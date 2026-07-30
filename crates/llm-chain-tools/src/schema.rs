//! Bridges [`Tool`](crate::Tool)s into the native tool-calling formats of the
//! provider crates.
//!
//! Every modern provider API takes the same three ingredients for a tool: a
//! name, a description and a JSON Schema for the arguments. [`ToolSchema`]
//! carries exactly those, generated from a tool's
//! [`ToolDescription`](crate::description::ToolDescription), so a
//! [`ToolCollection`](crate::ToolCollection) can be declared natively to any
//! provider:
//!
//! | Provider crate | Native definition |
//! |---|---|
//! | `llm-chain-openai` | `function_tool(&s.name, &s.description, s.parameters)` |
//! | `llm-chain-anthropic` | `ToolDefinition::new(&s.name, &s.description, s.parameters)` |
//! | `llm-chain-gemini` | `FunctionDeclaration::new(&s.name, &s.description, s.parameters)` |
//! | `llm-chain-bedrock` | `ToolSpec::new(&s.name, &s.description, s.parameters)` |
//! | `llm-chain-ollama` | `Tool::function(&s.name, &s.description, s.parameters)` |
//!
//! When the model calls a tool, run it with
//! [`ToolCollection::invoke_json`](crate::ToolCollection::invoke_json) and
//! send the JSON result back with the provider's continuation helper
//! (`with_tool_results` and friends).

use serde::{Deserialize, Serialize};

use crate::description::ToolDescription;

/// A provider-neutral, native tool definition: name, description and a JSON
/// Schema for the arguments.
///
/// Produced by [`ToolCollection::tool_schemas`](crate::ToolCollection::tool_schemas)
/// or [`From<ToolDescription>`](Self::from).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSchema {
    /// The name the model calls the tool by.
    pub name: String,
    /// What the tool does and when to use it.
    pub description: String,
    /// A [JSON Schema](https://json-schema.org/) object describing the tool's arguments.
    pub parameters: serde_json::Value,
}

impl From<ToolDescription> for ToolSchema {
    fn from(description: ToolDescription) -> Self {
        Self {
            name: description.name().to_string(),
            description: description.full_description(),
            parameters: description.input_schema(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::description::{Describe, Format};
    use crate::tool::{Tool, ToolError, gen_invoke_function};
    use crate::{ToolCollection, tools::BashTool};

    struct GreetTool;

    #[derive(Serialize, Deserialize)]
    struct GreetInput {
        name: String,
    }

    #[derive(Serialize, Deserialize)]
    struct GreetOutput {
        greeting: String,
    }

    impl Describe for GreetInput {
        fn describe() -> Format {
            vec![("name", "Who to greet.").into()].into()
        }
    }

    impl Describe for GreetOutput {
        fn describe() -> Format {
            vec![("greeting", "The greeting.").into()].into()
        }
    }

    impl GreetTool {
        fn invoke_typed(&self, input: &GreetInput) -> Result<GreetOutput, ToolError> {
            Ok(GreetOutput {
                greeting: format!("Hello, {}!", input.name),
            })
        }
    }

    impl Tool for GreetTool {
        gen_invoke_function!();
        fn description(&self) -> ToolDescription {
            ToolDescription::new(
                "GreetTool",
                "Greets a person by name.",
                "Use this to greet people.",
                GreetInput::describe(),
                GreetOutput::describe(),
            )
        }
    }

    #[test]
    fn tool_schemas_generate_json_schema() {
        let collection = ToolCollection::new(vec![Box::new(BashTool::new())]);
        let schemas = collection.tool_schemas();
        assert_eq!(schemas.len(), 1);
        let schema = &schemas[0];
        assert_eq!(schema.name, "BashTool");
        assert!(schema.description.contains("executes a bash command"));
        assert_eq!(schema.parameters["type"], "object");
        assert_eq!(schema.parameters["properties"]["cmd"]["type"], "string");
        assert_eq!(schema.parameters["required"][0], "cmd");
        // No provider-hostile keywords.
        assert!(schema.parameters.get("additionalProperties").is_none());
    }

    #[test]
    fn invoke_json_round_trips_through_yaml_tools() {
        let collection = ToolCollection::new(vec![Box::new(GreetTool)]);
        let output = collection
            .invoke_json("GreetTool", &serde_json::json!({"name": "world"}))
            .unwrap();
        assert_eq!(output, serde_json::json!({"greeting": "Hello, world!"}));
    }

    #[test]
    fn invoke_json_reports_missing_tools() {
        let collection = ToolCollection::new(vec![]);
        let error = collection
            .invoke_json("Nope", &serde_json::json!({}))
            .unwrap_err();
        assert!(matches!(error, ToolError::NotFound(name) if name == "Nope"));
    }

    #[test]
    fn schema_description_joins_context() {
        let schema = ToolSchema::from(GreetTool.description());
        assert_eq!(
            schema.description,
            "Greets a person by name. Use this to greet people."
        );
    }
}
