use thiserror::Error;

use crate::description::ToolDescription;

/// An error that occurred while invoking a tool.
#[derive(Debug, Error)]
pub enum ToolError {
    /// No tool with the given name exists in the collection.
    #[error("tool `{0}` not found")]
    NotFound(String),
    /// The model output contained an opening code fence but no closing one.
    #[error("no closed code block found in the model output")]
    UnclosedCodeBlock,
    /// The input or output could not be (de)serialized. The input must be valid YAML matching the tool's input format.
    #[error("invalid YAML: {0}")]
    Yaml(#[from] serde_yaml_ng::Error),
    /// An I/O error occurred while executing the tool.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// The tool failed for another reason.
    #[error("tool failed: {0}")]
    Execution(String),
}

macro_rules! gen_invoke_function {
    () => {
        fn invoke(
            &self,
            input: serde_yaml_ng::Value,
        ) -> Result<serde_yaml_ng::Value, $crate::ToolError> {
            let input = serde_yaml_ng::from_value(input)?;
            let output = self.invoke_typed(&input)?;
            Ok(serde_yaml_ng::to_value(output)?)
        }
    };
}
pub(crate) use gen_invoke_function;

/// A tool that an LLM can invoke by name with YAML input.
pub trait Tool {
    /// Describes the tool: its name, purpose and input/output formats.
    fn description(&self) -> ToolDescription;
    /// Invokes the tool with the given YAML input, returning YAML output.
    fn invoke(&self, input: serde_yaml_ng::Value) -> Result<serde_yaml_ng::Value, ToolError>;
    /// Returns `true` if this tool answers to the given name.
    fn matches(&self, name: &str) -> bool {
        self.description().name == name
    }
}
