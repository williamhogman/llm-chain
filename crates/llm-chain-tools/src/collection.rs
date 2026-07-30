use serde::{Deserialize, Serialize};

use crate::tool::{Tool, ToolError};

/// A collection of tools that can be invoked by name from model output.
pub struct ToolCollection {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolCollection {
    /// Creates a new collection from the given tools.
    pub fn new(tools: Vec<Box<dyn Tool>>) -> Self {
        Self { tools }
    }

    /// Invokes the named tool with the given YAML input.
    pub fn invoke(
        &self,
        name: &str,
        input: &serde_yaml_ng::Value,
    ) -> Result<serde_yaml_ng::Value, ToolError> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.matches(name))
            .ok_or_else(|| ToolError::NotFound(name.to_string()))?;
        tool.invoke(input.clone())
    }

    /// Extracts a tool invocation from model chat output and runs it, returning the YAML-serialized result.
    ///
    /// The invocation may be wrapped in a fenced code block (with or without a
    /// `yaml` language tag) or be the entire message.
    pub fn process_chat_input(&self, data: &str) -> Result<String, ToolError> {
        let yaml_str = extract_code_block(data)?;
        let input: ToolInvocationInput = serde_yaml_ng::from_str(yaml_str)?;
        let output = self.invoke(&input.command, &input.input)?;
        Ok(serde_yaml_ng::to_string(&output)?)
    }

    /// Describes all the tools in the collection as YAML, suitable for prompting.
    pub fn describe(&self) -> Result<String, ToolError> {
        let des: Vec<_> = self.tools.iter().map(|t| t.description()).collect();
        Ok(serde_yaml_ng::to_string(&des)?)
    }
}

/// Returns the contents of the first fenced code block in `data`, or the whole
/// string if there is no code fence. The language tag on the opening fence is
/// ignored.
fn extract_code_block(data: &str) -> Result<&str, ToolError> {
    let Some(start) = data.find("```") else {
        return Ok(data);
    };
    let after_fence = &data[start + 3..];
    // Everything up to the end of the line is a language tag (e.g. ```yaml).
    let content_start = after_fence
        .find('\n')
        .map(|i| i + 1)
        .ok_or(ToolError::UnclosedCodeBlock)?;
    let content = &after_fence[content_start..];
    let end = content.find("```").ok_or(ToolError::UnclosedCodeBlock)?;
    Ok(&content[..end])
}

#[derive(Serialize, Deserialize)]
struct ToolInvocationInput {
    command: String,
    input: serde_yaml_ng::Value,
}

#[cfg(test)]
mod tests {
    use super::extract_code_block;

    #[test]
    fn extracts_plain_fence() {
        let data = "```\ncommand: Foo\n```";
        assert_eq!(extract_code_block(data).unwrap(), "command: Foo\n");
    }

    #[test]
    fn extracts_yaml_fence() {
        let data = "Sure! Here you go:\n```yaml\ncommand: Foo\n```\ndone";
        assert_eq!(extract_code_block(data).unwrap(), "command: Foo\n");
    }

    #[test]
    fn passes_through_bare_yaml() {
        let data = "command: Foo\ninput:\n  cmd: ls";
        assert_eq!(extract_code_block(data).unwrap(), data);
    }

    #[test]
    fn errors_on_unclosed_fence() {
        let data = "```yaml\ncommand: Foo";
        assert!(extract_code_block(data).is_err());
    }
}
