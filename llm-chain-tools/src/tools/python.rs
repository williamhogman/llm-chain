use serde::{Deserialize, Serialize};
use std::process::Command;

use crate::description::{Describe, Format, ToolDescription};
use crate::tool::{Tool, ToolError, gen_invoke_function};

/// A tool that executes Python code.
pub struct PythonTool {}

impl PythonTool {
    /// Creates a new `PythonTool`.
    pub fn new() -> Self {
        PythonTool {}
    }
}

impl Default for PythonTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize, Deserialize)]
pub struct PythonToolInput {
    code: String,
}

#[derive(Serialize, Deserialize)]
pub struct PythonToolOutput {
    result: String,
    stderr: String,
}

impl Describe for PythonToolInput {
    fn describe() -> Format {
        vec![("code", "The Python code to execute.").into()].into()
    }
}

impl Describe for PythonToolOutput {
    fn describe() -> Format {
        vec![
            ("result", "The result of the executed Python code.").into(),
            ("stderr", "The stderr output of the Python code execution.").into(),
        ]
        .into()
    }
}

impl PythonTool {
    fn invoke_typed(&self, input: &PythonToolInput) -> Result<PythonToolOutput, ToolError> {
        let output = Command::new("python3")
            .arg("-c")
            .arg(&input.code)
            .output()?;
        Ok(PythonToolOutput {
            result: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

impl Tool for PythonTool {
    gen_invoke_function!();
    fn description(&self) -> ToolDescription {
        ToolDescription::new(
            "PythonTool",
            "A tool that executes Python code.",
            "Use this to execute Python code to solve your goals",
            PythonToolInput::describe(),
            PythonToolOutput::describe(),
        )
    }
}
