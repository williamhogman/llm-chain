use serde::{Deserialize, Serialize};

use crate::description::{Describe, Format, ToolDescription};
use crate::tool::{Tool, ToolError, gen_invoke_function};

/// A tool that exits the program with the given status code.
pub struct ExitTool {}

impl ExitTool {
    /// Creates a new `ExitTool`.
    pub fn new() -> Self {
        ExitTool {}
    }
}

impl Default for ExitTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize, Deserialize)]
pub struct ExitToolInput {
    status_code: i32,
}

#[derive(Serialize, Deserialize)]
pub struct ExitToolOutput {}

impl Describe for ExitToolInput {
    fn describe() -> Format {
        vec![("status_code", "<integer> UNIX status to exit with").into()].into()
    }
}

impl Describe for ExitToolOutput {
    fn describe() -> Format {
        vec![].into()
    }
}

impl ExitTool {
    fn invoke_typed(&self, input: &ExitToolInput) -> Result<ExitToolOutput, ToolError> {
        std::process::exit(input.status_code);
    }
}

impl Tool for ExitTool {
    gen_invoke_function!();
    fn description(&self) -> ToolDescription {
        ToolDescription::new(
            "ExitTool",
            "Exits the program with the given status code",
            "Use this when your task is complete and you want to exit the program.",
            ExitToolInput::describe(),
            ExitToolOutput::describe(),
        )
    }
}
