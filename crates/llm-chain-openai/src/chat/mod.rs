//! This module implements chains for OpenAI's Chat Completions API.
//!
//! The building blocks are:
//!
//! - [`Model`] — which model to call, from the GPT-5.6 family to any custom model id
//! - [`ChatPromptTemplate`] / [`MessagePromptTemplate`] — templated conversations with
//!   [`Role`]s (`System`, `Developer`, `User`, `Assistant`)
//! - [`Options`] — per-step request options such as temperature, reasoning effort and
//!   response format
//! - [`Step`] — a model, a prompt and options, ready to be chained
//! - [`Executor`] — runs steps against the API
//! - [`AzureExecutor`] / [`AzureV1Config`] — the same executor pointed at Azure
//!   OpenAI's OpenAI-compatible v1 surface
mod azure;
mod error;
mod executor;
mod options;
mod prompt;
mod step;
mod tools;

pub use async_openai::config::OpenAIConfig;
pub use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls, ChatCompletionTool,
    ChatCompletionToolChoiceOption, ChatCompletionTools, FunctionCall, FunctionObject,
    ReasoningEffort, ResponseFormat, ToolChoiceOptions, Verbosity,
};
pub use azure::{AZURE_API_KEY_HEADER, AZURE_V1_PATH, AzureV1Config};
pub use error::FormatError;
pub use executor::{AzureExecutor, Executor};
pub use options::Options;
pub use prompt::{ChatPromptTemplate, MessagePromptTemplate, Role};
pub use step::{Model, Step};
pub use tools::{
    assistant_tool_calls_message, function_calls, function_tool, tool_result_message,
};
