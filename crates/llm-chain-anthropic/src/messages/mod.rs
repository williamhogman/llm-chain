//! Steps, executors and prompt templates for Anthropic's
//! [Messages API](https://docs.anthropic.com/en/api/messages).
//!
//! - [`Step`] pairs a [`Model`] with a [`ChatPromptTemplate`] and optional [`Options`].
//! - [`Executor`] sends formatted requests to the API and returns [`MessagesResponse`]s.
//! - [`ChatPromptTemplate`] holds optional system instructions plus user/assistant
//!   message templates.
mod error;
mod executor;
mod options;
mod prompt;
mod step;
mod types;

pub use error::{AnthropicError, FormatError};
pub use executor::{ANTHROPIC_VERSION, API_KEY_ENV_VAR, DEFAULT_BASE_URL, Executor};
pub use options::{DEFAULT_MAX_TOKENS, Options};
pub use prompt::{ChatPromptTemplate, MessagePromptTemplate};
pub use step::{Model, Step};
pub use types::{
    ContentBlock, Effort, Message, MessageContent, MessagesRequest, MessagesResponse, Role,
    StopReason, Thinking, ToolChoice, ToolDefinition, ToolResult, ToolUse, Usage,
};
