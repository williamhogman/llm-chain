//! Steps, executors and prompt templates for Ollama's
//! [chat API](https://docs.ollama.com/api#generate-a-chat-completion).
//!
//! - [`Step`] pairs a [`Model`] with a [`ChatPromptTemplate`] and optional [`Options`].
//! - [`Executor`] sends formatted requests to an Ollama server and returns
//!   [`ChatResponse`]s.
//! - [`ChatPromptTemplate`] holds system/user/assistant message templates.
mod error;
mod executor;
mod options;
mod prompt;
mod step;
mod types;

pub use error::{FormatError, OllamaError};
pub use executor::{CLOUD_BASE_URL, DEFAULT_BASE_URL, Executor, HOST_ENV_VAR};
pub use options::Options;
pub use prompt::{ChatPromptTemplate, MessagePromptTemplate};
pub use step::{DEFAULT_MODEL, Model, Step};
pub use types::{
    ChatRequest, ChatResponse, DoneReason, Format, Message, ModelOptions, Role, Think,
};
