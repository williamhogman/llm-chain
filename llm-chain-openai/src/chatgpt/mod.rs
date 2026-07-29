//! This module implements chains for OpenAI's chat models.
mod error;
mod executor;
mod prompt;
mod step;

pub use async_openai::config::OpenAIConfig;
pub use async_openai::types::chat::Role;
pub use error::FormatError;
pub use executor::Executor;
pub use prompt::{ChatPromptTemplate, MessagePromptTemplate};
pub use step::{Model, Step};
