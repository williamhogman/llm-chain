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
mod error;
mod executor;
mod options;
mod prompt;
mod step;

pub use async_openai::config::OpenAIConfig;
pub use async_openai::types::chat::{ReasoningEffort, ResponseFormat, Verbosity};
pub use error::FormatError;
pub use executor::Executor;
pub use options::Options;
pub use prompt::{ChatPromptTemplate, MessagePromptTemplate, Role};
pub use step::{Model, Step};
