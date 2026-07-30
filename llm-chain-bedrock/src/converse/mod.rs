//! Steps, executors and prompt templates for Amazon Bedrock's
//! [Converse API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html).
//!
//! - [`Step`] pairs a [`Model`] with a [`ChatPromptTemplate`] and optional [`Options`].
//! - [`Executor`] sends formatted requests to the API and returns [`ConverseResponse`]s.
//! - [`ChatPromptTemplate`] holds optional system instructions plus user/assistant
//!   message templates.
mod error;
mod executor;
mod options;
mod prompt;
mod step;
mod types;

pub use error::{BedrockError, FormatError};
pub use executor::{
    BEARER_TOKEN_ENV_VAR, DEFAULT_REGION, Executor, REGION_ENV_VAR, REGION_FALLBACK_ENV_VAR,
};
pub use options::Options;
pub use prompt::{ChatPromptTemplate, MessagePromptTemplate};
pub use step::{DEFAULT_MODEL, Model, Step, models};
pub use types::{
    ContentBlock, ConverseOutput, ConverseRequest, ConverseResponse, InferenceConfig, Message,
    Metrics, ReasoningContent, ReasoningText, Role, StopReason, SystemContentBlock, TokenUsage,
};
