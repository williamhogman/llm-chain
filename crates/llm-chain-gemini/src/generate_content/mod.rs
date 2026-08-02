//! Steps, executors and prompt templates for the Gemini API's
//! [`generateContent`](https://ai.google.dev/api/generate-content) method.
//!
//! - [`Step`] pairs a [`Model`] with a [`ChatPromptTemplate`] and optional [`Options`].
//! - [`Executor`] sends formatted requests to the API and returns
//!   [`GenerateContentResponse`]s, or streams them chunk by chunk via
//!   [`StreamingExecutor`](llm_chain::traits::StreamingExecutor).
//! - [`ChatPromptTemplate`] holds optional system instructions plus user/model
//!   message templates.
mod error;
mod executor;
mod options;
mod prompt;
mod step;
mod stream;
mod types;

pub use error::{FormatError, GeminiError};
pub use executor::{
    API_KEY_ENV_VAR, API_KEY_FALLBACK_ENV_VAR, API_VERSION, DEFAULT_BASE_URL, Executor,
    VERTEX_API_VERSION, VERTEX_BASE_URL,
};
pub use options::Options;
pub use prompt::{ChatPromptTemplate, MessagePromptTemplate};
pub use step::{Model, Step};
pub use stream::ResponseAccumulator;
pub use types::{
    Candidate, Content, FinishReason, FunctionCall, FunctionCallingConfig, FunctionCallingMode,
    FunctionDeclaration, FunctionResponse, GenerateContentRequest, GenerateContentResponse,
    GenerationConfig, Part, PromptFeedback, Role, ThinkingConfig, ThinkingLevel, Tool, ToolConfig,
    UsageMetadata,
};
