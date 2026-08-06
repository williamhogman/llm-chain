//! Steps, executors and prompt templates for the Lovable AI Gateway's
//! OpenAI-compatible chat completions API.
//!
//! - [`Step`] pairs a [`Model`] with a [`ChatPromptTemplate`] and optional [`Options`].
//! - [`Executor`] sends formatted requests to the gateway and returns
//!   [`ChatResponse`]s, or streams [`ChatChunk`]s via
//!   [`StreamingExecutor`](llm_chain::traits::StreamingExecutor).
//! - [`ChatPromptTemplate`] holds system/user/assistant message templates.
mod error;
mod executor;
mod options;
mod prompt;
mod step;
mod stream;
mod types;

pub use error::{FormatError, LovableError};
pub use executor::{API_KEY_ENV_VAR, DEFAULT_BASE_URL, Executor, RUN_ID_HEADER, SDK_HEADER};
pub use options::Options;
pub use prompt::{ChatPromptTemplate, MessagePromptTemplate};
pub use step::{DEFAULT_MODEL, Model, Step, models};
pub use stream::{
    ChatChunk, ChunkChoice, Delta, FunctionCallDelta, ResponseAccumulator, ToolCallDelta,
};
pub use types::{
    ChatRequest, ChatResponse, Choice, FinishReason, FunctionCall, JsonSchema, Message, Reasoning,
    ReasoningEffort, ResponseFormat, Role, StreamOptions, Tool, ToolCall, ToolFunction, Usage,
};
