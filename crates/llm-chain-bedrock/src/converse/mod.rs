//! Steps, executors and prompt templates for Amazon Bedrock's
//! [Converse API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html).
//!
//! - [`Step`] pairs a [`Model`] with a [`ChatPromptTemplate`] and optional [`Options`].
//! - [`Executor`] sends formatted requests to the API and returns [`ConverseResponse`]s,
//!   buffered or streamed via
//!   [`StreamingExecutor`](llm_chain::traits::StreamingExecutor).
//! - [`ChatPromptTemplate`] holds optional system instructions plus user/assistant
//!   message templates.
//! - [`StreamEvent`] and [`ResponseAccumulator`] cover streamed responses;
//!   [`EventStreamDecoder`] decodes AWS's binary event stream framing.
mod error;
mod eventstream;
mod executor;
mod options;
mod prompt;
mod step;
mod stream;
mod types;

pub use error::{BedrockError, FormatError};
pub use eventstream::{EventStreamDecoder, EventStreamError, EventStreamMessage, HeaderValue};
pub use executor::{
    BEARER_TOKEN_ENV_VAR, DEFAULT_REGION, Executor, REGION_ENV_VAR, REGION_FALLBACK_ENV_VAR,
};
pub use options::Options;
pub use prompt::{ChatPromptTemplate, MessagePromptTemplate};
pub use step::{DEFAULT_MODEL, Model, Step, models};
pub use stream::{
    ContentBlockStart, ContentDelta, ReasoningDelta, ResponseAccumulator, StreamEvent,
    ToolUseDelta, ToolUseStart,
};
pub use types::{
    ContentBlock, ConverseOutput, ConverseRequest, ConverseResponse, InferenceConfig, Message,
    Metrics, ReasoningContent, ReasoningText, Role, StopReason, SystemContentBlock, TokenUsage,
    Tool, ToolChoice, ToolConfiguration, ToolInputSchema, ToolResultBlock, ToolResultContent,
    ToolResultStatus, ToolSpec, ToolUseBlock,
};
