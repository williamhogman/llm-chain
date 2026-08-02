use std::borrow::Cow;

use futures::StreamExt as _;
use llm_chain::streaming::{SseDecoder, SseEvent, frames};
use llm_chain::traits::BoxStream;
use llm_chain::{Parameters, traits};
use secrecy::{ExposeSecret, SecretString};

use super::error::AnthropicError;
use super::stream::StreamEvent;
use super::types::{ContentBlock, MessagesRequest, MessagesResponse, Usage};

/// The environment variable holding the API key.
pub const API_KEY_ENV_VAR: &str = "ANTHROPIC_API_KEY";
/// The default API endpoint.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// The Messages API version this crate speaks.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// The executor for Anthropic's Messages API.
///
/// Holds the HTTP client and credentials. Cheap to clone; clones share the
/// underlying connection pool.
///
/// # Examples
///
/// ```no_run
/// use llm_chain_anthropic::messages::Executor;
///
/// // Reads ANTHROPIC_API_KEY from the environment.
/// let exec = Executor::new_default().unwrap();
/// // Or provide the key explicitly:
/// let exec = Executor::with_api_key("sk-ant-...");
/// ```
#[derive(Clone)]
pub struct Executor {
    client: reqwest::Client,
    /// Kept in a [`SecretString`] so the key is redacted from any debug output
    /// and zeroized on drop.
    api_key: SecretString,
    base_url: String,
}

impl Executor {
    /// Creates an executor with the API key from the `ANTHROPIC_API_KEY`
    /// environment variable.
    ///
    /// # Errors
    ///
    /// Returns [`AnthropicError::MissingApiKey`] when the variable is unset or empty.
    pub fn new_default() -> Result<Self, AnthropicError> {
        match std::env::var(API_KEY_ENV_VAR) {
            Ok(key) if !key.trim().is_empty() => Ok(Self::with_api_key(key)),
            _ => Err(AnthropicError::MissingApiKey),
        }
    }

    /// Creates an executor with an explicit API key.
    pub fn with_api_key<S: Into<String>>(api_key: S) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: SecretString::from(api_key.into()),
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Overrides the API endpoint, e.g. for a gateway or proxy.
    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    async fn send(&self, request: &MessagesRequest) -> Result<MessagesResponse, AnthropicError> {
        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(request)
            .send()
            .await?;

        let status = response.status();
        if status.is_success() {
            Ok(response.json::<MessagesResponse>().await?)
        } else {
            Err(parse_api_error(status.as_u16(), response.text().await?))
        }
    }

    /// Sends the request with `stream: true` and returns the typed event
    /// stream once the response headers arrive.
    async fn send_stream(
        &self,
        request: &MessagesRequest,
    ) -> Result<BoxStream<StreamEvent, AnthropicError>, AnthropicError> {
        // `stream` is a transport concern, so it is injected here rather than
        // carried on the request type.
        let mut body = serde_json::to_value(request)?;
        body["stream"] = serde_json::Value::Bool(true);
        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("accept", "text/event-stream")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            return Err(parse_api_error(status.as_u16(), response.text().await?));
        }
        let bytes = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(AnthropicError::from));
        let events = frames(SseDecoder::new(), bytes);
        Ok(Box::pin(events.filter_map(|event| async move {
            match event {
                Ok(event) => parse_stream_event(event).transpose(),
                Err(error) => Some(Err(error)),
            }
        })))
    }
}

// Manual Debug: keeps the output stable and the API key visibly redacted.
impl std::fmt::Debug for Executor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Executor")
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .finish()
    }
}

fn parse_api_error(status: u16, body: String) -> AnthropicError {
    #[derive(serde::Deserialize)]
    struct ApiErrorBody {
        error: ApiErrorDetail,
    }
    #[derive(serde::Deserialize)]
    struct ApiErrorDetail {
        #[serde(rename = "type")]
        error_type: String,
        message: String,
    }
    match serde_json::from_str::<ApiErrorBody>(&body) {
        Ok(parsed) => AnthropicError::Api {
            status,
            error_type: parsed.error.error_type,
            message: parsed.error.message,
        },
        Err(_) => AnthropicError::Api {
            status,
            error_type: "unknown".to_string(),
            message: body,
        },
    }
}

/// Maps one SSE event to a typed stream event; `None` for empty keep-alives.
///
/// The API signals mid-stream failures as `event: error`, which becomes
/// [`AnthropicError::StreamError`].
fn parse_stream_event(event: SseEvent) -> Result<Option<StreamEvent>, AnthropicError> {
    if event.data.is_empty() {
        return Ok(None);
    }
    if event.event.as_deref() == Some("error") {
        return Err(parse_stream_error(&event.data));
    }
    Ok(Some(serde_json::from_str(&event.data)?))
}

fn parse_stream_error(data: &str) -> AnthropicError {
    #[derive(serde::Deserialize)]
    struct ErrorBody {
        error: ErrorDetail,
    }
    #[derive(serde::Deserialize)]
    struct ErrorDetail {
        #[serde(rename = "type")]
        error_type: String,
        message: String,
    }
    match serde_json::from_str::<ErrorBody>(data) {
        Ok(parsed) => AnthropicError::StreamError {
            error_type: parsed.error.error_type,
            message: parsed.error.message,
        },
        Err(_) => AnthropicError::StreamError {
            error_type: "unknown".to_string(),
            message: data.to_string(),
        },
    }
}

impl traits::Executor for Executor {
    type Step = super::step::Step;
    type Output = MessagesResponse;
    type Error = AnthropicError;

    async fn execute(&self, input: MessagesRequest) -> Result<MessagesResponse, AnthropicError> {
        self.send(&input).await
    }

    fn apply_output_to_parameters(parameters: Parameters, output: &MessagesResponse) -> Parameters {
        parameters.with_text(output.text())
    }

    fn combine_outputs(output: &MessagesResponse, other: &MessagesResponse) -> MessagesResponse {
        let mut combined = output.clone();
        combined.content = vec![ContentBlock::Text {
            text: format!("{}\n{}", output.text(), other.text()),
        }];
        combined.usage = Usage {
            input_tokens: output.usage.input_tokens + other.usage.input_tokens,
            output_tokens: output.usage.output_tokens + other.usage.output_tokens,
        };
        combined.stop_reason = other.stop_reason;
        combined.stop_sequence = other.stop_sequence.clone();
        combined
    }
}

impl traits::StreamingExecutor for Executor {
    type StreamEvent = StreamEvent;

    async fn execute_stream(
        &self,
        input: MessagesRequest,
    ) -> Result<BoxStream<StreamEvent, AnthropicError>, AnthropicError> {
        self.send_stream(&input).await
    }

    fn text_delta(event: &StreamEvent) -> Option<Cow<'_, str>> {
        event.text_delta().map(Cow::Borrowed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_chain::traits::Executor as _;

    #[test]
    fn debug_never_prints_the_api_key() {
        let exec = Executor::with_api_key("sk-ant-secret");
        let debug = format!("{exec:?}");
        assert!(!debug.contains("secret"));
        assert!(debug.contains("[REDACTED]"));
    }

    #[test]
    fn base_url_trailing_slash_is_normalized() {
        let exec = Executor::with_api_key("k").with_base_url("https://gateway.example.com/");
        assert_eq!(exec.base_url, "https://gateway.example.com");
    }

    #[test]
    fn api_errors_are_parsed() {
        let error = parse_api_error(
            429,
            r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#
                .to_string(),
        );
        match error {
            AnthropicError::Api {
                status,
                error_type,
                message,
            } => {
                assert_eq!(status, 429);
                assert_eq!(error_type, "rate_limit_error");
                assert_eq!(message, "slow down");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn malformed_error_bodies_fall_back_to_raw_text() {
        let error = parse_api_error(500, "upstream exploded".to_string());
        match error {
            AnthropicError::Api {
                status,
                error_type,
                message,
            } => {
                assert_eq!(status, 500);
                assert_eq!(error_type, "unknown");
                assert_eq!(message, "upstream exploded");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn combine_outputs_merges_text_and_usage() {
        let first = MessagesResponse {
            id: "msg_1".to_string(),
            model: "m".to_string(),
            content: vec![ContentBlock::Text {
                text: "Hello".to_string(),
            }],
            stop_reason: Some(super::super::types::StopReason::EndTurn),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };
        let second = MessagesResponse {
            id: "msg_2".to_string(),
            model: "m".to_string(),
            content: vec![ContentBlock::Text {
                text: "World".to_string(),
            }],
            stop_reason: Some(super::super::types::StopReason::MaxTokens),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 7,
                output_tokens: 3,
            },
        };
        let combined = Executor::combine_outputs(&first, &second);
        assert_eq!(combined.text(), "Hello\nWorld");
        assert_eq!(combined.usage.input_tokens, 17);
        assert_eq!(combined.usage.output_tokens, 8);
        assert_eq!(
            combined.stop_reason,
            Some(super::super::types::StopReason::MaxTokens)
        );
    }
}
