use std::borrow::Cow;

use futures::StreamExt as _;
use llm_chain::streaming::{SseDecoder, SseEvent, frames};
use llm_chain::traits::BoxStream;
use llm_chain::{Parameters, traits};
use secrecy::{ExposeSecret, SecretString};

use super::error::LovableError;
use super::stream::ChatChunk;
use super::types::{ChatRequest, ChatResponse, StreamOptions, Usage};

/// The environment variable holding the API key.
///
/// In Lovable Cloud projects the key is auto-provisioned under this name. It
/// is a server-side credential — never ship it to browsers or other
/// untrusted clients.
pub const API_KEY_ENV_VAR: &str = "LOVABLE_API_KEY";
/// The default API endpoint (OpenAI-compatible base URL).
pub const DEFAULT_BASE_URL: &str = "https://ai.gateway.lovable.dev/v1";
/// The response header carrying the gateway's run id, which correlates a
/// request with Lovable AI usage logs. Surfaced as
/// [`ChatResponse::run_id`] and [`ChatChunk::run_id`].
pub const RUN_ID_HEADER: &str = "x-lovable-aig-run-id";
/// The request header identifying the calling SDK to the gateway.
pub const SDK_HEADER: &str = "x-lovable-aig-sdk";

/// The SDK name this crate reports in [`SDK_HEADER`].
const SDK_NAME: &str = "llm-chain";

/// The executor for the Lovable AI Gateway's chat completions API.
///
/// Holds the HTTP client and credentials. Cheap to clone; clones share the
/// underlying connection pool.
///
/// # Examples
///
/// ```no_run
/// use llm_chain_lovable::chat::Executor;
///
/// // Reads LOVABLE_API_KEY from the environment.
/// let exec = Executor::new_default().unwrap();
/// // Or provide the key explicitly:
/// let exec = Executor::with_api_key("...");
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
    /// Creates an executor with the API key from the `LOVABLE_API_KEY`
    /// environment variable.
    ///
    /// # Errors
    ///
    /// Returns [`LovableError::MissingApiKey`] when the variable is unset or empty.
    pub fn new_default() -> Result<Self, LovableError> {
        match std::env::var(API_KEY_ENV_VAR) {
            Ok(key) if !key.trim().is_empty() => Ok(Self::with_api_key(key)),
            _ => Err(LovableError::MissingApiKey),
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

    /// Overrides the API endpoint, e.g. for a proxy or a mock in tests.
    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    fn post(&self, request: &ChatRequest) -> reqwest::RequestBuilder {
        self.client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Lovable-API-Key", self.api_key.expose_secret())
            .header(SDK_HEADER, SDK_NAME)
            .json(request)
    }

    async fn send(&self, request: &ChatRequest) -> Result<ChatResponse, LovableError> {
        let response = self.post(request).send().await?;

        let status = response.status();
        if !status.is_success() {
            return Err(parse_api_error(status.as_u16(), response.text().await?));
        }
        let run_id = header_value(&response, RUN_ID_HEADER);
        let mut parsed = response.json::<ChatResponse>().await?;
        parsed.run_id = run_id;
        Ok(parsed)
    }

    /// Sends the request with `stream: true` and returns the chunk stream
    /// once the response headers arrive.
    async fn send_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<BoxStream<ChatChunk, LovableError>, LovableError> {
        let mut request = request.clone();
        request.stream = true;
        // Ask for the final usage-bearing chunk so accumulated responses
        // carry token counts like buffered ones do.
        request.stream_options = Some(StreamOptions {
            include_usage: true,
        });
        let response = self
            .post(&request)
            .header("accept", "text/event-stream")
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            return Err(parse_api_error(status.as_u16(), response.text().await?));
        }
        let run_id = header_value(&response, RUN_ID_HEADER);
        let bytes = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(LovableError::from));
        let events = frames(SseDecoder::new(), bytes);
        Ok(Box::pin(events.filter_map(move |event| {
            let run_id = run_id.clone();
            async move {
                match event {
                    Ok(event) => parse_stream_event(&event, run_id).transpose(),
                    Err(error) => Some(Err(error)),
                }
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

fn header_value(response: &reqwest::Response, name: &str) -> Option<String> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}

fn parse_api_error(status: u16, body: String) -> LovableError {
    #[derive(serde::Deserialize)]
    struct NestedError {
        message: String,
    }
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum ErrorField {
        Nested(NestedError),
        Plain(String),
    }
    #[derive(serde::Deserialize)]
    struct ApiErrorBody {
        error: ErrorField,
    }
    let message = match serde_json::from_str::<ApiErrorBody>(&body) {
        Ok(ApiErrorBody {
            error: ErrorField::Nested(nested),
        }) => nested.message,
        Ok(ApiErrorBody {
            error: ErrorField::Plain(plain),
        }) => plain,
        Err(_) => body,
    };
    LovableError::Api { status, message }
}

/// Parses one SSE event into a chunk. The terminating `data: [DONE]` frame
/// yields `None`; an event carrying a top-level `error` (how the gateway
/// relays mid-stream failures) becomes a [`LovableError::StreamError`].
fn parse_stream_event(
    event: &SseEvent,
    run_id: Option<String>,
) -> Result<Option<ChatChunk>, LovableError> {
    let data = event.data.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }
    let value: serde_json::Value = serde_json::from_str(data)?;
    if let Some(error) = value.get("error") {
        let message = error
            .get("message")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .or_else(|| error.as_str().map(str::to_string))
            .unwrap_or_else(|| error.to_string());
        return Err(LovableError::StreamError(message));
    }
    let mut chunk: ChatChunk = serde_json::from_value(value)?;
    chunk.run_id = run_id;
    Ok(Some(chunk))
}

/// Sums two optional usages; `None` only when both sides are `None`.
fn merge_usage(a: Option<Usage>, b: Option<Usage>) -> Option<Usage> {
    match (a, b) {
        (None, None) => None,
        (a, b) => {
            let a = a.unwrap_or_default();
            let b = b.unwrap_or_default();
            Some(Usage {
                prompt_tokens: a.prompt_tokens + b.prompt_tokens,
                completion_tokens: a.completion_tokens + b.completion_tokens,
                total_tokens: a.total_tokens + b.total_tokens,
            })
        }
    }
}

impl traits::Executor for Executor {
    type Step = super::step::Step;
    type Output = ChatResponse;
    type Error = LovableError;

    async fn execute(&self, input: ChatRequest) -> Result<ChatResponse, LovableError> {
        self.send(&input).await
    }

    fn apply_output_to_parameters(parameters: Parameters, output: &ChatResponse) -> Parameters {
        parameters.with_text(output.text())
    }

    fn combine_outputs(output: &ChatResponse, other: &ChatResponse) -> ChatResponse {
        let mut combined = other.clone();
        let text = format!("{}\n{}", output.text(), other.text());
        match combined.choices.first_mut() {
            Some(choice) => choice.message.content = Some(text),
            None => {
                if let Some(choice) = output.choices.first() {
                    let mut choice = choice.clone();
                    choice.message.content = Some(text);
                    combined.choices.push(choice);
                }
            }
        }
        combined.usage = merge_usage(output.usage, other.usage);
        if combined.run_id.is_none() {
            combined.run_id = output.run_id.clone();
        }
        combined
    }
}

impl traits::StreamingExecutor for Executor {
    /// Streamed gateway responses arrive as OpenAI-style
    /// `chat.completion.chunk` events.
    type StreamEvent = ChatChunk;

    async fn execute_stream(
        &self,
        input: ChatRequest,
    ) -> Result<BoxStream<ChatChunk, LovableError>, LovableError> {
        self.send_stream(&input).await
    }

    fn text_delta(event: &ChatChunk) -> Option<Cow<'_, str>> {
        event
            .text()
            .filter(|text| !text.is_empty())
            .map(Cow::Borrowed)
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{Choice, FinishReason, Message, Role};
    use super::*;
    use llm_chain::traits::{Executor as _, StreamingExecutor as _};

    #[test]
    fn debug_never_prints_the_api_key() {
        let exec = Executor::with_api_key("sk-secret");
        let debug = format!("{exec:?}");
        assert!(!debug.contains("secret"));
        assert!(debug.contains("[REDACTED]"));
    }

    #[test]
    fn base_urls_are_normalized() {
        let exec = Executor::with_api_key("k").with_base_url("http://localhost:8080/v1/");
        assert_eq!(exec.base_url, "http://localhost:8080/v1");
        assert_eq!(
            Executor::with_api_key("k").base_url,
            "https://ai.gateway.lovable.dev/v1"
        );
    }

    #[test]
    fn api_errors_are_parsed_in_both_shapes() {
        let error = parse_api_error(
            402,
            r#"{"error":{"message":"Payment required: workspace is out of credits","type":"payment_required"}}"#
                .to_string(),
        );
        match &error {
            LovableError::Api { status, message } => {
                assert_eq!(*status, 402);
                assert!(message.starts_with("Payment required"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert!(error.is_credits_exhausted());

        let error = parse_api_error(429, r#"{"error":"Rate limit exceeded"}"#.to_string());
        match &error {
            LovableError::Api { status, message } => {
                assert_eq!(*status, 429);
                assert_eq!(message, "Rate limit exceeded");
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert!(error.is_rate_limit());
    }

    #[test]
    fn malformed_error_bodies_fall_back_to_raw_text() {
        let error = parse_api_error(500, "upstream exploded".to_string());
        match error {
            LovableError::Api { status, message } => {
                assert_eq!(status, 500);
                assert_eq!(message, "upstream exploded");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    fn event(data: &str) -> SseEvent {
        SseEvent {
            event: None,
            data: data.to_string(),
        }
    }

    #[test]
    fn done_frames_and_blank_events_end_quietly() {
        assert!(
            parse_stream_event(&event("[DONE]"), None)
                .unwrap()
                .is_none()
        );
        assert!(parse_stream_event(&event("  "), None).unwrap().is_none());
    }

    #[test]
    fn stream_chunks_carry_the_run_id() {
        let chunk = parse_stream_event(
            &event(r#"{"id":"c1","choices":[{"index":0,"delta":{"content":"hi"}}]}"#),
            Some("run-123".to_string()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(chunk.text(), Some("hi"));
        assert_eq!(chunk.run_id.as_deref(), Some("run-123"));
        assert_eq!(Executor::text_delta(&chunk).as_deref(), Some("hi"));
    }

    #[test]
    fn mid_stream_errors_are_typed() {
        let error = parse_stream_event(
            &event(r#"{"error":{"message":"upstream exploded","code":502}}"#),
            None,
        )
        .unwrap_err();
        match error {
            LovableError::StreamError(message) => assert_eq!(message, "upstream exploded"),
            other => panic!("unexpected error: {other:?}"),
        }

        let error = parse_stream_event(&event(r#"{"error":"plain"}"#), None).unwrap_err();
        match error {
            LovableError::StreamError(message) => assert_eq!(message, "plain"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    fn response(text: &str, usage: Option<Usage>) -> ChatResponse {
        ChatResponse {
            id: "chatcmpl-1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "google/gemini-3.6-flash".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message::new(Role::Assistant, text),
                finish_reason: Some(FinishReason::Stop),
            }],
            usage,
            run_id: None,
        }
    }

    #[test]
    fn combine_outputs_merges_text_and_usage() {
        let first = response(
            "Hello",
            Some(Usage {
                prompt_tokens: 5,
                completion_tokens: 2,
                total_tokens: 7,
            }),
        );
        let second = response(
            "World",
            Some(Usage {
                prompt_tokens: 3,
                completion_tokens: 4,
                total_tokens: 7,
            }),
        );
        let combined = Executor::combine_outputs(&first, &second);
        assert_eq!(combined.text(), "Hello\nWorld");
        let usage = combined.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 8);
        assert_eq!(usage.completion_tokens, 6);
        assert_eq!(usage.total_tokens, 14);

        // Usage absent on both sides stays absent.
        let combined = Executor::combine_outputs(&response("a", None), &response("b", None));
        assert_eq!(combined.usage, None);
    }
}
