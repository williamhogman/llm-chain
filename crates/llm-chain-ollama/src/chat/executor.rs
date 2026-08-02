use std::borrow::Cow;

use futures::StreamExt as _;
use llm_chain::streaming::{NdjsonDecoder, frames};
use llm_chain::traits::BoxStream;
use llm_chain::{Parameters, traits};
use secrecy::{ExposeSecret, SecretString};

use super::error::OllamaError;
use super::types::{ChatRequest, ChatResponse};

/// The environment variable holding the server address, honored like the
/// Ollama CLI does.
pub const HOST_ENV_VAR: &str = "OLLAMA_HOST";
/// The default local server address.
pub const DEFAULT_BASE_URL: &str = "http://localhost:11434";
/// The base URL for Ollama's cloud, used by [`Executor::cloud`].
pub const CLOUD_BASE_URL: &str = "https://ollama.com";

/// The executor for Ollama's chat API.
///
/// Holds the HTTP client, the server address and an optional bearer token.
/// Cheap to clone; clones share the underlying connection pool.
///
/// # Examples
///
/// ```no_run
/// use llm_chain_ollama::chat::Executor;
///
/// // Talks to the local server; honors OLLAMA_HOST when set.
/// let exec = Executor::new_default();
/// // Or point at a remote server:
/// let exec = Executor::new_default().with_base_url("http://gpu-box:11434");
/// // Or use Ollama's cloud with an API key:
/// let exec = Executor::cloud("sk-...");
/// ```
#[derive(Clone)]
pub struct Executor {
    client: reqwest::Client,
    base_url: String,
    /// Kept in a [`SecretString`] so the key is redacted from any debug output
    /// and zeroized on drop.
    api_key: Option<SecretString>,
}

impl Default for Executor {
    fn default() -> Self {
        Self::new_default()
    }
}

impl Executor {
    /// Creates an executor for the local Ollama server.
    ///
    /// Honors the `OLLAMA_HOST` environment variable like the Ollama CLI does
    /// (scheme optional), falling back to [`DEFAULT_BASE_URL`]. Unlike the
    /// hosted-API executors this cannot fail: no API key is required.
    pub fn new_default() -> Self {
        let base_url = match std::env::var(HOST_ENV_VAR) {
            Ok(host) if !host.trim().is_empty() => normalize_base_url(&host),
            _ => DEFAULT_BASE_URL.to_string(),
        };
        Self {
            client: reqwest::Client::new(),
            base_url,
            api_key: None,
        }
    }

    /// Creates an executor for Ollama's cloud with the given API key.
    pub fn cloud<S: Into<String>>(api_key: S) -> Self {
        Self::new_default()
            .with_base_url(CLOUD_BASE_URL)
            .with_api_key(api_key)
    }

    /// Overrides the server address, e.g. for a remote GPU box or a proxy.
    ///
    /// A scheme is optional; `host:port` becomes `http://host:port`.
    pub fn with_base_url<S: AsRef<str>>(mut self, base_url: S) -> Self {
        self.base_url = normalize_base_url(base_url.as_ref());
        self
    }

    /// Sets a bearer token, required by Ollama's cloud and by proxied servers.
    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(SecretString::from(api_key.into()));
        self
    }

    async fn send(&self, request: &ChatRequest) -> Result<ChatResponse, OllamaError> {
        let mut http_request = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(request);
        if let Some(api_key) = &self.api_key {
            http_request = http_request.bearer_auth(api_key.expose_secret());
        }
        let response = http_request.send().await.map_err(|source| {
            if source.is_connect() {
                OllamaError::Connection {
                    url: self.base_url.clone(),
                    source,
                }
            } else {
                OllamaError::Http(source)
            }
        })?;

        let status = response.status();
        if status.is_success() {
            Ok(response.json::<ChatResponse>().await?)
        } else {
            Err(parse_api_error(status.as_u16(), response.text().await?))
        }
    }

    /// Sends the request with `stream: true` and returns the chunk stream
    /// once the response headers arrive.
    async fn send_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<BoxStream<ChatResponse, OllamaError>, OllamaError> {
        let mut request = request.clone();
        request.stream = true;
        let mut http_request = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&request);
        if let Some(api_key) = &self.api_key {
            http_request = http_request.bearer_auth(api_key.expose_secret());
        }
        let response = http_request.send().await.map_err(|source| {
            if source.is_connect() {
                OllamaError::Connection {
                    url: self.base_url.clone(),
                    source,
                }
            } else {
                OllamaError::Http(source)
            }
        })?;

        let status = response.status();
        if !status.is_success() {
            return Err(parse_api_error(status.as_u16(), response.text().await?));
        }
        let bytes = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(OllamaError::from));
        let lines = frames(NdjsonDecoder::new(), bytes);
        Ok(Box::pin(lines.map(|line| {
            line.and_then(|line| parse_stream_line(&line))
        })))
    }
}

// Manual Debug: keeps the output stable and the API key visibly redacted.
impl std::fmt::Debug for Executor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Executor")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
}

/// Normalizes a server address: trims whitespace and trailing slashes and
/// defaults the scheme to `http://`, matching how the Ollama CLI reads
/// `OLLAMA_HOST` values like `0.0.0.0:11434`.
fn normalize_base_url(raw: &str) -> String {
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    }
}

fn parse_api_error(status: u16, body: String) -> OllamaError {
    #[derive(serde::Deserialize)]
    struct ApiErrorBody {
        error: String,
    }
    let message = match serde_json::from_str::<ApiErrorBody>(&body) {
        Ok(parsed) => parsed.error,
        Err(_) => body,
    };
    OllamaError::Api { status, message }
}

/// Parses one NDJSON line into a chunk. A line carrying a top-level `error`
/// string (how the server reports mid-stream failures) becomes an
/// [`OllamaError::StreamError`].
fn parse_stream_line(line: &str) -> Result<ChatResponse, OllamaError> {
    let value: serde_json::Value = serde_json::from_str(line)?;
    if let Some(message) = value.get("error").and_then(serde_json::Value::as_str) {
        return Err(OllamaError::StreamError(message.to_string()));
    }
    Ok(serde_json::from_value(value)?)
}

/// Sums two optional counters; `None` only when both sides are `None`.
fn merge_counts(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (None, None) => None,
        _ => Some(a.unwrap_or(0) + b.unwrap_or(0)),
    }
}

impl traits::Executor for Executor {
    type Step = super::step::Step;
    type Output = ChatResponse;
    type Error = OllamaError;

    async fn execute(&self, input: ChatRequest) -> Result<ChatResponse, OllamaError> {
        self.send(&input).await
    }

    fn apply_output_to_parameters(parameters: Parameters, output: &ChatResponse) -> Parameters {
        parameters.with_text(output.text())
    }

    fn combine_outputs(output: &ChatResponse, other: &ChatResponse) -> ChatResponse {
        let mut combined = other.clone();
        combined.message.content = format!("{}\n{}", output.message.content, other.message.content);
        combined.message.thinking = match (&output.message.thinking, &other.message.thinking) {
            (Some(a), Some(b)) => Some(format!("{a}\n{b}")),
            (thinking, None) => thinking.clone(),
            (None, thinking) => thinking.clone(),
        };
        combined.total_duration = merge_counts(output.total_duration, other.total_duration);
        combined.load_duration = merge_counts(output.load_duration, other.load_duration);
        combined.prompt_eval_count =
            merge_counts(output.prompt_eval_count, other.prompt_eval_count);
        combined.prompt_eval_duration =
            merge_counts(output.prompt_eval_duration, other.prompt_eval_duration);
        combined.eval_count = merge_counts(output.eval_count, other.eval_count);
        combined.eval_duration = merge_counts(output.eval_duration, other.eval_duration);
        combined
    }
}

impl traits::StreamingExecutor for Executor {
    /// Streamed Ollama responses arrive as partial [`ChatResponse`] chunks
    /// rather than a separate event type.
    type StreamEvent = ChatResponse;

    async fn execute_stream(
        &self,
        input: ChatRequest,
    ) -> Result<BoxStream<ChatResponse, OllamaError>, OllamaError> {
        self.send_stream(&input).await
    }

    fn text_delta(event: &ChatResponse) -> Option<Cow<'_, str>> {
        (!event.message.content.is_empty()).then(|| Cow::Borrowed(event.message.content.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{DoneReason, Message, Role};
    use super::*;
    use llm_chain::traits::Executor as _;

    #[test]
    fn debug_never_prints_the_api_key() {
        let exec = Executor::new_default().with_api_key("sk-secret");
        let debug = format!("{exec:?}");
        assert!(!debug.contains("secret"));
        assert!(debug.contains("[REDACTED]"));
    }

    #[test]
    fn base_urls_are_normalized() {
        assert_eq!(
            normalize_base_url("http://gpu-box:11434/"),
            "http://gpu-box:11434"
        );
        assert_eq!(normalize_base_url("0.0.0.0:11434"), "http://0.0.0.0:11434");
        assert_eq!(
            normalize_base_url(" https://ollama.com "),
            "https://ollama.com"
        );
        let exec = Executor::new_default().with_base_url("gpu-box:11434");
        assert_eq!(exec.base_url, "http://gpu-box:11434");
    }

    #[test]
    fn cloud_points_at_ollama_com_with_a_key() {
        let exec = Executor::cloud("sk-key");
        assert_eq!(exec.base_url, CLOUD_BASE_URL);
        let api_key = exec.api_key.as_ref().expect("api key set");
        assert_eq!(api_key.expose_secret(), "sk-key");
    }

    #[test]
    fn api_errors_are_parsed() {
        let error = parse_api_error(
            404,
            r#"{"error":"model 'nope' not found, try pulling it first"}"#.to_string(),
        );
        match error {
            OllamaError::Api { status, message } => {
                assert_eq!(status, 404);
                assert_eq!(message, "model 'nope' not found, try pulling it first");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn malformed_error_bodies_fall_back_to_raw_text() {
        let error = parse_api_error(500, "upstream exploded".to_string());
        match error {
            OllamaError::Api { status, message } => {
                assert_eq!(status, 500);
                assert_eq!(message, "upstream exploded");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    fn response(text: &str, eval_count: Option<u64>, done_reason: DoneReason) -> ChatResponse {
        ChatResponse {
            model: "qwen3".to_string(),
            created_at: String::new(),
            message: Message::new(Role::Assistant, text),
            done: true,
            done_reason: Some(done_reason),
            total_duration: None,
            load_duration: None,
            prompt_eval_count: None,
            prompt_eval_duration: None,
            eval_count,
            eval_duration: None,
        }
    }

    #[test]
    fn combine_outputs_merges_text_and_counters() {
        let first = response("Hello", Some(5), DoneReason::Stop);
        let second = response("World", Some(3), DoneReason::Length);
        let combined = Executor::combine_outputs(&first, &second);
        assert_eq!(combined.text(), "Hello\nWorld");
        assert_eq!(combined.eval_count, Some(8));
        assert_eq!(combined.done_reason, Some(DoneReason::Length));
        // Counters absent on both sides stay absent.
        assert_eq!(combined.total_duration, None);
    }
}
