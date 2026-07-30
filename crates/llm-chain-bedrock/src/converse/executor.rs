use llm_chain::{Parameters, traits};
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};
use secrecy::{ExposeSecret, SecretString};

use super::error::BedrockError;
use super::types::{ContentBlock, ConverseRequest, ConverseResponse, Message, Metrics, Role};

/// The environment variable holding the Bedrock API key (bearer token).
pub const BEARER_TOKEN_ENV_VAR: &str = "AWS_BEARER_TOKEN_BEDROCK";
/// The primary environment variable holding the AWS region.
pub const REGION_ENV_VAR: &str = "AWS_REGION";
/// The fallback environment variable holding the AWS region.
pub const REGION_FALLBACK_ENV_VAR: &str = "AWS_DEFAULT_REGION";
/// The region used when neither region variable is set.
pub const DEFAULT_REGION: &str = "us-east-1";

/// Everything percent-encoded in a model id except RFC 3986 unreserved
/// characters, so `:` in version suffixes and `/` in ARNs are escaped.
const MODEL_ID_ENCODE_SET: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'.')
    .remove(b'_')
    .remove(b'~');

/// The executor for Amazon Bedrock's Converse API.
///
/// Holds the HTTP client and credentials. Cheap to clone; clones share the
/// underlying connection pool.
///
/// Authenticates with a Bedrock API key sent as a bearer token — generate one
/// in the AWS console (Bedrock → API keys) and export it as
/// `AWS_BEARER_TOKEN_BEDROCK`. Deployments that require SigV4 request signing
/// (IAM roles, STS sessions) should use the official `aws-sdk-bedrockruntime`
/// crate instead.
///
/// # Examples
///
/// ```no_run
/// use llm_chain_bedrock::converse::Executor;
///
/// // Reads AWS_BEARER_TOKEN_BEDROCK and AWS_REGION from the environment.
/// let exec = Executor::new_default().unwrap();
/// // Or provide the key explicitly and pick a region:
/// let exec = Executor::with_bearer_token("bedrock-api-key-...").with_region("eu-central-1");
/// ```
#[derive(Clone)]
pub struct Executor {
    client: reqwest::Client,
    /// Kept in a [`SecretString`] so the token is redacted from any debug
    /// output and zeroized on drop.
    bearer_token: SecretString,
    base_url: String,
}

impl Executor {
    /// Creates an executor with the API key from the `AWS_BEARER_TOKEN_BEDROCK`
    /// environment variable, targeting the region from `AWS_REGION` (or
    /// `AWS_DEFAULT_REGION`, or [`DEFAULT_REGION`]).
    ///
    /// # Errors
    ///
    /// Returns [`BedrockError::MissingBearerToken`] when the token variable is
    /// unset or empty.
    pub fn new_default() -> Result<Self, BedrockError> {
        match std::env::var(BEARER_TOKEN_ENV_VAR) {
            Ok(token) if !token.trim().is_empty() => Ok(Self::with_bearer_token(token)),
            _ => Err(BedrockError::MissingBearerToken),
        }
    }

    /// Creates an executor with an explicit Bedrock API key, targeting the
    /// region from the environment (or [`DEFAULT_REGION`]).
    pub fn with_bearer_token<S: Into<String>>(bearer_token: S) -> Self {
        Self {
            client: reqwest::Client::new(),
            bearer_token: SecretString::from(bearer_token.into()),
            base_url: endpoint_for_region(&region_from_env()),
        }
    }

    /// Targets the Bedrock runtime in the given AWS region, e.g. `eu-central-1`.
    pub fn with_region<S: AsRef<str>>(mut self, region: S) -> Self {
        self.base_url = endpoint_for_region(region.as_ref());
        self
    }

    /// Overrides the API endpoint entirely, e.g. for a gateway or proxy.
    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    async fn send(&self, request: &ConverseRequest) -> Result<ConverseResponse, BedrockError> {
        let response = self
            .client
            .post(format!(
                "{}/model/{}/converse",
                self.base_url,
                utf8_percent_encode(&request.model_id, MODEL_ID_ENCODE_SET)
            ))
            .bearer_auth(self.bearer_token.expose_secret())
            .json(request)
            .send()
            .await?;

        let status = response.status();
        if status.is_success() {
            Ok(response.json::<ConverseResponse>().await?)
        } else {
            let error_type = response
                .headers()
                .get("x-amzn-errortype")
                .and_then(|value| value.to_str().ok())
                .map(|value| value.split(':').next().unwrap_or(value).to_string());
            Err(parse_api_error(
                status.as_u16(),
                error_type,
                response.text().await?,
            ))
        }
    }
}

// Manual Debug: keeps the output stable and the token visibly redacted.
impl std::fmt::Debug for Executor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Executor")
            .field("bearer_token", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .finish()
    }
}

fn region_from_env() -> String {
    for var in [REGION_ENV_VAR, REGION_FALLBACK_ENV_VAR] {
        if let Ok(region) = std::env::var(var)
            && !region.trim().is_empty()
        {
            return region.trim().to_string();
        }
    }
    DEFAULT_REGION.to_string()
}

fn endpoint_for_region(region: &str) -> String {
    format!("https://bedrock-runtime.{region}.amazonaws.com")
}

fn parse_api_error(status: u16, error_type: Option<String>, body: String) -> BedrockError {
    #[derive(serde::Deserialize)]
    struct ApiErrorBody {
        #[serde(alias = "Message")]
        message: String,
    }
    let message = match serde_json::from_str::<ApiErrorBody>(&body) {
        Ok(parsed) => parsed.message,
        Err(_) => body,
    };
    BedrockError::Api {
        status,
        error_type: error_type.unwrap_or_else(|| "UnknownException".to_string()),
        message,
    }
}

/// Sums two latency metrics; `None` only when both sides are `None`.
fn merge_metrics(a: Option<Metrics>, b: Option<Metrics>) -> Option<Metrics> {
    match (a, b) {
        (None, None) => None,
        _ => Some(Metrics {
            latency_ms: a.map(|m| m.latency_ms).unwrap_or(0) + b.map(|m| m.latency_ms).unwrap_or(0),
        }),
    }
}

impl traits::Executor for Executor {
    type Step = super::step::Step;
    type Output = ConverseResponse;
    type Error = BedrockError;

    async fn execute(&self, input: ConverseRequest) -> Result<ConverseResponse, BedrockError> {
        self.send(&input).await
    }

    fn apply_output_to_parameters(parameters: Parameters, output: &ConverseResponse) -> Parameters {
        parameters.with_text(output.text())
    }

    fn combine_outputs(output: &ConverseResponse, other: &ConverseResponse) -> ConverseResponse {
        let mut combined = other.clone();
        combined.output = Some(super::types::ConverseOutput {
            message: Some(Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: format!("{}\n{}", output.text(), other.text()),
                }],
            }),
        });
        combined.usage = super::types::TokenUsage {
            input_tokens: output.usage.input_tokens + other.usage.input_tokens,
            output_tokens: output.usage.output_tokens + other.usage.output_tokens,
            total_tokens: output.usage.total_tokens + other.usage.total_tokens,
            cache_read_input_tokens: output.usage.cache_read_input_tokens
                + other.usage.cache_read_input_tokens,
            cache_write_input_tokens: output.usage.cache_write_input_tokens
                + other.usage.cache_write_input_tokens,
        };
        combined.metrics = merge_metrics(output.metrics, other.metrics);
        combined
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{ConverseOutput, StopReason, TokenUsage};
    use super::*;
    use llm_chain::traits::Executor as _;

    #[test]
    fn debug_never_prints_the_api_key() {
        let exec = Executor::with_bearer_token("bedrock-secret");
        let debug = format!("{exec:?}");
        assert!(!debug.contains("secret"));
        assert!(debug.contains("[REDACTED]"));
    }

    #[test]
    fn regions_map_to_runtime_endpoints() {
        let exec = Executor::with_bearer_token("k").with_region("eu-central-1");
        assert_eq!(
            exec.base_url,
            "https://bedrock-runtime.eu-central-1.amazonaws.com"
        );
    }

    #[test]
    fn base_url_trailing_slash_is_normalized() {
        let exec = Executor::with_bearer_token("k").with_base_url("https://proxy.example.com/");
        assert_eq!(exec.base_url, "https://proxy.example.com");
    }

    #[test]
    fn model_ids_are_percent_encoded_for_the_path() {
        let encoded = utf8_percent_encode(
            "global.anthropic.claude-sonnet-5-20260203-v1:0",
            MODEL_ID_ENCODE_SET,
        )
        .to_string();
        assert_eq!(encoded, "global.anthropic.claude-sonnet-5-20260203-v1%3A0");
        let arn = utf8_percent_encode(
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-pro-v1:0",
            MODEL_ID_ENCODE_SET,
        )
        .to_string();
        assert!(!arn.contains(':'));
        assert!(!arn.contains('/'));
    }

    #[test]
    fn api_errors_are_parsed() {
        let error = parse_api_error(
            429,
            Some("ThrottlingException".to_string()),
            r#"{"message":"Too many requests, please wait before trying again."}"#.to_string(),
        );
        match error {
            BedrockError::Api {
                status,
                error_type,
                message,
            } => {
                assert_eq!(status, 429);
                assert_eq!(error_type, "ThrottlingException");
                assert_eq!(
                    message,
                    "Too many requests, please wait before trying again."
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn malformed_error_bodies_fall_back_to_raw_text() {
        let error = parse_api_error(500, None, "upstream exploded".to_string());
        match error {
            BedrockError::Api {
                status,
                error_type,
                message,
            } => {
                assert_eq!(status, 500);
                assert_eq!(error_type, "UnknownException");
                assert_eq!(message, "upstream exploded");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    fn response(text: &str, input: u32, output: u32, latency_ms: u64) -> ConverseResponse {
        ConverseResponse {
            output: Some(ConverseOutput {
                message: Some(Message::text(Role::Assistant, text)),
            }),
            stop_reason: Some(StopReason::EndTurn),
            usage: TokenUsage {
                input_tokens: input,
                output_tokens: output,
                total_tokens: input + output,
                cache_read_input_tokens: 0,
                cache_write_input_tokens: 0,
            },
            metrics: Some(Metrics { latency_ms }),
        }
    }

    #[test]
    fn combine_outputs_merges_text_usage_and_metrics() {
        let combined =
            Executor::combine_outputs(&response("Hello", 10, 5, 100), &response("World", 7, 3, 50));
        assert_eq!(combined.text(), "Hello\nWorld");
        assert_eq!(combined.usage.input_tokens, 17);
        assert_eq!(combined.usage.output_tokens, 8);
        assert_eq!(combined.usage.total_tokens, 25);
        assert_eq!(combined.metrics.unwrap().latency_ms, 150);
    }
}
