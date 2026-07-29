use llm_chain::{Parameters, traits};

use super::error::GeminiError;
use super::types::{Content, GenerateContentRequest, GenerateContentResponse, Role, UsageMetadata};

/// The primary environment variable holding the API key.
pub const API_KEY_ENV_VAR: &str = "GEMINI_API_KEY";
/// The fallback environment variable holding the API key.
pub const API_KEY_FALLBACK_ENV_VAR: &str = "GOOGLE_API_KEY";
/// The default API endpoint.
pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
/// The API version this crate speaks.
pub const API_VERSION: &str = "v1beta";

/// The executor for the Gemini API.
///
/// Holds the HTTP client and credentials. Cheap to clone; clones share the
/// underlying connection pool.
///
/// # Examples
///
/// ```no_run
/// use llm_chain_gemini::generate_content::Executor;
///
/// // Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.
/// let exec = Executor::new_default().unwrap();
/// // Or provide the key explicitly:
/// let exec = Executor::with_api_key("AIza...");
/// ```
#[derive(Clone)]
pub struct Executor {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl Executor {
    /// Creates an executor with the API key from the `GEMINI_API_KEY`
    /// environment variable, falling back to `GOOGLE_API_KEY`.
    ///
    /// # Errors
    ///
    /// Returns [`GeminiError::MissingApiKey`] when both variables are unset or empty.
    pub fn new_default() -> Result<Self, GeminiError> {
        for var in [API_KEY_ENV_VAR, API_KEY_FALLBACK_ENV_VAR] {
            if let Ok(key) = std::env::var(var)
                && !key.trim().is_empty()
            {
                return Ok(Self::with_api_key(key));
            }
        }
        Err(GeminiError::MissingApiKey)
    }

    /// Creates an executor with an explicit API key.
    pub fn with_api_key<S: Into<String>>(api_key: S) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Overrides the API endpoint, e.g. for a gateway or proxy.
    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    async fn send(
        &self,
        request: &GenerateContentRequest,
    ) -> Result<GenerateContentResponse, GeminiError> {
        let response = self
            .client
            .post(format!(
                "{}/{}/models/{}:generateContent",
                self.base_url, API_VERSION, request.model
            ))
            .header("x-goog-api-key", &self.api_key)
            .json(request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            return Err(parse_api_error(status.as_u16(), response.text().await?));
        }
        let response = response.json::<GenerateContentResponse>().await?;
        if response.candidates.is_empty() {
            let reason = response
                .prompt_feedback
                .as_ref()
                .and_then(|feedback| feedback.block_reason.clone())
                .unwrap_or_else(|| "no candidates returned".to_string());
            return Err(GeminiError::NoCandidates { reason });
        }
        Ok(response)
    }
}

// Never derive Debug: it would print the API key.
impl std::fmt::Debug for Executor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Executor")
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .finish()
    }
}

fn parse_api_error(http_status: u16, body: String) -> GeminiError {
    #[derive(serde::Deserialize)]
    struct ApiErrorBody {
        error: ApiErrorDetail,
    }
    #[derive(serde::Deserialize)]
    struct ApiErrorDetail {
        #[serde(default)]
        status: String,
        message: String,
    }
    match serde_json::from_str::<ApiErrorBody>(&body) {
        Ok(parsed) => GeminiError::Api {
            http_status,
            status: if parsed.error.status.is_empty() {
                "UNKNOWN".to_string()
            } else {
                parsed.error.status
            },
            message: parsed.error.message,
        },
        Err(_) => GeminiError::Api {
            http_status,
            status: "UNKNOWN".to_string(),
            message: body,
        },
    }
}

impl traits::Executor for Executor {
    type Step = super::step::Step;
    type Output = GenerateContentResponse;
    type Error = GeminiError;

    async fn execute(
        &self,
        input: GenerateContentRequest,
    ) -> Result<GenerateContentResponse, GeminiError> {
        self.send(&input).await
    }

    fn apply_output_to_parameters(
        parameters: Parameters,
        output: &GenerateContentResponse,
    ) -> Parameters {
        parameters.with_text(output.text())
    }

    fn combine_outputs(
        output: &GenerateContentResponse,
        other: &GenerateContentResponse,
    ) -> GenerateContentResponse {
        let mut combined = other.clone();
        combined.candidates = vec![super::types::Candidate {
            content: Some(Content::text(
                Role::Model,
                format!("{}\n{}", output.text(), other.text()),
            )),
            finish_reason: other.finish_reason(),
        }];
        combined.usage_metadata = UsageMetadata {
            prompt_token_count: output.usage_metadata.prompt_token_count
                + other.usage_metadata.prompt_token_count,
            candidates_token_count: output.usage_metadata.candidates_token_count
                + other.usage_metadata.candidates_token_count,
            thoughts_token_count: output.usage_metadata.thoughts_token_count
                + other.usage_metadata.thoughts_token_count,
            total_token_count: output.usage_metadata.total_token_count
                + other.usage_metadata.total_token_count,
        };
        combined
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{Candidate, FinishReason};
    use super::*;
    use llm_chain::traits::Executor as _;

    #[test]
    fn debug_never_prints_the_api_key() {
        let exec = Executor::with_api_key("AIza-secret");
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
            r#"{"error":{"code":429,"message":"slow down","status":"RESOURCE_EXHAUSTED"}}"#
                .to_string(),
        );
        match error {
            GeminiError::Api {
                http_status,
                status,
                message,
            } => {
                assert_eq!(http_status, 429);
                assert_eq!(status, "RESOURCE_EXHAUSTED");
                assert_eq!(message, "slow down");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn malformed_error_bodies_fall_back_to_raw_text() {
        let error = parse_api_error(500, "upstream exploded".to_string());
        match error {
            GeminiError::Api {
                http_status,
                status,
                message,
            } => {
                assert_eq!(http_status, 500);
                assert_eq!(status, "UNKNOWN");
                assert_eq!(message, "upstream exploded");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn combine_outputs_merges_text_and_usage() {
        let make = |text: &str, prompt: u32, output: u32| GenerateContentResponse {
            candidates: vec![Candidate {
                content: Some(Content::text(Role::Model, text)),
                finish_reason: Some(FinishReason::Stop),
            }],
            prompt_feedback: None,
            usage_metadata: UsageMetadata {
                prompt_token_count: prompt,
                candidates_token_count: output,
                thoughts_token_count: 0,
                total_token_count: prompt + output,
            },
            model_version: None,
            response_id: None,
        };
        let combined = Executor::combine_outputs(&make("Hello", 10, 5), &make("World", 7, 3));
        assert_eq!(combined.text(), "Hello\nWorld");
        assert_eq!(combined.usage_metadata.prompt_token_count, 17);
        assert_eq!(combined.usage_metadata.candidates_token_count, 8);
        assert_eq!(combined.usage_metadata.total_token_count, 25);
    }
}
