use llm_chain::{Parameters, traits};

use super::error::GeminiError;
use super::types::{Content, GenerateContentRequest, GenerateContentResponse, Role, UsageMetadata};

/// The primary environment variable holding the API key.
pub const API_KEY_ENV_VAR: &str = "GEMINI_API_KEY";
/// The fallback environment variable holding the API key.
pub const API_KEY_FALLBACK_ENV_VAR: &str = "GOOGLE_API_KEY";
/// The default API endpoint (the consumer Gemini API).
pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
/// The API version this crate speaks on the consumer Gemini API.
pub const API_VERSION: &str = "v1beta";
/// The endpoint serving Vertex AI's global location and express mode.
pub const VERTEX_BASE_URL: &str = "https://aiplatform.googleapis.com";
/// The API version this crate speaks on Vertex AI.
pub const VERTEX_API_VERSION: &str = "v1";

/// How requests authenticate. Never derives Debug: it holds credentials.
#[derive(Clone)]
enum Auth {
    /// `x-goog-api-key` header: consumer Gemini API keys and Vertex express keys.
    ApiKey(String),
    /// `Authorization: Bearer` header: Vertex OAuth2 access tokens.
    Bearer(String),
}

/// Which URL layout the endpoint uses. The wire format is identical on all of
/// them; only the path (and auth) differs.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Route {
    /// The consumer Gemini API: `/v1beta/models/{model}:generateContent`.
    GenerativeLanguage,
    /// Vertex AI, scoped to a project and location:
    /// `/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent`.
    Vertex { project: String, location: String },
    /// Vertex AI express mode: `/v1/publishers/google/models/{model}:generateContent`.
    VertexExpress,
}

/// The executor for Gemini models, on the consumer Gemini API or on Vertex AI.
///
/// Holds the HTTP client and credentials. Cheap to clone; clones share the
/// underlying connection pool.
///
/// # Examples
///
/// ```no_run
/// use llm_chain_gemini::generate_content::Executor;
///
/// // Consumer Gemini API: reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.
/// let exec = Executor::new_default().unwrap();
/// // Or provide the key explicitly:
/// let exec = Executor::with_api_key("AIza...");
///
/// // Vertex AI with an OAuth2 access token (`gcloud auth print-access-token`):
/// let exec = Executor::vertex("my-project", "europe-north1", "ya29....");
/// // Vertex AI express mode with an API key:
/// let exec = Executor::vertex_express("AQ....");
/// ```
#[derive(Clone)]
pub struct Executor {
    client: reqwest::Client,
    auth: Auth,
    base_url: String,
    route: Route,
}

impl Executor {
    /// Creates an executor for the consumer Gemini API with the API key from
    /// the `GEMINI_API_KEY` environment variable, falling back to
    /// `GOOGLE_API_KEY`.
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

    /// Creates an executor for the consumer Gemini API with an explicit API key.
    pub fn with_api_key<S: Into<String>>(api_key: S) -> Self {
        Self {
            client: reqwest::Client::new(),
            auth: Auth::ApiKey(api_key.into()),
            base_url: DEFAULT_BASE_URL.to_string(),
            route: Route::GenerativeLanguage,
        }
    }

    /// Creates an executor for Vertex AI, scoped to a Google Cloud project and
    /// location and authenticating with an OAuth2 access token.
    ///
    /// Get a token with `gcloud auth print-access-token` or from a service
    /// account / Application Default Credentials. Access tokens are
    /// short-lived (about an hour): mint a fresh executor when yours expires.
    ///
    /// The location picks the serving region, e.g. `us-central1` or
    /// `europe-north1`; pass `global` to let Google route to whichever region
    /// has capacity (recommended for the newest models).
    pub fn vertex<P, L, T>(project: P, location: L, access_token: T) -> Self
    where
        P: Into<String>,
        L: Into<String>,
        T: Into<String>,
    {
        let location = location.into();
        let base_url = if location == "global" {
            VERTEX_BASE_URL.to_string()
        } else {
            format!("https://{location}-aiplatform.googleapis.com")
        };
        Self {
            client: reqwest::Client::new(),
            auth: Auth::Bearer(access_token.into()),
            base_url,
            route: Route::Vertex {
                project: project.into(),
                location,
            },
        }
    }

    /// Creates an executor for Vertex AI express mode with an API key — no
    /// project or location setup required.
    pub fn vertex_express<S: Into<String>>(api_key: S) -> Self {
        Self {
            client: reqwest::Client::new(),
            auth: Auth::ApiKey(api_key.into()),
            base_url: VERTEX_BASE_URL.to_string(),
            route: Route::VertexExpress,
        }
    }

    /// Overrides the API endpoint, e.g. for a gateway or proxy. The URL layout
    /// (consumer API or Vertex) is preserved.
    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    fn url(&self, model: &str) -> String {
        match &self.route {
            Route::GenerativeLanguage => format!(
                "{}/{}/models/{}:generateContent",
                self.base_url, API_VERSION, model
            ),
            Route::Vertex { project, location } => format!(
                "{}/{}/projects/{}/locations/{}/publishers/google/models/{}:generateContent",
                self.base_url, VERTEX_API_VERSION, project, location, model
            ),
            Route::VertexExpress => format!(
                "{}/{}/publishers/google/models/{}:generateContent",
                self.base_url, VERTEX_API_VERSION, model
            ),
        }
    }

    async fn send(
        &self,
        request: &GenerateContentRequest,
    ) -> Result<GenerateContentResponse, GeminiError> {
        let mut http_request = self.client.post(self.url(&request.model)).json(request);
        http_request = match &self.auth {
            Auth::ApiKey(api_key) => http_request.header("x-goog-api-key", api_key),
            Auth::Bearer(access_token) => http_request.bearer_auth(access_token),
        };
        let response = http_request.send().await?;

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

// Never derive Debug: it would print the credentials.
impl std::fmt::Debug for Executor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Executor")
            .field(
                match self.auth {
                    Auth::ApiKey(_) => "api_key",
                    Auth::Bearer(_) => "access_token",
                },
                &"[REDACTED]",
            )
            .field("base_url", &self.base_url)
            .field("route", &self.route)
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
            cached_content_token_count: output.usage_metadata.cached_content_token_count
                + other.usage_metadata.cached_content_token_count,
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
    fn debug_never_prints_the_access_token() {
        let exec = Executor::vertex("my-project", "global", "ya29-secret");
        let debug = format!("{exec:?}");
        assert!(!debug.contains("secret"));
        assert!(debug.contains("[REDACTED]"));
        assert!(debug.contains("access_token"));
    }

    #[test]
    fn base_url_trailing_slash_is_normalized() {
        let exec = Executor::with_api_key("k").with_base_url("https://gateway.example.com/");
        assert_eq!(exec.base_url, "https://gateway.example.com");
    }

    #[test]
    fn consumer_api_urls_use_v1beta_models() {
        let exec = Executor::with_api_key("k");
        assert_eq!(
            exec.url("gemini-3.6-flash"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"
        );
    }

    #[test]
    fn vertex_urls_are_project_and_location_scoped() {
        let exec = Executor::vertex("my-project", "europe-north1", "token");
        assert_eq!(
            exec.url("gemini-3.6-flash"),
            "https://europe-north1-aiplatform.googleapis.com/v1/projects/my-project/locations/europe-north1/publishers/google/models/gemini-3.6-flash:generateContent"
        );
    }

    #[test]
    fn vertex_global_location_uses_the_global_endpoint() {
        let exec = Executor::vertex("my-project", "global", "token");
        assert_eq!(
            exec.url("gemini-3.6-flash"),
            "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/publishers/google/models/gemini-3.6-flash:generateContent"
        );
    }

    #[test]
    fn vertex_express_urls_have_no_project_scoping() {
        let exec = Executor::vertex_express("k");
        assert_eq!(
            exec.url("gemini-3.6-flash"),
            "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3.6-flash:generateContent"
        );
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
                cached_content_token_count: 0,
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
