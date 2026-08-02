use llm_chain::PromptTemplateError;
use thiserror::Error;

/// An error that occurred while formatting a chat prompt into a Gemini API request.
#[derive(Debug, Error)]
pub enum FormatError {
    /// The underlying prompt template failed to format.
    #[error(transparent)]
    Template(#[from] PromptTemplateError),
}

/// An error that occurred while executing a request against the Gemini API.
#[derive(Debug, Error)]
pub enum GeminiError {
    /// No API key was provided and neither `GEMINI_API_KEY` nor
    /// `GOOGLE_API_KEY` is set.
    #[error(
        "no API key: set the GEMINI_API_KEY environment variable or use Executor::with_api_key"
    )]
    MissingApiKey,
    /// The HTTP request failed (connection, TLS, timeout, or invalid response body).
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    /// A streamed chunk could not be deserialized.
    #[error("invalid JSON payload: {0}")]
    Json(#[from] serde_json::Error),
    /// The API returned an error response.
    #[error("gemini api error ({http_status} {status}): {message}")]
    Api {
        /// The HTTP status code.
        http_status: u16,
        /// The API status, e.g. `INVALID_ARGUMENT` or `RESOURCE_EXHAUSTED`.
        status: String,
        /// The human-readable error message.
        message: String,
    },
    /// The response contained no candidates (e.g. the prompt was blocked).
    #[error("gemini returned no candidates: {reason}")]
    NoCandidates {
        /// Why no candidates were returned, e.g. the prompt block reason.
        reason: String,
    },
}

impl GeminiError {
    /// The HTTP status code associated with this error, when there is one.
    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Api { http_status, .. } => Some(*http_status),
            Self::Http(error) => error.status().map(|status| status.as_u16()),
            Self::MissingApiKey | Self::Json(_) | Self::NoCandidates { .. } => None,
        }
    }

    /// Returns `true` when the request was rejected for rate limiting or
    /// quota exhaustion — worth retrying with backoff.
    pub fn is_rate_limit(&self) -> bool {
        match self {
            Self::Api {
                http_status,
                status,
                ..
            } => *http_status == 429 || status == "RESOURCE_EXHAUSTED",
            _ => self.status() == Some(429),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn api_error(http_status: u16, status: &str) -> GeminiError {
        GeminiError::Api {
            http_status,
            status: status.to_string(),
            message: "m".to_string(),
        }
    }

    #[test]
    fn status_is_exposed_for_api_errors() {
        assert_eq!(api_error(400, "INVALID_ARGUMENT").status(), Some(400));
        assert_eq!(GeminiError::MissingApiKey.status(), None);
        assert_eq!(
            GeminiError::NoCandidates {
                reason: "SAFETY".to_string()
            }
            .status(),
            None
        );
    }

    #[test]
    fn quota_exhaustion_is_retryable() {
        assert!(api_error(429, "RESOURCE_EXHAUSTED").is_rate_limit());
        assert!(api_error(429, "UNKNOWN").is_rate_limit());
        assert!(!api_error(400, "INVALID_ARGUMENT").is_rate_limit());
        assert!(!GeminiError::MissingApiKey.is_rate_limit());
    }
}
