use llm_chain::PromptTemplateError;
use thiserror::Error;

/// An error that occurred while formatting a chat prompt into a chat API request.
#[derive(Debug, Error)]
pub enum FormatError {
    /// The underlying prompt template failed to format.
    #[error(transparent)]
    Template(#[from] PromptTemplateError),
}

/// An error that occurred while executing a request against the Lovable AI Gateway.
#[derive(Debug, Error)]
pub enum LovableError {
    /// No API key was available.
    ///
    /// The gateway key lives in the `LOVABLE_API_KEY` environment variable
    /// (auto-provisioned in Lovable Cloud projects). It is a server-side
    /// credential — never ship it to browsers or other untrusted clients.
    #[error(
        "LOVABLE_API_KEY is not set (or empty) — provide the Lovable AI Gateway key via the environment or Executor::with_api_key"
    )]
    MissingApiKey,
    /// The HTTP request failed (connection, TLS, timeout, or invalid response body).
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    /// A response or streamed chunk could not be deserialized.
    #[error("invalid JSON payload: {0}")]
    Json(#[from] serde_json::Error),
    /// The gateway returned an error response.
    ///
    /// Only 429 (rate limited) and 5xx (transient upstream) are worth
    /// retrying, with backoff. Everything else is terminal: a 400 means the
    /// request itself is wrong (e.g. a model id not in the Lovable catalog,
    /// or an unsupported field for the selected model) and a 402 means the
    /// workspace is out of credits.
    #[error("lovable ai gateway error ({status}): {message}")]
    Api {
        /// The HTTP status code.
        status: u16,
        /// The error message relayed by the gateway.
        message: String,
    },
    /// The gateway reported an error mid-stream.
    #[error("lovable ai gateway stream error: {0}")]
    StreamError(String),
}

impl LovableError {
    /// The HTTP status code associated with this error, when there is one.
    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Api { status, .. } => Some(*status),
            Self::Http(error) => error.status().map(|status| status.as_u16()),
            Self::MissingApiKey | Self::Json(_) | Self::StreamError(_) => None,
        }
    }

    /// Returns `true` when the request was rejected for rate limiting
    /// (HTTP 429) — worth retrying with backoff.
    pub fn is_rate_limit(&self) -> bool {
        self.status() == Some(429)
    }

    /// Returns `true` when the workspace's Lovable credits are exhausted
    /// (HTTP 402). Not retryable — surface a billing error to the user.
    pub fn is_credits_exhausted(&self) -> bool {
        self.status() == Some(402)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_and_retry_helpers_are_exposed() {
        let error = LovableError::Api {
            status: 429,
            message: "rate limited".to_string(),
        };
        assert_eq!(error.status(), Some(429));
        assert!(error.is_rate_limit());
        assert!(!error.is_credits_exhausted());

        let error = LovableError::Api {
            status: 402,
            message: "credits exhausted".to_string(),
        };
        assert_eq!(error.status(), Some(402));
        assert!(error.is_credits_exhausted());
        assert!(!error.is_rate_limit());

        assert_eq!(LovableError::MissingApiKey.status(), None);
    }

    #[test]
    fn missing_key_error_names_the_env_var() {
        let message = LovableError::MissingApiKey.to_string();
        assert!(message.contains("LOVABLE_API_KEY"));
    }
}
