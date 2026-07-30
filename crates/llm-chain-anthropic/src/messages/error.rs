use llm_chain::PromptTemplateError;
use thiserror::Error;

/// An error that occurred while formatting a chat prompt into a Messages API request.
#[derive(Debug, Error)]
pub enum FormatError {
    /// The underlying prompt template failed to format.
    #[error(transparent)]
    Template(#[from] PromptTemplateError),
}

/// An error that occurred while executing a request against the Anthropic API.
#[derive(Debug, Error)]
pub enum AnthropicError {
    /// No API key was provided and `ANTHROPIC_API_KEY` is not set.
    #[error(
        "no API key: set the ANTHROPIC_API_KEY environment variable or use Executor::with_api_key"
    )]
    MissingApiKey,
    /// The HTTP request failed (connection, TLS, timeout, or invalid response body).
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    /// The API returned an error response.
    #[error("anthropic api error ({status} {error_type}): {message}")]
    Api {
        /// The HTTP status code.
        status: u16,
        /// The API error type, e.g. `invalid_request_error` or `overloaded_error`.
        error_type: String,
        /// The human-readable error message.
        message: String,
    },
}

impl AnthropicError {
    /// The HTTP status code associated with this error, when there is one.
    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Api { status, .. } => Some(*status),
            Self::Http(error) => error.status().map(|status| status.as_u16()),
            Self::MissingApiKey => None,
        }
    }

    /// Returns `true` when the request was rejected for rate limiting or
    /// because the API is overloaded — both are worth retrying with backoff.
    pub fn is_rate_limit(&self) -> bool {
        match self {
            Self::Api {
                status, error_type, ..
            } => {
                *status == 429
                    || error_type == "rate_limit_error"
                    || error_type == "overloaded_error"
            }
            _ => self.status() == Some(429),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn api_error(status: u16, error_type: &str) -> AnthropicError {
        AnthropicError::Api {
            status,
            error_type: error_type.to_string(),
            message: "m".to_string(),
        }
    }

    #[test]
    fn status_is_exposed_for_api_errors() {
        assert_eq!(api_error(400, "invalid_request_error").status(), Some(400));
        assert_eq!(AnthropicError::MissingApiKey.status(), None);
    }

    #[test]
    fn rate_limits_and_overloads_are_retryable() {
        assert!(api_error(429, "rate_limit_error").is_rate_limit());
        assert!(api_error(529, "overloaded_error").is_rate_limit());
        assert!(!api_error(400, "invalid_request_error").is_rate_limit());
        assert!(!AnthropicError::MissingApiKey.is_rate_limit());
    }
}
