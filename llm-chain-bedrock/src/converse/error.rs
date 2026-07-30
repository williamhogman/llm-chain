use llm_chain::PromptTemplateError;
use thiserror::Error;

/// An error that occurred while formatting a chat prompt into a Converse API request.
#[derive(Debug, Error)]
pub enum FormatError {
    /// The underlying prompt template failed to format.
    #[error(transparent)]
    Template(#[from] PromptTemplateError),
}

/// An error that occurred while executing a request against the Bedrock runtime.
#[derive(Debug, Error)]
pub enum BedrockError {
    /// No API key was provided and `AWS_BEARER_TOKEN_BEDROCK` is not set.
    #[error(
        "no Bedrock API key: set the AWS_BEARER_TOKEN_BEDROCK environment variable or use Executor::with_bearer_token"
    )]
    MissingBearerToken,
    /// The HTTP request failed (connection, TLS, timeout, or invalid response body).
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    /// The API returned an error response.
    #[error("bedrock api error ({status} {error_type}): {message}")]
    Api {
        /// The HTTP status code.
        status: u16,
        /// The AWS exception name from the `x-amzn-errortype` header,
        /// e.g. `ValidationException` or `ThrottlingException`.
        error_type: String,
        /// The human-readable error message.
        message: String,
    },
}

impl BedrockError {
    /// The HTTP status code associated with this error, when there is one.
    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Api { status, .. } => Some(*status),
            Self::Http(error) => error.status().map(|status| status.as_u16()),
            Self::MissingBearerToken => None,
        }
    }

    /// Returns `true` when the request was throttled — worth retrying with
    /// backoff.
    pub fn is_rate_limit(&self) -> bool {
        match self {
            Self::Api {
                status, error_type, ..
            } => *status == 429 || error_type == "ThrottlingException",
            _ => self.status() == Some(429),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn api_error(status: u16, error_type: &str) -> BedrockError {
        BedrockError::Api {
            status,
            error_type: error_type.to_string(),
            message: "m".to_string(),
        }
    }

    #[test]
    fn status_is_exposed_for_api_errors() {
        assert_eq!(api_error(400, "ValidationException").status(), Some(400));
        assert_eq!(BedrockError::MissingBearerToken.status(), None);
    }

    #[test]
    fn throttling_is_retryable() {
        assert!(api_error(429, "ThrottlingException").is_rate_limit());
        assert!(api_error(400, "ThrottlingException").is_rate_limit());
        assert!(!api_error(400, "ValidationException").is_rate_limit());
        assert!(!BedrockError::MissingBearerToken.is_rate_limit());
    }
}
