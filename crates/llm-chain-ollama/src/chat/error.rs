use llm_chain::PromptTemplateError;
use thiserror::Error;

/// An error that occurred while formatting a chat prompt into a chat API request.
#[derive(Debug, Error)]
pub enum FormatError {
    /// The underlying prompt template failed to format.
    #[error(transparent)]
    Template(#[from] PromptTemplateError),
}

/// An error that occurred while executing a request against an Ollama server.
#[derive(Debug, Error)]
pub enum OllamaError {
    /// The server could not be reached at all.
    ///
    /// For a local setup this almost always means the daemon is not running —
    /// start it with `ollama serve` (or the desktop app).
    #[error("could not reach the Ollama server at {url} — is `ollama serve` running? ({source})")]
    Connection {
        /// The base URL that was tried.
        url: String,
        /// The underlying connection error.
        source: reqwest::Error,
    },
    /// The HTTP request failed (TLS, timeout, or invalid response body).
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    /// A streamed chunk could not be deserialized.
    #[error("invalid JSON payload: {0}")]
    Json(#[from] serde_json::Error),
    /// The server returned an error response.
    ///
    /// A 404 with a "model not found" message means the model has not been
    /// pulled yet — run `ollama pull <model>` first.
    #[error("ollama api error ({status}): {message}")]
    Api {
        /// The HTTP status code.
        status: u16,
        /// The error message, e.g. `model 'nope' not found, try pulling it first`.
        message: String,
    },
    /// The server reported an error mid-stream.
    #[error("ollama stream error: {0}")]
    StreamError(String),
}

impl OllamaError {
    /// The HTTP status code associated with this error, when there is one.
    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Api { status, .. } => Some(*status),
            Self::Http(error) => error.status().map(|status| status.as_u16()),
            Self::Connection { .. } | Self::Json(_) | Self::StreamError(_) => None,
        }
    }

    /// Returns `true` when the request was rejected for rate limiting (e.g.
    /// by Ollama's cloud or a proxy) — worth retrying with backoff.
    pub fn is_rate_limit(&self) -> bool {
        self.status() == Some(429)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_and_rate_limits_are_exposed() {
        let error = OllamaError::Api {
            status: 429,
            message: "slow down".to_string(),
        };
        assert_eq!(error.status(), Some(429));
        assert!(error.is_rate_limit());

        let error = OllamaError::Api {
            status: 404,
            message: "model not found".to_string(),
        };
        assert_eq!(error.status(), Some(404));
        assert!(!error.is_rate_limit());
    }
}
