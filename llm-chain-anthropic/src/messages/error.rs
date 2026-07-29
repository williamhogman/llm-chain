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
