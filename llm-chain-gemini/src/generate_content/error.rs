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
