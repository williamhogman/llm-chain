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
