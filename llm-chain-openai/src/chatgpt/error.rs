use async_openai::error::OpenAIError;
use async_openai::types::chat::Role;
use llm_chain::PromptTemplateError;
use thiserror::Error;

/// An error that occurred while formatting a chat prompt into an OpenAI request.
#[derive(Debug, Error)]
pub enum FormatError {
    /// The underlying prompt template failed to format.
    #[error(transparent)]
    Template(#[from] PromptTemplateError),
    /// The role cannot be used in a templated message (e.g. `Tool`, which requires a tool call id).
    #[error("role {0:?} cannot be used in a prompt template")]
    UnsupportedRole(Role),
    /// Building the OpenAI request failed.
    #[error(transparent)]
    OpenAI(#[from] OpenAIError),
}
