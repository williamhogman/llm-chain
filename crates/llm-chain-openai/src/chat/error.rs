use async_openai::error::OpenAIError;
use llm_chain::PromptTemplateError;
use thiserror::Error;

/// An error that occurred while formatting a chat prompt into an OpenAI request.
#[derive(Debug, Error)]
pub enum FormatError {
    /// The underlying prompt template failed to format.
    #[error(transparent)]
    Template(#[from] PromptTemplateError),
    /// Building the OpenAI request failed.
    #[error(transparent)]
    OpenAI(#[from] OpenAIError),
}
