use llama_cpp_2::{
    DecodeError, LlamaContextLoadError, LlamaCppError, LlamaModelLoadError, StringToTokenError,
    TokenToStringError, llama_batch::BatchAddError,
};
use thiserror::Error;

/// Errors produced when loading or running a LLaMA-family model.
#[derive(Debug, Error)]
pub enum LlamaError {
    /// The llama.cpp backend could not be initialized.
    #[error("failed to initialize the llama.cpp backend: {0}")]
    Backend(#[from] LlamaCppError),
    /// The model file could not be loaded (missing file, not a GGUF file, ...).
    #[error("failed to load model: {0}")]
    ModelLoad(#[from] LlamaModelLoadError),
    /// A llama.cpp context could not be created for the model.
    #[error("failed to create llama.cpp context: {0}")]
    ContextCreation(#[from] LlamaContextLoadError),
    /// The prompt could not be tokenized.
    #[error("failed to tokenize prompt: {0}")]
    Tokenize(#[from] StringToTokenError),
    /// A generated token could not be converted back into text.
    #[error("failed to detokenize output: {0}")]
    Detokenize(#[from] TokenToStringError),
    /// A token could not be added to the decoding batch.
    #[error("failed to add token to batch: {0}")]
    BatchAdd(#[from] BatchAddError),
    /// llama.cpp failed to decode a batch.
    #[error("failed to decode batch: {0}")]
    Decode(#[from] DecodeError),
    /// The prompt does not fit in the context window.
    #[error("the prompt is too long for the context window ({prompt_tokens} tokens >= {n_ctx})")]
    PromptTooLong {
        /// The number of tokens in the prompt.
        prompt_tokens: usize,
        /// The size of the context window.
        n_ctx: usize,
    },
}
