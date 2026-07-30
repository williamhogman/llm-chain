use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use llama_cpp_2::LlamaCppError;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use llm_chain::Parameters;
use llm_chain::traits;

use crate::config::ModelConfig;
use crate::error::LlamaError;
use crate::output::Output;
use crate::step::{LlamaInvocation, Step as LlamaStep};

/// Buffer size (in bytes) used when converting a single token back to text.
/// Large enough for any token piece produced by current vocabularies.
const TOKEN_PIECE_BUFFER_SIZE: usize = 256;

/// Returns the process-wide llama.cpp backend, initializing it on first use.
///
/// llama.cpp only allows the backend to be initialized once per process, so it
/// is stored in a static and shared by every [`Executor`].
fn backend() -> Result<&'static LlamaBackend, LlamaCppError> {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    static INIT_LOCK: Mutex<()> = Mutex::new(());

    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    let _guard = INIT_LOCK.lock().expect("backend init lock poisoned");
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    let backend = LlamaBackend::init()?;
    Ok(BACKEND.get_or_init(|| backend))
}

/// Executor is responsible for running the LLaMA model and managing its context.
///
/// It loads a GGUF model file via [llama.cpp](https://github.com/ggml-org/llama.cpp)
/// and keeps the weights in memory; each invocation creates a fresh inference
/// context, so a single executor can be reused across chain runs.
pub struct Executor {
    backend: &'static LlamaBackend,
    model: LlamaModel,
    config: ModelConfig,
}

impl Executor {
    /// Loads the GGUF model at `model_path` with the given configuration.
    pub fn new_with_config(
        model_path: impl AsRef<Path>,
        config: ModelConfig,
    ) -> Result<Self, LlamaError> {
        let backend = backend()?;
        let model_params = LlamaModelParams::default().with_n_gpu_layers(config.n_gpu_layers);
        let model = LlamaModel::load_from_file(backend, model_path, &model_params)?;
        Ok(Self {
            backend,
            model,
            config,
        })
    }

    /// Loads the GGUF model at `model_path` with the default configuration.
    pub fn new(model_path: impl Into<PathBuf>) -> Result<Self, LlamaError> {
        Self::new_with_config(model_path.into(), ModelConfig::default())
    }

    /// Runs the model with the provided invocation and returns the generated output.
    fn run_model(&self, input: &LlamaInvocation) -> Result<Output, LlamaError> {
        // Tokenize the prompt.
        let tokens = self.model.str_to_token(&input.prompt, AddBos::Always)?;

        // Create a fresh context, sized to fit the prompt in a single decode.
        let n_batch = self.config.n_batch.max(tokens.len() as u32);
        let context_params = LlamaContextParams::default()
            .with_n_ctx(self.config.n_ctx.and_then(NonZeroU32::new))
            .with_n_batch(n_batch)
            .with_n_threads(input.n_threads)
            .with_n_threads_batch(input.n_threads);
        let mut ctx = self.model.new_context(self.backend, context_params)?;

        let n_ctx = ctx.n_ctx() as usize;
        if tokens.len() >= n_ctx {
            return Err(LlamaError::PromptTooLong {
                prompt_tokens: tokens.len(),
                n_ctx,
            });
        }

        // Decode the full prompt, requesting logits for the last token only.
        let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
        let last_index = tokens.len() as i32 - 1;
        for (i, token) in (0i32..).zip(tokens.iter()) {
            batch.add(*token, i, &[0], i == last_index)?;
        }
        ctx.decode(&mut batch)?;

        // Generate tokens one at a time until end-of-generation, the stop
        // sequence, the token budget or the context window is reached.
        let mut sampler = build_sampler(input);
        let max_new_tokens = if input.n_tok_predict == 0 {
            usize::MAX
        } else {
            input.n_tok_predict
        };
        let mut output_bytes: Vec<u8> = Vec::new();
        let mut n_cur = tokens.len();
        let mut n_generated = 0usize;

        while n_cur < n_ctx && n_generated < max_new_tokens {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            if self.model.is_eog_token(token) {
                break;
            }
            output_bytes.extend_from_slice(&self.model.token_to_piece_bytes(
                token,
                TOKEN_PIECE_BUFFER_SIZE,
                false,
                None,
            )?);
            n_generated += 1;

            if let Some(stop) = input.stop_sequence.as_deref() {
                let text = String::from_utf8_lossy(&output_bytes);
                if let Some(stop_index) = text.find(stop) {
                    return Ok(Output::from(text[..stop_index].to_string()));
                }
            }

            batch.clear();
            batch.add(token, n_cur as i32, &[0], true)?;
            n_cur += 1;
            ctx.decode(&mut batch)?;
        }

        Ok(Output::from(
            String::from_utf8_lossy(&output_bytes).into_owned(),
        ))
    }
}

/// Builds the sampler chain for an invocation. A temperature of 0.0 or lower
/// selects greedy sampling.
fn build_sampler(input: &LlamaInvocation) -> LlamaSampler {
    let mut samplers = vec![LlamaSampler::penalties(64, input.repeat_penalty, 0.0, 0.0)];
    if input.temp <= 0.0 {
        samplers.push(LlamaSampler::greedy());
    } else {
        samplers.push(LlamaSampler::top_k(input.top_k));
        samplers.push(LlamaSampler::top_p(input.top_p, 1));
        samplers.push(LlamaSampler::temp(input.temp));
        samplers.push(LlamaSampler::dist(input.seed));
    }
    LlamaSampler::chain_simple(samplers)
}

// Implement the Executor trait for the Executor, defining methods for handling input and output.
impl traits::Executor for Executor {
    type Step = LlamaStep;
    type Output = Output;
    type Error = LlamaError;

    /// Executes the model and returns the output.
    ///
    /// Note: inference runs on the calling thread. When driving this from an
    /// async runtime with other tasks, consider wrapping calls in
    /// `tokio::task::spawn_blocking` or a dedicated thread.
    async fn execute(
        &self,
        input: <<Executor as traits::Executor>::Step as traits::Step>::Output,
    ) -> Result<Self::Output, Self::Error> {
        self.run_model(&input)
    }

    // Applies the output to the given parameters.
    fn apply_output_to_parameters(parameters: Parameters, output: &Self::Output) -> Parameters {
        parameters.with_text(output.to_string())
    }

    // Combines two outputs into a single output.
    fn combine_outputs(output: &Self::Output, other: &Self::Output) -> Self::Output {
        output.combine(other)
    }
}
