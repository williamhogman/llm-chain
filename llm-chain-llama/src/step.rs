#[cfg(feature = "serialization")]
use llm_chain::serialization::StorableEntity;
use llm_chain::{Parameters, PromptTemplate, PromptTemplateError, traits};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

/// Represents a concrete call to the LLaMA model, with all the parameters specified, and no implicit behavior.
#[derive(Debug, Clone)]
pub struct LlamaInvocation {
    pub(crate) n_threads: i32,
    pub(crate) n_tok_predict: usize,
    pub(crate) top_k: i32,
    pub(crate) top_p: f32,
    pub(crate) temp: f32,
    pub(crate) repeat_penalty: f32,
    pub(crate) seed: u32,
    pub(crate) stop_sequence: Option<String>,
    pub(crate) prompt: String,
}

/// `LlamaConfig` is an overridable collection of sampling parameters for the
/// LLaMA model. It is combined with a prompt to create an invocation.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct LlamaConfig {
    /// Number of CPU threads to use. Defaults to the number of available cores.
    pub n_threads: Option<i32>,
    /// Maximum number of tokens to generate. `0` (the default) generates
    /// until the model emits an end-of-generation token or the context is full.
    pub n_tok_predict: Option<usize>,
    /// Top-K sampling cutoff. Defaults to 40.
    pub top_k: Option<i32>,
    /// Top-P (nucleus) sampling cutoff. Defaults to 0.95.
    pub top_p: Option<f32>,
    /// Sampling temperature. Defaults to 0.8. A value of 0.0 or lower selects
    /// greedy sampling.
    pub temp: Option<f32>,
    /// Repetition penalty. Defaults to 1.1.
    pub repeat_penalty: Option<f32>,
    /// Seed for the random sampler. Defaults to 1234; set for reproducible output.
    pub seed: Option<u32>,
    /// A sequence that stops generation when produced by the model. Defaults
    /// to none, generating until end-of-generation.
    pub stop_sequence: Option<String>,
}

impl LlamaConfig {
    /// Creates a new `LlamaConfig` instance with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Converts the current `LlamaConfig` instance to a [`LlamaInvocation`], using the given prompt.
    fn to_invocation(&self, prompt: String) -> LlamaInvocation {
        LlamaInvocation {
            n_threads: self.n_threads.unwrap_or_else(default_n_threads),
            n_tok_predict: self.n_tok_predict.unwrap_or(0),
            top_k: self.top_k.unwrap_or(40),
            top_p: self.top_p.unwrap_or(0.95),
            temp: self.temp.unwrap_or(0.8),
            repeat_penalty: self.repeat_penalty.unwrap_or(1.1),
            seed: self.seed.unwrap_or(1234),
            stop_sequence: self.stop_sequence.clone(),
            prompt,
        }
    }
}

fn default_n_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get().try_into().unwrap_or(i32::MAX))
        .unwrap_or(1)
}

/// A step in a chain of LLaMA invocations. It is a combination of a prompt and a configuration.
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct Step {
    prompt: PromptTemplate,
    config: LlamaConfig,
}

impl Step {
    /// Create a new step with the given prompt and configuration.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The prompt template for the step.
    /// * `config` - An optional configuration for the step. If `None`, the default configuration will be used.
    pub fn new_with_config(prompt: PromptTemplate, config: Option<LlamaConfig>) -> Self {
        Self {
            prompt,
            config: config.unwrap_or_default(),
        }
    }

    /// Create a new step with the given prompt and default configuration.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The prompt template for the step.
    pub fn new(prompt: PromptTemplate) -> Self {
        Self::new_with_config(prompt, None)
    }
}

impl traits::Step for Step {
    type Output = LlamaInvocation;
    type Error = PromptTemplateError;

    /// Formats the current step using the given parameters, creating a [`LlamaInvocation`] instance.
    fn format(&self, parameters: &Parameters) -> Result<Self::Output, Self::Error> {
        Ok(self.config.to_invocation(self.prompt.format(parameters)?))
    }
}

#[cfg(feature = "serialization")]
impl StorableEntity for Step {
    fn get_metadata() -> Vec<(String, String)> {
        vec![(
            "step-type".to_string(),
            "llm-chain-llama::step::Step".to_string(),
        )]
    }
}
