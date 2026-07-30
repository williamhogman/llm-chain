/// Configuration for how a model is loaded and how large its inference
/// context is. These settings apply to the model as a whole, while
/// [`crate::LlamaConfig`] controls per-invocation sampling.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// The size of the context window in tokens. `None` uses the model's own
    /// training context length.
    pub n_ctx: Option<u32>,
    /// The number of layers to offload to the GPU. Requires building with one
    /// of the GPU features (`cuda`, `metal`, `vulkan`) to have an effect.
    /// Use a large value such as `1_000_000` to offload all layers.
    pub n_gpu_layers: u32,
    /// The logical batch size used when decoding the prompt. Automatically
    /// grown to fit the prompt.
    pub n_batch: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_ctx: Some(4096),
            n_gpu_layers: 0,
            n_batch: 512,
        }
    }
}

impl ModelConfig {
    /// Creates a new `ModelConfig` with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the context window size in tokens.
    pub fn with_n_ctx(mut self, n_ctx: u32) -> Self {
        self.n_ctx = Some(n_ctx);
        self
    }

    /// Sets the number of layers to offload to the GPU.
    pub fn with_n_gpu_layers(mut self, n_gpu_layers: u32) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    /// Sets the logical batch size for prompt decoding.
    pub fn with_n_batch(mut self, n_batch: u32) -> Self {
        self.n_batch = n_batch;
        self
    }
}
