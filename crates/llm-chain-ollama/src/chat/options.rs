#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use super::types::{ChatRequest, Format, Think, Tool};

/// Per-step request options for Ollama's chat API.
///
/// Every option is off by default, which means the model's own defaults apply
/// (from its Modelfile). Options are set with a consuming builder style and
/// attached to a step with [`Step::with_options`](super::Step::with_options).
///
/// # Example
///
/// ```
/// use llm_chain_ollama::chat::{Options, Think};
///
/// let options = Options::new()
///     .with_temperature(0.2)
///     .with_num_predict(512)
///     .with_think(Think::Enabled)
///     .with_stop_sequences(["\n\n"]);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(
    feature = "serialization",
    derive(Serialize, Deserialize),
    serde(default)
)]
pub struct Options {
    /// Sampling temperature. Higher is more random.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    temperature: Option<f32>,
    /// Nucleus sampling: only tokens comprising the top `top_p` probability mass are considered.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    top_p: Option<f32>,
    /// Only sample from the `top_k` most likely tokens.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    top_k: Option<u32>,
    /// Minimum token probability relative to the most likely token.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    min_p: Option<f32>,
    /// Upper bound on generated tokens.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    num_predict: Option<i32>,
    /// The context window size in tokens.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    num_ctx: Option<u32>,
    /// Penalty for repeating tokens.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    repeat_penalty: Option<f32>,
    /// Random seed for reproducible generation.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    seed: Option<i64>,
    /// Sequences at which the model stops generating.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    stop_sequences: Option<Vec<String>>,
    /// Thinking control for reasoning-capable models.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    think: Option<Think>,
    /// Structured-output control.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    format: Option<Format>,
    /// How long to keep the model loaded after the request.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    keep_alive: Option<String>,
}

impl Options {
    /// Creates an empty set of options; the model defaults apply for everything.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true when no option is set, i.e. the defaults apply.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Sets the sampling temperature. Higher is more random.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the nucleus-sampling probability mass (0.0–1.0).
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Only sample from the `top_k` most likely tokens.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Sets the minimum token probability relative to the most likely token.
    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = Some(min_p);
        self
    }

    /// Caps the number of generated tokens (`num_predict`); `-1` means unlimited.
    pub fn with_num_predict(mut self, num_predict: i32) -> Self {
        self.num_predict = Some(num_predict);
        self
    }

    /// Sets the context window size in tokens (`num_ctx`).
    pub fn with_num_ctx(mut self, num_ctx: u32) -> Self {
        self.num_ctx = Some(num_ctx);
        self
    }

    /// Sets the penalty for repeating tokens.
    pub fn with_repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.repeat_penalty = Some(repeat_penalty);
        self
    }

    /// Sets the random seed for reproducible generation (with temperature 0).
    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets sequences at which the model stops generating.
    pub fn with_stop_sequences<S: Into<String>>(
        mut self,
        stop_sequences: impl IntoIterator<Item = S>,
    ) -> Self {
        self.stop_sequences = Some(stop_sequences.into_iter().map(Into::into).collect());
        self
    }

    /// Controls thinking on reasoning-capable models.
    ///
    /// Use [`Think::Enabled`]/[`Think::Disabled`] for models like qwen3 or
    /// deepseek-r1, and the leveled variants for models with thinking levels
    /// (e.g. gpt-oss). The reasoning arrives separately in
    /// [`ChatResponse::thinking`](super::ChatResponse::thinking).
    pub fn with_think(mut self, think: Think) -> Self {
        self.think = Some(think);
        self
    }

    /// Forces the model to answer with valid JSON.
    ///
    /// Also instruct the model to answer in JSON in the prompt, otherwise it
    /// may fill the output with whitespace.
    pub fn with_json_output(mut self) -> Self {
        self.format = Some(Format::Json);
        self
    }

    /// Constrains the answer to the given JSON schema.
    pub fn with_format(mut self, format: Format) -> Self {
        self.format = Some(format);
        self
    }

    /// Sets how long the model stays loaded after the request, e.g. `"5m"` or `"0"`.
    pub fn with_keep_alive<S: Into<String>>(mut self, keep_alive: S) -> Self {
        self.keep_alive = Some(keep_alive.into());
        self
    }

    /// Applies every set option to a request.
    pub(crate) fn apply(&self, request: &mut ChatRequest) {
        request.options.temperature = self.temperature;
        request.options.top_p = self.top_p;
        request.options.top_k = self.top_k;
        request.options.min_p = self.min_p;
        request.options.num_predict = self.num_predict;
        request.options.num_ctx = self.num_ctx;
        request.options.repeat_penalty = self.repeat_penalty;
        request.options.seed = self.seed;
        request.options.stop = self.stop_sequences.clone();
        request.think = self.think;
        request.format = self.format.clone();
        request.keep_alive = self.keep_alive.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{Message, ModelOptions, Role};
    use super::*;

    fn base_request() -> ChatRequest {
        ChatRequest {
            model: "qwen3".to_string(),
            messages: vec![Message::new(Role::User, "hi")],
            stream: false,
            think: None,
            format: None,
            keep_alive: None,
            options: ModelOptions::default(),
        }
    }

    #[test]
    fn default_options_leave_request_untouched() {
        let mut request = base_request();
        Options::new().apply(&mut request);
        assert_eq!(request, base_request());
    }

    #[test]
    fn options_are_applied_to_the_request() {
        let mut request = base_request();
        Options::new()
            .with_temperature(0.3)
            .with_top_k(40)
            .with_num_predict(256)
            .with_num_ctx(8192)
            .with_seed(42)
            .with_stop_sequences(["END"])
            .with_think(Think::High)
            .with_json_output()
            .with_keep_alive("5m")
            .apply(&mut request);
        assert_eq!(request.options.temperature, Some(0.3));
        assert_eq!(request.options.top_k, Some(40));
        assert_eq!(request.options.num_predict, Some(256));
        assert_eq!(request.options.num_ctx, Some(8192));
        assert_eq!(request.options.seed, Some(42));
        assert_eq!(request.options.stop, Some(vec!["END".to_string()]));
        assert_eq!(request.think, Some(Think::High));
        assert_eq!(request.format, Some(Format::Json));
        assert_eq!(request.keep_alive.as_deref(), Some("5m"));
    }

    #[test]
    fn is_default_detects_empty_options() {
        assert!(Options::new().is_default());
        assert!(!Options::new().with_temperature(0.1).is_default());
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn options_round_trip_through_yaml() {
        let options = Options::new()
            .with_temperature(0.5)
            .with_num_predict(2048)
            .with_think(Think::Enabled);
        let yaml = serde_yaml_ng::to_string(&options).unwrap();
        let parsed: Options = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed, options);
    }
}
