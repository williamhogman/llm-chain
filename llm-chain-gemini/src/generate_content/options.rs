#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use super::types::{GenerateContentRequest, GenerationConfig, ThinkingConfig, ThinkingLevel};

/// Per-step request options for the Gemini API.
///
/// Every option is off by default, which means the API default applies.
/// Options are set with a consuming builder style and attached to a step with
/// [`Step::with_options`](super::Step::with_options).
///
/// # Example
///
/// ```
/// use llm_chain_gemini::generate_content::{Options, ThinkingLevel};
///
/// let options = Options::new()
///     .with_temperature(0.2)
///     .with_max_output_tokens(2048)
///     .with_thinking_level(ThinkingLevel::Low)
///     .with_stop_sequences(["\n\n"]);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(
    feature = "serialization",
    derive(Serialize, Deserialize),
    serde(default)
)]
pub struct Options {
    /// Upper bound on generated tokens, including thinking tokens.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    max_output_tokens: Option<u32>,
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
    /// Sequences at which the model stops generating.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    stop_sequences: Option<Vec<String>>,
    /// Thinking depth for Gemini 3-generation models.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    thinking_level: Option<ThinkingLevel>,
    /// Thinking token budget for Gemini 2.5-generation models.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    thinking_budget: Option<i32>,
}

impl Options {
    /// Creates an empty set of options; the API defaults apply for everything.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true when no option is set, i.e. the defaults apply.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Caps the number of generated tokens, including thinking tokens.
    pub fn with_max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
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

    /// Sets sequences at which the model stops generating.
    pub fn with_stop_sequences<S: Into<String>>(
        mut self,
        stop_sequences: impl IntoIterator<Item = S>,
    ) -> Self {
        self.stop_sequences = Some(stop_sequences.into_iter().map(Into::into).collect());
        self
    }

    /// Sets the thinking depth (Gemini 3 generation).
    ///
    /// Gemini 3 models default to [`ThinkingLevel::High`]; lower it for
    /// faster, cheaper responses. Gemini 2.5-generation models use
    /// [`Options::with_thinking_budget`] instead.
    pub fn with_thinking_level(mut self, thinking_level: ThinkingLevel) -> Self {
        self.thinking_level = Some(thinking_level);
        self
    }

    /// Sets the thinking token budget (Gemini 2.5 generation).
    ///
    /// `0` disables thinking (where supported) and `-1` enables dynamic
    /// thinking. Gemini 3-generation models use
    /// [`Options::with_thinking_level`] instead.
    pub fn with_thinking_budget(mut self, thinking_budget: i32) -> Self {
        self.thinking_budget = Some(thinking_budget);
        self
    }

    /// Applies every set option to a request.
    pub(crate) fn apply(&self, request: &mut GenerateContentRequest) {
        let thinking_config = if self.thinking_level.is_some() || self.thinking_budget.is_some() {
            Some(ThinkingConfig {
                thinking_level: self.thinking_level,
                thinking_budget: self.thinking_budget,
            })
        } else {
            None
        };
        let config = GenerationConfig {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            max_output_tokens: self.max_output_tokens,
            stop_sequences: self.stop_sequences.clone(),
            thinking_config,
        };
        request.generation_config = if config.is_empty() { None } else { Some(config) };
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{Content, Role};
    use super::*;

    fn base_request() -> GenerateContentRequest {
        GenerateContentRequest {
            model: "gemini-3.6-flash".to_string(),
            contents: vec![Content::text(Role::User, "hi")],
            system_instruction: None,
            generation_config: None,
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
            .with_max_output_tokens(4096)
            .with_temperature(0.3)
            .with_top_k(40)
            .with_stop_sequences(["END"])
            .with_thinking_level(ThinkingLevel::Low)
            .apply(&mut request);
        let config = request.generation_config.expect("config set");
        assert_eq!(config.max_output_tokens, Some(4096));
        assert_eq!(config.temperature, Some(0.3));
        assert_eq!(config.top_k, Some(40));
        assert_eq!(config.stop_sequences, Some(vec!["END".to_string()]));
        assert_eq!(
            config.thinking_config,
            Some(ThinkingConfig {
                thinking_level: Some(ThinkingLevel::Low),
                thinking_budget: None,
            })
        );
    }

    #[test]
    fn thinking_budget_builds_a_thinking_config() {
        let mut request = base_request();
        Options::new().with_thinking_budget(2048).apply(&mut request);
        let config = request.generation_config.expect("config set");
        assert_eq!(
            config.thinking_config,
            Some(ThinkingConfig {
                thinking_level: None,
                thinking_budget: Some(2048),
            })
        );
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
            .with_max_output_tokens(2048);
        let yaml = serde_yaml_ng::to_string(&options).unwrap();
        let parsed: Options = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed, options);
    }
}
