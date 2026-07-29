#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use super::types::{Effort, MessagesRequest, Thinking};

/// The default `max_tokens` for a step that does not override it.
///
/// The Messages API requires `max_tokens` on every request; this crate defaults
/// to a generous but bounded value.
pub const DEFAULT_MAX_TOKENS: u32 = 1024;

/// Per-step request options for Anthropic's Messages API.
///
/// Every option is off by default, which means the API default applies (except
/// `max_tokens`, which the API requires and which defaults to
/// [`DEFAULT_MAX_TOKENS`]). Options are set with a consuming builder style and
/// attached to a step with [`Step::with_options`](super::Step::with_options).
///
/// # Example
///
/// ```
/// use llm_chain_anthropic::messages::Options;
///
/// let options = Options::new()
///     .with_temperature(0.2)
///     .with_max_tokens(2048)
///     .with_thinking_budget(1024)
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
    max_tokens: Option<u32>,
    /// Sampling temperature, between 0.0 and 1.0. Higher is more random.
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
    /// Extended-thinking token budget; enables thinking when set.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    thinking_budget: Option<u32>,
    /// Reasoning effort for Claude 5-generation models.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    effort: Option<Effort>,
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
    ///
    /// Defaults to [`DEFAULT_MAX_TOKENS`] when unset.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the sampling temperature (0.0–1.0). Higher is more random.
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

    /// Enables extended thinking with the given token budget.
    ///
    /// Supported by Claude Haiku 4.5 and the 4.x generation; the budget must be
    /// less than `max_tokens`, so raise [`Options::with_max_tokens`]
    /// accordingly. Claude 5-generation models think adaptively — use
    /// [`Options::with_effort`] for those instead.
    pub fn with_thinking_budget(mut self, thinking_budget: u32) -> Self {
        self.thinking_budget = Some(thinking_budget);
        self
    }

    /// Sets the reasoning effort (Claude 5 generation and Opus 4.8+).
    ///
    /// The API defaults to [`Effort::High`] on supported models; lower it for
    /// faster, cheaper responses.
    pub fn with_effort(mut self, effort: Effort) -> Self {
        self.effort = Some(effort);
        self
    }

    /// Applies every set option to a request.
    pub(crate) fn apply(&self, request: &mut MessagesRequest) {
        if let Some(max_tokens) = self.max_tokens {
            request.max_tokens = max_tokens;
        }
        request.temperature = self.temperature;
        request.top_p = self.top_p;
        request.top_k = self.top_k;
        request.stop_sequences = self.stop_sequences.clone();
        request.thinking = self
            .thinking_budget
            .map(|budget_tokens| Thinking::Enabled { budget_tokens });
        request.effort = self.effort;
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{Message, Role};
    use super::*;

    fn base_request() -> MessagesRequest {
        MessagesRequest {
            model: "claude-sonnet-5".to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            system: None,
            messages: vec![Message {
                role: Role::User,
                content: "hi".to_string(),
            }],
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            thinking: None,
            effort: None,
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
            .with_max_tokens(4096)
            .with_temperature(0.3)
            .with_top_k(40)
            .with_stop_sequences(["END"])
            .with_thinking_budget(2048)
            .with_effort(Effort::Medium)
            .apply(&mut request);
        assert_eq!(request.max_tokens, 4096);
        assert_eq!(request.temperature, Some(0.3));
        assert_eq!(request.top_k, Some(40));
        assert_eq!(request.stop_sequences, Some(vec!["END".to_string()]));
        assert_eq!(
            request.thinking,
            Some(Thinking::Enabled {
                budget_tokens: 2048
            })
        );
        assert_eq!(request.effort, Some(Effort::Medium));
    }

    #[test]
    fn is_default_detects_empty_options() {
        assert!(Options::new().is_default());
        assert!(!Options::new().with_temperature(0.1).is_default());
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn options_round_trip_through_yaml() {
        let options = Options::new().with_temperature(0.5).with_max_tokens(2048);
        let yaml = serde_yaml_ng::to_string(&options).unwrap();
        let parsed: Options = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed, options);
    }
}
