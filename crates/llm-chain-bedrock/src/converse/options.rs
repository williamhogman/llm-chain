#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use super::types::{ConverseRequest, InferenceConfig, Tool, ToolChoice, ToolConfiguration, ToolSpec};

/// Per-step request options for Bedrock's Converse API.
///
/// Every option is off by default, which means the model default applies.
/// Options are set with a consuming builder style and attached to a step with
/// [`Step::with_options`](super::Step::with_options).
///
/// The base options cover the parameters every Bedrock model family shares.
/// Family-specific parameters (Claude's `thinking` or `top_k`, Nova's
/// `inferenceConfig` extensions, …) pass through verbatim via
/// [`Options::with_additional_model_request_fields`].
///
/// # Example
///
/// ```
/// use llm_chain_bedrock::converse::Options;
///
/// let options = Options::new()
///     .with_temperature(0.2)
///     .with_max_tokens(2048)
///     .with_stop_sequences(["\n\n"]);
///
/// // Claude-specific extended thinking, passed through verbatim:
/// let reasoning = Options::new().with_max_tokens(4096).with_additional_model_request_fields(
///     serde_json::json!({"thinking": {"type": "enabled", "budget_tokens": 2048}}),
/// );
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(
    feature = "serialization",
    derive(Serialize, Deserialize),
    serde(default)
)]
pub struct Options {
    /// Upper bound on generated tokens.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    max_tokens: Option<u32>,
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
    /// Sequences at which the model stops generating.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    stop_sequences: Option<Vec<String>>,
    /// Model-family-specific request fields, passed through verbatim.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    additional_model_request_fields: Option<serde_json::Value>,
    /// Tools the model may call.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    tools: Option<Vec<ToolSpec>>,
    /// How the model chooses among the tools.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    tool_choice: Option<ToolChoice>,
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

    /// Caps the number of generated tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
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

    /// Sets sequences at which the model stops generating.
    pub fn with_stop_sequences<S: Into<String>>(
        mut self,
        stop_sequences: impl IntoIterator<Item = S>,
    ) -> Self {
        self.stop_sequences = Some(stop_sequences.into_iter().map(Into::into).collect());
        self
    }

    /// Passes model-family-specific request fields through verbatim, e.g.
    /// Claude's `thinking` or `top_k`.
    ///
    /// ```
    /// use llm_chain_bedrock::converse::Options;
    ///
    /// let options = Options::new().with_additional_model_request_fields(
    ///     serde_json::json!({"top_k": 40}),
    /// );
    /// ```
    pub fn with_additional_model_request_fields(mut self, fields: serde_json::Value) -> Self {
        self.additional_model_request_fields = Some(fields);
        self
    }

    /// Declares tools the model may call, sent as the request's `toolConfig`.
    ///
    /// When the model decides to call one, the response's
    /// [`stop_reason`](super::ConverseResponse::stop_reason) is
    /// [`StopReason::ToolUse`](super::StopReason::ToolUse) and the calls are
    /// available via [`ConverseResponse::tool_uses`](super::ConverseResponse::tool_uses).
    /// Answer them with
    /// [`ConverseRequest::with_tool_results`](super::ConverseRequest::with_tool_results).
    ///
    /// ```
    /// use llm_chain_bedrock::converse::{Options, ToolSpec};
    ///
    /// let options = Options::new().with_tools([ToolSpec::new(
    ///     "get_weather",
    ///     "Get the current weather in a city",
    ///     serde_json::json!({
    ///         "type": "object",
    ///         "properties": {"city": {"type": "string"}},
    ///         "required": ["city"]
    ///     }),
    /// )]);
    /// ```
    pub fn with_tools(mut self, tools: impl IntoIterator<Item = ToolSpec>) -> Self {
        self.tools = Some(tools.into_iter().collect());
        self
    }

    /// Sets how the model chooses among the tools declared with
    /// [`with_tools`](Self::with_tools). No effect unless tools are set.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Applies every set option to a request.
    pub(crate) fn apply(&self, request: &mut ConverseRequest) {
        let inference_config = InferenceConfig {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            stop_sequences: self.stop_sequences.clone(),
        };
        request.inference_config = (!inference_config.is_empty()).then_some(inference_config);
        request.additional_model_request_fields = self.additional_model_request_fields.clone();
        request.tool_config = self.tools.as_ref().map(|tools| ToolConfiguration {
            tools: tools
                .iter()
                .cloned()
                .map(|tool_spec| Tool { tool_spec })
                .collect(),
            tool_choice: self.tool_choice.clone(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{Message, Role};
    use super::*;

    fn base_request() -> ConverseRequest {
        ConverseRequest {
            model_id: "amazon.nova-pro-v1:0".to_string(),
            messages: vec![Message::text(Role::User, "hi")],
            system: None,
            inference_config: None,
            additional_model_request_fields: None,
            tool_config: None,
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
            .with_top_p(0.9)
            .with_stop_sequences(["END"])
            .with_additional_model_request_fields(serde_json::json!({"top_k": 40}))
            .apply(&mut request);
        let config = request.inference_config.expect("config set");
        assert_eq!(config.max_tokens, Some(4096));
        assert_eq!(config.temperature, Some(0.3));
        assert_eq!(config.top_p, Some(0.9));
        assert_eq!(config.stop_sequences, Some(vec!["END".to_string()]));
        assert_eq!(
            request.additional_model_request_fields,
            Some(serde_json::json!({"top_k": 40}))
        );
    }

    #[test]
    fn tools_are_applied_as_tool_config() {
        let mut request = base_request();
        Options::new()
            .with_tools([ToolSpec::new(
                "get_weather",
                "Get the weather",
                serde_json::json!({"type": "object"}),
            )])
            .with_tool_choice(ToolChoice::Any {})
            .apply(&mut request);
        let tool_config = request.tool_config.expect("tool config set");
        assert_eq!(tool_config.tools.len(), 1);
        assert_eq!(tool_config.tools[0].tool_spec.name, "get_weather");
        assert_eq!(tool_config.tool_choice, Some(ToolChoice::Any {}));
    }

    #[test]
    fn tool_choice_alone_does_not_create_tool_config() {
        let mut request = base_request();
        Options::new()
            .with_tool_choice(ToolChoice::Auto {})
            .apply(&mut request);
        assert_eq!(request.tool_config, None);
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
            .with_max_tokens(2048)
            .with_additional_model_request_fields(serde_json::json!({"top_k": 40}))
            .with_tools([ToolSpec::new(
                "get_weather",
                "Get the weather",
                serde_json::json!({"type": "object"}),
            )]);
        let yaml = serde_yaml_ng::to_string(&options).unwrap();
        let parsed: Options = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed, options);
    }
}
