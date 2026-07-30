use async_openai::types::chat::{
    ChatCompletionToolChoiceOption, ChatCompletionTools, CreateChatCompletionRequestArgs,
    ReasoningEffort, ResponseFormat, StopConfiguration, Verbosity,
};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

/// Per-step request options for OpenAI chat models.
///
/// Every option is off by default, which means the API default applies. Options are set
/// with a consuming builder style and attached to a step with
/// [`Step::with_options`](super::Step::with_options).
///
/// # Example
///
/// ```
/// use llm_chain_openai::chat::{Options, ReasoningEffort, Verbosity};
///
/// let options = Options::new()
///     .with_temperature(0.2)
///     .with_max_completion_tokens(1024)
///     .with_reasoning_effort(ReasoningEffort::Low)
///     .with_verbosity(Verbosity::Low)
///     .with_stop(["\n\n"]);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(
    feature = "serialization",
    derive(Serialize, Deserialize),
    serde(default)
)]
pub struct Options {
    /// Sampling temperature, between 0.0 and 2.0. Higher is more random.
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
    /// Upper bound on generated tokens, including reasoning tokens.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    max_completion_tokens: Option<u32>,
    /// Sequences at which the model stops generating (at most 4).
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    stop: Option<StopConfiguration>,
    /// Penalizes tokens by their frequency so far, between -2.0 and 2.0.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    frequency_penalty: Option<f32>,
    /// Penalizes tokens that already appeared, between -2.0 and 2.0.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    presence_penalty: Option<f32>,
    /// How much reasoning the model should do before answering (reasoning models only).
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    reasoning_effort: Option<ReasoningEffort>,
    /// How verbose the answer should be (GPT-5 and later).
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    verbosity: Option<Verbosity>,
    /// The output format: plain text, JSON mode, or a strict JSON schema.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    response_format: Option<ResponseFormat>,
    /// Tools the model may call.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    tools: Option<Vec<ChatCompletionTools>>,
    /// How the model chooses among the tools.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    /// Whether the model may call several tools in parallel.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    parallel_tool_calls: Option<bool>,
}

impl Options {
    /// Creates an empty set of options; the API defaults apply for everything.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true when no option is set, i.e. the API defaults apply.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Sets the sampling temperature (0.0–2.0). Higher is more random.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the nucleus-sampling probability mass (0.0–1.0).
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Caps the number of generated tokens, including reasoning tokens.
    pub fn with_max_completion_tokens(mut self, max_completion_tokens: u32) -> Self {
        self.max_completion_tokens = Some(max_completion_tokens);
        self
    }

    /// Sets up to four sequences at which the model stops generating.
    pub fn with_stop<S: Into<String>>(mut self, stop: impl IntoIterator<Item = S>) -> Self {
        self.stop = Some(StopConfiguration::StringArray(
            stop.into_iter().map(Into::into).collect(),
        ));
        self
    }

    /// Sets the frequency penalty (-2.0–2.0).
    pub fn with_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.frequency_penalty = Some(frequency_penalty);
        self
    }

    /// Sets the presence penalty (-2.0–2.0).
    pub fn with_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.presence_penalty = Some(presence_penalty);
        self
    }

    /// Sets the reasoning effort (reasoning models only).
    pub fn with_reasoning_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(reasoning_effort);
        self
    }

    /// Sets the answer verbosity (GPT-5 and later).
    pub fn with_verbosity(mut self, verbosity: Verbosity) -> Self {
        self.verbosity = Some(verbosity);
        self
    }

    /// Sets the response format: plain text, JSON mode, or a strict JSON schema.
    pub fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    /// Gives the model tools to call; build them with
    /// [`function_tool`](super::function_tool).
    ///
    /// When the model calls one, the response message's `tool_calls` is
    /// non-empty (see [`function_calls`](super::function_calls)); answer with
    /// [`assistant_tool_calls_message`](super::assistant_tool_calls_message)
    /// and [`tool_result_message`](super::tool_result_message)s, then execute
    /// the request again.
    pub fn with_tools<I: IntoIterator<Item = ChatCompletionTools>>(mut self, tools: I) -> Self {
        self.tools = Some(tools.into_iter().collect());
        self
    }

    /// Controls how the model chooses among the tools (default: `auto` when
    /// tools are present).
    pub fn with_tool_choice(mut self, tool_choice: ChatCompletionToolChoiceOption) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Sets whether the model may call several tools in parallel (the API
    /// default is `true`).
    pub fn with_parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = Some(parallel_tool_calls);
        self
    }

    /// Applies every set option to a request builder.
    pub(crate) fn apply(&self, args: &mut CreateChatCompletionRequestArgs) {
        if let Some(temperature) = self.temperature {
            args.temperature(temperature);
        }
        if let Some(top_p) = self.top_p {
            args.top_p(top_p);
        }
        if let Some(max_completion_tokens) = self.max_completion_tokens {
            args.max_completion_tokens(max_completion_tokens);
        }
        if let Some(stop) = &self.stop {
            args.stop(stop.clone());
        }
        if let Some(frequency_penalty) = self.frequency_penalty {
            args.frequency_penalty(frequency_penalty);
        }
        if let Some(presence_penalty) = self.presence_penalty {
            args.presence_penalty(presence_penalty);
        }
        if let Some(reasoning_effort) = &self.reasoning_effort {
            args.reasoning_effort(reasoning_effort.clone());
        }
        if let Some(verbosity) = &self.verbosity {
            args.verbosity(verbosity.clone());
        }
        if let Some(response_format) = &self.response_format {
            args.response_format(response_format.clone());
        }
        if let Some(tools) = &self.tools {
            args.tools(tools.clone());
        }
        if let Some(tool_choice) = &self.tool_choice {
            args.tool_choice(tool_choice.clone());
        }
        if let Some(parallel_tool_calls) = self.parallel_tool_calls {
            args.parallel_tool_calls(parallel_tool_calls);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_leave_request_untouched() {
        let mut args = CreateChatCompletionRequestArgs::default();
        args.model("gpt-5.6-terra").messages(vec![]);
        Options::new().apply(&mut args);
        let request = args.build().unwrap();
        assert_eq!(request.temperature, None);
        assert_eq!(request.max_completion_tokens, None);
        assert_eq!(request.reasoning_effort, None);
    }

    #[test]
    fn options_are_applied_to_the_request() {
        let mut args = CreateChatCompletionRequestArgs::default();
        args.model("gpt-5.6-terra").messages(vec![]);
        Options::new()
            .with_temperature(0.25)
            .with_top_p(0.9)
            .with_max_completion_tokens(256)
            .with_stop(["END"])
            .with_reasoning_effort(ReasoningEffort::Low)
            .with_verbosity(Verbosity::Low)
            .apply(&mut args);
        let request = args.build().unwrap();
        assert_eq!(request.temperature, Some(0.25));
        assert_eq!(request.top_p, Some(0.9));
        assert_eq!(request.max_completion_tokens, Some(256));
        assert_eq!(
            request.stop,
            Some(StopConfiguration::StringArray(vec!["END".to_string()]))
        );

        assert_eq!(request.reasoning_effort, Some(ReasoningEffort::Low));
        assert_eq!(request.verbosity, Some(Verbosity::Low));
    }

    #[test]
    fn tools_are_applied_to_the_request() {
        let tool = crate::chat::function_tool(
            "get_weather",
            "Get the current weather for a city.",
            serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        );
        let mut args = CreateChatCompletionRequestArgs::default();
        args.model("gpt-5.6-terra").messages(vec![]);
        Options::new()
            .with_tools([tool.clone()])
            .with_tool_choice(ChatCompletionToolChoiceOption::Mode(
                async_openai::types::chat::ToolChoiceOptions::Required,
            ))
            .with_parallel_tool_calls(false)
            .apply(&mut args);
        let request = args.build().unwrap();
        assert_eq!(request.tools, Some(vec![tool]));
        assert_eq!(request.parallel_tool_calls, Some(false));
        assert!(matches!(
            request.tool_choice,
            Some(ChatCompletionToolChoiceOption::Mode(
                async_openai::types::chat::ToolChoiceOptions::Required
            ))
        ));
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
            .with_reasoning_effort(ReasoningEffort::High);
        let yaml = serde_yaml_ng::to_string(&options).unwrap();
        let parsed: Options = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed, options);
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn tool_options_round_trip_through_yaml() {
        let options = Options::new().with_tools([crate::chat::function_tool(
            "get_weather",
            "Get the current weather for a city.",
            serde_json::json!({"type": "object"}),
        )]);
        let yaml = serde_yaml_ng::to_string(&options).unwrap();
        let parsed: Options = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed, options);
    }
}
