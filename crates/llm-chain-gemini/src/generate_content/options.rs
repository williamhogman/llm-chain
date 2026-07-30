#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use super::types::{
    FunctionCallingConfig, FunctionCallingMode, FunctionDeclaration, GenerateContentRequest,
    GenerationConfig, ThinkingConfig, ThinkingLevel, Tool, ToolConfig,
};

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
    /// Whether the API should include thought summaries in the response.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    include_thoughts: Option<bool>,
    /// MIME type of the response, e.g. `application/json`.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    response_mime_type: Option<String>,
    /// Functions the model may call.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    tools: Option<Vec<FunctionDeclaration>>,
    /// How the model chooses among the functions.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    function_calling_mode: Option<FunctionCallingMode>,
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

    /// Asks the API to include thought summaries in the response.
    ///
    /// Thought parts carry `thought == true` and are excluded from
    /// [`GenerateContentResponse::text`](super::types::GenerateContentResponse::text);
    /// walk the candidate's parts to read them.
    pub fn with_include_thoughts(mut self, include_thoughts: bool) -> Self {
        self.include_thoughts = Some(include_thoughts);
        self
    }

    /// Sets the response MIME type, e.g. `application/json` for JSON output.
    pub fn with_response_mime_type<S: Into<String>>(mut self, response_mime_type: S) -> Self {
        self.response_mime_type = Some(response_mime_type.into());
        self
    }

    /// Gives the model functions to call.
    ///
    /// When the model calls one, the response carries function-call parts (see
    /// [`GenerateContentResponse::function_calls`](super::types::GenerateContentResponse::function_calls));
    /// run the functions and continue with
    /// [`GenerateContentRequest::with_function_responses`](super::types::GenerateContentRequest::with_function_responses).
    ///
    /// Use [`ToolCollection::tool_specs`](https://docs.rs/llm-chain-tools) to
    /// bridge an existing `llm-chain-tools` collection into declarations.
    pub fn with_tools<I: IntoIterator<Item = FunctionDeclaration>>(mut self, tools: I) -> Self {
        self.tools = Some(tools.into_iter().collect());
        self
    }

    /// Controls how the model chooses among the functions (default:
    /// [`FunctionCallingMode::Auto`] when functions are present).
    pub fn with_function_calling_mode(mut self, mode: FunctionCallingMode) -> Self {
        self.function_calling_mode = Some(mode);
        self
    }

    /// Applies every set option to a request.
    pub(crate) fn apply(&self, request: &mut GenerateContentRequest) {
        let thinking_config = if self.thinking_level.is_some()
            || self.thinking_budget.is_some()
            || self.include_thoughts.is_some()
        {
            Some(ThinkingConfig {
                thinking_level: self.thinking_level,
                thinking_budget: self.thinking_budget,
                include_thoughts: self.include_thoughts,
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
            response_mime_type: self.response_mime_type.clone(),
            thinking_config,
        };
        request.generation_config = if config.is_empty() {
            None
        } else {
            Some(config)
        };
        request.tools = self.tools.clone().map(|function_declarations| {
            vec![Tool {
                function_declarations,
            }]
        });
        request.tool_config = self.function_calling_mode.map(|mode| ToolConfig {
            function_calling_config: Some(FunctionCallingConfig {
                mode: Some(mode),
                allowed_function_names: None,
            }),
        });
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
            tools: None,
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
                include_thoughts: None,
            })
        );
    }

    #[test]
    fn thinking_budget_builds_a_thinking_config() {
        let mut request = base_request();
        Options::new()
            .with_thinking_budget(2048)
            .apply(&mut request);
        let config = request.generation_config.expect("config set");
        assert_eq!(
            config.thinking_config,
            Some(ThinkingConfig {
                thinking_level: None,
                thinking_budget: Some(2048),
                include_thoughts: None,
            })
        );
    }

    #[test]
    fn include_thoughts_and_json_mode_are_applied() {
        let mut request = base_request();
        Options::new()
            .with_include_thoughts(true)
            .with_response_mime_type("application/json")
            .apply(&mut request);
        let config = request.generation_config.expect("config set");
        assert_eq!(
            config.response_mime_type,
            Some("application/json".to_string())
        );
        assert_eq!(
            config.thinking_config,
            Some(ThinkingConfig {
                thinking_level: None,
                thinking_budget: None,
                include_thoughts: Some(true),
            })
        );
    }

    #[test]
    fn is_default_detects_empty_options() {
        assert!(Options::new().is_default());
        assert!(!Options::new().with_temperature(0.1).is_default());
    }

    #[test]
    fn tools_are_applied_to_the_request() {
        let mut request = base_request();
        let declaration = FunctionDeclaration::new(
            "get_weather",
            "Get the current weather for a city.",
            serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        );
        Options::new()
            .with_tools([declaration.clone()])
            .with_function_calling_mode(FunctionCallingMode::Any)
            .apply(&mut request);
        assert_eq!(
            request.tools,
            Some(vec![Tool {
                function_declarations: vec![declaration]
            }])
        );
        assert_eq!(
            request
                .tool_config
                .as_ref()
                .and_then(|config| config.function_calling_config.as_ref())
                .and_then(|config| config.mode),
            Some(FunctionCallingMode::Any)
        );
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
