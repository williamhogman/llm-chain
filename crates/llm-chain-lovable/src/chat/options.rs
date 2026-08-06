#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use super::types::{ChatRequest, JsonSchema, Reasoning, ReasoningEffort, ResponseFormat, Tool};

/// Per-step request options for the Lovable AI Gateway's chat completions API.
///
/// Every option is off by default, which means the selected model's own
/// defaults apply. Options are set with a consuming builder style and
/// attached to a step with [`Step::with_options`](super::Step::with_options).
///
/// The gateway relays each request to the selected model's provider, so the
/// usual OpenAI-compatible caveats apply per model: OpenAI reasoning models
/// take [`with_max_completion_tokens`](Options::with_max_completion_tokens)
/// rather than [`with_max_tokens`](Options::with_max_tokens), and reasoning
/// is requested with [`with_reasoning_effort`](Options::with_reasoning_effort)
/// on `openai/*` models but [`with_reasoning`](Options::with_reasoning) on
/// everything else (e.g. `google/gemini-*`). A field the model does not
/// support is rejected with a 400 that names the offender.
///
/// # Example
///
/// ```
/// use llm_chain_lovable::chat::{Options, Reasoning, ReasoningEffort};
///
/// let options = Options::new()
///     .with_temperature(0.2)
///     .with_max_tokens(512)
///     .with_reasoning(Reasoning::effort(ReasoningEffort::Medium))
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
    /// Upper bound on generated tokens (`max_tokens`).
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    max_tokens: Option<u32>,
    /// Upper bound on generated tokens (`max_completion_tokens`).
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    max_completion_tokens: Option<u32>,
    /// Sequences at which the model stops generating.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    stop_sequences: Option<Vec<String>>,
    /// Random seed for best-effort reproducible generation.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    seed: Option<i64>,
    /// Structured-output control.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    response_format: Option<ResponseFormat>,
    /// OpenAI-style reasoning effort.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    reasoning_effort: Option<ReasoningEffort>,
    /// Unified reasoning option.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    reasoning: Option<Reasoning>,
    /// Tools the model may call.
    #[cfg_attr(
        feature = "serialization",
        serde(skip_serializing_if = "Option::is_none")
    )]
    tools: Option<Vec<Tool>>,
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

    /// Caps the number of generated tokens via `max_tokens`, the widely
    /// supported field. OpenAI reasoning models reject it — use
    /// [`with_max_completion_tokens`](Options::with_max_completion_tokens)
    /// for those.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Caps the number of generated tokens via `max_completion_tokens`, the
    /// field OpenAI reasoning models require. The cap covers reasoning
    /// tokens too.
    pub fn with_max_completion_tokens(mut self, max_completion_tokens: u32) -> Self {
        self.max_completion_tokens = Some(max_completion_tokens);
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

    /// Sets the random seed for best-effort reproducible generation.
    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Forces the model to answer with valid JSON (schema not enforced).
    ///
    /// Also instruct the model to answer in JSON in the prompt — providers
    /// reject `json_object` requests whose messages never mention JSON.
    /// Prefer [`with_json_schema`](Options::with_json_schema), which has no
    /// such requirement and guarantees the shape.
    pub fn with_json_output(mut self) -> Self {
        self.response_format = Some(ResponseFormat::JsonObject);
        self
    }

    /// Constrains the answer to the given JSON schema (strict mode).
    ///
    /// Keep the schema strict-compatible: an object at the root, every
    /// property required (optional fields typed nullable), and
    /// `additionalProperties: false` on every object.
    pub fn with_json_schema(mut self, name: impl Into<String>, schema: serde_json::Value) -> Self {
        self.response_format = Some(ResponseFormat::JsonSchema {
            json_schema: JsonSchema {
                name: name.into(),
                strict: Some(true),
                schema,
            },
        });
        self
    }

    /// Sets the structured-output control directly.
    pub fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    /// Sets the OpenAI-style `reasoning_effort` field, for `openai/*` chat
    /// models. For other vendors use [`with_reasoning`](Options::with_reasoning).
    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Sets the unified `reasoning` request option, for reasoning-capable
    /// non-OpenAI models (e.g. `google/gemini-*`). The reasoning arrives in
    /// [`ChatResponse::reasoning`](super::ChatResponse::reasoning) and
    /// streams as [`Delta::reasoning`](super::Delta::reasoning).
    pub fn with_reasoning(mut self, reasoning: Reasoning) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Declares tools the model may call.
    ///
    /// When the model decides to call one, the calls are available via
    /// [`ChatResponse::tool_calls`](super::ChatResponse::tool_calls); answer
    /// them with [`ChatRequest::with_tool_results`](super::ChatRequest::with_tool_results).
    ///
    /// ```
    /// use llm_chain_lovable::chat::{Options, Tool};
    ///
    /// let options = Options::new().with_tools([Tool::function(
    ///     "get_weather",
    ///     "Get the current weather in a city",
    ///     serde_json::json!({
    ///         "type": "object",
    ///         "properties": {"city": {"type": "string"}},
    ///         "required": ["city"]
    ///     }),
    /// )]);
    /// ```
    pub fn with_tools(mut self, tools: impl IntoIterator<Item = Tool>) -> Self {
        self.tools = Some(tools.into_iter().collect());
        self
    }

    /// Applies every set option to a request.
    pub(crate) fn apply(&self, request: &mut ChatRequest) {
        request.temperature = self.temperature;
        request.top_p = self.top_p;
        request.max_tokens = self.max_tokens;
        request.max_completion_tokens = self.max_completion_tokens;
        request.stop = self.stop_sequences.clone();
        request.seed = self.seed;
        request.response_format = self.response_format.clone();
        request.reasoning_effort = self.reasoning_effort;
        request.reasoning = self.reasoning;
        request.tools = self.tools.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{Message, Role};
    use super::*;

    fn base_request() -> ChatRequest {
        ChatRequest::new(
            "google/gemini-3.6-flash",
            vec![Message::new(Role::User, "hi")],
        )
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
            .with_top_p(0.9)
            .with_max_tokens(256)
            .with_seed(42)
            .with_stop_sequences(["END"])
            .with_reasoning(Reasoning::effort(ReasoningEffort::High))
            .with_json_output()
            .apply(&mut request);
        assert_eq!(request.temperature, Some(0.3));
        assert_eq!(request.top_p, Some(0.9));
        assert_eq!(request.max_tokens, Some(256));
        assert_eq!(request.seed, Some(42));
        assert_eq!(request.stop, Some(vec!["END".to_string()]));
        assert_eq!(
            request.reasoning,
            Some(Reasoning::effort(ReasoningEffort::High))
        );
        assert_eq!(request.response_format, Some(ResponseFormat::JsonObject));
    }

    #[test]
    fn json_schema_defaults_to_strict() {
        let mut request = base_request();
        Options::new()
            .with_json_schema("answer", serde_json::json!({"type": "object"}))
            .apply(&mut request);
        match request.response_format {
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                assert_eq!(json_schema.name, "answer");
                assert_eq!(json_schema.strict, Some(true));
            }
            other => panic!("expected json_schema, got: {other:?}"),
        }
    }

    #[test]
    fn tools_are_applied_to_the_request() {
        let mut request = base_request();
        let tool = Tool::function(
            "get_weather",
            "Get the weather",
            serde_json::json!({"type": "object"}),
        );
        Options::new()
            .with_tools([tool.clone()])
            .apply(&mut request);
        assert_eq!(request.tools, Some(vec![tool]));
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
            .with_max_completion_tokens(2048)
            .with_reasoning_effort(ReasoningEffort::Low)
            .with_tools([Tool::function(
                "get_weather",
                "Get the weather",
                serde_json::json!({"type": "object"}),
            )]);
        let yaml = serde_yaml_ng::to_string(&options).unwrap();
        let parsed: Options = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed, options);
    }
}
