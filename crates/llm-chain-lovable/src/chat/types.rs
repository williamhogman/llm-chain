//! Wire types for the Lovable AI Gateway's OpenAI-compatible chat
//! completions API (`POST /v1/chat/completions`).
//!
//! These types model the subset of the surface that llm-chain uses:
//! text-in, text-out conversations, native tool calling, structured output
//! and reasoning controls. They always derive `serde` traits because they
//! exist to be (de)serialized on the wire.

use serde::{Deserialize, Serialize};

/// The role of a message in a conversation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Sets model behavior. The gateway normalizes this per vendor (e.g. to
    /// a developer message on OpenAI reasoning models).
    System,
    /// End-user input.
    User,
    /// Previous model output, e.g. few-shot examples or an echoed tool-call turn.
    Assistant,
    /// The result of running a tool, answering an assistant tool call.
    Tool,
}

/// The function invocation inside a [`ToolCall`].
///
/// Unlike some other APIs, the OpenAI-compatible wire format carries the
/// arguments as a JSON-*encoded string*; use
/// [`parsed_arguments`](FunctionCall::parsed_arguments) to decode them.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// The name of the tool to invoke.
    pub name: String,
    /// The arguments as a JSON-encoded string, conforming to the tool's
    /// parameters schema.
    #[serde(default)]
    pub arguments: String,
}

impl FunctionCall {
    /// Decodes [`arguments`](Self::arguments) into a JSON value.
    pub fn parsed_arguments(&self) -> Result<serde_json::Value, serde_json::Error> {
        serde_json::from_str(&self.arguments)
    }
}

/// A tool call made by the model, carried in [`Message::tool_calls`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// The call id, echoed back in the answering [`Role::Tool`] message.
    pub id: String,
    /// The call type; always `function`.
    #[serde(rename = "type", default = "function_type")]
    pub call_type: String,
    /// The function to invoke.
    pub function: FunctionCall,
}

pub(crate) fn function_type() -> String {
    "function".to_string()
}

/// A single, fully formatted message in a chat API request or response.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Who authored the message.
    pub role: Role,
    /// The message text. `None` on assistant messages that only carry tool calls.
    pub content: Option<String>,
    /// The model's reasoning, present on responses from reasoning-capable
    /// models when reasoning is requested (see
    /// [`Options::with_reasoning`](super::Options::with_reasoning)).
    #[serde(
        default,
        alias = "reasoning_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub reasoning: Option<String>,
    /// Tool calls made by the model, present in responses when it decides to
    /// call tools declared with [`Options::with_tools`](super::Options::with_tools).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// The id of the [`ToolCall`] this message answers; set on [`Role::Tool`] messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Creates a message with the given role and content.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            reasoning: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a [`Role::Tool`] message carrying a tool's result back to the
    /// model. `content` is usually JSON, but any string works.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            reasoning: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// The function declaration inside a [`Tool`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolFunction {
    /// The tool name the model calls it by.
    pub name: String,
    /// What the tool does and when to use it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// A [JSON Schema](https://json-schema.org/) object describing the tool's arguments.
    #[serde(default)]
    pub parameters: serde_json::Value,
    /// Whether the provider must enforce the schema exactly (strict mode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// A tool the model may call, declared with
/// [`Options::with_tools`](super::Options::with_tools).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tool {
    /// The tool type; always `function`.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function declaration.
    pub function: ToolFunction,
}

impl Tool {
    /// Creates a function tool from a name, description and JSON Schema.
    ///
    /// ```
    /// use llm_chain_lovable::chat::Tool;
    ///
    /// let tool = Tool::function(
    ///     "get_weather",
    ///     "Get the current weather in a city",
    ///     serde_json::json!({
    ///         "type": "object",
    ///         "properties": {"city": {"type": "string"}},
    ///         "required": ["city"]
    ///     }),
    /// );
    /// ```
    pub fn function<N: Into<String>, D: Into<String>>(
        name: N,
        description: D,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: name.into(),
                description: Some(description.into()),
                parameters,
                strict: None,
            },
        }
    }

    /// Requests strict schema enforcement for this tool.
    ///
    /// Strict mode requires a strict-compatible schema: every object sets
    /// `additionalProperties: false`, every property is required (optional
    /// inputs are typed nullable), and no defaults.
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.function.strict = Some(strict);
        self
    }
}

/// Structured-output control (`response_format`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Force the model to answer with valid JSON (schema not enforced).
    ///
    /// Also instruct the model to answer in JSON in the prompt — providers
    /// reject `json_object` requests whose messages never mention JSON.
    JsonObject,
    /// Constrain the answer to the given JSON schema.
    JsonSchema {
        /// The named schema definition.
        json_schema: JsonSchema,
    },
}

/// A named JSON schema for [`ResponseFormat::JsonSchema`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JsonSchema {
    /// A name for the schema (identifier-like).
    pub name: String,
    /// Whether the provider must enforce the schema exactly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    /// The [JSON Schema](https://json-schema.org/) object itself.
    pub schema: serde_json::Value,
}

/// Reasoning effort for reasoning-capable models.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Disable reasoning on models that support turning it off.
    None,
    /// Light reasoning.
    Low,
    /// Balanced reasoning (a good default).
    Medium,
    /// Thorough reasoning.
    High,
}

/// The unified reasoning request option, for reasoning-capable non-OpenAI
/// models (e.g. the `google/gemini-*` family).
///
/// The reasoning summary arrives in [`Message::reasoning`] (and streams as
/// [`Delta::reasoning`](super::Delta::reasoning)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reasoning {
    /// How much effort to spend reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    /// Reason internally but leave the reasoning out of the response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
}

impl Reasoning {
    /// Creates a reasoning request with the given effort.
    pub fn effort(effort: ReasoningEffort) -> Self {
        Self {
            effort: Some(effort),
            exclude: None,
        }
    }

    /// Sets whether the reasoning is excluded from the response.
    pub fn with_exclude(mut self, exclude: bool) -> Self {
        self.exclude = Some(exclude);
        self
    }
}

/// Streaming options, sent only on streaming requests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamOptions {
    /// Ask for a final chunk carrying token [`Usage`].
    pub include_usage: bool,
}

/// A fully formatted chat completions request, produced by
/// [`Step::format`](llm_chain::traits::Step::format) and consumed by the
/// [`Executor`](super::Executor).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// The vendor-prefixed model id from the Lovable catalog,
    /// e.g. `google/gemini-3.6-flash`.
    pub model: String,
    /// The conversation so far, including any system message.
    pub messages: Vec<Message>,
    /// Whether to stream the response. [`Executor::execute`](llm_chain::traits::Executor::execute)
    /// sends `false`;
    /// [`execute_stream`](llm_chain::traits::StreamingExecutor::execute_stream)
    /// sends `true`.
    pub stream: bool,
    /// Streaming options; set by the streaming path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// Sampling temperature. Higher is more random.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability mass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Upper bound on generated tokens (widely supported field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Upper bound on generated tokens, the field OpenAI reasoning models
    /// require instead of `max_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Sequences at which the model stops generating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Random seed for best-effort reproducible generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Structured-output control.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// OpenAI-style reasoning effort, for `openai/*` chat models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Unified reasoning option, for non-OpenAI reasoning models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    /// Tools the model may call, set with
    /// [`Options::with_tools`](super::Options::with_tools).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

impl ChatRequest {
    /// Creates a minimal request: model, messages, everything else at the
    /// provider defaults.
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            stream: false,
            stream_options: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_completion_tokens: None,
            stop: None,
            seed: None,
            response_format: None,
            reasoning_effort: None,
            reasoning: None,
            tools: None,
        }
    }

    /// Extends this request to continue a tool-calling conversation: appends
    /// the assistant's message from `response` (echoing its `tool_calls`
    /// verbatim) followed by one [`Role::Tool`] message per result.
    ///
    /// `results` pairs each [`ToolCall::id`] with its output, in the same
    /// order as [`ChatResponse::tool_calls`]. Content is usually JSON, but
    /// any string works.
    pub fn with_tool_results<I, C>(
        mut self,
        response: &ChatResponse,
        results: impl IntoIterator<Item = (I, C)>,
    ) -> Self
    where
        I: Into<String>,
        C: Into<String>,
    {
        if let Some(message) = response.message() {
            self.messages.push(message.clone());
        }
        self.messages.extend(
            results
                .into_iter()
                .map(|(tool_call_id, content)| Message::tool_result(tool_call_id, content)),
        );
        self
    }
}

/// Why the model stopped generating.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model finished its turn naturally (or hit a stop sequence).
    Stop,
    /// The token limit was reached.
    Length,
    /// The model stopped to call tools; run them and continue with
    /// [`ChatRequest::with_tool_results`].
    ToolCalls,
    /// The provider filtered the content.
    ContentFilter,
    /// Any finish reason this crate does not model.
    #[serde(other)]
    Other,
}

/// Token usage for a request, reported by the gateway.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens in the prompt.
    #[serde(default)]
    pub prompt_tokens: u64,
    /// Tokens in the generated answer (reasoning included).
    #[serde(default)]
    pub completion_tokens: u64,
    /// Prompt plus completion tokens.
    #[serde(default)]
    pub total_tokens: u64,
}

/// One generated completion inside a [`ChatResponse`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    /// The choice index; `0` unless multiple completions were requested.
    #[serde(default)]
    pub index: u32,
    /// The generated message.
    pub message: Message,
    /// Why the model stopped.
    #[serde(default)]
    pub finish_reason: Option<FinishReason>,
}

/// A chat completions response (`object: "chat.completion"`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The response id.
    #[serde(default)]
    pub id: String,
    /// The object type; `chat.completion`.
    #[serde(default)]
    pub object: String,
    /// Unix timestamp of creation.
    #[serde(default)]
    pub created: u64,
    /// The model that produced the response.
    #[serde(default)]
    pub model: String,
    /// The generated completions; one unless multiple were requested.
    pub choices: Vec<Choice>,
    /// Token usage for the request.
    #[serde(default)]
    pub usage: Option<Usage>,
    /// The run id from the gateway's `X-Lovable-AIG-Run-ID` response header,
    /// correlating this call with Lovable AI usage logs. Set by the
    /// [`Executor`](super::Executor), not part of the JSON body.
    #[serde(skip)]
    pub run_id: Option<String>,
}

impl ChatResponse {
    /// The first choice's message, if any.
    pub fn message(&self) -> Option<&Message> {
        self.choices.first().map(|choice| &choice.message)
    }

    /// The generated answer text.
    pub fn text(&self) -> String {
        self.message()
            .and_then(|message| message.content.clone())
            .unwrap_or_default()
    }

    /// The model's reasoning, when reasoning was requested.
    pub fn reasoning(&self) -> Option<&str> {
        self.message()
            .and_then(|message| message.reasoning.as_deref())
    }

    /// The tool calls the model made, in order; empty when it answered
    /// directly. Run each tool and continue with
    /// [`ChatRequest::with_tool_results`].
    pub fn tool_calls(&self) -> &[ToolCall] {
        self.message()
            .and_then(|message| message.tool_calls.as_deref())
            .unwrap_or_default()
    }

    /// Why the first choice stopped, if reported.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.choices.first().and_then(|choice| choice.finish_reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_request() -> ChatRequest {
        ChatRequest::new(
            "google/gemini-3.6-flash",
            vec![Message::new(Role::User, "hi")],
        )
    }

    #[test]
    fn request_serializes_minimally() {
        let json = serde_json::to_value(minimal_request()).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "model": "google/gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false,
            })
        );
    }

    #[test]
    fn tools_serialize_in_openai_style() {
        let request = ChatRequest {
            tools: Some(vec![
                Tool::function(
                    "get_weather",
                    "Get the current weather",
                    serde_json::json!({
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }),
                )
                .with_strict(true),
            ]),
            ..minimal_request()
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["tools"][0]["type"], "function");
        assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(json["tools"][0]["function"]["strict"], true);
        assert_eq!(
            json["tools"][0]["function"]["parameters"]["required"][0],
            "city"
        );
    }

    #[test]
    fn response_format_serializes_tagged() {
        assert_eq!(
            serde_json::to_value(ResponseFormat::JsonObject).unwrap(),
            serde_json::json!({"type": "json_object"})
        );
        let schema = ResponseFormat::JsonSchema {
            json_schema: JsonSchema {
                name: "answer".to_string(),
                strict: Some(true),
                schema: serde_json::json!({"type": "object"}),
            },
        };
        assert_eq!(
            serde_json::to_value(&schema).unwrap(),
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": true, "schema": {"type": "object"}}
            })
        );
    }

    #[test]
    fn reasoning_serializes_in_both_shapes() {
        // OpenAI-style flat effort.
        assert_eq!(
            serde_json::to_value(ReasoningEffort::Medium).unwrap(),
            serde_json::json!("medium")
        );
        // Unified reasoning object for non-OpenAI models.
        assert_eq!(
            serde_json::to_value(Reasoning::effort(ReasoningEffort::Low).with_exclude(true))
                .unwrap(),
            serde_json::json!({"effort": "low", "exclude": true})
        );
    }

    #[test]
    fn tool_calls_parse_and_are_extracted() {
        let json = r#"{
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "google/gemini-3.6-flash",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\": \"Stockholm\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }"#;
        let response: ChatResponse = serde_json::from_str(json).unwrap();
        let calls = response.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            calls[0].function.parsed_arguments().unwrap()["city"],
            "Stockholm"
        );
        assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
        assert_eq!(response.text(), "");
    }

    #[test]
    fn with_tool_results_extends_the_conversation() {
        let response: ChatResponse = serde_json::from_str(
            r#"{
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{\"city\": \"Stockholm\"}"}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }"#,
        )
        .unwrap();

        let request = minimal_request()
            .with_tool_results(&response, [("call_1", r#"{"temperature_c": -3}"#)]);

        assert_eq!(request.messages.len(), 3);
        let json = serde_json::to_value(&request).unwrap();
        // The assistant's tool_calls are echoed back verbatim.
        assert_eq!(json["messages"][1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(
            json["messages"][2],
            serde_json::json!({
                "role": "tool",
                "content": "{\"temperature_c\": -3}",
                "tool_call_id": "call_1",
            })
        );
    }

    #[test]
    fn response_parses_and_extracts_text() {
        let json = r#"{
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "created": 1785000000,
            "model": "google/gemini-3.6-flash",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello, world", "reasoning": "hmm"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 40, "total_tokens": 52}
        }"#;
        let response: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.text(), "Hello, world");
        assert_eq!(response.reasoning(), Some("hmm"));
        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
        assert_eq!(response.usage.unwrap().total_tokens, 52);
        assert_eq!(response.run_id, None);
    }

    #[test]
    fn reasoning_content_alias_is_accepted() {
        let json = r#"{
            "choices": [{
                "message": {"role": "assistant", "content": "hi", "reasoning_content": "thought"}
            }]
        }"#;
        let response: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.reasoning(), Some("thought"));
    }

    #[test]
    fn unknown_finish_reasons_do_not_break_parsing() {
        let json = r#"{
            "choices": [{
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "some_future_reason"
            }]
        }"#;
        let response: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.finish_reason(), Some(FinishReason::Other));
    }
}
