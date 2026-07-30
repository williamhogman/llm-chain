//! Wire types for Ollama's chat API.
//!
//! These types model the subset of the API that llm-chain uses: non-streaming,
//! text-in, text-out conversations, optionally with thinking. They always
//! derive `serde` traits because they exist to be (de)serialized on the wire.

use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize};

/// The role of a message in a conversation.
///
/// Ollama accepts system instructions inline in the message list, so unlike
/// some hosted APIs there is no separate top-level system field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Sets model behavior.
    System,
    /// End-user input.
    User,
    /// Previous model output, e.g. few-shot examples.
    Assistant,
    /// The result of running a tool, answering an assistant tool call.
    Tool,
}

/// The function invocation inside a [`ToolCall`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// The name of the tool to invoke.
    pub name: String,
    /// The arguments as a JSON object, conforming to the tool's parameters schema.
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// A tool call made by the model, carried in [`Message::tool_calls`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// The function to invoke.
    pub function: FunctionCall,
}

/// A single, fully formatted message in a chat API request or response.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Who authored the message.
    pub role: Role,
    /// The message text.
    pub content: String,
    /// The model's thinking, present in responses when thinking is enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// Tool calls made by the model, present in responses when it decides to
    /// call tools declared with [`Options::with_tools`](super::Options::with_tools).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// The tool that produced this message; set on [`Role::Tool`] messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

impl Message {
    /// Creates a message with the given role and content, without thinking.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            thinking: None,
            tool_calls: None,
            tool_name: None,
        }
    }

    /// Creates a [`Role::Tool`] message carrying a tool's result back to the
    /// model. `content` is usually JSON, but any string works.
    pub fn tool_result(tool_name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            thinking: None,
            tool_calls: None,
            tool_name: Some(tool_name.into()),
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
    /// use llm_chain_ollama::chat::Tool;
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
            },
        }
    }
}

/// Thinking control for reasoning-capable models.
///
/// Serializes the way Ollama expects: `false`/`true` for [`Think::Disabled`]
/// and [`Think::Enabled`], and the strings `"low"`/`"medium"`/`"high"` for the
/// leveled variants (supported by gpt-oss-style models).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Think {
    /// Turn thinking off, even on models that think by default (e.g. qwen3, deepseek-r1).
    Disabled,
    /// Turn thinking on; the reasoning arrives separately in [`Message::thinking`].
    Enabled,
    /// Low thinking effort (models with thinking levels, e.g. gpt-oss).
    Low,
    /// Medium thinking effort (models with thinking levels, e.g. gpt-oss).
    Medium,
    /// High thinking effort (models with thinking levels, e.g. gpt-oss).
    High,
}

impl Serialize for Think {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Disabled => serializer.serialize_bool(false),
            Self::Enabled => serializer.serialize_bool(true),
            Self::Low => serializer.serialize_str("low"),
            Self::Medium => serializer.serialize_str("medium"),
            Self::High => serializer.serialize_str("high"),
        }
    }
}

impl<'de> Deserialize<'de> for Think {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ThinkVisitor;
        impl Visitor<'_> for ThinkVisitor {
            type Value = Think;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a boolean or one of \"low\", \"medium\", \"high\"")
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Think, E> {
                Ok(if v { Think::Enabled } else { Think::Disabled })
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Think, E> {
                match v {
                    "low" => Ok(Think::Low),
                    "medium" => Ok(Think::Medium),
                    "high" => Ok(Think::High),
                    other => Err(E::unknown_variant(other, &["low", "medium", "high"])),
                }
            }
        }
        deserializer.deserialize_any(ThinkVisitor)
    }
}

/// Structured-output control.
///
/// Serializes as the string `"json"` for [`Format::Json`] and as the schema
/// object itself for [`Format::Schema`].
#[derive(Clone, Debug, PartialEq)]
pub enum Format {
    /// Force the model to answer with valid JSON.
    Json,
    /// Constrain the answer to the given JSON schema.
    Schema(serde_json::Value),
}

impl Serialize for Format {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Json => serializer.serialize_str("json"),
            Self::Schema(schema) => schema.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Format {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::String(s) if s == "json" => Ok(Self::Json),
            serde_json::Value::String(other) => Err(de::Error::custom(format!(
                "unknown format string {other:?}, expected \"json\" or a JSON schema object"
            ))),
            schema @ serde_json::Value::Object(_) => Ok(Self::Schema(schema)),
            _ => Err(de::Error::custom(
                "expected \"json\" or a JSON schema object",
            )),
        }
    }
}

/// Model-level options, nested under `options` in a request.
///
/// These map to Ollama's Modelfile runtime parameters; anything left `None`
/// falls back to the model's own defaults.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelOptions {
    /// Sampling temperature. Higher is more random.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability mass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Only sample from the `top_k` most likely tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Minimum token probability relative to the most likely token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    /// Maximum number of tokens to generate. `-1` means unlimited.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    /// The context window size in tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    /// Penalty for repeating tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    /// Random seed for reproducible generation (with temperature 0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Sequences at which the model stops generating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

impl ModelOptions {
    /// Returns true when every option is unset, i.e. the model defaults apply.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// A fully formatted chat API request, produced by
/// [`Step::format`](llm_chain::traits::Step::format) and consumed by the
/// [`Executor`](super::Executor).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// The model name, e.g. `qwen3` or `gpt-oss:120b-cloud`.
    pub model: String,
    /// The conversation so far, including any system message.
    pub messages: Vec<Message>,
    /// Whether to stream the response. This crate always sends `false`.
    pub stream: bool,
    /// Thinking control for reasoning-capable models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub think: Option<Think>,
    /// Structured-output control (JSON mode or a JSON schema).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<Format>,
    /// How long to keep the model loaded after the request, e.g. `"5m"` or `"0"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    /// Tools the model may call, set with
    /// [`Options::with_tools`](super::Options::with_tools).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Model-level options; omitted from the wire when empty.
    #[serde(default, skip_serializing_if = "ModelOptions::is_empty")]
    pub options: ModelOptions,
}

impl ChatRequest {
    /// Extends this request to continue a tool-calling conversation: appends
    /// the assistant's message from `response` (echoing its `tool_calls`
    /// verbatim) followed by one [`Role::Tool`] message per result.
    ///
    /// `results` pairs each tool name with its output, in the same order as
    /// [`ChatResponse::tool_calls`]. Content is usually JSON, but any string
    /// works.
    pub fn with_tool_results<N, C>(
        mut self,
        response: &ChatResponse,
        results: impl IntoIterator<Item = (N, C)>,
    ) -> Self
    where
        N: Into<String>,
        C: Into<String>,
    {
        self.messages.push(response.message.clone());
        self.messages.extend(
            results
                .into_iter()
                .map(|(tool_name, content)| Message::tool_result(tool_name, content)),
        );
        self
    }
}

/// Why the model stopped generating.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DoneReason {
    /// The model finished its turn naturally (or hit a stop sequence).
    Stop,
    /// The `num_predict` limit or context window was reached.
    Length,
    /// The request only loaded the model, no generation was requested.
    Load,
    /// The request unloaded the model.
    Unload,
    /// Any done reason this crate does not model.
    #[serde(other)]
    Other,
}

/// A chat API response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The model that produced the response.
    pub model: String,
    /// When the response was created (RFC 3339).
    #[serde(default)]
    pub created_at: String,
    /// The generated message.
    pub message: Message,
    /// Whether generation finished. Always true for non-streaming requests.
    pub done: bool,
    /// Why the model stopped.
    #[serde(default)]
    pub done_reason: Option<DoneReason>,
    /// Total wall-clock time in nanoseconds.
    #[serde(default)]
    pub total_duration: Option<u64>,
    /// Time spent loading the model in nanoseconds.
    #[serde(default)]
    pub load_duration: Option<u64>,
    /// Tokens in the prompt.
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    /// Time spent evaluating the prompt in nanoseconds.
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    /// Tokens in the generated answer.
    #[serde(default)]
    pub eval_count: Option<u64>,
    /// Time spent generating in nanoseconds.
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

impl ChatResponse {
    /// The generated answer text.
    pub fn text(&self) -> String {
        self.message.content.clone()
    }

    /// The model's thinking, when thinking was enabled.
    pub fn thinking(&self) -> Option<&str> {
        self.message.thinking.as_deref()
    }

    /// Generation speed in tokens per second, when the server reported timings.
    pub fn eval_rate(&self) -> Option<f64> {
        match (self.eval_count, self.eval_duration) {
            (Some(count), Some(duration)) if duration > 0 => {
                Some(count as f64 / (duration as f64 / 1e9))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serializes_minimally() {
        let request = ChatRequest {
            model: "qwen3".to_string(),
            messages: vec![Message::new(Role::User, "hi")],
            stream: false,
            think: None,
            format: None,
            keep_alive: None,
            options: ModelOptions::default(),
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false,
            })
        );
    }

    #[test]
    fn think_serializes_as_bool_or_level() {
        assert_eq!(
            serde_json::to_value(Think::Disabled).unwrap(),
            serde_json::json!(false)
        );
        assert_eq!(
            serde_json::to_value(Think::Enabled).unwrap(),
            serde_json::json!(true)
        );
        assert_eq!(
            serde_json::to_value(Think::High).unwrap(),
            serde_json::json!("high")
        );
        for think in [
            Think::Disabled,
            Think::Enabled,
            Think::Low,
            Think::Medium,
            Think::High,
        ] {
            let json = serde_json::to_value(think).unwrap();
            let parsed: Think = serde_json::from_value(json).unwrap();
            assert_eq!(parsed, think);
        }
    }

    #[test]
    fn format_serializes_as_json_or_schema() {
        assert_eq!(
            serde_json::to_value(&Format::Json).unwrap(),
            serde_json::json!("json")
        );
        let schema =
            serde_json::json!({"type": "object", "properties": {"age": {"type": "integer"}}});
        assert_eq!(
            serde_json::to_value(Format::Schema(schema.clone())).unwrap(),
            schema
        );
        let parsed: Format = serde_json::from_value(schema.clone()).unwrap();
        assert_eq!(parsed, Format::Schema(schema));
        let parsed: Format = serde_json::from_value(serde_json::json!("json")).unwrap();
        assert_eq!(parsed, Format::Json);
    }

    #[test]
    fn response_parses_and_extracts_text() {
        let json = r#"{
            "model": "qwen3",
            "created_at": "2026-07-30T08:00:00.000000Z",
            "message": {"role": "assistant", "content": "Hello, world", "thinking": "hmm"},
            "done": true,
            "done_reason": "stop",
            "total_duration": 5000000000,
            "load_duration": 1000000,
            "prompt_eval_count": 12,
            "prompt_eval_duration": 200000000,
            "eval_count": 40,
            "eval_duration": 2000000000
        }"#;
        let response: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.text(), "Hello, world");
        assert_eq!(response.thinking(), Some("hmm"));
        assert_eq!(response.done_reason, Some(DoneReason::Stop));
        assert_eq!(response.eval_count, Some(40));
        assert_eq!(response.eval_rate(), Some(20.0));
    }

    #[test]
    fn minimal_responses_parse_without_timings() {
        let json = r#"{
            "model": "m",
            "message": {"role": "assistant", "content": "hi"},
            "done": true
        }"#;
        let response: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.text(), "hi");
        assert_eq!(response.thinking(), None);
        assert_eq!(response.done_reason, None);
        assert_eq!(response.eval_rate(), None);
    }

    #[test]
    fn unknown_done_reasons_do_not_break_parsing() {
        let json = r#"{
            "model": "m",
            "message": {"role": "assistant", "content": ""},
            "done": true,
            "done_reason": "some_future_reason"
        }"#;
        let response: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.done_reason, Some(DoneReason::Other));
    }
}
