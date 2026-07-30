//! Wire types for Anthropic's Messages API.
//!
//! These types model the subset of the API that llm-chain uses: text-in,
//! text-out conversations, optionally with extended thinking and tool use.
//! They always derive `serde` traits because they exist to be (de)serialized
//! on the wire.

use serde::{Deserialize, Serialize};

/// The role of a message in a conversation.
///
/// Anthropic's Messages API only accepts `user` and `assistant` messages in the
/// conversation itself; system instructions are a top-level request field (see
/// [`ChatPromptTemplate::with_system`](super::ChatPromptTemplate::with_system)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// End-user input.
    User,
    /// Previous model output, e.g. few-shot examples.
    Assistant,
}

/// The content of a [`Message`]: plain text or a list of content blocks.
///
/// Plain text serializes as a JSON string, exactly like the shorthand the API
/// accepts. Blocks are needed for tool use: assistant turns carry
/// [`ContentBlock::ToolUse`] blocks and the following user turn carries
/// [`ContentBlock::ToolResult`] blocks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text.
    Text(String),
    /// A list of content blocks.
    Blocks(Vec<ContentBlock>),
}

impl MessageContent {
    /// The plain text of this content, when it is [`MessageContent::Text`].
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Blocks(_) => None,
        }
    }

    /// The blocks of this content; plain text has no blocks.
    pub fn blocks(&self) -> &[ContentBlock] {
        match self {
            Self::Text(_) => &[],
            Self::Blocks(blocks) => blocks,
        }
    }
}

impl From<String> for MessageContent {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for MessageContent {
    fn from(text: &str) -> Self {
        Self::Text(text.to_string())
    }
}

impl From<Vec<ContentBlock>> for MessageContent {
    fn from(blocks: Vec<ContentBlock>) -> Self {
        Self::Blocks(blocks)
    }
}

/// A single, fully formatted message in a Messages API request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Who authored the message.
    pub role: Role,
    /// The message content: plain text or content blocks.
    pub content: MessageContent,
}

impl Message {
    /// Creates a plain-text message with the given role.
    pub fn text<S: Into<String>>(role: Role, text: S) -> Self {
        Self {
            role,
            content: MessageContent::Text(text.into()),
        }
    }

    /// Creates the user message that answers tool calls: one
    /// [`ContentBlock::ToolResult`] block per invoked tool.
    ///
    /// The API requires tool results to be the next message after the
    /// assistant's tool-use turn, and every `tool_use_id` must be answered.
    pub fn tool_results<I: IntoIterator<Item = ToolResult>>(results: I) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Blocks(
                results.into_iter().map(ContentBlock::ToolResult).collect(),
            ),
        }
    }
}

/// Extended-thinking configuration.
///
/// Supported by Claude Haiku 4.5 and the 4.x generation. Claude 5-generation
/// models (Fable, Opus 5, Sonnet 5) use adaptive thinking instead — control
/// them with [`Effort`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Thinking {
    /// The model may think for up to `budget_tokens` tokens before answering.
    Enabled {
        /// Upper bound on thinking tokens; must be less than `max_tokens`.
        budget_tokens: u32,
    },
}

/// Reasoning effort for Claude 5-generation models (Opus 4.8 and later).
///
/// Higher effort spends more reasoning for harder problems; the API default is
/// [`Effort::High`] on the models that support it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    /// Fastest, cheapest responses.
    Low,
    /// A balance of speed and depth.
    Medium,
    /// Maximum reasoning depth (the API default where supported).
    High,
}

/// A tool the model may call, sent in the request's `tools` array.
///
/// `input_schema` is a [JSON Schema](https://json-schema.org/) object
/// describing the tool's arguments; the model generates
/// [`ToolUse::input`] values conforming to it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// The tool name the model calls it by.
    pub name: String,
    /// What the tool does and when to use it. Be thorough — this is the
    /// model's main signal for choosing tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the tool's input object.
    pub input_schema: serde_json::Value,
}

impl ToolDefinition {
    /// Creates a tool definition from a name, description and JSON Schema.
    pub fn new<N: Into<String>, D: Into<String>>(
        name: N,
        description: D,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            input_schema,
        }
    }
}

/// How the model chooses among the request's tools.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolChoice {
    /// The model decides whether to call a tool (the API default when tools are present).
    Auto,
    /// The model must call one of the provided tools.
    Any,
    /// The model must call the named tool.
    Tool {
        /// The name of the tool to call.
        name: String,
    },
    /// The model must not call any tool.
    None,
}

/// A fully formatted Messages API request, produced by
/// [`Step::format`](llm_chain::traits::Step::format) and consumed by the
/// [`Executor`](super::Executor).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MessagesRequest {
    /// The model id, e.g. `claude-sonnet-4-5`.
    pub model: String,
    /// The maximum number of tokens to generate (required by the API).
    pub max_tokens: u32,
    /// System instructions, set with [`Step::with_system`](super::Step::with_system).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// The conversation so far.
    pub messages: Vec<Message>,
    /// Sampling temperature, between 0.0 and 1.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability mass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Only sample from the `top_k` most likely tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Sequences at which the model stops generating.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Extended-thinking configuration (Haiku 4.5 and the 4.x generation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
    /// Reasoning effort (Claude 5 generation and Opus 4.8+).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,
    /// Tools the model may call, set with
    /// [`Options::with_tools`](super::Options::with_tools).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    /// How the model chooses among the tools, set with
    /// [`Options::with_tool_choice`](super::Options::with_tool_choice).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

impl MessagesRequest {
    /// Continues a tool-use conversation: appends the assistant turn from
    /// `response` (preserving every content block, including thinking
    /// signatures) followed by a user turn carrying the tool results.
    ///
    /// Execute the returned request to get the model's next turn — either the
    /// final answer or another round of tool calls.
    pub fn with_tool_results<I: IntoIterator<Item = ToolResult>>(
        mut self,
        response: &MessagesResponse,
        results: I,
    ) -> Self {
        self.messages.push(response.to_message());
        self.messages.push(Message::tool_results(results));
        self
    }
}

/// A tool call made by the model, carried in a [`ContentBlock::ToolUse`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolUse {
    /// The unique id of this call; echo it back as
    /// [`ToolResult::tool_use_id`].
    pub id: String,
    /// The name of the tool to invoke.
    pub name: String,
    /// The arguments, conforming to the tool's `input_schema`.
    #[serde(default)]
    pub input: serde_json::Value,
}

/// The result of running a tool, carried in a [`ContentBlock::ToolResult`]
/// inside the user message that answers a tool-use turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// The id of the [`ToolUse`] this result answers.
    pub tool_use_id: String,
    /// The result, serialized as text (JSON is fine).
    pub content: String,
    /// Set to `true` when the tool failed; the model sees the content as an
    /// error message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl ToolResult {
    /// Creates a successful tool result.
    pub fn new<I: Into<String>, C: Into<String>>(tool_use_id: I, content: C) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error: None,
        }
    }

    /// Creates a failed tool result; the model sees `message` as an error.
    pub fn error<I: Into<String>, M: Into<String>>(tool_use_id: I, message: M) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: message.into(),
            is_error: Some(true),
        }
    }
}

/// One block of generated content in a response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Generated text.
    Text {
        /// The text itself.
        text: String,
    },
    /// The model's (summarized) reasoning, present when extended thinking is enabled.
    Thinking {
        /// The thinking text.
        thinking: String,
        /// Integrity signature for passing the block back to the API.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// A tool call made by the model; run the tool and answer with a
    /// [`ContentBlock::ToolResult`] in the next user message.
    ToolUse(ToolUse),
    /// The result of running a tool, sent back in a user message.
    ToolResult(ToolResult),
    /// Any block type this crate does not model (e.g. server tool use).
    #[serde(other)]
    Other,
}

/// Why the model stopped generating.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The model finished its turn naturally.
    EndTurn,
    /// The `max_tokens` limit was reached.
    MaxTokens,
    /// One of the `stop_sequences` was generated.
    StopSequence,
    /// The model invoked a tool.
    ToolUse,
    /// The model refused to answer.
    Refusal,
    /// The model paused a long-running turn.
    PauseTurn,
    /// Any stop reason this crate does not model.
    #[serde(other)]
    Other,
}

/// Token accounting for a response.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens in the prompt.
    #[serde(default)]
    pub input_tokens: u32,
    /// Tokens in the generated answer (including thinking tokens).
    #[serde(default)]
    pub output_tokens: u32,
}

/// A Messages API response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MessagesResponse {
    /// The response id.
    pub id: String,
    /// The model that produced the response.
    pub model: String,
    /// The generated content blocks.
    pub content: Vec<ContentBlock>,
    /// Why the model stopped.
    #[serde(default)]
    pub stop_reason: Option<StopReason>,
    /// Which stop sequence fired, if any.
    #[serde(default)]
    pub stop_sequence: Option<String>,
    /// Token accounting.
    #[serde(default)]
    pub usage: Usage,
}

impl MessagesResponse {
    /// Concatenates every text block into a single string, skipping thinking
    /// and unknown blocks.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for block in &self.content {
            if let ContentBlock::Text { text } = block {
                out.push_str(text);
            }
        }
        out
    }

    /// The tool calls the model made this turn, in order.
    ///
    /// Non-empty exactly when [`MessagesResponse::stop_reason`] is
    /// [`StopReason::ToolUse`]. Run each tool and answer with
    /// [`MessagesRequest::with_tool_results`].
    pub fn tool_uses(&self) -> impl Iterator<Item = &ToolUse> {
        self.content.iter().filter_map(|block| match block {
            ContentBlock::ToolUse(tool_use) => Some(tool_use),
            _ => None,
        })
    }

    /// Converts the response into an assistant [`Message`], preserving every
    /// content block (including thinking signatures and tool-use blocks) so
    /// it can be passed back when continuing the conversation.
    pub fn to_message(&self) -> Message {
        Message {
            role: Role::Assistant,
            content: MessageContent::Blocks(self.content.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serializes_minimally() {
        let request = MessagesRequest {
            model: "claude-sonnet-5".to_string(),
            max_tokens: 1024,
            system: None,
            messages: vec![Message::text(Role::User, "hi")],
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            thinking: None,
            effort: None,
            tools: None,
            tool_choice: None,
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "model": "claude-sonnet-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "hi"}],
            })
        );
    }

    #[test]
    fn response_parses_and_extracts_text() {
        let json = r#"{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": ", world"},
                {"type": "server_tool_use", "id": "tu_1", "name": "web_search", "input": {}}
            ],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 10, "output_tokens": 25, "cache_read_input_tokens": 0}
        }"#;
        let response: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.text(), "Hello, world");
        assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 25);
        assert!(matches!(response.content[3], ContentBlock::Other));
    }

    #[test]
    fn unknown_stop_reasons_do_not_break_parsing() {
        let json = r#"{
            "id": "msg_1",
            "model": "m",
            "content": [],
            "stop_reason": "some_future_reason",
            "usage": {"input_tokens": 1, "output_tokens": 2}
        }"#;
        let response: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.stop_reason, Some(StopReason::Other));
    }

    #[test]
    fn tools_serialize_on_the_wire_format() {
        let mut request = MessagesRequest {
            model: "claude-sonnet-5".to_string(),
            max_tokens: 1024,
            system: None,
            messages: vec![Message::text(Role::User, "weather in Stockholm?")],
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            thinking: None,
            effort: None,
            tools: Some(vec![ToolDefinition::new(
                "get_weather",
                "Get the current weather for a city.",
                serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }),
            )]),
            tool_choice: Some(ToolChoice::Auto),
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["tools"][0]["name"], "get_weather");
        assert_eq!(
            json["tools"][0]["input_schema"]["properties"]["city"]["type"],
            "string"
        );
        assert_eq!(json["tool_choice"], serde_json::json!({"type": "auto"}));

        request.tool_choice = Some(ToolChoice::Tool {
            name: "get_weather".to_string(),
        });
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(
            json["tool_choice"],
            serde_json::json!({"type": "tool", "name": "get_weather"})
        );
    }

    #[test]
    fn tool_use_responses_parse_and_expose_calls() {
        let json = r#"{
            "id": "msg_1",
            "model": "claude-sonnet-5",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {"city": "Stockholm"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 30, "output_tokens": 12}
        }"#;
        let response: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.stop_reason, Some(StopReason::ToolUse));
        let calls: Vec<_> = response.tool_uses().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "toolu_abc");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input["city"], "Stockholm");
    }

    #[test]
    fn tool_results_continue_the_conversation() {
        let response: MessagesResponse = serde_json::from_str(
            r#"{
                "id": "msg_1",
                "model": "m",
                "content": [{"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {"city": "Stockholm"}}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 2}
            }"#,
        )
        .unwrap();
        let request = MessagesRequest {
            model: "m".to_string(),
            max_tokens: 1024,
            system: None,
            messages: vec![Message::text(Role::User, "weather?")],
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            thinking: None,
            effort: None,
            tools: None,
            tool_choice: None,
        }
        .with_tool_results(&response, [ToolResult::new("toolu_abc", "8°C, cloudy")]);

        assert_eq!(request.messages.len(), 3);
        assert_eq!(request.messages[1].role, Role::Assistant);
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(
            json["messages"][2],
            serde_json::json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_abc",
                    "content": "8°C, cloudy"
                }]
            })
        );
    }

    #[test]
    fn error_tool_results_carry_the_flag() {
        let result = ToolResult::error("toolu_1", "city not found");
        let json = serde_json::to_value(ContentBlock::ToolResult(result)).unwrap();
        assert_eq!(json["is_error"], true);
        assert_eq!(json["type"], "tool_result");
    }

    #[test]
    fn message_content_round_trips_text_and_blocks() {
        let text: MessageContent = "hi".into();
        assert_eq!(
            serde_json::to_value(&text).unwrap(),
            serde_json::json!("hi")
        );
        assert_eq!(text.as_text(), Some("hi"));
        assert!(text.blocks().is_empty());

        let blocks = MessageContent::Blocks(vec![ContentBlock::Text {
            text: "hi".to_string(),
        }]);
        let json = serde_json::to_value(&blocks).unwrap();
        assert_eq!(json[0]["type"], "text");
        let parsed: MessageContent = serde_json::from_value(json).unwrap();
        assert_eq!(parsed, blocks);
        assert_eq!(parsed.as_text(), None);
    }
}
