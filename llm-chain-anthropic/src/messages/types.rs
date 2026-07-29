//! Wire types for Anthropic's Messages API.
//!
//! These types model the subset of the API that llm-chain uses: text-in,
//! text-out conversations, optionally with extended thinking. They always
//! derive `serde` traits because they exist to be (de)serialized on the wire.

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

/// A single, fully formatted message in a Messages API request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Who authored the message.
    pub role: Role,
    /// The message text.
    pub content: String,
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
    /// Any block type this crate does not model (e.g. tool use).
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
}
