//! Wire types for Amazon Bedrock's Converse API.
//!
//! These types model the subset of the API that llm-chain uses: text-in,
//! text-out conversations, optionally with reasoning. They always derive
//! `serde` traits because they exist to be (de)serialized on the wire.

use serde::{Deserialize, Serialize};

/// The role of a message in a conversation.
///
/// The Converse API only accepts `user` and `assistant` messages in the
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

/// The model's reasoning text, nested inside [`ReasoningContent`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReasoningText {
    /// The reasoning itself.
    #[serde(default)]
    pub text: String,
    /// Integrity signature for passing the block back to the API.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// A reasoning content block, returned when reasoning is enabled for models
/// that support it (via
/// [`Options::with_additional_model_request_fields`](super::Options::with_additional_model_request_fields)).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningContent {
    /// The reasoning text, absent when the block is redacted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_text: Option<ReasoningText>,
}

/// One block of content in a message.
///
/// The Converse API models content as a union with exactly one member set per
/// block; on the wire each block is an object with a single key.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContentBlock {
    /// Generated or provided text.
    Text {
        /// The text itself.
        text: String,
    },
    /// The model's reasoning, present when reasoning is enabled.
    Reasoning {
        /// The reasoning content.
        #[serde(rename = "reasoningContent")]
        reasoning_content: ReasoningContent,
    },
    /// Any block type this crate does not model (images, documents, tool use, …).
    Other(serde_json::Value),
}

/// A single, fully formatted message in a Converse API request or response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Who authored the message.
    pub role: Role,
    /// The content blocks making up the message.
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Creates a single-block text message with the given role.
    pub fn text<S: Into<String>>(role: Role, text: S) -> Self {
        Self {
            role,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// Concatenates every text block into a single string, skipping reasoning
    /// and unknown blocks.
    pub fn text_blocks(&self) -> String {
        let mut out = String::new();
        for block in &self.content {
            if let ContentBlock::Text { text } = block {
                out.push_str(text);
            }
        }
        out
    }
}

/// One block of system instructions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SystemContentBlock {
    /// The instruction text.
    pub text: String,
}

/// Base inference parameters, shared across every model family.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceConfig {
    /// Upper bound on generated tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability mass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Sequences at which the model stops generating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

impl InferenceConfig {
    /// Returns true when no field is set.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// A fully formatted Converse API request, produced by
/// [`Step::format`](llm_chain::traits::Step::format) and consumed by the
/// [`Executor`](super::Executor).
///
/// The Converse API takes the model id in the URL path rather than the request
/// body, so `model_id` is carried here for the executor but never serialized.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConverseRequest {
    /// The model or inference-profile id, e.g. `global.anthropic.claude-sonnet-5-20260203-v1:0`.
    /// Sent in the URL path, not the body.
    #[serde(skip)]
    pub model_id: String,
    /// The conversation so far.
    pub messages: Vec<Message>,
    /// System instructions, set with [`Step::with_system`](super::Step::with_system).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<SystemContentBlock>>,
    /// Base inference parameters, shared across model families.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_config: Option<InferenceConfig>,
    /// Model-family-specific parameters the base config does not cover,
    /// passed through verbatim (e.g. Claude's `thinking`, `top_k`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_model_request_fields: Option<serde_json::Value>,
}

/// Why the model stopped generating.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The model finished its turn naturally.
    EndTurn,
    /// The model invoked a tool.
    ToolUse,
    /// The `maxTokens` limit was reached.
    MaxTokens,
    /// One of the `stopSequences` was generated.
    StopSequence,
    /// A configured guardrail intervened.
    GuardrailIntervened,
    /// The content was filtered by the model provider.
    ContentFiltered,
    /// Any stop reason this crate does not model.
    #[serde(other)]
    Other,
}

/// Token accounting for a response.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenUsage {
    /// Tokens in the prompt.
    #[serde(default)]
    pub input_tokens: u32,
    /// Tokens in the generated answer (including reasoning tokens).
    #[serde(default)]
    pub output_tokens: u32,
    /// Total tokens for the request.
    #[serde(default)]
    pub total_tokens: u32,
    /// Prompt tokens served from the prompt cache.
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    /// Prompt tokens written to the prompt cache.
    #[serde(default)]
    pub cache_write_input_tokens: u32,
}

/// Latency metrics for a response.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Metrics {
    /// Wall-clock request latency in milliseconds, as measured by Bedrock.
    #[serde(default)]
    pub latency_ms: u64,
}

/// The result union of a Converse response; in practice the generated message.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConverseOutput {
    /// The generated message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,
}

/// A Converse API response.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConverseResponse {
    /// The generated output.
    #[serde(default)]
    pub output: Option<ConverseOutput>,
    /// Why the model stopped.
    #[serde(default)]
    pub stop_reason: Option<StopReason>,
    /// Token accounting.
    #[serde(default)]
    pub usage: TokenUsage,
    /// Latency metrics.
    #[serde(default)]
    pub metrics: Option<Metrics>,
}

impl ConverseResponse {
    /// Concatenates every text block of the output message into a single
    /// string, skipping reasoning and unknown blocks.
    pub fn text(&self) -> String {
        self.output
            .as_ref()
            .and_then(|output| output.message.as_ref())
            .map(Message::text_blocks)
            .unwrap_or_default()
    }

    /// Concatenates every reasoning block of the output message, or `None`
    /// when the model did not reason (or reasoning was redacted).
    pub fn reasoning(&self) -> Option<String> {
        let message = self.output.as_ref()?.message.as_ref()?;
        let mut out = String::new();
        for block in &message.content {
            if let ContentBlock::Reasoning { reasoning_content } = block
                && let Some(reasoning_text) = &reasoning_content.reasoning_text
            {
                out.push_str(&reasoning_text.text);
            }
        }
        (!out.is_empty()).then_some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serializes_minimally() {
        let request = ConverseRequest {
            model_id: "global.anthropic.claude-sonnet-5-20260203-v1:0".to_string(),
            messages: vec![Message::text(Role::User, "hi")],
            system: None,
            inference_config: None,
            additional_model_request_fields: None,
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            })
        );
    }

    #[test]
    fn system_and_inference_config_serialize_in_camel_case() {
        let request = ConverseRequest {
            model_id: "amazon.nova-pro-v1:0".to_string(),
            messages: vec![Message::text(Role::User, "hi")],
            system: Some(vec![SystemContentBlock {
                text: "be brief".to_string(),
            }]),
            inference_config: Some(InferenceConfig {
                max_tokens: Some(512),
                temperature: Some(0.5),
                top_p: None,
                stop_sequences: Some(vec!["END".to_string()]),
            }),
            additional_model_request_fields: None,
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["system"], serde_json::json!([{"text": "be brief"}]));
        assert_eq!(json["inferenceConfig"]["maxTokens"], 512);
        assert_eq!(json["inferenceConfig"]["stopSequences"][0], "END");
        assert!(json["inferenceConfig"].get("topP").is_none());
    }

    #[test]
    fn response_parses_and_extracts_text() {
        let json = r#"{
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"reasoningContent": {"reasoningText": {"text": "hmm", "signature": "sig"}}},
                        {"text": "Hello"},
                        {"text": ", world"},
                        {"toolUse": {"toolUseId": "t1", "name": "search", "input": {}}}
                    ]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 25, "totalTokens": 35},
            "metrics": {"latencyMs": 551}
        }"#;
        let response: ConverseResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.text(), "Hello, world");
        assert_eq!(response.reasoning().as_deref(), Some("hmm"));
        assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 25);
        assert_eq!(response.metrics.unwrap().latency_ms, 551);
        let message = response.output.unwrap().message.unwrap();
        assert!(matches!(message.content[3], ContentBlock::Other(_)));
    }

    #[test]
    fn unknown_stop_reasons_do_not_break_parsing() {
        let json = r#"{
            "output": {"message": {"role": "assistant", "content": [{"text": "x"}]}},
            "stopReason": "some_future_reason",
            "usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}
        }"#;
        let response: ConverseResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.stop_reason, Some(StopReason::Other));
    }

    #[test]
    fn cache_token_counts_parse_when_present() {
        let json = r#"{
            "output": {"message": {"role": "assistant", "content": [{"text": "x"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 5, "outputTokens": 2, "totalTokens": 7,
                "cacheReadInputTokens": 3, "cacheWriteInputTokens": 1
            }
        }"#;
        let response: ConverseResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.usage.cache_read_input_tokens, 3);
        assert_eq!(response.usage.cache_write_input_tokens, 1);
    }
}
