//! Wire types for the Gemini API's `generateContent` method.
//!
//! These types model the subset of the API that llm-chain uses: text-in,
//! text-out conversations, optionally with thinking. They always derive
//! `serde` traits because they exist to be (de)serialized on the wire.

use serde::{Deserialize, Serialize};

/// The role of a content entry in a conversation.
///
/// The Gemini API only accepts `user` and `model` turns in the conversation
/// itself; system instructions are a top-level request field (see
/// [`ChatPromptTemplate::with_system`](super::ChatPromptTemplate::with_system)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// End-user input.
    User,
    /// Previous model output, e.g. few-shot examples.
    Model,
}

/// One part of a content entry. This crate models text parts; anything else
/// (inline data, function calls, …) deserializes with `text` empty and is
/// skipped by [`Content::text`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Part {
    /// The text of this part.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text: String,
    /// True when this part is model reasoning rather than the answer
    /// (returned when thoughts are requested).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub thought: bool,
}

/// A content entry: a role plus a list of parts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Content {
    /// Who authored the entry. Absent on system instructions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    /// The parts making up the entry.
    #[serde(default)]
    pub parts: Vec<Part>,
}

impl Content {
    /// Creates a single-part text entry with the given role.
    pub fn text<S: Into<String>>(role: Role, text: S) -> Self {
        Self {
            role: Some(role),
            parts: vec![Part {
                text: text.into(),
                thought: false,
            }],
        }
    }

    /// Creates system instructions (a role-less single-part text entry).
    pub fn system<S: Into<String>>(text: S) -> Self {
        Self {
            role: None,
            parts: vec![Part {
                text: text.into(),
                thought: false,
            }],
        }
    }

    /// Concatenates every non-thought text part into a single string.
    pub fn text_parts(&self) -> String {
        let mut out = String::new();
        for part in &self.parts {
            if !part.thought {
                out.push_str(&part.text);
            }
        }
        out
    }
}

/// Thinking configuration, nested inside [`GenerationConfig`].
///
/// Gemini 3-generation models are controlled with [`ThinkingLevel`]; the 2.5
/// generation uses a token budget instead.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    /// Thinking depth for Gemini 3-generation models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<ThinkingLevel>,
    /// Thinking token budget for Gemini 2.5-generation models.
    /// `0` disables thinking (where supported) and `-1` enables dynamic thinking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<i32>,
    /// Ask the API to return thought summaries as parts with
    /// [`Part::thought`] set to `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_thoughts: Option<bool>,
}

/// Thinking depth for Gemini 3-generation models.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    /// The least thinking, for the lowest latency and cost. Not supported by
    /// every model (e.g. Gemini 3.1 Pro cannot go below [`ThinkingLevel::Low`]).
    Minimal,
    /// Little thinking, for latency-sensitive work.
    Low,
    /// A balance of depth and latency.
    Medium,
    /// Maximum reasoning depth (the API default on Gemini 3 Pro-class models).
    High,
}

/// Sampling and output controls, sent as `generationConfig`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability mass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Only sample from the `top_k` most likely tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Upper bound on generated tokens, including thinking tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Sequences at which the model stops generating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// The MIME type of the response, e.g. `application/json` for JSON output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    /// Thinking controls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
}

impl GenerationConfig {
    /// Returns true when no field is set.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// A fully formatted `generateContent` request, produced by
/// [`Step::format`](llm_chain::traits::Step::format) and consumed by the
/// [`Executor`](super::Executor).
///
/// The Gemini API takes the model id in the URL path rather than the request
/// body, so `model` is carried here for the executor but never serialized.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    /// The model id, e.g. `gemini-3.6-flash`. Sent in the URL path, not the body.
    #[serde(skip)]
    pub model: String,
    /// The conversation so far.
    pub contents: Vec<Content>,
    /// System instructions, set with [`Step::with_system`](super::Step::with_system).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    /// Sampling and output controls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
}

/// Why the model stopped generating.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
    /// The model finished its turn naturally or hit a stop sequence.
    Stop,
    /// The `maxOutputTokens` limit was reached.
    MaxTokens,
    /// The response was flagged by safety filters.
    Safety,
    /// The response was flagged for reciting training data.
    Recitation,
    /// The response used an unsupported language.
    Language,
    /// The response contained a term from a configured blocklist.
    Blocklist,
    /// The response was flagged as prohibited content.
    ProhibitedContent,
    /// The response was flagged for sensitive personally identifiable information.
    Spii,
    /// The model produced an invalid function call.
    MalformedFunctionCall,
    /// Any finish reason this crate does not model.
    #[serde(other)]
    Other,
}

/// Token accounting for a response.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    /// Tokens in the prompt.
    #[serde(default)]
    pub prompt_token_count: u32,
    /// Tokens in the generated answer.
    #[serde(default)]
    pub candidates_token_count: u32,
    /// Tokens spent thinking.
    #[serde(default)]
    pub thoughts_token_count: u32,
    /// Prompt tokens served from the context cache.
    #[serde(default)]
    pub cached_content_token_count: u32,
    /// Total tokens for the request.
    #[serde(default)]
    pub total_token_count: u32,
}

/// One generated candidate.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    /// The generated content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    /// Why the model stopped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

/// Feedback about the prompt, present when the prompt itself was blocked.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    /// Why the prompt was blocked, e.g. `SAFETY`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_reason: Option<String>,
}

/// A `generateContent` response.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    /// The generated candidates. In practice at most one unless requested otherwise.
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    /// Present when the prompt was blocked and no candidates were generated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_feedback: Option<PromptFeedback>,
    /// Token accounting.
    #[serde(default)]
    pub usage_metadata: UsageMetadata,
    /// The exact model version that served the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    /// The response id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
}

impl GenerateContentResponse {
    /// Concatenates every non-thought text part of the first candidate into a
    /// single string.
    pub fn text(&self) -> String {
        self.candidates
            .first()
            .and_then(|candidate| candidate.content.as_ref())
            .map(Content::text_parts)
            .unwrap_or_default()
    }

    /// The first candidate's finish reason, if any.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.candidates
            .first()
            .and_then(|candidate| candidate.finish_reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serializes_minimally() {
        let request = GenerateContentRequest {
            model: "gemini-3.6-flash".to_string(),
            contents: vec![Content::text(Role::User, "hi")],
            system_instruction: None,
            generation_config: None,
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            })
        );
    }

    #[test]
    fn system_instruction_serializes_without_role() {
        let request = GenerateContentRequest {
            model: "gemini-3.6-flash".to_string(),
            contents: vec![Content::text(Role::User, "hi")],
            system_instruction: Some(Content::system("be brief")),
            generation_config: None,
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(
            json["systemInstruction"],
            serde_json::json!({"parts": [{"text": "be brief"}]})
        );
    }

    #[test]
    fn response_parses_and_extracts_text() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "let me think", "thought": true},
                        {"text": "Hello"},
                        {"text": ", world"}
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 25,
                "thoughtsTokenCount": 7,
                "totalTokenCount": 42
            },
            "modelVersion": "gemini-3.6-flash",
            "responseId": "resp_123"
        }"#;
        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.text(), "Hello, world");
        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
        assert_eq!(response.usage_metadata.prompt_token_count, 10);
        assert_eq!(response.usage_metadata.candidates_token_count, 25);
        assert_eq!(response.usage_metadata.thoughts_token_count, 7);
    }

    #[test]
    fn unknown_finish_reasons_do_not_break_parsing() {
        let json = r#"{
            "candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": "SOME_FUTURE_REASON"}]
        }"#;
        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.finish_reason(), Some(FinishReason::Other));
    }

    #[test]
    fn non_text_parts_are_skipped() {
        let json = r#"{
            "candidates": [{
                "content": {"parts": [
                    {"inlineData": {"mimeType": "image/png", "data": ""}},
                    {"text": "caption"}
                ]}
            }]
        }"#;
        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.text(), "caption");
    }
}
