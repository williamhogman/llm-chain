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

/// A function call made by the model, carried in a [`Part`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCall {
    /// The name of the function to invoke.
    pub name: String,
    /// The arguments as a JSON object, conforming to the declaration's `parameters`.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub args: serde_json::Value,
}

/// The result of running a function, sent back to the model in a [`Part`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionResponse {
    /// The name of the function that was run.
    pub name: String,
    /// The result as a JSON object.
    pub response: serde_json::Value,
}

impl FunctionResponse {
    /// Creates a function response.
    ///
    /// The API requires `response` to be a JSON object; non-object values are
    /// wrapped as `{"result": value}`.
    pub fn new<N: Into<String>>(name: N, response: serde_json::Value) -> Self {
        let response = if response.is_object() {
            response
        } else {
            serde_json::json!({ "result": response })
        };
        Self {
            name: name.into(),
            response,
        }
    }
}

/// One part of a content entry: text, a function call, or a function
/// response. Anything else (inline data, …) deserializes with every field
/// empty and is skipped by [`Content::text_parts`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    /// The text of this part.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text: String,
    /// True when this part is model reasoning rather than the answer
    /// (returned when thoughts are requested).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub thought: bool,
    /// A function call made by the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    /// The result of running a function, sent back to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_response: Option<FunctionResponse>,
    /// Opaque reasoning signature; echo it back verbatim when continuing a
    /// function-calling conversation with a Gemini 3-generation model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

impl Part {
    /// Creates a text part.
    pub fn text<S: Into<String>>(text: S) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    /// Creates a function-response part.
    pub fn function_response(function_response: FunctionResponse) -> Self {
        Self {
            function_response: Some(function_response),
            ..Self::default()
        }
    }
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
            parts: vec![Part::text(text)],
        }
    }

    /// Creates system instructions (a role-less single-part text entry).
    pub fn system<S: Into<String>>(text: S) -> Self {
        Self {
            role: None,
            parts: vec![Part::text(text)],
        }
    }

    /// Creates the user entry that answers function calls: one
    /// [`Part::function_response`] per invoked function.
    pub fn function_responses<I: IntoIterator<Item = FunctionResponse>>(responses: I) -> Self {
        Self {
            role: Some(Role::User),
            parts: responses.into_iter().map(Part::function_response).collect(),
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

    /// The function calls in this entry, in order.
    pub fn function_calls(&self) -> impl Iterator<Item = &FunctionCall> {
        self.parts
            .iter()
            .filter_map(|part| part.function_call.as_ref())
    }
}

/// A function the model may call, declared in a [`Tool`].
///
/// `parameters` is a [JSON Schema](https://json-schema.org/) object (OpenAPI
/// 3.0 subset) describing the function's arguments.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclaration {
    /// The function name the model calls it by.
    pub name: String,
    /// What the function does and when to use it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the function's arguments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

impl FunctionDeclaration {
    /// Creates a function declaration from a name, description and JSON Schema.
    pub fn new<N: Into<String>, D: Into<String>>(
        name: N,
        description: D,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            parameters: Some(parameters),
        }
    }
}

/// A tool made available to the model: a set of function declarations.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    /// The functions in this tool.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub function_declarations: Vec<FunctionDeclaration>,
}

/// How the model chooses among the declared functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FunctionCallingMode {
    /// The model decides whether to call a function (the API default).
    Auto,
    /// The model must call a function.
    Any,
    /// The model must not call any function.
    None,
}

/// Function-calling behavior, nested inside [`ToolConfig`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    /// The function-calling mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<FunctionCallingMode>,
    /// With [`FunctionCallingMode::Any`], restricts the model to these functions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

/// Tool behavior controls, sent as `toolConfig`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    /// Function-calling behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_calling_config: Option<FunctionCallingConfig>,
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
    /// Tools the model may call, set with
    /// [`Options::with_tools`](super::Options::with_tools).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Tool behavior controls, set with
    /// [`Options::with_function_calling_mode`](super::Options::with_function_calling_mode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
}

impl GenerateContentRequest {
    /// Continues a function-calling conversation: appends the model turn from
    /// `response` (preserving thought signatures) followed by a user turn
    /// carrying the function responses.
    ///
    /// Execute the returned request to get the model's next turn — either the
    /// final answer or another round of function calls.
    pub fn with_function_responses<I: IntoIterator<Item = FunctionResponse>>(
        mut self,
        response: &GenerateContentResponse,
        responses: I,
    ) -> Self {
        if let Some(content) = response
            .candidates
            .first()
            .and_then(|candidate| candidate.content.as_ref())
        {
            self.contents.push(content.clone());
        }
        self.contents.push(Content::function_responses(responses));
        self
    }
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

    /// The function calls the model made this turn, in order.
    ///
    /// Non-empty when the model wants functions run. Run each function and
    /// answer with
    /// [`GenerateContentRequest::with_function_responses`].
    pub fn function_calls(&self) -> impl Iterator<Item = &FunctionCall> {
        self.candidates
            .first()
            .and_then(|candidate| candidate.content.as_ref())
            .into_iter()
            .flat_map(Content::function_calls)
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
            tools: None,
            tool_config: None,
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
            tools: None,
            tool_config: None,
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

    #[test]
    fn tools_serialize_on_the_wire_format() {
        let request = GenerateContentRequest {
            model: "gemini-3.6-flash".to_string(),
            contents: vec![Content::text(Role::User, "weather in Stockholm?")],
            system_instruction: None,
            generation_config: None,
            tools: Some(vec![Tool {
                function_declarations: vec![FunctionDeclaration::new(
                    "get_weather",
                    "Get the current weather for a city.",
                    serde_json::json!({
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }),
                )],
            }]),
            tool_config: Some(ToolConfig {
                function_calling_config: Some(FunctionCallingConfig {
                    mode: Some(FunctionCallingMode::Any),
                    allowed_function_names: None,
                }),
            }),
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(
            json["tools"][0]["functionDeclarations"][0]["name"],
            "get_weather"
        );
        assert_eq!(
            json["tools"][0]["functionDeclarations"][0]["parameters"]["properties"]["city"]["type"],
            "string"
        );
        assert_eq!(json["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
    }

    #[test]
    fn function_call_responses_parse_and_expose_calls() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {"name": "get_weather", "args": {"city": "Stockholm"}},
                        "thoughtSignature": "sig_abc"
                    }]
                },
                "finishReason": "STOP"
            }]
        }"#;
        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let calls: Vec<_> = response.function_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].args["city"], "Stockholm");
        assert_eq!(response.text(), "");
    }

    #[test]
    fn function_responses_continue_the_conversation() {
        let response: GenerateContentResponse = serde_json::from_str(
            r#"{
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{
                            "functionCall": {"name": "get_weather", "args": {"city": "Stockholm"}},
                            "thoughtSignature": "sig_abc"
                        }]
                    }
                }]
            }"#,
        )
        .unwrap();
        let request = GenerateContentRequest {
            model: "gemini-3.6-flash".to_string(),
            contents: vec![Content::text(Role::User, "weather?")],
            system_instruction: None,
            generation_config: None,
            tools: None,
            tool_config: None,
        }
        .with_function_responses(
            &response,
            [FunctionResponse::new(
                "get_weather",
                serde_json::json!({"temperature_c": 8, "sky": "cloudy"}),
            )],
        );

        assert_eq!(request.contents.len(), 3);
        let json = serde_json::to_value(&request).unwrap();
        // The model turn is echoed back verbatim, including the signature.
        assert_eq!(
            json["contents"][1]["parts"][0]["thoughtSignature"],
            "sig_abc"
        );
        assert_eq!(
            json["contents"][2],
            serde_json::json!({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": "get_weather",
                        "response": {"temperature_c": 8, "sky": "cloudy"}
                    }
                }]
            })
        );
    }

    #[test]
    fn non_object_function_responses_are_wrapped() {
        let response = FunctionResponse::new("get_time", serde_json::json!("09:41"));
        assert_eq!(response.response, serde_json::json!({"result": "09:41"}));
    }
}
