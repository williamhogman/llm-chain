//! Streaming support for the Chat Completions API.
//!
//! With [`StreamingExecutor::execute_stream`](llm_chain::traits::StreamingExecutor::execute_stream)
//! the API delivers the response as a series of `chat.completion.chunk`
//! objects ([`CreateChatCompletionStreamResponse`]): each carries a delta with
//! newly generated content (or tool-call fragments), and — with usage
//! reporting on, which the executor enables by default — a final chunk with
//! empty `choices` carries the token usage for the whole request.
//!
//! Print live output with the chunk's delta content; fold the full chunk
//! sequence back into a regular [`CreateChatCompletionResponse`] with
//! [`ResponseAccumulator`] when the final response is also wanted.

use std::collections::BTreeMap;

use async_openai::types::chat::{
    CompletionUsage, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
    FinishReason, Role, ServiceTier,
};

/// Folds a stream of `chat.completion.chunk` objects back into a
/// [`CreateChatCompletionResponse`].
///
/// Feed every chunk to [`apply`](ResponseAccumulator::apply); once the stream
/// ends, [`into_response`](ResponseAccumulator::into_response) yields a
/// response equivalent to what [`Executor::execute`](llm_chain::traits::Executor::execute)
/// would have returned — content concatenated, tool-call fragments assembled
/// by index, and the finish reason and usage carried over, so tool-calling
/// conversations can be continued from a streamed turn.
///
/// # Examples
///
/// ```no_run
/// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// use futures::StreamExt as _;
/// use llm_chain::traits::StreamingExecutor as _;
/// use llm_chain_openai::chat::{Executor, ResponseAccumulator};
///
/// # let executor = Executor::new_default();
/// # let request = todo!();
/// let mut stream = executor.execute_stream(request).await?;
/// let mut accumulator = ResponseAccumulator::new();
/// while let Some(chunk) = stream.next().await {
///     let chunk = chunk?;
///     if let Some(content) = chunk.choices.first().and_then(|c| c.delta.content.as_deref()) {
///         print!("{content}");
///     }
///     accumulator.apply(&chunk);
/// }
/// let response = accumulator.into_response().expect("stream produced chunks");
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct ResponseAccumulator {
    started: bool,
    id: String,
    created: u32,
    model: String,
    service_tier: Option<ServiceTier>,
    role: Option<Role>,
    content: String,
    refusal: String,
    /// Tool-call fragments keyed by the API's tool-call index.
    tool_calls: BTreeMap<u32, ToolCallDraft>,
    finish_reason: Option<FinishReason>,
    usage: Option<CompletionUsage>,
}

#[derive(Debug, Default)]
struct ToolCallDraft {
    id: String,
    name: String,
    arguments: String,
}

impl ResponseAccumulator {
    /// Creates an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies one streamed chunk to the response under construction.
    pub fn apply(&mut self, chunk: &CreateChatCompletionStreamResponse) {
        self.started = true;
        if !chunk.id.is_empty() {
            self.id = chunk.id.clone();
        }
        if chunk.created != 0 {
            self.created = chunk.created;
        }
        if !chunk.model.is_empty() {
            self.model = chunk.model.clone();
        }
        if chunk.service_tier.is_some() {
            self.service_tier = chunk.service_tier.clone();
        }
        // Usage arrives on a final chunk with empty `choices` when
        // `stream_options.include_usage` is on.
        if chunk.usage.is_some() {
            self.usage = chunk.usage.clone();
        }
        let Some(choice) = chunk.choices.first() else {
            return;
        };
        if choice.finish_reason.is_some() {
            self.finish_reason = choice.finish_reason;
        }
        if let Some(role) = choice.delta.role {
            self.role = Some(role);
        }
        if let Some(content) = &choice.delta.content {
            self.content.push_str(content);
        }
        if let Some(refusal) = &choice.delta.refusal {
            self.refusal.push_str(refusal);
        }
        if let Some(tool_calls) = &choice.delta.tool_calls {
            for fragment in tool_calls {
                let draft = self.tool_calls.entry(fragment.index).or_default();
                if let Some(id) = &fragment.id {
                    draft.id.push_str(id);
                }
                if let Some(function) = &fragment.function {
                    if let Some(name) = &function.name {
                        draft.name.push_str(name);
                    }
                    if let Some(arguments) = &function.arguments {
                        draft.arguments.push_str(arguments);
                    }
                }
            }
        }
    }

    /// Consumes the accumulator, yielding the assembled response.
    ///
    /// Returns `None` when no chunk was applied.
    pub fn into_response(self) -> Option<CreateChatCompletionResponse> {
        if !self.started {
            return None;
        }
        let tool_calls: Vec<serde_json::Value> = self
            .tool_calls
            .into_values()
            .map(|draft| {
                serde_json::json!({
                    "id": draft.id,
                    "type": "function",
                    "function": {"name": draft.name, "arguments": draft.arguments},
                })
            })
            .collect();
        // Assembled through the wire format: the response type's fields are
        // owned by async-openai, and this stays correct as fields are added.
        let response = serde_json::json!({
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "service_tier": self.service_tier,
            "choices": [{
                "index": 0,
                "message": {
                    "role": self.role.unwrap_or(Role::Assistant),
                    "content": if self.content.is_empty() && !tool_calls.is_empty() {
                        serde_json::Value::Null
                    } else {
                        serde_json::Value::String(self.content)
                    },
                    "refusal": if self.refusal.is_empty() {
                        serde_json::Value::Null
                    } else {
                        serde_json::Value::String(self.refusal)
                    },
                    "tool_calls": if tool_calls.is_empty() {
                        serde_json::Value::Null
                    } else {
                        serde_json::Value::Array(tool_calls)
                    },
                },
                "finish_reason": self.finish_reason,
                "logprobs": null,
            }],
            "usage": self.usage,
        });
        serde_json::from_value(response).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(json: &str) -> CreateChatCompletionStreamResponse {
        serde_json::from_str(json).expect("valid chunk fixture")
    }

    #[test]
    fn text_deltas_accumulate_into_a_response() {
        let chunks = [
            r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":100,"model":"gpt-5.6-terra","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":100,"model":"gpt-5.6-terra","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":100,"model":"gpt-5.6-terra","choices":[{"index":0,"delta":{"content":", world"},"finish_reason":"stop"}]}"#,
            r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":100,"model":"gpt-5.6-terra","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":3,"total_tokens":12}}"#,
        ];
        let mut accumulator = ResponseAccumulator::new();
        for json in chunks {
            accumulator.apply(&chunk(json));
        }
        let response = accumulator.into_response().unwrap();
        assert_eq!(response.id, "chatcmpl-1");
        assert_eq!(response.model, "gpt-5.6-terra");
        let choice = &response.choices[0];
        assert_eq!(choice.message.content.as_deref(), Some("Hello, world"));
        assert_eq!(choice.finish_reason, Some(FinishReason::Stop));
        let usage = response.usage.expect("usage present");
        assert_eq!(usage.prompt_tokens, 9);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.total_tokens, 12);
    }

    #[test]
    fn tool_call_fragments_assemble_by_index() {
        let chunks = [
            r#"{"id":"chatcmpl-2","object":"chat.completion.chunk","created":100,"model":"gpt-5.6-terra","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-2","object":"chat.completion.chunk","created":100,"model":"gpt-5.6-terra","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-2","object":"chat.completion.chunk","created":100,"model":"gpt-5.6-terra","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Stockholm\"}"}}]},"finish_reason":"tool_calls"}]}"#,
        ];
        let mut accumulator = ResponseAccumulator::new();
        for json in chunks {
            accumulator.apply(&chunk(json));
        }
        let response = accumulator.into_response().unwrap();
        let choice = &response.choices[0];
        assert_eq!(choice.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(choice.message.content, None);
        let calls = crate::chat::function_calls(&response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Stockholm"}"#);
    }

    #[test]
    fn empty_accumulator_yields_nothing() {
        assert!(ResponseAccumulator::new().into_response().is_none());
    }
}
