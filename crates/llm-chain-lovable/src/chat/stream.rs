//! Streaming support for the chat completions API.
//!
//! With [`StreamingExecutor::execute_stream`](llm_chain::traits::StreamingExecutor::execute_stream)
//! the gateway delivers the response as server-sent events: a series of
//! [`ChatChunk`]s (`object: "chat.completion.chunk"`) whose `delta` carries
//! the newly generated text (or reasoning, or tool-call fragments), followed
//! by a usage-bearing chunk and a terminating `data: [DONE]` frame (consumed
//! by the executor, never yielded).
//!
//! Print live output with [`Executor::text_delta`](llm_chain::traits::StreamingExecutor::text_delta);
//! fold the full chunk sequence back into one complete response with
//! [`ResponseAccumulator`] when the final response is also wanted.

use serde::{Deserialize, Serialize};

use super::types::{
    ChatResponse, Choice, FinishReason, FunctionCall, Message, Role, ToolCall, Usage, function_type,
};

/// A partial [`FunctionCall`] inside a [`ToolCallDelta`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    /// The tool name; usually arrives whole on the first fragment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// A fragment of the JSON-encoded arguments string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// A tool-call fragment inside a [`Delta`]; fragments with the same
/// [`index`](ToolCallDelta::index) belong to the same call.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Which tool call this fragment belongs to.
    #[serde(default)]
    pub index: u32,
    /// The call id; arrives on the first fragment of a call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The call type (`function`); arrives on the first fragment of a call.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    /// The partial function invocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

/// The incremental message content inside a [`ChunkChoice`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Delta {
    /// The message role; arrives on the first chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    /// Newly generated answer text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Newly generated reasoning, when reasoning was requested.
    #[serde(
        default,
        alias = "reasoning_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub reasoning: Option<String>,
    /// Tool-call fragments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// One choice's increment inside a [`ChatChunk`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkChoice {
    /// The choice index; `0` unless multiple completions were requested.
    #[serde(default)]
    pub index: u32,
    /// The incremental content.
    #[serde(default)]
    pub delta: Delta,
    /// Why the model stopped; set on the choice's final chunk.
    #[serde(default)]
    pub finish_reason: Option<FinishReason>,
}

/// A streamed chat completion chunk (`object: "chat.completion.chunk"`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatChunk {
    /// The response id, shared by every chunk of the stream.
    #[serde(default)]
    pub id: String,
    /// The object type; `chat.completion.chunk`.
    #[serde(default)]
    pub object: String,
    /// Unix timestamp of creation.
    #[serde(default)]
    pub created: u64,
    /// The model producing the response.
    #[serde(default)]
    pub model: String,
    /// The incremental choices; empty on the final usage-only chunk.
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    /// Token usage; arrives on a final chunk (the executor requests it via
    /// `stream_options.include_usage`).
    #[serde(default)]
    pub usage: Option<Usage>,
    /// The run id from the gateway's `X-Lovable-AIG-Run-ID` response header,
    /// correlating this stream with Lovable AI usage logs. Set by the
    /// [`Executor`](super::Executor) on every chunk, not part of the JSON body.
    #[serde(skip)]
    pub run_id: Option<String>,
}

impl ChatChunk {
    /// The newly generated answer text in this chunk, if any.
    pub fn text(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|choice| choice.delta.content.as_deref())
    }
}

/// Folds a stream of [`ChatChunk`]s back into one complete [`ChatResponse`].
///
/// Feed every chunk to [`apply`](ResponseAccumulator::apply); once the stream
/// ends, [`into_response`](ResponseAccumulator::into_response) yields a
/// response equal to what [`Executor::execute`](llm_chain::traits::Executor::execute)
/// would have returned — text and reasoning concatenated, tool-call fragments
/// reassembled by index, and the final chunk's finish reason and usage
/// carried over, so tool-calling conversations can be continued from a
/// streamed turn.
///
/// # Examples
///
/// ```no_run
/// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// use futures::StreamExt as _;
/// use llm_chain::traits::StreamingExecutor as _;
/// use llm_chain_lovable::chat::{Executor, ResponseAccumulator};
///
/// # let executor = Executor::new_default()?;
/// # let request = todo!();
/// let mut stream = executor.execute_stream(request).await?;
/// let mut accumulator = ResponseAccumulator::new();
/// while let Some(chunk) = stream.next().await {
///     let chunk = chunk?;
///     if let Some(text) = chunk.text() {
///         print!("{text}");
///     }
///     accumulator.apply(&chunk);
/// }
/// let response = accumulator.into_response().expect("stream produced chunks");
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct ResponseAccumulator {
    response: Option<ChatResponse>,
}

impl ResponseAccumulator {
    /// Creates an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies one streamed chunk to the response under construction.
    pub fn apply(&mut self, chunk: &ChatChunk) {
        let response = self.response.get_or_insert_with(|| ChatResponse {
            id: String::new(),
            object: "chat.completion".to_string(),
            created: 0,
            model: String::new(),
            choices: Vec::new(),
            usage: None,
            run_id: None,
        });
        if response.id.is_empty() && !chunk.id.is_empty() {
            response.id = chunk.id.clone();
        }
        if response.model.is_empty() && !chunk.model.is_empty() {
            response.model = chunk.model.clone();
        }
        if response.created == 0 {
            response.created = chunk.created;
        }
        if response.run_id.is_none() {
            response.run_id = chunk.run_id.clone();
        }
        if chunk.usage.is_some() {
            response.usage = chunk.usage;
        }

        for chunk_choice in &chunk.choices {
            let index = chunk_choice.index as usize;
            while response.choices.len() <= index {
                response.choices.push(Choice {
                    index: response.choices.len() as u32,
                    message: Message {
                        role: Role::Assistant,
                        content: None,
                        reasoning: None,
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    finish_reason: None,
                });
            }
            let choice = &mut response.choices[index];
            let delta = &chunk_choice.delta;

            if let Some(role) = delta.role {
                choice.message.role = role;
            }
            if let Some(content) = &delta.content {
                choice
                    .message
                    .content
                    .get_or_insert_with(String::new)
                    .push_str(content);
            }
            if let Some(reasoning) = &delta.reasoning {
                choice
                    .message
                    .reasoning
                    .get_or_insert_with(String::new)
                    .push_str(reasoning);
            }
            if let Some(fragments) = &delta.tool_calls {
                let calls = choice.message.tool_calls.get_or_insert_with(Vec::new);
                for fragment in fragments {
                    let call_index = fragment.index as usize;
                    while calls.len() <= call_index {
                        calls.push(ToolCall {
                            id: String::new(),
                            call_type: function_type(),
                            function: FunctionCall::default(),
                        });
                    }
                    let call = &mut calls[call_index];
                    if let Some(id) = &fragment.id {
                        call.id = id.clone();
                    }
                    if let Some(call_type) = &fragment.call_type {
                        call.call_type = call_type.clone();
                    }
                    if let Some(function) = &fragment.function {
                        if let Some(name) = &function.name {
                            call.function.name.push_str(name);
                        }
                        if let Some(arguments) = &function.arguments {
                            call.function.arguments.push_str(arguments);
                        }
                    }
                }
            }
            if chunk_choice.finish_reason.is_some() {
                choice.finish_reason = chunk_choice.finish_reason;
            }
        }
    }

    /// Whether a chunk carrying a finish reason has been seen.
    pub fn is_complete(&self) -> bool {
        self.response.as_ref().is_some_and(|response| {
            response
                .choices
                .iter()
                .any(|choice| choice.finish_reason.is_some())
        })
    }

    /// The response assembled so far, if any chunk has been applied.
    pub fn response(&self) -> Option<&ChatResponse> {
        self.response.as_ref()
    }

    /// Consumes the accumulator, yielding the assembled response.
    pub fn into_response(self) -> Option<ChatResponse> {
        self.response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(content: &str) -> ChatChunk {
        ChatChunk {
            id: "chatcmpl-1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1_785_000_000,
            model: "google/gemini-3.6-flash".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some(Role::Assistant),
                    content: Some(content.to_string()),
                    reasoning: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
            run_id: None,
        }
    }

    #[test]
    fn text_chunks_concatenate_and_final_metadata_wins() {
        let mut accumulator = ResponseAccumulator::new();
        accumulator.apply(&chunk("Hello"));
        assert!(!accumulator.is_complete());
        accumulator.apply(&chunk(", world"));

        let mut last = chunk("!");
        last.choices[0].finish_reason = Some(FinishReason::Stop);
        accumulator.apply(&last);

        // Usage arrives on a trailing choice-less chunk.
        let usage_chunk = ChatChunk {
            usage: Some(Usage {
                prompt_tokens: 7,
                completion_tokens: 9,
                total_tokens: 16,
            }),
            ..ChatChunk::default()
        };
        accumulator.apply(&usage_chunk);

        assert!(accumulator.is_complete());
        let response = accumulator.into_response().unwrap();
        assert_eq!(response.text(), "Hello, world!");
        assert_eq!(response.id, "chatcmpl-1");
        assert_eq!(response.model, "google/gemini-3.6-flash");
        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
        assert_eq!(response.usage.unwrap().total_tokens, 16);
    }

    #[test]
    fn reasoning_concatenates_separately_from_the_answer() {
        let mut first = chunk("");
        first.choices[0].delta.content = None;
        first.choices[0].delta.reasoning = Some("Let me ".to_string());
        let mut second = chunk("");
        second.choices[0].delta.content = None;
        second.choices[0].delta.reasoning = Some("think".to_string());

        let mut accumulator = ResponseAccumulator::new();
        accumulator.apply(&first);
        accumulator.apply(&second);
        accumulator.apply(&chunk("Answer"));

        let response = accumulator.into_response().unwrap();
        assert_eq!(response.reasoning(), Some("Let me think"));
        assert_eq!(response.text(), "Answer");
    }

    #[test]
    fn tool_call_fragments_reassemble_by_index() {
        let fragment = |index: u32,
                        id: Option<&str>,
                        name: Option<&str>,
                        arguments: Option<&str>| ToolCallDelta {
            index,
            id: id.map(str::to_string),
            call_type: id.map(|_| "function".to_string()),
            function: Some(FunctionCallDelta {
                name: name.map(str::to_string),
                arguments: arguments.map(str::to_string),
            }),
        };
        let with_fragments = |fragments: Vec<ToolCallDelta>| {
            let mut c = chunk("");
            c.choices[0].delta.content = None;
            c.choices[0].delta.tool_calls = Some(fragments);
            c
        };

        let mut accumulator = ResponseAccumulator::new();
        accumulator.apply(&with_fragments(vec![fragment(
            0,
            Some("call_1"),
            Some("get_weather"),
            Some("{\"city\": "),
        )]));
        accumulator.apply(&with_fragments(vec![
            fragment(0, None, None, Some("\"Stockholm\"}")),
            fragment(1, Some("call_2"), Some("get_time"), Some("{}")),
        ]));

        let response = accumulator.into_response().unwrap();
        let calls = response.tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].call_type, "function");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            calls[0].function.parsed_arguments().unwrap()["city"],
            "Stockholm"
        );
        assert_eq!(calls[1].id, "call_2");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn run_id_is_carried_into_the_response() {
        let mut first = chunk("hi");
        first.run_id = Some("run-123".to_string());
        let mut accumulator = ResponseAccumulator::new();
        accumulator.apply(&first);
        accumulator.apply(&chunk("!"));
        let response = accumulator.into_response().unwrap();
        assert_eq!(response.run_id.as_deref(), Some("run-123"));
    }

    #[test]
    fn chunks_parse_from_wire_json() {
        let json = r#"{
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1785000000,
            "model": "google/gemini-3.6-flash",
            "choices": [{"index": 0, "delta": {"content": "Hi", "reasoning_content": "hm"}, "finish_reason": null}]
        }"#;
        let chunk: ChatChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.text(), Some("Hi"));
        assert_eq!(chunk.choices[0].delta.reasoning.as_deref(), Some("hm"));
    }

    #[test]
    fn empty_accumulator_yields_nothing() {
        assert!(ResponseAccumulator::new().into_response().is_none());
        assert!(!ResponseAccumulator::new().is_complete());
    }
}
