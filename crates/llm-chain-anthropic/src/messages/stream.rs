//! Streaming events for the Messages API.
//!
//! With [`StreamingExecutor::execute_stream`](llm_chain::traits::StreamingExecutor::execute_stream)
//! the API delivers a response as a series of [`StreamEvent`]s: a
//! [`MessageStart`](StreamEvent::MessageStart) skeleton, then per-block
//! start/delta/stop events carrying the generated text (or thinking, or tool
//! arguments), then a final [`MessageDelta`](StreamEvent::MessageDelta) with
//! the stop reason and usage.
//!
//! Print live output with [`StreamEvent::text_delta`]; fold the full event
//! sequence back into a regular [`MessagesResponse`] with
//! [`ResponseAccumulator`] when the final response is also wanted.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::types::{ContentBlock, MessagesResponse, StopReason, Usage};

/// A single event in a streamed Messages API response.
///
/// Mirrors the API's SSE event types. Unknown event types parse as
/// [`StreamEvent::Other`] so new server-side additions never break a stream
/// mid-response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// The response skeleton: id, model and prompt-side usage, with empty content.
    MessageStart {
        /// The message skeleton; its `content` is empty at this point.
        message: MessagesResponse,
    },
    /// A new content block begins at `index`.
    ContentBlockStart {
        /// The position of the block in the final content array.
        index: usize,
        /// The empty block being started (its text/input arrives via deltas).
        content_block: ContentBlock,
    },
    /// New content for the block at `index`.
    ContentBlockDelta {
        /// The position of the block being extended.
        index: usize,
        /// The new content: text, thinking, or partial tool-input JSON.
        delta: ContentDelta,
    },
    /// The block at `index` is complete.
    ContentBlockStop {
        /// The position of the completed block.
        index: usize,
    },
    /// Top-level response changes: the stop reason and cumulative output usage.
    MessageDelta {
        /// The stop reason and stop sequence, once known.
        delta: MessageDelta,
        /// Cumulative usage; `output_tokens` grows as the response streams.
        #[serde(default)]
        usage: Usage,
    },
    /// The response is complete.
    MessageStop,
    /// Keep-alive; carries nothing.
    Ping,
    /// Any event type this crate does not model.
    #[serde(other)]
    Other,
}

impl StreamEvent {
    /// The new answer text carried by this event, if any.
    ///
    /// Concatenating every `text_delta` across the stream yields exactly
    /// [`MessagesResponse::text`] of the final response.
    pub fn text_delta(&self) -> Option<&str> {
        match self {
            Self::ContentBlockDelta {
                delta: ContentDelta::TextDelta { text },
                ..
            } => Some(text),
            _ => None,
        }
    }

    /// The new thinking text carried by this event, if any (extended
    /// thinking only).
    pub fn thinking_delta(&self) -> Option<&str> {
        match self {
            Self::ContentBlockDelta {
                delta: ContentDelta::ThinkingDelta { thinking },
                ..
            } => Some(thinking),
            _ => None,
        }
    }
}

/// The payload of a [`StreamEvent::ContentBlockDelta`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    /// New answer text for a text block.
    TextDelta {
        /// The text fragment.
        text: String,
    },
    /// A fragment of the JSON arguments for a tool-use block.
    InputJsonDelta {
        /// The JSON fragment; concatenate every fragment for the block and
        /// parse once the block stops.
        partial_json: String,
    },
    /// New thinking text for a thinking block.
    ThinkingDelta {
        /// The thinking fragment.
        thinking: String,
    },
    /// The integrity signature of a completed thinking block.
    SignatureDelta {
        /// The signature fragment.
        signature: String,
    },
    /// Any delta type this crate does not model.
    #[serde(other)]
    Other,
}

/// The top-level changes carried by a [`StreamEvent::MessageDelta`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MessageDelta {
    /// Why the model stopped, once known.
    #[serde(default)]
    pub stop_reason: Option<StopReason>,
    /// Which stop sequence fired, if any.
    #[serde(default)]
    pub stop_sequence: Option<String>,
}

/// Folds a stream of [`StreamEvent`]s back into a [`MessagesResponse`].
///
/// Feed every event to [`apply`](ResponseAccumulator::apply); once the stream
/// ends (or [`is_complete`](ResponseAccumulator::is_complete) reports the
/// `message_stop` event was seen), [`into_response`](ResponseAccumulator::into_response)
/// yields a response equal to what [`Executor::execute`](llm_chain::traits::Executor::execute)
/// would have returned — including assembled tool-use inputs and thinking
/// signatures, so tool-calling conversations can be continued from a streamed
/// turn.
///
/// # Examples
///
/// ```no_run
/// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// use futures::StreamExt as _;
/// use llm_chain::traits::StreamingExecutor as _;
/// use llm_chain_anthropic::messages::{Executor, ResponseAccumulator};
///
/// # let executor = Executor::with_api_key("sk-ant-...");
/// # let request = todo!();
/// let mut stream = executor.execute_stream(request).await?;
/// let mut accumulator = ResponseAccumulator::new();
/// while let Some(event) = stream.next().await {
///     let event = event?;
///     if let Some(text) = event.text_delta() {
///         print!("{text}");
///     }
///     accumulator.apply(&event);
/// }
/// let response = accumulator.into_response().expect("stream produced a message");
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct ResponseAccumulator {
    response: Option<MessagesResponse>,
    /// Tool-input JSON fragments per block index, parsed at block stop.
    partial_json: HashMap<usize, String>,
    complete: bool,
}

impl ResponseAccumulator {
    /// Creates an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies one event to the response under construction.
    pub fn apply(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::MessageStart { message } => {
                self.response = Some(message.clone());
            }
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                let Some(response) = &mut self.response else {
                    return;
                };
                // Pad with placeholders if indices arrive out of order.
                while response.content.len() < *index {
                    response.content.push(ContentBlock::Other);
                }
                if response.content.len() == *index {
                    response.content.push(content_block.clone());
                } else {
                    response.content[*index] = content_block.clone();
                }
            }
            StreamEvent::ContentBlockDelta { index, delta } => {
                if let ContentDelta::InputJsonDelta { partial_json } = delta {
                    self.partial_json
                        .entry(*index)
                        .or_default()
                        .push_str(partial_json);
                    return;
                }
                let Some(block) = self
                    .response
                    .as_mut()
                    .and_then(|response| response.content.get_mut(*index))
                else {
                    return;
                };
                match (block, delta) {
                    (ContentBlock::Text { text }, ContentDelta::TextDelta { text: fragment }) => {
                        text.push_str(fragment);
                    }
                    (
                        ContentBlock::Thinking { thinking, .. },
                        ContentDelta::ThinkingDelta { thinking: fragment },
                    ) => {
                        thinking.push_str(fragment);
                    }
                    (
                        ContentBlock::Thinking { signature, .. },
                        ContentDelta::SignatureDelta {
                            signature: fragment,
                        },
                    ) => {
                        signature.get_or_insert_with(String::new).push_str(fragment);
                    }
                    _ => {}
                }
            }
            StreamEvent::ContentBlockStop { index } => {
                let Some(raw) = self.partial_json.remove(index) else {
                    return;
                };
                if let Some(ContentBlock::ToolUse(tool_use)) = self
                    .response
                    .as_mut()
                    .and_then(|response| response.content.get_mut(*index))
                {
                    // An empty accumulation means a no-argument tool: `{}`.
                    let raw = if raw.is_empty() {
                        "{}".to_string()
                    } else {
                        raw
                    };
                    // Malformed JSON is preserved as a string rather than lost.
                    tool_use.input =
                        serde_json::from_str(&raw).unwrap_or(serde_json::Value::String(raw));
                }
            }
            StreamEvent::MessageDelta { delta, usage } => {
                let Some(response) = &mut self.response else {
                    return;
                };
                if delta.stop_reason.is_some() {
                    response.stop_reason = delta.stop_reason;
                }
                if delta.stop_sequence.is_some() {
                    response.stop_sequence = delta.stop_sequence.clone();
                }
                // Output usage is cumulative; input usage came with message_start.
                response.usage.output_tokens = usage.output_tokens;
                if usage.input_tokens != 0 {
                    response.usage.input_tokens = usage.input_tokens;
                }
            }
            StreamEvent::MessageStop => self.complete = true,
            StreamEvent::Ping | StreamEvent::Other => {}
        }
    }

    /// Whether the `message_stop` event has been seen.
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// The response assembled so far, if a `message_start` has been seen.
    pub fn response(&self) -> Option<&MessagesResponse> {
        self.response.as_ref()
    }

    /// Consumes the accumulator, yielding the assembled response.
    pub fn into_response(self) -> Option<MessagesResponse> {
        self.response
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::ToolUse;
    use super::*;

    fn event(json: &str) -> StreamEvent {
        serde_json::from_str(json).expect("valid event json")
    }

    #[test]
    fn events_parse_from_the_wire_format() {
        let start = event(
            r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5","content":[],"stop_reason":null,"usage":{"input_tokens":12,"output_tokens":1}}}"#,
        );
        match &start {
            StreamEvent::MessageStart { message } => {
                assert_eq!(message.id, "msg_1");
                assert_eq!(message.usage.input_tokens, 12);
            }
            other => panic!("unexpected: {other:?}"),
        }

        let delta = event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hej"}}"#,
        );
        assert_eq!(delta.text_delta(), Some("Hej"));

        assert_eq!(event(r#"{"type":"ping"}"#), StreamEvent::Ping);
        assert_eq!(
            event(r#"{"type":"some_future_event","payload":1}"#),
            StreamEvent::Other
        );
        assert_eq!(
            event(r#"{"type":"content_block_delta","index":0,"delta":{"type":"future_delta"}}"#),
            StreamEvent::ContentBlockDelta {
                index: 0,
                delta: ContentDelta::Other
            }
        );
    }

    #[test]
    fn accumulator_rebuilds_a_text_response() {
        let events = [
            r#"{"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-5","content":[],"usage":{"input_tokens":10,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":", world"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":9}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let mut accumulator = ResponseAccumulator::new();
        for json in events {
            accumulator.apply(&event(json));
        }
        assert!(accumulator.is_complete());
        let response = accumulator.into_response().unwrap();
        assert_eq!(response.text(), "Hello, world");
        assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 9);
    }

    #[test]
    fn accumulator_assembles_tool_use_input_from_fragments() {
        let events = [
            r#"{"type":"message_start","message":{"id":"msg_1","model":"m","content":[],"usage":{"input_tokens":5,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"get_weather","input":{}}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"Stockholm\"}"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":20}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let mut accumulator = ResponseAccumulator::new();
        for json in events {
            accumulator.apply(&event(json));
        }
        let response = accumulator.into_response().unwrap();
        assert_eq!(response.stop_reason, Some(StopReason::ToolUse));
        let tool_uses: Vec<&ToolUse> = response.tool_uses().collect();
        assert_eq!(tool_uses.len(), 1);
        assert_eq!(tool_uses[0].name, "get_weather");
        assert_eq!(tool_uses[0].input, serde_json::json!({"city": "Stockholm"}));
    }

    #[test]
    fn accumulator_assembles_thinking_blocks_with_signatures() {
        let events = [
            r#"{"type":"message_start","message":{"id":"msg_1","model":"m","content":[],"usage":{"input_tokens":5,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig123"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let mut accumulator = ResponseAccumulator::new();
        for json in events {
            accumulator.apply(&event(json));
        }
        let response = accumulator.into_response().unwrap();
        assert_eq!(response.text(), "Answer");
        match &response.content[0] {
            ContentBlock::Thinking {
                thinking,
                signature,
            } => {
                assert_eq!(thinking, "Let me think");
                assert_eq!(signature.as_deref(), Some("sig123"));
            }
            other => panic!("unexpected block: {other:?}"),
        }
    }

    #[test]
    fn accumulator_ignores_events_before_message_start() {
        let mut accumulator = ResponseAccumulator::new();
        accumulator.apply(&event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}"#,
        ));
        assert!(accumulator.response().is_none());
        assert!(accumulator.into_response().is_none());
    }
}
