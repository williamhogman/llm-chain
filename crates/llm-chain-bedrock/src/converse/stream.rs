//! Streaming events for the Converse API (ConverseStream).
//!
//! With [`StreamingExecutor::execute_stream`](llm_chain::traits::StreamingExecutor::execute_stream)
//! the API delivers a response as a series of [`StreamEvent`]s: a
//! [`MessageStart`](StreamEvent::MessageStart), then per-block
//! start/delta/stop events carrying the generated text (or reasoning, or tool
//! arguments), a [`MessageStop`](StreamEvent::MessageStop) with the stop
//! reason, and a final [`Metadata`](StreamEvent::Metadata) with usage and
//! latency.
//!
//! Print live output with [`StreamEvent::text_delta`]; fold the full event
//! sequence back into a regular [`ConverseResponse`] with
//! [`ResponseAccumulator`] when the final response is also wanted.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::types::{
    ContentBlock, ConverseOutput, ConverseResponse, Message, Metrics, ReasoningContent,
    ReasoningText, Role, StopReason, TokenUsage, ToolUseBlock,
};

/// A single event in a streamed Converse API response.
///
/// Mirrors the members of the API's `ConverseStreamOutput` union; the event
/// name arrives in the binary frame's `:event-type` header and the payload is
/// JSON. Unknown event types parse as [`StreamEvent::Other`] so new
/// server-side additions never break a stream mid-response.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamEvent {
    /// The response begins; the generated message will have this role.
    MessageStart {
        /// The role of the message being generated (always assistant).
        role: Role,
    },
    /// A new content block begins at `index`. Only tool-use blocks carry
    /// start information (the tool id and name); text blocks start implicitly
    /// with their first delta.
    ContentBlockStart {
        /// The position of the block in the final content array.
        index: usize,
        /// What kind of block is starting.
        start: ContentBlockStart,
    },
    /// New content for the block at `index`.
    ContentBlockDelta {
        /// The position of the block being extended.
        index: usize,
        /// The new content: text, tool-input JSON, or reasoning.
        delta: ContentDelta,
    },
    /// The block at `index` is complete.
    ContentBlockStop {
        /// The position of the completed block.
        index: usize,
    },
    /// The response is complete.
    MessageStop {
        /// Why the model stopped.
        stop_reason: Option<StopReason>,
    },
    /// Token usage and latency for the whole request, sent last.
    Metadata {
        /// Token accounting.
        usage: TokenUsage,
        /// Latency metrics.
        metrics: Option<Metrics>,
    },
    /// Any event type this crate does not model.
    Other,
}

impl StreamEvent {
    /// The new answer text carried by this event, if any.
    ///
    /// Concatenating every `text_delta` across the stream yields exactly
    /// [`ConverseResponse::text`] of the final response.
    pub fn text_delta(&self) -> Option<&str> {
        match self {
            Self::ContentBlockDelta { delta, .. } => delta.text.as_deref(),
            _ => None,
        }
    }

    /// The new reasoning text carried by this event, if any (models with
    /// reasoning enabled only).
    pub fn reasoning_delta(&self) -> Option<&str> {
        match self {
            Self::ContentBlockDelta { delta, .. } => {
                delta.reasoning_content.as_ref()?.text.as_deref()
            }
            _ => None,
        }
    }
}

/// The `start` union of a [`StreamEvent::ContentBlockStart`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentBlockStart {
    /// Present when the starting block is a tool call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_use: Option<ToolUseStart>,
}

/// The id and name of a tool call whose arguments will arrive as
/// [`ToolUseDelta`] fragments.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolUseStart {
    /// The unique id of this call.
    pub tool_use_id: String,
    /// The name of the tool being invoked.
    pub name: String,
}

/// The `delta` union of a [`StreamEvent::ContentBlockDelta`]. Exactly one
/// field is set per event.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentDelta {
    /// New answer text for a text block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// A fragment of the JSON arguments for a tool-use block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_use: Option<ToolUseDelta>,
    /// New reasoning content for a reasoning block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<ReasoningDelta>,
}

/// A fragment of a tool call's JSON arguments; concatenate every fragment for
/// the block and parse once the block stops.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolUseDelta {
    /// The JSON fragment.
    #[serde(default)]
    pub input: String,
}

/// A fragment of a reasoning block: text, or the integrity signature of a
/// completed block.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningDelta {
    /// New reasoning text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// A fragment of the block's integrity signature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// Parses one event's JSON payload by its `:event-type` header value.
///
/// Unknown event types yield [`StreamEvent::Other`]; a payload that fails to
/// parse for a known type is an error.
pub(crate) fn parse_event(
    event_type: &str,
    payload: &[u8],
) -> Result<StreamEvent, serde_json::Error> {
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct MessageStartPayload {
        #[serde(default)]
        role: Option<Role>,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct BlockStartPayload {
        #[serde(default)]
        content_block_index: usize,
        #[serde(default)]
        start: Option<ContentBlockStart>,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct BlockDeltaPayload {
        #[serde(default)]
        content_block_index: usize,
        #[serde(default)]
        delta: Option<ContentDelta>,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct BlockStopPayload {
        #[serde(default)]
        content_block_index: usize,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct MessageStopPayload {
        #[serde(default)]
        stop_reason: Option<StopReason>,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct MetadataPayload {
        #[serde(default)]
        usage: TokenUsage,
        #[serde(default)]
        metrics: Option<Metrics>,
    }

    Ok(match event_type {
        "messageStart" => {
            let payload: MessageStartPayload = serde_json::from_slice(payload)?;
            StreamEvent::MessageStart {
                role: payload.role.unwrap_or(Role::Assistant),
            }
        }
        "contentBlockStart" => {
            let payload: BlockStartPayload = serde_json::from_slice(payload)?;
            StreamEvent::ContentBlockStart {
                index: payload.content_block_index,
                start: payload.start.unwrap_or_default(),
            }
        }
        "contentBlockDelta" => {
            let payload: BlockDeltaPayload = serde_json::from_slice(payload)?;
            StreamEvent::ContentBlockDelta {
                index: payload.content_block_index,
                delta: payload.delta.unwrap_or_default(),
            }
        }
        "contentBlockStop" => {
            let payload: BlockStopPayload = serde_json::from_slice(payload)?;
            StreamEvent::ContentBlockStop {
                index: payload.content_block_index,
            }
        }
        "messageStop" => {
            let payload: MessageStopPayload = serde_json::from_slice(payload)?;
            StreamEvent::MessageStop {
                stop_reason: payload.stop_reason,
            }
        }
        "metadata" => {
            let payload: MetadataPayload = serde_json::from_slice(payload)?;
            StreamEvent::Metadata {
                usage: payload.usage,
                metrics: payload.metrics,
            }
        }
        _ => StreamEvent::Other,
    })
}

/// A content block being reassembled from deltas.
#[derive(Debug)]
enum BlockDraft {
    Text(String),
    Reasoning {
        text: String,
        signature: Option<String>,
    },
    ToolUse {
        tool_use_id: String,
        name: String,
        input_json: String,
    },
}

/// Folds a stream of [`StreamEvent`]s back into a [`ConverseResponse`].
///
/// Feed every event to [`apply`](ResponseAccumulator::apply); once the stream
/// ends, [`into_response`](ResponseAccumulator::into_response) yields a
/// response equivalent to what [`Executor::execute`](llm_chain::traits::Executor::execute)
/// would have returned — text concatenated, tool-argument fragments assembled
/// and parsed, and the stop reason, usage and metrics carried over, so
/// tool-calling conversations can be continued from a streamed turn.
///
/// # Examples
///
/// ```no_run
/// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// use futures::StreamExt as _;
/// use llm_chain::traits::StreamingExecutor as _;
/// use llm_chain_bedrock::converse::{Executor, ResponseAccumulator};
///
/// # let executor = Executor::new_default()?;
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
/// let response = accumulator.into_response().expect("stream produced events");
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct ResponseAccumulator {
    started: bool,
    role: Option<Role>,
    /// Blocks under construction, keyed by content block index.
    blocks: BTreeMap<usize, BlockDraft>,
    stop_reason: Option<StopReason>,
    usage: TokenUsage,
    metrics: Option<Metrics>,
}

impl ResponseAccumulator {
    /// Creates an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies one streamed event to the response under construction.
    pub fn apply(&mut self, event: &StreamEvent) {
        self.started = true;
        match event {
            StreamEvent::MessageStart { role } => self.role = Some(*role),
            StreamEvent::ContentBlockStart { index, start } => {
                if let Some(tool_use) = &start.tool_use {
                    self.blocks.insert(
                        *index,
                        BlockDraft::ToolUse {
                            tool_use_id: tool_use.tool_use_id.clone(),
                            name: tool_use.name.clone(),
                            input_json: String::new(),
                        },
                    );
                }
            }
            StreamEvent::ContentBlockDelta { index, delta } => self.apply_delta(*index, delta),
            StreamEvent::MessageStop { stop_reason } => self.stop_reason = *stop_reason,
            StreamEvent::Metadata { usage, metrics } => {
                self.usage = *usage;
                self.metrics = *metrics;
            }
            StreamEvent::ContentBlockStop { .. } | StreamEvent::Other => {}
        }
    }

    fn apply_delta(&mut self, index: usize, delta: &ContentDelta) {
        if let Some(fragment) = &delta.text
            && let BlockDraft::Text(text) = self
                .blocks
                .entry(index)
                .or_insert_with(|| BlockDraft::Text(String::new()))
        {
            text.push_str(fragment);
        }
        if let Some(tool_use) = &delta.tool_use
            && let BlockDraft::ToolUse { input_json, .. } =
                self.blocks.entry(index).or_insert_with(|| {
                    // A delta without a preceding start; reassemble what we can.
                    BlockDraft::ToolUse {
                        tool_use_id: String::new(),
                        name: String::new(),
                        input_json: String::new(),
                    }
                })
        {
            input_json.push_str(&tool_use.input);
        }
        if let Some(reasoning) = &delta.reasoning_content
            && let BlockDraft::Reasoning { text, signature } =
                self.blocks
                    .entry(index)
                    .or_insert_with(|| BlockDraft::Reasoning {
                        text: String::new(),
                        signature: None,
                    })
        {
            if let Some(fragment) = &reasoning.text {
                text.push_str(fragment);
            }
            if let Some(fragment) = &reasoning.signature {
                signature.get_or_insert_default().push_str(fragment);
            }
        }
    }

    /// Consumes the accumulator, yielding the assembled response.
    ///
    /// Tool-argument fragments are parsed as JSON (an empty argument stream
    /// parses as `{}`; unparseable arguments are preserved as a JSON string
    /// rather than dropped). Returns `None` when no event was applied.
    pub fn into_response(self) -> Option<ConverseResponse> {
        if !self.started {
            return None;
        }
        let content = self
            .blocks
            .into_values()
            .map(|draft| match draft {
                BlockDraft::Text(text) => ContentBlock::Text { text },
                BlockDraft::Reasoning { text, signature } => ContentBlock::Reasoning {
                    reasoning_content: ReasoningContent {
                        reasoning_text: Some(ReasoningText { text, signature }),
                    },
                },
                BlockDraft::ToolUse {
                    tool_use_id,
                    name,
                    input_json,
                } => {
                    let input = if input_json.trim().is_empty() {
                        serde_json::Value::Object(serde_json::Map::new())
                    } else {
                        serde_json::from_str(&input_json)
                            .unwrap_or(serde_json::Value::String(input_json))
                    };
                    ContentBlock::ToolUse {
                        tool_use: ToolUseBlock {
                            tool_use_id,
                            name,
                            input,
                        },
                    }
                }
            })
            .collect();
        Some(ConverseResponse {
            output: Some(ConverseOutput {
                message: Some(Message {
                    role: self.role.unwrap_or(Role::Assistant),
                    content,
                }),
            }),
            stop_reason: self.stop_reason,
            usage: self.usage,
            metrics: self.metrics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(event_type: &str, payload: &str) -> StreamEvent {
        parse_event(event_type, payload.as_bytes()).expect("valid payload")
    }

    #[test]
    fn events_parse_from_their_wire_payloads() {
        assert_eq!(
            parsed("messageStart", r#"{"role":"assistant"}"#),
            StreamEvent::MessageStart {
                role: Role::Assistant
            }
        );
        assert_eq!(
            parsed(
                "contentBlockDelta",
                r#"{"contentBlockIndex":2,"delta":{"text":"Hej"}}"#
            ),
            StreamEvent::ContentBlockDelta {
                index: 2,
                delta: ContentDelta {
                    text: Some("Hej".to_string()),
                    ..Default::default()
                }
            }
        );
        assert_eq!(
            parsed("messageStop", r#"{"stopReason":"end_turn"}"#),
            StreamEvent::MessageStop {
                stop_reason: Some(StopReason::EndTurn)
            }
        );
        assert_eq!(parsed("someFutureEvent", r#"{"x":1}"#), StreamEvent::Other);
        assert!(parse_event("messageStop", b"not json").is_err());
    }

    #[test]
    fn text_deltas_accumulate_into_a_response() {
        let events = [
            parsed("messageStart", r#"{"role":"assistant"}"#),
            parsed(
                "contentBlockDelta",
                r#"{"contentBlockIndex":0,"delta":{"text":"Hello"}}"#,
            ),
            parsed(
                "contentBlockDelta",
                r#"{"contentBlockIndex":0,"delta":{"text":", world"}}"#,
            ),
            parsed("contentBlockStop", r#"{"contentBlockIndex":0}"#),
            parsed("messageStop", r#"{"stopReason":"end_turn"}"#),
            parsed(
                "metadata",
                r#"{"usage":{"inputTokens":9,"outputTokens":3,"totalTokens":12},"metrics":{"latencyMs":210}}"#,
            ),
        ];
        let mut accumulator = ResponseAccumulator::new();
        let mut streamed = String::new();
        for event in &events {
            if let Some(text) = event.text_delta() {
                streamed.push_str(text);
            }
            accumulator.apply(event);
        }
        let response = accumulator.into_response().unwrap();
        assert_eq!(streamed, "Hello, world");
        assert_eq!(response.text(), "Hello, world");
        assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
        assert_eq!(response.usage.total_tokens, 12);
        assert_eq!(response.metrics.unwrap().latency_ms, 210);
    }

    #[test]
    fn tool_argument_fragments_assemble_and_parse() {
        let events = [
            parsed("messageStart", r#"{"role":"assistant"}"#),
            parsed(
                "contentBlockStart",
                r#"{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"tool-1","name":"get_weather"}}}"#,
            ),
            parsed(
                "contentBlockDelta",
                r#"{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"city\":"}}}"#,
            ),
            parsed(
                "contentBlockDelta",
                r#"{"contentBlockIndex":0,"delta":{"toolUse":{"input":"\"Norrtälje\"}"}}}"#,
            ),
            parsed("contentBlockStop", r#"{"contentBlockIndex":0}"#),
            parsed("messageStop", r#"{"stopReason":"tool_use"}"#),
        ];
        let mut accumulator = ResponseAccumulator::new();
        for event in &events {
            accumulator.apply(event);
        }
        let response = accumulator.into_response().unwrap();
        assert_eq!(response.stop_reason, Some(StopReason::ToolUse));
        let tool_uses = response.tool_uses();
        assert_eq!(tool_uses.len(), 1);
        assert_eq!(tool_uses[0].tool_use_id, "tool-1");
        assert_eq!(tool_uses[0].name, "get_weather");
        assert_eq!(tool_uses[0].input["city"], "Norrtälje");
    }

    #[test]
    fn reasoning_deltas_build_a_reasoning_block() {
        let events = [
            parsed("messageStart", r#"{"role":"assistant"}"#),
            parsed(
                "contentBlockDelta",
                r#"{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"Thinking"}}}"#,
            ),
            parsed(
                "contentBlockDelta",
                r#"{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":" hard"}}}"#,
            ),
            parsed(
                "contentBlockDelta",
                r#"{"contentBlockIndex":0,"delta":{"reasoningContent":{"signature":"sig=="}}}"#,
            ),
            parsed(
                "contentBlockDelta",
                r#"{"contentBlockIndex":1,"delta":{"text":"Answer"}}"#,
            ),
        ];
        let mut accumulator = ResponseAccumulator::new();
        for event in &events {
            assert_eq!(
                event.reasoning_delta().is_some(),
                matches!(
                    event,
                    StreamEvent::ContentBlockDelta { delta, .. }
                        if delta.reasoning_content.as_ref().is_some_and(|r| r.text.is_some())
                ),
            );
            accumulator.apply(event);
        }
        let response = accumulator.into_response().unwrap();
        assert_eq!(response.reasoning().as_deref(), Some("Thinking hard"));
        assert_eq!(response.text(), "Answer");
    }

    #[test]
    fn empty_accumulator_yields_nothing() {
        assert!(ResponseAccumulator::new().into_response().is_none());
    }
}
