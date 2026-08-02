//! Streaming support for the chat API.
//!
//! With [`StreamingExecutor::execute_stream`](llm_chain::traits::StreamingExecutor::execute_stream)
//! the server delivers the response as newline-delimited JSON: a series of
//! partial [`ChatResponse`] chunks whose `message` carries the newly generated
//! text (or thinking, or tool calls), followed by a final chunk with `done:
//! true` and the generation timings.
//!
//! Print live output with the chunk's [`text`](ChatResponse::text); fold the
//! full chunk sequence back into one complete response with
//! [`ResponseAccumulator`] when the final response is also wanted.

use super::types::ChatResponse;

/// Folds a stream of partial [`ChatResponse`] chunks back into one complete
/// response.
///
/// Feed every chunk to [`apply`](ResponseAccumulator::apply); once the stream
/// ends (or [`is_complete`](ResponseAccumulator::is_complete) reports the
/// `done` chunk was seen), [`into_response`](ResponseAccumulator::into_response)
/// yields a response equal to what [`Executor::execute`](llm_chain::traits::Executor::execute)
/// would have returned — text and thinking concatenated, tool calls collected,
/// and the final chunk's done reason and timings carried over, so tool-calling
/// conversations can be continued from a streamed turn.
///
/// # Examples
///
/// ```no_run
/// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// use futures::StreamExt as _;
/// use llm_chain::traits::StreamingExecutor as _;
/// use llm_chain_ollama::chat::{Executor, ResponseAccumulator};
///
/// # let executor = Executor::new_default();
/// # let request = todo!();
/// let mut stream = executor.execute_stream(request).await?;
/// let mut accumulator = ResponseAccumulator::new();
/// while let Some(chunk) = stream.next().await {
///     let chunk = chunk?;
///     print!("{}", chunk.message.content);
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
    pub fn apply(&mut self, chunk: &ChatResponse) {
        let Some(response) = &mut self.response else {
            self.response = Some(chunk.clone());
            return;
        };
        response.message.content.push_str(&chunk.message.content);
        if let Some(thinking) = &chunk.message.thinking {
            match &mut response.message.thinking {
                Some(existing) => existing.push_str(thinking),
                None => response.message.thinking = Some(thinking.clone()),
            }
        }
        if let Some(tool_calls) = &chunk.message.tool_calls {
            response
                .message
                .tool_calls
                .get_or_insert_with(Vec::new)
                .extend(tool_calls.iter().cloned());
        }
        if !chunk.model.is_empty() {
            response.model = chunk.model.clone();
        }
        if !chunk.created_at.is_empty() {
            response.created_at = chunk.created_at.clone();
        }
        response.done = chunk.done;
        if chunk.done_reason.is_some() {
            response.done_reason = chunk.done_reason;
        }
        // Timings and token counts arrive on the final chunk.
        if chunk.total_duration.is_some() {
            response.total_duration = chunk.total_duration;
        }
        if chunk.load_duration.is_some() {
            response.load_duration = chunk.load_duration;
        }
        if chunk.prompt_eval_count.is_some() {
            response.prompt_eval_count = chunk.prompt_eval_count;
        }
        if chunk.prompt_eval_duration.is_some() {
            response.prompt_eval_duration = chunk.prompt_eval_duration;
        }
        if chunk.eval_count.is_some() {
            response.eval_count = chunk.eval_count;
        }
        if chunk.eval_duration.is_some() {
            response.eval_duration = chunk.eval_duration;
        }
    }

    /// Whether a chunk with `done: true` has been seen.
    pub fn is_complete(&self) -> bool {
        self.response.as_ref().is_some_and(|response| response.done)
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
    use super::super::types::{DoneReason, FunctionCall, Message, Role, ToolCall};
    use super::*;

    fn chunk(content: &str) -> ChatResponse {
        ChatResponse {
            model: "qwen3".to_string(),
            created_at: String::new(),
            message: Message::new(Role::Assistant, content),
            done: false,
            done_reason: None,
            total_duration: None,
            load_duration: None,
            prompt_eval_count: None,
            prompt_eval_duration: None,
            eval_count: None,
            eval_duration: None,
        }
    }

    #[test]
    fn text_chunks_concatenate_and_final_metadata_wins() {
        let mut accumulator = ResponseAccumulator::new();
        accumulator.apply(&chunk("Hello"));
        assert!(!accumulator.is_complete());
        accumulator.apply(&chunk(", world"));

        let mut last = chunk("!");
        last.done = true;
        last.done_reason = Some(DoneReason::Stop);
        last.eval_count = Some(9);
        last.total_duration = Some(1_000);
        accumulator.apply(&last);

        assert!(accumulator.is_complete());
        let response = accumulator.into_response().unwrap();
        assert_eq!(response.text(), "Hello, world!");
        assert!(response.done);
        assert_eq!(response.done_reason, Some(DoneReason::Stop));
        assert_eq!(response.eval_count, Some(9));
        assert_eq!(response.total_duration, Some(1_000));
    }

    #[test]
    fn thinking_concatenates_separately_from_the_answer() {
        let mut thinking_chunk = chunk("");
        thinking_chunk.message.thinking = Some("Let me ".to_string());
        let mut thinking_chunk2 = chunk("");
        thinking_chunk2.message.thinking = Some("think".to_string());

        let mut accumulator = ResponseAccumulator::new();
        accumulator.apply(&thinking_chunk);
        accumulator.apply(&thinking_chunk2);
        accumulator.apply(&chunk("Answer"));

        let response = accumulator.into_response().unwrap();
        assert_eq!(response.thinking(), Some("Let me think"));
        assert_eq!(response.text(), "Answer");
    }

    #[test]
    fn tool_calls_are_collected_across_chunks() {
        let call = |name: &str| ToolCall {
            function: FunctionCall {
                name: name.to_string(),
                arguments: serde_json::json!({}),
            },
        };
        let mut first = chunk("");
        first.message.tool_calls = Some(vec![call("get_weather")]);
        let mut second = chunk("");
        second.message.tool_calls = Some(vec![call("get_time")]);

        let mut accumulator = ResponseAccumulator::new();
        accumulator.apply(&first);
        accumulator.apply(&second);

        let response = accumulator.into_response().unwrap();
        let names: Vec<_> = response
            .tool_calls()
            .iter()
            .map(|call| call.function.name.as_str())
            .collect();
        assert_eq!(names, ["get_weather", "get_time"]);
    }

    #[test]
    fn empty_accumulator_yields_nothing() {
        assert!(ResponseAccumulator::new().into_response().is_none());
        assert!(!ResponseAccumulator::new().is_complete());
    }
}
