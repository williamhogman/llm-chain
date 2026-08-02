//! Streaming support for `streamGenerateContent`.
//!
//! With [`StreamingExecutor::execute_stream`](llm_chain::traits::StreamingExecutor::execute_stream)
//! the API delivers the response as a series of partial
//! [`GenerateContentResponse`] chunks: each carries the newly generated parts
//! of the first candidate, and the final chunk carries the finish reason and
//! cumulative [`UsageMetadata`](super::UsageMetadata).
//!
//! Print live output with the chunk's [`text`](GenerateContentResponse::text);
//! fold the full chunk sequence back into one complete response with
//! [`ResponseAccumulator`] when the final response is also wanted.

use super::types::{Content, GenerateContentResponse, Part};

/// Folds a stream of partial [`GenerateContentResponse`] chunks back into one
/// complete response.
///
/// Feed every chunk to [`apply`](ResponseAccumulator::apply); once the stream
/// ends, [`into_response`](ResponseAccumulator::into_response) yields a
/// response equal to what [`Executor::execute`](llm_chain::traits::Executor::execute)
/// would have returned — consecutive text parts merged, thought parts kept
/// separate from answer parts, and function calls and thought signatures
/// preserved, so function-calling conversations can be continued from a
/// streamed turn.
///
/// # Examples
///
/// ```no_run
/// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// use futures::StreamExt as _;
/// use llm_chain::traits::StreamingExecutor as _;
/// use llm_chain_gemini::generate_content::{Executor, ResponseAccumulator};
///
/// # let executor = Executor::with_api_key("AIza...");
/// # let request = todo!();
/// let mut stream = executor.execute_stream(request).await?;
/// let mut accumulator = ResponseAccumulator::new();
/// while let Some(chunk) = stream.next().await {
///     let chunk = chunk?;
///     print!("{}", chunk.text());
///     accumulator.apply(&chunk);
/// }
/// let response = accumulator.into_response().expect("stream produced chunks");
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct ResponseAccumulator {
    response: Option<GenerateContentResponse>,
}

impl ResponseAccumulator {
    /// Creates an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies one streamed chunk to the response under construction.
    pub fn apply(&mut self, chunk: &GenerateContentResponse) {
        let Some(response) = &mut self.response else {
            self.response = Some(chunk.clone());
            return;
        };
        // Usage metadata on chunks is cumulative, so the latest chunk wins.
        response.usage_metadata = chunk.usage_metadata;
        if chunk.model_version.is_some() {
            response.model_version = chunk.model_version.clone();
        }
        if chunk.response_id.is_some() {
            response.response_id = chunk.response_id.clone();
        }
        if chunk.prompt_feedback.is_some() {
            response.prompt_feedback = chunk.prompt_feedback.clone();
        }

        let Some(new_candidate) = chunk.candidates.first() else {
            return;
        };
        if response.candidates.is_empty() {
            response.candidates.push(new_candidate.clone());
            return;
        }
        let candidate = &mut response.candidates[0];
        if new_candidate.finish_reason.is_some() {
            candidate.finish_reason = new_candidate.finish_reason;
        }
        let Some(new_content) = new_candidate.content.as_ref() else {
            return;
        };
        let content = candidate.content.get_or_insert_with(|| Content {
            role: new_content.role,
            parts: Vec::new(),
        });
        for part in &new_content.parts {
            if let Some(last) = content.parts.last_mut()
                && can_merge(last, part)
            {
                last.text.push_str(&part.text);
                if part.thought_signature.is_some() {
                    last.thought_signature = part.thought_signature.clone();
                }
                continue;
            }
            content.parts.push(part.clone());
        }
    }

    /// The response assembled so far, if any chunk has been applied.
    pub fn response(&self) -> Option<&GenerateContentResponse> {
        self.response.as_ref()
    }

    /// Consumes the accumulator, yielding the assembled response.
    pub fn into_response(self) -> Option<GenerateContentResponse> {
        self.response
    }
}

/// Whether `next` extends the text of `last` rather than starting a new part.
fn can_merge(last: &Part, next: &Part) -> bool {
    !last.text.is_empty()
        && !next.text.is_empty()
        && last.thought == next.thought
        && last.function_call.is_none()
        && next.function_call.is_none()
        && last.function_response.is_none()
        && next.function_response.is_none()
}

#[cfg(test)]
mod tests {
    use super::super::types::{Candidate, FinishReason, FunctionCall, Role, UsageMetadata};
    use super::*;

    fn chunk(parts: Vec<Part>) -> GenerateContentResponse {
        GenerateContentResponse {
            candidates: vec![Candidate {
                content: Some(Content {
                    role: Some(Role::Model),
                    parts,
                }),
                finish_reason: None,
            }],
            ..GenerateContentResponse::default()
        }
    }

    #[test]
    fn text_chunks_merge_into_one_part() {
        let mut accumulator = ResponseAccumulator::new();
        accumulator.apply(&chunk(vec![Part::text("Hello")]));
        accumulator.apply(&chunk(vec![Part::text(", world")]));

        let mut last = chunk(vec![Part::text("!")]);
        last.candidates[0].finish_reason = Some(FinishReason::Stop);
        last.usage_metadata = UsageMetadata {
            prompt_token_count: 4,
            candidates_token_count: 3,
            total_token_count: 7,
            ..UsageMetadata::default()
        };
        accumulator.apply(&last);

        let response = accumulator.into_response().unwrap();
        assert_eq!(response.text(), "Hello, world!");
        assert_eq!(
            response.candidates[0].content.as_ref().unwrap().parts.len(),
            1
        );
        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
        assert_eq!(response.usage_metadata.total_token_count, 7);
    }

    #[test]
    fn thoughts_stay_separate_from_the_answer() {
        let thought = Part {
            text: "thinking...".to_string(),
            thought: true,
            ..Part::default()
        };
        let mut accumulator = ResponseAccumulator::new();
        accumulator.apply(&chunk(vec![thought.clone()]));
        accumulator.apply(&chunk(vec![Part::text("Answer")]));
        accumulator.apply(&chunk(vec![Part::text(" text")]));

        let response = accumulator.into_response().unwrap();
        assert_eq!(response.text(), "Answer text");
        let parts = &response.candidates[0].content.as_ref().unwrap().parts;
        assert_eq!(parts.len(), 2);
        assert!(parts[0].thought);
        assert_eq!(parts[1].text, "Answer text");
    }

    #[test]
    fn function_calls_and_signatures_are_preserved() {
        let call = Part {
            function_call: Some(FunctionCall {
                name: "get_weather".to_string(),
                args: serde_json::json!({"city": "Stockholm"}),
            }),
            ..Part::default()
        };
        let signed = Part {
            text: "!".to_string(),
            thought_signature: Some("sig".to_string()),
            ..Part::default()
        };
        let mut accumulator = ResponseAccumulator::new();
        accumulator.apply(&chunk(vec![Part::text("Checking")]));
        accumulator.apply(&chunk(vec![signed]));
        accumulator.apply(&chunk(vec![call]));

        let response = accumulator.into_response().unwrap();
        let parts = &response.candidates[0].content.as_ref().unwrap().parts;
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].text, "Checking!");
        assert_eq!(parts[0].thought_signature.as_deref(), Some("sig"));
        let calls: Vec<_> = response.function_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn empty_accumulator_yields_nothing() {
        assert!(ResponseAccumulator::new().into_response().is_none());
    }
}
