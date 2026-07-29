use std::sync::Arc;

use async_openai::config::OpenAIConfig;
use async_openai::error::OpenAIError;
use async_openai::types::chat::{CompletionUsage, CreateChatCompletionResponse};
use llm_chain::Parameters;
use llm_chain::traits;

use super::step::Step;

/// The executor for OpenAI chat models. This executor uses the `async_openai` crate to communicate with the OpenAI API.
///
/// It authenticates using the `OPENAI_API_KEY` environment variable by default; use
/// [`Executor::with_api_key`] to pass a key explicitly, or [`Executor::new`] with a configured
/// client for anything custom (different base URLs, organizations, Azure, etc.).
#[derive(Clone)]
pub struct Executor {
    client: Arc<async_openai::Client<OpenAIConfig>>,
}

impl Executor {
    /// Creates a new executor with the given client.
    pub fn new(client: async_openai::Client<OpenAIConfig>) -> Self {
        let client = Arc::new(client);
        Self { client }
    }
    /// Creates a new executor with the default client, which uses the `OPENAI_API_KEY` environment variable.
    pub fn new_default() -> Self {
        Self::new(async_openai::Client::new())
    }
    /// Creates a new executor authenticating with the given API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self::new(async_openai::Client::with_config(
            OpenAIConfig::new().with_api_key(api_key),
        ))
    }
}

fn first_content(output: &CreateChatCompletionResponse) -> &str {
    output
        .choices
        .first()
        .and_then(|choice| choice.message.content.as_deref())
        .unwrap_or_default()
}

/// Sums token usage across two responses, so combined outputs keep honest accounting.
fn combine_usage(
    a: Option<&CompletionUsage>,
    b: Option<&CompletionUsage>,
) -> Option<CompletionUsage> {
    match (a, b) {
        (Some(a), Some(b)) => Some(CompletionUsage {
            prompt_tokens: a.prompt_tokens + b.prompt_tokens,
            completion_tokens: a.completion_tokens + b.completion_tokens,
            total_tokens: a.total_tokens + b.total_tokens,
            // Detailed breakdowns are not meaningfully additive across requests.
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }),
        (a, b) => a.or(b).cloned(),
    }
}

impl traits::Executor for Executor {
    type Step = Step;
    type Output = CreateChatCompletionResponse;
    type Error = OpenAIError;

    /// Executes the chat completion request and returns the response.
    async fn execute(
        &self,
        input: <<Executor as traits::Executor>::Step as traits::Step>::Output,
    ) -> Result<Self::Output, Self::Error> {
        self.client.chat().create(input).await
    }

    /// Applies the first choice's content to the parameters as the default `text`.
    fn apply_output_to_parameters(parameters: Parameters, output: &Self::Output) -> Parameters {
        parameters.with_text(first_content(output))
    }

    /// Combines two outputs into a single output by joining their contents with a newline
    /// and summing their token usage.
    fn combine_outputs(output: &Self::Output, other: &Self::Output) -> Self::Output {
        let mut combined = output.clone();
        let joined = [first_content(output), first_content(other)].join("\n");
        if let Some(choice) = combined.choices.first_mut() {
            choice.message.content = Some(joined);
        }
        combined.usage = combine_usage(output.usage.as_ref(), other.usage.as_ref());
        combined
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_chain::traits::Executor as _;

    fn response(content: &str, usage_tokens: Option<(u32, u32)>) -> CreateChatCompletionResponse {
        let usage = usage_tokens.map(|(prompt, completion)| {
            serde_json::json!({
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            })
        });
        serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5.6-terra",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": "stop",
            }],
            "usage": usage,
        }))
        .expect("valid response fixture")
    }

    #[test]
    fn combine_outputs_joins_content_and_sums_usage() {
        let a = response("first", Some((10, 5)));
        let b = response("second", Some((20, 7)));
        let combined = Executor::combine_outputs(&a, &b);
        assert_eq!(first_content(&combined), "first\nsecond");
        let usage = combined.usage.expect("usage present");
        assert_eq!(usage.prompt_tokens, 30);
        assert_eq!(usage.completion_tokens, 12);
        assert_eq!(usage.total_tokens, 42);
    }

    #[test]
    fn combine_outputs_keeps_the_only_available_usage() {
        let a = response("first", None);
        let b = response("second", Some((3, 4)));
        let combined = Executor::combine_outputs(&a, &b);
        assert_eq!(combined.usage.expect("usage present").total_tokens, 7);
    }

    #[test]
    fn apply_output_sets_text_parameter() {
        let output = response("hello", None);
        let parameters = Executor::apply_output_to_parameters(Parameters::new(), &output);
        assert_eq!(parameters.get_text(), Some("hello"));
    }
}
