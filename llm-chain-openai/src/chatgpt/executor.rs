use std::sync::Arc;

use async_openai::config::OpenAIConfig;
use async_openai::error::OpenAIError;
use async_openai::types::chat::CreateChatCompletionResponse;
use llm_chain::Parameters;
use llm_chain::traits;

use super::step::Step;

/// The executor for OpenAI chat models. This executor uses the `async_openai` crate to communicate with the OpenAI API.
///
/// It authenticates using the `OPENAI_API_KEY` environment variable by default; use [`Executor::new`] with a configured client for anything custom (different base URLs, organizations, Azure, etc.).
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
}

fn first_content(output: &CreateChatCompletionResponse) -> &str {
    output
        .choices
        .first()
        .and_then(|choice| choice.message.content.as_deref())
        .unwrap_or_default()
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

    /// Combines two outputs into a single output by joining their contents with a newline.
    fn combine_outputs(output: &Self::Output, other: &Self::Output) -> Self::Output {
        let mut combined = output.clone();
        let joined = [first_content(output), first_content(other)].join("\n");
        if let Some(choice) = combined.choices.first_mut() {
            choice.message.content = Some(joined);
        }
        combined
    }
}
