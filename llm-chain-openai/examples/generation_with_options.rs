use llm_chain::{Parameters, traits::StepExt};
use llm_chain_openai::chat::{Executor, Model, Options, ReasoningEffort, Role, Step, Verbosity};

/// Demonstrates per-step request options: sampling controls, token caps,
/// reasoning effort and verbosity.
#[tokio::main(flavor = "current_thread")]
async fn main() {
    let exec = Executor::new_default();
    let options = Options::new()
        .with_temperature(0.2)
        .with_max_completion_tokens(512)
        .with_reasoning_effort(ReasoningEffort::Low)
        .with_verbosity(Verbosity::Low);
    let chain = Step::new(
        Model::default(),
        [
            (Role::Developer, "You are a terse assistant."),
            (Role::User, "Explain what an LLM chain is in two sentences."),
        ],
    )
    .with_options(options)
    .to_chain();
    let res = chain.run(Parameters::new(), &exec).await.unwrap();
    println!(
        "{}",
        res.choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or_default()
    );
    if let Some(usage) = res.usage {
        println!(
            "({} prompt + {} completion = {} tokens)",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }
}
