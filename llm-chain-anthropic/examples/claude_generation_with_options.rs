//! Demonstrates per-step request options: sampling controls, token limits,
//! reasoning effort for the Claude 5 generation, and extended thinking for
//! models that support it.
use llm_chain::{Parameters, traits::StepExt};
use llm_chain_anthropic::messages::{Effort, Executor, Model, Options, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let exec = Executor::new_default().unwrap();

    // A deterministic-ish, tightly bounded answer from the balanced model,
    // with reasoning effort dialed down for latency.
    let quick = Step::new(Model::ClaudeSonnet5, [(Role::User, "Say hello to {}")]).with_options(
        Options::new()
            .with_temperature(0.0)
            .with_max_tokens(64)
            .with_effort(Effort::Low),
    );
    let res = quick
        .to_chain()
        .run(Parameters::new_with_text("Joe"), &exec)
        .await
        .unwrap();
    println!("quick: {}", res.text());
    println!(
        "usage: {} in / {} out",
        res.usage.input_tokens, res.usage.output_tokens
    );

    // Extended thinking on Haiku 4.5: the model may spend up to 2048 tokens
    // reasoning before it answers (thinking counts against max_tokens).
    let thoughtful = Step::new(
        Model::ClaudeHaiku45,
        [(Role::User, "What is 17 * 23? Explain briefly.")],
    )
    .with_options(
        Options::new()
            .with_max_tokens(4096)
            .with_thinking_budget(2048),
    );
    let res = thoughtful
        .to_chain()
        .run(Parameters::new(), &exec)
        .await
        .unwrap();
    println!("thoughtful: {}", res.text());
}
