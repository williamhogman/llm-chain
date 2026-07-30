//! Demonstrates per-step request options: shared inference parameters, plus
//! model-family-specific fields (Claude extended thinking) passed through
//! verbatim, and the usage/latency accounting Bedrock returns.
use llm_chain::{Parameters, traits::StepExt};
use llm_chain_bedrock::converse::{Executor, Options, Role, Step, models};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let exec = Executor::new_default().unwrap();

    // A deterministic-ish, tightly bounded answer from the cheapest Nova model.
    let quick = Step::new(models::NOVA_MICRO, [(Role::User, "Say hello to {}")]).with_options(
        Options::new()
            .with_temperature(0.0)
            .with_max_tokens(64)
            .with_stop_sequences(["\n\n"]),
    );
    let res = quick
        .to_chain()
        .run(Parameters::new_with_text("Joe"), &exec)
        .await
        .unwrap();
    println!("quick: {}", res.text());
    println!(
        "usage: {} in / {} out, latency: {} ms",
        res.usage.input_tokens,
        res.usage.output_tokens,
        res.metrics.map(|metrics| metrics.latency_ms).unwrap_or(0),
    );

    // Claude extended thinking, passed through verbatim as a family-specific
    // field the shared inference config does not cover.
    let thoughtful = Step::new(
        models::CLAUDE_HAIKU_4_5,
        [(Role::User, "What is 17 * 23? Explain briefly.")],
    )
    .with_options(
        Options::new()
            .with_max_tokens(4096)
            .with_additional_model_request_fields(serde_json::json!({
                "thinking": {"type": "enabled", "budget_tokens": 2048}
            })),
    );
    let res = thoughtful
        .to_chain()
        .run(Parameters::new(), &exec)
        .await
        .unwrap();
    if let Some(reasoning) = res.reasoning() {
        println!("reasoning: {reasoning}");
    }
    println!("thoughtful: {}", res.text());
}
