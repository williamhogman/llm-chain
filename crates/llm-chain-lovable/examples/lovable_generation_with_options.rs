//! Shows per-step request options: sampling controls, token limits,
//! reasoning and token usage, plus the run id that correlates the request
//! with Lovable AI usage logs.
use llm_chain::{Parameters, traits::StepExt};
use llm_chain_lovable::chat::{Executor, Model, Options, Reasoning, ReasoningEffort, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let exec = Executor::new_default().expect("set LOVABLE_API_KEY");
    let step = Step::new(
        Model::default(),
        [
            (Role::System, "You are a terse assistant. Answer briefly."),
            (Role::User, "Why is the sky blue?"),
        ],
    )
    .with_options(
        Options::new()
            .with_temperature(0.2)
            .with_max_tokens(512)
            .with_reasoning(Reasoning::effort(ReasoningEffort::Low)),
    );
    let res = step.to_chain().run(Parameters::new(), &exec).await.unwrap();
    if let Some(reasoning) = res.reasoning() {
        println!("--- reasoning ---\n{reasoning}\n--- answer ---");
    }
    println!("{}", res.text());
    if let Some(usage) = res.usage {
        println!(
            "[{} prompt + {} completion tokens]",
            usage.prompt_tokens, usage.completion_tokens
        );
    }
    if let Some(run_id) = &res.run_id {
        println!("(run id: {run_id})");
    }
}
