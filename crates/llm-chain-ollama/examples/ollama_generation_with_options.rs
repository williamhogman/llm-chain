//! Shows per-step request options: sampling controls, token limits, thinking
//! and generation timings.
use llm_chain::{Parameters, traits::StepExt};
use llm_chain_ollama::chat::{Executor, Model, Options, Role, Step, Think};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let exec = Executor::new_default();
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
            .with_num_predict(512)
            .with_think(Think::Enabled)
            .with_keep_alive("5m"),
    );
    let res = step.to_chain().run(Parameters::new(), &exec).await.unwrap();
    if let Some(thinking) = res.thinking() {
        println!("--- thinking ---\n{thinking}\n--- answer ---");
    }
    println!("{}", res.text());
    if let Some(rate) = res.eval_rate() {
        println!("({rate:.1} tokens/s)");
    }
}
