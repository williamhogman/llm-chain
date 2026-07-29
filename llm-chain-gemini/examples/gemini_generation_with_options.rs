//! Demonstrates per-step request options: sampling controls, output caps and
//! thinking depth.
//!
//! Requires `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) to be set.
//!
//! ```sh
//! cargo run --example gemini_generation_with_options -p llm-chain-gemini
//! ```
use llm_chain::{Parameters, traits::StepExt};
use llm_chain_gemini::generate_content::{Executor, Model, Options, Role, Step, ThinkingLevel};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let exec = Executor::new_default().expect("set GEMINI_API_KEY or GOOGLE_API_KEY");

    // Low temperature and a low thinking level for a fast, focused answer.
    let options = Options::new()
        .with_temperature(0.2)
        .with_max_output_tokens(512)
        .with_thinking_level(ThinkingLevel::Low);

    let chain = Step::new(
        Model::default(),
        [(Role::User, "In two sentences: why is the sky blue?")],
    )
    .with_system("You are a concise physics teacher")
    .with_options(options)
    .to_chain();

    let res = chain
        .run(Parameters::new(), &exec)
        .await
        .expect("chain failed");
    println!("{}", res.text());
    println!(
        "-- {} prompt + {} output tokens ({} thinking)",
        res.usage_metadata.prompt_token_count,
        res.usage_metadata.candidates_token_count,
        res.usage_metadata.thoughts_token_count,
    );
}
