//! A simple example generating a greeting with Gemini.
//!
//! Requires `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) to be set.
//!
//! ```sh
//! cargo run --example gemini_simple_generator -p llm-chain-gemini
//! ```
use llm_chain::{Parameters, traits::StepExt};
use llm_chain_gemini::generate_content::{Executor, Model, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let exec = Executor::new_default().expect("set GEMINI_API_KEY or GOOGLE_API_KEY");
    let chain = Step::new(
        Model::default(),
        [(Role::User, "Make a personalized greeting for Joe")],
    )
    .with_system("You are a bot for making personalized greetings")
    .to_chain();
    let res = chain
        .run(Parameters::new(), &exec)
        .await
        .expect("chain failed");
    println!("{}", res.text());
}
