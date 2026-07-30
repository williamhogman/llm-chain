//! A two-step sequential chain: write a rough draft, then polish it.
//!
//! Requires `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) to be set.
//!
//! ```sh
//! cargo run --example gemini_sequential_generation -p llm-chain-gemini
//! ```
use llm_chain::{Parameters, chains::sequential::Chain};
use llm_chain_gemini::generate_content::{Executor, Model, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let exec = Executor::new_default().expect("set GEMINI_API_KEY or GOOGLE_API_KEY");
    let chain = Chain::new(vec![
        Step::new(
            Model::default(),
            [(Role::User, "Write a rough tagline for a {} shop")],
        )
        .with_system("You are a punchy copywriter"),
        Step::new(
            Model::default(),
            [(
                Role::User,
                "Polish this tagline and output only the final version: {}",
            )],
        ),
    ]);
    let res = chain
        .run(Parameters::new_with_text("sourdough bakery"), &exec)
        .await
        .expect("chain failed");
    println!("{}", res.text());
}
