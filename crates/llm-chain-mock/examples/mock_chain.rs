//! Run a small two-step chain against the mock executor — no API key, no
//! network, fully deterministic.
//!
//! ```sh
//! cargo run --example mock_chain -p llm-chain-mock
//! ```

use llm_chain::Parameters;
use llm_chain::chains::sequential::Chain;
use llm_chain_mock::{Executor, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Scripted mode: the "model" answers from a fixed script.
    let chain: Chain<Step> = [
        Step::new("Give me three facts about {text}."),
        Step::new("Turn these facts into a haiku:\n{text}"),
    ]
    .into_iter()
    .collect();

    let executor = Executor::with_responses([
        "1. Crabs are crustaceans. 2. Crabs walk sideways. 3. Crabs molt.",
        "Sideways they wander,\nshedding shells beneath the moon —\nquiet crustaceans.",
    ]);

    let output = chain
        .run(Parameters::new_with_text("crabs"), &executor)
        .await?;
    println!("Final output:\n{output}\n");

    println!("Prompts the mock model saw:");
    for (i, call) in executor.calls().iter().enumerate() {
        println!("  [{i}] {call}");
    }
    Ok(())
}
