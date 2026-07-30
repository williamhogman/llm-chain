use llm_chain::{Parameters, traits::StepExt};
use llm_chain_llama::{Executor, Step, new_instruct_template};

/// Runs an Alpaca-style instruction prompt against a local GGUF model.
///
/// Usage: `cargo run --example alpaca -- /path/to/model.gguf`
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args()
        .nth(1)
        .ok_or("usage: alpaca <path-to-model.gguf>")?;
    let exec = Executor::new(model_path)?;
    let template = new_instruct_template("Answer the following question: {}");
    let chain = Step::new(template).to_chain();
    let res = chain
        .run(
            Parameters::new_with_text("Who was the first man on the moon?"),
            &exec,
        )
        .await?;
    println!("{}", res);
    Ok(())
}
