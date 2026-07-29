use llm_chain::{Parameters, traits::StepExt};
use llm_chain_llama::{Executor, Step};

/// Generates text from a local GGUF model.
///
/// Usage: `cargo run --example simple -- /path/to/model.gguf`
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args()
        .nth(1)
        .ok_or("usage: simple <path-to-model.gguf>")?;
    let exec = Executor::new(model_path)?;
    let chain = Step::new("The Colors of the Rainbow are (in order): ".into()).to_chain();
    let res = chain.run(Parameters::new(), &exec).await?;
    println!("{}", res);
    Ok(())
}
