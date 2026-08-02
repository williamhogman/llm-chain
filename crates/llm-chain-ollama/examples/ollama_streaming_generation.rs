//! Streams an Ollama response chunk by chunk, printing text as it arrives,
//! then reassembles the chunks into a complete response.
//!
//! Talks to the local Ollama server (respects `OLLAMA_HOST`).
//! Pull the model first: `ollama pull qwen3`.
//!
//! ```sh
//! cargo run --example ollama_streaming_generation -p llm-chain-ollama
//! ```
use std::io::Write as _;

use futures::StreamExt as _;
use llm_chain::Parameters;
use llm_chain::traits::{Step as _, StreamingExecutor as _};
use llm_chain_ollama::chat::{Executor, Model, ResponseAccumulator, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let exec = Executor::new_default();
    let step = Step::new(
        Model::default(),
        [(Role::User, "Tell a very short story about {{topic}}")],
    );
    let request = step
        .format(&Parameters::new().with("topic", "a crab learning Rust"))
        .expect("formatting failed");

    let mut stream = exec
        .execute_stream(request)
        .await
        .expect("request failed before any output");

    let mut accumulator = ResponseAccumulator::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("stream failed mid-generation");
        if let Some(text) = Executor::text_delta(&chunk) {
            print!("{text}");
            std::io::stdout().flush().ok();
        }
        accumulator.apply(&chunk);
    }
    println!();

    let response = accumulator
        .into_response()
        .expect("stream ended before completion");
    if let (Some(prompt), Some(eval)) = (response.prompt_eval_count, response.eval_count) {
        println!("[{prompt} prompt + {eval} generated tokens]");
    }
}
