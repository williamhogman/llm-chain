//! Streams a Gemini response chunk by chunk, printing text as it arrives,
//! then reassembles the chunks into a complete response.
//!
//! Requires `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) to be set.
//!
//! ```sh
//! cargo run --example gemini_streaming_generation -p llm-chain-gemini
//! ```
use std::io::Write as _;

use futures::StreamExt as _;
use llm_chain::Parameters;
use llm_chain::traits::{Step as _, StreamingExecutor as _};
use llm_chain_gemini::generate_content::{Executor, Model, ResponseAccumulator, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let exec = Executor::new_default().expect("set GEMINI_API_KEY or GOOGLE_API_KEY");
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
    if let Some(usage) = response.usage_metadata {
        println!(
            "[{} prompt + {} candidate tokens]",
            usage.prompt_token_count, usage.candidates_token_count
        );
    }
}
