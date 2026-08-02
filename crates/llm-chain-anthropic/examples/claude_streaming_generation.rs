//! Streams a Claude response token by token, printing text as it arrives,
//! then reassembles the events into a complete response.
//!
//! Requires `ANTHROPIC_API_KEY` to be set.
//!
//! ```sh
//! cargo run --example claude_streaming_generation -p llm-chain-anthropic
//! ```
use std::io::Write as _;

use futures::StreamExt as _;
use llm_chain::Parameters;
use llm_chain::traits::{Step as _, StreamingExecutor as _};
use llm_chain_anthropic::messages::{Executor, Model, ResponseAccumulator, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let exec = Executor::new_default().expect("set ANTHROPIC_API_KEY");
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
    while let Some(event) = stream.next().await {
        let event = event.expect("stream failed mid-generation");
        if let Some(text) = Executor::text_delta(&event) {
            print!("{text}");
            std::io::stdout().flush().ok();
        }
        accumulator.apply(&event);
    }
    println!();

    let response = accumulator
        .into_response()
        .expect("stream ended before completion");
    println!(
        "[{} input + {} output tokens]",
        response.usage.input_tokens, response.usage.output_tokens
    );
}
