//! Streams a Bedrock Converse response event by event, printing text as it
//! arrives, then reassembles the events into a complete response.
//!
//! Requires `AWS_BEARER_TOKEN_BEDROCK` to be set (and optionally
//! `AWS_REGION`).
//!
//! ```sh
//! cargo run --example bedrock_streaming_generation -p llm-chain-bedrock
//! ```
use std::io::Write as _;

use futures::StreamExt as _;
use llm_chain::Parameters;
use llm_chain::traits::{Step as _, StreamingExecutor as _};
use llm_chain_bedrock::converse::{Executor, Model, ResponseAccumulator, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let exec = Executor::new_default().expect("set AWS_BEARER_TOKEN_BEDROCK");
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
        if let Some(text) = event.text_delta() {
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
