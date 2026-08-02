//! Streams a chat completion token by token, printing text as it arrives,
//! then reassembles the chunks into a complete response.
//!
//! Requires `OPENAI_API_KEY` to be set.
//!
//! ```sh
//! cargo run --example streaming_generation -p llm-chain-openai
//! ```
use std::io::Write as _;

use futures::StreamExt as _;
use llm_chain::Parameters;
use llm_chain::traits::{Step as _, StreamingExecutor as _};
use llm_chain_openai::chat::{Executor, Model, ResponseAccumulator, Role, Step};

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
        if let Some(text) = chunk
            .choices
            .first()
            .and_then(|choice| choice.delta.content.as_deref())
        {
            print!("{text}");
            std::io::stdout().flush().ok();
        }
        accumulator.apply(&chunk);
    }
    println!();

    let response = accumulator
        .into_response()
        .expect("stream ended before completion");
    if let Some(usage) = response.usage {
        println!(
            "[{} prompt + {} completion tokens]",
            usage.prompt_tokens, usage.completion_tokens
        );
    }
}
