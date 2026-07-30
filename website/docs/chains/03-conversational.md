# Conversational Chains

Conversational chains — chains that maintain a running chat history across multiple exchanges — existed in the 0.13.x line of `llm-chain` but have **not yet been carried over to 0.14**.

## Doing multi-turn today

For many use cases a [sequential chain](./01-sequential-chains.md) covers the need: each step sees the previous step's output via the `{}` placeholder, so you can model a fixed series of exchanges without extra machinery.

For a free-form conversation, keep the history yourself and rebuild the step each turn — the drivers' prompt types convert straight from an iterator of `(Role, content)` pairs:

```rust
use llm_chain::{Parameters, traits::StepExt};
use llm_chain_openai::chat::{Executor, Model, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let exec = Executor::new_default();

    // The conversation so far, as (role, content) pairs
    let mut history: Vec<(Role, String)> = vec![(
        Role::System,
        "You are a helpful assistant.".to_string(),
    )];

    for user_message in ["Hi, my name is Emil!", "What was my name again?"] {
        history.push((Role::User, user_message.to_string()));

        // Build a step from the full history and run it
        let step = Step::new(
            Model::default(),
            history.iter().map(|(r, c)| (*r, c.as_str())),
        );
        let res = step.to_chain().run(Parameters::new(), &exec).await?;

        let reply = res
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or_default()
            .to_string();
        println!("assistant: {reply}");

        history.push((Role::Assistant, reply));
    }
    Ok(())
}
```

Trimming old messages to fit the context window is application-specific — token budgets differ per model — which is exactly why 0.14 has not yet frozen an API for it.

## Status

A first-class conversational chain (history management plus context-window trimming) is on the roadmap. If you need it, follow or open an issue on [GitHub](https://github.com/sobelio/llm-chain/issues) — input on the API shape is welcome.
