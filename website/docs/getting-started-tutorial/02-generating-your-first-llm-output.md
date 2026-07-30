# Generating Your First LLM Output

:::tip

Having problems? Don't worry, reach out on [discord](https://discord.gg/kewN9Gtjt2) and we will help you out.

:::

First, we need to install `tokio` in our project. Since this is a tutorial we will install the full `tokio` package crate; in production, of course, we should be more selective with what features we install.

```bash
cargo add tokio --features full
```

Now, let's write a simple Rust program that generates an LLM output using llm-chain and the OpenAI driver:

```rust
use llm_chain::{Parameters, traits::StepExt};
use llm_chain_openai::chat::{Executor, Model, Role, Step};

// Declare an async main function
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new OpenAI executor; reads OPENAI_API_KEY from the environment
    let exec = Executor::new_default();
    // Create our step containing our prompt, and wrap it in a one-step chain
    let chain = Step::new(
        Model::default(),
        [
            (
                Role::System,
                "You are a robot assistant for making personalized greetings",
            ),
            (Role::User, "Make a personalized greeting for Joe"),
        ],
    )
    .to_chain();
    // ...and run it
    let res = chain.run(Parameters::new(), &exec).await?;
    println!(
        "{}",
        res.choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or_default()
    );
    Ok(())
}
```

## Understanding the response

`chain.run` returns the provider's typed response — for OpenAI that includes `choices` with the generated messages, plus metadata such as token `usage`. Errors are typed too: a failed run returns a `ChainError` telling you whether prompt formatting or the API call failed.

## Error Handling and Common Issues

One common issue you might encounter is forgetting to set the OpenAI API key. Make sure you have set the API key in your `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="YOUR_OPEN_AI_KEY" # TIP: It starts with sk-
```

If you don't want to set an environment variable, or you need multiple API keys, you can pass the key explicitly:

```rust
use llm_chain_openai::chat::Executor;

let exec = Executor::with_api_key("sk-proj-...");
```

There is also `Executor::with_base_url` for OpenAI-compatible servers (vLLM, OpenRouter, local proxies) and `Executor::azure(...)` for Azure OpenAI.

In the next tutorial, we'll cover adding parameters to customize the LLM prompt to create more complicated interactions.
