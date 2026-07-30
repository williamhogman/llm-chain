# Using Prompt Templates and Parameters

:::tip

Having problems? Don't worry, reach out on [discord](https://discord.gg/kewN9Gtjt2) and we will help you out.

:::

In this part of the tutorial series, we'll explore how to use prompt templates and parameters with llm-chain. Prompt templates allow you to create dynamic prompts, and parameters are the text strings you put into your templates.

Templates use `{}` for the default parameter (named `text`) and `{name}` for named parameters. Literal braces are escaped by doubling: `{{` renders `{`.

Here's a simple Rust program demonstrating how to use prompt templates and parameters:

```rust
use llm_chain::{Parameters, traits::StepExt};
use llm_chain_openai::chat::{Executor, Model, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new OpenAI executor
    let exec = Executor::new_default();
    // Create our step containing our prompt template
    let chain = Step::new(
        Model::default(),
        [
            (
                Role::System,
                "You are a bot for making personalized greetings",
            ),
            (
                Role::User,
                "Make a personalized greeting tweet for {}", // {} is the default parameter
            ),
        ],
    )
    .to_chain();

    // A greeting for Emil!
    let res = chain.run(Parameters::new_with_text("Emil"), &exec).await?;
    println!(
        "{}",
        res.choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or_default()
    );

    // A greeting for you — named parameters work too, with `{name}` syntax
    let res = chain
        .run(vec![("text", "Your Name Here")].into(), &exec)
        .await?;
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

Let's break down the different parts of the code:

1. We start by importing the necessary libraries, including the traits and structs required for our program.
2. The main async function is defined, using Tokio as the runtime.
3. We create a new `Executor` with the default settings.
4. A `Step` is created containing our prompt template with a placeholder (`{}`) that will be replaced with a specific value later, and we wrap it into a one-step chain with `to_chain()`.
5. We run the chain with `Parameters::new_with_text("Emil")`, which binds the default parameter, and print the response.
6. We run the chain again, this time building the `Parameters` from a list of key-value pairs.

Formatting is fallible: if a template references a parameter that isn't provided, the chain returns a `ChainError::Format` instead of panicking — so typos in template names are caught cleanly at runtime.

In the next tutorial, we will combine multiple LLM invocations to solve more complicated problems.
