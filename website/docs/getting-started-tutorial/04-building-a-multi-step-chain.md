# Creating Your First Sequential Chain

:::tip

Having problems? Don't worry, reach out on [discord](https://discord.gg/kewN9Gtjt2) and we will help you out.

:::

Sequential chains in llm-chain allow you to execute a series of steps, with the output of each step feeding into the next one. This tutorial will guide you through creating a sequential chain, extending it with more steps, and provide some best practices and tips.

Here's a Rust program that demonstrates how to create a sequential chain:

```rust
use llm_chain::chains::sequential::Chain;
use llm_chain_openai::chat::{Executor, Model, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new OpenAI executor with the default settings
    let exec = Executor::new_default();

    // Create a chain of steps with two prompts
    let chain = Chain::new(vec![
        // First step: make a personalized birthday email
        Step::new(
            Model::default(),
            [
                (
                    Role::System,
                    "You are a bot for making personalized greetings",
                ),
                (
                    Role::User,
                    "Make personalized birthday e-mail to the whole company for {name} who has their birthday on {date}. Include their name",
                ),
            ],
        ),
        // Second step: summarize the email into a tweet. Importantly, `{}`
        // becomes the result of the previous step.
        Step::new(
            Model::default(),
            [
                (
                    Role::System,
                    "You are an assistant for managing social media accounts for a company",
                ),
                (
                    Role::User,
                    "Summarize this email into a tweet to be sent by the company, use emoji if you can. \n--\n{}",
                ),
            ],
        ),
    ]);

    // Run the chain with the provided parameters
    let res = chain
        .run(
            // Create a Parameters object with key-value pairs for the placeholders
            vec![("name", "Emil"), ("date", "February 30th 2023")].into(),
            &exec,
        )
        .await?;

    // Print the result to the console
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

1. We start by importing the necessary modules from `llm_chain` and `llm_chain_openai`.
2. We create a new OpenAI executor.
3. We build a `Chain` from a vector of `Step`s. Each step pairs a model with a prompt template. The output of the first step is fed into the second step's `{}` placeholder (the default `text` parameter).
4. We run the chain with a `Parameters` object providing values for the `{name}` and `{date}` placeholders, and print the final output.

You can also build chains incrementally — start with `Chain::of_one(step)` and add steps with `chain.push(step)`, or collect steps straight from an iterator, since `Chain` implements `FromIterator` and `Extend`.

With sequential chains, you can create complex pipelines where each step builds on the results of the previous one. In the next tutorial, we'll process large documents with map-reduce chains.
