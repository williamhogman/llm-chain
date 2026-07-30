# Sequential Chains

Sequential chains are a convenient way to apply large language models (LLMs) to a sequence of tasks. They connect multiple steps together, where the output of the first step becomes the input of the second step, and so on. This method allows for straightforward processing of information, where each step builds upon the results of the previous one.

In this guide, we'll explain how to create and execute a sequential chain using an example. The example demonstrates a two-step process, where the first step generates a personalized birthday email, and the second step summarizes the email into a tweet.

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
                (Role::System, "You are a bot for making personalized greetings"),
                (Role::User, "Make personalized birthday e-mail to the whole company for {name} who has their birthday on {date}. Include their name"),
            ],
        ),
        // Second step: summarize the email into a tweet. Importantly, the
        // `{}` placeholder becomes the result of the previous prompt.
        Step::new(
            Model::default(),
            [
                (Role::System, "You are an assistant for managing social media accounts for a company"),
                (Role::User, "Summarize this email into a tweet to be sent by the company, use emoji if you can. \n--\n{}"),
            ],
        ),
    ]);

    // Run the chain with the provided parameters
    let res = chain
        .run(
            vec![("name", "Emil"), ("date", "February 30th 2023")].into(),
            &exec,
        )
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

In this example, we start by importing the necessary modules and defining the main function. Then, we create a new OpenAI executor. The executor is responsible for running the LLM.

Next, we create a new `Chain` object by passing in a vector of `Step` objects. Each step pairs a model with a prompt template. In this case, we have two steps:

1. The first step generates a personalized birthday email using the provided `{name}` and `{date}` parameters.
2. The second step summarizes the previously generated email into a tweet. The `{}` placeholder (the default `text` parameter) is automatically filled with the result of the previous step.

After defining the chain, we execute it using `chain.run()`. We provide a `Parameters` object containing key-value pairs for the placeholders and a reference to the executor. The run returns a `Result` — a missing parameter or a failed API call surfaces as a typed `ChainError` rather than a panic.

Chains can also be built incrementally: `Chain::of_one(step)`, `.with_step(step)`, `.push(step)`, or collected from an iterator of steps.

Sequential chains offer an efficient and straightforward way to perform a series of tasks using LLMs. By organizing the steps in a specific order, you can create complex processing pipelines that leverage the capabilities of LLMs effectively.
