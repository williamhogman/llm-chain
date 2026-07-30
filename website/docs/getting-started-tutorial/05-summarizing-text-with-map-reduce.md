# Summarizing Text with Map-Reduce

:::tip

Having problems? Don't worry, reach out on [discord](https://discord.gg/kewN9Gtjt2) and we will help you out.

:::

Map-reduce chains let you process documents that are too large for a single prompt. The chain applies a "map" step to every document in parallel, combines the outputs, and then runs a "reduce" step over the combined result.

In this tutorial we'll summarize an article into bullet points:

```rust
use llm_chain::{Parameters, chains::map_reduce::Chain};
use llm_chain_openai::chat::{Executor, Model, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new OpenAI executor
    let exec = Executor::new_default();

    // The "map" step summarizes one chunk of text into bullet points
    let map_step = Step::new(
        Model::default(),
        [
            (
                Role::System,
                "You are a bot for summarizing wikipedia articles, you are terse and focus on accuracy",
            ),
            (Role::User, "Summarize this article into bullet points:\n{text}"),
        ],
    );

    // The "reduce" step combines multiple summaries into one
    let reduce_step = Step::new(
        Model::default(),
        [
            (Role::System, "You are a diligent bot that summarizes text"),
            (
                Role::User,
                "Please combine the articles below into one summary as bullet points:\n{text}",
            ),
        ],
    );

    // Create a map-reduce chain with the map and reduce steps
    let chain = Chain::new(map_step, reduce_step);

    // Load the article — in a real application you would split large
    // documents into chunks that fit the model's context window
    let article = include_str!("article_to_summarize.md");
    let docs = vec![Parameters::new_with_text(article)];

    // Run the chain: one Parameters per document, plus base parameters
    let res = chain.run(docs, Parameters::new(), &exec).await?;

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

How it works:

1. The **map** step is formatted once per document in `docs` and all invocations are executed concurrently.
2. The outputs are merged with the executor's `combine_outputs` implementation (for OpenAI this concatenates the choices and sums token usage).
3. The merged output becomes the `{text}` parameter for the **reduce** step, which produces the final result.

Running the chain with an empty document list returns `ChainError::Empty` rather than panicking, and each document's `Parameters` are combined with the `base_parameters` you pass to `run` — handy for threading shared context (like a target language or style guide) into every map invocation.

That concludes the getting started tutorial! From here, explore the [chains documentation](../chains/00-what-are-chains.md) or swap the OpenAI driver for [any other provider](../providers.md).
