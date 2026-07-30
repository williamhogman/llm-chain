# Map-Reduce Chains

Map-reduce chains are a powerful way to process large amounts of text using large language models (LLMs). They consist of two main steps: a "map" step, which processes each document independently and in parallel, and a "reduce" step, which combines the results of the map step into a single output. This approach enables the efficient processing of large documents that exceed the LLM's context window size.

In this guide, we'll explain how to create and execute a map-reduce chain using an example. The example demonstrates how to summarize an article into bullet points using a two-step process:

1. The "map" step summarizes each document into bullet points.
2. The "reduce" step combines all bullet point summaries into a single summary.

```rust
use llm_chain::{Parameters, chains::map_reduce::Chain};
use llm_chain_openai::chat::{Executor, Model, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new OpenAI executor with the default settings
    let exec = Executor::new_default();

    // Create the "map" step to summarize an article into bullet points
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

    // Create the "reduce" step to combine multiple summaries into one
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

    // Load the content of the article to be summarized
    let article = include_str!("article_to_summarize.md");

    // Create a vector with one Parameters object per document
    let docs = vec![Parameters::new_with_text(article)];

    // Run the chain with the documents and base parameters for the "reduce" step
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

In this example, we create the "map" and "reduce" steps with `Step::new`, pairing each with a model and prompt template. The "map" step is responsible for summarizing each document, while the "reduce" step combines the summaries into a single output.

After defining the steps, we create a new `Chain` object by passing in the "map" and "reduce" steps. We then load the content of the article and create one `Parameters` object per document.

Finally, we execute the map-reduce chain using `chain.run()`, passing in the documents, base `Parameters` that are combined into every invocation, and the executor. All map invocations run concurrently, their outputs are merged with the executor's `combine_outputs` (which also sums token usage), and the merged text becomes the `{text}` parameter of the reduce step.

An empty document list returns `ChainError::Empty` — no panics.

Map-reduce chains offer an effective way to handle large documents or multiple documents using LLMs. By breaking the text into manageable chunks and combining the results, you can create efficient pipelines for text processing tasks such as summarization, translation, and analysis.
