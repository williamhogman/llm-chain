//! Runs a chain against Azure OpenAI (Azure AI Foundry) instead of
//! `api.openai.com`.
//!
//! Azure's v1 surface is OpenAI-compatible: the deployment name goes in the
//! body's `model` field and no `api-version` pinning is needed — so the only
//! change versus the other examples is how the executor is constructed.
//!
//! ```sh
//! export AZURE_OPENAI_ENDPOINT=my-resource            # or the full https endpoint
//! export AZURE_OPENAI_API_KEY=...
//! export AZURE_OPENAI_DEPLOYMENT=gpt-5.6-terra        # your deployment's name
//! cargo run --example azure_generation
//! ```
use llm_chain::{Parameters, traits::StepExt};
use llm_chain_openai::chat::{AzureExecutor, Model, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT").expect("set AZURE_OPENAI_ENDPOINT");
    let api_key = std::env::var("AZURE_OPENAI_API_KEY").expect("set AZURE_OPENAI_API_KEY");
    // On Azure, `model` names your *deployment*; deployments are conventionally
    // named after the model they serve.
    let deployment = std::env::var("AZURE_OPENAI_DEPLOYMENT")
        .map(Model::Other)
        .unwrap_or_default();

    let exec = AzureExecutor::azure(endpoint, api_key);
    let chain = Step::new(
        deployment,
        [
            (
                Role::System,
                "You are a bot for making personalized greetings",
            ),
            (Role::User, "Make a personalized greet for Joe"),
        ],
    )
    .to_chain();
    let res = chain.run(Parameters::new(), &exec).await.unwrap();
    println!(
        "{}",
        res.choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or_default()
    );
}
