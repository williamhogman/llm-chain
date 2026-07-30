use llm_chain::{Parameters, traits::StepExt};
use llm_chain_ollama::chat::{Executor, Model, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Talks to the local Ollama server (respects OLLAMA_HOST).
    // Pull the model first: `ollama pull qwen3`.
    let exec = Executor::new_default();
    let chain = Step::new(
        Model::default(),
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
    println!("{}", res.text());
}
