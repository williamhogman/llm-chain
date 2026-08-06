use llm_chain::{Parameters, traits::StepExt};
use llm_chain_lovable::chat::{Executor, Model, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Reads LOVABLE_API_KEY from the environment.
    let exec = Executor::new_default().unwrap();
    let chain = Step::new(
        Model::default(), // google/gemini-3.6-flash
        [
            (
                Role::System,
                "You are a bot for making personalized greetings",
            ),
            (Role::User, "Make a personalized greeting for Joe"),
        ],
    )
    .to_chain();
    let res = chain.run(Parameters::new(), &exec).await.unwrap();
    println!("{}", res.text());
}
