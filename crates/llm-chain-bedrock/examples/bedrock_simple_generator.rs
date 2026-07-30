use llm_chain::{Parameters, traits::StepExt};
use llm_chain_bedrock::converse::{Executor, Model, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Reads AWS_BEARER_TOKEN_BEDROCK (and AWS_REGION) from the environment.
    let exec = Executor::new_default().unwrap();
    let chain = Step::new(
        Model::default(),
        [(Role::User, "Make a personalized greet for Joe")],
    )
    .with_system("You are a bot for making personalized greetings")
    .to_chain();
    let res = chain.run(Parameters::new(), &exec).await.unwrap();
    println!("{}", res.text());
}
