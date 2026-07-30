//! A two-step sequential chain: the first step invents a company name, the
//! second writes a slogan for it. The output of each step feeds the `{}`
//! placeholder of the next.
use llm_chain::{Parameters, chains::sequential::Chain};
use llm_chain_ollama::chat::{ChatPromptTemplate, Executor, Model, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let exec = Executor::new_default();
    let chain = Chain::new(vec![
        Step::new(
            Model::default(),
            ChatPromptTemplate::system_and_user(
                "You are a bot for making company names. Answer with the name only.",
                "Make a company name for {}",
            ),
        ),
        Step::new(
            Model::default(),
            ChatPromptTemplate::system_and_user(
                "You are a bot for making slogans. Answer with the slogan only.",
                "Make a slogan for {}",
            ),
        ),
    ]);
    let res = chain
        .run(
            Parameters::new_with_text("a cloud computing startup"),
            &exec,
        )
        .await
        .unwrap();
    println!("{}", res.text());
}
