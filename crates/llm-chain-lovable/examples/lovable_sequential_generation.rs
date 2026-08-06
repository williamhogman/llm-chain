//! A two-step sequential chain: the first step invents a company name, the
//! second writes a slogan for it. The output of each step feeds the `{}`
//! placeholder of the next — here across two different vendors through the
//! same gateway.
use llm_chain::{Parameters, chains::sequential::Chain};
use llm_chain_lovable::chat::{ChatPromptTemplate, Executor, Model, Step, models};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let exec = Executor::new_default().expect("set LOVABLE_API_KEY");
    let chain = Chain::new(vec![
        Step::new(
            Model::default(),
            ChatPromptTemplate::system_and_user(
                "You are a bot for making company names. Answer with the name only.",
                "Make a company name for {}",
            ),
        ),
        Step::new(
            Model::new(models::GPT_5_5),
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
