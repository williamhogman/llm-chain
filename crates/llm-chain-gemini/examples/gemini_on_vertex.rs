//! Runs the same chain against Vertex AI instead of the consumer Gemini API.
//!
//! Vertex serves the identical `generateContent` wire format behind
//! project/location-scoped URLs with OAuth2 bearer auth — so the only change
//! versus the other examples is how the [`Executor`] is constructed.
//!
//! ```sh
//! export GOOGLE_CLOUD_PROJECT=my-project
//! export VERTEX_ACCESS_TOKEN=$(gcloud auth print-access-token)
//! cargo run --example gemini_on_vertex
//! ```
use llm_chain::{Parameters, traits::StepExt};
use llm_chain_gemini::generate_content::{Executor, Model, Role, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let project = std::env::var("GOOGLE_CLOUD_PROJECT").expect("set GOOGLE_CLOUD_PROJECT");
    let token = std::env::var("VERTEX_ACCESS_TOKEN")
        .expect("set VERTEX_ACCESS_TOKEN, e.g. from `gcloud auth print-access-token`");

    // `global` lets Google route to whichever region has capacity; pass a
    // region like `europe-north1` instead to pin the serving location.
    let exec = Executor::vertex(project, "global", token);

    let chain = Step::new(
        Model::default(),
        [(Role::User, "Make a personalized greeting for Joe")],
    )
    .with_system("You are a bot for making personalized greetings")
    .to_chain();
    let res = chain.run(Parameters::new(), &exec).await.unwrap();
    println!("{}", res.text());
}
