//! # llm-chain-bedrock
//!
//! This crate implements `llm-chain` steps and executors for Amazon Bedrock
//! via the [Converse API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html).
//!
//! The Converse API is Bedrock's model-agnostic chat surface: one wire format
//! for every hosted model family (Anthropic Claude, Amazon Nova, Meta Llama,
//! Mistral, and more). Switching models is a one-line change.
//!
//! The crate ships its own minimal, dependency-light API client built on `reqwest` with
//! rustls — there is no heavyweight AWS SDK in the dependency tree. It authenticates
//! with Bedrock API keys (`Authorization: Bearer`), the credential mechanism AWS
//! introduced for Bedrock in 2025. If your deployment requires SigV4 request signing
//! (IAM roles, STS sessions), front the call with the official `aws-sdk-bedrockruntime`
//! instead.
//!
//! ## Getting started
//!
//! ```no_run
//! use llm_chain::{Parameters, traits::StepExt};
//! use llm_chain_bedrock::converse::{Executor, Model, Role, Step};
//!
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() {
//!     // Reads AWS_BEARER_TOKEN_BEDROCK (and AWS_REGION) from the environment.
//!     let exec = Executor::new_default().unwrap();
//!     let chain = Step::new(
//!         Model::default(),
//!         [
//!             (Role::User, "Make a personalized greeting for Joe"),
//!         ],
//!     )
//!     .with_system("You are a bot for making personalized greetings")
//!     .to_chain();
//!     let res = chain.run(Parameters::new(), &exec).await.unwrap();
//!     println!("{}", res.text());
//! }
//! ```
pub mod converse;
