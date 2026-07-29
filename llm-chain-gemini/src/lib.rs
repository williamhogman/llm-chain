//! # llm-chain-gemini
//!
//! This crate implements `llm-chain` steps and executors for Google's Gemini models
//! via the [Gemini API](https://ai.google.dev/api/generate-content).
//!
//! The crate ships its own minimal, dependency-light API client built on `reqwest` with
//! rustls — there is no unofficial SDK in the dependency tree to fall out of date.
//!
//! ## Getting started
//!
//! ```no_run
//! use llm_chain::{Parameters, traits::StepExt};
//! use llm_chain_gemini::generate_content::{Executor, Model, Role, Step};
//!
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() {
//!     // Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.
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
pub mod generate_content;
