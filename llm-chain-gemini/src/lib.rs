//! # llm-chain-gemini
//!
//! This crate implements `llm-chain` steps and executors for Google's Gemini models
//! via the [Gemini API](https://ai.google.dev/api/generate-content) — and, with the
//! same wire format, via **Vertex AI** on Google Cloud.
//!
//! The crate ships its own minimal, dependency-light API client built on `reqwest` with
//! rustls — there is no unofficial SDK in the dependency tree to fall out of date.
//!
//! It targets the long-stable `generateContent` REST surface (`v1beta` on the
//! consumer API, `v1` on Vertex). Google also ships a newer Interactions API;
//! the wire types here are scoped to the [`generate_content`] module so an
//! `interactions` module can sit alongside it once that API settles.
//!
//! Three front doors, one executor:
//!
//! - [`generate_content::Executor::new_default`] — the consumer Gemini API with a
//!   `GEMINI_API_KEY`/`GOOGLE_API_KEY`
//! - [`generate_content::Executor::vertex`] — Vertex AI, scoped to a Google Cloud
//!   project and location, authenticated with an OAuth2 access token
//! - [`generate_content::Executor::vertex_express`] — Vertex AI Express Mode with
//!   just an API key, no project setup
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
