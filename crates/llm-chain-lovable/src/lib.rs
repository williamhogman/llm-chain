//! # llm-chain-lovable
//!
//! This crate implements `llm-chain` steps and executors for the
//! [Lovable AI Gateway](https://docs.lovable.dev/features/ai), Lovable's
//! hosted model gateway.
//!
//! The gateway exposes one OpenAI-compatible chat-completions surface for
//! chat models across vendors — Google Gemini, OpenAI and more — behind a
//! single credential. Model ids are vendor-prefixed strings from the Lovable
//! model catalog (e.g. `google/gemini-3.6-flash`, `openai/gpt-5.5`), so
//! switching vendors is a one-string change with no new client, key or wire
//! format.
//!
//! Authentication uses the `LOVABLE_API_KEY` secret, sent in the
//! `Lovable-API-Key` header. In Lovable Cloud projects the key is
//! auto-provisioned; it is a server-side credential and must never be
//! shipped to browsers or other untrusted clients. Usage is billed in
//! Lovable credits: HTTP 429 means rate-limited (retry with backoff) and
//! HTTP 402 means the workspace is out of credits — both are surfaced as
//! typed errors with [`chat::LovableError::is_rate_limit`] and
//! [`chat::LovableError::is_credits_exhausted`].
//!
//! The crate ships its own minimal, dependency-light API client built on
//! `reqwest` with rustls — there is no unofficial SDK in the dependency tree
//! to fall out of date.
//!
//! ## Getting started
//!
//! ```no_run
//! use llm_chain::{Parameters, traits::StepExt};
//! use llm_chain_lovable::chat::{Executor, Model, Role, Step};
//!
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() {
//!     // Reads LOVABLE_API_KEY from the environment.
//!     let exec = Executor::new_default().unwrap();
//!     let chain = Step::new(
//!         Model::default(), // google/gemini-3.6-flash
//!         [
//!             (Role::System, "You are a bot for making personalized greetings"),
//!             (Role::User, "Make a personalized greeting for Joe"),
//!         ],
//!     )
//!     .to_chain();
//!     let res = chain.run(Parameters::new(), &exec).await.unwrap();
//!     println!("{}", res.text());
//! }
//! ```
pub mod chat;
