//! # llm-chain-ollama
//!
//! This crate implements `llm-chain` steps and executors for models served by
//! [Ollama](https://ollama.com) via its [chat API](https://docs.ollama.com/api#generate-a-chat-completion).
//!
//! Ollama runs open-weight models (Llama, Qwen, Gemma, DeepSeek, gpt-oss, …)
//! locally with zero API keys, and also serves larger variants through Ollama's
//! cloud. This crate speaks to both: it defaults to the local server at
//! `http://localhost:11434` (respecting `OLLAMA_HOST`) and supports
//! bearer-token auth for remote hosts.
//!
//! The crate ships its own minimal, dependency-light API client built on `reqwest` with
//! rustls — there is no unofficial SDK in the dependency tree to fall out of date.
//!
//! ## Getting started
//!
//! Pull a model first (`ollama pull qwen3`), then:
//!
//! ```no_run
//! use llm_chain::{Parameters, traits::StepExt};
//! use llm_chain_ollama::chat::{Executor, Model, Role, Step};
//!
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() {
//!     // Talks to the local Ollama server (respects OLLAMA_HOST).
//!     let exec = Executor::new_default();
//!     let chain = Step::new(
//!         Model::default(),
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
