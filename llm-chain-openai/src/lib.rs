//! # llm-chain-openai
//!
//! The OpenAI driver for [llm-chain](https://crates.io/crates/llm-chain): run chains against
//! OpenAI's Chat Completions API, from the GPT-5.6 family down to the classic GPT-4 era models.
//!
//! To get started you need an OpenAI API key — create one at
//! <https://platform.openai.com/api-keys> and export it as `OPENAI_API_KEY`, or pass it
//! explicitly with [`chat::Executor::with_api_key`].
//!
//! # What's inside? 🎁
//!
//! - [`chat::Step`] — a prompt template plus a [`chat::Model`] and optional [`chat::Options`]
//!   (temperature, reasoning effort, response format, …)
//! - [`chat::Executor`] — runs formatted steps against the API, ready to plug into
//!   sequential and map-reduce chains
//! - [`chat::AzureExecutor`] — the same executor pointed at Azure OpenAI's
//!   OpenAI-compatible v1 surface (`AzureExecutor::azure("my-resource", key)`),
//!   with both API-key and Microsoft Entra ID authentication
//! - Full serialization support, so chains can be stored as YAML and loaded back
//!
//! # Example
//!
//! ```no_run
//! use llm_chain::{Parameters, traits::StepExt};
//! use llm_chain_openai::chat::{Executor, Model, Role, Step};
//!
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() {
//!     let exec = Executor::new_default();
//!     let chain = Step::new(
//!         Model::default(),
//!         [
//!             (Role::System, "You are a helpful assistant."),
//!             (Role::User, "Tell me about the Rust programming language."),
//!         ],
//!     )
//!     .to_chain();
//!     let res = chain.run(Parameters::new(), &exec).await.unwrap();
//!     println!(
//!         "{}",
//!         res.choices
//!             .first()
//!             .and_then(|c| c.message.content.as_deref())
//!             .unwrap_or_default()
//!     );
//! }
//! ```
//!
//! Dive into the examples folder for more, including parameterized prompts, sequential
//! chains, request options and YAML round-trips. Happy coding! 🥳🚀

pub mod chat;

/// Deprecated alias for the [`chat`] module.
#[deprecated(
    since = "0.2.0",
    note = "the `chatgpt` module has been renamed to `chat`"
)]
pub mod chatgpt {
    pub use crate::chat::*;
}
