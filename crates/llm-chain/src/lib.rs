//! # llm-chain
//!
//! `llm-chain` is a toolbox for building LLM-powered applications in Rust: it
//! lets you chain prompts across steps and models, so the output of one
//! invocation feeds the next. Use it to summarize long documents, run
//! multi-step reasoning pipelines, or drive agents with tools.
//!
//! This crate contains the model-agnostic core:
//!
//! - [`PromptTemplate`] — prompt strings with `{}` / `{name}` placeholders
//! - [`Parameters`] — the key-value state threaded between steps
//! - [`traits::Step`] and [`traits::Executor`] — the contract every model
//!   driver implements, with native `async fn` and typed errors
//! - [`chains::sequential`] and [`chains::map_reduce`] — the chain runners
//! - [`serialization`] — store chains as YAML and load them back
//!
//! # Drivers
//!
//! Pick one or more driver crates for the models you want to run; they all
//! share the same `Step`/`Executor`/`Options` shape, so switching providers is
//! a matter of swapping imports:
//!
//! | Crate | Provider |
//! |-------|----------|
//! | [`llm-chain-openai`](https://crates.io/crates/llm-chain-openai) | OpenAI Chat Completions and Azure OpenAI |
//! | [`llm-chain-anthropic`](https://crates.io/crates/llm-chain-anthropic) | Anthropic's Messages API (Claude) |
//! | [`llm-chain-gemini`](https://crates.io/crates/llm-chain-gemini) | Google's Gemini API and Vertex AI |
//! | [`llm-chain-bedrock`](https://crates.io/crates/llm-chain-bedrock) | Amazon Bedrock's Converse API |
//! | [`llm-chain-ollama`](https://crates.io/crates/llm-chain-ollama) | Ollama, local or cloud |
//! | [`llm-chain-llama`](https://crates.io/crates/llm-chain-llama) | llama.cpp in-process (GGUF models) |
//! | [`llm-chain-tools`](https://crates.io/crates/llm-chain-tools) | Tool access for agents |
//!
//! # Example
//!
//! Templates and parameters work without any driver:
//!
//! ```
//! use llm_chain::{Parameters, PromptTemplate};
//!
//! let template: PromptTemplate = "Summarize this text: {text}".into();
//! let parameters: Parameters = vec![("text", "..a very long text..")].into();
//! assert_eq!(
//!     template.format(&parameters).unwrap(),
//!     "Summarize this text: ..a very long text.."
//! );
//! ```
//!
//! Add a driver to actually run a chain — see the driver crates for
//! end-to-end examples.
//!
//! # Cargo features
//!
//! - `serialization` *(default)* — YAML (de)serialization of chains and steps
//! - `async` — async file I/O for [`serialization`]

pub mod chains;

mod model_id;
mod parameters;
#[cfg(feature = "serialization")]
pub mod serialization;
mod templates;
pub mod traits;

pub use parameters::Parameters;

pub use templates::{PromptTemplate, PromptTemplateError};
