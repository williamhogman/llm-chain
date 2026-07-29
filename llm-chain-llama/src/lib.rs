//! # llm-chain-llama
//!
//! Welcome to the world of `llm-chain-llama`! This crate supercharges your applications with the power of LLaMA-family models (LLaMA, Mistral, Qwen, Gemma, and any other model in GGUF format), providing a robust framework for creating chains of models to generate human-like text.
//!
//! Built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) via the [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) crate, `llm-chain-llama` makes it a breeze to run modern open-weight models locally — no API keys, no network access required.
//!
//! # What's Inside? 🎁
//!
//! With `llm-chain-llama`, you'll be able to:
//!
//! - Generate text using any GGUF model, fully offline
//! - Create custom text summarization workflows
//! - Perform complex tasks by chaining together different prompts and models 🧠
//! - Offload computation to the GPU with the `cuda`, `metal` or `vulkan` features
//!
//! # Getting a model
//!
//! Download any GGUF model — for example from [Hugging Face](https://huggingface.co/models?library=gguf) — and pass its path to [`Executor::new`].
//!
//! # Examples 📚
//!
//! Dive into the examples folder to discover how to harness the power of this crate. You'll find detailed examples that showcase how to generate text using local GGUF models, as well as how to chain prompts together to create more complex workflows.
//!
//! Happy coding, and enjoy the amazing world of LLMs with llm-chain-llama! 🥳🚀

mod config;
mod error;
mod executor;
mod instruct;
mod output;
mod step;

pub use config::ModelConfig;
pub use error::LlamaError;
pub use executor::Executor;
pub use instruct::new_instruct_template;
pub use output::Output;
pub use step::{LlamaConfig, LlamaInvocation, Step};
