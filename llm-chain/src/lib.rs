//! # 🚀 llm-chain 🚀
//!
//! `llm-chain` is the *ultimate* toolbox for developers looking to supercharge their applications with the power of Large Language Models (LLMs)! 🎉
//!
//! This super handy crate lets you chain together LLMs, which is incredibly useful for:
//! - Summarizing lengthy documents with ease 📚
//! - Chaining together multiple prompts to tackle complex tasks like a boss 😎
//!
//! But wait, there's more! `llm-chain` is also your best friend when it comes to creating and managing prompts for LLMs. No more hassle, no more bloated syntax! Quickly create and manage prompts with our templating system, and let `llm-chain` do the rest! 🤩
//!
//! Heads up! This crate is just a library, meaning it doesn't come with any LLMs included. But don't worry! We also make the [llm-chain-openai](https://crates.io/crates/llm-chain-openai) crate, which brings the power of OpenAI's LLMs to your fingertips! 🪄 You should probably start with that crate. 😉
//!
//! So, gear up, and enjoy the amazing world of LLMs! Get ready to unlock the full potential of your applications with llm-chain! 🌈💥
//!
//! Happy coding, and may your LLM adventures be both exciting and productive! 🥳🚀
//!

pub mod chains;

mod parameters;
#[cfg(feature = "serialization")]
pub mod serialization;
mod templates;
pub mod traits;

pub use parameters::Parameters;

pub use templates::{PromptTemplate, PromptTemplateError};
