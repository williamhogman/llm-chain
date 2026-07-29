# llm-chain 🚀

`llm-chain` is a collection of Rust crates designed to help you work with Large Language Models (LLMs) more effectively. Our primary focus is on providing robust support for prompt templates and chaining together prompts in multi-step chains, enabling complex tasks that LLMs can't handle in a single step. This includes, but is not limited to, summarizing lengthy texts or performing advanced data processing tasks.

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/sobelio/llm-chain/cicd.yaml?branch=main?style=flat-square)
![Crates.io](https://img.shields.io/crates/v/llm-chain?style=flat-square)
![Crates.io](https://img.shields.io/crates/l/llm-chain-openai?style=flat-square)
![License](https://img.shields.io/github/license/sobelio/llm-chain)

## Examples 💡

To help you get started, here is an example demonstrating how to use `llm-chain`. You can find more examples in the [examples folder](/llm-chain-openai/examples) in the repository.

```rust
use llm_chain::{Parameters, traits::StepExt};
use llm_chain_openai::chat::{Executor, Model, Role, Step};

let exec = Executor::new_default();
let chain = Step::new(
    Model::default(), // gpt-5.6-terra
    [
        (Role::System, "You are a bot for making personalized greetings"),
        (Role::User, "Make a personalized greet for Joe"),
    ]
).to_chain();
let res = chain.run(Parameters::new(), &exec).await?;
println!("{:?}", res);
```

## Features 🌟

- **Prompt templates**: Create reusable and easily customizable prompt templates for consistent and structured interactions with LLMs.
- **Chains**: Build powerful chains of prompts that allow you to execute more complex tasks, step by step, leveraging the full potential of LLMs.
- **OpenAI support**: First-class support for OpenAI's current models — the GPT-5.6 family (sol/terra/luna) down through GPT-5.x, GPT-4.1, GPT-4o and the o-series reasoning models — plus any custom or fine-tuned model id, with per-step request options (temperature, reasoning effort, verbosity, JSON mode and more).
- **Anthropic support**: Claude via the Messages API — the Claude 5 generation (Fable, Opus, Sonnet) and Haiku 4.5 — with system prompts, sampling controls, reasoning effort and extended thinking, on a minimal built-in client (reqwest + rustls, no third-party SDK).
- **Google Gemini support**: Gemini via the `generateContent` API — the Gemini 3 generation (3.6/3.5 Flash, 3.1 Pro, Flash-Lite) and the 2.5 family — with system instructions, sampling controls and thinking level/budget, on the same minimal built-in client.
- **Local models via llama.cpp**: Run LLaMA, Mistral, Qwen, Gemma and any other GGUF model fully offline, with optional GPU acceleration (CUDA, Metal, Vulkan).
- **Tools**: Enhance your AI agents' capabilities by giving them access to various tools, such as running Bash commands or executing Python scripts, enabling more complex and powerful interactions.
- **Typed errors, no panics**: Formatting, execution and chain errors are all surfaced as typed `Result`s.
- **Modern async**: Built on native `async fn` in traits — no `async-trait` macro overhead.
- **Extensibility**: Designed with extensibility in mind, making it easy to integrate additional LLMs as the ecosystem grows.
- **Community-driven**: We welcome and encourage contributions from the community to help improve and expand the capabilities of `llm-chain`.

## Getting Started 🚀

To start using `llm-chain`, add it as a dependency in your `Cargo.toml` (requires Rust 1.85+):

```toml
[dependencies]
llm-chain = "0.2.0"
llm-chain-openai = "0.2.0"     # OpenAI (GPT)
llm-chain-anthropic = "0.2.0"  # Anthropic (Claude)
llm-chain-gemini = "0.2.0"     # Google (Gemini)
llm-chain-llama = "0.2.0"      # Local GGUF models via llama.cpp
```

Claude example:

```rust
use llm_chain::{Parameters, traits::StepExt};
use llm_chain_anthropic::messages::{Executor, Model, Role, Step};

let exec = Executor::new_default()?; // reads ANTHROPIC_API_KEY
let chain = Step::new(
    Model::default(), // claude-sonnet-5
    [(Role::User, "Make a personalized greet for Joe")],
)
.with_system("You are a bot for making personalized greetings")
.to_chain();
let res = chain.run(Parameters::new(), &exec).await?;
println!("{}", res.text());
```

Gemini example:

```rust
use llm_chain::{Parameters, traits::StepExt};
use llm_chain_gemini::generate_content::{Executor, Model, Role, Step};

let exec = Executor::new_default()?; // reads GEMINI_API_KEY (or GOOGLE_API_KEY)
let chain = Step::new(
    Model::default(), // gemini-3.6-flash
    [(Role::User, "Make a personalized greet for Joe")],
)
.with_system("You are a bot for making personalized greetings")
.to_chain();
let res = chain.run(Parameters::new(), &exec).await?;
println!("{}", res.text());
```

Then, refer to the [documentation](https://docs.rs/llm-chain) and the examples ([OpenAI](/llm-chain-openai/examples), [Anthropic](/llm-chain-anthropic/examples), [Gemini](/llm-chain-gemini/examples)) to learn how to create prompt templates, chains, and more.

## Contributing 🤝

We warmly welcome contributions from everyone! If you're interested in helping improve `llm-chain`, please check out our [`CONTRIBUTING.md`](/docs/CONTRIBUTING.md) file for guidelines and best practices.

## License 📄

`llm-chain` is licensed under the [MIT License](/LICENSE).

## Connect with Us 🌐

If you have any questions, suggestions, or feedback, feel free to open an issue or join our community discussions. We're always excited to hear from our users and learn about your experiences with `llm-chain`.

We hope you enjoy using `llm-chain` to unlock the full potential of Large Language Models in your projects. Happy coding! 🎉
