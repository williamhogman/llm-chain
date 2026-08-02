---
id: introduction
title: Introduction
sidebar_label: Introduction
sidebar_position: 0
---

# Welcome to llm-chain

`llm-chain` is a collection of Rust crates designed to help you work with Large Language Models (LLMs) more effectively. Our primary focus is on providing robust support for prompt templates and chaining together prompts in multi-step chains, enabling complex tasks that LLMs can't handle in a single step. This includes, but is not limited to, summarizing lengthy texts or performing advanced data processing tasks.

## Features

- **Prompt templates**: Create reusable and easily customizable prompt templates for consistent and structured interactions with LLMs.
- **Chains**: Build powerful chains of prompts that allow you to execute more complex tasks, step by step, leveraging the full potential of LLMs.
- **Every major provider**: First-class drivers for OpenAI (and Azure OpenAI), Anthropic, Google Gemini (and Vertex AI), AWS Bedrock, and Ollama — all sharing the same `Step`/`Executor`/`Options` architecture.
- **Local inference**: Run GGUF models in-process via llama.cpp with the `llm-chain-llama` driver — no server, no API key.
- **Native async, typed errors**: Built on Rust 2024 with native `async fn` in traits and typed errors end to end. No panics, no `async-trait`.
- **Streaming**: Stream responses token by token from every HTTP provider through one `StreamingExecutor` trait — see [Streaming](streaming.md).
- **Tools**: Enhance your AI agents' capabilities by giving them access to various tools, such as running Bash commands or executing Python scripts, enabling more complex and powerful interactions.
- **Extensibility**: Designed with extensibility in mind, making it easy to integrate additional LLMs as the ecosystem grows.
- **Community-driven**: We welcome and encourage contributions from the community to help improve and expand the capabilities of llm-chain.

## Picking a driver

| Crate | Provider |
|-------|----------|
| `llm-chain-openai` | OpenAI Chat Completions and Azure OpenAI |
| `llm-chain-anthropic` | Anthropic's Messages API (Claude) |
| `llm-chain-gemini` | Google's Gemini API and Vertex AI |
| `llm-chain-bedrock` | Amazon Bedrock's Converse API |
| `llm-chain-ollama` | Ollama, local or cloud |
| `llm-chain-llama` | llama.cpp in-process (GGUF models) |
| `llm-chain-mock` | Mock executor for unit tests |
| `llm-chain-tools` | Tool access for agents |

## Getting Started

To start using llm-chain, add it as a dependency along with the driver for your provider:

```bash
cargo add llm-chain llm-chain-openai
```

Then head over to the [getting started tutorial](/docs/getting-started-tutorial).

## Connect with Us

We're always excited to hear from our users and learn about your experiences with llm-chain. If you have any questions, suggestions, or feedback, feel free to open an issue or join our [Discord community](https://discord.gg/kewN9Gtjt2).

We hope you enjoy using llm-chain to unlock the full potential of Large Language Models in your projects. Happy coding! 🎉
