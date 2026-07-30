---
slug: llm-chain-0-14
title: "llm-chain 0.14: Rust 2024, native async, and every major provider"
authors: [whn]
tags: [llm-chain, release, rust, openai, anthropic, gemini, bedrock, ollama]
---

llm-chain is back. Version 0.14.0 is a ground-up modernization of the library for the Rust and LLM ecosystems of 2026 — and the biggest release in the project's history.

<!-- truncate -->

## What's new

### Six drivers, one architecture

Every provider now follows the exact same `Step` / `Executor` / `Options` shape, so switching providers is a matter of swapping imports:

- **`llm-chain-openai`** — OpenAI Chat Completions (GPT-5.6 family, o-series) plus **Azure OpenAI** and any OpenAI-compatible server (vLLM, OpenRouter).
- **`llm-chain-anthropic`** — Anthropic's Messages API with the Claude 5 generation, extended thinking budgets and effort control. *New.*
- **`llm-chain-gemini`** — Google's Gemini API (Gemini 3.x) with thinking levels and JSON mode, plus **Vertex AI** routes. *New.*
- **`llm-chain-bedrock`** — Amazon Bedrock's Converse API: one wire format for Claude, Nova, Llama, Mistral and more. *New.*
- **`llm-chain-ollama`** — local or cloud Ollama, with think levels and JSON-schema outputs. *New.*
- **`llm-chain-llama`** — rewritten on the maintained `llama-cpp-2` bindings: loads GGUF models in-process, with a modern sampling pipeline and `cuda`/`metal`/`vulkan` GPU offload.

Plus **`llm-chain-mock`**, a deterministic executor for unit-testing chains without network access.

### A modern core

- **Rust 2024 edition**, MSRV 1.85.
- **Native `async fn` in traits** — `async-trait` is gone; futures are guaranteed `Send`.
- **Typed errors end to end** — prompt formatting is fallible, steps and executors have typed error associated types, and chain runs return `Result<Output, ChainError>`. No panics.
- **Credential hygiene** — all API keys and tokens are held as `SecretString`: redacted in `Debug` output, zeroized on drop.
- **Retry-friendly errors** — every provider error exposes `.status()` and `.is_rate_limit()` for uniform backoff handling.

### First-party tool calling

Every HTTP provider speaks its native tool-calling dialect — OpenAI function tools, Anthropic `tool_use` blocks, Gemini function declarations, Bedrock `toolConfig`, Ollama tools — and `llm-chain-tools` bridges any `ToolCollection` into all of them: `tool_schemas()` generates a JSON Schema per tool, `invoke_json()` runs the calls the model makes. See the new [Tool calling](/docs/tool-calling) docs page.

### The wire, verified

The HTTP drivers are built on a minimal `reqwest` + rustls client — no heavyweight SDKs — and each ships a mock-API test suite that asserts on the exact wire format. The whole workspace (250+ tests) runs offline, including a real GGUF end-to-end inference test.

## Upgrading

0.14 is a breaking release from both the 0.1.x and 0.13.x lines. The [migration guide](https://github.com/sobelio/llm-chain/blob/main/docs/MIGRATION-0.14.md) covers both starting points, including a concept-mapping table for 0.13.x users (`prompt!`/`executor!` macros, conversation chains, vector stores).

```bash
cargo add llm-chain llm-chain-openai
```

Head over to the [getting started tutorial](/docs/getting-started-tutorial) — and as always, come say hi on [Discord](https://discord.gg/kewN9Gtjt2) or [GitHub](https://github.com/sobelio/llm-chain).
