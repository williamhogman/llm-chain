---
id: providers
title: Providers
sidebar_label: Providers
sidebar_position: 3
---

# Providers

Every driver follows the same architecture: a `Step` pairs a model with a prompt template, an `Executor` sends formatted requests to the provider, and an optional `Options` builder controls sampling and reasoning. Switching providers is a matter of swapping imports.

## OpenAI (`llm-chain-openai`)

```rust
use llm_chain_openai::chat::{Executor, Model, Options, ReasoningEffort, Role, Step};

let exec = Executor::new_default(); // reads OPENAI_API_KEY
let step = Step::new(
    Model::default(), // gpt-5.6-terra
    [
        (Role::Developer, "You are a helpful assistant."),
        (Role::User, "Tell me about {topic}."),
    ],
)
.with_options(Options::new().with_reasoning_effort(ReasoningEffort::Medium));
```

Also supports **Azure OpenAI** (`Executor::azure(resource, key)` or Entra ID tokens) and any OpenAI-compatible server via `Executor::with_base_url` (vLLM, OpenRouter, local proxies).

## Anthropic (`llm-chain-anthropic`)

```rust
use llm_chain_anthropic::messages::{Effort, Executor, Model, Options, Role, Step};

let exec = Executor::new_default()?; // reads ANTHROPIC_API_KEY
let step = Step::new(
    Model::default(), // claude-sonnet-5
    [(Role::User, "Tell me about {topic}.")],
)
.with_system("You are a helpful assistant.")
.with_options(Options::new().with_effort(Effort::High));
```

Supports extended thinking budgets (`with_thinking_budget`) on Claude 4.x models and the `effort` control on the Claude 5 generation.

## Google Gemini (`llm-chain-gemini`)

```rust
use llm_chain_gemini::generate_content::{Executor, Model, Options, Role, Step, ThinkingLevel};

let exec = Executor::new_default()?; // reads GEMINI_API_KEY or GOOGLE_API_KEY
let step = Step::new(
    Model::default(), // gemini-3.6-flash
    [(Role::User, "Tell me about {topic}.")],
)
.with_system("You are a helpful assistant.")
.with_options(Options::new().with_thinking_level(ThinkingLevel::Low));
```

Also routes to **Vertex AI**: `Executor::vertex(project, location, oauth_token)` or `Executor::vertex_express(api_key)`.

## AWS Bedrock (`llm-chain-bedrock`)

```rust
use llm_chain_bedrock::converse::{Executor, Model, Role, Step};

let exec = Executor::new_default()?; // reads AWS_BEARER_TOKEN_BEDROCK + AWS_REGION
let step = Step::new(
    Model::default(), // global.anthropic.claude-sonnet-5-v1:0
    [(Role::User, "Tell me about {topic}.")],
)
.with_system("You are a helpful assistant.");
```

One Converse wire format for every model on Bedrock — Claude, Nova, Llama, Mistral and more. Well-known model IDs ship as constants in the `models` module.

## Ollama (`llm-chain-ollama`)

```rust
use llm_chain_ollama::chat::{Executor, Model, Options, Role, Step, Think};

let exec = Executor::new_default(); // OLLAMA_HOST or http://localhost:11434
let step = Step::new(
    Model::from("qwen3"), // any name:tag from the Ollama registry
    [(Role::User, "Tell me about {topic}.")],
)
.with_options(Options::new().with_think(Think::Low));
```

Runs against a local Ollama server or Ollama cloud (`Executor::cloud(api_key)`), with JSON-schema constrained outputs via `Options::with_format`.

## llama.cpp in-process (`llm-chain-llama`)

```rust
use llm_chain_llama::{Executor, Step};

let exec = Executor::new("models/llama-3.2-1b-q4_k_m.gguf")?;
let step = Step::new("The colors of the rainbow are (in order): ".into());
```

Loads GGUF models directly — no server, no API key. GPU offload via the `cuda`, `metal` and `vulkan` features. See the [llama tutorial](./llama-tutorial.md) for a full walkthrough.

## Mock (`llm-chain-mock`)

```rust
use llm_chain_mock::Executor;

let echo = Executor::new();                            // echoes the prompt back
let scripted = Executor::with_responses(["a", "b"]);   // canned responses, in order
let failing = Executor::failing("simulated outage");   // always errors
```

A deterministic executor for unit-testing chains without network access. Every executed prompt is recorded and available via `Executor::calls()`, so tests can assert on exactly what the "model" was asked.
