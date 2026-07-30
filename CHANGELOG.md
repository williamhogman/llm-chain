# Changelog

All notable changes to the `llm-chain` workspace are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and all crates in the workspace share a single version number and adhere to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-30

A ground-up modernization of the entire workspace — the first release since
0.1.4 (April 2023). Every crate was brought up to the current state of the
Rust and LLM ecosystems, and five new provider crates were added.

**Breaking release.** See [`docs/MIGRATION-0.2.md`](docs/MIGRATION-0.2.md) for
a step-by-step upgrade guide from 0.1.x.

### Highlights

- **Rust 2024 edition**, resolver 3, MSRV 1.85, workspace-level dependencies
  and lints, thin LTO in release builds.
- **Native `async fn` in traits** — `async-trait` is gone. The `Executor`
  trait uses `#[trait_variant::make(Send)]` so futures stay `Send` on
  multi-threaded runtimes.
- **Typed errors, no panics** — formatting, execution and chain runs all
  return `Result`s with dedicated error types (`PromptTemplateError`,
  per-provider errors, `ChainError`).
- **Four new HTTP providers** — Anthropic, Google Gemini, Amazon Bedrock and
  Ollama, all on a minimal built-in client (`reqwest` 0.13 + rustls, no
  heavyweight SDKs).
- **Hyperscaler front doors** — Azure OpenAI (OpenAI-compatible v1 surface),
  Google Vertex AI (project-scoped and Express Mode) and AWS Bedrock
  (Converse API).
- **Credential hygiene** — every API key and token is held as
  `secrecy::SecretString`: redacted from `Debug`, zeroized on drop.

### Added

#### New crates

- **`llm-chain-anthropic`** — Anthropic's Messages API (Claude). Claude 5
  generation (Fable, Opus, Sonnet — Sonnet 5 is the default) plus pinned 4.x
  ids; system prompts; sampling controls (`temperature`, `top_p`, `top_k`);
  extended thinking budgets (`Thinking`) and the `effort` parameter
  (`Low`/`Medium`/`High`); typed stop reasons and token usage.
- **`llm-chain-gemini`** — Google's `generateContent` API. Gemini 3.6/3.5
  Flash, 3.1 Pro, Flash-Lite and the 2.5 family; system instructions;
  `ThinkingLevel` (Gemini 3) and `thinking_budget` (Gemini 2.5);
  `include_thoughts`; JSON output via `response_mime_type`; detailed
  `UsageMetadata` (prompt/candidates/thoughts/cached token counts). Also
  drives **Vertex AI**: `Executor::vertex(project, location, token)` for
  OAuth2 deployments and `Executor::vertex_express(key)` for Express Mode.
- **`llm-chain-bedrock`** — Amazon Bedrock's Converse API: one wire format
  for Claude, Nova, Llama, Mistral and every other hosted family. Bedrock API
  key auth (`AWS_BEARER_TOKEN_BEDROCK`), regional endpoint discovery
  (`AWS_REGION`), URL-encoded model-id paths, reasoning content blocks, and
  `additional_model_request_fields` for model-specific extensions.
- **`llm-chain-ollama`** — Ollama's `/api/chat`, local
  (`OLLAMA_HOST`, default `http://localhost:11434`) or cloud
  (`Executor::cloud(api_key)`). Arbitrary `name:tag` models (default
  `qwen3`); `Think` levels for reasoning models; `Format` for JSON mode and
  full JSON-schema constraints; `keep_alive`; generation timings merged
  across chain steps.

#### Core (`llm-chain`)

- `impl_model_id!` / `impl_model_id_serde!` macros: every driver's `Model`
  type is now generated from a single variant-to-id table, with a
  `KNOWN_IDS` constant for introspection and string-based serde.
- `Parameters`: `iter()`, `get()`, `FromIterator`, `Extend`, and
  `From<[(K, V); N]>`.
- `PromptTemplate`: `Display`, `as_str()`, `PartialEq`/`Eq`/`Hash`, and
  `{{`/`}}` escape sequences.
- `chains::sequential::Chain`: `push()`, `len()`, `is_empty()`, `steps()`,
  `FromIterator`, `Extend`, `IntoIterator`.
- `async` cargo feature: async file I/O for chain serialization.

#### Providers

- **OpenAI**: July 2026 model lineup — the GPT-5.6 family (Sol, Terra, Luna;
  `gpt-5.6-terra` is the default), GPT-5.4, GPT-5.2 Pro, GPT-4.1, GPT-4o and
  the o-series — plus any custom/fine-tuned id. New `Options`
  (`temperature`, `top_p`, `max_completion_tokens`, `reasoning_effort`,
  `verbosity`, `response_format`). Crate-owned `Role` enum including
  `Developer` for reasoning models. `Executor::with_api_key`. Token usage
  merged across combined outputs.
- **Azure OpenAI**: `AzureExecutor` / `AzureV1Config` targeting the
  OpenAI-compatible `/openai/v1` surface — no `api-version` pinning — with
  both `api-key` and Microsoft Entra ID bearer auth.
- **All HTTP providers**: `.status()` and `.is_rate_limit()` on error types
  for uniform retry/backoff handling across providers (429s, Anthropic
  `overloaded_error`, Gemini `RESOURCE_EXHAUSTED`, Bedrock throttling).

#### Tooling & CI

- Mock-API integration test suites for Anthropic, Gemini, Bedrock and Ollama
  asserting wire format, auth headers and error mapping — no API keys needed.
- End-to-end GGUF inference test for `llm-chain-llama` (tiny stories260K
  model).
- CI: fmt + clippy (`-D warnings`) + multi-OS test matrix (Ubuntu, macOS) +
  MSRV (1.85) verification + rustdoc with `-D warnings`.
- Tag-triggered release workflow publishing all crates to crates.io in
  dependency order.

### Changed

- **`llm-chain`**: prompt formatting is fallible
  (`PromptTemplate::format -> Result<String, PromptTemplateError>`) and no
  longer depends on `dynfmt`; chains return `Result<Output, ChainError>`;
  YAML serialization moved from the deprecated `serde_yaml` to
  `serde_yaml_ng`.
- **`llm-chain-openai`**: the `chatgpt` module is renamed to `chat` (a
  deprecated alias remains); `async-openai` upgraded 0.10 → 0.41.
- **`llm-chain-llama`**: rewritten on the maintained `llama-cpp-2` bindings
  (GGUF models) — the vendored `sys` FFI crate and the `llama.cpp` git
  submodule are gone. Modern sampling pipeline (top-k, top-p, temperature,
  repetition penalties, deterministic seeds), incremental stop-sequence
  checking, UTF-8-safe detokenization, `ModelConfig` for context window and
  GPU offload, and `cuda`/`metal`/`vulkan` cargo features.
- **`llm-chain-tools`**: `Tool::invoke` returns `Result<Value, ToolError>`;
  robust fenced-code-block extraction from model output;
  `gen_invoke_function!` to cut per-tool boilerplate.
- `thiserror` upgraded 1 → 2 across the workspace.

### Removed

- `async-trait` dependency (native async in traits).
- `dynfmt` dependency (custom fallible formatter).
- `llm-chain-llama-sys` vendored bindings and the `llama.cpp` submodule.
- OpenAI `seed` request option (deprecated upstream).

## [0.1.4] - 2023-04-10

Last release of the original 2023 line. See the git history for details.

[0.2.0]: https://github.com/sobelio/llm-chain/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/sobelio/llm-chain/releases/tag/v0.1.4
