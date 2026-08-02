# Changelog

All notable changes to the `llm-chain` workspace are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and all crates in the workspace share a single version number and adhere to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.14.0] - 2026-07-30

A ground-up modernization of the entire workspace. Every crate was brought up
to the current state of the Rust and LLM ecosystems, and five new provider
crates were added.

**Lineage note.** This release was developed from the April 2023 tree
(commit `51cce6b`, the 0.1.x line) and is a from-scratch replacement of that
architecture. The 0.3.0–0.13.0 versions published from upstream's divergent
2023–2024 line (last publish: 0.13.0, November 2023; last commit: October
2024) are not ancestors of this release. Of that line's additions, streaming
has been reimplemented on the new architecture; the vector stores and the
`prompt!`/`executor!` macros are not carried over — see the changelog entry
for that line below. 0.14.0 is numbered above 0.13.0 so crates.io resolution
moves forward.

**Breaking release.** See [`docs/MIGRATION-0.14.md`](docs/MIGRATION-0.14.md)
for a step-by-step upgrade guide from both 0.1.x and 0.13.x.

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
- **First-party tool calling** — native function/tool-use support on every
  HTTP provider, plus a provider-neutral bridge in `llm-chain-tools`
  (`tool_schemas()` / `invoke_json()`).
- **Token-by-token streaming** — a unified `StreamingExecutor` trait across
  all five HTTP providers, with typed per-provider events and accumulators
  that rebuild the full response (SSE, NDJSON and AWS's binary event stream
  all decoded natively).
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
- **`llm-chain-mock`** — deterministic in-process executor for testing
  chains without network access: Echo, Scripted and Failing behaviours, with
  every executed prompt recorded and available via `Executor::calls()`.
  (Rebuilt on the 0.14 architecture; the 0.13-era crate of the same name is
  not an ancestor.)


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
- `traits::StreamingExecutor`: the shared contract for token-by-token
  streaming — `execute_stream()` resolves to a `BoxStream` of typed
  per-provider events once the model starts answering, and `text_delta()`
  extracts answer text provider-agnostically.
- `streaming` module: sans-IO wire decoders shared by the drivers —
  `SseDecoder` (Server-Sent Events), `NdjsonDecoder` (newline-delimited
  JSON), and the `FrameDecoder` trait with the `frames()` adapter for
  custom framings.
- `async` cargo feature: async file I/O for chain serialization.

#### Providers

- **OpenAI**: July 2026 model lineup — the GPT-5.6 family (Sol, Terra, Luna;
  `gpt-5.6-terra` is the default), GPT-5.4, GPT-5.2 Pro, GPT-4.1, GPT-4o and
  the o-series — plus any custom/fine-tuned id. New `Options`
  (`temperature`, `top_p`, `max_completion_tokens`, `reasoning_effort`,
  `verbosity`, `response_format`). Crate-owned `Role` enum including
  `Developer` for reasoning models. `Executor::with_api_key`,
  `with_api_key_and_org`, and `with_base_url` for OpenAI-compatible servers
  (vLLM, OpenRouter, local proxies). Token usage merged across combined
  outputs.

- **Azure OpenAI**: `AzureExecutor` / `AzureV1Config` targeting the
  OpenAI-compatible `/openai/v1` surface — no `api-version` pinning — with
  both `api-key` and Microsoft Entra ID bearer auth.
- **Native tool calling on every HTTP provider**: OpenAI function tools,
  Anthropic `tool_use`/`tool_result` content blocks, Gemini function
  declarations, Bedrock Converse `toolConfig` and Ollama tools. Each driver
  gains `Options::with_tools` (and `with_tool_choice` where the API has one),
  response accessors for the calls the model made (`function_calls`,
  `tool_uses`, …) and continuation helpers for sending results back
  (`tool_result_message`, `with_tool_results`, `Message::tool`, …).
  `llm-chain-tools` bridges any `ToolCollection` into these APIs:
  `tool_schemas()` generates a JSON Schema per tool and `invoke_json()`
  executes the calls the model makes. Runnable `native_agent` example and a
  new website docs page.
- **Streaming on every HTTP provider**: each driver implements
  `StreamingExecutor` with events mirroring its wire protocol — OpenAI
  `chat.completion.chunk`s (with `stream_options.include_usage` on by
  default), Anthropic SSE events, Gemini `streamGenerateContent?alt=sse`
  chunks, Bedrock's binary `converse-stream` event stream (CRC-validated
  decoder built in) and Ollama NDJSON chunks. Each driver ships a
  `ResponseAccumulator` folding the events back into its regular response
  type (text, reasoning, tool calls, usage), plus a runnable
  `*_streaming_generation` example.
- **All HTTP providers**: `.status()` and `.is_rate_limit()` on error types
  for uniform retry/backoff handling across providers (429s, Anthropic
  `overloaded_error`, Gemini `RESOURCE_EXHAUSTED`, Bedrock throttling),
  including exceptions raised mid-stream.

#### Tooling & CI

- Mock-API integration test suites for Anthropic, Gemini, Bedrock and Ollama
  asserting wire format, auth headers and error mapping — no API keys needed
  — including streaming suites that replay SSE, NDJSON and chunked binary
  event-stream responses (mid-stream exceptions included).
- End-to-end GGUF inference test for `llm-chain-llama` (tiny stories260K
  model).
- CI: fmt + clippy (`-D warnings`) + multi-OS test matrix (Ubuntu, macOS) +
  MSRV (1.85) verification + rustdoc with `-D warnings`.
- Tag-triggered release workflow publishing all crates to crates.io in
  dependency order.
- Workspace reorganized into the upstream `crates/*` layout; `Cargo.lock`
  committed; `SECURITY.md` and a cargo-deny v2 `deny.toml` added and
  enforced by a CI audit job (advisories, bans, licenses, sources).
- Documentation website (`website/`, docs.llm-chain.xyz) ported to
  Docusaurus 3 (React 19, MDX v3): every docs page rewritten for the 0.14
  API, a new Providers page, the historical blog preserved plus a 0.14
  release post, and GitHub Pages deploy / PR-check workflows.


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
  `gen_invoke_function!` to cut per-tool boilerplate. The `description`
  module is now public, and `ToolSchema` / `ToolCollection::tool_schemas` /
  `ToolCollection::invoke_json` bridge collections into native tool calling.
- `thiserror` upgraded 1 → 2 across the workspace.

### Removed

- `async-trait` dependency (native async in traits).
- `dynfmt` dependency (custom fallible formatter).
- `llm-chain-llama-sys` vendored bindings and the `llama.cpp` submodule.
- OpenAI `seed` request option (deprecated upstream).
- Relative to the 0.13.x line (not ancestors of this release, see the
  lineage note): the `prompt!`/`executor!` macros, unified `Options` map,
  conversation chains, and the `llm-chain-local`,
  `llm-chain-macros`, `llm-chain-sagemaker-endpoint`,
  `llm-chain-gemma(-sys)`, `llm-chain-qdrant`, `llm-chain-milvus` and
  `llm-chain-hnsw` crates. (`llm-chain-mock` is carried over, rebuilt on the
  0.14 architecture; streaming is reimplemented via `StreamingExecutor`.)

## [0.3.0]–[0.13.0] - 2023-2024

Published from upstream's divergent 2023–2024 line (crates/* layout,
`prompt!`/`executor!` macros, unified options, streaming, vector stores,
gemma/sagemaker/local drivers). Development stopped in October 2024; the
last published version is 0.13.0 (November 2023). See the upstream git
history for details. These versions are not ancestors of 0.14.0.

## [0.1.4] - 2023-04-10

Last release of the original 0.1.x line, from which 0.14.0 descends. See the
git history for details.

[0.14.0]: https://github.com/sobelio/llm-chain/compare/v0.1.4...v0.14.0
[0.13.0]: https://github.com/sobelio/llm-chain/releases/tag/v0.13.0
[0.3.0]: https://github.com/sobelio/llm-chain/releases/tag/v0.3.0
[0.1.4]: https://github.com/sobelio/llm-chain/releases/tag/v0.1.4
