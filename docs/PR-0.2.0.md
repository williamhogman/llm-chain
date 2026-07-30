# Upstream PR draft — copy-paste ready

**Base:** `sobelio/llm-chain` `main` (last upstream commit `51cce6b`,
"Add support for serializing and deserializing chains (#14)")
**Head:** this repository's `main`
**Suggested merge strategy:** squash merge (the branch contains WIP
checkpoint commits), or rebase into the logical series listed at the bottom.

---

## Title

```
Modernize llm-chain for 2026: edition 2024, native async, 5 new providers, hyperscalers (v0.2.0)
```

## Body

This PR brings `llm-chain` from its April 2023 state (0.1.4) up to the
current state of the Rust and LLM ecosystems, and prepares the 0.2.0 release.

### Why

The 0.1.x line predates native async in traits, the GGUF model format, and
every major LLM API surface in use today. Its dependencies (`async-openai`
0.10, `serde_yaml`, `dynfmt`, `thiserror` 1, hand-rolled llama.cpp FFI with a
git submodule) are deprecated, unmaintained, or years out of date.

### What

**Toolchain & workspace**
- Rust 2024 edition, resolver 3, MSRV 1.85, workspace-level dependencies and
  lints, thin LTO for release builds.

**Core (`llm-chain`)**
- Native `async fn` in traits via `trait-variant` (drops `async-trait`);
  `Send` futures guaranteed.
- Typed errors end to end: fallible prompt formatting
  (`PromptTemplateError`, replaces unmaintained `dynfmt`), `Step::Error`,
  `Executor::Error`, and `ChainError` from chain runs. No panics.
- `serde_yaml` → `serde_yaml_ng`; optional `async` feature for tokio file I/O.
- New `impl_model_id!` / `impl_model_id_serde!` macros — every driver's
  `Model` type is generated from a one-line-per-model id table with
  `Display`/`FromStr`/serde/`KNOWN_IDS`.
- Quality-of-life: `Parameters` and `sequential::Chain` implement the
  standard collection traits; `PromptTemplate` gains `Display`/`Eq`/`Hash`
  and `{{` escapes.

**OpenAI (`llm-chain-openai`)**
- `async-openai` 0.10 → 0.41; `chatgpt` module renamed `chat` (deprecated
  alias kept).
- July 2026 model lineup (GPT-5.6 sol/terra/luna, 5.4, 5.2 Pro, 4.1, 4o,
  o-series) plus custom ids; `Developer` role; `Options` with
  temperature/top_p/max_completion_tokens/reasoning_effort/verbosity/
  response_format.
- **Azure OpenAI** support: `AzureExecutor` on the OpenAI-compatible
  `/openai/v1` surface, API-key or Microsoft Entra ID auth.

**New provider crates** — all on a minimal built-in client
(`reqwest` 0.13 + rustls, zero heavyweight SDKs), all following the same
`Step`/`Executor`/`Options` architecture:
- `llm-chain-anthropic` — Messages API, Claude 5 generation, extended
  thinking budgets and `effort`.
- `llm-chain-gemini` — `generateContent`, Gemini 3.x/2.5, thinking
  level/budget, JSON mode; **Vertex AI** routes (OAuth2 and Express Mode).
- `llm-chain-bedrock` — **AWS Bedrock** Converse API: one wire format for
  Claude, Nova, Llama, Mistral, …; Bedrock API-key auth; regional endpoints.
- `llm-chain-ollama` — local (`OLLAMA_HOST`) or Ollama cloud; think levels;
  JSON schema outputs; generation timings.

**LLaMA (`llm-chain-llama`)**
- Rewritten on the maintained `llama-cpp-2` bindings: GGUF models, modern
  sampling pipeline, stop sequences, UTF-8-safe detokenization,
  `cuda`/`metal`/`vulkan` features. The vendored `sys` crate and the
  `llama.cpp` submodule are removed.

**Tools (`llm-chain-tools`)**
- Fallible `Tool` trait with `ToolError`; robust fenced-code-block
  extraction; `gen_invoke_function!` macro.

**Security**
- All credentials held as `secrecy::SecretString` — redacted `Debug`,
  zeroized on drop.
- Uniform `.status()` / `.is_rate_limit()` on provider errors for
  retry/backoff.

**CI/CD & docs**
- fmt + clippy `-D warnings` + Ubuntu/macOS test matrix + MSRV (1.85) job +
  rustdoc `-D warnings`; tag-triggered crates.io release workflow.
- Rewritten README, `CHANGELOG.md`, migration guide
  (`docs/MIGRATION-0.2.md`), release checklist (`docs/RELEASING.md`), and
  runnable examples for every provider.

### Numbers

- 133 files changed, ~10,400 insertions, ~2,100 deletions.
- 21 test suites, ~180 unit/integration/doc tests — including mock-API wire
  format suites for Anthropic/Gemini/Bedrock/Ollama (no API keys needed) and
  a real GGUF end-to-end inference test.

### Breaking changes

0.1.x → 0.2.0 is breaking (fallible formatting, typed errors, module rename,
new model enums, GGUF-only llama). Full upgrade guide:
[`docs/MIGRATION-0.2.md`](MIGRATION-0.2.md).

### Test plan

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

All green on Linux (glibc) with stable and 1.85 (MSRV).

### Notes for reviewers

- `.lovable/`, `package.json`, `scripts/dev-preview.sh`,
  `scripts/doc-index.html` and `docs/site.tar.gz` are development-sandbox /
  docs-preview scaffolding. They are inert for library consumers and are not
  included in any published crate package. Happy to drop them from the PR if
  preferred.
- Suggested logical commit series if rebasing instead of squashing:
  1. Workspace: edition 2024, resolver 3, MSRV 1.85, shared deps/lints
  2. Core: native async traits, typed errors, fallible templates, model-id macros
  3. OpenAI: async-openai 0.41, 2026 models, options, Azure
  4. Anthropic: new crate
  5. Gemini: new crate + Vertex AI
  6. Ollama: new crate
  7. Bedrock: new crate
  8. LLaMA: rewrite on llama-cpp-2
  9. Tools: fallible trait, extraction hardening
  10. CI/CD, docs, changelog, release tooling
