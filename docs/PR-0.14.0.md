# Upstream PR draft — copy-paste ready

**Base:** `sobelio/llm-chain` `main`
**Head:** this repository's `main`
**Suggested merge strategy:** see "Merge strategy" below — this branch does
**not** descend from current upstream `main`.

---

## Honest lineage disclosure (read first)

This branch was forked from `51cce6b` ("Add support for serializing and
deserializing chains (#14)", April 2023 — the 0.1.x line). Upstream `main`
is **352 commits ahead** of that baseline: the 2023–2024 development line
that restructured the workspace into `crates/*`, added the
`prompt!`/`executor!` macros and unified options, SSE streaming,
conversation chains, vector stores (qdrant/milvus/hnsw), and the
local/mock/sagemaker/gemma drivers, publishing through **0.13.0**
(November 2023). Upstream's last commit is from **October 2024**; the
project has been dormant since.

A textual merge or rebase between the two lines is not meaningful — they
share almost no file in common (different layout, different core traits,
different everything). This PR therefore proposes the branch as a
**replacement line, versioned 0.14.0** (above the last published 0.13.0 so
crates.io resolution moves forward), not as an incremental diff on top of
`main`.

### What this branch does NOT carry over from 0.13.x

Called out explicitly so nobody discovers it post-merge: SSE streaming, the
`prompt!`/`executor!` macros, the unified `Options` map, conversation
chains, and the `local`, `macros`, `sagemaker-endpoint`,
`gemma(-sys)`, `qdrant`, `milvus` and `hnsw` crates. (`llm-chain-mock` *is*
carried over, rebuilt on the 0.14 architecture.)

`docs/MIGRATION-0.14.md` maps each to its 0.14 equivalent (or its absence).
Streaming is the top follow-up candidate; the vector-store crates are best
revisited against 2026-era store APIs rather than ported.

## Title

```
Modernize llm-chain for 2026: edition 2024, native async, 5 new providers, hyperscalers (v0.14.0)
```

## Body

This PR brings `llm-chain` up to the current state of the Rust and LLM
ecosystems and prepares the 0.14.0 release. It is a ground-up modernization
of the 0.1.x architecture, proposed as a replacement for the dormant 0.13.x
line (see the lineage disclosure in `docs/PR-0.14.0.md`).

### Why

Both existing lines are years out of date. The 0.13.x line (last commit
October 2024, last publish November 2023) predates native async in traits,
the current GGUF tooling, and every major LLM API surface in use today; its
dependencies (`async-trait`, `serde_yaml`, `thiserror` 1, hand-rolled
llama.cpp FFI with a git submodule) are deprecated or unmaintained.

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

**Mock (`llm-chain-mock`)**
- Rebuilt on the 0.14 architecture: Echo / Scripted / Failing behaviours with
  call recording, for unit-testing chains without network access.

**Tools (`llm-chain-tools`)**
- Fallible `Tool` trait with `ToolError`; robust fenced-code-block
  extraction; `gen_invoke_function!` macro.


**Security**
- All credentials held as `secrecy::SecretString` — redacted `Debug`,
  zeroized on drop.
- Uniform `.status()` / `.is_rate_limit()` on provider errors for
  retry/backoff.

**Repository & workspace**
- Crates laid out under `crates/*`, matching upstream's layout; `Cargo.lock`
  committed; `SECURITY.md` and a modern cargo-deny v2 `deny.toml` added,
  enforced by a CI audit job.

**Website**
- The Docusaurus site (`website/`, docs.llm-chain.xyz) ported and upgraded
  2.4 → 3.9 (React 19, MDX v3): every docs page rewritten for the 0.14 API,
  a new Providers page, the historical blog preserved plus a 0.14 release
  post, and GitHub Pages deploy / PR-check workflows.

**CI/CD & docs**
- fmt + clippy `-D warnings` + Ubuntu/macOS test matrix + MSRV (1.85) job +
  rustdoc `-D warnings`; tag-triggered crates.io release workflow.
- Rewritten README, `CHANGELOG.md`, migration guide
  (`docs/MIGRATION-0.14.md`, covering both 0.1.x and 0.13.x starting
  points), release checklist (`docs/RELEASING.md`), and runnable examples
  for every provider.


### Numbers

- Against the `51cce6b` fork point: 210 files changed, ~34,400 insertions,
  ~2,500 deletions (including the ported website and the committed
  `Cargo.lock` / `package-lock.json` lockfiles).
- 23 test suites, 219 unit/integration/doc tests — including mock-API wire
  format suites for Anthropic/Gemini/Bedrock/Ollama (no API keys needed) and
  a real GGUF end-to-end inference test.


### Versioning

The workspace is versioned **0.14.0** — above the last published 0.13.0 —
so the crates.io line moves forward even though this branch does not descend
from it. `CHANGELOG.md` documents the lineage split explicitly.

### Breaking changes

Breaking from both 0.1.x and 0.13.x (fallible formatting, typed errors,
per-provider typed options, module rename, new model enums, GGUF-only
llama). Full upgrade guide for both starting points:
[`docs/MIGRATION-0.14.md`](MIGRATION-0.14.md).

### Test plan

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

All green on Linux (glibc) with stable and 1.85 (MSRV).

## Merge strategy

Because the branch does not descend from current `main`, pick one:

1. **Replace `main` (recommended if maintainers agree the 0.13 line is
   retired):** merge with `-s ours` semantics reversed — i.e. land this tree
   as the new `main` (GitHub: merge the PR with "squash", accepting the
   full-tree diff), tag `v0.14.0`. History of the 0.13 line remains
   reachable via existing tags.
2. **Side-by-side:** land as a long-lived `next` branch, publish 0.14.0 from
   it, and switch `main` once consumers migrate.
3. **Port onto `main`:** we re-apply the work as a commit series on top of
   current HEAD. Honest cost estimate: near-total rewrite of every file
   touched, since the trees share almost nothing — this is the least
   practical option and effectively reproduces option 1 with more steps.

Suggested logical commit series if rebasing/re-authoring instead of
squashing:
1. Workspace: edition 2024, resolver 3, MSRV 1.85, shared deps/lints, `crates/*` layout
2. Core: native async traits, typed errors, fallible templates, model-id macros
3. OpenAI: async-openai 0.41, 2026 models, options, Azure
4. Anthropic: new crate
5. Gemini: new crate + Vertex AI
6. Ollama: new crate
7. Bedrock: new crate
8. LLaMA: rewrite on llama-cpp-2
9. Mock: rebuilt on the 0.14 architecture
10. Tools: fallible trait, extraction hardening
11. Website: Docusaurus 3 port, docs rewritten for 0.14
12. CI/CD, docs, changelog, release tooling


### Notes for reviewers

- `.lovable/`, `package.json`, `scripts/dev-preview.sh`,
  `scripts/doc-index.html` and `docs/site.tar.gz` are development-sandbox /
  docs-preview scaffolding. They are inert for library consumers and are not
  included in any published crate package. Happy to drop them from the PR if
  preferred.
- The workspace uses upstream's `crates/*` layout, and the 0.13-era
  Docusaurus website is carried forward under `website/` (upgraded to
  Docusaurus 3, docs rewritten for 0.14, domain and blog history preserved).
