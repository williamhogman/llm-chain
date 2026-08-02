---
id: dev-setup
title: Development Setup
sidebar_label: Development Setup
sidebar_position: 7
---

# Contributing to `llm-chain`

First of all, thank you for considering contributing to our project! 🎉 We are delighted to have you here and truly appreciate your interest in making our project even better. Your contributions and ideas are highly valued.

## Getting Started

1. Make your own fork of [`llm-chain`](https://github.com/sobelio/llm-chain).
2. `git clone` your fork to your local machine.
3. Follow the instructions on the [rustup website](https://rustup.rs/) to install Rust — you need **1.85.0 or newer** (edition 2024).
4. Install the native build dependencies used by `llm-chain-llama` (which compiles llama.cpp from source): `cmake`, a C/C++ toolchain and `libclang`. On Debian/Ubuntu: `sudo apt-get install build-essential cmake libclang-dev`. There are no git submodules to fetch.
5. Test that everything went well with `cargo test --workspace`.
6. Make your changes in a new branch, following the coding guidelines and best practices.
7. Before pushing, run the same checks CI runs:
   ```bash
   cargo fmt --all --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   ```
8. Commit and push your changes to your fork.
9. Open a pull request against the main repository. 🚀

## Repository layout

All crates live under `crates/` — the core `llm-chain` crate, one driver crate per provider (`llm-chain-openai`, `llm-chain-anthropic`, `llm-chain-gemini`, `llm-chain-bedrock`, `llm-chain-ollama`, `llm-chain-llama`), the testing `llm-chain-mock` crate and `llm-chain-tools`. This website lives in `website/`.

Driver crates follow a common module shape (`step.rs`, `executor.rs`, `options.rs`, `prompt.rs`, `types.rs`, `error.rs`) — when adding a provider, mirror an existing crate such as `llm-chain-anthropic`.

## Testing without API keys

The HTTP driver crates ship mock-API test suites that spin up a local server and assert on the exact wire format — no credentials needed. `llm-chain-llama`'s integration test uses a tiny GGUF model. `cargo test --workspace` runs everything offline.

## Before You Contribute

We are open to new ideas and contributions that align with the project's goals and vision. However, if you're planning on working on something significantly different from what's already in the project, we strongly recommend getting in touch with us before you start.

You can reach out to us by opening an issue, starting a discussion, or sending an email. This way, we can discuss your ideas, provide guidance, and ensure that your efforts are more likely to be merged into the project.
