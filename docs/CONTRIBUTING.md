# Contributing to `llm-chain`

First of all, thank you for considering contributing to our project! 🎉 We are delighted to have you here and truly appreciate your interest in making our project even better. Your contributions and ideas are highly valued.

## Getting Started

1. Fork the repository to your own account.
2. Clone your fork to your local machine.
3. Make your changes in a new branch, following the coding guidelines and best practices.
4. Commit and push your changes to your fork.
5. Open a pull request against the main repository.

## Development Requirements

- Rust 1.85 or later (the workspace uses the 2024 edition).
- Building `llm-chain-llama` compiles llama.cpp from source, which requires `cmake`, a C/C++ toolchain and `libclang` (`sudo apt-get install build-essential cmake libclang-dev` on Debian/Ubuntu).

Before opening a pull request, please make sure the same checks CI runs are green:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

If your change is user-visible, add a line to the `[Unreleased]`/upcoming
section of [`CHANGELOG.md`](/CHANGELOG.md). Maintainers cut releases
following [`docs/RELEASING.md`](/docs/RELEASING.md).



## Before You Contribute

We are open to new ideas and contributions that align with the project's goals and vision. However, if you're planning on working on something significantly different from what's already in the project, we strongly recommend getting in touch with us before you start.

You can reach out to us by opening an issue, starting a discussion, or sending an email. This way, we can discuss your ideas, provide guidance, and ensure that your efforts are more likely to be merged into the project.
