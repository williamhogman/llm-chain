# Setting up a project with llm-chain

:::tip

Having problems? Don't worry, reach out on [discord](https://discord.gg/kewN9Gtjt2) and we will help you out.

:::

Welcome to llm-chain, a Rust library designed to simplify working with large language models (LLMs) and help you create powerful applications. In this tutorial, we'll walk you through installing Rust, setting up a new project, and getting started with llm-chain.

## Installing Rust

To begin, you'll need to install Rust on your machine. We recommend using [rustup](https://rustup.rs/), the official Rust toolchain manager, to ensure you have the latest version and can manage your installations easily.

You need **Rust 1.85.0 or higher** — llm-chain uses the Rust 2024 edition. If you see errors about unstable features or the edition, please update your Rust version with `rustup update`.

1. Follow the instructions on the [rustup website](https://rustup.rs/) to install Rust.

## Creating a New Rust Project

Now that you have Rust installed, it's time to create a new Rust project. Run the following command to set up a new binary project:

```bash
cargo new --bin my-llm-project
```

This command will create a new directory called `my-llm-project` with the necessary files and directories for a Rust project.

## Installing llm-chain

With your Rust project set up, it's time to add llm-chain as a dependency. To do this, run the following command:

```bash
cd my-llm-project
cargo add llm-chain
```

This will add llm-chain to your project's `Cargo.toml` file.

## Choosing a Driver

llm-chain supports multiple drivers for working with different LLMs — OpenAI (and Azure), Anthropic, Gemini (and Vertex AI), AWS Bedrock, Ollama, and local llama.cpp. All drivers share the same `Step`/`Executor` architecture, so what you learn in this tutorial applies to every provider.

For ease of use and getting started quickly, we'll be using the OpenAI driver in this tutorial. To install it, run:

```bash
cargo add llm-chain-openai
```

In the next tutorial, we'll cover generating your first LLM output using the OpenAI driver.
