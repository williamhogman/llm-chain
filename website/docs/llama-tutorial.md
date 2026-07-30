---
id: llama-tutorial
title: "Tutorial: local inference with the LLaMA driver"
sidebar_label: LLaMA Tutorial
sidebar_position: 4
---

# Tutorial: Getting started with the LLaMA driver

In this tutorial, you will learn how to run an LLM **locally, in-process** using the `llm-chain-llama` driver, which embeds [llama.cpp](https://github.com/ggml-org/llama.cpp) via the maintained [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) bindings. If you wish to use the hosted drivers you can skip this part of the tutorial.

## Prerequisites

- Rust 1.85.0 or higher (`rustup update`)
- A C/C++ toolchain, `cmake`, and `libclang` — llama.cpp is compiled from source when the crate builds:

```bash
# Debian/Ubuntu
sudo apt-get install build-essential cmake libclang-dev

# macOS
xcode-select --install && brew install cmake
```

No git submodules, no Python conversion scripts, no Hugging Face account required — models are downloaded ready to run in the **GGUF** format.

## Step 1: Create a new Rust project

```bash
cargo new --bin llm-chain-demo
cd llm-chain-demo
cargo add llm-chain llm-chain-llama
cargo add tokio --features full
```

## Step 2: Download a GGUF model

GGUF is the standard single-file model format for llama.cpp. Thousands of pre-quantized models are available — for example Llama 3.2 1B:

```bash
mkdir -p models
curl -L -o models/llama-3.2-1b-q4_k_m.gguf \
  "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
```

Any GGUF model works: Llama, Mistral, Qwen, Gemma, Phi and more. `Q4_K_M` is a good default quantization — a solid quality/size trade-off.

## Step 3: Run inference

Replace the contents of `src/main.rs`:

```rust
use llm_chain::{Parameters, traits::StepExt};
use llm_chain_llama::{Executor, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let exec = Executor::new("models/llama-3.2-1b-q4_k_m.gguf")?;
    let chain = Step::new("The colors of the rainbow are (in order): ".into()).to_chain();
    let res = chain.run(Parameters::new(), &exec).await?;
    println!("{}", res);
    Ok(())
}
```

Then run it:

```bash
cargo run --release
```

The first build takes a few minutes while llama.cpp compiles; after that it's cached.

## Step 4: Tune the model and sampling

Context window and GPU offload are configured per executor:

```rust
use llm_chain_llama::{Executor, ModelConfig};

let exec = Executor::with_config(
    "models/llama-3.2-1b-q4_k_m.gguf",
    ModelConfig::new().with_n_ctx(8192),
)?;
```

Sampling is configured per step — top-k, top-p, temperature, repetition penalties, deterministic seeds and stop sequences:

```rust
use llm_chain_llama::{LlamaConfig, Step};

let step = Step::new_with_config(
    "Q: What is the capital of France?\nA:".into(),
    Some(LlamaConfig {
        temp: Some(0.7),
        top_p: Some(0.95),
        stop_sequence: Some("\n".into()),
        ..LlamaConfig::default()
    }),
);
```


## GPU acceleration

Enable the backend matching your hardware in `Cargo.toml`:

```toml
llm-chain-llama = { version = "0.14", features = ["metal"] }  # or "cuda", "vulkan"
```

With GPU offload enabled, layers are pushed to the accelerator via `ModelConfig::with_n_gpu_layers`.

That's it — the same `Step`/`Executor`/chain code you use with the hosted providers now runs entirely on your own machine.
