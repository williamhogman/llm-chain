# llm-chain-llama 🦙

Welcome to LLM-Chain-LLaMa, a powerful and versatile driver for LLaMA-family models! This crate leverages the amazing [llama.cpp](https://github.com/ggml-org/llama.cpp) library through the maintained [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) bindings, making it simple and efficient to run LLaMA, Mistral, Qwen, Gemma and any other GGUF model fully offline in a Rust environment.

## Getting Started 🏁

To begin, you'll need a model in **GGUF** format — the standard format used by llama.cpp today. Thousands of ready-to-use quantized models are available on [Hugging Face](https://huggingface.co/models?library=gguf). Download one and point the executor at it:

```rust,no_run
use llm_chain::{traits::StepExt, Parameters};
use llm_chain_llama::{Executor, Step};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let exec = Executor::new("path/to/model.gguf")?;
    let chain = Step::new("The Colors of the Rainbow are (in order): ".into()).to_chain();
    let res = chain.run(Parameters::new(), &exec).await?;
    println!("{}", res);
    Ok(())
}
```

## Features 🌟

LLM-Chain-LLaMa is packed with all the features you need to harness the full potential of local models. Here's a glimpse of what's inside:

- Running chained LLaMA-style models in a Rust environment, taking your applications to new heights 🌄
- Support for any GGUF model: LLaMA, Mistral, Qwen, Gemma, Phi, and more
- GPU offloading via the `cuda`, `metal` and `vulkan` cargo features ⚡
- Modern sampling (top-k, top-p, temperature, repetition penalties) with reproducible seeds
- Prompts for working with `instruct` models, empowering you to easily build virtual assistants and amazing applications 🧙‍♂️

## Building 🛠️

Building this crate compiles llama.cpp from source, which requires `cmake` and a C/C++ toolchain (and `libclang` for the generated bindings). To enable GPU acceleration, activate the matching feature:

```toml
[dependencies]
llm-chain-llama = { version = "0.2", features = ["metal"] } # or "cuda" / "vulkan"
```

So gear up and dive into the fantastic world of LLM-Chain-LLaMa! Let the power of local models propel your projects to the next level. Happy coding, and enjoy the ride! 🎉🥳
