# Migrating from 0.1.x to 0.2.0

0.2.0 is a breaking release that modernizes every crate in the workspace.
This guide covers each change you will hit when upgrading, with before/after
snippets.

## Requirements

- **Rust 1.85+** (the workspace uses the 2024 edition).
- Building `llm-chain-llama` now compiles llama.cpp from source via
  `llama-cpp-2`: you need `cmake`, a C/C++ toolchain and `libclang`
  (`sudo apt-get install build-essential cmake libclang-dev`).

## Core (`llm-chain`)

### Prompt formatting is fallible

`PromptTemplate::format` returns `Result<String, PromptTemplateError>`
instead of panicking on missing parameters:

```rust
// 0.1.x
let text = template.format(&parameters);

// 0.2.0
let text = template.format(&parameters)?;
```

Literal braces are escaped by doubling: `{{` renders `{`, `}}` renders `}`.

### Chains return typed errors

`Chain::run` (sequential and map-reduce) returns
`Result<Output, ChainError>` instead of a bare output:

```rust
// 0.1.x
let res = chain.run(Parameters::new(), exec).await;

// 0.2.0
let res = chain.run(Parameters::new(), &exec).await?;
```

Executors are now passed by reference.

### Traits use native async

The `Executor` trait no longer uses `#[async_trait]`. If you implement your
own driver, drop the attribute and add typed errors:

```rust
// 0.2.0
impl Step for MyStep {
    type Output = MyPrompt;
    type Error = MyFormatError;                    // new
    fn format(&self, p: &Parameters) -> Result<Self::Output, Self::Error> { ... }
}

#[trait_variant::make(Send)] // via llm_chain::traits::Executor
impl Executor for MyExecutor {
    type Step = MyStep;
    type Output = MyResponse;
    type Error = MyApiError;                       // new
    async fn execute(&self, input: MyPrompt) -> Result<MyResponse, MyApiError> { ... }
    ...
}
```

### YAML serialization

`serde_yaml` was replaced by `serde_yaml_ng`. The on-disk format is
unchanged; only the dependency changed. File I/O helpers are gated behind the
`serialization` (sync) and `async` (tokio file I/O) features.

## OpenAI (`llm-chain-openai`)

### Module rename: `chatgpt` → `chat`

```rust
// 0.1.x
use llm_chain_openai::chatgpt::{Executor, Model, Role, Step};

// 0.2.0
use llm_chain_openai::chat::{Executor, Model, Role, Step};
```

A deprecated `chatgpt` alias re-exports the `chat` module, so old imports
compile with a warning.

### Models

`Model::ChatGPT3_5Turbo` and friends are gone. The enum now covers the
GPT-5.6 family through the GPT-4 era, with a catch-all:

```rust
Model::default()                 // gpt-5.6-terra
Model::Gpt56Sol                  // strongest reasoning
Model::Other("my-fine-tune".into())
"gpt-5.6-luna".parse::<Model>()  // infallible; unknown ids -> Other
```

### Steps and options

`Step::new` takes a model plus `(Role, template)` pairs, and per-step
request options live in `Options`:

```rust
let step = Step::new(
    Model::default(),
    [
        (Role::Developer, "You are a helpful assistant."),
        (Role::User, "Tell me about {topic}."),
    ],
)
.with_options(
    Options::new()
        .with_temperature(0.2)
        .with_reasoning_effort(ReasoningEffort::Medium),
);
```

The `seed` option was removed (deprecated by OpenAI). `Role::Developer` is
the recommended replacement for `System` when targeting reasoning models.

### Azure OpenAI

```rust
let exec = AzureExecutor::azure("my-resource", api_key);
// or Entra ID:
let exec = AzureExecutor::azure_with_entra_token("my-resource", token);
```

## LLaMA (`llm-chain-llama`)

The crate now runs on `llama-cpp-2` and loads **GGUF** models (the old
ggml `model.bin` format is not supported):

```rust
let exec = Executor::new("models/llama-3.2-1b-q4_k_m.gguf")?;
// or with config:
let exec = Executor::with_config("model.gguf", ModelConfig::new().with_n_ctx(8192))?;
```

Sampling options (top-k, top-p, temperature, repetition penalties, seed) and
stop sequences are set per step. GPU offload is behind the `cuda`, `metal`
and `vulkan` features.

## Tools (`llm-chain-tools`)

`Tool::invoke` returns `Result<serde_json::Value, ToolError>`:

```rust
// 0.1.x
fn invoke(&self, input: Value) -> Value;

// 0.2.0
fn invoke(&self, input: Value) -> Result<Value, ToolError>;
```

## New providers

Anthropic, Gemini (+ Vertex AI), Bedrock and Ollama are new crates that
follow the exact same `Step` / `Executor` / `Options` shape — switching
providers is a matter of swapping imports. See the
[README](README.md) for a getting-started example per provider.
