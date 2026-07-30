# llm-chain-tools 🛠️

`llm-chain-tools` is an extension for the `llm-chain` crate, providing a collection of tools that can be used to give Large Language Models (LLMs) access to various utilities, such as running Bash commands on your computer or performing web searches.

## Examples 💡

To help you get started, here is an example demonstrating how to use `llm-chain-tools` with `llm-chain`. You can find more examples in the [examples folder](/llm-chain-tools/examples) in the repository.

```rust
use llm_chain::Parameters;
use llm_chain_tools::create_tool_prompt_segment;
use llm_chain_tools::tools::BashTool;
use llm_chain_tools::ToolCollection;
use std::boxed::Box;
// A simple example generating a prompt with some tools.

fn main() {
    let tool_collection = ToolCollection::new(vec![Box::new(BashTool::new())]);
    let prompt =
        create_tool_prompt_segment(&tool_collection, "Please perform the following task: {}")
            .unwrap();
    println!(
        "{}",
        prompt
            .format(&Parameters::new_with_text(
                "Find the file GOAL.txt and tell me its content."
            ))
            .unwrap()
    );
}

```

## Features 🌟

- **Native tool calling**: Bridge any `ToolCollection` into the first-party tool-calling APIs of OpenAI, Anthropic, Gemini, Bedrock and Ollama — `tool_schemas()` generates a JSON Schema per tool, `invoke_json()` runs the calls the model makes (see the `native_agent` example).
- **Prompt integration**: Or describe the tools as YAML in the prompt for models without tool training, and parse replies with `process_chat_input`.
- **Tool management**: Easily create and integrate a collection of tools that LLMs can use to perform various tasks.
- **Pre-defined tools**: A submodule providing a variety of pre-defined tools for common tasks, ready for use.
- **Extensibility**: Designed with extensibility in mind, making it easy to integrate additional tools as needed.

## Getting Started 🚀

To start using `llm-chain-tools`, add it as a dependency in your Cargo.toml:

```toml
[dependencies]
llm-chain = "0.14.0"
llm-chain-openai = "0.14.0"
llm-chain-tools = "0.14.0"
```

Then, refer to the documentation and examples to learn how to create and manage tools, integrate them into prompts, and more.
