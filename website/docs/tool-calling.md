---
id: tool-calling
title: Tool calling
sidebar_label: Tool calling
sidebar_position: 4
---

# Tool calling

`llm-chain-tools` gives models **tools** — actions they can trigger, like running a shell command or looking up data. There are two ways to wire them up:

1. **Native tool calling** (recommended): declare the tools through the provider's first-party tool-calling API. The model returns structured tool calls, you execute them, and send the results back. Supported by OpenAI, Anthropic, Gemini, Bedrock and Ollama.
2. **Prompt-based tools**: describe the tools in the prompt as YAML and parse the model's YAML reply. Works with any model, including local llama.cpp models without tool training.

## Defining a tool

A tool implements the `Tool` trait: typed input and output structs, a `Describe` implementation for each, and a description that tells the model what the tool does.

```rust
use llm_chain_tools::{Describe, Format, Tool, ToolDescription, ToolError};
use llm_chain_tools::gen_invoke_function;
use serde::{Deserialize, Serialize};

struct WeatherTool;

#[derive(Serialize, Deserialize)]
struct WeatherInput {
    city: String,
}

#[derive(Serialize, Deserialize)]
struct WeatherOutput {
    forecast: String,
}

impl Describe for WeatherInput {
    fn describe() -> Format {
        vec![("city", "The city to fetch the forecast for.").into()].into()
    }
}

impl Describe for WeatherOutput {
    fn describe() -> Format {
        vec![("forecast", "The weather forecast.").into()].into()
    }
}

impl WeatherTool {
    fn invoke_typed(&self, input: &WeatherInput) -> Result<WeatherOutput, ToolError> {
        Ok(WeatherOutput {
            forecast: format!("Sunny in {}", input.city),
        })
    }
}

impl Tool for WeatherTool {
    gen_invoke_function!();
    fn description(&self) -> ToolDescription {
        ToolDescription::new(
            "WeatherTool",
            "Fetches the weather forecast for a city.",
            "Use this when asked about the weather.",
            WeatherInput::describe(),
            WeatherOutput::describe(),
        )
    }
}
```

The crate ships `BashTool`, `PythonTool` and `ExitTool` ready-made.

## Native tool calling

`ToolCollection::tool_schemas()` bridges every tool into a provider-neutral `ToolSchema` — name, description and a JSON Schema for the arguments — which maps directly onto each provider's native tool definition:

| Provider | Native definition |
| --- | --- |
| `llm-chain-openai` | `function_tool(&s.name, &s.description, s.parameters)` |
| `llm-chain-anthropic` | `ToolDefinition::new(&s.name, &s.description, s.parameters)` |
| `llm-chain-gemini` | `FunctionDeclaration::new(&s.name, &s.description, s.parameters)` |
| `llm-chain-bedrock` | `ToolSpec::new(&s.name, &s.description, s.parameters)` |
| `llm-chain-ollama` | `Tool::function(&s.name, &s.description, s.parameters)` |

When the model calls a tool, `ToolCollection::invoke_json` executes it with the JSON arguments the model supplied and returns a JSON result to send back. The full agent loop with OpenAI:

```rust
use llm_chain::Parameters;
use llm_chain::traits::{Executor as _, Step as _};
use llm_chain_openai::chat::{
    Executor, Model, Options, Role, Step, assistant_tool_calls_message, function_calls,
    function_tool, tool_result_message,
};
use llm_chain_tools::ToolCollection;
use llm_chain_tools::tools::BashTool;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let tool_collection = ToolCollection::new(vec![Box::new(BashTool::new())]);
    let tools: Vec<_> = tool_collection
        .tool_schemas()
        .into_iter()
        .map(|s| function_tool(&s.name, &s.description, s.parameters))
        .collect();

    let exec = Executor::new_default();
    let step = Step::new(
        Model::default(),
        [
            (Role::Developer, "Use the available tools to complete the task."),
            (Role::User, "Figure out my IP address."),
        ],
    )
    .with_options(Options::new().with_tools(tools));
    let mut request = step.format(&Parameters::new()).unwrap();

    for _ in 0..5 {
        let response = exec.execute(request.clone()).await.unwrap();
        let calls: Vec<_> = function_calls(&response).cloned().collect();
        if calls.is_empty() {
            let answer = response.choices[0].message.content.as_deref().unwrap_or_default();
            println!("Assistant: {answer}");
            return;
        }
        request
            .messages
            .push(assistant_tool_calls_message(&response.choices[0].message));
        for call in calls {
            let content = serde_json::from_str(&call.function.arguments)
                .map_err(Into::into)
                .and_then(|input| tool_collection.invoke_json(&call.function.name, &input))
                .map(|output| output.to_string())
                .unwrap_or_else(|e| format!("error: {e}"));
            request.messages.push(tool_result_message(&call.id, content));
        }
    }
}
```

The same shape works everywhere: each provider crate has `Options::with_tools`, an accessor for the calls the model made (`function_calls`, `ConverseResponse::tool_uses`, `GenerateContentResponse::function_calls`, …) and a continuation helper for sending results back (`tool_result_message`, `ConverseRequest::with_tool_results`, `Message::tool`, …). Errors from `invoke_json` should be reported back to the model as the tool result so it can correct itself.

Runnable version: `cargo run --example native_agent` in `crates/llm-chain-tools`.

## Prompt-based tools

For models without native tool calling (many local GGUF models), describe the tools in the prompt and parse the reply:

```rust
use llm_chain::Parameters;
use llm_chain_tools::{ToolCollection, create_tool_prompt_segment};
use llm_chain_tools::tools::BashTool;

let tc = ToolCollection::new(vec![Box::new(BashTool::new())]);
let prompt = create_tool_prompt_segment(&tc, "Please perform the following task: {}")
    .unwrap()
    .format(&Parameters::new_with_text("Count the files in /tmp"))
    .unwrap();
// ...send `prompt` to any model, then:
// let result = tc.process_chat_input(&model_reply)?;
```

`process_chat_input` accepts bare YAML or fenced ``` ```yaml ``` blocks, invokes the named tool and returns its YAML output for the next turn. See `simple_agent.rs` in `crates/llm-chain-tools/examples` for the full loop.

## Choosing between them

| | Native | Prompt-based |
| --- | --- | --- |
| Reliability | High — structured, validated by the provider | Depends on the model following instructions |
| Model support | Tool-trained API models | Any model, including local GGUF |
| Parallel calls | Yes (provider-dependent) | One call per turn |
| Wire format | JSON Schema + JSON arguments | YAML in the prompt and reply |

Prefer native tool calling whenever the provider supports it.
