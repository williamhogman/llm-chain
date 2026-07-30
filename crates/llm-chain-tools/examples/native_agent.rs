//! An agent that uses native (first-party) tool calling instead of YAML
//! prompting.
//!
//! The tools are declared to the model through the API's own tool-calling
//! surface: [`ToolCollection::tool_schemas`] bridges each tool's description
//! into a JSON Schema, and [`ToolCollection::invoke_json`] runs the calls the
//! model makes. Compare with `simple_agent.rs`, which prompts for YAML and
//! parses the reply itself.
//!
//! Run with `OPENAI_API_KEY` set:
//!
//! ```sh
//! cargo run --example native_agent
//! ```

use llm_chain::Parameters;
use llm_chain::traits::{Executor as _, Step as _};
use llm_chain_openai::chat::{
    Executor, Model, Options, Role, Step, assistant_tool_calls_message, function_calls,
    function_tool, tool_result_message,
};
use llm_chain_tools::ToolCollection;
use llm_chain_tools::tools::BashTool;

const MAX_TURNS: usize = 5;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let tool_collection = ToolCollection::new(vec![Box::new(BashTool::new())]);

    // Bridge the collection into native function tools: one JSON Schema per tool.
    let tools: Vec<_> = tool_collection
        .tool_schemas()
        .into_iter()
        .map(|schema| function_tool(&schema.name, &schema.description, schema.parameters))
        .collect();

    let exec = Executor::new_default();
    let step = Step::new(
        Model::default(),
        [
            (
                Role::Developer,
                "You are an automated agent. Use the available tools to complete the task, \
                 then answer with a short summary.",
            ),
            (Role::User, "Figure out my IP address."),
        ],
    )
    .with_options(Options::new().with_tools(tools));
    let mut request = step.format(&Parameters::new()).unwrap();

    for _ in 0..MAX_TURNS {
        let response = exec.execute(request.clone()).await.unwrap();
        let calls: Vec<_> = function_calls(&response).cloned().collect();

        if calls.is_empty() {
            // No tool calls: the model is done and this is the final answer.
            let answer = response
                .choices
                .first()
                .and_then(|c| c.message.content.as_deref())
                .unwrap_or_default();
            println!("Assistant: {answer}");
            return;
        }

        // Echo the assistant's tool-call turn, then answer every call.
        let message = &response.choices[0].message;
        request.messages.push(assistant_tool_calls_message(message));
        for call in calls {
            println!("Tool call: {}({})", call.function.name, call.function.arguments);
            let result = serde_json::from_str(&call.function.arguments)
                .map_err(Into::into)
                .and_then(|input| tool_collection.invoke_json(&call.function.name, &input));
            let content = match result {
                Ok(output) => output.to_string(),
                // Report failures back to the model so it can correct itself.
                Err(e) => format!("error: {e}"),
            };
            println!("Tool result: {content}");
            request
                .messages
                .push(tool_result_message(&call.id, content));
        }
    }
    println!("Stopped after {MAX_TURNS} turns without a final answer.");
}
