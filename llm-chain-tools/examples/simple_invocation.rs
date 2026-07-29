use llm_chain::{Parameters, traits::StepExt};
use llm_chain_openai::chatgpt::{Executor, Model, Role, Step};
use llm_chain_tools::ToolCollection;
use llm_chain_tools::create_tool_prompt_segment;
use llm_chain_tools::tools::BashTool;
use std::boxed::Box;
// A simple example generating a prompt with some tools.

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let tool_collection = ToolCollection::new(vec![Box::new(BashTool::new())]);
    let template =
        create_tool_prompt_segment(&tool_collection, "Please perform the following task: {}")
            .unwrap();
    let prompt = template
        .format(&Parameters::new_with_text(
            "Find the file GOAL.txt and tell me its content.",
        ))
        .unwrap();

    let exec = Executor::new_default();
    let chain = Step::new(
        Model::default(),
        [
            (
                Role::System,
                "You are an automated agent for performing tasks. Your output must always be YAML.",
            ),
            (Role::User, &prompt),
        ],
    )
    .to_chain();
    let res = chain.run(Parameters::new(), &exec).await.unwrap();
    let message_text = res
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default();
    println!("{}", &message_text);
    match tool_collection.process_chat_input(&message_text) {
        Ok(output) => println!("{}", output),
        Err(e) => println!("Error: {}", e),
    }
}
