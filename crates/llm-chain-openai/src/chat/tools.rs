//! Helpers for native tool calling (function calling) with the Chat
//! Completions API.
//!
//! The flow is a loop:
//!
//! 1. Attach tools to a step with [`Options::with_tools`](super::Options::with_tools).
//! 2. Execute the step. When the model calls tools, the response message's
//!    `tool_calls` is non-empty (see [`function_calls`]).
//! 3. Append the assistant turn ([`assistant_tool_calls_message`]) and one
//!    [`tool_result_message`] per call to `request.messages`, then execute the
//!    request again for the model's next turn.
//!
//! ```
//! use llm_chain_openai::chat::function_tool;
//!
//! let tool = function_tool(
//!     "get_weather",
//!     "Get the current weather for a city.",
//!     serde_json::json!({
//!         "type": "object",
//!         "properties": {"city": {"type": "string"}},
//!         "required": ["city"]
//!     }),
//! );
//! ```

use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionResponseMessage, ChatCompletionTool,
    ChatCompletionTools, CreateChatCompletionResponse, FunctionObject,
};

/// Builds a function tool from a name, a description and a JSON Schema for
/// its parameters.
///
/// Use [`ToolCollection::tool_schemas`](https://docs.rs/llm-chain-tools) to
/// bridge an existing `llm-chain-tools` collection into function tools.
pub fn function_tool<N: Into<String>, D: Into<String>>(
    name: N,
    description: D,
    parameters: serde_json::Value,
) -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: name.into(),
            description: Some(description.into()),
            parameters: Some(parameters),
            strict: None,
        },
    })
}

/// The function calls the model made this turn, in order.
///
/// Looks at the first choice's message and skips non-function (custom) tool
/// calls. Non-empty exactly when the model wants tools run; answer each call
/// with a [`tool_result_message`].
pub fn function_calls(
    response: &CreateChatCompletionResponse,
) -> impl Iterator<Item = &ChatCompletionMessageToolCall> {
    response
        .choices
        .first()
        .and_then(|choice| choice.message.tool_calls.as_deref())
        .unwrap_or_default()
        .iter()
        .filter_map(|call| match call {
            ChatCompletionMessageToolCalls::Function(call) => Some(call),
            ChatCompletionMessageToolCalls::Custom(_) => None,
        })
}

/// Converts a response message into the assistant request message that echoes
/// its tool calls, preserving any text content.
///
/// The API requires this assistant turn to precede the tool results when
/// continuing a tool-calling conversation.
pub fn assistant_tool_calls_message(
    message: &ChatCompletionResponseMessage,
) -> ChatCompletionRequestMessage {
    ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
        content: message
            .content
            .clone()
            .map(ChatCompletionRequestAssistantMessageContent::Text),
        tool_calls: message.tool_calls.clone(),
        ..Default::default()
    })
}

/// Builds the tool message that answers one tool call.
///
/// `tool_call_id` must be the id of the call being answered; `content` is the
/// tool's result serialized as text (JSON is fine).
pub fn tool_result_message<I: Into<String>, C: Into<String>>(
    tool_call_id: I,
    content: C,
) -> ChatCompletionRequestMessage {
    ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage {
        content: ChatCompletionRequestToolMessageContent::Text(content.into()),
        tool_call_id: tool_call_id.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_tools_serialize_on_the_wire_format() {
        let tool = function_tool(
            "get_weather",
            "Get the current weather for a city.",
            serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }),
        );
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "get_weather");
        assert_eq!(
            json["function"]["parameters"]["properties"]["city"]["type"],
            "string"
        );
    }

    #[test]
    fn function_calls_extracts_calls_from_the_first_choice() {
        let response: CreateChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1768000000,
            "model": "gpt-5.6-terra",
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Stockholm\"}"}
                    }]
                }
            }]
        }))
        .unwrap();
        let calls: Vec<_> = function_calls(&response).collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "Stockholm");

        let assistant = assistant_tool_calls_message(&response.choices[0].message);
        let json = serde_json::to_value(&assistant).unwrap();
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["tool_calls"][0]["id"], "call_abc");
    }

    #[test]
    fn tool_result_messages_carry_the_call_id() {
        let message = tool_result_message("call_abc", "8°C, cloudy");
        let json = serde_json::to_value(&message).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": "8°C, cloudy"
            })
        );
    }
}
