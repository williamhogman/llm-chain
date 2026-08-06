---
id: streaming
title: Streaming
sidebar_label: Streaming
sidebar_position: 5
---

# Streaming

Every HTTP driver — OpenAI, Anthropic, Gemini, Bedrock, Ollama and the Lovable AI Gateway — can stream a response while the model generates it. Where `Executor::execute` buffers the whole response, `StreamingExecutor::execute_stream` resolves as soon as the model starts answering and yields typed events as they arrive.

```rust
use futures::StreamExt as _;
use llm_chain::Parameters;
use llm_chain::traits::{Step as _, StreamingExecutor as _};
use llm_chain_anthropic::messages::{Executor, Model, ResponseAccumulator, Role, Step};

let exec = Executor::new_default()?;
let step = Step::new(Model::default(), [(Role::User, "Tell a story about {{topic}}")]);
let request = step.format(&Parameters::new().with("topic", "a crab learning Rust"))?;

let mut stream = exec.execute_stream(request).await?;
let mut accumulator = ResponseAccumulator::new();
while let Some(event) = stream.next().await {
    let event = event?;
    if let Some(text) = Executor::text_delta(&event) {
        print!("{text}"); // live tokens
    }
    accumulator.apply(&event);
}
let response = accumulator.into_response().expect("stream ended early");
println!("\n[{} output tokens]", response.usage.output_tokens);
```

Swap the driver import and the same loop works against any provider — every crate ships a runnable `*_streaming_generation` example with exactly this shape.

## The `StreamingExecutor` trait

```rust
pub trait StreamingExecutor: Executor {
    type StreamEvent: Send + 'static;

    async fn execute_stream(&self, input: ...)
        -> Result<BoxStream<Self::StreamEvent, Self::Error>, Self::Error>;

    fn text_delta(event: &Self::StreamEvent) -> Option<Cow<'_, str>>;
}
```

- **`StreamEvent`** mirrors the provider's own streaming wire protocol, so nothing is lost in translation: reasoning deltas, tool-call fragments, usage reports and stop reasons all come through typed.
- **`text_delta`** extracts just the newly generated answer text from an event (and returns `None` for bookkeeping events and reasoning deltas), so provider-agnostic code can print tokens without knowing the event type.
- **Errors before any output** — bad credentials, unknown model, rate limits — are returned directly from `execute_stream`. **Errors during generation** (including mid-stream provider exceptions such as Bedrock throttling) are yielded inside the stream, which then ends. Both go through the driver's regular error type, so `.status()` and `.is_rate_limit()` work as usual.

## Accumulators

Streaming does not mean giving up the full response. Each driver ships a `ResponseAccumulator` that folds the events back into the driver's regular response type — the same one `execute` returns — so you can show live output *and* keep the final message, tool calls and token usage:

```rust
let mut accumulator = ResponseAccumulator::new();
// ... accumulator.apply(&event) for every event ...
let response = accumulator.into_response(); // Option<...>: None if the stream ended early
```

`into_response()` returns `None` when the stream ended before the provider finished the message, so an interrupted connection is detectable rather than silently truncated.

## What each driver speaks

| Driver | Wire protocol | Stream event type |
| --- | --- | --- |
| `llm-chain-openai` | SSE (`chat.completion.chunk`, with `stream_options.include_usage` on by default) | `CreateChatCompletionStreamResponse` |
| `llm-chain-anthropic` | SSE (`message_start` / `content_block_delta` / …) | `StreamEvent` |
| `llm-chain-gemini` | SSE (`streamGenerateContent?alt=sse`) | `GenerateContentResponse` |
| `llm-chain-bedrock` | AWS binary event stream (`application/vnd.amazon.eventstream`, CRC-validated) over `converse-stream` | `StreamEvent` |
| `llm-chain-ollama` | NDJSON | `ChatResponse` |
| `llm-chain-lovable` | SSE (OpenAI-style `chat.completion.chunk`) | `ChatChunk` |

Azure OpenAI and Vertex AI stream through the same executors as their base providers.

## Reasoning and tool calls stream too

Providers that expose reasoning (`thinking_delta` on Anthropic, thought parts on Gemini, `reasoning_delta` on Bedrock, `thinking` on Ollama, `reasoning` deltas on the Lovable Gateway) stream it as separate deltas, and tool-call arguments arrive as incremental fragments. `text_delta` deliberately skips both — use the event types directly for richer UIs, and let the accumulator reassemble complete tool calls for the [native tool-calling loop](tool-calling.md).

## Writing your own streaming driver

`llm_chain::streaming` houses the shared sans-IO wire decoders the built-in drivers use — `SseDecoder` for Server-Sent Events, `NdjsonDecoder` for newline-delimited JSON, and the `FrameDecoder` trait plus the `frames` adapter for turning any byte stream into a frame stream. Implement `FrameDecoder` for a custom framing (the Bedrock driver does exactly this for AWS's binary event stream) and `frames(decoder, byte_stream)` handles buffering, splitting and error passthrough.
