//! End-to-end tests for the executor against a local mock of the Lovable AI
//! Gateway's chat completions API.
//!
//! These verify the wire contract without a real gateway: the request path,
//! auth headers, JSON body, response parsing, run-id capture, and error
//! mapping.

use std::io::{Read, Write};
use std::net::TcpListener;

use llm_chain::Parameters;
use llm_chain::traits::{Executor as _, Step as _};
use llm_chain_lovable::chat::{
    Executor, FinishReason, LovableError, Model, Options, Reasoning, ReasoningEffort, Role, Step,
};

/// What the mock server saw in the one request it handled.
struct CapturedRequest {
    head: String,
    body: serde_json::Value,
}

/// Spawns a one-shot HTTP/1.1 server that answers a single request with the
/// given status line, extra headers and body, returning the address and a
/// handle that yields the captured request.
fn spawn_one_shot(
    status_line: &'static str,
    extra_headers: &'static str,
    response_body: &'static str,
) -> (String, std::thread::JoinHandle<CapturedRequest>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = format!("http://{}", listener.local_addr().expect("addr"));
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        // Read until the end of headers.
        let header_end = loop {
            let n = stream.read(&mut chunk).expect("read");
            buf.extend_from_slice(&chunk[..n]);
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break pos + 4;
            }
        };
        let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
        let content_length: usize = head
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse().ok())?
            })
            .expect("content-length header");
        while buf.len() < header_end + content_length {
            let n = stream.read(&mut chunk).expect("read body");
            buf.extend_from_slice(&chunk[..n]);
        }
        let body = serde_json::from_slice(&buf[header_end..header_end + content_length])
            .expect("json body");
        let response = format!(
            "{status_line}\r\ncontent-type: application/json\r\n{extra_headers}content-length: {}\r\nconnection: close\r\n\r\n{response_body}",
            response_body.len()
        );
        stream.write_all(response.as_bytes()).expect("write");
        stream.flush().expect("flush");
        CapturedRequest { head, body }
    });
    (addr, handle)
}

#[tokio::test(flavor = "current_thread")]
async fn executor_speaks_the_chat_completions_api() {
    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 200 OK",
        "x-lovable-aig-run-id: run-123\r\n",
        r#"{
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1785000000,
            "model": "google/gemini-3.6-flash",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello, Joe!", "reasoning": "greeting"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}
        }"#,
    );

    let exec = Executor::with_api_key("test-key").with_base_url(&addr);
    let step = Step::new(
        Model::default(),
        [(Role::System, "be friendly"), (Role::User, "Greet {}")],
    )
    .with_options(
        Options::new()
            .with_temperature(0.5)
            .with_reasoning(Reasoning::effort(ReasoningEffort::Low)),
    );
    let request = step.format(&Parameters::new_with_text("Joe")).unwrap();
    let response = exec.execute(request).await.unwrap();

    assert_eq!(response.text(), "Hello, Joe!");
    assert_eq!(response.reasoning(), Some("greeting"));
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.usage.unwrap().total_tokens, 16);
    // The run id comes from the response header, not the body.
    assert_eq!(response.run_id.as_deref(), Some("run-123"));

    let captured = server.join().unwrap();
    // Path and headers: the gateway authenticates on Lovable-API-Key, not Bearer.
    assert!(
        captured
            .head
            .starts_with("POST /chat/completions HTTP/1.1\r\n")
    );
    let head_lower = captured.head.to_lowercase();
    assert!(head_lower.contains("content-type: application/json"));
    assert!(head_lower.contains("lovable-api-key: test-key"));
    assert!(head_lower.contains("x-lovable-aig-sdk: llm-chain"));
    assert!(!head_lower.contains("authorization:"));
    // Body contents.
    assert_eq!(captured.body["model"], "google/gemini-3.6-flash");
    assert_eq!(captured.body["stream"], false);
    assert_eq!(captured.body["temperature"], 0.5);
    assert_eq!(captured.body["reasoning"]["effort"], "low");
    assert_eq!(captured.body["messages"][0]["role"], "system");
    assert_eq!(captured.body["messages"][0]["content"], "be friendly");
    assert_eq!(captured.body["messages"][1]["role"], "user");
    assert_eq!(captured.body["messages"][1]["content"], "Greet Joe");
    // Options that were not set must not be sent at all.
    assert!(captured.body.get("top_p").is_none());
    assert!(captured.body.get("response_format").is_none());
    assert!(captured.body.get("stream_options").is_none());
    assert!(captured.body.get("reasoning_effort").is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn credit_exhaustion_maps_to_a_typed_error() {
    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 402 Payment Required",
        "",
        r#"{"error":{"message":"Payment required: your workspace is out of AI credits.","type":"payment_required"}}"#,
    );

    let exec = Executor::with_api_key("test-key").with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "hi")]);
    let request = step.format(&Parameters::new()).unwrap();
    let error = exec.execute(request).await.unwrap_err();

    assert!(error.is_credits_exhausted());
    assert!(!error.is_rate_limit());
    match error {
        LovableError::Api { status, message } => {
            assert_eq!(status, 402);
            assert!(message.contains("out of AI credits"));
        }
        other => panic!("expected Api error, got: {other:?}"),
    }
    server.join().unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn rate_limits_map_to_a_typed_error() {
    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 429 Too Many Requests",
        "",
        r#"{"error":"Rate limit exceeded, please retry later"}"#,
    );

    let exec = Executor::with_api_key("test-key").with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "hi")]);
    let request = step.format(&Parameters::new()).unwrap();
    let error = exec.execute(request).await.unwrap_err();

    assert!(error.is_rate_limit());
    match error {
        LovableError::Api { status, message } => {
            assert_eq!(status, 429);
            assert_eq!(message, "Rate limit exceeded, please retry later");
        }
        other => panic!("expected Api error, got: {other:?}"),
    }
    server.join().unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn chains_run_end_to_end_against_the_mock() {
    use llm_chain::traits::StepExt;

    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 200 OK",
        "",
        r#"{
            "choices": [{
                "message": {"role": "assistant", "content": "chained"},
                "finish_reason": "stop"
            }]
        }"#,
    );

    let exec = Executor::with_api_key("test-key").with_base_url(&addr);
    let chain = Step::new(Model::default(), [(Role::User, "go")]).to_chain();
    let res = chain.run(Parameters::new(), &exec).await.unwrap();
    assert_eq!(res.text(), "chained");
    server.join().unwrap();
}

// SSE: OpenAI-style chunks, a trailing usage chunk, then `data: [DONE]`.
static STREAM_BODY: &str = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"google/gemini-3.6-flash\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Str\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"google/gemini-3.6-flash\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"eamed!\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"google/gemini-3.6-flash\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"google/gemini-3.6-flash\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":6,\"total_tokens\":13}}\n\n\
data: [DONE]\n\n";

#[tokio::test(flavor = "current_thread")]
async fn streaming_yields_chunks_and_reassembles_the_response() {
    use futures::StreamExt as _;
    use llm_chain::traits::StreamingExecutor as _;
    use llm_chain_lovable::chat::ResponseAccumulator;

    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 200 OK",
        "x-lovable-aig-run-id: run-456\r\n",
        STREAM_BODY,
    );

    let exec = Executor::with_api_key("test-key").with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "stream {}")]);
    let request = step.format(&Parameters::new_with_text("this")).unwrap();

    let mut stream = exec.execute_stream(request).await.unwrap();
    let mut live_text = String::new();
    let mut accumulator = ResponseAccumulator::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        assert_eq!(chunk.run_id.as_deref(), Some("run-456"));
        if let Some(text) = chunk.text() {
            live_text.push_str(text);
        }
        accumulator.apply(&chunk);
    }

    assert_eq!(live_text, "Streamed!");
    assert!(accumulator.is_complete());
    let response = accumulator.into_response().unwrap();
    assert_eq!(response.text(), "Streamed!");
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.usage.unwrap().total_tokens, 13);
    assert_eq!(response.run_id.as_deref(), Some("run-456"));

    let captured = server.join().unwrap();
    assert!(
        captured
            .head
            .starts_with("POST /chat/completions HTTP/1.1\r\n")
    );
    // The buffered path sends stream: false; the streaming path flips it and
    // asks for the final usage chunk.
    assert_eq!(captured.body["stream"], true);
    assert_eq!(captured.body["stream_options"]["include_usage"], true);
    assert_eq!(captured.body["messages"][0]["content"], "stream this");
}

static STREAM_ERROR_BODY: &str = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n\
data: {\"error\":{\"message\":\"upstream exploded\",\"code\":502}}\n\n";

#[tokio::test(flavor = "current_thread")]
async fn mid_stream_errors_surface_as_typed_stream_errors() {
    use futures::StreamExt as _;
    use llm_chain::traits::StreamingExecutor as _;

    let (addr, server) = spawn_one_shot("HTTP/1.1 200 OK", "", STREAM_ERROR_BODY);

    let exec = Executor::with_api_key("test-key").with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "hi")]);
    let request = step.format(&Parameters::new()).unwrap();

    let mut stream = exec.execute_stream(request).await.unwrap();
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.text(), Some("partial"));

    let error = stream.next().await.unwrap().unwrap_err();
    match &error {
        LovableError::StreamError(message) => {
            assert_eq!(message, "upstream exploded");
        }
        other => panic!("expected StreamError, got: {other:?}"),
    }
    server.join().unwrap();
}
