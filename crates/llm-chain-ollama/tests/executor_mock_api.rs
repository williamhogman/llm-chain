//! End-to-end tests for the executor against a local mock of Ollama's chat API.
//!
//! These verify the wire contract without a running Ollama server: the request
//! path, auth headers, JSON body, response parsing, and error mapping.

use std::io::{Read, Write};
use std::net::TcpListener;

use llm_chain::Parameters;
use llm_chain::traits::{Executor as _, Step as _};
use llm_chain_ollama::chat::{
    DoneReason, Executor, Model, OllamaError, Options, Role, Step, Think,
};

/// What the mock server saw in the one request it handled.
struct CapturedRequest {
    head: String,
    body: serde_json::Value,
}

/// Spawns a one-shot HTTP/1.1 server that answers a single request with the
/// given status line and JSON body, returning the address and a handle that
/// yields the captured request.
fn spawn_one_shot(
    status_line: &'static str,
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
            "{status_line}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{response_body}",
            response_body.len()
        );
        stream.write_all(response.as_bytes()).expect("write");
        stream.flush().expect("flush");
        CapturedRequest { head, body }
    });
    (addr, handle)
}

#[tokio::test(flavor = "current_thread")]
async fn executor_speaks_the_chat_api() {
    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 200 OK",
        r#"{
            "model": "qwen3",
            "created_at": "2026-07-30T08:00:00.000000Z",
            "message": {"role": "assistant", "content": "Hello, Joe!", "thinking": "greeting"},
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 12,
            "eval_count": 4,
            "eval_duration": 200000000
        }"#,
    );

    let exec = Executor::new_default().with_base_url(&addr);
    let step = Step::new(
        Model::default(),
        [(Role::System, "be friendly"), (Role::User, "Greet {}")],
    )
    .with_options(
        Options::new()
            .with_temperature(0.5)
            .with_think(Think::Enabled),
    );
    let request = step.format(&Parameters::new_with_text("Joe")).unwrap();
    let response = exec.execute(request).await.unwrap();

    assert_eq!(response.text(), "Hello, Joe!");
    assert_eq!(response.thinking(), Some("greeting"));
    assert_eq!(response.done_reason, Some(DoneReason::Stop));
    assert_eq!(response.eval_count, Some(4));

    let captured = server.join().unwrap();
    // Path and headers. No auth header without an API key.
    assert!(captured.head.starts_with("POST /api/chat HTTP/1.1\r\n"));
    let head_lower = captured.head.to_lowercase();
    assert!(head_lower.contains("content-type: application/json"));
    assert!(!head_lower.contains("authorization:"));
    // Body contents.
    assert_eq!(captured.body["model"], "qwen3");
    assert_eq!(captured.body["stream"], false);
    assert_eq!(captured.body["think"], true);
    assert_eq!(captured.body["messages"][0]["role"], "system");
    assert_eq!(captured.body["messages"][0]["content"], "be friendly");
    assert_eq!(captured.body["messages"][1]["role"], "user");
    assert_eq!(captured.body["messages"][1]["content"], "Greet Joe");
    assert_eq!(captured.body["options"]["temperature"], 0.5);
    // Options that were not set must not be sent at all.
    assert!(captured.body["options"].get("top_p").is_none());
    assert!(captured.body.get("format").is_none());
    assert!(captured.body.get("keep_alive").is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn bearer_auth_is_sent_when_an_api_key_is_set() {
    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 200 OK",
        r#"{
            "model": "gpt-oss:120b-cloud",
            "message": {"role": "assistant", "content": "from the cloud"},
            "done": true,
            "done_reason": "stop"
        }"#,
    );

    let exec = Executor::cloud("test-key").with_base_url(&addr);
    let step = Step::new("gpt-oss:120b-cloud", [(Role::User, "hi")]);
    let request = step.format(&Parameters::new()).unwrap();
    let response = exec.execute(request).await.unwrap();
    assert_eq!(response.text(), "from the cloud");

    let captured = server.join().unwrap();
    assert!(
        captured
            .head
            .to_lowercase()
            .contains("authorization: bearer test-key")
    );
    assert_eq!(captured.body["model"], "gpt-oss:120b-cloud");
}

#[tokio::test(flavor = "current_thread")]
async fn api_errors_map_to_typed_errors() {
    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 404 Not Found",
        r#"{"error":"model 'nope' not found, try pulling it first"}"#,
    );

    let exec = Executor::new_default().with_base_url(&addr);
    let step = Step::new("nope", [(Role::User, "hi")]);
    let request = step.format(&Parameters::new()).unwrap();
    let error = exec.execute(request).await.unwrap_err();

    match error {
        OllamaError::Api { status, message } => {
            assert_eq!(status, 404);
            assert_eq!(message, "model 'nope' not found, try pulling it first");
        }
        other => panic!("expected Api error, got: {other:?}"),
    }
    server.join().unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn unreachable_servers_map_to_connection_errors() {
    // Bind a port, then drop the listener so connecting to it is refused.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = format!("http://{}", listener.local_addr().expect("addr"));
    drop(listener);

    let exec = Executor::new_default().with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "hi")]);
    let request = step.format(&Parameters::new()).unwrap();
    let error = exec.execute(request).await.unwrap_err();

    match &error {
        OllamaError::Connection { url, .. } => {
            assert_eq!(url, &addr);
            let message = error.to_string();
            assert!(
                message.contains("ollama serve"),
                "unhelpful message: {message}"
            );
        }
        other => panic!("expected Connection error, got: {other:?}"),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn chains_run_end_to_end_against_the_mock() {
    use llm_chain::traits::StepExt;

    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 200 OK",
        r#"{
            "model": "qwen3",
            "message": {"role": "assistant", "content": "chained"},
            "done": true,
            "done_reason": "stop"
        }"#,
    );

    let exec = Executor::new_default().with_base_url(&addr);
    let chain = Step::new(Model::default(), [(Role::User, "go")]).to_chain();
    let res = chain.run(Parameters::new(), &exec).await.unwrap();
    assert_eq!(res.text(), "chained");
    server.join().unwrap();
}

// NDJSON: one complete JSON chunk per line, `done: true` on the last.
static STREAM_BODY: &str = r#"{"model":"qwen3","created_at":"2026-07-30T08:00:00Z","message":{"role":"assistant","content":"Str"},"done":false}
{"model":"qwen3","created_at":"2026-07-30T08:00:00Z","message":{"role":"assistant","content":"eamed!"},"done":false}
{"model":"qwen3","created_at":"2026-07-30T08:00:01Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":7,"eval_count":6,"total_duration":900000000}
"#;

#[tokio::test(flavor = "current_thread")]
async fn streaming_yields_chunks_and_reassembles_the_response() {
    use futures::StreamExt as _;
    use llm_chain::traits::StreamingExecutor as _;
    use llm_chain_ollama::chat::ResponseAccumulator;

    let (addr, server) = spawn_one_shot("HTTP/1.1 200 OK", STREAM_BODY);

    let exec = Executor::new_default().with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "stream {}")]);
    let request = step.format(&Parameters::new_with_text("this")).unwrap();

    let mut stream = exec.execute_stream(request).await.unwrap();
    let mut live_text = String::new();
    let mut accumulator = ResponseAccumulator::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        live_text.push_str(&chunk.message.content);
        accumulator.apply(&chunk);
    }

    assert_eq!(live_text, "Streamed!");
    assert!(accumulator.is_complete());
    let response = accumulator.into_response().unwrap();
    assert_eq!(response.text(), "Streamed!");
    assert_eq!(response.done_reason, Some(DoneReason::Stop));
    assert_eq!(response.prompt_eval_count, Some(7));
    assert_eq!(response.eval_count, Some(6));

    let captured = server.join().unwrap();
    assert!(captured.head.starts_with("POST /api/chat HTTP/1.1\r\n"));
    // The buffered path sends stream: false; the streaming path flips it.
    assert_eq!(captured.body["stream"], true);
    assert_eq!(captured.body["messages"][0]["content"], "stream this");
}

static STREAM_ERROR_BODY: &str = r#"{"model":"qwen3","message":{"role":"assistant","content":"partial"},"done":false}
{"error":"model runner has unexpectedly stopped"}
"#;

#[tokio::test(flavor = "current_thread")]
async fn mid_stream_errors_surface_as_typed_stream_errors() {
    use futures::StreamExt as _;
    use llm_chain::traits::StreamingExecutor as _;

    let (addr, server) = spawn_one_shot("HTTP/1.1 200 OK", STREAM_ERROR_BODY);

    let exec = Executor::new_default().with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "hi")]);
    let request = step.format(&Parameters::new()).unwrap();

    let mut stream = exec.execute_stream(request).await.unwrap();
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.message.content, "partial");

    let error = stream.next().await.unwrap().unwrap_err();
    match &error {
        OllamaError::StreamError(message) => {
            assert_eq!(message, "model runner has unexpectedly stopped");
        }
        other => panic!("expected StreamError, got: {other:?}"),
    }
    server.join().unwrap();
}
