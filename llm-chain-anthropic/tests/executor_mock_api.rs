//! End-to-end tests for the executor against a local mock of the Messages API.
//!
//! These verify the wire contract without a real API key: the request path,
//! authentication headers, JSON body, response parsing, and error mapping.

use std::io::{Read, Write};
use std::net::TcpListener;

use llm_chain::Parameters;
use llm_chain::traits::{Executor as _, Step as _};
use llm_chain_anthropic::messages::{
    AnthropicError, Executor, Model, Options, Role, Step, StopReason,
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
async fn executor_speaks_the_messages_api() {
    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 200 OK",
        r#"{
            "id": "msg_mock",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "Hello, Joe!"}],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 12, "output_tokens": 4}
        }"#,
    );

    let exec = Executor::with_api_key("test-key").with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "Greet {}")])
        .with_system("be friendly")
        .with_options(Options::new().with_temperature(0.5));
    let request = step.format(&Parameters::new_with_text("Joe")).unwrap();
    let response = exec.execute(request).await.unwrap();

    assert_eq!(response.text(), "Hello, Joe!");
    assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(response.usage.output_tokens, 4);

    let captured = server.join().unwrap();
    // Path and auth headers.
    assert!(captured.head.starts_with("POST /v1/messages HTTP/1.1\r\n"));
    let head_lower = captured.head.to_lowercase();
    assert!(head_lower.contains("x-api-key: test-key"));
    assert!(head_lower.contains("anthropic-version: 2023-06-01"));
    assert!(head_lower.contains("content-type: application/json"));
    // Body contents.
    assert_eq!(captured.body["model"], "claude-sonnet-5");
    assert_eq!(captured.body["system"], "be friendly");
    assert_eq!(captured.body["messages"][0]["role"], "user");
    assert_eq!(captured.body["messages"][0]["content"], "Greet Joe");
    assert_eq!(captured.body["temperature"], 0.5);
    // Options that were not set must not be sent at all.
    assert!(captured.body.get("top_p").is_none());
    assert!(captured.body.get("thinking").is_none());
    assert!(captured.body.get("effort").is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn api_errors_map_to_typed_errors() {
    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 529 Unknown",
        r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
    );

    let exec = Executor::with_api_key("test-key").with_base_url(&addr);
    let step = Step::new(Model::ClaudeHaiku45, [(Role::User, "hi")]);
    let request = step.format(&Parameters::new()).unwrap();
    let error = exec.execute(request).await.unwrap_err();

    match error {
        AnthropicError::Api {
            status,
            error_type,
            message,
        } => {
            assert_eq!(status, 529);
            assert_eq!(error_type, "overloaded_error");
            assert_eq!(message, "Overloaded");
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
        r#"{
            "id": "msg_chain",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "chained"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }"#,
    );

    let exec = Executor::with_api_key("test-key").with_base_url(&addr);
    let chain = Step::new(Model::default(), [(Role::User, "go")]).to_chain();
    let res = chain.run(Parameters::new(), &exec).await.unwrap();
    assert_eq!(res.text(), "chained");
    server.join().unwrap();
}
