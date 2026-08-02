//! End-to-end tests for the executor against a local mock of the Converse API.
//!
//! These verify the wire contract without a real API key: the request path,
//! authentication headers, JSON body, response parsing, and error mapping.

use std::io::{Read, Write};
use std::net::TcpListener;

use llm_chain::Parameters;
use llm_chain::traits::{Executor as _, Step as _};
use llm_chain_bedrock::converse::{
    BedrockError, Executor, Model, Options, Role, Step, StopReason, models,
};

/// What the mock server saw in the one request it handled.
struct CapturedRequest {
    head: String,
    body: serde_json::Value,
}

/// Spawns a one-shot HTTP/1.1 server that answers a single request with the
/// given status line, extra headers, and JSON body, returning the address and
/// a handle that yields the captured request.
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
            "{status_line}\r\n{extra_headers}content-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{response_body}",
            response_body.len()
        );
        stream.write_all(response.as_bytes()).expect("write");
        stream.flush().expect("flush");
        CapturedRequest { head, body }
    });
    (addr, handle)
}

#[tokio::test(flavor = "current_thread")]
async fn executor_speaks_the_converse_api() {
    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 200 OK",
        "",
        r#"{
            "output": {
                "message": {"role": "assistant", "content": [{"text": "Hello, Joe!"}]}
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 12, "outputTokens": 4, "totalTokens": 16},
            "metrics": {"latencyMs": 321}
        }"#,
    );

    let exec = Executor::with_bearer_token("test-key").with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "Greet {}")])
        .with_system("be friendly")
        .with_options(Options::new().with_temperature(0.5).with_max_tokens(128));
    let request = step.format(&Parameters::new_with_text("Joe")).unwrap();
    let response = exec.execute(request).await.unwrap();

    assert_eq!(response.text(), "Hello, Joe!");
    assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(response.usage.output_tokens, 4);
    assert_eq!(response.metrics.unwrap().latency_ms, 321);

    let captured = server.join().unwrap();
    // Path (model id percent-encoded) and auth headers.
    assert!(
        captured.head.starts_with(
            "POST /model/global.anthropic.claude-sonnet-5-v1%3A0/converse HTTP/1.1\r\n"
        )
    );
    let head_lower = captured.head.to_lowercase();
    assert!(head_lower.contains("authorization: bearer test-key"));
    assert!(head_lower.contains("content-type: application/json"));
    // Body contents. The model id goes in the URL, not the body.
    assert!(captured.body.get("modelId").is_none());
    assert_eq!(captured.body["system"][0]["text"], "be friendly");
    assert_eq!(captured.body["messages"][0]["role"], "user");
    assert_eq!(
        captured.body["messages"][0]["content"][0]["text"],
        "Greet Joe"
    );
    assert_eq!(captured.body["inferenceConfig"]["temperature"], 0.5);
    assert_eq!(captured.body["inferenceConfig"]["maxTokens"], 128);
    // Options that were not set must not be sent at all.
    assert!(captured.body["inferenceConfig"].get("topP").is_none());
    assert!(captured.body.get("additionalModelRequestFields").is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn api_errors_map_to_typed_errors() {
    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 429 Too Many Requests",
        "x-amzn-ErrorType: ThrottlingException:http://internal.amazon.com/coral/com.amazon.bedrock/\r\n",
        r#"{"message":"Too many requests, please wait before trying again."}"#,
    );

    let exec = Executor::with_bearer_token("test-key").with_base_url(&addr);
    let step = Step::new(models::NOVA_MICRO, [(Role::User, "hi")]);
    let request = step.format(&Parameters::new()).unwrap();
    let error = exec.execute(request).await.unwrap_err();

    match error {
        BedrockError::Api {
            status,
            error_type,
            message,
        } => {
            assert_eq!(status, 429);
            assert_eq!(error_type, "ThrottlingException");
            assert_eq!(
                message,
                "Too many requests, please wait before trying again."
            );
        }
        other => panic!("expected Api error, got: {other:?}"),
    }
    server.join().unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn additional_model_request_fields_pass_through_verbatim() {
    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 200 OK",
        "",
        r#"{
            "output": {
                "message": {"role": "assistant", "content": [
                    {"reasoningContent": {"reasoningText": {"text": "17*23 = 391", "signature": "sig"}}},
                    {"text": "391"}
                ]}
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 9, "outputTokens": 30, "totalTokens": 39}
        }"#,
    );

    let exec = Executor::with_bearer_token("test-key").with_base_url(&addr);
    let step = Step::new(models::CLAUDE_HAIKU_4_5, [(Role::User, "17 * 23?")]).with_options(
        Options::new()
            .with_max_tokens(4096)
            .with_additional_model_request_fields(serde_json::json!({
                "thinking": {"type": "enabled", "budget_tokens": 2048}
            })),
    );
    let request = step.format(&Parameters::new()).unwrap();
    let response = exec.execute(request).await.unwrap();

    assert_eq!(response.text(), "391");
    assert_eq!(response.reasoning().as_deref(), Some("17*23 = 391"));

    let captured = server.join().unwrap();
    assert_eq!(
        captured.body["additionalModelRequestFields"]["thinking"]["budget_tokens"],
        2048
    );
}

#[tokio::test(flavor = "current_thread")]
async fn chains_run_end_to_end_against_the_mock() {
    use llm_chain::traits::StepExt;

    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 200 OK",
        "",
        r#"{
            "output": {
                "message": {"role": "assistant", "content": [{"text": "chained"}]}
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
        }"#,
    );

    let exec = Executor::with_bearer_token("test-key").with_base_url(&addr);
    let chain = Step::new(Model::default(), [(Role::User, "go")]).to_chain();
    let res = chain.run(Parameters::new(), &exec).await.unwrap();
    assert_eq!(res.text(), "chained");
    server.join().unwrap();
}

// ---------------------------------------------------------------------------
// Streaming (ConverseStream, binary event stream framing)
// ---------------------------------------------------------------------------

use llm_chain_bedrock::converse::{EventStreamMessage, HeaderValue, ResponseAccumulator};

/// Encodes one `event` frame in AWS's binary event stream format.
fn event_frame(event_type: &str, payload: &str) -> Vec<u8> {
    EventStreamMessage {
        headers: vec![
            (
                ":message-type".to_string(),
                HeaderValue::String("event".to_string()),
            ),
            (
                ":event-type".to_string(),
                HeaderValue::String(event_type.to_string()),
            ),
        ],
        payload: payload.as_bytes().to_vec(),
    }
    .to_bytes()
}

/// Encodes one `exception` frame.
fn exception_frame(exception_type: &str, payload: &str) -> Vec<u8> {
    EventStreamMessage {
        headers: vec![
            (
                ":message-type".to_string(),
                HeaderValue::String("exception".to_string()),
            ),
            (
                ":exception-type".to_string(),
                HeaderValue::String(exception_type.to_string()),
            ),
        ],
        payload: payload.as_bytes().to_vec(),
    }
    .to_bytes()
}

/// Spawns a one-shot HTTP/1.1 server that answers a single request with the
/// given binary event stream body, written in small chunks to exercise
/// incremental decoding.
fn spawn_streaming(body: Vec<u8>) -> (String, std::thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = format!("http://{}", listener.local_addr().expect("addr"));
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
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
        let response_head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/vnd.amazon.eventstream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            body.len()
        );
        stream.write_all(response_head.as_bytes()).expect("write");
        // Write the frames in deliberately awkward chunks.
        for piece in body.chunks(7) {
            stream.write_all(piece).expect("write body");
            stream.flush().expect("flush");
        }
        head
    });
    (addr, handle)
}

#[tokio::test(flavor = "current_thread")]
async fn streaming_yields_deltas_and_reassembles_the_response() {
    use futures::StreamExt as _;
    use llm_chain::traits::StreamingExecutor as _;

    let mut body = Vec::new();
    body.extend(event_frame("messageStart", r#"{"role":"assistant"}"#));
    body.extend(event_frame(
        "contentBlockDelta",
        r#"{"contentBlockIndex":0,"delta":{"text":"Hello"}}"#,
    ));
    body.extend(event_frame(
        "contentBlockDelta",
        r#"{"contentBlockIndex":0,"delta":{"text":", Joe!"}}"#,
    ));
    body.extend(event_frame(
        "contentBlockStop",
        r#"{"contentBlockIndex":0}"#,
    ));
    body.extend(event_frame("messageStop", r#"{"stopReason":"end_turn"}"#));
    body.extend(event_frame(
        "metadata",
        r#"{"usage":{"inputTokens":12,"outputTokens":4,"totalTokens":16},"metrics":{"latencyMs":99}}"#,
    ));
    let (addr, server) = spawn_streaming(body);

    let exec = Executor::with_bearer_token("test-key").with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "Greet {}")]);
    let request = step.format(&Parameters::new_with_text("Joe")).unwrap();

    let mut stream = exec.execute_stream(request).await.unwrap();
    let mut accumulator = ResponseAccumulator::new();
    let mut streamed = String::new();
    while let Some(event) = stream.next().await {
        let event = event.unwrap();
        if let Some(text) = event.text_delta() {
            streamed.push_str(text);
        }
        accumulator.apply(&event);
    }
    let response = accumulator.into_response().unwrap();

    assert_eq!(streamed, "Hello, Joe!");
    assert_eq!(response.text(), "Hello, Joe!");
    assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(response.usage.total_tokens, 16);
    assert_eq!(response.metrics.unwrap().latency_ms, 99);

    let head = server.join().unwrap();
    assert!(head.contains("/converse-stream HTTP/1.1"));
    let head_lower = head.to_lowercase();
    assert!(head_lower.contains("authorization: bearer test-key"));
}

#[tokio::test(flavor = "current_thread")]
async fn mid_stream_exceptions_map_to_typed_errors() {
    use futures::StreamExt as _;
    use llm_chain::traits::StreamingExecutor as _;

    let mut body = Vec::new();
    body.extend(event_frame("messageStart", r#"{"role":"assistant"}"#));
    body.extend(event_frame(
        "contentBlockDelta",
        r#"{"contentBlockIndex":0,"delta":{"text":"Hel"}}"#,
    ));
    body.extend(exception_frame(
        "throttlingException",
        r#"{"message":"Too many tokens, slow down."}"#,
    ));
    let (addr, server) = spawn_streaming(body);

    let exec = Executor::with_bearer_token("test-key").with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "hi")]);
    let request = step.format(&Parameters::new()).unwrap();

    let mut stream = exec.execute_stream(request).await.unwrap();
    let mut texts = String::new();
    let mut error = None;
    while let Some(event) = stream.next().await {
        match event {
            Ok(event) => {
                if let Some(text) = event.text_delta() {
                    texts.push_str(text);
                }
            }
            Err(e) => error = Some(e),
        }
    }

    assert_eq!(texts, "Hel");
    let error = error.expect("stream carried an exception");
    assert!(error.is_rate_limit());
    match error {
        BedrockError::StreamException {
            exception_type,
            message,
        } => {
            assert_eq!(exception_type, "throttlingException");
            assert_eq!(message, "Too many tokens, slow down.");
        }
        other => panic!("expected StreamException, got: {other:?}"),
    }
    server.join().unwrap();
}
