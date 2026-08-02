//! End-to-end tests for the executor against a local mock of the Gemini API.
//!
//! These verify the wire contract without a real API key: the request path,
//! authentication headers, JSON body, response parsing, and error mapping.

use std::io::{Read, Write};
use std::net::TcpListener;

use llm_chain::Parameters;
use llm_chain::traits::{Executor as _, Step as _};
use llm_chain_gemini::generate_content::{
    Executor, FinishReason, GeminiError, Model, Options, Role, Step, ThinkingLevel,
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
async fn executor_speaks_the_gemini_api() {
    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 200 OK",
        r#"{
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "Hello, Joe!"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 4, "totalTokenCount": 16},
            "modelVersion": "gemini-3.6-flash"
        }"#,
    );

    let exec = Executor::with_api_key("test-key").with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "Greet {}")])
        .with_system("be friendly")
        .with_options(
            Options::new()
                .with_temperature(0.5)
                .with_thinking_level(ThinkingLevel::Low),
        );
    let request = step.format(&Parameters::new_with_text("Joe")).unwrap();
    let response = exec.execute(request).await.unwrap();

    assert_eq!(response.text(), "Hello, Joe!");
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.usage_metadata.candidates_token_count, 4);

    let captured = server.join().unwrap();
    // Path and auth headers. The model id goes in the URL, not the body.
    assert!(
        captured
            .head
            .starts_with("POST /v1beta/models/gemini-3.6-flash:generateContent HTTP/1.1\r\n")
    );
    let head_lower = captured.head.to_lowercase();
    assert!(head_lower.contains("x-goog-api-key: test-key"));
    assert!(head_lower.contains("content-type: application/json"));
    // Body contents.
    assert!(captured.body.get("model").is_none());
    assert_eq!(
        captured.body["systemInstruction"]["parts"][0]["text"],
        "be friendly"
    );
    assert_eq!(captured.body["contents"][0]["role"], "user");
    assert_eq!(
        captured.body["contents"][0]["parts"][0]["text"],
        "Greet Joe"
    );
    assert_eq!(captured.body["generationConfig"]["temperature"], 0.5);
    assert_eq!(
        captured.body["generationConfig"]["thinkingConfig"]["thinkingLevel"],
        "low"
    );
    // Options that were not set must not be sent at all.
    assert!(captured.body["generationConfig"].get("topP").is_none());
    assert!(
        captured.body["generationConfig"]
            .get("maxOutputTokens")
            .is_none()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn api_errors_map_to_typed_errors() {
    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 429 Too Many Requests",
        r#"{"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED"}}"#,
    );

    let exec = Executor::with_api_key("test-key").with_base_url(&addr);
    let step = Step::new(Model::Gemini31FlashLite, [(Role::User, "hi")]);
    let request = step.format(&Parameters::new()).unwrap();
    let error = exec.execute(request).await.unwrap_err();

    match error {
        GeminiError::Api {
            http_status,
            status,
            message,
        } => {
            assert_eq!(http_status, 429);
            assert_eq!(status, "RESOURCE_EXHAUSTED");
            assert_eq!(message, "Quota exceeded");
        }
        other => panic!("expected Api error, got: {other:?}"),
    }
    server.join().unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn blocked_prompts_map_to_no_candidates() {
    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 200 OK",
        r#"{"promptFeedback": {"blockReason": "SAFETY"}}"#,
    );

    let exec = Executor::with_api_key("test-key").with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "hi")]);
    let request = step.format(&Parameters::new()).unwrap();
    let error = exec.execute(request).await.unwrap_err();

    match error {
        GeminiError::NoCandidates { reason } => assert_eq!(reason, "SAFETY"),
        other => panic!("expected NoCandidates error, got: {other:?}"),
    }
    server.join().unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn chains_run_end_to_end_against_the_mock() {
    use llm_chain::traits::StepExt;

    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 200 OK",
        r#"{
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "chained"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
        }"#,
    );

    let exec = Executor::with_api_key("test-key").with_base_url(&addr);
    let chain = Step::new(Model::default(), [(Role::User, "go")]).to_chain();
    let res = chain.run(Parameters::new(), &exec).await.unwrap();
    assert_eq!(res.text(), "chained");
    server.join().unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn vertex_requests_are_project_scoped_with_bearer_auth() {
    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 200 OK",
        r#"{
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "Hello from Vertex"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4, "totalTokenCount": 7}
        }"#,
    );

    let exec = Executor::vertex("my-project", "europe-north1", "ya29-token").with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "hi")]);
    let request = step.format(&Parameters::new()).unwrap();
    let response = exec.execute(request).await.unwrap();
    assert_eq!(response.text(), "Hello from Vertex");

    let captured = server.join().unwrap();
    // Vertex scopes the path by project and location; same wire format otherwise.
    assert!(captured.head.starts_with(
        "POST /v1/projects/my-project/locations/europe-north1/publishers/google/models/gemini-3.6-flash:generateContent HTTP/1.1\r\n"
    ));
    let head_lower = captured.head.to_lowercase();
    assert!(head_lower.contains("authorization: bearer ya29-token"));
    assert!(!head_lower.contains("x-goog-api-key"));
    assert_eq!(captured.body["contents"][0]["parts"][0]["text"], "hi");
}

#[tokio::test(flavor = "current_thread")]
async fn vertex_express_requests_use_api_keys() {
    let (addr, server) = spawn_one_shot(
        "HTTP/1.1 200 OK",
        r#"{
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "Hello from express"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4, "totalTokenCount": 7}
        }"#,
    );

    let exec = Executor::vertex_express("express-key").with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "hi")]);
    let request = step.format(&Parameters::new()).unwrap();
    let response = exec.execute(request).await.unwrap();
    assert_eq!(response.text(), "Hello from express");

    let captured = server.join().unwrap();
    // Express mode drops the project/location scoping and authenticates like
    // the consumer API.
    assert!(captured.head.starts_with(
        "POST /v1/publishers/google/models/gemini-3.6-flash:generateContent HTTP/1.1\r\n"
    ));
    let head_lower = captured.head.to_lowercase();
    assert!(head_lower.contains("x-goog-api-key: express-key"));
    assert!(!head_lower.contains("authorization:"));
}

// SSE bodies must keep `data:` lines at column zero — leading whitespace
// would change the SSE field name.
static STREAM_BODY: &str = r#"data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]}}],"responseId":"r1"}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":"lo, world"}]}}]}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":"!"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":4,"totalTokenCount":9},"modelVersion":"gemini-3.6-flash"}

"#;

#[tokio::test(flavor = "current_thread")]
async fn streaming_yields_chunks_and_reassembles_the_response() {
    use futures::StreamExt as _;
    use llm_chain::traits::StreamingExecutor as _;
    use llm_chain_gemini::generate_content::ResponseAccumulator;

    let (addr, server) = spawn_one_shot("HTTP/1.1 200 OK", STREAM_BODY);

    let exec = Executor::with_api_key("test-key").with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "stream {}")]);
    let request = step.format(&Parameters::new_with_text("this")).unwrap();

    let mut stream = exec.execute_stream(request).await.unwrap();
    let mut live_text = String::new();
    let mut accumulator = ResponseAccumulator::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        live_text.push_str(&chunk.text());
        accumulator.apply(&chunk);
    }

    assert_eq!(live_text, "Hello, world!");
    let response = accumulator.into_response().unwrap();
    assert_eq!(response.text(), "Hello, world!");
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.usage_metadata.total_token_count, 9);
    assert_eq!(response.response_id.as_deref(), Some("r1"));
    assert_eq!(response.model_version.as_deref(), Some("gemini-3.6-flash"));

    let captured = server.join().unwrap();
    assert!(captured.head.starts_with(
        "POST /v1beta/models/gemini-3.6-flash:streamGenerateContent?alt=sse HTTP/1.1\r\n"
    ));
    let head_lower = captured.head.to_lowercase();
    assert!(head_lower.contains("x-goog-api-key: test-key"));
    assert!(head_lower.contains("accept: text/event-stream"));
    assert_eq!(captured.body["contents"][0]["parts"][0]["text"], "stream this");
}

static STREAM_ERROR_BODY: &str = r#"data: {"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]}}]}

data: {"error":{"code":429,"message":"quota exceeded","status":"RESOURCE_EXHAUSTED"}}

"#;

#[tokio::test(flavor = "current_thread")]
async fn mid_stream_errors_surface_as_typed_api_errors() {
    use futures::StreamExt as _;
    use llm_chain::traits::StreamingExecutor as _;

    let (addr, server) = spawn_one_shot("HTTP/1.1 200 OK", STREAM_ERROR_BODY);

    let exec = Executor::with_api_key("test-key").with_base_url(&addr);
    let step = Step::new(Model::default(), [(Role::User, "hi")]);
    let request = step.format(&Parameters::new()).unwrap();

    let mut stream = exec.execute_stream(request).await.unwrap();
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.text(), "partial");

    let error = stream.next().await.unwrap().unwrap_err();
    match &error {
        GeminiError::Api {
            http_status,
            status,
            message,
        } => {
            assert_eq!(*http_status, 429);
            assert_eq!(status, "RESOURCE_EXHAUSTED");
            assert_eq!(message, "quota exceeded");
        }
        other => panic!("expected Api error, got: {other:?}"),
    }
    assert!(error.is_rate_limit());
    server.join().unwrap();
}
