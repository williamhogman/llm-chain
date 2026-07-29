//! End-to-end integration test that runs a real GGUF model through the
//! executor. It only runs when the `LLM_CHAIN_TEST_MODEL` environment
//! variable points at a GGUF model file, so plain `cargo test` stays fast
//! and network-free.
//!
//! A tiny model suitable for this test (~1 MB, used by llama.cpp's own CI):
//! <https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf>

use llm_chain::{Parameters, traits::StepExt};
use llm_chain_llama::{Executor, LlamaConfig, ModelConfig, Step};

fn test_model_path() -> Option<String> {
    std::env::var("LLM_CHAIN_TEST_MODEL").ok()
}

#[tokio::test(flavor = "current_thread")]
async fn generates_text_from_gguf_model() {
    let Some(model_path) = test_model_path() else {
        eprintln!("skipping: set LLM_CHAIN_TEST_MODEL to a GGUF file to run this test");
        return;
    };

    let exec = Executor::new_with_config(&model_path, ModelConfig::new().with_n_ctx(512))
        .expect("model should load");

    let config = LlamaConfig {
        n_tok_predict: Some(32),
        seed: Some(42),
        ..Default::default()
    };
    let chain = Step::new_with_config("Once upon a time".into(), Some(config)).to_chain();
    let output = chain
        .run(Parameters::new(), &exec)
        .await
        .expect("generation should succeed");

    let text = output.to_string();
    assert!(!text.is_empty(), "the model should generate some text");
}

#[tokio::test(flavor = "current_thread")]
async fn greedy_sampling_is_deterministic() {
    let Some(model_path) = test_model_path() else {
        eprintln!("skipping: set LLM_CHAIN_TEST_MODEL to a GGUF file to run this test");
        return;
    };

    let exec = Executor::new_with_config(&model_path, ModelConfig::new().with_n_ctx(512))
        .expect("model should load");

    let config = LlamaConfig {
        n_tok_predict: Some(16),
        temp: Some(0.0), // greedy
        ..Default::default()
    };

    let mut outputs = Vec::new();
    for _ in 0..2 {
        let chain =
            Step::new_with_config("Once upon a time".into(), Some(config.clone())).to_chain();
        let output = chain
            .run(Parameters::new(), &exec)
            .await
            .expect("generation should succeed");
        outputs.push(output.to_string());
    }
    assert_eq!(
        outputs[0], outputs[1],
        "greedy sampling must be deterministic"
    );
}
