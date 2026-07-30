use std::collections::VecDeque;
use std::sync::Mutex;

use llm_chain::{Parameters, traits};

use crate::error::MockError;
use crate::step::Step;

/// The behaviour of a mock executor.
#[derive(Debug)]
enum Behavior {
    /// Echo the formatted prompt back as the response.
    Echo,
    /// Return canned responses in order; error when exhausted.
    Scripted(Mutex<VecDeque<String>>),
    /// Always fail with [`MockError::Forced`].
    Failing(String),
}

/// A deterministic in-process executor for testing chains.
///
/// See the [crate-level docs](crate) for the three behaviours. All executed
/// prompts are recorded and available via [`Executor::calls`], so tests can
/// assert on exactly what the "model" was asked.
#[derive(Debug)]
pub struct Executor {
    behavior: Behavior,
    calls: Mutex<Vec<String>>,
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor {
    /// Creates an echo executor: every call returns the formatted prompt verbatim.
    pub fn new() -> Self {
        Self {
            behavior: Behavior::Echo,
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Creates a scripted executor that returns the given responses in order.
    ///
    /// Once the script is exhausted, calls fail with [`MockError::OutOfResponses`].
    pub fn with_responses<I, S>(responses: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            behavior: Behavior::Scripted(Mutex::new(
                responses.into_iter().map(Into::into).collect(),
            )),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Creates an executor whose every call fails with [`MockError::Forced`].
    pub fn failing(message: impl Into<String>) -> Self {
        Self {
            behavior: Behavior::Failing(message.into()),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// The prompts executed so far, in order.
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().expect("calls mutex poisoned").clone()
    }
}

impl traits::Executor for Executor {
    type Step = Step;
    type Output = String;
    type Error = MockError;

    async fn execute(&self, input: String) -> Result<String, MockError> {
        self.calls
            .lock()
            .expect("calls mutex poisoned")
            .push(input.clone());
        match &self.behavior {
            Behavior::Echo => Ok(input),
            Behavior::Scripted(queue) => queue
                .lock()
                .expect("responses mutex poisoned")
                .pop_front()
                .ok_or(MockError::OutOfResponses),
            Behavior::Failing(message) => Err(MockError::Forced(message.clone())),
        }
    }

    fn apply_output_to_parameters(parameters: Parameters, output: &String) -> Parameters {
        parameters.with_text(output)
    }

    fn combine_outputs(output: &String, other: &String) -> String {
        format!("{output}\n{other}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_chain::chains::sequential::Chain;
    use llm_chain::traits::Executor as _;

    #[tokio::test]
    async fn echo_returns_the_prompt_and_records_calls() {
        let executor = Executor::new();
        let out = executor.execute("hello".to_string()).await.unwrap();
        assert_eq!(out, "hello");
        assert_eq!(executor.calls(), ["hello"]);
    }

    #[tokio::test]
    async fn scripted_responses_come_back_in_order_then_run_out() {
        let executor = Executor::with_responses(["one", "two"]);
        assert_eq!(executor.execute("a".into()).await.unwrap(), "one");
        assert_eq!(executor.execute("b".into()).await.unwrap(), "two");
        assert_eq!(
            executor.execute("c".into()).await.unwrap_err(),
            MockError::OutOfResponses
        );
        assert_eq!(executor.calls(), ["a", "b", "c"]);
    }

    #[tokio::test]
    async fn failing_always_fails() {
        let executor = Executor::failing("boom");
        assert_eq!(
            executor.execute("a".into()).await.unwrap_err(),
            MockError::Forced("boom".into())
        );
    }

    #[tokio::test]
    async fn sequential_chains_thread_outputs_between_steps() {
        let chain: Chain<Step> = [Step::new("first: {text}"), Step::new("second: {text}")]
            .into_iter()
            .collect();
        let executor = Executor::new();
        let out = chain.run(Parameters::new_with_text("go"), &executor).await;
        assert_eq!(out.unwrap(), "second: first: go");
    }

    #[tokio::test]
    async fn chain_errors_propagate_from_the_executor() {
        let chain = Chain::of_one(Step::new("{text}"));
        let executor = Executor::failing("down");
        let result = chain.run(Parameters::new_with_text("go"), &executor).await;
        assert!(result.is_err());
    }
}
