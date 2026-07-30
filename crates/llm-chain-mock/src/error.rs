use thiserror::Error;

/// Errors produced by the mock executor.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum MockError {
    /// A scripted executor (see [`Executor::with_responses`](crate::Executor::with_responses))
    /// ran out of canned responses.
    #[error("the mock executor ran out of scripted responses")]
    OutOfResponses,
    /// A failing executor (see [`Executor::failing`](crate::Executor::failing))
    /// produced its forced error.
    #[error("forced mock failure: {0}")]
    Forced(String),
}
