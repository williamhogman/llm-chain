//! # llm-chain-mock
//!
//! A mock driver for [`llm-chain`](https://crates.io/crates/llm-chain): run
//! chains against a deterministic, in-process fake model instead of a real
//! LLM. Useful for unit tests, CI pipelines and offline development, where you
//! want to exercise chain wiring, prompt formatting and error handling without
//! network access or API keys.
//!
//! Three behaviours are available:
//!
//! - [`Executor::new`] — **echo**: every call returns the formatted prompt
//!   verbatim, so you can assert on exactly what the model would have seen.
//! - [`Executor::with_responses`] — **scripted**: returns canned responses in
//!   order, and fails with [`MockError::OutOfResponses`] when the script runs
//!   dry.
//! - [`Executor::failing`] — **failing**: every call fails with
//!   [`MockError::Forced`], for testing error paths.
//!
//! Every executed prompt is recorded and can be inspected with
//! [`Executor::calls`].
//!
//! # Example
//!
//! ```
//! use llm_chain::{Parameters, traits::{Step as _, Executor as _}};
//! use llm_chain_mock::{Executor, Step};
//!
//! # futures_lite_block_on(async {
//! let step = Step::new("Summarize this text: {text}");
//! let executor = Executor::with_responses(["A concise summary."]);
//!
//! let prompt = step
//!     .format(&Parameters::new_with_text("..a very long text.."))
//!     .unwrap();
//! let response = executor.execute(prompt).await.unwrap();
//! assert_eq!(response, "A concise summary.");
//! assert_eq!(executor.calls(), ["Summarize this text: ..a very long text.."]);
//! # });
//! # fn futures_lite_block_on<F: std::future::Future>(fut: F) -> F::Output {
//! #     // A tiny single-future block_on so the doctest needs no runtime dep.
//! #     use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
//! #     fn raw() -> RawWaker { RawWaker::new(std::ptr::null(), &VTABLE) }
//! #     static VTABLE: RawWakerVTable = RawWakerVTable::new(|_| raw(), |_| {}, |_| {}, |_| {});
//! #     let waker = unsafe { Waker::from_raw(raw()) };
//! #     let mut cx = Context::from_waker(&waker);
//! #     let mut fut = std::pin::pin!(fut);
//! #     loop {
//! #         if let Poll::Ready(out) = fut.as_mut().poll(&mut cx) { return out; }
//! #     }
//! # }
//! ```

mod error;
mod executor;
mod step;

pub use error::MockError;
pub use executor::Executor;
pub use step::Step;
