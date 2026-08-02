//! Welcome to the `traits` module! This is where llm-chain houses its public traits, which define the essential behavior of steps and executors. These traits are the backbone of our library and are used to implement a new model.
//!
//! Let's break it down:
//! - **Steps**: These are the building blocks that make up the chains. Steps define the parameters, including the prompt that is sent to the LLM.
//! - **Executors**: These are the workhorses that perform the steps. They take the output of a step and invoke the model on it to get the output.
//!
//! By implementing these traits, you can set up a new model and use it in your application. Your step defines the input to the model, and your executor invokes the model and returns the output. The output of the executor is then passed to the next step in the chain, and so on.
//!
//! Both formatting a step and executing it are fallible: errors are surfaced as
//! typed associated errors rather than panics.

use crate::{Parameters, chains::sequential};

/// A step is a single step in a chain. It takes a set of parameters and returns a formatted prompt that can be used by an executor.
pub trait Step {
    /// The formatted prompt type produced by this step, consumed by a matching executor.
    type Output: Send;
    /// The error type produced when formatting fails (e.g. a missing template parameter).
    type Error: std::error::Error + Send + Sync + 'static;
    /// Formats the step with the given parameters, producing the executor input.
    fn format(&self, parameters: &Parameters) -> Result<Self::Output, Self::Error>;
}

impl<T: ?Sized> StepExt for T where T: Step {}
/// Convenience extensions available on every [`Step`].
pub trait StepExt: Step {
    /// Wraps this single step in a [`sequential::Chain`].
    fn to_chain(self) -> sequential::Chain<Self>
    where
        Self: Sized,
    {
        sequential::Chain::of_one(self)
    }
}

/// An executor performs a single step in a chain. It takes a step's formatted output, executes it against the model, and returns the result.
///
/// The `execute` method is a native `async fn`; the [`trait_variant`] attribute
/// guarantees the returned futures are `Send`, so chains can be driven from
/// multi-threaded runtimes.
#[trait_variant::make(Send)]
pub trait Executor {
    /// The step type this executor accepts.
    type Step: Step;
    /// The model output type.
    type Output: Send;
    /// The error type produced when execution fails.
    type Error: std::error::Error + Send + Sync + 'static;
    /// Executes the formatted input against the model.
    async fn execute(
        &self,
        input: <<Self as Executor>::Step as Step>::Output,
    ) -> Result<Self::Output, Self::Error>;
    /// Feeds an output back into the parameters for the next step in a chain.
    fn apply_output_to_parameters(parameters: Parameters, output: &Self::Output) -> Parameters;
    /// Combines two outputs into a single output, used by map-reduce chains.
    fn combine_outputs(output: &Self::Output, other: &Self::Output) -> Self::Output;
}

/// A boxed stream of results, as returned by
/// [`StreamingExecutor::execute_stream`].
///
/// Boxed because every driver's stream is an opaque chain of adapters over
/// its HTTP response; a nameable type keeps the trait object-safe to hold
/// and pass around.
pub type BoxStream<T, E> = std::pin::Pin<Box<dyn futures::Stream<Item = Result<T, E>> + Send>>;

/// An executor that can stream a response while the model generates it.
///
/// Where [`Executor::execute`] buffers the whole response,
/// [`execute_stream`](StreamingExecutor::execute_stream) resolves as soon as
/// the model starts answering and yields typed events as they arrive. Each
/// driver keeps its own event type ([`StreamEvent`](StreamingExecutor::StreamEvent))
/// mirroring its provider's wire protocol, and offers an accumulator to fold
/// the events back into the driver's regular
/// [`Output`](Executor::Output) when both live output and the final response
/// are wanted.
///
/// [`text_delta`](StreamingExecutor::text_delta) extracts the newly generated
/// text from an event, so provider-agnostic code can print tokens without
/// knowing the event type:
///
/// ```ignore
/// use futures::StreamExt as _;
/// use llm_chain::traits::StreamingExecutor;
///
/// let mut stream = executor.execute_stream(step.format(&parameters)?).await?;
/// while let Some(event) = stream.next().await {
///     if let Some(text) = E::text_delta(&event?) {
///         print!("{text}");
///     }
/// }
/// ```
#[trait_variant::make(Send)]
pub trait StreamingExecutor: Executor {
    /// The event type yielded while a response is being generated, mirroring
    /// the provider's streaming wire protocol.
    type StreamEvent: Send + 'static;

    /// Starts executing the formatted input, resolving to a stream of events
    /// once the model begins answering.
    ///
    /// Errors that occur before any output is produced (bad credentials,
    /// unknown model, rate limits) are returned directly; errors during
    /// generation are yielded inside the stream, which then ends.
    async fn execute_stream(
        &self,
        input: <<Self as Executor>::Step as Step>::Output,
    ) -> Result<BoxStream<Self::StreamEvent, <Self as Executor>::Error>, <Self as Executor>::Error>;

    /// The newly generated answer text carried by an event, if any.
    ///
    /// Returns `None` for bookkeeping events (message boundaries, usage
    /// reports, pings) and for non-answer output such as reasoning deltas.
    fn text_delta(event: &Self::StreamEvent) -> Option<std::borrow::Cow<'_, str>>;
}
