//! Incremental decoding of streaming model responses.
//!
//! Providers stream responses over HTTP in three framings: [server-sent
//! events](https://html.spec.whatwg.org/multipage/server-sent-events.html)
//! (OpenAI, Anthropic, Gemini), newline-delimited JSON (Ollama), and AWS's
//! binary event stream (Bedrock). This module houses the pieces every driver
//! shares:
//!
//! - [`SseDecoder`] and [`NdjsonDecoder`] — sans-IO push parsers: feed them
//!   raw bytes as they arrive, get complete frames back. They own all the
//!   fiddly parts (frames split across chunks, `\r\n` line endings, comment
//!   lines) and are directly unit-testable without a network.
//! - [`FrameDecoder`] — the trait both implement, so drivers with their own
//!   framing (Bedrock's binary event stream) plug into the same machinery.
//! - [`frames`] — adapts a fallible byte stream (e.g. `reqwest`'s
//!   `bytes_stream`) into a stream of decoded frames.
//!
//! Drivers layer their event types on top; applications usually interact with
//! streaming through
//! [`StreamingExecutor`](crate::traits::StreamingExecutor) instead of using
//! this module directly.

use std::collections::VecDeque;
use std::pin::Pin;

use futures::{Stream, StreamExt};

/// A single [server-sent event](https://html.spec.whatwg.org/multipage/server-sent-events.html):
/// an optional event name and the joined data payload.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SseEvent {
    /// The event name from the `event:` field, when the server sent one.
    pub event: Option<String>,
    /// The event payload: every `data:` line of the event joined with `\n`.
    pub data: String,
}

/// Buffers raw bytes and hands out complete text lines.
///
/// Understands `\n`, `\r\n` and lone-`\r` terminators, and holds back a
/// trailing `\r` until the next chunk shows whether it starts a `\r\n` pair.
/// Only complete lines are converted to text, so multi-byte UTF-8 sequences
/// split across chunk boundaries survive intact.
#[derive(Debug, Default)]
struct LineBuffer {
    buffer: Vec<u8>,
}

impl LineBuffer {
    fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Takes the next complete line, without its terminator.
    fn next_line(&mut self) -> Option<String> {
        let mut i = 0;
        while i < self.buffer.len() {
            let terminator_len = match self.buffer[i] {
                b'\n' => 1,
                b'\r' if i + 1 == self.buffer.len() => return None, // may be a split "\r\n"
                b'\r' if self.buffer[i + 1] == b'\n' => 2,
                b'\r' => 1,
                _ => {
                    i += 1;
                    continue;
                }
            };
            let line = String::from_utf8_lossy(&self.buffer[..i]).into_owned();
            self.buffer.drain(..i + terminator_len);
            return Some(line);
        }
        None
    }

    /// Takes whatever is left as a final, unterminated line.
    fn take_remainder(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }
        let line = String::from_utf8_lossy(&self.buffer).into_owned();
        self.buffer.clear();
        Some(line)
    }
}

/// An incremental decoder for a byte-stream framing: feed it raw bytes as
/// they arrive, get complete frames back.
///
/// Implementations are sans-IO — they never touch the network — which keeps
/// them directly unit-testable. Use [`frames`] to drive one from an async
/// byte stream.
pub trait FrameDecoder {
    /// The frame type this decoder produces.
    type Frame;
    /// Consumes a chunk of bytes, returning every frame it completed.
    fn feed(&mut self, bytes: &[u8]) -> Vec<Self::Frame>;
    /// Signals the end of input, returning any final buffered frames.
    fn finish(&mut self) -> Vec<Self::Frame>;
}

/// An incremental [server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html)
/// decoder.
///
/// Follows the WHATWG parsing rules: multiple `data:` lines join with `\n`,
/// comment lines (leading `:`) are ignored, `event:` names the event, and a
/// blank line dispatches it. `id:` and `retry:` fields are parsed and
/// discarded — reconnection is the HTTP client's concern. One deliberate
/// deviation: a final event missing its terminating blank line is flushed by
/// [`finish`](FrameDecoder::finish) rather than discarded, so a server that
/// closes the connection right after its last `data:` line loses nothing.
///
/// # Examples
///
/// ```
/// use llm_chain::streaming::{FrameDecoder, SseDecoder};
///
/// let mut decoder = SseDecoder::new();
/// // Chunk boundaries need not align with events:
/// let mut events = decoder.feed(b"event: delta\ndata: {\"text\":");
/// events.extend(decoder.feed(b" \"hi\"}\n\n"));
/// events.extend(decoder.finish());
/// assert_eq!(events.len(), 1);
/// assert_eq!(events[0].event.as_deref(), Some("delta"));
/// assert_eq!(events[0].data, "{\"text\": \"hi\"}");
/// ```
#[derive(Debug, Default)]
pub struct SseDecoder {
    lines: LineBuffer,
    event: Option<String>,
    data: String,
    /// Whether the UTF-8 BOM check for the very first bytes is still pending.
    at_start: bool,
}

impl SseDecoder {
    /// Creates a decoder at the start of a stream.
    pub fn new() -> Self {
        Self {
            lines: LineBuffer::default(),
            event: None,
            data: String::new(),
            at_start: true,
        }
    }

    /// Processes one complete line, returning a dispatched event on blank lines.
    fn process_line(&mut self, line: &str) -> Option<SseEvent> {
        if line.is_empty() {
            if self.data.is_empty() {
                // Nothing buffered: reset the event name and move on.
                self.event = None;
                return None;
            }
            let mut data = std::mem::take(&mut self.data);
            data.pop(); // remove the trailing newline the last data line added
            return Some(SseEvent {
                event: self.event.take(),
                data,
            });
        }
        if line.starts_with(':') {
            return None; // comment (also used as keep-alive by some servers)
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "data" => {
                self.data.push_str(value);
                self.data.push('\n');
            }
            "event" => self.event = Some(value.to_string()),
            // `id` and `retry` concern reconnection; unknown fields are ignored per spec.
            _ => {}
        }
        None
    }
}

impl FrameDecoder for SseDecoder {
    type Frame = SseEvent;

    fn feed(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        // Strip the UTF-8 BOM the spec allows before the first line.
        let bytes = if self.at_start && !bytes.is_empty() {
            self.at_start = false;
            bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes)
        } else {
            bytes
        };
        self.lines.push(bytes);
        let mut events = Vec::new();
        while let Some(line) = self.lines.next_line() {
            events.extend(self.process_line(&line));
        }
        events
    }

    fn finish(&mut self) -> Vec<SseEvent> {
        let mut events = Vec::new();
        if let Some(line) = self.lines.take_remainder() {
            events.extend(self.process_line(&line));
        }
        // Lenient flush: dispatch a final event that never got its blank line.
        if !self.data.is_empty() {
            events.extend(self.process_line(""));
        }
        events
    }
}

/// An incremental newline-delimited JSON decoder.
///
/// Yields each non-blank line as a `String`; parsing the JSON is the
/// caller's job, since the payload type is theirs. A final line missing its
/// terminating newline is flushed by [`finish`](FrameDecoder::finish).
///
/// # Examples
///
/// ```
/// use llm_chain::streaming::{FrameDecoder, NdjsonDecoder};
///
/// let mut decoder = NdjsonDecoder::new();
/// let mut lines = decoder.feed(b"{\"a\":1}\n{\"b\"");
/// lines.extend(decoder.feed(b":2}\n"));
/// lines.extend(decoder.finish());
/// assert_eq!(lines, vec!["{\"a\":1}", "{\"b\":2}"]);
/// ```
#[derive(Debug, Default)]
pub struct NdjsonDecoder {
    lines: LineBuffer,
}

impl NdjsonDecoder {
    /// Creates a decoder at the start of a stream.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FrameDecoder for NdjsonDecoder {
    type Frame = String;

    fn feed(&mut self, bytes: &[u8]) -> Vec<String> {
        self.lines.push(bytes);
        let mut out = Vec::new();
        while let Some(line) = self.lines.next_line() {
            if !line.trim().is_empty() {
                out.push(line);
            }
        }
        out
    }

    fn finish(&mut self) -> Vec<String> {
        self.lines
            .take_remainder()
            .filter(|line| !line.trim().is_empty())
            .into_iter()
            .collect()
    }
}

/// Adapts a fallible byte stream into a stream of decoded frames.
///
/// Bytes are pushed through `decoder` as they arrive; every completed frame
/// is yielded before the next chunk is read. When the source ends, the
/// decoder's [`finish`](FrameDecoder::finish) flushes any final frame. A
/// source error is yielded in place and ends the stream, since a byte-level
/// framing cannot be trusted after a transport failure.
///
/// This is the bridge drivers use between `reqwest`'s `bytes_stream()` and
/// their typed event streams.
pub fn frames<D, S, B, E>(decoder: D, source: S) -> impl Stream<Item = Result<D::Frame, E>> + Send
where
    D: FrameDecoder + Send + 'static,
    D::Frame: Send,
    S: Stream<Item = Result<B, E>> + Send + 'static,
    B: AsRef<[u8]> + Send,
    E: Send,
{
    struct State<D: FrameDecoder, S> {
        decoder: D,
        source: Pin<Box<S>>,
        pending: VecDeque<D::Frame>,
        done: bool,
    }
    let state = State {
        decoder,
        source: Box::pin(source),
        pending: VecDeque::new(),
        done: false,
    };
    futures::stream::unfold(state, |mut state| async move {
        loop {
            if let Some(frame) = state.pending.pop_front() {
                return Some((Ok(frame), state));
            }
            if state.done {
                return None;
            }
            match state.source.next().await {
                Some(Ok(bytes)) => state.pending.extend(state.decoder.feed(bytes.as_ref())),
                Some(Err(error)) => {
                    state.done = true;
                    return Some((Err(error), state));
                }
                None => {
                    state.done = true;
                    state.pending.extend(state.decoder.finish());
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_all(decoder: &mut SseDecoder, input: &[u8]) -> Vec<SseEvent> {
        let mut events = decoder.feed(input);
        events.extend(decoder.finish());
        events
    }

    #[test]
    fn parses_a_basic_event() {
        let mut decoder = SseDecoder::new();
        let events = feed_all(&mut decoder, b"data: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event, None);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn joins_multiple_data_lines_with_newlines() {
        let mut decoder = SseDecoder::new();
        let events = feed_all(&mut decoder, b"data: line one\ndata: line two\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line one\nline two");
    }

    #[test]
    fn carries_event_names_and_resets_between_events() {
        let mut decoder = SseDecoder::new();
        let events = feed_all(
            &mut decoder,
            b"event: message_start\ndata: {}\n\ndata: plain\n\n",
        );
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event.as_deref(), Some("message_start"));
        assert_eq!(events[1].event, None);
    }

    #[test]
    fn survives_arbitrary_chunk_boundaries() {
        let input = b"event: content_block_delta\ndata: {\"text\": \"\xC3\xA5\"}\n\n";
        // Split the input at every possible position, including mid-UTF-8.
        for split in 0..input.len() {
            let mut decoder = SseDecoder::new();
            let mut events = decoder.feed(&input[..split]);
            events.extend(decoder.feed(&input[split..]));
            events.extend(decoder.finish());
            assert_eq!(events.len(), 1, "split at {split}");
            assert_eq!(events[0].event.as_deref(), Some("content_block_delta"));
            assert_eq!(events[0].data, "{\"text\": \"å\"}", "split at {split}");
        }
    }

    #[test]
    fn handles_crlf_and_cr_line_endings() {
        let mut decoder = SseDecoder::new();
        let events = feed_all(&mut decoder, b"data: a\r\n\r\ndata: b\r\rdata: c\n\n");
        let data: Vec<_> = events.iter().map(|e| e.data.as_str()).collect();
        assert_eq!(data, ["a", "b", "c"]);
    }

    #[test]
    fn ignores_comments_and_unknown_fields() {
        let mut decoder = SseDecoder::new();
        let events = feed_all(
            &mut decoder,
            b": keep-alive\nid: 42\nretry: 100\nfuture: field\ndata: x\n\n",
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "x");
    }

    #[test]
    fn data_without_space_after_colon_is_kept_verbatim() {
        let mut decoder = SseDecoder::new();
        let events = feed_all(&mut decoder, b"data:no-space\ndata:  two-spaces\n\n");
        // Exactly one leading space is stripped per spec.
        assert_eq!(events[0].data, "no-space\n two-spaces");
    }

    #[test]
    fn flushes_a_final_unterminated_event() {
        let mut decoder = SseDecoder::new();
        let mut events = decoder.feed(b"data: tail");
        assert!(events.is_empty());
        events.extend(decoder.finish());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "tail");
    }

    #[test]
    fn blank_lines_without_data_dispatch_nothing() {
        let mut decoder = SseDecoder::new();
        let events = feed_all(&mut decoder, b"\n\nevent: named\n\n\n");
        assert!(events.is_empty());
    }

    #[test]
    fn strips_a_leading_utf8_bom() {
        let mut decoder = SseDecoder::new();
        let events = feed_all(&mut decoder, b"\xEF\xBB\xBFdata: x\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "x");
    }

    #[test]
    fn ndjson_splits_lines_and_skips_blanks() {
        let mut decoder = NdjsonDecoder::new();
        let mut lines = decoder.feed(b"{\"a\":1}\n\n{\"b\":2}\r\n{\"c\"");
        lines.extend(decoder.feed(b":3}"));
        lines.extend(decoder.finish());
        assert_eq!(lines, ["{\"a\":1}", "{\"b\":2}", "{\"c\":3}"]);
    }

    #[test]
    fn frames_adapter_decodes_across_chunks_and_flushes() {
        let chunks: Vec<Result<&[u8], std::convert::Infallible>> =
            vec![Ok(b"data: a\n\nda"), Ok(b"ta: b\n\ndata: c")];
        let stream = frames(SseDecoder::new(), futures::stream::iter(chunks));
        let events: Vec<_> = futures::executor::block_on(StreamExt::collect::<Vec<_>>(stream));
        let data: Vec<_> = events
            .into_iter()
            .map(|event| event.unwrap().data)
            .collect();
        assert_eq!(data, ["a", "b", "c"]);
    }

    #[test]
    fn frames_adapter_yields_source_errors_and_stops() {
        let chunks: Vec<Result<&[u8], String>> =
            vec![Ok(b"data: a\n\n"), Err("boom".to_string()), Ok(b"data: b\n\n")];
        let stream = frames(SseDecoder::new(), futures::stream::iter(chunks));
        let items: Vec<_> = futures::executor::block_on(StreamExt::collect::<Vec<_>>(stream));
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].as_ref().unwrap().data, "a");
        assert_eq!(items[1].as_ref().unwrap_err(), "boom");
    }
}
