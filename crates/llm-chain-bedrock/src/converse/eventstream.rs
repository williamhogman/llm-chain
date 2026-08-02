//! A sans-IO decoder for AWS's binary event stream framing
//! (`application/vnd.amazon.eventstream`), used by Bedrock's ConverseStream
//! API.
//!
//! Each message on the wire is a self-delimiting binary frame:
//!
//! ```text
//! +------------+-------------+-------------+---------+---------+-------------+
//! | total len  | headers len | prelude CRC | headers | payload | message CRC |
//! | u32 BE     | u32 BE      | u32 BE      |         |         | u32 BE      |
//! +------------+-------------+-------------+---------+---------+-------------+
//! ```
//!
//! The prelude CRC covers the first 8 bytes; the message CRC covers everything
//! before itself. Headers are name/typed-value pairs; Bedrock uses string
//! headers such as `:message-type` and `:event-type` to route payloads.
//!
//! [`EventStreamDecoder`] implements
//! [`FrameDecoder`](llm_chain::streaming::FrameDecoder), so it plugs into
//! [`llm_chain::streaming::frames`] like the SSE and NDJSON decoders. Because
//! a binary framing can genuinely be corrupt (unlike text framings, which
//! merely parse oddly), its frame type is a `Result`: a CRC mismatch or
//! malformed prelude yields one [`EventStreamError`] and poisons the decoder —
//! byte positions cannot be trusted after a framing failure.

use llm_chain::streaming::FrameDecoder;
use thiserror::Error;

/// Frames larger than this are rejected as malformed. The event stream spec
/// caps payloads well below this; anything bigger is a corrupt length field.
const MAX_FRAME_LEN: usize = 16 * 1024 * 1024;
/// Prelude: total length, headers length, prelude CRC — 4 bytes each.
const PRELUDE_LEN: usize = 12;
/// Prelude plus the trailing message CRC.
const OVERHEAD_LEN: usize = PRELUDE_LEN + 4;

/// An error in the binary event stream framing itself.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum EventStreamError {
    /// The prelude CRC did not match: the length fields cannot be trusted.
    #[error("event stream prelude CRC mismatch")]
    PreludeCrc,
    /// The message CRC did not match: the frame was corrupted in transit.
    #[error("event stream message CRC mismatch")]
    MessageCrc,
    /// A length field was inconsistent (frame too small, headers longer than
    /// the frame, or an absurd total length).
    #[error("malformed event stream frame: {0}")]
    Malformed(&'static str),
    /// The stream ended in the middle of a frame.
    #[error("event stream ended mid-frame ({0} bytes of an incomplete frame)")]
    Truncated(usize),
}

/// The value of one event stream header.
#[derive(Clone, Debug, PartialEq)]
pub enum HeaderValue {
    /// A boolean (wire types 0 and 1).
    Bool(bool),
    /// A single byte (wire type 2).
    Byte(i8),
    /// A 16-bit integer (wire type 3).
    Int16(i16),
    /// A 32-bit integer (wire type 4).
    Int32(i32),
    /// A 64-bit integer (wire type 5).
    Int64(i64),
    /// An opaque byte array (wire type 6).
    Bytes(Vec<u8>),
    /// A UTF-8 string (wire type 7) — the only type Bedrock uses in practice.
    String(String),
    /// A millisecond timestamp (wire type 8).
    Timestamp(i64),
    /// A UUID (wire type 9).
    Uuid([u8; 16]),
}

impl HeaderValue {
    /// The string value, when this header is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }
}

/// One decoded event stream message: its headers and raw payload.
///
/// Bedrock routes payloads by two string headers: `:message-type` (`event` or
/// `exception`) and `:event-type` / `:exception-type` (which event or
/// exception the JSON payload is).
#[derive(Clone, Debug, PartialEq)]
pub struct EventStreamMessage {
    /// The headers, in wire order.
    pub headers: Vec<(String, HeaderValue)>,
    /// The raw payload; JSON for every Bedrock message.
    pub payload: Vec<u8>,
}

impl EventStreamMessage {
    /// The value of the named header, when present.
    pub fn header(&self, name: &str) -> Option<&HeaderValue> {
        self.headers
            .iter()
            .find(|(header, _)| header == name)
            .map(|(_, value)| value)
    }

    /// The named header's string value, when present and a string.
    pub fn header_str(&self, name: &str) -> Option<&str> {
        self.header(name)?.as_str()
    }

    /// The `:message-type` header: `event`, `exception`, or `error`.
    pub fn message_type(&self) -> Option<&str> {
        self.header_str(":message-type")
    }

    /// The `:event-type` header, e.g. `contentBlockDelta`.
    pub fn event_type(&self) -> Option<&str> {
        self.header_str(":event-type")
    }

    /// The `:exception-type` header, e.g. `throttlingException`.
    pub fn exception_type(&self) -> Option<&str> {
        self.header_str(":exception-type")
    }

    /// Encodes this message into its wire format, computing both CRCs.
    ///
    /// The decoder's inverse; used by mocks and tests to produce byte-exact
    /// server responses.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut headers = Vec::new();
        for (name, value) in &self.headers {
            headers.push(name.len() as u8);
            headers.extend_from_slice(name.as_bytes());
            match value {
                HeaderValue::Bool(true) => headers.push(0),
                HeaderValue::Bool(false) => headers.push(1),
                HeaderValue::Byte(v) => {
                    headers.push(2);
                    headers.push(*v as u8);
                }
                HeaderValue::Int16(v) => {
                    headers.push(3);
                    headers.extend_from_slice(&v.to_be_bytes());
                }
                HeaderValue::Int32(v) => {
                    headers.push(4);
                    headers.extend_from_slice(&v.to_be_bytes());
                }
                HeaderValue::Int64(v) => {
                    headers.push(5);
                    headers.extend_from_slice(&v.to_be_bytes());
                }
                HeaderValue::Bytes(v) => {
                    headers.push(6);
                    headers.extend_from_slice(&(v.len() as u16).to_be_bytes());
                    headers.extend_from_slice(v);
                }
                HeaderValue::String(v) => {
                    headers.push(7);
                    headers.extend_from_slice(&(v.len() as u16).to_be_bytes());
                    headers.extend_from_slice(v.as_bytes());
                }
                HeaderValue::Timestamp(v) => {
                    headers.push(8);
                    headers.extend_from_slice(&v.to_be_bytes());
                }
                HeaderValue::Uuid(v) => {
                    headers.push(9);
                    headers.extend_from_slice(v);
                }
            }
        }
        let total = OVERHEAD_LEN + headers.len() + self.payload.len();
        let mut frame = Vec::with_capacity(total);
        frame.extend_from_slice(&(total as u32).to_be_bytes());
        frame.extend_from_slice(&(headers.len() as u32).to_be_bytes());
        frame.extend_from_slice(&crc32fast::hash(&frame[..8]).to_be_bytes());
        frame.extend_from_slice(&headers);
        frame.extend_from_slice(&self.payload);
        frame.extend_from_slice(&crc32fast::hash(&frame).to_be_bytes());
        frame
    }
}

/// An incremental decoder for AWS binary event stream frames.
///
/// Feed it raw bytes as they arrive; complete, CRC-verified messages come
/// back. Frames split across chunk boundaries are buffered until whole. On a
/// framing error (CRC mismatch, inconsistent lengths) it yields one `Err` and
/// poisons itself: byte positions cannot be trusted after corruption, so all
/// further input is dropped.
///
/// # Examples
///
/// ```
/// use llm_chain::streaming::FrameDecoder as _;
/// use llm_chain_bedrock::converse::{EventStreamDecoder, EventStreamMessage, HeaderValue};
///
/// let frame = EventStreamMessage {
///     headers: vec![
///         (":message-type".into(), HeaderValue::String("event".into())),
///         (":event-type".into(), HeaderValue::String("messageStop".into())),
///     ],
///     payload: br#"{"stopReason":"end_turn"}"#.to_vec(),
/// }
/// .to_bytes();
///
/// let mut decoder = EventStreamDecoder::new();
/// // Chunk boundaries need not align with frames:
/// let mut messages = decoder.feed(&frame[..7]);
/// messages.extend(decoder.feed(&frame[7..]));
/// messages.extend(decoder.finish());
/// assert_eq!(messages.len(), 1);
/// let message = messages.remove(0).unwrap();
/// assert_eq!(message.event_type(), Some("messageStop"));
/// ```
#[derive(Debug, Default)]
pub struct EventStreamDecoder {
    buffer: Vec<u8>,
    /// Set after a framing error; all further input is dropped.
    poisoned: bool,
}

impl EventStreamDecoder {
    /// Creates a decoder at the start of a stream.
    pub fn new() -> Self {
        Self::default()
    }

    /// Decodes the next complete frame from the buffer, if one is available.
    fn next_frame(&mut self) -> Option<Result<EventStreamMessage, EventStreamError>> {
        if self.buffer.len() < PRELUDE_LEN {
            return None;
        }
        let total = u32::from_be_bytes(self.buffer[0..4].try_into().unwrap()) as usize;
        let headers_len = u32::from_be_bytes(self.buffer[4..8].try_into().unwrap()) as usize;
        let prelude_crc = u32::from_be_bytes(self.buffer[8..12].try_into().unwrap());
        if crc32fast::hash(&self.buffer[0..8]) != prelude_crc {
            return Some(Err(EventStreamError::PreludeCrc));
        }
        if !(OVERHEAD_LEN..=MAX_FRAME_LEN).contains(&total) {
            return Some(Err(EventStreamError::Malformed(
                "total length out of range",
            )));
        }
        if headers_len > total - OVERHEAD_LEN {
            return Some(Err(EventStreamError::Malformed(
                "headers longer than frame",
            )));
        }
        if self.buffer.len() < total {
            return None;
        }
        let message_crc = u32::from_be_bytes(self.buffer[total - 4..total].try_into().unwrap());
        if crc32fast::hash(&self.buffer[..total - 4]) != message_crc {
            return Some(Err(EventStreamError::MessageCrc));
        }
        let headers = match parse_headers(&self.buffer[PRELUDE_LEN..PRELUDE_LEN + headers_len]) {
            Ok(headers) => headers,
            Err(error) => return Some(Err(error)),
        };
        let payload = self.buffer[PRELUDE_LEN + headers_len..total - 4].to_vec();
        self.buffer.drain(..total);
        Some(Ok(EventStreamMessage { headers, payload }))
    }
}

impl FrameDecoder for EventStreamDecoder {
    type Frame = Result<EventStreamMessage, EventStreamError>;

    fn feed(&mut self, bytes: &[u8]) -> Vec<Self::Frame> {
        if self.poisoned {
            return Vec::new();
        }
        self.buffer.extend_from_slice(bytes);
        let mut frames = Vec::new();
        while let Some(frame) = self.next_frame() {
            let errored = frame.is_err();
            frames.push(frame);
            if errored {
                self.poisoned = true;
                self.buffer.clear();
                break;
            }
        }
        frames
    }

    fn finish(&mut self) -> Vec<Self::Frame> {
        if self.poisoned || self.buffer.is_empty() {
            return Vec::new();
        }
        let leftover = self.buffer.len();
        self.buffer.clear();
        self.poisoned = true;
        vec![Err(EventStreamError::Truncated(leftover))]
    }
}

/// Parses the header block of a frame into name/value pairs.
fn parse_headers(mut bytes: &[u8]) -> Result<Vec<(String, HeaderValue)>, EventStreamError> {
    const MALFORMED: EventStreamError = EventStreamError::Malformed("truncated header block");
    let mut take = |n: usize| -> Result<&[u8], EventStreamError> {
        if bytes.len() < n {
            return Err(MALFORMED);
        }
        let (head, tail) = bytes.split_at(n);
        bytes = tail;
        Ok(head)
    };

    let mut headers = Vec::new();
    loop {
        let name_len = match take(1) {
            Ok(byte) => byte[0] as usize,
            // A clean end of the block: all headers parsed.
            Err(_) if bytes.is_empty() => return Ok(headers),
            Err(error) => return Err(error),
        };
        let name = String::from_utf8_lossy(take(name_len)?).into_owned();
        let value_type = take(1)?[0];
        let value = match value_type {
            0 => HeaderValue::Bool(true),
            1 => HeaderValue::Bool(false),
            2 => HeaderValue::Byte(take(1)?[0] as i8),
            3 => HeaderValue::Int16(i16::from_be_bytes(take(2)?.try_into().unwrap())),
            4 => HeaderValue::Int32(i32::from_be_bytes(take(4)?.try_into().unwrap())),
            5 => HeaderValue::Int64(i64::from_be_bytes(take(8)?.try_into().unwrap())),
            6 => {
                let len = u16::from_be_bytes(take(2)?.try_into().unwrap()) as usize;
                HeaderValue::Bytes(take(len)?.to_vec())
            }
            7 => {
                let len = u16::from_be_bytes(take(2)?.try_into().unwrap()) as usize;
                HeaderValue::String(String::from_utf8_lossy(take(len)?).into_owned())
            }
            8 => HeaderValue::Timestamp(i64::from_be_bytes(take(8)?.try_into().unwrap())),
            9 => HeaderValue::Uuid(take(16)?.try_into().unwrap()),
            _ => return Err(EventStreamError::Malformed("unknown header value type")),
        };
        headers.push((name, value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(event_type: &str, payload: &str) -> EventStreamMessage {
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
    }

    #[test]
    fn round_trips_a_message() {
        let message = event("contentBlockDelta", r#"{"delta":{"text":"hi"}}"#);
        let mut decoder = EventStreamDecoder::new();
        let mut frames = decoder.feed(&message.to_bytes());
        frames.extend(decoder.finish());
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].as_ref().unwrap(), &message);
    }

    #[test]
    fn survives_arbitrary_chunk_boundaries() {
        let mut wire = event("messageStart", r#"{"role":"assistant"}"#).to_bytes();
        wire.extend(event("messageStop", r#"{"stopReason":"end_turn"}"#).to_bytes());
        for split in 0..wire.len() {
            let mut decoder = EventStreamDecoder::new();
            let mut frames = decoder.feed(&wire[..split]);
            frames.extend(decoder.feed(&wire[split..]));
            frames.extend(decoder.finish());
            assert_eq!(frames.len(), 2, "split at {split}");
            assert_eq!(
                frames[0].as_ref().unwrap().event_type(),
                Some("messageStart")
            );
            assert_eq!(
                frames[1].as_ref().unwrap().event_type(),
                Some("messageStop")
            );
        }
    }

    #[test]
    fn all_header_value_types_round_trip() {
        let message = EventStreamMessage {
            headers: vec![
                ("b1".to_string(), HeaderValue::Bool(true)),
                ("b0".to_string(), HeaderValue::Bool(false)),
                ("byte".to_string(), HeaderValue::Byte(-5)),
                ("i16".to_string(), HeaderValue::Int16(-300)),
                ("i32".to_string(), HeaderValue::Int32(70_000)),
                ("i64".to_string(), HeaderValue::Int64(-5_000_000_000)),
                ("bytes".to_string(), HeaderValue::Bytes(vec![1, 2, 3])),
                ("str".to_string(), HeaderValue::String("hej".to_string())),
                ("ts".to_string(), HeaderValue::Timestamp(1_753_862_400_000)),
                ("uuid".to_string(), HeaderValue::Uuid([7; 16])),
            ],
            payload: b"{}".to_vec(),
        };
        let mut decoder = EventStreamDecoder::new();
        let frames = decoder.feed(&message.to_bytes());
        assert_eq!(frames[0].as_ref().unwrap(), &message);
    }

    #[test]
    fn corrupt_payload_fails_the_message_crc_and_poisons() {
        let mut wire = event("messageStop", r#"{"stopReason":"end_turn"}"#).to_bytes();
        let payload_start = wire.len() - 4 - 25;
        wire[payload_start] ^= 0xFF;
        let mut decoder = EventStreamDecoder::new();
        let frames = decoder.feed(&wire);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], Err(EventStreamError::MessageCrc));
        // Poisoned: further (valid) input is dropped.
        let more = decoder.feed(&event("ping", "{}").to_bytes());
        assert!(more.is_empty());
        assert!(decoder.finish().is_empty());
    }

    #[test]
    fn corrupt_prelude_is_detected_before_lengths_are_trusted() {
        let mut wire = event("messageStop", "{}").to_bytes();
        wire[0] ^= 0xFF; // absurd total length, but the CRC catches it first
        let mut decoder = EventStreamDecoder::new();
        let frames = decoder.feed(&wire);
        assert_eq!(frames, vec![Err(EventStreamError::PreludeCrc)]);
    }

    #[test]
    fn truncated_stream_is_reported_on_finish() {
        let wire = event("messageStop", "{}").to_bytes();
        let mut decoder = EventStreamDecoder::new();
        assert!(decoder.feed(&wire[..wire.len() - 3]).is_empty());
        let frames = decoder.finish();
        assert_eq!(
            frames,
            vec![Err(EventStreamError::Truncated(wire.len() - 3))]
        );
    }

    #[test]
    fn empty_stream_finishes_silently() {
        let mut decoder = EventStreamDecoder::new();
        assert!(decoder.finish().is_empty());
    }
}
