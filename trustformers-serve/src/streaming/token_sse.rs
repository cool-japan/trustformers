// Allow dead code for infrastructure under development
#![allow(dead_code)]

//! SSE (Server-Sent Events) streaming for real-time token generation
//!
//! This module provides token-by-token SSE streaming with OpenAI-compatible
//! chunk formats for real-time LLM inference responses.

use std::fmt;

/// A single SSE event with wire-format serialization
#[derive(Debug, Clone)]
pub struct SseEvent {
    /// Event type (e.g., "token", "done", "error")
    pub event_type: String,
    /// Event data (JSON-serializable string)
    pub data: String,
    /// Optional event ID for reconnection
    pub id: Option<String>,
    /// Optional retry interval in milliseconds
    pub retry_ms: Option<u64>,
}

impl SseEvent {
    /// Create a new SSE event
    pub fn new(event_type: &str, data: &str) -> Self {
        Self {
            event_type: event_type.to_string(),
            data: data.to_string(),
            id: None,
            retry_ms: None,
        }
    }

    /// Set the event ID for reconnection support
    pub fn with_id(mut self, id: &str) -> Self {
        self.id = Some(id.to_string());
        self
    }

    /// Set the retry interval in milliseconds
    pub fn with_retry(mut self, retry_ms: u64) -> Self {
        self.retry_ms = Some(retry_ms);
        self
    }

    /// Create a token event for streaming generation
    pub fn token(token: &str, index: usize) -> Self {
        let data = format!(
            r#"{{"token":{},"index":{}}}"#,
            json_escape_str(token),
            index
        );
        Self::new("token", &data)
    }

    /// Create a done event signaling end of stream
    pub fn done() -> Self {
        Self::new("done", r#"{"done":true}"#)
    }

    /// Create an error event
    pub fn error(message: &str) -> Self {
        let data = format!(r#"{{"error":{}}}"#, json_escape_str(message));
        Self::new("error", &data)
    }
}

impl fmt::Display for SseEvent {
    /// Format as SSE wire format:
    /// "event: {type}\ndata: {data}\nid: {id}\nretry: {ms}\n\n"
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "event: {}", self.event_type)?;
        writeln!(f, "data: {}", self.data)?;
        if let Some(ref id) = self.id {
            writeln!(f, "id: {}", id)?;
        }
        if let Some(retry_ms) = self.retry_ms {
            writeln!(f, "retry: {}", retry_ms)?;
        }
        writeln!(f)
    }
}

/// Token chunk for streaming response
#[derive(Debug, Clone)]
pub struct TokenChunk {
    /// The token text
    pub token: String,
    /// Token ID from vocabulary
    pub token_id: u32,
    /// Log-probability of this token
    pub log_prob: f32,
    /// Whether this is the last token
    pub is_last: bool,
    /// Optional finish reason for the last token
    pub finish_reason: Option<FinishReason>,
}

impl TokenChunk {
    /// Create a new token chunk
    pub fn new(token: &str, token_id: u32, log_prob: f32) -> Self {
        Self {
            token: token.to_string(),
            token_id,
            log_prob,
            is_last: false,
            finish_reason: None,
        }
    }

    /// Mark this as the last token with a finish reason
    pub fn as_last(mut self, reason: FinishReason) -> Self {
        self.is_last = true;
        self.finish_reason = Some(reason);
        self
    }
}

/// Reason for generation termination
#[derive(Debug, Clone, PartialEq)]
pub enum FinishReason {
    /// max_tokens limit was reached
    Length,
    /// Stop token was encountered
    Stop,
    /// An error occurred during generation
    Error,
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FinishReason::Length => write!(f, "length"),
            FinishReason::Stop => write!(f, "stop"),
            FinishReason::Error => write!(f, "error"),
        }
    }
}

/// Configuration for an SSE stream
pub struct SseStreamConfig {
    /// Include log_probs in token events
    pub include_log_probs: bool,
    /// Include token IDs in events
    pub include_token_ids: bool,
    /// Custom retry interval
    pub retry_ms: Option<u64>,
    /// Max tokens before forced finish
    pub max_tokens: usize,
}

impl Default for SseStreamConfig {
    fn default() -> Self {
        Self {
            include_log_probs: false,
            include_token_ids: false,
            retry_ms: None,
            max_tokens: 2048,
        }
    }
}

/// SSE stream builder — collects chunks and formats SSE events
pub struct SseStream {
    events: Vec<SseEvent>,
    config: SseStreamConfig,
    total_tokens: usize,
}

impl SseStream {
    /// Create a new SSE stream with the given configuration
    pub fn new(config: SseStreamConfig) -> Self {
        Self {
            events: Vec::new(),
            config,
            total_tokens: 0,
        }
    }

    /// Add a token chunk to the stream and get back the generated SSE event
    pub fn push_chunk(&mut self, chunk: TokenChunk) -> Result<SseEvent, SseError> {
        // Check if stream is already complete
        if self.is_complete() {
            return Err(SseError::StreamAlreadyComplete);
        }

        // Check max tokens
        if self.total_tokens >= self.config.max_tokens {
            return Err(SseError::MaxTokensExceeded {
                limit: self.config.max_tokens,
            });
        }

        // Validate chunk
        if chunk.token_id == u32::MAX {
            return Err(SseError::InvalidChunk(
                "token_id u32::MAX is reserved".to_string(),
            ));
        }

        self.total_tokens += 1;

        // Build data JSON based on config
        let data = build_token_event_data(
            &chunk,
            self.total_tokens - 1,
            self.config.include_log_probs,
            self.config.include_token_ids,
        );

        let mut event = SseEvent::new("token", &data);
        if let Some(retry_ms) = self.config.retry_ms {
            event = event.with_retry(retry_ms);
        }

        self.events.push(event.clone());

        // If this is the last chunk, append a done event
        if chunk.is_last {
            let done_event = SseEvent::done();
            self.events.push(done_event);
        }

        Ok(event)
    }

    /// Get all events in SSE wire format as a single string
    pub fn to_sse_string(&self) -> String {
        self.events.iter().map(|e| e.to_string()).collect()
    }

    /// Get a reference to the collected events
    pub fn events(&self) -> &[SseEvent] {
        &self.events
    }

    /// Total number of tokens generated
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Check if stream is complete (done or error event present)
    pub fn is_complete(&self) -> bool {
        self.events.iter().any(|e| e.event_type == "done" || e.event_type == "error")
    }
}

/// OpenAI-compatible streaming chunk format
#[derive(Debug, Clone)]
pub struct OpenAiStreamChunk {
    /// Unique chunk ID
    pub id: String,
    /// Object type — always "chat.completion.chunk"
    pub object: String,
    /// Unix timestamp
    pub created: u64,
    /// Model name
    pub model: String,
    /// List of choices (typically one)
    pub choices: Vec<StreamChoice>,
}

/// A single choice in a streaming chunk
#[derive(Debug, Clone)]
pub struct StreamChoice {
    /// Choice index
    pub index: usize,
    /// Delta content for this chunk
    pub delta: StreamDelta,
    /// Finish reason (only set in final chunk)
    pub finish_reason: Option<String>,
}

/// Delta content within a streaming choice
#[derive(Debug, Clone)]
pub struct StreamDelta {
    /// Role (only present in first chunk)
    pub role: Option<String>,
    /// Content token (absent in final chunk)
    pub content: Option<String>,
}

impl OpenAiStreamChunk {
    /// Create the first chunk which contains role information
    pub fn first_chunk(id: &str, model: &str, role: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp_secs(),
            model: model.to_string(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    role: Some(role.to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        }
    }

    /// Create a content chunk with token text
    pub fn content_chunk(id: &str, model: &str, content: &str, index: usize) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp_secs(),
            model: model.to_string(),
            choices: vec![StreamChoice {
                index,
                delta: StreamDelta {
                    role: None,
                    content: Some(content.to_string()),
                },
                finish_reason: None,
            }],
        }
    }

    /// Create the final chunk signaling end of stream
    pub fn final_chunk(id: &str, model: &str, finish_reason: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp_secs(),
            model: model.to_string(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some(finish_reason.to_string()),
            }],
        }
    }

    /// Serialize to a JSON string (manual serialization, no serde dependency required)
    pub fn to_json(&self) -> String {
        let choices_str: String =
            self.choices.iter().map(serialize_stream_choice).collect::<Vec<_>>().join(",");

        format!(
            r#"{{"id":{},"object":{},"created":{},"model":{},"choices":[{}]}}"#,
            json_escape_str(&self.id),
            json_escape_str(&self.object),
            self.created,
            json_escape_str(&self.model),
            choices_str,
        )
    }

    /// Format as SSE event: "data: {json}\n\n"
    pub fn to_sse(&self) -> String {
        format!("data: {}\n\n", self.to_json())
    }
}

/// Error types for SSE stream operations
#[derive(Debug)]
pub enum SseError {
    /// Stream has already received a done or error event
    StreamAlreadyComplete,
    /// Token limit was exceeded
    MaxTokensExceeded { limit: usize },
    /// Invalid chunk data
    InvalidChunk(String),
}

impl fmt::Display for SseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SseError::StreamAlreadyComplete => {
                write!(f, "SSE stream is already complete")
            },
            SseError::MaxTokensExceeded { limit } => {
                write!(f, "SSE stream exceeded max tokens limit of {}", limit)
            },
            SseError::InvalidChunk(msg) => {
                write!(f, "Invalid SSE chunk: {}", msg)
            },
        }
    }
}

impl std::error::Error for SseError {}

// ── private helpers ──────────────────────────────────────────────────────────

/// Build the JSON data payload for a token event
fn build_token_event_data(
    chunk: &TokenChunk,
    index: usize,
    include_log_probs: bool,
    include_token_ids: bool,
) -> String {
    let mut parts = vec![
        format!(r#""token":{}"#, json_escape_str(&chunk.token)),
        format!(r#""index":{}"#, index),
    ];

    if include_log_probs {
        parts.push(format!(r#""log_prob":{:.6}"#, chunk.log_prob));
    }

    if include_token_ids {
        parts.push(format!(r#""token_id":{}"#, chunk.token_id));
    }

    if chunk.is_last {
        parts.push(r#""is_last":true"#.to_string());
        if let Some(ref reason) = chunk.finish_reason {
            parts.push(format!(
                r#""finish_reason":{}"#,
                json_escape_str(&reason.to_string())
            ));
        }
    }

    format!("{{{}}}", parts.join(","))
}

/// Serialize a single StreamChoice to JSON
fn serialize_stream_choice(choice: &StreamChoice) -> String {
    let mut delta_parts = Vec::new();
    if let Some(ref role) = choice.delta.role {
        delta_parts.push(format!(r#""role":{}"#, json_escape_str(role)));
    }
    if let Some(ref content) = choice.delta.content {
        delta_parts.push(format!(r#""content":{}"#, json_escape_str(content)));
    }
    let delta_json = delta_parts.join(",");

    let finish_json = match &choice.finish_reason {
        Some(fr) => json_escape_str(fr),
        None => "null".to_string(),
    };

    format!(
        r#"{{"index":{},"delta":{{{}}},"finish_reason":{}}}"#,
        choice.index, delta_json, finish_json
    )
}

/// Escape a string for embedding in a JSON value (returns quoted string)
pub(crate) fn json_escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r#"\\"#),
            '\n' => out.push_str(r#"\n"#),
            '\r' => out.push_str(r#"\r"#),
            '\t' => out.push_str(r#"\t"#),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            },
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Get the current Unix timestamp in seconds
fn unix_timestamp_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_event_basic() {
        let event = SseEvent::new("token", "hello");
        assert_eq!(event.event_type, "token");
        assert_eq!(event.data, "hello");
        assert!(event.id.is_none());
        assert!(event.retry_ms.is_none());
    }

    #[test]
    fn test_sse_event_with_id() {
        let event = SseEvent::new("token", "world").with_id("evt-001");
        assert_eq!(event.id.as_deref(), Some("evt-001"));
    }

    #[test]
    fn test_sse_event_with_retry() {
        let event = SseEvent::new("token", "world").with_retry(3000);
        assert_eq!(event.retry_ms, Some(3000));
    }

    #[test]
    fn test_sse_event_token_format() {
        let event = SseEvent::token("hello", 0);
        assert_eq!(event.event_type, "token");
        assert!(event.data.contains("\"token\""));
        assert!(event.data.contains("\"index\""));
        assert!(event.data.contains("0"));
    }

    #[test]
    fn test_sse_event_done_format() {
        let event = SseEvent::done();
        assert_eq!(event.event_type, "done");
        assert!(event.data.contains("\"done\":true"));
    }

    #[test]
    fn test_sse_event_error_format() {
        let event = SseEvent::error("something went wrong");
        assert_eq!(event.event_type, "error");
        assert!(event.data.contains("\"error\""));
        assert!(event.data.contains("something went wrong"));
    }

    #[test]
    fn test_sse_event_display_wire_format() {
        let event = SseEvent::new("token", "hello").with_id("1").with_retry(1000);
        let wire = event.to_string();
        assert!(wire.starts_with("event: token\n"));
        assert!(wire.contains("data: hello\n"));
        assert!(wire.contains("id: 1\n"));
        assert!(wire.contains("retry: 1000\n"));
        assert!(wire.ends_with("\n\n"));
    }

    #[test]
    fn test_token_chunk_creation() {
        let chunk = TokenChunk::new("hello", 42, -0.5);
        assert_eq!(chunk.token, "hello");
        assert_eq!(chunk.token_id, 42);
        assert!((chunk.log_prob - (-0.5)).abs() < 1e-6);
        assert!(!chunk.is_last);
        assert!(chunk.finish_reason.is_none());
    }

    #[test]
    fn test_finish_reason_display() {
        assert_eq!(FinishReason::Length.to_string(), "length");
        assert_eq!(FinishReason::Stop.to_string(), "stop");
        assert_eq!(FinishReason::Error.to_string(), "error");
    }

    #[test]
    fn test_sse_stream_push_chunk() {
        let config = SseStreamConfig::default();
        let mut stream = SseStream::new(config);
        let chunk = TokenChunk::new("hello", 1, -0.3);
        let result = stream.push_chunk(chunk);
        assert!(result.is_ok());
        assert_eq!(stream.total_tokens(), 1);
        assert!(!stream.is_complete());
    }

    #[test]
    fn test_sse_stream_push_last_chunk() {
        let config = SseStreamConfig::default();
        let mut stream = SseStream::new(config);
        let chunk = TokenChunk::new("world", 2, -0.2).as_last(FinishReason::Stop);
        let result = stream.push_chunk(chunk);
        assert!(result.is_ok());
        // After a last chunk, the stream should be complete (done event was added)
        assert!(stream.is_complete());
        // Events: token + done
        assert_eq!(stream.events().len(), 2);
        assert_eq!(stream.events()[1].event_type, "done");
    }

    #[test]
    fn test_sse_stream_to_string() {
        let config = SseStreamConfig::default();
        let mut stream = SseStream::new(config);
        stream.push_chunk(TokenChunk::new("hi", 1, 0.0)).expect("push should succeed");
        let s = stream.to_sse_string();
        assert!(s.contains("event: token\n"));
        assert!(s.contains("data:"));
    }

    #[test]
    fn test_sse_stream_is_complete() {
        let config = SseStreamConfig::default();
        let mut stream = SseStream::new(config);
        assert!(!stream.is_complete());
        stream
            .push_chunk(TokenChunk::new("end", 5, 0.0).as_last(FinishReason::Length))
            .expect("push should succeed");
        assert!(stream.is_complete());
    }

    #[test]
    fn test_sse_stream_max_tokens() {
        let config = SseStreamConfig {
            max_tokens: 2,
            ..Default::default()
        };
        let mut stream = SseStream::new(config);
        stream.push_chunk(TokenChunk::new("a", 1, 0.0)).expect("push 1 should succeed");
        stream.push_chunk(TokenChunk::new("b", 2, 0.0)).expect("push 2 should succeed");
        let result = stream.push_chunk(TokenChunk::new("c", 3, 0.0));
        assert!(matches!(
            result,
            Err(SseError::MaxTokensExceeded { limit: 2 })
        ));
    }

    #[test]
    fn test_openai_stream_chunk_first() {
        let chunk = OpenAiStreamChunk::first_chunk("chatcmpl-001", "gpt-4", "assistant");
        assert_eq!(chunk.id, "chatcmpl-001");
        assert_eq!(chunk.object, "chat.completion.chunk");
        assert_eq!(chunk.model, "gpt-4");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.role.as_deref(), Some("assistant"));
        assert!(chunk.choices[0].delta.content.is_none());
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn test_openai_stream_chunk_content() {
        let chunk = OpenAiStreamChunk::content_chunk("chatcmpl-002", "gpt-4", "Hello", 0);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hello"));
        assert!(chunk.choices[0].delta.role.is_none());
    }

    #[test]
    fn test_openai_stream_chunk_final() {
        let chunk = OpenAiStreamChunk::final_chunk("chatcmpl-003", "gpt-4", "stop");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert!(chunk.choices[0].delta.content.is_none());
        assert!(chunk.choices[0].delta.role.is_none());
    }

    #[test]
    fn test_openai_stream_chunk_to_sse() {
        let chunk = OpenAiStreamChunk::content_chunk("id-1", "gpt-4", "Hi", 0);
        let sse = chunk.to_sse();
        assert!(sse.starts_with("data: "));
        assert!(sse.ends_with("\n\n"));
        assert!(sse.contains("\"object\":\"chat.completion.chunk\""));
        assert!(sse.contains("\"content\":\"Hi\""));
    }

    #[test]
    fn test_json_escape_special_chars() {
        let escaped = json_escape_str("say \"hello\"\nworld\\path");
        assert!(escaped.contains(r#"\""#));
        assert!(escaped.contains(r#"\n"#));
        assert!(escaped.contains(r#"\\"#));
    }

    #[test]
    fn test_sse_stream_already_complete_error() {
        let config = SseStreamConfig::default();
        let mut stream = SseStream::new(config);
        stream
            .push_chunk(TokenChunk::new("end", 1, 0.0).as_last(FinishReason::Stop))
            .expect("first push should succeed");
        let result = stream.push_chunk(TokenChunk::new("more", 2, 0.0));
        assert!(matches!(result, Err(SseError::StreamAlreadyComplete)));
    }
}
