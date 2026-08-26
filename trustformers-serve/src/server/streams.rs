//! Real streaming inference.
//!
//! `POST /v1/inference/stream` registers a stream, spawns a producer that runs
//! the real generation through the batching service, and pushes the resulting
//! chunks both into the SSE connection registry (so `GET /stream?request_id=…`
//! receives token events) and into an in-process store (so the chunks can be
//! inspected after the fact). No chunk is emitted that the model did not
//! produce.

use crate::batching::{
    aggregator::{ProcessingOutput, RequestInput},
    config::Priority,
    DynamicBatchingService, Request, RequestId,
};
use crate::streaming::{SseEvent, SseHandler};
use dashmap::DashMap;
use serde::Serialize;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Lifecycle of a streaming inference request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum StreamState {
    /// Registered; the producer has not emitted a chunk yet.
    Streaming,
    /// The producer emitted every chunk and closed the stream.
    Completed,
    /// Generation failed; `error` carries the reason.
    Failed,
}

impl StreamState {
    pub fn as_str(&self) -> &'static str {
        match self {
            StreamState::Streaming => "streaming",
            StreamState::Completed => "completed",
            StreamState::Failed => "failed",
        }
    }
}

/// Recorded state of one streaming inference request.
#[derive(Debug, Clone)]
pub struct StreamRecord {
    pub stream_id: String,
    pub state: StreamState,
    /// Chunks actually produced by the model, in order.
    pub chunks: Vec<String>,
    pub error: Option<String>,
    pub processing_time_ms: Option<f64>,
}

impl StreamRecord {
    /// JSON view returned by the stream status endpoint.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "stream_id": self.stream_id,
            "status": self.state.as_str(),
            "chunks": self.chunks,
            "text": self.chunks.concat(),
            "error": self.error,
            "processing_time_ms": self.processing_time_ms,
        })
    }
}

/// In-process registry of streaming inference requests.
#[derive(Debug, Default)]
pub struct StreamStore {
    streams: DashMap<String, StreamRecord>,
}

impl StreamStore {
    pub fn new() -> Self {
        Self {
            streams: DashMap::new(),
        }
    }

    /// Register a stream before its producer starts.
    pub fn register(&self, stream_id: String) {
        self.streams.insert(
            stream_id.clone(),
            StreamRecord {
                stream_id,
                state: StreamState::Streaming,
                chunks: Vec::new(),
                error: None,
                processing_time_ms: None,
            },
        );
    }

    /// Look up a stream. `None` means the id is genuinely unknown.
    pub fn get(&self, stream_id: &str) -> Option<StreamRecord> {
        self.streams.get(stream_id).map(|entry| entry.clone())
    }

    /// Number of registered streams.
    pub fn len(&self) -> usize {
        self.streams.len()
    }

    /// Whether the store holds no streams.
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// Count of streams currently in a given state.
    pub fn count_in_state(&self, state: StreamState) -> usize {
        self.streams.iter().filter(|entry| entry.state == state).count()
    }

    fn push_chunk(&self, stream_id: &str, chunk: String) {
        if let Some(mut entry) = self.streams.get_mut(stream_id) {
            entry.chunks.push(chunk);
        }
    }

    fn complete(&self, stream_id: &str, elapsed_ms: f64) {
        if let Some(mut entry) = self.streams.get_mut(stream_id) {
            entry.state = StreamState::Completed;
            entry.processing_time_ms = Some(elapsed_ms);
        }
    }

    fn fail(&self, stream_id: &str, error: String, elapsed_ms: f64) {
        if let Some(mut entry) = self.streams.get_mut(stream_id) {
            entry.state = StreamState::Failed;
            entry.error = Some(error);
            entry.processing_time_ms = Some(elapsed_ms);
        }
    }
}

/// Split generated text into stream chunks.
///
/// Whitespace-delimited chunks preserve the separator so that concatenating the
/// chunks reproduces the generated text exactly.
pub fn chunk_generated_text(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if ch.is_whitespace() {
            chunks.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Run one streaming inference request end to end.
///
/// Emits an SSE `token` event per produced chunk and a terminal `completion` or
/// `error` event. Waits briefly for a subscriber so a client that connects right
/// after receiving its `stream_id` does not miss the opening chunks.
pub async fn run_stream(
    store: Arc<StreamStore>,
    batching_service: Arc<DynamicBatchingService>,
    sse_handler: Arc<SseHandler>,
    stream_id: String,
    text: String,
    max_length: Option<usize>,
    subscriber_grace: Duration,
) {
    let started = Instant::now();

    // Give a client that has just received its stream_id a chance to connect.
    if !subscriber_grace.is_zero() {
        tokio::time::sleep(subscriber_grace).await;
    }

    let request = Request {
        id: RequestId::new(),
        input: RequestInput::Text { text, max_length },
        priority: Priority::Normal,
        submitted_at: Instant::now(),
        deadline: None,
        metadata: std::collections::HashMap::new(),
    };

    let outcome = batching_service.submit_request(request).await;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    let generated = match outcome {
        Ok(result) => match result.output {
            ProcessingOutput::Text(text) => Ok(text),
            ProcessingOutput::Tokens(tokens) => {
                Ok(tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "))
            },
            ProcessingOutput::Error(error) => Err(error),
            other => Err(format!("unsupported output type: {:?}", other)),
        },
        Err(e) => Err(e.to_string()),
    };

    match generated {
        Ok(text) => {
            for chunk in chunk_generated_text(&text) {
                store.push_chunk(&stream_id, chunk.clone());
                if let Err(e) =
                    sse_handler.send_to_request(&stream_id, SseEvent::token(chunk)).await
                {
                    tracing::debug!(
                        "stream {} could not deliver a token event: {}",
                        stream_id,
                        e
                    );
                }
            }
            store.complete(&stream_id, elapsed_ms);
            if let Err(e) =
                sse_handler.send_to_request(&stream_id, SseEvent::completion(text)).await
            {
                tracing::debug!(
                    "stream {} could not deliver the completion event: {}",
                    stream_id,
                    e
                );
            }
        },
        Err(error) => {
            store.fail(&stream_id, error.clone(), elapsed_ms);
            if let Err(e) = sse_handler.send_to_request(&stream_id, SseEvent::error(error)).await {
                tracing::debug!(
                    "stream {} could not deliver the error event: {}",
                    stream_id,
                    e
                );
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batching::config::BatchingConfig;
    use crate::batching::model_executor::{ByteTokenizer, Gpt2BatchModel};
    use crate::batching::ModelBatchExecutor;
    use crate::streaming::SseConfig;
    use trustformers_models::gpt2::Gpt2Config;

    fn tiny_executor() -> Arc<ModelBatchExecutor> {
        let config = Gpt2Config {
            vocab_size: ByteTokenizer::VOCAB_SIZE,
            n_positions: 32,
            n_embd: 8,
            n_layer: 1,
            n_head: 2,
            n_inner: Some(16),
            resid_pdrop: 0.0,
            embd_pdrop: 0.0,
            attn_pdrop: 0.0,
            ..Gpt2Config::default()
        };
        let model = Arc::new(Gpt2BatchModel::untrained(config).expect("model builds"));
        Arc::new(
            ModelBatchExecutor::new(model)
                .with_tokenizer(Arc::new(ByteTokenizer))
                .with_max_new_tokens(6),
        )
    }

    #[test]
    fn chunking_is_lossless() {
        for text in ["", "one", "one two", "a b  c\n"] {
            assert_eq!(chunk_generated_text(text).concat(), text);
        }
    }

    #[test]
    fn unknown_stream_is_unknown() {
        let store = StreamStore::new();
        assert!(store.get("nope").is_none());
        assert!(store.is_empty());
    }

    /// Regression: the streaming endpoint used to return a `stream_id` and never
    /// stream anything. A stream must now carry real generated chunks.
    #[tokio::test]
    async fn stream_carries_real_generated_chunks() {
        let store = Arc::new(StreamStore::new());
        let batching = Arc::new(DynamicBatchingService::with_executor(
            BatchingConfig::default(),
            tiny_executor(),
        ));
        batching.start().await.expect("service starts");
        let sse = Arc::new(SseHandler::new(SseConfig::default()));

        store.register("s-1".to_string());
        run_stream(
            Arc::clone(&store),
            batching,
            sse,
            "s-1".to_string(),
            "hello".to_string(),
            Some(6),
            Duration::ZERO,
        )
        .await;

        let record = store.get("s-1").expect("stream recorded");
        assert_eq!(
            record.state,
            StreamState::Completed,
            "stream failed: {:?}",
            record.error
        );
        assert!(
            !record.chunks.is_empty(),
            "a real stream must deliver at least one chunk"
        );
        assert!(record.processing_time_ms.unwrap_or(0.0) >= 0.0);
    }

    /// Regression: with no model the stream must fail loudly, not report success.
    #[tokio::test]
    async fn stream_without_model_fails_explicitly() {
        let store = Arc::new(StreamStore::new());
        let batching = Arc::new(DynamicBatchingService::new(BatchingConfig::default()));
        batching.start().await.expect("service starts");
        let sse = Arc::new(SseHandler::new(SseConfig::default()));

        store.register("s-2".to_string());
        run_stream(
            Arc::clone(&store),
            batching,
            sse,
            "s-2".to_string(),
            "hello".to_string(),
            Some(4),
            Duration::ZERO,
        )
        .await;

        let record = store.get("s-2").expect("stream recorded");
        assert_eq!(record.state, StreamState::Failed);
        assert!(record.chunks.is_empty());
        assert!(record.error.unwrap_or_default().contains("no model configured"));
    }
}
