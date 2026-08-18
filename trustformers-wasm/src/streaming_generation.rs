//! Streaming generation for real-time text generation
//!
//! This module provides streaming text generation capabilities, allowing models
//! to generate text progressively for improved user experience.

#![allow(dead_code)]
use crate::core::pipeline::TextGenerationPipeline;
use js_sys::{Array, Function, Object, Promise};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::format;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Wall-clock milliseconds since the Unix epoch. `js_sys::Date::now()`
/// unconditionally panics ("cannot call wasm-bindgen imported functions")
/// when called on non-wasm32 targets - there is no JS engine to call into -
/// so timing logic that needs to run under native `cargo test`/nextest
/// (which only ever computes *differences* between two readings, never the
/// absolute value itself) goes through this indirection instead. Same
/// pattern as `optimization::batch_processing::now_ms`.
#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    js_sys::Date::now()
}

#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

/// Streaming generation configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    chunk_size: usize,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    stream_delay_ms: u32,
    buffer_size: usize,
    enable_early_stopping: bool,
    stop_sequences: Vec<String>,
}

/// Streaming text generator.
///
/// `pipeline` is the real transformer model + tokenizer driving generation.
/// Before this fix, `StreamingGenerator` held no model reference at all -
/// `generate_token_chunk` fabricated tokens as `format!("token_{n}")` with
/// a `js_sys::Math::random()`-based "confidence" score, so every streamed
/// "generation" was entirely disconnected from any real model, weights, or
/// input prompt. A `StreamingGenerator` without a pipeline attached
/// (`pipeline: None`) now returns a structured error from
/// [`Self::start_streaming`] rather than silently falling back to
/// fabricated output.
#[wasm_bindgen]
pub struct StreamingGenerator {
    config: StreamingConfig,
    pipeline: Option<TextGenerationPipeline>,
    is_streaming: bool,
    current_session: Option<StreamingSession>,
    token_buffer: VecDeque<String>,
    callback_registry: Vec<StreamingCallback>,
    stats: StreamingStats,
}

/// Streaming session state
#[derive(Debug, Clone)]
struct StreamingSession {
    id: String,
    prompt: String,
    generated_tokens: Vec<String>,
    /// The full running token-id context (prompt + every real token
    /// generated so far), rebuilt into the model's input on every step -
    /// same "feed generated tokens back in" requirement as
    /// [`crate::core::pipeline::TextGenerationPipeline::generate_ids`].
    generated_ids: Vec<u32>,
    total_tokens: usize,
    start_time: f64,
    last_token_time: f64,
    is_complete: bool,
    completion_reason: CompletionReason,
}

/// Reason for completion
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CompletionReason {
    /// Reached maximum token limit
    MaxTokens,
    /// Hit stop sequence
    StopSequence,
    /// End of text token generated
    EndOfText,
    /// Manual stop requested
    ManualStop,
    /// Error occurred
    Error,
}

/// Streaming statistics
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct StreamingStats {
    total_sessions: u32,
    total_tokens_generated: u32,
    average_tokens_per_second: f32,
    current_session_tokens: u32,
    current_session_duration_ms: f32,
}

/// Token information for streaming
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct StreamingToken {
    token: String,
    confidence: f32,
    token_id: u32,
    timestamp: f64,
    is_stop_token: bool,
}

/// Streaming callback configuration
#[derive(Debug, Clone)]
struct StreamingCallback {
    callback_type: CallbackType,
    function: Function,
    enabled: bool,
}

/// Types of streaming callbacks
#[derive(Debug, Clone, Copy, PartialEq)]
enum CallbackType {
    Token,
    Chunk,
    Complete,
    Error,
    Progress,
}

/// Generation progress information
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct GenerationProgress {
    tokens_generated: u32,
    total_estimated_tokens: u32,
    progress_percentage: f32,
    tokens_per_second: f32,
    estimated_remaining_ms: f32,
}

#[wasm_bindgen]
impl StreamingConfig {
    /// Create a new streaming configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> StreamingConfig {
        StreamingConfig {
            chunk_size: 1,       // Generate 1 token at a time for real-time streaming
            max_tokens: 512,     // Maximum tokens to generate
            temperature: 0.8,    // Sampling temperature
            top_p: 0.9,          // Nucleus sampling
            top_k: 50,           // Top-K sampling
            stream_delay_ms: 50, // Delay between token emissions (50ms for smooth streaming)
            buffer_size: 32,     // Buffer size for token queue
            enable_early_stopping: true,
            stop_sequences: vec!["</s>".to_string(), "<|endoftext|>".to_string()],
        }
    }

    /// Set chunk size (tokens per streaming chunk)
    pub fn set_chunk_size(&mut self, chunk_size: usize) {
        self.chunk_size = chunk_size.clamp(1, 16); // Between 1 and 16 tokens
    }

    /// Set maximum tokens to generate
    pub fn set_max_tokens(&mut self, max_tokens: usize) {
        self.max_tokens = max_tokens.clamp(1, 2048); // Between 1 and 2048 tokens
    }

    /// Set sampling temperature
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature.clamp(0.1, 2.0); // Between 0.1 and 2.0
    }

    /// Set nucleus sampling probability
    pub fn set_top_p(&mut self, top_p: f32) {
        self.top_p = top_p.clamp(0.0, 1.0); // Between 0.0 and 1.0
    }

    /// Set top-K sampling value
    pub fn set_top_k(&mut self, top_k: u32) {
        self.top_k = top_k.clamp(1, 1000); // Between 1 and 1000
    }

    /// Set streaming delay in milliseconds
    pub fn set_stream_delay_ms(&mut self, delay_ms: u32) {
        self.stream_delay_ms = delay_ms.clamp(10, 1000); // Between 10ms and 1000ms
    }

    /// Set buffer size for token queue
    pub fn set_buffer_size(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size.clamp(4, 128); // Between 4 and 128 tokens
    }

    /// Enable or disable early stopping
    pub fn set_early_stopping(&mut self, enabled: bool) {
        self.enable_early_stopping = enabled;
    }

    /// Add stop sequence
    pub fn add_stop_sequence(&mut self, sequence: String) {
        if !self.stop_sequences.contains(&sequence) {
            self.stop_sequences.push(sequence);
        }
    }

    /// Clear stop sequences
    pub fn clear_stop_sequences(&mut self) {
        self.stop_sequences.clear();
    }

    #[wasm_bindgen(getter)]
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    #[wasm_bindgen(getter)]
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    #[wasm_bindgen(getter)]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    #[wasm_bindgen(getter)]
    pub fn top_p(&self) -> f32 {
        self.top_p
    }

    #[wasm_bindgen(getter)]
    pub fn top_k(&self) -> u32 {
        self.top_k
    }

    #[wasm_bindgen(getter)]
    pub fn stream_delay_ms(&self) -> u32 {
        self.stream_delay_ms
    }

    #[wasm_bindgen(getter)]
    pub fn early_stopping(&self) -> bool {
        self.enable_early_stopping
    }

    /// Get stop sequences as JavaScript array
    pub fn get_stop_sequences(&self) -> Array {
        let sequences = Array::new();
        for seq in &self.stop_sequences {
            sequences.push(&seq.into());
        }
        sequences
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl StreamingGenerator {
    /// Create a new streaming generator with no model attached yet. Call
    /// [`Self::set_pipeline`] before [`Self::start_streaming`], or
    /// `start_streaming` returns a structured "no model loaded" error
    /// rather than fabricating output.
    #[wasm_bindgen(constructor)]
    pub fn new(config: StreamingConfig) -> StreamingGenerator {
        StreamingGenerator {
            config,
            pipeline: None,
            is_streaming: false,
            current_session: None,
            token_buffer: VecDeque::new(),
            callback_registry: Vec::new(),
            stats: StreamingStats::new(),
        }
    }

    /// Attach the real model + tokenizer pipeline that
    /// [`Self::start_streaming`] will drive. Required before streaming can
    /// begin.
    pub fn set_pipeline(&mut self, pipeline: TextGenerationPipeline) {
        self.pipeline = Some(pipeline);
    }

    /// Whether a real generation pipeline has been attached via
    /// [`Self::set_pipeline`].
    #[wasm_bindgen(getter)]
    pub fn has_pipeline(&self) -> bool {
        self.pipeline.is_some()
    }

    /// Start streaming text generation.
    ///
    /// Returns a structured error - never fabricated text - when no
    /// pipeline has been attached via [`Self::set_pipeline`].
    pub async fn start_streaming(&mut self, prompt: &str) -> Result<String, JsValue> {
        if self.is_streaming {
            return Err("Already streaming. Stop current session first.".into());
        }
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Err(JsValue::from_str(
                "StreamingGenerator: no model pipeline attached; call set_pipeline() first",
            ));
        };

        let prompt_ids = pipeline.encode(prompt, true)?;

        let session_id = format!("session_{}", now_ms() as u64);
        let session = StreamingSession {
            id: session_id.clone(),
            prompt: prompt.to_string(),
            generated_tokens: Vec::new(),
            generated_ids: prompt_ids,
            total_tokens: 0,
            start_time: now_ms(),
            last_token_time: now_ms(),
            is_complete: false,
            completion_reason: CompletionReason::MaxTokens,
        };

        self.current_session = Some(session);
        self.is_streaming = true;
        self.token_buffer.clear();
        self.stats.total_sessions += 1;
        self.stats.current_session_tokens = 0;

        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(
            &format!("Starting streaming generation for prompt: {}", prompt).into(),
        );

        // Start the streaming process
        self.process_streaming_generation().await?;

        Ok(session_id)
    }

    /// Process streaming generation asynchronously
    async fn process_streaming_generation(&mut self) -> Result<(), JsValue> {
        let mut tokens_generated = 0;
        let start_time = now_ms();

        while self.is_streaming && tokens_generated < self.config.max_tokens {
            let generated_tokens = self.generate_token_chunk().await?;

            for token in generated_tokens {
                self.token_buffer.push_back(token.token.clone());

                if let Some(ref mut session) = self.current_session {
                    session.generated_tokens.push(token.token.clone());
                    session.total_tokens += 1;
                    session.last_token_time = now_ms();
                }

                self.stats.current_session_tokens += 1;
                tokens_generated += 1;

                // Check for stop sequences
                if self.should_stop(&token) {
                    if let Some(ref mut session) = self.current_session {
                        session.completion_reason = if token.is_stop_token {
                            CompletionReason::EndOfText
                        } else {
                            CompletionReason::StopSequence
                        };
                    }
                    break;
                }

                // Emit token callback
                self.emit_token_callback(&token).await?;

                // Process chunk if buffer is full or streaming delay
                if self.token_buffer.len() >= self.config.chunk_size {
                    self.emit_chunk_callback().await?;
                }

                // Streaming delay for smoother output
                if self.config.stream_delay_ms > 0 {
                    self.sleep(self.config.stream_delay_ms).await?;
                }
            }

            // Check if we should continue
            if !self.is_streaming {
                break;
            }
        }

        // Emit any remaining tokens in buffer
        if !self.token_buffer.is_empty() {
            self.emit_chunk_callback().await?;
        }

        // Complete the session
        if let Some(ref mut session) = self.current_session {
            session.is_complete = true;
            if session.completion_reason == CompletionReason::MaxTokens
                && tokens_generated >= self.config.max_tokens
            {
                session.completion_reason = CompletionReason::MaxTokens;
            }
        }

        self.is_streaming = false;
        self.update_stats(start_time);
        self.emit_completion_callback().await?;

        Ok(())
    }

    /// Generate a chunk of up to `self.config.chunk_size` tokens by running
    /// `self.config.chunk_size` real forward passes through the attached
    /// pipeline, feeding every newly generated token back into the running
    /// context before predicting the next one (same requirement as
    /// [`crate::core::pipeline::TextGenerationPipeline::generate_ids`] -
    /// this is the streaming, one-token-at-a-time equivalent of that same
    /// loop, sharing its `next_token_with_confidence` step).
    ///
    /// Previously this was a pure simulation: `token_text` was
    /// `format!("token_{n}")` (a counter, not vocabulary output) and
    /// `confidence` was `0.8 + Math::random() * 0.2` - no model, tokenizer,
    /// weights, or prompt was ever involved. `token_text` is now the
    /// tokenizer's real decoded output for the model's real predicted
    /// token id, and `confidence` is that token's real softmax probability
    /// from the model's own logits.
    ///
    /// Thin `JsValue`-wrapping shim around [`Self::generate_token_chunk_inner`]
    /// - see that method for why the split exists.
    async fn generate_token_chunk(&mut self) -> Result<Vec<StreamingToken>, JsValue> {
        self.generate_token_chunk_inner().await.map_err(|e| JsValue::from_str(&e))
    }

    /// Real logic behind [`Self::generate_token_chunk`], returning `String`
    /// errors rather than `JsValue`. `JsValue::from_str` (used to build the
    /// "no pipeline attached" / "no active session" error messages) panics
    /// on non-wasm32 native targets ("cannot call wasm-bindgen imported
    /// functions"), so - matching the "String-error inner / JsValue-wrapping
    /// outer" pattern used throughout this crate (see e.g.
    /// `lib.rs::require_loaded_model`) - native tests call this method
    /// directly to exercise the error paths without ever constructing a
    /// `JsValue`.
    async fn generate_token_chunk_inner(&mut self) -> Result<Vec<StreamingToken>, String> {
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Err(
                "StreamingGenerator: no model pipeline attached; call set_pipeline() first"
                    .to_string(),
            );
        };
        let max_position = pipeline.max_position_embeddings();

        let mut tokens = Vec::new();

        for i in 0..self.config.chunk_size {
            // Simulate token generation delay
            if i > 0 {
                self.sleep(10).await.map_err(|e| format!("{e:?}"))?; // Small delay between tokens in chunk
            }

            let context_ids =
                match self.current_session.as_ref() {
                    Some(session) => session.generated_ids.clone(),
                    None => return Err(
                        "StreamingGenerator: generate_token_chunk called with no active session"
                            .to_string(),
                    ),
                };
            if context_ids.len() >= max_position {
                break;
            }

            let pipeline = self.pipeline.as_ref().expect("checked above");
            let (token_id, confidence) = pipeline
                .next_token_with_confidence(&context_ids)
                .map_err(|e| e.as_string().unwrap_or_else(|| format!("{e:?}")))?;
            let token_text = pipeline
                .decode(std::vec![token_id], false)
                .map_err(|e| e.as_string().unwrap_or_else(|| format!("{e:?}")))?;
            let is_stop = self.is_stop_token(&token_text)
                || TextGenerationPipeline::is_eos_token(Some(token_id));

            if let Some(session) = self.current_session.as_mut() {
                session.generated_ids.push(token_id);
            }

            let token = StreamingToken {
                token: token_text,
                confidence,
                token_id,
                timestamp: now_ms(),
                is_stop_token: is_stop,
            };

            tokens.push(token);

            if is_stop {
                break;
            }
        }

        Ok(tokens)
    }

    /// Check if token is a stop token
    fn is_stop_token(&self, token: &str) -> bool {
        self.config.stop_sequences.iter().any(|seq| token.contains(seq))
    }

    /// Check if generation should stop
    fn should_stop(&self, token: &StreamingToken) -> bool {
        if token.is_stop_token {
            return true;
        }

        if self.config.enable_early_stopping {
            // Add additional early stopping logic here
            // For example, stop if confidence is too low
            if token.confidence < 0.3 {
                return true;
            }
        }

        false
    }

    /// Emit token callback
    async fn emit_token_callback(&self, token: &StreamingToken) -> Result<(), JsValue> {
        for callback in &self.callback_registry {
            if callback.callback_type == CallbackType::Token && callback.enabled {
                let token_obj = self.token_to_js_object(token)?;
                let _ = callback.function.call1(&JsValue::NULL, &token_obj);
            }
        }
        Ok(())
    }

    /// Emit chunk callback
    async fn emit_chunk_callback(&mut self) -> Result<(), JsValue> {
        if self.token_buffer.is_empty() {
            return Ok(());
        }

        let chunk_text: String = self.token_buffer.drain(..).collect::<Vec<_>>().join("");

        for callback in &self.callback_registry {
            if callback.callback_type == CallbackType::Chunk && callback.enabled {
                let chunk_obj = Object::new();
                js_sys::Reflect::set(&chunk_obj, &"text".into(), &JsValue::from_str(&chunk_text))?;
                js_sys::Reflect::set(&chunk_obj, &"timestamp".into(), &js_sys::Date::now().into())?;

                let _ = callback.function.call1(&JsValue::NULL, &chunk_obj);
            }
        }

        Ok(())
    }

    /// Emit completion callback
    async fn emit_completion_callback(&self) -> Result<(), JsValue> {
        for callback in &self.callback_registry {
            if callback.callback_type == CallbackType::Complete && callback.enabled {
                let result_obj = self.session_to_js_object()?;
                let _ = callback.function.call1(&JsValue::NULL, &result_obj);
            }
        }
        Ok(())
    }

    /// Convert token to JavaScript object
    fn token_to_js_object(&self, token: &StreamingToken) -> Result<Object, JsValue> {
        let obj = Object::new();
        js_sys::Reflect::set(&obj, &"token".into(), &token.token.clone().into())?;
        js_sys::Reflect::set(&obj, &"confidence".into(), &token.confidence.into())?;
        js_sys::Reflect::set(&obj, &"tokenId".into(), &token.token_id.into())?;
        js_sys::Reflect::set(&obj, &"timestamp".into(), &token.timestamp.into())?;
        js_sys::Reflect::set(&obj, &"isStopToken".into(), &token.is_stop_token.into())?;
        Ok(obj)
    }

    /// Convert session to JavaScript object
    fn session_to_js_object(&self) -> Result<Object, JsValue> {
        let obj = Object::new();

        if let Some(ref session) = self.current_session {
            js_sys::Reflect::set(&obj, &"id".into(), &session.id.clone().into())?;
            js_sys::Reflect::set(&obj, &"prompt".into(), &session.prompt.clone().into())?;
            js_sys::Reflect::set(
                &obj,
                &"generatedText".into(),
                &session.generated_tokens.join("").into(),
            )?;
            js_sys::Reflect::set(&obj, &"totalTokens".into(), &session.total_tokens.into())?;
            js_sys::Reflect::set(&obj, &"isComplete".into(), &session.is_complete.into())?;
            js_sys::Reflect::set(
                &obj,
                &"completionReason".into(),
                &format!("{:?}", session.completion_reason).into(),
            )?;
            js_sys::Reflect::set(
                &obj,
                &"durationMs".into(),
                &(session.last_token_time - session.start_time).into(),
            )?;
        }

        Ok(obj)
    }

    /// Sleep for specified milliseconds
    async fn sleep(&self, ms: u32) -> Result<(), JsValue> {
        let promise = Promise::new(&mut |resolve, _reject| {
            if let Some(window) = web_sys::window() {
                let _ = window
                    .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms as i32);
            }
        });

        wasm_bindgen_futures::JsFuture::from(promise).await?;
        Ok(())
    }

    /// Update statistics
    fn update_stats(&mut self, start_time: f64) {
        let duration_ms = now_ms() - start_time;
        self.stats.current_session_duration_ms = duration_ms as f32;

        if duration_ms > 0.0 {
            let tokens_per_second =
                (self.stats.current_session_tokens as f64 * 1000.0) / duration_ms;
            self.stats.average_tokens_per_second = tokens_per_second as f32;
        }

        self.stats.total_tokens_generated += self.stats.current_session_tokens;
    }

    /// Stop streaming generation
    pub fn stop_streaming(&mut self) {
        if self.is_streaming {
            self.is_streaming = false;
            if let Some(ref mut session) = self.current_session {
                session.completion_reason = CompletionReason::ManualStop;
            }
            #[cfg(target_arch = "wasm32")]
            web_sys::console::log_1(&"Streaming generation stopped manually".into());
        }
    }

    /// Register callback for streaming events
    pub fn on_token(&mut self, callback: Function) {
        self.callback_registry.push(StreamingCallback {
            callback_type: CallbackType::Token,
            function: callback,
            enabled: true,
        });
    }

    /// Register callback for chunk events
    pub fn on_chunk(&mut self, callback: Function) {
        self.callback_registry.push(StreamingCallback {
            callback_type: CallbackType::Chunk,
            function: callback,
            enabled: true,
        });
    }

    /// Register callback for completion events
    pub fn on_complete(&mut self, callback: Function) {
        self.callback_registry.push(StreamingCallback {
            callback_type: CallbackType::Complete,
            function: callback,
            enabled: true,
        });
    }

    /// Register callback for error events
    pub fn on_error(&mut self, callback: Function) {
        self.callback_registry.push(StreamingCallback {
            callback_type: CallbackType::Error,
            function: callback,
            enabled: true,
        });
    }

    /// Get current streaming status
    #[wasm_bindgen(getter)]
    pub fn is_streaming(&self) -> bool {
        self.is_streaming
    }

    /// Get current session ID
    pub fn get_current_session_id(&self) -> Option<String> {
        self.current_session.as_ref().map(|s| s.id.clone())
    }

    /// Get generation progress
    pub fn get_progress(&self) -> GenerationProgress {
        let tokens_generated = self.stats.current_session_tokens;
        let total_estimated = self.config.max_tokens as u32;
        let progress_percentage = if total_estimated > 0 {
            (tokens_generated as f32 / total_estimated as f32) * 100.0
        } else {
            0.0
        };

        let current_time = now_ms();
        let elapsed_time = if let Some(ref session) = self.current_session {
            current_time - session.start_time
        } else {
            0.0
        };

        let tokens_per_second = if elapsed_time > 0.0 {
            (tokens_generated as f64 * 1000.0) / elapsed_time
        } else {
            0.0
        };

        let remaining_tokens = total_estimated.saturating_sub(tokens_generated);
        let estimated_remaining_ms = if tokens_per_second > 0.0 {
            (remaining_tokens as f64 / tokens_per_second) * 1000.0
        } else {
            0.0
        };

        GenerationProgress {
            tokens_generated,
            total_estimated_tokens: total_estimated,
            progress_percentage,
            tokens_per_second: tokens_per_second as f32,
            estimated_remaining_ms: estimated_remaining_ms as f32,
        }
    }

    /// Get streaming statistics
    pub fn get_stats(&self) -> StreamingStats {
        self.stats.clone()
    }

    /// Clear callback registry
    pub fn clear_callbacks(&mut self) {
        self.callback_registry.clear();
    }
}

#[wasm_bindgen]
impl StreamingStats {
    /// Create new streaming statistics
    pub fn new() -> StreamingStats {
        StreamingStats {
            total_sessions: 0,
            total_tokens_generated: 0,
            average_tokens_per_second: 0.0,
            current_session_tokens: 0,
            current_session_duration_ms: 0.0,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn total_sessions(&self) -> u32 {
        self.total_sessions
    }

    #[wasm_bindgen(getter)]
    pub fn total_tokens_generated(&self) -> u32 {
        self.total_tokens_generated
    }

    #[wasm_bindgen(getter)]
    pub fn average_tokens_per_second(&self) -> f32 {
        self.average_tokens_per_second
    }

    #[wasm_bindgen(getter)]
    pub fn current_session_tokens(&self) -> u32 {
        self.current_session_tokens
    }

    #[wasm_bindgen(getter)]
    pub fn current_session_duration_ms(&self) -> f32 {
        self.current_session_duration_ms
    }

    /// Get statistics summary
    pub fn summary(&self) -> String {
        format!(
            "Sessions: {}, Total tokens: {}, Avg speed: {:.1} tokens/sec, Current: {} tokens in {:.1}ms",
            self.total_sessions,
            self.total_tokens_generated,
            self.average_tokens_per_second,
            self.current_session_tokens,
            self.current_session_duration_ms
        )
    }
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl StreamingToken {
    #[wasm_bindgen(getter)]
    pub fn token(&self) -> String {
        self.token.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    #[wasm_bindgen(getter)]
    pub fn token_id(&self) -> u32 {
        self.token_id
    }

    #[wasm_bindgen(getter)]
    pub fn timestamp(&self) -> f64 {
        self.timestamp
    }

    #[wasm_bindgen(getter)]
    pub fn is_stop_token(&self) -> bool {
        self.is_stop_token
    }
}

#[wasm_bindgen]
impl GenerationProgress {
    #[wasm_bindgen(getter)]
    pub fn tokens_generated(&self) -> u32 {
        self.tokens_generated
    }

    #[wasm_bindgen(getter)]
    pub fn total_estimated_tokens(&self) -> u32 {
        self.total_estimated_tokens
    }

    #[wasm_bindgen(getter)]
    pub fn progress_percentage(&self) -> f32 {
        self.progress_percentage
    }

    #[wasm_bindgen(getter)]
    pub fn tokens_per_second(&self) -> f32 {
        self.tokens_per_second
    }

    #[wasm_bindgen(getter)]
    pub fn estimated_remaining_ms(&self) -> f32 {
        self.estimated_remaining_ms
    }
}

/// Check if streaming generation is supported in the current environment
#[wasm_bindgen]
pub fn is_streaming_supported() -> bool {
    // Check for required APIs
    let js_code = r#"
        try {
            return typeof Promise !== 'undefined' &&
                   typeof setTimeout !== 'undefined' &&
                   typeof performance !== 'undefined' &&
                   typeof performance.now === 'function';
        } catch (e) {
            return false;
        }
    "#;

    js_sys::eval(js_code)
        .map(|result| result.as_bool().unwrap_or(false))
        .unwrap_or(false)
}

/// Get optimal streaming configuration for the current environment
#[wasm_bindgen]
pub fn get_optimal_streaming_config() -> StreamingConfig {
    let mut config = StreamingConfig::new();

    // Detect connection speed and adjust accordingly
    let js_code = r#"
        try {
            const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
            if (connection) {
                return connection.effectiveType || '4g';
            }
            return '4g';
        } catch (e) {
            return '4g';
        }
    "#;

    if let Ok(connection_type) = js_sys::eval(js_code) {
        if let Some(connection_str) = connection_type.as_string() {
            match connection_str.as_str() {
                "slow-2g" | "2g" => {
                    config.set_stream_delay_ms(200);
                    config.set_chunk_size(2);
                },
                "3g" => {
                    config.set_stream_delay_ms(100);
                    config.set_chunk_size(1);
                },
                "4g" => {
                    config.set_stream_delay_ms(50);
                    config.set_chunk_size(1);
                },
                _ => {
                    config.set_stream_delay_ms(50);
                    config.set_chunk_size(1);
                },
            }
        }
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::model::weights::{layer_prefix, NamedWeights};
    use crate::core::model::{ModelArchitecture, ModelConfig, WasmModel};
    use crate::core::pipeline::GenerationConfig;
    use crate::core::tensor::WasmTensor;
    use crate::core::tokenizer::TokenizerType;

    #[test]
    fn test_streaming_config() {
        let mut config = StreamingConfig::new();
        assert_eq!(config.chunk_size(), 1);
        assert_eq!(config.max_tokens(), 512);
        assert!(config.temperature() > 0.0);

        config.set_chunk_size(4);
        assert_eq!(config.chunk_size(), 4);

        config.set_temperature(1.0);
        assert_eq!(config.temperature(), 1.0);
    }

    #[test]
    fn test_streaming_stats() {
        let stats = StreamingStats::new();
        assert_eq!(stats.total_sessions(), 0);
        assert_eq!(stats.total_tokens_generated(), 0);
        assert_eq!(stats.average_tokens_per_second(), 0.0);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_feature_detection() {
        let _supported = is_streaming_supported();
        let _config = get_optimal_streaming_config();
    }

    // -----------------------------------------------------------------
    // `StreamingGenerator`: was a pure simulation (`format!("token_{n}")` +
    // `Math::random()`-based confidence) with no model reference at all.
    // These tests exercise the real (non-`#[wasm_bindgen]`-boundary)
    // `generate_token_chunk` step against a tiny real GPT2-shaped model.
    // -----------------------------------------------------------------

    /// Deterministic pseudo-random f32 generator, matching the pattern used
    /// in `core::pipeline`'s own tests.
    fn fill(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 8) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn tensor(data: Vec<f32>, shape: Vec<usize>) -> WasmTensor {
        WasmTensor::new(data, shape).expect("valid tensor")
    }

    /// Build a tiny, fully-populated GPT-2-shaped `TextGenerationPipeline`
    /// (real weights, real tokenizer, no network/wasm-bindgen boundary).
    fn build_test_pipeline() -> TextGenerationPipeline {
        let config = ModelConfig {
            architecture: ModelArchitecture::GPT2,
            vocab_size: 12,
            hidden_size: 8,
            num_layers: 2,
            num_heads: 2,
            max_position_embeddings: 16,
            intermediate_size: 10,
            hidden_dropout_prob: 0.0,
            attention_dropout_prob: 0.0,
        };
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let mut w = NamedWeights::new();
        let mut seed = 3u32;
        let mut next = |n: usize| {
            seed = seed.wrapping_add(211);
            fill(n, seed)
        };

        w.insert(
            "token_embeddings.weight",
            tensor(next(config.vocab_size * h), std::vec![config.vocab_size, h]),
        );
        w.insert(
            "position_embeddings.weight",
            tensor(
                next(config.max_position_embeddings * h),
                std::vec![config.max_position_embeddings, h],
            ),
        );
        for i in 0..config.num_layers {
            let p = layer_prefix(i);
            for name in ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"] {
                w.insert(
                    format!("{p}{name}.weight"),
                    tensor(next(h * h), std::vec![h, h]),
                );
            }
            w.insert(
                format!("{p}norm1.weight"),
                tensor(std::vec![1.0; h], std::vec![h]),
            );
            w.insert(
                format!("{p}norm2.weight"),
                tensor(std::vec![1.0; h], std::vec![h]),
            );
            w.insert(
                format!("{p}ffn.fc1.weight"),
                tensor(next(h * inter), std::vec![h, inter]),
            );
            w.insert(
                format!("{p}ffn.fc2.weight"),
                tensor(next(inter * h), std::vec![inter, h]),
            );
        }
        w.insert("final_norm.weight", tensor(std::vec![1.0; h], std::vec![h]));

        let vocab_size = config.vocab_size;
        let model = WasmModel::with_weights_for_test(config, w);
        let mut tokenizer = crate::core::tokenizer::WasmTokenizer::new(TokenizerType::BPE);
        // Real (if tiny) vocabulary, loaded through the `JsValue`-free
        // native-test path (`load_vocab_map`, not the `#[wasm_bindgen]`
        // `load_vocab`, which needs a real `JsValue` and would panic here) -
        // one single-byte-alphabet symbol per id, covering the whole
        // `vocab_size: 12` range so any argmax-selected token id decodes to
        // real text instead of erroring on "no vocabulary loaded".
        let vocab: std::collections::BTreeMap<String, u32> = (0u32..vocab_size as u32)
            .map(|id| (((b'a' + id as u8) as char).to_string(), id))
            .collect();
        tokenizer.load_vocab_map(vocab).expect("non-empty vocab");
        TextGenerationPipeline::new(model, tokenizer)
    }

    fn generator_for_test() -> StreamingGenerator {
        StreamingGenerator {
            config: StreamingConfig::new(),
            pipeline: None,
            is_streaming: false,
            current_session: None,
            token_buffer: VecDeque::new(),
            callback_registry: Vec::new(),
            stats: StreamingStats::new(),
        }
    }

    #[test]
    fn test_has_pipeline_reflects_set_pipeline() {
        let mut generator = generator_for_test();
        assert!(!generator.has_pipeline());
        generator.set_pipeline(build_test_pipeline());
        assert!(generator.has_pipeline());
    }

    /// Poll a `Future` to completion without a real async runtime. Every
    /// future used in these tests resolves quickly (no real timers/I/O
    /// beyond in-process `setTimeout`-driven sleeps, which are skipped by
    /// using `stream_delay_ms: 0` / a single-token chunk), mirroring the
    /// `pollster_block_on` helper used in `storage::streaming_loader`'s
    /// tests for the same "no async runtime available natively" reason.
    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        use std::pin::pin;
        use std::task::{Context, Poll, Waker};

        let mut future = pin!(future);
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        for _ in 0..10_000 {
            if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
                return output;
            }
        }
        panic!("future did not resolve within the poll budget");
    }

    #[test]
    fn test_generate_token_chunk_without_pipeline_errors_not_fabricates() {
        let mut generator = generator_for_test();
        generator.current_session = Some(StreamingSession {
            id: "s".to_string(),
            prompt: "hi".to_string(),
            generated_tokens: Vec::new(),
            generated_ids: std::vec![0u32],
            total_tokens: 0,
            start_time: 0.0,
            last_token_time: 0.0,
            is_complete: false,
            completion_reason: CompletionReason::MaxTokens,
        });

        // `generate_token_chunk_inner`, not `generate_token_chunk`: the
        // latter's `JsValue::from_str` error-wrapping panics on native,
        // non-wasm32 targets.
        let result = block_on(generator.generate_token_chunk_inner());
        assert!(
            result.is_err(),
            "must error, not fabricate tokens, when no pipeline is attached"
        );
    }

    #[test]
    fn test_generate_token_chunk_produces_real_decoded_tokens_not_counters() {
        // chunk_size 1 (not e.g. 3): `generate_token_chunk`'s inter-token
        // delay uses a real browser `setTimeout`-backed `Promise`
        // (`Self::sleep`), which panics on native, non-wasm32 targets
        // ("cannot call wasm-bindgen imported functions") - only reachable
        // once `i > 0` within the chunk loop, so a chunk size of 1 never
        // triggers it.
        let mut generator = generator_for_test();
        generator.config.set_chunk_size(1);
        let mut pipeline = build_test_pipeline();
        pipeline.set_config(GenerationConfig {
            max_length: 64,
            do_sample: false, // argmax: deterministic, no js_sys::Math::random()
            early_stopping: false,
            ..GenerationConfig::default()
        });
        generator.set_pipeline(pipeline);

        // Hand-crafted ids within the tiny test model's `vocab_size: 12`
        // (rather than `pipeline.encode("hello", true)`, whose BPE/vocab
        // token ids are unrelated to this model's tiny vocab and would
        // exceed it, making `forward` legitimately error).
        let prompt_ids: Vec<u32> = std::vec![1, 2, 3];
        generator.current_session = Some(StreamingSession {
            id: "s".to_string(),
            prompt: "hello".to_string(),
            generated_tokens: Vec::new(),
            generated_ids: prompt_ids.clone(),
            total_tokens: 0,
            start_time: 0.0,
            last_token_time: 0.0,
            is_complete: false,
            completion_reason: CompletionReason::MaxTokens,
        });

        let tokens = block_on(generator.generate_token_chunk()).expect("should generate");
        assert!(!tokens.is_empty());

        // Old fabricated tokens were always exactly `format!("token_{n}")`
        // for a monotonically increasing counter `n`, with confidence drawn
        // uniformly from [0.8, 1.0). Real tokens must not match that shape,
        // and confidence must be a genuine softmax probability in [0, 1].
        for token in &tokens {
            assert!(
                !token.token.starts_with("token_"),
                "token text looks like the old fabricated counter format: {}",
                token.token
            );
            assert!(
                (0.0..=1.0).contains(&token.confidence),
                "confidence must be a valid probability: {}",
                token.confidence
            );
        }

        // The session's running context must have grown by exactly the
        // number of tokens generated - proof the "feed generated tokens
        // back into context" requirement (mirroring
        // `TextGenerationPipeline::generate_ids`) is honored here too.
        let session = generator.current_session.as_ref().expect("session set");
        assert_eq!(session.generated_ids.len(), prompt_ids.len() + tokens.len());
    }

    #[test]
    fn test_generate_token_chunk_feeds_tokens_back_for_autoregressive_context() {
        // Regression test analogous to
        // `core::pipeline::tests::test_generate_ids_feeds_generated_tokens_back_into_context`:
        // generate two chunks back-to-back and confirm the second chunk's
        // starting context is strictly longer than the first's - i.e. each
        // call really does build on the previous one's output, rather than
        // e.g. always re-reading the original prompt.
        let mut generator = generator_for_test();
        generator.config.set_chunk_size(1);
        let mut pipeline = build_test_pipeline();
        pipeline.set_config(GenerationConfig {
            max_length: 64,
            do_sample: false,
            early_stopping: false,
            ..GenerationConfig::default()
        });
        generator.set_pipeline(pipeline);

        let prompt_ids: Vec<u32> = std::vec![1, 2];
        generator.current_session = Some(StreamingSession {
            id: "s".to_string(),
            prompt: "hi".to_string(),
            generated_tokens: Vec::new(),
            generated_ids: prompt_ids.clone(),
            total_tokens: 0,
            start_time: 0.0,
            last_token_time: 0.0,
            is_complete: false,
            completion_reason: CompletionReason::MaxTokens,
        });

        let _first = block_on(generator.generate_token_chunk()).expect("should generate");
        let len_after_first = generator.current_session.as_ref().unwrap().generated_ids.len();
        assert!(len_after_first > prompt_ids.len());

        let _second = block_on(generator.generate_token_chunk()).expect("should generate");
        let len_after_second = generator.current_session.as_ref().unwrap().generated_ids.len();
        assert!(len_after_second > len_after_first);
    }
}
