//! Streaming generation for real-time text generation
//!
//! This module provides streaming text generation capabilities, allowing models
//! to generate text progressively for improved user experience.

#![allow(dead_code)]

use js_sys::{Array, Function, Object, Promise};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::format;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

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

/// Streaming text generator
#[wasm_bindgen]
pub struct StreamingGenerator {
    config: StreamingConfig,
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
    /// Create a new streaming generator
    #[wasm_bindgen(constructor)]
    pub fn new(config: StreamingConfig) -> StreamingGenerator {
        StreamingGenerator {
            config,
            is_streaming: false,
            current_session: None,
            token_buffer: VecDeque::new(),
            callback_registry: Vec::new(),
            stats: StreamingStats::new(),
        }
    }

    /// Start streaming text generation
    pub async fn start_streaming(&mut self, prompt: &str) -> Result<String, JsValue> {
        if self.is_streaming {
            return Err("Already streaming. Stop current session first.".into());
        }

        let session_id = format!("session_{}", js_sys::Date::now() as u64);
        let session = StreamingSession {
            id: session_id.clone(),
            prompt: prompt.to_string(),
            generated_tokens: Vec::new(),
            total_tokens: 0,
            start_time: js_sys::Date::now(),
            last_token_time: js_sys::Date::now(),
            is_complete: false,
            completion_reason: CompletionReason::MaxTokens,
        };

        self.current_session = Some(session);
        self.is_streaming = true;
        self.token_buffer.clear();
        self.stats.total_sessions += 1;
        self.stats.current_session_tokens = 0;

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
        let start_time = js_sys::Date::now();

        while self.is_streaming && tokens_generated < self.config.max_tokens {
            // Simulate token generation (in real implementation, this would call the model)
            let generated_tokens = self.generate_token_chunk().await?;

            for token in generated_tokens {
                self.token_buffer.push_back(token.token.clone());

                if let Some(ref mut session) = self.current_session {
                    session.generated_tokens.push(token.token.clone());
                    session.total_tokens += 1;
                    session.last_token_time = js_sys::Date::now();
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

    /// Generate a chunk of tokens (simulated)
    async fn generate_token_chunk(&self) -> Result<Vec<StreamingToken>, JsValue> {
        // This is a simulation - in real implementation, this would call the actual model
        let mut tokens = Vec::new();

        for i in 0..self.config.chunk_size {
            // Simulate token generation delay
            if i > 0 {
                self.sleep(10).await?; // Small delay between tokens in chunk
            }

            let token_text = format!("token_{}", self.stats.current_session_tokens + i as u32);
            let confidence = 0.8 + (js_sys::Math::random() * 0.2); // Random confidence 0.8-1.0
            let is_stop = self.is_stop_token(&token_text);

            let token = StreamingToken {
                token: token_text,
                confidence: confidence as f32,
                token_id: self.stats.current_session_tokens + i as u32,
                timestamp: js_sys::Date::now(),
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
            let timeout_id = web_sys::window()
                .expect("window should be available in browser context")
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms as i32)
                .expect("set_timeout should succeed with valid callback");
            let _ = timeout_id;
        });

        wasm_bindgen_futures::JsFuture::from(promise).await?;
        Ok(())
    }

    /// Update statistics
    fn update_stats(&mut self, start_time: f64) {
        let duration_ms = js_sys::Date::now() - start_time;
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

        let current_time = js_sys::Date::now();
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
}
