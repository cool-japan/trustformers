//! WebAssembly-compatible serving backend.
//!
//! This module provides a WASM-compatible HTTP-like serving interface that:
//! - Works in Cloudflare Workers, Deno Deploy, and other edge runtimes
//! - Uses a simple request/response model without OS threads
//! - Serializes requests/responses as JSON
//! - Integrates with trustformers-wasm for inference

mod wasm_extra_tests;

use std::collections::HashMap;
use std::time::Instant;

use thiserror::Error;

/// Errors from the WASM serving layer
#[derive(Debug, Error)]
pub enum ServeError {
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("Model not allowed: {0}")]
    ModelNotAllowed(String),
    #[error("Sequence too long: got {actual}, max {max}")]
    SequenceTooLong { actual: usize, max: usize },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// A WASM-compatible inference request
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WasmInferenceRequest {
    pub request_id: String,
    pub model_id: String,
    pub task: WasmTask,
    pub inputs: serde_json::Value,
    pub parameters: WasmInferenceParameters,
}

/// Supported WASM inference tasks
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WasmTask {
    TextGeneration,
    TextClassification,
    TokenClassification,
    QuestionAnswering,
    Summarization,
    FeatureExtraction,
    ImageClassification,
    AudioClassification,
}

/// Inference parameters for WASM requests
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WasmInferenceParameters {
    pub max_new_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub return_full_text: bool,
    pub truncation: bool,
    pub padding: bool,
    pub max_length: Option<usize>,
}

impl Default for WasmInferenceParameters {
    fn default() -> Self {
        Self {
            max_new_tokens: Some(128),
            temperature: Some(1.0),
            top_p: None,
            top_k: None,
            return_full_text: false,
            truncation: true,
            padding: false,
            max_length: None,
        }
    }
}

/// Response from WASM inference
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WasmInferenceResponse {
    pub request_id: String,
    pub model_id: String,
    pub outputs: serde_json::Value,
    pub latency_ms: f32,
    pub tokens_generated: Option<u32>,
    pub cached: bool,
    pub error: Option<WasmServeError>,
}

/// Serializable error for WASM responses
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WasmServeError {
    pub code: String,
    pub message: String,
}

/// WASM serving handler — stateless, serializable
pub struct WasmServingHandler {
    config: WasmServingConfig,
}

/// Configuration for the WASM serving handler
#[derive(Debug, Clone)]
pub struct WasmServingConfig {
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub cache_responses: bool,
    pub allowed_models: Vec<String>,
    pub cors_origins: Vec<String>,
}

impl Default for WasmServingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_sequence_length: 2048,
            cache_responses: true,
            allowed_models: vec!["default".to_string()],
            cors_origins: vec!["*".to_string()],
        }
    }
}

impl WasmServingHandler {
    /// Create a new handler with the given config
    pub fn new(config: WasmServingConfig) -> Self {
        Self { config }
    }

    /// Process a single inference request (synchronous, WASM-safe)
    pub fn handle(&self, request: WasmInferenceRequest) -> WasmInferenceResponse {
        let start = Instant::now();
        let request_id = request.request_id.clone();
        let model_id = request.model_id.clone();

        if let Err(e) = self.validate_request(&request) {
            return Self::error_response(&request_id, "VALIDATION_ERROR", &e.to_string());
        }

        // Stub inference output — real impl routes to trustformers-wasm
        let outputs = serde_json::json!({
            "generated_text": "[wasm-stub output]",
            "task": format!("{:?}", request.task),
        });

        let latency_ms = start.elapsed().as_secs_f32() * 1000.0;

        WasmInferenceResponse {
            request_id,
            model_id,
            outputs,
            latency_ms,
            tokens_generated: request.parameters.max_new_tokens,
            cached: false,
            error: None,
        }
    }

    /// Process a batch of requests
    pub fn handle_batch(&self, requests: Vec<WasmInferenceRequest>) -> Vec<WasmInferenceResponse> {
        let capped = requests.into_iter().take(self.config.max_batch_size).collect::<Vec<_>>();
        capped.into_iter().map(|r| self.handle(r)).collect()
    }

    /// Validate a request against the current config
    pub fn validate_request(&self, request: &WasmInferenceRequest) -> Result<(), ServeError> {
        if request.request_id.is_empty() {
            return Err(ServeError::InvalidRequest(
                "request_id must not be empty".to_string(),
            ));
        }
        if !self.config.allowed_models.is_empty()
            && !self.config.allowed_models.contains(&request.model_id)
        {
            return Err(ServeError::ModelNotAllowed(request.model_id.clone()));
        }
        if let Some(max_len) = request.parameters.max_length {
            if max_len > self.config.max_sequence_length {
                return Err(ServeError::SequenceTooLong {
                    actual: max_len,
                    max: self.config.max_sequence_length,
                });
            }
        }
        Ok(())
    }

    /// Serialize handler config to JSON (for edge deployment)
    pub fn to_config_json(&self) -> Result<String, ServeError> {
        let value = serde_json::json!({
            "max_batch_size": self.config.max_batch_size,
            "max_sequence_length": self.config.max_sequence_length,
            "cache_responses": self.config.cache_responses,
            "allowed_models": self.config.allowed_models,
            "cors_origins": self.config.cors_origins,
        });
        serde_json::to_string_pretty(&value).map_err(ServeError::Serialization)
    }

    /// Build HTTP response headers for CORS etc.
    pub fn build_cors_headers(&self, origin: &str) -> Vec<(String, String)> {
        let allowed = self.cors_origin_allowed(origin);
        let allow_origin = if allowed {
            origin.to_string()
        } else {
            // fallback: first allowed origin or "*"
            self.config.cors_origins.first().cloned().unwrap_or_else(|| "*".to_string())
        };

        vec![
            ("Access-Control-Allow-Origin".to_string(), allow_origin),
            (
                "Access-Control-Allow-Methods".to_string(),
                "GET, POST, OPTIONS".to_string(),
            ),
            (
                "Access-Control-Allow-Headers".to_string(),
                "Content-Type, Authorization".to_string(),
            ),
            ("Access-Control-Max-Age".to_string(), "86400".to_string()),
        ]
    }

    fn cors_origin_allowed(&self, origin: &str) -> bool {
        self.config.cors_origins.iter().any(|o| o == "*" || o == origin)
    }

    /// Create a health check response
    pub fn health_check(&self) -> serde_json::Value {
        serde_json::json!({
            "status": "ok",
            "max_batch_size": self.config.max_batch_size,
            "max_sequence_length": self.config.max_sequence_length,
            "allowed_models": self.config.allowed_models,
        })
    }

    /// Create an error response for the given request ID
    pub fn error_response(request_id: &str, code: &str, message: &str) -> WasmInferenceResponse {
        WasmInferenceResponse {
            request_id: request_id.to_string(),
            model_id: String::new(),
            outputs: serde_json::Value::Null,
            latency_ms: 0.0,
            tokens_generated: None,
            cached: false,
            error: Some(WasmServeError {
                code: code.to_string(),
                message: message.to_string(),
            }),
        }
    }
}

/// Edge platform selector
#[derive(Debug, Clone, PartialEq)]
pub enum EdgePlatform {
    CloudflareWorkers,
    DenoDeployAPI,
    VercelEdge,
    AwsLambdaEdge,
    Generic,
}

/// Edge deployment configuration generator
pub struct EdgeDeploymentConfig {
    pub platform: EdgePlatform,
    pub handler: WasmServingConfig,
    pub env_vars: HashMap<String, String>,
}

impl EdgeDeploymentConfig {
    /// Create a new edge deployment config
    pub fn new(platform: EdgePlatform, handler: WasmServingConfig) -> Self {
        Self {
            platform,
            handler,
            env_vars: HashMap::new(),
        }
    }

    /// Serialize config to JSON
    pub fn to_json(&self) -> Result<String, ServeError> {
        let value = serde_json::json!({
            "platform": self.platform_name(),
            "max_batch_size": self.handler.max_batch_size,
            "max_sequence_length": self.handler.max_sequence_length,
            "cache_responses": self.handler.cache_responses,
            "allowed_models": self.handler.allowed_models,
            "cors_origins": self.handler.cors_origins,
            "env_vars": self.env_vars,
        });
        serde_json::to_string_pretty(&value).map_err(ServeError::Serialization)
    }

    /// Platform-specific notes for deployment
    pub fn platform_specific_notes(&self) -> Vec<String> {
        match &self.platform {
            EdgePlatform::CloudflareWorkers => vec![
                "Deploy with `wrangler publish`".to_string(),
                "Set TRUSTFORMERS_MODEL_URL in Cloudflare Workers environment".to_string(),
                "Enable Workers Unbound for long-running inference".to_string(),
            ],
            EdgePlatform::DenoDeployAPI => vec![
                "Deploy with `deployctl deploy`".to_string(),
                "WASM modules must be imported via URL in Deno Deploy".to_string(),
                "Use `Deno.env.get` for secrets".to_string(),
            ],
            EdgePlatform::VercelEdge => vec![
                "Add `export const runtime = 'edge'` to your handler".to_string(),
                "Set environment variables in Vercel dashboard".to_string(),
                "Max execution time: 30s on Edge Runtime".to_string(),
            ],
            EdgePlatform::AwsLambdaEdge => vec![
                "Deploy as Lambda@Edge function in us-east-1".to_string(),
                "Max response payload: 1 MB for viewer-facing functions".to_string(),
                "Use SSM Parameter Store for secrets".to_string(),
            ],
            EdgePlatform::Generic => vec![
                "Use standard WASM init + handle pattern".to_string(),
                "Ensure no OS-thread dependencies".to_string(),
            ],
        }
    }

    fn platform_name(&self) -> &'static str {
        match &self.platform {
            EdgePlatform::CloudflareWorkers => "cloudflare_workers",
            EdgePlatform::DenoDeployAPI => "deno_deploy",
            EdgePlatform::VercelEdge => "vercel_edge",
            EdgePlatform::AwsLambdaEdge => "aws_lambda_edge",
            EdgePlatform::Generic => "generic",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_handler() -> WasmServingHandler {
        WasmServingHandler::new(WasmServingConfig {
            allowed_models: vec!["bert-base".to_string(), "gpt2".to_string()],
            ..Default::default()
        })
    }

    fn make_request(model_id: &str) -> WasmInferenceRequest {
        WasmInferenceRequest {
            request_id: "req-001".to_string(),
            model_id: model_id.to_string(),
            task: WasmTask::TextGeneration,
            inputs: serde_json::json!({"text": "Hello world"}),
            parameters: WasmInferenceParameters::default(),
        }
    }

    #[test]
    fn test_handle_valid_request() {
        let handler = make_handler();
        let req = make_request("bert-base");
        let resp = handler.handle(req);
        assert!(resp.error.is_none());
        assert_eq!(resp.request_id, "req-001");
    }

    #[test]
    fn test_handle_disallowed_model() {
        let handler = make_handler();
        let req = make_request("unknown-model");
        let resp = handler.handle(req);
        assert!(resp.error.is_some());
        let err = resp.error.unwrap();
        assert_eq!(err.code, "VALIDATION_ERROR");
        assert!(err.message.contains("unknown-model"));
    }

    #[test]
    fn test_handle_empty_request_id() {
        let handler = make_handler();
        let mut req = make_request("bert-base");
        req.request_id = String::new();
        let resp = handler.handle(req);
        assert!(resp.error.is_some());
    }

    #[test]
    fn test_handle_batch() {
        let handler = make_handler();
        let requests = vec![
            make_request("bert-base"),
            make_request("gpt2"),
            make_request("bert-base"),
        ];
        let responses = handler.handle_batch(requests);
        assert_eq!(responses.len(), 3);
        assert!(responses.iter().all(|r| r.error.is_none()));
    }

    #[test]
    fn test_handle_batch_capped_at_max() {
        let handler = WasmServingHandler::new(WasmServingConfig {
            max_batch_size: 2,
            allowed_models: vec!["bert-base".to_string()],
            ..Default::default()
        });
        let requests = (0..5).map(|_| make_request("bert-base")).collect::<Vec<_>>();
        let responses = handler.handle_batch(requests);
        assert_eq!(responses.len(), 2);
    }

    #[test]
    fn test_validate_sequence_too_long() {
        let handler = WasmServingHandler::new(WasmServingConfig {
            max_sequence_length: 512,
            allowed_models: vec!["bert-base".to_string()],
            ..Default::default()
        });
        let mut req = make_request("bert-base");
        req.parameters.max_length = Some(1024);
        let result = handler.validate_request(&req);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("1024"));
    }

    #[test]
    fn test_to_config_json() {
        let handler = make_handler();
        let json_str = handler.to_config_json().expect("serialization failed");
        let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("invalid JSON");
        assert_eq!(parsed["max_batch_size"], 8);
        assert_eq!(parsed["cache_responses"], true);
    }

    #[test]
    fn test_build_cors_headers_allowed_origin() {
        let handler = WasmServingHandler::new(WasmServingConfig {
            cors_origins: vec!["https://example.com".to_string()],
            allowed_models: vec!["bert-base".to_string()],
            ..Default::default()
        });
        let headers = handler.build_cors_headers("https://example.com");
        let origin_header = headers
            .iter()
            .find(|(k, _)| k == "Access-Control-Allow-Origin")
            .map(|(_, v)| v.as_str());
        assert_eq!(origin_header, Some("https://example.com"));
    }

    #[test]
    fn test_health_check() {
        let handler = make_handler();
        let hc = handler.health_check();
        assert_eq!(hc["status"], "ok");
    }

    #[test]
    fn test_edge_deployment_config_json() {
        let config = EdgeDeploymentConfig::new(
            EdgePlatform::CloudflareWorkers,
            WasmServingConfig::default(),
        );
        let json_str = config.to_json().expect("serialization failed");
        let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("invalid JSON");
        assert_eq!(parsed["platform"], "cloudflare_workers");
    }

    #[test]
    fn test_platform_specific_notes_nonempty() {
        for platform in [
            EdgePlatform::CloudflareWorkers,
            EdgePlatform::DenoDeployAPI,
            EdgePlatform::VercelEdge,
            EdgePlatform::AwsLambdaEdge,
            EdgePlatform::Generic,
        ] {
            let config = EdgeDeploymentConfig::new(platform, WasmServingConfig::default());
            assert!(
                !config.platform_specific_notes().is_empty(),
                "platform notes should be non-empty"
            );
        }
    }

    #[test]
    fn test_error_response_static() {
        let resp = WasmServingHandler::error_response("req-xyz", "NOT_FOUND", "model missing");
        assert_eq!(resp.request_id, "req-xyz");
        let err = resp.error.unwrap();
        assert_eq!(err.code, "NOT_FOUND");
        assert_eq!(err.message, "model missing");
    }
}

// ─── WasmError ────────────────────────────────────────────────────────────────

/// Errors specific to WASM model lifecycle and inference.
#[derive(Debug, thiserror::Error)]
pub enum WasmError {
    #[error("Model not loaded")]
    ModelNotLoaded,
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    #[error("Out of memory: requested {requested}, available {available}")]
    OutOfMemory { requested: usize, available: usize },
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
    #[error("Generation failed: {0}")]
    GenerationFailed(String),
}

// ─── WasmInferenceConfig ─────────────────────────────────────────────────────

/// Configuration controlling inference behaviour inside the WASM runtime.
#[derive(Debug, Clone)]
pub struct WasmInferenceConfig {
    /// Maximum number of new tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature; 1.0 = unmodified logits.
    pub temperature: f32,
    /// Nucleus sampling threshold.
    pub top_p: f32,
    /// Number of execution threads (typically 1 in WASM single-thread targets).
    pub num_threads: usize,
}

impl Default for WasmInferenceConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            num_threads: 1,
        }
    }
}

// ─── WasmMemoryManager ───────────────────────────────────────────────────────

/// Tracks heap allocations within a fixed-capacity WASM linear memory region.
///
/// This is a simple bump-pointer allocator that additionally tracks individual
/// freed regions so that `total_allocated` stays accurate.  It does **not**
/// re-use freed space — it is designed for the pattern of loading once and
/// unloading entirely, not for general-purpose dynamic allocation.
pub struct WasmMemoryManager {
    capacity: usize,
    /// Map of offset → size for every live allocation.
    allocations: std::collections::HashMap<usize, usize>,
    next_offset: usize,
}

impl WasmMemoryManager {
    /// Create a new memory manager with the given capacity in bytes.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            allocations: std::collections::HashMap::new(),
            next_offset: 0,
        }
    }

    /// Allocate `size` bytes and return the starting offset.
    ///
    /// Returns [`WasmError::OutOfMemory`] if the request cannot be satisfied.
    pub fn allocate(&mut self, size: usize) -> Result<usize, WasmError> {
        let already_allocated = self.total_allocated();
        let available = self.capacity.saturating_sub(already_allocated);
        if size > available {
            return Err(WasmError::OutOfMemory {
                requested: size,
                available,
            });
        }
        let offset = self.next_offset;
        self.allocations.insert(offset, size);
        self.next_offset = self.next_offset.saturating_add(size);
        Ok(offset)
    }

    /// Free the allocation that starts at `offset` with `size` bytes.
    ///
    /// No-ops if the offset is not tracked (safe to call with stale handles).
    pub fn free(&mut self, offset: usize, _size: usize) {
        self.allocations.remove(&offset);
    }

    /// Return the total number of currently-allocated bytes.
    pub fn total_allocated(&self) -> usize {
        self.allocations.values().sum()
    }

    /// Return the total capacity of this memory region.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ─── WasmTokenizer ───────────────────────────────────────────────────────────

/// Minimal byte-level BPE-stub tokenizer suitable for WASM environments.
///
/// Each byte is mapped to a token ID via `byte % vocab_size`, making this a
/// deterministic bijection for ASCII content when `vocab_size >= 128`.
pub struct WasmTokenizer {
    vocab_size: u32,
}

impl WasmTokenizer {
    /// Create a new tokenizer with the given vocabulary size.
    pub fn new(vocab_size: u32) -> Self {
        Self { vocab_size }
    }

    /// Encode a UTF-8 string into token IDs.
    ///
    /// Each byte of the UTF-8 representation is mapped to `byte % vocab_size`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.as_bytes().iter().map(|&b| (b as u32) % self.vocab_size).collect()
    }

    /// Decode token IDs back to a (potentially lossy) UTF-8 string.
    ///
    /// Each token ID is mapped to `(id % 256) as u8`, then the byte buffer is
    /// interpreted as lossy UTF-8.
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().map(|&id| (id % 256) as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

// ─── WasmModelHandle ─────────────────────────────────────────────────────────

/// Manages the lifecycle of a model loaded into WASM linear memory.
pub struct WasmModelHandle {
    config: WasmInferenceConfig,
    /// Length in bytes of the original model data (stored instead of bytes to
    /// avoid holding a large allocation in the stub implementation).
    model_bytes_len: usize,
    loaded: bool,
    memory_manager: WasmMemoryManager,
}

impl WasmModelHandle {
    /// Load a model from a byte slice and apply the given inference config.
    ///
    /// Returns [`WasmError::InvalidInput`] if `bytes` is empty.
    pub fn load_model_from_bytes(
        bytes: &[u8],
        config: WasmInferenceConfig,
    ) -> Result<Self, WasmError> {
        if bytes.is_empty() {
            return Err(WasmError::InvalidInput("empty model bytes".to_string()));
        }
        // Reserve 4× model size as the logical WASM heap for activations.
        let memory_manager = WasmMemoryManager::new(bytes.len().saturating_mul(4));
        Ok(Self {
            config,
            model_bytes_len: bytes.len(),
            loaded: true,
            memory_manager,
        })
    }

    /// Run a forward pass and return raw logits as `f32` values.
    ///
    /// Returns [`WasmError::ModelNotLoaded`] if the handle has been unloaded.
    /// Returns [`WasmError::InvalidInput`] if `input` is empty.
    pub fn run_inference(&self, input: &[u32]) -> Result<Vec<f32>, WasmError> {
        if !self.loaded {
            return Err(WasmError::ModelNotLoaded);
        }
        if input.is_empty() {
            return Err(WasmError::InvalidInput("empty input".to_string()));
        }
        // Stub: normalise token IDs by `max_tokens` to produce pseudo-logits.
        let max_tokens_f = self.config.max_tokens as f32;
        Ok(input.iter().map(|&x| (x as f32) / max_tokens_f).collect())
    }

    /// Run autoregressive token generation.
    ///
    /// Returns the prompt IDs concatenated with up to `max_new_tokens` stub
    /// generated token IDs.
    pub fn run_generation(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, WasmError> {
        if !self.loaded {
            return Err(WasmError::ModelNotLoaded);
        }
        let mut output = prompt_ids.to_vec();
        for i in 0..max_new_tokens {
            output.push((i as u32 % 100) + 1);
        }
        Ok(output)
    }

    /// Unload the model, releasing the loaded state.
    ///
    /// After calling this method [`Self::run_inference`] and [`Self::run_generation`] will
    /// return [`WasmError::ModelNotLoaded`].
    pub fn unload(&mut self) {
        self.loaded = false;
    }

    /// Return whether the model is currently loaded.
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Return the size in bytes of the model that was loaded.
    pub fn model_bytes_len(&self) -> usize {
        self.model_bytes_len
    }

    /// Provide read access to the memory manager for diagnostics.
    pub fn memory_manager(&self) -> &WasmMemoryManager {
        &self.memory_manager
    }
}

// ─── WasmModelHandle + WasmMemoryManager + WasmTokenizer tests ───────────────

#[cfg(test)]
mod wasm_extended_tests {
    use super::*;

    // 1. WasmInferenceConfig::default() field values
    #[test]
    fn test_wasm_inference_config_defaults() {
        let cfg = WasmInferenceConfig::default();
        assert_eq!(cfg.max_tokens, 256);
        assert!((cfg.temperature - 1.0f32).abs() < f32::EPSILON);
        assert!((cfg.top_p - 1.0f32).abs() < f32::EPSILON);
        assert_eq!(cfg.num_threads, 1);
    }

    // 2. WasmMemoryManager allocate then total_allocated
    #[test]
    fn test_memory_manager_allocate_tracks_total() {
        let mut mm = WasmMemoryManager::new(1024);
        let off = mm.allocate(256).expect("allocate 256");
        assert_eq!(off, 0);
        assert_eq!(mm.total_allocated(), 256);
        let off2 = mm.allocate(128).expect("allocate 128");
        assert_eq!(off2, 256);
        assert_eq!(mm.total_allocated(), 384);
    }

    // 3. WasmMemoryManager allocate beyond capacity returns OutOfMemory
    #[test]
    fn test_memory_manager_out_of_memory() {
        let mut mm = WasmMemoryManager::new(64);
        let err = mm.allocate(128).unwrap_err();
        assert!(matches!(err, WasmError::OutOfMemory { .. }));
    }

    // 4. WasmMemoryManager free reduces total_allocated
    #[test]
    fn test_memory_manager_free_reduces_total() {
        let mut mm = WasmMemoryManager::new(512);
        let off = mm.allocate(200).expect("allocate");
        assert_eq!(mm.total_allocated(), 200);
        mm.free(off, 200);
        assert_eq!(mm.total_allocated(), 0);
    }

    // 5. WasmMemoryManager capacity is reported correctly
    #[test]
    fn test_memory_manager_capacity() {
        let mm = WasmMemoryManager::new(4096);
        assert_eq!(mm.capacity(), 4096);
    }

    // 6. WasmTokenizer encode/decode roundtrip for ASCII
    #[test]
    fn test_tokenizer_encode_decode_ascii_roundtrip() {
        let tok = WasmTokenizer::new(256);
        let text = "Hello, world!";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    // 7. WasmTokenizer encode empty string returns empty
    #[test]
    fn test_tokenizer_encode_empty_string() {
        let tok = WasmTokenizer::new(256);
        let ids = tok.encode("");
        assert!(ids.is_empty());
    }

    // 8. WasmModelHandle::load_model_from_bytes with valid bytes
    #[test]
    fn test_model_handle_load_valid() {
        let bytes = vec![1u8; 512];
        let handle = WasmModelHandle::load_model_from_bytes(&bytes, WasmInferenceConfig::default())
            .expect("load should succeed");
        assert!(handle.is_loaded());
        assert_eq!(handle.model_bytes_len(), 512);
        // Memory capacity should be 4× model bytes.
        assert_eq!(handle.memory_manager().capacity(), 2048);
    }

    // 9. WasmModelHandle::load_model_from_bytes with empty bytes returns error
    #[test]
    fn test_model_handle_load_empty_bytes_error() {
        let result = WasmModelHandle::load_model_from_bytes(&[], WasmInferenceConfig::default());
        match result {
            Err(WasmError::InvalidInput(_)) => {},
            Err(other) => panic!("expected InvalidInput, got {:?}", other),
            Ok(_) => panic!("expected Err, got Ok"),
        }
    }

    // 10. WasmModelHandle::run_inference happy path
    #[test]
    fn test_model_handle_run_inference_happy() {
        let bytes = vec![0u8; 256];
        let handle = WasmModelHandle::load_model_from_bytes(&bytes, WasmInferenceConfig::default())
            .expect("load");
        let logits = handle.run_inference(&[10, 20, 30]).expect("inference");
        assert_eq!(logits.len(), 3);
        // 10 / 256 ≈ 0.039
        assert!((logits[0] - 10.0f32 / 256.0).abs() < 1e-5);
    }

    // 11. WasmModelHandle::run_inference when not loaded returns ModelNotLoaded
    #[test]
    fn test_model_handle_inference_not_loaded() {
        let bytes = vec![1u8; 128];
        let mut handle =
            WasmModelHandle::load_model_from_bytes(&bytes, WasmInferenceConfig::default())
                .expect("load");
        handle.unload();
        let err = handle.run_inference(&[1, 2, 3]).unwrap_err();
        assert!(matches!(err, WasmError::ModelNotLoaded));
    }

    // 12. WasmModelHandle::run_generation returns prompt + generated tokens
    #[test]
    fn test_model_handle_run_generation() {
        let bytes = vec![0u8; 256];
        let handle = WasmModelHandle::load_model_from_bytes(&bytes, WasmInferenceConfig::default())
            .expect("load");
        let prompt = &[5u32, 10u32, 15u32];
        let output = handle.run_generation(prompt, 4).expect("generation");
        // Output should start with prompt ids.
        assert_eq!(&output[..3], prompt);
        // Then have 4 generated tokens.
        assert_eq!(output.len(), 7);
    }

    // 13. WasmModelHandle::unload then run_inference returns error
    #[test]
    fn test_model_handle_unload_then_inference_fails() {
        let bytes = vec![9u8; 64];
        let mut handle =
            WasmModelHandle::load_model_from_bytes(&bytes, WasmInferenceConfig::default())
                .expect("load");
        assert!(handle.is_loaded());
        handle.unload();
        assert!(!handle.is_loaded());
        let err = handle.run_inference(&[42]).unwrap_err();
        assert!(matches!(err, WasmError::ModelNotLoaded));
    }

    // 14. WasmModelHandle::run_inference empty input returns error
    #[test]
    fn test_model_handle_inference_empty_input() {
        let bytes = vec![1u8; 64];
        let handle = WasmModelHandle::load_model_from_bytes(&bytes, WasmInferenceConfig::default())
            .expect("load");
        let err = handle.run_inference(&[]).unwrap_err();
        assert!(matches!(err, WasmError::InvalidInput(_)));
    }
}
