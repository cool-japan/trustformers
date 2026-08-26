//! Batch-oriented tokenization sized and shaped for GPU-style workloads.
//!
//! # Honesty note on the "GPU" naming
//!
//! This crate is pure Rust with no unsafe FFI bindings to any GPU driver
//! (CUDA, ROCm, oneAPI, OpenCL, Vulkan), consistent with the workspace's
//! Pure-Rust policy. There is therefore **no compiled-in path that executes
//! tokenization on an actual GPU device** anywhere in this module, no matter
//! what hardware is detected on the machine. [`GpuTokenizer::tokenize_batch`]
//! always dispatches to the real [`Tokenizer`] this type wraps, either
//! sequentially or in parallel across CPU cores (via
//! `scirs2_core::parallel_ops`) depending on [`GpuTokenizerConfig::enable_gpu`].
//!
//! The type and field names keep the "GPU" vocabulary because the public API
//! shape (batching, padding, per-batch statistics) mirrors what a real GPU
//! tokenization backend would expose, and because callers may still want to
//! opt into a hard failure when no real GPU backend is available rather than
//! a silent, slower CPU fallback -- see
//! [`GpuTokenizerConfig::require_real_gpu`] and
//! [`GpuTokenizerError::BackendUnavailable`].
//!
//! An earlier version of this module simulated GPU kernel dispatch (fake
//! memory pointers, fake kernel function pointers, fake per-vendor "compute
//! capability" numbers) and its batch tokenization path returned the
//! hardcoded sequence `[1, 2, .., 10]` for every input text regardless of
//! content. That entire fake pipeline has been removed; every token ID this
//! module now returns comes from a real call into the wrapped [`Tokenizer`].

use scirs2_core::parallel_ops::*; // SciRS2 Integration Policy - replaces rayon
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// GPU-workload-shaped tokenization backend. See the module-level docs for
/// what "GPU" does and does not mean here.
#[derive(Clone)]
pub struct GpuTokenizer {
    /// Underlying tokenizer implementation that performs every real encode.
    tokenizer: Arc<dyn Tokenizer>,
    /// Configuration
    config: GpuTokenizerConfig,
    /// Batch processing configuration
    batch_config: BatchProcessingConfig,
    /// Real, atomically-updated usage counters backing [`Self::get_stats`].
    stats: Arc<StatsCounters>,
}

impl std::fmt::Debug for GpuTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTokenizer")
            .field("config", &self.config)
            .field("batch_config", &self.batch_config)
            .finish()
    }
}

/// Real usage counters. Updated from [`GpuTokenizer::tokenize_batch`] and
/// read back by [`GpuTokenizer::get_stats`]; never hardcoded.
#[derive(Debug, Default)]
struct StatsCounters {
    total_tokens: AtomicU64,
    total_batches: AtomicU64,
    total_time_us: AtomicU64,
}

/// Best-effort hint about which native GPU driver/toolkit (if any) appears
/// to be installed on this machine, from [`GpuTokenizerConfig::detect_best_backend`].
/// Purely informational: see the module-level docs.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuBackend {
    /// NVIDIA CUDA
    Cuda,
    /// AMD ROCm/HIP
    Rocm,
    /// Intel oneAPI
    OneApi,
    /// OpenCL (generic)
    OpenCL,
    /// Vulkan compute
    Vulkan,
}

/// GPU tokenizer configuration
#[derive(Debug, Clone)]
pub struct GpuTokenizerConfig {
    /// When `true` (the default), [`GpuTokenizer::tokenize_batch`] splits the
    /// input into `batch_size`-sized chunks and encodes every chunk in
    /// parallel across CPU cores. When `false`, texts are encoded
    /// sequentially on the calling thread. This never changes *what* is
    /// computed (every text is always encoded through the real
    /// [`Tokenizer`]), only how the work is scheduled.
    pub enable_gpu: bool,
    /// Best-effort detected GPU driver/toolkit hint. Informational only --
    /// see the module-level docs.
    pub backend: GpuBackend,
    /// Reserved for a future real GPU backend. Unused by the current
    /// (CPU-only) tokenization path.
    pub device_id: u32,
    /// Number of texts grouped into one parallel work unit by
    /// [`GpuTokenizer::tokenize_batch`]. Every input text is still encoded
    /// regardless of this value -- it controls chunk granularity, not how
    /// many texts are processed.
    pub batch_size: usize,
    /// Token sequences longer than this are truncated (from the end) after
    /// encoding. `0` disables truncation.
    pub max_sequence_length: usize,
    /// If `true`, [`GpuTokenizer::with_config`] fails with
    /// [`GpuTokenizerError::BackendUnavailable`] instead of transparently
    /// falling back to CPU execution. This build never links a real GPU
    /// tokenization kernel backend, so setting this to `true` always fails,
    /// regardless of what [`Self::backend`] detects -- it exists for
    /// callers that must be able to detect "no real GPU execution
    /// available" as a hard error rather than a silent, slower fallback.
    pub require_real_gpu: bool,
}

impl Default for GpuTokenizerConfig {
    fn default() -> Self {
        Self {
            enable_gpu: true,
            backend: Self::detect_best_backend(),
            device_id: 0,
            batch_size: 32,
            max_sequence_length: 512,
            require_real_gpu: false,
        }
    }
}

impl GpuTokenizerConfig {
    /// Detect the best available GPU backend
    pub fn detect_best_backend() -> GpuBackend {
        // Check for CUDA first (most common for ML)
        if Self::is_cuda_available() {
            GpuBackend::Cuda
        } else if Self::is_rocm_available() {
            GpuBackend::Rocm
        } else if Self::is_oneapi_available() {
            GpuBackend::OneApi
        } else if Self::is_opencl_available() {
            GpuBackend::OpenCL
        } else {
            GpuBackend::Vulkan // Fallback label when nothing else was detected
        }
    }

    /// Check if CUDA is available
    pub fn is_cuda_available() -> bool {
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
            || std::path::Path::new("/usr/local/cuda").exists()
    }

    /// Check if ROCm is available
    pub fn is_rocm_available() -> bool {
        std::env::var("ROCM_PATH").is_ok() || std::path::Path::new("/opt/rocm").exists()
    }

    /// Check if Intel oneAPI is available
    pub fn is_oneapi_available() -> bool {
        std::env::var("ONEAPI_ROOT").is_ok() || std::path::Path::new("/opt/intel/oneapi").exists()
    }

    /// Check if OpenCL is available
    pub fn is_opencl_available() -> bool {
        std::path::Path::new("/usr/lib/libOpenCL.so").exists()
            || std::path::Path::new("/System/Library/Frameworks/OpenCL.framework").exists()
    }
}

/// Padding strategies for batch processing
#[derive(Debug, Clone)]
pub enum PaddingStrategy {
    /// Pad to longest sequence in batch
    Longest,
    /// Pad to fixed length
    Fixed(usize),
    /// Pad to next power of 2
    NextPowerOf2,
    /// No padding
    None,
}

/// Batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchProcessingConfig {
    /// Maximum batch size (mirrors [`GpuTokenizerConfig::batch_size`])
    pub max_batch_size: usize,
    /// Padding strategy actually applied by `GpuTokenizer::apply_padding`
    pub padding_strategy: PaddingStrategy,
}

/// GPU tokenization results
#[derive(Debug, Clone)]
pub struct GpuTokenizationResult {
    /// Token IDs, one real-encoded sequence per input text (in input order).
    pub token_ids: Vec<Vec<u32>>,
    /// Attention masks
    pub attention_masks: Option<Vec<Vec<u8>>>,
    /// Token type IDs
    pub token_type_ids: Option<Vec<Vec<u32>>>,
    /// Wall-clock processing time in microseconds
    pub processing_time_us: u64,
    /// Estimated output buffer memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Number of input texts processed (always `texts.len()`; no text is
    /// ever silently dropped)
    pub batch_size: usize,
}

/// GPU tokenization statistics. `total_tokens`, `total_batches` and the
/// timing fields are real, atomically-accumulated counters. `memory_utilization`
/// and `gpu_utilization` are always `0.0`: this build has no GPU memory pool
/// or on-device execution to measure, so there is no real quantity to report
/// for them (an honest "not applicable", not a placeholder for a real
/// number).
#[derive(Debug, Clone)]
pub struct GpuTokenizationStats {
    /// Total tokens produced across every [`GpuTokenizer::tokenize_batch`] call
    pub total_tokens: u64,
    /// Total batches processed
    pub total_batches: u64,
    /// Average processing time per token, in microseconds
    pub avg_time_per_token_us: f64,
    /// Average processing time per batch, in microseconds
    pub avg_time_per_batch_us: f64,
    /// Always `0.0` in this build -- see struct docs.
    pub memory_utilization: f32,
    /// Always `0.0` in this build -- see struct docs.
    pub gpu_utilization: f32,
    /// Throughput in tokens per second, derived from real accumulated timing
    pub throughput_tokens_per_sec: f64,
}

impl GpuTokenizer {
    /// Create a new tokenizer with the default configuration.
    pub fn new(tokenizer: Arc<dyn Tokenizer>) -> Result<Self, GpuTokenizerError> {
        Self::with_config(tokenizer, GpuTokenizerConfig::default())
    }

    /// Create a tokenizer with an explicit configuration.
    ///
    /// Fails with [`GpuTokenizerError::BackendUnavailable`] only when
    /// `config.require_real_gpu` is `true` -- see that field's docs. Every
    /// other configuration always succeeds and dispatches through the real
    /// wrapped [`Tokenizer`].
    pub fn with_config(
        tokenizer: Arc<dyn Tokenizer>,
        config: GpuTokenizerConfig,
    ) -> Result<Self, GpuTokenizerError> {
        if config.require_real_gpu {
            return Err(GpuTokenizerError::BackendUnavailable(format!(
                "no real GPU tokenization kernel backend is compiled into this build \
                 (detected driver hint: {:?}); this crate is pure Rust with no unsafe \
                 FFI GPU driver bindings, so on-device execution is never available here \
                 regardless of installed hardware -- set `require_real_gpu = false` to use \
                 the real parallel-CPU implementation instead",
                config.backend
            )));
        }

        let batch_config = BatchProcessingConfig {
            max_batch_size: config.batch_size,
            padding_strategy: PaddingStrategy::Longest,
        };

        Ok(Self {
            tokenizer,
            config,
            batch_config,
            stats: Arc::new(StatsCounters::default()),
        })
    }

    /// Encode one text through the real wrapped tokenizer and apply
    /// `max_sequence_length` truncation.
    fn encode_and_truncate(&self, text: &str) -> Result<TokenizedInput, GpuTokenizerError> {
        let mut result = self
            .tokenizer
            .encode(text)
            .map_err(|e| GpuTokenizerError::TokenizationError(e.to_string()))?;

        let max_len = self.config.max_sequence_length;
        if max_len > 0 && result.input_ids.len() > max_len {
            result.input_ids.truncate(max_len);
            result.attention_mask.truncate(max_len);
            if let Some(type_ids) = result.token_type_ids.as_mut() {
                type_ids.truncate(max_len);
            }
            if let Some(mask) = result.special_tokens_mask.as_mut() {
                mask.truncate(max_len);
            }
        }

        Ok(result)
    }

    /// Tokenize a batch of texts.
    ///
    /// Every text in `texts` is encoded through the real wrapped
    /// [`Tokenizer`] -- none are ever dropped, and no token ID is ever
    /// invented. See the module-level docs for what "GPU" means here.
    pub fn tokenize_batch(
        &self,
        texts: &[String],
    ) -> Result<GpuTokenizationResult, GpuTokenizerError> {
        let start_time = std::time::Instant::now();

        let encoded: Vec<TokenizedInput> = if self.config.enable_gpu {
            let chunk_size = self.config.batch_size.max(1);
            texts
                .par_chunks(chunk_size)
                .map(|chunk| {
                    chunk
                        .iter()
                        .map(|text| self.encode_and_truncate(text))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<Vec<_>>, GpuTokenizerError>>()?
                .into_iter()
                .flatten()
                .collect()
        } else {
            texts
                .iter()
                .map(|text| self.encode_and_truncate(text))
                .collect::<Result<Vec<_>, _>>()?
        };

        let mut token_ids = Vec::with_capacity(encoded.len());
        let mut attention_masks = Vec::with_capacity(encoded.len());
        for result in encoded {
            token_ids.push(result.input_ids);
            attention_masks.push(result.attention_mask);
        }

        // Apply padding if needed
        if matches!(self.batch_config.padding_strategy, PaddingStrategy::Longest) {
            self.apply_padding(&mut token_ids, &mut attention_masks)?;
        }

        let processing_time = start_time.elapsed().as_micros() as u64;
        let memory_usage = self.estimate_memory_usage(&token_ids);
        let batch_size = texts.len();

        let total_tokens: u64 = token_ids.iter().map(|ids| ids.len() as u64).sum();
        self.stats.total_tokens.fetch_add(total_tokens, Ordering::Relaxed);
        self.stats.total_batches.fetch_add(1, Ordering::Relaxed);
        self.stats.total_time_us.fetch_add(processing_time, Ordering::Relaxed);

        Ok(GpuTokenizationResult {
            token_ids,
            attention_masks: Some(attention_masks),
            token_type_ids: None,
            processing_time_us: processing_time,
            memory_usage_bytes: memory_usage,
            batch_size,
        })
    }

    /// Apply padding to token sequences
    fn apply_padding(
        &self,
        token_ids: &mut [Vec<u32>],
        attention_masks: &mut [Vec<u8>],
    ) -> Result<(), GpuTokenizerError> {
        if token_ids.is_empty() {
            return Ok(());
        }

        let max_length = token_ids.iter().map(|seq| seq.len()).max().unwrap_or(0);

        for (tokens, mask) in token_ids.iter_mut().zip(attention_masks.iter_mut()) {
            let current_length = tokens.len();
            if current_length < max_length {
                tokens.resize(max_length, 0); // Pad with 0
                mask.resize(max_length, 0); // Pad attention mask with 0
            }
        }

        Ok(())
    }

    /// Estimate memory usage of the produced token-ID buffers (4 bytes per
    /// `u32` token).
    fn estimate_memory_usage(&self, token_ids: &[Vec<u32>]) -> usize {
        token_ids.iter().map(|seq| seq.len() * 4).sum()
    }

    /// Get real, accumulated tokenization statistics. Zero until the first
    /// [`Self::tokenize_batch`] call.
    pub fn get_stats(&self) -> GpuTokenizationStats {
        let total_tokens = self.stats.total_tokens.load(Ordering::Relaxed);
        let total_batches = self.stats.total_batches.load(Ordering::Relaxed);
        let total_time_us = self.stats.total_time_us.load(Ordering::Relaxed);

        let avg_time_per_batch_us = if total_batches > 0 {
            total_time_us as f64 / total_batches as f64
        } else {
            0.0
        };
        let avg_time_per_token_us =
            if total_tokens > 0 { total_time_us as f64 / total_tokens as f64 } else { 0.0 };
        let throughput_tokens_per_sec = if total_time_us > 0 {
            total_tokens as f64 / (total_time_us as f64 / 1_000_000.0)
        } else {
            0.0
        };

        GpuTokenizationStats {
            total_tokens,
            total_batches,
            avg_time_per_token_us,
            avg_time_per_batch_us,
            memory_utilization: 0.0,
            gpu_utilization: 0.0,
            throughput_tokens_per_sec,
        }
    }

    /// Enable/disable parallel (vs sequential) CPU dispatch. See
    /// [`GpuTokenizerConfig::enable_gpu`].
    pub fn set_gpu_enabled(&mut self, enabled: bool) {
        self.config.enable_gpu = enabled;
    }

    /// Set the chunk size used to parallelize `tokenize_batch`.
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.config.batch_size = batch_size;
        self.batch_config.max_batch_size = batch_size;
    }

    /// Set device ID (reserved; see [`GpuTokenizerConfig::device_id`]).
    pub fn set_device_id(&mut self, device_id: u32) {
        self.config.device_id = device_id;
    }
}

/// GPU tokenizer errors
#[derive(Debug, Clone)]
pub enum GpuTokenizerError {
    /// GPU initialization error
    GpuInitializationError(String),
    /// Memory allocation error
    MemoryAllocationError(String),
    /// Kernel launch error
    KernelLaunchError(String),
    /// Kernel not found
    KernelNotFound(String),
    /// Tokenization error, wrapping the real error from the underlying
    /// [`Tokenizer`]
    TokenizationError(String),
    /// Configuration error
    ConfigurationError(String),
    /// CUDA error
    CudaError(String),
    /// Invalid device
    InvalidDevice(u32),
    /// Out of memory
    OutOfMemory,
    /// Returned by [`GpuTokenizer::with_config`] when
    /// [`GpuTokenizerConfig::require_real_gpu`] is `true`. This build never
    /// has a real GPU tokenization backend available, so this error is
    /// always returned in that configuration.
    BackendUnavailable(String),
}

impl std::fmt::Display for GpuTokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuTokenizerError::GpuInitializationError(msg) => {
                write!(f, "GPU initialization error: {}", msg)
            },
            GpuTokenizerError::MemoryAllocationError(msg) => {
                write!(f, "Memory allocation error: {}", msg)
            },
            GpuTokenizerError::KernelLaunchError(msg) => {
                write!(f, "Kernel launch error: {}", msg)
            },
            GpuTokenizerError::KernelNotFound(name) => {
                write!(f, "Kernel not found: {}", name)
            },
            GpuTokenizerError::TokenizationError(msg) => {
                write!(f, "Tokenization error: {}", msg)
            },
            GpuTokenizerError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            },
            GpuTokenizerError::CudaError(msg) => {
                write!(f, "CUDA error: {}", msg)
            },
            GpuTokenizerError::InvalidDevice(id) => {
                write!(f, "Invalid device: {}", id)
            },
            GpuTokenizerError::OutOfMemory => {
                write!(f, "Out of memory")
            },
            GpuTokenizerError::BackendUnavailable(msg) => {
                write!(f, "GPU backend unavailable: {}", msg)
            },
        }
    }
}

impl std::error::Error for GpuTokenizerError {}

/// GPU tokenization benchmarks. Every measurement comes from a real
/// [`GpuTokenizer::tokenize_batch`] call's wall-clock timing.
pub struct GpuTokenizationBenchmark {
    /// Test configurations
    pub configs: Vec<GpuTokenizerConfig>,
    /// Test texts
    pub test_texts: Vec<String>,
    /// Results
    pub results: Vec<BenchmarkResult>,
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Configuration used
    pub config: GpuTokenizerConfig,
    /// Real wall-clock processing time
    pub processing_time_us: u64,
    /// Real throughput derived from `processing_time_us`
    pub throughput_tokens_per_sec: f64,
    /// Real estimated memory usage
    pub memory_usage_bytes: usize,
    /// Always `0.0`: no on-device GPU execution occurs in this build, so
    /// there is no real utilization figure to report.
    pub gpu_utilization: f32,
}

impl Default for GpuTokenizationBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuTokenizationBenchmark {
    /// Create new benchmark
    pub fn new() -> Self {
        Self {
            configs: vec![
                GpuTokenizerConfig::default(),
                GpuTokenizerConfig {
                    batch_size: 64,
                    ..Default::default()
                },
                GpuTokenizerConfig {
                    batch_size: 128,
                    ..Default::default()
                },
            ],
            test_texts: vec![
                "Hello world".to_string(),
                "This is a longer text for testing tokenization performance".to_string(),
                "The quick brown fox jumps over the lazy dog".repeat(10),
            ],
            results: Vec::new(),
        }
    }

    /// Run benchmark
    pub fn run(&mut self, tokenizer: Arc<dyn Tokenizer>) -> Result<(), GpuTokenizerError> {
        for config in &self.configs {
            let gpu_tokenizer = GpuTokenizer::with_config(tokenizer.clone(), config.clone())?;

            let start_time = std::time::Instant::now();
            let result = gpu_tokenizer.tokenize_batch(&self.test_texts)?;
            let processing_time = start_time.elapsed().as_micros() as u64;

            let total_tokens: usize = result.token_ids.iter().map(|seq| seq.len()).sum();
            let throughput = total_tokens as f64 / (processing_time as f64 / 1_000_000.0);

            self.results.push(BenchmarkResult {
                config: config.clone(),
                processing_time_us: processing_time,
                throughput_tokens_per_sec: throughput,
                memory_usage_bytes: result.memory_usage_bytes,
                gpu_utilization: 0.0,
            });
        }

        Ok(())
    }

    /// Get best configuration
    pub fn get_best_config(&self) -> Option<&GpuTokenizerConfig> {
        self.results
            .iter()
            .max_by(|a, b| {
                a.throughput_tokens_per_sec
                    .partial_cmp(&b.throughput_tokens_per_sec)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|result| &result.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::char::CharTokenizer;
    use std::sync::Arc;

    /// A deterministic character-level tokenizer: unlike a from-scratch BPE
    /// setup (whose GPT-2 byte-level merges need a much larger vocabulary to
    /// avoid every input collapsing to the same unknown-token fallback),
    /// each character below maps to a distinct, predictable ID, which is
    /// exactly what these regression tests need to distinguish "real
    /// tokenizer output" from "fabricated output".
    fn create_test_tokenizer() -> Arc<dyn Tokenizer> {
        let mut vocab = std::collections::HashMap::new();
        vocab.insert("h".to_string(), 0);
        vocab.insert("e".to_string(), 1);
        vocab.insert("l".to_string(), 2);
        vocab.insert("o".to_string(), 3);
        vocab.insert("w".to_string(), 4);
        vocab.insert("r".to_string(), 5);
        vocab.insert("d".to_string(), 6);
        vocab.insert(" ".to_string(), 7);
        vocab.insert("[UNK]".to_string(), 8);
        vocab.insert("[PAD]".to_string(), 9);

        Arc::new(CharTokenizer::new(vocab))
    }

    #[test]
    fn test_gpu_tokenizer_creation() {
        let tokenizer = create_test_tokenizer();
        let gpu_tokenizer = GpuTokenizer::new(tokenizer);
        assert!(gpu_tokenizer.is_ok());
    }

    #[test]
    fn test_gpu_tokenizer_config() {
        let config = GpuTokenizerConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.max_sequence_length, 512);
        assert!(config.enable_gpu);
        assert!(!config.require_real_gpu);
    }

    #[test]
    fn test_batch_tokenization() {
        let tokenizer = create_test_tokenizer();
        let gpu_tokenizer = GpuTokenizer::new(tokenizer).expect("Construction failed");
        let texts = vec!["Hello world".to_string(), "This is a test".to_string()];
        let result = gpu_tokenizer.tokenize_batch(&texts);
        assert!(result.is_ok());
    }

    /// Regression test for the fabricated `[1, 2, .., 10]` GPU tokenization
    /// path: real output must vary with real input.
    #[test]
    fn test_tokenize_batch_output_varies_with_input() {
        let tokenizer = create_test_tokenizer();
        let gpu_tokenizer = GpuTokenizer::new(tokenizer).expect("Construction failed");

        // Every character of "hello" and "world" is in the test vocab, so a
        // real tokenizer must map them to the differing sequences
        // [0,1,2,2,3] and [4,3,5,2,6] respectively (see
        // `create_test_tokenizer`), never the same sequence.
        let texts = vec!["hello".to_string(), "world".to_string()];
        let result = gpu_tokenizer.tokenize_batch(&texts).expect("tokenize_batch failed");

        // The old fake `copy_from_gpu` produced `[1, 2, .., 10]` (truncated
        // at the first non-positional zero) for *every* text, so both
        // outputs below would have been byte-for-byte identical apart from
        // padding. Real tokenization of different text must not collapse to
        // the same sequence.
        assert_ne!(
            result.token_ids[0], result.token_ids[1],
            "different input texts must not produce identical token ID sequences"
        );
        assert_ne!(result.token_ids[0], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    /// Regression test: `tokenize_batch` must delegate to the exact same
    /// real tokenizer `encode()` would produce, proving the GPU path is not
    /// an independent fake/hash-based implementation.
    #[test]
    fn test_tokenize_batch_matches_real_tokenizer() {
        let tokenizer = create_test_tokenizer();
        let reference = tokenizer.encode("hello world").expect("encode failed");

        let gpu_tokenizer = GpuTokenizer::new(tokenizer).expect("Construction failed");
        let texts = vec!["hello world".to_string()];
        let result = gpu_tokenizer.tokenize_batch(&texts).expect("tokenize_batch failed");

        assert_eq!(result.token_ids[0], reference.input_ids);
    }

    /// Regression test for "tokenize_batch silently discards every text
    /// beyond config.batch_size": every input text must appear in the
    /// output, however small `batch_size` (the parallel chunk size) is.
    #[test]
    fn test_tokenize_batch_processes_every_text_beyond_batch_size() {
        let tokenizer = create_test_tokenizer();
        let config = GpuTokenizerConfig {
            batch_size: 2,
            ..GpuTokenizerConfig::default()
        };
        let gpu_tokenizer =
            GpuTokenizer::with_config(tokenizer, config).expect("Construction failed");

        let texts: Vec<String> = (0..7).map(|i| format!("text number {}", i)).collect();
        let result = gpu_tokenizer.tokenize_batch(&texts).expect("tokenize_batch failed");

        assert_eq!(result.batch_size, 7);
        assert_eq!(result.token_ids.len(), 7);
    }

    #[test]
    fn test_max_sequence_length_truncates() {
        let tokenizer = create_test_tokenizer();
        let config = GpuTokenizerConfig {
            max_sequence_length: 3,
            ..GpuTokenizerConfig::default()
        };
        let gpu_tokenizer =
            GpuTokenizer::with_config(tokenizer, config).expect("Construction failed");

        let texts = vec!["hello world hello world hello world".to_string()];
        let result = gpu_tokenizer.tokenize_batch(&texts).expect("tokenize_batch failed");

        assert!(result.token_ids[0].len() <= 3);
    }

    #[test]
    fn test_padding_application() {
        let tokenizer = create_test_tokenizer();
        let gpu_tokenizer = GpuTokenizer::new(tokenizer).expect("Construction failed");
        let mut token_ids = vec![vec![1, 2, 3], vec![4, 5]];
        let mut attention_masks = vec![vec![1, 1, 1], vec![1, 1]];

        gpu_tokenizer
            .apply_padding(&mut token_ids, &mut attention_masks)
            .expect("Operation failed in test");

        assert_eq!(token_ids[0].len(), 3);
        assert_eq!(token_ids[1].len(), 3);
        assert_eq!(token_ids[1][2], 0); // Padding token
        assert_eq!(attention_masks[1][2], 0); // Padding mask
    }

    #[test]
    fn test_gpu_tokenization_stats_starts_at_zero() {
        let tokenizer = create_test_tokenizer();
        let gpu_tokenizer = GpuTokenizer::new(tokenizer).expect("Construction failed");
        let stats = gpu_tokenizer.get_stats();
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.total_batches, 0);
    }

    /// Regression test for `get_stats` returning hardcoded zeros
    /// unconditionally: after real tokenization work, the counters must
    /// reflect it.
    #[test]
    fn test_gpu_tokenization_stats_reflect_real_usage() {
        let tokenizer = create_test_tokenizer();
        let gpu_tokenizer = GpuTokenizer::new(tokenizer).expect("Construction failed");

        let texts = vec!["hello world".to_string(), "another text here".to_string()];
        gpu_tokenizer.tokenize_batch(&texts).expect("tokenize_batch failed");

        let stats = gpu_tokenizer.get_stats();
        assert_eq!(stats.total_batches, 1);
        assert!(
            stats.total_tokens > 0,
            "real tokenization must produce a nonzero token count"
        );

        gpu_tokenizer.tokenize_batch(&texts).expect("tokenize_batch failed");
        let stats_after_second_call = gpu_tokenizer.get_stats();
        assert_eq!(stats_after_second_call.total_batches, 2);
        assert_eq!(stats_after_second_call.total_tokens, stats.total_tokens * 2);
    }

    /// Regression test for `require_real_gpu`: this build never has a real
    /// GPU backend, so requiring one must fail loudly rather than silently
    /// falling back (which is the default, non-`require_real_gpu` behavior).
    #[test]
    fn test_require_real_gpu_returns_backend_unavailable() {
        let tokenizer = create_test_tokenizer();
        let config = GpuTokenizerConfig {
            require_real_gpu: true,
            ..GpuTokenizerConfig::default()
        };
        let result = GpuTokenizer::with_config(tokenizer, config);
        assert!(matches!(
            result,
            Err(GpuTokenizerError::BackendUnavailable(_))
        ));
    }

    #[test]
    fn test_gpu_tokenizer_configuration() {
        let tokenizer = create_test_tokenizer();
        let mut gpu_tokenizer = GpuTokenizer::new(tokenizer).expect("Construction failed");

        gpu_tokenizer.set_batch_size(64);
        gpu_tokenizer.set_device_id(1);
        gpu_tokenizer.set_gpu_enabled(false);

        assert_eq!(gpu_tokenizer.config.batch_size, 64);
        assert_eq!(gpu_tokenizer.config.device_id, 1);
        assert!(!gpu_tokenizer.config.enable_gpu);
    }

    /// `enable_gpu = false` (sequential) must produce identical results to
    /// `enable_gpu = true` (parallel chunks) -- scheduling must never change
    /// the real output.
    #[test]
    fn test_sequential_and_parallel_paths_agree() {
        let tokenizer = create_test_tokenizer();
        let texts: Vec<String> = (0..10).map(|i| format!("sample text {}", i)).collect();

        let parallel = GpuTokenizer::with_config(
            tokenizer.clone(),
            GpuTokenizerConfig {
                enable_gpu: true,
                batch_size: 3,
                ..GpuTokenizerConfig::default()
            },
        )
        .expect("Construction failed")
        .tokenize_batch(&texts)
        .expect("tokenize_batch failed");

        let sequential = GpuTokenizer::with_config(
            tokenizer,
            GpuTokenizerConfig {
                enable_gpu: false,
                ..GpuTokenizerConfig::default()
            },
        )
        .expect("Construction failed")
        .tokenize_batch(&texts)
        .expect("tokenize_batch failed");

        assert_eq!(parallel.token_ids, sequential.token_ids);
    }

    #[test]
    fn test_benchmark_creation() {
        let benchmark = GpuTokenizationBenchmark::new();
        assert_eq!(benchmark.configs.len(), 3);
        assert_eq!(benchmark.test_texts.len(), 3);
        assert_eq!(benchmark.results.len(), 0);
    }

    #[test]
    fn test_benchmark_run_produces_real_results() {
        let tokenizer = create_test_tokenizer();
        let mut benchmark = GpuTokenizationBenchmark::new();
        benchmark.run(tokenizer).expect("benchmark run failed");

        assert_eq!(benchmark.results.len(), benchmark.configs.len());
        for result in &benchmark.results {
            assert!(result.memory_usage_bytes > 0);
        }
        assert!(benchmark.get_best_config().is_some());
    }
}
