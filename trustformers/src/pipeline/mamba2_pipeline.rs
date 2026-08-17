//! Mamba-2 State Space Models (SSM) Pipeline - Cutting-Edge 2025 Architecture
//!
//! This module implements Mamba-2, the revolutionary state space model architecture for 2025:
//! - Ultra-long sequence processing (millions of tokens)
//! - Linear scaling with sequence length
//! - Superior performance vs Transformers on many tasks
//! - Hardware-efficient computation with selective state space
//! - Advanced selective scan mechanisms

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{Result, TrustformersError};
use crate::pipeline::{Pipeline, PipelineInput, PipelineOutput};
use trustformers_core::traits::Tokenizer;

/// Configuration for Mamba-2 State Space Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mamba2Config {
    /// Model dimension
    pub d_model: usize,
    /// State dimension for the SSM
    pub d_state: usize,
    /// Expansion factor for intermediate dimension
    pub expand_factor: usize,
    /// Dimension of the convolutional kernel
    pub d_conv: usize,
    /// Delta rank parameter for selective SSM
    pub dt_rank: Option<usize>,
    /// Number of SSM heads for multi-head selective scan
    pub n_heads: usize,
    /// Activation function for the SSM
    pub activation: Mamba2Activation,
    /// Whether to use bias in linear layers
    pub bias: bool,
    /// Whether to use simplified A initialization
    pub simplified_a_init: bool,
    /// Hardware optimization strategy
    pub hardware_strategy: HardwareStrategy,
    /// Sequence chunking strategy for ultra-long sequences
    pub chunking_strategy: ChunkingStrategy,
    /// Memory optimization level
    pub memory_optimization: MemoryOptimization,
}

impl Default for Mamba2Config {
    fn default() -> Self {
        Self {
            d_model: 768,
            d_state: 16,
            expand_factor: 2,
            d_conv: 4,
            dt_rank: None, // Auto-calculated as d_model / 16
            n_heads: 1,
            activation: Mamba2Activation::SiLU,
            bias: false,
            simplified_a_init: true,
            hardware_strategy: HardwareStrategy::Auto,
            chunking_strategy: ChunkingStrategy::Adaptive,
            memory_optimization: MemoryOptimization::Balanced,
        }
    }
}

/// Activation functions for Mamba-2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Mamba2Activation {
    SiLU,
    GELU,
    ReLU,
    Swish,
}

/// Hardware optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HardwareStrategy {
    /// Automatically detect and optimize for available hardware
    Auto,
    /// Optimize for CPU with SIMD
    CPU,
    /// Optimize for CUDA GPUs
    CUDA,
    /// Optimize for Apple Silicon with Metal
    Metal,
    /// Optimize for memory-constrained devices
    MemoryConstrained,
    /// Optimize for maximum throughput
    MaxThroughput,
}

/// Sequence chunking strategies for ultra-long sequences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// No chunking - process entire sequence
    None,
    /// Fixed-size chunks
    Fixed(usize),
    /// Adaptive chunking based on memory availability
    Adaptive,
    /// Overlapping chunks with state transfer
    Overlapping { chunk_size: usize, overlap: usize },
    /// Hierarchical chunking for very long sequences
    Hierarchical { levels: Vec<usize> },
}

/// Memory optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryOptimization {
    /// No optimization - fastest inference
    None,
    /// Balanced optimization
    Balanced,
    /// Aggressive optimization for limited memory
    Aggressive,
    /// Ultra optimization for extreme constraints
    Ultra,
}

/// Mamba-2 State Space Layer representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mamba2Layer {
    /// Layer index
    pub layer_id: usize,
    /// Current hidden state
    pub hidden_state: Vec<f32>,
    /// Convolutional state buffer
    pub conv_state: Vec<f32>,
    /// SSM state buffer
    pub ssm_state: Vec<f32>,
    /// Delta parameter for selective mechanism
    pub delta: Vec<f32>,
    /// A matrix for state evolution
    pub a_matrix: Vec<f32>,
    /// B matrix for input projection
    pub b_matrix: Vec<f32>,
    /// C matrix for output projection
    pub c_matrix: Vec<f32>,
    /// D skip connection parameter
    pub d_param: f32,
}

/// Number of Mamba-2 blocks the pipeline instantiates.
const MAMBA2_LAYER_COUNT: usize = 24;

/// Default parallel-scan chunk size when the strategy does not name one.
const DEFAULT_CHUNK_SIZE: usize = 256;

/// A deterministic, bounded parameter matrix.
///
/// Used for the embedding table and LM head so that no matrix is uniformly
/// zero. Values lie in `±scale` and depend only on their position, so runs are
/// reproducible.
fn deterministic_matrix(rows: usize, cols: usize, scale: f64) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    let phase = (r * 31 + c * 17) as f64 * 0.017_453_292_519_943_295;
                    phase.sin() * scale
                })
                .collect()
        })
        .collect()
}

/// Mamba-2 Model state and computation
///
/// The `blocks` are the models crate's real SSD blocks; `embeddings` and
/// `lm_head` are owned here because the models crate's `Mamba2ForCausalLM`
/// ships them zero-initialised.
pub struct Mamba2Model {
    config: Mamba2Config,
    models_config: trustformers_models::mamba2::Mamba2Config,
    blocks: Vec<trustformers_models::mamba2::Mamba2Block>,
    embeddings: Vec<Vec<f64>>,
    lm_head: Vec<Vec<f64>>,
    layers: Vec<Mamba2Layer>,
    performance_tracker: Arc<RwLock<Mamba2PerformanceTracker>>,
    state_manager: Arc<RwLock<StateManager>>,
}

impl std::fmt::Debug for Mamba2Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mamba2Model")
            .field("config", &self.config)
            .field("blocks", &self.blocks.len())
            .field("vocab_size", &self.embeddings.len())
            .finish()
    }
}

/// Performance tracking for Mamba-2
#[derive(Debug, Default)]
pub struct Mamba2PerformanceTracker {
    pub total_tokens_processed: u64,
    pub average_latency_per_token: f32,
    pub memory_usage_mb: f32,
    pub state_size_mb: f32,
    pub throughput_tokens_per_second: f32,
    pub hardware_utilization: f32,
    pub selective_scan_efficiency: f32,
}

/// State management for ultra-long sequences
#[derive(Debug, Default)]
pub struct StateManager {
    pub checkpoint_interval: usize,
    pub state_checkpoints: HashMap<usize, Vec<u8>>,
    pub compression_enabled: bool,
    pub max_state_memory_mb: f32,
}

/// Mamba-2 pipeline output with enhanced information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mamba2Output {
    /// Generated text or processed sequences
    pub text: String,
    /// Raw logits from the model
    pub logits: Vec<f32>,
    /// Final hidden states
    pub hidden_states: Vec<f32>,
    /// SSM states for continuation
    pub ssm_states: Vec<f32>,
    /// Performance metrics
    pub performance: Mamba2PerformanceMetrics,
    /// Token ids the model produced, when generation ran.
    pub token_ids: Vec<u32>,
    /// State evolution trajectory, when it was captured.
    ///
    /// `None` unless the caller asked for it — a Mamba block has no attention
    /// matrix to report, so nothing is synthesised in its place.
    pub state_trajectory: Option<Vec<Vec<f32>>>,
}

/// Performance metrics for Mamba-2 output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mamba2PerformanceMetrics {
    /// Wall-clock time of the run.
    pub inference_time_ms: f32,
    /// Tokens processed per second, measured.
    pub tokens_per_second: f32,
    /// Resident memory of the process, in megabytes.
    ///
    /// `None` when the operating system does not expose it.
    pub memory_usage_mb: Option<f32>,
    /// Recurrent state size relative to the KV cache an attention model of the
    /// same width would need for this sequence. Computed, not assumed.
    pub state_compression_ratio: f32,
}

/// Options that must be supplied explicitly when building a [`Mamba2Pipeline`].
#[derive(Clone)]
pub struct Mamba2PipelineOptions {
    /// Tokenizer used to encode prompts and decode generated ids.
    ///
    /// Required: without a real vocabulary there is no honest way to turn text
    /// into token ids.
    pub tokenizer: Arc<dyn Tokenizer>,
    /// Acknowledge that no Mamba-2 checkpoint is being loaded.
    ///
    /// There is no Mamba-2 weight loader in this workspace yet, so a pipeline
    /// built this way runs a *real* SSD scan over *deterministically
    /// initialised, untrained* parameters. Its outputs are therefore not
    /// predictions of anything. The flag exists so that can never happen by
    /// accident.
    pub allow_untrained_weights: bool,
}

impl std::fmt::Debug for Mamba2PipelineOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mamba2PipelineOptions")
            .field("vocab_size", &self.tokenizer.vocab_size())
            .field("allow_untrained_weights", &self.allow_untrained_weights)
            .finish()
    }
}

/// Main Mamba-2 Pipeline
///
/// The state-space computation is performed by the real SSD implementation in
/// `trustformers_models::mamba2` — the same `Mamba2Block` used by the models
/// crate's own Mamba-2 model — over a deterministically initialised embedding
/// table and language-model head owned by this pipeline.
pub struct Mamba2Pipeline {
    model: Arc<RwLock<Mamba2Model>>,
    config: Mamba2Config,
    tokenizer: Arc<dyn Tokenizer>,
    performance_monitor: Arc<RwLock<Mamba2PerformanceTracker>>,
}

impl Mamba2Pipeline {
    /// Build a pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::FeatureUnavailable`] unless
    /// [`Mamba2PipelineOptions::allow_untrained_weights`] is set, because no
    /// Mamba-2 checkpoint loader exists yet and silently returning an untrained
    /// network would make every output look like a prediction.
    pub fn new(config: Mamba2Config, options: Mamba2PipelineOptions) -> Result<Self> {
        if !options.allow_untrained_weights {
            return Err(TrustformersError::feature_unavailable(
                "no Mamba-2 checkpoint loader exists yet, so this pipeline can only run \
                 deterministically initialised, untrained parameters. Set \
                 `Mamba2PipelineOptions { allow_untrained_weights: true, .. }` to acknowledge \
                 that its outputs are not predictions."
                    .to_string(),
                "mamba2-weights",
            ));
        }
        tracing::warn!(
            d_model = config.d_model,
            "building a Mamba-2 pipeline over untrained parameters; outputs are not predictions"
        );

        let vocab_size = options.tokenizer.vocab_size();
        if vocab_size == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "the supplied tokenizer has an empty vocabulary".to_string(),
            ));
        }
        let model = Self::initialize_model(&config, vocab_size)?;
        let performance_monitor = Arc::new(RwLock::new(Mamba2PerformanceTracker::default()));

        Ok(Self {
            model: Arc::new(RwLock::new(model)),
            config,
            tokenizer: options.tokenizer,
            performance_monitor,
        })
    }

    /// Load a Mamba-2 checkpoint.
    ///
    /// # Errors
    ///
    /// Always fails today: no Mamba-2 weight loader has been implemented, and
    /// returning an untrained model from a `from_pretrained` call would be
    /// indistinguishable from a real load.
    pub fn from_pretrained(model_name_or_path: &str) -> Result<Self> {
        Err(TrustformersError::feature_unavailable(
            format!(
                "loading Mamba-2 checkpoints is not implemented, so `{model_name_or_path}` \
                 cannot be loaded. Use `Mamba2Pipeline::new` with \
                 `allow_untrained_weights` if an untrained network is what you want."
            ),
            "mamba2-checkpoint-loading",
        ))
    }

    /// Map the pipeline configuration onto the models crate's Mamba-2
    /// configuration, which drives the real SSD blocks.
    fn models_config(
        config: &Mamba2Config,
        vocab_size: usize,
    ) -> Result<trustformers_models::mamba2::Mamba2Config> {
        let expand = config.expand_factor.max(1);
        let inner_dim = config.d_model * expand;
        let nheads = config.n_heads.max(1);
        if !inner_dim.is_multiple_of(nheads) {
            return Err(TrustformersError::invalid_input_simple(format!(
                "inner dimension {inner_dim} (d_model {} × expand {expand}) is not divisible by \
                 n_heads {nheads}",
                config.d_model
            )));
        }
        Ok(trustformers_models::mamba2::Mamba2Config {
            vocab_size,
            d_model: config.d_model,
            n_layer: MAMBA2_LAYER_COUNT,
            d_state: config.d_state,
            d_conv: config.d_conv,
            expand,
            nheads,
            headdim: inner_dim / nheads,
            chunk_size: match config.chunking_strategy {
                ChunkingStrategy::Fixed(size) => size.max(1),
                ChunkingStrategy::Overlapping { chunk_size, .. } => chunk_size.max(1),
                _ => DEFAULT_CHUNK_SIZE,
            },
            rms_norm_eps: 1e-5,
            tie_embeddings: false,
        })
    }

    /// Initialize the Mamba-2 model with the given configuration.
    ///
    /// Every parameter gets a real, documented value: the SSD blocks come from
    /// the models crate, and the embedding table and LM head are seeded with a
    /// deterministic bounded pattern so that no matrix is uniformly zero (a zero
    /// `C` matrix would silently reduce the scan to the identity).
    fn initialize_model(config: &Mamba2Config, vocab_size: usize) -> Result<Mamba2Model> {
        let models_config = Self::models_config(config, vocab_size)?;
        let blocks: Vec<trustformers_models::mamba2::Mamba2Block> = (0..MAMBA2_LAYER_COUNT)
            .map(|_| trustformers_models::mamba2::Mamba2Block::new(&models_config))
            .collect();

        let embeddings = deterministic_matrix(vocab_size, config.d_model, 0.02);
        let lm_head = deterministic_matrix(vocab_size, config.d_model, 0.02);

        let dt_rank = config.dt_rank.unwrap_or(config.d_model / 16).max(1);
        let layer_states = (0..MAMBA2_LAYER_COUNT)
            .map(|layer_id| Mamba2Layer {
                layer_id,
                hidden_state: vec![0.0; config.d_model],
                conv_state: vec![0.0; config.d_conv * config.d_model],
                ssm_state: vec![0.0; config.d_state * config.d_model],
                delta: vec![1.0 / dt_rank as f32; dt_rank],
                a_matrix: Self::initialize_a_matrix(config.d_state, config.simplified_a_init),
                b_matrix: Self::initialize_projection(config.d_state, config.d_model, 1),
                c_matrix: Self::initialize_projection(config.d_state, config.d_model, 2),
                d_param: 1.0,
            })
            .collect();

        Ok(Mamba2Model {
            config: config.clone(),
            models_config,
            blocks,
            embeddings,
            lm_head,
            layers: layer_states,
            performance_tracker: Arc::new(RwLock::new(Mamba2PerformanceTracker::default())),
            state_manager: Arc::new(RwLock::new(StateManager::default())),
        })
    }

    /// Initialize A matrix for SSM with logarithmic spacing
    fn initialize_a_matrix(d_state: usize, simplified: bool) -> Vec<f32> {
        let mut a_matrix = Vec::with_capacity(d_state);

        if simplified {
            // Simplified initialization with exponentially spaced eigenvalues
            for i in 0..d_state {
                let val = -((i + 1) as f32).ln();
                a_matrix.push(val);
            }
        } else {
            // Complex initialization with random components
            for i in 0..d_state {
                let real_part = -((i + 1) as f32 / d_state as f32).ln();
                let imag_part = 2.0 * std::f32::consts::PI * (i as f32 / d_state as f32);
                a_matrix.push(real_part * imag_part.cos());
            }
        }

        a_matrix
    }

    /// Deterministic, non-degenerate initialization for the B and C projections.
    ///
    /// A zero matrix here is not a neutral choice: a zero `C` makes the scan
    /// output `D · x`, i.e. the input, which is why the previous all-zero
    /// initialization made the pipeline look like it worked while doing
    /// nothing. Values are bounded in `±1/sqrt(d_state)`.
    fn initialize_projection(d_state: usize, d_model: usize, seed: usize) -> Vec<f32> {
        let scale = 1.0 / (d_state.max(1) as f32).sqrt();
        (0..d_state * d_model)
            .map(|i| {
                let phase = (i * seed + seed) as f32 * 0.017_453_292;
                phase.sin() * scale
            })
            .collect()
    }

    /// Run the real SSD scan over `hidden`.
    ///
    /// Delegates to `trustformers_models::mamba2::Mamba2Block`, the same
    /// implementation the models crate uses, so this is the project's single
    /// selective-scan implementation rather than a second approximate copy.
    async fn selective_scan_sequence(&self, hidden: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>> {
        let model = self.model.read().await;
        let mut current = hidden;
        for block in &model.blocks {
            current = block.forward(&current).map_err(|e| {
                TrustformersError::pipeline(format!("Mamba-2 SSD scan failed: {e}"), "mamba2")
            })?;
        }
        Ok(current)
    }

    /// Apply the configured chunking strategy and run the scan.
    ///
    /// Overlapping and fixed chunking really split the sequence; the recorded
    /// chunk boundaries are the ones that were used.
    async fn process_with_chunking(
        &self,
        hidden: Vec<Vec<f64>>,
        strategy: &ChunkingStrategy,
    ) -> Result<Vec<Vec<f64>>> {
        let chunk_size = match strategy {
            ChunkingStrategy::None | ChunkingStrategy::Adaptive => None,
            ChunkingStrategy::Fixed(size) => Some((*size).max(1)),
            ChunkingStrategy::Overlapping { chunk_size, .. } => Some((*chunk_size).max(1)),
            ChunkingStrategy::Hierarchical { levels } => levels.first().map(|l| (*l).max(1)),
        };

        match chunk_size {
            None => self.selective_scan_sequence(hidden).await,
            Some(size) if hidden.len() <= size => self.selective_scan_sequence(hidden).await,
            Some(size) => {
                let mut output = Vec::with_capacity(hidden.len());
                for chunk in hidden.chunks(size) {
                    output.extend(self.selective_scan_sequence(chunk.to_vec()).await?);
                }
                Ok(output)
            },
        }
    }

    /// Measured performance metrics for one run.
    ///
    /// Everything here is derived from the run itself; the previous constants
    /// (`0.8` compression, `0.92` efficiency, `0.95` utilisation) are gone
    /// because nothing measured them.
    async fn compute_performance_metrics(
        &self,
        start_time: std::time::Instant,
        input_tokens: usize,
    ) -> Mamba2PerformanceMetrics {
        let elapsed = start_time.elapsed();
        let inference_time_ms = elapsed.as_secs_f32() * 1000.0;
        let tokens_per_second = if elapsed.as_secs_f32() > 0.0 {
            input_tokens as f32 / elapsed.as_secs_f32()
        } else {
            0.0
        };
        let memory_usage_mb = crate::profiler::read_process_memory()
            .map(|m| m.resident_bytes as f32 / (1024.0 * 1024.0));

        // A Mamba layer keeps `d_state × headdim` state per head regardless of
        // sequence length; the ratio against a length-proportional KV cache is
        // therefore a real, computed property of this run.
        let state_floats = self.config.d_state * self.config.d_model;
        let attention_cache_floats = input_tokens.max(1) * self.config.d_model * 2;
        let state_compression_ratio = state_floats as f32 / attention_cache_floats as f32;

        Mamba2PerformanceMetrics {
            inference_time_ms,
            tokens_per_second,
            memory_usage_mb,
            state_compression_ratio,
        }
    }
}

impl Pipeline for Mamba2Pipeline {
    type Input = PipelineInput;
    type Output = Mamba2Output;

    /// Run the pipeline from synchronous code.
    ///
    /// # Errors
    ///
    /// Building a nested runtime is impossible from inside a current-thread
    /// runtime, so that case returns an error pointing at
    /// [`Mamba2Pipeline::process_async`] instead of panicking, which is what
    /// the previous `Runtime::new()`-per-call did.
    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => match handle.runtime_flavor() {
                tokio::runtime::RuntimeFlavor::MultiThread => {
                    tokio::task::block_in_place(|| handle.block_on(self.process_async(input)))
                },
                _ => Err(TrustformersError::runtime_error(
                    "Mamba2Pipeline::__call__ was invoked from inside a current-thread Tokio \
                     runtime, where blocking is not possible. Await \
                     `Mamba2Pipeline::process_async` instead."
                        .to_string(),
                )),
            },
            Err(_) => {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| {
                        TrustformersError::runtime_error(format!(
                            "Failed to create tokio runtime: {e}"
                        ))
                    })?;
                runtime.block_on(self.process_async(input))
            },
        }
    }
}

impl Mamba2Pipeline {
    /// Encode `input` into token ids using the pipeline's real tokenizer.
    fn encode_input(&self, input: PipelineInput) -> Result<Vec<u32>> {
        match input {
            PipelineInput::Text(text) => {
                let encoded = self.tokenizer.encode(&text)?;
                if encoded.input_ids.is_empty() {
                    return Err(TrustformersError::invalid_input_simple(
                        "input text encoded to an empty token sequence".to_string(),
                    ));
                }
                Ok(encoded.input_ids)
            },
            PipelineInput::Tokens(tokens) => {
                if tokens.is_empty() {
                    return Err(TrustformersError::invalid_input_simple(
                        "token input was empty".to_string(),
                    ));
                }
                Ok(tokens)
            },
            _ => Err(TrustformersError::invalid_input(
                "Unsupported input type for Mamba-2".to_string(),
                None::<String>,
                None::<String>,
                None::<String>,
            )),
        }
    }

    /// Run the model over `input` and return its real outputs.
    ///
    /// # Errors
    ///
    /// Fails on unsupported input, on an empty encoding, or when the SSD scan
    /// rejects the sequence shape.
    pub async fn process_async(&self, input: PipelineInput) -> Result<Mamba2Output> {
        let start_time = std::time::Instant::now();
        let token_ids = self.encode_input(input)?;

        // Embed with the pipeline's own (non-degenerate) embedding table.
        let hidden: Vec<Vec<f64>> = {
            let model = self.model.read().await;
            let vocab_size = model.embeddings.len();
            token_ids
                .iter()
                .map(|&id| {
                    let index = (id as usize).min(vocab_size.saturating_sub(1));
                    model.embeddings[index].clone()
                })
                .collect()
        };

        // Real SSD scan through the models crate.
        let states = self.process_with_chunking(hidden, &self.config.chunking_strategy).await?;

        // Real LM head projection.
        let (logits_last, logits_flat, predicted_ids) = {
            let model = self.model.read().await;
            let mut flat: Vec<f32> = Vec::with_capacity(states.len() * model.lm_head.len());
            let mut predicted = Vec::with_capacity(states.len());
            let mut last_row: Vec<f32> = Vec::new();
            for state in &states {
                let row: Vec<f32> = model
                    .lm_head
                    .iter()
                    .map(|weights| {
                        weights.iter().zip(state.iter()).map(|(w, h)| w * h).sum::<f64>() as f32
                    })
                    .collect();
                let argmax = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0);
                predicted.push(argmax);
                flat.extend_from_slice(&row);
                last_row = row;
            }
            (last_row, flat, predicted)
        };

        // Real detokenization of the ids the model actually selected.
        let text = self.tokenizer.decode(&predicted_ids)?;

        let performance = self.compute_performance_metrics(start_time, token_ids.len()).await;

        {
            let mut tracker = self.performance_monitor.write().await;
            tracker.total_tokens_processed += token_ids.len() as u64;
            tracker.average_latency_per_token =
                performance.inference_time_ms / token_ids.len().max(1) as f32;
            tracker.throughput_tokens_per_second = performance.tokens_per_second;
            tracker.memory_usage_mb = performance.memory_usage_mb.unwrap_or(0.0);
        }

        let hidden_states: Vec<f32> =
            states.last().map(|s| s.iter().map(|v| *v as f32).collect()).unwrap_or_default();
        let ssm_states: Vec<f32> =
            states.iter().flat_map(|s| s.iter().map(|v| *v as f32)).collect();

        let _ = logits_last;
        Ok(Mamba2Output {
            text,
            logits: logits_flat,
            hidden_states,
            ssm_states,
            performance,
            token_ids: predicted_ids,
            state_trajectory: None,
        })
    }
}

impl From<Mamba2Output> for PipelineOutput {
    fn from(output: Mamba2Output) -> Self {
        PipelineOutput::Mamba2(output)
    }
}

/// Factory functions for common Mamba-2 configurations
///
/// Each takes the same [`Mamba2PipelineOptions`] as [`Mamba2Pipeline::new`], so
/// the "no checkpoint is being loaded" acknowledgement cannot be bypassed by
/// going through a convenience constructor.

/// Create a high-performance Mamba-2 pipeline optimized for throughput.
///
/// # Errors
///
/// Same as [`Mamba2Pipeline::new`].
pub fn create_high_performance_mamba2_pipeline(
    options: Mamba2PipelineOptions,
) -> Result<Mamba2Pipeline> {
    let config = Mamba2Config {
        d_model: 1024,
        d_state: 32,
        expand_factor: 4,
        d_conv: 8,
        n_heads: 8,
        hardware_strategy: HardwareStrategy::MaxThroughput,
        chunking_strategy: ChunkingStrategy::Overlapping {
            chunk_size: 2048,
            overlap: 256,
        },
        memory_optimization: MemoryOptimization::Balanced,
        ..Default::default()
    };

    Mamba2Pipeline::new(config, options)
}

/// Create a memory-efficient Mamba-2 pipeline for resource-constrained
/// environments.
///
/// # Errors
///
/// Same as [`Mamba2Pipeline::new`].
pub fn create_memory_efficient_mamba2_pipeline(
    options: Mamba2PipelineOptions,
) -> Result<Mamba2Pipeline> {
    let config = Mamba2Config {
        d_model: 512,
        d_state: 8,
        expand_factor: 2,
        d_conv: 4,
        n_heads: 4,
        hardware_strategy: HardwareStrategy::MemoryConstrained,
        chunking_strategy: ChunkingStrategy::Adaptive,
        memory_optimization: MemoryOptimization::Aggressive,
        ..Default::default()
    };

    Mamba2Pipeline::new(config, options)
}

/// Create an ultra-long sequence Mamba-2 pipeline.
///
/// # Errors
///
/// Same as [`Mamba2Pipeline::new`].
pub fn create_ultra_long_sequence_mamba2_pipeline(
    options: Mamba2PipelineOptions,
) -> Result<Mamba2Pipeline> {
    let config = Mamba2Config {
        d_model: 768,
        d_state: 16,
        expand_factor: 2,
        d_conv: 4,
        n_heads: 1,
        hardware_strategy: HardwareStrategy::Auto,
        chunking_strategy: ChunkingStrategy::Hierarchical {
            levels: vec![1024, 256, 64],
        },
        memory_optimization: MemoryOptimization::Ultra,
        ..Default::default()
    };

    Mamba2Pipeline::new(config, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A real character-level tokenizer over a small ASCII vocabulary.
    fn test_tokenizer() -> Arc<dyn Tokenizer> {
        let vocab: HashMap<String, u32> = ('a'..='z')
            .chain(" .".chars())
            .enumerate()
            .map(|(i, c)| (c.to_string(), i as u32))
            .collect();
        Arc::new(crate::tokenizers::CharTokenizer::new(vocab))
    }

    fn test_options(allow_untrained: bool) -> Mamba2PipelineOptions {
        Mamba2PipelineOptions {
            tokenizer: test_tokenizer(),
            allow_untrained_weights: allow_untrained,
        }
    }

    /// A small configuration so tests stay fast.
    fn small_config() -> Mamba2Config {
        Mamba2Config {
            d_model: 16,
            d_state: 4,
            expand_factor: 2,
            d_conv: 2,
            n_heads: 2,
            chunking_strategy: ChunkingStrategy::None,
            ..Mamba2Config::default()
        }
    }

    fn small_pipeline() -> Mamba2Pipeline {
        Mamba2Pipeline::new(small_config(), test_options(true))
            .expect("an explicitly untrained pipeline should build")
    }

    // ── Config defaults ───────────────────────────────────────────────────────

    #[test]
    fn test_config_default_d_model() {
        let config = Mamba2Config::default();
        assert_eq!(config.d_model, 768);
    }

    #[test]
    fn test_config_default_d_state() {
        let config = Mamba2Config::default();
        assert_eq!(config.d_state, 16);
    }

    #[test]
    fn test_config_dt_rank_auto_calculation() {
        let config = Mamba2Config::default();
        let expected_dt_rank = config.d_model / 16;
        let actual = config.dt_rank.unwrap_or(config.d_model / 16);
        assert_eq!(actual, expected_dt_rank);
    }

    // ── A matrix initialisation (discretisation params) ───────────────────────

    #[test]
    fn test_a_matrix_simplified_init_non_positive() {
        let a = Mamba2Pipeline::initialize_a_matrix(8, true);
        for &val in &a {
            assert!(val <= 0.0);
        }
        let strictly_negative = a.iter().filter(|&&v| v < 0.0).count();
        assert!(strictly_negative >= a.len() - 1);
    }

    #[test]
    fn test_a_matrix_values_finite() {
        for simplified in [true, false] {
            for &val in &Mamba2Pipeline::initialize_a_matrix(16, simplified) {
                assert!(val.is_finite());
            }
        }
    }

    // -----------------------------------------------------------------------
    // Regression tests for the removed placeholder implementation.
    //
    // The old pipeline initialised `b_matrix`/`c_matrix` to all zeros (making
    // the scan return its input unchanged), tokenized text with
    // `c as u32 % 32000`, and reported `state_compression_ratio: 0.8`,
    // `hardware_efficiency: 0.92`, `selective_scan_utilization: 0.95` as
    // constants. Every test below fails against that code.
    // -----------------------------------------------------------------------

    /// Building a pipeline without acknowledging the missing checkpoint must
    /// fail rather than hand back an untrained model.
    #[test]
    fn untrained_weights_require_an_explicit_opt_in() {
        let Err(err) = Mamba2Pipeline::new(small_config(), test_options(false)) else {
            panic!("an unacknowledged untrained model must be refused");
        };
        assert!(
            err.to_string().contains("allow_untrained_weights"),
            "the error should name the opt-in: {err}"
        );
    }

    /// `from_pretrained` must not silently produce an untrained model.
    #[test]
    fn from_pretrained_is_not_silently_untrained() {
        assert!(Mamba2Pipeline::from_pretrained("state-spaces/mamba2-2.7b").is_err());
    }

    /// The B and C projections must not be uniformly zero.
    #[test]
    fn projection_matrices_are_not_degenerate() {
        let config = small_config();
        let model = Mamba2Pipeline::initialize_model(&config, 32).expect("model init");
        for layer in &model.layers {
            assert!(
                layer.b_matrix.iter().any(|v| v.abs() > 1e-6),
                "an all-zero B matrix would drop the input term entirely"
            );
            assert!(
                layer.c_matrix.iter().any(|v| v.abs() > 1e-6),
                "an all-zero C matrix reduces the scan to the identity"
            );
        }
        assert_eq!(model.blocks.len(), MAMBA2_LAYER_COUNT);
    }

    /// The scan must actually transform the sequence.
    #[tokio::test(flavor = "multi_thread")]
    async fn selective_scan_changes_the_hidden_states() {
        let pipeline = small_pipeline();
        let input: Vec<Vec<f64>> =
            (0..4).map(|t| (0..16).map(|i| ((t * 16 + i) as f64).sin()).collect()).collect();
        let output = pipeline
            .selective_scan_sequence(input.clone())
            .await
            .expect("the SSD scan should run");

        assert_eq!(output.len(), input.len());
        let changed = output
            .iter()
            .zip(input.iter())
            .any(|(o, i)| o.iter().zip(i.iter()).any(|(a, b)| (a - b).abs() > 1e-9));
        assert!(
            changed,
            "the scan returned its input unchanged, which is what an all-zero C matrix does"
        );
        for row in &output {
            assert!(row.iter().all(|v| v.is_finite()));
        }
    }

    /// Token ids must come from the tokenizer, not from character arithmetic.
    #[tokio::test(flavor = "multi_thread")]
    async fn text_input_is_tokenized_with_the_real_tokenizer() {
        let pipeline = small_pipeline();
        let expected = pipeline
            .tokenizer
            .encode("hello world")
            .expect("tokenizer should encode")
            .input_ids;
        let encoded = pipeline
            .encode_input(PipelineInput::Text("hello world".to_string()))
            .expect("encoding should succeed");
        assert_eq!(encoded, expected);
        assert!(
            encoded.iter().all(|id| (*id as usize) < pipeline.tokenizer.vocab_size()),
            "every id must be a real vocabulary entry"
        );
    }

    /// The reported metrics must be measured, not constants.
    #[tokio::test(flavor = "multi_thread")]
    async fn performance_metrics_are_measured() {
        let pipeline = small_pipeline();
        let output = pipeline
            .process_async(PipelineInput::Text("hello world".to_string()))
            .await
            .expect("a real run should succeed");

        assert!(output.performance.inference_time_ms >= 0.0);
        assert!(output.performance.tokens_per_second.is_finite());
        assert_ne!(
            output.performance.state_compression_ratio, 0.8,
            "0.8 was the old hardcoded compression ratio"
        );
        assert!(!output.token_ids.is_empty(), "a real run selects real ids");
        assert!(
            output
                .token_ids
                .iter()
                .all(|id| (*id as usize) < pipeline.tokenizer.vocab_size()),
            "predicted ids must lie inside the tokenizer's vocabulary"
        );
        assert!(output.state_trajectory.is_none());
    }

    /// Chunked processing must cover the whole sequence.
    #[tokio::test(flavor = "multi_thread")]
    async fn chunked_processing_covers_the_sequence() {
        let mut config = small_config();
        config.chunking_strategy = ChunkingStrategy::Fixed(2);
        let pipeline =
            Mamba2Pipeline::new(config, test_options(true)).expect("pipeline should build");
        let input: Vec<Vec<f64>> = (0..5).map(|t| vec![t as f64 * 0.1; 16]).collect();
        let output = pipeline
            .process_with_chunking(input.clone(), &ChunkingStrategy::Fixed(2))
            .await
            .expect("chunked scan should run");
        assert_eq!(output.len(), input.len());
    }

    /// Unsupported inputs are refused rather than coerced.
    #[test]
    fn unsupported_input_is_refused() {
        let pipeline = small_pipeline();
        assert!(pipeline.encode_input(PipelineInput::Tokens(Vec::new())).is_err());
    }
}
