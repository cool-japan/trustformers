use crate::distributed::ProcessGroup;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use trustformers_core::tensor::Tensor;

/// Sequence Parallelism Configuration
///
/// Sequence parallelism distributes long sequences across multiple devices,
/// enabling the processing of sequences that are too long to fit on a single device.
/// This is particularly useful for very long document processing, DNA sequences,
/// or other sequential data that exceeds memory limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceParallelismConfig {
    /// Number of devices for sequence parallelism
    pub sequence_parallel_size: usize,
    /// Maximum sequence length per device
    pub max_sequence_length_per_device: usize,
    /// Overlap size between adjacent sequence chunks
    pub overlap_size: usize,
    /// Whether to use attention communication optimization
    pub attention_communication_opt: bool,
    /// Communication pattern for sequence parallelism
    pub communication_pattern: SequenceCommunicationPattern,
    /// Sequence splitting strategy
    pub splitting_strategy: SequenceSplittingStrategy,
    /// Whether to use gradient synchronization across sequence chunks
    pub sync_gradients: bool,
    /// Memory optimization for long sequences
    pub memory_optimization: SequenceMemoryOptimization,
    /// Whether to use checkpointing for sequence chunks
    pub use_checkpointing: bool,
}

impl Default for SequenceParallelismConfig {
    fn default() -> Self {
        Self {
            sequence_parallel_size: 1,
            max_sequence_length_per_device: 2048,
            overlap_size: 128,
            attention_communication_opt: true,
            communication_pattern: SequenceCommunicationPattern::RingAllReduce,
            splitting_strategy: SequenceSplittingStrategy::EqualChunks,
            sync_gradients: true,
            memory_optimization: SequenceMemoryOptimization::Medium,
            use_checkpointing: true,
        }
    }
}

/// Communication patterns for sequence parallelism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SequenceCommunicationPattern {
    /// Ring-based all-reduce for efficient communication
    RingAllReduce,
    /// Tree-based reduction
    TreeReduce,
    /// Point-to-point communication between adjacent chunks
    PointToPoint,
    /// All-to-all communication for global attention
    AllToAll,
    /// Hierarchical communication pattern
    Hierarchical,
}

/// Sequence splitting strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SequenceSplittingStrategy {
    /// Split into equal-sized chunks.
    EqualChunks,
    /// Split at the attention troughs measured by
    /// [`SequenceParallelism::observe_attention`]. Falls back to deterministic
    /// uniform chunking while no attention has been observed.
    AttentionBased,
    /// Split at sentence/paragraph boundaries registered with
    /// [`SequenceParallelism::set_segment_boundaries`]. Errors when none are
    /// registered — this crate cannot infer them from a sequence length.
    SemanticBoundaries,
    /// Uniform chunking whose chunk size shrinks with the measured memory
    /// pressure.
    Dynamic,
    /// Prefix-sum load balancing over the per-position costs registered with
    /// [`SequenceParallelism::set_position_costs`]. Errors when none are
    /// registered.
    ComplexityBased,
}

/// Memory optimization strategies for sequence parallelism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SequenceMemoryOptimization {
    None,
    Low,
    Medium,
    High,
    Extreme,
}

/// Sequence chunk information
#[derive(Debug, Clone)]
pub struct SequenceChunk {
    /// Chunk ID
    pub chunk_id: usize,
    /// Device rank where this chunk is processed
    pub device_rank: usize,
    /// Start position in the original sequence
    pub start_position: usize,
    /// End position in the original sequence (exclusive)
    pub end_position: usize,
    /// Effective length (excluding overlap)
    pub effective_length: usize,
    /// Overlap with previous chunk
    pub prev_overlap: usize,
    /// Overlap with next chunk
    pub next_overlap: usize,
    /// Whether this chunk needs attention communication
    pub needs_attention_comm: bool,
}

/// Attention communication info for cross-chunk attention
#[derive(Debug, Clone)]
pub struct AttentionCommunication {
    /// Source chunk ID
    pub source_chunk: usize,
    /// Target chunk ID
    pub target_chunk: usize,
    /// Attention scores that need to be communicated
    pub attention_positions: Vec<(usize, usize)>, // (query_pos, key_pos)
    /// Communication volume in bytes
    pub communication_size: usize,
}

/// Sequence parallelism coordinator
pub struct SequenceParallelism {
    config: SequenceParallelismConfig,
    global_rank: usize,
    world_size: usize,

    // Sequence chunk assignments
    sequence_chunks: Vec<SequenceChunk>,
    local_chunks: Vec<usize>, // Chunk IDs local to this device

    // Process groups for sequence parallelism
    sequence_group: Arc<dyn ProcessGroup>,

    // Attention communication management
    attention_comm_manager: Arc<RwLock<AttentionCommManager>>,

    // Measured attention structure driving attention-based partitioning.
    attention_profile: Arc<RwLock<Option<AttentionProfile>>>,

    // Caller-supplied semantic segment boundaries (token offsets) used by
    // `SequenceSplittingStrategy::SemanticBoundaries`.
    segment_boundaries: Arc<RwLock<Option<Vec<usize>>>>,

    // Caller-supplied per-position computational cost used by
    // `SequenceSplittingStrategy::ComplexityBased`.
    position_costs: Arc<RwLock<Option<Vec<f32>>>>,

    // Per-chunk computation supplied by the caller.
    chunk_processor: Arc<RwLock<Option<ChunkProcessor>>>,

    // Communication statistics
    communication_stats: Arc<Mutex<SequenceCommunicationStats>>,

    // Memory management for sequence chunks
    memory_manager: Arc<Mutex<SequenceMemoryManager>>,
}

/// Attention communication manager
#[derive(Debug, Default)]
struct AttentionCommManager {
    communication_plan: Vec<AttentionCommunication>,
    attention_cache: HashMap<(usize, usize), Tensor>, // (chunk_pair, cached_attention)
    cache_hits: u64,
    cache_misses: u64,
}

/// Communication statistics for sequence parallelism
#[derive(Debug, Default)]
struct SequenceCommunicationStats {
    total_communication_time: Duration,
    attention_communication_time: Duration,
    gradient_sync_time: Duration,
    total_bytes_communicated: u64,
    attention_cache_hit_rate: f32,
    communication_efficiency: f32,
}

/// Memory management for sequence chunks
#[derive(Debug, Default)]
struct SequenceMemoryManager {
    chunk_activations: HashMap<usize, Vec<Tensor>>,
    chunk_gradients: HashMap<usize, Vec<Tensor>>,
    checkpointed_chunks: HashMap<usize, Vec<Tensor>>,
    peak_memory_per_chunk: HashMap<usize, u64>,
    current_memory_usage: u64,
    memory_pressure: f32,
}

/// Comprehensive attention pattern analysis for intelligent sequence splitting
#[derive(Debug, Clone)]
pub struct AttentionPatternAnalysis {
    /// Total sequence length being analyzed
    pub total_length: usize,
    /// Positions identified as natural attention boundaries
    pub attention_boundaries: Vec<usize>,
    /// Attention intensity scores for different sequence regions
    pub attention_intensities: Vec<f32>,
    /// Cross-chunk attention strengths between adjacent chunks
    pub cross_chunk_attention: HashMap<(usize, usize), f32>,
    /// Token importance scores across the sequence
    pub token_importance: Vec<f32>,
    /// Attention head pattern analysis
    pub attention_head_patterns: Vec<AttentionHeadPattern>,
}

/// Computation applied to one sequence chunk: `(input, chunk) -> output`.
///
/// Sequence parallelism splits the sequence; it does not own the transformer
/// layers, so the caller supplies the per-chunk computation.
pub type ChunkProcessor = Box<dyn Fn(&Tensor, &SequenceChunk) -> Result<Tensor> + Send + Sync>;

/// Message tag for one step of the ring attention rotation.
fn ring_tag(chunk_id: usize, step: usize) -> u64 {
    (chunk_id as u64) << 20 | (step as u64 & 0xF_FFFF)
}

/// Message tag for a neighbour exchange between two adjacent ranks.
fn neighbour_tag(chunk_id: usize, lower_rank: usize, upper_rank: usize) -> u64 {
    1 << 40 | (chunk_id as u64) << 20 | (lower_rank as u64) << 10 | (upper_rank as u64 & 0x3FF)
}

/// Window used when scanning a sequence for attention structure.
const ANALYSIS_WINDOW: usize = 512;

/// Cross-window connectivity below which a seam is treated as a natural
/// sequence boundary.
const BOUNDARY_THRESHOLD: f32 = 0.3;

/// Observed attention structure for one sequence.
///
/// This is the *measured* input that drives attention-based partitioning.
/// Build it from a real attention matrix with
/// [`SequenceParallelism::observe_attention`]; without it the partitioner
/// falls back to deterministic uniform chunking rather than guessing.
#[derive(Debug, Clone)]
pub struct AttentionProfile {
    /// Sequence length the profile describes.
    pub sequence_length: usize,
    /// Row-major `[sequence_length, sequence_length]` attention weights,
    /// averaged over heads and batch.
    weights: Vec<f32>,
    /// Column mass: total attention each position receives, normalised so the
    /// mean is 1.0.
    received: Vec<f32>,
    /// Normalised entropy of each position's outgoing attention (0 = focused
    /// on one position, 1 = uniform).
    spread: Vec<f32>,
}

impl AttentionProfile {
    /// Build a profile from a `[sequence, sequence]` attention matrix.
    ///
    /// Rows are queries, columns are keys. Values must be non-negative.
    pub fn from_matrix(weights: Vec<f32>, sequence_length: usize) -> Result<Self> {
        if sequence_length == 0 {
            return Err(anyhow!("attention profile needs a non-empty sequence"));
        }
        if weights.len() != sequence_length * sequence_length {
            return Err(anyhow!(
                "attention matrix has {} values but sequence length {} needs {}",
                weights.len(),
                sequence_length,
                sequence_length * sequence_length
            ));
        }
        if weights.iter().any(|weight| *weight < 0.0 || !weight.is_finite()) {
            return Err(anyhow!("attention weights must be finite and non-negative"));
        }

        // Column mass, normalised so that a uniform matrix yields 1.0
        // everywhere and the fallback path stays comparable.
        let mut received = vec![0.0f32; sequence_length];
        for row in 0..sequence_length {
            for (column, slot) in received.iter_mut().enumerate() {
                *slot += weights[row * sequence_length + column];
            }
        }
        let total: f32 = received.iter().sum();
        if total > 0.0 {
            let scale = sequence_length as f32 / total;
            for slot in received.iter_mut() {
                *slot *= scale;
            }
        }

        // Normalised Shannon entropy of each row.
        let log_n = (sequence_length as f32).ln();
        let mut spread = vec![0.0f32; sequence_length];
        for (row, slot) in spread.iter_mut().enumerate() {
            let start = row * sequence_length;
            let row_sum: f32 = weights[start..start + sequence_length].iter().sum();
            if row_sum <= 0.0 || log_n <= 0.0 {
                *slot = 0.0;
                continue;
            }
            let mut entropy = 0.0f32;
            for weight in &weights[start..start + sequence_length] {
                let probability = weight / row_sum;
                if probability > 0.0 {
                    entropy -= probability * probability.ln();
                }
            }
            *slot = (entropy / log_n).clamp(0.0, 1.0);
        }

        Ok(Self {
            sequence_length,
            weights,
            received,
            spread,
        })
    }

    /// Mean attention weight over the block `rows x columns`.
    pub fn block_mean(
        &self,
        row_start: usize,
        row_end: usize,
        column_start: usize,
        column_end: usize,
    ) -> f32 {
        let row_end = row_end.min(self.sequence_length);
        let column_end = column_end.min(self.sequence_length);
        if row_start >= row_end || column_start >= column_end {
            return 0.0;
        }
        let mut total = 0.0f32;
        for row in row_start..row_end {
            let offset = row * self.sequence_length;
            for column in column_start..column_end {
                total += self.weights[offset + column];
            }
        }
        let cells = ((row_end - row_start) * (column_end - column_start)) as f32;
        // Scale by the sequence length so a uniform matrix maps to 1.0,
        // matching the fallback path's range.
        total / cells * self.sequence_length as f32
    }

    /// Normalised attention mass received by `position`.
    pub fn received_attention(&self, position: usize) -> f32 {
        self.received.get(position).copied().unwrap_or(0.0)
    }

    /// Normalised entropy of `position`'s outgoing attention.
    pub fn spread(&self, position: usize) -> f32 {
        self.spread.get(position).copied().unwrap_or(0.0)
    }
}

/// Attention pattern types identified by different attention heads
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionPatternType {
    /// Local attention patterns (within small windows)
    Local,
    /// Global attention patterns (across entire sequence)
    Global,
    /// Syntactic attention patterns (grammatical structures)
    Syntactic,
    /// Semantic attention patterns (meaning-based dependencies)
    Semantic,
}

/// Analysis of individual attention head patterns
#[derive(Debug, Clone)]
pub struct AttentionHeadPattern {
    /// Attention head identifier
    pub head_id: usize,
    /// Type of attention pattern this head exhibits
    pub pattern_type: AttentionPatternType,
    /// Typical attention span for this head
    pub attention_span: usize,
    /// Strength of the attention pattern (0.0 to 1.0)
    pub pattern_strength: f32,
    /// Communication requirement for distributed processing
    pub communication_requirement: f32,
}

impl SequenceParallelism {
    /// Create a new sequence parallelism coordinator
    pub fn new(
        config: SequenceParallelismConfig,
        global_rank: usize,
        world_size: usize,
        sequence_group: Arc<dyn ProcessGroup>,
    ) -> Result<Self> {
        // Validate configuration
        if config.sequence_parallel_size > world_size {
            return Err(anyhow!(
                "Sequence parallel size ({}) cannot exceed world size ({})",
                config.sequence_parallel_size,
                world_size
            ));
        }

        if config.overlap_size >= config.max_sequence_length_per_device {
            return Err(anyhow!(
                "Overlap size ({}) must be smaller than max sequence length per device ({})",
                config.overlap_size,
                config.max_sequence_length_per_device
            ));
        }

        Ok(Self {
            config,
            global_rank,
            world_size,
            sequence_chunks: Vec::new(),
            local_chunks: Vec::new(),
            sequence_group,
            attention_comm_manager: Arc::new(RwLock::new(AttentionCommManager::default())),
            attention_profile: Arc::new(RwLock::new(None)),
            segment_boundaries: Arc::new(RwLock::new(None)),
            position_costs: Arc::new(RwLock::new(None)),
            chunk_processor: Arc::new(RwLock::new(None)),
            communication_stats: Arc::new(Mutex::new(SequenceCommunicationStats::default())),
            memory_manager: Arc::new(Mutex::new(SequenceMemoryManager::default())),
        })
    }

    /// Record the attention matrix that attention-based partitioning should
    /// analyse.
    ///
    /// `attention_weights` may be `[sequence, sequence]`, `[heads, sequence,
    /// sequence]` or `[batch, heads, sequence, sequence]`; the leading axes are
    /// averaged away. Until this is called, attention-based splitting falls
    /// back to deterministic uniform chunking.
    pub fn observe_attention(&self, attention_weights: &Tensor) -> Result<()> {
        let shape = attention_weights.shape();
        let sequence_length = *shape
            .last()
            .ok_or_else(|| anyhow!("attention weights must have at least one dimension"))?;
        if shape.len() < 2 || shape[shape.len() - 2] != sequence_length {
            return Err(anyhow!(
                "attention weights must end in a square [sequence, sequence] block, got {shape:?}"
            ));
        }

        let values = attention_weights.to_vec_f32()?;
        let matrix_size = sequence_length * sequence_length;
        if values.len() % matrix_size != 0 {
            return Err(anyhow!(
                "attention weights hold {} values which is not a multiple of {matrix_size}",
                values.len()
            ));
        }
        let matrices = values.len() / matrix_size;

        // Average over batch and heads.
        let mut averaged = vec![0.0f32; matrix_size];
        for matrix in 0..matrices {
            let offset = matrix * matrix_size;
            for (slot, value) in averaged.iter_mut().zip(&values[offset..offset + matrix_size]) {
                *slot += *value;
            }
        }
        let scale = 1.0 / matrices as f32;
        for slot in averaged.iter_mut() {
            *slot *= scale;
        }

        let profile = AttentionProfile::from_matrix(averaged, sequence_length)?;
        let mut slot =
            self.attention_profile.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        *slot = Some(profile);
        Ok(())
    }

    /// Register the computation applied to each local sequence chunk.
    pub fn set_chunk_processor(&self, processor: ChunkProcessor) {
        let mut slot =
            self.chunk_processor.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        *slot = Some(processor);
    }

    /// Whether a chunk processor has been registered.
    pub fn has_chunk_processor(&self) -> bool {
        self.chunk_processor
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_some()
    }

    /// Install a pre-built attention profile.
    pub fn set_attention_profile(&self, profile: AttentionProfile) {
        let mut slot =
            self.attention_profile.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        *slot = Some(profile);
    }

    /// The currently registered attention profile, if any.
    pub fn attention_profile(&self) -> Option<AttentionProfile> {
        self.attention_profile
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Record the semantic segment boundaries that
    /// [`SequenceSplittingStrategy::SemanticBoundaries`] must respect.
    ///
    /// `boundaries` are token offsets at which a sentence/paragraph/document
    /// starts. This crate has no tokenizer or sentence splitter, so the
    /// boundaries have to come from whoever produced the token stream; without
    /// them semantic splitting reports an error rather than silently degrading
    /// to uniform chunks.
    ///
    /// Offsets are sorted and de-duplicated; `0` is implicit. An offset of `0`
    /// or a duplicate is accepted and ignored.
    pub fn set_segment_boundaries(
        &self,
        boundaries: impl IntoIterator<Item = usize>,
    ) -> Result<()> {
        let mut sorted: Vec<usize> = boundaries.into_iter().filter(|offset| *offset > 0).collect();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.is_empty() {
            return Err(anyhow!(
                "semantic segment boundaries must contain at least one interior offset (> 0)"
            ));
        }
        let mut slot =
            self.segment_boundaries.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        *slot = Some(sorted);
        Ok(())
    }

    /// The registered semantic segment boundaries, if any.
    pub fn segment_boundaries(&self) -> Option<Vec<usize>> {
        self.segment_boundaries
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Record the per-position computational cost that
    /// [`SequenceSplittingStrategy::ComplexityBased`] balances over.
    ///
    /// `costs[i]` is the relative work required by token `i` — for example the
    /// number of unmasked keys it attends to, or a measured per-token latency.
    /// Values must be finite and non-negative. Without a registration,
    /// complexity-based splitting reports an error instead of inventing scores.
    pub fn set_position_costs(&self, costs: Vec<f32>) -> Result<()> {
        if costs.is_empty() {
            return Err(anyhow!("position costs must not be empty"));
        }
        if costs.iter().any(|cost| !cost.is_finite() || *cost < 0.0) {
            return Err(anyhow!("position costs must be finite and non-negative"));
        }
        if costs.iter().sum::<f32>() <= 0.0 {
            return Err(anyhow!("position costs must not be all zero"));
        }
        let mut slot = self.position_costs.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        *slot = Some(costs);
        Ok(())
    }

    /// The registered per-position costs, if any.
    pub fn position_costs(&self) -> Option<Vec<f32>> {
        self.position_costs
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Split a sequence across multiple devices
    pub fn split_sequence(&mut self, total_sequence_length: usize) -> Result<Vec<SequenceChunk>> {
        let chunks = match self.config.splitting_strategy {
            SequenceSplittingStrategy::EqualChunks => {
                self.split_equal_chunks(total_sequence_length)?
            },
            SequenceSplittingStrategy::AttentionBased => {
                self.split_attention_based(total_sequence_length)?
            },
            SequenceSplittingStrategy::SemanticBoundaries => {
                self.split_semantic_boundaries(total_sequence_length)?
            },
            SequenceSplittingStrategy::Dynamic => self.split_dynamic(total_sequence_length)?,
            SequenceSplittingStrategy::ComplexityBased => {
                self.split_complexity_based(total_sequence_length)?
            },
        };

        // Update local chunk assignments
        self.sequence_chunks = chunks.clone();
        self.local_chunks = chunks
            .iter()
            .enumerate()
            .filter(|(_, chunk)| chunk.device_rank == self.global_rank)
            .map(|(i, _)| i)
            .collect();

        Ok(chunks)
    }

    /// Split sequence into equal chunks
    fn split_equal_chunks(&self, total_length: usize) -> Result<Vec<SequenceChunk>> {
        self.split_equal_chunks_of(total_length, self.config.max_sequence_length_per_device)
    }

    /// Uniform chunking with an explicit chunk size, so callers that adapt the
    /// size (see [`SequenceParallelism::split_dynamic`]) actually change the
    /// resulting partition instead of computing a size and discarding it.
    fn split_equal_chunks_of(
        &self,
        total_length: usize,
        chunk_size: usize,
    ) -> Result<Vec<SequenceChunk>> {
        let overlap = self.config.overlap_size;
        let num_devices = self.config.sequence_parallel_size;

        if chunk_size <= overlap {
            return Err(anyhow!(
                "chunk size ({chunk_size}) must exceed the overlap ({overlap}) or the partition \
                 cannot advance"
            ));
        }
        if num_devices == 0 {
            return Err(anyhow!("sequence_parallel_size must be >= 1"));
        }

        let mut chunks = Vec::new();
        let mut current_pos = 0;
        let mut chunk_id = 0;

        while current_pos < total_length {
            let end_pos = std::cmp::min(current_pos + chunk_size, total_length);
            let device_rank = chunk_id % num_devices;

            let prev_overlap = if chunk_id > 0 { overlap } else { 0 };
            let next_overlap = if end_pos < total_length { overlap } else { 0 };

            let chunk = SequenceChunk {
                chunk_id,
                device_rank,
                start_position: current_pos,
                end_position: end_pos,
                effective_length: end_pos - current_pos - prev_overlap,
                prev_overlap,
                next_overlap,
                needs_attention_comm: true,
            };

            chunks.push(chunk);
            current_pos = end_pos - overlap;
            chunk_id += 1;
        }

        Ok(chunks)
    }

    /// Split sequence based on attention patterns using intelligent analysis
    fn split_attention_based(&self, total_length: usize) -> Result<Vec<SequenceChunk>> {
        // Analyze attention patterns to determine optimal split points
        let attention_analysis = self.analyze_attention_patterns(total_length)?;

        // Find optimal split points based on attention boundaries
        let split_points = self.find_optimal_split_points(&attention_analysis, total_length)?;

        // Create chunks based on attention-aware split points
        self.create_attention_aware_chunks(total_length, &split_points)
    }

    /// Analyze attention patterns in the sequence to identify natural
    /// boundaries.
    ///
    /// When an attention profile has been registered with
    /// [`SequenceParallelism::observe_attention`], every quantity below is
    /// computed from the observed attention matrix. Without one, the analysis
    /// falls back to a **deterministic** position-only profile: uniform
    /// intensity, no interior boundaries, uniform token importance. It is never
    /// randomised — an earlier revision drew these from `fastrand`, which made
    /// the chosen partition differ run to run for identical input.
    fn analyze_attention_patterns(&self, total_length: usize) -> Result<AttentionPatternAnalysis> {
        let profile = self.attention_profile();
        let profile = profile.as_ref().filter(|profile| profile.sequence_length == total_length);
        if profile.is_none() {
            log::debug!(
                "no attention profile registered for sequence length {total_length}; \
                 falling back to deterministic uniform analysis"
            );
        }

        let mut attention_boundaries = Vec::new();
        let mut attention_intensities = Vec::new();
        let mut cross_chunk_attention = HashMap::new();

        let window_size = ANALYSIS_WINDOW.min(total_length.max(1));
        let num_windows = total_length.div_ceil(window_size.max(1));

        for window_idx in 0..num_windows {
            let start_pos = window_idx * window_size;
            let end_pos = (start_pos + window_size).min(total_length);

            let local_attention =
                self.calculate_local_attention_intensity(profile, start_pos, end_pos)?;
            let cross_window_attention = self.calculate_cross_window_attention(
                profile,
                window_idx,
                num_windows,
                window_size,
                total_length,
            )?;

            attention_intensities.push(local_attention);

            // A low-connectivity seam between two windows is a natural place to
            // cut the sequence.
            if window_idx > 0 && cross_window_attention < BOUNDARY_THRESHOLD {
                attention_boundaries.push(start_pos);
            }

            if window_idx < num_windows.saturating_sub(1) {
                cross_chunk_attention.insert((window_idx, window_idx + 1), cross_window_attention);
            }
        }

        // Add sequence boundaries
        if !attention_boundaries.contains(&0) {
            attention_boundaries.insert(0, 0);
        }
        if !attention_boundaries.contains(&total_length) {
            attention_boundaries.push(total_length);
        }

        attention_boundaries.sort();

        Ok(AttentionPatternAnalysis {
            total_length,
            attention_boundaries,
            attention_intensities,
            cross_chunk_attention,
            token_importance: self.calculate_token_importance(profile, total_length)?,
            attention_head_patterns: self.analyze_attention_head_patterns(total_length)?,
        })
    }

    /// Mean attention mass a window directs at itself.
    ///
    /// With a profile this is the average of the observed attention block
    /// `A[start..end, start..end]`. Without one it is a constant derived only
    /// from the window's length, which is deterministic by construction.
    fn calculate_local_attention_intensity(
        &self,
        profile: Option<&AttentionProfile>,
        start_pos: usize,
        end_pos: usize,
    ) -> Result<f32> {
        if end_pos <= start_pos {
            return Ok(0.0);
        }
        let window_length = end_pos - start_pos;
        let length_factor = (window_length as f32 / ANALYSIS_WINDOW as f32).min(1.0);

        match profile {
            Some(profile) => Ok(profile.block_mean(start_pos, end_pos, start_pos, end_pos)),
            None => Ok(length_factor),
        }
    }

    /// Attention connectivity between a window and its successor.
    ///
    /// With a profile this is the mean of the two off-diagonal blocks between
    /// the windows, i.e. how much the two halves actually attend to each other.
    fn calculate_cross_window_attention(
        &self,
        profile: Option<&AttentionProfile>,
        window_idx: usize,
        total_windows: usize,
        window_size: usize,
        total_length: usize,
    ) -> Result<f32> {
        if total_windows <= 1 || window_idx + 1 >= total_windows {
            return Ok(0.0);
        }

        let start = window_idx * window_size;
        let mid = ((window_idx + 1) * window_size).min(total_length);
        let end = ((window_idx + 2) * window_size).min(total_length);

        match profile {
            Some(profile) => {
                let forward = profile.block_mean(start, mid, mid, end);
                let backward = profile.block_mean(mid, end, start, mid);
                Ok((forward + backward) * 0.5)
            },
            None => {
                // Deterministic distance decay: interior seams are more
                // strongly connected than the ones next to the sequence edges.
                let decay = if window_idx == 0 || window_idx + 1 == total_windows {
                    0.8
                } else {
                    1.0 - (window_idx as f32 / total_windows as f32)
                };
                Ok(decay)
            },
        }
    }

    /// Per-position importance: how much attention each position *receives*.
    ///
    /// With a profile this is the column mass of the attention matrix, the
    /// standard measure of a token's influence. Without one it is a fixed
    /// position prior (higher at the sequence edges), which is deterministic.
    fn calculate_token_importance(
        &self,
        profile: Option<&AttentionProfile>,
        total_length: usize,
    ) -> Result<Vec<f32>> {
        let mut importance_scores = Vec::with_capacity(total_length);

        for pos in 0..total_length {
            let position_bias = if pos < total_length / 4 || pos > 3 * total_length / 4 {
                1.2 // Beginning and end tokens carry more positional weight
            } else {
                1.0
            };

            let content_importance = match profile {
                Some(profile) => profile.received_attention(pos),
                None => 1.0,
            };
            let attention_centrality =
                self.calculate_attention_centrality(profile, pos, total_length)?;

            importance_scores.push(position_bias * content_importance * attention_centrality);
        }

        Ok(importance_scores)
    }

    /// Attention centrality of a position.
    ///
    /// With a profile this combines the position's received attention with how
    /// evenly it distributes its own attention (normalised entropy): a position
    /// that both receives a lot and attends broadly is central. Without a
    /// profile only the deterministic positional term remains.
    fn calculate_attention_centrality(
        &self,
        profile: Option<&AttentionProfile>,
        pos: usize,
        total_length: usize,
    ) -> Result<f32> {
        if total_length == 0 {
            return Ok(0.0);
        }
        let relative_pos = pos as f32 / total_length as f32;
        let position_centrality = 1.0 - (2.0 * relative_pos - 1.0).abs();

        match profile {
            Some(profile) => Ok(position_centrality * profile.spread(pos)),
            None => Ok(position_centrality),
        }
    }

    /// Analyze attention head patterns to understand different types of attention
    fn analyze_attention_head_patterns(
        &self,
        total_length: usize,
    ) -> Result<Vec<AttentionHeadPattern>> {
        let num_heads = 12; // Typical number of attention heads
        let mut head_patterns = Vec::with_capacity(num_heads);

        for head_idx in 0..num_heads {
            let pattern_type = match head_idx % 4 {
                0 => AttentionPatternType::Local,     // Local attention patterns
                1 => AttentionPatternType::Global,    // Global attention patterns
                2 => AttentionPatternType::Syntactic, // Syntactic attention patterns
                3 => AttentionPatternType::Semantic,  // Semantic attention patterns
                _ => AttentionPatternType::Local,
            };

            let attention_span = match pattern_type {
                AttentionPatternType::Local => total_length / 8,
                AttentionPatternType::Global => total_length,
                AttentionPatternType::Syntactic => total_length / 4,
                AttentionPatternType::Semantic => total_length / 2,
            };

            // Deterministic per-pattern strength: heads whose span covers more
            // of the sequence exhibit a correspondingly stronger pattern. An
            // earlier revision drew this from `fastrand`, making the analysis
            // irreproducible.
            let pattern_strength = if total_length == 0 {
                0.0
            } else {
                (0.4 + 0.6 * (attention_span as f32 / total_length as f32)).clamp(0.0, 1.0)
            };
            let communication_requirement =
                self.calculate_communication_requirement(&pattern_type, total_length)?;

            head_patterns.push(AttentionHeadPattern {
                head_id: head_idx,
                pattern_type,
                attention_span,
                pattern_strength,
                communication_requirement,
            });
        }

        Ok(head_patterns)
    }

    /// Calculate communication requirement for an attention pattern type
    fn calculate_communication_requirement(
        &self,
        pattern_type: &AttentionPatternType,
        _total_length: usize,
    ) -> Result<f32> {
        match pattern_type {
            AttentionPatternType::Local => Ok(0.1), // Low communication for local patterns
            AttentionPatternType::Global => Ok(0.9), // High communication for global patterns
            AttentionPatternType::Syntactic => Ok(0.4), // Medium communication for syntactic patterns
            AttentionPatternType::Semantic => Ok(0.6), // Medium-high communication for semantic patterns
        }
    }

    /// Find optimal split points based on attention analysis
    fn find_optimal_split_points(
        &self,
        analysis: &AttentionPatternAnalysis,
        total_length: usize,
    ) -> Result<Vec<usize>> {
        let target_chunks = self.config.sequence_parallel_size;
        let min_chunk_size = self.config.max_sequence_length_per_device / 2;
        let max_chunk_size = self.config.max_sequence_length_per_device;

        if target_chunks == 1 {
            return Ok(vec![0, total_length]);
        }

        // Use dynamic programming to find optimal split points
        let mut split_points = vec![0];
        let mut remaining_length = total_length;
        let mut remaining_chunks = target_chunks;

        for chunk_idx in 0..target_chunks - 1 {
            let avg_remaining_chunk_size = remaining_length / remaining_chunks;
            let target_split_pos = split_points[chunk_idx] + avg_remaining_chunk_size;

            // Find the best boundary near the target position
            let best_boundary = self.find_best_boundary_near_position(
                analysis,
                target_split_pos,
                min_chunk_size,
                max_chunk_size,
            )?;

            split_points.push(best_boundary);
            remaining_length = total_length - best_boundary;
            remaining_chunks -= 1;
        }

        split_points.push(total_length);
        Ok(split_points)
    }

    /// Find the best attention boundary near a target position
    fn find_best_boundary_near_position(
        &self,
        analysis: &AttentionPatternAnalysis,
        target_pos: usize,
        min_chunk_size: usize,
        _max_chunk_size: usize,
    ) -> Result<usize> {
        let search_radius = min_chunk_size / 4;
        let start_search = target_pos.saturating_sub(search_radius);
        let end_search = (target_pos + search_radius).min(analysis.total_length);

        let mut best_pos = target_pos;
        let mut best_score = f32::NEG_INFINITY;

        // Evaluate each potential boundary position
        for candidate_pos in (start_search..=end_search).step_by(16) {
            if candidate_pos < min_chunk_size
                || candidate_pos > analysis.total_length - min_chunk_size
            {
                continue;
            }

            let boundary_score =
                self.calculate_boundary_score(analysis, candidate_pos, target_pos)?;

            if boundary_score > best_score {
                best_score = boundary_score;
                best_pos = candidate_pos;
            }
        }

        Ok(best_pos)
    }

    /// Calculate boundary score for a potential split position
    fn calculate_boundary_score(
        &self,
        analysis: &AttentionPatternAnalysis,
        pos: usize,
        target_pos: usize,
    ) -> Result<f32> {
        // Distance penalty (prefer positions close to target)
        let distance_penalty =
            1.0 - (pos as f32 - target_pos as f32).abs() / (target_pos as f32 + 1.0);

        // Attention boundary score (prefer positions with low cross-attention)
        let attention_score = if analysis.attention_boundaries.contains(&pos) {
            1.0
        } else {
            // Calculate interpolated attention score
            0.5 + 0.3 * (1.0 - self.get_cross_attention_at_position(analysis, pos)?)
        };

        // Token importance penalty (avoid splitting at important tokens)
        let importance_penalty = if pos < analysis.token_importance.len() {
            1.0 - analysis.token_importance[pos] * 0.3
        } else {
            1.0
        };

        // Communication cost consideration
        let communication_score =
            1.0 - self.estimate_communication_cost_at_boundary(analysis, pos)?;

        Ok(distance_penalty * 0.3
            + attention_score * 0.4
            + importance_penalty * 0.15
            + communication_score * 0.15)
    }

    /// Get cross-attention strength at a specific position
    fn get_cross_attention_at_position(
        &self,
        analysis: &AttentionPatternAnalysis,
        pos: usize,
    ) -> Result<f32> {
        let window_size = 512;
        let window_idx = pos / window_size;
        let next_window_idx = window_idx + 1;

        // Get cross-chunk attention for this window boundary
        if let Some(&cross_attention) =
            analysis.cross_chunk_attention.get(&(window_idx, next_window_idx))
        {
            Ok(cross_attention)
        } else {
            Ok(0.5) // Default moderate cross-attention
        }
    }

    /// Estimate communication cost if boundary is placed at this position
    fn estimate_communication_cost_at_boundary(
        &self,
        analysis: &AttentionPatternAnalysis,
        pos: usize,
    ) -> Result<f32> {
        let mut total_cost = 0.0;

        // Calculate cost based on attention head patterns
        for head_pattern in &analysis.attention_head_patterns {
            if head_pattern.attention_span > pos && pos > 0 {
                // This boundary would require communication for this attention head
                total_cost +=
                    head_pattern.communication_requirement * head_pattern.pattern_strength;
            }
        }

        // Normalize by number of heads
        Ok(total_cost / analysis.attention_head_patterns.len() as f32)
    }

    /// Create attention-aware chunks based on split points
    fn create_attention_aware_chunks(
        &self,
        _total_length: usize,
        split_points: &[usize],
    ) -> Result<Vec<SequenceChunk>> {
        let mut chunks = Vec::new();

        for i in 0..split_points.len() - 1 {
            let start_pos = split_points[i];
            let end_pos = split_points[i + 1];
            let chunk_length = end_pos - start_pos;

            // Calculate overlaps for attention communication
            let prev_overlap =
                if i > 0 { self.config.overlap_size.min(chunk_length / 4) } else { 0 };

            let next_overlap = if i < split_points.len() - 2 {
                self.config.overlap_size.min(chunk_length / 4)
            } else {
                0
            };

            // Determine if this chunk needs attention communication
            let needs_attention_comm =
                self.chunk_needs_attention_communication(i, split_points.len() - 1)?;

            chunks.push(SequenceChunk {
                chunk_id: i,
                device_rank: i % self.config.sequence_parallel_size,
                start_position: start_pos,
                end_position: end_pos,
                effective_length: chunk_length - prev_overlap - next_overlap,
                prev_overlap,
                next_overlap,
                needs_attention_comm,
            });
        }

        Ok(chunks)
    }

    /// Determine if a chunk needs attention communication with other chunks
    fn chunk_needs_attention_communication(
        &self,
        chunk_idx: usize,
        total_chunks: usize,
    ) -> Result<bool> {
        if total_chunks == 1 {
            return Ok(false);
        }

        // Chunks need attention communication if they have cross-chunk dependencies
        let has_prev_dependency = chunk_idx > 0;
        let has_next_dependency = chunk_idx < total_chunks - 1;

        // Enable attention communication optimization if configured
        if self.config.attention_communication_opt {
            Ok(has_prev_dependency || has_next_dependency)
        } else {
            Ok(false)
        }
    }

    /// Split the sequence so that every chunk boundary coincides with a
    /// registered semantic boundary.
    ///
    /// The algorithm walks the ideal uniform split positions and snaps each one
    /// to the nearest registered boundary that keeps every chunk non-empty and
    /// no longer than [`SequenceParallelismConfig::max_sequence_length_per_device`].
    /// Snapping is deterministic (ties resolve to the earlier boundary), so all
    /// ranks derive the same partition without communicating.
    ///
    /// # Errors
    ///
    /// Returns an error when no boundaries have been registered with
    /// [`SequenceParallelism::set_segment_boundaries`]: this crate cannot
    /// discover sentence or paragraph boundaries from a length alone, and
    /// silently falling back to uniform chunks would misreport the strategy
    /// that ran.
    fn split_semantic_boundaries(&self, total_length: usize) -> Result<Vec<SequenceChunk>> {
        let boundaries = self.segment_boundaries().ok_or_else(|| {
            anyhow!(
                "SequenceSplittingStrategy::SemanticBoundaries needs the segment offsets of the \
                 token stream; register them with SequenceParallelism::set_segment_boundaries, or \
                 select SequenceSplittingStrategy::EqualChunks for uniform partitioning"
            )
        })?;

        if total_length == 0 {
            return Ok(Vec::new());
        }

        let interior: Vec<usize> =
            boundaries.into_iter().filter(|offset| *offset < total_length).collect();
        if interior.is_empty() {
            return Err(anyhow!(
                "no registered semantic boundary falls inside a sequence of length {total_length}"
            ));
        }

        let target_chunks = self.config.sequence_parallel_size.max(1).min(interior.len() + 1);
        let max_chunk = self.config.max_sequence_length_per_device.max(1);

        let mut split_points = vec![0usize];
        for chunk_index in 1..target_chunks {
            let previous = split_points[chunk_index - 1];
            let ideal = total_length * chunk_index / target_chunks;
            let hard_limit = previous.saturating_add(max_chunk).min(total_length);

            // Candidates must advance past `previous`, stay within the per-device
            // budget, and leave at least one token for every remaining chunk.
            let remaining_chunks = target_chunks - chunk_index;
            let latest = hard_limit.min(total_length - remaining_chunks);
            let chosen = interior
                .iter()
                .copied()
                .filter(|offset| *offset > previous && *offset <= latest)
                .min_by_key(|offset| (offset.abs_diff(ideal), *offset));

            match chosen {
                Some(offset) => split_points.push(offset),
                // No boundary is reachable from here: stop splitting rather than
                // inventing a cut that is not a semantic boundary.
                None => break,
            }
        }
        split_points.push(total_length);

        self.create_attention_aware_chunks(total_length, &split_points)
    }

    /// Dynamic sequence splitting: the chunk size shrinks as memory pressure
    /// rises, so a device under pressure holds fewer tokens.
    fn split_dynamic(&self, total_length: usize) -> Result<Vec<SequenceChunk>> {
        let pressure = {
            let memory_manager =
                self.memory_manager.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            memory_manager.memory_pressure
        };

        let base_chunk_size = self.config.max_sequence_length_per_device;
        let adjusted_chunk_size = if pressure > 0.8 {
            base_chunk_size / 2
        } else if pressure > 0.6 {
            (base_chunk_size * 3) / 4
        } else {
            base_chunk_size
        };

        // The overlap must stay strictly inside the chunk or the walk below
        // cannot advance.
        let adjusted_chunk_size = adjusted_chunk_size.max(self.config.overlap_size + 1);

        self.split_equal_chunks_of(total_length, adjusted_chunk_size)
    }

    /// Split the sequence so that every chunk carries a comparable amount of
    /// work according to the registered per-position costs.
    ///
    /// Boundaries are placed where the cumulative cost crosses
    /// `k * total_cost / chunks`, which is the standard prefix-sum load-balanced
    /// partition. It is deterministic and depends only on the registered costs,
    /// so all ranks agree.
    ///
    /// # Errors
    ///
    /// Returns an error when no costs have been registered with
    /// [`SequenceParallelism::set_position_costs`], or when the registered
    /// vector does not cover `total_length` positions.
    fn split_complexity_based(&self, total_length: usize) -> Result<Vec<SequenceChunk>> {
        let costs = self.position_costs().ok_or_else(|| {
            anyhow!(
                "SequenceSplittingStrategy::ComplexityBased needs a per-position cost vector; \
                 register one with SequenceParallelism::set_position_costs, or select \
                 SequenceSplittingStrategy::EqualChunks for uniform partitioning"
            )
        })?;

        if total_length == 0 {
            return Ok(Vec::new());
        }
        if costs.len() < total_length {
            return Err(anyhow!(
                "position costs cover {} positions but the sequence is {} long",
                costs.len(),
                total_length
            ));
        }

        let target_chunks = self.config.sequence_parallel_size.max(1).min(total_length);
        if target_chunks == 1 {
            return self.create_attention_aware_chunks(total_length, &[0, total_length]);
        }

        let costs = &costs[..total_length];
        let total_cost: f64 = costs.iter().map(|cost| *cost as f64).sum();
        if total_cost <= 0.0 {
            return Err(anyhow!("registered position costs sum to zero"));
        }

        let mut split_points = vec![0usize];
        let mut cumulative = 0.0f64;
        let mut next_chunk = 1usize;

        for (position, cost) in costs.iter().enumerate() {
            cumulative += *cost as f64;
            while next_chunk < target_chunks
                && cumulative >= total_cost * next_chunk as f64 / target_chunks as f64
            {
                let candidate = position + 1;
                let remaining_chunks = target_chunks - next_chunk;
                let latest = total_length - remaining_chunks;
                let previous = split_points[next_chunk - 1];
                // Keep every chunk non-empty even when the cost mass is
                // concentrated in a few positions.
                let clamped = candidate.clamp(previous + 1, latest.max(previous + 1));
                split_points.push(clamped);
                next_chunk += 1;
            }
        }

        // Cost mass can run out before the last boundaries are placed (for
        // example when the tail has zero cost); finish with unit-width chunks.
        while next_chunk < target_chunks {
            let previous = split_points[next_chunk - 1];
            split_points.push((previous + 1).min(total_length));
            next_chunk += 1;
        }

        split_points.push(total_length);
        self.create_attention_aware_chunks(total_length, &split_points)
    }

    /// Process forward pass for a sequence chunk
    pub fn forward_chunk(
        &self,
        chunk_id: usize,
        input: &Tensor,
        _attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let start_time = Instant::now();

        if !self.local_chunks.contains(&chunk_id) {
            return Err(anyhow!("Chunk {} is not local to this device", chunk_id));
        }

        let chunk = &self.sequence_chunks[chunk_id];

        // Process the chunk locally
        let output = self.process_local_chunk(input, chunk)?;

        // Handle attention communication if needed
        let final_output = if chunk.needs_attention_comm {
            self.handle_attention_communication(chunk_id, &output)?
        } else {
            output
        };

        // Update statistics
        {
            let mut stats =
                self.communication_stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            stats.total_communication_time += start_time.elapsed();
        }

        Ok(final_output)
    }

    /// Apply the registered per-chunk computation.
    ///
    /// Without a registered processor this returns an error rather than echoing
    /// its input and implying the chunk was processed.
    fn process_local_chunk(&self, input: &Tensor, chunk: &SequenceChunk) -> Result<Tensor> {
        let processor =
            self.chunk_processor.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        match processor.as_ref() {
            Some(processor) => processor(input, chunk),
            None => Err(anyhow!(
                "no chunk processor registered for chunk {}: sequence parallelism partitions the \
                 sequence but does not own the model layers. Register the per-chunk computation \
                 with SequenceParallelism::set_chunk_processor",
                chunk.chunk_id
            )),
        }
    }

    /// Handle attention communication between chunks
    fn handle_attention_communication(
        &self,
        chunk_id: usize,
        chunk_output: &Tensor,
    ) -> Result<Tensor> {
        let start_time = Instant::now();

        match self.config.communication_pattern {
            SequenceCommunicationPattern::RingAllReduce => {
                self.ring_attention_communication(chunk_id, chunk_output)
            },
            SequenceCommunicationPattern::TreeReduce => {
                self.tree_attention_communication(chunk_id, chunk_output)
            },
            SequenceCommunicationPattern::PointToPoint => {
                self.point_to_point_attention(chunk_id, chunk_output)
            },
            SequenceCommunicationPattern::AllToAll => {
                self.all_to_all_attention(chunk_id, chunk_output)
            },
            SequenceCommunicationPattern::Hierarchical => {
                self.hierarchical_attention_communication(chunk_id, chunk_output)
            },
        }
        .inspect(|_result| {
            // Update attention communication statistics
            let mut stats =
                self.communication_stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            stats.attention_communication_time += start_time.elapsed();
        })
    }

    /// Ring-based cross-chunk attention exchange.
    ///
    /// The local chunk output is rotated around the sequence-parallel ring for
    /// `world_size - 1` steps; each rank therefore observes every other rank's
    /// chunk and assembles the full sequence in rank order. This is real data
    /// movement — every step sends and receives a tensor.
    fn ring_attention_communication(
        &self,
        chunk_id: usize,
        chunk_output: &Tensor,
    ) -> Result<Tensor> {
        let world_size = self.sequence_group.world_size();
        if world_size <= 1 {
            // A single rank already holds the whole sequence.
            return Ok(chunk_output.clone());
        }
        self.require_transport("ring attention communication")?;

        let rank = self.sequence_group.rank();
        let next = (rank + 1) % world_size;
        let previous = (rank + world_size - 1) % world_size;
        let shape = chunk_output.shape();

        let mut slots: Vec<Option<Tensor>> = vec![None; world_size];
        slots[rank] = Some(chunk_output.clone());
        let mut in_flight = chunk_output.clone();

        for step in 0..world_size - 1 {
            let tag = ring_tag(chunk_id, step);
            self.sequence_group.send(next, tag, &in_flight)?;
            let received = self.sequence_group.recv(previous, tag, &shape)?;
            let origin = (rank + world_size - step - 1) % world_size;
            slots[origin] = Some(received.clone());
            in_flight = received;
        }

        self.concatenate_chunk_outputs(slots)
    }

    /// Tree-based cross-chunk attention exchange.
    ///
    /// Uses the process group's all-gather (a broadcast tree for groups without
    /// a native ring all-gather) and concatenates the result in rank order.
    fn tree_attention_communication(
        &self,
        _chunk_id: usize,
        chunk_output: &Tensor,
    ) -> Result<Tensor> {
        if self.sequence_group.world_size() <= 1 {
            return Ok(chunk_output.clone());
        }
        let gathered = self.sequence_group.all_gather(chunk_output)?;
        self.concatenate_chunk_outputs(gathered.into_iter().map(Some).collect())
    }

    /// Exchange chunk outputs with the immediate neighbours only.
    ///
    /// This is the cheapest pattern: a chunk sees `[previous, local, next]`,
    /// which is sufficient when attention spans do not exceed one chunk plus
    /// the configured overlap.
    fn point_to_point_attention(&self, chunk_id: usize, chunk_output: &Tensor) -> Result<Tensor> {
        let world_size = self.sequence_group.world_size();
        if world_size <= 1 {
            return Ok(chunk_output.clone());
        }
        self.require_transport("point-to-point attention communication")?;

        let rank = self.sequence_group.rank();
        let shape = chunk_output.shape();
        let mut pieces: Vec<Option<Tensor>> = Vec::with_capacity(3);

        // Exchange with the previous neighbour, then with the next one. Both
        // sides use the same tags, so the order of the two halves does not
        // matter.
        if rank > 0 {
            let tag = neighbour_tag(chunk_id, rank - 1, rank);
            self.sequence_group.send(rank - 1, tag, chunk_output)?;
        }
        if rank + 1 < world_size {
            let tag = neighbour_tag(chunk_id, rank, rank + 1);
            self.sequence_group.send(rank + 1, tag, chunk_output)?;
        }

        if rank > 0 {
            let tag = neighbour_tag(chunk_id, rank - 1, rank);
            pieces.push(Some(self.sequence_group.recv(rank - 1, tag, &shape)?));
        }
        pieces.push(Some(chunk_output.clone()));
        if rank + 1 < world_size {
            let tag = neighbour_tag(chunk_id, rank, rank + 1);
            pieces.push(Some(self.sequence_group.recv(rank + 1, tag, &shape)?));
        }

        self.concatenate_chunk_outputs(pieces)
    }

    /// Gather every rank's chunk output and concatenate them in rank order.
    fn all_to_all_attention(&self, _chunk_id: usize, chunk_output: &Tensor) -> Result<Tensor> {
        if self.sequence_group.world_size() <= 1 {
            return Ok(chunk_output.clone());
        }
        let gathered = self.sequence_group.all_gather(chunk_output)?;
        self.concatenate_chunk_outputs(gathered.into_iter().map(Some).collect())
    }

    /// Hierarchical cross-chunk attention exchange.
    ///
    /// # Errors
    ///
    /// A two-level exchange needs an intra-node subgroup in addition to the
    /// global sequence group; this coordinator is constructed with a single
    /// group, so there is no node structure to exploit. Rather than silently
    /// performing a flat gather and reporting it as hierarchical, this returns
    /// an error naming the pattern to use instead.
    fn hierarchical_attention_communication(
        &self,
        _chunk_id: usize,
        chunk_output: &Tensor,
    ) -> Result<Tensor> {
        if self.sequence_group.world_size() <= 1 {
            return Ok(chunk_output.clone());
        }
        Err(anyhow!(
            "SequenceCommunicationPattern::Hierarchical requires an intra-node subgroup, which \
             SequenceParallelism is not given; use SequenceCommunicationPattern::RingAllReduce \
             or ::AllToAll"
        ))
    }

    /// Concatenate per-rank chunk outputs along the sequence (leading) axis.
    fn concatenate_chunk_outputs(&self, pieces: Vec<Option<Tensor>>) -> Result<Tensor> {
        let mut values = Vec::new();
        let mut rows = 0usize;
        let mut trailing: Vec<usize> = Vec::new();

        for piece in pieces.into_iter().flatten() {
            let shape = piece.shape();
            if trailing.is_empty() {
                trailing = shape.clone();
            }
            rows += shape.first().copied().unwrap_or(0);
            values.extend(piece.to_vec_f32()?);
        }

        if trailing.is_empty() {
            return Err(anyhow!("no chunk outputs to concatenate"));
        }
        trailing[0] = rows;

        let expected: usize = trailing.iter().product();
        if expected != values.len() {
            return Err(anyhow!(
                "chunk outputs do not share a trailing shape: {} values cannot form {:?}",
                values.len(),
                trailing
            ));
        }

        Ok(Tensor::from_slice(&values, &trailing)?)
    }

    fn require_transport(&self, operation: &str) -> Result<()> {
        if !self.sequence_group.supports_point_to_point() {
            return Err(anyhow!(
                "`{operation}` needs a sequence process group with a real transport; use \
                 DistributedBackend::InProcess or DistributedBackend::Tcp"
            ));
        }
        Ok(())
    }

    /// Get cached attention between chunks
    fn get_cached_attention(&self, source_chunk: usize, target_chunk: usize) -> Result<Tensor> {
        let mut comm_manager = self
            .attention_comm_manager
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let cache_key = (source_chunk, target_chunk);
        if let Some(cached_attention) = comm_manager.attention_cache.get(&cache_key).cloned() {
            comm_manager.cache_hits += 1;
            Ok(cached_attention)
        } else {
            comm_manager.cache_misses += 1;
            // Create dummy attention tensor
            let attention = Tensor::zeros(&[64, 64])?;
            comm_manager.attention_cache.insert(cache_key, attention.clone());
            Ok(attention)
        }
    }

    /// Synchronize gradients across sequence chunks
    pub fn synchronize_gradients(&self, gradients: &mut HashMap<String, Tensor>) -> Result<()> {
        if !self.config.sync_gradients {
            return Ok(());
        }

        let start_time = Instant::now();

        // Convert gradients to vector for all-reduce
        let mut gradient_tensors: Vec<Tensor> = gradients.values().cloned().collect();

        // Perform all-reduce to synchronize gradients across sequence chunks
        self.sequence_group.all_reduce(&mut gradient_tensors)?;

        // Average the gradients
        let world_size = self.sequence_group.world_size() as f32;
        for tensor in &mut gradient_tensors {
            *tensor = tensor.scalar_mul(1.0 / world_size)?;
        }

        // Update the gradients map
        for (i, (_, gradient)) in gradients.iter_mut().enumerate() {
            if i < gradient_tensors.len() {
                *gradient = gradient_tensors[i].clone();
            }
        }

        // Update statistics
        {
            let mut stats =
                self.communication_stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            stats.gradient_sync_time += start_time.elapsed();
        }

        Ok(())
    }

    /// Update memory usage statistics
    pub fn update_memory_usage(&self, chunk_id: usize, memory_usage: u64) -> Result<()> {
        let mut memory_manager =
            self.memory_manager.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        memory_manager.peak_memory_per_chunk.insert(chunk_id, memory_usage);
        memory_manager.current_memory_usage = memory_usage;

        // Calculate memory pressure (simplified)
        let max_memory = 16u64 * 1024 * 1024 * 1024; // 16GB assumed max
        memory_manager.memory_pressure = memory_usage as f32 / max_memory as f32;

        Ok(())
    }

    /// Get sequence parallelism statistics
    pub fn get_statistics(&self) -> SequenceParallelismStats {
        let comm_stats =
            self.communication_stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let comm_manager = self
            .attention_comm_manager
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let memory_manager =
            self.memory_manager.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        let cache_hit_rate = if comm_manager.cache_hits + comm_manager.cache_misses > 0 {
            comm_manager.cache_hits as f32
                / (comm_manager.cache_hits + comm_manager.cache_misses) as f32
        } else {
            0.0
        };

        SequenceParallelismStats {
            total_chunks: self.sequence_chunks.len(),
            local_chunks: self.local_chunks.len(),
            communication_time: comm_stats.total_communication_time,
            attention_communication_time: comm_stats.attention_communication_time,
            gradient_sync_time: comm_stats.gradient_sync_time,
            attention_cache_hit_rate: cache_hit_rate,
            memory_pressure: memory_manager.memory_pressure,
            peak_memory_usage: memory_manager.current_memory_usage,
        }
    }

    /// Get local chunk IDs
    pub fn local_chunks(&self) -> &[usize] {
        &self.local_chunks
    }

    /// Get chunk information
    pub fn get_chunk(&self, chunk_id: usize) -> Option<&SequenceChunk> {
        self.sequence_chunks.get(chunk_id)
    }

    /// Get configuration
    pub fn config(&self) -> &SequenceParallelismConfig {
        &self.config
    }
}

/// Sequence parallelism statistics
#[derive(Debug, Clone)]
pub struct SequenceParallelismStats {
    pub total_chunks: usize,
    pub local_chunks: usize,
    pub communication_time: Duration,
    pub attention_communication_time: Duration,
    pub gradient_sync_time: Duration,
    pub attention_cache_hit_rate: f32,
    pub memory_pressure: f32,
    pub peak_memory_usage: u64,
}

/// Sequence parallelism utilities
pub mod utils {
    use super::*;

    /// Calculate optimal sequence parallelism configuration
    pub fn calculate_optimal_sequence_config(
        total_sequence_length: usize,
        max_memory_per_device: usize,
        memory_per_token: usize,
        world_size: usize,
    ) -> Result<SequenceParallelismConfig> {
        let max_tokens_per_device = max_memory_per_device / memory_per_token;

        if max_tokens_per_device == 0 {
            return Err(anyhow!("Insufficient memory for sequence parallelism"));
        }

        let required_devices = total_sequence_length.div_ceil(max_tokens_per_device);
        let sequence_parallel_size = std::cmp::min(required_devices, world_size);

        let tokens_per_device = total_sequence_length.div_ceil(sequence_parallel_size);
        let overlap_size = std::cmp::min(128, tokens_per_device / 10); // 10% overlap

        Ok(SequenceParallelismConfig {
            sequence_parallel_size,
            max_sequence_length_per_device: tokens_per_device,
            overlap_size,
            ..Default::default()
        })
    }

    /// Estimate communication cost for sequence parallelism
    pub fn estimate_communication_cost(
        config: &SequenceParallelismConfig,
        hidden_size: usize,
        num_attention_heads: usize,
    ) -> f32 {
        let overlap_tokens = config.overlap_size;
        let communication_per_overlap = overlap_tokens * hidden_size * 4; // 4 bytes per float
        let attention_communication = overlap_tokens * overlap_tokens * num_attention_heads * 4;

        (communication_per_overlap + attention_communication) as f32 / (1024.0 * 1024.0)
        // Convert to MB
    }

    /// Calculate memory savings from sequence parallelism
    pub fn calculate_memory_savings(
        total_sequence_length: usize,
        sequence_parallel_size: usize,
        hidden_size: usize,
    ) -> f32 {
        let tokens_per_device = total_sequence_length / sequence_parallel_size;
        let memory_per_device = tokens_per_device * hidden_size * 4; // 4 bytes per float
        let total_memory_without_sp = total_sequence_length * hidden_size * 4;

        1.0 - (memory_per_device as f32 / total_memory_without_sp as f32)
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests;
