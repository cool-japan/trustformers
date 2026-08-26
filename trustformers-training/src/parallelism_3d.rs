use crate::distributed::{GradientCompressionConfig, ProcessGroup};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Model;

/// 3D Parallelism Configuration
///
/// Combines data parallelism (DP), model parallelism (MP), and pipeline parallelism (PP)
/// to efficiently scale transformer training across multiple GPUs and nodes.
///
/// Key concepts:
/// - Data Parallelism: Each process has a full copy of the model and trains on different data
/// - Model Parallelism: Model parameters are split across processes within a layer
/// - Pipeline Parallelism: Model layers are split across processes, enabling pipeline execution
///
/// The total number of processes = dp_size * mp_size * pp_size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelismConfig {
    /// Data parallel group size
    pub dp_size: usize,
    /// Model parallel group size
    pub mp_size: usize,
    /// Pipeline parallel group size
    pub pp_size: usize,
    /// Number of micro-batches for pipeline parallelism
    pub num_micro_batches: usize,
    /// Whether to use gradient accumulation
    pub gradient_accumulation: bool,
    /// Number of gradient accumulation steps
    pub accumulation_steps: usize,
    /// Whether to use activation checkpointing
    pub activation_checkpointing: bool,
    /// Communication backend preference
    pub comm_backend: CommBackend,
    /// Pipeline scheduling strategy
    pub pipeline_schedule: PipelineSchedule,
    /// Memory optimization level
    pub memory_optimization: MemoryOptimization,
}

impl Default for ParallelismConfig {
    fn default() -> Self {
        Self {
            dp_size: 1,
            mp_size: 1,
            pp_size: 1,
            num_micro_batches: 4,
            gradient_accumulation: true,
            accumulation_steps: 1,
            activation_checkpointing: true,
            comm_backend: CommBackend::NCCL,
            pipeline_schedule: PipelineSchedule::GPipe,
            memory_optimization: MemoryOptimization::Medium,
        }
    }
}

/// Communication backend options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommBackend {
    NCCL,
    Gloo,
    MPI,
    InfiniBand,
}

/// Pipeline scheduling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineSchedule {
    /// Google's GPipe scheduling
    GPipe,
    /// PipeDream scheduling
    PipeDream,
    /// PipeDream-2BW (bidirectional weight updates)
    PipeDream2BW,
    /// Interleaved 1F1B (One Forward One Backward)
    Interleaved1F1B,
    /// Adaptive scheduling based on communication patterns
    Adaptive,
}

/// Memory optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryOptimization {
    None,
    Low,
    Medium,
    High,
    Extreme,
}

/// 3D Parallelism Coordinator
///
/// Manages the coordination between different parallelism strategies
/// and handles communication between process groups.
pub struct Parallelism3D {
    config: ParallelismConfig,
    global_rank: usize,
    world_size: usize,

    // Process group ranks and mappings
    dp_rank: usize,
    mp_rank: usize,
    pp_rank: usize,

    // Process groups for different parallelism types
    dp_group: Arc<dyn ProcessGroup>,
    mp_group: Arc<dyn ProcessGroup>,
    pp_group: Arc<dyn ProcessGroup>,

    // Pipeline state management
    pipeline_state: Arc<RwLock<PipelineState>>,

    // Communication statistics
    comm_stats: Arc<Mutex<CommunicationStats>>,

    // Memory management
    memory_manager: Arc<Mutex<MemoryManager>>,

    // Per-stage computation supplied by the caller (see
    // `Parallelism3D::set_stage_executor`).
    stage_executor: Arc<RwLock<Option<StageExecutor>>>,

    // Per-stage backward computation supplied by the caller (see
    // `Parallelism3D::set_stage_backward`).
    stage_backward: Arc<RwLock<Option<StageExecutor>>>,

    // Lossy codec applied by `optimize_memory` for the Medium/High levels.
    gradient_compression: GradientCompressionConfig,

    // Measurements published by `optimize_pipeline_bubbles`.
    rebalance_plan: Arc<RwLock<RebalancePlan>>,
}

/// Which way a pipeline message travels. Encoded into the message tag so that
/// a forward activation and a backward gradient for the same micro-batch can
/// never be confused for one another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineDirection {
    /// Stage `i` → stage `i + 1` (activations).
    Forward = 0,
    /// Stage `i` → stage `i - 1` (activation gradients).
    Backward = 1,
}

/// What [`Parallelism3D::optimize_pipeline_bubbles`] measured.
///
/// This type carries observations and a derived recommendation only; nothing
/// here is applied automatically, because layer placement and the micro-batch
/// count belong to the caller.
#[derive(Debug, Clone, Default)]
pub struct RebalancePlan {
    /// Stages whose measured time exceeded twice the average, with that time.
    pub bottleneck_stages: Vec<(usize, Duration)>,
    /// `pipeline_bubbles / (forward + backward passes)` at the last call.
    pub measured_bubble_ratio: f32,
    /// Micro-batch count that would bring the GPipe bubble under 20%.
    pub recommended_micro_batches: usize,
}

/// Computation executed by one pipeline stage: `(inputs, stage) -> outputs`.
///
/// 3D parallelism cannot discover a model's per-stage layers through the
/// [`Model`] trait, so the caller supplies this.
pub type StageExecutor = Box<dyn Fn(&[Tensor], usize) -> Result<Vec<Tensor>> + Send + Sync>;

/// Pipeline execution state
#[derive(Debug, Default)]
struct PipelineState {
    current_micro_batch: usize,
    forward_passes_completed: usize,
    backward_passes_completed: usize,
    pipeline_bubbles: usize,
    stage_timings: HashMap<usize, Duration>,
    communication_overhead: Duration,
}

/// Communication statistics for 3D parallelism
#[derive(Debug, Default)]
struct CommunicationStats {
    dp_all_reduce_time: Duration,
    mp_all_reduce_time: Duration,
    pp_send_recv_time: Duration,
    total_bytes_communicated: u64,
    communication_efficiency: f32,
    bandwidth_utilization: f32,
}

/// Memory management for 3D parallelism
#[derive(Debug)]
struct MemoryManager {
    activation_memory_pool: HashMap<String, Vec<Tensor>>,
    gradient_memory_pool: HashMap<String, Vec<Tensor>>,
    peak_memory_usage: u64,
    current_memory_usage: u64,
    memory_optimization_level: MemoryOptimization,
    checkpointed_activations: HashMap<String, Vec<Tensor>>,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self {
            activation_memory_pool: HashMap::new(),
            gradient_memory_pool: HashMap::new(),
            peak_memory_usage: 0,
            current_memory_usage: 0,
            memory_optimization_level: MemoryOptimization::Medium,
            checkpointed_activations: HashMap::new(),
        }
    }
}

impl Parallelism3D {
    /// Create a new 3D parallelism coordinator
    pub fn new(
        config: ParallelismConfig,
        global_rank: usize,
        world_size: usize,
        dp_group: Arc<dyn ProcessGroup>,
        mp_group: Arc<dyn ProcessGroup>,
        pp_group: Arc<dyn ProcessGroup>,
    ) -> Result<Self> {
        // Validate configuration
        if config.dp_size * config.mp_size * config.pp_size != world_size {
            return Err(anyhow!(
                "Invalid parallelism configuration: dp_size ({}) * mp_size ({}) * pp_size ({}) != world_size ({})",
                config.dp_size, config.mp_size, config.pp_size, world_size
            ));
        }

        // Calculate local ranks for each parallelism type
        let dp_rank = global_rank / (config.mp_size * config.pp_size);
        let mp_rank = (global_rank / config.pp_size) % config.mp_size;
        let pp_rank = global_rank % config.pp_size;

        let memory_manager = MemoryManager {
            memory_optimization_level: config.memory_optimization.clone(),
            ..Default::default()
        };

        Ok(Self {
            config,
            global_rank,
            world_size,
            dp_rank,
            mp_rank,
            pp_rank,
            dp_group,
            mp_group,
            pp_group,
            pipeline_state: Arc::new(RwLock::new(PipelineState::default())),
            comm_stats: Arc::new(Mutex::new(CommunicationStats::default())),
            memory_manager: Arc::new(Mutex::new(memory_manager)),
            stage_executor: Arc::new(RwLock::new(None)),
            stage_backward: Arc::new(RwLock::new(None)),
            // 10% top-k is the classic bandwidth/accuracy compromise; override
            // with `with_gradient_compression`.
            gradient_compression: GradientCompressionConfig::default_when_enabled(),
            rebalance_plan: Arc::new(RwLock::new(RebalancePlan::default())),
        })
    }

    /// Register the computation each pipeline stage performs.
    ///
    /// The closure receives the stage's inputs and its index and returns the
    /// stage's outputs. Without it, [`Parallelism3D::forward_pass`] fails with
    /// an explicit error rather than passing its inputs through unchanged.
    pub fn set_stage_executor(&self, executor: StageExecutor) {
        let mut slot = self.stage_executor.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        *slot = Some(executor);
    }

    /// Whether a stage executor has been registered.
    pub fn has_stage_executor(&self) -> bool {
        self.stage_executor
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_some()
    }

    /// Register the backward computation each pipeline stage performs.
    ///
    /// The closure receives the gradient of this stage's *outputs* and the
    /// stage index, and returns the gradient of this stage's *inputs* — which
    /// is what gets sent to the preceding stage. Without it,
    /// [`Parallelism3D::backward_pass`] fails with an explicit error rather
    /// than echoing its input.
    pub fn set_stage_backward(&self, executor: StageExecutor) {
        let mut slot = self.stage_backward.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        *slot = Some(executor);
    }

    /// Whether a backward stage executor has been registered.
    pub fn has_stage_backward(&self) -> bool {
        self.stage_backward
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_some()
    }

    /// Choose the lossy codec [`Parallelism3D::optimize_memory`] applies at the
    /// [`MemoryOptimization::Medium`] and [`MemoryOptimization::High`] levels.
    pub fn with_gradient_compression(mut self, codec: GradientCompressionConfig) -> Self {
        self.gradient_compression = codec;
        self
    }

    /// The active gradient compression codec.
    pub fn gradient_compression(&self) -> GradientCompressionConfig {
        self.gradient_compression
    }

    /// The measurements and recommendation published by the last
    /// [`Parallelism3D::optimize_pipeline_bubbles`] call.
    pub fn rebalance_plan(&self) -> RebalancePlan {
        self.rebalance_plan
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Execute forward pass with 3D parallelism
    pub fn forward_pass<M: Model>(
        &self,
        model: &M,
        inputs: &[Tensor],
        micro_batch_id: usize,
    ) -> Result<Vec<Tensor>> {
        let _start_time = Instant::now();

        // Handle different pipeline scheduling strategies
        match self.config.pipeline_schedule {
            PipelineSchedule::GPipe => self.forward_gpipe(model, inputs, micro_batch_id),
            PipelineSchedule::PipeDream => self.forward_pipedream(model, inputs, micro_batch_id),
            PipelineSchedule::PipeDream2BW => {
                self.forward_pipedream_2bw(model, inputs, micro_batch_id)
            },
            PipelineSchedule::Interleaved1F1B => {
                self.forward_interleaved_1f1b(model, inputs, micro_batch_id)
            },
            PipelineSchedule::Adaptive => self.forward_adaptive(model, inputs, micro_batch_id),
        }
    }

    /// Execute backward pass with 3D parallelism
    pub fn backward_pass<M: Model>(
        &self,
        model: &mut M,
        gradients: &[Tensor],
        micro_batch_id: usize,
    ) -> Result<Vec<Tensor>> {
        let _start_time = Instant::now();

        // Handle different pipeline scheduling strategies for backward pass
        match self.config.pipeline_schedule {
            PipelineSchedule::GPipe => self.backward_gpipe(model, gradients, micro_batch_id),
            PipelineSchedule::PipeDream => {
                self.backward_pipedream(model, gradients, micro_batch_id)
            },
            PipelineSchedule::PipeDream2BW => {
                self.backward_pipedream_2bw(model, gradients, micro_batch_id)
            },
            PipelineSchedule::Interleaved1F1B => {
                self.backward_interleaved_1f1b(model, gradients, micro_batch_id)
            },
            PipelineSchedule::Adaptive => self.backward_adaptive(model, gradients, micro_batch_id),
        }
    }

    /// Synchronize gradients across all parallelism dimensions
    pub fn synchronize_gradients(&self, gradients: &mut [Tensor]) -> Result<()> {
        let start_time = Instant::now();

        // Step 1: Reduce-scatter within model parallel group
        let shapes = if self.config.mp_size > 1 {
            Some(self.mp_reduce_scatter_gradients(gradients)?)
        } else {
            None
        };

        // Step 2: All-reduce within data parallel group
        if self.config.dp_size > 1 {
            self.dp_all_reduce_gradients(gradients)?;
        }

        // Step 3: All-gather within model parallel group
        if let Some(shapes) = shapes {
            self.mp_all_gather_gradients(gradients, &shapes)?;
        }

        // Update communication statistics
        let mut stats = self.comm_stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        stats.dp_all_reduce_time += start_time.elapsed();

        Ok(())
    }

    /// Optimize memory usage based on configuration
    pub fn optimize_memory(&self, tensors: &mut [Tensor]) -> Result<()> {
        // Read the level and release the lock: every handler below takes the
        // same `Mutex`, and `std::sync::Mutex` is not reentrant — holding it
        // across the dispatch deadlocks the caller.
        let level = {
            let memory_manager =
                self.memory_manager.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            memory_manager.memory_optimization_level.clone()
        };

        match level {
            MemoryOptimization::None => {
                // No optimization
                Ok(())
            },
            MemoryOptimization::Low => {
                // Basic activation checkpointing
                self.apply_activation_checkpointing(tensors, 4)
            },
            MemoryOptimization::Medium => {
                // Activation checkpointing + gradient compression
                self.apply_activation_checkpointing(tensors, 2)?;
                self.apply_gradient_compression(tensors)
            },
            MemoryOptimization::High => {
                // All optimizations + CPU offloading
                self.apply_activation_checkpointing(tensors, 1)?;
                self.apply_gradient_compression(tensors)?;
                self.apply_cpu_offloading(tensors)
            },
            MemoryOptimization::Extreme => {
                // ZeRO-style optimization
                self.apply_zero_optimization(tensors)
            },
        }
    }

    /// Measure the pipeline's bubble behaviour and publish a rebalancing plan.
    ///
    /// See [`Parallelism3D::rebalance_plan`] for what is produced. The plan is
    /// a measurement plus a recommendation; nothing is silently mutated.
    pub fn optimize_pipeline_bubbles(&self) -> Result<()> {
        // The helpers below take the same `RwLock`; `std::sync::RwLock` gives no
        // reentrancy guarantee, so the guard is scoped and dropped first.
        let (bottleneck_stages, passes, bubbles) = {
            let state = self.pipeline_state.read().unwrap_or_else(|poisoned| poisoned.into_inner());

            let total_stages = self.config.pp_size.max(1);
            let avg_stage_time =
                state.stage_timings.values().sum::<Duration>() / total_stages as u32;

            let mut bottleneck_stages = Vec::new();
            for (stage, timing) in &state.stage_timings {
                if *timing > avg_stage_time * 2 {
                    bottleneck_stages.push(*stage);
                }
            }
            bottleneck_stages.sort_unstable();

            (
                bottleneck_stages,
                state.forward_passes_completed + state.backward_passes_completed,
                state.pipeline_bubbles,
            )
        };

        if !bottleneck_stages.is_empty() {
            self.apply_load_balancing(&bottleneck_stages)?;
        }

        // No passes yet means there is nothing to measure; reporting a
        // efficiency of 1.0 (or NaN) would be an invented number.
        if passes == 0 {
            return Ok(());
        }

        let pipeline_efficiency = 1.0 - (bubbles as f32 / passes as f32);
        if pipeline_efficiency < 0.8 {
            self.adjust_micro_batch_size()?;
        }

        Ok(())
    }

    /// Get comprehensive 3D parallelism statistics
    pub fn get_statistics(&self) -> Result<Parallelism3DStats> {
        let pipeline_state =
            self.pipeline_state.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        let comm_stats = self.comm_stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let memory_manager =
            self.memory_manager.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        Ok(Parallelism3DStats {
            dp_rank: self.dp_rank,
            mp_rank: self.mp_rank,
            pp_rank: self.pp_rank,
            pipeline_efficiency: self.calculate_pipeline_efficiency(&pipeline_state),
            communication_efficiency: comm_stats.communication_efficiency,
            memory_efficiency: self.calculate_memory_efficiency(&memory_manager),
            total_communication_time: comm_stats.dp_all_reduce_time
                + comm_stats.mp_all_reduce_time
                + comm_stats.pp_send_recv_time,
            peak_memory_usage: memory_manager.peak_memory_usage,
            pipeline_bubbles: pipeline_state.pipeline_bubbles,
            micro_batches_processed: pipeline_state.forward_passes_completed,
        })
    }

    // Private implementation methods for different pipeline strategies

    fn forward_gpipe<M: Model>(
        &self,
        model: &M,
        inputs: &[Tensor],
        micro_batch_id: usize,
    ) -> Result<Vec<Tensor>> {
        // GPipe: Sequential forward passes, then sequential backward passes

        if self.pp_rank == 0 {
            // First stage: process input
            let outputs = self.process_pipeline_stage(model, inputs, 0)?;

            // Send to next stage
            if self.config.pp_size > 1 {
                self.send_to_next_stage(&outputs, micro_batch_id)?;
            }

            Ok(outputs)
        } else {
            // Intermediate/final stages: receive from previous, process, send to next
            let received_inputs = self.receive_from_previous_stage(micro_batch_id)?;
            let outputs = self.process_pipeline_stage(model, &received_inputs, self.pp_rank)?;

            if self.pp_rank < self.config.pp_size - 1 {
                self.send_to_next_stage(&outputs, micro_batch_id)?;
            }

            Ok(outputs)
        }
    }

    /// PipeDream forward pass.
    ///
    /// The per-micro-batch *data flow* is identical to GPipe — the same
    /// activations move between the same stages — so this runs the GPipe path.
    /// What PipeDream adds is asynchronous 1F1B scheduling with weight
    /// stashing, which is a property of the driver loop rather than of a single
    /// micro-batch, and which this coordinator does not implement. Timing
    /// therefore matches GPipe; correctness does not differ.
    fn forward_pipedream<M: Model>(
        &self,
        model: &M,
        inputs: &[Tensor],
        micro_batch_id: usize,
    ) -> Result<Vec<Tensor>> {
        self.forward_gpipe(model, inputs, micro_batch_id)
    }

    /// PipeDream-2BW forward pass; see [`Self::forward_pipedream`] for the
    /// schedule caveat.
    fn forward_pipedream_2bw<M: Model>(
        &self,
        model: &M,
        inputs: &[Tensor],
        micro_batch_id: usize,
    ) -> Result<Vec<Tensor>> {
        self.forward_gpipe(model, inputs, micro_batch_id)
    }

    /// Interleaved-1F1B forward pass; see [`Self::forward_pipedream`] for the
    /// schedule caveat.
    fn forward_interleaved_1f1b<M: Model>(
        &self,
        model: &M,
        inputs: &[Tensor],
        micro_batch_id: usize,
    ) -> Result<Vec<Tensor>> {
        self.forward_gpipe(model, inputs, micro_batch_id)
    }

    fn forward_adaptive<M: Model>(
        &self,
        model: &M,
        inputs: &[Tensor],
        micro_batch_id: usize,
    ) -> Result<Vec<Tensor>> {
        // Adaptive scheduling based on runtime characteristics

        // Choose strategy based on current performance metrics
        let stats = self.comm_stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let communication_time_ratio = stats.pp_send_recv_time.as_millis() as f32
            / (stats.dp_all_reduce_time.as_millis() + stats.mp_all_reduce_time.as_millis() + 1)
                as f32;

        if communication_time_ratio > 2.0 {
            // High communication overhead, use GPipe
            self.forward_gpipe(model, inputs, micro_batch_id)
        } else {
            // Low communication overhead, use interleaved
            self.forward_interleaved_1f1b(model, inputs, micro_batch_id)
        }
    }

    /// GPipe backward pass: gradients flow from the last stage to the first.
    ///
    /// The last stage differentiates the incoming loss gradient; every other
    /// stage first receives the activation gradient produced by the stage after
    /// it. Each stage runs the registered backward executor (see
    /// [`Parallelism3D::set_stage_backward`]) and forwards the result to the
    /// stage before it.
    fn backward_gpipe<M: Model>(
        &self,
        _model: &mut M,
        gradients: &[Tensor],
        micro_batch_id: usize,
    ) -> Result<Vec<Tensor>> {
        let last_stage = self.config.pp_size.saturating_sub(1);

        let incoming = if self.pp_rank == last_stage {
            gradients.to_vec()
        } else {
            self.receive_from_next_stage(micro_batch_id)?
        };

        let outgoing = self.run_stage_backward(&incoming, self.pp_rank)?;

        if self.pp_rank > 0 {
            self.send_to_previous_stage(&outgoing, micro_batch_id)?;
        }

        Ok(outgoing)
    }

    /// PipeDream backward pass.
    ///
    /// The gradient plumbing is identical to GPipe; what PipeDream changes is
    /// the *schedule* (asynchronous 1F1B with weight stashing), which this
    /// coordinator does not implement. Selecting it therefore runs the GPipe
    /// data flow — correct gradients, no schedule overlap — and the difference
    /// is documented rather than silently claimed.
    fn backward_pipedream<M: Model>(
        &self,
        model: &mut M,
        gradients: &[Tensor],
        micro_batch_id: usize,
    ) -> Result<Vec<Tensor>> {
        self.backward_gpipe(model, gradients, micro_batch_id)
    }

    /// PipeDream-2BW backward pass; see [`Self::backward_pipedream`] for the
    /// schedule caveat.
    fn backward_pipedream_2bw<M: Model>(
        &self,
        model: &mut M,
        gradients: &[Tensor],
        micro_batch_id: usize,
    ) -> Result<Vec<Tensor>> {
        self.backward_gpipe(model, gradients, micro_batch_id)
    }

    /// Interleaved-1F1B backward pass; see [`Self::backward_pipedream`] for the
    /// schedule caveat.
    fn backward_interleaved_1f1b<M: Model>(
        &self,
        model: &mut M,
        gradients: &[Tensor],
        micro_batch_id: usize,
    ) -> Result<Vec<Tensor>> {
        self.backward_gpipe(model, gradients, micro_batch_id)
    }

    /// Adaptive backward pass; see [`Self::backward_pipedream`] for the
    /// schedule caveat.
    fn backward_adaptive<M: Model>(
        &self,
        model: &mut M,
        gradients: &[Tensor],
        micro_batch_id: usize,
    ) -> Result<Vec<Tensor>> {
        self.backward_gpipe(model, gradients, micro_batch_id)
    }

    /// Run this stage's registered backward computation.
    ///
    /// # Errors
    ///
    /// When no backward executor has been registered. Echoing the incoming
    /// gradient back would report a completed backward pass that never ran.
    fn run_stage_backward(&self, gradients: &[Tensor], stage: usize) -> Result<Vec<Tensor>> {
        let executor = self.stage_backward.read().unwrap_or_else(|poisoned| poisoned.into_inner());

        match executor.as_ref() {
            Some(executor) => executor(gradients, stage),
            None => Err(anyhow!(
                "no backward executor registered for pipeline stage {stage}: the Model trait \
                 exposes no per-layer autodiff hook, so 3D parallelism cannot differentiate the \
                 stage on its own. Register it with Parallelism3D::set_stage_backward"
            )),
        }
    }

    // Communication methods

    /// Reduce-scatter every gradient within the model-parallel group.
    ///
    /// Each MP rank ends up owning a contiguous slice of every gradient, which
    /// is what makes the subsequent data-parallel all-reduce cheaper by a factor
    /// of `mp_size`. The original shapes are returned so
    /// [`Self::mp_all_gather_gradients`] can restore them.
    fn mp_reduce_scatter_gradients(&self, gradients: &mut [Tensor]) -> Result<Vec<Vec<usize>>> {
        let mut shapes = Vec::with_capacity(gradients.len());
        for tensor in gradients.iter_mut() {
            shapes.push(tensor.shape());
            *tensor = self.mp_group.reduce_scatter(tensor)?;
        }
        Ok(shapes)
    }

    /// All-reduce and average the gradients within the data-parallel group.
    fn dp_all_reduce_gradients(&self, gradients: &mut [Tensor]) -> Result<()> {
        self.dp_group.all_reduce(gradients)?;

        // `all_reduce` sums; data parallelism needs the mean.
        let dp_size = self.dp_group.world_size().max(1);
        if dp_size > 1 {
            let scale = 1.0 / dp_size as f32;
            for tensor in gradients.iter_mut() {
                *tensor = tensor.scalar_mul(scale)?;
            }
        }

        Ok(())
    }

    /// All-gather the scattered gradient slices back into full tensors.
    fn mp_all_gather_gradients(
        &self,
        gradients: &mut [Tensor],
        shapes: &[Vec<usize>],
    ) -> Result<()> {
        for (index, tensor) in gradients.iter_mut().enumerate() {
            let shards = self.mp_group.all_gather(tensor)?;
            let mut values = Vec::new();
            for shard in &shards {
                values.extend(shard.to_vec_f32()?);
            }

            let shape = shapes.get(index).cloned().unwrap_or_else(|| vec![values.len()]);
            let expected: usize = shape.iter().product();
            if values.len() != expected {
                return Err(anyhow!(
                    "model-parallel all-gather reassembled {} elements but the original gradient \
                     had shape {:?} ({} elements)",
                    values.len(),
                    shape,
                    expected
                ));
            }
            *tensor = Tensor::from_slice(&values, &shape)?;
        }
        Ok(())
    }

    /// Message tag for one pipeline transfer.
    ///
    /// Both endpoints derive the same value from the micro-batch id, the
    /// direction and the message role, so no call-ordering convention is needed
    /// between stages that execute different code paths.
    fn pipeline_tag(micro_batch_id: usize, direction: PipelineDirection, part: u64) -> u64 {
        (micro_batch_id as u64) << 9 | (direction as u64) << 8 | part
    }

    /// Largest number of tensors that fit in one pipeline message.
    ///
    /// Parts 0 and 1 carry the header and manifest, and the part index must stay
    /// inside the 8 bits reserved for it in [`Self::pipeline_tag`].
    const MAX_PIPELINE_TENSORS: usize = 254;

    /// Send `tensors` to `peer`, labelled with `direction`.
    ///
    /// Wire protocol, in order:
    /// 1. a one-element header holding the manifest length,
    /// 2. the manifest `[count, ndim_0, dims_0…, ndim_1, dims_1…, …]`,
    /// 3. each tensor flattened to `[n]`.
    ///
    /// Shapes travel with the data, so the receiver reconstructs exactly what
    /// was sent.
    fn send_stage_tensors(
        &self,
        peer: usize,
        tensors: &[Tensor],
        micro_batch_id: usize,
        direction: PipelineDirection,
    ) -> Result<()> {
        if tensors.len() > Self::MAX_PIPELINE_TENSORS {
            return Err(anyhow!(
                "a pipeline message carries at most {} tensors, got {}",
                Self::MAX_PIPELINE_TENSORS,
                tensors.len()
            ));
        }

        let mut manifest: Vec<f32> = vec![tensors.len() as f32];
        for tensor in tensors {
            let shape = tensor.shape();
            manifest.push(shape.len() as f32);
            manifest.extend(shape.iter().map(|dim| *dim as f32));
        }

        let header = Tensor::from_slice(&[manifest.len() as f32], &[1])?;
        self.pp_group.send(
            peer,
            Self::pipeline_tag(micro_batch_id, direction, 0),
            &header,
        )?;

        let manifest_len = manifest.len();
        let manifest_tensor = Tensor::from_slice(&manifest, &[manifest_len])?;
        self.pp_group.send(
            peer,
            Self::pipeline_tag(micro_batch_id, direction, 1),
            &manifest_tensor,
        )?;

        for (index, tensor) in tensors.iter().enumerate() {
            let values = tensor.to_vec_f32()?;
            let length = values.len();
            let flat = Tensor::from_slice(&values, &[length])?;
            self.pp_group.send(
                peer,
                Self::pipeline_tag(micro_batch_id, direction, 2 + index as u64),
                &flat,
            )?;
        }

        {
            let mut stats = self.comm_stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            stats.total_bytes_communicated += tensors
                .iter()
                .map(|tensor| (tensor.len() * std::mem::size_of::<f32>()) as u64)
                .sum::<u64>();
        }
        Ok(())
    }

    /// Send `tensors` to the next pipeline stage (forward direction).
    fn send_to_next_stage(&self, tensors: &[Tensor], micro_batch_id: usize) -> Result<()> {
        let next_rank = self.pp_rank + 1;
        if next_rank >= self.config.pp_size {
            return Err(anyhow!(
                "pipeline stage {} is the last stage; there is no next stage to send to",
                self.pp_rank
            ));
        }
        self.require_pipeline_transport("send_to_next_stage")?;
        self.send_stage_tensors(
            next_rank,
            tensors,
            micro_batch_id,
            PipelineDirection::Forward,
        )
    }

    /// Send activation gradients to the previous pipeline stage (backward
    /// direction).
    fn send_to_previous_stage(&self, tensors: &[Tensor], micro_batch_id: usize) -> Result<()> {
        if self.pp_rank == 0 {
            return Err(anyhow!(
                "pipeline stage 0 is the first stage; there is no previous stage to send to"
            ));
        }
        self.require_pipeline_transport("send_to_previous_stage")?;
        self.send_stage_tensors(
            self.pp_rank - 1,
            tensors,
            micro_batch_id,
            PipelineDirection::Backward,
        )
    }

    /// Receive the tensors sent by the previous pipeline stage, restoring their
    /// original shapes.
    fn receive_from_previous_stage(&self, micro_batch_id: usize) -> Result<Vec<Tensor>> {
        if self.pp_rank == 0 {
            return Err(anyhow!(
                "pipeline stage 0 has no previous stage to receive from"
            ));
        }
        self.require_pipeline_transport("receive_from_previous_stage")?;
        self.recv_stage_tensors(self.pp_rank - 1, micro_batch_id, PipelineDirection::Forward)
    }

    /// Receive the activation gradients sent by the next pipeline stage.
    fn receive_from_next_stage(&self, micro_batch_id: usize) -> Result<Vec<Tensor>> {
        let next_rank = self.pp_rank + 1;
        if next_rank >= self.config.pp_size {
            return Err(anyhow!(
                "pipeline stage {} is the last stage; there is no next stage to receive from",
                self.pp_rank
            ));
        }
        self.require_pipeline_transport("receive_from_next_stage")?;
        self.recv_stage_tensors(next_rank, micro_batch_id, PipelineDirection::Backward)
    }

    /// Receive one pipeline message from `peer`.
    fn recv_stage_tensors(
        &self,
        previous_rank: usize,
        micro_batch_id: usize,
        direction: PipelineDirection,
    ) -> Result<Vec<Tensor>> {
        let header = self
            .pp_group
            .recv(
                previous_rank,
                Self::pipeline_tag(micro_batch_id, direction, 0),
                &[1],
            )?
            .to_vec_f32()?;
        let manifest_len = *header
            .first()
            .ok_or_else(|| anyhow!("pipeline header from stage {previous_rank} was empty"))?
            as usize;

        let manifest = self
            .pp_group
            .recv(
                previous_rank,
                Self::pipeline_tag(micro_batch_id, direction, 1),
                &[manifest_len],
            )?
            .to_vec_f32()?;

        let mut cursor = manifest.iter().copied();
        let count = cursor.next().ok_or_else(|| anyhow!("pipeline manifest was empty"))? as usize;

        let mut shapes = Vec::with_capacity(count);
        for _ in 0..count {
            let ndim = cursor
                .next()
                .ok_or_else(|| anyhow!("pipeline manifest ended before the rank count"))?
                as usize;
            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                shape.push(
                    cursor
                        .next()
                        .ok_or_else(|| anyhow!("pipeline manifest ended before the dimensions"))?
                        as usize,
                );
            }
            shapes.push(shape);
        }

        let mut tensors = Vec::with_capacity(count);
        for (index, shape) in shapes.into_iter().enumerate() {
            let elements: usize = shape.iter().product();
            let flat = self.pp_group.recv(
                previous_rank,
                Self::pipeline_tag(micro_batch_id, direction, 2 + index as u64),
                &[elements],
            )?;
            tensors.push(Tensor::from_slice(&flat.to_vec_f32()?, &shape)?);
        }

        Ok(tensors)
    }

    fn require_pipeline_transport(&self, operation: &str) -> Result<()> {
        if self.config.pp_size > 1 && !self.pp_group.supports_point_to_point() {
            return Err(anyhow!(
                "`{operation}` needs a pipeline process group with a real transport; use \
                 DistributedBackend::InProcess or DistributedBackend::Tcp"
            ));
        }
        Ok(())
    }

    /// Run one pipeline stage's computation.
    ///
    /// The [`Model`] trait exposes no per-layer API, so a stage's subset of
    /// layers cannot be discovered generically. Callers therefore register the
    /// per-stage computation with
    /// [`Parallelism3D::set_stage_executor`]; without one this returns an error
    /// rather than echoing its input and pretending the stage ran.
    fn process_pipeline_stage<M: Model>(
        &self,
        _model: &M,
        inputs: &[Tensor],
        stage: usize,
    ) -> Result<Vec<Tensor>> {
        let executor = self.stage_executor.read().unwrap_or_else(|poisoned| poisoned.into_inner());

        match executor.as_ref() {
            Some(executor) => executor(inputs, stage),
            None => Err(anyhow!(
                "no stage executor registered for pipeline stage {stage}: the Model trait exposes \
                 no per-layer API, so 3D parallelism cannot slice the model on its own. Register \
                 the per-stage computation with Parallelism3D::set_stage_executor"
            )),
        }
    }

    // Memory optimization methods
    fn apply_activation_checkpointing(
        &self,
        tensors: &mut [Tensor],
        checkpoint_ratio: usize,
    ) -> Result<()> {
        let mut memory_manager =
            self.memory_manager.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        // Save every Nth activation for checkpointing
        for (i, tensor) in tensors.iter().enumerate() {
            if i % checkpoint_ratio == 0 {
                memory_manager
                    .checkpointed_activations
                    .entry(format!("checkpoint_{}", i))
                    .or_default()
                    .push(tensor.clone());
            }
        }

        Ok(())
    }

    /// Apply the configured lossy gradient codec in place.
    ///
    /// The codec is [`GradientCompressionConfig`] from the distributed layer,
    /// so the arithmetic is shared with `DataParallelTrainer` rather than
    /// re-invented here. The values really change: top-k zeroes the small
    /// entries, quantization snaps to the reconstruction grid.
    fn apply_gradient_compression(&self, tensors: &mut [Tensor]) -> Result<()> {
        let codec = self.gradient_compression;
        if codec == GradientCompressionConfig::None {
            return Ok(());
        }
        for tensor in tensors.iter_mut() {
            let shape = tensor.shape();
            let mut values = tensor.to_vec_f32()?;
            codec.apply(&mut values)?;
            *tensor = Tensor::from_slice(&values, &shape)?;
        }
        Ok(())
    }

    /// CPU offloading.
    ///
    /// Tensors in the pure-Rust build already live in host memory — there is no
    /// device buffer to evict — so this is a genuine no-op and is documented as
    /// one rather than reported as a saving. It exists so the
    /// [`MemoryOptimization::High`] path stays explicit about what it does and
    /// does not do.
    fn apply_cpu_offloading(&self, _tensors: &mut [Tensor]) -> Result<()> {
        Ok(())
    }

    /// ZeRO-style optimizer-state partitioning.
    ///
    /// # Errors
    ///
    /// Always. ZeRO partitions *optimizer state*, which this type does not own:
    /// it never sees the optimizer. Silently returning `Ok(())` from
    /// [`MemoryOptimization::Extreme`] would report a memory saving that never
    /// happened, so the caller is pointed at the type that implements it.
    fn apply_zero_optimization(&self, _tensors: &mut [Tensor]) -> Result<()> {
        Err(anyhow!(
            "MemoryOptimization::Extreme requests ZeRO optimizer-state partitioning, which \
             Parallelism3D cannot perform: it holds no optimizer. Wrap your optimizer in \
             crate::distributed_zero::ZeroStage1Optimizer over the data-parallel process group, \
             and select a different MemoryOptimization level here"
        ))
    }

    /// Record a rebalancing plan for the stages that are running long.
    ///
    /// Layer-to-stage assignment lives in the caller's stage executor (see
    /// [`Parallelism3D::set_stage_executor`]), so this type cannot move layers
    /// itself. What it *can* do honestly is publish the measurement: the
    /// bottleneck stages and their observed times, retrievable with
    /// [`Parallelism3D::rebalance_plan`].
    fn apply_load_balancing(&self, bottleneck_stages: &[usize]) -> Result<()> {
        let timings = {
            let state = self.pipeline_state.read().unwrap_or_else(|poisoned| poisoned.into_inner());
            bottleneck_stages
                .iter()
                .map(|stage| {
                    (
                        *stage,
                        state.stage_timings.get(stage).copied().unwrap_or_default(),
                    )
                })
                .collect::<Vec<_>>()
        };

        let mut plan = self.rebalance_plan.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        plan.bottleneck_stages = timings;
        Ok(())
    }

    /// Recompute the recommended micro-batch count from the measured bubble
    /// ratio and publish it in the rebalance plan.
    ///
    /// More micro-batches shrink the pipeline bubble (`(pp_size - 1) / (m +
    /// pp_size - 1)` for GPipe), so the recommendation grows with the observed
    /// bubble fraction. The value is a recommendation, not an applied change:
    /// the micro-batch count is part of [`ParallelismConfig`] and belongs to
    /// whoever constructed this coordinator.
    fn adjust_micro_batch_size(&self) -> Result<()> {
        let (bubbles, passes) = {
            let state = self.pipeline_state.read().unwrap_or_else(|poisoned| poisoned.into_inner());
            (
                state.pipeline_bubbles,
                state.forward_passes_completed + state.backward_passes_completed,
            )
        };
        if passes == 0 {
            return Ok(());
        }

        let bubble_ratio = (bubbles as f32 / passes as f32).clamp(0.0, 0.99);
        // Target: keep the bubble fraction under 20%. GPipe's bubble is
        // (p - 1) / (m + p - 1), so the m that hits a target t is
        // m = (p - 1) * (1 - t) / t.
        let stages = self.config.pp_size.max(1) as f32;
        let target = 0.2f32;
        let recommended = if bubble_ratio > target {
            (((stages - 1.0) * (1.0 - target) / target).ceil() as usize)
                .max(self.config.num_micro_batches)
        } else {
            self.config.num_micro_batches
        };

        let mut plan = self.rebalance_plan.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        plan.measured_bubble_ratio = bubble_ratio;
        plan.recommended_micro_batches = recommended;
        Ok(())
    }

    // Statistics calculation methods
    fn calculate_pipeline_efficiency(&self, state: &PipelineState) -> f32 {
        if state.forward_passes_completed == 0 {
            return 0.0;
        }

        let total_passes = state.forward_passes_completed + state.backward_passes_completed;
        1.0 - (state.pipeline_bubbles as f32 / total_passes as f32)
    }

    fn calculate_memory_efficiency(&self, memory_manager: &MemoryManager) -> f32 {
        if memory_manager.peak_memory_usage == 0 {
            return 1.0;
        }

        memory_manager.current_memory_usage as f32 / memory_manager.peak_memory_usage as f32
    }
}

/// Comprehensive statistics for 3D parallelism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parallelism3DStats {
    pub dp_rank: usize,
    pub mp_rank: usize,
    pub pp_rank: usize,
    pub pipeline_efficiency: f32,
    pub communication_efficiency: f32,
    pub memory_efficiency: f32,
    pub total_communication_time: Duration,
    pub peak_memory_usage: u64,
    pub pipeline_bubbles: usize,
    pub micro_batches_processed: usize,
}

/// Manager for coordinating 3D parallelism across training
pub struct Parallelism3DManager {
    coordinators: HashMap<String, Arc<Parallelism3D>>,
    global_config: ParallelismConfig,
    performance_tracker: Arc<Mutex<PerformanceTracker>>,
}

#[derive(Debug, Default)]
struct PerformanceTracker {
    iteration_times: Vec<Duration>,
    communication_times: Vec<Duration>,
    memory_usage_samples: Vec<u64>,
    efficiency_scores: Vec<f32>,
}

impl Parallelism3DManager {
    /// Create a new 3D parallelism manager
    pub fn new(config: ParallelismConfig) -> Self {
        Self {
            coordinators: HashMap::new(),
            global_config: config,
            performance_tracker: Arc::new(Mutex::new(PerformanceTracker::default())),
        }
    }

    /// Register a new 3D parallelism coordinator
    pub fn register_coordinator(
        &mut self,
        name: String,
        coordinator: Arc<Parallelism3D>,
    ) -> Result<()> {
        self.coordinators.insert(name, coordinator);
        Ok(())
    }

    /// Get aggregate statistics across all coordinators
    pub fn get_aggregate_stats(&self) -> Result<AggregateParallelismStats> {
        let mut aggregate = AggregateParallelismStats::default();

        for coordinator in self.coordinators.values() {
            let stats = coordinator.get_statistics()?;
            aggregate.total_pipeline_efficiency += stats.pipeline_efficiency;
            aggregate.total_communication_time += stats.total_communication_time;
            aggregate.total_memory_usage += stats.peak_memory_usage;
            aggregate.total_micro_batches += stats.micro_batches_processed;
        }

        if !self.coordinators.is_empty() {
            aggregate.average_pipeline_efficiency =
                aggregate.total_pipeline_efficiency / self.coordinators.len() as f32;
        }

        aggregate.num_coordinators = self.coordinators.len();

        Ok(aggregate)
    }

    /// Optimize configuration based on performance metrics
    pub fn optimize_configuration(&mut self) -> Result<ParallelismConfig> {
        let tracker =
            self.performance_tracker.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        if tracker.efficiency_scores.is_empty() {
            return Ok(self.global_config.clone());
        }

        let avg_efficiency =
            tracker.efficiency_scores.iter().sum::<f32>() / tracker.efficiency_scores.len() as f32;
        let mut optimized_config = self.global_config.clone();

        // Adjust micro-batch size based on efficiency
        if avg_efficiency < 0.8 {
            optimized_config.num_micro_batches = (optimized_config.num_micro_batches * 2).min(16);
        } else if avg_efficiency > 0.95 {
            optimized_config.num_micro_batches = (optimized_config.num_micro_batches / 2).max(1);
        }

        // Adjust memory optimization based on usage patterns
        let avg_memory_usage = tracker.memory_usage_samples.iter().sum::<u64>()
            / tracker.memory_usage_samples.len() as u64;
        if avg_memory_usage > (0.9 * (32u64 * 1024 * 1024 * 1024) as f64) as u64 {
            // 32GB threshold
            optimized_config.memory_optimization = MemoryOptimization::High;
        }

        Ok(optimized_config)
    }
}

/// Aggregate statistics across multiple 3D parallelism coordinators
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AggregateParallelismStats {
    pub num_coordinators: usize,
    pub total_pipeline_efficiency: f32,
    pub average_pipeline_efficiency: f32,
    pub total_communication_time: Duration,
    pub total_memory_usage: u64,
    pub total_micro_batches: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::SimulatedProcessGroup;

    #[test]
    fn test_parallelism_config_validation() {
        let config = ParallelismConfig {
            dp_size: 2,
            mp_size: 2,
            pp_size: 2,
            ..Default::default()
        };

        let world_size = 8; // 2 * 2 * 2
        assert_eq!(config.dp_size * config.mp_size * config.pp_size, world_size);
    }

    #[test]
    fn test_rank_calculation() {
        let config = ParallelismConfig {
            dp_size: 2,
            mp_size: 2,
            pp_size: 2,
            ..Default::default()
        };

        let global_rank = 5;
        let _world_size = 8;

        let dp_rank = global_rank / (config.mp_size * config.pp_size);
        let mp_rank = (global_rank / config.pp_size) % config.mp_size;
        let pp_rank = global_rank % config.pp_size;

        assert_eq!(dp_rank, 1);
        assert_eq!(mp_rank, 0);
        assert_eq!(pp_rank, 1);
    }

    #[test]
    fn test_3d_parallelism_creation() {
        let config = ParallelismConfig {
            dp_size: 2,
            mp_size: 1,
            pp_size: 1,
            ..Default::default()
        };

        let dp_group = Arc::new(SimulatedProcessGroup::new(0, 2));
        let mp_group = Arc::new(SimulatedProcessGroup::new(0, 1));
        let pp_group = Arc::new(SimulatedProcessGroup::new(0, 1));

        let parallelism = Parallelism3D::new(config, 0, 2, dp_group, mp_group, pp_group);

        assert!(parallelism.is_ok());
        let p = parallelism.expect("operation failed in test");
        assert_eq!(p.dp_rank, 0);
        assert_eq!(p.mp_rank, 0);
        assert_eq!(p.pp_rank, 0);
    }

    #[test]
    fn test_memory_optimization_levels() {
        use MemoryOptimization::*;

        let levels = vec![None, Low, Medium, High, Extreme];

        for level in levels {
            let config = ParallelismConfig {
                memory_optimization: level,
                ..Default::default()
            };

            // Should be able to serialize/deserialize
            let json = serde_json::to_string(&config).expect("JSON serialization failed");
            let deserialized: ParallelismConfig =
                serde_json::from_str(&json).expect("JSON deserialization failed");

            assert!(matches!(deserialized.memory_optimization, _));
        }
    }

    #[test]
    fn test_pipeline_schedule_types() {
        use PipelineSchedule::*;

        let schedules = vec![GPipe, PipeDream, PipeDream2BW, Interleaved1F1B, Adaptive];

        for schedule in schedules {
            let config = ParallelismConfig {
                pipeline_schedule: schedule,
                ..Default::default()
            };

            // Should be able to serialize/deserialize
            let json = serde_json::to_string(&config).expect("JSON serialization failed");
            let deserialized: ParallelismConfig =
                serde_json::from_str(&json).expect("JSON deserialization failed");

            assert!(matches!(deserialized.pipeline_schedule, _));
        }
    }

    #[test]
    fn test_3d_parallelism_manager() {
        let config = ParallelismConfig::default();
        let mut manager = Parallelism3DManager::new(config);

        // Test configuration optimization with empty data
        let optimized_config = manager.optimize_configuration();
        assert!(optimized_config.is_ok());

        // Test aggregate stats with no coordinators
        let stats = manager.get_aggregate_stats();
        assert!(stats.is_ok());

        let stats = stats.expect("operation failed in test");
        assert_eq!(stats.num_coordinators, 0);
        assert_eq!(stats.average_pipeline_efficiency, 0.0);
    }

    #[test]
    fn test_config_serialization() {
        let config = ParallelismConfig {
            dp_size: 4,
            mp_size: 2,
            pp_size: 8,
            num_micro_batches: 16,
            gradient_accumulation: true,
            accumulation_steps: 4,
            activation_checkpointing: true,
            comm_backend: CommBackend::NCCL,
            pipeline_schedule: PipelineSchedule::Interleaved1F1B,
            memory_optimization: MemoryOptimization::High,
        };

        let json = serde_json::to_string(&config).expect("JSON serialization failed");
        let deserialized: ParallelismConfig =
            serde_json::from_str(&json).expect("JSON deserialization failed");

        assert_eq!(config.dp_size, deserialized.dp_size);
        assert_eq!(config.mp_size, deserialized.mp_size);
        assert_eq!(config.pp_size, deserialized.pp_size);
        assert_eq!(config.num_micro_batches, deserialized.num_micro_batches);
        assert_eq!(
            config.gradient_accumulation,
            deserialized.gradient_accumulation
        );
        assert_eq!(config.accumulation_steps, deserialized.accumulation_steps);
        assert_eq!(
            config.activation_checkpointing,
            deserialized.activation_checkpointing
        );
    }

    // ── Memory optimisation and pipeline scheduling ──────────────────────
    //
    // These assertions would all have failed against the previous
    // implementation, where `apply_gradient_compression` was an empty loop,
    // `apply_zero_optimization` returned `Ok(())` without partitioning
    // anything, and every backward pass returned its input unchanged.

    fn coordinator(memory_optimization: MemoryOptimization) -> Parallelism3D {
        let config = ParallelismConfig {
            dp_size: 1,
            mp_size: 1,
            pp_size: 1,
            memory_optimization,
            ..Default::default()
        };
        Parallelism3D::new(
            config,
            0,
            1,
            Arc::new(SimulatedProcessGroup::new(0, 1)),
            Arc::new(SimulatedProcessGroup::new(0, 1)),
            Arc::new(SimulatedProcessGroup::new(0, 1)),
        )
        .expect("coordinator must build in test")
    }

    #[test]
    fn gradient_compression_really_changes_the_values() {
        let parallelism = coordinator(MemoryOptimization::Medium)
            .with_gradient_compression(GradientCompressionConfig::TopK { ratio: 0.25 });

        let values: Vec<f32> = vec![0.01, -5.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07];
        let mut tensors =
            vec![Tensor::from_slice(&values, &[8]).expect("tensor must build in test")];

        parallelism
            .optimize_memory(&mut tensors)
            .expect("optimize_memory must succeed in test");

        let compressed = tensors[0].to_vec_f32().expect("tensor read must succeed in test");
        assert_ne!(
            compressed, values,
            "compression must actually modify the gradient"
        );
        // 25% of 8 elements = 2 survivors: the two largest magnitudes.
        let survivors = compressed.iter().filter(|value| **value != 0.0).count();
        assert_eq!(
            survivors, 2,
            "top-k must keep exactly ceil(8 * 0.25) entries"
        );
        approx::assert_relative_eq!(compressed[1], -5.0f32, epsilon = 1e-6);
        approx::assert_relative_eq!(compressed[7], 0.07f32, epsilon = 1e-6);
    }

    #[test]
    fn gradient_compression_none_is_the_identity() {
        let parallelism = coordinator(MemoryOptimization::Medium)
            .with_gradient_compression(GradientCompressionConfig::None);
        let values: Vec<f32> = vec![0.5, -0.25, 0.125, 1.0];
        let mut tensors =
            vec![Tensor::from_slice(&values, &[4]).expect("tensor must build in test")];
        parallelism
            .optimize_memory(&mut tensors)
            .expect("optimize_memory must succeed in test");
        assert_eq!(
            tensors[0].to_vec_f32().expect("tensor read must succeed in test"),
            values
        );
    }

    #[test]
    fn extreme_memory_optimization_refuses_to_claim_zero_partitioning() {
        let parallelism = coordinator(MemoryOptimization::Extreme);
        let mut tensors = vec![Tensor::ones(&[4]).expect("tensor must build in test")];
        let error = parallelism
            .optimize_memory(&mut tensors)
            .expect_err("Extreme must not silently do nothing in test");
        assert!(error.to_string().contains("ZeroStage1Optimizer"), "{error}");
    }

    #[test]
    fn backward_pass_without_an_executor_is_an_error() {
        let parallelism = coordinator(MemoryOptimization::None);
        let mut model = PassthroughModel::default();
        let gradients = vec![Tensor::ones(&[2]).expect("tensor must build in test")];

        let error = parallelism
            .backward_pass(&mut model, &gradients, 0)
            .expect_err("an unregistered backward must fail in test");
        assert!(error.to_string().contains("set_stage_backward"), "{error}");
    }

    #[test]
    fn backward_pass_runs_the_registered_executor() {
        let parallelism = coordinator(MemoryOptimization::None);
        parallelism.set_stage_backward(Box::new(|gradients: &[Tensor], _stage| {
            gradients.iter().map(|tensor| Ok(tensor.scalar_mul(2.0)?)).collect()
        }));

        let mut model = PassthroughModel::default();
        let gradients =
            vec![Tensor::from_slice(&[1.0f32, -2.0], &[2]).expect("tensor must build in test")];
        let out = parallelism
            .backward_pass(&mut model, &gradients, 0)
            .expect("backward must succeed in test");

        assert_eq!(
            out[0].to_vec_f32().expect("tensor read must succeed in test"),
            vec![2.0f32, -4.0],
            "the executor must run; echoing the input back would give [1, -2]"
        );
    }

    #[test]
    fn pipeline_backward_moves_gradients_from_the_last_stage_to_the_first() {
        use crate::distributed_collective::run_in_process;

        // Two pipeline stages as threads. Stage 1 differentiates the loss
        // gradient and sends the result to stage 0.
        let per_rank = run_in_process(2, |rank, group| -> Result<Vec<f32>> {
            let config = ParallelismConfig {
                dp_size: 1,
                mp_size: 1,
                pp_size: 2,
                ..Default::default()
            };
            let single: Arc<dyn ProcessGroup> = Arc::new(SimulatedProcessGroup::new(0, 1));
            let pp_group: Arc<dyn ProcessGroup> = group;
            let parallelism =
                Parallelism3D::new(config, rank, 2, Arc::clone(&single), single, pp_group)?;

            // Stage `s` multiplies the incoming gradient by (s + 2).
            parallelism.set_stage_backward(Box::new(|gradients: &[Tensor], stage: usize| {
                gradients
                    .iter()
                    .map(|tensor| Ok(tensor.scalar_mul(stage as f32 + 2.0)?))
                    .collect()
            }));

            let mut model = PassthroughModel::default();
            let seed = vec![Tensor::from_slice(&[1.0f32, 2.0], &[2])?];
            let out = parallelism.backward_pass(&mut model, &seed, 7)?;
            Ok(out[0].to_vec_f32()?)
        })
        .expect("in-process run must succeed in test");

        let stage0 = per_rank[0].as_ref().expect("stage 0 must complete in test");
        let stage1 = per_rank[1].as_ref().expect("stage 1 must complete in test");

        // Stage 1 sees the seed and scales by 3; stage 0 receives that and
        // scales by 2. Nothing here is reachable without a real transfer.
        assert_eq!(stage1, &vec![3.0f32, 6.0]);
        assert_eq!(stage0, &vec![6.0f32, 12.0]);
    }

    #[test]
    fn pipeline_bubble_optimisation_publishes_a_real_measurement() {
        let parallelism = coordinator(MemoryOptimization::None);
        assert_eq!(parallelism.rebalance_plan().recommended_micro_batches, 0);

        {
            let mut state = parallelism
                .pipeline_state
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.forward_passes_completed = 10;
            state.backward_passes_completed = 10;
            state.pipeline_bubbles = 12; // 60% bubbles, far above the 20% target
            state.stage_timings.insert(0, Duration::from_millis(10));
        }

        parallelism
            .optimize_pipeline_bubbles()
            .expect("optimisation must succeed in test");

        let plan = parallelism.rebalance_plan();
        approx::assert_relative_eq!(plan.measured_bubble_ratio, 0.6f32, epsilon = 1e-6);
        assert!(
            plan.recommended_micro_batches >= parallelism.config.num_micro_batches,
            "a high bubble ratio must recommend at least as many micro-batches"
        );
    }

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct PassthroughConfig;

    impl trustformers_core::traits::Config for PassthroughConfig {
        fn architecture(&self) -> &'static str {
            "passthrough"
        }
    }

    #[derive(Debug, Default)]
    struct PassthroughModel {
        config: PassthroughConfig,
    }

    impl Default for PassthroughConfig {
        fn default() -> Self {
            Self
        }
    }

    impl Model for PassthroughModel {
        type Config = PassthroughConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(
            &self,
            input: Self::Input,
        ) -> std::result::Result<Self::Output, trustformers_core::TrustformersError> {
            Ok(input)
        }

        fn load_pretrained(
            &mut self,
            _reader: &mut dyn std::io::Read,
        ) -> std::result::Result<(), trustformers_core::TrustformersError> {
            Ok(())
        }

        fn get_config(&self) -> &Self::Config {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            0
        }
    }
}
