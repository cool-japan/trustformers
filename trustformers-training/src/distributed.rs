//! Distributed data-parallel training primitives.
//!
//! # Backends
//!
//! Two collective backends are implemented in pure Rust and perform **real**
//! data movement and reduction arithmetic:
//!
//! * [`DistributedBackend::InProcess`] — multiple ranks running as threads of a
//!   single process, rendezvousing through shared memory. This is the backend
//!   exercised by the test-suite and by single-machine simulations.
//! * [`DistributedBackend::Tcp`] — multiple ranks running as separate processes
//!   (optionally on separate hosts), communicating over TCP sockets.
//!
//! Both are built on the transport/collective layer in
//! [`trustformers_optim::collective`], which implements ring all-reduce,
//! ring all-gather, ring reduce-scatter, binomial broadcast/reduce and a
//! rendezvous barrier.
//!
//! [`DistributedBackend::NCCL`], [`DistributedBackend::Gloo`] and
//! [`DistributedBackend::MPI`] wrap C/C++/CUDA libraries which the project's
//! pure-Rust policy forbids from the default build. They are retained as enum
//! variants so that configuration files remain loadable, but every constructor
//! and every collective returns [`DistributedError::BackendUnavailable`]. They
//! never fabricate a result.

use crate::gradient::GradientUtils;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Model;

/// Structured errors raised by the distributed training layer.
#[derive(Debug, thiserror::Error)]
pub enum DistributedError {
    /// The requested backend cannot be provided by this build.
    #[error("distributed backend `{backend}` is unavailable: {reason}")]
    BackendUnavailable {
        /// Human readable backend name (e.g. `"NCCL"`).
        backend: &'static str,
        /// Why the backend cannot be provided, and what to use instead.
        reason: String,
    },

    /// A collective was requested from a process group that performs no
    /// communication at all.
    #[error(
        "`{group}` performs no communication; `{operation}` is only valid for world_size == 1 \
         (got {world_size}). Use DistributedBackend::InProcess or DistributedBackend::Tcp."
    )]
    NoCommunication {
        /// Name of the process-group type.
        group: &'static str,
        /// Collective that was requested.
        operation: &'static str,
        /// Configured world size.
        world_size: usize,
    },

    /// No GPU enumeration is available on this host/build.
    #[error("no GPU devices could be enumerated: {reason}")]
    NoGpuDevices {
        /// Why enumeration failed.
        reason: String,
    },

    /// The configuration is internally inconsistent.
    #[error("invalid distributed configuration: {0}")]
    InvalidConfig(String),

    /// The process group cannot service this operation.
    #[error("`{operation}` is not supported by this process group: {reason}")]
    UnsupportedOperation {
        /// Operation that was requested.
        operation: &'static str,
        /// Why it is unavailable and what to use instead.
        reason: &'static str,
    },
}

/// Configuration for distributed training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Number of processes/nodes
    pub world_size: usize,
    /// Rank of current process (0 to world_size-1)
    pub rank: usize,
    /// Backend for communication (nccl, gloo, mpi)
    pub backend: DistributedBackend,
    /// Master address for coordination
    pub master_addr: String,
    /// Master port for coordination
    pub master_port: u16,
    /// Whether to compress gradients before the all-reduce.
    ///
    /// When `true`, [`DataParallelTrainer`] applies its configured codec (see
    /// [`DataParallelTrainer::with_gradient_compression`], default
    /// [`GradientCompressionConfig::default_when_enabled`]) to every gradient
    /// prior to the collective and reconstructs a dense tensor afterwards.
    /// Compression is lossy.
    pub gradient_compression: bool,
    /// Bucket size for gradient bucketing
    pub bucket_size_mb: usize,
}

/// Codec used to compress gradients before they are all-reduced.
///
/// Compression is *lossy*: the decompressed gradient is used for the parameter
/// update, so every variant here trades accuracy for bandwidth. The identity
/// codec [`GradientCompressionConfig::None`] is the default.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum GradientCompressionConfig {
    /// No compression; gradients are transmitted verbatim.
    #[default]
    None,
    /// Keep only the `ratio` fraction of entries with the largest magnitude,
    /// zeroing the rest. `ratio` is clamped to `(0, 1]`.
    TopK {
        /// Fraction of entries to retain.
        ratio: f32,
    },
    /// Uniform affine quantization to `bits` bits (2..=8) with a per-tensor
    /// scale and zero point.
    Quantize {
        /// Number of quantization bits.
        bits: u8,
    },
}

impl GradientCompressionConfig {
    /// Codec used when [`DistributedConfig::gradient_compression`] is enabled
    /// but no explicit codec was selected: 10% top-k sparsification, the
    /// classic bandwidth/accuracy compromise for data-parallel SGD.
    pub fn default_when_enabled() -> Self {
        Self::TopK { ratio: 0.1 }
    }

    /// Compress `values` in place, returning the reconstruction that the
    /// receiver would observe.
    ///
    /// Every codec is expressed as "compress then immediately decompress" so
    /// that the reduction operates on the exact values the peers exchanged.
    /// This is the standard way lossy gradient compression is modelled on a
    /// symmetric all-reduce: each rank contributes its *decompressed* payload.
    pub fn apply(&self, values: &mut [f32]) -> Result<()> {
        match *self {
            Self::None => Ok(()),
            Self::TopK { ratio } => {
                if !(ratio.is_finite() && ratio > 0.0) {
                    return Err(DistributedError::InvalidConfig(format!(
                        "TopK ratio must be finite and > 0, got {ratio}"
                    ))
                    .into());
                }
                if ratio >= 1.0 || values.is_empty() {
                    return Ok(());
                }
                let keep = ((values.len() as f32) * ratio).ceil() as usize;
                let keep = keep.clamp(1, values.len());
                if keep == values.len() {
                    return Ok(());
                }

                // Select exactly `keep` indices with the largest magnitude
                // (ties broken by index, so the result is deterministic and the
                // retained count is exact).
                let mut ranked: Vec<(usize, f32)> =
                    values.iter().enumerate().map(|(i, v)| (i, v.abs())).collect();
                ranked.select_nth_unstable_by(keep - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
                });

                let mut retain = vec![false; values.len()];
                for (index, _) in ranked.iter().take(keep) {
                    retain[*index] = true;
                }
                for (value, keep_it) in values.iter_mut().zip(retain) {
                    if !keep_it {
                        *value = 0.0;
                    }
                }
                Ok(())
            },
            Self::Quantize { bits } => {
                if !(2..=8).contains(&bits) {
                    return Err(DistributedError::InvalidConfig(format!(
                        "Quantize bits must be in 2..=8, got {bits}"
                    ))
                    .into());
                }
                if values.is_empty() {
                    return Ok(());
                }
                let mut min_val = f32::INFINITY;
                let mut max_val = f32::NEG_INFINITY;
                for &v in values.iter() {
                    min_val = min_val.min(v);
                    max_val = max_val.max(v);
                }
                let levels = (1u32 << bits) as f32;
                let span = max_val - min_val;
                if !span.is_finite() || span <= 0.0 {
                    // Constant tensor: quantization is exact, nothing to do.
                    return Ok(());
                }
                let scale = span / (levels - 1.0);
                for value in values.iter_mut() {
                    let level = ((*value - min_val) / scale).round().clamp(0.0, levels - 1.0);
                    *value = min_val + level * scale;
                }
                Ok(())
            },
        }
    }
}

/// Communication backend used for collective operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributedBackend {
    /// NVIDIA Collective Communications Library.
    ///
    /// Requires the proprietary CUDA/NCCL C libraries, which are excluded from
    /// the default (pure-Rust) build. Selecting it yields
    /// [`DistributedError::BackendUnavailable`]; use
    /// [`DistributedBackend::Tcp`] instead.
    NCCL,
    /// Gloo for CPU communication.
    ///
    /// Gloo is a C++ library and is excluded from the default (pure-Rust)
    /// build. [`DistributedBackend::Tcp`] provides the equivalent pure-Rust
    /// CPU collectives.
    Gloo,
    /// Message Passing Interface.
    ///
    /// Requires an MPI C implementation and is excluded from the default
    /// (pure-Rust) build. Use [`DistributedBackend::Tcp`].
    MPI,
    /// Single-rank group. All collectives are mathematically the identity for
    /// `world_size == 1`; any larger world size is rejected because no
    /// communication takes place.
    Simulated,
    /// Real collectives between ranks running as threads of one process,
    /// rendezvousing through shared memory.
    InProcess,
    /// Real collectives between ranks running as separate processes, over TCP.
    Tcp,
}

/// Process group for distributed communication.
///
/// Implementors provide the four primitive collectives; [`ProcessGroup::all_gather`]
/// and [`ProcessGroup::reduce_scatter`] ship real default implementations
/// expressed in terms of them, and point-to-point transfer is optional (see
/// [`ProcessGroup::send`]).
///
/// All collectives are SPMD: every rank must issue the same sequence of calls
/// with matching shapes.
pub trait ProcessGroup: Send + Sync {
    /// All-reduce operation to sum gradients across all processes
    fn all_reduce(&self, tensors: &mut [Tensor]) -> Result<()>;

    /// Broadcast tensor from source rank to all other ranks
    fn broadcast(&self, tensor: &mut Tensor, src_rank: usize) -> Result<()>;

    /// Reduce operation to sum tensors to a specific rank
    fn reduce(&self, tensor: &mut Tensor, dst_rank: usize) -> Result<()>;

    /// Barrier synchronization
    fn barrier(&self) -> Result<()>;

    /// Get rank of current process
    fn rank(&self) -> usize;

    /// Get total number of processes
    fn world_size(&self) -> usize;

    /// Gather every rank's `tensor`, returned in rank order.
    ///
    /// The default implementation performs `world_size` broadcasts, which is
    /// correct for any process group. Backends with a native ring all-gather
    /// should override it.
    fn all_gather(&self, tensor: &Tensor) -> Result<Vec<Tensor>> {
        let mut gathered = Vec::with_capacity(self.world_size());
        for root in 0..self.world_size() {
            let mut candidate = tensor.clone();
            self.broadcast(&mut candidate, root)?;
            gathered.push(candidate);
        }
        Ok(gathered)
    }

    /// Sum `tensor` across all ranks and return this rank's contiguous slice of
    /// the result.
    ///
    /// The default implementation all-reduces and then slices, which is correct
    /// though not bandwidth-optimal.
    fn reduce_scatter(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut reduced = vec![tensor.clone()];
        self.all_reduce(&mut reduced)?;
        let values = reduced[0].to_vec_f32()?;

        let world_size = self.world_size();
        let base = values.len() / world_size;
        let remainder = values.len() % world_size;
        let rank = self.rank();
        let start = rank * base + remainder.min(rank);
        let length = base + usize::from(rank < remainder);

        Ok(Tensor::from_slice(
            &values[start..start + length],
            &[length],
        )?)
    }

    /// Send `tensor` to `peer`, labelled with `tag`.
    ///
    /// `tag` lets both sides agree on which message is which without relying on
    /// call ordering, which matters when ranks follow different code paths (for
    /// example pipeline stages or hierarchical reductions).
    ///
    /// The default implementation reports
    /// [`DistributedError::UnsupportedOperation`]: point-to-point transfer is
    /// only meaningful for groups with a real transport.
    fn send(&self, _peer: usize, _tag: u64, _tensor: &Tensor) -> Result<()> {
        Err(DistributedError::UnsupportedOperation {
            operation: "send",
            reason: "this process group has no point-to-point transport; use \
                     DistributedBackend::InProcess or DistributedBackend::Tcp",
        }
        .into())
    }

    /// Receive a tensor of `shape` from `peer`, labelled with `tag`.
    fn recv(&self, _peer: usize, _tag: u64, _shape: &[usize]) -> Result<Tensor> {
        Err(DistributedError::UnsupportedOperation {
            operation: "recv",
            reason: "this process group has no point-to-point transport; use \
                     DistributedBackend::InProcess or DistributedBackend::Tcp",
        }
        .into())
    }

    /// Whether [`ProcessGroup::send`] / [`ProcessGroup::recv`] are available.
    fn supports_point_to_point(&self) -> bool {
        false
    }
}

/// Process group that performs no communication.
///
/// For `world_size == 1` every collective is genuinely the identity function
/// (all-reduce over a single contributor returns that contributor's data,
/// broadcast from self is a no-op, and so on), so this group is *correct* for
/// single-rank runs.
///
/// For `world_size > 1` there is no peer to exchange data with. Rather than
/// silently returning unsynchronised tensors, every collective returns
/// [`DistributedError::NoCommunication`]. Use
/// [`crate::distributed_collective::InProcessProcessGroup`] (threads) or
/// [`crate::distributed_collective::TcpProcessGroup`] (processes) for real
/// multi-rank collectives.
#[derive(Debug)]
pub struct SimulatedProcessGroup {
    rank: usize,
    world_size: usize,
}

impl SimulatedProcessGroup {
    /// Create a non-communicating process group.
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self { rank, world_size }
    }

    fn reject(&self, operation: &'static str) -> anyhow::Error {
        DistributedError::NoCommunication {
            group: "SimulatedProcessGroup",
            operation,
            world_size: self.world_size,
        }
        .into()
    }
}

impl ProcessGroup for SimulatedProcessGroup {
    fn all_reduce(&self, _tensors: &mut [Tensor]) -> Result<()> {
        // All-reduce over a single contributor is the identity.
        if self.world_size == 1 {
            return Ok(());
        }
        Err(self.reject("all_reduce"))
    }

    fn broadcast(&self, _tensor: &mut Tensor, src_rank: usize) -> Result<()> {
        // Broadcasting from oneself to oneself is the identity.
        if self.world_size == 1 && src_rank == self.rank {
            return Ok(());
        }
        Err(self.reject("broadcast"))
    }

    fn reduce(&self, _tensor: &mut Tensor, dst_rank: usize) -> Result<()> {
        if self.world_size == 1 && dst_rank == self.rank {
            return Ok(());
        }
        Err(self.reject("reduce"))
    }

    fn barrier(&self) -> Result<()> {
        // A barrier across a single rank is trivially satisfied.
        if self.world_size == 1 {
            return Ok(());
        }
        Err(self.reject("barrier"))
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }
}

/// Descriptor for a collective backend that this build cannot provide.
///
/// The three well-known HPC backends (NCCL, Gloo, MPI) are implemented as
/// C/C++/CUDA libraries. TrustformeRS ships a pure-Rust default build, so
/// linking them is not possible here. Instead of fabricating results, every
/// entry point of these process groups returns
/// [`DistributedError::BackendUnavailable`].
#[derive(Debug, Clone, Copy)]
struct UnavailableBackend {
    name: &'static str,
    reason: &'static str,
}

impl UnavailableBackend {
    const NCCL: Self = Self {
        name: "NCCL",
        reason: "NCCL is a proprietary CUDA C library and cannot be linked by the pure-Rust \
                 default build. Use DistributedBackend::Tcp for real multi-process collectives, \
                 or DistributedBackend::InProcess for multi-threaded ranks.",
    };
    const GLOO: Self = Self {
        name: "Gloo",
        reason: "Gloo is a C++ library and cannot be linked by the pure-Rust default build. \
                 DistributedBackend::Tcp provides equivalent pure-Rust CPU collectives \
                 (ring all-reduce/all-gather/reduce-scatter, binomial broadcast/reduce).",
    };
    const MPI: Self = Self {
        name: "MPI",
        reason: "MPI requires a C implementation (OpenMPI/MPICH) and cannot be linked by the \
                 pure-Rust default build. Use DistributedBackend::Tcp.",
    };

    fn error(self) -> anyhow::Error {
        DistributedError::BackendUnavailable {
            backend: self.name,
            reason: self.reason.to_string(),
        }
        .into()
    }
}

/// Configuration for an unavailable backend, retained so that callers can
/// inspect what *would* have been used.
#[derive(Debug, Clone)]
pub struct UnavailableBackendConfig {
    rank: usize,
    world_size: usize,
    device_id: usize,
    master_addr: String,
    master_port: u16,
}

impl UnavailableBackendConfig {
    /// Rank this group would have taken.
    pub fn rank(&self) -> usize {
        self.rank
    }
    /// World size this group would have taken.
    pub fn world_size(&self) -> usize {
        self.world_size
    }
    /// Device index this group would have bound to.
    pub fn device_id(&self) -> usize {
        self.device_id
    }
    /// Rendezvous host.
    pub fn master_addr(&self) -> &str {
        &self.master_addr
    }
    /// Rendezvous port.
    pub fn master_port(&self) -> u16 {
        self.master_port
    }
}

macro_rules! unavailable_process_group {
    ($ty:ident, $backend:expr, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug)]
        pub struct $ty {
            config: UnavailableBackendConfig,
        }

        impl $ty {
            /// Returns the configuration this group would have used.
            pub fn config(&self) -> &UnavailableBackendConfig {
                &self.config
            }

            /// The reason this backend cannot be provided.
            pub fn unavailable_reason() -> &'static str {
                $backend.reason
            }
        }

        impl ProcessGroup for $ty {
            fn all_reduce(&self, _tensors: &mut [Tensor]) -> Result<()> {
                Err($backend.error())
            }

            fn broadcast(&self, _tensor: &mut Tensor, _src_rank: usize) -> Result<()> {
                Err($backend.error())
            }

            fn reduce(&self, _tensor: &mut Tensor, _dst_rank: usize) -> Result<()> {
                Err($backend.error())
            }

            fn barrier(&self) -> Result<()> {
                Err($backend.error())
            }

            fn rank(&self) -> usize {
                self.config.rank
            }

            fn world_size(&self) -> usize {
                self.config.world_size
            }
        }
    };
}

unavailable_process_group!(
    NCCLProcessGroup,
    UnavailableBackend::NCCL,
    "NCCL-based process group for GPU distributed training.\n\n\
     Unavailable in the pure-Rust build: every constructor and collective returns\n\
     [`DistributedError::BackendUnavailable`]. See [`DistributedBackend::NCCL`]."
);

unavailable_process_group!(
    GlooProcessGroup,
    UnavailableBackend::GLOO,
    "Gloo-based process group for CPU distributed training.\n\n\
     Unavailable in the pure-Rust build: every constructor and collective returns\n\
     [`DistributedError::BackendUnavailable`]. Use [`DistributedBackend::Tcp`]."
);

unavailable_process_group!(
    MPIProcessGroup,
    UnavailableBackend::MPI,
    "MPI-based process group for distributed training.\n\n\
     Unavailable in the pure-Rust build: every constructor and collective returns\n\
     [`DistributedError::BackendUnavailable`]. Use [`DistributedBackend::Tcp`]."
);

impl NCCLProcessGroup {
    /// Always fails with [`DistributedError::BackendUnavailable`].
    ///
    /// NCCL cannot be linked by the pure-Rust default build; returning an error
    /// here is the only honest outcome.
    pub fn new(
        _rank: usize,
        _world_size: usize,
        _device_id: usize,
        _master_addr: String,
        _master_port: u16,
    ) -> Result<Self> {
        Err(UnavailableBackend::NCCL.error())
    }
}

impl GlooProcessGroup {
    /// Always fails with [`DistributedError::BackendUnavailable`].
    pub fn new(
        _rank: usize,
        _world_size: usize,
        _master_addr: String,
        _master_port: u16,
    ) -> Result<Self> {
        Err(UnavailableBackend::GLOO.error())
    }
}

impl MPIProcessGroup {
    /// Always fails with [`DistributedError::BackendUnavailable`].
    pub fn new(_rank: usize, _world_size: usize) -> Result<Self> {
        Err(UnavailableBackend::MPI.error())
    }
}

/// Data parallel trainer that wraps a model for distributed training.
///
/// # Parameter registry
///
/// The [`Model`] trait exposes no parameter iterator, so this trainer cannot
/// discover a model's weights on its own. Callers register the tensors they
/// want kept in sync via [`DataParallelTrainer::register_parameters`];
/// [`DataParallelTrainer::broadcast_parameters`] then broadcasts exactly those
/// from rank 0 and returns the synchronised values. Calling it without a
/// registry is an error rather than a fabricated parameter list.
pub struct DataParallelTrainer<M: Model<Input = Tensor, Output = Tensor>> {
    model: Arc<Mutex<M>>,
    process_group: Arc<dyn ProcessGroup>,
    config: DistributedConfig,
    compression: GradientCompressionConfig,
    parameters: Mutex<HashMap<String, Tensor>>,
}

impl<M: Model<Input = Tensor, Output = Tensor>> DataParallelTrainer<M> {
    /// Wrap `model` for data-parallel training over `process_group`.
    pub fn new(
        model: M,
        process_group: Arc<dyn ProcessGroup>,
        config: DistributedConfig,
    ) -> Result<Self> {
        if config.world_size != process_group.world_size() {
            return Err(DistributedError::InvalidConfig(format!(
                "config.world_size ({}) disagrees with the process group ({})",
                config.world_size,
                process_group.world_size()
            ))
            .into());
        }

        let compression = if config.gradient_compression {
            GradientCompressionConfig::default_when_enabled()
        } else {
            GradientCompressionConfig::None
        };

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            process_group,
            config,
            compression,
            parameters: Mutex::new(HashMap::new()),
        })
    }

    /// Override the gradient compression codec.
    ///
    /// Selecting anything other than [`GradientCompressionConfig::None`]
    /// implicitly enables compression regardless of
    /// [`DistributedConfig::gradient_compression`].
    pub fn with_gradient_compression(mut self, codec: GradientCompressionConfig) -> Self {
        self.compression = codec;
        self
    }

    /// The active gradient compression codec.
    pub fn gradient_compression(&self) -> GradientCompressionConfig {
        self.compression
    }

    /// Register the parameter tensors that participate in synchronisation.
    pub fn register_parameters(&self, parameters: HashMap<String, Tensor>) -> Result<()> {
        let mut guard = self.parameters.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = parameters;
        Ok(())
    }

    /// Snapshot the registered parameters.
    pub fn parameters(&self) -> HashMap<String, Tensor> {
        self.parameters.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
    }

    /// Forward pass through the model
    pub fn forward(&self, input: Tensor) -> Result<Tensor> {
        let model = self.model.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        model.forward(input).map_err(|e| anyhow::anyhow!(e))
    }

    /// Backward pass with gradient synchronization followed by global norm
    /// clipping.
    pub fn backward(&self, gradients: &mut HashMap<String, Tensor>) -> Result<()> {
        self.synchronize_gradients(gradients)?;

        // Clip using a deterministic, name-sorted ordering so that every rank
        // computes the identical global norm.
        let names: Vec<String> = sorted_names(gradients);
        let mut gradient_vec: Vec<Tensor> =
            names.iter().filter_map(|name| gradients.get(name).cloned()).collect();
        GradientUtils::clip_grad_norm(&mut gradient_vec, 1.0)?;

        for (name, clipped) in names.iter().zip(gradient_vec) {
            if let Some(slot) = gradients.get_mut(name) {
                *slot = clipped;
            }
        }

        Ok(())
    }

    /// Group parameter names into all-reduce buckets no larger than
    /// [`DistributedConfig::bucket_size_mb`].
    ///
    /// Bucketing is driven purely by the (sorted) names and tensor byte sizes,
    /// so every rank derives the identical plan and the collectives stay in
    /// lock-step.
    pub fn bucket_plan(&self, gradients: &HashMap<String, Tensor>) -> Vec<Vec<String>> {
        let budget = self.config.bucket_size_mb.max(1) * 1024 * 1024;
        let mut buckets: Vec<Vec<String>> = Vec::new();
        let mut current: Vec<String> = Vec::new();
        let mut current_bytes = 0usize;

        for name in sorted_names(gradients) {
            let bytes = gradients.get(&name).map(|t| t.len() * 4).unwrap_or(0);
            if !current.is_empty() && current_bytes + bytes > budget {
                buckets.push(std::mem::take(&mut current));
                current_bytes = 0;
            }
            current_bytes += bytes;
            current.push(name);
        }

        if !current.is_empty() {
            buckets.push(current);
        }
        buckets
    }

    /// Synchronize gradients across all processes: (optional) compression,
    /// bucketed all-reduce (sum), then division by the world size.
    fn synchronize_gradients(&self, gradients: &mut HashMap<String, Tensor>) -> Result<()> {
        if gradients.is_empty() {
            return Ok(());
        }

        // Lossy compression is applied before the collective so that each rank
        // contributes exactly the payload it would have transmitted.
        if self.compression != GradientCompressionConfig::None {
            for name in sorted_names(gradients) {
                if let Some(tensor) = gradients.get_mut(&name) {
                    let shape = tensor.shape();
                    let mut values = tensor.to_vec_f32()?;
                    self.compression.apply(&mut values)?;
                    *tensor = Tensor::from_slice(&values, &shape)?;
                }
            }
        }

        let world_size = self.process_group.world_size();

        for bucket in self.bucket_plan(gradients) {
            let mut bucket_tensors: Vec<Tensor> =
                bucket.iter().filter_map(|name| gradients.get(name).cloned()).collect();

            self.process_group.all_reduce(&mut bucket_tensors)?;

            for (name, reduced) in bucket.iter().zip(bucket_tensors) {
                let averaged = if world_size > 1 {
                    reduced.scalar_mul(1.0 / world_size as f32)?
                } else {
                    reduced
                };
                if let Some(slot) = gradients.get_mut(name) {
                    *slot = averaged;
                }
            }
        }

        Ok(())
    }

    /// Broadcast the registered parameters from rank 0 to all other ranks.
    ///
    /// Returns the synchronised parameters, which are also stored back into the
    /// registry. Errors with [`DistributedError::InvalidConfig`] when no
    /// parameters have been registered.
    pub fn broadcast_parameters(&self) -> Result<HashMap<String, Tensor>> {
        let snapshot = self.parameters();
        if snapshot.is_empty() {
            return Err(DistributedError::InvalidConfig(
                "no parameters registered; call DataParallelTrainer::register_parameters first \
                 (the Model trait exposes no parameter iterator, so they cannot be discovered)"
                    .to_string(),
            )
            .into());
        }

        let mut synchronized = HashMap::with_capacity(snapshot.len());
        // Sorted order keeps every rank issuing the same broadcast sequence.
        for name in sorted_names(&snapshot) {
            let mut tensor = snapshot
                .get(&name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("parameter `{name}` vanished during broadcast"))?;
            self.process_group.broadcast(&mut tensor, 0)?;
            synchronized.insert(name, tensor);
        }

        self.register_parameters(synchronized.clone())?;
        Ok(synchronized)
    }

    /// Get the wrapped model
    pub fn model(&self) -> Arc<Mutex<M>> {
        self.model.clone()
    }

    /// Get process group
    pub fn process_group(&self) -> Arc<dyn ProcessGroup> {
        self.process_group.clone()
    }

    /// The configuration this trainer was built with.
    pub fn config(&self) -> &DistributedConfig {
        &self.config
    }
}

/// Deterministic, rank-independent iteration order for a parameter map.
fn sorted_names(map: &HashMap<String, Tensor>) -> Vec<String> {
    let mut names: Vec<String> = map.keys().cloned().collect();
    names.sort();
    names
}

/// Initialize distributed training environment.
///
/// * [`DistributedBackend::Simulated`] yields a non-communicating group; valid
///   only for `world_size == 1`.
/// * [`DistributedBackend::InProcess`] rendezvouses with the other ranks of the
///   same process through a shared-memory registry keyed by
///   `master_addr:master_port`. Every rank must call this with the same key,
///   the same `world_size` and a distinct `rank`.
/// * [`DistributedBackend::Tcp`] binds `master_port + rank` and connects to the
///   remaining ranks at `master_addr:(master_port + peer)`.
/// * NCCL / Gloo / MPI return [`DistributedError::BackendUnavailable`].
pub fn init_distributed_training(config: DistributedConfig) -> Result<Arc<dyn ProcessGroup>> {
    if config.world_size == 0 {
        return Err(DistributedError::InvalidConfig("world_size must be >= 1".to_string()).into());
    }
    if config.rank >= config.world_size {
        return Err(DistributedError::InvalidConfig(format!(
            "rank {} is out of range for world_size {}",
            config.rank, config.world_size
        ))
        .into());
    }

    match config.backend {
        DistributedBackend::Simulated => Ok(Arc::new(SimulatedProcessGroup::new(
            config.rank,
            config.world_size,
        ))),
        DistributedBackend::InProcess => {
            let session = format!("{}:{}", config.master_addr, config.master_port);
            let pg = crate::distributed_collective::InProcessProcessGroup::join(
                &session,
                config.rank,
                config.world_size,
            )?;
            Ok(Arc::new(pg))
        },
        DistributedBackend::Tcp => {
            let pg = crate::distributed_collective::TcpProcessGroup::connect(
                config.rank,
                config.world_size,
                &config.master_addr,
                config.master_port,
            )?;
            Ok(Arc::new(pg))
        },
        DistributedBackend::NCCL => Err(UnavailableBackend::NCCL.error()),
        DistributedBackend::Gloo => Err(UnavailableBackend::GLOO.error()),
        DistributedBackend::MPI => Err(UnavailableBackend::MPI.error()),
    }
}

/// Detect the number of GPUs visible to this process.
///
/// The pure-Rust default build links neither CUDA nor ROCm, so the only
/// information available is the `CUDA_VISIBLE_DEVICES` environment variable set
/// by the job launcher. When it is absent, this returns
/// [`DistributedError::NoGpuDevices`] rather than inventing a device count —
/// guessing would map ranks onto devices that do not exist.
pub fn detect_gpu_count() -> Result<usize> {
    let devices =
        std::env::var("CUDA_VISIBLE_DEVICES").map_err(|_| DistributedError::NoGpuDevices {
            reason: "CUDA_VISIBLE_DEVICES is not set and this build links no GPU runtime; \
                     no device enumeration is possible"
                .to_string(),
        })?;

    let count = devices.split(',').filter(|entry| !entry.trim().is_empty()).count();

    if count == 0 {
        return Err(DistributedError::NoGpuDevices {
            reason: "CUDA_VISIBLE_DEVICES is set but lists no devices".to_string(),
        }
        .into());
    }

    Ok(count)
}

/// Utility functions for distributed training
pub mod utils {
    use super::*;

    /// Get local rank from environment variables
    pub fn get_local_rank() -> usize {
        std::env::var("LOCAL_RANK")
            .unwrap_or_else(|_| "0".to_string())
            .parse()
            .unwrap_or(0)
    }

    /// Get world size from environment variables
    pub fn get_world_size() -> usize {
        std::env::var("WORLD_SIZE")
            .unwrap_or_else(|_| "1".to_string())
            .parse()
            .unwrap_or(1)
    }

    /// Get rank from environment variables
    pub fn get_rank() -> usize {
        std::env::var("RANK").unwrap_or_else(|_| "0".to_string()).parse().unwrap_or(0)
    }

    /// Check if distributed training is enabled
    pub fn is_distributed() -> bool {
        get_world_size() > 1
    }

    /// Create default distributed config from environment
    pub fn default_distributed_config() -> DistributedConfig {
        DistributedConfig {
            world_size: get_world_size(),
            rank: get_rank(),
            backend: DistributedBackend::Simulated,
            master_addr: std::env::var("MASTER_ADDR").unwrap_or_else(|_| "localhost".to_string()),
            master_port: std::env::var("MASTER_PORT")
                .unwrap_or_else(|_| "29500".to_string())
                .parse()
                .unwrap_or(29500),
            gradient_compression: false,
            bucket_size_mb: 25,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use trustformers_core::tensor::Tensor;
    use trustformers_core::TrustformersError;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct DummyConfig;

    impl trustformers_core::traits::Config for DummyConfig {
        fn architecture(&self) -> &'static str {
            "dummy"
        }
    }

    #[derive(Debug, Clone)]
    struct DummyModel {
        config: DummyConfig,
    }

    impl DummyModel {
        fn new() -> Self {
            Self {
                config: DummyConfig,
            }
        }
    }

    impl Model for DummyModel {
        type Config = DummyConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, input: Self::Input) -> Result<Self::Output, TrustformersError> {
            Ok(input)
        }

        fn load_pretrained(
            &mut self,
            _reader: &mut dyn std::io::Read,
        ) -> Result<(), TrustformersError> {
            Ok(())
        }

        fn get_config(&self) -> &Self::Config {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            0 // DummyModel has no parameters
        }
    }

    #[test]
    fn test_simulated_process_group() {
        let pg = SimulatedProcessGroup::new(0, 1);
        assert_eq!(pg.rank(), 0);
        assert_eq!(pg.world_size(), 1);

        // Test barrier
        assert!(pg.barrier().is_ok());
    }

    #[test]
    fn test_data_parallel_trainer_creation() {
        let model = DummyModel::new();
        let config = DistributedConfig {
            world_size: 1,
            rank: 0,
            backend: DistributedBackend::Simulated,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            gradient_compression: false,
            bucket_size_mb: 25,
        };
        let pg = Arc::new(SimulatedProcessGroup::new(0, 1));

        let trainer = DataParallelTrainer::new(model, pg, config);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_gradient_synchronization() {
        let model = DummyModel::new();
        let config = DistributedConfig {
            world_size: 1,
            rank: 0,
            backend: DistributedBackend::Simulated,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            gradient_compression: false,
            bucket_size_mb: 25,
        };
        let pg = Arc::new(SimulatedProcessGroup::new(0, 1));

        let trainer =
            DataParallelTrainer::new(model, pg, config).expect("operation failed in test");

        let mut gradients = HashMap::new();
        gradients.insert(
            "test_param".to_string(),
            Tensor::ones(&[2, 2]).expect("tensor operation failed"),
        );

        let result = trainer.backward(&mut gradients);
        assert!(result.is_ok());
    }

    #[test]
    fn test_distributed_utils() {
        // Test environment variable parsing with defaults
        let world_size = utils::get_world_size();
        assert!(world_size >= 1);

        let rank = utils::get_rank();
        assert!(rank < world_size || world_size == 1);

        let config = utils::default_distributed_config();
        assert_eq!(config.world_size, world_size);
        assert_eq!(config.rank, rank);
    }

    #[test]
    fn test_init_distributed_training() {
        let config = DistributedConfig {
            world_size: 2,
            rank: 0,
            backend: DistributedBackend::Simulated,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            gradient_compression: false,
            bucket_size_mb: 25,
        };

        let pg = init_distributed_training(config);
        assert!(pg.is_ok());

        let pg = pg.expect("operation failed in test");
        assert_eq!(pg.rank(), 0);
        assert_eq!(pg.world_size(), 2);
    }

    // ── Additional tests ──────────────────────────────────────────────────────

    #[test]
    fn test_distributed_config_simulated_backend() {
        let cfg = DistributedConfig {
            world_size: 1,
            rank: 0,
            backend: DistributedBackend::Simulated,
            master_addr: "127.0.0.1".to_string(),
            master_port: 12345,
            gradient_compression: false,
            bucket_size_mb: 16,
        };
        assert!(matches!(cfg.backend, DistributedBackend::Simulated));
    }

    #[test]
    fn test_distributed_backend_variants() {
        let _ = DistributedBackend::NCCL;
        let _ = DistributedBackend::Gloo;
        let _ = DistributedBackend::MPI;
        let _ = DistributedBackend::Simulated;
    }

    #[test]
    fn test_simulated_process_group_rank() {
        let pg = SimulatedProcessGroup::new(2, 4);
        assert_eq!(pg.rank(), 2);
        assert_eq!(pg.world_size(), 4);
    }

    #[test]
    fn test_simulated_process_group_barrier_ok() {
        let pg = SimulatedProcessGroup::new(0, 1);
        let result = pg.barrier();
        assert!(result.is_ok());
    }

    #[test]
    fn test_simulated_process_group_single_node_all_reduce_ok() {
        let pg = SimulatedProcessGroup::new(0, 1);
        let mut tensors = vec![Tensor::ones(&[2]).expect("tensor failed")];
        let result = pg.all_reduce(&mut tensors);
        assert!(result.is_ok());
    }

    #[test]
    fn test_simulated_process_group_broadcast_ok() {
        let pg = SimulatedProcessGroup::new(0, 1);
        let mut tensor = Tensor::ones(&[3]).expect("tensor failed");
        let result = pg.broadcast(&mut tensor, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_simulated_process_group_reduce_ok() {
        let pg = SimulatedProcessGroup::new(0, 1);
        let mut tensor = Tensor::ones(&[2]).expect("tensor failed");
        let result = pg.reduce(&mut tensor, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_init_distributed_multi_node_returns_correct_rank() {
        let cfg = DistributedConfig {
            world_size: 3,
            rank: 1,
            backend: DistributedBackend::Simulated,
            master_addr: "localhost".to_string(),
            master_port: 29501,
            gradient_compression: false,
            bucket_size_mb: 32,
        };
        let pg = init_distributed_training(cfg).expect("init failed");
        assert_eq!(pg.rank(), 1);
        assert_eq!(pg.world_size(), 3);
    }

    #[test]
    fn test_data_parallel_trainer_with_gradient_compression() {
        let model = DummyModel::new();
        let cfg = DistributedConfig {
            world_size: 1,
            rank: 0,
            backend: DistributedBackend::Simulated,
            master_addr: "localhost".to_string(),
            master_port: 29502,
            gradient_compression: true,
            bucket_size_mb: 16,
        };
        let pg = Arc::new(SimulatedProcessGroup::new(0, 1));
        let trainer = DataParallelTrainer::new(model, pg, cfg);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_distributed_config_gradient_compression_field() {
        let cfg = DistributedConfig {
            world_size: 2,
            rank: 0,
            backend: DistributedBackend::Gloo,
            master_addr: "host".to_string(),
            master_port: 8080,
            gradient_compression: true,
            bucket_size_mb: 64,
        };
        assert!(cfg.gradient_compression);
        assert_eq!(cfg.bucket_size_mb, 64);
    }

    #[test]
    fn test_distributed_config_world_size_rank_consistency() {
        let cfg = DistributedConfig {
            world_size: 8,
            rank: 7,
            backend: DistributedBackend::Simulated,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            gradient_compression: false,
            bucket_size_mb: 25,
        };
        assert!(cfg.rank < cfg.world_size, "rank must be < world_size");
    }

    #[test]
    fn test_empty_gradients_backward_ok() {
        let model = DummyModel::new();
        let cfg = DistributedConfig {
            world_size: 1,
            rank: 0,
            backend: DistributedBackend::Simulated,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            gradient_compression: false,
            bucket_size_mb: 25,
        };
        let pg = Arc::new(SimulatedProcessGroup::new(0, 1));
        let trainer = DataParallelTrainer::new(model, pg, cfg).expect("trainer failed");
        let mut gradients = HashMap::new();
        let result = trainer.backward(&mut gradients);
        assert!(result.is_ok(), "empty gradient map should be fine");
    }

    // ── Unavailable backends must never fabricate a result ────────────────
    //
    // The previous implementation handed out NCCL/Gloo/MPI process groups whose
    // `all_reduce` was `scalar_mul(1.0)` and whose `broadcast` multiplied every
    // non-root rank's tensor by 0.99/0.98/0.97. These tests fail against that
    // code: they require an error, and they require the tensor to be untouched.

    fn unavailable_config(backend: DistributedBackend) -> DistributedConfig {
        DistributedConfig {
            world_size: 4,
            rank: 1,
            backend,
            master_addr: "127.0.0.1".to_string(),
            master_port: 29600,
            gradient_compression: false,
            bucket_size_mb: 25,
        }
    }

    #[test]
    fn nccl_gloo_and_mpi_are_reported_unavailable_not_simulated() {
        for (backend, name) in [
            (DistributedBackend::NCCL, "NCCL"),
            (DistributedBackend::Gloo, "Gloo"),
            (DistributedBackend::MPI, "MPI"),
        ] {
            let Err(error) = init_distributed_training(unavailable_config(backend)) else {
                panic!("an unavailable backend must not yield a process group in test");
            };
            let message = error.to_string();
            assert!(message.contains(name), "{message}");
            assert!(message.contains("unavailable"), "{message}");
        }
    }

    #[test]
    fn unavailable_backend_constructors_fail() {
        assert!(NCCLProcessGroup::new(0, 2, 0, "127.0.0.1".to_string(), 29600).is_err());
        assert!(GlooProcessGroup::new(0, 2, "127.0.0.1".to_string(), 29600).is_err());
        assert!(MPIProcessGroup::new(0, 2).is_err());

        for reason in [
            NCCLProcessGroup::unavailable_reason(),
            GlooProcessGroup::unavailable_reason(),
            MPIProcessGroup::unavailable_reason(),
        ] {
            assert!(
                reason.contains("Tcp"),
                "the error must point at the working backend: {reason}"
            );
        }
    }

    #[test]
    fn simulated_group_refuses_multi_rank_collectives_instead_of_no_oping() {
        let pg = SimulatedProcessGroup::new(1, 4);
        let original = vec![0.25f32, -0.5, 1.0];

        let mut tensor = Tensor::from_slice(&original, &[3]).expect("tensor must build in test");
        assert!(pg.broadcast(&mut tensor, 0).is_err());
        assert_eq!(
            tensor.to_vec_f32().expect("tensor read must succeed in test"),
            original,
            "a failed broadcast must not scale the tensor"
        );

        let mut tensors =
            vec![Tensor::from_slice(&original, &[3]).expect("tensor must build in test")];
        assert!(pg.all_reduce(&mut tensors).is_err());
        assert_eq!(
            tensors[0].to_vec_f32().expect("tensor read must succeed in test"),
            original
        );

        assert!(pg.reduce(&mut tensor, 0).is_err());
        assert!(pg.barrier().is_err());
    }

    #[test]
    fn detect_gpu_count_never_invents_devices() {
        // The previous implementation returned `Ok(8)` on any host without
        // `CUDA_VISIBLE_DEVICES`. The environment is read, never written, so
        // this test is safe to run in parallel with any other.
        match std::env::var("CUDA_VISIBLE_DEVICES") {
            Err(_) => {
                let error = detect_gpu_count()
                    .expect_err("no enumeration source means no device count in test");
                assert!(error.to_string().contains("no GPU devices"), "{error}");
            },
            Ok(devices) => {
                let expected = devices.split(',').filter(|entry| !entry.trim().is_empty()).count();
                if expected == 0 {
                    assert!(detect_gpu_count().is_err());
                } else {
                    assert_eq!(
                        detect_gpu_count().expect("a populated env var must parse in test"),
                        expected
                    );
                }
            },
        }
    }

    // ── Gradient compression is a real codec, not an inert flag ───────────

    #[test]
    fn topk_codec_keeps_exactly_the_largest_entries() {
        let mut values = vec![0.5f32, -0.01, 0.02, 0.03, -0.9, 0.04, 0.05, 0.06];
        GradientCompressionConfig::TopK { ratio: 0.25 }
            .apply(&mut values)
            .expect("codec must run in test");

        assert_eq!(values.iter().filter(|value| **value != 0.0).count(), 2);
        assert_eq!(values[0], 0.5);
        assert_eq!(values[4], -0.9);
    }

    #[test]
    fn quantize_codec_snaps_to_the_reconstruction_grid() {
        let original = vec![-1.0f32, -0.3, 0.2, 1.0];
        let mut values = original.clone();
        GradientCompressionConfig::Quantize { bits: 2 }
            .apply(&mut values)
            .expect("codec must run in test");

        // 2 bits over [-1, 1] gives levels {-1, -1/3, 1/3, 1}.
        assert_ne!(values, original, "quantization must move the values");
        let step = 2.0f32 / 3.0;
        for (quantized, raw) in values.iter().zip(&original) {
            let level = (quantized + 1.0) / step;
            assert!(
                (level - level.round()).abs() < 1e-4,
                "{quantized} is not on the grid"
            );
            assert!((quantized - raw).abs() <= step / 2.0 + 1e-5);
        }
    }

    #[test]
    fn codecs_reject_invalid_settings() {
        let mut values = vec![1.0f32; 4];
        assert!(GradientCompressionConfig::TopK { ratio: 0.0 }.apply(&mut values).is_err());
        assert!(GradientCompressionConfig::TopK { ratio: f32::NAN }.apply(&mut values).is_err());
        assert!(GradientCompressionConfig::Quantize { bits: 1 }.apply(&mut values).is_err());
        assert!(GradientCompressionConfig::Quantize { bits: 9 }.apply(&mut values).is_err());
    }

    #[test]
    fn enabling_gradient_compression_changes_the_synchronised_gradient() {
        // Magnitudes are chosen so the post-compression global norm stays below
        // the clipping threshold, isolating the codec's effect.
        let raw = vec![0.5f32, 0.01, 0.02, 0.03];

        let run = |compress: bool| -> Vec<f32> {
            let config = DistributedConfig {
                world_size: 1,
                rank: 0,
                backend: DistributedBackend::Simulated,
                master_addr: "localhost".to_string(),
                master_port: 29500,
                gradient_compression: compress,
                bucket_size_mb: 25,
            };
            let pg = Arc::new(SimulatedProcessGroup::new(0, 1));
            let trainer = DataParallelTrainer::new(DummyModel::new(), pg, config)
                .expect("trainer must build in test");

            let mut gradients = HashMap::new();
            gradients.insert(
                "w".to_string(),
                Tensor::from_slice(&raw, &[4]).expect("tensor must build in test"),
            );
            trainer.backward(&mut gradients).expect("backward must succeed in test");
            gradients["w"].to_vec_f32().expect("tensor read must succeed in test")
        };

        let dense = run(false);
        let compressed = run(true);

        assert_eq!(dense, raw, "without compression the gradient is untouched");
        assert_eq!(
            compressed,
            vec![0.5f32, 0.0, 0.0, 0.0],
            "10% top-k must keep one entry and zero the rest"
        );
    }

    #[test]
    fn bucket_plan_respects_the_byte_budget_and_is_rank_independent() {
        let config = DistributedConfig {
            world_size: 1,
            rank: 0,
            backend: DistributedBackend::Simulated,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            gradient_compression: false,
            bucket_size_mb: 1,
        };
        let pg = Arc::new(SimulatedProcessGroup::new(0, 1));
        let trainer = DataParallelTrainer::new(DummyModel::new(), pg, config)
            .expect("trainer must build in test");

        // Three tensors of 512 KiB each: two fit in a 1 MiB bucket, the third
        // starts a new one.
        let elements = 128 * 1024;
        let mut gradients = HashMap::new();
        for name in ["a", "b", "c"] {
            gradients.insert(
                name.to_string(),
                Tensor::zeros(&[elements]).expect("tensor must build in test"),
            );
        }

        let plan = trainer.bucket_plan(&gradients);
        assert_eq!(
            plan,
            vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["c".to_string()]
            ]
        );
    }

    // ── Real multi-rank data-parallel training ────────────────────────────

    #[test]
    fn data_parallel_trainer_averages_gradients_across_real_ranks() {
        use crate::distributed_collective::run_in_process;

        let per_rank = run_in_process(4, |rank, group| -> Vec<f32> {
            let config = DistributedConfig {
                world_size: 4,
                rank,
                backend: DistributedBackend::InProcess,
                master_addr: "localhost".to_string(),
                master_port: 29500,
                gradient_compression: false,
                bucket_size_mb: 25,
            };
            let pg: Arc<dyn ProcessGroup> = group;
            let trainer = DataParallelTrainer::new(DummyModel::new(), pg, config)
                .expect("trainer must build in test");

            let mut gradients = HashMap::new();
            gradients.insert(
                "w".to_string(),
                Tensor::from_slice(&[rank as f32, 2.0 * rank as f32], &[2])
                    .expect("tensor must build in test"),
            );
            // `backward` also clips; the mean here has norm < 1 so clipping is
            // the identity and the assertion isolates the collective.
            trainer.backward(&mut gradients).expect("backward must succeed in test");
            gradients["w"].to_vec_f32().expect("tensor read must succeed in test")
        })
        .expect("in-process run must succeed in test");

        // mean(0,1,2,3) = 1.5 -> norm of [1.5, 3.0] is > 1, so clipping scales
        // both by 1/norm. The ratio between the components is what the
        // collective determines, and every rank must agree exactly.
        for values in &per_rank {
            assert_eq!(
                values, &per_rank[0],
                "every rank must leave with the same gradient"
            );
            approx::assert_relative_eq!(values[1] / values[0], 2.0f32, epsilon = 1e-5);
        }
    }

    #[test]
    fn broadcast_parameters_requires_a_registry_and_then_synchronises() {
        use crate::distributed_collective::run_in_process;

        let config_for = |rank: usize| DistributedConfig {
            world_size: 3,
            rank,
            backend: DistributedBackend::InProcess,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            gradient_compression: false,
            bucket_size_mb: 25,
        };

        let per_rank = run_in_process(3, move |rank, group| -> Vec<f32> {
            let pg: Arc<dyn ProcessGroup> = group;
            let trainer = DataParallelTrainer::new(DummyModel::new(), pg, config_for(rank))
                .expect("trainer must build in test");

            assert!(
                trainer.broadcast_parameters().is_err(),
                "an empty registry must error, not invent parameters"
            );

            let mut parameters = HashMap::new();
            parameters.insert(
                "w".to_string(),
                Tensor::from_slice(&[rank as f32 + 1.0, 10.0 * (rank as f32 + 1.0)], &[2])
                    .expect("tensor must build in test"),
            );
            trainer
                .register_parameters(parameters)
                .expect("registration must succeed in test");

            let synchronized =
                trainer.broadcast_parameters().expect("broadcast must succeed in test");
            synchronized["w"].to_vec_f32().expect("tensor read must succeed in test")
        })
        .expect("in-process run must succeed in test");

        // Rank 0 owned [1, 10]; every rank must end up with exactly that.
        for values in &per_rank {
            assert_eq!(values, &vec![1.0f32, 10.0]);
        }
    }
}
