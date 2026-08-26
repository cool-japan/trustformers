//! Advanced GPU profiling and kernel optimization tools
//!
//! This module provides comprehensive GPU memory analysis, kernel optimization
//! suggestions, and advanced profiling capabilities for CUDA/ROCm/OpenCL kernels.
// reason: debug/profiling scaffolding — structs are constructed and their fields/methods
// are retained for the data model, serialization completeness, and future consumers that
// do not yet read every member. Consolidated from many item-level #[allow(dead_code)].
#![allow(dead_code)]

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// Advanced GPU memory profiler with fragmentation analysis
#[derive(Debug)]
pub struct AdvancedGpuMemoryProfiler {
    device_count: i32,
    memory_pools: HashMap<i32, GpuMemoryPool>,
    memory_allocations: HashMap<Uuid, GpuMemoryAllocation>,
    fragmentation_history: VecDeque<MemoryFragmentationSnapshot>,
    bandwidth_monitors: HashMap<i32, GpuBandwidthMonitor>,
    memory_pressure_monitor: MemoryPressureMonitor,
    cross_device_transfers: Vec<CrossDeviceTransfer>,
    /// Real host-OS telemetry handle (`sysinfo`), used ONLY to compute
    /// [`MemoryPressureSnapshot::swap_activity`] -- see that field's doc
    /// comment for why this is host-wide rather than per-GPU.
    system_info: sysinfo::System,
    /// Previous real `used_swap()` reading, so `swap_activity` can report
    /// a genuine delta instead of a single instantaneous level. `None`
    /// until the first [`Self::update_memory_pressure`] call.
    last_used_swap_bytes: Option<u64>,
}

/// GPU memory allocation with detailed tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryAllocation {
    pub allocation_id: Uuid,
    pub device_id: i32,
    pub size_bytes: usize,
    pub alignment: usize,
    pub memory_type: GpuMemoryType,
    pub allocation_context: AllocationContext,
    pub timestamp: SystemTime,
    pub freed: bool,
    pub free_timestamp: Option<SystemTime>,
    pub access_pattern: MemoryAccessPattern,
    pub usage_statistics: MemoryUsageStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuMemoryType {
    Global,
    Shared,
    Constant,
    Texture,
    Local,
    Unified,
    Pinned,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationContext {
    pub kernel_name: Option<String>,
    pub tensor_name: Option<String>,
    pub layer_name: Option<String>,
    pub allocation_source: AllocationSource,
    pub stack_trace: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationSource {
    TensorCreation,
    KernelLaunch,
    IntermediateBuffer,
    GradientBuffer,
    WeightBuffer,
    ActivationBuffer,
    CacheBuffer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAccessPattern {
    pub access_frequency: f64,
    pub read_ratio: f64,
    pub write_ratio: f64,
    pub sequential_access_ratio: f64,
    pub random_access_ratio: f64,
    pub coalesced_access_ratio: f64,
    pub cache_hit_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryUsageStats {
    pub total_accesses: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub lifetime_duration: Option<Duration>,
    pub peak_concurrent_usage: usize,
}

/// Memory fragmentation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFragmentationSnapshot {
    pub timestamp: DateTime<Utc>,
    pub device_id: i32,
    /// Capacity this pool was configured with -- see `GpuMemoryPool`'s
    /// own doc comment for how that number is obtained (this crate has no
    /// pure-Rust GPU memory query API).
    pub total_memory: usize,
    /// Real remaining capacity: `total_memory` minus the real sum of
    /// currently-outstanding allocations tracked via
    /// [`AdvancedGpuMemoryProfiler::track_allocation`] /
    /// [`AdvancedGpuMemoryProfiler::track_deallocation`] -- genuine
    /// bookkeeping from real call arguments, not fabricated.
    pub free_memory: usize,
    /// Size of the largest contiguous free block, when measurable. This
    /// pool tracks only a running free-BYTE COUNT, never the placement of
    /// individual allocations in address space, so it has no way to know
    /// whether that free capacity is one block or many small ones --
    /// `None`, never a claim that all free memory forms one contiguous
    /// block (the previous behavior).
    pub largest_free_block: Option<usize>,
    /// `None` for the same reason as [`Self::largest_free_block`]: real
    /// fragmentation is a function of block placement and allocator
    /// policy, neither of which this crate tracks or simulates.
    pub fragmentation_ratio: Option<f64>,
    /// Per-block free-space sizes, when known. `None` -- not an empty
    /// `Vec`, which would misleadingly read as "zero free blocks exist"
    /// -- see [`Self::largest_free_block`].
    pub free_block_distribution: Option<Vec<usize>>,
    /// `None` for the same reason as [`Self::fragmentation_ratio`].
    pub external_fragmentation: Option<f64>,
    /// `None` for the same reason as [`Self::fragmentation_ratio`].
    pub internal_fragmentation: Option<f64>,
}

/// GPU bandwidth monitoring
#[derive(Debug)]
pub struct GpuBandwidthMonitor {
    device_id: i32,
    bandwidth_samples: VecDeque<BandwidthSample>,
    theoretical_bandwidth: f64, // GB/s
    peak_observed_bandwidth: f64,
    sustained_bandwidth_history: Vec<SustainedBandwidthMeasurement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthSample {
    pub timestamp: SystemTime,
    pub memory_type: GpuMemoryType,
    pub operation_type: MemoryOperationType,
    pub bytes_transferred: usize,
    pub duration: Duration,
    pub achieved_bandwidth_gb_s: f64,
    pub efficiency_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryOperationType {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    KernelMemoryAccess,
    PeerToPeer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SustainedBandwidthMeasurement {
    pub duration: Duration,
    pub avg_bandwidth_gb_s: f64,
    pub min_bandwidth_gb_s: f64,
    pub max_bandwidth_gb_s: f64,
    pub bandwidth_variability: f64,
}

/// Memory pressure monitoring
#[derive(Debug)]
pub struct MemoryPressureMonitor {
    pressure_history: VecDeque<MemoryPressureSnapshot>,
    pressure_thresholds: MemoryPressureThresholds,
    auto_optimization_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureSnapshot {
    pub timestamp: DateTime<Utc>,
    pub device_id: i32,
    pub pressure_level: MemoryPressureLevel,
    pub available_memory_ratio: f64,
    pub allocation_rate: f64, // allocations per second
    pub deallocation_rate: f64,
    /// `None`: Rust has no garbage collector, and this crate has no hook
    /// into any framework-level GC, so there is no real signal to report
    /// here. Never a fabricated `0.0`.
    pub gc_pressure: Option<f64>,
    /// Real change in HOST OS swap usage (bytes, signed -- positive means
    /// swap grew) since the previous snapshot, read via `sysinfo`
    /// (already a workspace dependency; see
    /// `AdvancedGpuMemoryProfiler::last_used_swap_bytes`). `None` only
    /// for the very first snapshot, when there is no previous reading to
    /// diff against. This is deliberately HOST-wide, not
    /// `device_id`-scoped: no GPU vendor exposes a per-device "swap"
    /// concept to userspace, so a genuinely per-GPU number does not
    /// exist to measure. A single instantaneous reading would be a
    /// LEVEL, not "activity" -- this is a real delta, not a level
    /// wearing that name.
    pub swap_activity: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryPressureLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureThresholds {
    pub medium_threshold: f64, // 0.7 = 70% memory usage triggers medium pressure
    pub high_threshold: f64,   // 0.85 = 85% memory usage triggers high pressure
    pub critical_threshold: f64, // 0.95 = 95% memory usage triggers critical pressure
}

/// Cross-device memory transfer tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDeviceTransfer {
    pub transfer_id: Uuid,
    pub source_device: i32,
    pub target_device: i32,
    pub bytes_transferred: usize,
    pub transfer_type: CrossDeviceTransferType,
    pub duration: Duration,
    pub bandwidth_achieved: f64,
    /// Whether this transfer used peer-to-peer DMA. `None` when unknown:
    /// this crate has no pure-Rust API to query real GPU P2P capability
    /// (see `AdvancedGpuMemoryProfiler::detect_p2p_capability`) -- never
    /// a guessed `true`.
    pub p2p_enabled: Option<bool>,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossDeviceTransferType {
    DirectMemoryAccess,
    PeerToPeer,
    HostBounced,
    NvLink,
    Infinity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelExecutionProfile {
    pub kernel_name: String,
    pub execution_count: usize,
    pub total_execution_time: Duration,
    pub avg_execution_time: Duration,
    pub min_execution_time: Duration,
    pub max_execution_time: Duration,
    pub grid_sizes: Vec<(u32, u32, u32)>,
    pub block_sizes: Vec<(u32, u32, u32)>,
    pub shared_memory_usage: Vec<usize>,
    pub register_usage: Vec<u32>,
    pub occupancy_measurements: Vec<f64>,
    pub compute_utilization: Vec<f64>,
    pub memory_bandwidth_utilization: Vec<f64>,
    pub warp_efficiency: Vec<f64>,
    pub memory_efficiency: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelOptimization {
    pub optimization_type: OptimizationType,
    pub current_value: OptimizationValue,
    pub suggested_value: OptimizationValue,
    pub expected_improvement: ExpectedImprovement,
    pub confidence: f64,
    pub explanation: String,
    pub implementation_difficulty: ImplementationDifficulty,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    BlockSize,
    GridSize,
    SharedMemory,
    RegisterOptimization,
    MemoryCoalescing,
    WarpDivergence,
    KernelFusion,
    MemoryLayoutOptimization,
    ComputeIntensityBalance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationValue {
    IntegerValue(u32),
    FloatValue(f64),
    TupleValue((u32, u32, u32)),
    LayoutPattern(String),
    BooleanValue(bool),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedImprovement {
    pub performance_gain_percentage: f64,
    pub memory_usage_reduction_percentage: f64,
    pub energy_efficiency_improvement: f64,
    pub scalability_improvement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationDifficulty {
    Trivial,
    Easy,
    Moderate,
    Difficult,
    Expert,
}

/// Launch configuration analysis
#[derive(Debug)]
pub struct LaunchConfigAnalyzer {
    optimal_configs: HashMap<String, OptimalLaunchConfig>,
    config_performance_history: HashMap<String, Vec<ConfigPerformanceMeasurement>>,
    autotuning_enabled: bool,
    search_space_cache: HashMap<String, LaunchConfigSearchSpace>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaunchConfigSearchSpace {
    pub kernel_name: String,
    pub min_block_size: (u32, u32, u32),
    pub max_block_size: (u32, u32, u32),
    pub min_grid_size: (u32, u32, u32),
    pub max_grid_size: (u32, u32, u32),
    pub min_shared_memory: usize,
    pub max_shared_memory: usize,
    pub search_constraints: Vec<LaunchConstraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalLaunchConfig {
    pub kernel_name: String,
    pub optimal_block_size: (u32, u32, u32),
    pub optimal_grid_size: (u32, u32, u32),
    pub optimal_shared_memory: usize,
    pub expected_occupancy: f64,
    pub expected_performance: f64,
    pub constraints: Vec<LaunchConstraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigPerformanceMeasurement {
    pub block_size: (u32, u32, u32),
    pub grid_size: (u32, u32, u32),
    pub shared_memory: usize,
    pub achieved_occupancy: f64,
    pub execution_time: Duration,
    pub memory_bandwidth: f64,
    pub compute_utilization: f64,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LaunchConstraint {
    MaxSharedMemory(usize),
    MaxRegisters(u32),
    MinOccupancy(f64),
    WorkgroupSizeLimit(u32),
    MemoryBandwidthLimit(f64),
}

/// Memory access pattern analysis
#[derive(Debug)]
pub struct MemoryAccessAnalyzer {
    access_patterns: HashMap<String, MemoryAccessAnalysis>,
    coalescing_analysis: HashMap<String, CoalescingAnalysis>,
    cache_performance: HashMap<String, CachePerformanceAnalysis>,
    stride_analysis: HashMap<String, StrideAnalysisResult>,
    bank_conflict_analyzer: BankConflictAnalyzer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrideAnalysisResult {
    pub kernel_name: String,
    pub average_stride: f64,
    pub stride_pattern: StridePattern,
    pub optimization_potential: f64,
    pub recommended_changes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StridePattern {
    Sequential,
    Strided(i32),
    Random,
    Broadcast,
}

#[derive(Debug)]
pub struct BankConflictAnalyzer {
    conflict_patterns: HashMap<String, BankConflictPattern>,
    resolution_strategies: HashMap<String, Vec<ConflictResolutionStrategy>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BankConflictPattern {
    pub kernel_name: String,
    pub conflicts_detected: usize,
    pub conflict_severity: ConflictSeverity,
    pub affected_warps: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResolutionStrategy {
    pub strategy_type: ResolutionStrategyType,
    pub description: String,
    pub expected_improvement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionStrategyType {
    DataPadding,
    AccessReordering,
    SharedMemoryBanking,
    AlgorithmicChange,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAccessAnalysis {
    pub kernel_name: String,
    pub total_memory_transactions: u64,
    pub coalesced_transactions: u64,
    pub uncoalesced_transactions: u64,
    pub stride_patterns: Vec<StridePattern>,
    pub access_locality: AccessLocalityMetrics,
    pub bank_conflicts: u64,
    pub cache_line_utilization: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedStride {
    pub stride_size: usize,
    pub frequency: u64,
    pub efficiency_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessLocalityMetrics {
    pub temporal_locality_score: f64,
    pub spatial_locality_score: f64,
    pub working_set_size: usize,
    pub reuse_distance_avg: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoalescingAnalysis {
    pub kernel_name: String,
    pub coalescing_efficiency: f64,
    pub uncoalesced_regions: Vec<UncoalescedRegion>,
    pub suggested_improvements: Vec<CoalescingImprovement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncoalescedRegion {
    pub memory_region: String,
    pub access_pattern: String,
    pub efficiency_loss: f64,
    pub fix_difficulty: ImplementationDifficulty,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoalescingImprovement {
    pub improvement_type: CoalescingImprovementType,
    pub description: String,
    pub expected_speedup: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoalescingImprovementType {
    DataLayoutReorganization,
    AccessPatternOptimization,
    SharedMemoryBuffering,
    VectorizedAccess,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePerformanceAnalysis {
    pub kernel_name: String,
    pub l1_cache_hit_rate: f64,
    pub l2_cache_hit_rate: f64,
    pub texture_cache_hit_rate: f64,
    pub shared_memory_bank_conflicts: u64,
    pub cache_thrashing_detected: bool,
    pub recommended_cache_optimizations: Vec<CacheOptimization>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheOptimization {
    pub optimization_type: CacheOptimizationType,
    pub description: String,
    pub expected_improvement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheOptimizationType {
    DataPrefetching,
    CacheBlockingStrategy,
    SharedMemoryUsage,
    TextureMemoryUsage,
    ConstantMemoryUsage,
}

/// Compute utilization analysis
#[derive(Debug)]
pub struct ComputeUtilizationAnalyzer {
    utilization_profiles: HashMap<String, ComputeUtilizationProfile>,
    bottleneck_analysis: HashMap<String, ComputeBottleneckAnalysis>,
    arithmetic_intensity_analyzer: ArithmeticIntensityAnalyzer,
    resource_balancer: ResourceBalancer,
}

#[derive(Debug)]
pub struct ArithmeticIntensityAnalyzer {
    intensity_profiles: HashMap<String, ArithmeticIntensityProfile>,
    roofline_models: HashMap<i32, RooflineModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArithmeticIntensityProfile {
    pub kernel_name: String,
    pub arithmetic_intensity: f64,
    pub operations_per_byte: f64,
    pub peak_performance_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RooflineModel {
    pub device_id: i32,
    pub peak_compute_flops: f64,
    pub peak_memory_bandwidth: f64,
    pub ridge_point: f64,
}

#[derive(Debug)]
pub struct ResourceBalancer {
    resource_profiles: HashMap<String, ResourceProfile>,
    balancing_strategies: HashMap<String, BalancingStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceProfile {
    pub kernel_name: String,
    pub register_usage: f64,
    pub shared_memory_usage: f64,
    pub occupancy: f64,
    pub limiting_factor: LimitingFactor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LimitingFactor {
    Registers,
    SharedMemory,
    Blocks,
    Warps,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalancingStrategy {
    pub strategy_name: String,
    pub description: String,
    pub expected_improvement: f64,
    pub trade_offs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeUtilizationProfile {
    pub kernel_name: String,
    pub arithmetic_intensity: f64,
    pub compute_throughput: f64,
    pub memory_throughput: f64,
    pub compute_to_memory_ratio: f64,
    pub warp_execution_efficiency: f64,
    pub instruction_mix: InstructionMixAnalysis,
    pub resource_utilization: ResourceUtilizationMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionMixAnalysis {
    pub integer_ops_percentage: f64,
    pub float_ops_percentage: f64,
    pub double_ops_percentage: f64,
    pub special_function_ops_percentage: f64,
    pub memory_ops_percentage: f64,
    pub control_flow_ops_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationMetrics {
    pub register_utilization: f64,
    pub shared_memory_utilization: f64,
    pub constant_memory_utilization: f64,
    pub texture_cache_utilization: f64,
    pub compute_unit_utilization: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeBottleneckAnalysis {
    pub kernel_name: String,
    pub primary_bottleneck: ComputeBottleneckType,
    pub bottleneck_severity: f64,
    pub contributing_factors: Vec<BottleneckFactor>,
    pub optimization_opportunities: Vec<ComputeOptimizationOpportunity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeBottleneckType {
    MemoryBandwidth,
    ComputeThroughput,
    Latency,
    Occupancy,
    WarpDivergence,
    SynchronizationOverhead,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckFactor {
    pub factor_type: String,
    pub impact_percentage: f64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeOptimizationOpportunity {
    pub opportunity_type: ComputeOptimizationType,
    pub description: String,
    pub expected_speedup: f64,
    pub implementation_effort: ImplementationDifficulty,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeOptimizationType {
    KernelFusion,
    MemoryOptimization,
    ParallelismIncrease,
    AlgorithmicImprovement,
    ResourceBalancing,
}

impl AdvancedGpuMemoryProfiler {
    pub fn new(device_count: i32) -> Result<Self> {
        let mut memory_pools = HashMap::new();
        let mut bandwidth_monitors = HashMap::new();

        for device_id in 0..device_count {
            memory_pools.insert(device_id, GpuMemoryPool::new(device_id)?);
            bandwidth_monitors.insert(device_id, GpuBandwidthMonitor::new(device_id)?);
        }

        Ok(Self {
            device_count,
            memory_pools,
            memory_allocations: HashMap::new(),
            fragmentation_history: VecDeque::with_capacity(1000),
            bandwidth_monitors,
            memory_pressure_monitor: MemoryPressureMonitor::new(),
            cross_device_transfers: Vec::new(),
            system_info: sysinfo::System::new(),
            last_used_swap_bytes: None,
        })
    }

    /// Track a GPU memory allocation with detailed context
    pub fn track_allocation(
        &mut self,
        device_id: i32,
        size_bytes: usize,
        memory_type: GpuMemoryType,
        context: AllocationContext,
    ) -> Result<Uuid> {
        let allocation_id = Uuid::new_v4();
        let allocation = GpuMemoryAllocation {
            allocation_id,
            device_id,
            size_bytes,
            alignment: self.calculate_optimal_alignment(size_bytes),
            memory_type,
            allocation_context: context,
            timestamp: SystemTime::now(),
            freed: false,
            free_timestamp: None,
            access_pattern: MemoryAccessPattern::default(),
            usage_statistics: MemoryUsageStats::default(),
        };

        // Update memory pool
        if let Some(pool) = self.memory_pools.get_mut(&device_id) {
            pool.allocate(size_bytes)?;
        }

        self.memory_allocations.insert(allocation_id, allocation);

        // Check for memory pressure
        self.update_memory_pressure(device_id);

        Ok(allocation_id)
    }

    /// Track memory deallocation
    pub fn track_deallocation(&mut self, allocation_id: Uuid) -> Result<()> {
        let device_id = if let Some(allocation) = self.memory_allocations.get_mut(&allocation_id) {
            allocation.freed = true;
            allocation.free_timestamp = Some(SystemTime::now());

            // Get the device_id and size_bytes before dropping the mutable reference
            let device_id = allocation.device_id;
            let size_bytes = allocation.size_bytes;

            // Update memory pool
            if let Some(pool) = self.memory_pools.get_mut(&device_id) {
                pool.deallocate(size_bytes)?;
            }

            Some(device_id)
        } else {
            None
        };

        // Update memory pressure after dropping the mutable reference
        if let Some(device_id) = device_id {
            self.update_memory_pressure(device_id);
        }

        Ok(())
    }

    /// Analyze memory fragmentation across all devices
    pub fn analyze_fragmentation(&mut self) -> Result<Vec<MemoryFragmentationSnapshot>> {
        let mut snapshots = Vec::new();

        for (&_device_id, pool) in &self.memory_pools {
            let snapshot = pool.get_fragmentation_snapshot()?;
            snapshots.push(snapshot.clone());

            // Store in history
            self.fragmentation_history.push_back(snapshot);
            if self.fragmentation_history.len() > 1000 {
                self.fragmentation_history.pop_front();
            }
        }

        Ok(snapshots)
    }

    /// Monitor memory bandwidth utilization
    pub fn record_bandwidth_sample(
        &mut self,
        device_id: i32,
        sample: BandwidthSample,
    ) -> Result<()> {
        if let Some(monitor) = self.bandwidth_monitors.get_mut(&device_id) {
            monitor.add_sample(sample)?;
        }
        Ok(())
    }

    /// Track cross-device memory transfer
    pub fn track_cross_device_transfer(
        &mut self,
        source_device: i32,
        target_device: i32,
        bytes_transferred: usize,
        transfer_type: CrossDeviceTransferType,
        duration: Duration,
    ) -> Result<Uuid> {
        let transfer_id = Uuid::new_v4();
        let bandwidth_achieved =
            bytes_transferred as f64 / (1024.0 * 1024.0 * 1024.0) / duration.as_secs_f64();

        let transfer = CrossDeviceTransfer {
            transfer_id,
            source_device,
            target_device,
            bytes_transferred,
            transfer_type,
            duration,
            bandwidth_achieved,
            p2p_enabled: self.detect_p2p_capability(source_device, target_device),
            timestamp: SystemTime::now(),
        };

        self.cross_device_transfers.push(transfer);
        Ok(transfer_id)
    }

    /// Get comprehensive memory analysis report
    pub fn get_memory_analysis_report(&self) -> MemoryAnalysisReport {
        let fragmentation_summary = self.analyze_fragmentation_trends();
        let bandwidth_summary = self.analyze_bandwidth_utilization();
        let pressure_summary = self.analyze_memory_pressure();
        let allocation_summary = self.analyze_allocation_patterns();
        let cross_device_summary = self.analyze_cross_device_transfers();

        MemoryAnalysisReport {
            fragmentation_summary,
            bandwidth_summary,
            pressure_summary,
            allocation_summary,
            cross_device_summary,
            optimization_recommendations: self.generate_memory_optimization_recommendations(),
        }
    }

    fn calculate_optimal_alignment(&self, size_bytes: usize) -> usize {
        // Calculate optimal memory alignment for GPU access
        if size_bytes >= 128 {
            128 // Cache line alignment
        } else if size_bytes >= 64 {
            64
        } else if size_bytes >= 32 {
            32
        } else {
            16
        }
    }

    fn update_memory_pressure(&mut self, device_id: i32) {
        let Some((pressure_level, available_memory_ratio)) =
            self.memory_pools.get(&device_id).map(|pool| {
                (
                    pool.calculate_pressure_level(),
                    pool.get_available_memory_ratio(),
                )
            })
        else {
            return;
        };
        let allocation_rate = self.calculate_allocation_rate(device_id);
        let deallocation_rate = self.calculate_deallocation_rate(device_id);
        let swap_activity = self.compute_swap_activity_delta();

        let pressure_snapshot = MemoryPressureSnapshot {
            timestamp: Utc::now(),
            device_id,
            pressure_level,
            available_memory_ratio,
            allocation_rate,
            deallocation_rate,
            // Rust has no garbage collector and this crate has no hook
            // into any framework-level GC -- see the field's own doc
            // comment.
            gc_pressure: None,
            swap_activity,
        };

        self.memory_pressure_monitor.add_snapshot(pressure_snapshot);
    }

    /// Real change in HOST OS swap usage in bytes since the previous call
    /// (positive = swap grew), via `sysinfo` (already a workspace
    /// dependency -- same pattern as `realtime_dashboard.rs`). `None` only
    /// on the very first call, when there is no previous reading yet. See
    /// [`MemoryPressureSnapshot::swap_activity`] for the host-wide-not-
    /// per-GPU caveat.
    fn compute_swap_activity_delta(&mut self) -> Option<f64> {
        self.system_info.refresh_memory();
        let used = self.system_info.used_swap();
        let delta = self.last_used_swap_bytes.map(|prev| used as f64 - prev as f64);
        self.last_used_swap_bytes = Some(used);
        delta
    }

    /// Whether `source`/`target` support peer-to-peer DMA, when knowable.
    /// This crate has no pure-Rust GPU capability query API (a real one
    /// would need vendor FFI -- CUDA/ROCm/NVML -- which the COOLJAPAN
    /// pure-Rust policy keeps out of the default build), so there is
    /// nothing to honestly detect here today: always `None`, never a
    /// guessed `true`.
    fn detect_p2p_capability(&self, _source: i32, _target: i32) -> Option<bool> {
        None
    }

    fn calculate_allocation_rate(&self, device_id: i32) -> f64 {
        // Calculate allocations per second for the device
        let recent_allocations = self
            .memory_allocations
            .values()
            .filter(|a| a.device_id == device_id)
            .filter(|a| a.timestamp.elapsed().unwrap_or_default().as_secs() < 60)
            .count();

        recent_allocations as f64 / 60.0
    }

    fn calculate_deallocation_rate(&self, device_id: i32) -> f64 {
        // Calculate deallocations per second for the device
        let recent_deallocations = self
            .memory_allocations
            .values()
            .filter(|a| a.device_id == device_id && a.freed)
            .filter(|a| {
                if let Some(free_time) = a.free_timestamp {
                    free_time.elapsed().unwrap_or_default().as_secs() < 60
                } else {
                    false
                }
            })
            .count();

        recent_deallocations as f64 / 60.0
    }

    fn analyze_fragmentation_trends(&self) -> FragmentationSummary {
        // Analyze fragmentation trends from history
        FragmentationSummary::new(&self.fragmentation_history)
    }

    fn analyze_bandwidth_utilization(&self) -> BandwidthSummary {
        BandwidthSummary::new(&self.bandwidth_monitors)
    }

    fn analyze_memory_pressure(&self) -> MemoryPressureSummary {
        self.memory_pressure_monitor.get_summary()
    }

    fn analyze_allocation_patterns(&self) -> AllocationPatternSummary {
        AllocationPatternSummary::new(&self.memory_allocations)
    }

    fn analyze_cross_device_transfers(&self) -> CrossDeviceTransferSummary {
        CrossDeviceTransferSummary::new(&self.cross_device_transfers)
    }

    fn generate_memory_optimization_recommendations(
        &self,
    ) -> Vec<MemoryOptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Analyze fragmentation and suggest optimizations, when this
        // pool's fragmentation was actually measurable for that snapshot
        // -- see `MemoryFragmentationSnapshot::fragmentation_ratio`'s doc
        // comment. No allocator/placement model exists in this crate
        // today, so this loop is currently a no-op in practice; it is
        // still real code, ready the moment a real ratio is ever
        // populated, rather than fabricating one to keep it "working".
        for snapshot in self.fragmentation_history.iter().take(10) {
            let Some(ratio) = snapshot.fragmentation_ratio else {
                continue;
            };
            if ratio > 0.3 {
                recommendations.push(MemoryOptimizationRecommendation {
                    recommendation_type: MemoryOptimizationType::DefragmentationStrategy,
                    priority: OptimizationPriority::High,
                    description: format!(
                        "High fragmentation detected on device {}: {:.1}%",
                        snapshot.device_id,
                        ratio * 100.0
                    ),
                    expected_benefit: ExpectedBenefit {
                        performance_improvement: 15.0,
                        memory_efficiency_improvement: 25.0,
                        implementation_effort: ImplementationDifficulty::Moderate,
                    },
                    implementation_steps: vec![
                        "Implement memory pooling with fixed-size blocks".to_string(),
                        "Add periodic defragmentation during idle periods".to_string(),
                        "Consider memory compaction strategies".to_string(),
                    ],
                });
            }
        }

        recommendations
    }
}

// Helper structures for analysis reports

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysisReport {
    pub fragmentation_summary: FragmentationSummary,
    pub bandwidth_summary: BandwidthSummary,
    pub pressure_summary: MemoryPressureSummary,
    pub allocation_summary: AllocationPatternSummary,
    pub cross_device_summary: CrossDeviceTransferSummary,
    pub optimization_recommendations: Vec<MemoryOptimizationRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationSummary {
    /// Mean of the real (`Some`) fragmentation ratios recorded across the
    /// summarised history. `None` when none of that history carries a
    /// measured ratio -- see
    /// [`MemoryFragmentationSnapshot::fragmentation_ratio`]. Never a
    /// fabricated `0.1`.
    pub avg_fragmentation_ratio: Option<f64>,
    pub peak_fragmentation_ratio: Option<f64>,
    pub fragmentation_trend: FragmentationTrend,
    pub most_fragmented_device: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FragmentationTrend {
    Improving,
    Stable,
    Worsening,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthSummary {
    /// A "utilization" ratio needs a real theoretical peak bandwidth to
    /// divide by; this crate has no pure-Rust GPU capability query for
    /// one (see [`GpuBandwidthMonitor`]'s `theoretical_bandwidth`, itself
    /// a documented assumption, not a measurement), so honestly `None`
    /// rather than a ratio against an invented denominator.
    pub avg_bandwidth_utilization: Option<f64>,
    /// Real maximum `achieved_bandwidth_gb_s` across all recorded
    /// samples on every device. `0.0`, not `None`, when no sample has
    /// ever been recorded -- a genuine "nothing observed yet".
    pub peak_bandwidth_achieved: f64,
    pub bandwidth_efficiency_by_operation: HashMap<String, f64>,
    pub underutilized_devices: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureSummary {
    pub current_pressure_levels: HashMap<i32, MemoryPressureLevel>,
    pub pressure_trend: PressureTrend,
    pub devices_under_pressure: Vec<i32>,
    pub time_in_high_pressure: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PressureTrend {
    Decreasing,
    Stable,
    Increasing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPatternSummary {
    pub total_allocations: usize,
    pub avg_allocation_size: usize,
    pub largest_allocation: usize,
    pub allocation_size_distribution: HashMap<String, usize>,
    pub memory_leaks_detected: usize,
    pub allocation_hot_spots: Vec<AllocationHotSpot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationHotSpot {
    pub location: String,
    pub allocation_frequency: f64,
    pub total_memory_allocated: usize,
    pub avg_allocation_lifetime: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDeviceTransferSummary {
    pub total_transfers: usize,
    pub total_bytes_transferred: usize,
    pub avg_transfer_bandwidth: f64,
    /// `None`: computing this needs to know which transfers were really
    /// peer-to-peer, which this crate cannot determine -- see
    /// [`CrossDeviceTransfer::p2p_enabled`] / `detect_p2p_capability`.
    pub p2p_efficiency: Option<f64>,
    pub transfer_bottlenecks: Vec<TransferBottleneck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferBottleneck {
    pub device_pair: (i32, i32),
    pub bottleneck_type: TransferBottleneckType,
    pub impact_severity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferBottleneckType {
    BandwidthLimited,
    LatencyBound,
    SynchronizationOverhead,
    P2PNotAvailable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationRecommendation {
    pub recommendation_type: MemoryOptimizationType,
    pub priority: OptimizationPriority,
    pub description: String,
    pub expected_benefit: ExpectedBenefit,
    pub implementation_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryOptimizationType {
    DefragmentationStrategy,
    MemoryPoolingOptimization,
    AllocationPatternOptimization,
    CrossDeviceTransferOptimization,
    PressureReliefStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedBenefit {
    pub performance_improvement: f64,
    pub memory_efficiency_improvement: f64,
    pub implementation_effort: ImplementationDifficulty,
}

// Default implementations for helper structures

impl Default for MemoryAccessPattern {
    fn default() -> Self {
        Self {
            access_frequency: 0.0,
            read_ratio: 0.5,
            write_ratio: 0.5,
            sequential_access_ratio: 0.8,
            random_access_ratio: 0.2,
            coalesced_access_ratio: 0.9,
            cache_hit_rate: 0.85,
        }
    }
}

// Constructors and helpers for the remaining structures.

/// This crate has no pure-Rust API to query a real GPU's memory capacity
/// (a real query needs vendor FFI -- CUDA/ROCm/Metal -- kept out of the
/// default build by the COOLJAPAN pure-Rust policy). [`GpuMemoryPool::new`]
/// therefore uses this ASSUMED capacity as a documented fallback rather
/// than silently pretending to have queried real hardware. It is real
/// bookkeeping arithmetic from here on (`allocate`/`deallocate` track
/// genuine caller-supplied sizes against it), just seeded from an
/// assumption instead of a measurement.
const ASSUMED_DEVICE_MEMORY_BYTES: usize = 8 * 1024 * 1024 * 1024; // 8GB

impl GpuMemoryPool {
    fn new(device_id: i32) -> Result<Self> {
        Ok(Self {
            device_id,
            total_memory: ASSUMED_DEVICE_MEMORY_BYTES,
            free_memory: ASSUMED_DEVICE_MEMORY_BYTES,
        })
    }

    fn allocate(&mut self, size: usize) -> Result<()> {
        if self.free_memory >= size {
            self.free_memory -= size;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Insufficient memory"))
        }
    }

    fn deallocate(&mut self, size: usize) -> Result<()> {
        self.free_memory += size;
        Ok(())
    }

    /// `total_memory`/`free_memory` are real (see their own doc comments
    /// on [`MemoryFragmentationSnapshot`]); the block-placement fields are
    /// honestly `None` -- this pool tracks a free-byte COUNT only, never
    /// individual allocation placement, so it cannot know the true shape
    /// of its free space.
    fn get_fragmentation_snapshot(&self) -> Result<MemoryFragmentationSnapshot> {
        Ok(MemoryFragmentationSnapshot {
            timestamp: Utc::now(),
            device_id: self.device_id,
            total_memory: self.total_memory,
            free_memory: self.free_memory,
            largest_free_block: None,
            fragmentation_ratio: None,
            free_block_distribution: None,
            external_fragmentation: None,
            internal_fragmentation: None,
        })
    }

    fn calculate_pressure_level(&self) -> MemoryPressureLevel {
        let usage_ratio = 1.0 - (self.free_memory as f64 / self.total_memory as f64);

        if usage_ratio > 0.95 {
            MemoryPressureLevel::Critical
        } else if usage_ratio > 0.85 {
            MemoryPressureLevel::High
        } else if usage_ratio > 0.70 {
            MemoryPressureLevel::Medium
        } else {
            MemoryPressureLevel::Low
        }
    }

    fn get_available_memory_ratio(&self) -> f64 {
        self.free_memory as f64 / self.total_memory as f64
    }
}

impl GpuBandwidthMonitor {
    fn new(device_id: i32) -> Result<Self> {
        Ok(Self {
            device_id,
            bandwidth_samples: VecDeque::with_capacity(1000),
            theoretical_bandwidth: 900.0, // GB/s for high-end GPU
            peak_observed_bandwidth: 0.0,
            sustained_bandwidth_history: Vec::new(),
        })
    }

    fn add_sample(&mut self, sample: BandwidthSample) -> Result<()> {
        if sample.achieved_bandwidth_gb_s > self.peak_observed_bandwidth {
            self.peak_observed_bandwidth = sample.achieved_bandwidth_gb_s;
        }

        self.bandwidth_samples.push_back(sample);
        if self.bandwidth_samples.len() > 1000 {
            self.bandwidth_samples.pop_front();
        }

        Ok(())
    }
}

impl MemoryPressureMonitor {
    fn new() -> Self {
        Self {
            pressure_history: VecDeque::with_capacity(1000),
            pressure_thresholds: MemoryPressureThresholds {
                medium_threshold: 0.7,
                high_threshold: 0.85,
                critical_threshold: 0.95,
            },
            auto_optimization_enabled: true,
        }
    }

    fn add_snapshot(&mut self, snapshot: MemoryPressureSnapshot) {
        self.pressure_history.push_back(snapshot);
        if self.pressure_history.len() > 1000 {
            self.pressure_history.pop_front();
        }
    }

    /// Real per-device most-recent level, real pressure trend, real
    /// devices-under-pressure list and a real (attributed-by-gap)
    /// time-in-high-pressure -- all from `self.pressure_history`, which
    /// this crate already collects on every real `track_allocation`/
    /// `track_deallocation` call. Previously discarded its own real
    /// history entirely (`_history` was never read).
    fn get_summary(&self) -> MemoryPressureSummary {
        // `pressure_history` is a `VecDeque` filled via `push_back`, so
        // iterating oldest->newest and letting each insert overwrite the
        // last correctly leaves the MOST RECENT snapshot per device.
        let mut current_pressure_levels: HashMap<i32, MemoryPressureLevel> = HashMap::new();
        for snapshot in &self.pressure_history {
            current_pressure_levels.insert(snapshot.device_id, snapshot.pressure_level.clone());
        }

        let devices_under_pressure: Vec<i32> = current_pressure_levels
            .iter()
            .filter(|(_, level)| {
                matches!(
                    level,
                    MemoryPressureLevel::High | MemoryPressureLevel::Critical
                )
            })
            .map(|(&device_id, _)| device_id)
            .collect();

        // Real trend: mean pressure ordinal of the recent half of history
        // vs. the older half -- the same "insufficient data -> Stable"
        // convention used by `GradientAnomalyDetector::analyze_recent_trend`
        // elsewhere in this crate.
        let pressure_trend = if self.pressure_history.len() < 4 {
            PressureTrend::Stable
        } else {
            let ordinals: Vec<f64> = self
                .pressure_history
                .iter()
                .map(|s| pressure_ordinal(&s.pressure_level))
                .collect();
            let mid = ordinals.len() / 2;
            let older_avg = ordinals[..mid].iter().sum::<f64>() / mid as f64;
            let recent_avg = ordinals[mid..].iter().sum::<f64>() / (ordinals.len() - mid) as f64;
            let trend_threshold = 0.25;
            if recent_avg > older_avg + trend_threshold {
                PressureTrend::Increasing
            } else if recent_avg < older_avg - trend_threshold {
                PressureTrend::Decreasing
            } else {
                PressureTrend::Stable
            }
        };

        // Real time spent at High/Critical: attribute each real
        // inter-snapshot gap (real timestamps) to the level held at the
        // START of that gap, per device, and sum the gaps that started
        // High or Critical.
        let mut by_device: HashMap<i32, Vec<&MemoryPressureSnapshot>> = HashMap::new();
        for snapshot in &self.pressure_history {
            by_device.entry(snapshot.device_id).or_default().push(snapshot);
        }
        let mut time_in_high_pressure = Duration::from_secs(0);
        for snapshots in by_device.values_mut() {
            snapshots.sort_by_key(|s| s.timestamp);
            for pair in snapshots.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                if matches!(
                    a.pressure_level,
                    MemoryPressureLevel::High | MemoryPressureLevel::Critical
                ) {
                    if let Ok(gap) = (b.timestamp - a.timestamp).to_std() {
                        time_in_high_pressure += gap;
                    }
                }
            }
        }

        MemoryPressureSummary {
            current_pressure_levels,
            pressure_trend,
            devices_under_pressure,
            time_in_high_pressure,
        }
    }
}

/// Ordinal encoding of [`MemoryPressureLevel`] for trend averaging (higher
/// = more pressure). An internal convenience, not a claim of any inherent
/// numeric scale in the real telemetry.
fn pressure_ordinal(level: &MemoryPressureLevel) -> f64 {
    match level {
        MemoryPressureLevel::Low => 0.0,
        MemoryPressureLevel::Medium => 1.0,
        MemoryPressureLevel::High => 2.0,
        MemoryPressureLevel::Critical => 3.0,
    }
}

// Real summary aggregation from the real data each analyzer already
// collects (previously discarded via an unused `_history`/`_monitors`/
// `_allocations`/`_transfers` parameter in every one of the four `new`s
// below).

impl FragmentationSummary {
    fn new(history: &VecDeque<MemoryFragmentationSnapshot>) -> Self {
        let measured: Vec<(i32, f64)> = history
            .iter()
            .filter_map(|s| s.fragmentation_ratio.map(|r| (s.device_id, r)))
            .collect();

        let Some(&(first_device, _)) = measured.first() else {
            // No snapshot in this history has ever carried a real
            // fragmentation ratio -- see that field's own doc comment.
            // Honestly absent, never the old `0.1`/`0.2` constants.
            return Self {
                avg_fragmentation_ratio: None,
                peak_fragmentation_ratio: None,
                fragmentation_trend: FragmentationTrend::Stable,
                most_fragmented_device: None,
            };
        };

        let avg = measured.iter().map(|(_, r)| r).sum::<f64>() / measured.len() as f64;
        let (peak_device, peak_ratio) =
            measured.iter().fold((first_device, f64::MIN), |(bd, br), &(d, r)| {
                if r > br {
                    (d, r)
                } else {
                    (bd, br)
                }
            });

        let fragmentation_trend = if measured.len() < 4 {
            FragmentationTrend::Stable
        } else {
            let mid = measured.len() / 2;
            let older_avg = measured[..mid].iter().map(|(_, r)| r).sum::<f64>() / mid as f64;
            let recent_avg =
                measured[mid..].iter().map(|(_, r)| r).sum::<f64>() / (measured.len() - mid) as f64;
            let trend_threshold = 0.05;
            if recent_avg > older_avg + trend_threshold {
                FragmentationTrend::Worsening
            } else if recent_avg < older_avg - trend_threshold {
                FragmentationTrend::Improving
            } else {
                FragmentationTrend::Stable
            }
        };

        Self {
            avg_fragmentation_ratio: Some(avg),
            peak_fragmentation_ratio: Some(peak_ratio),
            fragmentation_trend,
            most_fragmented_device: Some(peak_device),
        }
    }
}

impl BandwidthSummary {
    fn new(monitors: &HashMap<i32, GpuBandwidthMonitor>) -> Self {
        let peak_bandwidth_achieved =
            monitors.values().map(|m| m.peak_observed_bandwidth).fold(0.0_f64, f64::max);

        // Real average `efficiency_percentage` per real operation type --
        // each sample's value comes from whatever called
        // `record_bandwidth_sample`, not fabricated by this crate.
        let mut efficiency_sum: HashMap<String, (f64, usize)> = HashMap::new();
        for monitor in monitors.values() {
            for sample in &monitor.bandwidth_samples {
                let entry = efficiency_sum
                    .entry(format!("{:?}", sample.operation_type))
                    .or_insert((0.0, 0));
                entry.0 += sample.efficiency_percentage;
                entry.1 += 1;
            }
        }
        let bandwidth_efficiency_by_operation: HashMap<String, f64> = efficiency_sum
            .into_iter()
            .map(|(op, (sum, count))| (op, sum / count as f64))
            .collect();

        // Real, RELATIVE under-utilization: a device whose peak observed
        // bandwidth sits well below the best peak observed anywhere,
        // among devices with at least one real sample -- never compared
        // against `theoretical_bandwidth` (a documented assumption, not a
        // measurement). Zero samples is "unproven", not "underutilized".
        let underutilized_devices: Vec<i32> = if peak_bandwidth_achieved > 0.0 {
            monitors
                .iter()
                .filter(|(_, m)| !m.bandwidth_samples.is_empty())
                .filter(|(_, m)| m.peak_observed_bandwidth < peak_bandwidth_achieved * 0.5)
                .map(|(&device_id, _)| device_id)
                .collect()
        } else {
            Vec::new()
        };

        Self {
            avg_bandwidth_utilization: None,
            peak_bandwidth_achieved,
            bandwidth_efficiency_by_operation,
            underutilized_devices,
        }
    }
}

impl AllocationPatternSummary {
    fn new(allocations: &HashMap<Uuid, GpuMemoryAllocation>) -> Self {
        let total_allocations = allocations.len();
        let total_bytes: u128 = allocations.values().map(|a| a.size_bytes as u128).sum();
        let avg_allocation_size = if total_allocations > 0 {
            (total_bytes / total_allocations as u128) as usize
        } else {
            0
        };
        let largest_allocation = allocations.values().map(|a| a.size_bytes).max().unwrap_or(0);

        let mut allocation_size_distribution: HashMap<String, usize> = HashMap::new();
        for allocation in allocations.values() {
            *allocation_size_distribution
                .entry(format!("{:?}", allocation.memory_type))
                .or_insert(0) += 1;
        }

        // Heuristic, documented as such (not a certainty): an allocation
        // still outstanding after this long is flagged as a possible
        // leak. Real elapsed time from the real allocation timestamp;
        // never a fabricated count -- the old code reported `0` always,
        // even with real un-freed allocations on record.
        const LEAK_SUSPECT_THRESHOLD: Duration = Duration::from_secs(300);
        let memory_leaks_detected = allocations
            .values()
            .filter(|a| {
                !a.freed && a.timestamp.elapsed().unwrap_or_default() > LEAK_SUSPECT_THRESHOLD
            })
            .count();

        // Real hot spots: group by the most specific real context label
        // available, summing real bytes. `allocation_frequency` is a real
        // rate (count / the real observed timestamp span across ALL
        // tracked allocations), falling back to the raw real count only
        // when that span is degenerate (e.g. a single allocation).
        // `avg_allocation_lifetime` averages real `free_timestamp -
        // timestamp` deltas over allocations that HAVE been freed at that
        // location -- one still outstanding has no real lifetime yet.
        let observation_span_secs = {
            let timestamps: Vec<SystemTime> = allocations.values().map(|a| a.timestamp).collect();
            match (timestamps.iter().min(), timestamps.iter().max()) {
                (Some(&min_t), Some(&max_t)) => {
                    max_t.duration_since(min_t).unwrap_or_default().as_secs_f64()
                },
                _ => 0.0,
            }
        };

        let mut by_location: HashMap<String, (usize, u64, Duration, usize)> = HashMap::new();
        for allocation in allocations.values() {
            let location = allocation
                .allocation_context
                .tensor_name
                .clone()
                .or_else(|| allocation.allocation_context.layer_name.clone())
                .or_else(|| allocation.allocation_context.kernel_name.clone())
                .unwrap_or_else(|| {
                    format!("{:?}", allocation.allocation_context.allocation_source)
                });
            let entry = by_location.entry(location).or_insert((0, 0, Duration::ZERO, 0));
            entry.0 += 1;
            entry.1 += allocation.size_bytes as u64;
            if let Some(free_time) = allocation.free_timestamp {
                if let Ok(lifetime) = free_time.duration_since(allocation.timestamp) {
                    entry.2 += lifetime;
                    entry.3 += 1;
                }
            }
        }
        let mut allocation_hot_spots: Vec<AllocationHotSpot> = by_location
            .into_iter()
            .map(
                |(location, (count, bytes, total_lifetime, freed_count))| AllocationHotSpot {
                    location,
                    allocation_frequency: if observation_span_secs > 0.0 {
                        count as f64 / observation_span_secs
                    } else {
                        count as f64
                    },
                    total_memory_allocated: bytes as usize,
                    avg_allocation_lifetime: if freed_count > 0 {
                        total_lifetime / freed_count as u32
                    } else {
                        Duration::ZERO
                    },
                },
            )
            .collect();
        allocation_hot_spots.sort_by_key(|h| std::cmp::Reverse(h.total_memory_allocated));
        allocation_hot_spots.truncate(10);

        Self {
            total_allocations,
            avg_allocation_size,
            largest_allocation,
            allocation_size_distribution,
            memory_leaks_detected,
            allocation_hot_spots,
        }
    }
}

impl CrossDeviceTransferSummary {
    fn new(transfers: &[CrossDeviceTransfer]) -> Self {
        let total_transfers = transfers.len();
        let total_bytes_transferred: usize = transfers.iter().map(|t| t.bytes_transferred).sum();
        let avg_transfer_bandwidth = if total_transfers > 0 {
            transfers.iter().map(|t| t.bandwidth_achieved).sum::<f64>() / total_transfers as f64
        } else {
            0.0
        };

        // Real, RELATIVE bottleneck detection: a (source, target) device
        // pair whose average achieved bandwidth sits well below the
        // overall average across every pair -- a genuine comparison
        // against other real measurements, never an absolute claim this
        // crate cannot verify (e.g. "P2P not available", which needs a
        // real capability query -- see `detect_p2p_capability`).
        let mut by_pair: HashMap<(i32, i32), Vec<f64>> = HashMap::new();
        for t in transfers {
            by_pair
                .entry((t.source_device, t.target_device))
                .or_default()
                .push(t.bandwidth_achieved);
        }
        let transfer_bottlenecks: Vec<TransferBottleneck> = if avg_transfer_bandwidth > 0.0 {
            by_pair
                .into_iter()
                .filter_map(|(pair, bandwidths)| {
                    let pair_avg = bandwidths.iter().sum::<f64>() / bandwidths.len() as f64;
                    if pair_avg < avg_transfer_bandwidth * 0.5 {
                        Some(TransferBottleneck {
                            device_pair: pair,
                            bottleneck_type: TransferBottleneckType::BandwidthLimited,
                            impact_severity: (1.0 - pair_avg / avg_transfer_bandwidth)
                                .clamp(0.0, 1.0),
                        })
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        Self {
            total_transfers,
            total_bytes_transferred,
            avg_transfer_bandwidth,
            // Needs to know which transfers were really P2P, which this
            // crate cannot determine -- see `CrossDeviceTransfer::p2p_enabled`.
            p2p_efficiency: None,
            transfer_bottlenecks,
        }
    }
}

#[derive(Debug)]
struct GpuMemoryPool {
    device_id: i32,
    total_memory: usize,
    free_memory: usize,
}

/// Configuration for advanced GPU profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedGpuProfilingConfig {
    /// Enable GPU profiling
    pub enable_gpu_profiling: bool,
    /// Number of GPU devices to profile
    pub device_count: i32,
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
    /// Enable kernel profiling
    pub enable_kernel_profiling: bool,
    /// Enable bandwidth monitoring
    pub enable_bandwidth_monitoring: bool,
    /// Maximum number of allocations to track
    pub max_tracked_allocations: usize,
    /// Sampling rate for profiling (0.0 to 1.0)
    pub profiling_sampling_rate: f32,
    /// Enable fragmentation analysis
    pub enable_fragmentation_analysis: bool,
}

impl Default for AdvancedGpuProfilingConfig {
    fn default() -> Self {
        Self {
            enable_gpu_profiling: true,
            device_count: 1,
            enable_memory_profiling: true,
            enable_kernel_profiling: true,
            enable_bandwidth_monitoring: true,
            max_tracked_allocations: 10000,
            profiling_sampling_rate: 1.0,
            enable_fragmentation_analysis: true,
        }
    }
}

/// Summary report for kernel optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelOptimizationSummaryReport {
    pub total_kernels_analyzed: usize,
    pub optimization_opportunities_found: usize,
    pub high_impact_optimizations: Vec<HighImpactOptimization>,
    pub fusion_opportunities: usize,
    pub regression_alerts: usize,
    /// Composite score in `[0, 100]`, or `None` when no kernel has been
    /// analysed yet and there is therefore nothing to score.
    pub overall_optimization_score: Option<f64>,
    pub top_recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighImpactOptimization {
    pub kernel_name: String,
    pub optimization_type: String,
    pub expected_speedup: f64,
    pub implementation_difficulty: String,
    pub description: String,
}

#[cfg(test)]
#[path = "advanced_gpu_profiler_tests.rs"]
mod advanced_gpu_profiler_tests;
