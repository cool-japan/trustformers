//! CPU-side analytical kernel optimization computations.
//!
//! This module contains the *real* analysis logic that backs the analyzers in
//! the parent [`crate::kernel_optimizer`] module. Every routine here is a pure
//! CPU-side computation over kernel/launch descriptors — none of them require a
//! live CUDA device (no `cudarc`, no driver calls). The device constants used
//! are documented NVIDIA Ampere `sm_86`-class limits (e.g. RTX A4000 /
//! RTX 30-series / A10), which is the GPU class available in this environment.
//!
//! The four analytical entry points are:
//!
//! * [`analyze_launch_config`] — occupancy estimation (CUDA occupancy-calculator
//!   algorithm) plus block-size / register / shared-memory recommendations.
//! * [`analyze_memory_access`] — coalescing / warp-divergence heuristics from the
//!   measured memory- and warp-efficiency metrics.
//! * [`analyze_compute_utilization`] — roofline / arithmetic-intensity bottleneck
//!   classification (memory-bound vs compute-bound vs latency-bound).
//! * [`find_fusion_opportunities`] — kernel-sequence fusion detection using
//!   producer/consumer semantics and an Amdahl-style memory-traffic speedup model.
//! * [`establish_baseline`] / [`compare_to_baseline`] / [`detect_regression`] —
//!   performance-regression detection: a real baseline execution-time
//!   distribution estimated from measured samples, tested against later
//!   samples with a genuine Welch's t-test (unequal variances, unequal
//!   sample sizes) rather than a constant "stable" verdict.

// reason: some `AmpereDeviceLimits` fields document device constants that are not
// read by every analyzer (they exist so the model is complete and auditable).
#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use statrs::distribution::{ContinuousCDF, StudentsT};
use uuid::Uuid;

use crate::advanced_gpu_profiler::{
    ExpectedImprovement, ImplementationDifficulty, KernelOptimization, OptimizationType,
    OptimizationValue,
};

use super::{
    BaselineComparison, BaselineProfile, DataDependency, DependencyType, FusionFeasibility,
    FusionOpportunity, FusionType, KernelProfileData, PerformanceDistribution, PerformanceTrend,
    RegressionAlert, RegressionSeverity, RegressionThresholds, RegressionType,
    SynchronizationComplexity,
};

/// Documented NVIDIA Ampere `sm_86`-class device limits used for CPU-side
/// occupancy / roofline estimation.
///
/// Values follow the CUDA C Programming Guide "Compute Capability 8.6" table and
/// public RTX A4000 specifications. They are deliberately conservative defaults
/// rather than queried from a live device, so the analysis runs anywhere.
#[derive(Debug, Clone)]
pub struct AmpereDeviceLimits {
    /// Maximum threads per thread block.
    pub max_threads_per_block: u32,
    /// Threads per warp.
    pub warp_size: u32,
    /// Maximum resident warps per SM (1536 threads / 32).
    pub max_warps_per_sm: u32,
    /// Maximum resident threads per SM.
    pub max_threads_per_sm: u32,
    /// Maximum resident thread blocks per SM.
    pub max_blocks_per_sm: u32,
    /// Maximum registers per thread.
    pub max_registers_per_thread: u32,
    /// Total 32-bit registers in the SM register file.
    pub registers_per_sm: u32,
    /// Register allocation granularity (registers allocated per warp, rounded up).
    pub register_alloc_unit: u32,
    /// Default maximum shared memory per block (bytes).
    pub max_shared_memory_per_block: usize,
    /// Maximum shared memory per SM (bytes, sm_86 carve-out limit).
    pub max_shared_memory_per_sm: usize,
    /// Shared-memory allocation granularity (bytes).
    pub shared_mem_alloc_unit: usize,
    /// L2 cache line size (bytes) — a fully coalesced warp transaction.
    pub l2_cache_line_bytes: usize,
    /// Peak FP32 throughput (GFLOP/s). RTX A4000 ≈ 19.17 TFLOPS.
    pub peak_fp32_gflops: f64,
    /// Peak global-memory bandwidth (GB/s). RTX A4000 ≈ 448 GB/s.
    pub peak_memory_bandwidth_gb_s: f64,
    /// Occupancy below this fraction is treated as "needs improvement".
    pub target_occupancy: f64,
}

impl AmpereDeviceLimits {
    /// Construct the documented `sm_86` (Ampere GA10x) default limits.
    pub fn sm_86() -> Self {
        Self {
            max_threads_per_block: 1024,
            warp_size: 32,
            max_warps_per_sm: 48,
            max_threads_per_sm: 1536,
            max_blocks_per_sm: 16,
            max_registers_per_thread: 255,
            registers_per_sm: 65_536,
            register_alloc_unit: 256,
            max_shared_memory_per_block: 49_152,
            max_shared_memory_per_sm: 102_400,
            shared_mem_alloc_unit: 128,
            l2_cache_line_bytes: 128,
            peak_fp32_gflops: 19_170.0,
            peak_memory_bandwidth_gb_s: 448.0,
            target_occupancy: 0.75,
        }
    }

    /// Roofline ridge point in FLOP/byte (peak compute / peak bandwidth).
    pub fn ridge_point(&self) -> f64 {
        self.peak_fp32_gflops / self.peak_memory_bandwidth_gb_s
    }
}

impl Default for AmpereDeviceLimits {
    fn default() -> Self {
        Self::sm_86()
    }
}

/// Which hardware resource binds the achievable occupancy of a launch config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyLimiter {
    /// Limited by the per-SM register file.
    Registers,
    /// Limited by the per-SM shared-memory budget.
    SharedMemory,
    /// Limited by the per-SM resident-warp ceiling (a function of block size).
    Warps,
    /// Limited by the per-SM resident-block ceiling.
    Blocks,
    /// No single resource binds — already at the device maximum.
    None,
}

/// Result of the CUDA-style occupancy computation for one launch configuration.
#[derive(Debug, Clone, Copy)]
pub struct OccupancyEstimate {
    /// Achieved occupancy as a fraction in `[0, 1]`.
    pub occupancy: f64,
    /// Concurrently resident thread blocks per SM.
    pub active_blocks_per_sm: u32,
    /// Concurrently resident warps per SM.
    pub active_warps_per_sm: u32,
    /// Warps required per thread block.
    pub warps_per_block: u32,
    /// The binding resource for this configuration.
    pub limiter: OccupancyLimiter,
}

/// Sensitivity factor mapping occupancy gains to performance gains. Occupancy
/// improvements have diminishing returns (latency hiding saturates), so a gain
/// in occupancy is discounted when translated to an expected speedup.
const OCCUPANCY_PERF_SENSITIVITY: f64 = 0.5;

/// Memory efficiency below this is treated as "poorly coalesced".
const COALESCING_GOOD_THRESHOLD: f64 = 0.6;
/// Memory efficiency below this (but above [`COALESCING_GOOD_THRESHOLD`]) is
/// treated as "partially coalesced — vectorization may help".
const COALESCING_EXCELLENT_THRESHOLD: f64 = 0.85;
/// Warp execution efficiency below this indicates harmful branch divergence.
const WARP_EFFICIENCY_THRESHOLD: f64 = 0.7;
/// Arithmetic intensity (FLOP/byte) below this is treated as memory-bound.
const MEMORY_BOUND_INTENSITY: f64 = 10.0;

/// Representative size (bytes) of an intermediate activation tensor exchanged
/// between two fused kernels: 1M FP32 elements. Used to estimate the global
/// memory traffic eliminated by fusion.
const NOMINAL_TENSOR_BYTES: usize = 4 * 1024 * 1024;

// ── shared helpers ───────────────────────────────────────────────────────────

#[inline]
fn round_up_u32(value: u32, multiple: u32) -> u32 {
    if multiple == 0 {
        value
    } else {
        value.div_ceil(multiple) * multiple
    }
}

#[inline]
fn round_up_usize(value: usize, multiple: usize) -> usize {
    if multiple == 0 {
        value
    } else {
        value.div_ceil(multiple) * multiple
    }
}

/// Total threads in a block from its `(x, y, z)` dimensions.
#[inline]
fn block_threads(block: (u32, u32, u32)) -> u32 {
    block.0.saturating_mul(block.1).saturating_mul(block.2)
}

/// Build an [`ExpectedImprovement`] from a performance gain and a memory-usage
/// reduction (both as percentages). Energy efficiency is modeled as a fraction
/// of the performance gain (fewer cycles ≈ less energy).
fn improvement(performance_gain: f64, memory_reduction: f64) -> ExpectedImprovement {
    let perf = performance_gain.clamp(0.0, 95.0);
    ExpectedImprovement {
        performance_gain_percentage: perf,
        memory_usage_reduction_percentage: memory_reduction.clamp(0.0, 95.0),
        energy_efficiency_improvement: (perf * 0.4).clamp(0.0, 60.0),
        scalability_improvement: (perf * 0.3).clamp(0.0, 60.0),
    }
}

#[allow(clippy::too_many_arguments)]
fn make_opt(
    optimization_type: OptimizationType,
    current_value: OptimizationValue,
    suggested_value: OptimizationValue,
    expected_improvement: ExpectedImprovement,
    confidence: f64,
    explanation: String,
    implementation_difficulty: ImplementationDifficulty,
) -> KernelOptimization {
    KernelOptimization {
        optimization_type,
        current_value,
        suggested_value,
        expected_improvement,
        confidence: confidence.clamp(0.0, 1.0),
        explanation,
        implementation_difficulty,
    }
}

/// Translate an occupancy delta into an expected performance gain percentage.
fn occupancy_perf_gain(current: f64, target: f64) -> f64 {
    if target <= current {
        return 0.0;
    }
    let denom = current.max(0.05);
    ((target - current) / denom) * 100.0 * OCCUPANCY_PERF_SENSITIVITY
}

// ── occupancy model ──────────────────────────────────────────────────────────

/// Estimate achievable occupancy for a launch configuration using the standard
/// CUDA occupancy-calculator algorithm (the same one `cudaOccupancyMaxActive
/// BlocksPerMultiprocessor` implements), driven entirely on the CPU.
pub fn estimate_occupancy(
    threads_per_block: u32,
    registers_per_thread: u32,
    shared_memory_per_block: usize,
    limits: &AmpereDeviceLimits,
) -> OccupancyEstimate {
    let warps_per_block =
        round_up_u32(threads_per_block.max(1), limits.warp_size) / limits.warp_size;

    // An over-sized block (more warps than fit on an SM, or more threads than the
    // device allows) cannot be launched at all.
    if warps_per_block == 0
        || threads_per_block > limits.max_threads_per_block
        || warps_per_block > limits.max_warps_per_sm
    {
        return OccupancyEstimate {
            occupancy: 0.0,
            active_blocks_per_sm: 0,
            active_warps_per_sm: 0,
            warps_per_block,
            limiter: OccupancyLimiter::Warps,
        };
    }

    // Block-count ceiling.
    let blocks_by_blocks = limits.max_blocks_per_sm;

    // Resident-warp ceiling.
    let blocks_by_warps = limits.max_warps_per_sm / warps_per_block;

    // Register-file ceiling. Registers are allocated per warp, rounded up to the
    // allocation granularity.
    let blocks_by_registers = if registers_per_thread == 0 {
        limits.max_blocks_per_sm
    } else {
        let regs_per_warp = round_up_u32(
            registers_per_thread.saturating_mul(limits.warp_size),
            limits.register_alloc_unit,
        );
        // `warps_per_block` is guaranteed non-zero here (handled by the early
        // return above); a zero register allocation falls back to no constraint.
        limits
            .registers_per_sm
            .checked_div(regs_per_warp)
            .map(|warps_by_regs| warps_by_regs / warps_per_block)
            .unwrap_or(limits.max_blocks_per_sm)
    };

    // Shared-memory ceiling.
    let blocks_by_shared = if shared_memory_per_block == 0 {
        limits.max_blocks_per_sm
    } else {
        let smem = round_up_usize(shared_memory_per_block, limits.shared_mem_alloc_unit);
        limits
            .max_shared_memory_per_sm
            .checked_div(smem)
            .map(|blocks| blocks as u32)
            .unwrap_or(limits.max_blocks_per_sm)
    };

    // The binding constraint is the smallest ceiling.
    let active_blocks = blocks_by_blocks
        .min(blocks_by_warps)
        .min(blocks_by_registers)
        .min(blocks_by_shared);

    let limiter = if active_blocks == 0 {
        // Some resource forbids even one resident block.
        if blocks_by_registers == 0 {
            OccupancyLimiter::Registers
        } else if blocks_by_shared == 0 {
            OccupancyLimiter::SharedMemory
        } else {
            OccupancyLimiter::Warps
        }
    } else if active_blocks == blocks_by_registers && registers_per_thread > 0 {
        OccupancyLimiter::Registers
    } else if active_blocks == blocks_by_shared && shared_memory_per_block > 0 {
        OccupancyLimiter::SharedMemory
    } else if active_blocks == blocks_by_warps && blocks_by_warps < blocks_by_blocks {
        OccupancyLimiter::Warps
    } else if active_blocks == blocks_by_blocks {
        OccupancyLimiter::Blocks
    } else {
        OccupancyLimiter::None
    };

    let active_warps = active_blocks.saturating_mul(warps_per_block);
    let occupancy = (f64::from(active_warps) / f64::from(limits.max_warps_per_sm)).clamp(0.0, 1.0);

    OccupancyEstimate {
        occupancy,
        active_blocks_per_sm: active_blocks,
        active_warps_per_sm: active_warps,
        warps_per_block,
        limiter,
    }
}

/// Search the legal block-size space (multiples of the warp size) for the
/// configuration that maximizes occupancy, holding per-thread registers and
/// per-block shared memory fixed. Returns `(threads_per_block, occupancy)`.
fn best_block_size(profile: &KernelProfileData, limits: &AmpereDeviceLimits) -> Option<(u32, f64)> {
    let mut best: Option<(u32, f64)> = None;
    let mut threads = limits.warp_size;
    while threads <= limits.max_threads_per_block {
        let est = estimate_occupancy(
            threads,
            profile.registers_per_thread,
            profile.shared_memory_bytes,
            limits,
        );
        match best {
            Some((_, best_occ)) if est.occupancy <= best_occ => {},
            _ => best = Some((threads, est.occupancy)),
        }
        threads += limits.warp_size;
    }
    best
}

/// Largest per-thread register count `<= current` that still reaches the target
/// occupancy (holding block size and shared memory fixed). `None` if even the
/// 8-register floor cannot reach the target.
fn registers_for_target(profile: &KernelProfileData, limits: &AmpereDeviceLimits) -> Option<u32> {
    let current = profile.registers_per_thread;
    if current <= 8 {
        return None;
    }
    let mut regs = current.saturating_sub(1);
    while regs >= 8 {
        let est = estimate_occupancy(
            block_threads(profile.block_size),
            regs,
            profile.shared_memory_bytes,
            limits,
        );
        if est.occupancy >= limits.target_occupancy {
            return Some(regs);
        }
        regs -= 1;
    }
    None
}

/// Largest per-block shared-memory budget (bytes) `<= current` that still reaches
/// the target occupancy. `None` if zero shared memory still cannot reach it.
fn shared_mem_for_target(
    profile: &KernelProfileData,
    limits: &AmpereDeviceLimits,
) -> Option<usize> {
    let current = profile.shared_memory_bytes;
    if current == 0 {
        return None;
    }
    let step = limits.shared_mem_alloc_unit.max(1);
    let mut smem = current.saturating_sub(step);
    loop {
        let est = estimate_occupancy(
            block_threads(profile.block_size),
            profile.registers_per_thread,
            smem,
            limits,
        );
        if est.occupancy >= limits.target_occupancy {
            return Some(smem);
        }
        if smem < step {
            return None;
        }
        smem -= step;
    }
}

// ── launch-config analyzer ─────────────────────────────────────────────────────

/// Analyze a launch configuration and produce concrete, ranked optimization
/// recommendations. Driven by the occupancy model above.
pub fn analyze_launch_config(profile: &KernelProfileData) -> Vec<KernelOptimization> {
    let limits = AmpereDeviceLimits::sm_86();
    let mut out = Vec::new();
    let threads = block_threads(profile.block_size);
    if threads == 0 {
        return out;
    }

    let current = estimate_occupancy(
        threads,
        profile.registers_per_thread,
        profile.shared_memory_bytes,
        &limits,
    );

    // (1) Partial-warp waste: a block size that is not a multiple of the warp
    // size leaves lanes in the final warp permanently idle.
    if !threads.is_multiple_of(limits.warp_size) {
        let rounded = round_up_u32(threads, limits.warp_size);
        let waste = f64::from(rounded - threads) / f64::from(rounded);
        out.push(make_opt(
            OptimizationType::BlockSize,
            OptimizationValue::IntegerValue(threads),
            OptimizationValue::IntegerValue(rounded),
            improvement(waste * 100.0, 0.0),
            0.9,
            format!(
                "Block size {threads} is not a multiple of the warp size ({}); {:.1}% of \
                 lanes in the trailing warp are idle. Round up to {rounded} threads/block.",
                limits.warp_size,
                waste * 100.0
            ),
            ImplementationDifficulty::Trivial,
        ));
    }

    // (2) Too few warps per block to hide memory/instruction latency.
    if threads < 2 * limits.warp_size {
        let suggested = 256u32.min(limits.max_threads_per_block);
        let suggested_occ = estimate_occupancy(
            suggested,
            profile.registers_per_thread,
            profile.shared_memory_bytes,
            &limits,
        )
        .occupancy;
        out.push(make_opt(
            OptimizationType::BlockSize,
            OptimizationValue::IntegerValue(threads),
            OptimizationValue::IntegerValue(suggested),
            improvement(
                occupancy_perf_gain(current.occupancy, suggested_occ.max(0.5)),
                0.0,
            ),
            0.8,
            format!(
                "Only {} warp(s) per block: too few to hide latency. Increase to {suggested} \
                 threads/block to expose more instruction-level and warp-level parallelism.",
                current.warps_per_block
            ),
            ImplementationDifficulty::Easy,
        ));
    }

    // (3) Occupancy below target — address the binding resource directly.
    if current.occupancy < limits.target_occupancy {
        match current.limiter {
            OccupancyLimiter::Registers => {
                if let Some(target_regs) = registers_for_target(profile, &limits) {
                    let new_occ = estimate_occupancy(
                        threads,
                        target_regs,
                        profile.shared_memory_bytes,
                        &limits,
                    )
                    .occupancy;
                    out.push(make_opt(
                        OptimizationType::RegisterOptimization,
                        OptimizationValue::IntegerValue(profile.registers_per_thread),
                        OptimizationValue::IntegerValue(target_regs),
                        improvement(occupancy_perf_gain(current.occupancy, new_occ), 0.0),
                        0.7,
                        format!(
                            "Occupancy {:.0}% is register-limited ({} regs/thread). Cap registers \
                             at {target_regs} (e.g. via -maxrregcount / __launch_bounds__) to raise \
                             occupancy to {:.0}%.",
                            current.occupancy * 100.0,
                            profile.registers_per_thread,
                            new_occ * 100.0
                        ),
                        ImplementationDifficulty::Moderate,
                    ));
                }
            },
            OccupancyLimiter::SharedMemory => {
                if let Some(target_smem) = shared_mem_for_target(profile, &limits) {
                    let new_occ = estimate_occupancy(
                        threads,
                        profile.registers_per_thread,
                        target_smem,
                        &limits,
                    )
                    .occupancy;
                    let mem_reduction = if profile.shared_memory_bytes > 0 {
                        (1.0 - target_smem as f64 / profile.shared_memory_bytes as f64) * 100.0
                    } else {
                        0.0
                    };
                    out.push(make_opt(
                        OptimizationType::SharedMemory,
                        OptimizationValue::IntegerValue(profile.shared_memory_bytes as u32),
                        OptimizationValue::IntegerValue(target_smem as u32),
                        improvement(
                            occupancy_perf_gain(current.occupancy, new_occ),
                            mem_reduction,
                        ),
                        0.7,
                        format!(
                            "Occupancy {:.0}% is shared-memory-limited ({} B/block). Reduce the \
                             per-block shared-memory footprint to {target_smem} B (e.g. smaller \
                             tiles / reuse) to raise occupancy to {:.0}%.",
                            current.occupancy * 100.0,
                            profile.shared_memory_bytes,
                            new_occ * 100.0
                        ),
                        ImplementationDifficulty::Difficult,
                    ));
                }
            },
            _ => {},
        }

        // A different block size may relieve a warp/block-count limit (and is
        // worth surfacing even alongside a register/shared-memory fix).
        if let Some((best_threads, best_occ)) = best_block_size(profile, &limits) {
            if best_threads != threads && best_occ > current.occupancy + 0.05 {
                out.push(make_opt(
                    OptimizationType::BlockSize,
                    OptimizationValue::IntegerValue(threads),
                    OptimizationValue::IntegerValue(best_threads),
                    improvement(occupancy_perf_gain(current.occupancy, best_occ), 0.0),
                    0.75,
                    format!(
                        "A block size of {best_threads} threads maximizes theoretical occupancy \
                         ({:.0}%) versus {:.0}% at the current {threads} threads/block.",
                        best_occ * 100.0,
                        current.occupancy * 100.0
                    ),
                    ImplementationDifficulty::Easy,
                ));
            }
        }
    }

    out
}

// ── memory-access analyzer ─────────────────────────────────────────────────────

/// Analyze the measured memory- and warp-efficiency metrics and recommend
/// coalescing / layout / divergence fixes.
pub fn analyze_memory_access(profile: &KernelProfileData) -> Vec<KernelOptimization> {
    let limits = AmpereDeviceLimits::sm_86();
    let mut out = Vec::new();
    let mem_eff = profile.memory_efficiency.clamp(0.0, 1.0);
    let warp_eff = profile.warp_efficiency.clamp(0.0, 1.0);
    let bw_util = profile.memory_bandwidth_utilization.clamp(0.0, 1.0);

    if mem_eff < COALESCING_GOOD_THRESHOLD {
        // Most of the bandwidth wasted by uncoalesced access can be recovered.
        let recovered = (1.0 - mem_eff) * 100.0 * 0.6;
        out.push(make_opt(
            OptimizationType::MemoryCoalescing,
            OptimizationValue::FloatValue(mem_eff),
            OptimizationValue::FloatValue(0.9),
            improvement(recovered, (1.0 - mem_eff) * 40.0),
            0.75,
            format!(
                "Memory efficiency is only {:.0}%: consecutive threads are not reading \
                 consecutive addresses, so each {}-byte L2 cache line carries useful data for \
                 few lanes. Make the innermost (thread) index map to the contiguous memory \
                 dimension so a warp fills whole cache lines.",
                mem_eff * 100.0,
                limits.l2_cache_line_bytes
            ),
            ImplementationDifficulty::Moderate,
        ));
        out.push(make_opt(
            OptimizationType::MemoryLayoutOptimization,
            OptimizationValue::LayoutPattern("array-of-structs".to_string()),
            OptimizationValue::LayoutPattern("struct-of-arrays, 128B-aligned".to_string()),
            improvement(recovered * 0.5, (1.0 - mem_eff) * 25.0),
            0.6,
            "Reorganize the data layout to struct-of-arrays and pad rows to 128-byte \
             boundaries so warp accesses align to cache-line transactions."
                .to_string(),
            ImplementationDifficulty::Difficult,
        ));
    } else if mem_eff < COALESCING_EXCELLENT_THRESHOLD {
        let recovered = (1.0 - mem_eff) * 100.0 * 0.4;
        out.push(make_opt(
            OptimizationType::MemoryCoalescing,
            OptimizationValue::LayoutPattern("scalar loads".to_string()),
            OptimizationValue::LayoutPattern("vectorized float4 loads".to_string()),
            improvement(recovered, 0.0),
            0.65,
            format!(
                "Memory efficiency {:.0}% is good but not optimal; widen loads/stores to \
                 128-bit (float4) so each thread issues fewer, larger transactions.",
                mem_eff * 100.0
            ),
            ImplementationDifficulty::Easy,
        ));
    }

    if warp_eff < WARP_EFFICIENCY_THRESHOLD {
        let gain = (1.0 - warp_eff) * 100.0 * 0.5;
        out.push(make_opt(
            OptimizationType::WarpDivergence,
            OptimizationValue::FloatValue(warp_eff),
            OptimizationValue::FloatValue(0.9),
            improvement(gain, 0.0),
            0.7,
            format!(
                "Warp execution efficiency {:.0}% indicates branch divergence: lanes within a \
                 warp follow different control-flow paths and serialize. Restructure branches so \
                 a warp takes a uniform path, or sort/partition work by branch outcome.",
                warp_eff * 100.0
            ),
            ImplementationDifficulty::Moderate,
        ));
    }

    if bw_util > 0.7 && mem_eff < 0.9 {
        out.push(make_opt(
            OptimizationType::MemoryCoalescing,
            OptimizationValue::BooleanValue(false),
            OptimizationValue::BooleanValue(true),
            improvement((0.9 - mem_eff) * 100.0 * 0.4, 0.0),
            0.6,
            format!(
                "Bandwidth utilization is high ({:.0}%) while memory efficiency is {:.0}%: the \
                 kernel is bandwidth-bound and wasting transactions. Coalesce accesses and use \
                 wide vector loads to move the same data in fewer transactions.",
                bw_util * 100.0,
                mem_eff * 100.0
            ),
            ImplementationDifficulty::Moderate,
        ));
    }

    out
}

// ── compute-utilization analyzer ────────────────────────────────────────────────

/// Classify the kernel on the roofline model and recommend bottleneck-specific
/// optimizations (memory-bound vs compute-bound vs latency/occupancy-bound).
pub fn analyze_compute_utilization(profile: &KernelProfileData) -> Vec<KernelOptimization> {
    let limits = AmpereDeviceLimits::sm_86();
    let mut out = Vec::new();
    let compute_util = profile.compute_utilization.clamp(0.0, 1.0);
    let bw_util = profile.memory_bandwidth_utilization.clamp(0.0, 1.0);
    let occupancy = profile.occupancy.clamp(0.0, 1.0);

    let achieved_compute = compute_util * limits.peak_fp32_gflops;
    let achieved_bw = bw_util * limits.peak_memory_bandwidth_gb_s;
    let intensity = if achieved_bw > f64::EPSILON {
        achieved_compute / achieved_bw
    } else {
        f64::INFINITY
    };
    let ridge = limits.ridge_point();

    let memory_bound = (intensity < MEMORY_BOUND_INTENSITY && bw_util >= compute_util)
        || (bw_util > 0.8 && compute_util < 0.5);
    let compute_bound = compute_util > 0.8 && intensity >= ridge;

    if memory_bound {
        let gain = (1.0 - compute_util).clamp(0.0, 1.0) * 100.0 * 0.4 + 10.0;
        out.push(make_opt(
            OptimizationType::ComputeIntensityBalance,
            OptimizationValue::FloatValue(intensity.min(1e6)),
            OptimizationValue::FloatValue((intensity * 2.0).min(ridge)),
            improvement(gain, 0.0),
            0.7,
            format!(
                "Memory-bound: arithmetic intensity ≈ {:.1} FLOP/byte is far below the {:.0} \
                 FLOP/byte roofline ridge (compute {:.0}% vs bandwidth {:.0}%). Raise reuse with \
                 register/shared-memory tiling and fuse adjacent kernels so each loaded operand \
                 does more work before being evicted.",
                intensity,
                ridge,
                compute_util * 100.0,
                bw_util * 100.0
            ),
            ImplementationDifficulty::Difficult,
        ));
        out.push(make_opt(
            OptimizationType::MemoryCoalescing,
            OptimizationValue::FloatValue(bw_util),
            OptimizationValue::FloatValue(1.0),
            improvement(
                (1.0 - profile.memory_efficiency.clamp(0.0, 1.0)) * 100.0 * 0.5,
                0.0,
            ),
            0.65,
            "While memory-bound, ensure every byte fetched is used: coalesce global accesses \
             and prefer 128-bit vector loads to approach peak effective bandwidth."
                .to_string(),
            ImplementationDifficulty::Moderate,
        ));
    }

    if compute_bound {
        let gain = (1.0 - compute_util) * 100.0 * 0.8 + 5.0;
        out.push(make_opt(
            OptimizationType::ComputeIntensityBalance,
            OptimizationValue::FloatValue(compute_util),
            OptimizationValue::FloatValue(0.95),
            improvement(gain, 0.0),
            0.5,
            format!(
                "Compute-bound: compute units are {:.0}% utilized at intensity {:.1} FLOP/byte \
                 (≥ ridge {:.0}). Reduce the instruction count (strength-reduction, fast-math \
                 intrinsics) or move dense matmul work onto tensor cores / mixed precision.",
                compute_util * 100.0,
                intensity,
                ridge
            ),
            ImplementationDifficulty::Difficult,
        ));
    }

    if occupancy < 0.5 && compute_util < 0.6 && bw_util < 0.6 {
        let gain = occupancy_perf_gain(occupancy, limits.target_occupancy);
        out.push(make_opt(
            OptimizationType::GridSize,
            OptimizationValue::FloatValue(occupancy),
            OptimizationValue::FloatValue(limits.target_occupancy),
            improvement(gain, 0.0),
            0.6,
            format!(
                "Latency-bound: occupancy {:.0}% with low compute ({:.0}%) and bandwidth ({:.0}%) \
                 utilization — the SMs are starved. Launch more blocks / larger grids and expose \
                 more independent work per thread (instruction-level parallelism) to hide latency.",
                occupancy * 100.0,
                compute_util * 100.0,
                bw_util * 100.0
            ),
            ImplementationDifficulty::Moderate,
        ));
    }

    out
}

// ── kernel-fusion analyzer ─────────────────────────────────────────────────────

/// Coarse semantic category of a kernel inferred from its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KernelCategory {
    /// Dense matrix-multiply / convolution style (compute-bound producer).
    Matmul,
    /// Element-wise op (activation, bias, scale, residual, cast …).
    Elementwise,
    /// Reduction / normalization (softmax, sum, layernorm …).
    Reduction,
    /// Unrecognized.
    Other,
}

fn classify_kernel(name: &str) -> KernelCategory {
    let n = name.to_ascii_lowercase();
    const MATMUL: &[&str] = &[
        "matmul",
        "gemm",
        "linear",
        "conv",
        "dense",
        "bmm",
        "qk",
        "attention_scores",
    ];
    const REDUCTION: &[&str] = &[
        "softmax",
        "layernorm",
        "rmsnorm",
        "norm",
        "reduce",
        "sum",
        "mean",
        "argmax",
        "logsumexp",
    ];
    const ELEMENTWISE: &[&str] = &[
        "relu",
        "gelu",
        "silu",
        "swish",
        "sigmoid",
        "tanh",
        "bias",
        "scale",
        "dropout",
        "activation",
        "residual",
        "cast",
        "copy",
        "mask",
        "elementwise",
        "add",
        "mul",
        "sub",
    ];
    if MATMUL.iter().any(|k| n.contains(k)) {
        KernelCategory::Matmul
    } else if REDUCTION.iter().any(|k| n.contains(k)) {
        KernelCategory::Reduction
    } else if ELEMENTWISE.iter().any(|k| n.contains(k)) {
        KernelCategory::Elementwise
    } else {
        KernelCategory::Other
    }
}

/// Per-pair fusion plan: (type, memory-bound fraction `f`, traffic reduction
/// ratio `r`, sync complexity, confidence, difficulty).
struct FusionPlan {
    fusion_type: FusionType,
    mem_bound_fraction: f64,
    traffic_ratio: f64,
    sync: SynchronizationComplexity,
    confidence: f64,
    difficulty: ImplementationDifficulty,
    access_pattern: &'static str,
}

/// Decide whether two adjacent kernels (producer `a`, consumer `b`) can be fused
/// and, if so, with which strategy. Returns `None` when fusion is not profitable
/// or feasible (e.g. matmul→matmul, or unrecognized kernels).
fn fusion_plan(a: KernelCategory, b: KernelCategory) -> Option<FusionPlan> {
    use ImplementationDifficulty::{Difficult, Easy, Moderate};
    use KernelCategory::{Elementwise, Matmul, Reduction};
    use SynchronizationComplexity::{Minimal, Moderate as SyncModerate, None as SyncNone};

    let plan = match (a, b) {
        (Elementwise, Elementwise) => FusionPlan {
            fusion_type: FusionType::ElementwiseFusion,
            mem_bound_fraction: 0.95,
            traffic_ratio: 0.5,
            sync: SyncNone,
            confidence: 0.9,
            difficulty: Easy,
            access_pattern: "sequential",
        },
        (Matmul, Elementwise) => FusionPlan {
            fusion_type: FusionType::ProducerConsumerFusion,
            mem_bound_fraction: 0.3,
            traffic_ratio: 0.4,
            sync: Minimal,
            confidence: 0.8,
            difficulty: Moderate,
            access_pattern: "tiled producer-consumer (epilogue)",
        },
        (Matmul, Reduction) => FusionPlan {
            fusion_type: FusionType::AttentionFusion,
            mem_bound_fraction: 0.45,
            traffic_ratio: 0.45,
            sync: SyncModerate,
            confidence: 0.7,
            difficulty: Difficult,
            access_pattern: "score-then-reduce (attention)",
        },
        (Reduction, Elementwise) => FusionPlan {
            fusion_type: FusionType::ReductionFusion,
            mem_bound_fraction: 0.85,
            traffic_ratio: 0.5,
            sync: Minimal,
            confidence: 0.75,
            difficulty: Moderate,
            access_pattern: "reduce-then-scale",
        },
        (Elementwise, Reduction) => FusionPlan {
            fusion_type: FusionType::ReductionFusion,
            mem_bound_fraction: 0.8,
            traffic_ratio: 0.5,
            sync: SyncModerate,
            confidence: 0.7,
            difficulty: Moderate,
            access_pattern: "map-then-reduce",
        },
        (Reduction, Reduction) => FusionPlan {
            fusion_type: FusionType::ReductionFusion,
            mem_bound_fraction: 0.8,
            traffic_ratio: 0.6,
            sync: SyncModerate,
            confidence: 0.65,
            difficulty: Difficult,
            access_pattern: "fused multi-reduction",
        },
        _ => return None,
    };
    Some(plan)
}

/// Amdahl-style speedup: only the memory-bound fraction of the runtime shrinks,
/// and it shrinks by the traffic-reduction ratio achieved by eliminating the
/// intermediate global-memory round trip.
fn fusion_speedup(plan: &FusionPlan) -> f64 {
    let f = plan.mem_bound_fraction.clamp(0.0, 1.0);
    let r = plan.traffic_ratio.clamp(0.0, 1.0);
    1.0 / ((1.0 - f) + f * r)
}

/// Detect kernel-fusion opportunities across an ordered kernel sequence by
/// examining each adjacent producer→consumer pair.
pub fn find_fusion_opportunities(kernel_sequence: &[String]) -> Vec<FusionOpportunity> {
    let mut out = Vec::new();
    for pair in kernel_sequence.windows(2) {
        let (producer, consumer) = (&pair[0], &pair[1]);
        let cat_a = classify_kernel(producer);
        let cat_b = classify_kernel(consumer);
        let Some(plan) = fusion_plan(cat_a, cat_b) else {
            continue;
        };

        let speedup = fusion_speedup(&plan);
        // Eliminating the intermediate avoids one global write and one global read.
        let memory_savings = NOMINAL_TENSOR_BYTES.saturating_mul(2);

        let dependency = DataDependency {
            source_kernel: producer.clone(),
            target_kernel: consumer.clone(),
            dependency_type: DependencyType::ReadAfterWrite,
            data_size: NOMINAL_TENSOR_BYTES,
            access_pattern: plan.access_pattern.to_string(),
        };

        // Feasibility: element-wise fusion needs no extra shared memory and
        // little register pressure; producer→reduction fusions may stress shared
        // memory and require synchronization.
        let needs_shared = matches!(
            plan.fusion_type,
            FusionType::AttentionFusion | FusionType::ReductionFusion
        );
        let feasibility = FusionFeasibility {
            resource_constraints_satisfied: true,
            register_usage_feasible: true,
            shared_memory_feasible: !needs_shared
                || NOMINAL_TENSOR_BYTES
                    <= AmpereDeviceLimits::sm_86().max_shared_memory_per_block * 64,
            synchronization_complexity: plan.sync,
            fusion_confidence: plan.confidence,
        };

        out.push(FusionOpportunity {
            opportunity_id: Uuid::new_v4(),
            kernel_group: vec![producer.clone(), consumer.clone()],
            fusion_type: plan.fusion_type,
            data_dependencies: vec![dependency],
            expected_speedup: speedup,
            memory_savings,
            implementation_complexity: plan.difficulty,
            fusion_feasibility: feasibility,
        });
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────
// Performance regression detection
// ─────────────────────────────────────────────────────────────────────────
//
// Real statistics backing `PerformanceRegressionDetector`: establish a
// baseline execution-time distribution from a kernel's early measurements,
// then test later measurements against it with a genuine Welch's t-test
// (unequal variances, unequal sample sizes) -- the same technique already
// used by `crate::differential_debugging::welch_t_test`. No constant ever
// stands in for a value that was not actually computed from real samples;
// every "not enough data yet" case returns `None` rather than a fabricated
// "stable" / "0.95 significant" verdict.

/// Minimum number of execution-time samples required before a baseline
/// performance distribution can be established for a kernel. Below this,
/// [`establish_baseline`] must not be called -- there is nothing yet to
/// compare against.
pub const MIN_BASELINE_SAMPLES: usize = 10;

/// Minimum number of *recent* (post-baseline, within the configured
/// detection window) samples required before a comparison against the
/// established baseline is attempted.
pub const MIN_COMPARISON_SAMPLES: usize = 5;

/// Convert a [`SystemTime`] to nanoseconds since the Unix epoch, saturating
/// to 0 rather than panicking if the clock ever reads before the epoch.
pub fn system_time_to_ns(t: SystemTime) -> u64 {
    t.duration_since(SystemTime::UNIX_EPOCH).unwrap_or(Duration::ZERO).as_nanos() as u64
}

/// Build a [`BaselineProfile`] from a kernel's first `samples_secs.len()`
/// real execution-time measurements (in seconds). Percentiles use the
/// nearest-rank method; the outlier threshold is the conventional
/// mean + 3*std_dev; the confidence interval on the mean uses the exact
/// Student's t critical value for `n - 1` degrees of freedom (not a fixed
/// z=1.96, which is only exact as n -> infinity).
///
/// Panics if `samples_secs` is empty -- callers must gate on
/// `samples_secs.len() >= MIN_BASELINE_SAMPLES` first (see
/// `PerformanceRegressionDetector::check_regression`).
pub fn establish_baseline(kernel_name: &str, samples_secs: &[f64]) -> BaselineProfile {
    let n = samples_secs.len();
    assert!(n > 0, "establish_baseline requires at least one sample");

    let mean = samples_secs.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        samples_secs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    let mut sorted = samples_secs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let percentile_secs = |p: f64| -> f64 {
        let rank = (p / 100.0 * (n - 1) as f64).round() as usize;
        sorted[rank.min(n - 1)]
    };
    let percentiles: HashMap<u8, f64> = [50u8, 90, 95, 99]
        .iter()
        .map(|&p| (p, percentile_secs(p as f64).max(0.0)))
        .collect();

    let outlier_threshold_secs = (mean + 3.0 * std_dev).max(0.0);

    let confidence_interval = if n > 1 && std_dev > 0.0 {
        StudentsT::new(0.0, 1.0, n as f64 - 1.0)
            .ok()
            .map(|t_dist| {
                let margin = t_dist.inverse_cdf(0.975) * std_dev / (n as f64).sqrt();
                (
                    Duration::from_secs_f64((mean - margin).max(0.0)),
                    Duration::from_secs_f64((mean + margin).max(0.0)),
                )
            })
            .unwrap_or((
                Duration::from_secs_f64(mean.max(0.0)),
                Duration::from_secs_f64(mean.max(0.0)),
            ))
    } else {
        (
            Duration::from_secs_f64(mean.max(0.0)),
            Duration::from_secs_f64(mean.max(0.0)),
        )
    };

    BaselineProfile {
        kernel_name: kernel_name.to_string(),
        baseline_performance: Duration::from_secs_f64(mean.max(0.0)),
        performance_distribution: PerformanceDistribution {
            mean_secs: mean.max(0.0),
            std_dev_secs: std_dev.max(0.0),
            percentiles,
            outlier_threshold_secs,
            sample_count: n,
        },
        established_date: SystemTime::now(),
        confidence_interval,
    }
}

/// Real outcome of comparing a recent execution-time sample against an
/// established baseline via Welch's t-test.
#[derive(Debug, Clone, Copy)]
pub struct BaselineTestOutcome {
    /// `(recent_mean - baseline_mean) / baseline_mean`, as a fraction
    /// (`0.10` == 10% slower, `-0.10` == 10% faster).
    pub relative_change: f64,
    /// 95% confidence interval on `relative_change`, same units.
    pub relative_change_ci: (f64, f64),
    pub t_statistic: f64,
    /// Two-tailed p-value from the Welch-Satterthwaite Student's
    /// t-distribution.
    pub p_value: f64,
    pub degrees_of_freedom: f64,
    /// `recent_variance / baseline_variance`. `>> 1.0` means the recent
    /// window is meaningfully MORE erratic than the baseline
    /// (independent of whether the mean moved) -- see
    /// [`detect_regression`]'s one-sided `is_volatile` gate, which only
    /// fires on this side: a ratio `<< 1.0` means the kernel got MORE
    /// consistent, which is not volatility.
    pub variance_ratio: f64,
}

/// Real Welch's t-test (unequal variances, unequal sample sizes) between a
/// kernel's baseline execution-time distribution (summarised by its mean,
/// std-dev and sample count) and a window of recent raw measurements (both
/// in seconds).
///
/// Returns `None` -- never a fabricated result -- when there are fewer
/// than 2 baseline or recent samples, when either variance is not
/// finite, or when the resulting degrees of freedom or `StudentsT`
/// parameters are not finite/positive (all `statrs` invariants). Callers
/// must treat `None` as "no comparison possible right now", never as
/// "stable".
///
/// When *both* the baseline and the recent window are exactly constant
/// (zero variance on both sides -- common with synthetic/mocked timings,
/// or a kernel with genuinely deterministic execution time), the
/// classical Welch standard error is zero and the textbook t-statistic
/// is 0/0. That is not "no data": it is the most repeatable measurement
/// possible on both sides, so the well-defined limiting outcome is
/// reported directly rather than discarded as `None`. Two equal
/// constants are a real, maximally-confident "no difference"
/// (`p_value = 1.0`); two different constants are a real,
/// maximally-confident difference (`t -> +/-infinity`, `p_value = 0.0`,
/// the limit as variance -> 0 of the ordinary formula below).
pub fn compare_to_baseline(
    baseline_mean_secs: f64,
    baseline_std_secs: f64,
    baseline_n: usize,
    recent_samples_secs: &[f64],
) -> Option<BaselineTestOutcome> {
    let n_recent = recent_samples_secs.len();
    if baseline_n < 2 || n_recent < 2 {
        return None;
    }

    let recent_mean = recent_samples_secs.iter().sum::<f64>() / n_recent as f64;
    let recent_var = recent_samples_secs.iter().map(|v| (v - recent_mean).powi(2)).sum::<f64>()
        / (n_recent as f64 - 1.0);
    let baseline_var = baseline_std_secs * baseline_std_secs;

    if !baseline_var.is_finite() || !recent_var.is_finite() {
        return None;
    }

    if baseline_var <= 0.0 && recent_var <= 0.0 {
        let relative_change = if baseline_mean_secs != 0.0 {
            (recent_mean - baseline_mean_secs) / baseline_mean_secs
        } else {
            0.0
        };
        let degrees_of_freedom = (baseline_n + n_recent) as f64 - 2.0;
        let identical = (recent_mean - baseline_mean_secs).abs()
            <= f64::EPSILON * baseline_mean_secs.abs().max(recent_mean.abs()).max(1.0);
        return Some(if identical {
            BaselineTestOutcome {
                relative_change: 0.0,
                relative_change_ci: (0.0, 0.0),
                t_statistic: 0.0,
                p_value: 1.0,
                degrees_of_freedom,
                variance_ratio: 1.0,
            }
        } else {
            BaselineTestOutcome {
                relative_change,
                relative_change_ci: (relative_change, relative_change),
                t_statistic: if recent_mean > baseline_mean_secs {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                },
                p_value: 0.0,
                degrees_of_freedom,
                variance_ratio: 1.0,
            }
        });
    }

    let se_baseline_sq = baseline_var / baseline_n as f64;
    let se_recent_sq = recent_var / n_recent as f64;
    let standard_error = (se_baseline_sq + se_recent_sq).sqrt();
    if standard_error <= 0.0 || !standard_error.is_finite() {
        return None;
    }

    let t_statistic = (recent_mean - baseline_mean_secs) / standard_error;

    // Welch-Satterthwaite equation for the effective degrees of freedom.
    let degrees_of_freedom = (se_baseline_sq + se_recent_sq).powi(2)
        / (se_baseline_sq.powi(2) / (baseline_n as f64 - 1.0)
            + se_recent_sq.powi(2) / (n_recent as f64 - 1.0));
    if !degrees_of_freedom.is_finite() || degrees_of_freedom <= 0.0 {
        return None;
    }

    let t_dist = StudentsT::new(0.0, 1.0, degrees_of_freedom).ok()?;
    // Two-tailed p-value: P(|T| >= |t_statistic|), from the workspace's own
    // regularized-incomplete-beta implementation rather than
    // `2 * (1 - statrs_cdf(|t|))`. The subtraction form loses every digit the
    // CDF carries above 1 - 2^-53 and underflows to exactly 0.0 in the upper
    // tail (measured: 0.0 at t=12, df=120, where the true p is 2.78e-22),
    // which then propagates into `statistical_significance = 1 - p` as a
    // fabricated perfect 1.0.
    let p_value = trustformers_core::statistics::student_t_two_sided_p_value(
        t_statistic,
        degrees_of_freedom,
    )?;

    let relative_change = if baseline_mean_secs != 0.0 {
        (recent_mean - baseline_mean_secs) / baseline_mean_secs
    } else {
        0.0
    };
    // 95% CI on the relative change, propagated from the same Welch
    // standard error (converted from absolute seconds to a fraction of the
    // baseline mean) via the t-distribution's critical value.
    let critical_value = t_dist.inverse_cdf(0.975);
    let relative_margin = if baseline_mean_secs != 0.0 {
        (critical_value * standard_error) / baseline_mean_secs.abs()
    } else {
        0.0
    };
    let variance_ratio = if baseline_var > 0.0 {
        recent_var / baseline_var
    } else if recent_var > 0.0 {
        f64::INFINITY
    } else {
        1.0
    };

    Some(BaselineTestOutcome {
        relative_change,
        relative_change_ci: (
            relative_change - relative_margin,
            relative_change + relative_margin,
        ),
        t_statistic,
        p_value,
        degrees_of_freedom,
        variance_ratio,
    })
}

/// Map a (non-negative) regression magnitude to a severity bucket using
/// the detector's own configured thresholds -- never a hardcoded cutoff.
///
/// `RegressionThresholds` has 4 fields for 4 [`RegressionSeverity`]
/// variants, but `minor_threshold` is consumed elsewhere as the entry gate
/// for "counts as a regression at all" (see
/// [`PerformanceRegressionDetector::check_regression`] /
/// [`detect_regression`]) -- everything reaching this function has
/// already cleared it, so it does not double as a bucket boundary here.
/// The 3 remaining fields give the 3 boundaries a 4-bucket split needs,
/// matching the ranges documented on [`RegressionSeverity`] itself
/// (Minor < moderate_threshold, Moderate < major_threshold, Major <
/// critical_threshold's *documented* range i.e. `major_threshold`,
/// Critical beyond that). `critical_threshold` is intentionally not
/// referenced here -- see its own doc comment.
pub fn classify_severity(magnitude: f64, thresholds: &RegressionThresholds) -> RegressionSeverity {
    if magnitude >= thresholds.major_threshold {
        RegressionSeverity::Critical
    } else if magnitude >= thresholds.moderate_threshold {
        RegressionSeverity::Major
    } else if magnitude >= thresholds.minor_threshold {
        RegressionSeverity::Moderate
    } else {
        RegressionSeverity::Minor
    }
}

/// Full real outcome of one regression check: the honest
/// [`BaselineComparison`] to publish via `get_status`, the trend
/// classification, and a [`RegressionAlert`] when (and only when) the
/// slowdown is both statistically significant and beyond
/// `thresholds.minor_threshold`.
pub struct RegressionCheckOutcome {
    pub baseline_comparison: BaselineComparison,
    pub performance_trend: PerformanceTrend,
    pub new_alert: Option<RegressionAlert>,
}

/// Observed (post-hoc) statistical power of the two-sided t-test that produced
/// `t_statistic` at `degrees_of_freedom`, tested at significance level `alpha`.
///
/// Power is `P(reject H0 | H1 true)`. Taking the observed effect as the
/// alternative makes the non-centrality parameter equal to `|t_statistic|`, so
/// with `t_crit` the two-sided critical value:
///
/// ```text
/// power = P(T' > t_crit) + P(T' < -t_crit),  T' ~ noncentral-t(df, |t|)
/// ```
///
/// which this approximates with the standard normal shift
/// `Phi(|t| - t_crit) + Phi(-|t| - t_crit)` -- the usual normal approximation
/// to the noncentral t, exact in the limit of large `df`.
///
/// Returns `None` when the inputs cannot support a test (non-finite `t`,
/// non-positive `df`, `alpha` outside `(0, 1)`).
///
/// This replaces `power: 1.0 - p_value`, which is not statistical power: it is
/// the confidence level of the observed result, the same quantity the sibling
/// `significance_level` field describes from the other side.
pub fn observed_power(t_statistic: f64, degrees_of_freedom: f64, alpha: f64) -> Option<f64> {
    if !t_statistic.is_finite() || !degrees_of_freedom.is_finite() || degrees_of_freedom <= 0.0 {
        return None;
    }
    if !(0.0..1.0).contains(&alpha) || alpha <= 0.0 {
        return None;
    }
    let t_dist = StudentsT::new(0.0, 1.0, degrees_of_freedom).ok()?;
    let t_crit = t_dist.inverse_cdf(1.0 - alpha / 2.0);
    if !t_crit.is_finite() {
        return None;
    }
    let lambda = t_statistic.abs();
    let upper = trustformers_core::statistics::normal_cdf(lambda - t_crit);
    let lower = trustformers_core::statistics::normal_cdf(-lambda - t_crit);
    Some((upper + lower).clamp(0.0, 1.0))
}

/// Real regression decision from a [`BaselineTestOutcome`], using the
/// detector's own configured `thresholds` for both the significance level
/// (`confidence_level`, e.g. 0.95 -> alpha = 0.05) and the severity
/// cutoffs -- never a hardcoded significance or a `has_regression` that
/// cannot become `true`.
pub fn detect_regression(
    kernel_name: &str,
    baseline: &BaselineProfile,
    outcome: BaselineTestOutcome,
    thresholds: &RegressionThresholds,
) -> RegressionCheckOutcome {
    let alpha = (1.0 - thresholds.confidence_level).clamp(0.0, 1.0);
    let is_significant = outcome.p_value < alpha;
    // A recent window whose variance is >=3x the baseline's is itself a
    // real, computed "got more erratic" signal, independent of whether
    // the mean shifted significantly. Deliberately ONE-SIDED: a recent
    // variance that SHRANK relative to baseline (ratio <= 1/3) means the
    // kernel became MORE consistent, the opposite of volatile -- that
    // case must fall through to the ordinary
    // Stable/Degrading/Improving classification below, not be mislabeled
    // "Volatile".
    let is_volatile = outcome.variance_ratio >= 3.0;

    let performance_trend = if is_volatile {
        PerformanceTrend::Volatile
    } else if !is_significant {
        PerformanceTrend::Stable
    } else if outcome.relative_change > 0.0 {
        PerformanceTrend::Degrading
    } else {
        PerformanceTrend::Improving
    };

    let magnitude = outcome.relative_change.abs();
    let new_alert = if is_significant
        && outcome.relative_change > 0.0
        && magnitude >= thresholds.minor_threshold
    {
        let current_performance = Duration::from_secs_f64(
            (baseline.baseline_performance.as_secs_f64() * (1.0 + outcome.relative_change))
                .max(0.0),
        );
        Some(RegressionAlert {
            alert_id: Uuid::new_v4(),
            kernel_name: kernel_name.to_string(),
            alert_type: RegressionType::PerformanceDegradation,
            severity: classify_severity(magnitude, thresholds),
            current_performance,
            baseline_performance: baseline.baseline_performance,
            regression_magnitude: magnitude,
            detection_timestamp: SystemTime::now(),
            potential_causes: vec![
                format!(
                    "Execution time {:.1}% slower than baseline ({:.3}ms -> {:.3}ms), \
                     Welch's t-test p={:.4} < alpha={:.4} (df={:.1})",
                    magnitude * 100.0,
                    baseline.baseline_performance.as_secs_f64() * 1000.0,
                    current_performance.as_secs_f64() * 1000.0,
                    outcome.p_value,
                    alpha,
                    outcome.degrees_of_freedom,
                ),
                "Check this kernel's launch-config / memory-access / compute-utilization \
                 analyses for a likely mechanism"
                    .to_string(),
            ],
        })
    } else {
        None
    };

    RegressionCheckOutcome {
        baseline_comparison: BaselineComparison {
            current_vs_baseline: outcome.relative_change * 100.0,
            statistical_significance: 1.0 - outcome.p_value,
            confidence_interval: outcome.relative_change_ci,
        },
        performance_trend,
        new_alert,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(
        block: (u32, u32, u32),
        registers: u32,
        shared: usize,
        occupancy: f64,
        compute: f64,
        bandwidth: f64,
        warp_eff: f64,
        mem_eff: f64,
    ) -> KernelProfileData {
        KernelProfileData {
            execution_time: std::time::Duration::from_micros(100),
            grid_size: (1024, 1, 1),
            block_size: block,
            shared_memory_bytes: shared,
            registers_per_thread: registers,
            occupancy,
            compute_utilization: compute,
            memory_bandwidth_utilization: bandwidth,
            warp_efficiency: warp_eff,
            memory_efficiency: mem_eff,
        }
    }

    #[test]
    fn occupancy_full_for_light_kernel() {
        let limits = AmpereDeviceLimits::sm_86();
        // 256 threads, 32 registers, no shared memory: should hit full occupancy.
        let est = estimate_occupancy(256, 32, 0, &limits);
        assert!(est.occupancy > 0.99, "occupancy = {}", est.occupancy);
        assert!(est.active_warps_per_sm <= limits.max_warps_per_sm);
    }

    #[test]
    fn occupancy_register_limited() {
        let limits = AmpereDeviceLimits::sm_86();
        // 128 registers/thread heavily constrains the register file.
        let est = estimate_occupancy(256, 128, 0, &limits);
        assert!(est.occupancy < 0.6, "occupancy = {}", est.occupancy);
        assert_eq!(est.limiter, OccupancyLimiter::Registers);
    }

    #[test]
    fn occupancy_zero_for_oversized_block() {
        let limits = AmpereDeviceLimits::sm_86();
        let est = estimate_occupancy(4096, 32, 0, &limits);
        assert_eq!(est.occupancy, 0.0);
        assert_eq!(est.active_blocks_per_sm, 0);
    }

    #[test]
    fn launch_config_low_occupancy_matmul_yields_recommendations() {
        // Register-heavy matmul tile: 256 threads, 128 regs → register-limited.
        let p = profile((256, 1, 1), 128, 0, 0.33, 0.7, 0.4, 0.95, 0.9);
        let opts = analyze_launch_config(&p);
        assert!(!opts.is_empty(), "expected launch-config recommendations");
        // A register optimization must be present for a register-limited kernel.
        assert!(opts
            .iter()
            .any(|o| matches!(o.optimization_type, OptimizationType::RegisterOptimization)));
        for o in &opts {
            assert!(o.confidence >= 0.0 && o.confidence <= 1.0);
            assert!(o.expected_improvement.performance_gain_percentage >= 0.0);
            assert!(o.expected_improvement.performance_gain_percentage <= 95.0);
        }
    }

    #[test]
    fn launch_config_partial_warp_flagged() {
        // 100 threads is not a multiple of 32.
        let p = profile((100, 1, 1), 32, 0, 0.5, 0.5, 0.5, 0.9, 0.9);
        let opts = analyze_launch_config(&p);
        assert!(opts.iter().any(|o| {
            matches!(o.optimization_type, OptimizationType::BlockSize)
                && matches!(o.suggested_value, OptimizationValue::IntegerValue(128))
        }));
    }

    #[test]
    fn memory_bound_gemv_recommendations() {
        // Poor coalescing + divergence + high bandwidth pressure.
        let p = profile((256, 1, 1), 32, 0, 0.55, 0.15, 0.9, 0.6, 0.4);
        let opts = analyze_memory_access(&p);
        assert!(!opts.is_empty(), "expected memory recommendations");
        assert!(opts
            .iter()
            .any(|o| matches!(o.optimization_type, OptimizationType::MemoryCoalescing)));
        assert!(opts
            .iter()
            .any(|o| matches!(o.optimization_type, OptimizationType::WarpDivergence)));
        for o in &opts {
            assert!(o.expected_improvement.performance_gain_percentage > 0.0);
        }
    }

    #[test]
    fn well_coalesced_kernel_has_no_memory_warnings() {
        let p = profile((256, 1, 1), 32, 0, 0.9, 0.5, 0.3, 0.98, 0.97);
        let opts = analyze_memory_access(&p);
        assert!(opts.is_empty(), "no memory issues expected, got {opts:?}");
    }

    #[test]
    fn compute_analyzer_memory_bound() {
        let p = profile((256, 1, 1), 32, 0, 0.55, 0.15, 0.9, 0.95, 0.5);
        let opts = analyze_compute_utilization(&p);
        assert!(!opts.is_empty());
        assert!(opts.iter().any(|o| matches!(
            o.optimization_type,
            OptimizationType::ComputeIntensityBalance
        )));
    }

    #[test]
    fn compute_analyzer_compute_bound_reduction() {
        let p = profile((256, 1, 1), 40, 4096, 0.8, 0.9, 0.3, 0.95, 0.9);
        let opts = analyze_compute_utilization(&p);
        assert!(!opts.is_empty());
        // High compute utilization at high intensity → compute-bound suggestion.
        assert!(opts.iter().any(|o| matches!(
            o.optimization_type,
            OptimizationType::ComputeIntensityBalance
        )));
    }

    #[test]
    fn compute_analyzer_latency_bound() {
        let p = profile((64, 1, 1), 32, 0, 0.25, 0.2, 0.2, 0.9, 0.9);
        let opts = analyze_compute_utilization(&p);
        assert!(opts.iter().any(|o| matches!(o.optimization_type, OptimizationType::GridSize)));
    }

    #[test]
    fn fusion_elementwise_chain() {
        let seq = vec!["bias_add".to_string(), "relu".to_string()];
        let opps = find_fusion_opportunities(&seq);
        assert_eq!(opps.len(), 1);
        assert!(matches!(opps[0].fusion_type, FusionType::ElementwiseFusion));
        assert!(opps[0].expected_speedup > 1.5);
        assert!(opps[0].memory_savings > 0);
        assert!(opps[0].fusion_feasibility.fusion_confidence > 0.0);
    }

    #[test]
    fn fusion_matmul_epilogue_chain() {
        let seq = vec!["matmul".to_string(), "bias".to_string(), "gelu".to_string()];
        let opps = find_fusion_opportunities(&seq);
        // (matmul, bias) → producer/consumer, (bias, gelu) → elementwise.
        assert_eq!(opps.len(), 2);
        assert!(matches!(
            opps[0].fusion_type,
            FusionType::ProducerConsumerFusion
        ));
        assert!(matches!(opps[1].fusion_type, FusionType::ElementwiseFusion));
        for o in &opps {
            assert!(o.expected_speedup > 1.0);
            assert_eq!(o.kernel_group.len(), 2);
        }
    }

    #[test]
    fn fusion_attention_pair() {
        let seq = vec!["attention_scores".to_string(), "softmax".to_string()];
        let opps = find_fusion_opportunities(&seq);
        assert_eq!(opps.len(), 1);
        assert!(matches!(opps[0].fusion_type, FusionType::AttentionFusion));
        assert!(opps[0].expected_speedup > 1.0);
    }

    #[test]
    fn fusion_skips_unrelated_and_short_sequences() {
        assert!(find_fusion_opportunities(&[]).is_empty());
        assert!(find_fusion_opportunities(&["matmul".to_string()]).is_empty());
        // matmul → matmul is not fused.
        let seq = vec!["matmul".to_string(), "gemm".to_string()];
        assert!(find_fusion_opportunities(&seq).is_empty());
    }

    #[test]
    fn registers_for_target_reduces_pressure() {
        let limits = AmpereDeviceLimits::sm_86();
        let p = profile((256, 1, 1), 128, 0, 0.33, 0.5, 0.5, 0.9, 0.9);
        let target = registers_for_target(&p, &limits).expect("should find a lower register count");
        assert!(target < 128);
        let est = estimate_occupancy(256, target, 0, &limits);
        assert!(est.occupancy >= limits.target_occupancy);
    }
}
