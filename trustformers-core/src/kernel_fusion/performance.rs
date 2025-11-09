//! Performance tracking and database for kernel fusion
//!
//! This module provides structures for tracking operation costs, device
//! characteristics, and fusion statistics for performance optimization.

use crate::kernel_fusion::graph::Device;
use crate::kernel_fusion::operation_types::OperationType;
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct PerformanceDatabase {
    pub operation_costs: HashMap<OperationType, OperationCost>,
    pub fusion_benefits: HashMap<String, f64>, // pattern hash -> speedup
    pub device_characteristics: HashMap<Device, DeviceCharacteristics>,
}

#[derive(Debug, Clone)]
pub struct OperationCost {
    pub ops_per_element: f64,
    pub memory_bandwidth_factor: f64,
    pub launch_overhead_ns: u64,
    pub parallelization_efficiency: f64,
}

#[derive(Debug, Clone)]
pub struct DeviceCharacteristics {
    pub peak_compute_ops: f64,      // GFLOPS
    pub memory_bandwidth_gbps: f64, // GB/s
    pub cache_size_kb: usize,
    pub warp_size: usize,
    pub max_threads_per_block: usize,
    pub register_file_size: usize,
}

#[derive(Debug, Clone, Default)]
pub struct FusionStatistics {
    pub total_fusions_attempted: u64,
    pub successful_fusions: u64,
    pub total_speedup: f64,
    pub memory_saved_bytes: u64,
    pub patterns_used: HashMap<String, u64>,
}

impl PerformanceDatabase {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_operation_cost(&mut self, op_type: OperationType, cost: OperationCost) {
        self.operation_costs.insert(op_type, cost);
    }

    pub fn get_operation_cost(&self, op_type: &OperationType) -> Option<&OperationCost> {
        self.operation_costs.get(op_type)
    }

    pub fn record_fusion_benefit(&mut self, pattern_hash: String, speedup: f64) {
        self.fusion_benefits.insert(pattern_hash, speedup);
    }

    pub fn get_fusion_benefit(&self, pattern_hash: &str) -> Option<f64> {
        self.fusion_benefits.get(pattern_hash).copied()
    }

    pub fn add_device_characteristics(
        &mut self,
        device: Device,
        characteristics: DeviceCharacteristics,
    ) {
        self.device_characteristics.insert(device, characteristics);
    }

    pub fn get_device_characteristics(&self, device: &Device) -> Option<&DeviceCharacteristics> {
        self.device_characteristics.get(device)
    }
}

impl OperationCost {
    pub fn new(ops_per_element: f64, memory_bandwidth_factor: f64) -> Self {
        Self {
            ops_per_element,
            memory_bandwidth_factor,
            launch_overhead_ns: 1000,        // Default 1µs
            parallelization_efficiency: 0.8, // Default 80% efficiency
        }
    }

    pub fn with_launch_overhead(mut self, overhead_ns: u64) -> Self {
        self.launch_overhead_ns = overhead_ns;
        self
    }

    pub fn with_parallelization_efficiency(mut self, efficiency: f64) -> Self {
        self.parallelization_efficiency = efficiency;
        self
    }
}

impl DeviceCharacteristics {
    pub fn new(peak_compute_ops: f64, memory_bandwidth_gbps: f64) -> Self {
        Self {
            peak_compute_ops,
            memory_bandwidth_gbps,
            cache_size_kb: 256,          // Default cache size
            warp_size: 32,               // Default warp size for NVIDIA GPUs
            max_threads_per_block: 1024, // Default max threads
            register_file_size: 65536,   // Default register file size
        }
    }

    pub fn cpu_characteristics() -> Self {
        Self {
            peak_compute_ops: 100.0,     // ~100 GFLOPS for modern CPU
            memory_bandwidth_gbps: 50.0, // ~50 GB/s for DDR4
            cache_size_kb: 32768,        // 32MB L3 cache
            warp_size: 1,                // No warp concept for CPU
            max_threads_per_block: std::thread::available_parallelism().unwrap().get(),
            register_file_size: 16 * 32, // 16 registers * 32 bits
        }
    }

    pub fn gpu_characteristics() -> Self {
        Self {
            peak_compute_ops: 10000.0,    // ~10 TFLOPS for modern GPU
            memory_bandwidth_gbps: 900.0, // ~900 GB/s for high-end GPU
            cache_size_kb: 6144,          // 6MB L2 cache for high-end GPU
            warp_size: 32,                // NVIDIA warp size
            max_threads_per_block: 1024,  // Max threads per block
            register_file_size: 65536,    // 64KB register file per SM
        }
    }

    pub fn is_compute_bound(&self, ops_per_byte: f64) -> bool {
        // If ops per byte is high, it's likely compute bound
        ops_per_byte > (self.peak_compute_ops / self.memory_bandwidth_gbps)
    }
}

impl FusionStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_fusion_attempt(&mut self) {
        self.total_fusions_attempted += 1;
    }

    pub fn record_successful_fusion(
        &mut self,
        pattern_name: &str,
        speedup: f64,
        memory_saved: u64,
    ) {
        self.successful_fusions += 1;
        self.total_speedup += speedup;
        self.memory_saved_bytes += memory_saved;
        *self.patterns_used.entry(pattern_name.to_string()).or_insert(0) += 1;
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_fusions_attempted == 0 {
            0.0
        } else {
            self.successful_fusions as f64 / self.total_fusions_attempted as f64
        }
    }

    pub fn average_speedup(&self) -> f64 {
        if self.successful_fusions == 0 {
            1.0
        } else {
            self.total_speedup / self.successful_fusions as f64
        }
    }
}
