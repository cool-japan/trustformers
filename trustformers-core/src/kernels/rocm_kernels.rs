// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! ROCm kernel operation surface for transformer computations.
//!
//! This module defines the operation shapes (matmul, flash attention,
//! layer norm, GELU, reduce-sum) that a real HIP kernel backend would
//! implement, plus reference HIP source generators for what each kernel
//! would look like. It does **not** contain a real ROCm/HIP runtime
//! binding: there is no `libloading`/`dlopen` call and no compiled kernel
//! ever actually runs on a GPU here. `kernels/rocm_impl.rs` is the module
//! that dlopen's the real `libamdhip64` runtime (behind `target_os =
//! "linux"`, since that is where ROCm ships); until every operation below
//! is routed through those real bindings, `enumerate_devices` honestly
//! reports zero devices and every operation returns a structured
//! [`TrustformersError::hardware_error`] instead of fabricating GPU
//! hardware or silently leaving its output tensor untouched.

#![allow(unused_variables)] // Operation shapes with reserved parameters for the future real backend

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Direct ROCm kernel operation surface for performance-critical computations
///
/// See the module docs: no HIP runtime is bound here, so every operation
/// returns an honest "backend unavailable" error rather than fabricated
/// results.
pub struct RocmKernel {
    /// HIP context. Always `None` today: no public constructor path in
    /// this module ever detects a real device (see `enumerate_devices`).
    #[allow(dead_code)]
    context: Option<HipContext>,
    /// Available GPU devices (always empty - see `enumerate_devices`).
    devices: Vec<RocmDevice>,
    /// Memory pools for different devices (always empty in lockstep with
    /// `devices`).
    memory_pools: HashMap<usize, Arc<Mutex<RocmMemoryPool>>>,
}

/// ROCm device information
#[derive(Debug, Clone)]
pub struct RocmDevice {
    pub id: usize,
    pub name: String,
    pub gfx_version: String,
    pub memory_total: u64,
    pub memory_free: u64,
    pub compute_unit_count: u32,
    pub max_threads_per_block: u32,
    pub wavefront_size: u32,
    pub max_shared_memory_per_block: u32,
}

/// HIP context for kernel execution
#[derive(Debug)]
pub struct HipContext {
    #[allow(dead_code)]
    device_id: usize,
    _stream: HipStream,
}

/// HIP stream for asynchronous operations
#[derive(Debug)]
pub struct HipStream {
    #[allow(dead_code)]
    id: usize,
    _priority: i32,
}

/// Memory pool for efficient GPU memory management
#[derive(Debug)]
pub struct RocmMemoryPool {
    #[allow(dead_code)]
    device_id: usize,
    _allocated_blocks: HashMap<usize, RocmMemoryBlock>,
    free_blocks: Vec<RocmMemoryBlock>,
    total_allocated: u64,
    peak_allocated: u64,
}

/// ROCm memory block
#[derive(Debug, Clone)]
pub struct RocmMemoryBlock {
    #[allow(dead_code)]
    ptr: usize,
    size: u64,
    _device_id: usize,
}

/// A compiled HIP kernel, as a real ROCm backend would represent one.
///
/// Nothing in this module constructs one today (see the module docs); the
/// type is kept as part of the public API shape for `kernels/rocm_impl.rs`
/// (or a future real backend) to produce.
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    #[allow(dead_code)]
    name: String,
    _hsaco_code: String,
    _function_name: String,
    _grid_size: (u32, u32, u32),
    _block_size: (u32, u32, u32),
    _shared_memory_size: u32,
}

/// ROCm kernel configuration
#[derive(Debug, Clone)]
pub struct KernelConfig {
    pub grid_size: (u32, u32, u32),
    pub block_size: (u32, u32, u32),
    pub shared_memory_size: u32,
    pub stream_id: Option<usize>,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            grid_size: (1, 1, 1),
            block_size: (256, 1, 1),
            shared_memory_size: 0,
            stream_id: None,
        }
    }
}

impl RocmKernel {
    /// Initialize the ROCm kernel operation surface.
    ///
    /// Always succeeds: enumerating zero devices is not itself an error
    /// (mirrors `hipGetDeviceCount` succeeding with `count == 0`).
    /// Operations that need a real device (`matmul`, `flash_attention`,
    /// ...) check device availability themselves and return
    /// [`TrustformersError::hardware_error`] when none exists, instead of
    /// running a fabricated pipeline.
    pub fn new() -> Result<Self> {
        let devices = Self::enumerate_devices()?;
        let context = if !devices.is_empty() { Some(HipContext::new(0)?) } else { None };

        let mut memory_pools = HashMap::new();
        for device in &devices {
            memory_pools.insert(
                device.id,
                Arc::new(Mutex::new(RocmMemoryPool::new(device.id)?)),
            );
        }

        Ok(Self {
            context,
            devices,
            memory_pools,
        })
    }

    /// Enumerate available ROCm devices.
    ///
    /// This module has no real HIP runtime binding (see the module docs),
    /// so it honestly reports zero devices rather than fabricating AMD
    /// hardware that may not exist on the host. `kernels/rocm_impl.rs`
    /// dlopen's the real `hipGetDeviceCount` on Linux; a future version of
    /// this function should delegate to it.
    fn enumerate_devices() -> Result<Vec<RocmDevice>> {
        Ok(Vec::new())
    }

    /// Return an error if no real ROCm device is available. Called first
    /// by every operation below so a caller gets a clear, structured
    /// failure instead of a fabricated result.
    fn ensure_device_available(&self) -> Result<()> {
        if self.devices.is_empty() {
            return Err(TrustformersError::hardware_error(
                "ROCm backend unavailable: no HIP device was detected (this module has no real \
                 HIP runtime binding - see kernels/rocm_impl.rs for the dlopen'd path on Linux, \
                 or use the oxiblas CPU path via gpu_ops::rocm)",
                "RocmKernel::ensure_device_available",
            ));
        }
        Ok(())
    }

    /// Matrix multiplication. Errors unless a real device is available
    /// (see [`RocmKernel::ensure_device_available`]); never fabricates a
    /// result or silently leaves `c` untouched while returning `Ok`.
    pub fn matmul(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        c: &mut Tensor,
        config: Option<KernelConfig>,
    ) -> Result<()> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let c_shape = c.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 || c_shape.len() != 2 {
            return Err(TrustformersError::tensor_op_error(
                "Matrix multiplication requires 2D tensors",
                "RocmKernels::gemm",
            ));
        }

        if a_shape[1] != b_shape[0] {
            return Err(TrustformersError::tensor_op_error(
                "Matrix dimensions incompatible for multiplication",
                "RocmKernels::gemm",
            ));
        }

        if c_shape[0] != a_shape[0] || c_shape[1] != b_shape[1] {
            return Err(TrustformersError::tensor_op_error(
                "Output matrix has incorrect dimensions",
                "RocmKernels::gemm",
            ));
        }

        self.ensure_device_available()
    }

    /// Flash attention. Errors unless a real device is available (see
    /// [`RocmKernel::ensure_device_available`]).
    pub fn flash_attention(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        output: &mut Tensor,
        config: Option<KernelConfig>,
    ) -> Result<()> {
        let q_shape = query.shape();
        let k_shape = key.shape();
        let v_shape = value.shape();

        if q_shape.len() != 3 || k_shape.len() != 3 || v_shape.len() != 3 {
            return Err(TrustformersError::tensor_op_error(
                "Flash attention requires 3D tensors",
                "RocmKernels::flash_attention",
            ));
        }

        if q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] {
            return Err(TrustformersError::tensor_op_error(
                "Batch dimensions must match for attention",
                "RocmKernels::flash_attention",
            ));
        }

        self.ensure_device_available()
    }

    /// Layer normalization. Errors unless a real device is available (see
    /// [`RocmKernel::ensure_device_available`]).
    pub fn layer_norm(
        &mut self,
        input: &Tensor,
        gamma: &Tensor,
        beta: &Tensor,
        output: &mut Tensor,
        epsilon: f32,
        config: Option<KernelConfig>,
    ) -> Result<()> {
        self.ensure_device_available()
    }

    /// Fused GELU activation. Errors unless a real device is available
    /// (see [`RocmKernel::ensure_device_available`]).
    pub fn fused_gelu(
        &mut self,
        input: &Tensor,
        output: &mut Tensor,
        config: Option<KernelConfig>,
    ) -> Result<()> {
        self.ensure_device_available()
    }

    /// Reduce-sum. Errors unless a real device is available (see
    /// [`RocmKernel::ensure_device_available`]).
    pub fn reduce_sum(
        &mut self,
        input: &Tensor,
        output: &mut Tensor,
        dim: usize,
        config: Option<KernelConfig>,
    ) -> Result<()> {
        self.ensure_device_available()
    }

    /// Get memory statistics for a device.
    pub fn get_memory_stats(&self, device_id: usize) -> Result<(u64, u64, u64)> {
        if let Some(pool) = self.memory_pools.get(&device_id) {
            let pool = pool.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            Ok(pool.stats())
        } else {
            Err(TrustformersError::tensor_op_error(
                &format!("Device {} not found", device_id),
                "RocmKernels::get_device",
            ))
        }
    }
}

impl HipContext {
    fn new(device_id: usize) -> Result<Self> {
        Ok(Self {
            device_id,
            _stream: HipStream {
                id: 0,
                _priority: 0,
            },
        })
    }
}

impl RocmMemoryPool {
    fn new(device_id: usize) -> Result<Self> {
        Ok(Self {
            device_id,
            _allocated_blocks: HashMap::new(),
            free_blocks: Vec::new(),
            total_allocated: 0,
            peak_allocated: 0,
        })
    }

    fn stats(&self) -> (u64, u64, u64) {
        (
            self.total_allocated,
            self.peak_allocated,
            self.free_blocks.iter().map(|b| b.size).sum(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rocm_kernel_creation_succeeds_with_zero_devices() {
        // Creating the kernel surface itself must not fail just because no
        // ROCm hardware exists (mirrors `hipGetDeviceCount` succeeding with
        // `count == 0`).
        let kernel = RocmKernel::new();
        assert!(kernel.is_ok());
    }

    /// Regression test: before this fix, `enumerate_devices` fabricated
    /// three specific AMD GPUs (RX 6800 XT, RX 7900 XTX, MI300X) on every
    /// machine, ROCm hardware or not.
    #[test]
    fn test_rocm_device_enumeration_reports_no_phantom_devices() {
        let devices = RocmKernel::enumerate_devices().expect("operation failed in test");
        assert!(
            devices.is_empty(),
            "must not fabricate AMD devices when no real HIP runtime is bound: got {:?}",
            devices
        );
    }

    /// Regression test: before this fix, `matmul` ran a fully simulated
    /// pipeline (fake compile/allocate/launch/copy) and returned `Ok(())`
    /// while leaving `c` completely untouched - silently wrong, not
    /// merely unimplemented. It must now return a structured error instead
    /// of claiming success.
    #[test]
    fn test_matmul_without_device_errors_instead_of_faking_success() {
        let mut kernel = RocmKernel::new().expect("operation failed in test");
        let a = Tensor::ones(&[2, 3]).expect("tensor creation failed");
        let b = Tensor::ones(&[3, 4]).expect("tensor creation failed");
        let mut c = Tensor::zeros(&[2, 4]).expect("tensor creation failed");

        let result = kernel.matmul(&a, &b, &mut c, None);
        assert!(
            result.is_err(),
            "matmul must error when no real ROCm device is available, not silently succeed"
        );
    }

    /// Regression test: before this fix, `flash_attention` similarly ran a
    /// fake pipeline and reported success without writing `output`.
    #[test]
    fn test_flash_attention_without_device_errors() {
        let mut kernel = RocmKernel::new().expect("operation failed in test");
        let q = Tensor::ones(&[1, 4, 8]).expect("tensor creation failed");
        let k = Tensor::ones(&[1, 4, 8]).expect("tensor creation failed");
        let v = Tensor::ones(&[1, 4, 8]).expect("tensor creation failed");
        let mut output = Tensor::zeros(&[1, 4, 8]).expect("tensor creation failed");

        let result = kernel.flash_attention(&q, &k, &v, &mut output, None);
        assert!(
            result.is_err(),
            "flash_attention must error without a real device"
        );
    }

    #[test]
    fn test_kernel_config_default() {
        let config = KernelConfig::default();
        assert_eq!(config.grid_size, (1, 1, 1));
        assert_eq!(config.block_size, (256, 1, 1));
        assert_eq!(config.shared_memory_size, 0);
    }
}
