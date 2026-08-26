use crate::errors::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Supported GPU backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum GpuBackend {
    /// NVIDIA CUDA backend
    Cuda,
    /// AMD ROCm backend
    Rocm,
    /// Apple Metal Performance Shaders
    #[default]
    Metal,
    /// Vulkan compute backend
    Vulkan,
    /// WebGPU for browser/WASM
    WebGpu,
    /// OpenCL backend
    OpenCl,
    /// Intel oneAPI backend
    Intel,
    /// CPU fallback
    Cpu,
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub id: usize,
    pub name: String,
    pub backend: GpuBackend,
    pub memory_total: u64,
    pub memory_free: u64,
    pub compute_capability: Option<String>,
    pub is_available: bool,
}

impl GpuDevice {
    /// Create a CPU device as fallback
    pub fn cpu() -> Self {
        Self {
            id: 0,
            name: "CPU".to_string(),
            backend: GpuBackend::Cpu,
            memory_total: 0,
            memory_free: 0,
            compute_capability: None,
            is_available: true,
        }
    }

    /// Check if this device supports tensor cores (NVIDIA)
    pub fn supports_tensor_cores(&self) -> bool {
        matches!(self.backend, GpuBackend::Cuda)
            && self.compute_capability.as_ref().map(|cc| cc.as_str() >= "7.0").unwrap_or(false)
    }

    /// Get device memory utilization ratio
    pub fn memory_utilization(&self) -> f32 {
        if self.memory_total == 0 {
            0.0
        } else {
            1.0 - (self.memory_free as f32 / self.memory_total as f32)
        }
    }
}

/// GPU memory pool for efficient allocation
#[derive(Debug)]
pub struct GpuMemoryPool {
    #[allow(dead_code)]
    backend: GpuBackend,
    allocated_blocks: HashMap<usize, u64>,
    free_blocks: Vec<(usize, u64)>,
    total_allocated: u64,
    peak_allocated: u64,
}

impl GpuMemoryPool {
    pub fn new(backend: GpuBackend) -> Self {
        Self {
            backend,
            allocated_blocks: HashMap::new(),
            free_blocks: Vec::new(),
            total_allocated: 0,
            peak_allocated: 0,
        }
    }

    /// Allocate memory block
    pub fn allocate(&mut self, size: u64) -> Result<usize> {
        // Find a free block of sufficient size
        if let Some(pos) = self.free_blocks.iter().position(|(_, block_size)| *block_size >= size) {
            let (ptr, block_size) = self.free_blocks.remove(pos);
            self.allocated_blocks.insert(ptr, size);

            // Split block if much larger than needed
            if block_size > size + 1024 {
                self.free_blocks.push((ptr + size as usize, block_size - size));
            }

            Ok(ptr)
        } else {
            // Allocate new block
            let ptr = self.allocated_blocks.len() + 1;
            self.allocated_blocks.insert(ptr, size);
            self.total_allocated += size;
            self.peak_allocated = self.peak_allocated.max(self.total_allocated);
            Ok(ptr)
        }
    }

    /// Deallocate memory block
    pub fn deallocate(&mut self, ptr: usize) -> Result<()> {
        if let Some(size) = self.allocated_blocks.remove(&ptr) {
            self.free_blocks.push((ptr, size));
            self.total_allocated -= size;
            Ok(())
        } else {
            Err(TrustformersError::tensor_op_error(
                "Invalid memory pointer",
                "deallocate",
            ))
        }
    }

    /// Get memory statistics
    pub fn stats(&self) -> (u64, u64, u64) {
        (
            self.total_allocated,
            self.peak_allocated,
            self.free_blocks.iter().map(|(_, size)| size).sum(),
        )
    }
}

/// GPU context for managing device operations
#[derive(Debug)]
pub struct GpuContext {
    pub device: GpuDevice,
    memory_pool: Arc<Mutex<GpuMemoryPool>>,
    stream_count: usize,
    async_enabled: bool,
}

impl GpuContext {
    /// Create a new GPU context for the given device
    pub fn new(device: GpuDevice) -> Result<Self> {
        let memory_pool = Arc::new(Mutex::new(GpuMemoryPool::new(device.backend)));

        Ok(Self {
            device,
            memory_pool,
            stream_count: 1,
            async_enabled: false,
        })
    }

    /// Create CPU-only context
    pub fn cpu() -> Self {
        Self {
            device: GpuDevice::cpu(),
            memory_pool: Arc::new(Mutex::new(GpuMemoryPool::new(GpuBackend::Cpu))),
            stream_count: 1,
            async_enabled: false,
        }
    }

    /// Enable asynchronous operations
    pub fn enable_async(&mut self, stream_count: usize) {
        self.async_enabled = true;
        self.stream_count = stream_count;
    }

    /// Allocate device memory
    pub fn allocate(&self, size: u64) -> Result<usize> {
        let mut pool = self.memory_pool.lock().map_err(|_| {
            TrustformersError::tensor_op_error("Failed to acquire memory pool lock", "gpu_memory")
        })?;
        pool.allocate(size)
    }

    /// Deallocate device memory
    pub fn deallocate(&self, ptr: usize) -> Result<()> {
        let mut pool = self.memory_pool.lock().map_err(|_| {
            TrustformersError::tensor_op_error("Failed to acquire memory pool lock", "gpu_memory")
        })?;
        pool.deallocate(ptr)
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> Result<(u64, u64, u64)> {
        let pool = self.memory_pool.lock().map_err(|_| {
            TrustformersError::tensor_op_error("Failed to acquire memory pool lock", "gpu_memory")
        })?;
        Ok(pool.stats())
    }

    /// Synchronize all operations on this context
    pub fn synchronize(&self) -> Result<()> {
        // Platform-specific synchronization would go here
        match self.device.backend {
            GpuBackend::Cuda => {
                // cudaDeviceSynchronize() equivalent
                Ok(())
            },
            GpuBackend::Rocm => {
                // hipDeviceSynchronize() equivalent
                Ok(())
            },
            GpuBackend::Metal => {
                // Metal command buffer wait until completed
                Ok(())
            },
            GpuBackend::Vulkan => {
                // vkQueueWaitIdle() equivalent
                Ok(())
            },
            _ => Ok(()),
        }
    }
}

/// GPU manager for device detection and context creation
#[derive(Debug)]
pub struct GpuManager {
    available_devices: Vec<GpuDevice>,
    active_contexts: HashMap<usize, Arc<GpuContext>>,
}

impl GpuManager {
    pub fn new() -> Self {
        let available_devices = Self::detect_devices();
        Self {
            available_devices,
            active_contexts: HashMap::new(),
        }
    }

    /// Detect available GPU devices
    fn detect_devices() -> Vec<GpuDevice> {
        // Always start with CPU as the fallback device. `mut` is used only when a
        // GPU-backend feature (metal/cuda/rocm/vulkan/…) appends to the list below;
        // with no such feature active (e.g. `cuda-oxicuda` alone) the list is never
        // appended to, so the `mut` is conditionally unused.
        #[allow(unused_mut)]
        let mut devices = vec![GpuDevice::cpu()];

        // Platform-specific device detection
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            // Detect Metal devices
            if let Ok(metal_devices) = Self::detect_metal_devices() {
                devices.extend(metal_devices);
            }
        }

        #[cfg(feature = "cuda")]
        {
            // Detect CUDA devices
            if let Ok(cuda_devices) = Self::detect_cuda_devices() {
                devices.extend(cuda_devices);
            }
        }

        #[cfg(feature = "rocm")]
        {
            // Detect ROCm devices
            if let Ok(rocm_devices) = Self::detect_rocm_devices() {
                devices.extend(rocm_devices);
            }
        }

        #[cfg(feature = "vulkan")]
        {
            // Detect Vulkan devices
            if let Ok(vulkan_devices) = Self::detect_vulkan_devices() {
                devices.extend(vulkan_devices);
            }
        }

        devices
    }

    /// Enumerate the system's Metal device.
    ///
    /// Queries the real device through the Metal backend. Without the `metal`
    /// feature no Metal bindings are linked and no device is reported — an
    /// empty list, never a placeholder "Apple GPU" with invented memory.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn detect_metal_devices() -> Result<Vec<GpuDevice>> {
        let Some(device) = metal::Device::system_default() else {
            return Ok(Vec::new());
        };

        // `recommended_max_working_set_size` is the amount of memory Metal will
        // let this process hold resident — the closest real figure to a device
        // memory budget that Metal exposes. Free memory is not exposed at all,
        // so it is reported as unknown (0) rather than guessed.
        let memory_total = device.recommended_max_working_set_size();

        Ok(vec![GpuDevice {
            id: 1,
            name: device.name().to_string(),
            backend: GpuBackend::Metal,
            memory_total,
            memory_free: 0,
            compute_capability: None,
            is_available: true,
        }])
    }

    /// Enumerate CUDA devices.
    ///
    /// The CUDA feature links the oxicuda driver bindings but this crate has no
    /// device-property query wired to them, so no device is reported rather
    /// than a fabricated one.
    #[cfg(feature = "cuda")]
    fn detect_cuda_devices() -> Result<Vec<GpuDevice>> {
        tracing::debug!(
            "CUDA device enumeration is not wired to the driver bindings; reporting no devices"
        );
        Ok(Vec::new())
    }

    /// Enumerate ROCm devices.
    ///
    /// No HIP device-property query is linked, so no device is reported. This
    /// previously invented two specific consumer cards.
    #[cfg(feature = "rocm")]
    fn detect_rocm_devices() -> Result<Vec<GpuDevice>> {
        tracing::debug!(
            "ROCm device enumeration is not wired to hipGetDeviceProperties; reporting no devices"
        );
        Ok(Vec::new())
    }

    /// Enumerate Vulkan devices through `vulkano`.
    #[cfg(feature = "vulkan")]
    fn detect_vulkan_devices() -> Result<Vec<GpuDevice>> {
        use vulkano::instance::{Instance, InstanceCreateInfo};
        use vulkano::VulkanLibrary;

        let Ok(library) = VulkanLibrary::new() else {
            // No Vulkan loader on this system.
            return Ok(Vec::new());
        };
        let Ok(instance) = Instance::new(library, InstanceCreateInfo::default()) else {
            return Ok(Vec::new());
        };
        let Ok(physical_devices) = instance.enumerate_physical_devices() else {
            return Ok(Vec::new());
        };

        let mut devices = Vec::new();
        for (index, physical) in physical_devices.enumerate() {
            let properties = physical.properties();
            // Device-local heap size is the real VRAM figure Vulkan reports.
            let memory_total = physical
                .memory_properties()
                .memory_heaps
                .iter()
                .filter(|heap| {
                    heap.flags.intersects(vulkano::memory::MemoryHeapFlags::DEVICE_LOCAL)
                })
                .map(|heap| heap.size)
                .max()
                .unwrap_or(0);

            devices.push(GpuDevice {
                id: 100 + index,
                name: properties.device_name.clone(),
                backend: GpuBackend::Vulkan,
                memory_total,
                // Vulkan exposes no free-memory query without an extension.
                memory_free: 0,
                compute_capability: Some(format!("Vulkan {}", properties.api_version)),
                is_available: true,
            });
        }

        Ok(devices)
    }

    /// Get all available devices
    pub fn available_devices(&self) -> &[GpuDevice] {
        &self.available_devices
    }

    /// Get the best available device
    pub fn best_device(&self) -> &GpuDevice {
        // Prefer GPU over CPU, and newer/more capable GPUs
        self.available_devices
            .iter()
            .filter(|d| d.is_available)
            .max_by_key(|d| {
                let backend_score = match d.backend {
                    GpuBackend::Cuda => 100,
                    GpuBackend::Metal => 90,
                    GpuBackend::Vulkan => 80,
                    GpuBackend::Rocm => 70,
                    GpuBackend::OpenCl => 60,
                    GpuBackend::WebGpu => 50,
                    GpuBackend::Intel => 40,
                    GpuBackend::Cpu => 10,
                };
                (backend_score, d.memory_total)
            })
            .unwrap_or(&self.available_devices[0])
    }

    /// Create context for specified device
    pub fn create_context(&mut self, device_id: usize) -> Result<Arc<GpuContext>> {
        let device =
            self.available_devices.iter().find(|d| d.id == device_id).cloned().ok_or_else(
                || {
                    TrustformersError::tensor_op_error(
                        &format!("Device {} not found", device_id),
                        "create_context",
                    )
                },
            )?;

        let context = Arc::new(GpuContext::new(device)?);
        self.active_contexts.insert(device_id, context.clone());
        Ok(context)
    }

    /// Get existing context or create new one
    pub fn get_or_create_context(&mut self, device_id: Option<usize>) -> Result<Arc<GpuContext>> {
        let device_id = device_id.unwrap_or_else(|| self.best_device().id);

        if let Some(context) = self.active_contexts.get(&device_id) {
            Ok(context.clone())
        } else {
            self.create_context(device_id)
        }
    }

    /// List available GPU devices (backward compatibility)
    pub fn list_devices() -> Result<Vec<GpuDevice>> {
        Ok(Self::detect_devices())
    }
}

impl Default for GpuManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global GPU manager instance
static GPU_MANAGER: OnceLock<Arc<Mutex<GpuManager>>> = OnceLock::new();

/// Get the global GPU manager
pub fn gpu_manager() -> Arc<Mutex<GpuManager>> {
    GPU_MANAGER.get_or_init(|| Arc::new(Mutex::new(GpuManager::new()))).clone()
}

/// Initialize GPU subsystem with optional device preference
pub fn init_gpu(preferred_backend: Option<GpuBackend>) -> Result<Arc<GpuContext>> {
    let manager = gpu_manager();
    let manager_lock = manager.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

    let device_id = if let Some(backend) = preferred_backend {
        manager_lock
            .available_devices()
            .iter()
            .find(|d| d.backend == backend && d.is_available)
            .map(|d| d.id)
    } else {
        Some(manager_lock.best_device().id)
    };

    let device_id = device_id.unwrap_or_else(|| manager_lock.best_device().id);
    drop(manager_lock); // Release the lock before calling get_or_create_context

    let mut manager_lock = manager.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    manager_lock.get_or_create_context(Some(device_id))
}

/// Trait for types that can be moved to GPU
pub trait ToGpu: Sized {
    type Output;

    /// Move this object to the specified GPU context
    fn to_gpu(&self, context: &GpuContext) -> Result<Self::Output>;

    /// Move this object back to CPU
    fn to_cpu(&self) -> Result<Self>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: the macOS path reported an 8 GiB "Apple GPU" without
    /// asking Metal anything, and the CUDA/ROCm/Vulkan paths invented specific
    /// cards. Without the corresponding feature, no GPU may be reported at all.
    #[test]
    fn test_no_gpu_is_invented_without_a_backend() {
        let devices = GpuManager::detect_devices();

        // The CPU fallback is always present and is honest about having no
        // dedicated memory.
        let cpu = devices
            .iter()
            .find(|device| device.backend == GpuBackend::Cpu)
            .expect("CPU fallback is always present");
        assert_eq!(cpu.memory_total, 0);

        for device in &devices {
            assert_ne!(
                device.name, "Apple GPU",
                "the placeholder Metal device must not reappear"
            );
            assert_ne!(device.name, "NVIDIA GPU");
            assert_ne!(device.name, "Vulkan GPU");
            assert_ne!(device.name, "AMD Radeon RX 6800 XT");
            assert_ne!(device.name, "AMD Radeon RX 7900 XTX");
        }

        // Without any GPU feature the only device is the CPU fallback.
        #[cfg(not(any(
            all(target_os = "macos", feature = "metal"),
            feature = "cuda",
            feature = "rocm",
            feature = "vulkan"
        )))]
        assert_eq!(
            devices.len(),
            1,
            "no GPU backend is compiled in, so no GPU may be enumerated: {devices:?}"
        );
    }

    /// `best_device` must never pick a device that was never enumerated.
    #[test]
    fn test_best_device_is_one_of_the_detected_devices() {
        let manager = GpuManager::new();
        let best = manager.best_device();
        assert!(
            manager.available_devices().iter().any(|device| device.name == best.name),
            "best_device must come from the detected list"
        );
    }

    #[test]
    fn test_gpu_device_creation() {
        let device = GpuDevice::cpu();
        assert_eq!(device.backend, GpuBackend::Cpu);
        assert!(device.is_available);
    }

    #[test]
    fn test_memory_pool_allocation() {
        let mut pool = GpuMemoryPool::new(GpuBackend::Cpu);

        let ptr1 = pool.allocate(1024).expect("operation failed in test");
        let ptr2 = pool.allocate(2048).expect("operation failed in test");

        assert_ne!(ptr1, ptr2);

        pool.deallocate(ptr1).expect("operation failed in test");
        pool.deallocate(ptr2).expect("operation failed in test");
    }

    #[test]
    fn test_gpu_context_creation() {
        let device = GpuDevice::cpu();
        let context = GpuContext::new(device).expect("operation failed in test");

        assert_eq!(context.device.backend, GpuBackend::Cpu);
        assert!(!context.async_enabled);
    }

    #[test]
    fn test_gpu_manager() {
        let manager = GpuManager::new();
        assert!(!manager.available_devices().is_empty());

        let best_device = manager.best_device();
        assert!(best_device.is_available);
    }

    #[test]
    fn test_gpu_backend_default() {
        let backend = GpuBackend::default();

        #[cfg(target_os = "macos")]
        assert_eq!(backend, GpuBackend::Metal);

        // On non-macOS platforms, any valid backend is acceptable
        // The actual backend depends on available hardware
        #[cfg(not(target_os = "macos"))]
        assert!(matches!(
            backend,
            GpuBackend::Cuda
                | GpuBackend::Rocm
                | GpuBackend::Vulkan
                | GpuBackend::Metal
                | GpuBackend::WebGpu
                | GpuBackend::Cpu
        ));
    }

    #[test]
    fn test_tensor_cores_support() {
        let cuda_device = GpuDevice {
            id: 1,
            name: "RTX 4090".to_string(),
            backend: GpuBackend::Cuda,
            memory_total: 24 * 1024 * 1024 * 1024,
            memory_free: 20 * 1024 * 1024 * 1024,
            compute_capability: Some("8.9".to_string()),
            is_available: true,
        };

        assert!(cuda_device.supports_tensor_cores());

        let old_cuda_device = GpuDevice {
            id: 2,
            name: "GTX 1080".to_string(),
            backend: GpuBackend::Cuda,
            memory_total: 8 * 1024 * 1024 * 1024,
            memory_free: 6 * 1024 * 1024 * 1024,
            compute_capability: Some("6.1".to_string()),
            is_available: true,
        };

        assert!(!old_cuda_device.supports_tensor_cores());
    }

    #[test]
    fn test_memory_utilization() {
        let device = GpuDevice {
            id: 1,
            name: "Test GPU".to_string(),
            backend: GpuBackend::Cuda,
            memory_total: 1000,
            memory_free: 300,
            compute_capability: None,
            is_available: true,
        };

        assert_eq!(device.memory_utilization(), 0.7);
    }

    #[test]
    fn test_gpu_initialization() {
        let context = init_gpu(None).expect("operation failed in test");
        assert!(context.device.is_available);
    }

    #[test]
    fn test_context_memory_operations() {
        let context = GpuContext::cpu();

        let ptr = context.allocate(1024).expect("operation failed in test");
        assert!(ptr > 0);

        let stats = context.memory_stats().expect("operation failed in test");
        assert_eq!(stats.0, 1024); // total allocated

        context.deallocate(ptr).expect("operation failed in test");

        let stats_after = context.memory_stats().expect("operation failed in test");
        assert_eq!(stats_after.0, 0); // total allocated after free
    }

    #[test]
    fn test_async_context() {
        let mut context = GpuContext::cpu();
        assert!(!context.async_enabled);

        context.enable_async(4);
        assert!(context.async_enabled);
        assert_eq!(context.stream_count, 4);
    }

    #[test]
    fn test_manager_context_management() {
        let mut manager = GpuManager::new();

        let context1 = manager.get_or_create_context(Some(0)).expect("operation failed in test");
        let context2 = manager.get_or_create_context(Some(0)).expect("operation failed in test");

        // Should return the same context for the same device
        assert!(Arc::ptr_eq(&context1, &context2));
    }
}
