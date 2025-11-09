//! Platform-specific optimizations for TrustformeRS-C
//!
//! This module provides platform-specific optimizations for different
//! architectures and operating systems.

#[cfg(target_arch = "aarch64")]
pub mod arm64;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "linux")]
pub mod linux;

/// Platform information and capabilities
#[derive(Debug, Clone)]
pub struct PlatformInfo {
    pub architecture: String,
    pub operating_system: String,
    pub cpu_features: Vec<String>,
    pub has_gpu: bool,
    pub memory_mb: usize,
    pub cpu_cores: usize,
}

impl PlatformInfo {
    /// Detect current platform capabilities
    pub fn detect() -> Self {
        let architecture = std::env::consts::ARCH.to_string();
        let operating_system = std::env::consts::OS.to_string();

        let cpu_features = Self::detect_cpu_features();
        let has_gpu = Self::detect_gpu();
        let memory_mb = Self::detect_memory();
        let cpu_cores = Self::detect_cpu_cores();

        Self {
            architecture,
            operating_system,
            cpu_features,
            has_gpu,
            memory_mb,
            cpu_cores,
        }
    }

    fn detect_cpu_features() -> Vec<String> {
        let mut features = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                features.push("avx2".to_string());
            }
            if is_x86_feature_detected!("fma") {
                features.push("fma".to_string());
            }
            if is_x86_feature_detected!("sse4.1") {
                features.push("sse4.1".to_string());
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            features.push("neon".to_string());
            // Additional ARM64 features would be detected here
        }

        #[cfg(target_arch = "wasm32")]
        {
            #[cfg(target_feature = "simd128")]
            features.push("simd128".to_string());

            #[cfg(target_feature = "atomics")]
            features.push("atomics".to_string());

            #[cfg(target_feature = "bulk-memory")]
            features.push("bulk-memory".to_string());
        }

        features
    }

    fn detect_gpu() -> bool {
        #[cfg(target_os = "macos")]
        {
            // On macOS, most devices have Metal-capable GPUs
            true
        }
        #[cfg(target_os = "windows")]
        {
            // On Windows, assume GPU is available
            // In a real implementation, this would check for DirectX/OpenGL
            true
        }
        #[cfg(target_os = "linux")]
        {
            // On Linux, check for NVIDIA/AMD drivers
            // Simplified for now
            false
        }
        #[cfg(target_arch = "wasm32")]
        {
            // WASM can potentially use WebGL/WebGPU
            // Check for WebGPU support in browser environment
            false // Conservative default, can be detected at runtime
        }
        #[cfg(not(any(
            target_os = "macos",
            target_os = "windows",
            target_os = "linux",
            target_arch = "wasm32"
        )))]
        {
            false
        }
    }

    fn detect_memory() -> usize {
        // System memory detection
        // This is a simplified implementation
        #[cfg(target_arch = "wasm32")]
        {
            512 // 512MB conservative default for WASM
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            4096 // 4GB default for native platforms
        }
    }

    fn detect_cpu_cores() -> usize {
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
    }
}

/// Platform-specific optimization selector
pub struct PlatformOptimizer {
    info: PlatformInfo,
}

impl PlatformOptimizer {
    pub fn new() -> Self {
        Self {
            info: PlatformInfo::detect(),
        }
    }

    pub fn get_info(&self) -> &PlatformInfo {
        &self.info
    }

    /// Get the best matrix multiplication implementation for this platform
    pub fn get_matrix_multiply_fn(&self) -> fn(&[f32], &[f32], &mut [f32], usize, usize, usize) {
        match self.info.architecture.as_str() {
            "aarch64" => {
                #[cfg(target_arch = "aarch64")]
                {
                    |a, b, c, m, n, k| {
                        let optimizer = arm64::Arm64Optimizer::new();
                        optimizer.matrix_multiply(a, b, c, m, n, k);
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    Self::generic_matrix_multiply
                }
            },
            "x86_64" => {
                #[cfg(target_arch = "x86_64")]
                {
                    if self.info.cpu_features.contains(&"avx2".to_string()) {
                        x86_64::avx2_matrix_multiply
                    } else {
                        Self::generic_matrix_multiply
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    Self::generic_matrix_multiply
                }
            },
            "wasm32" => {
                #[cfg(target_arch = "wasm32")]
                {
                    |a, b, c, m, n, k| {
                        let optimizer = wasm::WasmOptimizer::new();
                        optimizer.matrix_multiply(a, b, c, m, n, k);
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    Self::generic_matrix_multiply
                }
            },
            _ => Self::generic_matrix_multiply,
        }
    }

    fn generic_matrix_multiply(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for ki in 0..k {
                    sum += a[i * k + ki] * b[ki * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
}

/// C API exports for platform information
#[no_mangle]
pub extern "C" fn trustformers_get_platform_info() -> *const PlatformInfo {
    let info = Box::new(PlatformInfo::detect());
    Box::into_raw(info)
}

#[no_mangle]
pub extern "C" fn trustformers_free_platform_info(info: *mut PlatformInfo) {
    if !info.is_null() {
        unsafe {
            let _ = Box::from_raw(info);
        }
    }
}

#[no_mangle]
pub extern "C" fn trustformers_get_architecture() -> *const std::os::raw::c_char {
    std::ffi::CString::new(std::env::consts::ARCH).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn trustformers_get_os() -> *const std::os::raw::c_char {
    std::ffi::CString::new(std::env::consts::OS).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn trustformers_get_cpu_cores() -> usize {
    PlatformInfo::detect_cpu_cores()
}
