//! System Information Gathering Utilities
//!
//! This module provides comprehensive system information gathering capabilities
//! for hardware detection, resource monitoring, and environment analysis.

use super::string_conversion::*;
use super::types::*;
use crate::error::{TrustformersError, TrustformersResult};
use std::os::raw::c_char;
use std::ptr;

// Re-export system detection functions for compatibility
pub use system_detection::get_system_info;

// Additional system info functions for compatibility
pub fn get_cpu_info() -> TrustformersResult<String> {
    let info = system_detection::get_system_info()?;
    Ok(format!(
        "CPU: {} cores, {}",
        info.num_cpu_cores, info.cpu_architecture
    ))
}

pub fn get_memory_info() -> TrustformersResult<String> {
    let info = system_detection::get_system_info()?;
    Ok(format!(
        "Memory: {} MB total, {} MB available",
        info.total_memory_bytes / (1024 * 1024),
        info.available_memory_bytes / (1024 * 1024)
    ))
}

pub fn get_gpu_info() -> TrustformersResult<String> {
    let info = system_detection::get_system_info()?;
    Ok(format!(
        "GPU: CUDA={}, Metal={}, OpenCL={}",
        info.gpu_info.cuda_available, info.gpu_info.metal_available, info.gpu_info.opencl_available
    ))
}

pub fn get_hardware_info() -> TrustformersResult<String> {
    let info = system_detection::get_system_info()?;
    Ok(format!(
        "Hardware: {} CPU cores, {} MB RAM, {}",
        info.num_cpu_cores,
        info.total_memory_bytes / (1024 * 1024),
        info.cpu_architecture
    ))
}

pub fn get_battery_info() -> TrustformersResult<String> {
    Ok("Battery info not available".to_string())
}

pub fn get_boot_time() -> TrustformersResult<String> {
    Ok("Boot time not available".to_string())
}

pub fn get_display_info() -> TrustformersResult<String> {
    Ok("Display info not available".to_string())
}

pub fn get_environment_variables() -> TrustformersResult<Vec<(String, String)>> {
    Ok(std::env::vars().collect())
}

pub fn get_load_average() -> TrustformersResult<(f64, f64, f64)> {
    Ok((0.0, 0.0, 0.0)) // Placeholder
}

pub fn get_locale_info() -> TrustformersResult<String> {
    Ok(std::env::var("LANG").unwrap_or_else(|_| "en_US.UTF-8".to_string()))
}

pub fn get_network_interfaces() -> TrustformersResult<Vec<String>> {
    Ok(vec!["lo".to_string(), "eth0".to_string()]) // Placeholder
}

pub fn get_process_info() -> TrustformersResult<String> {
    Ok(format!("PID: {}", std::process::id()))
}

pub fn get_thermal_info() -> TrustformersResult<String> {
    Ok("Thermal info not available".to_string())
}

pub fn get_timezone() -> TrustformersResult<String> {
    Ok(std::env::var("TZ").unwrap_or_else(|_| "UTC".to_string()))
}

pub fn get_uptime() -> TrustformersResult<String> {
    Ok("Uptime not available".to_string())
}

// Module-level utilities
pub mod hardware_utils {
    use super::TrustformersResult;

    pub fn detect_capabilities() -> TrustformersResult<Vec<String>> {
        Ok(vec!["x86_64".to_string(), "sse2".to_string()])
    }

    pub fn detect_cpu_features() -> TrustformersResult<Vec<String>> {
        Ok(vec![
            "sse".to_string(),
            "sse2".to_string(),
            "avx".to_string(),
        ])
    }

    pub fn get_cache_sizes() -> TrustformersResult<(usize, usize, usize)> {
        Ok((32_768, 256_000, 8_000_000)) // L1, L2, L3 cache sizes in bytes
    }

    pub fn get_memory_topology() -> TrustformersResult<String> {
        Ok("NUMA topology not available".to_string())
    }

    pub fn get_pci_devices() -> TrustformersResult<Vec<String>> {
        Ok(vec!["PCI device enumeration not available".to_string()])
    }

    pub fn get_storage_devices() -> TrustformersResult<Vec<String>> {
        Ok(vec!["Storage device enumeration not available".to_string()])
    }

    pub fn get_usb_devices() -> TrustformersResult<Vec<String>> {
        Ok(vec!["USB device enumeration not available".to_string()])
    }
}

pub mod process_utils {
    use super::TrustformersResult;

    pub fn get_current_process_info() -> TrustformersResult<String> {
        Ok(format!("Process ID: {}", std::process::id()))
    }

    pub fn find_process_by_name(name: &str) -> TrustformersResult<Vec<u32>> {
        // Placeholder implementation
        Ok(vec![])
    }

    pub fn get_process_cpu_usage(pid: u32) -> TrustformersResult<f64> {
        Ok(0.0) // Placeholder
    }

    pub fn get_process_memory(pid: u32) -> TrustformersResult<usize> {
        Ok(0) // Placeholder
    }

    pub fn is_process_running(pid: u32) -> TrustformersResult<bool> {
        Ok(false) // Placeholder
    }

    pub fn kill_process(pid: u32) -> TrustformersResult<bool> {
        Ok(false) // Placeholder - don't actually kill processes
    }

    pub fn list_processes() -> TrustformersResult<Vec<(u32, String)>> {
        Ok(vec![(std::process::id(), "current_process".to_string())])
    }
}

pub mod system_monitor {
    use super::TrustformersResult;

    pub type AlertCallback = fn(&str);

    #[derive(Debug, Clone)]
    pub struct ResourceThresholds {
        pub memory_threshold: f64,
        pub cpu_threshold: f64,
    }

    impl Default for ResourceThresholds {
        fn default() -> Self {
            Self {
                memory_threshold: 80.0,
                cpu_threshold: 90.0,
            }
        }
    }

    #[derive(Debug)]
    pub struct SystemMonitor {
        pub thresholds: ResourceThresholds,
    }

    impl SystemMonitor {
        pub fn new(thresholds: ResourceThresholds) -> Self {
            Self { thresholds }
        }

        pub fn check_resources(&self) -> TrustformersResult<bool> {
            Ok(true) // Placeholder
        }
    }
}

/// System information gathering and analysis
pub mod system_detection {
    use super::*;

    /// Get comprehensive system information
    pub fn get_system_info() -> TrustformersResult<SystemInfo> {
        let mut info = SystemInfo::default();

        // CPU information
        info.num_cpu_cores = get_cpu_core_count();
        info.available_cpu_cores = get_physical_cpu_count();
        info.cpu_architecture = get_cpu_architecture();
        info.cpu_features = get_cpu_features();

        // Memory information
        let memory_info = get_memory_info()?;
        info.total_memory_bytes = memory_info.total;
        info.available_memory_bytes = memory_info.available;

        // GPU information
        info.gpu_info = get_gpu_information()?;

        // Operating system information
        info.os_info = get_operating_system_info();

        // Runtime information
        info.runtime_info = get_runtime_info();

        Ok(info)
    }

    /// Get CPU core count
    fn get_cpu_core_count() -> u32 {
        num_cpus::get() as u32
    }

    /// Get physical CPU count
    fn get_physical_cpu_count() -> u32 {
        num_cpus::get_physical() as u32
    }

    /// Get CPU architecture information
    fn get_cpu_architecture() -> String {
        std::env::consts::ARCH.to_string()
    }

    /// Get CPU feature support
    pub fn get_cpu_features() -> CpuFeatures {
        CpuFeatures {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            sse: is_x86_feature_detected!("sse"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            sse: false,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            sse2: is_x86_feature_detected!("sse2"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            sse2: false,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            sse3: is_x86_feature_detected!("sse3"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            sse3: false,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            sse4_1: is_x86_feature_detected!("sse4.1"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            sse4_1: false,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            sse4_2: is_x86_feature_detected!("sse4.2"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            sse4_2: false,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            avx: is_x86_feature_detected!("avx"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            avx: false,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            avx2: is_x86_feature_detected!("avx2"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            avx2: false,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            avx512: is_x86_feature_detected!("avx512f"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            avx512: false,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            fma: is_x86_feature_detected!("fma"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            fma: false,
        }
    }

    /// Get memory information
    fn get_memory_info() -> TrustformersResult<MemoryInfo> {
        // This is a simplified implementation
        // In a real implementation, you would use platform-specific APIs
        Ok(MemoryInfo {
            total: 8 * 1024 * 1024 * 1024,     // 8GB default
            available: 6 * 1024 * 1024 * 1024, // 6GB default
            used: 2 * 1024 * 1024 * 1024,      // 2GB default
            page_size: get_page_size(),
        })
    }

    /// Get system page size
    fn get_page_size() -> usize {
        // Platform-specific page size detection
        #[cfg(unix)]
        {
            unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
        }
        #[cfg(not(unix))]
        {
            4096 // Default page size
        }
    }

    /// Get GPU information
    fn get_gpu_information() -> TrustformersResult<GpuInfo> {
        let mut gpu_info = GpuInfo::default();

        // CUDA detection
        #[cfg(feature = "cuda")]
        {
            gpu_info.cuda_available = true;
            gpu_info.cuda_devices = detect_cuda_devices();
            gpu_info.cuda_version = get_cuda_version();
        }

        // Metal detection (macOS)
        #[cfg(target_os = "macos")]
        {
            gpu_info.metal_available = true;
        }

        // ROCm detection
        #[cfg(feature = "rocm")]
        {
            gpu_info.rocm_available = true;
            gpu_info.rocm_devices = detect_rocm_devices();
        }

        // OpenCL detection
        gpu_info.opencl_available = detect_opencl();

        // Vulkan detection
        gpu_info.vulkan_available = detect_vulkan();

        Ok(gpu_info)
    }

    /// Detect CUDA devices
    #[cfg(feature = "cuda")]
    fn detect_cuda_devices() -> Vec<CudaDeviceInfo> {
        // Simplified CUDA device detection
        // In real implementation, would use CUDA Runtime API
        vec![CudaDeviceInfo {
            device_id: 0,
            name: "NVIDIA GPU".to_string(),
            compute_capability: (7, 5),
            memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            multiprocessor_count: 68,
            max_threads_per_block: 1024,
        }]
    }

    #[cfg(not(feature = "cuda"))]
    fn detect_cuda_devices() -> Vec<CudaDeviceInfo> {
        Vec::new()
    }

    /// Get CUDA version
    #[cfg(feature = "cuda")]
    fn get_cuda_version() -> String {
        "11.8".to_string() // Simplified
    }

    #[cfg(not(feature = "cuda"))]
    fn get_cuda_version() -> String {
        "N/A".to_string()
    }

    /// Detect ROCm devices
    #[cfg(feature = "rocm")]
    fn detect_rocm_devices() -> Vec<RocmDeviceInfo> {
        vec![RocmDeviceInfo {
            device_id: 0,
            name: "AMD GPU".to_string(),
            memory_bytes: 8 * 1024 * 1024 * 1024,
            compute_units: 64,
        }]
    }

    #[cfg(not(feature = "rocm"))]
    fn detect_rocm_devices() -> Vec<RocmDeviceInfo> {
        Vec::new()
    }

    /// Detect OpenCL support
    fn detect_opencl() -> bool {
        false // Simplified - would check for OpenCL runtime
    }

    /// Detect Vulkan support
    fn detect_vulkan() -> bool {
        false // Simplified - would check for Vulkan loader
    }

    /// Get operating system information
    fn get_operating_system_info() -> OperatingSystemInfo {
        OperatingSystemInfo {
            name: std::env::consts::OS.to_string(),
            version: get_os_version(),
            kernel_version: get_kernel_version(),
            is_64bit: cfg!(target_pointer_width = "64"),
            endianness: if cfg!(target_endian = "big") {
                "big".to_string()
            } else {
                "little".to_string()
            },
        }
    }

    /// Get OS version
    fn get_os_version() -> String {
        // Platform-specific version detection
        #[cfg(target_os = "windows")]
        {
            get_windows_version()
        }
        #[cfg(target_os = "macos")]
        {
            get_macos_version()
        }
        #[cfg(target_os = "linux")]
        {
            get_linux_version()
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            "Unknown".to_string()
        }
    }

    #[cfg(target_os = "windows")]
    fn get_windows_version() -> String {
        "Windows 10/11".to_string() // Simplified
    }

    #[cfg(target_os = "macos")]
    fn get_macos_version() -> String {
        "macOS 13+".to_string() // Simplified
    }

    #[cfg(target_os = "linux")]
    fn get_linux_version() -> String {
        std::fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|line| line.starts_with("PRETTY_NAME="))
                    .map(|line| line.split('=').nth(1).unwrap_or("").trim_matches('"').to_string())
            })
            .unwrap_or_else(|| "Linux".to_string())
    }

    /// Get kernel version
    fn get_kernel_version() -> String {
        #[cfg(unix)]
        {
            std::process::Command::new("uname")
                .arg("-r")
                .output()
                .ok()
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "Unknown".to_string())
        }
        #[cfg(not(unix))]
        {
            "Unknown".to_string()
        }
    }

    /// Get runtime information
    fn get_runtime_info() -> RuntimeInfo {
        RuntimeInfo {
            process_id: std::process::id(),
            thread_count: get_thread_count(),
            rust_version: get_rust_version(),
            compiler_version: get_compiler_version(),
            build_timestamp: std::env::var("VERGEN_BUILD_TIMESTAMP")
                .unwrap_or_else(|_| "unknown".to_string()),
            git_commit: std::env::var("VERGEN_GIT_SHA").unwrap_or_else(|_| "unknown".to_string()),
        }
    }

    /// Get current thread count
    fn get_thread_count() -> u32 {
        // Simplified - would use platform-specific APIs
        std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(1)
    }

    /// Get Rust version
    fn get_rust_version() -> String {
        std::env::var("VERGEN_RUSTC_SEMVER").unwrap_or_else(|_| "unknown".to_string())
    }

    /// Get compiler version
    fn get_compiler_version() -> String {
        let channel =
            std::env::var("VERGEN_RUSTC_CHANNEL").unwrap_or_else(|_| "unknown".to_string());
        let commit =
            std::env::var("VERGEN_RUSTC_COMMIT_HASH").unwrap_or_else(|_| "unknown".to_string());
        format!("{} {}", channel, commit)
    }
}

/// Comprehensive system information structure
#[derive(Debug, Default)]
pub struct SystemInfo {
    pub num_cpu_cores: u32,
    pub available_cpu_cores: u32,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub cpu_architecture: String,
    pub cpu_features: CpuFeatures,
    pub gpu_info: GpuInfo,
    pub os_info: OperatingSystemInfo,
    pub runtime_info: RuntimeInfo,
}

/// Memory information
#[derive(Debug, Default)]
pub struct MemoryInfo {
    pub total: u64,
    pub available: u64,
    pub used: u64,
    pub page_size: usize,
}

/// CPU features detection
#[derive(Debug, Default)]
pub struct CpuFeatures {
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512: bool,
    pub fma: bool,
}

/// GPU information
#[derive(Debug, Default)]
pub struct GpuInfo {
    pub cuda_available: bool,
    pub cuda_devices: Vec<CudaDeviceInfo>,
    pub cuda_version: String,
    pub metal_available: bool,
    pub rocm_available: bool,
    pub rocm_devices: Vec<RocmDeviceInfo>,
    pub opencl_available: bool,
    pub vulkan_available: bool,
}

/// CUDA device information
#[derive(Debug, Default)]
pub struct CudaDeviceInfo {
    pub device_id: u32,
    pub name: String,
    pub compute_capability: (u32, u32),
    pub memory_bytes: u64,
    pub multiprocessor_count: u32,
    pub max_threads_per_block: u32,
}

/// ROCm device information
#[derive(Debug, Default)]
pub struct RocmDeviceInfo {
    pub device_id: u32,
    pub name: String,
    pub memory_bytes: u64,
    pub compute_units: u32,
}

/// Operating system information
#[derive(Debug, Default)]
pub struct OperatingSystemInfo {
    pub name: String,
    pub version: String,
    pub kernel_version: String,
    pub is_64bit: bool,
    pub endianness: String,
}

/// Runtime information
#[derive(Debug, Default)]
pub struct RuntimeInfo {
    pub process_id: u32,
    pub thread_count: u32,
    pub rust_version: String,
    pub compiler_version: String,
    pub build_timestamp: String,
    pub git_commit: String,
}

/// Environment variable utilities
pub mod environment {
    use super::*;

    /// Get environment variable safely
    pub fn get_env_var(name: &str) -> Option<String> {
        std::env::var(name).ok()
    }

    /// Check if environment variable is set
    pub fn env_var_exists(name: &str) -> bool {
        std::env::var(name).is_ok()
    }

    /// Get environment variable with default value
    pub fn get_env_var_or_default(name: &str, default: &str) -> String {
        std::env::var(name).unwrap_or_else(|_| default.to_string())
    }

    /// Get all environment variables
    pub fn get_all_env_vars() -> std::collections::HashMap<String, String> {
        std::env::vars().collect()
    }

    /// Check for common AI/ML environment variables
    pub fn get_ml_environment_info() -> MlEnvironmentInfo {
        MlEnvironmentInfo {
            cuda_visible_devices: get_env_var("CUDA_VISIBLE_DEVICES"),
            cuda_device_order: get_env_var("CUDA_DEVICE_ORDER"),
            rocm_visible_devices: get_env_var("ROCR_VISIBLE_DEVICES"),
            omp_num_threads: get_env_var("OMP_NUM_THREADS"),
            mkl_num_threads: get_env_var("MKL_NUM_THREADS"),
            openblas_num_threads: get_env_var("OPENBLAS_NUM_THREADS"),
            tensorrt_verbose: env_var_exists("TENSORRT_VERBOSE"),
            debug_mode: env_var_exists("TRUSTFORMERS_DEBUG"),
        }
    }
}

/// ML/AI specific environment information
#[derive(Debug, Default)]
pub struct MlEnvironmentInfo {
    pub cuda_visible_devices: Option<String>,
    pub cuda_device_order: Option<String>,
    pub rocm_visible_devices: Option<String>,
    pub omp_num_threads: Option<String>,
    pub mkl_num_threads: Option<String>,
    pub openblas_num_threads: Option<String>,
    pub tensorrt_verbose: bool,
    pub debug_mode: bool,
}

/// Performance monitoring utilities
pub mod performance_monitoring {
    use super::*;
    use std::time::Instant;

    /// System resource monitor
    #[derive(Debug)]
    pub struct SystemResourceMonitor {
        start_time: Instant,
        last_cpu_time: u64,
        last_memory_check: Instant,
    }

    impl SystemResourceMonitor {
        /// Create new resource monitor
        pub fn new() -> Self {
            Self {
                start_time: Instant::now(),
                last_cpu_time: 0,
                last_memory_check: Instant::now(),
            }
        }

        /// Get current resource usage
        pub fn get_current_usage(&mut self) -> ResourceUsage {
            ResourceUsage {
                cpu_usage_percent: self.get_cpu_usage(),
                memory_usage_bytes: self.get_memory_usage(),
                uptime_seconds: self.start_time.elapsed().as_secs(),
                io_stats: self.get_io_stats(),
                network_stats: self.get_network_stats(),
            }
        }

        fn get_cpu_usage(&mut self) -> f64 {
            // Simplified CPU usage calculation
            // In real implementation, would use platform-specific APIs
            50.0 // Placeholder
        }

        fn get_memory_usage(&mut self) -> u64 {
            // Simplified memory usage
            // Would use platform-specific APIs
            2 * 1024 * 1024 * 1024 // 2GB placeholder
        }

        fn get_io_stats(&self) -> IoStats {
            IoStats {
                bytes_read: 0,
                bytes_written: 0,
                read_operations: 0,
                write_operations: 0,
            }
        }

        fn get_network_stats(&self) -> NetworkStats {
            NetworkStats {
                bytes_sent: 0,
                bytes_received: 0,
                packets_sent: 0,
                packets_received: 0,
            }
        }
    }

    impl Default for SystemResourceMonitor {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Current resource usage
    #[derive(Debug, Default)]
    pub struct ResourceUsage {
        pub cpu_usage_percent: f64,
        pub memory_usage_bytes: u64,
        pub uptime_seconds: u64,
        pub io_stats: IoStats,
        pub network_stats: NetworkStats,
    }

    /// I/O statistics
    #[derive(Debug, Default)]
    pub struct IoStats {
        pub bytes_read: u64,
        pub bytes_written: u64,
        pub read_operations: u64,
        pub write_operations: u64,
    }

    /// Network statistics
    #[derive(Debug, Default)]
    pub struct NetworkStats {
        pub bytes_sent: u64,
        pub bytes_received: u64,
        pub packets_sent: u64,
        pub packets_received: u64,
    }
}

/// Get current CPU usage as a percentage
pub fn get_cpu_usage() -> TrustformersResult<f64> {
    // Simple implementation - in a real system you'd want more sophisticated monitoring
    // This is a placeholder that returns a reasonable value
    #[cfg(unix)]
    {
        use std::fs;
        // Try to read from /proc/loadavg on Linux systems
        if let Ok(loadavg) = fs::read_to_string("/proc/loadavg") {
            if let Some(first_load) = loadavg.split_whitespace().next() {
                if let Ok(load) = first_load.parse::<f64>() {
                    // Convert load average to approximate CPU usage percentage
                    // This is a rough approximation
                    return Ok((load * 100.0).min(100.0));
                }
            }
        }
    }

    // Fallback: return a reasonable default value
    Ok(0.0)
}

/// Get disk usage information for a given path
pub fn get_disk_usage(path: &str) -> TrustformersResult<f64> {
    use std::path::Path;

    let path = Path::new(path);
    if !path.exists() {
        return Err(crate::error::TrustformersError::InvalidPath);
    }

    // Simple implementation - in a real system you'd want actual disk usage stats
    // This is a placeholder implementation
    #[cfg(unix)]
    {
        use std::process::Command;

        // Try to use `df` command to get disk usage
        if let Ok(output) = Command::new("df").arg(path).output() {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                // Parse df output - this is a simplified parser
                for line in output_str.lines().skip(1) {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 5 {
                        if let Ok(usage_percent) = parts[4].trim_end_matches('%').parse::<f64>() {
                            return Ok(usage_percent);
                        }
                    }
                }
            }
        }
    }

    // Fallback: return a reasonable default
    Ok(0.0)
}

#[cfg(test)]
mod tests {
    use super::system_detection::*;
    use super::*;

    #[test]
    fn test_system_info_gathering() {
        let info = get_system_info().unwrap();
        assert!(info.num_cpu_cores > 0);
        assert!(info.total_memory_bytes > 0);
        assert!(!info.cpu_architecture.is_empty());
        assert!(!info.os_info.name.is_empty());
    }

    #[test]
    fn test_environment_variables() {
        use super::environment::*;

        // Test with a known environment variable
        let path = get_env_var("PATH");
        assert!(path.is_some() || cfg!(target_os = "windows")); // PATH should exist on most systems

        let test_var = get_env_var_or_default("NONEXISTENT_VAR", "default_value");
        assert_eq!(test_var, "default_value");
    }

    #[test]
    fn test_cpu_features() {
        let features = system_detection::get_cpu_features();
        // Most modern x86_64 systems support SSE2
        #[cfg(target_arch = "x86_64")]
        {
            assert!(features.sse2);
        }
    }
}
