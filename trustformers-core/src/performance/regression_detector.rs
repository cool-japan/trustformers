//! Performance regression detection system
//!
//! This module provides automated detection of performance regressions by comparing
//! current benchmark results with historical baselines.

use crate::errors::{performance_error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Performance baseline for a specific benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Benchmark name/identifier
    pub benchmark_name: String,
    /// Mean execution time in nanoseconds
    pub mean_time_ns: u64,
    /// Standard deviation in nanoseconds
    pub std_dev_ns: u64,
    /// Throughput in operations per second (if applicable)
    pub throughput_ops_per_sec: Option<f64>,
    /// Memory usage in bytes
    pub memory_usage_bytes: Option<u64>,
    /// Timestamp when baseline was recorded
    pub timestamp: u64,
    /// Git commit hash (if available)
    pub commit_hash: Option<String>,
    /// Hardware configuration when recorded
    pub hardware_config: HardwareConfig,
}

/// Hardware configuration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// CPU model information
    pub cpu_model: Option<String>,
    /// Number of CPU cores
    pub cpu_cores: u32,
    /// Available system memory in bytes
    pub system_memory_bytes: u64,
    /// GPU information (if available)
    pub gpu_info: Option<String>,
    /// Operating system
    pub os: String,
}

/// Performance regression detection result
#[derive(Debug, Clone)]
pub struct RegressionResult {
    /// Whether a regression was detected
    pub is_regression: bool,
    /// Severity of the regression (0.0 = no regression, 1.0 = severe)
    pub severity: f64,
    /// Performance change percentage (negative = improvement, positive = regression)
    pub performance_change_percent: f64,
    /// Current measurement
    pub current_measurement: PerformanceMeasurement,
    /// Baseline used for comparison
    pub baseline: PerformanceBaseline,
    /// Detailed analysis
    pub analysis: String,
}

/// Current performance measurement
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    /// Execution time in nanoseconds
    pub time_ns: u64,
    /// Throughput in operations per second (if applicable)
    pub throughput_ops_per_sec: Option<f64>,
    /// Memory usage in bytes
    pub memory_usage_bytes: Option<u64>,
}

/// Configuration for regression detection
#[derive(Debug, Clone)]
pub struct RegressionConfig {
    /// Threshold for detecting performance regression (e.g., 0.1 = 10% slower)
    pub regression_threshold: f64,
    /// Number of standard deviations to consider significant
    pub std_dev_threshold: f64,
    /// Whether to consider memory regressions
    pub check_memory_regression: bool,
    /// Whether to consider throughput regressions
    pub check_throughput_regression: bool,
    /// Minimum number of measurements needed for reliable detection
    pub min_measurements: usize,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            regression_threshold: 0.05, // 5% performance degradation
            std_dev_threshold: 2.0,     // 2 standard deviations
            check_memory_regression: true,
            check_throughput_regression: true,
            min_measurements: 5,
        }
    }
}

/// Performance regression detector
pub struct RegressionDetector {
    /// Configuration for detection
    config: RegressionConfig,
    /// Storage path for baselines
    storage_path: PathBuf,
    /// In-memory cache of baselines
    baselines: HashMap<String, PerformanceBaseline>,
}

impl RegressionDetector {
    /// Create a new regression detector
    pub fn new(storage_path: impl AsRef<Path>, config: RegressionConfig) -> Result<Self> {
        let storage_path = storage_path.as_ref().to_path_buf();

        // Create storage directory if it doesn't exist
        if let Some(parent) = storage_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                performance_error(format!(
                    "Failed to create baseline storage directory: {}",
                    e
                ))
            })?;
        }

        let mut detector = Self {
            config,
            storage_path,
            baselines: HashMap::new(),
        };

        // Load existing baselines
        detector.load_baselines()?;

        Ok(detector)
    }

    /// Record a new baseline measurement
    pub fn record_baseline(
        &mut self,
        benchmark_name: impl Into<String>,
        measurement: PerformanceMeasurement,
    ) -> Result<()> {
        let benchmark_name = benchmark_name.into();
        let hardware_config = Self::detect_hardware_config()?;

        let baseline = PerformanceBaseline {
            benchmark_name: benchmark_name.clone(),
            mean_time_ns: measurement.time_ns,
            std_dev_ns: 0, // Will be updated with multiple measurements
            throughput_ops_per_sec: measurement.throughput_ops_per_sec,
            memory_usage_bytes: measurement.memory_usage_bytes,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
            commit_hash: Self::get_git_commit_hash(),
            hardware_config,
        };

        self.baselines.insert(benchmark_name, baseline);
        self.save_baselines()?;

        Ok(())
    }

    /// Check for performance regression against baseline
    pub fn check_regression(
        &self,
        benchmark_name: &str,
        current_measurement: PerformanceMeasurement,
    ) -> Result<Option<RegressionResult>> {
        let baseline = match self.baselines.get(benchmark_name) {
            Some(baseline) => baseline,
            None => return Ok(None), // No baseline available
        };

        let time_change = if baseline.mean_time_ns > 0 {
            (current_measurement.time_ns as f64 - baseline.mean_time_ns as f64)
                / baseline.mean_time_ns as f64
        } else {
            0.0
        };

        let mut is_regression = false;
        let mut severity = 0.0;
        let mut analysis_parts = Vec::new();

        // Check execution time regression
        if time_change > self.config.regression_threshold {
            is_regression = true;
            severity = (time_change / self.config.regression_threshold).min(1.0);
            analysis_parts.push(format!(
                "Execution time increased by {:.1}% (threshold: {:.1}%)",
                time_change * 100.0,
                self.config.regression_threshold * 100.0
            ));
        }

        // Check throughput regression
        if self.config.check_throughput_regression {
            if let (Some(current_throughput), Some(baseline_throughput)) = (
                current_measurement.throughput_ops_per_sec,
                baseline.throughput_ops_per_sec,
            ) {
                let throughput_change =
                    (baseline_throughput - current_throughput) / baseline_throughput;
                if throughput_change > self.config.regression_threshold {
                    is_regression = true;
                    severity = severity
                        .max((throughput_change / self.config.regression_threshold).min(1.0));
                    analysis_parts.push(format!(
                        "Throughput decreased by {:.1}% (threshold: {:.1}%)",
                        throughput_change * 100.0,
                        self.config.regression_threshold * 100.0
                    ));
                }
            }
        }

        // Check memory usage regression
        if self.config.check_memory_regression {
            if let (Some(current_memory), Some(baseline_memory)) = (
                current_measurement.memory_usage_bytes,
                baseline.memory_usage_bytes,
            ) {
                let memory_change = if baseline_memory > 0 {
                    (current_memory as f64 - baseline_memory as f64) / baseline_memory as f64
                } else {
                    0.0
                };

                if memory_change > self.config.regression_threshold {
                    is_regression = true;
                    severity =
                        severity.max((memory_change / self.config.regression_threshold).min(1.0));
                    analysis_parts.push(format!(
                        "Memory usage increased by {:.1}% (threshold: {:.1}%)",
                        memory_change * 100.0,
                        self.config.regression_threshold * 100.0
                    ));
                }
            }
        }

        let analysis = if analysis_parts.is_empty() {
            "No performance regression detected".to_string()
        } else {
            analysis_parts.join("; ")
        };

        Ok(Some(RegressionResult {
            is_regression,
            severity,
            performance_change_percent: time_change * 100.0,
            current_measurement,
            baseline: baseline.clone(),
            analysis,
        }))
    }

    /// Update baseline with new measurement (incremental statistics)
    pub fn update_baseline(
        &mut self,
        benchmark_name: &str,
        measurement: PerformanceMeasurement,
    ) -> Result<()> {
        if let Some(baseline) = self.baselines.get_mut(benchmark_name) {
            // Simple moving average update (could be improved with proper incremental statistics)
            let old_mean = baseline.mean_time_ns as f64;
            let new_value = measurement.time_ns as f64;
            let new_mean = (old_mean + new_value) / 2.0;

            // Update standard deviation (simplified)
            let old_variance = (baseline.std_dev_ns as f64).powi(2);
            let new_variance = (old_variance + (new_value - old_mean).powi(2)) / 2.0;

            baseline.mean_time_ns = new_mean as u64;
            baseline.std_dev_ns = new_variance.sqrt() as u64;
            baseline.timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs();

            if let Some(throughput) = measurement.throughput_ops_per_sec {
                baseline.throughput_ops_per_sec = Some(throughput);
            }

            if let Some(memory) = measurement.memory_usage_bytes {
                baseline.memory_usage_bytes = Some(memory);
            }

            self.save_baselines()?;
        }

        Ok(())
    }

    /// Get all available baselines
    pub fn get_baselines(&self) -> &HashMap<String, PerformanceBaseline> {
        &self.baselines
    }

    /// Load baselines from storage
    fn load_baselines(&mut self) -> Result<()> {
        if !self.storage_path.exists() {
            return Ok(());
        }

        let content = std::fs::read_to_string(&self.storage_path)
            .map_err(|e| performance_error(format!("Failed to read baselines file: {}", e)))?;

        let baselines: HashMap<String, PerformanceBaseline> = serde_json::from_str(&content)
            .map_err(|e| performance_error(format!("Failed to parse baselines file: {}", e)))?;

        self.baselines = baselines;
        Ok(())
    }

    /// Save baselines to storage
    fn save_baselines(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.baselines)
            .map_err(|e| performance_error(format!("Failed to serialize baselines: {}", e)))?;

        std::fs::write(&self.storage_path, content)
            .map_err(|e| performance_error(format!("Failed to write baselines file: {}", e)))?;

        Ok(())
    }

    /// Detect current hardware configuration
    fn detect_hardware_config() -> Result<HardwareConfig> {
        Ok(HardwareConfig {
            cpu_model: Self::detect_cpu_model(),
            cpu_cores: num_cpus::get() as u32,
            system_memory_bytes: Self::get_system_memory(),
            gpu_info: Self::detect_gpu_info(),
            os: format!("{} {}", std::env::consts::OS, Self::get_os_version()),
        })
    }

    /// Get system memory size with proper detection
    fn get_system_memory() -> u64 {
        #[cfg(target_os = "linux")]
        {
            // Read from /proc/meminfo on Linux
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Use sysctl on macOS
            let output = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output();

            if let Ok(output) = output {
                if let Ok(memory_str) = String::from_utf8(output.stdout) {
                    if let Ok(memory_bytes) = memory_str.trim().parse::<u64>() {
                        return memory_bytes;
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // On Windows, we can use GlobalMemoryStatusEx
            // For now, fall back to default as we'd need windows-sys crate
        }

        // Fallback: estimate based on num_cpus (rough heuristic)
        let cpu_count = num_cpus::get() as u64;
        match cpu_count {
            1..=2 => 4 * 1024 * 1024 * 1024,  // 4GB for low-end systems
            3..=4 => 8 * 1024 * 1024 * 1024,  // 8GB for mid-range systems
            5..=8 => 16 * 1024 * 1024 * 1024, // 16GB for higher-end systems
            _ => 32 * 1024 * 1024 * 1024,     // 32GB for high-end systems
        }
    }

    /// Detect CPU model information
    fn detect_cpu_model() -> Option<String> {
        #[cfg(target_os = "linux")]
        {
            // Read from /proc/cpuinfo on Linux
            if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in cpuinfo.lines() {
                    if line.starts_with("model name") {
                        if let Some(model) = line.split(':').nth(1) {
                            return Some(model.trim().to_string());
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Use sysctl on macOS
            let output = std::process::Command::new("sysctl")
                .args(["-n", "machdep.cpu.brand_string"])
                .output();

            if let Ok(output) = output {
                if let Ok(cpu_model) = String::from_utf8(output.stdout) {
                    return Some(cpu_model.trim().to_string());
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Use wmic on Windows
            let output = std::process::Command::new("wmic")
                .args(["cpu", "get", "name", "/format:list"])
                .output();

            if let Ok(output) = output {
                if let Ok(cpu_info) = String::from_utf8(output.stdout) {
                    for line in cpu_info.lines() {
                        if line.starts_with("Name=") {
                            return Some(line[5..].trim().to_string());
                        }
                    }
                }
            }
        }

        None
    }

    /// Detect GPU information
    fn detect_gpu_info() -> Option<String> {
        #[cfg(target_os = "linux")]
        {
            // Try to detect NVIDIA GPU first
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=gpu_name", "--format=csv,noheader,nounits"])
                .output()
            {
                if output.status.success() {
                    if let Ok(gpu_name) = String::from_utf8(output.stdout) {
                        let gpu_name = gpu_name.trim();
                        if !gpu_name.is_empty() {
                            return Some(format!("NVIDIA {}", gpu_name));
                        }
                    }
                }
            }

            // Try to detect AMD GPU
            if let Ok(output) =
                std::process::Command::new("rocm-smi").args(["--showproductname"]).output()
            {
                if output.status.success() {
                    if let Ok(gpu_info) = String::from_utf8(output.stdout) {
                        // Parse rocm-smi output
                        for line in gpu_info.lines() {
                            if line.contains("Card series:") {
                                if let Some(series) = line.split(':').nth(1) {
                                    return Some(format!("AMD {}", series.trim()));
                                }
                            }
                        }
                    }
                }
            }

            // Fallback: check lspci for GPU info
            if let Ok(output) = std::process::Command::new("lspci").args(["-nn"]).output() {
                if output.status.success() {
                    if let Ok(pci_info) = String::from_utf8(output.stdout) {
                        for line in pci_info.lines() {
                            if line.contains("VGA compatible controller")
                                || line.contains("3D controller")
                            {
                                return Some(
                                    line.split(':')
                                        .last()
                                        .unwrap_or("Unknown GPU")
                                        .trim()
                                        .to_string(),
                                );
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Use system_profiler on macOS
            let output = std::process::Command::new("system_profiler")
                .args(["SPDisplaysDataType", "-xml"])
                .output();

            if let Ok(output) = output {
                if let Ok(display_info) = String::from_utf8(output.stdout) {
                    // Simple parsing for GPU name
                    if display_info.contains("Apple") {
                        if display_info.contains("M1") {
                            return Some("Apple M1 GPU".to_string());
                        } else if display_info.contains("M2") {
                            return Some("Apple M2 GPU".to_string());
                        } else if display_info.contains("M3") {
                            return Some("Apple M3 GPU".to_string());
                        } else {
                            return Some("Apple Silicon GPU".to_string());
                        }
                    }
                    // Could add more parsing for discrete GPUs
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Use wmic on Windows
            let output = std::process::Command::new("wmic")
                .args([
                    "path",
                    "win32_VideoController",
                    "get",
                    "name",
                    "/format:list",
                ])
                .output();

            if let Ok(output) = output {
                if let Ok(gpu_info) = String::from_utf8(output.stdout) {
                    for line in gpu_info.lines() {
                        if line.starts_with("Name=") && line.len() > 5 {
                            let name = &line[5..];
                            if !name.trim().is_empty() {
                                return Some(name.trim().to_string());
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Get OS version information
    fn get_os_version() -> String {
        #[cfg(target_os = "linux")]
        {
            // Try to read from /etc/os-release
            if let Ok(os_release) = std::fs::read_to_string("/etc/os-release") {
                let mut name = None;
                let mut version = None;

                for line in os_release.lines() {
                    if line.starts_with("NAME=") {
                        name = Some(line[5..].trim_matches('"').to_string());
                    } else if line.starts_with("VERSION=") {
                        version = Some(line[8..].trim_matches('"').to_string());
                    }
                }

                match (name, version) {
                    (Some(n), Some(v)) => return format!("{} {}", n, v),
                    (Some(n), None) => return n,
                    _ => {},
                }
            }

            // Fallback: try uname
            if let Ok(output) = std::process::Command::new("uname").args(["-r"]).output() {
                if let Ok(version) = String::from_utf8(output.stdout) {
                    return version.trim().to_string();
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(output) =
                std::process::Command::new("sw_vers").args(["-productVersion"]).output()
            {
                if let Ok(version) = String::from_utf8(output.stdout) {
                    return version.trim().to_string();
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            if let Ok(output) = std::process::Command::new("ver").output() {
                if let Ok(version) = String::from_utf8(output.stdout) {
                    return version.trim().to_string();
                }
            }
        }

        "Unknown".to_string()
    }

    /// Get current git commit hash
    fn get_git_commit_hash() -> Option<String> {
        // Simple implementation - could use git2 crate for better integration
        std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    String::from_utf8(output.stdout).ok().map(|s| s.trim().to_string())
                } else {
                    None
                }
            })
    }
}

/// Helper macro for easily recording performance measurements in tests
#[macro_export]
macro_rules! measure_performance {
    ($detector:expr, $benchmark_name:expr, $code:block) => {{
        let start = std::time::Instant::now();
        let result = $code;
        let duration = start.elapsed();

        let measurement = $crate::performance::regression_detector::PerformanceMeasurement {
            time_ns: duration.as_nanos() as u64,
            throughput_ops_per_sec: None,
            memory_usage_bytes: None,
        };

        let _ = $detector.record_baseline($benchmark_name, measurement);
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_regression_detector_creation() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("baselines.json");

        let detector = RegressionDetector::new(storage_path, RegressionConfig::default());
        assert!(detector.is_ok());
    }

    #[test]
    fn test_baseline_recording() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("baselines.json");

        let mut detector =
            RegressionDetector::new(storage_path, RegressionConfig::default()).unwrap();

        let measurement = PerformanceMeasurement {
            time_ns: 1_000_000, // 1ms
            throughput_ops_per_sec: Some(1000.0),
            memory_usage_bytes: Some(1024),
        };

        assert!(detector.record_baseline("test_benchmark", measurement).is_ok());
        assert!(detector.baselines.contains_key("test_benchmark"));
    }

    #[test]
    fn test_regression_detection() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("baselines.json");

        let mut detector =
            RegressionDetector::new(storage_path, RegressionConfig::default()).unwrap();

        // Record baseline
        let baseline_measurement = PerformanceMeasurement {
            time_ns: 1_000_000, // 1ms
            throughput_ops_per_sec: Some(1000.0),
            memory_usage_bytes: Some(1024),
        };
        detector.record_baseline("test_benchmark", baseline_measurement).unwrap();

        // Test with faster performance (no regression)
        let faster_measurement = PerformanceMeasurement {
            time_ns: 900_000, // 0.9ms - 10% improvement
            throughput_ops_per_sec: Some(1100.0),
            memory_usage_bytes: Some(1000),
        };

        let result = detector.check_regression("test_benchmark", faster_measurement).unwrap();
        assert!(result.is_some());
        assert!(!result.unwrap().is_regression);

        // Test with slower performance (regression)
        let slower_measurement = PerformanceMeasurement {
            time_ns: 1_200_000, // 1.2ms - 20% slower
            throughput_ops_per_sec: Some(800.0),
            memory_usage_bytes: Some(1200),
        };

        let result = detector.check_regression("test_benchmark", slower_measurement).unwrap();
        assert!(result.is_some());
        let regression = result.unwrap();
        assert!(regression.is_regression);
        assert!(regression.severity > 0.0);
    }
}
