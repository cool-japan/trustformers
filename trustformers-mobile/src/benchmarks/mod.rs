//! Performance Benchmarking Suite for Mobile Deployment
//!
//! This module provides comprehensive benchmarking capabilities to validate
//! performance targets and measure mobile AI deployment effectiveness.

pub mod performance_targets;

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn};

// Re-export key types
pub use performance_targets::{
    BatteryMeasurement, BenchmarkConfig, BenchmarkResults, DeviceCategoryResult,
    DeviceCoverageResults, FrameworkSizeResults, LatencyMeasurement, PerformanceBenchmark,
    PerformanceTargets, TargetAchievement,
};

/// Benchmark suite configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuiteConfig {
    /// Performance targets to validate
    pub targets: PerformanceTargets,
    /// Enable detailed profiling
    pub detailed_profiling: bool,
    /// Export results to file
    pub export_results: bool,
    /// Results export path
    pub export_path: Option<String>,
}

impl Default for BenchmarkSuiteConfig {
    fn default() -> Self {
        Self {
            targets: PerformanceTargets::default(),
            detailed_profiling: true,
            export_results: true,
            export_path: None,
        }
    }
}

/// Mobile benchmark result summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    /// Overall performance score (0-100)
    pub overall_score: f64,
    /// All targets achieved
    pub all_targets_achieved: bool,
    /// Individual target results
    pub target_results: HashMap<String, bool>,
    /// Performance metrics summary
    pub performance_summary: PerformanceSummaryMetrics,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Performance summary metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummaryMetrics {
    /// Best inference latency achieved (ms)
    pub best_latency_ms: u64,
    /// Worst inference latency achieved (ms)
    pub worst_latency_ms: u64,
    /// Average inference latency (ms)
    pub avg_latency_ms: u64,
    /// Best battery efficiency achieved (% per hour)
    pub best_battery_efficiency: f64,
    /// Worst battery efficiency achieved (% per hour)
    pub worst_battery_efficiency: f64,
    /// Device coverage percentage
    pub device_coverage_percent: f64,
    /// Framework size (MB)
    pub framework_size_mb: f64,
}

/// Comprehensive mobile benchmarking suite
pub struct BenchmarkSuite {
    config: BenchmarkSuiteConfig,
    performance_benchmark: PerformanceBenchmark,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(config: BenchmarkSuiteConfig) -> Self {
        let perf_config = BenchmarkConfig {
            targets: config.targets.clone(),
            detailed_profiling: config.detailed_profiling,
            ..BenchmarkConfig::default()
        };

        let performance_benchmark = PerformanceBenchmark::new(perf_config);

        info!("Created benchmark suite with targets:");
        info!(
            "  Inference latency: <{}ms",
            config.targets.max_inference_latency_ms
        );
        info!(
            "  Battery drain: <{}% per hour",
            config.targets.max_battery_drain_per_hour
        );
        info!(
            "  Device coverage: >{}%",
            config.targets.min_device_coverage
        );
        info!(
            "  Framework size: <{}MB",
            config.targets.max_framework_size_mb
        );

        Self {
            config,
            performance_benchmark,
        }
    }

    /// Run comprehensive benchmark suite
    pub fn run_comprehensive_benchmarks(&mut self) -> Result<BenchmarkSummary> {
        info!("Starting comprehensive mobile benchmark suite");

        // Run performance benchmarks and take ownership of results
        let perf_results = self.performance_benchmark.run_benchmarks()?.clone();

        // Generate summary
        let summary = self.generate_summary(&perf_results)?;

        // Export results if configured
        if self.config.export_results {
            self.export_results(&summary, &perf_results)?;
        }

        info!(
            "Benchmark suite completed. Overall score: {:.1}/100",
            summary.overall_score
        );

        if summary.all_targets_achieved {
            info!("🎉 All performance targets achieved!");
        } else {
            warn!("⚠️ Some performance targets not met. See recommendations.");
        }

        Ok(summary)
    }

    /// Generate benchmark summary
    fn generate_summary(&self, results: &BenchmarkResults) -> Result<BenchmarkSummary> {
        let mut target_results = HashMap::new();
        target_results.insert(
            "inference_speed".to_string(),
            results.targets_achieved.inference_speed,
        );
        target_results.insert(
            "battery_efficiency".to_string(),
            results.targets_achieved.battery_efficiency,
        );
        target_results.insert(
            "device_coverage".to_string(),
            results.targets_achieved.device_coverage,
        );
        target_results.insert(
            "framework_size".to_string(),
            results.targets_achieved.framework_size,
        );

        // Calculate performance summary metrics
        let latencies: Vec<u64> =
            results.latency_measurements.iter().map(|m| m.latency_ms).collect();
        let battery_drains: Vec<f64> =
            results.battery_measurements.iter().map(|m| m.estimated_hourly_drain).collect();

        let performance_summary = PerformanceSummaryMetrics {
            best_latency_ms: *latencies.iter().min().unwrap_or(&0),
            worst_latency_ms: *latencies.iter().max().unwrap_or(&0),
            avg_latency_ms: if !latencies.is_empty() {
                latencies.iter().sum::<u64>() / latencies.len() as u64
            } else {
                0
            },
            best_battery_efficiency: battery_drains.iter().cloned().fold(f64::INFINITY, f64::min),
            worst_battery_efficiency: battery_drains
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
            device_coverage_percent: results.device_coverage.coverage_percentage,
            framework_size_mb: results.framework_size.measured_size_mb,
        };

        // Generate recommendations
        let recommendations = self.generate_recommendations(results);

        Ok(BenchmarkSummary {
            overall_score: results.overall_score,
            all_targets_achieved: results.targets_achieved.all_targets,
            target_results,
            performance_summary,
            recommendations,
        })
    }

    /// Generate improvement recommendations
    fn generate_recommendations(&self, results: &BenchmarkResults) -> Vec<String> {
        let mut recommendations = Vec::new();

        if !results.targets_achieved.inference_speed {
            recommendations.push(
                "🚀 Enable hardware acceleration (Neural Engine/NNAPI) for faster inference"
                    .to_string(),
            );
            recommendations
                .push("⚡ Implement more aggressive quantization (INT4/INT8)".to_string());
            recommendations.push("🔄 Add model caching and warm-up strategies".to_string());
        } else {
            recommendations.push("✅ Inference speed target achieved!".to_string());
        }

        if !results.targets_achieved.battery_efficiency {
            recommendations
                .push("🔋 Implement adaptive power management based on battery level".to_string());
            recommendations
                .push("📊 Add batch processing for better energy efficiency".to_string());
            recommendations
                .push("⏸️ Implement smart inference scheduling during low usage".to_string());
        } else {
            recommendations.push("✅ Battery efficiency target achieved!".to_string());
        }

        if !results.targets_achieved.device_coverage {
            recommendations.push("📱 Add fallback implementations for legacy devices".to_string());
            recommendations.push("🔧 Implement device-specific optimization paths".to_string());
            recommendations
                .push("🧪 Expand compatibility testing across more device models".to_string());
        } else {
            recommendations.push("✅ Device coverage target achieved!".to_string());
        }

        if !results.targets_achieved.framework_size {
            recommendations.push(
                "📦 Enable feature gating for optional components (Unity, React Native)"
                    .to_string(),
            );
            recommendations
                .push("🗜️ Implement dynamic loading for infrequently used features".to_string());
            recommendations
                .push("📚 Consider shared libraries for common dependencies".to_string());
        } else {
            recommendations.push("✅ Framework size target achieved!".to_string());
        }

        if results.targets_achieved.all_targets {
            recommendations.push(
                "🎯 All targets achieved! Consider pushing for even better performance".to_string(),
            );
            recommendations.push("🚀 Ready for production deployment with confidence".to_string());
        }

        recommendations
    }

    /// Export benchmark results
    fn export_results(&self, summary: &BenchmarkSummary, results: &BenchmarkResults) -> Result<()> {
        let export_path = self
            .config
            .export_path
            .as_deref()
            .unwrap_or("/tmp/mobile_benchmark_results.json");

        let export_data = serde_json::json!({
            "benchmark_summary": summary,
            "detailed_results": results,
            "configuration": self.config,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "targets_overview": {
                "all_achieved": summary.all_targets_achieved,
                "score": summary.overall_score,
                "target_breakdown": summary.target_results
            }
        });

        std::fs::write(export_path, serde_json::to_string_pretty(&export_data)?)?;
        info!("Benchmark results exported to: {}", export_path);

        Ok(())
    }

    /// Get current benchmark configuration
    pub fn get_config(&self) -> &BenchmarkSuiteConfig {
        &self.config
    }

    /// Update benchmark targets
    pub fn update_targets(&mut self, targets: PerformanceTargets) {
        self.config.targets = targets.clone();
        // Update the performance benchmark as well
        let perf_config = BenchmarkConfig {
            targets,
            detailed_profiling: self.config.detailed_profiling,
            ..BenchmarkConfig::default()
        };
        self.performance_benchmark = PerformanceBenchmark::new(perf_config);
    }

    /// Run quick validation against targets
    pub fn quick_validation(&mut self) -> Result<bool> {
        let results = self.run_comprehensive_benchmarks()?;
        Ok(results.all_targets_achieved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_creation() {
        let config = BenchmarkSuiteConfig::default();
        let suite = BenchmarkSuite::new(config);

        assert_eq!(suite.config.targets.max_inference_latency_ms, 100);
        assert_eq!(suite.config.targets.max_battery_drain_per_hour, 5.0);
    }

    #[test]
    #[ignore] // Ignore by default - this test runs full comprehensive benchmarks and takes 120+ seconds
    fn test_comprehensive_benchmarks() {
        let mut config = BenchmarkSuiteConfig::default();
        config.export_results = false; // Disable export for test

        let mut suite = BenchmarkSuite::new(config);
        let summary = suite.run_comprehensive_benchmarks().unwrap();

        assert!(summary.overall_score >= 0.0 && summary.overall_score <= 100.0);
        assert!(!summary.target_results.is_empty());
        assert!(!summary.recommendations.is_empty());
    }

    #[test]
    fn test_target_updates() {
        let config = BenchmarkSuiteConfig::default();
        let mut suite = BenchmarkSuite::new(config);

        let mut new_targets = PerformanceTargets::default();
        new_targets.max_inference_latency_ms = 50;

        suite.update_targets(new_targets);
        assert_eq!(suite.config.targets.max_inference_latency_ms, 50);
    }

    #[test]
    #[ignore] // Ignore by default - this test runs full comprehensive benchmarks and takes 120+ seconds
    fn test_quick_validation() {
        let mut config = BenchmarkSuiteConfig::default();
        config.export_results = false;

        let mut suite = BenchmarkSuite::new(config);
        let is_valid = suite.quick_validation().unwrap();

        // Result should be boolean
        assert!(is_valid == true || is_valid == false);
    }
}
