//! Statistics tracking and utility functions for network adaptation.
//!
//! This module provides comprehensive statistics collection, analysis utilities,
//! and helper functions for optimizing network adaptation in mobile federated learning.

use std::collections::HashMap;

use super::types::{
    CompressionStats, FederatedTask, FederatedTaskType, GradientCompressionAlgorithm,
    NetworkAdaptationConfig, NetworkConditions, NetworkQuality,
};
use crate::profiler::NetworkConnectionType;

use crate::device_info::{MobileDeviceInfo, PerformanceTier};

/// Comprehensive statistics for network adaptation
#[derive(Debug, Clone)]
pub struct NetworkAdaptationStats {
    /// Total number of federated tasks scheduled
    pub total_tasks_scheduled: u64,
    /// Number of tasks completed successfully
    pub tasks_completed: u64,
    /// Number of tasks that failed
    pub tasks_failed: u64,
    /// Average completion time in milliseconds
    pub avg_completion_time_ms: f32,
    /// Data usage broken down by network type
    pub data_usage_by_network: HashMap<NetworkConnectionType, u64>,
    /// Compression statistics
    pub compression_stats: CompressionStats,
    /// Distribution of network quality assessments
    pub quality_distribution: HashMap<NetworkQuality, u32>,
    /// Accuracy of adaptation decisions
    pub adaptation_accuracy: f32,
    /// Battery impact in milliwatt-hours
    pub battery_impact_mwh: f32,
}

/// Utility functions for network adaptation optimization
pub struct NetworkAdaptationUtils;

/// Performance metrics for optimization
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Throughput in tasks per minute
    pub throughput_tasks_per_minute: f32,
    /// Average network utilization (0.0 to 1.0)
    pub network_utilization: f32,
    /// Battery efficiency in tasks per mAh
    pub battery_efficiency: f32,
    /// Compression efficiency ratio
    pub compression_efficiency: f32,
    /// Prediction accuracy
    pub prediction_accuracy: f32,
}

/// Network health assessment
#[derive(Debug, Clone)]
pub struct NetworkHealthAssessment {
    /// Overall health score (0.0 to 100.0)
    pub overall_health_score: f32,
    /// Individual metric scores
    pub bandwidth_score: f32,
    pub latency_score: f32,
    pub stability_score: f32,
    pub reliability_score: f32,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Data usage analysis
#[derive(Debug, Clone)]
pub struct DataUsageAnalysis {
    /// Total data usage in MB
    pub total_usage_mb: f32,
    /// Usage by task type
    pub usage_by_task_type: HashMap<FederatedTaskType, f32>,
    /// Usage trends over time
    pub usage_trends: Vec<(u64, f32)>, // (timestamp, cumulative_usage)
    /// Projected usage for next period
    pub projected_usage_mb: f32,
    /// Efficiency metrics
    pub bytes_per_completed_task: f32,
}

/// Adaptation optimization recommendations
#[derive(Debug, Clone)]
pub struct OptimizationRecommendations {
    /// Recommended configuration changes
    pub config_recommendations: Vec<ConfigRecommendation>,
    /// Priority level (1-10, higher is more urgent)
    pub priority: u8,
    /// Expected impact description
    pub expected_impact: String,
    /// Implementation complexity (1-5, higher is more complex)
    pub implementation_complexity: u8,
}

/// Individual configuration recommendation
#[derive(Debug, Clone)]
pub struct ConfigRecommendation {
    /// What to change
    pub parameter: String,
    /// Current value
    pub current_value: String,
    /// Recommended value
    pub recommended_value: String,
    /// Reasoning for change
    pub reasoning: String,
    /// Expected improvement percentage
    pub expected_improvement: f32,
}

impl NetworkAdaptationStats {
    /// Create new statistics tracker
    pub fn new() -> Self {
        Self {
            total_tasks_scheduled: 0,
            tasks_completed: 0,
            tasks_failed: 0,
            avg_completion_time_ms: 0.0,
            data_usage_by_network: HashMap::new(),
            compression_stats: CompressionStats::default(),
            quality_distribution: HashMap::new(),
            adaptation_accuracy: 0.0,
            battery_impact_mwh: 0.0,
        }
    }

    /// Record a scheduled task
    pub fn record_task_scheduled(&mut self, task: &FederatedTask) {
        self.total_tasks_scheduled += 1;

        // Update quality distribution if network conditions are available
        // This would typically be passed in with the task context
    }

    /// Record a completed task
    pub fn record_task_completed(
        &mut self,
        task: &FederatedTask,
        completion_time_ms: u64,
        data_used_bytes: u64,
        network_type: NetworkConnectionType,
    ) {
        self.tasks_completed += 1;

        // Update average completion time
        let total_time = self.avg_completion_time_ms * (self.tasks_completed - 1) as f32
            + completion_time_ms as f32;
        self.avg_completion_time_ms = total_time / self.tasks_completed as f32;

        // Update data usage by network type
        *self.data_usage_by_network.entry(network_type).or_insert(0) += data_used_bytes;

        // Update battery impact (simplified estimation)
        let estimated_battery_mwh = match network_type {
            NetworkConnectionType::WiFi => completion_time_ms as f32 * 0.1,
            NetworkConnectionType::Cellular4G => completion_time_ms as f32 * 0.4,
            NetworkConnectionType::Cellular5G => completion_time_ms as f32 * 0.3,
            NetworkConnectionType::Ethernet => completion_time_ms as f32 * 0.05,
            NetworkConnectionType::Offline => completion_time_ms as f32 * 0.0,
            NetworkConnectionType::Unknown => completion_time_ms as f32 * 0.3,
        };
        self.battery_impact_mwh += estimated_battery_mwh;
    }

    /// Record a failed task
    pub fn record_task_failed(&mut self, task: &FederatedTask, reason: &str) {
        self.tasks_failed += 1;
        // Could also track failure reasons for analysis
    }

    /// Update compression statistics
    pub fn update_compression_stats(&mut self, stats: CompressionStats) {
        self.compression_stats = stats;
    }

    /// Record quality assessment
    pub fn record_quality_assessment(&mut self, quality: NetworkQuality) {
        *self.quality_distribution.entry(quality).or_insert(0) += 1;
    }

    /// Update adaptation accuracy
    pub fn update_adaptation_accuracy(&mut self, predicted_outcome: f32, actual_outcome: f32) {
        let error = (predicted_outcome - actual_outcome).abs() / actual_outcome.max(1.0);
        let new_accuracy = 1.0 - error;

        // Update running average
        if self.adaptation_accuracy == 0.0 {
            self.adaptation_accuracy = new_accuracy;
        } else {
            self.adaptation_accuracy = (self.adaptation_accuracy * 0.9) + (new_accuracy * 0.1);
        }
    }

    /// Get success rate
    pub fn get_success_rate(&self) -> f32 {
        if self.total_tasks_scheduled == 0 {
            return 0.0;
        }
        (self.tasks_completed as f32) / (self.total_tasks_scheduled as f32)
    }

    /// Get failure rate
    pub fn get_failure_rate(&self) -> f32 {
        if self.total_tasks_scheduled == 0 {
            return 0.0;
        }
        (self.tasks_failed as f32) / (self.total_tasks_scheduled as f32)
    }

    /// Get total data usage in MB
    pub fn get_total_data_usage_mb(&self) -> f32 {
        let total_bytes: u64 = self.data_usage_by_network.values().sum();
        total_bytes as f32 / (1024.0 * 1024.0)
    }

    /// Get average data usage per task in MB
    pub fn get_avg_data_per_task_mb(&self) -> f32 {
        if self.tasks_completed == 0 {
            return 0.0;
        }
        self.get_total_data_usage_mb() / self.tasks_completed as f32
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        // Calculate throughput (assuming stats cover 1 hour period)
        let throughput = if self.avg_completion_time_ms > 0.0 {
            60000.0 / self.avg_completion_time_ms // tasks per minute
        } else {
            0.0
        };

        // Calculate network utilization (simplified)
        let network_utilization = if self.total_tasks_scheduled > 0 {
            self.get_success_rate() * 0.8 // Rough approximation
        } else {
            0.0
        };

        // Calculate battery efficiency
        let battery_efficiency = if self.battery_impact_mwh > 0.0 {
            self.tasks_completed as f32 / self.battery_impact_mwh
        } else {
            0.0
        };

        PerformanceMetrics {
            throughput_tasks_per_minute: throughput,
            network_utilization,
            battery_efficiency,
            compression_efficiency: 1.0 - self.compression_stats.compression_ratio,
            prediction_accuracy: self.adaptation_accuracy,
        }
    }

    /// Generate summary report
    pub fn generate_summary(&self) -> String {
        format!(
            "Network Adaptation Statistics Summary:\n\
             - Tasks Scheduled: {}\n\
             - Tasks Completed: {} ({:.1}% success rate)\n\
             - Tasks Failed: {} ({:.1}% failure rate)\n\
             - Avg Completion Time: {:.1}ms\n\
             - Total Data Usage: {:.2}MB\n\
             - Adaptation Accuracy: {:.1}%\n\
             - Battery Impact: {:.2}mWh\n\
             - Compression Ratio: {:.1}%",
            self.total_tasks_scheduled,
            self.tasks_completed,
            self.get_success_rate() * 100.0,
            self.tasks_failed,
            self.get_failure_rate() * 100.0,
            self.avg_completion_time_ms,
            self.get_total_data_usage_mb(),
            self.adaptation_accuracy * 100.0,
            self.battery_impact_mwh,
            self.compression_stats.compression_ratio * 100.0
        )
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl NetworkAdaptationUtils {
    /// Create optimized configuration for device and network conditions
    pub fn create_optimized_config(
        device_info: &MobileDeviceInfo,
        current_conditions: &NetworkConditions,
    ) -> NetworkAdaptationConfig {
        let mut config = NetworkAdaptationConfig::default();

        // Adjust based on connection type
        match current_conditions.connection_type {
            NetworkConnectionType::WiFi => {
                config.communication_strategy.wifi_strategy.enable_high_frequency_updates = true;
                config.sync_frequency.base_frequency_minutes = 30;
            },
            NetworkConnectionType::Cellular5G => {
                config.communication_strategy.cellular_strategy.g5_config.max_sync_size_mb = 50;
                config.sync_frequency.base_frequency_minutes = 60;
            },
            NetworkConnectionType::Cellular4G => {
                config.communication_strategy.cellular_strategy.g4_config.max_sync_size_mb = 20;
                config.sync_frequency.base_frequency_minutes = 120;
                config.communication_strategy.compression_config.model_compression_ratio = 0.5;
            },
            _ => {
                // Conservative settings for unknown connections
                config.sync_frequency.base_frequency_minutes = 180;
                config.communication_strategy.compression_config.model_compression_ratio = 0.3;
            },
        }

        // Adjust based on device performance tier
        match device_info.performance_scores.overall_tier {
            PerformanceTier::VeryLow => {
                config.communication_strategy.compression_config.enable_gradient_compression = true;
                config.communication_strategy.compression_config.gradient_compression_algo =
                    GradientCompressionAlgorithm::TopK { k: 10 };
                config.sync_frequency.base_frequency_minutes *= 4;
            },
            PerformanceTier::Low => {
                config.communication_strategy.compression_config.enable_gradient_compression = true;
                config.communication_strategy.compression_config.gradient_compression_algo =
                    GradientCompressionAlgorithm::TopK { k: 50 };
                config.sync_frequency.base_frequency_minutes *= 3;
            },
            PerformanceTier::Budget => {
                config.communication_strategy.compression_config.enable_gradient_compression = true;
                config.communication_strategy.compression_config.gradient_compression_algo =
                    GradientCompressionAlgorithm::TopK { k: 100 };
                config.sync_frequency.base_frequency_minutes *= 2;
            },
            PerformanceTier::Medium => {
                config.communication_strategy.compression_config.gradient_compression_algo =
                    GradientCompressionAlgorithm::Adaptive;
                config.sync_frequency.base_frequency_minutes =
                    (config.sync_frequency.base_frequency_minutes as f32 * 1.5) as u32;
            },
            PerformanceTier::Mid => {
                config.communication_strategy.compression_config.gradient_compression_algo =
                    GradientCompressionAlgorithm::Adaptive;
            },
            PerformanceTier::High | PerformanceTier::VeryHigh | PerformanceTier::Flagship => {
                config.enable_bandwidth_optimization = true;
                config.prediction_config.enable_ml_predictions = true;
                config.communication_strategy.wifi_strategy.max_concurrent_connections = 5;
            },
        }

        // Adjust based on battery level
        if let Some(battery_level) = device_info.power_info.battery_level_percent {
            if battery_level < 20 {
                // Aggressive power saving
                config.sync_frequency.base_frequency_minutes *= 3;
                config.communication_strategy.compression_config.model_compression_ratio *= 0.7;
            } else if battery_level < 50 {
                // Moderate power saving
                config.sync_frequency.base_frequency_minutes *= 2;
                config.communication_strategy.compression_config.model_compression_ratio *= 0.85;
            }
        }

        // Adjust based on thermal state
        match device_info.thermal_info.current_state {
            crate::device_info::ThermalState::Critical => {
                // Minimal activity
                config.sync_frequency.base_frequency_minutes *= 5;
                config.enable_adaptive_scheduling = false;
            },
            crate::device_info::ThermalState::Serious => {
                // Reduced activity
                config.sync_frequency.base_frequency_minutes *= 3;
            },
            crate::device_info::ThermalState::Fair => {
                // Slightly reduced activity
                config.sync_frequency.base_frequency_minutes *= 2;
            },
            _ => {
                // Normal operation
            },
        }

        config
    }

    /// Calculate network efficiency score (0.0 to 100.0)
    pub fn calculate_network_efficiency(conditions: &NetworkConditions) -> f32 {
        let bandwidth_score = (conditions.bandwidth_mbps / 100.0).min(1.0) * 30.0;
        let latency_score = ((200.0 - conditions.latency_ms) / 200.0).max(0.0) * 25.0;
        let stability_score = conditions.stability_score * 25.0;
        let loss_score = ((5.0 - conditions.packet_loss_percent) / 5.0).max(0.0) * 20.0;

        bandwidth_score + latency_score + stability_score + loss_score
    }

    /// Estimate data usage for federated task in bytes
    pub fn estimate_data_usage(task: &FederatedTask, compression_ratio: f32) -> usize {
        let base_size = task.estimated_size_mb;
        let compressed_size = (base_size as f32 * compression_ratio) as usize;

        match task.task_type {
            FederatedTaskType::ModelDownload => compressed_size * 1024 * 1024,
            FederatedTaskType::GradientUpload => compressed_size * 1024 * 1024 / 2, // Gradients are typically smaller
            FederatedTaskType::FullModelSync => compressed_size * 1024 * 1024,
            FederatedTaskType::IncrementalSync => compressed_size * 1024 * 1024 / 4, // Only diffs
            FederatedTaskType::Heartbeat => 1024, // Minimal data (1KB)
            FederatedTaskType::Checkpoint => compressed_size * 1024 * 1024 / 3, // Checkpoint metadata
        }
    }

    /// Determine optimal compression strategy for network conditions
    pub fn determine_compression_strategy(
        conditions: &NetworkConditions,
    ) -> GradientCompressionAlgorithm {
        match conditions.quality_assessment {
            NetworkQuality::Excellent => {
                if conditions.bandwidth_mbps > 50.0 {
                    GradientCompressionAlgorithm::None
                } else {
                    GradientCompressionAlgorithm::Quantized { bits: 8 }
                }
            },
            NetworkQuality::Good => GradientCompressionAlgorithm::Adaptive,
            NetworkQuality::Fair => GradientCompressionAlgorithm::TopK { k: 100 },
            NetworkQuality::Poor => GradientCompressionAlgorithm::TopK { k: 50 },
        }
    }

    /// Analyze network health and provide assessment
    pub fn analyze_network_health(conditions: &NetworkConditions) -> NetworkHealthAssessment {
        let bandwidth_score = (conditions.bandwidth_mbps / 100.0).min(1.0) * 100.0;
        let latency_score = ((200.0 - conditions.latency_ms) / 200.0).max(0.0) * 100.0;
        let stability_score = conditions.stability_score * 100.0;
        let reliability_score = ((5.0 - conditions.packet_loss_percent) / 5.0).max(0.0) * 100.0;

        let overall_health_score =
            (bandwidth_score + latency_score + stability_score + reliability_score) / 4.0;

        let mut recommendations = Vec::new();

        if bandwidth_score < 50.0 {
            recommendations
                .push("Consider using compression to reduce bandwidth usage".to_string());
        }
        if latency_score < 50.0 {
            recommendations.push(
                "High latency detected - consider scheduling less time-sensitive tasks".to_string(),
            );
        }
        if stability_score < 50.0 {
            recommendations
                .push("Network instability detected - implement retry mechanisms".to_string());
        }
        if reliability_score < 50.0 {
            recommendations.push(
                "High packet loss - consider switching to more reliable connection".to_string(),
            );
        }

        if overall_health_score > 80.0 {
            recommendations
                .push("Network conditions are excellent - can use full capabilities".to_string());
        }

        NetworkHealthAssessment {
            overall_health_score,
            bandwidth_score,
            latency_score,
            stability_score,
            reliability_score,
            recommendations,
        }
    }

    /// Analyze data usage patterns
    pub fn analyze_data_usage(stats: &NetworkAdaptationStats) -> DataUsageAnalysis {
        let total_usage_mb = stats.get_total_data_usage_mb();

        // Create usage by task type (simplified - would need more data in practice)
        let mut usage_by_task_type = HashMap::new();
        usage_by_task_type.insert(FederatedTaskType::ModelDownload, total_usage_mb * 0.4);
        usage_by_task_type.insert(FederatedTaskType::GradientUpload, total_usage_mb * 0.3);
        usage_by_task_type.insert(FederatedTaskType::FullModelSync, total_usage_mb * 0.2);
        usage_by_task_type.insert(FederatedTaskType::IncrementalSync, total_usage_mb * 0.08);
        usage_by_task_type.insert(FederatedTaskType::Heartbeat, total_usage_mb * 0.01);
        usage_by_task_type.insert(FederatedTaskType::Checkpoint, total_usage_mb * 0.01);

        // Simple trend analysis (would be more sophisticated in practice)
        let usage_trends = vec![
            (0, total_usage_mb * 0.2),
            (3600, total_usage_mb * 0.5),
            (7200, total_usage_mb * 0.8),
            (10800, total_usage_mb),
        ];

        // Project future usage based on current rate
        let projected_usage_mb = if stats.tasks_completed > 0 {
            total_usage_mb * 1.2 // 20% increase projection
        } else {
            0.0
        };

        let bytes_per_completed_task = if stats.tasks_completed > 0 {
            (total_usage_mb * 1024.0 * 1024.0) / stats.tasks_completed as f32
        } else {
            0.0
        };

        DataUsageAnalysis {
            total_usage_mb,
            usage_by_task_type,
            usage_trends,
            projected_usage_mb,
            bytes_per_completed_task,
        }
    }

    /// Generate optimization recommendations
    pub fn generate_optimization_recommendations(
        stats: &NetworkAdaptationStats,
        current_config: &NetworkAdaptationConfig,
        device_info: &MobileDeviceInfo,
    ) -> OptimizationRecommendations {
        let mut recommendations = Vec::new();
        let mut priority = 1u8;

        // Analyze success rate
        if stats.get_success_rate() < 0.8 {
            recommendations.push(ConfigRecommendation {
                parameter: "sync_frequency.base_frequency_minutes".to_string(),
                current_value: current_config.sync_frequency.base_frequency_minutes.to_string(),
                recommended_value: (current_config.sync_frequency.base_frequency_minutes * 2)
                    .to_string(),
                reasoning: "Low success rate indicates network stress - increase sync interval"
                    .to_string(),
                expected_improvement: 15.0,
            });
            priority = priority.max(7);
        }

        // Analyze battery impact
        if stats.battery_impact_mwh > 1000.0 {
            recommendations.push(ConfigRecommendation {
                parameter: "communication_strategy.compression_config.model_compression_ratio"
                    .to_string(),
                current_value: current_config
                    .communication_strategy
                    .compression_config
                    .model_compression_ratio
                    .to_string(),
                recommended_value: (current_config
                    .communication_strategy
                    .compression_config
                    .model_compression_ratio
                    * 0.8)
                    .to_string(),
                reasoning: "High battery impact - increase compression to reduce transmission time"
                    .to_string(),
                expected_improvement: 20.0,
            });
            priority = priority.max(6);
        }

        // Analyze data usage efficiency
        let avg_data_per_task = stats.get_avg_data_per_task_mb();
        if avg_data_per_task > 10.0 {
            recommendations.push(ConfigRecommendation {
                parameter: "communication_strategy.compression_config.enable_gradient_compression"
                    .to_string(),
                current_value: current_config
                    .communication_strategy
                    .compression_config
                    .enable_gradient_compression
                    .to_string(),
                recommended_value: "true".to_string(),
                reasoning: "High data usage per task - enable gradient compression".to_string(),
                expected_improvement: 30.0,
            });
            priority = priority.max(8);
        }

        // Analyze completion time
        if stats.avg_completion_time_ms > 60000.0 {
            // > 1 minute
            recommendations.push(ConfigRecommendation {
                parameter: "enable_bandwidth_optimization".to_string(),
                current_value: current_config.enable_bandwidth_optimization.to_string(),
                recommended_value: "true".to_string(),
                reasoning: "Long completion times - enable bandwidth optimization".to_string(),
                expected_improvement: 25.0,
            });
            priority = priority.max(5);
        }

        let expected_impact = if recommendations.is_empty() {
            "Current configuration appears optimal".to_string()
        } else {
            format!(
                "Expected overall improvement: {:.1}%",
                recommendations.iter().map(|r| r.expected_improvement).sum::<f32>()
                    / recommendations.len() as f32
            )
        };

        let implementation_complexity = if recommendations.len() > 3 {
            4 // High complexity
        } else if recommendations.len() > 1 {
            3 // Medium complexity
        } else {
            2 // Low complexity
        };

        OptimizationRecommendations {
            config_recommendations: recommendations,
            priority,
            expected_impact,
            implementation_complexity,
        }
    }

    /// Calculate optimal sync frequency based on conditions
    pub fn calculate_optimal_sync_frequency(
        device_info: &MobileDeviceInfo,
        network_conditions: &NetworkConditions,
        current_stats: &NetworkAdaptationStats,
    ) -> u32 {
        #[allow(dead_code)]
        let mut base_frequency = 60u32; // Default 60 minutes

        // Adjust for network quality
        match network_conditions.quality_assessment {
            NetworkQuality::Excellent => base_frequency = 30,
            NetworkQuality::Good => base_frequency = 45,
            NetworkQuality::Fair => base_frequency = 90,
            NetworkQuality::Poor => base_frequency = 180,
        }

        // Adjust for device performance
        match device_info.performance_scores.overall_tier {
            PerformanceTier::Flagship | PerformanceTier::VeryHigh => {
                base_frequency = (base_frequency as f32 * 0.7) as u32
            },
            PerformanceTier::High => base_frequency = (base_frequency as f32 * 0.8) as u32,
            PerformanceTier::Medium | PerformanceTier::Mid => {}, // No change
            PerformanceTier::Budget | PerformanceTier::Low => {
                base_frequency = (base_frequency as f32 * 1.5) as u32
            },
            PerformanceTier::VeryLow => base_frequency = (base_frequency as f32 * 2.0) as u32,
        }

        // Adjust for battery level
        if let Some(battery_level) = device_info.power_info.battery_level_percent {
            if battery_level < 20 {
                base_frequency *= 3;
            } else if battery_level < 50 {
                base_frequency *= 2;
            }
        }

        // Adjust based on historical success rate
        if current_stats.total_tasks_scheduled > 10 {
            let success_rate = current_stats.get_success_rate();
            if success_rate < 0.5 {
                base_frequency *= 3; // Much longer intervals for poor success rate
            } else if success_rate < 0.8 {
                base_frequency = (base_frequency as f32 * 1.5) as u32;
            }
        }

        base_frequency.max(15).min(720) // Clamp between 15 minutes and 12 hours
    }
}

// Default implementations for convenience
impl Default for NetworkAdaptationStats {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            throughput_tasks_per_minute: 0.0,
            network_utilization: 0.0,
            battery_efficiency: 0.0,
            compression_efficiency: 0.0,
            prediction_accuracy: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_info::{
        BasicDeviceInfo, ChargingStatus, CpuInfo, MemoryInfo, PerformanceScores, PerformanceTier,
        PowerInfo, ThermalInfo, ThermalState,
    };
    use crate::network_adaptation::types::{TaskPriority, TaskStatus};
    use crate::MobilePlatform;

    fn create_test_device_info() -> MobileDeviceInfo {
        MobileDeviceInfo {
            platform: MobilePlatform::Generic,
            basic_info: BasicDeviceInfo {
                platform: MobilePlatform::Generic,
                manufacturer: "Test".to_string(),
                model: "TestDevice".to_string(),
                os_version: "1.0".to_string(),
                hardware_id: "test123".to_string(),
                device_generation: Some(2023),
            },
            cpu_info: CpuInfo {
                architecture: "arm64".to_string(),
                total_cores: 8,
                core_count: 8,
                performance_cores: 4,
                efficiency_cores: 4,
                max_frequency_mhz: Some(3000),
                l1_cache_kb: Some(64),
                l2_cache_kb: Some(512),
                l3_cache_kb: Some(8192),
                features: vec!["NEON".to_string()],
                simd_support: crate::device_info::SimdSupport::Advanced,
            },
            memory_info: MemoryInfo {
                total_mb: 4096,
                available_mb: 2048,
                total_memory: 4096,
                available_memory: 2048,
                bandwidth_mbps: Some(25600),
                memory_type: "LPDDR5".to_string(),
                frequency_mhz: Some(6400),
                is_low_memory_device: false,
            },
            gpu_info: None,
            npu_info: None,
            thermal_info: ThermalInfo {
                current_state: ThermalState::Nominal,
                state: ThermalState::Nominal,
                throttling_supported: true,
                temperature_sensors: vec![],
                thermal_zones: vec![],
            },
            power_info: PowerInfo {
                battery_capacity_mah: Some(3000),
                battery_level_percent: Some(75),
                battery_level: Some(75),
                battery_health_percent: Some(95),
                charging_status: ChargingStatus::NotCharging,
                is_charging: false,
                power_save_mode: false,
                low_power_mode_available: true,
            },
            available_backends: vec![crate::MobileBackend::CPU],
            performance_scores: PerformanceScores {
                cpu_single_core: Some(1200),
                cpu_multi_core: Some(8500),
                gpu_score: None,
                memory_score: Some(8500),
                overall_tier: PerformanceTier::Mid,
                tier: PerformanceTier::Mid,
            },
        }
    }

    fn create_test_network_conditions() -> NetworkConditions {
        NetworkConditions {
            bandwidth_mbps: 25.0,
            latency_ms: 40.0,
            packet_loss_percent: 0.5,
            jitter_ms: 5.0,
            stability_score: 0.8,
            connection_type: NetworkConnectionType::WiFi,
            signal_strength_dbm: Some(-50),
            available_data_mb: Some(1000),
            quality_assessment: NetworkQuality::Good,
            timestamp: std::time::Instant::now(),
        }
    }

    #[test]
    fn test_network_efficiency_calculation() {
        let conditions = create_test_network_conditions();
        let efficiency = NetworkAdaptationUtils::calculate_network_efficiency(&conditions);

        assert!(efficiency > 0.0);
        assert!(efficiency <= 100.0);
    }

    #[test]
    fn test_optimized_config_creation() {
        let device_info = create_test_device_info();
        let conditions = create_test_network_conditions();

        let config = NetworkAdaptationUtils::create_optimized_config(&device_info, &conditions);

        // Should have reasonable sync frequency
        assert!(config.sync_frequency.base_frequency_minutes > 0);
        assert!(config.sync_frequency.base_frequency_minutes < 1000);
    }

    #[test]
    fn test_data_usage_estimation() {
        let task = FederatedTask {
            task_id: "test_task".to_string(),
            task_type: FederatedTaskType::ModelDownload,
            estimated_size_mb: 10,
            priority: TaskPriority::High,
            network_requirements: Default::default(),
            scheduled_time: std::time::Instant::now(),
            deadline: std::time::Instant::now(),
            retry_count: 0,
            status: TaskStatus::Pending,
        };

        let usage = NetworkAdaptationUtils::estimate_data_usage(&task, 0.8);
        assert!(usage > 0);
    }

    #[test]
    fn test_stats_recording() {
        let mut stats = NetworkAdaptationStats::new();
        let task = FederatedTask {
            task_id: "test_task".to_string(),
            task_type: FederatedTaskType::GradientUpload,
            estimated_size_mb: 5,
            priority: TaskPriority::Normal,
            network_requirements: Default::default(),
            scheduled_time: std::time::Instant::now(),
            deadline: std::time::Instant::now(),
            retry_count: 0,
            status: TaskStatus::Pending,
        };

        stats.record_task_scheduled(&task);
        assert_eq!(stats.total_tasks_scheduled, 1);

        stats.record_task_completed(&task, 5000, 1024000, NetworkConnectionType::WiFi);
        assert_eq!(stats.tasks_completed, 1);
        assert!(stats.avg_completion_time_ms > 0.0);
    }

    #[test]
    fn test_network_health_analysis() {
        let conditions = create_test_network_conditions();
        let health = NetworkAdaptationUtils::analyze_network_health(&conditions);

        assert!(health.overall_health_score > 0.0);
        assert!(health.overall_health_score <= 100.0);
        assert!(!health.recommendations.is_empty());
    }
}
