//! # UtilizationTrackingConfig - Trait Implementations
//!
//! This module contains trait implementations for `UtilizationTrackingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{
    CpuMonitorConfig, GpuMonitorConfig, IoMonitorConfig, MemoryMonitorConfig, NetworkMonitorConfig,
    UtilizationTrackingConfig,
};

impl Default for UtilizationTrackingConfig {
    fn default() -> Self {
        Self {
            sample_interval: Duration::from_secs(1),
            detailed_monitoring: true,
            history_retention: Duration::from_secs(86400),
            max_history_size: 86400,
            enable_trend_analysis: true,
            enable_alerting: true,
            enable_compression: true,
            monitoring_priority: 0,
            cpu_config: CpuMonitorConfig::default(),
            memory_config: MemoryMonitorConfig::default(),
            io_config: IoMonitorConfig::default(),
            network_config: NetworkMonitorConfig::default(),
            gpu_config: GpuMonitorConfig::default(),
        }
    }
}
