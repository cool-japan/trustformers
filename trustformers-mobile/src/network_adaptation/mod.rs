//! Network Condition Adaptation for Federated Learning
//!
//! This module provides comprehensive network adaptation capabilities for mobile
//! federated learning scenarios, including intelligent scheduling, bandwidth optimization,
//! and adaptive communication strategies based on real-time network conditions.
//!
//! ## Architecture
//!
//! The network adaptation system is organized into several focused modules:
//! - [`config`] - Configuration management for network adaptation parameters
//! - [`types`] - Core types, enums, and data structures for network adaptation
//! - [`monitoring`] - Network monitoring and condition assessment
//! - [`scheduling`] - Federated task scheduling and coordination
//! - [`bandwidth`] - Bandwidth optimization and traffic shaping
//! - [`prediction`] - Network performance prediction and modeling
//! - [`sync`] - Model synchronization coordination
//! - [`stats`] - Statistics tracking and metrics collection
//! - [`manager`] - Main NetworkAdaptationManager orchestrating all components

pub mod bandwidth;
pub mod config;
pub mod manager;
pub mod monitoring;
pub mod prediction;
pub mod scheduling;
pub mod stats;
pub mod sync;
pub mod types;

// Re-export main types for backward compatibility
pub use config::NetworkAdaptationConfig as ConfigNetworkAdaptationConfig;
pub use manager::NetworkAdaptationManager;
pub use types::{
    CellularConfig, CellularStrategy, CommunicationStrategy, DataUsageAwareness, DataUsageLimits,
    FailureRecoveryConfig, GradientCompressionAlgorithm, NetworkAdaptationConfig,
    NetworkCompressionConfig, NetworkPredictionConfig, NetworkQualityThresholds,
    NetworkQuantizationConfig, PoorNetworkStrategy, RetryConfig, SyncFrequencyConfig,
    TimeBasedScheduling, WiFiStrategy,
};

// Re-export component types for easy access
pub use bandwidth::{BandwidthOptimizer, DataUsageTracker, TrafficShaper};
pub use monitoring::NetworkMonitor;
pub use prediction::NetworkPredictor;
pub use scheduling::FederatedScheduler;
pub use stats::NetworkAdaptationStats;
pub use sync::ModelSyncCoordinator;
pub use types::SyncStrategy;
pub use types::{FederatedTaskType, SchedulingDecision};
pub use types::{NetworkConditions, NetworkQuality};
