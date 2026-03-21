//! # MultiModelConfig - Trait Implementations
//!
//! This module contains trait implementations for `MultiModelConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::Duration;

use super::types::{
    ABTestingConfig, AlertingThresholds, EnsembleConfig, EnsembleOptimizationConfig,
    FallbackConfig, FallbackTrigger, ModelCascadingConfig, ModelRoutingConfig,
    ModelSelectionCriteria, MonitoredMetric, MultiModelConfig, PerformanceMonitoringConfig,
    QualityAssessmentConfig, QualityThresholds, ResourceBudget, ResourceConstraints,
    RoutingStrategy, StatisticalThresholds, TrafficSplit, TrafficSplittingConfig, VotingStrategy,
};

impl Default for MultiModelConfig {
    fn default() -> Self {
        Self {
            routing: ModelRoutingConfig {
                default_strategy: RoutingStrategy::RoundRobin,
                route_strategies: HashMap::new(),
                selection_criteria: ModelSelectionCriteria {
                    preferred_characteristics: Vec::new(),
                    quality_thresholds: QualityThresholds {
                        min_accuracy: None,
                        max_latency: None,
                        max_error_rate: None,
                        min_throughput: None,
                    },
                    resource_constraints: ResourceConstraints {
                        max_memory_usage: None,
                        max_gpu_memory: None,
                        max_cpu_usage: None,
                        required_gpu_count: None,
                    },
                },
                fallback: FallbackConfig {
                    fallback_model: "default".to_string(),
                    enabled: true,
                    triggers: vec![
                        FallbackTrigger::ModelUnavailable,
                        FallbackTrigger::HighErrorRate(0.1),
                    ],
                },
            },
            ensemble: EnsembleConfig {
                enabled: false,
                methods: Vec::new(),
                voting_strategy: VotingStrategy::SimpleMajority,
                quality_assessment: QualityAssessmentConfig {
                    enabled: false,
                    methods: Vec::new(),
                    thresholds: QualityThresholds {
                        min_accuracy: None,
                        max_latency: None,
                        max_error_rate: None,
                        min_throughput: None,
                    },
                },
                optimization: EnsembleOptimizationConfig {
                    enabled: false,
                    strategies: Vec::new(),
                    resource_budget: ResourceBudget {
                        max_latency: Duration::from_secs(10),
                        max_memory: 1024 * 1024 * 1024,
                        max_compute_cost: 1.0,
                    },
                },
            },
            ab_testing: ABTestingConfig {
                enabled: false,
                experiments: Vec::new(),
                significance_thresholds: StatisticalThresholds {
                    p_value: 0.05,
                    confidence_level: 0.95,
                    minimum_sample_size: 1000,
                    minimum_effect_size: 0.1,
                },
            },
            traffic_splitting: TrafficSplittingConfig {
                enabled: false,
                split_rules: Vec::new(),
                default_split: TrafficSplit {
                    splits: HashMap::from([("default".to_string(), 100.0)]),
                    sticky_sessions: false,
                },
            },
            model_cascading: ModelCascadingConfig {
                enabled: false,
                cascade_chains: Vec::new(),
                exit_strategies: Vec::new(),
            },
            performance_monitoring: PerformanceMonitoringConfig {
                enabled: true,
                collection_interval: Duration::from_secs(60),
                monitored_metrics: vec![
                    MonitoredMetric::ModelLatency,
                    MonitoredMetric::ErrorRates,
                    MonitoredMetric::ResourceUtilization,
                ],
                alerting_thresholds: AlertingThresholds {
                    latency_threshold: Duration::from_secs(5),
                    error_rate_threshold: 0.05,
                    accuracy_threshold: 0.9,
                    resource_threshold: 0.8,
                },
            },
        }
    }
}
