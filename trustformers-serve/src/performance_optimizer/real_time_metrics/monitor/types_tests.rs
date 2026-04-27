//! Tests for real-time metrics monitor types

use super::types::*;
use chrono::Utc;
use std::collections::HashMap;

/// Simple LCG for deterministic pseudo-random values
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() % 10000) as f32 / 10000.0
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() % 100000) as f64 / 100000.0
    }
}

#[test]
fn test_variability_bounds_creation() {
    let bounds = VariabilityBounds {
        throughput_lower: 100.0,
        throughput_upper: 500.0,
        latency_lower: 0.001,
        latency_upper: 0.1,
        cpu_lower: 0.1,
        cpu_upper: 0.9,
        memory_lower: 0.2,
        memory_upper: 0.8,
        efficiency_lower: 0.5,
        efficiency_upper: 0.95,
        network_lower: 1000.0,
        network_upper: 10000.0,
        io_lower: 50.0,
        io_upper: 5000.0,
        response_time_lower: 0.001,
        response_time_upper: 1.0,
        error_rate_lower: 0.0,
        error_rate_upper: 0.05,
    };
    assert!(bounds.throughput_lower < bounds.throughput_upper);
    assert!(bounds.cpu_lower < bounds.cpu_upper);
    assert!(bounds.memory_lower < bounds.memory_upper);
}

#[test]
fn test_variability_bounds_ranges_valid() {
    let bounds = VariabilityBounds {
        throughput_lower: 50.0,
        throughput_upper: 200.0,
        latency_lower: 0.0,
        latency_upper: 0.5,
        cpu_lower: 0.0,
        cpu_upper: 1.0,
        memory_lower: 0.0,
        memory_upper: 1.0,
        efficiency_lower: 0.0,
        efficiency_upper: 1.0,
        network_lower: 0.0,
        network_upper: 100000.0,
        io_lower: 0.0,
        io_upper: 10000.0,
        response_time_lower: 0.0,
        response_time_upper: 10.0,
        error_rate_lower: 0.0,
        error_rate_upper: 1.0,
    };
    assert!(bounds.error_rate_lower <= bounds.error_rate_upper);
    assert!(bounds.io_lower <= bounds.io_upper);
}

#[test]
fn test_load_balancing_algorithm_equality() {
    assert_eq!(LoadBalancingAlgorithm::RoundRobin, LoadBalancingAlgorithm::RoundRobin);
    assert_ne!(LoadBalancingAlgorithm::RoundRobin, LoadBalancingAlgorithm::LeastConnections);
}

#[test]
fn test_load_balancing_algorithm_variants() {
    let algos = [
        LoadBalancingAlgorithm::RoundRobin,
        LoadBalancingAlgorithm::LeastConnections,
        LoadBalancingAlgorithm::WeightedRoundRobin,
        LoadBalancingAlgorithm::LoadBased,
        LoadBalancingAlgorithm::PerformanceBased,
    ];
    assert_eq!(algos.len(), 5);
}

#[test]
fn test_baseline_validation_status_variants() {
    let statuses = [
        BaselineValidationStatus::Pending,
        BaselineValidationStatus::Valid,
        BaselineValidationStatus::Validating,
        BaselineValidationStatus::Invalid,
        BaselineValidationStatus::Invalid,
    ];
    assert_eq!(statuses.len(), 5);
    assert_ne!(BaselineValidationStatus::Valid, BaselineValidationStatus::Invalid);
}

#[test]
fn test_threshold_anomaly_detector_new() {
    let detector = ThresholdAnomalyDetector::new();
    assert!((detector.throughput_threshold - 0.3).abs() < f32::EPSILON);
    assert!((detector.latency_threshold - 0.5).abs() < f32::EPSILON);
    assert!((detector.cpu_threshold - 0.2).abs() < f32::EPSILON);
    assert!((detector.memory_threshold - 0.25).abs() < f32::EPSILON);
}

#[test]
fn test_threshold_detector_exceeds_threshold_normal() {
    let detector = ThresholdAnomalyDetector::new();
    // Current close to baseline: should not exceed
    assert!(!detector.exceeds_threshold(100.0, 100.0, 0.3));
}

#[test]
fn test_threshold_detector_exceeds_threshold_high_deviation() {
    let detector = ThresholdAnomalyDetector::new();
    // 50% deviation should exceed 0.3 threshold
    assert!(detector.exceeds_threshold(150.0, 100.0, 0.3));
}

#[test]
fn test_threshold_detector_zero_baseline() {
    let detector = ThresholdAnomalyDetector::new();
    // Zero baseline should return false
    assert!(!detector.exceeds_threshold(100.0, 0.0, 0.3));
}

#[test]
fn test_pattern_anomaly_detector_new() {
    let detector = PatternAnomalyDetector::new(10);
    assert!(detector.patterns.is_empty());
    assert!((detector.similarity_threshold - 0.7).abs() < f32::EPSILON);
}

#[test]
fn test_pattern_similarity_identical_patterns() {
    let detector = PatternAnomalyDetector::new(10);
    let pattern = PatternSignature {
        throughput_trend: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        latency_trend: vec![0.1, 0.2, 0.3, 0.4, 0.5],
        resource_trend: vec![10.0, 20.0, 30.0, 40.0, 50.0],
        timestamp: Utc::now(),
        quality: 0.9,
    };
    let sim = detector.calculate_similarity(&pattern, &pattern);
    // Identical patterns should have high similarity
    assert!(sim > 0.99, "expected high similarity for identical patterns, got {}", sim);
}

#[test]
fn test_pattern_similarity_different_patterns() {
    let detector = PatternAnomalyDetector::new(10);
    let p1 = PatternSignature {
        throughput_trend: vec![1.0, 2.0, 3.0],
        latency_trend: vec![0.1, 0.2, 0.3],
        resource_trend: vec![10.0, 20.0, 30.0],
        timestamp: Utc::now(),
        quality: 0.9,
    };
    let p2 = PatternSignature {
        throughput_trend: vec![3.0, 2.0, 1.0],
        latency_trend: vec![0.3, 0.2, 0.1],
        resource_trend: vec![30.0, 20.0, 10.0],
        timestamp: Utc::now(),
        quality: 0.5,
    };
    let sim = detector.calculate_similarity(&p1, &p2);
    // Inverse patterns should still produce a result (absolute correlation)
    assert!(sim >= 0.0 && sim <= 1.0);
}

#[test]
fn test_pattern_similarity_empty_patterns() {
    let detector = PatternAnomalyDetector::new(10);
    let p1 = PatternSignature {
        throughput_trend: vec![],
        latency_trend: vec![],
        resource_trend: vec![],
        timestamp: Utc::now(),
        quality: 0.0,
    };
    let sim = detector.calculate_similarity(&p1, &p1);
    assert!((sim - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_pattern_similarity_different_lengths() {
    let detector = PatternAnomalyDetector::new(10);
    let p1 = PatternSignature {
        throughput_trend: vec![1.0, 2.0],
        latency_trend: vec![0.1],
        resource_trend: vec![10.0, 20.0, 30.0],
        timestamp: Utc::now(),
        quality: 0.5,
    };
    let p2 = PatternSignature {
        throughput_trend: vec![1.0],
        latency_trend: vec![0.1, 0.2],
        resource_trend: vec![10.0],
        timestamp: Utc::now(),
        quality: 0.5,
    };
    let sim = detector.calculate_similarity(&p1, &p2);
    assert!((sim - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_anomaly_event_to_event_data() {
    let event = AnomalyEvent {
        timestamp: Utc::now(),
        anomaly_type: "latency_spike".to_string(),
        severity: SeverityLevel::High,
        description: "High latency detected".to_string(),
        affected_metrics: vec!["p99_latency".to_string(), "avg_latency".to_string()],
        score: 0.85,
        confidence: 0.92,
        expected_value: 10.0,
        actual_value: 50.0,
        deviation: 40.0,
        detection_algorithm: "threshold".to_string(),
        context: HashMap::new(),
        recommendations: vec!["Scale up".to_string()],
    };
    let data = event.to_event_data();
    assert_eq!(data.get("type"), Some(&"latency_spike".to_string()));
    assert!(data.contains_key("score"));
    assert!(data.contains_key("confidence"));
    assert!(data.contains_key("affected_metrics"));
}

#[test]
fn test_anomaly_event_requires_immediate_attention_high() {
    let event = AnomalyEvent {
        timestamp: Utc::now(),
        anomaly_type: "critical_failure".to_string(),
        severity: SeverityLevel::Critical,
        description: "Critical failure".to_string(),
        affected_metrics: vec![],
        score: 1.0,
        confidence: 1.0,
        expected_value: 0.0,
        actual_value: 100.0,
        deviation: 100.0,
        detection_algorithm: "threshold".to_string(),
        context: HashMap::new(),
        recommendations: vec![],
    };
    assert!(event.requires_immediate_attention());
}

#[test]
fn test_anomaly_event_impact_score() {
    let event = AnomalyEvent {
        timestamp: Utc::now(),
        anomaly_type: "test".to_string(),
        severity: SeverityLevel::Low,
        description: "test".to_string(),
        affected_metrics: vec![],
        score: 0.5,
        confidence: 0.8,
        expected_value: 0.0,
        actual_value: 0.0,
        deviation: 0.0,
        detection_algorithm: "test".to_string(),
        context: HashMap::new(),
        recommendations: vec![],
    };
    let impact = event.impact_score();
    assert!((impact - 0.4).abs() < 0.001);
}

#[test]
fn test_thread_pool_config_creation() {
    let config = ThreadPoolConfig {
        min_threads: 2,
        max_threads: 16,
        scaling_threshold: 0.8,
        scale_up_delay: Duration::from_secs(5),
        scale_down_delay: Duration::from_secs(30),
        load_balancing: LoadBalancingAlgorithm::RoundRobin,
    };
    assert!(config.min_threads < config.max_threads);
    assert!(config.scaling_threshold > 0.0 && config.scaling_threshold <= 1.0);
}

#[test]
fn test_thread_resource_limits_creation() {
    let limits = ThreadResourceLimits {
        max_memory: 1024 * 1024 * 512,
        max_cpu: 0.5,
        max_iops: 10000,
        max_bandwidth: 100 * 1024 * 1024,
    };
    assert!(limits.max_cpu <= 1.0);
    assert!(limits.max_memory > 0);
}

#[test]
fn test_monitoring_status_creation() {
    let status = MonitoringStatus {
        active: true,
        thread_count: 4,
        processing_rate: 1000.0,
        health_score: 0.95,
        last_activity: Utc::now(),
        memory_usage: 1024 * 1024,
        cpu_utilization: 0.3,
        active_anomalies: 0,
        baseline_status: BaselineValidationStatus::Valid,
        error_rate: 0.001,
        collection_stats: CollectionStatsSummary {
            total_data_points: 0,
            collection_rate: 100.0,
            success_rate: 0.99,
            avg_collection_time: 10.0, data_quality: 0.95,
        },
    };
    assert!(status.active);
    assert!(status.health_score > 0.0);
}

#[test]
fn test_trend_info_creation() {
    let trend = TrendInfo {
        direction: crate::performance_optimizer::real_time_metrics::types::TrendDirection::Increasing,
        strength: TrendStrength::Strong,
        slope: 0.05,
        r_squared: 0.85,
        confidence: 0.92,
        significance: true,
    };
    assert!(trend.significance);
    assert!(trend.r_squared > 0.0 && trend.r_squared <= 1.0);
}

#[test]
fn test_pattern_signature_quality_bounds() {
    let mut lcg = Lcg::new(42);
    for _ in 0..10 {
        let quality = lcg.next_f32();
        let sig = PatternSignature {
            throughput_trend: vec![lcg.next_f64()],
            latency_trend: vec![lcg.next_f64()],
            resource_trend: vec![lcg.next_f64()],
            timestamp: Utc::now(),
            quality,
        };
        assert!(sig.quality >= 0.0 && sig.quality <= 1.0);
    }
}

#[test]
fn test_thread_pool_statistics_default() {
    let stats = ThreadPoolStatistics::default();
    assert_eq!(stats.current_threads, 0);
    assert_eq!(stats.scaling_events, 0);
    assert_eq!(stats.load_balance_events, 0);
}

#[test]
fn test_trend_analysis_result_creation() {
    let make_trend = |dir: crate::performance_optimizer::real_time_metrics::types::TrendDirection| TrendInfo {
        direction: dir,
        strength: TrendStrength::Moderate,
        slope: 0.01,
        r_squared: 0.7,
        confidence: 0.8,
        significance: true,
    };
    let result = TrendAnalysisResult {
        throughput_trend: make_trend(crate::performance_optimizer::real_time_metrics::types::TrendDirection::Increasing),
        latency_trend: make_trend(crate::performance_optimizer::real_time_metrics::types::TrendDirection::Stable),
        cpu_trend: make_trend(crate::performance_optimizer::real_time_metrics::types::TrendDirection::Decreasing),
        memory_trend: make_trend(crate::performance_optimizer::real_time_metrics::types::TrendDirection::Stable),
        overall_trend: make_trend(crate::performance_optimizer::real_time_metrics::types::TrendDirection::Stable),
        analysis_confidence: 0.85,
        recommendation: "Scale CPU".to_string(),
    };
    assert!(result.analysis_confidence > 0.0);
}

#[test]
fn test_thread_configuration_creation() {
    let config = ThreadConfiguration {
        collection_interval: Duration::from_millis(100),
        timeout: Duration::from_secs(30),
        buffer_size: 1024,
        priority: ThreadPriority::Normal,
        resource_limits: ThreadResourceLimits {
            max_memory: 256 * 1024 * 1024,
            max_cpu: 0.25,
            max_iops: 5000,
            max_bandwidth: 50 * 1024 * 1024,
        },
        retry_config: RetryConfiguration {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0, jitter_enabled: false,
        },
    };
    assert_eq!(config.buffer_size, 1024);
    assert_eq!(config.retry_config.max_attempts, 3);
}
