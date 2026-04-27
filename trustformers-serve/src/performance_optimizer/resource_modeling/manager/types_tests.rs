//! Tests for resource modeling manager types

use super::*;
use chrono::Utc;
use std::collections::HashMap;
use std::time::Duration;

/// Simple LCG for deterministic pseudo-random values
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() % 10000) as f32 / 10000.0
    }
}

#[test]
fn test_alert_level_ordering() {
    assert!(AlertLevel::Info < AlertLevel::Warning);
    assert!(AlertLevel::Warning < AlertLevel::Error);
    assert!(AlertLevel::Error < AlertLevel::Critical);
}

#[test]
fn test_alert_level_equality() {
    let a = AlertLevel::Info;
    let b = AlertLevel::Info;
    assert_eq!(a, b);
}

#[test]
fn test_difficulty_level_variants() {
    let levels = [
        DifficultyLevel::Easy,
        DifficultyLevel::Medium,
        DifficultyLevel::Hard,
        DifficultyLevel::Expert,
    ];
    for (i, level) in levels.iter().enumerate() {
        for (j, other) in levels.iter().enumerate() {
            if i == j {
                assert_eq!(level, other);
            } else {
                assert_ne!(level, other);
            }
        }
    }
}

#[test]
fn test_analysis_priority_ordering() {
    assert!(AnalysisPriority::Background < AnalysisPriority::Low);
    assert!(AnalysisPriority::Low < AnalysisPriority::Normal);
    assert!(AnalysisPriority::Normal < AnalysisPriority::High);
    assert!(AnalysisPriority::High < AnalysisPriority::Critical);
}

#[test]
fn test_analysis_quality_variants() {
    let qualities = [
        AnalysisQuality::Fast,
        AnalysisQuality::Balanced,
        AnalysisQuality::Comprehensive,
        AnalysisQuality::Research,
    ];
    for q in &qualities {
        let cloned = *q;
        assert_eq!(*q, cloned);
    }
}

#[test]
fn test_trend_direction_variants() {
    let dirs = [
        TrendDirection::Improving,
        TrendDirection::Stable,
        TrendDirection::Degrading,
        TrendDirection::Declining,
    ];
    assert_eq!(dirs.len(), 4);
    assert_ne!(TrendDirection::Improving, TrendDirection::Declining);
}

#[test]
fn test_task_execution_status_variants() {
    let statuses = [
        TaskExecutionStatus::Pending,
        TaskExecutionStatus::Running,
        TaskExecutionStatus::Completed,
        TaskExecutionStatus::Failed,
        TaskExecutionStatus::Cancelled,
        TaskExecutionStatus::TimedOut,
        TaskExecutionStatus::Retried,
    ];
    assert_eq!(statuses.len(), 7);
}

#[test]
fn test_component_status_variants() {
    let statuses = [
        ComponentStatus::Healthy,
        ComponentStatus::Warning,
        ComponentStatus::Degraded,
        ComponentStatus::Failed,
        ComponentStatus::Uninitialized,
    ];
    assert_eq!(statuses.len(), 5);
    assert_ne!(ComponentStatus::Healthy, ComponentStatus::Failed);
}

#[test]
fn test_workflow_execution_status_variants() {
    let statuses = [
        WorkflowExecutionStatus::Pending,
        WorkflowExecutionStatus::Running,
        WorkflowExecutionStatus::Completed,
        WorkflowExecutionStatus::Failed,
        WorkflowExecutionStatus::Cancelled,
    ];
    assert_eq!(statuses.len(), 5);
}

#[test]
fn test_cost_impact_variants() {
    assert_ne!(CostImpact::Low, CostImpact::Critical);
    assert_eq!(CostImpact::Medium, CostImpact::Medium);
}

#[test]
fn test_impact_level_variants() {
    assert_ne!(ImpactLevel::Low, ImpactLevel::High);
    assert_eq!(ImpactLevel::Critical, ImpactLevel::Critical);
}

#[test]
fn test_analysis_task_type_custom() {
    let custom = AnalysisTaskType::Custom("my_task".to_string());
    if let AnalysisTaskType::Custom(name) = &custom {
        assert_eq!(name, "my_task");
    } else {
        panic!("expected Custom variant");
    }
}

#[test]
fn test_analysis_task_type_hash() {
    let mut map = HashMap::new();
    map.insert(AnalysisTaskType::PerformanceProfiling, 1);
    map.insert(AnalysisTaskType::TemperatureMonitoring, 2);
    map.insert(AnalysisTaskType::Custom("x".into()), 3);
    assert_eq!(map.len(), 3);
}

#[test]
fn test_performance_recommendation_creation() {
    let rec = PerformanceRecommendation {
        title: "Optimize memory".to_string(),
        description: "Use memory pooling".to_string(),
        expected_improvement: 0.25,
        complexity: DifficultyLevel::Medium,
        required_resources: vec!["dev_time".to_string()],
        implementation_time: Duration::from_secs(3600),
    };
    assert!(rec.expected_improvement > 0.0);
    assert_eq!(rec.complexity, DifficultyLevel::Medium);
}

#[test]
fn test_task_resource_usage_creation() {
    let usage = TaskResourceUsage {
        cpu_usage: 45.5,
        memory_usage_mb: 1024,
        io_operations: 500,
        network_operations: 100,
    };
    assert!(usage.cpu_usage < 100.0);
    assert!(usage.memory_usage_mb > 0);
}

#[test]
fn test_resource_modeling_config_builder_detailed_detection() {
    let config = ResourceModelingConfig {
        detailed_detection: false,
        enable_profiling: true,
        enable_temperature_monitoring: false,
        enable_numa_analysis: false,
        update_interval: Duration::from_secs(30),
        profiling_samples: 100,
        temperature_threshold: 80.0,
        cache_profiling_results: true,
        max_concurrent_tasks: 4,
        task_timeout: Duration::from_secs(300),
        enable_predictive_analysis: false,
        cache_size_limit_mb: 256,
        enable_error_recovery: true,
        reporting_interval: Duration::from_secs(60),
        analysis_quality: AnalysisQuality::Balanced,
    };
    let updated = config.with_detailed_detection(true);
    assert!(updated.detailed_detection);
}

#[test]
fn test_resource_modeling_config_builder_profiling() {
    let config = ResourceModelingConfig {
        detailed_detection: false,
        enable_profiling: false,
        enable_temperature_monitoring: false,
        enable_numa_analysis: false,
        update_interval: Duration::from_secs(30),
        profiling_samples: 100,
        temperature_threshold: 80.0,
        cache_profiling_results: false,
        max_concurrent_tasks: 4,
        task_timeout: Duration::from_secs(300),
        enable_predictive_analysis: false,
        cache_size_limit_mb: 256,
        enable_error_recovery: true,
        reporting_interval: Duration::from_secs(60),
        analysis_quality: AnalysisQuality::Fast,
    };
    let updated = config
        .with_profiling_enabled(true)
        .with_profiling_samples(200)
        .with_cache_profiling_results(true);
    assert!(updated.enable_profiling);
    assert_eq!(updated.profiling_samples, 200);
    assert!(updated.cache_profiling_results);
}

#[test]
fn test_resource_modeling_config_builder_temperature() {
    let config = ResourceModelingConfig {
        detailed_detection: false,
        enable_profiling: false,
        enable_temperature_monitoring: false,
        enable_numa_analysis: false,
        update_interval: Duration::from_secs(30),
        profiling_samples: 100,
        temperature_threshold: 80.0,
        cache_profiling_results: false,
        max_concurrent_tasks: 4,
        task_timeout: Duration::from_secs(300),
        enable_predictive_analysis: false,
        cache_size_limit_mb: 256,
        enable_error_recovery: true,
        reporting_interval: Duration::from_secs(60),
        analysis_quality: AnalysisQuality::Balanced,
    };
    let updated = config.with_temperature_monitoring(true);
    assert!(updated.enable_temperature_monitoring);
}

#[test]
fn test_resource_modeling_config_builder_numa() {
    let config = ResourceModelingConfig {
        detailed_detection: false,
        enable_profiling: false,
        enable_temperature_monitoring: false,
        enable_numa_analysis: false,
        update_interval: Duration::from_secs(30),
        profiling_samples: 100,
        temperature_threshold: 80.0,
        cache_profiling_results: false,
        max_concurrent_tasks: 4,
        task_timeout: Duration::from_secs(300),
        enable_predictive_analysis: false,
        cache_size_limit_mb: 256,
        enable_error_recovery: true,
        reporting_interval: Duration::from_secs(60),
        analysis_quality: AnalysisQuality::Balanced,
    };
    let updated = config.with_numa_analysis(true);
    assert!(updated.enable_numa_analysis);
}

#[test]
fn test_resource_modeling_config_builder_update_interval() {
    let config = ResourceModelingConfig {
        detailed_detection: false,
        enable_profiling: false,
        enable_temperature_monitoring: false,
        enable_numa_analysis: false,
        update_interval: Duration::from_secs(30),
        profiling_samples: 100,
        temperature_threshold: 80.0,
        cache_profiling_results: false,
        max_concurrent_tasks: 4,
        task_timeout: Duration::from_secs(300),
        enable_predictive_analysis: false,
        cache_size_limit_mb: 256,
        enable_error_recovery: true,
        reporting_interval: Duration::from_secs(60),
        analysis_quality: AnalysisQuality::Balanced,
    };
    let updated = config.with_update_interval(Duration::from_secs(10));
    assert_eq!(updated.update_interval, Duration::from_secs(10));
}

#[test]
fn test_retry_policy_creation() {
    let policy = RetryPolicy {
        max_retries: 3,
        retry_delay: Duration::from_millis(500),
        exponential_backoff: true,
    };
    assert_eq!(policy.max_retries, 3);
    assert!(policy.exponential_backoff);
}

#[test]
fn test_resource_requirement_bounds() {
    let req = ResourceRequirement {
        minimum: 10.0,
        recommended: 50.0,
        maximum: 100.0,
        growth_rate: 0.05,
    };
    assert!(req.minimum <= req.recommended);
    assert!(req.recommended <= req.maximum);
    assert!(req.growth_rate > 0.0);
}

#[test]
fn test_executive_summary_creation() {
    let summary = ExecutiveSummary {
        key_findings: vec!["finding1".to_string(), "finding2".to_string()],
        health_score: 0.95,
        top_recommendations: vec!["rec1".to_string()],
        performance_highlights: vec!["highlight1".to_string()],
    };
    assert_eq!(summary.key_findings.len(), 2);
    assert!(summary.health_score <= 1.0);
}

#[test]
fn test_system_alert_creation() {
    let alert = SystemAlert {
        id: "alert_001".to_string(),
        level: AlertLevel::Warning,
        message: "High CPU usage".to_string(),
        component: "cpu_monitor".to_string(),
        timestamp: Utc::now(),
        acknowledged: false,
    };
    assert!(!alert.acknowledged);
    assert_eq!(alert.level, AlertLevel::Warning);
}

#[test]
fn test_health_trend_creation() {
    let trend = HealthTrend {
        component: "memory".to_string(),
        direction: TrendDirection::Stable,
        magnitude: 0.5,
        period: Duration::from_secs(3600),
    };
    assert!(trend.magnitude >= 0.0 && trend.magnitude <= 1.0);
}

#[test]
fn test_performance_trend_prediction_creation() {
    let pred = PerformanceTrendPrediction {
        cpu_trend: TrendDirection::Stable,
        memory_trend: TrendDirection::Improving,
        io_trend: TrendDirection::Degrading,
        network_trend: TrendDirection::Declining,
        system_trend: TrendDirection::Stable,
    };
    assert_eq!(pred.cpu_trend, TrendDirection::Stable);
    assert_ne!(pred.io_trend, pred.memory_trend);
}

#[test]
fn test_analysis_task_creation_with_lcg() {
    let mut lcg = Lcg::new(42);
    let task = AnalysisTask {
        id: lcg.next_u64(),
        task_type: AnalysisTaskType::PerformanceProfiling,
        priority: AnalysisPriority::High,
        parameters: HashMap::new(),
        estimated_duration: Duration::from_secs(60),
        deadline: None,
        dependencies: vec![],
        created_at: Utc::now(),
        retry_count: 0,
        max_retries: 3,
    };
    assert!(task.id > 0);
    assert_eq!(task.priority, AnalysisPriority::High);
}

#[test]
fn test_workflow_step_creation() {
    let step = WorkflowStep {
        name: "profiling_step".to_string(),
        task_type: AnalysisTaskType::PerformanceProfiling,
        priority: AnalysisPriority::Normal,
        parameters: HashMap::new(),
        dependencies: vec!["init".to_string()],
        timeout: Duration::from_secs(120),
        required: true,
    };
    assert!(step.required);
    assert_eq!(step.dependencies.len(), 1);
}

#[test]
fn test_component_health_creation() {
    let health = ComponentHealth {
        name: "profiler".to_string(),
        status: ComponentStatus::Healthy,
        last_check: Utc::now(),
        error_count: 0,
        performance_metrics: ComponentPerformanceMetrics {
            avg_response_time: Duration::from_millis(50),
            success_rate: 0.999,
            resource_usage: TaskResourceUsage {
                cpu_usage: 0.45,
                memory_usage_mb: 512,
                io_operations: 100,
                network_operations: 50,
            },
            throughput: 1000.0,
        },
    };
    assert_eq!(health.status, ComponentStatus::Healthy);
    assert_eq!(health.error_count, 0);
}
