#[cfg(test)]
mod tests {
    use crate::mobile_performance_profiler::profiler_split::types::*;
    use crate::mobile_performance_profiler::types::*;
    use crate::device_info::ThermalState;
    use std::collections::HashMap;
    use std::time::Duration;

    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }
        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1u64 << 53) as f32
        }
    }

    // Test 1: CacheType variants
    #[test]
    fn test_cache_type_variants() {
        let types = vec![
            CacheType::Model,
            CacheType::Tensor,
            CacheType::Computation,
            CacheType::Network,
            CacheType::General,
        ];
        assert_eq!(types.len(), 5);
        // Verify debug output works
        for cache_type in &types {
            let debug_str = format!("{:?}", cache_type);
            assert!(!debug_str.is_empty());
        }
    }

    // Test 2: BottleneckRule construction
    #[test]
    fn test_bottleneck_rule_construction() {
        let rule = BottleneckRule {
            id: "rule_001".to_string(),
            name: "High Memory Usage".to_string(),
            condition: BottleneckCondition::MemoryUsageHigh {
                threshold_percent: 85.0,
                duration_ms: 5000,
            },
            severity: BottleneckSeverity::High,
            suggestion: "Reduce memory usage".to_string(),
            confidence: 0.9,
            enabled: true,
        };
        assert_eq!(rule.id, "rule_001");
        assert!(rule.enabled);
        assert!((rule.confidence - 0.9).abs() < f32::EPSILON);
    }

    // Test 3: BottleneckCondition variants
    #[test]
    fn test_bottleneck_condition_variants() {
        let conditions = vec![
            BottleneckCondition::MemoryUsageHigh {
                threshold_percent: 80.0,
                duration_ms: 1000,
            },
            BottleneckCondition::CPUUsageHigh {
                threshold_percent: 90.0,
                duration_ms: 2000,
            },
            BottleneckCondition::GPUUsageHigh {
                threshold_percent: 95.0,
                duration_ms: 500,
            },
            BottleneckCondition::LatencyHigh {
                threshold_ms: 100.0,
                sample_count: 10,
            },
            BottleneckCondition::MemoryPressure { pressure_level: 3 },
            BottleneckCondition::Custom {
                name: "custom_rule".to_string(),
                evaluator: "my_evaluator".to_string(),
            },
        ];
        assert_eq!(conditions.len(), 6);
    }

    // Test 4: MemoryUsagePattern variants
    #[test]
    fn test_memory_usage_pattern_variants() {
        let patterns = vec![
            MemoryUsagePattern::SteadyHigh,
            MemoryUsagePattern::RapidGrowth,
            MemoryUsagePattern::MemoryLeaks,
            MemoryUsagePattern::Fragmentation,
            MemoryUsagePattern::LargeAllocations,
        ];
        for pattern in &patterns {
            let debug_str = format!("{:?}", pattern);
            assert!(!debug_str.is_empty());
        }
    }

    // Test 5: AnalysisResult construction
    #[test]
    fn test_analysis_result_construction() {
        let mut results = HashMap::new();
        results.insert("memory_usage".to_string(), 75.5f32);
        results.insert("cpu_usage".to_string(), 45.2f32);

        let analysis = AnalysisResult {
            analysis_id: "analysis_001".to_string(),
            timestamp: std::time::Instant::now(),
            analysis_type: "performance".to_string(),
            results,
            recommendations: vec!["Optimize memory".to_string()],
            confidence_score: 0.85,
        };
        assert_eq!(analysis.analysis_id, "analysis_001");
        assert_eq!(analysis.results.len(), 2);
        assert_eq!(analysis.recommendations.len(), 1);
        assert!((analysis.confidence_score - 0.85).abs() < f32::EPSILON);
    }

    // Test 6: OptimizationEvent construction
    #[test]
    fn test_optimization_event_construction() {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "auto".to_string());

        let event = OptimizationEvent {
            timestamp: std::time::Instant::now(),
            suggestion_id: "sugg_001".to_string(),
            event_type: "applied".to_string(),
            performance_impact: 0.15,
            implementation_status: "completed".to_string(),
            metadata,
        };
        assert_eq!(event.suggestion_id, "sugg_001");
        assert!((event.performance_impact - 0.15).abs() < f32::EPSILON);
        assert_eq!(event.metadata.len(), 1);
    }

    // Test 7: BottleneckDetectionStats default
    #[test]
    fn test_bottleneck_detection_stats_default() {
        let stats = BottleneckDetectionStats::default();
        assert_eq!(stats.total_detections, 0);
        assert_eq!(stats.true_positives, 0);
        assert_eq!(stats.false_positives, 0);
        assert!((stats.avg_detection_latency_ms - 0.0).abs() < f32::EPSILON);
        assert!((stats.accuracy_rate - 0.0).abs() < f32::EPSILON);
    }

    // Test 8: OptimizationCondition variants
    #[test]
    fn test_optimization_condition_variants() {
        let condition = OptimizationCondition::HighMemoryUsage {
            threshold_percent: 85.0,
            pattern: MemoryUsagePattern::SteadyHigh,
        };
        let debug_str = format!("{:?}", condition);
        assert!(debug_str.contains("HighMemoryUsage"));

        let condition_battery = OptimizationCondition::BatteryDrainHigh {
            threshold_mw: 1500.0,
            context: BatteryContext::Inference,
        };
        let debug_str2 = format!("{:?}", condition_battery);
        assert!(debug_str2.contains("BatteryDrainHigh"));
    }

    // Test 9: WorkloadType variants
    #[test]
    fn test_workload_type_variants() {
        let types = vec![
            WorkloadType::ComputerVision,
            WorkloadType::NLP,
            WorkloadType::Audio,
            WorkloadType::General,
        ];
        for workload in &types {
            let debug_str = format!("{:?}", workload);
            assert!(!debug_str.is_empty());
        }
    }

    // Test 10: CPUUsagePattern variants
    #[test]
    fn test_cpu_usage_pattern_variants() {
        let patterns = vec![
            CPUUsagePattern::SingleCoreHigh,
            CPUUsagePattern::PoorMultiCore,
            CPUUsagePattern::FrequentSwitching,
            CPUUsagePattern::ThermalLimited,
            CPUUsagePattern::InefficientAlgorithms,
        ];
        assert_eq!(patterns.len(), 5);
    }

    // Test 11: BatteryContext variants
    #[test]
    fn test_battery_context_variants() {
        let contexts = vec![
            BatteryContext::Inference,
            BatteryContext::ModelLoading,
            BatteryContext::Background,
            BatteryContext::Network,
            BatteryContext::General,
        ];
        assert_eq!(contexts.len(), 5);
    }

    // Test 12: MonitoringStats default
    #[test]
    fn test_monitoring_stats_default() {
        let stats = MonitoringStats::default();
        assert_eq!(stats.total_monitor_time, Duration::ZERO);
        assert_eq!(stats.alerts_generated, 0);
        assert_eq!(stats.critical_alerts, 0);
        assert!((stats.avg_alert_response_time_ms - 0.0).abs() < f32::EPSILON);
        assert!((stats.false_alarm_rate - 0.0).abs() < f32::EPSILON);
    }

    // Test 13: ExportManagerStats default
    #[test]
    fn test_export_manager_stats_default() {
        let stats = ExportManagerStats::default();
        assert_eq!(stats.exports_completed, 0);
        assert_eq!(stats.total_data_exported, 0);
        assert!((stats.avg_export_time_ms - 0.0).abs() < f32::EPSILON);
        assert!((stats.success_rate - 0.0).abs() < f32::EPSILON);
        assert!((stats.avg_compression_ratio - 0.0).abs() < f32::EPSILON);
    }

    // Test 14: ProfilingError construction
    #[test]
    fn test_profiling_error_construction() {
        let error = ProfilingError {
            error_id: "err_001".to_string(),
            timestamp: std::time::Instant::now(),
            error_type: "collection_failure".to_string(),
            message: "Failed to collect CPU metrics".to_string(),
            source: Some("metrics_collector".to_string()),
            severity: "high".to_string(),
        };
        assert_eq!(error.error_id, "err_001");
        assert!(error.source.is_some());
        if let Some(ref src) = error.source {
            assert_eq!(src, "metrics_collector");
        }
    }

    // Test 15: ExportRecord construction
    #[test]
    fn test_export_record_construction() {
        let record = ExportRecord {
            export_id: "exp_001".to_string(),
            timestamp: std::time::Instant::now(),
            export_format: "json".to_string(),
            file_path: "/tmp/profiling_data.json".to_string(),
            data_size_bytes: 1024 * 1024,
            export_duration_ms: 250,
            success: true,
        };
        assert_eq!(record.export_id, "exp_001");
        assert!(record.success);
        assert_eq!(record.data_size_bytes, 1024 * 1024);
    }

    // Test 16: AlertRecord construction
    #[test]
    fn test_alert_record_construction() {
        let record = AlertRecord {
            alert_id: "alert_001".to_string(),
            timestamp: std::time::Instant::now(),
            alert_type: "memory_pressure".to_string(),
            severity: "critical".to_string(),
            message: "Memory pressure detected".to_string(),
            resolved: false,
            resolution_time: None,
        };
        assert_eq!(record.alert_id, "alert_001");
        assert!(!record.resolved);
        assert!(record.resolution_time.is_none());
    }

    // Test 17: AlertRule construction
    #[test]
    fn test_alert_rule_construction() {
        let rule = AlertRule {
            rule_id: "rule_001".to_string(),
            name: "Memory Alert".to_string(),
            condition: "memory_usage > 80%".to_string(),
            threshold_value: 80.0,
            severity: "warning".to_string(),
            enabled: true,
            created_at: std::time::Instant::now(),
        };
        assert_eq!(rule.rule_id, "rule_001");
        assert!(rule.enabled);
        assert!((rule.threshold_value - 80.0).abs() < f32::EPSILON);
    }

    // Test 18: ProfilingState construction
    #[test]
    fn test_profiling_state_construction() {
        let state = ProfilingState {
            is_active: false,
            current_session_id: None,
            start_time: None,
            total_duration: Duration::ZERO,
            events_recorded: 0,
            snapshots_taken: 0,
            last_error: None,
        };
        assert!(!state.is_active);
        assert!(state.current_session_id.is_none());
        assert!(state.start_time.is_none());
        assert_eq!(state.events_recorded, 0);
    }

    // Test 19: BottleneckCondition with thermal state
    #[test]
    fn test_bottleneck_condition_thermal() {
        let condition = BottleneckCondition::ThermalThrottling {
            severity: ThermalState::Critical,
        };
        let debug_str = format!("{:?}", condition);
        assert!(debug_str.contains("ThermalThrottling"));
    }

    // Test 20: RealTimeState construction
    #[test]
    fn test_real_time_state_construction() {
        let state = RealTimeState {
            performance_score: 85.0,
            active_alerts: Vec::new(),
            trending_metrics: TrendingMetrics::default(),
            system_health: SystemHealth::default(),
            last_update: None,
            uptime: Duration::from_secs(3600),
        };
        assert!((state.performance_score - 85.0).abs() < f32::EPSILON);
        assert!(state.active_alerts.is_empty());
        assert!(state.last_update.is_none());
        assert_eq!(state.uptime, Duration::from_secs(3600));
    }

    // Test 21: Multiple BottleneckRules with LCG values
    #[test]
    fn test_multiple_bottleneck_rules_with_lcg() {
        let mut lcg = Lcg::new(777);
        let mut rules = Vec::new();
        for i in 0..5 {
            let rule = BottleneckRule {
                id: format!("rule_{}", i),
                name: format!("Rule {}", i),
                condition: BottleneckCondition::MemoryUsageHigh {
                    threshold_percent: lcg.next_f32() * 100.0,
                    duration_ms: (lcg.next() % 10000) + 100,
                },
                severity: if i < 2 {
                    BottleneckSeverity::Low
                } else {
                    BottleneckSeverity::High
                },
                suggestion: format!("Suggestion for rule {}", i),
                confidence: lcg.next_f32(),
                enabled: i % 2 == 0,
            };
            rules.push(rule);
        }
        assert_eq!(rules.len(), 5);
        let enabled_count = rules.iter().filter(|r| r.enabled).count();
        assert_eq!(enabled_count, 3);
    }

    // Test 22: OptimizationCondition GPU underutilized
    #[test]
    fn test_optimization_condition_gpu_underutilized() {
        let condition = OptimizationCondition::GPUUnderutilized {
            threshold_percent: 30.0,
            workload_type: WorkloadType::ComputerVision,
        };
        let debug_str = format!("{:?}", condition);
        assert!(debug_str.contains("GPUUnderutilized"));
        assert!(debug_str.contains("ComputerVision"));
    }

    // Test 23: LCG deterministic output
    #[test]
    fn test_lcg_deterministic() {
        let mut lcg1 = Lcg::new(42);
        let mut lcg2 = Lcg::new(42);
        for _ in 0..20 {
            assert_eq!(lcg1.next(), lcg2.next());
        }
    }

    // Test 24: BottleneckCondition network latency high
    #[test]
    fn test_bottleneck_condition_network_latency() {
        let condition = BottleneckCondition::NetworkLatencyHigh {
            threshold_ms: 200.0,
            sample_count: 50,
        };
        let debug_str = format!("{:?}", condition);
        assert!(debug_str.contains("NetworkLatencyHigh"));
    }

    // Test 25: BottleneckCondition cache hit rate low
    #[test]
    fn test_bottleneck_condition_cache_hit_rate_low() {
        let condition = BottleneckCondition::CacheHitRateLow {
            threshold_percent: 50.0,
            sample_count: 100,
        };
        let debug_str = format!("{:?}", condition);
        assert!(debug_str.contains("CacheHitRateLow"));
    }
}
