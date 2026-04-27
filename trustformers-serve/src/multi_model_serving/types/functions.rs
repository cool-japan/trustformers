//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::collections::HashMap;
    use std::time::Duration;
    fn make_performance_stats(
        avg_latency_ms: u64,
        throughput: f64,
        error_rate: f64,
    ) -> PerformanceStats {
        PerformanceStats {
            avg_latency: Duration::from_millis(avg_latency_ms),
            p95_latency: Duration::from_millis(avg_latency_ms * 2),
            p99_latency: Duration::from_millis(avg_latency_ms * 3),
            throughput,
            error_rate,
            accuracy: Some(0.95),
            request_count: 100,
        }
    }
    fn make_resource_usage(cpu: f64, mem: u64, gpu_mem: u64) -> ResourceUsage {
        ResourceUsage {
            cpu_usage: cpu,
            memory_usage: mem,
            gpu_memory_usage: gpu_mem,
            network_io: 0.0,
        }
    }
    fn make_model_info(id: &str, status: ModelStatus) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            name: format!("Model {}", id),
            version: "1.0.0".to_string(),
            characteristics: vec![],
            capabilities: vec!["text-generation".to_string()],
            status,
            performance_stats: make_performance_stats(50, 100.0, 0.01),
            resource_usage: make_resource_usage(0.3, 1024, 2048),
            metadata: HashMap::new(),
        }
    }
    fn make_minimal_config(strategy: RoutingStrategy) -> MultiModelConfig {
        MultiModelConfig {
            routing: ModelRoutingConfig {
                default_strategy: strategy,
                route_strategies: HashMap::new(),
                selection_criteria: ModelSelectionCriteria {
                    preferred_characteristics: vec![],
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
                    fallback_model: "fallback".to_string(),
                    enabled: false,
                    triggers: vec![],
                },
            },
            ensemble: EnsembleConfig {
                enabled: false,
                methods: vec![],
                voting_strategy: VotingStrategy::SimpleMajority,
                quality_assessment: QualityAssessmentConfig {
                    enabled: false,
                    methods: vec![],
                    thresholds: QualityThresholds {
                        min_accuracy: None,
                        max_latency: None,
                        max_error_rate: None,
                        min_throughput: None,
                    },
                },
                optimization: EnsembleOptimizationConfig {
                    enabled: false,
                    strategies: vec![],
                    resource_budget: ResourceBudget {
                        max_latency: Duration::from_secs(5),
                        max_memory: 4096,
                        max_compute_cost: 1.0,
                    },
                },
            },
            ab_testing: ABTestingConfig {
                enabled: false,
                experiments: vec![],
                significance_thresholds: StatisticalThresholds {
                    p_value: 0.05,
                    confidence_level: 0.95,
                    minimum_sample_size: 100,
                    minimum_effect_size: 0.05,
                },
            },
            traffic_splitting: TrafficSplittingConfig {
                enabled: false,
                split_rules: vec![],
                default_split: TrafficSplit {
                    splits: HashMap::new(),
                    sticky_sessions: false,
                },
            },
            model_cascading: ModelCascadingConfig {
                enabled: false,
                cascade_chains: vec![],
                exit_strategies: vec![],
            },
            performance_monitoring: PerformanceMonitoringConfig {
                enabled: false,
                collection_interval: Duration::from_secs(60),
                monitored_metrics: vec![],
                alerting_thresholds: AlertingThresholds {
                    latency_threshold: Duration::from_millis(500),
                    error_rate_threshold: 0.05,
                    accuracy_threshold: 0.9,
                    resource_threshold: 0.8,
                },
            },
        }
    }
    #[tokio::test]
    async fn test_register_and_route_single_model() {
        let config = make_minimal_config(RoutingStrategy::RoundRobin);
        let server = MultiModelServer::new(config);
        let model = make_model_info("m1", ModelStatus::Available);
        server.register_model(model).await.expect("register_model should succeed");
        let request = InferenceRequest {
            input_text: "hello".to_string(),
            path: "/infer".to_string(),
            headers: HashMap::new(),
            user_id: None,
            metadata: HashMap::new(),
        };
        let result = server.route_request(&request).await.expect("route_request should succeed");
        match result {
            RoutingResult::SingleModel { model_id } => assert_eq!(model_id, "m1"),
            _ => panic!("expected SingleModel routing result"),
        }
    }
    #[tokio::test]
    async fn test_unregister_model_removes_it() {
        let config = make_minimal_config(RoutingStrategy::RoundRobin);
        let server = MultiModelServer::new(config);
        let model = make_model_info("m1", ModelStatus::Available);
        server.register_model(model).await.expect("register should succeed");
        server.unregister_model("m1").await.expect("unregister should succeed");
        let request = InferenceRequest {
            input_text: "hello".to_string(),
            path: "/infer".to_string(),
            headers: HashMap::new(),
            user_id: None,
            metadata: HashMap::new(),
        };
        let result = server.route_request(&request).await;
        assert!(result.is_err(), "routing with no models should fail");
    }
    #[tokio::test]
    async fn test_unregister_nonexistent_model_returns_error() {
        let config = make_minimal_config(RoutingStrategy::RoundRobin);
        let server = MultiModelServer::new(config);
        let result = server.unregister_model("nonexistent").await;
        assert!(result.is_err(), "should fail for nonexistent model");
    }
    #[tokio::test]
    async fn test_route_request_no_models_returns_error() {
        let config = make_minimal_config(RoutingStrategy::RoundRobin);
        let server = MultiModelServer::new(config);
        let request = InferenceRequest {
            input_text: "test".to_string(),
            path: "/".to_string(),
            headers: HashMap::new(),
            user_id: None,
            metadata: HashMap::new(),
        };
        assert!(server.route_request(&request).await.is_err());
    }
    #[tokio::test]
    async fn test_route_request_unavailable_model_excluded() {
        let config = make_minimal_config(RoutingStrategy::RoundRobin);
        let server = MultiModelServer::new(config);
        server
            .register_model(make_model_info("m_down", ModelStatus::Unavailable))
            .await
            .expect("register");
        server
            .register_model(make_model_info("m_up", ModelStatus::Available))
            .await
            .expect("register");
        let request = InferenceRequest {
            input_text: "hello".to_string(),
            path: "/".to_string(),
            headers: HashMap::new(),
            user_id: None,
            metadata: HashMap::new(),
        };
        let result = server.route_request(&request).await.expect("should succeed");
        match result {
            RoutingResult::SingleModel { model_id } => assert_eq!(model_id, "m_up"),
            _ => panic!("expected single model"),
        }
    }
    #[tokio::test]
    async fn test_metrics_increment_on_routing() {
        let config = make_minimal_config(RoutingStrategy::RoundRobin);
        let server = MultiModelServer::new(config);
        server
            .register_model(make_model_info("m1", ModelStatus::Available))
            .await
            .expect("register");
        let request = InferenceRequest {
            input_text: "hello".to_string(),
            path: "/".to_string(),
            headers: HashMap::new(),
            user_id: None,
            metadata: HashMap::new(),
        };
        server.route_request(&request).await.expect("route");
        server.route_request(&request).await.expect("route");
        let metrics = server.get_metrics().await;
        assert_eq!(metrics.total_requests, 2, "total_requests should be 2");
    }
    #[tokio::test]
    async fn test_size_based_routing_selects_by_text_length() {
        let thresholds = vec![
            SizeThreshold {
                max_size: 10,
                target_model: "small".to_string(),
            },
            SizeThreshold {
                max_size: 100,
                target_model: "large".to_string(),
            },
        ];
        let config = make_minimal_config(RoutingStrategy::SizeBased {
            size_thresholds: thresholds,
        });
        let server = MultiModelServer::new(config);
        server
            .register_model(make_model_info("small", ModelStatus::Available))
            .await
            .expect("register");
        server
            .register_model(make_model_info("large", ModelStatus::Available))
            .await
            .expect("register");
        let short_req = InferenceRequest {
            input_text: "hi".to_string(),
            path: "/".to_string(),
            headers: HashMap::new(),
            user_id: None,
            metadata: HashMap::new(),
        };
        let result = server.route_request(&short_req).await.expect("route");
        match result {
            RoutingResult::SingleModel { model_id } => assert_eq!(model_id, "small"),
            _ => panic!("expected single model"),
        }
    }
    #[tokio::test]
    async fn test_user_based_routing_uses_user_map() {
        let mut user_map = HashMap::new();
        user_map.insert("alice".to_string(), "model_a".to_string());
        let strategy = RoutingStrategy::UserBased {
            user_model_map: user_map,
            default_model: "default_model".to_string(),
        };
        let config = make_minimal_config(strategy);
        let server = MultiModelServer::new(config);
        server
            .register_model(make_model_info("model_a", ModelStatus::Available))
            .await
            .expect("register");
        server
            .register_model(make_model_info("default_model", ModelStatus::Available))
            .await
            .expect("register");
        let req = InferenceRequest {
            input_text: "hello".to_string(),
            path: "/".to_string(),
            headers: HashMap::new(),
            user_id: Some("alice".to_string()),
            metadata: HashMap::new(),
        };
        let result = server.route_request(&req).await.expect("route");
        match result {
            RoutingResult::SingleModel { model_id } => assert_eq!(model_id, "model_a"),
            _ => panic!("expected single model"),
        }
    }
    #[tokio::test]
    async fn test_user_based_routing_falls_back_to_default() {
        let strategy = RoutingStrategy::UserBased {
            user_model_map: HashMap::new(),
            default_model: "default_model".to_string(),
        };
        let config = make_minimal_config(strategy);
        let server = MultiModelServer::new(config);
        server
            .register_model(make_model_info("default_model", ModelStatus::Available))
            .await
            .expect("register");
        let req = InferenceRequest {
            input_text: "hello".to_string(),
            path: "/".to_string(),
            headers: HashMap::new(),
            user_id: Some("unknown_user".to_string()),
            metadata: HashMap::new(),
        };
        let result = server.route_request(&req).await.expect("route");
        match result {
            RoutingResult::SingleModel { model_id } => {
                assert_eq!(model_id, "default_model")
            },
            _ => panic!("expected single model"),
        }
    }
    #[tokio::test]
    async fn test_performance_based_routing_picks_best_score() {
        let strategy = RoutingStrategy::PerformanceBased {
            metrics: vec![PerformanceMetric::Throughput],
            weights: HashMap::new(),
        };
        let config = make_minimal_config(strategy);
        let server = MultiModelServer::new(config);
        let mut fast_model = make_model_info("fast", ModelStatus::Available);
        fast_model.performance_stats.throughput = 500.0;
        let mut slow_model = make_model_info("slow", ModelStatus::Available);
        slow_model.performance_stats.throughput = 50.0;
        server.register_model(fast_model).await.expect("register");
        server.register_model(slow_model).await.expect("register");
        let req = InferenceRequest {
            input_text: "test".to_string(),
            path: "/".to_string(),
            headers: HashMap::new(),
            user_id: None,
            metadata: HashMap::new(),
        };
        let result = server.route_request(&req).await.expect("route");
        match result {
            RoutingResult::SingleModel { model_id } => assert_eq!(model_id, "fast"),
            _ => panic!("expected single model"),
        }
    }
    #[tokio::test]
    async fn test_ab_testing_disabled_rejects_start() {
        let config = make_minimal_config(RoutingStrategy::RoundRobin);
        let server = MultiModelServer::new(config);
        let experiment = ABTestExperiment {
            id: "exp1".to_string(),
            name: "Test Experiment".to_string(),
            control_model: "ctrl".to_string(),
            variant_models: vec![],
            traffic_allocation: TrafficAllocation {
                control_percentage: 1.0,
                variant_percentages: HashMap::new(),
                allocation_method: AllocationMethod::Random,
            },
            success_metrics: vec![SuccessMetric::Accuracy],
            duration: Duration::from_secs(3600),
            statistical_power: 0.8,
        };
        let result = server.start_ab_test(experiment).await;
        assert!(result.is_err(), "ab testing disabled should return error");
    }
    #[tokio::test]
    async fn test_performance_stats_default_zeroes() {
        let stats = PerformanceStats::default();
        assert_eq!(stats.avg_latency, Duration::from_millis(0));
        assert_eq!(stats.error_rate, 0.0);
        assert_eq!(stats.throughput, 0.0);
    }
    #[tokio::test]
    async fn test_route_picks_strategy_by_path_prefix() {
        let mut route_strategies = HashMap::new();
        route_strategies.insert("/v2/".to_string(), RoutingStrategy::RoundRobin);
        let mut config = make_minimal_config(RoutingStrategy::RoundRobin);
        config.routing.route_strategies = route_strategies;
        let server = MultiModelServer::new(config);
        server
            .register_model(make_model_info("m1", ModelStatus::Available))
            .await
            .expect("register");
        let req = InferenceRequest {
            input_text: "test".to_string(),
            path: "/v2/infer".to_string(),
            headers: HashMap::new(),
            user_id: None,
            metadata: HashMap::new(),
        };
        let result = server.route_request(&req).await.expect("route");
        assert!(matches!(result, RoutingResult::SingleModel { .. }));
    }
    #[tokio::test]
    async fn test_content_based_routing_respects_priority() {
        let rules = vec![
            ContentRoutingRule {
                name: "low_priority".to_string(),
                condition: RoutingCondition::TextLength {
                    min: Some(1),
                    max: None,
                },
                target_model: "generic".to_string(),
                priority: 1,
            },
            ContentRoutingRule {
                name: "high_priority".to_string(),
                condition: RoutingCondition::Keywords {
                    keywords: vec!["special".to_string()],
                    match_all: false,
                },
                target_model: "specialized".to_string(),
                priority: 10,
            },
        ];
        let strategy = RoutingStrategy::ContentBased { rules };
        let config = make_minimal_config(strategy);
        let server = MultiModelServer::new(config);
        server
            .register_model(make_model_info("specialized", ModelStatus::Available))
            .await
            .expect("register");
        server
            .register_model(make_model_info("generic", ModelStatus::Available))
            .await
            .expect("register");
        let req = InferenceRequest {
            input_text: "special request".to_string(),
            path: "/".to_string(),
            headers: HashMap::new(),
            user_id: None,
            metadata: HashMap::new(),
        };
        let result = server.route_request(&req).await.expect("route");
        match result {
            RoutingResult::SingleModel { model_id } => {
                assert_eq!(model_id, "specialized")
            },
            _ => panic!("expected single model"),
        }
    }
    #[tokio::test]
    async fn test_resource_based_routing_selects_low_utilization_model() {
        let strategy = RoutingStrategy::ResourceBased {
            cpu_threshold: 0.5,
            memory_threshold: 2000.0,
            gpu_threshold: 4000.0,
        };
        let config = make_minimal_config(strategy);
        let server = MultiModelServer::new(config);
        let mut heavy = make_model_info("heavy", ModelStatus::Available);
        heavy.resource_usage.cpu_usage = 0.9;
        let mut light = make_model_info("light", ModelStatus::Available);
        light.resource_usage.cpu_usage = 0.1;
        server.register_model(heavy).await.expect("register");
        server.register_model(light).await.expect("register");
        let req = InferenceRequest {
            input_text: "test".to_string(),
            path: "/".to_string(),
            headers: HashMap::new(),
            user_id: None,
            metadata: HashMap::new(),
        };
        let result = server.route_request(&req).await.expect("route");
        match result {
            RoutingResult::SingleModel { model_id } => assert_eq!(model_id, "light"),
            _ => panic!("expected single model"),
        }
    }
    #[tokio::test]
    async fn test_capability_based_routing_matches_model_by_capability() {
        let mut cap_map = HashMap::new();
        cap_map.insert(
            "cap_model".to_string(),
            vec!["text-generation".to_string(), "translation".to_string()],
        );
        let strategy = RoutingStrategy::CapabilityBased {
            capability_map: cap_map,
        };
        let config = make_minimal_config(strategy);
        let server = MultiModelServer::new(config);
        server
            .register_model(make_model_info("cap_model", ModelStatus::Available))
            .await
            .expect("register");
        let req = InferenceRequest {
            input_text: "translate this".to_string(),
            path: "/".to_string(),
            headers: HashMap::new(),
            user_id: None,
            metadata: HashMap::new(),
        };
        let result = server.route_request(&req).await.expect("route");
        match result {
            RoutingResult::SingleModel { model_id } => assert_eq!(model_id, "cap_model"),
            _ => panic!("expected single model"),
        }
    }
    #[tokio::test]
    async fn test_traffic_split_default_zero_splits() {
        let split = TrafficSplit {
            splits: HashMap::new(),
            sticky_sessions: false,
        };
        assert!(split.splits.is_empty());
        assert!(!split.sticky_sessions);
    }
    #[tokio::test]
    async fn test_alerting_thresholds_values() {
        let thresholds = AlertingThresholds {
            latency_threshold: Duration::from_millis(200),
            error_rate_threshold: 0.1,
            accuracy_threshold: 0.85,
            resource_threshold: 0.75,
        };
        assert_eq!(thresholds.latency_threshold.as_millis(), 200);
        assert!((thresholds.error_rate_threshold - 0.1).abs() < 1e-9);
    }
    #[tokio::test]
    async fn test_model_status_variants_distinct() {
        let statuses = vec![
            ModelStatus::Available,
            ModelStatus::Loading,
            ModelStatus::Unavailable,
            ModelStatus::Maintenance,
            ModelStatus::Deprecated,
        ];
        assert_eq!(statuses.len(), 5);
    }
    #[tokio::test]
    async fn test_routing_result_ensemble_variant() {
        let result = RoutingResult::Ensemble {
            method: EnsembleMethod::MajorityVoting {
                models: vec!["m1".to_string(), "m2".to_string()],
                weights: None,
            },
            models: vec!["m1".to_string(), "m2".to_string()],
        };
        match result {
            RoutingResult::Ensemble { models, .. } => assert_eq!(models.len(), 2),
            _ => panic!("expected ensemble"),
        }
    }
    #[tokio::test]
    async fn test_ab_test_experiment_traffic_percentage_validation() {
        let config = {
            let mut c = make_minimal_config(RoutingStrategy::RoundRobin);
            c.ab_testing.enabled = true;
            c
        };
        let server = MultiModelServer::new(config);
        server
            .register_model(make_model_info("ctrl", ModelStatus::Available))
            .await
            .expect("register");
        server
            .register_model(make_model_info("var1", ModelStatus::Available))
            .await
            .expect("register");
        let bad_experiment = ABTestExperiment {
            id: "bad".to_string(),
            name: "bad exp".to_string(),
            control_model: "ctrl".to_string(),
            variant_models: vec![ABTestVariant {
                id: "v1".to_string(),
                model: "var1".to_string(),
                traffic_percentage: 0.8,
                configuration: HashMap::new(),
            }],
            traffic_allocation: TrafficAllocation {
                control_percentage: 0.5,
                variant_percentages: HashMap::new(),
                allocation_method: AllocationMethod::Random,
            },
            success_metrics: vec![SuccessMetric::Accuracy],
            duration: Duration::from_secs(3600),
            statistical_power: 0.8,
        };
        let result = server.start_ab_test(bad_experiment).await;
        assert!(
            result.is_err(),
            "percentages not summing to 1.0 should fail"
        );
    }
    #[test]
    fn test_performance_stats_latency_scoring() {
        let fast = make_performance_stats(10, 200.0, 0.0);
        let slow = make_performance_stats(1000, 200.0, 0.0);
        let fast_score = 1.0 / (fast.avg_latency.as_millis() as f64 + 1.0);
        let slow_score = 1.0 / (slow.avg_latency.as_millis() as f64 + 1.0);
        assert!(
            fast_score > slow_score,
            "lower latency should yield higher score"
        );
    }
    #[test]
    fn test_model_characteristic_variants() {
        let chars = vec![
            ModelCharacteristic::Size(ModelSize::Large),
            ModelCharacteristic::Accuracy(0.97),
            ModelCharacteristic::Latency(Duration::from_millis(50)),
            ModelCharacteristic::Language("en".to_string()),
            ModelCharacteristic::Domain("nlp".to_string()),
            ModelCharacteristic::Task("classification".to_string()),
        ];
        assert_eq!(chars.len(), 6);
    }
    #[test]
    fn test_traffic_split_sticky_sessions_flag() {
        let split = TrafficSplit {
            splits: [("m1".to_string(), 0.7), ("m2".to_string(), 0.3)].iter().cloned().collect(),
            sticky_sessions: true,
        };
        assert!(split.sticky_sessions);
        let total: f64 = split.splits.values().sum();
        assert!((total - 1.0).abs() < 1e-9);
    }
    #[tokio::test]
    async fn test_metrics_model_request_counts_tracked() {
        let config = make_minimal_config(RoutingStrategy::RoundRobin);
        let server = MultiModelServer::new(config);
        server
            .register_model(make_model_info("m1", ModelStatus::Available))
            .await
            .expect("register");
        let req = InferenceRequest {
            input_text: "hello".to_string(),
            path: "/".to_string(),
            headers: HashMap::new(),
            user_id: None,
            metadata: HashMap::new(),
        };
        for _ in 0..3 {
            server.route_request(&req).await.expect("route");
        }
        let metrics = server.get_metrics().await;
        let count = metrics.model_request_counts.get("m1").copied().unwrap_or(0);
        assert_eq!(count, 3, "model request count should be 3");
    }
    #[test]
    fn test_quality_thresholds_construction() {
        let qt = QualityThresholds {
            min_accuracy: Some(0.9),
            max_latency: Some(Duration::from_millis(100)),
            max_error_rate: Some(0.05),
            min_throughput: Some(50.0),
        };
        assert!(qt.min_accuracy.is_some());
        assert!((qt.min_accuracy.expect("min_accuracy should be set") - 0.9).abs() < 1e-9);
        assert_eq!(
            qt.max_latency.expect("max_latency should be set").as_millis(),
            100
        );
    }
}
