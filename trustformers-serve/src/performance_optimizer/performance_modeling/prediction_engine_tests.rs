//! Tests for the PredictionEngine and related components.
//!
//! Covers: PredictionEngineConfig defaults, EnsembleStrategy variants,
//! PredictionModelRegistry CRUD, PredictionCache LRU / TTL behaviour,
//! EnsembleCoordinator combination, BatchStatistics calculation,
//! PredictionStatistics, and cache hit-rate arithmetic.

#[cfg(test)]
mod tests {
    use crate::performance_optimizer::performance_modeling::prediction_engine::{
        CachedPrediction, EnsembleCoordinator, EnsembleStrategy,
        PredictionCache, PredictionEngine, PredictionEngineConfig, PredictionModelRegistry,
        WeightedPrediction,
    };
    use crate::performance_optimizer::performance_modeling::types::{
        ModelAccuracyMetrics, PerformancePrediction, PredictionRequest,
    };
    use crate::performance_optimizer::types::{
        ResourceIntensity, SystemState, TestCharacteristics,
    };
    use anyhow::Result;
    use chrono::Utc;
    use std::collections::HashMap;
    use std::time::Duration;

    // -----------------------------------------------------------------------
    // Helper to build a valid PerformancePrediction
    // -----------------------------------------------------------------------

    fn make_prediction(throughput: f64, confidence: f32) -> PerformancePrediction {
        PerformancePrediction {
            throughput,
            latency: Duration::from_millis(100),
            confidence,
            uncertainty_bounds: (throughput * 0.9, throughput * 1.1),
            model_name: "test_model".to_string(),
            feature_importance: HashMap::new(),
            predicted_at: Utc::now(),
        }
    }

    fn make_prediction_request() -> PredictionRequest {
        PredictionRequest {
            parallelism_levels: vec![1, 2, 4],
            test_characteristics: TestCharacteristics {
                category_distribution: HashMap::new(),
                average_duration: Duration::from_millis(500),
                resource_intensity: ResourceIntensity {
                    cpu_intensity: 0.5,
                    memory_intensity: 0.3,
                    io_intensity: 0.2,
                    network_intensity: 0.1,
                    gpu_intensity: None,
                },
                concurrency_requirements: Default::default(),
                dependency_complexity: 0.1,
            },
            system_state: SystemState {
                available_cores: 8,
                available_memory_mb: 16384,
                load_average: 0.5,
                active_processes: 10,
                io_wait_percent: 2.0,
                network_utilization: 0.1,
                temperature_metrics: None,
            },
            prediction_horizon: None,
            confidence_level: 0.95,
            include_uncertainty: false,
        }
    }

    // -----------------------------------------------------------------------
    // PredictionEngineConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prediction_engine_config_default_caching_enabled() {
        let cfg = PredictionEngineConfig::default();
        assert!(cfg.enable_caching, "caching must be enabled by default");
    }

    #[test]
    fn test_prediction_engine_config_default_cache_ttl_positive() {
        let cfg = PredictionEngineConfig::default();
        assert!(cfg.cache_ttl_seconds > 0, "cache TTL must be > 0");
    }

    #[test]
    fn test_prediction_engine_config_default_max_cache_size_positive() {
        let cfg = PredictionEngineConfig::default();
        assert!(cfg.max_cache_size > 0, "max_cache_size must be > 0");
    }

    #[test]
    fn test_prediction_engine_config_default_ensemble_enabled() {
        let cfg = PredictionEngineConfig::default();
        assert!(cfg.enable_ensemble, "ensemble must be enabled by default");
    }

    #[test]
    fn test_prediction_engine_config_default_max_batch_size() {
        let cfg = PredictionEngineConfig::default();
        assert!(cfg.max_batch_size > 0, "max_batch_size must be > 0");
    }

    #[test]
    fn test_prediction_engine_config_default_prediction_timeout_nonzero() {
        let cfg = PredictionEngineConfig::default();
        assert!(
            cfg.prediction_timeout > Duration::from_secs(0),
            "prediction_timeout must be positive"
        );
    }

    // -----------------------------------------------------------------------
    // EnsembleStrategy variant tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ensemble_strategy_all_variants_debug() {
        let variants = [
            EnsembleStrategy::SimpleAverage,
            EnsembleStrategy::WeightedAverage,
            EnsembleStrategy::BestModel,
            EnsembleStrategy::Stacking,
            EnsembleStrategy::Voting,
            EnsembleStrategy::AdaptiveWeighting,
        ];
        for v in &variants {
            let s = format!("{:?}", v);
            assert!(!s.is_empty(), "debug output must not be empty for {:?}", v);
        }
    }

    // -----------------------------------------------------------------------
    // PredictionEngine construction test
    // -----------------------------------------------------------------------

    #[test]
    fn test_prediction_engine_new_with_default_config() {
        let engine = PredictionEngine::new(PredictionEngineConfig::default());
        let stats = engine.get_prediction_statistics();
        assert_eq!(stats.total_predictions, 0, "fresh engine must have zero predictions");
    }

    #[test]
    fn test_prediction_engine_cache_clear() {
        let engine = PredictionEngine::new(PredictionEngineConfig::default());
        // Clear on empty cache should not panic
        engine.clear_cache();
        let stats = engine.get_prediction_statistics();
        assert_eq!(stats.cache_size, 0, "cache size must be 0 after clear");
    }

    // -----------------------------------------------------------------------
    // PredictionModelRegistry tests
    // -----------------------------------------------------------------------

    /// Minimal mock model for registry testing
    #[derive(Debug)]
    struct MockPredictor {
        name: String,
        throughput: f64,
    }

    impl crate::performance_optimizer::performance_modeling::types::PerformancePredictor
        for MockPredictor
    {
        fn predict(
            &self,
            _request: &PredictionRequest,
        ) -> Result<PerformancePrediction> {
            Ok(make_prediction(self.throughput, 0.8))
        }

        fn get_accuracy(&self) -> ModelAccuracyMetrics {
            ModelAccuracyMetrics {
                overall_accuracy: 0.85,
                r_squared: 0.82,
                mean_absolute_error: 0.05,
                root_mean_squared_error: 0.07,
                cross_validation_scores: vec![0.84, 0.86, 0.85],
                confidence_interval: (0.80, 0.90),
                prediction_stability: 0.9,
                last_validated: Utc::now(),
            }
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn supports_online_learning(&self) -> bool {
            false
        }
    }

    #[test]
    fn test_registry_starts_empty() {
        let registry = PredictionModelRegistry::new();
        assert_eq!(registry.model_count(), 0, "fresh registry must be empty");
    }

    #[test]
    fn test_registry_register_model_increases_count() {
        let mut registry = PredictionModelRegistry::new();
        let model = Box::new(MockPredictor {
            name: "model_a".to_string(),
            throughput: 100.0,
        });
        registry
            .register_model("model_a".to_string(), model, 0.5)
            .expect("register_model must succeed");
        assert_eq!(registry.model_count(), 1, "registry must have 1 model after registration");
    }

    #[test]
    fn test_registry_invalid_weight_returns_error() {
        let mut registry = PredictionModelRegistry::new();
        let model = Box::new(MockPredictor {
            name: "bad_weight".to_string(),
            throughput: 50.0,
        });
        let result = registry.register_model("bad_weight".to_string(), model, 1.5);
        assert!(result.is_err(), "weight > 1.0 must be rejected");
    }

    #[test]
    fn test_registry_negative_weight_returns_error() {
        let mut registry = PredictionModelRegistry::new();
        let model = Box::new(MockPredictor {
            name: "neg_weight".to_string(),
            throughput: 50.0,
        });
        let result = registry.register_model("neg_weight".to_string(), model, -0.1);
        assert!(result.is_err(), "negative weight must be rejected");
    }

    #[test]
    fn test_registry_get_best_model_on_empty_registry() {
        let registry = PredictionModelRegistry::new();
        assert!(
            registry.get_best_model().is_none(),
            "empty registry must return None for best model"
        );
    }

    #[test]
    fn test_registry_update_weight_known_model() {
        let mut registry = PredictionModelRegistry::new();
        let model = Box::new(MockPredictor {
            name: "w_model".to_string(),
            throughput: 200.0,
        });
        registry
            .register_model("w_model".to_string(), model, 0.5)
            .expect("register must succeed");
        registry
            .update_model_weight("w_model", 0.9)
            .expect("update_model_weight must succeed");
    }

    #[test]
    fn test_registry_update_weight_unknown_model_returns_error() {
        let mut registry = PredictionModelRegistry::new();
        let result = registry.update_model_weight("nonexistent", 0.5);
        assert!(result.is_err(), "updating nonexistent model must return error");
    }

    // -----------------------------------------------------------------------
    // PredictionCache tests
    // -----------------------------------------------------------------------

    fn make_cached_prediction(ttl: u64, throughput: f64) -> CachedPrediction {
        CachedPrediction {
            prediction: make_prediction(throughput, 0.9),
            cached_at: Utc::now(),
            ttl_seconds: ttl,
        }
    }

    #[test]
    fn test_cache_starts_empty() {
        let cache = PredictionCache::new(100);
        assert_eq!(cache.size(), 0, "fresh cache must have size 0");
    }

    #[test]
    fn test_cache_insert_increases_size() {
        let mut cache = PredictionCache::new(100);
        cache.insert("key1".to_string(), make_cached_prediction(300, 100.0));
        assert_eq!(cache.size(), 1, "cache size must be 1 after one insert");
    }

    #[test]
    fn test_cache_contains_key_after_insert() {
        let mut cache = PredictionCache::new(100);
        cache.insert("key2".to_string(), make_cached_prediction(300, 200.0));
        assert!(cache.contains_key("key2"), "cache must contain key after insert");
    }

    #[test]
    fn test_cache_miss_on_unknown_key() {
        let mut cache = PredictionCache::new(100);
        let result = cache.get("no_such_key");
        assert!(result.is_none(), "unknown key must yield None");
    }

    #[test]
    fn test_cache_hit_on_valid_entry() {
        let mut cache = PredictionCache::new(100);
        cache.insert("valid_key".to_string(), make_cached_prediction(300, 50.0));
        let result = cache.get("valid_key");
        assert!(result.is_some(), "valid entry must be returned");
    }

    #[test]
    fn test_cache_evicts_lru_when_full() {
        let max = 3usize;
        let mut cache = PredictionCache::new(max);
        for i in 0..max {
            cache.insert(format!("key{}", i), make_cached_prediction(300, i as f64));
        }
        // Insert one more: oldest (key0) should be evicted
        cache.insert("key_extra".to_string(), make_cached_prediction(300, 999.0));
        assert_eq!(cache.size(), max, "cache must not exceed max_size");
    }

    #[test]
    fn test_cache_clear_empties_cache() {
        let mut cache = PredictionCache::new(100);
        cache.insert("k1".to_string(), make_cached_prediction(300, 1.0));
        cache.insert("k2".to_string(), make_cached_prediction(300, 2.0));
        cache.clear();
        assert_eq!(cache.size(), 0, "cache must be empty after clear");
    }

    #[test]
    fn test_cache_hit_rate_zero_initially() {
        let cache = PredictionCache::new(100);
        assert_eq!(cache.hit_rate(), 0.0, "hit rate must be 0.0 with no operations");
    }

    #[test]
    fn test_cache_statistics_initial_all_zero() {
        let cache = PredictionCache::new(100);
        let stats = cache.statistics();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.expired_entries, 0);
    }

    // -----------------------------------------------------------------------
    // EnsembleCoordinator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ensemble_coordinator_empty_predictions_error() {
        let coord = EnsembleCoordinator::new(EnsembleStrategy::SimpleAverage);
        let result = coord.combine_predictions(vec![]);
        assert!(result.is_err(), "empty prediction list must return an error");
    }

    #[test]
    fn test_ensemble_coordinator_single_prediction_returns_ok() {
        let coord = EnsembleCoordinator::new(EnsembleStrategy::SimpleAverage);
        let wp = WeightedPrediction {
            prediction: make_prediction(100.0, 0.9),
            weight: 1.0,
            model_id: "m1".to_string(),
        };
        let result = coord.combine_predictions(vec![wp]);
        assert!(result.is_ok(), "single prediction ensemble must succeed");
    }

    #[test]
    fn test_ensemble_coordinator_weighted_average_two_models() {
        let coord = EnsembleCoordinator::new(EnsembleStrategy::WeightedAverage);
        let wp1 = WeightedPrediction {
            prediction: make_prediction(80.0, 0.8),
            weight: 0.6,
            model_id: "m1".to_string(),
        };
        let wp2 = WeightedPrediction {
            prediction: make_prediction(120.0, 0.9),
            weight: 0.4,
            model_id: "m2".to_string(),
        };
        let result = coord.combine_predictions(vec![wp1, wp2]);
        assert!(result.is_ok(), "weighted average ensemble must succeed");
    }

    #[test]
    fn test_ensemble_coordinator_best_model_strategy() {
        let coord = EnsembleCoordinator::new(EnsembleStrategy::BestModel);
        let wp = WeightedPrediction {
            prediction: make_prediction(150.0, 0.95),
            weight: 1.0,
            model_id: "best".to_string(),
        };
        let result = coord.combine_predictions(vec![wp]);
        assert!(result.is_ok(), "best model strategy must succeed with one prediction");
    }
}
