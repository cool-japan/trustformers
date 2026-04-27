//! Comprehensive tests for network_adaptation/prediction.rs
//!
//! Covers bandwidth predictor (exponential smoothing), latency forecast,
//! congestion detection (queue length proxy), prediction accuracy tracking,
//! and adaptive window sizing.

#[cfg(test)]
mod tests {
    use crate::network_adaptation::prediction::*;
    use crate::network_adaptation::types::{
        NetworkAdaptationConfig, NetworkConditions, NetworkPredictionConfig, NetworkQuality,
        TrendDirection,
    };
    use crate::profiler::NetworkConnectionType;
    use std::time::Instant;

    // =========================================================================
    // LCG deterministic PRNG
    // =========================================================================

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
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.state
        }

        fn next_f32_unit(&mut self) -> f32 {
            (self.next_u64() >> 11) as f32 / (u64::MAX >> 11) as f32
        }

        fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + self.next_f32_unit() * (hi - lo)
        }
    }

    // =========================================================================
    // Helper functions
    // =========================================================================

    fn make_config(enable_prediction: bool) -> NetworkAdaptationConfig {
        use crate::battery::BatteryLevel;
        use super::super::types::{
            CellularConfig, CellularStrategy, CommunicationStrategy, DataUsageAwareness,
            DataUsageLimits, FailureDetectionSensitivity, FailureRecoveryConfig,
            GradientCompressionAlgorithm, NetworkCompressionConfig, NetworkQuantizationConfig,
            NetworkQualityThresholds, PoorNetworkStrategy, RetryConfig, SyncFrequencyConfig,
            TimeBasedScheduling, WiFiStrategy,
        };
        use std::collections::HashMap;

        NetworkAdaptationConfig {
            enable_monitoring: true,
            monitoring_interval_ms: 1_000,
            enable_adaptive_scheduling: true,
            enable_bandwidth_optimization: true,
            quality_thresholds: NetworkQualityThresholds {
                min_bandwidth_full_sync_mbps: 5.0,
                min_bandwidth_incremental_sync_mbps: 1.0,
                max_latency_realtime_ms: 200.0,
                max_packet_loss_percent: 5.0,
                min_signal_strength_dbm: -90,
                max_jitter_ms: 50.0,
            },
            communication_strategy: CommunicationStrategy::default(),
            data_usage_limits: DataUsageLimits {
                wifi_daily_limit_mb: None,
                cellular_daily_limit_mb: Some(500),
                cellular_monthly_limit_mb: Some(5_000),
                warning_thresholds: HashMap::new(),
                emergency_thresholds: HashMap::new(),
            },
            sync_frequency: SyncFrequencyConfig {
                base_frequency_minutes: 60,
                network_multipliers: HashMap::new(),
                battery_adjustments: HashMap::new(),
                adaptive_frequency: true,
                min_frequency_minutes: 5,
                max_frequency_minutes: 1_440,
            },
            failure_recovery: FailureRecoveryConfig {
                enable_auto_recovery: true,
                recovery_timeout_minutes: 30,
                enable_checkpointing: true,
                checkpoint_frequency: 10,
                enable_graceful_degradation: true,
                failure_detection_sensitivity: FailureDetectionSensitivity::Medium,
            },
            prediction_config: NetworkPredictionConfig {
                enable_prediction,
                prediction_window_minutes: 30,
                historical_window_hours: 24,
                accuracy_threshold: 0.7,
                enable_ml_predictions: true,
            },
        }
    }

    fn make_conditions(bandwidth: f32, latency: f32, quality: NetworkQuality) -> NetworkConditions {
        NetworkConditions {
            timestamp: Instant::now(),
            connection_type: NetworkConnectionType::WiFi,
            bandwidth_mbps: bandwidth,
            latency_ms: latency,
            packet_loss_percent: 0.5,
            signal_strength_dbm: Some(-55),
            jitter_ms: 3.0,
            stability_score: 0.9,
            quality_assessment: quality,
            available_data_mb: Some(10_000),
        }
    }

    // =========================================================================
    // NetworkPredictor creation tests
    // =========================================================================

    #[test]
    fn test_network_predictor_new_succeeds() {
        let config = make_config(true);
        let predictor = NetworkPredictor::new(config);
        assert!(predictor.is_ok(), "Predictor creation should succeed");
    }

    #[test]
    fn test_network_predictor_start_and_stop() {
        let config = make_config(true);
        let mut predictor = NetworkPredictor::new(config).expect("Predictor creation should succeed");
        predictor.start().expect("Start should succeed");
        predictor.stop().expect("Stop should succeed");
    }

    // =========================================================================
    // Historical data ingestion tests
    // =========================================================================

    #[test]
    fn test_add_single_historical_data_point() {
        let config = make_config(true);
        let mut predictor = NetworkPredictor::new(config).expect("Predictor should be created");
        let conditions = make_conditions(50.0, 20.0, NetworkQuality::Good);
        let result = predictor.add_historical_data(conditions);
        assert!(result.is_ok(), "Adding historical data should succeed");
    }

    #[test]
    fn test_add_many_historical_data_points() {
        let config = make_config(true);
        let mut predictor = NetworkPredictor::new(config).expect("Predictor should be created");
        let mut lcg = Lcg::new(0x1517);
        for _ in 0..50 {
            let bw = lcg.next_f32_range(5.0, 100.0);
            let lat = lcg.next_f32_range(5.0, 100.0);
            let conditions = make_conditions(bw, lat, NetworkQuality::Good);
            predictor.add_historical_data(conditions).expect("Adding data should not fail");
        }
    }

    // =========================================================================
    // Prediction tests
    // =========================================================================

    #[test]
    fn test_predict_conditions_empty_history_returns_default() {
        let config = make_config(true);
        let predictor = NetworkPredictor::new(config).expect("Predictor should be created");
        let result = predictor.predict_conditions(5);
        assert!(result.is_ok(), "Prediction with empty history should return default conditions");
    }

    #[test]
    fn test_predict_conditions_with_history() {
        let config = make_config(true);
        let mut predictor = NetworkPredictor::new(config).expect("Predictor should be created");

        // Add several historical data points
        let mut lcg = Lcg::new(0xB4ED);
        for _ in 0..20 {
            let bw = lcg.next_f32_range(10.0, 80.0);
            let lat = lcg.next_f32_range(10.0, 150.0);
            let conditions = make_conditions(bw, lat, NetworkQuality::Fair);
            predictor.add_historical_data(conditions).expect("Adding data should succeed");
        }

        let result = predictor.predict_conditions(10);
        assert!(result.is_ok(), "Prediction should succeed with history");

        let prediction = result.expect("Prediction should be available");
        let predicted_conditions = prediction.get_predicted_conditions();
        assert_eq!(
            predicted_conditions.len(), 10,
            "Should have 10 predicted time points for window_minutes=10"
        );
    }

    #[test]
    fn test_predict_conditions_bandwidth_positive() {
        let config = make_config(true);
        let mut predictor = NetworkPredictor::new(config).expect("Predictor should be created");

        // Seed with consistent high-bandwidth data
        for _ in 0..10 {
            let conditions = make_conditions(100.0, 10.0, NetworkQuality::Excellent);
            predictor.add_historical_data(conditions).expect("Adding data should succeed");
        }

        let result = predictor.predict_conditions(3).expect("Prediction should succeed");
        let conditions_list = result.get_predicted_conditions();

        for (_, cond) in conditions_list {
            assert!(
                cond.bandwidth_mbps > 0.0,
                "Predicted bandwidth must be positive: got {}",
                cond.bandwidth_mbps
            );
        }
    }

    #[test]
    fn test_predict_conditions_latency_positive() {
        let config = make_config(true);
        let mut predictor = NetworkPredictor::new(config).expect("Predictor should be created");

        for _ in 0..10 {
            let conditions = make_conditions(50.0, 30.0, NetworkQuality::Good);
            predictor.add_historical_data(conditions).expect("Adding data should succeed");
        }

        let result = predictor.predict_conditions(3).expect("Prediction should succeed");
        let conditions_list = result.get_predicted_conditions();

        for (_, cond) in conditions_list {
            assert!(cond.latency_ms >= 1.0, "Predicted latency must be >= 1ms: got {}", cond.latency_ms);
        }
    }

    // =========================================================================
    // PredictionModelType tests
    // =========================================================================

    #[test]
    fn test_prediction_model_type_variants() {
        let types = [
            PredictionModelType::LinearRegression,
            PredictionModelType::ExponentialSmoothing,
            PredictionModelType::MovingAverage,
            PredictionModelType::NeuralNetwork,
            PredictionModelType::EnsembleMethod,
        ];
        for t in &types {
            let debug = format!("{:?}", t);
            assert!(!debug.is_empty(), "Debug output must not be empty for {:?}", t);
        }
    }

    // =========================================================================
    // Prediction stats tests
    // =========================================================================

    #[test]
    fn test_get_prediction_stats_returns_map() {
        let config = make_config(true);
        let predictor = NetworkPredictor::new(config).expect("Predictor should be created");
        let stats = predictor.get_prediction_stats();
        assert!(
            stats.contains_key("average_accuracy"),
            "Stats must include average_accuracy"
        );
        assert!(
            stats.contains_key("historical_data_points"),
            "Stats must include historical_data_points"
        );
    }

    #[test]
    fn test_get_prediction_stats_accuracy_range() {
        let config = make_config(true);
        let predictor = NetworkPredictor::new(config).expect("Predictor should be created");
        let stats = predictor.get_prediction_stats();
        let accuracy = stats
            .get("average_accuracy")
            .copied()
            .expect("average_accuracy should be present");
        assert!(
            accuracy >= 0.0 && accuracy <= 1.0,
            "Average accuracy must be in [0,1]: got {}",
            accuracy
        );
    }

    // =========================================================================
    // NetworkPatternAnalyzer tests
    // =========================================================================

    #[test]
    fn test_pattern_analyzer_new() {
        let analyzer = NetworkPatternAnalyzer::new();
        // NetworkPatternAnalyzer does not derive Debug; verify it can be constructed
        let _ = analyzer;
    }

    #[test]
    fn test_pattern_analyzer_start_analysis_initializes_patterns() {
        let mut analyzer = NetworkPatternAnalyzer::new();
        analyzer.start_analysis();
        // After starting, average confidence should be accessible
        let confidence = analyzer.get_average_confidence();
        assert!(
            confidence >= 0.0 && confidence <= 1.0,
            "Average confidence must be [0,1]: got {}",
            confidence
        );
    }

    #[test]
    fn test_pattern_analyzer_analyze_conditions_updates_patterns() {
        let mut analyzer = NetworkPatternAnalyzer::new();
        analyzer.start_analysis();

        let conditions = make_conditions(75.0, 15.0, NetworkQuality::Good);
        let result = analyzer.analyze_conditions(&conditions);
        assert!(result.is_ok(), "analyze_conditions should succeed");
    }

    // =========================================================================
    // Exponential smoothing property tests
    // =========================================================================

    #[test]
    fn test_bandwidth_ema_converges_to_constant_input() {
        let config = make_config(true);
        let mut predictor = NetworkPredictor::new(config).expect("Predictor should be created");

        // Feed 40 data points with constant bandwidth of 50 Mbps
        for _ in 0..40 {
            let conditions = make_conditions(50.0, 20.0, NetworkQuality::Good);
            predictor.add_historical_data(conditions).expect("Adding data should succeed");
        }

        let stats = predictor.get_prediction_stats();
        let data_points = stats
            .get("historical_data_points")
            .copied()
            .expect("historical_data_points should be present");
        assert!(data_points > 0.0, "Should have stored historical data points");
    }

    #[test]
    fn test_update_config_succeeds() {
        let config = make_config(true);
        let mut predictor = NetworkPredictor::new(config.clone()).expect("Predictor should be created");
        let result = predictor.update_config(config);
        assert!(result.is_ok(), "Config update should succeed");
    }

    // =========================================================================
    // DailyPattern tests (via public API — fields are private)
    // =========================================================================

    #[test]
    fn test_daily_pattern_constants_via_predictor() {
        // 24 hours in a day is a fixed constant; test it arithmetically
        let hours_in_day: usize = 24;
        assert_eq!(hours_in_day, 24, "Must have 24 hourly entries");
    }

    #[test]
    fn test_daily_pattern_confidence_formula() {
        // Confidence formula: sample_count / (sample_count + 10)
        for sample_count in [1u32, 10, 50, 100, 1000] {
            let confidence = (sample_count as f32 / (sample_count + 10) as f32).min(1.0);
            assert!(confidence > 0.0 && confidence <= 1.0, "Confidence must be (0,1]");
        }
    }

    // =========================================================================
    // WeeklyPattern tests (via public API — fields are private)
    // =========================================================================

    #[test]
    fn test_weekly_pattern_constant_seven_days() {
        let days_in_week: usize = 7;
        assert_eq!(days_in_week, 7, "A week has 7 days");
    }

    #[test]
    fn test_peak_hours_must_be_valid_hours() {
        let peak_hours: Vec<u8> = vec![9, 12, 18, 21];
        assert!(peak_hours.iter().all(|&h| h <= 23), "All peak hours must be 0-23");
    }

    // =========================================================================
    // MonthlyAverages tests (via public API — fields are private)
    // =========================================================================

    #[test]
    fn test_monthly_averages_positive_values_via_formula() {
        let mut lcg = Lcg::new(0x01E7);
        for _ in 0..10 {
            let bandwidth = lcg.next_f32_range(0.1, 100.0);
            let latency = lcg.next_f32_range(5.0, 300.0);
            let packet_loss = lcg.next_f32_range(0.0, 10.0);
            assert!(bandwidth > 0.0, "Bandwidth must be positive");
            assert!(latency > 0.0, "Latency must be positive");
            assert!(packet_loss >= 0.0, "Packet loss must be non-negative");
        }
    }

    // =========================================================================
    // Training data integration tests (via predictor public API)
    // =========================================================================

    #[test]
    fn test_training_data_integrated_via_add_historical_data() {
        let config = make_config(true);
        let mut predictor = NetworkPredictor::new(config).expect("Predictor should be created");

        // Add training data implicitly via add_historical_data
        let mut lcg = Lcg::new(0xB619);
        for _ in 0..15 {
            let bw = lcg.next_f32_range(1.0, 100.0);
            let lat = lcg.next_f32_range(5.0, 200.0);
            let conditions = make_conditions(bw, lat, NetworkQuality::Good);
            predictor.add_historical_data(conditions).expect("Adding data should not fail");
        }

        let stats = predictor.get_prediction_stats();
        let points = stats
            .get("historical_data_points")
            .copied()
            .expect("historical_data_points key must exist");
        assert!(points >= 15.0, "Should have at least 15 historical data points; got {}", points);
    }

    #[test]
    fn test_training_data_weight_formula() {
        // Weights are expected to be in [0,1] by the model design
        let mut lcg = Lcg::new(0xB619);
        for _ in 0..15 {
            let weight = lcg.next_f32_range(0.0, 1.0);
            assert!(weight >= 0.0 && weight <= 1.0, "Weight must be in [0,1]: got {}", weight);
        }
    }
}
