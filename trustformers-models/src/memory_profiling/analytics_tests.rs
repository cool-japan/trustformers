#[cfg(test)]
mod tests {
    use crate::memory_profiling::analytics::*;
    use crate::memory_profiling::types::MemoryMetrics;
    use std::time::SystemTime;

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

        fn next_f64(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn make_metrics(rng: &mut Lcg, count: usize) -> Vec<MemoryMetrics> {
        (0..count)
            .map(|_| MemoryMetrics {
                timestamp: SystemTime::now(),
                total_memory_mb: 100.0 + rng.next_f64() * 200.0,
                heap_memory_mb: 80.0 + rng.next_f64() * 100.0,
                stack_memory_mb: 2.0 + rng.next_f64() * 5.0,
                gpu_memory_mb: None,
                peak_memory_mb: 300.0,
                allocated_objects: rng.next() % 10000,
                deallocated_objects: rng.next() % 10000,
                active_allocations: rng.next() % 5000,
                memory_fragmentation_ratio: rng.next_f64() * 0.5,
                gc_collections: rng.next() % 100,
                gc_time_ms: rng.next_f64() * 500.0,
                memory_growth_rate_mb_per_sec: rng.next_f64() * 20.0 - 5.0,
            })
            .collect()
    }

    // --- AlertRecommendations tests ---

    #[test]
    fn test_alert_recommendations_new() {
        let recs = AlertRecommendations::new();
        assert!(!recs.high_memory.is_empty());
        assert!(!recs.rapid_growth.is_empty());
        assert!(!recs.fragmentation.is_empty());
        assert!(!recs.memory_leak.is_empty());
        assert!(!recs.gc_pressure.is_empty());
    }

    #[test]
    fn test_alert_recommendations_default() {
        let recs = AlertRecommendations::default();
        assert!(!recs.high_memory.is_empty());
    }

    #[test]
    fn test_alert_recommendations_clone() {
        let recs = AlertRecommendations::new();
        let cloned = recs.clone();
        assert_eq!(recs.high_memory.len(), cloned.high_memory.len());
    }

    // --- AdaptiveThresholds tests ---

    #[test]
    fn test_adaptive_thresholds_default() {
        let thresholds = AdaptiveThresholds::default();
        assert!(thresholds.base_memory_threshold > 0.0);
        assert!(thresholds.growth_rate_threshold > 0.0);
        assert!(thresholds.fragmentation_threshold > 0.0);
        assert!(thresholds.adaptation_factor > 0.0);
    }

    #[test]
    fn test_adaptive_thresholds_update_empty_metrics() {
        let mut thresholds = AdaptiveThresholds::default();
        let original_base = thresholds.base_memory_threshold;
        thresholds.update_thresholds(&[]);
        assert!((thresholds.base_memory_threshold - original_base).abs() < f64::EPSILON);
    }

    #[test]
    fn test_adaptive_thresholds_update_with_data() {
        let mut thresholds = AdaptiveThresholds::default();
        let mut rng = Lcg::new(42);
        let metrics = make_metrics(&mut rng, 20);
        let original_base = thresholds.base_memory_threshold;
        thresholds.update_thresholds(&metrics);
        // Threshold should change after update
        assert!((thresholds.base_memory_threshold - original_base).abs() > 0.0);
    }

    #[test]
    fn test_adaptive_thresholds_clone() {
        let thresholds = AdaptiveThresholds::default();
        let cloned = thresholds.clone();
        assert!(
            (thresholds.base_memory_threshold - cloned.base_memory_threshold).abs() < f64::EPSILON
        );
    }

    // --- LeakDetectionHeuristics tests ---

    #[test]
    fn test_leak_detection_default() {
        let heuristics = LeakDetectionHeuristics::default();
        assert!(heuristics.sustained_growth_threshold > 0.0);
        assert!(heuristics.allocation_pattern_threshold > 0);
        assert!(heuristics.false_positive_filter > 0.0);
    }

    #[test]
    fn test_leak_detection_insufficient_data() {
        let heuristics = LeakDetectionHeuristics::default();
        let mut rng = Lcg::new(10);
        let metrics = make_metrics(&mut rng, 5); // Less than 10
        let alerts = heuristics.detect_potential_leaks(&metrics);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_leak_detection_no_leak() {
        let heuristics = LeakDetectionHeuristics::default();
        let mut rng = Lcg::new(20);
        let metrics: Vec<MemoryMetrics> = (0..100)
            .map(|_| MemoryMetrics {
                timestamp: SystemTime::now(),
                total_memory_mb: 100.0 + rng.next_f64() * 10.0,
                heap_memory_mb: 80.0,
                stack_memory_mb: 2.0,
                gpu_memory_mb: None,
                peak_memory_mb: 120.0,
                allocated_objects: 500,
                deallocated_objects: 500,
                active_allocations: 100,
                memory_fragmentation_ratio: 0.1,
                gc_collections: 5,
                gc_time_ms: 10.0,
                memory_growth_rate_mb_per_sec: 0.5, // Low growth
            })
            .collect();
        let alerts = heuristics.detect_potential_leaks(&metrics);
        // Should not detect sustained growth at low rates
        let sustained_alerts: Vec<_> = alerts
            .iter()
            .filter(|a| matches!(a.alert_type, LeakAlertType::SustainedGrowth))
            .collect();
        assert!(sustained_alerts.is_empty());
    }

    #[test]
    fn test_leak_detection_with_sustained_growth() {
        let heuristics = LeakDetectionHeuristics::default();
        let metrics: Vec<MemoryMetrics> = (0..100)
            .map(|_| MemoryMetrics {
                timestamp: SystemTime::now(),
                total_memory_mb: 500.0,
                heap_memory_mb: 400.0,
                stack_memory_mb: 2.0,
                gpu_memory_mb: None,
                peak_memory_mb: 500.0,
                allocated_objects: 10000,
                deallocated_objects: 5000,
                active_allocations: 5000,
                memory_fragmentation_ratio: 0.1,
                gc_collections: 5,
                gc_time_ms: 10.0,
                memory_growth_rate_mb_per_sec: 50.0, // High growth
            })
            .collect();
        let alerts = heuristics.detect_potential_leaks(&metrics);
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_leak_detection_allocation_imbalance() {
        let heuristics = LeakDetectionHeuristics::default();
        let metrics: Vec<MemoryMetrics> = (0..100)
            .map(|_| MemoryMetrics {
                timestamp: SystemTime::now(),
                total_memory_mb: 200.0,
                heap_memory_mb: 180.0,
                stack_memory_mb: 2.0,
                gpu_memory_mb: None,
                peak_memory_mb: 250.0,
                allocated_objects: 20000,
                deallocated_objects: 5000, // Much less than allocated
                active_allocations: 15000,
                memory_fragmentation_ratio: 0.1,
                gc_collections: 5,
                gc_time_ms: 10.0,
                memory_growth_rate_mb_per_sec: 1.0,
            })
            .collect();
        let alerts = heuristics.detect_potential_leaks(&metrics);
        let imbalance_alerts: Vec<_> = alerts
            .iter()
            .filter(|a| matches!(a.alert_type, LeakAlertType::AllocationImbalance))
            .collect();
        assert!(!imbalance_alerts.is_empty());
    }

    // --- MemoryPredictor tests ---

    #[test]
    fn test_memory_predictor_default() {
        let _predictor = MemoryPredictor::default();
        // MemoryPredictor created with defaults
    }

    #[test]
    fn test_memory_predictor_insufficient_data() {
        let mut predictor = MemoryPredictor::default();
        let mut rng = Lcg::new(30);
        let metrics = make_metrics(&mut rng, 10); // Less than trend_window
        let prediction = predictor.predict_memory_usage(&metrics, None);
        assert!(prediction.is_none());
    }

    #[test]
    fn test_memory_predictor_with_clear_trend() {
        let mut predictor = MemoryPredictor::default();
        let metrics: Vec<MemoryMetrics> = (0..100)
            .map(|i| MemoryMetrics {
                timestamp: SystemTime::now(),
                total_memory_mb: 100.0 + (i as f64) * 2.0, // Clear linear trend
                heap_memory_mb: 80.0 + (i as f64) * 1.5,
                stack_memory_mb: 2.0,
                gpu_memory_mb: None,
                peak_memory_mb: 300.0,
                allocated_objects: 1000,
                deallocated_objects: 900,
                active_allocations: 100,
                memory_fragmentation_ratio: 0.1,
                gc_collections: 5,
                gc_time_ms: 10.0,
                memory_growth_rate_mb_per_sec: 2.0,
            })
            .collect();
        let prediction = predictor.predict_memory_usage(&metrics, None);
        assert!(prediction.is_some());
        if let Some(pred) = prediction {
            assert!(pred.predicted_memory_mb > 0.0);
            assert!(pred.confidence > 0.0);
            assert!(pred.trend_slope > 0.0);
        }
    }

    #[test]
    fn test_memory_predictor_custom_horizon() {
        let mut predictor = MemoryPredictor::default();
        let metrics: Vec<MemoryMetrics> = (0..100)
            .map(|i| MemoryMetrics {
                timestamp: SystemTime::now(),
                total_memory_mb: 100.0 + (i as f64) * 1.0,
                heap_memory_mb: 80.0,
                stack_memory_mb: 2.0,
                gpu_memory_mb: None,
                peak_memory_mb: 250.0,
                allocated_objects: 1000,
                deallocated_objects: 1000,
                active_allocations: 100,
                memory_fragmentation_ratio: 0.1,
                gc_collections: 5,
                gc_time_ms: 10.0,
                memory_growth_rate_mb_per_sec: 1.0,
            })
            .collect();
        let prediction = predictor.predict_memory_usage(&metrics, Some(600));
        assert!(prediction.is_some());
        if let Some(pred) = prediction {
            assert_eq!(pred.horizon_secs, 600);
        }
    }

    // --- StatisticalAnalyzer tests ---

    #[test]
    fn test_statistical_analyzer_creation() {
        let _analyzer = StatisticalAnalyzer::new(0.95);
        // StatisticalAnalyzer created successfully
    }

    #[test]
    fn test_statistical_analyzer_empty_metrics() {
        let analyzer = StatisticalAnalyzer::new(0.95);
        let stats = analyzer.calculate_usage_statistics(&[]);
        assert!((stats.mean - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_statistical_analyzer_basic_statistics() {
        let analyzer = StatisticalAnalyzer::new(0.95);
        let metrics: Vec<MemoryMetrics> = (0..50)
            .map(|i| MemoryMetrics {
                timestamp: SystemTime::now(),
                total_memory_mb: 100.0 + (i as f64),
                heap_memory_mb: 80.0,
                stack_memory_mb: 2.0,
                gpu_memory_mb: None,
                peak_memory_mb: 200.0,
                allocated_objects: 1000,
                deallocated_objects: 1000,
                active_allocations: 100,
                memory_fragmentation_ratio: 0.1,
                gc_collections: 5,
                gc_time_ms: 10.0,
                memory_growth_rate_mb_per_sec: 1.0,
            })
            .collect();
        let stats = analyzer.calculate_usage_statistics(&metrics);
        assert!(stats.mean > 100.0);
        assert!(stats.median > 100.0);
        assert!(stats.std_dev > 0.0);
        assert!(stats.min >= 100.0);
        assert!(stats.max <= 150.0);
    }

    #[test]
    fn test_statistical_analyzer_detect_anomalies_insufficient_data() {
        let analyzer = StatisticalAnalyzer::new(0.95);
        let mut rng = Lcg::new(50);
        let metrics = make_metrics(&mut rng, 5);
        let anomalies = analyzer.detect_anomalies(&metrics);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_statistical_analyzer_detect_spike() {
        let analyzer = StatisticalAnalyzer::new(0.95);
        let mut metrics: Vec<MemoryMetrics> = (0..50)
            .map(|_| MemoryMetrics {
                timestamp: SystemTime::now(),
                total_memory_mb: 100.0,
                heap_memory_mb: 80.0,
                stack_memory_mb: 2.0,
                gpu_memory_mb: None,
                peak_memory_mb: 200.0,
                allocated_objects: 1000,
                deallocated_objects: 1000,
                active_allocations: 100,
                memory_fragmentation_ratio: 0.1,
                gc_collections: 5,
                gc_time_ms: 10.0,
                memory_growth_rate_mb_per_sec: 0.0,
            })
            .collect();
        // Add a large spike
        metrics[25].total_memory_mb = 10000.0;
        let anomalies = analyzer.detect_anomalies(&metrics);
        let spikes: Vec<_> = anomalies
            .iter()
            .filter(|a| a.anomaly_type == AnomalyType::SuddenSpike)
            .collect();
        assert!(!spikes.is_empty());
    }

    // --- MemoryStatistics tests ---

    #[test]
    fn test_memory_statistics_default() {
        let stats = MemoryStatistics::default();
        assert!((stats.mean - 0.0).abs() < f64::EPSILON);
        assert!((stats.std_dev - 0.0).abs() < f64::EPSILON);
        assert_eq!(stats.outlier_count, 0);
    }

    // --- LinearRegression tests ---

    #[test]
    fn test_linear_regression_fields() {
        let reg = LinearRegression {
            slope: 1.5,
            intercept: 10.0,
            correlation: 0.99,
            last_computed: SystemTime::now(),
        };
        assert!((reg.slope - 1.5).abs() < f64::EPSILON);
        assert!((reg.intercept - 10.0).abs() < f64::EPSILON);
        assert!((reg.correlation - 0.99).abs() < f64::EPSILON);
    }

    // --- MemoryPrediction tests ---

    #[test]
    fn test_memory_prediction_fields() {
        let pred = MemoryPrediction {
            predicted_memory_mb: 500.0,
            confidence: 0.95,
            horizon_secs: 300,
            trend_slope: 2.5,
        };
        assert!((pred.predicted_memory_mb - 500.0).abs() < f64::EPSILON);
        assert_eq!(pred.horizon_secs, 300);
    }
}
