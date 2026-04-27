//! Tests for resource analyzer types

use super::types::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};

struct Lcg { state: u64 }
impl Lcg {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f64(&mut self) -> f64 { (self.next_u64() % 100000) as f64 / 100000.0 }
}

#[test]
fn test_performance_based_strategy_new() { let _ = PerformanceBasedStrategy::new(); }

#[test]
fn test_hybrid_selection_strategy_new() { let _ = HybridSelectionStrategy::new(); }

#[test]
fn test_characteristic_based_strategy_new() { let _ = CharacteristicBasedStrategy::new(); }

#[test]
fn test_peak_intensity_algorithm_new() { let _ = PeakIntensityAlgorithm::new(); }

#[test]
fn test_mean_intensity_algorithm_new() { let _ = MeanIntensityAlgorithm::new(); }

#[test]
fn test_exponential_intensity_algorithm_new() {
    let algo = ExponentialIntensityAlgorithm::new();
    let _ = format!("{:?}", algo);
}

#[test]
fn test_weighted_intensity_algorithm_new() { let _ = WeightedIntensityAlgorithm::new(); }

#[test]
fn test_adaptive_intensity_algorithm_new() { let _ = AdaptiveIntensityAlgorithm::new(); }

#[test]
fn test_collection_statistics_new() {
    let stats = CollectionStatistics::new();
    assert_eq!(stats.total_snapshots.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(stats.failed_collections.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn test_analysis_statistics_new() {
    let stats = AnalysisStatistics::new();
    let summary = stats.get_statistics();
    assert_eq!(summary.total_analyses, 0);
    assert_eq!(summary.failed_analyses, 0);
}

#[test]
fn test_statistics_summary_fields() {
    let summary = StatisticsSummary {
        total_analyses: 100,
        successful_analyses: 95,
        failed_analyses: 5,
        success_rate: 0.95,
        cache_hit_rate: 0.75,
    };
    assert_eq!(summary.successful_analyses + summary.failed_analyses, summary.total_analyses);
}

#[test]
fn test_resource_intensity_has_default() {
    let ri: crate::performance_optimizer::test_characterization::types::ResourceIntensity = Default::default();
    assert!((ri.io_intensity - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_collection_statistics_zero_initial() {
    let stats = CollectionStatistics::new();
    assert_eq!(stats.collection_started.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn test_collection_statistics_zero_failed() {
    let stats = CollectionStatistics::new();
    assert_eq!(stats.collection_stopped.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn test_cached_calculation_validity() {
    let calc = CachedCalculation {
        result: Default::default(),
        algorithm_id: "mean".to_string(),
        calculated_at: Instant::now(),
        calculation_duration: Duration::from_millis(10),
    };
    assert!(calc.is_valid());
}

#[test]
fn test_selection_preferences_creation() {
    let prefs = SelectionPreferences {
        primary_strategy: "performance".to_string(),
        total_selections: 100,
        algorithm_usage: HashMap::new(),
        quality_threshold: 0.85,
        performance_threshold: Duration::from_secs(5),
    };
    assert_eq!(prefs.total_selections, 100);
}

#[test]
fn test_lcg_deterministic() {
    let mut a = Lcg::new(42);
    let mut b = Lcg::new(42);
    for _ in 0..50 { assert_eq!(a.next_u64(), b.next_u64()); }
}

#[test]
fn test_lcg_f64_range() {
    let mut lcg = Lcg::new(42);
    for _ in 0..100 {
        let v = lcg.next_f64();
        assert!(v >= 0.0 && v < 1.0);
    }
}

#[test]
fn test_resource_intensity_default() {
    let ri: crate::performance_optimizer::test_characterization::types::ResourceIntensity = Default::default();
    assert!((ri.cpu_intensity - 0.0).abs() < f64::EPSILON);
    assert!((ri.memory_intensity - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_pid_sample_rate_algorithm() {
    let _ = crate::performance_optimizer::real_time_metrics::collector::PidSampleRateAlgorithm::new();
}

#[test]
fn test_impact_recommendation_engine() {
    let _ = crate::performance_optimizer::real_time_metrics::collector::ImpactRecommendationEngine::new();
}

#[test]
fn test_statistics_summary_success_rate_bounds() {
    let summary = StatisticsSummary {
        total_analyses: 200,
        successful_analyses: 190,
        failed_analyses: 10,
        success_rate: 0.95,
        cache_hit_rate: 0.8,
    };
    assert!(summary.success_rate >= 0.0 && summary.success_rate <= 1.0);
    assert!(summary.cache_hit_rate >= 0.0 && summary.cache_hit_rate <= 1.0);
}

#[test]
fn test_multiple_statistics_summaries() {
    for total in [10u64, 50, 100, 500] {
        let failed = total / 10;
        let summary = StatisticsSummary {
            total_analyses: total,
            successful_analyses: total - failed,
            failed_analyses: failed,
            success_rate: (total - failed) as f64 / total as f64,
            cache_hit_rate: 0.5,
        };
        assert!(summary.success_rate > 0.0 && summary.success_rate <= 1.0);
    }
}
