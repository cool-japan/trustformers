//! Regression tests for the analyzers that replaced the placeholder macro.
//!
//! Every test here would have failed against the pre-0.2.1 code: each one
//! either asserts a value that is derived from the input (the placeholders
//! ignored their input entirely) or asserts a structured refusal where the
//! placeholders returned a fabricated `Ok`.

use std::time::Duration;

use chrono::{Duration as ChronoDuration, Utc};

use super::super::super::types::data_structures::TimestampedMetrics;
use super::super::types::{PerformanceThresholds, QualityThresholds};
use super::{
    AnomalyDetector, CorrelationAnalyzer, DistributionAnalyzer, ForecastingEngine, PatternAnalyzer,
    PerformanceAnalyzer, QualityAnalyzer, TrendAnalyzer,
};

/// Build a window of `n` samples, deriving each metric from its index.
///
/// `latency` and `throughput` are supplied by the caller so an individual test
/// can shape the series it needs. Timestamps are spaced one second apart,
/// ending at "now", so the quality analyzer sees a fresh, regular window.
fn window<L, T>(n: usize, latency: L, throughput: T) -> Vec<TimestampedMetrics>
where
    L: Fn(usize) -> f64,
    T: Fn(usize) -> f64,
{
    let start = Utc::now() - ChronoDuration::seconds(n as i64);
    (0..n)
        .map(|i| {
            let mut sample = TimestampedMetrics::default();
            sample.timestamp = start + ChronoDuration::seconds(i as i64);
            sample.metrics.throughput = throughput(i);
            sample.metrics.latency = Duration::from_secs_f64(latency(i));
            sample.metrics.error_rate = 0.01;
            // Every remaining series is held constant so a test only ever sees
            // the structure it planted in `latency` and `throughput`; a varying
            // filler series would be picked up as a finding of its own.
            sample.metrics.resource_usage.cpu_usage = 0.25;
            sample.metrics.resource_usage.memory_usage = 100.0;
            sample.metrics.resource_usage.io_usage = 0.1;
            sample.metrics.resource_usage.network_usage = 0.05;
            let _ = i;
            sample
        })
        .collect()
}

#[tokio::test]
async fn distribution_analyzer_refuses_a_window_too_small_to_fit() {
    let analyzer = DistributionAnalyzer::new().await.expect("construction never fails");
    // The placeholder returned a full "normal-like" report for any input,
    // including none at all.
    let error = analyzer
        .analyze(&window(3, |i| 0.01 * (i + 1) as f64, |_| 100.0))
        .await
        .expect_err("three samples cannot support a distribution fit");
    let message = error.to_string();
    assert!(message.contains("at least 8"), "{message}");
    assert!(message.contains("latency_seconds"), "{message}");
}

#[tokio::test]
async fn distribution_analyzer_fits_the_supplied_latency_series() {
    let analyzer = DistributionAnalyzer::new().await.expect("construction never fails");
    // A strictly increasing latency series: the fitted mean must land inside
    // the observed range, which the placeholder's canned report could not do.
    let samples = window(40, |i| 0.010 + 0.001 * i as f64, |_| 100.0);
    let result = analyzer.analyze(&samples).await.expect("40 samples is enough to fit");

    assert_eq!(result.series, "latency_seconds");
    assert!(
        !result.distribution_fits.is_empty(),
        "at least the normal family fits"
    );
    let best = result.best_fit.as_ref().expect("a best fit exists when fits do");
    let mean = best
        .parameters
        .get("mean")
        .or_else(|| best.parameters.get("log_mean"))
        .copied()
        .expect("every fitted family reports a location parameter");
    if best.distribution_name == "normal" {
        // Mean of 0.010..0.049 is 0.0295.
        assert!(
            (mean - 0.0295).abs() < 1e-6,
            "fitted mean {mean} must match the samples"
        );
    }
    // The placeholder always claimed a Shapiro-Wilk statistic of 0.95.
    assert!(
        result.normality_assessment.shapiro_wilk.is_none(),
        "Shapiro-Wilk needs coefficient tables this crate does not carry"
    );
    // A uniform ramp is not normal; the D'Agostino test must have run (n >= 20).
    assert!(result.normality_assessment.dagostino.is_some());
    assert!(result.histogram_analysis.optimal_bins >= 1);
    let counted: u64 = result.histogram_analysis.histogram.bin_counts.iter().sum();
    assert_eq!(counted, 40, "every sample lands in exactly one bin");
}

#[tokio::test]
async fn correlation_analyzer_recovers_a_planted_relationship() {
    let analyzer = CorrelationAnalyzer::new().await.expect("construction never fails");
    // throughput rises while latency rises: a perfect positive correlation.
    let samples = window(30, |i| 0.01 * i as f64, |i| 10.0 * i as f64);
    let result = analyzer.analyze(&samples).await.expect("30 samples is enough");

    let key = ("throughput".to_string(), "latency_seconds".to_string());
    let measure = result
        .pairwise_correlations
        .get(&key)
        .expect("both series vary, so the pair is reported");
    assert!(
        (measure.pearson - 1.0).abs() < 1e-9,
        "a linear relationship must give r = 1, got {}",
        measure.pearson
    );
    assert!(measure.p_value < 0.001, "p = {}", measure.p_value);
    // The placeholder hardcoded determinant = 1.0 for an empty matrix; a matrix
    // containing a perfectly collinear pair is singular.
    assert!(
        result.correlation_matrix.determinant.abs() < 1e-6,
        "determinant {} must collapse for collinear series",
        result.correlation_matrix.determinant
    );
    assert!(
        result.dependency_analysis.causal_relationships.is_empty(),
        "correlation is never reported as causation"
    );
}

#[tokio::test]
async fn correlation_analyzer_refuses_a_window_of_constants() {
    let analyzer = CorrelationAnalyzer::new().await.expect("construction never fails");
    let samples = window(20, |_| 0.01, |_| 100.0);
    let error = analyzer
        .analyze(&samples)
        .await
        .expect_err("constant series have undefined correlation");
    assert!(error.to_string().contains("varying"), "{error}");
}

#[tokio::test]
async fn forecasting_engine_scores_models_on_a_holdout() {
    let engine = ForecastingEngine::new().await.expect("construction never fails");
    // A clean ramp: the OLS model should reconstruct it almost exactly.
    let samples = window(40, |_| 0.01, |i| 5.0 + 2.0 * i as f64);
    let result = engine.analyze(&samples).await.expect("40 samples is enough");

    assert!(!result.models.is_empty());
    let best = result.best_model.as_deref().expect("a best model is selected");
    assert_eq!(best, "ordinary_least_squares", "a ramp is exactly linear");
    // The placeholder always reported mae 0.1 / mape 0.05 / confidence 0.85.
    assert!(
        result.accuracy_metrics.mae < 1e-6,
        "an exact fit has near-zero MAE, got {}",
        result.accuracy_metrics.mae
    );
    assert!(result.confidence > 0.99, "confidence {}", result.confidence);
    let model = result
        .models
        .iter()
        .find(|m| m.name == "ordinary_least_squares")
        .expect("just asserted it was selected");
    let slope = model.parameters.get("slope").copied().expect("OLS reports a slope");
    assert!(
        (slope - 2.0).abs() < 1e-9,
        "planted slope is 2.0, got {slope}"
    );
    assert_eq!(model.forecast_points.len(), 10);
}

#[tokio::test]
async fn forecasting_engine_refuses_a_window_it_cannot_split() {
    let engine = ForecastingEngine::new().await.expect("construction never fails");
    let error = engine
        .analyze(&window(6, |_| 0.01, |i| i as f64))
        .await
        .expect_err("six samples leave no holdout tail");
    assert!(error.to_string().contains("at least 12"), "{error}");
}

#[tokio::test]
async fn quality_analyzer_measures_the_window_it_is_given() {
    let thresholds = QualityThresholds::default();
    let analyzer = QualityAnalyzer::new(thresholds).await.expect("construction never fails");
    let samples = window(40, |i| 0.01 * (i + 1) as f64, |i| 100.0 + i as f64);
    let result = analyzer.analyze(&samples).await.expect("a populated window is valid");

    // The placeholder always reported overall_score 0.9 with seven invented
    // dimension scores.
    assert!(
        result.dimensions.accuracy.is_none(),
        "no reference measurement exists to compare against"
    );
    assert!(
        result.dimensions.integrity.is_none(),
        "no referential constraints are declared"
    );
    assert!(
        (result.dimensions.validity - 1.0).abs() < f64::EPSILON,
        "every sample is in-domain"
    );
    assert!(
        (result.dimensions.uniqueness - 1.0).abs() < f64::EPSILON,
        "timestamps are one second apart"
    );
    assert!(
        result.dimensions.consistency > 0.99,
        "a perfectly regular interval, got {}",
        result.dimensions.consistency
    );
    assert!(
        result.issues.is_empty(),
        "a clean window raises no issue: {:?}",
        result.issues
    );
}

#[tokio::test]
async fn quality_analyzer_flags_stale_and_duplicated_samples() {
    let mut thresholds = QualityThresholds::default();
    thresholds.max_staleness = Duration::from_secs(5);
    let analyzer = QualityAnalyzer::new(thresholds).await.expect("construction never fails");

    // Two samples share a timestamp, and all of them predate the threshold.
    let old = Utc::now() - ChronoDuration::seconds(600);
    let samples: Vec<TimestampedMetrics> = (0..8)
        .map(|i| {
            let mut sample = TimestampedMetrics::default();
            sample.timestamp = old + ChronoDuration::seconds((i / 2) as i64);
            sample.metrics.throughput = 100.0;
            sample
        })
        .collect();

    let result = analyzer.analyze(&samples).await.expect("a populated window is valid");
    assert!(
        (result.dimensions.timeliness - 0.0).abs() < f64::EPSILON,
        "every sample is older than the 5s threshold"
    );
    assert!(
        result.dimensions.uniqueness < 1.0,
        "pairs share timestamps, got {}",
        result.dimensions.uniqueness
    );
    let kinds: Vec<String> =
        result.issues.iter().map(|issue| format!("{:?}", issue.issue_type)).collect();
    assert!(
        kinds.iter().any(|kind| kind.contains("StaleData")),
        "{kinds:?}"
    );
    assert!(
        kinds.iter().any(|kind| kind.contains("DuplicateData")),
        "{kinds:?}"
    );
}

#[tokio::test]
async fn pattern_analyzer_detects_a_planted_trend() {
    let analyzer = PatternAnalyzer::new().await.expect("construction never fails");
    let samples = window(30, |_| 0.01, |i| 3.0 * i as f64);
    let result = analyzer.analyze(&samples).await.expect("30 samples is enough");

    // The placeholder always classified the window "normal" with confidence 0.8
    // and reported no patterns at all.
    let trend = result
        .patterns
        .iter()
        .find(|p| p.name == "throughput_trend")
        .expect("a perfect ramp is a trend");
    assert!(
        result.classification.features.contains_key("trending"),
        "the trend must appear in the classification: {:?}",
        result.classification.features
    );
    let slope = trend
        .characteristics
        .get("slope_per_sample")
        .copied()
        .expect("the trend reports its slope");
    assert!(
        (slope - 3.0).abs() < 1e-9,
        "planted slope is 3.0, got {slope}"
    );
    assert!(
        trend.strength > 0.99,
        "R^2 of an exact line: {}",
        trend.strength
    );
}

#[tokio::test]
async fn pattern_analyzer_reports_nothing_for_a_featureless_window() {
    let analyzer = PatternAnalyzer::new().await.expect("construction never fails");
    let samples = window(20, |_| 0.01, |_| 100.0);
    let result = analyzer.analyze(&samples).await.expect("20 samples is enough");
    assert_eq!(result.classification.primary_class, "no_pattern_detected");
    assert!((result.confidence - 0.0).abs() < f64::EPSILON);
}

#[tokio::test]
async fn performance_analyzer_reports_measured_percentiles_and_no_sla() {
    let analyzer = PerformanceAnalyzer::new(PerformanceThresholds::default())
        .await
        .expect("construction never fails");
    // Latency 0.010..0.049 in even steps.
    let samples = window(40, |i| 0.010 + 0.001 * i as f64, |_| 100.0);
    let result = analyzer.analyze(&samples).await.expect("40 samples is enough");

    let stats = &result.metrics_analysis.latency.current_stats;
    assert!(
        (stats.max - 0.049).abs() < 1e-9,
        "observed max, got {}",
        stats.max
    );
    assert!(
        (stats.mean - 0.0295).abs() < 1e-9,
        "observed mean, got {}",
        stats.mean
    );
    // The placeholder reported p95 = 80.0 and p99 = 100.0 for every input.
    assert!(stats.p95 > stats.median && stats.p95 <= stats.max);
    assert!(stats.p99 >= stats.p95);
    // Availability and SLA compliance are not derivable from a metrics window.
    assert!(result.metrics_analysis.availability.is_none());
    assert!(result.metrics_analysis.latency.sla_compliance.is_none());
    // Nor are energy and cost efficiency.
    assert!(result.efficiency_analysis.components.energy_efficiency.is_none());
    assert!(result.efficiency_analysis.components.cost_efficiency.is_none());
    assert!(
        result.optimization_opportunities.is_empty(),
        "ROI and effort are not observable from metrics"
    );
}

#[tokio::test]
async fn anomaly_detector_flags_only_the_planted_outlier() {
    let detector = AnomalyDetector::new().await.expect("construction never fails");
    // Latency alternates tightly around 10ms, giving a non-zero MAD, and one
    // sample is planted far outside that band.
    let mut samples = window(40, |i| 0.010 + 0.001 * (i % 4) as f64, |_| 100.0);
    if let Some(sample) = samples.get_mut(20) {
        sample.metrics.latency = Duration::from_secs_f64(5.0);
    }
    let result = detector.analyze(&samples).await.expect("40 samples is enough");

    // The placeholder reported anomaly_rate 0.02 and detection_confidence 0.9
    // for every input, with an empty anomaly list.
    assert_eq!(result.anomalies.len(), 1, "exactly one reading was planted");
    let anomaly = result.anomalies.first().expect("just asserted one");
    assert!(
        anomaly.affected_metrics.contains(&"latency_seconds".to_string()),
        "{:?}",
        anomaly.affected_metrics
    );
    assert!((result.anomaly_rate - 1.0 / 40.0).abs() < 1e-9);
    assert!(
        result.baseline_performance.is_none(),
        "no labelled ground truth exists to score a classifier against"
    );
}

#[tokio::test]
async fn anomaly_detector_refuses_a_window_with_no_spread() {
    let detector = AnomalyDetector::new().await.expect("construction never fails");
    let samples = window(20, |_| 0.01, |_| 100.0);
    let error = detector
        .analyze(&samples)
        .await
        .expect_err("a constant window has a zero MAD in every series");
    assert!(
        error.to_string().contains("median absolute deviation"),
        "{error}"
    );
}

#[tokio::test]
async fn trend_analyzer_recovers_the_planted_slope() {
    let analyzer = TrendAnalyzer::new().await.expect("construction never fails");
    let samples = window(30, |_| 0.01, |i| 7.0 * i as f64);
    let result = analyzer.analyze(&samples).await.expect("30 samples is enough");

    // The placeholder reported overall_trend Stable, trend_strength 0.5 and
    // trend_significance 0.8 for every input.
    let throughput = result
        .trends
        .iter()
        .find(|t| t.metric == "throughput")
        .expect("throughput varies, so it is trended");
    assert!(
        (throughput.slope - 7.0).abs() < 1e-9,
        "planted slope, got {}",
        throughput.slope
    );
    assert!(
        throughput.strength > 0.99,
        "R^2 of an exact line: {}",
        throughput.strength
    );
    assert_eq!(throughput.data_points.len(), 30, "one point per sample");
    assert!(!result.forecasts.is_empty());
}

#[tokio::test]
async fn every_analyzer_refuses_after_shutdown() {
    let samples = window(40, |i| 0.010 + 0.001 * i as f64, |i| 100.0 + i as f64);

    let distribution = DistributionAnalyzer::new().await.expect("construction never fails");
    distribution.shutdown().await.expect("shutdown never fails");
    assert!(distribution.is_shut_down());
    assert!(distribution.analyze(&samples).await.is_err());

    let correlation = CorrelationAnalyzer::new().await.expect("construction never fails");
    correlation.shutdown().await.expect("shutdown never fails");
    assert!(correlation.analyze(&samples).await.is_err());

    let forecasting = ForecastingEngine::new().await.expect("construction never fails");
    forecasting.shutdown().await.expect("shutdown never fails");
    assert!(forecasting.analyze(&samples).await.is_err());

    let quality = QualityAnalyzer::new(QualityThresholds::default())
        .await
        .expect("construction never fails");
    quality.shutdown().await.expect("shutdown never fails");
    assert!(quality.analyze(&samples).await.is_err());

    let pattern = PatternAnalyzer::new().await.expect("construction never fails");
    pattern.shutdown().await.expect("shutdown never fails");
    assert!(pattern.analyze(&samples).await.is_err());

    let performance = PerformanceAnalyzer::new(PerformanceThresholds::default())
        .await
        .expect("construction never fails");
    performance.shutdown().await.expect("shutdown never fails");
    assert!(performance.analyze(&samples).await.is_err());

    let anomaly = AnomalyDetector::new().await.expect("construction never fails");
    anomaly.shutdown().await.expect("shutdown never fails");
    assert!(anomaly.analyze(&samples).await.is_err());

    let trend = TrendAnalyzer::new().await.expect("construction never fails");
    trend.shutdown().await.expect("shutdown never fails");
    assert!(trend.analyze(&samples).await.is_err());
}
