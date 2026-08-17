//! Tests for [`super`]'s streaming quality analyzer.
//!
//! Split out of `quality_analyzer.rs` to keep that file under the workspace's
//! 2000-line limit.

use super::*;

// -----------------------------------------------------------------------
// Regression: three sub-analyzers used to return a constant 0.9 while
// ignoring their chunk argument, so part of the reported quality score was
// fixed. These assert the surviving analyses actually read the chunk.
// -----------------------------------------------------------------------

#[tokio::test]
async fn content_flow_depends_on_the_chunk() {
    let analyzer = QualityAnalyzer::new();
    let clean = make_chunk("and the result follows.", 0.5);
    let ragged = make_chunk("xyzzy", 0.5);
    let clean_score = analyzer.analyze_content_flow(&clean).await;
    let ragged_score = analyzer.analyze_content_flow(&ragged).await;
    assert!(
        (clean_score - ragged_score).abs() > f32::EPSILON,
        "content flow must vary with the chunk: {clean_score} vs {ragged_score}"
    );
    assert_ne!(clean_score, 0.9, "0.9 was the old constant");
}

#[tokio::test]
async fn context_consistency_is_absent_without_history() {
    let analyzer = QualityAnalyzer::new();
    let chunk = make_chunk("streaming quality analysis chunk", 0.5);
    assert!(
        analyzer.analyze_context_consistency(&chunk).await.is_none(),
        "with no history there is nothing to compare against, so no score may be reported"
    );

    analyzer.remember_chunk(&chunk).await;
    let related = make_chunk(&chunk.content, 0.5);
    let score = analyzer
        .analyze_context_consistency(&related)
        .await
        .expect("identical content shares vocabulary with the history");
    assert!(score > 0.75, "full overlap should score above the floor");
}
use std::time::Duration;

fn make_chunk(content: &str, complexity: f32) -> StreamChunk {
    StreamChunk {
        content: content.to_string(),
        index: 0,
        chunk_type: ChunkType::Content,
        timing: ChunkTiming::default(),
        metadata: ChunkMetadata::with_complexity(complexity),
    }
}

// --- StreamingQuality tests ---

#[test]
fn test_streaming_quality_default_values_in_range() {
    let quality = StreamingQuality::default();
    assert!(
        quality.smoothness >= 0.0 && quality.smoothness <= 1.0,
        "smoothness must be in [0.0, 1.0]"
    );
    assert!(
        quality.naturalness >= 0.0 && quality.naturalness <= 1.0,
        "naturalness must be in [0.0, 1.0]"
    );
    assert!(
        quality.responsiveness >= 0.0 && quality.responsiveness <= 1.0,
        "responsiveness must be in [0.0, 1.0]"
    );
    assert!(
        quality.coherence >= 0.0 && quality.coherence <= 1.0,
        "coherence must be in [0.0, 1.0]"
    );
    assert!(
        quality.overall_quality >= 0.0 && quality.overall_quality <= 1.0,
        "overall_quality must be in [0.0, 1.0]"
    );
}

#[test]
fn test_streaming_quality_default_chunk_consistency_in_range() {
    let quality = StreamingQuality::default();
    assert!(quality.chunk_consistency >= 0.0 && quality.chunk_consistency <= 1.0);
    assert!(quality.flow_smoothness >= 0.0 && quality.flow_smoothness <= 1.0);
    assert!(quality.timing_accuracy >= 0.0 && quality.timing_accuracy <= 1.0);
    assert!(quality.buffer_efficiency >= 0.0 && quality.buffer_efficiency <= 1.0);
}

// --- QualityThresholds tests ---

#[test]
fn test_quality_thresholds_default_min_overall_quality_in_range() {
    let thresholds = QualityThresholds::default();
    assert!(
        thresholds.min_overall_quality >= 0.0 && thresholds.min_overall_quality <= 1.0,
        "min_overall_quality must be in [0.0, 1.0]"
    );
}

#[test]
fn test_quality_thresholds_minimum_acceptable_less_than_target() {
    let thresholds = QualityThresholds::default();
    assert!(
        thresholds.minimum_acceptable <= thresholds.target_quality,
        "minimum_acceptable ({}) should be <= target_quality ({})",
        thresholds.minimum_acceptable,
        thresholds.target_quality
    );
}

#[test]
fn test_quality_thresholds_target_less_than_excellent() {
    let thresholds = QualityThresholds::default();
    assert!(
        thresholds.target_quality <= thresholds.excellent_threshold,
        "target_quality ({}) should be <= excellent_threshold ({})",
        thresholds.target_quality,
        thresholds.excellent_threshold
    );
}

#[test]
fn test_quality_thresholds_max_latency_positive() {
    let thresholds = QualityThresholds::default();
    assert!(
        thresholds.max_latency_ms > 0.0,
        "max_latency_ms should be positive, got {}",
        thresholds.max_latency_ms
    );
}

// --- QualityAnalyzer tests ---

#[test]
fn test_quality_analyzer_new() {
    let analyzer = QualityAnalyzer::new();
    assert_eq!(
        analyzer.window_size(),
        100,
        "default window size should be 100"
    );
}

#[test]
fn test_quality_analyzer_default() {
    let analyzer = QualityAnalyzer::default();
    assert_eq!(analyzer.window_size(), 100);
}

#[tokio::test]
async fn test_quality_analyzer_analyze_chunk_quality_returns_measurement() {
    let analyzer = QualityAnalyzer::new();
    let chunk = make_chunk("Hello world, this is a test sentence.", 0.5);
    let delivery_time = Duration::from_millis(80);
    let measurement = analyzer.analyze_chunk_quality(&chunk, delivery_time).await;
    assert!(
        measurement.smoothness >= 0.0 && measurement.smoothness <= 1.0,
        "smoothness must be in [0.0, 1.0]"
    );
    assert!(
        measurement.naturalness >= 0.0 && measurement.naturalness <= 1.0,
        "naturalness must be in [0.0, 1.0]"
    );
    assert!(
        measurement.responsiveness >= 0.0 && measurement.responsiveness <= 1.0,
        "responsiveness must be in [0.0, 1.0]"
    );
    assert!(
        measurement.coherence >= 0.0 && measurement.coherence <= 1.0,
        "coherence must be in [0.0, 1.0]"
    );
}

#[tokio::test]
async fn test_quality_analyzer_score_normalized() {
    let analyzer = QualityAnalyzer::new();
    let chunk = make_chunk("Test content for score normalization check.", 0.4);
    let delivery_time = Duration::from_millis(50);
    let measurement = analyzer.analyze_chunk_quality(&chunk, delivery_time).await;
    assert!(
        measurement.score >= 0.0 && measurement.score <= 1.0,
        "overall score must be normalized in [0.0, 1.0], got {}",
        measurement.score
    );
}

#[tokio::test]
async fn test_quality_analyzer_high_latency_lowers_responsiveness() {
    let analyzer = QualityAnalyzer::new();
    let chunk = make_chunk("Test content for latency measurement.", 0.5);
    let fast_delivery = Duration::from_millis(50);
    let slow_delivery = Duration::from_millis(500);
    let fast_measurement = analyzer.analyze_chunk_quality(&chunk, fast_delivery).await;
    let slow_measurement = analyzer.analyze_chunk_quality(&chunk, slow_delivery).await;
    assert!(
        fast_measurement.responsiveness >= slow_measurement.responsiveness,
        "faster delivery should produce >= responsiveness score: {} >= {}",
        fast_measurement.responsiveness,
        slow_measurement.responsiveness
    );
}

#[tokio::test]
async fn test_quality_analyzer_calculate_overall_quality_bounded() {
    let analyzer = QualityAnalyzer::new();
    // Populate with some measurements
    let chunk = make_chunk("Some streaming content for quality check.", 0.5);
    for _ in 0..5 {
        analyzer.analyze_chunk_quality(&chunk, Duration::from_millis(100)).await;
    }
    let quality = analyzer.calculate_overall_quality().await;
    assert!(
        quality.overall_quality >= 0.0 && quality.overall_quality <= 1.0,
        "overall_quality must be in [0.0, 1.0]"
    );
    assert!(quality.smoothness >= 0.0 && quality.smoothness <= 1.0);
}

#[tokio::test]
async fn test_quality_analyzer_meets_quality_thresholds_initially() {
    let analyzer = QualityAnalyzer::new();
    // Empty window - should meet thresholds with default quality
    let meets = analyzer.meets_quality_thresholds().await;
    // No strict assertion on value - just verify no panic
    let _ = meets;
}

#[tokio::test]
async fn test_quality_analyzer_quality_trends_stable_initially() {
    let analyzer = QualityAnalyzer::new();
    let trends = analyzer.get_quality_trends().await;
    // Without measurements, overall trend should be Stable
    assert_eq!(
        std::mem::discriminant(&trends.overall_trend),
        std::mem::discriminant(&TrendDirection::Stable),
        "initial trend should be Stable"
    );
}

#[tokio::test]
async fn test_quality_analyzer_get_optimization_recommendations() {
    let analyzer = QualityAnalyzer::new();
    let recommendations = analyzer.generate_optimization_recommendations().await;
    // Should return a Vec (possibly empty) without panicking
    let _ = recommendations.len();
}

#[tokio::test]
async fn test_quality_analyzer_accumulates_window() {
    let analyzer = QualityAnalyzer::new();
    let chunk = make_chunk("Accumulate measurements in the quality window", 0.5);
    for i in 0..10 {
        let delivery_time = Duration::from_millis(50 + i * 10);
        analyzer.analyze_chunk_quality(&chunk, delivery_time).await;
    }
    let window = analyzer.metrics_window().read().await;
    assert_eq!(
        window.len(),
        10,
        "window should contain exactly 10 measurements"
    );
}

#[tokio::test]
async fn test_quality_analyzer_window_respects_capacity() {
    let analyzer = QualityAnalyzer::new();
    let chunk = make_chunk("Testing window capacity limit", 0.5);
    // Push more than window_size (100) measurements
    for i in 0..150u64 {
        let delivery_time = Duration::from_millis(50 + i % 100);
        analyzer.analyze_chunk_quality(&chunk, delivery_time).await;
    }
    let window = analyzer.metrics_window().read().await;
    assert!(
        window.len() <= analyzer.window_size(),
        "window should not exceed capacity: {} <= {}",
        window.len(),
        analyzer.window_size()
    );
}

// --- PerceptualQuality tests ---

#[test]
fn test_perceptual_quality_default_values_in_range() {
    let pq = PerceptualQuality::default();
    assert!(pq.fluency >= 0.0 && pq.fluency <= 1.0);
    assert!(pq.engagement >= 0.0 && pq.engagement <= 1.0);
    assert!(pq.clarity >= 0.0 && pq.clarity <= 1.0);
    assert!(pq.user_experience_score >= 0.0 && pq.user_experience_score <= 1.0);
    assert!(pq.cognitive_load >= 0.0 && pq.cognitive_load <= 1.0);
}

// --- StatisticalAnalysis tests ---

#[test]
fn test_statistical_analysis_default_chunk_count_zero() {
    let stats = StatisticalAnalysis::default();
    assert_eq!(stats.chunk_count, 0);
    assert_eq!(stats.total_characters, 0);
}

// --- DegradationIndicators tests ---

#[test]
fn test_degradation_indicators_default_not_degrading() {
    let indicators = DegradationIndicators::default();
    assert!(!indicators.is_degrading, "default should not be degrading");
    assert_eq!(indicators.degradation_rate, 0.0);
    assert!(indicators.time_to_threshold_breach.is_none());
}

// --- Quality trait delegation tests ---

#[tokio::test]
async fn test_quality_analysis_trait_analyze_quality() {
    let analyzer = QualityAnalyzer::new();
    let chunk = make_chunk("Trait delegation test content", 0.5);
    let measurement = analyzer.analyze_quality(&chunk, Duration::from_millis(75)).await;
    assert!(measurement.score >= 0.0 && measurement.score <= 1.0);
}

#[tokio::test]
async fn test_quality_analysis_trait_get_overall_quality() {
    let analyzer = QualityAnalyzer::new();
    let quality = analyzer.get_overall_quality().await;
    assert!(quality.overall_quality >= 0.0 && quality.overall_quality <= 1.0);
}

#[tokio::test]
async fn test_quality_analysis_trait_meets_thresholds() {
    let analyzer = QualityAnalyzer::new();
    let result = analyzer.meets_thresholds().await;
    let _ = result; // no panic
}

#[tokio::test]
async fn test_quality_analysis_trait_get_trends() {
    let analyzer = QualityAnalyzer::new();
    let trends = analyzer.get_trends().await;
    let _ = trends; // no panic, verify all sub-trends accessible
}

#[tokio::test]
async fn test_quality_analysis_trait_get_recommendations() {
    let analyzer = QualityAnalyzer::new();
    let recs = analyzer.get_recommendations().await;
    let _ = recs.len();
}

// --- Score normalization edge cases ---

#[tokio::test]
async fn test_quality_score_very_slow_delivery_still_bounded() {
    let analyzer = QualityAnalyzer::new();
    let chunk = make_chunk("Edge case: extremely slow delivery", 0.9);
    // 10 second delivery
    let measurement = analyzer.analyze_chunk_quality(&chunk, Duration::from_secs(10)).await;
    assert!(
        measurement.responsiveness >= 0.0 && measurement.responsiveness <= 1.0,
        "responsiveness must stay in [0.0, 1.0] even with very slow delivery"
    );
    assert!(
        measurement.score >= 0.0 && measurement.score <= 1.0,
        "overall score must stay in [0.0, 1.0] even with very slow delivery"
    );
}

#[tokio::test]
async fn test_quality_score_instant_delivery_bounded() {
    let analyzer = QualityAnalyzer::new();
    let chunk = make_chunk("Edge case: instant delivery", 0.1);
    let measurement = analyzer.analyze_chunk_quality(&chunk, Duration::from_millis(1)).await;
    assert!(
        measurement.responsiveness >= 0.0 && measurement.responsiveness <= 1.0,
        "responsiveness must be in [0.0, 1.0] for instant delivery"
    );
}
