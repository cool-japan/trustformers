//! Quality analyzer and streaming algorithms for conversational AI pipeline.
//!
//! This module provides comprehensive quality assessment for streaming conversational responses,
//! including real-time quality metrics, trend analysis, performance evaluation, and optimization
//! recommendations for natural and coherent streaming delivery.

use super::types::*;
use crate::pipeline::conversational::types::TrendDirection;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

// ================================================================================================
// QUALITY MEASUREMENT AND ANALYSIS TYPES
// ================================================================================================

/// Quality indicators for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingQuality {
    /// Smoothness score (0.0 to 1.0)
    pub smoothness: f32,
    /// Naturalness score (0.0 to 1.0)
    pub naturalness: f32,
    /// Responsiveness score (0.0 to 1.0)
    pub responsiveness: f32,
    /// Coherence score (0.0 to 1.0)
    pub coherence: f32,
    /// Overall quality score (0.0 to 1.0)
    pub overall_quality: f32,
    /// Chunk consistency score (0.0 to 1.0)
    pub chunk_consistency: f32,
    /// Flow smoothness score (0.0 to 1.0)
    pub flow_smoothness: f32,
    /// Timing accuracy score (0.0 to 1.0)
    pub timing_accuracy: f32,
    /// Buffer efficiency score (0.0 to 1.0)
    pub buffer_efficiency: f32,
}

impl Default for StreamingQuality {
    fn default() -> Self {
        Self {
            smoothness: 0.8,
            naturalness: 0.8,
            responsiveness: 0.8,
            coherence: 0.8,
            overall_quality: 0.8,
            chunk_consistency: 0.8,
            flow_smoothness: 0.8,
            timing_accuracy: 0.8,
            buffer_efficiency: 0.8,
        }
    }
}

/// Individual quality measurement
#[derive(Debug, Clone)]
pub struct QualityMeasurement {
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Smoothness score (0.0 to 1.0)
    pub smoothness: f32,
    /// Naturalness score (0.0 to 1.0)
    pub naturalness: f32,
    /// Responsiveness score (0.0 to 1.0)
    pub responsiveness: f32,
    /// Coherence score (0.0 to 1.0)
    pub coherence: f32,
    /// Latency (ms)
    pub latency_ms: f64,
    /// Chunk size consistency
    pub chunk_consistency: f32,
    /// Overall quality score (0.0 to 1.0)
    pub score: f32,
    /// Confidence in the measurement (0.0 to 1.0)
    pub confidence: f32,
    /// Quality trend direction
    pub trend: TrendDirection,
}

/// Quality thresholds for different aspects
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    /// Minimum smoothness threshold
    pub min_smoothness: f32,
    /// Minimum naturalness threshold
    pub min_naturalness: f32,
    /// Minimum responsiveness threshold
    pub min_responsiveness: f32,
    /// Minimum coherence threshold
    pub min_coherence: f32,
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: f64,
    /// Minimum overall quality threshold
    pub min_overall_quality: f32,
    /// Minimum acceptable quality level
    pub minimum_acceptable: f32,
    /// Target quality level
    pub target_quality: f32,
    /// Excellent quality threshold
    pub excellent_threshold: f32,
    /// Quality degradation threshold
    pub degradation_threshold: f32,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_smoothness: 0.7,
            min_naturalness: 0.6,
            min_responsiveness: 0.8,
            min_coherence: 0.7,
            max_latency_ms: 200.0,
            min_overall_quality: 0.7,
            minimum_acceptable: 0.6,
            target_quality: 0.8,
            excellent_threshold: 0.9,
            degradation_threshold: 0.5,
        }
    }
}

/// Quality trends analysis
#[derive(Debug, Clone)]
pub struct QualityTrends {
    /// Overall quality trend
    pub overall_trend: TrendDirection,
    /// Smoothness trend
    pub smoothness_trend: TrendDirection,
    /// Naturalness trend
    pub naturalness_trend: TrendDirection,
    /// Responsiveness trend
    pub responsiveness_trend: TrendDirection,
    /// Coherence trend
    pub coherence_trend: TrendDirection,
}

impl Default for QualityTrends {
    fn default() -> Self {
        Self {
            overall_trend: TrendDirection::Stable,
            smoothness_trend: TrendDirection::Stable,
            naturalness_trend: TrendDirection::Stable,
            responsiveness_trend: TrendDirection::Stable,
            coherence_trend: TrendDirection::Stable,
        }
    }
}

/// Advanced quality metrics for comprehensive analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdvancedQualityMetrics {
    /// Perceptual quality scores
    pub perceptual_quality: PerceptualQuality,
    /// Statistical analysis
    pub statistical_analysis: StatisticalAnalysis,
    /// Performance benchmarks
    pub performance_benchmarks: PerformanceBenchmarks,
    /// Quality degradation indicators
    pub degradation_indicators: DegradationIndicators,
}

/// Perceptual quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualQuality {
    /// Fluency assessment (how natural the flow feels)
    pub fluency: f32,
    /// Engagement level (how engaging the content is)
    pub engagement: f32,
    /// Clarity score (how clear and understandable)
    pub clarity: f32,
    /// Emotional appropriateness
    pub emotional_tone: f32,
    /// Conversational flow quality
    pub conversation_flow: f32,
    /// User experience score
    pub user_experience_score: f32,
    /// Cognitive load assessment
    pub cognitive_load: f32,
    /// Attention retention score
    pub attention_retention: f32,
    /// Engagement flow quality
    pub engagement_flow: f32,
}

/// Statistical analysis of quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysis {
    /// Mean quality scores over time window
    pub mean_scores: QualityScores,
    /// Standard deviation of quality metrics
    pub std_deviation: QualityScores,
    /// Quality variance indicators
    pub variance: QualityScores,
    /// Quality distribution percentiles
    pub percentiles: QualityPercentiles,
    /// Correlation analysis between metrics
    pub correlations: QualityCorrelations,
    /// Total number of chunks
    pub chunk_count: usize,
    /// Total number of characters
    pub total_characters: usize,
    /// Average chunk size
    pub average_chunk_size: f32,
    /// Distribution skew
    pub distribution_skew: f32,
}

/// Quality scores structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScores {
    pub smoothness: f32,
    pub naturalness: f32,
    pub responsiveness: f32,
    pub coherence: f32,
}

/// Quality percentiles for distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityPercentiles {
    pub p25: QualityScores,
    pub p50: QualityScores,
    pub p75: QualityScores,
    pub p90: QualityScores,
    pub p95: QualityScores,
}

/// Correlation analysis between quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityCorrelations {
    pub smoothness_naturalness: f32,
    pub responsiveness_coherence: f32,
    pub latency_quality: f32,
    pub consistency_smoothness: f32,
}

/// Performance benchmarks and comparisons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmarks {
    /// Benchmark scores against target performance
    pub target_comparison: BenchmarkComparison,
    /// Historical performance comparison
    pub historical_comparison: BenchmarkComparison,
    /// Peer comparison (if available)
    pub peer_comparison: Option<BenchmarkComparison>,
    /// Performance ranking percentile
    pub performance_percentile: f32,
    /// Latency percentiles (p50, p95, p99)
    pub latency_percentiles: (f32, f32, f32),
    /// Throughput in megabits per second
    pub throughput_mbps: f32,
    /// Resource efficiency score
    pub resource_efficiency: f32,
    /// Scalability factor
    pub scalability_factor: f32,
}

/// Benchmark comparison structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub relative_performance: f32,  // -1.0 to 1.0 (worse to better)
    pub improvement_potential: f32, // 0.0 to 1.0
    pub confidence_level: f32,      // 0.0 to 1.0
}

/// Quality degradation indicators and alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationIndicators {
    /// Is quality degrading?
    pub is_degrading: bool,
    /// Rate of degradation (per hour)
    pub degradation_rate: f32,
    /// Predicted time to threshold breach
    pub time_to_threshold_breach: Option<Duration>,
    /// Critical quality areas
    pub critical_areas: Vec<QualityArea>,
    /// Recommended actions
    pub recommended_actions: Vec<OptimizationRecommendation>,
    /// Number of buffer overflows
    pub buffer_overflows: usize,
    /// Number of quality drops
    pub quality_drops: usize,
    /// Number of recovery events
    pub recovery_events: usize,
    /// Number of timing violations
    pub timing_violations: usize,
}

/// Quality areas for targeted analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityArea {
    Smoothness,
    Naturalness,
    Responsiveness,
    Coherence,
    Consistency,
    OverallPerformance,
    Performance,
    Reliability,
}

/// Optimization recommendations based on quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Type of optimization
    pub optimization_type: OptimizationType,
    /// Recommendation type (alias for optimization_type for backward compatibility)
    pub recommendation_type: OptimizationType,
    /// Priority level (0.0 to 1.0)
    pub priority: f32,
    /// Expected improvement
    pub expected_improvement: f32,
    /// Implementation complexity
    pub complexity: ComplexityLevel,
    /// Description of the recommendation
    pub description: String,
    /// Affected quality areas
    pub affected_areas: Vec<QualityArea>,
}

/// Types of optimizations available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    ChunkSizeAdjustment,
    TimingOptimization,
    FlowRateAdjustment,
    BufferManagement,
    PipelineReorganization,
    ModelParameterTuning,
    InfrastructureUpgrade,
    Performance,
}

/// Complexity levels for implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Low,    // Quick configuration changes
    Medium, // Code modifications required
    High,   // Significant architectural changes
}

// ================================================================================================
// QUALITY ANALYZER IMPLEMENTATION
// ================================================================================================

/// Quality analyzer for streaming performance
#[derive(Debug)]
pub struct QualityAnalyzer {
    /// Quality metrics window
    metrics_window: Arc<RwLock<VecDeque<QualityMeasurement>>>,
    /// Window size for analysis
    window_size: usize,
    /// Quality thresholds
    pub thresholds: QualityThresholds,
    /// Advanced analysis enabled
    advanced_analysis_enabled: bool,
    /// Historical metrics for trend analysis
    historical_metrics: Arc<RwLock<VecDeque<StreamingQuality>>>,
    /// Performance baselines
    performance_baselines: Arc<RwLock<Option<StreamingQuality>>>,
    /// Overall quality measurement
    pub overall_quality: QualityMeasurement,
    /// Current streaming quality metrics
    pub streaming_quality: StreamingQuality,
    /// Advanced metrics data
    pub advanced_metrics: AdvancedQualityMetrics,
    /// Optimization recommendations
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
    /// Quality thresholds configuration
    pub quality_thresholds: QualityThresholds,
    /// Assessment metadata
    pub assessment_metadata: std::collections::HashMap<String, String>,
}

impl QualityAnalyzer {
    /// Create a new quality analyzer
    pub fn new() -> Self {
        use crate::pipeline::conversational::types::TrendDirection;

        Self {
            metrics_window: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            window_size: 100,
            thresholds: QualityThresholds::default(),
            advanced_analysis_enabled: true,
            historical_metrics: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            performance_baselines: Arc::new(RwLock::new(None)),
            overall_quality: QualityMeasurement {
                timestamp: Instant::now(),
                smoothness: 0.8,
                naturalness: 0.8,
                responsiveness: 0.8,
                coherence: 0.8,
                latency_ms: 100.0,
                chunk_consistency: 0.8,
                score: 0.8,
                confidence: 0.7,
                trend: TrendDirection::Stable,
            },
            streaming_quality: StreamingQuality::default(),
            advanced_metrics: AdvancedQualityMetrics::default(),
            optimization_recommendations: Vec::new(),
            quality_thresholds: QualityThresholds::default(),
            assessment_metadata: std::collections::HashMap::new(),
        }
    }

    /// Create a new quality analyzer with custom configuration
    pub fn with_config(window_size: usize, thresholds: QualityThresholds) -> Self {
        use crate::pipeline::conversational::types::TrendDirection;

        Self {
            metrics_window: Arc::new(RwLock::new(VecDeque::with_capacity(window_size))),
            window_size,
            thresholds: thresholds.clone(),
            advanced_analysis_enabled: true,
            historical_metrics: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            performance_baselines: Arc::new(RwLock::new(None)),
            overall_quality: QualityMeasurement {
                timestamp: Instant::now(),
                smoothness: 0.8,
                naturalness: 0.8,
                responsiveness: 0.8,
                coherence: 0.8,
                latency_ms: 100.0,
                chunk_consistency: 0.8,
                score: 0.8,
                confidence: 0.7,
                trend: TrendDirection::Stable,
            },
            streaming_quality: StreamingQuality::default(),
            advanced_metrics: AdvancedQualityMetrics::default(),
            optimization_recommendations: Vec::new(),
            quality_thresholds: thresholds,
            assessment_metadata: std::collections::HashMap::new(),
        }
    }

    /// Get metrics window for external access
    pub fn metrics_window(&self) -> &Arc<RwLock<VecDeque<QualityMeasurement>>> {
        &self.metrics_window
    }

    /// Get window size
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get quality thresholds
    pub fn thresholds(&self) -> &QualityThresholds {
        &self.thresholds
    }

    /// Analyze chunk quality with comprehensive metrics
    pub async fn analyze_chunk_quality(
        &self,
        chunk: &StreamChunk,
        delivery_time: Duration,
    ) -> QualityMeasurement {
        let smoothness = self.calculate_smoothness(chunk, delivery_time).await;
        let naturalness = self.calculate_naturalness(chunk).await;
        let responsiveness = self.calculate_responsiveness(delivery_time);
        let coherence = self.calculate_coherence(chunk).await;
        let chunk_consistency = self.calculate_chunk_consistency(chunk).await;

        // Calculate overall score as weighted average
        let score =
            smoothness * 0.25 + naturalness * 0.25 + responsiveness * 0.25 + coherence * 0.25;

        let measurement = QualityMeasurement {
            timestamp: Instant::now(),
            smoothness,
            naturalness,
            responsiveness,
            coherence,
            latency_ms: delivery_time.as_millis() as f64,
            chunk_consistency,
            score,
            confidence: (score * 0.8 + chunk_consistency * 0.2).min(1.0).max(0.0),
            trend: TrendDirection::Stable,
        };

        // Add to metrics window
        let mut window = self.metrics_window.write().await;
        window.push_back(measurement.clone());

        // Keep window size
        if window.len() > self.window_size {
            window.pop_front();
        }

        measurement
    }

    /// Calculate overall streaming quality
    pub async fn calculate_overall_quality(&self) -> StreamingQuality {
        let window = self.metrics_window.read().await;

        if window.is_empty() {
            return StreamingQuality::default();
        }

        let count = window.len() as f32;
        let smoothness = window.iter().map(|m| m.smoothness).sum::<f32>() / count;
        let naturalness = window.iter().map(|m| m.naturalness).sum::<f32>() / count;
        let responsiveness = window.iter().map(|m| m.responsiveness).sum::<f32>() / count;
        let coherence = window.iter().map(|m| m.coherence).sum::<f32>() / count;

        let overall_quality = (smoothness + naturalness + responsiveness + coherence) / 4.0;

        let quality = StreamingQuality {
            smoothness,
            naturalness,
            responsiveness,
            coherence,
            overall_quality,
            chunk_consistency: 0.8,
            flow_smoothness: 0.8,
            timing_accuracy: 0.8,
            buffer_efficiency: 0.8,
        };

        // Add to historical metrics
        let mut historical = self.historical_metrics.write().await;
        historical.push_back(quality.clone());
        if historical.len() > 1000 {
            historical.pop_front();
        }

        quality
    }

    /// Calculate advanced quality metrics with comprehensive analysis
    pub async fn calculate_advanced_metrics(&self) -> AdvancedQualityMetrics {
        if !self.advanced_analysis_enabled {
            return AdvancedQualityMetrics::default();
        }

        let window = self.metrics_window.read().await;
        let historical = self.historical_metrics.read().await;

        AdvancedQualityMetrics {
            perceptual_quality: self.calculate_perceptual_quality(&window).await,
            statistical_analysis: self.calculate_statistical_analysis(&window).await,
            performance_benchmarks: self.calculate_performance_benchmarks(&historical).await,
            degradation_indicators: self
                .calculate_degradation_indicators(&window, &historical)
                .await,
        }
    }

    /// Check if quality meets thresholds
    pub async fn meets_quality_thresholds(&self) -> bool {
        let quality = self.calculate_overall_quality().await;

        quality.smoothness >= self.thresholds.min_smoothness
            && quality.naturalness >= self.thresholds.min_naturalness
            && quality.responsiveness >= self.thresholds.min_responsiveness
            && quality.coherence >= self.thresholds.min_coherence
            && quality.overall_quality >= self.thresholds.min_overall_quality
    }

    /// Get quality trends with sophisticated analysis
    pub async fn get_quality_trends(&self) -> QualityTrends {
        let window = self.metrics_window.read().await;

        if window.len() < 10 {
            return QualityTrends::default();
        }

        let recent = &window.as_slices().0[window.len() - 5..];
        let earlier = &window.as_slices().0[window.len() - 10..window.len() - 5];

        let recent_avg = recent
            .iter()
            .map(|m| (m.smoothness + m.naturalness + m.responsiveness + m.coherence) / 4.0)
            .sum::<f32>()
            / recent.len() as f32;
        let earlier_avg = earlier
            .iter()
            .map(|m| (m.smoothness + m.naturalness + m.responsiveness + m.coherence) / 4.0)
            .sum::<f32>()
            / earlier.len() as f32;

        let overall_trend = self.calculate_trend_direction(recent_avg, earlier_avg);

        // Calculate individual metric trends
        let smoothness_trend = self.calculate_metric_trend(recent, earlier, |m| m.smoothness);
        let naturalness_trend = self.calculate_metric_trend(recent, earlier, |m| m.naturalness);
        let responsiveness_trend =
            self.calculate_metric_trend(recent, earlier, |m| m.responsiveness);
        let coherence_trend = self.calculate_metric_trend(recent, earlier, |m| m.coherence);

        QualityTrends {
            overall_trend,
            smoothness_trend,
            naturalness_trend,
            responsiveness_trend,
            coherence_trend,
        }
    }

    /// Generate optimization recommendations based on quality analysis
    pub async fn generate_optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let quality = self.calculate_overall_quality().await;
        let trends = self.get_quality_trends().await;
        let mut recommendations = Vec::new();

        // Analyze each quality dimension and provide recommendations
        if quality.smoothness < self.thresholds.min_smoothness {
            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::ChunkSizeAdjustment,
                recommendation_type: OptimizationType::ChunkSizeAdjustment,
                priority: 0.8,
                expected_improvement: 0.15,
                complexity: ComplexityLevel::Low,
                description: "Adjust chunk sizes to improve smoothness - consider smaller, more consistent chunks".to_string(),
                affected_areas: vec![QualityArea::Smoothness],
            });
        }

        if quality.responsiveness < self.thresholds.min_responsiveness {
            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::TimingOptimization,
                recommendation_type: OptimizationType::TimingOptimization,
                priority: 0.9,
                expected_improvement: 0.2,
                complexity: ComplexityLevel::Medium,
                description:
                    "Optimize timing algorithms to reduce latency and improve responsiveness"
                        .to_string(),
                affected_areas: vec![QualityArea::Responsiveness],
            });
        }

        if quality.naturalness < self.thresholds.min_naturalness {
            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::ModelParameterTuning,
                recommendation_type: OptimizationType::ModelParameterTuning,
                priority: 0.7,
                expected_improvement: 0.1,
                complexity: ComplexityLevel::High,
                description:
                    "Fine-tune model parameters to improve naturalness of generated content"
                        .to_string(),
                affected_areas: vec![QualityArea::Naturalness],
            });
        }

        if quality.coherence < self.thresholds.min_coherence {
            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::PipelineReorganization,
                recommendation_type: OptimizationType::PipelineReorganization,
                priority: 0.6,
                expected_improvement: 0.12,
                complexity: ComplexityLevel::High,
                description:
                    "Reorganize processing pipeline to maintain better coherence across chunks"
                        .to_string(),
                affected_areas: vec![QualityArea::Coherence],
            });
        }

        // Check for declining trends
        if trends.overall_trend == TrendDirection::Declining {
            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::InfrastructureUpgrade,
                recommendation_type: OptimizationType::InfrastructureUpgrade,
                priority: 0.5,
                expected_improvement: 0.25,
                complexity: ComplexityLevel::High,
                description:
                    "Consider infrastructure upgrades to address declining performance trends"
                        .to_string(),
                affected_areas: vec![QualityArea::OverallPerformance],
            });
        }

        // Sort by priority
        recommendations.sort_by(|a, b| {
            b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal)
        });
        recommendations
    }

    /// Set performance baseline for comparison
    pub async fn set_performance_baseline(&self, baseline: StreamingQuality) {
        let mut baselines = self.performance_baselines.write().await;
        *baselines = Some(baseline);
    }

    /// Clear quality metrics and reset analyzer
    pub async fn reset(&self) {
        let mut window = self.metrics_window.write().await;
        window.clear();

        let mut historical = self.historical_metrics.write().await;
        historical.clear();
    }

    // ================================================================================================
    // PRIVATE QUALITY CALCULATION METHODS
    // ================================================================================================

    /// Calculate smoothness with advanced algorithms
    async fn calculate_smoothness(&self, chunk: &StreamChunk, delivery_time: Duration) -> f32 {
        let mut smoothness = 0.8;

        // Timing consistency analysis
        let window = self.metrics_window.read().await;
        if window.len() > 1 {
            let timing_variance = self.calculate_timing_variance(&window);
            smoothness *= (1.0 - timing_variance.min(0.5)) * 2.0;
        }

        // Content length consistency
        let length_factor = self.calculate_length_consistency_factor(chunk);
        smoothness *= length_factor;

        // Delivery time smoothness
        let target_time_ms = 100.0;
        let actual_time_ms = delivery_time.as_millis() as f64;
        let timing_factor = if actual_time_ms <= target_time_ms * 1.5 {
            1.0
        } else {
            (target_time_ms / actual_time_ms).max(0.3) as f32
        };
        smoothness *= timing_factor;

        smoothness.max(0.0).min(1.0)
    }

    /// Calculate naturalness with linguistic analysis
    async fn calculate_naturalness(&self, chunk: &StreamChunk) -> f32 {
        let content = &chunk.content;
        let mut naturalness = 0.8;

        // Linguistic pattern analysis
        naturalness *= self.analyze_linguistic_patterns(content);

        // Content flow analysis
        naturalness *= self.analyze_content_flow(chunk).await;

        // Punctuation and structure analysis
        naturalness *= self.analyze_punctuation_structure(content);

        naturalness.max(0.0).min(1.0)
    }

    /// Calculate responsiveness based on delivery metrics
    fn calculate_responsiveness(&self, delivery_time: Duration) -> f32 {
        let target_time_ms = 100.0;
        let actual_time_ms = delivery_time.as_millis() as f64;

        if actual_time_ms <= target_time_ms {
            1.0
        } else if actual_time_ms <= target_time_ms * 2.0 {
            (target_time_ms / actual_time_ms).max(0.5) as f32
        } else {
            (target_time_ms / actual_time_ms).max(0.1) as f32
        }
    }

    /// Calculate coherence with context awareness
    async fn calculate_coherence(&self, chunk: &StreamChunk) -> f32 {
        let mut coherence = 0.8;

        // Content coherence analysis
        coherence *= self.analyze_content_coherence(chunk);

        // Context consistency (if available from previous chunks)
        coherence *= self.analyze_context_consistency(chunk).await;

        // Semantic coherence
        coherence *= self.analyze_semantic_coherence(chunk);

        coherence.max(0.0).min(1.0)
    }

    /// Calculate chunk consistency with statistical analysis
    async fn calculate_chunk_consistency(&self, chunk: &StreamChunk) -> f32 {
        let window = self.metrics_window.read().await;

        if window.len() < 2 {
            return 1.0;
        }

        // Calculate variance in chunk characteristics
        let recent_window: Vec<_> = window.iter().rev().take(10).collect();
        let sizes: Vec<usize> = recent_window.iter().map(|_| chunk.content.len()).collect();

        if sizes.is_empty() {
            return 1.0;
        }

        let mean_size = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;
        let variance = sizes
            .iter()
            .map(|&size| {
                let diff = size as f32 - mean_size;
                diff * diff
            })
            .sum::<f32>()
            / sizes.len() as f32;

        // Convert variance to consistency score
        let consistency = 1.0 / (1.0 + variance / (mean_size * mean_size + 1.0));
        consistency.max(0.0).min(1.0)
    }

    /// Calculate perceptual quality metrics
    async fn calculate_perceptual_quality(
        &self,
        window: &VecDeque<QualityMeasurement>,
    ) -> PerceptualQuality {
        if window.is_empty() {
            return PerceptualQuality::default();
        }

        let count = window.len() as f32;

        PerceptualQuality {
            fluency: window.iter().map(|m| m.smoothness * m.coherence).sum::<f32>() / count,
            engagement: window.iter().map(|m| m.naturalness * 0.9).sum::<f32>() / count,
            clarity: window.iter().map(|m| m.coherence * 0.95).sum::<f32>() / count,
            emotional_tone: window.iter().map(|m| m.naturalness * 0.8).sum::<f32>() / count,
            conversation_flow: window
                .iter()
                .map(|m| (m.smoothness + m.responsiveness) / 2.0)
                .sum::<f32>()
                / count,
            user_experience_score: window
                .iter()
                .map(|m| (m.smoothness + m.naturalness + m.responsiveness) / 3.0)
                .sum::<f32>()
                / count,
            cognitive_load: window.iter().map(|m| (1.0 - m.coherence) * 0.5).sum::<f32>() / count,
            attention_retention: window.iter().map(|m| m.responsiveness * m.coherence).sum::<f32>()
                / count,
            engagement_flow: window
                .iter()
                .map(|m| (m.naturalness + m.responsiveness) / 2.0)
                .sum::<f32>()
                / count,
        }
    }

    /// Calculate statistical analysis of quality metrics
    async fn calculate_statistical_analysis(
        &self,
        window: &VecDeque<QualityMeasurement>,
    ) -> StatisticalAnalysis {
        if window.is_empty() {
            return StatisticalAnalysis::default();
        }

        let smoothness_values: Vec<f32> = window.iter().map(|m| m.smoothness).collect();
        let naturalness_values: Vec<f32> = window.iter().map(|m| m.naturalness).collect();
        let responsiveness_values: Vec<f32> = window.iter().map(|m| m.responsiveness).collect();
        let coherence_values: Vec<f32> = window.iter().map(|m| m.coherence).collect();

        let total_chars: usize = window.iter().map(|m| (m.latency_ms * 10.0) as usize).sum();
        let chunk_count = window.len();
        let avg_size = if chunk_count > 0 { total_chars as f32 / chunk_count as f32 } else { 0.0 };

        StatisticalAnalysis {
            mean_scores: QualityScores {
                smoothness: self.calculate_mean(&smoothness_values),
                naturalness: self.calculate_mean(&naturalness_values),
                responsiveness: self.calculate_mean(&responsiveness_values),
                coherence: self.calculate_mean(&coherence_values),
            },
            std_deviation: QualityScores {
                smoothness: self.calculate_std_dev(&smoothness_values),
                naturalness: self.calculate_std_dev(&naturalness_values),
                responsiveness: self.calculate_std_dev(&responsiveness_values),
                coherence: self.calculate_std_dev(&coherence_values),
            },
            variance: QualityScores {
                smoothness: self.calculate_variance(&smoothness_values),
                naturalness: self.calculate_variance(&naturalness_values),
                responsiveness: self.calculate_variance(&responsiveness_values),
                coherence: self.calculate_variance(&coherence_values),
            },
            percentiles: self.calculate_percentiles(
                &smoothness_values,
                &naturalness_values,
                &responsiveness_values,
                &coherence_values,
            ),
            correlations: self.calculate_correlations(window),
            chunk_count,
            total_characters: total_chars,
            average_chunk_size: avg_size,
            distribution_skew: self.calculate_std_dev(&smoothness_values)
                / (self.calculate_mean(&smoothness_values) + 0.001),
        }
    }

    /// Calculate performance benchmarks
    async fn calculate_performance_benchmarks(
        &self,
        historical: &VecDeque<StreamingQuality>,
    ) -> PerformanceBenchmarks {
        let baselines = self.performance_baselines.read().await;
        let window = self.metrics_window.read().await;

        if let Some(baseline) = baselines.as_ref() {
            if let Some(current) = historical.back() {
                let target_comparison = self.compare_to_baseline(current, baseline);
                let historical_comparison = self.compare_to_historical(current, historical);

                let latency_values: Vec<f32> = window.iter().map(|m| m.latency_ms as f32).collect();
                let p50 = self.calculate_percentile_value(&latency_values, 0.50);
                let p95 = self.calculate_percentile_value(&latency_values, 0.95);
                let p99 = self.calculate_percentile_value(&latency_values, 0.99);

                // Calculate throughput: measurements per second, converted to approximate MB/s
                // Assuming avg chunk size of ~100 bytes per measurement
                let time_window_secs = if window.len() > 1 {
                    let elapsed = window
                        .back()
                        .unwrap()
                        .timestamp
                        .duration_since(window.front().unwrap().timestamp);
                    elapsed.as_secs_f32().max(1.0)
                } else {
                    1.0
                };
                let measurements_per_sec = window.len() as f32 / time_window_secs;
                let throughput_mbps = (measurements_per_sec * 100.0) / 1_000_000.0; // bytes to MB

                // Calculate resource efficiency: inverse of latency normalized by quality
                // Higher quality with lower latency = better efficiency
                let avg_latency = latency_values.iter().sum::<f32>() / latency_values.len() as f32;
                let avg_quality = window.iter().map(|m| m.score).sum::<f32>() / window.len() as f32;
                let resource_efficiency = if avg_latency > 0.0 {
                    (avg_quality * 100.0 / avg_latency).min(1.0).max(0.0)
                } else {
                    avg_quality
                };

                // Calculate scalability factor: consistency of quality across window
                // Lower variance = better scalability
                let quality_variance = {
                    let qualities: Vec<f32> = window.iter().map(|m| m.score).collect();
                    let mean = avg_quality;
                    let variance = qualities.iter().map(|&q| (q - mean).powi(2)).sum::<f32>()
                        / qualities.len() as f32;
                    variance
                };
                let scalability_factor = (1.0 - quality_variance).max(0.0).min(1.0);

                PerformanceBenchmarks {
                    target_comparison,
                    historical_comparison,
                    peer_comparison: None, // Would require external data
                    performance_percentile: self
                        .calculate_performance_percentile(current, historical),
                    latency_percentiles: (p50, p95, p99),
                    throughput_mbps,
                    resource_efficiency,
                    scalability_factor,
                }
            } else {
                PerformanceBenchmarks::default()
            }
        } else {
            PerformanceBenchmarks::default()
        }
    }

    /// Calculate degradation indicators
    async fn calculate_degradation_indicators(
        &self,
        window: &VecDeque<QualityMeasurement>,
        historical: &VecDeque<StreamingQuality>,
    ) -> DegradationIndicators {
        if historical.len() < 10 {
            return DegradationIndicators::default();
        }

        let recent_quality =
            historical.iter().rev().take(5).map(|q| q.overall_quality).sum::<f32>() / 5.0;
        let earlier_quality =
            historical.iter().rev().skip(5).take(5).map(|q| q.overall_quality).sum::<f32>() / 5.0;

        let is_degrading = recent_quality < earlier_quality - 0.05;
        let degradation_rate = if is_degrading {
            (earlier_quality - recent_quality) / 5.0 // Per measurement period
        } else {
            0.0
        };

        let critical_areas = self.identify_critical_areas(window).await;
        let recommended_actions = if is_degrading {
            self.generate_optimization_recommendations().await
        } else {
            Vec::new()
        };

        // Track buffer overflows by detecting when metrics window is full and we're adding more
        let buffer_overflows = if window.len() >= self.window_size {
            // Count measurements where the window is nearly full (above 95% capacity)
            if window.len() as f32 / self.window_size as f32 > 0.95 {
                1
            } else {
                0
            }
        } else {
            0
        };

        // Track recovery events by detecting quality improvements after drops
        // Convert VecDeque to Vec to use windows() method
        let window_vec: Vec<&QualityMeasurement> = window.iter().collect();
        let recovery_events = window_vec
            .windows(2)
            .filter(|pair| {
                // Recovery: quality was below threshold and then improved significantly
                pair[0].score < 0.7 && pair[1].score >= 0.7
            })
            .count();

        DegradationIndicators {
            is_degrading,
            degradation_rate,
            time_to_threshold_breach: self.estimate_time_to_threshold_breach(degradation_rate),
            critical_areas,
            recommended_actions,
            buffer_overflows,
            quality_drops: window.iter().filter(|m| m.score < 0.7).count(),
            recovery_events,
            timing_violations: window.iter().filter(|m| m.latency_ms > 200.0).count(),
        }
    }

    // ================================================================================================
    // UTILITY AND HELPER METHODS
    // ================================================================================================

    fn calculate_trend_direction(&self, recent: f32, earlier: f32) -> TrendDirection {
        let threshold = 0.05;
        if recent > earlier + threshold {
            TrendDirection::Improving
        } else if recent < earlier - threshold {
            TrendDirection::Declining
        } else {
            TrendDirection::Stable
        }
    }

    fn calculate_metric_trend<F>(
        &self,
        recent: &[QualityMeasurement],
        earlier: &[QualityMeasurement],
        extractor: F,
    ) -> TrendDirection
    where
        F: Fn(&QualityMeasurement) -> f32,
    {
        let recent_avg = recent.iter().map(&extractor).sum::<f32>() / recent.len() as f32;
        let earlier_avg = earlier.iter().map(&extractor).sum::<f32>() / earlier.len() as f32;
        self.calculate_trend_direction(recent_avg, earlier_avg)
    }

    fn calculate_timing_variance(&self, window: &VecDeque<QualityMeasurement>) -> f32 {
        if window.len() < 2 {
            return 0.0;
        }

        let latencies: Vec<f64> = window.iter().map(|m| m.latency_ms).collect();
        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let variance =
            latencies.iter().map(|&l| (l - mean).powi(2)).sum::<f64>() / latencies.len() as f64;

        (variance.sqrt() / mean.max(1.0)) as f32
    }

    fn calculate_length_consistency_factor(&self, chunk: &StreamChunk) -> f32 {
        let length = chunk.content.len();
        if length < 5 {
            0.6
        } else if length > 200 {
            0.8
        } else {
            1.0
        }
    }

    fn analyze_linguistic_patterns(&self, content: &str) -> f32 {
        let mut score: f32 = 1.0;

        // Check for natural hesitation markers
        if content.contains("um") || content.contains("uh") || content.contains("er") {
            score *= 1.1; // These can be natural
        }

        // Check for proper capitalization
        if let Some(first_char) = content.chars().next() {
            if first_char.is_lowercase() && content.len() > 1 {
                score *= 0.9;
            }
        }

        // Check for excessive repetition
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.len() > 1 {
            let mut repeated = 0;
            for i in 1..words.len() {
                if words[i] == words[i - 1] {
                    repeated += 1;
                }
            }
            if repeated > words.len() / 3 {
                score *= 0.7;
            }
        }

        score.max(0.0_f32).min(1.0_f32)
    }

    async fn analyze_content_flow(&self, _chunk: &StreamChunk) -> f32 {
        // Placeholder for sophisticated content flow analysis
        // Would analyze semantic coherence, topic consistency, etc.
        0.9
    }

    fn analyze_punctuation_structure(&self, content: &str) -> f32 {
        let mut score: f32 = 1.0;

        // Check sentence ending
        if content.trim().ends_with(['.', '!', '?']) {
            score *= 1.1;
        }

        // Check for proper comma usage (basic)
        let comma_count = content.matches(',').count();
        let word_count = content.split_whitespace().count();
        if word_count > 10 && comma_count == 0 {
            score *= 0.9; // Long sentences without commas might be unnatural
        }

        score.max(0.0_f32).min(1.0_f32)
    }

    fn analyze_content_coherence(&self, chunk: &StreamChunk) -> f32 {
        let content = &chunk.content.trim();

        if content.is_empty() {
            return 0.0;
        }

        let mut coherence = 0.9;

        // Check for sentence fragments
        if matches!(chunk.chunk_type, ChunkType::Sentence) && !content.ends_with(['.', '!', '?']) {
            coherence *= 0.8;
        }

        // Check for proper word boundaries
        if chunk.content.starts_with(' ') || chunk.content.ends_with(' ') {
            coherence *= 0.95; // Minor penalty for boundary issues
        }

        coherence
    }

    async fn analyze_context_consistency(&self, _chunk: &StreamChunk) -> f32 {
        // Placeholder for context consistency analysis
        // Would check against previous chunks for topic drift, style consistency, etc.
        0.9
    }

    fn analyze_semantic_coherence(&self, _chunk: &StreamChunk) -> f32 {
        // Placeholder for semantic coherence analysis
        // Would use NLP techniques to analyze semantic consistency
        0.9
    }

    fn calculate_mean(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        }
    }

    fn calculate_std_dev(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = self.calculate_mean(values);
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        variance.sqrt()
    }

    fn calculate_variance(&self, values: &[f32]) -> f32 {
        self.calculate_std_dev(values).powi(2)
    }

    fn calculate_percentile_value(&self, values: &[f32], percentile: f32) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let index = ((percentile * (sorted.len() - 1) as f32) as usize).min(sorted.len() - 1);
        sorted[index]
    }

    fn calculate_percentiles(
        &self,
        smoothness: &[f32],
        naturalness: &[f32],
        responsiveness: &[f32],
        coherence: &[f32],
    ) -> QualityPercentiles {
        QualityPercentiles {
            p25: QualityScores {
                smoothness: self.percentile(smoothness, 0.25),
                naturalness: self.percentile(naturalness, 0.25),
                responsiveness: self.percentile(responsiveness, 0.25),
                coherence: self.percentile(coherence, 0.25),
            },
            p50: QualityScores {
                smoothness: self.percentile(smoothness, 0.50),
                naturalness: self.percentile(naturalness, 0.50),
                responsiveness: self.percentile(responsiveness, 0.50),
                coherence: self.percentile(coherence, 0.50),
            },
            p75: QualityScores {
                smoothness: self.percentile(smoothness, 0.75),
                naturalness: self.percentile(naturalness, 0.75),
                responsiveness: self.percentile(responsiveness, 0.75),
                coherence: self.percentile(coherence, 0.75),
            },
            p90: QualityScores {
                smoothness: self.percentile(smoothness, 0.90),
                naturalness: self.percentile(naturalness, 0.90),
                responsiveness: self.percentile(responsiveness, 0.90),
                coherence: self.percentile(coherence, 0.90),
            },
            p95: QualityScores {
                smoothness: self.percentile(smoothness, 0.95),
                naturalness: self.percentile(naturalness, 0.95),
                responsiveness: self.percentile(responsiveness, 0.95),
                coherence: self.percentile(coherence, 0.95),
            },
        }
    }

    fn percentile(&self, values: &[f32], p: f32) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = ((sorted.len() - 1) as f32 * p) as usize;
        sorted[index]
    }

    fn calculate_correlations(&self, window: &VecDeque<QualityMeasurement>) -> QualityCorrelations {
        if window.len() < 2 {
            return QualityCorrelations::default();
        }

        let smoothness: Vec<f32> = window.iter().map(|m| m.smoothness).collect();
        let naturalness: Vec<f32> = window.iter().map(|m| m.naturalness).collect();
        let responsiveness: Vec<f32> = window.iter().map(|m| m.responsiveness).collect();
        let coherence: Vec<f32> = window.iter().map(|m| m.coherence).collect();
        let latency: Vec<f32> = window.iter().map(|m| m.latency_ms as f32).collect();
        let consistency: Vec<f32> = window.iter().map(|m| m.chunk_consistency).collect();

        QualityCorrelations {
            smoothness_naturalness: self.correlation(&smoothness, &naturalness),
            responsiveness_coherence: self.correlation(&responsiveness, &coherence),
            latency_quality: -self.correlation(&latency, &smoothness), // Negative correlation expected
            consistency_smoothness: self.correlation(&consistency, &smoothness),
        }
    }

    fn correlation(&self, x: &[f32], y: &[f32]) -> f32 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let mean_x = self.calculate_mean(x);
        let mean_y = self.calculate_mean(y);

        let numerator: f32 =
            x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y)).sum();
        let sum_sq_x: f32 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f32 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn compare_to_baseline(
        &self,
        current: &StreamingQuality,
        baseline: &StreamingQuality,
    ) -> BenchmarkComparison {
        let relative_performance = (current.overall_quality - baseline.overall_quality)
            / baseline.overall_quality.max(0.1);

        BenchmarkComparison {
            relative_performance: relative_performance.max(-1.0).min(1.0),
            improvement_potential: (1.0 - current.overall_quality).max(0.0),
            confidence_level: 0.8, // Static for now, could be dynamic
        }
    }

    fn compare_to_historical(
        &self,
        current: &StreamingQuality,
        historical: &VecDeque<StreamingQuality>,
    ) -> BenchmarkComparison {
        if historical.len() < 5 {
            return BenchmarkComparison::default();
        }

        let historical_avg =
            historical.iter().map(|q| q.overall_quality).sum::<f32>() / historical.len() as f32;
        let relative_performance =
            (current.overall_quality - historical_avg) / historical_avg.max(0.1);

        BenchmarkComparison {
            relative_performance: relative_performance.max(-1.0).min(1.0),
            improvement_potential: (1.0 - current.overall_quality).max(0.0),
            confidence_level: 0.9,
        }
    }

    fn calculate_performance_percentile(
        &self,
        current: &StreamingQuality,
        historical: &VecDeque<StreamingQuality>,
    ) -> f32 {
        if historical.is_empty() {
            return 0.5;
        }

        let better_count = historical
            .iter()
            .filter(|q| q.overall_quality < current.overall_quality)
            .count();
        better_count as f32 / historical.len() as f32
    }

    async fn identify_critical_areas(
        &self,
        window: &VecDeque<QualityMeasurement>,
    ) -> Vec<QualityArea> {
        let mut critical_areas = Vec::new();

        if let Some(latest) = window.back() {
            if latest.smoothness < self.thresholds.min_smoothness {
                critical_areas.push(QualityArea::Smoothness);
            }
            if latest.naturalness < self.thresholds.min_naturalness {
                critical_areas.push(QualityArea::Naturalness);
            }
            if latest.responsiveness < self.thresholds.min_responsiveness {
                critical_areas.push(QualityArea::Responsiveness);
            }
            if latest.coherence < self.thresholds.min_coherence {
                critical_areas.push(QualityArea::Coherence);
            }
            if latest.chunk_consistency < 0.7 {
                critical_areas.push(QualityArea::Consistency);
            }
        }

        critical_areas
    }

    fn estimate_time_to_threshold_breach(&self, degradation_rate: f32) -> Option<Duration> {
        if degradation_rate <= 0.0 {
            return None;
        }

        // Estimate based on current degradation rate
        let time_periods = (self.thresholds.min_overall_quality / degradation_rate) as u64;
        Some(Duration::from_secs(time_periods * 60)) // Assuming 1-minute periods
    }
}

// ================================================================================================
// DEFAULT IMPLEMENTATIONS
// ================================================================================================

impl Default for PerceptualQuality {
    fn default() -> Self {
        Self {
            fluency: 0.8,
            engagement: 0.8,
            clarity: 0.8,
            emotional_tone: 0.8,
            conversation_flow: 0.8,
            user_experience_score: 0.8,
            cognitive_load: 0.3,
            attention_retention: 0.8,
            engagement_flow: 0.8,
        }
    }
}

impl Default for StatisticalAnalysis {
    fn default() -> Self {
        Self {
            mean_scores: QualityScores::default(),
            std_deviation: QualityScores::default(),
            variance: QualityScores::default(),
            percentiles: QualityPercentiles::default(),
            correlations: QualityCorrelations::default(),
            chunk_count: 0,
            total_characters: 0,
            average_chunk_size: 0.0,
            distribution_skew: 0.0,
        }
    }
}

impl Default for QualityScores {
    fn default() -> Self {
        Self {
            smoothness: 0.8,
            naturalness: 0.8,
            responsiveness: 0.8,
            coherence: 0.8,
        }
    }
}

impl Default for QualityCorrelations {
    fn default() -> Self {
        Self {
            smoothness_naturalness: 0.0,
            responsiveness_coherence: 0.0,
            latency_quality: 0.0,
            consistency_smoothness: 0.0,
        }
    }
}

impl Default for PerformanceBenchmarks {
    fn default() -> Self {
        Self {
            target_comparison: BenchmarkComparison::default(),
            historical_comparison: BenchmarkComparison::default(),
            peer_comparison: None,
            performance_percentile: 0.5,
            latency_percentiles: (100.0, 200.0, 300.0),
            throughput_mbps: 1.0,
            resource_efficiency: 0.8,
            scalability_factor: 1.0,
        }
    }
}

impl Default for BenchmarkComparison {
    fn default() -> Self {
        Self {
            relative_performance: 0.0,
            improvement_potential: 0.2,
            confidence_level: 0.5,
        }
    }
}

impl Default for DegradationIndicators {
    fn default() -> Self {
        Self {
            is_degrading: false,
            degradation_rate: 0.0,
            time_to_threshold_breach: None,
            critical_areas: Vec::new(),
            recommended_actions: Vec::new(),
            buffer_overflows: 0,
            quality_drops: 0,
            recovery_events: 0,
            timing_violations: 0,
        }
    }
}

impl Default for QualityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// QUALITY ANALYSIS TRAIT
// ================================================================================================

/// Trait for quality analysis functionality
#[async_trait]
pub trait QualityAnalysis {
    /// Analyze the quality of a stream chunk
    async fn analyze_quality(
        &self,
        chunk: &StreamChunk,
        delivery_time: Duration,
    ) -> QualityMeasurement;

    /// Get overall streaming quality assessment
    async fn get_overall_quality(&self) -> StreamingQuality;

    /// Check if current quality meets defined thresholds
    async fn meets_thresholds(&self) -> bool;

    /// Get quality trends and predictions
    async fn get_trends(&self) -> QualityTrends;

    /// Generate actionable optimization recommendations
    async fn get_recommendations(&self) -> Vec<OptimizationRecommendation>;
}

#[async_trait]
impl QualityAnalysis for QualityAnalyzer {
    async fn analyze_quality(
        &self,
        chunk: &StreamChunk,
        delivery_time: Duration,
    ) -> QualityMeasurement {
        self.analyze_chunk_quality(chunk, delivery_time).await
    }

    async fn get_overall_quality(&self) -> StreamingQuality {
        self.calculate_overall_quality().await
    }

    async fn meets_thresholds(&self) -> bool {
        self.meets_quality_thresholds().await
    }

    async fn get_trends(&self) -> QualityTrends {
        self.get_quality_trends().await
    }

    async fn get_recommendations(&self) -> Vec<OptimizationRecommendation> {
        self.generate_optimization_recommendations().await
    }
}
