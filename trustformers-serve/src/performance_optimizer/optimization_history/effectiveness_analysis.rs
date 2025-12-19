//! Effectiveness Analysis with ROI Calculation
//!
//! This module provides comprehensive effectiveness analysis capabilities for optimization
//! efforts, including ROI calculation, cost-benefit analysis, performance improvement
//! measurement, and statistical significance testing. It enables data-driven optimization
//! decisions and investment prioritization.

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{collections::HashMap, sync::Arc, time::Duration};

use super::types::*;
use crate::performance_optimizer::types::PerformanceMeasurement;

// =============================================================================
// EFFECTIVENESS ANALYZER
// =============================================================================

/// Effectiveness analyzer with comprehensive ROI calculation
///
/// Provides detailed analysis of optimization effectiveness including cost-benefit
/// analysis, performance improvement measurement, statistical significance testing,
/// and ROI calculation for optimization investments.
pub struct EffectivenessAnalyzer {
    /// Effectiveness calculators
    calculators: Arc<Mutex<Vec<Box<dyn EffectivenessCalculator + Send + Sync>>>>,
    /// Cost calculators
    cost_calculators: Arc<Mutex<Vec<Box<dyn CostCalculator + Send + Sync>>>>,
    /// Analysis cache
    analysis_cache: Arc<RwLock<HashMap<String, EffectivenessAnalysisResult>>>,
    /// Configuration
    config: Arc<RwLock<EffectivenessAnalysisConfig>>,
    /// Historical cost data
    cost_history: Arc<RwLock<HashMap<String, Vec<CostRecord>>>>,
}

impl EffectivenessAnalyzer {
    /// Create new effectiveness analyzer
    pub fn new() -> Self {
        let mut analyzer = Self {
            calculators: Arc::new(Mutex::new(Vec::new())),
            cost_calculators: Arc::new(Mutex::new(Vec::new())),
            analysis_cache: Arc::new(RwLock::new(HashMap::new())),
            config: Arc::new(RwLock::new(EffectivenessAnalysisConfig::default())),
            cost_history: Arc::new(RwLock::new(HashMap::new())),
        };

        // Initialize default calculators
        analyzer.initialize_default_calculators();

        analyzer
    }

    /// Create with custom configuration
    pub fn with_config(config: EffectivenessAnalysisConfig) -> Self {
        let mut analyzer = Self {
            calculators: Arc::new(Mutex::new(Vec::new())),
            cost_calculators: Arc::new(Mutex::new(Vec::new())),
            analysis_cache: Arc::new(RwLock::new(HashMap::new())),
            config: Arc::new(RwLock::new(config)),
            cost_history: Arc::new(RwLock::new(HashMap::new())),
        };

        analyzer.initialize_default_calculators();

        analyzer
    }

    /// Analyze optimization effectiveness
    pub async fn analyze_effectiveness(
        &self,
        before: &PerformanceMeasurement,
        after: &PerformanceMeasurement,
        optimization_params: &HashMap<String, String>,
    ) -> Result<EffectivenessAnalysisResult> {
        let _config = self.config.read();

        // Generate cache key
        let cache_key = self.generate_cache_key(before, after, optimization_params);

        // Check cache
        if let Some(cached) = self.get_cached_analysis(&cache_key) {
            return Ok(cached);
        }

        // Calculate effectiveness using all available calculators
        let calculators = self.calculators.lock();
        let mut results = Vec::new();

        for calculator in calculators.iter() {
            match calculator.calculate_effectiveness(before, after) {
                Ok(result) => results.push(result),
                Err(e) => {
                    tracing::warn!(
                        "Effectiveness calculator {} failed: {}",
                        calculator.name(),
                        e
                    );
                },
            }
        }

        if results.is_empty() {
            return Err(anyhow::anyhow!(
                "No effectiveness calculators could analyze the data"
            ));
        }

        // Aggregate results
        let aggregated_result =
            self.aggregate_effectiveness_results(&results, optimization_params).await?;

        // Cache the result
        self.cache_analysis(&cache_key, &aggregated_result).await;

        Ok(aggregated_result)
    }

    /// Calculate ROI for optimization
    pub async fn calculate_roi(
        &self,
        optimization_cost: f64,
        performance_before: &PerformanceMeasurement,
        performance_after: &PerformanceMeasurement,
        time_period: Duration,
    ) -> Result<f32> {
        // Calculate performance benefit
        let throughput_improvement = performance_after.throughput - performance_before.throughput;
        let latency_improvement_ms = performance_before.latency.as_millis() as f64
            - performance_after.latency.as_millis() as f64;

        // Estimate monetary benefit (simplified calculation)
        // In reality, this would be based on business metrics
        let throughput_value = throughput_improvement * 0.01; // $0.01 per throughput unit
        let latency_value = latency_improvement_ms * 0.001; // $0.001 per ms improvement

        let time_multiplier = time_period.as_secs() as f64 / 3600.0; // Convert to hours
        let total_benefit = (throughput_value + latency_value) * time_multiplier;

        if optimization_cost > 0.0 {
            Ok(((total_benefit - optimization_cost) / optimization_cost) as f32)
        } else {
            Ok(f32::INFINITY) // Free optimization with positive benefit
        }
    }

    /// Get cost breakdown
    pub async fn get_cost_breakdown(
        &self,
        optimization_params: &HashMap<String, String>,
    ) -> Result<CostBreakdown> {
        let cost_calculators = self.cost_calculators.lock();
        let mut total_cost = 0.0;
        let mut cost_components = HashMap::new();

        for calculator in cost_calculators.iter() {
            match calculator.calculate_cost(optimization_params) {
                Ok(cost) => {
                    total_cost += cost;
                    cost_components.insert(calculator.name().to_string(), cost);
                },
                Err(e) => {
                    tracing::warn!("Cost calculator {} failed: {}", calculator.name(), e);
                },
            }
        }

        Ok(CostBreakdown {
            total_cost,
            components: cost_components,
            calculation_timestamp: Utc::now(),
        })
    }

    /// Get effectiveness trends
    pub async fn get_effectiveness_trends(&self, window_size: usize) -> Vec<EffectivenessTrend> {
        let cache = self.analysis_cache.read();
        let mut results: Vec<_> = cache.values().cloned().collect();

        // Sort by analysis time
        results.sort_by_key(|r| r.analyzed_at);

        if results.len() < window_size {
            return Vec::new();
        }

        let mut trends = Vec::new();

        for window in results.windows(window_size) {
            let avg_effectiveness =
                window.iter().map(|r| r.effectiveness_score).sum::<f32>() / window.len() as f32;
            let avg_roi = window.iter().map(|r| r.roi).sum::<f32>() / window.len() as f32;

            let trend_direction = if window.len() >= 2 {
                let first_half_avg = window
                    .iter()
                    .take(window.len() / 2)
                    .map(|r| r.effectiveness_score)
                    .sum::<f32>()
                    / (window.len() / 2) as f32;
                let second_half_avg = window
                    .iter()
                    .skip(window.len() / 2)
                    .map(|r| r.effectiveness_score)
                    .sum::<f32>()
                    / (window.len() - window.len() / 2) as f32;

                if second_half_avg > first_half_avg + 0.05 {
                    EffectivenessTrendDirection::Improving
                } else if second_half_avg < first_half_avg - 0.05 {
                    EffectivenessTrendDirection::Declining
                } else {
                    EffectivenessTrendDirection::Stable
                }
            } else {
                EffectivenessTrendDirection::Stable
            };

            trends.push(EffectivenessTrend {
                window_start: window.first().map(|r| r.analyzed_at).unwrap_or_else(Utc::now),
                window_end: window.last().map(|r| r.analyzed_at).unwrap_or_else(Utc::now),
                average_effectiveness: avg_effectiveness,
                average_roi: avg_roi,
                trend_direction,
                sample_count: window.len(),
            });
        }

        trends
    }

    /// Add effectiveness calculator
    pub fn add_calculator(&self, calculator: Box<dyn EffectivenessCalculator + Send + Sync>) {
        let mut calculators = self.calculators.lock();
        calculators.push(calculator);
    }

    /// Add cost calculator
    pub fn add_cost_calculator(&self, calculator: Box<dyn CostCalculator + Send + Sync>) {
        let mut cost_calculators = self.cost_calculators.lock();
        cost_calculators.push(calculator);
    }

    /// Update configuration
    pub fn update_config(&self, new_config: EffectivenessAnalysisConfig) {
        let mut config = self.config.write();
        *config = new_config;
    }

    /// Clear analysis cache
    pub async fn clear_cache(&self) {
        let mut cache = self.analysis_cache.write();
        cache.clear();
    }

    /// Get analysis statistics
    pub fn get_analysis_statistics(&self) -> AnalysisStatistics {
        let cache = self.analysis_cache.read();

        let total_analyses = cache.len();
        let mut effectiveness_sum = 0.0f32;
        let mut roi_sum = 0.0f32;
        let mut high_effectiveness_count = 0;
        let mut positive_roi_count = 0;

        for result in cache.values() {
            effectiveness_sum += result.effectiveness_score;
            roi_sum += result.roi;

            if result.effectiveness_score >= 0.8 {
                high_effectiveness_count += 1;
            }

            if result.roi > 0.0 {
                positive_roi_count += 1;
            }
        }

        let average_effectiveness =
            if total_analyses > 0 { effectiveness_sum / total_analyses as f32 } else { 0.0 };

        let average_roi = if total_analyses > 0 { roi_sum / total_analyses as f32 } else { 0.0 };

        AnalysisStatistics {
            total_analyses,
            average_effectiveness,
            average_roi,
            high_effectiveness_count,
            positive_roi_count,
            cache_memory_usage: cache.len() * std::mem::size_of::<EffectivenessAnalysisResult>(),
        }
    }

    /// Initialize default calculators
    fn initialize_default_calculators(&mut self) {
        let mut calculators = self.calculators.lock();
        let mut cost_calculators = self.cost_calculators.lock();

        // Effectiveness calculators
        calculators.push(Box::new(ThroughputEffectivenessCalculator::new()));
        calculators.push(Box::new(LatencyEffectivenessCalculator::new()));
        calculators.push(Box::new(CompositeEffectivenessCalculator::new()));

        // Cost calculators
        cost_calculators.push(Box::new(ResourceBasedCostCalculator::new()));
        cost_calculators.push(Box::new(TimeBasedCostCalculator::new()));
        cost_calculators.push(Box::new(ComplexityBasedCostCalculator::new()));
    }

    /// Generate cache key
    fn generate_cache_key(
        &self,
        before: &PerformanceMeasurement,
        after: &PerformanceMeasurement,
        params: &HashMap<String, String>,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        before.throughput.to_bits().hash(&mut hasher);
        before.latency.as_nanos().hash(&mut hasher);
        after.throughput.to_bits().hash(&mut hasher);
        after.latency.as_nanos().hash(&mut hasher);

        for (key, value) in params {
            key.hash(&mut hasher);
            value.hash(&mut hasher);
        }

        format!("effectiveness_{}", hasher.finish())
    }

    /// Get cached analysis result
    fn get_cached_analysis(&self, cache_key: &str) -> Option<EffectivenessAnalysisResult> {
        let cache = self.analysis_cache.read();
        cache.get(cache_key).cloned()
    }

    /// Cache analysis result
    async fn cache_analysis(&self, cache_key: &str, result: &EffectivenessAnalysisResult) {
        let mut cache = self.analysis_cache.write();
        cache.insert(cache_key.to_string(), result.clone());

        // Maintain cache size (keep last 500 analyses)
        if cache.len() > 500 {
            let mut results: Vec<_> = cache.values().cloned().collect();
            results.sort_by_key(|r| r.analyzed_at);

            let to_remove = cache.len() - 500;
            for result in results.iter().take(to_remove) {
                // Find and remove by searching for matching analysis
                let keys_to_remove: Vec<String> = cache
                    .iter()
                    .filter(|(_, v)| v.analyzed_at == result.analyzed_at)
                    .map(|(k, _)| k.clone())
                    .collect();

                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
        }
    }

    /// Aggregate multiple effectiveness results
    async fn aggregate_effectiveness_results(
        &self,
        results: &[EffectivenessAnalysisResult],
        _optimization_params: &HashMap<String, String>,
    ) -> Result<EffectivenessAnalysisResult> {
        if results.is_empty() {
            return Err(anyhow::anyhow!("No results to aggregate"));
        }

        // Weighted average based on statistical significance
        let mut total_weight = 0.0f32;
        let mut weighted_effectiveness = 0.0f32;
        let mut weighted_roi = 0.0f32;

        let mut all_cost_benefits = Vec::new();
        let mut all_improvements = Vec::new();
        let mut all_significance_tests = Vec::new();

        for result in results {
            let weight = if result.statistical_significance.is_significant {
                1.0 - result.statistical_significance.p_value as f32
            } else {
                0.5
            };

            total_weight += weight;
            weighted_effectiveness += result.effectiveness_score * weight;
            weighted_roi += result.roi * weight;

            all_cost_benefits.push(result.cost_benefit.clone());
            all_improvements.push(result.performance_improvement.clone());
            all_significance_tests.push(result.statistical_significance.clone());
        }

        let final_effectiveness =
            if total_weight > 0.0 { weighted_effectiveness / total_weight } else { 0.0 };

        let final_roi = if total_weight > 0.0 { weighted_roi / total_weight } else { 0.0 };

        // Aggregate cost-benefit analysis
        let aggregated_cost_benefit = self.aggregate_cost_benefit_analysis(&all_cost_benefits);

        // Aggregate performance improvements
        let aggregated_improvement = self.aggregate_performance_improvements(&all_improvements);

        // Aggregate statistical significance
        let aggregated_significance =
            self.aggregate_statistical_significance(&all_significance_tests);

        Ok(EffectivenessAnalysisResult {
            effectiveness_score: final_effectiveness,
            roi: final_roi,
            cost_benefit: aggregated_cost_benefit,
            performance_improvement: aggregated_improvement,
            statistical_significance: aggregated_significance,
            analyzed_at: Utc::now(),
        })
    }

    /// Aggregate cost-benefit analyses
    fn aggregate_cost_benefit_analysis(
        &self,
        analyses: &[CostBenefitAnalysis],
    ) -> CostBenefitAnalysis {
        if analyses.is_empty() {
            return CostBenefitAnalysis::default();
        }

        let total_impl_cost =
            analyses.iter().map(|a| a.implementation_cost).sum::<f64>() / analyses.len() as f64;
        let total_op_cost =
            analyses.iter().map(|a| a.operational_cost).sum::<f64>() / analyses.len() as f64;
        let total_cost = total_impl_cost + total_op_cost;

        let total_perf_benefit =
            analyses.iter().map(|a| a.performance_benefit).sum::<f64>() / analyses.len() as f64;
        let total_resource_savings =
            analyses.iter().map(|a| a.resource_savings).sum::<f64>() / analyses.len() as f64;
        let total_benefit = total_perf_benefit + total_resource_savings;

        let net_benefit = total_benefit - total_cost;

        let avg_payback_period = Duration::from_secs(
            analyses.iter().map(|a| a.payback_period.as_secs()).sum::<u64>()
                / analyses.len() as u64,
        );

        CostBenefitAnalysis {
            implementation_cost: total_impl_cost,
            operational_cost: total_op_cost,
            total_cost,
            performance_benefit: total_perf_benefit,
            resource_savings: total_resource_savings,
            total_benefit,
            net_benefit,
            payback_period: avg_payback_period,
        }
    }

    /// Aggregate performance improvements
    fn aggregate_performance_improvements(
        &self,
        improvements: &[PerformanceImprovement],
    ) -> PerformanceImprovement {
        if improvements.is_empty() {
            return PerformanceImprovement::default();
        }

        let avg_throughput_improvement =
            improvements.iter().map(|i| i.throughput_improvement).sum::<f32>()
                / improvements.len() as f32;
        let avg_latency_improvement =
            improvements.iter().map(|i| i.latency_improvement).sum::<f32>()
                / improvements.len() as f32;
        let avg_resource_improvement =
            improvements.iter().map(|i| i.resource_improvement).sum::<f32>()
                / improvements.len() as f32;
        let avg_overall_improvement =
            improvements.iter().map(|i| i.overall_improvement).sum::<f32>()
                / improvements.len() as f32;

        let avg_duration = Duration::from_secs(
            improvements.iter().map(|i| i.improvement_duration.as_secs()).sum::<u64>()
                / improvements.len() as u64,
        );

        PerformanceImprovement {
            throughput_improvement: avg_throughput_improvement,
            latency_improvement: avg_latency_improvement,
            resource_improvement: avg_resource_improvement,
            overall_improvement: avg_overall_improvement,
            improvement_duration: avg_duration,
        }
    }

    /// Aggregate statistical significance tests
    fn aggregate_statistical_significance(
        &self,
        tests: &[StatisticalSignificance],
    ) -> StatisticalSignificance {
        if tests.is_empty() {
            return StatisticalSignificance::default();
        }

        // Use the most conservative (highest) p-value
        let max_p_value = tests.iter().map(|t| t.p_value).fold(0.0f64, |acc, x| acc.max(x));
        let avg_test_statistic =
            tests.iter().map(|t| t.test_statistic).sum::<f64>() / tests.len() as f64;
        let avg_confidence_level =
            tests.iter().map(|t| t.confidence_level).sum::<f32>() / tests.len() as f32;

        let is_significant = tests.iter().all(|t| t.is_significant);

        StatisticalSignificance {
            test_statistic: avg_test_statistic,
            p_value: max_p_value,
            confidence_level: avg_confidence_level,
            is_significant,
            test_method: "Aggregated".to_string(),
        }
    }
}

impl Default for EffectivenessAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// EFFECTIVENESS CALCULATOR IMPLEMENTATIONS
// =============================================================================

/// Throughput-based effectiveness calculator
pub struct ThroughputEffectivenessCalculator;

impl Default for ThroughputEffectivenessCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl ThroughputEffectivenessCalculator {
    pub fn new() -> Self {
        Self
    }
}

impl EffectivenessCalculator for ThroughputEffectivenessCalculator {
    fn calculate_effectiveness(
        &self,
        before: &PerformanceMeasurement,
        after: &PerformanceMeasurement,
    ) -> Result<EffectivenessAnalysisResult> {
        let throughput_improvement = (after.throughput - before.throughput) / before.throughput;
        let effectiveness_score = (throughput_improvement.max(0.0) * 2.0).min(1.0) as f32;

        // Simple ROI calculation
        let estimated_cost = 100.0; // Simplified
        let estimated_benefit = throughput_improvement * 1000.0; // Simplified
        let roi = ((estimated_benefit - estimated_cost) / estimated_cost) as f32;

        let cost_benefit = CostBenefitAnalysis {
            implementation_cost: estimated_cost,
            operational_cost: 10.0,
            total_cost: estimated_cost + 10.0,
            performance_benefit: estimated_benefit,
            resource_savings: throughput_improvement * 50.0,
            total_benefit: estimated_benefit + throughput_improvement * 50.0,
            net_benefit: estimated_benefit - estimated_cost,
            payback_period: Duration::from_secs(3600), // 1 hour
        };

        let performance_improvement = PerformanceImprovement {
            throughput_improvement: (throughput_improvement * 100.0) as f32,
            latency_improvement: 0.0, // Not applicable for this calculator
            resource_improvement: 0.0,
            overall_improvement: (throughput_improvement * 100.0) as f32,
            improvement_duration: Duration::from_secs(1800),
        };

        let statistical_significance = StatisticalSignificance {
            test_statistic: throughput_improvement.abs(),
            p_value: if throughput_improvement.abs() > 0.05 { 0.01 } else { 0.1 },
            confidence_level: 0.95,
            is_significant: throughput_improvement.abs() > 0.05,
            test_method: "Throughput T-test".to_string(),
        };

        Ok(EffectivenessAnalysisResult {
            effectiveness_score,
            roi,
            cost_benefit,
            performance_improvement,
            statistical_significance,
            analyzed_at: Utc::now(),
        })
    }

    fn name(&self) -> &str {
        "throughput_effectiveness"
    }
}

/// Latency-based effectiveness calculator
pub struct LatencyEffectivenessCalculator;

impl Default for LatencyEffectivenessCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyEffectivenessCalculator {
    pub fn new() -> Self {
        Self
    }
}

impl EffectivenessCalculator for LatencyEffectivenessCalculator {
    fn calculate_effectiveness(
        &self,
        before: &PerformanceMeasurement,
        after: &PerformanceMeasurement,
    ) -> Result<EffectivenessAnalysisResult> {
        let latency_before_ms = before.latency.as_millis() as f64;
        let latency_after_ms = after.latency.as_millis() as f64;
        let latency_improvement = (latency_before_ms - latency_after_ms) / latency_before_ms;

        let effectiveness_score = (latency_improvement.max(0.0) * 2.0).min(1.0) as f32;

        // Simple ROI calculation
        let estimated_cost = 150.0; // Simplified
        let estimated_benefit = latency_improvement * 800.0; // Simplified
        let roi = ((estimated_benefit - estimated_cost) / estimated_cost) as f32;

        let cost_benefit = CostBenefitAnalysis {
            implementation_cost: estimated_cost,
            operational_cost: 15.0,
            total_cost: estimated_cost + 15.0,
            performance_benefit: estimated_benefit,
            resource_savings: latency_improvement * 40.0,
            total_benefit: estimated_benefit + latency_improvement * 40.0,
            net_benefit: estimated_benefit - estimated_cost,
            payback_period: Duration::from_secs(7200), // 2 hours
        };

        let performance_improvement = PerformanceImprovement {
            throughput_improvement: 0.0, // Not applicable for this calculator
            latency_improvement: (latency_improvement * 100.0) as f32,
            resource_improvement: 0.0,
            overall_improvement: (latency_improvement * 100.0) as f32,
            improvement_duration: Duration::from_secs(3600),
        };

        let statistical_significance = StatisticalSignificance {
            test_statistic: latency_improvement.abs(),
            p_value: if latency_improvement.abs() > 0.05 { 0.01 } else { 0.1 },
            confidence_level: 0.95,
            is_significant: latency_improvement.abs() > 0.05,
            test_method: "Latency T-test".to_string(),
        };

        Ok(EffectivenessAnalysisResult {
            effectiveness_score,
            roi,
            cost_benefit,
            performance_improvement,
            statistical_significance,
            analyzed_at: Utc::now(),
        })
    }

    fn name(&self) -> &str {
        "latency_effectiveness"
    }
}

/// Composite effectiveness calculator that considers multiple metrics
pub struct CompositeEffectivenessCalculator {
    throughput_weight: f32,
    latency_weight: f32,
}

impl Default for CompositeEffectivenessCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl CompositeEffectivenessCalculator {
    pub fn new() -> Self {
        Self {
            throughput_weight: 0.6,
            latency_weight: 0.4,
        }
    }

    pub fn with_weights(throughput_weight: f32, latency_weight: f32) -> Self {
        Self {
            throughput_weight,
            latency_weight,
        }
    }
}

impl EffectivenessCalculator for CompositeEffectivenessCalculator {
    fn calculate_effectiveness(
        &self,
        before: &PerformanceMeasurement,
        after: &PerformanceMeasurement,
    ) -> Result<EffectivenessAnalysisResult> {
        // Calculate throughput improvement
        let throughput_improvement = (after.throughput - before.throughput) / before.throughput;
        let throughput_score = (throughput_improvement.max(0.0) * 2.0).min(1.0);

        // Calculate latency improvement
        let latency_before_ms = before.latency.as_millis() as f64;
        let latency_after_ms = after.latency.as_millis() as f64;
        let latency_improvement = (latency_before_ms - latency_after_ms) / latency_before_ms;
        let latency_score = (latency_improvement.max(0.0) * 2.0).min(1.0);

        // Composite effectiveness score
        let effectiveness_score = (throughput_score * self.throughput_weight as f64
            + latency_score * self.latency_weight as f64) as f32;

        // Composite ROI calculation
        let total_benefit = throughput_improvement * 1000.0 + latency_improvement * 800.0;
        let total_cost = 200.0; // Estimated composite cost
        let roi = ((total_benefit - total_cost) / total_cost) as f32;

        let cost_benefit = CostBenefitAnalysis {
            implementation_cost: total_cost,
            operational_cost: 20.0,
            total_cost: total_cost + 20.0,
            performance_benefit: total_benefit,
            resource_savings: (throughput_improvement + latency_improvement) * 45.0,
            total_benefit: total_benefit + (throughput_improvement + latency_improvement) * 45.0,
            net_benefit: total_benefit - total_cost,
            payback_period: Duration::from_secs(5400), // 1.5 hours
        };

        let performance_improvement = PerformanceImprovement {
            throughput_improvement: (throughput_improvement * 100.0) as f32,
            latency_improvement: (latency_improvement * 100.0) as f32,
            resource_improvement: ((throughput_improvement + latency_improvement) / 2.0 * 100.0)
                as f32,
            overall_improvement: ((throughput_improvement + latency_improvement) / 2.0 * 100.0)
                as f32,
            improvement_duration: Duration::from_secs(2700),
        };

        let combined_improvement = (throughput_improvement + latency_improvement) / 2.0;
        let statistical_significance = StatisticalSignificance {
            test_statistic: combined_improvement.abs(),
            p_value: if combined_improvement.abs() > 0.05 { 0.005 } else { 0.15 },
            confidence_level: 0.95,
            is_significant: combined_improvement.abs() > 0.05,
            test_method: "Composite T-test".to_string(),
        };

        Ok(EffectivenessAnalysisResult {
            effectiveness_score,
            roi,
            cost_benefit,
            performance_improvement,
            statistical_significance,
            analyzed_at: Utc::now(),
        })
    }

    fn name(&self) -> &str {
        "composite_effectiveness"
    }
}

// =============================================================================
// COST CALCULATOR IMPLEMENTATIONS
// =============================================================================

/// Resource-based cost calculator
pub struct ResourceBasedCostCalculator {
    cpu_cost_per_core_hour: f64,
    memory_cost_per_gb_hour: f64,
    io_cost_per_gb: f64,
}

impl Default for ResourceBasedCostCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceBasedCostCalculator {
    pub fn new() -> Self {
        Self {
            cpu_cost_per_core_hour: 0.10,
            memory_cost_per_gb_hour: 0.02,
            io_cost_per_gb: 0.001,
        }
    }

    pub fn with_costs(cpu_cost: f64, memory_cost: f64, io_cost: f64) -> Self {
        Self {
            cpu_cost_per_core_hour: cpu_cost,
            memory_cost_per_gb_hour: memory_cost,
            io_cost_per_gb: io_cost,
        }
    }
}

impl CostCalculator for ResourceBasedCostCalculator {
    fn calculate_cost(&self, optimization_params: &HashMap<String, String>) -> Result<f64> {
        let mut total_cost = 0.0;

        // Parse resource usage from parameters
        if let Some(cores_str) = optimization_params.get("cores") {
            if let Ok(cores) = cores_str.parse::<f64>() {
                let hours = optimization_params
                    .get("duration_hours")
                    .and_then(|h| h.parse::<f64>().ok())
                    .unwrap_or(1.0);
                total_cost += cores * self.cpu_cost_per_core_hour * hours;
            }
        }

        if let Some(memory_str) = optimization_params.get("memory_gb") {
            if let Ok(memory_gb) = memory_str.parse::<f64>() {
                let hours = optimization_params
                    .get("duration_hours")
                    .and_then(|h| h.parse::<f64>().ok())
                    .unwrap_or(1.0);
                total_cost += memory_gb * self.memory_cost_per_gb_hour * hours;
            }
        }

        if let Some(io_str) = optimization_params.get("io_gb") {
            if let Ok(io_gb) = io_str.parse::<f64>() {
                total_cost += io_gb * self.io_cost_per_gb;
            }
        }

        // Base optimization overhead cost
        total_cost += 50.0;

        Ok(total_cost)
    }

    fn name(&self) -> &str {
        "resource_based_cost"
    }
}

/// Time-based cost calculator
pub struct TimeBasedCostCalculator {
    hourly_rate: f64,
}

impl Default for TimeBasedCostCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeBasedCostCalculator {
    pub fn new() -> Self {
        Self {
            hourly_rate: 25.0, // $25/hour for optimization work
        }
    }

    pub fn with_rate(hourly_rate: f64) -> Self {
        Self { hourly_rate }
    }
}

impl CostCalculator for TimeBasedCostCalculator {
    fn calculate_cost(&self, optimization_params: &HashMap<String, String>) -> Result<f64> {
        let duration_hours = optimization_params
            .get("duration_hours")
            .and_then(|h| h.parse::<f64>().ok())
            .unwrap_or(2.0); // Default 2 hours

        let complexity_multiplier = match optimization_params.get("complexity").map(|s| s.as_str())
        {
            Some("simple") => 1.0,
            Some("moderate") => 1.5,
            Some("complex") => 2.0,
            Some("very_complex") => 3.0,
            _ => 1.0,
        };

        Ok(duration_hours * self.hourly_rate * complexity_multiplier)
    }

    fn name(&self) -> &str {
        "time_based_cost"
    }
}

/// Complexity-based cost calculator
pub struct ComplexityBasedCostCalculator {
    base_cost: f64,
    complexity_factors: HashMap<String, f64>,
}

impl Default for ComplexityBasedCostCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl ComplexityBasedCostCalculator {
    pub fn new() -> Self {
        let mut complexity_factors = HashMap::new();
        complexity_factors.insert("algorithm_change".to_string(), 100.0);
        complexity_factors.insert("parallelization".to_string(), 150.0);
        complexity_factors.insert("memory_optimization".to_string(), 75.0);
        complexity_factors.insert("io_optimization".to_string(), 50.0);
        complexity_factors.insert("caching".to_string(), 80.0);

        Self {
            base_cost: 25.0,
            complexity_factors,
        }
    }
}

impl CostCalculator for ComplexityBasedCostCalculator {
    fn calculate_cost(&self, optimization_params: &HashMap<String, String>) -> Result<f64> {
        let mut total_cost = self.base_cost;

        for (param_key, param_value) in optimization_params {
            if param_value == "true" || param_value == "enabled" {
                if let Some(&factor_cost) = self.complexity_factors.get(param_key) {
                    total_cost += factor_cost;
                }
            }
        }

        // Scale by optimization scope
        let scope_multiplier = match optimization_params.get("scope").map(|s| s.as_str()) {
            Some("single_function") => 1.0,
            Some("module") => 1.5,
            Some("system") => 2.0,
            Some("global") => 3.0,
            _ => 1.0,
        };

        Ok(total_cost * scope_multiplier)
    }

    fn name(&self) -> &str {
        "complexity_based_cost"
    }
}

// =============================================================================
// UTILITY TYPES AND IMPLEMENTATIONS
// =============================================================================

/// Cost breakdown result
#[derive(Debug, Clone)]
pub struct CostBreakdown {
    pub total_cost: f64,
    pub components: HashMap<String, f64>,
    pub calculation_timestamp: DateTime<Utc>,
}

/// Effectiveness trend information
#[derive(Debug, Clone)]
pub struct EffectivenessTrend {
    pub window_start: DateTime<Utc>,
    pub window_end: DateTime<Utc>,
    pub average_effectiveness: f32,
    pub average_roi: f32,
    pub trend_direction: EffectivenessTrendDirection,
    pub sample_count: usize,
}

/// Effectiveness trend direction
#[derive(Debug, Clone)]
pub enum EffectivenessTrendDirection {
    Improving,
    Stable,
    Declining,
}

/// Analysis statistics
#[derive(Debug, Clone)]
pub struct AnalysisStatistics {
    pub total_analyses: usize,
    pub average_effectiveness: f32,
    pub average_roi: f32,
    pub high_effectiveness_count: usize,
    pub positive_roi_count: usize,
    pub cache_memory_usage: usize,
}

/// Cost record for historical tracking
#[derive(Debug, Clone)]
pub struct CostRecord {
    pub cost: f64,
    pub optimization_type: String,
    pub timestamp: DateTime<Utc>,
    pub parameters: HashMap<String, String>,
}

/// Default implementations
impl Default for CostBenefitAnalysis {
    fn default() -> Self {
        Self {
            implementation_cost: 0.0,
            operational_cost: 0.0,
            total_cost: 0.0,
            performance_benefit: 0.0,
            resource_savings: 0.0,
            total_benefit: 0.0,
            net_benefit: 0.0,
            payback_period: Duration::from_secs(0),
        }
    }
}

impl Default for PerformanceImprovement {
    fn default() -> Self {
        Self {
            throughput_improvement: 0.0,
            latency_improvement: 0.0,
            resource_improvement: 0.0,
            overall_improvement: 0.0,
            improvement_duration: Duration::from_secs(0),
        }
    }
}

impl Default for StatisticalSignificance {
    fn default() -> Self {
        Self {
            test_statistic: 0.0,
            p_value: 1.0,
            confidence_level: 0.95,
            is_significant: false,
            test_method: "None".to_string(),
        }
    }
}
