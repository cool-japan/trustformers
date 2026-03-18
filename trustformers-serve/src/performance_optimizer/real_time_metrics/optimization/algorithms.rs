//! Base Optimization Algorithms
//!
//! Multiple strategies for different optimization scenarios

use anyhow::Result;
use chrono::{DateTime, Utc};
use std::{
    collections::{HashMap, VecDeque},
    time::Duration,
};

use super::super::types::*;

// =============================================================================

/// Parallelism optimization algorithm
///
/// Optimizes system parallelism based on CPU utilization, core availability,
/// and workload characteristics to maximize throughput while minimizing contention.
pub struct ParallelismOptimizationAlgorithm {
    /// Algorithm statistics
    stats: AlgorithmStatistics,

    /// Historical data for trend analysis
    historical_data: VecDeque<ParallelismMetrics>,

    /// Configuration parameters
    config: ParallelismConfig,
}

#[derive(Debug, Clone)]
struct ParallelismMetrics {
    timestamp: DateTime<Utc>,
    cpu_utilization: f32,
    throughput: f64,
    parallelism_level: usize,
    contention_score: f32,
}

#[derive(Debug, Clone)]
struct ParallelismConfig {
    min_parallelism: usize,
    max_parallelism: usize,
    utilization_threshold_low: f32,
    utilization_threshold_high: f32,
    throughput_threshold: f64,
}

impl Default for ParallelismOptimizationAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelismOptimizationAlgorithm {
    pub fn new() -> Self {
        Self {
            stats: AlgorithmStatistics::default(),
            historical_data: VecDeque::new(),
            config: ParallelismConfig {
                min_parallelism: 1,
                max_parallelism: num_cpus::get() * 2,
                utilization_threshold_low: 0.3,
                utilization_threshold_high: 0.85,
                throughput_threshold: 50.0,
            },
        }
    }

    fn calculate_optimal_parallelism(
        &self,
        cpu_utilization: f32,
        throughput: f64,
        context: &OptimizationContext,
    ) -> usize {
        let available_cores = context.system_state.available_cores;
        // TODO: SystemState no longer has current_parallelism field
        // Using available_cores as placeholder; should track actual parallelism level
        let current_parallelism = available_cores;

        // Analyze historical trends
        let trend_factor = self.analyze_parallelism_trends();

        // Calculate base recommendation
        let mut optimal_parallelism = if cpu_utilization < self.config.utilization_threshold_low
            && throughput < self.config.throughput_threshold
        {
            // Low utilization, consider increasing parallelism
            ((current_parallelism as f32 * 1.5) as usize).min(available_cores * 2)
        } else if cpu_utilization > self.config.utilization_threshold_high {
            // High utilization, consider decreasing parallelism
            ((current_parallelism as f32 * 0.8) as usize).max(1)
        } else {
            current_parallelism
        };

        // Apply trend factor
        optimal_parallelism = ((optimal_parallelism as f32 * trend_factor) as usize)
            .max(self.config.min_parallelism)
            .min(self.config.max_parallelism);

        optimal_parallelism
    }

    fn analyze_parallelism_trends(&self) -> f32 {
        if self.historical_data.len() < 3 {
            return 1.0;
        }

        let recent_data: Vec<_> = self.historical_data.iter().rev().take(5).collect();
        let throughput_trend = self.calculate_throughput_trend(&recent_data);
        let utilization_trend = self.calculate_utilization_trend(&recent_data);

        // Combine trends to determine adjustment factor
        match (throughput_trend > 0.0, utilization_trend < 0.9) {
            (true, true) => 1.1,   // Increasing throughput, manageable utilization
            (false, false) => 0.9, // Decreasing throughput, high utilization
            _ => 1.0,              // Stable or mixed signals
        }
    }

    fn calculate_throughput_trend(&self, data: &[&ParallelismMetrics]) -> f32 {
        if data.len() < 2 {
            return 0.0;
        }

        let recent_avg = data[..2].iter().map(|d| d.throughput).sum::<f64>() / 2.0;
        let older_avg =
            data[2..].iter().map(|d| d.throughput).sum::<f64>() / (data.len() - 2) as f64;

        ((recent_avg - older_avg) / older_avg) as f32
    }

    fn calculate_utilization_trend(&self, data: &[&ParallelismMetrics]) -> f32 {
        if data.is_empty() {
            return 0.5;
        }

        data.iter().map(|d| d.cpu_utilization).sum::<f32>() / data.len() as f32
    }
}

impl LiveOptimizationAlgorithm for ParallelismOptimizationAlgorithm {
    fn optimize(
        &self,
        metrics: &RealTimeMetrics,
        history: &[TimestampedMetrics],
        context: &OptimizationContext,
    ) -> Result<Vec<OptimizationRecommendation>, RealTimeMetricsError> {
        let mut recommendations = Vec::new();

        // Analyze current parallelism efficiency
        let cpu_utilization = metrics.current_cpu_utilization;
        let throughput = metrics.current_throughput;
        // TODO: SystemState no longer has current_parallelism field
        let current_parallelism = context.system_state.available_cores;

        let optimal_parallelism =
            self.calculate_optimal_parallelism(cpu_utilization, throughput, context);

        if optimal_parallelism != current_parallelism {
            let _recommendation_type = if optimal_parallelism > current_parallelism {
                RecommendationType::IncreaseParallelism {
                    target_level: optimal_parallelism,
                }
            } else {
                RecommendationType::DecreaseParallelism {
                    target_level: optimal_parallelism,
                }
            };

            let confidence = self.calculate_confidence(cpu_utilization, throughput, history);
            let impact = self.estimate_impact(current_parallelism, optimal_parallelism, metrics);

            let action_type = if optimal_parallelism > current_parallelism {
                ActionType::IncreaseParallelism
            } else {
                ActionType::DecreaseParallelism
            };

            recommendations.push(OptimizationRecommendation {
                id: format!("parallelism_{}", Utc::now().timestamp()),
                timestamp: Utc::now(),
                actions: vec![RecommendedAction {
                    action_type,
                    parameters: {
                        let mut params = HashMap::new();
                        params.insert(
                            "target_parallelism".to_string(),
                            optimal_parallelism.to_string(),
                        );
                        params.insert(
                            "current_parallelism".to_string(),
                            current_parallelism.to_string(),
                        );
                        params.insert("cpu_utilization".to_string(), cpu_utilization.to_string());
                        params.insert("throughput".to_string(), throughput.to_string());
                        params
                    },
                    priority: if (optimal_parallelism as i32 - current_parallelism as i32).abs() > 2
                    {
                        1.0
                    } else {
                        2.0
                    },
                    expected_impact: impact.performance_impact,
                    estimated_duration: Duration::from_secs(30),
                    reversible: true,
                }],
                expected_impact: impact,
                confidence,
                analysis: format!(
                    "Parallelism optimization: Current={}, Optimal={}, CPU={}%, Throughput={}",
                    current_parallelism,
                    optimal_parallelism,
                    cpu_utilization * 100.0,
                    throughput
                ),
                risks: self.assess_parallelism_risks(current_parallelism, optimal_parallelism),
                priority: 1,
                implementation_time: Duration::from_secs(30),
            });
        }

        Ok(recommendations)
    }

    fn name(&self) -> &str {
        "parallelism_optimization"
    }

    fn confidence(&self, data_quality: f32) -> f32 {
        data_quality * 0.9 * if self.historical_data.len() >= 5 { 1.0 } else { 0.8 }
    }

    fn is_applicable(&self, context: &OptimizationContext) -> bool {
        context.system_state.available_cores > 1
    }

    fn update_with_feedback(
        &mut self,
        _feedback: &PerformanceFeedback,
    ) -> Result<(), RealTimeMetricsError> {
        // TODO: AlgorithmStatistics no longer has feedback_count, positive_feedback, negative_feedback fields
        // Need to implement feedback tracking differently or add these fields back to AlgorithmStatistics
        // self.stats.feedback_count += 1;
        // match feedback.feedback_type {
        //     FeedbackType::Positive => self.stats.positive_feedback += 1,
        //     FeedbackType::Negative => self.stats.negative_feedback += 1,
        //     FeedbackType::Neutral => {},
        // }

        Ok(())
    }

    fn statistics(&self) -> AlgorithmStatistics {
        self.stats.clone()
    }
}

impl ParallelismOptimizationAlgorithm {
    fn calculate_confidence(
        &self,
        cpu_utilization: f32,
        throughput: f64,
        history: &[TimestampedMetrics],
    ) -> f32 {
        let base_confidence = 0.8;

        // Adjust based on data quality
        let data_quality = if history.len() >= 10 { 1.0 } else { 0.7 };

        // Adjust based on utilization stability
        let utilization_stability =
            if cpu_utilization > 0.1 && cpu_utilization < 0.95 { 1.0 } else { 0.8 };

        // Adjust based on throughput
        let throughput_factor = if throughput > 10.0 { 1.0 } else { 0.9 };

        base_confidence * data_quality * utilization_stability * throughput_factor
    }

    fn estimate_impact(
        &self,
        current: usize,
        optimal: usize,
        metrics: &RealTimeMetrics,
    ) -> ImpactAssessment {
        let change_ratio = optimal as f32 / current.max(1) as f32;

        let performance_impact = if change_ratio > 1.0 {
            // Increasing parallelism
            ((change_ratio - 1.0) * 0.3).min(0.5)
        } else {
            // Decreasing parallelism
            ((1.0 - change_ratio) * -0.2).max(-0.3)
        };

        let resource_impact = (change_ratio - 1.0) * 0.1;
        let complexity = if (optimal as i32 - current as i32).abs() > 2 { 0.6 } else { 0.3 };

        ImpactAssessment {
            performance_impact,
            resource_impact,
            complexity,
            risk_level: if !(0.5..=2.0).contains(&change_ratio) { 0.7 } else { 0.3 },
            estimated_benefit: performance_impact.abs() * metrics.current_throughput as f32 / 100.0,
            implementation_time: Duration::from_secs(30),
        }
    }

    fn assess_parallelism_risks(&self, current: usize, optimal: usize) -> Vec<RiskFactor> {
        let mut risks = Vec::new();

        if optimal > current * 2 {
            risks.push(RiskFactor {
                risk_type: "PerformanceRegression".to_string(),
                description: "Dramatic parallelism increase may cause thread contention"
                    .to_string(),
                probability: 0.3,
                impact: 0.6,
                severity: SeverityLevel::Medium,
                mitigation: vec![
                    "Monitor CPU utilization and rollback if contention increases".to_string(),
                ],
            });
        }

        if optimal < current / 2 {
            risks.push(RiskFactor {
                risk_type: "ResourceWaste".to_string(),
                description: "Significant parallelism reduction may underutilize available cores"
                    .to_string(),
                probability: 0.2,
                impact: 0.4,
                severity: SeverityLevel::Low,
                mitigation: vec![
                    "Monitor throughput and adjust if underperformance detected".to_string()
                ],
            });
        }

        risks
    }
}

/// Resource optimization algorithm
///
/// Optimizes system resource allocation including memory, CPU affinity,
/// and I/O scheduling to improve overall system efficiency.
pub struct ResourceOptimizationAlgorithm {
    /// Algorithm statistics
    stats: AlgorithmStatistics,

    /// Resource utilization history
    resource_history: VecDeque<ResourceSnapshot>,

    /// Configuration parameters
    config: ResourceConfig,
}

#[derive(Debug, Clone)]
struct ResourceSnapshot {
    timestamp: DateTime<Utc>,
    memory_utilization: f32,
    cpu_utilization: f32,
    io_utilization: f32,
    network_utilization: f32,
}

#[derive(Debug, Clone)]
struct ResourceConfig {
    memory_threshold_critical: f32,
    memory_threshold_warning: f32,
    cpu_threshold_high: f32,
    io_threshold_high: f32,
    optimization_interval: Duration,
}

impl Default for ResourceOptimizationAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceOptimizationAlgorithm {
    pub fn new() -> Self {
        Self {
            stats: AlgorithmStatistics::default(),
            resource_history: VecDeque::new(),
            config: ResourceConfig {
                memory_threshold_critical: 0.9,
                memory_threshold_warning: 0.8,
                cpu_threshold_high: 0.85,
                io_threshold_high: 0.8,
                optimization_interval: Duration::from_secs(60),
            },
        }
    }

    fn analyze_memory_pressure(&self, metrics: &RealTimeMetrics) -> Option<RecommendedAction> {
        let memory_util = metrics.current_memory_utilization;

        if memory_util > self.config.memory_threshold_critical {
            Some(RecommendedAction {
                action_type: ActionType::AdjustResourceAllocation,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("resource_type".to_string(), "memory".to_string());
                    params.insert("action".to_string(), "increase_limit".to_string());
                    params.insert("current_utilization".to_string(), memory_util.to_string());
                    params.insert("recommended_increase".to_string(), "20%".to_string());
                    params
                },
                priority: 1.0,
                expected_impact: 0.8, // High impact for critical memory issues
                estimated_duration: Duration::from_secs(60),
                reversible: true,
            })
        } else if memory_util > self.config.memory_threshold_warning {
            Some(RecommendedAction {
                action_type: ActionType::TuneParameters,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("strategy".to_string(), "garbage_collection".to_string());
                    params.insert("current_utilization".to_string(), memory_util.to_string());
                    params
                },
                priority: 2.0,
                expected_impact: 0.5, // Medium impact for memory warning
                estimated_duration: Duration::from_secs(30),
                reversible: true,
            })
        } else {
            None
        }
    }

    fn analyze_cpu_affinity(
        &self,
        metrics: &RealTimeMetrics,
        context: &OptimizationContext,
    ) -> Option<RecommendedAction> {
        let cpu_util = metrics.current_cpu_utilization;

        if cpu_util > self.config.cpu_threshold_high && context.system_state.available_cores > 2 {
            Some(RecommendedAction {
                action_type: ActionType::TuneParameters,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("strategy".to_string(), "spread_load".to_string());
                    params.insert("current_utilization".to_string(), cpu_util.to_string());
                    params.insert(
                        "available_cores".to_string(),
                        context.system_state.available_cores.to_string(),
                    );
                    params
                },
                priority: 2.0,
                expected_impact: 0.6, // Medium-high impact for CPU affinity
                estimated_duration: Duration::from_secs(45),
                reversible: true,
            })
        } else {
            None
        }
    }

    fn analyze_io_optimization(&self, metrics: &RealTimeMetrics) -> Option<RecommendedAction> {
        // Estimate I/O pressure from latency patterns
        let io_pressure = if metrics.current_latency.as_millis() > 1000 { 0.8 } else { 0.3 };

        if io_pressure > self.config.io_threshold_high {
            Some(RecommendedAction {
                action_type: ActionType::TuneParameters,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("strategy".to_string(), "async_io".to_string());
                    params.insert(
                        "current_latency".to_string(),
                        metrics.current_latency.as_millis().to_string(),
                    );
                    params.insert(
                        "optimization_target".to_string(),
                        "reduce_blocking".to_string(),
                    );
                    params
                },
                priority: 3.0,
                expected_impact: 0.7, // High impact for I/O optimization
                estimated_duration: Duration::from_secs(120),
                reversible: true,
            })
        } else {
            None
        }
    }
}

impl LiveOptimizationAlgorithm for ResourceOptimizationAlgorithm {
    fn optimize(
        &self,
        metrics: &RealTimeMetrics,
        _history: &[TimestampedMetrics],
        context: &OptimizationContext,
    ) -> Result<Vec<OptimizationRecommendation>, RealTimeMetricsError> {
        let mut recommendations = Vec::new();
        let mut actions = Vec::new();

        // Analyze different resource aspects
        if let Some(memory_action) = self.analyze_memory_pressure(metrics) {
            actions.push(memory_action);
        }

        if let Some(cpu_action) = self.analyze_cpu_affinity(metrics, context) {
            actions.push(cpu_action);
        }

        if let Some(io_action) = self.analyze_io_optimization(metrics) {
            actions.push(io_action);
        }

        if !actions.is_empty() {
            let confidence = self.calculate_resource_confidence(metrics);
            let impact = self.estimate_resource_impact(&actions, metrics);

            recommendations.push(OptimizationRecommendation {
                id: format!("resource_opt_{}", Utc::now().timestamp()),
                timestamp: Utc::now(),
                actions,
                expected_impact: impact,
                confidence,
                analysis: format!(
                    "Resource optimization: Memory={}%, CPU={}%, Latency={}ms",
                    metrics.current_memory_utilization * 100.0,
                    metrics.current_cpu_utilization * 100.0,
                    metrics.current_latency.as_millis()
                ),
                risks: self.assess_resource_risks(metrics),
                priority: 1,
                implementation_time: Duration::from_secs(120),
            });
        }

        Ok(recommendations)
    }

    fn name(&self) -> &str {
        "resource_optimization"
    }

    fn confidence(&self, data_quality: f32) -> f32 {
        data_quality * 0.85
    }

    fn is_applicable(&self, _context: &OptimizationContext) -> bool {
        true
    }

    fn update_with_feedback(
        &mut self,
        _feedback: &PerformanceFeedback,
    ) -> Result<(), RealTimeMetricsError> {
        // TODO: AlgorithmStatistics no longer has feedback_count, positive_feedback, negative_feedback fields
        // Need to implement feedback tracking differently or add these fields back to AlgorithmStatistics
        // self.stats.feedback_count += 1;
        // match feedback.feedback_type {
        //     FeedbackType::Positive => self.stats.positive_feedback += 1,
        //     FeedbackType::Negative => self.stats.negative_feedback += 1,
        //     FeedbackType::Neutral => {},
        // }

        Ok(())
    }

    fn statistics(&self) -> AlgorithmStatistics {
        self.stats.clone()
    }
}

impl ResourceOptimizationAlgorithm {
    fn calculate_resource_confidence(&self, metrics: &RealTimeMetrics) -> f32 {
        let base_confidence = 0.8;

        // Higher confidence for clear resource pressure indicators
        let memory_factor = if metrics.current_memory_utilization > 0.8 { 1.0 } else { 0.9 };
        let cpu_factor = if metrics.current_cpu_utilization > 0.8 { 1.0 } else { 0.9 };

        base_confidence * memory_factor * cpu_factor
    }

    fn estimate_resource_impact(
        &self,
        actions: &[RecommendedAction],
        metrics: &RealTimeMetrics,
    ) -> ImpactAssessment {
        let action_count = actions.len();
        let performance_impact = (action_count as f32 * 0.15).min(0.6);
        let resource_impact = action_count as f32 * 0.1;
        let complexity = if action_count > 2 { 0.7 } else { 0.4 };

        ImpactAssessment {
            performance_impact,
            resource_impact,
            complexity,
            risk_level: if metrics.current_memory_utilization > 0.9 { 0.3 } else { 0.5 },
            estimated_benefit: performance_impact * metrics.current_throughput as f32 / 100.0,
            implementation_time: Duration::from_secs(120),
        }
    }

    fn assess_resource_risks(&self, metrics: &RealTimeMetrics) -> Vec<RiskFactor> {
        let mut risks = Vec::new();

        if metrics.current_memory_utilization > 0.95 {
            risks.push(RiskFactor {
                risk_type: "SystemInstability".to_string(),
                description: "Critical memory pressure may cause system instability".to_string(),
                probability: 0.6,
                impact: 0.8,
                severity: SeverityLevel::High,
                mitigation: vec!["Implement gradual optimization with close monitoring".to_string()],
            });
        }

        if metrics.current_cpu_utilization > 0.9 {
            risks.push(RiskFactor {
                risk_type: "PerformanceRegression".to_string(),
                description: "High CPU utilization may impact optimization effectiveness"
                    .to_string(),
                probability: 0.4,
                impact: 0.6,
                severity: SeverityLevel::Medium,
                mitigation: vec![
                    "Schedule optimization during lower utilization periods".to_string()
                ],
            });
        }

        risks
    }
}
