//! Main Gradient Debugger Implementation
//!
//! This module provides the main GradientDebugger that orchestrates all gradient
//! debugging capabilities including monitoring, anomaly detection, performance tracking,
//! conflict analysis, visualization, and enhanced analysis.

use super::anomaly_detection::*;
use super::conflict_analysis::*;
use super::enhanced_analysis::*;
use super::monitoring::*;
use super::performance_tracking::*;
use super::types::*;
use super::visualization::*;
use crate::DebugConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Flow analysis for gradient flow patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowAnalysis {
    pub layer_analyses: HashMap<String, LayerFlowAnalysis>,
}

/// Analysis of gradient flow for a specific layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerFlowAnalysis {
    pub layer_name: String,
    pub is_vanishing: bool,
    pub is_exploding: bool,
    pub gradient_norm: f64,
    pub flow_consistency: f64,
}

/// Main gradient debugger
#[derive(Debug)]
pub struct GradientDebugger {
    #[allow(dead_code)]
    config: DebugConfig,
    gradient_config: GradientDebugConfig,
    gradient_histories: HashMap<String, GradientHistory>,
    current_step: usize,
    alerts: Vec<GradientAlert>,
    layer_no_gradient_count: HashMap<String, usize>,

    // Advanced features
    adaptive_thresholds: HashMap<String, AdaptiveThresholds>,
    real_time_monitors: HashMap<String, RealTimeGradientMonitor>,
    anomaly_detector: GradientAnomalyDetector,
    performance_tracker: GradientPerformanceTracker,
    conflict_analyzer: GradientConflictAnalyzer,
    flow_visualizer: GradientFlowVisualizer,
    enhanced_analyzer: EnhancedGradientAnalyzer,
}

impl GradientDebugger {
    /// Create a new gradient debugger
    pub fn new(config: DebugConfig) -> Self {
        let gradient_config = GradientDebugConfig::default();

        Self {
            config,
            gradient_config: gradient_config.clone(),
            gradient_histories: HashMap::new(),
            current_step: 0,
            alerts: Vec::new(),
            layer_no_gradient_count: HashMap::new(),
            adaptive_thresholds: HashMap::new(),
            real_time_monitors: HashMap::new(),
            anomaly_detector: GradientAnomalyDetector::default(),
            performance_tracker: GradientPerformanceTracker::default(),
            conflict_analyzer: GradientConflictAnalyzer::default(),
            flow_visualizer: GradientFlowVisualizer::default(),
            enhanced_analyzer: EnhancedGradientAnalyzer::default(),
        }
    }

    /// Create with custom gradient configuration
    pub fn with_gradient_config(config: DebugConfig, gradient_config: GradientDebugConfig) -> Self {
        Self {
            config,
            gradient_config: gradient_config.clone(),
            gradient_histories: HashMap::new(),
            current_step: 0,
            alerts: Vec::new(),
            layer_no_gradient_count: HashMap::new(),
            adaptive_thresholds: HashMap::new(),
            real_time_monitors: HashMap::new(),
            anomaly_detector: GradientAnomalyDetector::default(),
            performance_tracker: GradientPerformanceTracker::default(),
            conflict_analyzer: GradientConflictAnalyzer::default(),
            flow_visualizer: GradientFlowVisualizer::default(),
            enhanced_analyzer: EnhancedGradientAnalyzer::default(),
        }
    }

    /// Record gradient flow for a layer
    pub fn record_gradient_flow(
        &mut self,
        layer_name: &str,
        gradient_norm: f64,
        gradient_mean: f64,
        gradient_std: f64,
    ) -> Result<()> {
        let flow = GradientFlow {
            layer_name: layer_name.to_string(),
            step: self.current_step,
            gradient_norm,
            gradient_mean,
            gradient_std,
            gradient_max: gradient_mean + gradient_std,
            gradient_min: gradient_mean - gradient_std,
            dead_neurons_ratio: self.estimate_dead_neurons_ratio(gradient_norm),
            active_neurons_ratio: 1.0 - self.estimate_dead_neurons_ratio(gradient_norm),
            timestamp: chrono::Utc::now(),
        };

        // Update gradient history
        {
            let history = self
                .gradient_histories
                .entry(layer_name.to_string())
                .or_insert_with(|| GradientHistory::new(layer_name.to_string(), 1000));
            history.add_gradient_flow(&flow);
        }

        // Update adaptive thresholds
        let thresholds =
            self.adaptive_thresholds.entry(layer_name.to_string()).or_insert_with(|| {
                AdaptiveThresholds::new(
                    layer_name.to_string(),
                    self.gradient_config.vanishing_threshold,
                    self.gradient_config.exploding_threshold,
                )
            });
        thresholds.update_thresholds(gradient_norm);

        // Update real-time monitor
        let monitor = self
            .real_time_monitors
            .entry(layer_name.to_string())
            .or_insert_with(|| RealTimeGradientMonitor::new(layer_name.to_string()));
        monitor.update(gradient_norm);

        // Check for alerts
        self.check_gradient_alerts(layer_name, &flow)?;

        // Record performance metrics
        let timer = self.performance_tracker.start_timing(layer_name);
        let (_, computation_time) = timer.finish();
        self.performance_tracker
            .record_layer_performance(layer_name, computation_time, 0); // Memory usage simplified

        // Detect anomalies
        let anomalies =
            self.anomaly_detector
                .detect_anomalies(layer_name, gradient_norm, self.current_step);
        for anomaly in anomalies {
            self.alerts.push(GradientAlert::GradientOscillation {
                layer_name: anomaly.layer_name,
                variance: anomaly.severity,
            });
        }

        // Establish baseline if needed
        if let Some(history) = self.gradient_histories.get(layer_name) {
            if history.gradient_norms.len() == 50 {
                let gradient_values: Vec<f64> = history.gradient_norms.iter().cloned().collect();
                self.anomaly_detector.establish_baseline(layer_name, &gradient_values);
            }
        }

        Ok(())
    }

    /// Get current gradient debugging status
    pub fn get_status(&self) -> GradientDebugStatus {
        let layer_statuses: HashMap<String, LayerGradientStatus> = self
            .gradient_histories
            .iter()
            .map(|(layer_name, history)| {
                let status = self.compute_layer_status(layer_name, history);
                (layer_name.clone(), status)
            })
            .collect();

        let overall_health = self.compute_overall_health(&layer_statuses);
        let recent_alerts: Vec<GradientAlert> =
            self.alerts.iter().rev().take(10).cloned().collect();

        GradientDebugStatus {
            current_step: self.current_step,
            overall_health,
            layer_statuses,
            recent_alerts,
            total_alerts: self.alerts.len(),
            active_layers: self.gradient_histories.len(),
        }
    }

    /// Generate flow analysis for report generation
    fn generate_flow_analysis(&self) -> FlowAnalysis {
        let mut layer_analyses = HashMap::new();

        for (layer_name, history) in &self.gradient_histories {
            let latest_gradient = history.gradient_norms.back().cloned().unwrap_or(0.0);

            // Determine if gradients are vanishing or exploding
            let is_vanishing = latest_gradient < 1e-8
                || (history.gradient_norms.len() > 5
                    && history.gradient_norms.iter().rev().take(5).all(|&g| g < 1e-6));

            let is_exploding = latest_gradient > 100.0
                || (history.gradient_norms.len() > 3
                    && history.gradient_norms.iter().rev().take(3).any(|&g| g > 50.0));

            // Calculate flow consistency (variance in gradient norms)
            let flow_consistency = if history.gradient_norms.len() > 1 {
                let mean = history.gradient_norms.iter().sum::<f64>()
                    / history.gradient_norms.len() as f64;
                let variance =
                    history.gradient_norms.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                        / history.gradient_norms.len() as f64;
                1.0 / (1.0 + variance) // Higher consistency = lower variance
            } else {
                1.0
            };

            layer_analyses.insert(
                layer_name.clone(),
                LayerFlowAnalysis {
                    layer_name: layer_name.clone(),
                    is_vanishing,
                    is_exploding,
                    gradient_norm: latest_gradient,
                    flow_consistency,
                },
            );
        }

        FlowAnalysis { layer_analyses }
    }

    /// Generate comprehensive debugging report
    pub fn generate_comprehensive_report(&self) -> Result<ComprehensiveGradientReport> {
        let status = self.get_status();
        let conflict_analysis = self.conflict_analyzer.analyze_conflicts(&self.gradient_histories);
        let visualization = self
            .flow_visualizer
            .generate_visualization(&self.gradient_histories, self.current_step);
        let enhanced_analysis =
            self.enhanced_analyzer.generate_enhanced_analysis(&self.gradient_histories);
        let performance_snapshot = self.performance_tracker.take_performance_snapshot();
        let anomaly_summary = self.anomaly_detector.get_anomaly_summary(None);

        let flow_analysis = self.generate_flow_analysis();

        Ok(ComprehensiveGradientReport {
            timestamp: chrono::Utc::now(),
            status,
            conflict_analysis,
            visualization,
            enhanced_analysis,
            flow_analysis,
            performance_snapshot,
            anomaly_summary,
            recommendations: self.generate_comprehensive_recommendations()?,
        })
    }

    /// Analyze gradient conflicts between layers
    pub fn analyze_gradient_conflicts(&self) -> GradientConflictAnalysis {
        self.conflict_analyzer.analyze_conflicts(&self.gradient_histories)
    }

    /// Generate gradient flow visualization
    pub fn generate_gradient_flow_visualization(&self) -> GradientFlowVisualization {
        self.flow_visualizer
            .generate_visualization(&self.gradient_histories, self.current_step)
    }

    /// Generate enhanced layer analysis
    pub fn generate_enhanced_layer_analysis(&self) -> EnhancedLayerGradientAnalysis {
        self.enhanced_analyzer.generate_enhanced_analysis(&self.gradient_histories)
    }

    /// Get performance insights
    pub fn get_performance_insights(&self) -> PerformanceInsights {
        let trends = self.performance_tracker.get_performance_trends();
        let recommendations = self.performance_tracker.generate_optimization_recommendations();
        let bottlenecks = self.performance_tracker.bottleneck_layers.clone();

        PerformanceInsights {
            trends,
            recommendations,
            bottlenecks,
            current_throughput: self.performance_tracker.throughput_gradients_per_second,
            memory_usage: self.performance_tracker.memory_usage_bytes,
        }
    }

    /// Advance to next step
    pub fn next_step(&mut self) {
        self.current_step += 1;

        // Clear old alerts (keep last 100)
        if self.alerts.len() > 100 {
            self.alerts.drain(0..self.alerts.len() - 100);
        }

        // Update no-gradient counters
        for (layer_name, history) in &self.gradient_histories {
            if let Some(latest_norm) = history.gradient_norms.back() {
                if *latest_norm < 1e-8 {
                    *self.layer_no_gradient_count.entry(layer_name.clone()).or_insert(0) += 1;
                } else {
                    self.layer_no_gradient_count.insert(layer_name.clone(), 0);
                }
            }
        }

        // Check for no-gradient alerts
        for (layer_name, &count) in &self.layer_no_gradient_count {
            if count >= self.gradient_config.no_gradient_steps_threshold {
                self.alerts.push(GradientAlert::NoGradientFlow {
                    layer_name: layer_name.clone(),
                    steps_without_gradient: count,
                });
            }
        }
    }

    /// Reset debugger state
    pub fn reset(&mut self) {
        self.gradient_histories.clear();
        self.current_step = 0;
        self.alerts.clear();
        self.layer_no_gradient_count.clear();
        self.adaptive_thresholds.clear();
        self.real_time_monitors.clear();
        self.anomaly_detector = GradientAnomalyDetector::default();
        self.performance_tracker = GradientPerformanceTracker::default();
    }

    /// Get alerts for a specific layer
    pub fn get_layer_alerts(&self, layer_name: &str) -> Vec<&GradientAlert> {
        self.alerts
            .iter()
            .filter(|alert| match alert {
                GradientAlert::VanishingGradients {
                    layer_name: name, ..
                } => name == layer_name,
                GradientAlert::ExplodingGradients {
                    layer_name: name, ..
                } => name == layer_name,
                GradientAlert::DeadNeurons {
                    layer_name: name, ..
                } => name == layer_name,
                GradientAlert::GradientOscillation {
                    layer_name: name, ..
                } => name == layer_name,
                GradientAlert::NoGradientFlow {
                    layer_name: name, ..
                } => name == layer_name,
            })
            .collect()
    }

    /// Get gradient history for a layer
    pub fn get_layer_history(&self, layer_name: &str) -> Option<&GradientHistory> {
        self.gradient_histories.get(layer_name)
    }

    /// Get all monitored layers
    pub fn get_monitored_layers(&self) -> Vec<&String> {
        self.gradient_histories.keys().collect()
    }

    // Private helper methods

    fn estimate_dead_neurons_ratio(&self, gradient_norm: f64) -> f64 {
        // Simplified estimation - in practice would analyze individual neuron gradients
        if gradient_norm < 1e-6 {
            0.9 // Assume 90% dead if very low gradient
        } else if gradient_norm < 1e-4 {
            0.3 // Assume 30% dead if low gradient
        } else {
            0.05 // Assume 5% dead for normal gradients
        }
    }

    fn check_gradient_alerts(&mut self, layer_name: &str, flow: &GradientFlow) -> Result<()> {
        // Check adaptive thresholds first
        if let Some(thresholds) = self.adaptive_thresholds.get(layer_name) {
            let threshold_alerts = thresholds.check_thresholds(flow.gradient_norm);
            self.alerts.extend(threshold_alerts);
        } else {
            // Fallback to static thresholds
            if flow.gradient_norm < self.gradient_config.vanishing_threshold {
                self.alerts.push(GradientAlert::VanishingGradients {
                    layer_name: layer_name.to_string(),
                    norm: flow.gradient_norm,
                    threshold: self.gradient_config.vanishing_threshold,
                });
            }

            if flow.gradient_norm > self.gradient_config.exploding_threshold {
                self.alerts.push(GradientAlert::ExplodingGradients {
                    layer_name: layer_name.to_string(),
                    norm: flow.gradient_norm,
                    threshold: self.gradient_config.exploding_threshold,
                });
            }
        }

        // Check dead neurons
        if flow.dead_neurons_ratio > self.gradient_config.dead_neuron_threshold {
            self.alerts.push(GradientAlert::DeadNeurons {
                layer_name: layer_name.to_string(),
                ratio: flow.dead_neurons_ratio,
                threshold: self.gradient_config.dead_neuron_threshold,
            });
        }

        // Check oscillation
        if let Some(monitor) = self.real_time_monitors.get(layer_name) {
            if monitor.is_oscillating() {
                self.alerts.push(GradientAlert::GradientOscillation {
                    layer_name: layer_name.to_string(),
                    variance: monitor.get_stability_score(),
                });
            }
        }

        Ok(())
    }

    fn compute_layer_status(
        &self,
        layer_name: &str,
        history: &GradientHistory,
    ) -> LayerGradientStatus {
        let latest_norm = history.gradient_norms.back().cloned().unwrap_or(0.0);
        let health = self.classify_layer_health(layer_name, history);
        let alerts = self.get_layer_alerts(layer_name).len();
        let trend = history.get_trend_slope().unwrap_or(0.0);

        LayerGradientStatus {
            layer_name: layer_name.to_string(),
            health,
            latest_gradient_norm: latest_norm,
            gradient_trend: trend,
            alert_count: alerts,
            steps_recorded: history.gradient_norms.len(),
        }
    }

    fn classify_layer_health(&self, layer_name: &str, history: &GradientHistory) -> LayerHealth {
        let latest_norm = history.gradient_norms.back().cloned().unwrap_or(0.0);
        let alert_count = self.get_layer_alerts(layer_name).len();

        if !(1e-7..=100.0).contains(&latest_norm) || alert_count > 3 {
            LayerHealth::Critical
        } else if !(1e-5..=10.0).contains(&latest_norm) || alert_count > 0 {
            LayerHealth::Warning
        } else {
            LayerHealth::Healthy
        }
    }

    fn compute_overall_health(
        &self,
        layer_statuses: &HashMap<String, LayerGradientStatus>,
    ) -> LayerHealth {
        if layer_statuses.is_empty() {
            return LayerHealth::Healthy;
        }

        let critical_count =
            layer_statuses.values().filter(|s| s.health == LayerHealth::Critical).count();
        let warning_count =
            layer_statuses.values().filter(|s| s.health == LayerHealth::Warning).count();
        let total = layer_statuses.len();

        if critical_count > 0 || warning_count as f64 / total as f64 > 0.5 {
            LayerHealth::Critical
        } else if warning_count > 0 {
            LayerHealth::Warning
        } else {
            LayerHealth::Healthy
        }
    }

    fn generate_comprehensive_recommendations(&self) -> Result<Vec<GradientRecommendation>> {
        let mut recommendations = Vec::new();

        // Performance recommendations
        let perf_recs = self.performance_tracker.generate_optimization_recommendations();
        for rec in perf_recs {
            recommendations.push(GradientRecommendation {
                recommendation_type: RecommendationType::Performance,
                title: rec.layer_name,
                description: format!("{:?}: {}", rec.issue_type, rec.recommendations.join(", ")),
                priority: match rec.severity {
                    OptimizationSeverity::Critical => GradientRecommendationPriority::High,
                    OptimizationSeverity::High => GradientRecommendationPriority::High,
                    OptimizationSeverity::Medium => GradientRecommendationPriority::Medium,
                    OptimizationSeverity::Low => GradientRecommendationPriority::Low,
                },
                expected_impact: rec.expected_improvement,
            });
        }

        // Conflict recommendations
        let conflict_analysis = self.conflict_analyzer.analyze_conflicts(&self.gradient_histories);
        for strategy in conflict_analysis.mitigation_strategies {
            recommendations.push(GradientRecommendation {
                recommendation_type: RecommendationType::Conflict,
                title: strategy.strategy_name,
                description: strategy.description,
                priority: match strategy.implementation_complexity {
                    MitigationComplexity::Simple => GradientRecommendationPriority::High,
                    MitigationComplexity::Moderate => GradientRecommendationPriority::Medium,
                    MitigationComplexity::Complex => GradientRecommendationPriority::Medium,
                    MitigationComplexity::RequiresArchitectureChange => {
                        GradientRecommendationPriority::Low
                    },
                },
                expected_impact: strategy.effectiveness,
            });
        }

        // Anomaly recommendations
        let anomaly_summary = self.anomaly_detector.get_anomaly_summary(None);
        for rec_text in anomaly_summary.recommendations {
            recommendations.push(GradientRecommendation {
                recommendation_type: RecommendationType::Anomaly,
                title: "Anomaly Mitigation".to_string(),
                description: rec_text,
                priority: if anomaly_summary.average_severity > 0.7 {
                    GradientRecommendationPriority::High
                } else {
                    GradientRecommendationPriority::Medium
                },
                expected_impact: 1.0 - anomaly_summary.average_severity,
            });
        }

        // Sort by priority and expected impact
        recommendations.sort_by(|a, b| {
            let priority_cmp = b.priority.cmp(&a.priority);
            if priority_cmp == std::cmp::Ordering::Equal {
                b.expected_impact.partial_cmp(&a.expected_impact).unwrap()
            } else {
                priority_cmp
            }
        });

        Ok(recommendations)
    }

    /// Generate recommendations based on current analysis
    pub fn generate_recommendations(&self) -> Result<Vec<GradientRecommendation>> {
        self.generate_comprehensive_recommendations()
    }

    /// Start the gradient debugger
    pub async fn start(&mut self) -> Result<()> {
        // Initialize monitoring systems
        self.performance_tracker.start_monitoring();

        // Reset state for a new debugging session
        self.current_step = 0;
        self.alerts.clear();

        // Initialize adaptive thresholds for existing histories
        for (layer_name, history) in &self.gradient_histories {
            if !history.gradient_norms.is_empty() {
                let thresholds = AdaptiveThresholds::from_history(history);
                self.adaptive_thresholds.insert(layer_name.clone(), thresholds);
            }
        }

        Ok(())
    }

    /// Generate comprehensive gradient report
    pub async fn generate_report(&self) -> Result<ComprehensiveGradientReport> {
        let status = GradientDebugStatus {
            current_step: self.current_step,
            overall_health: self.evaluate_overall_health(),
            layer_statuses: self.get_layer_statuses(),
            recent_alerts: self.alerts.iter().rev().take(10).cloned().collect(),
            total_alerts: self.alerts.len(),
            active_layers: self.gradient_histories.len(),
        };

        let conflict_analysis = self.conflict_analyzer.analyze_conflicts(&self.gradient_histories);
        let visualization = self.flow_visualizer.create_visualization(&self.gradient_histories);
        let enhanced_analysis = self.enhanced_analyzer.analyze_gradients(&self.gradient_histories);
        let performance_snapshot = self.performance_tracker.take_performance_snapshot();
        let anomaly_summary = self.anomaly_detector.get_anomaly_summary(None);
        let recommendations = self.generate_recommendations().unwrap_or_default();

        let flow_analysis = self.generate_flow_analysis();

        Ok(ComprehensiveGradientReport {
            timestamp: chrono::Utc::now(),
            status,
            conflict_analysis,
            visualization,
            enhanced_analysis,
            flow_analysis,
            performance_snapshot,
            anomaly_summary,
            recommendations,
        })
    }

    /// Quick analysis for immediate insights
    pub async fn quick_analysis(&self) -> Result<GradientQuickAnalysis> {
        let mut problematic_layers = Vec::new();
        let mut total_gradients = 0f64;
        let mut active_layers = 0;

        for (layer_name, history) in &self.gradient_histories {
            if !history.gradient_norms.is_empty() {
                active_layers += 1;
                let latest_norm = history.gradient_norms.back().unwrap();
                total_gradients += latest_norm;

                // Check for basic problems
                if *latest_norm < 1e-8 {
                    problematic_layers.push(format!("{}: Vanishing gradients", layer_name));
                } else if *latest_norm > 100.0 {
                    problematic_layers.push(format!("{}: Exploding gradients", layer_name));
                }
            }
        }

        let average_gradient =
            if active_layers > 0 { total_gradients / active_layers as f64 } else { 0.0 };

        let health_score = self.calculate_quick_health_score();

        Ok(GradientQuickAnalysis {
            overall_health: if health_score > 0.8 {
                LayerHealth::Healthy
            } else if health_score > 0.5 {
                LayerHealth::Warning
            } else {
                LayerHealth::Critical
            },
            active_layers,
            problematic_layers,
            average_gradient_norm: average_gradient,
            recent_alerts_count: self.alerts.len(),
            timestamp: chrono::Utc::now(),
        })
    }

    /// Evaluate overall gradient health
    fn evaluate_overall_health(&self) -> LayerHealth {
        if self.gradient_histories.is_empty() {
            return LayerHealth::Unknown;
        }

        let mut healthy_count = 0;
        let mut warning_count = 0;
        let mut critical_count = 0;

        for history in self.gradient_histories.values() {
            if let Some(latest_norm) = history.gradient_norms.back() {
                if *latest_norm < 1e-8 || *latest_norm > 100.0 {
                    critical_count += 1;
                } else if *latest_norm < 1e-6 || *latest_norm > 10.0 {
                    warning_count += 1;
                } else {
                    healthy_count += 1;
                }
            }
        }

        let total = healthy_count + warning_count + critical_count;
        let critical_ratio = critical_count as f64 / total as f64;
        let warning_ratio = (warning_count + critical_count) as f64 / total as f64;

        if critical_ratio > 0.3 {
            LayerHealth::Critical
        } else if warning_ratio > 0.5 {
            LayerHealth::Warning
        } else {
            LayerHealth::Healthy
        }
    }

    /// Get status for each layer
    fn get_layer_statuses(&self) -> HashMap<String, LayerGradientStatus> {
        let mut statuses = HashMap::new();

        for (layer_name, history) in &self.gradient_histories {
            let status = if let Some(latest_norm) = history.gradient_norms.back() {
                LayerGradientStatus {
                    layer_name: layer_name.clone(),
                    latest_gradient_norm: *latest_norm,
                    gradient_trend: self.calculate_trend_value(history),
                    health: if *latest_norm < 1e-8 {
                        LayerHealth::Critical
                    } else if *latest_norm > 100.0 {
                        LayerHealth::Critical
                    } else if *latest_norm < 1e-6 || *latest_norm > 10.0 {
                        LayerHealth::Warning
                    } else {
                        LayerHealth::Healthy
                    },
                    alert_count: self.get_layer_alerts(layer_name).len(),
                    steps_recorded: history.gradient_norms.len(),
                }
            } else {
                LayerGradientStatus {
                    layer_name: layer_name.clone(),
                    latest_gradient_norm: 0.0,
                    gradient_trend: 0.0,
                    health: LayerHealth::Unknown,
                    alert_count: 0,
                    steps_recorded: 0,
                }
            };

            statuses.insert(layer_name.clone(), status);
        }

        statuses
    }

    /// Calculate gradient trend for a layer
    #[allow(dead_code)]
    fn calculate_trend(&self, history: &GradientHistory) -> GradientTrend {
        if history.gradient_norms.len() < 3 {
            return GradientTrend::Unknown;
        }

        let recent: Vec<f64> = history.gradient_norms.iter().rev().take(3).cloned().collect();

        if recent[0] > recent[1] && recent[1] > recent[2] {
            GradientTrend::Increasing
        } else if recent[0] < recent[1] && recent[1] < recent[2] {
            GradientTrend::Decreasing
        } else {
            GradientTrend::Stable
        }
    }

    /// Calculate gradient trend as numeric value for a layer
    fn calculate_trend_value(&self, history: &GradientHistory) -> f64 {
        if history.gradient_norms.len() < 2 {
            return 0.0;
        }

        let recent: Vec<f64> = history.gradient_norms.iter().rev().take(10).cloned().collect();
        if recent.len() < 2 {
            return 0.0;
        }

        // Calculate linear trend slope
        let n = recent.len() as f64;
        let sum_x = (0..recent.len()).sum::<usize>() as f64;
        let sum_y = recent.iter().sum::<f64>();
        let sum_xy = recent.iter().enumerate().map(|(i, &y)| i as f64 * y).sum::<f64>();
        let sum_x2 = (0..recent.len()).map(|i| (i * i) as f64).sum::<f64>();

        
        (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    }

    /// Calculate quick health score
    fn calculate_quick_health_score(&self) -> f64 {
        if self.gradient_histories.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;
        let mut count = 0;

        for history in self.gradient_histories.values() {
            if let Some(latest_norm) = history.gradient_norms.back() {
                // Score based on gradient magnitude (ideal range: 1e-4 to 1.0)
                let norm_score = if *latest_norm >= 1e-4 && *latest_norm <= 1.0 {
                    1.0
                } else if *latest_norm >= 1e-6 && *latest_norm <= 10.0 {
                    0.7
                } else if *latest_norm >= 1e-8 && *latest_norm <= 100.0 {
                    0.3
                } else {
                    0.0
                };

                score += norm_score;
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            score / count as f64
        }
    }
}

/// Current gradient debugging status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GradientDebugStatus {
    pub current_step: usize,
    pub overall_health: LayerHealth,
    pub layer_statuses: HashMap<String, LayerGradientStatus>,
    pub recent_alerts: Vec<GradientAlert>,
    pub total_alerts: usize,
    pub active_layers: usize,
}

/// Comprehensive gradient debugging report
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComprehensiveGradientReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub status: GradientDebugStatus,
    pub conflict_analysis: GradientConflictAnalysis,
    pub visualization: GradientFlowVisualization,
    pub enhanced_analysis: EnhancedLayerGradientAnalysis,
    pub flow_analysis: FlowAnalysis,
    pub performance_snapshot: PerformanceSnapshot,
    pub anomaly_summary: AnomalySummary,
    pub recommendations: Vec<GradientRecommendation>,
}

impl ComprehensiveGradientReport {
    /// Check if there are vanishing gradient issues
    pub fn has_vanishing_gradients(&self) -> bool {
        // Check if any layers have very small gradients
        for layer_status in self.status.layer_statuses.values() {
            if layer_status.latest_gradient_norm < 1e-8 {
                return true;
            }
        }

        // Check anomaly summary for vanishing gradient patterns
        for anomaly in &self.anomaly_summary.anomalies {
            if matches!(
                anomaly.anomaly_type,
                crate::anomaly_detector::AnomalyType::GradientVanishing
            ) {
                return true;
            }
        }

        false
    }

    /// Check if there are exploding gradient issues
    pub fn has_exploding_gradients(&self) -> bool {
        // Check if any layers have very large gradients
        for layer_status in self.status.layer_statuses.values() {
            if layer_status.latest_gradient_norm > 100.0 {
                return true;
            }
        }

        // Check anomaly summary for exploding gradient patterns
        for anomaly in &self.anomaly_summary.anomalies {
            if matches!(
                anomaly.anomaly_type,
                crate::anomaly_detector::AnomalyType::GradientExplosion
                    | crate::anomaly_detector::AnomalyType::NumericalInstability
            ) {
                return true;
            }
        }

        false
    }
}

/// Performance insights summary
#[derive(Debug, Clone)]
pub struct PerformanceInsights {
    pub trends: PerformanceTrends,
    pub recommendations: Vec<OptimizationRecommendation>,
    pub bottlenecks: Vec<String>,
    pub current_throughput: f64,
    pub memory_usage: usize,
}

/// Gradient debugging recommendation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GradientRecommendation {
    pub recommendation_type: RecommendationType,
    pub title: String,
    pub description: String,
    pub priority: GradientRecommendationPriority,
    pub expected_impact: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum RecommendationType {
    Performance,
    Conflict,
    Anomaly,
    Architecture,
    Optimization,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum GradientRecommendationPriority {
    Low,
    Medium,
    High,
}

/// Quick analysis results for immediate insights
#[derive(Debug, Clone)]
pub struct GradientQuickAnalysis {
    pub overall_health: LayerHealth,
    pub active_layers: usize,
    pub problematic_layers: Vec<String>,
    pub average_gradient_norm: f64,
    pub recent_alerts_count: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Status for individual layer gradients
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerGradientStatus {
    pub layer_name: String,
    pub health: LayerHealth,
    pub latest_gradient_norm: f64,
    pub gradient_trend: f64,
    pub alert_count: usize,
    pub steps_recorded: usize,
}

/// Gradient trend indicators
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GradientTrend {
    Unknown,
    Increasing,
    Decreasing,
    Stable,
}
