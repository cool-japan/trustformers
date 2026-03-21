//! Lock Contention Analyzer
//!
//! Detects and analyzes lock contention patterns, providing optimization
//! recommendations for reducing synchronization overhead.

use super::super::types::*;
use anyhow::Result;
use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

pub struct LockContentionAnalyzer {
    /// Contention analysis algorithms
    analysis_algorithms: Arc<Mutex<Vec<Box<dyn LockAnalysisAlgorithm + Send + Sync>>>>,

    /// Lock usage pattern database
    pattern_database: Arc<RwLock<LockUsagePatternDatabase>>,

    /// Contention metrics collector
    metrics_collector: Arc<RwLock<LockContentionMetrics>>,

    /// Configuration
    config: LockAnalysisConfig,
}

impl LockContentionAnalyzer {
    /// Creates a new lock contention analyzer
    pub async fn new(config: LockAnalysisConfig) -> Result<Self> {
        let mut analysis_algorithms: Vec<Box<dyn LockAnalysisAlgorithm + Send + Sync>> = Vec::new();

        // Initialize lock analysis algorithms
        // TODO: ContentionFrequencyAnalysis::new requires frequency: f64, hotspots: Vec<String>
        analysis_algorithms.push(Box::new(ContentionFrequencyAnalysis::new(0.0, Vec::new())));
        analysis_algorithms.push(Box::new(HoldTimeAnalysis::new()));
        // TODO: WaitTimeAnalysis::new requires avg_wait_time_us: u64, max_wait_time_us: u64
        analysis_algorithms.push(Box::new(WaitTimeAnalysis::new(0, 0)));
        analysis_algorithms.push(Box::new(DeadlockPotentialAnalysis::new()));

        Ok(Self {
            analysis_algorithms: Arc::new(Mutex::new(analysis_algorithms)),
            pattern_database: Arc::new(RwLock::new(LockUsagePatternDatabase::new())),
            metrics_collector: Arc::new(RwLock::new(LockContentionMetrics::new())),
            config,
        })
    }

    /// Analyzes lock contention in test execution data
    pub async fn analyze_lock_contention(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<LockAnalysisResult> {
        let start_time = Utc::now();

        // Extract lock usage data
        let lock_events = self.extract_lock_events(test_data)?;

        // Execute synchronously to avoid lifetime issues with mutex guards
        let analysis_task_results: Vec<_> = {
            let algorithms = self.analysis_algorithms.lock();
            algorithms
                .iter()
                .map(|algorithm| {
                    let algorithm_name = algorithm.name().to_string();
                    let analysis_start = Instant::now();
                    let result_string = algorithm.analyze_locks();
                    let result: Result<String> = Ok(result_string);
                    let analysis_duration = analysis_start.elapsed();
                    (algorithm_name, result, analysis_duration)
                })
                .collect()
        };

        // Collect analysis results
        let mut lock_analyses_structs = Vec::new(); // For helper methods
        let mut algorithm_results = Vec::new();

        for (algorithm_name, result, duration) in analysis_task_results {
            match result {
                Ok(analysis_string) => {
                    // Create a placeholder LockAnalysis for helper methods
                    let lock_analysis = LockAnalysis {
                        lock_id: analysis_string.clone(),
                        lock_events: Vec::new(),
                        contention_metrics: LockContentionMetrics::default(),
                        dependencies: Vec::new(),
                        analysis_timestamp: chrono::Utc::now(),
                        average_contention_level: 0.0,
                        average_hold_time: Duration::from_secs(0),
                        contention_events: Vec::new(),
                        max_wait_time: Duration::from_secs(0),
                        min_wait_time: Duration::from_secs(0),
                        average_wait_time: Duration::from_secs(0),
                    };

                    algorithm_results.push(LockAlgorithmResult {
                        algorithm: algorithm_name,
                        analysis: analysis_string,
                        analysis_duration: duration,
                        confidence: self.calculate_lock_analysis_confidence(&lock_analysis) as f64,
                    });
                    lock_analyses_structs.push(lock_analysis);
                },
                Err(e) => {
                    log::warn!("Lock analysis algorithm failed: {}", e);
                },
            }
        }

        // Synthesize results
        let _contention_summary_struct = self.synthesize_contention_summary(&lock_analyses_structs);
        let _latency_bounds_struct = self.calculate_latency_bounds(&lock_analyses_structs);
        let optimization_recommendations_vec =
            self.generate_lock_optimizations(&lock_analyses_structs);

        // Convert to expected types
        let contention_summary: HashMap<String, f64> = HashMap::new(); // TODO: extract from contention_summary_struct
        let latency_bounds: HashMap<String, Duration> = HashMap::new(); // TODO: extract from latency_bounds_struct
        let optimization_recommendations: Vec<String> =
            optimization_recommendations_vec.iter().map(|o| format!("{:?}", o)).collect();

        Ok(LockAnalysisResult {
            lock_events,
            contention_summary,
            latency_bounds,
            optimization_recommendations,
            algorithm_results,
            analysis_duration: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
            confidence: self.calculate_overall_lock_confidence(&lock_analyses_structs) as f64,
        })
    }

    /// Extracts lock events from test execution traces
    fn extract_lock_events(&self, test_data: &TestExecutionData) -> Result<Vec<LockEvent>> {
        let mut events = Vec::new();

        for trace in &test_data.execution_traces {
            match trace.operation.as_str() {
                "LockAcquire" | "LockAcquisition" => {
                    events.push(LockEvent {
                        timestamp: trace.timestamp,
                        lock_id: trace.resource.clone(),
                        event_type: "acquire".to_string(),
                        thread_id: trace.thread_id,
                        duration: Duration::from_secs(0), // Duration not available in trace
                        wait_time: None,
                        contention_level: 0.0,
                        performance_impact: 0.0,
                        deadlock_risk: 0.0,
                        alternatives: Vec::new(),
                        // TODO: ExecutionTrace no longer has result field
                        success: true, // Assume success if not specified
                    });
                },
                "LockRelease" => {
                    events.push(LockEvent {
                        timestamp: trace.timestamp,
                        lock_id: trace.resource.clone(),
                        event_type: "release".to_string(),
                        thread_id: trace.thread_id,
                        duration: Duration::from_secs(0), // Duration not available in trace
                        wait_time: None,
                        contention_level: 0.0,
                        performance_impact: 0.0,
                        deadlock_risk: 0.0,
                        alternatives: Vec::new(),
                        // TODO: ExecutionTrace no longer has result field
                        success: true, // Assume success if not specified
                    });
                },
                _ => {},
            }
        }

        Ok(events)
    }

    /// Calculates lock analysis confidence
    fn calculate_lock_analysis_confidence(&self, analysis: &LockAnalysis) -> f32 {
        let contention_factor = 1.0 - analysis.average_contention_level;
        let hold_time_factor =
            if analysis.average_hold_time > Duration::from_millis(100) { 0.7 } else { 0.9 };

        ((contention_factor + hold_time_factor) / 2.0) as f32
    }

    /// Synthesizes contention summary
    fn synthesize_contention_summary(&self, analyses: &[LockAnalysis]) -> ContentionSummary {
        let total_contentions = analyses.iter().map(|a| a.contention_events.len()).sum::<usize>();

        let avg_contention_level = analyses.iter().map(|a| a.average_contention_level).sum::<f64>()
            as f32
            / analyses.len().max(1) as f32;

        let _max_wait_time = analyses
            .iter()
            .map(|a| a.max_wait_time)
            .max()
            .unwrap_or(Duration::from_millis(0));

        let hotspot_ids = self
            .identify_contention_hotspots(analyses)
            .iter()
            .map(|h| h.resource_id.clone())
            .collect();

        let contention_by_resource = analyses
            .iter()
            .map(|a| (a.lock_id.clone(), a.contention_events.len()))
            .collect();

        ContentionSummary {
            total_contentions,
            contention_hotspots: hotspot_ids,
            average_contention_duration: Duration::from_secs_f64(avg_contention_level as f64),
            contention_by_resource,
        }
    }

    /// Identifies contention hotspots
    fn identify_contention_hotspots(&self, analyses: &[LockAnalysis]) -> Vec<ContentionHotspot> {
        let mut hotspots = Vec::new();

        for analysis in analyses {
            if analysis.average_contention_level > 0.7 {
                hotspots.push(ContentionHotspot {
                    resource_id: analysis.lock_id.clone(),
                    contention_frequency: analysis.average_contention_level,
                    average_wait_time: analysis.max_wait_time,
                    affected_threads: Vec::new(),
                });
            }
        }

        hotspots
    }

    /// Calculates severity distribution
    fn calculate_severity_distribution(&self, analyses: &[LockAnalysis]) -> SeverityDistribution {
        let mut low = 0;
        let mut medium = 0;
        let mut high = 0;
        let mut critical = 0;

        for analysis in analyses {
            match analysis.average_contention_level {
                x if x < 0.25 => low += 1,
                x if x < 0.5 => medium += 1,
                x if x < 0.75 => high += 1,
                _ => critical += 1,
            }
        }

        let total = low + medium + high + critical;
        let mut distribution = HashMap::new();
        distribution.insert("Low".to_string(), low);
        distribution.insert("Medium".to_string(), medium);
        distribution.insert("High".to_string(), high);
        distribution.insert("Critical".to_string(), critical);

        let most_common = if critical > high && critical > medium && critical > low {
            "Critical"
        } else if high > medium && high > low {
            "High"
        } else if medium > low {
            "Medium"
        } else {
            "Low"
        };

        SeverityDistribution {
            distribution,
            total_count: total,
            most_common_severity: most_common.to_string(),
        }
    }

    /// Calculates latency bounds
    fn calculate_latency_bounds(&self, analyses: &[LockAnalysis]) -> LatencyBounds {
        let min_latency = analyses
            .iter()
            .map(|a| a.min_wait_time)
            .min()
            .unwrap_or(Duration::from_millis(0));

        let max_latency = analyses
            .iter()
            .map(|a| a.max_wait_time)
            .max()
            .unwrap_or(Duration::from_millis(0));

        let avg_latency = Duration::from_millis(
            analyses.iter().map(|a| a.average_wait_time.as_millis() as u64).sum::<u64>()
                / analyses.len().max(1) as u64,
        );

        LatencyBounds {
            min_latency,
            max_latency,
            average_latency: avg_latency,
        }
    }

    /// Calculates percentile latency
    fn calculate_percentile_latency(&self, analyses: &[LockAnalysis], percentile: u8) -> Duration {
        if analyses.is_empty() {
            return Duration::from_millis(0);
        }

        let mut wait_times: Vec<Duration> = analyses.iter().map(|a| a.average_wait_time).collect();

        wait_times.sort();

        let index = ((percentile as f32 / 100.0) * wait_times.len() as f32) as usize;
        wait_times
            .get(index.min(wait_times.len() - 1))
            .copied()
            .unwrap_or(Duration::from_millis(0))
    }

    /// Generates lock optimization recommendations
    fn generate_lock_optimizations(
        &self,
        analyses: &[LockAnalysis],
    ) -> Vec<LockOptimizationRecommendation> {
        let mut recommendations = Vec::new();

        for analysis in analyses {
            if analysis.average_contention_level > 0.5 {
                recommendations.push(LockOptimizationRecommendation {
                    lock_id: analysis.lock_id.clone(),
                    recommendation_type: "ReduceContention".to_string(),
                    expected_improvement: 0.4,
                    implementation_effort: "Medium".to_string(),
                });
            }

            if analysis.average_hold_time > Duration::from_millis(100) {
                recommendations.push(LockOptimizationRecommendation {
                    lock_id: analysis.lock_id.clone(),
                    recommendation_type: "ReduceHoldTime".to_string(),
                    expected_improvement: 0.3,
                    implementation_effort: "Low".to_string(),
                });
            }
        }

        recommendations
    }

    /// Calculates overall lock confidence
    fn calculate_overall_lock_confidence(&self, analyses: &[LockAnalysis]) -> f32 {
        if analyses.is_empty() {
            return 0.0;
        }

        let confidences: Vec<f32> =
            analyses.iter().map(|a| self.calculate_lock_analysis_confidence(a)).collect();

        confidences.iter().map(|&x| x as f64).sum::<f64>() as f32 / confidences.len() as f32
    }
}
