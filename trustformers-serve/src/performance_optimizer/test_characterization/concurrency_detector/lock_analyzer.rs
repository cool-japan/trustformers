//! Lock Contention Analyzer
//!
//! Detects and analyzes lock contention patterns, providing optimization
//! recommendations for reducing synchronization overhead.

use super::super::types::*;
use anyhow::Result;
use chrono::Utc;
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

pub struct LockContentionAnalyzer {
    /// Contention analysis algorithms
    analysis_algorithms: Arc<Mutex<Vec<Box<dyn LockAnalysisAlgorithm + Send + Sync>>>>,

    /// Configuration
    config: LockAnalysisConfig,
}

impl LockContentionAnalyzer {
    /// Creates a new lock contention analyzer
    pub async fn new(config: LockAnalysisConfig) -> Result<Self> {
        let mut analysis_algorithms: Vec<Box<dyn LockAnalysisAlgorithm + Send + Sync>> = Vec::new();

        // Initialize lock analysis algorithms with zero values for the algorithm objects;
        // actual analysis values are computed per-run from trace data
        analysis_algorithms.push(Box::new(ContentionFrequencyAnalysis::new(0.0, Vec::new())));
        analysis_algorithms.push(Box::new(HoldTimeAnalysis::new()));
        analysis_algorithms.push(Box::new(WaitTimeAnalysis::new(0, 0)));
        analysis_algorithms.push(Box::new(DeadlockPotentialAnalysis::new()));

        Ok(Self {
            analysis_algorithms: Arc::new(Mutex::new(analysis_algorithms)),
            config,
        })
    }

    /// Analyzes lock contention in test execution data
    pub async fn analyze_lock_contention(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<LockAnalysisResult> {
        let start_time = Utc::now();

        if !self.config.enable_contention_analysis {
            return Ok(LockAnalysisResult {
                lock_events: Vec::new(),
                contention_summary: HashMap::new(),
                latency_bounds: HashMap::new(),
                optimization_recommendations: Vec::new(),
                algorithm_results: Vec::new(),
                analysis_duration: std::time::Duration::from_secs(0),
                confidence: 0.0,
            });
        }

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
                    // Build LockAnalysis with real values computed from lock_events
                    let related_events: Vec<&LockEvent> =
                        lock_events.iter().filter(|e| !e.lock_id.is_empty()).collect();

                    let avg_contention = if !related_events.is_empty() {
                        related_events.iter().map(|e| e.contention_level).sum::<f64>()
                            / related_events.len() as f64
                    } else {
                        0.0
                    };

                    let max_wait = related_events
                        .iter()
                        .filter_map(|e| e.wait_time)
                        .max()
                        .unwrap_or(Duration::from_secs(0));

                    let min_wait = related_events
                        .iter()
                        .filter_map(|e| e.wait_time)
                        .min()
                        .unwrap_or(Duration::from_secs(0));

                    let avg_wait = if !related_events.is_empty() {
                        let total_us: u64 = related_events
                            .iter()
                            .filter_map(|e| e.wait_time)
                            .map(|d| u64::try_from(d.as_micros()).unwrap_or(u64::MAX))
                            .sum();
                        Duration::from_micros(total_us / related_events.len() as u64)
                    } else {
                        Duration::from_secs(0)
                    };

                    let avg_hold = related_events
                        .iter()
                        .map(|e| e.duration)
                        .max()
                        .unwrap_or(Duration::from_secs(0));

                    let lock_analysis = LockAnalysis {
                        lock_id: algorithm_name.clone(),
                        lock_events: related_events.iter().map(|e| (*e).clone()).collect(),
                        contention_metrics: LockContentionMetrics::default(),
                        dependencies: Vec::new(),
                        analysis_timestamp: chrono::Utc::now(),
                        average_contention_level: avg_contention,
                        average_hold_time: avg_hold,
                        contention_events: Vec::new(),
                        max_wait_time: max_wait,
                        min_wait_time: min_wait,
                        average_wait_time: avg_wait,
                    };

                    algorithm_results.push(LockAlgorithmResult {
                        algorithm: algorithm_name,
                        analysis: analysis_string,
                        analysis_duration: duration,
                        confidence: self.calculate_lock_analysis_confidence(&lock_analysis),
                    });
                    lock_analyses_structs.push(lock_analysis);
                },
                Err(e) => {
                    log::warn!("Lock analysis algorithm failed: {}", e);
                },
            }
        }

        // Synthesize results
        let optimization_recommendations_vec =
            self.generate_lock_optimizations(&lock_analyses_structs);

        // Convert to expected types using real data from helper methods
        let contention_summary: HashMap<String, f64> = {
            let summary = self.synthesize_contention_summary(&lock_analyses_structs);
            let mut m = HashMap::new();
            m.insert(
                "total_contentions".to_string(),
                summary.total_contentions as f64,
            );
            for (resource, count) in &summary.contention_by_resource {
                m.insert(format!("contention_{}", resource), *count as f64);
            }
            m
        };
        let latency_bounds: HashMap<String, Duration> = if self.config.enable_dependency_analysis {
            let bounds = self.calculate_latency_bounds(&lock_analyses_structs);
            let mut m = HashMap::new();
            m.insert("min_latency".to_string(), bounds.min_latency);
            m.insert("max_latency".to_string(), bounds.max_latency);
            m.insert("average_latency".to_string(), bounds.average_latency);
            m
        } else {
            HashMap::new()
        };
        let optimization_recommendations: Vec<String> =
            optimization_recommendations_vec.iter().map(|o| format!("{:?}", o)).collect();

        let elapsed = Utc::now().signed_duration_since(start_time).to_std().unwrap_or_default();
        if elapsed > self.config.max_analysis_duration {
            log::warn!(
                "Lock analysis exceeded configured duration limit ({:?} > {:?})",
                elapsed,
                self.config.max_analysis_duration
            );
        }

        Ok(LockAnalysisResult {
            lock_events,
            contention_summary,
            latency_bounds,
            optimization_recommendations,
            algorithm_results,
            analysis_duration: elapsed,
            confidence: self.calculate_overall_lock_confidence(&lock_analyses_structs),
        })
    }

    /// Extracts lock events from test execution traces
    fn extract_lock_events(&self, test_data: &TestExecutionData) -> Result<Vec<LockEvent>> {
        let mut events = Vec::new();

        for trace in &test_data.execution_traces {
            match trace.operation.as_str() {
                "LockAcquire" | "LockAcquisition" => {
                    // Try to compute duration from the trace's lock_timeline entry for this resource
                    let duration = trace
                        .lock_timeline
                        .iter()
                        .find(|e| e.lock_id == trace.resource)
                        .map(|e| e.duration)
                        .unwrap_or(Duration::from_secs(0));

                    events.push(LockEvent {
                        timestamp: trace.timestamp,
                        lock_id: trace.resource.clone(),
                        event_type: "acquire".to_string(),
                        thread_id: trace.thread_id,
                        duration,
                        wait_time: None,
                        contention_level: 0.0,
                        performance_impact: 0.0,
                        deadlock_risk: 0.0,
                        alternatives: Vec::new(),
                        success: true,
                    });
                },
                "LockRelease" => {
                    // Try to compute duration from the trace's lock_timeline entry for this resource
                    let duration = trace
                        .lock_timeline
                        .iter()
                        .find(|e| e.lock_id == trace.resource)
                        .map(|e| e.duration)
                        .unwrap_or(Duration::from_secs(0));

                    events.push(LockEvent {
                        timestamp: trace.timestamp,
                        lock_id: trace.resource.clone(),
                        event_type: "release".to_string(),
                        thread_id: trace.thread_id,
                        duration,
                        wait_time: None,
                        contention_level: 0.0,
                        performance_impact: 0.0,
                        deadlock_risk: 0.0,
                        alternatives: Vec::new(),
                        success: true,
                    });
                },
                _ => {},
            }
        }

        Ok(events)
    }

    /// Calculates lock analysis confidence
    fn calculate_lock_analysis_confidence(&self, analysis: &LockAnalysis) -> f64 {
        let contention_factor = 1.0_f64 - analysis.average_contention_level;
        let hold_time_factor: f64 =
            if analysis.average_hold_time > Duration::from_millis(100) { 0.7 } else { 0.9 };

        (contention_factor + hold_time_factor) / 2.0
    }

    /// Synthesizes contention summary
    fn synthesize_contention_summary(&self, analyses: &[LockAnalysis]) -> ContentionSummary {
        let total_contentions = analyses.iter().map(|a| a.contention_events.len()).sum::<usize>();

        let avg_contention_level = analyses.iter().map(|a| a.average_contention_level).sum::<f64>()
            / analyses.len().max(1) as f64;

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
            average_contention_duration: Duration::from_secs_f64(avg_contention_level),
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
            analyses
                .iter()
                .map(|a| u64::try_from(a.average_wait_time.as_millis()).unwrap_or(u64::MAX))
                .sum::<u64>()
                / analyses.len().max(1) as u64,
        );

        LatencyBounds {
            min_latency,
            max_latency,
            average_latency: avg_latency,
        }
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
    fn calculate_overall_lock_confidence(&self, analyses: &[LockAnalysis]) -> f64 {
        if analyses.is_empty() {
            return 0.0;
        }

        let confidences: Vec<f64> =
            analyses.iter().map(|a| self.calculate_lock_analysis_confidence(a)).collect();

        confidences.iter().sum::<f64>() / confidences.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn make_trace(thread_id: u64, resource: &str, operation: &str) -> ExecutionTrace {
        ExecutionTrace {
            thread_id,
            resource: resource.to_string(),
            operation: operation.to_string(),
            timestamp: Instant::now(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_lock_contention_yields_events_on_acquire_traces() {
        let config = LockAnalysisConfig {
            enable_contention_analysis: true,
            enable_dependency_analysis: true,
            max_analysis_duration: std::time::Duration::from_secs(30),
        };
        let analyzer = LockContentionAnalyzer::new(config).await.expect("should create");

        let mut test_data = TestExecutionData::default();
        // Simulate two threads acquiring the same lock
        test_data.execution_traces.push(make_trace(1, "mutex_x", "LockAcquire"));
        test_data.execution_traces.push(make_trace(2, "mutex_x", "LockAcquire"));
        test_data.execution_traces.push(make_trace(1, "mutex_x", "LockRelease"));
        test_data.execution_traces.push(make_trace(2, "mutex_x", "LockRelease"));

        let result = analyzer.analyze_lock_contention(&test_data).await.expect("should analyze");

        // Should have detected lock events
        assert!(
            !result.lock_events.is_empty(),
            "should extract lock events from traces"
        );
        // All events should be for mutex_x
        assert!(
            result.lock_events.iter().all(|e| e.lock_id == "mutex_x"),
            "lock events should reference mutex_x"
        );
    }

    #[test]
    fn test_contention_frequency_analysis_construction() {
        let cfa = ContentionFrequencyAnalysis::new(5.0, vec!["mutex_x".to_string()]);
        assert!((cfa.frequency - 5.0).abs() < f64::EPSILON);
        assert_eq!(cfa.hotspots.len(), 1);
    }

    #[test]
    fn test_wait_time_analysis_construction() {
        let wta = WaitTimeAnalysis::new(100, 500);
        assert_eq!(wta.avg_wait_time_us, 100);
        assert_eq!(wta.max_wait_time_us, 500);
    }
}
