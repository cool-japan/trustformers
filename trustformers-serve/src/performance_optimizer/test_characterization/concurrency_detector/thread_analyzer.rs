//! Thread Interaction Analyzer
//!
//! Analyzes thread interactions, synchronization patterns, and communication
//! mechanisms for optimal concurrent execution.

use super::super::types::*;
use anyhow::Result;
use chrono::Utc;
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

pub struct ThreadInteractionAnalyzer {
    /// Analysis algorithms
    analysis_algorithms: Arc<Mutex<Vec<Box<dyn ThreadAnalysisAlgorithm + Send + Sync>>>>,

    /// Configuration
    config: ThreadAnalysisConfig,
}

impl ThreadInteractionAnalyzer {
    /// Creates a new thread interaction analyzer
    pub async fn new(config: ThreadAnalysisConfig) -> Result<Self> {
        let mut analysis_algorithms: Vec<Box<dyn ThreadAnalysisAlgorithm + Send + Sync>> =
            Vec::new();

        // Initialize thread analysis algorithms
        analysis_algorithms.push(Box::new(CommunicationPatternAnalysis::new()));
        analysis_algorithms.push(Box::new(SynchronizationAnalysis::new()));
        analysis_algorithms.push(Box::new(PerformanceImpactAnalysis::new(0.0, Vec::new())));
        analysis_algorithms.push(Box::new(ScalabilityAnalysis::new()));

        Ok(Self {
            analysis_algorithms: Arc::new(Mutex::new(analysis_algorithms)),
            config,
        })
    }

    /// Analyzes thread interactions in test execution data
    pub async fn analyze_thread_interactions(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<ThreadAnalysisResult> {
        let start_time = Utc::now();

        if !self.config.enable_interaction_analysis {
            return Ok(ThreadAnalysisResult {
                thread_interactions: Vec::new(),
                throughput_analysis: HashMap::new(),
                efficiency_metrics: HashMap::new(),
                interaction_patterns: Vec::new(),
                optimization_opportunities: Vec::new(),
                algorithm_results: Vec::new(),
                analysis_window: std::time::Duration::from_secs(0),
                confidence: 1.0,
            });
        }

        // Extract thread interaction data
        let thread_interactions = self.extract_thread_interactions(test_data)?;

        // Compute metrics from thread_interactions before entering the algorithm lock
        let thread_count = thread_interactions.len();
        let interaction_count = thread_interactions.len();
        let cpu_efficiency = if thread_count > 0 {
            (interaction_count as f64 / thread_count.max(1) as f64).min(1.0)
        } else {
            0.0
        };
        let sync_interactions = thread_interactions
            .iter()
            .filter(|i| matches!(i.interaction_type, InteractionType::Synchronization))
            .count();
        let synchronization_efficiency =
            1.0 - (sync_interactions as f64 / interaction_count.max(1) as f64).min(0.9);
        let performance_impact = if cpu_efficiency < 0.6 { 0.4 } else { 0.1 };

        // Run analysis algorithms synchronously to avoid lifetime issues
        let thread_analysis_results: Vec<_> = {
            let algorithms = self.analysis_algorithms.lock();
            algorithms
                .iter()
                .map(|algorithm| {
                    let algorithm_name = algorithm.name().to_string();
                    let analysis_start = Instant::now();
                    let result_string = algorithm.analyze_threads();
                    // Convert String result to ThreadAnalysis using computed metrics
                    let performance_metrics_map = {
                        let mut m: HashMap<u64, f64> = HashMap::new();
                        for interaction in &thread_interactions {
                            m.entry(interaction.source_thread)
                                .and_modify(|v| *v += interaction.strength)
                                .or_insert(interaction.strength);
                        }
                        m
                    };
                    let bottlenecks = if synchronization_efficiency < 0.5 {
                        vec!["High synchronization overhead".to_string()]
                    } else {
                        Vec::new()
                    };
                    let detected_patterns: Vec<String> =
                        vec![result_string].into_iter().filter(|s| !s.contains("No ")).collect();
                    let result: Result<ThreadAnalysis> = Ok(ThreadAnalysis {
                        thread_count,
                        interactions: thread_interactions.clone(),
                        performance_metrics: performance_metrics_map,
                        bottlenecks,
                        detected_patterns,
                        performance_impact,
                        baseline_throughput: thread_count as f64,
                        projected_throughput: thread_count as f64 * (1.0 + cpu_efficiency * 0.5),
                        scalability_factor: cpu_efficiency,
                        estimated_saturation_point: (thread_count * 2).max(4),
                        optimal_thread_count: thread_count.max(1),
                        cpu_efficiency,
                        memory_efficiency: 0.8,
                        synchronization_efficiency,
                    });
                    let analysis_duration = analysis_start.elapsed();
                    (algorithm_name, result, analysis_duration)
                })
                .collect()
        };

        // Collect analysis results
        let mut thread_analyses = Vec::new();
        let mut algorithm_results = Vec::new();

        for (algorithm_name, result, duration) in thread_analysis_results {
            match result {
                Ok(analysis) => {
                    algorithm_results.push(ThreadAlgorithmResult {
                        algorithm: algorithm_name,
                        analysis: analysis.clone(),
                        analysis_duration: duration,
                        confidence: self.calculate_thread_analysis_confidence(&analysis),
                    });
                    thread_analyses.push(analysis);
                },
                Err(e) => {
                    log::warn!("Thread analysis algorithm failed: {}", e);
                },
            }
        }

        // Synthesize results
        let _throughput_analysis_struct = self.synthesize_throughput_analysis(&thread_analyses);
        let _efficiency_metrics_struct = self.calculate_efficiency_metrics(&thread_analyses);
        let interaction_patterns_vec = self.identify_interaction_patterns(&thread_interactions);
        let optimization_opportunities_vec =
            self.identify_optimization_opportunities(&thread_analyses);

        // Convert to expected types
        let throughput_analysis: HashMap<u64, f64> = {
            let mut m = HashMap::new();
            for interaction in &thread_interactions {
                m.entry(interaction.source_thread)
                    .and_modify(|v| *v += interaction.frequency)
                    .or_insert(interaction.frequency);
            }
            m
        };
        let efficiency_metrics: HashMap<String, f64> = {
            let mut m = HashMap::new();
            if !thread_analyses.is_empty() {
                let avg_cpu = thread_analyses.iter().map(|a| a.cpu_efficiency).sum::<f64>()
                    / thread_analyses.len() as f64;
                let avg_mem = thread_analyses.iter().map(|a| a.memory_efficiency).sum::<f64>()
                    / thread_analyses.len() as f64;
                let avg_sync =
                    thread_analyses.iter().map(|a| a.synchronization_efficiency).sum::<f64>()
                        / thread_analyses.len() as f64;
                m.insert("cpu_efficiency".to_string(), avg_cpu);
                m.insert("memory_efficiency".to_string(), avg_mem);
                m.insert("synchronization_efficiency".to_string(), avg_sync);
            }
            m
        };
        let interaction_patterns: Vec<String> = interaction_patterns_vec
            .iter()
            .map(|p| {
                format!(
                    "{} (freq:{:.2}, confidence:{:.2})",
                    p.description, p.frequency, p.confidence
                )
            })
            .collect();
        let optimization_opportunities: Vec<String> =
            optimization_opportunities_vec.iter().map(|o| format!("{:?}", o)).collect();

        // Filter interactions below the minimum frequency threshold
        let filtered_interactions: Vec<ThreadInteraction> = thread_interactions
            .into_iter()
            .filter(|i| i.frequency >= self.config.min_interaction_frequency)
            .collect();

        let elapsed = Utc::now().signed_duration_since(start_time).to_std().unwrap_or_default();
        if elapsed > self.config.analysis_timeout {
            log::warn!(
                "Thread analysis exceeded configured timeout ({:?} > {:?})",
                elapsed,
                self.config.analysis_timeout
            );
        }

        Ok(ThreadAnalysisResult {
            thread_interactions: filtered_interactions,
            throughput_analysis,
            efficiency_metrics,
            interaction_patterns,
            optimization_opportunities,
            algorithm_results,
            analysis_window: elapsed,
            confidence: self.calculate_overall_thread_confidence(&thread_analyses),
        })
    }

    /// Extracts thread interaction data from test execution traces
    fn extract_thread_interactions(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<Vec<ThreadInteraction>> {
        let mut interactions = Vec::new();
        let mut thread_activities: HashMap<u64, Vec<&ExecutionTrace>> = HashMap::new();

        // Group traces by thread
        for trace in &test_data.execution_traces {
            thread_activities.entry(trace.thread_id).or_default().push(trace);
        }

        // Identify interactions between threads
        for (thread_id, traces) in &thread_activities {
            for trace in traces {
                // Look for interactions with other threads
                for (other_thread_id, other_traces) in &thread_activities {
                    if thread_id != other_thread_id {
                        for other_trace in other_traces {
                            if self.traces_interact(trace, other_trace) {
                                // Convert ThreadInteractionType to InteractionType
                                let thread_interaction_type =
                                    self.determine_interaction_type(trace, other_trace);
                                let interaction_type =
                                    match thread_interaction_type.interaction_type.as_str() {
                                        "ReadWrite" | "WriteRead" | "WriteWrite" | "ReadRead" => {
                                            InteractionType::SharedMemory
                                        },
                                        "LockContention" => InteractionType::Synchronization,
                                        "ChannelCommunication" => InteractionType::MessagePassing,
                                        _ => InteractionType::SharedMemory, // Default fallback
                                    };

                                interactions.push(ThreadInteraction {
                                    source_thread: *thread_id,
                                    target_thread: *other_thread_id,
                                    from_thread: *thread_id,
                                    to_thread: *other_thread_id,
                                    interaction_type,
                                    frequency: 1.0,
                                    // Calculate analysis duration from thread_timeline
                                    analysis_duration: self.calculate_trace_duration(trace),
                                    data_patterns: vec![],
                                    sync_requirements: vec![],
                                    performance_impact: 0.0,
                                    optimization_opportunities: vec![],
                                    safety_considerations: vec![],
                                    timestamp: chrono::Utc::now(),
                                    resource: trace.resource.clone(),
                                    strength: self
                                        .calculate_interaction_strength(trace, other_trace),
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(interactions)
    }

    /// Checks if two traces represent an interaction
    fn traces_interact(&self, trace1: &ExecutionTrace, trace2: &ExecutionTrace) -> bool {
        // Check if traces access the same resource within a time window
        let resource_match = trace1.resource == trace2.resource;
        let time_window = Duration::from_millis(self.config.interaction_time_window_ms);
        let time_diff = if trace1.timestamp > trace2.timestamp {
            trace1
                .timestamp
                .checked_duration_since(trace2.timestamp)
                .unwrap_or(Duration::ZERO)
        } else {
            trace2
                .timestamp
                .checked_duration_since(trace1.timestamp)
                .unwrap_or(Duration::ZERO)
        };
        let time_proximity = time_diff < time_window;

        resource_match && time_proximity
    }

    /// Determines the type of interaction between threads
    fn determine_interaction_type(
        &self,
        trace1: &ExecutionTrace,
        trace2: &ExecutionTrace,
    ) -> ThreadInteractionType {
        // Match on string operation fields
        match (trace1.operation.as_str(), trace2.operation.as_str()) {
            ("Read", "Write") => ThreadInteractionType {
                interaction_type: "ReadWrite".to_string(),
                synchronization_required: true,
                typical_duration: Duration::from_millis(10),
            },
            ("Write", "Read") => ThreadInteractionType {
                interaction_type: "WriteRead".to_string(),
                synchronization_required: true,
                typical_duration: Duration::from_millis(10),
            },
            ("Write", "Write") => ThreadInteractionType {
                interaction_type: "WriteWrite".to_string(),
                synchronization_required: true,
                typical_duration: Duration::from_millis(15),
            },
            ("LockAcquire" | "LockAcquisition", "LockAcquire" | "LockAcquisition") => {
                ThreadInteractionType {
                    interaction_type: "LockContention".to_string(),
                    synchronization_required: true,
                    typical_duration: Duration::from_millis(20),
                }
            },
            _ => ThreadInteractionType {
                interaction_type: "Other".to_string(),
                synchronization_required: false,
                typical_duration: Duration::from_millis(1),
            },
        }
    }

    /// Calculates interaction strength
    fn calculate_interaction_strength(
        &self,
        trace1: &ExecutionTrace,
        trace2: &ExecutionTrace,
    ) -> f64 {
        let time_diff = if trace1.timestamp > trace2.timestamp {
            trace1
                .timestamp
                .checked_duration_since(trace2.timestamp)
                .unwrap_or(Duration::ZERO)
        } else {
            trace2
                .timestamp
                .checked_duration_since(trace1.timestamp)
                .unwrap_or(Duration::ZERO)
        };
        let time_diff_ms = time_diff.as_millis() as f64;
        let duration1 = self.calculate_trace_duration(trace1);
        let duration2 = self.calculate_trace_duration(trace2);
        let duration_factor = (duration1.as_millis() as f64 + duration2.as_millis() as f64) / 2.0;

        // Closer in time and longer duration = stronger interaction
        let time_strength = 1.0 / (1.0 + time_diff_ms / 1000.0);
        let duration_strength = (duration_factor / 100.0).min(1.0);

        (time_strength + duration_strength) / 2.0
    }

    /// Calculate duration of an execution trace from its timeline
    fn calculate_trace_duration(&self, trace: &ExecutionTrace) -> Duration {
        // Calculate from thread_timeline if available
        if let Some(timeline) = trace.thread_timeline.get(&trace.thread_id) {
            if timeline.len() >= 2 {
                let first = timeline.first().map(|(t, _)| *t);
                let last = timeline.last().map(|(t, _)| *t);
                if let (Some(start), Some(end)) = (first, last) {
                    return end.duration_since(start);
                }
            }
        }
        // Default to 100ms if we can't calculate
        Duration::from_millis(100)
    }

    /// Calculates thread analysis confidence
    fn calculate_thread_analysis_confidence(&self, analysis: &ThreadAnalysis) -> f64 {
        // Count-based heuristic: more patterns = higher confidence
        let pattern_confidence: f64 = if analysis.detected_patterns.is_empty() {
            0.5
        } else if analysis.detected_patterns.len() > 3 {
            0.9
        } else {
            0.7
        };

        let metrics_confidence: f64 = if analysis.performance_impact > 0.0 {
            1.0 - analysis.performance_impact.abs()
        } else {
            0.8
        };

        (pattern_confidence + metrics_confidence) / 2.0
    }

    /// Synthesizes throughput analysis from multiple thread analyses
    fn synthesize_throughput_analysis(&self, analyses: &[ThreadAnalysis]) -> ThroughputAnalysis {
        let baseline_throughput = analyses.iter().map(|a| a.baseline_throughput).sum::<f64>()
            / analyses.len().max(1) as f64;

        let projected_throughput = analyses.iter().map(|a| a.projected_throughput).sum::<f64>()
            / analyses.len().max(1) as f64;

        ThroughputAnalysis {
            average_throughput: baseline_throughput,
            peak_throughput: projected_throughput,
            throughput_variance: (projected_throughput - baseline_throughput).abs(),
            throughput_trend: if projected_throughput > baseline_throughput {
                "Improving".to_string()
            } else {
                "Stable".to_string()
            },
        }
    }

    /// Calculates efficiency metrics
    fn calculate_efficiency_metrics(&self, analyses: &[ThreadAnalysis]) -> EfficiencyMetrics {
        let cpu_efficiency =
            analyses.iter().map(|a| a.cpu_efficiency).sum::<f64>() / analyses.len().max(1) as f64;

        let memory_efficiency = analyses.iter().map(|a| a.memory_efficiency).sum::<f64>()
            / analyses.len().max(1) as f64;

        let synchronization_efficiency =
            analyses.iter().map(|a| a.synchronization_efficiency).sum::<f64>()
                / analyses.len().max(1) as f64;

        let overall_efficiency =
            (cpu_efficiency + memory_efficiency + synchronization_efficiency) / 3.0;

        let efficiency_rating = if overall_efficiency >= 0.9 {
            EfficiencyRating::VeryHigh
        } else if overall_efficiency >= 0.75 {
            EfficiencyRating::High
        } else if overall_efficiency >= 0.5 {
            EfficiencyRating::Medium
        } else if overall_efficiency >= 0.25 {
            EfficiencyRating::Low
        } else {
            EfficiencyRating::VeryLow
        };

        EfficiencyMetrics {
            cpu_efficiency,
            memory_efficiency,
            io_efficiency: synchronization_efficiency,
            overall_efficiency,
            efficiency_rating,
        }
    }

    /// Identifies interaction patterns
    fn identify_interaction_patterns(
        &self,
        interactions: &[ThreadInteraction],
    ) -> Vec<InteractionPattern> {
        let mut patterns = Vec::new();

        // Identify common patterns: count SharedMemory interactions as read-write patterns
        let read_write_count = interactions
            .iter()
            .filter(|i| matches!(i.interaction_type, InteractionType::SharedMemory))
            .count();

        if read_write_count > 0 {
            let freq = read_write_count as f64 / interactions.len().max(1) as f64;
            patterns.push(InteractionPattern {
                pattern_type: "Concurrency".to_string(),
                interacting_components: vec![],
                interaction_frequency: freq,
                frequency: freq,
                confidence: 0.8,
                description: "Read-write interaction pattern detected".to_string(),
                impact: PatternImpact::Neutral,
            });
        }

        // Add more pattern detection logic as needed
        patterns
    }

    /// Identifies optimization opportunities
    fn identify_optimization_opportunities(
        &self,
        analyses: &[ThreadAnalysis],
    ) -> Vec<ThreadOptimizationOpportunity> {
        let mut opportunities = Vec::new();

        for analysis in analyses {
            if analysis.synchronization_efficiency < 0.7 {
                opportunities.push(ThreadOptimizationOpportunity {
                    opportunity_type: "reduce_synchronization".to_string(),
                    affected_threads: vec![],
                    expected_improvement: 0.3,
                    implementation_cost: "medium".to_string(),
                    description: "High synchronization overhead detected".to_string(),
                    implementation_effort: "medium".to_string(),
                    recommendations: vec![
                        "Consider using lock-free data structures".to_string(),
                        "Reduce critical section size".to_string(),
                    ],
                });
            }

            if analysis.cpu_efficiency < 0.6 {
                opportunities.push(ThreadOptimizationOpportunity {
                    opportunity_type: "improve_load_balancing".to_string(),
                    affected_threads: vec![],
                    expected_improvement: 0.4,
                    implementation_cost: "high".to_string(),
                    description: "Poor CPU utilization detected".to_string(),
                    implementation_effort: "high".to_string(),
                    recommendations: vec![
                        "Implement work stealing algorithms".to_string(),
                        "Balance workload distribution".to_string(),
                    ],
                });
            }
        }

        opportunities
    }

    /// Calculates overall thread confidence
    fn calculate_overall_thread_confidence(&self, analyses: &[ThreadAnalysis]) -> f64 {
        if analyses.is_empty() {
            return 0.0;
        }

        let confidences: Vec<f64> =
            analyses.iter().map(|a| self.calculate_thread_analysis_confidence(a)).collect();

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
    async fn test_thread_analysis_detects_patterns_on_shared_resource() {
        let config = ThreadAnalysisConfig {
            enable_interaction_analysis: true,
            min_interaction_frequency: 0.0,
            analysis_timeout: std::time::Duration::from_secs(30),
            interaction_time_window_ms: 10000,
        };
        let analyzer =
            ThreadInteractionAnalyzer::new(config).await.expect("should create analyzer");

        let mut test_data = TestExecutionData::default();
        // Thread 1 and thread 2 both access "shared_resource"
        test_data.execution_traces.push(make_trace(1, "shared_resource", "Read"));
        test_data.execution_traces.push(make_trace(2, "shared_resource", "Write"));

        let result =
            analyzer.analyze_thread_interactions(&test_data).await.expect("should analyze");

        // Should have detected interactions between thread 1 and thread 2
        assert!(
            !result.thread_interactions.is_empty(),
            "should detect thread interactions for shared resource"
        );
    }
}
