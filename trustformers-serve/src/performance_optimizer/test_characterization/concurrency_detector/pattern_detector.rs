//! Concurrency Pattern Detector
//!
//! Recognizes and analyzes common concurrency patterns to optimize parallel
//! execution strategies and identify potential issues.

use super::super::types::*;
use anyhow::Result;
use chrono::Utc;
use parking_lot::Mutex;
use std::{sync::Arc, time::Instant};

pub struct ConcurrencyPatternDetector {
    /// Pattern detection algorithms
    detection_algorithms: Arc<Mutex<Vec<Box<dyn PatternDetectionAlgorithm + Send + Sync>>>>,
    /// Detection configuration.
    ///
    /// Until 0.2.1 `new` took this and dropped it (`_config`), so
    /// `detection_enabled`, `min_confidence` and `max_patterns_to_detect` had
    /// no effect on anything.
    config: PatternDetectionConfig,
}

// `build_pattern_from_string` was deleted in 0.2.1. It turned a detector's
// output string into a `ConcurrencyPattern` by looking the pattern name up in a
// table of invented numbers -- confidence 0.85/0.90/0.80/0.75, thread_count
// 4/8/6/4, applicability 0.85/0.90/0.80/0.75 -- none of which came from the
// test being analysed. The detectors now return a `ConcurrencyPattern` whose
// every field is measured from the recorded thread interactions.

/// Parses a pattern type string into the ConcurrencyPatternType enum
fn pattern_type_from_str(s: &str) -> ConcurrencyPatternType {
    match s {
        "ProducerConsumer" => ConcurrencyPatternType::ProducerConsumer,
        "MasterWorker" => ConcurrencyPatternType::MasterWorker,
        "Pipeline" => ConcurrencyPatternType::Pipeline,
        "ForkJoin" => ConcurrencyPatternType::ForkJoin,
        other => ConcurrencyPatternType::Custom(other.to_string()),
    }
}

impl ConcurrencyPatternDetector {
    /// Creates a new concurrency pattern detector
    pub async fn new(config: PatternDetectionConfig) -> Result<Self> {
        let mut detection_algorithms: Vec<Box<dyn PatternDetectionAlgorithm + Send + Sync>> =
            Vec::new();

        // Initialize pattern detection algorithms
        detection_algorithms.push(Box::new(ProducerConsumerDetection::new()));
        detection_algorithms.push(Box::new(MasterWorkerDetection::new()));
        detection_algorithms.push(Box::new(PipelineDetection::new()));
        detection_algorithms.push(Box::new(ForkJoinDetection::new()));

        Ok(Self {
            detection_algorithms: Arc::new(Mutex::new(detection_algorithms)),
            config,
        })
    }

    /// Detects concurrency patterns in test execution data.
    ///
    /// Returns an error when the test recorded no thread interactions: with
    /// nothing to analyse, no pattern can be confirmed and none can be ruled
    /// out. Before 0.2.1 this method ignored `test_data` outright and reported
    /// "no pattern detected" with confidence 1.0 for every test.
    pub async fn detect_concurrency_patterns(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<PatternAnalysisResult> {
        let start_time = Utc::now();

        if !self.config.detection_enabled {
            anyhow::bail!(
                "concurrency-pattern detection is disabled by configuration; no pattern \
                 analysis was performed for test '{}'",
                test_data.test_id
            );
        }

        if test_data.thread_interactions.is_empty() {
            anyhow::bail!(
                "test '{}' recorded no thread interactions; concurrency-pattern analysis has \
                 nothing to examine",
                test_data.test_id
            );
        }

        // Execute synchronously to avoid lifetime issues with mutex guards
        let detection_results: Vec<_> = {
            let algorithms = self.detection_algorithms.lock();
            algorithms
                .iter()
                .map(|algorithm| {
                    let algorithm_name = algorithm.name().to_string();
                    let detection_start = Instant::now();
                    let result = algorithm.detect(test_data);
                    let detection_duration = detection_start.elapsed();
                    (algorithm_name, result, detection_duration)
                })
                .collect()
        };

        // Collect detection results
        let mut detected_patterns_structs = Vec::new(); // For helper methods
        let mut algorithm_results = Vec::new();

        for (algorithm_name, result, duration) in detection_results {
            match result {
                Ok(found) => {
                    let pattern_structs: Vec<ConcurrencyPattern> = found.into_iter().collect();
                    algorithm_results.push(PatternAlgorithmResult {
                        algorithm: algorithm_name,
                        patterns: pattern_structs
                            .iter()
                            .map(|pattern| pattern.pattern_type.clone())
                            .collect(),
                        detection_duration: duration,
                        confidence: self.calculate_pattern_detection_confidence(&pattern_structs)
                            as f64,
                    });
                    detected_patterns_structs.extend(pattern_structs);
                },
                Err(e) => {
                    log::warn!("Pattern detection algorithm failed: {}", e);
                },
            }
        }

        // Deduplicate, then drop anything below the configured confidence bar
        // and cap the count. Until 0.2.1 the config was dropped in `new`, so
        // neither limit did anything.
        let mut detected_patterns = self.deduplicate_patterns(&detected_patterns_structs);
        detected_patterns.retain(|pattern| pattern.confidence >= self.config.min_confidence);
        detected_patterns.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
        detected_patterns.truncate(self.config.max_patterns_to_detect);

        let pattern_recommendations =
            self.generate_pattern_recommendations(&detected_patterns).await?;

        Ok(PatternAnalysisResult {
            // Scalability characterization needs a scaling experiment -- the
            // same workload run at several thread counts. This analyzer runs
            // the test once, so it has nothing to report here. Until 0.2.1 this
            // carried a 32-point efficiency curve computed from a lookup table
            // keyed on the pattern's name; see the note further down.
            scalability_patterns: Vec::new(),
            confidence: self.calculate_overall_pattern_confidence(&detected_patterns) as f64,
            detected_patterns,
            pattern_recommendations,
            algorithm_results,
            timeout_requirements: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
        })
    }

    /// Calculates pattern detection confidence
    fn calculate_pattern_detection_confidence(&self, patterns: &[ConcurrencyPattern]) -> f32 {
        if patterns.is_empty() {
            return 1.0;
        }

        patterns.iter().map(|p| p.confidence).sum::<f64>() as f32 / patterns.len() as f32
    }

    /// Deduplicates detected patterns
    fn deduplicate_patterns(&self, patterns: &[ConcurrencyPattern]) -> Vec<ConcurrencyPattern> {
        let mut unique_patterns = Vec::new();

        for pattern in patterns {
            let is_duplicate = unique_patterns
                .iter()
                .any(|existing: &ConcurrencyPattern| self.patterns_are_similar(existing, pattern));

            if !is_duplicate {
                unique_patterns.push(pattern.clone());
            }
        }

        unique_patterns
    }

    /// Checks if two patterns are similar
    fn patterns_are_similar(&self, a: &ConcurrencyPattern, b: &ConcurrencyPattern) -> bool {
        a.pattern_type == b.pattern_type
            && a.thread_count == b.thread_count
            && (a.confidence - b.confidence).abs() < 0.2
    }

    // ------------------------------------------------------------------
    // Deleted in 0.2.1: the lookup-table classification stage
    // ------------------------------------------------------------------
    //
    // Between detection and reporting sat fifteen methods that took a detected
    // `ConcurrencyPattern`, matched on its `pattern_type` *string*, and read
    // numbers out of a hardcoded table: `classify_patterns`,
    // `classify_single_pattern`, `assess_pattern_complexity`,
    // `assess_pattern_scalability`, `assess_pattern_efficiency`,
    // `analyze_pattern_performance`, `estimate_throughput_factor`,
    // `estimate_latency_impact`, `estimate_resource_utilization`,
    // `analyze_scaling_behavior`, `assess_optimization_potential`,
    // `estimate_throughput_improvement_potential`,
    // `estimate_latency_reduction_potential`,
    // `estimate_resource_efficiency_potential`,
    // `estimate_optimization_complexity`, `estimate_optimal_threads`,
    // `estimate_saturation_point`, `model_efficiency_curve` and
    // `analyze_scalability_patterns`.
    //
    // Nothing in any of them looked at the test. "Pipeline" always scored
    // throughput 0.95, latency 0.30, complexity 0.80, optimal thread count 6
    // and saturation point 12; "MasterWorker" always 0.90/0.10/0.30/8/16. The
    // efficiency curve was 32 points of arithmetic over those two constants,
    // and `analyze_scalability_patterns` published it as
    // `PatternAnalysisResult::scalability_patterns`, which
    // `ConcurrencyAnalyzer::derive_performance_guarantees` turns into strings
    // like "Scalability pattern: ..." -- performance guarantees derived from a
    // table keyed on a pattern name.
    //
    // Measuring any of this needs a scaling experiment: run the workload at
    // several thread counts and observe the throughput. The characterizer runs
    // a test once. So the stage is gone rather than reimplemented, and
    // `PatternAnalysisResult::scalability_patterns` is now always empty --
    // honestly so. What survives is what the detectors measure: the patterns
    // themselves, each with a confidence and thread count taken from the
    // recorded interaction graph.
    //
    // `ClassifiedConcurrencyPattern`, `PatternClassification`,
    // `ScalabilityRating`, `ScalingBehavior`, `OptimizationPotential`,
    // `OptimizationComplexity` and `EfficiencyCurve` still exist as types --
    // several are used elsewhere -- but this detector no longer manufactures
    // instances of them.

    /// Playbook entries for the patterns that were detected.
    ///
    /// ## Changed in 0.2.1
    ///
    /// This used to gate on `pattern.optimization_potential > 0.2` / `> 0.3`
    /// and publish that same number as `expected_improvement` -- a figure that
    /// came from the deleted lookup table above, not from the test, and was
    /// presented as an expected percentage gain. The advice itself is a static
    /// playbook keyed on the pattern kind, which is legitimate as advice; the
    /// numeric prediction attached to it was not, and `expected_improvement` is
    /// now `None`.
    pub(crate) async fn generate_pattern_recommendations(
        &self,
        patterns: &[ConcurrencyPattern],
    ) -> Result<Vec<PatternOptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        for pattern in patterns {
            let pattern_type = pattern_type_from_str(&pattern.pattern_type);
            recommendations.push(PatternOptimizationRecommendation {
                pattern_type: pattern.pattern_type.clone(),
                optimization_type: "ThroughputOptimization".to_string(),
                description: format!(
                    "Throughput playbook for the detected {} pattern",
                    pattern.pattern_type
                ),
                expected_improvement: None,
                implementation_effort: None,
                recommendations: self.generate_throughput_recommendations(&pattern_type),
            });
            recommendations.push(PatternOptimizationRecommendation {
                pattern_type: pattern.pattern_type.clone(),
                optimization_type: "LatencyOptimization".to_string(),
                description: format!(
                    "Latency playbook for the detected {} pattern",
                    pattern.pattern_type
                ),
                expected_improvement: None,
                implementation_effort: None,
                recommendations: self.generate_latency_recommendations(&pattern_type),
            });
        }

        Ok(recommendations)
    }

    // `convert_complexity_to_effort` was deleted with its only caller: it
    // bucketed the constant 0.5 that `generate_pattern_recommendations` passed
    // it, so every recommendation ever produced reported effort "Medium".

    /// Generates throughput recommendations
    fn generate_throughput_recommendations(
        &self,
        pattern_type: &ConcurrencyPatternType,
    ) -> Vec<String> {
        match pattern_type {
            ConcurrencyPatternType::ProducerConsumer => vec![
                "Implement bounded queues with optimal capacity".to_string(),
                "Use multiple producer/consumer threads".to_string(),
                "Consider lock-free queue implementations".to_string(),
            ],
            ConcurrencyPatternType::MasterWorker => vec![
                "Implement work stealing algorithms".to_string(),
                "Balance workload distribution".to_string(),
                "Use thread pool sizing based on workload".to_string(),
            ],
            ConcurrencyPatternType::Pipeline => vec![
                "Optimize pipeline stage parallelism".to_string(),
                "Balance pipeline stage processing times".to_string(),
                "Implement dynamic pipeline scaling".to_string(),
            ],
            ConcurrencyPatternType::ForkJoin => vec![
                "Optimize task granularity".to_string(),
                "Implement efficient join synchronization".to_string(),
                "Use recursive task decomposition".to_string(),
            ],
            ConcurrencyPatternType::Custom(_) => vec![
                "Analyze custom pattern for optimization opportunities".to_string(),
                "Consider standard pattern alternatives".to_string(),
            ],
        }
    }

    /// Generates latency recommendations
    fn generate_latency_recommendations(
        &self,
        pattern_type: &ConcurrencyPatternType,
    ) -> Vec<String> {
        match pattern_type {
            ConcurrencyPatternType::ProducerConsumer => vec![
                "Reduce queue wait times".to_string(),
                "Implement priority queues for urgent tasks".to_string(),
                "Minimize synchronization overhead".to_string(),
            ],
            ConcurrencyPatternType::MasterWorker => vec![
                "Reduce task distribution overhead".to_string(),
                "Implement worker affinity".to_string(),
                "Use batched task assignment".to_string(),
            ],
            ConcurrencyPatternType::Pipeline => vec![
                "Reduce inter-stage latency".to_string(),
                "Implement pipeline bypassing for urgent tasks".to_string(),
                "Optimize stage transition overhead".to_string(),
            ],
            ConcurrencyPatternType::ForkJoin => vec![
                "Minimize fork overhead".to_string(),
                "Optimize join synchronization".to_string(),
                "Implement early termination strategies".to_string(),
            ],
            ConcurrencyPatternType::Custom(_) => vec![
                "Analyze critical path for latency bottlenecks".to_string(),
                "Implement asynchronous operations where possible".to_string(),
            ],
        }
    }

    /// Calculates overall pattern confidence
    fn calculate_overall_pattern_confidence(&self, patterns: &[ConcurrencyPattern]) -> f32 {
        if patterns.is_empty() {
            return 0.0;
        }

        let confidences: Vec<f32> = patterns.iter().map(|p| p.confidence as f32).collect();

        confidences.iter().map(|&x| x as f64).sum::<f64>() as f32 / confidences.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `test_build_pattern_from_string_known_type` and `..._unknown_type` were
    // deleted in 0.2.1. They asserted that the lookup table's invented
    // confidence/applicability/thread_count sat "above placeholder 0.5" -- they
    // were regression tests *for* the fabrication. The detectors' real
    // behaviour is covered in `types/core/pattern_algorithms_tests.rs`, and the
    // end-to-end path is covered below.

    /// Builds a data-flow interaction for the end-to-end tests.
    fn flow(source_thread: u64, target_thread: u64) -> ThreadInteraction {
        ThreadInteraction {
            source_thread,
            target_thread,
            interaction_type: InteractionType::DataExchange,
            frequency: 4.0,
            analysis_duration: std::time::Duration::from_millis(1),
            data_patterns: Vec::new(),
            sync_requirements: Vec::new(),
            performance_impact: 0.0,
            optimization_opportunities: Vec::new(),
            safety_considerations: Vec::new(),
            from_thread: source_thread,
            to_thread: target_thread,
            timestamp: Utc::now(),
            resource: String::new(),
            strength: 1.0,
        }
    }

    /// A pattern fixture with values that are stated, not implied to be
    /// measured.
    fn fixture_pattern() -> ConcurrencyPattern {
        ConcurrencyPattern {
            pattern_type: "ProducerConsumer".to_string(),
            description: "fixture".to_string(),
            characteristics: vec!["fixture".to_string()],
            applicability: 0.5,
            confidence: 0.8,
            thread_count: 3,
        }
    }

    /// Regression: `detect_concurrency_patterns` ignored its `test_data`
    /// argument entirely and reported "no patterns, confidence 1.0" for every
    /// test ever passed to it. With nothing recorded it must now say so.
    #[tokio::test]
    async fn detect_concurrency_patterns_refuses_a_test_with_no_interactions() {
        let detector = ConcurrencyPatternDetector::new(PatternDetectionConfig {
            detection_enabled: true,
            min_confidence: 0.5,
            max_patterns_to_detect: 10,
        })
        .await
        .expect("detector constructs");
        let data = TestExecutionData {
            test_id: "empty".to_string(),
            ..TestExecutionData::default()
        };
        let error = detector
            .detect_concurrency_patterns(&data)
            .await
            .expect_err("no interactions means nothing to analyse");
        assert!(
            error.to_string().contains("no thread interactions"),
            "{error}"
        );
    }

    /// Regression: `new` took a `PatternDetectionConfig` and dropped it
    /// (`_config`), so `detection_enabled` had no effect and a detector
    /// configured off still analysed and reported.
    #[tokio::test]
    async fn detection_can_actually_be_switched_off() {
        let detector = ConcurrencyPatternDetector::new(PatternDetectionConfig {
            detection_enabled: false,
            min_confidence: 0.5,
            max_patterns_to_detect: 10,
        })
        .await
        .expect("detector constructs");
        let data = TestExecutionData {
            test_id: "disabled".to_string(),
            thread_interactions: vec![flow(1, 2), flow(1, 3), flow(2, 4), flow(3, 4)],
            ..TestExecutionData::default()
        };
        let error = detector
            .detect_concurrency_patterns(&data)
            .await
            .expect_err("a disabled detector must not report an analysis");
        assert!(
            error.to_string().contains("disabled by configuration"),
            "{error}"
        );
    }

    /// Regression: `min_confidence` was dropped along with the rest of the
    /// config, so low-confidence findings were reported unfiltered.
    #[tokio::test]
    async fn the_configured_confidence_bar_is_applied() {
        let data = TestExecutionData {
            test_id: "crosstalk".to_string(),
            // A star with one worker-to-worker edge: the master/worker finding
            // scores 3/4 = 0.75.
            thread_interactions: vec![flow(1, 2), flow(1, 3), flow(1, 4), flow(2, 3)],
            ..TestExecutionData::default()
        };

        let permissive = ConcurrencyPatternDetector::new(PatternDetectionConfig {
            detection_enabled: true,
            min_confidence: 0.5,
            max_patterns_to_detect: 10,
        })
        .await
        .expect("detector constructs");
        let found = permissive
            .detect_concurrency_patterns(&data)
            .await
            .expect("analysis runs")
            .detected_patterns;
        assert!(
            found.iter().any(|pattern| pattern.pattern_type == "MasterWorker"),
            "{found:?}"
        );

        let strict = ConcurrencyPatternDetector::new(PatternDetectionConfig {
            detection_enabled: true,
            min_confidence: 0.9,
            max_patterns_to_detect: 10,
        })
        .await
        .expect("detector constructs");
        let filtered = strict
            .detect_concurrency_patterns(&data)
            .await
            .expect("analysis runs")
            .detected_patterns;
        assert!(
            !filtered.iter().any(|pattern| pattern.pattern_type == "MasterWorker"),
            "a 0.75-confidence finding is below a 0.9 bar: {filtered:?}"
        );
    }

    /// Regression: `scalability_patterns` used to carry a 32-point efficiency
    /// curve computed from a lookup table keyed on the pattern's name, which
    /// `ConcurrencyAnalyzer` then published as a "performance guarantee".
    #[tokio::test]
    async fn no_scalability_curve_is_invented_from_a_single_run() {
        let detector = ConcurrencyPatternDetector::new(PatternDetectionConfig {
            detection_enabled: true,
            min_confidence: 0.5,
            max_patterns_to_detect: 10,
        })
        .await
        .expect("detector constructs");
        let data = TestExecutionData {
            test_id: "scalability".to_string(),
            thread_interactions: vec![flow(1, 2), flow(1, 3), flow(2, 4), flow(3, 4)],
            ..TestExecutionData::default()
        };
        let result = detector.detect_concurrency_patterns(&data).await.expect("analysis runs");
        assert!(
            result.scalability_patterns.is_empty(),
            "one run cannot characterize scaling: {:?}",
            result.scalability_patterns
        );
        assert!(
            !result.detected_patterns.is_empty(),
            "the patterns themselves are still measured and reported"
        );
    }

    /// Regression: the detected patterns' thread counts came from a lookup
    /// table (4/8/6/4), not from the test. A four-thread scatter/gather must
    /// report four threads.
    #[tokio::test]
    async fn detect_concurrency_patterns_measures_the_supplied_graph() {
        let detector = ConcurrencyPatternDetector::new(PatternDetectionConfig {
            detection_enabled: true,
            min_confidence: 0.5,
            max_patterns_to_detect: 10,
        })
        .await
        .expect("detector constructs");
        let data = TestExecutionData {
            test_id: "fork_join".to_string(),
            thread_interactions: vec![flow(1, 2), flow(1, 3), flow(2, 4), flow(3, 4)],
            ..TestExecutionData::default()
        };
        let result = detector.detect_concurrency_patterns(&data).await.expect("analysis runs");
        let fork_join = result
            .detected_patterns
            .iter()
            .find(|pattern| pattern.pattern_type == "ForkJoin")
            .expect("the scatter/gather shape is present");
        assert_eq!(fork_join.thread_count, 4);
        assert!(fork_join.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_generate_pattern_recommendations_throughput() {
        let config = PatternDetectionConfig {
            detection_enabled: true,
            min_confidence: 0.5,
            max_patterns_to_detect: 10,
        };
        let detector = ConcurrencyPatternDetector::new(config).await.expect("detector constructs");
        let patterns = vec![fixture_pattern()];
        let recs = detector
            .generate_pattern_recommendations(&patterns)
            .await
            .expect("playbook lookup never fails");
        assert!(!recs.is_empty());
        let throughput_recs: Vec<_> = recs
            .iter()
            .filter(|r| r.optimization_type == "ThroughputOptimization")
            .collect();
        assert!(!throughput_recs.is_empty());
        let rec = &throughput_recs[0];
        assert!(
            rec.recommendations.len() > 1,
            "should have multiple specific recommendations, not just one generic"
        );
        // Regression: `expected_improvement` used to carry a number from a
        // lookup table keyed on the pattern name, presented as a predicted gain.
        assert!(
            rec.expected_improvement.is_none(),
            "nothing here measures an improvement, so none is predicted"
        );
        assert!(rec.implementation_effort.is_none());
    }
}
