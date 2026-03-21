//! Concurrency Pattern Detector
//!
//! Recognizes and analyzes common concurrency patterns to optimize parallel
//! execution strategies and identify potential issues.

use super::super::types::*;
use anyhow::Result;
use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

pub struct ConcurrencyPatternDetector {
    /// Pattern detection algorithms
    detection_algorithms: Arc<Mutex<Vec<Box<dyn PatternDetectionAlgorithm + Send + Sync>>>>,

    /// Known pattern library
    pattern_library: Arc<RwLock<ConcurrencyPatternLibrary>>,

    /// Pattern performance database
    performance_db: Arc<RwLock<PatternPerformanceDatabase>>,

    /// Configuration
    config: PatternDetectionConfig,
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

        let pattern_library = ConcurrencyPatternLibrary::new();
        let performance_db = PatternPerformanceDatabase::new();

        Ok(Self {
            detection_algorithms: Arc::new(Mutex::new(detection_algorithms)),
            pattern_library: Arc::new(RwLock::new(pattern_library)),
            performance_db: Arc::new(RwLock::new(performance_db)),
            config,
        })
    }

    /// Detects concurrency patterns in test execution data
    pub async fn detect_concurrency_patterns(
        &self,
        _test_data: &TestExecutionData,
    ) -> Result<PatternAnalysisResult> {
        let start_time = Utc::now();

        // Execute synchronously to avoid lifetime issues with mutex guards
        let detection_results: Vec<_> = {
            let algorithms = self.detection_algorithms.lock();
            algorithms
                .iter()
                .map(|algorithm| {
                    let algorithm_name = algorithm.name().to_string();
                    let detection_start = Instant::now();
                    let result_string = algorithm.detect_patterns();
                    let result: Result<Vec<String>> = Ok(vec![result_string]);
                    let detection_duration = detection_start.elapsed();
                    (algorithm_name, result, detection_duration)
                })
                .collect()
        };

        // Collect detection results
        let mut detected_patterns_strings = Vec::new();
        let mut detected_patterns_structs = Vec::new(); // For helper methods
        let mut algorithm_results = Vec::new();

        for (algorithm_name, result, duration) in detection_results {
            match result {
                Ok(mut patterns) => {
                    // Create placeholder ConcurrencyPattern structs from String results
                    let pattern_structs: Vec<ConcurrencyPattern> = patterns
                        .iter()
                        .map(|pattern_str| ConcurrencyPattern {
                            pattern_type: pattern_str.clone(),
                            description: pattern_str.clone(),
                            characteristics: vec![pattern_str.clone()],
                            applicability: 0.5,
                            confidence: 0.5,
                            thread_count: 1,
                        })
                        .collect();

                    algorithm_results.push(PatternAlgorithmResult {
                        algorithm: algorithm_name,
                        patterns: patterns.clone(),
                        detection_duration: duration,
                        confidence: self.calculate_pattern_detection_confidence(&pattern_structs)
                            as f64,
                    });
                    detected_patterns_strings.append(&mut patterns);
                    detected_patterns_structs.extend(pattern_structs);
                },
                Err(e) => {
                    log::warn!("Pattern detection algorithm failed: {}", e);
                },
            }
        }

        // Deduplicate and classify patterns using structs
        let unique_patterns = self.deduplicate_patterns(&detected_patterns_structs);
        let classified_patterns = self.classify_patterns(&unique_patterns);

        // Analyze scalability characteristics
        let scalability_patterns_vec = self.analyze_scalability_patterns(&classified_patterns);

        // Generate pattern-based recommendations
        let pattern_recommendations =
            self.generate_pattern_recommendations(&classified_patterns).await?;

        // Convert types
        let scalability_patterns: Vec<String> =
            scalability_patterns_vec.iter().map(|p| format!("{:?}", p)).collect();

        // Extract ConcurrencyPattern from ClassifiedConcurrencyPattern
        let detected_patterns: Vec<ConcurrencyPattern> =
            classified_patterns.into_iter().map(|cp| cp.pattern).collect();

        Ok(PatternAnalysisResult {
            detected_patterns,
            scalability_patterns,
            pattern_recommendations,
            algorithm_results,
            timeout_requirements: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
            confidence: self.calculate_overall_pattern_confidence(&unique_patterns) as f64,
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

    /// Classifies patterns by type and characteristics
    fn classify_patterns(
        &self,
        patterns: &[ConcurrencyPattern],
    ) -> Vec<ClassifiedConcurrencyPattern> {
        patterns
            .iter()
            .map(|pattern| ClassifiedConcurrencyPattern {
                pattern: pattern.clone(),
                classification: self.classify_single_pattern(pattern),
                performance_characteristics: self.analyze_pattern_performance(pattern),
                optimization_potential: self.assess_optimization_potential(pattern).potential_score,
            })
            .collect()
    }

    /// Classifies a single pattern
    fn classify_single_pattern(&self, pattern: &ConcurrencyPattern) -> PatternClassification {
        let scalability = self.assess_pattern_scalability(pattern);
        let efficiency = self.assess_pattern_efficiency(pattern);

        PatternClassification {
            classification_type: pattern.pattern_type.clone(),
            confidence: pattern.confidence,
            categories: vec![pattern.pattern_type.clone()],
            primary_type: pattern.pattern_type.clone(),
            complexity_level: self.assess_pattern_complexity(pattern),
            // TODO: ScalabilityRating is now a struct, use its score field
            scalability_rating: scalability.score,
            efficiency_rating: match efficiency {
                EfficiencyRating::VeryLow => 0.1,
                EfficiencyRating::Low => 0.3,
                EfficiencyRating::Medium => 0.6,
                EfficiencyRating::High => 0.9,
                EfficiencyRating::VeryHigh => 1.0,
            },
        }
    }

    /// Assesses pattern complexity
    fn assess_pattern_complexity(&self, pattern: &ConcurrencyPattern) -> ComplexityLevel {
        // Match on string pattern_type field
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => ComplexityLevel::Medium,
            "MasterWorker" => ComplexityLevel::Simple,
            "Pipeline" => ComplexityLevel::Complex,
            "ForkJoin" => ComplexityLevel::Medium,
            _ => ComplexityLevel::Complex, // Default for Custom and unknown patterns
        }
    }

    /// Assesses pattern scalability
    fn assess_pattern_scalability(&self, pattern: &ConcurrencyPattern) -> ScalabilityRating {
        // TODO: ScalabilityRating is now a struct with rating: String and score: f64
        if pattern.thread_count > 8 {
            ScalabilityRating {
                rating: "High".to_string(),
                score: 0.9,
            }
        } else if pattern.thread_count > 4 {
            ScalabilityRating {
                rating: "Medium".to_string(),
                score: 0.6,
            }
        } else {
            ScalabilityRating {
                rating: "Low".to_string(),
                score: 0.3,
            }
        }
    }

    /// Assesses pattern efficiency
    fn assess_pattern_efficiency(&self, pattern: &ConcurrencyPattern) -> EfficiencyRating {
        // Match on string pattern_type field
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => EfficiencyRating::High,
            "MasterWorker" => EfficiencyRating::Medium,
            "Pipeline" => EfficiencyRating::High,
            "ForkJoin" => EfficiencyRating::Medium,
            _ => EfficiencyRating::Low, // Default for Custom and unknown patterns
        }
    }

    /// Analyzes pattern performance characteristics
    fn analyze_pattern_performance(
        &self,
        pattern: &ConcurrencyPattern,
    ) -> PatternPerformanceCharacteristics {
        let scaling = self.analyze_scaling_behavior(pattern);
        let throughput_factor = self.estimate_throughput_factor(pattern);
        let latency_impact = self.estimate_latency_impact(pattern);
        let resource_utilization = self.estimate_resource_utilization(pattern);

        PatternPerformanceCharacteristics {
            throughput: throughput_factor as f64,
            latency: Duration::from_millis((latency_impact * 1000.0) as u64),
            resource_efficiency: resource_utilization as f64,
            throughput_factor: throughput_factor as f64,
            latency_impact: latency_impact as f64,
            resource_utilization: resource_utilization as f64,
            // TODO: ScalingBehavior is now a struct with scaling_type: String
            scaling_behavior: scaling.scaling_type.clone(),
        }
    }

    /// Estimates throughput factor
    fn estimate_throughput_factor(&self, pattern: &ConcurrencyPattern) -> f32 {
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => 0.8,
            "MasterWorker" => 0.9,
            "Pipeline" => 0.95,
            "ForkJoin" => 0.7,
            _ => 0.6, // Custom or unknown
        }
    }

    /// Estimates latency impact
    fn estimate_latency_impact(&self, pattern: &ConcurrencyPattern) -> f32 {
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => 0.2,
            "MasterWorker" => 0.1,
            "Pipeline" => 0.3,
            "ForkJoin" => 0.4,
            _ => 0.5, // Custom or unknown
        }
    }

    /// Estimates resource utilization
    fn estimate_resource_utilization(&self, pattern: &ConcurrencyPattern) -> f32 {
        let base_utilization = match pattern.pattern_type.as_str() {
            "ProducerConsumer" => 0.75,
            "MasterWorker" => 0.85,
            "Pipeline" => 0.9,
            "ForkJoin" => 0.65,
            _ => 0.5, // Custom or unknown
        };

        // Adjust based on thread count
        let thread_factor = (pattern.thread_count as f32).log2() / 4.0;
        (base_utilization * (1.0 + thread_factor)).min(1.0)
    }

    /// Analyzes scaling behavior
    fn analyze_scaling_behavior(&self, pattern: &ConcurrencyPattern) -> ScalingBehavior {
        // TODO: ScalingBehavior is now a struct, not an enum
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => ScalingBehavior {
                scaling_type: "Linear".to_string(),
                scaling_efficiency: 0.9,
                optimal_scale: 8,
                scaling_limits: (1, 64),
            },
            "MasterWorker" => ScalingBehavior {
                scaling_type: "SubLinear".to_string(),
                scaling_efficiency: 0.7,
                optimal_scale: 4,
                scaling_limits: (1, 32),
            },
            "Pipeline" => ScalingBehavior {
                scaling_type: "Linear".to_string(),
                scaling_efficiency: 0.85,
                optimal_scale: 8,
                scaling_limits: (1, 64),
            },
            "ForkJoin" => ScalingBehavior {
                scaling_type: "SubLinear".to_string(),
                scaling_efficiency: 0.75,
                optimal_scale: 6,
                scaling_limits: (1, 48),
            },
            _ => ScalingBehavior {
                scaling_type: "Unknown".to_string(),
                scaling_efficiency: 0.5,
                optimal_scale: 4,
                scaling_limits: (1, 16),
            },
        }
    }

    /// Assesses optimization potential
    fn assess_optimization_potential(&self, pattern: &ConcurrencyPattern) -> OptimizationPotential {
        let throughput_improvement = self.estimate_throughput_improvement_potential(pattern);
        let latency_reduction = self.estimate_latency_reduction_potential(pattern);
        let resource_efficiency = self.estimate_resource_efficiency_potential(pattern);
        let complexity = self.estimate_optimization_complexity(pattern);

        let potential_score =
            (throughput_improvement + latency_reduction + resource_efficiency) / 3.0;

        let mut optimization_areas = Vec::new();
        if throughput_improvement > 0.3 {
            optimization_areas.push("Throughput".to_string());
        }
        if latency_reduction > 0.3 {
            optimization_areas.push("Latency".to_string());
        }
        if resource_efficiency > 0.3 {
            optimization_areas.push("ResourceEfficiency".to_string());
        }

        // TODO: OptimizationComplexity is now a struct with complexity_level: ComplexityLevel
        let feasibility = match complexity.complexity_level {
            ComplexityLevel::VerySimple | ComplexityLevel::Simple => 0.9,
            ComplexityLevel::Medium => 0.6,
            ComplexityLevel::Complex | ComplexityLevel::VeryComplex => 0.3,
            ComplexityLevel::HighlyComplex => 0.2,
            ComplexityLevel::ExtremelyComplex => 0.1,
        };

        OptimizationPotential {
            potential_score: potential_score as f64,
            optimization_areas,
            expected_improvement: ((throughput_improvement + latency_reduction) / 2.0) as f64,
            feasibility,
        }
    }

    /// Estimates throughput improvement potential
    fn estimate_throughput_improvement_potential(&self, pattern: &ConcurrencyPattern) -> f32 {
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => 0.3,
            "MasterWorker" => 0.2,
            "Pipeline" => 0.4,
            "ForkJoin" => 0.5,
            _ => 0.6, // Custom or other patterns
        }
    }

    /// Estimates latency reduction potential
    fn estimate_latency_reduction_potential(&self, pattern: &ConcurrencyPattern) -> f32 {
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => 0.2,
            "MasterWorker" => 0.1,
            "Pipeline" => 0.3,
            "ForkJoin" => 0.4,
            _ => 0.5, // Custom or other patterns
        }
    }

    /// Estimates resource efficiency potential
    fn estimate_resource_efficiency_potential(&self, pattern: &ConcurrencyPattern) -> f32 {
        let current_efficiency = self.estimate_resource_utilization(pattern);
        1.0 - current_efficiency
    }

    /// Estimates optimization complexity
    fn estimate_optimization_complexity(
        &self,
        pattern: &ConcurrencyPattern,
    ) -> OptimizationComplexity {
        // TODO: OptimizationComplexity is now a struct, not an enum
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => OptimizationComplexity {
                complexity_level: ComplexityLevel::Medium,
                complexity_score: 0.5,
                complexity_factors: vec!["Coordination overhead".to_string()],
            },
            "MasterWorker" => OptimizationComplexity {
                complexity_level: ComplexityLevel::Simple,
                complexity_score: 0.3,
                complexity_factors: vec!["Simple distribution".to_string()],
            },
            "Pipeline" => OptimizationComplexity {
                complexity_level: ComplexityLevel::Complex,
                complexity_score: 0.8,
                complexity_factors: vec!["Stage synchronization".to_string()],
            },
            "ForkJoin" => OptimizationComplexity {
                complexity_level: ComplexityLevel::Medium,
                complexity_score: 0.6,
                complexity_factors: vec!["Join coordination".to_string()],
            },
            _ => OptimizationComplexity {
                complexity_level: ComplexityLevel::Complex,
                complexity_score: 0.9,
                complexity_factors: vec!["Unknown pattern".to_string()],
            },
        }
    }

    /// Analyzes scalability patterns
    fn analyze_scalability_patterns(
        &self,
        patterns: &[ClassifiedConcurrencyPattern],
    ) -> Vec<ScalabilityPattern> {
        let mut scalability_patterns = Vec::new();

        for pattern in patterns {
            let pattern_type_str = format!("{:?}", pattern.pattern.pattern_type);
            let efficiency_curve_data = self.model_efficiency_curve(&pattern.pattern);
            let scalability_pattern = ScalabilityPattern {
                pattern_type: pattern_type_str,
                efficiency_curve: efficiency_curve_data
                    .data_points
                    .iter()
                    .map(|(_, y)| *y)
                    .collect(),
            };

            scalability_patterns.push(scalability_pattern);
        }

        scalability_patterns
    }

    /// Calculates scaling factor
    fn calculate_scaling_factor(&self, pattern: &ConcurrencyPattern) -> f32 {
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => 0.8,
            "MasterWorker" => 0.9,
            "Pipeline" => 0.85,
            "ForkJoin" => 0.75,
            _ => 0.6, // Custom or unknown
        }
    }

    /// Estimates optimal thread count
    fn estimate_optimal_threads(&self, pattern: &ConcurrencyPattern) -> usize {
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => 4,
            "MasterWorker" => 8,
            "Pipeline" => 6,
            "ForkJoin" => 4,
            _ => 2, // Custom or unknown
        }
    }

    /// Estimates saturation point
    fn estimate_saturation_point(&self, pattern: &ConcurrencyPattern) -> usize {
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => 8,
            "MasterWorker" => 16,
            "Pipeline" => 12,
            "ForkJoin" => 8,
            _ => 4, // Custom or unknown
        }
    }

    /// Models efficiency curve
    fn model_efficiency_curve(&self, pattern: &ConcurrencyPattern) -> EfficiencyCurve {
        let optimal_threads = self.estimate_optimal_threads(pattern) as f64;
        let saturation_point = self.estimate_saturation_point(pattern) as f64;

        // Generate data points for the efficiency curve
        let mut data_points = Vec::new();
        for i in 1..=32 {
            let threads = i as f64;
            let efficiency = if threads <= optimal_threads {
                threads / optimal_threads // Linear growth in optimal region
            } else if threads <= saturation_point {
                1.0 - (threads - optimal_threads) / (saturation_point - optimal_threads) * 0.2
            // Slight degradation
            } else {
                0.8 - (threads - saturation_point) / 32.0 * 0.3 // Further degradation
            };
            data_points.push((threads, efficiency));
        }

        EfficiencyCurve {
            data_points,
            curve_type: "Logarithmic".to_string(),
            peak_efficiency: 1.0,
            optimal_point: (optimal_threads, 1.0),
        }
    }

    /// Generates efficiency function
    fn generate_efficiency_function(&self, pattern: &ConcurrencyPattern) -> EfficiencyFunction {
        EfficiencyFunction {
            function_type: "Logarithmic".to_string(),
            parameters: vec![
                self.calculate_scaling_factor(pattern) as f64,
                self.estimate_optimal_threads(pattern) as f64,
                self.estimate_saturation_point(pattern) as f64,
            ],
            domain: (1.0, 128.0), // Thread count domain (1 to 128 threads)
            range: (0.0, 1.0),    // Efficiency range (0% to 100%)
        }
    }

    /// Generates pattern-based recommendations
    async fn generate_pattern_recommendations(
        &self,
        patterns: &[ClassifiedConcurrencyPattern],
    ) -> Result<Vec<PatternOptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        for pattern in patterns {
            // optimization_potential is f64, use it directly for threshold comparison
            if pattern.optimization_potential > 0.2 {
                let effort = self.convert_complexity_to_effort(0.5);
                recommendations.push(PatternOptimizationRecommendation {
                    pattern_type: pattern.pattern.pattern_type.clone(),
                    optimization_type: "ThroughputOptimization".to_string(),
                    description: "Significant throughput improvement potential detected"
                        .to_string(),
                    expected_improvement: pattern.optimization_potential,
                    implementation_effort: format!("{:?}", effort),
                    recommendations: vec!["Optimize throughput".to_string()], // TODO: parse pattern_type to enum
                });
            }

            if pattern.optimization_potential > 0.3 {
                let effort = self.convert_complexity_to_effort(0.5);
                recommendations.push(PatternOptimizationRecommendation {
                    pattern_type: pattern.pattern.pattern_type.clone(),
                    optimization_type: "LatencyOptimization".to_string(),
                    description: "Significant latency reduction potential detected".to_string(),
                    expected_improvement: pattern.optimization_potential,
                    implementation_effort: format!("{:?}", effort),
                    recommendations: vec!["Reduce latency".to_string()], // TODO: parse pattern_type to enum
                });
            }
        }

        Ok(recommendations)
    }

    /// Converts optimization complexity to effort
    fn convert_complexity_to_effort(&self, complexity: f64) -> OptimizationEffort {
        if complexity < 0.3 {
            OptimizationEffort::Low
        } else if complexity < 0.6 {
            OptimizationEffort::Medium
        } else {
            OptimizationEffort::High
        }
    }

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
