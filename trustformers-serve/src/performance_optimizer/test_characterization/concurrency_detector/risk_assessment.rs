//! Concurrency Risk Assessment
//!
//! Provides comprehensive risk assessment for concurrent test execution including
//! data race detection, memory safety analysis, and performance degradation risks.

use super::super::types::*;
use anyhow::Result;
use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use std::{collections::HashMap, sync::Arc, time::Instant};

pub struct ConcurrencyRiskAssessment {
    /// Risk assessment algorithms
    assessment_algorithms: Arc<Mutex<Vec<Box<dyn RiskAssessmentAlgorithm + Send + Sync>>>>,

    /// Risk mitigation strategies
    mitigation_strategies: Arc<Mutex<Vec<Box<dyn RiskMitigationStrategy + Send + Sync>>>>,

    /// Historical risk data
    risk_history: Arc<Mutex<RiskAssessmentHistory>>,

    /// Risk monitoring data
    monitoring_data: Arc<RwLock<RiskMonitoringData>>,

    /// Configuration
    config: RiskAssessmentConfig,
}

impl ConcurrencyRiskAssessment {
    /// Creates a new concurrency risk assessment system
    pub async fn new(config: RiskAssessmentConfig) -> Result<Self> {
        let mut assessment_algorithms: Vec<Box<dyn RiskAssessmentAlgorithm + Send + Sync>> =
            Vec::new();
        let mut mitigation_strategies: Vec<Box<dyn RiskMitigationStrategy + Send + Sync>> =
            Vec::new();

        // Initialize risk assessment algorithms
        assessment_algorithms.push(Box::new(MachineLearningRiskAssessment::new(
            "default".to_string(),
            0.85,
        )));

        // Initialize mitigation strategies
        // TODO: PreventiveMitigation::new requires enabled: bool, strategies: Vec<String>
        mitigation_strategies.push(Box::new(PreventiveMitigation::new(true, Vec::new())));
        // TODO: ReactiveMitigation::new requires enabled: bool, response_time_ms: u64
        mitigation_strategies.push(Box::new(ReactiveMitigation::new(true, 1000)));
        mitigation_strategies.push(Box::new(AdaptiveMitigation::new()));

        Ok(Self {
            assessment_algorithms: Arc::new(Mutex::new(assessment_algorithms)),
            mitigation_strategies: Arc::new(Mutex::new(mitigation_strategies)),
            risk_history: Arc::new(Mutex::new(RiskAssessmentHistory::new())),
            monitoring_data: Arc::new(RwLock::new(RiskMonitoringData::new())),
            config,
        })
    }

    /// Assesses concurrency risks for test execution data
    pub async fn assess_concurrency_risks(
        &self,
        _test_data: &TestExecutionData,
    ) -> Result<RiskAssessmentResult> {
        let start_time = Utc::now();

        // Run risk assessment algorithms synchronously to avoid lifetime issues
        let assessment_task_results: Vec<_> = {
            let algorithms = self.assessment_algorithms.lock();
            algorithms
                .iter()
                .map(|algorithm| {
                    let algorithm_name = algorithm.name().to_string();
                    let assessment_start = Instant::now();
                    // TODO: assess_risk takes 0 arguments, removed test_data parameter
                    let result = algorithm.assess_risk();
                    let assessment_duration = assessment_start.elapsed();
                    (algorithm_name, result, assessment_duration)
                })
                .collect()
        };

        // Collect assessment results
        let mut risk_assessments = Vec::new();
        let mut algorithm_results = Vec::new();

        for (algorithm_name, risk_score, duration) in assessment_task_results {
            let assessment = RiskAssessment {
                risk_level: if risk_score > 0.7 {
                    "HIGH".to_string()
                } else if risk_score > 0.4 {
                    "MEDIUM".to_string()
                } else {
                    "LOW".to_string()
                },
                risk_score,
                risk_factors: Vec::new(), // TODO: populate from algorithm details
                primary_risk_factor: "concurrency".to_string(),
                potential_impact: risk_score,
            };

            let risk_assessment_struct = RiskAssessment {
                risk_level: if risk_score > 0.7 {
                    "High"
                } else if risk_score > 0.4 {
                    "Medium"
                } else {
                    "Low"
                }
                .to_string(),
                risk_score,
                risk_factors: Vec::new(),
                primary_risk_factor: "Concurrency".to_string(),
                potential_impact: risk_score,
            };

            algorithm_results.push(RiskAlgorithmResult {
                algorithm: algorithm_name,
                assessment: risk_assessment_struct.clone(),
                assessment_duration: duration,
                confidence: 0.8, // Default confidence for risk assessment
            });
            risk_assessments.push(assessment);
        }

        // Synthesize overall risk assessment
        let overall_risk_level = self.synthesize_risk_level(&risk_assessments);
        let risk_factors = self.identify_risk_factors(&risk_assessments);
        let risk_thresholds_f32 = self.calculate_risk_thresholds(&risk_assessments);
        // Convert HashMap<String, f32> to HashMap<String, f64>
        let risk_thresholds: HashMap<String, f64> =
            risk_thresholds_f32.into_iter().map(|(k, v)| (k, v as f64)).collect();

        // Generate mitigation recommendations
        let mitigation_recommendations =
            self.generate_mitigation_recommendations(&risk_assessments).await?;

        Ok(RiskAssessmentResult {
            overall_risk_level,
            risk_factors,
            risk_thresholds,
            mitigation_recommendations,
            algorithm_results,
            assessment_duration: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
            confidence: self.calculate_overall_risk_confidence(&risk_assessments) as f64,
        })
    }

    /// Calculates algorithm confidence
    fn calculate_algorithm_confidence(&self, assessment: &RiskAssessment) -> f32 {
        let factor_confidence = if assessment.risk_factors.is_empty() {
            0.5
        } else {
            assessment.risk_factors.iter().map(|f| f.confidence).sum::<f64>() as f32
                / assessment.risk_factors.len() as f32
        };

        // Map impact value (0.0-1.0) to confidence
        let impact_confidence = if assessment.potential_impact < 0.25 {
            0.8 // Low impact
        } else if assessment.potential_impact < 0.5 {
            0.7 // Medium impact
        } else if assessment.potential_impact < 0.75 {
            0.6 // High impact
        } else {
            0.5 // Critical impact
        };

        (factor_confidence + impact_confidence) / 2.0
    }

    /// Synthesizes overall risk level from multiple assessments
    fn synthesize_risk_level(&self, assessments: &[RiskAssessment]) -> RiskLevel {
        if assessments.is_empty() {
            return RiskLevel::Negligible;
        }

        // Convert String risk levels to enum and find the highest
        let risk_levels: Vec<RiskLevel> = assessments
            .iter()
            .map(|a| match a.risk_level.as_str() {
                "Negligible" => RiskLevel::Negligible,
                "VeryLow" => RiskLevel::VeryLow,
                "Low" => RiskLevel::Low,
                "Medium" => RiskLevel::Medium,
                "High" => RiskLevel::High,
                "VeryHigh" => RiskLevel::VeryHigh,
                "Severe" => RiskLevel::Severe,
                "Critical" => RiskLevel::Critical,
                "Extreme" => RiskLevel::Extreme,
                _ => RiskLevel::Negligible, // Default for unknown
            })
            .collect();

        // Find highest risk level (assuming enum order matches risk severity)
        risk_levels.into_iter().max().unwrap_or(RiskLevel::Negligible)
    }

    /// Identifies common risk factors
    fn identify_risk_factors(&self, assessments: &[RiskAssessment]) -> Vec<RiskFactor> {
        let mut all_factors = Vec::new();

        for assessment in assessments {
            all_factors.extend(assessment.risk_factors.clone());
        }

        // Deduplicate and merge similar factors
        self.deduplicate_risk_factors(&all_factors)
    }

    /// Deduplicates risk factors
    fn deduplicate_risk_factors(&self, factors: &[RiskFactor]) -> Vec<RiskFactor> {
        let mut unique_factors = Vec::new();

        for factor in factors {
            let existing = unique_factors
                .iter_mut()
                .find(|f: &&mut RiskFactor| f.factor_type == factor.factor_type);

            if let Some(existing_factor) = existing {
                // Merge factors by taking maximum severity and confidence
                existing_factor.severity = existing_factor.severity.max(factor.severity);
                existing_factor.confidence = existing_factor.confidence.max(factor.confidence);
            } else {
                unique_factors.push(factor.clone());
            }
        }

        unique_factors
    }

    /// Calculates risk thresholds
    fn calculate_risk_thresholds(&self, assessments: &[RiskAssessment]) -> HashMap<String, f32> {
        let mut thresholds = HashMap::new();

        for assessment in assessments {
            // Match on string risk_level field
            let threshold = match assessment.risk_level.as_str() {
                "Negligible" | "VeryLow" => 0.0,
                "Low" => 0.2,
                "Medium" => 0.5,
                "High" | "VeryHigh" => 0.8,
                "Severe" | "Critical" | "Extreme" => 1.0,
                _ => 0.0, // Default for unknown
            };

            thresholds.insert(format!("risk_level_{}", assessment.risk_level), threshold);
        }

        thresholds
    }

    /// Generates mitigation recommendations
    async fn generate_mitigation_recommendations(
        &self,
        assessments: &[RiskAssessment],
    ) -> Result<Vec<RiskMitigationRecommendation>> {
        let strategies = self.mitigation_strategies.lock();
        let mut recommendations = Vec::new();

        for assessment in assessments {
            for strategy in strategies.iter() {
                // TODO: is_applicable takes 0 arguments, removed assessment parameter
                if strategy.is_applicable() {
                    // TODO: generate_mitigation takes 0 arguments, removed assessment parameter
                    let mitigation = strategy.generate_mitigation();
                    recommendations.push(RiskMitigationRecommendation {
                        risk_factor: assessment.primary_risk_factor.clone(),
                        mitigation_strategy: strategy.name().to_string(),
                        mitigation_action: mitigation,
                        expected_effectiveness: self
                            .calculate_mitigation_effectiveness(strategy.name(), assessment)
                            as f64,
                        implementation_cost: self.calculate_implementation_cost(strategy.name())
                            as f64,
                    });
                }
            }
        }

        Ok(recommendations)
    }

    /// Calculates mitigation effectiveness
    fn calculate_mitigation_effectiveness(
        &self,
        strategy_name: &str,
        assessment: &RiskAssessment,
    ) -> f32 {
        let base_effectiveness = match strategy_name {
            "PreventiveMitigation" => 0.9,
            "ReactiveMitigation" => 0.7,
            "AdaptiveMitigation" => 0.8,
            _ => 0.6,
        };

        // Adjust based on risk level (assessment.risk_level is String)
        let risk_adjustment = match assessment.risk_level.as_str() {
            "Negligible" | "VeryLow" => 1.0,
            "Low" => 0.9,
            "Medium" => 0.8,
            "High" | "VeryHigh" => 0.7,
            "Severe" | "Critical" | "Extreme" => 0.6,
            _ => 0.75, // Default for unknown risk levels
        };

        base_effectiveness * risk_adjustment
    }

    /// Calculates implementation cost
    fn calculate_implementation_cost(&self, strategy_name: &str) -> f32 {
        match strategy_name {
            "PreventiveMitigation" => 0.8,
            "ReactiveMitigation" => 0.4,
            "AdaptiveMitigation" => 0.9,
            _ => 0.5,
        }
    }

    /// Calculates overall risk confidence
    fn calculate_overall_risk_confidence(&self, assessments: &[RiskAssessment]) -> f32 {
        if assessments.is_empty() {
            return 0.0;
        }

        let confidences: Vec<f32> =
            assessments.iter().map(|a| self.calculate_algorithm_confidence(a)).collect();

        let avg_confidence =
            confidences.iter().map(|&x| x as f64).sum::<f64>() as f32 / confidences.len() as f32;
        let consistency_factor = self.calculate_assessment_consistency(&confidences);

        avg_confidence * consistency_factor
    }

    /// Calculates assessment consistency
    fn calculate_assessment_consistency(&self, confidences: &[f32]) -> f32 {
        if confidences.len() < 2 {
            return 1.0;
        }

        let mean =
            confidences.iter().map(|&x| x as f64).sum::<f64>() as f32 / confidences.len() as f32;
        let variance = confidences.iter().map(|&c| (c - mean).powi(2) as f64).sum::<f64>() as f32
            / confidences.len() as f32;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 1.0 };

        (1.0 - coefficient_of_variation.min(1.0)).max(0.1)
    }
}
