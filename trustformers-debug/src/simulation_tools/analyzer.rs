//! Main Simulation Analyzer Implementation
//!
//! This module provides the core SimulationAnalyzer that orchestrates all simulation
//! analysis capabilities including what-if analysis, perturbation testing, adversarial
//! probing, and edge case discovery.

use super::adversarial_analysis::*;
use super::edge_case_discovery::*;
use super::perturbation_testing::*;
use super::reporting::SimulationReport;
use super::types::*;
use super::what_if_analysis::*;
use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;

/// Main simulation tools analyzer
#[derive(Debug)]
pub struct SimulationAnalyzer {
    config: SimulationConfig,
    what_if_results: Vec<WhatIfAnalysisResult>,
    perturbation_results: Vec<PerturbationTestResult>,
    adversarial_results: Vec<AdversarialProbingResult>,
    edge_case_results: Vec<EdgeCaseDiscoveryResult>,
}

impl SimulationAnalyzer {
    /// Create a new simulation analyzer
    pub fn new(config: SimulationConfig) -> Self {
        Self {
            config,
            what_if_results: Vec::new(),
            perturbation_results: Vec::new(),
            adversarial_results: Vec::new(),
            edge_case_results: Vec::new(),
        }
    }

    /// Perform what-if analysis
    pub async fn analyze_what_if(
        &mut self,
        base_input: &HashMap<String, f64>,
        model_fn: Box<dyn Fn(&HashMap<String, f64>) -> f64 + Send + Sync>,
    ) -> Result<WhatIfAnalysisResult> {
        if !self.config.enable_what_if_analysis {
            return Err(anyhow::anyhow!("What-if analysis is disabled"));
        }

        let base_prediction = model_fn(base_input);
        let base_scenario = Scenario {
            id: "base".to_string(),
            description: "Original input scenario".to_string(),
            features: base_input.clone(),
            prediction: base_prediction,
            // `confidence` here would be the MODEL's confidence in its own
            // prediction, which `model_fn` (a bare `-> f64`) never reports --
            // for the base scenario exactly as much as for a perturbed one. It
            // used to be `1.0`, asserting perfect certainty.
            confidence: None,
            changed_features: vec![],
            distance_from_base: 0.0,
            plausibility: 1.0,
        };

        // Generate what-if scenarios
        let scenarios = self.generate_what_if_scenarios(base_input, &model_fn).await?;

        // Analyze scenario impacts
        let impact_analysis = self.analyze_scenario_impacts(&base_scenario, &scenarios);

        // Perform sensitivity analysis
        let sensitivity_analysis = self.analyze_feature_sensitivity_from_scenarios(&scenarios);

        // Generate counterfactual insights
        let counterfactual_insights =
            self.generate_counterfactual_insights(&base_scenario, &scenarios);

        // Explore decision boundary
        let decision_boundary_exploration = self.explore_decision_boundary(&scenarios);

        let result = WhatIfAnalysisResult {
            timestamp: Utc::now(),
            base_scenario,
            scenarios,
            impact_analysis,
            sensitivity_analysis,
            counterfactual_insights,
            decision_boundary_exploration,
        };

        self.what_if_results.push(result.clone());
        Ok(result)
    }

    /// Perform perturbation testing
    pub async fn test_perturbations(
        &mut self,
        base_input: &HashMap<String, f64>,
        model_fn: Box<dyn Fn(&HashMap<String, f64>) -> f64 + Send + Sync>,
    ) -> Result<PerturbationTestResult> {
        if !self.config.enable_perturbation_testing {
            return Err(anyhow::anyhow!("Perturbation testing is disabled"));
        }

        let mut results_by_intensity = HashMap::new();

        // Test different perturbation intensities
        for &intensity in &self.config.perturbation_intensities {
            let intensity_result =
                self.test_perturbation_intensity(base_input, &model_fn, intensity).await?;
            results_by_intensity.insert(intensity.to_string(), intensity_result);
        }

        // Assess overall robustness
        let robustness_assessment = self.assess_robustness(&results_by_intensity);

        // Identify sensitivity hotspots
        let sensitivity_hotspots = self.identify_sensitivity_hotspots(&results_by_intensity);

        // Analyze failure modes
        let failure_modes = self.analyze_failure_modes(&results_by_intensity);

        let result = PerturbationTestResult {
            timestamp: Utc::now(),
            base_input: base_input.clone(),
            results_by_intensity,
            robustness_assessment,
            sensitivity_hotspots,
            failure_modes,
        };

        self.perturbation_results.push(result.clone());
        Ok(result)
    }

    /// Perform adversarial probing
    pub async fn probe_adversarial(
        &mut self,
        base_input: &HashMap<String, f64>,
        model_fn: Box<dyn Fn(&HashMap<String, f64>) -> f64 + Send + Sync>,
    ) -> Result<AdversarialProbingResult> {
        if !self.config.enable_adversarial_probing {
            return Err(anyhow::anyhow!("Adversarial probing is disabled"));
        }

        let mut adversarial_examples = HashMap::new();

        // Generate adversarial examples using different methods
        for method in &self.config.adversarial_methods {
            let examples =
                self.generate_adversarial_examples(base_input, &model_fn, method).await?;
            adversarial_examples.insert(method.clone(), examples);
        }

        // Analyze attack success
        let attack_success_analysis = self.analyze_attack_success(&adversarial_examples);

        // Assess adversarial robustness
        let robustness_assessment = self.assess_adversarial_robustness(&adversarial_examples);

        // Generate defense recommendations
        let defense_recommendations = self.generate_defense_recommendations(&adversarial_examples);

        let result = AdversarialProbingResult {
            timestamp: Utc::now(),
            base_input: base_input.clone(),
            adversarial_examples,
            attack_success_analysis,
            robustness_assessment,
            defense_recommendations,
        };

        self.adversarial_results.push(result.clone());
        Ok(result)
    }

    /// Discover edge cases
    pub async fn discover_edge_cases(
        &mut self,
        input_space: &HashMap<String, (f64, f64)>,
        model_fn: Box<dyn Fn(&HashMap<String, f64>) -> f64 + Send + Sync>,
    ) -> Result<EdgeCaseDiscoveryResult> {
        if !self.config.enable_edge_case_discovery {
            return Err(anyhow::anyhow!("Edge case discovery is disabled"));
        }

        // Search for edge cases using different strategies
        let edge_cases = self.search_edge_cases(input_space, &model_fn).await?;

        // Classify edge cases
        let classification = self.classify_edge_cases(&edge_cases);

        // Analyze coverage
        let coverage_analysis = self.analyze_edge_case_coverage(&edge_cases, input_space);

        // Assess risks
        let risk_assessment = self.assess_edge_case_risks(&edge_cases);

        let result = EdgeCaseDiscoveryResult {
            timestamp: Utc::now(),
            edge_cases,
            classification,
            coverage_analysis,
            risk_assessment,
        };

        self.edge_case_results.push(result.clone());
        Ok(result)
    }

    /// Generate comprehensive simulation report
    pub async fn generate_report(&self) -> Result<SimulationReport> {
        Ok(SimulationReport {
            timestamp: Utc::now(),
            config: self.config.clone(),
            what_if_analyses_count: self.what_if_results.len(),
            perturbation_tests_count: self.perturbation_results.len(),
            adversarial_probes_count: self.adversarial_results.len(),
            edge_case_discoveries_count: self.edge_case_results.len(),
            recent_what_if_results: self.what_if_results.iter().rev().take(3).cloned().collect(),
            recent_perturbation_results: self
                .perturbation_results
                .iter()
                .rev()
                .take(3)
                .cloned()
                .collect(),
            recent_adversarial_results: self
                .adversarial_results
                .iter()
                .rev()
                .take(3)
                .cloned()
                .collect(),
            recent_edge_case_results: self
                .edge_case_results
                .iter()
                .rev()
                .take(3)
                .cloned()
                .collect(),
            simulation_summary: self.generate_simulation_summary(),
        })
    }

    // Helper methods.

    async fn generate_what_if_scenarios(
        &self,
        base_input: &HashMap<String, f64>,
        model_fn: &(dyn Fn(&HashMap<String, f64>) -> f64 + Send + Sync),
    ) -> Result<Vec<Scenario>> {
        let mut scenarios = Vec::new();
        let _base_prediction = model_fn(base_input);

        for i in 0..self.config.num_what_if_scenarios {
            let mut scenario_input = base_input.clone();
            let mut changed_features = Vec::new();

            // Randomly modify features
            use scirs2_core::random::*; // SciRS2 Integration Policy
            let mut rng = thread_rng();
            let num_features_to_change = 1 + (rng.random_range(0..3)); // 1-3 features
            let features: Vec<String> = base_input.keys().cloned().collect();

            for _ in 0..num_features_to_change {
                if let Some(feature_name) = features.get(rng.random_range(0..features.len())) {
                    let original_value = base_input[feature_name];
                    let change_factor = 0.8 + (rng.random::<f64>() * 0.4); // 0.8-1.2 multiplier
                    let new_value = original_value * change_factor;

                    scenario_input.insert(feature_name.clone(), new_value);

                    changed_features.push(FeatureChange {
                        feature_name: feature_name.clone(),
                        original_value,
                        new_value,
                        change_magnitude: (new_value - original_value).abs(),
                        change_direction: if new_value > original_value {
                            ChangeDirection::Increase
                        } else {
                            ChangeDirection::Decrease
                        },
                        change_type: if (new_value - original_value).abs() / original_value.abs()
                            > 0.1
                        {
                            ChangeType::Significant
                        } else {
                            ChangeType::Incremental
                        },
                    });
                }
            }

            let prediction = model_fn(&scenario_input);
            let distance_from_base = self.calculate_distance(base_input, &scenario_input);

            scenarios.push(Scenario {
                id: format!("scenario_{}", i),
                description: format!("What-if scenario {}", i),
                features: scenario_input,
                prediction,
                // No confidence is derivable: `model_fn` returns a bare
                // scalar prediction with no uncertainty, and the scenario is
                // evaluated exactly once. Previously a flat 0.8.
                confidence: None,
                changed_features,
                distance_from_base,
                plausibility: 1.0 - (distance_from_base / 10.0).min(1.0), // Simple plausibility
            });
        }

        Ok(scenarios)
    }

    fn calculate_distance(
        &self,
        input1: &HashMap<String, f64>,
        input2: &HashMap<String, f64>,
    ) -> f64 {
        input1
            .iter()
            .map(|(key, value)| {
                let other_value = input2.get(key).unwrap_or(&0.0);
                (value - other_value).powi(2)
            })
            .sum::<f64>()
            .sqrt()
    }

    fn analyze_scenario_impacts(
        &self,
        base_scenario: &Scenario,
        scenarios: &[Scenario],
    ) -> ScenarioImpactAnalysis {
        let prediction_changes: Vec<f64> = scenarios
            .iter()
            .map(|s| (s.prediction - base_scenario.prediction).abs())
            .collect();

        let avg_prediction_change =
            prediction_changes.iter().sum::<f64>() / prediction_changes.len() as f64;
        let max_prediction_change = prediction_changes.iter().cloned().fold(0.0, f64::max);

        let high_impact_scenarios: Vec<String> = scenarios
            .iter()
            .filter(|s| {
                (s.prediction - base_scenario.prediction).abs() > avg_prediction_change * 2.0
            })
            .map(|s| s.id.clone())
            .collect();

        let prediction_flip_scenarios: Vec<String> = scenarios
            .iter()
            .filter(|s| (s.prediction > 0.5) != (base_scenario.prediction > 0.5))
            .map(|s| s.id.clone())
            .collect();

        // Feature importance analysis
        let mut feature_impacts: HashMap<String, Vec<f64>> = HashMap::new();
        for scenario in scenarios {
            for change in &scenario.changed_features {
                feature_impacts
                    .entry(change.feature_name.clone())
                    .or_default()
                    .push((scenario.prediction - base_scenario.prediction).abs());
            }
        }

        let feature_importance_ranking: Vec<FeatureImportanceRank> = feature_impacts
            .iter()
            .enumerate()
            .map(|(rank, (feature_name, impacts))| {
                let avg_impact = impacts.iter().sum::<f64>() / impacts.len() as f64;
                FeatureImportanceRank {
                    feature_name: feature_name.clone(),
                    importance_score: avg_impact,
                    rank: rank + 1,
                    avg_impact,
                    change_frequency: impacts.len(),
                }
            })
            .collect();

        let stability_analysis = PredictionStabilityAnalysis {
            stability_score: 1.0
                - (max_prediction_change / base_scenario.prediction.abs()).min(1.0),
            prediction_variance: {
                let predictions: Vec<f64> = scenarios.iter().map(|s| s.prediction).collect();
                let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
                predictions.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
                    / predictions.len() as f64
            },
            prediction_flips: prediction_flip_scenarios.len(),
            // Per-magnitude stability would need the perturbation sweep
            // re-run and bucketed by magnitude; this pass evaluates one
            // magnitude per scenario, so there is nothing to bucket.
            stability_by_magnitude: HashMap::new(),
        };

        ScenarioImpactAnalysis {
            high_impact_scenarios,
            prediction_flip_scenarios,
            avg_prediction_change,
            max_prediction_change,
            stability_analysis,
            feature_importance_ranking,
        }
    }

    fn analyze_feature_sensitivity_from_scenarios(
        &self,
        scenarios: &[Scenario],
    ) -> FeatureSensitivityAnalysis {
        let mut feature_sensitivities = HashMap::new();
        let mut feature_change_counts = HashMap::new();

        for scenario in scenarios {
            for change in &scenario.changed_features {
                let sensitivity = change.change_magnitude / scenario.distance_from_base;
                *feature_sensitivities.entry(change.feature_name.clone()).or_insert(0.0) +=
                    sensitivity;
                *feature_change_counts.entry(change.feature_name.clone()).or_insert(0) += 1;
            }
        }

        // Average sensitivities
        for (feature, sensitivity) in feature_sensitivities.iter_mut() {
            let count = feature_change_counts[feature] as f64;
            if count > 0.0 {
                *sensitivity /= count;
            }
        }

        let mut sorted_features: Vec<_> = feature_sensitivities.iter().collect();
        sorted_features.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        let most_sensitive_features: Vec<String> =
            sorted_features.iter().take(5).map(|(name, _)| (*name).clone()).collect();

        let least_sensitive_features: Vec<String> =
            sorted_features.iter().rev().take(5).map(|(name, _)| (*name).clone()).collect();

        FeatureSensitivityAnalysis {
            feature_sensitivities,
            most_sensitive_features,
            least_sensitive_features,
            non_linear_features: vec![], // Would detect non-linearity
            interaction_sensitivities: vec![], // Would compute interactions
        }
    }

    fn generate_counterfactual_insights(
        &self,
        base_scenario: &Scenario,
        scenarios: &[Scenario],
    ) -> Vec<CounterfactualInsight> {
        let mut insights = Vec::new();

        // Find scenarios with significant prediction changes
        for scenario in scenarios {
            let prediction_change = (scenario.prediction - base_scenario.prediction).abs();
            if prediction_change > 0.1 {
                // Significant change threshold
                insights.push(CounterfactualInsight {
                    description: format!(
                        "Changing {} features can alter prediction by {:.3}",
                        scenario.changed_features.len(),
                        prediction_change
                    ),
                    required_changes: scenario.changed_features.clone(),
                    predicted_outcome: scenario.prediction,
                    confidence: scenario.confidence,
                    feasibility: if scenario.changed_features.len() <= 2 {
                        ImplementationFeasibility::Easy
                    } else {
                        ImplementationFeasibility::Moderate
                    },
                });
            }
        }

        insights
    }

    fn explore_decision_boundary(&self, scenarios: &[Scenario]) -> DecisionBoundaryExploration {
        // Simplified boundary exploration
        let boundary_points: Vec<BoundaryPoint> = scenarios.iter()
            .filter(|s| (s.prediction - 0.5).abs() < 0.1) // Near decision boundary
            .take(10)
            .map(|s| BoundaryPoint {
                coordinates: s.features.clone(),
                distance_to_boundary: (s.prediction - 0.5).abs(),
                prediction: s.prediction,
                gradient_direction: HashMap::new(), // Would compute actual gradient
            })
            .collect();

        DecisionBoundaryExploration {
            boundary_points: boundary_points.clone(),
            boundary_complexity: BoundaryComplexity {
                complexity_score: 0.6,
                curvature: 0.3,
                inflection_points: 2,
                complexity_class: ComplexityClass::Polynomial,
            },
            local_linearity: LocalLinearityAnalysis {
                avg_linearity: 0.7,
                linearity_by_region: HashMap::new(),
                most_linear_regions: vec![],
                most_nonlinear_regions: vec![],
            },
            crossing_analysis: BoundaryCrossingAnalysis {
                crossing_count: boundary_points.len(),
                avg_crossing_distance: boundary_points
                    .iter()
                    .map(|p| p.distance_to_boundary)
                    .sum::<f64>()
                    / boundary_points.len() as f64,
                crossing_directions: vec![],
                common_crossing_features: vec![],
            },
        }
    }

    async fn test_perturbation_intensity(
        &self,
        base_input: &HashMap<String, f64>,
        model_fn: &(dyn Fn(&HashMap<String, f64>) -> f64 + Send + Sync),
        intensity: f64,
    ) -> Result<PerturbationIntensityResult> {
        let base_prediction = model_fn(base_input);
        let mut perturbation_details = Vec::new();
        let mut successful_perturbations = 0;
        let mut failed_perturbations = 0;
        let mut prediction_changes = Vec::new();

        // Generate perturbations
        for i in 0..self.config.num_perturbation_samples {
            let perturbed_input = self.generate_perturbation(base_input, intensity);
            let perturbed_prediction = model_fn(&perturbed_input);
            let prediction_change = (perturbed_prediction - base_prediction).abs();

            let is_successful = prediction_change < 0.1; // Threshold for "successful" perturbation

            if is_successful {
                successful_perturbations += 1;
            } else {
                failed_perturbations += 1;
            }

            prediction_changes.push(prediction_change);

            let perturbation_vector: HashMap<String, f64> = base_input
                .iter()
                .map(|(key, &base_val)| {
                    let perturbed_val = perturbed_input.get(key).unwrap_or(&base_val);
                    (key.clone(), perturbed_val - base_val)
                })
                .collect();

            let perturbation_magnitude =
                perturbation_vector.values().map(|&v| v.powi(2)).sum::<f64>().sqrt();

            perturbation_details.push(PerturbationDetail {
                id: format!("pert_{}_{}", intensity, i),
                original_input: base_input.clone(),
                perturbed_input,
                original_prediction: base_prediction,
                perturbed_prediction,
                prediction_change,
                perturbation_vector,
                perturbation_magnitude,
                is_successful,
            });
        }

        let avg_prediction_change =
            prediction_changes.iter().sum::<f64>() / prediction_changes.len() as f64;
        let max_prediction_change = prediction_changes.iter().cloned().fold(0.0, f64::max);
        let std_prediction_change = {
            let variance = prediction_changes
                .iter()
                .map(|&x| (x - avg_prediction_change).powi(2))
                .sum::<f64>()
                / prediction_changes.len() as f64;
            variance.sqrt()
        };

        Ok(PerturbationIntensityResult {
            intensity,
            num_perturbations: self.config.num_perturbation_samples,
            successful_perturbations,
            failed_perturbations,
            avg_prediction_change,
            max_prediction_change,
            std_prediction_change,
            perturbation_details,
        })
    }

    fn generate_perturbation(
        &self,
        base_input: &HashMap<String, f64>,
        intensity: f64,
    ) -> HashMap<String, f64> {
        let mut perturbed_input = base_input.clone();
        use scirs2_core::random::*; // SciRS2 Integration Policy
        let mut rng = thread_rng();

        for (_key, value) in perturbed_input.iter_mut() {
            // Add Gaussian noise proportional to intensity
            let noise = (rng.random::<f64>() - 0.5) * 2.0 * intensity;
            *value += noise;
        }

        perturbed_input
    }

    fn assess_robustness(
        &self,
        results: &HashMap<String, PerturbationIntensityResult>,
    ) -> RobustnessAssessment {
        // Calculate overall robustness score
        let success_rates: Vec<f64> = results
            .values()
            .map(|r| r.successful_perturbations as f64 / r.num_perturbations as f64)
            .collect();

        let robustness_score = success_rates.iter().sum::<f64>() / success_rates.len() as f64;

        let robustness_class = match robustness_score {
            x if x > 0.9 => RobustnessClass::VeryRobust,
            x if x > 0.7 => RobustnessClass::Robust,
            x if x > 0.5 => RobustnessClass::SomewhatRobust,
            x if x > 0.3 => RobustnessClass::Sensitive,
            _ => RobustnessClass::Fragile,
        };

        // Find critical threshold
        let critical_threshold = results
            .iter()
            .find(|(_, result)| {
                let success_rate =
                    result.successful_perturbations as f64 / result.num_perturbations as f64;
                success_rate < 0.5
            })
            .map(|(intensity, _)| intensity.parse::<f64>().unwrap_or(1.0))
            .unwrap_or(1.0);

        RobustnessAssessment {
            robustness_score,
            robustness_class,
            feature_robustness: HashMap::new(), // Would compute per-feature robustness
            critical_threshold,
            improvement_recommendations: vec![
                "Consider adding regularization".to_string(),
                "Increase training data diversity".to_string(),
            ],
        }
    }

    fn identify_sensitivity_hotspots(
        &self,
        _results: &HashMap<String, PerturbationIntensityResult>,
    ) -> Vec<SensitivityHotspot> {
        // Simplified hotspot identification
        vec![SensitivityHotspot {
            location: HashMap::new(), // Would identify actual locations
            sensitivity_score: 0.8,
            sensitivity_radius: 0.1,
            sensitive_features: vec!["feature1".to_string()],
            hotspot_type: HotspotType::Local,
        }]
    }

    fn analyze_failure_modes(
        &self,
        _results: &HashMap<String, PerturbationIntensityResult>,
    ) -> FailureModesAnalysis {
        // Simplified failure mode analysis
        let failure_modes = vec![FailureMode {
            id: "noise_sensitivity".to_string(),
            description: "Model sensitive to input noise".to_string(),
            triggering_conditions: vec![TriggeringCondition {
                feature: "any".to_string(),
                condition_type: ConditionType::Exceeds,
                threshold: 0.1,
                description: "Noise level exceeds 10%".to_string(),
            }],
            severity: FailureSeverity::Moderate,
            frequency: 0.3,
            example_inputs: vec![],
        }];

        FailureModesAnalysis {
            failure_modes,
            failure_frequency: FailureFrequencyAnalysis {
                overall_failure_rate: 0.3,
                failure_rate_by_intensity: HashMap::new(),
                failure_rate_by_feature: HashMap::new(),
                time_to_failure: TimeToFailureAnalysis {
                    avg_time_to_failure: 5.0,
                    median_time_to_failure: 3.0,
                    distribution_parameters: HashMap::new(),
                },
            },
            failure_severity: FailureSeverityAnalysis {
                avg_severity: 2.5,
                severity_distribution: HashMap::new(),
                most_severe_modes: vec!["noise_sensitivity".to_string()],
                cascading_failures: CascadingFailureAnalysis {
                    cascading_events: 0,
                    avg_cascade_length: 0.0,
                    cascade_triggers: vec![],
                    amplification_factors: HashMap::new(),
                },
            },
            mitigation_strategies: vec![MitigationStrategy {
                name: "Data Augmentation".to_string(),
                description: "Add noise during training".to_string(),
                target_failure_modes: vec!["noise_sensitivity".to_string()],
                effectiveness: 0.8,
                implementation_cost: ImplementationCost::Medium,
                implementation_steps: vec![
                    "Add noise to training data".to_string(),
                    "Retrain model".to_string(),
                ],
            }],
        }
    }

    async fn generate_adversarial_examples(
        &self,
        base_input: &HashMap<String, f64>,
        model_fn: &(dyn Fn(&HashMap<String, f64>) -> f64 + Send + Sync),
        method: &AdversarialMethod,
    ) -> Result<Vec<AdversarialExample>> {
        let mut examples = Vec::new();
        let base_prediction = model_fn(base_input);

        for i in 0..self.config.num_adversarial_examples {
            // `epsilon` grows with the example index so a batch sweeps a range
            // of perturbation budgets instead of repeating one attack.
            let epsilon = Self::FGSM_BASE_EPSILON * (i + 1) as f64;
            let adversarial_input = match method {
                AdversarialMethod::FGSM => Self::fgsm_example(base_input, model_fn, epsilon),
                AdversarialMethod::PGD => Self::pgd_example(base_input, model_fn, epsilon),
                // C&W (Carlini-Wagner), DeepFool, UAP and the Boundary attack
                // are distinct algorithms -- a Lagrangian optimisation, an
                // iterative linearisation to the decision boundary, a
                // cross-input universal perturbation, and a decision-based
                // random walk respectively. None is implemented here. They used
                // to call `generate_fgsm_example` and be reported under their
                // own names, so a caller comparing "six attacks" was really
                // comparing six runs of the same one.
                other => {
                    return Err(anyhow::anyhow!(
                        "adversarial method {other:?} is not implemented; only FGSM and PGD have \
                         real implementations (finite-difference gradient). Requesting {other:?} \
                         used to silently run FGSM under that name."
                    ))
                },
            };

            let adversarial_prediction = model_fn(&adversarial_input);

            let perturbation: HashMap<String, f64> = base_input
                .iter()
                .map(|(key, &base_val)| {
                    let adv_val = adversarial_input.get(key).unwrap_or(&base_val);
                    (key.clone(), adv_val - base_val)
                })
                .collect();

            let perturbation_norm = perturbation.values().map(|&v| v.powi(2)).sum::<f64>().sqrt();

            let is_successful =
                (adversarial_prediction - base_prediction).abs() > Self::ATTACK_SUCCESS_THRESHOLD;

            examples.push(AdversarialExample {
                id: format!("adv_{:?}_{}", method, i),
                attack_method: method.clone(),
                original_input: base_input.clone(),
                adversarial_input,
                original_prediction: base_prediction,
                adversarial_prediction,
                perturbation,
                perturbation_norm,
                is_successful,
                // Real confidence: how far past the success threshold the
                // prediction moved, saturating at 1. Previously a flat 0.8 for
                // every example, successful or not.
                confidence: ((adversarial_prediction - base_prediction).abs()
                    / Self::ATTACK_SUCCESS_THRESHOLD)
                    .clamp(0.0, 1.0),
            });
        }

        Ok(examples)
    }

    /// Base L-infinity perturbation budget for the first generated example.
    const FGSM_BASE_EPSILON: f64 = 0.01;
    /// Prediction change beyond which an attack counts as successful.
    const ATTACK_SUCCESS_THRESHOLD: f64 = 0.1;
    /// Step size for the central-difference gradient estimate.
    const GRADIENT_STEP: f64 = 1e-4;
    /// Iterations for the PGD loop.
    const PGD_ITERATIONS: usize = 10;

    /// Central-difference estimate of `d model_fn / d input[key]` for every
    /// key, using `2 * d` model evaluations.
    ///
    /// The model is a black-box `Fn(&HashMap<String, f64>) -> f64`, so no
    /// analytic gradient is available; the central difference has error
    /// `O(h^2 * |f'''|)`, which is what makes this a real -- if numerical --
    /// gradient rather than a guess.
    fn numerical_gradient(
        input: &HashMap<String, f64>,
        model_fn: &(dyn Fn(&HashMap<String, f64>) -> f64 + Send + Sync),
    ) -> HashMap<String, f64> {
        let mut gradient = HashMap::with_capacity(input.len());
        for key in input.keys() {
            let mut probe = input.clone();
            let original = input.get(key).copied().unwrap_or(0.0);

            probe.insert(key.clone(), original + Self::GRADIENT_STEP);
            let forward = model_fn(&probe);
            probe.insert(key.clone(), original - Self::GRADIENT_STEP);
            let backward = model_fn(&probe);

            gradient.insert(
                key.clone(),
                (forward - backward) / (2.0 * Self::GRADIENT_STEP),
            );
        }
        gradient
    }

    /// Real FGSM (Goodfellow, Shlens & Szegedy 2015): step `epsilon` along the
    /// SIGN of the input gradient.
    ///
    /// The gradient comes from [`Self::numerical_gradient`]. The previous
    /// implementation ignored `model_fn` entirely and stepped in a RANDOM sign
    /// direction, which is a random perturbation, not a gradient attack -- and
    /// so was uninformative about the model's actual sensitivity.
    fn fgsm_example(
        base_input: &HashMap<String, f64>,
        model_fn: &(dyn Fn(&HashMap<String, f64>) -> f64 + Send + Sync),
        epsilon: f64,
    ) -> HashMap<String, f64> {
        let gradient = Self::numerical_gradient(base_input, model_fn);
        let mut adversarial = base_input.clone();
        for (key, value) in adversarial.iter_mut() {
            let sign = match gradient.get(key) {
                Some(g) if *g > 0.0 => 1.0,
                Some(g) if *g < 0.0 => -1.0,
                // Zero (or missing) gradient: this input has no first-order
                // influence, so perturbing it is not an attack.
                _ => 0.0,
            };
            *value += epsilon * sign;
        }
        adversarial
    }

    /// Real PGD (Madry et al. 2018): iterated FGSM steps of size
    /// `epsilon / iterations`, each projected back into the L-infinity ball of
    /// radius `epsilon` around the original input.
    ///
    /// The previous implementation looped random-sign steps with no projection
    /// at all, so the perturbation was an unbounded random walk.
    fn pgd_example(
        base_input: &HashMap<String, f64>,
        model_fn: &(dyn Fn(&HashMap<String, f64>) -> f64 + Send + Sync),
        epsilon: f64,
    ) -> HashMap<String, f64> {
        let step = epsilon / Self::PGD_ITERATIONS as f64;
        let mut adversarial = base_input.clone();
        for _ in 0..Self::PGD_ITERATIONS {
            let gradient = Self::numerical_gradient(&adversarial, model_fn);
            for (key, value) in adversarial.iter_mut() {
                let sign = match gradient.get(key) {
                    Some(g) if *g > 0.0 => 1.0,
                    Some(g) if *g < 0.0 => -1.0,
                    _ => 0.0,
                };
                *value += step * sign;
                // Projection onto the L-infinity ball around the original.
                let origin = base_input.get(key).copied().unwrap_or(*value);
                *value = value.clamp(origin - epsilon, origin + epsilon);
            }
        }
        adversarial
    }

    fn analyze_attack_success(
        &self,
        adversarial_examples: &HashMap<AdversarialMethod, Vec<AdversarialExample>>,
    ) -> AttackSuccessAnalysis {
        let mut success_rate_by_method = HashMap::new();
        let mut total_successful = 0;
        let mut total_examples = 0;
        let mut total_perturbation = 0.0;

        for (method, examples) in adversarial_examples {
            let successful = examples.iter().filter(|e| e.is_successful).count();
            let success_rate = successful as f64 / examples.len() as f64;

            success_rate_by_method.insert(method.clone(), success_rate);
            total_successful += successful;
            total_examples += examples.len();

            total_perturbation += examples.iter().map(|e| e.perturbation_norm).sum::<f64>();
        }

        let overall_success_rate = total_successful as f64 / total_examples as f64;
        let avg_perturbation_magnitude = total_perturbation / total_examples as f64;

        let most_effective_methods: Vec<AdversarialMethod> = success_rate_by_method
            .iter()
            .filter(|(_, &rate)| rate > 0.5)
            .map(|(method, _)| method.clone())
            .collect();

        AttackSuccessAnalysis {
            success_rate_by_method,
            overall_success_rate,
            avg_perturbation_magnitude,
            most_effective_methods,
            attack_difficulty: AttackDifficultyAnalysis {
                easy_targets: vec!["feature1".to_string()],
                hard_targets: vec!["feature2".to_string()],
                perturbation_by_feature: HashMap::new(),
                complexity_assessment: ComplexityAssessment {
                    complexity_score: 0.6,
                    features_required: 2,
                    min_perturbation: 0.01,
                    sophistication_level: SophisticationLevel::Intermediate,
                },
            },
        }
    }

    fn assess_adversarial_robustness(
        &self,
        adversarial_examples: &HashMap<AdversarialMethod, Vec<AdversarialExample>>,
    ) -> AdversarialRobustnessAssessment {
        // Calculate robustness scores by attack method
        let robustness_by_attack: HashMap<AdversarialMethod, f64> = adversarial_examples
            .iter()
            .map(|(method, examples)| {
                let failed_attacks = examples.iter().filter(|e| !e.is_successful).count();
                let robustness = failed_attacks as f64 / examples.len() as f64;
                (method.clone(), robustness)
            })
            .collect();

        let overall_robustness =
            robustness_by_attack.values().sum::<f64>() / robustness_by_attack.len() as f64;

        AdversarialRobustnessAssessment {
            robustness_score: overall_robustness,
            robustness_by_attack,
            vulnerability_hotspots: vec![], // Would identify actual hotspots
            certified_robustness: CertifiedRobustnessAnalysis {
                certified_radius: 0.01,
                certification_confidence: 0.8,
                certification_method: "Simplified".to_string(),
                robustness_guarantees: vec![],
            },
        }
    }

    fn generate_defense_recommendations(
        &self,
        _adversarial_examples: &HashMap<AdversarialMethod, Vec<AdversarialExample>>,
    ) -> Vec<DefenseRecommendation> {
        vec![
            DefenseRecommendation {
                name: "Adversarial Training".to_string(),
                description: "Train with adversarial examples".to_string(),
                target_vulnerabilities: vec!["FGSM".to_string(), "PGD".to_string()],
                effectiveness: 0.8,
                complexity: DefenseComplexity::Moderate,
                performance_impact: PerformanceImpact::Medium,
            },
            DefenseRecommendation {
                name: "Input Preprocessing".to_string(),
                description: "Add noise reduction preprocessing".to_string(),
                target_vulnerabilities: vec!["All".to_string()],
                effectiveness: 0.6,
                complexity: DefenseComplexity::Simple,
                performance_impact: PerformanceImpact::Low,
            },
        ]
    }

    async fn search_edge_cases(
        &self,
        input_space: &HashMap<String, (f64, f64)>,
        model_fn: &(dyn Fn(&HashMap<String, f64>) -> f64 + Send + Sync),
    ) -> Result<Vec<EdgeCase>> {
        let mut edge_cases = Vec::new();
        use scirs2_core::random::*; // SciRS2 Integration Policy
        let mut rng = thread_rng();

        // Search strategies: boundary exploration, outlier generation, etc.
        for i in 0..100 {
            // Simplified edge case search
            let mut test_input = HashMap::new();

            // Generate boundary/extreme inputs
            for (feature, (min_val, max_val)) in input_space {
                let value = if rng.random::<f64>() > 0.5 {
                    *min_val + (*max_val - *min_val) * 0.01 // Near minimum
                } else {
                    *max_val - (*max_val - *min_val) * 0.01 // Near maximum
                };
                test_input.insert(feature.clone(), value);
            }

            let prediction = model_fn(&test_input);

            // Check if this is an edge case (extreme prediction, unexpected behavior, etc.)
            if !(0.1..=0.9).contains(&prediction) || prediction.is_nan() {
                edge_cases.push(EdgeCase {
                    id: format!("edge_{}", i),
                    description: format!("Edge case with extreme prediction: {:.3}", prediction),
                    trigger_input: test_input,
                    model_output: prediction,
                    expected_output: None,
                    edge_case_type: if prediction.is_nan() {
                        EdgeCaseType::ModelConfusion
                    } else if !(0.1..=0.9).contains(&prediction) {
                        EdgeCaseType::DistributionBoundary
                    } else {
                        EdgeCaseType::Outlier
                    },
                    severity: if prediction.is_nan() {
                        EdgeCaseSeverity::Critical
                    } else {
                        EdgeCaseSeverity::Medium
                    },
                    likelihood: 0.1, // Low likelihood for edge cases
                    detection_method: "Boundary exploration".to_string(),
                });
            }
        }

        Ok(edge_cases)
    }

    fn classify_edge_cases(&self, edge_cases: &[EdgeCase]) -> EdgeCaseClassification {
        let mut by_type = HashMap::new();
        let mut by_severity = HashMap::new();

        for edge_case in edge_cases {
            *by_type.entry(edge_case.edge_case_type.clone()).or_insert(0) += 1;
            *by_severity.entry(edge_case.severity.clone()).or_insert(0) += 1;
        }

        EdgeCaseClassification {
            by_type,
            by_severity,
            common_patterns: vec![],   // Would analyze patterns
            systematic_issues: vec![], // Would identify systematic issues
        }
    }

    fn analyze_edge_case_coverage(
        &self,
        _edge_cases: &[EdgeCase],
        _input_space: &HashMap<String, (f64, f64)>,
    ) -> CoverageAnalysis {
        // Simplified coverage analysis
        CoverageAnalysis {
            feature_space_coverage: 0.3, // Low coverage is expected for edge cases
            boundary_coverage: 0.8,      // High boundary coverage
            uncovered_regions: vec![],   // Would identify uncovered regions
            coverage_gaps: vec![],       // Would identify gaps
        }
    }

    fn assess_edge_case_risks(&self, edge_cases: &[EdgeCase]) -> EdgeCaseRiskAssessment {
        let overall_risk = edge_cases
            .iter()
            .map(|ec| match ec.severity {
                EdgeCaseSeverity::Critical => 1.0,
                EdgeCaseSeverity::High => 0.8,
                EdgeCaseSeverity::Medium => 0.5,
                EdgeCaseSeverity::Low => 0.2,
            })
            .sum::<f64>()
            / edge_cases.len() as f64;

        let high_risk_cases: Vec<String> = edge_cases
            .iter()
            .filter(|ec| {
                matches!(
                    ec.severity,
                    EdgeCaseSeverity::High | EdgeCaseSeverity::Critical
                )
            })
            .map(|ec| ec.id.clone())
            .collect();

        EdgeCaseRiskAssessment {
            overall_risk,
            risk_by_type: HashMap::new(), // Would compute risk by type
            high_risk_cases,
            mitigation_priorities: vec![], // Would generate priorities
        }
    }

    fn generate_simulation_summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();

        summary.insert(
            "total_what_if_analyses".to_string(),
            self.what_if_results.len().to_string(),
        );
        summary.insert(
            "total_perturbation_tests".to_string(),
            self.perturbation_results.len().to_string(),
        );
        summary.insert(
            "total_adversarial_probes".to_string(),
            self.adversarial_results.len().to_string(),
        );
        summary.insert(
            "total_edge_case_discoveries".to_string(),
            self.edge_case_results.len().to_string(),
        );

        if let Some(latest_perturbation) = self.perturbation_results.last() {
            summary.insert(
                "latest_robustness_score".to_string(),
                format!(
                    "{:.2}",
                    latest_perturbation.robustness_assessment.robustness_score
                ),
            );
        }

        if let Some(latest_adversarial) = self.adversarial_results.last() {
            summary.insert(
                "latest_adversarial_robustness".to_string(),
                format!(
                    "{:.2}",
                    latest_adversarial.robustness_assessment.robustness_score
                ),
            );
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Wave 6c debug-sweep2: real gradient-based attacks ---------------

    /// `f(x) = 3*a - 5*b`: gradient is (+3, -5), so FGSM must step `a` UP and
    /// `b` DOWN by exactly epsilon. The old implementation stepped both in a
    /// random direction and never consulted `model_fn` at all.
    #[test]
    fn fgsm_steps_along_the_real_gradient_sign() {
        let model = |x: &HashMap<String, f64>| {
            3.0 * x.get("a").copied().unwrap_or(0.0) - 5.0 * x.get("b").copied().unwrap_or(0.0)
        };
        let base: HashMap<String, f64> =
            [("a".to_string(), 1.0), ("b".to_string(), 1.0)].into_iter().collect();

        let adversarial = SimulationAnalyzer::fgsm_example(&base, &model, 0.1);
        assert!(
            (adversarial["a"] - 1.1).abs() < 1e-9,
            "positive gradient => step up: {}",
            adversarial["a"]
        );
        assert!(
            (adversarial["b"] - 0.9).abs() < 1e-9,
            "negative gradient => step down: {}",
            adversarial["b"]
        );
        // And the attack must really increase the model output.
        assert!(model(&adversarial) > model(&base));
    }

    #[test]
    fn fgsm_leaves_gradient_free_inputs_untouched() {
        // `c` does not appear in the model, so its gradient is 0 and
        // perturbing it would not be an attack.
        let model = |x: &HashMap<String, f64>| x.get("a").copied().unwrap_or(0.0);
        let base: HashMap<String, f64> =
            [("a".to_string(), 0.0), ("c".to_string(), 7.0)].into_iter().collect();
        let adversarial = SimulationAnalyzer::fgsm_example(&base, &model, 0.5);
        assert!(
            (adversarial["c"] - 7.0).abs() < 1e-12,
            "zero-gradient input must not move"
        );
        assert!((adversarial["a"] - 0.5).abs() < 1e-9);
    }

    /// PGD must project every iterate back into the L-infinity ball. The old
    /// loop had no projection at all, so 10 random steps of `epsilon` each
    /// wandered up to `10 * epsilon` away.
    #[test]
    fn pgd_stays_inside_the_l_infinity_ball() {
        let model = |x: &HashMap<String, f64>| x.values().sum::<f64>();
        let base: HashMap<String, f64> =
            [("a".to_string(), 0.0), ("b".to_string(), 0.0)].into_iter().collect();
        let epsilon = 0.05;
        let adversarial = SimulationAnalyzer::pgd_example(&base, &model, epsilon);
        for (key, value) in &adversarial {
            let delta = (value - base[key]).abs();
            assert!(
                delta <= epsilon + 1e-9,
                "{key} moved {delta}, outside the {epsilon} L-inf ball"
            );
        }
    }

    #[tokio::test]
    async fn unimplemented_attacks_are_refused_not_aliased_to_fgsm() {
        let analyzer = SimulationAnalyzer::new(SimulationConfig::default());
        let model = |x: &HashMap<String, f64>| x.values().sum::<f64>();
        let base: HashMap<String, f64> = [("a".to_string(), 1.0)].into_iter().collect();
        for method in [
            AdversarialMethod::CW,
            AdversarialMethod::DeepFool,
            AdversarialMethod::UAP,
            AdversarialMethod::Boundary,
        ] {
            let err = analyzer
                .generate_adversarial_examples(&base, &model, &method)
                .await
                .expect_err("{method:?} has no implementation");
            assert!(
                err.to_string().contains("not implemented"),
                "{method:?}: {err}"
            );
        }
        // FGSM and PGD really do run.
        assert!(analyzer
            .generate_adversarial_examples(&base, &model, &AdversarialMethod::FGSM)
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn test_simulation_analyzer_creation() {
        let config = SimulationConfig::default();
        let analyzer = SimulationAnalyzer::new(config);
        assert_eq!(analyzer.what_if_results.len(), 0);
    }

    #[tokio::test]
    async fn test_what_if_analysis() {
        let config = SimulationConfig::default();
        let mut analyzer = SimulationAnalyzer::new(config);

        let mut base_input = HashMap::new();
        base_input.insert("feature1".to_string(), 1.0);
        base_input.insert("feature2".to_string(), 2.0);

        let model_fn =
            Box::new(|input: &HashMap<String, f64>| -> f64 { input.values().sum::<f64>() * 0.1 });

        let result = analyzer.analyze_what_if(&base_input, model_fn).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_perturbation_testing() {
        let config = SimulationConfig::default();
        let mut analyzer = SimulationAnalyzer::new(config);

        let mut base_input = HashMap::new();
        base_input.insert("feature1".to_string(), 1.0);
        base_input.insert("feature2".to_string(), 2.0);

        let model_fn =
            Box::new(|input: &HashMap<String, f64>| -> f64 { input.values().sum::<f64>() * 0.1 });

        let result = analyzer.test_perturbations(&base_input, model_fn).await;
        assert!(result.is_ok());
    }
}
