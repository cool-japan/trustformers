// Activation statistics monitoring for detecting dead neurons and activation patterns
use crate::tensor::Tensor;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Activation statistics analyzer for monitoring neural network activations
#[derive(Debug, Clone)]
pub struct ActivationStatsAnalyzer {
    config: ActivationStatsConfig,
    layer_stats: HashMap<String, LayerActivationStats>,
    history: Vec<ActivationSnapshot>,
    current_step: u64,
}

/// Configuration for activation statistics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationStatsConfig {
    /// Track activation distributions
    pub track_distributions: bool,
    /// Track dead neuron statistics
    pub track_dead_neurons: bool,
    /// Track activation sparsity
    pub track_sparsity: bool,
    /// Threshold for considering a neuron dead (activation below this value)
    pub dead_neuron_threshold: f32,
    /// Threshold for considering an activation saturated
    pub saturation_threshold: f32,
    /// Number of histogram bins for activation distributions
    pub histogram_bins: usize,
    /// Maximum history to keep
    pub max_history: usize,
    /// Sample size for statistics (0 = use all)
    pub sample_size: usize,
}

impl Default for ActivationStatsConfig {
    fn default() -> Self {
        Self {
            track_distributions: true,
            track_dead_neurons: true,
            track_sparsity: true,
            dead_neuron_threshold: 1e-6,
            saturation_threshold: 0.95,
            histogram_bins: 50,
            max_history: 500,
            sample_size: 10000, // Sample 10k activations for efficiency
        }
    }
}

/// Activation statistics for a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerActivationStats {
    pub layer_name: String,
    pub activation_type: String,
    pub basic_stats: ActivationBasicStats,
    pub distribution_stats: Option<ActivationDistributionStats>,
    pub sparsity_stats: Option<ActivationSparsityStats>,
    pub dead_neuron_stats: Option<DeadNeuronStats>,
    pub health_assessment: ActivationHealthAssessment,
    pub update_count: u64,
}

/// Basic activation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationBasicStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub median: f32,
    pub range: f32,
    pub variance: f32,
    pub total_elements: usize,
    pub zero_count: usize,
    pub negative_count: usize,
    pub positive_count: usize,
}

/// Activation distribution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationDistributionStats {
    pub histogram: Vec<u32>,
    pub bin_edges: Vec<f32>,
    pub skewness: f32,
    pub kurtosis: f32,
    pub percentiles: Vec<f32>, // P5, P10, P25, P50, P75, P90, P95
    pub entropy: f32,
    pub effective_rank: f32,
}

/// Activation sparsity statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationSparsityStats {
    pub sparsity_ratio: f32,           // Fraction of near-zero activations
    pub density_ratio: f32,            // 1 - sparsity_ratio
    pub gini_coefficient: f32,         // Measure of inequality in activation distribution
    pub activation_concentration: f32, // How concentrated activations are
    pub hoyer_sparsity: f32,           // Hoyer's sparsity measure
}

/// Dead neuron detection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadNeuronStats {
    pub dead_neuron_count: usize,
    pub total_neurons: usize,
    pub dead_ratio: f32,
    pub saturated_neuron_count: usize,
    pub saturated_ratio: f32,
    pub effective_neurons: usize,
    pub neuron_utilization: f32,
}

/// Health assessment for layer activations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationHealthAssessment {
    pub overall_health_score: f32, // 0.0 (poor) to 1.0 (excellent)
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
    pub is_healthy: bool,
    pub risk_level: String, // Low, Medium, High, Critical
}

/// Snapshot of activation statistics at a specific step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationSnapshot {
    pub step: u64,
    pub timestamp: f64,
    pub layer_stats: HashMap<String, LayerActivationStats>,
    pub global_stats: GlobalActivationStats,
}

/// Global activation statistics across all layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalActivationStats {
    pub total_layers: usize,
    pub total_neurons: usize,
    pub total_dead_neurons: usize,
    pub global_dead_ratio: f32,
    pub avg_sparsity: f32,
    pub avg_activation_magnitude: f32,
    pub layers_with_issues: usize,
    pub most_problematic_layers: Vec<String>,
}

impl Default for ActivationStatsAnalyzer {
    fn default() -> Self {
        Self::new(ActivationStatsConfig::default())
    }
}

impl ActivationStatsAnalyzer {
    /// Create a new activation statistics analyzer
    pub fn new(config: ActivationStatsConfig) -> Self {
        Self {
            config,
            layer_stats: HashMap::new(),
            history: Vec::new(),
            current_step: 0,
        }
    }

    /// Analyze activations for a specific layer
    pub fn analyze_layer_activations(
        &mut self,
        layer_name: &str,
        activations: &Tensor,
        activation_type: &str,
    ) -> Result<()> {
        let activation_data = activations.data()?;

        // Sample data if needed for efficiency
        let sampled_data =
            if self.config.sample_size > 0 && activation_data.len() > self.config.sample_size {
                self.sample_activations(&activation_data)
            } else {
                activation_data.to_vec()
            };

        // Calculate basic statistics
        let basic_stats = self.calculate_basic_stats(&sampled_data);

        // Calculate distribution statistics if enabled
        let distribution_stats = if self.config.track_distributions {
            Some(self.calculate_distribution_stats(&sampled_data)?)
        } else {
            None
        };

        // Calculate sparsity statistics if enabled
        let sparsity_stats = if self.config.track_sparsity {
            Some(self.calculate_sparsity_stats(&sampled_data))
        } else {
            None
        };

        // Calculate dead neuron statistics if enabled
        let dead_neuron_stats = if self.config.track_dead_neurons {
            Some(self.calculate_dead_neuron_stats(&sampled_data, &activations.shape()))
        } else {
            None
        };

        // Assess activation health
        let health_assessment =
            self.assess_activation_health(&basic_stats, &sparsity_stats, &dead_neuron_stats);

        // Update layer statistics
        let layer_stats = LayerActivationStats {
            layer_name: layer_name.to_string(),
            activation_type: activation_type.to_string(),
            basic_stats,
            distribution_stats,
            sparsity_stats,
            dead_neuron_stats,
            health_assessment,
            update_count: self.layer_stats.get(layer_name).map(|s| s.update_count + 1).unwrap_or(1),
        };

        self.layer_stats.insert(layer_name.to_string(), layer_stats);

        Ok(())
    }

    /// Sample activations for efficiency
    fn sample_activations(&self, data: &[f32]) -> Vec<f32> {
        let step = data.len() / self.config.sample_size;
        if step <= 1 {
            return data.to_vec();
        }

        data.iter().step_by(step).take(self.config.sample_size).copied().collect()
    }

    /// Calculate basic activation statistics
    fn calculate_basic_stats(&self, activations: &[f32]) -> ActivationBasicStats {
        if activations.is_empty() {
            return ActivationBasicStats {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
                range: 0.0,
                variance: 0.0,
                total_elements: 0,
                zero_count: 0,
                negative_count: 0,
                positive_count: 0,
            };
        }

        let n = activations.len() as f32;

        // Basic statistics
        let sum: f32 = activations.iter().sum();
        let mean = sum / n;

        let variance: f32 = activations.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();

        let min = activations.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = activations.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max - min;

        // Calculate median
        let mut sorted_activations = activations.to_vec();
        sorted_activations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted_activations.len() % 2 == 0 {
            let mid = sorted_activations.len() / 2;
            (sorted_activations[mid - 1] + sorted_activations[mid]) / 2.0
        } else {
            sorted_activations[sorted_activations.len() / 2]
        };

        // Count different types of activations
        let zero_count = activations
            .iter()
            .filter(|&&x| x.abs() < self.config.dead_neuron_threshold)
            .count();
        let negative_count = activations.iter().filter(|&&x| x < 0.0).count();
        let positive_count = activations.iter().filter(|&&x| x > 0.0).count();

        ActivationBasicStats {
            mean,
            std,
            min,
            max,
            median,
            range,
            variance,
            total_elements: activations.len(),
            zero_count,
            negative_count,
            positive_count,
        }
    }

    /// Calculate activation distribution statistics
    fn calculate_distribution_stats(
        &self,
        activations: &[f32],
    ) -> Result<ActivationDistributionStats> {
        if activations.is_empty() {
            return Ok(ActivationDistributionStats {
                histogram: vec![0; self.config.histogram_bins],
                bin_edges: vec![0.0; self.config.histogram_bins + 1],
                skewness: 0.0,
                kurtosis: 0.0,
                percentiles: vec![0.0; 7],
                entropy: 0.0,
                effective_rank: 0.0,
            });
        }

        // Sort activations for percentile calculation
        let mut sorted_activations = activations.to_vec();
        sorted_activations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate percentiles (P5, P10, P25, P50, P75, P90, P95)
        let percentiles = vec![
            Self::percentile(&sorted_activations, 5.0),
            Self::percentile(&sorted_activations, 10.0),
            Self::percentile(&sorted_activations, 25.0),
            Self::percentile(&sorted_activations, 50.0),
            Self::percentile(&sorted_activations, 75.0),
            Self::percentile(&sorted_activations, 90.0),
            Self::percentile(&sorted_activations, 95.0),
        ];

        // Create histogram
        let min_val = sorted_activations[0];
        let max_val = sorted_activations[sorted_activations.len() - 1];
        let range = max_val - min_val;

        let mut histogram = vec![0u32; self.config.histogram_bins];
        let mut bin_edges = Vec::with_capacity(self.config.histogram_bins + 1);

        // Calculate bin edges
        for i in 0..=self.config.histogram_bins {
            let edge = min_val + (i as f32 * range / self.config.histogram_bins as f32);
            bin_edges.push(edge);
        }

        // Fill histogram
        for &value in activations {
            if range > 0.0 {
                let bin_idx = ((value - min_val) / range * self.config.histogram_bins as f32)
                    .floor() as usize;
                let bin_idx = bin_idx.min(self.config.histogram_bins - 1);
                histogram[bin_idx] += 1;
            } else {
                histogram[0] = activations.len() as u32;
                break;
            }
        }

        // Calculate skewness and kurtosis
        let mean = activations.iter().sum::<f32>() / activations.len() as f32;
        let variance =
            activations.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / activations.len() as f32;
        let std_dev = variance.sqrt();

        let skewness = if std_dev > 0.0 {
            activations.iter().map(|&x| ((x - mean) / std_dev).powi(3)).sum::<f32>()
                / activations.len() as f32
        } else {
            0.0
        };

        let kurtosis = if std_dev > 0.0 {
            activations.iter().map(|&x| ((x - mean) / std_dev).powi(4)).sum::<f32>()
                / activations.len() as f32
                - 3.0 // Excess kurtosis
        } else {
            0.0
        };

        // Calculate entropy
        let entropy = self.calculate_entropy(&histogram);

        // Calculate effective rank (approximate)
        let effective_rank = self.calculate_effective_rank(activations);

        Ok(ActivationDistributionStats {
            histogram,
            bin_edges,
            skewness,
            kurtosis,
            percentiles,
            entropy,
            effective_rank,
        })
    }

    /// Calculate sparsity statistics
    fn calculate_sparsity_stats(&self, activations: &[f32]) -> ActivationSparsityStats {
        if activations.is_empty() {
            return ActivationSparsityStats {
                sparsity_ratio: 0.0,
                density_ratio: 0.0,
                gini_coefficient: 0.0,
                activation_concentration: 0.0,
                hoyer_sparsity: 0.0,
            };
        }

        let n = activations.len() as f32;

        // Calculate sparsity ratio (fraction of near-zero activations)
        let near_zero_count = activations
            .iter()
            .filter(|&&x| x.abs() < self.config.dead_neuron_threshold)
            .count() as f32;
        let sparsity_ratio = near_zero_count / n;
        let density_ratio = 1.0 - sparsity_ratio;

        // Calculate Gini coefficient
        let gini_coefficient = self.calculate_gini_coefficient(activations);

        // Calculate activation concentration (how much activation is concentrated in few neurons)
        let abs_activations: Vec<f32> = activations.iter().map(|&x| x.abs()).collect();
        let total_activation: f32 = abs_activations.iter().sum();
        let activation_concentration = if total_activation > 0.0 {
            let mut sorted_abs = abs_activations.clone();
            sorted_abs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            // Calculate what fraction of total activation is in top 20% of neurons
            let top_20_percent = (n * 0.2) as usize;
            let top_20_activation: f32 = sorted_abs.iter().take(top_20_percent).sum();
            top_20_activation / total_activation
        } else {
            0.0
        };

        // Calculate Hoyer's sparsity measure
        let hoyer_sparsity = self.calculate_hoyer_sparsity(activations);

        ActivationSparsityStats {
            sparsity_ratio,
            density_ratio,
            gini_coefficient,
            activation_concentration,
            hoyer_sparsity,
        }
    }

    /// Calculate dead neuron statistics
    fn calculate_dead_neuron_stats(&self, activations: &[f32], shape: &[usize]) -> DeadNeuronStats {
        if activations.is_empty() {
            return DeadNeuronStats {
                dead_neuron_count: 0,
                total_neurons: 0,
                dead_ratio: 0.0,
                saturated_neuron_count: 0,
                saturated_ratio: 0.0,
                effective_neurons: 0,
                neuron_utilization: 0.0,
            };
        }

        // For simplicity, assume the last dimension represents neurons
        let total_neurons = shape.last().copied().unwrap_or(activations.len());
        let neurons_per_batch = activations.len() / total_neurons;

        let mut dead_count = 0;
        let mut saturated_count = 0;

        // Check each neuron across the batch
        for neuron_idx in 0..total_neurons {
            let mut neuron_dead = true;
            let mut neuron_saturated = false;

            for batch_idx in 0..neurons_per_batch {
                let activation_idx = batch_idx * total_neurons + neuron_idx;
                if activation_idx < activations.len() {
                    let activation = activations[activation_idx].abs();

                    if activation >= self.config.dead_neuron_threshold {
                        neuron_dead = false;
                    }

                    if activation >= self.config.saturation_threshold {
                        neuron_saturated = true;
                    }
                }
            }

            if neuron_dead {
                dead_count += 1;
            }
            if neuron_saturated {
                saturated_count += 1;
            }
        }

        let dead_ratio = dead_count as f32 / total_neurons as f32;
        let saturated_ratio = saturated_count as f32 / total_neurons as f32;
        let effective_neurons = total_neurons - dead_count;
        let neuron_utilization = effective_neurons as f32 / total_neurons as f32;

        DeadNeuronStats {
            dead_neuron_count: dead_count,
            total_neurons,
            dead_ratio,
            saturated_neuron_count: saturated_count,
            saturated_ratio,
            effective_neurons,
            neuron_utilization,
        }
    }

    /// Assess activation health
    fn assess_activation_health(
        &self,
        basic_stats: &ActivationBasicStats,
        sparsity_stats: &Option<ActivationSparsityStats>,
        dead_neuron_stats: &Option<DeadNeuronStats>,
    ) -> ActivationHealthAssessment {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        let mut health_score = 1.0f32;

        // Check for dead neurons
        if let Some(dead_stats) = dead_neuron_stats {
            if dead_stats.dead_ratio > 0.5 {
                issues.push("High proportion of dead neurons (>50%)".to_string());
                recommendations
                    .push("Consider reducing learning rate or using LeakyReLU".to_string());
                health_score *= 0.3;
            } else if dead_stats.dead_ratio > 0.2 {
                issues.push("Moderate proportion of dead neurons (>20%)".to_string());
                recommendations.push("Monitor neuron activation patterns".to_string());
                health_score *= 0.6;
            }

            if dead_stats.saturated_ratio > 0.3 {
                issues.push("High proportion of saturated neurons (>30%)".to_string());
                recommendations
                    .push("Consider batch normalization or gradient clipping".to_string());
                health_score *= 0.5;
            }
        }

        // Check activation magnitude
        if basic_stats.mean.abs() < 1e-6 {
            issues.push("Very low activation magnitudes".to_string());
            recommendations.push("Check weight initialization and learning rate".to_string());
            health_score *= 0.4;
        }

        if basic_stats.std < 1e-6 {
            issues.push("Very low activation variance".to_string());
            recommendations
                .push("Activations are not varying much - check input diversity".to_string());
            health_score *= 0.5;
        }

        // Check sparsity
        if let Some(sparsity) = sparsity_stats {
            if sparsity.sparsity_ratio > 0.9 {
                issues.push("Extremely high sparsity (>90%)".to_string());
                recommendations.push(
                    "Most activations are near zero - check for dying ReLU problem".to_string(),
                );
                health_score *= 0.2;
            } else if sparsity.sparsity_ratio > 0.8 {
                issues.push("High sparsity (>80%)".to_string());
                recommendations.push("Consider using different activation function".to_string());
                health_score *= 0.6;
            }

            if sparsity.activation_concentration > 0.9 {
                issues.push("Activation highly concentrated in few neurons".to_string());
                recommendations.push("Consider regularization or dropout".to_string());
                health_score *= 0.7;
            }
        }

        // Determine risk level
        let risk_level = if health_score >= 0.8 {
            "Low"
        } else if health_score >= 0.6 {
            "Medium"
        } else if health_score >= 0.3 {
            "High"
        } else {
            "Critical"
        };

        let is_healthy = health_score >= 0.7 && issues.is_empty();

        ActivationHealthAssessment {
            overall_health_score: health_score,
            issues,
            recommendations,
            is_healthy,
            risk_level: risk_level.to_string(),
        }
    }

    /// Calculate percentile value
    fn percentile(sorted_values: &[f32], percentile: f32) -> f32 {
        if sorted_values.is_empty() {
            return 0.0;
        }

        let idx = (percentile / 100.0 * (sorted_values.len() - 1) as f32).round() as usize;
        let idx = idx.min(sorted_values.len() - 1);
        sorted_values[idx]
    }

    /// Calculate entropy of histogram
    fn calculate_entropy(&self, histogram: &[u32]) -> f32 {
        let total: u32 = histogram.iter().sum();
        if total == 0 {
            return 0.0;
        }

        histogram
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f32 / total as f32;
                -p * p.log2()
            })
            .sum()
    }

    /// Calculate effective rank (simplified version)
    fn calculate_effective_rank(&self, activations: &[f32]) -> f32 {
        if activations.is_empty() {
            return 0.0;
        }

        // Use a simplified effective rank based on activation diversity
        let unique_values: std::collections::HashSet<_> = activations.iter()
            .map(|&x| (x * 1000.0).round() as i32) // Discretize for uniqueness
            .collect();

        unique_values.len() as f32 / activations.len() as f32 * 100.0
    }

    /// Calculate Gini coefficient
    fn calculate_gini_coefficient(&self, values: &[f32]) -> f32 {
        if values.len() <= 1 {
            return 0.0;
        }

        let mut sorted_values = values.iter().map(|&x| x.abs()).collect::<Vec<_>>();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_values.len() as f32;
        let sum: f32 = sorted_values.iter().sum();

        if sum == 0.0 {
            return 0.0;
        }

        let mut gini_sum = 0.0;
        for (i, &value) in sorted_values.iter().enumerate() {
            gini_sum += (2.0 * (i as f32 + 1.0) - n - 1.0) * value;
        }

        gini_sum / (n * sum)
    }

    /// Calculate Hoyer's sparsity measure
    fn calculate_hoyer_sparsity(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let n = values.len() as f32;
        let l1_norm: f32 = values.iter().map(|&x| x.abs()).sum();
        let l2_norm: f32 = values.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if l2_norm == 0.0 {
            return 0.0;
        }

        (n.sqrt() - l1_norm / l2_norm) / (n.sqrt() - 1.0)
    }

    /// Create a snapshot of current activation statistics
    pub fn create_snapshot(&mut self) -> Result<()> {
        let global_stats = self.calculate_global_stats();

        let snapshot = ActivationSnapshot {
            step: self.current_step,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs_f64(),
            layer_stats: self.layer_stats.clone(),
            global_stats,
        };

        self.history.push(snapshot);

        // Maintain history size limit
        while self.history.len() > self.config.max_history {
            self.history.remove(0);
        }

        self.current_step += 1;
        Ok(())
    }

    /// Calculate global activation statistics
    fn calculate_global_stats(&self) -> GlobalActivationStats {
        let total_layers = self.layer_stats.len();

        if total_layers == 0 {
            return GlobalActivationStats {
                total_layers: 0,
                total_neurons: 0,
                total_dead_neurons: 0,
                global_dead_ratio: 0.0,
                avg_sparsity: 0.0,
                avg_activation_magnitude: 0.0,
                layers_with_issues: 0,
                most_problematic_layers: Vec::new(),
            };
        }

        let total_neurons: usize = self
            .layer_stats
            .values()
            .filter_map(|stats| stats.dead_neuron_stats.as_ref())
            .map(|stats| stats.total_neurons)
            .sum();

        let total_dead_neurons: usize = self
            .layer_stats
            .values()
            .filter_map(|stats| stats.dead_neuron_stats.as_ref())
            .map(|stats| stats.dead_neuron_count)
            .sum();

        let global_dead_ratio = if total_neurons > 0 {
            total_dead_neurons as f32 / total_neurons as f32
        } else {
            0.0
        };

        let avg_sparsity: f32 = self
            .layer_stats
            .values()
            .filter_map(|stats| stats.sparsity_stats.as_ref())
            .map(|stats| stats.sparsity_ratio)
            .sum::<f32>()
            / total_layers as f32;

        let avg_activation_magnitude: f32 =
            self.layer_stats.values().map(|stats| stats.basic_stats.mean.abs()).sum::<f32>()
                / total_layers as f32;

        let layers_with_issues = self
            .layer_stats
            .values()
            .filter(|stats| !stats.health_assessment.is_healthy)
            .count();

        let mut problematic_layers: Vec<_> = self
            .layer_stats
            .iter()
            .filter(|(_, stats)| !stats.health_assessment.is_healthy)
            .map(|(name, stats)| (name.clone(), stats.health_assessment.overall_health_score))
            .collect();

        problematic_layers
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let most_problematic_layers: Vec<String> =
            problematic_layers.into_iter().take(5).map(|(name, _)| name).collect();

        GlobalActivationStats {
            total_layers,
            total_neurons,
            total_dead_neurons,
            global_dead_ratio,
            avg_sparsity,
            avg_activation_magnitude,
            layers_with_issues,
            most_problematic_layers,
        }
    }

    /// Get activation statistics for a specific layer
    pub fn get_layer_stats(&self, layer_name: &str) -> Option<&LayerActivationStats> {
        self.layer_stats.get(layer_name)
    }

    /// Get all layer statistics
    pub fn get_all_layer_stats(&self) -> &HashMap<String, LayerActivationStats> {
        &self.layer_stats
    }

    /// Get activation history
    pub fn get_history(&self) -> &[ActivationSnapshot] {
        &self.history
    }

    /// Get the most recent snapshot
    pub fn get_latest_snapshot(&self) -> Option<&ActivationSnapshot> {
        self.history.last()
    }

    /// Reset the analyzer
    pub fn reset(&mut self) {
        self.layer_stats.clear();
        self.history.clear();
        self.current_step = 0;
    }

    /// Generate a comprehensive activation analysis report
    pub fn generate_report(&self) -> ActivationAnalysisReport {
        let global_health = self.assess_global_health();
        let problematic_layers = self.identify_problematic_layers();
        let recommendations = self.generate_global_recommendations(&global_health);

        ActivationAnalysisReport {
            global_health,
            layer_count: self.layer_stats.len(),
            problematic_layers,
            recommendations,
            latest_snapshot: self.get_latest_snapshot().cloned(),
        }
    }

    /// Assess global activation health
    fn assess_global_health(&self) -> GlobalActivationHealth {
        let global_stats = self.calculate_global_stats();

        let avg_health_score = if !self.layer_stats.is_empty() {
            self.layer_stats
                .values()
                .map(|stats| stats.health_assessment.overall_health_score)
                .sum::<f32>()
                / self.layer_stats.len() as f32
        } else {
            0.0
        };

        let health_status = if avg_health_score >= 0.8 {
            "Excellent"
        } else if avg_health_score >= 0.6 {
            "Good"
        } else if avg_health_score >= 0.4 {
            "Fair"
        } else if avg_health_score >= 0.2 {
            "Poor"
        } else {
            "Critical"
        };

        GlobalActivationHealth {
            overall_score: avg_health_score,
            health_status: health_status.to_string(),
            dead_neuron_percentage: global_stats.global_dead_ratio * 100.0,
            avg_sparsity_percentage: global_stats.avg_sparsity * 100.0,
            layers_with_issues: global_stats.layers_with_issues,
            total_layers: global_stats.total_layers,
        }
    }

    /// Identify problematic layers
    fn identify_problematic_layers(&self) -> Vec<ProblematicActivationLayer> {
        let mut problematic = Vec::new();

        for (layer_name, stats) in &self.layer_stats {
            if !stats.health_assessment.is_healthy {
                problematic.push(ProblematicActivationLayer {
                    layer_name: layer_name.clone(),
                    health_score: stats.health_assessment.overall_health_score,
                    issues: stats.health_assessment.issues.clone(),
                    dead_ratio: stats
                        .dead_neuron_stats
                        .as_ref()
                        .map(|s| s.dead_ratio)
                        .unwrap_or(0.0),
                    sparsity_ratio: stats
                        .sparsity_stats
                        .as_ref()
                        .map(|s| s.sparsity_ratio)
                        .unwrap_or(0.0),
                });
            }
        }

        // Sort by health score (worst first)
        problematic.sort_by(|a, b| {
            a.health_score.partial_cmp(&b.health_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        problematic
    }

    /// Generate global recommendations
    fn generate_global_recommendations(
        &self,
        global_health: &GlobalActivationHealth,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if global_health.dead_neuron_percentage > 50.0 {
            recommendations.push("Critical: Over 50% of neurons are dead".to_string());
            recommendations.push("- Consider using LeakyReLU or ELU instead of ReLU".to_string());
            recommendations.push("- Reduce learning rate significantly".to_string());
            recommendations.push("- Check weight initialization scheme".to_string());
        } else if global_health.dead_neuron_percentage > 20.0 {
            recommendations.push("Warning: High proportion of dead neurons".to_string());
            recommendations.push("- Monitor learning rate and consider reduction".to_string());
            recommendations.push("- Consider activation function alternatives".to_string());
        }

        if global_health.avg_sparsity_percentage > 80.0 {
            recommendations.push("Very high activation sparsity detected".to_string());
            recommendations.push("- This may indicate dying ReLU problem".to_string());
            recommendations.push("- Consider batch normalization".to_string());
        }

        if global_health.overall_score < 0.5 {
            recommendations.push("Overall poor activation health".to_string());
            recommendations.push("- Review network architecture".to_string());
            recommendations.push("- Consider different activation functions".to_string());
            recommendations.push("- Add regularization techniques".to_string());
        }

        recommendations
    }
}

/// Global activation health assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalActivationHealth {
    pub overall_score: f32,
    pub health_status: String,
    pub dead_neuron_percentage: f32,
    pub avg_sparsity_percentage: f32,
    pub layers_with_issues: usize,
    pub total_layers: usize,
}

/// Information about a problematic layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblematicActivationLayer {
    pub layer_name: String,
    pub health_score: f32,
    pub issues: Vec<String>,
    pub dead_ratio: f32,
    pub sparsity_ratio: f32,
}

/// Comprehensive activation analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationAnalysisReport {
    pub global_health: GlobalActivationHealth,
    pub layer_count: usize,
    pub problematic_layers: Vec<ProblematicActivationLayer>,
    pub recommendations: Vec<String>,
    pub latest_snapshot: Option<ActivationSnapshot>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_stats_analyzer_creation() {
        let analyzer = ActivationStatsAnalyzer::default();
        assert_eq!(analyzer.current_step, 0);
        assert!(analyzer.layer_stats.is_empty());
    }

    #[test]
    fn test_basic_stats_calculation() -> Result<()> {
        let analyzer = ActivationStatsAnalyzer::default();
        let activations = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = analyzer.calculate_basic_stats(&activations);

        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.total_elements, 5);

        Ok(())
    }

    #[test]
    fn test_dead_neuron_detection() -> Result<()> {
        let mut analyzer = ActivationStatsAnalyzer::default();

        // Create activations with some dead neurons (consistently small values across batches)
        // Shape [batch=2, neurons=3], neurons 0 and 2 are consistently dead
        let activation_data = vec![
            1e-8, 0.5, 1e-9, // Batch 0: neuron 0 dead, neuron 1 alive, neuron 2 dead
            1e-7, 0.3, 1e-8, // Batch 1: neuron 0 dead, neuron 1 alive, neuron 2 dead
        ];
        let activations = Tensor::from_vec(activation_data, &[2, 3])?;

        analyzer.analyze_layer_activations("test_layer", &activations, "relu")?;

        let stats = analyzer.get_layer_stats("test_layer").unwrap();
        if let Some(dead_stats) = &stats.dead_neuron_stats {
            assert!(dead_stats.dead_neuron_count > 0);
            assert!(dead_stats.dead_ratio > 0.0);
            assert_eq!(dead_stats.total_neurons, 3);
        }

        Ok(())
    }

    #[test]
    fn test_sparsity_calculation() -> Result<()> {
        let analyzer = ActivationStatsAnalyzer::default();

        // Create sparse activations (many zeros)
        let activations = vec![0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let sparsity_stats = analyzer.calculate_sparsity_stats(&activations);

        assert!(sparsity_stats.sparsity_ratio > 0.5); // More than half are zeros
        assert!(sparsity_stats.density_ratio < 0.5);
        assert!(sparsity_stats.gini_coefficient > 0.0);

        Ok(())
    }

    #[test]
    fn test_activation_health_assessment() -> Result<()> {
        let mut analyzer = ActivationStatsAnalyzer::default();

        // Create healthy activations
        let healthy_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let healthy_activations = Tensor::from_vec(healthy_data, &[2, 3])?;

        analyzer.analyze_layer_activations("healthy_layer", &healthy_activations, "relu")?;

        let stats = analyzer.get_layer_stats("healthy_layer").unwrap();
        assert!(stats.health_assessment.overall_health_score > 0.5);

        Ok(())
    }

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(ActivationStatsAnalyzer::percentile(&values, 50.0), 3.0);
        assert_eq!(ActivationStatsAnalyzer::percentile(&values, 0.0), 1.0);
        assert_eq!(ActivationStatsAnalyzer::percentile(&values, 100.0), 5.0);
    }

    #[test]
    fn test_gini_coefficient() {
        let analyzer = ActivationStatsAnalyzer::default();

        // Perfect equality
        let equal_values = vec![1.0, 1.0, 1.0, 1.0];
        assert!((analyzer.calculate_gini_coefficient(&equal_values)).abs() < 1e-6);

        // Perfect inequality
        let unequal_values = vec![0.0, 0.0, 0.0, 1.0];
        assert!(analyzer.calculate_gini_coefficient(&unequal_values) > 0.5);
    }

    #[test]
    fn test_activation_analysis_report() -> Result<()> {
        let mut analyzer = ActivationStatsAnalyzer::default();

        // Add some layers with different activation patterns
        let normal_data = vec![0.1, 0.2, 0.3, 0.4];
        let sparse_data = vec![0.0, 0.0, 1.0, 0.0];

        let normal_activations = Tensor::from_vec(normal_data, &[2, 2])?;
        let sparse_activations = Tensor::from_vec(sparse_data, &[2, 2])?;

        analyzer.analyze_layer_activations("normal_layer", &normal_activations, "relu")?;
        analyzer.analyze_layer_activations("sparse_layer", &sparse_activations, "relu")?;

        let report = analyzer.generate_report();
        assert_eq!(report.layer_count, 2);
        assert!(!report.recommendations.is_empty());

        Ok(())
    }
}
