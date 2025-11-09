//! Model architecture analysis and optimization.
//!
//! This module provides comprehensive model architecture analysis including
//! parameter efficiency assessment, computational complexity analysis,
//! memory efficiency evaluation, and architecture optimization recommendations.

use anyhow::Result;
use std::collections::HashMap;

use super::types::{ArchitecturalAnalysis, ModelArchitectureInfo};

/// Architecture analyzer for evaluating model design and efficiency.
#[derive(Debug)]
pub struct ArchitectureAnalyzer {
    /// Current architecture information
    architecture_info: Option<ModelArchitectureInfo>,
    /// Analysis configuration
    config: ArchitectureAnalysisConfig,
}

/// Configuration for architecture analysis.
#[derive(Debug, Clone)]
pub struct ArchitectureAnalysisConfig {
    /// Target parameter efficiency threshold
    pub target_parameter_efficiency: f64,
    /// Target memory efficiency threshold
    pub target_memory_efficiency: f64,
    /// Maximum acceptable model size in MB
    pub max_model_size_mb: f64,
    /// Preferred layer types for optimization recommendations
    pub preferred_layer_types: Vec<String>,
}

impl Default for ArchitectureAnalysisConfig {
    fn default() -> Self {
        Self {
            target_parameter_efficiency: 0.7,
            target_memory_efficiency: 0.8,
            max_model_size_mb: 1024.0, // 1GB
            preferred_layer_types: vec![
                "Attention".to_string(),
                "Linear".to_string(),
                "Normalization".to_string(),
            ],
        }
    }
}

impl ArchitectureAnalyzer {
    /// Create a new architecture analyzer.
    pub fn new() -> Self {
        Self {
            architecture_info: None,
            config: ArchitectureAnalysisConfig::default(),
        }
    }

    /// Create a new architecture analyzer with custom configuration.
    pub fn with_config(config: ArchitectureAnalysisConfig) -> Self {
        Self {
            architecture_info: None,
            config,
        }
    }

    /// Record architecture information.
    pub fn record_architecture(&mut self, arch_info: ModelArchitectureInfo) {
        self.architecture_info = Some(arch_info);
    }

    /// Get current architecture information.
    pub fn get_architecture_info(&self) -> Option<&ModelArchitectureInfo> {
        self.architecture_info.as_ref()
    }

    /// Perform comprehensive architecture analysis.
    pub fn analyze_architecture(&self) -> Result<ArchitecturalAnalysis> {
        let arch_info = self
            .architecture_info
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No architecture information available"))?;

        let parameter_efficiency = self.calculate_parameter_efficiency(arch_info);
        let computational_complexity = self.assess_computational_complexity(arch_info);
        let memory_efficiency = self.calculate_memory_efficiency(arch_info);
        let recommendations = self.generate_architecture_recommendations(arch_info);
        let bottlenecks = self.identify_architectural_bottlenecks(arch_info);

        Ok(ArchitecturalAnalysis {
            parameter_efficiency,
            computational_complexity,
            memory_efficiency,
            recommendations,
            bottlenecks,
        })
    }

    /// Calculate parameter efficiency score.
    pub fn calculate_parameter_efficiency(&self, arch_info: &ModelArchitectureInfo) -> f64 {
        if arch_info.total_parameters == 0 {
            return 0.0;
        }

        let trainable_ratio =
            arch_info.trainable_parameters as f64 / arch_info.total_parameters as f64;
        let size_penalty = if arch_info.model_size_mb > self.config.max_model_size_mb {
            0.8 // Penalize oversized models
        } else {
            1.0
        };

        // Consider layer type distribution
        let layer_efficiency = self.calculate_layer_type_efficiency(arch_info);

        (trainable_ratio * size_penalty * layer_efficiency).min(1.0)
    }

    /// Assess computational complexity of the architecture.
    pub fn assess_computational_complexity(&self, arch_info: &ModelArchitectureInfo) -> String {
        let param_count = arch_info.total_parameters;
        let depth = arch_info.depth;
        let width = arch_info.width;

        // Estimate computational complexity based on parameters and architecture
        let complexity_score = (param_count as f64).log10() + (depth as f64 * width as f64).log10();

        match complexity_score {
            x if x < 8.0 => "Low".to_string(),
            x if x < 10.0 => "Medium".to_string(),
            x if x < 12.0 => "High".to_string(),
            _ => "Very High".to_string(),
        }
    }

    /// Calculate memory efficiency score.
    pub fn calculate_memory_efficiency(&self, arch_info: &ModelArchitectureInfo) -> f64 {
        if arch_info.model_size_mb == 0.0 {
            return 0.0;
        }

        // Theoretical minimum memory based on parameters
        let theoretical_min_mb = (arch_info.total_parameters as f64 * 4.0) / (1024.0 * 1024.0); // 4 bytes per float32
        let efficiency = theoretical_min_mb / arch_info.model_size_mb;

        // Factor in layer organization efficiency
        let layer_organization_bonus = self.calculate_layer_organization_efficiency(arch_info);

        (efficiency * layer_organization_bonus).min(1.0)
    }

    /// Generate architecture optimization recommendations.
    pub fn generate_architecture_recommendations(
        &self,
        arch_info: &ModelArchitectureInfo,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Parameter efficiency recommendations
        let param_efficiency = self.calculate_parameter_efficiency(arch_info);
        if param_efficiency < self.config.target_parameter_efficiency {
            recommendations.push(
                "Consider reducing model size or improving parameter utilization".to_string(),
            );
            recommendations.push("Evaluate layer pruning opportunities".to_string());
        }

        // Memory efficiency recommendations
        let memory_efficiency = self.calculate_memory_efficiency(arch_info);
        if memory_efficiency < self.config.target_memory_efficiency {
            recommendations.push("Consider weight quantization to reduce memory usage".to_string());
            recommendations.push("Evaluate model compression techniques".to_string());
        }

        // Model size recommendations
        if arch_info.model_size_mb > self.config.max_model_size_mb {
            recommendations.push("Model size exceeds recommended limits".to_string());
            recommendations.push("Consider architectural changes to reduce model size".to_string());
        }

        // Layer type recommendations
        let layer_recommendations = self.analyze_layer_type_distribution(arch_info);
        recommendations.extend(layer_recommendations);

        // Depth and width recommendations
        if arch_info.depth > 50 {
            recommendations
                .push("Very deep model detected - consider residual connections".to_string());
        }

        if arch_info.width > 4096 {
            recommendations
                .push("Very wide model detected - consider factorization techniques".to_string());
        }

        recommendations
    }

    /// Identify architectural bottlenecks.
    pub fn identify_architectural_bottlenecks(
        &self,
        arch_info: &ModelArchitectureInfo,
    ) -> Vec<String> {
        let mut bottlenecks = Vec::new();

        // Check for imbalanced layer distribution
        if let Some(dominant_layer) = self.find_dominant_layer_type(arch_info) {
            if arch_info.layer_types.get(&dominant_layer).unwrap_or(&0)
                > &(arch_info.layer_count / 2)
            {
                bottlenecks.push(format!("Over-reliance on {} layers", dominant_layer));
            }
        }

        // Check for activation function bottlenecks
        if let Some(dominant_activation) = self.find_dominant_activation(arch_info) {
            if arch_info.activation_functions.get(&dominant_activation).unwrap_or(&0)
                > &(arch_info.layer_count * 3 / 4)
            {
                bottlenecks.push(format!(
                    "Limited activation function diversity: {} dominates",
                    dominant_activation
                ));
            }
        }

        // Check for depth/width imbalance
        let aspect_ratio = arch_info.depth as f64 / arch_info.width as f64;
        if aspect_ratio > 0.1 {
            bottlenecks.push("Model may be too deep relative to width".to_string());
        } else if aspect_ratio < 0.001 {
            bottlenecks.push("Model may be too wide relative to depth".to_string());
        }

        // Check for parameter distribution
        let params_per_layer = arch_info.total_parameters as f64 / arch_info.layer_count as f64;
        if params_per_layer > 1_000_000.0 {
            bottlenecks.push("High parameter density per layer detected".to_string());
        }

        bottlenecks
    }

    /// Calculate efficiency based on layer types.
    fn calculate_layer_type_efficiency(&self, arch_info: &ModelArchitectureInfo) -> f64 {
        let total_layers = arch_info.layer_count as f64;
        if total_layers == 0.0 {
            return 0.0;
        }

        let mut efficiency_score = 0.0;
        for (layer_type, count) in &arch_info.layer_types {
            let weight =
                if self.config.preferred_layer_types.contains(layer_type) { 1.0 } else { 0.8 };
            efficiency_score += (*count as f64 / total_layers) * weight;
        }

        efficiency_score.min(1.0)
    }

    /// Calculate layer organization efficiency.
    fn calculate_layer_organization_efficiency(&self, arch_info: &ModelArchitectureInfo) -> f64 {
        // Bonus for good layer type diversity
        let diversity_bonus = (arch_info.layer_types.len() as f64 / 10.0).min(1.2);

        // Bonus for activation function diversity
        let activation_bonus = (arch_info.activation_functions.len() as f64 / 5.0).min(1.1);

        // Penalty for extreme aspect ratios
        let aspect_ratio = arch_info.depth as f64 / arch_info.width as f64;
        let aspect_penalty = if aspect_ratio > 0.05 || aspect_ratio < 0.002 { 0.9 } else { 1.0 };

        diversity_bonus * activation_bonus * aspect_penalty
    }

    /// Analyze layer type distribution for recommendations.
    fn analyze_layer_type_distribution(&self, arch_info: &ModelArchitectureInfo) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check for missing important layer types
        if !arch_info.layer_types.contains_key("Normalization") {
            recommendations
                .push("Consider adding normalization layers for training stability".to_string());
        }

        if !arch_info.layer_types.contains_key("Dropout") {
            recommendations.push("Consider adding dropout layers for regularization".to_string());
        }

        // Check for layer type imbalances
        let total_layers = arch_info.layer_count;
        for (layer_type, count) in &arch_info.layer_types {
            let ratio = *count as f64 / total_layers as f64;
            match layer_type.as_str() {
                "Linear" if ratio > 0.8 => {
                    recommendations.push(
                        "High proportion of linear layers - consider adding non-linearity"
                            .to_string(),
                    );
                },
                "Convolutional" if ratio > 0.9 => {
                    recommendations.push(
                        "Very CNN-heavy architecture - consider hybrid approaches".to_string(),
                    );
                },
                "Attention" if ratio > 0.7 => {
                    recommendations.push(
                        "Attention-heavy architecture - consider computational efficiency"
                            .to_string(),
                    );
                },
                _ => {},
            }
        }

        recommendations
    }

    /// Find the dominant layer type.
    fn find_dominant_layer_type(&self, arch_info: &ModelArchitectureInfo) -> Option<String> {
        arch_info
            .layer_types
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(layer_type, _)| layer_type.clone())
    }

    /// Find the dominant activation function.
    fn find_dominant_activation(&self, arch_info: &ModelArchitectureInfo) -> Option<String> {
        arch_info
            .activation_functions
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(activation, _)| activation.clone())
    }

    /// Generate detailed architecture report.
    pub fn generate_architecture_report(&self) -> Result<ArchitectureReport> {
        let arch_info = self
            .architecture_info
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No architecture information available"))?;

        let analysis = self.analyze_architecture()?;

        let overall_score = self.calculate_overall_architecture_score(&analysis);

        Ok(ArchitectureReport {
            model_summary: ModelSummary {
                total_parameters: arch_info.total_parameters,
                trainable_parameters: arch_info.trainable_parameters,
                model_size_mb: arch_info.model_size_mb,
                layer_count: arch_info.layer_count,
                depth: arch_info.depth,
                width: arch_info.width,
            },
            efficiency_metrics: EfficiencyMetrics {
                parameter_efficiency: analysis.parameter_efficiency,
                memory_efficiency: analysis.memory_efficiency,
                computational_complexity: analysis.computational_complexity,
            },
            layer_distribution: arch_info.layer_types.clone(),
            activation_distribution: arch_info.activation_functions.clone(),
            recommendations: analysis.recommendations,
            bottlenecks: analysis.bottlenecks,
            overall_score,
        })
    }

    /// Calculate overall architecture score.
    fn calculate_overall_architecture_score(&self, analysis: &ArchitecturalAnalysis) -> f64 {
        let complexity_penalty = match analysis.computational_complexity.as_str() {
            "Low" => 1.0,
            "Medium" => 0.9,
            "High" => 0.8,
            "Very High" => 0.7,
            _ => 0.8,
        };

        let bottleneck_penalty = 1.0 - (analysis.bottlenecks.len() as f64 * 0.1).min(0.5);

        (analysis.parameter_efficiency * 0.4
            + analysis.memory_efficiency * 0.4
            + complexity_penalty * 0.2)
            * bottleneck_penalty
    }

    /// Clear architecture information.
    pub fn clear(&mut self) {
        self.architecture_info = None;
    }
}

impl Default for ArchitectureAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive architecture report.
#[derive(Debug, Clone)]
pub struct ArchitectureReport {
    /// Model summary statistics
    pub model_summary: ModelSummary,
    /// Efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics,
    /// Layer type distribution
    pub layer_distribution: HashMap<String, usize>,
    /// Activation function distribution
    pub activation_distribution: HashMap<String, usize>,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
    /// Identified bottlenecks
    pub bottlenecks: Vec<String>,
    /// Overall architecture score (0.0 to 1.0)
    pub overall_score: f64,
}

/// Model summary information.
#[derive(Debug, Clone)]
pub struct ModelSummary {
    /// Total number of parameters
    pub total_parameters: usize,
    /// Number of trainable parameters
    pub trainable_parameters: usize,
    /// Model size in megabytes
    pub model_size_mb: f64,
    /// Total number of layers
    pub layer_count: usize,
    /// Model depth
    pub depth: usize,
    /// Model width
    pub width: usize,
}

/// Efficiency metrics for the architecture.
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    /// Parameter efficiency score
    pub parameter_efficiency: f64,
    /// Memory efficiency score
    pub memory_efficiency: f64,
    /// Computational complexity assessment
    pub computational_complexity: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_architecture() -> ModelArchitectureInfo {
        let mut layer_types = HashMap::new();
        layer_types.insert("Linear".to_string(), 10);
        layer_types.insert("Attention".to_string(), 5);
        layer_types.insert("Normalization".to_string(), 15);

        let mut activation_functions = HashMap::new();
        activation_functions.insert("ReLU".to_string(), 10);
        activation_functions.insert("GELU".to_string(), 20);

        ModelArchitectureInfo {
            total_parameters: 1_000_000,
            trainable_parameters: 950_000,
            model_size_mb: 50.0,
            layer_count: 30,
            layer_types,
            depth: 12,
            width: 768,
            activation_functions,
        }
    }

    #[test]
    fn test_architecture_analyzer_creation() {
        let analyzer = ArchitectureAnalyzer::new();
        assert!(analyzer.architecture_info.is_none());
    }

    #[test]
    fn test_record_architecture() {
        let mut analyzer = ArchitectureAnalyzer::new();
        let arch_info = create_test_architecture();

        analyzer.record_architecture(arch_info);
        assert!(analyzer.architecture_info.is_some());
    }

    #[test]
    fn test_parameter_efficiency_calculation() {
        let analyzer = ArchitectureAnalyzer::new();
        let arch_info = create_test_architecture();

        let efficiency = analyzer.calculate_parameter_efficiency(&arch_info);
        assert!(efficiency > 0.0 && efficiency <= 1.0);
    }

    #[test]
    fn test_computational_complexity_assessment() {
        let analyzer = ArchitectureAnalyzer::new();
        let arch_info = create_test_architecture();

        let complexity = analyzer.assess_computational_complexity(&arch_info);
        assert!(["Low", "Medium", "High", "Very High"].contains(&complexity.as_str()));
    }

    #[test]
    fn test_memory_efficiency_calculation() {
        let analyzer = ArchitectureAnalyzer::new();
        let arch_info = create_test_architecture();

        let efficiency = analyzer.calculate_memory_efficiency(&arch_info);
        assert!(efficiency > 0.0 && efficiency <= 1.0);
    }

    #[test]
    fn test_architecture_analysis() {
        let mut analyzer = ArchitectureAnalyzer::new();
        let arch_info = create_test_architecture();

        analyzer.record_architecture(arch_info);
        let analysis = analyzer.analyze_architecture().unwrap();

        assert!(analysis.parameter_efficiency > 0.0);
        assert!(analysis.memory_efficiency > 0.0);
        assert!(!analysis.computational_complexity.is_empty());
    }
}
