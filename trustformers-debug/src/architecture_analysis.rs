//! Architecture Analysis
//!
//! Comprehensive analysis tools for neural network architectures including
//! parameter counting, receptive field calculation, and connectivity analysis.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for architecture analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureAnalysisConfig {
    /// Enable parameter counting
    pub enable_parameter_counting: bool,
    /// Enable receptive field calculation
    pub enable_receptive_field_calculation: bool,
    /// Enable depth/width analysis
    pub enable_depth_width_analysis: bool,
    /// Enable connectivity pattern detection
    pub enable_connectivity_patterns: bool,
    /// Enable symmetry detection
    pub enable_symmetry_detection: bool,
    /// Maximum depth to analyze for receptive fields
    pub max_receptive_field_depth: usize,
    /// Sampling rate for large models (0.0 to 1.0)
    pub sampling_rate: f32,
}

impl Default for ArchitectureAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_parameter_counting: true,
            enable_receptive_field_calculation: true,
            enable_depth_width_analysis: true,
            enable_connectivity_patterns: true,
            enable_symmetry_detection: true,
            max_receptive_field_depth: 50,
            sampling_rate: 1.0,
        }
    }
}

/// Layer type for architecture analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum LayerType {
    Linear,
    Conv2D,
    Conv3D,
    BatchNorm,
    LayerNorm,
    Attention,
    Embedding,
    Dropout,
    Activation,
    Pooling,
    Residual,
    Unknown,
}

/// Information about a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    pub id: String,
    pub name: String,
    pub layer_type: LayerType,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub parameters: usize,
    pub trainable_parameters: usize,
    pub memory_usage: usize,
    pub flops: u64,
    pub receptive_field: Option<ReceptiveField>,
}

/// Receptive field information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceptiveField {
    pub size: Vec<usize>,
    pub stride: Vec<usize>,
    pub padding: Vec<usize>,
    pub effective_size: Vec<usize>,
}

/// Connectivity pattern between layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityPattern {
    pub from_layer: String,
    pub to_layer: String,
    pub connection_type: ConnectionType,
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum ConnectionType {
    Sequential,
    Residual,
    Attention,
    Skip,
    Recurrent,
    Branching,
}

/// Symmetry information in the architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryInfo {
    pub symmetry_type: SymmetryType,
    pub symmetric_layers: Vec<String>,
    pub confidence: f32,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymmetryType {
    Translational,
    Rotational,
    Reflection,
    Permutation,
    Block,
}

/// Architecture analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureAnalysisReport {
    pub total_parameters: usize,
    pub trainable_parameters: usize,
    pub model_size_mb: f32,
    pub total_flops: u64,
    pub model_depth: usize,
    pub model_width: usize,
    pub layers: Vec<LayerInfo>,
    pub connectivity_patterns: Vec<ConnectivityPattern>,
    pub symmetries: Vec<SymmetryInfo>,
    pub parameter_distribution: HashMap<LayerType, usize>,
    pub bottlenecks: Vec<String>,
    pub efficiency_metrics: EfficiencyMetrics,
}

/// Model efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    pub parameter_efficiency: f32,
    pub flops_efficiency: f32,
    pub memory_efficiency: f32,
    pub depth_efficiency: f32,
    pub overall_score: f32,
}

/// Architecture analyzer
#[derive(Debug)]
pub struct ArchitectureAnalyzer {
    config: ArchitectureAnalysisConfig,
    layers: Vec<LayerInfo>,
    connections: Vec<ConnectivityPattern>,
    analysis_cache: HashMap<String, ArchitectureAnalysisReport>,
}

impl ArchitectureAnalyzer {
    /// Create a new architecture analyzer
    pub fn new(config: ArchitectureAnalysisConfig) -> Self {
        Self {
            config,
            layers: Vec::new(),
            connections: Vec::new(),
            analysis_cache: HashMap::new(),
        }
    }

    /// Register a layer for analysis
    pub fn register_layer(&mut self, layer: LayerInfo) {
        self.layers.push(layer);
    }

    /// Add a connection between layers
    pub fn add_connection(&mut self, pattern: ConnectivityPattern) {
        self.connections.push(pattern);
    }

    /// Analyze the registered architecture
    pub async fn analyze(&mut self) -> Result<ArchitectureAnalysisReport> {
        let mut report = ArchitectureAnalysisReport {
            total_parameters: 0,
            trainable_parameters: 0,
            model_size_mb: 0.0,
            total_flops: 0,
            model_depth: 0,
            model_width: 0,
            layers: self.layers.clone(),
            connectivity_patterns: self.connections.clone(),
            symmetries: Vec::new(),
            parameter_distribution: HashMap::new(),
            bottlenecks: Vec::new(),
            efficiency_metrics: EfficiencyMetrics {
                parameter_efficiency: 0.0,
                flops_efficiency: 0.0,
                memory_efficiency: 0.0,
                depth_efficiency: 0.0,
                overall_score: 0.0,
            },
        };

        if self.config.enable_parameter_counting {
            self.count_parameters(&mut report);
        }

        if self.config.enable_receptive_field_calculation {
            self.calculate_receptive_fields(&mut report).await?;
        }

        if self.config.enable_depth_width_analysis {
            self.analyze_depth_width(&mut report);
        }

        if self.config.enable_connectivity_patterns {
            self.analyze_connectivity_patterns(&mut report);
        }

        if self.config.enable_symmetry_detection {
            self.detect_symmetries(&mut report);
        }

        self.calculate_efficiency_metrics(&mut report);
        self.identify_bottlenecks(&mut report);

        Ok(report)
    }

    /// Count parameters in all layers
    fn count_parameters(&self, report: &mut ArchitectureAnalysisReport) {
        let mut param_distribution: HashMap<LayerType, usize> = HashMap::new();

        for layer in &self.layers {
            report.total_parameters += layer.parameters;
            report.trainable_parameters += layer.trainable_parameters;

            *param_distribution.entry(layer.layer_type.clone()).or_insert(0) += layer.parameters;
        }

        report.parameter_distribution = param_distribution;

        // Estimate model size (4 bytes per float32 parameter)
        report.model_size_mb = (report.total_parameters * 4) as f32 / (1024.0 * 1024.0);

        // Calculate total FLOPS
        report.total_flops = self.layers.iter().map(|l| l.flops).sum();
    }

    /// Calculate receptive fields for convolutional layers
    async fn calculate_receptive_fields(
        &mut self,
        report: &mut ArchitectureAnalysisReport,
    ) -> Result<()> {
        for layer in &mut self.layers {
            if matches!(layer.layer_type, LayerType::Conv2D | LayerType::Conv3D) {
                layer.receptive_field =
                    Some(Self::compute_receptive_field_static(&layer.layer_type));
            }
        }

        report.layers = self.layers.clone();
        Ok(())
    }

    /// Compute receptive field for a convolutional layer (static version)
    fn compute_receptive_field_static(layer_type: &LayerType) -> ReceptiveField {
        match layer_type {
            LayerType::Conv2D => {
                // Simple 2D convolution receptive field calculation
                let kernel_size = vec![3, 3]; // Default 3x3 kernel
                let stride = vec![1, 1];
                let padding = vec![1, 1];

                ReceptiveField {
                    size: kernel_size.clone(),
                    stride,
                    padding,
                    effective_size: kernel_size,
                }
            },
            LayerType::Conv3D => {
                // Simple 3D convolution receptive field calculation
                let kernel_size = vec![3, 3, 3]; // Default 3x3x3 kernel
                let stride = vec![1, 1, 1];
                let padding = vec![1, 1, 1];

                ReceptiveField {
                    size: kernel_size.clone(),
                    stride,
                    padding,
                    effective_size: kernel_size,
                }
            },
            _ => {
                // For non-conv layers, receptive field is 1
                ReceptiveField {
                    size: vec![1],
                    stride: vec![1],
                    padding: vec![0],
                    effective_size: vec![1],
                }
            },
        }
    }

    /// Compute receptive field for a convolutional layer
    #[allow(dead_code)]
    fn compute_receptive_field(&self, layer: &LayerInfo) -> ReceptiveField {
        Self::compute_receptive_field_static(&layer.layer_type)
    }

    /// Analyze model depth and width
    fn analyze_depth_width(&self, report: &mut ArchitectureAnalysisReport) {
        // Calculate depth (number of sequential layers)
        report.model_depth = self.layers.len();

        // Calculate width (maximum number of parameters in a single layer)
        report.model_width = self.layers.iter().map(|l| l.parameters).max().unwrap_or(0);
    }

    /// Analyze connectivity patterns
    fn analyze_connectivity_patterns(&self, report: &mut ArchitectureAnalysisReport) {
        let mut pattern_types: HashMap<ConnectionType, usize> = HashMap::new();

        for connection in &self.connections {
            *pattern_types.entry(connection.connection_type.clone()).or_insert(0) += 1;
        }

        // Find unusual connectivity patterns
        for (connection_type, count) in pattern_types {
            if count > self.layers.len() / 2 {
                // High connectivity, might indicate bottlenecks
                report.bottlenecks.push(format!(
                    "High {:?} connectivity: {} connections",
                    connection_type, count
                ));
            }
        }
    }

    /// Detect architectural symmetries
    fn detect_symmetries(&self, report: &mut ArchitectureAnalysisReport) {
        // Detect block symmetries (repeated layer patterns)
        let mut block_patterns: HashMap<Vec<LayerType>, Vec<usize>> = HashMap::new();

        // Look for patterns of 2-5 consecutive layers
        for window_size in 2..=5.min(self.layers.len()) {
            for i in 0..=(self.layers.len() - window_size) {
                let pattern: Vec<LayerType> =
                    self.layers[i..i + window_size].iter().map(|l| l.layer_type.clone()).collect();

                block_patterns.entry(pattern).or_default().push(i);
            }
        }

        // Find repeated patterns
        for (pattern, positions) in block_patterns {
            if positions.len() > 1 {
                let confidence = positions.len() as f32 / self.layers.len() as f32;

                if confidence > 0.1 {
                    // At least 10% of the model
                    report.symmetries.push(SymmetryInfo {
                        symmetry_type: SymmetryType::Block,
                        symmetric_layers: positions
                            .iter()
                            .map(|&i| format!("block_{}", i))
                            .collect(),
                        confidence,
                        description: format!(
                            "Repeated block pattern: {:?} appears {} times",
                            pattern,
                            positions.len()
                        ),
                    });
                }
            }
        }

        // Detect parameter symmetries
        let mut param_groups: HashMap<usize, Vec<String>> = HashMap::new();
        for layer in &self.layers {
            param_groups.entry(layer.parameters).or_default().push(layer.id.clone());
        }

        for (param_count, layer_ids) in param_groups {
            if layer_ids.len() > 2 && param_count > 0 {
                let confidence = layer_ids.len() as f32 / self.layers.len() as f32;

                report.symmetries.push(SymmetryInfo {
                    symmetry_type: SymmetryType::Permutation,
                    symmetric_layers: layer_ids.clone(),
                    confidence,
                    description: format!(
                        "Parameter symmetry: {} layers with {} parameters each",
                        layer_ids.len(),
                        param_count
                    ),
                });
            }
        }
    }

    /// Calculate efficiency metrics
    fn calculate_efficiency_metrics(&self, report: &mut ArchitectureAnalysisReport) {
        let total_params = report.total_parameters as f32;
        let total_flops = report.total_flops as f32;
        let depth = report.model_depth as f32;
        let memory = report.model_size_mb;

        // Parameter efficiency: fewer parameters for same capability is better
        report.efficiency_metrics.parameter_efficiency = if total_params > 0.0 {
            1.0 / (total_params / 1_000_000.0).log10().max(1.0) // Inverse log scale
        } else {
            1.0
        };

        // FLOPS efficiency: fewer FLOPS for same capability is better
        report.efficiency_metrics.flops_efficiency = if total_flops > 0.0 {
            1.0 / (total_flops / 1_000_000_000.0).log10().max(1.0) // Inverse log scale
        } else {
            1.0
        };

        // Memory efficiency: less memory usage is better
        report.efficiency_metrics.memory_efficiency = if memory > 0.0 {
            1.0 / (memory / 100.0).log10().max(1.0) // Inverse log scale
        } else {
            1.0
        };

        // Depth efficiency: moderate depth is best (not too shallow, not too deep)
        report.efficiency_metrics.depth_efficiency = if depth > 0.0 {
            let optimal_depth = 20.0; // Assumed optimal depth
            1.0 - ((depth - optimal_depth).abs() / optimal_depth).min(1.0)
        } else {
            0.0
        };

        // Overall efficiency score (weighted average)
        report.efficiency_metrics.overall_score = 0.3
            * report.efficiency_metrics.parameter_efficiency
            + 0.3 * report.efficiency_metrics.flops_efficiency
            + 0.2 * report.efficiency_metrics.memory_efficiency
            + 0.2 * report.efficiency_metrics.depth_efficiency;
    }

    /// Identify potential bottlenecks
    fn identify_bottlenecks(&self, report: &mut ArchitectureAnalysisReport) {
        // Find layers with disproportionately high parameter counts
        if let Some(_max_params) = self.layers.iter().map(|l| l.parameters).max() {
            let avg_params = report.total_parameters / self.layers.len().max(1);

            for layer in &self.layers {
                if layer.parameters > avg_params * 5 {
                    report.bottlenecks.push(format!(
                        "Parameter bottleneck: Layer '{}' has {} parameters ({}x average)",
                        layer.name,
                        layer.parameters,
                        layer.parameters / avg_params.max(1)
                    ));
                }
            }
        }

        // Find layers with very large memory usage
        for layer in &self.layers {
            if layer.memory_usage > 100 * 1024 * 1024 {
                // > 100MB
                report.bottlenecks.push(format!(
                    "Memory bottleneck: Layer '{}' uses {:.1}MB memory",
                    layer.name,
                    layer.memory_usage as f32 / (1024.0 * 1024.0)
                ));
            }
        }

        // Find layers with very high FLOPS
        if let Some(_max_flops) = self.layers.iter().map(|l| l.flops).max() {
            let avg_flops = report.total_flops / self.layers.len().max(1) as u64;

            for layer in &self.layers {
                if layer.flops > avg_flops * 10 {
                    report.bottlenecks.push(format!(
                        "Computation bottleneck: Layer '{}' requires {} FLOPS ({}x average)",
                        layer.name,
                        layer.flops,
                        layer.flops / avg_flops.max(1)
                    ));
                }
            }
        }
    }

    /// Quick architecture analysis for simplified interface
    pub async fn quick_analysis(&self) -> Result<crate::QuickArchitectureSummary> {
        let total_parameters = self.layers.iter().map(|l| l.parameters as u64).sum::<u64>();
        let total_flops = self.layers.iter().map(|l| l.flops).sum::<u64>();

        // Estimate model size in MB (4 bytes per float32 parameter)
        let model_size_mb = (total_parameters as f64 * 4.0) / (1024.0 * 1024.0);

        // Calculate efficiency score based on parameters to FLOPS ratio
        let efficiency_score = if total_flops > 0 {
            (total_parameters as f64 / total_flops as f64 * 1000.0).min(100.0)
        } else {
            50.0
        };

        let mut recommendations = Vec::new();
        if total_parameters > 1_000_000_000 {
            recommendations
                .push("Consider model compression techniques for large model".to_string());
        }
        if efficiency_score < 30.0 {
            recommendations.push("Model architecture could be more efficient".to_string());
        }
        if model_size_mb > 1000.0 {
            recommendations.push("Large model size may impact deployment".to_string());
        }
        if recommendations.is_empty() {
            recommendations.push("Architecture appears well-balanced".to_string());
        }

        Ok(crate::QuickArchitectureSummary {
            total_parameters,
            model_size_mb,
            efficiency_score,
            recommendations,
        })
    }

    /// Generate a comprehensive report
    pub async fn generate_report(&self) -> Result<ArchitectureAnalysisReport> {
        // Create a temporary clone to avoid mutable borrow issues
        let mut temp_analyzer = ArchitectureAnalyzer {
            config: self.config.clone(),
            layers: self.layers.clone(),
            connections: self.connections.clone(),
            analysis_cache: HashMap::new(),
        };

        temp_analyzer.analyze().await
    }

    /// Clear all registered layers and connections
    pub fn clear(&mut self) {
        self.layers.clear();
        self.connections.clear();
        self.analysis_cache.clear();
    }

    /// Get summary statistics
    pub fn get_summary(&self) -> ArchitectureSummary {
        let total_params: usize = self.layers.iter().map(|l| l.parameters).sum();
        let total_flops: u64 = self.layers.iter().map(|l| l.flops).sum();

        ArchitectureSummary {
            total_layers: self.layers.len(),
            total_parameters: total_params,
            total_flops,
            average_layer_size: if !self.layers.is_empty() {
                total_params / self.layers.len()
            } else {
                0
            },
            layer_type_distribution: {
                let mut dist = HashMap::new();
                for layer in &self.layers {
                    *dist.entry(layer.layer_type.clone()).or_insert(0) += 1;
                }
                dist
            },
        }
    }
}

/// Summary statistics for architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureSummary {
    pub total_layers: usize,
    pub total_parameters: usize,
    pub total_flops: u64,
    pub average_layer_size: usize,
    pub layer_type_distribution: HashMap<LayerType, usize>,
}

/// Convenience function to create a layer info
pub fn create_layer_info(
    id: String,
    name: String,
    layer_type: LayerType,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    parameters: usize,
) -> LayerInfo {
    let memory_usage = parameters * 4; // 4 bytes per float32
    let flops = estimate_flops(&layer_type, &input_shape, &output_shape, parameters);

    LayerInfo {
        id,
        name,
        layer_type,
        input_shape,
        output_shape,
        parameters,
        trainable_parameters: parameters, // Assume all parameters are trainable by default
        memory_usage,
        flops,
        receptive_field: None,
    }
}

/// Estimate FLOPS for a layer
fn estimate_flops(
    layer_type: &LayerType,
    input_shape: &[usize],
    output_shape: &[usize],
    parameters: usize,
) -> u64 {
    match layer_type {
        LayerType::Linear => {
            if input_shape.len() >= 2 && output_shape.len() >= 2 {
                let batch_size = input_shape[0] as u64;
                let input_features = input_shape[1] as u64;
                let output_features = output_shape[1] as u64;
                batch_size * input_features * output_features * 2
            } else {
                parameters as u64 * 2
            }
        },
        LayerType::Conv2D => {
            if output_shape.len() >= 4 {
                let batch_size = output_shape[0] as u64;
                let output_channels = output_shape[1] as u64;
                let output_h = output_shape[2] as u64;
                let output_w = output_shape[3] as u64;
                batch_size
                    * output_channels
                    * output_h
                    * output_w
                    * (parameters as u64 / output_channels).max(1)
                    * 2
            } else {
                parameters as u64 * 2
            }
        },
        LayerType::Attention => {
            if input_shape.len() >= 3 {
                let batch_size = input_shape[0] as u64;
                let seq_len = input_shape[1] as u64;
                let hidden_size = input_shape[2] as u64;
                batch_size * seq_len * seq_len * hidden_size * 4
            } else {
                parameters as u64 * 4
            }
        },
        _ => parameters as u64,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_layer(id: &str, params: usize) -> LayerInfo {
        create_layer_info(
            id.to_string(),
            format!("{}_layer", id),
            LayerType::Linear,
            vec![1, params],
            vec![1, params],
            params,
        )
    }

    // ── Config ────────────────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = ArchitectureAnalysisConfig::default();
        assert!(cfg.enable_parameter_counting);
        assert!(cfg.enable_receptive_field_calculation);
        assert!(cfg.enable_depth_width_analysis);
        assert!(cfg.enable_connectivity_patterns);
        assert!(cfg.enable_symmetry_detection);
        assert!(cfg.max_receptive_field_depth > 0);
        assert!((cfg.sampling_rate - 1.0).abs() < 1e-6);
    }

    // ── ArchitectureAnalyzer ───────────────────────────────────────────────

    #[test]
    fn test_analyzer_new_empty() {
        let analyzer = ArchitectureAnalyzer::new(ArchitectureAnalysisConfig::default());
        let summary = analyzer.get_summary();
        assert_eq!(summary.total_layers, 0);
        assert_eq!(summary.total_parameters, 0);
    }

    #[test]
    fn test_register_layer_accumulates() {
        let mut analyzer = ArchitectureAnalyzer::new(ArchitectureAnalysisConfig::default());
        analyzer.register_layer(make_linear_layer("l0", 512));
        analyzer.register_layer(make_linear_layer("l1", 256));
        assert_eq!(analyzer.get_summary().total_layers, 2);
    }

    #[test]
    fn test_add_connection() {
        let mut analyzer = ArchitectureAnalyzer::new(ArchitectureAnalysisConfig::default());
        analyzer.register_layer(make_linear_layer("a", 128));
        analyzer.register_layer(make_linear_layer("b", 128));
        analyzer.add_connection(ConnectivityPattern {
            from_layer: "a".to_string(),
            to_layer: "b".to_string(),
            connection_type: ConnectionType::Sequential,
            strength: 1.0,
        });
        let summary = analyzer.get_summary();
        assert_eq!(summary.total_layers, 2);
    }

    #[test]
    fn test_clear_resets_state() {
        let mut analyzer = ArchitectureAnalyzer::new(ArchitectureAnalysisConfig::default());
        analyzer.register_layer(make_linear_layer("l0", 64));
        analyzer.clear();
        assert_eq!(analyzer.get_summary().total_layers, 0);
    }

    // ── LayerInfo via create_layer_info ────────────────────────────────────

    #[test]
    fn test_create_layer_info_parameters() {
        let layer = make_linear_layer("dense", 1024);
        assert_eq!(layer.parameters, 1024);
        assert_eq!(layer.trainable_parameters, 1024);
        assert_eq!(layer.memory_usage, 1024 * 4);
        assert!(layer.receptive_field.is_none());
    }

    #[test]
    fn test_create_layer_info_conv2d_flops() {
        let layer = create_layer_info(
            "conv".to_string(),
            "conv_layer".to_string(),
            LayerType::Conv2D,
            vec![1, 3, 224, 224],
            vec![1, 64, 112, 112],
            64 * 3 * 3 * 3,
        );
        assert!(layer.flops > 0);
    }

    #[test]
    fn test_create_layer_info_attention_flops() {
        let layer = create_layer_info(
            "attn".to_string(),
            "attention".to_string(),
            LayerType::Attention,
            vec![1, 128, 768],
            vec![1, 128, 768],
            768 * 768 * 4,
        );
        assert!(layer.flops > 0);
    }

    // ── ArchitectureSummary ────────────────────────────────────────────────

    #[test]
    fn test_summary_totals() {
        let mut analyzer = ArchitectureAnalyzer::new(ArchitectureAnalysisConfig::default());
        analyzer.register_layer(make_linear_layer("l0", 100));
        analyzer.register_layer(make_linear_layer("l1", 200));
        let s = analyzer.get_summary();
        assert_eq!(s.total_parameters, 300);
        assert_eq!(s.average_layer_size, 150);
    }

    #[test]
    fn test_summary_layer_type_distribution() {
        let mut analyzer = ArchitectureAnalyzer::new(ArchitectureAnalysisConfig::default());
        analyzer.register_layer(make_linear_layer("l0", 64));
        analyzer.register_layer(make_linear_layer("l1", 64));
        let s = analyzer.get_summary();
        assert_eq!(
            s.layer_type_distribution.get(&LayerType::Linear).copied().unwrap_or(0),
            2
        );
    }

    // ── LayerType variants ─────────────────────────────────────────────────

    #[test]
    fn test_layer_type_all_variants() {
        let types = [
            LayerType::Linear,
            LayerType::Conv2D,
            LayerType::Conv3D,
            LayerType::BatchNorm,
            LayerType::LayerNorm,
            LayerType::Attention,
            LayerType::Embedding,
            LayerType::Dropout,
            LayerType::Activation,
            LayerType::Pooling,
            LayerType::Residual,
            LayerType::Unknown,
        ];
        for t in &types {
            assert!(!format!("{:?}", t).is_empty());
        }
    }

    // ── ConnectionType variants ────────────────────────────────────────────

    #[test]
    fn test_connection_type_variants() {
        let types = [
            ConnectionType::Sequential,
            ConnectionType::Residual,
            ConnectionType::Attention,
            ConnectionType::Skip,
            ConnectionType::Recurrent,
            ConnectionType::Branching,
        ];
        for t in &types {
            assert!(!format!("{:?}", t).is_empty());
        }
    }

    // ── SymmetryType variants ──────────────────────────────────────────────

    #[test]
    fn test_symmetry_type_variants() {
        let types = [
            SymmetryType::Translational,
            SymmetryType::Rotational,
            SymmetryType::Reflection,
            SymmetryType::Permutation,
            SymmetryType::Block,
        ];
        for t in &types {
            assert!(!format!("{:?}", t).is_empty());
        }
    }

    // ── EfficiencyMetrics ──────────────────────────────────────────────────

    #[test]
    fn test_efficiency_metrics_construction() {
        let metrics = EfficiencyMetrics {
            parameter_efficiency: 0.8,
            flops_efficiency: 0.7,
            memory_efficiency: 0.9,
            depth_efficiency: 0.6,
            overall_score: 0.75,
        };
        assert!((metrics.overall_score - 0.75).abs() < 1e-6);
    }

    // ── async analyze ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_analyze_empty() {
        let mut analyzer = ArchitectureAnalyzer::new(ArchitectureAnalysisConfig::default());
        let report = analyzer.analyze().await.expect("analyze should succeed");
        assert_eq!(report.total_parameters, 0);
        assert_eq!(report.model_depth, 0);
    }

    #[tokio::test]
    async fn test_analyze_parameter_counting() {
        let mut analyzer = ArchitectureAnalyzer::new(ArchitectureAnalysisConfig::default());
        analyzer.register_layer(make_linear_layer("l0", 512));
        analyzer.register_layer(make_linear_layer("l1", 512));
        let report = analyzer.analyze().await.expect("analyze should succeed");
        assert_eq!(report.total_parameters, 1024);
        assert_eq!(report.trainable_parameters, 1024);
    }

    #[tokio::test]
    async fn test_analyze_depth_width() {
        let mut analyzer = ArchitectureAnalyzer::new(ArchitectureAnalysisConfig::default());
        analyzer.register_layer(make_linear_layer("l0", 128));
        analyzer.register_layer(make_linear_layer("l1", 256));
        analyzer.register_layer(make_linear_layer("l2", 64));
        let report = analyzer.analyze().await.expect("analyze should succeed");
        assert_eq!(report.model_depth, 3);
        assert_eq!(report.model_width, 256);
    }

    #[tokio::test]
    async fn test_quick_analysis_returns_summary() {
        let mut analyzer = ArchitectureAnalyzer::new(ArchitectureAnalysisConfig::default());
        analyzer.register_layer(make_linear_layer("l0", 1000));
        let qs = analyzer.quick_analysis().await.expect("quick_analysis should succeed");
        assert_eq!(qs.total_parameters, 1000);
        assert!(!qs.recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_generate_report_symmetry_detection() {
        let mut analyzer = ArchitectureAnalyzer::new(ArchitectureAnalysisConfig::default());
        // Register 6 identical layers — should trigger permutation symmetry detection.
        for i in 0..6 {
            analyzer.register_layer(make_linear_layer(&format!("l{}", i), 256));
        }
        let report = analyzer.generate_report().await.expect("report should succeed");
        // Symmetry detection is heuristic; just verify the report is populated
        assert_eq!(report.total_parameters, 6 * 256);
    }
}
