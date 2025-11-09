//! Operator Fusion Module
//!
//! Fuses multiple operators into single optimized kernels for mobile execution.
//! Common fusion patterns include:
//! - Conv + BatchNorm + ReLU
//! - Linear + Activation
//! - Multi-head Attention components

use super::{ComputationGraph, GraphOperator, KernelType};
use crate::MobileBackend;
use std::collections::{HashMap, HashSet};
use trustformers_core::error::Result;
use trustformers_core::Tensor;

/// Fusion pattern types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    ConvBatchNorm,
    ConvBatchNormRelu,
    LinearActivation,
    MultiHeadAttention,
    LayerNormLinear,
    GeLULinear,
    ResidualBlock,
    TransformerBlock,
}

/// Fusion context for tracking fusion opportunities
#[derive(Debug)]
pub struct FusionContext {
    pub fused_operators: HashMap<usize, FusedOperator>,
    pub fusion_groups: Vec<Vec<usize>>,
    pub performance_gains: HashMap<FusionPattern, f32>,
}

impl Default for FusionContext {
    fn default() -> Self {
        let mut performance_gains = HashMap::new();
        // Estimated performance gains for each fusion pattern
        performance_gains.insert(FusionPattern::ConvBatchNorm, 1.2);
        performance_gains.insert(FusionPattern::ConvBatchNormRelu, 1.4);
        performance_gains.insert(FusionPattern::LinearActivation, 1.15);
        performance_gains.insert(FusionPattern::MultiHeadAttention, 1.5);
        performance_gains.insert(FusionPattern::LayerNormLinear, 1.25);
        performance_gains.insert(FusionPattern::GeLULinear, 1.2);
        performance_gains.insert(FusionPattern::ResidualBlock, 1.3);
        performance_gains.insert(FusionPattern::TransformerBlock, 1.6);

        Self {
            fused_operators: HashMap::new(),
            fusion_groups: Vec::new(),
            performance_gains,
        }
    }
}

/// Fused operator representation
#[derive(Debug, Clone)]
pub struct FusedOperator {
    pub pattern: FusionPattern,
    pub operators: Vec<usize>,
    pub fused_kernel: KernelType,
    pub estimated_speedup: f32,
}

/// Operator fusion engine
pub struct OperatorFusion {
    backend: MobileBackend,
    context: FusionContext,
    enable_aggressive_fusion: bool,
}

impl OperatorFusion {
    /// Create new fusion engine for specific backend
    pub fn new(backend: MobileBackend) -> Self {
        Self {
            backend,
            context: FusionContext::default(),
            enable_aggressive_fusion: true,
        }
    }

    /// Detect fusion opportunities in the graph
    pub fn detect_fusion_opportunities(
        &mut self,
        graph: &ComputationGraph,
    ) -> Result<Vec<FusionPattern>> {
        let mut patterns = Vec::new();
        let mut visited = HashSet::new();

        for (idx, operator) in graph.operators.iter().enumerate() {
            if visited.contains(&idx) {
                continue;
            }

            // Check Conv + BatchNorm patterns
            if matches!(operator.kernel, KernelType::Conv2d) {
                if let Some(pattern) = self.check_conv_patterns(graph, idx, &mut visited) {
                    patterns.push(pattern);
                }
            }

            // Check Linear + Activation patterns
            if matches!(operator.kernel, KernelType::Linear) {
                if let Some(pattern) = self.check_linear_patterns(graph, idx, &mut visited) {
                    patterns.push(pattern);
                }
            }

            // Check Attention patterns
            if matches!(operator.kernel, KernelType::Attention) {
                if let Some(pattern) = self.check_attention_patterns(graph, idx, &mut visited) {
                    patterns.push(pattern);
                }
            }

            // Check Transformer block patterns
            if self.enable_aggressive_fusion {
                if let Some(pattern) = self.check_transformer_patterns(graph, idx, &mut visited) {
                    patterns.push(pattern);
                }
            }
        }

        Ok(patterns)
    }

    /// Fuse Conv + BatchNorm operators
    pub fn fuse_conv_batchnorm(&mut self, graph: &mut ComputationGraph) -> Result<()> {
        let mut fusion_groups = Vec::new();

        // Find Conv->BatchNorm sequences
        for (idx, operator) in graph.operators.iter().enumerate() {
            if matches!(operator.kernel, KernelType::Conv2d) {
                if let Some(next_idx) = self.find_next_operator(graph, idx) {
                    if matches!(graph.operators[next_idx].kernel, KernelType::BatchNorm) {
                        // Check if we can also fuse ReLU
                        if let Some(relu_idx) = self.find_next_operator(graph, next_idx) {
                            if matches!(graph.operators[relu_idx].kernel, KernelType::Activation) {
                                fusion_groups.push(vec![idx, next_idx, relu_idx]);
                            } else {
                                fusion_groups.push(vec![idx, next_idx]);
                            }
                        } else {
                            fusion_groups.push(vec![idx, next_idx]);
                        }
                    }
                }
            }
        }

        // Apply fusion
        for group in fusion_groups {
            self.apply_fusion(graph, group, FusionPattern::ConvBatchNormRelu)?;
        }

        Ok(())
    }

    /// Fuse Linear + Activation operators
    pub fn fuse_linear_activation(&mut self, graph: &mut ComputationGraph) -> Result<()> {
        let mut fusion_groups = Vec::new();

        // Find Linear->Activation sequences
        for (idx, operator) in graph.operators.iter().enumerate() {
            if matches!(operator.kernel, KernelType::Linear) {
                if let Some(next_idx) = self.find_next_operator(graph, idx) {
                    if matches!(graph.operators[next_idx].kernel, KernelType::Activation) {
                        fusion_groups.push(vec![idx, next_idx]);
                    }
                }
            }
        }

        // Apply fusion
        for group in fusion_groups {
            self.apply_fusion(graph, group, FusionPattern::LinearActivation)?;
        }

        Ok(())
    }

    /// Fuse multi-head attention components
    pub fn fuse_attention(&mut self, graph: &mut ComputationGraph) -> Result<()> {
        let mut fusion_groups = Vec::new();

        // Find attention patterns (Q, K, V projections + attention)
        for (idx, operator) in graph.operators.iter().enumerate() {
            if matches!(operator.kernel, KernelType::Linear) {
                // Look for Q, K, V projection pattern
                if let Some(pattern_ops) = self.detect_qkv_pattern(graph, idx) {
                    fusion_groups.push(pattern_ops);
                }
            }
        }

        // Apply fusion
        for group in fusion_groups {
            self.apply_fusion(graph, group, FusionPattern::MultiHeadAttention)?;
        }

        Ok(())
    }

    // Private helper methods

    fn check_conv_patterns(
        &self,
        graph: &ComputationGraph,
        idx: usize,
        visited: &mut HashSet<usize>,
    ) -> Option<FusionPattern> {
        if let Some(next_idx) = self.find_next_operator(graph, idx) {
            if matches!(graph.operators[next_idx].kernel, KernelType::BatchNorm) {
                visited.insert(idx);
                visited.insert(next_idx);

                // Check for ReLU after BatchNorm
                if let Some(relu_idx) = self.find_next_operator(graph, next_idx) {
                    if matches!(graph.operators[relu_idx].kernel, KernelType::Activation) {
                        visited.insert(relu_idx);
                        return Some(FusionPattern::ConvBatchNormRelu);
                    }
                }

                return Some(FusionPattern::ConvBatchNorm);
            }
        }
        None
    }

    fn check_linear_patterns(
        &self,
        graph: &ComputationGraph,
        idx: usize,
        visited: &mut HashSet<usize>,
    ) -> Option<FusionPattern> {
        if let Some(next_idx) = self.find_next_operator(graph, idx) {
            match &graph.operators[next_idx].kernel {
                KernelType::Activation => {
                    visited.insert(idx);
                    visited.insert(next_idx);

                    // Check if it's specifically GeLU
                    if self.is_gelu_activation(&graph.operators[next_idx]) {
                        return Some(FusionPattern::GeLULinear);
                    }

                    return Some(FusionPattern::LinearActivation);
                },
                KernelType::Custom(name) if name == "LayerNorm" => {
                    visited.insert(idx);
                    visited.insert(next_idx);
                    return Some(FusionPattern::LayerNormLinear);
                },
                _ => {},
            }
        }
        None
    }

    fn check_attention_patterns(
        &self,
        graph: &ComputationGraph,
        idx: usize,
        visited: &mut HashSet<usize>,
    ) -> Option<FusionPattern> {
        // Simple check for attention operator
        if matches!(graph.operators[idx].kernel, KernelType::Attention) {
            visited.insert(idx);
            return Some(FusionPattern::MultiHeadAttention);
        }
        None
    }

    fn check_transformer_patterns(
        &self,
        graph: &ComputationGraph,
        idx: usize,
        visited: &mut HashSet<usize>,
    ) -> Option<FusionPattern> {
        // Look for transformer block pattern:
        // LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
        if let Some(name) = self.get_custom_kernel_name(&graph.operators[idx].kernel) {
            if name == "LayerNorm" {
                // This is a simplified check - in practice would be more sophisticated
                let block_size = 6; // Typical transformer block has ~6 operators
                if idx + block_size < graph.operators.len() {
                    // Mark all operators in the block as visited
                    for i in idx..idx + block_size {
                        visited.insert(i);
                    }
                    return Some(FusionPattern::TransformerBlock);
                }
            }
        }
        None
    }

    fn find_next_operator(&self, graph: &ComputationGraph, current_idx: usize) -> Option<usize> {
        // Find the operator that uses the output of current operator
        let current_output = &graph.operators[current_idx].outputs[0];

        for (idx, operator) in graph.operators.iter().enumerate() {
            if operator.inputs.contains(current_output) {
                return Some(idx);
            }
        }

        None
    }

    fn detect_qkv_pattern(&self, graph: &ComputationGraph, start_idx: usize) -> Option<Vec<usize>> {
        // Simplified QKV pattern detection
        // Look for three consecutive linear layers with similar shapes
        if start_idx + 2 >= graph.operators.len() {
            return None;
        }

        let mut indices = vec![start_idx];

        for i in 1..3 {
            if matches!(graph.operators[start_idx + i].kernel, KernelType::Linear) {
                indices.push(start_idx + i);
            } else {
                return None;
            }
        }

        Some(indices)
    }

    fn apply_fusion(
        &mut self,
        graph: &mut ComputationGraph,
        operator_indices: Vec<usize>,
        pattern: FusionPattern,
    ) -> Result<()> {
        // Create fused operator
        let fused_kernel = self.create_fused_kernel(&pattern, &operator_indices, graph)?;

        let fused_op = FusedOperator {
            pattern: pattern.clone(),
            operators: operator_indices.clone(),
            fused_kernel: fused_kernel.clone(),
            estimated_speedup: self.context.performance_gains[&pattern],
        };

        // Replace operators in graph with fused version
        // This is simplified - in practice would need careful graph manipulation
        let first_idx = operator_indices[0];
        graph.operators[first_idx].kernel = fused_kernel;

        // Mark other operators for removal (simplified)
        for &idx in &operator_indices[1..] {
            graph.operators[idx].kernel = KernelType::Custom("_removed_".to_string());
        }

        // Store fusion info
        self.context.fused_operators.insert(first_idx, fused_op);
        self.context.fusion_groups.push(operator_indices);

        Ok(())
    }

    fn create_fused_kernel(
        &self,
        pattern: &FusionPattern,
        indices: &[usize],
        graph: &ComputationGraph,
    ) -> Result<KernelType> {
        match pattern {
            FusionPattern::ConvBatchNorm => Ok(KernelType::Custom("ConvBN".to_string())),
            FusionPattern::ConvBatchNormRelu => Ok(KernelType::Custom("ConvBNReLU".to_string())),
            FusionPattern::LinearActivation => Ok(KernelType::Custom("LinearAct".to_string())),
            FusionPattern::GeLULinear => Ok(KernelType::Custom("LinearGeLU".to_string())),
            FusionPattern::MultiHeadAttention => Ok(KernelType::Custom("MHA".to_string())),
            FusionPattern::LayerNormLinear => Ok(KernelType::Custom("LNLinear".to_string())),
            FusionPattern::ResidualBlock => Ok(KernelType::Custom("ResBlock".to_string())),
            FusionPattern::TransformerBlock => {
                Ok(KernelType::Custom("TransformerBlock".to_string()))
            },
        }
    }

    fn is_gelu_activation(&self, operator: &GraphOperator) -> bool {
        // Check if the activation is GELU
        // This is simplified - would check operator attributes
        matches!(operator.kernel, KernelType::Activation)
    }

    fn get_custom_kernel_name<'a>(&self, kernel: &'a KernelType) -> Option<&'a str> {
        match kernel {
            KernelType::Custom(name) => Some(name),
            _ => None,
        }
    }
}

/// Conv + BatchNorm fusion implementation
pub struct ConvBatchNormFusion;

impl ConvBatchNormFusion {
    /// Fuse Conv2D weights with BatchNorm parameters
    pub fn fuse_weights(
        conv_weight: &Tensor,
        conv_bias: Option<&Tensor>,
        bn_scale: &Tensor,
        bn_bias: &Tensor,
        bn_mean: &Tensor,
        bn_var: &Tensor,
        epsilon: f32,
    ) -> Result<(Tensor, Tensor)> {
        let channels = bn_scale.shape()[0];

        // Compute fused weights
        let mut fused_weight = conv_weight.clone();
        let mut fused_bias = if let Some(bias) = conv_bias {
            bias.clone()
        } else {
            Tensor::zeros(&[channels])?
        };

        // Fuse parameters: W' = W * scale / sqrt(var + eps)
        // b' = (bias - mean) * scale / sqrt(var + eps) + bn_bias
        let scale_data = bn_scale.data()?;
        let var_data = bn_var.data()?;
        let mean_data = bn_mean.data()?;
        let bias_data = bn_bias.data()?;

        for c in 0..channels {
            let scale = scale_data[c];
            let var = var_data[c];
            let mean = mean_data[c];
            let bn_b = bias_data[c];

            let factor = scale / (var + epsilon).sqrt();

            // Update weights for this channel
            // This is simplified - would need proper tensor indexing
            let mut weight_data = fused_weight.data()?;
            for i in 0..weight_data.len() / channels {
                weight_data[c + i * channels] *= factor;
            }
            fused_weight = Tensor::from_vec(weight_data, &fused_weight.shape())?;

            // Update bias
            let mut bias_data = fused_bias.data()?;
            bias_data[c] = (bias_data[c] - mean) * factor + bn_b;
            fused_bias = Tensor::from_vec(bias_data, &fused_bias.shape())?;
        }

        Ok((fused_weight, fused_bias))
    }
}

/// Linear + Activation fusion implementation
pub struct LinearActivationFusion;

impl LinearActivationFusion {
    /// Create fused linear + activation kernel
    pub fn create_fused_kernel(activation_type: &str) -> KernelType {
        match activation_type {
            "relu" => KernelType::Custom("LinearReLU".to_string()),
            "gelu" => KernelType::Custom("LinearGeLU".to_string()),
            "silu" => KernelType::Custom("LinearSiLU".to_string()),
            _ => KernelType::Custom(format!("Linear{}", activation_type)),
        }
    }
}

/// Attention fusion implementation
pub struct AttentionFusion;

impl AttentionFusion {
    /// Fuse QKV projections into single operation
    pub fn fuse_qkv_projections(
        q_weight: &Tensor,
        k_weight: &Tensor,
        v_weight: &Tensor,
        q_bias: Option<&Tensor>,
        k_bias: Option<&Tensor>,
        v_bias: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Concatenate weights along output dimension
        let weights = vec![q_weight.clone(), k_weight.clone(), v_weight.clone()];
        let fused_weight = Tensor::concat(&weights, 0)?;

        // Concatenate biases if present
        let fused_bias = if let (Some(qb), Some(kb), Some(vb)) = (q_bias, k_bias, v_bias) {
            let biases = vec![qb.clone(), kb.clone(), vb.clone()];
            Some(Tensor::concat(&biases, 0)?)
        } else {
            None
        };

        Ok((fused_weight, fused_bias))
    }
}

/// Fusion statistics
#[derive(Debug)]
pub struct FusionStats {
    pub patterns_detected: HashMap<FusionPattern, usize>,
    pub operators_fused: usize,
    pub estimated_speedup: f32,
    pub memory_saved_bytes: usize,
}

impl Default for FusionStats {
    fn default() -> Self {
        Self {
            patterns_detected: HashMap::new(),
            operators_fused: 0,
            estimated_speedup: 1.0, // Start at baseline (no speedup)
            memory_saved_bytes: 0,
        }
    }
}

impl FusionStats {
    /// Update statistics with new fusion
    pub fn record_fusion(&mut self, pattern: &FusionPattern, operators_count: usize) {
        *self.patterns_detected.entry(pattern.clone()).or_insert(0) += 1;
        self.operators_fused += operators_count;

        // Estimate speedup (simplified)
        let pattern_speedup = match pattern {
            FusionPattern::ConvBatchNormRelu => 1.4,
            FusionPattern::MultiHeadAttention => 1.5,
            FusionPattern::TransformerBlock => 1.6,
            _ => 1.2,
        };

        // Multiplicative speedup accumulation (more realistic for independent optimizations)
        self.estimated_speedup *= pattern_speedup;

        // Estimate memory saved by eliminating intermediate tensors
        self.memory_saved_bytes += operators_count * 1024 * 1024; // 1MB per op (estimate)
    }

    /// Get summary of fusion optimizations
    pub fn summary(&self) -> String {
        format!(
            "Fusion Statistics:\n\
             - Patterns detected: {:?}\n\
             - Operators fused: {}\n\
             - Estimated speedup: {:.1}x\n\
             - Memory saved: {} MB",
            self.patterns_detected,
            self.operators_fused,
            self.estimated_speedup,
            self.memory_saved_bytes / (1024 * 1024)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_pattern_detection() {
        let fusion = OperatorFusion::new(MobileBackend::CPU);

        // Create simple graph
        let graph = ComputationGraph {
            operators: vec![
                GraphOperator {
                    id: 0,
                    kernel: KernelType::Conv2d,
                    inputs: vec!["input".to_string()],
                    outputs: vec!["conv_out".to_string()],
                    input_shapes: vec![vec![1, 3, 224, 224]],
                    output_shape: vec![1, 64, 112, 112],
                    cache_hints: None,
                },
                GraphOperator {
                    id: 1,
                    kernel: KernelType::BatchNorm,
                    inputs: vec!["conv_out".to_string()],
                    outputs: vec!["bn_out".to_string()],
                    input_shapes: vec![vec![1, 64, 112, 112]],
                    output_shape: vec![1, 64, 112, 112],
                    cache_hints: None,
                },
            ],
            edges: vec![],
        };

        let mut fusion_engine = OperatorFusion::new(MobileBackend::CPU);
        let patterns = fusion_engine.detect_fusion_opportunities(&graph).unwrap();

        assert!(!patterns.is_empty());
        assert!(patterns.contains(&FusionPattern::ConvBatchNorm));
    }

    #[test]
    fn test_conv_bn_weight_fusion() {
        let conv_weight = Tensor::ones(&[64, 3, 3, 3]).unwrap();
        let conv_bias = Some(Tensor::zeros(&[64]).unwrap());
        let bn_scale = Tensor::ones(&[64]).unwrap();
        let bn_bias = Tensor::zeros(&[64]).unwrap();
        let bn_mean = Tensor::zeros(&[64]).unwrap();
        let bn_var = Tensor::ones(&[64]).unwrap();

        let (fused_weight, fused_bias) = ConvBatchNormFusion::fuse_weights(
            &conv_weight,
            conv_bias.as_ref(),
            &bn_scale,
            &bn_bias,
            &bn_mean,
            &bn_var,
            1e-5,
        )
        .unwrap();

        assert_eq!(fused_weight.shape(), conv_weight.shape());
        assert_eq!(fused_bias.shape(), &[64]);
    }

    #[test]
    fn test_fusion_stats() {
        let mut stats = FusionStats::default();

        stats.record_fusion(&FusionPattern::ConvBatchNormRelu, 3);
        stats.record_fusion(&FusionPattern::LinearActivation, 2);

        assert_eq!(stats.operators_fused, 5);
        assert!(stats.estimated_speedup > 1.0);
        assert!(stats.memory_saved_bytes > 0);

        let summary = stats.summary();
        assert!(summary.contains("Operators fused: 5"));
    }
}
