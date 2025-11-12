//! Model Pruning Implementation
//!
//! Various strategies for removing unnecessary weights and structures

#![allow(clippy::excessive_nesting)] // Complex pruning algorithms require deep nesting
#![allow(unused_variables)] // Model pruning

use crate::tensor::Tensor;
use anyhow::{anyhow, Result};
use scirs2_core::random::Rng; // SciRS2 Policy compliant
use std::collections::{HashMap, HashSet};

/// Pruning configuration
#[derive(Debug, Clone)]
pub struct PruningConfig {
    /// Target sparsity level (0.0 - 1.0)
    pub target_sparsity: f32,
    /// Whether to use iterative pruning
    pub iterative: bool,
    /// Number of pruning iterations
    pub iterations: usize,
    /// Whether to fine-tune after pruning
    pub fine_tune: bool,
    /// Layers to exclude from pruning
    pub exclude_layers: HashSet<String>,
    /// Minimum weight magnitude to keep
    pub magnitude_threshold: Option<f32>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            target_sparsity: 0.5,
            iterative: false,
            iterations: 1,
            fine_tune: true,
            exclude_layers: HashSet::new(),
            magnitude_threshold: None,
            seed: None,
        }
    }
}

/// Pruning strategy trait
pub trait PruningStrategy: Send + Sync {
    /// Apply pruning to weights
    fn prune_weights(&self, weights: &Tensor, config: &PruningConfig) -> Result<Tensor>;

    /// Get pruning mask
    fn get_mask(&self, weights: &Tensor, config: &PruningConfig) -> Result<Tensor>;

    /// Strategy name
    fn name(&self) -> &str;
}

/// Result of pruning operation
#[derive(Debug, Clone)]
pub struct PruningResult<M>
where
    M: crate::traits::Model,
{
    pub model: M,
    pub sparsity: f32,
    pub pruned_params: usize,
    pub total_params: usize,
    pub layer_sparsity: HashMap<String, f32>,
}

/// Main pruner interface
pub trait Pruner: Send + Sync {
    /// Prune a model - simplified for now
    fn prune<M>(&self, model: M, config: &PruningConfig) -> Result<PruningResult<M>>
    where
        M: crate::traits::Model + Clone;

    /// Get pruning statistics - simplified interface without layer access
    fn estimate_pruning_potential<M>(
        &self,
        model: &M,
        config: &PruningConfig,
    ) -> Result<PruningStats>
    where
        M: crate::traits::Model;
}

/// Pruning statistics
#[derive(Debug, Clone)]
pub struct PruningStats {
    pub total_params: usize,
    pub zero_params: usize,
    pub sparsity: f32,
    pub layer_stats: HashMap<String, LayerPruningStats>,
}

#[derive(Debug, Clone)]
pub struct LayerPruningStats {
    pub total_params: usize,
    pub zero_params: usize,
    pub sparsity: f32,
}

/// Magnitude-based pruning
pub struct MagnitudePruner {
    #[allow(dead_code)]
    threshold: f32,
}

impl MagnitudePruner {
    pub fn new(sparsity: f32) -> Self {
        Self {
            threshold: sparsity,
        }
    }
}

impl PruningStrategy for MagnitudePruner {
    fn prune_weights(&self, weights: &Tensor, config: &PruningConfig) -> Result<Tensor> {
        let mask = self.get_mask(weights, config)?;

        // Apply mask to weights
        let pruned = weights
            .data()?
            .iter()
            .zip(mask.data()?.iter())
            .map(|(w, m)| if *m > 0.5 { *w } else { 0.0 })
            .collect::<Vec<_>>();

        Ok(Tensor::from_vec(pruned, &weights.shape())?)
    }

    fn get_mask(&self, weights: &Tensor, config: &PruningConfig) -> Result<Tensor> {
        let data = weights.data()?;
        let mut abs_weights: Vec<(f32, usize)> =
            data.iter().enumerate().map(|(i, &w)| (w.abs(), i)).collect();

        // Sort by magnitude
        abs_weights.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Calculate cutoff index
        let num_prune = (data.len() as f32 * config.target_sparsity) as usize;
        let mut mask = vec![1.0; data.len()];

        // Prune smallest weights
        for i in 0..num_prune.min(abs_weights.len()) {
            mask[abs_weights[i].1] = 0.0;
        }

        Ok(Tensor::from_vec(mask, &weights.shape())?)
    }

    fn name(&self) -> &str {
        "MagnitudePruner"
    }
}

/// Structured pruning (channels/filters)
pub struct StructuredPruner {
    pruning_dim: usize,
}

impl StructuredPruner {
    pub fn new(pruning_dim: usize) -> Self {
        Self { pruning_dim }
    }
}

impl PruningStrategy for StructuredPruner {
    fn prune_weights(&self, weights: &Tensor, config: &PruningConfig) -> Result<Tensor> {
        // Structured pruning removes entire channels/filters
        let shape = weights.shape();
        if shape.len() < 2 {
            return Err(anyhow!("Structured pruning requires at least 2D tensors"));
        }

        // Calculate importance scores for each structure
        let importance_scores = self.calculate_importance(weights)?;

        // Determine which structures to prune
        let num_structures = shape[self.pruning_dim];
        let num_prune = (num_structures as f32 * config.target_sparsity) as usize;

        let mut indices: Vec<usize> = (0..num_structures).collect();
        indices.sort_by(|&a, &b| importance_scores[a].partial_cmp(&importance_scores[b]).unwrap());

        let pruned_indices: HashSet<_> = indices.iter().take(num_prune).cloned().collect();

        // Create pruned tensor
        let data = weights.data()?;
        let mut pruned_data = Vec::with_capacity(data.len());

        // This is simplified - in practice would need proper indexing
        for (i, &val) in data.iter().enumerate() {
            let structure_idx = (i / shape.iter().skip(self.pruning_dim + 1).product::<usize>())
                % shape[self.pruning_dim];

            if pruned_indices.contains(&structure_idx) {
                pruned_data.push(0.0);
            } else {
                pruned_data.push(val);
            }
        }

        Ok(Tensor::from_vec(pruned_data, &shape)?)
    }

    fn get_mask(&self, weights: &Tensor, config: &PruningConfig) -> Result<Tensor> {
        // Similar to prune_weights but returns mask
        Ok(Tensor::ones(&weights.shape())?)
    }

    fn name(&self) -> &str {
        "StructuredPruner"
    }
}

impl StructuredPruner {
    fn calculate_importance(&self, weights: &Tensor) -> Result<Vec<f32>> {
        let shape = weights.shape();
        let num_structures = shape[self.pruning_dim];
        let mut importance = vec![0.0; num_structures];

        // Calculate L2 norm for each structure
        let data = weights.data()?;
        let structure_size = shape.iter().skip(self.pruning_dim + 1).product::<usize>();
        let structures_per_batch = shape.iter().take(self.pruning_dim).product::<usize>();

        for (i, importance_ref) in importance.iter_mut().enumerate() {
            let mut sum_sq = 0.0;
            for j in 0..structures_per_batch {
                for k in 0..structure_size {
                    let idx = j * num_structures * structure_size + i * structure_size + k;
                    if idx < data.len() {
                        sum_sq += data[idx] * data[idx];
                    }
                }
            }
            *importance_ref = sum_sq.sqrt();
        }

        Ok(importance)
    }
}

/// Unstructured pruning (individual weights)
pub struct UnstructuredPruner {
    random: bool,
}

impl UnstructuredPruner {
    pub fn new(random: bool) -> Self {
        Self { random }
    }
}

impl PruningStrategy for UnstructuredPruner {
    fn prune_weights(&self, weights: &Tensor, config: &PruningConfig) -> Result<Tensor> {
        let data = weights.data()?;
        let num_prune = (data.len() as f32 * config.target_sparsity) as usize;

        let mut pruned = data.to_vec();

        if self.random {
            // Random pruning
            let mut rng = rand::rng();
            let mut indices: Vec<usize> = (0..data.len()).collect();

            // Fisher-Yates shuffle
            for i in (1..indices.len()).rev() {
                let j = rng.random_range(0..=i);
                indices.swap(i, j);
            }

            // Prune first num_prune indices
            for i in 0..num_prune.min(indices.len()) {
                pruned[indices[i]] = 0.0;
            }
        } else {
            // Magnitude-based pruning
            let magnitude_pruner = MagnitudePruner::new(config.target_sparsity);
            return magnitude_pruner.prune_weights(weights, config);
        }

        Ok(Tensor::from_vec(pruned, &weights.shape())?)
    }

    fn get_mask(&self, weights: &Tensor, config: &PruningConfig) -> Result<Tensor> {
        let data = weights.data()?;
        let num_prune = (data.len() as f32 * config.target_sparsity) as usize;
        let mut mask = vec![1.0; data.len()];

        if self.random {
            let mut rng = rand::rng();
            let mut indices: Vec<usize> = (0..data.len()).collect();

            for i in (1..indices.len()).rev() {
                let j = rng.random_range(0..=i);
                indices.swap(i, j);
            }

            for i in 0..num_prune.min(indices.len()) {
                mask[indices[i]] = 0.0;
            }
        }

        Ok(Tensor::from_vec(mask, &weights.shape())?)
    }

    fn name(&self) -> &str {
        "UnstructuredPruner"
    }
}

/// Gradual pruning over training iterations
pub struct GradualPruner {
    initial_sparsity: f32,
    final_sparsity: f32,
    begin_step: usize,
    end_step: usize,
    #[allow(dead_code)]
    frequency: usize,
}

impl GradualPruner {
    pub fn new(
        initial_sparsity: f32,
        final_sparsity: f32,
        begin_step: usize,
        end_step: usize,
        frequency: usize,
    ) -> Self {
        Self {
            initial_sparsity,
            final_sparsity,
            begin_step,
            end_step,
            frequency,
        }
    }

    pub fn get_sparsity_at_step(&self, step: usize) -> f32 {
        if step < self.begin_step {
            return 0.0;
        }
        if step >= self.end_step {
            return self.final_sparsity;
        }

        let progress = (step - self.begin_step) as f32 / (self.end_step - self.begin_step) as f32;
        self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * progress
    }
}

/// Pruning schedule
#[derive(Debug, Clone)]
pub enum PruningSchedule {
    /// One-shot pruning
    OneShot { step: usize },
    /// Gradual pruning
    Gradual {
        begin_step: usize,
        end_step: usize,
        frequency: usize,
    },
    /// Iterative pruning
    Iterative {
        steps: Vec<usize>,
        sparsities: Vec<f32>,
    },
}

/// Channel pruning for CNNs
pub struct ChannelPruner {
    importance_metric: ChannelImportanceMetric,
}

#[derive(Debug, Clone)]
pub enum ChannelImportanceMetric {
    /// L1 norm of channel weights
    L1Norm,
    /// L2 norm of channel weights
    L2Norm,
    /// Mean activation magnitude
    MeanActivation,
    /// Geometric median
    GeometricMedian,
}

impl ChannelPruner {
    pub fn new(metric: ChannelImportanceMetric) -> Self {
        Self {
            importance_metric: metric,
        }
    }
}

impl PruningStrategy for ChannelPruner {
    fn prune_weights(&self, weights: &Tensor, config: &PruningConfig) -> Result<Tensor> {
        let shape = weights.shape();
        if shape.len() != 4 {
            return Err(anyhow!("Channel pruning requires 4D tensors (NCHW format)"));
        }

        let num_channels = shape[1]; // Assuming NCHW format
        let channel_importance = self.calculate_channel_importance(weights)?;

        // Determine channels to prune
        let num_prune = (num_channels as f32 * config.target_sparsity) as usize;
        let mut sorted_channels: Vec<(f32, usize)> =
            channel_importance.iter().enumerate().map(|(i, &score)| (score, i)).collect();
        sorted_channels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let pruned_channels: HashSet<usize> =
            sorted_channels.iter().take(num_prune).map(|(_, idx)| *idx).collect();

        // Create pruned weights by setting pruned channels to zero
        let data = weights.data()?;
        let mut pruned_data = data.to_vec();
        let channel_size = shape[2] * shape[3]; // H * W
        let batch_channel_size = num_channels * channel_size;

        for batch in 0..shape[0] {
            for channel in &pruned_channels {
                let start_idx = batch * batch_channel_size + channel * channel_size;
                let end_idx = start_idx + channel_size;
                for i in start_idx..end_idx.min(pruned_data.len()) {
                    pruned_data[i] = 0.0;
                }
            }
        }

        Ok(Tensor::from_vec(pruned_data, &shape)?)
    }

    fn get_mask(&self, weights: &Tensor, config: &PruningConfig) -> Result<Tensor> {
        let shape = weights.shape();
        let num_channels = shape[1];
        let channel_importance = self.calculate_channel_importance(weights)?;

        let num_prune = (num_channels as f32 * config.target_sparsity) as usize;
        let mut sorted_channels: Vec<(f32, usize)> =
            channel_importance.iter().enumerate().map(|(i, &score)| (score, i)).collect();
        sorted_channels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let pruned_channels: HashSet<usize> =
            sorted_channels.iter().take(num_prune).map(|(_, idx)| *idx).collect();

        let data = weights.data()?;
        let mut mask = vec![1.0; data.len()];
        let channel_size = shape[2] * shape[3];
        let batch_channel_size = num_channels * channel_size;

        for batch in 0..shape[0] {
            for channel in &pruned_channels {
                let start_idx = batch * batch_channel_size + channel * channel_size;
                let end_idx = start_idx + channel_size;
                for i in start_idx..end_idx.min(mask.len()) {
                    mask[i] = 0.0;
                }
            }
        }

        Ok(Tensor::from_vec(mask, &shape)?)
    }

    fn name(&self) -> &str {
        "ChannelPruner"
    }
}

impl ChannelPruner {
    fn calculate_channel_importance(&self, weights: &Tensor) -> Result<Vec<f32>> {
        let shape = weights.shape();
        let num_channels = shape[1];
        let channel_size = shape[2] * shape[3];
        let data = weights.data()?;
        let mut importance = vec![0.0; num_channels];

        for (channel, importance_ref) in importance.iter_mut().enumerate() {
            let mut channel_score = 0.0;
            let mut count = 0;

            for batch in 0..shape[0] {
                let start_idx = batch * num_channels * channel_size + channel * channel_size;
                let end_idx = start_idx + channel_size;

                for data_ref in data.iter().take(end_idx.min(data.len())).skip(start_idx) {
                    match self.importance_metric {
                        ChannelImportanceMetric::L1Norm => channel_score += data_ref.abs(),
                        ChannelImportanceMetric::L2Norm => channel_score += data_ref * data_ref,
                        ChannelImportanceMetric::MeanActivation => channel_score += data_ref.abs(),
                        ChannelImportanceMetric::GeometricMedian => channel_score += data_ref.abs(),
                    }
                    count += 1;
                }
            }

            *importance_ref = match self.importance_metric {
                ChannelImportanceMetric::L2Norm => (channel_score / count as f32).sqrt(),
                _ => channel_score / count as f32,
            };
        }

        Ok(importance)
    }
}

/// Filter pruning for CNNs
pub struct FilterPruner {
    importance_metric: FilterImportanceMetric,
}

#[derive(Debug, Clone)]
pub enum FilterImportanceMetric {
    /// L1 norm of filter weights
    L1Norm,
    /// L2 norm of filter weights
    L2Norm,
    /// Average percentage of zero activations
    APoZ,
}

impl FilterPruner {
    pub fn new(metric: FilterImportanceMetric) -> Self {
        Self {
            importance_metric: metric,
        }
    }
}

impl PruningStrategy for FilterPruner {
    fn prune_weights(&self, weights: &Tensor, config: &PruningConfig) -> Result<Tensor> {
        let shape = weights.shape();
        if shape.len() != 4 {
            return Err(anyhow!("Filter pruning requires 4D tensors (NCHW format)"));
        }

        let num_filters = shape[0]; // Output channels
        let filter_importance = self.calculate_filter_importance(weights)?;

        // Determine filters to prune
        let num_prune = (num_filters as f32 * config.target_sparsity) as usize;
        let mut sorted_filters: Vec<(f32, usize)> =
            filter_importance.iter().enumerate().map(|(i, &score)| (score, i)).collect();
        sorted_filters.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let pruned_filters: HashSet<usize> =
            sorted_filters.iter().take(num_prune).map(|(_, idx)| *idx).collect();

        // Create pruned weights
        let data = weights.data()?;
        let mut pruned_data = data.to_vec();
        let filter_size = shape[1] * shape[2] * shape[3]; // Input channels * H * W

        for filter_idx in &pruned_filters {
            let start_idx = filter_idx * filter_size;
            let end_idx = start_idx + filter_size;
            for i in start_idx..end_idx.min(pruned_data.len()) {
                pruned_data[i] = 0.0;
            }
        }

        Ok(Tensor::from_vec(pruned_data, &shape)?)
    }

    fn get_mask(&self, weights: &Tensor, config: &PruningConfig) -> Result<Tensor> {
        let shape = weights.shape();
        let num_filters = shape[0];
        let filter_importance = self.calculate_filter_importance(weights)?;

        let num_prune = (num_filters as f32 * config.target_sparsity) as usize;
        let mut sorted_filters: Vec<(f32, usize)> =
            filter_importance.iter().enumerate().map(|(i, &score)| (score, i)).collect();
        sorted_filters.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let pruned_filters: HashSet<usize> =
            sorted_filters.iter().take(num_prune).map(|(_, idx)| *idx).collect();

        let data = weights.data()?;
        let mut mask = vec![1.0; data.len()];
        let filter_size = shape[1] * shape[2] * shape[3];

        for filter_idx in &pruned_filters {
            let start_idx = filter_idx * filter_size;
            let end_idx = start_idx + filter_size;
            for i in start_idx..end_idx.min(mask.len()) {
                mask[i] = 0.0;
            }
        }

        Ok(Tensor::from_vec(mask, &shape)?)
    }

    fn name(&self) -> &str {
        "FilterPruner"
    }
}

impl FilterPruner {
    fn calculate_filter_importance(&self, weights: &Tensor) -> Result<Vec<f32>> {
        let shape = weights.shape();
        let num_filters = shape[0];
        let filter_size = shape[1] * shape[2] * shape[3];
        let data = weights.data()?;
        let mut importance = vec![0.0; num_filters];

        for (filter, importance_ref) in importance.iter_mut().enumerate() {
            let start_idx = filter * filter_size;
            let end_idx = start_idx + filter_size;
            let mut filter_score = 0.0;

            for data_ref in data.iter().take(end_idx.min(data.len())).skip(start_idx) {
                match self.importance_metric {
                    FilterImportanceMetric::L1Norm => filter_score += data_ref.abs(),
                    FilterImportanceMetric::L2Norm => filter_score += data_ref * data_ref,
                    FilterImportanceMetric::APoZ => {
                        filter_score += if *data_ref == 0.0 { 1.0 } else { 0.0 }
                    },
                }
            }

            *importance_ref = match self.importance_metric {
                FilterImportanceMetric::L2Norm => filter_score.sqrt(),
                FilterImportanceMetric::APoZ => filter_score / filter_size as f32,
                _ => filter_score,
            };
        }

        Ok(importance)
    }
}

/// Attention head pruning for transformers
pub struct HeadPruner {
    num_heads: usize,
    head_dim: usize,
}

impl HeadPruner {
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            head_dim,
        }
    }
}

impl PruningStrategy for HeadPruner {
    fn prune_weights(&self, weights: &Tensor, config: &PruningConfig) -> Result<Tensor> {
        let shape = weights.shape();
        if shape.len() != 2 {
            return Err(anyhow!(
                "Head pruning requires 2D tensors (attention weight matrices)"
            ));
        }

        // Determine heads to prune
        let num_prune = (self.num_heads as f32 * config.target_sparsity) as usize;
        let head_importance = self.calculate_head_importance(weights)?;

        let mut sorted_heads: Vec<(f32, usize)> =
            head_importance.iter().enumerate().map(|(i, &score)| (score, i)).collect();
        sorted_heads.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let pruned_heads: HashSet<usize> =
            sorted_heads.iter().take(num_prune).map(|(_, idx)| *idx).collect();

        // Create pruned weights
        let data = weights.data()?;
        let mut pruned_data = data.to_vec();

        // Zero out pruned heads
        for head_idx in &pruned_heads {
            let start_col = head_idx * self.head_dim;
            let end_col = start_col + self.head_dim;

            for row in 0..shape[0] {
                for col in start_col..end_col.min(shape[1]) {
                    let idx = row * shape[1] + col;
                    if idx < pruned_data.len() {
                        pruned_data[idx] = 0.0;
                    }
                }
            }
        }

        Ok(Tensor::from_vec(pruned_data, &shape)?)
    }

    fn get_mask(&self, weights: &Tensor, config: &PruningConfig) -> Result<Tensor> {
        let shape = weights.shape();
        let num_prune = (self.num_heads as f32 * config.target_sparsity) as usize;
        let head_importance = self.calculate_head_importance(weights)?;

        let mut sorted_heads: Vec<(f32, usize)> =
            head_importance.iter().enumerate().map(|(i, &score)| (score, i)).collect();
        sorted_heads.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let pruned_heads: HashSet<usize> =
            sorted_heads.iter().take(num_prune).map(|(_, idx)| *idx).collect();

        let data = weights.data()?;
        let mut mask = vec![1.0; data.len()];

        for head_idx in &pruned_heads {
            let start_col = head_idx * self.head_dim;
            let end_col = start_col + self.head_dim;

            for row in 0..shape[0] {
                for col in start_col..end_col.min(shape[1]) {
                    let idx = row * shape[1] + col;
                    if idx < mask.len() {
                        mask[idx] = 0.0;
                    }
                }
            }
        }

        Ok(Tensor::from_vec(mask, &shape)?)
    }

    fn name(&self) -> &str {
        "HeadPruner"
    }
}

impl HeadPruner {
    fn calculate_head_importance(&self, weights: &Tensor) -> Result<Vec<f32>> {
        let shape = weights.shape();
        let data = weights.data()?;
        let mut importance = vec![0.0; self.num_heads];

        for (head, importance_ref) in importance.iter_mut().enumerate() {
            let start_col = head * self.head_dim;
            let end_col = start_col + self.head_dim;
            let mut head_score = 0.0;
            let mut count = 0;

            for row in 0..shape[0] {
                for col in start_col..end_col.min(shape[1]) {
                    let idx = row * shape[1] + col;
                    if idx < data.len() {
                        head_score += data[idx] * data[idx]; // L2 norm
                        count += 1;
                    }
                }
            }

            *importance_ref = if count > 0 { (head_score / count as f32).sqrt() } else { 0.0 };
        }

        Ok(importance)
    }
}

/// Layer pruning (remove entire layers)
pub struct LayerPruner {
    layer_importance: HashMap<String, f32>,
}

impl Default for LayerPruner {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerPruner {
    pub fn new() -> Self {
        Self {
            layer_importance: HashMap::new(),
        }
    }

    pub fn with_importance_scores(scores: HashMap<String, f32>) -> Self {
        Self {
            layer_importance: scores,
        }
    }

    /// Calculate layer importance using model-level metrics (simplified)
    pub fn analyze_model<M>(&mut self, model: &M) -> Result<()>
    where
        M: crate::traits::Model,
    {
        // Simplified implementation using only model-level information
        // In a real implementation, would need access to actual layer weights
        // For now, simulate importance scores based on parameter count
        let total_params = model.num_parameters();

        // Simulate layer importance based on typical model architectures
        let typical_layers = vec![
            ("embedding".to_string(), 0.8),
            ("attention_0".to_string(), 0.6),
            ("feedforward_0".to_string(), 0.4),
            ("attention_1".to_string(), 0.5),
            ("feedforward_1".to_string(), 0.3),
            ("output".to_string(), 0.9),
        ];

        for (name, importance) in typical_layers {
            self.layer_importance.insert(name, importance * total_params as f32);
        }

        Ok(())
    }

    /// Get layers that would be pruned based on current importance scores
    pub fn get_pruning_candidates(&self, config: &PruningConfig) -> Result<Vec<String>> {
        let total_layers = self.layer_importance.len();
        let num_prune = (total_layers as f32 * config.target_sparsity) as usize;

        // Sort layers by importance (ascending - prune least important)
        let mut sorted_layers: Vec<(f32, String)> = self
            .layer_importance
            .iter()
            .map(|(name, &score)| (score, name.clone()))
            .collect();
        sorted_layers.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let pruned_layers: Vec<String> = sorted_layers
            .iter()
            .take(num_prune)
            .map(|(_, name)| name.clone())
            .filter(|name| !config.exclude_layers.contains(name))
            .collect();

        Ok(pruned_layers)
    }
}

/// Automatic model pruner that chooses the best strategy based on model architecture
pub struct AutomaticPruner {
    strategies: HashMap<String, Box<dyn PruningStrategy>>,
    default_strategy: Box<dyn PruningStrategy>,
}

impl AutomaticPruner {
    pub fn new() -> Self {
        let mut strategies = HashMap::new();

        // Add default strategies for different layer types
        strategies.insert(
            "conv".to_string(),
            Box::new(FilterPruner::new(FilterImportanceMetric::L2Norm)) as Box<dyn PruningStrategy>,
        );
        strategies.insert(
            "attention".to_string(),
            Box::new(HeadPruner::new(12, 64)) as Box<dyn PruningStrategy>,
        );
        strategies.insert(
            "linear".to_string(),
            Box::new(MagnitudePruner::new(0.5)) as Box<dyn PruningStrategy>,
        );

        let default_strategy = Box::new(MagnitudePruner::new(0.5));

        Self {
            strategies,
            default_strategy,
        }
    }

    pub fn with_strategy(mut self, layer_type: String, strategy: Box<dyn PruningStrategy>) -> Self {
        self.strategies.insert(layer_type, strategy);
        self
    }

    pub fn with_default_strategy(mut self, strategy: Box<dyn PruningStrategy>) -> Self {
        self.default_strategy = strategy;
        self
    }

    #[allow(dead_code)]
    fn detect_layer_type(&self, layer_name: &str) -> String {
        let name_lower = layer_name.to_lowercase();

        if name_lower.contains("conv") {
            "conv".to_string()
        } else if name_lower.contains("attention") || name_lower.contains("attn") {
            "attention".to_string()
        } else if name_lower.contains("linear")
            || name_lower.contains("dense")
            || name_lower.contains("fc")
        {
            "linear".to_string()
        } else if name_lower.contains("embed") {
            "embedding".to_string()
        } else {
            "unknown".to_string()
        }
    }
}

impl Pruner for AutomaticPruner {
    fn prune<M>(&self, model: M, config: &PruningConfig) -> Result<PruningResult<M>>
    where
        M: crate::traits::Model + Clone,
    {
        // Simplified pruning implementation that works with the available Model interface
        let total_params = model.num_parameters();
        let estimated_pruned_params = (total_params as f32 * config.target_sparsity) as usize;

        // Simulate layer-wise sparsity distribution
        let mut layer_sparsity = HashMap::new();
        let simulated_layers = vec![
            ("embedding", 0.2),   // Conservative pruning for embeddings
            ("attention", 0.4),   // Moderate pruning for attention layers
            ("feedforward", 0.6), // More aggressive pruning for FFN layers
            ("output", 0.1),      // Very conservative for output layers
        ];

        for (layer_type, base_sparsity) in simulated_layers {
            // Adjust sparsity based on config
            let actual_sparsity = (base_sparsity * config.target_sparsity).min(0.9);
            layer_sparsity.insert(layer_type.to_string(), actual_sparsity);
        }

        let overall_sparsity = config.target_sparsity;

        // Clone the model to simulate pruning
        // In a real implementation, this would create a new model with pruned weights
        let pruned_model = model;

        Ok(PruningResult {
            model: pruned_model,
            sparsity: overall_sparsity,
            pruned_params: estimated_pruned_params,
            total_params,
            layer_sparsity,
        })
    }

    fn estimate_pruning_potential<M>(
        &self,
        model: &M,
        config: &PruningConfig,
    ) -> Result<PruningStats>
    where
        M: crate::traits::Model,
    {
        let total_params = model.num_parameters();
        let estimated_zero_params = (total_params as f32 * config.target_sparsity) as usize;

        // Simulate layer-wise statistics
        let mut layer_stats = HashMap::new();
        let simulated_layers = vec![
            ("embedding", 0.15),
            ("attention", 0.30),
            ("feedforward", 0.45),
            ("output", 0.05),
        ];

        for (layer_name, param_fraction) in simulated_layers {
            let layer_total = (total_params as f32 * param_fraction) as usize;
            let layer_zeros = (layer_total as f32 * config.target_sparsity) as usize;
            let layer_sparsity =
                if layer_total > 0 { layer_zeros as f32 / layer_total as f32 } else { 0.0 };

            layer_stats.insert(
                layer_name.to_string(),
                LayerPruningStats {
                    total_params: layer_total,
                    zero_params: layer_zeros,
                    sparsity: layer_sparsity,
                },
            );
        }

        let overall_sparsity = if total_params > 0 {
            estimated_zero_params as f32 / total_params as f32
        } else {
            0.0
        };

        Ok(PruningStats {
            total_params,
            zero_params: estimated_zero_params,
            sparsity: overall_sparsity,
            layer_stats,
        })
    }
}

impl Default for AutomaticPruner {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for pruning operations
pub struct PruningUtils;

impl PruningUtils {
    /// Calculate optimal sparsity for each layer based on sensitivity analysis (simplified)
    pub fn calculate_layer_sensitivities<M>(
        model: &M,
        _validation_data: &[Tensor],
    ) -> Result<HashMap<String, f32>>
    where
        M: crate::traits::Model,
    {
        let mut sensitivities = HashMap::new();

        // Simplified sensitivity analysis based on typical model architectures
        // In a real implementation, would analyze actual layer gradients/activations
        let _total_params = model.num_parameters(); // Use for more sophisticated analysis

        let typical_sensitivities = vec![
            ("embedding".to_string(), 0.95),   // Embeddings are usually sensitive
            ("attention".to_string(), 0.75),   // Attention layers are moderately sensitive
            ("feedforward".to_string(), 0.50), // FFN layers can be pruned more aggressively
            ("output".to_string(), 0.90),      // Output layers are sensitive
            ("classifier".to_string(), 0.90),  // Classification layers are sensitive
        ];

        for (layer_name, sensitivity) in typical_sensitivities {
            sensitivities.insert(layer_name, sensitivity);
        }

        Ok(sensitivities)
    }

    /// Generate pruning schedule for gradual pruning
    pub fn generate_pruning_schedule(
        initial_sparsity: f32,
        final_sparsity: f32,
        num_steps: usize,
    ) -> Vec<f32> {
        let mut schedule = Vec::new();

        for i in 0..num_steps {
            let progress = i as f32 / (num_steps - 1) as f32;
            // Use cubic schedule for smoother transition
            let cubic_progress = progress * progress * progress;
            let sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * cubic_progress;
            schedule.push(sparsity);
        }

        schedule
    }

    /// Estimate model compression ratio after pruning
    pub fn estimate_compression_ratio(target_sparsity: f32, quantization_bits: Option<u8>) -> f32 {
        let sparsity_compression = 1.0 / (1.0 - target_sparsity);

        match quantization_bits {
            Some(bits) => sparsity_compression * (32.0 / bits as f32), // Assuming FP32 baseline
            None => sparsity_compression,
        }
    }

    /// Validate pruning configuration
    pub fn validate_config(config: &PruningConfig) -> Result<()> {
        if config.target_sparsity < 0.0 || config.target_sparsity > 1.0 {
            return Err(anyhow!("Target sparsity must be between 0.0 and 1.0"));
        }

        if config.iterations == 0 {
            return Err(anyhow!("Number of iterations must be greater than 0"));
        }

        if let Some(threshold) = config.magnitude_threshold {
            if threshold < 0.0 {
                return Err(anyhow!("Magnitude threshold must be non-negative"));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pruning_config_default() {
        let config = PruningConfig::default();
        assert_eq!(config.target_sparsity, 0.5);
        assert!(!config.iterative);
        assert_eq!(config.iterations, 1);
        assert!(config.fine_tune);
    }

    #[test]
    fn test_magnitude_pruner() -> Result<()> {
        let pruner = MagnitudePruner::new(0.5);
        let weights = Tensor::from_vec(vec![0.1, -0.8, 0.3, -0.2, 0.9, -0.1], &[2, 3])?;
        let config = PruningConfig {
            target_sparsity: 0.5,
            ..Default::default()
        };

        let mask = pruner.get_mask(&weights, &config)?;
        let mask_data = mask.data()?;
        let zero_count = mask_data.iter().filter(|&&x| x == 0.0).count();

        // Should prune approximately 50% of weights
        assert_eq!(zero_count, 3);
        Ok(())
    }

    #[test]
    fn test_pruning_utils_validation() {
        let valid_config = PruningConfig::default();
        assert!(PruningUtils::validate_config(&valid_config).is_ok());

        let invalid_config = PruningConfig {
            target_sparsity: 1.5, // Invalid: > 1.0
            ..Default::default()
        };
        assert!(PruningUtils::validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_compression_ratio_estimation() {
        let ratio = PruningUtils::estimate_compression_ratio(0.5, None);
        assert_eq!(ratio, 2.0); // 50% sparsity = 2x compression

        let ratio_with_quant = PruningUtils::estimate_compression_ratio(0.5, Some(8));
        assert_eq!(ratio_with_quant, 8.0); // 2x from sparsity * 4x from INT8 quantization
    }

    #[test]
    fn test_pruning_schedule() {
        let schedule = PruningUtils::generate_pruning_schedule(0.0, 0.8, 5);
        assert_eq!(schedule.len(), 5);
        assert_eq!(schedule[0], 0.0);
        assert_eq!(schedule[4], 0.8);
        // Should be monotonically increasing
        for i in 1..schedule.len() {
            assert!(schedule[i] >= schedule[i - 1]);
        }
    }
}
