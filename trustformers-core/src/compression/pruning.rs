//! Model Pruning Implementation
//!
//! Various strategies for removing unnecessary weights and structures

#![allow(clippy::excessive_nesting)] // Complex pruning algorithms require deep nesting

use crate::tensor::Tensor;
use anyhow::{anyhow, Result};
use scirs2_core::random::*; // SciRS2 Policy compliant
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
        abs_weights.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(::std::cmp::Ordering::Equal));

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

impl StructuredPruner {
    /// Indices of the structures (channels / filters) this configuration prunes.
    fn pruned_structures(
        &self,
        weights: &Tensor,
        config: &PruningConfig,
    ) -> Result<HashSet<usize>> {
        let shape = weights.shape();
        if shape.len() < 2 {
            return Err(anyhow!("Structured pruning requires at least 2D tensors"));
        }
        if self.pruning_dim >= shape.len() {
            return Err(anyhow!(
                "pruning dimension {} is out of range for shape {:?}",
                self.pruning_dim,
                shape
            ));
        }

        let importance_scores = self.calculate_importance(weights)?;
        let num_structures = shape[self.pruning_dim];
        let num_prune = (num_structures as f32 * config.target_sparsity) as usize;

        let mut indices: Vec<usize> = (0..num_structures).collect();
        indices.sort_by(|&a, &b| {
            importance_scores[a]
                .partial_cmp(&importance_scores[b])
                .unwrap_or(::std::cmp::Ordering::Equal)
        });

        Ok(indices.iter().take(num_prune).copied().collect())
    }
}

impl PruningStrategy for StructuredPruner {
    fn prune_weights(&self, weights: &Tensor, config: &PruningConfig) -> Result<Tensor> {
        // Structured pruning removes entire channels/filters
        let shape = weights.shape();
        let pruned_indices = self.pruned_structures(weights, config)?;

        // Create pruned tensor
        let data = weights.data()?;
        let mut pruned_data = Vec::with_capacity(data.len());

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

    /// Binary keep-mask matching exactly what [`Self::prune_weights`] zeroes.
    fn get_mask(&self, weights: &Tensor, config: &PruningConfig) -> Result<Tensor> {
        let shape = weights.shape();
        let pruned_indices = self.pruned_structures(weights, config)?;
        let structure_stride = shape.iter().skip(self.pruning_dim + 1).product::<usize>();
        let num_structures = shape[self.pruning_dim];

        let element_count = weights.data()?.len();
        let mask: Vec<f32> = (0..element_count)
            .map(|i| {
                let structure_idx = (i / structure_stride) % num_structures;
                if pruned_indices.contains(&structure_idx) {
                    0.0
                } else {
                    1.0
                }
            })
            .collect();

        Ok(Tensor::from_vec(mask, &shape)?)
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
            let mut rng = thread_rng();
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
            let mut rng = thread_rng();
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
        sorted_channels
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(::std::cmp::Ordering::Equal));

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
        sorted_channels
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(::std::cmp::Ordering::Equal));

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
        sorted_filters.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(::std::cmp::Ordering::Equal));

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
        sorted_filters.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(::std::cmp::Ordering::Equal));

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
        sorted_heads.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(::std::cmp::Ordering::Equal));

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
        sorted_heads.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(::std::cmp::Ordering::Equal));

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

    /// Score each layer by the L2 norm of its real weights.
    ///
    /// The scores come from [`crate::traits::Model::named_tensors`], so they
    /// describe the model in front of you. A model with no named tensors is an
    /// error rather than an excuse to invent a layer table.
    pub fn analyze_model<M>(&mut self, model: &M) -> Result<()>
    where
        M: crate::traits::Model,
    {
        let tensors = model.named_tensors();
        if tensors.is_empty() {
            return Err(anyhow!(
                "LayerPruner::analyze_model needs weight access: this model exposes no tensors \
                 through Model::named_tensors"
            ));
        }

        self.layer_importance.clear();
        for (name, tensor) in tensors {
            let l2_norm = tensor
                .data()?
                .iter()
                .map(|value| (*value as f64) * (*value as f64))
                .sum::<f64>()
                .sqrt() as f32;
            self.layer_importance.insert(name, l2_norm);
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
        sorted_layers.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(::std::cmp::Ordering::Equal));

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
    /// Prune a model in place by rewriting its named tensors.
    ///
    /// Every parameter tensor exposed by [`crate::traits::Model::named_tensors_mut`]
    /// is passed through the strategy registered for its detected layer type
    /// and written back, so the model really changes. The reported sparsity is
    /// then *measured* from the resulting weights, never assumed from
    /// `config.target_sparsity`.
    ///
    /// Errors when the model exposes no named tensors: without weight access
    /// nothing can be pruned, and reporting a sparsity for an untouched model
    /// would be a fabrication.
    fn prune<M>(&self, model: M, config: &PruningConfig) -> Result<PruningResult<M>>
    where
        M: crate::traits::Model + Clone,
    {
        let mut model = model;
        let total_params_reported = model.num_parameters();

        let mut layer_sparsity = HashMap::new();
        let mut total_elements = 0usize;
        let mut total_zeros = 0usize;
        let mut pruned_layers = 0usize;

        {
            let tensors = model.named_tensors_mut();
            if tensors.is_empty() {
                return Err(anyhow!(
                    "AutomaticPruner::prune needs weight access: this model exposes no tensors \
                     through Model::named_tensors_mut, so nothing can be pruned. Implement \
                     named_tensors_mut on the model rather than reporting a sparsity for \
                     untouched weights."
                ));
            }

            for (name, tensor) in tensors {
                let element_count = tensor.data()?.len();
                total_elements += element_count;

                if config.exclude_layers.contains(&name) {
                    let zeros = count_zeros(tensor)?;
                    total_zeros += zeros;
                    layer_sparsity.insert(name, sparsity_of(zeros, element_count));
                    continue;
                }

                let layer_type = self.detect_layer_type(&name);
                let strategy = self
                    .strategies
                    .get(&layer_type)
                    .map(|boxed| boxed.as_ref())
                    .unwrap_or(self.default_strategy.as_ref());

                match strategy.prune_weights(tensor, config) {
                    Ok(pruned) => {
                        *tensor = pruned;
                        pruned_layers += 1;
                    },
                    Err(error) => {
                        // A strategy that cannot handle this tensor's shape
                        // (for example structured pruning of a 1-D bias) leaves
                        // it untouched; its real sparsity is still measured.
                        tracing::debug!(
                            layer = %name,
                            strategy = strategy.name(),
                            "pruning strategy skipped this tensor: {}",
                            error
                        );
                    },
                }

                let zeros = count_zeros(tensor)?;
                total_zeros += zeros;
                layer_sparsity.insert(name, sparsity_of(zeros, element_count));
            }
        }

        if pruned_layers == 0 {
            return Err(anyhow!(
                "no tensor could be pruned: every registered strategy rejected every parameter \
                 tensor"
            ));
        }

        let total_params = if total_elements > 0 { total_elements } else { total_params_reported };

        Ok(PruningResult {
            model,
            sparsity: sparsity_of(total_zeros, total_params),
            pruned_params: total_zeros,
            total_params,
            layer_sparsity,
        })
    }

    /// Measure the model's *current* sparsity, per layer and overall.
    ///
    /// This reads the live weights; it does not predict what pruning would
    /// achieve, and it does not derive numbers from `config.target_sparsity`.
    fn estimate_pruning_potential<M>(
        &self,
        model: &M,
        _config: &PruningConfig,
    ) -> Result<PruningStats>
    where
        M: crate::traits::Model,
    {
        let tensors = model.named_tensors();
        if tensors.is_empty() {
            return Err(anyhow!(
                "estimate_pruning_potential needs weight access: this model exposes no tensors \
                 through Model::named_tensors"
            ));
        }

        let mut layer_stats = HashMap::new();
        let mut total_params = 0usize;
        let mut zero_params = 0usize;

        for (name, tensor) in tensors {
            let element_count = tensor.data()?.len();
            let zeros = count_zeros(tensor)?;
            total_params += element_count;
            zero_params += zeros;

            layer_stats.insert(
                name,
                LayerPruningStats {
                    total_params: element_count,
                    zero_params: zeros,
                    sparsity: sparsity_of(zeros, element_count),
                },
            );
        }

        Ok(PruningStats {
            total_params,
            zero_params,
            sparsity: sparsity_of(zero_params, total_params),
            layer_stats,
        })
    }
}

/// Number of exactly-zero elements in a tensor.
fn count_zeros(tensor: &Tensor) -> Result<usize> {
    Ok(tensor.data()?.iter().filter(|value| **value == 0.0).count())
}

/// Fraction of `zeros` among `total`, or 0.0 for an empty tensor.
fn sparsity_of(zeros: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        zeros as f32 / total as f32
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
    /// Measure how sensitive the model's output is to each parameter tensor,
    /// by real ablation.
    ///
    /// For every tensor exposed through
    /// [`crate::traits::Model::named_tensors_mut`] the tensor is temporarily
    /// zeroed, the model is run over `validation_data`, and the mean absolute
    /// output change against the un-ablated baseline is recorded; the original
    /// weights are then restored. Scores are normalised to `[0, 1]` by the
    /// largest observed change, so 1.0 marks the most sensitive tensor.
    ///
    /// Errors when the model exposes no named tensors or when
    /// `validation_data` is empty — either way there is nothing to measure.
    pub fn calculate_layer_sensitivities<M>(
        model: &mut M,
        validation_data: &[Tensor],
    ) -> Result<HashMap<String, f32>>
    where
        M: crate::traits::Model<Input = Tensor, Output = Tensor>,
    {
        if validation_data.is_empty() {
            return Err(anyhow!(
                "layer sensitivity analysis needs validation data to ablate against"
            ));
        }
        if model.named_tensors().is_empty() {
            return Err(anyhow!(
                "layer sensitivity analysis needs weight access: this model exposes no tensors \
                 through Model::named_tensors_mut"
            ));
        }

        // Baseline outputs before any ablation.
        let mut baselines = Vec::with_capacity(validation_data.len());
        for input in validation_data {
            baselines.push(model.forward(input.clone())?.data()?);
        }

        let layer_names: Vec<String> =
            model.named_tensors().into_iter().map(|(name, _)| name).collect();

        let mut raw_sensitivities: HashMap<String, f64> = HashMap::new();

        for layer_name in layer_names {
            // Zero this tensor, keeping the original values for restoration.
            let original = {
                let mut saved = None;
                for (name, tensor) in model.named_tensors_mut() {
                    if name == layer_name {
                        saved = Some(tensor.clone());
                        *tensor = Tensor::zeros(&tensor.shape())?;
                        break;
                    }
                }
                match saved {
                    Some(tensor) => tensor,
                    None => continue,
                }
            };

            let mut total_change = 0.0f64;
            let mut counted = 0usize;
            let mut failure = None;

            for (input, baseline) in validation_data.iter().zip(baselines.iter()) {
                match model.forward(input.clone()).and_then(|output| output.data()) {
                    Ok(ablated) => {
                        if ablated.len() != baseline.len() {
                            failure = Some(anyhow!(
                                "ablating '{}' changed the output length from {} to {}",
                                layer_name,
                                baseline.len(),
                                ablated.len()
                            ));
                            break;
                        }
                        let change: f64 = baseline
                            .iter()
                            .zip(ablated.iter())
                            .map(|(base, value)| (base - value).abs() as f64)
                            .sum();
                        total_change += change / baseline.len().max(1) as f64;
                        counted += 1;
                    },
                    Err(error) => {
                        failure = Some(anyhow!("{}", error));
                        break;
                    },
                }
            }

            // Always restore the weights, even if the forward pass failed.
            for (name, tensor) in model.named_tensors_mut() {
                if name == layer_name {
                    *tensor = original;
                    break;
                }
            }

            if let Some(error) = failure {
                return Err(error);
            }

            let mean_change = if counted > 0 { total_change / counted as f64 } else { 0.0 };
            raw_sensitivities.insert(layer_name, mean_change);
        }

        let max_change = raw_sensitivities.values().copied().fold(0.0f64, f64::max);

        Ok(raw_sensitivities
            .into_iter()
            .map(|(name, change)| {
                let normalised = if max_change > 0.0 { (change / max_change) as f32 } else { 0.0 };
                (name, normalised)
            })
            .collect())
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
    use crate::traits::{Config, Model};
    use std::io::Read;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct TinyConfig;

    impl Config for TinyConfig {
        fn architecture(&self) -> &'static str {
            "tiny"
        }
    }

    /// A model that exposes its real weights.
    #[derive(Debug, Clone)]
    struct TinyModel {
        config: TinyConfig,
        linear_weight: Tensor,
        embedding_weight: Tensor,
    }

    impl TinyModel {
        fn new() -> Self {
            Self {
                config: TinyConfig,
                linear_weight: Tensor::from_vec(
                    vec![0.9, -0.8, 0.05, -0.02, 0.7, -0.6, 0.01, -0.03],
                    &[2, 4],
                )
                .expect("from_vec failed"),
                embedding_weight: Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
                    .expect("from_vec failed"),
            }
        }
    }

    impl Model for TinyModel {
        type Config = TinyConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, input: Self::Input) -> crate::errors::Result<Self::Output> {
            // A real dependency on the weights, so ablating one changes the output.
            let scale: f32 = self.linear_weight.data()?.iter().sum::<f32>()
                + self.embedding_weight.data()?.iter().sum::<f32>();
            input.scalar_mul(scale)
        }

        fn load_pretrained(&mut self, _reader: &mut dyn Read) -> crate::errors::Result<()> {
            Ok(())
        }

        fn get_config(&self) -> &Self::Config {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            12
        }

        fn named_tensors(&self) -> Vec<(String, &Tensor)> {
            vec![
                ("linear.weight".to_string(), &self.linear_weight),
                ("embedding.weight".to_string(), &self.embedding_weight),
            ]
        }

        fn named_tensors_mut(&mut self) -> Vec<(String, &mut Tensor)> {
            vec![
                ("linear.weight".to_string(), &mut self.linear_weight),
                ("embedding.weight".to_string(), &mut self.embedding_weight),
            ]
        }
    }

    /// A model with no weight access at all.
    #[derive(Debug, Clone)]
    struct OpaqueModel;

    impl Model for OpaqueModel {
        type Config = TinyConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, input: Self::Input) -> crate::errors::Result<Self::Output> {
            Ok(input)
        }

        fn load_pretrained(&mut self, _reader: &mut dyn Read) -> crate::errors::Result<()> {
            Ok(())
        }

        fn get_config(&self) -> &Self::Config {
            &TinyConfig
        }

        fn num_parameters(&self) -> usize {
            1_000_000
        }
    }

    /// Regression test: `AutomaticPruner::prune` used to return the model
    /// untouched while reporting `sparsity = config.target_sparsity` and a
    /// hardcoded per-layer table of embedding/attention/feedforward/output.
    #[test]
    fn test_automatic_pruner_really_zeroes_weights() -> Result<()> {
        let pruner = AutomaticPruner::new();
        let config = PruningConfig {
            target_sparsity: 0.5,
            ..Default::default()
        };

        let model = TinyModel::new();
        let before = model.linear_weight.data()?;
        let result = pruner.prune(model, &config)?;

        let after = result.model.linear_weight.data()?;
        assert_ne!(before, after, "the weights must actually change");
        assert!(
            after.iter().filter(|value| **value == 0.0).count() > 0,
            "pruning must produce real zeros: {after:?}"
        );

        // The sparsity must be measured, and the layer map must name the real
        // tensors, not the old invented layer types.
        assert!(result.sparsity > 0.0);
        assert_eq!(result.pruned_params, {
            let mut zeros = 0;
            for (_, tensor) in result.model.named_tensors() {
                zeros += tensor.data()?.iter().filter(|value| **value == 0.0).count();
            }
            zeros
        });
        assert!(result.layer_sparsity.contains_key("linear.weight"));
        assert!(result.layer_sparsity.contains_key("embedding.weight"));
        assert!(!result.layer_sparsity.contains_key("feedforward"));
        assert_eq!(result.total_params, 12);
        Ok(())
    }

    /// Excluded layers must be left alone but still measured honestly.
    #[test]
    fn test_excluded_layers_are_not_pruned() -> Result<()> {
        let pruner = AutomaticPruner::new();
        let mut exclude = HashSet::new();
        exclude.insert("embedding.weight".to_string());
        let config = PruningConfig {
            target_sparsity: 0.9,
            exclude_layers: exclude,
            ..Default::default()
        };

        let original = TinyModel::new().embedding_weight.data()?;
        let result = pruner.prune(TinyModel::new(), &config)?;
        assert_eq!(result.model.embedding_weight.data()?, original);
        assert_eq!(result.layer_sparsity.get("embedding.weight"), Some(&0.0));
        Ok(())
    }

    /// Regression test: a model with no weight access must not get a sparsity
    /// report at all.
    #[test]
    fn test_pruning_a_model_without_weight_access_is_refused() {
        let pruner = AutomaticPruner::new();
        let config = PruningConfig::default();
        let error = pruner
            .prune(OpaqueModel, &config)
            .expect_err("nothing can be pruned without weight access");
        assert!(error.to_string().contains("named_tensors_mut"));

        assert!(pruner.estimate_pruning_potential(&OpaqueModel, &config).is_err());
    }

    /// Regression test: `estimate_pruning_potential` used to derive per-layer
    /// statistics from a fixed `embedding/attention/feedforward/output` table.
    #[test]
    fn test_estimate_pruning_potential_measures_current_sparsity() -> Result<()> {
        let pruner = AutomaticPruner::new();
        let config = PruningConfig::default();

        let mut model = TinyModel::new();
        // Zero three of the twelve weights.
        model.linear_weight =
            Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 4])?;

        let stats = pruner.estimate_pruning_potential(&model, &config)?;
        assert_eq!(stats.total_params, 12);
        assert_eq!(stats.zero_params, 3);
        assert!((stats.sparsity - 0.25).abs() < 1e-6);
        assert!(stats.layer_stats.contains_key("linear.weight"));
        assert_eq!(
            stats.layer_stats.get("linear.weight").map(|s| s.zero_params),
            Some(3)
        );
        Ok(())
    }

    /// Regression test: `LayerPruner::analyze_model` used to insert a fixed
    /// six-entry layer table scaled by `num_parameters()`.
    #[test]
    fn test_layer_pruner_scores_real_tensors() -> Result<()> {
        let mut pruner = LayerPruner::new();
        pruner.analyze_model(&TinyModel::new())?;

        let candidates = pruner.get_pruning_candidates(&PruningConfig {
            target_sparsity: 0.5,
            ..Default::default()
        })?;
        assert!(!candidates.is_empty());
        for name in &candidates {
            assert!(
                name == "linear.weight" || name == "embedding.weight",
                "unexpected layer name {name}"
            );
        }

        assert!(LayerPruner::new().analyze_model(&OpaqueModel).is_err());
        Ok(())
    }

    /// Regression test: `calculate_layer_sensitivities` returned the same five
    /// constants for every model.
    #[test]
    fn test_layer_sensitivities_are_measured_by_ablation() -> Result<()> {
        let mut model = TinyModel::new();
        let validation = vec![Tensor::from_vec(vec![1.0, 1.0], &[1, 2])?];

        let sensitivities = PruningUtils::calculate_layer_sensitivities(&mut model, &validation)?;

        assert_eq!(sensitivities.len(), 2);
        assert!(sensitivities.contains_key("linear.weight"));
        assert!(sensitivities.contains_key("embedding.weight"));
        assert!(!sensitivities.contains_key("classifier"));

        // The embedding weights sum to 10.0 and the linear ones to ~0.21, so
        // ablating the embedding must matter far more.
        let embedding = sensitivities["embedding.weight"];
        let linear = sensitivities["linear.weight"];
        assert!(
            embedding > linear,
            "embedding ({embedding}) should dominate linear ({linear})"
        );
        assert!(
            (embedding - 1.0).abs() < 1e-6,
            "the max must normalise to 1.0"
        );

        // The ablation must restore the weights it borrowed.
        assert_eq!(model.embedding_weight.data()?, vec![1.0, 2.0, 3.0, 4.0]);

        assert!(PruningUtils::calculate_layer_sensitivities(&mut model, &[]).is_err());
        Ok(())
    }

    /// Regression test: `StructuredPruner::get_mask` returned all ones, so the
    /// mask never matched what `prune_weights` actually zeroed.
    #[test]
    fn test_structured_pruner_mask_matches_pruned_weights() -> Result<()> {
        let pruner = StructuredPruner::new(0);
        let weights = Tensor::from_vec(vec![10.0, 10.0, 0.1, 0.1], &[2, 2])?;
        let config = PruningConfig {
            target_sparsity: 0.5,
            ..Default::default()
        };

        let mask = pruner.get_mask(&weights, &config)?.data()?;
        let pruned = pruner.prune_weights(&weights, &config)?.data()?;

        assert!(mask.contains(&0.0), "mask must not be all ones");
        for (index, (mask_value, pruned_value)) in mask.iter().zip(pruned.iter()).enumerate() {
            if *mask_value == 0.0 {
                assert_eq!(*pruned_value, 0.0, "element {index} masked but not zeroed");
            }
        }
        Ok(())
    }

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
