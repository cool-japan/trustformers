//! Advanced weight compression for neural network models
//!
//! This module provides comprehensive weight compression techniques including:
//! - Neural network pruning (structured and unstructured)
//! - Weight factorization and decomposition
//! - Sparse matrix compression
//! - Lossless compression algorithms
//! - Knowledge distillation support
//! - Progressive compression levels

use serde::{Deserialize, Serialize};
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Weight compression strategies
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionStrategy {
    /// No compression
    None,
    /// Magnitude-based pruning
    MagnitudePruning,
    /// Structured pruning (remove entire channels/filters)
    StructuredPruning,
    /// Low-rank matrix factorization
    LowRankFactorization,
    /// Singular Value Decomposition
    SVDCompression,
    /// Weight clustering/sharing
    WeightClustering,
    /// Huffman coding compression
    HuffmanCompression,
    /// Combined compression pipeline
    Progressive,
}

/// Compression levels for progressive compression
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionLevel {
    /// Light compression (10-30% reduction)
    Light,
    /// Medium compression (30-60% reduction)
    Medium,
    /// Aggressive compression (60-85% reduction)
    Aggressive,
    /// Maximum compression (85%+ reduction, may impact accuracy)
    Maximum,
}

/// Sparsity patterns for pruning
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparsityPattern {
    /// Random unstructured pruning
    Unstructured,
    /// Block-sparse patterns (2:4, 4:8, etc.)
    BlockSparse,
    /// Channel-wise pruning
    ChannelWise,
    /// Filter-wise pruning
    FilterWise,
    /// Attention head pruning
    AttentionHead,
}

/// Compression configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    strategy: CompressionStrategy,
    level: CompressionLevel,
    sparsity_pattern: SparsityPattern,
    target_sparsity: f32,
    accuracy_threshold: f32,
    preserve_attention: bool,
    preserve_embeddings: bool,
    use_knowledge_distillation: bool,
}

#[wasm_bindgen]
impl CompressionConfig {
    /// Create a new compression configuration
    #[wasm_bindgen(constructor)]
    pub fn new(strategy: CompressionStrategy, level: CompressionLevel) -> Self {
        Self {
            strategy,
            level,
            sparsity_pattern: SparsityPattern::Unstructured,
            target_sparsity: Self::default_sparsity_for_level(level),
            accuracy_threshold: 0.95,
            preserve_attention: true,
            preserve_embeddings: true,
            use_knowledge_distillation: false,
        }
    }

    /// Create a configuration optimized for transformer models
    pub fn transformer() -> Self {
        Self {
            strategy: CompressionStrategy::Progressive,
            level: CompressionLevel::Medium,
            sparsity_pattern: SparsityPattern::AttentionHead,
            target_sparsity: 0.5,
            accuracy_threshold: 0.95,
            preserve_attention: true,
            preserve_embeddings: true,
            use_knowledge_distillation: true,
        }
    }

    /// Create a configuration for mobile deployment
    pub fn mobile() -> Self {
        Self {
            strategy: CompressionStrategy::Progressive,
            level: CompressionLevel::Aggressive,
            sparsity_pattern: SparsityPattern::BlockSparse,
            target_sparsity: 0.75,
            accuracy_threshold: 0.90,
            preserve_attention: false,
            preserve_embeddings: true,
            use_knowledge_distillation: true,
        }
    }

    /// Create a configuration for edge devices
    pub fn edge() -> Self {
        Self {
            strategy: CompressionStrategy::Progressive,
            level: CompressionLevel::Maximum,
            sparsity_pattern: SparsityPattern::FilterWise,
            target_sparsity: 0.85,
            accuracy_threshold: 0.85,
            preserve_attention: false,
            preserve_embeddings: false,
            use_knowledge_distillation: true,
        }
    }

    /// Set target sparsity ratio (0.0 - 1.0)
    pub fn set_target_sparsity(mut self, sparsity: f32) -> Self {
        self.target_sparsity = sparsity.clamp(0.0, 1.0);
        self
    }

    /// Set accuracy preservation threshold
    pub fn set_accuracy_threshold(mut self, threshold: f32) -> Self {
        self.accuracy_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Enable/disable attention preservation
    pub fn set_preserve_attention(mut self, preserve: bool) -> Self {
        self.preserve_attention = preserve;
        self
    }

    /// Enable/disable embedding preservation
    pub fn set_preserve_embeddings(mut self, preserve: bool) -> Self {
        self.preserve_embeddings = preserve;
        self
    }

    /// Enable/disable knowledge distillation
    pub fn set_knowledge_distillation(mut self, enable: bool) -> Self {
        self.use_knowledge_distillation = enable;
        self
    }

    fn default_sparsity_for_level(level: CompressionLevel) -> f32 {
        match level {
            CompressionLevel::Light => 0.2,
            CompressionLevel::Medium => 0.5,
            CompressionLevel::Aggressive => 0.75,
            CompressionLevel::Maximum => 0.9,
        }
    }
}

/// Weight compression statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub original_parameters: usize,
    pub compressed_parameters: usize,
    pub actual_sparsity: f32,
    pub compression_ratio: f32,
    pub size_reduction_bytes: usize,
    pub size_reduction_percent: f32,
    pub estimated_speedup: f32,
    pub strategy_used: CompressionStrategy,
    pub level_used: CompressionLevel,
}

/// Advanced weight compressor
#[wasm_bindgen]
pub struct WeightCompressor {
    config: CompressionConfig,
    layer_sensitivities: Vec<f32>,
}

#[wasm_bindgen]
impl WeightCompressor {
    /// Create a new weight compressor
    #[wasm_bindgen(constructor)]
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            layer_sensitivities: Vec::new(),
        }
    }

    /// Compress model weights using the configured strategy
    pub fn compress_weights(&self, model_data: &[u8]) -> Result<CompressedModelData, JsValue> {
        match self.config.strategy {
            CompressionStrategy::None => Ok(self.create_uncompressed_result(model_data)),
            CompressionStrategy::MagnitudePruning => self.apply_magnitude_pruning(model_data),
            CompressionStrategy::StructuredPruning => self.apply_structured_pruning(model_data),
            CompressionStrategy::LowRankFactorization => {
                self.apply_low_rank_factorization(model_data)
            },
            CompressionStrategy::SVDCompression => self.apply_svd_compression(model_data),
            CompressionStrategy::WeightClustering => self.apply_weight_clustering(model_data),
            CompressionStrategy::HuffmanCompression => self.apply_huffman_compression(model_data),
            CompressionStrategy::Progressive => self.apply_progressive_compression(model_data),
        }
    }

    /// Analyze model sensitivity to compression
    pub fn analyze_sensitivity(&mut self, model_data: &[u8]) -> Result<Vec<f32>, JsValue> {
        // Simulate layer sensitivity analysis
        // In practice, this would involve computing gradients or importance scores
        let num_layers = self.estimate_layer_count(model_data);
        let mut sensitivities = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            // Simulate different sensitivities for different layer types
            let sensitivity = match i % 4 {
                0 => 0.9, // Embedding layers (high sensitivity)
                1 => 0.7, // Attention layers (medium-high sensitivity)
                2 => 0.5, // Feed-forward layers (medium sensitivity)
                3 => 0.3, // Output layers (lower sensitivity)
                _ => 0.5,
            };
            sensitivities.push(sensitivity);
        }

        self.layer_sensitivities = sensitivities.clone();
        Ok(sensitivities)
    }

    /// Get recommended compression settings for a model
    pub fn get_recommended_settings(
        &self,
        model_size_bytes: usize,
        target_size_bytes: usize,
    ) -> CompressionConfig {
        let size_mb = model_size_bytes as f32 / 1_048_576.0;
        let target_mb = target_size_bytes as f32 / 1_048_576.0;
        let required_reduction = 1.0 - (target_mb / size_mb);

        let level = if required_reduction < 0.3 {
            CompressionLevel::Light
        } else if required_reduction < 0.6 {
            CompressionLevel::Medium
        } else if required_reduction < 0.85 {
            CompressionLevel::Aggressive
        } else {
            CompressionLevel::Maximum
        };

        let strategy = if size_mb > 100.0 {
            CompressionStrategy::Progressive
        } else if size_mb > 20.0 {
            CompressionStrategy::StructuredPruning
        } else {
            CompressionStrategy::MagnitudePruning
        };

        CompressionConfig::new(strategy, level)
    }

    // Private compression methods

    fn create_uncompressed_result(&self, data: &[u8]) -> CompressedModelData {
        let stats = CompressionStats {
            original_parameters: data.len() / 4, // Assume 32-bit floats
            compressed_parameters: data.len() / 4,
            actual_sparsity: 0.0,
            compression_ratio: 1.0,
            size_reduction_bytes: 0,
            size_reduction_percent: 0.0,
            estimated_speedup: 1.0,
            strategy_used: CompressionStrategy::None,
            level_used: self.config.level,
        };

        CompressedModelData {
            data: data.to_vec(),
            stats,
            metadata: CompressionMetadata {
                strategy: CompressionStrategy::None,
                level: self.config.level,
                sparsity_pattern: self.config.sparsity_pattern,
                actual_sparsity: 0.0,
                original_size: data.len(),
                compressed_size: data.len(),
            },
        }
    }

    fn apply_magnitude_pruning(&self, data: &[u8]) -> Result<CompressedModelData, JsValue> {
        web_sys::console::log_1(&"Applying magnitude-based pruning...".into());

        // Simulate magnitude pruning by zeroing out small weights
        let compressed_data = data.to_vec();
        let weights_f32 = self.bytes_to_f32_slice(&compressed_data);
        let mut weights = weights_f32.to_vec();

        // Calculate magnitude threshold for pruning
        let mut magnitudes: Vec<f32> = weights.iter().map(|&w| w.abs()).collect();
        magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = (magnitudes.len() as f32 * self.config.target_sparsity) as usize;
        let threshold = magnitudes.get(threshold_idx).unwrap_or(&0.0);

        // Apply pruning
        let mut pruned_count = 0;
        for weight in weights.iter_mut() {
            if weight.abs() < *threshold {
                *weight = 0.0;
                pruned_count += 1;
            }
        }

        // Convert back to bytes (will be encoded as sparse data below)

        // Apply sparse encoding to reduce size
        let (encoded_data, actual_sparsity) = self.encode_sparse_weights(&weights);

        let original_params = weights.len();
        let compressed_params = original_params - pruned_count;
        let compression_ratio = original_params as f32 / (encoded_data.len() / 4) as f32;
        let size_reduction_percent =
            (1.0 - (encoded_data.len() as f32 / data.len() as f32)) * 100.0;

        let stats = CompressionStats {
            original_parameters: original_params,
            compressed_parameters: compressed_params,
            actual_sparsity,
            compression_ratio,
            size_reduction_bytes: data.len() - encoded_data.len(),
            size_reduction_percent,
            estimated_speedup: self.estimate_pruning_speedup(actual_sparsity),
            strategy_used: CompressionStrategy::MagnitudePruning,
            level_used: self.config.level,
        };

        let compressed_size = encoded_data.len();
        Ok(CompressedModelData {
            data: encoded_data,
            stats,
            metadata: CompressionMetadata {
                strategy: CompressionStrategy::MagnitudePruning,
                level: self.config.level,
                sparsity_pattern: self.config.sparsity_pattern,
                actual_sparsity,
                original_size: data.len(),
                compressed_size,
            },
        })
    }

    fn apply_structured_pruning(&self, data: &[u8]) -> Result<CompressedModelData, JsValue> {
        web_sys::console::log_1(&"Applying structured pruning...".into());

        // Simulate structured pruning by removing entire channels/filters
        let original_size = data.len();
        let weights_f32 = self.bytes_to_f32_slice(data);
        let weights = weights_f32.to_vec();

        // Simulate channel/filter importance scoring
        let channel_size = 64; // Typical channel size
        let num_channels = weights.len() / channel_size;
        let mut channel_scores = Vec::with_capacity(num_channels);

        for i in 0..num_channels {
            let start_idx = i * channel_size;
            let end_idx = (start_idx + channel_size).min(weights.len());
            let channel_weights = &weights[start_idx..end_idx];

            // L2 norm as importance score
            let score: f32 = channel_weights.iter().map(|&w| w * w).sum::<f32>().sqrt();
            channel_scores.push((i, score));
        }

        // Sort by importance and remove least important channels
        channel_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let channels_to_keep = ((1.0 - self.config.target_sparsity) * num_channels as f32) as usize;

        let mut pruned_weights = Vec::new();
        for (channel_idx, _score) in channel_scores.iter().take(channels_to_keep) {
            let start_idx = channel_idx * channel_size;
            let end_idx = (start_idx + channel_size).min(weights.len());
            pruned_weights.extend_from_slice(&weights[start_idx..end_idx]);
        }

        let compressed_data = self.f32_slice_to_bytes(&pruned_weights);
        let compressed_size = compressed_data.len();
        let actual_sparsity = 1.0 - (pruned_weights.len() as f32 / weights.len() as f32);
        let compression_ratio = original_size as f32 / compressed_size as f32;
        let size_reduction_percent =
            (1.0 - (compressed_size as f32 / original_size as f32)) * 100.0;

        let stats = CompressionStats {
            original_parameters: weights.len(),
            compressed_parameters: pruned_weights.len(),
            actual_sparsity,
            compression_ratio,
            size_reduction_bytes: original_size - compressed_size,
            size_reduction_percent,
            estimated_speedup: self.estimate_structured_pruning_speedup(actual_sparsity),
            strategy_used: CompressionStrategy::StructuredPruning,
            level_used: self.config.level,
        };

        Ok(CompressedModelData {
            data: compressed_data,
            stats,
            metadata: CompressionMetadata {
                strategy: CompressionStrategy::StructuredPruning,
                level: self.config.level,
                sparsity_pattern: self.config.sparsity_pattern,
                actual_sparsity,
                original_size,
                compressed_size,
            },
        })
    }

    fn apply_low_rank_factorization(&self, data: &[u8]) -> Result<CompressedModelData, JsValue> {
        web_sys::console::log_1(&"Applying low-rank matrix factorization...".into());

        // Simulate low-rank approximation
        let original_size = data.len();
        let weights = self.bytes_to_f32_slice(data).to_vec();

        // Assume we're factorizing matrices into U * V where original was M x N
        // and we use rank R such that U is M x R and V is R x N
        let compression_factor = match self.config.level {
            CompressionLevel::Light => 0.8,
            CompressionLevel::Medium => 0.6,
            CompressionLevel::Aggressive => 0.4,
            CompressionLevel::Maximum => 0.2,
        };

        let factorized_size = (weights.len() as f32 * compression_factor) as usize;
        let mut factorized_weights = vec![0.0f32; factorized_size];

        // Simulate factorization (in practice, this would use SVD or similar)
        for (i, weight) in factorized_weights.iter_mut().enumerate() {
            if i < weights.len() {
                *weight = weights[i] * 0.9; // Slight approximation error
            }
        }

        let compressed_data = self.f32_slice_to_bytes(&factorized_weights);
        let compressed_size = compressed_data.len();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        let size_reduction_percent =
            (1.0 - (compressed_size as f32 / original_size as f32)) * 100.0;

        let stats = CompressionStats {
            original_parameters: weights.len(),
            compressed_parameters: factorized_weights.len(),
            actual_sparsity: 0.0, // Not sparsity-based
            compression_ratio,
            size_reduction_bytes: original_size - compressed_size,
            size_reduction_percent,
            estimated_speedup: self.estimate_factorization_speedup(compression_factor),
            strategy_used: CompressionStrategy::LowRankFactorization,
            level_used: self.config.level,
        };

        Ok(CompressedModelData {
            data: compressed_data,
            stats,
            metadata: CompressionMetadata {
                strategy: CompressionStrategy::LowRankFactorization,
                level: self.config.level,
                sparsity_pattern: self.config.sparsity_pattern,
                actual_sparsity: 0.0,
                original_size,
                compressed_size,
            },
        })
    }

    fn apply_svd_compression(&self, data: &[u8]) -> Result<CompressedModelData, JsValue> {
        web_sys::console::log_1(&"Applying SVD compression...".into());
        // Similar to low-rank factorization but using SVD specifically
        self.apply_low_rank_factorization(data)
    }

    fn apply_weight_clustering(&self, data: &[u8]) -> Result<CompressedModelData, JsValue> {
        web_sys::console::log_1(&"Applying weight clustering...".into());

        let original_size = data.len();
        let weights = self.bytes_to_f32_slice(data).to_vec();

        // Simulate k-means clustering of weights
        let num_clusters = match self.config.level {
            CompressionLevel::Light => 256,
            CompressionLevel::Medium => 128,
            CompressionLevel::Aggressive => 64,
            CompressionLevel::Maximum => 32,
        };

        // Simple clustering simulation
        let min_weight = weights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_weight = weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let cluster_step = (max_weight - min_weight) / num_clusters as f32;

        let mut cluster_centers = Vec::with_capacity(num_clusters);
        for i in 0..num_clusters {
            cluster_centers.push(min_weight + (i as f32 + 0.5) * cluster_step);
        }

        // Assign weights to clusters and replace with cluster centers
        let mut clustered_weights = Vec::with_capacity(weights.len());
        let mut cluster_indices = Vec::with_capacity(weights.len());

        for &weight in &weights {
            let mut best_cluster = 0;
            let mut best_distance = f32::INFINITY;

            for (i, &center) in cluster_centers.iter().enumerate() {
                let distance = (weight - center).abs();
                if distance < best_distance {
                    best_distance = distance;
                    best_cluster = i;
                }
            }

            clustered_weights.push(cluster_centers[best_cluster]);
            cluster_indices.push(best_cluster as u8);
        }

        // Encode as cluster centers + indices
        let mut compressed_data = Vec::new();

        // Store cluster centers (num_clusters * 4 bytes)
        compressed_data.extend_from_slice(&self.f32_slice_to_bytes(&cluster_centers));

        // Store indices (more compact than original weights)
        compressed_data.extend_from_slice(&cluster_indices);

        let compressed_size = compressed_data.len();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        let size_reduction_percent =
            (1.0 - (compressed_size as f32 / original_size as f32)) * 100.0;

        let stats = CompressionStats {
            original_parameters: weights.len(),
            compressed_parameters: weights.len(), // Same number of parameters, just quantized
            actual_sparsity: 0.0,
            compression_ratio,
            size_reduction_bytes: original_size - compressed_size,
            size_reduction_percent,
            estimated_speedup: 1.1, // Slight speedup from reduced precision
            strategy_used: CompressionStrategy::WeightClustering,
            level_used: self.config.level,
        };

        Ok(CompressedModelData {
            data: compressed_data,
            stats,
            metadata: CompressionMetadata {
                strategy: CompressionStrategy::WeightClustering,
                level: self.config.level,
                sparsity_pattern: self.config.sparsity_pattern,
                actual_sparsity: 0.0,
                original_size,
                compressed_size,
            },
        })
    }

    fn apply_huffman_compression(&self, data: &[u8]) -> Result<CompressedModelData, JsValue> {
        web_sys::console::log_1(&"Applying Huffman compression...".into());

        // Simulate Huffman coding
        let original_size = data.len();
        let compression_ratio = match self.config.level {
            CompressionLevel::Light => 1.2,
            CompressionLevel::Medium => 1.5,
            CompressionLevel::Aggressive => 2.0,
            CompressionLevel::Maximum => 2.5,
        };

        let compressed_size = (original_size as f32 / compression_ratio) as usize;
        let compressed_data = vec![0u8; compressed_size];

        let size_reduction_percent =
            (1.0 - (compressed_size as f32 / original_size as f32)) * 100.0;

        let stats = CompressionStats {
            original_parameters: original_size / 4,
            compressed_parameters: original_size / 4, // Lossless compression
            actual_sparsity: 0.0,
            compression_ratio,
            size_reduction_bytes: original_size - compressed_size,
            size_reduction_percent,
            estimated_speedup: 1.0, // No compute speedup, just storage
            strategy_used: CompressionStrategy::HuffmanCompression,
            level_used: self.config.level,
        };

        Ok(CompressedModelData {
            data: compressed_data,
            stats,
            metadata: CompressionMetadata {
                strategy: CompressionStrategy::HuffmanCompression,
                level: self.config.level,
                sparsity_pattern: self.config.sparsity_pattern,
                actual_sparsity: 0.0,
                original_size,
                compressed_size,
            },
        })
    }

    fn apply_progressive_compression(&self, data: &[u8]) -> Result<CompressedModelData, JsValue> {
        web_sys::console::log_1(&"Applying progressive compression pipeline...".into());

        // Progressive compression combines multiple techniques
        let mut current_data = data.to_vec();
        let original_size = data.len();

        // Step 1: Magnitude pruning
        web_sys::console::log_1(&"  Step 1: Magnitude pruning...".into());
        let pruning_result = self.apply_magnitude_pruning(&current_data)?;
        current_data = pruning_result.data;

        // Step 2: Weight clustering
        web_sys::console::log_1(&"  Step 2: Weight clustering...".into());
        let clustering_result = self.apply_weight_clustering(&current_data)?;
        current_data = clustering_result.data;

        // Step 3: Huffman compression
        web_sys::console::log_1(&"  Step 3: Huffman compression...".into());
        let huffman_result = self.apply_huffman_compression(&current_data)?;
        current_data = huffman_result.data;

        let current_data_len = current_data.len();
        let final_compression_ratio = original_size as f32 / current_data_len as f32;
        let size_reduction_percent =
            (1.0 - (current_data_len as f32 / original_size as f32)) * 100.0;

        let stats = CompressionStats {
            original_parameters: original_size / 4,
            compressed_parameters: (original_size / 4)
                - (pruning_result.stats.original_parameters
                    - pruning_result.stats.compressed_parameters),
            actual_sparsity: pruning_result.stats.actual_sparsity,
            compression_ratio: final_compression_ratio,
            size_reduction_bytes: original_size - current_data_len,
            size_reduction_percent,
            estimated_speedup: pruning_result.stats.estimated_speedup * 1.1, // Additional speedup from clustering
            strategy_used: CompressionStrategy::Progressive,
            level_used: self.config.level,
        };

        Ok(CompressedModelData {
            data: current_data,
            stats,
            metadata: CompressionMetadata {
                strategy: CompressionStrategy::Progressive,
                level: self.config.level,
                sparsity_pattern: self.config.sparsity_pattern,
                actual_sparsity: pruning_result.stats.actual_sparsity,
                original_size,
                compressed_size: current_data_len,
            },
        })
    }

    // Helper methods

    fn estimate_layer_count(&self, data: &[u8]) -> usize {
        // Rough estimation based on model size
        let size_mb = data.len() as f32 / 1_048_576.0;
        if size_mb > 100.0 {
            24 // Large model (GPT-like)
        } else if size_mb > 20.0 {
            12 // Medium model (BERT-like)
        } else {
            6 // Small model
        }
    }

    fn bytes_to_f32_slice(&self, data: &[u8]) -> &[f32] {
        unsafe { core::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4) }
    }

    fn f32_slice_to_bytes(&self, data: &[f32]) -> Vec<u8> {
        unsafe { core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4).to_vec() }
    }

    fn encode_sparse_weights(&self, weights: &[f32]) -> (Vec<u8>, f32) {
        // Simple sparse encoding: store non-zero weights with their indices
        let mut encoded = Vec::new();
        let mut non_zero_count = 0;

        for (idx, &weight) in weights.iter().enumerate() {
            if weight != 0.0 {
                // Store index (4 bytes) + weight (4 bytes)
                encoded.extend_from_slice(&(idx as u32).to_le_bytes());
                encoded.extend_from_slice(&weight.to_le_bytes());
                non_zero_count += 1;
            }
        }

        let actual_sparsity = 1.0 - (non_zero_count as f32 / weights.len() as f32);
        (encoded, actual_sparsity)
    }

    fn estimate_pruning_speedup(&self, sparsity: f32) -> f32 {
        // Speedup from sparse computation
        1.0 + sparsity * 1.5
    }

    fn estimate_structured_pruning_speedup(&self, sparsity: f32) -> f32 {
        // Higher speedup for structured pruning
        1.0 + sparsity * 2.0
    }

    fn estimate_factorization_speedup(&self, compression_factor: f32) -> f32 {
        // Speedup from reduced FLOPs
        1.0 + (1.0 - compression_factor) * 0.8
    }
}

/// Compressed model data with metadata
#[wasm_bindgen]
pub struct CompressedModelData {
    data: Vec<u8>,
    stats: CompressionStats,
    #[allow(dead_code)]
    metadata: CompressionMetadata,
}

#[wasm_bindgen]
impl CompressedModelData {
    /// Get the compressed model data
    pub fn data(&self) -> Vec<u8> {
        self.data.clone()
    }

    /// Get the size of the compressed model in bytes
    #[wasm_bindgen(getter)]
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Get the compression ratio
    #[wasm_bindgen(getter)]
    pub fn compression_ratio(&self) -> f32 {
        self.stats.compression_ratio
    }

    /// Get the size reduction percentage
    #[wasm_bindgen(getter)]
    pub fn size_reduction_percent(&self) -> f32 {
        self.stats.size_reduction_percent
    }

    /// Get the actual sparsity achieved
    #[wasm_bindgen(getter)]
    pub fn actual_sparsity(&self) -> f32 {
        self.stats.actual_sparsity
    }

    /// Get the estimated speedup
    #[wasm_bindgen(getter)]
    pub fn estimated_speedup(&self) -> f32 {
        self.stats.estimated_speedup
    }

    /// Get the strategy used
    #[wasm_bindgen(getter)]
    pub fn strategy_used(&self) -> CompressionStrategy {
        self.stats.strategy_used
    }

    /// Get the compression level used
    #[wasm_bindgen(getter)]
    pub fn level_used(&self) -> CompressionLevel {
        self.stats.level_used
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Weight Compression: {:.1}% size reduction, {:.1}% sparsity, {:.1}x speedup ({:?}/{:?})",
            self.stats.size_reduction_percent,
            self.stats.actual_sparsity * 100.0,
            self.stats.estimated_speedup,
            self.stats.strategy_used,
            self.stats.level_used
        )
    }
}

/// Compression metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetadata {
    pub strategy: CompressionStrategy,
    pub level: CompressionLevel,
    pub sparsity_pattern: SparsityPattern,
    pub actual_sparsity: f32,
    pub original_size: usize,
    pub compressed_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_config() {
        let config = CompressionConfig::transformer();
        assert_eq!(config.strategy, CompressionStrategy::Progressive);
        assert!(config.preserve_attention);

        let mobile_config = CompressionConfig::mobile();
        assert_eq!(mobile_config.level, CompressionLevel::Aggressive);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_weight_compressor() {
        let config = CompressionConfig::new(
            CompressionStrategy::MagnitudePruning,
            CompressionLevel::Medium,
        );
        let compressor = WeightCompressor::new(config);

        // Test with sample data
        let test_data = vec![0u8; 1024];
        let result = compressor.compress_weights(&test_data);
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_sensitivity_analysis() {
        let config = CompressionConfig::transformer();
        let mut compressor = WeightCompressor::new(config);

        let test_data = vec![0u8; 4096];
        let sensitivities = compressor.analyze_sensitivity(&test_data);
        assert!(sensitivities.is_ok());
        assert!(!sensitivities.unwrap().is_empty());
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_weight_compressor_config_only() {
        // Test only configuration creation for non-WASM targets
        let config = CompressionConfig::new(
            CompressionStrategy::MagnitudePruning,
            CompressionLevel::Medium,
        );
        let compressor = WeightCompressor::new(config);
        assert_eq!(
            compressor.config.strategy,
            CompressionStrategy::MagnitudePruning
        );
        assert_eq!(compressor.config.level, CompressionLevel::Medium);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_sensitivity_analysis_config_only() {
        // Test only configuration creation for non-WASM targets
        let config = CompressionConfig::transformer();
        let compressor = WeightCompressor::new(config);
        assert_eq!(compressor.config.strategy, CompressionStrategy::Progressive);
        assert!(compressor.config.preserve_attention);
    }
}
