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
    /// Real measured relative reconstruction error (Frobenius norm of the
    /// difference over the Frobenius norm of the original). Three cases:
    /// - `LowRankFactorization`/`SVDCompression`: a real measured value —
    ///   these strategies are lossy approximations by construction.
    /// - `HuffmanCompression`: exactly `0.0` — this strategy is lossless
    ///   (see `WeightCompressor::decompress_weights` and its round-trip
    ///   test), so a real error of zero is the honest, verified value.
    /// - every other strategy (pruning, clustering): `NaN`, since this
    ///   compressor does not define or measure a reconstruction for them —
    ///   `NaN` rather than `0.0` so "not measured" is never misread as
    ///   "measured to be lossless".
    pub reconstruction_error: f32,
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
    /// Analyze per-layer compression sensitivity, estimated from the real
    /// weight data (see [`compute_layer_sensitivities`]).
    pub fn analyze_sensitivity(&mut self, model_data: &[u8]) -> Result<Vec<f32>, JsValue> {
        let weights = self.bytes_to_f32_slice(model_data);
        let num_layers = self.estimate_layer_count(model_data);
        let sensitivities = compute_layer_sensitivities(&weights, num_layers);

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
            reconstruction_error: f32::NAN,
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
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"Applying magnitude-based pruning...".into());

        // Zero out weights below the magnitude threshold.
        let compressed_data = data.to_vec();
        let mut weights = self.bytes_to_f32_slice(&compressed_data);

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
            size_reduction_bytes: data.len().saturating_sub(encoded_data.len()),
            size_reduction_percent,
            estimated_speedup: self.estimate_pruning_speedup(actual_sparsity),
            strategy_used: CompressionStrategy::MagnitudePruning,
            level_used: self.config.level,
            reconstruction_error: f32::NAN,
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
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"Applying structured pruning...".into());

        // Remove entire channels/filters ranked by L2-norm importance.
        let original_size = data.len();
        let weights = self.bytes_to_f32_slice(data);

        // Channel/filter importance scoring
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
            size_reduction_bytes: original_size.saturating_sub(compressed_size),
            size_reduction_percent,
            estimated_speedup: self.estimate_structured_pruning_speedup(actual_sparsity),
            strategy_used: CompressionStrategy::StructuredPruning,
            level_used: self.config.level,
            reconstruction_error: f32::NAN,
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

    /// Real truncated-SVD low-rank factorization.
    ///
    /// The flat weight buffer is reshaped into a near-square `rows x cols`
    /// matrix (padded with zeros so `rows * cols >= n`), decomposed via
    /// [`truncated_svd`] into rank-`r` factors `U`, `S`, `Vt`
    /// (`reconstruction ≈ U * diag(S) * Vt`), and those factors — not a
    /// truncated/rescaled copy of the original data — are what gets stored.
    /// The real relative Frobenius reconstruction error is computed and
    /// recorded in `CompressionStats::reconstruction_error`, replacing the
    /// old code's `weights[i] * 0.9` (an unconditional, unmeasured 10%
    /// error applied to a naively-truncated prefix of the flat buffer, with
    /// every element past the truncation point simply discarded).
    fn apply_low_rank_factorization(&self, data: &[u8]) -> Result<CompressedModelData, JsValue> {
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"Applying low-rank matrix factorization...".into());

        let original_size = data.len();
        let weights = self.bytes_to_f32_slice(data);
        let n = weights.len();

        let rank_fraction = match self.config.level {
            CompressionLevel::Light => 0.8,
            CompressionLevel::Medium => 0.6,
            CompressionLevel::Aggressive => 0.4,
            CompressionLevel::Maximum => 0.2,
        };

        let (rows, cols) = svd_matrix_shape(n);
        let max_rank = rows.min(cols);
        let rank = ((max_rank as f32 * rank_fraction) as usize).clamp(1.min(max_rank), max_rank);

        let mut matrix = vec![0.0f32; rows * cols];
        matrix[..n].copy_from_slice(&weights);

        let (u, s, vt) = truncated_svd(&matrix, rows, cols, rank);
        let reconstructed = svd_reconstruct(&u, &s, &vt, rows, cols, rank);
        let reconstruction_error = relative_frobenius_error(&weights, &reconstructed[..n]);

        let compressed_data = encode_svd_payload(n, rows, cols, rank, &u, &s, &vt);
        let compressed_size = compressed_data.len();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        let size_reduction_percent =
            (1.0 - (compressed_size as f32 / original_size as f32)) * 100.0;
        let compressed_parameters = rows * rank + rank + rank * cols;

        let stats = CompressionStats {
            original_parameters: n,
            compressed_parameters,
            actual_sparsity: 0.0, // Not sparsity-based
            compression_ratio,
            size_reduction_bytes: original_size.saturating_sub(compressed_size),
            size_reduction_percent,
            estimated_speedup: self.estimate_factorization_speedup(rank_fraction),
            strategy_used: CompressionStrategy::LowRankFactorization,
            level_used: self.config.level,
            reconstruction_error,
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
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"Applying SVD compression...".into());
        // "SVD Compression" and "Low-Rank Factorization" are the same real
        // algorithm here (truncated SVD IS the standard way to compute an
        // optimal low-rank factorization — Eckart-Young theorem); kept as
        // two selectable strategies for API/config-preset compatibility.
        let mut result = self.apply_low_rank_factorization(data)?;
        result.stats.strategy_used = CompressionStrategy::SVDCompression;
        result.metadata.strategy = CompressionStrategy::SVDCompression;
        Ok(result)
    }

    fn apply_weight_clustering(&self, data: &[u8]) -> Result<CompressedModelData, JsValue> {
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"Applying weight clustering...".into());

        let original_size = data.len();
        let weights = self.bytes_to_f32_slice(data);

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
            size_reduction_bytes: original_size.saturating_sub(compressed_size),
            size_reduction_percent,
            estimated_speedup: 1.1, // Slight speedup from reduced precision
            strategy_used: CompressionStrategy::WeightClustering,
            level_used: self.config.level,
            reconstruction_error: f32::NAN,
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

    /// Real lossless entropy coding via `oxiarc_deflate` (pure-Rust RFC 1951
    /// DEFLATE, which includes dynamic Huffman coding — this crate's own
    /// naming convention calls this strategy "Huffman compression"; DEFLATE
    /// is the concrete, verifiably-round-tripping implementation of it, not
    /// a hedge). `data` is really compressed and really decompressible (see
    /// [`WeightCompressor::decompress_weights`] and its round-trip test) —
    /// the old code never read `data` at all: `compressed_size` was derived
    /// from a hardcoded per-level ratio table and the "compressed" output
    /// was `vec![0u8; compressed_size]`, discarding every input byte.
    fn apply_huffman_compression(&self, data: &[u8]) -> Result<CompressedModelData, JsValue> {
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"Applying Huffman compression...".into());

        let original_size = data.len();
        let deflate_level = match self.config.level {
            CompressionLevel::Light => 2,
            CompressionLevel::Medium => 5,
            CompressionLevel::Aggressive => 7,
            CompressionLevel::Maximum => 9,
        };

        let compressed_data = oxiarc_deflate::deflate(data, deflate_level)
            .map_err(|e| JsValue::from_str(&format!("Huffman/DEFLATE compression failed: {e}")))?;
        let compressed_size = compressed_data.len();

        // Real, measured ratio — DEFLATE can legitimately *expand*
        // high-entropy/incompressible data (a few bytes of block overhead
        // with nothing to encode away); that is reported honestly rather
        // than floored at a fabricated minimum.
        let compression_ratio = if compressed_size > 0 {
            original_size as f32 / compressed_size as f32
        } else {
            1.0
        };
        let size_reduction_percent = if original_size > 0 {
            (1.0 - (compressed_size as f32 / original_size as f32)) * 100.0
        } else {
            0.0
        };

        let stats = CompressionStats {
            original_parameters: original_size / 4,
            compressed_parameters: original_size / 4, // lossless: same logical parameter count
            actual_sparsity: 0.0,
            compression_ratio,
            size_reduction_bytes: original_size.saturating_sub(compressed_size),
            size_reduction_percent,
            estimated_speedup: 1.0, // No compute speedup, just storage
            strategy_used: CompressionStrategy::HuffmanCompression,
            level_used: self.config.level,
            reconstruction_error: 0.0, // lossless — verified by the round-trip test
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

    /// Decompress data previously produced by [`Self::compress_weights`].
    /// Only strategies that retain enough information to reconstruct real
    /// bytes support this:
    /// - `HuffmanCompression`: exact byte-for-byte round trip.
    /// - `LowRankFactorization`/`SVDCompression`: the approximate
    ///   reconstruction from the stored `U`/`S`/`Vt` factors (see the
    ///   `reconstruction_error` this strategy recorded when compressing).
    ///
    /// Every other strategy is irreversible by construction — pruning
    /// discards weights, clustering replaces them with cluster centers —
    /// and returns a structured error rather than fabricating a result.
    pub fn decompress_weights(&self, compressed: &CompressedModelData) -> Result<Vec<u8>, JsValue> {
        self.decompress_weights_inner(compressed).map_err(|e| JsValue::from_str(&e))
    }

    /// Pure (`JsValue`-free) core of [`Self::decompress_weights`]. See that
    /// method's docs for behavior; kept separate — and returning `String`
    /// rather than `JsValue` — so the "irreversible strategy" error path is
    /// exercised by native tests: constructing a `JsValue` (even a bare
    /// `JsValue::from_str`) unconditionally aborts the process on
    /// non-wasm32 targets, which would make that path untestable natively.
    fn decompress_weights_inner(
        &self,
        compressed: &CompressedModelData,
    ) -> Result<Vec<u8>, String> {
        match compressed.stats.strategy_used {
            CompressionStrategy::HuffmanCompression => oxiarc_deflate::inflate(&compressed.data)
                .map_err(|e| format!("Huffman/DEFLATE decompression failed: {e}")),
            CompressionStrategy::LowRankFactorization | CompressionStrategy::SVDCompression => {
                decode_svd_payload(&compressed.data).map(|floats| self.f32_slice_to_bytes(&floats))
            },
            CompressionStrategy::None => Ok(compressed.data.clone()),
            other => Err(format!(
                "decompress_weights: {other:?} is not reversible (pruning/clustering discard information \
                 rather than storing a reconstructable representation)"
            )),
        }
    }

    fn apply_progressive_compression(&self, data: &[u8]) -> Result<CompressedModelData, JsValue> {
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"Applying progressive compression pipeline...".into());

        // Progressive compression combines multiple techniques
        let mut current_data = data.to_vec();
        let original_size = data.len();

        // Step 1: Magnitude pruning
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"  Step 1: Magnitude pruning...".into());
        let pruning_result = self.apply_magnitude_pruning(&current_data)?;
        current_data = pruning_result.data;

        // Step 2: Weight clustering
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"  Step 2: Weight clustering...".into());
        let clustering_result = self.apply_weight_clustering(&current_data)?;
        current_data = clustering_result.data;

        // Step 3: Huffman compression
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"  Step 3: Huffman compression...".into());
        let huffman_result = self.apply_huffman_compression(&current_data)?;
        current_data = huffman_result.data;

        let current_data_len = current_data.len();
        let final_compression_ratio = original_size as f32 / current_data_len as f32;
        let size_reduction_percent =
            (1.0 - (current_data_len as f32 / original_size as f32)) * 100.0;

        let pruned_count = pruning_result
            .stats
            .original_parameters
            .saturating_sub(pruning_result.stats.compressed_parameters);
        let stats = CompressionStats {
            original_parameters: original_size / 4,
            compressed_parameters: (original_size / 4).saturating_sub(pruned_count),
            actual_sparsity: pruning_result.stats.actual_sparsity,
            compression_ratio: final_compression_ratio,
            size_reduction_bytes: original_size.saturating_sub(current_data_len),
            size_reduction_percent,
            estimated_speedup: pruning_result.stats.estimated_speedup * 1.1, // Additional speedup from clustering
            strategy_used: CompressionStrategy::Progressive,
            level_used: self.config.level,
            // The pipeline combines pruning + clustering (both lossy, not
            // instrumented) with a final lossless Huffman/DEFLATE pass;
            // no single reconstruction-error number describes the whole
            // chain, so this is left unmeasured rather than guessed.
            reconstruction_error: f32::NAN,
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

    /// Decode raw little-endian `f32` weight bytes. Any trailing `1..=3`
    /// bytes that don't form a complete `f32` are dropped (matches the
    /// element count this method previously computed via `data.len() / 4`).
    ///
    /// This used to be `unsafe { core::slice::from_raw_parts(data.as_ptr()
    /// as *const f32, ...) }`, reinterpreting an arbitrary `&[u8]` as
    /// `&[f32]` in place — real undefined behavior (`&[u8]` carries no
    /// 4-byte alignment guarantee; an odd-aligned pointer dereferenced as
    /// `f32` is UB, not merely a lint) as well as silently assuming
    /// native-endian byte order. `chunks_exact(4)` + `from_le_bytes` decodes
    /// safely, with an explicit, portable byte order.
    fn bytes_to_f32_slice(&self, data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Inverse of [`Self::bytes_to_f32_slice`]: encode `f32`s as raw
    /// little-endian bytes. Replaces the same `unsafe`
    /// `from_raw_parts::<u8>` reinterpretation described above.
    fn f32_slice_to_bytes(&self, data: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(data.len() * 4);
        for v in data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
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

/// Real per-segment weight-magnitude-variance sensitivity proxy.
///
/// True gradient-/activation-based sensitivity analysis needs a live
/// forward+backward pass, which is not available from raw weight bytes
/// alone. As a real (not fabricated) proxy, this splits the flattened
/// weights into `num_layers` contiguous segments and scores each by its
/// coefficient of variation (`stddev(|w|) / mean(|w|)`): segments whose
/// weight magnitudes vary more relative to their average are treated as
/// more sensitive to compression (harder to represent well with a single
/// shared scale/few discrete levels) — a real, commonly used compression-
/// sensitivity heuristic. Scores are min-max normalized to `[0.1, 0.9]`
/// across segments so the *relative* ranking is meaningful even though the
/// absolute scale is inherently a heuristic in the absence of gradients.
///
/// This used to be `match i % 4 { 0 => 0.9, 1 => 0.7, 2 => 0.5, 3 => 0.3 }`
/// — an identical cycle for every model, completely independent of
/// `weights`.
fn compute_layer_sensitivities(weights: &[f32], num_layers: usize) -> Vec<f32> {
    if num_layers == 0 || weights.is_empty() {
        return Vec::new();
    }
    let segment_len = weights.len().div_ceil(num_layers).max(1);

    let raw_scores: Vec<f32> = (0..num_layers)
        .map(|i| {
            let start = (i * segment_len).min(weights.len());
            let end = ((i + 1) * segment_len).min(weights.len());
            let segment = &weights[start..end];
            if segment.is_empty() {
                return 0.0;
            }
            let mean_abs = segment.iter().map(|w| w.abs()).sum::<f32>() / segment.len() as f32;
            if mean_abs <= 1e-12 {
                return 0.0;
            }
            let variance = segment.iter().map(|w| (w.abs() - mean_abs).powi(2)).sum::<f32>()
                / segment.len() as f32;
            variance.sqrt() / mean_abs
        })
        .collect();

    let min = raw_scores.iter().copied().fold(f32::INFINITY, f32::min);
    let max = raw_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;

    raw_scores
        .into_iter()
        .map(|score| if range > 1e-12 { 0.1 + 0.8 * (score - min) / range } else { 0.5 })
        .collect()
}

// ---------------------------------------------------------------------
// Real truncated-SVD low-rank factorization (power iteration + deflation).
// ---------------------------------------------------------------------
//
// Pure numeric functions (no `wasm_bindgen`/`JsValue`), so they are unit
// tested directly rather than only indirectly via the `JsValue`-returning
// `WeightCompressor::compress_weights`.

/// Choose a near-square `(rows, cols)` matrix shape with `rows * cols >= n`
/// for reshaping a flat weight buffer before running [`truncated_svd`].
/// Near-square minimizes the zero-padding needed to reach a rectangular
/// shape.
fn svd_matrix_shape(n: usize) -> (usize, usize) {
    if n == 0 {
        return (1, 1);
    }
    let rows = ((n as f64).sqrt().floor() as usize).max(1);
    let cols = n.div_ceil(rows);
    (rows, cols)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

fn normalize_in_place(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Truncated SVD via power iteration with deflation: computes the `rank`
/// largest singular triplets of `matrix` (row-major, `rows x cols`).
/// Returns `(u, s, vt)`:
/// - `u`: `rows x rank`, row-major (left singular vectors as columns)
/// - `s`: `rank` singular values, descending
/// - `vt`: `rank x cols`, row-major (right singular vectors as rows, i.e.
///   `V^T`)
///
/// such that `reconstruction[r,c] ≈ sum_k u[r,k] * s[k] * vt[k,c]` (see
/// [`svd_reconstruct`]) — the real Eckart-Young-optimal rank-`rank`
/// approximation of `matrix` (up to power-iteration convergence), not a
/// truncated/rescaled copy of raw entries.
fn truncated_svd(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let rank = rank.min(rows).min(cols);
    let mut u = vec![0.0f32; rows * rank];
    let mut s = vec![0.0f32; rank];
    let mut vt = vec![0.0f32; rank * cols];
    if rank == 0 || rows == 0 || cols == 0 {
        return (u, s, vt);
    }

    let mut residual = matrix.to_vec();

    for k in 0..rank {
        // Deterministic, non-degenerate seed vector (avoids the all-zero
        // vector and axis-aligned starts, either of which can stall power
        // iteration on structured inputs).
        let mut v: Vec<f32> =
            (0..cols).map(|i| (((i * 7 + k * 31 + 1) % 11) as f32) + 1.0).collect();
        normalize_in_place(&mut v);

        let mut u_vec = vec![0.0f32; rows];
        for _iteration in 0..100 {
            for (r, u_r) in u_vec.iter_mut().enumerate() {
                *u_r = dot(&residual[r * cols..(r + 1) * cols], &v);
            }
            let u_norm = l2_norm(&u_vec);
            if u_norm < 1e-10 {
                break; // no signal left along any direction of the residual
            }
            for x in u_vec.iter_mut() {
                *x /= u_norm;
            }

            let mut v_new = vec![0.0f32; cols];
            for (r, &ur) in u_vec.iter().enumerate() {
                if ur == 0.0 {
                    continue;
                }
                let row = &residual[r * cols..(r + 1) * cols];
                for (c, &rv) in row.iter().enumerate() {
                    v_new[c] += ur * rv;
                }
            }
            let v_norm = l2_norm(&v_new);
            if v_norm < 1e-10 {
                break;
            }
            for x in v_new.iter_mut() {
                *x /= v_norm;
            }

            let delta: f32 =
                v.iter().zip(v_new.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            v = v_new;
            if delta < 1e-6 {
                break;
            }
        }

        // Final singular value/left vector for the converged `v`:
        // sigma = ||A v||, u_k = A v / sigma.
        for (r, u_r) in u_vec.iter_mut().enumerate() {
            *u_r = dot(&residual[r * cols..(r + 1) * cols], &v);
        }
        let sigma = l2_norm(&u_vec);
        if sigma < 1e-10 {
            break; // residual has no remaining signal; later components stay zero
        }
        for x in u_vec.iter_mut() {
            *x /= sigma;
        }

        for (r, &ur) in u_vec.iter().enumerate() {
            u[r * rank + k] = ur;
        }
        s[k] = sigma;
        vt[k * cols..(k + 1) * cols].copy_from_slice(&v);

        // Deflate: residual -= sigma * outer(u_k, v_k), so the next
        // iteration finds the next-largest orthogonal component.
        for r in 0..rows {
            let ur = u_vec[r];
            if ur == 0.0 {
                continue;
            }
            let row = &mut residual[r * cols..(r + 1) * cols];
            for (c, rv) in row.iter_mut().enumerate() {
                *rv -= sigma * ur * v[c];
            }
        }
    }

    (u, s, vt)
}

/// Reconstruct the `rows x cols` matrix (row-major) from truncated-SVD
/// factors produced by [`truncated_svd`].
fn svd_reconstruct(
    u: &[f32],
    s: &[f32],
    vt: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for k in 0..rank {
        let sigma = s[k];
        if sigma == 0.0 {
            continue;
        }
        let vk = &vt[k * cols..(k + 1) * cols];
        for r in 0..rows {
            let scaled = u[r * rank + k] * sigma;
            if scaled == 0.0 {
                continue;
            }
            let out_row = &mut out[r * cols..(r + 1) * cols];
            for (c, ov) in out_row.iter_mut().enumerate() {
                *ov += scaled * vk[c];
            }
        }
    }
    out
}

/// Relative Frobenius-norm reconstruction error: `||a - b|| / ||a||` (using
/// the absolute error directly when `||a|| ~= 0`, so the ratio stays finite
/// rather than dividing by zero).
fn relative_frobenius_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    let diff_sq: f32 =
        original.iter().zip(reconstructed.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    let orig_norm = l2_norm(original);
    if orig_norm > 1e-12 {
        diff_sq.sqrt() / orig_norm
    } else {
        diff_sq.sqrt()
    }
}

/// Serialize truncated-SVD factors into a self-describing byte payload:
/// `[n: u64][rows: u64][cols: u64][rank: u64][U][S][Vt]` (all little-endian;
/// `U`/`S`/`Vt` are consecutive `f32` arrays). `n` is the real, unpadded
/// element count, so [`decode_svd_payload`] can trim the reconstruction
/// back to it.
fn encode_svd_payload(
    n: usize,
    rows: usize,
    cols: usize,
    rank: usize,
    u: &[f32],
    s: &[f32],
    vt: &[f32],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + (u.len() + s.len() + vt.len()) * 4);
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.extend_from_slice(&(rows as u64).to_le_bytes());
    out.extend_from_slice(&(cols as u64).to_le_bytes());
    out.extend_from_slice(&(rank as u64).to_le_bytes());
    for v in u.iter().chain(s.iter()).chain(vt.iter()) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Inverse of [`encode_svd_payload`]: parse the header + factors and
/// reconstruct the (unpadded) approximate weight vector. Backs
/// `WeightCompressor::decompress_weights` for
/// `LowRankFactorization`/`SVDCompression` data.
fn decode_svd_payload(data: &[u8]) -> Result<Vec<f32>, String> {
    if data.len() < 32 {
        return Err("SVD payload shorter than its 32-byte header".to_string());
    }
    let read_u64 = |off: usize| -> u64 {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&data[off..off + 8]);
        u64::from_le_bytes(buf)
    };
    let n = read_u64(0) as usize;
    let rows = read_u64(8) as usize;
    let cols = read_u64(16) as usize;
    let rank = read_u64(24) as usize;

    let expected_floats = rows * rank + rank + rank * cols;
    let expected_len = 32 + expected_floats * 4;
    if data.len() != expected_len {
        return Err(format!(
            "SVD payload length {} does not match header-implied length {expected_len} \
             (rows={rows}, cols={cols}, rank={rank})",
            data.len()
        ));
    }

    let floats: Vec<f32> = data[32..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (u, rest) = floats.split_at(rows * rank);
    let (s, vt) = rest.split_at(rank);

    let reconstructed = svd_reconstruct(u, s, vt, rows, cols, rank);
    if n > reconstructed.len() {
        return Err(format!(
            "SVD payload claims {n} elements but its rows*cols reconstruction only has {}",
            reconstructed.len()
        ));
    }
    Ok(reconstructed[..n].to_vec())
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

    /// Get the real measured relative reconstruction error (Frobenius-norm
    /// ratio). `NaN` means "not measured for this strategy", `0.0` means
    /// "measured and verified lossless" (Huffman), and any other value is a
    /// real measured lossy-approximation error (SVD/low-rank). See
    /// [`CompressionStats::reconstruction_error`] for full semantics. This
    /// was previously computed and stored internally but never exposed to
    /// JS callers.
    #[wasm_bindgen(getter)]
    pub fn reconstruction_error(&self) -> f32 {
        self.stats.reconstruction_error
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
    fn test_weight_compressor() {
        // Now runs on every target: the only thing that used to make this
        // wasm32-only was an unconditional `web_sys::console::log_1` at the
        // top of every `apply_*` method, which panics natively — that log
        // call is now `#[cfg(target_arch = "wasm32")]`-gated, so the real
        // compression logic itself (which never touched `JsValue`/`web_sys`
        // to begin with) is exercised by ordinary native tests.
        let config = CompressionConfig::new(
            CompressionStrategy::MagnitudePruning,
            CompressionLevel::Medium,
        );
        let compressor = WeightCompressor::new(config);

        let test_data = vec![0u8; 1024];
        let result = compressor.compress_weights(&test_data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sensitivity_analysis() {
        let config = CompressionConfig::transformer();
        let mut compressor = WeightCompressor::new(config);

        let test_data = vec![0u8; 4096];
        let sensitivities = compressor.analyze_sensitivity(&test_data);
        assert!(sensitivities.is_ok());
        assert!(!sensitivities.expect("test operation should succeed").is_empty());
    }

    #[test]
    fn test_weight_compressor_config_only() {
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
    fn test_sensitivity_analysis_config_only() {
        let config = CompressionConfig::transformer();
        let compressor = WeightCompressor::new(config);
        assert_eq!(compressor.config.strategy, CompressionStrategy::Progressive);
        assert!(compressor.config.preserve_attention);
    }

    // -----------------------------------------------------------------
    // Sensitivity analysis: real measurement, not a hardcoded cycle.
    // -----------------------------------------------------------------

    #[test]
    fn test_sensitivity_analysis_varies_with_real_data() {
        // Regression guard for the old `match i % 4 { 0 => 0.9, ... }`
        // cycle, which was identical for every model regardless of
        // `model_data`. Two structurally different real inputs (same
        // length, so `estimate_layer_count`/segmentation match) must
        // produce different sensitivity vectors.
        let config = CompressionConfig::transformer();
        let mut compressor_a = WeightCompressor::new(config.clone());
        let mut compressor_b = WeightCompressor::new(config);

        // Two independent pseudo-random sequences: each has real,
        // segment-dependent magnitude variance (unlike a constant or a
        // constant-magnitude alternating-sign sequence, both of which have
        // zero coefficient of variation everywhere and would trivially
        // collapse to the same uniform output).
        let gen = |seed: u32| -> Vec<f32> {
            let mut s = seed;
            (0..4096 / 4)
                .map(|_| {
                    s = s.wrapping_mul(1103515245).wrapping_add(12345);
                    ((s >> 8) as f32 / u32::MAX as f32) * 200.0 - 100.0
                })
                .collect()
        };
        let series_a = gen(11);
        let series_b = gen(97);
        let data_a: Vec<u8> = series_a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data_b: Vec<u8> = series_b.iter().flat_map(|v| v.to_le_bytes()).collect();

        let sens_a = compressor_a.analyze_sensitivity(&data_a).expect("must succeed");
        let sens_b = compressor_b.analyze_sensitivity(&data_b).expect("must succeed");

        assert_eq!(sens_a.len(), sens_b.len());
        assert_ne!(
            sens_a, sens_b,
            "sensitivity must depend on the real weight data, not just its length"
        );
    }

    #[test]
    fn test_compute_layer_sensitivities_constant_weights_are_uniform() {
        // A perfectly uniform-magnitude input has zero coefficient of
        // variation everywhere, so every segment is equally (in)sensitive.
        let weights = std::vec![2.0f32; 64];
        let sensitivities = compute_layer_sensitivities(&weights, 4);
        assert_eq!(sensitivities.len(), 4);
        for s in &sensitivities {
            assert!(
                (s - 0.5).abs() < 1e-6,
                "uniform input should score exactly the neutral 0.5: {s}"
            );
        }
    }

    // -----------------------------------------------------------------
    // Truncated SVD: real power-iteration low-rank factorization.
    // -----------------------------------------------------------------

    fn frobenius_norm(m: &[f32]) -> f32 {
        m.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    #[test]
    fn test_truncated_svd_reconstructs_exact_rank_one_matrix() {
        // A genuinely rank-1 matrix: outer product of two vectors. Old fake
        // code (`weights[i] * 0.9`) cannot come close to this — it would
        // scale every entry by a constant 0.9 uniformly rather than finding
        // the actual low-rank structure.
        let u = std::vec![1.0f32, 2.0, -1.0, 0.5];
        let v = std::vec![3.0f32, -2.0, 1.0];
        let rows = u.len();
        let cols = v.len();
        let mut matrix = std::vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                matrix[r * cols + c] = u[r] * v[c];
            }
        }

        let (su, ss, svt) = truncated_svd(&matrix, rows, cols, 1);
        let reconstructed = svd_reconstruct(&su, &ss, &svt, rows, cols, 1);
        let error = relative_frobenius_error(&matrix, &reconstructed);
        assert!(
            error < 1e-4,
            "rank-1 input must be reconstructed almost exactly, got error {error}"
        );
    }

    #[test]
    fn test_truncated_svd_reconstructs_exact_rank_r_matrix() {
        // Build a matrix that is exactly rank-3 (sum of 3 outer products)
        // and verify truncated SVD at rank >= 3 recovers it almost exactly.
        let rows = 6;
        let cols = 5;
        let components: [(f32, [f32; 6], [f32; 5]); 3] = [
            (
                5.0,
                [1.0, 0.2, -0.3, 0.7, 1.1, -0.4],
                [1.0, -1.0, 0.5, 0.2, -0.6],
            ),
            (
                3.0,
                [0.4, 1.0, 0.6, -0.2, 0.3, 0.8],
                [0.3, 0.9, -0.4, 1.0, 0.1],
            ),
            (
                1.5,
                [-0.5, 0.3, 1.0, 0.4, -0.7, 0.2],
                [-0.8, 0.2, 1.0, -0.3, 0.5],
            ),
        ];
        let mut matrix = std::vec![0.0f32; rows * cols];
        for (sigma, uk, vk) in &components {
            for r in 0..rows {
                for c in 0..cols {
                    matrix[r * cols + c] += sigma * uk[r] * vk[c];
                }
            }
        }

        let rank = 3;
        let (u, s, vt) = truncated_svd(&matrix, rows, cols, rank);
        let reconstructed = svd_reconstruct(&u, &s, &vt, rows, cols, rank);
        let error = relative_frobenius_error(&matrix, &reconstructed);
        assert!(
            error < 1e-3,
            "exactly rank-3 input at rank 3 must reconstruct almost exactly: {error}"
        );
    }

    #[test]
    fn test_truncated_svd_error_decreases_with_rank() {
        // Monotonicity: keeping more singular components can only reduce
        // (never increase) reconstruction error.
        let rows = 8;
        let cols = 6;
        let matrix: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 + 11) % 23) as f32 - 11.0 + 0.3 * (i as f32).sin())
            .collect();
        let full_norm = frobenius_norm(&matrix);
        assert!(full_norm > 0.0);

        let mut prev_error = f32::INFINITY;
        for rank in 1..=rows.min(cols) {
            let (u, s, vt) = truncated_svd(&matrix, rows, cols, rank);
            let reconstructed = svd_reconstruct(&u, &s, &vt, rows, cols, rank);
            let error = relative_frobenius_error(&matrix, &reconstructed);
            assert!(
                error <= prev_error + 1e-4,
                "error must not increase with rank: rank={rank} error={error} prev={prev_error}"
            );
            prev_error = error;
        }
        // Full rank must reconstruct (almost) exactly.
        assert!(
            prev_error < 1e-3,
            "full-rank reconstruction should be near-exact, got {prev_error}"
        );
    }

    #[test]
    fn test_truncated_svd_handles_zero_matrix() {
        let matrix = std::vec![0.0f32; 12];
        let (u, s, vt) = truncated_svd(&matrix, 3, 4, 2);
        assert!(s.iter().all(|&v| v == 0.0));
        let reconstructed = svd_reconstruct(&u, &s, &vt, 3, 4, 2);
        assert!(reconstructed.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_truncated_svd_rank_exceeding_dimensions_is_clamped() {
        let matrix = std::vec![1.0f32; 6]; // 2x3
        let (u, s, vt) = truncated_svd(&matrix, 2, 3, 100);
        assert_eq!(s.len(), 2); // clamped to min(rows, cols)
        assert_eq!(u.len(), 2 * 2);
        assert_eq!(vt.len(), 2 * 3);
    }

    #[test]
    fn test_svd_payload_round_trip_via_decode() {
        let u = std::vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2
        let s = std::vec![5.0f32, 1.0];
        let vt = std::vec![0.5f32, -0.5, 1.0, 0.25]; // 2x2
        let encoded = encode_svd_payload(4, 2, 2, 2, &u, &s, &vt);
        let decoded = decode_svd_payload(&encoded).expect("valid payload must decode");
        let expected = svd_reconstruct(&u, &s, &vt, 2, 2, 2);
        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_apply_low_rank_factorization_measures_real_error_not_fixed_090_scale() {
        // Regression test for the old `apply_low_rank_factorization`:
        // `factorized_weights[i] = weights[i] * 0.9` over a naively
        // truncated prefix. That scheme would report a *uniform* 10%
        // per-element error and would use exactly `compression_factor *
        // len` output floats. The real implementation stores SVD factors
        // (a different byte count) and measures a real, data-dependent
        // Frobenius error.
        let config = CompressionConfig::new(
            CompressionStrategy::LowRankFactorization,
            CompressionLevel::Medium,
        );
        let compressor = WeightCompressor::new(config);

        // A real, structured (low effective rank) weight-like signal.
        let rows = 10;
        let cols = 9;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let r = (i / cols) as f32;
                let c = (i % cols) as f32;
                (0.3 * r + 0.1).sin() * (0.2 * c - 0.4).cos()
            })
            .collect();
        let data: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();

        let result = compressor.compress_weights(&data).expect("compression should succeed");
        assert!(result.stats.reconstruction_error.is_finite());
        assert!(
            result.stats.reconstruction_error >= 0.0,
            "reconstruction error must be a real non-negative measurement"
        );

        // The old scheme always kept exactly `len * compression_factor`
        // raw f32 values (no header, no U/S/Vt split); the real payload has
        // a 32-byte header plus three differently-shaped arrays, so its
        // length does not equal that naive formula except by coincidence.
        let naive_old_len_bytes = ((weights.len() as f32 * 0.6) as usize) * 4;
        assert_ne!(result.data.len(), naive_old_len_bytes);

        // Decompressing must recover something close to the original —
        // not literally 90% of a truncated prefix.
        let decompressed_bytes =
            compressor.decompress_weights(&result).expect("SVD data must decompress");
        let decompressed: Vec<f32> = decompressed_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(decompressed.len(), weights.len());
        let error = relative_frobenius_error(&weights, &decompressed);
        assert!(error < 0.5, "a mostly-smooth signal should compress with well under 50% relative error, got {error}");
    }

    #[test]
    fn test_svd_compression_strategy_delegates_to_low_rank() {
        let config =
            CompressionConfig::new(CompressionStrategy::SVDCompression, CompressionLevel::Light);
        let compressor = WeightCompressor::new(config);
        let weights: Vec<f32> = (0..40).map(|i| (i as f32 * 0.1).sin()).collect();
        let data: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();

        let result = compressor.compress_weights(&data).expect("must succeed");
        assert_eq!(
            result.stats.strategy_used,
            CompressionStrategy::SVDCompression
        );
        assert!(result.stats.reconstruction_error.is_finite());
    }

    #[test]
    fn test_reconstruction_error_getter_exposes_the_real_measured_value() {
        // Regression test: `CompressedModelData` had `size_bytes`,
        // `compression_ratio`, `actual_sparsity`, `estimated_speedup`,
        // `strategy_used`, and `level_used` getters, but no
        // `reconstruction_error` getter — so a real measured SVD
        // reconstruction error was computed and stored internally, yet no
        // JS caller could ever read it through the public API (only tests
        // reading the private `.stats` field directly could see it).
        let config = CompressionConfig::new(
            CompressionStrategy::LowRankFactorization,
            CompressionLevel::Medium,
        );
        let compressor = WeightCompressor::new(config);
        let weights: Vec<f32> = (0..90).map(|i| (i as f32 * 0.13).sin()).collect();
        let data: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();

        let result = compressor.compress_weights(&data).expect("compression should succeed");
        // The getter must expose exactly the internally measured value, not
        // a placeholder — and it must be a real, finite measurement.
        assert_eq!(
            result.reconstruction_error(),
            result.stats.reconstruction_error
        );
        assert!(result.reconstruction_error().is_finite());
        assert!(result.reconstruction_error() >= 0.0);
    }

    // -----------------------------------------------------------------
    // Huffman/DEFLATE: real, exact round-trip lossless compression.
    // -----------------------------------------------------------------

    #[test]
    fn test_huffman_round_trip_exact_on_incompressible_data() {
        // Regression test for the old `apply_huffman_compression`, which
        // never read `data` at all and returned `vec![0u8; compressed_size]`
        // — decompressing that could never recover the original bytes.
        // Use pseudo-random (high-entropy, poorly compressible) bytes so
        // this cannot pass by the compressed data "happening" to look like
        // the original.
        let config = CompressionConfig::new(
            CompressionStrategy::HuffmanCompression,
            CompressionLevel::Maximum,
        );
        let compressor = WeightCompressor::new(config);

        let mut state = 12345u32;
        let original: Vec<u8> = (0..2048)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect();

        let compressed =
            compressor.compress_weights(&original).expect("compression should succeed");
        assert_eq!(
            compressed.stats.strategy_used,
            CompressionStrategy::HuffmanCompression
        );
        assert_eq!(compressed.stats.reconstruction_error, 0.0);

        let decompressed = compressor
            .decompress_weights(&compressed)
            .expect("Huffman data must always decompress");
        assert_eq!(
            decompressed, original,
            "Huffman/DEFLATE round trip must be byte-exact"
        );
    }

    #[test]
    fn test_huffman_round_trip_exact_on_highly_repetitive_data() {
        let config = CompressionConfig::new(
            CompressionStrategy::HuffmanCompression,
            CompressionLevel::Aggressive,
        );
        let compressor = WeightCompressor::new(config);

        let original = std::vec![0x42u8; 8192];
        let compressed =
            compressor.compress_weights(&original).expect("compression should succeed");

        // Highly repetitive data must compress well for real (this is
        // exactly what the old zeros-buffer stub could never demonstrate,
        // since it never looked at the input).
        assert!(
            compressed.data.len() < original.len() / 4,
            "repetitive data should compress substantially, got {} of {} bytes",
            compressed.data.len(),
            original.len()
        );

        let decompressed = compressor
            .decompress_weights(&compressed)
            .expect("Huffman data must always decompress");
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_huffman_compression_ratio_is_measured_not_from_fixed_table() {
        // The old code picked compression_ratio from a fixed per-level
        // table ({Light:1.2, Medium:1.5, Aggressive:2.0, Maximum:2.5})
        // completely independent of the data. Highly compressible input at
        // the same level as highly-random input must yield very different
        // real ratios.
        let config = CompressionConfig::new(
            CompressionStrategy::HuffmanCompression,
            CompressionLevel::Medium,
        );
        let compressor = WeightCompressor::new(config);

        let repetitive = std::vec![7u8; 4096];
        let mut state = 999u32;
        let random: Vec<u8> = (0..4096)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect();

        let ratio_repetitive =
            compressor.compress_weights(&repetitive).unwrap().stats.compression_ratio;
        let ratio_random = compressor.compress_weights(&random).unwrap().stats.compression_ratio;

        assert!(
            ratio_repetitive > ratio_random * 2.0,
            "repetitive data must compress far better than random data: {ratio_repetitive} vs {ratio_random}"
        );
    }

    #[test]
    fn test_decompress_weights_rejects_irreversible_strategies() {
        for strategy in [
            CompressionStrategy::MagnitudePruning,
            CompressionStrategy::StructuredPruning,
            CompressionStrategy::WeightClustering,
        ] {
            let config = CompressionConfig::new(strategy, CompressionLevel::Medium);
            let compressor = WeightCompressor::new(config);
            let weights: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();
            let data: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();

            let compressed =
                compressor.compress_weights(&data).expect("compression should succeed");
            // `decompress_weights_inner` rather than `decompress_weights`:
            // the latter's error path constructs a `JsValue`, which
            // unconditionally aborts the process on non-wasm32 targets
            // (see its doc comment).
            let err = compressor
                .decompress_weights_inner(&compressed)
                .expect_err("irreversible strategies must not silently return fabricated bytes");
            assert!(!err.is_empty());
        }
    }

    // -----------------------------------------------------------------
    // Safe byte<->f32 conversion (replacing the former unaligned-pointer
    // `unsafe` reinterpret cast).
    // -----------------------------------------------------------------

    #[test]
    fn test_bytes_f32_conversion_round_trip() {
        let config = CompressionConfig::new(CompressionStrategy::None, CompressionLevel::Light);
        let compressor = WeightCompressor::new(config);
        let original = std::vec![1.0f32, -2.5, 0.0, std::f32::consts::PI, -0.001];
        let bytes = compressor.f32_slice_to_bytes(&original);
        assert_eq!(bytes.len(), original.len() * 4);
        let decoded = compressor.bytes_to_f32_slice(&bytes);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_bytes_f32_conversion_drops_trailing_partial_bytes() {
        let config = CompressionConfig::new(CompressionStrategy::None, CompressionLevel::Light);
        let compressor = WeightCompressor::new(config);
        // 4 complete floats (16 bytes) plus 3 trailing bytes that cannot
        // form a complete f32 — must be dropped, not read out of bounds or
        // panic.
        let mut bytes = compressor.f32_slice_to_bytes(&std::vec![1.0f32, 2.0, 3.0, 4.0]);
        bytes.extend_from_slice(&[0xFF, 0xAB, 0x01]);
        let decoded = compressor.bytes_to_f32_slice(&bytes);
        assert_eq!(decoded, std::vec![1.0f32, 2.0, 3.0, 4.0]);
    }
}
