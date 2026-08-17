use anyhow::{anyhow, Result};
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

/// Gradient compression algorithms for distributed training.
///
/// Reduces communication overhead by compressing gradients before
/// sending them across the network in distributed training setups.

#[derive(Debug, Clone)]
pub enum CompressionMethod {
    /// Top-K sparsification: only send the K largest gradients
    TopK { k: usize },
    /// Random-K sparsification: randomly sample K gradients
    RandomK { k: usize },
    /// Threshold-based sparsification: send gradients above threshold
    Threshold { threshold: f32 },
    /// Quantization-based compression
    Quantization { bits: u8 },
    /// SignSGD: send only the sign of gradients
    SignSGD,
    /// Error feedback compression
    ErrorFeedback { base_method: Box<CompressionMethod> },
}

#[derive(Debug)]
pub struct GradientCompressor {
    method: CompressionMethod,
    compression_ratio: f32,
    error_buffer: HashMap<String, Vec<f32>>, // For error feedback
}

/// One gradient after compression, in the form it would be put on the wire.
///
/// `indices` is **empty for dense methods** (quantization, SignSGD): those transmit a
/// value for every coordinate, so an explicit `0..n` index list would be pure
/// overhead. Sparse methods (top-k, random-k, threshold) list the coordinates they
/// kept.
///
/// `values` always holds the *dequantized* `f32` values so callers can do arithmetic
/// with them directly; `value_bits` records how many bits each value actually needs on
/// the wire, which is what [`CompressedGradient::payload_bytes`] uses.
#[derive(Debug, Clone)]
pub struct CompressedGradient {
    /// Kept coordinates, or empty for a dense positional payload.
    pub indices: Vec<usize>,
    /// Dequantized values, in `indices` order (or coordinate order when dense).
    pub values: Vec<f32>,
    /// Number of elements in the uncompressed gradient.
    pub original_size: usize,
    /// Fraction of the original payload this representation occupies.
    pub compression_ratio: f32,
    /// Bits each value needs on the wire (`32` for an untransformed `f32`, `8` for
    /// 8-bit quantization, `1` for a sign).
    pub value_bits: u8,
}

impl CompressedGradient {
    /// Bytes this representation would put on the wire.
    ///
    /// Indices cost `size_of::<usize>()` each; values cost `value_bits` bits each,
    /// rounded up to whole bytes.
    pub fn payload_bytes(&self) -> usize {
        let index_bytes = self.indices.len() * std::mem::size_of::<usize>();
        let value_bits = self.values.len() * self.value_bits as usize;
        index_bytes + value_bits.div_ceil(8)
    }

    /// Bytes the uncompressed `f32` gradient would occupy.
    pub fn dense_bytes(&self) -> usize {
        self.original_size * std::mem::size_of::<f32>()
    }
}

impl GradientCompressor {
    pub fn new(method: CompressionMethod) -> Self {
        Self {
            method,
            compression_ratio: 0.0,
            error_buffer: HashMap::new(),
        }
    }

    pub fn compress(
        &mut self,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, CompressedGradient>> {
        let mut compressed = HashMap::new();

        for (name, gradient) in gradients.iter() {
            let grad_data = gradient.data()?;
            let compressed_grad = self.compress_single(&grad_data, name)?;
            compressed.insert(name.clone(), compressed_grad);
        }

        Ok(compressed)
    }

    pub fn decompress(
        &self,
        compressed: &HashMap<String, CompressedGradient>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut decompressed = HashMap::new();

        for (name, compressed_grad) in compressed.iter() {
            let grad_data = self.decompress_single(compressed_grad)?;
            decompressed.insert(name.clone(), Tensor::new(grad_data)?);
        }

        Ok(decompressed)
    }

    fn compress_single(
        &mut self,
        gradient: &[f32],
        param_name: &str,
    ) -> Result<CompressedGradient> {
        match self.method.clone() {
            CompressionMethod::TopK { k } => self.compress_topk(gradient, k),
            CompressionMethod::RandomK { k } => self.compress_randomk(gradient, k),
            CompressionMethod::Threshold { threshold } => {
                self.compress_threshold(gradient, threshold)
            },
            CompressionMethod::Quantization { bits } => self.compress_quantized(gradient, bits),
            CompressionMethod::SignSGD => self.compress_signsgd(gradient),
            CompressionMethod::ErrorFeedback { base_method } => {
                self.compress_with_error_feedback(gradient, param_name, &base_method)
            },
        }
    }

    fn compress_topk(&self, gradient: &[f32], k: usize) -> Result<CompressedGradient> {
        let mut indexed_grads: Vec<(usize, f32)> =
            gradient.iter().enumerate().map(|(i, &val)| (i, val.abs())).collect();

        // Sort by absolute value in descending order
        indexed_grads.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = k.min(gradient.len());
        let mut indices = Vec::with_capacity(k);
        let mut values = Vec::with_capacity(k);

        for i in 0..k {
            let (idx, _) = indexed_grads[i];
            indices.push(idx);
            values.push(gradient[idx]);
        }

        Ok(CompressedGradient {
            indices,
            values,
            original_size: gradient.len(),
            compression_ratio: k as f32 / gradient.len() as f32,
            value_bits: 32,
        })
    }

    fn compress_randomk(&self, gradient: &[f32], k: usize) -> Result<CompressedGradient> {
        use std::collections::HashSet;

        let k = k.min(gradient.len());
        let mut indices = Vec::with_capacity(k);
        let mut values = Vec::with_capacity(k);
        let mut selected_indices = HashSet::new();

        // Simple random sampling (in practice, would use proper RNG)
        let step = gradient.len() / k.max(1);
        for i in (0..gradient.len()).step_by(step) {
            if indices.len() < k && !selected_indices.contains(&i) {
                indices.push(i);
                values.push(gradient[i]);
                selected_indices.insert(i);
            }
        }

        Ok(CompressedGradient {
            indices,
            values,
            original_size: gradient.len(),
            compression_ratio: k as f32 / gradient.len() as f32,
            value_bits: 32,
        })
    }

    fn compress_threshold(&self, gradient: &[f32], threshold: f32) -> Result<CompressedGradient> {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &val) in gradient.iter().enumerate() {
            if val.abs() > threshold {
                indices.push(i);
                values.push(val);
            }
        }

        let compression_ratio = indices.len() as f32 / gradient.len() as f32;

        Ok(CompressedGradient {
            indices,
            values,
            original_size: gradient.len(),
            compression_ratio,
            value_bits: 32,
        })
    }

    fn compress_quantized(&self, gradient: &[f32], bits: u8) -> Result<CompressedGradient> {
        let levels = (1 << bits) - 1;
        let min_val = gradient.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = gradient.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let scale = (max_val - min_val) / levels as f32;

        let mut quantized_values = Vec::with_capacity(gradient.len());

        for &val in gradient.iter() {
            let quantized = if scale > 0.0 { ((val - min_val) / scale).round() } else { 0.0 };
            quantized_values.push(min_val + quantized * scale);
        }

        Ok(CompressedGradient {
            // Dense: every coordinate is transmitted, so no index list is needed.
            indices: Vec::new(),
            values: quantized_values,
            original_size: gradient.len(),
            compression_ratio: (bits as f32) / 32.0, // Assuming f32 gradients
            value_bits: bits,
        })
    }

    fn compress_signsgd(&self, gradient: &[f32]) -> Result<CompressedGradient> {
        let values: Vec<f32> =
            gradient.iter().map(|&val| if val >= 0.0 { 1.0 } else { -1.0 }).collect();

        Ok(CompressedGradient {
            // Dense: one sign bit per coordinate, so no index list is needed.
            indices: Vec::new(),
            values,
            original_size: gradient.len(),
            compression_ratio: 1.0 / 32.0, // 1 bit vs 32 bits
            value_bits: 1,
        })
    }

    fn compress_with_error_feedback(
        &mut self,
        gradient: &[f32],
        param_name: &str,
        base_method: &Box<CompressionMethod>,
    ) -> Result<CompressedGradient> {
        // Add accumulated error to current gradient
        let mut corrected_gradient = gradient.to_vec();

        if let Some(error) = self.error_buffer.get(param_name) {
            for i in 0..corrected_gradient.len().min(error.len()) {
                corrected_gradient[i] += error[i];
            }
        }

        // Compress the corrected gradient
        let mut temp_compressor = GradientCompressor::new((**base_method).clone());
        let compressed = temp_compressor.compress_single(&corrected_gradient, param_name)?;

        // Compute and store the new error
        let decompressed = self.decompress_single(&compressed)?;
        let mut new_error = vec![0.0; corrected_gradient.len()];

        for i in 0..new_error.len() {
            new_error[i] = corrected_gradient[i] - decompressed.get(i).copied().unwrap_or(0.0);
        }

        self.error_buffer.insert(param_name.to_string(), new_error);

        Ok(compressed)
    }

    fn decompress_single(&self, compressed: &CompressedGradient) -> Result<Vec<f32>> {
        let mut gradient = vec![0.0; compressed.original_size];

        if compressed.indices.is_empty() {
            // Dense payload: values are already in coordinate order.
            if compressed.values.len() != compressed.original_size && !compressed.values.is_empty()
            {
                return Err(anyhow!(
                    "dense compressed payload has {} values but the gradient has {}",
                    compressed.values.len(),
                    compressed.original_size
                ));
            }
            gradient[..compressed.values.len()].copy_from_slice(&compressed.values);
            return Ok(gradient);
        }

        for (&i, &value) in compressed.indices.iter().zip(compressed.values.iter()) {
            if i < gradient.len() {
                gradient[i] = value;
            }
        }

        Ok(gradient)
    }

    pub fn get_compression_ratio(&self) -> f32 {
        self.compression_ratio
    }

    pub fn reset_error_buffer(&mut self) {
        self.error_buffer.clear();
    }
}

/// Distributed gradient aggregator with compression support
#[derive(Debug)]
pub struct CompressedAllReduce {
    compressor: GradientCompressor,
    world_size: usize,
}

impl CompressedAllReduce {
    pub fn new(compression_method: CompressionMethod, world_size: usize) -> Self {
        Self {
            compressor: GradientCompressor::new(compression_method),
            world_size,
        }
    }

    /// Compresses, aggregates and averages gradients across the process group.
    ///
    /// # Aggregation semantics
    ///
    /// No transport is wired up in this crate, so there is exactly one honest thing
    /// this can do:
    ///
    /// * `world_size == 1` — the local rank *is* the group. The gradient is compressed
    ///   and decompressed (so the caller sees the real compression error) and returned.
    /// * `world_size > 1` with peer contributions supplied — see
    ///   [`CompressedAllReduce::all_reduce_with_peers`], which performs the real sum.
    /// * `world_size > 1` with no peers — an error. Scaling the local gradient by
    ///   `world_size` (what this used to do) is not an approximation of a sum across
    ///   workers; it is numerically worse than doing nothing.
    ///
    /// # Errors
    ///
    /// Returns an error when `world_size > 1`, because no communicator is configured.
    pub fn all_reduce(
        &mut self,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        if self.world_size != 1 {
            return Err(anyhow!(
                "CompressedAllReduce has no communicator: cannot aggregate across {} ranks. \
                 Use all_reduce_with_peers to supply the other ranks' compressed gradients.",
                self.world_size
            ));
        }

        let compressed = self.compressor.compress(gradients)?;
        self.compressor.decompress(&compressed)
    }

    /// Performs a genuine compressed all-reduce given the peers' compressed gradients.
    ///
    /// `peers` holds one map per *other* rank. Each parameter's contributions are
    /// summed in the dense domain and divided by the number of contributing ranks, so
    /// the result is the true average of the compressed gradients — not a rescaling of
    /// the local one.
    ///
    /// # Errors
    ///
    /// Returns an error when a peer supplies a gradient of a different length, or when
    /// the number of contributions does not match `world_size`.
    pub fn all_reduce_with_peers(
        &mut self,
        gradients: &HashMap<String, Tensor>,
        peers: &[HashMap<String, CompressedGradient>],
    ) -> Result<HashMap<String, Tensor>> {
        if peers.len() + 1 != self.world_size {
            return Err(anyhow!(
                "expected {} peer contributions for world_size {}, got {}",
                self.world_size - 1,
                self.world_size,
                peers.len()
            ));
        }

        let local = self.compressor.compress(gradients)?;
        let mut summed: HashMap<String, Vec<f32>> = HashMap::new();

        for contribution in std::iter::once(&local).chain(peers.iter()) {
            for (name, compressed) in contribution {
                let dense = self.compressor.decompress_single(compressed)?;
                match summed.get_mut(name) {
                    Some(accumulator) => {
                        if accumulator.len() != dense.len() {
                            return Err(anyhow!(
                                "rank contributions for '{name}' disagree on length: {} vs {}",
                                accumulator.len(),
                                dense.len()
                            ));
                        }
                        for (slot, value) in accumulator.iter_mut().zip(dense.iter()) {
                            *slot += value;
                        }
                    },
                    None => {
                        summed.insert(name.clone(), dense);
                    },
                }
            }
        }

        let divisor = self.world_size as f32;
        let mut result = HashMap::new();
        for (name, mut values) in summed {
            for value in values.iter_mut() {
                *value /= divisor;
            }
            let shape = gradients
                .get(&name)
                .map(|t| t.shape().to_vec())
                .unwrap_or_else(|| vec![values.len()]);
            result.insert(name, Tensor::from_vec(values, &shape)?);
        }

        Ok(result)
    }

    /// Compresses `gradients` for transmission to the other ranks.
    ///
    /// # Errors
    ///
    /// Returns an error when a gradient cannot be read.
    pub fn compress_for_transmission(
        &mut self,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, CompressedGradient>> {
        self.compressor.compress(gradients)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topk_compression() {
        let mut compressor = GradientCompressor::new(CompressionMethod::TopK { k: 3 });
        let gradient = vec![0.1, 0.8, 0.2, -0.9, 0.3, -0.1];

        let compressed =
            compressor.compress_single(&gradient, "test").expect("Operation failed in test");

        assert_eq!(compressed.indices.len(), 3);
        assert_eq!(compressed.values.len(), 3);
        assert_eq!(compressed.original_size, 6);
        assert!(compressed.compression_ratio < 1.0);

        // Should include the largest magnitude values: -0.9, 0.8, 0.3
        assert!(compressed.values.contains(&-0.9));
        assert!(compressed.values.contains(&0.8));
        assert!(compressed.values.contains(&0.3));
    }

    #[test]
    fn test_threshold_compression() {
        let mut compressor =
            GradientCompressor::new(CompressionMethod::Threshold { threshold: 0.5 });
        let gradient = vec![0.1, 0.8, 0.2, -0.9, 0.3, -0.1];

        let compressed =
            compressor.compress_single(&gradient, "test").expect("Operation failed in test");

        // Only values with abs > 0.5 should be included: 0.8, -0.9
        assert_eq!(compressed.values.len(), 2);
        assert!(compressed.values.contains(&0.8));
        assert!(compressed.values.contains(&-0.9));
    }

    #[test]
    fn test_signsgd_compression() {
        let mut compressor = GradientCompressor::new(CompressionMethod::SignSGD);
        let gradient = vec![0.1, -0.8, 0.2, -0.9, 0.3, -0.1];

        let compressed =
            compressor.compress_single(&gradient, "test").expect("Operation failed in test");

        assert_eq!(compressed.values.len(), gradient.len());
        assert_eq!(compressed.compression_ratio, 1.0 / 32.0);

        // All values should be either 1.0 or -1.0
        for &val in &compressed.values {
            assert!(val == 1.0 || val == -1.0);
        }
    }

    #[test]
    fn test_compression_decompression_roundtrip() {
        let mut compressor = GradientCompressor::new(CompressionMethod::TopK { k: 3 });
        let mut gradients = HashMap::new();

        let grad_data = vec![0.1, 0.8, 0.2, -0.9, 0.3, -0.1];
        gradients.insert(
            "param1".to_string(),
            Tensor::new(grad_data.clone()).expect("Failed to create tensor"),
        );

        let compressed = compressor.compress(&gradients).expect("Operation failed in test");
        let decompressed = compressor.decompress(&compressed).expect("Operation failed in test");

        let result_data = decompressed
            .get("param1")
            .expect("Key not found")
            .data()
            .expect("Operation failed in test");
        assert_eq!(result_data.len(), grad_data.len());

        // Check that the largest values are preserved
        assert!(result_data.contains(&0.8));
        assert!(result_data.contains(&-0.9));
    }

    #[test]
    fn test_compressed_all_reduce() {
        // Regression: `all_reduce` used to multiply the *local* gradient by
        // `world_size` and call the result an aggregation across workers.
        let mut all_reduce = CompressedAllReduce::new(CompressionMethod::TopK { k: 2 }, 4);

        let mut gradients = HashMap::new();
        gradients.insert(
            "param1".to_string(),
            Tensor::from_vec(vec![0.4_f32, 0.8, 0.2, -0.6], &[4]).expect("tensor"),
        );

        assert!(
            all_reduce.all_reduce(&gradients).is_err(),
            "aggregating across 4 ranks with no communicator must be an error"
        );
    }

    #[test]
    fn test_single_rank_all_reduce_is_a_round_trip() {
        let mut all_reduce = CompressedAllReduce::new(CompressionMethod::SignSGD, 1);

        let mut gradients = HashMap::new();
        gradients.insert(
            "param1".to_string(),
            Tensor::from_vec(vec![0.4_f32, -0.8, 0.2, -0.6], &[4]).expect("tensor"),
        );

        let result = all_reduce.all_reduce(&gradients).expect("single-rank all-reduce");
        let values = result.get("param1").expect("param1").data_f32().expect("data");
        assert_eq!(
            values,
            vec![1.0, -1.0, 1.0, -1.0],
            "SignSGD keeps only the sign"
        );
    }

    #[test]
    fn test_all_reduce_with_peers_sums_and_averages() {
        let mut all_reduce = CompressedAllReduce::new(CompressionMethod::SignSGD, 3);

        let mut local = HashMap::new();
        local.insert(
            "w".to_string(),
            Tensor::from_vec(vec![1.0_f32, 1.0], &[2]).expect("tensor"),
        );

        // Two peers, both reporting the opposite sign on the second coordinate.
        let mut peer = GradientCompressor::new(CompressionMethod::SignSGD);
        let mut peer_gradients = HashMap::new();
        peer_gradients.insert(
            "w".to_string(),
            Tensor::from_vec(vec![1.0_f32, -1.0], &[2]).expect("tensor"),
        );
        let peer_payload = peer.compress(&peer_gradients).expect("compress");

        let result = all_reduce
            .all_reduce_with_peers(&local, &[peer_payload.clone(), peer_payload])
            .expect("all reduce");
        let values = result.get("w").expect("w").data_f32().expect("data");

        // Coordinate 0: (1 + 1 + 1)/3 = 1. Coordinate 1: (1 − 1 − 1)/3 = −1/3.
        assert!((values[0] - 1.0).abs() < 1e-6, "{}", values[0]);
        assert!((values[1] + 1.0 / 3.0).abs() < 1e-6, "{}", values[1]);
    }

    #[test]
    fn test_payload_bytes_reflects_the_method() {
        let mut compressor = GradientCompressor::new(CompressionMethod::SignSGD);
        let gradient = vec![0.1_f32; 64];
        let compressed = compressor.compress_single(&gradient, "w").expect("compress");

        // One sign bit per coordinate, no index list.
        assert!(compressed.indices.is_empty());
        assert_eq!(compressed.payload_bytes(), 8);
        assert_eq!(compressed.dense_bytes(), 256);

        let mut sparse = GradientCompressor::new(CompressionMethod::TopK { k: 4 });
        let compressed = sparse.compress_single(&gradient, "w").expect("compress");
        // 4 indices (8 bytes each) plus 4 f32 values.
        assert_eq!(compressed.payload_bytes(), 4 * 8 + 4 * 4);
    }

    #[test]
    fn test_dense_round_trip_preserves_positions() {
        let mut compressor = GradientCompressor::new(CompressionMethod::Quantization { bits: 8 });
        let gradient = vec![-1.0_f32, -0.5, 0.0, 0.5, 1.0];
        let compressed = compressor.compress_single(&gradient, "w").expect("compress");
        assert!(
            compressed.indices.is_empty(),
            "a dense method needs no index list"
        );

        let restored = compressor.decompress_single(&compressed).expect("decompress");
        assert_eq!(restored.len(), gradient.len());
        for (a, b) in restored.iter().zip(gradient.iter()) {
            assert!((a - b).abs() < 0.02, "8-bit round trip: {a} vs {b}");
        }
    }
}
