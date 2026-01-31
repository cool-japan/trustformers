use anyhow::Result;
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

#[derive(Debug, Clone)]
pub struct CompressedGradient {
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
    pub original_size: usize,
    pub compression_ratio: f32,
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
        })
    }

    fn compress_quantized(&self, gradient: &[f32], bits: u8) -> Result<CompressedGradient> {
        let levels = (1 << bits) - 1;
        let min_val = gradient.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = gradient.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let scale = (max_val - min_val) / levels as f32;

        let mut quantized_values = Vec::new();
        let mut indices = Vec::new();

        for (i, &val) in gradient.iter().enumerate() {
            let quantized = ((val - min_val) / scale).round() as i32;
            let dequantized = min_val + quantized as f32 * scale;

            indices.push(i);
            quantized_values.push(dequantized);
        }

        Ok(CompressedGradient {
            indices,
            values: quantized_values,
            original_size: gradient.len(),
            compression_ratio: (bits as f32) / 32.0, // Assuming f32 gradients
        })
    }

    fn compress_signsgd(&self, gradient: &[f32]) -> Result<CompressedGradient> {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &val) in gradient.iter().enumerate() {
            indices.push(i);
            values.push(if val >= 0.0 { 1.0 } else { -1.0 });
        }

        Ok(CompressedGradient {
            indices,
            values,
            original_size: gradient.len(),
            compression_ratio: 1.0 / 32.0, // 1 bit vs 32 bits
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

    pub fn all_reduce(
        &mut self,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        // Compress gradients
        let compressed = self.compressor.compress(gradients)?;

        // Simulate all-reduce operation (in practice, this would use MPI/NCCL)
        let aggregated = self.simulate_all_reduce(&compressed)?;

        // Decompress and average
        let mut result = self.compressor.decompress(&aggregated)?;

        // Average across all workers
        for (_, gradient) in result.iter_mut() {
            let mut data = gradient.data()?;
            for val in data.iter_mut() {
                *val /= self.world_size as f32;
            }
            *gradient = Tensor::new(data)?;
        }

        Ok(result)
    }

    fn simulate_all_reduce(
        &self,
        compressed: &HashMap<String, CompressedGradient>,
    ) -> Result<HashMap<String, CompressedGradient>> {
        // In a real implementation, this would:
        // 1. Send compressed gradients to all other workers
        // 2. Receive compressed gradients from all other workers
        // 3. Aggregate the sparse representations
        // 4. Return the aggregated result

        // For simulation, just return the input scaled by world_size
        let mut result = HashMap::new();

        for (name, grad) in compressed.iter() {
            let mut aggregated_values = grad.values.clone();
            for val in aggregated_values.iter_mut() {
                *val *= self.world_size as f32; // Simulate sum across workers
            }

            result.insert(
                name.clone(),
                CompressedGradient {
                    indices: grad.indices.clone(),
                    values: aggregated_values,
                    original_size: grad.original_size,
                    compression_ratio: grad.compression_ratio,
                },
            );
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topk_compression() {
        let mut compressor = GradientCompressor::new(CompressionMethod::TopK { k: 3 });
        let gradient = vec![0.1, 0.8, 0.2, -0.9, 0.3, -0.1];

        let compressed = compressor.compress_single(&gradient, "test").unwrap();

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

        let compressed = compressor.compress_single(&gradient, "test").unwrap();

        // Only values with abs > 0.5 should be included: 0.8, -0.9
        assert_eq!(compressed.values.len(), 2);
        assert!(compressed.values.contains(&0.8));
        assert!(compressed.values.contains(&-0.9));
    }

    #[test]
    fn test_signsgd_compression() {
        let mut compressor = GradientCompressor::new(CompressionMethod::SignSGD);
        let gradient = vec![0.1, -0.8, 0.2, -0.9, 0.3, -0.1];

        let compressed = compressor.compress_single(&gradient, "test").unwrap();

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
            Tensor::new(grad_data.clone()).unwrap(),
        );

        let compressed = compressor.compress(&gradients).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        let result_data = decompressed.get("param1").unwrap().data().unwrap();
        assert_eq!(result_data.len(), grad_data.len());

        // Check that the largest values are preserved
        assert!(result_data.contains(&0.8));
        assert!(result_data.contains(&-0.9));
    }

    #[test]
    fn test_compressed_all_reduce() {
        let mut all_reduce = CompressedAllReduce::new(
            CompressionMethod::TopK { k: 2 },
            4, // 4 workers
        );

        let mut gradients = HashMap::new();
        let grad_data = vec![0.4, 0.8, 0.2, -0.6];
        gradients.insert("param1".to_string(), Tensor::new(grad_data).unwrap());

        let result = all_reduce.all_reduce(&gradients).unwrap();

        let result_data = result.get("param1").unwrap().data().unwrap();
        assert_eq!(result_data.len(), 4);

        // Values should be averaged across workers (divided by world_size)
        for &val in &result_data {
            if val != 0.0 {
                // Non-zero values should be the original values (since we simulated identity operation)
                assert!(val.abs() <= 1.0); // Reasonable bound
            }
        }
    }
}
