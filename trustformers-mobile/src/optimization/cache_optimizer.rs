//! Cache Optimization Module
//!
//! Optimizes memory access patterns and data layouts for mobile cache hierarchies

use super::{ComputationGraph, GraphOperator, KernelType};
use crate::MobilePlatform;
use std::collections::HashMap;
use trustformers_core::errors::Result;
use trustformers_core::Tensor;
use trustformers_core::TrustformersError;

/// Cache optimization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    /// Optimize for L1 cache
    L1Optimized,
    /// Optimize for L2 cache
    L2Optimized,
    /// Streaming (bypass cache)
    Streaming,
    /// Prefetch-heavy
    Prefetch,
}

/// Loop tiling configuration
#[derive(Debug, Clone)]
pub struct TilingConfig {
    /// Tile dimensions
    pub tile_sizes: Vec<usize>,
    /// Nested loop order
    pub loop_order: Vec<usize>,
    /// Unroll factors
    pub unroll_factors: Vec<usize>,
}

/// Data layout for cache efficiency
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataLayout {
    /// NCHW - batch, channels, height, width
    NCHW,
    /// NHWC - batch, height, width, channels
    NHWC,
    /// NC4HW4 - packed format for SIMD
    NC4HW4,
    /// Custom layout
    Custom,
}

/// Memory access pattern analysis
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// Stride pattern
    pub strides: Vec<isize>,
    /// Access order (sequential, strided, random)
    pub access_type: AccessType,
    /// Reuse distance
    pub reuse_distance: usize,
    /// Working set size
    pub working_set_size: usize,
}

/// Access type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    Sequential,
    Strided,
    Random,
    Broadcast,
}

/// Cache hints for operators
#[derive(Debug, Clone)]
pub struct CacheHints {
    /// Prefetch distance
    pub prefetch_distance: usize,
    /// Cache bypass for certain tensors
    pub bypass_cache: Vec<String>,
    /// Temporal locality hints
    pub temporal_hints: HashMap<String, TemporalHint>,
    /// Spatial locality hints
    pub spatial_hints: HashMap<String, SpatialHint>,
}

/// Temporal locality hint
#[derive(Debug, Clone, Copy)]
pub enum TemporalHint {
    /// High reuse - keep in cache
    HighReuse,
    /// Medium reuse
    MediumReuse,
    /// Low reuse - consider streaming
    LowReuse,
    /// No reuse - bypass cache
    NoReuse,
}

/// Spatial locality hint
#[derive(Debug, Clone, Copy)]
pub enum SpatialHint {
    /// Contiguous access
    Contiguous,
    /// Strided access
    Strided { stride: usize },
    /// Blocked access
    Blocked { block_size: usize },
}

/// Cache optimizer
pub struct CacheOptimizer {
    platform: MobilePlatform,
    cache_hierarchy: CacheHierarchy,
    optimization_cache: HashMap<String, CacheStrategy>,
}

/// Cache hierarchy description
#[derive(Debug, Clone)]
struct CacheHierarchy {
    l1_size: usize,
    l1_line_size: usize,
    l1_associativity: usize,
    l2_size: usize,
    l2_line_size: usize,
    l2_associativity: usize,
    l3_size: Option<usize>,
}

impl CacheOptimizer {
    /// Create new cache optimizer
    pub fn new(platform: MobilePlatform) -> Self {
        let cache_hierarchy = Self::detect_cache_hierarchy(&platform);

        Self {
            platform,
            cache_hierarchy,
            optimization_cache: HashMap::new(),
        }
    }

    /// Optimize tensor layout for cache efficiency
    pub fn optimize_layout(&self, tensor: &Tensor, pattern: &AccessPattern) -> Result<Tensor> {
        let layout = self.select_optimal_layout(&tensor.shape(), pattern)?;

        match layout {
            DataLayout::NHWC if self.current_layout_is_nchw(tensor) => {
                self.transpose_nchw_to_nhwc(tensor)
            },
            DataLayout::NC4HW4 => self.pack_to_nc4hw4(tensor),
            _ => Ok(tensor.clone()),
        }
    }

    /// Generate cache hints for kernel
    pub fn generate_hints(
        &self,
        kernel: &KernelType,
        input_shapes: &[Vec<usize>],
    ) -> Result<CacheHints> {
        let prefetch_distance = self.calculate_prefetch_distance(kernel, input_shapes)?;
        let bypass_cache = self.identify_streaming_tensors(kernel, input_shapes)?;
        let temporal_hints = self.analyze_temporal_locality(kernel, input_shapes)?;
        let spatial_hints = self.analyze_spatial_locality(kernel, input_shapes)?;

        Ok(CacheHints {
            prefetch_distance,
            bypass_cache,
            temporal_hints,
            spatial_hints,
        })
    }

    /// Apply loop tiling optimization
    pub fn apply_tiling(&self, graph: &mut ComputationGraph) -> Result<()> {
        for operator in &mut graph.operators {
            if self.can_tile(&operator.kernel) {
                let tiling_config = self.compute_tiling_config(operator)?;

                // Store tiling configuration in operator metadata
                if let Some(ref mut hints) = operator.cache_hints {
                    // Apply tiling configuration
                    self.apply_tiling_to_operator(operator, &tiling_config)?;
                }
            }
        }

        Ok(())
    }

    /// Analyze access pattern for tensor
    pub fn analyze(&self, tensor_name: &str, kernel: &KernelType) -> Result<AccessPattern> {
        let access_type = match kernel {
            KernelType::Conv2d => AccessType::Strided,
            KernelType::Linear => AccessType::Sequential,
            KernelType::Attention => AccessType::Random,
            _ => AccessType::Sequential,
        };

        let pattern = AccessPattern {
            strides: self.compute_strides(kernel)?,
            access_type,
            reuse_distance: self.estimate_reuse_distance(kernel)?,
            working_set_size: self.estimate_working_set(kernel)?,
        };

        Ok(pattern)
    }

    // Private helper methods

    fn detect_cache_hierarchy(platform: &MobilePlatform) -> CacheHierarchy {
        match platform {
            MobilePlatform::Ios => CacheHierarchy {
                l1_size: 64 * 1024, // 64KB L1
                l1_line_size: 64,
                l1_associativity: 4,
                l2_size: 3 * 1024 * 1024, // 3MB L2
                l2_line_size: 128,
                l2_associativity: 8,
                l3_size: None,
            },
            MobilePlatform::Android => CacheHierarchy {
                l1_size: 32 * 1024, // 32KB L1 (conservative)
                l1_line_size: 64,
                l1_associativity: 4,
                l2_size: 1024 * 1024, // 1MB L2
                l2_line_size: 64,
                l2_associativity: 8,
                l3_size: None,
            },
            MobilePlatform::Generic => CacheHierarchy {
                l1_size: 32 * 1024,
                l1_line_size: 64,
                l1_associativity: 4,
                l2_size: 256 * 1024,
                l2_line_size: 64,
                l2_associativity: 8,
                l3_size: None,
            },
        }
    }

    fn select_optimal_layout(
        &self,
        shape: &[usize],
        pattern: &AccessPattern,
    ) -> Result<DataLayout> {
        match pattern.access_type {
            AccessType::Sequential => Ok(DataLayout::NCHW),
            AccessType::Strided => {
                // NHWC is often better for strided access on mobile
                if self.platform == MobilePlatform::Android {
                    Ok(DataLayout::NHWC)
                } else {
                    Ok(DataLayout::NCHW)
                }
            },
            AccessType::Random => Ok(DataLayout::Custom),
            AccessType::Broadcast => Ok(DataLayout::NC4HW4), // Packed for SIMD
        }
    }

    fn current_layout_is_nchw(&self, tensor: &Tensor) -> bool {
        // Simplified check - would need actual layout metadata
        tensor.shape().len() == 4
    }

    fn transpose_nchw_to_nhwc(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.shape().len() != 4 {
            return Err(TrustformersError::tensor_op_error(
                "Expected 4D tensor",
                "transpose_nchw_to_nhwc",
            ));
        }

        let [n, c, h, w] = [
            tensor.shape()[0],
            tensor.shape()[1],
            tensor.shape()[2],
            tensor.shape()[3],
        ];
        let mut transposed_data = vec![0.0f32; n * h * w * c];
        let src_data = tensor.data()?;

        // Transpose from NCHW to NHWC
        for batch in 0..n {
            for channel in 0..c {
                for row in 0..h {
                    for col in 0..w {
                        let src_idx = batch * c * h * w + channel * h * w + row * w + col;
                        let dst_idx = batch * h * w * c + row * w * c + col * c + channel;
                        transposed_data[dst_idx] = src_data[src_idx];
                    }
                }
            }
        }

        Tensor::from_vec(transposed_data, &[n, h, w, c])
    }

    fn pack_to_nc4hw4(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.shape().len() != 4 {
            return Err(TrustformersError::tensor_op_error(
                "Expected 4D tensor",
                "transpose_nchw_to_nhwc",
            ));
        }

        let [n, c, h, w] = [
            tensor.shape()[0],
            tensor.shape()[1],
            tensor.shape()[2],
            tensor.shape()[3],
        ];
        let c_padded = c.div_ceil(4) * 4; // Round up to multiple of 4

        let mut packed_data = vec![0.0f32; n * c_padded * h * w];
        let src_data = tensor.data()?;

        // Pack channels in groups of 4 for SIMD
        for batch in 0..n {
            for c_group in 0..c.div_ceil(4) {
                for row in 0..h {
                    for col in 0..w {
                        for c_offset in 0..4 {
                            let c_idx = c_group * 4 + c_offset;
                            if c_idx < c {
                                let src_idx = batch * c * h * w + c_idx * h * w + row * w + col;
                                let dst_idx = batch * c_padded * h * w
                                    + c_group * 4 * h * w
                                    + row * w * 4
                                    + col * 4
                                    + c_offset;
                                packed_data[dst_idx] = src_data[src_idx];
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_vec(packed_data, &[n, c_padded, h, w])
    }

    fn calculate_prefetch_distance(
        &self,
        kernel: &KernelType,
        input_shapes: &[Vec<usize>],
    ) -> Result<usize> {
        // Calculate based on memory bandwidth and compute intensity
        let compute_intensity = match kernel {
            KernelType::Conv2d => 10.0,
            KernelType::Linear => 2.0,
            KernelType::Attention => 5.0,
            _ => 1.0,
        };

        // Higher compute intensity = larger prefetch distance
        Ok((compute_intensity * 2.0) as usize)
    }

    fn identify_streaming_tensors(
        &self,
        kernel: &KernelType,
        input_shapes: &[Vec<usize>],
    ) -> Result<Vec<String>> {
        let mut streaming = Vec::new();

        // Large tensors that won't fit in cache should stream
        for (idx, shape) in input_shapes.iter().enumerate() {
            let size_bytes = shape.iter().product::<usize>() * 4; // 4 bytes per float

            if size_bytes > self.cache_hierarchy.l2_size {
                streaming.push(format!("input_{}", idx));
            }
        }

        Ok(streaming)
    }

    fn analyze_temporal_locality(
        &self,
        kernel: &KernelType,
        input_shapes: &[Vec<usize>],
    ) -> Result<HashMap<String, TemporalHint>> {
        let mut hints = HashMap::new();

        match kernel {
            KernelType::Conv2d => {
                hints.insert("weights".to_string(), TemporalHint::HighReuse);
                hints.insert("input".to_string(), TemporalHint::MediumReuse);
            },
            KernelType::Linear => {
                hints.insert("weights".to_string(), TemporalHint::HighReuse);
                hints.insert("input".to_string(), TemporalHint::LowReuse);
            },
            KernelType::BatchNorm => {
                hints.insert("mean".to_string(), TemporalHint::HighReuse);
                hints.insert("variance".to_string(), TemporalHint::HighReuse);
            },
            _ => {},
        }

        Ok(hints)
    }

    fn analyze_spatial_locality(
        &self,
        kernel: &KernelType,
        input_shapes: &[Vec<usize>],
    ) -> Result<HashMap<String, SpatialHint>> {
        let mut hints = HashMap::new();

        match kernel {
            KernelType::Conv2d => {
                hints.insert("input".to_string(), SpatialHint::Blocked { block_size: 16 });
                hints.insert("output".to_string(), SpatialHint::Contiguous);
            },
            KernelType::Linear => {
                hints.insert("input".to_string(), SpatialHint::Contiguous);
                hints.insert(
                    "weights".to_string(),
                    SpatialHint::Strided {
                        stride: input_shapes[0][1],
                    },
                );
            },
            _ => {
                hints.insert("default".to_string(), SpatialHint::Contiguous);
            },
        }

        Ok(hints)
    }

    fn can_tile(&self, kernel: &KernelType) -> bool {
        matches!(
            kernel,
            KernelType::Conv2d | KernelType::Linear | KernelType::Attention
        )
    }

    fn compute_tiling_config(&self, operator: &GraphOperator) -> Result<TilingConfig> {
        let tile_size = self.compute_optimal_tile_size(operator)?;

        let config = match operator.kernel {
            KernelType::Conv2d => TilingConfig {
                tile_sizes: vec![1, tile_size, tile_size, tile_size], // N, C, H, W
                loop_order: vec![0, 2, 3, 1], // N, H, W, C for better locality
                unroll_factors: vec![1, 4, 1, 1],
            },
            KernelType::Linear => TilingConfig {
                tile_sizes: vec![tile_size, tile_size], // M, N
                loop_order: vec![0, 1],
                unroll_factors: vec![4, 4],
            },
            _ => TilingConfig {
                tile_sizes: vec![tile_size],
                loop_order: vec![0],
                unroll_factors: vec![4],
            },
        };

        Ok(config)
    }

    fn compute_optimal_tile_size(&self, operator: &GraphOperator) -> Result<usize> {
        // Calculate tile size to fit in L1 cache
        let working_set_elements = operator
            .input_shapes
            .iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum::<usize>();

        let element_size = 4; // 4 bytes per float
        let working_set_bytes = working_set_elements * element_size;

        // Use 75% of L1 cache
        let available_cache = (self.cache_hierarchy.l1_size as f32 * 0.75) as usize;

        // Find tile size that fits
        let mut tile_size = 64;
        while tile_size * tile_size * element_size > available_cache && tile_size > 8 {
            tile_size /= 2;
        }

        Ok(tile_size)
    }

    fn apply_tiling_to_operator(
        &self,
        operator: &mut GraphOperator,
        config: &TilingConfig,
    ) -> Result<()> {
        // This would modify the operator to use tiled execution
        // For now, just store the configuration
        operator.kernel = match operator.kernel.clone() {
            KernelType::Conv2d => KernelType::Custom("TiledConv2d".to_string()),
            KernelType::Linear => KernelType::Custom("TiledLinear".to_string()),
            other => other,
        };

        Ok(())
    }

    fn compute_strides(&self, kernel: &KernelType) -> Result<Vec<isize>> {
        Ok(match kernel {
            KernelType::Conv2d => vec![1, 1],  // Typical conv stride
            KernelType::Pooling => vec![2, 2], // Typical pooling stride
            _ => vec![1],                      // Sequential access
        })
    }

    fn estimate_reuse_distance(&self, kernel: &KernelType) -> Result<usize> {
        Ok(match kernel {
            KernelType::Conv2d => 1024,  // High reuse for weights
            KernelType::Linear => 256,   // Medium reuse
            KernelType::BatchNorm => 64, // Low reuse
            _ => 128,
        })
    }

    fn estimate_working_set(&self, kernel: &KernelType) -> Result<usize> {
        Ok(match kernel {
            KernelType::Conv2d => 256 * 1024,    // 256KB typical
            KernelType::Linear => 128 * 1024,    // 128KB
            KernelType::Attention => 512 * 1024, // 512KB
            _ => 64 * 1024,                      // 64KB
        })
    }
}

impl AccessPattern {
    /// Analyze access pattern for given tensor and kernel
    pub fn analyze(tensor_name: &str, kernel: &KernelType) -> Result<Self> {
        let access_type = if tensor_name.contains("weight") {
            AccessType::Broadcast
        } else if matches!(kernel, KernelType::Attention) {
            AccessType::Random
        } else {
            AccessType::Sequential
        };

        Ok(Self {
            strides: vec![1],
            access_type,
            reuse_distance: 128,
            working_set_size: 64 * 1024,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_optimizer_creation() {
        let optimizer = CacheOptimizer::new(MobilePlatform::Generic);
        assert_eq!(optimizer.platform, MobilePlatform::Generic);
    }

    #[test]
    fn test_layout_optimization() {
        let optimizer = CacheOptimizer::new(MobilePlatform::Android);
        let tensor = Tensor::ones(&[1, 64, 32, 32]).unwrap();
        let pattern = AccessPattern {
            strides: vec![1, 1],
            access_type: AccessType::Strided,
            reuse_distance: 128,
            working_set_size: 64 * 1024,
        };

        let optimized = optimizer.optimize_layout(&tensor, &pattern).unwrap();
        // On Android with strided access, should transpose to NHWC
        assert_eq!(optimized.shape(), &[1, 32, 32, 64]);
    }

    #[test]
    fn test_tiling_config() {
        let optimizer = CacheOptimizer::new(MobilePlatform::Ios);
        let operator = GraphOperator {
            id: 0,
            kernel: KernelType::Conv2d,
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            input_shapes: vec![vec![1, 64, 32, 32]],
            output_shape: vec![1, 128, 16, 16],
            cache_hints: None,
        };

        let config = optimizer.compute_tiling_config(&operator).unwrap();
        assert!(!config.tile_sizes.is_empty());
        assert!(!config.loop_order.is_empty());
    }

    #[test]
    fn test_cache_hints_generation() {
        let optimizer = CacheOptimizer::new(MobilePlatform::Generic);
        let kernel = KernelType::Conv2d;
        let input_shapes = vec![vec![1, 3, 224, 224]];

        let hints = optimizer.generate_hints(&kernel, &input_shapes).unwrap();
        assert!(hints.prefetch_distance > 0);
        assert!(!hints.temporal_hints.is_empty());
    }
}
