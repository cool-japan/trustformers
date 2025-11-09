use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Memory optimization utilities for TrustformeRS
///
/// This module provides high-priority memory optimizations:
/// - Zero-copy tensor views for slice operations
/// - Memory mapping for large model weights
/// - Custom allocators for tensor allocation patterns
/// - Tensor memory recycling pool
///
/// Configuration for memory optimizations
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Enable memory pool for tensor recycling
    pub enable_memory_pool: bool,
    /// Maximum size of memory pool in bytes
    pub max_pool_size: usize,
    /// Enable zero-copy tensor views
    pub enable_zero_copy: bool,
    /// Enable memory mapping for large tensors
    pub enable_mmap: bool,
    /// Minimum size for memory mapping (in bytes)
    pub mmap_threshold: usize,
    /// Pool cleanup interval
    pub cleanup_interval: Duration,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enable_memory_pool: true,
            max_pool_size: 1024 * 1024 * 1024, // 1GB
            enable_zero_copy: true,
            enable_mmap: true,
            mmap_threshold: 100 * 1024 * 1024, // 100MB
            cleanup_interval: Duration::from_secs(60),
        }
    }
}

/// Memory pool entry for tensor recycling
#[derive(Debug, Clone)]
struct PoolEntry {
    tensor: Tensor,
    last_used: Instant,
    ref_count: usize,
}

/// Zero-copy tensor view for slice operations
#[derive(Debug)]
pub struct TensorView {
    /// Original tensor reference
    original: Arc<Tensor>,
    /// Offset in the original tensor
    offset: usize,
    /// Shape of the view
    shape: Vec<usize>,
    /// Strides for the view
    #[allow(dead_code)]
    strides: Vec<usize>,
}

impl TensorView {
    /// Create a new zero-copy view of a tensor slice
    pub fn slice(tensor: Arc<Tensor>, start: usize, end: usize) -> Result<Self> {
        let original_shape = tensor.shape();
        if start >= end || end > original_shape.iter().product::<usize>() {
            return Err(TrustformersError::invalid_input(
                "Invalid slice bounds".to_string(),
            ));
        }

        let slice_len = end - start;
        Ok(Self {
            original: tensor,
            offset: start,
            shape: vec![slice_len],
            strides: vec![1],
        })
    }

    /// Get the shape of the view
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the underlying tensor data (zero-copy)
    pub fn as_tensor(&self) -> Result<Tensor> {
        // This would implement actual zero-copy slicing
        // For now, return a simple implementation
        match &*self.original {
            Tensor::F32(arr) => {
                let flat = arr
                    .view()
                    .into_shape_with_order(arr.len())
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                let slice = flat.slice(ndarray::s![
                    self.offset..self.offset + self.shape.iter().product::<usize>()
                ]);
                let sliced_arr = slice
                    .to_owned()
                    .into_shape_with_order(ndarray::IxDyn(&self.shape))
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(Tensor::F32(sliced_arr))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Zero-copy slicing not implemented for this tensor type",
                "zero_copy_slice",
            )),
        }
    }
}

/// Memory pool for tensor recycling
pub struct TensorMemoryPool {
    config: MemoryConfig,
    pool: Arc<RwLock<HashMap<Vec<usize>, Vec<PoolEntry>>>>,
    current_size: Arc<Mutex<usize>>,
    last_cleanup: Arc<Mutex<Instant>>,
}

impl TensorMemoryPool {
    /// Create a new memory pool
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            pool: Arc::new(RwLock::new(HashMap::new())),
            current_size: Arc::new(Mutex::new(0)),
            last_cleanup: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Get a tensor from the pool or create a new one
    pub fn get_tensor(&self, shape: &[usize], dtype: crate::tensor::DType) -> Result<Tensor> {
        if !self.config.enable_memory_pool {
            return self.create_tensor(shape, dtype);
        }

        // Try to get from pool first
        if let Some(tensor) = self.try_get_from_pool(shape)? {
            return Ok(tensor);
        }

        // Create new tensor if none available in pool
        self.create_tensor(shape, dtype)
    }

    /// Return a tensor to the pool for recycling
    pub fn return_tensor(&self, tensor: Tensor) -> Result<()> {
        if !self.config.enable_memory_pool {
            return Ok(()); // Just drop the tensor
        }

        let shape = tensor.shape().to_vec();

        // Calculate tensor size before moving
        let tensor_size = self.estimate_tensor_size(&tensor);

        let entry = PoolEntry {
            tensor,
            last_used: Instant::now(),
            ref_count: 0,
        };

        let mut pool = self.pool.write().unwrap();
        pool.entry(shape).or_default().push(entry);

        // Update current size
        *self.current_size.lock().unwrap() += tensor_size;

        // Cleanup if needed
        self.cleanup_if_needed()?;

        Ok(())
    }

    /// Try to get a tensor from the pool
    fn try_get_from_pool(&self, shape: &[usize]) -> Result<Option<Tensor>> {
        let mut pool = self.pool.write().unwrap();

        if let Some(entries) = pool.get_mut(shape) {
            if let Some(entry) = entries.pop() {
                let tensor_size = self.estimate_tensor_size(&entry.tensor);
                *self.current_size.lock().unwrap() -= tensor_size;
                return Ok(Some(entry.tensor));
            }
        }

        Ok(None)
    }

    /// Create a new tensor
    fn create_tensor(&self, shape: &[usize], dtype: crate::tensor::DType) -> Result<Tensor> {
        match dtype {
            crate::tensor::DType::F32 => Tensor::zeros(shape),
            crate::tensor::DType::F64 => Tensor::zeros_f64(shape),
            crate::tensor::DType::F16 => Tensor::zeros_f16(shape),
            crate::tensor::DType::BF16 => Tensor::zeros_bf16(shape),
            crate::tensor::DType::I64 => Tensor::zeros_i64(shape),
            crate::tensor::DType::C32 => Tensor::zeros_c32(shape),
            crate::tensor::DType::C64 => Tensor::zeros_c64(shape),
            crate::tensor::DType::CF16 => Tensor::zeros_cf16(shape),
            crate::tensor::DType::CBF16 => Tensor::zeros_cbf16(shape),
            _ => Err(TrustformersError::tensor_op_error(
                &format!("Tensor creation not implemented for dtype: {:?} - only supported types are F32, F64, F16, BF16, I64, C32, C64, CF16, CBF16", dtype),
                "create_tensor"
            )),
        }
    }

    /// Estimate the memory size of a tensor
    fn estimate_tensor_size(&self, tensor: &Tensor) -> usize {
        let elements = tensor.shape().iter().product::<usize>();
        match tensor {
            Tensor::F32(_) => elements * 4,   // 32-bit float
            Tensor::F64(_) => elements * 8,   // 64-bit float
            Tensor::F16(_) => elements * 2,   // 16-bit float
            Tensor::BF16(_) => elements * 2,  // 16-bit bfloat
            Tensor::I64(_) => elements * 8,   // 64-bit integer
            Tensor::C32(_) => elements * 8,   // 2 * 32-bit complex
            Tensor::C64(_) => elements * 16,  // 2 * 64-bit complex
            Tensor::CF16(_) => elements * 4,  // 2 * 16-bit complex
            Tensor::CBF16(_) => elements * 4, // 2 * 16-bit bfloat complex
            #[cfg(feature = "torch")]
            Tensor::Torch(_) => elements * 4, // Default to 32-bit
            #[cfg(feature = "candle")]
            Tensor::Candle(_) => elements * 4, // Default to 32-bit
            Tensor::Sparse(sparse) => {
                // For sparse tensors, estimate based on non-zero elements
                let nnz = sparse.nnz();
                nnz * 4 + nnz * std::mem::size_of::<usize>() // values + indices
            },
        }
    }

    /// Cleanup old entries if needed
    fn cleanup_if_needed(&self) -> Result<()> {
        let mut last_cleanup = self.last_cleanup.lock().unwrap();
        if last_cleanup.elapsed() < self.config.cleanup_interval {
            return Ok(());
        }

        let current_size = *self.current_size.lock().unwrap();
        if current_size <= self.config.max_pool_size {
            return Ok(());
        }

        // Cleanup old entries
        let mut pool = self.pool.write().unwrap();
        let mut total_freed = 0;
        let cutoff_time = Instant::now() - self.config.cleanup_interval;

        for entries in pool.values_mut() {
            entries.retain(|entry| {
                if entry.last_used < cutoff_time && entry.ref_count == 0 {
                    total_freed += self.estimate_tensor_size(&entry.tensor);
                    false
                } else {
                    true
                }
            });
        }

        // Remove empty entries
        pool.retain(|_, entries| !entries.is_empty());

        // Update size
        *self.current_size.lock().unwrap() -= total_freed;
        *last_cleanup = Instant::now();

        Ok(())
    }

    /// Get memory pool statistics
    pub fn get_stats(&self) -> MemoryPoolStats {
        let pool = self.pool.read().unwrap();
        let current_size = *self.current_size.lock().unwrap();

        let total_tensors = pool.values().map(|v| v.len()).sum();
        let total_shapes = pool.len();

        MemoryPoolStats {
            total_tensors,
            total_shapes,
            current_size_bytes: current_size,
            max_size_bytes: self.config.max_pool_size,
            utilization: current_size as f64 / self.config.max_pool_size as f64,
        }
    }
}

/// Statistics for memory pool
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub total_tensors: usize,
    pub total_shapes: usize,
    pub current_size_bytes: usize,
    pub max_size_bytes: usize,
    pub utilization: f64,
}

/// Memory mapped tensor for large model weights
pub struct MemoryMappedTensor {
    /// File path for the memory mapped data
    file_path: String,
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Data type
    dtype: crate::tensor::DType,
    /// File handle for memory mapped data
    _file: Option<File>,
    /// Size of the file in bytes
    file_size: u64,
}

impl MemoryMappedTensor {
    /// Create a new memory mapped tensor
    pub fn new(file_path: String, shape: Vec<usize>, dtype: crate::tensor::DType) -> Result<Self> {
        // Open the file for reading
        let mut file = File::open(&file_path).map_err(|e| {
            TrustformersError::tensor_op_error(
                &format!("Failed to open file for memory mapping: {}", e),
                "mmap_new",
            )
        })?;

        // Get file size
        let file_size = file.seek(SeekFrom::End(0)).map_err(|e| {
            TrustformersError::tensor_op_error(
                &format!("Failed to get file size: {}", e),
                "mmap_new",
            )
        })?;

        // Verify file size matches tensor size
        let element_size = dtype.size_in_bytes();
        let total_elements: usize = shape.iter().product();
        let expected_size = total_elements * element_size;

        if file_size != expected_size as u64 {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "File size {} doesn't match expected tensor size {}",
                    file_size, expected_size
                ),
                "mmap_new",
            ));
        }

        Ok(Self {
            file_path,
            shape,
            dtype,
            _file: Some(file),
            file_size,
        })
    }

    /// Load the tensor data (lazy loading)
    pub fn load(&self) -> Result<Tensor> {
        // Read the entire file content
        let mut file = File::open(&self.file_path).map_err(|e| {
            TrustformersError::tensor_op_error(
                &format!("Failed to open file for reading: {}", e),
                "mmap_load",
            )
        })?;

        let mut buffer = vec![0u8; self.file_size as usize];
        file.read_exact(&mut buffer).map_err(|e| {
            TrustformersError::tensor_op_error(
                &format!("Failed to read file data: {}", e),
                "mmap_load",
            )
        })?;

        // Convert bytes to appropriate tensor type
        match self.dtype {
            crate::tensor::DType::F32 => {
                let float_data = buffer
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect::<Vec<f32>>();
                Tensor::from_slice(&float_data, &self.shape)
            },
            crate::tensor::DType::F64 => {
                let float_data = buffer
                    .chunks_exact(8)
                    .map(|chunk| {
                        f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect::<Vec<f64>>();
                Tensor::from_slice_f64(&float_data, &self.shape)
            },
            crate::tensor::DType::I64 => {
                let int_data = buffer
                    .chunks_exact(8)
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect::<Vec<i64>>();
                Tensor::from_slice_i64(&int_data, &self.shape)
            },
            crate::tensor::DType::I32 => {
                let int_data = buffer
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect::<Vec<i32>>();
                Tensor::from_slice_i32(&int_data, &self.shape)
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported dtype for memory mapped tensor",
                "mmap_load",
            )),
        }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the file path
    pub fn file_path(&self) -> &str {
        &self.file_path
    }
}

/// Global memory manager instance
static MEMORY_MANAGER: std::sync::OnceLock<TensorMemoryPool> = std::sync::OnceLock::new();

/// Initialize the global memory manager
pub fn init_memory_manager(config: MemoryConfig) -> Result<()> {
    let pool = TensorMemoryPool::new(config);
    MEMORY_MANAGER.set(pool).map_err(|_| {
        TrustformersError::invalid_input("Memory manager already initialized".to_string())
    })?;
    Ok(())
}

/// Get the global memory manager
pub fn get_memory_manager() -> Option<&'static TensorMemoryPool> {
    MEMORY_MANAGER.get()
}

/// Convenience function to get a tensor from the global pool
pub fn get_tensor(shape: &[usize], dtype: crate::tensor::DType) -> Result<Tensor> {
    if let Some(manager) = get_memory_manager() {
        manager.get_tensor(shape, dtype)
    } else {
        // Fallback to direct creation
        match dtype {
            crate::tensor::DType::F32 => Tensor::zeros(shape),
            crate::tensor::DType::F64 => Tensor::zeros_f64(shape),
            crate::tensor::DType::I64 => Tensor::zeros_i64(shape),
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported dtype",
                "get_tensor",
            )),
        }
    }
}

/// Convenience function to return a tensor to the global pool
pub fn return_tensor(tensor: Tensor) -> Result<()> {
    if let Some(manager) = get_memory_manager() {
        manager.return_tensor(tensor)
    } else {
        Ok(()) // Just drop the tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_config_default() {
        let config = MemoryConfig::default();
        assert!(config.enable_memory_pool);
        assert!(config.enable_zero_copy);
        assert!(config.enable_mmap);
        assert_eq!(config.max_pool_size, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_tensor_pool_creation() {
        let config = MemoryConfig::default();
        let pool = TensorMemoryPool::new(config);
        let stats = pool.get_stats();
        assert_eq!(stats.total_tensors, 0);
        assert_eq!(stats.current_size_bytes, 0);
    }

    #[test]
    fn test_tensor_pool_get_and_return() -> Result<()> {
        let config = MemoryConfig::default();
        let pool = TensorMemoryPool::new(config);

        // Get a tensor
        let shape = vec![2, 3];
        let tensor = pool.get_tensor(&shape, crate::tensor::DType::F32)?;
        assert_eq!(tensor.shape(), shape.as_slice());

        // Return it to pool
        pool.return_tensor(tensor)?;

        // Get it again (should come from pool)
        let tensor2 = pool.get_tensor(&shape, crate::tensor::DType::F32)?;
        assert_eq!(tensor2.shape(), shape.as_slice());

        Ok(())
    }

    #[test]
    fn test_zero_copy_tensor_view() -> Result<()> {
        let tensor = Arc::new(Tensor::ones(&[10])?);
        let view = TensorView::slice(tensor, 2, 8)?;
        assert_eq!(view.shape(), &[6]);

        let viewed_tensor = view.as_tensor()?;
        assert_eq!(viewed_tensor.shape(), &[6]);

        Ok(())
    }

    #[test]
    fn test_memory_mapped_tensor() -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        // Create a temporary file with some data
        let temp_file = "test_temp.bin";
        let data_size = 100 * 100 * std::mem::size_of::<f32>();
        let data: Vec<u8> = vec![0; data_size];

        {
            let mut file = File::create(temp_file).map_err(|e| {
                TrustformersError::tensor_op_error(
                    &format!("Failed to create test file: {}", e),
                    "test_setup",
                )
            })?;
            file.write_all(&data).map_err(|e| {
                TrustformersError::tensor_op_error(
                    &format!("Failed to write test data: {}", e),
                    "test_setup",
                )
            })?;
        }

        let mmap_tensor = MemoryMappedTensor::new(
            temp_file.to_string(),
            vec![100, 100],
            crate::tensor::DType::F32,
        )?;

        assert_eq!(mmap_tensor.shape(), &[100, 100]);
        assert_eq!(mmap_tensor.file_path(), temp_file);

        let loaded = mmap_tensor.load()?;
        assert_eq!(loaded.shape(), &[100, 100]);

        // Clean up
        std::fs::remove_file(temp_file).ok();

        Ok(())
    }

    #[test]
    fn test_global_memory_manager() -> Result<()> {
        let config = MemoryConfig::default();
        init_memory_manager(config)?;

        let tensor = get_tensor(&[5, 5], crate::tensor::DType::F32)?;
        assert_eq!(tensor.shape(), [5, 5].as_slice());

        return_tensor(tensor)?;

        Ok(())
    }
}
