/// Streaming Weight Loader
///
/// This module provides streaming weight loading for models that don't fit entirely in memory.
/// It loads and manages chunks of tensors, evicting old chunks as needed using an LRU strategy.
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use trustformers_core::{
    errors::{ErrorKind, Result, TrustformersError},
    tensor::{DType, Tensor},
};

use super::config::WeightLoadingConfig;
use super::huggingface::{HuggingFaceLoader, TensorMetadata, WeightLoader};

/// Chunk information for streaming loader
#[derive(Debug, Clone)]
pub struct ChunkInfo {
    pub chunk_id: usize,
    pub tensor_names: Vec<String>,
    pub memory_usage: usize,
    pub last_accessed: std::time::Instant,
}

/// Streaming weight loader for models that don't fit in memory
pub struct StreamingLoader {
    config: WeightLoadingConfig,
    model_path: PathBuf,
    chunk_size: usize,
    max_memory_usage: usize,
    current_chunks: HashMap<usize, ChunkInfo>,
    tensor_to_chunk: HashMap<String, usize>,
    loaded_tensors: HashMap<String, Tensor>,
    chunk_access_order: VecDeque<usize>,
    total_memory_usage: usize,
    tensor_metadata_cache: HashMap<String, TensorMetadata>,
    underlying_loader: Arc<Mutex<Option<HuggingFaceLoader>>>,
}

impl StreamingLoader {
    pub fn new(
        config: WeightLoadingConfig,
        model_path: PathBuf,
        chunk_size: usize,
        max_memory_mb: usize,
    ) -> Self {
        Self {
            config,
            model_path,
            chunk_size,
            max_memory_usage: max_memory_mb * 1024 * 1024, // Convert MB to bytes
            current_chunks: HashMap::new(),
            tensor_to_chunk: HashMap::new(),
            loaded_tensors: HashMap::new(),
            chunk_access_order: VecDeque::new(),
            total_memory_usage: 0,
            tensor_metadata_cache: HashMap::new(),
            underlying_loader: Arc::new(Mutex::new(None)),
        }
    }

    /// Initialize the streaming loader by analyzing the model structure
    pub fn initialize(&mut self) -> Result<()> {
        // Create underlying loader to analyze tensor structure
        let loader = HuggingFaceLoader::new(&self.model_path, self.config.clone())?;

        // Get all tensor information
        let tensor_names = loader.list_tensors()?;
        let mut current_chunk_id = 0;
        let mut current_chunk_size: usize = 0;
        let mut current_chunk_tensors = Vec::new();

        // Group tensors into chunks based on size
        for tensor_name in tensor_names {
            if let Some(metadata) = loader.tensor_info(&tensor_name)? {
                self.tensor_metadata_cache.insert(tensor_name.clone(), metadata.clone());

                // Check if adding this tensor would exceed chunk size
                if current_chunk_size + (metadata.size_bytes as usize) > self.chunk_size
                    && !current_chunk_tensors.is_empty()
                {
                    // Finalize current chunk
                    self.finalize_chunk(
                        current_chunk_id,
                        current_chunk_tensors,
                        current_chunk_size,
                    );

                    // Start new chunk
                    current_chunk_id += 1;
                    current_chunk_tensors = Vec::new();
                    current_chunk_size = 0;
                }

                current_chunk_tensors.push(tensor_name.clone());
                current_chunk_size += metadata.size_bytes as usize;
                self.tensor_to_chunk.insert(tensor_name, current_chunk_id);
            }
        }

        // Finalize the last chunk
        if !current_chunk_tensors.is_empty() {
            self.finalize_chunk(current_chunk_id, current_chunk_tensors, current_chunk_size);
        }

        // Store the underlying loader for later use
        *self.underlying_loader.lock().unwrap() = Some(loader);

        Ok(())
    }

    fn finalize_chunk(&mut self, chunk_id: usize, tensor_names: Vec<String>, memory_usage: usize) {
        let chunk_info = ChunkInfo {
            chunk_id,
            tensor_names,
            memory_usage,
            last_accessed: std::time::Instant::now(),
        };
        self.current_chunks.insert(chunk_id, chunk_info);
    }

    pub fn load_chunk(&mut self, chunk_id: usize) -> Result<()> {
        // Check if chunk is already loaded
        if self.chunk_access_order.contains(&chunk_id) {
            // Move to front of access order (most recently used)
            self.chunk_access_order.retain(|&x| x != chunk_id);
            self.chunk_access_order.push_front(chunk_id);
            return Ok(());
        }

        // Get chunk info
        let chunk_info = self
            .current_chunks
            .get(&chunk_id)
            .ok_or_else(|| {
                TrustformersError::invalid_operation(format!("Chunk {} not found", chunk_id))
            })?
            .clone();

        // Calculate memory needed for this chunk
        let chunk_memory = chunk_info.memory_usage;

        // Evict chunks if necessary to make room
        while self.total_memory_usage + chunk_memory > self.max_memory_usage
            && !self.chunk_access_order.is_empty()
        {
            let oldest_chunk = self.chunk_access_order.pop_back().unwrap();
            self.evict_chunk_internal(oldest_chunk)?;
        }

        // Load tensors for this chunk
        let mut loader_guard = self.underlying_loader.lock().unwrap();
        if let Some(loader) = loader_guard.as_mut() {
            for tensor_name in &chunk_info.tensor_names {
                let tensor = loader.load_tensor(tensor_name)?;
                let tensor_size = self.calculate_tensor_memory_usage(&tensor);
                self.loaded_tensors.insert(tensor_name.clone(), tensor);
                self.total_memory_usage += tensor_size;
            }
        } else {
            return Err(TrustformersError::invalid_operation(
                "Streaming loader not initialized".to_string(),
            ));
        }

        // Add to access order (most recently used at front)
        self.chunk_access_order.push_front(chunk_id);

        // Update chunk access time
        if let Some(chunk) = self.current_chunks.get_mut(&chunk_id) {
            chunk.last_accessed = std::time::Instant::now();
        }

        Ok(())
    }

    fn calculate_tensor_memory_usage(&self, tensor: &Tensor) -> usize {
        let element_count: usize = tensor.shape().iter().product();
        let bytes_per_element = match tensor.dtype() {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::F64 => 8,
            DType::I32 => 4,
            DType::I64 => 8,
            _ => 4, // Default to 4 bytes
        };
        element_count * bytes_per_element
    }

    pub fn evict_chunk(&mut self, chunk_id: usize) -> Result<()> {
        self.evict_chunk_internal(chunk_id)
    }

    fn evict_chunk_internal(&mut self, chunk_id: usize) -> Result<()> {
        // Get chunk info
        if let Some(chunk_info) = self.current_chunks.get(&chunk_id) {
            let tensor_names = chunk_info.tensor_names.clone();

            // Remove tensors from memory and update memory usage
            for tensor_name in &tensor_names {
                if let Some(tensor) = self.loaded_tensors.remove(tensor_name) {
                    let tensor_size = self.calculate_tensor_memory_usage(&tensor);
                    self.total_memory_usage = self.total_memory_usage.saturating_sub(tensor_size);
                }
            }

            // Remove from access order
            self.chunk_access_order.retain(|&x| x != chunk_id);
        }

        Ok(())
    }

    pub fn get_memory_usage(&self) -> usize {
        self.total_memory_usage
    }

    /// Get memory usage as a percentage of maximum allowed
    pub fn get_memory_usage_percentage(&self) -> f32 {
        if self.max_memory_usage == 0 {
            0.0
        } else {
            (self.total_memory_usage as f32 / self.max_memory_usage as f32) * 100.0
        }
    }

    /// Get detailed memory statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        MemoryStats {
            current_usage_bytes: self.total_memory_usage,
            max_usage_bytes: self.max_memory_usage,
            usage_percentage: self.get_memory_usage_percentage(),
            loaded_chunks: self.chunk_access_order.len(),
            total_chunks: self.current_chunks.len(),
            loaded_tensors: self.loaded_tensors.len(),
        }
    }

    pub fn is_chunk_loaded(&self, chunk_id: usize) -> bool {
        self.chunk_access_order.contains(&chunk_id)
    }

    /// Get information about all chunks
    pub fn get_chunk_info(&self) -> Vec<ChunkInfo> {
        self.current_chunks.values().cloned().collect()
    }

    /// Force garbage collection of unused tensors
    pub fn garbage_collect(&mut self) -> Result<()> {
        // This could be enhanced with more sophisticated GC logic
        // For now, just ensure memory usage tracking is accurate
        let mut actual_usage = 0;
        for tensor in self.loaded_tensors.values() {
            actual_usage += self.calculate_tensor_memory_usage(tensor);
        }
        self.total_memory_usage = actual_usage;
        Ok(())
    }
}

impl WeightLoader for StreamingLoader {
    fn load_tensor(&mut self, name: &str) -> Result<Tensor> {
        // Check if tensor is already loaded
        if let Some(tensor) = self.loaded_tensors.get(name) {
            // Update access time for the chunk containing this tensor
            if let Some(&chunk_id) = self.tensor_to_chunk.get(name) {
                if let Some(chunk) = self.current_chunks.get_mut(&chunk_id) {
                    chunk.last_accessed = std::time::Instant::now();
                }
                // Move chunk to front of access order
                self.chunk_access_order.retain(|&x| x != chunk_id);
                self.chunk_access_order.push_front(chunk_id);
            }
            return Ok(tensor.clone());
        }

        // Find which chunk contains this tensor
        if let Some(&chunk_id) = self.tensor_to_chunk.get(name) {
            // Load the chunk
            self.load_chunk(chunk_id)?;

            // Now the tensor should be loaded
            if let Some(tensor) = self.loaded_tensors.get(name) {
                Ok(tensor.clone())
            } else {
                Err(TrustformersError::new(ErrorKind::WeightLoadingError {
                    reason: format!("Failed to load tensor {} from chunk {}", name, chunk_id),
                }))
            }
        } else {
            Err(TrustformersError::new(ErrorKind::WeightLoadingError {
                reason: format!("Tensor {} not found in any chunk", name),
            }))
        }
    }

    fn list_tensors(&self) -> Result<Vec<String>> {
        // Return all tensor names across all chunks
        Ok(self.tensor_to_chunk.keys().cloned().collect())
    }

    fn tensor_info(&self, name: &str) -> Result<Option<TensorMetadata>> {
        Ok(self.tensor_metadata_cache.get(name).cloned())
    }

    fn close(&mut self) -> Result<()> {
        // Clear all loaded data
        self.loaded_tensors.clear();
        self.chunk_access_order.clear();
        self.total_memory_usage = 0;

        // Close underlying loader
        if let Some(mut loader) = self.underlying_loader.lock().unwrap().take() {
            loader.close()?;
        }

        Ok(())
    }
}

/// Additional methods for streaming-specific functionality
impl StreamingLoader {
    /// List only currently loaded tensors
    pub fn list_loaded_tensors(&self) -> Vec<String> {
        self.loaded_tensors.keys().cloned().collect()
    }
}

/// Memory usage statistics for the streaming loader
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub current_usage_bytes: usize,
    pub max_usage_bytes: usize,
    pub usage_percentage: f32,
    pub loaded_chunks: usize,
    pub total_chunks: usize,
    pub loaded_tensors: usize,
}

impl std::fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Memory Usage: {:.1}% ({} / {} bytes), Chunks: {} / {}, Tensors: {}",
            self.usage_percentage,
            self.current_usage_bytes,
            self.max_usage_bytes,
            self.loaded_chunks,
            self.total_chunks,
            self.loaded_tensors
        )
    }
}
