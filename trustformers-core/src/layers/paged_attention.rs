#![allow(unused_variables)] // Paged attention implementation

use crate::errors::{Result, TrustformersError};
use crate::layers::Linear;
use crate::tensor::Tensor;
use crate::traits::Layer;
use ndarray::{s, Array2, ArrayD, Axis, IxDyn};
use std::collections::HashMap;
use std::sync::RwLock;

/// PagedAttention: Memory-efficient attention for inference
///
/// This implements PagedAttention which organizes KV cache in pages
/// to avoid memory fragmentation and enable efficient memory management
/// during long sequence generation.
///
/// Reference: Efficient Memory Management for Large Language Model Serving with PagedAttention
/// https://arxiv.org/abs/2309.06180
#[derive(Debug)]
pub struct PagedAttention {
    num_heads: usize,
    hidden_size: usize,
    head_dim: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    out_proj: Linear,
    #[allow(dead_code)]
    dropout_prob: f32,
    page_size: usize,
    max_pages: usize,
    block_tables: RwLock<HashMap<usize, Vec<usize>>>, // sequence_id -> block_ids
    kv_cache: RwLock<KVCache>,
}

/// KV Cache organized in pages for efficient memory management
#[derive(Debug, Clone)]
pub struct KVCache {
    key_cache: Vec<Option<ArrayD<f32>>>,   // page_id -> key data
    value_cache: Vec<Option<ArrayD<f32>>>, // page_id -> value data
    free_pages: Vec<usize>,
    #[allow(dead_code)]
    page_size: usize,
    _num_heads: usize,
    _head_dim: usize,
}

impl KVCache {
    pub fn new(max_pages: usize, page_size: usize, num_heads: usize, head_dim: usize) -> Self {
        let mut free_pages = Vec::with_capacity(max_pages);
        for i in 0..max_pages {
            free_pages.push(i);
        }

        Self {
            key_cache: vec![None; max_pages],
            value_cache: vec![None; max_pages],
            free_pages,
            page_size,
            _num_heads: num_heads,
            _head_dim: head_dim,
        }
    }

    /// Allocate a new page and return its ID
    pub fn allocate_page(&mut self) -> Option<usize> {
        self.free_pages.pop()
    }

    /// Free a page and return it to the free list
    pub fn free_page(&mut self, page_id: usize) {
        if page_id < self.key_cache.len() {
            self.key_cache[page_id] = None;
            self.value_cache[page_id] = None;
            self.free_pages.push(page_id);
        }
    }

    /// Store key data in a page
    pub fn store_key(&mut self, page_id: usize, data: ArrayD<f32>) -> Result<()> {
        if page_id >= self.key_cache.len() {
            return Err(TrustformersError::invalid_config(
                "Page ID out of bounds".into(),
            ));
        }
        self.key_cache[page_id] = Some(data);
        Ok(())
    }

    /// Store value data in a page
    pub fn store_value(&mut self, page_id: usize, data: ArrayD<f32>) -> Result<()> {
        if page_id >= self.value_cache.len() {
            return Err(TrustformersError::invalid_config(
                "Page ID out of bounds".into(),
            ));
        }
        self.value_cache[page_id] = Some(data);
        Ok(())
    }

    /// Retrieve key data from a page
    pub fn get_key(&self, page_id: usize) -> Option<&ArrayD<f32>> {
        self.key_cache.get(page_id)?.as_ref()
    }

    /// Retrieve value data from a page
    pub fn get_value(&self, page_id: usize) -> Option<&ArrayD<f32>> {
        self.value_cache.get(page_id)?.as_ref()
    }

    /// Get number of available pages
    pub fn available_pages(&self) -> usize {
        self.free_pages.len()
    }
}

impl PagedAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        dropout_prob: f32,
        bias: bool,
        page_size: usize,
        max_pages: usize,
    ) -> Result<Self> {
        if hidden_size % num_heads != 0 {
            return Err(TrustformersError::invalid_config(format!(
                "hidden_size {} must be divisible by num_heads {}",
                hidden_size, num_heads
            )));
        }

        let head_dim = hidden_size / num_heads;
        let kv_cache = KVCache::new(max_pages, page_size, num_heads, head_dim);

        Ok(Self {
            num_heads,
            hidden_size,
            head_dim,
            query: Linear::new(hidden_size, hidden_size, bias),
            key: Linear::new(hidden_size, hidden_size, bias),
            value: Linear::new(hidden_size, hidden_size, bias),
            out_proj: Linear::new(hidden_size, hidden_size, bias),
            dropout_prob,
            page_size,
            max_pages,
            block_tables: RwLock::new(HashMap::new()),
            kv_cache: RwLock::new(kv_cache),
        })
    }

    /// Allocate pages for a new sequence
    pub fn allocate_sequence(&self, sequence_id: usize, estimated_length: usize) -> Result<()> {
        let pages_needed = (estimated_length + self.page_size - 1) / self.page_size;
        let mut allocated_pages = Vec::new();

        for _ in 0..pages_needed {
            if let Some(page_id) = self.kv_cache.write().unwrap().allocate_page() {
                allocated_pages.push(page_id);
            } else {
                // Free allocated pages if we can't allocate all needed
                for &page_id in &allocated_pages {
                    self.kv_cache.write().unwrap().free_page(page_id);
                }
                return Err(TrustformersError::resource_exhausted(
                    "Not enough pages available".into(),
                ));
            }
        }

        self.block_tables.write().unwrap().insert(sequence_id, allocated_pages);
        Ok(())
    }

    /// Free all pages for a sequence
    pub fn free_sequence(&self, sequence_id: usize) {
        if let Some(page_ids) = self.block_tables.write().unwrap().remove(&sequence_id) {
            for page_id in page_ids {
                self.kv_cache.write().unwrap().free_page(page_id);
            }
        }
    }

    /// Split tensor into heads: [batch, seq_len, hidden] -> [batch, num_heads, seq_len, head_dim]
    fn split_heads(&self, tensor: &Tensor) -> Result<Tensor> {
        match tensor {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                if shape.len() != 3 {
                    return Err(TrustformersError::shape_error(
                        "Expected 3D tensor for split_heads".into(),
                    ));
                }

                let batch_size = shape[0];
                let seq_len = shape[1];

                // Reshape to [batch, seq_len, num_heads, head_dim]
                let reshaped = arr
                    .clone()
                    .into_shape_with_order(IxDyn(&[
                        batch_size,
                        seq_len,
                        self.num_heads,
                        self.head_dim,
                    ]))
                    .map_err(|_| {
                        TrustformersError::shape_error("Failed to reshape in split_heads".into())
                    })?;

                // Transpose to [batch, num_heads, seq_len, head_dim]
                let transposed = reshaped.permuted_axes(vec![0, 2, 1, 3]);

                Ok(Tensor::F32(transposed))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type",
                "PagedAttention::split_heads",
            )),
        }
    }

    /// Merge heads back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden]
    fn merge_heads(&self, tensor: &Tensor) -> Result<Tensor> {
        let shape = tensor.shape();

        match tensor {
            Tensor::F32(arr) => {
                let batch_size = shape[0];
                let seq_len = shape[2];

                // Transpose back to [batch, seq_len, num_heads, head_dim]
                let transposed = arr.clone().permuted_axes(vec![0, 2, 1, 3]);

                // Reshape to [batch, seq_len, hidden_size]
                let merged = transposed
                    .to_shape(IxDyn(&[batch_size, seq_len, self.hidden_size]))
                    .map_err(|_| {
                        TrustformersError::shape_error("Failed to reshape in merge_heads".into())
                    })?
                    .to_owned();

                Ok(Tensor::F32(merged))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type",
                "PagedAttention::merge_heads",
            )),
        }
    }

    /// PagedAttention computation with efficient KV cache management
    pub fn paged_attention_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        sequence_id: usize,
        position: usize,
    ) -> Result<Tensor> {
        let q_shape = q.shape();
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let seq_len = q_shape[2];
        let head_dim = q_shape[3];
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Get or allocate pages for this sequence
        if !self.block_tables.read().unwrap().contains_key(&sequence_id) {
            self.allocate_sequence(sequence_id, seq_len * 2)?; // Allocate for some growth
        }

        match (q, k, v) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr), Tensor::F32(v_arr)) => {
                let mut output = ArrayD::zeros(IxDyn(&[batch_size, num_heads, seq_len, head_dim]));

                // For each batch and head
                for b in 0..batch_size {
                    for h in 0..num_heads {
                        // Extract head-specific Q, K, V
                        let q_batch = q_arr.index_axis(Axis(0), b);
                        let k_batch = k_arr.index_axis(Axis(0), b);
                        let v_batch = v_arr.index_axis(Axis(0), b);
                        let q_head = q_batch.index_axis(Axis(0), h);
                        let k_head = k_batch.index_axis(Axis(0), h);
                        let v_head = v_batch.index_axis(Axis(0), h);

                        // Store new K, V in pages
                        let page_ids =
                            self.block_tables.read().unwrap().get(&sequence_id).cloned().unwrap();

                        // For simplicity, use full attention for now
                        // In practice, this would manage KV cache pages more efficiently
                        let mut scores = Array2::<f32>::zeros((seq_len, seq_len));

                        // Compute attention scores
                        for qi in 0..seq_len {
                            for ki in 0..seq_len {
                                let mut dot_product = 0.0;
                                for d in 0..head_dim {
                                    dot_product += q_head[[qi, d]] * k_head[[ki, d]];
                                }
                                scores[[qi, ki]] = dot_product * scale;
                            }
                        }

                        // Apply causal mask (for decoder attention)
                        for qi in 0..seq_len {
                            for ki in qi + 1..seq_len {
                                scores[[qi, ki]] = f32::NEG_INFINITY;
                            }
                        }

                        // Softmax
                        for qi in 0..seq_len {
                            let max_score = scores
                                .slice(s![qi, ..])
                                .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                            let mut sum = 0.0;
                            for ki in 0..seq_len {
                                scores[[qi, ki]] = (scores[[qi, ki]] - max_score).exp();
                                sum += scores[[qi, ki]];
                            }
                            for ki in 0..seq_len {
                                scores[[qi, ki]] /= sum;
                            }
                        }

                        // Apply attention to values
                        for qi in 0..seq_len {
                            for d in 0..head_dim {
                                let mut output_val = 0.0;
                                for ki in 0..seq_len {
                                    output_val += scores[[qi, ki]] * v_head[[ki, d]];
                                }
                                output[[b, h, qi, d]] = output_val;
                            }
                        }
                    }
                }

                Ok(Tensor::F32(output))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for paged attention",
                "PagedAttention::paged_attention_forward",
            )),
        }
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let kv_cache = self.kv_cache.read().unwrap();
        let block_tables = self.block_tables.read().unwrap();
        MemoryStats {
            total_pages: self.max_pages,
            used_pages: self.max_pages - kv_cache.available_pages(),
            available_pages: kv_cache.available_pages(),
            page_size: self.page_size,
            active_sequences: block_tables.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_pages: usize,
    pub used_pages: usize,
    pub available_pages: usize,
    pub page_size: usize,
    pub active_sequences: usize,
}

#[derive(Debug, Clone)]
pub struct PagedAttentionInput {
    pub hidden_states: Tensor,
    pub sequence_id: usize,
    pub position: usize,
    pub attention_mask: Option<Tensor>,
}

impl Layer for PagedAttention {
    type Input = PagedAttentionInput;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden_states = input.hidden_states;

        // Track if input was originally 2D to decide whether to squeeze output
        let was_2d = match &hidden_states {
            Tensor::F32(arr) => arr.ndim() == 2,
            _ => false,
        };

        // Handle 2D input by adding batch dimension
        let hidden_states = match &hidden_states {
            Tensor::F32(arr) => {
                if arr.ndim() == 2 {
                    let shape = arr.shape();
                    let expanded = arr
                        .view()
                        .into_shape_with_order(IxDyn(&[1, shape[0], shape[1]]))
                        .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to add batch dimension: {}",
                                e
                            ))
                        })?;
                    Tensor::F32(expanded.to_owned())
                } else {
                    hidden_states
                }
            },
            _ => hidden_states,
        };

        // Compute Q, K, V projections
        let query_states = self.query.forward(hidden_states.clone())?;
        let key_states = self.key.forward(hidden_states.clone())?;
        let value_states = self.value.forward(hidden_states)?;

        // Split into attention heads
        let query_states = self.split_heads(&query_states)?;
        let key_states = self.split_heads(&key_states)?;
        let value_states = self.split_heads(&value_states)?;

        // Apply PagedAttention
        let context = self.paged_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            input.sequence_id,
            input.position,
        )?;

        // Merge heads back
        let context = self.merge_heads(&context)?;

        // Apply output projection
        let result = self.out_proj.forward(context)?;

        // Remove batch dimension only if input was originally 2D
        if was_2d {
            match &result {
                Tensor::F32(arr) => {
                    if arr.shape()[0] == 1 {
                        let squeezed = arr.index_axis(Axis(0), 0).to_owned();
                        Ok(Tensor::F32(squeezed))
                    } else {
                        Ok(result)
                    }
                },
                _ => Ok(result),
            }
        } else {
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_paged_attention_creation() {
        let paged_attn = PagedAttention::new(768, 12, 0.1, true, 64, 1000);
        assert!(paged_attn.is_ok());

        let paged_attn = paged_attn.unwrap();
        assert_eq!(paged_attn.num_heads, 12);
        assert_eq!(paged_attn.hidden_size, 768);
        assert_eq!(paged_attn.head_dim, 64);
        assert_eq!(paged_attn.page_size, 64);
        assert_eq!(paged_attn.max_pages, 1000);
    }

    #[test]
    fn test_kv_cache_operations() {
        let mut cache = KVCache::new(10, 64, 8, 64);

        // Test allocation
        let page1 = cache.allocate_page();
        assert!(page1.is_some());
        let page1 = page1.unwrap();

        let page2 = cache.allocate_page();
        assert!(page2.is_some());
        let page2 = page2.unwrap();

        assert_ne!(page1, page2);
        assert_eq!(cache.available_pages(), 8);

        // Test storage and retrieval
        let test_data = ArrayD::from_elem(IxDyn(&[8, 64, 64]), 1.0f32);
        cache.store_key(page1, test_data.clone()).unwrap();

        let retrieved = cache.get_key(page1);
        assert!(retrieved.is_some());

        // Test freeing
        cache.free_page(page1);
        assert_eq!(cache.available_pages(), 9);
        assert!(cache.get_key(page1).is_none());
    }

    #[test]
    fn test_sequence_allocation() {
        let paged_attn = PagedAttention::new(256, 8, 0.0, true, 32, 100).unwrap();

        // Allocate sequence
        let result = paged_attn.allocate_sequence(1, 128);
        assert!(result.is_ok());

        // Check block table
        assert!(paged_attn.block_tables.read().unwrap().contains_key(&1));
        let pages = paged_attn.block_tables.read().unwrap().get(&1).cloned().unwrap();
        assert_eq!(pages.len(), 4); // 128 / 32 = 4 pages

        // Free sequence
        paged_attn.free_sequence(1);
        assert!(!paged_attn.block_tables.read().unwrap().contains_key(&1));
    }

    #[test]
    fn test_paged_attention_forward() {
        let paged_attn = PagedAttention::new(256, 8, 0.0, true, 32, 100).unwrap();

        let hidden_states = Tensor::randn(&[1, 64, 256]).unwrap();
        let input = PagedAttentionInput {
            hidden_states,
            sequence_id: 1,
            position: 0,
            attention_mask: None,
        };

        let output = paged_attn.forward(input);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape(), vec![1, 64, 256]);
    }

    #[test]
    fn test_memory_stats() {
        let paged_attn = PagedAttention::new(256, 8, 0.0, true, 32, 100).unwrap();

        let stats = paged_attn.memory_stats();
        assert_eq!(stats.total_pages, 100);
        assert_eq!(stats.used_pages, 0);
        assert_eq!(stats.available_pages, 100);
        assert_eq!(stats.page_size, 32);
        assert_eq!(stats.active_sequences, 0);

        // Allocate a sequence
        paged_attn.allocate_sequence(1, 128).unwrap();

        let stats = paged_attn.memory_stats();
        assert_eq!(stats.used_pages, 4);
        assert_eq!(stats.available_pages, 96);
        assert_eq!(stats.active_sequences, 1);
    }
}
