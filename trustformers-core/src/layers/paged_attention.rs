//! PagedAttention: attention backed by a real paged KV cache.
//!
//! The cache is organised in fixed-size pages (`page_size` tokens each). A
//! sequence owns a [`BlockTable`] mapping its logical page index to a physical
//! page id, so the memory for one sequence never has to be contiguous and no
//! space is wasted on the unused tail of a pre-allocated maximum length.
//!
//! Reference: *Efficient Memory Management for Large Language Model Serving
//! with PagedAttention* <https://arxiv.org/abs/2309.06180>

use crate::errors::{Result, TrustformersError};
use crate::layers::attention::mask::MaskView;
use crate::layers::Linear;
use crate::tensor::Tensor;
use crate::traits::Layer;
use scirs2_core::ndarray::{ArrayD, Axis, IxDyn};
use std::collections::HashMap;
use std::sync::RwLock;

/// PagedAttention: memory-efficient attention for inference.
///
/// Keys and values produced by [`PagedAttention::paged_attention_forward`] are
/// written into the pages owned by the sequence and every subsequent call
/// attends over the whole cached prefix, which is what makes incremental
/// decoding (`q_len == 1`) correct *and* cheap.
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
    /// `sequence_id -> block table`.
    ///
    /// Lock ordering across the struct is always `block_tables` before
    /// `kv_cache`; every method in this file obeys it so the two locks can be
    /// held together without risking a deadlock.
    block_tables: RwLock<HashMap<usize, BlockTable>>,
    kv_cache: RwLock<KVCache>,
}

/// Logical-to-physical page mapping for a single sequence.
///
/// Also records how many tokens have actually been written, which is the only
/// admissible source of the attendable context length - it must never be
/// inferred from the query tensor, whose length is 1 during decoding.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockTable {
    pages: Vec<usize>,
    length: usize,
}

impl BlockTable {
    /// Physical page ids backing this sequence, in logical order.
    pub fn pages(&self) -> &[usize] {
        &self.pages
    }

    /// Number of tokens written into the cache for this sequence.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Whether no token has been cached yet.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Number of tokens the currently allocated pages can hold.
    pub fn capacity(&self, page_size: usize) -> usize {
        self.pages.len() * page_size
    }

    /// Physical page holding the token at logical `position`.
    fn page_for(&self, position: usize, page_size: usize) -> Result<usize> {
        let logical = position / page_size;
        self.pages.get(logical).copied().ok_or_else(|| {
            TrustformersError::invalid_config(format!(
                "block table has {} pages, but token {} needs logical page {}",
                self.pages.len(),
                position,
                logical
            ))
        })
    }
}

/// KV cache organised in pages for efficient memory management.
///
/// Each page stores `page_size` consecutive tokens as an array whose last three
/// axes are `[num_heads, page_size, head_dim]`; the paged attention path uses
/// a leading batch axis (`[batch, num_heads, page_size, head_dim]`).
#[derive(Debug, Clone)]
pub struct KVCache {
    key_cache: Vec<Option<ArrayD<f32>>>,   // page_id -> key data
    value_cache: Vec<Option<ArrayD<f32>>>, // page_id -> value data
    free_pages: Vec<usize>,
    page_size: usize,
    num_heads: usize,
    head_dim: usize,
}

impl KVCache {
    /// Create a cache with `max_pages` pages of `page_size` tokens each.
    pub fn new(max_pages: usize, page_size: usize, num_heads: usize, head_dim: usize) -> Self {
        let free_pages = (0..max_pages).rev().collect();

        Self {
            key_cache: vec![None; max_pages],
            value_cache: vec![None; max_pages],
            free_pages,
            page_size,
            num_heads,
            head_dim,
        }
    }

    /// Tokens held by one page.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Number of attention heads each page stores.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Per-head feature width each page stores.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Allocate a new page and return its ID.
    pub fn allocate_page(&mut self) -> Option<usize> {
        self.free_pages.pop()
    }

    /// Free a page and return it to the free list.
    pub fn free_page(&mut self, page_id: usize) {
        if page_id < self.key_cache.len() && !self.free_pages.contains(&page_id) {
            self.key_cache[page_id] = None;
            self.value_cache[page_id] = None;
            self.free_pages.push(page_id);
        }
    }

    /// Validate that `shape` can address this cache's page geometry.
    fn validate_page_shape(&self, shape: &[usize], kind: &str) -> Result<()> {
        let expected = [self.num_heads, self.page_size, self.head_dim];
        if shape.len() < 3 || shape[shape.len() - 3..] != expected {
            return Err(TrustformersError::shape_error(format!(
                "{} page shape {:?} must end with [num_heads, page_size, head_dim] = {:?}",
                kind, shape, expected
            )));
        }
        Ok(())
    }

    /// Store key data in a page.
    pub fn store_key(&mut self, page_id: usize, data: ArrayD<f32>) -> Result<()> {
        self.validate_page_shape(data.shape(), "key")?;
        if page_id >= self.key_cache.len() {
            return Err(TrustformersError::invalid_config(
                "Page ID out of bounds".into(),
            ));
        }
        self.key_cache[page_id] = Some(data);
        Ok(())
    }

    /// Store value data in a page.
    pub fn store_value(&mut self, page_id: usize, data: ArrayD<f32>) -> Result<()> {
        self.validate_page_shape(data.shape(), "value")?;
        if page_id >= self.value_cache.len() {
            return Err(TrustformersError::invalid_config(
                "Page ID out of bounds".into(),
            ));
        }
        self.value_cache[page_id] = Some(data);
        Ok(())
    }

    /// Retrieve key data from a page.
    pub fn get_key(&self, page_id: usize) -> Option<&ArrayD<f32>> {
        self.key_cache.get(page_id)?.as_ref()
    }

    /// Retrieve value data from a page.
    pub fn get_value(&self, page_id: usize) -> Option<&ArrayD<f32>> {
        self.value_cache.get(page_id)?.as_ref()
    }

    /// Borrow a key page for writing, zero-filling it on first use.
    pub fn key_page_mut(&mut self, page_id: usize, batch_size: usize) -> Result<&mut ArrayD<f32>> {
        let shape = [batch_size, self.num_heads, self.page_size, self.head_dim];
        Self::page_slot_mut(&mut self.key_cache, page_id, &shape, "key")
    }

    /// Borrow a value page for writing, zero-filling it on first use.
    pub fn value_page_mut(
        &mut self,
        page_id: usize,
        batch_size: usize,
    ) -> Result<&mut ArrayD<f32>> {
        let shape = [batch_size, self.num_heads, self.page_size, self.head_dim];
        Self::page_slot_mut(&mut self.value_cache, page_id, &shape, "value")
    }

    fn page_slot_mut<'a>(
        slots: &'a mut [Option<ArrayD<f32>>],
        page_id: usize,
        shape: &[usize],
        kind: &str,
    ) -> Result<&'a mut ArrayD<f32>> {
        let slot = slots.get_mut(page_id).ok_or_else(|| {
            TrustformersError::invalid_config(format!("{} page id {} out of bounds", kind, page_id))
        })?;

        match slot.as_ref() {
            Some(existing) if existing.shape() != shape => {
                return Err(TrustformersError::shape_error(format!(
                    "{} page {} holds shape {:?} but {:?} was requested",
                    kind,
                    page_id,
                    existing.shape(),
                    shape
                )));
            },
            Some(_) => {},
            None => *slot = Some(ArrayD::zeros(IxDyn(shape))),
        }

        slot.as_mut().ok_or_else(|| {
            TrustformersError::runtime_error(format!(
                "{} page {} vanished after init",
                kind, page_id
            ))
        })
    }

    /// Get number of available pages.
    pub fn available_pages(&self) -> usize {
        self.free_pages.len()
    }
}

impl PagedAttention {
    /// Build a paged-attention layer.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        dropout_prob: f32,
        bias: bool,
        page_size: usize,
        max_pages: usize,
    ) -> Result<Self> {
        if !hidden_size.is_multiple_of(num_heads) {
            return Err(TrustformersError::invalid_config(format!(
                "hidden_size {} must be divisible by num_heads {}",
                hidden_size, num_heads
            )));
        }
        if page_size == 0 {
            return Err(TrustformersError::invalid_config(
                "page_size must be greater than zero".into(),
            ));
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

    fn tables_read(&self) -> Result<std::sync::RwLockReadGuard<'_, HashMap<usize, BlockTable>>> {
        self.block_tables
            .read()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock poisoned: {}", e)))
    }

    fn tables_write(&self) -> Result<std::sync::RwLockWriteGuard<'_, HashMap<usize, BlockTable>>> {
        self.block_tables
            .write()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock poisoned: {}", e)))
    }

    fn cache_read(&self) -> Result<std::sync::RwLockReadGuard<'_, KVCache>> {
        self.kv_cache
            .read()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock poisoned: {}", e)))
    }

    fn cache_write(&self) -> Result<std::sync::RwLockWriteGuard<'_, KVCache>> {
        self.kv_cache
            .write()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock poisoned: {}", e)))
    }

    /// Allocate pages for a new sequence.
    ///
    /// Idempotent and incremental: calling it again with a larger
    /// `estimated_length` appends the missing pages instead of reallocating.
    pub fn allocate_sequence(&self, sequence_id: usize, estimated_length: usize) -> Result<()> {
        self.ensure_capacity(sequence_id, estimated_length)
    }

    /// Make sure `sequence_id` owns enough pages to hold `tokens` tokens.
    fn ensure_capacity(&self, sequence_id: usize, tokens: usize) -> Result<()> {
        let needed_pages = tokens.div_ceil(self.page_size);
        let mut tables = self.tables_write()?;
        let table = tables.entry(sequence_id).or_default();
        if table.pages.len() >= needed_pages {
            return Ok(());
        }

        let mut cache = self.cache_write()?;
        let mut fresh = Vec::with_capacity(needed_pages - table.pages.len());
        while table.pages.len() + fresh.len() < needed_pages {
            match cache.allocate_page() {
                Some(page_id) => fresh.push(page_id),
                None => {
                    for page_id in fresh {
                        cache.free_page(page_id);
                    }
                    return Err(TrustformersError::resource_exhausted(format!(
                        "Not enough pages available: {} more needed for sequence {}",
                        needed_pages - table.pages.len(),
                        sequence_id
                    )));
                },
            }
        }
        table.pages.extend(fresh);
        Ok(())
    }

    /// Free all pages for a sequence.
    pub fn free_sequence(&self, sequence_id: usize) {
        let removed = self
            .block_tables
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&sequence_id);
        if let Some(table) = removed {
            let mut cache = self.kv_cache.write().unwrap_or_else(|poisoned| poisoned.into_inner());
            for page_id in table.pages {
                cache.free_page(page_id);
            }
        }
    }

    /// Number of tokens currently cached for `sequence_id`.
    pub fn sequence_length(&self, sequence_id: usize) -> Result<usize> {
        Ok(self.tables_read()?.get(&sequence_id).map_or(0, BlockTable::len))
    }

    /// Snapshot of the block table of `sequence_id`, if the sequence exists.
    pub fn block_table(&self, sequence_id: usize) -> Result<Option<BlockTable>> {
        Ok(self.tables_read()?.get(&sequence_id).cloned())
    }

    /// Materialise the cached keys and values of `sequence_id` as dense
    /// `[batch, num_heads, length, head_dim]` tensors.
    ///
    /// Returns `Ok(None)` when nothing has been cached for the sequence yet.
    /// This is a debugging/inspection helper - the attention path reads the
    /// pages in place and never builds these arrays.
    pub fn cached_kv(&self, sequence_id: usize) -> Result<Option<(Tensor, Tensor)>> {
        let tables = self.tables_read()?;
        let Some(table) = tables.get(&sequence_id) else {
            return Ok(None);
        };
        if table.length == 0 {
            return Ok(None);
        }
        let cache = self.cache_read()?;
        let first_page = table.page_for(0, self.page_size)?;
        let Some(first) = cache.get_key(first_page) else {
            return Ok(None);
        };
        let batch = if first.ndim() == 4 { first.shape()[0] } else { 1 };
        let shape = IxDyn(&[batch, self.num_heads, table.length, self.head_dim]);
        let mut keys = ArrayD::<f32>::zeros(shape.clone());
        let mut values = ArrayD::<f32>::zeros(shape);

        for position in 0..table.length {
            let page_id = table.page_for(position, self.page_size)?;
            let slot = position % self.page_size;
            let key_page = cache.get_key(page_id).ok_or_else(|| {
                TrustformersError::runtime_error(format!("key page {} is not resident", page_id))
            })?;
            let value_page = cache.get_value(page_id).ok_or_else(|| {
                TrustformersError::runtime_error(format!("value page {} is not resident", page_id))
            })?;
            for b in 0..batch {
                for h in 0..self.num_heads {
                    for d in 0..self.head_dim {
                        keys[[b, h, position, d]] = key_page[[b, h, slot, d]];
                        values[[b, h, position, d]] = value_page[[b, h, slot, d]];
                    }
                }
            }
        }

        Ok(Some((Tensor::F32(keys), Tensor::F32(values))))
    }

    /// Split tensor into heads: `[batch, seq_len, hidden] -> [batch, num_heads, seq_len, head_dim]`.
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

                let reshaped = arr
                    .as_standard_layout()
                    .into_owned()
                    .into_shape_with_order(IxDyn(&[
                        batch_size,
                        seq_len,
                        self.num_heads,
                        self.head_dim,
                    ]))
                    .map_err(|_| {
                        TrustformersError::shape_error("Failed to reshape in split_heads".into())
                    })?;

                Ok(Tensor::F32(reshaped.permuted_axes(vec![0, 2, 1, 3])))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type",
                "PagedAttention::split_heads",
            )),
        }
    }

    /// Merge heads back: `[batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden]`.
    fn merge_heads(&self, tensor: &Tensor) -> Result<Tensor> {
        let shape = tensor.shape();

        match tensor {
            Tensor::F32(arr) => {
                let batch_size = shape[0];
                let seq_len = shape[2];

                let transposed = arr.view().permuted_axes(vec![0, 2, 1, 3]);

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

    /// Paged attention over the persistent KV cache.
    ///
    /// `k`/`v` carry the **new** tokens, whose absolute positions in the
    /// sequence are `position .. position + kv_len`; they are written into the
    /// sequence's pages before attention runs. `q` carries the queries for the
    /// last `q_len` of those positions, and each query attends over every
    /// cached token up to and including its own absolute position.
    ///
    /// With `q_len == kv_len == 1` this is one decode step against the whole
    /// cached prefix; with `position == 0` and `q_len == kv_len` it is a
    /// prefill. Both produce identical results to dense causal attention over
    /// the concatenated sequence.
    pub fn paged_attention_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        sequence_id: usize,
        position: usize,
    ) -> Result<Tensor> {
        self.paged_attention_forward_masked(q, k, v, sequence_id, position, None)
    }

    /// [`PagedAttention::paged_attention_forward`] with an additional attention
    /// mask broadcast over `[batch, num_heads, q_len, context_len]`, where
    /// `context_len == position + kv_len` is the cached prefix length.
    pub fn paged_attention_forward_masked(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        sequence_id: usize,
        position: usize,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (q_arr, k_arr, v_arr) = match (q, k, v) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr), Tensor::F32(v_arr)) => (q_arr, k_arr, v_arr),
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Unsupported tensor types for paged attention",
                    "PagedAttention::paged_attention_forward",
                ));
            },
        };

        for (name, arr) in [("query", q_arr), ("key", k_arr), ("value", v_arr)] {
            if arr.ndim() != 4 {
                return Err(TrustformersError::shape_error(format!(
                    "PagedAttention expects a 4-D {} tensor [batch, heads, seq, head_dim], got {:?}",
                    name,
                    arr.shape()
                )));
            }
        }

        let q_shape = q_arr.shape().to_vec();
        let (batch, heads, q_len, head_dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
        if k_arr.shape() != v_arr.shape() {
            return Err(TrustformersError::shape_error(format!(
                "key shape {:?} and value shape {:?} must match",
                k_arr.shape(),
                v_arr.shape()
            )));
        }
        let kv_shape = k_arr.shape().to_vec();
        if kv_shape[0] != batch || kv_shape[1] != heads || kv_shape[3] != head_dim {
            return Err(TrustformersError::shape_error(format!(
                "query shape {:?} is incompatible with key/value shape {:?}",
                q_shape, kv_shape
            )));
        }
        if heads != self.num_heads || head_dim != self.head_dim {
            return Err(TrustformersError::shape_error(format!(
                "PagedAttention was built for {} heads of width {}, got {} heads of width {}",
                self.num_heads, self.head_dim, heads, head_dim
            )));
        }
        let kv_len = kv_shape[2];
        if kv_len == 0 {
            return Err(TrustformersError::shape_error(
                "PagedAttention needs at least one new key/value token".into(),
            ));
        }
        if q_len > kv_len {
            return Err(TrustformersError::shape_error(format!(
                "query length {} may not exceed the {} new key/value tokens",
                q_len, kv_len
            )));
        }

        let context_len = position + kv_len;
        let query_start = context_len - q_len;
        self.ensure_capacity(sequence_id, context_len)?;
        self.write_kv_pages(sequence_id, position, batch, heads, head_dim, k_arr, v_arr)?;

        let mask_view = match attention_mask {
            Some(mask) => Some(MaskView::new(mask, batch, heads, q_len, context_len)?),
            None => None,
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = ArrayD::<f32>::zeros(IxDyn(&[batch, heads, q_len, head_dim]));
        let mut scores = vec![0.0f32; context_len];
        let mut query = vec![0.0f32; head_dim];

        let tables = self.tables_read()?;
        let table = tables.get(&sequence_id).ok_or_else(|| {
            crate::errors::compute_error(
                "paged_attention_forward",
                "sequence must have allocated pages",
            )
        })?;
        let cache = self.cache_read()?;

        for b in 0..batch {
            for h in 0..heads {
                for qi in 0..q_len {
                    // Every query sees the cached prefix up to its own position.
                    let visible = query_start + qi + 1;
                    for (d, slot) in query.iter_mut().enumerate() {
                        *slot = q_arr[[b, h, qi, d]];
                    }

                    let mut position_cursor = 0usize;
                    while position_cursor < visible {
                        let page_id = table.page_for(position_cursor, self.page_size)?;
                        let page_slot = position_cursor % self.page_size;
                        let take = (self.page_size - page_slot).min(visible - position_cursor);
                        let key_page = cache.get_key(page_id).ok_or_else(|| {
                            TrustformersError::runtime_error(format!(
                                "key page {} is not resident",
                                page_id
                            ))
                        })?;
                        for t in 0..take {
                            let mut dot = 0.0f32;
                            for (d, &qd) in query.iter().enumerate() {
                                dot += qd * key_page[[b, h, page_slot + t, d]];
                            }
                            scores[position_cursor + t] = dot * scale;
                        }
                        position_cursor += take;
                    }

                    if let Some(view) = mask_view.as_ref() {
                        for (key_index, score) in scores.iter_mut().take(visible).enumerate() {
                            let penalty = view.additive(b, h, qi, key_index);
                            if penalty != 0.0 {
                                *score += penalty;
                            }
                        }
                    }

                    let max_score =
                        scores[..visible].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    if max_score == f32::NEG_INFINITY {
                        // Every visible position is masked out: the row stays zero.
                        continue;
                    }
                    if !max_score.is_finite() {
                        return Err(TrustformersError::runtime_error(
                            "paged attention produced a non-finite score".to_string(),
                        ));
                    }

                    let mut sum = 0.0f32;
                    for score in scores.iter_mut().take(visible) {
                        let weight = (*score - max_score).exp();
                        *score = weight;
                        sum += weight;
                    }
                    if !(sum > 0.0 && sum.is_finite()) {
                        return Err(TrustformersError::runtime_error(
                            "paged attention softmax denominator is not usable".to_string(),
                        ));
                    }
                    let inv_sum = 1.0 / sum;

                    let mut position_cursor = 0usize;
                    while position_cursor < visible {
                        let page_id = table.page_for(position_cursor, self.page_size)?;
                        let page_slot = position_cursor % self.page_size;
                        let take = (self.page_size - page_slot).min(visible - position_cursor);
                        let value_page = cache.get_value(page_id).ok_or_else(|| {
                            TrustformersError::runtime_error(format!(
                                "value page {} is not resident",
                                page_id
                            ))
                        })?;
                        for t in 0..take {
                            let weight = scores[position_cursor + t] * inv_sum;
                            if weight == 0.0 {
                                continue;
                            }
                            for d in 0..head_dim {
                                output[[b, h, qi, d]] +=
                                    weight * value_page[[b, h, page_slot + t, d]];
                            }
                        }
                        position_cursor += take;
                    }
                }
            }
        }

        Ok(Tensor::F32(output))
    }

    /// Copy the new keys/values into the sequence's pages and advance its length.
    #[allow(clippy::too_many_arguments)]
    fn write_kv_pages(
        &self,
        sequence_id: usize,
        position: usize,
        batch: usize,
        heads: usize,
        head_dim: usize,
        k_arr: &ArrayD<f32>,
        v_arr: &ArrayD<f32>,
    ) -> Result<()> {
        let kv_len = k_arr.shape()[2];
        let mut tables = self.tables_write()?;
        let table = tables.get_mut(&sequence_id).ok_or_else(|| {
            crate::errors::compute_error("paged_attention_forward", "sequence has no block table")
        })?;
        let mut cache = self.cache_write()?;

        let mut written = 0usize;
        while written < kv_len {
            let absolute = position + written;
            let page_id = table.page_for(absolute, self.page_size)?;
            let page_slot = absolute % self.page_size;
            let take = (self.page_size - page_slot).min(kv_len - written);

            {
                let key_page = cache.key_page_mut(page_id, batch)?;
                for t in 0..take {
                    for b in 0..batch {
                        for h in 0..heads {
                            for d in 0..head_dim {
                                key_page[[b, h, page_slot + t, d]] = k_arr[[b, h, written + t, d]];
                            }
                        }
                    }
                }
            }
            {
                let value_page = cache.value_page_mut(page_id, batch)?;
                for t in 0..take {
                    for b in 0..batch {
                        for h in 0..heads {
                            for d in 0..head_dim {
                                value_page[[b, h, page_slot + t, d]] =
                                    v_arr[[b, h, written + t, d]];
                            }
                        }
                    }
                }
            }

            written += take;
        }

        table.length = table.length.max(position + kv_len);
        Ok(())
    }

    /// Get memory usage statistics.
    pub fn memory_stats(&self) -> MemoryStats {
        let block_tables =
            self.block_tables.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        let kv_cache = self.kv_cache.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        MemoryStats {
            total_pages: self.max_pages,
            used_pages: self.max_pages - kv_cache.available_pages(),
            available_pages: kv_cache.available_pages(),
            page_size: self.page_size,
            active_sequences: block_tables.len(),
        }
    }
}

/// Page-level occupancy of a [`PagedAttention`] cache.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_pages: usize,
    pub used_pages: usize,
    pub available_pages: usize,
    pub page_size: usize,
    pub active_sequences: usize,
}

/// Input bundle for [`PagedAttention`] as a [`Layer`].
#[derive(Debug, Clone)]
pub struct PagedAttentionInput {
    /// `[batch, seq_len, hidden]` (a 2-D `[seq_len, hidden]` input is accepted
    /// and the batch axis is restored on the way out).
    pub hidden_states: Tensor,
    /// Identifies the cached sequence these tokens belong to.
    pub sequence_id: usize,
    /// Absolute position of the first token of `hidden_states`.
    pub position: usize,
    /// Optional mask broadcast over `[batch, heads, seq_len, position + seq_len]`.
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

        // Apply PagedAttention against the persistent cache
        let context = self.paged_attention_forward_masked(
            &query_states,
            &key_states,
            &value_states,
            input.sequence_id,
            input.position,
            input.attention_mask.as_ref(),
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

    /// Dense causal attention reference, used to pin the paged path down.
    ///
    /// `q_offset` is the absolute position of the first query row.
    fn reference_causal_attention(
        q: &ArrayD<f32>,
        k: &ArrayD<f32>,
        v: &ArrayD<f32>,
        q_offset: usize,
    ) -> ArrayD<f32> {
        let (batch, heads, q_len, head_dim) =
            (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
        let kv_len = k.shape()[2];
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = ArrayD::<f32>::zeros(IxDyn(&[batch, heads, q_len, head_dim]));

        for b in 0..batch {
            for h in 0..heads {
                for qi in 0..q_len {
                    let visible = (q_offset + qi + 1).min(kv_len);
                    let mut scores = vec![0.0f32; visible];
                    for (ki, score) in scores.iter_mut().enumerate() {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[[b, h, qi, d]] * k[[b, h, ki, d]];
                        }
                        *score = dot * scale;
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for score in scores.iter_mut() {
                        *score = (*score - max).exp();
                        sum += *score;
                    }
                    for (ki, score) in scores.iter().enumerate() {
                        let weight = score / sum;
                        for d in 0..head_dim {
                            out[[b, h, qi, d]] += weight * v[[b, h, ki, d]];
                        }
                    }
                }
            }
        }
        out
    }

    fn ramp(shape: &[usize], start: f32, step: f32) -> ArrayD<f32> {
        let total: usize = shape.iter().product();
        let data: Vec<f32> = (0..total).map(|i| start + step * i as f32).collect();
        ArrayD::from_shape_vec(IxDyn(shape), data).expect("ramp shape")
    }

    fn max_abs_diff(a: &ArrayD<f32>, b: &ArrayD<f32>) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    #[test]
    fn test_paged_attention_creation() {
        let paged_attn = PagedAttention::new(768, 12, 0.1, true, 64, 1000);
        assert!(paged_attn.is_ok());

        let paged_attn = paged_attn.expect("Failed to create PagedAttention");
        assert_eq!(paged_attn.num_heads, 12);
        assert_eq!(paged_attn.hidden_size, 768);
        assert_eq!(paged_attn.head_dim, 64);
        assert_eq!(paged_attn.page_size, 64);
        assert_eq!(paged_attn.max_pages, 1000);
    }

    #[test]
    fn zero_page_size_is_rejected() {
        assert!(PagedAttention::new(64, 4, 0.0, false, 0, 8).is_err());
    }

    #[test]
    fn test_kv_cache_operations() {
        let mut cache = KVCache::new(10, 64, 8, 64);

        // Test allocation
        let page1 = cache.allocate_page();
        assert!(page1.is_some());
        let page1 = page1.expect("Failed to allocate page1");

        let page2 = cache.allocate_page();
        assert!(page2.is_some());
        let page2 = page2.expect("Failed to allocate page2");

        assert_ne!(page1, page2);
        assert_eq!(cache.available_pages(), 8);

        // Test storage and retrieval
        let test_data = ArrayD::from_elem(IxDyn(&[8, 64, 64]), 1.0f32);
        cache.store_key(page1, test_data.clone()).expect("Failed to store key");

        let retrieved = cache.get_key(page1);
        assert!(retrieved.is_some());

        // Test freeing
        cache.free_page(page1);
        assert_eq!(cache.available_pages(), 9);
        assert!(cache.get_key(page1).is_none());
    }

    #[test]
    fn store_key_rejects_a_shape_that_does_not_match_the_page_geometry() {
        let mut cache = KVCache::new(4, 8, 2, 3);
        let page = cache.allocate_page().expect("page");
        let wrong = ArrayD::from_elem(IxDyn(&[2, 4, 3]), 0.0f32);
        assert!(cache.store_key(page, wrong).is_err());
        let right = ArrayD::from_elem(IxDyn(&[1, 2, 8, 3]), 0.0f32);
        assert!(cache.store_key(page, right).is_ok());
    }

    #[test]
    fn test_sequence_allocation() {
        let paged_attn =
            PagedAttention::new(256, 8, 0.0, true, 32, 100).expect("operation failed in test");

        // Allocate sequence
        let result = paged_attn.allocate_sequence(1, 128);
        assert!(result.is_ok());

        // Check block table
        let table = paged_attn.block_table(1).expect("lock").expect("sequence exists");
        assert_eq!(table.pages().len(), 4); // 128 / 32 = 4 pages
        assert!(table.is_empty()); // nothing written yet

        // Growing the sequence appends pages instead of reallocating
        paged_attn.allocate_sequence(1, 160).expect("grow");
        let grown = paged_attn.block_table(1).expect("lock").expect("sequence exists");
        assert_eq!(grown.pages().len(), 5);
        assert_eq!(&grown.pages()[..4], table.pages());

        // Free sequence
        paged_attn.free_sequence(1);
        assert!(paged_attn.block_table(1).expect("lock").is_none());
        assert_eq!(paged_attn.memory_stats().available_pages, 100);
    }

    #[test]
    fn test_paged_attention_forward() {
        let paged_attn = PagedAttention::new(256, 8, 0.0, true, 32, 100)
            .expect("Failed to create PagedAttention");

        let hidden_states = Tensor::randn(&[1, 64, 256]).expect("Failed to create random tensor");
        let input = PagedAttentionInput {
            hidden_states,
            sequence_id: 1,
            position: 0,
            attention_mask: None,
        };

        let output = paged_attn.forward(input);
        assert!(output.is_ok());

        let output = output.expect("Forward pass failed");
        assert_eq!(output.shape(), vec![1, 64, 256]);
        assert_eq!(paged_attn.sequence_length(1).expect("lock"), 64);
    }

    #[test]
    fn test_memory_stats() {
        let paged_attn = PagedAttention::new(256, 8, 0.0, true, 32, 100)
            .expect("Failed to create PagedAttention");

        let stats = paged_attn.memory_stats();
        assert_eq!(stats.total_pages, 100);
        assert_eq!(stats.used_pages, 0);
        assert_eq!(stats.available_pages, 100);
        assert_eq!(stats.page_size, 32);
        assert_eq!(stats.active_sequences, 0);

        // Allocate a sequence
        paged_attn.allocate_sequence(1, 128).expect("Failed to allocate sequence");

        let stats = paged_attn.memory_stats();
        assert_eq!(stats.used_pages, 4);
        assert_eq!(stats.available_pages, 96);
        assert_eq!(stats.active_sequences, 1);
    }

    /// Regression test for the fake paged path: the previous implementation
    /// never touched `kv_cache`, so the pages stayed `None` after a forward.
    #[test]
    fn prefill_writes_the_new_keys_and_values_into_the_pages() {
        let heads = 2;
        let head_dim = 3;
        let attention = PagedAttention::new(heads * head_dim, heads, 0.0, false, 4, 8)
            .expect("construct PagedAttention");

        let shape = [1, heads, 5, head_dim];
        let q = ramp(&shape, 0.01, 0.01);
        let k = ramp(&shape, -0.2, 0.03);
        let v = ramp(&shape, 0.5, -0.02);

        attention
            .paged_attention_forward(
                &Tensor::F32(q),
                &Tensor::F32(k.clone()),
                &Tensor::F32(v.clone()),
                7,
                0,
            )
            .expect("paged forward");

        assert_eq!(attention.sequence_length(7).expect("lock"), 5);
        let (cached_k, cached_v) = attention
            .cached_kv(7)
            .expect("lock")
            .expect("cache must be populated after a forward");
        match (cached_k, cached_v) {
            (Tensor::F32(ck), Tensor::F32(cv)) => {
                assert_eq!(ck.shape(), &shape[..]);
                assert!(
                    max_abs_diff(&ck, &k) < 1e-6,
                    "cached keys differ from input"
                );
                assert!(
                    max_abs_diff(&cv, &v) < 1e-6,
                    "cached values differ from input"
                );
            },
            _ => panic!("cached kv must be F32"),
        }
    }

    /// The decisive regression test: a decode step must attend over the cached
    /// prefix. The old implementation ran attention over the single new token
    /// only, which made the output exactly the new value vector.
    #[test]
    fn decode_step_attends_over_the_cached_prefix() {
        let heads = 2;
        let head_dim = 4;
        // page_size 3 forces the 5-token prefix to straddle two pages.
        let attention = PagedAttention::new(heads * head_dim, heads, 0.0, false, 3, 16)
            .expect("construct PagedAttention");

        let prefix_len = 5;
        let prefix_shape = [1, heads, prefix_len, head_dim];
        let q_prefix = ramp(&prefix_shape, 0.02, 0.011);
        let k_prefix = ramp(&prefix_shape, -0.3, 0.017);
        let v_prefix = ramp(&prefix_shape, 0.7, -0.023);

        attention
            .paged_attention_forward(
                &Tensor::F32(q_prefix),
                &Tensor::F32(k_prefix.clone()),
                &Tensor::F32(v_prefix.clone()),
                3,
                0,
            )
            .expect("prefill");

        let step_shape = [1, heads, 1, head_dim];
        let q_step = ramp(&step_shape, 0.05, 0.03);
        let k_step = ramp(&step_shape, 0.13, -0.007);
        let v_step = ramp(&step_shape, -0.4, 0.05);

        let out = attention
            .paged_attention_forward(
                &Tensor::F32(q_step.clone()),
                &Tensor::F32(k_step.clone()),
                &Tensor::F32(v_step.clone()),
                3,
                prefix_len,
            )
            .expect("decode step");

        // Reference: dense causal attention over the concatenated 6 tokens.
        let mut full_k = ArrayD::<f32>::zeros(IxDyn(&[1, heads, prefix_len + 1, head_dim]));
        let mut full_v = ArrayD::<f32>::zeros(IxDyn(&[1, heads, prefix_len + 1, head_dim]));
        for h in 0..heads {
            for t in 0..prefix_len {
                for d in 0..head_dim {
                    full_k[[0, h, t, d]] = k_prefix[[0, h, t, d]];
                    full_v[[0, h, t, d]] = v_prefix[[0, h, t, d]];
                }
            }
            for d in 0..head_dim {
                full_k[[0, h, prefix_len, d]] = k_step[[0, h, 0, d]];
                full_v[[0, h, prefix_len, d]] = v_step[[0, h, 0, d]];
            }
        }
        let expected = reference_causal_attention(&q_step, &full_k, &full_v, prefix_len);

        match out {
            Tensor::F32(actual) => {
                assert_eq!(actual.shape(), &step_shape[..]);
                assert!(
                    max_abs_diff(&actual, &expected) < 1e-5,
                    "decode output {:?} does not match dense reference {:?}",
                    actual,
                    expected
                );
                // And it must NOT be the degenerate "attend to the new token only"
                // answer the old code produced.
                let degenerate = v_step.clone();
                assert!(
                    max_abs_diff(&actual, &degenerate) > 1e-3,
                    "decode output ignored the cached prefix"
                );
            },
            _ => panic!("output must be F32"),
        }
        assert_eq!(attention.sequence_length(3).expect("lock"), prefix_len + 1);
    }

    /// Token-by-token decoding must reproduce a single dense prefill exactly.
    #[test]
    fn incremental_decoding_matches_a_single_prefill() {
        let heads = 2;
        let head_dim = 3;
        let seq_len = 7;
        let attention = PagedAttention::new(heads * head_dim, heads, 0.0, false, 2, 32)
            .expect("construct PagedAttention");

        let shape = [1, heads, seq_len, head_dim];
        let q = ramp(&shape, 0.03, 0.009);
        let k = ramp(&shape, -0.25, 0.013);
        let v = ramp(&shape, 0.6, -0.019);

        let prefill = attention
            .paged_attention_forward(
                &Tensor::F32(q.clone()),
                &Tensor::F32(k.clone()),
                &Tensor::F32(v.clone()),
                11,
                0,
            )
            .expect("prefill");
        let prefill = match prefill {
            Tensor::F32(arr) => arr,
            _ => panic!("F32 expected"),
        };

        let mut stepwise = ArrayD::<f32>::zeros(IxDyn(&shape));
        for t in 0..seq_len {
            let mut q_step = ArrayD::<f32>::zeros(IxDyn(&[1, heads, 1, head_dim]));
            let mut k_step = ArrayD::<f32>::zeros(IxDyn(&[1, heads, 1, head_dim]));
            let mut v_step = ArrayD::<f32>::zeros(IxDyn(&[1, heads, 1, head_dim]));
            for h in 0..heads {
                for d in 0..head_dim {
                    q_step[[0, h, 0, d]] = q[[0, h, t, d]];
                    k_step[[0, h, 0, d]] = k[[0, h, t, d]];
                    v_step[[0, h, 0, d]] = v[[0, h, t, d]];
                }
            }
            let out = attention
                .paged_attention_forward(
                    &Tensor::F32(q_step),
                    &Tensor::F32(k_step),
                    &Tensor::F32(v_step),
                    12,
                    t,
                )
                .expect("decode step");
            let out = match out {
                Tensor::F32(arr) => arr,
                _ => panic!("F32 expected"),
            };
            for h in 0..heads {
                for d in 0..head_dim {
                    stepwise[[0, h, t, d]] = out[[0, h, 0, d]];
                }
            }
        }

        assert!(
            max_abs_diff(&prefill, &stepwise) < 1e-5,
            "incremental decoding diverged from prefill"
        );
    }

    /// Hand-computed check with one head of width 1, so softmax is exact on paper.
    #[test]
    fn matches_a_hand_computed_two_token_example() {
        let attention =
            PagedAttention::new(1, 1, 0.0, false, 4, 4).expect("construct PagedAttention");

        // head_dim = 1 => scale = 1. Keys [1, 2], values [10, 20], query [1].
        let q = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0f32]).expect("q");
        let k = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 1]), vec![1.0f32, 2.0]).expect("k");
        let v = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 1]), vec![10.0f32, 20.0]).expect("v");

        let out = attention
            .paged_attention_forward(&Tensor::F32(q), &Tensor::F32(k), &Tensor::F32(v), 1, 0)
            .expect("forward");

        // scores = [1, 2]; softmax = [1/(1+e), e/(1+e)]
        let e = 1.0f32.exp();
        let expected = (10.0 + 20.0 * e) / (1.0 + e);
        match out {
            Tensor::F32(arr) => {
                assert_eq!(arr.shape(), &[1, 1, 1, 1]);
                assert!(
                    (arr[[0, 0, 0, 0]] - expected).abs() < 1e-5,
                    "got {}, expected {}",
                    arr[[0, 0, 0, 0]],
                    expected
                );
            },
            _ => panic!("F32 expected"),
        }
    }

    /// The `position` argument used to be ignored entirely.
    #[test]
    fn position_changes_the_attended_context() {
        let heads = 1;
        let head_dim = 2;
        let attention = PagedAttention::new(heads * head_dim, heads, 0.0, false, 4, 16)
            .expect("construct PagedAttention");

        let prefix_shape = [1, heads, 4, head_dim];
        let q_prefix = ramp(&prefix_shape, 0.1, 0.05);
        let k_prefix = ramp(&prefix_shape, 0.2, 0.07);
        let v_prefix = ramp(&prefix_shape, 1.0, 0.3);
        for sequence_id in [21usize, 22] {
            attention
                .paged_attention_forward(
                    &Tensor::F32(q_prefix.clone()),
                    &Tensor::F32(k_prefix.clone()),
                    &Tensor::F32(v_prefix.clone()),
                    sequence_id,
                    0,
                )
                .expect("prefill");
        }

        let step_shape = [1, heads, 1, head_dim];
        let q_step = ramp(&step_shape, 0.4, 0.1);
        let k_step = ramp(&step_shape, 0.9, 0.1);
        let v_step = ramp(&step_shape, -1.0, 0.4);

        let at_end = attention
            .paged_attention_forward(
                &Tensor::F32(q_step.clone()),
                &Tensor::F32(k_step.clone()),
                &Tensor::F32(v_step.clone()),
                21,
                4,
            )
            .expect("append at position 4");
        let overwriting = attention
            .paged_attention_forward(
                &Tensor::F32(q_step),
                &Tensor::F32(k_step),
                &Tensor::F32(v_step),
                22,
                0,
            )
            .expect("write at position 0");

        match (at_end, overwriting) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                assert!(
                    max_abs_diff(&a, &b) > 1e-3,
                    "position must change the attended context"
                );
            },
            _ => panic!("F32 expected"),
        }
    }

    #[test]
    fn a_keep_mask_removes_the_masked_key_positions() {
        let heads = 1;
        let head_dim = 1;
        let attention =
            PagedAttention::new(1, 1, 0.0, false, 4, 4).expect("construct PagedAttention");

        let q = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0f32]).expect("q");
        let k = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 1]), vec![1.0f32, 2.0]).expect("k");
        let v = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 1]), vec![10.0f32, 20.0]).expect("v");
        // Keep only the first key position.
        let mask =
            Tensor::F32(ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0f32, 0.0]).expect("mask"));

        let out = attention
            .paged_attention_forward_masked(
                &Tensor::F32(q),
                &Tensor::F32(k),
                &Tensor::F32(v),
                5,
                0,
                Some(&mask),
            )
            .expect("masked forward");

        let _ = (heads, head_dim);
        match out {
            Tensor::F32(arr) => {
                assert!(
                    (arr[[0, 0, 0, 0]] - 10.0).abs() < 1e-5,
                    "masked-out key still contributed: {}",
                    arr[[0, 0, 0, 0]]
                );
            },
            _ => panic!("F32 expected"),
        }
    }

    #[test]
    fn running_out_of_pages_is_reported_not_silently_truncated() {
        let attention =
            PagedAttention::new(4, 2, 0.0, false, 2, 2).expect("construct PagedAttention");
        // 2 pages x 2 tokens = 4 tokens of capacity; ask for 6.
        let shape = [1, 2, 6, 2];
        let q = ramp(&shape, 0.1, 0.01);
        let err = attention.paged_attention_forward(
            &Tensor::F32(q.clone()),
            &Tensor::F32(q.clone()),
            &Tensor::F32(q),
            1,
            0,
        );
        assert!(err.is_err(), "over-long sequence must be refused");
    }

    #[test]
    fn shape_mismatches_are_refused() {
        let attention =
            PagedAttention::new(4, 2, 0.0, false, 4, 8).expect("construct PagedAttention");
        let q = ramp(&[1, 2, 2, 2], 0.1, 0.01);
        let k = ramp(&[1, 2, 3, 2], 0.1, 0.01);
        let v = ramp(&[1, 2, 2, 2], 0.1, 0.01);
        assert!(attention
            .paged_attention_forward(
                &Tensor::F32(q.clone()),
                &Tensor::F32(k),
                &Tensor::F32(v),
                1,
                0
            )
            .is_err());

        let three_d = ramp(&[2, 2, 2], 0.1, 0.01);
        assert!(attention
            .paged_attention_forward(
                &Tensor::F32(three_d.clone()),
                &Tensor::F32(three_d.clone()),
                &Tensor::F32(three_d),
                1,
                0
            )
            .is_err());
    }
}
