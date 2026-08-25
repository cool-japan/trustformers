//! # MetalBackend - attention_with_cache_gpu_to_gpu_group Methods
//!
//! This module contains method implementations for `MetalBackend`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
use super::common::*;

use super::metalbackend_type::MetalBackend;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
use super::types::{BufferCache, BufferId};

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalBackend {
    /// Execute multi-head attention with KV-cache on GPU (supports different Q vs K/V seq lengths)
    ///
    /// This version takes pre-reshaped tensors in multi-head format and supports
    /// different sequence lengths for Q (current tokens) vs K/V (cached + current).
    ///
    /// # Arguments
    ///
    /// * `q_heads_id` - Query tensor: [batch, num_heads, q_seq_len, head_dim]
    /// * `k_heads_id` - Key tensor: [batch, num_heads, kv_seq_len, head_dim]
    /// * `v_heads_id` - Value tensor: [batch, num_heads, kv_seq_len, head_dim]
    /// * `batch_size` - Batch size
    /// * `q_seq_len` - Query sequence length (typically 1 during generation)
    /// * `kv_seq_len` - Key/Value sequence length (cached + new tokens)
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    ///
    /// # Returns
    ///
    /// Buffer ID containing output: [batch, num_heads, q_seq_len, head_dim]
    pub fn attention_with_cache_gpu_to_gpu(
        &self,
        q_heads_id: &BufferId,
        k_heads_id: &BufferId,
        v_heads_id: &BufferId,
        batch_size: usize,
        q_seq_len: usize,
        kv_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<BufferId> {
        tracing::trace!(
            batch_size,
            q_seq_len,
            kv_seq_len,
            num_heads,
            head_dim,
            "metal: cached multi-head attention (gpu-to-gpu)"
        );
        if batch_size != 1 {
            return Err(TrustformersError::tensor_op_error(
                "GPU cached attention currently only supports batch_size=1",
                "attention_with_cache_gpu_to_gpu",
            ));
        }
        if q_seq_len == 0 || kv_seq_len == 0 || num_heads == 0 || head_dim == 0 {
            return Err(TrustformersError::tensor_op_error(
                "GPU cached attention requires non-zero q_seq_len, kv_seq_len, \
                 num_heads and head_dim",
                "attention_with_cache_gpu_to_gpu",
            ));
        }
        if q_seq_len > kv_seq_len {
            return Err(TrustformersError::tensor_op_error(
                "GPU cached attention requires q_seq_len <= kv_seq_len (the KV cache \
                 always contains at least the current query positions)",
                "attention_with_cache_gpu_to_gpu",
            ));
        }
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Both intermediates are dead once the output exists; release them on every
        // exit path so a decode loop does not accumulate two GPU buffers per token.
        let mut scratch: Vec<BufferId> = Vec::with_capacity(2);
        let result = (|| -> Result<BufferId> {
            let k_heads_t =
                self.batched_transpose_gpu_to_gpu(k_heads_id, num_heads, kv_seq_len, head_dim)?;
            scratch.push(k_heads_t);

            // Kernel selection is a causality question, not a shape-convenience one.
            // The query block always covers the LAST `q_seq_len` positions of the key
            // sequence, so query row i sits at absolute key position
            // `(kv_seq_len - q_seq_len) + i` and may attend to key columns
            // `0..=(kv_seq_len - q_seq_len) + i`.
            let attn_weights = if q_seq_len == kv_seq_len {
                // Full prefill: the offset is zero, which is exactly the plain causal
                // mask the fused causal kernel applies.
                self.batched_scaled_matmul_softmax_causal_gpu_to_gpu(
                    q_heads_id, &k_heads_t, num_heads, q_seq_len, head_dim, scale,
                )?
            } else if q_seq_len == 1 {
                // Single-token decode: the one query row is the last key position, so
                // every key column is in its past and the unmasked generation kernel is
                // already correct (and is the hot path, so leave it untouched).
                self.batched_scaled_matmul_softmax_gen_gpu_to_gpu(
                    q_heads_id, &k_heads_t, num_heads, q_seq_len, kv_seq_len, head_dim, scale,
                )?
            } else {
                // Chunked continuation (1 < q_seq_len < kv_seq_len): a block of new
                // tokens against a warm cache. Routing this to the UNMASKED generation
                // kernel - which is what this selection used to do - let every row of
                // the chunk attend to the later rows of its own chunk: measured on
                // (q_seq 3, kv_seq 5) as an exact match to a non-causal reference and a
                // 0.29 miss (on |signal| 0.9) against the causal one. The offset-masked
                // kernel carries the cached prefix length so each row stops at its own
                // absolute position.
                self.batched_scaled_matmul_softmax_gen_causal_gpu_to_gpu(
                    q_heads_id,
                    &k_heads_t,
                    num_heads,
                    q_seq_len,
                    kv_seq_len,
                    head_dim,
                    scale,
                    kv_seq_len - q_seq_len,
                )?
            };
            scratch.push(attn_weights);

            self.batched_matmul_gpu_to_gpu(
                &attn_weights,
                v_heads_id,
                num_heads,
                q_seq_len,
                kv_seq_len,
                head_dim,
            )
        })();

        self.release_buffers(&scratch)?;
        result
    }
    /// Fused scaled matmul + softmax with an **offset causal mask**.
    ///
    /// The masked sibling of
    /// [`batched_scaled_matmul_softmax_gen_gpu_to_gpu`](Self::batched_scaled_matmul_softmax_gen_gpu_to_gpu).
    /// Query row `i` is taken to be absolute key position `q_offset + i`, so it attends
    /// to key columns `0..=(q_offset + i)` and the rest of its row is zeroed.
    ///
    /// With `q_offset == 0` this is the ordinary causal mask; with
    /// `q_offset == kv_seq_len - q_seq_len` it is the mask a chunk of new tokens needs
    /// against a warm KV cache. The unmasked generation kernel coincides with this one
    /// only when `q_seq_len == 1`.
    ///
    /// # Arguments
    ///
    /// * `q_buffer_id` - Q: `[num_heads, q_seq_len, head_dim]`
    /// * `k_t_buffer_id` - K transposed: `[num_heads, head_dim, kv_seq_len]`
    /// * `alpha` - score scaling, normally `1/sqrt(head_dim)`
    /// * `q_offset` - absolute key position of query row 0
    ///
    /// # Returns
    ///
    /// Buffer ID containing attention weights: `[num_heads, q_seq_len, kv_seq_len]`
    pub fn batched_scaled_matmul_softmax_gen_causal_gpu_to_gpu(
        &self,
        q_buffer_id: &BufferId,
        k_t_buffer_id: &BufferId,
        num_heads: usize,
        q_seq_len: usize,
        kv_seq_len: usize,
        head_dim: usize,
        alpha: f32,
        q_offset: usize,
    ) -> Result<BufferId> {
        let q_buffer = self.get_persistent_buffer(q_buffer_id)?;
        let k_t_buffer = self.get_persistent_buffer(k_t_buffer_id)?;
        // Same contract as the two unmasked variants: the kernel is unbounded in
        // `kv_seq_len`, but the declared shape must match the operand buffers.
        Self::validate_fused_attention_shapes(
            &q_buffer,
            &k_t_buffer,
            num_heads,
            q_seq_len,
            kv_seq_len,
            head_dim,
            "batched_scaled_matmul_softmax_gen_causal_gpu_to_gpu",
        )?;
        // The last query row must land inside the key sequence. The kernel clamps
        // rather than reading out of bounds, but a caller who gets this wrong is asking
        // for a mask that does not exist, so say so instead of silently un-masking.
        let last_position = q_offset.checked_add(q_seq_len).ok_or_else(|| {
            TrustformersError::shape_error(format!(
                "batched_scaled_matmul_softmax_gen_causal_gpu_to_gpu: q_offset \
                 {q_offset} + q_seq_len {q_seq_len} overflows usize"
            ))
        })?;
        if last_position > kv_seq_len {
            return Err(TrustformersError::shape_error(format!(
                "batched_scaled_matmul_softmax_gen_causal_gpu_to_gpu: query rows \
                 {q_offset}..{last_position} do not fit in a key sequence of \
                 {kv_seq_len} positions"
            )));
        }

        // Shared, not Private: the kernel parks raw scores in this buffer and the
        // result is read back by tests and CPU fallbacks (see the causal variant).
        let output_bytes = num_heads * q_seq_len * kv_seq_len * mem::size_of::<f32>();
        self.validate_allocation_bytes(
            output_bytes,
            "batched_scaled_matmul_softmax_gen_causal_gpu_to_gpu",
        )?;
        let output_buffer = Arc::new(
            self.device
                .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared),
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.batched_scaled_matmul_softmax_gen_causal_pipeline);
        encoder.set_buffer(0, Some(&*q_buffer), 0);
        encoder.set_buffer(1, Some(&*k_t_buffer), 0);
        encoder.set_buffer(2, Some(&*output_buffer), 0);

        let num_heads_u32 = num_heads as u32;
        let q_seq_len_u32 = q_seq_len as u32;
        let kv_seq_len_u32 = kv_seq_len as u32;
        let head_dim_u32 = head_dim as u32;
        let q_offset_u32 = q_offset as u32;

        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &num_heads_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &q_seq_len_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &kv_seq_len_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            mem::size_of::<u32>() as u64,
            &head_dim_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            7,
            mem::size_of::<u32>() as u64,
            &alpha as *const f32 as *const _,
        );
        encoder.set_bytes(
            8,
            mem::size_of::<u32>() as u64,
            &q_offset_u32 as *const u32 as *const _,
        );

        // Dispatch: one thread per (q_row, head) pair.
        let threadgroup_size = metal::MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (q_seq_len as u64).div_ceil(64),
            height: num_heads as u64,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);

        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "batched_scaled_matmul_softmax_gen_causal_gpu_to_gpu",
            )
        })?;
        cache.insert(output_id, output_buffer);

        Ok(output_id)
    }
}

/// Causality regression suite for the cached-attention composition.
///
/// # What went wrong
///
/// `attention_with_cache_gpu_to_gpu` used to pick its softmax kernel by asking only
/// "are the two sequence lengths equal?", sending everything else to the *unmasked*
/// `batched_scaled_matmul_softmax_gen` kernel. That is correct for a single-token
/// decode and wrong for every longer query block against a warm KV cache: with
/// `1 < q_seq_len < kv_seq_len` each row of the chunk also attended to the later rows
/// of its own chunk. Measured on the real GPU at (q_seq 3, kv_seq 5): an exact match
/// to a NON-causal reference and a 0.290 miss against the causal one, on a signal
/// whose magnitude is 0.9.
///
/// # Why these tests discriminate
///
/// Every case is checked against *both* references and the suite first proves the two
/// references actually disagree for the shape under test (`separation`), so a pass
/// cannot be vacuous. The single-token shape is kept in the matrix precisely because
/// the two references coincide there - which is the justification for leaving the hot
/// decode path on the unmasked kernel.
#[cfg(all(target_os = "macos", feature = "metal", test))]
mod chunked_causal_tests {
    use super::*;
    use crate::gpu_ops::metal::functions::get_metal_backend;

    /// Deterministic, non-degenerate fill in `[-0.9, 0.9]`.
    fn pattern(n: usize, phase: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.317 + phase).sin() * 0.9).collect()
    }

    /// Host attention reference over `[num_heads, seq, head_dim]` operands, in f64.
    ///
    /// `causal == true` applies the mask the KV cache implies - query row `i` is
    /// absolute key position `(kv_seq_len - q_seq_len) + i` and may attend to key
    /// columns `0..=that`. `causal == false` is the unmasked generation kernel's
    /// semantics, where every query row sees every key column.
    fn reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        num_heads: usize,
        q_seq_len: usize,
        kv_seq_len: usize,
        head_dim: usize,
        causal: bool,
    ) -> Vec<f32> {
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let q_offset = kv_seq_len - q_seq_len;
        let mut out = vec![0.0f32; num_heads * q_seq_len * head_dim];
        for h in 0..num_heads {
            for i in 0..q_seq_len {
                let last = if causal { q_offset + i } else { kv_seq_len - 1 };
                let mut scores = Vec::with_capacity(last + 1);
                for j in 0..=last {
                    let mut dot = 0.0f64;
                    for d in 0..head_dim {
                        dot += f64::from(q[h * q_seq_len * head_dim + i * head_dim + d])
                            * f64::from(k[h * kv_seq_len * head_dim + j * head_dim + d]);
                    }
                    scores.push(dot * scale);
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for d in 0..head_dim {
                    let mut acc = 0.0f64;
                    for (j, e) in exps.iter().enumerate() {
                        acc +=
                            (e / sum) * f64::from(v[h * kv_seq_len * head_dim + j * head_dim + d]);
                    }
                    out[h * q_seq_len * head_dim + i * head_dim + d] = acc as f32;
                }
            }
        }
        out
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "compared slices must have equal length");
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    /// Every `(q_seq_len, kv_seq_len)` the composition can be handed must produce the
    /// KV-cache causal mask, not the unmasked one.
    #[test]
    fn cached_attention_is_causal_for_every_query_block_shape() -> Result<()> {
        let backend = get_metal_backend()?;
        let (num_heads, head_dim) = (3usize, 5usize);

        for (q_seq_len, kv_seq_len) in [(3usize, 5usize), (2, 6), (4, 4), (1, 7)] {
            let q = pattern(num_heads * q_seq_len * head_dim, 0.7);
            let k = pattern(num_heads * kv_seq_len * head_dim, 1.3);
            let v = pattern(num_heads * kv_seq_len * head_dim, 2.9);

            let q_id = backend.create_persistent_buffer(&q)?;
            let k_id = backend.create_persistent_buffer(&k)?;
            let v_id = backend.create_persistent_buffer(&v)?;
            let out_id = backend.attention_with_cache_gpu_to_gpu(
                &q_id, &k_id, &v_id, 1, q_seq_len, kv_seq_len, num_heads, head_dim,
            )?;
            let got =
                backend.download_buffer_via_staging(&out_id, num_heads * q_seq_len * head_dim)?;
            backend.release_buffers(&[q_id, k_id, v_id, out_id])?;

            let causal = reference(&q, &k, &v, num_heads, q_seq_len, kv_seq_len, head_dim, true);
            let non_causal = reference(
                &q, &k, &v, num_heads, q_seq_len, kv_seq_len, head_dim, false,
            );
            let separation = max_abs_diff(&causal, &non_causal);
            let vs_causal = max_abs_diff(&got, &causal);
            let vs_non_causal = max_abs_diff(&got, &non_causal);

            assert!(
                vs_causal < 2e-5,
                "(q_seq {q_seq_len}, kv_seq {kv_seq_len}): GPU output is {vs_causal} away \
                 from the KV-cache causal reference (non-causal distance {vs_non_causal})"
            );

            if q_seq_len == 1 {
                // The one query row IS the last key position, so both references agree
                // exactly. This is why the decode path may keep the unmasked kernel.
                assert_eq!(
                    separation, 0.0,
                    "(q_seq 1, kv_seq {kv_seq_len}): the causal and non-causal references \
                     must coincide for a single query row"
                );
                assert!(
                    vs_non_causal < 2e-5,
                    "(q_seq 1, kv_seq {kv_seq_len}): a single query row must attend to the \
                     whole cache, but the output is {vs_non_causal} from that reference"
                );
            } else {
                // The old failure mode: the output WAS the non-causal answer. Prove the
                // two references disagree materially first, so this cannot pass vacuously.
                assert!(
                    separation > 1e-2,
                    "(q_seq {q_seq_len}, kv_seq {kv_seq_len}): the two references differ by \
                     only {separation}; this case cannot discriminate and needs new data"
                );
                assert!(
                    vs_non_causal > separation * 0.5,
                    "(q_seq {q_seq_len}, kv_seq {kv_seq_len}): GPU output is only \
                     {vs_non_causal} from the NON-causal reference (references differ by \
                     {separation}) - the chunk is attending to its own future again"
                );
            }
        }
        Ok(())
    }

    /// The offset-masked kernel itself: weights must match a host softmax over the
    /// visible prefix, and the masked tail must be exactly zero.
    #[test]
    fn offset_masked_kernel_matches_its_reference_and_zeroes_the_tail() -> Result<()> {
        let backend = get_metal_backend()?;
        let (num_heads, head_dim) = (2usize, 4usize);

        for (q_seq_len, kv_seq_len) in [(3usize, 5usize), (2, 6), (4, 4), (1, 7), (5, 600)] {
            let q_offset = kv_seq_len - q_seq_len;
            let alpha = 1.0f32 / (head_dim as f32).sqrt();
            let q = pattern(num_heads * q_seq_len * head_dim, 0.11);
            // K arrives transposed: [num_heads, head_dim, kv_seq_len].
            let k_t = pattern(num_heads * head_dim * kv_seq_len, 0.53);

            let q_id = backend.create_persistent_buffer(&q)?;
            let k_id = backend.create_persistent_buffer(&k_t)?;
            let out_id = backend.batched_scaled_matmul_softmax_gen_causal_gpu_to_gpu(
                &q_id, &k_id, num_heads, q_seq_len, kv_seq_len, head_dim, alpha, q_offset,
            )?;
            let got =
                backend.download_buffer_via_staging(&out_id, num_heads * q_seq_len * kv_seq_len)?;
            backend.release_buffers(&[q_id, k_id, out_id])?;

            for h in 0..num_heads {
                for row in 0..q_seq_len {
                    let last = q_offset + row;
                    let base = h * q_seq_len * kv_seq_len + row * kv_seq_len;

                    let mut scores = Vec::with_capacity(last + 1);
                    for col in 0..=last {
                        let mut dot = 0.0f64;
                        for d in 0..head_dim {
                            dot += f64::from(q[h * q_seq_len * head_dim + row * head_dim + d])
                                * f64::from(k_t[h * head_dim * kv_seq_len + d * kv_seq_len + col]);
                        }
                        scores.push(dot * f64::from(alpha));
                    }
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                    let sum: f64 = exps.iter().sum();

                    for (col, e) in exps.iter().enumerate() {
                        let want = (e / sum) as f32;
                        let diff = (got[base + col] - want).abs();
                        assert!(
                            diff < 2e-5,
                            "q_seq={q_seq_len} kv_seq={kv_seq_len} head {h} row {row} col \
                             {col}: got {} want {want}",
                            got[base + col]
                        );
                    }
                    for col in (last + 1)..kv_seq_len {
                        assert_eq!(
                            got[base + col],
                            0.0,
                            "q_seq={q_seq_len} kv_seq={kv_seq_len} head {h} row {row}: key \
                             {col} is strictly in the future of absolute position {last} \
                             and must be masked to exactly zero"
                        );
                    }
                    let visible_sum: f32 = got[base..=base + last].iter().sum();
                    assert!(
                        (visible_sum - 1.0).abs() < 1e-3,
                        "q_seq={q_seq_len} kv_seq={kv_seq_len} head {h} row {row}: visible \
                         weights sum to {visible_sum}, not 1"
                    );
                }
            }
        }
        Ok(())
    }

    /// A query block that does not fit inside the key sequence is a structured error,
    /// not a silently unmasked dispatch.
    #[test]
    fn offset_masked_kernel_rejects_a_block_that_does_not_fit() -> Result<()> {
        let backend = get_metal_backend()?;
        let q_id = backend.create_persistent_buffer(&pattern(2 * 3 * 4, 0.2))?;
        let k_id = backend.create_persistent_buffer(&pattern(2 * 4 * 6, 0.4))?;

        let error = backend
            .batched_scaled_matmul_softmax_gen_causal_gpu_to_gpu(&q_id, &k_id, 2, 3, 6, 4, 0.5, 4)
            .expect_err("q_offset 4 + q_seq_len 3 overruns a 6-position key sequence");
        let rendered = format!("{error}");
        assert!(
            rendered.contains("do not fit"),
            "the error must name the mismatch, got: {rendered}"
        );

        // The same call with a fitting offset must succeed.
        let ok_id = backend.batched_scaled_matmul_softmax_gen_causal_gpu_to_gpu(
            &q_id, &k_id, 2, 3, 6, 4, 0.5, 3,
        )?;
        backend.release_buffers(&[q_id, k_id, ok_id])?;
        Ok(())
    }

    /// `flash_attention_with_cache` masked by the query's *block-relative* index, so a
    /// query block shorter than the key sequence - the only case its name is about -
    /// threw away the cached prefix: a single-token decode against a 100-token cache
    /// saw only key position 0. It must mask by absolute position instead.
    #[test]
    fn flash_attention_with_cache_masks_by_absolute_position() -> Result<()> {
        let backend = get_metal_backend()?;
        let (num_heads, head_dim) = (2usize, 8usize);

        for (q_seq_len, kv_seq_len) in [(2usize, 6usize), (1, 5), (4, 4)] {
            let q = pattern(num_heads * q_seq_len * head_dim, 0.31);
            let k = pattern(num_heads * kv_seq_len * head_dim, 1.7);
            let v = pattern(num_heads * kv_seq_len * head_dim, 2.3);

            let q_id = backend.create_persistent_buffer(&q)?;
            let k_id = backend.create_persistent_buffer(&k)?;
            let v_id = backend.create_persistent_buffer(&v)?;
            let out_id = backend.flash_attention_with_cache(
                &q_id, &k_id, &v_id, 1, q_seq_len, kv_seq_len, num_heads, head_dim,
            )?;
            let got =
                backend.download_buffer_via_staging(&out_id, num_heads * q_seq_len * head_dim)?;
            backend.release_buffers(&[q_id, k_id, v_id, out_id])?;

            let causal = reference(&q, &k, &v, num_heads, q_seq_len, kv_seq_len, head_dim, true);
            let vs_causal = max_abs_diff(&got, &causal);
            assert!(
                vs_causal < 1e-4,
                "flash (q_seq {q_seq_len}, kv_seq {kv_seq_len}): output is {vs_causal} from \
                 the KV-cache causal reference"
            );

            // Discrimination: with the old block-relative mask, row 0 could only see key
            // 0, i.e. exactly V[h, 0, :]. Prove that answer is materially different.
            if kv_seq_len > q_seq_len {
                let first_key_only: Vec<f32> = (0..num_heads)
                    .flat_map(|h| {
                        v[h * kv_seq_len * head_dim..h * kv_seq_len * head_dim + head_dim].to_vec()
                    })
                    .collect();
                let row0: Vec<f32> = (0..num_heads)
                    .flat_map(|h| {
                        got[h * q_seq_len * head_dim..h * q_seq_len * head_dim + head_dim].to_vec()
                    })
                    .collect();
                let vs_broken = max_abs_diff(&row0, &first_key_only);
                assert!(
                    vs_broken > 1e-2,
                    "flash (q_seq {q_seq_len}, kv_seq {kv_seq_len}): row 0 is {vs_broken} from \
                     'attends to key 0 only' - the cached prefix is still being masked away"
                );
            }
        }
        Ok(())
    }
}
