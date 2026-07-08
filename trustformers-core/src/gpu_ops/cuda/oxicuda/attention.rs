//! GPU-resident attention chain for the oxicuda CUDA backend.
//!
//! This module gives the CUDA backend the resident method set the Metal backend
//! already has (attention, RoPE, causal softmax, residual add), so transformer
//! models can keep a whole attention block on the device instead of bouncing
//! every intermediate through the host. All methods operate on persistent
//! buffers owned by the backend's cache and return new resident buffer ids
//! (callers wrap them in the refcounted handle lifecycle exactly like the other
//! `*_gpu_to_gpu` ops).
//!
//! # Why the chain is composed from primitives (verified upstream constraints)
//!
//! The routing below is built from oxicuda entry points whose published
//! sources this codebase has verified. Several oxicuda 0.4.0 defects that
//! originally shaped this module were fixed upstream in oxicuda 0.4.1
//! (verified against the published 0.4.1 sources, 2026-07-07); the remaining
//! constraints are:
//!
//! - **GEMM transpose flags work in 0.4.1** (transposed operands are routed
//!   to a SIMT kernel honouring all four `(trans_a, trans_b)` combinations),
//!   so `Q @ K^T` is expressed directly with `Transpose::Trans` on a
//!   heads-major `[kv, d]` key block — the pitched-copy `K^T` materialisation
//!   this module used against 0.4.0 has been removed.
//! - **GEMM still requires tightly packed row-major operands.** The 0.4.1
//!   dispatcher rejects (`InvalidArgument`) any operand with `ld != cols` or
//!   a column-major layout, so per-head matrices must be packed contiguously;
//!   strided views into `[seq, hidden]` activations are still not an option.
//!   Head splitting therefore remains a pitched device-to-device copy.
//! - **`oxicuda_dnn::attn::mha::multi_head_attention` is real in 0.4.1**
//!   (0.4.0 shipped empty kernel bodies; 0.4.1 has genuine QK^T/softmax/PV
//!   kernels and a dedicated score scratch buffer) but is still not used
//!   here: it exposes no KV-cache/decode surface, takes an additive mask
//!   tensor rather than a causal flag, and its scale+mask kernel is hardcoded
//!   to f32 addressing (the f64 instantiation mis-addresses 8-byte elements —
//!   still open upstream; harmless here because this backend is f32-only).
//!   // TODO(oxicuda): revisit adopting `multi_head_attention` for the bulk
//!   // prefill once upstream ships a causal-mask/KV-cache surface and fixes
//!   // the f64 `generate_scale_mask_ptx` addressing.
//! - **`causal_softmax` takes an explicit `seq_len` in 0.4.1** and masks by
//!   `row % seq_len`, so flattened `[H * seq, seq]` score batches are now
//!   supported upstream. The prefill below still calls it per head — not as a
//!   correctness workaround but to reuse one `[seq, seq]` scratch instead of
//!   materialising all `H` score matrices at once. The rectangular decode
//!   case (`q = 1`, every key live) uses the plain row-wise `softmax`,
//!   batched over heads as a `[H, kv]` matrix.
//! - **`rope_neox_half_split_f32` positions are implicit `0..seq_len`**, so
//!   decode-time RoPE at a cache offset is realised by running the kernel over
//!   a zero-padded `[offset + seq, H, d]` buffer and slicing the tail (rotating
//!   zeros yields zeros, and only the tail rows are read back).
//!
//! # Pitched copies and stream ordering
//!
//! Head split/merge and KV-cache concatenation are pitched
//! device-to-device copies issued through the raw `cuMemcpy2D` driver entry
//! point (`oxicuda_memory::copy_2d_dtod` exposes no source/destination offsets,
//! which these gathers need). The synchronous memcpy API is ordered on the
//! legacy NULL stream, while GEMM/softmax kernels run on the backend's
//! non-blocking BLAS stream and RoPE on a per-call DNN stream. To keep those
//! three queues ordered, every public method in this module **enters and exits
//! context-synchronized** (`cuCtxSynchronize`). This is deliberately
//! correctness-first: a handful of synchronizations per transformer layer, no
//! host round-trips. Single-stream execution (via an async 2D memcpy, once
//! upstream exposes one) is the recorded optimisation path.

use oxicuda_blas::level3::gemm_api::gemm;
use oxicuda_blas::{Layout, MatrixDesc, MatrixDescMut, Transpose};
use oxicuda_dnn::DnnHandle;
use oxicuda_memory::DeviceBuffer;

use super::{OxiCudaBufferId, OxicudaCudaBackend};
use crate::errors::TrustformersError;

// ---------------------------------------------------------------------------
// Pitched-copy planning (pure host-side logic, hardware-free)
// ---------------------------------------------------------------------------

/// One pitched 2D device-to-device copy, in **elements** (not bytes).
///
/// Copies `height` rows of `width` elements each: row `r` moves from
/// `src_offset + r * src_pitch` to `dst_offset + r * dst_pitch`. This is the
/// element-level description of a `cuMemcpy2D` call; [`PitchedCopy::validate`]
/// re-checks every bound because the raw driver call performs none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PitchedCopy {
    /// Element offset of the first copied row inside the source buffer.
    pub src_offset: usize,
    /// Element offset of the first written row inside the destination buffer.
    pub dst_offset: usize,
    /// Source row stride in elements (must be >= `width`).
    pub src_pitch: usize,
    /// Destination row stride in elements (must be >= `width`).
    pub dst_pitch: usize,
    /// Elements copied per row (> 0).
    pub width: usize,
    /// Number of rows (> 0).
    pub height: usize,
}

impl PitchedCopy {
    /// Bounds-check this copy against the two buffer lengths (in elements).
    ///
    /// The raw `cuMemcpy2D` path validates nothing, so this is the single
    /// gatekeeper: zero-sized copies, pitches narrower than the row width, and
    /// any access past either buffer end are rejected with a shape error.
    pub(crate) fn validate(&self, src_len: usize, dst_len: usize) -> crate::errors::Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(TrustformersError::shape_error(format!(
                "Pitched copy must be non-empty (width={}, height={})",
                self.width, self.height
            )));
        }
        if self.src_pitch < self.width || self.dst_pitch < self.width {
            return Err(TrustformersError::shape_error(format!(
                "Pitched copy pitches ({}, {}) must be >= width {}",
                self.src_pitch, self.dst_pitch, self.width
            )));
        }
        let src_end = self
            .height
            .checked_sub(1)
            .and_then(|rows| rows.checked_mul(self.src_pitch))
            .and_then(|off| off.checked_add(self.width))
            .and_then(|off| off.checked_add(self.src_offset));
        match src_end {
            Some(end) if end <= src_len => {},
            _ => {
                return Err(TrustformersError::shape_error(format!(
                    "Pitched copy source region (offset {}, pitch {}, {}x{}) exceeds buffer length {}",
                    self.src_offset, self.src_pitch, self.height, self.width, src_len
                )));
            },
        }
        let dst_end = self
            .height
            .checked_sub(1)
            .and_then(|rows| rows.checked_mul(self.dst_pitch))
            .and_then(|off| off.checked_add(self.width))
            .and_then(|off| off.checked_add(self.dst_offset));
        match dst_end {
            Some(end) if end <= dst_len => {},
            _ => {
                return Err(TrustformersError::shape_error(format!(
                    "Pitched copy destination region (offset {}, pitch {}, {}x{}) exceeds buffer length {}",
                    self.dst_offset, self.dst_pitch, self.height, self.width, dst_len
                )));
            },
        }
        Ok(())
    }
}

/// Checked product helper for plan sizing.
fn checked_len(factors: &[usize], what: &str) -> crate::errors::Result<usize> {
    factors.iter().try_fold(1usize, |acc, &f| {
        acc.checked_mul(f).ok_or_else(|| {
            TrustformersError::shape_error(format!("{} size {:?} overflows usize", what, factors))
        })
    })
}

/// Plan the head-gather permutation used for QKV splitting, `[seq, H, d] ->
/// [H, seq, d]` re-packing, and `[H, seq, d] -> [seq, H*d]` head merging.
///
/// The source is a flat row-major buffer of `rows` rows with `src_row_stride`
/// elements each; head `h`'s slice starts at column `comp_offset +
/// h * head_stride` and spans `head_dim` columns. One copy is emitted per head.
///
/// - `heads_major = true` packs the output as `[num_heads, rows, head_dim]`
///   (per-head matrices packed back to back — the GEMM operand layout).
/// - `heads_major = false` packs it as `[rows, num_heads * head_dim]` (heads
///   interleaved per row — the RoPE layout, and the head-merge layout).
pub(crate) fn gather_heads_plan(
    rows: usize,
    src_row_stride: usize,
    comp_offset: usize,
    head_stride: usize,
    num_heads: usize,
    head_dim: usize,
    heads_major: bool,
) -> crate::errors::Result<Vec<PitchedCopy>> {
    if rows == 0 || num_heads == 0 || head_dim == 0 {
        return Err(TrustformersError::shape_error(format!(
            "gather_heads requires non-zero dims (rows={}, heads={}, head_dim={})",
            rows, num_heads, head_dim
        )));
    }
    let mut plan = Vec::with_capacity(num_heads);
    for h in 0..num_heads {
        let src_offset = h
            .checked_mul(head_stride)
            .and_then(|off| off.checked_add(comp_offset))
            .ok_or_else(|| {
                TrustformersError::shape_error("gather_heads source offset overflows".to_string())
            })?;
        let (dst_offset, dst_pitch) = if heads_major {
            (
                checked_len(&[h, rows, head_dim], "gather_heads dst")?,
                head_dim,
            )
        } else {
            (
                h.checked_mul(head_dim).ok_or_else(|| {
                    TrustformersError::shape_error("gather_heads dst offset overflows".to_string())
                })?,
                checked_len(&[num_heads, head_dim], "gather_heads dst pitch")?,
            )
        };
        plan.push(PitchedCopy {
            src_offset,
            dst_offset,
            src_pitch: src_row_stride,
            dst_pitch,
            width: head_dim,
            height: rows,
        });
    }
    Ok(plan)
}

/// Plan appending `kv_new` key or value rows to a heads-major KV cache:
/// `[H, kv_old, d] ++ [H, kv_new, d] -> [H, kv_old + kv_new, d]`.
///
/// Per head both chunks are contiguous, so each side is one flat (height-1)
/// copy per head. Returns `(old_plan, new_plan)`: `old_plan` re-packs the
/// previous cache into the wider output (empty when `kv_old == 0`),
/// `new_plan` appends the new rows behind the old ones. Keys and values share
/// this plan since oxicuda 0.4.1's transpose-aware GEMM removed the need for
/// a column-oriented `K^T` cache layout.
pub(crate) fn concat_v_plan(
    num_heads: usize,
    kv_old: usize,
    kv_new: usize,
    head_dim: usize,
) -> crate::errors::Result<(Vec<PitchedCopy>, Vec<PitchedCopy>)> {
    if num_heads == 0 || head_dim == 0 || kv_new == 0 {
        return Err(TrustformersError::shape_error(format!(
            "concat_v requires non-zero dims (heads={}, head_dim={}, kv_new={})",
            num_heads, head_dim, kv_new
        )));
    }
    let kv_total = kv_old.checked_add(kv_new).ok_or_else(|| {
        TrustformersError::shape_error("concat_v total length overflows".to_string())
    })?;
    let old_chunk = checked_len(&[kv_old, head_dim], "concat_v old chunk")?;
    let new_chunk = checked_len(&[kv_new, head_dim], "concat_v new chunk")?;
    let total_chunk = checked_len(&[kv_total, head_dim], "concat_v total chunk")?;
    let mut old_plan = Vec::new();
    let mut new_plan = Vec::with_capacity(num_heads);
    for h in 0..num_heads {
        let dst_head = h.checked_mul(total_chunk).ok_or_else(|| {
            TrustformersError::shape_error("concat_v dst offset overflows".to_string())
        })?;
        if kv_old > 0 {
            old_plan.push(PitchedCopy {
                src_offset: h * old_chunk,
                dst_offset: dst_head,
                src_pitch: old_chunk,
                dst_pitch: old_chunk,
                width: old_chunk,
                height: 1,
            });
        }
        new_plan.push(PitchedCopy {
            src_offset: h * new_chunk,
            dst_offset: dst_head + old_chunk,
            src_pitch: new_chunk,
            dst_pitch: new_chunk,
            width: new_chunk,
            height: 1,
        });
    }
    Ok((old_plan, new_plan))
}

// ---------------------------------------------------------------------------
// Raw pitched-copy execution
// ---------------------------------------------------------------------------

/// Execute a batch of pitched device-to-device copies via `cuMemcpy2D`.
///
/// Every copy is bounds-checked first ([`PitchedCopy::validate`]) because the
/// driver call trusts its pointers. Offsets/pitches are converted to bytes
/// here; the source and destination pointers are offset directly (the
/// `CUDA_MEMCPY2D` x/y fields are left at their zero defaults).
///
/// Ordering: these copies run on the legacy NULL stream; callers are
/// responsible for the context synchronization brackets described in the
/// module docs.
fn run_pitched_copies(
    src: &DeviceBuffer<f32>,
    dst: &mut DeviceBuffer<f32>,
    plan: &[PitchedCopy],
    op: &str,
) -> crate::errors::Result<()> {
    use oxicuda_driver::ffi::{CUmemorytype, CUDA_MEMCPY2D};

    let api = oxicuda_driver::loader::try_driver().map_err(|e| {
        TrustformersError::hardware_error(&format!("Failed to load CUDA driver: {}", e), op)
    })?;
    let memcpy_2d = api.cu_memcpy_2d.ok_or_else(|| {
        TrustformersError::hardware_error("cuMemcpy2D is unavailable in this driver", op)
    })?;

    let elem = std::mem::size_of::<f32>();
    for copy in plan {
        copy.validate(src.len(), dst.len())?;
        let desc = CUDA_MEMCPY2D {
            src_memory_type: CUmemorytype::Device as u32,
            src_device: src.as_device_ptr() + (copy.src_offset * elem) as u64,
            src_pitch: copy.src_pitch * elem,
            dst_memory_type: CUmemorytype::Device as u32,
            dst_device: dst.as_device_ptr() + (copy.dst_offset * elem) as u64,
            dst_pitch: copy.dst_pitch * elem,
            width_in_bytes: copy.width * elem,
            height: copy.height,
            ..CUDA_MEMCPY2D::default()
        };
        // SAFETY: both pointers address live device allocations owned by `src`
        // and `dst`, and `validate` proved the pitched region lies inside them.
        oxicuda_driver::check(unsafe { memcpy_2d(&desc) }).map_err(|e| {
            TrustformersError::hardware_error(&format!("cuMemcpy2D failed: {}", e), op)
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Backend methods
// ---------------------------------------------------------------------------

/// Convert a dimension to `u32` for the kernel APIs, with a precise error.
fn dim_u32(value: usize, what: &str, op: &str) -> crate::errors::Result<u32> {
    u32::try_from(value).map_err(|_| {
        TrustformersError::shape_error(format!("{}: {} {} exceeds u32", op, what, value))
    })
}

impl OxicudaCudaBackend {
    /// Synchronize the whole CUDA context (all streams plus the legacy queue).
    ///
    /// The resident attention chain mixes three queues (BLAS stream, per-call
    /// DNN stream, NULL-stream pitched copies); every public method in this
    /// module brackets its work with this call so cross-queue reads never race.
    fn sync_context(&self, op: &str) -> crate::errors::Result<()> {
        self.context().synchronize().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("CUDA context synchronization failed: {}", e),
                op,
            )
        })
    }

    /// Gather per-head slices out of a resident row-major buffer.
    ///
    /// The source is treated as `rows` rows of `src_row_stride` elements; head
    /// `h` occupies columns `comp_offset + h * head_stride ..+ head_dim`. With
    /// `heads_major = true` the output is packed `[num_heads, rows, head_dim]`
    /// (GEMM operand layout); with `false` it is `[rows, num_heads * head_dim]`
    /// (RoPE / merged-heads layout). This one primitive covers QKV splitting
    /// (GPT-2: `comp_offset` in `{0, hidden, 2*hidden}`, `head_stride =
    /// head_dim`; GPT-NeoX: `comp_offset` in `{0, d, 2d}`, `head_stride = 3d`),
    /// the `[seq, H, d] -> [H, seq, d]` re-pack, and head merging
    /// (`src_row_stride = head_dim`, `head_stride = rows * head_dim`).
    #[allow(clippy::too_many_arguments)] // Mirrors the pitched-gather geometry; no natural grouping struct.
    pub fn gather_heads_gpu_to_gpu(
        &self,
        src_id: &OxiCudaBufferId,
        rows: usize,
        src_row_stride: usize,
        comp_offset: usize,
        head_stride: usize,
        num_heads: usize,
        head_dim: usize,
        heads_major: bool,
    ) -> crate::errors::Result<OxiCudaBufferId> {
        let op = "gather_heads_gpu_to_gpu";
        let plan = gather_heads_plan(
            rows,
            src_row_stride,
            comp_offset,
            head_stride,
            num_heads,
            head_dim,
            heads_major,
        )?;
        let out_len = checked_len(&[num_heads, rows, head_dim], op)?;

        self.sync_context(op)?;
        let mut cache = self
            .buffer_cache
            .lock()
            .map_err(|_| TrustformersError::hardware_error("Failed to lock buffer cache", op))?;
        let src = cache.get(src_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Source buffer {:?} not found in cache", src_id),
                op,
            )
        })?;
        // Every output element is written by exactly one copy, so a plain
        // allocation suffices (validated below against both buffer lengths).
        let mut out = DeviceBuffer::<f32>::alloc(out_len).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer on device: {}", e),
                op,
            )
        })?;
        run_pitched_copies(src, &mut out, &plan, op)?;

        // Final sync before publishing the id: on failure the still-local
        // output buffer drops (freed) instead of leaking in the cache.
        self.sync_context(op)?;
        let output_id = OxiCudaBufferId::new();
        cache.insert(output_id, out);
        Ok(output_id)
    }

    /// GPT-NeoX half-split partial RoPE on a resident `[seq, H, d]` buffer,
    /// result resident. `position_offset` is the absolute position of the
    /// first row (0 for prefill; the cached length during decode).
    ///
    /// The device kernel's positions are implicit `0..seq_len`, so a non-zero
    /// offset is realised by zero-padding `offset` leading rows, rotating the
    /// padded `[offset + seq, H, d]` buffer (rotated zeros stay zero) and
    /// copying only the tail rows out. That trades `O(offset)` extra kernel
    /// work for staying entirely on the device.
    #[allow(clippy::too_many_arguments)] // Mirrors the host `rope_f32` signature plus the offset.
    pub fn rope_neox_gpu_to_gpu(
        &self,
        input_id: &OxiCudaBufferId,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_ndims: usize,
        base: f32,
        position_offset: usize,
    ) -> crate::errors::Result<OxiCudaBufferId> {
        let op = "rope_neox_gpu_to_gpu";
        let row = checked_len(&[num_heads, head_dim], op)?;
        let total = checked_len(&[seq_len, row], op)?;
        if total == 0 {
            return Err(TrustformersError::shape_error(format!(
                "{}: dimensions must be non-zero (seq_len={}, heads={}, head_dim={})",
                op, seq_len, num_heads, head_dim
            )));
        }
        let seq_u32 = dim_u32(
            position_offset.checked_add(seq_len).ok_or_else(|| {
                TrustformersError::shape_error(format!("{}: padded length overflows", op))
            })?,
            "padded seq_len",
            op,
        )?;
        let heads_u32 = dim_u32(num_heads, "num_heads", op)?;
        let dim_u32_ = dim_u32(head_dim, "head_dim", op)?;
        let rot_u32 = dim_u32(rotary_ndims, "rotary_ndims", op)?;

        self.sync_context(op)?;
        let mut cache = self
            .buffer_cache
            .lock()
            .map_err(|_| TrustformersError::hardware_error("Failed to lock buffer cache", op))?;
        let input = cache.get(input_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Input buffer {:?} not found in cache", input_id),
                op,
            )
        })?;
        if input.len() != total {
            return Err(TrustformersError::shape_error(format!(
                "{}: input length {} doesn't match seq_len {} * num_heads {} * head_dim {}",
                op,
                input.len(),
                seq_len,
                num_heads,
                head_dim
            )));
        }

        let dnn = DnnHandle::new(self.context()).map_err(|e| {
            TrustformersError::hardware_error(&format!("Failed to create cuDNN handle: {}", e), op)
        })?;

        let out = if position_offset == 0 {
            // Direct case: rotate in place-shape, one kernel launch.
            let mut out = DeviceBuffer::<f32>::alloc(total).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to allocate output buffer on device: {}", e),
                    op,
                )
            })?;
            oxicuda_dnn::attn::rope_neox_half_split_f32(
                &dnn, input, &mut out, seq_u32, heads_u32, dim_u32_, rot_u32, base,
            )
            .map_err(|e| {
                TrustformersError::hardware_error(&format!("RoPE execution failed: {}", e), op)
            })?;
            self.sync_context(op)?;
            out
        } else {
            // Offset case: pad, rotate, slice the tail back out.
            let padded_total = checked_len(&[position_offset + seq_len, row], op)?;
            let mut padded_in = DeviceBuffer::<f32>::zeroed(padded_total).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to allocate padded input on device: {}", e),
                    op,
                )
            })?;
            let pad = position_offset * row;
            run_pitched_copies(
                input,
                &mut padded_in,
                &[PitchedCopy {
                    src_offset: 0,
                    dst_offset: pad,
                    src_pitch: total,
                    dst_pitch: total,
                    width: total,
                    height: 1,
                }],
                op,
            )?;
            self.sync_context(op)?;

            let mut padded_out = DeviceBuffer::<f32>::alloc(padded_total).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to allocate padded output on device: {}", e),
                    op,
                )
            })?;
            oxicuda_dnn::attn::rope_neox_half_split_f32(
                &dnn,
                &padded_in,
                &mut padded_out,
                seq_u32,
                heads_u32,
                dim_u32_,
                rot_u32,
                base,
            )
            .map_err(|e| {
                TrustformersError::hardware_error(&format!("RoPE execution failed: {}", e), op)
            })?;
            self.sync_context(op)?;

            let mut out = DeviceBuffer::<f32>::alloc(total).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to allocate output buffer on device: {}", e),
                    op,
                )
            })?;
            run_pitched_copies(
                &padded_out,
                &mut out,
                &[PitchedCopy {
                    src_offset: pad,
                    dst_offset: 0,
                    src_pitch: total,
                    dst_pitch: total,
                    width: total,
                    height: 1,
                }],
                op,
            )?;
            out
        };

        // Final sync before publishing the id: on failure the still-local
        // output buffer drops (freed) instead of leaking in the cache.
        self.sync_context(op)?;
        let output_id = OxiCudaBufferId::new();
        cache.insert(output_id, out);
        Ok(output_id)
    }

    /// Causal (lower-triangular) row-wise softmax over a resident `[rows,
    /// cols]` matrix, result resident. Row `r` normalizes over columns
    /// `j <= r` and zeroes the rest — single-matrix semantics (`seq_len =
    /// rows` is passed to the kernel). oxicuda 0.4.1's `causal_softmax` also
    /// supports flattened `[batch * seq, cols]` batches via its `seq_len`
    /// parameter; extend this wrapper if a batched caller appears.
    pub fn softmax_causal_gpu_to_gpu(
        &self,
        input_id: &OxiCudaBufferId,
        rows: usize,
        cols: usize,
    ) -> crate::errors::Result<OxiCudaBufferId> {
        let op = "softmax_causal_gpu_to_gpu";
        let total = checked_len(&[rows, cols], op)?;
        let rows_u32 = dim_u32(rows, "rows", op)?;
        let cols_u32 = dim_u32(cols, "cols", op)?;

        self.sync_context(op)?;
        let mut cache = self
            .buffer_cache
            .lock()
            .map_err(|_| TrustformersError::hardware_error("Failed to lock buffer cache", op))?;
        let input = cache.get(input_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Input buffer {:?} not found in cache", input_id),
                op,
            )
        })?;
        if input.len() != total {
            return Err(TrustformersError::shape_error(format!(
                "{}: input length {} doesn't match rows {} * cols {}",
                op,
                input.len(),
                rows,
                cols
            )));
        }
        // The kernel writes every cell (masked cells as 0), so alloc suffices.
        let mut out = DeviceBuffer::<f32>::alloc(total).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer on device: {}", e),
                op,
            )
        })?;
        // seq_len = rows: one causal matrix, mask row is the global row.
        oxicuda_blas::reduction::causal_softmax::<f32>(
            &self.handle,
            rows_u32,
            cols_u32,
            rows_u32,
            input,
            &mut out,
        )
        .map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Causal softmax execution failed: {}", e),
                op,
            )
        })?;

        // Final sync before publishing the id: on failure the still-local
        // output buffer drops (freed) instead of leaking in the cache.
        self.sync_context(op)?;
        let output_id = OxiCudaBufferId::new();
        cache.insert(output_id, out);
        Ok(output_id)
    }

    /// Plain (unmasked) row-wise softmax over a resident `[rows, cols]`
    /// matrix, result resident. Rows are independent, so packing one head per
    /// row batches the decode-time attention softmax into a single launch.
    pub fn softmax_rows_gpu_to_gpu(
        &self,
        input_id: &OxiCudaBufferId,
        rows: usize,
        cols: usize,
    ) -> crate::errors::Result<OxiCudaBufferId> {
        let op = "softmax_rows_gpu_to_gpu";
        let total = checked_len(&[rows, cols], op)?;
        let rows_u32 = dim_u32(rows, "rows", op)?;
        let cols_u32 = dim_u32(cols, "cols", op)?;

        self.sync_context(op)?;
        let mut cache = self
            .buffer_cache
            .lock()
            .map_err(|_| TrustformersError::hardware_error("Failed to lock buffer cache", op))?;
        let input = cache.get(input_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Input buffer {:?} not found in cache", input_id),
                op,
            )
        })?;
        if input.len() != total {
            return Err(TrustformersError::shape_error(format!(
                "{}: input length {} doesn't match rows {} * cols {}",
                op,
                input.len(),
                rows,
                cols
            )));
        }
        let mut out = DeviceBuffer::<f32>::alloc(total).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer on device: {}", e),
                op,
            )
        })?;
        oxicuda_blas::reduction::softmax::<f32>(&self.handle, rows_u32, cols_u32, input, &mut out)
            .map_err(|e| {
                TrustformersError::hardware_error(&format!("Softmax execution failed: {}", e), op)
            })?;

        // Final sync before publishing the id: on failure the still-local
        // output buffer drops (freed) instead of leaking in the cache.
        self.sync_context(op)?;
        let output_id = OxiCudaBufferId::new();
        cache.insert(output_id, out);
        Ok(output_id)
    }

    /// Element-wise addition of two equal-length resident buffers, result
    /// resident — the residual-add primitive (`hidden + attention_output`).
    /// Mirrors the Metal backend's `add_gpu_to_gpu` signature.
    pub fn add_gpu_to_gpu(
        &self,
        a_id: &OxiCudaBufferId,
        b_id: &OxiCudaBufferId,
        size: usize,
    ) -> crate::errors::Result<OxiCudaBufferId> {
        let op = "add_gpu_to_gpu";
        let size_u32 = dim_u32(size, "size", op)?;
        if size == 0 {
            return Err(TrustformersError::shape_error(format!(
                "{}: size must be non-zero",
                op
            )));
        }

        self.sync_context(op)?;
        let mut cache = self
            .buffer_cache
            .lock()
            .map_err(|_| TrustformersError::hardware_error("Failed to lock buffer cache", op))?;
        let a = cache.get(a_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Input buffer {:?} not found in cache", a_id),
                op,
            )
        })?;
        let b = cache.get(b_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Input buffer {:?} not found in cache", b_id),
                op,
            )
        })?;
        if a.len() != size || b.len() != size {
            return Err(TrustformersError::shape_error(format!(
                "{}: operand lengths ({}, {}) must both equal size {}",
                op,
                a.len(),
                b.len(),
                size
            )));
        }
        let mut out = DeviceBuffer::<f32>::alloc(size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer on device: {}", e),
                op,
            )
        })?;
        oxicuda_blas::elementwise::add::<f32>(&self.handle, size_u32, a, b, &mut out).map_err(
            |e| {
                TrustformersError::hardware_error(
                    &format!("Elementwise add execution failed: {}", e),
                    op,
                )
            },
        )?;

        // Final sync before publishing the id: on failure the still-local
        // output buffer drops (freed) instead of leaking in the cache.
        self.sync_context(op)?;
        let output_id = OxiCudaBufferId::new();
        cache.insert(output_id, out);
        Ok(output_id)
    }

    /// Full multi-head **prefill** attention on resident buffers, result
    /// resident: `softmax(scale * Q_h @ K_h^T, causal) @ V_h` per head.
    ///
    /// Layouts (all packed, batch 1): `q`, `k` and `v` are `[H, seq, d]`;
    /// the output is `[H, seq, d]`. `K^T` is expressed with
    /// `Transpose::Trans` on the score GEMM (transpose-aware since oxicuda
    /// 0.4.1), so keys are consumed in the same heads-major layout as Q/V.
    /// Per head this runs one scaled GEMM, one causal softmax and one GEMM,
    /// all on the backend stream; the two `[seq, seq]` scratch buffers are
    /// reused across heads and freed after a final synchronization (so the
    /// async launches can never race the free).
    ///
    /// The causal mask is the from-empty-cache one (`query i` sees keys
    /// `0..=i`); prefills appended behind an existing cache must use the
    /// decode path per token instead.
    #[allow(clippy::too_many_arguments)] // Attention geometry (3 operands + 4 dims); grouping adds nothing.
    pub fn attention_prefill_gpu_to_gpu(
        &self,
        q_id: &OxiCudaBufferId,
        k_id: &OxiCudaBufferId,
        v_id: &OxiCudaBufferId,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> crate::errors::Result<OxiCudaBufferId> {
        let op = "attention_prefill_gpu_to_gpu";
        let head_mat = checked_len(&[seq_len, head_dim], op)?;
        let total = checked_len(&[num_heads, head_mat], op)?;
        let scores = checked_len(&[seq_len, seq_len], op)?;
        if total == 0 {
            return Err(TrustformersError::shape_error(format!(
                "{}: dimensions must be non-zero (heads={}, seq={}, head_dim={})",
                op, num_heads, seq_len, head_dim
            )));
        }
        let seq_u32 = dim_u32(seq_len, "seq_len", op)?;
        let dim_u32_ = dim_u32(head_dim, "head_dim", op)?;

        self.sync_context(op)?;
        let mut cache = self
            .buffer_cache
            .lock()
            .map_err(|_| TrustformersError::hardware_error("Failed to lock buffer cache", op))?;
        let missing = |id: &OxiCudaBufferId| {
            TrustformersError::hardware_error(&format!("Buffer {:?} not found in cache", id), op)
        };
        let q = cache.get(q_id).ok_or_else(|| missing(q_id))?;
        let k = cache.get(k_id).ok_or_else(|| missing(k_id))?;
        let v = cache.get(v_id).ok_or_else(|| missing(v_id))?;
        if q.len() != total || k.len() != total || v.len() != total {
            return Err(TrustformersError::shape_error(format!(
                "{}: operand lengths ({}, {}, {}) must all equal heads {} * seq {} * head_dim {}",
                op,
                q.len(),
                k.len(),
                v.len(),
                num_heads,
                seq_len,
                head_dim
            )));
        }

        // Scratch matrices reused across heads. The score buffer is zeroed
        // because the GEMM epilogue reads C_old even with beta = 0; the
        // probability buffer is fully written by the softmax kernel.
        let s_buf = DeviceBuffer::<f32>::zeroed(scores).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate score scratch on device: {}", e),
                op,
            )
        })?;
        let mut p_buf = DeviceBuffer::<f32>::alloc(scores).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate probability scratch on device: {}", e),
                op,
            )
        })?;
        let out = DeviceBuffer::<f32>::zeroed(total).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer on device: {}", e),
                op,
            )
        })?;

        let elem = std::mem::size_of::<f32>() as u64;
        for h in 0..num_heads {
            let head_off = (h * head_mat) as u64 * elem;
            // S = scale * Q_h @ K_h^T   ([seq, d] @ [seq, d]^T -> [seq, seq]);
            // the transpose is a GEMM flag (transpose-aware since 0.4.1).
            let q_desc = MatrixDesc::<f32>::from_raw(
                q.as_device_ptr() + head_off,
                seq_u32,
                dim_u32_,
                dim_u32_,
                Layout::RowMajor,
            );
            let k_desc = MatrixDesc::<f32>::from_raw(
                k.as_device_ptr() + head_off,
                seq_u32,
                dim_u32_,
                dim_u32_,
                Layout::RowMajor,
            );
            let mut s_desc = MatrixDescMut::<f32>::from_raw(
                s_buf.as_device_ptr(),
                seq_u32,
                seq_u32,
                seq_u32,
                Layout::RowMajor,
            );
            gemm::<f32>(
                &self.handle,
                Transpose::NoTrans,
                Transpose::Trans,
                scale,
                &q_desc,
                &k_desc,
                0.0f32,
                &mut s_desc,
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Score GEMM failed at head {}: {}", h, e),
                    op,
                )
            })?;

            // P = causal_softmax(S) — per head (scratch reuse; see module
            // docs), so seq_len == rows: one square causal matrix per call.
            oxicuda_blas::reduction::causal_softmax::<f32>(
                &self.handle,
                seq_u32,
                seq_u32,
                seq_u32,
                &s_buf,
                &mut p_buf,
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Causal softmax failed at head {}: {}", h, e),
                    op,
                )
            })?;

            // O_h = P @ V_h   ([seq, seq] @ [seq, d] -> [seq, d])
            let p_desc = MatrixDesc::<f32>::from_raw(
                p_buf.as_device_ptr(),
                seq_u32,
                seq_u32,
                seq_u32,
                Layout::RowMajor,
            );
            let v_desc = MatrixDesc::<f32>::from_raw(
                v.as_device_ptr() + head_off,
                seq_u32,
                dim_u32_,
                dim_u32_,
                Layout::RowMajor,
            );
            let mut o_desc = MatrixDescMut::<f32>::from_raw(
                out.as_device_ptr() + head_off,
                seq_u32,
                dim_u32_,
                dim_u32_,
                Layout::RowMajor,
            );
            gemm::<f32>(
                &self.handle,
                Transpose::NoTrans,
                Transpose::NoTrans,
                1.0f32,
                &p_desc,
                &v_desc,
                0.0f32,
                &mut o_desc,
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Output GEMM failed at head {}: {}", h, e),
                    op,
                )
            })?;
        }

        // Quiesce the stream before the scratch buffers drop (the async
        // launches must not race cuMemFree) and before the caller reads `out`.
        self.sync_context(op)?;
        drop(s_buf);
        drop(p_buf);

        let output_id = OxiCudaBufferId::new();
        cache.insert(output_id, out);
        Ok(output_id)
    }

    /// Single-query **decode** attention on resident buffers, result resident:
    /// one new token attends to the whole KV cache (no mask — every cached key
    /// is in the past).
    ///
    /// Layouts (packed, batch 1): `q` is `[H, d]` (one query row per head),
    /// `k` and `v` are `[H, kv, d]` (the heads-major KV-cache layout; `K^T`
    /// is a GEMM transpose flag, transpose-aware since oxicuda 0.4.1); the
    /// output is `[H, d]` — a `[1, H * d]` hidden row. Scores for all heads
    /// are packed as `[H, kv]` so the softmax over every head is one kernel
    /// launch.
    #[allow(clippy::too_many_arguments)] // Attention geometry (3 operands + 4 dims); grouping adds nothing.
    pub fn attention_decode_gpu_to_gpu(
        &self,
        q_id: &OxiCudaBufferId,
        k_id: &OxiCudaBufferId,
        v_id: &OxiCudaBufferId,
        num_heads: usize,
        kv_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> crate::errors::Result<OxiCudaBufferId> {
        let op = "attention_decode_gpu_to_gpu";
        let q_len = checked_len(&[num_heads, head_dim], op)?;
        let kv_mat = checked_len(&[head_dim, kv_len], op)?;
        let kv_total = checked_len(&[num_heads, kv_mat], op)?;
        let scores = checked_len(&[num_heads, kv_len], op)?;
        if q_len == 0 || kv_len == 0 {
            return Err(TrustformersError::shape_error(format!(
                "{}: dimensions must be non-zero (heads={}, kv={}, head_dim={})",
                op, num_heads, kv_len, head_dim
            )));
        }
        let heads_u32 = dim_u32(num_heads, "num_heads", op)?;
        let kv_u32 = dim_u32(kv_len, "kv_len", op)?;
        let dim_u32_ = dim_u32(head_dim, "head_dim", op)?;

        self.sync_context(op)?;
        let mut cache = self
            .buffer_cache
            .lock()
            .map_err(|_| TrustformersError::hardware_error("Failed to lock buffer cache", op))?;
        let missing = |id: &OxiCudaBufferId| {
            TrustformersError::hardware_error(&format!("Buffer {:?} not found in cache", id), op)
        };
        let q = cache.get(q_id).ok_or_else(|| missing(q_id))?;
        let k = cache.get(k_id).ok_or_else(|| missing(k_id))?;
        let v = cache.get(v_id).ok_or_else(|| missing(v_id))?;
        if q.len() != q_len {
            return Err(TrustformersError::shape_error(format!(
                "{}: query length {} doesn't match heads {} * head_dim {}",
                op,
                q.len(),
                num_heads,
                head_dim
            )));
        }
        if k.len() != kv_total || v.len() != kv_total {
            return Err(TrustformersError::shape_error(format!(
                "{}: K/V lengths ({}, {}) must equal heads {} * head_dim {} * kv {}",
                op,
                k.len(),
                v.len(),
                num_heads,
                head_dim,
                kv_len
            )));
        }

        let s_buf = DeviceBuffer::<f32>::zeroed(scores).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate score scratch on device: {}", e),
                op,
            )
        })?;
        let mut p_buf = DeviceBuffer::<f32>::alloc(scores).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate probability scratch on device: {}", e),
                op,
            )
        })?;
        let out = DeviceBuffer::<f32>::zeroed(q_len).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer on device: {}", e),
                op,
            )
        })?;

        let elem = std::mem::size_of::<f32>() as u64;
        // S[h, :] = scale * q_h @ K_h^T   ([1, d] @ [kv, d]^T -> [1, kv])
        for h in 0..num_heads {
            let q_desc = MatrixDesc::<f32>::from_raw(
                q.as_device_ptr() + (h * head_dim) as u64 * elem,
                1,
                dim_u32_,
                dim_u32_,
                Layout::RowMajor,
            );
            let k_desc = MatrixDesc::<f32>::from_raw(
                k.as_device_ptr() + (h * kv_mat) as u64 * elem,
                kv_u32,
                dim_u32_,
                dim_u32_,
                Layout::RowMajor,
            );
            let mut s_desc = MatrixDescMut::<f32>::from_raw(
                s_buf.as_device_ptr() + (h * kv_len) as u64 * elem,
                1,
                kv_u32,
                kv_u32,
                Layout::RowMajor,
            );
            gemm::<f32>(
                &self.handle,
                Transpose::NoTrans,
                Transpose::Trans,
                scale,
                &q_desc,
                &k_desc,
                0.0f32,
                &mut s_desc,
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Score GEMM failed at head {}: {}", h, e),
                    op,
                )
            })?;
        }

        // One softmax over all heads: rows are independent, one row per head.
        oxicuda_blas::reduction::softmax::<f32>(
            &self.handle,
            heads_u32,
            kv_u32,
            &s_buf,
            &mut p_buf,
        )
        .map_err(|e| {
            TrustformersError::hardware_error(&format!("Softmax execution failed: {}", e), op)
        })?;

        // O[h, :] = P[h, :] @ V_h   ([1, kv] @ [kv, d] -> [1, d])
        for h in 0..num_heads {
            let p_desc = MatrixDesc::<f32>::from_raw(
                p_buf.as_device_ptr() + (h * kv_len) as u64 * elem,
                1,
                kv_u32,
                kv_u32,
                Layout::RowMajor,
            );
            let v_desc = MatrixDesc::<f32>::from_raw(
                v.as_device_ptr() + (h * kv_mat) as u64 * elem,
                kv_u32,
                dim_u32_,
                dim_u32_,
                Layout::RowMajor,
            );
            let mut o_desc = MatrixDescMut::<f32>::from_raw(
                out.as_device_ptr() + (h * head_dim) as u64 * elem,
                1,
                dim_u32_,
                dim_u32_,
                Layout::RowMajor,
            );
            gemm::<f32>(
                &self.handle,
                Transpose::NoTrans,
                Transpose::NoTrans,
                1.0f32,
                &p_desc,
                &v_desc,
                0.0f32,
                &mut o_desc,
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Output GEMM failed at head {}: {}", h, e),
                    op,
                )
            })?;
        }

        self.sync_context(op)?;
        drop(s_buf);
        drop(p_buf);

        let output_id = OxiCudaBufferId::new();
        cache.insert(output_id, out);
        Ok(output_id)
    }

    /// Append new rows to a heads-major KV-cache buffer:
    /// `[H, kv_old, d] ++ [H, kv_new, d] -> [H, kv_old + kv_new, d]`, all
    /// resident. Serves both keys and values (they share the heads-major
    /// layout now that `K^T` is a GEMM transpose flag). `prev_id = None`
    /// (with `kv_old = 0`) starts a fresh cache. The inputs are left
    /// untouched (the caller's handles still own them); the returned buffer
    /// is new.
    #[allow(clippy::too_many_arguments)] // Cache-append geometry; no natural grouping struct.
    pub fn concat_v_cache_gpu_to_gpu(
        &self,
        prev_id: Option<&OxiCudaBufferId>,
        new_id: &OxiCudaBufferId,
        num_heads: usize,
        kv_old: usize,
        kv_new: usize,
        head_dim: usize,
    ) -> crate::errors::Result<OxiCudaBufferId> {
        let op = "concat_v_cache_gpu_to_gpu";
        if prev_id.is_some() != (kv_old > 0) {
            return Err(TrustformersError::shape_error(format!(
                "{}: previous cache presence must match kv_old {} (prev given: {})",
                op,
                kv_old,
                prev_id.is_some()
            )));
        }
        let (old_plan, new_plan) = concat_v_plan(num_heads, kv_old, kv_new, head_dim)?;
        let out_len = checked_len(&[num_heads, kv_old + kv_new, head_dim], op)?;

        self.sync_context(op)?;
        let mut cache = self
            .buffer_cache
            .lock()
            .map_err(|_| TrustformersError::hardware_error("Failed to lock buffer cache", op))?;
        let missing = |id: &OxiCudaBufferId| {
            TrustformersError::hardware_error(&format!("Buffer {:?} not found in cache", id), op)
        };
        let mut out = DeviceBuffer::<f32>::alloc(out_len).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer on device: {}", e),
                op,
            )
        })?;
        if let Some(prev_id) = prev_id {
            let prev = cache.get(prev_id).ok_or_else(|| missing(prev_id))?;
            run_pitched_copies(prev, &mut out, &old_plan, op)?;
        }
        let new_buf = cache.get(new_id).ok_or_else(|| missing(new_id))?;
        run_pitched_copies(new_buf, &mut out, &new_plan, op)?;

        // Final sync before publishing the id: on failure the still-local
        // output buffer drops (freed) instead of leaking in the cache.
        self.sync_context(op)?;
        let output_id = OxiCudaBufferId::new();
        cache.insert(output_id, out);
        Ok(output_id)
    }
}

// ---------------------------------------------------------------------------
// Hardware-free plan tests (run everywhere, including macOS)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod plan_tests {
    use super::*;

    #[test]
    fn pitched_copy_validation_accepts_exact_fit() -> crate::errors::Result<()> {
        // 2 rows of 3 elements out of a 2x5 source into a 2x3 destination.
        let copy = PitchedCopy {
            src_offset: 1,
            dst_offset: 0,
            src_pitch: 5,
            dst_pitch: 3,
            width: 3,
            height: 2,
        };
        copy.validate(10, 6)?;
        Ok(())
    }

    #[test]
    fn pitched_copy_validation_rejects_bad_regions() {
        let base = PitchedCopy {
            src_offset: 0,
            dst_offset: 0,
            src_pitch: 4,
            dst_pitch: 4,
            width: 4,
            height: 2,
        };
        // Empty copies.
        assert!(PitchedCopy {
            width: 0,
            ..base.clone()
        }
        .validate(8, 8)
        .is_err());
        assert!(PitchedCopy {
            height: 0,
            ..base.clone()
        }
        .validate(8, 8)
        .is_err());
        // Pitch narrower than width.
        assert!(PitchedCopy {
            src_pitch: 3,
            ..base.clone()
        }
        .validate(8, 8)
        .is_err());
        assert!(PitchedCopy {
            dst_pitch: 3,
            ..base.clone()
        }
        .validate(8, 8)
        .is_err());
        // Source / destination overrun (needs 8, offset pushes past the end).
        assert!(PitchedCopy {
            src_offset: 1,
            ..base.clone()
        }
        .validate(8, 8)
        .is_err());
        assert!(PitchedCopy {
            dst_offset: 1,
            ..base.clone()
        }
        .validate(8, 8)
        .is_err());
        // Exact fit passes.
        assert!(base.validate(8, 8).is_ok());
    }

    #[test]
    fn gather_plan_gpt2_qkv_split_heads_major() -> crate::errors::Result<()> {
        // GPT-2 layout: row = [Q(hidden) | K(hidden) | V(hidden)], hidden = 4,
        // H = 2, d = 2, seq = 3. Gather the K component ([H, seq, d] packed).
        let plan = gather_heads_plan(3, 12, 4, 2, 2, 2, true)?;
        assert_eq!(
            plan,
            vec![
                PitchedCopy {
                    src_offset: 4,
                    dst_offset: 0,
                    src_pitch: 12,
                    dst_pitch: 2,
                    width: 2,
                    height: 3,
                },
                PitchedCopy {
                    src_offset: 6,
                    dst_offset: 6,
                    src_pitch: 12,
                    dst_pitch: 2,
                    width: 2,
                    height: 3,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn gather_plan_neox_qkv_split_rope_layout() -> crate::errors::Result<()> {
        // GPT-NeoX layout: row = [h0: q k v | h1: q k v], d = 2, H = 2,
        // hidden = 4, row stride = 12. Gather Q into the [seq, H, d] RoPE
        // layout (heads interleaved per row).
        let plan = gather_heads_plan(2, 12, 0, 6, 2, 2, false)?;
        assert_eq!(
            plan,
            vec![
                PitchedCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    src_pitch: 12,
                    dst_pitch: 4,
                    width: 2,
                    height: 2,
                },
                PitchedCopy {
                    src_offset: 6,
                    dst_offset: 2,
                    src_pitch: 12,
                    dst_pitch: 4,
                    width: 2,
                    height: 2,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn gather_plan_head_merge_layout() -> crate::errors::Result<()> {
        // Merge [H=2, seq=3, d=2] -> [seq, H*d]: source rows are the packed
        // per-head matrices (pitch d), heads step by seq*d.
        let plan = gather_heads_plan(3, 2, 0, 6, 2, 2, false)?;
        assert_eq!(
            plan,
            vec![
                PitchedCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    src_pitch: 2,
                    dst_pitch: 4,
                    width: 2,
                    height: 3,
                },
                PitchedCopy {
                    src_offset: 6,
                    dst_offset: 2,
                    src_pitch: 2,
                    dst_pitch: 4,
                    width: 2,
                    height: 3,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn concat_v_plan_appends_contiguous_chunks() -> crate::errors::Result<()> {
        // H = 2, kv_old = 2, kv_new = 1, d = 3 -> per-head chunks of 6/3/9.
        let (old_plan, new_plan) = concat_v_plan(2, 2, 1, 3)?;
        assert_eq!(old_plan.len(), 2);
        assert_eq!(new_plan.len(), 2);
        assert_eq!((old_plan[0].src_offset, old_plan[0].dst_offset), (0, 0));
        assert_eq!((old_plan[1].src_offset, old_plan[1].dst_offset), (6, 9));
        assert_eq!((new_plan[0].src_offset, new_plan[0].dst_offset), (0, 6));
        assert_eq!((new_plan[1].src_offset, new_plan[1].dst_offset), (3, 15));
        assert!(old_plan.iter().chain(new_plan.iter()).all(|c| c.height == 1));
        Ok(())
    }

    #[test]
    fn concat_plans_reject_empty_or_mismatched_dims() {
        assert!(concat_v_plan(0, 1, 1, 2).is_err());
        assert!(concat_v_plan(2, 1, 0, 2).is_err());
        assert!(concat_v_plan(2, 0, 0, 2).is_err());
        assert!(gather_heads_plan(0, 4, 0, 2, 2, 2, true).is_err());
    }

    #[test]
    fn fresh_cache_concat_has_no_old_copies() -> crate::errors::Result<()> {
        let (old_plan, new_plan) = concat_v_plan(2, 0, 3, 2)?;
        assert!(old_plan.is_empty());
        assert_eq!(new_plan.len(), 2);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Availability-gated GPU tests (skip cleanly without a CUDA device)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod gpu_tests {
    use super::super::{oxicuda_backend, oxicuda_cuda_available, OxiCudaBufferHandle};
    use crate::tensor::Tensor;

    fn assert_close(result: &[f32], expected: &[f32]) {
        assert_eq!(result.len(), expected.len());
        for (idx, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "mismatch at {}: got {} expected {}",
                idx,
                got,
                want
            );
        }
    }

    /// CPU reference for the GPT-NeoX half-split partial RoPE at an absolute
    /// position offset (matches the device kernel's frequency convention).
    fn rope_reference(
        input: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_ndims: usize,
        base: f32,
        position_offset: usize,
    ) -> Vec<f32> {
        let mut out = input.to_vec();
        let half = rotary_ndims / 2;
        for pos in 0..seq_len {
            for h in 0..num_heads {
                let off = (pos * num_heads + h) * head_dim;
                for i in 0..half {
                    let freq = base.powf(-2.0 * (i as f32) / (rotary_ndims as f32));
                    let angle = ((pos + position_offset) as f32) * freq;
                    let (s, c) = angle.sin_cos();
                    let x_i = input[off + i];
                    let x_j = input[off + i + half];
                    out[off + i] = x_i * c - x_j * s;
                    out[off + i + half] = x_i * s + x_j * c;
                }
            }
        }
        out
    }

    /// Naive CPU multi-head causal prefill attention over the packed layouts
    /// used by `attention_prefill_gpu_to_gpu` (q/v: `[H, seq, d]`).
    fn prefill_reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        num_heads: usize,
        seq: usize,
        d: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; num_heads * seq * d];
        for h in 0..num_heads {
            let base = h * seq * d;
            for i in 0..seq {
                // Scores for query i over keys 0..=i.
                let mut scores = vec![0.0f32; i + 1];
                for (j, score) in scores.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for l in 0..d {
                        acc += q[base + i * d + l] * k[base + j * d + l];
                    }
                    *score = acc * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for score in scores.iter_mut() {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                for l in 0..d {
                    let mut acc = 0.0f32;
                    for (j, &p) in scores.iter().enumerate() {
                        acc += (p / sum) * v[base + j * d + l];
                    }
                    out[base + i * d + l] = acc;
                }
            }
        }
        out
    }

    #[test]
    fn oxicuda_gather_heads_round_trip() -> crate::errors::Result<()> {
        if !oxicuda_cuda_available() {
            eprintln!("Skipping oxicuda gather-heads test: no CUDA device available");
            return Ok(());
        }

        // GPT-2 style QKV rows: seq = 3, hidden = 4 (H = 2, d = 2), row = 12.
        let (seq, hidden, num_heads, head_dim) = (3usize, 4usize, 2usize, 2usize);
        let row = 3 * hidden;
        let qkv: Vec<f32> = (0..seq * row).map(|i| i as f32).collect();

        let backend = oxicuda_backend(0)?;
        let qkv_id = backend.create_persistent_buffer(&qkv)?;
        let qkv_handle = OxiCudaBufferHandle::new(qkv_id, 0);

        // K heads-major gather: expected [H, seq, d].
        let k_id = backend.gather_heads_gpu_to_gpu(
            &qkv_handle.id(),
            seq,
            row,
            hidden,
            head_dim,
            num_heads,
            head_dim,
            true,
        )?;
        let k_handle = OxiCudaBufferHandle::new(k_id, 0);
        let mut expected_k = Vec::new();
        for h in 0..num_heads {
            for r in 0..seq {
                for l in 0..head_dim {
                    expected_k.push(qkv[r * row + hidden + h * head_dim + l]);
                }
            }
        }
        assert_close(&backend.download_buffer(&k_handle.id())?, &expected_k);

        // Merge back: [H, seq, d] -> [seq, H*d] must equal the K rows.
        let merged_id = backend.gather_heads_gpu_to_gpu(
            &k_handle.id(),
            seq,
            head_dim,
            0,
            seq * head_dim,
            num_heads,
            head_dim,
            false,
        )?;
        let merged_handle = OxiCudaBufferHandle::new(merged_id, 0);
        let mut expected_merged = Vec::new();
        for r in 0..seq {
            for c in 0..hidden {
                expected_merged.push(qkv[r * row + hidden + c]);
            }
        }
        assert_close(
            &backend.download_buffer(&merged_handle.id())?,
            &expected_merged,
        );
        Ok(())
    }

    #[test]
    fn oxicuda_resident_rope_parity_with_offset() -> crate::errors::Result<()> {
        if !oxicuda_cuda_available() {
            eprintln!("Skipping oxicuda resident-RoPE test: no CUDA device available");
            return Ok(());
        }

        let (seq, num_heads, head_dim, rotary, base) = (2usize, 2usize, 6usize, 4usize, 10000.0);
        let total = seq * num_heads * head_dim;
        let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.31 - 1.5).collect();

        let backend = oxicuda_backend(0)?;
        for offset in [0usize, 3] {
            let in_id = backend.create_persistent_buffer(&input)?;
            let in_handle = OxiCudaBufferHandle::new(in_id, 0);
            let out_id = backend.rope_neox_gpu_to_gpu(
                &in_handle.id(),
                seq,
                num_heads,
                head_dim,
                rotary,
                base,
                offset,
            )?;
            let out_handle = OxiCudaBufferHandle::new(out_id, 0);
            let expected = rope_reference(&input, seq, num_heads, head_dim, rotary, base, offset);
            assert_close(&backend.download_buffer(&out_handle.id())?, &expected);
        }
        Ok(())
    }

    #[test]
    fn oxicuda_resident_softmax_parity() -> crate::errors::Result<()> {
        if !oxicuda_cuda_available() {
            eprintln!("Skipping oxicuda resident-softmax test: no CUDA device available");
            return Ok(());
        }

        let (rows, cols) = (3usize, 4usize);
        let input: Vec<f32> = (0..rows * cols).map(|i| ((i * 7 % 5) as f32) * 0.4 - 1.0).collect();
        let backend = oxicuda_backend(0)?;

        // Causal variant.
        let in_id = backend.create_persistent_buffer(&input)?;
        let in_handle = OxiCudaBufferHandle::new(in_id, 0);
        let causal_id = backend.softmax_causal_gpu_to_gpu(&in_handle.id(), rows, cols)?;
        let causal_handle = OxiCudaBufferHandle::new(causal_id, 0);
        let mut expected_causal = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let live = (r + 1).min(cols);
            let base = r * cols;
            let max = input[base..base + live].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = input[base..base + live].iter().map(|&x| (x - max).exp()).sum();
            for j in 0..live {
                expected_causal[base + j] = (input[base + j] - max).exp() / sum;
            }
        }
        assert_close(
            &backend.download_buffer(&causal_handle.id())?,
            &expected_causal,
        );

        // Plain row-wise variant.
        let rows_id = backend.softmax_rows_gpu_to_gpu(&in_handle.id(), rows, cols)?;
        let rows_handle = OxiCudaBufferHandle::new(rows_id, 0);
        let mut expected_rows = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let base = r * cols;
            let max = input[base..base + cols].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = input[base..base + cols].iter().map(|&x| (x - max).exp()).sum();
            for j in 0..cols {
                expected_rows[base + j] = (input[base + j] - max).exp() / sum;
            }
        }
        assert_close(&backend.download_buffer(&rows_handle.id())?, &expected_rows);
        Ok(())
    }

    #[test]
    fn oxicuda_resident_add_and_tensor_add() -> crate::errors::Result<()> {
        if !oxicuda_cuda_available() {
            eprintln!("Skipping oxicuda resident-add test: no CUDA device available");
            return Ok(());
        }

        let a: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..8).map(|i| 4.0 - i as f32).collect();
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();

        // Backend-level primitive.
        let backend = oxicuda_backend(0)?;
        let a_id = backend.create_persistent_buffer(&a)?;
        let a_handle = OxiCudaBufferHandle::new(a_id, 0);
        let b_id = backend.create_persistent_buffer(&b)?;
        let b_handle = OxiCudaBufferHandle::new(b_id, 0);
        let sum_id = backend.add_gpu_to_gpu(&a_handle.id(), &b_handle.id(), a.len())?;
        let sum_handle = OxiCudaBufferHandle::new(sum_id, 0);
        assert_close(&backend.download_buffer(&sum_handle.id())?, &expected);

        // Tensor-level residual add must stay resident (no host bounce).
        let device = crate::device::Device::CUDA(0);
        let a_dev = Tensor::from_vec(a.clone(), &[2, 4])?.to_device_enum(&device)?;
        let b_dev = Tensor::from_vec(b.clone(), &[2, 4])?.to_device_enum(&device)?;
        let sum_dev = a_dev.add(&b_dev)?;
        match &sum_dev {
            Tensor::CUDA(data) => assert_eq!(data.shape, vec![2, 4]),
            other => panic!("expected resident CUDA sum, got {:?}", other),
        }
        match sum_dev.to_device_enum(&crate::device::Device::CPU)? {
            Tensor::F32(arr) => {
                let flat: Vec<f32> = arr.iter().copied().collect();
                assert_close(&flat, &expected);
            },
            other => panic!("expected downloadable F32 sum, got {:?}", other),
        }
        Ok(())
    }

    #[test]
    fn oxicuda_resident_prefill_attention_parity() -> crate::errors::Result<()> {
        if !oxicuda_cuda_available() {
            eprintln!("Skipping oxicuda prefill-attention test: no CUDA device available");
            return Ok(());
        }

        let (num_heads, seq, d) = (2usize, 3usize, 2usize);
        let total = num_heads * seq * d;
        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.37).sin()).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.53).cos()).collect();
        let v: Vec<f32> = (0..total).map(|i| i as f32 * 0.25 - 1.0).collect();
        let scale = 1.0 / (d as f32).sqrt();

        let backend = oxicuda_backend(0)?;
        let q_id = backend.create_persistent_buffer(&q)?;
        let q_handle = OxiCudaBufferHandle::new(q_id, 0);
        let v_id = backend.create_persistent_buffer(&v)?;
        let v_handle = OxiCudaBufferHandle::new(v_id, 0);
        // K stays heads-major [H, seq, d]; K^T is a GEMM transpose flag.
        let k_id = backend.create_persistent_buffer(&k)?;
        let k_handle = OxiCudaBufferHandle::new(k_id, 0);

        let out_id = backend.attention_prefill_gpu_to_gpu(
            &q_handle.id(),
            &k_handle.id(),
            &v_handle.id(),
            num_heads,
            seq,
            d,
            scale,
        )?;
        let out_handle = OxiCudaBufferHandle::new(out_id, 0);

        let expected = prefill_reference(&q, &k, &v, num_heads, seq, d, scale);
        assert_close(&backend.download_buffer(&out_handle.id())?, &expected);
        Ok(())
    }

    #[test]
    fn oxicuda_resident_decode_attention_parity() -> crate::errors::Result<()> {
        if !oxicuda_cuda_available() {
            eprintln!("Skipping oxicuda decode-attention test: no CUDA device available");
            return Ok(());
        }

        // Decode must equal the last row of an equivalent prefill: the final
        // query of a causal prefill attends to every key, which is exactly
        // the unmasked decode softmax.
        let (num_heads, seq, d) = (2usize, 3usize, 2usize);
        let total = num_heads * seq * d;
        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.41).sin()).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.29).cos()).collect();
        let v: Vec<f32> = (0..total).map(|i| 1.5 - i as f32 * 0.2).collect();
        let scale = 1.0 / (d as f32).sqrt();

        // Last-token query per head: [H, d].
        let mut q_last = Vec::with_capacity(num_heads * d);
        for h in 0..num_heads {
            q_last.extend_from_slice(&q[h * seq * d + (seq - 1) * d..h * seq * d + seq * d]);
        }

        let backend = oxicuda_backend(0)?;
        let q_id = backend.create_persistent_buffer(&q_last)?;
        let q_handle = OxiCudaBufferHandle::new(q_id, 0);
        let v_id = backend.create_persistent_buffer(&v)?;
        let v_handle = OxiCudaBufferHandle::new(v_id, 0);
        // K stays heads-major [H, kv, d]; K^T is a GEMM transpose flag.
        let k_id = backend.create_persistent_buffer(&k)?;
        let k_handle = OxiCudaBufferHandle::new(k_id, 0);

        let out_id = backend.attention_decode_gpu_to_gpu(
            &q_handle.id(),
            &k_handle.id(),
            &v_handle.id(),
            num_heads,
            seq,
            d,
            scale,
        )?;
        let out_handle = OxiCudaBufferHandle::new(out_id, 0);

        let full = prefill_reference(&q, &k, &v, num_heads, seq, d, scale);
        let mut expected = Vec::with_capacity(num_heads * d);
        for h in 0..num_heads {
            expected.extend_from_slice(&full[h * seq * d + (seq - 1) * d..h * seq * d + seq * d]);
        }
        assert_close(&backend.download_buffer(&out_handle.id())?, &expected);
        Ok(())
    }

    #[test]
    fn oxicuda_resident_kv_cache_concat_parity() -> crate::errors::Result<()> {
        if !oxicuda_cuda_available() {
            eprintln!("Skipping oxicuda kv-concat test: no CUDA device available");
            return Ok(());
        }

        let (num_heads, d) = (2usize, 2usize);
        let (kv_old, kv_new) = (2usize, 1usize);

        // K pieces share the heads-major V layout: [H, kv_old, d] and
        // [H, kv_new, d] (K^T is a GEMM transpose flag, not a cache layout).
        let k_old: Vec<f32> = (0..num_heads * kv_old * d).map(|i| i as f32).collect();
        let k_new: Vec<f32> = (0..num_heads * kv_new * d).map(|i| 100.0 + i as f32).collect();
        let mut expected_k = Vec::new();
        for h in 0..num_heads {
            expected_k.extend_from_slice(&k_old[h * kv_old * d..(h + 1) * kv_old * d]);
            expected_k.extend_from_slice(&k_new[h * kv_new * d..(h + 1) * kv_new * d]);
        }

        let backend = oxicuda_backend(0)?;
        let old_id = backend.create_persistent_buffer(&k_old)?;
        let old_handle = OxiCudaBufferHandle::new(old_id, 0);
        let new_id = backend.create_persistent_buffer(&k_new)?;
        let new_handle = OxiCudaBufferHandle::new(new_id, 0);
        let cat_id = backend.concat_v_cache_gpu_to_gpu(
            Some(&old_handle.id()),
            &new_handle.id(),
            num_heads,
            kv_old,
            kv_new,
            d,
        )?;
        let cat_handle = OxiCudaBufferHandle::new(cat_id, 0);
        assert_close(&backend.download_buffer(&cat_handle.id())?, &expected_k);

        // V pieces: [H, kv_old, d] and [H, kv_new, d].
        let v_old: Vec<f32> = (0..num_heads * kv_old * d).map(|i| i as f32 * 0.5).collect();
        let v_new: Vec<f32> = (0..num_heads * kv_new * d).map(|i| 50.0 + i as f32).collect();
        let mut expected_v = Vec::new();
        for h in 0..num_heads {
            expected_v.extend_from_slice(&v_old[h * kv_old * d..(h + 1) * kv_old * d]);
            expected_v.extend_from_slice(&v_new[h * kv_new * d..(h + 1) * kv_new * d]);
        }
        let vold_id = backend.create_persistent_buffer(&v_old)?;
        let vold_handle = OxiCudaBufferHandle::new(vold_id, 0);
        let vnew_id = backend.create_persistent_buffer(&v_new)?;
        let vnew_handle = OxiCudaBufferHandle::new(vnew_id, 0);
        let vcat_id = backend.concat_v_cache_gpu_to_gpu(
            Some(&vold_handle.id()),
            &vnew_handle.id(),
            num_heads,
            kv_old,
            kv_new,
            d,
        )?;
        let vcat_handle = OxiCudaBufferHandle::new(vcat_id, 0);
        assert_close(&backend.download_buffer(&vcat_handle.id())?, &expected_v);
        Ok(())
    }
}
