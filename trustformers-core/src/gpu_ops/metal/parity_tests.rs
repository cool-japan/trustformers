//! GPU-versus-CPU parity tests for the Metal fused-attention kernels, plus the
//! buffer-cache lifetime assertions.
//!
//! # Why these exist
//!
//! Before this suite the only Metal parity coverage was a 2x2 matmul and two GEMM
//! cases (4x5x3 and 64x64x64 - both comfortably under 256). Nothing exercised
//! `batched_scaled_matmul_softmax_causal_gpu_to_gpu`,
//! `batched_scaled_matmul_softmax_gen_gpu_to_gpu`, `attention_gpu_to_gpu` or
//! `attention_with_cache_gpu_to_gpu`, which is precisely why two out-of-bounds writes
//! and a total absence of buffer eviction went unnoticed:
//!
//! * the causal kernel kept the score row in a `float scores[256]` thread-private
//!   array, so every `seq_len > 256` overran it (GPT-2's context is 1024); and
//! * the decode kernel kept a `float scores[512]`, so any generation whose KV cache
//!   crossed 512 tokens overran that one.
//!
//! The sweeps below deliberately straddle both old limits (255/256/257 and
//! 511/512/513) and compare against a CPU reference. Run against the pre-fix kernels
//! they fail; against the online-softmax rewrite they pass.

#![cfg(all(test, target_os = "macos", feature = "metal"))]

use super::common::*;
use super::functions::get_metal_backend;
use super::metalbackend_type::MetalBackend;
use super::types::BufferId;

/// Deterministic pseudo-random fill in `[-1, 1)`, so a failure is reproducible.
fn fill(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 40) as u32; // 24 bits
            (bits as f32 / (1u32 << 23) as f32) - 1.0
        })
        .collect()
}

/// Read `n_elems` floats out of a resident buffer regardless of its storage mode.
fn read_back(backend: &MetalBackend, id: &BufferId, n_elems: usize) -> Result<Vec<f32>> {
    backend.download_buffer_via_staging(id, n_elems)
}

/// CPU reference for `batched_scaled_matmul_softmax_causal`.
///
/// Q is `[heads, seq, head_dim]`, K^T is `[heads, head_dim, seq]`; the result is
/// `[heads, seq, seq]` row-wise softmax with a causal mask.
fn cpu_causal_reference(
    q: &[f32],
    k_t: &[f32],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    alpha: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_heads * seq_len * seq_len];
    for h in 0..num_heads {
        let q_base = h * seq_len * head_dim;
        let k_base = h * head_dim * seq_len;
        let o_base = h * seq_len * seq_len;
        for row in 0..seq_len {
            let mut scores = vec![0.0f32; row + 1];
            let mut max_score = f32::NEG_INFINITY;
            for (col, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_base + row * head_dim + d] * k_t[k_base + d * seq_len + col];
                }
                *score = dot * alpha;
                max_score = max_score.max(*score);
            }
            let mut sum = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            for col in 0..seq_len {
                out[o_base + row * seq_len + col] =
                    if col <= row { scores[col] / sum } else { 0.0 };
            }
        }
    }
    out
}

/// CPU reference for `batched_scaled_matmul_softmax_gen` (no causal mask).
fn cpu_gen_reference(
    q: &[f32],
    k_t: &[f32],
    num_heads: usize,
    q_seq_len: usize,
    kv_seq_len: usize,
    head_dim: usize,
    alpha: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_heads * q_seq_len * kv_seq_len];
    for h in 0..num_heads {
        let q_base = h * q_seq_len * head_dim;
        let k_base = h * head_dim * kv_seq_len;
        let o_base = h * q_seq_len * kv_seq_len;
        for row in 0..q_seq_len {
            let mut scores = vec![0.0f32; kv_seq_len];
            let mut max_score = f32::NEG_INFINITY;
            for (col, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_base + row * head_dim + d] * k_t[k_base + d * kv_seq_len + col];
                }
                *score = dot * alpha;
                max_score = max_score.max(*score);
            }
            let mut sum = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            for (col, score) in scores.iter().enumerate() {
                out[o_base + row * kv_seq_len + col] = score / sum;
            }
        }
    }
    out
}

/// Assert every element agrees within `tolerance`, reporting the worst offender.
fn assert_close(got: &[f32], want: &[f32], tolerance: f32, context: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{context}: length {} != reference length {}",
        got.len(),
        want.len()
    );
    let mut worst = (0usize, 0.0f32);
    for (index, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let delta = (a - b).abs();
        if delta > worst.1 {
            worst = (index, delta);
        }
    }
    assert!(
        worst.1 <= tolerance,
        "{context}: worst mismatch {} at index {} (GPU {} vs CPU {}); tolerance {}",
        worst.1,
        worst.0,
        got[worst.0],
        want[worst.0],
        tolerance
    );
}

/// One causal-kernel case at a given sequence length.
fn run_causal_case(num_heads: usize, seq_len: usize, head_dim: usize) -> Result<()> {
    let backend = get_metal_backend()?;
    let alpha = 1.0f32 / (head_dim as f32).sqrt();
    let q = fill(num_heads * seq_len * head_dim, seq_len as u64 * 31 + 7);
    let k_t = fill(num_heads * head_dim * seq_len, seq_len as u64 * 17 + 3);

    let q_id = backend.create_transient_buffer(&q)?;
    let k_id = backend.create_transient_buffer(&k_t)?;
    let out_id = backend.batched_scaled_matmul_softmax_causal_gpu_to_gpu(
        &q_id, &k_id, num_heads, seq_len, head_dim, alpha,
    )?;
    let got = read_back(&backend, &out_id, num_heads * seq_len * seq_len)?;
    backend.release_buffers(&[q_id, k_id, out_id])?;

    let want = cpu_causal_reference(&q, &k_t, num_heads, seq_len, head_dim, alpha);
    assert_close(
        &got,
        &want,
        2e-5,
        &format!("causal heads={num_heads} seq_len={seq_len} head_dim={head_dim}"),
    );

    // Every unmasked row must still be a probability distribution.
    for h in 0..num_heads {
        for row in 0..seq_len {
            let base = h * seq_len * seq_len + row * seq_len;
            let sum: f32 = got[base..base + seq_len].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "causal seq_len={seq_len}: head {h} row {row} sums to {sum}, not 1"
            );
            for col in (row + 1)..seq_len {
                assert_eq!(
                    got[base + col],
                    0.0,
                    "causal seq_len={seq_len}: future position ({row}, {col}) must be masked"
                );
            }
        }
    }
    Ok(())
}

/// One decode-kernel case at a given KV length.
fn run_gen_case(
    num_heads: usize,
    q_seq_len: usize,
    kv_seq_len: usize,
    head_dim: usize,
) -> Result<()> {
    let backend = get_metal_backend()?;
    let alpha = 1.0f32 / (head_dim as f32).sqrt();
    let q = fill(num_heads * q_seq_len * head_dim, kv_seq_len as u64 * 13 + 5);
    let k_t = fill(
        num_heads * head_dim * kv_seq_len,
        kv_seq_len as u64 * 29 + 11,
    );

    let q_id = backend.create_transient_buffer(&q)?;
    let k_id = backend.create_transient_buffer(&k_t)?;
    let out_id = backend.batched_scaled_matmul_softmax_gen_gpu_to_gpu(
        &q_id, &k_id, num_heads, q_seq_len, kv_seq_len, head_dim, alpha,
    )?;
    let got = read_back(&backend, &out_id, num_heads * q_seq_len * kv_seq_len)?;
    backend.release_buffers(&[q_id, k_id, out_id])?;

    let want = cpu_gen_reference(&q, &k_t, num_heads, q_seq_len, kv_seq_len, head_dim, alpha);
    assert_close(
        &got,
        &want,
        2e-5,
        &format!("gen heads={num_heads} q_seq={q_seq_len} kv_seq={kv_seq_len}"),
    );
    for h in 0..num_heads {
        for row in 0..q_seq_len {
            let base = h * q_seq_len * kv_seq_len + row * kv_seq_len;
            let sum: f32 = got[base..base + kv_seq_len].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "gen kv_seq={kv_seq_len}: head {h} row {row} sums to {sum}, not 1"
            );
        }
    }
    Ok(())
}

/// P0 regression: the causal kernel at, around and far past its old 256 limit.
///
/// Against the pre-fix kernel every case with `seq_len > 256` wrote past
/// `float scores[256]` and produced garbage (or corrupted neighbouring state); the
/// 255/256 cases passed. This test therefore fails on the old code and passes on the
/// online-softmax rewrite.
#[test]
fn causal_fused_attention_parity_across_the_old_256_limit() -> Result<()> {
    for seq_len in [1usize, 2, 31, 64, 255, 256, 257, 300, 512, 513] {
        run_causal_case(2, seq_len, 16)?;
    }
    Ok(())
}

/// The same kernel at GPT-2's real context length.
#[test]
fn causal_fused_attention_parity_at_gpt2_context_length() -> Result<()> {
    run_causal_case(1, 1024, 64)
}

/// P0 regression: the decode kernel at, around and past its old 512 limit.
#[test]
fn generation_fused_attention_parity_across_the_old_512_limit() -> Result<()> {
    for kv_seq_len in [1usize, 8, 255, 511, 512, 513, 800, 1024] {
        run_gen_case(2, 1, kv_seq_len, 16)?;
    }
    Ok(())
}

/// A multi-token query against a long cache, as a prompt-continuation would produce.
#[test]
fn generation_fused_attention_parity_with_multi_token_query() -> Result<()> {
    run_gen_case(3, 4, 600, 32)
}

/// Degenerate shapes must be refused with a structured error, not dispatched.
#[test]
fn fused_attention_rejects_degenerate_and_undersized_shapes() -> Result<()> {
    let backend = get_metal_backend()?;
    let q_id = backend.create_transient_buffer(&fill(2 * 4 * 8, 1))?;
    let k_id = backend.create_transient_buffer(&fill(2 * 8 * 4, 2))?;

    // Zero dimension.
    let error = backend
        .batched_scaled_matmul_softmax_causal_gpu_to_gpu(&q_id, &k_id, 2, 0, 8, 0.5)
        .expect_err("seq_len = 0 must be rejected");
    assert!(format!("{error}").contains("non-zero"), "got: {error}");

    // Declared shape larger than the operand buffers.
    let error = backend
        .batched_scaled_matmul_softmax_causal_gpu_to_gpu(&q_id, &k_id, 2, 4096, 8, 0.5)
        .expect_err("an oversized declared shape must be rejected");
    let rendered = format!("{error}");
    assert!(
        rendered.contains("Q buffer holds") || rendered.contains("K^T buffer holds"),
        "error must name the undersized operand, got: {rendered}"
    );

    // Same for the generation variant.
    let error = backend
        .batched_scaled_matmul_softmax_gen_gpu_to_gpu(&q_id, &k_id, 2, 1, 4096, 8, 0.5)
        .expect_err("an oversized declared kv length must be rejected");
    assert!(format!("{error}").contains("buffer holds"), "got: {error}");

    backend.release_buffers(&[q_id, k_id])?;
    Ok(())
}

/// Full multi-head attention against a CPU reference, at a length that used to
/// overrun the fused kernel.
#[test]
fn attention_gpu_to_gpu_matches_cpu_reference_past_the_old_limit() -> Result<()> {
    let backend = get_metal_backend()?;
    let (seq_len, num_heads, head_dim) = (300usize, 2usize, 8usize);
    let hidden = num_heads * head_dim;

    let q = fill(seq_len * hidden, 101);
    let k = fill(seq_len * hidden, 202);
    let v = fill(seq_len * hidden, 303);

    let q_id = backend.create_transient_buffer(&q)?;
    let k_id = backend.create_transient_buffer(&k)?;
    let v_id = backend.create_transient_buffer(&v)?;
    let out_id =
        backend.attention_gpu_to_gpu(&q_id, &k_id, &v_id, 1, seq_len, num_heads, head_dim)?;
    let got = read_back(&backend, &out_id, seq_len * hidden)?;
    backend.release_buffers(&[q_id, k_id, v_id, out_id])?;

    // CPU reference: per-head causal softmax attention over [seq_len, hidden].
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut want = vec![0.0f32; seq_len * hidden];
    for h in 0..num_heads {
        let off = h * head_dim;
        for row in 0..seq_len {
            let mut scores = vec![0.0f32; row + 1];
            let mut max_score = f32::NEG_INFINITY;
            for (col, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[row * hidden + off + d] * k[col * hidden + off + d];
                }
                *score = dot * scale;
                max_score = max_score.max(*score);
            }
            let mut sum = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (col, score) in scores.iter().enumerate() {
                    acc += (score / sum) * v[col * hidden + off + d];
                }
                want[row * hidden + off + d] = acc;
            }
        }
    }

    assert_close(&got, &want, 5e-4, "attention_gpu_to_gpu seq_len=300");
    Ok(())
}

/// P0 regression: `attention_gpu_to_gpu` used to leak six GPU buffers per call - they
/// were inserted into the process-global cache and never removed, so a decode loop
/// grew the cache without bound. Assert the cache returns to its baseline.
#[test]
fn attention_releases_every_intermediate_buffer() -> Result<()> {
    let backend = get_metal_backend()?;
    let (seq_len, num_heads, head_dim) = (16usize, 2usize, 8usize);
    let hidden = num_heads * head_dim;

    let q_id = backend.create_transient_buffer(&fill(seq_len * hidden, 11))?;
    let k_id = backend.create_transient_buffer(&fill(seq_len * hidden, 22))?;
    let v_id = backend.create_transient_buffer(&fill(seq_len * hidden, 33))?;

    let baseline = backend.buffer_cache_stats()?;
    for _ in 0..8 {
        let out_id =
            backend.attention_gpu_to_gpu(&q_id, &k_id, &v_id, 1, seq_len, num_heads, head_dim)?;
        backend.release_buffers(&[out_id])?;
    }
    let after = backend.buffer_cache_stats()?;

    assert_eq!(
        after.entries, baseline.entries,
        "8 attention calls must leave the cache at its baseline entry count \
         (was {}, now {}); the old code added 6 entries per call",
        baseline.entries, after.entries
    );
    assert_eq!(
        after.live_bytes, baseline.live_bytes,
        "byte accounting must also return to baseline"
    );

    backend.release_buffers(&[q_id, k_id, v_id])?;
    Ok(())
}

/// Same assertion for the KV-cached decode path, driven as a real generation loop.
#[test]
fn cached_attention_decode_loop_does_not_grow_the_cache() -> Result<()> {
    let backend = get_metal_backend()?;
    let (num_heads, head_dim) = (2usize, 8usize);

    let baseline = backend.buffer_cache_stats()?;
    // 32 decode steps against a growing KV cache, exactly as generation does.
    for step in 1..=32usize {
        let kv_seq_len = step;
        let q_id = backend.create_transient_buffer(&fill(num_heads * head_dim, step as u64))?;
        let k_id = backend
            .create_transient_buffer(&fill(num_heads * kv_seq_len * head_dim, step as u64 + 1))?;
        let v_id = backend
            .create_transient_buffer(&fill(num_heads * kv_seq_len * head_dim, step as u64 + 2))?;
        let out_id = backend.attention_with_cache_gpu_to_gpu(
            &q_id, &k_id, &v_id, 1, 1, kv_seq_len, num_heads, head_dim,
        )?;
        backend.release_buffers(&[q_id, k_id, v_id, out_id])?;
    }
    let after = backend.buffer_cache_stats()?;

    assert_eq!(
        after.entries, baseline.entries,
        "a 32-step decode loop must not grow the buffer cache (was {}, now {})",
        baseline.entries, after.entries
    );
    Ok(())
}

/// The LRU cap must actually evict once the cache is driven past it, and pinned
/// weights must survive that pressure. Nothing evicted anything before this change.
/// Only the explicitly-`Evictable` scratch tier participates - see `types.rs`.
#[test]
fn buffer_cache_evicts_scratch_under_pressure_but_keeps_pinned_weights() -> Result<()> {
    let backend = get_metal_backend()?;
    let original_capacity = backend.buffer_cache_stats()?.capacity_bytes;

    // 1 MiB cap; 1024-float (4 KiB) buffers; 512 of them is 2 MiB.
    backend.set_buffer_cache_capacity_bytes(1024 * 1024)?;
    let weight_id = backend.create_persistent_buffer(&[1.0f32; 1024])?;
    let before = backend.buffer_cache_stats()?;

    let mut transient_ids = Vec::new();
    for index in 0..512u64 {
        transient_ids.push(backend.create_evictable_buffer(&fill(1024, index))?);
    }
    let after = backend.buffer_cache_stats()?;

    assert!(
        after.live_bytes <= after.capacity_bytes,
        "cache must respect its byte cap: {after:?}"
    );
    assert!(
        after.evicted_entries > before.evicted_entries,
        "driving 2 MiB of evictable scratch through a 1 MiB cap must evict: {after:?}"
    );
    assert!(
        backend.has_buffer(&weight_id)?,
        "a pinned weight buffer must survive eviction pressure"
    );
    // The earliest transients are gone; the most recent survive.
    assert!(
        !backend.has_buffer(&transient_ids[0])?,
        "the least-recently-used scratch buffer must have been evicted"
    );
    assert!(
        backend.has_buffer(&transient_ids[511])?,
        "the most recent scratch buffer must still be resident"
    );

    // Restore the process-global cache for the other tests in this binary.
    backend.remove_persistent_buffer(&weight_id)?;
    backend.release_buffers(&transient_ids)?;
    backend.set_buffer_cache_capacity_bytes(original_capacity)?;
    Ok(())
}

/// A `MetalBufferHandle` keeps its entry alive under eviction pressure and frees it
/// when the last clone drops.
#[test]
fn retained_buffers_survive_pressure_and_free_on_drop() -> Result<()> {
    let backend = get_metal_backend()?;
    let original_capacity = backend.buffer_cache_stats()?.capacity_bytes;
    backend.set_buffer_cache_capacity_bytes(512 * 1024)?;

    // Deliberately in the evictable tier so only the handle protects it.
    let id = backend.create_evictable_buffer(&[2.0f32; 1024])?;
    let handle = backend.retain_buffer(&id)?;
    let mut ids = Vec::new();
    for index in 0..256u64 {
        ids.push(backend.create_evictable_buffer(&fill(1024, index + 9000))?);
    }
    assert!(
        backend.has_buffer(&id)?,
        "a retained buffer must not be evicted"
    );
    assert_eq!(handle.id(), id, "the handle must address the same buffer");

    drop(handle);
    assert!(
        !backend.has_buffer(&id)?,
        "dropping the last handle must free the cache entry"
    );

    backend.release_buffers(&ids)?;
    backend.set_buffer_cache_capacity_bytes(original_capacity)?;
    Ok(())
}

/// `download_buffer_to_vec` must refuse a GPU-private buffer.
///
/// Note what this machine actually does: on Apple Silicon's unified memory,
/// `MTLBuffer::contents` on a `StorageModePrivate` allocation returns a *non-null*
/// pointer that reads as zeroes rather than the documented null. A null check alone
/// would therefore have handed the caller a silently zero-filled tensor. The guard
/// inspects `MTLResource.storageMode` instead, which is deterministic.
#[test]
fn downloading_a_private_buffer_is_refused_not_silently_zero_filled() -> Result<()> {
    let backend = get_metal_backend()?;
    let private = Arc::new(
        backend
            .device_for_tests()
            .new_buffer(4096, MTLResourceOptions::StorageModePrivate),
    );
    let id = backend.insert_buffer_for_tests(Arc::clone(&private))?;

    // Demonstrate the hazard the storage-mode guard exists for.
    assert!(
        !private.contents().is_null(),
        "on this machine a Private buffer hands back a non-null mapping, which is          exactly why a null check is not a sufficient guard"
    );

    let error = backend
        .download_buffer_to_vec(&id)
        .expect_err("a StorageModePrivate buffer must not be read as if CPU-mappable");
    assert!(
        format!("{error}").contains("not CPU-mappable"),
        "got: {error}"
    );

    // The staging path handles it correctly: blit into Shared, then read.
    let values = backend.download_buffer_via_staging(&id, 1024)?;
    assert_eq!(values.len(), 1024);
    backend.release_buffers(&[id])?;
    Ok(())
}

/// A Shared buffer still downloads normally through the fast path.
#[test]
fn downloading_a_shared_buffer_returns_its_contents() -> Result<()> {
    let backend = get_metal_backend()?;
    let payload = fill(256, 4242);
    let id = backend.create_transient_buffer(&payload)?;
    let got = backend.download_buffer_to_vec(&id)?;
    assert_eq!(got, payload, "Shared buffers must round-trip exactly");
    backend.release_buffers(&[id])?;
    Ok(())
}

/// The device query must return real facts about this machine.
#[test]
fn device_info_reports_the_live_device() -> Result<()> {
    let backend = get_metal_backend()?;
    let info = backend.device_info();
    assert!(!info.name.is_empty(), "MTLDevice.name must be non-empty");
    assert!(info.registry_id != 0, "registryID must be real");
    assert!(
        info.max_buffer_length > 64 * 1024 * 1024,
        "maxBufferLength looks wrong: {}",
        info.max_buffer_length
    );
    assert!(
        info.max_threads_per_threadgroup.0 >= 32,
        "maxThreadsPerThreadgroup.width looks wrong: {:?}",
        info.max_threads_per_threadgroup
    );
    println!(
        "Metal device: {} | registry {} | unified={} | apple_family={:?} | metal3={} \
         | max_buffer={} MiB | working_set={} MiB | oxicuda={}",
        info.name,
        info.registry_id,
        info.has_unified_memory,
        info.apple_gpu_family,
        info.supports_metal3,
        info.max_buffer_length / (1024 * 1024),
        info.recommended_max_working_set_size / (1024 * 1024),
        info.oxicuda_metal_available
    );
    Ok(())
}

/// Cross-queue ordering: trustformers' kernels and oxicuda-metal's GEMM run on two
/// *different* `MTLCommandQueue`s (oxicuda-metal creates its own in
/// `oxicuda-metal-0.5.5/src/device.rs`), and Metal guarantees no ordering across
/// queues. This interleaves `layernorm_gpu_to_gpu` (our queue, committed
/// asynchronously) straight into `matmul_gpu_to_gpu_mps` (oxicuda's queue) and
/// compares against a CPU reference, which is the shape of the hazard the `flush()`
/// before every oxicuda hand-off exists to close.
#[test]
fn layernorm_then_oxicuda_gemm_is_correctly_ordered() -> Result<()> {
    let backend = get_metal_backend()?;
    let (rows, hidden, out_dim) = (64usize, 128usize, 32usize);
    let eps = 1e-5f32;

    let input = fill(rows * hidden, 777);
    let gamma = fill(hidden, 778);
    let beta = fill(hidden, 779);
    let weight = fill(hidden * out_dim, 780);

    // Repeat so a race would have many chances to show up.
    for iteration in 0..16 {
        let input_id = backend.create_transient_buffer(&input)?;
        let gamma_id = backend.create_transient_buffer(&gamma)?;
        let beta_id = backend.create_transient_buffer(&beta)?;
        let weight_id = backend.create_transient_buffer(&weight)?;

        // Our queue, async commit ...
        let normed_id =
            backend.layernorm_gpu_to_gpu(&input_id, &gamma_id, &beta_id, rows, hidden, eps)?;
        // ... immediately consumed by oxicuda's queue.
        let out_id =
            backend.matmul_gpu_to_gpu_mps(&normed_id, &weight_id, rows, hidden, out_dim)?;
        let got = read_back(&backend, &out_id, rows * out_dim)?;
        backend.release_buffers(&[input_id, gamma_id, beta_id, weight_id, normed_id, out_id])?;

        // CPU reference: layernorm over the last dim, then GEMM.
        let mut normed = vec![0.0f32; rows * hidden];
        for row in 0..rows {
            let slice = &input[row * hidden..(row + 1) * hidden];
            let mean = slice.iter().sum::<f32>() / hidden as f32;
            let variance =
                slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden as f32;
            let inv_std = 1.0 / (variance + eps).sqrt();
            for col in 0..hidden {
                normed[row * hidden + col] = (slice[col] - mean) * inv_std * gamma[col] + beta[col];
            }
        }
        let mut want = vec![0.0f32; rows * out_dim];
        for row in 0..rows {
            for col in 0..out_dim {
                let mut acc = 0.0f32;
                for inner in 0..hidden {
                    acc += normed[row * hidden + inner] * weight[inner * out_dim + col];
                }
                want[row * out_dim + col] = acc;
            }
        }

        assert_close(
            &got,
            &want,
            5e-3,
            &format!("layernorm -> oxicuda gemm, iteration {iteration}"),
        );
    }
    Ok(())
}

/// **Discriminating test for the eviction policy.**
///
/// LRU eviction is only safe if nothing that is still *live* can be evicted. A
/// `Tensor::Metal` holds a bare `BufferId`, not a handle, so an op output that is
/// still referenced by a live tensor is - by construction - an unpinned, unretained
/// cache entry. This drives the cache far past its cap while holding such a tensor
/// and then uses it, which is exactly the failure mode a cap-and-evict design risks.
#[test]
fn a_live_op_output_survives_eviction_pressure() -> Result<()> {
    let backend = get_metal_backend()?;
    let original_capacity = backend.buffer_cache_stats()?.capacity_bytes;

    let (rows, hidden, out_dim) = (8usize, 16usize, 16usize);
    let a = fill(rows * hidden, 4001);
    let b = fill(hidden * out_dim, 4002);
    let a_id = backend.create_transient_buffer(&a)?;
    let b_id = backend.create_transient_buffer(&b)?;

    // The output of a real GPU op - the same thing `Linear::forward` puts inside a
    // `Tensor::Metal` and hands back to the model.
    let live_output = backend.matmul_gpu_to_gpu_mps(&a_id, &b_id, rows, hidden, out_dim)?;

    // Squeeze the cache hard: 256 KiB cap, then push 2 MiB of transients through it.
    backend.set_buffer_cache_capacity_bytes(256 * 1024)?;
    let mut churn = Vec::new();
    for index in 0..512u64 {
        churn.push(backend.create_evictable_buffer(&fill(1024, index + 50_000))?);
    }
    let stats = backend.buffer_cache_stats()?;
    assert!(
        stats.evicted_entries > 0,
        "the test must actually apply eviction pressure: {stats:?}"
    );

    // Now use the live output. If eviction can claim a live op result, this fails.
    let survived = backend.has_buffer(&live_output)?;
    let readback = backend.download_buffer_to_vec(&live_output);

    backend.release_buffers(&churn)?;
    backend.release_buffers(&[a_id, b_id, live_output])?;
    backend.set_buffer_cache_capacity_bytes(original_capacity)?;

    assert!(
        survived,
        "a live GPU op output was evicted under pressure; `Tensor::Metal` holds a bare \
         BufferId, so this would surface as 'Buffer not found in cache' mid-forward. \
         Op results must stay resident while the caller can still name them: {stats:?}"
    );
    let values = readback?;
    assert_eq!(values.len(), rows * out_dim);
    Ok(())
}
