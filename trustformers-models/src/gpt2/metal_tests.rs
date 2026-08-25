//! GPT-2 Metal GPU tests that really execute on the GPU.
//!
//! Every test here calls [`Gpt2LMHeadModel::weights_to_gpu`] (or drives the Metal
//! kernels directly) and then *proves* the GPU path ran, either by asserting the
//! result tensor is `Tensor::Metal` or by watching
//! [`metal_attention_call_count`](crate::gpt2::model::metal_attention_call_count)
//! advance. Constructing a model with `Device::Metal(0)` alone leaves every weight on
//! the host and computes the whole forward pass in `ndarray`, which is how a suite of
//! "Metal" tests previously passed while exercising nothing but CPU arithmetic.

#![cfg(all(target_os = "macos", feature = "metal"))]

use trustformers_core::device::Device;
use trustformers_core::gpu_ops::metal::{get_metal_backend, BufferId, MetalBackend};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Model, TokenizedInput};

use crate::gpt2::generation::GenerativeModel;
use crate::gpt2::model::metal_attention_call_count;
use crate::gpt2::{Gpt2Config, Gpt2LMHeadModel};

/// Small deterministic-shape config; the weights are random but every comparison in
/// this file is CPU-vs-GPU on the *same* instance, so randomness is irrelevant.
fn small_config(n_layer: usize, n_head: usize) -> Gpt2Config {
    Gpt2Config {
        vocab_size: 50,
        n_positions: 32,
        n_embd: 32,
        n_layer,
        n_head,
        ..Default::default()
    }
}

fn tokenized(ids: Vec<u32>) -> TokenizedInput {
    let len = ids.len();
    TokenizedInput {
        input_ids: ids,
        attention_mask: vec![1u8; len],
        token_type_ids: None,
        special_tokens_mask: None,
        offset_mapping: None,
        overflowing_tokens: None,
    }
}

/// `Some(device)` when this machine really has a Metal GPU, `None` (with a notice) when
/// the suite is running somewhere the GPU tests cannot mean anything.
fn metal_device_or_skip(test: &str) -> Option<Device> {
    let device = Device::metal_if_available(0);
    if matches!(device, Device::Metal(_)) && get_metal_backend().is_ok() {
        Some(device)
    } else {
        eprintln!("{test}: no Metal device on this machine, skipping");
        None
    }
}

fn host_values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_device_enum(&Device::CPU)
        .expect("download tensor to host")
        .to_vec_f32()
        .expect("tensor is f32")
}

fn max_abs(values: &[f32]) -> f32 {
    values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared tensors have different lengths");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

/// Deterministic, non-repeating test pattern (no PRNG, so failures reproduce exactly).
fn pattern(len: usize, phase: f32) -> Vec<f32> {
    (0..len).map(|i| ((i as f32) * 0.317 + phase).sin() * 0.9).collect()
}

// ---------------------------------------------------------------------------
// Mission 1 - end-to-end prefill parity
// ---------------------------------------------------------------------------

/// A GPU-resident GPT-2 must produce the same prefill logits as the identical model on
/// the CPU.
///
/// Regression guard for a chain of defects that made this differ by 69-81% of the
/// signal magnitude: LayerNorm silently switching variance estimator by tensor rank on
/// Metal builds, and `Tensor::to_device_enum(Metal -> CPU)` reading `MTLBuffer::
/// contents()` without waiting for the queue (so the readback saw a freshly zeroed
/// allocation).
#[test]
fn metal_gpu_prefill_logits_match_cpu() {
    let Some(device) = metal_device_or_skip("metal_gpu_prefill_logits_match_cpu") else {
        return;
    };

    let mut model = Gpt2LMHeadModel::new_with_device(small_config(2, 2), Device::CPU)
        .expect("build model on CPU");
    let prompt = vec![1u32, 2, 3, 4, 5];

    let cpu_out = model.forward(tokenized(prompt.clone())).expect("cpu forward");
    assert!(
        matches!(cpu_out.logits, Tensor::F32(_)),
        "the reference forward must run on the CPU"
    );
    let cpu_logits = host_values(&cpu_out.logits);

    let calls_before = metal_attention_call_count();
    model.weights_to_gpu(&device).expect("upload weights to the GPU");
    let gpu_out = model.forward(tokenized(prompt)).expect("gpu forward");

    assert!(
        matches!(gpu_out.logits, Tensor::Metal(_)),
        "logits should still be GPU-resident after weights_to_gpu"
    );
    assert!(
        metal_attention_call_count() > calls_before,
        "the Metal attention fast path never ran - this test would be measuring CPU math"
    );

    let gpu_logits = host_values(&gpu_out.logits);
    let magnitude = max_abs(&cpu_logits);
    // Guard against a vacuous pass: a degenerate all-zero or constant logits tensor
    // would satisfy any relative-difference bound.
    let spread = cpu_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - cpu_logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        magnitude > 1e-3 && spread > 1e-3,
        "reference logits are degenerate (max|x| = {magnitude}, spread = {spread}); \
         a relative comparison against them would prove nothing"
    );
    let relative = max_abs_diff(&cpu_logits, &gpu_logits) / magnitude;
    assert!(
        relative < 1e-3,
        "GPU prefill logits diverge from CPU by {relative} of signal magnitude \
         (max|cpu| = {magnitude})"
    );
}

// ---------------------------------------------------------------------------
// Mission 1 - stage-by-stage kernel parity
// ---------------------------------------------------------------------------

/// Hand-written host reference for `split_qkv_gpu` (batch = 1).
fn reference_split_qkv(
    qkv: &[f32],
    seq_len: usize,
    hidden: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0; seq_len * hidden];
    let mut k = vec![0.0; seq_len * hidden];
    let mut v = vec![0.0; seq_len * hidden];
    for s in 0..seq_len {
        for h in 0..hidden {
            let base = s * 3 * hidden;
            q[s * hidden + h] = qkv[base + h];
            k[s * hidden + h] = qkv[base + hidden + h];
            v[s * hidden + h] = qkv[base + 2 * hidden + h];
        }
    }
    (q, k, v)
}

/// `[seq, num_heads * head_dim]` -> `[num_heads, seq, head_dim]`.
fn reference_to_heads(flat: &[f32], seq_len: usize, num_heads: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; seq_len * num_heads * head_dim];
    for h in 0..num_heads {
        for s in 0..seq_len {
            for d in 0..head_dim {
                out[h * seq_len * head_dim + s * head_dim + d] =
                    flat[s * num_heads * head_dim + h * head_dim + d];
            }
        }
    }
    out
}

/// `[num_heads, seq, head_dim]` -> `[seq, num_heads * head_dim]`.
fn reference_from_heads(
    heads: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; seq_len * num_heads * head_dim];
    for h in 0..num_heads {
        for s in 0..seq_len {
            for d in 0..head_dim {
                out[s * num_heads * head_dim + h * head_dim + d] =
                    heads[h * seq_len * head_dim + s * head_dim + d];
            }
        }
    }
    out
}

/// Textbook scaled dot-product attention over `[num_heads, *, head_dim]` buffers,
/// accumulated in `f64` so the comparison is against real arithmetic rather than
/// against another f32 kernel. `causal` masks key position `j > q_offset + i`.
fn reference_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    q_seq: usize,
    kv_seq: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0_f64 / (head_dim as f64).sqrt();
    // Query row i of a q_seq-long block sits at absolute position kv_seq - q_seq + i.
    let q_offset = kv_seq - q_seq;
    let mut out = vec![0.0_f32; num_heads * q_seq * head_dim];
    for h in 0..num_heads {
        for i in 0..q_seq {
            let last_key = if causal { q_offset + i } else { kv_seq - 1 };
            let mut scores = Vec::with_capacity(last_key + 1);
            for j in 0..=last_key {
                let mut dot = 0.0_f64;
                for d in 0..head_dim {
                    dot += q[h * q_seq * head_dim + i * head_dim + d] as f64
                        * k[h * kv_seq * head_dim + j * head_dim + d] as f64;
                }
                scores.push(dot * scale);
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for d in 0..head_dim {
                let mut acc = 0.0_f64;
                for (j, e) in exps.iter().enumerate() {
                    acc += (e / sum) * v[h * kv_seq * head_dim + j * head_dim + d] as f64;
                }
                out[h * q_seq * head_dim + i * head_dim + d] = acc as f32;
            }
        }
    }
    out
}

fn upload(backend: &MetalBackend, data: &[f32]) -> BufferId {
    backend.create_persistent_buffer(data).expect("upload buffer")
}

fn download(backend: &MetalBackend, id: &BufferId) -> Vec<f32> {
    backend.download_buffer_to_vec(id).expect("download buffer")
}

/// Every stage of the GPT-2 Metal attention block, in isolation, against a host
/// reference: `split_qkv_gpu` -> `reshape_to_heads_gpu` ->
/// `attention_with_cache_gpu_to_gpu` -> `reshape_from_heads_gpu`.
///
/// Deliberately uses non-power-of-two dimensions (3 heads x 5 dims, 7 positions) so a
/// threadgroup rounding or stride bug cannot hide behind a convenient shape.
#[test]
fn metal_attention_stages_match_host_reference_on_prefill() {
    let Some(_device) =
        metal_device_or_skip("metal_attention_stages_match_host_reference_on_prefill")
    else {
        return;
    };
    let backend = get_metal_backend().expect("metal backend");

    let (seq_len, num_heads, head_dim) = (7usize, 3usize, 5usize);
    let hidden = num_heads * head_dim;
    let qkv = pattern(seq_len * 3 * hidden, 0.11);
    let qkv_id = upload(&backend, &qkv);

    // Stage 1: split the fused projection.
    let (q_id, k_id, v_id) =
        backend.split_qkv_gpu(&qkv_id, 1, seq_len, hidden).expect("split_qkv_gpu");
    let (q_ref, k_ref, v_ref) = reference_split_qkv(&qkv, seq_len, hidden);
    assert_eq!(download(&backend, &q_id), q_ref, "split_qkv_gpu: Q slice");
    assert_eq!(download(&backend, &k_id), k_ref, "split_qkv_gpu: K slice");
    assert_eq!(download(&backend, &v_id), v_ref, "split_qkv_gpu: V slice");

    // Stage 2: interleaved hidden dim -> heads-major.
    let q_heads = backend
        .reshape_to_heads_gpu(&q_id, seq_len, num_heads, head_dim)
        .expect("reshape_to_heads_gpu");
    let k_heads = backend
        .reshape_to_heads_gpu(&k_id, seq_len, num_heads, head_dim)
        .expect("reshape_to_heads_gpu");
    let v_heads = backend
        .reshape_to_heads_gpu(&v_id, seq_len, num_heads, head_dim)
        .expect("reshape_to_heads_gpu");
    let q_heads_ref = reference_to_heads(&q_ref, seq_len, num_heads, head_dim);
    let k_heads_ref = reference_to_heads(&k_ref, seq_len, num_heads, head_dim);
    let v_heads_ref = reference_to_heads(&v_ref, seq_len, num_heads, head_dim);
    assert_eq!(
        download(&backend, &q_heads),
        q_heads_ref,
        "reshape_to_heads_gpu: Q"
    );
    assert_eq!(
        download(&backend, &k_heads),
        k_heads_ref,
        "reshape_to_heads_gpu: K"
    );
    assert_eq!(
        download(&backend, &v_heads),
        v_heads_ref,
        "reshape_to_heads_gpu: V"
    );

    // Stage 3: causal multi-head attention (prefill: q_seq == kv_seq).
    let attn_heads = backend
        .attention_with_cache_gpu_to_gpu(
            &q_heads, &k_heads, &v_heads, 1, seq_len, seq_len, num_heads, head_dim,
        )
        .expect("attention_with_cache_gpu_to_gpu");
    let attn_ref = reference_attention(
        &q_heads_ref,
        &k_heads_ref,
        &v_heads_ref,
        num_heads,
        seq_len,
        seq_len,
        head_dim,
        true,
    );
    let attn_gpu = download(&backend, &attn_heads);
    assert!(
        max_abs(&attn_ref) > 1e-3,
        "the host reference is degenerate; the comparison below would prove nothing"
    );
    let attn_err = max_abs_diff(&attn_gpu, &attn_ref);
    assert!(
        attn_err < 1e-5,
        "causal attention kernel differs from the host reference by {attn_err}"
    );
    // The causal mask must actually bite: row 0 attends to one key only, so its output
    // is exactly V[h, 0, :]. If the kernel silently dropped masking this would fail.
    for h in 0..num_heads {
        let row0 = &attn_gpu[h * seq_len * head_dim..h * seq_len * head_dim + head_dim];
        let v_row0 = &v_heads_ref[h * seq_len * head_dim..h * seq_len * head_dim + head_dim];
        assert!(
            max_abs_diff(row0, v_row0) < 1e-5,
            "head {h}: the first query row must attend to the first key alone"
        );
    }

    // Stage 4: heads-major -> flat hidden.
    let merged = backend
        .reshape_from_heads_gpu(&attn_heads, seq_len, num_heads, head_dim)
        .expect("reshape_from_heads_gpu");
    let merged_ref = reference_from_heads(&attn_ref, seq_len, num_heads, head_dim);
    let merged_err = max_abs_diff(&download(&backend, &merged), &merged_ref);
    assert!(
        merged_err < 1e-5,
        "reshape_from_heads_gpu differs from the host reference by {merged_err}"
    );

    backend
        .release_buffers(&[
            qkv_id, q_id, k_id, v_id, q_heads, k_heads, v_heads, attn_heads, merged,
        ])
        .expect("release test buffers");
}

/// The decode-shaped attention stage: one query row against a concatenated KV cache.
///
/// Covers `concat_kv_cache` plus the `q_seq_len != kv_seq_len` branch of
/// `attention_with_cache_gpu_to_gpu`, which the prefill test above never reaches.
#[test]
fn metal_attention_stages_match_host_reference_on_decode() {
    let Some(_device) =
        metal_device_or_skip("metal_attention_stages_match_host_reference_on_decode")
    else {
        return;
    };
    let backend = get_metal_backend().expect("metal backend");

    let (num_heads, head_dim) = (3usize, 5usize);
    let (cached_seq, new_seq) = (6usize, 1usize);
    let total_seq = cached_seq + new_seq;

    let cached_k = pattern(num_heads * cached_seq * head_dim, 0.23);
    let cached_v = pattern(num_heads * cached_seq * head_dim, 1.07);
    let new_k = pattern(num_heads * new_seq * head_dim, 2.31);
    let new_v = pattern(num_heads * new_seq * head_dim, 3.19);
    let q = pattern(num_heads * new_seq * head_dim, 4.05);

    let cached_k_id = upload(&backend, &cached_k);
    let cached_v_id = upload(&backend, &cached_v);
    let new_k_id = upload(&backend, &new_k);
    let new_v_id = upload(&backend, &new_v);
    let q_id = upload(&backend, &q);

    let k_id = backend
        .concat_kv_cache(
            Some(&cached_k_id),
            &new_k_id,
            1,
            num_heads,
            cached_seq,
            new_seq,
            head_dim,
        )
        .expect("concat_kv_cache K");
    let v_id = backend
        .concat_kv_cache(
            Some(&cached_v_id),
            &new_v_id,
            1,
            num_heads,
            cached_seq,
            new_seq,
            head_dim,
        )
        .expect("concat_kv_cache V");

    // Reference concatenation along the sequence axis of a heads-major buffer.
    let concat = |old: &[f32], new: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0_f32; num_heads * total_seq * head_dim];
        for h in 0..num_heads {
            for s in 0..cached_seq {
                for d in 0..head_dim {
                    out[h * total_seq * head_dim + s * head_dim + d] =
                        old[h * cached_seq * head_dim + s * head_dim + d];
                }
            }
            for s in 0..new_seq {
                for d in 0..head_dim {
                    out[h * total_seq * head_dim + (cached_seq + s) * head_dim + d] =
                        new[h * new_seq * head_dim + s * head_dim + d];
                }
            }
        }
        out
    };
    let k_ref = concat(&cached_k, &new_k);
    let v_ref = concat(&cached_v, &new_v);
    assert_eq!(download(&backend, &k_id), k_ref, "concat_kv_cache: K");
    assert_eq!(download(&backend, &v_id), v_ref, "concat_kv_cache: V");

    let attn = backend
        .attention_with_cache_gpu_to_gpu(
            &q_id, &k_id, &v_id, 1, new_seq, total_seq, num_heads, head_dim,
        )
        .expect("attention_with_cache_gpu_to_gpu (decode)");
    // A single query at the newest position attends to every cached key, so causal
    // masking is a no-op here; assert against the unmasked reference.
    let attn_ref = reference_attention(
        &q, &k_ref, &v_ref, num_heads, new_seq, total_seq, head_dim, true,
    );
    let err = max_abs_diff(&download(&backend, &attn), &attn_ref);
    assert!(
        err < 1e-5,
        "decode attention differs from the host reference by {err}"
    );

    backend
        .release_buffers(&[
            cached_k_id,
            cached_v_id,
            new_k_id,
            new_v_id,
            q_id,
            k_id,
            v_id,
            attn,
        ])
        .expect("release test buffers");
}

// ---------------------------------------------------------------------------
// Mission 2 - KV-cache parity
// ---------------------------------------------------------------------------

/// Greedy decoding is deterministic, so the KV-cached generator must emit exactly the
/// same tokens as the uncached one, and as the CPU.
///
/// Regression guard for `forward_internal` deriving the positional offset from
/// `metal_cache.shape[1]`. The Metal KV cache is heads-major
/// `[batch, n_head, kv_seq, head_dim]`, so axis 1 is `n_head`: every decode step after
/// the first re-used the same position embedding and generation degenerated into a
/// repeated token. The `n_head`/prompt-length pairs below are chosen so that the bug's
/// "offset == n_head" coincidence cannot hide it.
#[test]
fn metal_generate_greedy_with_cache_matches_uncached() {
    let Some(device) = metal_device_or_skip("metal_generate_greedy_with_cache_matches_uncached")
    else {
        return;
    };

    for (n_head, prompt) in [
        (2usize, vec![1u32, 2]),
        (4usize, vec![1u32, 2]),
        (2usize, vec![1u32, 2, 3, 4]),
    ] {
        let mut model = Gpt2LMHeadModel::new_with_device(small_config(2, n_head), Device::CPU)
            .expect("build model on CPU");
        let cpu_cached = model
            .generate_greedy_with_cache(prompt.clone(), 9)
            .expect("cpu cached generate");

        let calls_before = metal_attention_call_count();
        model.weights_to_gpu(&device).expect("upload weights to the GPU");
        let gpu_plain = model.generate_greedy(prompt.clone(), 9).expect("gpu uncached generate");
        let gpu_cached = model
            .generate_greedy_with_cache(prompt.clone(), 9)
            .expect("gpu cached generate");
        assert!(
            metal_attention_call_count() > calls_before,
            "the Metal attention fast path never ran for n_head={n_head}"
        );

        assert_eq!(
            gpu_plain, cpu_cached,
            "GPU uncached greedy diverged from the CPU reference (n_head={n_head}, \
             prompt={prompt:?})"
        );
        assert_eq!(
            gpu_cached, gpu_plain,
            "GPU KV-cached greedy diverged from GPU uncached greedy (n_head={n_head}, \
             prompt={prompt:?})"
        );
    }
}

// ---------------------------------------------------------------------------
// Mission 3 - the GenerativeModel trait on a GPU-resident model
// ---------------------------------------------------------------------------

/// `GenerativeModel`'s methods used to reject GPU-resident logits outright
/// ("Unsupported tensor type for logits") while the inherent generators on the very
/// same model worked, because each carried its own copy of the last-token decode.
#[test]
fn metal_generative_model_trait_works_after_weights_to_gpu() {
    let Some(device) =
        metal_device_or_skip("metal_generative_model_trait_works_after_weights_to_gpu")
    else {
        return;
    };

    let mut model = Gpt2LMHeadModel::new_with_device(small_config(2, 2), Device::CPU)
        .expect("build model on CPU");
    let prompt = vec![1u32, 2, 3];
    let cpu_reference =
        GenerativeModel::generate_greedy(&model, prompt.clone(), 8).expect("cpu trait greedy");

    let calls_before = metal_attention_call_count();
    model.weights_to_gpu(&device).expect("upload weights to the GPU");

    let gpu_trait_greedy =
        GenerativeModel::generate_greedy(&model, prompt.clone(), 8).expect("gpu trait greedy");
    assert!(
        metal_attention_call_count() > calls_before,
        "the Metal attention fast path never ran"
    );
    assert_eq!(
        gpu_trait_greedy, cpu_reference,
        "GenerativeModel::generate_greedy on the GPU diverged from the CPU reference"
    );
    assert_eq!(
        gpu_trait_greedy,
        model.generate_greedy(prompt.clone(), 8).expect("inherent greedy"),
        "the trait generator and the inherent generator disagree on the same GPU model"
    );

    // The sampling entry points only have to succeed and stay in range - they consume
    // randomness, so their token ids are not comparable across calls.
    for (label, tokens) in [
        (
            "generate_top_k",
            GenerativeModel::generate_top_k(&model, prompt.clone(), 8, 5, 1.0)
                .expect("gpu trait top-k"),
        ),
        (
            "generate_top_p",
            GenerativeModel::generate_top_p(&model, prompt.clone(), 8, 0.9, 1.0)
                .expect("gpu trait top-p"),
        ),
    ] {
        assert_eq!(
            &tokens[..prompt.len()],
            &prompt[..],
            "{label} rewrote the prompt"
        );
        assert!(tokens.len() <= 8, "{label} overran max_length");
        assert!(
            tokens.iter().all(|t| (*t as usize) < model.get_config().vocab_size),
            "{label} produced an out-of-vocabulary token"
        );
    }

    let beams = GenerativeModel::generate_beam_search(&model, prompt.clone(), 8, 3)
        .expect("gpu trait beam search");
    assert!(
        !beams.is_empty(),
        "beam search returned no sequences on the GPU"
    );

    // Contrastive search needs hidden states as well as logits; it went through the
    // same host-only reader.
    let (logits, hidden) = model
        .logits_and_hidden_states(&prompt)
        .expect("logits_and_hidden_states on the GPU");
    assert_eq!(logits.len(), model.get_config().vocab_size);
    assert_eq!(hidden.len(), prompt.len());
    assert!(
        hidden.iter().all(|row| row.len() == model.get_config().n_embd),
        "hidden state rows have the wrong width"
    );
}

// ---------------------------------------------------------------------------
// Mission 5 - the attention fast path must not leak GPU buffers
// ---------------------------------------------------------------------------

/// Each Metal attention forward used to park seven `MTLBuffer`s in the process-global
/// buffer cache forever (three from `split_qkv_gpu`, three from `reshape_to_heads_gpu`,
/// one attention output), i.e. seven per layer per token of a generation loop.
#[test]
fn metal_attention_forward_releases_its_scratch_buffers() {
    let Some(device) = metal_device_or_skip("metal_attention_forward_releases_its_scratch_buffers")
    else {
        return;
    };
    let backend = get_metal_backend().expect("metal backend");

    let n_layer = 2usize;
    let mut model = Gpt2LMHeadModel::new_with_device(small_config(n_layer, 2), Device::CPU)
        .expect("build model on CPU");
    model.weights_to_gpu(&device).expect("upload weights to the GPU");

    // One warm-up pass: the first forward legitimately grows the cache (lazily
    // uploaded weights, cached transposes). Steady state is what must not drift.
    let calls_before = metal_attention_call_count();
    drop(model.forward(tokenized(vec![1, 2, 3])).expect("warm-up forward"));
    let warmup_calls = metal_attention_call_count() - calls_before;
    assert_eq!(
        warmup_calls, n_layer,
        "expected one Metal attention call per layer, saw {warmup_calls}"
    );

    let baseline = backend.buffer_cache_stats().expect("buffer cache stats").entries;
    for round in 0..4 {
        drop(model.forward(tokenized(vec![1, 2, 3])).expect("forward"));
        let entries = backend.buffer_cache_stats().expect("buffer cache stats").entries;
        assert_eq!(
            entries, baseline,
            "buffer cache grew from {baseline} to {entries} entries after forward {round}; \
             the attention fast path is leaking GPU scratch buffers"
        );
    }

    // A KV-cached generation holds K/V per layer while it runs, but must give every
    // one of them back when the cache is dropped at the end of the call.
    let before_generation = backend.buffer_cache_stats().expect("buffer cache stats").entries;
    let generated = model.generate_greedy_with_cache(vec![1, 2, 3], 8).expect("cached generation");
    assert_eq!(
        generated.len(),
        8,
        "generation stopped early; the leak check is not meaningful"
    );
    let after_generation = backend.buffer_cache_stats().expect("buffer cache stats").entries;
    assert_eq!(
        after_generation,
        before_generation,
        "KV-cached generation leaked {} buffer cache entries",
        after_generation as i64 - before_generation as i64
    );
}

// ── Multi-token continuation against a GPU-resident cache ────────────────────

/// A chunk of new tokens fed to a warm GPU KV cache must give the same answer as one
/// uncached forward over the whole sequence, must actually run on the GPU, and must
/// not lose the cached history.
///
/// `MetalBackend::attention_with_cache_gpu_to_gpu` used to send
/// `1 < q_seq_len < kv_seq_len` to the *unmasked* generation kernel, so every query
/// row in the chunk also attended to the later rows of its own chunk (measured:
/// exactly the non-causal reference, ~32% of signal magnitude away from the causal
/// one), and this fast path declined the shape rather than return wrong numbers. The
/// composition now selects an offset-masked kernel, so the chunk belongs on the GPU
/// again - which is why every stage below is bracketed by the Metal attention
/// counter. Without those assertions the test would still pass if admission control
/// silently sent the whole thing back to the host path.
#[test]
fn metal_multi_token_cache_continuation_matches_uncached_forward() {
    use crate::gpt2::model::KVCache;

    let Some(device) =
        metal_device_or_skip("metal_multi_token_cache_continuation_matches_uncached_forward")
    else {
        return;
    };

    let config = small_config(2, 2);
    let mut model =
        Gpt2LMHeadModel::new_with_device(config.clone(), Device::CPU).expect("build model on CPU");
    model.weights_to_gpu(&device).expect("upload weights to the GPU");

    let prompt = vec![1u32, 2, 3];
    let continuation = vec![4u32, 5];
    let whole: Vec<u32> = prompt.iter().chain(continuation.iter()).copied().collect();

    let calls_before = metal_attention_call_count();
    let mut cache = Some(KVCache::new(config.n_layer));
    model
        .forward_with_cache(tokenized(prompt), &mut cache)
        .expect("prefill through the GPU cache");
    assert_eq!(
        metal_attention_call_count() - calls_before,
        config.n_layer,
        "the prefill must have run on the Metal fast path, once per layer"
    );

    // THE mission-critical assertion: the chunk must be served by the GPU, by the
    // offset-masked kernel, not by the host fallback. `nextest` gives each test its
    // own process, so this process-global counter belongs to this test alone and the
    // per-layer count can be asserted exactly.
    let calls_before_chunk = metal_attention_call_count();
    let chunked = model
        .forward_with_cache(tokenized(continuation), &mut cache)
        .expect("multi-token continuation against a GPU-resident cache");
    assert_eq!(
        metal_attention_call_count() - calls_before_chunk,
        config.n_layer,
        "the multi-token continuation must run on the Metal fast path (one attention \
         dispatch per layer); a smaller count means it fell back to the host"
    );
    let reference = model.forward(tokenized(whole)).expect("uncached forward");

    // Compare EVERY row of the chunk, not just the last one: the last query row of a
    // chunk sits at the end of the key sequence, so causal and non-causal attention
    // agree there and it cannot see this bug at all.
    let chunked_rows = logits_rows(&chunked.logits);
    let reference_rows = logits_rows(&reference.logits);
    assert_eq!(
        chunked_rows.len(),
        2,
        "the continuation must return one row per new token"
    );
    let magnitude = reference_rows.iter().flatten().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(
        magnitude > 0.0,
        "reference logits are all zero; nothing was computed"
    );
    for (offset, chunk_row) in chunked_rows.iter().enumerate() {
        let reference_row = &reference_rows[reference_rows.len() - chunked_rows.len() + offset];
        let relative = max_abs_diff(chunk_row, reference_row) / magnitude;
        assert!(
            relative < 1e-3,
            "cached chunk row {offset} diverged from the uncached forward by {relative} \
             of signal magnitude (max|reference| = {magnitude})"
        );
    }

    // Weight-independent causality check. Row 0 of the chunk is at absolute position
    // 3; the token at position 4 is strictly in its future, so replacing it must leave
    // that row bit-identical. Attending to it - as the unmasked generation kernel this
    // composition used to select would - changes the row. The all-rows comparison above only catches this when the
    // randomly initialised weights make the difference big enough to clear 1e-3, which
    // measurement showed happens most but not all of the time; this assertion holds
    // for every weight set.
    let causal_row = |last_token: u32| -> Vec<f32> {
        let mut probe_cache = Some(KVCache::new(config.n_layer));
        model
            .forward_with_cache(tokenized(vec![1, 2, 3]), &mut probe_cache)
            .expect("prefill through the GPU cache");
        let out = model
            .forward_with_cache(tokenized(vec![4, last_token]), &mut probe_cache)
            .expect("continuation through the GPU cache");
        logits_rows(&out.logits).swap_remove(0)
    };
    assert_eq!(
        causal_row(5),
        causal_row(9),
        "row 0 of the chunk changed when a strictly later token changed: the chunk is \
         attending to its own future"
    );

    // The cache must still be usable, still GPU-resident, and still describe all five
    // positions.
    let calls_before_decode = metal_attention_call_count();
    let after = model
        .forward_with_cache(tokenized(vec![6u32]), &mut cache)
        .expect("single-token decode after the chunked continuation");
    assert_eq!(logits_rows(&after.logits)[0].len(), config.vocab_size);
    assert_eq!(
        metal_attention_call_count() - calls_before_decode,
        config.n_layer,
        "the decode after the chunk must still run on the Metal fast path; a fallback \
         here would mean the chunk left the cache in a host layout"
    );
}

/// Rows of a `[batch, seq, vocab]` logits tensor (first batch element), as host floats.
fn logits_rows(logits: &Tensor) -> Vec<Vec<f32>> {
    let host = logits.to_device_enum(&Device::CPU).expect("download logits");
    let shape = host.shape().to_vec();
    assert_eq!(
        shape.len(),
        3,
        "expected [batch, seq, vocab] logits, got {shape:?}"
    );
    let (seq_len, vocab_size) = (shape[1], shape[2]);
    let values = host.to_vec_f32().expect("logits must be f32");
    (0..seq_len)
        .map(|row| values[row * vocab_size..(row + 1) * vocab_size].to_vec())
        .collect()
}
