use crate::gpt2::{Gpt2Config, Gpt2LMHeadModel, Gpt2Model};
use trustformers_core::{
    tensor::Tensor,
    traits::{Model, TokenizedInput},
};

#[test]
fn test_gpt2_model_creation() {
    let config = Gpt2Config::default();
    let model = Gpt2Model::new(config.clone()).expect("operation failed");
    assert_eq!(model.get_config().n_layer, 12);
    assert_eq!(model.get_config().n_head, 12);
}

#[test]
fn test_gpt2_lm_head_model_creation() {
    let config = Gpt2Config::default();
    let model = Gpt2LMHeadModel::new(config.clone()).expect("operation failed");
    assert_eq!(model.get_config().n_layer, 12);
}

#[test]
fn test_gpt2_forward_pass() {
    let config = Gpt2Config {
        vocab_size: 100,
        n_positions: 64,
        n_embd: 32,
        n_layer: 2,
        n_head: 4,
        ..Default::default()
    };

    let model = Gpt2Model::new(config).expect("operation failed");
    let input = TokenizedInput {
        input_ids: vec![1, 2, 3, 4, 5],
        attention_mask: vec![1u8; 5],
        token_type_ids: None,
        offset_mapping: None,
        special_tokens_mask: None,
        overflowing_tokens: None,
    };

    let output = model.forward(input).expect("operation failed");
    match &output.last_hidden_state {
        Tensor::F32(arr) => {
            assert_eq!(arr.shape(), &[1, 5, 32]);
        },
        _ => panic!("Expected F32 tensor"),
    }
}

#[test]
fn test_gpt2_lm_forward_pass() {
    let config = Gpt2Config {
        vocab_size: 100,
        n_positions: 64,
        n_embd: 32,
        n_layer: 2,
        n_head: 4,
        ..Default::default()
    };

    let model = Gpt2LMHeadModel::new(config).expect("operation failed");
    let input = TokenizedInput {
        input_ids: vec![1, 2, 3, 4, 5],
        attention_mask: vec![1u8; 5],
        token_type_ids: None,
        offset_mapping: None,
        special_tokens_mask: None,
        overflowing_tokens: None,
    };

    let output = model.forward(input).expect("operation failed");
    match &output.logits {
        Tensor::F32(arr) => {
            assert_eq!(arr.shape(), &[1, 5, 100]);
        },
        _ => panic!("Expected F32 tensor"),
    }
}

#[test]
fn test_gpt2_generate_greedy() {
    let config = Gpt2Config {
        vocab_size: 100,
        n_positions: 64,
        n_embd: 32,
        n_layer: 2,
        n_head: 4,
        ..Default::default()
    };

    let model = Gpt2LMHeadModel::new(config).expect("operation failed");
    let input_ids = vec![1, 2, 3];
    let generated = model.generate_greedy(input_ids.clone(), 10).expect("operation failed");

    assert!(generated.len() >= input_ids.len());
    assert!(generated.len() <= 10);
    assert_eq!(&generated[..3], &input_ids[..]);
}

#[test]
fn test_gpt2_beam_search() {
    // Use even smaller config to reduce memory pressure
    let config = Gpt2Config {
        vocab_size: 30,  // Reduce from 50
        n_positions: 32, // Reduce from 64
        n_embd: 16,      // Reduce from 32
        n_layer: 1,
        n_head: 2, // Reduce from 4
        ..Default::default()
    };

    let model = Gpt2LMHeadModel::new(config).expect("operation failed");
    let input_ids = vec![1, 2];
    let max_length = 6; // Reduce from 10
    let num_beams = 2; // Reduce from 3

    let generated = model
        .generate_beam_search(input_ids.clone(), max_length, num_beams)
        .expect("operation failed");

    assert!(generated.len() >= input_ids.len());
    assert!(generated.len() <= max_length);
    assert_eq!(&generated[..2], &input_ids[..]);

    // Explicit cleanup
    drop(generated);
    drop(model);
    std::hint::black_box(());
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn test_gpt2_metal_sampling() {
    use crate::gpt2::generation::GenerativeModel;
    use crate::gpt2::model::metal_attention_call_count;
    use std::time::Instant;
    use trustformers_core::Device;

    // Very small model config to prevent Metal SIGTRAP
    let config = Gpt2Config {
        vocab_size: 50,  // Reduce from 100
        n_positions: 32, // Reduce from 64
        n_embd: 32,      // Reduce from 64
        n_layer: 1,      // Reduce from 2
        n_head: 2,       // Reduce from 4
        ..Default::default()
    };

    // Test Metal device availability
    let device = Device::metal_if_available(0);

    // Skip test if Metal is not available
    if !matches!(device, Device::Metal(_)) {
        eprintln!("test_gpt2_metal_sampling: no Metal device on this machine, skipping");
        return;
    }

    // Build on the CPU and then MOVE THE WEIGHTS. `new_with_device(_, Metal)` alone
    // leaves every weight in host memory and computes the whole forward pass in
    // ndarray, so a test written that way is a CPU test wearing a Metal name - which
    // is exactly what this one used to be.
    let mut model =
        Gpt2LMHeadModel::new_with_device(config.clone(), Device::CPU).expect("operation failed");
    let calls_before = metal_attention_call_count();
    model.weights_to_gpu(&device).expect("weights_to_gpu");

    // Test input - use smaller values
    let input_ids = vec![1, 2];
    let max_length = 8; // Reduce from 20
    let k = 5; // Reduce from 10
    let temperature = 1.0;

    // Measure generation time
    let start = Instant::now();
    let generated = model
        .generate_top_k(input_ids.clone(), max_length, k, temperature)
        .expect("operation failed");
    let elapsed = start.elapsed();

    let gpu_attention_calls = metal_attention_call_count() - calls_before;
    println!(
        "Generated {} tokens in {:?} ({} GPU attention dispatches)",
        generated.len() - input_ids.len(),
        elapsed,
        gpu_attention_calls
    );

    // The point of the test: the GPU really ran.
    assert!(
        gpu_attention_calls > 0,
        "no Metal attention dispatch happened - this test would be measuring CPU math"
    );

    // Verify output
    assert!(generated.len() >= input_ids.len());
    assert!(generated.len() <= max_length);
    assert_eq!(&generated[..2], &input_ids[..]);
    assert!(
        generated.iter().all(|t| (*t as usize) < config.vocab_size),
        "sampling produced an out-of-vocabulary token"
    );

    // Explicit cleanup
    drop(generated);
    drop(model);
    std::hint::black_box(());
}

/// CPU vs GPU on identical weights: same tokens, and both sides really ran where they
/// claim to.
///
/// This used to be `test_gpt2_metal_vs_cpu_performance` and compared a CPU model
/// against a second model that had merely been *constructed* with `Device::Metal(0)`,
/// so both halves executed the same ndarray kernels and the reported "speedup" was
/// CPU-vs-CPU timing noise. Wall-clock numbers are printed for information but are
/// not asserted on: a 1-layer, 32-wide model is dominated by dispatch overhead, and a
/// timing assertion there would be a coin flip, not a measurement.
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn test_gpt2_metal_matches_cpu_on_identical_weights() {
    use crate::gpt2::model::metal_attention_call_count;
    use std::time::Instant;
    use trustformers_core::Device;

    // Very small model config to prevent Metal SIGTRAP
    let config = Gpt2Config {
        vocab_size: 50,  // Reduce from 100
        n_positions: 32, // Reduce from 128
        n_embd: 32,      // Reduce from 128
        n_layer: 1,      // Reduce from 4
        n_head: 2,       // Reduce from 8
        ..Default::default()
    };

    let input_ids = vec![1, 2, 3];
    let max_length = 8; // Reduce from 25

    let device = Device::metal_if_available(0);
    if !matches!(device, Device::Metal(_)) {
        eprintln!(
            "test_gpt2_metal_matches_cpu_on_identical_weights: no Metal device on this \
             machine, skipping"
        );
        return;
    }

    // ONE model, two devices: the GPU run must reproduce the CPU run bit-for-bit in
    // token space (greedy decoding is deterministic), which is only a meaningful
    // claim because the weights are literally the same tensors moved across.
    let mut model =
        Gpt2LMHeadModel::new_with_device(config.clone(), Device::CPU).expect("operation failed");

    let cpu_start = Instant::now();
    let cpu_generated =
        model.generate_greedy(input_ids.clone(), max_length).expect("operation failed");
    let cpu_elapsed = cpu_start.elapsed();
    let cpu_tokens = cpu_generated.len() - input_ids.len();
    println!("CPU: generated {} tokens in {:?}", cpu_tokens, cpu_elapsed);

    let calls_before = metal_attention_call_count();
    model.weights_to_gpu(&device).expect("weights_to_gpu");

    let metal_start = Instant::now();
    let metal_generated =
        model.generate_greedy(input_ids.clone(), max_length).expect("operation failed");
    let metal_elapsed = metal_start.elapsed();
    let metal_tokens = metal_generated.len() - input_ids.len();
    let gpu_attention_calls = metal_attention_call_count() - calls_before;
    println!(
        "Metal: generated {} tokens in {:?} ({} GPU attention dispatches)",
        metal_tokens, metal_elapsed, gpu_attention_calls
    );

    assert!(
        gpu_attention_calls > 0,
        "no Metal attention dispatch happened - the 'Metal' half ran on the CPU"
    );
    assert!(metal_tokens > 0);
    assert_eq!(
        metal_tokens, cpu_tokens,
        "the two devices generated different lengths"
    );
    assert_eq!(
        metal_generated, cpu_generated,
        "greedy decoding is deterministic, so identical weights must give identical \
         tokens on both devices"
    );

    // Explicit cleanup
    drop(metal_generated);
    drop(cpu_generated);
    drop(model);
    std::hint::black_box(());
}

// ── Contrastive search ───────────────────────────────────────────────────────

mod contrastive_search {
    use super::*;
    use crate::generation_utils::{GenerationConfig, GenerationUtils};
    use crate::gpt2::generation::GenerativeModel;

    fn tiny_model() -> (Gpt2LMHeadModel, Gpt2Config) {
        let config = Gpt2Config {
            vocab_size: 32,
            n_positions: 32,
            n_embd: 16,
            n_layer: 2,
            n_head: 4,
            ..Default::default()
        };
        let model =
            Gpt2LMHeadModel::new(config.clone()).expect("tiny GPT-2 model must be constructible");
        (model, config)
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    #[test]
    fn contrastive_search_generates_instead_of_erroring() {
        // Regression: the `ContrastiveSearch` arm used to return
        // "Contrastive search not yet implemented for GPT-2" for every call,
        // while the preset constructor advertised the mode.
        let (model, config) = tiny_model();
        let prompt = vec![1u32, 2, 3];
        let mut generation_config = GenerationConfig::contrastive_search(4, 0.6);
        // `max_length` counts the prompt; `GenerationUtils::should_stop` compares
        // `max_new_tokens` against the whole sequence, so it cannot express
        // "three more tokens" for a non-empty prompt.
        generation_config.max_length = prompt.len() + 3;

        let sequences = model
            .generate_with_config(prompt.clone(), generation_config)
            .expect("contrastive search must generate");
        assert_eq!(sequences.len(), 1);
        let sequence = &sequences[0];
        assert_eq!(sequence.len(), prompt.len() + 3);
        assert_eq!(&sequence[..prompt.len()], prompt.as_slice());
        assert!(sequence.iter().all(|&token| (token as usize) < config.vocab_size));
    }

    #[test]
    fn contrastive_search_selects_the_reference_scored_token() {
        // Independent re-implementation of the degeneration penalty:
        //   score(v) = (1 - alpha) * p(v) - alpha * max_j cos(h_v, h_j)
        let (model, _) = tiny_model();
        let prompt = vec![5u32, 7, 11];
        let top_k = 5usize;
        let alpha = 0.6f32;

        let (logits, context_states) = model
            .logits_and_hidden_states(&prompt)
            .expect("hidden states must be available");
        let probabilities = GenerationUtils::softmax(&logits);
        let mut ranked: Vec<(usize, f32)> = probabilities.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);

        let mut expected: Option<(u32, f32)> = None;
        for (token_id, probability) in ranked {
            let mut candidate = prompt.clone();
            candidate.push(token_id as u32);
            let (_, states) = model
                .logits_and_hidden_states(&candidate)
                .expect("candidate hidden states must be available");
            let candidate_state = states.last().expect("candidate must have a representation");
            let max_similarity = context_states
                .iter()
                .map(|state| cosine(candidate_state, state))
                .fold(f32::NEG_INFINITY, f32::max);
            let score = (1.0 - alpha) * probability - alpha * max_similarity;
            if expected.is_none_or(|(_, best)| score > best) {
                expected = Some((token_id as u32, score));
            }
        }
        let (expected_token, _) = expected.expect("reference scoring must pick a token");

        let mut generation_config = GenerationConfig::contrastive_search(top_k, alpha);
        generation_config.max_length = prompt.len() + 1;
        let sequences = model
            .generate_with_config(prompt.clone(), generation_config)
            .expect("contrastive search must generate");
        assert_eq!(
            sequences[0][prompt.len()],
            expected_token,
            "contrastive search must pick the token the penalised score ranks first"
        );
    }

    #[test]
    fn contrastive_search_penalty_can_override_the_most_likely_token() {
        // With alpha = 0 the penalty vanishes and the highest-probability
        // candidate wins; the scoring must therefore be sensitive to alpha.
        let (model, _) = tiny_model();
        let prompt = vec![3u32, 3, 3, 3];

        let (logits, _) = model
            .logits_and_hidden_states(&prompt)
            .expect("hidden states must be available");
        let probabilities = GenerationUtils::softmax(&logits);
        let most_likely = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index as u32)
            .expect("softmax must produce a maximum");

        let mut greedy_like = GenerationConfig::contrastive_search(8, 0.0);
        greedy_like.max_length = prompt.len() + 1;
        let sequences = model
            .generate_with_config(prompt.clone(), greedy_like)
            .expect("contrastive search must generate");
        assert_eq!(
            sequences[0][prompt.len()],
            most_likely,
            "alpha = 0 must reduce contrastive search to greedy decoding"
        );
    }

    #[test]
    fn contrastive_search_rejects_a_zero_top_k() {
        let (model, _) = tiny_model();
        let mut generation_config = GenerationConfig::contrastive_search(0, 0.5);
        generation_config.max_length = 3;
        let err = model
            .generate_with_config(vec![1, 2], generation_config)
            .expect_err("top_k = 0 must be rejected");
        assert!(err.to_string().contains("top_k"), "unexpected error: {err}");
    }

    #[test]
    fn contrastive_search_rejects_an_out_of_range_alpha() {
        let (model, _) = tiny_model();
        let mut generation_config = GenerationConfig::contrastive_search(4, 1.5);
        generation_config.max_length = 3;
        let err = model
            .generate_with_config(vec![1, 2], generation_config)
            .expect_err("alpha outside [0, 1] must be rejected");
        assert!(err.to_string().contains("alpha"), "unexpected error: {err}");
    }
}

// ── Weight loading from a byte stream ────────────────────────────────────────

mod weight_loading {
    use super::*;
    use crate::weight_loading::test_support::{build_safetensors, F32Tensor};

    fn tiny_config() -> Gpt2Config {
        Gpt2Config {
            vocab_size: 16,
            n_positions: 16,
            n_embd: 8,
            n_layer: 2,
            n_head: 2,
            ..Default::default()
        }
    }

    /// A GPT-2 checkpoint in HuggingFace's own layout.
    ///
    /// HF stores the block projections as `Conv1D` weights of shape
    /// `[in_features, out_features]`, the transpose of this crate's `Linear`.
    fn gpt2_fixture(config: &Gpt2Config, prefix: &str, with_lm_head: bool) -> Vec<F32Tensor> {
        let embd = config.n_embd;
        let mut seed = 0.0f32;
        let mut next_seed = || {
            seed += 1.0;
            seed
        };

        let mut tensors = vec![
            F32Tensor::ramp(
                &format!("{prefix}wte.weight"),
                &[config.vocab_size, embd],
                next_seed(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}wpe.weight"),
                &[config.n_positions, embd],
                next_seed(),
            ),
        ];

        for layer in 0..config.n_layer {
            let p = format!("{prefix}h.{layer}.");
            for norm in ["ln_1", "ln_2"] {
                tensors.push(F32Tensor::ramp(
                    &format!("{p}{norm}.weight"),
                    &[embd],
                    next_seed(),
                ));
                tensors.push(F32Tensor::ramp(
                    &format!("{p}{norm}.bias"),
                    &[embd],
                    next_seed(),
                ));
            }
            tensors.push(F32Tensor::ramp(
                &format!("{p}attn.c_attn.weight"),
                &[embd, 3 * embd],
                next_seed(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{p}attn.c_attn.bias"),
                &[3 * embd],
                next_seed(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{p}attn.c_proj.weight"),
                &[embd, embd],
                next_seed(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{p}attn.c_proj.bias"),
                &[embd],
                next_seed(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{p}mlp.c_fc.weight"),
                &[embd, 4 * embd],
                next_seed(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{p}mlp.c_fc.bias"),
                &[4 * embd],
                next_seed(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{p}mlp.c_proj.weight"),
                &[4 * embd, embd],
                next_seed(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{p}mlp.c_proj.bias"),
                &[embd],
                next_seed(),
            ));
        }

        tensors.push(F32Tensor::ramp(
            &format!("{prefix}ln_f.weight"),
            &[embd],
            next_seed(),
        ));
        tensors.push(F32Tensor::ramp(
            &format!("{prefix}ln_f.bias"),
            &[embd],
            next_seed(),
        ));

        if with_lm_head {
            tensors.push(F32Tensor::ramp(
                "lm_head.weight",
                &[config.vocab_size, embd],
                next_seed(),
            ));
        }
        tensors
    }

    fn hidden(model: &Gpt2Model) -> Vec<f32> {
        let input = TokenizedInput {
            input_ids: vec![1, 2, 3],
            attention_mask: vec![1u8; 3],
            token_type_ids: None,
            offset_mapping: None,
            special_tokens_mask: None,
            overflowing_tokens: None,
        };
        match model.forward(input).expect("forward must succeed").last_hidden_state {
            Tensor::F32(arr) => arr.iter().copied().collect(),
            other => panic!("expected an F32 hidden state, got {other:?}"),
        }
    }

    #[test]
    fn gpt2_load_pretrained_reads_a_real_checkpoint() {
        // Regression: `Model::load_pretrained` used to return
        // "Use load_weights_from_reader instead" for every call.
        let config = tiny_config();
        let bytes = build_safetensors(&gpt2_fixture(&config, "", false));

        let mut first = Gpt2Model::new(config.clone()).expect("model must build");
        let mut second = Gpt2Model::new(config).expect("model must build");
        assert_ne!(hidden(&first), hidden(&second), "random inits must differ");

        first.load_pretrained(&mut bytes.as_slice()).expect("checkpoint must load");
        second.load_pretrained(&mut bytes.as_slice()).expect("checkpoint must load");
        assert_eq!(
            hidden(&first),
            hidden(&second),
            "after loading, both models must be the checkpoint's model"
        );
    }

    #[test]
    fn gpt2_load_pretrained_accepts_the_transformer_prefix() {
        let config = tiny_config();
        let bytes = build_safetensors(&gpt2_fixture(&config, "transformer.", true));
        let mut model = Gpt2LMHeadModel::new(config).expect("model must build");
        model
            .load_pretrained(&mut bytes.as_slice())
            .expect("a task checkpoint must load");
    }

    #[test]
    fn gpt2_lm_head_is_tied_to_the_embeddings_when_absent() {
        let config = tiny_config();
        let tensors = gpt2_fixture(&config, "transformer.", false);
        let embeddings = tensors
            .iter()
            .find(|t| t.name == "transformer.wte.weight")
            .expect("fixture must hold the embedding table")
            .clone();
        let bytes = build_safetensors(&tensors);

        let mut model = Gpt2LMHeadModel::new(config).expect("model must build");
        model
            .load_pretrained(&mut bytes.as_slice())
            .expect("a tied-embedding checkpoint must load");
        // Round-tripping through the model is enough: the head must now produce
        // logits derived from the checkpoint's embedding table rather than from
        // its random initialisation.
        let input = TokenizedInput {
            input_ids: vec![0, 1],
            attention_mask: vec![1u8; 2],
            token_type_ids: None,
            offset_mapping: None,
            special_tokens_mask: None,
            overflowing_tokens: None,
        };
        let logits = model.forward(input).expect("forward must succeed").logits;
        assert_eq!(logits.shape(), vec![1, 2, embeddings.shape[0]]);
    }

    #[test]
    fn gpt2_load_pretrained_rejects_an_incomplete_checkpoint() {
        let config = tiny_config();
        let mut tensors = gpt2_fixture(&config, "", false);
        tensors.retain(|t| t.name != "h.1.mlp.c_fc.weight");
        let bytes = build_safetensors(&tensors);

        let mut model = Gpt2Model::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("a missing tensor must fail the load");
        assert!(
            err.to_string().contains("h.1.mlp.c_fc.weight"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn gpt2_load_pretrained_rejects_bytes_that_are_not_a_checkpoint() {
        let config = tiny_config();
        let mut model = Gpt2Model::new(config).expect("model must build");
        let garbage = vec![0x3C_u8; 4096];
        let err = model
            .load_pretrained(&mut garbage.as_slice())
            .expect_err("garbage must not be accepted as weights");
        assert!(
            err.to_string().contains("unrecognised checkpoint container"),
            "unexpected error: {err}"
        );
    }
}

// ── KV-cache continuation with more than one new token ───────────────────────

/// Feeding a *chunk* of new tokens to a warm KV cache must give the same answer as
/// one uncached forward over the whole sequence.
///
/// This used to abort the process: `Gpt2Attention::forward_with_cache` added the
/// caller's `create_causal_mask(seq_len)` (shaped `[1, 1, new, new]`) straight into a
/// `[batch, heads, new, cached + new]` score matrix, and `ndarray` panicked with
/// "could not broadcast array from shape [1, 1, 2, 2] to [1, 2, 2, 5]". No test
/// covered a multi-token continuation, because the greedy generators only ever feed
/// one token at a time after the prompt.
#[test]
fn gpt2_multi_token_cache_continuation_matches_uncached_forward() {
    use crate::gpt2::model::KVCache;

    let config = Gpt2Config {
        vocab_size: 50,
        n_positions: 32,
        n_embd: 32,
        n_layer: 2,
        n_head: 2,
        ..Default::default()
    };
    let model = Gpt2LMHeadModel::new(config.clone()).expect("model creation should succeed");

    let prompt = vec![1u32, 2, 3];
    let continuation = vec![4u32, 5];
    let whole: Vec<u32> = prompt.iter().chain(continuation.iter()).copied().collect();

    let mut cache = Some(KVCache::new(config.n_layer));
    model
        .forward_with_cache(tokenized_input(prompt.clone()), &mut cache)
        .expect("prefill through the cache");
    let chunked = model
        .forward_with_cache(tokenized_input(continuation), &mut cache)
        .expect("multi-token continuation through the cache");

    let reference = model.forward(tokenized_input(whole)).expect("uncached forward");

    // Compare EVERY row of the chunk: the last query row of a chunk sits at the end of
    // the key sequence, so causal and non-causal attention agree there and it cannot
    // see a within-chunk masking bug at all.
    let chunked_rows = logits_rows(&chunked.logits);
    let reference_rows = logits_rows(&reference.logits);
    assert_eq!(
        chunked_rows.len(),
        2,
        "the continuation must return one row per new token"
    );
    assert_eq!(chunked_rows[0].len(), config.vocab_size);
    let magnitude = reference_rows.iter().flatten().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(
        magnitude > 0.0,
        "reference logits are all zero; nothing was computed"
    );
    for (offset, chunk_row) in chunked_rows.iter().enumerate() {
        let reference_row = &reference_rows[reference_rows.len() - chunked_rows.len() + offset];
        let deviation = chunk_row
            .iter()
            .zip(reference_row.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            deviation / magnitude < 1e-4,
            "cached chunk row {offset} diverged from the uncached forward by {deviation} \
             (max|reference| = {magnitude})"
        );
    }

    // Weight-independent causality check: row 0 of the chunk is at absolute position 3,
    // so the token at position 4 is strictly in its future and replacing it must leave
    // that row bit-identical.
    let causal_row = |last_token: u32| -> Vec<f32> {
        let mut probe_cache = Some(KVCache::new(config.n_layer));
        model
            .forward_with_cache(tokenized_input(vec![1, 2, 3]), &mut probe_cache)
            .expect("prefill through the cache");
        let out = model
            .forward_with_cache(tokenized_input(vec![4, last_token]), &mut probe_cache)
            .expect("continuation through the cache");
        logits_rows(&out.logits).swap_remove(0)
    };
    assert_eq!(
        causal_row(5),
        causal_row(9),
        "row 0 of the chunk changed when a strictly later token changed: the chunk is \
         attending to its own future"
    );
}

fn tokenized_input(ids: Vec<u32>) -> TokenizedInput {
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

/// Rows of a `[batch, seq, vocab]` logits tensor (first batch element), as host floats.
fn logits_rows(logits: &Tensor) -> Vec<Vec<f32>> {
    let host = logits
        .to_device_enum(&trustformers_core::device::Device::CPU)
        .expect("logits must be readable on the host");
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
