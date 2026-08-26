#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::mistral::config::MistralConfig;
    use crate::mistral::model::{
        MistralAttention, MistralForCausalLM, MistralModel, MixtralExpert, MixtralSparseMoE,
    };
    use trustformers_core::tensor::Tensor;
    use trustformers_core::traits::{Config, Layer};

    #[test]
    fn test_mistral_config_validation() {
        // Use minimal config to reduce memory usage
        let config = MistralConfig {
            num_hidden_layers: 2,
            vocab_size: 1000,
            hidden_size: 64,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            intermediate_size: 256,
            ..MistralConfig::default()
        };
        assert!(config.validate().is_ok());

        // Test head dimension calculation
        assert_eq!(config.head_dim(), 8); // 64 / 8

        // Test grouped-query attention
        assert_eq!(config.num_query_groups(), 4); // 8 / 2

        // Explicit cleanup
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_mistral_sliding_window() {
        // Use minimal config with sliding window to reduce memory usage
        let config = MistralConfig {
            num_hidden_layers: 2,
            vocab_size: 1000,
            hidden_size: 64,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            intermediate_size: 256,
            sliding_window: Some(512), // Reduced from 4096
            ..MistralConfig::default()
        };
        assert!(config.uses_sliding_window());
        assert_eq!(config.sliding_window_size(), 512);

        // Explicit cleanup
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_mixtral_config() {
        // Use minimal Mixtral config to reduce memory usage
        let config = MistralConfig {
            num_hidden_layers: 2,
            vocab_size: 1000,
            hidden_size: 64,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            intermediate_size: 256,
            model_type: "mixtral".to_string(),
            sliding_window: None, // Mixtral doesn't use sliding window
            ..MistralConfig::default()
        };
        assert!(config.validate().is_ok());
        assert!(!config.uses_sliding_window()); // Mixtral doesn't use sliding window
        assert_eq!(config.model_type, "mixtral");

        // Explicit cleanup
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_mistral_architecture() {
        let config = MistralConfig::default();
        assert_eq!(config.architecture(), "Mistral");
    }

    #[test]
    fn test_invalid_mistral_config() {
        let config = MistralConfig {
            num_attention_heads: 31, // Not divisible by num_key_value_heads (8)
            ..MistralConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_mistral_model_creation() {
        let config = MistralConfig {
            num_hidden_layers: 2, // Smaller for testing
            vocab_size: 1000,
            hidden_size: 64,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            intermediate_size: 256,
            ..MistralConfig::default()
        };

        let model = MistralModel::new(config);
        assert!(model.is_ok());

        // Explicit cleanup
        if let Ok(model) = model {
            drop(model);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_mistral_for_causal_lm_creation() {
        let config = MistralConfig {
            num_hidden_layers: 1, // Minimal for testing
            vocab_size: 100,
            hidden_size: 32,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            ..MistralConfig::default()
        };

        let model = MistralForCausalLM::new(config);
        assert!(model.is_ok());

        // Explicit cleanup
        if let Ok(model) = model {
            drop(model);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_mixtral_sparse_moe_creation() {
        let config = MistralConfig {
            hidden_size: 64,
            intermediate_size: 256,
            ..MistralConfig::default()
        };

        // Create experts
        let mut experts = Vec::new();
        for i in 0..4 {
            experts.push(MixtralExpert::new(i, &config).expect("operation failed"));
        }

        // Create MoE config
        let moe_config = crate::moe::MoEConfig {
            num_experts: 4,
            num_experts_per_token: 2,
            hidden_size: config.hidden_size,
            expert_capacity: None,
            ..Default::default()
        };

        let moe = MixtralSparseMoE::new(experts, moe_config);
        assert!(moe.is_ok());

        // Explicit cleanup
        if let Ok(moe) = moe {
            drop(moe);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_mixtral_expert_creation() {
        let config = MistralConfig {
            hidden_size: 64,
            intermediate_size: 256,
            ..MistralConfig::default()
        };

        let expert = MixtralExpert::new(0, &config);
        assert!(expert.is_ok());

        // Explicit cleanup
        if let Ok(expert) = expert {
            drop(expert);
        }
        std::hint::black_box(());
    }

    // ── Real attention regression tests ───────────────────────────────────────
    //
    // These exercise real RoPE + grouped-query scaled dot-product attention
    // with a real sliding-window mask, replacing a no-op RoPE and a
    // direction-inverted (always no-op) sliding-window mask.

    fn attn_test_config() -> MistralConfig {
        MistralConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 8,
            num_key_value_heads: 2, // GQA
            sliding_window: Some(4096),
            ..MistralConfig::default()
        }
    }

    fn lcg_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn f32_data(t: &Tensor) -> Vec<f32> {
        match t {
            Tensor::F32(arr) => arr.as_slice().expect("contiguous").to_vec(),
            _ => panic!("expected F32 tensor"),
        }
    }

    #[test]
    fn test_mistral_attention_output_shape() {
        let config = attn_test_config();
        let attn = MistralAttention::new(&config).expect("attention");
        let seq_len = 4;
        let hidden = config.hidden_size;
        let input =
            Tensor::from_vec(lcg_vec(seq_len * hidden, 1), &[seq_len, hidden]).expect("tensor");
        let output = attn.forward(input).expect("forward");
        assert_eq!(output.shape(), vec![seq_len, hidden]);
    }

    /// Changing an EARLY token must change a LATER position's output — the
    /// discriminating test that only passes for real QK^T/softmax/V
    /// attention (the old RoPE no-op still ran real attention math, so this
    /// specifically targets the RoPE fix via position sensitivity below;
    /// this test targets the attention mixing itself, which was already
    /// real prior to this fix and must remain so).
    #[test]
    fn test_mistral_attention_early_token_change_propagates_forward() {
        let config = attn_test_config();
        let attn = MistralAttention::new(&config).expect("attention");
        let seq_len = 4;
        let hidden = config.hidden_size;

        let base = lcg_vec(seq_len * hidden, 21);
        let mut modified = base.clone();
        for x in modified[0..hidden].iter_mut() {
            *x += 5.0;
        }

        let out_base = attn
            .forward(Tensor::from_vec(base, &[seq_len, hidden]).expect("t"))
            .expect("fwd");
        let out_mod = attn
            .forward(Tensor::from_vec(modified, &[seq_len, hidden]).expect("t"))
            .expect("fwd");

        let a = f32_data(&out_base);
        let b = f32_data(&out_mod);
        let last_a = &a[3 * hidden..4 * hidden];
        let last_b = &b[3 * hidden..4 * hidden];
        let differs = last_a.iter().zip(last_b.iter()).any(|(x, y)| (x - y).abs() > 1e-5);
        assert!(differs, "changing token 0 must change token 3's output");
    }

    /// Causal masking: changing the LAST token must not change any earlier
    /// position's output.
    #[test]
    fn test_mistral_attention_causal_mask_future_does_not_leak_backward() {
        let config = attn_test_config();
        let attn = MistralAttention::new(&config).expect("attention");
        let seq_len = 4;
        let hidden = config.hidden_size;

        let base = lcg_vec(seq_len * hidden, 22);
        let mut modified = base.clone();
        for x in modified[3 * hidden..4 * hidden].iter_mut() {
            *x += 5.0;
        }

        let out_base = attn
            .forward(Tensor::from_vec(base, &[seq_len, hidden]).expect("t"))
            .expect("fwd");
        let out_mod = attn
            .forward(Tensor::from_vec(modified, &[seq_len, hidden]).expect("t"))
            .expect("fwd");

        let a = f32_data(&out_base);
        let b = f32_data(&out_mod);
        for row in 0..3 {
            let ra = &a[row * hidden..(row + 1) * hidden];
            let rb = &b[row * hidden..(row + 1) * hidden];
            for (x, y) in ra.iter().zip(rb.iter()) {
                assert!(
                    (x - y).abs() < 1e-6,
                    "row {row} must be unaffected by a later change"
                );
            }
        }
    }

    /// The sliding window must actually exclude distant keys: with
    /// `sliding_window = 1`, a window of size 1 covers only distance 0 (each
    /// query attends only to itself — see `apply_sliding_window_mask`'s
    /// `distance < window_size` boundary), so perturbing token 0 (distance 3
    /// from token 3) must leave token 3's output unchanged.
    #[test]
    fn test_mistral_attention_sliding_window_excludes_distant_tokens() {
        let mut config = attn_test_config();
        config.sliding_window = Some(1);
        let attn = MistralAttention::new(&config).expect("attention");
        let seq_len = 4;
        let hidden = config.hidden_size;

        let base = lcg_vec(seq_len * hidden, 23);
        let mut modified = base.clone();
        for x in modified[0..hidden].iter_mut() {
            *x += 5.0; // token 0 is outside token 3's window
        }

        let out_base = attn
            .forward(Tensor::from_vec(base, &[seq_len, hidden]).expect("t"))
            .expect("fwd");
        let out_mod = attn
            .forward(Tensor::from_vec(modified, &[seq_len, hidden]).expect("t"))
            .expect("fwd");

        let a = f32_data(&out_base);
        let b = f32_data(&out_mod);
        let last_a = &a[3 * hidden..4 * hidden];
        let last_b = &b[3 * hidden..4 * hidden];
        for (x, y) in last_a.iter().zip(last_b.iter()) {
            assert!(
                (x - y).abs() < 1e-6,
                "token 3 with sliding_window=1 must not see token 0 at all"
            );
        }
    }

    /// Boundary regression: a window of size W covers exactly W positions
    /// (distances `0..W`), matching `gemma2`/`qwen`/`qwen2_5`'s `i - j >= w`
    /// exclusion convention. With `sliding_window = 2` and 4 tokens, token 3
    /// (index 3) must see token 1 (distance 2 — excluded, `2 >= 2`) not at
    /// all, but must still see token 2 (distance 1 — included, `1 < 2`).
    /// This pins down the exact boundary that
    /// `apply_sliding_window_mask`'s old `diff > window_size` condition got
    /// wrong (it included distance == window_size, one position too many);
    /// this test would have FAILED against that old boundary since
    /// perturbing token 1 would NOT have changed token 3's output there
    /// either (both conventions exclude it), but perturbing token 2 must
    /// differ from perturbing token 1 in whether it's visible, which this
    /// test also verifies via the "still sees token 2" half.
    #[test]
    fn test_mistral_attention_sliding_window_boundary_distance_equals_window_excluded() {
        let mut config = attn_test_config();
        config.sliding_window = Some(2);
        let attn = MistralAttention::new(&config).expect("attention");
        let seq_len = 4;
        let hidden = config.hidden_size;

        let base = lcg_vec(seq_len * hidden, 63);

        // Perturb token 1 (distance 2 from token 3 == window_size -> excluded).
        let mut modified_far = base.clone();
        for x in modified_far[hidden..2 * hidden].iter_mut() {
            *x += 5.0;
        }
        // Perturb token 2 (distance 1 from token 3 < window_size -> included).
        let mut modified_near = base.clone();
        for x in modified_near[2 * hidden..3 * hidden].iter_mut() {
            *x += 5.0;
        }

        let out_base = attn
            .forward(Tensor::from_vec(base, &[seq_len, hidden]).expect("t"))
            .expect("fwd base");
        let out_far = attn
            .forward(Tensor::from_vec(modified_far, &[seq_len, hidden]).expect("t"))
            .expect("fwd far");
        let out_near = attn
            .forward(Tensor::from_vec(modified_near, &[seq_len, hidden]).expect("t"))
            .expect("fwd near");

        let a = f32_data(&out_base);
        let far = f32_data(&out_far);
        let near = f32_data(&out_near);

        let last_a = &a[3 * hidden..4 * hidden];
        let last_far = &far[3 * hidden..4 * hidden];
        let last_near = &near[3 * hidden..4 * hidden];

        for (x, y) in last_a.iter().zip(last_far.iter()) {
            assert!(
                (x - y).abs() < 1e-6,
                "token 3 must NOT see token 1 (distance == window_size=2)"
            );
        }
        let differs = last_a.iter().zip(last_near.iter()).any(|(x, y)| (x - y).abs() > 1e-5);
        assert!(
            differs,
            "token 3 MUST see token 2 (distance 1 < window_size=2)"
        );
    }

    /// Without a sliding window, a change to token 0 CAN affect token 3
    /// (contrast with the windowed test above) — confirms the window is
    /// what causes the exclusion, not some other masking bug.
    #[test]
    fn test_mistral_attention_no_window_lets_distant_tokens_influence_output() {
        let mut config = attn_test_config();
        config.sliding_window = None;
        let attn = MistralAttention::new(&config).expect("attention");
        let seq_len = 4;
        let hidden = config.hidden_size;

        let base = lcg_vec(seq_len * hidden, 24);
        let mut modified = base.clone();
        for x in modified[0..hidden].iter_mut() {
            *x += 5.0;
        }

        let out_base = attn
            .forward(Tensor::from_vec(base, &[seq_len, hidden]).expect("t"))
            .expect("fwd");
        let out_mod = attn
            .forward(Tensor::from_vec(modified, &[seq_len, hidden]).expect("t"))
            .expect("fwd");

        let a = f32_data(&out_base);
        let b = f32_data(&out_mod);
        let last_a = &a[3 * hidden..4 * hidden];
        let last_b = &b[3 * hidden..4 * hidden];
        let differs = last_a.iter().zip(last_b.iter()).any(|(x, y)| (x - y).abs() > 1e-5);
        assert!(differs, "without a window, token 3 must see token 0");
    }

    /// Causal property with a growing sequence: forwarding a prefix of
    /// length N and then forwarding that same prefix plus one appended
    /// token must leave rows `0..N` of the output bit-identical. This is
    /// the variable-length analogue of "appending a token to the KV cache
    /// does not change earlier positions' outputs" for this crate's
    /// stateless `Layer::forward` (there is no incremental KV cache in the
    /// `Layer` API; `seq_len` and `position_ids` are recomputed fresh from
    /// the input every call). It's a stronger check than the fixed-length
    /// "modify a token" tests above: it also catches a mask or RoPE angle
    /// that was accidentally computed from total `seq_len` instead of each
    /// row's own position. `attn_test_config`'s `sliding_window = Some(4096)`
    /// is far larger than this test's sequence length, so the window never
    /// engages here.
    #[test]
    fn test_mistral_attention_prefix_extension_preserves_earlier_outputs() {
        let config = attn_test_config();
        let attn = MistralAttention::new(&config).expect("attention");
        let hidden = config.hidden_size;
        let prefix_len = 3;

        let prefix = lcg_vec(prefix_len * hidden, 61);
        let mut extended = prefix.clone();
        extended.extend(lcg_vec(hidden, 62));

        let out_prefix = attn
            .forward(Tensor::from_vec(prefix, &[prefix_len, hidden]).expect("t"))
            .expect("fwd prefix");
        let out_extended = attn
            .forward(Tensor::from_vec(extended, &[prefix_len + 1, hidden]).expect("t"))
            .expect("fwd extended");

        let a = f32_data(&out_prefix);
        let b = f32_data(&out_extended);
        for row in 0..prefix_len {
            let ra = &a[row * hidden..(row + 1) * hidden];
            let rb = &b[row * hidden..(row + 1) * hidden];
            for (x, y) in ra.iter().zip(rb.iter()) {
                assert!(
                    (x - y).abs() < 1e-5,
                    "row {row} must be unchanged when a new token is appended after it"
                );
            }
        }
    }

    // -- RoPE primitive regression tests --
    //
    // `MistralAttention::forward`'s attention-mixing tests above don't
    // discriminate RoPE correctness on their own: with `sliding_window`
    // large enough, a no-op RoPE still passes them, because the rest of the
    // attention path (GQA repeat, QK^T, causal mask) was already real. So
    // RoPE itself is verified directly against `apply_rope_rotate_half`,
    // the same way gemma/qwen/gpt_j test their RoPE primitives. (A tempting
    // alternative — feed identical tokens at every position and check that
    // attention output differs by position — does NOT work: when V is
    // identical at every position, any softmax-weighted average of V is
    // still exactly V, regardless of what RoPE does to the weights.)

    use crate::mistral::model::apply_rope_rotate_half;

    #[test]
    fn test_mistral_rope_position_zero_is_identity() {
        let data_orig = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut data = data_orig.clone();
        apply_rope_rotate_half(&mut data, 1, 4, 10000.0, &[0]);
        for (a, b) in data.iter().zip(data_orig.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "position 0 must be an identity rotation"
            );
        }
    }

    #[test]
    fn test_mistral_rope_different_positions_differ() {
        let mut at_pos0 = vec![1.0f32; 4];
        let mut at_pos5 = vec![1.0f32; 4];
        apply_rope_rotate_half(&mut at_pos0, 1, 4, 10000.0, &[0]);
        apply_rope_rotate_half(&mut at_pos5, 1, 4, 10000.0, &[5]);
        let differs = at_pos0.iter().zip(at_pos5.iter()).any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            differs,
            "RoPE must rotate differently at different positions"
        );
    }

    /// Regression: RoPE must rotate every head, not just the first
    /// `head_dim`-wide block of a multi-head row.
    #[test]
    fn test_mistral_rope_rotates_every_head() {
        let head_dim = 4;
        let mut data = vec![1.0f32; 2 * head_dim]; // seq_len=1, num_heads=2
        let original = data.clone();
        apply_rope_rotate_half(&mut data, 2, head_dim, 10000.0, &[7]);
        let head0_changed = data[0..head_dim]
            .iter()
            .zip(&original[0..head_dim])
            .any(|(a, b)| (a - b).abs() > 1e-4);
        let head1_changed = data[head_dim..2 * head_dim]
            .iter()
            .zip(&original[head_dim..2 * head_dim])
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(head0_changed, "head 0 must rotate at a non-zero position");
        assert!(head1_changed, "head 1 must ALSO rotate, not just head 0");
    }
}
