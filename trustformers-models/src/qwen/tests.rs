#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::qwen::config::{QwenConfig, RopeScaling};
    use crate::qwen::model::{
        QwenAttention, QwenForCausalLM, QwenModel, QwenRMSNorm, QwenRotaryEmbedding,
    };
    use trustformers_core::tensor::Tensor;
    use trustformers_core::traits::{Config, Layer};

    // ── LCG ───────────────────────────────────────────────────────────────────
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }

        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005_u64)
                .wrapping_add(1_442_695_040_888_963_407_u64);
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1_u64 << 53) as f32
        }
    }

    // ── Minimal config ────────────────────────────────────────────────────────

    fn minimal_qwen_config() -> QwenConfig {
        QwenConfig {
            vocab_size: 512,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: Some(2),
            hidden_act: "silu".to_string(),
            max_position_embeddings: 256,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
            pad_token_id: None,
            bos_token_id: 1,
            eos_token_id: 2,
            rope_theta: 1_000_000.0,
            rope_scaling: None,
            attention_dropout: 0.0,
            use_sliding_window: false,
            sliding_window: None,
            max_window_layers: None,
            use_logn_attn: false,
            logn_list: None,
            model_type: "qwen2".to_string(),
        }
    }

    // ── Default config tests ──────────────────────────────────────────────────

    #[test]
    fn test_qwen_default_config_is_valid() {
        let config = QwenConfig::default();
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen_default_config_params() {
        let config = QwenConfig::default();
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.model_type, "qwen2");
        drop(config);
        std::hint::black_box(());
    }

    // ── Preset configs ────────────────────────────────────────────────────────

    #[test]
    fn test_qwen2_0_5b_config() {
        let config = QwenConfig::qwen2_0_5b();
        assert_eq!(config.hidden_size, 896);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 14);
        assert_eq!(config.num_key_value_heads, Some(2));
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen2_1_5b_config() {
        let config = QwenConfig::qwen2_1_5b();
        assert_eq!(config.hidden_size, 1536);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.num_key_value_heads, Some(2));
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen2_7b_config() {
        let config = QwenConfig::qwen2_7b();
        assert_eq!(config.hidden_size, 3584);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 28);
        assert_eq!(config.num_key_value_heads, Some(4));
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen2_72b_config() {
        let config = QwenConfig::qwen2_72b();
        assert_eq!(config.hidden_size, 8192);
        assert_eq!(config.num_hidden_layers, 80);
        assert_eq!(config.num_attention_heads, 64);
        assert_eq!(config.num_key_value_heads, Some(8));
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen2_5_7b_config() {
        let config = QwenConfig::qwen2_5_7b();
        assert_eq!(config.max_position_embeddings, 131072);
        assert_eq!(config.model_type, "qwen2.5");
        assert!(config.is_qwen2_5());
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen2_5_14b_config() {
        let config = QwenConfig::qwen2_5_14b();
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.num_hidden_layers, 48);
        assert_eq!(config.model_type, "qwen2.5");
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen2_5_32b_config() {
        let config = QwenConfig::qwen2_5_32b();
        assert_eq!(config.num_hidden_layers, 64);
        assert_eq!(config.max_position_embeddings, 131072);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen2_5_72b_config() {
        let config = QwenConfig::qwen2_5_72b();
        assert_eq!(config.hidden_size, 8192);
        assert_eq!(config.num_hidden_layers, 80);
        assert_eq!(config.model_type, "qwen2.5");
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen2_5_coder_7b_config() {
        let config = QwenConfig::qwen2_5_coder_7b();
        assert_eq!(config.model_type, "qwen2.5-coder");
        assert_eq!(config.max_position_embeddings, 131072);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    // ── Architecture string ───────────────────────────────────────────────────

    #[test]
    fn test_qwen_architecture_string() {
        let config = QwenConfig::default();
        assert_eq!(config.architecture(), "Qwen");
    }

    // ── Validation failure tests ──────────────────────────────────────────────

    #[test]
    fn test_qwen_invalid_hidden_size_not_divisible_by_heads() {
        let mut config = minimal_qwen_config();
        config.hidden_size = 129; // 129 not divisible by 8
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_qwen_invalid_heads_not_divisible_by_kv_heads() {
        let mut config = minimal_qwen_config();
        config.num_key_value_heads = Some(3); // 8 not divisible by 3
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_qwen_invalid_vocab_size_zero() {
        let mut config = minimal_qwen_config();
        config.vocab_size = 0;
        assert!(config.validate().is_err());
    }

    // ── Helper method tests ───────────────────────────────────────────────────

    #[test]
    fn test_qwen_head_dim() {
        let config = QwenConfig::qwen2_7b();
        // head_dim = 3584 / 28 = 128
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn test_qwen_num_kv_heads_explicit() {
        let config = QwenConfig::qwen2_7b();
        assert_eq!(config.num_kv_heads(), 4);
    }

    #[test]
    fn test_qwen_num_kv_heads_fallback() {
        let mut config = minimal_qwen_config();
        config.num_key_value_heads = None;
        // Should fall back to num_attention_heads
        assert_eq!(config.num_kv_heads(), config.num_attention_heads);
    }

    #[test]
    fn test_qwen_num_query_groups() {
        let config = QwenConfig::qwen2_7b();
        // 28 heads / 4 kv heads = 7 groups
        assert_eq!(config.num_query_groups(), 7);
    }

    #[test]
    fn test_qwen_uses_grouped_query_attention_true() {
        let config = QwenConfig::qwen2_7b();
        assert!(config.uses_grouped_query_attention());
    }

    #[test]
    fn test_qwen_uses_grouped_query_attention_false_when_equal() {
        let config = QwenConfig::default(); // num_kv_heads = Some(32) == num_attention_heads = 32
        assert!(!config.uses_grouped_query_attention());
    }

    #[test]
    fn test_qwen_uses_grouped_query_attention_false_when_none() {
        let mut config = minimal_qwen_config();
        config.num_key_value_heads = None;
        assert!(!config.uses_grouped_query_attention());
    }

    #[test]
    fn test_qwen_sliding_window_false_by_default() {
        let config = QwenConfig::default();
        assert!(!config.uses_sliding_window());
    }

    #[test]
    fn test_qwen_sliding_window_true_when_both_set() {
        let mut config = minimal_qwen_config();
        config.use_sliding_window = true;
        config.sliding_window = Some(256);
        assert!(config.uses_sliding_window());
        assert_eq!(config.sliding_window_size(), 256);
    }

    #[test]
    fn test_qwen_sliding_window_false_when_flag_false() {
        let mut config = minimal_qwen_config();
        config.use_sliding_window = false;
        config.sliding_window = Some(256); // window set but flag false
        assert!(!config.uses_sliding_window());
    }

    #[test]
    fn test_qwen_sliding_window_size_falls_back_to_max_position() {
        let config = minimal_qwen_config(); // sliding_window = None
        assert_eq!(config.sliding_window_size(), config.max_position_embeddings);
    }

    #[test]
    fn test_qwen_is_qwen2_5_true() {
        let config = QwenConfig::qwen2_5_7b();
        assert!(config.is_qwen2_5());
    }

    #[test]
    fn test_qwen_is_qwen2_5_false_for_qwen2() {
        let config = QwenConfig::qwen2_7b();
        assert!(!config.is_qwen2_5());
    }

    // ── RMSNorm creation ──────────────────────────────────────────────────────

    #[test]
    fn test_qwen_rmsnorm_creation() {
        let norm = QwenRMSNorm::new(128, 1e-6);
        assert!(norm.is_ok());
        if let Ok(n) = norm {
            assert_eq!(n.parameter_count(), 128);
        }
        std::hint::black_box(());
    }

    // ── Model creation tests ──────────────────────────────────────────────────

    #[test]
    fn test_qwen_model_creation_minimal() {
        let config = minimal_qwen_config();
        let model = QwenModel::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen_for_causal_lm_creation_minimal() {
        let config = minimal_qwen_config();
        let model = QwenForCausalLM::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    // ── LCG reproducibility ───────────────────────────────────────────────────

    #[test]
    fn test_lcg_reproducibility() {
        let mut rng1 = Lcg::new(0xDEAD_BEEF);
        let mut rng2 = Lcg::new(0xDEAD_BEEF);
        for _ in 0..50 {
            assert_eq!(rng1.next_f32(), rng2.next_f32());
        }
    }

    #[test]
    fn test_lcg_range() {
        let mut rng = Lcg::new(42);
        for _ in 0..200 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }

    // ── Real attention / RoPE regression tests ────────────────────────────────
    //
    // These exercise the real RoPE (including "linear" and "dynamic" NTK
    // rope_scaling) and real scaled dot-product attention that replaced a
    // no-op RoPE and `o_proj(scale*Q + V)` fake attention. Every test below
    // would have FAILED against that old code.

    fn qwen_lcg_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Lcg::new(seed);
        (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
    }

    fn qwen_f32_data(t: &Tensor) -> Vec<f32> {
        match t {
            Tensor::F32(arr) => arr.as_slice().expect("contiguous").to_vec(),
            _ => panic!("expected F32 tensor"),
        }
    }

    #[test]
    fn test_qwen_rope_position_zero_is_identity() {
        let rope = QwenRotaryEmbedding::new(4, 32, 10000.0, None);
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let q = Tensor::from_vec(data.clone(), &[1, 4]).expect("tensor");
        let k = q.clone();
        let (q_out, _) = rope.apply_rotary_emb(&q, &k, &[0]).expect("rope");
        let out = qwen_f32_data(&q_out);
        for (a, b) in data.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-5, "position 0 must be identity");
        }
    }

    #[test]
    fn test_qwen_rope_rotates_every_head() {
        let head_dim = 4;
        let rope = QwenRotaryEmbedding::new(head_dim, 32, 10000.0, None);
        let data = vec![1.0f32; 8]; // seq_len=1, num_heads=2
        let q = Tensor::from_vec(data.clone(), &[1, 8]).expect("tensor");
        let k = q.clone();
        let (q_out, _) = rope.apply_rotary_emb(&q, &k, &[7]).expect("rope");
        let out = qwen_f32_data(&q_out);
        let head0_changed = out[0..4].iter().zip(&data[0..4]).any(|(a, b)| (a - b).abs() > 1e-4);
        let head1_changed = out[4..8].iter().zip(&data[4..8]).any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(head0_changed, "head 0 must rotate");
        assert!(head1_changed, "head 1 must ALSO rotate, not just head 0");
    }

    /// Hand-verifiable property of "linear" (position-interpolation) RoPE
    /// scaling: rotating at position `P` with `scaling_factor = F` must be
    /// numerically identical to rotating the SAME vector, unscaled, at
    /// position `P / F` (both are exactly representable here: P=4, F=2).
    #[test]
    fn test_qwen_rope_linear_scaling_matches_unscaled_at_divided_position() {
        let head_dim = 8;
        let data = vec![0.3f32, -0.7, 1.1, 0.4, -0.2, 0.9, -1.3, 0.6];

        let scaled_rope = QwenRotaryEmbedding::new(
            head_dim,
            64,
            10000.0,
            Some(RopeScaling {
                scaling_type: "linear".to_string(),
                scaling_factor: 2.0,
            }),
        );
        let unscaled_rope = QwenRotaryEmbedding::new(head_dim, 64, 10000.0, None);

        let q = Tensor::from_vec(data.clone(), &[1, head_dim]).expect("tensor");
        let k = q.clone();
        let (scaled_out, _) = scaled_rope.apply_rotary_emb(&q, &k, &[4]).expect("rope scaled");
        let (unscaled_out, _) =
            unscaled_rope.apply_rotary_emb(&q, &k, &[2]).expect("rope unscaled");

        let a = qwen_f32_data(&scaled_out);
        let b = qwen_f32_data(&unscaled_out);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x - y).abs() < 1e-4,
                "linear scaling(factor=2) at pos=4 must equal unscaled at pos=2: {x} vs {y}"
            );
        }
    }

    /// Hand-verifiable property of "dynamic" NTK RoPE scaling (matches HF
    /// transformers' `_compute_dynamic_ntk_parameters`): it is a no-op while
    /// the sequence length stays within `max_position_embeddings`, and must
    /// diverge from unscaled RoPE once positions exceed it.
    #[test]
    fn test_qwen_rope_dynamic_scaling_is_noop_within_max_position() {
        let head_dim = 8;
        let max_seq_len = 16;
        let data = vec![0.3f32, -0.7, 1.1, 0.4, -0.2, 0.9, -1.3, 0.6];

        let dynamic_rope = QwenRotaryEmbedding::new(
            head_dim,
            max_seq_len,
            10000.0,
            Some(RopeScaling {
                scaling_type: "dynamic".to_string(),
                scaling_factor: 4.0,
            }),
        );
        let unscaled_rope = QwenRotaryEmbedding::new(head_dim, max_seq_len, 10000.0, None);

        let q = Tensor::from_vec(data.clone(), &[1, head_dim]).expect("tensor");
        let k = q.clone();
        // Position 5 is well within max_seq_len=16, so the dynamic factor's
        // seq_len clamp keeps it a no-op.
        let (dyn_out, _) = dynamic_rope.apply_rotary_emb(&q, &k, &[5]).expect("rope dynamic");
        let (base_out, _) = unscaled_rope.apply_rotary_emb(&q, &k, &[5]).expect("rope base");
        let a = qwen_f32_data(&dyn_out);
        let b = qwen_f32_data(&base_out);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x - y).abs() < 1e-4,
                "dynamic scaling must be a no-op within max_position_embeddings: {x} vs {y}"
            );
        }
    }

    #[test]
    fn test_qwen_rope_dynamic_scaling_diverges_beyond_max_position() {
        let head_dim = 8;
        let max_seq_len = 16;
        let data = vec![0.3f32, -0.7, 1.1, 0.4, -0.2, 0.9, -1.3, 0.6];

        let dynamic_rope = QwenRotaryEmbedding::new(
            head_dim,
            max_seq_len,
            10000.0,
            Some(RopeScaling {
                scaling_type: "dynamic".to_string(),
                scaling_factor: 4.0,
            }),
        );
        let unscaled_rope = QwenRotaryEmbedding::new(head_dim, max_seq_len, 10000.0, None);

        let q = Tensor::from_vec(data.clone(), &[1, head_dim]).expect("tensor");
        let k = q.clone();
        // Position 31 exceeds max_seq_len=16, so the NTK base adjustment
        // must kick in and diverge from plain unscaled RoPE.
        let (dyn_out, _) = dynamic_rope.apply_rotary_emb(&q, &k, &[31]).expect("rope dynamic");
        let (base_out, _) = unscaled_rope.apply_rotary_emb(&q, &k, &[31]).expect("rope base");
        let a = qwen_f32_data(&dyn_out);
        let b = qwen_f32_data(&base_out);
        let differs = a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-4);
        assert!(
            differs,
            "dynamic scaling must diverge from unscaled RoPE beyond max_position_embeddings"
        );
    }

    #[test]
    fn test_qwen_rope_unsupported_scaling_type_is_a_structured_error() {
        let rope = QwenRotaryEmbedding::new(
            8,
            32,
            10000.0,
            Some(RopeScaling {
                scaling_type: "yarn".to_string(),
                scaling_factor: 2.0,
            }),
        );
        let q = Tensor::from_vec(vec![0.0f32; 8], &[1, 8]).expect("tensor");
        let k = q.clone();
        let result = rope.apply_rotary_emb(&q, &k, &[1]);
        assert!(
            result.is_err(),
            "an unrecognized rope_scaling.scaling_type must return an error, not silently apply a guessed formula"
        );
    }

    fn qwen_attn_config() -> QwenConfig {
        minimal_qwen_config()
    }

    #[test]
    fn test_qwen_attention_output_shape() {
        let config = qwen_attn_config();
        let attn = QwenAttention::new(&config).expect("attention");
        let seq_len = 4;
        let hidden = config.hidden_size;
        let input =
            Tensor::from_vec(qwen_lcg_vec(seq_len * hidden, 1), &[seq_len, hidden]).expect("t");
        let out = attn.forward(input).expect("forward");
        assert_eq!(out.shape(), vec![seq_len, hidden]);
    }

    /// Changing an EARLY token must change a LATER position's output — the
    /// discriminating test that only real QK^T/softmax/V attention passes;
    /// the old `o_proj(scale*Q + V)` fake path is purely position-local.
    #[test]
    fn test_qwen_attention_early_token_change_propagates_forward() {
        let config = qwen_attn_config();
        let attn = QwenAttention::new(&config).expect("attention");
        let seq_len = 4;
        let hidden = config.hidden_size;

        let base = qwen_lcg_vec(seq_len * hidden, 31);
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

        let a = qwen_f32_data(&out_base);
        let b = qwen_f32_data(&out_mod);
        let last_a = &a[3 * hidden..4 * hidden];
        let last_b = &b[3 * hidden..4 * hidden];
        let differs = last_a.iter().zip(last_b.iter()).any(|(x, y)| (x - y).abs() > 1e-5);
        assert!(differs, "changing token 0 must change token 3's output");
    }

    #[test]
    fn test_qwen_attention_causal_mask_future_does_not_leak_backward() {
        let config = qwen_attn_config();
        let attn = QwenAttention::new(&config).expect("attention");
        let seq_len = 4;
        let hidden = config.hidden_size;

        let base = qwen_lcg_vec(seq_len * hidden, 32);
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

        let a = qwen_f32_data(&out_base);
        let b = qwen_f32_data(&out_mod);
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

    #[test]
    fn test_qwen_attention_sliding_window_excludes_distant_tokens() {
        let mut config = qwen_attn_config();
        config.use_sliding_window = true;
        config.sliding_window = Some(1);
        let attn = QwenAttention::new(&config).expect("attention");
        let seq_len = 4;
        let hidden = config.hidden_size;

        let base = qwen_lcg_vec(seq_len * hidden, 33);
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

        let a = qwen_f32_data(&out_base);
        let b = qwen_f32_data(&out_mod);
        let last_a = &a[3 * hidden..4 * hidden];
        let last_b = &b[3 * hidden..4 * hidden];
        for (x, y) in last_a.iter().zip(last_b.iter()) {
            assert!(
                (x - y).abs() < 1e-6,
                "token 3 with sliding_window=1 must not see token 0 at all"
            );
        }
    }

    /// Causal property with a growing sequence: forwarding a prefix of
    /// length N and then forwarding that same prefix plus one appended
    /// token must leave rows `0..N` of the output bit-identical. This is
    /// the variable-length analogue of "appending a token to the KV cache
    /// does not change earlier positions' outputs" for this crate's
    /// stateless `Layer::forward` (there is no incremental KV cache in the
    /// `Layer` API; `seq_len` and `position_ids` are recomputed fresh from
    /// the input on every call, so this is the correctness property an
    /// incremental cache would need to preserve).
    ///
    /// This complements (and is stronger than) the fixed-length
    /// "modify a token, check earlier rows" tests above: it also catches
    /// bugs where a per-row computation leaks *total* sequence length
    /// rather than depending only on each row's own position (e.g. a mask
    /// or RoPE base computed from `seq_len` instead of `position_ids[row]`).
    ///
    /// Uses `rope_scaling: None` (via `qwen_attn_config`/`minimal_qwen_config`)
    /// so RoPE angles depend only on each row's absolute position, not on
    /// `seq_len`. Qwen's `"dynamic"` NTK scaling deliberately recomputes the
    /// RoPE base from `max_position_seen` once the sequence exceeds
    /// `max_position_embeddings` (see `effective_base_and_scale`), which
    /// would legitimately change earlier rows too — that is correct
    /// architecture behavior, not a bug, so this test does not exercise it.
    #[test]
    fn test_qwen_attention_prefix_extension_preserves_earlier_outputs() {
        let config = qwen_attn_config();
        let attn = QwenAttention::new(&config).expect("attention");
        let hidden = config.hidden_size;
        let prefix_len = 3;

        let prefix = qwen_lcg_vec(prefix_len * hidden, 71);
        let mut extended = prefix.clone();
        extended.extend(qwen_lcg_vec(hidden, 72));

        let out_prefix = attn
            .forward(Tensor::from_vec(prefix, &[prefix_len, hidden]).expect("t"))
            .expect("fwd prefix");
        let out_extended = attn
            .forward(Tensor::from_vec(extended, &[prefix_len + 1, hidden]).expect("t"))
            .expect("fwd extended");

        let a = qwen_f32_data(&out_prefix);
        let b = qwen_f32_data(&out_extended);
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
}
