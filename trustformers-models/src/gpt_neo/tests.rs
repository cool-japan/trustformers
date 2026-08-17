#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::gpt_neo::config::GptNeoConfig;
    use crate::gpt_neo::model::{GptNeoAttention, GptNeoLMHeadModel, GptNeoModel};
    use trustformers_core::tensor::Tensor;
    use trustformers_core::traits::Config;

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

    fn minimal_gpt_neo_config() -> GptNeoConfig {
        GptNeoConfig {
            vocab_size: 512,
            hidden_size: 64,
            num_layers: 2,
            attention_types: vec!["global".to_string(), "local".to_string()],
            num_heads: 8,
            intermediate_size: 256,
            window_size: 32,
            activation_function: "gelu_new".to_string(),
            resid_dropout: 0.0,
            embed_dropout: 0.0,
            attention_dropout: 0.0,
            max_position_embeddings: 128,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            use_cache: true,
            bos_token_id: 50256,
            eos_token_id: 50256,
            model_type: "gpt_neo".to_string(),
        }
    }

    // ── Default config tests ──────────────────────────────────────────────────

    #[test]
    fn test_gpt_neo_default_config_is_valid() {
        let config = GptNeoConfig::default();
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_default_config_params() {
        let config = GptNeoConfig::default();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.window_size, 256);
        assert_eq!(config.model_type, "gpt_neo");
        drop(config);
        std::hint::black_box(());
    }

    // ── Preset configs ────────────────────────────────────────────────────────

    #[test]
    fn test_gpt_neo_125m_config() {
        let config = GptNeoConfig::gpt_neo_125m();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.attention_types.len(), 12);
        // Alternating global/local
        assert_eq!(config.attention_types[0], "global");
        assert_eq!(config.attention_types[1], "local");
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_1_3b_config() {
        let config = GptNeoConfig::gpt_neo_1_3b();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.intermediate_size, 8192);
        assert_eq!(config.attention_types.len(), 24);
        // Even indices are global
        assert_eq!(config.attention_types[0], "global");
        assert_eq!(config.attention_types[1], "local");
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_2_7b_config() {
        let config = GptNeoConfig::gpt_neo_2_7b();
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 20);
        assert_eq!(config.intermediate_size, 10240);
        assert_eq!(config.attention_types.len(), 32);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    // ── Architecture string ───────────────────────────────────────────────────

    #[test]
    fn test_gpt_neo_architecture_string() {
        let config = GptNeoConfig::default();
        assert_eq!(config.architecture(), "GPT-Neo");
    }

    // ── Validation failure tests ──────────────────────────────────────────────

    #[test]
    fn test_gpt_neo_invalid_hidden_size_not_divisible_by_heads() {
        let mut config = minimal_gpt_neo_config();
        config.hidden_size = 65; // not divisible by 8
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gpt_neo_invalid_empty_attention_types() {
        let mut config = minimal_gpt_neo_config();
        config.attention_types = vec![];
        assert!(config.validate().is_err());
    }

    // ── from_pretrained_name tests ────────────────────────────────────────────

    #[test]
    fn test_gpt_neo_from_pretrained_name_125m() {
        let config = GptNeoConfig::from_pretrained_name("gpt-neo-125M");
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_heads, 12);
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_from_pretrained_name_1_3b() {
        let config = GptNeoConfig::from_pretrained_name("gpt-neo-1.3b");
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_heads, 16);
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_from_pretrained_name_1_3b_underscore() {
        let config = GptNeoConfig::from_pretrained_name("gpt-neo-1_3b");
        assert_eq!(config.hidden_size, 2048);
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_from_pretrained_name_2_7b() {
        let config = GptNeoConfig::from_pretrained_name("gpt-neo-2.7b");
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_heads, 20);
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_from_pretrained_name_unknown_defaults_to_125m() {
        let config = GptNeoConfig::from_pretrained_name("unknown-variant");
        assert_eq!(config.hidden_size, 768);
        drop(config);
        std::hint::black_box(());
    }

    // ── Attention types alternation ───────────────────────────────────────────

    #[test]
    fn test_gpt_neo_125m_attention_type_pattern() {
        let config = GptNeoConfig::gpt_neo_125m();
        for (i, attn_type) in config.attention_types.iter().enumerate() {
            let expected = if i % 2 == 0 { "global" } else { "local" };
            assert_eq!(attn_type.as_str(), expected, "layer {} mismatch", i);
        }
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_1_3b_attention_type_pattern() {
        let config = GptNeoConfig::gpt_neo_1_3b();
        for (i, attn_type) in config.attention_types.iter().enumerate() {
            let expected = if i % 2 == 0 { "global" } else { "local" };
            assert_eq!(attn_type.as_str(), expected, "layer {} mismatch", i);
        }
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_attention_types_count_matches_layers_125m() {
        let config = GptNeoConfig::gpt_neo_125m();
        assert_eq!(config.attention_types.len(), config.num_layers);
    }

    #[test]
    fn test_gpt_neo_attention_types_count_matches_layers_1_3b() {
        let config = GptNeoConfig::gpt_neo_1_3b();
        assert_eq!(config.attention_types.len(), config.num_layers);
    }

    #[test]
    fn test_gpt_neo_attention_types_count_matches_layers_2_7b() {
        let config = GptNeoConfig::gpt_neo_2_7b();
        assert_eq!(config.attention_types.len(), config.num_layers);
    }

    // ── Model creation tests ──────────────────────────────────────────────────

    #[test]
    fn test_gpt_neo_model_creation_minimal() {
        let config = minimal_gpt_neo_config();
        let model = GptNeoModel::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_lm_head_model_creation_minimal() {
        let config = minimal_gpt_neo_config();
        let model = GptNeoLMHeadModel::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    // ── Token IDs ─────────────────────────────────────────────────────────────

    #[test]
    fn test_gpt_neo_125m_token_ids() {
        let config = GptNeoConfig::gpt_neo_125m();
        assert_eq!(config.bos_token_id, 50256);
        assert_eq!(config.eos_token_id, 50256);
    }

    #[test]
    fn test_gpt_neo_dropout_values_in_pretrained() {
        let config = GptNeoConfig::gpt_neo_125m();
        // Pre-trained configs use 0.0 dropout
        assert_eq!(config.resid_dropout, 0.0);
        assert_eq!(config.embed_dropout, 0.0);
        assert_eq!(config.attention_dropout, 0.0);
    }

    // ── Config cloning ────────────────────────────────────────────────────────

    #[test]
    fn test_gpt_neo_config_clone() {
        let config = minimal_gpt_neo_config();
        let cloned = config.clone();
        assert_eq!(config.vocab_size, cloned.vocab_size);
        assert_eq!(config.hidden_size, cloned.hidden_size);
        assert_eq!(config.attention_types.len(), cloned.attention_types.len());
        drop(config);
        drop(cloned);
        std::hint::black_box(());
    }

    // ── LCG reproducibility ───────────────────────────────────────────────────

    #[test]
    fn test_lcg_reproducibility() {
        let mut rng1 = Lcg::new(1111);
        let mut rng2 = Lcg::new(1111);
        for _ in 0..30 {
            assert_eq!(rng1.next_f32(), rng2.next_f32());
        }
    }

    #[test]
    fn test_lcg_range() {
        let mut rng = Lcg::new(7777);
        for _ in 0..100 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }

    // ── Real attention regression tests ───────────────────────────────────────
    //
    // These exercise real multi-head scaled dot-product attention with
    // causal masking and a real alternating local/global sliding window,
    // replacing `out_proj(v)` (Q and K discarded, no local/global
    // distinction at all). Every test below would have FAILED against that
    // old code.

    fn gptneo_lcg_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Lcg::new(seed);
        (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
    }

    fn gptneo_f32_data(t: &Tensor) -> Vec<f32> {
        match t {
            Tensor::F32(arr) => arr.as_slice().expect("contiguous").to_vec(),
            _ => panic!("expected F32 tensor"),
        }
    }

    #[test]
    fn test_gptneo_global_and_local_window_sizes() {
        let config = minimal_gpt_neo_config();
        let global = GptNeoAttention::new(&config, "global").expect("global attn");
        let local = GptNeoAttention::new(&config, "local").expect("local attn");
        assert_eq!(
            global.window_size(),
            None,
            "global layer must have no window"
        );
        assert_eq!(
            local.window_size(),
            Some(config.window_size),
            "local layer must use config.window_size"
        );
    }

    #[test]
    fn test_gptneo_attention_output_shape() {
        let config = minimal_gpt_neo_config();
        let attn = GptNeoAttention::new(&config, "global").expect("attn");
        let seq_len = 4;
        let hidden = config.hidden_size;
        let input =
            Tensor::from_vec(gptneo_lcg_vec(seq_len * hidden, 1), &[seq_len, hidden]).expect("t");
        let out = attn.forward(input, None).expect("forward");
        assert_eq!(out.shape(), vec![seq_len, hidden]);
    }

    /// Changing an EARLY token must change a LATER position's output — the
    /// discriminating test that only real QK^T/softmax/V attention passes;
    /// the old `out_proj(v)` fake path never even read Q or K.
    #[test]
    fn test_gptneo_attention_early_token_change_propagates_forward() {
        let config = minimal_gpt_neo_config();
        let attn = GptNeoAttention::new(&config, "global").expect("attn");
        let seq_len = 4;
        let hidden = config.hidden_size;

        let base = gptneo_lcg_vec(seq_len * hidden, 41);
        let mut modified = base.clone();
        for x in modified[0..hidden].iter_mut() {
            *x += 5.0;
        }

        let out_base = attn
            .forward(Tensor::from_vec(base, &[seq_len, hidden]).expect("t"), None)
            .expect("fwd");
        let out_mod = attn
            .forward(
                Tensor::from_vec(modified, &[seq_len, hidden]).expect("t"),
                None,
            )
            .expect("fwd");

        let a = gptneo_f32_data(&out_base);
        let b = gptneo_f32_data(&out_mod);
        let last_a = &a[3 * hidden..4 * hidden];
        let last_b = &b[3 * hidden..4 * hidden];
        let differs = last_a.iter().zip(last_b.iter()).any(|(x, y)| (x - y).abs() > 1e-5);
        assert!(differs, "changing token 0 must change token 3's output");
    }

    #[test]
    fn test_gptneo_attention_causal_mask_future_does_not_leak_backward() {
        let config = minimal_gpt_neo_config();
        let attn = GptNeoAttention::new(&config, "global").expect("attn");
        let seq_len = 4;
        let hidden = config.hidden_size;

        let base = gptneo_lcg_vec(seq_len * hidden, 42);
        let mut modified = base.clone();
        for x in modified[3 * hidden..4 * hidden].iter_mut() {
            *x += 5.0;
        }

        let out_base = attn
            .forward(Tensor::from_vec(base, &[seq_len, hidden]).expect("t"), None)
            .expect("fwd");
        let out_mod = attn
            .forward(
                Tensor::from_vec(modified, &[seq_len, hidden]).expect("t"),
                None,
            )
            .expect("fwd");

        let a = gptneo_f32_data(&out_base);
        let b = gptneo_f32_data(&out_mod);
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

    /// A "local" layer's window must actually exclude distant keys: with
    /// `window_size = 1`, token 3 can only see itself, so perturbing token 0
    /// must leave token 3's output completely unchanged.
    #[test]
    fn test_gptneo_local_layer_window_excludes_distant_tokens() {
        let mut config = minimal_gpt_neo_config();
        config.window_size = 1;
        let attn = GptNeoAttention::new(&config, "local").expect("attn");
        let seq_len = 4;
        let hidden = config.hidden_size;

        let base = gptneo_lcg_vec(seq_len * hidden, 43);
        let mut modified = base.clone();
        for x in modified[0..hidden].iter_mut() {
            *x += 5.0;
        }

        let out_base = attn
            .forward(Tensor::from_vec(base, &[seq_len, hidden]).expect("t"), None)
            .expect("fwd");
        let out_mod = attn
            .forward(
                Tensor::from_vec(modified, &[seq_len, hidden]).expect("t"),
                None,
            )
            .expect("fwd");

        let a = gptneo_f32_data(&out_base);
        let b = gptneo_f32_data(&out_mod);
        let last_a = &a[3 * hidden..4 * hidden];
        let last_b = &b[3 * hidden..4 * hidden];
        for (x, y) in last_a.iter().zip(last_b.iter()) {
            assert!(
                (x - y).abs() < 1e-6,
                "token 3 with window_size=1 must not see token 0 at all"
            );
        }
    }

    /// Contrast with the windowed test above: a "global" layer over the
    /// SAME perturbation must let token 0 influence token 3.
    #[test]
    fn test_gptneo_global_layer_has_no_window_exclusion() {
        let config = minimal_gpt_neo_config();
        let attn = GptNeoAttention::new(&config, "global").expect("attn");
        let seq_len = 4;
        let hidden = config.hidden_size;

        let base = gptneo_lcg_vec(seq_len * hidden, 44);
        let mut modified = base.clone();
        for x in modified[0..hidden].iter_mut() {
            *x += 5.0;
        }

        let out_base = attn
            .forward(Tensor::from_vec(base, &[seq_len, hidden]).expect("t"), None)
            .expect("fwd");
        let out_mod = attn
            .forward(
                Tensor::from_vec(modified, &[seq_len, hidden]).expect("t"),
                None,
            )
            .expect("fwd");

        let a = gptneo_f32_data(&out_base);
        let b = gptneo_f32_data(&out_mod);
        let last_a = &a[3 * hidden..4 * hidden];
        let last_b = &b[3 * hidden..4 * hidden];
        let differs = last_a.iter().zip(last_b.iter()).any(|(x, y)| (x - y).abs() > 1e-5);
        assert!(differs, "a global layer must let token 0 influence token 3");
    }

    /// Causal property with a growing sequence: forwarding a prefix of
    /// length N and then forwarding that same prefix plus one appended
    /// token must leave rows `0..N` of the output bit-identical. This is
    /// the variable-length analogue of "appending a token to the KV cache
    /// does not change earlier positions' outputs" for this crate's
    /// stateless `Layer::forward` (there is no incremental KV cache in the
    /// attention API; `seq_len` is recomputed fresh from the input on every
    /// call). Checked on the "global" attention type; GPT-Neo has no RoPE,
    /// so this specifically exercises the causal-mask construction rather
    /// than a position-dependent rotation.
    #[test]
    fn test_gptneo_attention_prefix_extension_preserves_earlier_outputs() {
        let config = minimal_gpt_neo_config();
        let attn = GptNeoAttention::new(&config, "global").expect("attn");
        let hidden = config.hidden_size;
        let prefix_len = 3;

        let prefix = gptneo_lcg_vec(prefix_len * hidden, 81);
        let mut extended = prefix.clone();
        extended.extend(gptneo_lcg_vec(hidden, 82));

        let out_prefix = attn
            .forward(
                Tensor::from_vec(prefix, &[prefix_len, hidden]).expect("t"),
                None,
            )
            .expect("fwd prefix");
        let out_extended = attn
            .forward(
                Tensor::from_vec(extended, &[prefix_len + 1, hidden]).expect("t"),
                None,
            )
            .expect("fwd extended");

        let a = gptneo_f32_data(&out_prefix);
        let b = gptneo_f32_data(&out_extended);
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
