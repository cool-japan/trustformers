#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::gemma::config::GemmaConfig;
    use crate::gemma::model::{
        GemmaAttention, GemmaForCausalLM, GemmaModel, GemmaRMSNorm, GemmaRotaryEmbedding,
    };
    use trustformers_core::tensor::Tensor;
    use trustformers_core::traits::{Config, Layer};

    // ── LCG for deterministic pseudo-random data ──────────────────────────────
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

    // ── Minimal config helper ─────────────────────────────────────────────────

    fn minimal_gemma_config() -> GemmaConfig {
        GemmaConfig {
            vocab_size: 512,
            hidden_size: 64, // 8 heads * 8 head_dim
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 8, // hidden_size(64) == num_attention_heads(8) * head_dim(8)
            hidden_act: "gelu".to_string(),
            max_position_embeddings: 128,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
            pad_token_id: Some(0),
            bos_token_id: 2,
            eos_token_id: 1,
            rope_theta: 10000.0,
            attention_bias: false,
            attention_dropout: 0.0,
            model_type: "gemma".to_string(),
        }
    }

    // ── Default config tests ──────────────────────────────────────────────────

    #[test]
    fn test_gemma_default_config_is_valid() {
        let config = GemmaConfig::default();
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gemma_default_config_params() {
        let config = GemmaConfig::default();
        assert_eq!(config.vocab_size, 256000);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 18);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_key_value_heads, 1);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.model_type, "gemma");
        drop(config);
        std::hint::black_box(());
    }

    // ── Preset configs ────────────────────────────────────────────────────────

    #[test]
    fn test_gemma_2b_config() {
        let config = GemmaConfig::gemma_2b();
        assert_eq!(config.vocab_size, 256000);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 18);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_key_value_heads, 1);
        assert_eq!(config.head_dim, 256);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gemma_7b_config() {
        let config = GemmaConfig::gemma_7b();
        assert_eq!(config.vocab_size, 256000);
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 16);
        assert_eq!(config.head_dim, 256);
        // Note: hidden_size (3072) != num_attention_heads * head_dim (16*256=4096)
        // This is a known configuration quirk; validation reflects the strict rule.
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gemma_code_2b_config() {
        let config = GemmaConfig::gemma_code_2b();
        assert_eq!(config.model_type, "gemma-code");
        assert_eq!(config.hidden_size, 2048);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gemma_code_7b_config() {
        let config = GemmaConfig::gemma_code_7b();
        assert_eq!(config.model_type, "gemma-code");
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_attention_heads, 16);
        // Not asserting validate().is_ok() - same quirk as gemma_7b preset
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gemma_2b_instruct_same_as_2b() {
        let base = GemmaConfig::gemma_2b();
        let instruct = GemmaConfig::gemma_2b_instruct();
        assert_eq!(base.hidden_size, instruct.hidden_size);
        assert_eq!(base.num_hidden_layers, instruct.num_hidden_layers);
        assert_eq!(base.num_attention_heads, instruct.num_attention_heads);
    }

    #[test]
    fn test_gemma_7b_instruct_same_as_7b() {
        let base = GemmaConfig::gemma_7b();
        let instruct = GemmaConfig::gemma_7b_instruct();
        assert_eq!(base.hidden_size, instruct.hidden_size);
        assert_eq!(base.num_hidden_layers, instruct.num_hidden_layers);
    }

    // ── Architecture string ───────────────────────────────────────────────────

    #[test]
    fn test_gemma_architecture_string() {
        let config = GemmaConfig::default();
        assert_eq!(config.architecture(), "Gemma");
    }

    // ── Validation failure tests ──────────────────────────────────────────────

    #[test]
    fn test_gemma_invalid_hidden_size_mismatch() {
        // hidden_size must equal num_attention_heads * head_dim
        let mut config = minimal_gemma_config();
        config.hidden_size = 65; // 8 * 8 = 64, so 65 is invalid
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gemma_invalid_heads_not_divisible_by_kv_heads() {
        let mut config = minimal_gemma_config();
        config.num_key_value_heads = 3; // 8 not divisible by 3
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gemma_invalid_vocab_size_zero() {
        let mut config = minimal_gemma_config();
        config.vocab_size = 0;
        assert!(config.validate().is_err());
    }

    // ── Helper methods ────────────────────────────────────────────────────────

    #[test]
    fn test_gemma_num_query_groups_2b() {
        let config = GemmaConfig::gemma_2b();
        // 8 attention heads / 1 kv head = 8 query groups
        assert_eq!(config.num_query_groups(), 8);
    }

    #[test]
    fn test_gemma_num_query_groups_7b() {
        let config = GemmaConfig::gemma_7b();
        // 16 attention heads / 16 kv heads = 1 query group (MHA)
        assert_eq!(config.num_query_groups(), 1);
    }

    #[test]
    fn test_gemma_uses_multi_query_attention_2b() {
        let config = GemmaConfig::gemma_2b();
        // 1 kv head < 8 attention heads => multi-query
        assert!(config.uses_multi_query_attention());
    }

    #[test]
    fn test_gemma_uses_multi_query_attention_7b_false() {
        let config = GemmaConfig::gemma_7b();
        // 16 kv heads == 16 attention heads => NOT multi-query
        assert!(!config.uses_multi_query_attention());
    }

    #[test]
    fn test_gemma_effective_head_dim() {
        let config = GemmaConfig::gemma_2b();
        assert_eq!(config.effective_head_dim(), 256);
    }

    #[test]
    fn test_gemma_hidden_size_consistency_2b() {
        let config = GemmaConfig::gemma_2b();
        // hidden_size == num_attention_heads * head_dim
        assert_eq!(
            config.hidden_size,
            config.num_attention_heads * config.head_dim
        );
    }

    #[test]
    fn test_gemma_2b_hidden_size_consistency() {
        let config = GemmaConfig::gemma_2b();
        // hidden_size == num_attention_heads * head_dim
        assert_eq!(
            config.hidden_size,
            config.num_attention_heads * config.head_dim
        );
    }

    // ── RMSNorm creation ──────────────────────────────────────────────────────

    #[test]
    fn test_gemma_rmsnorm_creation() {
        let norm = GemmaRMSNorm::new(64, 1e-6);
        assert!(norm.is_ok());
        if let Ok(n) = norm {
            assert_eq!(n.parameter_count(), 64);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_gemma_rmsnorm_parameter_count() {
        let dim = 128usize;
        let norm_result = GemmaRMSNorm::new(dim, 1e-6);
        if let Ok(norm) = norm_result {
            assert_eq!(norm.parameter_count(), dim);
        }
        std::hint::black_box(());
    }

    // ── Model creation tests ──────────────────────────────────────────────────

    #[test]
    fn test_gemma_model_creation_minimal() {
        let config = minimal_gemma_config();
        let model = GemmaModel::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_gemma_for_causal_lm_creation_minimal() {
        let config = minimal_gemma_config();
        let model = GemmaForCausalLM::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    // ── LCG range and reproducibility ────────────────────────────────────────

    #[test]
    fn test_lcg_reproducibility() {
        let mut rng1 = Lcg::new(9876);
        let mut rng2 = Lcg::new(9876);
        for _ in 0..30 {
            assert_eq!(rng1.next_f32(), rng2.next_f32());
        }
    }

    #[test]
    fn test_lcg_range() {
        let mut rng = Lcg::new(0xCAFE_BABE);
        for _ in 0..200 {
            let v = rng.next_f32();
            assert!(v >= 0.0);
            assert!(v < 1.0);
        }
    }

    // ── Misc config properties ────────────────────────────────────────────────

    #[test]
    fn test_gemma_token_ids() {
        let config = GemmaConfig::default();
        assert_eq!(config.eos_token_id, 1);
        assert_eq!(config.bos_token_id, 2);
        assert_eq!(config.pad_token_id, Some(0));
    }

    #[test]
    fn test_gemma_rope_theta() {
        let config = GemmaConfig::default();
        assert!((config.rope_theta - 10000.0).abs() < 1e-3);
    }

    #[test]
    fn test_gemma_config_clone() {
        let config = minimal_gemma_config();
        let cloned = config.clone();
        assert_eq!(config.vocab_size, cloned.vocab_size);
        assert_eq!(config.hidden_size, cloned.hidden_size);
        assert_eq!(config.head_dim, cloned.head_dim);
        drop(config);
        drop(cloned);
        std::hint::black_box(());
    }

    // ── Real attention regression tests ───────────────────────────────────────
    //
    // These exercise the RoPE + scaled dot-product attention implementation
    // that replaced the previous no-op RoPE / `o_proj(scale*Q + V)` fake
    // attention. Every test here would have FAILED against that old code.

    fn lcg_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Lcg::new(seed);
        (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
    }

    /// RoPE at position 0 must be the identity rotation (angle = 0).
    #[test]
    fn test_gemma_rope_position_zero_is_identity() {
        let rope = GemmaRotaryEmbedding::new(4, 32, 10000.0);
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let q = Tensor::from_vec(data.clone(), &[1, 4]).expect("tensor");
        let k = q.clone();
        let (q_out, _) = rope.apply_rotary_emb(&q, &k, &[0]).expect("rope");
        let out = match q_out {
            Tensor::F32(arr) => arr.as_slice().expect("contiguous").to_vec(),
            _ => panic!("expected F32"),
        };
        for (a, b) in data.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "position 0 must be identity: {a} vs {b}"
            );
        }
    }

    /// RoPE must actually rotate: two different positions on the same input
    /// vector must produce different outputs (this fails against a no-op
    /// RoPE that just clones q/k unchanged).
    #[test]
    fn test_gemma_rope_different_positions_differ() {
        let rope = GemmaRotaryEmbedding::new(4, 32, 10000.0);
        let data = vec![1.0f32, 1.0, 1.0, 1.0];
        let q = Tensor::from_vec(data, &[1, 4]).expect("tensor");
        let k = q.clone();

        let (q_pos0, _) = rope.apply_rotary_emb(&q, &k, &[0]).expect("rope pos0");
        let (q_pos5, _) = rope.apply_rotary_emb(&q, &k, &[5]).expect("rope pos5");

        let out0 = match q_pos0 {
            Tensor::F32(arr) => arr.as_slice().expect("contiguous").to_vec(),
            _ => panic!("expected F32"),
        };
        let out5 = match q_pos5 {
            Tensor::F32(arr) => arr.as_slice().expect("contiguous").to_vec(),
            _ => panic!("expected F32"),
        };
        let differs = out0.iter().zip(out5.iter()).any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            differs,
            "RoPE must rotate differently at different positions"
        );
    }

    /// A multi-head tensor's SECOND head must also be rotated, not just the
    /// first `head_dim` channels of the row. This is exactly the bug class
    /// where only "head 0" receives positional information.
    #[test]
    fn test_gemma_rope_rotates_every_head_not_just_the_first() {
        let head_dim = 4;
        let rope = GemmaRotaryEmbedding::new(head_dim, 32, 10000.0);
        // seq_len=1, num_heads=2 => row width = 8
        let data = vec![1.0f32; 8];
        let q = Tensor::from_vec(data.clone(), &[1, 8]).expect("tensor");
        let k = q.clone();
        let (q_out, _) = rope.apply_rotary_emb(&q, &k, &[7]).expect("rope");
        let out = match q_out {
            Tensor::F32(arr) => arr.as_slice().expect("contiguous").to_vec(),
            _ => panic!("expected F32"),
        };
        let head0_changed = out[0..4].iter().zip(&data[0..4]).any(|(a, b)| (a - b).abs() > 1e-4);
        let head1_changed = out[4..8].iter().zip(&data[4..8]).any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            head0_changed,
            "head 0 must be rotated at a non-zero position"
        );
        assert!(
            head1_changed,
            "head 1 must ALSO be rotated at a non-zero position, not left untouched"
        );
    }

    fn attn_config() -> GemmaConfig {
        minimal_gemma_config()
    }

    /// Output shape must be `[seq_len, hidden_size]`.
    #[test]
    fn test_gemma_attention_output_shape() {
        let config = attn_config();
        let attn = GemmaAttention::new(&config).expect("attention");
        let seq_len = 4;
        let input = Tensor::from_vec(
            lcg_vec(seq_len * config.hidden_size, 1),
            &[seq_len, config.hidden_size],
        )
        .expect("tensor");
        let output = attn.forward(input).expect("forward");
        assert_eq!(output.shape(), vec![seq_len, config.hidden_size]);
    }

    /// Changing an EARLY token must change a LATER position's output. This
    /// is the discriminating test that only passes for real QK^T/softmax/V
    /// attention: the old `o_proj(scale*Q + V)` fake path is purely
    /// position-local and would leave row 3 unaffected by a change at row 0.
    #[test]
    fn test_gemma_attention_early_token_change_propagates_forward() {
        let config = attn_config();
        let attn = GemmaAttention::new(&config).expect("attention");
        let seq_len = 4;
        let hidden = config.hidden_size;

        let base = lcg_vec(seq_len * hidden, 2);
        let mut modified = base.clone();
        for x in modified[0..hidden].iter_mut() {
            *x += 5.0; // perturb only token 0
        }

        let out_base = attn
            .forward(Tensor::from_vec(base, &[seq_len, hidden]).expect("tensor"))
            .expect("forward base");
        let out_mod = attn
            .forward(Tensor::from_vec(modified, &[seq_len, hidden]).expect("tensor"))
            .expect("forward modified");

        let (base_data, mod_data) = match (&out_base, &out_mod) {
            (Tensor::F32(a), Tensor::F32(b)) => (
                a.as_slice().expect("contiguous").to_vec(),
                b.as_slice().expect("contiguous").to_vec(),
            ),
            _ => panic!("expected F32 outputs"),
        };

        // Row 3 (last token) attends over rows 0..=3, so it must change when
        // row 0's input changes.
        let last_row = &base_data[3 * hidden..4 * hidden];
        let last_row_mod = &mod_data[3 * hidden..4 * hidden];
        let differs = last_row.iter().zip(last_row_mod.iter()).any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(
            differs,
            "changing token 0 must change token 3's attention output for real attention"
        );
    }

    /// Causal masking: changing the LAST token must NOT change any earlier
    /// position's output (the fake position-local path also happens to pass
    /// this one, so it is only meaningful together with the test above).
    #[test]
    fn test_gemma_attention_causal_mask_future_does_not_leak_backward() {
        let config = attn_config();
        let attn = GemmaAttention::new(&config).expect("attention");
        let seq_len = 4;
        let hidden = config.hidden_size;

        let base = lcg_vec(seq_len * hidden, 3);
        let mut modified = base.clone();
        for x in modified[3 * hidden..4 * hidden].iter_mut() {
            *x += 5.0; // perturb only the LAST token
        }

        let out_base = attn
            .forward(Tensor::from_vec(base, &[seq_len, hidden]).expect("tensor"))
            .expect("forward base");
        let out_mod = attn
            .forward(Tensor::from_vec(modified, &[seq_len, hidden]).expect("tensor"))
            .expect("forward modified");

        let (base_data, mod_data) = match (&out_base, &out_mod) {
            (Tensor::F32(a), Tensor::F32(b)) => (
                a.as_slice().expect("contiguous").to_vec(),
                b.as_slice().expect("contiguous").to_vec(),
            ),
            _ => panic!("expected F32 outputs"),
        };

        for row in 0..3 {
            let a = &base_data[row * hidden..(row + 1) * hidden];
            let b = &mod_data[row * hidden..(row + 1) * hidden];
            for (x, y) in a.iter().zip(b.iter()) {
                assert!(
                    (x - y).abs() < 1e-6,
                    "row {row} must be unaffected by a change to a later token (causal masking)"
                );
            }
        }
    }

    /// Causal property with a growing sequence: forwarding a prefix of
    /// length N and then forwarding that same prefix plus one appended
    /// token must leave rows `0..N` of the output bit-identical. This is
    /// the variable-length analogue of "appending a token to the KV cache
    /// does not change earlier positions' outputs" for this crate's
    /// stateless `Layer::forward` (there is no incremental KV cache in the
    /// `Layer` API; `seq_len` and `position_ids` are recomputed fresh from
    /// the input on every call). Stronger than the fixed-length "modify a
    /// token" tests above: it also catches a mask or RoPE angle that
    /// leaked total `seq_len` instead of depending only on each row's own
    /// position. `attn_config`'s GQA grouping (8 heads / 2 kv heads) is
    /// exercised here too since every head, including the repeated KV
    /// heads, must respect the same per-row positional indexing.
    #[test]
    fn test_gemma_attention_prefix_extension_preserves_earlier_outputs() {
        let config = attn_config();
        let attn = GemmaAttention::new(&config).expect("attention");
        let hidden = config.hidden_size;
        let prefix_len = 3;

        let prefix = lcg_vec(prefix_len * hidden, 41);
        let mut extended = prefix.clone();
        extended.extend(lcg_vec(hidden, 42));

        let out_prefix = attn
            .forward(Tensor::from_vec(prefix, &[prefix_len, hidden]).expect("tensor"))
            .expect("forward prefix");
        let out_extended = attn
            .forward(Tensor::from_vec(extended, &[prefix_len + 1, hidden]).expect("tensor"))
            .expect("forward extended");

        let (a, b) = match (&out_prefix, &out_extended) {
            (Tensor::F32(x), Tensor::F32(y)) => (
                x.as_slice().expect("contiguous").to_vec(),
                y.as_slice().expect("contiguous").to_vec(),
            ),
            _ => panic!("expected F32 outputs"),
        };
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
