#[cfg(test)]
mod tests {
    use crate::cross_attention::config::*;
    use crate::cross_attention::layers::*;
    use crate::cross_attention::utils::{create_attention_mask, MaskType};
    use trustformers_core::tensor::Tensor;

    // --- CrossAttentionConfig tests ---

    #[test]
    fn test_config_default() {
        let config = CrossAttentionConfig::default();
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_heads, 8);
        assert!(config.bias);
    }

    #[test]
    fn test_config_standard() {
        let config = CrossAttentionConfig::standard(256, 4);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_heads, 4);
    }

    #[test]
    fn test_config_sparse() {
        let config = CrossAttentionConfig::sparse(256, 4, 0.5);
        assert!(config.sparse_config.is_some());
        if let Some(sc) = &config.sparse_config {
            assert!((sc.sparsity_ratio - 0.5).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_config_get_head_dim() {
        let config = CrossAttentionConfig::default();
        assert_eq!(config.get_head_dim(), 64); // 512 / 8
    }

    #[test]
    fn test_config_get_head_dim_custom() {
        let config = CrossAttentionConfig {
            head_dim: Some(32),
            ..CrossAttentionConfig::default()
        };
        assert_eq!(config.get_head_dim(), 32);
    }

    #[test]
    fn test_config_get_scale() {
        let config = CrossAttentionConfig::default();
        let expected = 1.0 / (64.0_f32).sqrt();
        assert!((config.get_scale() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_config_validate_valid() {
        let config = CrossAttentionConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_hidden_not_divisible() {
        let config = CrossAttentionConfig {
            hidden_size: 100,
            num_heads: 3,
            ..CrossAttentionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_dropout() {
        let config = CrossAttentionConfig {
            attention_dropout: 1.5,
            ..CrossAttentionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    // --- CrossAttention creation tests ---

    #[test]
    fn test_cross_attention_creation() {
        let config = CrossAttentionConfig::standard(64, 4);
        let attn = CrossAttention::new(config);
        assert!(attn.is_ok());
    }

    #[test]
    fn test_multi_head_cross_attention_creation() {
        let config = CrossAttentionConfig::standard(64, 4);
        let attn = MultiHeadCrossAttention::new(config);
        assert!(attn.is_ok());
    }

    #[test]
    fn test_sparse_cross_attention_creation() {
        let config = CrossAttentionConfig::sparse(64, 4, 0.3);
        let attn = SparseCrossAttention::new(config);
        assert!(attn.is_ok());
    }

    #[test]
    fn test_hierarchical_cross_attention_creation() {
        let config = CrossAttentionConfig {
            hidden_size: 64,
            num_heads: 4,
            hierarchical_config: Some(HierarchicalAttentionConfig::default()),
            ..CrossAttentionConfig::default()
        };
        let attn = HierarchicalCrossAttention::new(config);
        assert!(attn.is_ok());
    }

    /// Wiring/plumbing regression test, **not** a numeric proof that pooling is mathematically
    /// correct -- that proof is the 18 closed-form tests in
    /// `cross_attention::utils::pooling_and_interpolation_tests` (they exercise
    /// `average_pool_1d`/`max_pool_1d`/`linear_interpolate`/`nearest_interpolate` directly against
    /// hand-computed values). This test cannot distinguish "pooling genuinely shrank the
    /// key/value sequence axis at every level" from "pooling silently stayed a no-op": cross
    /// attention never requires `query_len == key_len`, so `HierarchicalCrossAttention`'s
    /// `WeightedSum`-aggregated output shape is `[batch, query_len, hidden]` regardless of what
    /// key_len each level actually attended over, and `Linear`'s randomly-initialized weights
    /// rule out a closed-form check on the output values here. What this test *does* prove: now
    /// that `pool_tensor` genuinely changes `current_key`/`current_value`'s shape between
    /// levels (previously it could not, since pooling was a no-op), the multi-level forward pass
    /// -- which calls `pool_tensor` `num_levels - 1` times and re-feeds the result into the next
    /// level's `MultiHeadCrossAttention::forward` -- still completes without an internal shape
    /// error and still returns the expected output shape.
    #[test]
    fn test_hierarchical_forward_with_no_mask_succeeds_and_preserves_query_shape() {
        let config = CrossAttentionConfig {
            hidden_size: 4,
            num_heads: 2,
            hierarchical_config: Some(HierarchicalAttentionConfig {
                num_levels: 3,
                pooling_factor: 2,
                learnable_pooling: false,
                aggregation_method: AggregationMethod::WeightedSum,
            }),
            bias: false,
            ..CrossAttentionConfig::default()
        };
        let attn = HierarchicalCrossAttention::new(config).expect("construction must succeed");

        // seq_len=8 pools to 4, then 2, across the 3 levels (query stays unpooled).
        let query = Tensor::zeros(&[1, 5, 4]).expect("tensor");
        let key = Tensor::zeros(&[1, 8, 4]).expect("tensor");
        let value = Tensor::zeros(&[1, 8, 4]).expect("tensor");

        let output = attn
            .forward(query, key, value, None)
            .expect("forward with no mask must succeed now that pooling genuinely shrinks the sequence axis");
        // WeightedSum aggregation keeps the per-level output shape: [batch, query_len, hidden].
        assert_eq!(output.output.shape(), vec![1, 5, 4]);
    }

    /// Regression: `MultiHeadCrossAttention::forward` used to build its per-head mask via
    /// `mask.unsqueeze(1)?.broadcast_to(&[batch, num_heads, query_len, key_len])`, which
    /// assumed a 3-D `[batch, query_len, key_len]` input mask (so the new axis lands between
    /// batch and query_len). `create_attention_mask` -- this crate's own mask constructor,
    /// right in this module -- produces a 2-D `[query_len, key_len]` mask with no batch
    /// dimension, so `unsqueeze(1)` instead landed the new axis between query_len and key_len,
    /// producing `[query_len, 1, key_len]`; broadcasting that against a 4-D target compared
    /// `query_len` against `num_heads`, which errored whenever they differed (the
    /// overwhelmingly common case, `num_heads = 2` and `query_len = 5` here).
    ///
    /// Now that `forward` rank-dispatches the mask before broadcasting, this must succeed --
    /// and the test hand-verifies the *broadcast is correct*, not merely that it doesn't error:
    /// with `bias: false` and all-zero query/key/value, every projected Q/K/V is exactly zero
    /// (a zero vector times any weight matrix is zero), so every raw attention score is exactly
    /// zero before masking. Adding the causal mask therefore leaves unmasked positions at
    /// `0.0` and masked ones at `-inf`; softmax over a row with `i + 1` zeros and the rest
    /// `-inf` is `1 / (i + 1)` on the zeros and `0.0` on the `-inf`s -- a closed-form
    /// prediction, not an approximation.
    #[test]
    fn test_multi_head_forward_with_2d_mask_broadcasts_correctly_across_mismatched_heads() {
        let config = CrossAttentionConfig {
            hidden_size: 4,
            num_heads: 2, // != query_len below, which is what exposed the defect
            bias: false,
            // `forward` hardcodes `training: true` in its `scaled_dot_product_attention`
            // call (see the "training flag - would be configurable" comment on the sibling
            // `CrossAttention::forward`), so the crate's own default `attention_dropout:
            // 0.1` would genuinely apply inverted dropout here (scaling surviving weights
            // by `1 / (1 - 0.1)`), which breaks the hand-computed "softmax row sums to
            // exactly 1" math below on more than an expected value across many samples.
            // Zero it so this test's per-call assertions are deterministic.
            attention_dropout: 0.0,
            ..CrossAttentionConfig::default()
        };
        let attn = MultiHeadCrossAttention::new(config).expect("construction must succeed");

        let query = Tensor::zeros(&[1, 5, 4]).expect("tensor");
        let key = Tensor::zeros(&[1, 5, 4]).expect("tensor");
        let value = Tensor::zeros(&[1, 5, 4]).expect("tensor");
        let mask = create_attention_mask(5, 5, MaskType::Causal).expect("mask construction");

        let output = attn.forward(query, key, value, Some(mask)).expect(
            "a 2-D mask must broadcast correctly across all heads even when num_heads != query_len",
        );

        let weights = output
            .attention_weights
            .expect("scaled_dot_product_attention always reports attention_weights");
        let (batch, heads, q_len, k_len) = (1usize, 2usize, 5usize, 5usize);
        assert_eq!(weights.shape(), vec![batch, heads, q_len, k_len]);

        let data = weights.data_f32().expect("attention weights must be readable");
        for b in 0..batch {
            for h in 0..heads {
                for i in 0..q_len {
                    let row_start = ((b * heads + h) * q_len + i) * k_len;
                    let row = &data[row_start..row_start + k_len];
                    let row_sum: f32 = row.iter().sum();
                    assert!(
                        (row_sum - 1.0).abs() < 1e-4,
                        "batch {b} head {h} query {i}: softmax row must sum to 1, got {row_sum} ({row:?})"
                    );
                    let expected_unmasked = 1.0 / (i + 1) as f32;
                    for (j, &w) in row.iter().enumerate() {
                        if j > i {
                            assert!(
                                w < 1e-6,
                                "batch {b} head {h} query {i}: position {j} > {i} is causally \
                                 masked and must carry ~0 weight, got {w}"
                            );
                        } else {
                            assert!(
                                (w - expected_unmasked).abs() < 1e-4,
                                "batch {b} head {h} query {i}: unmasked position {j} must be \
                                 uniform at {expected_unmasked}, got {w}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Regression: `HierarchicalCrossAttention::forward` used to pass the SAME
    /// caller-supplied mask to every level unchanged while `current_key`/`current_value`
    /// shrink via pooling between levels -- so by level 1 the mask's key axis (still the
    /// original, larger `key_len`) no longer matched `current_key`'s pooled (smaller) length,
    /// and `MultiHeadCrossAttention::forward` (inside each level) would fail to broadcast it.
    /// Now the mask is pooled in lockstep with K/V via `pool_mask_key_axis`, so a
    /// non-hierarchical-shaped 2-D mask must survive all `num_levels` levels.
    #[test]
    fn test_hierarchical_forward_with_2d_mask_pools_the_mask_key_axis_per_level() {
        let config = CrossAttentionConfig {
            hidden_size: 4,
            num_heads: 2,
            hierarchical_config: Some(HierarchicalAttentionConfig {
                num_levels: 3,
                pooling_factor: 2,
                learnable_pooling: false,
                aggregation_method: AggregationMethod::WeightedSum,
            }),
            bias: false,
            ..CrossAttentionConfig::default()
        };
        let attn = HierarchicalCrossAttention::new(config).expect("construction must succeed");

        // Level 0 key_len=8, level 1 key_len=4, level 2 key_len=2 (pooling_factor=2 each
        // step): the mask's key axis must track every one of those or forward() errors.
        let query = Tensor::zeros(&[1, 5, 4]).expect("tensor");
        let key = Tensor::zeros(&[1, 8, 4]).expect("tensor");
        let value = Tensor::zeros(&[1, 8, 4]).expect("tensor");
        let mask = create_attention_mask(5, 8, MaskType::None).expect("mask construction");

        let output = attn
            .forward(query, key, value, Some(mask))
            .expect("a 2-D mask must be pooled per level to track the shrinking key/value axis");
        assert_eq!(output.output.shape(), vec![1, 5, 4]);
    }

    // --- Attention mask tests ---

    #[test]
    fn test_none_mask() {
        let mask = create_attention_mask(4, 4, MaskType::None);
        assert!(mask.is_ok());
        if let Ok(m) = mask {
            assert_eq!(m.shape(), &[4, 4]);
        }
    }

    #[test]
    fn test_causal_mask() {
        let mask = create_attention_mask(4, 4, MaskType::Causal);
        assert!(mask.is_ok());
        if let Ok(m) = mask {
            assert_eq!(m.shape(), &[4, 4]);
        }
    }

    #[test]
    fn test_local_mask() {
        let mask = create_attention_mask(8, 8, MaskType::Local(3));
        assert!(mask.is_ok());
        if let Ok(m) = mask {
            assert_eq!(m.shape(), &[8, 8]);
        }
    }

    // --- SparseAttentionConfig defaults ---

    #[test]
    fn test_sparse_config_default() {
        let sc = SparseAttentionConfig::default();
        assert!((sc.sparsity_ratio - 0.1).abs() < f32::EPSILON);
        assert_eq!(sc.block_size, Some(64));
    }

    #[test]
    fn test_hierarchical_config_default() {
        let hc = HierarchicalAttentionConfig::default();
        assert_eq!(hc.num_levels, 3);
        assert_eq!(hc.pooling_factor, 2);
        assert!(hc.learnable_pooling);
    }

    #[test]
    fn test_adaptive_config_default() {
        let ac = AdaptiveAttentionConfig::default();
        assert_eq!(ac.num_patterns, 4);
        assert!((ac.temperature - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gated_config_default() {
        let gc = GatedAttentionConfig::default();
        assert!(gc.gate_bias);
        assert!(!gc.separate_gates);
    }

    // --- Config validation edge cases ---

    #[test]
    fn test_config_validate_bad_sparsity() {
        let config = CrossAttentionConfig {
            sparse_config: Some(SparseAttentionConfig {
                sparsity_ratio: -0.5,
                ..SparseAttentionConfig::default()
            }),
            ..CrossAttentionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_hierarchical() {
        let config = CrossAttentionConfig {
            hierarchical_config: Some(HierarchicalAttentionConfig {
                num_levels: 0,
                ..HierarchicalAttentionConfig::default()
            }),
            ..CrossAttentionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_adaptive() {
        let config = CrossAttentionConfig {
            adaptive_config: Some(AdaptiveAttentionConfig {
                temperature: 0.0,
                ..AdaptiveAttentionConfig::default()
            }),
            ..CrossAttentionConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
