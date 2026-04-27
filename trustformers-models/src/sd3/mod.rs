mod config;
mod model;
mod tasks;

pub use config::{Sd3Config, Sd3ConfigError};
pub use model::{
    ClipTextEncoder, Sd3Error, Sd3TextEmbeddings, Sd3TextEncoderPipeline, T5Attention, T5Encoder,
    T5EncoderLayer, T5FeedForward, T5RelativePositionBias,
};
pub use tasks::{Sd3TaskError, Sd3TextEncoder};

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: T5 relative position bucket — exact region (small distances)
    #[test]
    fn test_relative_position_bucket_exact_region() {
        let num_buckets = 32;
        let max_distance = 128;

        // Bidirectional mode:
        //   n = -relative_position
        //   half_buckets = 16
        //   if n < 0: ret = 16, n = -n   (rel_pos > 0 means q BEFORE k)
        //   effective_buckets = 16, max_exact = 8
        //
        // rel_pos = -3: n = 3 (not < 0), ret = 0, exact region → bucket = 3
        let bucket_q_after_k = T5RelativePositionBias::relative_position_bucket(
            -3, // query 3 steps AFTER key
            true,
            num_buckets,
            max_distance,
        );
        // n=3, is_small (3 < 8), ret=0 → bucket = 3
        assert_eq!(
            bucket_q_after_k, 3,
            "Query 3 steps after key: bucket should be 3"
        );

        // rel_pos = 3: n = -3 < 0 → ret = 16, n = 3, exact region → bucket = 16 + 3 = 19
        let bucket_q_before_k = T5RelativePositionBias::relative_position_bucket(
            3, // query 3 steps BEFORE key
            true,
            num_buckets,
            max_distance,
        );
        assert_eq!(
            bucket_q_before_k, 19,
            "Query 3 steps before key: bucket should be 19 (16 + 3)"
        );

        // Zero relative position → bucket 0
        let bucket_zero =
            T5RelativePositionBias::relative_position_bucket(0, true, num_buckets, max_distance);
        assert_eq!(
            bucket_zero, 0,
            "Zero relative position should map to bucket 0"
        );
    }

    // Test 2: T5 relative position bucket — log-spaced region (large distances)
    #[test]
    fn test_relative_position_bucket_log_region() {
        let num_buckets = 32;
        let max_distance = 128;

        // Large relative position (query far after key, rel_pos = -64):
        //   n = 64 (not < 0 since rel_pos < 0), ret = 0, effective_buckets = 16
        //   max_exact = 8, n = 64 >= 8 → log region: bucket in [8, 15]
        let bucket_far =
            T5RelativePositionBias::relative_position_bucket(-64, true, num_buckets, max_distance);
        // ret=0, log region: bucket must be >= max_exact (8) and < effective_buckets (16)
        assert!(
            bucket_far >= 8,
            "Far q-after-k position should be in log-spaced region (>= 8)"
        );
        assert!(
            bucket_far < 16,
            "Far q-after-k bucket should be < 16 (effective half)"
        );
        assert!(
            bucket_far < num_buckets,
            "Bucket must be within num_buckets range"
        );

        // Very large position should be clamped to max bucket
        let bucket_max = T5RelativePositionBias::relative_position_bucket(
            -10000,
            true,
            num_buckets,
            max_distance,
        );
        assert!(
            bucket_max < num_buckets,
            "Very far position must not exceed num_buckets"
        );
    }

    // Test 3: relative position bias shape
    #[test]
    fn test_relative_position_bias_shape() {
        let num_heads = 8;
        let num_buckets = 32;
        let max_distance = 128;
        let rpb = T5RelativePositionBias::new(num_heads, num_buckets, max_distance);

        let seq_len = 5;
        let bias = rpb.compute_bias(seq_len, true);

        // Shape: seq_len * seq_len entries, each of size num_heads
        assert_eq!(
            bias.len(),
            seq_len * seq_len,
            "Bias should have seq_len^2 entries"
        );
        for entry in &bias {
            assert_eq!(
                entry.len(),
                num_heads,
                "Each bias entry should have num_heads values"
            );
        }

        // Self-positions (diagonal: q == k, rel_pos=0) should all map to bucket 0
        for i in 0..seq_len {
            let diagonal_entry = &bias[i * seq_len + i];
            // All heads at bucket 0 should have the same initialization pattern
            assert_eq!(
                diagonal_entry.len(),
                num_heads,
                "Diagonal entries must have num_heads values"
            );
        }
    }

    // Test 4: T5 attention heads consistency
    #[test]
    fn test_t5_attention_num_heads() {
        let cfg = Sd3Config::default();
        // T5 has 64 heads
        assert_eq!(cfg.t5_num_heads, 64);
        // Head dim = hidden_size / num_heads = 4096 / 64 = 64
        assert_eq!(cfg.t5_head_dim(), 64);
    }

    // Test 5: config defaults
    #[test]
    fn test_config_defaults() {
        let cfg = Sd3Config::default();
        assert_eq!(cfg.t5_vocab_size, 32128);
        assert_eq!(cfg.t5_hidden_size, 4096);
        assert_eq!(cfg.t5_num_layers, 24);
        assert_eq!(cfg.t5_num_heads, 64);
        assert_eq!(cfg.t5_intermediate_size, 10240);
        assert_eq!(cfg.t5_relative_attn_buckets, 32);
        assert_eq!(cfg.t5_max_distance, 128);
        assert_eq!(cfg.clip_vocab_size, 49408);
        assert_eq!(cfg.clip_hidden_size, 768);
        assert_eq!(cfg.clip_num_layers, 12);
        assert_eq!(cfg.clip_num_heads, 12);
        assert_eq!(cfg.clip_intermediate_size, 3072);
        assert_eq!(cfg.clip_g_hidden_size, 1280);
        assert_eq!(cfg.clip_g_num_layers, 32);
        assert_eq!(cfg.clip_g_num_heads, 20);
        assert_eq!(cfg.text_embedding_dim, 4096);
        assert_eq!(cfg.pooled_embedding_dim, 2048);
        assert_eq!(cfg.max_sequence_length, 77);
        assert_eq!(cfg.max_t5_sequence_length, 256);
    }

    // Test 6: pooled embedding dim = CLIP-L + CLIP-G (768 + 1280 = 2048)
    #[test]
    fn test_pooled_embedding_dim() {
        let cfg = Sd3Config::default();
        let expected = cfg.clip_hidden_size + cfg.clip_g_hidden_size;
        assert_eq!(expected, 2048);
        assert_eq!(cfg.pooled_embedding_dim, expected);
    }

    // Test 7: Sd3TextEmbeddings struct fields and dimensions
    #[test]
    fn test_text_embeddings_struct() {
        let t5_hidden = 4096usize;
        let max_t5_seq = 16usize; // small for test
        let pooled_dim = 2048usize;

        let emb = Sd3TextEmbeddings {
            t5_embeddings: vec![0.0f32; max_t5_seq * t5_hidden],
            pooled_embeddings: vec![0.0f32; pooled_dim],
            seq_len: 10,
        };

        assert_eq!(emb.t5_embeddings.len(), max_t5_seq * t5_hidden);
        assert_eq!(emb.pooled_embeddings.len(), pooled_dim);
        assert_eq!(emb.seq_len, 10);
        assert_eq!(emb.t5_embedding_dim(max_t5_seq), t5_hidden);
        assert_eq!(emb.pooled_dim(), pooled_dim);
    }

    // Test 8: CLIP-L vs T5 parameter differences
    #[test]
    fn test_clip_vs_t5_params() {
        let cfg = Sd3Config::default();
        // T5 has much larger hidden size
        assert!(cfg.t5_hidden_size > cfg.clip_hidden_size);
        assert!(cfg.t5_hidden_size > cfg.clip_g_hidden_size);
        // T5 has more layers than CLIP-L (though fewer than CLIP-G)
        assert!(cfg.t5_num_layers > cfg.clip_num_layers);
        // T5 has far more heads
        assert!(cfg.t5_num_heads > cfg.clip_num_heads);
        // validate should pass for default config
        assert!(cfg.validate().is_ok());
    }
}
