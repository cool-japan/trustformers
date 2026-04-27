/// Tests for generation cache types: KVCache and Beam
#[cfg(test)]
mod tests {
    use crate::generation::cache::{Beam, KVCache};

    // ---- KVCache tests ----

    #[test]
    fn test_kvcache_new_is_empty() {
        let cache = KVCache::new();
        assert!(cache.keys.is_empty());
        assert!(cache.values.is_empty());
        assert_eq!(cache.seq_len, 0);
    }

    #[test]
    fn test_kvcache_default_matches_new() {
        let a = KVCache::new();
        let b = KVCache::default();
        assert_eq!(a.seq_len, b.seq_len);
        assert_eq!(a.keys.len(), b.keys.len());
    }

    #[test]
    fn test_kvcache_clear_resets_all_fields() {
        let mut cache = KVCache::new();
        // Start with seq_len > 0 by manipulating directly (no append needed for clearing test)
        cache.seq_len = 5;
        cache.clear();
        assert_eq!(cache.seq_len, 0);
        assert!(cache.keys.is_empty());
        assert!(cache.values.is_empty());
    }

    #[test]
    fn test_kvcache_get_layer_out_of_bounds_returns_none() {
        let cache = KVCache::new();
        let result = cache.get_layer(0);
        assert!(result.is_none());
    }

    #[test]
    fn test_kvcache_get_layer_large_index_returns_none() {
        let cache = KVCache::new();
        assert!(cache.get_layer(999).is_none());
    }

    #[test]
    fn test_kvcache_seq_len_starts_at_zero() {
        let cache = KVCache::new();
        assert_eq!(cache.seq_len, 0);
    }

    #[test]
    fn test_kvcache_clear_idempotent() {
        let mut cache = KVCache::new();
        cache.clear();
        cache.clear();
        assert_eq!(cache.seq_len, 0);
        assert!(cache.keys.is_empty());
    }

    // ---- Beam tests ----

    #[test]
    fn test_beam_new_creates_unfinished_beam() {
        let beam = Beam::new(vec![1, 2, 3], -0.5);
        assert_eq!(beam.tokens, vec![1, 2, 3]);
        assert!((beam.score - (-0.5)).abs() < 1e-6);
        assert!(!beam.finished);
        assert!(beam.cache.is_none());
    }

    #[test]
    fn test_beam_new_empty_tokens() {
        let beam = Beam::new(vec![], 0.0);
        assert!(beam.tokens.is_empty());
        assert_eq!(beam.score, 0.0);
        assert!(!beam.finished);
    }

    #[test]
    fn test_beam_extend_appends_token_and_accumulates_score() {
        let beam = Beam::new(vec![10, 20], -1.0);
        let extended = beam.extend(30, -0.5);
        assert_eq!(extended.tokens, vec![10, 20, 30]);
        assert!((extended.score - (-1.5)).abs() < 1e-6);
        assert!(!extended.finished);
    }

    #[test]
    fn test_beam_extend_does_not_mutate_original() {
        let beam = Beam::new(vec![1], 0.0);
        let _extended = beam.extend(2, -1.0);
        assert_eq!(beam.tokens, vec![1]);
        assert_eq!(beam.score, 0.0);
    }

    #[test]
    fn test_beam_finalize_sets_finished_flag() {
        let mut beam = Beam::new(vec![5], -0.3);
        assert!(!beam.finished);
        beam.finalize();
        assert!(beam.finished);
    }

    #[test]
    fn test_beam_finalize_idempotent() {
        let mut beam = Beam::new(vec![1], 0.0);
        beam.finalize();
        beam.finalize();
        assert!(beam.finished);
    }

    #[test]
    fn test_beam_normalized_score_single_token() {
        let beam = Beam::new(vec![42], -2.0);
        let normalized = beam.get_normalized_score();
        assert!((normalized - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_beam_normalized_score_multiple_tokens() {
        let beam = Beam::new(vec![1, 2, 3, 4], -8.0);
        let normalized = beam.get_normalized_score();
        assert!((normalized - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_beam_normalized_score_empty_tokens_returns_zero() {
        let beam = Beam::new(vec![], 0.0);
        let normalized = beam.get_normalized_score();
        assert_eq!(normalized, 0.0);
    }

    #[test]
    fn test_beam_extend_chain() {
        let mut beam = Beam::new(vec![], 0.0);
        // Use LCG for deterministic "random" scores
        let mut s = 42u64;
        for i in 0..5usize {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let score = -((s % 1000) as f32) / 1000.0;
            beam = beam.extend(i, score);
        }
        assert_eq!(beam.tokens.len(), 5);
    }

    #[test]
    fn test_beam_clone_is_independent() {
        let beam = Beam::new(vec![1, 2], -0.5);
        let mut cloned = beam.clone();
        cloned.finalize();
        assert!(!beam.finished);
        assert!(cloned.finished);
    }

    #[test]
    fn test_beam_extend_preserves_cache_none() {
        let beam = Beam::new(vec![1], 0.0);
        let extended = beam.extend(2, -1.0);
        assert!(extended.cache.is_none());
    }

    #[test]
    fn test_beam_normalized_score_positive() {
        let beam = Beam::new(vec![1, 2], 4.0);
        let normalized = beam.get_normalized_score();
        assert!((normalized - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_beam_score_accumulation_zero() {
        let beam = Beam::new(vec![1], 0.0);
        let extended = beam.extend(2, 0.0);
        assert_eq!(extended.score, 0.0);
    }

    #[test]
    fn test_kvcache_multiple_clears() {
        let mut cache = KVCache::new();
        for _ in 0..10 {
            cache.clear();
        }
        assert_eq!(cache.seq_len, 0);
    }
}
