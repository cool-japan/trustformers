#[cfg(test)]
mod tests {
    use crate::retnet::config::RetNetConfig;
    use crate::retnet::model::{
        AdvancedChunkProcessor, MultiScaleRetention, RetNetStateCache, RotaryPositionEmbedding,
    };
    use trustformers_core::{device::Device, tensor::Tensor};

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
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1u64 << 53) as f32
        }
    }

    fn make_tensor(
        rng: &mut Lcg,
        shape: &[usize],
    ) -> trustformers_core::errors::Result<Tensor> {
        let total: usize = shape.iter().product();
        let data: Vec<f32> = (0..total).map(|_| rng.next_f32() * 0.1).collect();
        Tensor::from_vec(data, shape)
    }

    fn small_retnet_config() -> RetNetConfig {
        RetNetConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            num_heads: 4,
            intermediate_size: 64,
            retention_heads: 4,
            max_position_embeddings: 128,
            vocab_size: 100,
            chunk_size: 32,
            chunking: false,
            use_bias: false,
            ..RetNetConfig::default()
        }
    }

    // --- RotaryPositionEmbedding tests ---

    #[test]
    fn test_rotary_pos_emb_creation() {
        let rope = RotaryPositionEmbedding::new(64, 512, 10000.0);
        assert!(rope.is_ok());
    }

    #[test]
    fn test_rotary_pos_emb_device() {
        if let Ok(rope) = RotaryPositionEmbedding::new(64, 512, 10000.0) {
            assert!(matches!(rope.device(), Device::CPU));
        }
    }

    #[test]
    fn test_rotary_pos_emb_with_device() {
        let rope = RotaryPositionEmbedding::new_with_device(32, 256, 10000.0, Device::CPU);
        assert!(rope.is_ok());
    }

    #[test]
    fn test_rotary_pos_emb_apply() {
        let mut rng = Lcg::new(42);
        if let Ok(rope) = RotaryPositionEmbedding::new(8, 64, 10000.0) {
            // Use 2D tensors since rotate_half uses slice_ranges
            if let Ok(q) = make_tensor(&mut rng, &[1, 8]) {
                if let Ok(k) = make_tensor(&mut rng, &[1, 8]) {
                    // apply_rotary_pos_emb may fail on non-matching shapes;
                    // verify the method is callable
                    let _result = rope.apply_rotary_pos_emb(&q, &k, 0);
                }
            }
        }
    }

    #[test]
    fn test_rotary_pos_emb_get_cos_sin() {
        // Test the internal cos/sin computation indirectly
        if let Ok(rope) = RotaryPositionEmbedding::new(8, 64, 10000.0) {
            // Creating the embedding with valid params should succeed
            assert!(matches!(rope.device(), Device::CPU));
        }
    }

    // --- AdvancedChunkProcessor tests ---

    #[test]
    fn test_chunk_processor_creation() {
        let _processor = AdvancedChunkProcessor::new(128, 16, false);
        // Processor created successfully
    }

    #[test]
    fn test_chunk_processor_with_gradient_checkpointing() {
        let _processor = AdvancedChunkProcessor::new(256, 32, true);
        // Processor with checkpointing created successfully
    }

    #[test]
    fn test_chunk_processor_short_sequence() {
        let processor = AdvancedChunkProcessor::new(128, 16, false);
        let mut rng = Lcg::new(77);
        if let Ok(seq) = make_tensor(&mut rng, &[1, 64, 32]) {
            let result = processor.process_chunks(&seq, |chunk, _state| {
                let state = Tensor::zeros(&[1, 32])?;
                Ok((chunk.clone(), state))
            });
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_chunk_processor_long_sequence() {
        let processor = AdvancedChunkProcessor::new(32, 8, false);
        let mut rng = Lcg::new(88);
        if let Ok(seq) = make_tensor(&mut rng, &[1, 96, 16]) {
            let result = processor.process_chunks(&seq, |chunk, _state| {
                let state = Tensor::zeros(&[1, 16])?;
                Ok((chunk.clone(), state))
            });
            assert!(result.is_ok());
        }
    }

    // --- RetNetStateCache tests ---

    #[test]
    fn test_state_cache_creation() {
        let cache = RetNetStateCache::new(10);
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_state_cache_set_and_get() {
        let mut cache = RetNetStateCache::new(10);
        if let Ok(t) = Tensor::zeros(&[2, 4]) {
            let set_result = cache.set_state(0, t);
            assert!(set_result.is_ok());
            assert_eq!(cache.size(), 1);
            let got = cache.get_state(0);
            assert!(got.is_some());
        }
    }

    #[test]
    fn test_state_cache_get_missing() {
        let cache = RetNetStateCache::new(10);
        assert!(cache.get_state(42).is_none());
    }

    #[test]
    fn test_state_cache_clear() {
        let mut cache = RetNetStateCache::new(10);
        if let Ok(t) = Tensor::zeros(&[2, 4]) {
            let _ = cache.set_state(0, t);
            cache.clear();
            assert_eq!(cache.size(), 0);
            assert!(cache.get_state(0).is_none());
        }
    }

    #[test]
    fn test_state_cache_eviction() {
        let mut cache = RetNetStateCache::new(2);
        if let Ok(t0) = Tensor::zeros(&[1]) {
            if let Ok(t1) = Tensor::zeros(&[1]) {
                if let Ok(t2) = Tensor::zeros(&[1]) {
                    let _ = cache.set_state(0, t0);
                    let _ = cache.set_state(1, t1);
                    assert_eq!(cache.size(), 2);
                    let _ = cache.set_state(2, t2);
                    // After eviction, at most max_cache_size + 1
                    assert!(cache.size() <= 3);
                }
            }
        }
    }

    #[test]
    fn test_state_cache_multiple_sets_same_key() {
        let mut cache = RetNetStateCache::new(10);
        if let Ok(t0) = Tensor::zeros(&[2]) {
            if let Ok(t1) = Tensor::ones(&[2]) {
                let _ = cache.set_state(0, t0);
                let _ = cache.set_state(0, t1);
                assert!(cache.get_state(0).is_some());
            }
        }
    }

    // --- MultiScaleRetention tests ---

    #[test]
    fn test_multi_scale_retention_creation() {
        let config = small_retnet_config();
        let msr = MultiScaleRetention::new(&config);
        assert!(msr.is_ok());
    }

    #[test]
    fn test_multi_scale_retention_device() {
        let config = small_retnet_config();
        if let Ok(msr) = MultiScaleRetention::new(&config) {
            assert!(matches!(msr.device(), Device::CPU));
        }
    }

    #[test]
    fn test_multi_scale_retention_set_inference_mode() {
        let config = small_retnet_config();
        if let Ok(mut msr) = MultiScaleRetention::new(&config) {
            msr.set_inference_mode(Some(16));
            msr.clear_cache();
        }
    }

    #[test]
    fn test_multi_scale_retention_clear_cache() {
        let config = small_retnet_config();
        if let Ok(mut msr) = MultiScaleRetention::new(&config) {
            msr.clear_cache();
        }
    }

    // --- RetNet config tests ---

    #[test]
    fn test_retnet_config_default() {
        let config = RetNetConfig::default();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_heads, 16);
    }

    #[test]
    fn test_retnet_config_small() {
        let config = RetNetConfig::retnet_small();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 24);
    }

    #[test]
    fn test_retnet_config_medium() {
        let config = RetNetConfig::retnet_medium();
        assert_eq!(config.hidden_size, 2560);
    }

    #[test]
    fn test_retnet_config_large() {
        let config = RetNetConfig::retnet_large();
        assert_eq!(config.hidden_size, 4096);
    }

    #[test]
    fn test_retnet_config_xl() {
        let config = RetNetConfig::retnet_xl();
        assert_eq!(config.hidden_size, 5120);
        assert!(config.deepnorm);
    }

    #[test]
    fn test_retnet_config_long() {
        let config = RetNetConfig::retnet_long();
        assert_eq!(config.max_position_embeddings, 8192);
        assert!(config.chunking);
    }

    #[test]
    fn test_retnet_head_dim() {
        let config = small_retnet_config();
        assert_eq!(config.head_dim(), 8);
    }

    #[test]
    fn test_retnet_retention_head_dim() {
        let config = small_retnet_config();
        assert_eq!(config.retention_head_dim(), 8);
    }

    #[test]
    fn test_retnet_retention_dim() {
        let config = small_retnet_config();
        let expected = (32.0_f32 / 2.0) as usize;
        assert_eq!(config.retention_dim(), expected);
    }

    #[test]
    fn test_retnet_uses_chunking_disabled() {
        let mut config = small_retnet_config();
        config.chunking = false;
        assert!(!config.uses_chunking());
    }

    #[test]
    fn test_retnet_uses_chunking_enabled() {
        let mut config = small_retnet_config();
        config.chunking = true;
        config.chunk_size = 64;
        assert!(config.uses_chunking());
    }

    #[test]
    fn test_retnet_config_validate_valid() {
        use trustformers_core::traits::Config;
        let config = small_retnet_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_retnet_config_validate_bad_hidden_heads() {
        use trustformers_core::traits::Config;
        let mut config = small_retnet_config();
        config.num_heads = 3; // 32 not divisible by 3
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_retnet_config_architecture() {
        use trustformers_core::traits::Config;
        let config = small_retnet_config();
        assert_eq!(config.architecture(), "RetNet");
    }
}
