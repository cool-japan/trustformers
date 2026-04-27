use super::config::FNetConfig;
use super::model::{FNetForMaskedLM, FNetForSequenceClassification, FNetModel};
use trustformers_core::traits::{Config, Model};

// LCG random number generator (no rand dependency)
#[allow(dead_code)]
struct Lcg {
    state: u64,
}

#[allow(dead_code)]
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

// ─── Config Tests ─────────────────────────────────────────────────────────────

#[test]
fn test_fnet_config_default() {
    let config = FNetConfig::default();
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_hidden_layers, 12);
    assert_eq!(config.intermediate_size, 3072);
    assert!(config.use_fourier_transform);
    assert!(!config.use_tpu_optimized_fft);
    assert_eq!(config.fourier_transform_type, "dft");
    assert!(config.use_bias_in_fourier);
    assert_eq!(config.fourier_dropout_prob, 0.0);
}

#[test]
fn test_fnet_config_validate_ok() {
    let config = FNetConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_fnet_config_invalid_transform_type() {
    let config = FNetConfig {
        fourier_transform_type: "wavelet".to_string(),
        ..FNetConfig::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_fnet_config_max_pos_too_large() {
    let config = FNetConfig {
        max_position_embeddings: 16384,
        ..FNetConfig::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_fnet_valid_transform_types() {
    for transform_type in &["dft", "real_dft", "dct"] {
        let config = FNetConfig {
            fourier_transform_type: transform_type.to_string(),
            ..FNetConfig::default()
        };
        assert!(
            config.validate().is_ok(),
            "Transform type {} should be valid",
            transform_type
        );
    }
}

#[test]
fn test_fnet_config_base_preset() {
    let config = FNetConfig::fnet_base();
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_hidden_layers, 12);
    assert!(config.validate().is_ok());
}

#[test]
fn test_fnet_config_large_preset() {
    let config = FNetConfig::fnet_large();
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.num_hidden_layers, 24);
    assert_eq!(config.intermediate_size, 4096);
    assert!(config.validate().is_ok());
}

#[test]
fn test_fnet_config_tpu_preset() {
    let config = FNetConfig::fnet_tpu();
    assert!(config.use_tpu_optimized_fft);
    assert_eq!(config.fourier_transform_type, "real_dft");
    assert_eq!(config.max_position_embeddings, 1024);
    assert!(config.validate().is_ok());
}

#[test]
fn test_fnet_config_dct_preset() {
    let config = FNetConfig::fnet_dct();
    assert_eq!(config.fourier_transform_type, "dct");
    assert_eq!(config.max_position_embeddings, 1024);
    assert!(config.validate().is_ok());
}

#[test]
fn test_fnet_config_long_preset() {
    let config = FNetConfig::fnet_long();
    assert_eq!(config.max_position_embeddings, 4096);
    assert_eq!(config.fourier_transform_type, "real_dft");
    assert!(config.validate().is_ok());
}

#[test]
fn test_fnet_config_complexity_advantage() {
    let config = FNetConfig {
        max_position_embeddings: 512,
        ..FNetConfig::default()
    };
    let advantage = config.complexity_advantage();
    // O(n^2) / O(n log n) = n / log2(n)
    let n = 512.0_f32;
    let expected = (n * n) / (n * n.log2());
    assert!((advantage - expected).abs() < 1e-3);
}

#[test]
fn test_fnet_config_is_efficient_power_of_two() {
    let power_of_two = FNetConfig {
        max_position_embeddings: 512,
        ..FNetConfig::default()
    };
    assert!(power_of_two.is_efficient_config());

    let non_power = FNetConfig {
        max_position_embeddings: 500,
        ..FNetConfig::default()
    };
    assert!(!non_power.is_efficient_config());
}

#[test]
fn test_fnet_config_is_efficient_various() {
    for &n in &[64usize, 128, 256, 512, 1024, 2048, 4096] {
        let config = FNetConfig {
            max_position_embeddings: n,
            ..FNetConfig::default()
        };
        assert!(config.is_efficient_config(), "n={} should be power of 2", n);
    }
}

#[test]
fn test_fnet_config_recommended_batch_size() {
    let base_config = FNetConfig {
        hidden_size: 768,
        ..FNetConfig::default()
    };
    assert_eq!(base_config.recommended_batch_size(), 64);

    let large_config = FNetConfig {
        hidden_size: 1024,
        ..FNetConfig::default()
    };
    assert_eq!(large_config.recommended_batch_size(), 32);

    let other_config = FNetConfig {
        hidden_size: 512,
        ..FNetConfig::default()
    };
    assert_eq!(other_config.recommended_batch_size(), 16);
}

#[test]
fn test_fnet_config_architecture_name() {
    let config = FNetConfig::default();
    assert_eq!(config.architecture(), "FNet");
}

// ─── Model Construction Tests ─────────────────────────────────────────────────

fn tiny_fnet_config() -> FNetConfig {
    FNetConfig {
        vocab_size: 100,
        hidden_size: 32,
        num_hidden_layers: 1,
        intermediate_size: 64,
        max_position_embeddings: 64,
        hidden_dropout_prob: 0.0,
        fourier_dropout_prob: 0.0,
        use_bias_in_fourier: true,
        use_fourier_transform: true,
        use_tpu_optimized_fft: false,
        fourier_transform_type: "dft".to_string(),
        ..FNetConfig::default()
    }
}

#[test]
fn test_fnet_model_construction() {
    let config = tiny_fnet_config();
    let result = FNetModel::new(config);
    assert!(
        result.is_ok(),
        "FNetModel construction failed: {:?}",
        result.err()
    );
}

#[test]
fn test_fnet_model_no_bias_construction() {
    let config = FNetConfig {
        vocab_size: 100,
        hidden_size: 32,
        num_hidden_layers: 1,
        intermediate_size: 64,
        max_position_embeddings: 64,
        hidden_dropout_prob: 0.0,
        fourier_dropout_prob: 0.0,
        use_bias_in_fourier: false,
        ..FNetConfig::default()
    };
    let result = FNetModel::new(config);
    assert!(
        result.is_ok(),
        "FNet without Fourier bias failed: {:?}",
        result.err()
    );
}

#[test]
fn test_fnet_model_num_parameters() {
    let config = tiny_fnet_config();
    if let Ok(model) = FNetModel::new(config) {
        assert!(model.num_parameters() > 0);
    }
}

#[test]
fn test_fnet_classification_construction() {
    let config = tiny_fnet_config();
    let result = FNetForSequenceClassification::new(config, 5);
    assert!(
        result.is_ok(),
        "FNet classification failed: {:?}",
        result.err()
    );
}

#[test]
fn test_fnet_masked_lm_construction() {
    let config = tiny_fnet_config();
    let result = FNetForMaskedLM::new(config);
    assert!(result.is_ok(), "FNet MaskedLM failed: {:?}", result.err());
}

#[test]
fn test_fnet_model_device() {
    let config = tiny_fnet_config();
    if let Ok(model) = FNetModel::new(config) {
        assert_eq!(model.device(), trustformers_core::device::Device::CPU);
    }
}

#[test]
fn test_fnet_model_config_access() {
    let config = tiny_fnet_config();
    let vocab_size = config.vocab_size;
    if let Ok(model) = FNetModel::new(config) {
        assert_eq!(model.get_config().vocab_size, vocab_size);
    }
}
