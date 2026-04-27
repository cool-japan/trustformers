use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// Whisper model configuration
/// Reference: "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    // Audio encoder
    pub num_mel_bins: usize,         // 80
    pub max_source_positions: usize, // 1500 (30s / 20ms = 1500 frames)
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub d_model: usize,         // hidden size (encoder + decoder)
    pub encoder_ffn_dim: usize, // 4 * d_model typically
    // Text decoder
    pub vocab_size: usize,           // 51865 for multilingual
    pub max_target_positions: usize, // 448
    pub decoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub decoder_ffn_dim: usize,
    // Common
    pub dropout: f32,
    pub attention_dropout: f32,
    pub activation_dropout: f32,
    pub scale_embedding: bool,
    pub model_type: String,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        // Whisper small defaults
        Self {
            num_mel_bins: 80,
            max_source_positions: 1500,
            encoder_layers: 6,
            encoder_attention_heads: 8,
            d_model: 512,
            encoder_ffn_dim: 2048,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_layers: 6,
            decoder_attention_heads: 8,
            decoder_ffn_dim: 2048,
            dropout: 0.0,
            attention_dropout: 0.0,
            activation_dropout: 0.0,
            scale_embedding: false,
            model_type: "whisper".to_string(),
        }
    }
}

impl Config for WhisperConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if !self.d_model.is_multiple_of(self.encoder_attention_heads) {
            return Err(trustformers_core::errors::invalid_config(
                "d_model",
                "d_model must be divisible by encoder_attention_heads",
            ));
        }
        if !self.d_model.is_multiple_of(self.decoder_attention_heads) {
            return Err(trustformers_core::errors::invalid_config(
                "d_model",
                "d_model must be divisible by decoder_attention_heads",
            ));
        }
        if self.num_mel_bins == 0 {
            return Err(trustformers_core::errors::invalid_config(
                "num_mel_bins",
                "num_mel_bins must be greater than 0",
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Whisper"
    }
}

impl WhisperConfig {
    /// Whisper Tiny (~39M params)
    pub fn whisper_tiny() -> Self {
        Self {
            encoder_layers: 4,
            encoder_attention_heads: 6,
            d_model: 384,
            encoder_ffn_dim: 1536,
            decoder_layers: 4,
            decoder_attention_heads: 6,
            decoder_ffn_dim: 1536,
            model_type: "whisper".to_string(),
            ..Self::default()
        }
    }

    /// Whisper Base (~74M params)
    pub fn whisper_base() -> Self {
        Self {
            encoder_layers: 6,
            encoder_attention_heads: 8,
            d_model: 512,
            encoder_ffn_dim: 2048,
            decoder_layers: 6,
            decoder_attention_heads: 8,
            decoder_ffn_dim: 2048,
            model_type: "whisper".to_string(),
            ..Self::default()
        }
    }

    /// Whisper Small (~244M params)
    pub fn whisper_small() -> Self {
        Self {
            encoder_layers: 12,
            encoder_attention_heads: 12,
            d_model: 768,
            encoder_ffn_dim: 3072,
            decoder_layers: 12,
            decoder_attention_heads: 12,
            decoder_ffn_dim: 3072,
            model_type: "whisper".to_string(),
            ..Self::default()
        }
    }

    /// Whisper Medium (~769M params)
    pub fn whisper_medium() -> Self {
        Self {
            encoder_layers: 24,
            encoder_attention_heads: 16,
            d_model: 1024,
            encoder_ffn_dim: 4096,
            decoder_layers: 24,
            decoder_attention_heads: 16,
            decoder_ffn_dim: 4096,
            model_type: "whisper".to_string(),
            ..Self::default()
        }
    }

    /// Whisper Large v2 (~1550M params)
    pub fn whisper_large_v2() -> Self {
        Self {
            encoder_layers: 32,
            encoder_attention_heads: 20,
            d_model: 1280,
            encoder_ffn_dim: 5120,
            decoder_layers: 32,
            decoder_attention_heads: 20,
            decoder_ffn_dim: 5120,
            model_type: "whisper".to_string(),
            ..Self::default()
        }
    }

    /// Whisper Tiny English-only variant
    pub fn whisper_tiny_en() -> Self {
        Self {
            vocab_size: 50257, // English-only vocabulary
            ..Self::whisper_tiny()
        }
    }

    /// Head dimension for encoder attention
    pub fn encoder_head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }

    /// Head dimension for decoder attention
    pub fn decoder_head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    #[test]
    fn test_default_is_small() {
        let cfg = WhisperConfig::default();
        assert_eq!(cfg.d_model, 512);
        assert_eq!(cfg.encoder_layers, 6);
        assert_eq!(cfg.decoder_layers, 6);
        assert_eq!(cfg.num_mel_bins, 80);
        assert_eq!(cfg.vocab_size, 51865);
    }

    #[test]
    fn test_tiny_preset() {
        let cfg = WhisperConfig::whisper_tiny();
        assert_eq!(cfg.d_model, 384);
        assert_eq!(cfg.encoder_layers, 4);
        assert_eq!(cfg.decoder_layers, 4);
        assert_eq!(cfg.encoder_attention_heads, 6);
    }

    #[test]
    fn test_medium_preset() {
        let cfg = WhisperConfig::whisper_medium();
        assert_eq!(cfg.d_model, 1024);
        assert_eq!(cfg.encoder_layers, 24);
        assert_eq!(cfg.encoder_ffn_dim, 4096);
    }

    #[test]
    fn test_large_v2_preset() {
        let cfg = WhisperConfig::whisper_large_v2();
        assert_eq!(cfg.d_model, 1280);
        assert_eq!(cfg.encoder_layers, 32);
        assert_eq!(cfg.encoder_ffn_dim, 5120);
    }

    #[test]
    fn test_tiny_en_vocab_size() {
        assert_eq!(WhisperConfig::whisper_tiny_en().vocab_size, 50257);
    }

    #[test]
    fn test_encoder_head_dim_tiny() {
        // 384 / 6 = 64
        assert_eq!(WhisperConfig::whisper_tiny().encoder_head_dim(), 64);
    }

    #[test]
    fn test_decoder_head_dim_small() {
        // 512 / 8 = 64
        assert_eq!(WhisperConfig::default().decoder_head_dim(), 64);
    }

    #[test]
    fn test_head_dims_all_equal_64() {
        for cfg in [
            WhisperConfig::whisper_tiny(),
            WhisperConfig::whisper_base(),
            WhisperConfig::whisper_small(),
            WhisperConfig::whisper_medium(),
            WhisperConfig::whisper_large_v2(),
        ] {
            assert_eq!(cfg.encoder_head_dim(), 64);
            assert_eq!(cfg.decoder_head_dim(), 64);
        }
    }

    #[test]
    fn test_architecture_label() {
        assert_eq!(WhisperConfig::default().architecture(), "Whisper");
    }

    #[test]
    fn test_model_type_is_whisper() {
        assert_eq!(WhisperConfig::default().model_type, "whisper");
    }

    #[test]
    fn test_num_mel_bins_is_80() {
        assert_eq!(WhisperConfig::default().num_mel_bins, 80);
    }

    #[test]
    fn test_max_source_positions() {
        assert_eq!(WhisperConfig::default().max_source_positions, 1500);
    }

    #[test]
    fn test_dropout_zero_by_default() {
        let cfg = WhisperConfig::default();
        assert_eq!(cfg.dropout, 0.0);
        assert_eq!(cfg.attention_dropout, 0.0);
        assert_eq!(cfg.activation_dropout, 0.0);
    }

    #[test]
    fn test_validate_default_ok() {
        assert!(WhisperConfig::default().validate().is_ok());
    }

    #[test]
    fn test_validate_large_v2_ok() {
        assert!(WhisperConfig::whisper_large_v2().validate().is_ok());
    }

    #[test]
    fn test_validate_d_model_not_divisible_by_encoder_heads() {
        let cfg = WhisperConfig {
            encoder_attention_heads: 7,
            ..WhisperConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_d_model_not_divisible_by_decoder_heads() {
        let cfg = WhisperConfig {
            decoder_attention_heads: 7,
            ..WhisperConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_mel_bins() {
        let cfg = WhisperConfig {
            num_mel_bins: 0,
            ..WhisperConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_clone_preserves_fields() {
        let cfg = WhisperConfig::whisper_medium();
        let cloned = cfg.clone();
        assert_eq!(cfg.d_model, cloned.d_model);
        assert_eq!(cfg.vocab_size, cloned.vocab_size);
        assert_eq!(cfg.model_type, cloned.model_type);
    }

    #[test]
    fn test_lcg_varied_d_models() {
        let mut s = 97u64;
        let candidates = [256usize, 384, 512, 768, 1024];
        for _ in 0..4 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = (s % candidates.len() as u64) as usize;
            let d_model = candidates[idx];
            let heads = if d_model.is_multiple_of(8) { 8 } else { 4 };
            let cfg = WhisperConfig {
                d_model,
                encoder_attention_heads: heads,
                decoder_attention_heads: heads,
                ..WhisperConfig::default()
            };
            assert!(cfg.validate().is_ok(), "d_model={d_model} failed");
        }
    }
}
