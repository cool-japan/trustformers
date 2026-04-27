use serde::{Deserialize, Serialize};
use trustformers_core::errors::{invalid_config, Result};
use trustformers_core::traits::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub pad_token_id: u32,
    pub position_embedding_type: Option<String>,
    pub use_cache: Option<bool>,
    pub classifier_dropout: Option<f32>,
}

impl Default for BertConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: Some("absolute".to_string()),
            use_cache: Some(true),
            classifier_dropout: None,
        }
    }
}

impl Config for BertConfig {
    fn validate(&self) -> Result<()> {
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(invalid_config(
                "hidden_size",
                format!(
                    "hidden_size {} must be divisible by num_attention_heads {}",
                    self.hidden_size, self.num_attention_heads
                ),
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "BERT"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    // --- LCG deterministic pseudo-random for reproducible test data ---
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
        *state
    }

    fn lcg_f32(state: &mut u64) -> f32 {
        let v = lcg_next(state);
        (v >> 33) as f32 / u32::MAX as f32
    }

    // --- Default / BERT-base ---

    #[test]
    fn test_bert_base_default_vocab_size() {
        let cfg = BertConfig::default();
        assert_eq!(cfg.vocab_size, 30522, "BERT-base vocab_size must be 30522");
    }

    #[test]
    fn test_bert_base_default_hidden_size() {
        let cfg = BertConfig::default();
        assert_eq!(cfg.hidden_size, 768, "BERT-base hidden_size must be 768");
    }

    #[test]
    fn test_bert_base_default_num_hidden_layers() {
        let cfg = BertConfig::default();
        assert_eq!(
            cfg.num_hidden_layers, 12,
            "BERT-base must have 12 hidden layers"
        );
    }

    #[test]
    fn test_bert_base_default_num_attention_heads() {
        let cfg = BertConfig::default();
        assert_eq!(
            cfg.num_attention_heads, 12,
            "BERT-base must have 12 attention heads"
        );
    }

    #[test]
    fn test_bert_base_default_max_position_embeddings() {
        let cfg = BertConfig::default();
        assert_eq!(
            cfg.max_position_embeddings, 512,
            "BERT-base max_position_embeddings must be 512"
        );
    }

    #[test]
    fn test_bert_base_intermediate_size_is_4x_hidden() {
        let cfg = BertConfig::default();
        assert_eq!(
            cfg.intermediate_size,
            4 * cfg.hidden_size,
            "intermediate_size must equal 4 * hidden_size"
        );
    }

    #[test]
    fn test_bert_base_type_vocab_size() {
        let cfg = BertConfig::default();
        assert_eq!(cfg.type_vocab_size, 2, "type_vocab_size must be 2 for BERT");
    }

    #[test]
    fn test_bert_base_hidden_act_is_gelu() {
        let cfg = BertConfig::default();
        assert_eq!(
            cfg.hidden_act, "gelu",
            "BERT default activation must be gelu"
        );
    }

    #[test]
    fn test_bert_base_position_embedding_type_is_absolute() {
        let cfg = BertConfig::default();
        assert_eq!(
            cfg.position_embedding_type.as_deref(),
            Some("absolute"),
            "default positional embedding type must be absolute"
        );
    }

    #[test]
    fn test_bert_base_use_cache_default_true() {
        let cfg = BertConfig::default();
        assert_eq!(cfg.use_cache, Some(true), "use_cache must default to true");
    }

    #[test]
    fn test_bert_base_classifier_dropout_is_none_by_default() {
        let cfg = BertConfig::default();
        assert!(
            cfg.classifier_dropout.is_none(),
            "classifier_dropout must default to None"
        );
    }

    #[test]
    fn test_bert_base_pad_token_id_is_zero() {
        let cfg = BertConfig::default();
        assert_eq!(cfg.pad_token_id, 0, "pad_token_id must default to 0");
    }

    // --- BERT-large ---

    #[test]
    fn test_bert_large_hidden_size() {
        let cfg = BertConfig {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            ..BertConfig::default()
        };
        assert_eq!(cfg.hidden_size, 1024, "BERT-large hidden_size must be 1024");
    }

    #[test]
    fn test_bert_large_num_layers() {
        let cfg = BertConfig {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            ..BertConfig::default()
        };
        assert_eq!(
            cfg.num_hidden_layers, 24,
            "BERT-large must have 24 hidden layers"
        );
    }

    #[test]
    fn test_bert_large_num_heads() {
        let cfg = BertConfig {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            ..BertConfig::default()
        };
        assert_eq!(
            cfg.num_attention_heads, 16,
            "BERT-large must have 16 attention heads"
        );
    }

    #[test]
    fn test_bert_large_intermediate_size_is_4x_hidden() {
        let cfg = BertConfig {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            ..BertConfig::default()
        };
        assert_eq!(
            cfg.intermediate_size,
            4 * cfg.hidden_size,
            "BERT-large: intermediate_size must equal 4 * hidden_size"
        );
    }

    // --- Architecture name ---

    #[test]
    fn test_architecture_name() {
        let cfg = BertConfig::default();
        assert_eq!(cfg.architecture(), "BERT");
    }

    // --- Validation ---

    #[test]
    fn test_validate_passes_for_bert_base() {
        let cfg = BertConfig::default();
        assert!(
            cfg.validate().is_ok(),
            "BERT-base config must pass validation"
        );
    }

    #[test]
    fn test_validate_fails_when_hidden_not_divisible_by_heads() {
        let cfg = BertConfig {
            hidden_size: 769, // not divisible by 12
            num_attention_heads: 12,
            ..BertConfig::default()
        };
        let result = cfg.validate();
        assert!(
            result.is_err(),
            "Validation must fail when hidden_size is not divisible by num_attention_heads"
        );
    }

    #[test]
    fn test_head_size_relationship() {
        let cfg = BertConfig::default();
        let head_size = cfg.hidden_size / cfg.num_attention_heads;
        assert_eq!(head_size, 64, "BERT-base head size must be 64");
    }

    #[test]
    fn test_lcg_deterministic_values() {
        let mut state: u64 = 42;
        let v1 = lcg_f32(&mut state);
        let v2 = lcg_f32(&mut state);
        // LCG must be deterministic and in [0,1)
        assert!((0.0..1.0).contains(&v1), "LCG value must be in [0, 1)");
        assert!((0.0..1.0).contains(&v2), "LCG value must be in [0, 1)");
        assert_ne!(v1, v2, "Consecutive LCG values must differ");
    }

    #[test]
    fn test_custom_config_roundtrip_serde() {
        let original = BertConfig {
            vocab_size: 1000,
            hidden_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            intermediate_size: 1024,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 128,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: Some("absolute".to_string()),
            use_cache: Some(true),
            classifier_dropout: None,
        };
        let json = serde_json::to_string(&original).expect("Serialization must succeed");
        let restored: BertConfig =
            serde_json::from_str(&json).expect("Deserialization must succeed");
        assert_eq!(original.vocab_size, restored.vocab_size);
        assert_eq!(original.hidden_size, restored.hidden_size);
        assert_eq!(original.num_hidden_layers, restored.num_hidden_layers);
    }
}
