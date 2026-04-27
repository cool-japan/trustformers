use serde::{Deserialize, Serialize};
use trustformers_core::errors::invalid_config;
use trustformers_core::traits::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_inner: Option<usize>,
    pub activation_function: String,
    pub resid_pdrop: f32,
    pub embd_pdrop: f32,
    pub attn_pdrop: f32,
    pub layer_norm_epsilon: f32,
    pub initializer_range: f32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub model_type: String,
}

impl Default for Gpt2Config {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            n_inner: None,
            activation_function: "gelu".to_string(),
            resid_pdrop: 0.1,
            embd_pdrop: 0.1,
            attn_pdrop: 0.1,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            bos_token_id: 50256,
            eos_token_id: 50256,
            model_type: "gpt2".to_string(),
        }
    }
}

impl Config for Gpt2Config {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if !self.n_embd.is_multiple_of(self.n_head) {
            return Err(invalid_config(
                "config_field",
                "n_embd must be divisible by n_head".to_string(),
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "GPT2"
    }
}

impl Gpt2Config {
    pub fn small() -> Self {
        Self::default()
    }

    pub fn medium() -> Self {
        Self {
            n_embd: 1024,
            n_head: 16,
            n_layer: 24,
            ..Self::default()
        }
    }

    pub fn large() -> Self {
        Self {
            n_embd: 1280,
            n_head: 20,
            n_layer: 36,
            ..Self::default()
        }
    }

    pub fn xl() -> Self {
        Self {
            n_embd: 1600,
            n_head: 25,
            n_layer: 48,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    #[test]
    fn test_default_is_small() {
        let cfg = Gpt2Config::default();
        assert_eq!(cfg.vocab_size, 50257);
        assert_eq!(cfg.n_embd, 768);
        assert_eq!(cfg.n_layer, 12);
        assert_eq!(cfg.n_head, 12);
        assert_eq!(cfg.n_positions, 1024);
    }

    #[test]
    fn test_medium_preset() {
        let cfg = Gpt2Config::medium();
        assert_eq!(cfg.n_embd, 1024);
        assert_eq!(cfg.n_head, 16);
        assert_eq!(cfg.n_layer, 24);
    }

    #[test]
    fn test_large_preset() {
        let cfg = Gpt2Config::large();
        assert_eq!(cfg.n_embd, 1280);
        assert_eq!(cfg.n_head, 20);
        assert_eq!(cfg.n_layer, 36);
    }

    #[test]
    fn test_xl_preset() {
        let cfg = Gpt2Config::xl();
        assert_eq!(cfg.n_embd, 1600);
        assert_eq!(cfg.n_head, 25);
        assert_eq!(cfg.n_layer, 48);
    }

    #[test]
    fn test_head_dim_small() {
        // 768 / 12 = 64
        assert_eq!(Gpt2Config::small().n_embd / Gpt2Config::small().n_head, 64);
    }

    #[test]
    fn test_head_dim_xl() {
        // 1600 / 25 = 64
        assert_eq!(Gpt2Config::xl().n_embd / Gpt2Config::xl().n_head, 64);
    }

    #[test]
    fn test_bos_eos_are_same() {
        let cfg = Gpt2Config::default();
        assert_eq!(cfg.bos_token_id, 50256);
        assert_eq!(cfg.eos_token_id, 50256);
    }

    #[test]
    fn test_n_inner_none_by_default() {
        assert!(Gpt2Config::default().n_inner.is_none());
    }

    #[test]
    fn test_activation_is_gelu() {
        assert_eq!(Gpt2Config::default().activation_function, "gelu");
    }

    #[test]
    fn test_model_type_is_gpt2() {
        assert_eq!(Gpt2Config::default().model_type, "gpt2");
    }

    #[test]
    fn test_architecture_label() {
        assert_eq!(Gpt2Config::default().architecture(), "GPT2");
    }

    #[test]
    fn test_validate_small_ok() {
        assert!(Gpt2Config::small().validate().is_ok());
    }

    #[test]
    fn test_validate_medium_ok() {
        assert!(Gpt2Config::medium().validate().is_ok());
    }

    #[test]
    fn test_validate_xl_ok() {
        assert!(Gpt2Config::xl().validate().is_ok());
    }

    #[test]
    fn test_validate_n_embd_not_divisible_by_n_head() {
        let cfg = Gpt2Config {
            n_embd: 769,
            ..Gpt2Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_resid_pdrop_default() {
        assert!((Gpt2Config::default().resid_pdrop - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_clone_preserves_all_fields() {
        let cfg = Gpt2Config::medium();
        let cloned = cfg.clone();
        assert_eq!(cfg.n_embd, cloned.n_embd);
        assert_eq!(cfg.n_layer, cloned.n_layer);
        assert_eq!(cfg.activation_function, cloned.activation_function);
    }

    #[test]
    fn test_lcg_varied_n_embd() {
        let mut s = 101u64;
        for _ in 0..5 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let heads = 8usize;
            let mult = ((s % 8) + 1) as usize;
            let n_embd = heads * mult * 8;
            let cfg = Gpt2Config {
                n_embd,
                n_head: heads,
                ..Gpt2Config::default()
            };
            assert!(cfg.validate().is_ok(), "n_embd={n_embd} failed");
        }
    }
}
