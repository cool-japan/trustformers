//! # CodeLlama
//!
//! CodeLlama is Meta AI's family of code-specialised LLMs built by fine-tuning
//! LLaMA-2 on a large corpus of code (Rozière et al., 2023).  The architecture
//! is identical to LLaMA-2 but adds:
//!
//! * **RoPE scaling** — CodeLlama-34B uses linear frequency scaling to achieve
//!   effective context lengths up to 100K tokens.
//! * **Fill-in-the-Middle (FIM)** — models trained with the `<FILL_ME>` token
//!   can complete spans given prefix and suffix context.
//! * **Repository-level context** — CodeLlama-70B supports 4K-token windows
//!   designed for cross-file understanding.
//!
//! ## RoPE scaling strategies
//!
//! ### Linear scaling
//! ```text
//! inv_freq_scaled[i] = (theta ^ (-2i/d)) / factor
//! ```
//!
//! ### Dynamic NTK scaling
//! ```text
//! inv_freq[i] = ((alpha * theta) ^ (-2i/d))
//! ```
//!
//! ## Model variants
//!
//! | Variant  | Layers | Hidden | Q-heads | KV-heads | Max context |
//! |----------|--------|--------|---------|----------|-------------|
//! | 7B       | 32     | 4096   | 32      | 32       | 16K         |
//! | 13B      | 40     | 5120   | 40      | 40       | 16K         |
//! | 34B      | 48     | 8192   | 64      | 8 (GQA)  | ~100K (4×)  |
//! | 70B      | 80     | 8192   | 64      | 8 (GQA)  | 4K          |

pub mod config;
pub mod model;
pub mod tasks;

#[cfg(test)]
mod tests;

pub use config::{CodeLlamaConfig, RopeScalingConfig, RopeScalingType};
pub use model::{
    CodeLlamaAttention, CodeLlamaDecoderLayer, CodeLlamaForCausalLM, CodeLlamaMLP, CodeLlamaModel,
    CodeLlamaRMSNorm, CodeLlamaRotaryEmbedding,
};
pub use tasks::{CodeLMOutput, CodeLlamaCompletion, CodeLlamaInfilling, CodeLlamaRepoLevel};

// ── Additional tests beyond tests.rs ─────────────────────────────────────────

#[cfg(test)]
mod extra_tests {
    use super::*;
    use trustformers_core::traits::Config;

    fn mini_config() -> CodeLlamaConfig {
        CodeLlamaConfig {
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 8,
            vocab_size: 512,
            max_position_embeddings: 128,
            rope_scaling: None,
            infilling: false,
            programming_languages: vec!["rust".to_string()],
            ..CodeLlamaConfig::default()
        }
    }

    // ── Preset field validation ───────────────────────────────────────────────

    #[test]
    fn test_codellama_13b_preset_fields() {
        let cfg = CodeLlamaConfig::codellama_13b();
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.intermediate_size, 13824);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.num_attention_heads, 40);
        assert_eq!(cfg.num_key_value_heads, 40);
        assert!(!cfg.uses_gqa(), "13B uses full MHA");
        assert_eq!(cfg.num_query_groups(), 1);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_codellama_7b_instruct_infilling_enabled() {
        let cfg = CodeLlamaConfig::codellama_7b_instruct();
        assert!(cfg.infilling, "7B-Instruct must have infilling=true");
        assert_eq!(cfg.hidden_size, 4096);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_codellama_7b_infilling_disabled() {
        let cfg = CodeLlamaConfig::codellama_7b();
        assert!(!cfg.infilling, "7B base must have infilling=false");
    }

    #[test]
    fn test_codellama_70b_native_context() {
        let cfg = CodeLlamaConfig::codellama_70b();
        assert_eq!(cfg.max_position_embeddings, 4096);
        // No rope scaling → effective context equals native context
        assert!(cfg.rope_scaling.is_none());
        assert_eq!(cfg.effective_max_context(), 4096);
    }

    // ── from_pretrained_name lookups ──────────────────────────────────────────

    #[test]
    fn test_from_pretrained_name_7b() {
        let cfg = CodeLlamaConfig::from_pretrained_name("codellama/CodeLlama-7b-hf");
        assert!(cfg.is_some());
        let cfg = cfg.expect("must match");
        assert_eq!(cfg.hidden_size, 4096);
    }

    #[test]
    fn test_from_pretrained_name_short_alias() {
        let cfg = CodeLlamaConfig::from_pretrained_name("codellama-34b");
        assert!(cfg.is_some());
        let cfg = cfg.expect("must match");
        assert_eq!(cfg.num_hidden_layers, 48);
    }

    #[test]
    fn test_from_pretrained_name_unknown_returns_none() {
        let cfg = CodeLlamaConfig::from_pretrained_name("unknown-model-xyz");
        assert!(cfg.is_none());
    }

    #[test]
    fn test_from_pretrained_name_instruct() {
        let cfg = CodeLlamaConfig::from_pretrained_name("codellama-7b-instruct");
        assert!(cfg.is_some());
        let cfg = cfg.expect("must match");
        assert!(cfg.infilling);
    }

    // ── RopeScalingType Display ───────────────────────────────────────────────

    #[test]
    fn test_rope_scaling_type_display_linear() {
        assert_eq!(format!("{}", RopeScalingType::Linear), "linear");
    }

    #[test]
    fn test_rope_scaling_type_display_dynamic() {
        assert_eq!(format!("{}", RopeScalingType::Dynamic), "dynamic");
    }

    #[test]
    fn test_rope_scaling_type_equality() {
        assert_eq!(RopeScalingType::Linear, RopeScalingType::Linear);
        assert_ne!(RopeScalingType::Linear, RopeScalingType::Dynamic);
    }

    // ── Config accessors ──────────────────────────────────────────────────────

    #[test]
    fn test_rope_theta_default_value() {
        let cfg = CodeLlamaConfig::default();
        assert!(
            (cfg.rope_theta - 10000.0).abs() < 1e-3,
            "CodeLlama default rope_theta should be 10000.0"
        );
    }

    #[test]
    fn test_bos_eos_token_ids() {
        let cfg = CodeLlamaConfig::default();
        assert_eq!(cfg.bos_token_id, 1);
        assert_eq!(cfg.eos_token_id, 2);
    }

    #[test]
    fn test_programming_languages_default_nonempty() {
        let cfg = CodeLlamaConfig::default();
        assert!(
            !cfg.programming_languages.is_empty(),
            "default config should have programming languages"
        );
        assert!(cfg.programming_languages.contains(&"rust".to_string()));
    }

    #[test]
    fn test_effective_max_context_no_scaling() {
        let cfg = mini_config();
        assert_eq!(cfg.effective_max_context(), cfg.max_position_embeddings);
    }

    #[test]
    fn test_head_dim_computation() {
        let cfg = mini_config();
        // hidden_size=64, num_attention_heads=8 → head_dim=8
        assert_eq!(cfg.head_dim(), 8);
    }

    #[test]
    fn test_architecture_label() {
        assert_eq!(mini_config().architecture(), "CodeLlama");
    }

    // ── Task wrappers ─────────────────────────────────────────────────────────

    #[test]
    fn test_code_llama_completion_construction() {
        let task = CodeLlamaCompletion::new(mini_config());
        assert!(
            task.is_ok(),
            "CodeLlamaCompletion must construct successfully"
        );
    }

    #[test]
    fn test_code_llama_infilling_enabled_flag() {
        let mut cfg = mini_config();
        cfg.infilling = true;
        let task = CodeLlamaInfilling::new(cfg).expect("construction");
        assert!(
            task.infilling_enabled,
            "infilling_enabled must reflect config"
        );
    }

    #[test]
    fn test_code_llama_infilling_disabled_flag() {
        let task = CodeLlamaInfilling::new(mini_config()).expect("construction");
        assert!(
            !task.infilling_enabled,
            "infilling_enabled must be false for mini_config"
        );
    }

    #[test]
    fn test_code_llama_repo_level_construction() {
        let task = CodeLlamaRepoLevel::new(mini_config());
        assert!(
            task.is_ok(),
            "CodeLlamaRepoLevel must construct successfully"
        );
        let task = task.expect("checked");
        assert_eq!(
            task.repo_context_limit,
            mini_config().effective_max_context()
        );
    }

    // ── Config clone and serialization ────────────────────────────────────────

    #[test]
    fn test_config_clone() {
        let cfg = mini_config();
        let cloned = cfg.clone();
        assert_eq!(cfg.hidden_size, cloned.hidden_size);
        assert_eq!(cfg.vocab_size, cloned.vocab_size);
    }

    #[test]
    fn test_config_json_round_trip() {
        let cfg = CodeLlamaConfig::codellama_34b();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let restored: CodeLlamaConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg.hidden_size, restored.hidden_size);
        assert_eq!(cfg.num_key_value_heads, restored.num_key_value_heads);
        assert!(restored.rope_scaling.is_some());
    }

    #[test]
    fn test_rope_scaling_factor_invalid_zero() {
        let cfg = CodeLlamaConfig {
            rope_scaling: Some(RopeScalingConfig {
                scaling_type: RopeScalingType::Linear,
                factor: 0.0,
            }),
            ..mini_config()
        };
        assert!(
            cfg.validate().is_err(),
            "rope_scaling factor=0.0 must be rejected"
        );
    }
}
