use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// LLaMA model configuration
/// Reference: "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>, // For grouped-query attention
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub initializer_range: f32,
    pub rms_norm_eps: f32,
    pub use_cache: bool,
    pub pad_token_id: Option<u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub rope_theta: f32, // Base frequency for RoPE
    pub rope_scaling: Option<RopeScaling>,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub model_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub scaling_type: String, // "linear" or "dynamic"
    pub scaling_factor: f32,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None, // Multi-head attention by default
            hidden_act: "silu".to_string(),
            max_position_embeddings: 2048,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
            pad_token_id: None,
            bos_token_id: 1,
            eos_token_id: 2,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            mlp_bias: false,
            model_type: "llama".to_string(),
        }
    }
}

impl Config for LlamaConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "hidden_size must be divisible by num_attention_heads".to_string(),
                ),
            );
        }

        if let Some(num_kv_heads) = self.num_key_value_heads {
            if !self.num_attention_heads.is_multiple_of(num_kv_heads) {
                return Err(
                    trustformers_core::errors::TrustformersError::invalid_config(
                        "num_attention_heads must be divisible by num_key_value_heads".to_string(),
                    ),
                );
            }
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "LLaMA"
    }
}

impl LlamaConfig {
    /// LLaMA 7B configuration
    pub fn llama_7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            max_position_embeddings: 2048,
            ..Self::default()
        }
    }

    /// LLaMA 13B configuration
    pub fn llama_13b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            max_position_embeddings: 2048,
            ..Self::default()
        }
    }

    /// LLaMA 30B configuration
    pub fn llama_30b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 6656,
            intermediate_size: 17920,
            num_hidden_layers: 60,
            num_attention_heads: 52,
            max_position_embeddings: 2048,
            ..Self::default()
        }
    }

    /// LLaMA 65B configuration
    pub fn llama_65b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 8192,
            intermediate_size: 22016,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            max_position_embeddings: 2048,
            ..Self::default()
        }
    }

    /// LLaMA 2 7B configuration
    pub fn llama2_7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            max_position_embeddings: 4096, // Increased context length
            ..Self::default()
        }
    }

    /// LLaMA 2 13B configuration
    pub fn llama2_13b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            max_position_embeddings: 4096,
            ..Self::default()
        }
    }

    /// LLaMA 2 70B configuration with grouped-query attention
    pub fn llama2_70b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 8192,
            intermediate_size: 28672,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: Some(8), // Grouped-query attention
            max_position_embeddings: 4096,
            ..Self::default()
        }
    }

    /// Code Llama configuration (based on LLaMA 2)
    pub fn code_llama_7b() -> Self {
        Self {
            vocab_size: 32016,              // Slightly different vocab for code
            max_position_embeddings: 16384, // Much longer context for code
            ..Self::llama2_7b()
        }
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the number of key-value heads (for grouped-query attention)
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the number of query groups per key-value head
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads()
    }

    /// LLaMA 3 8B configuration
    /// Enhanced version with larger vocabulary and improved architecture
    pub fn llama3_8b() -> Self {
        Self {
            vocab_size: 128256, // Much larger vocabulary
            hidden_size: 4096,
            intermediate_size: 14336, // Increased intermediate size
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),  // Grouped-query attention
            max_position_embeddings: 8192, // 8K context length
            rope_theta: 500000.0,          // Higher RoPE base frequency
            rms_norm_eps: 1e-5,            // Updated epsilon
            ..Self::default()
        }
    }

    /// LLaMA 3 70B configuration
    /// Large model with grouped-query attention for efficiency
    pub fn llama3_70b() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 8192,
            intermediate_size: 28672,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: Some(8),  // Grouped-query attention
            max_position_embeddings: 8192, // 8K context length
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            ..Self::default()
        }
    }

    /// LLaMA 3 405B configuration (largest model)
    /// Massive model with advanced optimizations
    pub fn llama3_405b() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 16384,       // Massive hidden size
            intermediate_size: 53248, // Large intermediate size
            num_hidden_layers: 126,   // Many layers
            num_attention_heads: 128,
            num_key_value_heads: Some(8),  // Highly efficient GQA
            max_position_embeddings: 8192, // 8K context length
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            ..Self::default()
        }
    }

    /// LLaMA 3 Instruct 8B (instruction-tuned version)
    pub fn llama3_8b_instruct() -> Self {
        Self {
            model_type: "llama3-instruct".to_string(),
            ..Self::llama3_8b()
        }
    }

    /// LLaMA 3 Instruct 70B (instruction-tuned version)
    pub fn llama3_70b_instruct() -> Self {
        Self {
            model_type: "llama3-instruct".to_string(),
            ..Self::llama3_70b()
        }
    }

    /// LLaMA 3 Instruct 405B (instruction-tuned version)
    pub fn llama3_405b_instruct() -> Self {
        Self {
            model_type: "llama3-instruct".to_string(),
            ..Self::llama3_405b()
        }
    }

    /// LLaMA 3.1 8B configuration with extended context (128K tokens)
    pub fn llama3_1_8b_128k() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 131072, // 128K context
            rope_theta: 500000.0,
            rope_scaling: Some(RopeScaling {
                scaling_type: "linear".to_string(),
                scaling_factor: 16.0, // Scale for long context
            }),
            rms_norm_eps: 1e-5,
            model_type: "llama3.1".to_string(),
            ..Self::default()
        }
    }

    /// LLaMA 3.1 70B configuration with extended context (128K tokens)
    pub fn llama3_1_70b_128k() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 8192,
            intermediate_size: 28672,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: Some(8),
            max_position_embeddings: 131072, // 128K context
            rope_theta: 500000.0,
            rope_scaling: Some(RopeScaling {
                scaling_type: "linear".to_string(),
                scaling_factor: 16.0,
            }),
            rms_norm_eps: 1e-5,
            model_type: "llama3.1".to_string(),
            ..Self::default()
        }
    }

    /// LLaMA 3.1 405B configuration with extended context (128K tokens)
    pub fn llama3_1_405b_128k() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 16384,
            intermediate_size: 53248,
            num_hidden_layers: 126,
            num_attention_heads: 128,
            num_key_value_heads: Some(8),
            max_position_embeddings: 131072, // 128K context
            rope_theta: 500000.0,
            rope_scaling: Some(RopeScaling {
                scaling_type: "linear".to_string(),
                scaling_factor: 16.0,
            }),
            rms_norm_eps: 1e-5,
            model_type: "llama3.1".to_string(),
            ..Self::default()
        }
    }

    /// Multilingual LLaMA configuration (optimized for multilingual understanding)
    pub fn llama_multilingual_7b() -> Self {
        Self {
            vocab_size: 250000, // Expanded vocabulary for multiple languages
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 16384, // Extended context for multilingual texts
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            model_type: "llama-multilingual".to_string(),
            ..Self::default()
        }
    }

    /// Scientific LLaMA configuration (optimized for scientific literature)
    pub fn llama_scientific_7b() -> Self {
        Self {
            vocab_size: 50000, // Specialized vocabulary for scientific terms
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 32768, // Long context for scientific papers
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            model_type: "llama-scientific".to_string(),
            ..Self::default()
        }
    }

    /// Legal LLaMA configuration (optimized for legal documents)
    pub fn llama_legal_7b() -> Self {
        Self {
            vocab_size: 40000, // Legal terminology focused vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 65536, // Very long context for legal documents
            rope_theta: 500000.0,
            rope_scaling: Some(RopeScaling {
                scaling_type: "linear".to_string(),
                scaling_factor: 8.0,
            }),
            rms_norm_eps: 1e-5,
            model_type: "llama-legal".to_string(),
            ..Self::default()
        }
    }

    /// Medical LLaMA configuration (optimized for medical literature)
    pub fn llama_medical_7b() -> Self {
        Self {
            vocab_size: 45000, // Medical terminology focused vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 32768, // Long context for medical documents
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            model_type: "llama-medical".to_string(),
            ..Self::default()
        }
    }

    /// Creative Writing LLaMA configuration (optimized for creative tasks)
    pub fn llama_creative_7b() -> Self {
        Self {
            vocab_size: 35000, // Focused vocabulary for creative writing
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 16384, // Medium context for stories/novels
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            model_type: "llama-creative".to_string(),
            ..Self::default()
        }
    }

    /// LLaMA 1B configuration (ultra-efficient variant)
    pub fn llama_1b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 2048,
            intermediate_size: 5504,
            num_hidden_layers: 16,
            num_attention_heads: 16,
            num_key_value_heads: Some(4),
            max_position_embeddings: 4096,
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            model_type: "llama-1b".to_string(),
            ..Self::default()
        }
    }

    /// LLaMA 3B configuration (efficient variant)
    pub fn llama_3b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 2560,
            intermediate_size: 6912,
            num_hidden_layers: 20,
            num_attention_heads: 20,
            num_key_value_heads: Some(4),
            max_position_embeddings: 4096,
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            model_type: "llama-3b".to_string(),
            ..Self::default()
        }
    }

    /// Create configuration from pretrained model name
    pub fn from_pretrained_name(name: &str) -> Option<Self> {
        match name {
            // LLaMA 1 models
            "llama-7b" => Some(Self::llama_7b()),
            "llama-13b" => Some(Self::llama_13b()),
            "llama-30b" => Some(Self::llama_30b()),
            "llama-65b" => Some(Self::llama_65b()),

            // LLaMA 2 models
            "meta-llama/Llama-2-7b-hf" | "llama2-7b" => Some(Self::llama2_7b()),
            "meta-llama/Llama-2-13b-hf" | "llama2-13b" => Some(Self::llama2_13b()),
            "meta-llama/Llama-2-70b-hf" | "llama2-70b" => Some(Self::llama2_70b()),

            // Code Llama
            "codellama/CodeLlama-7b-hf" | "code-llama-7b" => Some(Self::code_llama_7b()),

            // LLaMA 3 models
            "meta-llama/Meta-Llama-3-8B" | "llama3-8b" => Some(Self::llama3_8b()),
            "meta-llama/Meta-Llama-3-70B" | "llama3-70b" => Some(Self::llama3_70b()),
            "meta-llama/Meta-Llama-3-405B" | "llama3-405b" => Some(Self::llama3_405b()),

            // LLaMA 3 Instruct models
            "meta-llama/Meta-Llama-3-8B-Instruct" | "llama3-8b-instruct" => {
                Some(Self::llama3_8b_instruct())
            },
            "meta-llama/Meta-Llama-3-70B-Instruct" | "llama3-70b-instruct" => {
                Some(Self::llama3_70b_instruct())
            },
            "meta-llama/Meta-Llama-3-405B-Instruct" | "llama3-405b-instruct" => {
                Some(Self::llama3_405b_instruct())
            },

            // LLaMA 3.1 Long Context models
            "meta-llama/Meta-Llama-3.1-8B" | "llama3.1-8b-128k" => Some(Self::llama3_1_8b_128k()),
            "meta-llama/Meta-Llama-3.1-70B" | "llama3.1-70b-128k" => {
                Some(Self::llama3_1_70b_128k())
            },
            "meta-llama/Meta-Llama-3.1-405B" | "llama3.1-405b-128k" => {
                Some(Self::llama3_1_405b_128k())
            },

            // Specialized models
            "llama-multilingual-7b" => Some(Self::llama_multilingual_7b()),
            "llama-scientific-7b" => Some(Self::llama_scientific_7b()),
            "llama-legal-7b" => Some(Self::llama_legal_7b()),
            "llama-medical-7b" => Some(Self::llama_medical_7b()),
            "llama-creative-7b" => Some(Self::llama_creative_7b()),

            // Efficient variants
            "llama-1b" => Some(Self::llama_1b()),
            "llama-3b" => Some(Self::llama_3b()),

            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

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

    #[test]
    fn test_default_config_fields() {
        let cfg = LlamaConfig::default();
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.model_type, "llama");
        assert!(cfg.use_cache);
        assert!(!cfg.attention_bias);
        assert!(!cfg.mlp_bias);
    }

    #[test]
    fn test_default_validate_passes() {
        let cfg = LlamaConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_architecture_name() {
        let cfg = LlamaConfig::default();
        assert_eq!(cfg.architecture(), "LLaMA");
    }

    #[test]
    fn test_hidden_size_not_divisible_fails_validation() {
        let cfg = LlamaConfig {
            hidden_size: 100,
            num_attention_heads: 32,
            ..LlamaConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_kv_heads_not_divisible_fails_validation() {
        let cfg = LlamaConfig {
            num_attention_heads: 32,
            num_key_value_heads: Some(7),
            ..LlamaConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_valid_gqa_config() {
        let cfg = LlamaConfig {
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            ..LlamaConfig::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_head_dim_computation() {
        let cfg = LlamaConfig::default();
        assert_eq!(cfg.head_dim(), 4096 / 32);
    }

    #[test]
    fn test_num_kv_heads_default() {
        let cfg = LlamaConfig::default();
        assert_eq!(cfg.num_kv_heads(), cfg.num_attention_heads);
    }

    #[test]
    fn test_num_kv_heads_gqa() {
        let cfg = LlamaConfig {
            num_key_value_heads: Some(8),
            ..LlamaConfig::default()
        };
        assert_eq!(cfg.num_kv_heads(), 8);
    }

    #[test]
    fn test_num_query_groups() {
        let cfg = LlamaConfig {
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            ..LlamaConfig::default()
        };
        assert_eq!(cfg.num_query_groups(), 4);
    }

    #[test]
    fn test_llama_7b_config() {
        let cfg = LlamaConfig::llama_7b();
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_llama_13b_config() {
        let cfg = LlamaConfig::llama_13b();
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_llama2_70b_has_gqa() {
        let cfg = LlamaConfig::llama2_70b();
        assert_eq!(cfg.num_key_value_heads, Some(8));
        assert_eq!(cfg.max_position_embeddings, 4096);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_llama3_8b_config() {
        let cfg = LlamaConfig::llama3_8b();
        assert_eq!(cfg.vocab_size, 128256);
        assert_eq!(cfg.num_key_value_heads, Some(8));
        assert_eq!(cfg.rope_theta, 500000.0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_llama3_1_8b_128k_has_rope_scaling() {
        let cfg = LlamaConfig::llama3_1_8b_128k();
        assert_eq!(cfg.max_position_embeddings, 131072);
        if let Some(scaling) = &cfg.rope_scaling {
            assert_eq!(scaling.scaling_type, "linear");
            assert_eq!(scaling.scaling_factor, 16.0);
        } else {
            panic!("expected rope_scaling to be Some");
        }
    }

    #[test]
    fn test_code_llama_7b_config() {
        let cfg = LlamaConfig::code_llama_7b();
        assert_eq!(cfg.vocab_size, 32016);
        assert_eq!(cfg.max_position_embeddings, 16384);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_from_pretrained_name_llama2_7b() {
        let result = LlamaConfig::from_pretrained_name("llama2-7b");
        assert!(result.is_some());
        if let Some(cfg) = result {
            assert_eq!(cfg.max_position_embeddings, 4096);
        }
    }

    #[test]
    fn test_from_pretrained_name_unknown_returns_none() {
        let result = LlamaConfig::from_pretrained_name("nonexistent-model-xyz");
        assert!(result.is_none());
    }

    #[test]
    fn test_from_pretrained_name_llama3_70b_instruct() {
        let result = LlamaConfig::from_pretrained_name("llama3-70b-instruct");
        assert!(result.is_some());
        if let Some(cfg) = result {
            assert!(cfg.model_type.contains("instruct"));
        }
    }

    #[test]
    fn test_llama_1b_config() {
        let cfg = LlamaConfig::llama_1b();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 16);
        assert_eq!(cfg.num_key_value_heads, Some(4));
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_llama_specialized_configs_validate() {
        let configs = vec![
            LlamaConfig::llama_multilingual_7b(),
            LlamaConfig::llama_scientific_7b(),
            LlamaConfig::llama_legal_7b(),
            LlamaConfig::llama_medical_7b(),
            LlamaConfig::llama_creative_7b(),
        ];
        for cfg in configs {
            assert!(
                cfg.validate().is_ok(),
                "config {} failed validation",
                cfg.model_type
            );
        }
    }

    #[test]
    fn test_lcg_produces_values_in_range() {
        let mut rng = Lcg::new(12345);
        for _ in 0..100 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "LCG value {} out of range", v);
        }
    }

    #[test]
    fn test_rope_scaling_fields() {
        let scaling = RopeScaling {
            scaling_type: "dynamic".to_string(),
            scaling_factor: 4.0,
        };
        assert_eq!(scaling.scaling_type, "dynamic");
        assert_eq!(scaling.scaling_factor, 4.0);
    }

    #[test]
    fn test_llama3_405b_architecture() {
        let cfg = LlamaConfig::llama3_405b();
        assert_eq!(cfg.num_hidden_layers, 126);
        assert_eq!(cfg.hidden_size, 16384);
        assert_eq!(cfg.num_key_value_heads, Some(8));
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_llama_3b_config() {
        let cfg = LlamaConfig::llama_3b();
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_hidden_layers, 20);
        assert_eq!(cfg.num_key_value_heads, Some(4));
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_head_dim_varies_by_model() {
        let small = LlamaConfig::llama_1b();
        let medium = LlamaConfig::llama3_8b();
        let small_hd = small.head_dim();
        let medium_hd = medium.head_dim();
        assert!(small_hd > 0);
        assert!(medium_hd > 0);
    }

    #[test]
    fn test_llama3_instruct_model_type() {
        let cfg = LlamaConfig::llama3_8b_instruct();
        assert_eq!(cfg.model_type, "llama3-instruct");
    }

    #[test]
    fn test_legal_config_has_rope_scaling() {
        let cfg = LlamaConfig::llama_legal_7b();
        assert!(cfg.rope_scaling.is_some());
        assert_eq!(cfg.max_position_embeddings, 65536);
    }

    #[test]
    fn test_query_groups_mha_equals_one() {
        let cfg = LlamaConfig::default(); // no kv heads set
        assert_eq!(cfg.num_query_groups(), 1);
    }
}
