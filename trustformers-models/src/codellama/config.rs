use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// RoPE position scaling strategy for long-context CodeLlama variants
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum RopeScalingType {
    /// Linear position scaling: divide all frequencies by `factor`.
    ///
    /// `freq_scaled = base_freq / factor`
    ///
    /// Used by CodeLlama-34B to extend context to 100K tokens.
    Linear,
    /// Dynamic NTK-aware scaling: increase the RoPE base period.
    ///
    /// `base_freq = (alpha * theta)^(-2i/d)`, where `alpha = factor`
    ///
    /// Maintains quality at long range better than linear scaling.
    Dynamic,
}

impl std::fmt::Display for RopeScalingType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RopeScalingType::Linear => write!(f, "linear"),
            RopeScalingType::Dynamic => write!(f, "dynamic"),
        }
    }
}

/// RoPE scaling configuration for long-context variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScalingConfig {
    /// Scaling strategy (linear or dynamic NTK)
    pub scaling_type: RopeScalingType,
    /// Multiplicative scale factor (e.g. 4.0 for 4× context extension)
    pub factor: f32,
}

/// CodeLlama model configuration
///
/// CodeLlama is a family of code-optimised models built by fine-tuning LLaMA-2
/// on code-heavy data (Meta AI, 2023).  The architecture is identical to
/// LLaMA-2, but adds:
///
/// * **Optional RoPE scaling**: CodeLlama-34B supports up to 100K tokens via
///   linear frequency scaling.
/// * **Fill-in-the-Middle (FIM) / infilling**: Models trained with the
///   `<FILL_ME>` sentinel token.
/// * **Programming language metadata**: tracks which languages the model was
///   trained on (informational).
///
/// Reference: "Code Llama: Open Foundation Models for Code" (Rozière et al., 2023)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeLlamaConfig {
    // ── Architecture (identical to LLaMA-2) ──────────────────────────────────
    /// Hidden dimension
    pub hidden_size: usize,
    /// FFN intermediate dimension
    pub intermediate_size: usize,
    /// Number of decoder layers
    pub num_hidden_layers: usize,
    /// Number of query attention heads
    pub num_attention_heads: usize,
    /// Number of KV heads (GQA when < num_attention_heads)
    pub num_key_value_heads: usize,
    /// Vocabulary size (32016 for most CodeLlama variants)
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// RoPE base frequency
    pub rope_theta: f32,
    /// RMSNorm epsilon
    pub rms_norm_eps: f64,
    /// Whether to use attention bias
    pub attention_bias: bool,
    /// Whether to use MLP bias
    pub mlp_bias: bool,
    /// BOS token ID
    pub bos_token_id: u32,
    /// EOS token ID
    pub eos_token_id: u32,
    /// Whether to use KV cache
    pub use_cache: bool,
    /// Optional pad token ID
    pub pad_token_id: Option<u32>,

    // ── CodeLlama-specific ────────────────────────────────────────────────────
    /// Optional RoPE scaling for long-context variants (e.g. 34B / 100K)
    pub rope_scaling: Option<RopeScalingConfig>,
    /// Whether the model supports Fill-in-the-Middle (FIM) infilling
    pub infilling: bool,
    /// Programming languages included in the training mix (informational)
    pub programming_languages: Vec<String>,
}

impl Default for CodeLlamaConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            vocab_size: 32016, // CodeLlama extends LLaMA-2 vocab slightly
            max_position_embeddings: 16384,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            attention_bias: false,
            mlp_bias: false,
            bos_token_id: 1,
            eos_token_id: 2,
            use_cache: true,
            pad_token_id: None,
            rope_scaling: None,
            infilling: true,
            programming_languages: vec![
                "python".to_string(),
                "java".to_string(),
                "c++".to_string(),
                "javascript".to_string(),
                "rust".to_string(),
            ],
        }
    }
}

impl Config for CodeLlamaConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "hidden_size must be divisible by num_attention_heads".to_string(),
                ),
            );
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "num_attention_heads must be divisible by num_key_value_heads".to_string(),
                ),
            );
        }
        if self.vocab_size == 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "vocab_size must be greater than 0".to_string(),
                ),
            );
        }
        if let Some(ref scaling) = self.rope_scaling {
            if scaling.factor <= 0.0 {
                return Err(
                    trustformers_core::errors::TrustformersError::invalid_config(
                        "rope_scaling.factor must be positive".to_string(),
                    ),
                );
            }
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "CodeLlama"
    }
}

impl CodeLlamaConfig {
    /// Compute head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Number of query groups per KV head
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Whether this variant uses GQA
    pub fn uses_gqa(&self) -> bool {
        self.num_key_value_heads < self.num_attention_heads
    }

    /// Effective maximum context after optional RoPE scaling
    pub fn effective_max_context(&self) -> usize {
        match &self.rope_scaling {
            Some(s) => (self.max_position_embeddings as f32 * s.factor) as usize,
            None => self.max_position_embeddings,
        }
    }

    // ── Named presets ─────────────────────────────────────────────────────────

    /// CodeLlama-7B — general-purpose code model (no infilling)
    pub fn codellama_7b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            max_position_embeddings: 16384,
            infilling: false,
            ..Self::default()
        }
    }

    /// CodeLlama-13B
    pub fn codellama_13b() -> Self {
        Self {
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: 40,
            max_position_embeddings: 16384,
            infilling: false,
            ..Self::default()
        }
    }

    /// CodeLlama-34B — linear RoPE scaling for 100K token context
    ///
    /// The scaling factor of 4.0 extends the native 16384 training context
    /// to ~100K tokens via linear frequency division.
    pub fn codellama_34b() -> Self {
        Self {
            hidden_size: 8192,
            intermediate_size: 22016,
            num_hidden_layers: 48,
            num_attention_heads: 64,
            num_key_value_heads: 8, // GQA
            max_position_embeddings: 16384,
            rope_scaling: Some(RopeScalingConfig {
                scaling_type: RopeScalingType::Linear,
                factor: 4.0, // 16384 * 4 ≈ 65536; further extrapolation to 100K
            }),
            infilling: false,
            ..Self::default()
        }
    }

    /// CodeLlama-70B — 4K native context, repository-level tasks
    pub fn codellama_70b() -> Self {
        Self {
            hidden_size: 8192,
            intermediate_size: 28672,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: 8, // GQA
            max_position_embeddings: 4096,
            infilling: false,
            ..Self::default()
        }
    }

    /// CodeLlama-7B-Instruct — instruction-tuned for chat-style code assistance
    pub fn codellama_7b_instruct() -> Self {
        Self {
            infilling: true, // Instruct variant keeps FIM capability
            ..Self::codellama_7b()
        }
    }

    /// Create from HuggingFace model name
    pub fn from_pretrained_name(name: &str) -> Option<Self> {
        match name {
            "codellama/CodeLlama-7b-hf" | "codellama-7b" => Some(Self::codellama_7b()),
            "codellama/CodeLlama-13b-hf" | "codellama-13b" => Some(Self::codellama_13b()),
            "codellama/CodeLlama-34b-hf" | "codellama-34b" => Some(Self::codellama_34b()),
            "codellama/CodeLlama-70b-hf" | "codellama-70b" => Some(Self::codellama_70b()),
            "codellama/CodeLlama-7b-Instruct-hf" | "codellama-7b-instruct" => {
                Some(Self::codellama_7b_instruct())
            },
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codellama_default_vocab_size() {
        let cfg = CodeLlamaConfig::default();
        assert_eq!(cfg.vocab_size, 32016);
    }

    #[test]
    fn test_codellama_default_hidden_size() {
        let cfg = CodeLlamaConfig::default();
        assert_eq!(cfg.hidden_size, 4096);
    }

    #[test]
    fn test_codellama_default_num_hidden_layers() {
        let cfg = CodeLlamaConfig::default();
        assert_eq!(cfg.num_hidden_layers, 32);
    }

    #[test]
    fn test_codellama_default_num_attention_heads() {
        let cfg = CodeLlamaConfig::default();
        assert_eq!(cfg.num_attention_heads, 32);
    }

    #[test]
    fn test_codellama_default_max_position_embeddings() {
        let cfg = CodeLlamaConfig::default();
        assert_eq!(cfg.max_position_embeddings, 16384);
    }

    #[test]
    fn test_codellama_default_infilling() {
        let cfg = CodeLlamaConfig::default();
        assert!(cfg.infilling);
    }

    #[test]
    fn test_codellama_default_programming_languages() {
        let cfg = CodeLlamaConfig::default();
        assert!(!cfg.programming_languages.is_empty());
        assert!(cfg.programming_languages.contains(&"python".to_string()));
    }

    #[test]
    fn test_codellama_default_rope_scaling_none() {
        let cfg = CodeLlamaConfig::default();
        assert!(cfg.rope_scaling.is_none());
    }

    #[test]
    fn test_codellama_validate_passes_default() {
        let cfg = CodeLlamaConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_codellama_validate_fails_zero_vocab_size() {
        let cfg = CodeLlamaConfig {
            vocab_size: 0,
            ..CodeLlamaConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_codellama_validate_fails_hidden_not_divisible_by_heads() {
        let cfg = CodeLlamaConfig {
            hidden_size: 4096,
            num_attention_heads: 7,
            num_key_value_heads: 7,
            ..CodeLlamaConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_codellama_validate_fails_heads_not_divisible_by_kv_heads() {
        let cfg = CodeLlamaConfig {
            num_attention_heads: 32,
            num_key_value_heads: 7,
            ..CodeLlamaConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_codellama_7b_preset() {
        let cfg = CodeLlamaConfig::codellama_7b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert!(!cfg.infilling);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_codellama_13b_preset() {
        let cfg = CodeLlamaConfig::codellama_13b();
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_codellama_34b_preset_with_rope_scaling() {
        let cfg = CodeLlamaConfig::codellama_34b();
        assert_eq!(cfg.hidden_size, 8192);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert!(cfg.rope_scaling.is_some());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_codellama_34b_rope_scaling_type() {
        let cfg = CodeLlamaConfig::codellama_34b();
        if let Some(ref scaling) = cfg.rope_scaling {
            assert_eq!(scaling.scaling_type, RopeScalingType::Linear);
            assert!((scaling.factor - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_codellama_70b_preset() {
        let cfg = CodeLlamaConfig::codellama_70b();
        assert_eq!(cfg.hidden_size, 8192);
        assert_eq!(cfg.num_hidden_layers, 80);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_codellama_head_dim() {
        let cfg = CodeLlamaConfig::codellama_7b();
        assert_eq!(cfg.head_dim(), 4096 / 32);
    }

    #[test]
    fn test_codellama_uses_gqa_7b_false() {
        let cfg = CodeLlamaConfig::codellama_7b();
        assert!(!cfg.uses_gqa());
    }

    #[test]
    fn test_codellama_uses_gqa_34b_true() {
        let cfg = CodeLlamaConfig::codellama_34b();
        assert!(cfg.uses_gqa());
    }

    #[test]
    fn test_codellama_effective_max_context_no_scaling() {
        let cfg = CodeLlamaConfig::codellama_7b();
        assert_eq!(cfg.effective_max_context(), cfg.max_position_embeddings);
    }

    #[test]
    fn test_codellama_effective_max_context_with_scaling() {
        let cfg = CodeLlamaConfig::codellama_34b();
        let effective = cfg.effective_max_context();
        assert!(effective > cfg.max_position_embeddings);
    }

    #[test]
    fn test_codellama_from_pretrained_name_7b() {
        let cfg = CodeLlamaConfig::from_pretrained_name("codellama-7b");
        assert!(cfg.is_some());
    }

    #[test]
    fn test_codellama_from_pretrained_name_unknown() {
        let cfg = CodeLlamaConfig::from_pretrained_name("unknown");
        assert!(cfg.is_none());
    }

    #[test]
    fn test_codellama_architecture_name() {
        let cfg = CodeLlamaConfig::default();
        assert_eq!(cfg.architecture(), "CodeLlama");
    }

    #[test]
    fn test_codellama_7b_instruct_preserves_infilling() {
        let cfg = CodeLlamaConfig::codellama_7b_instruct();
        assert!(cfg.infilling);
    }

    #[test]
    fn test_codellama_lcg_values_in_range() {
        let mut s = 42u64;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (s % 1000) as f32 / 1000.0;
        assert!((0.0..1.0).contains(&v));
    }
}
