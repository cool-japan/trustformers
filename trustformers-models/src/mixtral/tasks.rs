use crate::mixtral::{config::MixtralConfig, model::MixtralForCausalLM};
use trustformers_core::{errors::Result, tensor::Tensor, traits::Model};

/// High-level causal language modeling task wrapper for Mixtral.
pub struct MixtralCausalLMTask {
    model: MixtralForCausalLM,
}

impl MixtralCausalLMTask {
    pub fn new(config: MixtralConfig) -> Result<Self> {
        let model = MixtralForCausalLM::new(config)?;
        Ok(Self { model })
    }

    /// Forward pass: token IDs → logits [batch=1, seq_len, vocab_size]
    pub fn forward(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        self.model.forward(input_ids)
    }

    /// Access underlying model.
    pub fn model(&self) -> &MixtralForCausalLM {
        &self.model
    }

    /// Access config.
    pub fn config(&self) -> &MixtralConfig {
        self.model.get_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mixtral::config::MixtralConfig;
    use trustformers_core::traits::Model;

    fn small_config() -> MixtralConfig {
        MixtralConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_local_experts: 4,
            num_experts_per_tok: 2,
            sliding_window: None,
            vocab_size: 256,
            max_position_embeddings: 64,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-5,
            hidden_act: "silu".to_string(),
            router_aux_loss_coef: 0.02,
            model_type: "mixtral".to_string(),
        }
    }

    // ── 1. MixtralCausalLMTask constructs successfully ────────────────────────

    #[test]
    fn test_task_construction() {
        let result = MixtralCausalLMTask::new(small_config());
        assert!(result.is_ok(), "MixtralCausalLMTask must construct");
    }

    // ── 2. config accessor returns the correct config ─────────────────────────

    #[test]
    fn test_config_accessor() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let task = MixtralCausalLMTask::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        assert_eq!(task.config().vocab_size, vocab);
    }

    // ── 3. model() accessor returns the inner model ───────────────────────────

    #[test]
    fn test_model_accessor_num_parameters() {
        let task =
            MixtralCausalLMTask::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        assert!(
            task.model().num_parameters() > 0,
            "model must have parameters"
        );
    }

    // ── 4. forward produces non-empty output ─────────────────────────────────

    #[test]
    fn test_forward_nonempty_output() {
        let task =
            MixtralCausalLMTask::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let result = task.forward(vec![1u32, 2, 3]);
        assert!(result.is_ok(), "forward must succeed");
    }

    // ── 5. num_local_experts field in config ──────────────────────────────────

    #[test]
    fn test_config_num_experts() {
        let cfg = small_config();
        assert_eq!(cfg.num_local_experts, 4);
        assert_eq!(cfg.num_experts_per_tok, 2);
    }

    // ── 6. config has correct model_type ─────────────────────────────────────

    #[test]
    fn test_config_model_type() {
        let cfg = small_config();
        assert_eq!(cfg.model_type, "mixtral");
    }

    // ── 7. router_aux_loss_coef is set correctly ──────────────────────────────

    #[test]
    fn test_config_router_aux_loss_coef() {
        let cfg = small_config();
        assert!((cfg.router_aux_loss_coef - 0.02).abs() < 1e-6);
    }

    // ── 8. forward output tensor is non-empty ────────────────────────────────

    #[test]
    fn test_forward_tensor_non_empty() {
        let task =
            MixtralCausalLMTask::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        if let Ok(Tensor::F32(arr)) = task.forward(vec![0u32, 1]).as_ref() {
            assert!(!arr.is_empty(), "output tensor must be non-empty");
        }
    }

    // ── 9. forward output is finite ──────────────────────────────────────────

    #[test]
    fn test_forward_output_finite() {
        let task =
            MixtralCausalLMTask::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        if let Ok(Tensor::F32(arr)) = task.forward(vec![1u32]).as_ref() {
            for &v in arr.iter() {
                assert!(v.is_finite(), "logit {v} must be finite");
            }
        }
    }

    // ── 10. forward is deterministic ─────────────────────────────────────────

    #[test]
    fn test_forward_deterministic() {
        let task =
            MixtralCausalLMTask::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let ids = vec![1u32, 2, 3];
        let r1 = task.forward(ids.clone());
        let r2 = task.forward(ids);
        if let (Ok(a), Ok(b)) = (r1, r2) {
            if let (Tensor::F32(arr_a), Tensor::F32(arr_b)) = (&a, &b) {
                let v1: Vec<f32> = arr_a.iter().copied().collect();
                let v2: Vec<f32> = arr_b.iter().copied().collect();
                assert_eq!(v1, v2, "forward must be deterministic");
            }
        }
    }

    // ── 11. num_parameters > 0 via model accessor ────────────────────────────

    #[test]
    fn test_num_parameters_via_model() {
        let task =
            MixtralCausalLMTask::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let params = task.model().num_parameters();
        assert!(params > 0, "model must have non-zero parameters");
    }

    // ── 12. default config has 8 experts ─────────────────────────────────────

    #[test]
    fn test_default_config_experts() {
        let cfg = MixtralConfig::default();
        assert_eq!(cfg.num_local_experts, 8);
        assert_eq!(cfg.num_experts_per_tok, 2);
    }
}
