use crate::granite::config::{GraniteConfig, GraniteError};
use crate::granite::model::{DenseLayer, GraniteModel};

// ─── LM Head ─────────────────────────────────────────────────────────────────

/// Linear projection from `hidden_size` to `vocab_size` (no bias by default).
#[derive(Debug, Clone)]
pub struct GraniteLmHead {
    weight: Vec<f32>,
    in_features: usize,
    out_features: usize,
}

impl GraniteLmHead {
    fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weight: vec![0.0_f32; out_features * in_features],
            in_features,
            out_features,
        }
    }

    /// Project a single hidden vector to logits.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, GraniteError> {
        if x.len() != self.in_features {
            return Err(GraniteError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }
        let mut out = vec![0.0_f32; self.out_features];
        for o in 0..self.out_features {
            let row_start = o * self.in_features;
            let acc: f32 = self.weight[row_start..row_start + self.in_features]
                .iter()
                .zip(x.iter())
                .map(|(w, v)| w * v)
                .sum();
            out[o] = acc;
        }
        Ok(out)
    }
}

// ─── Causal Language Model ────────────────────────────────────────────────────

/// Granite model with a causal language-modelling head.
///
/// The final logits are scaled by `config.logits_scaling` before they are
/// returned or used for greedy token selection.
#[derive(Debug, Clone)]
pub struct GraniteForCausalLm {
    model: GraniteModel,
    lm_head: GraniteLmHead,
    logits_scaling: f32,
}

impl GraniteForCausalLm {
    /// Construct from config.
    pub fn new(config: &GraniteConfig) -> Result<Self, GraniteError> {
        config.validate()?;
        let model = GraniteModel::new(config)?;
        let lm_head = GraniteLmHead::new(config.hidden_size, config.vocab_size);
        Ok(Self {
            model,
            lm_head,
            logits_scaling: config.logits_scaling,
        })
    }

    /// Compute scaled logits for the last position in a token sequence.
    ///
    /// Returns a `vocab_size`-length vector.
    pub fn forward_last_logits(&self, token_ids: &[u32]) -> Result<Vec<f32>, GraniteError> {
        if token_ids.is_empty() {
            return Err(GraniteError::EmptyInput);
        }
        let hidden = self.model.forward(token_ids)?;
        let seq_len = token_ids.len();
        let hidden_size = self.model.hidden_size();
        let last = &hidden[(seq_len - 1) * hidden_size..seq_len * hidden_size];
        let mut logits = self.lm_head.forward(last)?;
        for v in &mut logits {
            *v *= self.logits_scaling;
        }
        Ok(logits)
    }

    /// Greedy decoding: iteratively pick the argmax token until `max_new`
    /// tokens are generated.
    ///
    /// `vocab_size` is used as an upper bound to prevent OOB token ids.
    pub fn generate_greedy(
        &self,
        prompt: &[u32],
        max_new: usize,
        vocab_size: usize,
    ) -> Result<Vec<u32>, GraniteError> {
        if prompt.is_empty() {
            return Err(GraniteError::EmptyInput);
        }
        let mut tokens: Vec<u32> = prompt.to_vec();
        for _ in 0..max_new {
            let logits = self.forward_last_logits(&tokens)?;
            let next = argmax_token(&logits, vocab_size)?;
            tokens.push(next);
        }
        // Return only the newly generated tokens.
        Ok(tokens[prompt.len()..].to_vec())
    }

    /// The logit scaling factor from config.
    pub fn logits_scaling(&self) -> f32 {
        self.logits_scaling
    }
}

/// Return the argmax index from `logits`, bounded by `vocab_size`.
fn argmax_token(logits: &[f32], vocab_size: usize) -> Result<u32, GraniteError> {
    if logits.is_empty() {
        return Err(GraniteError::EmptyInput);
    }
    let effective_len = logits.len().min(vocab_size);
    let best = logits[..effective_len]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32);
    best.ok_or(GraniteError::EmptyInput)
}

// ─── Sequence Classification ─────────────────────────────────────────────────

/// Granite model with a sequence-classification head.
///
/// Pools the hidden state at the last token position and projects to `num_labels`.
#[derive(Debug, Clone)]
pub struct GraniteForSequenceClassification {
    model: GraniteModel,
    classifier: DenseLayer,
    num_labels: usize,
}

impl GraniteForSequenceClassification {
    /// Construct from config and the desired number of output labels.
    pub fn new(config: &GraniteConfig, num_labels: usize) -> Result<Self, GraniteError> {
        config.validate()?;
        if num_labels == 0 {
            return Err(GraniteError::InvalidConfig(
                "num_labels must be greater than 0".to_string(),
            ));
        }
        let model = GraniteModel::new(config)?;
        let classifier = DenseLayer::new(config.hidden_size, num_labels, true, 0xAAAA);
        Ok(Self {
            model,
            classifier,
            num_labels,
        })
    }

    /// Forward pass returning classification logits of length `num_labels`.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>, GraniteError> {
        if token_ids.is_empty() {
            return Err(GraniteError::EmptyInput);
        }
        let hidden = self.model.forward(token_ids)?;
        let seq_len = token_ids.len();
        let hidden_size = self.model.hidden_size();
        // Pool at the last token.
        let last = &hidden[(seq_len - 1) * hidden_size..seq_len * hidden_size];
        self.classifier.forward(last)
    }

    /// The number of classification labels.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::granite::config::GraniteConfig;

    fn small_config() -> GraniteConfig {
        GraniteConfig {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_bias: false,
            mlp_bias: false,
            tie_word_embeddings: false,
            hidden_act: "silu".to_string(),
            attention_dropout: 0.0,
            initializer_range: 0.02,
            embedding_multiplier: 1.0,
            logits_scaling: 1.0,
            residual_multiplier: 1.0,
            attention_multiplier: 1.0,
        }
    }

    // ── 1. GraniteForCausalLm constructs without error ────────────────────────

    #[test]
    fn test_causal_lm_construction() {
        let cfg = small_config();
        let result = GraniteForCausalLm::new(&cfg);
        assert!(result.is_ok(), "GraniteForCausalLm must construct");
    }

    // ── 2. logits_scaling accessor returns config value ───────────────────────

    #[test]
    fn test_causal_lm_logits_scaling() {
        let mut cfg = small_config();
        cfg.logits_scaling = 0.5;
        let model = GraniteForCausalLm::new(&cfg).unwrap_or_else(|_| panic!("init failed"));
        assert!((model.logits_scaling() - 0.5).abs() < 1e-6);
    }

    // ── 3. forward_last_logits returns vocab-sized vector ─────────────────────

    #[test]
    fn test_forward_last_logits_length() {
        let cfg = small_config();
        let model = GraniteForCausalLm::new(&cfg).unwrap_or_else(|_| panic!("init failed"));
        let result = model.forward_last_logits(&[1u32, 2, 3]);
        assert!(result.is_ok(), "forward_last_logits must succeed");
        let logits = result.unwrap_or_else(|_| panic!("forward failed"));
        assert_eq!(
            logits.len(),
            cfg.vocab_size,
            "logits length must equal vocab_size"
        );
    }

    // ── 4. forward_last_logits on empty input returns error ───────────────────

    #[test]
    fn test_forward_last_logits_empty_input_error() {
        let cfg = small_config();
        let model = GraniteForCausalLm::new(&cfg).unwrap_or_else(|_| panic!("init failed"));
        let err = model.forward_last_logits(&[]);
        assert!(
            matches!(err, Err(GraniteError::EmptyInput)),
            "empty input must return EmptyInput error"
        );
    }

    // ── 5. generate_greedy returns correct number of tokens ───────────────────

    #[test]
    fn test_generate_greedy_token_count() {
        let cfg = small_config();
        let model = GraniteForCausalLm::new(&cfg).unwrap_or_else(|_| panic!("init failed"));
        let result = model.generate_greedy(&[1u32, 2], 3, cfg.vocab_size);
        assert!(result.is_ok(), "generate_greedy must succeed");
        let tokens = result.unwrap_or_else(|_| panic!("generate failed"));
        assert_eq!(tokens.len(), 3, "must generate exactly 3 new tokens");
    }

    // ── 6. generate_greedy on empty prompt returns error ─────────────────────

    #[test]
    fn test_generate_greedy_empty_prompt_error() {
        let cfg = small_config();
        let model = GraniteForCausalLm::new(&cfg).unwrap_or_else(|_| panic!("init failed"));
        let err = model.generate_greedy(&[], 1, cfg.vocab_size);
        assert!(
            matches!(err, Err(GraniteError::EmptyInput)),
            "empty prompt must return EmptyInput"
        );
    }

    // ── 7. generate_greedy zero new tokens returns empty vec ─────────────────

    #[test]
    fn test_generate_greedy_zero_new_tokens() {
        let cfg = small_config();
        let model = GraniteForCausalLm::new(&cfg).unwrap_or_else(|_| panic!("init failed"));
        let tokens = model.generate_greedy(&[1u32], 0, cfg.vocab_size).unwrap_or_default();
        assert!(tokens.is_empty(), "zero new tokens must return empty vec");
    }

    // ── 8. generated tokens are within vocab bounds ───────────────────────────

    #[test]
    fn test_generate_tokens_within_vocab() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let model = GraniteForCausalLm::new(&cfg).unwrap_or_else(|_| panic!("init failed"));
        if let Ok(tokens) = model.generate_greedy(&[1u32, 2], 5, vocab) {
            for &t in &tokens {
                assert!((t as usize) < vocab, "token {t} must be within vocab");
            }
        }
    }

    // ── 9. GraniteForSequenceClassification construction ──────────────────────

    #[test]
    fn test_seq_cls_construction() {
        let cfg = small_config();
        let result = GraniteForSequenceClassification::new(&cfg, 3);
        assert!(
            result.is_ok(),
            "GraniteForSequenceClassification must construct"
        );
    }

    // ── 10. GraniteForSequenceClassification zero labels error ────────────────

    #[test]
    fn test_seq_cls_zero_labels_error() {
        let cfg = small_config();
        let err = GraniteForSequenceClassification::new(&cfg, 0);
        assert!(err.is_err(), "zero labels must return error");
    }

    // ── 11. num_labels accessor is correct ────────────────────────────────────

    #[test]
    fn test_seq_cls_num_labels_accessor() {
        let cfg = small_config();
        let model = GraniteForSequenceClassification::new(&cfg, 5)
            .unwrap_or_else(|_| panic!("init failed"));
        assert_eq!(model.num_labels(), 5);
    }

    // ── 12. seq cls forward returns correct length ────────────────────────────

    #[test]
    fn test_seq_cls_forward_length() {
        let cfg = small_config();
        let model = GraniteForSequenceClassification::new(&cfg, 4)
            .unwrap_or_else(|_| panic!("init failed"));
        let result = model.forward(&[1u32, 2, 3]);
        assert!(result.is_ok(), "seq cls forward must succeed");
        let logits = result.unwrap_or_else(|_| panic!("forward failed"));
        assert_eq!(logits.len(), 4, "must return 4 logits for 4 labels");
    }

    // ── 13. seq cls forward empty input error ────────────────────────────────

    #[test]
    fn test_seq_cls_forward_empty_input_error() {
        let cfg = small_config();
        let model = GraniteForSequenceClassification::new(&cfg, 2)
            .unwrap_or_else(|_| panic!("init failed"));
        let err = model.forward(&[]);
        assert!(
            matches!(err, Err(GraniteError::EmptyInput)),
            "empty input must return EmptyInput"
        );
    }

    // ── 14. logits are finite ─────────────────────────────────────────────────

    #[test]
    fn test_causal_lm_logits_finite() {
        let cfg = small_config();
        let model = GraniteForCausalLm::new(&cfg).unwrap_or_else(|_| panic!("init failed"));
        if let Ok(logits) = model.forward_last_logits(&[0u32, 1]) {
            for &v in &logits {
                assert!(v.is_finite(), "logit {v} must be finite");
            }
        }
    }

    // ── 15. logits_scaling = 2.0 doubles logit values ─────────────────────────

    #[test]
    fn test_logits_scaling_applied() {
        let mut cfg1 = small_config();
        cfg1.logits_scaling = 1.0;
        let mut cfg2 = small_config();
        cfg2.logits_scaling = 2.0;

        let m1 = GraniteForCausalLm::new(&cfg1).unwrap_or_else(|_| panic!("init failed"));
        let m2 = GraniteForCausalLm::new(&cfg2).unwrap_or_else(|_| panic!("init failed"));

        if let (Ok(l1), Ok(l2)) = (
            m1.forward_last_logits(&[1u32]),
            m2.forward_last_logits(&[1u32]),
        ) {
            // l2 should be approximately 2x l1
            for (&v1, &v2) in l1.iter().zip(l2.iter()) {
                if v1.abs() > 1e-6 {
                    let ratio = v2 / v1;
                    assert!(
                        (ratio - 2.0).abs() < 0.01,
                        "scaling=2.0 should double logits: ratio {ratio}"
                    );
                }
            }
        }
    }

    // ── 16. GraniteLmHead forward via generate_greedy is deterministic ────────

    #[test]
    fn test_generate_greedy_deterministic() {
        let cfg = small_config();
        let model = GraniteForCausalLm::new(&cfg).unwrap_or_else(|_| panic!("init failed"));
        let prompt = vec![1u32, 2, 3];
        let r1 = model.generate_greedy(&prompt, 3, cfg.vocab_size).unwrap_or_default();
        let r2 = model.generate_greedy(&prompt, 3, cfg.vocab_size).unwrap_or_default();
        assert_eq!(r1, r2, "generation must be deterministic");
    }

    // ── 17. validate fails on invalid granite config ──────────────────────────

    #[test]
    fn test_causal_lm_rejects_invalid_config() {
        let mut cfg = small_config();
        cfg.vocab_size = 0;
        let result = GraniteForCausalLm::new(&cfg);
        assert!(result.is_err(), "invalid config must be rejected");
    }

    // ── 18. seq cls forward output is finite ─────────────────────────────────

    #[test]
    fn test_seq_cls_forward_finite() {
        let cfg = small_config();
        let model = GraniteForSequenceClassification::new(&cfg, 3)
            .unwrap_or_else(|_| panic!("init failed"));
        if let Ok(logits) = model.forward(&[0u32, 1]) {
            for &v in &logits {
                assert!(v.is_finite(), "classification logit {v} must be finite");
            }
        }
    }
}
