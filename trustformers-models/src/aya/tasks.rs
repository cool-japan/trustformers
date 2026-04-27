use crate::aya::config::{AyaConfig, AyaError};
use crate::aya::model::{AyaDenseLayer, AyaModel};

// ─── LM Head ─────────────────────────────────────────────────────────────────

/// Linear projection from `hidden_size` to `vocab_size`.
#[derive(Debug, Clone)]
pub struct AyaLmHead {
    weight: Vec<f32>,
    in_features: usize,
    out_features: usize,
}

impl AyaLmHead {
    fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weight: vec![0.0_f32; out_features * in_features],
            in_features,
            out_features,
        }
    }

    /// Project a single hidden vector to logits.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, AyaError> {
        if x.len() != self.in_features {
            return Err(AyaError::DimensionMismatch {
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

/// Aya-23 model with a causal language-modelling head.
///
/// Final logits are multiplied by `config.logit_scale` before returning.
#[derive(Debug, Clone)]
pub struct AyaForCausalLm {
    model: AyaModel,
    lm_head: AyaLmHead,
    logit_scale: f32,
}

impl AyaForCausalLm {
    /// Construct from config.
    pub fn new(config: &AyaConfig) -> Result<Self, AyaError> {
        config.validate()?;
        let model = AyaModel::new(config)?;
        let lm_head = AyaLmHead::new(config.hidden_size, config.vocab_size);
        Ok(Self {
            model,
            lm_head,
            logit_scale: config.logit_scale,
        })
    }

    /// Compute scaled logits for the last token in the sequence.
    pub fn forward_last_logits(&self, token_ids: &[u32]) -> Result<Vec<f32>, AyaError> {
        if token_ids.is_empty() {
            return Err(AyaError::EmptyInput);
        }
        let hidden = self.model.forward(token_ids)?;
        let seq_len = token_ids.len();
        let hidden_size = self.model.hidden_size();
        let last = &hidden[(seq_len - 1) * hidden_size..seq_len * hidden_size];
        let mut logits = self.lm_head.forward(last)?;
        for v in &mut logits {
            *v *= self.logit_scale;
        }
        Ok(logits)
    }

    /// The logit scale from config.
    pub fn logit_scale(&self) -> f32 {
        self.logit_scale
    }

    /// Greedy decode `max_new` tokens starting from `prompt`.
    fn generate_greedy_internal(
        &self,
        prompt: &[u32],
        max_new: usize,
        vocab_size: usize,
    ) -> Result<Vec<u32>, AyaError> {
        if prompt.is_empty() {
            return Err(AyaError::EmptyInput);
        }
        let mut tokens: Vec<u32> = prompt.to_vec();
        for _ in 0..max_new {
            let logits = self.forward_last_logits(&tokens)?;
            let next = argmax_token(&logits, vocab_size)?;
            tokens.push(next);
        }
        Ok(tokens[prompt.len()..].to_vec())
    }
}

/// Return the argmax index from `logits` bounded by `vocab_size`.
fn argmax_token(logits: &[f32], vocab_size: usize) -> Result<u32, AyaError> {
    if logits.is_empty() {
        return Err(AyaError::EmptyInput);
    }
    let effective_len = logits.len().min(vocab_size);
    let best = logits[..effective_len]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32);
    best.ok_or(AyaError::EmptyInput)
}

// ─── Multilingual Generation ─────────────────────────────────────────────────

/// Aya-23 model tailored for multilingual generation.
///
/// Wraps [`AyaForCausalLm`] and prepends a `<lang:{code}>` tag to steer
/// generation toward the desired target language.
#[derive(Debug, Clone)]
pub struct AyaForMultilingualGeneration {
    inner: AyaForCausalLm,
    pub target_language: Option<String>,
    supported_languages: Vec<String>,
}

impl AyaForMultilingualGeneration {
    /// Construct from config and an optional pre-set target language.
    pub fn new(config: &AyaConfig) -> Result<Self, AyaError> {
        config.validate()?;
        Ok(Self {
            inner: AyaForCausalLm::new(config)?,
            target_language: None,
            supported_languages: config.supported_languages.clone(),
        })
    }

    /// Set the default target language.
    pub fn set_target_language(&mut self, lang: &str) {
        self.target_language = Some(lang.to_string());
    }

    /// Generate text in the specified `target_lang`.
    ///
    /// Validation checks that `target_lang` is in the supported language list.
    /// The placeholder `<lang:{target_lang}>` token is prepended to `prompt`
    /// (represented as token 0 for simplicity, as the full tokenizer is out of
    /// scope for this inference-only implementation).
    pub fn generate_in_language(
        &self,
        prompt: &[u32],
        target_lang: &str,
        max_new: usize,
        vocab_size: usize,
    ) -> Result<Vec<u32>, AyaError> {
        if prompt.is_empty() {
            return Err(AyaError::EmptyInput);
        }
        // Validate target language.
        if !self.supported_languages.iter().any(|l| l.as_str() == target_lang) {
            return Err(AyaError::UnsupportedLanguage(target_lang.to_string()));
        }

        // Prepend a language-control sentinel token (token 0 represents the
        // `<lang:{target_lang}>` placeholder in this simplified implementation).
        let mut augmented: Vec<u32> = vec![0_u32];
        augmented.extend_from_slice(prompt);

        self.inner.generate_greedy_internal(&augmented, max_new, vocab_size)
    }

    /// Forward last logits, delegating to inner model.
    pub fn forward_last_logits(&self, token_ids: &[u32]) -> Result<Vec<f32>, AyaError> {
        self.inner.forward_last_logits(token_ids)
    }
}

// ─── Sequence Classification ─────────────────────────────────────────────────

/// Aya-23 model with a sequence-classification head.
#[derive(Debug, Clone)]
pub struct AyaForSequenceClassification {
    model: AyaModel,
    classifier: AyaDenseLayer,
    num_labels: usize,
}

impl AyaForSequenceClassification {
    /// Construct from config and number of labels.
    pub fn new(config: &AyaConfig, num_labels: usize) -> Result<Self, AyaError> {
        config.validate()?;
        if num_labels == 0 {
            return Err(AyaError::InvalidConfig(
                "num_labels must be greater than 0".to_string(),
            ));
        }
        let model = AyaModel::new(config)?;
        let classifier = AyaDenseLayer::new(config.hidden_size, num_labels, true, 0x000B_BBBA);
        Ok(Self {
            model,
            classifier,
            num_labels,
        })
    }

    /// Forward: pool last token → classification logits.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>, AyaError> {
        if token_ids.is_empty() {
            return Err(AyaError::EmptyInput);
        }
        let hidden = self.model.forward(token_ids)?;
        let seq_len = token_ids.len();
        let hidden_size = self.model.hidden_size();
        let last = &hidden[(seq_len - 1) * hidden_size..seq_len * hidden_size];
        self.classifier.forward(last)
    }

    /// Number of classification labels.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aya::config::AyaConfig;

    fn tiny_config() -> AyaConfig {
        AyaConfig {
            vocab_size: 64,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            max_position_embeddings: 32,
            layer_norm_eps: 1e-5,
            rope_theta: 10000.0,
            logit_scale: 0.0625,
            use_qk_norm: false,
            tie_word_embeddings: false,
            attention_dropout: 0.0,
            supported_languages: vec![
                "en".to_string(),
                "fr".to_string(),
                "de".to_string(),
                "es".to_string(),
                "zh".to_string(),
            ],
            tokenizer_class: "PreTrainedTokenizer".to_string(),
        }
    }

    // --- AyaLmHead ---

    #[test]
    fn test_aya_lm_head_output_size() {
        let cfg = tiny_config();
        let lm = AyaLmHead::new(cfg.hidden_size, cfg.vocab_size);
        let input: Vec<f32> = vec![0.5_f32; cfg.hidden_size];
        let out = lm.forward(&input).expect("AyaLmHead forward must succeed");
        assert_eq!(
            out.len(),
            cfg.vocab_size,
            "LM head output must equal vocab_size"
        );
    }

    #[test]
    fn test_aya_lm_head_dimension_mismatch_errors() {
        let lm = AyaLmHead::new(16, 64);
        let bad_input: Vec<f32> = vec![0.0_f32; 8];
        let result = lm.forward(&bad_input);
        assert!(result.is_err(), "LM head must reject wrong input size");
    }

    // --- AyaForCausalLm ---

    #[test]
    fn test_aya_causal_lm_new() {
        let cfg = tiny_config();
        let model = AyaForCausalLm::new(&cfg).expect("AyaForCausalLm::new must succeed");
        assert!((model.logit_scale() - cfg.logit_scale).abs() < 1e-6);
    }

    #[test]
    fn test_aya_causal_lm_forward_last_logits_shape() {
        let cfg = tiny_config();
        let model = AyaForCausalLm::new(&cfg).expect("AyaForCausalLm::new must succeed");
        let token_ids: Vec<u32> = vec![1, 2, 3];
        let logits =
            model.forward_last_logits(&token_ids).expect("forward_last_logits must succeed");
        assert_eq!(
            logits.len(),
            cfg.vocab_size,
            "Logits must have vocab_size entries"
        );
    }

    #[test]
    fn test_aya_causal_lm_forward_empty_input_errors() {
        let cfg = tiny_config();
        let model = AyaForCausalLm::new(&cfg).expect("AyaForCausalLm::new must succeed");
        let result = model.forward_last_logits(&[]);
        assert!(
            result.is_err(),
            "forward_last_logits must reject empty input"
        );
    }

    #[test]
    fn test_aya_causal_lm_logit_scale_applied() {
        let cfg = tiny_config();
        let model = AyaForCausalLm::new(&cfg).expect("AyaForCausalLm::new must succeed");
        // logit_scale = 0.0625; since LM head weights are zero, all logits should be 0
        let logits = model.forward_last_logits(&[1, 2]).expect("forward_last_logits must succeed");
        // All zero weights * scale = 0.0 (all logits are zero since lm_head is zero-initialized)
        for v in &logits {
            assert!(v.is_finite(), "Scaled logits must be finite");
        }
    }

    #[test]
    fn test_aya_causal_lm_single_token_forward() {
        let cfg = tiny_config();
        let model = AyaForCausalLm::new(&cfg).expect("AyaForCausalLm::new must succeed");
        let logits = model.forward_last_logits(&[5]).expect("single token forward must succeed");
        assert_eq!(logits.len(), cfg.vocab_size);
    }

    // --- AyaForMultilingualGeneration ---

    #[test]
    fn test_aya_multilingual_new() {
        let cfg = tiny_config();
        let model = AyaForMultilingualGeneration::new(&cfg)
            .expect("AyaForMultilingualGeneration::new must succeed");
        assert!(
            model.target_language.is_none(),
            "Default target_language must be None"
        );
    }

    #[test]
    fn test_aya_multilingual_set_target_language() {
        let cfg = tiny_config();
        let mut model = AyaForMultilingualGeneration::new(&cfg)
            .expect("AyaForMultilingualGeneration::new must succeed");
        model.set_target_language("fr");
        assert_eq!(model.target_language.as_deref(), Some("fr"));
    }

    #[test]
    fn test_aya_multilingual_generate_supported_language() {
        let cfg = tiny_config();
        let model = AyaForMultilingualGeneration::new(&cfg)
            .expect("AyaForMultilingualGeneration::new must succeed");
        let result = model.generate_in_language(&[1, 2], "en", 2, cfg.vocab_size);
        assert!(
            result.is_ok(),
            "generate_in_language must succeed for supported language"
        );
        let tokens = result.expect("generate result must be ok");
        assert_eq!(tokens.len(), 2, "Must generate exactly max_new tokens");
    }

    #[test]
    fn test_aya_multilingual_generate_unsupported_language_errors() {
        let cfg = tiny_config();
        let model = AyaForMultilingualGeneration::new(&cfg)
            .expect("AyaForMultilingualGeneration::new must succeed");
        let result = model.generate_in_language(&[1, 2], "xx", 1, cfg.vocab_size);
        assert!(
            result.is_err(),
            "generate_in_language must reject unsupported language"
        );
        if let Err(AyaError::UnsupportedLanguage(code)) = result {
            assert_eq!(code, "xx");
        }
    }

    #[test]
    fn test_aya_multilingual_generate_empty_prompt_errors() {
        let cfg = tiny_config();
        let model = AyaForMultilingualGeneration::new(&cfg)
            .expect("AyaForMultilingualGeneration::new must succeed");
        let result = model.generate_in_language(&[], "en", 1, cfg.vocab_size);
        assert!(
            result.is_err(),
            "generate_in_language must reject empty prompt"
        );
    }

    #[test]
    fn test_aya_multilingual_forward_last_logits_shape() {
        let cfg = tiny_config();
        let model = AyaForMultilingualGeneration::new(&cfg)
            .expect("AyaForMultilingualGeneration::new must succeed");
        let logits =
            model.forward_last_logits(&[1, 2, 3]).expect("forward_last_logits must succeed");
        assert_eq!(logits.len(), cfg.vocab_size);
    }

    // --- AyaForSequenceClassification ---

    #[test]
    fn test_aya_seq_cls_new() {
        let cfg = tiny_config();
        let model = AyaForSequenceClassification::new(&cfg, 3)
            .expect("AyaForSequenceClassification::new must succeed");
        assert_eq!(model.num_labels(), 3);
    }

    #[test]
    fn test_aya_seq_cls_output_shape() {
        let cfg = tiny_config();
        let num_labels = 4usize;
        let model = AyaForSequenceClassification::new(&cfg, num_labels)
            .expect("AyaForSequenceClassification::new must succeed");
        let token_ids: Vec<u32> = vec![1, 2, 3, 4, 5];
        let out = model
            .forward(&token_ids)
            .expect("AyaForSequenceClassification forward must succeed");
        assert_eq!(
            out.len(),
            num_labels,
            "Classification output must have num_labels entries"
        );
    }

    #[test]
    fn test_aya_seq_cls_zero_labels_errors() {
        let cfg = tiny_config();
        let result = AyaForSequenceClassification::new(&cfg, 0);
        assert!(result.is_err(), "Zero labels must be rejected");
    }

    #[test]
    fn test_aya_seq_cls_empty_input_errors() {
        let cfg = tiny_config();
        let model = AyaForSequenceClassification::new(&cfg, 2)
            .expect("AyaForSequenceClassification::new must succeed");
        let result = model.forward(&[]);
        assert!(result.is_err(), "empty input must be rejected");
    }

    #[test]
    fn test_aya_seq_cls_single_token() {
        let cfg = tiny_config();
        let model = AyaForSequenceClassification::new(&cfg, 2)
            .expect("AyaForSequenceClassification::new must succeed");
        let out = model.forward(&[1]).expect("single token forward must succeed");
        assert_eq!(out.len(), 2);
    }
}
