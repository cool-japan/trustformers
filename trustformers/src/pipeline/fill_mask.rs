use crate::automodel::AutoModelType;
use crate::core::traits::{Model, Tokenizer};
use crate::error::{Result, TrustformersError};
use crate::pipeline::{BasePipeline, FillMaskOutput, Pipeline, PipelineOutput};
use crate::{AutoModel, AutoTokenizer};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// MaskPrediction — public output type for enhanced fill-mask
// ---------------------------------------------------------------------------

/// A single predicted token filling a `[MASK]` position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskPrediction {
    /// The predicted token string.
    pub token: String,
    /// Vocabulary id of the predicted token.
    pub token_id: u32,
    /// Probability score (0..1).
    pub score: f32,
    /// Full input sequence with the mask replaced by this token.
    pub sequence: String,
}

// ---------------------------------------------------------------------------
// FillMaskProcessor — pure numeric helpers (no model required)
// ---------------------------------------------------------------------------

/// Stateless helper for fill-mask post-processing arithmetic.
pub struct FillMaskProcessor;

impl FillMaskProcessor {
    /// Return every position in `token_ids` that equals `mask_token_id`.
    pub fn find_mask_positions(token_ids: &[u32], mask_token_id: u32) -> Vec<usize> {
        token_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| if id == mask_token_id { Some(i) } else { None })
            .collect()
    }

    /// For each token id in `predictions`, produce a copy of `template` where
    /// position `mask_pos` has been replaced with that prediction id.
    pub fn apply_predictions(
        template: &[u32],
        mask_pos: usize,
        predictions: &[u32],
    ) -> Vec<Vec<u32>> {
        predictions
            .iter()
            .map(|&pred| {
                let mut seq = template.to_vec();
                if mask_pos < seq.len() {
                    seq[mask_pos] = pred;
                }
                seq
            })
            .collect()
    }

    /// Numerically-stable softmax over a logit slice.
    ///
    /// Returns a probability distribution (sums to 1).
    pub fn score_to_probability(logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum < f32::EPSILON {
            return vec![1.0 / logits.len() as f32; logits.len()];
        }
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Return the top-`k` (token_id, probability) pairs sorted by probability descending.
    pub fn top_k_predictions(probs: &[f32], k: usize) -> Vec<(u32, f32)> {
        if probs.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut indexed: Vec<(u32, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i as u32, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }
}

/// Configuration for fill-mask pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillMaskConfig {
    /// Maximum sequence length
    pub max_length: usize,
    /// Mask token
    pub mask_token: String,
    /// Number of top predictions to return
    pub top_k: usize,
}

impl Default for FillMaskConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            mask_token: "[MASK]".to_string(),
            top_k: 5,
        }
    }
}

/// Pipeline for fill-mask tasks (masked language modeling)
#[derive(Clone)]
pub struct FillMaskPipeline {
    base: BasePipeline<AutoModel, AutoTokenizer>,
    mask_token: String,
    top_k: usize,
}

impl FillMaskPipeline {
    pub fn new(model: AutoModel, tokenizer: AutoTokenizer) -> Result<Self> {
        Ok(Self {
            base: BasePipeline::new(model, tokenizer),
            mask_token: "[MASK]".to_string(),
            top_k: 5,
        })
    }

    pub fn with_mask_token(mut self, token: String) -> Self {
        self.mask_token = token;
        self
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    fn fill_mask(&self, text: &str) -> Result<Vec<FillMaskOutput>> {
        // Check if mask token is present
        if !text.contains(&self.mask_token) {
            return Err(TrustformersError::invalid_input_simple(format!(
                "Input must contain mask token '{}'",
                self.mask_token
            )));
        }

        let (tokenized, mask_position) = self.locate_mask(text)?;

        // Every supported branch runs the checkpoint's real masked-LM head and
        // reads the top-k of its own softmax over the tokenizer's vocabulary.
        // Architectures without an MLM head cannot answer this question at all,
        // so they get a structured error rather than an invented word list.
        let logits = match &self.base.model.model_type {
            #[cfg(feature = "bert")]
            AutoModelType::BertForMaskedLM(model) => model.forward(tokenized)?.logits,
            #[cfg(feature = "roberta")]
            AutoModelType::RobertaForMaskedLM(model) => model.forward(tokenized)?.logits,
            #[cfg(feature = "albert")]
            AutoModelType::AlbertForMaskedLM(model) => model.forward(tokenized)?.logits,
            _ => {
                return Err(TrustformersError::feature_unavailable(
                    format!(
                        "the loaded model ({}) has no masked-language-modelling head, so it \
                         cannot predict a token for '{}'. Load a *ForMaskedLM checkpoint \
                         (BERT / RoBERTa / ALBERT).",
                        crate::core::traits::Config::architecture(&self.base.model.config),
                        self.mask_token
                    ),
                    "fill-mask",
                ));
            },
        };

        self.extract_predictions_from_logits(
            &logits,
            mask_position,
            text,
            &self.mask_token,
            self.top_k,
        )
    }

    /// Tokenize `text` and locate the mask token inside the token stream.
    ///
    /// # Errors
    ///
    /// Fails when the tokenizer's vocabulary has no entry for the configured
    /// mask token, or when the encoded sequence does not contain it.
    fn locate_mask(&self, text: &str) -> Result<(crate::core::traits::TokenizedInput, usize)> {
        let tokenized = self.base.tokenizer.encode(text)?;

        let mask_token_id = self.base.tokenizer.token_to_id(&self.mask_token).ok_or_else(|| {
            TrustformersError::invalid_input_simple(format!(
                "Mask token '{}' not found in tokenizer vocabulary",
                self.mask_token
            ))
        })?;

        let mask_position =
            tokenized.input_ids.iter().position(|&id| id == mask_token_id).ok_or_else(|| {
                TrustformersError::invalid_input_simple(
                    "Mask token not found in tokenized input".to_string(),
                )
            })?;

        Ok((tokenized, mask_position))
    }

    fn fill_mask_batch(&self, texts: &[String]) -> Result<Vec<Vec<FillMaskOutput>>> {
        texts.iter().map(|text| self.fill_mask(text)).collect()
    }

    /// Slice the mask position out of a real `[batch, seq, vocab]` logits
    /// tensor and turn it into ranked predictions.
    ///
    /// # Errors
    ///
    /// Fails when the tensor is not rank-3, when the mask position falls
    /// outside the sequence, or when the tensor is shorter than its own shape
    /// claims.
    fn extract_predictions_from_logits(
        &self,
        logits_tensor: &crate::Tensor,
        mask_position: usize,
        original_text: &str,
        mask_token: &str,
        top_k: usize,
    ) -> Result<Vec<FillMaskOutput>> {
        let logits_data = logits_tensor.data()?;

        // Ensure the tensor has the expected shape [batch_size, seq_len, vocab_size]
        let shape = logits_tensor.shape();
        if shape.len() < 3 {
            return Err(TrustformersError::runtime_error(
                "Logits tensor must have at least 3 dimensions [batch, seq, vocab]".to_string(),
            ));
        }

        let seq_len = shape[1];
        let vocab_len = shape[2];

        if mask_position >= seq_len {
            return Err(TrustformersError::invalid_input_simple(format!(
                "Mask position {} exceeds sequence length {}",
                mask_position, seq_len
            )));
        }

        // Extract logits for the mask position (batch index 0)
        let start_idx = mask_position * vocab_len;
        let end_idx = start_idx + vocab_len;

        if end_idx > logits_data.len() {
            return Err(TrustformersError::runtime_error(format!(
                "Logits tensor holds {} values but shape {:?} requires at least {}",
                logits_data.len(),
                shape,
                end_idx
            )));
        }

        let mask_logits = &logits_data[start_idx..end_idx];

        // Convert logits to predictions
        self.logits_to_predictions(mask_logits, original_text, mask_token, top_k)
    }

    /// Convert the model's own mask-position logits into ranked predictions.
    ///
    /// The probabilities are a softmax over the model's full output
    /// distribution and the token ids/strings come from the loaded tokenizer's
    /// vocabulary — nothing here is synthesised. Special tokens (`[CLS]`,
    /// `<s>`, sub-word continuations, …) are dropped because they cannot stand
    /// in for the mask in the returned sequence; the list is therefore allowed
    /// to be shorter than `top_k`, and empty when the model puts all its mass
    /// on special tokens.
    fn logits_to_predictions(
        &self,
        logits: &[f32],
        original_text: &str,
        mask_token: &str,
        top_k: usize,
    ) -> Result<Vec<FillMaskOutput>> {
        // Apply softmax to convert logits to probabilities
        let probs = self.softmax(logits);

        // Create (probability, token_id) pairs and sort by probability
        let mut prob_pairs: Vec<(f32, usize)> =
            probs.iter().enumerate().map(|(idx, &prob)| (prob, idx)).collect();

        prob_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Rank first, then keep the top_k *usable* candidates.
        let mut predictions = Vec::with_capacity(top_k);
        for (prob, token_id) in prob_pairs {
            if predictions.len() >= top_k {
                break;
            }
            let Some(token_str) = self.base.tokenizer.id_to_token(token_id as u32) else {
                continue;
            };
            if self.is_special_token(&token_str) {
                continue;
            }
            let sequence = original_text.replace(mask_token, &token_str);
            predictions.push(FillMaskOutput {
                sequence,
                score: prob,
                token: token_id as u32,
                token_str,
            });
        }

        Ok(predictions)
    }

    /// Apply softmax function to logits
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();

        exp_logits.iter().map(|&x| x / sum_exp).collect()
    }

    /// Check if a token is a special token that should be filtered out
    fn is_special_token(&self, token: &str) -> bool {
        token.starts_with('[') && token.ends_with(']')
            || token.starts_with('<') && token.ends_with('>')
            || token == self.mask_token
            || token.trim().is_empty()
            || token.contains("##") // WordPiece subword tokens
    }
}

impl Pipeline for FillMaskPipeline {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        let results = self.fill_mask(&input)?;
        Ok(PipelineOutput::FillMask(results))
    }

    fn batch(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        let batch_results = self.fill_mask_batch(&inputs)?;
        Ok(batch_results.into_iter().map(PipelineOutput::FillMask).collect())
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl crate::pipeline::AsyncPipeline for FillMaskPipeline {
    type Input = String;
    type Output = PipelineOutput;

    async fn __call_async__(&self, input: Self::Input) -> Result<Self::Output> {
        let pipeline = self.clone();
        tokio::task::spawn_blocking(move || pipeline.__call__(input))
            .await
            .map_err(|e| TrustformersError::runtime_error(e.to_string()))?
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Regression tests for the removed keyword→word fallback table.
    //
    // The old `_ =>` arm answered any non-BERT model from a hardcoded list such
    // as `("said", 0.85, 2056)`, with token ids that had nothing to do with the
    // loaded tokenizer. Both tests below fail against that implementation.
    // -----------------------------------------------------------------------

    #[cfg(feature = "bert")]
    fn tiny_vocab() -> std::collections::HashMap<String, u32> {
        // `mask` rather than `[MASK]`: the WordPiece tokenizer under test
        // lower-cases and splits on punctuation, so a bracketed token would not
        // survive encoding. The pipeline's mask token is configured to match.
        [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "mask",
            "the",
            "president",
            "said",
            "hello",
        ]
        .iter()
        .enumerate()
        .map(|(i, w)| ((*w).to_string(), i as u32))
        .collect()
    }

    #[cfg(feature = "bert")]
    fn tiny_bert_config() -> crate::models::bert::BertConfig {
        crate::models::bert::BertConfig {
            vocab_size: 9,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 32,
            ..crate::models::bert::BertConfig::default()
        }
    }

    /// A model without an MLM head must be refused outright.
    #[cfg(feature = "bert")]
    #[test]
    fn model_without_mlm_head_is_refused() {
        let tokenizer = AutoTokenizer::WordPiece(crate::tokenizers::WordPieceTokenizer::new(
            tiny_vocab(),
            true,
        ));
        let model = AutoModel::from_config(crate::AutoConfig::Bert(tiny_bert_config()))
            .expect("tiny bert should build");
        let pipeline = FillMaskPipeline::new(model, tokenizer)
            .expect("pipeline should build")
            .with_mask_token("mask".to_string());

        let Err(err) = pipeline.fill_mask("the president mask hello") else {
            panic!("a headless encoder cannot fill a mask");
        };
        let message = err.to_string();
        for fabricated in ["said", "announced", "declared", "stated", "confirmed"] {
            assert!(
                !message.contains(fabricated),
                "the error must not carry the old keyword table: {message}"
            );
        }
    }

    /// Every prediction must be a real entry of the loaded vocabulary.
    #[cfg(feature = "bert")]
    #[test]
    fn predictions_come_from_the_loaded_vocabulary() {
        let vocab = tiny_vocab();
        let vocab_size = vocab.len() as u32;
        let tokenizer =
            AutoTokenizer::WordPiece(crate::tokenizers::WordPieceTokenizer::new(vocab, true));
        let config = tiny_bert_config();
        let model = AutoModel::from_parts(
            crate::AutoConfig::Bert(config.clone()),
            crate::automodel::AutoModelType::BertForMaskedLM(
                crate::models::bert::BertForMaskedLM::new(config).expect("MLM head should build"),
            ),
        );
        let pipeline = FillMaskPipeline::new(model, tokenizer.clone())
            .expect("pipeline should build")
            .with_mask_token("mask".to_string())
            .with_top_k(3);

        let predictions = pipeline
            .fill_mask("the president mask hello")
            .expect("a real MLM head must answer");
        for prediction in &predictions {
            assert!(
                prediction.token < vocab_size,
                "token id {} is outside the {vocab_size}-entry vocabulary and therefore \
                 cannot have come from the model",
                prediction.token
            );
            assert_eq!(
                tokenizer.id_to_token(prediction.token).as_deref(),
                Some(prediction.token_str.as_str()),
                "the reported string must be the vocabulary entry for the reported id"
            );
            assert!(
                (0.0..=1.0).contains(&prediction.score),
                "probabilities must come from a softmax: {}",
                prediction.score
            );
        }
    }

    // -----------------------------------------------------------------------
    // FillMaskProcessor::find_mask_positions
    // -----------------------------------------------------------------------

    #[test]
    fn find_mask_positions_single() {
        let ids = vec![101u32, 2054, 103, 2003, 102];
        let positions = FillMaskProcessor::find_mask_positions(&ids, 103);
        assert_eq!(positions, vec![2]);
    }

    #[test]
    fn find_mask_positions_none() {
        let ids = vec![101u32, 2054, 2003, 102];
        let positions = FillMaskProcessor::find_mask_positions(&ids, 103);
        assert!(positions.is_empty());
    }

    #[test]
    fn find_mask_positions_multiple() {
        let ids = vec![101u32, 103, 2003, 103, 102];
        let positions = FillMaskProcessor::find_mask_positions(&ids, 103);
        assert_eq!(positions, vec![1, 3]);
    }

    #[test]
    fn find_mask_positions_empty_input() {
        let positions = FillMaskProcessor::find_mask_positions(&[], 103);
        assert!(positions.is_empty());
    }

    #[test]
    fn find_mask_positions_all_masks() {
        let ids = vec![103u32, 103, 103];
        let positions = FillMaskProcessor::find_mask_positions(&ids, 103);
        assert_eq!(positions, vec![0, 1, 2]);
    }

    // -----------------------------------------------------------------------
    // FillMaskProcessor::apply_predictions
    // -----------------------------------------------------------------------

    #[test]
    fn apply_predictions_basic() {
        let template = vec![101u32, 103, 2003, 102];
        let predictions = vec![2054u32, 2002, 2001];
        let filled = FillMaskProcessor::apply_predictions(&template, 1, &predictions);
        assert_eq!(filled.len(), 3);
        assert_eq!(filled[0][1], 2054);
        assert_eq!(filled[1][1], 2002);
        assert_eq!(filled[2][1], 2001);
        // Other positions unchanged
        assert_eq!(filled[0][0], 101);
        assert_eq!(filled[0][2], 2003);
    }

    #[test]
    fn apply_predictions_mask_out_of_bounds() {
        let template = vec![101u32, 103];
        // mask_pos = 10 which is past the end — no panic, template returned unchanged
        let filled = FillMaskProcessor::apply_predictions(&template, 10, &[999]);
        assert_eq!(filled.len(), 1);
        assert_eq!(filled[0], template);
    }

    #[test]
    fn apply_predictions_empty_predictions() {
        let template = vec![101u32, 103, 102];
        let filled = FillMaskProcessor::apply_predictions(&template, 1, &[]);
        assert!(filled.is_empty());
    }

    // -----------------------------------------------------------------------
    // FillMaskProcessor::score_to_probability (softmax)
    // -----------------------------------------------------------------------

    #[test]
    fn score_to_probability_sums_to_one() {
        let logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let probs = FillMaskProcessor::score_to_probability(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum was {sum}");
    }

    #[test]
    fn score_to_probability_all_equal_logits() {
        let logits = vec![0.0f32; 4];
        let probs = FillMaskProcessor::score_to_probability(&logits);
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn score_to_probability_highest_logit_wins() {
        let logits = vec![0.0f32, 0.0, 10.0, 0.0];
        let probs = FillMaskProcessor::score_to_probability(&logits);
        assert!(probs[2] > probs[0]);
        assert!(probs[2] > probs[1]);
        assert!(probs[2] > probs[3]);
        assert!(probs[2] > 0.99);
    }

    #[test]
    fn score_to_probability_empty() {
        let probs = FillMaskProcessor::score_to_probability(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn score_to_probability_single_element() {
        let probs = FillMaskProcessor::score_to_probability(&[5.0]);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn score_to_probability_negative_logits() {
        let logits = vec![-10.0f32, -1.0, -5.0];
        let probs = FillMaskProcessor::score_to_probability(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // -1.0 should be the highest probability
        assert!(probs[1] > probs[0]);
        assert!(probs[1] > probs[2]);
    }

    // -----------------------------------------------------------------------
    // FillMaskProcessor::top_k_predictions
    // -----------------------------------------------------------------------

    #[test]
    fn top_k_predictions_ordering() {
        let probs = vec![0.1f32, 0.5, 0.2, 0.8, 0.05];
        let top = FillMaskProcessor::top_k_predictions(&probs, 3);
        assert_eq!(top.len(), 3);
        // Must be sorted descending
        assert!(top[0].1 >= top[1].1);
        assert!(top[1].1 >= top[2].1);
        // Top token id should be 3 (prob 0.8)
        assert_eq!(top[0].0, 3);
    }

    #[test]
    fn top_k_predictions_k_larger_than_vocab() {
        let probs = vec![0.3f32, 0.7];
        let top = FillMaskProcessor::top_k_predictions(&probs, 100);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn top_k_predictions_k_zero() {
        let probs = vec![0.3f32, 0.7];
        let top = FillMaskProcessor::top_k_predictions(&probs, 0);
        assert!(top.is_empty());
    }

    #[test]
    fn top_k_predictions_empty_probs() {
        let top = FillMaskProcessor::top_k_predictions(&[], 5);
        assert!(top.is_empty());
    }

    #[test]
    fn top_k_predictions_exact_k() {
        let probs = vec![0.1f32, 0.2, 0.3, 0.4];
        let top = FillMaskProcessor::top_k_predictions(&probs, 2);
        assert_eq!(top.len(), 2);
        // Top two are token_id 3 (0.4) and token_id 2 (0.3)
        assert_eq!(top[0].0, 3);
        assert_eq!(top[1].0, 2);
    }

    // -----------------------------------------------------------------------
    // MaskPrediction struct
    // -----------------------------------------------------------------------

    #[test]
    fn mask_prediction_fields() {
        let pred = MaskPrediction {
            token: "cat".to_string(),
            token_id: 4231,
            score: 0.92,
            sequence: "The cat sat on the mat.".to_string(),
        };
        assert_eq!(pred.token, "cat");
        assert_eq!(pred.token_id, 4231);
        assert!((pred.score - 0.92).abs() < 1e-6);
        assert!(pred.sequence.contains("cat"));
    }

    #[test]
    fn mask_prediction_serde_roundtrip() {
        let pred = MaskPrediction {
            token: "dog".to_string(),
            token_id: 3914,
            score: 0.75,
            sequence: "The dog runs fast.".to_string(),
        };
        let json = serde_json::to_string(&pred).expect("serialize");
        let back: MaskPrediction = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.token, pred.token);
        assert_eq!(back.token_id, pred.token_id);
        assert!((back.score - pred.score).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // End-to-end: softmax + top_k together
    // -----------------------------------------------------------------------

    #[test]
    fn softmax_then_top_k_pipeline() {
        let logits = vec![0.5f32, 1.5, 0.2, 3.0, -1.0, 0.0];
        let probs = FillMaskProcessor::score_to_probability(&logits);
        let top = FillMaskProcessor::top_k_predictions(&probs, 2);
        // Token id 3 has the highest logit (3.0) so must be top-1
        assert_eq!(top[0].0, 3);
        assert_eq!(top.len(), 2);
        // Probabilities should sum to something less than 1 (only top-2 returned)
        assert!(top[0].1 > top[1].1);
    }

    #[test]
    fn find_then_apply_then_top_k() {
        let template = vec![101u32, 103, 2003, 2035, 102]; // [CLS] [MASK] is all [SEP]
        let mask_id = 103u32;
        let positions = FillMaskProcessor::find_mask_positions(&template, mask_id);
        assert_eq!(positions.len(), 1);
        let logits = vec![0.0f32; 30522]; // BERT vocab size
                                          // Set logit of token 2023 to be highest
        let mut logits_mut = logits;
        logits_mut[2023] = 10.0;
        let probs = FillMaskProcessor::score_to_probability(&logits_mut);
        let top = FillMaskProcessor::top_k_predictions(&probs, 3);
        assert_eq!(top[0].0, 2023);
        let filled = FillMaskProcessor::apply_predictions(&template, positions[0], &[top[0].0]);
        assert_eq!(filled[0][positions[0]], 2023);
    }

    #[test]
    fn multiple_masks_independent_positions() {
        let template = vec![101u32, 103, 2003, 103, 102];
        let positions = FillMaskProcessor::find_mask_positions(&template, 103);
        assert_eq!(positions.len(), 2);
        // Each mask position can be filled independently
        let p1 = FillMaskProcessor::apply_predictions(&template, positions[0], &[500, 600]);
        let p2 = FillMaskProcessor::apply_predictions(&template, positions[1], &[700, 800]);
        assert_eq!(p1[0][1], 500);
        assert_eq!(p1[1][1], 600);
        assert_eq!(p2[0][3], 700);
        assert_eq!(p2[1][3], 800);
    }
}
