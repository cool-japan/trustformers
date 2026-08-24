//! Pure-Rust scoring for the classification pipeline.
//!
//! This replaces the hardcoded `[("POSITIVE", 0.7), ("NEGATIVE", 0.3)]` the
//! `text-classification` pipeline used to return for *every* input, without
//! ever running a model: the numbers were the same for "I loved it" and "I
//! hated it", and the labels were the same whatever the checkpoint's
//! `id2label` said.
//!
//! Free of the Python C API so it is unit-testable with a plain `cargo test`
//! (the test binary does not link `libpython`; see `models/weights.rs` for the
//! same split).

use trustformers_core::errors::{runtime_error, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Model, TokenizedInput};
use trustformers_models::bert::BertForSequenceClassification;

/// One scored class, as the pipeline reports it to Python.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScoredLabel {
    /// The class name (`id2label` from the checkpoint, or `LABEL_n`).
    pub label: String,
    /// The softmax probability of that class.
    pub score: f32,
}

/// Numerically stable softmax over one row of logits.
///
/// Shifting by the row maximum keeps `exp` in range for the large logits a
/// real classification head produces.
///
/// # Errors
///
/// Fails on an empty row, or on a row whose values are not finite (a `NaN`
/// logit would otherwise poison the whole distribution silently).
pub(crate) fn softmax(row: &[f32]) -> Result<Vec<f32>, TrustformersError> {
    if row.is_empty() {
        return Err(runtime_error(
            "cannot take a softmax over an empty logits row".to_string(),
        ));
    }
    if let Some(bad) = row.iter().find(|value| !value.is_finite()) {
        return Err(runtime_error(format!(
            "logits row holds the non-finite value {bad}"
        )));
    }

    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<f32> = row.iter().map(|&value| (value - max).exp()).collect();
    let total: f32 = exponentials.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return Err(runtime_error(
            "softmax denominator is not a positive finite number".to_string(),
        ));
    }
    Ok(exponentials.into_iter().map(|value| value / total).collect())
}

/// The single row of class logits held by a `[num_labels]` or
/// `[1, num_labels]` classification output.
///
/// # Errors
///
/// Fails when the tensor cannot be read, when it has no dimensions, or when it
/// holds more than one row -- the pipeline classifies one text at a time, so a
/// multi-row tensor means the caller and the model disagree about the batch.
pub(crate) fn single_logits_row(logits: &Tensor) -> Result<Vec<f32>, TrustformersError> {
    let num_classes = *logits
        .shape()
        .last()
        .ok_or_else(|| runtime_error("logits tensor has no dimensions".to_string()))?;
    if num_classes == 0 {
        return Err(runtime_error(
            "logits tensor has an empty class axis".to_string(),
        ));
    }
    let values = logits.to_vec_f32()?;
    if values.len() != num_classes {
        return Err(runtime_error(format!(
            "expected one row of {num_classes} class logits but the tensor holds {} values",
            values.len()
        )));
    }
    Ok(values)
}

/// Rank `labels` by their softmax probability under `logits`, best first.
///
/// # Errors
///
/// Fails when the label count does not match the width of the logits row --
/// the checkpoint's `id2label` and its classification head disagreeing is a
/// real configuration error, not something to paper over with `LABEL_n`
/// placeholders for the surplus classes.
pub(crate) fn rank_labels(
    logits: &[f32],
    labels: &[String],
) -> Result<Vec<ScoredLabel>, TrustformersError> {
    if logits.len() != labels.len() {
        return Err(runtime_error(format!(
            "the classification head produced {} logits but the model carries {} label names",
            logits.len(),
            labels.len()
        )));
    }

    let scores = softmax(logits)?;
    let mut ranked: Vec<ScoredLabel> = labels
        .iter()
        .zip(scores)
        .map(|(label, score)| ScoredLabel {
            label: label.clone(),
            score,
        })
        .collect();
    // Descending by score; `total_cmp` gives a total order without the
    // `partial_cmp(..).unwrap()` this crate's no-unwrap policy forbids.
    ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
    Ok(ranked)
}

/// Classify one tokenized text with a real BERT sequence-classification head.
///
/// The whole non-Python half of the `text-classification` pipeline: a real
/// forward pass through the encoder, its pooler and the loaded linear head,
/// then a real softmax over the resulting `[num_labels]` logits, ranked
/// best-first and optionally truncated to `top_k`.
///
/// Kept here, free of the Python C API, so the pipeline's actual behaviour --
/// not just its plumbing -- is unit-testable with a plain `cargo test`.
///
/// # Errors
///
/// Fails when the forward pass fails, when the head's output is not a single
/// row of class logits, when the label names do not match the head's width, or
/// when `top_k` is zero.
pub(crate) fn classify_with_bert(
    model: &BertForSequenceClassification,
    input: TokenizedInput,
    labels: &[String],
    top_k: Option<usize>,
) -> Result<Vec<ScoredLabel>, TrustformersError> {
    if input.input_ids.is_empty() {
        return Err(runtime_error(
            "the tokenizer produced no tokens for this text, so there is nothing to classify"
                .to_string(),
        ));
    }
    if let Some(0) = top_k {
        return Err(runtime_error("top_k must be at least 1".to_string()));
    }

    let outputs = model.forward(input)?;
    let logits = single_logits_row(&outputs.logits)?;
    let mut ranked = rank_labels(&logits, labels)?;
    if let Some(limit) = top_k {
        ranked.truncate(limit);
    }
    Ok(ranked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{ArrayD, IxDyn};

    fn labels(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn softmax_is_a_distribution() {
        let scores = softmax(&[1.0, 2.0, 3.0]).expect("finite logits");
        let total: f32 = scores.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "scores must sum to 1, got {total}");
        assert!(scores[2] > scores[1] && scores[1] > scores[0]);
    }

    /// The max-shift is what keeps this from overflowing to `NaN`.
    #[test]
    fn softmax_survives_large_logits() {
        let scores = softmax(&[1000.0, 999.0]).expect("large logits");
        let total: f32 = scores.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert!(scores.iter().all(|score| score.is_finite()));
    }

    #[test]
    fn softmax_rejects_empty_and_non_finite_rows() {
        assert!(softmax(&[]).is_err());
        assert!(softmax(&[1.0, f32::NAN]).is_err());
        assert!(softmax(&[1.0, f32::INFINITY]).is_err());
    }

    /// The replaced pipeline returned `0.7`/`0.3` for every input. Real scores
    /// come from the logits, sum to 1, and are ordered best-first.
    #[test]
    fn ranks_labels_by_probability() {
        let ranked = rank_labels(&[0.5, 2.5, 1.0], &labels(&["NEGATIVE", "POSITIVE", "NEUTRAL"]))
            .expect("matching widths");
        assert_eq!(ranked[0].label, "POSITIVE");
        assert_eq!(ranked[1].label, "NEUTRAL");
        assert_eq!(ranked[2].label, "NEGATIVE");
        let total: f32 = ranked.iter().map(|scored| scored.score).sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert!(ranked.windows(2).all(|pair| pair[0].score >= pair[1].score));
    }

    /// Different logits must give different scores -- the property the
    /// hardcoded `0.7`/`0.3` could never satisfy.
    #[test]
    fn different_logits_give_different_scores() {
        let names = labels(&["NEGATIVE", "POSITIVE"]);
        let first = rank_labels(&[3.0, 0.0], &names).expect("valid");
        let second = rank_labels(&[0.0, 3.0], &names).expect("valid");
        assert_eq!(first[0].label, "NEGATIVE");
        assert_eq!(second[0].label, "POSITIVE");
        assert_ne!(first[0].score, second[1].score * 2.0);
    }

    #[test]
    fn rejects_a_label_count_that_does_not_match_the_head() {
        assert!(rank_labels(&[1.0, 2.0, 3.0], &labels(&["A", "B"])).is_err());
    }

    #[test]
    fn reads_a_single_logits_row_in_either_shape() {
        let flat = Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).expect("shape matches"),
        );
        let batched = Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).expect("shape matches"),
        );
        assert_eq!(
            single_logits_row(&flat).expect("flat row"),
            vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            single_logits_row(&batched).expect("batched row"),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn rejects_a_multi_row_logits_tensor() {
        let two_rows = Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).expect("shape"),
        );
        assert!(single_logits_row(&two_rows).is_err());
    }

    // ---- classify_with_bert: the real forward path ----

    /// A BERT small enough to run a real forward pass in a unit test.
    fn tiny_classifier(num_labels: usize) -> BertForSequenceClassification {
        let config = trustformers_models::bert::BertConfig {
            vocab_size: 64,
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 32,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 16,
            ..trustformers_models::bert::BertConfig::default()
        };
        BertForSequenceClassification::new(config, num_labels)
            .expect("tiny BERT classifier config is valid")
    }

    fn tokenized(ids: &[u32]) -> TokenizedInput {
        TokenizedInput {
            input_ids: ids.to_vec(),
            attention_mask: vec![1u8; ids.len()],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        }
    }

    /// The replaced pipeline never called a model at all. This runs the real
    /// encoder + pooler + linear head and checks the contract: one score per
    /// label, summing to 1, ranked best-first.
    #[test]
    fn classifies_with_a_real_forward_pass() {
        let model = tiny_classifier(3);
        let names = labels(&["NEGATIVE", "NEUTRAL", "POSITIVE"]);
        let ranked = classify_with_bert(&model, tokenized(&[1, 2, 3, 4]), &names, None)
            .expect("forward pass succeeds");

        assert_eq!(ranked.len(), 3);
        let total: f32 = ranked.iter().map(|scored| scored.score).sum();
        assert!((total - 1.0).abs() < 1e-5, "scores must sum to 1, got {total}");
        assert!(ranked.windows(2).all(|pair| pair[0].score >= pair[1].score));
        let mut returned: Vec<&str> = ranked.iter().map(|scored| scored.label.as_str()).collect();
        returned.sort_unstable();
        assert_eq!(returned, vec!["NEGATIVE", "NEUTRAL", "POSITIVE"]);
    }

    /// The head has `num_labels` outputs, so a 2-label model must never
    /// produce the 2 fixed classes the placeholder always returned regardless
    /// of how many labels the checkpoint declared.
    #[test]
    fn label_count_follows_the_head_width() {
        let model = tiny_classifier(5);
        let names = labels(&["A", "B", "C", "D", "E"]);
        let ranked =
            classify_with_bert(&model, tokenized(&[7, 8]), &names, None).expect("forward pass");
        assert_eq!(ranked.len(), 5);

        // A head of a different width than the label list is a configuration
        // error, not something to pad over.
        assert!(classify_with_bert(&model, tokenized(&[7, 8]), &labels(&["A", "B"]), None).is_err());
    }

    /// Different inputs must give different scores -- the property the fixed
    /// `0.7`/`0.3` pair could never satisfy.
    #[test]
    fn different_inputs_give_different_scores() {
        let model = tiny_classifier(2);
        let names = labels(&["NEGATIVE", "POSITIVE"]);
        let first = classify_with_bert(&model, tokenized(&[1, 2, 3]), &names, None)
            .expect("first forward pass");
        let second = classify_with_bert(&model, tokenized(&[40, 41, 42]), &names, None)
            .expect("second forward pass");
        let first_positive = first
            .iter()
            .find(|scored| scored.label == "POSITIVE")
            .map(|scored| scored.score)
            .expect("POSITIVE is scored");
        let second_positive = second
            .iter()
            .find(|scored| scored.label == "POSITIVE")
            .map(|scored| scored.score)
            .expect("POSITIVE is scored");
        assert_ne!(
            first_positive, second_positive,
            "a real model gives different inputs different scores"
        );
    }

    #[test]
    fn classification_is_deterministic_for_one_model() {
        let model = tiny_classifier(2);
        let names = labels(&["NEGATIVE", "POSITIVE"]);
        let first =
            classify_with_bert(&model, tokenized(&[5, 6]), &names, None).expect("first run");
        let second =
            classify_with_bert(&model, tokenized(&[5, 6]), &names, None).expect("second run");
        assert_eq!(first, second);
    }

    #[test]
    fn top_k_truncates_and_rejects_zero() {
        let model = tiny_classifier(4);
        let names = labels(&["A", "B", "C", "D"]);
        let ranked = classify_with_bert(&model, tokenized(&[1, 2]), &names, Some(2))
            .expect("forward pass");
        assert_eq!(ranked.len(), 2);
        assert!(classify_with_bert(&model, tokenized(&[1, 2]), &names, Some(0)).is_err());
    }

    #[test]
    fn rejects_an_empty_tokenization() {
        let model = tiny_classifier(2);
        assert!(
            classify_with_bert(&model, tokenized(&[]), &labels(&["A", "B"]), None).is_err(),
            "an empty tokenization must be an error, not a classified empty string"
        );
    }
}
