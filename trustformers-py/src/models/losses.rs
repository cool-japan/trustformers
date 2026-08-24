//! Real cross-entropy losses for the task-model bindings.
//!
//! Both functions here replace implementations that took a `labels` argument,
//! named it `_labels`, and never looked at it: the classification loss was
//! `-mean(log(softmax(logits)))` and the language-modeling loss was the same
//! expression again. That number is not a cross-entropy against anything -- it
//! is a function of the logits alone, so `forward(..., labels=y)` returned the
//! *same* "loss" for every possible `y`, including deliberately wrong labels.
//!
//! These are pure-Rust and deliberately free of the Python C API so they are
//! unit-testable with a plain `cargo test` (the test binary does not link
//! `libpython`; see `weights.rs` for the same split).

use trustformers_core::errors::{runtime_error, TrustformersError};
use trustformers_core::tensor::Tensor;

/// Turn a class-index tensor into `usize` indices, rejecting anything that is
/// not a valid class index instead of silently truncating it.
fn label_indices(labels: &Tensor, num_classes: usize) -> Result<Vec<usize>, TrustformersError> {
    labels
        .to_vec_f32()?
        .into_iter()
        .map(|value| {
            if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
                return Err(runtime_error(format!(
                    "label {value} is not a non-negative integer class index"
                )));
            }
            let index = value as usize;
            if index >= num_classes {
                return Err(runtime_error(format!(
                    "label {index} is out of range for {num_classes} classes"
                )));
            }
            Ok(index)
        })
        .collect()
}

/// The number of classes (last axis) and the flattened logits of `logits`.
fn logit_rows(logits: &Tensor) -> Result<(usize, Vec<f32>), TrustformersError> {
    let num_classes = *logits
        .shape()
        .last()
        .ok_or_else(|| runtime_error("logits tensor has no dimensions".to_string()))?;
    if num_classes == 0 {
        return Err(runtime_error(
            "logits tensor has an empty class axis".to_string(),
        ));
    }
    let data = logits.to_vec_f32()?;
    if data.len() % num_classes != 0 {
        return Err(runtime_error(format!(
            "logits hold {} values, which is not a multiple of the {} classes on the last axis",
            data.len(),
            num_classes
        )));
    }
    Ok((num_classes, data))
}

/// `log(sum(exp(row)))` computed in the numerically stable max-shifted form.
fn log_sum_exp(row: &[f32]) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return max;
    }
    let sum: f32 = row.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

/// Mean negative log-likelihood of `targets` under `rows` of logits.
fn mean_nll(rows: &[&[f32]], targets: &[usize]) -> Result<f32, TrustformersError> {
    if rows.is_empty() {
        return Err(runtime_error(
            "loss requires at least one logits row".to_string(),
        ));
    }
    let mut total = 0.0f32;
    for (row, &target) in rows.iter().zip(targets.iter()) {
        let logit = *row
            .get(target)
            .ok_or_else(|| runtime_error(format!("class index {target} is out of range")))?;
        total += log_sum_exp(row) - logit;
    }
    Ok(total / rows.len() as f32)
}

/// Cross-entropy of `[.., num_classes]` classification logits against
/// class-index `labels` (one index per logits row).
///
/// # Errors
///
/// Fails when the label count does not match the number of logits rows, or
/// when a label is not a valid class index -- both of which the previous
/// label-ignoring implementation accepted silently.
pub(crate) fn classification_cross_entropy(
    logits: &Tensor,
    labels: &Tensor,
) -> Result<f32, TrustformersError> {
    let (num_classes, data) = logit_rows(logits)?;
    let rows: Vec<&[f32]> = data.chunks(num_classes).collect();
    let targets = label_indices(labels, num_classes)?;
    if targets.len() != rows.len() {
        return Err(runtime_error(format!(
            "got {} labels for {} logits rows; classification labels must hold one class index \
             per row",
            targets.len(),
            rows.len()
        )));
    }
    mean_nll(&rows, &targets)
}

/// Causal-LM cross-entropy: the loss of predicting `labels[t + 1]` from the
/// logits at position `t`, averaged over the sequence.
///
/// This is the shift HuggingFace applies inside `GPT2LMHeadModel.forward` when
/// `labels` is passed; the caller hands in the same `labels` tensor as the
/// input ids, exactly as there.
///
/// # Errors
///
/// Fails when the label count does not match the number of positions in the
/// logits, when the sequence is too short to have a single (input, next-token)
/// pair, or when a label is not a valid vocabulary index.
pub(crate) fn language_modeling_cross_entropy(
    logits: &Tensor,
    labels: &Tensor,
) -> Result<f32, TrustformersError> {
    let (vocab_size, data) = logit_rows(logits)?;
    let positions: Vec<&[f32]> = data.chunks(vocab_size).collect();
    let targets = label_indices(labels, vocab_size)?;
    if targets.len() != positions.len() {
        return Err(runtime_error(format!(
            "got {} labels for {} sequence positions; language-modeling labels must be aligned \
             with the input ids",
            targets.len(),
            positions.len()
        )));
    }
    if positions.len() < 2 {
        return Err(runtime_error(
            "language-modeling loss needs at least two positions: with a single token there is \
             no next token to predict"
                .to_string(),
        ));
    }
    let shifted_rows: Vec<&[f32]> = positions[..positions.len() - 1].to_vec();
    let shifted_targets: Vec<usize> = targets[1..].to_vec();
    mean_nll(&shifted_rows, &shifted_targets)
}

/// Extractive question-answering loss: the mean of the start-position and
/// end-position cross-entropies, exactly as HuggingFace's
/// `BertForQuestionAnswering.forward` computes it when `start_positions` /
/// `end_positions` are given.
///
/// Unlike [`classification_cross_entropy`], the "class axis" here is the
/// sequence position, not the tensor's trailing dimension: `start_logits` /
/// `end_logits` carry exactly one score per input position (whatever the
/// tensor's exact rank -- `[seq_len]` or `[1, seq_len]` or `[1, seq_len, 1]`,
/// depending on how the head's `split` leaves the singleton axis), so this
/// flattens the whole tensor into one row of `seq_len` scores rather than
/// chunking it by a trailing "num_classes" dimension the way
/// [`classification_cross_entropy`] does.
///
/// # Errors
///
/// Fails when `start_logits` and `end_logits` do not cover the same number of
/// positions, when either position tensor does not hold exactly one index (a
/// single-example forward pass has exactly one start and one end position),
/// or when a position is not a valid index into the sequence.
pub(crate) fn qa_span_cross_entropy(
    start_logits: &Tensor,
    end_logits: &Tensor,
    start_position: &Tensor,
    end_position: &Tensor,
) -> Result<f32, TrustformersError> {
    let start_row = start_logits.to_vec_f32()?;
    let end_row = end_logits.to_vec_f32()?;
    if start_row.len() != end_row.len() {
        return Err(runtime_error(format!(
            "start_logits and end_logits must cover the same number of sequence positions, got {} \
             and {}",
            start_row.len(),
            end_row.len()
        )));
    }
    let seq_len = start_row.len();
    if seq_len == 0 {
        return Err(runtime_error(
            "start_logits/end_logits have no sequence positions".to_string(),
        ));
    }

    let start_target = label_indices(start_position, seq_len)?;
    let end_target = label_indices(end_position, seq_len)?;
    if start_target.len() != 1 {
        return Err(runtime_error(format!(
            "start_positions must hold exactly one index for this single-example forward pass, \
             got {}",
            start_target.len()
        )));
    }
    if end_target.len() != 1 {
        return Err(runtime_error(format!(
            "end_positions must hold exactly one index for this single-example forward pass, got {}",
            end_target.len()
        )));
    }

    let start_loss = mean_nll(&[start_row.as_slice()], &start_target)?;
    let end_loss = mean_nll(&[end_row.as_slice()], &end_target)?;
    Ok((start_loss + end_loss) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{ArrayD, IxDyn};

    fn tensor(shape: &[usize], values: Vec<f32>) -> Tensor {
        Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(shape), values).expect("fixture shape matches value count"),
        )
    }

    #[test]
    fn classification_loss_depends_on_the_labels() {
        // The regression test for the old implementation: it computed
        // `-mean(log(softmax(logits)))`, so these two calls -- confident
        // prediction of class 1, scored once against the right label and once
        // against the wrong one -- returned the *identical* number.
        let logits = tensor(&[1, 2], vec![0.0, 5.0]);
        let right = classification_cross_entropy(&logits, &tensor(&[1], vec![1.0]))
            .expect("a valid label scores");
        let wrong = classification_cross_entropy(&logits, &tensor(&[1], vec![0.0]))
            .expect("a valid label scores");
        assert!(
            wrong > right,
            "predicting the wrong class must cost more ({wrong} vs {right})"
        );
    }

    #[test]
    fn classification_loss_matches_the_closed_form_value() {
        // Two logits 5 apart: -log(softmax) of the larger is log(1 + e^-5).
        let logits = tensor(&[1, 2], vec![0.0, 5.0]);
        let loss = classification_cross_entropy(&logits, &tensor(&[1], vec![1.0]))
            .expect("a valid label scores");
        let expected = (1.0f32 + (-5.0f32).exp()).ln();
        assert!(
            (loss - expected).abs() < 1e-6,
            "expected {expected}, got {loss}"
        );
    }

    #[test]
    fn classification_loss_of_a_uniform_distribution_is_ln_num_classes() {
        let logits = tensor(&[1, 4], vec![2.0, 2.0, 2.0, 2.0]);
        let loss = classification_cross_entropy(&logits, &tensor(&[1], vec![3.0]))
            .expect("a valid label scores");
        assert!((loss - 4.0f32.ln()).abs() < 1e-6, "got {loss}");
    }

    #[test]
    fn classification_loss_averages_over_a_batch() {
        let logits = tensor(&[2, 2], vec![0.0, 5.0, 5.0, 0.0]);
        let both_right = classification_cross_entropy(&logits, &tensor(&[2], vec![1.0, 0.0]))
            .expect("valid labels score");
        let one_wrong = classification_cross_entropy(&logits, &tensor(&[2], vec![1.0, 1.0]))
            .expect("valid labels score");
        assert!(one_wrong > both_right);
    }

    #[test]
    fn classification_loss_rejects_a_label_count_that_does_not_match_the_batch() {
        let logits = tensor(&[2, 2], vec![0.0, 1.0, 1.0, 0.0]);
        let result = classification_cross_entropy(&logits, &tensor(&[3], vec![0.0, 1.0, 0.0]));
        assert!(result.is_err(), "a label/row count mismatch must be an error");
    }

    #[test]
    fn classification_loss_rejects_an_out_of_range_label() {
        let logits = tensor(&[1, 2], vec![0.0, 1.0]);
        assert!(classification_cross_entropy(&logits, &tensor(&[1], vec![7.0])).is_err());
    }

    #[test]
    fn classification_loss_rejects_a_non_integer_label() {
        let logits = tensor(&[1, 2], vec![0.0, 1.0]);
        assert!(classification_cross_entropy(&logits, &tensor(&[1], vec![0.5])).is_err());
    }

    #[test]
    fn language_modeling_loss_shifts_labels_by_one_position() {
        // Position 0 predicts token 1, position 1 predicts token 2. Give
        // position 0 a confident (correct) prediction of token 1 and position 1
        // a confident (correct) prediction of token 2: the loss must be near
        // zero. The label at index 0 is never scored, which is what "shift"
        // means -- the old implementation ignored labels entirely and could not
        // express this at all.
        let logits = tensor(&[3, 3], vec![
            -9.0, 9.0, -9.0, // position 0 -> token 1
            -9.0, -9.0, 9.0, // position 1 -> token 2
            0.0, 0.0, 0.0, // position 2 is dropped by the shift
        ]);
        let labels = tensor(&[3], vec![0.0, 1.0, 2.0]);
        let loss = language_modeling_cross_entropy(&logits, &labels).expect("aligned labels score");
        assert!(loss < 1e-3, "a perfectly predicted sequence should cost ~0, got {loss}");

        // Same logits, labels rotated: now every prediction is wrong.
        let wrong_labels = tensor(&[3], vec![0.0, 2.0, 1.0]);
        let wrong_loss =
            language_modeling_cross_entropy(&logits, &wrong_labels).expect("aligned labels score");
        assert!(
            wrong_loss > loss + 1.0,
            "wrong labels must cost far more ({wrong_loss} vs {loss})"
        );
    }

    #[test]
    fn language_modeling_loss_ignores_the_final_position() {
        // The last position has no next token, so changing its logits must not
        // change the loss at all.
        let labels = tensor(&[3], vec![0.0, 1.0, 2.0]);
        let base = tensor(&[3, 3], vec![
            -1.0, 2.0, 0.5, -0.5, 0.0, 3.0, 0.0, 0.0, 0.0,
        ]);
        let perturbed = tensor(&[3, 3], vec![
            -1.0, 2.0, 0.5, -0.5, 0.0, 3.0, 40.0, -40.0, 12.0,
        ]);
        let a = language_modeling_cross_entropy(&base, &labels).expect("aligned labels score");
        let b = language_modeling_cross_entropy(&perturbed, &labels).expect("aligned labels score");
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    #[test]
    fn language_modeling_loss_rejects_misaligned_labels() {
        let logits = tensor(&[3, 3], vec![0.0; 9]);
        let labels = tensor(&[2], vec![0.0, 1.0]);
        assert!(language_modeling_cross_entropy(&logits, &labels).is_err());
    }

    #[test]
    fn language_modeling_loss_rejects_a_single_token_sequence() {
        let logits = tensor(&[1, 3], vec![0.0, 1.0, 2.0]);
        let labels = tensor(&[1], vec![1.0]);
        let result = language_modeling_cross_entropy(&logits, &labels);
        assert!(
            result.is_err(),
            "one token has no next token to predict; that must be an error, not a number"
        );
    }

    // ---- qa_span_cross_entropy ----

    #[test]
    fn qa_loss_depends_on_both_positions() {
        // Five positions; a confident (and correct) start-at-1/end-at-3 must
        // score much better than the same logits scored against a rotated,
        // wrong pair of positions.
        let start_logits = tensor(&[5], vec![-9.0, 9.0, -9.0, -9.0, -9.0]);
        let end_logits = tensor(&[5], vec![-9.0, -9.0, -9.0, 9.0, -9.0]);
        let right = qa_span_cross_entropy(
            &start_logits,
            &end_logits,
            &tensor(&[1], vec![1.0]),
            &tensor(&[1], vec![3.0]),
        )
        .expect("valid positions score");
        let wrong = qa_span_cross_entropy(
            &start_logits,
            &end_logits,
            &tensor(&[1], vec![0.0]),
            &tensor(&[1], vec![4.0]),
        )
        .expect("valid positions score");
        assert!(
            wrong > right + 1.0,
            "the wrong span must cost far more ({wrong} vs {right})"
        );
    }

    #[test]
    fn qa_loss_matches_the_closed_form_value() {
        // Two positions, 5 apart, exactly like the classification closed-form
        // test: -log(softmax) of the larger is log(1 + e^-5), for both start
        // and end, so the mean is the same value again.
        let start_logits = tensor(&[2], vec![0.0, 5.0]);
        let end_logits = tensor(&[2], vec![0.0, 5.0]);
        let loss = qa_span_cross_entropy(
            &start_logits,
            &end_logits,
            &tensor(&[1], vec![1.0]),
            &tensor(&[1], vec![1.0]),
        )
        .expect("valid positions score");
        let expected = (1.0f32 + (-5.0f32).exp()).ln();
        assert!((loss - expected).abs() < 1e-6, "expected {expected}, got {loss}");
    }

    /// A tampered start position must change the loss: the replaced
    /// implementation this guards against ignored `start_positions`/
    /// `end_positions` entirely (no such implementation ever existed here,
    /// but this is the exact property `BertForSequenceClassification`'s loss
    /// was once missing -- see `classification_loss_depends_on_the_labels`).
    #[test]
    fn qa_loss_changes_when_the_start_position_is_tampered_with() {
        let start_logits = tensor(&[4], vec![0.0, 8.0, 0.0, 0.0]);
        let end_logits = tensor(&[4], vec![0.0, 0.0, 0.0, 8.0]);
        let correct = qa_span_cross_entropy(
            &start_logits,
            &end_logits,
            &tensor(&[1], vec![1.0]),
            &tensor(&[1], vec![3.0]),
        )
        .expect("valid positions score");
        let tampered = qa_span_cross_entropy(
            &start_logits,
            &end_logits,
            &tensor(&[1], vec![2.0]),
            &tensor(&[1], vec![3.0]),
        )
        .expect("valid positions score");
        assert_ne!(correct, tampered);
    }

    #[test]
    fn qa_loss_rejects_mismatched_position_counts() {
        let start_logits = tensor(&[3], vec![0.0, 1.0, 2.0]);
        let end_logits = tensor(&[3], vec![0.0, 1.0, 2.0]);
        // Two start positions for a single-example forward pass is invalid.
        assert!(qa_span_cross_entropy(
            &start_logits,
            &end_logits,
            &tensor(&[2], vec![0.0, 1.0]),
            &tensor(&[1], vec![2.0])
        )
        .is_err());
    }

    #[test]
    fn qa_loss_rejects_logits_of_different_lengths() {
        let start_logits = tensor(&[3], vec![0.0, 1.0, 2.0]);
        let end_logits = tensor(&[4], vec![0.0, 1.0, 2.0, 3.0]);
        assert!(qa_span_cross_entropy(
            &start_logits,
            &end_logits,
            &tensor(&[1], vec![0.0]),
            &tensor(&[1], vec![0.0])
        )
        .is_err());
    }

    #[test]
    fn qa_loss_rejects_an_out_of_range_position() {
        let start_logits = tensor(&[3], vec![0.0, 1.0, 2.0]);
        let end_logits = tensor(&[3], vec![0.0, 1.0, 2.0]);
        assert!(qa_span_cross_entropy(
            &start_logits,
            &end_logits,
            &tensor(&[1], vec![9.0]),
            &tensor(&[1], vec![0.0])
        )
        .is_err());
    }
}
