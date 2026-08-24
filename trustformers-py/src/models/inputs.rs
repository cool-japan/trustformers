//! Pure-Rust construction of [`TokenizedInput`] from caller-supplied tensors.
//!
//! Every `forward(input_ids, attention_mask, token_type_ids, ...)` binding in
//! `models/mod.rs` needs the same conversion, and every one of them used to
//! open-code it. The GPT-2 copy did this for the mask:
//!
//! ```text
//! mask.inner.to_vec_f32().unwrap_or_default().iter().map(|&x| x as u8).collect()
//! ```
//!
//! which turns a *failed* conversion (a dtype `to_vec_f32` cannot handle) into
//! an empty mask, silently making every real token invisible to attention
//! instead of reporting the error -- and it derived the default mask length
//! from `input_ids.shape()[0]`, which panics on a 0-D tensor and is the wrong
//! length for anything but a 1-D one.
//!
//! These functions are deliberately free of the Python C API so they are
//! unit-testable with a plain `cargo test` (the test binary does not link
//! `libpython`; see `weights.rs` and `losses.rs` for the same split).

use trustformers_core::errors::{runtime_error, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::TokenizedInput;

/// Read a tensor's values as exact non-negative integers.
///
/// `I64` tensors are read directly rather than through `to_vec_f32`, whose
/// `i64 -> f32` step silently rounds any value above 2^24.
fn exact_non_negative_values(
    tensor: &Tensor,
    what: &str,
) -> Result<Vec<u64>, TrustformersError> {
    let raw: Vec<f64> = match tensor {
        Tensor::I64(array) => {
            return array
                .iter()
                .map(|&value| {
                    u64::try_from(value).map_err(|_| {
                        runtime_error(format!("{what} value {value} is negative"))
                    })
                })
                .collect();
        },
        _ => tensor.to_vec_f32()?.into_iter().map(f64::from).collect(),
    };

    raw.into_iter()
        .map(|value| {
            if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
                return Err(runtime_error(format!(
                    "{what} value {value} is not a non-negative integer"
                )));
            }
            Ok(value as u64)
        })
        .collect()
}

/// Narrow `values` to `u32`, the width [`TokenizedInput`] stores ids in.
fn narrow_to_u32(values: Vec<u64>, what: &str) -> Result<Vec<u32>, TrustformersError> {
    values
        .into_iter()
        .map(|value| {
            u32::try_from(value).map_err(|_| {
                runtime_error(format!(
                    "{what} value {value} does not fit in the u32 vocabulary index range"
                ))
            })
        })
        .collect()
}

/// Token ids from a tensor of integer (or integral float) token values.
pub(crate) fn token_ids_from_tensor(tensor: &Tensor) -> Result<Vec<u32>, TrustformersError> {
    let ids = narrow_to_u32(exact_non_negative_values(tensor, "input_ids")?, "input_ids")?;
    if ids.is_empty() {
        return Err(runtime_error(
            "input_ids is empty; a forward pass needs at least one token".to_string(),
        ));
    }
    Ok(ids)
}

/// Attention-mask flags from a tensor, checked against `expected_len`.
///
/// Only 0 and 1 are accepted: an "attention mask" holding anything else is a
/// caller mistake, and `x as u8` (what this replaces) would have truncated it
/// into a valid-looking flag.
fn attention_mask_from_tensor(
    tensor: &Tensor,
    expected_len: usize,
) -> Result<Vec<u8>, TrustformersError> {
    let values = exact_non_negative_values(tensor, "attention_mask")?;
    if values.len() != expected_len {
        return Err(runtime_error(format!(
            "attention_mask holds {} values but input_ids holds {}; they must be aligned",
            values.len(),
            expected_len
        )));
    }
    values
        .into_iter()
        .map(|value| match value {
            0 | 1 => Ok(value as u8),
            other => Err(runtime_error(format!(
                "attention_mask value {other} is not 0 or 1"
            ))),
        })
        .collect()
}

/// Segment ids from a tensor, checked against `expected_len`.
fn token_type_ids_from_tensor(
    tensor: &Tensor,
    expected_len: usize,
) -> Result<Vec<u32>, TrustformersError> {
    let values = narrow_to_u32(
        exact_non_negative_values(tensor, "token_type_ids")?,
        "token_type_ids",
    )?;
    if values.len() != expected_len {
        return Err(runtime_error(format!(
            "token_type_ids holds {} values but input_ids holds {}; they must be aligned",
            values.len(),
            expected_len
        )));
    }
    Ok(values)
}

/// Build a [`TokenizedInput`] from the tensors a `forward` binding receives.
///
/// An absent `attention_mask` becomes an all-ones mask of the *token* length
/// (not of `shape()[0]`). An absent `token_type_ids` stays `None`, which is
/// what a single-segment BERT input and every decoder-only model want.
///
/// # Errors
///
/// Fails when a tensor's dtype cannot be read, when it holds a value that is
/// not a valid non-negative index, or when `attention_mask`/`token_type_ids`
/// disagree in length with `input_ids` -- all of which the open-coded
/// conversions this replaces accepted silently.
pub(crate) fn tokenized_input_from_parts(
    input_ids: &Tensor,
    attention_mask: Option<&Tensor>,
    token_type_ids: Option<&Tensor>,
) -> Result<TokenizedInput, TrustformersError> {
    let input_ids = token_ids_from_tensor(input_ids)?;
    let token_count = input_ids.len();

    let attention_mask = match attention_mask {
        Some(mask) => attention_mask_from_tensor(mask, token_count)?,
        None => vec![1u8; token_count],
    };

    let token_type_ids = match token_type_ids {
        Some(ids) => Some(token_type_ids_from_tensor(ids, token_count)?),
        None => None,
    };

    Ok(TokenizedInput {
        input_ids,
        attention_mask,
        token_type_ids,
        special_tokens_mask: None,
        offset_mapping: None,
        overflowing_tokens: None,
    })
}

/// Build a [`TokenizedInput`] for a run of already-decoded token ids, with an
/// all-ones attention mask and no segment ids.
///
/// This is the shape every autoregressive decoding step feeds back into the
/// model.
pub(crate) fn tokenized_input_from_ids(ids: &[u32]) -> TokenizedInput {
    TokenizedInput {
        input_ids: ids.to_vec(),
        attention_mask: vec![1u8; ids.len()],
        token_type_ids: None,
        special_tokens_mask: None,
        offset_mapping: None,
        overflowing_tokens: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{ArrayD, IxDyn};

    fn f32_tensor(shape: &[usize], values: Vec<f32>) -> Tensor {
        Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(shape), values).expect("test tensor shape matches values"),
        )
    }

    fn i64_tensor(shape: &[usize], values: Vec<i64>) -> Tensor {
        Tensor::I64(
            ArrayD::from_shape_vec(IxDyn(shape), values).expect("test tensor shape matches values"),
        )
    }

    #[test]
    fn defaults_the_mask_to_the_token_count() {
        let input = tokenized_input_from_parts(&i64_tensor(&[3], vec![5, 6, 7]), None, None)
            .expect("valid ids");
        assert_eq!(input.input_ids, vec![5, 6, 7]);
        assert_eq!(input.attention_mask, vec![1, 1, 1]);
        assert!(input.token_type_ids.is_none());
    }

    /// The replaced GPT-2 code derived the default mask length from
    /// `input_ids.shape()[0]`, so a `[1, 4]` batch-shaped input produced a
    /// 1-element mask for 4 tokens (and a 0-D tensor panicked outright).
    #[test]
    fn mask_length_follows_tokens_not_the_first_axis() {
        let input =
            tokenized_input_from_parts(&i64_tensor(&[1, 4], vec![1, 2, 3, 4]), None, None)
                .expect("valid ids");
        assert_eq!(input.attention_mask.len(), 4);
    }

    /// The replaced code did `to_vec_f32().unwrap_or_default()`, turning a
    /// failed conversion into an empty (all-tokens-masked) attention mask.
    #[test]
    fn propagates_a_mask_conversion_failure() {
        // `F16` is a dtype `Tensor::to_vec_f32` genuinely refuses.
        let mask = Tensor::F16(ArrayD::from_elem(IxDyn(&[2]), half::f16::from_f32(1.0)));
        let result = tokenized_input_from_parts(&i64_tensor(&[2], vec![1, 2]), Some(&mask), None);
        assert!(
            result.is_err(),
            "an unreadable mask dtype must be an error, not an empty mask"
        );
    }

    #[test]
    fn rejects_a_mask_of_the_wrong_length() {
        let result = tokenized_input_from_parts(
            &i64_tensor(&[3], vec![1, 2, 3]),
            Some(&i64_tensor(&[2], vec![1, 1])),
            None,
        );
        assert!(result.is_err(), "mask/ids length mismatch must be rejected");
    }

    #[test]
    fn rejects_a_mask_value_that_is_not_a_flag() {
        let result = tokenized_input_from_parts(
            &i64_tensor(&[2], vec![1, 2]),
            Some(&i64_tensor(&[2], vec![1, 7])),
            None,
        );
        assert!(result.is_err(), "mask values other than 0/1 must be rejected");
    }

    #[test]
    fn keeps_token_type_ids_when_given() {
        let input = tokenized_input_from_parts(
            &i64_tensor(&[4], vec![1, 2, 3, 4]),
            None,
            Some(&i64_tensor(&[4], vec![0, 0, 1, 1])),
        )
        .expect("valid inputs");
        assert_eq!(input.token_type_ids, Some(vec![0, 0, 1, 1]));
    }

    #[test]
    fn rejects_token_type_ids_of_the_wrong_length() {
        let result = tokenized_input_from_parts(
            &i64_tensor(&[4], vec![1, 2, 3, 4]),
            None,
            Some(&i64_tensor(&[3], vec![0, 0, 1])),
        );
        assert!(result.is_err());
    }

    #[test]
    fn rejects_negative_and_fractional_ids() {
        assert!(tokenized_input_from_parts(&i64_tensor(&[1], vec![-1]), None, None).is_err());
        assert!(tokenized_input_from_parts(&f32_tensor(&[1], vec![1.5]), None, None).is_err());
        assert!(tokenized_input_from_parts(&f32_tensor(&[1], vec![f32::NAN]), None, None).is_err());
    }

    #[test]
    fn rejects_empty_input_ids() {
        assert!(tokenized_input_from_parts(&i64_tensor(&[0], vec![]), None, None).is_err());
    }

    /// `i64 -> f32 -> u32` rounds anything above 2^24; ids are read from an
    /// `I64` tensor exactly.
    #[test]
    fn reads_large_i64_ids_exactly() {
        let id = 16_777_217i64; // 2^24 + 1, not representable in f32
        let input = tokenized_input_from_parts(&i64_tensor(&[1], vec![id]), None, None)
            .expect("valid id");
        assert_eq!(input.input_ids, vec![16_777_217u32]);
    }

    #[test]
    fn tokenized_input_from_ids_masks_every_token() {
        let input = tokenized_input_from_ids(&[9, 8, 7]);
        assert_eq!(input.input_ids, vec![9, 8, 7]);
        assert_eq!(input.attention_mask, vec![1, 1, 1]);
        assert!(input.token_type_ids.is_none());
    }
}
