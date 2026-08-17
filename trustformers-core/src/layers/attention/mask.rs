//! Interpretation and broadcasting of attention masks.
//!
//! Two mask conventions are in active use across the ecosystem and both are
//! supported here:
//!
//! * **Keep masks** hold `1` for positions that may be attended to and `0` for
//!   positions that must be ignored (this is what HuggingFace tokenizers emit
//!   and what [`crate::layers::SDPA`] historically expected).
//! * **Additive masks** are added directly to the pre-softmax scores: `0.0`
//!   keeps a position, a large negative value (`-inf`, `-1e9`, `-10000.0`)
//!   removes it. This is what "extended" transformer masks and hand-built
//!   structural masks (e.g. tree masks) look like.
//!
//! Because the two are numerically incompatible, the convention is detected
//! from the mask data with one documented rule (see
//! [`MaskSemantics::detect`]) rather than guessed per call site.

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use scirs2_core::ndarray::ArrayD;

/// How the numeric values of an attention mask are to be interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskSemantics {
    /// `0` marks a masked-out position; every other value keeps the position.
    Keep,
    /// The value is added to the pre-softmax score (`0.0` keeps a position, a
    /// large negative value removes it).
    Additive,
}

impl MaskSemantics {
    /// Detect the convention used by a mask tensor.
    ///
    /// The rule is deliberately simple and total: a mask is a [`Keep`] mask
    /// **iff** every element is exactly `0.0` or `1.0` *and* at least one
    /// element is `1.0`. Everything else - including all-zero masks, masks
    /// containing `-inf`/`-1e9`, and masks with fractional values - is treated
    /// as [`Additive`].
    ///
    /// The `at least one 1.0` clause matters: an all-zero tensor is a no-op
    /// additive mask but would be a "mask everything" keep mask, and treating
    /// it as the latter would silently produce empty attention rows.
    ///
    /// [`Keep`]: MaskSemantics::Keep
    /// [`Additive`]: MaskSemantics::Additive
    pub fn detect(values: &ArrayD<f32>) -> Self {
        let mut saw_one = false;
        for &value in values.iter() {
            if value == 1.0 {
                saw_one = true;
            } else if value != 0.0 {
                return MaskSemantics::Additive;
            }
        }
        if saw_one {
            MaskSemantics::Keep
        } else {
            MaskSemantics::Additive
        }
    }
}

/// Resolved broadcasting plan for a mask tensor.
#[derive(Debug, Clone, Copy)]
enum MaskKind {
    /// `[seq_q, seq_k]`
    QueryKey,
    /// `[batch, seq_k]` - a padding mask shared by every query position.
    BatchKey,
    /// `[batch | 1, seq_q | 1, seq_k]`
    BatchQueryKey {
        broadcast_batch: bool,
        broadcast_query: bool,
    },
    /// `[batch | 1, heads | 1, seq_q | 1, seq_k]`
    Full {
        broadcast_batch: bool,
        broadcast_head: bool,
        broadcast_query: bool,
    },
}

/// A borrowed attention mask with a resolved shape and value convention.
///
/// Constructing a `MaskView` validates the mask against the attention shape
/// once; [`MaskView::additive`] then yields the value that must be *added* to
/// the pre-softmax score at `(batch, head, query, key)`.
pub(crate) struct MaskView<'a> {
    values: &'a ArrayD<f32>,
    semantics: MaskSemantics,
    kind: MaskKind,
}

impl<'a> MaskView<'a> {
    /// Validate `mask` against an attention problem of shape
    /// `[batch, heads, seq_q, seq_k]`.
    pub(crate) fn new(
        mask: &'a Tensor,
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_k: usize,
    ) -> Result<Self> {
        let values = match mask {
            Tensor::F32(array) => array,
            other => {
                return Err(TrustformersError::tensor_op_error(
                    &format!(
                        "Attention masks must be F32 tensors, got {:?}",
                        other.dtype()
                    ),
                    "MaskView::new",
                ));
            },
        };

        let shape = values.shape();
        let kind = match shape.len() {
            2 => {
                if shape[0] == seq_q && shape[1] == seq_k {
                    MaskKind::QueryKey
                } else if shape[0] == batch && shape[1] == seq_k {
                    MaskKind::BatchKey
                } else {
                    return Err(Self::shape_error(shape, batch, heads, seq_q, seq_k));
                }
            },
            3 => {
                let batch_ok = shape[0] == batch || shape[0] == 1;
                let query_ok = shape[1] == seq_q || shape[1] == 1;
                if !batch_ok || !query_ok || shape[2] != seq_k {
                    return Err(Self::shape_error(shape, batch, heads, seq_q, seq_k));
                }
                MaskKind::BatchQueryKey {
                    broadcast_batch: shape[0] == 1 && batch != 1,
                    broadcast_query: shape[1] == 1 && seq_q != 1,
                }
            },
            4 => {
                let batch_ok = shape[0] == batch || shape[0] == 1;
                let head_ok = shape[1] == heads || shape[1] == 1;
                let query_ok = shape[2] == seq_q || shape[2] == 1;
                if !batch_ok || !head_ok || !query_ok || shape[3] != seq_k {
                    return Err(Self::shape_error(shape, batch, heads, seq_q, seq_k));
                }
                MaskKind::Full {
                    broadcast_batch: shape[0] == 1 && batch != 1,
                    broadcast_head: shape[1] == 1 && heads != 1,
                    broadcast_query: shape[2] == 1 && seq_q != 1,
                }
            },
            _ => return Err(Self::shape_error(shape, batch, heads, seq_q, seq_k)),
        };

        Ok(Self {
            values,
            semantics: MaskSemantics::detect(values),
            kind,
        })
    }

    fn shape_error(
        shape: &[usize],
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_k: usize,
    ) -> TrustformersError {
        TrustformersError::tensor_op_error(
            &format!(
                "Attention mask shape {:?} is not broadcastable to [{}, {}, {}, {}]",
                shape, batch, heads, seq_q, seq_k
            ),
            "MaskView::new",
        )
    }

    #[inline]
    fn raw(&self, batch: usize, head: usize, query: usize, key: usize) -> f32 {
        match self.kind {
            MaskKind::QueryKey => self.values[[query, key]],
            MaskKind::BatchKey => self.values[[batch, key]],
            MaskKind::BatchQueryKey {
                broadcast_batch,
                broadcast_query,
            } => {
                let b = if broadcast_batch { 0 } else { batch };
                let q = if broadcast_query { 0 } else { query };
                self.values[[b, q, key]]
            },
            MaskKind::Full {
                broadcast_batch,
                broadcast_head,
                broadcast_query,
            } => {
                let b = if broadcast_batch { 0 } else { batch };
                let h = if broadcast_head { 0 } else { head };
                let q = if broadcast_query { 0 } else { query };
                self.values[[b, h, q, key]]
            },
        }
    }

    /// Value to add to the pre-softmax score at `(batch, head, query, key)`.
    ///
    /// Returns `0.0` for kept positions and [`f32::NEG_INFINITY`] (or the raw
    /// additive penalty) for masked ones.
    #[inline]
    pub(crate) fn additive(&self, batch: usize, head: usize, query: usize, key: usize) -> f32 {
        let value = self.raw(batch, head, query, key);
        match self.semantics {
            MaskSemantics::Keep => {
                if value == 0.0 {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            },
            MaskSemantics::Additive => value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::IxDyn;

    fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(shape), data).expect("test mask shape must be valid"),
        )
    }

    #[test]
    fn detects_keep_masks() {
        let mask = tensor(vec![1.0, 0.0, 1.0, 1.0], &[2, 2]);
        let view = MaskView::new(&mask, 1, 1, 2, 2).expect("2-D keep mask must be accepted");
        assert_eq!(view.semantics, MaskSemantics::Keep);
        assert_eq!(view.additive(0, 0, 0, 0), 0.0);
        assert_eq!(view.additive(0, 0, 0, 1), f32::NEG_INFINITY);
    }

    #[test]
    fn detects_additive_masks() {
        let mask = tensor(vec![0.0, f32::NEG_INFINITY, 0.0, 0.0], &[2, 2]);
        let view = MaskView::new(&mask, 1, 1, 2, 2).expect("2-D additive mask must be accepted");
        assert_eq!(view.semantics, MaskSemantics::Additive);
        assert_eq!(view.additive(0, 0, 0, 0), 0.0);
        assert_eq!(view.additive(0, 0, 0, 1), f32::NEG_INFINITY);
    }

    #[test]
    fn all_zero_mask_is_treated_as_additive_no_op() {
        // An all-zero tensor is a no-op additive mask; reading it as a keep
        // mask would silently blank out every attention row.
        let mask = tensor(vec![0.0; 4], &[2, 2]);
        let view = MaskView::new(&mask, 1, 1, 2, 2).expect("all-zero mask must be accepted");
        assert_eq!(view.semantics, MaskSemantics::Additive);
        assert_eq!(view.additive(0, 0, 1, 1), 0.0);
    }

    #[test]
    fn broadcasts_padding_masks() {
        // HuggingFace-style [1, 1, 1, seq_k] padding mask.
        let mask = tensor(vec![1.0, 1.0, 0.0], &[1, 1, 1, 3]);
        let view = MaskView::new(&mask, 2, 4, 3, 3).expect("padding mask must broadcast");
        assert_eq!(view.semantics, MaskSemantics::Keep);
        assert_eq!(view.additive(1, 3, 2, 0), 0.0);
        assert_eq!(view.additive(1, 3, 2, 2), f32::NEG_INFINITY);
    }

    #[test]
    fn rejects_incompatible_shapes() {
        let mask = tensor(vec![1.0; 6], &[2, 3]);
        assert!(MaskView::new(&mask, 4, 2, 5, 5).is_err());
    }
}
