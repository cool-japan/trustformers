//! Reusable binding helpers for `Model::load_pretrained` implementations.
//!
//! Every architecture in this crate maps HuggingFace tensor names onto its own
//! layers, and every one of those maps needs the same three things: take a
//! `[out, in]` weight (plus an optional `[out]` bias) for a `Linear`, take a
//! `[dim]` vector for a norm, and check the shape rather than reshaping. Doing
//! that by hand per model is how a loader ends up quietly skipping a parameter,
//! so the primitives live here and each model contributes only its name map.
//!
//! The strictness of [`WeightBinder`](crate::weight_loading::checkpoint::WeightBinder) is
//! preserved throughout: a tensor the checkpoint does not hold is *recorded as
//! missing* and the parameter is left untouched, never filled with a
//! substitute, and `WeightBinder::finish` turns the accumulated gaps into one
//! error naming all of them.

use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::layers::{Embedding, LayerNorm, Linear};
use trustformers_core::tensor::Tensor;

use super::checkpoint::{Checkpoint, LoadReport, WeightBinder};

/// Bind a `[out_features, in_features]` weight and an optional `[out_features]`
/// bias into a [`Linear`].
///
/// Both HuggingFace's `nn.Linear` and this crate's [`Linear`] store the weight
/// as `[out_features, in_features]`, so no transposition is involved; the shape
/// is checked instead of assumed.
///
/// `expect_bias` distinguishes "this architecture has a bias here, report it if
/// the checkpoint lacks it" from "this architecture has no bias, do not look".
///
/// # Errors
///
/// Fails when a tensor exists under the expected name but has the wrong shape.
pub fn bind_linear(
    binder: &mut WeightBinder<'_>,
    name: &str,
    out_features: usize,
    in_features: usize,
    expect_bias: bool,
    layer: &mut Linear,
) -> Result<()> {
    if let Some(weight) =
        binder.take_shaped(&format!("{name}.weight"), &[out_features, in_features])?
    {
        layer.set_weight(weight)?;
    }
    if expect_bias {
        if let Some(bias) = binder.take_shaped(&format!("{name}.bias"), &[out_features])? {
            layer.set_bias(bias)?;
        }
    }
    Ok(())
}

/// Bind a `[num_embeddings, embedding_dim]` matrix into an [`Embedding`].
///
/// # Errors
///
/// Fails when the tensor exists but has the wrong shape.
pub fn bind_embedding(
    binder: &mut WeightBinder<'_>,
    name: &str,
    num_embeddings: usize,
    embedding_dim: usize,
    layer: &mut Embedding,
) -> Result<()> {
    if let Some(weight) =
        binder.take_shaped(&format!("{name}.weight"), &[num_embeddings, embedding_dim])?
    {
        layer.set_weight(weight)?;
    }
    Ok(())
}

/// Take a `[dim]` normalisation weight, checking its shape.
///
/// Returned rather than assigned because each architecture wraps its norm in a
/// different type; the caller stores it through that type's own setter.
///
/// # Errors
///
/// Fails when the tensor exists but has the wrong shape.
pub fn take_norm_weight(
    binder: &mut WeightBinder<'_>,
    name: &str,
    dim: usize,
) -> Result<Option<Tensor>> {
    binder.take_shaped(&format!("{name}.weight"), &[dim])
}

/// Take a `[dim]` normalisation bias, checking its shape.
///
/// # Errors
///
/// Fails when the tensor exists but has the wrong shape.
pub fn take_norm_bias(
    binder: &mut WeightBinder<'_>,
    name: &str,
    dim: usize,
) -> Result<Option<Tensor>> {
    binder.take_shaped(&format!("{name}.bias"), &[dim])
}

/// Bind a `[out, in]` task-head projection and its bias from a checkpoint.
///
/// Task heads are bound *after* the backbone, straight off the parsed
/// [`Checkpoint`] rather than through a [`WeightBinder`], because the backbone's
/// binder has already finished and deliberately tolerated the head's namespace.
///
/// A head the checkpoint does not carry is legitimate — a pretrained backbone
/// ships without a fine-tuned head — but it is recorded in
/// [`LoadReport::missing`] so the caller can see that the layer kept its
/// constructor initialisation. A head that *is* present must reach the model:
/// silently leaving it behind is exactly the failure this crate is being
/// cleaned of.
///
/// # Errors
///
/// Fails when a tensor is present with the wrong shape.
pub fn bind_head_linear(
    checkpoint: &Checkpoint,
    report: &mut LoadReport,
    name: &str,
    weight_shape: [usize; 2],
    layer: &mut Linear,
) -> Result<()> {
    let weight_name = format!("{name}.weight");
    match checkpoint.take_shaped(&weight_name, &weight_shape)? {
        Some(weight) => {
            layer.set_weight(weight)?;
            report.mark_loaded(&weight_name);
        },
        None => report.note_absent(&weight_name),
    }

    let bias_name = format!("{name}.bias");
    match checkpoint.take_shaped(&bias_name, &[weight_shape[0]])? {
        Some(bias) => {
            layer.set_bias(bias)?;
            report.mark_loaded(&bias_name);
        },
        None => report.note_absent(&bias_name),
    }
    Ok(())
}

/// Bind a task-head [`LayerNorm`] from a checkpoint, recording what was found.
///
/// # Errors
///
/// Fails when a tensor is present with the wrong shape.
pub fn bind_head_layer_norm(
    checkpoint: &Checkpoint,
    report: &mut LoadReport,
    name: &str,
    hidden_size: usize,
    norm: &mut LayerNorm,
) -> Result<()> {
    let weight_name = format!("{name}.weight");
    match checkpoint.take_shaped(&weight_name, &[hidden_size])? {
        Some(weight) => {
            norm.set_weight(weight)?;
            report.mark_loaded(&weight_name);
        },
        None => report.note_absent(&weight_name),
    }

    let bias_name = format!("{name}.bias");
    match checkpoint.take_shaped(&bias_name, &[hidden_size])? {
        Some(bias) => {
            norm.set_bias(bias)?;
            report.mark_loaded(&bias_name);
        },
        None => report.note_absent(&bias_name),
    }
    Ok(())
}

/// The tensor shapes a decoder-only transformer's layers must have.
///
/// Grouped-query attention means `k_proj`/`v_proj` are narrower than
/// `q_proj`: their output width is `num_key_value_heads * head_dim`. Getting
/// that wrong is the single most common way a "successful" load ends up with
/// transposed or truncated attention weights, so it is a first-class field here
/// rather than an ad-hoc expression per model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderShapes {
    /// Vocabulary size — rows of the embedding matrix and the LM head.
    pub vocab_size: usize,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// Feed-forward inner width.
    pub intermediate_size: usize,
    /// Number of decoder layers.
    pub num_layers: usize,
    /// Output width of `q_proj` (`num_attention_heads * head_dim`).
    pub q_width: usize,
    /// Output width of `k_proj`/`v_proj` (`num_key_value_heads * head_dim`).
    pub kv_width: usize,
}

impl DecoderShapes {
    /// Build the shapes for a model whose `head_dim` is `hidden_size / heads`.
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            q_width: num_attention_heads * head_dim,
            kv_width: num_key_value_heads * head_dim,
        }
    }
}

/// The checkpoint namespaces a task wrapper binds for itself.
///
/// # Why strictness has to be contextual
///
/// A bare encoder load has to tolerate whole head namespaces: HuggingFace ships
/// `cls.predictions.*`, `classifier.*` and `qa_outputs.*` inside the same file
/// as the encoder, so `BertModel::load_from_checkpoint` would fail on every real
/// `bert-base-uncased` checkpoint if it refused them. That tolerance lives in
/// each architecture's `ALLOWED_UNUSED_PREFIXES`.
///
/// A *task wrapper* is in the opposite position. It binds those very namespaces,
/// so an entry left over inside one is not "a head this model does not have" —
/// it is a name this model failed to recognise. Under the encoder's policy alone
/// a typo such as `cls.predictions.transform.dens.weight` starts with `cls.`,
/// lands in [`LoadReport::ignored`], and the load reports success while the
/// dense layer silently keeps its random initialisation.
///
/// [`BoundNamespaces::verify`] closes exactly that gap: after the wrapper has
/// bound its head, every checkpoint entry still unconsumed *inside a namespace
/// the wrapper claims* becomes an error, while namespaces it does not claim stay
/// as tolerant as before. A wrapper that claims `cls.predictions.` therefore
/// still accepts a checkpoint's `cls.seq_relationship.*` NSP head, and a
/// checkpoint with no head at all still loads (nothing is left over — the gap is
/// reported through [`LoadReport::missing`] instead).
#[derive(Debug, Clone, Copy)]
pub struct BoundNamespaces<'a> {
    /// Namespaces (matched with `starts_with`) this wrapper binds itself.
    prefixes: &'a [&'a str],
    /// Non-parameter buffer names (matched with `ends_with`) that stay tolerated
    /// even inside a claimed namespace.
    tolerated_suffixes: &'a [&'a str],
}

impl<'a> BoundNamespaces<'a> {
    /// Claim a set of namespaces with no buffer exceptions.
    pub const fn new(prefixes: &'a [&'a str]) -> Self {
        Self {
            prefixes,
            tolerated_suffixes: &[],
        }
    }

    /// Claim namespaces, excepting registered buffers matched by suffix.
    ///
    /// Needed where a head namespace legitimately carries a non-parameter entry
    /// (a `position_ids` range, a cached mask) that no setter consumes.
    pub const fn with_tolerated_suffixes(
        prefixes: &'a [&'a str],
        tolerated_suffixes: &'a [&'a str],
    ) -> Self {
        Self {
            prefixes,
            tolerated_suffixes,
        }
    }

    /// The namespaces this wrapper claims.
    pub fn prefixes(&self) -> &'a [&'a str] {
        self.prefixes
    }

    /// Whether this entry falls inside a claimed namespace and is not an
    /// explicitly tolerated buffer.
    fn claims(&self, name: &str) -> bool {
        self.prefixes.iter().any(|prefix| name.starts_with(prefix))
            && !self.tolerated_suffixes.iter().any(|suffix| name.ends_with(suffix))
    }

    /// Fail when the checkpoint holds an unrecognised tensor inside a namespace
    /// this wrapper binds.
    ///
    /// Call it *after* every head tensor has been bound, so that
    /// [`LoadReport::mark_loaded`] has already moved the recognised names out of
    /// [`LoadReport::ignored`].
    ///
    /// # Errors
    ///
    /// Fails when any entry under a claimed namespace was left unconsumed,
    /// naming every offender and the namespaces that were claimed.
    pub fn verify(&self, report: &LoadReport) -> Result<()> {
        let unrecognised: Vec<&str> = report
            .ignored
            .iter()
            .map(String::as_str)
            .filter(|name| self.claims(name))
            .collect();
        if unrecognised.is_empty() {
            return Ok(());
        }
        Err(TrustformersError::weight_load_error(format!(
            "checkpoint holds {} tensor(s) this model does not recognise inside the head \
             namespace(s) it binds ({}): {}. A bare encoder load tolerates a whole head \
             namespace it does not bind, but this model binds these names, so an unconsumed \
             entry here is a name it failed to match — most often a misspelling — not an \
             absent head",
            unrecognised.len(),
            self.prefixes.join(", "),
            unrecognised.join(", "),
        )))
    }
}

/// Checkpoint entries a decoder-only HuggingFace export carries that are not
/// learnable parameters.
///
/// `rotary_emb.inv_freq` is a cached frequency table recomputed at construction
/// time, `masked_bias`/`bias` under an attention block are causal-mask buffers,
/// and `position_ids` is a range. Refusing a load over any of these would make
/// every real checkpoint fail, while still refusing over an unknown *weight* is
/// exactly the strictness worth keeping.
pub const DECODER_BUFFER_SUFFIXES: &[&str] = &[
    "rotary_emb.inv_freq",
    "attn.masked_bias",
    "attn.bias",
    "embeddings.position_ids",
    "rotary_emb.cos_cached",
    "rotary_emb.sin_cached",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_loading::checkpoint::{Checkpoint, UnusedTensors};
    use crate::weight_loading::test_support::{build_safetensors, F32Tensor};

    #[test]
    fn bind_linear_copies_the_exact_checkpoint_values() {
        let weight = F32Tensor::ramp("proj.weight", &[3, 2], 1.0);
        let bias = F32Tensor::ramp("proj.bias", &[3], 40.0);
        let bytes = build_safetensors(&[weight.clone(), bias.clone()]);
        let checkpoint = Checkpoint::from_bytes(&bytes).expect("checkpoint must parse");

        let mut layer = Linear::new(2, 3, true);
        let mut binder = checkpoint.binder("");
        bind_linear(&mut binder, "proj", 3, 2, true, &mut layer).expect("binding must succeed");
        binder.finish(UnusedTensors::default()).expect("everything must be consumed");

        assert_eq!(
            layer.weight().data().expect("readable"),
            weight.values,
            "the bound weight must be the checkpoint's, byte for byte"
        );
        assert_eq!(
            layer.bias().expect("bias must be set").data().expect("readable"),
            bias.values
        );
    }

    #[test]
    fn bind_linear_reports_a_shape_mismatch_instead_of_reshaping() {
        let bytes = build_safetensors(&[F32Tensor::ramp("proj.weight", &[6, 1], 1.0)]);
        let checkpoint = Checkpoint::from_bytes(&bytes).expect("checkpoint must parse");
        let mut layer = Linear::new(2, 3, false);
        let mut binder = checkpoint.binder("");
        let err = bind_linear(&mut binder, "proj", 3, 2, false, &mut layer)
            .expect_err("a [6,1] tensor is not a [3,2] weight");
        assert!(err.to_string().contains("[6, 1]"), "unexpected: {err}");
    }

    #[test]
    fn a_missing_parameter_is_reported_rather_than_invented() {
        let bytes = build_safetensors(&[F32Tensor::ramp("other.weight", &[1], 1.0)]);
        let checkpoint = Checkpoint::from_bytes(&bytes).expect("checkpoint must parse");
        let mut layer = Linear::new(2, 3, false);
        let before = layer.weight().data().expect("readable");
        let mut binder = checkpoint.binder("");
        bind_linear(&mut binder, "proj", 3, 2, false, &mut layer).expect("binding must not fail");
        assert_eq!(
            layer.weight().data().expect("readable"),
            before,
            "an absent tensor must leave the layer untouched"
        );
        let err = binder
            .finish(UnusedTensors::default())
            .expect_err("the gap must be reported at finish time");
        assert!(err.to_string().contains("proj.weight"), "unexpected: {err}");
    }

    #[test]
    fn decoder_shapes_track_grouped_query_attention() {
        // 8 query heads, 2 KV heads, head_dim 16: q is 128 wide, k/v are 32.
        let shapes = DecoderShapes::new(100, 128, 256, 2, 8, 2, 16);
        assert_eq!(shapes.q_width, 128);
        assert_eq!(shapes.kv_width, 32);
    }
}
