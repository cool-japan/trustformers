//! # Shared media primitives
//!
//! Real, model-independent signal and image processing shared by the media
//! pipelines (speech-to-text, audio classification, image classification,
//! detection, depth, …).
//!
//! Everything in this module does the arithmetic it advertises:
//!
//! - [`audio_dsp`] decodes RIFF/WAVE containers, resamples, and computes a
//!   Hann-windowed STFT (via the pure-Rust `oxifft` crate) followed by a
//!   Slaney mel filterbank and log compression.
//! - [`image_proc`] decodes Netpbm images (and, with the `vision` feature, any
//!   format the `image` crate supports), resizes with real bilinear
//!   interpolation, centre-crops, and normalises to a CHW `f32` tensor.
//!
//! Nothing here fabricates data. When a required capability is missing the
//! helpers return a structured [`TrustformersError`], and
//! [`unsupported_model`] builds the canonical "this architecture has no real
//! implementation" error used by every media pipeline.

pub mod audio_dsp;
pub mod image_proc;

use crate::error::TrustformersError;

/// Build the canonical "architecture not supported" error for a media task.
///
/// Media pipelines advertise a task (`"image-classification"`, `"depth-estimation"`, …)
/// but can only actually execute the architectures that have a real
/// implementation in `trustformers-models` *and* a wired-up backend here. When
/// a caller asks for anything else the pipeline must fail loudly rather than
/// return a synthesised result.
///
/// The returned [`TrustformersError::FeatureUnavailable`] carries the full list
/// of architectures that *are* supported in its `alternatives` field, so the
/// caller can recover programmatically.
///
/// `supported` may be empty — that is the honest answer for tasks whose
/// backbones are not implemented at all yet.
pub fn unsupported_model(task: &str, requested: &str, supported: &[&str]) -> TrustformersError {
    let supported_list = if supported.is_empty() {
        "none (no backbone for this task is implemented yet)".to_string()
    } else {
        supported.join(", ")
    };
    TrustformersError::FeatureUnavailable {
        message: format!(
            "no real model implementation for `{requested}` in the `{task}` pipeline; \
             supported architectures: {supported_list}. This pipeline never returns \
             synthesised results, so the request cannot be served."
        ),
        feature: format!("{task}:{requested}"),
        suggestion: Some(if supported.is_empty() {
            format!(
                "The `{task}` pipeline has real pre-/post-processing but no model backend. \
                 Use the preprocessing helpers directly and run inference with your own model."
            )
        } else {
            format!("Supply one of: {supported_list}.")
        }),
        alternatives: supported.iter().map(|s| (*s).to_string()).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_model_lists_alternatives() {
        let err = unsupported_model("image-classification", "resnet-50", &["vit", "clip"]);
        match err {
            TrustformersError::FeatureUnavailable {
                ref alternatives,
                ref message,
                ..
            } => {
                assert_eq!(alternatives, &["vit".to_string(), "clip".to_string()]);
                assert!(message.contains("resnet-50"), "message: {message}");
                assert!(message.contains("vit"), "message: {message}");
            },
            other => panic!("expected FeatureUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_model_with_no_backends_is_explicit() {
        let err = unsupported_model("depth-estimation", "Intel/dpt-large", &[]);
        let text = err.to_string();
        assert!(text.contains("no backbone"), "text: {text}");
    }
}
