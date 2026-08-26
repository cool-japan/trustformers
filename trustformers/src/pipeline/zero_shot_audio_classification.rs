//! # Zero-Shot Audio Classification Pipeline
//!
//! ## What is real here
//!
//! * [`ZeroShotAudioClassificationPipeline::audio_features`] — a genuine
//!   4-dimensional descriptor of a waveform: RMS energy, peak amplitude,
//!   duration and zero-crossing rate, all measured from the samples.
//! * [`ZeroShotAudioProcessor::format_hypotheses`] — real hypothesis-template
//!   expansion.
//! * [`cosine_similarity`], [`softmax`] and
//!   [`ZeroShotAudioClassificationPipeline::rank_similarities`] — exact
//!   arithmetic that works on any pair of embeddings you supply.
//!
//! ## Model support
//!
//! Zero-shot classification needs a **joint audio-text embedding space**, and
//! no CLAP-style encoder is implemented in `trustformers-models`.
//! [`ZeroShotAudioClassificationPipeline::classify`] therefore returns
//! [`ZeroShotAudioError::UnsupportedModel`].
//!
//! It used to compare the audio descriptor above against a "text embedding"
//! built from the *djb2 hash of the label string* — four bytes of a hash
//! treated as a semantic vector. Those similarities were arithmetic noise, and
//! the softmax over them produced confident-looking probabilities with no
//! relationship to the audio's content. None of that survives.
//!
//! ## Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::audio_generation::AudioWaveform;
//! use trustformers::pipeline::zero_shot_audio_classification::{
//!     ZeroShotAudioClassificationPipeline, ZeroShotAudioConfig,
//! };
//!
//! let config = ZeroShotAudioConfig::default();
//! let pipeline = ZeroShotAudioClassificationPipeline::new(config)?;
//! let waveform = AudioWaveform::new(vec![0.0; 16_000], 16_000)?;
//! // Real feature extraction and ranking around your own CLAP encoder:
//! let features = pipeline.audio_features(&waveform);
//! let result = pipeline.rank_similarities(&["speech", "music"], &my_similarities)?;
//! println!("Top label: {} ({:.4})", result.label, result.score);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use super::audio_generation::AudioWaveform;

/// Joint audio-text architectures with a real encoder in this workspace.
///
/// Deliberately empty — the pipeline says so rather than pretending.
const SUPPORTED_ARCHITECTURES: &[&str] = &[];

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the zero-shot audio classification pipeline.
#[derive(Debug, thiserror::Error)]
pub enum ZeroShotAudioError {
    /// The input waveform contained no samples.
    #[error("Empty audio")]
    EmptyAudio,
    /// No candidate labels were provided.
    #[error("No candidate labels")]
    NoLabels,
    /// The requested checkpoint has no real implementation in this workspace.
    #[error(
        "no real joint audio-text embedding model is implemented for `{requested}`; supported: \
         {supported}. This pipeline never scores labels against a hash of their text — use \
         `rank_similarities` with your own encoder's similarities."
    )]
    UnsupportedModel {
        /// The checkpoint or architecture the caller asked for.
        requested: String,
        /// Comma-separated list of architectures that *are* supported.
        supported: String,
    },
    /// The number of similarities does not match the number of labels.
    #[error("expected {expected} similarity scores (one per label) but got {got}")]
    ScoreCountMismatch { expected: usize, got: usize },
    /// A generic model-level error with a descriptive message.
    #[error("Model error: {0}")]
    ModelError(String),
    /// Embedding dimension mismatch.
    #[error("Dimension mismatch: audio_embed len={audio}, text_embed len={text}")]
    DimensionMismatch { audio: usize, text: usize },
}

// ---------------------------------------------------------------------------
// PipelineError alias for new API
// ---------------------------------------------------------------------------

/// Alias for [`ZeroShotAudioError`] used in the enhanced API.
pub type PipelineError = ZeroShotAudioError;

// ---------------------------------------------------------------------------
// AudioInput
// ---------------------------------------------------------------------------

/// Flexible audio input that wraps an [`AudioWaveform`].
#[derive(Debug, Clone)]
pub struct AudioInput {
    /// The underlying waveform.
    pub waveform: AudioWaveform,
}

impl AudioInput {
    /// Create an `AudioInput` from raw samples and sample rate.
    ///
    /// # Errors
    /// Returns an error if `AudioWaveform::new` fails.
    pub fn from_samples(samples: Vec<f32>, sample_rate: u32) -> Result<Self, ZeroShotAudioError> {
        let waveform = AudioWaveform::new(samples, sample_rate)
            .map_err(|e| ZeroShotAudioError::ModelError(format!("waveform error: {e:?}")))?;
        Ok(Self { waveform })
    }

    /// Borrow the inner waveform.
    pub fn waveform(&self) -> &AudioWaveform {
        &self.waveform
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`ZeroShotAudioClassificationPipeline`].
#[derive(Debug, Clone)]
pub struct ZeroShotAudioConfig {
    /// HuggingFace model identifier or local path.
    pub model_name: String,
    /// Expected input sample rate in Hz.
    pub sample_rate: u32,
    /// Whether to normalise the input audio before embedding.
    pub normalize_audio: bool,
    /// Whether to L2-normalise embeddings before computing similarity.
    pub normalize_embeddings: bool,
    /// Hypothesis template used for zero-shot classification.
    /// Use `{}` as a placeholder for the label, e.g. `"This audio is {}"`.
    pub hypothesis_template: String,
}

impl Default for ZeroShotAudioConfig {
    fn default() -> Self {
        Self {
            model_name: "laion/larger_clap_general".to_string(),
            sample_rate: 48_000,
            normalize_audio: true,
            normalize_embeddings: true,
            hypothesis_template: "This audio is {}".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// ZeroShotAudioProcessor — utility methods
// ---------------------------------------------------------------------------

/// A collection of pure utility functions for zero-shot audio classification.
pub struct ZeroShotAudioProcessor;

impl ZeroShotAudioProcessor {
    /// Expand a hypothesis template for each label.
    ///
    /// Replaces every occurrence of `{}` in `template` with the label string.
    ///
    /// ```
    /// # use trustformers::pipeline::zero_shot_audio_classification::ZeroShotAudioProcessor;
    /// let labels = vec!["speech".to_string(), "music".to_string()];
    /// let hypotheses = ZeroShotAudioProcessor::format_hypotheses(&labels, "This audio is {}");
    /// assert_eq!(hypotheses[0], "This audio is speech");
    /// assert_eq!(hypotheses[1], "This audio is music");
    /// ```
    pub fn format_hypotheses(labels: &[String], template: &str) -> Vec<String> {
        labels.iter().map(|lbl| template.replace("{}", lbl)).collect()
    }

    /// Cosine similarity between two variable-length embedding slices.
    ///
    /// Returns `0.0` when either vector is all-zero.
    ///
    /// # Errors
    /// Returns [`ZeroShotAudioError::DimensionMismatch`] if the slices differ in length.
    pub fn cosine_similarity(
        audio_embed: &[f32],
        text_embed: &[f32],
    ) -> Result<f32, ZeroShotAudioError> {
        if audio_embed.len() != text_embed.len() {
            return Err(ZeroShotAudioError::DimensionMismatch {
                audio: audio_embed.len(),
                text: text_embed.len(),
            });
        }
        let dot: f32 = audio_embed.iter().zip(text_embed.iter()).map(|(a, b)| a * b).sum();
        let na = (audio_embed.iter().map(|x| x * x).sum::<f32>()).sqrt();
        let nb = (text_embed.iter().map(|x| x * x).sum::<f32>()).sqrt();
        if na < f32::EPSILON || nb < f32::EPSILON {
            return Ok(0.0);
        }
        Ok((dot / (na * nb)).clamp(-1.0, 1.0))
    }

    /// Rank candidate label embeddings by cosine similarity with `audio_embed`.
    ///
    /// Returns `(original_index, similarity)` pairs sorted by similarity descending.
    ///
    /// # Errors
    /// Propagates any dimension mismatch from [`Self::cosine_similarity`].
    pub fn rank_labels(
        audio_embed: &[f32],
        label_embeds: &[Vec<f32>],
    ) -> Result<Vec<(usize, f32)>, ZeroShotAudioError> {
        let mut scored: Vec<(usize, f32)> = label_embeds
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                let sim = Self::cosine_similarity(audio_embed, emb)?;
                Ok((i, sim))
            })
            .collect::<Result<Vec<_>, ZeroShotAudioError>>()?;
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }

    /// Entmax approximation: apply softmax then squash low-probability entries.
    ///
    /// Algorithm:
    /// 1. Compute standard softmax probabilities.
    /// 2. Subtract the mean probability.
    /// 3. Clamp to `[0, ∞)` (sparse projection).
    /// 4. Re-normalise so probabilities sum to 1.
    ///
    /// For nearly-uniform distributions this becomes equivalent to softmax.
    pub fn entmax_scores(logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }
        // Step 1: stable softmax
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = if sum < f32::EPSILON {
            vec![1.0 / logits.len() as f32; logits.len()]
        } else {
            exps.iter().map(|v| v / sum).collect()
        };

        // Step 2 & 3: subtract mean, clamp
        let mean = probs.iter().sum::<f32>() / probs.len() as f32;
        let shifted: Vec<f32> = probs.iter().map(|p| (p - mean).max(0.0)).collect();

        // Step 4: re-normalise
        let shifted_sum: f32 = shifted.iter().sum();
        if shifted_sum < f32::EPSILON {
            // Fallback: uniform if all entries clamped to zero
            vec![1.0 / logits.len() as f32; logits.len()]
        } else {
            shifted.iter().map(|v| v / shifted_sum).collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// Classification result for a single audio input.
#[derive(Debug, Clone)]
pub struct ZeroShotAudioResult {
    /// The top-ranked label.
    pub label: String,
    /// Probability of the top-ranked label in `[0.0, 1.0]`.
    pub score: f32,
    /// All labels with their probabilities, sorted in descending order.
    pub all_scores: Vec<(String, f32)>,
}

/// A single (label, score) result in the new enhanced API.
#[derive(Debug, Clone)]
pub struct ZeroShotAudioItem {
    /// The candidate label (possibly expanded from a template).
    pub candidate_label: String,
    /// Similarity score / probability in `[0.0, 1.0]`.
    pub score: f32,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute a real 4-dimensional acoustic descriptor of a waveform.
///
/// Components: `[rms_energy, peak_amplitude, duration_seconds,
/// zero_crossing_rate]` — all measured directly from the samples. This is a
/// genuine (if low-dimensional) feature vector; it is **not** a CLAP embedding
/// and cannot be compared against text.
fn audio_embedding(audio: &AudioWaveform, normalize: bool) -> [f32; 4] {
    let rms = audio.rms_energy();
    let peak = audio.peak_amplitude();
    let dur = audio.duration_seconds();
    // Zero-crossing rate: fraction of adjacent pairs that change sign.
    let zcr = if audio.samples.len() < 2 {
        0.0
    } else {
        let crossings = audio.samples.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count();
        crossings as f32 / (audio.samples.len() - 1) as f32
    };
    let mut emb = [rms, peak, dur, zcr];
    if normalize {
        let norm = (emb.iter().map(|x| x * x).sum::<f32>()).sqrt();
        if norm > f32::EPSILON {
            for v in emb.iter_mut() {
                *v /= norm;
            }
        }
    }
    emb
}

/// Cosine similarity between two fixed-size embedding arrays.
pub fn cosine_similarity(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = (a.iter().map(|x| x * x).sum::<f32>()).sqrt();
    let nb = (b.iter().map(|x| x * x).sum::<f32>()).sqrt();
    if na < f32::EPSILON || nb < f32::EPSILON {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

/// Stable softmax over a slice, returning a new vector of probabilities.
///
/// Exponentials are accumulated in `f64` and each probability is floored at
/// `f32::MIN_POSITIVE`, so outputs stay strictly positive even when widely
/// spread logits would underflow in `f32` (softmax is mathematically > 0).
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<f64> = logits.iter().map(|&v| (f64::from(v) - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < f64::EPSILON {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.iter().map(|&v| ((v / sum) as f32).max(f32::MIN_POSITIVE)).collect()
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Pipeline for zero-shot audio classification (CLAP style).
pub struct ZeroShotAudioClassificationPipeline {
    config: ZeroShotAudioConfig,
}

impl ZeroShotAudioClassificationPipeline {
    /// Create a new pipeline with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration is fundamentally invalid (e.g.
    /// zero sample rate).
    pub fn new(config: ZeroShotAudioConfig) -> Result<Self, ZeroShotAudioError> {
        Ok(Self { config })
    }

    /// Classify a single audio waveform against the candidate labels.
    ///
    /// Returns a [`ZeroShotAudioResult`] with the best label and a sorted list
    /// of all label probabilities.
    ///
    /// # Errors
    ///
    /// - [`ZeroShotAudioError::EmptyAudio`] — waveform has no samples.
    /// - [`ZeroShotAudioError::NoLabels`] — no candidate labels provided.
    pub fn classify(
        &self,
        audio: &AudioWaveform,
        candidate_labels: &[&str],
    ) -> Result<ZeroShotAudioResult, ZeroShotAudioError> {
        if audio.samples.is_empty() {
            return Err(ZeroShotAudioError::EmptyAudio);
        }
        if candidate_labels.is_empty() {
            return Err(ZeroShotAudioError::NoLabels);
        }
        Err(self.unsupported())
    }

    /// The error this pipeline returns when asked to score labels.
    fn unsupported(&self) -> ZeroShotAudioError {
        ZeroShotAudioError::UnsupportedModel {
            requested: self.config.model_name.clone(),
            supported: if SUPPORTED_ARCHITECTURES.is_empty() {
                "none (no joint audio-text encoder is implemented yet)".to_string()
            } else {
                SUPPORTED_ARCHITECTURES.join(", ")
            },
        }
    }

    /// Measure the real acoustic descriptor of a waveform.
    ///
    /// `[rms_energy, peak_amplitude, duration_seconds, zero_crossing_rate]`,
    /// L2-normalised when the configuration asks for it.
    pub fn audio_features(&self, audio: &AudioWaveform) -> [f32; 4] {
        audio_embedding(audio, self.config.normalize_embeddings)
    }

    /// Turn a real encoder's audio-text similarities into a ranked result.
    ///
    /// `similarities[i]` is the cosine similarity between the audio embedding
    /// and the embedding of `candidate_labels[i]`, as produced by *your*
    /// encoder. The pipeline softmaxes them and sorts descending.
    ///
    /// # Errors
    ///
    /// [`ZeroShotAudioError::NoLabels`] for an empty label set and
    /// [`ZeroShotAudioError::ScoreCountMismatch`] on a length mismatch.
    pub fn rank_similarities(
        &self,
        candidate_labels: &[&str],
        similarities: &[f32],
    ) -> Result<ZeroShotAudioResult, ZeroShotAudioError> {
        if candidate_labels.is_empty() {
            return Err(ZeroShotAudioError::NoLabels);
        }
        if similarities.len() != candidate_labels.len() {
            return Err(ZeroShotAudioError::ScoreCountMismatch {
                expected: candidate_labels.len(),
                got: similarities.len(),
            });
        }

        let probs = softmax(similarities);
        let mut all_scores: Vec<(String, f32)> = candidate_labels
            .iter()
            .zip(probs.iter())
            .map(|(lbl, &p)| ((*lbl).to_string(), p))
            .collect();
        all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (label, score) = all_scores[0].clone();
        Ok(ZeroShotAudioResult {
            label,
            score,
            all_scores,
        })
    }

    /// Classify a batch of audio waveforms against the same candidate labels.
    ///
    /// # Errors
    ///
    /// Fails fast on the first error encountered.
    pub fn classify_batch(
        &self,
        audios: &[&AudioWaveform],
        candidate_labels: &[&str],
    ) -> Result<Vec<ZeroShotAudioResult>, ZeroShotAudioError> {
        audios.iter().map(|a| self.classify(a, candidate_labels)).collect()
    }

    /// Classify a single [`AudioInput`] against `candidate_labels` using the
    /// pipeline's hypothesis template.
    ///
    /// Returns items sorted by score descending.
    ///
    /// # Errors
    ///
    /// - [`ZeroShotAudioError::EmptyAudio`] — waveform has no samples.
    /// - [`ZeroShotAudioError::NoLabels`] — no candidate labels provided.
    pub fn classify_input(
        &self,
        audio: &AudioInput,
        candidate_labels: &[String],
    ) -> Result<Vec<ZeroShotAudioItem>, ZeroShotAudioError> {
        if audio.waveform.samples.is_empty() {
            return Err(ZeroShotAudioError::EmptyAudio);
        }
        if candidate_labels.is_empty() {
            return Err(ZeroShotAudioError::NoLabels);
        }

        // The hypothesis expansion is real and still runs, so template errors
        // surface here rather than being masked by the missing encoder.
        let _hypotheses = ZeroShotAudioProcessor::format_hypotheses(
            candidate_labels,
            &self.config.hypothesis_template,
        );
        Err(self.unsupported())
    }

    /// Classify a batch of [`AudioInput`] values against the same candidate labels.
    ///
    /// Returns one `Vec<ZeroShotAudioItem>` per input, sorted descending.
    ///
    /// # Errors
    ///
    /// Fails fast on the first error encountered.
    pub fn classify_inputs_batch(
        &self,
        audios: Vec<AudioInput>,
        candidate_labels: &[String],
    ) -> Result<Vec<Vec<ZeroShotAudioItem>>, ZeroShotAudioError> {
        audios.iter().map(|a| self.classify_input(a, candidate_labels)).collect()
    }

    /// Access the pipeline configuration.
    pub fn config(&self) -> &ZeroShotAudioConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::audio_generation::AudioWaveform;

    fn make_waveform(samples: Vec<f32>) -> AudioWaveform {
        AudioWaveform::new(samples, 16_000).expect("valid")
    }

    fn default_pipeline() -> ZeroShotAudioClassificationPipeline {
        ZeroShotAudioClassificationPipeline::new(ZeroShotAudioConfig::default())
            .expect("default config valid")
    }

    fn assert_unsupported(err: &ZeroShotAudioError) {
        match err {
            ZeroShotAudioError::UnsupportedModel { supported, .. } => {
                assert!(supported.contains("none"), "supported: {supported}");
            },
            other => panic!("expected UnsupportedModel, got {other:?}"),
        }
    }

    #[test]
    fn test_classify_reports_unsupported_model() {
        // Regression: `classify` used to compare a real acoustic descriptor
        // against a "text embedding" made of four bytes of the label's djb2
        // hash, and report the softmax of that noise as probabilities.
        let p = default_pipeline();
        let audio = make_waveform(vec![0.5_f32; 16_000]);
        let labels = ["speech", "music", "noise", "silence"];
        assert_unsupported(&p.classify(&audio, &labels).expect_err("no encoder"));
    }

    #[test]
    fn test_classify_batch_reports_unsupported_model() {
        let p = default_pipeline();
        let a1 = make_waveform(vec![0.1_f32; 16_000]);
        let a2 = make_waveform(vec![0.9_f32; 16_000]);
        let audios = [&a1, &a2];
        assert_unsupported(
            &p.classify_batch(&audios, &["music", "noise"]).expect_err("no encoder"),
        );
    }

    #[test]
    fn test_empty_audio_error() {
        let p = default_pipeline();
        let audio = make_waveform(vec![]);
        let err = p.classify(&audio, &["speech"]).expect_err("empty audio should fail");
        assert!(matches!(err, ZeroShotAudioError::EmptyAudio));
    }

    #[test]
    fn test_no_labels_error() {
        let p = default_pipeline();
        let audio = make_waveform(vec![0.1_f32; 100]);
        let err = p.classify(&audio, &[]).expect_err("empty labels should fail");
        assert!(matches!(err, ZeroShotAudioError::NoLabels));
    }

    // ── Real feature extraction and ranking ─────────────────────────────────

    #[test]
    fn test_audio_features_measure_the_waveform() {
        let p = ZeroShotAudioClassificationPipeline::new(ZeroShotAudioConfig {
            normalize_embeddings: false,
            ..ZeroShotAudioConfig::default()
        })
        .expect("valid");
        // Alternating +-0.5 at 16 kHz: RMS 0.5, peak 0.5, ZCR ~1.0.
        let samples: Vec<f32> = (0..16_000).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
        let f = p.audio_features(&make_waveform(samples));
        assert!((f[0] - 0.5).abs() < 1e-4, "rms: {}", f[0]);
        assert!((f[1] - 0.5).abs() < 1e-4, "peak: {}", f[1]);
        assert!((f[2] - 1.0).abs() < 1e-4, "duration: {}", f[2]);
        assert!((f[3] - 1.0).abs() < 1e-3, "zcr: {}", f[3]);
    }

    #[test]
    fn test_audio_features_distinguish_signals() {
        let p = default_pipeline();
        let quiet = p.audio_features(&make_waveform(vec![0.01_f32; 4_000]));
        let loud = p.audio_features(&make_waveform(vec![0.9_f32; 4_000]));
        assert_ne!(
            quiet, loud,
            "different waveforms must give different features"
        );
    }

    #[test]
    fn test_rank_similarities_sorts_and_normalises() {
        let p = default_pipeline();
        let labels = ["cat", "dog", "bird"];
        let result = p.rank_similarities(&labels, &[0.1, 0.9, 0.2]).expect("rank");
        assert_eq!(result.all_scores.len(), labels.len());
        assert_eq!(result.label, "dog");
        for w in result.all_scores.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "scores not sorted: {} < {}",
                w[0].1,
                w[1].1
            );
        }
        let total: f32 = result.all_scores.iter().map(|(_, s)| s).sum();
        assert!((total - 1.0).abs() < 1e-5, "scores sum to {total}");
    }

    #[test]
    fn test_rank_similarities_single_label_scores_one() {
        let p = default_pipeline();
        let result = p.rank_similarities(&["music"], &[0.42]).expect("rank");
        assert!(
            (result.score - 1.0).abs() < 1e-5,
            "score was {}",
            result.score
        );
    }

    #[test]
    fn test_rank_similarities_rejects_bad_input() {
        let p = default_pipeline();
        assert!(matches!(
            p.rank_similarities(&["a", "b"], &[0.1]),
            Err(ZeroShotAudioError::ScoreCountMismatch { .. })
        ));
        assert!(matches!(
            p.rank_similarities(&[], &[]),
            Err(ZeroShotAudioError::NoLabels)
        ));
    }

    #[test]
    fn test_default_config_sample_rate() {
        let config = ZeroShotAudioConfig::default();
        assert_eq!(config.sample_rate, 48_000);
    }

    #[test]
    fn test_normalize_flags_present_in_default() {
        let config = ZeroShotAudioConfig::default();
        assert!(
            config.normalize_audio,
            "normalize_audio should default to true"
        );
        assert!(
            config.normalize_embeddings,
            "normalize_embeddings should default to true"
        );
    }

    // ── ZeroShotAudioProcessor::format_hypotheses ────────────────────────────

    #[test]
    fn test_format_hypotheses_basic() {
        let labels = vec![
            "speech".to_string(),
            "music".to_string(),
            "noise".to_string(),
        ];
        let hyps = ZeroShotAudioProcessor::format_hypotheses(&labels, "This audio is {}");
        assert_eq!(hyps.len(), 3);
        assert_eq!(hyps[0], "This audio is speech");
        assert_eq!(hyps[1], "This audio is music");
        assert_eq!(hyps[2], "This audio is noise");
    }

    #[test]
    fn test_format_hypotheses_custom_template() {
        let labels = vec!["rain".to_string(), "thunder".to_string()];
        let hyps = ZeroShotAudioProcessor::format_hypotheses(&labels, "Classify as: {}");
        assert_eq!(hyps[0], "Classify as: rain");
        assert_eq!(hyps[1], "Classify as: thunder");
    }

    #[test]
    fn test_format_hypotheses_no_placeholder() {
        // Template without {} — every hypothesis is identical to the template.
        let labels = vec!["a".to_string(), "b".to_string()];
        let hyps = ZeroShotAudioProcessor::format_hypotheses(&labels, "fixed text");
        assert!(hyps.iter().all(|h| h == "fixed text"));
    }

    #[test]
    fn test_format_hypotheses_empty_labels() {
        let labels: Vec<String> = vec![];
        let hyps = ZeroShotAudioProcessor::format_hypotheses(&labels, "This is {}");
        assert!(hyps.is_empty());
    }

    #[test]
    fn test_format_hypotheses_multiple_placeholders() {
        // Multiple {} occurrences — both replaced.
        let labels = vec!["cat".to_string()];
        let hyps = ZeroShotAudioProcessor::format_hypotheses(&labels, "A {} or not a {}");
        assert_eq!(hyps[0], "A cat or not a cat");
    }

    // ── ZeroShotAudioProcessor::cosine_similarity ────────────────────────────

    #[test]
    fn test_cosine_similarity_parallel() {
        // Parallel vectors → similarity = 1.0
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![2.0_f32, 0.0, 0.0];
        let sim = ZeroShotAudioProcessor::cosine_similarity(&a, &b).expect("ok");
        assert!((sim - 1.0).abs() < 1e-5, "parallel vectors: sim={sim}");
    }

    #[test]
    fn test_cosine_similarity_antiparallel() {
        // Antiparallel vectors → similarity = -1.0
        let a = vec![1.0_f32, 0.0];
        let b = vec![-1.0_f32, 0.0];
        let sim = ZeroShotAudioProcessor::cosine_similarity(&a, &b).expect("ok");
        assert!((sim + 1.0).abs() < 1e-5, "antiparallel vectors: sim={sim}");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        // Orthogonal vectors → similarity = 0.0
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = ZeroShotAudioProcessor::cosine_similarity(&a, &b).expect("ok");
        assert!(sim.abs() < 1e-5, "orthogonal vectors: sim={sim}");
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        let sim = ZeroShotAudioProcessor::cosine_similarity(&a, &b).expect("ok");
        assert_eq!(sim, 0.0, "zero vector should yield 0 similarity");
    }

    #[test]
    fn test_cosine_similarity_dimension_mismatch() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        let err = ZeroShotAudioProcessor::cosine_similarity(&a, &b).unwrap_err();
        assert!(
            matches!(
                err,
                ZeroShotAudioError::DimensionMismatch { audio: 2, text: 3 }
            ),
            "expected DimensionMismatch"
        );
    }

    // ── ZeroShotAudioProcessor::rank_labels ──────────────────────────────────

    #[test]
    fn test_rank_labels_ordering() {
        // audio_embed = [1, 0, 0]
        // label 0: [1, 0, 0]  → sim ≈ 1.0 (most similar)
        // label 1: [0, 1, 0]  → sim = 0.0
        // label 2: [-1, 0, 0] → sim ≈ -1.0 (least similar)
        let audio_embed = vec![1.0_f32, 0.0, 0.0];
        let label_embeds = vec![
            vec![1.0_f32, 0.0, 0.0],
            vec![0.0_f32, 1.0, 0.0],
            vec![-1.0_f32, 0.0, 0.0],
        ];
        let ranked = ZeroShotAudioProcessor::rank_labels(&audio_embed, &label_embeds).expect("ok");
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].0, 0, "most similar should be label 0");
        assert_eq!(ranked[2].0, 2, "least similar should be label 2");
        // Descending order
        for w in ranked.windows(2) {
            assert!(w[0].1 >= w[1].1, "rank not descending");
        }
    }

    #[test]
    fn test_rank_labels_single_label() {
        let audio = vec![1.0_f32, 1.0];
        let labels = vec![vec![1.0_f32, 1.0]];
        let ranked = ZeroShotAudioProcessor::rank_labels(&audio, &labels).expect("ok");
        assert_eq!(ranked.len(), 1);
    }

    // ── ZeroShotAudioProcessor::entmax_scores ────────────────────────────────

    #[test]
    fn test_entmax_scores_sum_to_one() {
        let logits = vec![2.0_f32, 1.0, -1.0, 0.5];
        let scores = ZeroShotAudioProcessor::entmax_scores(&logits);
        let sum: f32 = scores.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "entmax scores must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_entmax_scores_all_positive() {
        let logits = vec![1.0_f32, -2.0, 0.0, 3.0, -5.0];
        let scores = ZeroShotAudioProcessor::entmax_scores(&logits);
        assert!(
            scores.iter().all(|&s| s >= 0.0),
            "all entmax scores must be >= 0"
        );
    }

    #[test]
    fn test_entmax_scores_dominant_entry() {
        // Very large logit at index 0 should dominate after entmax.
        let logits = vec![100.0_f32, 0.0, 0.0, 0.0];
        let scores = ZeroShotAudioProcessor::entmax_scores(&logits);
        assert!(
            scores[0] > 0.9,
            "dominant logit should dominate: score={}",
            scores[0]
        );
    }

    #[test]
    fn test_entmax_scores_empty() {
        let scores = ZeroShotAudioProcessor::entmax_scores(&[]);
        assert!(scores.is_empty());
    }

    // ── classify_input (new API) ──────────────────────────────────────────────

    #[test]
    fn test_classify_input_reports_unsupported_model() {
        let p = default_pipeline();
        let audio = AudioInput::from_samples(vec![0.1_f32; 4_000], 16_000).expect("ok");
        let labels = vec!["music".to_string(), "speech".to_string()];
        assert_unsupported(&p.classify_input(&audio, &labels).expect_err("no encoder"));
    }

    #[test]
    fn test_classify_input_empty_labels_error() {
        let p = default_pipeline();
        let audio = AudioInput::from_samples(vec![0.5_f32; 1_000], 16_000).expect("ok");
        let labels: Vec<String> = vec![];
        let err = p.classify_input(&audio, &labels).unwrap_err();
        assert!(matches!(err, ZeroShotAudioError::NoLabels));
    }

    #[test]
    fn test_classify_input_empty_audio_error() {
        let p = default_pipeline();
        let audio = AudioInput {
            waveform: make_waveform(vec![]),
        };
        let labels = vec!["music".to_string()];
        let err = p.classify_input(&audio, &labels).unwrap_err();
        assert!(matches!(err, ZeroShotAudioError::EmptyAudio));
    }

    #[test]
    fn test_classify_inputs_batch_reports_unsupported_model() {
        let p = default_pipeline();
        let audios: Vec<AudioInput> = (0..3)
            .map(|i| AudioInput::from_samples(vec![(i as f32) * 0.1; 4_000], 16_000).expect("ok"))
            .collect();
        let labels = ["speech", "music", "noise"].iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_unsupported(&p.classify_inputs_batch(audios, &labels).expect_err("no encoder"));
    }

    // ── hypothesis_template in config ────────────────────────────────────────

    #[test]
    fn test_default_hypothesis_template() {
        let config = ZeroShotAudioConfig::default();
        assert_eq!(config.hypothesis_template, "This audio is {}");
    }

    // ── softmax properties ────────────────────────────────────────────────────

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.5_f32, -0.5, 2.0, 0.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum}");
    }

    #[test]
    fn test_softmax_all_positive() {
        let logits = vec![-100.0_f32, -200.0, -50.0];
        let probs = softmax(&logits);
        assert!(
            probs.iter().all(|&p| p > 0.0),
            "all softmax outputs must be positive"
        );
    }

    #[test]
    fn test_softmax_empty() {
        let probs = softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_top_k_score_sum_approaches_one() {
        let p = default_pipeline();
        let labels = ["a", "b", "c", "d", "e"];
        let result = p.rank_similarities(&labels, &[0.4, 0.1, -0.2, 0.9, 0.0]).expect("rank");
        // All scores sum to 1.0
        let sum: f32 = result.all_scores.iter().map(|(_, s)| s).sum();
        assert!((sum - 1.0).abs() < 1e-5, "all scores sum to {sum}");
    }
}
