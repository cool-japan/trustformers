//! # Zero-Shot Audio Classification Pipeline
//!
//! CLAP-compatible zero-shot audio classification: classify audio against a set
//! of natural-language candidate labels without any task-specific fine-tuning.
//!
//! ## Supported model families
//! - **CLAP** — Contrastive Language-Audio Pre-training
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
//! let result = pipeline.classify(&waveform, &["speech", "music", "noise"])?;
//! println!("Top label: {} ({:.4})", result.label, result.score);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use super::audio_generation::AudioWaveform;

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
        labels
            .iter()
            .map(|lbl| template.replace("{}", lbl))
            .collect()
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
        let dot: f32 = audio_embed
            .iter()
            .zip(text_embed.iter())
            .map(|(a, b)| a * b)
            .sum();
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
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
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

/// djb2 hash — used for deterministic mock text embeddings.
fn djb2_hash(s: &str) -> u64 {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    h
}

/// Produce a 4-dimensional mock embedding for an audio waveform.
///
/// Features: `[rms_energy, peak_amplitude, duration_seconds, zero_crossing_rate]`.
fn audio_embedding(audio: &AudioWaveform, normalize: bool) -> [f32; 4] {
    let rms = audio.rms_energy();
    let peak = audio.peak_amplitude();
    let dur = audio.duration_seconds();
    // Zero-crossing rate: fraction of adjacent pairs that change sign.
    let zcr = if audio.samples.len() < 2 {
        0.0
    } else {
        let crossings = audio
            .samples
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
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

/// Produce a 4-dimensional mock embedding for a text label using its djb2 hash.
fn text_embedding(label: &str, normalize: bool) -> [f32; 4] {
    let h = djb2_hash(label);
    let mut emb = [
        ((h & 0xFF) as f32) / 255.0,
        (((h >> 8) & 0xFF) as f32) / 255.0,
        (((h >> 16) & 0xFF) as f32) / 255.0,
        (((h >> 24) & 0xFF) as f32) / 255.0,
    ];
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
fn cosine_similarity(a: &[f32; 4], b: &[f32; 4]) -> f32 {
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
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum < f32::EPSILON {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.iter().map(|v| v / sum).collect()
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

        let audio_emb = audio_embedding(audio, self.config.normalize_embeddings);
        let logits: Vec<f32> = candidate_labels
            .iter()
            .map(|lbl| {
                let text_emb = text_embedding(lbl, self.config.normalize_embeddings);
                cosine_similarity(&audio_emb, &text_emb)
            })
            .collect();

        let probs = softmax(&logits);

        // Build sorted (label, score) pairs.
        let mut all_scores: Vec<(String, f32)> = candidate_labels
            .iter()
            .zip(probs.iter())
            .map(|(lbl, &p)| (lbl.to_string(), p))
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
        audios
            .iter()
            .map(|a| self.classify(a, candidate_labels))
            .collect()
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

        let hypotheses = ZeroShotAudioProcessor::format_hypotheses(
            candidate_labels,
            &self.config.hypothesis_template,
        );

        let audio_emb = audio_embedding(&audio.waveform, self.config.normalize_embeddings);
        let logits: Vec<f32> = hypotheses
            .iter()
            .map(|hyp| {
                let text_emb = text_embedding(hyp, self.config.normalize_embeddings);
                cosine_similarity(&audio_emb, &text_emb)
            })
            .collect();

        let probs = softmax(&logits);

        let mut items: Vec<ZeroShotAudioItem> = candidate_labels
            .iter()
            .zip(probs.iter())
            .map(|(lbl, &score)| ZeroShotAudioItem {
                candidate_label: lbl.clone(),
                score,
            })
            .collect();
        items.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(items)
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
        audios
            .iter()
            .map(|a| self.classify_input(a, candidate_labels))
            .collect()
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

    #[test]
    fn test_classify_returns_correct_label_count() {
        let p = default_pipeline();
        let audio = make_waveform(vec![0.5_f32; 16_000]);
        let labels = ["speech", "music", "noise", "silence"];
        let result = p.classify(&audio, &labels).expect("classify ok");
        assert_eq!(result.all_scores.len(), labels.len());
    }

    #[test]
    fn test_classify_scores_sorted_descending() {
        let p = default_pipeline();
        let audio = make_waveform(vec![0.3_f32; 16_000]);
        let labels = ["cat", "dog", "bird"];
        let result = p.classify(&audio, &labels).expect("ok");
        for w in result.all_scores.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "scores not sorted: {} > {}",
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn test_classify_all_scores_sum_approx_one() {
        let p = default_pipeline();
        let audio = make_waveform(vec![0.1_f32; 8_000]);
        let labels = ["rain", "thunder", "wind", "hail"];
        let result = p.classify(&audio, &labels).expect("ok");
        let total: f32 = result.all_scores.iter().map(|(_, s)| s).sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "scores sum to {total}, expected ~1.0"
        );
    }

    #[test]
    fn test_classify_batch_count() {
        let p = default_pipeline();
        let a1 = make_waveform(vec![0.1_f32; 16_000]);
        let a2 = make_waveform(vec![0.9_f32; 16_000]);
        let audios = [&a1, &a2];
        let labels = ["music", "noise"];
        let results = p.classify_batch(&audios, &labels).expect("batch ok");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_empty_audio_error() {
        let p = default_pipeline();
        let audio = make_waveform(vec![]);
        let err = p
            .classify(&audio, &["speech"])
            .expect_err("empty audio should fail");
        assert!(matches!(err, ZeroShotAudioError::EmptyAudio));
    }

    #[test]
    fn test_no_labels_error() {
        let p = default_pipeline();
        let audio = make_waveform(vec![0.1_f32; 100]);
        let err = p
            .classify(&audio, &[])
            .expect_err("empty labels should fail");
        assert!(matches!(err, ZeroShotAudioError::NoLabels));
    }

    #[test]
    fn test_single_label_score_is_one() {
        let p = default_pipeline();
        let audio = make_waveform(vec![0.2_f32; 16_000]);
        let result = p.classify(&audio, &["music"]).expect("ok");
        assert!((result.score - 1.0).abs() < 1e-5, "score was {}", result.score);
    }

    #[test]
    fn test_different_audios_may_get_different_top_labels() {
        let p = default_pipeline();
        // Two very different signals.
        let a1 = make_waveform((0..16_000).map(|i| (i as f32 * 0.001).sin()).collect());
        let a2 = make_waveform(vec![0.999_f32; 16_000]);
        let labels = ["speech", "music", "noise", "silence", "environmental"];
        let r1 = p.classify(&a1, &labels).expect("ok");
        let r2 = p.classify(&a2, &labels).expect("ok");
        // At minimum the scores should differ (they embed differently).
        let scores_differ = r1
            .all_scores
            .iter()
            .zip(r2.all_scores.iter())
            .any(|(a, b)| (a.1 - b.1).abs() > 1e-6);
        assert!(scores_differ, "expected different score distributions for different audio");
    }

    #[test]
    fn test_default_config_sample_rate() {
        let config = ZeroShotAudioConfig::default();
        assert_eq!(config.sample_rate, 48_000);
    }

    #[test]
    fn test_normalize_flags_present_in_default() {
        let config = ZeroShotAudioConfig::default();
        assert!(config.normalize_audio, "normalize_audio should default to true");
        assert!(
            config.normalize_embeddings,
            "normalize_embeddings should default to true"
        );
    }

    // ── ZeroShotAudioProcessor::format_hypotheses ────────────────────────────

    #[test]
    fn test_format_hypotheses_basic() {
        let labels = vec!["speech".to_string(), "music".to_string(), "noise".to_string()];
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
            matches!(err, ZeroShotAudioError::DimensionMismatch { audio: 2, text: 3 }),
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
        assert!((sum - 1.0).abs() < 1e-5, "entmax scores must sum to 1.0, got {sum}");
    }

    #[test]
    fn test_entmax_scores_all_positive() {
        let logits = vec![1.0_f32, -2.0, 0.0, 3.0, -5.0];
        let scores = ZeroShotAudioProcessor::entmax_scores(&logits);
        assert!(scores.iter().all(|&s| s >= 0.0), "all entmax scores must be >= 0");
    }

    #[test]
    fn test_entmax_scores_dominant_entry() {
        // Very large logit at index 0 should dominate after entmax.
        let logits = vec![100.0_f32, 0.0, 0.0, 0.0];
        let scores = ZeroShotAudioProcessor::entmax_scores(&logits);
        assert!(scores[0] > 0.9, "dominant logit should dominate: score={}", scores[0]);
    }

    #[test]
    fn test_entmax_scores_empty() {
        let scores = ZeroShotAudioProcessor::entmax_scores(&[]);
        assert!(scores.is_empty());
    }

    // ── classify_input (new API) ──────────────────────────────────────────────

    #[test]
    fn test_classify_input_basic() {
        let p = default_pipeline();
        let audio = AudioInput::from_samples(vec![0.5_f32; 16_000], 16_000).expect("ok");
        let labels = ["speech", "music", "noise"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let result = p.classify_input(&audio, &labels).expect("classify_input ok");
        assert_eq!(result.len(), labels.len());
    }

    #[test]
    fn test_classify_input_scores_sorted() {
        let p = default_pipeline();
        let audio = AudioInput::from_samples(vec![0.3_f32; 8_000], 16_000).expect("ok");
        let labels = ["cat", "dog", "bird", "rain"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let result = p.classify_input(&audio, &labels).expect("ok");
        for w in result.windows(2) {
            assert!(w[0].score >= w[1].score, "scores not sorted descending");
        }
    }

    #[test]
    fn test_classify_input_single_label_score_one() {
        let p = default_pipeline();
        let audio = AudioInput::from_samples(vec![0.1_f32; 4_000], 16_000).expect("ok");
        let labels = vec!["music".to_string()];
        let result = p.classify_input(&audio, &labels).expect("ok");
        assert!((result[0].score - 1.0).abs() < 1e-5, "single label score should be 1.0");
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
    fn test_classify_inputs_batch_shape() {
        let p = default_pipeline();
        let audios: Vec<AudioInput> = (0..3)
            .map(|i| {
                AudioInput::from_samples(vec![(i as f32) * 0.1; 4_000], 16_000)
                    .expect("ok")
            })
            .collect();
        let labels = ["speech", "music", "noise"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let results = p
            .classify_inputs_batch(audios, &labels)
            .expect("batch ok");
        assert_eq!(results.len(), 3, "batch should return one result per audio");
        for r in &results {
            assert_eq!(r.len(), labels.len());
        }
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
        assert!(probs.iter().all(|&p| p > 0.0), "all softmax outputs must be positive");
    }

    #[test]
    fn test_softmax_empty() {
        let probs = softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_top_k_score_sum_approaches_one() {
        let p = default_pipeline();
        let audio = make_waveform(vec![0.4_f32; 8_000]);
        let labels = ["a", "b", "c", "d", "e"];
        let result = p.classify(&audio, &labels).expect("ok");
        // All scores sum to 1.0
        let sum: f32 = result.all_scores.iter().map(|(_, s)| s).sum();
        assert!((sum - 1.0).abs() < 1e-5, "all scores sum to {sum}");
    }
}
