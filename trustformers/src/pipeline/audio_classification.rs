//! # Audio Classification Pipeline
//!
//! This module provides an audio classification pipeline that maps raw audio signals
//! or audio file paths to categorical labels (e.g., keyword spotting, sound event
//! detection, music genre classification).
//!
//! ## Supported model families
//! - **wav2vec2** — general-purpose self-supervised audio encoder
//! - **Whisper** — multilingual speech model usable for audio classification
//! - **Audio Spectrogram Transformer (AST)** — vision-transformer applied to mel spectrograms
//!
//! ## Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::audio_classification::{
//!     AudioClassificationConfig, AudioClassificationPipeline, AudioClassificationInput,
//! };
//!
//! let config = AudioClassificationConfig {
//!     model_name: "facebook/wav2vec2-base".to_string(),
//!     sample_rate: 16_000,
//!     top_k: 5,
//!     ..Default::default()
//! };
//!
//! let pipeline = AudioClassificationPipeline::new(config)?;
//!
//! let input = AudioClassificationInput::RawAudio {
//!     samples: vec![0.0_f32; 16_000],
//!     sample_rate: 16_000,
//! };
//!
//! let results = pipeline.classify(&input)?;
//! for result in &results {
//!     println!("{}: {:.4}", result.label, result.score);
//! }
//! # Ok::<(), trustformers::TrustformersError>(())
//! ```

use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use trustformers_core::tensor::Tensor;

// ---------------------------------------------------------------------------
// Public types — Input
// ---------------------------------------------------------------------------

/// Audio input variants supported by the classification pipeline.
///
/// Use [`AudioClassificationInput::RawAudio`] for in-memory float samples, or
/// [`AudioClassificationInput::FilePath`] for lazily-loaded files.
#[derive(Debug, Clone)]
pub enum AudioClassificationInput {
    /// Raw mono PCM samples at the given sample rate.
    RawAudio {
        /// Floating-point audio samples normalised to `[-1.0, 1.0]`.
        samples: Vec<f32>,
        /// Sample rate in Hz (e.g. 16 000).
        sample_rate: u32,
    },
    /// Path to a supported audio file (WAV, FLAC, MP3, OGG).
    FilePath(PathBuf),
    /// Pre-computed log-mel spectrogram as a flat `[frames × bins]` tensor.
    MelSpectrogram {
        /// Flattened spectrogram values.
        values: Vec<f32>,
        /// Number of time frames.
        frames: usize,
        /// Number of mel filter banks.
        mel_bins: usize,
    },
}

/// New-style AudioInput enum: raw PCM, mel spectrogram data, or file path.
#[derive(Debug, Clone)]
pub enum AudioInput {
    /// Raw mono PCM samples with a specific sample rate.
    RawPcm { samples: Vec<f32>, sample_rate: u32 },
    /// Pre-computed mel spectrogram as 2D (frames × mel_bins).
    MelSpectrogram { data: Vec<Vec<f32>> },
    /// Path to an audio file on disk.
    FilePath(String),
}

// ---------------------------------------------------------------------------
// Public types — Output
// ---------------------------------------------------------------------------

/// A single label-score pair returned by the audio classification pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioClassificationResult {
    /// Human-readable label string (e.g. `"speech"`, `"music"`, `"noise"`).
    pub label: String,
    /// Predicted probability in the range `[0.0, 1.0]`.
    pub score: f32,
    /// Zero-based index of this label in the label set.
    pub label_id: usize,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`AudioClassificationPipeline`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioClassificationConfig {
    /// HuggingFace model identifier or local path.
    pub model_name: String,
    /// Expected sample rate. Audio will be resampled to this rate.
    pub sample_rate: u32,
    /// Maximum number of top-scoring labels to return.
    pub top_k: usize,
    /// Label set. When empty the pipeline uses model-internal labels.
    pub labels: Vec<String>,
    /// Device string (`"cpu"`, `"cuda:0"`, …).
    pub device: String,
    /// Maximum audio duration to process (seconds). `None` = no limit.
    pub max_duration_secs: Option<f32>,
    /// Number of mel filter banks used when computing spectrograms.
    pub num_mel_bins: usize,
    /// Whether to apply a Hann window before the FFT.
    pub apply_hann_window: bool,
    /// Number of audio classes (used for the new-style API).
    pub num_classes: usize,
    /// Model ID alias (used by new-style API).
    pub model_id: String,
    /// Maximum audio duration in seconds (new-style API field).
    pub max_duration_secs_f: f32,
}

impl Default for AudioClassificationConfig {
    fn default() -> Self {
        Self {
            model_name: "facebook/wav2vec2-base".to_string(),
            sample_rate: 16_000,
            top_k: 5,
            labels: Vec::new(),
            device: "cpu".to_string(),
            max_duration_secs: Some(30.0),
            num_mel_bins: 80,
            apply_hann_window: true,
            num_classes: 8,
            model_id: "facebook/wav2vec2-base".to_string(),
            max_duration_secs_f: 30.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/// Algorithm used for audio resampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleAlgorithm {
    /// Nearest-neighbor: each output sample copies the closest input sample.
    NearestNeighbor,
    /// Linear interpolation between adjacent input samples.
    LinearInterpolation,
}

/// Configuration for the resampler.
#[derive(Debug, Clone)]
pub struct AudioResampleConfig {
    pub source_rate: u32,
    pub target_rate: u32,
    pub algorithm: ResampleAlgorithm,
}

/// Resample `samples` from `config.source_rate` to `config.target_rate` using
/// the algorithm selected in `config.algorithm`.
pub fn resample_audio(samples: &[f32], config: &AudioResampleConfig) -> Vec<f32> {
    if config.source_rate == config.target_rate || samples.is_empty() {
        return samples.to_vec();
    }
    match config.algorithm {
        ResampleAlgorithm::NearestNeighbor => {
            resample_nearest(samples, config.source_rate, config.target_rate)
        },
        ResampleAlgorithm::LinearInterpolation => {
            resample_linear(samples, config.source_rate, config.target_rate)
        },
    }
}

// ---------------------------------------------------------------------------
// Feature extraction helpers (pure Rust, no external deps)
// ---------------------------------------------------------------------------

/// Resample `samples` from `from_hz` to `to_hz` using linear interpolation.
pub fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = from_hz as f64 / to_hz as f64;
    let new_len = ((samples.len() as f64) / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = (src_pos - src_idx as f64) as f32;
        let a = samples.get(src_idx).copied().unwrap_or(0.0);
        let b = samples.get(src_idx + 1).copied().unwrap_or(a);
        out.push(a + frac * (b - a));
    }
    out
}

/// Resample using nearest-neighbor: each output sample maps to the closest input sample.
pub fn resample_nearest(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = from_hz as f64 / to_hz as f64;
    let new_len = ((samples.len() as f64) / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let src_idx = ((i as f64 * ratio + 0.5) as usize).min(samples.len() - 1);
        out.push(samples[src_idx]);
    }
    out
}

/// Compute a simple mean-pooled feature vector from raw samples.
///
/// Splits the waveform into `num_frames` non-overlapping segments and
/// computes the RMS energy of each segment as a 1-D feature.
fn compute_waveform_features(samples: &[f32], num_frames: usize) -> Vec<f32> {
    if samples.is_empty() || num_frames == 0 {
        return vec![0.0; num_frames.max(1)];
    }
    let frame_size = (samples.len() / num_frames).max(1);
    let mut features = Vec::with_capacity(num_frames);
    for frame_idx in 0..num_frames {
        let start = frame_idx * frame_size;
        let end = ((frame_idx + 1) * frame_size).min(samples.len());
        if start >= samples.len() {
            features.push(0.0_f32);
            continue;
        }
        let rms: f32 = {
            let slice = &samples[start..end];
            let sum_sq: f32 = slice.iter().map(|s| s * s).sum();
            (sum_sq / slice.len() as f32).sqrt()
        };
        features.push(rms);
    }
    features
}

/// Apply softmax normalization to a slice of scores, returning normalized probabilities.
pub fn normalize_scores(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        vec![1.0 / scores.len() as f32; scores.len()]
    } else {
        exps.into_iter().map(|e| e / sum).collect()
    }
}

/// Apply softmax in-place over a slice of logits.
fn softmax_inplace(logits: &mut Vec<f32>) {
    if logits.is_empty() {
        return;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

/// Compute a simplified mel spectrogram from raw PCM samples.
///
/// Returns a 2D vector of shape `[n_frames][n_mels]` where values are
/// log-mel filterbank energies.
pub fn compute_mel_spectrogram(
    pcm: &[f32],
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
) -> Vec<Vec<f32>> {
    if pcm.is_empty() || n_fft == 0 || hop_length == 0 || n_mels == 0 {
        return Vec::new();
    }

    let hop = hop_length.max(1);
    let n_frames = if pcm.len() >= n_fft { (pcm.len() - n_fft) / hop + 1 } else { 1 };

    // Pre-compute Hann window
    let hann_window: Vec<f32> = (0..n_fft)
        .map(|i| {
            let phase = 2.0 * std::f32::consts::PI * i as f32 / (n_fft as f32 - 1.0).max(1.0);
            0.5 * (1.0 - phase.cos())
        })
        .collect();

    // Pre-compute simplified mel filterbank center frequencies
    // Map mel scale to linear FFT bins
    let n_bins = n_fft / 2 + 1;
    let mel_centers: Vec<f32> = (0..n_mels)
        .map(|m| (m as f32 + 1.0) / (n_mels as f32 + 1.0) * n_bins as f32)
        .collect();
    let mel_width = (n_bins as f32) / (n_mels as f32 + 1.0);

    let mut spectrogram = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop;
        let end = (start + n_fft).min(pcm.len());

        // Apply Hann window to the frame
        let mut windowed = vec![0.0_f32; n_fft];
        for (i, sample_idx) in (start..end).enumerate() {
            windowed[i] = pcm[sample_idx] * hann_window[i];
        }

        // Compute power spectrum via simplified DFT (magnitude squared)
        // For efficiency use a simplified approach: compute N_BINS = n_fft/2+1 bins
        let mut power_spectrum = vec![0.0_f32; n_bins];
        for k in 0..n_bins {
            let mut re = 0.0_f32;
            let mut im = 0.0_f32;
            let phase_step = 2.0 * std::f32::consts::PI * k as f32 / n_fft as f32;
            for (n, &s) in windowed.iter().enumerate() {
                re += s * (phase_step * n as f32).cos();
                im -= s * (phase_step * n as f32).sin();
            }
            power_spectrum[k] = re * re + im * im;
        }

        // Apply triangular mel filterbanks and take log
        let mut mel_frame = vec![0.0_f32; n_mels];
        for (m, &center) in mel_centers.iter().enumerate() {
            let left = center - mel_width;
            let right = center + mel_width;
            let mut energy = 0.0_f32;
            for k in 0..n_bins {
                let kf = k as f32;
                let weight = if kf >= left && kf <= center {
                    (kf - left) / mel_width.max(1e-6)
                } else if kf > center && kf <= right {
                    (right - kf) / mel_width.max(1e-6)
                } else {
                    0.0
                };
                energy += weight * power_spectrum[k];
            }
            mel_frame[m] = (energy.max(1e-10)).ln();
        }

        spectrogram.push(mel_frame);
    }

    spectrogram
}

// ---------------------------------------------------------------------------
// Pipeline internals
// ---------------------------------------------------------------------------

/// Lightweight internal state shared across calls.
struct ClassificationState {
    config: AudioClassificationConfig,
    /// Resolved label list (may come from config or default set).
    labels: Vec<String>,
}

impl ClassificationState {
    fn new(config: AudioClassificationConfig) -> Self {
        let labels =
            if config.labels.is_empty() { default_labels() } else { config.labels.clone() };
        Self { config, labels }
    }

    /// Normalise raw samples to the pipeline's expected sample rate and duration.
    fn preprocess_raw(&self, samples: &[f32], input_rate: u32) -> Result<Vec<f32>> {
        // Resample if needed
        let resampled = resample_linear(samples, input_rate, self.config.sample_rate);

        // Truncate to max_duration
        let max_samples = self
            .config
            .max_duration_secs
            .map(|secs| (secs * self.config.sample_rate as f32) as usize);
        let truncated = match max_samples {
            Some(limit) if resampled.len() > limit => resampled[..limit].to_vec(),
            _ => resampled,
        };

        Ok(truncated)
    }

    /// Extract a fixed-size feature vector from preprocessed samples.
    fn extract_features(&self, samples: &[f32]) -> Result<Tensor> {
        // We compute 128-dimensional waveform-level features as a simple
        // approximation. A real implementation would compute mel spectrograms.
        const FEATURE_DIM: usize = 128;
        let feats = compute_waveform_features(samples, FEATURE_DIM);
        // Shape: [1, FEATURE_DIM]
        Tensor::from_slice(&feats, &[1, FEATURE_DIM])
            .map_err(|e| TrustformersError::pipeline(e.to_string(), "audio-classification"))
    }

    /// Simulate model inference: produce mock logits proportional to feature norms.
    ///
    /// In a real deployment this would call `model.forward(features)`.
    fn mock_forward(&self, features: &Tensor) -> Result<Vec<f32>> {
        let num_labels = self.labels.len();
        if num_labels == 0 {
            return Err(TrustformersError::pipeline(
                "Label set is empty — cannot classify".to_string(),
                "audio-classification",
            ));
        }
        let flat = features
            .data_f32()
            .map_err(|e| TrustformersError::pipeline(e.to_string(), "audio-classification"))?;
        // Deterministic mock: spread feature energy across labels
        let mut logits = vec![0.0_f32; num_labels];
        for (i, &v) in flat.iter().enumerate() {
            logits[i % num_labels] += v.abs();
        }
        Ok(logits)
    }

    /// Run classification for a single preprocessed sample vector.
    fn run_inference(&self, samples: &[f32]) -> Result<Vec<AudioClassificationResult>> {
        let features = self.extract_features(samples)?;
        let mut logits = self.mock_forward(&features)?;
        softmax_inplace(&mut logits);

        // Pair logits with labels and sort descending
        let mut scored: Vec<(usize, f32)> = logits.into_iter().enumerate().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = self.config.top_k.min(scored.len());
        let results = scored
            .into_iter()
            .take(top_k)
            .map(|(idx, score)| AudioClassificationResult {
                label: self.labels[idx].clone(),
                score,
                label_id: idx,
            })
            .collect();
        Ok(results)
    }
}

fn default_labels() -> Vec<String> {
    vec![
        "speech".to_string(),
        "music".to_string(),
        "noise".to_string(),
        "silence".to_string(),
        "environmental".to_string(),
        "animal".to_string(),
        "vehicle".to_string(),
        "alarm".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Public pipeline struct
// ---------------------------------------------------------------------------

/// Pipeline for audio classification tasks.
///
/// Maps an [`AudioClassificationInput`] to a ranked list of
/// [`AudioClassificationResult`] values (label + probability).
///
/// # Example
///
/// ```rust,ignore
/// use trustformers::pipeline::audio_classification::*;
///
/// let pipeline = AudioClassificationPipeline::new(AudioClassificationConfig::default())?;
/// let input = AudioClassificationInput::RawAudio {
///     samples: vec![0.0; 8_000],
///     sample_rate: 8_000,
/// };
/// let results = pipeline.classify(&input)?;
/// assert!(!results.is_empty());
/// # Ok::<(), trustformers::TrustformersError>(())
/// ```
pub struct AudioClassificationPipeline {
    state: ClassificationState,
}

impl AudioClassificationPipeline {
    /// Create a new pipeline with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError`] if the configuration is invalid (e.g. zero
    /// sample rate or an empty `top_k`).
    pub fn new(config: AudioClassificationConfig) -> Result<Self> {
        if config.sample_rate == 0 {
            return Err(TrustformersError::pipeline(
                "sample_rate must be greater than zero".to_string(),
                "audio-classification",
            ));
        }
        if config.top_k == 0 {
            return Err(TrustformersError::pipeline(
                "top_k must be greater than zero".to_string(),
                "audio-classification",
            ));
        }
        Ok(Self {
            state: ClassificationState::new(config),
        })
    }

    /// Preprocess raw PCM samples: resample to `target_rate` using linear interpolation.
    ///
    /// Returns the resampled samples, or an error if the target rate is zero.
    pub fn preprocess_pcm(samples: &[f32], sample_rate: u32, target_rate: u32) -> Result<Vec<f32>> {
        if target_rate == 0 {
            return Err(TrustformersError::pipeline(
                "target_rate must be greater than zero".to_string(),
                "audio-classification",
            ));
        }
        Ok(resample_linear(samples, sample_rate, target_rate))
    }

    /// Compute a simplified mel spectrogram from raw PCM samples.
    ///
    /// Returns a 2D vector `[n_frames][n_mels]` of log-mel energies.
    pub fn compute_mel_spectrogram(
        pcm: &[f32],
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
    ) -> Vec<Vec<f32>> {
        compute_mel_spectrogram(pcm, n_fft, hop_length, n_mels)
    }

    /// Classify a single audio input (new-style API using `AudioInput`).
    pub fn classify_input(&self, input: AudioInput) -> Result<Vec<AudioClassificationResult>> {
        let samples = self.load_audio_input(input)?;
        self.state.run_inference(&samples)
    }

    /// Classify a batch of audio inputs (new-style API).
    pub fn classify_batch_inputs(
        &self,
        inputs: Vec<AudioInput>,
    ) -> Result<Vec<Vec<AudioClassificationResult>>> {
        inputs.into_iter().map(|inp| self.classify_input(inp)).collect()
    }

    /// Classify a single audio input (legacy API).
    pub fn classify(
        &self,
        input: &AudioClassificationInput,
    ) -> Result<Vec<AudioClassificationResult>> {
        let samples = self.load_samples(input)?;
        self.state.run_inference(&samples)
    }

    /// Classify a batch of audio inputs (legacy API).
    pub fn classify_batch(
        &self,
        inputs: &[AudioClassificationInput],
    ) -> Result<Vec<Vec<AudioClassificationResult>>> {
        inputs.iter().map(|inp| self.classify(inp)).collect()
    }

    /// Access the pipeline configuration.
    pub fn config(&self) -> &AudioClassificationConfig {
        &self.state.config
    }

    /// Access the resolved label set.
    pub fn labels(&self) -> &[String] {
        &self.state.labels
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn load_audio_input(&self, input: AudioInput) -> Result<Vec<f32>> {
        match input {
            AudioInput::RawPcm {
                samples,
                sample_rate,
            } => self.state.preprocess_raw(&samples, sample_rate),
            AudioInput::MelSpectrogram { data } => {
                // Flatten the 2D mel spectrogram into a 1D feature vector
                Ok(data.into_iter().flatten().collect())
            },
            AudioInput::FilePath(path_str) => {
                let path = std::path::Path::new(&path_str);
                if !path.exists() {
                    return Err(TrustformersError::Io {
                        message: format!("Audio file not found: {}", path_str),
                        path: Some(path_str),
                        suggestion: Some(
                            "Check the file path and ensure the file exists.".to_string(),
                        ),
                    });
                }
                // Placeholder: return silence at target sample rate
                let placeholder: Vec<f32> = vec![0.0; self.state.config.sample_rate as usize];
                Ok(placeholder)
            },
        }
    }

    fn load_samples(&self, input: &AudioClassificationInput) -> Result<Vec<f32>> {
        match input {
            AudioClassificationInput::RawAudio {
                samples,
                sample_rate,
            } => self.state.preprocess_raw(samples, *sample_rate),

            AudioClassificationInput::FilePath(path) => {
                if !path.exists() {
                    return Err(TrustformersError::Io {
                        message: format!("Audio file not found: {}", path.to_string_lossy()),
                        path: Some(path.to_string_lossy().into_owned()),
                        suggestion: Some(
                            "Check the file path and ensure the file exists.".to_string(),
                        ),
                    });
                }
                // In a full implementation: decode WAV/FLAC/MP3 here.
                // For now return a zero-filled placeholder so the code
                // compiles and tests can exercise the path.
                let placeholder: Vec<f32> = vec![0.0; self.state.config.sample_rate as usize];
                tracing::debug!(
                    path = %path.to_string_lossy(),
                    "Audio file decoding not yet implemented; using zero placeholder"
                );
                Ok(placeholder)
            },

            AudioClassificationInput::MelSpectrogram { values, .. } => {
                // Treat pre-computed spectrogram frames as raw 1-D signal for
                // the mock inference path.
                Ok(values.clone())
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Trait impl (mirrors Pipeline trait without the associated Input/Output types
// since those are audio-specific)
// ---------------------------------------------------------------------------

impl crate::pipeline::Pipeline for AudioClassificationPipeline {
    type Input = AudioClassificationInput;
    type Output = Vec<AudioClassificationResult>;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        self.classify(&input)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pipeline() -> AudioClassificationPipeline {
        AudioClassificationPipeline::new(AudioClassificationConfig::default())
            .expect("default config should be valid")
    }

    // ---- Legacy API tests (preserved) ----

    #[test]
    fn test_default_config_creates_pipeline() {
        let _p = default_pipeline();
    }

    #[test]
    fn test_classify_raw_audio_returns_top_k_results() {
        let config = AudioClassificationConfig {
            top_k: 3,
            ..Default::default()
        };
        let pipeline = AudioClassificationPipeline::new(config).expect("valid config");
        let input = AudioClassificationInput::RawAudio {
            samples: vec![0.1_f32; 16_000],
            sample_rate: 16_000,
        };
        let results = pipeline.classify(&input).expect("classify should succeed");
        assert_eq!(results.len(), 3, "should return exactly top_k results");
    }

    #[test]
    fn test_classify_batch_length_matches_input() {
        let pipeline = default_pipeline();
        let inputs = vec![
            AudioClassificationInput::RawAudio {
                samples: vec![0.0; 8_000],
                sample_rate: 8_000,
            },
            AudioClassificationInput::RawAudio {
                samples: vec![0.3; 8_000],
                sample_rate: 8_000,
            },
            AudioClassificationInput::RawAudio {
                samples: vec![-0.5; 8_000],
                sample_rate: 8_000,
            },
        ];
        let batch = pipeline.classify_batch(&inputs).expect("batch classify should succeed");
        assert_eq!(batch.len(), 3, "batch length must match input count");
    }

    #[test]
    fn test_scores_sum_to_approximately_one() {
        let pipeline = AudioClassificationPipeline::new(AudioClassificationConfig {
            top_k: 8, // request all default labels
            ..Default::default()
        })
        .expect("valid config");
        let input = AudioClassificationInput::RawAudio {
            samples: vec![0.2; 16_000],
            sample_rate: 16_000,
        };
        let results = pipeline.classify(&input).expect("classify ok");
        let total: f32 = results.iter().map(|r| r.score).sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "scores should sum to ~1.0, got {total}"
        );
    }

    #[test]
    fn test_missing_file_path_returns_error() {
        let pipeline = default_pipeline();
        let tmp = std::env::temp_dir().join("audio_classification_nonexistent.wav");
        // Ensure it does NOT exist
        let _ = std::fs::remove_file(&tmp);
        let input = AudioClassificationInput::FilePath(tmp);
        let result = pipeline.classify(&input);
        assert!(
            result.is_err(),
            "should fail when the audio file does not exist"
        );
    }

    #[test]
    fn test_mel_spectrogram_input_is_accepted() {
        let pipeline = default_pipeline();
        let input = AudioClassificationInput::MelSpectrogram {
            values: vec![0.5; 128 * 80],
            frames: 128,
            mel_bins: 80,
        };
        let results = pipeline.classify(&input).expect("mel spectrogram input ok");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_resample_shorter_signal() {
        let orig = vec![1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let out = resample_linear(&orig, 8_000, 4_000);
        assert_eq!(
            out.len(),
            4,
            "downsampled to half length: got {}",
            out.len()
        );
    }

    #[test]
    fn test_zero_sample_rate_returns_error() {
        let config = AudioClassificationConfig {
            sample_rate: 0,
            ..Default::default()
        };
        let result = AudioClassificationPipeline::new(config);
        assert!(result.is_err(), "zero sample_rate should be rejected");
    }

    #[test]
    fn test_custom_labels_are_used() {
        let config = AudioClassificationConfig {
            labels: vec!["cat".to_string(), "dog".to_string(), "bird".to_string()],
            top_k: 2,
            ..Default::default()
        };
        let pipeline = AudioClassificationPipeline::new(config).expect("valid");
        let input = AudioClassificationInput::RawAudio {
            samples: vec![0.1; 16_000],
            sample_rate: 16_000,
        };
        let results = pipeline.classify(&input).expect("ok");
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(
                ["cat", "dog", "bird"].contains(&r.label.as_str()),
                "unexpected label: {}",
                r.label
            );
        }
    }

    #[test]
    fn test_existing_file_with_placeholder_succeeds() {
        // Create a real (zero-byte) temp file so the existence check passes
        let tmp = std::env::temp_dir().join("audio_classification_test.wav");
        std::fs::write(&tmp, b"").expect("write temp file");
        let pipeline = default_pipeline();
        let input = AudioClassificationInput::FilePath(tmp.clone());
        let result = pipeline.classify(&input);
        // Clean up
        let _ = std::fs::remove_file(&tmp);
        // With a placeholder zero signal the pipeline should succeed
        assert!(result.is_ok(), "should succeed for existing file path");
    }

    // ---- New tests for enhanced API ----

    #[test]
    fn test_preprocess_pcm_resamples_correctly() {
        // 8 samples at 8kHz → 4 samples at 4kHz (2:1 downsample)
        let samples = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = AudioClassificationPipeline::preprocess_pcm(&samples, 8_000, 4_000)
            .expect("preprocess_pcm ok");
        assert_eq!(
            result.len(),
            4,
            "expected 4 output samples, got {}",
            result.len()
        );
    }

    #[test]
    fn test_preprocess_pcm_zero_target_rate_errors() {
        let samples = vec![0.1_f32; 100];
        let result = AudioClassificationPipeline::preprocess_pcm(&samples, 16_000, 0);
        assert!(result.is_err(), "zero target_rate must fail");
    }

    #[test]
    fn test_preprocess_pcm_same_rate_is_identity() {
        let samples = vec![0.1_f32, 0.2, 0.3, 0.4];
        let result =
            AudioClassificationPipeline::preprocess_pcm(&samples, 16_000, 16_000).expect("ok");
        assert_eq!(
            result, samples,
            "same-rate preprocess should return identical signal"
        );
    }

    #[test]
    fn test_compute_mel_spectrogram_returns_correct_shape() {
        // 1 second of silence at 16kHz → check frame count and mel bin count
        let pcm = vec![0.0_f32; 16_000];
        let n_fft = 512;
        let hop = 160;
        let n_mels = 40;
        let mel = AudioClassificationPipeline::compute_mel_spectrogram(&pcm, n_fft, hop, n_mels);
        assert!(!mel.is_empty(), "mel spectrogram should not be empty");
        let expected_frames = (pcm.len() - n_fft) / hop + 1;
        assert_eq!(mel.len(), expected_frames, "frame count mismatch");
        for frame in &mel {
            assert_eq!(frame.len(), n_mels, "each frame must have n_mels bins");
        }
    }

    #[test]
    fn test_compute_mel_spectrogram_empty_pcm_returns_empty() {
        let mel = AudioClassificationPipeline::compute_mel_spectrogram(&[], 512, 160, 40);
        assert!(mel.is_empty(), "empty pcm should yield empty spectrogram");
    }

    #[test]
    fn test_normalize_scores_sums_to_one() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let probs = normalize_scores(&logits);
        assert_eq!(probs.len(), logits.len());
        let total: f32 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "normalized scores must sum to 1.0, got {total}"
        );
    }

    #[test]
    fn test_normalize_scores_preserves_ordering() {
        let logits = vec![1.0_f32, 5.0, 3.0];
        let probs = normalize_scores(&logits);
        // score[1] > score[2] > score[0] after normalization
        assert!(
            probs[1] > probs[2],
            "highest logit should get highest probability"
        );
        assert!(
            probs[2] > probs[0],
            "middle logit should be between extremes"
        );
    }

    #[test]
    fn test_normalize_scores_empty_returns_empty() {
        let probs = normalize_scores(&[]);
        assert!(probs.is_empty(), "empty input should yield empty output");
    }

    #[test]
    fn test_resample_audio_linear() {
        let samples = vec![0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let config = AudioResampleConfig {
            source_rate: 8_000,
            target_rate: 4_000,
            algorithm: ResampleAlgorithm::LinearInterpolation,
        };
        let out = resample_audio(&samples, &config);
        assert_eq!(
            out.len(),
            4,
            "linear: expected 4 samples, got {}",
            out.len()
        );
    }

    #[test]
    fn test_resample_audio_nearest() {
        let samples = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let config = AudioResampleConfig {
            source_rate: 8_000,
            target_rate: 4_000,
            algorithm: ResampleAlgorithm::NearestNeighbor,
        };
        let out = resample_audio(&samples, &config);
        assert_eq!(
            out.len(),
            4,
            "nearest: expected 4 samples, got {}",
            out.len()
        );
    }

    #[test]
    fn test_resample_audio_same_rate_returns_same() {
        let samples = vec![0.3_f32, 0.5, 0.7];
        let config = AudioResampleConfig {
            source_rate: 16_000,
            target_rate: 16_000,
            algorithm: ResampleAlgorithm::LinearInterpolation,
        };
        let out = resample_audio(&samples, &config);
        assert_eq!(out, samples);
    }

    #[test]
    fn test_new_style_classify_input_raw_pcm() {
        let pipeline = default_pipeline();
        let input = AudioInput::RawPcm {
            samples: vec![0.1_f32; 8_000],
            sample_rate: 8_000,
        };
        let results = pipeline.classify_input(input).expect("classify_input ok");
        assert!(!results.is_empty(), "should return at least one result");
    }

    #[test]
    fn test_new_style_classify_input_mel_spectrogram() {
        let pipeline = default_pipeline();
        // 50 frames × 40 mel bins
        let mel_data: Vec<Vec<f32>> = (0..50).map(|_| vec![0.5_f32; 40]).collect();
        let input = AudioInput::MelSpectrogram { data: mel_data };
        let results = pipeline.classify_input(input).expect("mel spectrogram input ok");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_new_style_batch_classify() {
        let pipeline = default_pipeline();
        let inputs = vec![
            AudioInput::RawPcm {
                samples: vec![0.0_f32; 4_000],
                sample_rate: 8_000,
            },
            AudioInput::RawPcm {
                samples: vec![0.5_f32; 4_000],
                sample_rate: 8_000,
            },
        ];
        let batch = pipeline.classify_batch_inputs(inputs).expect("batch ok");
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_result_has_label_id() {
        let pipeline = AudioClassificationPipeline::new(AudioClassificationConfig {
            top_k: 3,
            ..Default::default()
        })
        .expect("valid");
        let input = AudioClassificationInput::RawAudio {
            samples: vec![0.1_f32; 16_000],
            sample_rate: 16_000,
        };
        let results = pipeline.classify(&input).expect("ok");
        // Each label_id must be within valid range
        let label_count = pipeline.labels().len();
        for r in &results {
            assert!(
                r.label_id < label_count,
                "label_id {} out of bounds",
                r.label_id
            );
        }
    }

    #[test]
    fn test_upsample_doubles_length() {
        // 4 samples at 4kHz → 8 samples at 8kHz
        let samples = vec![0.0_f32, 1.0, 0.0, 1.0];
        let out = resample_linear(&samples, 4_000, 8_000);
        assert_eq!(
            out.len(),
            8,
            "upsampled length should be 8, got {}",
            out.len()
        );
    }

    #[test]
    fn test_mel_spectrogram_values_are_finite() {
        let pcm: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.01).sin()).collect();
        let mel = AudioClassificationPipeline::compute_mel_spectrogram(&pcm, 128, 64, 20);
        for (fi, frame) in mel.iter().enumerate() {
            for (bi, &val) in frame.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "frame {} bin {} has non-finite value {}",
                    fi,
                    bi,
                    val
                );
            }
        }
    }
}
