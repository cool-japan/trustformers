//! # Audio Classification Pipeline
//!
//! Maps raw audio signals or audio files to categorical labels (keyword
//! spotting, sound-event detection, music genre classification).
//!
//! ## What is real here
//!
//! * **Decoding** — uncompressed RIFF/WAVE containers (PCM 8/16/24/32-bit and
//!   IEEE float 32/64-bit) are decoded for real; compressed containers report a
//!   structured error instead of being replaced by silence.
//! * **Resampling** — nearest-neighbour and linear interpolation.
//! * **Feature extraction** — Hann-windowed STFT via the pure-Rust `oxifft`
//!   crate, a Slaney-scale triangular mel filterbank, and log compression
//!   (see [`crate::pipeline::media::audio_dsp`]).
//! * **Post-processing** — softmax and top-k ranking.
//!
//! ## Model support
//!
//! No audio classification backbone (wav2vec2, AST, …) has a real,
//! weight-loadable implementation in `trustformers-models` yet. Consequently
//! [`AudioClassificationPipeline::classify`] returns a structured
//! [`TrustformersError::FeatureUnavailable`] rather than inventing labels and
//! confidence scores. The preprocessing above is fully usable on its own —
//! call [`AudioClassificationPipeline::extract_log_mel`] and run your own model
//! on the result.
//!
//! ## Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::audio_classification::{
//!     AudioClassificationConfig, AudioClassificationPipeline,
//! };
//!
//! let pipeline = AudioClassificationPipeline::new(AudioClassificationConfig::default())?;
//! // Real feature extraction:
//! let mel = pipeline.extract_log_mel(&samples, 16_000)?;
//! // Classification reports that no backbone is available:
//! assert!(pipeline.classify(&input).is_err());
//! # Ok::<(), trustformers::TrustformersError>(())
//! ```

use crate::error::{Result, TrustformersError};
use crate::pipeline::media::audio_dsp;
use crate::pipeline::media::unsupported_model;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use trustformers_core::tensor::Tensor;

/// Architectures with a real, usable audio-classification backbone.
///
/// Deliberately empty: none exists yet, and the pipeline says so instead of
/// pretending otherwise.
const SUPPORTED_ARCHITECTURES: &[&str] = &[];

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
    /// Path to an audio file. Only uncompressed RIFF/WAVE can be decoded
    /// without an external codec; anything else reports an error.
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
// Feature extraction helpers (pure Rust, no C/C++ deps)
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

/// Compute a log-mel spectrogram from raw PCM samples.
///
/// Uses a real Hann-windowed STFT (`oxifft`) followed by a Slaney-scale
/// triangular mel filterbank and a natural-log compression. Frames are *not*
/// centred: frame `t` starts at sample `t · hop_length`, so the frame count is
/// `(len - n_fft) / hop_length + 1` (or 1 when the signal is shorter than one
/// window).
///
/// Returns a 2D vector of shape `[n_frames][n_mels]`, or an empty vector for
/// degenerate parameters.
///
/// The `sample_rate` used for the mel scale is fixed at 16 kHz, matching the
/// pipeline default; use [`AudioClassificationPipeline::extract_log_mel`] for a
/// rate-aware version.
pub fn compute_mel_spectrogram(
    pcm: &[f32],
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
) -> Vec<Vec<f32>> {
    compute_mel_spectrogram_at(pcm, 16_000, n_fft, hop_length, n_mels).unwrap_or_default()
}

/// Rate-aware log-mel spectrogram; see [`compute_mel_spectrogram`].
///
/// # Errors
///
/// Returns an error when the STFT or filterbank parameters are invalid.
pub fn compute_mel_spectrogram_at(
    pcm: &[f32],
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
) -> Result<Vec<Vec<f32>>> {
    if pcm.is_empty() || n_fft == 0 || hop_length == 0 || n_mels == 0 || sample_rate == 0 {
        return Ok(Vec::new());
    }

    // Match the historical (uncentred) framing so callers keep their frame counts.
    let padded: Vec<f32> = if pcm.len() >= n_fft {
        pcm.to_vec()
    } else {
        let mut v = pcm.to_vec();
        v.resize(n_fft, 0.0);
        v
    };

    let power = audio_dsp::stft_power(&padded, n_fft, hop_length, false)?;
    if power.is_empty() {
        return Ok(Vec::new());
    }
    let bank = audio_dsp::mel_filterbank(
        sample_rate,
        n_fft,
        n_mels,
        0.0,
        f64::from(sample_rate) / 2.0,
    )?;
    let mel = audio_dsp::apply_mel_filterbank(&power, &bank)?;
    Ok(mel
        .into_iter()
        .map(|frame| frame.into_iter().map(|v| v.max(1e-10).ln()).collect())
        .collect())
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
        if input_rate == 0 {
            return Err(TrustformersError::pipeline(
                "input sample rate must be greater than zero".to_string(),
                "audio-classification",
            ));
        }
        let resampled = resample_linear(samples, input_rate, self.config.sample_rate);

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

    /// Compute real log-mel features for preprocessed samples.
    fn extract_log_mel(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        if samples.is_empty() {
            return Err(TrustformersError::pipeline(
                "cannot extract features from an empty audio buffer".to_string(),
                "audio-classification",
            ));
        }
        let mel = audio_dsp::MelConfig {
            n_fft: 400,
            hop_length: 160,
            n_mels: self.config.num_mel_bins,
        };
        audio_dsp::log_mel_spectrogram(samples, self.config.sample_rate, mel)
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
/// Performs real decoding, resampling and log-mel feature extraction. Because
/// no audio classification backbone is implemented, the classification methods
/// return a structured [`TrustformersError::FeatureUnavailable`] instead of a
/// fabricated label ranking. See the module documentation.
pub struct AudioClassificationPipeline {
    state: ClassificationState,
}

impl AudioClassificationPipeline {
    /// Create a new pipeline with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError`] if the configuration is invalid (e.g. zero
    /// sample rate, zero `top_k`, or zero mel bins).
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
        if config.num_mel_bins == 0 {
            return Err(TrustformersError::pipeline(
                "num_mel_bins must be greater than zero".to_string(),
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

    /// Compute a log-mel spectrogram from raw PCM samples.
    pub fn compute_mel_spectrogram(
        pcm: &[f32],
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
    ) -> Vec<Vec<f32>> {
        compute_mel_spectrogram(pcm, n_fft, hop_length, n_mels)
    }

    /// Resample, truncate and compute real log-mel features for `samples`.
    ///
    /// This is the pipeline's genuine, reusable front-end: it works regardless
    /// of whether a classification backbone is available.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty buffer, a zero input rate, or audio too
    /// short for a single STFT frame.
    pub fn extract_log_mel(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<Vec<f32>>> {
        let prepared = self.state.preprocess_raw(samples, sample_rate)?;
        self.state.extract_log_mel(&prepared)
    }

    /// Decode an audio file into mono PCM at the pipeline's sample rate.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::Io`] when the file is missing and
    /// [`TrustformersError::FeatureUnavailable`] for containers that need an
    /// external codec. It never substitutes silence.
    pub fn decode_audio_file(&self, path: &Path) -> Result<Vec<f32>> {
        if !path.exists() {
            return Err(TrustformersError::Io {
                message: format!("Audio file not found: {}", path.to_string_lossy()),
                path: Some(path.to_string_lossy().into_owned()),
                suggestion: Some("Check the file path and ensure the file exists.".to_string()),
            });
        }
        let bytes = std::fs::read(path).map_err(|e| TrustformersError::Io {
            message: format!("failed to read audio file: {e}"),
            path: Some(path.to_string_lossy().into_owned()),
            suggestion: Some("Check file permissions.".to_string()),
        })?;
        if !audio_dsp::is_wav(&bytes) {
            return Err(TrustformersError::feature_unavailable(
                format!(
                    "cannot decode `{}`: only uncompressed RIFF/WAVE is decodable without an \
                     external codec",
                    path.to_string_lossy()
                ),
                "audio-codec",
            ));
        }
        let audio = audio_dsp::decode_wav(&bytes)?;
        audio_dsp::resample_linear(
            &audio.samples,
            audio.sample_rate,
            self.state.config.sample_rate,
        )
    }

    /// Rank `logits` against the pipeline's label set and return the top-k.
    ///
    /// Real post-processing, exposed so callers that run their own model can
    /// reuse it.
    ///
    /// # Errors
    ///
    /// Returns an error when the label set is empty or the logit count does not
    /// match it.
    pub fn rank_logits(&self, logits: &[f32]) -> Result<Vec<AudioClassificationResult>> {
        let labels = &self.state.labels;
        if labels.is_empty() {
            return Err(TrustformersError::pipeline(
                "Label set is empty — cannot classify".to_string(),
                "audio-classification",
            ));
        }
        if logits.len() != labels.len() {
            return Err(TrustformersError::pipeline(
                format!(
                    "model produced {} logits but the label set has {} entries",
                    logits.len(),
                    labels.len()
                ),
                "audio-classification",
            ));
        }
        let probabilities = normalize_scores(logits);
        let mut scored: Vec<(usize, f32)> = probabilities.into_iter().enumerate().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k = self.state.config.top_k.min(scored.len());
        Ok(scored
            .into_iter()
            .take(top_k)
            .map(|(idx, score)| AudioClassificationResult {
                label: labels[idx].clone(),
                score,
                label_id: idx,
            })
            .collect())
    }

    /// Classify a single audio input (new-style API using `AudioInput`).
    ///
    /// # Errors
    ///
    /// Always returns [`TrustformersError::FeatureUnavailable`] once the input
    /// has been validated and preprocessed: no audio classification backbone is
    /// implemented. Preprocessing errors surface first.
    pub fn classify_input(&self, input: AudioInput) -> Result<Vec<AudioClassificationResult>> {
        let features = self.features_for_new_style(input)?;
        self.run_inference(&features)
    }

    /// Classify a batch of audio inputs (new-style API).
    pub fn classify_batch_inputs(
        &self,
        inputs: Vec<AudioInput>,
    ) -> Result<Vec<Vec<AudioClassificationResult>>> {
        inputs.into_iter().map(|inp| self.classify_input(inp)).collect()
    }

    /// Classify a single audio input (legacy API).
    ///
    /// # Errors
    ///
    /// See [`Self::classify_input`].
    pub fn classify(
        &self,
        input: &AudioClassificationInput,
    ) -> Result<Vec<AudioClassificationResult>> {
        let features = self.features_for(input)?;
        self.run_inference(&features)
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

    /// Convert an input into log-mel features, doing all the real work.
    pub fn features_for(&self, input: &AudioClassificationInput) -> Result<Vec<Vec<f32>>> {
        match input {
            AudioClassificationInput::RawAudio {
                samples,
                sample_rate,
            } => self.extract_log_mel(samples, *sample_rate),

            AudioClassificationInput::FilePath(path) => {
                let pcm = self.decode_audio_file(path)?;
                self.state.extract_log_mel(&pcm)
            },

            AudioClassificationInput::MelSpectrogram {
                values,
                frames,
                mel_bins,
            } => reshape_mel(values, *frames, *mel_bins),
        }
    }

    fn features_for_new_style(&self, input: AudioInput) -> Result<Vec<Vec<f32>>> {
        match input {
            AudioInput::RawPcm {
                samples,
                sample_rate,
            } => self.extract_log_mel(&samples, sample_rate),
            AudioInput::MelSpectrogram { data } => {
                if data.is_empty() {
                    return Err(TrustformersError::pipeline(
                        "mel spectrogram input is empty".to_string(),
                        "audio-classification",
                    ));
                }
                Ok(data)
            },
            AudioInput::FilePath(path_str) => {
                let pcm = self.decode_audio_file(Path::new(&path_str))?;
                self.state.extract_log_mel(&pcm)
            },
        }
    }

    /// Run the (currently unavailable) classification backbone on real features.
    fn run_inference(&self, features: &[Vec<f32>]) -> Result<Vec<AudioClassificationResult>> {
        if features.is_empty() {
            return Err(TrustformersError::pipeline(
                "no feature frames were produced".to_string(),
                "audio-classification",
            ));
        }
        if self.state.labels.is_empty() {
            return Err(TrustformersError::pipeline(
                "Label set is empty — cannot classify".to_string(),
                "audio-classification",
            ));
        }
        Err(unsupported_model(
            "audio-classification",
            &self.state.config.model_name,
            SUPPORTED_ARCHITECTURES,
        ))
    }
}

/// Reshape a flat spectrogram buffer into `[frames][mel_bins]`.
fn reshape_mel(values: &[f32], frames: usize, mel_bins: usize) -> Result<Vec<Vec<f32>>> {
    if frames == 0 || mel_bins == 0 {
        return Err(TrustformersError::pipeline(
            "mel spectrogram dimensions must be non-zero".to_string(),
            "audio-classification",
        ));
    }
    if values.len() != frames * mel_bins {
        return Err(TrustformersError::pipeline(
            format!(
                "mel spectrogram buffer has {} values but {frames}x{mel_bins} = {} were expected",
                values.len(),
                frames * mel_bins
            ),
            "audio-classification",
        ));
    }
    Ok(values.chunks_exact(mel_bins).map(<[f32]>::to_vec).collect())
}

/// Build a `[1, mel_bins, frames]` tensor from log-mel frames.
///
/// This is the layout audio encoders (Whisper, AST) expect; exposed so callers
/// running their own model can reuse the pipeline's front-end.
///
/// # Errors
///
/// Returns an error for empty or ragged input.
pub fn mel_frames_to_tensor(frames: &[Vec<f32>]) -> Result<Tensor> {
    if frames.is_empty() {
        return Err(TrustformersError::pipeline(
            "no mel frames to convert".to_string(),
            "audio-classification",
        ));
    }
    let n_mels = frames[0].len();
    if n_mels == 0 || frames.iter().any(|f| f.len() != n_mels) {
        return Err(TrustformersError::pipeline(
            "mel frames must be non-empty and rectangular".to_string(),
            "audio-classification",
        ));
    }
    let n_frames = frames.len();
    let mut flat = vec![0.0f32; n_mels * n_frames];
    for (t, frame) in frames.iter().enumerate() {
        for (m, &v) in frame.iter().enumerate() {
            flat[m * n_frames + t] = v;
        }
    }
    Tensor::from_slice(&flat, &[1, n_mels, n_frames])
        .map_err(|e| TrustformersError::pipeline(e.to_string(), "audio-classification"))
}

// ---------------------------------------------------------------------------
// Trait impl
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
    use crate::pipeline::media::audio_dsp::encode_wav_pcm16;

    fn default_pipeline() -> AudioClassificationPipeline {
        AudioClassificationPipeline::new(AudioClassificationConfig::default())
            .expect("default config should be valid")
    }

    fn tone(freq: f32, sample_rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                0.7 * (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin()
            })
            .collect()
    }

    fn assert_unsupported(err: &TrustformersError) {
        match err {
            TrustformersError::FeatureUnavailable { message, .. } => {
                assert!(
                    message.contains("no real model implementation"),
                    "unexpected message: {message}"
                );
            },
            other => panic!("expected FeatureUnavailable, got {other:?}"),
        }
    }

    // ---- Construction ----

    #[test]
    fn test_default_config_creates_pipeline() {
        let _p = default_pipeline();
    }

    #[test]
    fn test_zero_sample_rate_returns_error() {
        let config = AudioClassificationConfig {
            sample_rate: 0,
            ..Default::default()
        };
        assert!(AudioClassificationPipeline::new(config).is_err());
    }

    #[test]
    fn test_zero_mel_bins_returns_error() {
        let config = AudioClassificationConfig {
            num_mel_bins: 0,
            ..Default::default()
        };
        assert!(AudioClassificationPipeline::new(config).is_err());
    }

    // ---- Honesty: no fabricated classification ----

    #[test]
    fn test_classify_raw_audio_reports_unsupported_model() {
        // Regression: `classify` used to return confident labels from
        // `mock_forward` without ever loading a model.
        let config = AudioClassificationConfig {
            top_k: 3,
            ..Default::default()
        };
        let pipeline = AudioClassificationPipeline::new(config).expect("valid config");
        let input = AudioClassificationInput::RawAudio {
            samples: tone(440.0, 16_000, 16_000),
            sample_rate: 16_000,
        };
        let err = pipeline.classify(&input).expect_err("no backbone is available");
        assert_unsupported(&err);
    }

    #[test]
    fn test_classify_batch_reports_unsupported_model() {
        let pipeline = default_pipeline();
        let inputs = vec![
            AudioClassificationInput::RawAudio {
                samples: tone(200.0, 8_000, 8_000),
                sample_rate: 8_000,
            },
            AudioClassificationInput::RawAudio {
                samples: tone(600.0, 8_000, 8_000),
                sample_rate: 8_000,
            },
        ];
        let err = pipeline.classify_batch(&inputs).expect_err("no backbone is available");
        assert_unsupported(&err);
    }

    #[test]
    fn test_mel_spectrogram_input_also_reports_unsupported_model() {
        let pipeline = default_pipeline();
        let input = AudioClassificationInput::MelSpectrogram {
            values: vec![0.5; 128 * 80],
            frames: 128,
            mel_bins: 80,
        };
        let err = pipeline.classify(&input).expect_err("no backbone is available");
        assert_unsupported(&err);
    }

    #[test]
    fn test_new_style_classify_input_reports_unsupported_model() {
        let pipeline = default_pipeline();
        let input = AudioInput::RawPcm {
            samples: tone(300.0, 8_000, 8_000),
            sample_rate: 8_000,
        };
        let err = pipeline.classify_input(input).expect_err("no backbone is available");
        assert_unsupported(&err);
    }

    #[test]
    fn test_unsupported_error_names_the_requested_model() {
        let config = AudioClassificationConfig {
            model_name: "MIT/ast-finetuned-audioset".to_string(),
            ..Default::default()
        };
        let pipeline = AudioClassificationPipeline::new(config).expect("valid");
        let input = AudioClassificationInput::RawAudio {
            samples: tone(440.0, 16_000, 8_000),
            sample_rate: 16_000,
        };
        let err = pipeline.classify(&input).expect_err("no backbone");
        assert!(
            err.to_string().contains("MIT/ast-finetuned-audioset"),
            "err: {err}"
        );
    }

    // ---- Real preprocessing ----

    #[test]
    fn test_extract_log_mel_is_not_all_zero() {
        let pipeline = default_pipeline();
        let mel = pipeline
            .extract_log_mel(&tone(440.0, 16_000, 16_000), 16_000)
            .expect("feature extraction");
        assert_eq!(mel.len(), 100);
        assert_eq!(mel[0].len(), 80);
        let flat: Vec<f32> = mel.iter().flatten().copied().collect();
        assert!(
            flat.iter().any(|&v| v != 0.0),
            "features must not be all zeros"
        );
    }

    #[test]
    fn test_extract_log_mel_differs_between_tones() {
        let pipeline = default_pipeline();
        let a = pipeline.extract_log_mel(&tone(220.0, 16_000, 8_000), 16_000).expect("a");
        let b = pipeline.extract_log_mel(&tone(3000.0, 16_000, 8_000), 16_000).expect("b");
        assert_ne!(a, b, "different tones must produce different features");
    }

    #[test]
    fn test_extract_log_mel_rejects_empty_audio() {
        let pipeline = default_pipeline();
        assert!(pipeline.extract_log_mel(&[], 16_000).is_err());
    }

    #[test]
    fn test_extract_log_mel_rejects_zero_input_rate() {
        let pipeline = default_pipeline();
        assert!(pipeline.extract_log_mel(&[0.1; 100], 0).is_err());
    }

    // ---- File handling ----

    #[test]
    fn test_missing_file_path_returns_error() {
        let pipeline = default_pipeline();
        let tmp = std::env::temp_dir().join("audio_classification_nonexistent.wav");
        let _ = std::fs::remove_file(&tmp);
        let input = AudioClassificationInput::FilePath(tmp);
        assert!(matches!(
            pipeline.classify(&input),
            Err(TrustformersError::Io { .. })
        ));
    }

    #[test]
    fn test_existing_nonwav_file_is_rejected_not_treated_as_silence() {
        // Regression: an existing file used to yield `vec![0.0; sample_rate]`
        // silence and a confident classification.
        let tmp = std::env::temp_dir().join("audio_classification_not_a_wav.wav");
        std::fs::write(&tmp, b"").expect("write temp file");
        let pipeline = default_pipeline();
        let result = pipeline.classify(&AudioClassificationInput::FilePath(tmp.clone()));
        let _ = std::fs::remove_file(&tmp);
        match result {
            Err(TrustformersError::FeatureUnavailable { feature, .. }) => {
                assert_eq!(feature, "audio-codec");
            },
            other => panic!("expected an audio-codec error, got {other:?}"),
        }
    }

    #[test]
    fn test_real_wav_file_decodes_to_real_samples() {
        let tmp = std::env::temp_dir().join("audio_classification_real.wav");
        std::fs::write(&tmp, encode_wav_pcm16(&tone(500.0, 16_000, 8_000), 16_000))
            .expect("write fixture");
        let pipeline = default_pipeline();
        let pcm = pipeline.decode_audio_file(&tmp).expect("decode");
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(pcm.len(), 8_000);
        assert!(
            pcm.iter().any(|&s| s.abs() > 0.5),
            "decoded audio must not be silence"
        );
    }

    // ---- Post-processing (real, reusable) ----

    #[test]
    fn test_rank_logits_orders_and_normalises() {
        let pipeline = default_pipeline();
        let mut logits = vec![0.0f32; pipeline.labels().len()];
        logits[2] = 5.0;
        let ranked = pipeline.rank_logits(&logits).expect("rank");
        assert_eq!(ranked.len(), 5, "top_k defaults to 5");
        assert_eq!(ranked[0].label_id, 2);
        assert!(ranked[0].score > ranked[1].score);
        let total: f32 = normalize_scores(&logits).iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rank_logits_rejects_mismatched_length() {
        let pipeline = default_pipeline();
        assert!(pipeline.rank_logits(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_mel_frames_to_tensor_transposes() {
        let frames = vec![vec![1.0f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let tensor = mel_frames_to_tensor(&frames).expect("tensor");
        assert_eq!(tensor.shape(), vec![1, 2, 3]);
        assert_eq!(
            tensor.to_vec_f32().expect("values"),
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
        );
    }

    #[test]
    fn test_mel_frames_to_tensor_rejects_ragged_input() {
        let frames = vec![vec![1.0f32, 2.0], vec![3.0]];
        assert!(mel_frames_to_tensor(&frames).is_err());
        assert!(mel_frames_to_tensor(&[]).is_err());
    }

    #[test]
    fn test_reshape_mel_validates_dimensions() {
        assert!(reshape_mel(&[1.0, 2.0, 3.0, 4.0], 2, 2).is_ok());
        assert!(reshape_mel(&[1.0, 2.0, 3.0], 2, 2).is_err());
        assert!(reshape_mel(&[], 0, 2).is_err());
    }

    // ---- Resampling ----

    #[test]
    fn test_resample_shorter_signal() {
        let orig = vec![1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let out = resample_linear(&orig, 8_000, 4_000);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_preprocess_pcm_resamples_correctly() {
        let samples = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = AudioClassificationPipeline::preprocess_pcm(&samples, 8_000, 4_000)
            .expect("preprocess_pcm ok");
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_preprocess_pcm_zero_target_rate_errors() {
        let samples = vec![0.1_f32; 100];
        assert!(AudioClassificationPipeline::preprocess_pcm(&samples, 16_000, 0).is_err());
    }

    #[test]
    fn test_preprocess_pcm_same_rate_is_identity() {
        let samples = vec![0.1_f32, 0.2, 0.3, 0.4];
        let result =
            AudioClassificationPipeline::preprocess_pcm(&samples, 16_000, 16_000).expect("ok");
        assert_eq!(result, samples);
    }

    #[test]
    fn test_upsample_doubles_length() {
        let samples = vec![0.0_f32, 1.0, 0.0, 1.0];
        let out = resample_linear(&samples, 4_000, 8_000);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_resample_audio_linear() {
        let samples = vec![0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let config = AudioResampleConfig {
            source_rate: 8_000,
            target_rate: 4_000,
            algorithm: ResampleAlgorithm::LinearInterpolation,
        };
        assert_eq!(resample_audio(&samples, &config).len(), 4);
    }

    #[test]
    fn test_resample_audio_nearest() {
        let samples = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let config = AudioResampleConfig {
            source_rate: 8_000,
            target_rate: 4_000,
            algorithm: ResampleAlgorithm::NearestNeighbor,
        };
        assert_eq!(resample_audio(&samples, &config).len(), 4);
    }

    #[test]
    fn test_resample_audio_same_rate_returns_same() {
        let samples = vec![0.3_f32, 0.5, 0.7];
        let config = AudioResampleConfig {
            source_rate: 16_000,
            target_rate: 16_000,
            algorithm: ResampleAlgorithm::LinearInterpolation,
        };
        assert_eq!(resample_audio(&samples, &config), samples);
    }

    // ---- Mel spectrogram helper ----

    #[test]
    fn test_compute_mel_spectrogram_returns_correct_shape() {
        let pcm = tone(440.0, 16_000, 16_000);
        let n_fft = 512;
        let hop = 160;
        let n_mels = 40;
        let mel = AudioClassificationPipeline::compute_mel_spectrogram(&pcm, n_fft, hop, n_mels);
        assert!(!mel.is_empty());
        let expected_frames = (pcm.len() - n_fft) / hop + 1;
        assert_eq!(mel.len(), expected_frames, "frame count mismatch");
        for frame in &mel {
            assert_eq!(frame.len(), n_mels);
        }
    }

    #[test]
    fn test_compute_mel_spectrogram_empty_pcm_returns_empty() {
        let mel = AudioClassificationPipeline::compute_mel_spectrogram(&[], 512, 160, 40);
        assert!(mel.is_empty());
    }

    #[test]
    fn test_compute_mel_spectrogram_locates_a_tone() {
        // A 3 kHz tone must land in a higher mel band than a 250 Hz tone.
        let peak_band = |freq: f32| {
            let mel = compute_mel_spectrogram_at(&tone(freq, 16_000, 16_000), 16_000, 400, 160, 40)
                .expect("mel");
            let mid = &mel[mel.len() / 2];
            mid.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        assert!(
            peak_band(250.0) < peak_band(3000.0),
            "mel band ordering must follow frequency"
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
                    "frame {fi} bin {bi} has non-finite value {val}"
                );
            }
        }
    }

    // ---- Score normalisation ----

    #[test]
    fn test_normalize_scores_sums_to_one() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let probs = normalize_scores(&logits);
        assert_eq!(probs.len(), logits.len());
        let total: f32 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "got {total}");
    }

    #[test]
    fn test_normalize_scores_preserves_ordering() {
        let logits = vec![1.0_f32, 5.0, 3.0];
        let probs = normalize_scores(&logits);
        assert!(probs[1] > probs[2]);
        assert!(probs[2] > probs[0]);
    }

    #[test]
    fn test_normalize_scores_empty_returns_empty() {
        assert!(normalize_scores(&[]).is_empty());
    }

    #[test]
    fn test_custom_labels_are_resolved() {
        let config = AudioClassificationConfig {
            labels: vec!["cat".to_string(), "dog".to_string(), "bird".to_string()],
            top_k: 2,
            ..Default::default()
        };
        let pipeline = AudioClassificationPipeline::new(config).expect("valid");
        assert_eq!(pipeline.labels(), ["cat", "dog", "bird"]);
        let ranked = pipeline.rank_logits(&[0.1, 5.0, 0.2]).expect("rank");
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].label, "dog");
    }
}
