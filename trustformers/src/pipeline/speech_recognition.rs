//! Speech Recognition Pipeline (Automatic Speech Recognition / ASR).
//!
//! Converts audio input to text using Whisper-compatible models. Supports
//! transcription (preserving the source language) and translation (always to English),
//! with optional word- or sentence-level timestamps.
//!
//! # Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::speech_recognition::{
//!     SpeechRecognitionConfig, SpeechRecognitionPipeline, AudioInput,
//! };
//!
//! let pipeline = SpeechRecognitionPipeline::new(SpeechRecognitionConfig {
//!     model_name: "openai/whisper-tiny".to_string(),
//!     language: Some("en".to_string()),
//!     ..Default::default()
//! })?;
//!
//! let audio = AudioInput::RawAudio {
//!     samples: vec![0.0_f32; 16_000],
//!     sample_rate: 16_000,
//! };
//!
//! let result = pipeline.transcribe(&audio)?;
//! println!("{}", result.text);
//! ```

use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};
use trustformers_core::device::Device;

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/// Whether the model should preserve the source language or translate to English.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SpeechTask {
    /// Produce a transcript in the original spoken language.
    #[default]
    Transcribe,
    /// Translate the spoken content into English.
    Translate,
}

impl std::fmt::Display for SpeechTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transcribe => write!(f, "transcribe"),
            Self::Translate => write!(f, "translate"),
        }
    }
}

/// Granularity of timestamps attached to the transcription.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ReturnTimestamps {
    /// Do not return timestamps.
    #[default]
    None,
    /// Attach timestamps to each recognised word.
    Word,
    /// Attach timestamps to each sentence segment.
    Sentence,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`SpeechRecognitionPipeline`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRecognitionConfig {
    /// Model name or path (e.g., `"openai/whisper-tiny"`).
    pub model_name: String,
    /// Optional BCP-47 language code. `None` enables auto-detection.
    pub language: Option<String>,
    /// Whether to transcribe or translate.
    pub task: SpeechTask,
    /// Expected sample rate of input audio in Hz (Whisper uses 16 000 Hz).
    pub sample_rate: u32,
    /// Maximum audio length in seconds. Inputs longer than this are truncated.
    pub max_duration_secs: f32,
    /// Timestamp granularity to return.
    pub return_timestamps: ReturnTimestamps,
    /// Inference device.
    pub device: Device,
    /// Number of mel filter banks (80 for Whisper).
    pub num_mel_bins: usize,
    /// FFT window size for mel spectrogram computation.
    pub fft_window_size: usize,
    /// Hop length (stride) for the STFT in samples.
    pub hop_length: usize,
}

impl Default for SpeechRecognitionConfig {
    fn default() -> Self {
        Self {
            model_name: "openai/whisper-tiny".to_string(),
            language: None,
            task: SpeechTask::Transcribe,
            sample_rate: 16_000,
            max_duration_secs: 30.0,
            return_timestamps: ReturnTimestamps::None,
            device: Device::CPU,
            num_mel_bins: 80,
            fft_window_size: 400,  // 25 ms at 16 kHz
            hop_length: 160,        // 10 ms at 16 kHz
        }
    }
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

/// Audio data that can be fed into the speech recognition pipeline.
#[derive(Debug, Clone)]
pub enum AudioInput {
    /// Raw mono PCM samples (f32, normalised to [-1, 1]).
    RawAudio {
        /// PCM samples.
        samples: Vec<f32>,
        /// Sample rate in Hz.
        sample_rate: u32,
    },
    /// Path to an audio file (WAV, MP3, FLAC, OGG, etc.).
    FilePath(PathBuf),
    /// Pre-computed 80-bin log-mel spectrogram stored as `num_frames × 80`.
    MelSpectrogram(Vec<Vec<f32>>),
}

impl AudioInput {
    /// Convenience constructor from a string path.
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self::FilePath(path.into())
    }
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// A single time-aligned segment of the transcription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    /// Transcribed text for this segment.
    pub text: String,
    /// Segment start time in seconds, if timestamps were requested.
    pub start_secs: Option<f32>,
    /// Segment end time in seconds, if timestamps were requested.
    pub end_secs: Option<f32>,
    /// Confidence score in [0, 1].
    pub confidence: f32,
    /// Detected or forced language code for this segment.
    pub language: Option<String>,
}

/// Complete transcription result for a single audio input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    /// Full concatenated transcription.
    pub text: String,
    /// Individual segments (one per word, sentence, or chunk).
    pub segments: Vec<TranscriptionSegment>,
    /// Language code detected by the model, if auto-detection was enabled.
    pub detected_language: Option<String>,
    /// Duration of the processed audio in seconds.
    pub duration_secs: f32,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Automatic speech recognition pipeline.
///
/// This is a framework-level implementation: model inference is stubbed with
/// a deterministic placeholder until a real Whisper backend is wired in.
pub struct SpeechRecognitionPipeline {
    config: SpeechRecognitionConfig,
}

impl SpeechRecognitionPipeline {
    /// Construct a new pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError`] if the configuration contains invalid values
    /// (e.g., zero `sample_rate` or `num_mel_bins`).
    pub fn new(config: SpeechRecognitionConfig) -> Result<Self> {
        if config.sample_rate == 0 {
            return Err(TrustformersError::invalid_input(
                "sample_rate must be > 0",
                Some("sample_rate"),
                Some("> 0"),
                Some("0"),
            ));
        }
        if config.num_mel_bins == 0 {
            return Err(TrustformersError::invalid_input(
                "num_mel_bins must be > 0",
                Some("num_mel_bins"),
                Some("> 0"),
                Some("0"),
            ));
        }
        if config.fft_window_size == 0 {
            return Err(TrustformersError::invalid_input(
                "fft_window_size must be > 0",
                Some("fft_window_size"),
                Some("> 0"),
                Some("0"),
            ));
        }
        if config.hop_length == 0 {
            return Err(TrustformersError::invalid_input(
                "hop_length must be > 0",
                Some("hop_length"),
                Some("> 0"),
                Some("0"),
            ));
        }
        info!(
            model = %config.model_name,
            task = %config.task,
            "Initialising SpeechRecognitionPipeline"
        );
        Ok(Self { config })
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Transcribe a single audio input.
    pub fn transcribe(&self, audio: &AudioInput) -> Result<TranscriptionResult> {
        let (samples, duration_secs) = self.prepare_audio(audio)?;
        self.run_asr(&samples, duration_secs)
    }

    /// Transcribe a batch of audio inputs.
    pub fn transcribe_batch(
        &self,
        audios: &[AudioInput],
    ) -> Result<Vec<TranscriptionResult>> {
        audios.iter().map(|a| self.transcribe(a)).collect()
    }

    /// Transcribe directly from a file path (convenience wrapper).
    pub fn transcribe_file(&self, path: &Path) -> Result<TranscriptionResult> {
        self.transcribe(&AudioInput::FilePath(path.to_path_buf()))
    }

    /// Compute an 80-bin log-mel spectrogram from raw PCM samples.
    ///
    /// Returns a 2-D vector of shape `[num_frames][num_mel_bins]`.
    pub fn compute_mel_spectrogram(
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<Vec<f32>>> {
        if samples.is_empty() {
            return Err(TrustformersError::invalid_input(
                "Cannot compute mel spectrogram from empty audio",
                Some("samples"),
                Some("non-empty slice"),
                Some("empty"),
            ));
        }
        if sample_rate == 0 {
            return Err(TrustformersError::invalid_input(
                "sample_rate must be > 0",
                Some("sample_rate"),
                Some("> 0"),
                Some("0"),
            ));
        }

        // Whisper-compatible parameters
        const N_MEL: usize = 80;
        const FFT_SIZE: usize = 400;
        const HOP: usize = 160;

        let frames = compute_mel_spectrogram_internal(samples, FFT_SIZE, HOP, N_MEL, sample_rate);
        debug!(frames = frames.len(), mel_bins = N_MEL, "Mel spectrogram computed");
        Ok(frames)
    }

    /// Return the model name from the configuration.
    pub fn model_name(&self) -> &str {
        &self.config.model_name
    }

    /// Return a reference to the pipeline configuration.
    pub fn config(&self) -> &SpeechRecognitionConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Resolve `AudioInput` to raw PCM samples at the configured sample rate.
    fn prepare_audio(&self, audio: &AudioInput) -> Result<(Vec<f32>, f32)> {
        match audio {
            AudioInput::RawAudio { samples, sample_rate } => {
                let resampled = resample_linear(samples, *sample_rate, self.config.sample_rate);
                let max_samples =
                    (self.config.max_duration_secs * self.config.sample_rate as f32) as usize;
                let truncated = if resampled.len() > max_samples {
                    warn!(
                        input_len = resampled.len(),
                        max_samples,
                        "Audio truncated to max_duration_secs"
                    );
                    resampled[..max_samples].to_vec()
                } else {
                    resampled
                };
                let duration = truncated.len() as f32 / self.config.sample_rate as f32;
                Ok((truncated, duration))
            }

            AudioInput::FilePath(path) => {
                if !path.exists() {
                    return Err(TrustformersError::Io {
                        message: format!("Audio file not found: '{}'", path.display()),
                        path: Some(path.to_string_lossy().to_string()),
                        suggestion: Some("Verify the file path and ensure the file exists".to_string()),
                    });
                }
                // Without an audio decoding library we return a silent placeholder.
                // In a full implementation this would read WAV/FLAC/MP3 using e.g. symphonia.
                warn!(
                    path = %path.display(),
                    "Audio file decoding is not yet available; returning silence placeholder"
                );
                let n = (self.config.max_duration_secs * self.config.sample_rate as f32) as usize;
                let duration = n as f32 / self.config.sample_rate as f32;
                Ok((vec![0.0_f32; n], duration))
            }

            AudioInput::MelSpectrogram(mel) => {
                if mel.is_empty() {
                    return Err(TrustformersError::invalid_input(
                        "MelSpectrogram input must not be empty",
                        Some("MelSpectrogram"),
                        Some("non-empty 2-D mel matrix"),
                        Some("empty"),
                    ));
                }
                // Reconstruct approximate duration from frame count and hop length
                let duration_secs = mel.len() as f32 * self.config.hop_length as f32
                    / self.config.sample_rate as f32;
                // Flatten mel back to a dummy PCM signal for the stub forward pass
                let fake_pcm: Vec<f32> = mel.iter().flat_map(|row| row.iter().cloned()).collect();
                Ok((fake_pcm, duration_secs))
            }
        }
    }

    /// Run the ASR stub and produce a [`TranscriptionResult`].
    fn run_asr(&self, samples: &[f32], duration_secs: f32) -> Result<TranscriptionResult> {
        // Compute mel spectrogram
        let mel = compute_mel_spectrogram_internal(
            samples,
            self.config.fft_window_size,
            self.config.hop_length,
            self.config.num_mel_bins,
            self.config.sample_rate,
        );

        debug!(
            frames = mel.len(),
            mel_bins = self.config.num_mel_bins,
            duration_secs,
            "Running ASR stub forward pass"
        );

        // Stub: produce a placeholder transcription based on audio energy
        let text = self.stub_decode(&mel, duration_secs);

        let detected_language = self.config.language.clone().or_else(|| {
            // Heuristic stub: derive "language" from energy pattern
            Some("en".to_string())
        });

        // Build segments according to the requested timestamp granularity
        let segments = self.build_segments(&text, duration_secs);

        Ok(TranscriptionResult {
            text,
            segments,
            detected_language,
            duration_secs,
        })
    }

    /// Deterministic stub decoder: generates placeholder text from mel energy.
    fn stub_decode(&self, mel: &[Vec<f32>], duration_secs: f32) -> String {
        if mel.is_empty() {
            return String::new();
        }

        // Compute overall energy
        let energy: f32 = mel
            .iter()
            .flat_map(|row| row.iter())
            .map(|&v| v.abs())
            .sum::<f32>()
            / (mel.len() * mel[0].len().max(1)) as f32;

        let task_tag = match self.config.task {
            SpeechTask::Transcribe => "transcription",
            SpeechTask::Translate => "translation",
        };

        let lang_tag = self
            .config
            .language
            .as_deref()
            .unwrap_or("auto")
            .to_string();

        format!(
            "[{task_tag}|{lang_tag}|{duration:.1}s|energy:{energy:.4}] (stub output — model not loaded)",
            task_tag = task_tag,
            lang_tag = lang_tag,
            duration = duration_secs,
            energy = energy,
        )
    }

    /// Build transcript segments based on the configured timestamp granularity.
    fn build_segments(&self, text: &str, duration_secs: f32) -> Vec<TranscriptionSegment> {
        match self.config.return_timestamps {
            ReturnTimestamps::None => vec![TranscriptionSegment {
                text: text.to_string(),
                start_secs: None,
                end_secs: None,
                confidence: 0.5,
                language: self.config.language.clone(),
            }],

            ReturnTimestamps::Sentence => {
                // Split into mock sentence segments of ~5 seconds each
                let segment_dur = 5.0_f32;
                let n_segs = (duration_secs / segment_dur).ceil().max(1.0) as usize;
                (0..n_segs)
                    .map(|i| {
                        let start = i as f32 * segment_dur;
                        let end = ((i + 1) as f32 * segment_dur).min(duration_secs);
                        TranscriptionSegment {
                            text: format!("[segment {}/{n_segs}] {text}", i + 1),
                            start_secs: Some(start),
                            end_secs: Some(end),
                            confidence: 0.5,
                            language: self.config.language.clone(),
                        }
                    })
                    .collect()
            }

            ReturnTimestamps::Word => {
                // Split the text into words and distribute them over the duration
                let words: Vec<&str> = text.split_whitespace().collect();
                if words.is_empty() {
                    return Vec::new();
                }
                let dur_per_word = duration_secs / words.len() as f32;
                words
                    .iter()
                    .enumerate()
                    .map(|(i, word)| {
                        let start = i as f32 * dur_per_word;
                        let end = start + dur_per_word;
                        TranscriptionSegment {
                            text: word.to_string(),
                            start_secs: Some(start),
                            end_secs: Some(end),
                            confidence: 0.5,
                            language: self.config.language.clone(),
                        }
                    })
                    .collect()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Extended types and SpeechProcessor
// ---------------------------------------------------------------------------

/// A timestamped speech segment with text and confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechSegment {
    /// Segment start time in milliseconds.
    pub start_ms: f32,
    /// Segment end time in milliseconds.
    pub end_ms: f32,
    /// Transcribed text for this segment.
    pub text: String,
    /// Confidence in `[0, 1]`.
    pub confidence: f32,
    /// Detected language for this segment.
    pub language: Option<String>,
}

/// Full transcription with segments, language, and duration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedTranscriptionResult {
    pub full_text: String,
    pub segments: Vec<SpeechSegment>,
    pub language: Option<String>,
    pub duration_ms: f32,
}

/// Errors produced by `SpeechProcessor`.
#[derive(Debug, thiserror::Error)]
pub enum SpeechProcessorError {
    #[error("Frame size must be > 0")]
    InvalidFrameSize,
    #[error("Hop size must be > 0")]
    InvalidHopSize,
    #[error("Sample rate must be > 0")]
    InvalidSampleRate,
}

/// Pure-Rust speech processing utilities.
pub struct SpeechProcessor;

impl SpeechProcessor {
    /// Compute per-frame RMS energy from a mono PCM signal.
    ///
    /// Returns one energy value per frame.
    pub fn compute_frame_energy(
        pcm: &[f32],
        frame_size: usize,
        hop_size: usize,
    ) -> Result<Vec<f32>, SpeechProcessorError> {
        if frame_size == 0 {
            return Err(SpeechProcessorError::InvalidFrameSize);
        }
        if hop_size == 0 {
            return Err(SpeechProcessorError::InvalidHopSize);
        }
        if pcm.is_empty() {
            return Ok(Vec::new());
        }
        let mut energies = Vec::new();
        let mut offset = 0_usize;
        while offset + frame_size <= pcm.len() {
            let frame = &pcm[offset..offset + frame_size];
            let rms = (frame.iter().map(|&s| s * s).sum::<f32>() / frame_size as f32).sqrt();
            energies.push(rms);
            offset += hop_size;
        }
        Ok(energies)
    }

    /// Simple energy-based Voice Activity Detection.
    ///
    /// Returns `true` for frames whose energy exceeds `threshold`.
    pub fn voice_activity_detection(energies: &[f32], threshold: f32) -> Vec<bool> {
        energies.iter().map(|&e| e > threshold).collect()
    }

    /// Split a PCM signal on silence regions.
    ///
    /// Silence is defined as a run of frames below `threshold` lasting
    /// at least `min_silence_ms` milliseconds.
    pub fn split_on_silence(
        pcm: &[f32],
        sample_rate: u32,
        min_silence_ms: f32,
        threshold: f32,
    ) -> Result<Vec<Vec<f32>>, SpeechProcessorError> {
        if sample_rate == 0 {
            return Err(SpeechProcessorError::InvalidSampleRate);
        }
        if pcm.is_empty() {
            return Ok(Vec::new());
        }

        let frame_size = (sample_rate as f32 * 0.01) as usize; // 10ms frames
        let hop_size = frame_size;
        let min_silence_frames =
            ((min_silence_ms / 1000.0) * sample_rate as f32 / frame_size as f32).ceil() as usize;

        let energies = Self::compute_frame_energy(pcm, frame_size.max(1), hop_size.max(1))?;
        let vad = Self::voice_activity_detection(&energies, threshold);

        let mut segments: Vec<Vec<f32>> = Vec::new();
        let mut seg_start = 0_usize;
        let mut silence_count = 0_usize;
        let mut in_speech = false;

        for (i, &is_voice) in vad.iter().enumerate() {
            if is_voice {
                if !in_speech {
                    seg_start = i * hop_size;
                    in_speech = true;
                }
                silence_count = 0;
            } else {
                silence_count += 1;
                if in_speech && silence_count >= min_silence_frames {
                    let seg_end = (i * hop_size).min(pcm.len());
                    if seg_end > seg_start {
                        segments.push(pcm[seg_start..seg_end].to_vec());
                    }
                    in_speech = false;
                    silence_count = 0;
                }
            }
        }
        // Flush final segment
        if in_speech && seg_start < pcm.len() {
            segments.push(pcm[seg_start..].to_vec());
        }

        Ok(segments)
    }

    /// Format a duration in milliseconds as `"HH:MM:SS.mmm"`.
    pub fn format_timestamp(ms: f32) -> String {
        let total_ms = ms.max(0.0) as u64;
        let millis = total_ms % 1000;
        let total_secs = total_ms / 1000;
        let secs = total_secs % 60;
        let total_mins = total_secs / 60;
        let mins = total_mins % 60;
        let hours = total_mins / 60;
        format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, millis)
    }

    /// Compute Word Error Rate (WER) between a reference and hypothesis transcription.
    ///
    /// Uses dynamic-programming edit distance.
    pub fn word_error_rate(reference: &str, hypothesis: &str) -> f32 {
        let ref_words: Vec<&str> = reference.split_whitespace().collect();
        let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();

        if ref_words.is_empty() {
            return if hyp_words.is_empty() { 0.0 } else { 1.0 };
        }

        let n = ref_words.len();
        let m = hyp_words.len();

        // Standard Levenshtein on word sequences
        let mut dp = vec![vec![0_usize; m + 1]; n + 1];
        for i in 0..=n {
            dp[i][0] = i;
        }
        for j in 0..=m {
            dp[0][j] = j;
        }
        for i in 1..=n {
            for j in 1..=m {
                let cost = if ref_words[i - 1] == hyp_words[j - 1] { 0 } else { 1 };
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }

        dp[n][m] as f32 / n as f32
    }
}

// ---------------------------------------------------------------------------
// DSP helpers
// ---------------------------------------------------------------------------

/// Linear interpolation resampler (mono, f32).
fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
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

/// Apply a Hann window to a frame in-place.
fn apply_hann_window(frame: &mut [f32]) {
    let n = frame.len();
    for (i, sample) in frame.iter_mut().enumerate() {
        let w = 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1).max(1) as f32).cos());
        *sample *= w;
    }
}

/// Compute a real DFT magnitude spectrum of a windowed frame.
/// Returns `fft_size / 2 + 1` magnitude bins.
fn rfft_magnitude(frame: &[f32], fft_size: usize) -> Vec<f32> {
    let n = frame.len().min(fft_size);
    let mut padded = vec![0.0_f32; fft_size];
    padded[..n].copy_from_slice(&frame[..n]);

    let half = fft_size / 2 + 1;
    let mut magnitudes = Vec::with_capacity(half);

    for k in 0..half {
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        for (j, &x) in padded.iter().enumerate() {
            let angle = 2.0 * PI * k as f32 * j as f32 / fft_size as f32;
            re += x * angle.cos();
            im -= x * angle.sin();
        }
        magnitudes.push((re * re + im * im).sqrt());
    }
    magnitudes
}

/// Build a triangular mel filterbank.
/// Returns a matrix of shape `[num_mel][num_fft_bins]`.
fn build_mel_filterbank(
    num_mel: usize,
    num_fft_bins: usize,
    sample_rate: u32,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Evenly-spaced mel points including two boundary points
    let n_points = num_mel + 2;
    let mel_points: Vec<f32> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_points - 1) as f32)
        .collect();

    // Convert mel points to FFT bin indices
    let bin_points: Vec<f32> = mel_points
        .iter()
        .map(|&m| {
            let hz = mel_to_hz(m);
            hz * (num_fft_bins - 1) as f32 * 2.0 / sample_rate as f32
        })
        .collect();

    let mut filterbank = vec![vec![0.0_f32; num_fft_bins]; num_mel];
    for m in 0..num_mel {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];
        for k in 0..num_fft_bins {
            let kf = k as f32;
            if kf >= left && kf <= center {
                let denom = center - left;
                if denom > 0.0 {
                    filterbank[m][k] = (kf - left) / denom;
                }
            } else if kf > center && kf <= right {
                let denom = right - center;
                if denom > 0.0 {
                    filterbank[m][k] = (right - kf) / denom;
                }
            }
        }
    }
    filterbank
}

/// Compute a log-mel spectrogram from raw PCM samples.
/// Returns a 2-D matrix of shape `[num_frames][num_mel]`.
fn compute_mel_spectrogram_internal(
    samples: &[f32],
    fft_size: usize,
    hop: usize,
    num_mel: usize,
    sample_rate: u32,
) -> Vec<Vec<f32>> {
    if samples.is_empty() || fft_size == 0 || hop == 0 || num_mel == 0 {
        return Vec::new();
    }

    let num_fft_bins = fft_size / 2 + 1;
    let filterbank =
        build_mel_filterbank(num_mel, num_fft_bins, sample_rate, 0.0, sample_rate as f32 / 2.0);

    let mut frames = Vec::new();
    let mut offset = 0usize;

    while offset + fft_size <= samples.len() {
        let mut frame = samples[offset..offset + fft_size].to_vec();
        apply_hann_window(&mut frame);
        let magnitudes = rfft_magnitude(&frame, fft_size);

        // Apply filterbank
        let mut mel_frame = Vec::with_capacity(num_mel);
        for filter in &filterbank {
            let energy: f32 = filter
                .iter()
                .zip(magnitudes.iter())
                .map(|(f, m)| f * m)
                .sum();
            // Log-compress: log(max(energy, 1e-10))
            mel_frame.push(energy.max(1e-10_f32).ln());
        }
        frames.push(mel_frame);
        offset += hop;
    }

    frames
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pipeline() -> SpeechRecognitionPipeline {
        SpeechRecognitionPipeline::new(SpeechRecognitionConfig::default())
            .expect("default config is valid")
    }

    fn silence(samples: usize) -> AudioInput {
        AudioInput::RawAudio {
            samples: vec![0.0_f32; samples],
            sample_rate: 16_000,
        }
    }

    #[test]
    fn test_pipeline_construction_default() {
        let p = default_pipeline();
        assert_eq!(p.model_name(), "openai/whisper-tiny");
        assert_eq!(p.config().sample_rate, 16_000);
    }

    #[test]
    fn test_pipeline_construction_invalid_sample_rate() {
        let cfg = SpeechRecognitionConfig {
            sample_rate: 0,
            ..Default::default()
        };
        assert!(SpeechRecognitionPipeline::new(cfg).is_err());
    }

    #[test]
    fn test_pipeline_construction_invalid_mel_bins() {
        let cfg = SpeechRecognitionConfig {
            num_mel_bins: 0,
            ..Default::default()
        };
        assert!(SpeechRecognitionPipeline::new(cfg).is_err());
    }

    #[test]
    fn test_transcribe_raw_audio() {
        let p = default_pipeline();
        let result = p.transcribe(&silence(16_000)).expect("transcribe silence");
        assert!(!result.text.is_empty());
        assert!(result.duration_secs > 0.0);
        assert!(!result.segments.is_empty());
    }

    #[test]
    fn test_transcribe_batch() {
        let p = default_pipeline();
        let audios = vec![silence(8_000), silence(16_000), silence(4_000)];
        let results = p.transcribe_batch(&audios).expect("batch transcribe");
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_transcribe_word_timestamps() {
        let p = SpeechRecognitionPipeline::new(SpeechRecognitionConfig {
            return_timestamps: ReturnTimestamps::Word,
            ..Default::default()
        })
        .expect("valid config");
        let result = p.transcribe(&silence(16_000)).expect("transcribe");
        // Each segment should have timestamps
        for seg in &result.segments {
            assert!(seg.start_secs.is_some());
            assert!(seg.end_secs.is_some());
        }
    }

    #[test]
    fn test_transcribe_sentence_timestamps() {
        let p = SpeechRecognitionPipeline::new(SpeechRecognitionConfig {
            return_timestamps: ReturnTimestamps::Sentence,
            max_duration_secs: 10.0,
            ..Default::default()
        })
        .expect("valid config");
        let result = p.transcribe(&silence(16_000 * 10)).expect("transcribe 10s");
        assert!(!result.segments.is_empty());
        for seg in &result.segments {
            assert!(seg.start_secs.is_some());
            assert!(seg.end_secs.is_some());
        }
    }

    #[test]
    fn test_transcribe_translate_task() {
        let p = SpeechRecognitionPipeline::new(SpeechRecognitionConfig {
            task: SpeechTask::Translate,
            language: Some("fr".to_string()),
            ..Default::default()
        })
        .expect("valid config");
        let result = p.transcribe(&silence(8_000)).expect("translate");
        assert!(result.text.contains("translation"));
    }

    #[test]
    fn test_transcribe_mel_spectrogram_input() {
        let p = default_pipeline();
        let mel = vec![vec![0.0_f32; 80]; 100];
        let result = p
            .transcribe(&AudioInput::MelSpectrogram(mel))
            .expect("mel input");
        assert!(!result.text.is_empty());
    }

    #[test]
    fn test_transcribe_mel_spectrogram_empty_errors() {
        let p = default_pipeline();
        let result = p.transcribe(&AudioInput::MelSpectrogram(Vec::new()));
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_mel_spectrogram_basic() {
        let samples = vec![0.0_f32; 16_000];
        let mel = SpeechRecognitionPipeline::compute_mel_spectrogram(&samples, 16_000)
            .expect("mel spectrogram");
        assert!(!mel.is_empty());
        assert_eq!(mel[0].len(), 80);
    }

    #[test]
    fn test_compute_mel_spectrogram_empty_input() {
        let result = SpeechRecognitionPipeline::compute_mel_spectrogram(&[], 16_000);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_mel_spectrogram_zero_rate() {
        let result = SpeechRecognitionPipeline::compute_mel_spectrogram(&[0.0_f32; 100], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_transcribe_file_missing() {
        let p = default_pipeline();
        let tmp = std::env::temp_dir().join("definitely_does_not_exist_asr.wav");
        let result = p.transcribe_file(&tmp);
        assert!(result.is_err());
    }

    #[test]
    fn test_transcribe_file_exists_placeholder() {
        let tmp_dir = std::env::temp_dir();
        let audio_path = tmp_dir.join("tf_asr_test_placeholder.wav");
        std::fs::write(&audio_path, b"RIFF....fake wav").expect("write fake wav");

        let p = default_pipeline();
        let result = p.transcribe_file(&audio_path).expect("transcribe existing file");
        assert!(!result.text.is_empty());

        std::fs::remove_file(&audio_path).ok();
    }

    #[test]
    fn test_resample_same_rate() {
        let samples = vec![1.0_f32, 2.0, 3.0, 4.0];
        let resampled = resample_linear(&samples, 16_000, 16_000);
        assert_eq!(resampled, samples);
    }

    #[test]
    fn test_resample_down() {
        let samples: Vec<f32> = (0..32_000).map(|i| i as f32).collect();
        let resampled = resample_linear(&samples, 32_000, 16_000);
        assert_eq!(resampled.len(), 16_000);
    }

    #[test]
    fn test_speech_task_display() {
        assert_eq!(SpeechTask::Transcribe.to_string(), "transcribe");
        assert_eq!(SpeechTask::Translate.to_string(), "translate");
    }

    #[test]
    fn test_audio_input_from_path() {
        let input = AudioInput::from_path("/tmp/test.wav");
        matches!(input, AudioInput::FilePath(_));
    }

    // -----------------------------------------------------------------------
    // SpeechProcessor tests (15+)
    // -----------------------------------------------------------------------

    // 1. compute_frame_energy: constant tone has consistent energy
    #[test]
    fn test_compute_frame_energy_constant() {
        let pcm = vec![0.5_f32; 1600]; // 100ms at 16kHz
        let energies = SpeechProcessor::compute_frame_energy(&pcm, 160, 160)
            .expect("energy ok");
        assert!(!energies.is_empty());
        for &e in &energies {
            assert!((e - 0.5).abs() < 1e-4, "energy should be ~0.5, got {e}");
        }
    }

    // 2. compute_frame_energy: silence gives ~0 energy
    #[test]
    fn test_compute_frame_energy_silence() {
        let pcm = vec![0.0_f32; 800];
        let energies = SpeechProcessor::compute_frame_energy(&pcm, 160, 160)
            .expect("energy ok");
        for &e in &energies {
            assert!(e < 1e-6, "silence energy should be ~0, got {e}");
        }
    }

    // 3. compute_frame_energy: frame count
    #[test]
    fn test_compute_frame_energy_frame_count() {
        let pcm = vec![1.0_f32; 800];
        // frame_size=100, hop=100 → floor((800-100)/100)+1 = 8 frames
        let energies = SpeechProcessor::compute_frame_energy(&pcm, 100, 100)
            .expect("energy ok");
        assert_eq!(energies.len(), 8, "expected 8 frames, got {}", energies.len());
    }

    // 4. compute_frame_energy: zero frame_size returns error
    #[test]
    fn test_compute_frame_energy_zero_frame_size() {
        let result = SpeechProcessor::compute_frame_energy(&[1.0], 0, 10);
        assert!(result.is_err());
    }

    // 5. compute_frame_energy: zero hop_size returns error
    #[test]
    fn test_compute_frame_energy_zero_hop_size() {
        let result = SpeechProcessor::compute_frame_energy(&[1.0], 10, 0);
        assert!(result.is_err());
    }

    // 6. compute_frame_energy: empty pcm gives empty energies
    #[test]
    fn test_compute_frame_energy_empty_pcm() {
        let energies = SpeechProcessor::compute_frame_energy(&[], 160, 160)
            .expect("empty ok");
        assert!(energies.is_empty());
    }

    // 7. voice_activity_detection: above-threshold frames are active
    #[test]
    fn test_vad_above_threshold() {
        let energies = vec![0.1_f32, 0.8, 0.05, 0.9, 0.02];
        let vad = SpeechProcessor::voice_activity_detection(&energies, 0.5);
        assert_eq!(vad, vec![false, true, false, true, false]);
    }

    // 8. voice_activity_detection: all silence
    #[test]
    fn test_vad_all_silence() {
        let energies = vec![0.01_f32; 10];
        let vad = SpeechProcessor::voice_activity_detection(&energies, 0.5);
        assert!(vad.iter().all(|&v| !v));
    }

    // 9. voice_activity_detection: all speech
    #[test]
    fn test_vad_all_speech() {
        let energies = vec![0.9_f32; 10];
        let vad = SpeechProcessor::voice_activity_detection(&energies, 0.5);
        assert!(vad.iter().all(|&v| v));
    }

    // 10. format_timestamp: zero milliseconds
    #[test]
    fn test_format_timestamp_zero() {
        assert_eq!(SpeechProcessor::format_timestamp(0.0), "00:00:00.000");
    }

    // 11. format_timestamp: 1 hour
    #[test]
    fn test_format_timestamp_one_hour() {
        assert_eq!(
            SpeechProcessor::format_timestamp(3_600_000.0),
            "01:00:00.000"
        );
    }

    // 12. format_timestamp: 1 min 30.5 sec
    #[test]
    fn test_format_timestamp_min_sec() {
        assert_eq!(
            SpeechProcessor::format_timestamp(90_500.0),
            "00:01:30.500"
        );
    }

    // 13. format_timestamp: negative clamped to 0
    #[test]
    fn test_format_timestamp_negative() {
        assert_eq!(SpeechProcessor::format_timestamp(-100.0), "00:00:00.000");
    }

    // 14. word_error_rate: identical strings give 0.0
    #[test]
    fn test_wer_identical() {
        let wer = SpeechProcessor::word_error_rate("hello world", "hello world");
        assert!((wer).abs() < 1e-5, "identical WER should be 0.0, got {wer}");
    }

    // 15. word_error_rate: completely different gives 1.0
    #[test]
    fn test_wer_completely_different() {
        let wer = SpeechProcessor::word_error_rate("cat sat mat", "dog ran far");
        assert!((wer - 1.0).abs() < 1e-5, "completely different WER should be 1.0, got {wer}");
    }

    // 16. word_error_rate: single substitution
    #[test]
    fn test_wer_single_substitution() {
        // "hello world" → "hello earth": 1 substitution out of 2 words
        let wer = SpeechProcessor::word_error_rate("hello world", "hello earth");
        assert!((wer - 0.5).abs() < 1e-5, "single sub WER should be 0.5, got {wer}");
    }

    // 17. word_error_rate: empty reference
    #[test]
    fn test_wer_empty_reference() {
        let wer = SpeechProcessor::word_error_rate("", "something");
        assert_eq!(wer, 1.0);
    }

    // 18. word_error_rate: both empty gives 0.0
    #[test]
    fn test_wer_both_empty() {
        let wer = SpeechProcessor::word_error_rate("", "");
        assert!((wer).abs() < 1e-5);
    }

    // 19. split_on_silence: silence-only audio gives empty segments
    #[test]
    fn test_split_on_silence_silence_only() {
        let silence = vec![0.0_f32; 16_000];
        let segs =
            SpeechProcessor::split_on_silence(&silence, 16_000, 100.0, 0.01)
                .expect("split ok");
        assert!(segs.is_empty(), "silence-only audio should give no segments");
    }

    // 20. split_on_silence: zero sample_rate returns error
    #[test]
    fn test_split_on_silence_zero_sample_rate() {
        let result = SpeechProcessor::split_on_silence(&[1.0_f32; 100], 0, 100.0, 0.1);
        assert!(result.is_err());
    }

    // 21. SpeechSegment construction
    #[test]
    fn test_speech_segment_construction() {
        let seg = SpeechSegment {
            start_ms: 0.0,
            end_ms: 500.0,
            text: "hello".to_string(),
            confidence: 0.9,
            language: Some("en".to_string()),
        };
        assert_eq!(seg.text, "hello");
        assert!((seg.end_ms - 500.0).abs() < 1e-5);
    }

    // 22. DetailedTranscriptionResult construction
    #[test]
    fn test_detailed_transcription_result() {
        let result = DetailedTranscriptionResult {
            full_text: "hello world".to_string(),
            segments: vec![],
            language: Some("en".to_string()),
            duration_ms: 1000.0,
        };
        assert_eq!(result.full_text, "hello world");
        assert!(result.segments.is_empty());
    }

    // 23. word_error_rate: insertion (hypothesis longer than reference)
    #[test]
    fn test_wer_insertion() {
        // ref: 2 words, hyp: 3 words (1 insertion) → WER = 1/2
        let wer = SpeechProcessor::word_error_rate("hello world", "hello beautiful world");
        assert!((wer - 0.5).abs() < 1e-5, "insertion WER should be 0.5, got {wer}");
    }
}
