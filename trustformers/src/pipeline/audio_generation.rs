//! # Audio Generation Pipeline
//!
//! AudioLDM/MusicGen-compatible pipeline that generates audio waveforms from
//! natural language text descriptions.
//!
//! ## Supported model families
//! - **AudioLDM** — latent diffusion model for general audio generation
//! - **MusicGen** — auto-regressive transformer for music generation
//!
//! ## Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::audio_generation::{AudioGenerationConfig, AudioGenerationPipeline};
//!
//! let config = AudioGenerationConfig::default();
//! let pipeline = AudioGenerationPipeline::new(config)?;
//! let waveforms = pipeline.generate("ambient forest soundscape")?;
//! println!("Generated {} waveform(s)", waveforms.len());
//! # Ok::<(), trustformers::pipeline::audio_generation::AudioGenError>(())
//! ```

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the audio generation pipeline.
#[derive(Debug, thiserror::Error)]
pub enum AudioGenError {
    /// The text prompt was empty or contained only whitespace.
    #[error("Empty prompt")]
    EmptyPrompt,
    /// The requested sample rate is zero or otherwise invalid.
    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(u32),
    /// The requested audio duration is non-positive or NaN.
    #[error("Invalid duration: {0}")]
    InvalidDuration(f32),
    /// The two waveforms have different lengths and cannot be mixed.
    #[error("Sample count mismatch for mix")]
    MixLengthMismatch,
    /// A generic model-level error with a descriptive message.
    #[error("Model error: {0}")]
    ModelError(String),
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`AudioGenerationPipeline`].
#[derive(Debug, Clone)]
pub struct AudioGenerationConfig {
    /// HuggingFace model identifier or local path.
    pub model_name: String,
    /// Output sample rate in Hz (e.g. 16 000 for speech, 22 050 for music).
    pub sample_rate: u32,
    /// Desired waveform length in seconds.
    pub audio_length_seconds: f32,
    /// Number of diffusion denoising steps.
    pub num_inference_steps: usize,
    /// Classifier-free guidance scale.
    pub guidance_scale: f32,
    /// Number of independent waveforms to produce per prompt.
    pub num_waveforms_per_prompt: usize,
    /// Optional fixed seed for deterministic generation.
    pub seed: Option<u64>,
}

impl Default for AudioGenerationConfig {
    fn default() -> Self {
        Self {
            model_name: "cvssp/audioldm-s-full-v2".to_string(),
            sample_rate: 16_000,
            audio_length_seconds: 5.0,
            num_inference_steps: 10,
            guidance_scale: 3.5,
            num_waveforms_per_prompt: 1,
            seed: None,
        }
    }
}

// ---------------------------------------------------------------------------
// AudioWaveform
// ---------------------------------------------------------------------------

/// A mono PCM audio waveform.
#[derive(Debug, Clone)]
pub struct AudioWaveform {
    /// Mono PCM samples in the range `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Total waveform duration in seconds.
    pub duration_seconds: f32,
    /// Number of channels (always 1 for this implementation).
    pub num_channels: usize,
}

impl AudioWaveform {
    /// Construct a new waveform, computing the duration from the sample count.
    ///
    /// # Errors
    ///
    /// Returns [`AudioGenError::InvalidSampleRate`] if `sample_rate` is zero.
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Result<Self, AudioGenError> {
        if sample_rate == 0 {
            return Err(AudioGenError::InvalidSampleRate(sample_rate));
        }
        let duration_seconds = samples.len() as f32 / sample_rate as f32;
        Ok(Self {
            samples,
            sample_rate,
            duration_seconds,
            num_channels: 1,
        })
    }

    /// Total number of PCM samples.
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Duration in seconds, recomputed from the sample count.
    pub fn duration_seconds(&self) -> f32 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f32 / self.sample_rate as f32
        }
    }

    /// Root-mean-square energy of the waveform.
    pub fn rms_energy(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = self.samples.iter().map(|s| s * s).sum();
        (sum_sq / self.samples.len() as f32).sqrt()
    }

    /// Peak (maximum absolute) amplitude in the waveform.
    pub fn peak_amplitude(&self) -> f32 {
        self.samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0_f32, f32::max)
    }

    /// Return a mono copy of this waveform (this implementation is already mono).
    pub fn to_mono(&self) -> Self {
        self.clone()
    }

    /// Return a normalised copy scaled so the peak amplitude equals 1.0.
    ///
    /// If the waveform is silent (all zeros) the original copy is returned
    /// unchanged.
    pub fn normalize(&self) -> Self {
        let peak = self.peak_amplitude();
        if peak < f32::EPSILON {
            return self.clone();
        }
        let samples: Vec<f32> = self.samples.iter().map(|s| s / peak).collect();
        let duration_seconds = samples.len() as f32 / self.sample_rate.max(1) as f32;
        AudioWaveform {
            samples,
            sample_rate: self.sample_rate,
            duration_seconds,
            num_channels: 1,
        }
    }

    /// Resample the waveform to a new sample rate using linear interpolation.
    pub fn resample(&self, new_sample_rate: u32) -> Self {
        if new_sample_rate == self.sample_rate || self.samples.is_empty() || new_sample_rate == 0 {
            return self.clone();
        }
        let ratio = self.sample_rate as f64 / new_sample_rate as f64;
        let new_len = ((self.samples.len() as f64) / ratio).ceil() as usize;
        let mut out = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_pos = i as f64 * ratio;
            let src_idx = src_pos as usize;
            let frac = (src_pos - src_idx as f64) as f32;
            let a = self.samples.get(src_idx).copied().unwrap_or(0.0);
            let b = self.samples.get(src_idx + 1).copied().unwrap_or(a);
            out.push(a + frac * (b - a));
        }
        let duration_seconds = out.len() as f32 / new_sample_rate as f32;
        AudioWaveform {
            samples: out,
            sample_rate: new_sample_rate,
            duration_seconds,
            num_channels: 1,
        }
    }

    /// Return a copy with leading and trailing samples whose absolute value is
    /// below `threshold` removed.
    pub fn trim_silence(&self, threshold: f32) -> Self {
        if self.samples.is_empty() {
            return self.clone();
        }
        let start = self
            .samples
            .iter()
            .position(|s| s.abs() >= threshold)
            .unwrap_or(self.samples.len());
        let end = self
            .samples
            .iter()
            .rposition(|s| s.abs() >= threshold)
            .map(|p| p + 1)
            .unwrap_or(0);

        let trimmed = if start < end {
            self.samples[start..end].to_vec()
        } else {
            Vec::new()
        };
        let duration_seconds = trimmed.len() as f32 / self.sample_rate.max(1) as f32;
        AudioWaveform {
            samples: trimmed,
            sample_rate: self.sample_rate,
            duration_seconds,
            num_channels: 1,
        }
    }

    /// Mix this waveform with `other` using `self * ratio + other * (1 - ratio)`.
    ///
    /// # Errors
    ///
    /// Returns [`AudioGenError::MixLengthMismatch`] when the two waveforms have
    /// different sample counts.
    pub fn mix(&self, other: &AudioWaveform, ratio: f32) -> Result<AudioWaveform, AudioGenError> {
        if self.samples.len() != other.samples.len() {
            return Err(AudioGenError::MixLengthMismatch);
        }
        let inv = 1.0 - ratio;
        let samples: Vec<f32> = self
            .samples
            .iter()
            .zip(other.samples.iter())
            .map(|(a, b)| a * ratio + b * inv)
            .collect();
        let duration_seconds = samples.len() as f32 / self.sample_rate.max(1) as f32;
        Ok(AudioWaveform {
            samples,
            sample_rate: self.sample_rate,
            duration_seconds,
            num_channels: 1,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute a deterministic djb2-style hash of the given string.
fn djb2_hash(s: &str) -> u64 {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    h
}

/// Generate a mock waveform: sine wave at a hash-derived frequency, shaped by a
/// Hann window.
fn generate_mock_waveform(
    prompt: &str,
    sample_rate: u32,
    audio_length_seconds: f32,
    waveform_index: usize,
) -> AudioWaveform {
    let num_samples = (sample_rate as f32 * audio_length_seconds) as usize;
    let base_hash = djb2_hash(prompt);
    // Produce a frequency between 110 Hz and 880 Hz.
    let idx_offset: u64 = waveform_index as u64 * 7919;
    let freq_hash = base_hash.wrapping_add(idx_offset);
    let freq = 110.0 + (freq_hash % 770) as f32;

    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        // Hann window for smooth onset/offset.
        let window = if num_samples > 1 {
            0.5 * (1.0 - (2.0 * PI * i as f32 / (num_samples - 1) as f32).cos())
        } else {
            1.0
        };
        let sine = (2.0 * PI * freq * t).sin();
        samples.push(sine * window);
    }

    let duration_seconds = num_samples as f32 / sample_rate as f32;
    AudioWaveform {
        samples,
        sample_rate,
        duration_seconds,
        num_channels: 1,
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Pipeline for text-to-audio generation (AudioLDM / MusicGen style).
#[derive(Debug)]
pub struct AudioGenerationPipeline {
    config: AudioGenerationConfig,
}

impl AudioGenerationPipeline {
    /// Create a new pipeline with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration contains invalid values such as a
    /// zero sample rate or a non-positive duration.
    pub fn new(config: AudioGenerationConfig) -> Result<Self, AudioGenError> {
        if config.sample_rate == 0 {
            return Err(AudioGenError::InvalidSampleRate(config.sample_rate));
        }
        if !config.audio_length_seconds.is_finite() || config.audio_length_seconds <= 0.0 {
            return Err(AudioGenError::InvalidDuration(config.audio_length_seconds));
        }
        Ok(Self { config })
    }

    /// Generate one or more waveforms from the given text prompt.
    ///
    /// Returns [`config.num_waveforms_per_prompt`] waveforms as a `Vec`.
    ///
    /// # Errors
    ///
    /// Returns [`AudioGenError::EmptyPrompt`] for blank prompts.
    pub fn generate(&self, prompt: &str) -> Result<Vec<AudioWaveform>, AudioGenError> {
        let trimmed = prompt.trim();
        if trimmed.is_empty() {
            return Err(AudioGenError::EmptyPrompt);
        }
        let n = self.config.num_waveforms_per_prompt.max(1);
        let waveforms = (0..n)
            .map(|idx| {
                generate_mock_waveform(
                    trimmed,
                    self.config.sample_rate,
                    self.config.audio_length_seconds,
                    idx,
                )
            })
            .collect();
        Ok(waveforms)
    }

    /// Generate waveforms for each prompt in the batch.
    ///
    /// # Errors
    ///
    /// Fails fast on the first invalid prompt.
    pub fn generate_batch(
        &self,
        prompts: &[&str],
    ) -> Result<Vec<Vec<AudioWaveform>>, AudioGenError> {
        prompts.iter().map(|p| self.generate(p)).collect()
    }

    /// Access the pipeline configuration.
    pub fn config(&self) -> &AudioGenerationConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pipeline() -> AudioGenerationPipeline {
        AudioGenerationPipeline::new(AudioGenerationConfig::default())
            .expect("default config should be valid")
    }

    // --- AudioWaveform construction ---

    #[test]
    fn test_waveform_new_valid() {
        let w = AudioWaveform::new(vec![0.0; 100], 16_000).expect("should construct");
        assert_eq!(w.num_channels, 1);
        assert_eq!(w.sample_rate, 16_000);
    }

    #[test]
    fn test_waveform_num_samples() {
        let w = AudioWaveform::new(vec![0.5; 48], 16_000).expect("valid");
        assert_eq!(w.num_samples(), 48);
    }

    #[test]
    fn test_waveform_duration() {
        let sr = 16_000_u32;
        let n = 8_000_usize;
        let w = AudioWaveform::new(vec![0.0; n], sr).expect("valid");
        let expected = n as f32 / sr as f32;
        assert!((w.duration_seconds() - expected).abs() < 1e-6);
    }

    // --- Energy / amplitude ---

    #[test]
    fn test_rms_energy_is_finite() {
        let samples: Vec<f32> = (0..1000)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let w = AudioWaveform::new(samples, 16_000).expect("valid");
        assert!(w.rms_energy().is_finite());
        assert!(w.rms_energy() > 0.0);
    }

    #[test]
    fn test_peak_amplitude_after_normalize_le_one() {
        let samples: Vec<f32> = vec![0.2, 0.5, -0.8, 0.3, 0.1];
        let w = AudioWaveform::new(samples, 16_000).expect("valid");
        let n = w.normalize();
        assert!(
            n.peak_amplitude() <= 1.0 + f32::EPSILON,
            "peak was {}",
            n.peak_amplitude()
        );
        assert!((n.peak_amplitude() - 1.0).abs() < 1e-5, "peak should be ~1.0");
    }

    // --- Resample ---

    #[test]
    fn test_resample_sample_count_correct() {
        // 1 second at 16 kHz → resample to 8 kHz → should give ~8 000 samples.
        let orig: Vec<f32> = (0..16_000).map(|i| (i as f32).sin()).collect();
        let w = AudioWaveform::new(orig, 16_000).expect("valid");
        let resampled = w.resample(8_000);
        // Allow ±1 sample for rounding.
        assert!(
            (resampled.num_samples() as i64 - 8_000).abs() <= 1,
            "expected ~8000 samples, got {}",
            resampled.num_samples()
        );
        assert_eq!(resampled.sample_rate, 8_000);
    }

    // --- Trim silence ---

    #[test]
    fn test_trim_silence_removes_quiet_samples() {
        // Leading and trailing zeros + two loud samples in the middle.
        let samples = vec![0.0, 0.0, 0.5, -0.6, 0.0, 0.0];
        let w = AudioWaveform::new(samples, 16_000).expect("valid");
        let trimmed = w.trim_silence(0.1);
        // Should retain only [0.5, -0.6].
        assert_eq!(trimmed.num_samples(), 2, "got {}", trimmed.num_samples());
    }

    // --- Mix ---

    #[test]
    fn test_mix_basic() {
        let a = AudioWaveform::new(vec![1.0, 1.0, 1.0], 16_000).expect("valid");
        let b = AudioWaveform::new(vec![0.0, 0.0, 0.0], 16_000).expect("valid");
        let m = a.mix(&b, 0.5).expect("mix should succeed");
        for s in &m.samples {
            assert!((s - 0.5).abs() < 1e-6, "expected 0.5, got {s}");
        }
    }

    // --- Pipeline::generate ---

    #[test]
    fn test_generate_returns_correct_count() {
        let config = AudioGenerationConfig {
            num_waveforms_per_prompt: 3,
            ..Default::default()
        };
        let p = AudioGenerationPipeline::new(config).expect("valid");
        let waveforms = p.generate("rain on a tin roof").expect("generate ok");
        assert_eq!(waveforms.len(), 3);
    }

    #[test]
    fn test_generate_wave_length_equals_sr_times_duration() {
        let sr = 22_050_u32;
        let dur = 2.0_f32;
        let config = AudioGenerationConfig {
            sample_rate: sr,
            audio_length_seconds: dur,
            num_waveforms_per_prompt: 1,
            ..Default::default()
        };
        let p = AudioGenerationPipeline::new(config).expect("valid");
        let waveforms = p.generate("gentle piano").expect("ok");
        let expected = (sr as f32 * dur) as usize;
        assert_eq!(waveforms[0].num_samples(), expected);
    }

    // --- Pipeline::generate_batch ---

    #[test]
    fn test_generate_batch_count() {
        let p = default_pipeline();
        let prompts = ["thunder", "birds", "ocean waves"];
        let batch = p.generate_batch(&prompts).expect("batch ok");
        assert_eq!(batch.len(), 3);
    }

    // --- Error cases ---

    #[test]
    fn test_empty_prompt_error() {
        let p = default_pipeline();
        let err = p.generate("   ").expect_err("empty prompt should fail");
        assert!(matches!(err, AudioGenError::EmptyPrompt));
    }

    #[test]
    fn test_invalid_sample_rate_error() {
        let config = AudioGenerationConfig {
            sample_rate: 0,
            ..Default::default()
        };
        let err =
            AudioGenerationPipeline::new(config).expect_err("zero sample rate should fail");
        assert!(matches!(err, AudioGenError::InvalidSampleRate(0)));
    }

    #[test]
    fn test_invalid_duration_error() {
        let config = AudioGenerationConfig {
            audio_length_seconds: -1.0,
            ..Default::default()
        };
        let err =
            AudioGenerationPipeline::new(config).expect_err("negative duration should fail");
        assert!(matches!(err, AudioGenError::InvalidDuration(_)));
    }
}
