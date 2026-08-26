//! # Speech-to-text (ASR) pipeline
//!
//! Real audio front-end, honest back-end.
//!
//! ## What is real here
//!
//! * **Container decoding** — RIFF/WAVE (PCM 8/16/24/32-bit and IEEE float
//!   32/64-bit, any channel count) via [`audio_dsp::decode_wav`].
//!   Base64 payloads are decoded with the `base64` crate and then *sniffed*,
//!   not assumed to be WAV.
//! * **Resampling** — linear interpolation to the configured rate.
//! * **Feature extraction** — Hann-windowed STFT (pure-Rust `oxifft`), Slaney
//!   mel filterbank, Whisper-style log compression.
//!
//! ## What is not available
//!
//! The pipeline runs inference only when a real speech model is attached. The
//! generic [`AutoModel`] carries no speech architecture, so
//! [`SpeechToTextPipeline::new`] produces a pipeline whose
//! [`Pipeline::__call__`] returns a structured
//! [`TrustformersError::FeatureUnavailable`] listing the supported backends.
//!
//! With the `whisper` feature enabled, [`SpeechToTextPipeline::with_whisper`]
//! accepts a caller-constructed
//! [`trustformers_models::whisper::SpeechRecognitionTask`] and the pipeline
//! then runs the real encoder–decoder forward pass and decodes the produced
//! token IDs with the pipeline's tokenizer.
//!
//! Under no configuration does this pipeline invent a transcript, a language,
//! or a confidence score.

use crate::error::{Result, TrustformersError};
use crate::pipeline::media::audio_dsp::{self, MelConfig};
use crate::pipeline::media::unsupported_model;
use crate::pipeline::{BasePipeline, Pipeline};
use crate::{AutoModel, AutoTokenizer};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::path::Path;
#[cfg(feature = "whisper")]
use std::sync::Arc;

/// Architectures this pipeline can actually execute.
const SUPPORTED_ARCHITECTURES: &[&str] = &["whisper (requires the `whisper` feature)"];

/// Audio input for speech-to-text pipeline
#[derive(Debug, Clone)]
pub enum AudioInput {
    /// File path to audio file
    FilePath(String),
    /// Raw audio samples (f32) with sample rate
    RawAudio { samples: Vec<f32>, sample_rate: u32 },
    /// Base64 encoded audio data
    Base64(String),
    /// Audio bytes with format info
    Bytes {
        data: Vec<u8>,
        format: AudioFormat,
        sample_rate: u32,
    },
}

/// Supported audio formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    Wav,
    Flac,
    Mp3,
    M4a,
    Ogg,
    WebM,
}

impl AudioFormat {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "wav" => Some(Self::Wav),
            "flac" => Some(Self::Flac),
            "mp3" => Some(Self::Mp3),
            "m4a" => Some(Self::M4a),
            "ogg" => Some(Self::Ogg),
            "webm" => Some(Self::WebM),
            _ => None,
        }
    }

    /// Whether this crate can decode the container without external codecs.
    ///
    /// Only uncompressed RIFF/WAVE is decodable in pure Rust here; the
    /// compressed formats need a codec that is not part of this workspace.
    pub fn is_decodable(self) -> bool {
        matches!(self, AudioFormat::Wav)
    }
}

/// Speech-to-text output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechToTextOutput {
    /// Transcribed text
    pub text: String,
    /// Mean per-token probability of the emitted tokens, or `None` when the
    /// backend does not expose token scores. Never a fabricated constant.
    pub confidence: Option<f32>,
    /// Word-level timestamps, when the backend produces them.
    pub word_timestamps: Option<Vec<WordTimestamp>>,
    /// Language detected (if multi-language model)
    pub language: Option<String>,
    /// Measured wall-clock processing time in milliseconds.
    pub processing_time_ms: Option<u64>,
}

/// Word-level timestamp information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start_time: f64, // seconds
    pub end_time: f64,   // seconds
    pub confidence: f32, // 0.0 to 1.0
}

/// Configuration for speech-to-text processing
#[derive(Clone, Debug)]
pub struct SpeechToTextConfig {
    /// Target sample rate for audio preprocessing
    pub sample_rate: u32,
    /// Maximum audio duration in seconds
    pub max_duration: Option<f64>,
    /// Return word-level timestamps
    pub return_timestamps: bool,
    /// Target language (for multilingual models)
    pub language: Option<String>,
    /// Task type (transcribe or translate)
    pub task: SpeechTask,
    /// Beam search configuration
    pub num_beams: usize,
    /// Use temperature sampling
    pub temperature: f32,
    /// Length penalty for beam search
    pub length_penalty: f32,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// No repeat n-gram size
    pub no_repeat_ngram_size: usize,
    /// Chunk length for long audio (in seconds)
    pub chunk_length_s: Option<f64>,
    /// Stride length for overlapping chunks
    pub stride_length_s: Option<f64>,
}

/// Default decoder token budget (Whisper's decoder context length).
pub const DEFAULT_MAX_NEW_TOKENS: usize = 448;

/// Whisper's start-of-transcript special token.
///
/// Its numeric id differs between checkpoints, so it is always resolved through
/// the tokenizer (or an explicit override) rather than hardcoded.
pub const WHISPER_SOT_TOKEN: &str = "<|startoftranscript|>";

impl Default for SpeechToTextConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,       // Whisper default
            max_duration: Some(30.0), // 30 seconds max
            return_timestamps: false,
            language: None, // Auto-detect
            task: SpeechTask::Transcribe,
            num_beams: 1,     // Greedy decoding by default
            temperature: 0.0, // Deterministic
            length_penalty: 1.0,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: 0,
            chunk_length_s: Some(30.0), // 30-second chunks
            stride_length_s: Some(5.0), // 5-second stride
        }
    }
}

/// Speech recognition task type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SpeechTask {
    /// Transcribe in the same language
    Transcribe,
    /// Translate to English
    Translate,
}

/// The inference backend attached to a [`SpeechToTextPipeline`].
#[derive(Clone)]
pub enum SpeechToTextBackend {
    /// No speech model attached — inference reports the architecture as
    /// unsupported instead of returning a synthesised transcript.
    Unavailable,
    /// A caller-supplied Whisper encoder–decoder with real weights.
    #[cfg(feature = "whisper")]
    Whisper(Arc<trustformers_models::whisper::SpeechRecognitionTask>),
}

impl std::fmt::Debug for SpeechToTextBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable => f.write_str("SpeechToTextBackend::Unavailable"),
            #[cfg(feature = "whisper")]
            Self::Whisper(_) => f.write_str("SpeechToTextBackend::Whisper"),
        }
    }
}

/// Pipeline for speech-to-text tasks (ASR)
#[derive(Clone)]
pub struct SpeechToTextPipeline {
    base: BasePipeline<AutoModel, AutoTokenizer>,
    config: SpeechToTextConfig,
    backend: SpeechToTextBackend,
    max_new_tokens: usize,
    decoder_start_token: Option<u32>,
}

impl SpeechToTextPipeline {
    /// Create a new speech-to-text pipeline.
    ///
    /// [`AutoModel`] carries no speech architecture, so the resulting pipeline
    /// has [`SpeechToTextBackend::Unavailable`]: audio preprocessing works, but
    /// transcription returns a structured error. Attach a real backend with
    /// [`Self::with_whisper`].
    pub fn new(model: AutoModel, tokenizer: AutoTokenizer) -> Result<Self> {
        let base = BasePipeline::new(model, tokenizer);
        let config = SpeechToTextConfig::default();
        // Validate the default rate eagerly so the error surfaces at construction.
        AudioFeatureExtractor::new(config.sample_rate)?;

        Ok(Self {
            base,
            config,
            backend: SpeechToTextBackend::Unavailable,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            decoder_start_token: None,
        })
    }

    /// Build the feature extractor for the pipeline's current sample rate.
    fn extractor(&self) -> Result<AudioFeatureExtractor> {
        AudioFeatureExtractor::new(self.config.sample_rate)
    }

    /// Cap the number of tokens the decoder may emit per utterance.
    pub fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }

    /// Set the decoder's start-of-transcript token id explicitly.
    ///
    /// Overrides the tokenizer lookup of [`WHISPER_SOT_TOKEN`]. Use this when
    /// the tokenizer does not expose Whisper's special tokens.
    pub fn with_decoder_start_token(mut self, token_id: u32) -> Self {
        self.decoder_start_token = Some(token_id);
        self
    }

    /// The configured decoder start token override, if any.
    pub fn decoder_start_token(&self) -> Option<u32> {
        self.decoder_start_token
    }

    /// Attach a real Whisper model (with caller-loaded weights) as the backend.
    #[cfg(feature = "whisper")]
    pub fn with_whisper(
        mut self,
        task: Arc<trustformers_models::whisper::SpeechRecognitionTask>,
    ) -> Self {
        self.backend = SpeechToTextBackend::Whisper(task);
        self
    }

    /// The currently attached backend.
    pub fn backend(&self) -> &SpeechToTextBackend {
        &self.backend
    }

    /// Create pipeline with custom configuration.
    ///
    /// An invalid sample rate is not rejected here (the builder is infallible);
    /// it surfaces as a structured error the first time audio is preprocessed.
    pub fn with_config(mut self, config: SpeechToTextConfig) -> Self {
        self.config = config;
        self
    }

    /// Set target language for transcription
    pub fn with_language(mut self, language: String) -> Self {
        self.config.language = Some(language);
        self
    }

    /// Enable word-level timestamps
    pub fn with_timestamps(mut self, enable: bool) -> Self {
        self.config.return_timestamps = enable;
        self
    }

    /// Set task type (transcribe or translate)
    pub fn with_task(mut self, task: SpeechTask) -> Self {
        self.config.task = task;
        self
    }

    /// Set audio chunk length for processing long audio
    pub fn with_chunk_length(mut self, chunk_length_s: f64) -> Self {
        self.config.chunk_length_s = Some(chunk_length_s);
        self
    }

    /// Process audio file from path
    pub fn transcribe_file<P: AsRef<Path>>(&self, audio_path: P) -> Result<SpeechToTextOutput> {
        let input = AudioInput::FilePath(audio_path.as_ref().to_string_lossy().to_string());
        self.__call__(input)
    }

    /// Process raw audio samples
    pub fn transcribe_samples(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> Result<SpeechToTextOutput> {
        let input = AudioInput::RawAudio {
            samples,
            sample_rate,
        };
        self.__call__(input)
    }

    /// Process audio in streaming fashion (for real-time)
    pub fn transcribe_streaming(&self, audio_chunk: &[f32]) -> Result<SpeechToTextOutput> {
        // For streaming, we process shorter chunks
        let input = AudioInput::RawAudio {
            samples: audio_chunk.to_vec(),
            sample_rate: self.config.sample_rate,
        };
        self.__call__(input)
    }

    /// Decode an [`AudioInput`] into mono PCM at the pipeline's sample rate.
    ///
    /// # Errors
    ///
    /// Returns a structured error when the container cannot be decoded. It
    /// never substitutes silence for undecodable audio.
    pub fn decode_input(&self, input: &AudioInput) -> Result<Vec<f32>> {
        let extractor = self.extractor()?;
        let target = self.config.sample_rate;
        match input {
            AudioInput::FilePath(path) => extractor.load_pcm(path, target),
            AudioInput::RawAudio {
                samples,
                sample_rate,
            } => audio_dsp::resample_linear(samples, *sample_rate, target),
            AudioInput::Base64(encoded) => {
                let decoded = base64::engine::general_purpose::STANDARD
                    .decode(encoded.trim())
                    .map_err(|e| {
                        TrustformersError::invalid_input_simple(format!(
                            "Failed to decode base64 audio: {e}"
                        ))
                    })?;
                if decoded.is_empty() {
                    return Err(TrustformersError::invalid_input_simple(
                        "base64 audio payload decoded to zero bytes".to_string(),
                    ));
                }
                // Sniff the container instead of assuming WAV.
                if !audio_dsp::is_wav(&decoded) {
                    return Err(TrustformersError::feature_unavailable(
                        "base64 audio payload is not a RIFF/WAVE stream; only uncompressed WAV \
                         can be decoded without an external codec"
                            .to_string(),
                        "audio-codec",
                    ));
                }
                let audio = audio_dsp::decode_wav(&decoded)?;
                audio_dsp::resample_linear(&audio.samples, audio.sample_rate, target)
            },
            AudioInput::Bytes {
                data,
                format,
                sample_rate,
            } => {
                let audio = extractor.decode_bytes(data, *format)?;
                // Trust the container's own rate; fall back to the caller's hint
                // only when the container did not carry one.
                let source_rate =
                    if audio.sample_rate > 0 { audio.sample_rate } else { *sample_rate };
                audio_dsp::resample_linear(&audio.samples, source_rate, target)
            },
        }
    }

    /// Pre-process audio input into log-mel features.
    pub fn preprocess_audio(&self, input: &AudioInput) -> Result<AudioFeatures> {
        let pcm = self.decode_input(input)?;
        self.extractor()?.extract_features(&pcm)
    }

    /// Run the attached backend on the extracted features.
    fn run_backend(&self, features: &AudioFeatures) -> Result<SpeechToTextOutput> {
        match &self.backend {
            SpeechToTextBackend::Unavailable => {
                use trustformers_core::traits::Tokenizer as _;
                tracing::trace!(
                    duration_s = features.duration_s,
                    tokenizer_vocab_size = self.base.tokenizer.vocab_size(),
                    "no speech architecture attached; refusing rather than fabricating a \
                     transcript"
                );
                Err(unsupported_model(
                    "speech-to-text",
                    "AutoModel (no speech architecture attached)",
                    SUPPORTED_ARCHITECTURES,
                ))
            },
            #[cfg(feature = "whisper")]
            SpeechToTextBackend::Whisper(task) => self.run_whisper(task, features),
        }
    }

    /// Resolve the decoder's `<|startoftranscript|>` token id.
    ///
    /// Uses the explicit override from [`Self::with_decoder_start_token`] when
    /// set, otherwise looks the special token up in the pipeline's tokenizer.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::Model`] when neither source supplies an id
    /// — the pipeline refuses to guess a constant.
    #[cfg(feature = "whisper")]
    fn resolve_decoder_start_token(&self) -> Result<u32> {
        use trustformers_core::traits::Tokenizer as _;

        if let Some(id) = self.decoder_start_token {
            return Ok(id);
        }
        self.base.tokenizer.token_to_id(WHISPER_SOT_TOKEN).ok_or_else(|| {
            TrustformersError::model(
                format!(
                    "the tokenizer does not define `{WHISPER_SOT_TOKEN}`, so the decoder \
                         start token is unknown; set it explicitly with \
                         `with_decoder_start_token`"
                ),
                "whisper",
            )
        })
    }

    #[cfg(feature = "whisper")]
    fn run_whisper(
        &self,
        task: &trustformers_models::whisper::SpeechRecognitionTask,
        features: &AudioFeatures,
    ) -> Result<SpeechToTextOutput> {
        use trustformers_core::traits::Tokenizer as _;

        let mel = features.to_whisper_mel_tensor()?;
        let config = task.config();
        let vocab_size = config.vocab_size;
        if vocab_size == 0 {
            return Err(TrustformersError::model(
                "Whisper config reports a zero-sized vocabulary".to_string(),
                "whisper",
            ));
        }
        let eos_token = (vocab_size - 1) as u32;
        // The decoder prompt must start with the checkpoint's real
        // <|startoftranscript|> id. Resolve it from the tokenizer (or from an
        // explicit override) rather than assuming a constant — guessing here
        // would silently decode from the wrong prefix.
        let start_token = self.resolve_decoder_start_token()?;
        let mut decoder_ids: Vec<u32> = vec![start_token];
        let mut generated: Vec<u32> = Vec::new();
        let mut token_probs: Vec<f32> = Vec::new();

        for _ in 0..self.max_new_tokens {
            let logits = task
                .forward(&mel, &decoder_ids)
                .map_err(|e| TrustformersError::model(e.to_string(), "whisper"))?;
            let (next, prob) = argmax_last_position(&logits)?;
            if next == eos_token {
                break;
            }
            generated.push(next);
            token_probs.push(prob);
            decoder_ids.push(next);
        }

        let text = if generated.is_empty() {
            String::new()
        } else {
            self.base.tokenizer.decode(&generated).map_err(|e| {
                TrustformersError::model(
                    format!("failed to decode Whisper token IDs: {e}"),
                    "whisper",
                )
            })?
        };

        let confidence = if token_probs.is_empty() {
            None
        } else {
            Some(token_probs.iter().sum::<f32>() / token_probs.len() as f32)
        };

        Ok(SpeechToTextOutput {
            text,
            confidence,
            // Word alignment requires cross-attention timestamps, which this
            // backend does not expose; reporting `None` beats inventing spans.
            word_timestamps: None,
            language: self.config.language.clone(),
            processing_time_ms: None,
        })
    }
}

/// Extract the argmax token id and its softmax probability from the final
/// position of a `[batch, seq, vocab]` logits tensor.
#[cfg(feature = "whisper")]
fn argmax_last_position(logits: &crate::core::tensor::Tensor) -> Result<(u32, f32)> {
    let shape = logits.shape().to_vec();
    if shape.len() != 3 {
        return Err(TrustformersError::model(
            format!("expected [batch, seq, vocab] logits, got shape {shape:?}"),
            "whisper",
        ));
    }
    let (seq_len, vocab) = (shape[1], shape[2]);
    if seq_len == 0 || vocab == 0 {
        return Err(TrustformersError::model(
            "decoder produced an empty logits tensor".to_string(),
            "whisper",
        ));
    }
    let flat = logits.to_vec_f32().map_err(TrustformersError::from)?;
    let offset = (seq_len - 1) * vocab;
    let row = flat.get(offset..offset + vocab).ok_or_else(|| {
        TrustformersError::model("logits buffer too small".to_string(), "whisper")
    })?;

    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    let sum_exp: f32 = row.iter().map(|&v| (v - best_val).exp()).sum();
    let prob = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
    Ok((best as u32, prob))
}

impl Pipeline for SpeechToTextPipeline {
    type Input = AudioInput;
    type Output = SpeechToTextOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        let start_time = std::time::Instant::now();

        // 1. Decode + resample + extract real log-mel features.
        let audio_features = self.preprocess_audio(&input)?;
        let audio_duration = audio_features.duration();

        // 2. Check audio duration limits
        if let Some(max_duration) = self.config.max_duration {
            if audio_duration > max_duration {
                return Err(TrustformersError::invalid_input_simple(format!(
                    "Audio duration ({audio_duration:.2}s) exceeds maximum allowed \
                     ({max_duration:.2}s)"
                )));
            }
        }

        // 3. Run the attached backend (or report that none is attached).
        let mut result = self.run_backend(&audio_features)?;

        // 4. Report the measured wall-clock time.
        result.processing_time_ms = Some(start_time.elapsed().as_millis() as u64);

        Ok(result)
    }
}

/// Audio feature extractor for speech models.
///
/// Computes a real Whisper-style log-mel spectrogram; see
/// [`crate::pipeline::media::audio_dsp`] for the exact conventions.
#[derive(Debug, Clone)]
pub struct AudioFeatureExtractor {
    sample_rate: u32,
    mel: MelConfig,
}

impl AudioFeatureExtractor {
    /// Create an extractor for audio at `sample_rate` with Whisper's mel settings.
    pub fn new(sample_rate: u32) -> Result<Self> {
        if sample_rate == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "AudioFeatureExtractor: sample rate must be non-zero".to_string(),
            ));
        }
        Ok(Self {
            sample_rate,
            mel: MelConfig::default(),
        })
    }

    /// The extractor's target sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Number of mel bands produced.
    pub fn n_mels(&self) -> usize {
        self.mel.n_mels
    }

    /// Decode an audio file from disk into mono PCM at `target_rate`.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::Io`] when the file cannot be read and
    /// [`TrustformersError::FeatureUnavailable`] for containers this crate
    /// cannot decode without an external codec.
    pub fn load_pcm(&self, path: &str, target_rate: u32) -> Result<Vec<f32>> {
        let bytes = std::fs::read(path).map_err(|e| TrustformersError::Io {
            message: format!("failed to read audio file: {e}"),
            path: Some(path.to_string()),
            suggestion: Some("Check that the file exists and is readable".to_string()),
        })?;

        if !audio_dsp::is_wav(&bytes) {
            let ext = Path::new(path)
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string());
            return Err(TrustformersError::feature_unavailable(
                format!(
                    "cannot decode `{path}` (extension `{ext}`): only uncompressed RIFF/WAVE is \
                     decodable without an external codec"
                ),
                "audio-codec",
            ));
        }

        let audio = audio_dsp::decode_wav(&bytes)?;
        audio_dsp::resample_linear(&audio.samples, audio.sample_rate, target_rate)
    }

    /// Decode an audio file and extract log-mel features from it.
    pub fn load_and_extract(&self, path: &str) -> Result<AudioFeatures> {
        let pcm = self.load_pcm(path, self.sample_rate)?;
        self.extract_features(&pcm)
    }

    /// Compute a Whisper-style log-mel spectrogram from mono PCM.
    pub fn extract_features(&self, samples: &[f32]) -> Result<AudioFeatures> {
        let duration_s = samples.len() as f64 / f64::from(self.sample_rate);
        let features = audio_dsp::log_mel_spectrogram(samples, self.sample_rate, self.mel)?;
        Ok(AudioFeatures {
            features,
            sample_rate: self.sample_rate,
            duration_s,
        })
    }

    /// Resample mono PCM between sample rates.
    pub fn resample(&self, samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
        audio_dsp::resample_linear(samples, from_rate, to_rate)
    }

    /// Decode a byte buffer in the declared container format.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::FeatureUnavailable`] for compressed
    /// containers (FLAC/MP3/M4A/Ogg/WebM), which need a codec that this
    /// workspace does not ship.
    pub fn decode_bytes(
        &self,
        data: &[u8],
        format: AudioFormat,
    ) -> Result<audio_dsp::DecodedAudio> {
        if !format.is_decodable() {
            return Err(TrustformersError::feature_unavailable(
                format!(
                    "{format:?} decoding is not implemented; only uncompressed WAV is decodable \
                     in pure Rust here"
                ),
                "audio-codec",
            ));
        }
        audio_dsp::decode_wav(data)
    }

    /// Decode a byte buffer and extract features from it.
    pub fn decode_and_extract(&self, data: &[u8], format: AudioFormat) -> Result<AudioFeatures> {
        let audio = self.decode_bytes(data, format)?;
        let pcm = audio_dsp::resample_linear(&audio.samples, audio.sample_rate, self.sample_rate)?;
        self.extract_features(&pcm)
    }
}

/// Log-mel features: `[time_frames][n_mels]`.
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    pub features: Vec<Vec<f32>>,
    pub sample_rate: u32,
    pub duration_s: f64,
}

impl AudioFeatures {
    pub fn duration(&self) -> f64 {
        self.duration_s
    }

    /// Number of mel bands per frame (0 when there are no frames).
    pub fn n_mels(&self) -> usize {
        self.features.first().map_or(0, Vec::len)
    }

    /// Convert to a `[1, frames, n_mels]` tensor.
    pub fn to_tensor(&self) -> Result<crate::core::tensor::Tensor> {
        use crate::core::tensor::Tensor;

        if self.features.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "AudioFeatures: no frames to convert".to_string(),
            ));
        }
        let n_mels = self.features[0].len();
        let flat: Vec<f32> = self.features.iter().flatten().copied().collect();
        let shape = vec![1, self.features.len(), n_mels];
        Tensor::from_vec(flat, &shape).map_err(Into::into)
    }

    /// Convert to the `[1, n_mels, frames]` layout Whisper's encoder expects.
    pub fn to_whisper_mel_tensor(&self) -> Result<crate::core::tensor::Tensor> {
        use crate::core::tensor::Tensor;

        if self.features.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "AudioFeatures: no frames to convert".to_string(),
            ));
        }
        let frames = self.features.len();
        let n_mels = self.features[0].len();
        let mut flat = vec![0.0f32; frames * n_mels];
        for (t, frame) in self.features.iter().enumerate() {
            if frame.len() != n_mels {
                return Err(TrustformersError::invalid_input_simple(
                    "AudioFeatures: ragged mel frames".to_string(),
                ));
            }
            for (m, &v) in frame.iter().enumerate() {
                flat[m * frames + t] = v;
            }
        }
        Tensor::from_vec(flat, &[1, n_mels, frames]).map_err(Into::into)
    }

    /// Reject a sample-rate change after feature extraction.
    ///
    /// Mel features cannot be resampled meaningfully once computed — the
    /// filterbank is tied to the waveform's rate. Resample the waveform before
    /// extraction instead. Returns `self` unchanged when the rate already
    /// matches, and an error otherwise.
    pub fn resample_to(self, target_rate: u32) -> Result<Self> {
        if self.sample_rate == target_rate {
            return Ok(self);
        }
        Err(TrustformersError::invalid_input_simple(format!(
            "AudioFeatures were computed at {} Hz and cannot be converted to {} Hz after the \
             fact; resample the waveform before extracting features",
            self.sample_rate, target_rate
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::media::audio_dsp::encode_wav_pcm16;

    fn tone(freq: f32, sample_rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                0.8 * (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin()
            })
            .collect()
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(name);
        p
    }

    // ---- AudioFormat tests ----

    #[test]
    fn test_audio_format_from_extension_wav() {
        let fmt = AudioFormat::from_extension("wav");
        assert!(matches!(fmt, Some(AudioFormat::Wav)));
    }

    #[test]
    fn test_audio_format_from_extension_flac() {
        let fmt = AudioFormat::from_extension("flac");
        assert!(matches!(fmt, Some(AudioFormat::Flac)));
    }

    #[test]
    fn test_audio_format_from_extension_mp3() {
        let fmt = AudioFormat::from_extension("mp3");
        assert!(matches!(fmt, Some(AudioFormat::Mp3)));
    }

    #[test]
    fn test_audio_format_from_extension_case_insensitive() {
        let fmt = AudioFormat::from_extension("WAV");
        assert!(matches!(fmt, Some(AudioFormat::Wav)));
    }

    #[test]
    fn test_audio_format_from_extension_unknown() {
        let fmt = AudioFormat::from_extension("xyz");
        assert!(fmt.is_none());
    }

    #[test]
    fn test_audio_format_all_variants() {
        let exts = ["wav", "flac", "mp3", "m4a", "ogg", "webm"];
        for ext in &exts {
            assert!(
                AudioFormat::from_extension(ext).is_some(),
                "missing: {}",
                ext
            );
        }
    }

    #[test]
    fn test_only_wav_is_decodable() {
        assert!(AudioFormat::Wav.is_decodable());
        for fmt in [
            AudioFormat::Flac,
            AudioFormat::Mp3,
            AudioFormat::M4a,
            AudioFormat::Ogg,
            AudioFormat::WebM,
        ] {
            assert!(!fmt.is_decodable(), "{fmt:?} must not claim decodability");
        }
    }

    // ---- SpeechToTextConfig tests ----

    #[test]
    fn test_config_default_values() {
        let cfg = SpeechToTextConfig::default();
        assert_eq!(cfg.sample_rate, 16000);
        assert_eq!(cfg.max_duration, Some(30.0));
        assert!(!cfg.return_timestamps);
        assert!(cfg.language.is_none());
        assert!(matches!(cfg.task, SpeechTask::Transcribe));
        assert_eq!(cfg.num_beams, 1);
        assert!((cfg.temperature - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_chunk_length() {
        let cfg = SpeechToTextConfig::default();
        assert_eq!(cfg.chunk_length_s, Some(30.0));
        assert_eq!(cfg.stride_length_s, Some(5.0));
    }

    // ---- AudioFeatureExtractor tests ----

    #[test]
    fn test_extractor_creates_successfully() {
        let extractor = AudioFeatureExtractor::new(16000).expect("extractor creation succeeded");
        assert_eq!(extractor.sample_rate(), 16000);
        assert_eq!(extractor.n_mels(), 80);
    }

    #[test]
    fn test_extractor_rejects_zero_sample_rate() {
        assert!(AudioFeatureExtractor::new(0).is_err());
    }

    #[test]
    fn test_extract_features_duration_calculation() {
        let extractor = AudioFeatureExtractor::new(16000).expect("ok");
        let samples = tone(300.0, 16000, 16000); // 1 second
        let features = extractor.extract_features(&samples).expect("ok");
        assert!((features.duration_s - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_extract_features_frame_count() {
        let extractor = AudioFeatureExtractor::new(16000).expect("ok");
        let samples = tone(300.0, 16000, 1600); // 0.1 seconds
        let features = extractor.extract_features(&samples).expect("ok");
        // Whisper convention: centred STFT with the final frame dropped →
        // len / hop = 1600 / 160 = 10 frames.
        assert_eq!(features.features.len(), 10);
    }

    #[test]
    fn test_extract_features_mel_dims() {
        let extractor = AudioFeatureExtractor::new(16000).expect("ok");
        let samples = tone(300.0, 16000, 3200);
        let features = extractor.extract_features(&samples).expect("ok");
        for frame in &features.features {
            assert_eq!(frame.len(), 80);
        }
    }

    #[test]
    fn test_extract_features_are_not_all_zero() {
        // Regression: `extract_features` used to return `vec![vec![0.0; 80]; n]`.
        let extractor = AudioFeatureExtractor::new(16000).expect("ok");
        let features = extractor.extract_features(&tone(440.0, 16000, 8000)).expect("ok");
        let flat: Vec<f32> = features.features.iter().flatten().copied().collect();
        assert!(
            flat.iter().any(|&v| v != 0.0),
            "mel features must not be all zeros"
        );
        let min = flat.iter().copied().fold(f32::INFINITY, f32::min);
        let max = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(max - min > 0.5, "mel features must vary: {min}..{max}");
    }

    #[test]
    fn test_extract_features_differ_between_tones() {
        let extractor = AudioFeatureExtractor::new(16000).expect("ok");
        let a = extractor.extract_features(&tone(220.0, 16000, 8000)).expect("a");
        let b = extractor.extract_features(&tone(3000.0, 16000, 8000)).expect("b");
        assert_ne!(
            a.features, b.features,
            "different tones must yield different features"
        );
    }

    // ---- Resampling tests ----

    #[test]
    fn test_resample_same_rate_no_op() {
        let extractor = AudioFeatureExtractor::new(16000).expect("ok");
        let samples = vec![0.1_f32, 0.2, 0.3];
        let resampled = extractor.resample(&samples, 16000, 16000).expect("ok");
        assert_eq!(resampled.len(), samples.len());
        for (a, b) in resampled.iter().zip(samples.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_resample_upsample_increases_length() {
        let extractor = AudioFeatureExtractor::new(16000).expect("ok");
        let samples = tone(100.0, 8000, 100);
        let resampled = extractor.resample(&samples, 8000, 16000).expect("ok");
        assert_eq!(resampled.len(), 200);
    }

    #[test]
    fn test_resample_downsample_decreases_length() {
        let extractor = AudioFeatureExtractor::new(16000).expect("ok");
        let samples = tone(100.0, 16000, 200);
        let resampled = extractor.resample(&samples, 16000, 8000).expect("ok");
        assert_eq!(resampled.len(), 100);
    }

    // ---- AudioFeatures tests ----

    #[test]
    fn test_audio_features_duration() {
        let af = AudioFeatures {
            features: vec![vec![0.0; 80]; 50],
            sample_rate: 16000,
            duration_s: 3.5,
        };
        assert!((af.duration() - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_audio_features_to_tensor_shape() {
        let n_frames = 10;
        let n_mels = 80;
        let af = AudioFeatures {
            features: vec![vec![0.1; n_mels]; n_frames],
            sample_rate: 16000,
            duration_s: 1.0,
        };
        let tensor = af.to_tensor().expect("tensor creation succeeded");
        let shape = tensor.shape();
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], n_frames);
        assert_eq!(shape[2], n_mels);
    }

    #[test]
    fn test_audio_features_whisper_layout_transposes() {
        let af = AudioFeatures {
            features: vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            sample_rate: 16000,
            duration_s: 1.0,
        };
        let tensor = af.to_whisper_mel_tensor().expect("tensor");
        assert_eq!(tensor.shape(), &[1, 2, 3]);
        let flat = tensor.to_vec_f32().expect("values");
        // mel-major: [1,3,5, 2,4,6]
        assert_eq!(flat, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_audio_features_resample_to_same_rate() {
        let af = AudioFeatures {
            features: vec![vec![0.0; 80]; 5],
            sample_rate: 16000,
            duration_s: 1.0,
        };
        let result = af.resample_to(16000).expect("ok");
        assert_eq!(result.sample_rate, 16000);
    }

    #[test]
    fn test_audio_features_resample_to_other_rate_errors() {
        // Regression: `resample_to` used to silently return `self` unchanged
        // while claiming to have resampled.
        let af = AudioFeatures {
            features: vec![vec![0.0; 80]; 5],
            sample_rate: 16000,
            duration_s: 1.0,
        };
        let err = af.resample_to(8000).expect_err("must not silently no-op");
        assert!(
            err.to_string().contains("cannot be converted"),
            "err: {err}"
        );
    }

    // ---- WordTimestamp tests ----

    #[test]
    fn test_word_timestamp_time_ordering() {
        let ts = WordTimestamp {
            word: "hello".to_string(),
            start_time: 0.0,
            end_time: 0.5,
            confidence: 0.95,
        };
        assert!(ts.start_time < ts.end_time);
        assert!(ts.confidence >= 0.0 && ts.confidence <= 1.0);
    }

    #[test]
    fn test_word_timestamp_confidence_range() {
        let ts = WordTimestamp {
            word: "world".to_string(),
            start_time: 0.5,
            end_time: 1.0,
            confidence: 0.87,
        };
        assert!(ts.confidence >= 0.0 && ts.confidence <= 1.0);
    }

    // ---- SpeechTask enum ----

    #[test]
    fn test_speech_task_transcribe_variant() {
        let task = SpeechTask::Transcribe;
        assert!(matches!(task, SpeechTask::Transcribe));
    }

    #[test]
    fn test_speech_task_translate_variant() {
        let task = SpeechTask::Translate;
        assert!(matches!(task, SpeechTask::Translate));
    }

    // ---- File / byte decoding ----

    #[test]
    fn test_load_and_extract_rejects_missing_file() {
        // Regression: `load_and_extract` used to return 100 zero frames and a
        // hardcoded 5-second duration for any path, including nonexistent ones.
        let extractor = AudioFeatureExtractor::new(16000).expect("ok");
        let err = extractor
            .load_and_extract("definitely-not-a-real-file.wav")
            .expect_err("missing file must error");
        assert!(matches!(err, TrustformersError::Io { .. }), "err: {err}");
    }

    #[test]
    fn test_load_and_extract_reads_a_real_wav() {
        let extractor = AudioFeatureExtractor::new(16000).expect("ok");
        let path = temp_path("trustformers-stt-load.wav");
        std::fs::write(&path, encode_wav_pcm16(&tone(500.0, 16000, 8000), 16000))
            .expect("write fixture");
        let features =
            extractor.load_and_extract(&path.to_string_lossy()).expect("decode real wav");
        assert_eq!(features.features.len(), 50);
        assert!(features.features.iter().flatten().any(|&v| v != 0.0));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_and_extract_rejects_compressed_container() {
        let extractor = AudioFeatureExtractor::new(16000).expect("ok");
        let path = temp_path("trustformers-stt-fake.mp3");
        std::fs::write(&path, b"ID3\x04\x00\x00\x00\x00\x00\x00").expect("write fixture");
        let err = extractor
            .load_and_extract(&path.to_string_lossy())
            .expect_err("mp3 must not be decoded as silence");
        assert!(matches!(err, TrustformersError::FeatureUnavailable { .. }));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_decode_and_extract_rejects_garbage_wav() {
        // Regression: `decode_and_extract` used to ignore `data` entirely and
        // extract features from `&[0.0; 16000]`.
        let extractor = AudioFeatureExtractor::new(16000).expect("ok");
        let err = extractor
            .decode_and_extract(&[0u8; 512], AudioFormat::Wav)
            .expect_err("512 zero bytes are not a WAV file");
        assert!(err.to_string().contains("RIFF"), "err: {err}");
    }

    #[test]
    fn test_decode_and_extract_accepts_real_wav() {
        let extractor = AudioFeatureExtractor::new(16000).expect("ok");
        let bytes = encode_wav_pcm16(&tone(700.0, 16000, 4800), 16000);
        let features = extractor
            .decode_and_extract(&bytes, AudioFormat::Wav)
            .expect("real wav decodes");
        assert_eq!(features.features.len(), 30);
        assert!(features.features.iter().flatten().any(|&v| v != 0.0));
    }

    #[test]
    fn test_decode_bytes_rejects_compressed_formats() {
        let extractor = AudioFeatureExtractor::new(16000).expect("ok");
        for fmt in [AudioFormat::Flac, AudioFormat::Mp3, AudioFormat::Ogg] {
            let err = extractor
                .decode_bytes(&[0u8; 16], fmt)
                .expect_err("compressed containers are not decodable");
            assert!(matches!(err, TrustformersError::FeatureUnavailable { .. }));
        }
    }

    #[test]
    fn test_decode_bytes_resamples_from_container_rate() {
        // The container's own 8 kHz rate must win over any caller hint.
        let extractor = AudioFeatureExtractor::new(16000).expect("ok");
        let bytes = encode_wav_pcm16(&tone(400.0, 8000, 4000), 8000);
        let audio = extractor.decode_bytes(&bytes, AudioFormat::Wav).expect("decode");
        assert_eq!(audio.sample_rate, 8000);
        let resampled = extractor.resample(&audio.samples, audio.sample_rate, 16000).expect("rs");
        assert_eq!(resampled.len(), 8000);
    }

    // ---- base64 ----

    #[test]
    fn test_base64_decoding_is_real() {
        // Regression: a private `mod base64` shadowed the real crate and
        // returned `Ok(vec![])` for every input.
        let wav = encode_wav_pcm16(&tone(440.0, 16000, 1600), 16000);
        let encoded = base64::engine::general_purpose::STANDARD.encode(&wav);
        let decoded =
            base64::engine::general_purpose::STANDARD.decode(&encoded).expect("round trip");
        assert_eq!(decoded, wav);
        assert!(!decoded.is_empty(), "real base64 decode must not be empty");
    }

    #[test]
    fn test_base64_invalid_payload_errors() {
        let err = base64::engine::general_purpose::STANDARD
            .decode("!!!not base64!!!")
            .expect_err("invalid base64 must error");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn test_backend_defaults_to_unavailable() {
        let backend = SpeechToTextBackend::Unavailable;
        assert_eq!(format!("{backend:?}"), "SpeechToTextBackend::Unavailable");
    }

    fn stub_pipeline() -> SpeechToTextPipeline {
        use trustformers_tokenizers::CharTokenizer;
        let mut vocab = std::collections::HashMap::new();
        vocab.insert(WHISPER_SOT_TOKEN.to_string(), 50_258_u32);
        let tokenizer = AutoTokenizer::Char(CharTokenizer::new(vocab));
        let model = AutoModel::from_config(crate::AutoConfig::Bert(Default::default()))
            .expect("bert config builds");
        SpeechToTextPipeline::new(model, tokenizer).expect("pipeline")
    }

    #[test]
    fn test_decoder_start_token_defaults_to_tokenizer_lookup() {
        // Regression: the whisper decode loop used to hardcode token id 0 as
        // `<|startoftranscript|>`, which is wrong for every real checkpoint.
        let pipeline = stub_pipeline();
        assert!(
            pipeline.decoder_start_token().is_none(),
            "no override is set by default; the id comes from the tokenizer"
        );
        assert_eq!(WHISPER_SOT_TOKEN, "<|startoftranscript|>");
    }

    #[test]
    fn test_decoder_start_token_override_is_stored() {
        let pipeline = stub_pipeline().with_decoder_start_token(50_258);
        assert_eq!(pipeline.decoder_start_token(), Some(50_258));
    }

    #[cfg(feature = "whisper")]
    #[test]
    fn test_resolve_decoder_start_token_uses_the_vocabulary() {
        let pipeline = stub_pipeline();
        assert_eq!(
            pipeline.resolve_decoder_start_token().expect("token present"),
            50_258
        );
    }

    #[cfg(feature = "whisper")]
    #[test]
    fn test_resolve_decoder_start_token_errors_without_the_special_token() {
        use trustformers_tokenizers::CharTokenizer;
        let tokenizer = AutoTokenizer::Char(CharTokenizer::new(std::collections::HashMap::new()));
        let model = AutoModel::from_config(crate::AutoConfig::Bert(Default::default()))
            .expect("bert config builds");
        let pipeline = SpeechToTextPipeline::new(model, tokenizer).expect("pipeline");
        let err = pipeline.resolve_decoder_start_token().expect_err("the id must not be guessed");
        assert!(err.to_string().contains(WHISPER_SOT_TOKEN), "err: {err}");
    }

    #[test]
    fn test_unsupported_backend_error_lists_whisper() {
        let err = crate::pipeline::media::unsupported_model(
            "speech-to-text",
            "AutoModel (no speech architecture attached)",
            SUPPORTED_ARCHITECTURES,
        );
        match err {
            TrustformersError::FeatureUnavailable { alternatives, .. } => {
                assert!(alternatives.iter().any(|a| a.contains("whisper")));
            },
            other => panic!("expected FeatureUnavailable, got {other:?}"),
        }
    }
}
