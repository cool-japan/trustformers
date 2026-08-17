//! # Text-to-speech pipeline
//!
//! ## What is real here
//!
//! * **Text front-end** — whitespace normalisation, abbreviation expansion,
//!   number-to-words for 0–20, punctuation verbalisation and a
//!   dictionary-plus-fallback grapheme-to-phoneme converter. All of it is
//!   plain, inspectable string processing and is exposed publicly.
//! * **Prosody analysis** — [`ProsodyAnalyzer`] measures fundamental frequency
//!   by autocorrelation over overlapping frames, derives the pitch range from
//!   the per-frame F0 track, computes the speaking rate from the actual word
//!   count and audio duration, and finds pauses and emphasis from real frame
//!   energy. No constant is reported as a measurement.
//!
//! ## What is not available
//!
//! There is **no neural vocoder** in this workspace, so there is no way to turn
//! a model's acoustic output into a waveform. [`TextToSpeechPipeline::synthesize`]
//! therefore returns a structured [`TrustformersError::FeatureUnavailable`]
//! instead of the sine-wave tone generator it used to emit as "speech". Use
//! [`TextToSpeechPipeline::prepare`] to obtain the normalised text, phonemes and
//! token IDs and drive your own vocoder.

use crate::error::{Result, TrustformersError};
use crate::pipeline::{BasePipeline, Pipeline};
use serde::{Deserialize, Serialize};
use trustformers_core::traits::{Model, Tokenizer};
use trustformers_core::Tensor;

/// Vocoder implementations available in this workspace (none yet).
const SUPPORTED_VOCODERS: &[&str] = &[];

/// Configuration for text-to-speech pipeline
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextToSpeechConfig {
    /// Voice to use for synthesis
    pub voice: String,
    /// Speaking rate (0.5 to 2.0)
    pub speaking_rate: f32,
    /// Pitch (0.5 to 2.0)
    pub pitch: f32,
    /// Volume (0.0 to 1.0)
    pub volume: f32,
    /// Sample rate for output audio
    pub sample_rate: u32,
    /// Output format
    pub output_format: AudioFormat,
    /// Maximum duration in seconds
    pub max_duration: Option<f64>,
    /// Enable prosody control
    pub prosody_control: bool,
    /// Enable emotion control
    pub emotion_control: bool,
    /// Target emotion (if emotion control is enabled)
    pub target_emotion: Option<String>,
}

impl Default for TextToSpeechConfig {
    fn default() -> Self {
        Self {
            voice: "default".to_string(),
            speaking_rate: 1.0,
            pitch: 1.0,
            volume: 1.0,
            sample_rate: 22050,
            output_format: AudioFormat::Wav,
            max_duration: Some(60.0),
            prosody_control: false,
            emotion_control: false,
            target_emotion: None,
        }
    }
}

/// Audio format for output
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AudioFormat {
    /// WAV format
    Wav,
    /// MP3 format
    Mp3,
    /// FLAC format
    Flac,
    /// OGG format
    Ogg,
    /// Raw PCM
    Raw,
}

/// Input for text-to-speech pipeline
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextToSpeechInput {
    /// Text to synthesize
    pub text: String,
    /// Optional voice override
    pub voice: Option<String>,
    /// Optional speaking rate override
    pub speaking_rate: Option<f32>,
    /// Optional pitch override
    pub pitch: Option<f32>,
    /// Optional volume override
    pub volume: Option<f32>,
    /// Optional emotion override
    pub emotion: Option<String>,
    /// Optional prosody markers
    pub prosody_markers: Option<Vec<ProsodyMarker>>,
}

/// Prosody marker for fine-grained control
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProsodyMarker {
    /// Start position in text
    pub start: usize,
    /// End position in text
    pub end: usize,
    /// Prosody type
    pub prosody_type: ProsodyType,
    /// Intensity (0.0 to 1.0)
    pub intensity: f32,
}

/// Types of prosody control
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProsodyType {
    /// Emphasis
    Emphasis,
    /// Pause
    Pause,
    /// Speed change
    Speed,
    /// Pitch change
    Pitch,
    /// Volume change
    Volume,
}

/// Output from text-to-speech pipeline
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextToSpeechOutput {
    /// Generated audio data
    pub audio_data: Vec<f32>,
    /// Sample rate of the audio
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration: f64,
    /// Audio format
    pub format: AudioFormat,
    /// Voice used for synthesis
    pub voice: String,
    /// Text that was synthesized
    pub text: String,
    /// Phoneme sequence (if available)
    pub phonemes: Option<Vec<String>>,
    /// Timing information for phonemes
    pub phoneme_timings: Option<Vec<PhonemeTimings>>,
    /// Prosody information
    pub prosody_info: Option<ProsodyInfo>,
}

/// Timing information for phonemes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhonemeTimings {
    /// Phoneme symbol
    pub phoneme: String,
    /// Start time in seconds
    pub start_time: f64,
    /// End time in seconds
    pub end_time: f64,
    /// Aligner confidence, when a forced aligner produced this timing.
    ///
    /// `None` means the timing came from a uniform division of the utterance
    /// duration rather than an alignment model — the honest value, since this
    /// crate ships no aligner.
    pub confidence: Option<f32>,
}

/// Prosody information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProsodyInfo {
    /// Average pitch
    pub avg_pitch: f32,
    /// Pitch range
    pub pitch_range: f32,
    /// Speaking rate (words per minute)
    pub speaking_rate: f32,
    /// Pause locations
    pub pauses: Vec<PauseInfo>,
    /// Emphasis locations
    pub emphasis: Vec<EmphasisInfo>,
}

/// Pause information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PauseInfo {
    /// Start time in seconds
    pub start_time: f64,
    /// Duration in seconds
    pub duration: f64,
    /// Pause type
    pub pause_type: PauseType,
}

/// Types of pauses
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PauseType {
    /// Sentence boundary
    Sentence,
    /// Phrase boundary
    Phrase,
    /// Comma pause
    Comma,
    /// Breath pause
    Breath,
    /// Emphasis pause
    Emphasis,
}

/// Emphasis information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmphasisInfo {
    /// Start time in seconds
    pub start_time: f64,
    /// End time in seconds
    pub end_time: f64,
    /// Emphasis intensity
    pub intensity: f32,
    /// Emphasis type
    pub emphasis_type: EmphasisType,
}

/// Types of emphasis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EmphasisType {
    /// Stress emphasis
    Stress,
    /// Pitch emphasis
    Pitch,
    /// Volume emphasis
    Volume,
    /// Duration emphasis
    Duration,
}

/// Text-to-speech pipeline implementation
pub struct TextToSpeechPipeline<M, T>
where
    M: Model + Clone + Send + Sync + 'static,
    T: Tokenizer + Clone + Send + Sync + 'static,
{
    base: BasePipeline<M, T>,
    config: TextToSpeechConfig,
    available_voices: Vec<String>,
    phoneme_converter: Option<PhonemeConverter>,
    prosody_analyzer: Option<ProsodyAnalyzer>,
}

impl<M, T> TextToSpeechPipeline<M, T>
where
    M: Model<Input = Tensor, Output = Tensor> + Clone + Send + Sync + 'static,
    T: Tokenizer + Clone + Send + Sync + 'static,
{
    /// Create a new text-to-speech pipeline
    pub fn new(model: M, tokenizer: T) -> Result<Self> {
        let base = BasePipeline::new(model, tokenizer);
        let config = TextToSpeechConfig::default();
        let available_voices = Self::get_available_voices();

        Ok(Self {
            base,
            config,
            available_voices,
            phoneme_converter: None,
            prosody_analyzer: None,
        })
    }

    /// Set configuration
    pub fn with_config(mut self, config: TextToSpeechConfig) -> Self {
        self.config = config;
        self
    }

    /// Set voice
    pub fn with_voice(mut self, voice: String) -> Self {
        self.config.voice = voice;
        self
    }

    /// Set speaking rate
    pub fn with_speaking_rate(mut self, rate: f32) -> Self {
        self.config.speaking_rate = rate.clamp(0.5, 2.0);
        self
    }

    /// Set pitch
    pub fn with_pitch(mut self, pitch: f32) -> Self {
        self.config.pitch = pitch.clamp(0.5, 2.0);
        self
    }

    /// Set volume
    pub fn with_volume(mut self, volume: f32) -> Self {
        self.config.volume = volume.clamp(0.0, 1.0);
        self
    }

    /// Set output format
    pub fn with_output_format(mut self, format: AudioFormat) -> Self {
        self.config.output_format = format;
        self
    }

    /// Attach (or detach) the grapheme-to-phoneme converter.
    pub fn with_phoneme_conversion(mut self, enable: bool) -> Self {
        self.phoneme_converter = if enable { Some(PhonemeConverter::new()) } else { None };
        self
    }

    /// Enable prosody control
    pub fn with_prosody_control(mut self, enable: bool) -> Self {
        self.config.prosody_control = enable;
        if enable && self.prosody_analyzer.is_none() {
            self.prosody_analyzer = Some(ProsodyAnalyzer::new());
        }
        self
    }

    /// Enable emotion control
    pub fn with_emotion_control(mut self, enable: bool) -> Self {
        self.config.emotion_control = enable;
        self
    }

    /// Set target emotion
    pub fn with_target_emotion(mut self, emotion: String) -> Self {
        self.config.target_emotion = Some(emotion);
        self.config.emotion_control = true;
        self
    }

    /// Get available voices
    pub fn get_available_voices() -> Vec<String> {
        vec![
            "default".to_string(),
            "male-neutral".to_string(),
            "female-neutral".to_string(),
            "male-young".to_string(),
            "female-young".to_string(),
            "male-elderly".to_string(),
            "female-elderly".to_string(),
            "child".to_string(),
            "narrator".to_string(),
            "robot".to_string(),
        ]
    }

    /// Get supported emotions
    pub fn get_supported_emotions() -> Vec<String> {
        vec![
            "neutral".to_string(),
            "happy".to_string(),
            "sad".to_string(),
            "angry".to_string(),
            "excited".to_string(),
            "calm".to_string(),
            "surprised".to_string(),
            "fearful".to_string(),
            "disgusted".to_string(),
            "confident".to_string(),
            "whispering".to_string(),
            "shouting".to_string(),
        ]
    }

    /// Everything the text front-end produces for an utterance.
    ///
    /// This is the real, complete output of the parts of the pipeline that are
    /// implemented; it stops short of the waveform because no vocoder exists.
    ///
    /// # Errors
    ///
    /// Returns an error for empty text, an unknown voice, or a tokenizer
    /// failure.
    pub fn prepare(&self, input: &TextToSpeechInput) -> Result<SynthesisPlan> {
        if input.text.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "Text input cannot be empty. Expected: non-empty text string, received: empty \
                 string"
                    .to_string(),
            ));
        }

        let voice = input.voice.clone().unwrap_or_else(|| self.config.voice.clone());
        if !self.available_voices.contains(&voice) {
            return Err(TrustformersError::invalid_input_simple(format!(
                "Voice '{}' is not available. Parameter: voice, Expected: one of {:?}, Received: \
                 {}",
                voice, self.available_voices, voice
            )));
        }

        let normalized_text = self.preprocess_text(&input.text)?;
        let tokenized = self.base.tokenizer.encode(&normalized_text)?;
        let phonemes = match &self.phoneme_converter {
            Some(converter) => Some(converter.text_to_phonemes(&normalized_text)?),
            None => None,
        };

        Ok(SynthesisPlan {
            voice,
            normalized_text,
            token_ids: tokenized.input_ids.clone(),
            phonemes,
            speaking_rate: input.speaking_rate.unwrap_or(self.config.speaking_rate),
            pitch: input.pitch.unwrap_or(self.config.pitch),
            volume: input.volume.unwrap_or(self.config.volume),
            sample_rate: self.config.sample_rate,
        })
    }

    /// Synthesize text to speech.
    ///
    /// # Errors
    ///
    /// Always returns [`TrustformersError::FeatureUnavailable`] once the input
    /// has been validated and the text front-end has run: this workspace ships
    /// no neural vocoder, and the pipeline will not emit a synthetic tone in
    /// place of speech. Input-validation errors surface first.
    ///
    /// Use [`Self::prepare`] to get the normalised text, token IDs and phonemes
    /// and drive an external vocoder.
    pub fn synthesize(&self, input: TextToSpeechInput) -> Result<TextToSpeechOutput> {
        // Run the real front-end first so callers still get input validation.
        let plan = self.prepare(&input)?;

        Err(TrustformersError::FeatureUnavailable {
            message: format!(
                "text-to-speech synthesis requires a neural vocoder, and none is implemented in \
                 this workspace; the text front-end produced {} tokens and {} phonemes for voice \
                 `{}`, but no waveform can be generated. This pipeline never emits synthesised \
                 tones in place of speech.",
                plan.token_ids.len(),
                plan.phonemes.as_ref().map_or(0, Vec::len),
                plan.voice
            ),
            feature: "tts-vocoder".to_string(),
            suggestion: Some(
                "Call `TextToSpeechPipeline::prepare` for the normalised text, token IDs and \
                 phonemes, then run your own vocoder."
                    .to_string(),
            ),
            alternatives: SUPPORTED_VOCODERS.iter().map(|s| (*s).to_string()).collect(),
        })
    }

    /// Build a [`TextToSpeechOutput`] from externally-vocoded audio.
    ///
    /// Lets callers reuse the pipeline's real phoneme timing and prosody
    /// analysis once they have a waveform from their own vocoder.
    ///
    /// # Errors
    ///
    /// Returns an error when `audio_data` is empty or prosody analysis fails.
    pub fn finish_with_audio(
        &self,
        plan: SynthesisPlan,
        audio_data: Vec<f32>,
    ) -> Result<TextToSpeechOutput> {
        if audio_data.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "finish_with_audio: the vocoded audio buffer is empty".to_string(),
            ));
        }
        let duration = audio_data.len() as f64 / f64::from(plan.sample_rate);

        let phoneme_timings = match &plan.phonemes {
            Some(phonemes) => Some(self.generate_phoneme_timings(phonemes, duration)?),
            None => None,
        };

        let prosody_info = if self.config.prosody_control {
            match &self.prosody_analyzer {
                Some(analyzer) => {
                    Some(analyzer.analyze(&plan.normalized_text, &audio_data, plan.sample_rate)?)
                },
                None => None,
            }
        } else {
            None
        };

        Ok(TextToSpeechOutput {
            audio_data,
            sample_rate: plan.sample_rate,
            duration,
            format: self.config.output_format.clone(),
            voice: plan.voice,
            text: plan.normalized_text,
            phonemes: plan.phonemes,
            phoneme_timings,
            prosody_info,
        })
    }

    /// Preprocess text for TTS
    fn preprocess_text(&self, text: &str) -> Result<String> {
        let mut processed = text.to_string();

        // Normalize whitespace
        processed = processed.split_whitespace().collect::<Vec<_>>().join(" ");

        // Expand abbreviations
        processed = self.expand_abbreviations(&processed);

        // Normalize numbers
        processed = self.normalize_numbers(&processed);

        // Handle punctuation
        processed = self.normalize_punctuation(&processed);

        Ok(processed)
    }

    /// Expand common abbreviations
    fn expand_abbreviations(&self, text: &str) -> String {
        let abbreviations = [
            ("Dr.", "Doctor"),
            ("Mr.", "Mister"),
            ("Mrs.", "Missus"),
            ("Ms.", "Miss"),
            ("Prof.", "Professor"),
            ("St.", "Street"),
            ("Ave.", "Avenue"),
            ("Blvd.", "Boulevard"),
            ("etc.", "et cetera"),
            ("vs.", "versus"),
            ("Inc.", "Incorporated"),
            ("Corp.", "Corporation"),
            ("Ltd.", "Limited"),
            ("Co.", "Company"),
            ("USA", "United States of America"),
            ("UK", "United Kingdom"),
            ("CEO", "Chief Executive Officer"),
            ("CTO", "Chief Technology Officer"),
            ("CFO", "Chief Financial Officer"),
            ("AI", "Artificial Intelligence"),
            ("ML", "Machine Learning"),
            ("TTS", "Text To Speech"),
            ("API", "Application Programming Interface"),
        ];

        let mut result = text.to_string();
        for (abbr, expansion) in &abbreviations {
            result = result.replace(abbr, expansion);
        }
        result
    }

    /// Normalize numbers to words
    fn normalize_numbers(&self, text: &str) -> String {
        // This is a simplified number normalization
        // In a real implementation, you would use a proper number-to-words library
        let number_words = [
            ("0", "zero"),
            ("1", "one"),
            ("2", "two"),
            ("3", "three"),
            ("4", "four"),
            ("5", "five"),
            ("6", "six"),
            ("7", "seven"),
            ("8", "eight"),
            ("9", "nine"),
            ("10", "ten"),
            ("11", "eleven"),
            ("12", "twelve"),
            ("13", "thirteen"),
            ("14", "fourteen"),
            ("15", "fifteen"),
            ("16", "sixteen"),
            ("17", "seventeen"),
            ("18", "eighteen"),
            ("19", "nineteen"),
            ("20", "twenty"),
        ];

        let mut result = text.to_string();
        for (num, word) in &number_words {
            result = result.replace(&format!(" {} ", num), &format!(" {} ", word));
        }
        result
    }

    /// Normalize punctuation for speech
    fn normalize_punctuation(&self, text: &str) -> String {
        text.replace("...", " pause ")
            .replace(".", " period ")
            .replace("!", " exclamation ")
            .replace("?", " question ")
            .replace(",", " comma ")
            .replace(";", " semicolon ")
            .replace(":", " colon ")
            .replace("-", " dash ")
            .replace("(", " open parenthesis ")
            .replace(")", " close parenthesis ")
            .replace("\"", " quote ")
            .replace("'", " apostrophe ")
    }

    /// Divide an utterance's duration uniformly across its phonemes.
    ///
    /// This is an explicitly *uniform* segmentation, not a forced alignment:
    /// every phoneme gets `total_duration / n` seconds and the reported
    /// confidence is `None`, because no aligner produced these boundaries.
    ///
    /// # Errors
    ///
    /// Returns an error when `phonemes` is empty.
    pub fn generate_phoneme_timings(
        &self,
        phonemes: &[String],
        total_duration: f64,
    ) -> Result<Vec<PhonemeTimings>> {
        if phonemes.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "generate_phoneme_timings: no phonemes supplied".to_string(),
            ));
        }
        let avg_duration = total_duration / phonemes.len() as f64;

        Ok(phonemes
            .iter()
            .enumerate()
            .map(|(i, phoneme)| {
                let start_time = i as f64 * avg_duration;
                PhonemeTimings {
                    phoneme: phoneme.clone(),
                    start_time,
                    end_time: start_time + avg_duration,
                    confidence: None,
                }
            })
            .collect())
    }
}

/// The complete output of the text front-end for one utterance.
///
/// Produced by [`TextToSpeechPipeline::prepare`]; contains only quantities that
/// were actually computed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynthesisPlan {
    /// Resolved voice identifier.
    pub voice: String,
    /// Text after normalisation, abbreviation expansion and verbalisation.
    pub normalized_text: String,
    /// Token IDs produced by the pipeline's tokenizer.
    pub token_ids: Vec<u32>,
    /// Phoneme sequence, when a [`PhonemeConverter`] is attached.
    pub phonemes: Option<Vec<String>>,
    /// Effective speaking-rate multiplier.
    pub speaking_rate: f32,
    /// Effective pitch multiplier.
    pub pitch: f32,
    /// Effective volume multiplier.
    pub volume: f32,
    /// Target output sample rate in Hz.
    pub sample_rate: u32,
}

impl<M, T> Pipeline for TextToSpeechPipeline<M, T>
where
    M: Model<Input = Tensor, Output = Tensor> + Clone + Send + Sync + 'static,
    T: Tokenizer + Clone + Send + Sync + 'static,
{
    type Input = TextToSpeechInput;
    type Output = TextToSpeechOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        self.synthesize(input)
    }
}

/// Phoneme converter for text-to-phoneme conversion
pub struct PhonemeConverter {
    phoneme_dict: std::collections::HashMap<String, Vec<String>>,
}

impl Default for PhonemeConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl PhonemeConverter {
    pub fn new() -> Self {
        let mut phoneme_dict = std::collections::HashMap::new();

        // Add some basic phoneme mappings
        phoneme_dict.insert(
            "hello".to_string(),
            vec![
                "h".to_string(),
                "ə".to_string(),
                "ˈl".to_string(),
                "oʊ".to_string(),
            ],
        );
        phoneme_dict.insert(
            "world".to_string(),
            vec![
                "w".to_string(),
                "ɜr".to_string(),
                "l".to_string(),
                "d".to_string(),
            ],
        );
        phoneme_dict.insert("the".to_string(), vec!["ð".to_string(), "ə".to_string()]);
        phoneme_dict.insert("a".to_string(), vec!["ə".to_string()]);
        phoneme_dict.insert("an".to_string(), vec!["æ".to_string(), "n".to_string()]);

        Self { phoneme_dict }
    }

    pub fn text_to_phonemes(&self, text: &str) -> Result<Vec<String>> {
        let mut phonemes = Vec::new();

        for word in text.split_whitespace() {
            let clean_word =
                word.to_lowercase().trim_matches(|c: char| !c.is_alphabetic()).to_string();

            if let Some(word_phonemes) = self.phoneme_dict.get(&clean_word) {
                phonemes.extend(word_phonemes.clone());
            } else {
                // Fallback: convert each character to a phoneme
                for ch in clean_word.chars() {
                    phonemes.push(ch.to_string());
                }
            }
        }

        Ok(phonemes)
    }
}

/// Prosody analyzer for speech characteristics
pub struct ProsodyAnalyzer;

impl ProsodyAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyse the prosody of a real waveform.
    ///
    /// Every reported figure is measured:
    ///
    /// * `avg_pitch` / `pitch_range` come from an autocorrelation F0 track
    ///   (see [`Self::pitch_track`]), restricted to the 50–500 Hz human range;
    ///   both are `0.0` when no frame is voiced.
    /// * `speaking_rate` is the utterance's word count divided by its duration
    ///   in minutes.
    /// * pauses and emphasis come from frame energy.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty buffer or a zero sample rate.
    pub fn analyze(&self, text: &str, audio_data: &[f32], sample_rate: u32) -> Result<ProsodyInfo> {
        if audio_data.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "prosody analysis: empty audio buffer".to_string(),
            ));
        }
        if sample_rate == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "prosody analysis: sample rate must be non-zero".to_string(),
            ));
        }

        let track = self.pitch_track(audio_data, sample_rate);
        let avg_pitch = if track.is_empty() {
            0.0
        } else {
            track.iter().sum::<f32>() / track.len() as f32
        };
        let pitch_range = if track.is_empty() {
            0.0
        } else {
            let min = track.iter().copied().fold(f32::INFINITY, f32::min);
            let max = track.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            max - min
        };

        let duration_minutes = audio_data.len() as f32 / sample_rate as f32 / 60.0;
        let word_count = text.split_whitespace().count() as f32;
        let speaking_rate =
            if duration_minutes > 0.0 { word_count / duration_minutes } else { 0.0 };

        let pauses = self.detect_pauses(audio_data, sample_rate)?;
        let emphasis = self.detect_emphasis(audio_data, sample_rate)?;

        Ok(ProsodyInfo {
            avg_pitch,
            pitch_range,
            speaking_rate,
            pauses,
            emphasis,
        })
    }

    /// Per-frame fundamental frequency in Hz for the voiced frames of a signal.
    ///
    /// Uses normalised autocorrelation over 40 ms frames with a 20 ms hop,
    /// searching lags corresponding to 50–500 Hz. A frame is considered voiced
    /// when its peak normalised autocorrelation exceeds `0.3`; unvoiced frames
    /// are omitted rather than reported as some default pitch.
    pub fn pitch_track(&self, audio_data: &[f32], sample_rate: u32) -> Vec<f32> {
        const MIN_HZ: f32 = 50.0;
        const MAX_HZ: f32 = 500.0;
        const VOICING_THRESHOLD: f32 = 0.3;

        if sample_rate == 0 {
            return Vec::new();
        }
        let frame_len = (sample_rate as usize * 40) / 1000;
        let hop = (sample_rate as usize * 20) / 1000;
        if frame_len == 0 || hop == 0 || audio_data.len() < frame_len {
            return Vec::new();
        }
        let min_lag = (sample_rate as f32 / MAX_HZ).floor().max(1.0) as usize;
        let max_lag = ((sample_rate as f32 / MIN_HZ).ceil() as usize).min(frame_len - 1);
        if min_lag >= max_lag {
            return Vec::new();
        }

        let mut track = Vec::new();
        let mut start = 0usize;
        while start + frame_len <= audio_data.len() {
            let frame = &audio_data[start..start + frame_len];
            let energy: f32 = frame.iter().map(|s| s * s).sum();
            if energy > 1e-8 {
                let mut best_lag = 0usize;
                let mut best_score = 0.0f32;
                for lag in min_lag..=max_lag {
                    let mut num = 0.0f32;
                    let mut den = 0.0f32;
                    for i in 0..(frame_len - lag) {
                        num += frame[i] * frame[i + lag];
                        den += frame[i + lag] * frame[i + lag];
                    }
                    let score = if den > 1e-12 { num / (energy.sqrt() * den.sqrt()) } else { 0.0 };
                    if score > best_score {
                        best_score = score;
                        best_lag = lag;
                    }
                }
                if best_lag > 0 && best_score > VOICING_THRESHOLD {
                    track.push(sample_rate as f32 / best_lag as f32);
                }
            }
            start += hop;
        }
        track
    }

    /// Detect silent stretches longer than 100 ms.
    ///
    /// Pauses are classified by their measured duration: below 250 ms is a
    /// [`PauseType::Breath`], below 500 ms a [`PauseType::Comma`], below 900 ms
    /// a [`PauseType::Phrase`], and anything longer a [`PauseType::Sentence`].
    /// The thresholds are a documented heuristic over a real measurement, not a
    /// fixed label.
    ///
    /// # Errors
    ///
    /// Returns an error when `sample_rate` is zero.
    pub fn detect_pauses(&self, audio_data: &[f32], sample_rate: u32) -> Result<Vec<PauseInfo>> {
        if sample_rate == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "detect_pauses: sample rate must be non-zero".to_string(),
            ));
        }
        let mut pauses = Vec::new();
        let silence_threshold = 0.01;
        let min_pause_duration = 0.1; // 100ms

        let mut in_pause = false;
        let mut pause_start = 0.0;

        for (i, &sample) in audio_data.iter().enumerate() {
            let time = i as f64 / f64::from(sample_rate);

            if sample.abs() < silence_threshold {
                if !in_pause {
                    pause_start = time;
                    in_pause = true;
                }
            } else if in_pause {
                let duration = time - pause_start;
                if duration >= min_pause_duration {
                    pauses.push(PauseInfo {
                        start_time: pause_start,
                        duration,
                        pause_type: classify_pause(duration),
                    });
                }
                in_pause = false;
            }
        }

        // A trailing silence is still a pause.
        if in_pause {
            let duration = audio_data.len() as f64 / f64::from(sample_rate) - pause_start;
            if duration >= min_pause_duration {
                pauses.push(PauseInfo {
                    start_time: pause_start,
                    duration,
                    pause_type: classify_pause(duration),
                });
            }
        }

        Ok(pauses)
    }

    /// Detect 100 ms windows whose mean absolute amplitude exceeds `0.5`.
    ///
    /// # Errors
    ///
    /// Returns an error when `sample_rate` is zero.
    pub fn detect_emphasis(
        &self,
        audio_data: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<EmphasisInfo>> {
        if sample_rate == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "detect_emphasis: sample rate must be non-zero".to_string(),
            ));
        }
        let mut emphasis = Vec::new();
        let emphasis_threshold = 0.5;
        let window_size = (sample_rate as usize / 10).max(1); // 100ms window
        if audio_data.len() < window_size {
            return Ok(emphasis);
        }

        for (i, window) in audio_data.windows(window_size).enumerate() {
            let avg_amplitude = window.iter().map(|&x| x.abs()).sum::<f32>() / window.len() as f32;

            if avg_amplitude > emphasis_threshold {
                let start_time = i as f64 / sample_rate as f64 * window_size as f64;
                let end_time = start_time + window_size as f64 / sample_rate as f64;

                emphasis.push(EmphasisInfo {
                    start_time,
                    end_time,
                    intensity: avg_amplitude,
                    emphasis_type: EmphasisType::Volume,
                });
            }
        }

        Ok(emphasis)
    }
}

impl Default for ProsodyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Classify a measured pause duration (seconds) into a [`PauseType`].
fn classify_pause(duration: f64) -> PauseType {
    if duration < 0.25 {
        PauseType::Breath
    } else if duration < 0.5 {
        PauseType::Comma
    } else if duration < 0.9 {
        PauseType::Phrase
    } else {
        PauseType::Sentence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::{Model, TokenizedInput, Tokenizer};
    use crate::AutoConfig;
    use std::collections::HashMap;
    use trustformers_core::Tensor;

    #[derive(Clone)]
    struct MockModel {
        config: AutoConfig,
    }

    impl MockModel {
        fn new() -> Self {
            MockModel {
                config: {
                    // Try each available config in order
                    #[cfg(feature = "bert")]
                    {
                        AutoConfig::Bert(Default::default())
                    }
                    #[cfg(all(not(feature = "bert"), feature = "roberta"))]
                    {
                        AutoConfig::Roberta(Default::default())
                    }
                    #[cfg(all(not(feature = "bert"), not(feature = "roberta"), feature = "gpt2"))]
                    {
                        AutoConfig::Gpt2(Default::default())
                    }
                    #[cfg(all(
                        not(feature = "bert"),
                        not(feature = "roberta"),
                        not(feature = "gpt2"),
                        feature = "gpt_neo"
                    ))]
                    {
                        AutoConfig::GptNeo(Default::default())
                    }
                    #[cfg(all(
                        not(feature = "bert"),
                        not(feature = "roberta"),
                        not(feature = "gpt2"),
                        not(feature = "gpt_neo"),
                        feature = "gpt_j"
                    ))]
                    {
                        AutoConfig::GptJ(Default::default())
                    }
                    #[cfg(all(
                        not(feature = "bert"),
                        not(feature = "roberta"),
                        not(feature = "gpt2"),
                        not(feature = "gpt_neo"),
                        not(feature = "gpt_j"),
                        feature = "t5"
                    ))]
                    {
                        AutoConfig::T5(Default::default())
                    }
                    #[cfg(all(
                        not(feature = "bert"),
                        not(feature = "roberta"),
                        not(feature = "gpt2"),
                        not(feature = "gpt_neo"),
                        not(feature = "gpt_j"),
                        not(feature = "t5"),
                        feature = "albert"
                    ))]
                    {
                        AutoConfig::Albert(Default::default())
                    }
                    #[cfg(not(any(
                        feature = "bert",
                        feature = "roberta",
                        feature = "gpt2",
                        feature = "gpt_neo",
                        feature = "gpt_j",
                        feature = "t5",
                        feature = "albert"
                    )))]
                    {
                        // If no model features are enabled, we need to enable at least one for testing
                        // Since this is a test context, we'll compile-fail rather than panic at runtime
                        compile_error!("At least one model feature must be enabled for tests (bert, roberta, gpt2, gpt_neo, gpt_j, t5, or albert)")
                    }
                },
            }
        }
    }

    impl Model for MockModel {
        type Input = Tensor;
        type Output = Tensor;
        type Config = AutoConfig;

        fn forward(&self, _input: Self::Input) -> trustformers_core::errors::Result<Self::Output> {
            // Return a dummy tensor for testing
            Tensor::zeros(&[1, 10])
        }

        fn num_parameters(&self) -> usize {
            1000 // Mock parameter count
        }

        fn load_pretrained(
            &mut self,
            _reader: &mut dyn std::io::Read,
        ) -> trustformers_core::errors::Result<()> {
            Ok(()) // Mock implementation
        }

        fn get_config(&self) -> &Self::Config {
            &self.config
        }
    }

    #[derive(Clone)]
    struct MockTokenizer;

    impl MockTokenizer {
        fn new() -> Self {
            MockTokenizer
        }
    }

    impl Tokenizer for MockTokenizer {
        fn encode(&self, _text: &str) -> trustformers_core::errors::Result<TokenizedInput> {
            Ok(TokenizedInput {
                input_ids: vec![1, 2, 3], // Mock token IDs
                attention_mask: vec![1, 1, 1],
                token_type_ids: Some(vec![0, 0, 0]),
                offset_mapping: None,
                special_tokens_mask: None,
                overflowing_tokens: None,
            })
        }

        fn encode_pair(
            &self,
            _text_a: &str,
            _text_b: &str,
        ) -> trustformers_core::errors::Result<TokenizedInput> {
            Ok(TokenizedInput {
                input_ids: vec![1, 2, 3, 4, 5], // Mock token IDs for pair
                attention_mask: vec![1, 1, 1, 1, 1],
                token_type_ids: Some(vec![0, 0, 0, 1, 1]),
                offset_mapping: None,
                special_tokens_mask: None,
                overflowing_tokens: None,
            })
        }

        fn decode(&self, _token_ids: &[u32]) -> trustformers_core::errors::Result<String> {
            Ok("mock decoded text".to_string())
        }

        fn vocab_size(&self) -> usize {
            1000
        }

        fn get_vocab(&self) -> HashMap<String, u32> {
            let mut vocab = HashMap::new();
            vocab.insert("test".to_string(), 1);
            vocab.insert("mock".to_string(), 2);
            vocab.insert("token".to_string(), 3);
            vocab
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                "test" => Some(1),
                "mock" => Some(2),
                "token" => Some(3),
                _ => None,
            }
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            match id {
                1 => Some("test".to_string()),
                2 => Some("mock".to_string()),
                3 => Some("token".to_string()),
                _ => None,
            }
        }
    }

    #[test]
    fn test_text_to_speech_pipeline_creation() {
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = TextToSpeechPipeline::new(model, tokenizer);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_text_to_speech_config() {
        let config = TextToSpeechConfig::default();
        assert_eq!(config.voice, "default");
        assert_eq!(config.speaking_rate, 1.0);
        assert_eq!(config.pitch, 1.0);
        assert_eq!(config.volume, 1.0);
    }

    #[test]
    fn test_available_voices() {
        let voices = TextToSpeechPipeline::<MockModel, MockTokenizer>::get_available_voices();
        assert!(voices.contains(&"default".to_string()));
        assert!(voices.contains(&"male-neutral".to_string()));
        assert!(voices.contains(&"female-neutral".to_string()));
    }

    #[test]
    fn test_supported_emotions() {
        let emotions = TextToSpeechPipeline::<MockModel, MockTokenizer>::get_supported_emotions();
        assert!(emotions.contains(&"neutral".to_string()));
        assert!(emotions.contains(&"happy".to_string()));
        assert!(emotions.contains(&"sad".to_string()));
    }

    #[test]
    fn test_phoneme_converter() {
        let converter = PhonemeConverter::new();
        let phonemes = converter.text_to_phonemes("hello world").expect("operation failed in test");
        assert!(!phonemes.is_empty());
    }

    /// A pure tone at `freq` Hz, used to check the pitch tracker against a
    /// known ground truth.
    fn tone(freq: f32, sample_rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_prosody_analyzer_measures_a_known_pitch() {
        // Regression: `avg_pitch` used to be `mean|x| * 440.0`, which is not a
        // frequency at all. A real 200 Hz tone must be measured as ~200 Hz.
        let analyzer = ProsodyAnalyzer::new();
        let audio = tone(200.0, 16_000, 16_000);
        let prosody = analyzer.analyze("one two three", &audio, 16_000).expect("prosody");
        assert!(
            (prosody.avg_pitch - 200.0).abs() < 10.0,
            "expected ~200 Hz, measured {}",
            prosody.avg_pitch
        );
    }

    #[test]
    fn test_prosody_analyzer_distinguishes_pitches() {
        let analyzer = ProsodyAnalyzer::new();
        let low = analyzer
            .analyze("x", &tone(120.0, 16_000, 16_000), 16_000)
            .expect("low")
            .avg_pitch;
        let high = analyzer
            .analyze("x", &tone(330.0, 16_000, 16_000), 16_000)
            .expect("high")
            .avg_pitch;
        assert!(
            low < high,
            "120 Hz ({low}) must measure below 330 Hz ({high})"
        );
    }

    #[test]
    fn test_prosody_analyzer_rejects_empty_audio() {
        let analyzer = ProsodyAnalyzer::new();
        assert!(analyzer.analyze("x", &[], 22050).is_err());
        assert!(analyzer.analyze("x", &[0.1, 0.2], 0).is_err());
    }

    #[test]
    fn test_prosody_speaking_rate_is_measured_not_constant() {
        // Regression: `speaking_rate` used to be the literal 150.0.
        let analyzer = ProsodyAnalyzer::new();
        let audio = tone(200.0, 16_000, 16_000); // exactly 1 second
        let four_words = analyzer
            .analyze("one two three four", &audio, 16_000)
            .expect("prosody")
            .speaking_rate;
        let two_words = analyzer.analyze("one two", &audio, 16_000).expect("prosody").speaking_rate;
        // 4 words in 1/60 minute = 240 wpm; 2 words = 120 wpm.
        assert!((four_words - 240.0).abs() < 1.0, "got {four_words}");
        assert!((two_words - 120.0).abs() < 1.0, "got {two_words}");
        assert!(four_words > two_words);
    }

    #[test]
    fn test_pitch_track_is_empty_for_silence() {
        let analyzer = ProsodyAnalyzer::new();
        assert!(analyzer.pitch_track(&vec![0.0f32; 16_000], 16_000).is_empty());
    }

    #[test]
    fn test_detect_pauses_classifies_by_measured_duration() {
        // Regression: every pause used to be labelled `PauseType::Phrase`.
        let analyzer = ProsodyAnalyzer::new();
        let sample_rate = 1000u32;
        let mut audio = vec![1.0f32; 100];
        audio.extend(vec![0.0f32; 150]); // 150 ms → Breath
        audio.extend(vec![1.0f32; 100]);
        audio.extend(vec![0.0f32; 1000]); // 1 s → Sentence
        audio.extend(vec![1.0f32; 100]);
        let pauses = analyzer.detect_pauses(&audio, sample_rate).expect("pauses");
        assert_eq!(pauses.len(), 2, "expected two pauses, got {pauses:?}");
        assert!(matches!(pauses[0].pause_type, PauseType::Breath));
        assert!(matches!(pauses[1].pause_type, PauseType::Sentence));
    }

    #[test]
    fn test_synthesize_refuses_to_emit_a_synthetic_tone() {
        // Regression: `synthesize` used to return a sine-wave "waveform" as
        // speech. It must now fail loudly.
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = TextToSpeechPipeline::new(model, tokenizer).expect("pipeline ok");
        let input = TextToSpeechInput {
            text: "Hello world".to_string(),
            voice: None,
            speaking_rate: None,
            pitch: None,
            volume: None,
            emotion: None,
            prosody_markers: None,
        };
        match pipeline.synthesize(input) {
            Err(TrustformersError::FeatureUnavailable {
                feature, message, ..
            }) => {
                assert_eq!(feature, "tts-vocoder");
                assert!(message.contains("vocoder"), "message: {message}");
            },
            other => panic!("expected a vocoder FeatureUnavailable error, got {other:?}"),
        }
    }

    #[test]
    fn test_synthesize_still_validates_input_first() {
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = TextToSpeechPipeline::new(model, tokenizer).expect("pipeline ok");
        let empty = TextToSpeechInput {
            text: String::new(),
            voice: None,
            speaking_rate: None,
            pitch: None,
            volume: None,
            emotion: None,
            prosody_markers: None,
        };
        assert!(matches!(
            pipeline.synthesize(empty),
            Err(TrustformersError::InvalidInput { .. })
        ));

        let bad_voice = TextToSpeechInput {
            text: "hi".to_string(),
            voice: Some("nonexistent-voice".to_string()),
            speaking_rate: None,
            pitch: None,
            volume: None,
            emotion: None,
            prosody_markers: None,
        };
        assert!(matches!(
            pipeline.synthesize(bad_voice),
            Err(TrustformersError::InvalidInput { .. })
        ));
    }

    #[test]
    fn test_prepare_returns_the_real_front_end_output() {
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = TextToSpeechPipeline::new(model, tokenizer)
            .expect("pipeline ok")
            .with_phoneme_conversion(true);
        let input = TextToSpeechInput {
            text: "Dr. Smith has 5 cats.".to_string(),
            voice: None,
            speaking_rate: None,
            pitch: None,
            volume: None,
            emotion: None,
            prosody_markers: None,
        };
        let plan = pipeline.prepare(&input).expect("prepare");
        assert!(plan.normalized_text.contains("Doctor"));
        assert!(plan.normalized_text.contains("five"));
        assert_eq!(plan.token_ids, vec![1, 2, 3]);
        assert!(plan.phonemes.is_some_and(|p| !p.is_empty()));
    }

    #[test]
    fn test_finish_with_audio_uses_caller_supplied_waveform() {
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = TextToSpeechPipeline::new(model, tokenizer)
            .expect("pipeline ok")
            .with_config(TextToSpeechConfig {
                sample_rate: 16_000,
                ..Default::default()
            });
        let input = TextToSpeechInput {
            text: "hello".to_string(),
            voice: None,
            speaking_rate: None,
            pitch: None,
            volume: None,
            emotion: None,
            prosody_markers: None,
        };
        let plan = pipeline.prepare(&input).expect("prepare");
        let audio = tone(200.0, 16_000, 8_000);
        let output = pipeline.finish_with_audio(plan, audio.clone()).expect("finish");
        assert_eq!(output.audio_data, audio);
        assert!((output.duration - 0.5).abs() < 1e-6);
        assert!(pipeline
            .finish_with_audio(pipeline.prepare(&input).expect("prepare"), Vec::new())
            .is_err());
    }

    #[test]
    fn test_text_preprocessing() {
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline =
            TextToSpeechPipeline::new(model, tokenizer).expect("operation failed in test");

        let processed = pipeline
            .preprocess_text("Dr. Smith said 5 words.")
            .expect("operation failed in test");
        assert!(processed.contains("Doctor"));
        assert!(processed.contains("five"));
    }

    #[test]
    fn test_pipeline_configuration() {
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = TextToSpeechPipeline::new(model, tokenizer)
            .expect("operation failed in test")
            .with_voice("female-neutral".to_string())
            .with_speaking_rate(1.5)
            .with_pitch(1.2)
            .with_volume(0.8);

        assert_eq!(pipeline.config.voice, "female-neutral");
        assert_eq!(pipeline.config.speaking_rate, 1.5);
        assert_eq!(pipeline.config.pitch, 1.2);
        assert_eq!(pipeline.config.volume, 0.8);
    }

    // ── Additional tests ──────────────────────────────────────────────────────

    #[test]
    fn test_default_config_sample_rate() {
        let cfg = TextToSpeechConfig::default();
        assert_eq!(cfg.sample_rate, 22050);
    }

    #[test]
    fn test_default_config_prosody_disabled() {
        let cfg = TextToSpeechConfig::default();
        assert!(!cfg.prosody_control);
        assert!(!cfg.emotion_control);
        assert!(cfg.target_emotion.is_none());
    }

    #[test]
    fn test_default_config_max_duration_positive() {
        let cfg = TextToSpeechConfig::default();
        if let Some(d) = cfg.max_duration {
            assert!(d > 0.0, "max_duration must be positive when set");
        }
    }

    #[test]
    fn test_audio_format_variants_constructable() {
        let formats = [
            AudioFormat::Wav,
            AudioFormat::Mp3,
            AudioFormat::Flac,
            AudioFormat::Ogg,
            AudioFormat::Raw,
        ];
        assert_eq!(formats.len(), 5);
    }

    #[test]
    fn test_prosody_type_variants_constructable() {
        let types = [
            ProsodyType::Emphasis,
            ProsodyType::Pause,
            ProsodyType::Speed,
            ProsodyType::Pitch,
            ProsodyType::Volume,
        ];
        assert_eq!(types.len(), 5);
    }

    #[test]
    fn test_prosody_marker_fields() {
        let marker = ProsodyMarker {
            start: 0,
            end: 5,
            prosody_type: ProsodyType::Emphasis,
            intensity: 0.8,
        };
        assert_eq!(marker.start, 0);
        assert_eq!(marker.end, 5);
        assert!(marker.intensity >= 0.0 && marker.intensity <= 1.0);
    }

    #[test]
    fn test_tts_input_with_all_fields() {
        let input = TextToSpeechInput {
            text: "Hello world".to_string(),
            voice: Some("male-neutral".to_string()),
            speaking_rate: Some(1.2),
            pitch: Some(0.9),
            volume: Some(0.7),
            emotion: Some("happy".to_string()),
            prosody_markers: Some(vec![ProsodyMarker {
                start: 0,
                end: 5,
                prosody_type: ProsodyType::Emphasis,
                intensity: 0.5,
            }]),
        };
        assert_eq!(input.text, "Hello world");
        assert!(input.voice.is_some());
        assert!(input.speaking_rate.is_some());
        assert!(input.prosody_markers.is_some());
    }

    #[test]
    fn test_phoneme_converter_known_words() {
        let converter = PhonemeConverter::new();
        let hello_phonemes =
            converter.text_to_phonemes("hello").expect("phoneme conversion should succeed");
        // "hello" is in the dict → should return 4 phonemes
        assert_eq!(hello_phonemes.len(), 4, "hello → h ə ˈl oʊ (4 phonemes)");
    }

    #[test]
    fn test_phoneme_converter_unknown_word_fallback() {
        let converter = PhonemeConverter::new();
        let phonemes = converter
            .text_to_phonemes("zzz")
            .expect("phoneme conversion should succeed for unknown word");
        // Fallback: each character becomes a phoneme
        assert_eq!(phonemes.len(), 3, "unknown word → 1 phoneme per character");
    }

    #[test]
    fn test_expand_abbreviations_dr() {
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = TextToSpeechPipeline::new(model, tokenizer).expect("pipeline creation ok");
        let result = pipeline.expand_abbreviations("Dr. Smith");
        assert!(result.contains("Doctor"), "Dr. should expand to Doctor");
    }

    #[test]
    fn test_expand_abbreviations_ai() {
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = TextToSpeechPipeline::new(model, tokenizer).expect("pipeline creation ok");
        let result = pipeline.expand_abbreviations("AI systems");
        assert!(
            result.contains("Artificial Intelligence"),
            "AI should expand to Artificial Intelligence"
        );
    }

    #[test]
    fn test_normalize_numbers_single_digit() {
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = TextToSpeechPipeline::new(model, tokenizer).expect("pipeline creation ok");
        let result = pipeline.normalize_numbers(" 5 steps");
        assert!(result.contains("five"), "5 should become 'five'");
    }

    #[test]
    fn test_normalize_punctuation_period() {
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = TextToSpeechPipeline::new(model, tokenizer).expect("pipeline creation ok");
        let result = pipeline.normalize_punctuation("End.");
        assert!(
            result.contains("period"),
            "period should be expanded to 'period'"
        );
    }

    #[test]
    fn test_normalize_punctuation_question_mark() {
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = TextToSpeechPipeline::new(model, tokenizer).expect("pipeline creation ok");
        let result = pipeline.normalize_punctuation("Really?");
        assert!(result.contains("question"), "? should expand to 'question'");
    }

    #[test]
    fn test_prosody_analyzer_avg_pitch_non_negative() {
        let analyzer = ProsodyAnalyzer::new();
        // LCG-generated samples
        let mut s = 42u64;
        let samples: Vec<f32> = (0..22050)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s % 1000) as f32 / 1000.0
            })
            .collect();
        let prosody = analyzer.analyze("test text", &samples, 22050).expect("prosody ok");
        assert!(prosody.avg_pitch >= 0.0, "avg_pitch must be non-negative");
    }

    #[test]
    fn test_prosody_analyzer_speaking_rate_positive() {
        let analyzer = ProsodyAnalyzer::new();
        let samples = vec![0.3_f32; 22050];
        let prosody = analyzer.analyze("test", &samples, 22050).expect("prosody should succeed");
        assert!(
            prosody.speaking_rate > 0.0,
            "speaking_rate must be positive"
        );
    }

    #[test]
    fn test_phoneme_timings_ordering() {
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = TextToSpeechPipeline::new(model, tokenizer).expect("pipeline ok");
        let phonemes = vec!["h".to_string(), "e".to_string(), "l".to_string()];
        let timings =
            pipeline.generate_phoneme_timings(&phonemes, 1.5).expect("phoneme timings ok");
        assert_eq!(timings.len(), 3, "one timing per phoneme");
        // Verify start times are non-decreasing
        for w in timings.windows(2) {
            assert!(
                w[1].start_time >= w[0].start_time,
                "phoneme start times should be non-decreasing"
            );
        }
    }

    #[test]
    fn test_phoneme_timings_report_no_aligner_confidence() {
        // Regression: timings used to carry a hardcoded `confidence: 0.8`
        // despite no aligner having produced them.
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = TextToSpeechPipeline::new(model, tokenizer).expect("pipeline ok");
        let phonemes = vec!["a".to_string(), "b".to_string()];
        let timings =
            pipeline.generate_phoneme_timings(&phonemes, 1.0).expect("phoneme timings ok");
        assert_eq!(timings.len(), 2);
        for t in &timings {
            assert!(
                t.confidence.is_none(),
                "no aligner exists, so confidence must be None"
            );
        }
        assert!((timings[0].end_time - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_phoneme_timings_rejects_empty_input() {
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = TextToSpeechPipeline::new(model, tokenizer).expect("pipeline ok");
        assert!(pipeline.generate_phoneme_timings(&[], 1.0).is_err());
    }
}
