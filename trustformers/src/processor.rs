use crate::error::{Result as TrustformersResult, TrustformersError};
use crate::pipeline::multimodal::MultiModalInput;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Auto processor that can handle multiple input modalities automatically
#[derive(Debug, Clone)]
pub struct AutoProcessor {
    /// Configuration for processing
    pub config: ProcessorConfig,
    /// Supported modalities
    pub supported_modalities: Vec<Modality>,
    /// Text processing configuration
    pub text_config: Option<TextProcessorConfig>,
    /// Image processing configuration
    pub image_config: Option<ImageProcessorConfig>,
    /// Audio processing configuration
    pub audio_config: Option<AudioProcessorConfig>,
    /// Video processing configuration
    pub video_config: Option<VideoProcessorConfig>,
}

/// Configuration for the auto processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Maximum input size per modality (in bytes)
    pub max_input_size: HashMap<String, usize>,
    /// Whether to validate inputs
    pub validate_inputs: bool,
    /// Whether to normalize inputs automatically
    pub auto_normalize: bool,
    /// Preprocessing steps to apply
    pub preprocessing_steps: Vec<PreprocessingStep>,
    /// Whether to extract metadata
    pub extract_metadata: bool,
    /// Quality checks to perform
    pub quality_checks: Vec<QualityCheck>,
    /// Default timeouts per modality (in milliseconds)
    pub processing_timeouts: HashMap<String, u64>,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        let mut max_input_size = HashMap::new();
        max_input_size.insert("text".to_string(), 1_000_000); // 1MB
        max_input_size.insert("image".to_string(), 50_000_000); // 50MB
        max_input_size.insert("audio".to_string(), 100_000_000); // 100MB
        max_input_size.insert("video".to_string(), 500_000_000); // 500MB

        let mut processing_timeouts = HashMap::new();
        processing_timeouts.insert("text".to_string(), 5000); // 5s
        processing_timeouts.insert("image".to_string(), 30000); // 30s
        processing_timeouts.insert("audio".to_string(), 60000); // 60s
        processing_timeouts.insert("video".to_string(), 120000); // 120s

        Self {
            max_input_size,
            validate_inputs: true,
            auto_normalize: true,
            preprocessing_steps: vec![
                PreprocessingStep::ValidateFormat,
                PreprocessingStep::CheckSize,
                PreprocessingStep::ExtractMetadata,
                PreprocessingStep::QualityCheck,
            ],
            extract_metadata: true,
            quality_checks: vec![
                QualityCheck::CheckCorruption,
                QualityCheck::ValidateEncoding,
                QualityCheck::CheckDimensions,
            ],
            processing_timeouts,
        }
    }
}

/// Supported modalities
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    MultiModal,
}

/// Preprocessing steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreprocessingStep {
    ValidateFormat,
    CheckSize,
    ExtractMetadata,
    QualityCheck,
    Normalize,
    Resize,
    Resample,
    ConvertFormat,
    RemoveNoise,
    AugmentData,
}

/// Quality check types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityCheck {
    CheckCorruption,
    ValidateEncoding,
    CheckDimensions,
    ValidateContent,
    CheckAudioQuality,
    ValidateVideoCodec,
    DetectLanguage,
    CheckTextEncoding,
}

/// Text processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextProcessorConfig {
    pub max_length: usize,
    pub truncation: bool,
    pub padding: bool,
    pub lowercase: bool,
    pub remove_special_chars: bool,
    pub supported_languages: Vec<String>,
    pub encoding: String,
}

impl Default for TextProcessorConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            truncation: true,
            padding: true,
            lowercase: false,
            remove_special_chars: false,
            supported_languages: vec![
                "en".to_string(),
                "es".to_string(),
                "fr".to_string(),
                "de".to_string(),
            ],
            encoding: "utf-8".to_string(),
        }
    }
}

/// Image processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProcessorConfig {
    pub target_size: (usize, usize),
    pub normalize: bool,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub supported_formats: Vec<String>,
    pub color_space: String,
    pub interpolation: String,
}

impl Default for ImageProcessorConfig {
    fn default() -> Self {
        Self {
            target_size: (224, 224),
            normalize: true,
            mean: vec![0.485, 0.456, 0.406], // ImageNet mean
            std: vec![0.229, 0.224, 0.225],  // ImageNet std
            supported_formats: vec![
                "jpeg".to_string(),
                "png".to_string(),
                "webp".to_string(),
                "bmp".to_string(),
            ],
            color_space: "RGB".to_string(),
            interpolation: "bilinear".to_string(),
        }
    }
}

/// Audio processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioProcessorConfig {
    pub sample_rate: usize,
    pub channels: usize,
    pub max_duration: f64,
    pub normalize: bool,
    pub supported_formats: Vec<String>,
    pub bit_depth: usize,
    pub frame_size: usize,
}

impl Default for AudioProcessorConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            max_duration: 30.0,
            normalize: true,
            supported_formats: vec![
                "wav".to_string(),
                "mp3".to_string(),
                "flac".to_string(),
                "ogg".to_string(),
            ],
            bit_depth: 16,
            frame_size: 400,
        }
    }
}

/// Video processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoProcessorConfig {
    pub target_fps: f32,
    pub target_resolution: (usize, usize),
    pub max_duration: f64,
    pub max_frames: usize,
    pub supported_codecs: Vec<String>,
    pub color_space: String,
    pub frame_sampling: String,
}

impl Default for VideoProcessorConfig {
    fn default() -> Self {
        Self {
            target_fps: 25.0,
            target_resolution: (224, 224),
            max_duration: 60.0,
            max_frames: 1500,
            supported_codecs: vec!["h264".to_string(), "h265".to_string(), "vp9".to_string()],
            color_space: "RGB".to_string(),
            frame_sampling: "uniform".to_string(),
        }
    }
}

/// Input validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub modality: Modality,
    pub detected_format: Option<String>,
    pub size_bytes: usize,
    pub issues: Vec<ValidationIssue>,
    pub metadata: HashMap<String, String>,
    pub quality_score: f32,
}

/// Validation issue
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub issue_type: String,
    pub severity: IssueSeverity,
    pub message: String,
    pub suggestion: Option<String>,
}

/// Issue severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Processing result with metadata
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub input: MultiModalInput,
    pub validation: ValidationResult,
    pub processing_time_ms: u64,
    pub preprocessing_applied: Vec<PreprocessingStep>,
    pub warnings: Vec<String>,
}

impl AutoProcessor {
    /// Create a new auto processor with default configuration
    pub fn new() -> Self {
        Self {
            config: ProcessorConfig::default(),
            supported_modalities: vec![
                Modality::Text,
                Modality::Image,
                Modality::Audio,
                Modality::Video,
                Modality::MultiModal,
            ],
            text_config: Some(TextProcessorConfig::default()),
            image_config: Some(ImageProcessorConfig::default()),
            audio_config: Some(AudioProcessorConfig::default()),
            video_config: Some(VideoProcessorConfig::default()),
        }
    }

    /// Create auto processor from pretrained configuration
    pub fn from_pretrained(model_name: &str) -> TrustformersResult<Self> {
        let mut processor = Self::new();

        // Configure based on known model types
        match model_name {
            name if name.contains("clip") => {
                processor = processor.for_vision_language_model();
            },
            name if name.contains("wav2vec") => {
                processor = processor.for_speech_model();
            },
            name if name.contains("videomae") || name.contains("video") => {
                processor = processor.for_video_model();
            },
            name if name.contains("layoutlm") => {
                processor = processor.for_document_model();
            },
            _ => {
                processor = processor.for_text_model();
            },
        }

        Ok(processor)
    }

    /// Configure processor for vision-language models
    pub fn for_vision_language_model(mut self) -> Self {
        // Optimize for CLIP-style models
        if let Some(ref mut image_config) = self.image_config {
            image_config.target_size = (224, 224);
            image_config.normalize = true;
        }

        if let Some(ref mut text_config) = self.text_config {
            text_config.max_length = 77; // CLIP text encoder length
        }

        self.supported_modalities = vec![Modality::Text, Modality::Image, Modality::MultiModal];
        self
    }

    /// Configure processor for speech models
    pub fn for_speech_model(mut self) -> Self {
        // Optimize for Wav2Vec/Whisper-style models
        if let Some(ref mut audio_config) = self.audio_config {
            audio_config.sample_rate = 16000;
            audio_config.channels = 1;
            audio_config.max_duration = 30.0;
        }

        self.supported_modalities = vec![Modality::Audio, Modality::Text];
        self.image_config = None;
        self.video_config = None;
        self
    }

    /// Configure processor for video models
    pub fn for_video_model(mut self) -> Self {
        // Optimize for video understanding models
        if let Some(ref mut video_config) = self.video_config {
            video_config.target_fps = 8.0; // Common for video transformers
            video_config.max_frames = 16;
            video_config.target_resolution = (224, 224);
        }

        self.supported_modalities = vec![Modality::Video, Modality::Text];
        self
    }

    /// Configure processor for document understanding models
    pub fn for_document_model(mut self) -> Self {
        // Optimize for LayoutLM-style models
        if let Some(ref mut image_config) = self.image_config {
            image_config.target_size = (224, 224);
            image_config.supported_formats.push("pdf".to_string());
        }

        if let Some(ref mut text_config) = self.text_config {
            text_config.max_length = 512;
        }

        self.supported_modalities = vec![Modality::Text, Modality::Image, Modality::MultiModal];
        self
    }

    /// Configure processor for text-only models
    pub fn for_text_model(mut self) -> Self {
        self.supported_modalities = vec![Modality::Text];
        self.image_config = None;
        self.audio_config = None;
        self.video_config = None;
        self
    }

    /// Automatically detect input modality
    pub fn detect_modality(&self, input: &[u8]) -> TrustformersResult<Modality> {
        if input.is_empty() {
            return Err(TrustformersError::invalid_input(
                "Empty input provided",
                Some("input"),
                Some("Non-empty data"),
                Some("Empty bytes"),
            ));
        }

        // Check for text (UTF-8)
        if std::str::from_utf8(input).is_ok() {
            return Ok(Modality::Text);
        }

        // Check for common image formats
        if self.is_image_format(input) {
            return Ok(Modality::Image);
        }

        // Check for common audio formats
        if self.is_audio_format(input) {
            return Ok(Modality::Audio);
        }

        // Check for common video formats
        if self.is_video_format(input) {
            return Ok(Modality::Video);
        }

        Err(TrustformersError::invalid_input(
            "Could not detect input modality",
            Some("input"),
            Some("Supported format (text, image, audio, video)"),
            Some("Unknown binary format"),
        ))
    }

    /// Validate input for specific modality
    pub fn validate_input(
        &self,
        input: &[u8],
        modality: &Modality,
    ) -> TrustformersResult<ValidationResult> {
        let start_time = std::time::Instant::now();

        let mut result = ValidationResult {
            is_valid: true,
            modality: modality.clone(),
            detected_format: None,
            size_bytes: input.len(),
            issues: Vec::new(),
            metadata: HashMap::new(),
            quality_score: 1.0,
        };

        // Check size limits
        let max_size = self
            .config
            .max_input_size
            .get(&format!("{:?}", modality).to_lowercase())
            .copied()
            .unwrap_or(10_000_000); // 10MB default

        if input.len() > max_size {
            result.issues.push(ValidationIssue {
                issue_type: "size_exceeded".to_string(),
                severity: IssueSeverity::Error,
                message: format!("Input size {} exceeds maximum {}", input.len(), max_size),
                suggestion: Some("Reduce input size or increase limit".to_string()),
            });
            result.is_valid = false;
        }

        // Modality-specific validation
        match modality {
            Modality::Text => self.validate_text_input(input, &mut result)?,
            Modality::Image => self.validate_image_input(input, &mut result)?,
            Modality::Audio => self.validate_audio_input(input, &mut result)?,
            Modality::Video => self.validate_video_input(input, &mut result)?,
            Modality::MultiModal => {
                // For multimodal, we need structured input
                result.issues.push(ValidationIssue {
                    issue_type: "multimodal_structure".to_string(),
                    severity: IssueSeverity::Info,
                    message: "Use process_multimodal() for structured multimodal input".to_string(),
                    suggestion: Some("Provide MultiModalInput structure".to_string()),
                });
            },
        }

        // Calculate quality score based on issues
        let error_count =
            result.issues.iter().filter(|i| i.severity == IssueSeverity::Error).count();
        let warning_count =
            result.issues.iter().filter(|i| i.severity == IssueSeverity::Warning).count();

        result.quality_score =
            (1.0 - (error_count as f32 * 0.5) - (warning_count as f32 * 0.1)).max(0.0);

        Ok(result)
    }

    /// Process input automatically detecting modality
    pub fn process(&self, input: &[u8]) -> TrustformersResult<ProcessingResult> {
        let start_time = std::time::Instant::now();

        // Detect modality
        let modality = self.detect_modality(input)?;

        // Validate input
        let validation = self.validate_input(input, &modality)?;

        if !validation.is_valid {
            let critical_issues: Vec<_> = validation
                .issues
                .iter()
                .filter(|i| {
                    i.severity == IssueSeverity::Critical || i.severity == IssueSeverity::Error
                })
                .collect();

            if !critical_issues.is_empty() {
                return Err(TrustformersError::invalid_input(
                    format!("Input validation failed: {}", critical_issues[0].message),
                    Some("input"),
                    Some("Valid input data"),
                    Some(&format!("{:?} data", modality)),
                ));
            }
        }

        // Create MultiModalInput based on detected modality
        let multimodal_input = match modality {
            Modality::Text => {
                let text = std::str::from_utf8(input).map_err(|_| {
                    TrustformersError::invalid_input(
                        "Invalid UTF-8 text",
                        Some("text"),
                        Some("Valid UTF-8 text"),
                        Some("Invalid encoding"),
                    )
                })?;

                MultiModalInput {
                    text: Some(text.to_string()),
                    image: None,
                    audio: None,
                    video: None,
                    metadata: HashMap::new(),
                    modality_weights: None,
                }
            },
            Modality::Image => MultiModalInput {
                text: None,
                image: Some(input.to_vec()),
                audio: None,
                video: None,
                metadata: HashMap::new(),
                modality_weights: None,
            },
            Modality::Audio => MultiModalInput {
                text: None,
                image: None,
                audio: Some(input.to_vec()),
                video: None,
                metadata: HashMap::new(),
                modality_weights: None,
            },
            Modality::Video => MultiModalInput {
                text: None,
                image: None,
                audio: None,
                video: Some(input.to_vec()),
                metadata: HashMap::new(),
                modality_weights: None,
            },
            Modality::MultiModal => {
                return Err(TrustformersError::invalid_input(
                    "Use process_multimodal() for structured multimodal input",
                    Some("input"),
                    Some("Structured MultiModalInput"),
                    Some("Raw bytes"),
                ));
            },
        };

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(ProcessingResult {
            input: multimodal_input,
            validation,
            processing_time_ms: processing_time,
            preprocessing_applied: self.config.preprocessing_steps.clone(),
            warnings: Vec::new(),
        })
    }

    /// Process structured multimodal input
    pub fn process_multimodal(
        &self,
        input: MultiModalInput,
    ) -> TrustformersResult<ProcessingResult> {
        let start_time = std::time::Instant::now();
        let mut warnings = Vec::new();

        // Validate each modality present in input
        if let Some(ref text) = input.text {
            let validation = self.validate_input(text.as_bytes(), &Modality::Text)?;
            if !validation.is_valid {
                warnings.push(format!(
                    "Text validation issues: {}",
                    validation
                        .issues
                        .iter()
                        .map(|i| &i.message)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }

        if let Some(ref image) = input.image {
            let validation = self.validate_input(image, &Modality::Image)?;
            if !validation.is_valid {
                warnings.push(format!(
                    "Image validation issues: {}",
                    validation
                        .issues
                        .iter()
                        .map(|i| &i.message)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }

        if let Some(ref audio) = input.audio {
            let validation = self.validate_input(audio, &Modality::Audio)?;
            if !validation.is_valid {
                warnings.push(format!(
                    "Audio validation issues: {}",
                    validation
                        .issues
                        .iter()
                        .map(|i| &i.message)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }

        if let Some(ref video) = input.video {
            let validation = self.validate_input(video, &Modality::Video)?;
            if !validation.is_valid {
                warnings.push(format!(
                    "Video validation issues: {}",
                    validation
                        .issues
                        .iter()
                        .map(|i| &i.message)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(ProcessingResult {
            input,
            validation: ValidationResult {
                is_valid: warnings.is_empty(),
                modality: Modality::MultiModal,
                detected_format: Some("multimodal".to_string()),
                size_bytes: 0, // Would calculate total size
                issues: Vec::new(),
                metadata: HashMap::new(),
                quality_score: if warnings.is_empty() { 1.0 } else { 0.8 },
            },
            processing_time_ms: processing_time,
            preprocessing_applied: self.config.preprocessing_steps.clone(),
            warnings,
        })
    }

    // Helper methods for format detection
    fn is_image_format(&self, input: &[u8]) -> bool {
        if input.len() < 4 {
            return false;
        }

        // Check common image magic numbers
        match &input[0..4] {
            [0xFF, 0xD8, 0xFF, _] => true,    // JPEG
            [0x89, 0x50, 0x4E, 0x47] => true, // PNG
            [0x47, 0x49, 0x46, 0x38] => true, // GIF
            [0x52, 0x49, 0x46, 0x46] if input.len() >= 12 && &input[8..12] == b"WEBP" => true, // WebP
            [0x42, 0x4D, _, _] => true, // BMP
            _ => false,
        }
    }

    fn is_audio_format(&self, input: &[u8]) -> bool {
        if input.len() < 4 {
            return false;
        }

        match &input[0..4] {
            [0x52, 0x49, 0x46, 0x46] if input.len() >= 12 && &input[8..12] == b"WAVE" => true, // WAV
            [0xFF, 0xFB, _, _] | [0xFF, 0xFA, _, _] | [0xFF, 0xF3, _, _] | [0xFF, 0xF2, _, _] => {
                true
            }, // MP3
            [0x66, 0x4C, 0x61, 0x43] => true, // FLAC
            [0x4F, 0x67, 0x67, 0x53] => true, // OGG
            _ => false,
        }
    }

    fn is_video_format(&self, input: &[u8]) -> bool {
        if input.len() < 8 {
            return false;
        }

        // Check for MP4/MOV
        if input.len() >= 8 && &input[4..8] == b"ftyp" {
            return true;
        }

        // Check for AVI
        if input.len() >= 12 && &input[0..4] == b"RIFF" && &input[8..12] == b"AVI " {
            return true;
        }

        // Check for WebM/MKV
        if input.len() >= 4 && input[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
            return true;
        }

        false
    }

    fn validate_text_input(
        &self,
        input: &[u8],
        result: &mut ValidationResult,
    ) -> TrustformersResult<()> {
        // Check if valid UTF-8
        match std::str::from_utf8(input) {
            Ok(text) => {
                result.detected_format = Some("utf-8".to_string());
                result.metadata.insert("length".to_string(), text.len().to_string());
                result.metadata.insert("lines".to_string(), text.lines().count().to_string());

                // Check text length limits
                if let Some(text_config) = &self.text_config {
                    if text.len() > text_config.max_length {
                        result.issues.push(ValidationIssue {
                            issue_type: "text_too_long".to_string(),
                            severity: IssueSeverity::Warning,
                            message: format!(
                                "Text length {} exceeds recommended {}",
                                text.len(),
                                text_config.max_length
                            ),
                            suggestion: Some("Consider truncation".to_string()),
                        });
                    }
                }
            },
            Err(_) => {
                result.issues.push(ValidationIssue {
                    issue_type: "invalid_encoding".to_string(),
                    severity: IssueSeverity::Error,
                    message: "Text is not valid UTF-8".to_string(),
                    suggestion: Some("Ensure text is properly encoded".to_string()),
                });
                result.is_valid = false;
            },
        }

        Ok(())
    }

    fn validate_image_input(
        &self,
        input: &[u8],
        result: &mut ValidationResult,
    ) -> TrustformersResult<()> {
        // Basic format detection already done, add more detailed validation
        if let Some(image_config) = &self.image_config {
            // Would integrate with actual image processing library for detailed validation
            result.detected_format = Some("image".to_string());
            result.metadata.insert("size_bytes".to_string(), input.len().to_string());

            // Placeholder for actual image dimension checking
            result.metadata.insert("width".to_string(), "unknown".to_string());
            result.metadata.insert("height".to_string(), "unknown".to_string());
        }

        Ok(())
    }

    fn validate_audio_input(
        &self,
        input: &[u8],
        result: &mut ValidationResult,
    ) -> TrustformersResult<()> {
        if let Some(audio_config) = &self.audio_config {
            result.detected_format = Some("audio".to_string());
            result.metadata.insert("size_bytes".to_string(), input.len().to_string());

            // Placeholder for actual audio analysis
            result.metadata.insert("duration".to_string(), "unknown".to_string());
            result.metadata.insert("sample_rate".to_string(), "unknown".to_string());
        }

        Ok(())
    }

    fn validate_video_input(
        &self,
        input: &[u8],
        result: &mut ValidationResult,
    ) -> TrustformersResult<()> {
        if let Some(video_config) = &self.video_config {
            result.detected_format = Some("video".to_string());
            result.metadata.insert("size_bytes".to_string(), input.len().to_string());

            // Placeholder for actual video analysis
            result.metadata.insert("duration".to_string(), "unknown".to_string());
            result.metadata.insert("fps".to_string(), "unknown".to_string());
            result.metadata.insert("resolution".to_string(), "unknown".to_string());
        }

        Ok(())
    }
}

impl Default for AutoProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_processor_creation() {
        let processor = AutoProcessor::new();
        assert_eq!(processor.supported_modalities.len(), 5);
        assert!(processor.text_config.is_some());
        assert!(processor.image_config.is_some());
    }

    #[test]
    fn test_modality_detection() {
        let processor = AutoProcessor::new();

        // Test text detection
        let text_input = "Hello, world!".as_bytes();
        let modality = processor.detect_modality(text_input).unwrap();
        assert_eq!(modality, Modality::Text);
    }

    #[test]
    fn test_vision_language_config() {
        let processor = AutoProcessor::new().for_vision_language_model();
        assert_eq!(processor.supported_modalities.len(), 3);

        if let Some(text_config) = processor.text_config {
            assert_eq!(text_config.max_length, 77);
        }
    }

    #[test]
    fn test_input_validation() {
        let processor = AutoProcessor::new();
        let text_input = "Test input".as_bytes();

        let validation = processor.validate_input(text_input, &Modality::Text).unwrap();
        assert!(validation.is_valid);
        assert_eq!(validation.modality, Modality::Text);
    }

    #[test]
    fn test_from_pretrained() {
        let processor = AutoProcessor::from_pretrained("clip-vit-base-patch32").unwrap();
        assert!(processor.supported_modalities.contains(&Modality::Image));
        assert!(processor.supported_modalities.contains(&Modality::Text));
    }
}
