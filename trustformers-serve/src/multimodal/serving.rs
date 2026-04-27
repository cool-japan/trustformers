// Allow dead code for infrastructure under development
#![allow(dead_code)]

//! Multi-modal serving — image+text request handling
//!
//! Provides validated multi-modal request/response types and a processor
//! that estimates token costs and validates inputs for image+text LLM serving.

use std::collections::HashMap;
use std::fmt;

// ── modalities ────────────────────────────────────────────────────────────────

/// Supported input modalities
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
}

impl fmt::Display for Modality {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Modality::Text => write!(f, "text"),
            Modality::Image => write!(f, "image"),
            Modality::Audio => write!(f, "audio"),
            Modality::Video => write!(f, "video"),
        }
    }
}

// ── image format ─────────────────────────────────────────────────────────────

/// Image encoding format
#[derive(Debug, Clone, PartialEq)]
pub enum ImageFormat {
    Jpeg,
    Png,
    Webp,
    Gif,
    Unknown,
}

impl ImageFormat {
    /// Detect format from magic bytes
    pub fn from_bytes(data: &[u8]) -> Self {
        if data.len() >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
            return ImageFormat::Jpeg;
        }
        if data.len() >= 4
            && data[0] == 0x89
            && data[1] == 0x50
            && data[2] == 0x4E
            && data[3] == 0x47
        {
            return ImageFormat::Png;
        }
        // WebP: "RIFF????WEBP"
        if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP" {
            return ImageFormat::Webp;
        }
        // GIF: "GIF8"
        if data.len() >= 3 && data[0] == 0x47 && data[1] == 0x49 && data[2] == 0x46 {
            return ImageFormat::Gif;
        }
        ImageFormat::Unknown
    }

    /// Return the MIME type string for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "image/jpeg",
            ImageFormat::Png => "image/png",
            ImageFormat::Webp => "image/webp",
            ImageFormat::Gif => "image/gif",
            ImageFormat::Unknown => "application/octet-stream",
        }
    }
}

// ── multi-modal input ─────────────────────────────────────────────────────────

/// A single input in a multi-modal request
#[derive(Debug, Clone)]
pub enum MultiModalServingInput {
    /// Plain text content
    Text(String),
    /// Image referenced by URL or data URI
    ImageUrl(String),
    /// Raw image bytes with optional dimensions
    ImageBytes {
        data: Vec<u8>,
        format: ImageFormat,
        width: Option<u32>,
        height: Option<u32>,
    },
    /// Audio referenced by URL
    AudioUrl(String),
    /// Raw PCM/encoded audio bytes
    AudioBytes {
        data: Vec<u8>,
        sample_rate: u32,
        channels: u8,
    },
}

impl MultiModalServingInput {
    /// Return the modality of this input
    pub fn modality(&self) -> Modality {
        match self {
            MultiModalServingInput::Text(_) => Modality::Text,
            MultiModalServingInput::ImageUrl(_) | MultiModalServingInput::ImageBytes { .. } => {
                Modality::Image
            },
            MultiModalServingInput::AudioUrl(_) | MultiModalServingInput::AudioBytes { .. } => {
                Modality::Audio
            },
        }
    }

    /// Extract text content if this is a text input
    pub fn as_text(&self) -> Option<&str> {
        match self {
            MultiModalServingInput::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get the approximate size of the input in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            MultiModalServingInput::Text(s) => s.len(),
            MultiModalServingInput::ImageUrl(u) => u.len(),
            MultiModalServingInput::ImageBytes { data, .. } => data.len(),
            MultiModalServingInput::AudioUrl(u) => u.len(),
            MultiModalServingInput::AudioBytes { data, .. } => data.len(),
        }
    }

    /// Return true if this input carries image data
    pub fn is_image(&self) -> bool {
        matches!(
            self,
            MultiModalServingInput::ImageUrl(_) | MultiModalServingInput::ImageBytes { .. }
        )
    }

    /// Parse a data URI `data:<mime>;base64,<encoded>` into (mime_type, decoded_bytes)
    pub fn parse_data_uri(uri: &str) -> Result<(String, Vec<u8>), MultiModalServingError> {
        if !uri.starts_with("data:") {
            return Err(MultiModalServingError::InvalidDataUri(
                "URI must start with 'data:'".to_string(),
            ));
        }

        let rest = &uri[5..]; // strip "data:"

        let comma_pos = rest.find(',').ok_or_else(|| {
            MultiModalServingError::InvalidDataUri("missing ',' separator".to_string())
        })?;

        let header = &rest[..comma_pos];
        let encoded = &rest[comma_pos + 1..];

        // Header format: "<mime>;base64" or just "<mime>"
        let mime_type = if header.ends_with(";base64") {
            header[..header.len() - 7].to_string()
        } else {
            header.to_string()
        };

        if mime_type.is_empty() {
            return Err(MultiModalServingError::InvalidDataUri(
                "empty MIME type in data URI".to_string(),
            ));
        }

        let decoded = base64_decode(encoded)?;
        Ok((mime_type, decoded))
    }
}

// ── multi-modal message ───────────────────────────────────────────────────────

/// A multi-modal message (role + list of inputs)
#[derive(Debug, Clone)]
pub struct MultiModalMessage {
    /// Role: "user", "assistant", or "system"
    pub role: String,
    /// Ordered list of inputs for this message
    pub inputs: Vec<MultiModalServingInput>,
}

impl MultiModalMessage {
    /// Create an empty message with a role
    pub fn new(role: &str) -> Self {
        Self {
            role: role.to_string(),
            inputs: Vec::new(),
        }
    }

    /// Append a text input (builder style)
    pub fn with_text(mut self, text: &str) -> Self {
        self.inputs.push(MultiModalServingInput::Text(text.to_string()));
        self
    }

    /// Append an image URL input (builder style)
    pub fn with_image_url(mut self, url: &str) -> Self {
        self.inputs.push(MultiModalServingInput::ImageUrl(url.to_string()));
        self
    }

    /// Append raw image bytes (format auto-detected from magic bytes)
    pub fn with_image_bytes(mut self, data: Vec<u8>) -> Self {
        let format = ImageFormat::from_bytes(&data);
        self.inputs.push(MultiModalServingInput::ImageBytes {
            data,
            format,
            width: None,
            height: None,
        });
        self
    }

    /// Count inputs grouped by modality name
    pub fn count_by_modality(&self) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for input in &self.inputs {
            *counts.entry(input.modality().to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Return all text inputs concatenated with spaces
    pub fn all_text(&self) -> String {
        self.inputs.iter().filter_map(|i| i.as_text()).collect::<Vec<_>>().join(" ")
    }

    /// Return true if any input is an image
    pub fn has_image(&self) -> bool {
        self.inputs.iter().any(|i| i.is_image())
    }
}

// ── multi-modal request ───────────────────────────────────────────────────────

/// A complete multi-modal inference request
#[derive(Debug, Clone)]
pub struct MultiModalServingRequest {
    /// Conversation messages
    pub messages: Vec<MultiModalMessage>,
    /// Model identifier
    pub model: String,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature in [0, 2]
    pub temperature: f32,
    /// Whether to stream the response
    pub stream: bool,
}

impl MultiModalServingRequest {
    /// Create a new request for the given model with defaults
    pub fn new(model: &str) -> Self {
        Self {
            messages: Vec::new(),
            model: model.to_string(),
            max_tokens: 256,
            temperature: 1.0,
            stream: false,
        }
    }

    /// Append a message (builder style)
    pub fn with_message(mut self, message: MultiModalMessage) -> Self {
        self.messages.push(message);
        self
    }

    /// Set max tokens (builder style)
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Set sampling temperature (builder style)
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Enable or disable streaming (builder style)
    pub fn with_stream(mut self, s: bool) -> Self {
        self.stream = s;
        self
    }

    /// Collect all distinct modalities present in this request
    pub fn modalities(&self) -> Vec<Modality> {
        let mut seen: Vec<Modality> = Vec::new();
        for msg in &self.messages {
            for input in &msg.inputs {
                let m = input.modality();
                if !seen.contains(&m) {
                    seen.push(m);
                }
            }
        }
        seen
    }

    /// Validate the request and return an error if invalid
    pub fn validate(&self) -> Result<(), MultiModalServingError> {
        if self.model.is_empty() {
            return Err(MultiModalServingError::EmptyRequest);
        }
        if self.messages.is_empty() {
            return Err(MultiModalServingError::EmptyRequest);
        }
        if !(0.0..=2.0).contains(&self.temperature) {
            return Err(MultiModalServingError::InvalidTemperature(self.temperature));
        }
        if self.max_tokens == 0 {
            return Err(MultiModalServingError::InvalidMaxTokens(self.max_tokens));
        }
        Ok(())
    }

    /// Count total inputs across all messages
    pub fn total_input_count(&self) -> usize {
        self.messages.iter().map(|m| m.inputs.len()).sum()
    }
}

// ── multi-modal response ──────────────────────────────────────────────────────

/// A multi-modal inference response
#[derive(Debug, Clone)]
pub struct MultiModalServingResponse {
    /// Unique response ID
    pub id: String,
    /// Model that generated this response
    pub model: String,
    /// Generated text content
    pub content: String,
    /// Reason generation stopped
    pub finish_reason: String,
    /// Number of input tokens consumed
    pub input_tokens: usize,
    /// Number of tokens generated
    pub output_tokens: usize,
    /// Modalities that were processed
    pub modalities_processed: Vec<Modality>,
}

impl MultiModalServingResponse {
    /// Create a new response with default token counts
    pub fn new(id: &str, model: &str, content: &str) -> Self {
        Self {
            id: id.to_string(),
            model: model.to_string(),
            content: content.to_string(),
            finish_reason: "stop".to_string(),
            input_tokens: 0,
            output_tokens: 0,
            modalities_processed: Vec::new(),
        }
    }

    /// Serialize to a JSON string (manual, no serde)
    pub fn to_json(&self) -> String {
        let modalities_json: String = self
            .modalities_processed
            .iter()
            .map(|m| format!("\"{}\"", m))
            .collect::<Vec<_>>()
            .join(",");

        format!(
            r#"{{"id":{},"model":{},"content":{},"finish_reason":{},"input_tokens":{},"output_tokens":{},"modalities_processed":[{}]}}"#,
            json_escape_str(&self.id),
            json_escape_str(&self.model),
            json_escape_str(&self.content),
            json_escape_str(&self.finish_reason),
            self.input_tokens,
            self.output_tokens,
            modalities_json,
        )
    }
}

// ── multi-modal processor ─────────────────────────────────────────────────────

/// Validates and pre-processes multi-modal requests before forwarding to the
/// model backend.
pub struct MultiModalProcessor {
    /// Maximum image size in bytes (default: 20 MB)
    pub max_image_bytes: usize,
    /// Maximum number of images per request
    pub max_images_per_request: usize,
    /// Modalities that this processor accepts
    pub supported_modalities: Vec<Modality>,
}

impl Default for MultiModalProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiModalProcessor {
    /// Create a processor with default limits
    pub fn new() -> Self {
        Self {
            max_image_bytes: 20 * 1024 * 1024, // 20 MB
            max_images_per_request: 10,
            supported_modalities: vec![Modality::Text, Modality::Image, Modality::Audio],
        }
    }

    /// Validate the request and return a report
    pub fn validate_request(
        &self,
        request: &MultiModalServingRequest,
    ) -> Result<ValidationReport, MultiModalServingError> {
        // Basic request validation first
        request.validate()?;

        let mut warnings = Vec::new();
        let mut image_count = 0usize;

        for msg in &request.messages {
            for input in &msg.inputs {
                let modality = input.modality();

                // Check modality is supported
                if !self.supported_modalities.contains(&modality) {
                    return Err(MultiModalServingError::UnsupportedModality(modality));
                }

                // Image-specific checks
                if let MultiModalServingInput::ImageBytes { data, .. } = input {
                    image_count += 1;
                    if data.len() > self.max_image_bytes {
                        return Err(MultiModalServingError::ImageTooLarge {
                            size_bytes: data.len(),
                            limit_bytes: self.max_image_bytes,
                        });
                    }
                } else if input.is_image() {
                    image_count += 1;
                }

                if image_count > self.max_images_per_request {
                    return Err(MultiModalServingError::TooManyImages {
                        count: image_count,
                        limit: self.max_images_per_request,
                    });
                }

                // Warn about large audio (heuristic: > 5 MB)
                if matches!(input, MultiModalServingInput::AudioBytes { data, .. } if data.len() > 5 * 1024 * 1024)
                {
                    warnings.push("Audio input larger than 5 MB; latency may increase".to_string());
                }
            }
        }

        let estimated_tokens = self.estimate_token_cost(request);
        let modalities = request.modalities();

        Ok(ValidationReport {
            is_valid: true,
            warnings,
            estimated_tokens,
            modalities,
        })
    }

    /// Extract all image inputs from a request
    pub fn extract_images<'a>(
        &self,
        request: &'a MultiModalServingRequest,
    ) -> Vec<&'a MultiModalServingInput> {
        request
            .messages
            .iter()
            .flat_map(|m| m.inputs.iter())
            .filter(|i| i.is_image())
            .collect()
    }

    /// Estimate the total processing cost in tokens
    ///
    /// - Text: approximately `len / 4` tokens
    /// - Image 224×224: 256 tokens (ViT-style patch encoding)
    /// - Image 336×336: 576 tokens (LLaVA-style)
    /// - Default image (unknown size): 256 tokens
    /// - Audio: `sample_rate / 16000 * 25` tokens per second (Whisper-style estimate)
    pub fn estimate_token_cost(&self, request: &MultiModalServingRequest) -> usize {
        let mut total = 0usize;

        for msg in &request.messages {
            for input in &msg.inputs {
                match input {
                    MultiModalServingInput::Text(s) => {
                        total += (s.len() / 4).max(1);
                    },
                    MultiModalServingInput::ImageUrl(_) => {
                        // URL references: assume default patch count
                        total += 256;
                    },
                    MultiModalServingInput::ImageBytes { width, height, .. } => {
                        let tokens = match (width, height) {
                            (Some(224), Some(224)) => 256,
                            (Some(336), Some(336)) => 576,
                            (Some(w), Some(h)) => {
                                // Scale proportionally from 224×224 baseline
                                let scale = (*w as f64 * *h as f64) / (224.0 * 224.0);
                                (256.0 * scale) as usize
                            },
                            _ => 256, // default
                        };
                        total += tokens;
                    },
                    MultiModalServingInput::AudioUrl(_) => {
                        // Unknown duration, use 25 token default estimate
                        total += 25;
                    },
                    MultiModalServingInput::AudioBytes {
                        data, sample_rate, ..
                    } => {
                        // Estimate duration from byte count (PCM 16-bit mono)
                        let bytes_per_sample = 2usize; // 16-bit
                        let total_samples = data.len() / bytes_per_sample.max(1);
                        let duration_secs = if *sample_rate > 0 {
                            total_samples as f64 / *sample_rate as f64
                        } else {
                            1.0
                        };
                        let tokens_per_sec = (*sample_rate as f64 / 16000.0) * 25.0;
                        total += (duration_secs * tokens_per_sec).ceil() as usize;
                    },
                }
            }
        }

        total
    }
}

/// Report produced by `MultiModalProcessor::validate_request`
pub struct ValidationReport {
    /// Whether the request passed all validation checks
    pub is_valid: bool,
    /// Non-fatal issues that the caller may wish to surface
    pub warnings: Vec<String>,
    /// Estimated token budget required to process this request
    pub estimated_tokens: usize,
    /// Modalities found in the request
    pub modalities: Vec<Modality>,
}

// ── error type ────────────────────────────────────────────────────────────────

/// Error type for multi-modal serving operations
#[derive(Debug)]
pub enum MultiModalServingError {
    /// The request contained a modality not supported by this processor
    UnsupportedModality(Modality),
    /// An image exceeded the size limit
    ImageTooLarge {
        size_bytes: usize,
        limit_bytes: usize,
    },
    /// Too many images in a single request
    TooManyImages { count: usize, limit: usize },
    /// Data URI could not be parsed
    InvalidDataUri(String),
    /// Base64 data in a data URI was invalid
    InvalidBase64(String),
    /// Request had no model or no messages
    EmptyRequest,
    /// Temperature was outside the valid [0, 2] range
    InvalidTemperature(f32),
    /// max_tokens was zero
    InvalidMaxTokens(usize),
}

impl fmt::Display for MultiModalServingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MultiModalServingError::UnsupportedModality(m) => {
                write!(f, "unsupported modality: {}", m)
            },
            MultiModalServingError::ImageTooLarge {
                size_bytes,
                limit_bytes,
            } => write!(
                f,
                "image too large: {} bytes (limit {} bytes)",
                size_bytes, limit_bytes
            ),
            MultiModalServingError::TooManyImages { count, limit } => {
                write!(f, "too many images: {} (limit {})", count, limit)
            },
            MultiModalServingError::InvalidDataUri(msg) => {
                write!(f, "invalid data URI: {}", msg)
            },
            MultiModalServingError::InvalidBase64(msg) => {
                write!(f, "invalid base64: {}", msg)
            },
            MultiModalServingError::EmptyRequest => {
                write!(
                    f,
                    "request must have a non-empty model and at least one message"
                )
            },
            MultiModalServingError::InvalidTemperature(t) => {
                write!(f, "temperature {} is out of range [0, 2]", t)
            },
            MultiModalServingError::InvalidMaxTokens(n) => {
                write!(f, "max_tokens must be > 0, got {}", n)
            },
        }
    }
}

impl std::error::Error for MultiModalServingError {}

// ── base64 decoder ────────────────────────────────────────────────────────────

/// Pure-Rust base64 decoder (standard alphabet with padding)
pub(crate) fn base64_decode(input: &str) -> Result<Vec<u8>, MultiModalServingError> {
    // Standard base64 alphabet
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    // Build reverse lookup: character byte → 6-bit value
    let mut table = [0xFFu8; 256];
    for (i, &c) in ALPHABET.iter().enumerate() {
        table[c as usize] = i as u8;
    }

    // Strip whitespace and padding for the decode loop
    let clean: Vec<u8> = input
        .bytes()
        .filter(|&b| b != b'=' && b != b'\n' && b != b'\r' && b != b' ')
        .collect();

    if clean.len() % 4 == 1 {
        return Err(MultiModalServingError::InvalidBase64(
            "input length mod 4 == 1, which is never valid base64".to_string(),
        ));
    }

    let mut out = Vec::with_capacity((clean.len() * 3) / 4);

    let mut i = 0;
    while i + 3 < clean.len() {
        let b0 = table[clean[i] as usize];
        let b1 = table[clean[i + 1] as usize];
        let b2 = table[clean[i + 2] as usize];
        let b3 = table[clean[i + 3] as usize];

        if b0 == 0xFF || b1 == 0xFF {
            return Err(MultiModalServingError::InvalidBase64(format!(
                "invalid character at position {}",
                i
            )));
        }

        out.push((b0 << 2) | (b1 >> 4));

        if b2 != 0xFF {
            out.push((b1 << 4) | (b2 >> 2));
        }
        if b3 != 0xFF {
            out.push((b2 << 6) | b3);
        }

        i += 4;
    }

    // Handle tail: 2-char or 3-char groups (after stripping padding)
    let tail = clean.len() - i;
    match tail {
        0 => {},
        2 => {
            let b0 = table[clean[i] as usize];
            let b1 = table[clean[i + 1] as usize];
            if b0 == 0xFF || b1 == 0xFF {
                return Err(MultiModalServingError::InvalidBase64(
                    "invalid character in 2-byte tail".to_string(),
                ));
            }
            out.push((b0 << 2) | (b1 >> 4));
        },
        3 => {
            let b0 = table[clean[i] as usize];
            let b1 = table[clean[i + 1] as usize];
            let b2 = table[clean[i + 2] as usize];
            if b0 == 0xFF || b1 == 0xFF || b2 == 0xFF {
                return Err(MultiModalServingError::InvalidBase64(
                    "invalid character in 3-byte tail".to_string(),
                ));
            }
            out.push((b0 << 2) | (b1 >> 4));
            out.push((b1 << 4) | (b2 >> 2));
        },
        _ => {
            return Err(MultiModalServingError::InvalidBase64(
                "unexpected tail length after stripping padding".to_string(),
            ));
        },
    }

    Ok(out)
}

/// Escape a string for embedding in a JSON value (returns quoted string)
fn json_escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r#"\\"#),
            '\n' => out.push_str(r#"\n"#),
            '\r' => out.push_str(r#"\r"#),
            '\t' => out.push_str(r#"\t"#),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            },
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // JPEG magic bytes (FF D8 FF E0 ...)
    const JPEG_MAGIC: &[u8] = &[0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
    // PNG magic bytes (89 50 4E 47 0D 0A 1A 0A)
    const PNG_MAGIC: &[u8] = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    #[test]
    fn test_image_format_from_bytes_jpeg() {
        let fmt = ImageFormat::from_bytes(JPEG_MAGIC);
        assert_eq!(fmt, ImageFormat::Jpeg);
    }

    #[test]
    fn test_image_format_from_bytes_png() {
        let fmt = ImageFormat::from_bytes(PNG_MAGIC);
        assert_eq!(fmt, ImageFormat::Png);
    }

    #[test]
    fn test_image_format_mime_type() {
        assert_eq!(ImageFormat::Jpeg.mime_type(), "image/jpeg");
        assert_eq!(ImageFormat::Png.mime_type(), "image/png");
        assert_eq!(ImageFormat::Webp.mime_type(), "image/webp");
        assert_eq!(ImageFormat::Gif.mime_type(), "image/gif");
        assert_eq!(ImageFormat::Unknown.mime_type(), "application/octet-stream");
    }

    #[test]
    fn test_multimodal_input_modality_text() {
        let input = MultiModalServingInput::Text("hello".to_string());
        assert_eq!(input.modality(), Modality::Text);
    }

    #[test]
    fn test_multimodal_input_modality_image() {
        let input = MultiModalServingInput::ImageUrl("https://example.com/img.png".to_string());
        assert_eq!(input.modality(), Modality::Image);
    }

    #[test]
    fn test_multimodal_input_size_bytes() {
        let text = MultiModalServingInput::Text("hello".to_string());
        assert_eq!(text.size_bytes(), 5);

        let img = MultiModalServingInput::ImageBytes {
            data: vec![0u8; 1024],
            format: ImageFormat::Jpeg,
            width: None,
            height: None,
        };
        assert_eq!(img.size_bytes(), 1024);
    }

    #[test]
    fn test_multimodal_input_is_image() {
        let text = MultiModalServingInput::Text("hello".to_string());
        assert!(!text.is_image());

        let img = MultiModalServingInput::ImageUrl("http://example.com/img.png".to_string());
        assert!(img.is_image());

        let img_bytes = MultiModalServingInput::ImageBytes {
            data: vec![],
            format: ImageFormat::Png,
            width: None,
            height: None,
        };
        assert!(img_bytes.is_image());
    }

    #[test]
    fn test_parse_data_uri_basic() {
        // Base64 for "Hello"
        let uri = "data:text/plain;base64,SGVsbG8=";
        let result = MultiModalServingInput::parse_data_uri(uri);
        assert!(result.is_ok(), "parse should succeed: {:?}", result.err());
        let (mime, bytes) = result.expect("valid data URI should parse successfully");
        assert_eq!(mime, "text/plain");
        assert_eq!(bytes, b"Hello");
    }

    #[test]
    fn test_parse_data_uri_invalid() {
        // Missing "data:" prefix
        let result = MultiModalServingInput::parse_data_uri("blob:something");
        assert!(result.is_err());

        // Missing comma
        let result = MultiModalServingInput::parse_data_uri("data:image/png;base64");
        assert!(result.is_err());
    }

    #[test]
    fn test_multimodal_message_builder() {
        let msg = MultiModalMessage::new("user")
            .with_text("Describe this image")
            .with_image_url("https://example.com/cat.png");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.inputs.len(), 2);
    }

    #[test]
    fn test_multimodal_message_has_image() {
        let msg = MultiModalMessage::new("user")
            .with_text("Hello")
            .with_image_url("http://example.com/img.jpg");
        assert!(msg.has_image());

        let text_only = MultiModalMessage::new("user").with_text("No image here");
        assert!(!text_only.has_image());
    }

    #[test]
    fn test_multimodal_message_all_text() {
        let msg = MultiModalMessage::new("user")
            .with_text("hello")
            .with_image_url("http://x.com/a.png")
            .with_text("world");
        let text = msg.all_text();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_multimodal_request_builder() {
        let req = MultiModalServingRequest::new("gpt-4-vision")
            .with_message(MultiModalMessage::new("user").with_text("describe"))
            .with_max_tokens(512)
            .with_temperature(0.7)
            .with_stream(true);

        assert_eq!(req.model, "gpt-4-vision");
        assert_eq!(req.max_tokens, 512);
        assert!((req.temperature - 0.7).abs() < 1e-6);
        assert!(req.stream);
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_multimodal_request_validate_ok() {
        let req = MultiModalServingRequest::new("gpt-4-vision")
            .with_message(MultiModalMessage::new("user").with_text("hello"))
            .with_max_tokens(100)
            .with_temperature(1.0);
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_multimodal_request_validate_invalid_temp() {
        let req = MultiModalServingRequest::new("gpt-4-vision")
            .with_message(MultiModalMessage::new("user").with_text("hi"))
            .with_temperature(3.0); // out of range
        let err = req.validate();
        assert!(err.is_err());
        assert!(matches!(
            err,
            Err(MultiModalServingError::InvalidTemperature(_))
        ));
    }

    #[test]
    fn test_multimodal_processor_estimate_tokens_text() {
        let processor = MultiModalProcessor::new();
        let req = MultiModalServingRequest::new("gpt-4")
            .with_message(MultiModalMessage::new("user").with_text("hello world foo bar")); // 19 chars → 4 tokens
        let cost = processor.estimate_token_cost(&req);
        // "hello world foo bar" → 19 chars / 4 = 4 (integer)
        assert_eq!(cost, 4);
    }

    #[test]
    fn test_multimodal_processor_estimate_tokens_image() {
        let processor = MultiModalProcessor::new();
        // 224×224 image should cost 256 tokens
        let mut msg = MultiModalMessage::new("user");
        msg.inputs.push(MultiModalServingInput::ImageBytes {
            data: vec![0u8; 100],
            format: ImageFormat::Jpeg,
            width: Some(224),
            height: Some(224),
        });
        let req = MultiModalServingRequest::new("gpt-4-vision").with_message(msg);
        let cost = processor.estimate_token_cost(&req);
        assert_eq!(cost, 256);
    }

    #[test]
    fn test_multimodal_processor_validate_request() {
        let processor = MultiModalProcessor::new();
        let req = MultiModalServingRequest::new("gpt-4-vision")
            .with_message(MultiModalMessage::new("user").with_text("hello"))
            .with_max_tokens(256);
        let report = processor.validate_request(&req);
        assert!(report.is_ok(), "validation should pass: {:?}", report.err());
        let report = report.expect("report should be valid");
        assert!(report.is_valid);
        assert!(report.warnings.is_empty());
        assert!(report.estimated_tokens > 0);
    }

    #[test]
    fn test_multimodal_response_to_json() {
        let mut resp = MultiModalServingResponse::new("resp-001", "gpt-4-vision", "A cat");
        resp.input_tokens = 10;
        resp.output_tokens = 5;
        resp.modalities_processed = vec![Modality::Text, Modality::Image];
        let json = resp.to_json();
        assert!(json.contains("\"id\":\"resp-001\""));
        assert!(json.contains("\"content\":\"A cat\""));
        assert!(json.contains("\"input_tokens\":10"));
        assert!(json.contains("\"output_tokens\":5"));
        assert!(json.contains("\"text\""));
        assert!(json.contains("\"image\""));
    }

    #[test]
    fn test_base64_decode_basic() {
        // "Man" in base64 is "TWFu"
        let decoded = base64_decode("TWFu").expect("valid base64 should decode");
        assert_eq!(decoded, b"Man");
    }

    #[test]
    fn test_base64_decode_with_padding() {
        // "Hello" → "SGVsbG8="
        let decoded = base64_decode("SGVsbG8=").expect("valid padded base64 should decode");
        assert_eq!(decoded, b"Hello");
    }

    #[test]
    fn test_base64_decode_invalid() {
        // '!' is not a valid base64 character
        let result = base64_decode("SGVs!G8=");
        assert!(result.is_err());
    }
}
