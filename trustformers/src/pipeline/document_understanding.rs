//! # Document understanding pipeline
//!
//! ## What is real here
//!
//! Everything that operates on *text you supply*:
//!
//! * [`DocumentUnderstandingPipeline::analyze_text`] — the full text-side
//!   analysis: language-specific normalisation (Chinese/Japanese/Arabic/Latin),
//!   regex key-value extraction with a confidence model that rewards common
//!   form fields and structured values, de-duplication by key, and
//!   [`DocumentUnderstandingPipeline::extract_entities`], a real regex NER for
//!   emails, phone numbers, URLs, dates and monetary amounts.
//! * [`DocumentUnderstandingPipeline::preprocess_text`] and the per-script
//!   helpers.
//!
//! ## What is not available
//!
//! There is **no OCR engine and no document layout model** in this workspace.
//! [`DocumentUnderstandingPipeline::extract_text`],
//! [`DocumentUnderstandingPipeline::perform_ocr`],
//! [`DocumentUnderstandingPipeline::extract_layout`] and
//! [`DocumentUnderstandingPipeline::extract_tables`] therefore return a
//! structured [`TrustformersError::FeatureUnavailable`], and the `Pipeline`
//! implementation fails for any non-empty document.
//!
//! Previously these methods returned fixtures — `"Sample OCR text"`, a
//! `"John Doe"` PERSON entity, two invented financial tables and
//! `format!("Answer to '{{question}}' based on document content")` — regardless
//! of the input. None of that survives.
//!
//! Run your own OCR and hand the text to
//! [`DocumentUnderstandingPipeline::analyze_text`].

use crate::core::traits::{Model, Tokenizer};
use crate::error::{Result, TrustformersError};
use crate::pipeline::{BasePipeline, Device, Pipeline};
use serde::{Deserialize, Serialize};
use trustformers_core::cache::CacheKeyBuilder;

/// Build the "no OCR / layout engine" error shared by this module.
fn ocr_unavailable(operation: &str) -> TrustformersError {
    TrustformersError::FeatureUnavailable {
        message: format!(
            "{operation} requires an OCR / document-layout engine, and none is implemented in \
             this workspace. This pipeline never returns canned text, entities or tables — run \
             your own OCR and call `analyze_text` with the result."
        ),
        feature: "document-ocr".to_string(),
        suggestion: Some(
            "Use `DocumentUnderstandingPipeline::analyze_text` with text you extracted yourself."
                .to_string(),
        ),
        alternatives: Vec::new(),
    }
}

/// Configuration for document understanding pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentUnderstandingConfig {
    /// Maximum number of tokens to process
    pub max_length: usize,
    /// Whether to return OCR results
    pub return_ocr_results: bool,
    /// Whether to return layout information
    pub return_layout: bool,
    /// Whether to return key-value pairs
    pub return_key_value_pairs: bool,
    /// Whether to return entities
    pub return_entities: bool,
    /// Confidence threshold for extraction
    pub confidence_threshold: f32,
    /// Whether to return raw text
    pub return_text: bool,
    /// Language hints for OCR
    pub language_hints: Vec<String>,
    /// Whether to apply text preprocessing
    pub preprocess_text: bool,
}

impl Default for DocumentUnderstandingConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            return_ocr_results: true,
            return_layout: true,
            return_key_value_pairs: true,
            return_entities: true,
            confidence_threshold: 0.5,
            return_text: true,
            language_hints: vec!["en".to_string()],
            preprocess_text: true,
        }
    }
}

/// Input for document understanding pipeline
#[derive(Debug, Clone)]
pub struct DocumentUnderstandingInput {
    /// Document image as bytes
    pub image: Vec<u8>,
    /// MIME type of the image
    pub image_type: String,
    /// Optional question about the document
    pub question: Option<String>,
    /// Optional specific extraction targets
    pub extraction_targets: Option<Vec<String>>,
}

/// Bounding box for layout information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Text block with layout information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBlock {
    pub text: String,
    pub bounding_box: BoundingBox,
    pub confidence: f32,
    pub block_type: TextBlockType,
}

/// Type of text block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextBlockType {
    Title,
    Heading,
    Paragraph,
    List,
    Table,
    Footer,
    Header,
    Caption,
    Other,
}

/// Key-value pair extracted from document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyValuePair {
    pub key: String,
    pub value: String,
    pub key_bbox: BoundingBox,
    pub value_bbox: BoundingBox,
    pub confidence: f32,
}

/// Named entity extracted from document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentEntity {
    pub text: String,
    pub entity_type: String,
    pub bounding_box: BoundingBox,
    pub confidence: f32,
}

/// Table structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    pub rows: Vec<Vec<String>>,
    pub headers: Option<Vec<String>>,
    pub bounding_box: BoundingBox,
    pub confidence: f32,
}

/// OCR result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRResult {
    pub text: String,
    pub bounding_box: BoundingBox,
    pub confidence: f32,
    pub word_level_boxes: Option<Vec<(String, BoundingBox)>>,
}

/// Output from document understanding pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentUnderstandingOutput {
    /// Raw text extracted from document
    pub text: Option<String>,
    /// Text blocks with layout information
    pub text_blocks: Option<Vec<TextBlock>>,
    /// Key-value pairs extracted
    pub key_value_pairs: Option<Vec<KeyValuePair>>,
    /// Named entities found
    pub entities: Option<Vec<DocumentEntity>>,
    /// Tables found in document
    pub tables: Option<Vec<Table>>,
    /// OCR results
    pub ocr_results: Option<Vec<OCRResult>>,
    /// Answer to question if provided
    pub answer: Option<String>,
    /// Processing metadata
    pub metadata: DocumentMetadata,
}

/// Metadata about the document processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub page_count: usize,
    /// Measured wall-clock processing time in milliseconds.
    pub processing_time_ms: u64,
    /// Language detected from the text, or `None` when detection was not run.
    ///
    /// Never a hardcoded `"en"`.
    pub detected_language: Option<String>,
    /// Page rotation in degrees, or `None` when no deskew stage ran.
    pub text_orientation: Option<f32>,
    /// Scan-quality estimate, or `None` when no quality model ran.
    pub quality_score: Option<f32>,
}

/// Document understanding pipeline
pub struct DocumentUnderstandingPipeline<M, T> {
    base: BasePipeline<M, T>,
    config: DocumentUnderstandingConfig,
}

impl<M, T> DocumentUnderstandingPipeline<M, T>
where
    M: Model + Send + Sync + 'static,
    T: Tokenizer + Send + Sync + 'static,
{
    pub fn new(model: M, tokenizer: T) -> Result<Self> {
        Ok(Self {
            base: BasePipeline::new(model, tokenizer),
            config: DocumentUnderstandingConfig::default(),
        })
    }

    pub fn with_config(mut self, config: DocumentUnderstandingConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.config.max_length = max_length;
        self
    }

    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.config.confidence_threshold = threshold;
        self
    }

    pub fn with_language_hints(mut self, hints: Vec<String>) -> Self {
        self.config.language_hints = hints;
        self
    }

    pub fn to_device(mut self, device: Device) -> Self {
        self.base = self.base.to_device(device);
        self
    }

    /// Extract text from a document image using OCR.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::FeatureUnavailable`] for any non-empty
    /// document: no OCR engine is implemented in this workspace. An empty
    /// buffer yields an empty string.
    pub fn extract_text(&self, image: &[u8]) -> Result<String> {
        if image.is_empty() {
            return Ok(String::new());
        }
        Err(ocr_unavailable("text extraction"))
    }

    /// Whether `image` looks like a PDF container (`%PDF` magic).
    pub fn is_pdf_image(&self, image: &[u8]) -> bool {
        image.len() > 4 && &image[0..4] == b"%PDF"
    }

    /// Apply language-specific processing
    fn apply_language_processing(&self, text: &str) -> Result<String> {
        let mut processed_text = text.to_string();

        for lang in &self.config.language_hints {
            match lang.as_str() {
                "zh" | "zh-CN" | "zh-TW" => {
                    // Chinese text processing
                    processed_text = self.process_chinese_text(&processed_text);
                },
                "ja" => {
                    // Japanese text processing
                    processed_text = self.process_japanese_text(&processed_text);
                },
                "ar" => {
                    // Arabic text processing (RTL)
                    processed_text = self.process_arabic_text(&processed_text);
                },
                _ => {
                    // Default Latin text processing
                    processed_text = self.process_latin_text(&processed_text);
                },
            }
        }

        Ok(processed_text)
    }

    fn process_chinese_text(&self, text: &str) -> String {
        // Chinese text normalization
        text.chars()
            .filter(|c| !c.is_whitespace() || c == &' ')
            .collect::<String>()
            .trim()
            .to_string()
    }

    fn process_japanese_text(&self, text: &str) -> String {
        // Japanese text processing
        text.lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("")
    }

    fn process_arabic_text(&self, text: &str) -> String {
        // Arabic text processing (RTL support)
        text.trim().to_string()
    }

    fn process_latin_text(&self, text: &str) -> String {
        // Standard Latin text processing
        text.lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Extract layout information (text blocks with bounding boxes).
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::FeatureUnavailable`] for any non-empty
    /// document: no layout model is implemented. Previously this returned four
    /// fixed regions carrying the strings `"Document Header"`,
    /// `"Main Document Title"`, a canned body paragraph and
    /// `"Page 1 | Footer Information"`, independent of the input.
    pub fn extract_layout(&self, image: &[u8]) -> Result<Vec<TextBlock>> {
        if image.is_empty() {
            return Ok(Vec::new());
        }
        Err(ocr_unavailable("layout analysis"))
    }

    /// Sort text blocks into reading order (top-to-bottom, left-to-right).
    ///
    /// Real geometry, reusable with blocks from any layout engine: blocks whose
    /// vertical centres are within 20 units are treated as one line and ordered
    /// by `x`.
    pub fn sort_reading_order(&self, mut blocks: Vec<TextBlock>) -> Vec<TextBlock> {
        blocks.sort_by(|a, b| {
            let y_diff = (a.bounding_box.y - b.bounding_box.y).abs();
            if y_diff < 20.0 {
                a.bounding_box
                    .x
                    .partial_cmp(&b.bounding_box.x)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.bounding_box
                    .y
                    .partial_cmp(&b.bounding_box.y)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        blocks
    }

    /// Extract key-value pairs from already-extracted document text.
    ///
    /// Real regex matching over `Key: Value`, `Key = Value`, `Key - Value` and
    /// whitespace-separated forms, scored by `Self::calculate_kv_confidence`
    /// and de-duplicated by normalised key.
    ///
    /// # Errors
    ///
    /// Propagates de-duplication failures (currently infallible).
    pub fn extract_key_value_pairs(&self, text: &str) -> Result<Vec<KeyValuePair>> {
        let mut pairs = Vec::new();

        // Extract key-value pairs using regex patterns
        let kv_patterns = [
            // Common form field patterns
            (r"([A-Za-z\s]+):\s*(.+)", 1.0),    // "Name: John Doe"
            (r"([A-Za-z\s]+)\s*=\s*(.+)", 0.9), // "Name = John Doe"
            (r"([A-Za-z\s]+)\s*-\s*(.+)", 0.8), // "Name - John Doe"
            (r"([A-Za-z\s]+)\s+(.+?)(?:\n|$)", 0.7), // "Name John Doe"
        ];

        for line in text.lines() {
            for (pattern, base_confidence) in &kv_patterns {
                if let Ok(re) = regex::Regex::new(pattern) {
                    if let Some(captures) = re.captures(line.trim()) {
                        if let (Some(key_match), Some(value_match)) =
                            (captures.get(1), captures.get(2))
                        {
                            let key = key_match.as_str().trim();
                            let value = value_match.as_str().trim();

                            // Skip empty or very short values
                            if value.len() < 2 || key.len() < 2 {
                                continue;
                            }

                            // Calculate confidence based on pattern and content quality
                            let confidence =
                                self.calculate_kv_confidence(key, value, *base_confidence);

                            if confidence >= self.config.confidence_threshold {
                                let pair = KeyValuePair {
                                    key: key.to_string(),
                                    value: value.to_string(),
                                    key_bbox: self.estimate_text_bbox(
                                        key,
                                        100.0,
                                        200.0 + pairs.len() as f32 * 25.0,
                                    ),
                                    value_bbox: self.estimate_text_bbox(
                                        value,
                                        200.0,
                                        200.0 + pairs.len() as f32 * 25.0,
                                    ),
                                    confidence,
                                };
                                pairs.push(pair);
                                break; // Use first matching pattern
                            }
                        }
                    }
                }
            }
        }

        // Remove duplicate keys (keep highest confidence)
        self.deduplicate_key_value_pairs(pairs)
    }

    /// Calculate confidence score for key-value pair
    fn calculate_kv_confidence(&self, key: &str, value: &str, base_confidence: f32) -> f32 {
        let mut confidence = base_confidence;

        // Boost confidence for common form fields
        let common_keys = [
            "name",
            "address",
            "phone",
            "email",
            "date",
            "amount",
            "total",
            "quantity",
            "price",
            "description",
            "company",
        ];

        if common_keys.iter().any(|&k| key.to_lowercase().contains(k)) {
            confidence += 0.1;
        }

        // Reduce confidence for very long keys or values
        if key.len() > 50 || value.len() > 200 {
            confidence -= 0.2;
        }

        // Boost confidence for structured values (dates, emails, phones)
        if self.is_structured_value(value) {
            confidence += 0.15;
        }

        confidence.clamp(0.0, 1.0)
    }

    /// Check if value follows a structured format
    fn is_structured_value(&self, value: &str) -> bool {
        // Date patterns
        if regex::Regex::new(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")
            .map(|re| re.is_match(value))
            .unwrap_or(false)
        {
            return true;
        }

        // Email pattern
        if regex::Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
            .map(|re| re.is_match(value))
            .unwrap_or(false)
        {
            return true;
        }

        // Phone number pattern
        if regex::Regex::new(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
            .map(|re| re.is_match(value))
            .unwrap_or(false)
        {
            return true;
        }

        false
    }

    /// Estimate bounding box for text
    fn estimate_text_bbox(&self, text: &str, x: f32, y: f32) -> BoundingBox {
        let char_width = 8.0; // Approximate character width
        let line_height = 20.0;

        BoundingBox {
            x,
            y,
            width: text.len() as f32 * char_width,
            height: line_height,
        }
    }

    /// Remove duplicate key-value pairs
    fn deduplicate_key_value_pairs(&self, pairs: Vec<KeyValuePair>) -> Result<Vec<KeyValuePair>> {
        use std::collections::HashMap;

        let mut best_pairs: HashMap<String, KeyValuePair> = HashMap::new();

        for pair in pairs {
            let key_normalized = pair.key.to_lowercase().trim().to_string();

            match best_pairs.get(&key_normalized) {
                Some(existing) if existing.confidence >= pair.confidence => {
                    // Keep existing
                },
                _ => {
                    // Insert new or replace existing
                    best_pairs.insert(key_normalized, pair);
                },
            }
        }

        Ok(best_pairs.into_values().collect())
    }

    /// Extract named entities from already-extracted document text.
    ///
    /// A real, pattern-based recogniser over the text the caller supplies. It
    /// finds `EMAIL`, `URL`, `PHONE`, `DATE` and `MONEY` spans — the classes a
    /// regex can identify reliably — and reports the *character offsets* of
    /// each match as its bounding box (`x` = start offset, `width` = length),
    /// because without OCR there are no pixel coordinates to report.
    ///
    /// Confidences are fixed per class and documented as pattern-precision
    /// estimates, not model outputs. Entities below
    /// `config.confidence_threshold` are dropped.
    ///
    /// Previously this returned a single hardcoded `"John Doe"` PERSON entity
    /// for every input.
    ///
    /// # Errors
    ///
    /// Returns an error only if a built-in pattern fails to compile.
    pub fn extract_entities(&self, text: &str) -> Result<Vec<DocumentEntity>> {
        // (name, pattern, pattern-precision estimate)
        const PATTERNS: &[(&str, &str, f32)] = &[
            (
                "EMAIL",
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                0.95,
            ),
            ("URL", r#"https?://[^\s<>"]+"#, 0.95),
            (
                "PHONE",
                r"\+?\d{1,3}[-. ]?\(?\d{2,4}\)?[-. ]?\d{3,4}[-. ]?\d{3,4}",
                0.75,
            ),
            ("DATE", r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", 0.85),
            ("MONEY", r"[$£€¥]\s?\d[\d,]*(?:\.\d{1,2})?", 0.90),
        ];

        let mut entities = Vec::new();
        for (entity_type, pattern, confidence) in PATTERNS {
            if *confidence < self.config.confidence_threshold {
                continue;
            }
            let re = regex::Regex::new(pattern).map_err(|e| {
                TrustformersError::pipeline(
                    format!("built-in {entity_type} pattern failed to compile: {e}"),
                    "document-understanding",
                )
            })?;
            for m in re.find_iter(text) {
                entities.push(DocumentEntity {
                    text: m.as_str().to_string(),
                    entity_type: (*entity_type).to_string(),
                    bounding_box: BoundingBox {
                        x: m.start() as f32,
                        y: 0.0,
                        width: (m.end() - m.start()) as f32,
                        height: 0.0,
                    },
                    confidence: *confidence,
                });
            }
        }
        entities.sort_by_key(|e| e.bounding_box.x as usize);
        Ok(entities)
    }

    /// Extract tables from a document.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::FeatureUnavailable`] for any non-empty
    /// document: no table-structure detector is implemented. Previously this
    /// returned two invented tables — a four-column "financial" one and a
    /// three-column "contact" one — for every input.
    pub fn extract_tables(&self, image: &[u8]) -> Result<Vec<Table>> {
        if image.is_empty() {
            return Ok(Vec::new());
        }
        Err(ocr_unavailable("table extraction"))
    }

    /// Perform OCR on a document image.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::FeatureUnavailable`] for any non-empty
    /// document: no OCR engine is implemented. Previously this returned a fixed
    /// `"Sample OCR text"` result with a 0.92 confidence and two word boxes,
    /// regardless of the image.
    pub fn perform_ocr(&self, image: &[u8]) -> Result<Vec<OCRResult>> {
        if image.is_empty() {
            return Ok(Vec::new());
        }
        Err(ocr_unavailable("OCR"))
    }

    /// Answer a question about a document.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::FeatureUnavailable`]: no document QA model
    /// is wired in. Previously this returned
    /// `format!("Answer to '{{question}}' based on document content")`, which
    /// contained no information from the document at all.
    pub fn answer_question(&self, _text: &str, question: &str) -> Result<String> {
        Err(TrustformersError::FeatureUnavailable {
            message: format!(
                "document question answering is not implemented; the question {question:?} \
                 cannot be answered. This pipeline never returns a templated non-answer."
            ),
            feature: "document-qa".to_string(),
            suggestion: Some(
                "Run your own document QA model over the text from `analyze_text`.".to_string(),
            ),
            alternatives: Vec::new(),
        })
    }

    /// Analyse already-extracted document text.
    ///
    /// This is the pipeline's real, fully-implemented half: normalisation,
    /// key-value extraction and pattern-based entity recognition, all driven by
    /// the same configuration flags the `Pipeline` implementation uses.
    ///
    /// Fields the pipeline cannot produce without OCR (`text_blocks`,
    /// `ocr_results`, `tables`, `answer`) are left `None` rather than filled
    /// with fixtures.
    ///
    /// # Errors
    ///
    /// Propagates errors from entity extraction.
    pub fn analyze_text(&self, text: &str) -> Result<DocumentUnderstandingOutput> {
        let start_time = std::time::Instant::now();
        let processed = self.preprocess_text(&self.apply_language_processing(text)?);

        let key_value_pairs = if self.config.return_key_value_pairs {
            Some(self.extract_key_value_pairs(&processed)?)
        } else {
            None
        };
        let entities = if self.config.return_entities {
            Some(self.extract_entities(&processed)?)
        } else {
            None
        };

        Ok(DocumentUnderstandingOutput {
            text: if self.config.return_text { Some(processed) } else { None },
            text_blocks: None,
            key_value_pairs,
            entities,
            tables: None,
            ocr_results: None,
            answer: None,
            metadata: DocumentMetadata {
                page_count: 1,
                processing_time_ms: start_time.elapsed().as_millis() as u64,
                detected_language: None,
                text_orientation: None,
                quality_score: None,
            },
        })
    }

    /// Preprocess text
    pub fn preprocess_text(&self, text: &str) -> String {
        if self.config.preprocess_text {
            // Basic text preprocessing
            text.lines()
                .map(|line| line.trim())
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            text.to_string()
        }
    }
}

impl<M, T> Pipeline for DocumentUnderstandingPipeline<M, T>
where
    M: Model + Send + Sync + 'static,
    T: Tokenizer + Send + Sync + 'static,
{
    type Input = DocumentUnderstandingInput;
    type Output = DocumentUnderstandingOutput;

    /// Run the pipeline over a document image.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::FeatureUnavailable`] for any non-empty
    /// document: without an OCR engine there is no text to analyse, and this
    /// pipeline will not substitute fixtures. Call
    /// [`DocumentUnderstandingPipeline::analyze_text`] with text you extracted
    /// yourself.
    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        let start_time = std::time::Instant::now();

        // Extract text from the image. This is where the pipeline stops without
        // an OCR engine — the error names the alternative.
        let text = self.extract_text(&input.image)?;

        // An empty document is genuinely analysable (there is nothing to read),
        // so the caller still gets a well-formed, empty result.
        let cache_key = if let Some(cache) = &self.base.cache {
            let mut builder = CacheKeyBuilder::new("document_understanding", "image_analysis")
                .with_param("image_type", &input.image_type)
                .with_param("image_hash", &input.image.len())
                .with_param(
                    "config",
                    &serde_json::to_string(&self.config).unwrap_or_default(),
                );

            if let Some(question) = &input.question {
                builder = builder.with_text(question);
            }

            let key = builder.build();
            if let Some(cached) = cache.get(&key) {
                if let Ok(output) = serde_json::from_slice::<DocumentUnderstandingOutput>(&cached) {
                    return Ok(output);
                }
            }
            Some(key)
        } else {
            None
        };

        let mut output = self.analyze_text(&text)?;

        if let Some(question) = &input.question {
            output.answer = Some(self.answer_question(&text, question)?);
        }

        output.metadata.processing_time_ms = start_time.elapsed().as_millis() as u64;

        if let (Some(cache), Some(key)) = (&self.base.cache, cache_key) {
            if let Ok(serialized) = serde_json::to_vec(&output) {
                cache.insert(key, serialized);
            }
        }

        Ok(output)
    }
}

/// Factory function for document understanding pipeline
pub fn document_understanding_pipeline<M, T>(
    model: M,
    tokenizer: T,
) -> Result<DocumentUnderstandingPipeline<M, T>>
where
    M: Model + Send + Sync + 'static,
    T: Tokenizer + Send + Sync + 'static,
{
    DocumentUnderstandingPipeline::new(model, tokenizer)
}

// ================================================================================================
// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use trustformers_core::traits::{Config as CoreConfig, TokenizedInput};
    use trustformers_core::Tensor;

    // ---- Minimal model/tokenizer stand-ins -------------------------------
    //
    // The document pipeline never consults either one (its work is all
    // text-side); they exist only to satisfy the generic bounds.

    #[derive(Debug, Default, Serialize, Deserialize)]
    struct StubConfig;

    impl CoreConfig for StubConfig {
        fn architecture(&self) -> &'static str {
            "stub"
        }
    }

    struct StubModel;

    impl Model for StubModel {
        type Input = Tensor;
        type Output = Tensor;
        type Config = StubConfig;

        fn forward(&self, input: Self::Input) -> trustformers_core::errors::Result<Self::Output> {
            Ok(input)
        }

        fn num_parameters(&self) -> usize {
            0
        }

        fn load_pretrained(
            &mut self,
            _reader: &mut dyn std::io::Read,
        ) -> trustformers_core::errors::Result<()> {
            Ok(())
        }

        fn get_config(&self) -> &Self::Config {
            &StubConfig
        }
    }

    struct StubTokenizer;

    impl Tokenizer for StubTokenizer {
        fn encode(&self, _text: &str) -> trustformers_core::errors::Result<TokenizedInput> {
            Ok(TokenizedInput::new(vec![0], vec![1]))
        }

        fn encode_pair(
            &self,
            _a: &str,
            _b: &str,
        ) -> trustformers_core::errors::Result<TokenizedInput> {
            Ok(TokenizedInput::new(vec![0], vec![1]))
        }

        fn decode(&self, _ids: &[u32]) -> trustformers_core::errors::Result<String> {
            Ok(String::new())
        }

        fn vocab_size(&self) -> usize {
            1
        }

        fn get_vocab(&self) -> HashMap<String, u32> {
            HashMap::new()
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }

        fn id_to_token(&self, _id: u32) -> Option<String> {
            None
        }
    }

    fn stub_pipeline() -> DocumentUnderstandingPipeline<StubModel, StubTokenizer> {
        DocumentUnderstandingPipeline::new(StubModel, StubTokenizer).expect("pipeline")
    }

    // ---- Honesty regressions ---------------------------------------------

    #[test]
    fn test_extract_text_reports_missing_ocr() {
        // Regression: `extract_text` used to return the fixed strings
        // "Document Header", "Main content paragraph …" and "Footer
        // information" for any image, and "Extracted text from PDF document"
        // for anything starting with %PDF.
        let pipeline = stub_pipeline();
        match pipeline.extract_text(b"%PDF-1.7 some bytes") {
            Err(TrustformersError::FeatureUnavailable {
                feature, message, ..
            }) => {
                assert_eq!(feature, "document-ocr");
                assert!(
                    message.contains("never returns canned"),
                    "message: {message}"
                );
            },
            other => panic!("expected a document-ocr error, got {other:?}"),
        }
        assert_eq!(pipeline.extract_text(&[]).expect("empty is empty"), "");
    }

    #[test]
    fn test_perform_ocr_reports_missing_engine() {
        // Regression: this returned `OCRResult { text: "Sample OCR text", .. }`.
        let pipeline = stub_pipeline();
        assert!(matches!(
            pipeline.perform_ocr(&[1, 2, 3]),
            Err(TrustformersError::FeatureUnavailable { .. })
        ));
        assert!(pipeline.perform_ocr(&[]).expect("empty").is_empty());
    }

    #[test]
    fn test_extract_layout_reports_missing_engine() {
        let pipeline = stub_pipeline();
        assert!(matches!(
            pipeline.extract_layout(&[1, 2, 3]),
            Err(TrustformersError::FeatureUnavailable { .. })
        ));
    }

    #[test]
    fn test_extract_tables_reports_missing_detector() {
        // Regression: this returned two invented tables (a "financial" one
        // totalling $145.00 and a "contact" one with three fake employees).
        let pipeline = stub_pipeline();
        assert!(matches!(
            pipeline.extract_tables(&[1, 2, 3]),
            Err(TrustformersError::FeatureUnavailable { .. })
        ));
    }

    #[test]
    fn test_answer_question_refuses_to_template_an_answer() {
        // Regression: this returned
        // `format!("Answer to '{}' based on document content", question)`.
        let pipeline = stub_pipeline();
        match pipeline.answer_question("some text", "What is the total?") {
            Err(TrustformersError::FeatureUnavailable {
                feature, message, ..
            }) => {
                assert_eq!(feature, "document-qa");
                assert!(
                    !message.contains("based on document content"),
                    "must not echo the old template: {message}"
                );
            },
            other => panic!("expected a document-qa error, got {other:?}"),
        }
    }

    #[test]
    fn test_pipeline_call_fails_for_a_real_document() {
        let pipeline = stub_pipeline();
        let input = DocumentUnderstandingInput {
            image: vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10],
            image_type: "image/jpeg".to_string(),
            question: Some("Who signed this?".to_string()),
            extraction_targets: None,
        };
        assert!(matches!(
            pipeline.__call__(input),
            Err(TrustformersError::FeatureUnavailable { .. })
        ));
    }

    // ---- Real text-side analysis -----------------------------------------

    #[test]
    fn test_extract_entities_finds_real_spans_only() {
        // Regression: this always returned exactly one `"John Doe"` PERSON
        // entity, whatever the text said.
        let pipeline = stub_pipeline();
        let text =
            "Contact ada@example.com or call 555-123-4567 before 2024-05-01. Total $1,234.56";
        let entities = pipeline.extract_entities(text).expect("entities");

        let types: Vec<&str> = entities.iter().map(|e| e.entity_type.as_str()).collect();
        assert!(types.contains(&"EMAIL"), "{types:?}");
        assert!(types.contains(&"PHONE"), "{types:?}");
        assert!(types.contains(&"DATE"), "{types:?}");
        assert!(types.contains(&"MONEY"), "{types:?}");
        assert!(
            !entities.iter().any(|e| e.text == "John Doe"),
            "the hardcoded fixture must be gone"
        );
        for e in &entities {
            assert!(
                text.contains(&e.text),
                "entity {:?} is not a substring of the input",
                e.text
            );
        }
    }

    #[test]
    fn test_extract_entities_returns_nothing_for_plain_prose() {
        let pipeline = stub_pipeline();
        let entities = pipeline
            .extract_entities("the quick brown fox jumps over the lazy dog")
            .expect("entities");
        assert!(entities.is_empty(), "unexpected entities: {entities:?}");
    }

    #[test]
    fn test_extract_key_value_pairs_reads_the_supplied_text() {
        let pipeline = stub_pipeline();
        let pairs = pipeline
            .extract_key_value_pairs("Name: Ada Lovelace\nEmail: ada@example.com")
            .expect("pairs");
        assert!(!pairs.is_empty(), "expected key-value pairs");
        assert!(
            pairs.iter().any(|p| p.value.contains("Ada Lovelace")),
            "pairs: {pairs:?}"
        );
    }

    #[test]
    fn test_analyze_text_leaves_ocr_only_fields_empty() {
        let pipeline = stub_pipeline();
        let output = pipeline.analyze_text("Email: ada@example.com").expect("analysis");
        assert!(output.text.is_some());
        assert!(output.entities.is_some_and(|e| !e.is_empty()));
        assert!(output.text_blocks.is_none(), "no layout model exists");
        assert!(output.ocr_results.is_none(), "no OCR engine exists");
        assert!(output.tables.is_none(), "no table detector exists");
        assert!(output.answer.is_none(), "no QA model exists");
        assert!(
            output.metadata.detected_language.is_none(),
            "language must not be hardcoded to `en`"
        );
        assert!(
            output.metadata.quality_score.is_none(),
            "quality score must not be invented"
        );
    }

    #[test]
    fn test_sort_reading_order_is_top_to_bottom_then_left_to_right() {
        let pipeline = stub_pipeline();
        let block = |text: &str, x: f32, y: f32| TextBlock {
            text: text.to_string(),
            bounding_box: BoundingBox {
                x,
                y,
                width: 10.0,
                height: 10.0,
            },
            confidence: 0.9,
            block_type: TextBlockType::Paragraph,
        };
        let sorted = pipeline.sort_reading_order(vec![
            block("c", 10.0, 200.0),
            block("b", 90.0, 10.0),
            block("a", 10.0, 12.0),
        ]);
        let order: Vec<&str> = sorted.iter().map(|b| b.text.as_str()).collect();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    // --- DocumentUnderstandingConfig tests ---

    #[test]
    fn test_config_default_values() {
        let config = DocumentUnderstandingConfig::default();
        assert_eq!(config.max_length, 512, "default max_length should be 512");
        assert!(
            config.return_ocr_results,
            "default should return OCR results"
        );
        assert!(config.return_layout, "default should return layout");
        assert!(
            config.return_key_value_pairs,
            "default should return key-value pairs"
        );
        assert!(config.return_entities, "default should return entities");
        assert!(config.return_text, "default should return text");
        assert!(config.preprocess_text, "default should preprocess text");
    }

    #[test]
    fn test_config_confidence_threshold_default_in_range() {
        let config = DocumentUnderstandingConfig::default();
        assert!(
            config.confidence_threshold >= 0.0 && config.confidence_threshold <= 1.0,
            "confidence_threshold should be in [0.0, 1.0], got {}",
            config.confidence_threshold
        );
    }

    #[test]
    fn test_config_language_hints_default_contains_english() {
        let config = DocumentUnderstandingConfig::default();
        assert!(
            config.language_hints.contains(&"en".to_string()),
            "default language_hints should contain 'en'"
        );
    }

    // --- BoundingBox tests ---

    #[test]
    fn test_bounding_box_construction() {
        let bbox = BoundingBox {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 50.0,
        };
        assert!((bbox.x - 10.0).abs() < 1e-6);
        assert!((bbox.y - 20.0).abs() < 1e-6);
        assert!((bbox.width - 100.0).abs() < 1e-6);
        assert!((bbox.height - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_bounding_box_dimensions_non_negative() {
        let bbox = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: 50.0,
            height: 30.0,
        };
        assert!(
            bbox.width >= 0.0,
            "bounding box width should be non-negative"
        );
        assert!(
            bbox.height >= 0.0,
            "bounding box height should be non-negative"
        );
    }

    // --- TextBlock tests ---

    #[test]
    fn test_text_block_confidence_in_range() {
        let block = TextBlock {
            text: "Sample paragraph text".to_string(),
            bounding_box: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 200.0,
                height: 30.0,
            },
            confidence: 0.88,
            block_type: TextBlockType::Paragraph,
        };
        assert!(
            block.confidence >= 0.0 && block.confidence <= 1.0,
            "confidence must be in [0.0, 1.0]"
        );
    }

    #[test]
    fn test_text_block_heading_type() {
        let block = TextBlock {
            text: "Chapter 1: Introduction".to_string(),
            bounding_box: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 300.0,
                height: 40.0,
            },
            confidence: 0.95,
            block_type: TextBlockType::Heading,
        };
        assert!(
            matches!(block.block_type, TextBlockType::Heading),
            "block_type should be Heading"
        );
    }

    #[test]
    fn test_text_block_title_type() {
        let block = TextBlock {
            text: "Annual Report 2024".to_string(),
            bounding_box: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 400.0,
                height: 60.0,
            },
            confidence: 0.97,
            block_type: TextBlockType::Title,
        };
        assert!(matches!(block.block_type, TextBlockType::Title));
    }

    // --- Table tests ---

    #[test]
    fn test_table_row_col_count() {
        let headers = vec!["Name".to_string(), "Value".to_string()];
        let rows = vec![
            vec!["Row1".to_string(), "100".to_string()],
            vec!["Row2".to_string(), "200".to_string()],
            vec!["Row3".to_string(), "300".to_string()],
        ];
        let table = Table {
            rows: rows.clone(),
            headers: Some(headers),
            bounding_box: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 300.0,
                height: 100.0,
            },
            confidence: 0.92,
        };
        assert_eq!(table.rows.len(), 3, "table should have 3 rows");
        assert_eq!(table.rows[0].len(), 2, "each row should have 2 columns");
    }

    #[test]
    fn test_table_headers_present() {
        let headers = vec!["Item".to_string(), "Qty".to_string(), "Price".to_string()];
        let table = Table {
            rows: vec![headers.clone()],
            headers: Some(headers.clone()),
            bounding_box: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 400.0,
                height: 200.0,
            },
            confidence: 0.90,
        };
        assert!(table.headers.is_some(), "table should have headers");
        assert_eq!(
            table.headers.as_ref().expect("headers present").len(),
            3,
            "table should have 3 column headers"
        );
    }

    #[test]
    fn test_table_confidence_in_range() {
        let table = Table {
            rows: vec![vec!["data".to_string()]],
            headers: None,
            bounding_box: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 50.0,
            },
            confidence: 0.85,
        };
        assert!(
            table.confidence >= 0.0 && table.confidence <= 1.0,
            "table confidence must be in [0.0, 1.0]"
        );
    }

    // --- OCRResult tests ---

    #[test]
    fn test_ocr_result_confidence_threshold() {
        let ocr = OCRResult {
            text: "Extracted text here".to_string(),
            bounding_box: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 200.0,
                height: 25.0,
            },
            confidence: 0.92,
            word_level_boxes: None,
        };
        let threshold = 0.5;
        assert!(
            ocr.confidence >= threshold,
            "OCR result with confidence {} should pass threshold {}",
            ocr.confidence,
            threshold
        );
    }

    #[test]
    fn test_ocr_result_with_word_boxes() {
        let ocr = OCRResult {
            text: "Sample OCR".to_string(),
            bounding_box: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 150.0,
                height: 25.0,
            },
            confidence: 0.95,
            word_level_boxes: Some(vec![
                (
                    "Sample".to_string(),
                    BoundingBox {
                        x: 0.0,
                        y: 0.0,
                        width: 70.0,
                        height: 25.0,
                    },
                ),
                (
                    "OCR".to_string(),
                    BoundingBox {
                        x: 75.0,
                        y: 0.0,
                        width: 50.0,
                        height: 25.0,
                    },
                ),
            ]),
        };
        let boxes = ocr.word_level_boxes.as_ref().expect("word level boxes should be present");
        assert_eq!(boxes.len(), 2, "should have 2 word-level bounding boxes");
    }

    // --- KeyValuePair tests ---

    #[test]
    fn test_key_value_pair_fields() {
        let kv = KeyValuePair {
            key: "Invoice Number".to_string(),
            value: "INV-12345".to_string(),
            key_bbox: BoundingBox {
                x: 10.0,
                y: 50.0,
                width: 100.0,
                height: 20.0,
            },
            value_bbox: BoundingBox {
                x: 120.0,
                y: 50.0,
                width: 80.0,
                height: 20.0,
            },
            confidence: 0.88,
        };
        assert_eq!(kv.key, "Invoice Number");
        assert_eq!(kv.value, "INV-12345");
        assert!(kv.confidence >= 0.0 && kv.confidence <= 1.0);
    }

    // --- DocumentMetadata tests ---

    #[test]
    fn test_document_metadata_quality_score_in_range() {
        let meta = DocumentMetadata {
            page_count: 1,
            processing_time_ms: 150,
            detected_language: Some("en".to_string()),
            text_orientation: Some(0.0),
            quality_score: Some(0.92),
        };
        let score = meta.quality_score.expect("quality score present");
        assert!(
            (0.0..=1.0).contains(&score),
            "quality_score must be in [0.0, 1.0]"
        );
    }

    #[test]
    fn test_document_metadata_page_count_positive() {
        let meta = DocumentMetadata {
            page_count: 5,
            processing_time_ms: 500,
            detected_language: Some("en".to_string()),
            text_orientation: Some(0.0),
            quality_score: Some(0.85),
        };
        assert!(meta.page_count > 0, "page_count should be at least 1");
    }

    // --- DocumentUnderstandingOutput tests ---

    #[test]
    fn test_document_understanding_output_construction() {
        let output = DocumentUnderstandingOutput {
            text: Some("Sample document text".to_string()),
            text_blocks: None,
            key_value_pairs: None,
            entities: None,
            tables: None,
            ocr_results: None,
            answer: None,
            metadata: DocumentMetadata {
                page_count: 1,
                processing_time_ms: 200,
                detected_language: None,
                text_orientation: None,
                quality_score: None,
            },
        };
        assert!(output.text.is_some(), "output should have text");
        assert_eq!(output.metadata.page_count, 1);
    }

    // --- Reading order / layout order tests ---

    #[test]
    fn test_layout_reading_order_top_to_bottom() {
        // Simulate multiple text blocks and verify they can be sorted top-to-bottom
        let blocks = vec![
            TextBlock {
                text: "Header text".to_string(),
                bounding_box: BoundingBox {
                    x: 10.0,
                    y: 10.0,
                    width: 500.0,
                    height: 30.0,
                },
                confidence: 0.95,
                block_type: TextBlockType::Header,
            },
            TextBlock {
                text: "Body text paragraph".to_string(),
                bounding_box: BoundingBox {
                    x: 10.0,
                    y: 100.0,
                    width: 500.0,
                    height: 60.0,
                },
                confidence: 0.90,
                block_type: TextBlockType::Paragraph,
            },
            TextBlock {
                text: "Footer text".to_string(),
                bounding_box: BoundingBox {
                    x: 10.0,
                    y: 900.0,
                    width: 500.0,
                    height: 20.0,
                },
                confidence: 0.85,
                block_type: TextBlockType::Footer,
            },
        ];
        // Sort by y-coordinate (top-to-bottom reading order)
        let mut sorted = blocks.clone();
        sorted.sort_by(|a, b| {
            a.bounding_box
                .y
                .partial_cmp(&b.bounding_box.y)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        assert_eq!(
            sorted[0].bounding_box.y, 10.0,
            "first block should have smallest y"
        );
        assert_eq!(
            sorted[2].bounding_box.y, 900.0,
            "last block should have largest y"
        );
    }

    #[test]
    fn test_layout_reading_order_left_to_right() {
        // Two columns at same height should be sorted left-to-right
        let left_block = TextBlock {
            text: "Left column".to_string(),
            bounding_box: BoundingBox {
                x: 10.0,
                y: 100.0,
                width: 200.0,
                height: 50.0,
            },
            confidence: 0.90,
            block_type: TextBlockType::Paragraph,
        };
        let right_block = TextBlock {
            text: "Right column".to_string(),
            bounding_box: BoundingBox {
                x: 300.0,
                y: 100.0,
                width: 200.0,
                height: 50.0,
            },
            confidence: 0.88,
            block_type: TextBlockType::Paragraph,
        };
        // Left block should come before right block in reading order
        assert!(
            left_block.bounding_box.x < right_block.bounding_box.x,
            "left column x ({}) should be less than right column x ({})",
            left_block.bounding_box.x,
            right_block.bounding_box.x
        );
    }

    // --- TextBlockType variants test ---

    #[test]
    fn test_text_block_type_variants_accessible() {
        let variants = [
            TextBlockType::Title,
            TextBlockType::Heading,
            TextBlockType::Paragraph,
            TextBlockType::List,
            TextBlockType::Table,
            TextBlockType::Footer,
            TextBlockType::Header,
            TextBlockType::Caption,
            TextBlockType::Other,
        ];
        // H1/H2/H3 hierarchy via heading level detection can be represented
        // via the same Heading variant. Verify Heading is among variants.
        let has_heading = variants.iter().any(|v| matches!(v, TextBlockType::Heading));
        assert!(
            has_heading,
            "TextBlockType should include Heading variant for H1/H2/H3 detection"
        );
    }
}
