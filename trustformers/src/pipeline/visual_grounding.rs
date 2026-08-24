//! # Visual Grounding Pipeline
//!
//! ## What is real here
//!
//! Query parsing ([`parse_phrases`], which splits a GroundingDINO-style prompt
//! on `.` and `,`) and the post-processing chain in
//! [`VisualGroundingPipeline::postprocess`]: score thresholding, descending
//! sort and truncation to `max_detections`.
//!
//! ## Model support
//!
//! No open-set grounding backbone (GroundingDINO, …) is implemented in
//! `trustformers-models`, so [`VisualGroundingPipeline::ground`] returns
//! [`GroundingError::UnsupportedModel`] instead of the boxes it used to derive
//! from a djb2 hash of the phrase while ignoring the image entirely.
//!
//! ## Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::visual_grounding::{VisualGroundingConfig, VisualGroundingPipeline};
//!
//! let config = VisualGroundingConfig::default();
//! let pipeline = VisualGroundingPipeline::new(config)?;
//! // Real post-processing over boxes produced by your own model:
//! let result = pipeline.postprocess(my_boxes, "a cat . a dog", 800, 1333);
//! for b in &result.boxes {
//!     println!("{}: {:.3}", b.phrase, b.score);
//! }
//! # Ok::<(), trustformers::pipeline::visual_grounding::GroundingError>(())
//! ```

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the visual grounding pipeline.
#[derive(Debug, thiserror::Error)]
pub enum GroundingError {
    /// The image slice was empty.
    #[error("Empty image")]
    EmptyImage,
    /// The text query was empty or contained only whitespace.
    #[error("Empty text query")]
    EmptyQuery,
    /// A generic model-level error with a descriptive message.
    #[error("Model error: {0}")]
    ModelError(String),
    /// The requested checkpoint has no real implementation in this workspace.
    #[error(
        "no real visual grounding model is implemented for `{requested}`; supported: \
         {supported}. This pipeline never returns synthesised boxes — use `postprocess` with \
         your own model's output."
    )]
    UnsupportedModel {
        /// The checkpoint or architecture the caller asked for.
        requested: String,
        /// Comma-separated list of architectures that *are* supported.
        supported: String,
    },
}

/// Grounding architectures with a real backbone in this workspace.
///
/// Deliberately empty — the pipeline says so rather than pretending.
const SUPPORTED_ARCHITECTURES: &[&str] = &[];

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`VisualGroundingPipeline`].
#[derive(Debug, Clone)]
pub struct VisualGroundingConfig {
    /// HuggingFace model identifier or local path.
    pub model_name: String,
    /// Minimum confidence required to keep a predicted box.
    pub box_threshold: f32,
    /// Minimum text-region similarity required to keep a box.
    pub text_threshold: f32,
    /// Maximum number of predicted boxes to return per image.
    pub max_detections: usize,
    /// Expected model input size as `(height, width)`.
    pub input_size: (usize, usize),
}

impl Default for VisualGroundingConfig {
    fn default() -> Self {
        Self {
            model_name: "IDEA-Research/grounding-dino-tiny".to_string(),
            box_threshold: 0.3,
            text_threshold: 0.25,
            max_detections: 256,
            input_size: (800, 1333),
        }
    }
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// A grounded detection: a text phrase associated with a normalised bounding box.
#[derive(Debug, Clone)]
pub struct GroundedBox {
    /// The text phrase that produced this detection.
    pub phrase: String,
    /// Normalised bounding box `(x1, y1, x2, y2)` in `[0.0, 1.0]`.
    pub bbox: (f32, f32, f32, f32),
    /// Overall detection confidence.
    pub score: f32,
    /// Similarity between the phrase and the detected region.
    pub phrase_score: f32,
}

impl GroundedBox {
    /// Area of the bounding box (in normalised coordinates).
    pub fn area(&self) -> f32 {
        let (x1, y1, x2, y2) = self.bbox;
        let w = (x2 - x1).max(0.0);
        let h = (y2 - y1).max(0.0);
        w * h
    }

    /// Centre point of the bounding box.
    pub fn center(&self) -> (f32, f32) {
        let (x1, y1, x2, y2) = self.bbox;
        ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    }

    /// Intersection-over-Union with another [`GroundedBox`].
    pub fn iou(&self, other: &GroundedBox) -> f32 {
        let (x1, y1, x2, y2) = self.bbox;
        let (ox1, oy1, ox2, oy2) = other.bbox;

        let ix1 = x1.max(ox1);
        let iy1 = y1.max(oy1);
        let ix2 = x2.min(ox2);
        let iy2 = y2.min(oy2);

        let inter_w = (ix2 - ix1).max(0.0);
        let inter_h = (iy2 - iy1).max(0.0);
        let intersection = inter_w * inter_h;

        let union = self.area() + other.area() - intersection;
        if union < f32::EPSILON {
            0.0
        } else {
            intersection / union
        }
    }
}

/// Grounding result for a single image-query pair.
#[derive(Debug, Clone)]
pub struct GroundingResult {
    /// All grounded boxes for this image.
    pub boxes: Vec<GroundedBox>,
    /// The text query that was used.
    pub query: String,
    /// Height of the source image in pixels.
    pub image_height: usize,
    /// Width of the source image in pixels.
    pub image_width: usize,
}

impl GroundingResult {
    /// Return a copy keeping only boxes whose `score >= threshold`.
    pub fn filter_by_score(&self, threshold: f32) -> Self {
        let boxes: Vec<GroundedBox> =
            self.boxes.iter().filter(|b| b.score >= threshold).cloned().collect();
        GroundingResult {
            boxes,
            query: self.query.clone(),
            image_height: self.image_height,
            image_width: self.image_width,
        }
    }

    /// Return a copy keeping only the top-`k` boxes sorted by score descending.
    pub fn top_k(&self, k: usize) -> Self {
        let mut sorted = self.boxes.clone();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(k);
        GroundingResult {
            boxes: sorted,
            query: self.query.clone(),
            image_height: self.image_height,
            image_width: self.image_width,
        }
    }

    /// Return the deduplicated set of phrase strings from all boxes.
    pub fn unique_phrases(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for b in &self.boxes {
            if seen.insert(b.phrase.clone()) {
                out.push(b.phrase.clone());
            }
        }
        out
    }

    /// Split `query` (the original GroundingDINO-style prompt, e.g.
    /// `"a cat . a dog"`) into the individual phrases that were requested.
    ///
    /// This is [`unique_phrases`](Self::unique_phrases)'s counterpart on the
    /// *input* side: `unique_phrases` reports what the (external) model
    /// actually returned boxes for, `requested_phrases` reports what was
    /// asked for. Comparing the two can surface a model that grounded a
    /// phrase nobody requested, or requested phrases it never answered.
    pub fn requested_phrases(&self) -> Vec<String> {
        parse_phrases(&self.query)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// djb2 hash — deterministic numeric fingerprint for strings.
fn djb2_hash(s: &str) -> u64 {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    h
}

/// Parse free-form text into a list of trimmed, non-empty phrases.
///
/// Phrases are separated by `.` or `,`.
fn parse_phrases(text: &str) -> Vec<String> {
    text.split(['.', ','])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Pipeline for visual grounding (GroundingDINO style).
pub struct VisualGroundingPipeline {
    config: VisualGroundingConfig,
}

impl VisualGroundingPipeline {
    /// Create a new pipeline with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration is fundamentally invalid.
    pub fn new(config: VisualGroundingConfig) -> Result<Self, GroundingError> {
        Ok(Self { config })
    }

    /// Ground a text query in a single image.
    ///
    /// # Errors
    ///
    /// - [`GroundingError::EmptyImage`] — `image` is empty.
    /// - [`GroundingError::EmptyQuery`] — `text_query` is blank.
    /// - [`GroundingError::UnsupportedModel`] — otherwise: no grounding
    ///   backbone is implemented and this pipeline will not invent boxes.
    pub fn ground(
        &self,
        image: &[f32],
        _height: usize,
        _width: usize,
        text_query: &str,
    ) -> Result<GroundingResult, GroundingError> {
        if image.is_empty() {
            return Err(GroundingError::EmptyImage);
        }
        let trimmed = text_query.trim();
        if trimmed.is_empty() {
            return Err(GroundingError::EmptyQuery);
        }
        Err(GroundingError::UnsupportedModel {
            requested: self.config.model_name.clone(),
            supported: if SUPPORTED_ARCHITECTURES.is_empty() {
                "none (no grounding backbone is implemented yet)".to_string()
            } else {
                SUPPORTED_ARCHITECTURES.join(", ")
            },
        })
    }

    /// Apply the pipeline's real post-processing to boxes from your own model.
    ///
    /// Drops boxes scoring below `box_threshold`, sorts the rest by descending
    /// score, and truncates to `max_detections`.
    pub fn postprocess(
        &self,
        mut boxes: Vec<GroundedBox>,
        query: &str,
        image_height: usize,
        image_width: usize,
    ) -> GroundingResult {
        boxes.retain(|b| b.score >= self.config.box_threshold);
        boxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        boxes.truncate(self.config.max_detections);
        GroundingResult {
            boxes,
            query: query.trim().to_string(),
            image_height,
            image_width,
        }
    }

    /// Ground the same text query across a batch of images.
    ///
    /// Each element of `images` is `(pixel_data, height, width)`.
    ///
    /// # Errors
    ///
    /// Fails fast on the first error encountered.
    pub fn ground_batch(
        &self,
        images: &[(&[f32], usize, usize)],
        text_query: &str,
    ) -> Result<Vec<GroundingResult>, GroundingError> {
        images.iter().map(|(img, h, w)| self.ground(img, *h, *w, text_query)).collect()
    }

    /// Access the pipeline configuration.
    pub fn config(&self) -> &VisualGroundingConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Extended types and processor
// ---------------------------------------------------------------------------

/// A normalised bounding box `(x1, y1, x2, y2)` in `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BoundingBox {
    /// Area of this bounding box.
    pub fn area(&self) -> f32 {
        let w = (self.x2 - self.x1).max(0.0);
        let h = (self.y2 - self.y1).max(0.0);
        w * h
    }

    /// Intersection-over-Union with another bounding box.
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);
        let inter_w = (ix2 - ix1).max(0.0);
        let inter_h = (iy2 - iy1).max(0.0);
        let intersection = inter_w * inter_h;
        let union = self.area() + other.area() - intersection;
        if union < f32::EPSILON {
            0.0
        } else {
            intersection / union
        }
    }
}

/// Grounding result pairing a phrase with a bounding box and score.
#[derive(Debug, Clone)]
pub struct GroundingResultNew {
    pub phrase: String,
    pub bbox: BoundingBox,
    pub score: f32,
}

/// Input for the grounding pipeline.
#[derive(Debug, Clone)]
pub struct GroundingInput {
    /// Raw image bytes (RGB, row-major).
    pub image: Vec<u8>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Phrases to localise.
    pub phrases: Vec<String>,
}

/// Grounding processor providing pure-Rust helpers.
pub struct GroundingProcessor;

impl GroundingProcessor {
    /// Word-level tokenization; unknown words get a djb2 hash id.
    pub fn encode_phrase(phrase: &str) -> Vec<u32> {
        phrase
            .split_whitespace()
            .map(|word| {
                let lower = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
                (djb2_hash(&lower) % 30_000) as u32 + 1
            })
            .collect()
    }

    /// Cosine similarity between an image patch (f32 values) and a phrase encoding (u32 ids).
    ///
    /// The phrase encoding is cast to f32 for the dot product.
    pub fn score_region(image_patch: &[f32], phrase_encoding: &[u32]) -> f32 {
        if image_patch.is_empty() || phrase_encoding.is_empty() {
            return 0.0;
        }
        let phrase_f32: Vec<f32> = phrase_encoding.iter().map(|&id| id as f32).collect();

        let len = image_patch.len().min(phrase_f32.len());
        let dot: f32 = image_patch[..len]
            .iter()
            .zip(phrase_f32[..len].iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f32 = image_patch.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_b: f32 = phrase_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
            0.0
        } else {
            (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
        }
    }

    /// Extract a rectangular patch from a flat RGB byte image.
    ///
    /// Returns an empty `Vec` if any index is out of range.
    pub fn extract_image_patch(
        image: &[u8],
        img_w: usize,
        img_h: usize,
        bbox: &BoundingBox,
    ) -> Vec<u8> {
        let x1 = (bbox.x1 * img_w as f32) as usize;
        let y1 = (bbox.y1 * img_h as f32) as usize;
        let x2 = ((bbox.x2 * img_w as f32) as usize).min(img_w);
        let y2 = ((bbox.y2 * img_h as f32) as usize).min(img_h);

        if x2 <= x1 || y2 <= y1 {
            return Vec::new();
        }

        let channels = 3_usize;
        let mut patch = Vec::new();
        for row in y1..y2 {
            for col in x1..x2 {
                let base = (row * img_w + col) * channels;
                if base + channels <= image.len() {
                    patch.extend_from_slice(&image[base..base + channels]);
                }
            }
        }
        patch
    }

    /// Generate sliding-window bounding box proposals.
    ///
    /// For each `size` in `sizes`, slides a window of that pixel size
    /// across the image by `step` pixels in both dimensions.
    ///
    /// All coordinates are returned normalised to `[0, 1]`.
    pub fn sliding_window_proposals(
        width: usize,
        height: usize,
        step: usize,
        sizes: &[usize],
    ) -> Vec<BoundingBox> {
        if width == 0 || height == 0 || step == 0 || sizes.is_empty() {
            return Vec::new();
        }
        let mut proposals = Vec::new();
        for &sz in sizes {
            if sz == 0 || sz > width || sz > height {
                continue;
            }
            let mut y = 0_usize;
            while y + sz <= height {
                let mut x = 0_usize;
                while x + sz <= width {
                    proposals.push(BoundingBox {
                        x1: x as f32 / width as f32,
                        y1: y as f32 / height as f32,
                        x2: (x + sz) as f32 / width as f32,
                        y2: (y + sz) as f32 / height as f32,
                    });
                    x += step;
                }
                y += step;
            }
        }
        proposals
    }
}

/// Phrase grounding evaluation metrics.
pub struct PhraseGroundingMetrics;

impl PhraseGroundingMetrics {
    /// Recall@IoU: fraction of ground-truth boxes that have at least one prediction
    /// with IoU >= `iou_threshold`.
    pub fn recall_at_iou(
        predictions: &[GroundingResultNew],
        ground_truth: &[BoundingBox],
        iou_threshold: f32,
    ) -> f32 {
        if ground_truth.is_empty() {
            return 1.0;
        }
        let matched = ground_truth
            .iter()
            .filter(|gt| predictions.iter().any(|pred| pred.bbox.iou(gt) >= iou_threshold));
        matched.count() as f32 / ground_truth.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pipeline() -> VisualGroundingPipeline {
        VisualGroundingPipeline::new(VisualGroundingConfig::default())
            .expect("default config valid")
    }

    fn dummy_image(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 / n as f32).collect()
    }

    // --- GroundedBox geometry ---

    #[test]
    fn test_grounded_box_area() {
        let b = GroundedBox {
            phrase: "cat".to_string(),
            bbox: (0.1, 0.1, 0.5, 0.6),
            score: 0.9,
            phrase_score: 0.8,
        };
        let expected = (0.5 - 0.1) * (0.6 - 0.1);
        assert!((b.area() - expected).abs() < 1e-6, "area was {}", b.area());
    }

    #[test]
    fn test_grounded_box_center() {
        let b = GroundedBox {
            phrase: "dog".to_string(),
            bbox: (0.0, 0.0, 1.0, 1.0),
            score: 0.8,
            phrase_score: 0.7,
        };
        let (cx, cy) = b.center();
        assert!((cx - 0.5).abs() < 1e-6);
        assert!((cy - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_grounded_box_iou_identical() {
        let b = GroundedBox {
            phrase: "bird".to_string(),
            bbox: (0.1, 0.1, 0.5, 0.5),
            score: 0.9,
            phrase_score: 0.8,
        };
        let iou = b.iou(&b);
        assert!((iou - 1.0).abs() < 1e-5, "iou of identical box was {iou}");
    }

    // --- GroundingResult filters ---

    #[test]
    fn test_filter_by_score() {
        let boxes = vec![
            GroundedBox {
                phrase: "a".to_string(),
                bbox: (0.0, 0.0, 0.1, 0.1),
                score: 0.8,
                phrase_score: 0.7,
            },
            GroundedBox {
                phrase: "b".to_string(),
                bbox: (0.2, 0.2, 0.3, 0.3),
                score: 0.2,
                phrase_score: 0.3,
            },
        ];
        let result = GroundingResult {
            boxes,
            query: "a . b".to_string(),
            image_height: 100,
            image_width: 100,
        };
        let filtered = result.filter_by_score(0.5);
        assert_eq!(filtered.boxes.len(), 1);
        assert_eq!(filtered.boxes[0].phrase, "a");
    }

    #[test]
    fn test_top_k() {
        let boxes: Vec<GroundedBox> = ["x", "y", "z"]
            .iter()
            .enumerate()
            .map(|(i, lbl)| GroundedBox {
                phrase: lbl.to_string(),
                bbox: (0.0, 0.0, 0.1, 0.1),
                score: (i + 1) as f32 * 0.2,
                phrase_score: 0.5,
            })
            .collect();
        let result = GroundingResult {
            boxes,
            query: "x . y . z".to_string(),
            image_height: 100,
            image_width: 100,
        };
        let top = result.top_k(2);
        assert_eq!(top.boxes.len(), 2);
        // Scores should be sorted descending.
        assert!(top.boxes[0].score >= top.boxes[1].score);
    }

    #[test]
    fn test_unique_phrases_dedup() {
        let boxes: Vec<GroundedBox> = vec!["cat", "dog", "cat", "bird", "dog"]
            .into_iter()
            .map(|p| GroundedBox {
                phrase: p.to_string(),
                bbox: (0.0, 0.0, 0.1, 0.1),
                score: 0.9,
                phrase_score: 0.8,
            })
            .collect();
        let result = GroundingResult {
            boxes,
            query: "cat . dog . cat . bird . dog".to_string(),
            image_height: 100,
            image_width: 100,
        };
        let phrases = result.unique_phrases();
        assert_eq!(
            phrases.len(),
            3,
            "expected 3 unique phrases, got {:?}",
            phrases
        );
    }

    #[test]
    fn test_requested_phrases_parses_the_query_not_the_boxes() {
        // The model answered for "cat" only, but the query asked for both
        // "cat" and "dog" - requested_phrases must report what was asked,
        // independent of unique_phrases (what was answered).
        let boxes = vec![GroundedBox {
            phrase: "cat".to_string(),
            bbox: (0.0, 0.0, 0.1, 0.1),
            score: 0.9,
            phrase_score: 0.8,
        }];
        let result = GroundingResult {
            boxes,
            query: "a cat . a dog".to_string(),
            image_height: 100,
            image_width: 100,
        };
        assert_eq!(
            result.requested_phrases(),
            vec!["a cat".to_string(), "a dog".to_string()]
        );
        assert_eq!(result.unique_phrases(), vec!["cat".to_string()]);
    }

    // --- Pipeline::ground ---

    #[test]
    fn test_ground_reports_unsupported_model() {
        // Regression: `ground` used to derive boxes from a djb2 hash of the
        // phrase, ignoring the image, and report them as detections.
        let config = VisualGroundingConfig {
            box_threshold: 0.0,
            model_name: "IDEA-Research/grounding-dino-tiny".to_string(),
            ..Default::default()
        };
        let p = VisualGroundingPipeline::new(config).expect("valid");
        let img = dummy_image(800 * 600 * 3);
        match p.ground(&img, 800, 600, "a cat . a dog . a bird") {
            Err(GroundingError::UnsupportedModel {
                requested,
                supported,
            }) => {
                assert_eq!(requested, "IDEA-Research/grounding-dino-tiny");
                assert!(supported.contains("none"), "supported: {supported}");
            },
            other => panic!("expected UnsupportedModel, got {other:?}"),
        }
    }

    #[test]
    fn test_ground_batch_reports_unsupported_model() {
        let p = default_pipeline();
        let img1 = dummy_image(100 * 100 * 3);
        let img2 = dummy_image(200 * 200 * 3);
        let images: Vec<(&[f32], usize, usize)> = vec![(&img1, 100, 100), (&img2, 200, 200)];
        assert!(matches!(
            p.ground_batch(&images, "cat"),
            Err(GroundingError::UnsupportedModel { .. })
        ));
    }

    #[test]
    fn test_postprocess_filters_sorts_and_truncates() {
        let config = VisualGroundingConfig {
            box_threshold: 0.4,
            max_detections: 2,
            ..Default::default()
        };
        let p = VisualGroundingPipeline::new(config).expect("valid");
        let make = |phrase: &str, score: f32| GroundedBox {
            phrase: phrase.to_string(),
            bbox: (0.1, 0.1, 0.5, 0.5),
            score,
            phrase_score: 0.5,
        };
        let result = p.postprocess(
            vec![
                make("cat", 0.9),
                make("dog", 0.2),
                make("bird", 0.7),
                make("tree", 0.6),
            ],
            " a cat . a dog ",
            800,
            600,
        );
        assert_eq!(result.boxes.len(), 2, "{:?}", result.boxes);
        assert_eq!(result.boxes[0].phrase, "cat");
        assert_eq!(result.boxes[1].phrase, "bird");
        assert_eq!(result.query, "a cat . a dog");
        assert_eq!(result.image_height, 800);
        assert_eq!(result.image_width, 600);
    }

    // --- Error cases ---

    #[test]
    fn test_empty_image_error() {
        let p = default_pipeline();
        let err = p.ground(&[], 100, 100, "cat").expect_err("empty image should fail");
        assert!(matches!(err, GroundingError::EmptyImage));
    }

    #[test]
    fn test_empty_query_error() {
        let p = default_pipeline();
        let img = dummy_image(100);
        let err = p.ground(&img, 10, 10, "   ").expect_err("empty query should fail");
        assert!(matches!(err, GroundingError::EmptyQuery));
    }

    // --- Bbox / score invariants ---

    // -----------------------------------------------------------------------
    // BoundingBox extended type tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bounding_box_area() {
        let bb = BoundingBox {
            x1: 0.1,
            y1: 0.1,
            x2: 0.5,
            y2: 0.6,
        };
        let expected = (0.5 - 0.1) * (0.6 - 0.1);
        assert!(
            (bb.area() - expected).abs() < 1e-6,
            "area was {}",
            bb.area()
        );
    }

    #[test]
    fn test_bounding_box_iou_identical() {
        let bb = BoundingBox {
            x1: 0.1,
            y1: 0.1,
            x2: 0.5,
            y2: 0.5,
        };
        assert!(
            (bb.iou(&bb) - 1.0).abs() < 1e-5,
            "iou of identical box should be 1.0"
        );
    }

    #[test]
    fn test_bounding_box_iou_no_overlap() {
        let a = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 0.3,
            y2: 0.3,
        };
        let b = BoundingBox {
            x1: 0.5,
            y1: 0.5,
            x2: 0.8,
            y2: 0.8,
        };
        assert!(
            (a.iou(&b)).abs() < 1e-6,
            "non-overlapping boxes should have iou ~0"
        );
    }

    #[test]
    fn test_bounding_box_iou_partial_overlap() {
        let a = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 0.6,
            y2: 0.6,
        };
        let b = BoundingBox {
            x1: 0.4,
            y1: 0.4,
            x2: 1.0,
            y2: 1.0,
        };
        let iou = a.iou(&b);
        assert!(
            iou > 0.0 && iou < 1.0,
            "partial overlap iou should be in (0,1), got {iou}"
        );
    }

    // -----------------------------------------------------------------------
    // GroundingProcessor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_phrase_nonempty() {
        let tokens = GroundingProcessor::encode_phrase("a black cat sitting");
        assert_eq!(tokens.len(), 4);
        for &t in &tokens {
            assert!(t > 0, "token id should be > 0");
        }
    }

    #[test]
    fn test_encode_phrase_empty() {
        let tokens = GroundingProcessor::encode_phrase("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_encode_phrase_deterministic() {
        let t1 = GroundingProcessor::encode_phrase("the red car");
        let t2 = GroundingProcessor::encode_phrase("the red car");
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_score_region_same_patch() {
        // Cosine similarity of a vector with itself should be 1.0
        let patch: Vec<f32> = (1..=4).map(|i| i as f32).collect();
        let phrase: Vec<u32> = (1..=4).collect();
        let sim = GroundingProcessor::score_region(&patch, &phrase);
        assert!(
            sim > 0.9,
            "self-similarity should be close to 1.0, got {sim}"
        );
    }

    #[test]
    fn test_score_region_empty_inputs() {
        assert_eq!(GroundingProcessor::score_region(&[], &[1, 2]), 0.0);
        assert_eq!(GroundingProcessor::score_region(&[1.0, 2.0], &[]), 0.0);
    }

    #[test]
    fn test_extract_image_patch_basic() {
        // 4x4 RGB image
        let image: Vec<u8> = (0..(4 * 4 * 3)).map(|i| i as u8).collect();
        let bbox = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 0.5,
            y2: 0.5,
        };
        let patch = GroundingProcessor::extract_image_patch(&image, 4, 4, &bbox);
        // 2x2 pixels * 3 channels = 12 bytes
        assert_eq!(patch.len(), 12, "expected 12 bytes, got {}", patch.len());
    }

    #[test]
    fn test_extract_image_patch_full_image() {
        let image: Vec<u8> = vec![255u8; 3 * 3 * 3];
        let bbox = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
        };
        let patch = GroundingProcessor::extract_image_patch(&image, 3, 3, &bbox);
        assert_eq!(patch.len(), 3 * 3 * 3);
    }

    #[test]
    fn test_extract_image_patch_inverted_bbox() {
        let image: Vec<u8> = vec![0u8; 10 * 10 * 3];
        let bbox = BoundingBox {
            x1: 0.8,
            y1: 0.8,
            x2: 0.2,
            y2: 0.2,
        };
        let patch = GroundingProcessor::extract_image_patch(&image, 10, 10, &bbox);
        assert!(patch.is_empty(), "inverted bbox should return empty patch");
    }

    #[test]
    fn test_sliding_window_proposals_count() {
        // 100x100 image, step 10, single size 10
        // x positions: 0,10,20,...,90 -> 10 positions
        // y positions: 0,10,20,...,90 -> 10 positions
        // total: 10 * 10 = 100 proposals
        let proposals = GroundingProcessor::sliding_window_proposals(100, 100, 10, &[10]);
        assert_eq!(
            proposals.len(),
            100,
            "expected 100 proposals, got {}",
            proposals.len()
        );
    }

    #[test]
    fn test_sliding_window_proposals_multiple_sizes() {
        let proposals = GroundingProcessor::sliding_window_proposals(20, 20, 5, &[5, 10]);
        // size=5: (20-5)/5 + 1 = 4 positions per axis -> 16 proposals
        // size=10: (20-10)/5 + 1 = 3 positions per axis -> 9 proposals
        // total = 25
        assert_eq!(
            proposals.len(),
            25,
            "expected 25 proposals, got {}",
            proposals.len()
        );
    }

    #[test]
    fn test_sliding_window_proposals_normalised() {
        let proposals = GroundingProcessor::sliding_window_proposals(50, 50, 25, &[25]);
        for p in &proposals {
            assert!(p.x1 >= 0.0 && p.x2 <= 1.0, "x out of range: {:?}", p);
            assert!(p.y1 >= 0.0 && p.y2 <= 1.0, "y out of range: {:?}", p);
        }
    }

    #[test]
    fn test_sliding_window_proposals_zero_step() {
        let proposals = GroundingProcessor::sliding_window_proposals(10, 10, 0, &[5]);
        assert!(proposals.is_empty());
    }

    // -----------------------------------------------------------------------
    // PhraseGroundingMetrics tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_recall_at_iou_perfect() {
        let gt = vec![BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 0.5,
            y2: 0.5,
        }];
        let pred = vec![GroundingResultNew {
            phrase: "cat".to_string(),
            bbox: BoundingBox {
                x1: 0.0,
                y1: 0.0,
                x2: 0.5,
                y2: 0.5,
            },
            score: 0.9,
        }];
        let recall = PhraseGroundingMetrics::recall_at_iou(&pred, &gt, 0.5);
        assert!(
            (recall - 1.0).abs() < 1e-5,
            "perfect match should give recall 1.0"
        );
    }

    #[test]
    fn test_recall_at_iou_no_match() {
        let gt = vec![BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 0.3,
            y2: 0.3,
        }];
        let pred = vec![GroundingResultNew {
            phrase: "dog".to_string(),
            bbox: BoundingBox {
                x1: 0.7,
                y1: 0.7,
                x2: 1.0,
                y2: 1.0,
            },
            score: 0.8,
        }];
        let recall = PhraseGroundingMetrics::recall_at_iou(&pred, &gt, 0.5);
        assert!((recall).abs() < 1e-5, "no overlap should give recall 0.0");
    }

    #[test]
    fn test_recall_at_iou_empty_gt() {
        let recall = PhraseGroundingMetrics::recall_at_iou(&[], &[], 0.5);
        assert!(
            (recall - 1.0).abs() < 1e-5,
            "empty gt should give recall 1.0"
        );
    }
}
