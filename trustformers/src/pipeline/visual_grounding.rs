//! # Visual Grounding Pipeline
//!
//! GroundingDINO-compatible visual grounding pipeline: locate objects in images
//! described by free-form text queries and return normalised bounding boxes.
//!
//! ## Supported model families
//! - **GroundingDINO** — open-set object detection with language conditioning
//!
//! ## Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::visual_grounding::{VisualGroundingConfig, VisualGroundingPipeline};
//!
//! let config = VisualGroundingConfig::default();
//! let pipeline = VisualGroundingPipeline::new(config)?;
//! let pixels = vec![0.5_f32; 800 * 1333 * 3];
//! let result = pipeline.ground(&pixels, 800, 1333, "a cat . a dog")?;
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
}

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
        let boxes: Vec<GroundedBox> = self
            .boxes
            .iter()
            .filter(|b| b.score >= threshold)
            .cloned()
            .collect();
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

/// Generate a mock bounding box for `phrase` given image `height` and `width`.
///
/// Position is deterministic based on the phrase hash; size shrinks with phrase
/// length (longer phrase → more specific → smaller box).
fn mock_box_for_phrase(phrase: &str, _height: usize, _width: usize) -> (f32, f32, f32, f32) {
    let h = djb2_hash(phrase);

    // Decode position from hash bytes (normalised to [0, 1]).
    let cx = ((h & 0xFF) as f32) / 255.0;
    let cy = (((h >> 8) & 0xFF) as f32) / 255.0;

    // Shorter phrases → larger boxes (max ~0.6, min ~0.1).
    let max_side = 0.60_f32;
    let min_side = 0.10_f32;
    let len_factor = (phrase.len() as f32 / 30.0_f32).clamp(0.0, 1.0);
    let side = max_side - len_factor * (max_side - min_side);

    let half = side / 2.0;
    let x1 = (cx - half).clamp(0.0, 1.0 - side);
    let y1 = (cy - half).clamp(0.0, 1.0 - side);
    let x2 = (x1 + side).min(1.0);
    let y2 = (y1 + side).min(1.0);

    (x1, y1, x2, y2)
}

/// Compute a mock confidence score for a phrase-image pair in `[0, 1]`.
fn mock_score(phrase: &str, image: &[f32]) -> f32 {
    let h = djb2_hash(phrase);
    let base = (((h >> 16) & 0xFF) as f32) / 255.0;
    // Blend with mean pixel intensity to vary by image content.
    let img_mean = if image.is_empty() {
        0.5
    } else {
        let sum: f32 = image.iter().take(256).sum();
        (sum / image.len().min(256) as f32).clamp(0.0, 1.0)
    };
    (base * 0.7 + img_mean * 0.3).clamp(0.0, 1.0)
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
    /// `image` is a flat pixel buffer (any channel layout is accepted; only
    /// its length is used for mock scoring).
    ///
    /// # Errors
    ///
    /// - [`GroundingError::EmptyImage`] — `image` is empty.
    /// - [`GroundingError::EmptyQuery`] — `text_query` is blank.
    pub fn ground(
        &self,
        image: &[f32],
        height: usize,
        width: usize,
        text_query: &str,
    ) -> Result<GroundingResult, GroundingError> {
        if image.is_empty() {
            return Err(GroundingError::EmptyImage);
        }
        let trimmed = text_query.trim();
        if trimmed.is_empty() {
            return Err(GroundingError::EmptyQuery);
        }

        let phrases = parse_phrases(trimmed);
        let mut boxes: Vec<GroundedBox> = Vec::new();

        for phrase in &phrases {
            let score = mock_score(phrase, image);
            if score < self.config.box_threshold {
                continue;
            }
            let phrase_h = djb2_hash(phrase);
            let phrase_score = (((phrase_h >> 32) & 0xFF) as f32 / 255.0)
                .clamp(self.config.text_threshold, 1.0);
            let bbox = mock_box_for_phrase(phrase, height, width);
            boxes.push(GroundedBox {
                phrase: phrase.clone(),
                bbox,
                score,
                phrase_score,
            });
        }

        // Sort by score descending and cap at max_detections.
        boxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        boxes.truncate(self.config.max_detections);

        Ok(GroundingResult {
            boxes,
            query: trimmed.to_string(),
            image_height: height,
            image_width: width,
        })
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
        images
            .iter()
            .map(|(img, h, w)| self.ground(img, *h, *w, text_query))
            .collect()
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
                let lower = word
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase();
                let mut h: u64 = 5381;
                for b in lower.bytes() {
                    h = h.wrapping_mul(33).wrapping_add(b as u64);
                }
                (h % 30_000) as u32 + 1
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
        let matched = ground_truth.iter().filter(|gt| {
            predictions.iter().any(|pred| pred.bbox.iou(gt) >= iou_threshold)
        });
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
        assert_eq!(phrases.len(), 3, "expected 3 unique phrases, got {:?}", phrases);
    }

    // --- Pipeline::ground ---

    #[test]
    fn test_ground_multi_phrase_query_produces_boxes() {
        // Use a low box_threshold so mock boxes are kept.
        let config = VisualGroundingConfig {
            box_threshold: 0.0,
            ..Default::default()
        };
        let p = VisualGroundingPipeline::new(config).expect("valid");
        let img = dummy_image(800 * 600 * 3);
        let result = p
            .ground(&img, 800, 600, "a cat . a dog . a bird")
            .expect("ground ok");
        // Three phrases → at most three boxes (before filtering).
        assert!(
            !result.boxes.is_empty(),
            "expected at least one box"
        );
    }

    // --- Pipeline::ground_batch ---

    #[test]
    fn test_ground_batch_count() {
        let config = VisualGroundingConfig {
            box_threshold: 0.0,
            ..Default::default()
        };
        let p = VisualGroundingPipeline::new(config).expect("valid");
        let img1 = dummy_image(100 * 100 * 3);
        let img2 = dummy_image(200 * 200 * 3);
        let images: Vec<(&[f32], usize, usize)> =
            vec![(&img1, 100, 100), (&img2, 200, 200)];
        let results = p.ground_batch(&images, "cat").expect("batch ok");
        assert_eq!(results.len(), 2);
    }

    // --- Error cases ---

    #[test]
    fn test_empty_image_error() {
        let p = default_pipeline();
        let err = p
            .ground(&[], 100, 100, "cat")
            .expect_err("empty image should fail");
        assert!(matches!(err, GroundingError::EmptyImage));
    }

    #[test]
    fn test_empty_query_error() {
        let p = default_pipeline();
        let img = dummy_image(100);
        let err = p
            .ground(&img, 10, 10, "   ")
            .expect_err("empty query should fail");
        assert!(matches!(err, GroundingError::EmptyQuery));
    }

    // --- Bbox / score invariants ---

    #[test]
    fn test_boxes_within_unit_square() {
        let config = VisualGroundingConfig {
            box_threshold: 0.0,
            ..Default::default()
        };
        let p = VisualGroundingPipeline::new(config).expect("valid");
        let img = dummy_image(800 * 600 * 3);
        let result = p
            .ground(&img, 800, 600, "tree, car, person, building")
            .expect("ok");
        for b in &result.boxes {
            let (x1, y1, x2, y2) = b.bbox;
            assert!(x1 >= 0.0 && y1 >= 0.0 && x2 <= 1.0 && y2 <= 1.0,
                "box out of unit square: {:?}", b.bbox);
            assert!(x2 >= x1 && y2 >= y1, "box coordinates inverted: {:?}", b.bbox);
        }
    }

    #[test]
    fn test_box_score_le_one() {
        let config = VisualGroundingConfig {
            box_threshold: 0.0,
            ..Default::default()
        };
        let p = VisualGroundingPipeline::new(config).expect("valid");
        let img = dummy_image(400 * 300 * 3);
        let result = p
            .ground(&img, 400, 300, "window, door, roof")
            .expect("ok");
        for b in &result.boxes {
            assert!(
                b.score <= 1.0,
                "score {} > 1.0 for phrase '{}'",
                b.score,
                b.phrase
            );
        }
    }

    // -----------------------------------------------------------------------
    // BoundingBox extended type tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bounding_box_area() {
        let bb = BoundingBox { x1: 0.1, y1: 0.1, x2: 0.5, y2: 0.6 };
        let expected = (0.5 - 0.1) * (0.6 - 0.1);
        assert!((bb.area() - expected).abs() < 1e-6, "area was {}", bb.area());
    }

    #[test]
    fn test_bounding_box_iou_identical() {
        let bb = BoundingBox { x1: 0.1, y1: 0.1, x2: 0.5, y2: 0.5 };
        assert!((bb.iou(&bb) - 1.0).abs() < 1e-5, "iou of identical box should be 1.0");
    }

    #[test]
    fn test_bounding_box_iou_no_overlap() {
        let a = BoundingBox { x1: 0.0, y1: 0.0, x2: 0.3, y2: 0.3 };
        let b = BoundingBox { x1: 0.5, y1: 0.5, x2: 0.8, y2: 0.8 };
        assert!((a.iou(&b)).abs() < 1e-6, "non-overlapping boxes should have iou ~0");
    }

    #[test]
    fn test_bounding_box_iou_partial_overlap() {
        let a = BoundingBox { x1: 0.0, y1: 0.0, x2: 0.6, y2: 0.6 };
        let b = BoundingBox { x1: 0.4, y1: 0.4, x2: 1.0, y2: 1.0 };
        let iou = a.iou(&b);
        assert!(iou > 0.0 && iou < 1.0, "partial overlap iou should be in (0,1), got {iou}");
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
        assert!(sim > 0.9, "self-similarity should be close to 1.0, got {sim}");
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
        let bbox = BoundingBox { x1: 0.0, y1: 0.0, x2: 0.5, y2: 0.5 };
        let patch = GroundingProcessor::extract_image_patch(&image, 4, 4, &bbox);
        // 2x2 pixels * 3 channels = 12 bytes
        assert_eq!(patch.len(), 12, "expected 12 bytes, got {}", patch.len());
    }

    #[test]
    fn test_extract_image_patch_full_image() {
        let image: Vec<u8> = vec![255u8; 3 * 3 * 3];
        let bbox = BoundingBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
        let patch = GroundingProcessor::extract_image_patch(&image, 3, 3, &bbox);
        assert_eq!(patch.len(), 3 * 3 * 3);
    }

    #[test]
    fn test_extract_image_patch_inverted_bbox() {
        let image: Vec<u8> = vec![0u8; 10 * 10 * 3];
        let bbox = BoundingBox { x1: 0.8, y1: 0.8, x2: 0.2, y2: 0.2 };
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
        assert_eq!(proposals.len(), 100, "expected 100 proposals, got {}", proposals.len());
    }

    #[test]
    fn test_sliding_window_proposals_multiple_sizes() {
        let proposals =
            GroundingProcessor::sliding_window_proposals(20, 20, 5, &[5, 10]);
        // size=5: (20-5)/5 + 1 = 4 positions per axis -> 16 proposals
        // size=10: (20-10)/5 + 1 = 3 positions per axis -> 9 proposals
        // total = 25
        assert_eq!(proposals.len(), 25, "expected 25 proposals, got {}", proposals.len());
    }

    #[test]
    fn test_sliding_window_proposals_normalised() {
        let proposals =
            GroundingProcessor::sliding_window_proposals(50, 50, 25, &[25]);
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
        let gt = vec![BoundingBox { x1: 0.0, y1: 0.0, x2: 0.5, y2: 0.5 }];
        let pred = vec![GroundingResultNew {
            phrase: "cat".to_string(),
            bbox: BoundingBox { x1: 0.0, y1: 0.0, x2: 0.5, y2: 0.5 },
            score: 0.9,
        }];
        let recall = PhraseGroundingMetrics::recall_at_iou(&pred, &gt, 0.5);
        assert!((recall - 1.0).abs() < 1e-5, "perfect match should give recall 1.0");
    }

    #[test]
    fn test_recall_at_iou_no_match() {
        let gt = vec![BoundingBox { x1: 0.0, y1: 0.0, x2: 0.3, y2: 0.3 }];
        let pred = vec![GroundingResultNew {
            phrase: "dog".to_string(),
            bbox: BoundingBox { x1: 0.7, y1: 0.7, x2: 1.0, y2: 1.0 },
            score: 0.8,
        }];
        let recall = PhraseGroundingMetrics::recall_at_iou(&pred, &gt, 0.5);
        assert!((recall).abs() < 1e-5, "no overlap should give recall 0.0");
    }

    #[test]
    fn test_recall_at_iou_empty_gt() {
        let recall =
            PhraseGroundingMetrics::recall_at_iou(&[], &[], 0.5);
        assert!((recall - 1.0).abs() < 1e-5, "empty gt should give recall 1.0");
    }
}
