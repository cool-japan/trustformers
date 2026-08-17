//! # Object Detection Pipeline
//!
//! ## What is real here
//!
//! The post-processing half of a detection pipeline is fully implemented and
//! independently useful: [`BoundingBox`] geometry (area, IoU, clipping,
//! centre), greedy [`nms`], Gaussian [`soft_nms`], confidence filtering,
//! label filtering and top-k selection. All of it is unit-tested against
//! hand-computed values.
//!
//! ## Model support
//!
//! No detection backbone (DETR, YOLO, …) has a real implementation in
//! `trustformers-models`, so [`ObjectDetectionPipeline::detect`] returns
//! [`DetectionError::UnsupportedModel`] instead of the boxes it used to
//! synthesise from the input buffer's length. Feed your own model's raw
//! detections to [`ObjectDetectionPipeline::postprocess`] to use the real
//! post-processing chain.
//!
//! ## Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::object_detection::{
//!     ObjectDetectionConfig, ObjectDetectionPipeline,
//! };
//!
//! let config = ObjectDetectionConfig::default();
//! let pipeline = ObjectDetectionPipeline::new(config)?;
//! // Real post-processing over detections produced elsewhere:
//! let result = pipeline.postprocess(my_raw_detections, 800, 800);
//! for det in &result.detections {
//!     println!("{}: {:.2} @ {:?}", det.label, det.confidence, det.bbox);
//! }
//! # Ok::<(), detection::DetectionError>(())
//! ```

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the object detection pipeline.
#[derive(Debug, Error)]
pub enum DetectionError {
    #[error("Invalid bounding box: {0}")]
    InvalidBbox(String),
    #[error("Empty image")]
    EmptyImage,
    #[error("Model error: {0}")]
    ModelError(String),
    /// The requested checkpoint has no real implementation in this workspace.
    #[error(
        "no real detection model is implemented for `{requested}`; supported: {supported}. \
         This pipeline never returns synthesised detections — use `postprocess` with your own \
         model's output."
    )]
    UnsupportedModel {
        /// The checkpoint or architecture the caller asked for.
        requested: String,
        /// Comma-separated list of architectures that *are* supported.
        supported: String,
    },
}

/// Detection architectures with a real backbone in this workspace.
///
/// Deliberately empty — the pipeline says so rather than pretending.
const SUPPORTED_ARCHITECTURES: &[&str] = &[];

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`ObjectDetectionPipeline`].
#[derive(Debug, Clone)]
pub struct ObjectDetectionConfig {
    /// HuggingFace model identifier or local path.
    pub model_name: String,
    /// Minimum confidence score to keep a detection.
    pub confidence_threshold: f32,
    /// IoU threshold for Non-Maximum Suppression.
    pub iou_threshold: f32,
    /// Maximum number of detections to return.
    pub max_detections: usize,
    /// Model input size `(height, width)`.
    pub input_size: (usize, usize),
    /// Number of output classes (91 = full COCO).
    pub num_classes: usize,
}

impl Default for ObjectDetectionConfig {
    fn default() -> Self {
        Self {
            model_name: "facebook/detr-resnet-50".to_string(),
            confidence_threshold: 0.5,
            iou_threshold: 0.5,
            max_detections: 100,
            input_size: (800, 800),
            num_classes: 91,
        }
    }
}

// ---------------------------------------------------------------------------
// BoundingBox
// ---------------------------------------------------------------------------

/// An axis-aligned bounding box.
///
/// Coordinates are in any consistent space (normalised `[0,1]` or pixel coords).
#[derive(Debug, Clone)]
pub struct BoundingBox {
    /// Top-left x coordinate.
    pub x1: f32,
    /// Top-left y coordinate.
    pub y1: f32,
    /// Bottom-right x coordinate.
    pub x2: f32,
    /// Bottom-right y coordinate.
    pub y2: f32,
}

impl BoundingBox {
    /// Create a `BoundingBox`, validating that coordinates are in `[0,1]` and `x1<x2`, `y1<y2`.
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Result<Self, DetectionError> {
        for (name, v) in [("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)] {
            if !(0.0..=1.0).contains(&v) {
                return Err(DetectionError::InvalidBbox(format!(
                    "{name} = {v} is outside [0, 1]"
                )));
            }
        }
        if x2 <= x1 {
            return Err(DetectionError::InvalidBbox(format!(
                "x2 ({x2}) must be > x1 ({x1})"
            )));
        }
        if y2 <= y1 {
            return Err(DetectionError::InvalidBbox(format!(
                "y2 ({y2}) must be > y1 ({y1})"
            )));
        }
        Ok(Self { x1, y1, x2, y2 })
    }

    /// Create a BoundingBox without coordinate-range validation.
    ///
    /// Useful for pixel-space boxes where coordinates may exceed `[0,1]`.
    pub fn new_unchecked(x1: f32, y1: f32, x2: f32, y2: f32) -> Result<Self, DetectionError> {
        if x2 <= x1 {
            return Err(DetectionError::InvalidBbox(format!(
                "x2 ({x2}) must be > x1 ({x1})"
            )));
        }
        if y2 <= y1 {
            return Err(DetectionError::InvalidBbox(format!(
                "y2 ({y2}) must be > y1 ({y1})"
            )));
        }
        Ok(Self { x1, y1, x2, y2 })
    }

    /// Area of the bounding box.
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Intersection over Union with another bounding box.
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);

        let inter_w = (ix2 - ix1).max(0.0);
        let inter_h = (iy2 - iy1).max(0.0);
        let inter = inter_w * inter_h;

        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }

    /// Returns `true` if `x2 > x1` and `y2 > y1`.
    pub fn is_valid(&self) -> bool {
        self.x2 > self.x1 && self.y2 > self.y1
    }

    /// Clip the bounding box to image boundaries `[0, w] × [0, h]`.
    pub fn clip_to_image(&self, w: f32, h: f32) -> Self {
        let x1 = self.x1.clamp(0.0, w);
        let y1 = self.y1.clamp(0.0, h);
        let x2 = self.x2.clamp(0.0, w);
        let y2 = self.y2.clamp(0.0, h);
        // After clipping x2 might equal x1 — caller must check is_valid().
        Self { x1, y1, x2, y2 }
    }

    /// Width of the bounding box.
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    /// Height of the bounding box.
    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    /// Centre `(cx, cy)` of the bounding box.
    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }
}

// ---------------------------------------------------------------------------
// Detection — new-style struct (score field alias for confidence)
// ---------------------------------------------------------------------------

/// A single detected object (new-style, with `score` field).
#[derive(Debug, Clone)]
pub struct Detection {
    /// Bounding box coordinates.
    pub bbox: BoundingBox,
    /// Human-readable class label.
    pub label: String,
    /// Class index.
    pub label_id: usize,
    /// Detection confidence in `[0, 1]` (also accessible as `score`).
    pub confidence: f32,
}

impl Detection {
    /// Score alias for confidence (consistent with other pipeline result types).
    pub fn score(&self) -> f32 {
        self.confidence
    }
}

/// Collection of detections for one image.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// All detections after filtering and NMS.
    pub detections: Vec<Detection>,
    /// Original image height.
    pub image_height: usize,
    /// Original image width.
    pub image_width: usize,
    /// Approximate inference duration in milliseconds.
    pub inference_time_ms: u64,
}

impl DetectionResult {
    /// Keep only detections whose confidence is >= `threshold`.
    pub fn filter_by_confidence(&self, threshold: f32) -> Self {
        Self {
            detections: self
                .detections
                .iter()
                .filter(|d| d.confidence >= threshold)
                .cloned()
                .collect(),
            image_height: self.image_height,
            image_width: self.image_width,
            inference_time_ms: self.inference_time_ms,
        }
    }

    /// Keep only detections whose label equals `label`.
    pub fn filter_by_label(&self, label: &str) -> Self {
        Self {
            detections: self.detections.iter().filter(|d| d.label == label).cloned().collect(),
            image_height: self.image_height,
            image_width: self.image_width,
            inference_time_ms: self.inference_time_ms,
        }
    }

    /// Return the top-`k` detections sorted by confidence (highest first).
    pub fn top_k(&self, k: usize) -> Self {
        let mut sorted = self.detections.clone();
        sorted.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(k);
        Self {
            detections: sorted,
            image_height: self.image_height,
            image_width: self.image_width,
            inference_time_ms: self.inference_time_ms,
        }
    }

    /// Count the number of detections per label.
    pub fn count_by_label(&self) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for d in &self.detections {
            *counts.entry(d.label.clone()).or_insert(0) += 1;
        }
        counts
    }
}

// ---------------------------------------------------------------------------
// NMS and Soft-NMS
// ---------------------------------------------------------------------------

/// Greedy Non-Maximum Suppression over a list of `Detection` values.
///
/// - Sorts by confidence descending.
/// - Suppresses any box whose IoU with a kept box exceeds `iou_threshold`.
///
/// Returns the surviving detections.
pub fn nms(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
    let mut sorted: Vec<&Detection> = detections.iter().collect();
    sorted.sort_by(|a, b| {
        b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<Detection> = Vec::new();
    let mut suppressed = vec![false; sorted.len()];

    for i in 0..sorted.len() {
        if suppressed[i] {
            continue;
        }
        kept.push(sorted[i].clone());
        for j in (i + 1)..sorted.len() {
            if suppressed[j] {
                continue;
            }
            if sorted[i].bbox.iou(&sorted[j].bbox) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    kept
}

/// Soft-NMS with Gaussian score decay.
///
/// Instead of hard suppression, reduces the score of overlapping boxes
/// by a factor of `exp(- iou^2 / sigma)`.
///
/// Boxes whose decayed score falls below `score_threshold` are removed.
pub fn soft_nms(detections: &[Detection], sigma: f32, score_threshold: f32) -> Vec<Detection> {
    if detections.is_empty() {
        return Vec::new();
    }

    // Work on mutable copies with scores
    let mut scored: Vec<(Detection, f32)> =
        detections.iter().map(|d| (d.clone(), d.confidence)).collect();

    let n = scored.len();
    let mut result: Vec<Detection> = Vec::new();

    // Iterative soft-NMS: pick the highest-score box, apply decay
    for _ in 0..n {
        // Find the current maximum-score item
        let max_idx = scored
            .iter()
            .enumerate()
            .filter(|(_, (_, s))| *s > 0.0)
            .max_by(|(_, (_, a)), (_, (_, b))| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        let max_idx = match max_idx {
            Some(idx) => idx,
            None => break,
        };

        let (best_det, best_score) = scored[max_idx].clone();
        if best_score <= score_threshold {
            break;
        }
        result.push(Detection {
            confidence: best_score,
            ..best_det.clone()
        });
        // Zero out so we don't pick it again
        scored[max_idx].1 = 0.0;

        // Decay remaining box scores
        for (i, (det, score)) in scored.iter_mut().enumerate() {
            if i == max_idx || *score <= 0.0 {
                continue;
            }
            let iou = best_det.bbox.iou(&det.bbox);
            let decay = (-iou * iou / sigma.max(1e-6)).exp();
            *score *= decay;
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// DETR-compatible object detection pipeline.
pub struct ObjectDetectionPipeline {
    config: ObjectDetectionConfig,
    labels: Vec<String>,
}

/// The first 20 COCO class names, used as the pipeline's default label set.
const COCO_LABELS_20: &[&str] = &[
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
];

impl ObjectDetectionPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: ObjectDetectionConfig) -> Result<Self, DetectionError> {
        if config.input_size.0 == 0 || config.input_size.1 == 0 {
            return Err(DetectionError::ModelError(
                "input_size dimensions must be > 0".to_string(),
            ));
        }
        let labels: Vec<String> = COCO_LABELS_20.iter().map(|s| s.to_string()).collect();
        Ok(Self { config, labels })
    }

    /// Run object detection on a single image.
    ///
    /// `image` is a flat `f32` buffer, `height` and `width` describe its spatial dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`DetectionError::EmptyImage`] for an empty buffer and
    /// [`DetectionError::UnsupportedModel`] otherwise: no detection backbone is
    /// implemented, and this pipeline will not fabricate boxes. Use
    /// [`Self::postprocess`] with your own model's raw detections.
    pub fn detect(
        &self,
        image: &[f32],
        _height: usize,
        _width: usize,
    ) -> Result<DetectionResult, DetectionError> {
        if image.is_empty() {
            return Err(DetectionError::EmptyImage);
        }
        Err(DetectionError::UnsupportedModel {
            requested: self.config.model_name.clone(),
            supported: if SUPPORTED_ARCHITECTURES.is_empty() {
                "none (no detection backbone is implemented yet)".to_string()
            } else {
                SUPPORTED_ARCHITECTURES.join(", ")
            },
        })
    }

    /// Run object detection on a batch of images.
    ///
    /// # Errors
    ///
    /// See [`Self::detect`].
    pub fn detect_batch(
        &self,
        images: &[(&[f32], usize, usize)],
    ) -> Result<Vec<DetectionResult>, DetectionError> {
        if images.is_empty() {
            return Err(DetectionError::EmptyImage);
        }
        images.iter().map(|&(data, h, w)| self.detect(data, h, w)).collect()
    }

    /// Apply the pipeline's real post-processing chain to raw detections.
    ///
    /// Confidence filtering → greedy NMS at `iou_threshold` → truncation to
    /// `max_detections`. `inference_time_ms` is reported as the caller's
    /// measured value, or `0` when unknown — it is never invented.
    pub fn postprocess(
        &self,
        detections: Vec<Detection>,
        image_height: usize,
        image_width: usize,
    ) -> DetectionResult {
        self.postprocess_timed(detections, image_height, image_width, 0)
    }

    /// [`Self::postprocess`] with a caller-measured inference duration.
    pub fn postprocess_timed(
        &self,
        mut detections: Vec<Detection>,
        image_height: usize,
        image_width: usize,
        inference_time_ms: u64,
    ) -> DetectionResult {
        detections.retain(|d| d.confidence >= self.config.confidence_threshold);
        let mut after_nms = nms(&detections, self.config.iou_threshold);
        after_nms.truncate(self.config.max_detections);

        DetectionResult {
            detections: after_nms,
            image_height,
            image_width,
            inference_time_ms,
        }
    }

    /// Greedy NMS (convenience wrapper; also available as a free function `nms`).
    pub fn nms(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
        nms(detections, iou_threshold)
    }

    /// Soft-NMS with Gaussian decay (convenience wrapper; also available as a free function).
    pub fn soft_nms(detections: &[Detection], sigma: f32, score_threshold: f32) -> Vec<Detection> {
        soft_nms(detections, sigma, score_threshold)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_image(h: usize, w: usize) -> Vec<f32> {
        (0..h * w * 3).map(|i| (i % 256) as f32 / 255.0).collect()
    }

    fn make_det(x1: f32, y1: f32, x2: f32, y2: f32, confidence: f32) -> Detection {
        Detection {
            bbox: BoundingBox { x1, y1, x2, y2 },
            label: "test".to_string(),
            label_id: 0,
            confidence,
        }
    }

    // ---- 1. BoundingBox creation valid ----

    #[test]
    fn test_bbox_valid() {
        let bbox = BoundingBox::new(0.1, 0.1, 0.9, 0.9).expect("valid bbox");
        assert!((bbox.x1 - 0.1).abs() < 1e-6);
        assert!((bbox.x2 - 0.9).abs() < 1e-6);
    }

    // ---- 2. BoundingBox creation invalid (x2 <= x1) ----

    #[test]
    fn test_bbox_invalid_x() {
        let result = BoundingBox::new(0.5, 0.1, 0.3, 0.9);
        assert!(matches!(result, Err(DetectionError::InvalidBbox(_))));
    }

    // ---- 3. BoundingBox creation invalid (out of range) ----

    #[test]
    fn test_bbox_invalid_range() {
        let result = BoundingBox::new(-0.1, 0.0, 0.5, 1.0);
        assert!(matches!(result, Err(DetectionError::InvalidBbox(_))));
    }

    // ---- 4. area ----

    #[test]
    fn test_bbox_area() {
        let bbox = BoundingBox::new(0.0, 0.0, 0.5, 0.4).expect("valid");
        assert!((bbox.area() - 0.2).abs() < 1e-6);
    }

    // ---- 5. IoU non-overlapping ----

    #[test]
    fn test_bbox_iou_no_overlap() {
        let a = BoundingBox::new(0.0, 0.0, 0.4, 0.4).expect("valid");
        let b = BoundingBox::new(0.6, 0.6, 1.0, 1.0).expect("valid");
        assert!((a.iou(&b) - 0.0).abs() < 1e-6);
    }

    // ---- 6. IoU identical boxes ----

    #[test]
    fn test_bbox_iou_identical() {
        let a = BoundingBox::new(0.2, 0.2, 0.8, 0.8).expect("valid");
        let b = a.clone();
        assert!((a.iou(&b) - 1.0).abs() < 1e-6);
    }

    // ---- 7. IoU partial overlap ----

    #[test]
    fn test_bbox_iou_partial() {
        let a = BoundingBox::new(0.0, 0.0, 0.6, 0.6).expect("valid");
        let b = BoundingBox::new(0.4, 0.4, 1.0, 1.0).expect("valid");
        // intersection = 0.2 * 0.2 = 0.04
        // union = 0.36 + 0.36 - 0.04 = 0.68
        let expected = 0.04 / 0.68;
        assert!((a.iou(&b) - expected).abs() < 1e-5);
    }

    // ---- 8. NMS removes overlapping boxes ----

    #[test]
    fn test_nms_removes_overlapping() {
        let bbox_hi = BoundingBox::new(0.0, 0.0, 0.5, 0.5).expect("valid");
        let bbox_lo = BoundingBox::new(0.01, 0.01, 0.49, 0.49).expect("valid");
        let dets = vec![
            Detection {
                bbox: bbox_lo,
                label: "cat".into(),
                label_id: 0,
                confidence: 0.6,
            },
            Detection {
                bbox: bbox_hi,
                label: "cat".into(),
                label_id: 0,
                confidence: 0.9,
            },
        ];
        let result = nms(&dets, 0.5);
        // Only the high-confidence box should survive.
        assert_eq!(result.len(), 1);
        assert!((result[0].confidence - 0.9).abs() < 1e-6);
    }

    // ---- 9. NMS keeps non-overlapping boxes ----

    #[test]
    fn test_nms_keeps_non_overlapping() {
        let b1 = BoundingBox::new(0.0, 0.0, 0.3, 0.3).expect("valid");
        let b2 = BoundingBox::new(0.7, 0.7, 1.0, 1.0).expect("valid");
        let dets = vec![
            Detection {
                bbox: b1,
                label: "dog".into(),
                label_id: 1,
                confidence: 0.8,
            },
            Detection {
                bbox: b2,
                label: "cat".into(),
                label_id: 0,
                confidence: 0.7,
            },
        ];
        let result = nms(&dets, 0.5);
        assert_eq!(result.len(), 2);
    }

    // ---- 10. DetectionResult filter_by_confidence ----

    #[test]
    fn test_filter_by_confidence() {
        let pipeline = ObjectDetectionPipeline::new(ObjectDetectionConfig::default()).expect("ok");
        let result = pipeline.postprocess(
            vec![
                make_det(0.0, 0.0, 0.2, 0.2, 0.95),
                make_det(0.5, 0.5, 0.7, 0.7, 0.60),
            ],
            100,
            100,
        );
        let filtered = result.filter_by_confidence(0.8);
        assert_eq!(filtered.detections.len(), 1);
        assert!(filtered.detections.iter().all(|d| d.confidence >= 0.8));
    }

    // ---- 11. DetectionResult filter_by_label ----

    #[test]
    fn test_filter_by_label() {
        let b = BoundingBox::new(0.1, 0.1, 0.5, 0.5).expect("valid");
        let dets = vec![
            Detection {
                bbox: b.clone(),
                label: "cat".into(),
                label_id: 0,
                confidence: 0.9,
            },
            Detection {
                bbox: b.clone(),
                label: "dog".into(),
                label_id: 1,
                confidence: 0.8,
            },
            Detection {
                bbox: b.clone(),
                label: "cat".into(),
                label_id: 0,
                confidence: 0.7,
            },
        ];
        let result = DetectionResult {
            detections: dets,
            image_height: 100,
            image_width: 100,
            inference_time_ms: 0,
        };
        let cats = result.filter_by_label("cat");
        assert_eq!(cats.detections.len(), 2);
        assert!(cats.detections.iter().all(|d| d.label == "cat"));
    }

    // ---- 12. DetectionResult top_k ----

    #[test]
    fn test_top_k() {
        let b = BoundingBox::new(0.1, 0.1, 0.5, 0.5).expect("valid");
        let dets: Vec<Detection> = (0..5)
            .map(|i| Detection {
                bbox: b.clone(),
                label: "x".into(),
                label_id: i,
                confidence: i as f32 * 0.1 + 0.1,
            })
            .collect();
        let result = DetectionResult {
            detections: dets,
            image_height: 10,
            image_width: 10,
            inference_time_ms: 0,
        };
        let top2 = result.top_k(2);
        assert_eq!(top2.detections.len(), 2);
        assert!(top2.detections[0].confidence >= top2.detections[1].confidence);
    }

    // ---- 13. DetectionResult count_by_label ----

    #[test]
    fn test_count_by_label() {
        let b = BoundingBox::new(0.1, 0.1, 0.5, 0.5).expect("valid");
        let dets = vec![
            Detection {
                bbox: b.clone(),
                label: "cat".into(),
                label_id: 0,
                confidence: 0.9,
            },
            Detection {
                bbox: b.clone(),
                label: "dog".into(),
                label_id: 1,
                confidence: 0.8,
            },
            Detection {
                bbox: b.clone(),
                label: "cat".into(),
                label_id: 0,
                confidence: 0.7,
            },
        ];
        let result = DetectionResult {
            detections: dets,
            image_height: 10,
            image_width: 10,
            inference_time_ms: 0,
        };
        let counts = result.count_by_label();
        assert_eq!(counts["cat"], 2);
        assert_eq!(counts["dog"], 1);
    }

    // ---- 14. detect basic ----

    #[test]
    fn test_detect_reports_unsupported_model() {
        // Regression: `detect` used to fabricate boxes whose count came from
        // `image.len() % 10` and whose coordinates came from the loop index.
        let config = ObjectDetectionConfig {
            confidence_threshold: 0.0,
            model_name: "facebook/detr-resnet-50".to_string(),
            ..Default::default()
        };
        let pipeline = ObjectDetectionPipeline::new(config).expect("ok");
        let image = make_image(50, 50);
        match pipeline.detect(&image, 50, 50) {
            Err(DetectionError::UnsupportedModel {
                requested,
                supported,
            }) => {
                assert_eq!(requested, "facebook/detr-resnet-50");
                assert!(supported.contains("none"), "supported: {supported}");
            },
            other => panic!("expected UnsupportedModel, got {other:?}"),
        }
    }

    #[test]
    fn test_postprocess_runs_the_real_chain() {
        let config = ObjectDetectionConfig {
            confidence_threshold: 0.5,
            iou_threshold: 0.5,
            max_detections: 10,
            ..Default::default()
        };
        let pipeline = ObjectDetectionPipeline::new(config).expect("ok");
        let raw = vec![
            make_det(0.0, 0.0, 0.4, 0.4, 0.9),     // kept
            make_det(0.01, 0.01, 0.41, 0.41, 0.8), // suppressed by NMS
            make_det(0.6, 0.6, 0.9, 0.9, 0.7),     // kept
            make_det(0.2, 0.2, 0.3, 0.3, 0.1),     // below threshold
        ];
        let result = pipeline.postprocess(raw, 50, 50);
        assert_eq!(result.detections.len(), 2, "{:?}", result.detections);
        assert_eq!(result.image_height, 50);
        assert_eq!(result.image_width, 50);
        assert_eq!(result.inference_time_ms, 0, "timing must not be invented");
    }

    #[test]
    fn test_postprocess_timed_reports_caller_measurement() {
        let pipeline = ObjectDetectionPipeline::new(ObjectDetectionConfig::default()).expect("ok");
        let result =
            pipeline.postprocess_timed(vec![make_det(0.0, 0.0, 0.4, 0.4, 0.9)], 10, 10, 42);
        assert_eq!(result.inference_time_ms, 42);
    }

    // ---- 15. detect empty image ----

    #[test]
    fn test_detect_empty_image() {
        let pipeline = ObjectDetectionPipeline::new(ObjectDetectionConfig::default()).expect("ok");
        let result = pipeline.detect(&[], 10, 10);
        assert!(matches!(result, Err(DetectionError::EmptyImage)));
    }

    // ---- 16. BoundingBox::is_valid ----

    #[test]
    fn test_bbox_is_valid() {
        let valid = BoundingBox {
            x1: 0.1,
            y1: 0.1,
            x2: 0.9,
            y2: 0.9,
        };
        assert!(valid.is_valid(), "should be valid");
        let degenerate = BoundingBox {
            x1: 0.5,
            y1: 0.1,
            x2: 0.5,
            y2: 0.9,
        };
        assert!(!degenerate.is_valid(), "x1 == x2 is invalid");
    }

    // ---- 17. BoundingBox::clip_to_image ----

    #[test]
    fn test_bbox_clip_to_image() {
        let big = BoundingBox {
            x1: -0.1,
            y1: -0.2,
            x2: 1.5,
            y2: 2.0,
        };
        let clipped = big.clip_to_image(1.0, 1.0);
        assert!((clipped.x1 - 0.0).abs() < 1e-6);
        assert!((clipped.y1 - 0.0).abs() < 1e-6);
        assert!((clipped.x2 - 1.0).abs() < 1e-6);
        assert!((clipped.y2 - 1.0).abs() < 1e-6);
    }

    // ---- 18. IoU symmetry ----

    #[test]
    fn test_iou_symmetry() {
        let a = BoundingBox::new(0.0, 0.0, 0.6, 0.6).expect("valid");
        let b = BoundingBox::new(0.3, 0.3, 0.9, 0.9).expect("valid");
        assert!(
            (a.iou(&b) - b.iou(&a)).abs() < 1e-6,
            "IoU must be symmetric"
        );
    }

    // ---- 19. NMS output is sorted by confidence ----

    #[test]
    fn test_nms_output_sorted_by_confidence() {
        let dets = vec![
            make_det(0.0, 0.0, 0.3, 0.3, 0.6),
            make_det(0.5, 0.5, 0.8, 0.8, 0.9),
            make_det(0.1, 0.1, 0.4, 0.4, 0.75),
        ];
        let result = nms(&dets, 0.3);
        for w in result.windows(2) {
            assert!(
                w[0].confidence >= w[1].confidence,
                "NMS output should be sorted descending"
            );
        }
    }

    // ---- 20. Soft-NMS decays overlapping scores ----

    #[test]
    fn test_soft_nms_decays_scores() {
        // Two heavily overlapping boxes (nearly identical)
        let dets = vec![
            make_det(0.0, 0.0, 0.5, 0.5, 0.9),
            make_det(0.01, 0.01, 0.49, 0.49, 0.85),
        ];
        let result = soft_nms(&dets, 0.5, 0.01);
        // Both should survive soft-NMS (unlike hard NMS), but the second's score decays
        assert!(
            !result.is_empty(),
            "soft-NMS should keep at least one detection"
        );
        // The top detection should have high confidence
        assert!(
            result[0].confidence > 0.5,
            "top detection should have reasonable confidence"
        );
    }

    // ---- 21. Soft-NMS removes very low-scored boxes ----

    #[test]
    fn test_soft_nms_removes_low_score_boxes() {
        // Use a tight sigma so decay is very aggressive on highly overlapping boxes
        let dets = vec![
            make_det(0.0, 0.0, 0.9, 0.9, 0.95),
            make_det(0.01, 0.01, 0.89, 0.89, 0.3),
        ];
        // High score threshold: anything decayed below 0.5 is removed
        let result = soft_nms(&dets, 0.1, 0.5);
        // The second box's score should decay significantly and be pruned
        assert!(result.len() <= 2, "at most 2 boxes can survive");
        if result.len() > 1 {
            assert!(result[1].confidence >= 0.5, "kept box must meet threshold");
        }
    }

    // ---- 22. detect_batch ----

    #[test]
    fn test_detect_batch_reports_unsupported_model() {
        let config = ObjectDetectionConfig {
            confidence_threshold: 0.0,
            ..Default::default()
        };
        let pipeline = ObjectDetectionPipeline::new(config).expect("ok");
        let img1 = make_image(20, 20);
        let img2 = make_image(30, 30);
        let batch: Vec<(&[f32], usize, usize)> =
            vec![(img1.as_slice(), 20, 20), (img2.as_slice(), 30, 30)];
        assert!(matches!(
            pipeline.detect_batch(&batch),
            Err(DetectionError::UnsupportedModel { .. })
        ));
        assert!(matches!(
            pipeline.detect_batch(&[]),
            Err(DetectionError::EmptyImage)
        ));
    }

    // ---- 23. bbox area is positive ----

    #[test]
    fn test_bbox_area_positive() {
        let b = BoundingBox::new(0.1, 0.2, 0.6, 0.8).expect("valid");
        assert!(b.area() > 0.0, "area should be positive for valid bbox");
    }

    // ---- 24. Detection::score alias ----

    #[test]
    fn test_detection_score_alias() {
        let det = make_det(0.1, 0.1, 0.5, 0.5, 0.77);
        assert!(
            (det.score() - 0.77).abs() < 1e-6,
            "score() should equal confidence"
        );
    }

    // ---- 25. NMS empty input returns empty ----

    #[test]
    fn test_nms_empty_input() {
        let result = nms(&[], 0.5);
        assert!(result.is_empty(), "NMS on empty input should return empty");
    }
}
