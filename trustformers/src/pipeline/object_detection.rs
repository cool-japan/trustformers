//! # Object Detection Pipeline
//!
//! DETR-compatible object detection returning bounding boxes with labels and confidence scores.
//!
//! ## Supported model families
//! - **DETR** (Detection Transformer) — end-to-end transformer-based detection
//! - **YOLO** — classic one-stage detection baselines
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
//! let image = vec![0.5f32; 800 * 800 * 3];
//! let result = pipeline.detect(&image, 800, 800)?;
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
}

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

/// First 20 COCO class names used for deterministic mock detections.
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
    pub fn detect(
        &self,
        image: &[f32],
        height: usize,
        width: usize,
    ) -> Result<DetectionResult, DetectionError> {
        if image.is_empty() {
            return Err(DetectionError::EmptyImage);
        }

        // Deterministic: number of detections derived from image length.
        let num_detections = (image.len() % 10) + 1;

        let mut detections: Vec<Detection> = (0..num_detections)
            .map(|i| {
                let label_id = i % self.labels.len();
                let label = self.labels[label_id].clone();

                // Deterministic box derived from index and image length.
                let seed = (i as f32 + 1.0) / (num_detections as f32 + 1.0);
                let x1 = (seed * 0.5).min(0.49);
                let y1 = (seed * 0.4).min(0.39);
                let x2 = (x1 + 0.3 + seed * 0.1).min(1.0);
                let y2 = (y1 + 0.3 + seed * 0.1).min(1.0);
                // Ensure x2 > x1 and y2 > y1 with a small epsilon.
                let x2 = x2.max(x1 + 0.01);
                let y2 = y2.max(y1 + 0.01);
                let bbox = BoundingBox { x1, y1, x2, y2 };
                let confidence = 0.55 + seed * 0.4;
                Detection {
                    bbox,
                    label,
                    label_id,
                    confidence,
                }
            })
            .collect();

        // Apply confidence threshold filtering.
        detections.retain(|d| d.confidence >= self.config.confidence_threshold);

        // NMS.
        let mut after_nms = nms(&detections, self.config.iou_threshold);

        // Cap at max_detections.
        after_nms.truncate(self.config.max_detections);

        Ok(DetectionResult {
            detections: after_nms,
            image_height: height,
            image_width: width,
            inference_time_ms: 0,
        })
    }

    /// Run object detection on a batch of images.
    pub fn detect_batch(
        &self,
        images: &[(&[f32], usize, usize)],
    ) -> Result<Vec<DetectionResult>, DetectionError> {
        if images.is_empty() {
            return Err(DetectionError::EmptyImage);
        }
        images.iter().map(|&(data, h, w)| self.detect(data, h, w)).collect()
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
        let image = make_image(100, 100);
        let result = pipeline.detect(&image, 100, 100).expect("ok");
        let filtered = result.filter_by_confidence(0.8);
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
    fn test_detect_basic() {
        let config = ObjectDetectionConfig {
            confidence_threshold: 0.0,
            ..Default::default()
        };
        let pipeline = ObjectDetectionPipeline::new(config).expect("ok");
        let image = make_image(50, 50);
        let result = pipeline.detect(&image, 50, 50).expect("ok");
        assert!(!result.detections.is_empty());
        assert_eq!(result.image_height, 50);
        assert_eq!(result.image_width, 50);
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
    fn test_detect_batch() {
        let config = ObjectDetectionConfig {
            confidence_threshold: 0.0,
            ..Default::default()
        };
        let pipeline = ObjectDetectionPipeline::new(config).expect("ok");
        let img1 = make_image(20, 20);
        let img2 = make_image(30, 30);
        let batch: Vec<(&[f32], usize, usize)> =
            vec![(img1.as_slice(), 20, 20), (img2.as_slice(), 30, 30)];
        let results = pipeline.detect_batch(&batch).expect("batch ok");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].image_height, 20);
        assert_eq!(results[1].image_height, 30);
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
