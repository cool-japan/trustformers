//! # Image Segmentation Pipeline
//!
//! SegFormer-compatible semantic segmentation that assigns a class label to each pixel.
//!
//! ## Supported model families
//! - **SegFormer** — hierarchical transformer for semantic segmentation
//! - **Mask2Former** — universal image segmentation architecture
//!
//! ## Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::image_segmentation::{
//!     ImageSegmentationConfig, ImageSegmentationPipeline,
//! };
//!
//! let config = ImageSegmentationConfig::default();
//! let pipeline = ImageSegmentationPipeline::new(config)?;
//! let image = vec![0.5f32; 512 * 512 * 3];
//! let result = pipeline.segment(&image, 512, 512)?;
//! println!("Dominant class: {:?}", result.mask.dominant_class());
//! # Ok::<(), image_segmentation::SegmentationError>(())
//! ```

use std::cmp::Reverse;
use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the image segmentation pipeline.
#[derive(Debug, Error)]
pub enum SegmentationError {
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),
    #[error("Empty image")]
    EmptyImage,
    #[error("Model error: {0}")]
    ModelError(String),
    #[error("Dimension mismatch: predictions and ground truth must have the same shape")]
    DimensionMismatch,
}

// ---------------------------------------------------------------------------
// Segmentation type
// ---------------------------------------------------------------------------

/// Segmentation task type.
#[derive(Debug, Clone, PartialEq)]
pub enum SegmentationType {
    /// Assign a class label to every pixel.
    Semantic,
    /// Detect individual object instances with per-instance masks.
    Instance,
    /// Unified semantic + instance segmentation (stuff + things).
    Panoptic,
}

// ---------------------------------------------------------------------------
// BoundingBox
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box in pixel coordinates.
#[derive(Debug, Clone)]
pub struct BoundingBox {
    /// Minimum column (left).
    pub x_min: usize,
    /// Minimum row (top).
    pub y_min: usize,
    /// Maximum column (right, exclusive).
    pub x_max: usize,
    /// Maximum row (bottom, exclusive).
    pub y_max: usize,
}

impl BoundingBox {
    /// Compute the bounding box of a 2-D boolean mask.
    ///
    /// Returns `None` if no `true` pixel is present.
    pub fn from_mask(mask: &[Vec<bool>]) -> Option<Self> {
        let mut y_min = usize::MAX;
        let mut y_max = 0_usize;
        let mut x_min = usize::MAX;
        let mut x_max = 0_usize;
        let mut found = false;
        for (row_idx, row) in mask.iter().enumerate() {
            for (col_idx, &v) in row.iter().enumerate() {
                if v {
                    found = true;
                    y_min = y_min.min(row_idx);
                    y_max = y_max.max(row_idx + 1);
                    x_min = x_min.min(col_idx);
                    x_max = x_max.max(col_idx + 1);
                }
            }
        }
        if found {
            Some(Self { x_min, y_min, x_max, y_max })
        } else {
            None
        }
    }

    /// Width of the bounding box.
    pub fn width(&self) -> usize {
        self.x_max.saturating_sub(self.x_min)
    }

    /// Height of the bounding box.
    pub fn height(&self) -> usize {
        self.y_max.saturating_sub(self.y_min)
    }

    /// Area (width × height).
    pub fn area(&self) -> usize {
        self.width() * self.height()
    }
}

// ---------------------------------------------------------------------------
// SegmentationInstance
// ---------------------------------------------------------------------------

/// A single instance in instance segmentation.
#[derive(Debug, Clone)]
pub struct SegmentationInstance {
    /// 2-D boolean mask (row-major), `true` where the instance is present.
    pub mask: Vec<Vec<bool>>,
    /// Class label of this instance.
    pub label: String,
    /// Confidence / detection score.
    pub score: f32,
    /// Axis-aligned bounding box.
    pub bbox: BoundingBox,
    /// Number of pixels in the mask.
    pub area: usize,
}

// ---------------------------------------------------------------------------
// PanopticSegment
// ---------------------------------------------------------------------------

/// A single segment in panoptic segmentation.
#[derive(Debug, Clone)]
pub struct PanopticSegment {
    /// 2-D boolean mask.
    pub mask: Vec<Vec<bool>>,
    /// Semantic label.
    pub label: String,
    /// Unique segment identifier.
    pub segment_id: u32,
    /// `true` if this is a "stuff" (background / amorphous) segment,
    /// `false` if it is a "thing" (countable object) instance.
    pub is_stuff: bool,
}

// ---------------------------------------------------------------------------
// SemanticSegmentationMap
// ---------------------------------------------------------------------------

/// A full semantic segmentation map: each pixel is assigned a class index.
#[derive(Debug, Clone)]
pub struct SemanticSegmentationMap {
    /// `labels_per_pixel[row][col]` is the class index.
    pub labels_per_pixel: Vec<Vec<usize>>,
    /// Human-readable class names indexed by class id.
    pub label_names: Vec<String>,
}

impl SemanticSegmentationMap {
    /// Compute the fraction of pixels assigned to each class.
    ///
    /// Returns `(class_name, fraction)` pairs sorted by fraction descending.
    /// Classes that appear zero times are omitted.
    pub fn class_frequency(&self) -> Vec<(String, f32)> {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        let mut total = 0_usize;
        for row in &self.labels_per_pixel {
            for &cls in row {
                *counts.entry(cls).or_insert(0) += 1;
                total += 1;
            }
        }
        if total == 0 {
            return Vec::new();
        }
        let mut freq: Vec<(String, f32)> = counts
            .into_iter()
            .map(|(cls, count)| {
                let name = self
                    .label_names
                    .get(cls)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{cls}"));
                let frac = count as f32 / total as f32;
                (name, frac)
            })
            .collect();
        freq.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        freq
    }

    /// Mean Intersection-over-Union between `predictions` and `ground_truth`.
    ///
    /// mIoU = mean over all classes of (TP / (TP + FP + FN)).
    ///
    /// Only classes present in `ground_truth` are included in the average.
    ///
    /// # Errors
    /// Returns [`SegmentationError::DimensionMismatch`] if the two maps differ
    /// in shape.
    pub fn mean_iou(
        predictions: &SemanticSegmentationMap,
        ground_truth: &SemanticSegmentationMap,
    ) -> Result<f32, SegmentationError> {
        let pred_rows = predictions.labels_per_pixel.len();
        let gt_rows = ground_truth.labels_per_pixel.len();
        if pred_rows != gt_rows {
            return Err(SegmentationError::DimensionMismatch);
        }
        for (p_row, g_row) in predictions
            .labels_per_pixel
            .iter()
            .zip(ground_truth.labels_per_pixel.iter())
        {
            if p_row.len() != g_row.len() {
                return Err(SegmentationError::DimensionMismatch);
            }
        }

        // Collect unique classes from ground truth.
        let gt_classes: std::collections::HashSet<usize> = ground_truth
            .labels_per_pixel
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        if gt_classes.is_empty() {
            return Ok(0.0);
        }

        let mut iou_sum = 0.0_f32;
        for &cls in &gt_classes {
            let mut tp = 0_usize;
            let mut fp = 0_usize;
            let mut fn_ = 0_usize;
            for (p_row, g_row) in predictions
                .labels_per_pixel
                .iter()
                .zip(ground_truth.labels_per_pixel.iter())
            {
                for (&p, &g) in p_row.iter().zip(g_row.iter()) {
                    let pred_is_cls = p == cls;
                    let gt_is_cls = g == cls;
                    if pred_is_cls && gt_is_cls {
                        tp += 1;
                    } else if pred_is_cls {
                        fp += 1;
                    } else if gt_is_cls {
                        fn_ += 1;
                    }
                }
            }
            let denom = tp + fp + fn_;
            if denom > 0 {
                iou_sum += tp as f32 / denom as f32;
            }
        }
        Ok(iou_sum / gt_classes.len() as f32)
    }

    /// Height (number of rows).
    pub fn height(&self) -> usize {
        self.labels_per_pixel.len()
    }

    /// Width (number of columns) — taken from the first row; 0 if empty.
    pub fn width(&self) -> usize {
        self.labels_per_pixel.first().map(|r| r.len()).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Convert a `SemanticSegmentationMap` to a flat RGB byte buffer using a
/// deterministic colour palette (indexed by class id).
///
/// Returns a buffer of length `height * width * 3`.
pub fn apply_colormap_segmentation(seg_map: &SemanticSegmentationMap) -> Vec<u8> {
    let h = seg_map.height();
    let w = seg_map.width();
    let mut out = Vec::with_capacity(h * w * 3);
    for row in &seg_map.labels_per_pixel {
        for &cls in row {
            // Simple palette derived from class id.
            let r = ((cls * 79) % 256) as u8;
            let g = ((cls * 131) % 256) as u8;
            let b = ((cls * 197) % 256) as u8;
            out.push(r);
            out.push(g);
            out.push(b);
        }
    }
    out
}

/// Convert a set of binary instance masks (plus corresponding class ids) into
/// a `SemanticSegmentationMap`.
///
/// Later masks in the slice override earlier ones for overlapping pixels.
///
/// # Parameters
/// - `masks`: slice of 2-D boolean masks, each `true` where the instance is present.
/// - `class_ids`: class id for each mask (must be the same length as `masks`).
/// - `h`, `w`: output map dimensions.
/// - `label_names`: class name strings indexed by class id.
pub fn masks_to_semantic(
    masks: &[Vec<Vec<bool>>],
    class_ids: &[usize],
    h: usize,
    w: usize,
    label_names: Vec<String>,
) -> SemanticSegmentationMap {
    // Background class = 0 (unlabelled).
    let mut pixel_labels = vec![vec![0_usize; w]; h];
    for (mask, &cls) in masks.iter().zip(class_ids.iter()) {
        for (row_idx, row) in mask.iter().enumerate() {
            if row_idx >= h {
                break;
            }
            for (col_idx, &v) in row.iter().enumerate() {
                if col_idx >= w {
                    break;
                }
                if v {
                    pixel_labels[row_idx][col_idx] = cls;
                }
            }
        }
    }
    SemanticSegmentationMap {
        labels_per_pixel: pixel_labels,
        label_names,
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`ImageSegmentationPipeline`].
#[derive(Debug, Clone)]
pub struct ImageSegmentationConfig {
    /// HuggingFace model identifier or local path.
    pub model_name: String,
    /// Number of output classes (150 = ADE20K).
    pub num_classes: usize,
    /// Model input height.
    pub input_height: usize,
    /// Model input width.
    pub input_width: usize,
    /// Output is downsampled by this factor (e.g. 4 → output is 1/4 input resolution).
    pub output_stride: usize,
    /// If `true`, the pipeline also returns a per-pixel maximum confidence map.
    pub return_confidence_map: bool,
    /// Which segmentation type this pipeline produces.
    pub segmentation_type: SegmentationType,
}

impl Default for ImageSegmentationConfig {
    fn default() -> Self {
        Self {
            model_name: "nvidia/segformer-b0-finetuned-ade-512-512".to_string(),
            num_classes: 150,
            input_height: 512,
            input_width: 512,
            output_stride: 4,
            return_confidence_map: false,
            segmentation_type: SegmentationType::Semantic,
        }
    }
}

// ---------------------------------------------------------------------------
// ImageInput
// ---------------------------------------------------------------------------

/// A simple image input wrapper.
#[derive(Debug, Clone)]
pub struct ImageInput {
    /// Flat f32 pixel buffer in row-major `[H * W * channels]` format.
    pub data: Vec<f32>,
    /// Image height.
    pub height: usize,
    /// Image width.
    pub width: usize,
}

impl ImageInput {
    /// Construct an `ImageInput`.
    pub fn new(data: Vec<f32>, height: usize, width: usize) -> Self {
        Self { data, height, width }
    }
}

// ---------------------------------------------------------------------------
// SegmentationMask
// ---------------------------------------------------------------------------

/// A 2-D class-id mask produced by the segmentation pipeline.
#[derive(Debug, Clone)]
pub struct SegmentationMask {
    /// Flattened class-id map in row-major order, shape `[height, width]`.
    pub class_ids: Vec<u32>,
    /// Mask height (= `input_height / output_stride`).
    pub height: usize,
    /// Mask width (= `input_width / output_stride`).
    pub width: usize,
    /// Total number of classes the model knows about.
    pub num_classes: usize,
}

impl SegmentationMask {
    /// Create a new `SegmentationMask`.
    pub fn new(class_ids: Vec<u32>, height: usize, width: usize, num_classes: usize) -> Self {
        Self { class_ids, height, width, num_classes }
    }

    /// Return the class ID at `(row, col)`.
    ///
    /// Returns `0` for out-of-bounds indices.
    pub fn get(&self, row: usize, col: usize) -> u32 {
        let idx = row * self.width + col;
        self.class_ids.get(idx).copied().unwrap_or(0)
    }

    /// Count the number of pixels for each class ID.
    pub fn class_pixel_counts(&self) -> HashMap<u32, usize> {
        let mut counts: HashMap<u32, usize> = HashMap::new();
        for &id in &self.class_ids {
            *counts.entry(id).or_insert(0) += 1;
        }
        counts
    }

    /// Return the class ID that covers the most pixels, or `None` if the mask is empty.
    pub fn dominant_class(&self) -> Option<u32> {
        self.class_pixel_counts()
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(id, _)| id)
    }

    /// Nearest-neighbour upsample by `scale` in both dimensions.
    pub fn upsample(&self, scale: usize) -> Self {
        if scale == 0 {
            return self.clone();
        }
        let new_h = self.height * scale;
        let new_w = self.width * scale;
        let mut out = vec![0u32; new_h * new_w];
        for oy in 0..new_h {
            for ox in 0..new_w {
                let sy = oy / scale;
                let sx = ox / scale;
                out[oy * new_w + ox] = self.get(sy, sx);
            }
        }
        Self::new(out, new_h, new_w, self.num_classes)
    }

    /// Render the mask as ASCII art where each pixel is represented by `class_id % 10`.
    pub fn to_ascii(&self) -> String {
        let mut out = String::with_capacity(self.height * (self.width + 1));
        for row in 0..self.height {
            for col in 0..self.width {
                let id = self.get(row, col);
                let ch = char::from_digit(id % 10, 10).unwrap_or('?');
                out.push(ch);
            }
            out.push('\n');
        }
        out
    }
}

// ---------------------------------------------------------------------------
// SegmentStats
// ---------------------------------------------------------------------------

/// Per-class statistics for a segmentation result.
#[derive(Debug, Clone)]
pub struct SegmentStats {
    /// Class identifier.
    pub class_id: u32,
    /// Human-readable class name.
    pub class_name: String,
    /// Number of pixels labelled with this class.
    pub pixel_count: usize,
    /// Fraction of the total pixels covered by this class.
    pub coverage_ratio: f32,
}

// ---------------------------------------------------------------------------
// SegmentationResult
// ---------------------------------------------------------------------------

/// The full output of one segmentation pass.
#[derive(Debug, Clone)]
pub struct SegmentationResult {
    /// The predicted class-id mask.
    pub mask: SegmentationMask,
    /// Optional per-pixel maximum confidence scores.
    pub confidence_map: Option<Vec<f32>>,
    /// Class names indexed by class ID.
    pub class_names: Vec<String>,
    /// Approximate inference duration in milliseconds.
    pub inference_time_ms: u64,
}

impl SegmentationResult {
    /// Compute per-class coverage statistics, sorted by pixel count (descending).
    pub fn segment_stats(&self) -> Vec<SegmentStats> {
        let total_pixels = self.mask.height * self.mask.width;
        let counts = self.mask.class_pixel_counts();
        let mut stats: Vec<SegmentStats> = counts
            .into_iter()
            .map(|(class_id, pixel_count)| {
                let class_name = self
                    .class_names
                    .get(class_id as usize)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{class_id}"));
                let coverage_ratio = if total_pixels == 0 {
                    0.0
                } else {
                    pixel_count as f32 / total_pixels as f32
                };
                SegmentStats { class_id, class_name, pixel_count, coverage_ratio }
            })
            .collect();
        stats.sort_by_key(|s| Reverse(s.pixel_count));
        stats
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// SegFormer-compatible semantic segmentation pipeline.
pub struct ImageSegmentationPipeline {
    config: ImageSegmentationConfig,
    class_names: Vec<String>,
}

impl ImageSegmentationPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: ImageSegmentationConfig) -> Result<Self, SegmentationError> {
        if config.input_height == 0 || config.input_width == 0 {
            return Err(SegmentationError::InvalidDimensions(
                "input_height and input_width must be > 0".to_string(),
            ));
        }
        if config.output_stride == 0 {
            return Err(SegmentationError::InvalidDimensions(
                "output_stride must be > 0".to_string(),
            ));
        }
        // Generate synthetic class names for the first `num_classes` slots.
        let class_names: Vec<String> = (0..config.num_classes)
            .map(|i| format!("class_{i}"))
            .collect();
        Ok(Self { config, class_names })
    }

    /// Run semantic segmentation on a single image.
    ///
    /// `image` is a flat `f32` buffer with shape `[height * width * channels]`,
    /// values typically in `[0, 1]`.
    pub fn segment(
        &self,
        image: &[f32],
        height: usize,
        width: usize,
    ) -> Result<SegmentationResult, SegmentationError> {
        if image.is_empty() {
            return Err(SegmentationError::EmptyImage);
        }
        if height == 0 || width == 0 {
            return Err(SegmentationError::InvalidDimensions(
                "height and width must be > 0".to_string(),
            ));
        }

        let out_h = (self.config.input_height / self.config.output_stride).max(1);
        let out_w = (self.config.input_width / self.config.output_stride).max(1);

        // Mock: class assignment derived from pixel value at the nearest source location.
        // class_id = (pixel_value * num_classes) as u32, clamped to [0, num_classes - 1].
        let num_classes = self.config.num_classes as u32;
        let mut class_ids = Vec::with_capacity(out_h * out_w);
        let mut confidence_map: Option<Vec<f32>> =
            if self.config.return_confidence_map { Some(Vec::with_capacity(out_h * out_w)) } else { None };

        let channels = image.len().checked_div(height * width).unwrap_or(1).max(1);

        for oy in 0..out_h {
            for ox in 0..out_w {
                // Map output pixel to source image pixel.
                let sy = (oy * self.config.output_stride).min(height - 1);
                let sx = (ox * self.config.output_stride).min(width - 1);
                let src_idx = (sy * width + sx) * channels;
                let pixel_val = image.get(src_idx).copied().unwrap_or(0.0);

                let class_id = if num_classes == 0 {
                    0u32
                } else {
                    ((pixel_val.clamp(0.0, 1.0) * num_classes as f32) as u32)
                        .min(num_classes - 1)
                };
                class_ids.push(class_id);

                if let Some(ref mut cm) = confidence_map {
                    // Confidence is a simple function of distance from class boundary.
                    let fractional = pixel_val.clamp(0.0, 1.0) * num_classes as f32;
                    let dist_to_boundary = (fractional - fractional.round()).abs();
                    let conf = 1.0 - dist_to_boundary * 2.0;
                    cm.push(conf.clamp(0.0, 1.0));
                }
            }
        }

        let mask = SegmentationMask::new(class_ids, out_h, out_w, self.config.num_classes);
        Ok(SegmentationResult {
            mask,
            confidence_map,
            class_names: self.class_names.clone(),
            inference_time_ms: 0,
        })
    }

    /// Run semantic segmentation on a batch of images.
    ///
    /// Each element is `(image_data, height, width)`.
    pub fn segment_batch(
        &self,
        images: &[(&[f32], usize, usize)],
    ) -> Result<Vec<SegmentationResult>, SegmentationError> {
        if images.is_empty() {
            return Err(SegmentationError::EmptyImage);
        }
        images
            .iter()
            .map(|&(data, h, w)| self.segment(data, h, w))
            .collect()
    }

    /// Run semantic segmentation on an [`ImageInput`], returning a [`SemanticSegmentationMap`].
    pub fn segment_semantic(
        &self,
        input: &ImageInput,
    ) -> Result<SemanticSegmentationMap, SegmentationError> {
        let result = self.segment(&input.data, input.height, input.width)?;
        let h = result.mask.height;
        let w = result.mask.width;
        let mut labels_per_pixel = vec![vec![0_usize; w]; h];
        for row in 0..h {
            for col in 0..w {
                labels_per_pixel[row][col] = result.mask.get(row, col) as usize;
            }
        }
        Ok(SemanticSegmentationMap {
            labels_per_pixel,
            label_names: result.class_names,
        })
    }

    /// Run instance segmentation on an [`ImageInput`].
    ///
    /// Returns one [`SegmentationInstance`] per detected class region, derived
    /// from the semantic segmentation mask by treating each contiguous class as
    /// a separate instance (mock implementation).
    pub fn segment_instance(
        &self,
        input: &ImageInput,
    ) -> Result<Vec<SegmentationInstance>, SegmentationError> {
        let result = self.segment(&input.data, input.height, input.width)?;
        let h = result.mask.height;
        let w = result.mask.width;

        // Group pixels by class id → one instance per class.
        let counts = result.mask.class_pixel_counts();
        let total = h * w;

        let mut instances: Vec<SegmentationInstance> = counts
            .into_iter()
            .map(|(class_id, pixel_count)| {
                let mut mask = vec![vec![false; w]; h];
                for row in 0..h {
                    for col in 0..w {
                        if result.mask.get(row, col) == class_id {
                            mask[row][col] = true;
                        }
                    }
                }
                let bbox = BoundingBox::from_mask(&mask).unwrap_or(BoundingBox {
                    x_min: 0,
                    y_min: 0,
                    x_max: w,
                    y_max: h,
                });
                let label = result
                    .class_names
                    .get(class_id as usize)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{class_id}"));
                let score = if total > 0 {
                    pixel_count as f32 / total as f32
                } else {
                    0.0
                };
                SegmentationInstance {
                    mask,
                    label,
                    score,
                    bbox,
                    area: pixel_count,
                }
            })
            .collect();

        // Sort by area descending for stable output.
        instances.sort_by(|a, b| b.area.cmp(&a.area));
        Ok(instances)
    }

    /// Run panoptic segmentation on an [`ImageInput`].
    ///
    /// Mock: classes with even id are treated as "stuff" (background), odd as "things".
    pub fn segment_panoptic(
        &self,
        input: &ImageInput,
    ) -> Result<Vec<PanopticSegment>, SegmentationError> {
        let result = self.segment(&input.data, input.height, input.width)?;
        let h = result.mask.height;
        let w = result.mask.width;

        let counts = result.mask.class_pixel_counts();
        let mut segments: Vec<PanopticSegment> = counts
            .into_iter()
            .enumerate()
            .map(|(seg_idx, (class_id, _))| {
                let mut mask = vec![vec![false; w]; h];
                for row in 0..h {
                    for col in 0..w {
                        if result.mask.get(row, col) == class_id {
                            mask[row][col] = true;
                        }
                    }
                }
                let label = result
                    .class_names
                    .get(class_id as usize)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{class_id}"));
                PanopticSegment {
                    mask,
                    label,
                    segment_id: seg_idx as u32,
                    is_stuff: class_id % 2 == 0,
                }
            })
            .collect();

        segments.sort_by_key(|s| s.segment_id);
        Ok(segments)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_image(h: usize, w: usize) -> Vec<f32> {
        (0..h * w * 3).map(|i| (i % 100) as f32 / 100.0).collect()
    }

    // ---- 1. config defaults ----

    #[test]
    fn test_config_defaults() {
        let cfg = ImageSegmentationConfig::default();
        assert_eq!(cfg.num_classes, 150);
        assert_eq!(cfg.input_height, 512);
        assert_eq!(cfg.input_width, 512);
        assert_eq!(cfg.output_stride, 4);
        assert!(!cfg.return_confidence_map);
    }

    // ---- 2. mask get ----

    #[test]
    fn test_mask_get() {
        let ids = vec![0u32, 1, 2, 3];
        let mask = SegmentationMask::new(ids, 2, 2, 10);
        assert_eq!(mask.get(0, 0), 0);
        assert_eq!(mask.get(0, 1), 1);
        assert_eq!(mask.get(1, 0), 2);
        assert_eq!(mask.get(1, 1), 3);
        // out of bounds → 0
        assert_eq!(mask.get(5, 5), 0);
    }

    // ---- 3. class_pixel_counts ----

    #[test]
    fn test_class_pixel_counts() {
        let ids = vec![0u32, 1, 0, 2, 0, 1];
        let mask = SegmentationMask::new(ids, 2, 3, 5);
        let counts = mask.class_pixel_counts();
        assert_eq!(*counts.get(&0).unwrap(), 3);
        assert_eq!(*counts.get(&1).unwrap(), 2);
        assert_eq!(*counts.get(&2).unwrap(), 1);
    }

    // ---- 4. dominant_class ----

    #[test]
    fn test_dominant_class() {
        let ids = vec![0u32, 1, 0, 0, 2, 1];
        let mask = SegmentationMask::new(ids, 2, 3, 5);
        assert_eq!(mask.dominant_class(), Some(0));
    }

    // ---- 5. dominant_class empty ----

    #[test]
    fn test_dominant_class_empty() {
        let mask = SegmentationMask::new(vec![], 0, 0, 5);
        assert_eq!(mask.dominant_class(), None);
    }

    // ---- 6. upsample ----

    #[test]
    fn test_upsample_dimensions() {
        let ids = vec![1u32, 2, 3, 4];
        let mask = SegmentationMask::new(ids, 2, 2, 10);
        let up = mask.upsample(3);
        assert_eq!(up.height, 6);
        assert_eq!(up.width, 6);
        assert_eq!(up.class_ids.len(), 36);
    }

    #[test]
    fn test_upsample_values() {
        let ids = vec![7u32];
        let mask = SegmentationMask::new(ids, 1, 1, 10);
        let up = mask.upsample(4);
        assert!(up.class_ids.iter().all(|&v| v == 7));
    }

    // ---- 7. to_ascii ----

    #[test]
    fn test_to_ascii_shape() {
        let ids = vec![0u32; 3 * 5];
        let mask = SegmentationMask::new(ids, 3, 5, 10);
        let ascii = mask.to_ascii();
        let lines: Vec<&str> = ascii.lines().collect();
        assert_eq!(lines.len(), 3);
        for line in &lines {
            assert_eq!(line.len(), 5);
        }
    }

    #[test]
    fn test_to_ascii_values() {
        let ids = vec![0u32, 13, 5, 21];
        let mask = SegmentationMask::new(ids, 2, 2, 30);
        let ascii = mask.to_ascii();
        let chars: Vec<char> = ascii.chars().filter(|&c| c != '\n').collect();
        assert_eq!(chars[0], '0');
        assert_eq!(chars[1], '3'); // 13 % 10
        assert_eq!(chars[2], '5');
        assert_eq!(chars[3], '1'); // 21 % 10
    }

    // ---- 8. segment basic ----

    #[test]
    fn test_segment_basic() {
        let config = ImageSegmentationConfig {
            input_height: 16,
            input_width: 16,
            output_stride: 4,
            num_classes: 10,
            ..Default::default()
        };
        let pipeline = ImageSegmentationPipeline::new(config).unwrap();
        let image = make_image(16, 16);
        let result = pipeline.segment(&image, 16, 16).unwrap();
        assert_eq!(result.mask.height, 4);
        assert_eq!(result.mask.width, 4);
        assert_eq!(result.mask.class_ids.len(), 16);
        assert!(result.confidence_map.is_none());
    }

    // ---- 9. segment_stats coverage_ratio ----

    #[test]
    fn test_segment_stats_coverage() {
        let ids = vec![0u32, 0, 1, 1, 0, 0, 1, 0, 0]; // 6 zeros, 3 ones
        let mask = SegmentationMask::new(ids, 3, 3, 5);
        let result = SegmentationResult {
            mask,
            confidence_map: None,
            class_names: vec!["a".into(), "b".into(), "c".into()],
            inference_time_ms: 0,
        };
        let stats = result.segment_stats();
        let class0 = stats.iter().find(|s| s.class_id == 0).unwrap();
        let class1 = stats.iter().find(|s| s.class_id == 1).unwrap();
        assert!((class0.coverage_ratio - 6.0 / 9.0).abs() < 1e-5);
        assert!((class1.coverage_ratio - 3.0 / 9.0).abs() < 1e-5);
    }

    // ---- 10. segment_batch ----

    #[test]
    fn test_segment_batch() {
        let config = ImageSegmentationConfig {
            input_height: 8,
            input_width: 8,
            output_stride: 2,
            num_classes: 5,
            ..Default::default()
        };
        let pipeline = ImageSegmentationPipeline::new(config).unwrap();
        let img1 = make_image(8, 8);
        let img2 = make_image(12, 12);
        let batch: Vec<(&[f32], usize, usize)> = vec![
            (img1.as_slice(), 8, 8),
            (img2.as_slice(), 12, 12),
        ];
        let results = pipeline.segment_batch(&batch).unwrap();
        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(r.mask.height, 4);
            assert_eq!(r.mask.width, 4);
        }
    }

    // ---- 11. confidence_map returned when configured ----

    #[test]
    fn test_confidence_map_returned() {
        let config = ImageSegmentationConfig {
            input_height: 8,
            input_width: 8,
            output_stride: 4,
            num_classes: 10,
            return_confidence_map: true,
            ..Default::default()
        };
        let pipeline = ImageSegmentationPipeline::new(config).unwrap();
        let image = make_image(8, 8);
        let result = pipeline.segment(&image, 8, 8).unwrap();
        assert!(result.confidence_map.is_some());
        let cm = result.confidence_map.unwrap();
        assert_eq!(cm.len(), result.mask.height * result.mask.width);
        assert!(cm.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    // ---- 12. segment empty input error ----

    #[test]
    fn test_segment_empty_input() {
        let pipeline =
            ImageSegmentationPipeline::new(ImageSegmentationConfig::default()).unwrap();
        let result = pipeline.segment(&[], 10, 10);
        assert!(matches!(result, Err(SegmentationError::EmptyImage)));
    }

    // ---- 13. segment invalid dimensions ----

    #[test]
    fn test_segment_invalid_dimensions() {
        let pipeline =
            ImageSegmentationPipeline::new(ImageSegmentationConfig::default()).unwrap();
        let result = pipeline.segment(&[0.5f32; 10], 0, 10);
        assert!(matches!(result, Err(SegmentationError::InvalidDimensions(_))));
    }

    // ---- 14. SemanticSegmentationMap::class_frequency ----

    #[test]
    fn test_class_frequency_basic() {
        let map = SemanticSegmentationMap {
            labels_per_pixel: vec![vec![0, 1, 0], vec![1, 1, 0]],
            label_names: vec!["bg".to_string(), "fg".to_string()],
        };
        let freq = map.class_frequency();
        // "fg" (3/6) should beat "bg" (3/6) — tied, but we have both.
        assert_eq!(freq.len(), 2);
        let total: f32 = freq.iter().map(|(_, f)| f).sum();
        assert!((total - 1.0).abs() < 1e-5, "frequencies must sum to 1");
    }

    #[test]
    fn test_class_frequency_single_class() {
        let map = SemanticSegmentationMap {
            labels_per_pixel: vec![vec![3, 3], vec![3, 3]],
            label_names: vec!["a".into(), "b".into(), "c".into(), "d".into()],
        };
        let freq = map.class_frequency();
        assert_eq!(freq.len(), 1);
        assert!((freq[0].1 - 1.0).abs() < 1e-5);
        assert_eq!(freq[0].0, "d");
    }

    #[test]
    fn test_class_frequency_empty_map() {
        let map = SemanticSegmentationMap {
            labels_per_pixel: vec![],
            label_names: vec![],
        };
        assert!(map.class_frequency().is_empty());
    }

    // ---- 15. SemanticSegmentationMap::mean_iou ----

    #[test]
    fn test_mean_iou_perfect_match() {
        let data = vec![vec![0_usize, 1], vec![1, 0]];
        let pred = SemanticSegmentationMap {
            labels_per_pixel: data.clone(),
            label_names: vec!["a".into(), "b".into()],
        };
        let gt = SemanticSegmentationMap {
            labels_per_pixel: data,
            label_names: vec!["a".into(), "b".into()],
        };
        let miou = SemanticSegmentationMap::mean_iou(&pred, &gt).expect("ok");
        assert!((miou - 1.0).abs() < 1e-5, "perfect prediction: mIoU={miou}");
    }

    #[test]
    fn test_mean_iou_completely_wrong() {
        // All pixels predicted as class 1, all GT pixels are class 0.
        let pred_data = vec![vec![1_usize, 1], vec![1, 1]];
        let gt_data = vec![vec![0_usize, 0], vec![0, 0]];
        let pred = SemanticSegmentationMap {
            labels_per_pixel: pred_data,
            label_names: vec!["a".into(), "b".into()],
        };
        let gt = SemanticSegmentationMap {
            labels_per_pixel: gt_data,
            label_names: vec!["a".into(), "b".into()],
        };
        let miou = SemanticSegmentationMap::mean_iou(&pred, &gt).expect("ok");
        assert!((miou - 0.0).abs() < 1e-5, "completely wrong: mIoU={miou}");
    }

    #[test]
    fn test_mean_iou_dimension_mismatch() {
        let pred = SemanticSegmentationMap {
            labels_per_pixel: vec![vec![0_usize, 1]],
            label_names: vec![],
        };
        let gt = SemanticSegmentationMap {
            labels_per_pixel: vec![vec![0_usize], vec![1]],
            label_names: vec![],
        };
        let result = SemanticSegmentationMap::mean_iou(&pred, &gt);
        assert!(matches!(result, Err(SegmentationError::DimensionMismatch)));
    }

    // ---- 16. apply_colormap_segmentation ----

    #[test]
    fn test_apply_colormap_segmentation_length() {
        let map = SemanticSegmentationMap {
            labels_per_pixel: vec![vec![0, 1, 2], vec![3, 4, 5]],
            label_names: (0..6).map(|i| format!("c{i}")).collect(),
        };
        let pixels = apply_colormap_segmentation(&map);
        assert_eq!(pixels.len(), 2 * 3 * 3, "should be h*w*3 bytes");
    }

    #[test]
    fn test_apply_colormap_class_zero_is_black() {
        // Class 0 → R=0, G=0, B=0 since (0*79)%256 = 0 etc.
        let map = SemanticSegmentationMap {
            labels_per_pixel: vec![vec![0]],
            label_names: vec!["bg".into()],
        };
        let pixels = apply_colormap_segmentation(&map);
        assert_eq!(pixels.len(), 3);
        assert_eq!((pixels[0], pixels[1], pixels[2]), (0, 0, 0));
    }

    #[test]
    fn test_apply_colormap_different_classes_different_colors() {
        let map = SemanticSegmentationMap {
            labels_per_pixel: vec![vec![1, 2]],
            label_names: vec!["bg".into(), "a".into(), "b".into()],
        };
        let pixels = apply_colormap_segmentation(&map);
        assert_eq!(pixels.len(), 6);
        let c1 = (pixels[0], pixels[1], pixels[2]);
        let c2 = (pixels[3], pixels[4], pixels[5]);
        assert_ne!(c1, c2, "different classes should produce different colours");
    }

    // ---- 17. masks_to_semantic ----

    #[test]
    fn test_masks_to_semantic_basic() {
        // 2x2 output; mask 0 covers (0,0), mask 1 covers (1,1).
        let masks = vec![
            vec![vec![true, false], vec![false, false]],
            vec![vec![false, false], vec![false, true]],
        ];
        let class_ids = vec![1_usize, 2];
        let label_names = vec!["bg".into(), "a".into(), "b".into()];
        let map = masks_to_semantic(&masks, &class_ids, 2, 2, label_names);
        assert_eq!(map.labels_per_pixel[0][0], 1);
        assert_eq!(map.labels_per_pixel[1][1], 2);
        // Unset pixels should be 0 (background).
        assert_eq!(map.labels_per_pixel[0][1], 0);
    }

    #[test]
    fn test_masks_to_semantic_overlap_last_wins() {
        // Two masks covering the same pixel — later one should win.
        let masks = vec![
            vec![vec![true]],
            vec![vec![true]],
        ];
        let class_ids = vec![1_usize, 2];
        let label_names = vec!["bg".into(), "a".into(), "b".into()];
        let map = masks_to_semantic(&masks, &class_ids, 1, 1, label_names);
        assert_eq!(map.labels_per_pixel[0][0], 2);
    }

    // ---- 18. SegmentationType variants ----

    #[test]
    fn test_segmentation_type_variants() {
        assert_eq!(SegmentationType::Semantic, SegmentationType::Semantic);
        assert_ne!(SegmentationType::Semantic, SegmentationType::Instance);
        assert_ne!(SegmentationType::Instance, SegmentationType::Panoptic);
    }

    // ---- 19. BoundingBox::from_mask ----

    #[test]
    fn test_bounding_box_from_mask_basic() {
        let mask = vec![
            vec![false, false, false],
            vec![false, true,  true ],
            vec![false, false, true ],
        ];
        let bbox = BoundingBox::from_mask(&mask).expect("should find bbox");
        assert_eq!(bbox.x_min, 1);
        assert_eq!(bbox.x_max, 3);
        assert_eq!(bbox.y_min, 1);
        assert_eq!(bbox.y_max, 3);
    }

    #[test]
    fn test_bounding_box_from_mask_empty() {
        let mask = vec![vec![false, false], vec![false, false]];
        assert!(BoundingBox::from_mask(&mask).is_none());
    }

    // ---- 20. segment_semantic ----

    #[test]
    fn test_segment_semantic_shape() {
        let config = ImageSegmentationConfig {
            input_height: 8,
            input_width: 8,
            output_stride: 4,
            num_classes: 5,
            ..Default::default()
        };
        let pipeline = ImageSegmentationPipeline::new(config).expect("ok");
        let input = ImageInput::new(make_image(8, 8), 8, 8);
        let map = pipeline.segment_semantic(&input).expect("ok");
        assert_eq!(map.height(), 2);
        assert_eq!(map.width(), 2);
    }

    // ---- 21. segment_instance ----

    #[test]
    fn test_segment_instance_non_empty() {
        let config = ImageSegmentationConfig {
            input_height: 8,
            input_width: 8,
            output_stride: 4,
            num_classes: 5,
            ..Default::default()
        };
        let pipeline = ImageSegmentationPipeline::new(config).expect("ok");
        let input = ImageInput::new(make_image(8, 8), 8, 8);
        let instances = pipeline.segment_instance(&input).expect("ok");
        assert!(!instances.is_empty(), "should return at least one instance");
        for inst in &instances {
            assert!(inst.area > 0, "each instance must have non-zero area");
            assert!(inst.score > 0.0);
        }
    }

    // ---- 22. segment_panoptic ----

    #[test]
    fn test_segment_panoptic_non_empty() {
        let config = ImageSegmentationConfig {
            input_height: 8,
            input_width: 8,
            output_stride: 4,
            num_classes: 6,
            ..Default::default()
        };
        let pipeline = ImageSegmentationPipeline::new(config).expect("ok");
        let input = ImageInput::new(make_image(8, 8), 8, 8);
        let segments = pipeline.segment_panoptic(&input).expect("ok");
        assert!(!segments.is_empty(), "should return at least one segment");
        // At least one stuff and possibly one thing.
        let has_stuff = segments.iter().any(|s| s.is_stuff);
        let has_thing = segments.iter().any(|s| !s.is_stuff);
        assert!(has_stuff || has_thing, "segments should be classified");
    }

    // ---- 23. SegmentationConfig includes segmentation_type ----

    #[test]
    fn test_config_includes_segmentation_type() {
        let config = ImageSegmentationConfig::default();
        assert_eq!(config.segmentation_type, SegmentationType::Semantic);
    }

    // ---- 24. class_frequency sorted descending ----

    #[test]
    fn test_class_frequency_sorted_descending() {
        let map = SemanticSegmentationMap {
            labels_per_pixel: vec![
                vec![0, 0, 0, 1],
                vec![0, 0, 2, 1],
            ],
            label_names: vec!["a".into(), "b".into(), "c".into()],
        };
        let freq = map.class_frequency();
        for w in freq.windows(2) {
            assert!(w[0].1 >= w[1].1, "frequency not sorted descending");
        }
    }

    // ---- 25. mean_iou partial overlap ----

    #[test]
    fn test_mean_iou_partial_overlap() {
        // 2x2 grid: class 0 everywhere in GT; pred has class 0 in (0,0) and (0,1), class 1 in (1,0) and (1,1).
        let pred = SemanticSegmentationMap {
            labels_per_pixel: vec![vec![0_usize, 0], vec![1, 1]],
            label_names: vec!["a".into(), "b".into()],
        };
        let gt = SemanticSegmentationMap {
            labels_per_pixel: vec![vec![0_usize, 0], vec![0, 0]],
            label_names: vec!["a".into(), "b".into()],
        };
        let miou = SemanticSegmentationMap::mean_iou(&pred, &gt).expect("ok");
        // Only class 0 in GT. IoU for class 0 = TP/(TP+FP+FN) = 2/(2+2+0) = 0.5.
        assert!((miou - 0.5).abs() < 1e-5, "partial overlap mIoU expected 0.5, got {miou}");
    }
}
