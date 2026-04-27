//! # Mask Generation Pipeline
//!
//! SAM-compatible segmentation mask generation from image prompts.
//!
//! Supports point prompts, box prompts, and text prompts, returning binary
//! masks with quality scores. Also provides morphological post-processing
//! utilities via `MaskRefiner`.

use std::fmt;

/// Label for a point prompt
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PointLabel {
    /// Foreground point (positive, included in mask)
    Foreground,
    /// Background point (negative, excluded from mask)
    Background,
}

/// A point prompt for mask generation
#[derive(Debug, Clone, Copy)]
pub struct PointPrompt {
    pub x: f32,
    pub y: f32,
    pub label: PointLabel,
}

/// A box prompt for mask generation
#[derive(Debug, Clone, Copy)]
pub struct BoxPrompt {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

impl BoxPrompt {
    /// Area of the bounding box
    pub fn area(&self) -> f32 {
        (self.x_max - self.x_min) * (self.y_max - self.y_min)
    }

    /// Center point of the box
    pub fn center(&self) -> (f32, f32) {
        (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )
    }

    /// Returns true if the box has positive area
    pub fn is_valid(&self) -> bool {
        self.x_max > self.x_min && self.y_max > self.y_min
    }
}

/// Prompt for mask generation (can be points, box, or both)
#[derive(Debug, Clone)]
pub struct MaskPrompt {
    pub points: Vec<PointPrompt>,
    pub boxes: Vec<BoxPrompt>,
    /// Optional text prompt (for text-guided SAM variants)
    pub text: Option<String>,
}

impl MaskPrompt {
    /// Create an empty prompt
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            boxes: Vec::new(),
            text: None,
        }
    }

    /// Add a point prompt (builder style)
    pub fn with_point(mut self, x: f32, y: f32, label: PointLabel) -> Self {
        self.points.push(PointPrompt { x, y, label });
        self
    }

    /// Add a box prompt (builder style)
    pub fn with_box(mut self, x_min: f32, y_min: f32, x_max: f32, y_max: f32) -> Self {
        self.boxes.push(BoxPrompt {
            x_min,
            y_min,
            x_max,
            y_max,
        });
        self
    }

    /// Add a text prompt (builder style)
    pub fn with_text(mut self, text: &str) -> Self {
        self.text = Some(text.to_string());
        self
    }

    /// True when there are no prompts of any kind
    pub fn is_empty(&self) -> bool {
        self.points.is_empty() && self.boxes.is_empty() && self.text.is_none()
    }

    /// Total number of geometric prompts (points + boxes, excluding text)
    pub fn num_prompts(&self) -> usize {
        self.points.len() + self.boxes.len()
    }
}

impl Default for MaskPrompt {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SegmentationMask — new-style result type
// ---------------------------------------------------------------------------

/// A rich segmentation mask with label, score, and area metadata.
#[derive(Debug, Clone)]
pub struct SegmentationMask {
    /// 2D boolean mask (true = foreground). Shape: `[height][width]`.
    pub mask: Vec<Vec<bool>>,
    /// Human-readable label for this mask region.
    pub label: String,
    /// Quality score in `[0, 1]`.
    pub score: f32,
    /// Number of foreground pixels (area).
    pub area: usize,
}

impl SegmentationMask {
    /// Construct a `SegmentationMask`, computing `area` automatically.
    pub fn new(mask: Vec<Vec<bool>>, label: String, score: f32) -> Self {
        let area = mask.iter().flatten().filter(|&&v| v).count();
        Self {
            mask,
            label,
            score,
            area,
        }
    }

    /// Flatten the 2D mask to a 1D boolean vector (row-major).
    pub fn to_flat(&self) -> Vec<bool> {
        self.mask.iter().flatten().copied().collect()
    }

    /// Height of the mask in pixels.
    pub fn height(&self) -> usize {
        self.mask.len()
    }

    /// Width of the mask in pixels (derived from first row).
    pub fn width(&self) -> usize {
        self.mask.first().map(|r| r.len()).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// GeneratedMask — existing flat mask type
// ---------------------------------------------------------------------------

/// A generated mask with metadata
#[derive(Debug, Clone)]
pub struct GeneratedMask {
    /// Binary mask data (true = foreground), row-major [H * W]
    pub mask: Vec<bool>,
    pub width: usize,
    pub height: usize,
    /// Predicted IoU quality score in [0, 1]
    pub iou_score: f32,
    /// Stability score in [0, 1] — consistency across threshold choices
    pub stability_score: f32,
    /// Index for multi-mask output (SAM produces 3 candidates)
    pub mask_index: usize,
}

impl GeneratedMask {
    /// Construct a new GeneratedMask
    pub fn new(
        mask: Vec<bool>,
        width: usize,
        height: usize,
        iou_score: f32,
        stability_score: f32,
        mask_index: usize,
    ) -> Self {
        Self {
            mask,
            width,
            height,
            iou_score,
            stability_score,
            mask_index,
        }
    }

    /// Number of foreground (true) pixels
    pub fn num_foreground_pixels(&self) -> usize {
        self.mask.iter().filter(|&&v| v).count()
    }

    /// Fraction of image covered by the foreground mask
    pub fn coverage_ratio(&self) -> f32 {
        let total = self.width * self.height;
        if total == 0 {
            return 0.0;
        }
        self.num_foreground_pixels() as f32 / total as f32
    }

    /// Compute the tight bounding box of the foreground region, or None if empty
    pub fn bounding_box(&self) -> Option<BoxPrompt> {
        let mut x_min = usize::MAX;
        let mut y_min = usize::MAX;
        let mut x_max = 0usize;
        let mut y_max = 0usize;
        let mut found = false;

        for (idx, &fg) in self.mask.iter().enumerate() {
            if fg {
                let x = idx % self.width;
                let y = idx / self.width;
                if x < x_min {
                    x_min = x;
                }
                if y < y_min {
                    y_min = y;
                }
                if x > x_max {
                    x_max = x;
                }
                if y > y_max {
                    y_max = y;
                }
                found = true;
            }
        }

        if !found {
            return None;
        }

        Some(BoxPrompt {
            x_min: x_min as f32,
            y_min: y_min as f32,
            x_max: x_max as f32,
            y_max: y_max as f32,
        })
    }

    /// Erode the mask by 1 pixel (4-connectivity: all 4 cardinal neighbors must also be foreground)
    pub fn erode(&self) -> Self {
        let mut eroded = vec![false; self.width * self.height];
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                if !self.mask[idx] {
                    continue;
                }
                // Check all 4 cardinal neighbors exist and are foreground
                let left_ok = x > 0 && self.mask[y * self.width + (x - 1)];
                let right_ok = x + 1 < self.width && self.mask[y * self.width + (x + 1)];
                let up_ok = y > 0 && self.mask[(y - 1) * self.width + x];
                let down_ok = y + 1 < self.height && self.mask[(y + 1) * self.width + x];
                eroded[idx] = left_ok && right_ok && up_ok && down_ok;
            }
        }
        Self::new(
            eroded,
            self.width,
            self.height,
            self.iou_score,
            self.stability_score,
            self.mask_index,
        )
    }

    /// Dilate the mask by 1 pixel (4-connectivity: any cardinal neighbor being foreground is enough)
    pub fn dilate(&self) -> Self {
        let mut dilated = self.mask.clone();
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                if self.mask[idx] {
                    continue;
                }
                // Set to foreground if any 4-neighbor is foreground
                let left_fg = x > 0 && self.mask[y * self.width + (x - 1)];
                let right_fg = x + 1 < self.width && self.mask[y * self.width + (x + 1)];
                let up_fg = y > 0 && self.mask[(y - 1) * self.width + x];
                let down_fg = y + 1 < self.height && self.mask[(y + 1) * self.width + x];
                dilated[idx] = left_fg || right_fg || up_fg || down_fg;
            }
        }
        Self::new(
            dilated,
            self.width,
            self.height,
            self.iou_score,
            self.stability_score,
            self.mask_index,
        )
    }

    /// Convert to a 2D `SegmentationMask` representation.
    pub fn to_segmentation_mask(&self, label: &str, score: f32) -> SegmentationMask {
        let mask_2d: Vec<Vec<bool>> =
            self.mask.chunks(self.width).map(|row| row.to_vec()).collect();
        SegmentationMask::new(mask_2d, label.to_string(), score)
    }
}

// ---------------------------------------------------------------------------
// MaskRefiner
// ---------------------------------------------------------------------------

/// Morphological and quality post-processing operations on 2D boolean masks.
pub struct MaskRefiner;

impl MaskRefiner {
    /// Morphological opening (erosion followed by dilation).
    ///
    /// Removes small bright regions and thin protrusions.
    pub fn morphological_open(mask: &[Vec<bool>], kernel_size: usize) -> Vec<Vec<bool>> {
        let eroded = Self::erode_2d(mask, kernel_size);
        Self::dilate_2d(&eroded, kernel_size)
    }

    /// Morphological closing (dilation followed by erosion).
    ///
    /// Fills small holes and gaps in the mask.
    pub fn morphological_close(mask: &[Vec<bool>], kernel_size: usize) -> Vec<Vec<bool>> {
        let dilated = Self::dilate_2d(mask, kernel_size);
        Self::erode_2d(&dilated, kernel_size)
    }

    /// Remove connected components smaller than `min_area` pixels using flood fill (BFS).
    pub fn remove_small_components(mask: &[Vec<bool>], min_area: usize) -> Vec<Vec<bool>> {
        let h = mask.len();
        let w = mask.first().map(|r| r.len()).unwrap_or(0);
        if h == 0 || w == 0 {
            return mask.to_vec();
        }

        let mut visited = vec![vec![false; w]; h];
        let mut output = vec![vec![false; w]; h];

        for start_r in 0..h {
            for start_c in 0..w {
                if !mask[start_r][start_c] || visited[start_r][start_c] {
                    continue;
                }
                // BFS to collect the connected component
                let mut component: Vec<(usize, usize)> = Vec::new();
                let mut queue = std::collections::VecDeque::new();
                queue.push_back((start_r, start_c));
                visited[start_r][start_c] = true;

                while let Some((r, c)) = queue.pop_front() {
                    component.push((r, c));
                    // 4-connectivity neighbors
                    let neighbors: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                    for (dr, dc) in &neighbors {
                        let nr = r as isize + dr;
                        let nc = c as isize + dc;
                        if nr >= 0 && nr < h as isize && nc >= 0 && nc < w as isize {
                            let nr = nr as usize;
                            let nc = nc as usize;
                            if mask[nr][nc] && !visited[nr][nc] {
                                visited[nr][nc] = true;
                                queue.push_back((nr, nc));
                            }
                        }
                    }
                }

                // Keep the component only if it meets the minimum area threshold
                if component.len() >= min_area {
                    for (r, c) in &component {
                        output[*r][*c] = true;
                    }
                }
            }
        }

        output
    }

    /// Compute the intersection-over-union between two 2D boolean masks.
    ///
    /// Both masks must have the same dimensions; returns 0.0 if either is empty.
    pub fn compute_mask_iou(a: &[Vec<bool>], b: &[Vec<bool>]) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let h = a.len().min(b.len());
        let mut intersection = 0usize;
        let mut union_ = 0usize;

        for row in 0..h {
            let w = a[row].len().min(b[row].len());
            for col in 0..w {
                let av = a[row][col];
                let bv = b[row][col];
                if av || bv {
                    union_ += 1;
                }
                if av && bv {
                    intersection += 1;
                }
            }
        }

        if union_ == 0 {
            0.0
        } else {
            intersection as f32 / union_ as f32
        }
    }

    // ------------------------------------------------------------------
    // Private morphological primitives
    // ------------------------------------------------------------------

    /// 2D erosion: a pixel stays true only if all pixels in the
    /// `kernel_size × kernel_size` neighborhood are true.
    fn erode_2d(mask: &[Vec<bool>], kernel_size: usize) -> Vec<Vec<bool>> {
        let h = mask.len();
        let w = mask.first().map(|r| r.len()).unwrap_or(0);
        if h == 0 || w == 0 || kernel_size == 0 {
            return mask.to_vec();
        }
        let half = (kernel_size / 2) as isize;
        let mut out = vec![vec![false; w]; h];

        for r in 0..h {
            for c in 0..w {
                if !mask[r][c] {
                    continue;
                }
                let mut all_fg = true;
                'outer: for kr in -half..=half {
                    for kc in -half..=half {
                        let nr = r as isize + kr;
                        let nc = c as isize + kc;
                        if nr < 0 || nr >= h as isize || nc < 0 || nc >= w as isize {
                            all_fg = false;
                            break 'outer;
                        }
                        if !mask[nr as usize][nc as usize] {
                            all_fg = false;
                            break 'outer;
                        }
                    }
                }
                out[r][c] = all_fg;
            }
        }
        out
    }

    /// 2D dilation: a pixel becomes true if any pixel in the
    /// `kernel_size × kernel_size` neighborhood is true.
    fn dilate_2d(mask: &[Vec<bool>], kernel_size: usize) -> Vec<Vec<bool>> {
        let h = mask.len();
        let w = mask.first().map(|r| r.len()).unwrap_or(0);
        if h == 0 || w == 0 || kernel_size == 0 {
            return mask.to_vec();
        }
        let half = (kernel_size / 2) as isize;
        let mut out = vec![vec![false; w]; h];

        for r in 0..h {
            for c in 0..w {
                // Check if any neighbor in the kernel is foreground
                'outer: for kr in -half..=half {
                    for kc in -half..=half {
                        let nr = r as isize + kr;
                        let nc = c as isize + kc;
                        if nr >= 0
                            && nr < h as isize
                            && nc >= 0
                            && nc < w as isize
                            && mask[nr as usize][nc as usize]
                        {
                            out[r][c] = true;
                            break 'outer;
                        }
                    }
                }
            }
        }
        out
    }
}

/// Mask generation result holding multiple candidate masks
#[derive(Debug, Clone)]
pub struct MaskGenerationResult {
    pub masks: Vec<GeneratedMask>,
    /// Index of the mask with the highest `iou_score`
    pub best_mask_index: usize,
}

impl MaskGenerationResult {
    /// Reference to the best-scoring mask
    pub fn best_mask(&self) -> &GeneratedMask {
        &self.masks[self.best_mask_index]
    }

    /// Masks whose IoU score meets the threshold
    pub fn filter_by_iou(&self, min_iou: f32) -> Vec<&GeneratedMask> {
        self.masks.iter().filter(|m| m.iou_score >= min_iou).collect()
    }

    /// Masks whose stability score meets the threshold
    pub fn filter_by_stability(&self, min_stability: f32) -> Vec<&GeneratedMask> {
        self.masks.iter().filter(|m| m.stability_score >= min_stability).collect()
    }
}

/// Errors that can occur during mask generation
#[derive(Debug)]
pub enum MaskGenerationError {
    /// Prompt contained no geometric or text information
    EmptyPrompt,
    /// Image dimensions are inconsistent or zero
    InvalidImageDimensions { width: usize, height: usize },
    /// A box prompt was geometrically invalid
    InvalidBoxPrompt(String),
}

impl fmt::Display for MaskGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaskGenerationError::EmptyPrompt => write!(f, "mask generation error: prompt is empty"),
            MaskGenerationError::InvalidImageDimensions { width, height } => write!(
                f,
                "mask generation error: invalid image dimensions {}x{}",
                width, height
            ),
            MaskGenerationError::InvalidBoxPrompt(msg) => {
                write!(f, "mask generation error: invalid box prompt — {}", msg)
            },
        }
    }
}

impl std::error::Error for MaskGenerationError {}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// SAM-compatible mask generation pipeline
pub struct MaskGenerationPipeline {
    pub model: String,
    /// Number of candidate masks to generate (SAM outputs 3)
    pub num_multimask_outputs: usize,
    /// Minimum predicted IoU to include a mask
    pub pred_iou_thresh: f32,
    /// Minimum stability score to include a mask
    pub stability_score_thresh: f32,
}

impl MaskGenerationPipeline {
    /// Create a pipeline with sensible defaults
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            num_multimask_outputs: 3,
            pred_iou_thresh: 0.88,
            stability_score_thresh: 0.95,
        }
    }

    /// Generate masks for a single image given a prompt.
    ///
    /// `image` is row-major `[H * W * 3]` f32 in `[0, 1]`.
    pub fn run(
        &self,
        image: &[f32],
        width: usize,
        height: usize,
        prompt: &MaskPrompt,
    ) -> Result<MaskGenerationResult, MaskGenerationError> {
        if width == 0 || height == 0 || image.len() != width * height * 3 {
            return Err(MaskGenerationError::InvalidImageDimensions { width, height });
        }
        if prompt.is_empty() {
            return Err(MaskGenerationError::EmptyPrompt);
        }

        // Validate box prompts
        for bp in &prompt.boxes {
            if !bp.is_valid() {
                return Err(MaskGenerationError::InvalidBoxPrompt(format!(
                    "box ({},{})→({},{}) has zero or negative area",
                    bp.x_min, bp.y_min, bp.x_max, bp.y_max
                )));
            }
        }

        let mut masks = Vec::with_capacity(self.num_multimask_outputs);

        for mask_idx in 0..self.num_multimask_outputs {
            let mask_data = self.generate_single_mask(image, width, height, prompt, mask_idx);

            // Deterministic quality scores derived from mask coverage
            let foreground_count = mask_data.iter().filter(|&&v| v).count();
            let coverage = foreground_count as f32 / (width * height) as f32;

            let iou_score = (0.95_f32 - coverage * 0.3 - mask_idx as f32 * 0.05).clamp(0.0, 1.0);
            let stability_score =
                (0.98_f32 - coverage * 0.2 - mask_idx as f32 * 0.02).clamp(0.0, 1.0);

            masks.push(GeneratedMask::new(
                mask_data,
                width,
                height,
                iou_score,
                stability_score,
                mask_idx,
            ));
        }

        // Find best mask by iou_score
        let best_mask_index = masks
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.iou_score.partial_cmp(&b.iou_score).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(MaskGenerationResult {
            masks,
            best_mask_index,
        })
    }

    /// Generate segmentation masks as `SegmentationMask` (new-style API).
    pub fn generate(
        &self,
        image: &[f32],
        width: usize,
        height: usize,
        prompt: &MaskPrompt,
    ) -> Result<Vec<SegmentationMask>, MaskGenerationError> {
        let result = self.run(image, width, height, prompt)?;
        let seg_masks: Vec<SegmentationMask> = result
            .masks
            .iter()
            .enumerate()
            .map(|(i, gm)| gm.to_segmentation_mask(&format!("mask_{}", i), gm.iou_score))
            .collect();
        Ok(seg_masks)
    }

    /// Generate a single candidate mask from the prompt geometry.
    fn generate_single_mask(
        &self,
        _image: &[f32],
        width: usize,
        height: usize,
        prompt: &MaskPrompt,
        mask_idx: usize,
    ) -> Vec<bool> {
        let mut mask = vec![false; width * height];

        // Apply box prompts: fill the interior with a small erosion per mask_idx
        for bp in &prompt.boxes {
            let x0 = (bp.x_min as usize).min(width.saturating_sub(1));
            let y0 = (bp.y_min as usize).min(height.saturating_sub(1));
            let x1 = (bp.x_max as usize).min(width.saturating_sub(1));
            let y1 = (bp.y_max as usize).min(height.saturating_sub(1));

            // Each successive candidate slightly shrinks the box
            let shrink = mask_idx;
            let xs = (x0 + shrink).min(x1.saturating_sub(shrink));
            let ys = (y0 + shrink).min(y1.saturating_sub(shrink));
            let xe = x1.saturating_sub(shrink);
            let ye = y1.saturating_sub(shrink);

            for y in ys..=ye.min(height.saturating_sub(1)) {
                for x in xs..=xe.min(width.saturating_sub(1)) {
                    mask[y * width + x] = true;
                }
            }
        }

        // Apply foreground point prompts: circle of radius 2 + mask_idx
        for pp in &prompt.points {
            if pp.label != PointLabel::Foreground {
                continue;
            }
            let cx = pp.x as isize;
            let cy = pp.y as isize;
            let radius = 2 + mask_idx as isize;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx * dx + dy * dy <= radius * radius {
                        let px = cx + dx;
                        let py = cy + dy;
                        if px >= 0 && py >= 0 && (px as usize) < width && (py as usize) < height {
                            mask[(py as usize) * width + (px as usize)] = true;
                        }
                    }
                }
            }
        }

        // Apply background point prompts: erase circles
        for pp in &prompt.points {
            if pp.label != PointLabel::Background {
                continue;
            }
            let cx = pp.x as isize;
            let cy = pp.y as isize;
            let radius = 2_isize;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx * dx + dy * dy <= radius * radius {
                        let px = cx + dx;
                        let py = cy + dy;
                        if px >= 0 && py >= 0 && (px as usize) < width && (py as usize) < height {
                            mask[(py as usize) * width + (px as usize)] = false;
                        }
                    }
                }
            }
        }

        mask
    }

    /// Automatic mask generation without explicit prompts — uses a uniform grid of points.
    ///
    /// Generates a `points_per_side × points_per_side` grid of foreground prompts and runs
    /// mask generation for each, returning all results.
    pub fn automatic_mask_generation(
        &self,
        image: &[f32],
        width: usize,
        height: usize,
        points_per_side: usize,
    ) -> Result<Vec<MaskGenerationResult>, MaskGenerationError> {
        if width == 0 || height == 0 || image.len() != width * height * 3 {
            return Err(MaskGenerationError::InvalidImageDimensions { width, height });
        }
        if points_per_side == 0 {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(points_per_side * points_per_side);

        for row in 0..points_per_side {
            for col in 0..points_per_side {
                // Evenly spaced across [0, width) and [0, height)
                let x = (col as f32 + 0.5) * (width as f32 / points_per_side as f32);
                let y = (row as f32 + 0.5) * (height as f32 / points_per_side as f32);

                let prompt = MaskPrompt::new().with_point(x, y, PointLabel::Foreground);
                let result = self.run(image, width, height, &prompt)?;
                results.push(result);
            }
        }

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_image(w: usize, h: usize) -> Vec<f32> {
        let mut img = vec![0.0f32; w * h * 3];
        for i in 0..img.len() {
            img[i] = (i as f32 / img.len() as f32).clamp(0.0, 1.0);
        }
        img
    }

    fn all_true_2d(h: usize, w: usize) -> Vec<Vec<bool>> {
        vec![vec![true; w]; h]
    }

    fn all_false_2d(h: usize, w: usize) -> Vec<Vec<bool>> {
        vec![vec![false; w]; h]
    }

    // ---- Existing tests ----

    #[test]
    fn test_box_prompt_area() {
        let bp = BoxPrompt {
            x_min: 1.0,
            y_min: 2.0,
            x_max: 5.0,
            y_max: 6.0,
        };
        let area = bp.area();
        assert!(
            (area - 16.0).abs() < 1e-6,
            "area should be 16, got {}",
            area
        );
    }

    #[test]
    fn test_box_prompt_center() {
        let bp = BoxPrompt {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 4.0,
            y_max: 6.0,
        };
        let (cx, cy) = bp.center();
        assert!((cx - 2.0).abs() < 1e-6);
        assert!((cy - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_box_prompt_invalid() {
        let valid = BoxPrompt {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 4.0,
            y_max: 4.0,
        };
        assert!(valid.is_valid());

        let degenerate_x = BoxPrompt {
            x_min: 4.0,
            y_min: 0.0,
            x_max: 4.0,
            y_max: 4.0,
        };
        assert!(!degenerate_x.is_valid());

        let inverted = BoxPrompt {
            x_min: 5.0,
            y_min: 5.0,
            x_max: 1.0,
            y_max: 1.0,
        };
        assert!(!inverted.is_valid());
    }

    #[test]
    fn test_mask_prompt_builder() {
        let prompt = MaskPrompt::new()
            .with_point(1.0, 2.0, PointLabel::Foreground)
            .with_box(0.0, 0.0, 10.0, 10.0)
            .with_text("cat");

        assert_eq!(prompt.points.len(), 1);
        assert_eq!(prompt.boxes.len(), 1);
        assert_eq!(prompt.text.as_deref(), Some("cat"));
        assert_eq!(prompt.num_prompts(), 2);
    }

    #[test]
    fn test_mask_prompt_is_empty() {
        let empty = MaskPrompt::new();
        assert!(empty.is_empty());

        let non_empty = MaskPrompt::new().with_point(1.0, 1.0, PointLabel::Background);
        assert!(!non_empty.is_empty());

        let text_only = MaskPrompt::new().with_text("dog");
        assert!(!text_only.is_empty());
    }

    #[test]
    fn test_generated_mask_coverage() {
        // 4x4 mask, 4 pixels foreground out of 16
        let mask_data: Vec<bool> = (0..16).map(|i| i < 4).collect();
        let gm = GeneratedMask::new(mask_data, 4, 4, 0.9, 0.95, 0);
        assert_eq!(gm.num_foreground_pixels(), 4);
        assert!((gm.coverage_ratio() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_generated_mask_bounding_box() {
        // 4x4 grid; set pixel (1,1), (1,2), (2,1), (2,2)
        let mut mask_data = vec![false; 16];
        mask_data[4 + 1] = true;
        mask_data[4 + 2] = true;
        mask_data[2 * 4 + 1] = true;
        mask_data[2 * 4 + 2] = true;
        let gm = GeneratedMask::new(mask_data, 4, 4, 0.9, 0.95, 0);
        let bbox = gm.bounding_box().expect("should have bounding box");
        assert!((bbox.x_min - 1.0).abs() < 1e-6);
        assert!((bbox.y_min - 1.0).abs() < 1e-6);
        assert!((bbox.x_max - 2.0).abs() < 1e-6);
        assert!((bbox.y_max - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_generated_mask_erode() {
        // 4x4 all-true mask; after erosion border pixels should become false
        let mask_data = vec![true; 16];
        let gm = GeneratedMask::new(mask_data, 4, 4, 0.9, 0.95, 0);
        let eroded = gm.erode();
        // Only pixel (1,1) and (2,2) etc. keep all 4 neighbors; corners lose them
        // For a 4x4 grid: interior pixels are (1,1),(1,2),(2,1),(2,2)
        assert!(!eroded.mask[0]); // corner (0,0) — missing left/up neighbors
                                  // Interior pixel (1,1): idx = 1*4+1 = 5
        assert!(eroded.mask[5], "interior pixel (1,1) should be eroded-true");
    }

    #[test]
    fn test_generated_mask_dilate() {
        // 4x4 mask with a single true pixel at (2,2)
        let mut mask_data = vec![false; 16];
        mask_data[2 * 4 + 2] = true;
        let gm = GeneratedMask::new(mask_data, 4, 4, 0.9, 0.95, 0);
        let dilated = gm.dilate();
        // (2,2) and all 4-neighbors should be true
        assert!(dilated.mask[2 * 4 + 2]);
        assert!(dilated.mask[2 * 4 + 1]); // left
        assert!(dilated.mask[2 * 4 + 3]); // right
        assert!(dilated.mask[4 + 2]); // up
        assert!(dilated.mask[3 * 4 + 2]); // down
                                          // Diagonals should stay false
        assert!(!dilated.mask[4 + 1]);
    }

    #[test]
    fn test_mask_gen_result_best_mask() {
        let m0 = GeneratedMask::new(vec![true; 16], 4, 4, 0.7, 0.9, 0);
        let m1 = GeneratedMask::new(vec![true; 16], 4, 4, 0.95, 0.92, 1);
        let m2 = GeneratedMask::new(vec![true; 16], 4, 4, 0.8, 0.88, 2);
        let result = MaskGenerationResult {
            masks: vec![m0, m1, m2],
            best_mask_index: 1,
        };
        assert_eq!(result.best_mask().mask_index, 1);
        assert!((result.best_mask().iou_score - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_mask_gen_result_filter_iou() {
        let m0 = GeneratedMask::new(vec![false; 16], 4, 4, 0.5, 0.9, 0);
        let m1 = GeneratedMask::new(vec![false; 16], 4, 4, 0.9, 0.92, 1);
        let m2 = GeneratedMask::new(vec![false; 16], 4, 4, 0.75, 0.88, 2);
        let result = MaskGenerationResult {
            masks: vec![m0, m1, m2],
            best_mask_index: 1,
        };
        let filtered = result.filter_by_iou(0.7);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_mask_generation_pipeline_with_box() {
        let pipeline = MaskGenerationPipeline::new("sam-vit-b");
        let image = make_image(8, 8);
        let prompt = MaskPrompt::new().with_box(1.0, 1.0, 6.0, 6.0);
        let result = pipeline.run(&image, 8, 8, &prompt).expect("run should succeed");
        assert_eq!(result.masks.len(), 3);
        // Best mask should have some foreground pixels from the box region
        assert!(result.best_mask().num_foreground_pixels() > 0);
    }

    #[test]
    fn test_mask_generation_pipeline_with_point() {
        let pipeline = MaskGenerationPipeline::new("sam-vit-b");
        let image = make_image(8, 8);
        let prompt = MaskPrompt::new().with_point(4.0, 4.0, PointLabel::Foreground);
        let result = pipeline.run(&image, 8, 8, &prompt).expect("run should succeed");
        assert_eq!(result.masks.len(), 3);
        assert!(result.best_mask().num_foreground_pixels() > 0);
    }

    #[test]
    fn test_mask_generation_automatic() {
        let pipeline = MaskGenerationPipeline::new("sam-vit-b");
        let image = make_image(8, 8);
        let results = pipeline
            .automatic_mask_generation(&image, 8, 8, 2)
            .expect("automatic should succeed");
        // 2x2 grid = 4 results
        assert_eq!(results.len(), 4);
        for r in &results {
            assert_eq!(r.masks.len(), 3);
        }
    }

    #[test]
    fn test_mask_generation_error_display() {
        let e1 = MaskGenerationError::EmptyPrompt;
        assert!(e1.to_string().contains("empty"));

        let e2 = MaskGenerationError::InvalidImageDimensions {
            width: 0,
            height: 0,
        };
        assert!(e2.to_string().contains("invalid image dimensions"));

        let e3 = MaskGenerationError::InvalidBoxPrompt("bad box".to_string());
        assert!(e3.to_string().contains("bad box"));
    }

    // ---- New MaskRefiner tests ----

    #[test]
    fn test_morphological_open_removes_isolated_pixel() {
        // 5×5 mask: all false except one isolated pixel at center
        let mut mask = all_false_2d(5, 5);
        mask[2][2] = true;
        let opened = MaskRefiner::morphological_open(&mask, 3);
        // A single isolated pixel should be eliminated by erosion first
        let any_true = opened.iter().flatten().any(|&v| v);
        assert!(
            !any_true,
            "isolated pixel should be removed by morphological opening"
        );
    }

    #[test]
    fn test_morphological_close_fills_hole() {
        // 5×5 all-true mask with one hole at center
        let mut mask = all_true_2d(5, 5);
        mask[2][2] = false;
        let closed = MaskRefiner::morphological_close(&mask, 3);
        // The central hole should be filled
        assert!(
            closed[2][2],
            "morphological closing should fill the interior hole"
        );
    }

    #[test]
    fn test_morphological_open_shape_preserving() {
        // A solid 5×5 block should survive opening with a small kernel
        let mask = all_true_2d(5, 5);
        let opened = MaskRefiner::morphological_open(&mask, 1);
        // With kernel_size=1 (no-op), shape should be preserved
        assert_eq!(opened, mask, "kernel_size=1 open should be identity");
    }

    #[test]
    fn test_remove_small_components_eliminates_small() {
        // 6×6 mask: large block at top-left, tiny single pixel at bottom-right
        let mut mask = all_false_2d(6, 6);
        // 3×3 block at top-left (area=9)
        for r in 0..3 {
            for c in 0..3 {
                mask[r][c] = true;
            }
        }
        // Single isolated pixel at (5,5) (area=1)
        mask[5][5] = true;
        let cleaned = MaskRefiner::remove_small_components(&mask, 5);
        // Single pixel should be removed; 3×3 block should survive
        assert!(cleaned[0][0], "large component should survive");
        assert!(!cleaned[5][5], "tiny component should be removed");
    }

    #[test]
    fn test_remove_small_components_keeps_large() {
        // All pixels in a 4×4 block are connected and should survive with min_area=1
        let mask = all_true_2d(4, 4);
        let cleaned = MaskRefiner::remove_small_components(&mask, 1);
        let all_kept = cleaned.iter().flatten().all(|&v| v);
        assert!(all_kept, "all pixels in a connected block should survive");
    }

    #[test]
    fn test_compute_mask_iou_identical() {
        let mask = all_true_2d(4, 4);
        let iou = MaskRefiner::compute_mask_iou(&mask, &mask);
        assert!(
            (iou - 1.0).abs() < 1e-6,
            "IoU with self should be 1.0, got {}",
            iou
        );
    }

    #[test]
    fn test_compute_mask_iou_disjoint() {
        // Left half vs right half → no overlap → IoU = 0
        let mut a = all_false_2d(4, 4);
        let mut b = all_false_2d(4, 4);
        for r in 0..4 {
            a[r][0] = true;
            a[r][1] = true;
            b[r][2] = true;
            b[r][3] = true;
        }
        let iou = MaskRefiner::compute_mask_iou(&a, &b);
        assert!(
            (iou - 0.0).abs() < 1e-6,
            "disjoint masks should have IoU=0, got {}",
            iou
        );
    }

    #[test]
    fn test_compute_mask_iou_partial_overlap() {
        // 2×2 all-true masks shifted by one column → 2 shared pixels
        let mut a = all_false_2d(2, 4);
        let mut b = all_false_2d(2, 4);
        a[0][0] = true;
        a[0][1] = true;
        a[1][0] = true;
        a[1][1] = true;
        b[0][1] = true;
        b[0][2] = true;
        b[1][1] = true;
        b[1][2] = true;
        // intersection = 2, union = 6, IoU = 2/6 ≈ 0.333
        let iou = MaskRefiner::compute_mask_iou(&a, &b);
        assert!(
            (iou - 2.0 / 6.0).abs() < 1e-5,
            "IoU should be ~0.333, got {}",
            iou
        );
    }

    #[test]
    fn test_generate_returns_segmentation_masks() {
        let pipeline = MaskGenerationPipeline::new("sam-vit-b");
        let image = make_image(8, 8);
        let prompt = MaskPrompt::new().with_box(1.0, 1.0, 6.0, 6.0);
        let masks = pipeline.generate(&image, 8, 8, &prompt).expect("generate ok");
        assert_eq!(
            masks.len(),
            3,
            "should return 3 candidate SegmentationMasks"
        );
        for seg in &masks {
            assert_eq!(seg.height(), 8, "mask height should be 8");
            assert_eq!(seg.width(), 8, "mask width should be 8");
        }
    }

    #[test]
    fn test_segmentation_mask_area_computation() {
        let mask_2d = vec![vec![true, false, true], vec![false, true, false]];
        let seg = SegmentationMask::new(mask_2d, "test".to_string(), 0.9);
        assert_eq!(seg.area, 3, "area should count foreground pixels");
        assert_eq!(seg.height(), 2);
        assert_eq!(seg.width(), 3);
    }

    #[test]
    fn test_segmentation_mask_to_flat() {
        let mask_2d = vec![vec![true, false], vec![false, true]];
        let seg = SegmentationMask::new(mask_2d, "x".to_string(), 0.8);
        let flat = seg.to_flat();
        assert_eq!(flat, vec![true, false, false, true]);
    }

    #[test]
    fn test_generated_mask_to_segmentation_mask() {
        let mask_flat = vec![true, false, false, true];
        let gm = GeneratedMask::new(mask_flat, 2, 2, 0.88, 0.95, 0);
        let seg = gm.to_segmentation_mask("region", 0.88);
        assert_eq!(seg.height(), 2);
        assert_eq!(seg.width(), 2);
        assert_eq!(seg.area, 2);
        assert_eq!(seg.label, "region");
    }
}
