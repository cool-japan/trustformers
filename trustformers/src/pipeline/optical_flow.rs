//! Optical flow pipeline — dense motion estimation between frames (RAFT/FlowFormer-compatible)

use std::fmt;

/// A 2D displacement vector representing per-pixel motion
#[derive(Debug, Clone, Copy)]
pub struct FlowVector {
    /// Horizontal displacement in pixels
    pub dx: f32,
    /// Vertical displacement in pixels
    pub dy: f32,
}

impl FlowVector {
    /// Construct a new flow vector
    pub fn new(dx: f32, dy: f32) -> Self {
        Self { dx, dy }
    }

    /// Euclidean magnitude of the displacement
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Angle of the displacement in degrees, measured from positive-x axis
    pub fn angle_degrees(&self) -> f32 {
        self.dy.atan2(self.dx).to_degrees()
    }

    /// Zero-motion vector
    pub fn zero() -> Self {
        Self { dx: 0.0, dy: 0.0 }
    }
}

/// Dense 2-D optical flow field with one `FlowVector` per pixel
#[derive(Debug, Clone)]
pub struct FlowField {
    /// Row-major `[H * W]` flow vectors
    pub flows: Vec<FlowVector>,
    pub width: usize,
    pub height: usize,
}

impl FlowField {
    /// Construct from a pre-built vector; length must equal `width * height`
    pub fn new(flows: Vec<FlowVector>, width: usize, height: usize) -> Self {
        Self {
            flows,
            width,
            height,
        }
    }

    /// All-zero flow field of the given dimensions
    pub fn zeros(width: usize, height: usize) -> Self {
        Self::new(vec![FlowVector::zero(); width * height], width, height)
    }

    /// Retrieve the flow vector at pixel `(x, y)`, or `None` if out of bounds
    pub fn get(&self, x: usize, y: usize) -> Option<FlowVector> {
        if x < self.width && y < self.height {
            Some(self.flows[y * self.width + x])
        } else {
            None
        }
    }

    /// Mean magnitude across all pixels
    pub fn mean_magnitude(&self) -> f32 {
        if self.flows.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.flows.iter().map(|f| f.magnitude()).sum();
        sum / self.flows.len() as f32
    }

    /// Maximum magnitude across all pixels
    pub fn max_magnitude(&self) -> f32 {
        self.flows.iter().map(|f| f.magnitude()).fold(0.0_f32, f32::max)
    }

    /// Average End-Point Error between `self` and `reference` (must have identical dimensions)
    pub fn epe(&self, reference: &FlowField) -> Result<f32, FlowError> {
        if self.width != reference.width || self.height != reference.height {
            return Err(FlowError::DimensionMismatch {
                self_shape: (self.width, self.height),
                other_shape: (reference.width, reference.height),
            });
        }
        let sum: f32 = self
            .flows
            .iter()
            .zip(reference.flows.iter())
            .map(|(a, b)| {
                let dx = a.dx - b.dx;
                let dy = a.dy - b.dy;
                (dx * dx + dy * dy).sqrt()
            })
            .sum();
        Ok(sum / self.flows.len() as f32)
    }

    /// Backward-warp `frame` (row-major `[H * W * 3]` f32) by this flow field using bilinear
    /// interpolation.  For each output pixel `(x, y)` the source coordinates are
    /// `(x + dx, y + dy)` clamped to the frame boundaries.
    pub fn warp_frame(&self, frame: &[f32]) -> Result<Vec<f32>, FlowError> {
        let expected = self.width * self.height * 3;
        if frame.len() != expected {
            return Err(FlowError::InvalidDimensions);
        }
        if self.flows.is_empty() {
            return Err(FlowError::EmptyFrame);
        }

        let w = self.width as f32;
        let h = self.height as f32;
        let mut output = vec![0.0_f32; expected];

        for y in 0..self.height {
            for x in 0..self.width {
                let fv = self.flows[y * self.width + x];
                let src_x = (x as f32 + fv.dx).clamp(0.0, w - 1.0);
                let src_y = (y as f32 + fv.dy).clamp(0.0, h - 1.0);

                // Bilinear interpolation
                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = (x0 + 1).min(self.width - 1);
                let y1 = (y0 + 1).min(self.height - 1);

                let wx = src_x - x0 as f32;
                let wy = src_y - y0 as f32;

                let out_base = (y * self.width + x) * 3;
                for c in 0..3 {
                    let v00 = frame[(y0 * self.width + x0) * 3 + c];
                    let v10 = frame[(y0 * self.width + x1) * 3 + c];
                    let v01 = frame[(y1 * self.width + x0) * 3 + c];
                    let v11 = frame[(y1 * self.width + x1) * 3 + c];
                    output[out_base + c] = v00 * (1.0 - wx) * (1.0 - wy)
                        + v10 * wx * (1.0 - wy)
                        + v01 * (1.0 - wx) * wy
                        + v11 * wx * wy;
                }
            }
        }
        Ok(output)
    }

    /// Visualise flow as an HSV-encoded color image.
    ///
    /// Hue encodes direction (0–360° → 0–255), saturation encodes normalised magnitude.
    /// Returns `[H * W * 3]` as u8 RGB.
    pub fn to_color_visualization(&self) -> Vec<u8> {
        if self.flows.is_empty() {
            return Vec::new();
        }

        let max_mag = self.max_magnitude().max(1e-8);
        let mut rgb = vec![0u8; self.width * self.height * 3];

        for (i, fv) in self.flows.iter().enumerate() {
            // Hue from angle [0, 360) mapped to [0, 1)
            let angle = fv.angle_degrees();
            let hue = ((angle + 360.0) % 360.0) / 360.0;
            // Saturation from normalised magnitude
            let sat = (fv.magnitude() / max_mag).clamp(0.0, 1.0);
            // Full value
            let val = 1.0_f32;

            let (r, g, b) = hsv_to_rgb(hue, sat, val);
            rgb[i * 3] = (r * 255.0) as u8;
            rgb[i * 3 + 1] = (g * 255.0) as u8;
            rgb[i * 3 + 2] = (b * 255.0) as u8;
        }
        rgb
    }

    /// Downsample the flow field by a factor of 2 (average 2×2 blocks; divide vectors by 2).
    pub fn downsample(&self) -> Self {
        if self.width < 2 || self.height < 2 {
            return Self::zeros(1, 1);
        }
        let new_w = self.width / 2;
        let new_h = self.height / 2;
        let mut flows = Vec::with_capacity(new_w * new_h);

        for y in 0..new_h {
            for x in 0..new_w {
                let sx = x * 2;
                let sy = y * 2;
                let f00 = self.flows[sy * self.width + sx];
                let f10 = self.flows[sy * self.width + (sx + 1).min(self.width - 1)];
                let f01 = self.flows[(sy + 1).min(self.height - 1) * self.width + sx];
                let f11 = self.flows
                    [(sy + 1).min(self.height - 1) * self.width + (sx + 1).min(self.width - 1)];
                // Average the block, then halve magnitude (scale change at coarser resolution)
                flows.push(FlowVector::new(
                    (f00.dx + f10.dx + f01.dx + f11.dx) * 0.25,
                    (f00.dy + f10.dy + f01.dy + f11.dy) * 0.25,
                ));
            }
        }
        Self::new(flows, new_w, new_h)
    }
}

// ---------------------------------------------------------------------------
// HSV → RGB helper (pure arithmetic, no external dependencies)
// ---------------------------------------------------------------------------

/// Convert HSV (all in [0, 1]) to RGB (all in [0, 1])
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    if s <= 0.0 {
        return (v, v, v);
    }
    let hh = (h * 6.0).rem_euclid(6.0);
    let i = hh.floor() as u32;
    let ff = hh - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * ff);
    let t = v * (1.0 - s * (1.0 - ff));
    match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

// ---------------------------------------------------------------------------
// Flow pyramid
// ---------------------------------------------------------------------------

/// Multi-scale flow pyramid for coarse-to-fine estimation
pub struct FlowPyramid {
    pub levels: Vec<FlowField>,
}

impl FlowPyramid {
    /// Build a pyramid by iteratively downsampling the base flow field.
    /// `levels[0]` is the finest (full-resolution) level.
    pub fn from_base(base: FlowField, num_levels: usize) -> Self {
        if num_levels == 0 {
            return Self { levels: Vec::new() };
        }
        let mut levels = Vec::with_capacity(num_levels);
        levels.push(base);
        for i in 1..num_levels {
            // Index-based access avoids last().expect()
            let downsampled = levels[i - 1].downsample();
            levels.push(downsampled);
        }
        Self { levels }
    }

    /// Return the finest (full-resolution) flow field
    pub fn collapse(&self) -> &FlowField {
        &self.levels[0]
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Optical flow errors
#[derive(Debug)]
pub enum FlowError {
    /// Two flow fields have incompatible spatial dimensions
    DimensionMismatch {
        self_shape: (usize, usize),
        other_shape: (usize, usize),
    },
    /// Frame buffer is empty
    EmptyFrame,
    /// Frame dimensions are inconsistent
    InvalidDimensions,
}

impl fmt::Display for FlowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlowError::DimensionMismatch {
                self_shape,
                other_shape,
            } => write!(
                f,
                "optical flow error: dimension mismatch {}x{} vs {}x{}",
                self_shape.0, self_shape.1, other_shape.0, other_shape.1
            ),
            FlowError::EmptyFrame => write!(f, "optical flow error: frame buffer is empty"),
            FlowError::InvalidDimensions => {
                write!(f, "optical flow error: frame has invalid dimensions")
            },
        }
    }
}

impl std::error::Error for FlowError {}

/// RAFT/FlowFormer-compatible optical flow pipeline
pub struct OpticalFlowPipeline {
    pub model: String,
    /// Number of recurrent refinement iterations (RAFT default: 12)
    pub num_iterations: usize,
    /// Enable pyramid-based coarse-to-fine estimation
    pub use_pyramid: bool,
    /// Number of pyramid levels when `use_pyramid` is true
    pub pyramid_levels: usize,
}

impl OpticalFlowPipeline {
    /// Create a pipeline with sensible defaults
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            num_iterations: 12,
            use_pyramid: true,
            pyramid_levels: 4,
        }
    }

    /// Estimate dense optical flow from `frame1` to `frame2`.
    ///
    /// Both frames are row-major `[H * W * 3]` f32 in `[0, 1]`.
    /// Uses a Horn-Schunck-inspired approximation: flow proportional to the negative
    /// spatial gradient of the temporal difference image.
    pub fn run(
        &self,
        frame1: &[f32],
        frame2: &[f32],
        width: usize,
        height: usize,
    ) -> Result<FlowField, FlowError> {
        if frame1.is_empty() || frame2.is_empty() {
            return Err(FlowError::EmptyFrame);
        }
        let expected = width * height * 3;
        if frame1.len() != expected || frame2.len() != expected {
            return Err(FlowError::InvalidDimensions);
        }

        // Per-pixel luminance difference
        let diff: Vec<f32> = frame1
            .iter()
            .zip(frame2.iter())
            .enumerate()
            .map(|(i, (&a, &b))| {
                // Accumulate all three channels into the pixel's luminance slot
                // We'll average into a single luma value per pixel below
                let _ = i;
                a - b
            })
            .collect();

        // Reduce RGB diff to per-pixel luma diff
        let luma_diff: Vec<f32> = (0..width * height)
            .map(|px| {
                let base = px * 3;
                (diff[base] + diff[base + 1] + diff[base + 2]) / 3.0
            })
            .collect();

        let mut flows = Vec::with_capacity(width * height);

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let it = luma_diff[idx];

                // Forward-difference spatial gradients of luma_diff
                let ix = if x + 1 < width { luma_diff[y * width + (x + 1)] - it } else { 0.0 };
                let iy = if y + 1 < height { luma_diff[(y + 1) * width + x] - it } else { 0.0 };

                // Horn-Schunck: u = -It * Ix / (Ix^2 + Iy^2 + alpha^2)
                let alpha_sq = 0.01_f32;
                let denom = ix * ix + iy * iy + alpha_sq;
                let scale = -it / denom;
                let dx = scale * ix;
                let dy = scale * iy;

                flows.push(FlowVector::new(dx, dy));
            }
        }

        if self.use_pyramid {
            let base = FlowField::new(flows, width, height);
            let pyramid = FlowPyramid::from_base(base, self.pyramid_levels);
            // For the mock: just return the finest level
            return pyramid.levels.into_iter().next().ok_or(FlowError::EmptyFrame);
        }

        Ok(FlowField::new(flows, width, height))
    }

    /// Compute forward and backward optical flow between two frames.
    pub fn run_bidirectional(
        &self,
        frame1: &[f32],
        frame2: &[f32],
        width: usize,
        height: usize,
    ) -> Result<(FlowField, FlowField), FlowError> {
        let forward = self.run(frame1, frame2, width, height)?;
        let backward = self.run(frame2, frame1, width, height)?;
        Ok((forward, backward))
    }

    /// Compute per-pixel occlusion mask from forward/backward flow consistency.
    ///
    /// A pixel is considered occluded when:
    /// `|forward + warped(backward)| > 0.01 * (|forward|^2 + |backward|^2) + 0.5`
    pub fn occlusion_mask(
        &self,
        forward: &FlowField,
        backward: &FlowField,
    ) -> Result<Vec<bool>, FlowError> {
        if forward.width != backward.width || forward.height != backward.height {
            return Err(FlowError::DimensionMismatch {
                self_shape: (forward.width, forward.height),
                other_shape: (backward.width, backward.height),
            });
        }

        let w = forward.width;
        let h = forward.height;
        let mut mask = vec![false; w * h];

        for y in 0..h {
            for x in 0..w {
                let fv = forward.flows[y * w + x];

                // Sample backward flow at destination pixel (clamp to bounds)
                let dst_x = (x as f32 + fv.dx).clamp(0.0, (w - 1) as f32) as usize;
                let dst_y = (y as f32 + fv.dy).clamp(0.0, (h - 1) as f32) as usize;
                let bv = backward.flows[dst_y * w + dst_x];

                // Consistency residual
                let rx = fv.dx + bv.dx;
                let ry = fv.dy + bv.dy;
                let residual_sq = rx * rx + ry * ry;

                let fmag_sq = fv.dx * fv.dx + fv.dy * fv.dy;
                let bmag_sq = bv.dx * bv.dx + bv.dy * bv.dy;
                let threshold = 0.01 * (fmag_sq + bmag_sq) + 0.5;

                mask[y * w + x] = residual_sq > threshold;
            }
        }

        Ok(mask)
    }
}

// ---------------------------------------------------------------------------
// OpticalFlowField — 2-D grid alias with enhanced API
// ---------------------------------------------------------------------------

/// A 2-D optical flow field represented as a `Vec<Vec<FlowVector>>` (row-major outer index).
///
/// This is a higher-level wrapper around row-major storage for callers that prefer
/// 2-D indexing.  Internally the data mirrors `FlowField` but is stored as nested `Vec`.
#[derive(Debug, Clone)]
pub struct OpticalFlowField {
    /// Row-major 2-D flow vectors: `vectors[row][col]`
    pub vectors: Vec<Vec<FlowVector>>,
    pub width: usize,
    pub height: usize,
}

impl OpticalFlowField {
    /// Construct from a row-major `Vec<Vec<FlowVector>>`.
    pub fn new(vectors: Vec<Vec<FlowVector>>, width: usize, height: usize) -> Self {
        Self {
            vectors,
            width,
            height,
        }
    }

    /// All-zero field of size `width × height`.
    pub fn zeros(width: usize, height: usize) -> Self {
        let vectors = (0..height).map(|_| vec![FlowVector::zero(); width]).collect();
        Self {
            vectors,
            width,
            height,
        }
    }

    /// Return a 2-D grid of per-pixel magnitudes.
    pub fn magnitude_map(&self) -> Vec<Vec<f32>> {
        self.vectors
            .iter()
            .map(|row| row.iter().map(|fv| fv.magnitude()).collect())
            .collect()
    }

    /// Mean of all pixel magnitudes.
    pub fn mean_magnitude(&self) -> f32 {
        let total_pixels = self.width * self.height;
        if total_pixels == 0 {
            return 0.0;
        }
        let sum: f32 =
            self.vectors.iter().flat_map(|row| row.iter().map(|fv| fv.magnitude())).sum();
        sum / total_pixels as f32
    }

    /// Maximum of all pixel magnitudes.
    pub fn max_magnitude(&self) -> f32 {
        self.vectors
            .iter()
            .flat_map(|row| row.iter().map(|fv| fv.magnitude()))
            .fold(0.0_f32, f32::max)
    }

    /// Encode the flow as an HSV image (RGB bytes, `[H × W × 3]`).
    ///
    /// - Hue encodes direction (angle mapped to 0–360°→0–1→byte).
    /// - Saturation = 1 (fully saturated).
    /// - Value = normalised magnitude (magnitude / max_magnitude).
    pub fn to_hsv_image(&self) -> Vec<u8> {
        let max_mag = self.max_magnitude().max(f32::EPSILON);
        let mut output = Vec::with_capacity(self.width * self.height * 3);
        for row in &self.vectors {
            for fv in row {
                let angle = fv.angle_degrees();
                let hue = ((angle + 360.0) % 360.0) / 360.0;
                let saturation = 1.0_f32;
                let value = (fv.magnitude() / max_mag).clamp(0.0, 1.0);
                let (r, g, b) = hsv_to_rgb(hue, saturation, value);
                output.push((r * 255.0) as u8);
                output.push((g * 255.0) as u8);
                output.push((b * 255.0) as u8);
            }
        }
        output
    }
}

// ---------------------------------------------------------------------------
// OpticalFlowMetrics — quality metrics between flow fields
// ---------------------------------------------------------------------------

/// Metrics for comparing predicted optical flow fields against ground truth.
pub struct OpticalFlowMetrics;

impl OpticalFlowMetrics {
    /// Mean End-Point Error (EPE): mean Euclidean distance between predicted
    /// and ground-truth flow vectors.
    ///
    /// Both fields must have identical `width` and `height`.  Returns `None`
    /// when dimensions differ.
    pub fn endpoint_error(pred: &OpticalFlowField, gt: &OpticalFlowField) -> Option<f32> {
        if pred.width != gt.width || pred.height != gt.height {
            return None;
        }
        let n = (pred.width * pred.height) as f32;
        if n < f32::EPSILON {
            return Some(0.0);
        }
        let sum: f32 = pred
            .vectors
            .iter()
            .zip(gt.vectors.iter())
            .flat_map(|(pr, gr)| {
                pr.iter().zip(gr.iter()).map(|(p, g)| {
                    let dx = p.dx - g.dx;
                    let dy = p.dy - g.dy;
                    (dx * dx + dy * dy).sqrt()
                })
            })
            .sum();
        Some(sum / n)
    }

    /// Mean Angular Error (in degrees) between predicted and ground-truth flow vectors.
    ///
    /// Uses the formula: `arccos(dot(u_p, u_g) / (|u_p| * |u_g|))` where vectors are
    /// lifted to 3-D with a constant z=1 (standard optical-flow convention).
    ///
    /// Returns `None` when dimensions differ.
    pub fn angular_error(pred: &OpticalFlowField, gt: &OpticalFlowField) -> Option<f32> {
        if pred.width != gt.width || pred.height != gt.height {
            return None;
        }
        let n = (pred.width * pred.height) as f32;
        if n < f32::EPSILON {
            return Some(0.0);
        }
        let sum: f32 = pred
            .vectors
            .iter()
            .zip(gt.vectors.iter())
            .flat_map(|(pr, gr)| {
                pr.iter().zip(gr.iter()).map(|(p, g)| {
                    // Lift to 3-D: (dx, dy, 1)
                    let dp = (p.dx * p.dx + p.dy * p.dy + 1.0).sqrt();
                    let dg = (g.dx * g.dx + g.dy * g.dy + 1.0).sqrt();
                    let dot = p.dx * g.dx + p.dy * g.dy + 1.0;
                    let cos_theta = (dot / (dp * dg)).clamp(-1.0, 1.0);
                    cos_theta.acos().to_degrees()
                })
            })
            .sum();
        Some(sum / n)
    }

    /// Flow coverage: fraction of pixels where `|flow| > threshold`.
    pub fn flow_coverage(flow: &OpticalFlowField, threshold: f32) -> f32 {
        let n = flow.width * flow.height;
        if n == 0 {
            return 0.0;
        }
        let count = flow
            .vectors
            .iter()
            .flat_map(|row| row.iter())
            .filter(|fv| fv.magnitude() > threshold)
            .count();
        count as f32 / n as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(w: usize, h: usize, fill: f32) -> Vec<f32> {
        vec![fill; w * h * 3]
    }

    fn make_gradient_frame(w: usize, h: usize) -> Vec<f32> {
        let len = w * h * 3;
        (0..len).map(|i| (i as f32 / len as f32).clamp(0.0, 1.0)).collect()
    }

    #[test]
    fn test_flow_vector_magnitude() {
        let fv = FlowVector::new(3.0, 4.0);
        assert!((fv.magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_flow_vector_angle() {
        let right = FlowVector::new(1.0, 0.0);
        assert!((right.angle_degrees() - 0.0).abs() < 1e-4);

        let up = FlowVector::new(0.0, -1.0); // negative-y is upward in image coords
        assert!((up.angle_degrees() - (-90.0)).abs() < 1e-4);
    }

    #[test]
    fn test_flow_field_zeros() {
        let ff = FlowField::zeros(4, 4);
        assert_eq!(ff.flows.len(), 16);
        assert!((ff.mean_magnitude() - 0.0).abs() < 1e-8);
    }

    #[test]
    fn test_flow_field_get() {
        let mut flows = vec![FlowVector::zero(); 16];
        flows[4 + 2] = FlowVector::new(1.0, -1.0);
        let ff = FlowField::new(flows, 4, 4);
        let fv = ff.get(2, 1).expect("should be in bounds");
        assert!((fv.dx - 1.0).abs() < 1e-6);
        assert!((fv.dy - (-1.0)).abs() < 1e-6);
        assert!(ff.get(4, 0).is_none());
    }

    #[test]
    fn test_flow_field_mean_magnitude() {
        // 4 vectors of magnitude 5 (3-4-5 triangle) and 12 zeros = mean = 4*5/16 = 1.25
        let mut flows = vec![FlowVector::zero(); 16];
        for i in 0..4 {
            flows[i] = FlowVector::new(3.0, 4.0);
        }
        let ff = FlowField::new(flows, 4, 4);
        assert!((ff.mean_magnitude() - 1.25).abs() < 1e-5);
    }

    #[test]
    fn test_flow_field_max_magnitude() {
        let mut flows = vec![FlowVector::zero(); 9];
        flows[4] = FlowVector::new(6.0, 8.0); // magnitude = 10
        let ff = FlowField::new(flows, 3, 3);
        assert!((ff.max_magnitude() - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_flow_field_epe_identical() {
        let ff1 = FlowField::zeros(4, 4);
        let ff2 = FlowField::zeros(4, 4);
        let epe = ff1.epe(&ff2).expect("epe should succeed");
        assert!((epe - 0.0).abs() < 1e-8);
    }

    #[test]
    fn test_flow_field_epe_known_value() {
        // All vectors differ by (3, 4) → EPE per pixel = 5
        let ff1 = FlowField::zeros(2, 2);
        let flows2 = vec![FlowVector::new(3.0, 4.0); 4];
        let ff2 = FlowField::new(flows2, 2, 2);
        let epe = ff1.epe(&ff2).expect("epe should succeed");
        assert!((epe - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_flow_field_epe_dimension_mismatch() {
        let ff1 = FlowField::zeros(4, 4);
        let ff2 = FlowField::zeros(2, 2);
        let result = ff1.epe(&ff2);
        assert!(result.is_err());
        match result.unwrap_err() {
            FlowError::DimensionMismatch { .. } => {},
            other => panic!("unexpected error: {}", other),
        }
    }

    #[test]
    fn test_flow_field_warp_frame() {
        // Constant frame; warping should return the same constant
        let frame = make_frame(4, 4, 0.5);
        let ff = FlowField::zeros(4, 4);
        let warped = ff.warp_frame(&frame).expect("warp should succeed");
        assert_eq!(warped.len(), frame.len());
        for &v in &warped {
            assert!((v - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_flow_field_color_visualization() {
        let ff = FlowField::zeros(4, 4);
        let vis = ff.to_color_visualization();
        assert_eq!(vis.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_flow_pyramid_from_base() {
        let base = FlowField::zeros(8, 8);
        let pyramid = FlowPyramid::from_base(base, 3);
        assert_eq!(pyramid.levels.len(), 3);
        assert_eq!(pyramid.levels[0].width, 8);
        assert_eq!(pyramid.levels[0].height, 8);
        assert_eq!(pyramid.levels[1].width, 4);
        assert_eq!(pyramid.levels[1].height, 4);
        assert_eq!(pyramid.levels[2].width, 2);
        assert_eq!(pyramid.levels[2].height, 2);
    }

    #[test]
    fn test_flow_pyramid_levels() {
        let base = FlowField::zeros(4, 4);
        let pyramid = FlowPyramid::from_base(base, 2);
        let collapsed = pyramid.collapse();
        assert_eq!(collapsed.width, 4);
        assert_eq!(collapsed.height, 4);
    }

    #[test]
    fn test_optical_flow_pipeline_run() {
        let pipeline = OpticalFlowPipeline::new("raft");
        let frame1 = make_gradient_frame(4, 4);
        let frame2 = make_frame(4, 4, 0.3);
        let flow = pipeline.run(&frame1, &frame2, 4, 4).expect("run should succeed");
        assert_eq!(flow.width, 4);
        assert_eq!(flow.height, 4);
        assert_eq!(flow.flows.len(), 16);
    }

    #[test]
    fn test_optical_flow_bidirectional() {
        let pipeline = OpticalFlowPipeline::new("raft");
        let frame1 = make_gradient_frame(4, 4);
        let frame2 = make_frame(4, 4, 0.3);
        let (fwd, bwd) = pipeline
            .run_bidirectional(&frame1, &frame2, 4, 4)
            .expect("bidirectional should succeed");
        assert_eq!(fwd.width, 4);
        assert_eq!(bwd.width, 4);
    }

    #[test]
    fn test_optical_flow_occlusion_mask() {
        let pipeline = OpticalFlowPipeline::new("raft");
        let frame1 = make_gradient_frame(4, 4);
        let frame2 = make_frame(4, 4, 0.3);
        let (fwd, bwd) = pipeline
            .run_bidirectional(&frame1, &frame2, 4, 4)
            .expect("bidirectional should succeed");
        let occ = pipeline.occlusion_mask(&fwd, &bwd).expect("occlusion should succeed");
        assert_eq!(occ.len(), 16);
        // All pixels should have a boolean value — just verify no panic and correct length
    }

    #[test]
    fn test_flow_error_display() {
        let e1 = FlowError::DimensionMismatch {
            self_shape: (4, 4),
            other_shape: (8, 8),
        };
        assert!(e1.to_string().contains("4x4"));
        assert!(e1.to_string().contains("8x8"));

        let e2 = FlowError::EmptyFrame;
        assert!(e2.to_string().contains("empty"));

        let e3 = FlowError::InvalidDimensions;
        assert!(e3.to_string().contains("invalid"));
    }

    // -----------------------------------------------------------------------
    // OpticalFlowField tests
    // -----------------------------------------------------------------------

    #[test]
    fn optical_flow_field_zeros_dimensions() {
        let f = OpticalFlowField::zeros(5, 3);
        assert_eq!(f.width, 5);
        assert_eq!(f.height, 3);
        assert_eq!(f.vectors.len(), 3);
        assert_eq!(f.vectors[0].len(), 5);
    }

    #[test]
    fn optical_flow_field_zeros_magnitude_zero() {
        let f = OpticalFlowField::zeros(4, 4);
        assert!((f.mean_magnitude() - 0.0).abs() < 1e-8);
        assert!((f.max_magnitude() - 0.0).abs() < 1e-8);
    }

    #[test]
    fn optical_flow_field_magnitude_map_shape() {
        let f = OpticalFlowField::zeros(3, 2);
        let map = f.magnitude_map();
        assert_eq!(map.len(), 2);
        assert_eq!(map[0].len(), 3);
    }

    #[test]
    fn optical_flow_field_magnitude_map_values() {
        let mut f = OpticalFlowField::zeros(2, 2);
        f.vectors[0][0] = FlowVector::new(3.0, 4.0); // magnitude = 5
        let map = f.magnitude_map();
        assert!((map[0][0] - 5.0).abs() < 1e-5);
        assert!((map[0][1]).abs() < 1e-8);
    }

    #[test]
    fn optical_flow_field_mean_magnitude() {
        // 4 pixels, one has magnitude 5, rest are 0 → mean = 5/4 = 1.25
        let mut f = OpticalFlowField::zeros(2, 2);
        f.vectors[0][0] = FlowVector::new(3.0, 4.0);
        assert!((f.mean_magnitude() - 1.25).abs() < 1e-5);
    }

    #[test]
    fn optical_flow_field_max_magnitude() {
        let mut f = OpticalFlowField::zeros(3, 3);
        f.vectors[1][1] = FlowVector::new(6.0, 8.0); // magnitude = 10
        assert!((f.max_magnitude() - 10.0).abs() < 1e-5);
    }

    #[test]
    fn optical_flow_field_to_hsv_image_size() {
        let f = OpticalFlowField::zeros(4, 3);
        let img = f.to_hsv_image();
        assert_eq!(img.len(), 4 * 3 * 3);
    }

    #[test]
    fn optical_flow_field_to_hsv_image_zero_flow_is_black() {
        // Zero magnitude → value = 0 → black pixel
        let f = OpticalFlowField::zeros(2, 2);
        let img = f.to_hsv_image();
        // All bytes should be 0 (black)
        for &byte in &img {
            assert_eq!(byte, 0);
        }
    }

    #[test]
    fn optical_flow_field_to_hsv_nonzero_flow() {
        let mut f = OpticalFlowField::zeros(1, 1);
        f.vectors[0][0] = FlowVector::new(1.0, 0.0); // rightward flow
        let img = f.to_hsv_image();
        assert_eq!(img.len(), 3);
        // At least one channel should be non-zero
        assert!(img.iter().any(|&b| b > 0));
    }

    // -----------------------------------------------------------------------
    // OpticalFlowMetrics tests
    // -----------------------------------------------------------------------

    #[test]
    fn epe_identical_fields_zero() {
        let pred = OpticalFlowField::zeros(4, 4);
        let gt = OpticalFlowField::zeros(4, 4);
        let epe = OpticalFlowMetrics::endpoint_error(&pred, &gt).expect("epe");
        assert!((epe).abs() < 1e-8);
    }

    #[test]
    fn epe_known_displacement() {
        // All vectors differ by (3, 4) → EPE = 5
        let pred = OpticalFlowField::zeros(2, 2);
        let gt = {
            let row = vec![FlowVector::new(3.0, 4.0); 2];
            let vectors = vec![row.clone(), row];
            OpticalFlowField::new(vectors, 2, 2)
        };
        let epe = OpticalFlowMetrics::endpoint_error(&pred, &gt).expect("epe");
        assert!((epe - 5.0).abs() < 1e-5);
    }

    #[test]
    fn epe_dimension_mismatch_returns_none() {
        let pred = OpticalFlowField::zeros(4, 4);
        let gt = OpticalFlowField::zeros(2, 2);
        assert!(OpticalFlowMetrics::endpoint_error(&pred, &gt).is_none());
    }

    #[test]
    fn angular_error_identical_fields_zero() {
        let pred = {
            let row = vec![FlowVector::new(1.0, 0.0); 2];
            OpticalFlowField::new(vec![row.clone(), row], 2, 2)
        };
        let gt = pred.clone();
        let ae = OpticalFlowMetrics::angular_error(&pred, &gt).expect("angular_error");
        assert!(
            ae < 1e-3,
            "angular error should be ~0 for identical fields, got {ae}"
        );
    }

    #[test]
    fn angular_error_dimension_mismatch_returns_none() {
        let pred = OpticalFlowField::zeros(3, 3);
        let gt = OpticalFlowField::zeros(2, 2);
        assert!(OpticalFlowMetrics::angular_error(&pred, &gt).is_none());
    }

    #[test]
    fn flow_coverage_zero_field_zero_threshold() {
        let f = OpticalFlowField::zeros(4, 4);
        let cov = OpticalFlowMetrics::flow_coverage(&f, 0.0);
        // Zero-magnitude vectors are NOT > 0 threshold, but threshold 0.0 means > 0.0 which zero vectors fail
        assert!((cov).abs() < 1e-8);
    }

    #[test]
    fn flow_coverage_full_coverage() {
        let row = vec![FlowVector::new(3.0, 4.0); 3];
        let f = OpticalFlowField::new(vec![row.clone(), row], 3, 2);
        // All magnitudes = 5, threshold = 1 → full coverage
        let cov = OpticalFlowMetrics::flow_coverage(&f, 1.0);
        assert!((cov - 1.0).abs() < 1e-5);
    }

    #[test]
    fn flow_coverage_partial() {
        let mut f = OpticalFlowField::zeros(2, 2);
        f.vectors[0][0] = FlowVector::new(3.0, 4.0); // magnitude 5
                                                     // 1 out of 4 pixels has magnitude > 2 → coverage = 0.25
        let cov = OpticalFlowMetrics::flow_coverage(&f, 2.0);
        assert!((cov - 0.25).abs() < 1e-5);
    }
}
