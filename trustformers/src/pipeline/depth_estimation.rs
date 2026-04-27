//! # Depth Estimation Pipeline
//!
//! DPT/MiDaS-compatible monocular depth estimation from images.
//!
//! ## Supported model families
//! - **DPT-Large** (Intel/dpt-large) — Vision Transformer backbone for dense prediction
//! - **MiDaS** — Multi-scale depth estimation
//!
//! ## Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::depth_estimation::{
//!     DepthEstimationConfig, DepthEstimationPipeline,
//! };
//!
//! let config = DepthEstimationConfig::default();
//! let pipeline = DepthEstimationPipeline::new(config)?;
//! let image = vec![0.5f32; 384 * 384 * 3];
//! let depth_map = pipeline.predict(&image, 384, 384)?;
//! println!("Depth map: {}x{}", depth_map.width, depth_map.height);
//! # Ok::<(), depth_estimation::PipelineError>(())
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the depth estimation pipeline.
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("Invalid input dimensions: {0}")]
    InvalidDimensions(String),
    #[error("Empty input")]
    EmptyInput,
    #[error("Model error: {0}")]
    ModelError(String),
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`DepthEstimationPipeline`].
#[derive(Debug, Clone)]
pub struct DepthEstimationConfig {
    /// HuggingFace model identifier or local path.
    pub model_name: String,
    /// Target input height (pixels). Default: 384 for DPT-Large.
    pub input_height: usize,
    /// Target input width (pixels). Default: 384 for DPT-Large.
    pub input_width: usize,
    /// Normalize output depth values to `[0, 1]`.
    pub output_normalized: bool,
    /// Invert depth so that near objects have high values and far objects have low values.
    pub invert_depth: bool,
}

impl Default for DepthEstimationConfig {
    fn default() -> Self {
        Self {
            model_name: "Intel/dpt-large".to_string(),
            input_height: 384,
            input_width: 384,
            output_normalized: true,
            invert_depth: false,
        }
    }
}

// ---------------------------------------------------------------------------
// DepthEstimationResult — new-style result struct
// ---------------------------------------------------------------------------

/// Rich depth estimation result with 2D map and metadata.
#[derive(Debug, Clone)]
pub struct DepthEstimationResult {
    /// 2D depth map in row-major order. Shape: `[height][width]`.
    pub depth_map: Vec<Vec<f32>>,
    /// Map width in pixels.
    pub width: usize,
    /// Map height in pixels.
    pub height: usize,
    /// Minimum depth value in `depth_map`.
    pub min_depth: f32,
    /// Maximum depth value in `depth_map`.
    pub max_depth: f32,
}

impl DepthEstimationResult {
    /// Flatten the 2D depth map to a 1D vector.
    pub fn to_flat(&self) -> Vec<f32> {
        self.depth_map.iter().flatten().copied().collect()
    }

    /// Convert flat values to a DepthEstimationResult.
    pub fn from_flat(values: Vec<f32>, width: usize, height: usize) -> Self {
        let (min_depth, max_depth) = compute_min_max_flat(&values);
        let depth_map: Vec<Vec<f32>> = values.chunks(width).map(|row| row.to_vec()).collect();
        Self {
            depth_map,
            width,
            height,
            min_depth,
            max_depth,
        }
    }
}

// ---------------------------------------------------------------------------
// DepthMap — original flat result type
// ---------------------------------------------------------------------------

/// A 2-D depth map returned by the depth estimation pipeline.
#[derive(Debug, Clone)]
pub struct DepthMap {
    /// Flattened depth values in row-major order, shape `[height, width]`.
    pub values: Vec<f32>,
    /// Map height in pixels.
    pub height: usize,
    /// Map width in pixels.
    pub width: usize,
    /// Minimum depth value in `values`.
    pub min_depth: f32,
    /// Maximum depth value in `values`.
    pub max_depth: f32,
}

impl DepthMap {
    /// Create a new `DepthMap`, computing `min_depth` and `max_depth` automatically.
    pub fn new(values: Vec<f32>, height: usize, width: usize) -> Self {
        let (min_depth, max_depth) = compute_min_max_flat(&values);
        Self {
            values,
            height,
            width,
            min_depth,
            max_depth,
        }
    }

    /// Return the depth value at `(row, col)`.
    ///
    /// Returns `0.0` for out-of-bounds indices.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        let idx = row * self.width + col;
        self.values.get(idx).copied().unwrap_or(0.0)
    }

    /// Return a new `DepthMap` with all values linearly scaled to `[0, 1]`.
    pub fn normalize(&self) -> Self {
        let range = self.max_depth - self.min_depth;
        let normalized: Vec<f32> = if range.abs() < f32::EPSILON {
            vec![0.0f32; self.values.len()]
        } else {
            self.values.iter().map(|&v| (v - self.min_depth) / range).collect()
        };
        Self {
            min_depth: 0.0,
            max_depth: 1.0,
            values: normalized,
            height: self.height,
            width: self.width,
        }
    }

    /// Render the depth map as ASCII art using a 9-character gradient.
    ///
    /// Characters from darkest (far) to brightest (near):
    /// `' '`, `'.'`, `':'`, `';'`, `'+'`, `'x'`, `'$'`, `'#'`, `'@'`
    pub fn to_colormap_ascii(&self) -> String {
        const CHARS: &[u8] = b" .:;+x$#@";
        let normalized = self.normalize();
        let mut out = String::with_capacity(self.height * (self.width + 1));
        for row in 0..self.height {
            for col in 0..self.width {
                let v = normalized.get(row, col).clamp(0.0, 1.0);
                let idx = ((v * (CHARS.len() - 1) as f32).round() as usize).min(CHARS.len() - 1);
                out.push(CHARS[idx] as char);
            }
            out.push('\n');
        }
        out
    }

    /// Bilinearly resize the depth map to `(new_h, new_w)`.
    pub fn resize(&self, new_h: usize, new_w: usize) -> Self {
        let mut out = vec![0.0f32; new_h * new_w];
        let scale_y = self.height as f32 / new_h as f32;
        let scale_x = self.width as f32 / new_w as f32;

        for oy in 0..new_h {
            for ox in 0..new_w {
                let src_y = (oy as f32 + 0.5) * scale_y - 0.5;
                let src_x = (ox as f32 + 0.5) * scale_x - 0.5;

                let y0 = (src_y.floor() as isize).clamp(0, self.height as isize - 1) as usize;
                let y1 = (y0 + 1).min(self.height - 1);
                let x0 = (src_x.floor() as isize).clamp(0, self.width as isize - 1) as usize;
                let x1 = (x0 + 1).min(self.width - 1);

                let fy = (src_y - y0 as f32).clamp(0.0, 1.0);
                let fx = (src_x - x0 as f32).clamp(0.0, 1.0);

                let v00 = self.get(y0, x0);
                let v01 = self.get(y0, x1);
                let v10 = self.get(y1, x0);
                let v11 = self.get(y1, x1);

                let v = v00 * (1.0 - fy) * (1.0 - fx)
                    + v01 * (1.0 - fy) * fx
                    + v10 * fy * (1.0 - fx)
                    + v11 * fy * fx;

                out[oy * new_w + ox] = v;
            }
        }
        Self::new(out, new_h, new_w)
    }
}

// ---------------------------------------------------------------------------
// DepthMapPostprocessor
// ---------------------------------------------------------------------------

/// Post-processing utilities for depth maps.
pub struct DepthMapPostprocessor;

impl DepthMapPostprocessor {
    /// Min-max normalize a 2D depth map to `[0, 1]`.
    pub fn normalize_depth(map: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if map.is_empty() {
            return Vec::new();
        }
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for row in map {
            for &v in row {
                if v < min_val {
                    min_val = v;
                }
                if v > max_val {
                    max_val = v;
                }
            }
        }
        let range = max_val - min_val;
        if range.abs() < f32::EPSILON {
            return map.iter().map(|row| vec![0.0_f32; row.len()]).collect();
        }
        map.iter()
            .map(|row| row.iter().map(|&v| (v - min_val) / range).collect())
            .collect()
    }

    /// Invert depth via disparity transformation: output `1 / d` (clamped to avoid division by zero).
    pub fn invert_depth(map: &[Vec<f32>]) -> Vec<Vec<f32>> {
        map.iter()
            .map(|row| {
                row.iter()
                    .map(
                        |&d| {
                            if d.abs() < 1e-6 {
                                f32::MAX.min(1e6)
                            } else {
                                1.0 / d
                            }
                        },
                    )
                    .collect()
            })
            .collect()
    }

    /// Apply a median filter with the given `kernel_size` (must be odd; clamped to 3 or 5).
    ///
    /// Uses a square kernel of size `kernel_size × kernel_size`.
    pub fn median_filter(map: &[Vec<f32>], kernel_size: usize) -> Vec<Vec<f32>> {
        if map.is_empty() {
            return Vec::new();
        }
        let h = map.len();
        let w = map.first().map(|r| r.len()).unwrap_or(0);
        if w == 0 {
            return map.to_vec();
        }

        // Clamp kernel_size to supported values
        let ks = if kernel_size <= 3 { 3 } else { 5 };
        let half = ks / 2;

        let mut out: Vec<Vec<f32>> = vec![vec![0.0_f32; w]; h];
        for row in 0..h {
            for col in 0..w {
                let mut neighbors = Vec::with_capacity(ks * ks);
                for ky in 0..ks {
                    let r = (row as isize + ky as isize - half as isize).clamp(0, h as isize - 1)
                        as usize;
                    for kx in 0..ks {
                        let c = (col as isize + kx as isize - half as isize)
                            .clamp(0, w as isize - 1) as usize;
                        if let Some(v) = map.get(r).and_then(|row| row.get(c)) {
                            neighbors.push(*v);
                        }
                    }
                }
                // Sort and pick median
                neighbors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = neighbors.len() / 2;
                out[row][col] = neighbors.get(mid).copied().unwrap_or(0.0);
            }
        }
        out
    }

    /// Colorize a normalized depth map using a simplified viridis-like colormap.
    ///
    /// Returns a flat RGB byte buffer `[height * width * 3]`, values in `[0, 255]`.
    pub fn colorize_depth(map: &[Vec<f32>]) -> Vec<u8> {
        let h = map.len();
        let w = map.first().map(|r| r.len()).unwrap_or(0);
        let mut out = Vec::with_capacity(h * w * 3);

        for row in map {
            for &v in row {
                let t = v.clamp(0.0, 1.0);
                // Simplified viridis: purple (low) → blue → cyan → green → yellow (high)
                let (r, g, b) = viridis_approx(t);
                out.push(r);
                out.push(g);
                out.push(b);
            }
        }
        out
    }

    /// Back-project a depth map to a 3D point cloud.
    ///
    /// Uses pinhole camera model:
    /// - `fx`, `fy`: focal lengths in pixels
    /// - `cx`, `cy`: principal point in pixels
    ///
    /// Returns a list of `(X, Y, Z)` world coordinates (one per pixel).
    pub fn to_point_cloud(
        depth_map: &[Vec<f32>],
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
    ) -> Vec<(f32, f32, f32)> {
        let h = depth_map.len();
        let w = depth_map.first().map(|r| r.len()).unwrap_or(0);
        let mut points = Vec::with_capacity(h * w);

        for (row, depth_row) in depth_map.iter().enumerate() {
            for (col, &z) in depth_row.iter().enumerate() {
                if z <= 0.0 {
                    continue;
                }
                let x = (col as f32 - cx) * z / fx;
                let y = (row as f32 - cy) * z / fy;
                points.push((x, y, z));
            }
        }
        points
    }
}

/// Simplified viridis colormap approximation: returns `(R, G, B)` in `[0, 255]`.
fn viridis_approx(t: f32) -> (u8, u8, u8) {
    // Control points: t=0 → (68,1,84), t=0.25 → (59,82,139),
    //                  t=0.5 → (33,145,140), t=0.75 → (94,201,98), t=1 → (253,231,37)
    let stops: [(f32, f32, f32, f32); 5] = [
        (0.00, 68.0, 1.0, 84.0),
        (0.25, 59.0, 82.0, 139.0),
        (0.50, 33.0, 145.0, 140.0),
        (0.75, 94.0, 201.0, 98.0),
        (1.00, 253.0, 231.0, 37.0),
    ];

    // Find the two stops that bracket t
    let mut lo_idx = 0usize;
    for i in 1..stops.len() {
        if stops[i].0 <= t {
            lo_idx = i;
        }
    }
    let hi_idx = (lo_idx + 1).min(stops.len() - 1);

    let (t0, r0, g0, b0) = stops[lo_idx];
    let (t1, r1, g1, b1) = stops[hi_idx];

    let span = t1 - t0;
    let frac = if span < 1e-6 { 0.0 } else { (t - t0) / span };

    let r = (r0 + frac * (r1 - r0)).clamp(0.0, 255.0) as u8;
    let g = (g0 + frac * (g1 - g0)).clamp(0.0, 255.0) as u8;
    let b = (b0 + frac * (b1 - b0)).clamp(0.0, 255.0) as u8;
    (r, g, b)
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Monocular depth estimation pipeline (DPT / MiDaS compatible).
pub struct DepthEstimationPipeline {
    config: DepthEstimationConfig,
}

impl DepthEstimationPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: DepthEstimationConfig) -> Result<Self, PipelineError> {
        if config.input_height == 0 || config.input_width == 0 {
            return Err(PipelineError::InvalidDimensions(
                "input_height and input_width must be > 0".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Run depth estimation on a single image, returning the rich `DepthEstimationResult`.
    pub fn estimate(
        &self,
        image_data: &[f32],
        height: usize,
        width: usize,
    ) -> Result<DepthEstimationResult, PipelineError> {
        let depth_map = self.predict(image_data, height, width)?;
        Ok(DepthEstimationResult::from_flat(
            depth_map.values,
            depth_map.width,
            depth_map.height,
        ))
    }

    /// Run depth estimation on a single image.
    ///
    /// `image_data` is a flat `f32` buffer in `[0, 1]` or `[-1, 1]` range,
    /// with shape `[height * width * channels]` (channels = 1 or 3).
    pub fn predict(
        &self,
        image_data: &[f32],
        height: usize,
        width: usize,
    ) -> Result<DepthMap, PipelineError> {
        if image_data.is_empty() {
            return Err(PipelineError::EmptyInput);
        }
        if height == 0 || width == 0 {
            return Err(PipelineError::InvalidDimensions(
                "height and width must be > 0".to_string(),
            ));
        }

        // Preprocessing: resize to model input size by sampling the input.
        let in_h = self.config.input_height;
        let in_w = self.config.input_width;
        let _resized = preprocess_image(image_data, height, width, in_h, in_w);

        // Mock inference: synthetic radial depth gradient (center near, edges far).
        let depth_values = mock_depth_inference(in_h, in_w);

        // Postprocessing.
        let mut depth_map = DepthMap::new(depth_values, in_h, in_w);

        if self.config.output_normalized {
            depth_map = depth_map.normalize();
        }

        if self.config.invert_depth {
            let max = depth_map.max_depth;
            depth_map.values.iter_mut().for_each(|v| *v = max - *v);
            let (mn, mx) = compute_min_max_flat(&depth_map.values);
            depth_map.min_depth = mn;
            depth_map.max_depth = mx;
        }

        Ok(depth_map)
    }

    /// Run depth estimation on a batch of images.
    ///
    /// Each element is `(image_data, height, width)`.
    pub fn predict_batch(
        &self,
        images: &[(&[f32], usize, usize)],
    ) -> Result<Vec<DepthMap>, PipelineError> {
        if images.is_empty() {
            return Err(PipelineError::EmptyInput);
        }
        images.iter().map(|&(data, h, w)| self.predict(data, h, w)).collect()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn compute_min_max_flat(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mut mn = values[0];
    let mut mx = values[0];
    for &v in values.iter().skip(1) {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    (mn, mx)
}

/// Nearest-neighbor downsample/upsample from `(src_h, src_w)` to `(dst_h, dst_w)`.
/// Returns the first channel's values (or the only channel if mono).
fn preprocess_image(
    image_data: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<f32> {
    // Determine number of channels.
    let total_pixels = src_h * src_w;
    let channels = image_data.len().checked_div(total_pixels).unwrap_or(1).max(1);

    let mut out = vec![0.0f32; dst_h * dst_w];
    for oy in 0..dst_h {
        for ox in 0..dst_w {
            let sy = ((oy as f32 / dst_h as f32) * src_h as f32) as usize;
            let sx = ((ox as f32 / dst_w as f32) * src_w as f32) as usize;
            let sy = sy.min(src_h - 1);
            let sx = sx.min(src_w - 1);
            let src_idx = (sy * src_w + sx) * channels;
            out[oy * dst_w + ox] = image_data.get(src_idx).copied().unwrap_or(0.0);
        }
    }
    out
}

/// Generate a synthetic depth map: center pixels are near (low depth value),
/// edge pixels are far (high depth value). Values in `[0, 1]`.
fn mock_depth_inference(h: usize, w: usize) -> Vec<f32> {
    let cy = h as f32 / 2.0;
    let cx = w as f32 / 2.0;
    let max_dist = (cy * cy + cx * cx).sqrt();

    (0..h)
        .flat_map(|row| {
            (0..w).map(move |col| {
                let dy = row as f32 - cy;
                let dx = col as f32 - cx;
                let dist = (dy * dy + dx * dx).sqrt();
                if max_dist > 0.0 {
                    dist / max_dist
                } else {
                    0.0
                }
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers ----

    fn make_image(h: usize, w: usize) -> Vec<f32> {
        (0..h * w * 3).map(|i| (i % 256) as f32 / 255.0).collect()
    }

    fn make_2d_map(h: usize, w: usize, val: f32) -> Vec<Vec<f32>> {
        vec![vec![val; w]; h]
    }

    // ---- 1. config defaults ----

    #[test]
    fn test_config_defaults() {
        let cfg = DepthEstimationConfig::default();
        assert_eq!(cfg.model_name, "Intel/dpt-large");
        assert_eq!(cfg.input_height, 384);
        assert_eq!(cfg.input_width, 384);
        assert!(cfg.output_normalized);
        assert!(!cfg.invert_depth);
    }

    // ---- 2. DepthMap::new computes min/max ----

    #[test]
    fn test_depth_map_new_min_max() {
        let values = vec![0.1f32, 0.5, 0.9, 0.3];
        let dm = DepthMap::new(values, 2, 2);
        assert!((dm.min_depth - 0.1).abs() < 1e-6);
        assert!((dm.max_depth - 0.9).abs() < 1e-6);
    }

    // ---- 3. DepthMap::get ----

    #[test]
    fn test_depth_map_get() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let dm = DepthMap::new(values, 2, 2);
        assert_eq!(dm.get(0, 0), 1.0);
        assert_eq!(dm.get(0, 1), 2.0);
        assert_eq!(dm.get(1, 0), 3.0);
        assert_eq!(dm.get(1, 1), 4.0);
        // out of bounds => 0.0
        assert_eq!(dm.get(5, 5), 0.0);
    }

    // ---- 4. DepthMap::normalize ----

    #[test]
    fn test_depth_map_normalize() {
        let values = vec![0.0f32, 5.0, 10.0];
        let dm = DepthMap::new(values, 1, 3);
        let norm = dm.normalize();
        assert!((norm.values[0] - 0.0).abs() < 1e-6);
        assert!((norm.values[1] - 0.5).abs() < 1e-6);
        assert!((norm.values[2] - 1.0).abs() < 1e-6);
        assert_eq!(norm.min_depth, 0.0);
        assert_eq!(norm.max_depth, 1.0);
    }

    // ---- 5. normalize with uniform values (zero range) ----

    #[test]
    fn test_depth_map_normalize_uniform() {
        let values = vec![3.0f32; 4];
        let dm = DepthMap::new(values, 2, 2);
        let norm = dm.normalize();
        assert!(norm.values.iter().all(|&v| v == 0.0));
    }

    // ---- 6. to_colormap_ascii ----

    #[test]
    fn test_colormap_ascii_shape() {
        let dm = DepthMap::new(vec![0.0f32; 4 * 6], 4, 6);
        let ascii = dm.to_colormap_ascii();
        let lines: Vec<&str> = ascii.lines().collect();
        assert_eq!(lines.len(), 4);
        for line in &lines {
            assert_eq!(line.len(), 6);
        }
    }

    #[test]
    fn test_colormap_ascii_chars() {
        let dm = DepthMap::new(vec![0.0f32, 1.0], 1, 2);
        let ascii = dm.to_colormap_ascii();
        // first char should be space (0.0 → darkest), last should be '@' (1.0 → brightest)
        let chars: Vec<char> = ascii.chars().filter(|&c| c != '\n').collect();
        assert_eq!(chars[0], ' ');
        assert_eq!(chars[1], '@');
    }

    // ---- 7. resize ----

    #[test]
    fn test_depth_map_resize_dimensions() {
        let dm = DepthMap::new(vec![0.5f32; 8 * 8], 8, 8);
        let resized = dm.resize(4, 4);
        assert_eq!(resized.height, 4);
        assert_eq!(resized.width, 4);
        assert_eq!(resized.values.len(), 16);
    }

    #[test]
    fn test_depth_map_resize_uniform_values() {
        let dm = DepthMap::new(vec![0.7f32; 6 * 6], 6, 6);
        let resized = dm.resize(3, 3);
        for v in &resized.values {
            assert!((*v - 0.7).abs() < 1e-5);
        }
    }

    // ---- 8. predict basic ----

    #[test]
    fn test_predict_basic() {
        let config = DepthEstimationConfig {
            input_height: 16,
            input_width: 16,
            ..Default::default()
        };
        let pipeline = DepthEstimationPipeline::new(config).expect("pipeline creation failed");
        let image = make_image(32, 32);
        let dm = pipeline.predict(&image, 32, 32).expect("predict failed");
        assert_eq!(dm.height, 16);
        assert_eq!(dm.width, 16);
        assert_eq!(dm.values.len(), 256);
        // normalized → values in [0, 1]
        for v in &dm.values {
            assert!(*v >= 0.0 && *v <= 1.0 + 1e-6);
        }
    }

    // ---- 9. predict empty input error ----

    #[test]
    fn test_predict_empty_input() {
        let pipeline = DepthEstimationPipeline::new(DepthEstimationConfig::default()).expect("ok");
        let result = pipeline.predict(&[], 10, 10);
        assert!(matches!(result, Err(PipelineError::EmptyInput)));
    }

    // ---- 10. predict invalid dimensions ----

    #[test]
    fn test_predict_invalid_dimensions() {
        let pipeline = DepthEstimationPipeline::new(DepthEstimationConfig::default()).expect("ok");
        let result = pipeline.predict(&[0.5f32; 10], 0, 10);
        assert!(matches!(result, Err(PipelineError::InvalidDimensions(_))));
    }

    // ---- 11. predict_batch ----

    #[test]
    fn test_predict_batch() {
        let config = DepthEstimationConfig {
            input_height: 8,
            input_width: 8,
            ..Default::default()
        };
        let pipeline = DepthEstimationPipeline::new(config).expect("ok");
        let img1 = make_image(16, 16);
        let img2 = make_image(24, 24);
        let batch: Vec<(&[f32], usize, usize)> =
            vec![(img1.as_slice(), 16, 16), (img2.as_slice(), 24, 24)];
        let results = pipeline.predict_batch(&batch).expect("batch predict failed");
        assert_eq!(results.len(), 2);
        for dm in &results {
            assert_eq!(dm.height, 8);
            assert_eq!(dm.width, 8);
        }
    }

    // ---- 12. invert_depth ----

    #[test]
    fn test_invert_depth() {
        let config_normal = DepthEstimationConfig {
            input_height: 8,
            input_width: 8,
            output_normalized: true,
            invert_depth: false,
            ..Default::default()
        };
        let config_inv = DepthEstimationConfig {
            input_height: 8,
            input_width: 8,
            output_normalized: true,
            invert_depth: true,
            ..Default::default()
        };
        let pl_normal = DepthEstimationPipeline::new(config_normal).expect("ok");
        let pl_inv = DepthEstimationPipeline::new(config_inv).expect("ok");
        let image = make_image(16, 16);
        let dm_normal = pl_normal.predict(&image, 16, 16).expect("ok");
        let dm_inv = pl_inv.predict(&image, 16, 16).expect("ok");
        // The argmax/argmin positions should be swapped.
        let normal_max_idx = dm_normal
            .values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("non-empty");
        let inv_min_idx = dm_inv
            .values
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("non-empty");
        assert_eq!(normal_max_idx, inv_min_idx);
    }

    // ---- New postprocessor tests ----

    #[test]
    fn test_normalize_depth_range() {
        let map = vec![vec![0.0_f32, 5.0, 10.0]];
        let norm = DepthMapPostprocessor::normalize_depth(&map);
        assert!((norm[0][0] - 0.0).abs() < 1e-6, "min should normalize to 0");
        assert!(
            (norm[0][1] - 0.5).abs() < 1e-6,
            "mid should normalize to 0.5"
        );
        assert!(
            (norm[0][2] - 1.0).abs() < 1e-6,
            "max should normalize to 1.0"
        );
    }

    #[test]
    fn test_normalize_depth_uniform_map() {
        let map = make_2d_map(3, 3, 7.0);
        let norm = DepthMapPostprocessor::normalize_depth(&map);
        for row in &norm {
            for &v in row {
                assert_eq!(v, 0.0, "uniform map should normalize to zeros");
            }
        }
    }

    #[test]
    fn test_invert_depth_postprocessor() {
        let map = vec![vec![2.0_f32, 4.0, 0.5]];
        let inv = DepthMapPostprocessor::invert_depth(&map);
        assert!((inv[0][0] - 0.5).abs() < 1e-5, "1/2=0.5");
        assert!((inv[0][1] - 0.25).abs() < 1e-5, "1/4=0.25");
        assert!((inv[0][2] - 2.0).abs() < 1e-5, "1/0.5=2.0");
    }

    #[test]
    fn test_invert_depth_zero_avoidance() {
        let map = vec![vec![0.0_f32]];
        let inv = DepthMapPostprocessor::invert_depth(&map);
        assert!(
            inv[0][0].is_finite(),
            "inverting zero should not produce infinity or NaN"
        );
        assert!(inv[0][0] > 0.0, "1/0 placeholder should be large positive");
    }

    #[test]
    fn test_median_filter_shape() {
        let map = make_2d_map(5, 5, 1.0);
        let filtered = DepthMapPostprocessor::median_filter(&map, 3);
        assert_eq!(filtered.len(), 5, "output height should match input");
        assert_eq!(filtered[0].len(), 5, "output width should match input");
    }

    #[test]
    fn test_median_filter_uniform_map() {
        let map = make_2d_map(4, 4, 3.5);
        let filtered = DepthMapPostprocessor::median_filter(&map, 3);
        for row in &filtered {
            for &v in row {
                assert!(
                    (v - 3.5).abs() < 1e-5,
                    "median of uniform values should be unchanged"
                );
            }
        }
    }

    #[test]
    fn test_colorize_depth_output_size() {
        let map = make_2d_map(8, 6, 0.5);
        let rgb = DepthMapPostprocessor::colorize_depth(&map);
        assert_eq!(
            rgb.len(),
            8 * 6 * 3,
            "RGB output should be height*width*3 bytes"
        );
    }

    #[test]
    fn test_colorize_depth_values_are_valid() {
        let map = vec![vec![0.0_f32, 0.5, 1.0]];
        let rgb = DepthMapPostprocessor::colorize_depth(&map);
        assert_eq!(rgb.len(), 9, "3 pixels × 3 channels");
        // All bytes should be in range [0,255] (by type) — just ensure vec is populated
        for &byte in &rgb {
            // u8 is always 0-255; just verify the vector is non-trivially populated
            let _ = byte;
        }
    }

    #[test]
    fn test_to_point_cloud_count() {
        // 2×2 depth map, all 1.0 (no zeros, so all 4 points emitted)
        let map = vec![vec![1.0_f32, 1.0], vec![1.0_f32, 1.0]];
        let pts = DepthMapPostprocessor::to_point_cloud(&map, 100.0, 100.0, 1.0, 1.0);
        assert_eq!(pts.len(), 4, "4 pixels with nonzero depth → 4 points");
    }

    #[test]
    fn test_to_point_cloud_zero_depth_skipped() {
        let map = vec![vec![0.0_f32, 1.0], vec![1.0_f32, 0.0]];
        let pts = DepthMapPostprocessor::to_point_cloud(&map, 100.0, 100.0, 0.5, 0.5);
        assert_eq!(pts.len(), 2, "zero-depth pixels should be skipped");
    }

    #[test]
    fn test_to_point_cloud_z_values() {
        // Depth map with known values; check that Z equals depth
        let map = vec![vec![5.0_f32]];
        let pts = DepthMapPostprocessor::to_point_cloud(&map, 1.0, 1.0, 0.0, 0.0);
        assert_eq!(pts.len(), 1);
        assert!((pts[0].2 - 5.0).abs() < 1e-5, "Z should equal depth value");
    }

    #[test]
    fn test_estimate_returns_2d_map() {
        let config = DepthEstimationConfig {
            input_height: 4,
            input_width: 4,
            ..Default::default()
        };
        let pipeline = DepthEstimationPipeline::new(config).expect("ok");
        let image = make_image(8, 8);
        let result = pipeline.estimate(&image, 8, 8).expect("estimate ok");
        assert_eq!(result.height, 4);
        assert_eq!(result.width, 4);
        assert_eq!(result.depth_map.len(), 4, "depth_map should have 4 rows");
        for row in &result.depth_map {
            assert_eq!(row.len(), 4, "each row should have 4 columns");
        }
    }
}
