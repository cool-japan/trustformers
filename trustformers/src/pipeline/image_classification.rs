//! # Image Classification Pipeline
//!
//! Maps an image (raw RGB bytes, a file path, or a pre-computed feature vector)
//! to a ranked list of category labels with associated probabilities.
//!
//! ## Supported model families
//! - **ViT** (Vision Transformer) — patch-based transformer for image classification
//! - **CLIP** — contrastive language-image pre-training; zero-shot classification
//! - **ResNet / EfficientNet** — classic convolutional baselines
//!
//! ## Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::image_classification::{
//!     ImageClassificationConfig, ImageClassificationPipeline, ImageClassificationInput,
//! };
//!
//! let config = ImageClassificationConfig {
//!     model_name: "google/vit-base-patch16-224".to_string(),
//!     top_k: 5,
//!     ..Default::default()
//! };
//!
//! let pipeline = ImageClassificationPipeline::new(config)?;
//!
//! let input = ImageClassificationInput::RgbImage {
//!     data: vec![128u8; 224 * 224 * 3],
//!     width: 224,
//!     height: 224,
//! };
//!
//! let results = pipeline.classify(&input)?;
//! for result in &results {
//!     println!("{}: {:.4}", result.label, result.score);
//! }
//! # Ok::<(), trustformers::TrustformersError>(())
//! ```

use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use trustformers_core::tensor::Tensor;

// ---------------------------------------------------------------------------
// Public types — Input
// ---------------------------------------------------------------------------

/// Image input variants supported by the classification pipeline.
#[derive(Debug, Clone)]
pub enum ImageClassificationInput {
    /// Raw RGB image bytes (no header, row-major, 3 channels per pixel).
    RgbImage {
        /// Flat byte buffer: `height × width × 3` values in `[0, 255]`.
        data: Vec<u8>,
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
    /// Path to a supported image file (JPEG, PNG, BMP, GIF, WEBP).
    FilePath(PathBuf),
    /// Pre-normalised floating-point pixel values in `[-1, 1]` or `[0, 1]`.
    NormalisedTensor {
        /// Flattened `C × H × W` float values.
        values: Vec<f32>,
        /// Number of channels (typically 3 for RGB).
        channels: u32,
        /// Height in pixels.
        height: u32,
        /// Width in pixels.
        width: u32,
    },
}

/// New-style `ImageInput` enum for the enhanced preprocessor API.
#[derive(Debug, Clone)]
pub enum ImageInput {
    /// Raw RGB pixels in HWC layout (height × width × 3 channels), values in [0, 255].
    RgbPixels {
        data: Vec<u8>,
        width: usize,
        height: usize,
    },
    /// Pre-normalized float tensor with explicit channel dimension.
    FloatTensor {
        data: Vec<f32>,
        width: usize,
        height: usize,
        channels: usize,
    },
    /// Path to an image file.
    FilePath(String),
}

// ---------------------------------------------------------------------------
// Public types — Output
// ---------------------------------------------------------------------------

/// A single label-score pair returned by the image classification pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageClassificationResult {
    /// Human-readable category label.
    pub label: String,
    /// Predicted probability in `[0.0, 1.0]`.
    pub score: f32,
    /// Zero-based label index.
    pub label_id: usize,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`ImageClassificationPipeline`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageClassificationConfig {
    /// HuggingFace model identifier or local path.
    pub model_name: String,
    /// Number of top labels to return.
    pub top_k: usize,
    /// Label set. When empty the pipeline uses built-in ImageNet-1k labels.
    pub labels: Vec<String>,
    /// Target image width after resizing.
    pub image_size: u32,
    /// Device string (`"cpu"`, `"cuda:0"`, …).
    pub device: String,
    /// Whether to apply standard ImageNet normalisation (mean/std per channel).
    pub apply_imagenet_norm: bool,
}

impl Default for ImageClassificationConfig {
    fn default() -> Self {
        Self {
            model_name: "google/vit-base-patch16-224".to_string(),
            top_k: 5,
            labels: Vec::new(),
            image_size: 224,
            device: "cpu".to_string(),
            apply_imagenet_norm: true,
        }
    }
}

// ---------------------------------------------------------------------------
// ImageNet normalization constants
// ---------------------------------------------------------------------------

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

// ---------------------------------------------------------------------------
// ImagePreprocessor
// ---------------------------------------------------------------------------

/// Image preprocessing utilities for classification pipelines.
pub struct ImagePreprocessor;

impl ImagePreprocessor {
    /// Bicubic resize from `(src_w, src_h)` to `(dst_w, dst_h)`.
    ///
    /// Uses a 4×4 neighborhood weighted by the Mitchell-Netravali cubic kernel
    /// (a=0.5, b=0). Input/output buffers are flat RGB byte arrays (HWC).
    pub fn resize_bicubic(
        data: &[u8],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Vec<u8> {
        if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
            return Vec::new();
        }
        if src_w == dst_w && src_h == dst_h {
            return data.to_vec();
        }

        let scale_x = src_w as f32 / dst_w as f32;
        let scale_y = src_h as f32 / dst_h as f32;

        let mut out = vec![0u8; dst_w * dst_h * 3];

        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let sx_f = (dx as f32 + 0.5) * scale_x - 0.5;
                let sy_f = (dy as f32 + 0.5) * scale_y - 0.5;

                let sx_int = sx_f.floor() as isize;
                let sy_int = sy_f.floor() as isize;

                let tx = sx_f - sx_f.floor();
                let ty = sy_f - sy_f.floor();

                let mut acc = [0.0_f32; 3];
                let mut weight_sum = 0.0_f32;

                for ky in -1isize..=2isize {
                    for kx in -1isize..=2isize {
                        let px = (sx_int + kx).clamp(0, src_w as isize - 1) as usize;
                        let py = (sy_int + ky).clamp(0, src_h as isize - 1) as usize;
                        let wx = cubic_weight(kx as f32 - tx);
                        let wy = cubic_weight(ky as f32 - ty);
                        let w = wx * wy;
                        let src_base = (py * src_w + px) * 3;
                        for c in 0..3usize {
                            acc[c] += w * data.get(src_base + c).copied().unwrap_or(0) as f32;
                        }
                        weight_sum += w;
                    }
                }

                let dst_base = (dy * dst_w + dx) * 3;
                for c in 0..3usize {
                    let v = if weight_sum.abs() > 1e-6 { acc[c] / weight_sum } else { 0.0 };
                    out[dst_base + c] = v.clamp(0.0, 255.0).round() as u8;
                }
            }
        }

        out
    }

    /// Normalize RGB pixels to `f32` using ImageNet mean and std per channel.
    ///
    /// Input: flat HWC `u8` buffer (R, G, B interleaved).
    /// Output: flat HWC `f32` buffer with each channel normalized:
    /// `(pixel / 255.0 - mean[c]) / std[c]`.
    pub fn normalize_imagenet(pixels: &[u8]) -> Vec<f32> {
        let n_pixels = pixels.len() / 3;
        let mut out = Vec::with_capacity(pixels.len());
        for p in 0..n_pixels {
            for c in 0..3usize {
                let raw = pixels.get(p * 3 + c).copied().unwrap_or(0) as f32 / 255.0;
                let normalized = (raw - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                out.push(normalized);
            }
        }
        out
    }

    /// Center-crop an RGB image to `(crop_size × crop_size)`.
    ///
    /// Returns `(cropped_data, actual_width, actual_height)`.
    /// If the image is smaller than `crop_size` in either dimension, the full
    /// image is returned with its original dimensions.
    pub fn center_crop(
        data: &[u8],
        src_w: usize,
        src_h: usize,
        crop_size: usize,
    ) -> (Vec<u8>, usize, usize) {
        let cw = crop_size.min(src_w);
        let ch = crop_size.min(src_h);
        let x_start = (src_w.saturating_sub(cw)) / 2;
        let y_start = (src_h.saturating_sub(ch)) / 2;

        let mut out = Vec::with_capacity(cw * ch * 3);
        for row in y_start..(y_start + ch) {
            for col in x_start..(x_start + cw) {
                let src_base = (row * src_w + col) * 3;
                out.push(data.get(src_base).copied().unwrap_or(0));
                out.push(data.get(src_base + 1).copied().unwrap_or(0));
                out.push(data.get(src_base + 2).copied().unwrap_or(0));
            }
        }

        (out, cw, ch)
    }

    /// Convert a float image buffer from HWC to CHW layout.
    ///
    /// PyTorch / ViT models expect channels-first (C, H, W) tensors.
    /// Input shape: `[H * W * C]` in HWC order.
    /// Output shape: `[C * H * W]` in CHW order.
    pub fn to_chw_format(hwc: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
        let mut chw = vec![0.0_f32; c * h * w];
        for row in 0..h {
            for col in 0..w {
                for ch in 0..c {
                    let src_idx = (row * w + col) * c + ch;
                    let dst_idx = ch * h * w + row * w + col;
                    chw[dst_idx] = hwc.get(src_idx).copied().unwrap_or(0.0);
                }
            }
        }
        chw
    }
}

/// Cubic interpolation weight using the Catmull-Rom kernel (`a = -0.5`).
fn cubic_weight(t: f32) -> f32 {
    let t = t.abs();
    if t < 1.0 {
        1.5 * t * t * t - 2.5 * t * t + 1.0
    } else if t < 2.0 {
        -0.5 * t * t * t + 2.5 * t * t - 4.0 * t + 2.0
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Bilinear resize of a flat RGB byte buffer to `(target × target)` pixels.
fn resize_rgb_bilinear(data: &[u8], src_w: u32, src_h: u32, target: u32) -> Vec<u8> {
    if src_w == target && src_h == target {
        return data.to_vec();
    }
    let tw = target as usize;
    let th = target as usize;
    let sw = src_w as usize;
    let sh = src_h as usize;
    let mut out = vec![0u8; tw * th * 3];
    for ty in 0..th {
        for tx in 0..tw {
            let sx = (tx as f32 * sw as f32 / tw as f32) as usize;
            let sy = (ty as f32 * sh as f32 / th as f32) as usize;
            let sx = sx.min(sw - 1);
            let sy = sy.min(sh - 1);
            let src_base = (sy * sw + sx) * 3;
            let dst_base = (ty * tw + tx) * 3;
            out[dst_base] = data.get(src_base).copied().unwrap_or(0);
            out[dst_base + 1] = data.get(src_base + 1).copied().unwrap_or(0);
            out[dst_base + 2] = data.get(src_base + 2).copied().unwrap_or(0);
        }
    }
    out
}

/// Convert a raw RGB `u8` buffer to a normalised `f32` vector in `[0.0, 1.0]`.
fn normalise_rgb(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| b as f32 / 255.0).collect()
}

/// Apply softmax in-place.
fn softmax_inplace(logits: &mut Vec<f32>) {
    if logits.is_empty() {
        return;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline internals
// ---------------------------------------------------------------------------

struct ClassificationState {
    config: ImageClassificationConfig,
    labels: Vec<String>,
}

impl ClassificationState {
    fn new(config: ImageClassificationConfig) -> Self {
        let labels = if config.labels.is_empty() {
            default_imagenet_labels()
        } else {
            config.labels.clone()
        };
        Self { config, labels }
    }

    /// Preprocess raw RGB bytes into a normalised feature tensor.
    fn preprocess_rgb(&self, data: &[u8], width: u32, height: u32) -> Result<Tensor> {
        let sz = self.config.image_size;
        let resized = resize_rgb_bilinear(data, width, height, sz);
        let norm = normalise_rgb(&resized);
        // Shape: [1, 3, sz, sz]  (channels-first, as expected by ViT/CLIP)
        let sz = sz as usize;
        let expected = sz * sz * 3;
        if norm.len() != expected {
            return Err(TrustformersError::pipeline(
                format!(
                    "normalised buffer has length {} but expected {}",
                    norm.len(),
                    expected
                ),
                "image-classification",
            ));
        }
        Tensor::from_slice(&norm, &[1, 3, sz, sz])
            .map_err(|e| TrustformersError::pipeline(e.to_string(), "image-classification"))
    }

    /// Preprocess an already-normalised float tensor.
    fn preprocess_normalised(
        &self,
        values: &[f32],
        channels: u32,
        height: u32,
        width: u32,
    ) -> Result<Tensor> {
        Tensor::from_slice(
            values,
            &[1, channels as usize, height as usize, width as usize],
        )
        .map_err(|e| TrustformersError::pipeline(e.to_string(), "image-classification"))
    }

    /// Mock forward pass: produce logits proportional to mean pixel intensity
    /// per spatial region. A real pipeline would call `model.forward(features)`.
    fn mock_forward(&self, features: &Tensor) -> Result<Vec<f32>> {
        let num_labels = self.labels.len();
        if num_labels == 0 {
            return Err(TrustformersError::pipeline(
                "Label set is empty — cannot classify".to_string(),
                "image-classification",
            ));
        }
        let flat = features
            .data_f32()
            .map_err(|e| TrustformersError::pipeline(e.to_string(), "image-classification"))?;
        let mut logits = vec![0.0_f32; num_labels];
        for (i, &v) in flat.iter().enumerate() {
            logits[i % num_labels] += v;
        }
        Ok(logits)
    }

    fn run_inference(&self, features: Tensor) -> Result<Vec<ImageClassificationResult>> {
        let mut logits = self.mock_forward(&features)?;
        softmax_inplace(&mut logits);

        let mut scored: Vec<(usize, f32)> = logits.into_iter().enumerate().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = self.config.top_k.min(scored.len());
        let results = scored
            .into_iter()
            .take(top_k)
            .map(|(idx, score)| ImageClassificationResult {
                label: self.labels[idx].clone(),
                score,
                label_id: idx,
            })
            .collect();
        Ok(results)
    }
}

/// A minimal subset of ImageNet-1k class names used as the default label set.
fn default_imagenet_labels() -> Vec<String> {
    vec![
        "tench".to_string(),
        "goldfish".to_string(),
        "great_white_shark".to_string(),
        "tiger_shark".to_string(),
        "hammerhead".to_string(),
        "electric_ray".to_string(),
        "stingray".to_string(),
        "cock".to_string(),
        "hen".to_string(),
        "ostrich".to_string(),
        "brambling".to_string(),
        "goldfinch".to_string(),
        "house_finch".to_string(),
        "junco".to_string(),
        "indigo_bunting".to_string(),
        "robin".to_string(),
        "bulbul".to_string(),
        "jay".to_string(),
        "magpie".to_string(),
        "chickadee".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Public pipeline struct
// ---------------------------------------------------------------------------

/// Pipeline for image classification tasks.
///
/// Maps an [`ImageClassificationInput`] to a ranked list of
/// [`ImageClassificationResult`] values.
pub struct ImageClassificationPipeline {
    state: ClassificationState,
}

impl ImageClassificationPipeline {
    /// Create a new image classification pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError`] if `top_k` is zero or `image_size` is zero.
    pub fn new(config: ImageClassificationConfig) -> Result<Self> {
        if config.top_k == 0 {
            return Err(TrustformersError::pipeline(
                "top_k must be greater than zero".to_string(),
                "image-classification",
            ));
        }
        if config.image_size == 0 {
            return Err(TrustformersError::pipeline(
                "image_size must be greater than zero".to_string(),
                "image-classification",
            ));
        }
        Ok(Self {
            state: ClassificationState::new(config),
        })
    }

    /// Classify a single image using the new-style `ImageInput` enum.
    pub fn classify_image(&self, input: ImageInput) -> Result<Vec<ImageClassificationResult>> {
        let features = self.build_features_from_image_input(input)?;
        self.state.run_inference(features)
    }

    /// Classify a batch of images using the new-style `ImageInput` enum.
    pub fn classify_image_batch(
        &self,
        inputs: Vec<ImageInput>,
    ) -> Result<Vec<Vec<ImageClassificationResult>>> {
        inputs.into_iter().map(|inp| self.classify_image(inp)).collect()
    }

    /// Classify a single image input (legacy API).
    pub fn classify(
        &self,
        input: &ImageClassificationInput,
    ) -> Result<Vec<ImageClassificationResult>> {
        let features = self.build_features(input)?;
        self.state.run_inference(features)
    }

    /// Classify a batch of images in one call (legacy API).
    pub fn classify_batch(
        &self,
        inputs: &[ImageClassificationInput],
    ) -> Result<Vec<Vec<ImageClassificationResult>>> {
        inputs.iter().map(|inp| self.classify(inp)).collect()
    }

    /// Access the pipeline configuration.
    pub fn config(&self) -> &ImageClassificationConfig {
        &self.state.config
    }

    /// Access the resolved label set.
    pub fn labels(&self) -> &[String] {
        &self.state.labels
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn build_features_from_image_input(&self, input: ImageInput) -> Result<Tensor> {
        match input {
            ImageInput::RgbPixels {
                data,
                width,
                height,
            } => self.state.preprocess_rgb(&data, width as u32, height as u32),
            ImageInput::FloatTensor {
                data,
                width,
                height,
                channels,
            } => self.state.preprocess_normalised(
                &data,
                channels as u32,
                height as u32,
                width as u32,
            ),
            ImageInput::FilePath(path_str) => {
                let path = std::path::Path::new(&path_str);
                if !path.exists() {
                    return Err(TrustformersError::Io {
                        message: format!("Image file not found: {}", path_str),
                        path: Some(path_str),
                        suggestion: Some(
                            "Check the file path and ensure the file exists.".to_string(),
                        ),
                    });
                }
                let sz = self.state.config.image_size as usize;
                let placeholder = vec![0.0_f32; sz * sz * 3];
                Tensor::from_slice(&placeholder, &[1, 3, sz, sz])
                    .map_err(|e| TrustformersError::pipeline(e.to_string(), "image-classification"))
            },
        }
    }

    fn build_features(&self, input: &ImageClassificationInput) -> Result<Tensor> {
        match input {
            ImageClassificationInput::RgbImage {
                data,
                width,
                height,
            } => self.state.preprocess_rgb(data, *width, *height),

            ImageClassificationInput::FilePath(path) => {
                if !path.exists() {
                    return Err(TrustformersError::Io {
                        message: format!("Image file not found: {}", path.to_string_lossy()),
                        path: Some(path.to_string_lossy().into_owned()),
                        suggestion: Some(
                            "Check the file path and ensure the file exists.".to_string(),
                        ),
                    });
                }
                // Full decoding (JPEG/PNG/etc.) is not yet implemented.
                // Use a zero-pixel placeholder so the code path can be tested.
                let sz = self.state.config.image_size as usize;
                let placeholder = vec![0.0_f32; sz * sz * 3];
                tracing::debug!(
                    path = %path.to_string_lossy(),
                    "Image file decoding not yet implemented; using zero placeholder"
                );
                Tensor::from_slice(&placeholder, &[1, 3, sz, sz])
                    .map_err(|e| TrustformersError::pipeline(e.to_string(), "image-classification"))
            },

            ImageClassificationInput::NormalisedTensor {
                values,
                channels,
                height,
                width,
            } => self.state.preprocess_normalised(values, *channels, *height, *width),
        }
    }
}

// ---------------------------------------------------------------------------
// Trait impl
// ---------------------------------------------------------------------------

impl crate::pipeline::Pipeline for ImageClassificationPipeline {
    type Input = ImageClassificationInput;
    type Output = Vec<ImageClassificationResult>;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        self.classify(&input)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pipeline() -> ImageClassificationPipeline {
        ImageClassificationPipeline::new(ImageClassificationConfig::default())
            .expect("default config should be valid")
    }

    // ---- Legacy API tests (preserved) ----

    #[test]
    fn test_default_config_creates_pipeline() {
        let _p = default_pipeline();
    }

    #[test]
    fn test_classify_rgb_returns_top_k() {
        let config = ImageClassificationConfig {
            top_k: 3,
            image_size: 64,
            ..Default::default()
        };
        let pipeline = ImageClassificationPipeline::new(config).expect("valid");
        let input = ImageClassificationInput::RgbImage {
            data: vec![128u8; 64 * 64 * 3],
            width: 64,
            height: 64,
        };
        let results = pipeline.classify(&input).expect("classify ok");
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_batch_classify_length_matches() {
        let pipeline = default_pipeline();
        let inputs: Vec<ImageClassificationInput> = (0..4)
            .map(|i| ImageClassificationInput::RgbImage {
                data: vec![(i * 60) as u8; 224 * 224 * 3],
                width: 224,
                height: 224,
            })
            .collect();
        let batch = pipeline.classify_batch(&inputs).expect("batch ok");
        assert_eq!(batch.len(), 4);
    }

    #[test]
    fn test_scores_sum_to_approximately_one() {
        let config = ImageClassificationConfig {
            top_k: 20,
            image_size: 32,
            ..Default::default()
        };
        let pipeline = ImageClassificationPipeline::new(config).expect("valid");
        let input = ImageClassificationInput::RgbImage {
            data: vec![100u8; 32 * 32 * 3],
            width: 32,
            height: 32,
        };
        let results = pipeline.classify(&input).expect("ok");
        let total: f32 = results.iter().map(|r| r.score).sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "scores should sum to ~1.0, got {total}"
        );
    }

    #[test]
    fn test_missing_file_returns_error() {
        let pipeline = default_pipeline();
        let tmp = std::env::temp_dir().join("image_classification_nonexistent.jpg");
        let _ = std::fs::remove_file(&tmp);
        let input = ImageClassificationInput::FilePath(tmp);
        let result = pipeline.classify(&input);
        assert!(result.is_err(), "should fail for non-existent file");
    }

    #[test]
    fn test_normalised_tensor_input() {
        let pipeline = default_pipeline();
        let sz = pipeline.config().image_size as usize;
        let input = ImageClassificationInput::NormalisedTensor {
            values: vec![0.5_f32; 3 * sz * sz],
            channels: 3,
            height: sz as u32,
            width: sz as u32,
        };
        let results = pipeline.classify(&input).expect("ok");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_custom_labels_respected() {
        let config = ImageClassificationConfig {
            labels: vec!["cat".to_string(), "dog".to_string()],
            top_k: 2,
            image_size: 32,
            ..Default::default()
        };
        let pipeline = ImageClassificationPipeline::new(config).expect("valid");
        let input = ImageClassificationInput::RgbImage {
            data: vec![50u8; 32 * 32 * 3],
            width: 32,
            height: 32,
        };
        let results = pipeline.classify(&input).expect("ok");
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(["cat", "dog"].contains(&r.label.as_str()));
        }
    }

    #[test]
    fn test_zero_top_k_is_rejected() {
        let config = ImageClassificationConfig {
            top_k: 0,
            ..Default::default()
        };
        assert!(
            ImageClassificationPipeline::new(config).is_err(),
            "top_k=0 should be rejected"
        );
    }

    #[test]
    fn test_zero_image_size_is_rejected() {
        let config = ImageClassificationConfig {
            image_size: 0,
            ..Default::default()
        };
        assert!(
            ImageClassificationPipeline::new(config).is_err(),
            "image_size=0 should be rejected"
        );
    }

    #[test]
    fn test_existing_file_with_placeholder_succeeds() {
        let tmp = std::env::temp_dir().join("image_classification_test.jpg");
        std::fs::write(&tmp, b"").expect("write temp file");
        let pipeline = default_pipeline();
        let input = ImageClassificationInput::FilePath(tmp.clone());
        let result = pipeline.classify(&input);
        let _ = std::fs::remove_file(&tmp);
        assert!(result.is_ok(), "should succeed for existing file path");
    }

    #[test]
    fn test_resize_bilinear_preserves_size_when_already_target() {
        let data = vec![255u8; 4 * 4 * 3];
        let out = resize_rgb_bilinear(&data, 4, 4, 4);
        assert_eq!(out.len(), data.len());
    }

    // ---- New ImagePreprocessor tests ----

    #[test]
    fn test_resize_bicubic_output_dimensions() {
        let src = vec![128u8; 8 * 8 * 3];
        let out = ImagePreprocessor::resize_bicubic(&src, 8, 8, 4, 4);
        assert_eq!(out.len(), 4 * 4 * 3, "output should be 4×4×3 bytes");
    }

    #[test]
    fn test_resize_bicubic_same_size_returns_same() {
        let src = vec![100u8; 6 * 6 * 3];
        let out = ImagePreprocessor::resize_bicubic(&src, 6, 6, 6, 6);
        assert_eq!(out, src, "same-size bicubic should be identity");
    }

    #[test]
    fn test_resize_bicubic_upscale_dimensions() {
        let src = vec![200u8; 4 * 4 * 3];
        let out = ImagePreprocessor::resize_bicubic(&src, 4, 4, 8, 8);
        assert_eq!(out.len(), 8 * 8 * 3, "upscaled output dimensions wrong");
    }

    #[test]
    fn test_normalize_imagenet_range() {
        // All-128 image: pixel=0.502, normalized for R: (0.502-0.485)/0.229 ≈ 0.074
        let pixels = vec![128u8; 4 * 3]; // 4 pixels
        let out = ImagePreprocessor::normalize_imagenet(&pixels);
        assert_eq!(out.len(), 12, "output length should match input length");
        // Values should be small (near 0) for near-mean pixels
        for v in &out {
            assert!(
                v.abs() < 3.0,
                "normalized value {} is unexpectedly large",
                v
            );
        }
    }

    #[test]
    fn test_normalize_imagenet_black_pixel() {
        // Black pixel (0): all channels should be negative after normalization
        let pixels = vec![0u8; 3];
        let out = ImagePreprocessor::normalize_imagenet(&pixels);
        // (0/255 - mean[c]) / std[c] < 0 for all c since mean > 0
        for v in &out {
            assert!(
                *v < 0.0,
                "black pixel should normalize to negative value, got {}",
                v
            );
        }
    }

    #[test]
    fn test_normalize_imagenet_white_pixel() {
        // White pixel (255): all channels should be positive after normalization
        let pixels = vec![255u8; 3];
        let out = ImagePreprocessor::normalize_imagenet(&pixels);
        // (1.0 - mean[c]) / std[c] > 0 since mean < 1.0
        for v in &out {
            assert!(
                *v > 0.0,
                "white pixel should normalize to positive value, got {}",
                v
            );
        }
    }

    #[test]
    fn test_center_crop_output_dimensions() {
        let src = vec![128u8; 10 * 10 * 3];
        let (out, w, h) = ImagePreprocessor::center_crop(&src, 10, 10, 6);
        assert_eq!(w, 6, "cropped width should be 6");
        assert_eq!(h, 6, "cropped height should be 6");
        assert_eq!(out.len(), 6 * 6 * 3, "cropped data length wrong");
    }

    #[test]
    fn test_center_crop_smaller_than_crop_size() {
        let src = vec![200u8; 4 * 4 * 3];
        let (out, w, h) = ImagePreprocessor::center_crop(&src, 4, 4, 8);
        // Crop size > image size: should return the full 4×4 image
        assert_eq!(w, 4);
        assert_eq!(h, 4);
        assert_eq!(out.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_to_chw_format_shape() {
        // HWC: 2×2×3
        let hwc = vec![
            1.0_f32, 2.0, 3.0, // pixel (0,0): R=1, G=2, B=3
            4.0, 5.0, 6.0, // pixel (0,1): R=4, G=5, B=6
            7.0, 8.0, 9.0, // pixel (1,0): R=7, G=8, B=9
            10.0, 11.0, 12.0, // pixel (1,1): R=10, G=11, B=12
        ];
        let chw = ImagePreprocessor::to_chw_format(&hwc, 2, 2, 3);
        assert_eq!(chw.len(), 12, "CHW output length should be 12");
        // Channel 0 (R): [1, 4, 7, 10]
        assert!(
            (chw[0] - 1.0).abs() < 1e-6,
            "CHW[0] (R@00) should be 1.0, got {}",
            chw[0]
        );
        assert!(
            (chw[1] - 4.0).abs() < 1e-6,
            "CHW[1] (R@01) should be 4.0, got {}",
            chw[1]
        );
        // Channel 1 (G): [2, 5, 8, 11]
        assert!(
            (chw[4] - 2.0).abs() < 1e-6,
            "CHW[4] (G@00) should be 2.0, got {}",
            chw[4]
        );
    }

    #[test]
    fn test_to_chw_format_roundtrip_values() {
        // Single pixel, 3 channels
        let hwc = vec![0.1_f32, 0.2, 0.3];
        let chw = ImagePreprocessor::to_chw_format(&hwc, 1, 1, 3);
        assert_eq!(chw.len(), 3);
        assert!((chw[0] - 0.1).abs() < 1e-6);
        assert!((chw[1] - 0.2).abs() < 1e-6);
        assert!((chw[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_new_style_classify_rgb_pixels() {
        let pipeline = default_pipeline();
        let sz = pipeline.config().image_size as usize;
        let input = ImageInput::RgbPixels {
            data: vec![100u8; sz * sz * 3],
            width: sz,
            height: sz,
        };
        let results = pipeline.classify_image(input).expect("classify_image ok");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_new_style_classify_float_tensor() {
        let pipeline = default_pipeline();
        let sz = pipeline.config().image_size as usize;
        let input = ImageInput::FloatTensor {
            data: vec![0.5_f32; sz * sz * 3],
            width: sz,
            height: sz,
            channels: 3,
        };
        let results = pipeline.classify_image(input).expect("classify_image ok");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_new_style_batch_classify() {
        let pipeline = default_pipeline();
        let sz = pipeline.config().image_size as usize;
        let inputs = vec![
            ImageInput::RgbPixels {
                data: vec![0u8; sz * sz * 3],
                width: sz,
                height: sz,
            },
            ImageInput::RgbPixels {
                data: vec![128u8; sz * sz * 3],
                width: sz,
                height: sz,
            },
        ];
        let batch = pipeline.classify_image_batch(inputs).expect("batch ok");
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_result_has_label_id() {
        let config = ImageClassificationConfig {
            top_k: 5,
            image_size: 32,
            ..Default::default()
        };
        let pipeline = ImageClassificationPipeline::new(config).expect("valid");
        let input = ImageClassificationInput::RgbImage {
            data: vec![50u8; 32 * 32 * 3],
            width: 32,
            height: 32,
        };
        let results = pipeline.classify(&input).expect("ok");
        let label_count = pipeline.labels().len();
        for r in &results {
            assert!(
                r.label_id < label_count,
                "label_id {} out of bounds (max {})",
                r.label_id,
                label_count
            );
        }
    }

    #[test]
    fn test_results_sorted_descending() {
        let config = ImageClassificationConfig {
            top_k: 5,
            image_size: 32,
            ..Default::default()
        };
        let pipeline = ImageClassificationPipeline::new(config).expect("valid");
        let input = ImageClassificationInput::RgbImage {
            data: vec![200u8; 32 * 32 * 3],
            width: 32,
            height: 32,
        };
        let results = pipeline.classify(&input).expect("ok");
        for window in results.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "results should be sorted by descending score: {} < {}",
                window[0].score,
                window[1].score
            );
        }
    }
}
