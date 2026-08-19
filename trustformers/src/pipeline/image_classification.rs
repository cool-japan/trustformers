//! # Image Classification Pipeline
//!
//! Maps an image (raw RGB bytes, a file path, or a pre-computed feature vector)
//! to a ranked list of category labels with associated probabilities.
//!
//! ## What is real here
//!
//! * **Decoding** — binary Netpbm always, and every format the pure-Rust
//!   `image` crate supports when the `vision` feature is on. A file that cannot
//!   be decoded produces a structured error; it is never replaced by a black
//!   image.
//! * **Preprocessing** — real bilinear resize with half-pixel centres, real
//!   bicubic resize (Catmull-Rom), centre-crop, ImageNet/CLIP normalisation and
//!   HWC → CHW conversion.
//! * **Post-processing** — softmax and top-k ranking over model logits.
//!
//! ## Model support
//!
//! Inference runs only when a real vision backbone is attached:
//!
//! * with the `vit` feature, [`ImageClassificationPipeline::with_vit`] accepts a
//!   caller-constructed [`trustformers_models::vit::ViTForImageClassification`]
//!   (weights are the caller's responsibility) and the pipeline runs its real
//!   forward pass;
//! * otherwise [`ImageClassificationPipeline::classify`] returns a structured
//!   [`TrustformersError::FeatureUnavailable`] naming the supported
//!   architectures.
//!
//! Nothing in this module fabricates labels, scores or timings.
//!
//! ## Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::image_classification::{
//!     ImageClassificationConfig, ImageClassificationPipeline, ImageClassificationInput,
//! };
//!
//! let pipeline = ImageClassificationPipeline::new(ImageClassificationConfig::default())?;
//!
//! let input = ImageClassificationInput::RgbImage {
//!     data: vec![128u8; 224 * 224 * 3],
//!     width: 224,
//!     height: 224,
//! };
//!
//! // Real preprocessing, always available:
//! let features = pipeline.preprocess(&input)?;
//! // Classification needs a real backbone:
//! assert!(pipeline.classify(&input).is_err());
//! # Ok::<(), trustformers::TrustformersError>(())
//! ```

use crate::error::{Result, TrustformersError};
use crate::pipeline::media::image_proc;
use crate::pipeline::media::unsupported_model;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
#[cfg(feature = "vit")]
use std::sync::Arc;
use trustformers_core::tensor::Tensor;

/// Architectures this pipeline can actually execute.
const SUPPORTED_ARCHITECTURES: &[&str] = &["vit (requires the `vit` feature)"];

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
///
/// Real bilinear interpolation with half-pixel centres — the previous
/// implementation of this function sampled the nearest source pixel despite its
/// name, which produced visibly aliased inputs to the model.
///
/// # Errors
///
/// Returns an error when the buffer does not match `src_w × src_h × 3` or any
/// dimension is zero.
///
/// Not currently called from production code (`ClassificationState::preprocess_rgb`
/// goes straight from bytes to a normalised `Tensor` via `preprocess_image`,
/// never materializing a resized byte buffer) — kept `#[cfg(test)]` as a
/// correctness check on the underlying `image_proc::resize_bilinear` this
/// module depends on.
#[cfg(test)]
fn resize_rgb_bilinear(data: &[u8], src_w: u32, src_h: u32, target: u32) -> Result<Vec<u8>> {
    let image = bytes_to_rgb_image(data, src_w, src_h)?;
    let resized = image_proc::resize_bilinear(&image, target as usize, target as usize)?;
    Ok(resized
        .data
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect())
}

/// Wrap a flat HWC RGB byte buffer as an [`image_proc::RgbImage`].
fn bytes_to_rgb_image(data: &[u8], width: u32, height: u32) -> Result<image_proc::RgbImage> {
    let expected = width as usize * height as usize * 3;
    if data.len() != expected {
        return Err(TrustformersError::pipeline(
            format!(
                "RGB buffer has {} bytes but {width}x{height}x3 = {expected} were expected",
                data.len()
            ),
            "image-classification",
        ));
    }
    let floats: Vec<f32> = data.iter().map(|&b| f32::from(b) / 255.0).collect();
    image_proc::RgbImage::new(floats, height as usize, width as usize)
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

    /// Preprocess an [`image_proc::RgbImage`] into a `[1, 3, size, size]` tensor.
    ///
    /// Real bilinear resize followed (when `apply_imagenet_norm` is set) by
    /// per-channel ImageNet normalisation, laid out channels-first as ViT and
    /// CLIP expect.
    fn preprocess_image(&self, image: &image_proc::RgbImage) -> Result<Tensor> {
        let sz = self.config.image_size as usize;
        let resized = image_proc::resize_bilinear(image, sz, sz)?;
        let chw = if self.config.apply_imagenet_norm {
            image_proc::normalize_to_chw(&resized, IMAGENET_MEAN, IMAGENET_STD)?
        } else {
            image_proc::normalize_to_chw(&resized, [0.0; 3], [1.0; 3])?
        };
        Tensor::from_slice(&chw, &[1, 3, sz, sz])
            .map_err(|e| TrustformersError::pipeline(e.to_string(), "image-classification"))
    }

    /// Preprocess raw RGB bytes into a normalised feature tensor.
    fn preprocess_rgb(&self, data: &[u8], width: u32, height: u32) -> Result<Tensor> {
        let image = bytes_to_rgb_image(data, width, height)?;
        self.preprocess_image(&image)
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

    /// Rank raw model logits into the pipeline's top-k label list.
    fn rank_logits(&self, logits: &[f32]) -> Result<Vec<ImageClassificationResult>> {
        if self.labels.is_empty() {
            return Err(TrustformersError::pipeline(
                "Label set is empty — cannot classify".to_string(),
                "image-classification",
            ));
        }
        if logits.len() != self.labels.len() {
            return Err(TrustformersError::pipeline(
                format!(
                    "model produced {} logits but the label set has {} entries",
                    logits.len(),
                    self.labels.len()
                ),
                "image-classification",
            ));
        }
        let mut probabilities = logits.to_vec();
        softmax_inplace(&mut probabilities);

        let mut scored: Vec<(usize, f32)> = probabilities.into_iter().enumerate().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = self.config.top_k.min(scored.len());
        Ok(scored
            .into_iter()
            .take(top_k)
            .map(|(idx, score)| ImageClassificationResult {
                label: self.labels[idx].clone(),
                score,
                label_id: idx,
            })
            .collect())
    }
}

/// The inference backend attached to an [`ImageClassificationPipeline`].
#[derive(Clone)]
pub enum ImageClassificationBackend {
    /// No vision backbone attached — classification reports the architecture as
    /// unsupported instead of returning fabricated labels.
    Unavailable,
    /// A caller-supplied Vision Transformer classifier.
    #[cfg(feature = "vit")]
    Vit(Arc<trustformers_models::vit::ViTForImageClassification>),
}

impl std::fmt::Debug for ImageClassificationBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable => f.write_str("ImageClassificationBackend::Unavailable"),
            #[cfg(feature = "vit")]
            Self::Vit(_) => f.write_str("ImageClassificationBackend::Vit"),
        }
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
    backend: ImageClassificationBackend,
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
            backend: ImageClassificationBackend::Unavailable,
        })
    }

    /// Attach a caller-supplied Vision Transformer classifier as the backend.
    ///
    /// Weight provenance is the caller's responsibility: the pipeline runs the
    /// model's real forward pass on real preprocessed pixels.
    #[cfg(feature = "vit")]
    pub fn with_vit(
        mut self,
        model: Arc<trustformers_models::vit::ViTForImageClassification>,
    ) -> Self {
        self.backend = ImageClassificationBackend::Vit(model);
        self
    }

    /// The currently attached backend.
    pub fn backend(&self) -> &ImageClassificationBackend {
        &self.backend
    }

    /// Preprocess an input into the `[1, 3, size, size]` tensor a vision
    /// backbone expects.
    ///
    /// This is the pipeline's real, reusable front-end: decoding, bilinear
    /// resize, normalisation and CHW layout. It works with or without a
    /// backbone attached.
    pub fn preprocess(&self, input: &ImageClassificationInput) -> Result<Tensor> {
        self.build_features(input)
    }

    /// Rank raw model logits with the pipeline's label set and `top_k`.
    ///
    /// # Errors
    ///
    /// Returns an error when the logit count does not match the label set.
    pub fn rank_logits(&self, logits: &[f32]) -> Result<Vec<ImageClassificationResult>> {
        self.state.rank_logits(logits)
    }

    /// Classify a single image using the new-style `ImageInput` enum.
    pub fn classify_image(&self, input: ImageInput) -> Result<Vec<ImageClassificationResult>> {
        let features = self.build_features_from_image_input(input)?;
        self.run_inference(&features)
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
        self.run_inference(&features)
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

    /// Decode an image file into the pipeline's feature tensor.
    ///
    /// # Errors
    ///
    /// [`TrustformersError::Io`] for a missing/unreadable file, and
    /// [`TrustformersError::FeatureUnavailable`] when the format needs the
    /// `vision` feature. Never a blank image.
    fn decode_file(&self, path: &Path) -> Result<Tensor> {
        if !path.exists() {
            return Err(TrustformersError::Io {
                message: format!("Image file not found: {}", path.to_string_lossy()),
                path: Some(path.to_string_lossy().into_owned()),
                suggestion: Some("Check the file path and ensure the file exists.".to_string()),
            });
        }
        let image = image_proc::decode_image_file(path)?;
        self.state.preprocess_image(&image)
    }

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
            ImageInput::FilePath(path_str) => self.decode_file(Path::new(&path_str)),
        }
    }

    fn build_features(&self, input: &ImageClassificationInput) -> Result<Tensor> {
        match input {
            ImageClassificationInput::RgbImage {
                data,
                width,
                height,
            } => self.state.preprocess_rgb(data, *width, *height),

            ImageClassificationInput::FilePath(path) => self.decode_file(path),

            ImageClassificationInput::NormalisedTensor {
                values,
                channels,
                height,
                width,
            } => self.state.preprocess_normalised(values, *channels, *height, *width),
        }
    }

    /// Run the attached backbone on real preprocessed features.
    fn run_inference(&self, features: &Tensor) -> Result<Vec<ImageClassificationResult>> {
        match &self.backend {
            ImageClassificationBackend::Unavailable => {
                tracing::trace!(
                    feature_shape = ?features.shape(),
                    "no vision backbone attached; refusing rather than fabricating labels"
                );
                Err(unsupported_model(
                    "image-classification",
                    &self.state.config.model_name,
                    SUPPORTED_ARCHITECTURES,
                ))
            },
            #[cfg(feature = "vit")]
            ImageClassificationBackend::Vit(model) => {
                let logits = run_vit(model, features)?;
                self.state.rank_logits(&logits)
            },
        }
    }
}

/// Run a Vision Transformer classifier over a `[1, 3, H, W]` feature tensor.
#[cfg(feature = "vit")]
fn run_vit(
    model: &trustformers_models::vit::ViTForImageClassification,
    features: &Tensor,
) -> Result<Vec<f32>> {
    use scirs2_core::ndarray::Array4;

    let shape = features.shape().to_vec();
    if shape.len() != 4 {
        return Err(TrustformersError::pipeline(
            format!("expected a [batch, 3, H, W] feature tensor, got {shape:?}"),
            "image-classification",
        ));
    }
    let flat = features
        .to_vec_f32()
        .map_err(|e| TrustformersError::pipeline(e.to_string(), "image-classification"))?;
    let images = Array4::from_shape_vec((shape[0], shape[1], shape[2], shape[3]), flat)
        .map_err(|e| TrustformersError::pipeline(e.to_string(), "image-classification"))?;

    let logits = model
        .forward(&images)
        .map_err(|e| TrustformersError::model(e.to_string(), "vit"))?;
    Ok(logits.iter().copied().collect())
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
    use crate::pipeline::media::image_proc::{encode_ppm, RgbImage};

    fn default_pipeline() -> ImageClassificationPipeline {
        ImageClassificationPipeline::new(ImageClassificationConfig::default())
            .expect("default config should be valid")
    }

    fn gradient_bytes(h: usize, w: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(h * w * 3);
        for y in 0..h {
            for x in 0..w {
                data.push((x * 255 / w.max(1)) as u8);
                data.push((y * 255 / h.max(1)) as u8);
                data.push(64u8);
            }
        }
        data
    }

    fn assert_unsupported(err: &TrustformersError) {
        match err {
            TrustformersError::FeatureUnavailable { message, .. } => {
                assert!(
                    message.contains("no real model implementation"),
                    "unexpected message: {message}"
                );
            },
            other => panic!("expected FeatureUnavailable, got {other:?}"),
        }
    }

    // ---- Construction ----

    #[test]
    fn test_default_config_creates_pipeline() {
        let _p = default_pipeline();
    }

    #[test]
    fn test_zero_top_k_is_rejected() {
        let config = ImageClassificationConfig {
            top_k: 0,
            ..Default::default()
        };
        assert!(ImageClassificationPipeline::new(config).is_err());
    }

    #[test]
    fn test_zero_image_size_is_rejected() {
        let config = ImageClassificationConfig {
            image_size: 0,
            ..Default::default()
        };
        assert!(ImageClassificationPipeline::new(config).is_err());
    }

    #[test]
    fn test_backend_defaults_to_unavailable() {
        let pipeline = default_pipeline();
        assert_eq!(
            format!("{:?}", pipeline.backend()),
            "ImageClassificationBackend::Unavailable"
        );
    }

    // ---- Honesty: no fabricated classification ----

    #[test]
    fn test_classify_rgb_reports_unsupported_model() {
        // Regression: `classify` used to return confident ImageNet labels from
        // `mock_forward` without ever loading a model.
        let config = ImageClassificationConfig {
            top_k: 3,
            image_size: 64,
            ..Default::default()
        };
        let pipeline = ImageClassificationPipeline::new(config).expect("valid");
        let input = ImageClassificationInput::RgbImage {
            data: gradient_bytes(64, 64),
            width: 64,
            height: 64,
        };
        let err = pipeline.classify(&input).expect_err("no backbone attached");
        assert_unsupported(&err);
    }

    #[test]
    fn test_batch_classify_reports_unsupported_model() {
        let pipeline = default_pipeline();
        let inputs: Vec<ImageClassificationInput> = (0..2)
            .map(|_| ImageClassificationInput::RgbImage {
                data: gradient_bytes(224, 224),
                width: 224,
                height: 224,
            })
            .collect();
        let err = pipeline.classify_batch(&inputs).expect_err("no backbone attached");
        assert_unsupported(&err);
    }

    #[test]
    fn test_normalised_tensor_input_reports_unsupported_model() {
        let pipeline = default_pipeline();
        let sz = pipeline.config().image_size as usize;
        let input = ImageClassificationInput::NormalisedTensor {
            values: vec![0.5_f32; 3 * sz * sz],
            channels: 3,
            height: sz as u32,
            width: sz as u32,
        };
        let err = pipeline.classify(&input).expect_err("no backbone attached");
        assert_unsupported(&err);
    }

    #[test]
    fn test_new_style_classify_reports_unsupported_model() {
        let pipeline = default_pipeline();
        let sz = pipeline.config().image_size as usize;
        let input = ImageInput::RgbPixels {
            data: gradient_bytes(sz, sz),
            width: sz,
            height: sz,
        };
        let err = pipeline.classify_image(input).expect_err("no backbone attached");
        assert_unsupported(&err);
    }

    #[test]
    fn test_unsupported_error_names_the_requested_model() {
        let config = ImageClassificationConfig {
            model_name: "microsoft/resnet-50".to_string(),
            image_size: 32,
            ..Default::default()
        };
        let pipeline = ImageClassificationPipeline::new(config).expect("valid");
        let input = ImageClassificationInput::RgbImage {
            data: gradient_bytes(32, 32),
            width: 32,
            height: 32,
        };
        let err = pipeline.classify(&input).expect_err("no backbone");
        assert!(
            err.to_string().contains("microsoft/resnet-50"),
            "err: {err}"
        );
        assert!(err.to_string().contains("vit"), "err: {err}");
    }

    // ---- Real preprocessing ----

    #[test]
    fn test_preprocess_produces_nonzero_chw_tensor() {
        let config = ImageClassificationConfig {
            image_size: 8,
            ..Default::default()
        };
        let pipeline = ImageClassificationPipeline::new(config).expect("valid");
        let input = ImageClassificationInput::RgbImage {
            data: gradient_bytes(16, 16),
            width: 16,
            height: 16,
        };
        let tensor = pipeline.preprocess(&input).expect("preprocess");
        assert_eq!(tensor.shape(), vec![1, 3, 8, 8]);
        let values = tensor.to_vec_f32().expect("values");
        assert!(
            values.iter().any(|&v| v != 0.0),
            "features must not be all zeros"
        );
        // Channel-major layout: the red plane varies along x, the green plane along y.
        let plane = 8 * 8;
        assert!(
            (values[1] - values[0]).abs() > 1e-3,
            "red plane must vary along x"
        );
        assert!(
            (values[plane + 8] - values[plane]).abs() > 1e-3,
            "green plane must vary along y"
        );
    }

    #[test]
    fn test_preprocess_distinguishes_images() {
        let config = ImageClassificationConfig {
            image_size: 8,
            ..Default::default()
        };
        let pipeline = ImageClassificationPipeline::new(config).expect("valid");
        let a = pipeline
            .preprocess(&ImageClassificationInput::RgbImage {
                data: gradient_bytes(16, 16),
                width: 16,
                height: 16,
            })
            .expect("a")
            .to_vec_f32()
            .expect("a values");
        let mut reversed = gradient_bytes(16, 16);
        reversed.reverse();
        let b = pipeline
            .preprocess(&ImageClassificationInput::RgbImage {
                data: reversed,
                width: 16,
                height: 16,
            })
            .expect("b")
            .to_vec_f32()
            .expect("b values");
        assert_ne!(a, b, "different images must preprocess differently");
    }

    #[test]
    fn test_preprocess_rejects_wrong_buffer_length() {
        let pipeline = default_pipeline();
        let input = ImageClassificationInput::RgbImage {
            data: vec![0u8; 10],
            width: 16,
            height: 16,
        };
        assert!(pipeline.preprocess(&input).is_err());
    }

    // ---- File handling ----

    #[test]
    fn test_missing_file_returns_error() {
        let pipeline = default_pipeline();
        let tmp = std::env::temp_dir().join("image_classification_nonexistent.ppm");
        let _ = std::fs::remove_file(&tmp);
        let input = ImageClassificationInput::FilePath(tmp);
        assert!(matches!(
            pipeline.classify(&input),
            Err(TrustformersError::Io { .. })
        ));
    }

    #[test]
    fn test_existing_undecodable_file_is_rejected_not_treated_as_black() {
        // Regression: an existing file used to be replaced by an all-zero
        // (black) tensor and classified with confident labels.
        let tmp = std::env::temp_dir().join("image_classification_not_an_image.jpg");
        std::fs::write(&tmp, b"").expect("write temp file");
        let pipeline = default_pipeline();
        let result = pipeline.preprocess(&ImageClassificationInput::FilePath(tmp.clone()));
        let _ = std::fs::remove_file(&tmp);
        assert!(
            result.is_err(),
            "an empty file must not decode to a black image"
        );
    }

    #[test]
    fn test_real_image_file_decodes_to_real_pixels() {
        let tmp = std::env::temp_dir().join("image_classification_real.ppm");
        let image = RgbImage::new(
            gradient_bytes(12, 12).iter().map(|&b| f32::from(b) / 255.0).collect(),
            12,
            12,
        )
        .expect("fixture");
        std::fs::write(&tmp, encode_ppm(&image)).expect("write fixture");
        let config = ImageClassificationConfig {
            image_size: 8,
            ..Default::default()
        };
        let pipeline = ImageClassificationPipeline::new(config).expect("valid");
        let tensor = pipeline
            .preprocess(&ImageClassificationInput::FilePath(tmp.clone()))
            .expect("decode + preprocess");
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(tensor.shape(), vec![1, 3, 8, 8]);
        let values = tensor.to_vec_f32().expect("values");
        assert!(values.iter().any(|&v| v != 0.0));
    }

    // ---- Post-processing (real, reusable) ----

    #[test]
    fn test_rank_logits_orders_and_normalises() {
        let pipeline = default_pipeline();
        let mut logits = vec![0.0f32; pipeline.labels().len()];
        logits[3] = 6.0;
        let ranked = pipeline.rank_logits(&logits).expect("rank");
        assert_eq!(ranked.len(), 5);
        assert_eq!(ranked[0].label_id, 3);
        for window in ranked.windows(2) {
            assert!(window[0].score >= window[1].score);
        }
        for r in &ranked {
            assert!(r.label_id < pipeline.labels().len());
        }
    }

    #[test]
    fn test_rank_logits_rejects_length_mismatch() {
        let pipeline = default_pipeline();
        assert!(pipeline.rank_logits(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_rank_logits_scores_sum_to_one() {
        let config = ImageClassificationConfig {
            top_k: 20,
            ..Default::default()
        };
        let pipeline = ImageClassificationPipeline::new(config).expect("valid");
        let logits: Vec<f32> = (0..pipeline.labels().len()).map(|i| i as f32 * 0.1).collect();
        let ranked = pipeline.rank_logits(&logits).expect("rank");
        let total: f32 = ranked.iter().map(|r| r.score).sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "scores should sum to ~1.0, got {total}"
        );
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
        let ranked = pipeline.rank_logits(&[0.1, 4.0]).expect("rank");
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].label, "dog");
    }

    // ---- Resize helpers ----

    #[test]
    fn test_resize_bilinear_preserves_size_when_already_target() {
        let data = gradient_bytes(4, 4);
        let out = resize_rgb_bilinear(&data, 4, 4, 4).expect("resize");
        assert_eq!(out, data);
    }

    #[test]
    fn test_resize_bilinear_interpolates_not_nearest() {
        // 2x2 red ramp: 0, 255 / 0, 255. Upsampling to 4 px must produce
        // intermediate values that nearest-neighbour sampling never yields.
        let data: Vec<u8> = vec![
            0, 0, 0, 255, 0, 0, // row 0
            0, 0, 0, 255, 0, 0, // row 1
        ];
        let out = resize_rgb_bilinear(&data, 2, 2, 4).expect("resize");
        assert_eq!(out.len(), 4 * 4 * 3);
        let reds: Vec<u8> = out.chunks_exact(3).map(|p| p[0]).collect();
        assert!(
            reds.iter().any(|&r| r > 5 && r < 250),
            "bilinear resize must produce intermediate values, got {reds:?}"
        );
    }

    #[test]
    fn test_resize_bilinear_rejects_bad_buffer() {
        assert!(resize_rgb_bilinear(&[0u8; 5], 4, 4, 4).is_err());
    }

    // ---- ImagePreprocessor ----

    #[test]
    fn test_resize_bicubic_output_dimensions() {
        let src = vec![128u8; 8 * 8 * 3];
        let out = ImagePreprocessor::resize_bicubic(&src, 8, 8, 4, 4);
        assert_eq!(out.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_resize_bicubic_same_size_returns_same() {
        let src = vec![100u8; 6 * 6 * 3];
        assert_eq!(ImagePreprocessor::resize_bicubic(&src, 6, 6, 6, 6), src);
    }

    #[test]
    fn test_resize_bicubic_upscale_dimensions() {
        let src = vec![200u8; 4 * 4 * 3];
        let out = ImagePreprocessor::resize_bicubic(&src, 4, 4, 8, 8);
        assert_eq!(out.len(), 8 * 8 * 3);
    }

    #[test]
    fn test_normalize_imagenet_range() {
        let pixels = vec![128u8; 4 * 3];
        let out = ImagePreprocessor::normalize_imagenet(&pixels);
        assert_eq!(out.len(), 12);
        for v in &out {
            assert!(v.abs() < 3.0, "normalized value {v} is unexpectedly large");
        }
    }

    #[test]
    fn test_normalize_imagenet_black_pixel() {
        let out = ImagePreprocessor::normalize_imagenet(&[0u8; 3]);
        for v in &out {
            assert!(*v < 0.0, "black pixel should normalize negative, got {v}");
        }
    }

    #[test]
    fn test_normalize_imagenet_white_pixel() {
        let out = ImagePreprocessor::normalize_imagenet(&[255u8; 3]);
        for v in &out {
            assert!(*v > 0.0, "white pixel should normalize positive, got {v}");
        }
    }

    #[test]
    fn test_center_crop_output_dimensions() {
        let src = vec![128u8; 10 * 10 * 3];
        let (out, w, h) = ImagePreprocessor::center_crop(&src, 10, 10, 6);
        assert_eq!((w, h), (6, 6));
        assert_eq!(out.len(), 6 * 6 * 3);
    }

    #[test]
    fn test_center_crop_smaller_than_crop_size() {
        let src = vec![200u8; 4 * 4 * 3];
        let (out, w, h) = ImagePreprocessor::center_crop(&src, 4, 4, 8);
        assert_eq!((w, h), (4, 4));
        assert_eq!(out.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_to_chw_format_shape() {
        let hwc = vec![
            1.0_f32, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
            10.0, 11.0, 12.0,
        ];
        let chw = ImagePreprocessor::to_chw_format(&hwc, 2, 2, 3);
        assert_eq!(chw.len(), 12);
        assert!((chw[0] - 1.0).abs() < 1e-6);
        assert!((chw[1] - 4.0).abs() < 1e-6);
        assert!((chw[4] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_to_chw_format_roundtrip_values() {
        let chw = ImagePreprocessor::to_chw_format(&[0.1_f32, 0.2, 0.3], 1, 1, 3);
        assert_eq!(chw.len(), 3);
        assert!((chw[0] - 0.1).abs() < 1e-6);
        assert!((chw[1] - 0.2).abs() < 1e-6);
        assert!((chw[2] - 0.3).abs() < 1e-6);
    }
}
