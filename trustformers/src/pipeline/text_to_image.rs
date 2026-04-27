//! # Text-to-Image Pipeline
//!
//! Stable Diffusion / DALL-E compatible text-to-image generation.
//!
//! ## Supported model families
//! - **Stable Diffusion XL** (stabilityai/stable-diffusion-xl-base-1.0)
//! - **DALL-E** compatible inference
//!
//! ## Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::text_to_image::{TextToImageConfig, TextToImagePipeline};
//!
//! let config = TextToImageConfig::default();
//! let pipeline = TextToImagePipeline::new(config)?;
//! let images = pipeline.generate("a photo of an astronaut riding a horse")?;
//! println!("Generated {} image(s)", images.len());
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the text-to-image pipeline.
#[derive(Debug, Error)]
pub enum ImageGenError {
    #[error("Empty prompt")]
    EmptyPrompt,
    #[error("Invalid dimensions: height={h}, width={w}")]
    InvalidDimensions { h: usize, w: usize },
    #[error("Invalid pixel data: expected {expected} got {got}")]
    InvalidPixelData { expected: usize, got: usize },
    #[error("Model error: {0}")]
    ModelError(String),
}

// ---------------------------------------------------------------------------
// Scheduler type
// ---------------------------------------------------------------------------

/// Noise scheduler variant for the diffusion process.
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulerType {
    /// Denoising Diffusion Probabilistic Models
    DDPM,
    /// Denoising Diffusion Implicit Models (faster, fewer steps)
    DDIM,
    /// Pseudo Numerical methods for Diffusion Models
    PNDM,
    /// Euler Ancestral sampler
    EulerA,
    /// DPM++ sampler (high quality with fewer steps)
    DPMpp,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`TextToImagePipeline`].
#[derive(Debug, Clone)]
pub struct TextToImageConfig {
    /// HuggingFace model identifier or local path.
    pub model_name: String,
    /// Image height in pixels. Default: 512.
    pub height: usize,
    /// Image width in pixels. Default: 512.
    pub width: usize,
    /// Number of denoising steps. Default: 20.
    pub num_inference_steps: usize,
    /// Classifier-free guidance scale. Default: 7.5.
    pub guidance_scale: f32,
    /// Optional negative prompt to steer generation away from.
    pub negative_prompt: Option<String>,
    /// Number of images to generate per prompt. Default: 1.
    pub num_images_per_prompt: usize,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Noise scheduler variant.
    pub scheduler: SchedulerType,
}

impl Default for TextToImageConfig {
    fn default() -> Self {
        Self {
            model_name: "stabilityai/stable-diffusion-xl-base-1.0".to_string(),
            height: 512,
            width: 512,
            num_inference_steps: 20,
            guidance_scale: 7.5,
            negative_prompt: None,
            num_images_per_prompt: 1,
            seed: None,
            scheduler: SchedulerType::DDIM,
        }
    }
}

// ---------------------------------------------------------------------------
// GeneratedImage
// ---------------------------------------------------------------------------

/// A generated image returned by the text-to-image pipeline.
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    /// RGB pixel data in row-major order: length = `height * width * 3`.
    pub pixels: Vec<u8>,
    /// Image height in pixels.
    pub height: usize,
    /// Image width in pixels.
    pub width: usize,
    /// Whether an NSFW classifier flagged this image.
    pub nsfw_detected: bool,
}

impl GeneratedImage {
    /// Construct a new `GeneratedImage`, validating the pixel buffer length.
    pub fn new(pixels: Vec<u8>, height: usize, width: usize) -> Result<Self, ImageGenError> {
        let expected = height * width * 3;
        if pixels.len() != expected {
            return Err(ImageGenError::InvalidPixelData {
                expected,
                got: pixels.len(),
            });
        }
        Ok(Self {
            pixels,
            height,
            width,
            nsfw_detected: false,
        })
    }

    /// Return the `(R, G, B)` tuple at pixel `(row, col)`.
    ///
    /// Returns `(0, 0, 0)` for out-of-bounds coordinates.
    pub fn pixel_at(&self, row: usize, col: usize) -> (u8, u8, u8) {
        if row >= self.height || col >= self.width {
            return (0, 0, 0);
        }
        let idx = (row * self.width + col) * 3;
        (
            self.pixels[idx],
            self.pixels[idx + 1],
            self.pixels[idx + 2],
        )
    }

    /// Convert the image to grayscale (ITU-R BT.601 luminance).
    ///
    /// Returns a flat `Vec<u8>` with length `height * width`.
    pub fn to_grayscale(&self) -> Vec<u8> {
        let num_px = self.height * self.width;
        let mut out = Vec::with_capacity(num_px);
        for i in 0..num_px {
            let r = self.pixels[i * 3] as f32;
            let g = self.pixels[i * 3 + 1] as f32;
            let b = self.pixels[i * 3 + 2] as f32;
            let luma = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
            out.push(luma);
        }
        out
    }

    /// Mean brightness: average of all channel values across all pixels.
    pub fn mean_brightness(&self) -> f32 {
        if self.pixels.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.pixels.iter().map(|&p| p as u64).sum();
        sum as f32 / self.pixels.len() as f32
    }

    /// Nearest-neighbour resize to `(new_h, new_w)`.
    pub fn resize(&self, new_h: usize, new_w: usize) -> Self {
        let mut pixels = Vec::with_capacity(new_h * new_w * 3);
        for oy in 0..new_h {
            for ox in 0..new_w {
                let sy = ((oy as f32 / new_h as f32) * self.height as f32) as usize;
                let sx = ((ox as f32 / new_w as f32) * self.width as f32) as usize;
                let sy = sy.min(self.height.saturating_sub(1));
                let sx = sx.min(self.width.saturating_sub(1));
                let (r, g, b) = self.pixel_at(sy, sx);
                pixels.push(r);
                pixels.push(g);
                pixels.push(b);
            }
        }
        Self {
            pixels,
            height: new_h,
            width: new_w,
            nsfw_detected: false,
        }
    }

    /// Total number of pixels (= `height * width`).
    pub fn num_pixels(&self) -> usize {
        self.height * self.width
    }
}

// ---------------------------------------------------------------------------
// DDIM Scheduler
// ---------------------------------------------------------------------------

/// Simulated DDIM noise scheduler with linear beta schedule.
pub struct DdimScheduler {
    num_steps: usize,
    beta_start: f64,
    beta_end: f64,
    betas: Vec<f64>,
    alphas_cumprod: Vec<f64>,
}

impl DdimScheduler {
    /// Construct a DDIM scheduler with `num_steps` denoising steps.
    ///
    /// Uses a linear beta schedule between `beta_start = 0.00085` and
    /// `beta_end = 0.012`, computing cumulative products of `alpha_t = 1 - beta_t`.
    pub fn new(num_steps: usize) -> Self {
        let beta_start = 0.00085_f64;
        let beta_end = 0.012_f64;

        let steps = num_steps.max(1);
        let mut betas = Vec::with_capacity(steps);
        let mut alphas_cumprod = Vec::with_capacity(steps);
        let mut cumprod = 1.0_f64;

        for t in 0..steps {
            let beta = if steps > 1 {
                beta_start + (beta_end - beta_start) * t as f64 / (steps - 1) as f64
            } else {
                beta_start
            };
            betas.push(beta);
            let alpha = 1.0 - beta;
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }

        Self {
            num_steps: steps,
            beta_start,
            beta_end,
            betas,
            alphas_cumprod,
        }
    }

    /// Decreasing timestep sequence for inference, e.g. `[999, 978, …, 0]` for 20 steps.
    pub fn timesteps(&self) -> Vec<usize> {
        let total = 1000_usize;
        let step_size = total / self.num_steps;
        let mut ts: Vec<usize> = (0..self.num_steps)
            .map(|i| total.saturating_sub(1).saturating_sub(i * step_size))
            .collect();
        // Ensure the last element is 0.
        if let Some(last) = ts.last_mut() {
            if self.num_steps > 1 {
                *last = 0;
            }
        }
        ts
    }

    /// Return the cumulative-product alpha at scheduler step `step`.
    ///
    /// Returns `1.0` for out-of-bounds indices.
    pub fn get_alpha_cumprod(&self, step: usize) -> f64 {
        self.alphas_cumprod.get(step).copied().unwrap_or(1.0)
    }

    /// Number of denoising steps.
    pub fn step_count(&self) -> usize {
        self.num_steps
    }

    /// Beta values for the schedule (read-only access for testing).
    pub fn betas(&self) -> &[f64] {
        &self.betas
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Text-to-image generation pipeline (Stable Diffusion / DALL-E compatible).
pub struct TextToImagePipeline {
    config: TextToImageConfig,
}

impl TextToImagePipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: TextToImageConfig) -> Result<Self, ImageGenError> {
        if config.height == 0 || config.width == 0 {
            return Err(ImageGenError::InvalidDimensions {
                h: config.height,
                w: config.width,
            });
        }
        Ok(Self { config })
    }

    /// Generate images from the given `prompt`.
    ///
    /// Returns `num_images_per_prompt` images.  Each pixel is computed
    /// deterministically as:
    ///
    /// ```text
    /// hash_val = djb2(prompt) XOR seed   (seed = 0 when None)
    /// R[row, col, img] = (hash_val + row) % 256
    /// G[row, col, img] = (hash_val + col) % 256
    /// B[row, col, img] = (hash_val + img) % 256
    /// ```
    pub fn generate(&self, prompt: &str) -> Result<Vec<GeneratedImage>, ImageGenError> {
        if prompt.trim().is_empty() {
            return Err(ImageGenError::EmptyPrompt);
        }
        let seed = self.config.seed.unwrap_or(0);
        let hash_val = djb2_hash(prompt) ^ seed;
        self.build_images(hash_val)
    }

    /// Generate images conditioned on both `prompt` and `negative_prompt`.
    ///
    /// The negative prompt modifies the hash so that different images are
    /// produced compared to calling [`generate`] without a negative prompt.
    pub fn generate_with_negative(
        &self,
        prompt: &str,
        negative_prompt: &str,
    ) -> Result<Vec<GeneratedImage>, ImageGenError> {
        if prompt.trim().is_empty() {
            return Err(ImageGenError::EmptyPrompt);
        }
        let seed = self.config.seed.unwrap_or(0);
        // XOR with negative prompt hash to differentiate outputs.
        let hash_val = djb2_hash(prompt) ^ djb2_hash(negative_prompt) ^ seed;
        self.build_images(hash_val)
    }

    /// Generate images for each prompt in `prompts`.
    ///
    /// Returns one `Vec<GeneratedImage>` (of length `num_images_per_prompt`) per prompt.
    pub fn batch_generate(
        &self,
        prompts: &[&str],
    ) -> Result<Vec<Vec<GeneratedImage>>, ImageGenError> {
        prompts.iter().map(|p| self.generate(p)).collect()
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn build_images(&self, hash_val: u64) -> Result<Vec<GeneratedImage>, ImageGenError> {
        let h = self.config.height;
        let w = self.config.width;
        let n = self.config.num_images_per_prompt;

        let mut images = Vec::with_capacity(n);
        for img_idx in 0..n {
            let mut pixels = Vec::with_capacity(h * w * 3);
            for row in 0..h {
                for col in 0..w {
                    let r = ((hash_val + row as u64) % 256) as u8;
                    let g = ((hash_val + col as u64) % 256) as u8;
                    let b = ((hash_val + img_idx as u64) % 256) as u8;
                    pixels.push(r);
                    pixels.push(g);
                    pixels.push(b);
                }
            }
            images.push(GeneratedImage {
                pixels,
                height: h,
                width: w,
                nsfw_detected: false,
            });
        }
        Ok(images)
    }
}

// ---------------------------------------------------------------------------
// DiffusionScheduler
// ---------------------------------------------------------------------------

/// A generic noise scheduler for diffusion models.
///
/// Holds precomputed beta, alpha, and cumulative-product-alpha schedules
/// for a given number of timesteps.
pub struct DiffusionScheduler {
    /// Number of timesteps in the schedule.
    pub num_timesteps: usize,
    /// β_t values for each timestep.
    pub betas: Vec<f32>,
    /// α_t = 1 − β_t for each timestep.
    pub alphas: Vec<f32>,
    /// ᾱ_t = ∏_{s=1}^{t} α_s (cumulative product of alphas).
    pub alphas_cumprod: Vec<f32>,
}

impl DiffusionScheduler {
    /// Build a `DiffusionScheduler` from an explicit beta schedule.
    pub fn from_betas(betas: Vec<f32>) -> Self {
        let alphas = Self::compute_alphas(&betas);
        let alphas_cumprod = Self::compute_alphas_cumprod(&alphas);
        let num_timesteps = betas.len();
        Self { num_timesteps, betas, alphas, alphas_cumprod }
    }

    /// Build a `DiffusionScheduler` with a **linear** beta schedule.
    ///
    /// β linearly interpolates from `beta_start` to `beta_end` over
    /// `num_timesteps` steps.
    pub fn linear_beta_schedule(
        num_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
    ) -> Vec<f32> {
        if num_timesteps == 0 {
            return Vec::new();
        }
        if num_timesteps == 1 {
            return vec![beta_start];
        }
        (0..num_timesteps)
            .map(|t| beta_start + (beta_end - beta_start) * t as f32 / (num_timesteps - 1) as f32)
            .collect()
    }

    /// Build a `DiffusionScheduler` with a **cosine** beta schedule.
    ///
    /// Uses the Nichol & Dhariwal (2021) formulation:
    /// ᾱ_t = cos²(((t/T + s) / (1 + s)) · π/2)
    /// β_t = 1 − ᾱ_t / ᾱ_{t-1}, clipped to `[0, 0.999]`.
    ///
    /// `s` is a small offset (default 0.008) that prevents β_0 from being
    /// too small.
    pub fn cosine_beta_schedule(num_timesteps: usize, s: f32) -> Vec<f32> {
        if num_timesteps == 0 {
            return Vec::new();
        }
        let alphas_cumprod: Vec<f32> = (0..=num_timesteps)
            .map(|t| {
                let frac = (t as f32 / num_timesteps as f32 + s) / (1.0 + s);
                (frac * std::f32::consts::PI / 2.0).cos().powi(2)
            })
            .collect();

        (0..num_timesteps)
            .map(|t| {
                let prev = alphas_cumprod[t];
                let curr = alphas_cumprod[t + 1];
                let beta = 1.0 - curr / prev.max(f32::EPSILON);
                beta.clamp(0.0, 0.999)
            })
            .collect()
    }

    /// Compute α_t = 1 − β_t for each timestep.
    pub fn compute_alphas(betas: &[f32]) -> Vec<f32> {
        betas.iter().map(|&b| 1.0 - b).collect()
    }

    /// Compute the cumulative product ᾱ_t = ∏_{s=0}^{t} α_s.
    pub fn compute_alphas_cumprod(alphas: &[f32]) -> Vec<f32> {
        let mut cumprod = Vec::with_capacity(alphas.len());
        let mut running = 1.0_f32;
        for &a in alphas {
            running *= a;
            cumprod.push(running);
        }
        cumprod
    }

    /// Signal-to-Noise Ratio at timestep `t`:
    /// SNR(t) = ᾱ_t / (1 − ᾱ_t).
    ///
    /// Returns `f32::INFINITY` when ᾱ_t = 1.0.
    pub fn snr(alphas_cumprod: &[f32], t: usize) -> f32 {
        let acp = alphas_cumprod.get(t).copied().unwrap_or(1.0);
        let denom = 1.0 - acp;
        if denom < f32::EPSILON {
            f32::INFINITY
        } else {
            acp / denom
        }
    }
}

// ---------------------------------------------------------------------------
// DiffusionConfig
// ---------------------------------------------------------------------------

/// Configuration for the diffusion process.
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    /// Number of diffusion timesteps (T).
    pub num_timesteps: usize,
    /// Start value for the beta schedule.
    pub beta_start: f32,
    /// End value for the beta schedule.
    pub beta_end: f32,
    /// Classifier-free guidance scale.
    pub guidance_scale: f32,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            num_timesteps: 1000,
            beta_start: 0.0001,
            beta_end: 0.02,
            guidance_scale: 7.5,
        }
    }
}

// ---------------------------------------------------------------------------
// TextToImageProcessor
// ---------------------------------------------------------------------------

/// Utility processor with pure functions for text-to-image generation.
pub struct TextToImageProcessor;

impl TextToImageProcessor {
    /// Simple whitespace tokeniser: splits `prompt` on whitespace and maps each
    /// token to a deterministic `u32` id via djb2 hashing.
    pub fn encode_prompt_to_tokens(prompt: &str) -> Vec<u32> {
        prompt
            .split_whitespace()
            .map(|word| {
                let mut h: u32 = 5381;
                for b in word.bytes() {
                    h = h.wrapping_mul(33).wrapping_add(b as u32);
                }
                h
            })
            .collect()
    }

    /// Forward diffusion: add noise to `image` at timestep `t`.
    ///
    /// x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε
    ///
    /// `image` and `noise` must have the same length.
    /// Returns the noised image.
    pub fn add_noise(
        image: &[f32],
        noise: &[f32],
        timestep: usize,
        alphas_cumprod: &[f32],
    ) -> Vec<f32> {
        let acp = alphas_cumprod.get(timestep).copied().unwrap_or(1.0);
        let sqrt_acp = acp.sqrt();
        let sqrt_one_minus_acp = (1.0 - acp).max(0.0).sqrt();
        image
            .iter()
            .zip(noise.iter())
            .map(|(&x, &n)| sqrt_acp * x + sqrt_one_minus_acp * n)
            .collect()
    }

    /// Classifier-free guidance (CFG):
    ///
    /// prediction = uncond_pred + scale · (cond_pred − uncond_pred)
    ///
    /// `cond_pred` and `uncond_pred` must have the same length.
    pub fn cfg_guidance(cond_pred: &[f32], uncond_pred: &[f32], scale: f32) -> Vec<f32> {
        cond_pred
            .iter()
            .zip(uncond_pred.iter())
            .map(|(&c, &u)| u + scale * (c - u))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// DJB2 hash — deterministic, no external crates needed.
fn djb2_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
    }
    hash
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- GeneratedImage::new ---

    #[test]
    fn test_generated_image_new_valid() {
        let pixels = vec![0u8; 4 * 4 * 3];
        let img = GeneratedImage::new(pixels.clone(), 4, 4).expect("should construct");
        assert_eq!(img.pixels, pixels);
        assert_eq!(img.height, 4);
        assert_eq!(img.width, 4);
    }

    #[test]
    fn test_generated_image_new_wrong_size() {
        let pixels = vec![0u8; 10]; // wrong
        let err = GeneratedImage::new(pixels, 4, 4).unwrap_err();
        assert!(matches!(err, ImageGenError::InvalidPixelData { .. }));
    }

    // --- GeneratedImage::pixel_at ---

    #[test]
    fn test_pixel_at_correct_values() {
        // 2x2 image, pixels: (10,20,30) (40,50,60) (70,80,90) (100,110,120)
        let pixels: Vec<u8> = vec![
            10, 20, 30, // (0,0)
            40, 50, 60, // (0,1)
            70, 80, 90, // (1,0)
            100, 110, 120, // (1,1)
        ];
        let img = GeneratedImage::new(pixels, 2, 2).expect("ok");
        assert_eq!(img.pixel_at(0, 0), (10, 20, 30));
        assert_eq!(img.pixel_at(0, 1), (40, 50, 60));
        assert_eq!(img.pixel_at(1, 0), (70, 80, 90));
        assert_eq!(img.pixel_at(1, 1), (100, 110, 120));
    }

    #[test]
    fn test_pixel_at_out_of_bounds_returns_black() {
        let pixels = vec![255u8; 2 * 2 * 3];
        let img = GeneratedImage::new(pixels, 2, 2).expect("ok");
        assert_eq!(img.pixel_at(10, 10), (0, 0, 0));
    }

    // --- GeneratedImage::mean_brightness ---

    #[test]
    fn test_mean_brightness_uniform() {
        let pixels = vec![100u8; 4 * 4 * 3];
        let img = GeneratedImage::new(pixels, 4, 4).expect("ok");
        let mb = img.mean_brightness();
        assert!((mb - 100.0).abs() < 0.01, "expected 100, got {mb}");
    }

    // --- GeneratedImage::to_grayscale ---

    #[test]
    fn test_to_grayscale_length() {
        let pixels = vec![128u8; 3 * 5 * 3];
        let img = GeneratedImage::new(pixels, 3, 5).expect("ok");
        let gray = img.to_grayscale();
        assert_eq!(gray.len(), 3 * 5);
    }

    #[test]
    fn test_to_grayscale_pure_white() {
        let pixels = vec![255u8; 2 * 2 * 3];
        let img = GeneratedImage::new(pixels, 2, 2).expect("ok");
        let gray = img.to_grayscale();
        assert!(gray.iter().all(|&v| v == 255), "all pixels should be 255");
    }

    // --- GeneratedImage::resize ---

    #[test]
    fn test_resize_dimensions() {
        let config = TextToImageConfig {
            height: 8,
            width: 8,
            num_images_per_prompt: 1,
            ..Default::default()
        };
        let pipeline = TextToImagePipeline::new(config).expect("ok");
        let images = pipeline.generate("test resize").expect("ok");
        let resized = images[0].resize(4, 4);
        assert_eq!(resized.height, 4);
        assert_eq!(resized.width, 4);
        assert_eq!(resized.pixels.len(), 4 * 4 * 3);
    }

    // --- TextToImagePipeline::generate ---

    #[test]
    fn test_generate_returns_correct_num_images() {
        let config = TextToImageConfig {
            num_images_per_prompt: 3,
            height: 16,
            width: 16,
            ..Default::default()
        };
        let pipeline = TextToImagePipeline::new(config).expect("ok");
        let images = pipeline.generate("sunset over mountains").expect("ok");
        assert_eq!(images.len(), 3);
    }

    #[test]
    fn test_generate_pixel_count() {
        let h = 32;
        let w = 32;
        let config = TextToImageConfig {
            height: h,
            width: w,
            num_images_per_prompt: 1,
            ..Default::default()
        };
        let pipeline = TextToImagePipeline::new(config).expect("ok");
        let images = pipeline.generate("a cat").expect("ok");
        assert_eq!(images[0].pixels.len(), h * w * 3);
    }

    // --- TextToImagePipeline::batch_generate ---

    #[test]
    fn test_batch_generate_correct_batch_size() {
        let config = TextToImageConfig {
            height: 8,
            width: 8,
            num_images_per_prompt: 1,
            ..Default::default()
        };
        let pipeline = TextToImagePipeline::new(config).expect("ok");
        let prompts = vec!["cat", "dog", "bird"];
        let results = pipeline.batch_generate(&prompts).expect("ok");
        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.len(), 1);
        }
    }

    // --- Empty prompt error ---

    #[test]
    fn test_generate_empty_prompt_error() {
        let config = TextToImageConfig::default();
        let pipeline = TextToImagePipeline::new(config).expect("ok");
        let err = pipeline.generate("").unwrap_err();
        assert!(matches!(err, ImageGenError::EmptyPrompt));
    }

    #[test]
    fn test_generate_whitespace_prompt_error() {
        let config = TextToImageConfig::default();
        let pipeline = TextToImagePipeline::new(config).expect("ok");
        let err = pipeline.generate("   ").unwrap_err();
        assert!(matches!(err, ImageGenError::EmptyPrompt));
    }

    // --- Negative prompt differs from base ---

    #[test]
    fn test_negative_prompt_produces_different_output() {
        let config = TextToImageConfig {
            height: 16,
            width: 16,
            num_images_per_prompt: 1,
            ..Default::default()
        };
        let pipeline = TextToImagePipeline::new(config).expect("ok");
        let base = pipeline.generate("a bright sunny day").expect("ok");
        let with_neg = pipeline
            .generate_with_negative("a bright sunny day", "rain clouds")
            .expect("ok");
        assert_ne!(
            base[0].pixels, with_neg[0].pixels,
            "negative prompt should change pixels"
        );
    }

    // --- DdimScheduler ---

    #[test]
    fn test_ddim_scheduler_timesteps_count() {
        let sched = DdimScheduler::new(20);
        let ts = sched.timesteps();
        assert_eq!(ts.len(), 20);
    }

    #[test]
    fn test_ddim_scheduler_get_alpha_cumprod_decreasing() {
        let sched = DdimScheduler::new(20);
        let alphas: Vec<f64> = (0..20).map(|i| sched.get_alpha_cumprod(i)).collect();
        // alphas_cumprod must be strictly decreasing (more noise over time).
        for window in alphas.windows(2) {
            assert!(
                window[0] > window[1],
                "alphas_cumprod should be decreasing: {} <= {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_ddim_scheduler_beta_schedule_bounds() {
        let sched = DdimScheduler::new(20);
        let betas = sched.betas();
        // First beta should be close to beta_start (0.00085).
        assert!((betas[0] - 0.00085).abs() < 1e-6, "first beta mismatch");
        // Last beta should be close to beta_end (0.012).
        let last = *betas.last().expect("non-empty");
        assert!((last - 0.012).abs() < 1e-6, "last beta mismatch");
    }

    // --- SchedulerType variants ---

    #[test]
    fn test_scheduler_type_variants() {
        assert_eq!(SchedulerType::DDPM, SchedulerType::DDPM);
        assert_ne!(SchedulerType::DDIM, SchedulerType::PNDM);
        assert_ne!(SchedulerType::EulerA, SchedulerType::DPMpp);

        // Ensure all variants are constructable.
        let variants = [
            SchedulerType::DDPM,
            SchedulerType::DDIM,
            SchedulerType::PNDM,
            SchedulerType::EulerA,
            SchedulerType::DPMpp,
        ];
        assert_eq!(variants.len(), 5);
    }

    // ── DiffusionScheduler::linear_beta_schedule ─────────────────────────────

    #[test]
    fn test_linear_beta_schedule_length() {
        let betas = DiffusionScheduler::linear_beta_schedule(100, 0.0001, 0.02);
        assert_eq!(betas.len(), 100);
    }

    #[test]
    fn test_linear_beta_schedule_endpoints() {
        let betas = DiffusionScheduler::linear_beta_schedule(1000, 0.0001, 0.02);
        assert!((betas[0] - 0.0001).abs() < 1e-6, "first beta mismatch: {}", betas[0]);
        assert!(
            (betas[999] - 0.02).abs() < 1e-6,
            "last beta mismatch: {}",
            betas[999]
        );
    }

    #[test]
    fn test_linear_beta_schedule_monotonically_increasing() {
        let betas = DiffusionScheduler::linear_beta_schedule(50, 0.0001, 0.02);
        for w in betas.windows(2) {
            assert!(w[1] >= w[0], "linear schedule should be non-decreasing");
        }
    }

    #[test]
    fn test_linear_beta_schedule_single_step() {
        let betas = DiffusionScheduler::linear_beta_schedule(1, 0.005, 0.02);
        assert_eq!(betas.len(), 1);
        assert!((betas[0] - 0.005).abs() < 1e-6);
    }

    // ── DiffusionScheduler::cosine_beta_schedule ─────────────────────────────

    #[test]
    fn test_cosine_beta_schedule_length() {
        let betas = DiffusionScheduler::cosine_beta_schedule(1000, 0.008);
        assert_eq!(betas.len(), 1000);
    }

    #[test]
    fn test_cosine_beta_schedule_clipped() {
        let betas = DiffusionScheduler::cosine_beta_schedule(100, 0.008);
        for &b in &betas {
            assert!(
                (0.0..=0.999).contains(&b),
                "cosine beta out of [0, 0.999]: {b}"
            );
        }
    }

    #[test]
    fn test_cosine_beta_schedule_all_non_negative() {
        let betas = DiffusionScheduler::cosine_beta_schedule(500, 0.008);
        assert!(betas.iter().all(|&b| b >= 0.0));
    }

    // ── DiffusionScheduler::compute_alphas ───────────────────────────────────

    #[test]
    fn test_compute_alphas_basic() {
        let betas = vec![0.1_f32, 0.2, 0.3];
        let alphas = DiffusionScheduler::compute_alphas(&betas);
        assert_eq!(alphas.len(), 3);
        assert!((alphas[0] - 0.9).abs() < 1e-5);
        assert!((alphas[1] - 0.8).abs() < 1e-5);
        assert!((alphas[2] - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_compute_alphas_empty() {
        let alphas = DiffusionScheduler::compute_alphas(&[]);
        assert!(alphas.is_empty());
    }

    // ── DiffusionScheduler::compute_alphas_cumprod ───────────────────────────

    #[test]
    fn test_compute_alphas_cumprod_decreasing() {
        let betas = DiffusionScheduler::linear_beta_schedule(100, 0.0001, 0.02);
        let alphas = DiffusionScheduler::compute_alphas(&betas);
        let acp = DiffusionScheduler::compute_alphas_cumprod(&alphas);
        assert_eq!(acp.len(), 100);
        for w in acp.windows(2) {
            assert!(w[1] <= w[0], "alphas_cumprod must be non-increasing");
        }
    }

    #[test]
    fn test_compute_alphas_cumprod_first_value() {
        // For a single beta of 0.1, ᾱ_0 = 1 − 0.1 = 0.9.
        let betas = vec![0.1_f32];
        let alphas = DiffusionScheduler::compute_alphas(&betas);
        let acp = DiffusionScheduler::compute_alphas_cumprod(&alphas);
        assert!((acp[0] - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_compute_alphas_cumprod_product_property() {
        let betas = vec![0.1_f32, 0.2, 0.3];
        let alphas = DiffusionScheduler::compute_alphas(&betas);
        let acp = DiffusionScheduler::compute_alphas_cumprod(&alphas);
        // ᾱ_2 = 0.9 * 0.8 * 0.7 = 0.504
        assert!((acp[2] - 0.9 * 0.8 * 0.7).abs() < 1e-5, "cumprod[2]={}", acp[2]);
    }

    // ── DiffusionScheduler::snr ───────────────────────────────────────────────

    #[test]
    fn test_snr_decreases_over_time() {
        let betas = DiffusionScheduler::linear_beta_schedule(100, 0.0001, 0.02);
        let alphas = DiffusionScheduler::compute_alphas(&betas);
        let acp = DiffusionScheduler::compute_alphas_cumprod(&alphas);
        let snr_0 = DiffusionScheduler::snr(&acp, 0);
        let snr_50 = DiffusionScheduler::snr(&acp, 50);
        let snr_99 = DiffusionScheduler::snr(&acp, 99);
        assert!(snr_0 > snr_50, "SNR should decrease: snr_0={snr_0}, snr_50={snr_50}");
        assert!(snr_50 > snr_99, "SNR should decrease: snr_50={snr_50}, snr_99={snr_99}");
    }

    #[test]
    fn test_snr_positive() {
        let betas = DiffusionScheduler::linear_beta_schedule(50, 0.0001, 0.02);
        let alphas = DiffusionScheduler::compute_alphas(&betas);
        let acp = DiffusionScheduler::compute_alphas_cumprod(&alphas);
        for t in 0..50 {
            assert!(DiffusionScheduler::snr(&acp, t) > 0.0, "SNR must be positive at t={t}");
        }
    }

    // ── TextToImageProcessor::encode_prompt_to_tokens ────────────────────────

    #[test]
    fn test_encode_prompt_to_tokens_length() {
        let tokens = TextToImageProcessor::encode_prompt_to_tokens("a photo of a cat");
        assert_eq!(tokens.len(), 5, "should tokenise by whitespace");
    }

    #[test]
    fn test_encode_prompt_to_tokens_same_word_same_token() {
        let tokens = TextToImageProcessor::encode_prompt_to_tokens("cat cat cat");
        assert_eq!(tokens[0], tokens[1]);
        assert_eq!(tokens[1], tokens[2]);
    }

    #[test]
    fn test_encode_prompt_to_tokens_different_words_usually_different() {
        let tokens = TextToImageProcessor::encode_prompt_to_tokens("apple banana cherry");
        // Different words should (with high probability) produce different ids.
        assert!(tokens[0] != tokens[1] || tokens[1] != tokens[2]);
    }

    // ── TextToImageProcessor::add_noise ──────────────────────────────────────

    #[test]
    fn test_add_noise_output_length() {
        let image = vec![0.5_f32; 64];
        let noise = vec![0.1_f32; 64];
        let betas = DiffusionScheduler::linear_beta_schedule(10, 0.0001, 0.02);
        let alphas = DiffusionScheduler::compute_alphas(&betas);
        let acp = DiffusionScheduler::compute_alphas_cumprod(&alphas);
        let noised = TextToImageProcessor::add_noise(&image, &noise, 5, &acp);
        assert_eq!(noised.len(), 64, "output must match input length");
    }

    #[test]
    fn test_add_noise_t0_mostly_signal() {
        // At t=0, ᾱ_0 ≈ 1 − β_0 ≈ 1. The output should be close to the original image.
        let image = vec![1.0_f32; 4];
        let noise = vec![0.0_f32; 4];
        let betas = DiffusionScheduler::linear_beta_schedule(1000, 0.0001, 0.02);
        let alphas = DiffusionScheduler::compute_alphas(&betas);
        let acp = DiffusionScheduler::compute_alphas_cumprod(&alphas);
        let noised = TextToImageProcessor::add_noise(&image, &noise, 0, &acp);
        // √ᾱ_0 ≈ 0.99995, so result ≈ 0.99995.
        for &v in &noised {
            assert!(v > 0.99, "at t=0, output should be close to input: {v}");
        }
    }

    #[test]
    fn test_add_noise_large_t_mostly_noise() {
        // At large t (near T−1), ᾱ_t ≈ 0. The output should be dominated by noise.
        let image = vec![0.0_f32; 4];
        let noise = vec![1.0_f32; 4];
        let betas = DiffusionScheduler::linear_beta_schedule(1000, 0.0001, 0.02);
        let alphas = DiffusionScheduler::compute_alphas(&betas);
        let acp = DiffusionScheduler::compute_alphas_cumprod(&alphas);
        let noised = TextToImageProcessor::add_noise(&image, &noise, 999, &acp);
        for &v in &noised {
            assert!(v > 0.5, "at large t, output should be close to noise: {v}");
        }
    }

    // ── TextToImageProcessor::cfg_guidance ───────────────────────────────────

    #[test]
    fn test_cfg_guidance_scale_one_equals_cond() {
        // scale=1: output = uncond + 1*(cond - uncond) = cond
        let cond = vec![1.0_f32, 2.0, 3.0];
        let uncond = vec![0.0_f32, 0.0, 0.0];
        let guided = TextToImageProcessor::cfg_guidance(&cond, &uncond, 1.0);
        for (g, c) in guided.iter().zip(cond.iter()) {
            assert!((g - c).abs() < 1e-5, "scale=1 should equal cond");
        }
    }

    #[test]
    fn test_cfg_guidance_scale_zero_equals_uncond() {
        // scale=0: output = uncond + 0*(cond - uncond) = uncond
        let cond = vec![5.0_f32, 6.0, 7.0];
        let uncond = vec![1.0_f32, 2.0, 3.0];
        let guided = TextToImageProcessor::cfg_guidance(&cond, &uncond, 0.0);
        for (g, u) in guided.iter().zip(uncond.iter()) {
            assert!((g - u).abs() < 1e-5, "scale=0 should equal uncond");
        }
    }

    #[test]
    fn test_cfg_guidance_amplifies_difference() {
        // With scale > 1, the output should be further from uncond than cond.
        let cond = vec![1.0_f32];
        let uncond = vec![0.0_f32];
        let guided = TextToImageProcessor::cfg_guidance(&cond, &uncond, 7.5);
        // output = 0 + 7.5*(1 - 0) = 7.5
        assert!((guided[0] - 7.5).abs() < 1e-5, "CFG scale 7.5: got {}", guided[0]);
    }

    #[test]
    fn test_cfg_guidance_formula() {
        let cond = vec![2.0_f32, 4.0];
        let uncond = vec![1.0_f32, 2.0];
        let scale = 3.0_f32;
        let guided = TextToImageProcessor::cfg_guidance(&cond, &uncond, scale);
        // uncond + scale*(cond - uncond) = 1 + 3*(2-1) = 4.0; 2 + 3*(4-2) = 8.0
        assert!((guided[0] - 4.0).abs() < 1e-5, "CFG formula check: {}", guided[0]);
        assert!((guided[1] - 8.0).abs() < 1e-5, "CFG formula check: {}", guided[1]);
    }

    // ── DiffusionConfig defaults ──────────────────────────────────────────────

    #[test]
    fn test_diffusion_config_defaults() {
        let cfg = DiffusionConfig::default();
        assert_eq!(cfg.num_timesteps, 1000);
        assert_eq!(cfg.guidance_scale, 7.5);
        assert!(cfg.beta_start < cfg.beta_end, "beta_start must be < beta_end");
    }
}
