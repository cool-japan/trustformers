//! # Video Classification Pipeline
//!
//! VideoMAE / TimeSformer-compatible video clip classification.
//!
//! ## Supported model families
//! - **VideoMAE** (MCG-NJU/videomae-base-finetuned-kinetics)
//! - **TimeSformer** — divide-and-conquer space-time attention
//!
//! ## Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::video_classification::{
//!     VideoClassificationConfig, VideoClassificationPipeline, VideoClip, FrameSamplingStrategy,
//! };
//!
//! let config = VideoClassificationConfig::default();
//! let pipeline = VideoClassificationPipeline::new(config)?;
//! let frames = vec![vec![0.5f32; 224 * 224 * 3]; 16];
//! let clip = VideoClip::new(frames, 224, 224, 25.0)?;
//! let result = pipeline.classify(&clip)?;
//! println!("Top label: {} ({:.2})", result.label, result.score);
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the video classification pipeline.
#[derive(Debug, Error)]
pub enum VideoError {
    #[error("Empty video: no frames")]
    EmptyVideo,
    #[error("Invalid frame at index {0}: wrong size")]
    InvalidFrame(usize),
    #[error("Invalid FPS: {0}")]
    InvalidFps(f32),
    #[error("Not enough frames: need {need}, have {have}")]
    NotEnoughFrames { need: usize, have: usize },
    #[error("Model error: {0}")]
    ModelError(String),
}

// ---------------------------------------------------------------------------
// Frame sampling strategy
// ---------------------------------------------------------------------------

/// How frames are sampled from a [`VideoClip`] prior to classification.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameSamplingStrategy {
    /// Sample frames uniformly spaced across the full clip.
    Uniform,
    /// Sample frames centred around the middle of the clip.
    Center,
    /// Sample frames deterministically using a fixed seed.
    Random { seed: u64 },
    /// Sample frames with higher density around motion-heavy regions (simulated).
    MotionBased,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`VideoClassificationPipeline`].
#[derive(Debug, Clone)]
pub struct VideoClassificationConfig {
    /// HuggingFace model identifier or local path.
    pub model_name: String,
    /// Number of frames to sample from the clip. Default: 16.
    pub num_frames: usize,
    /// Expected frame height in pixels. Default: 224.
    pub frame_height: usize,
    /// Expected frame width in pixels. Default: 224.
    pub frame_width: usize,
    /// Action-class label strings.
    pub labels: Vec<String>,
    /// Number of top-k results to return. Default: 5.
    pub top_k: usize,
    /// Frame sampling strategy.
    pub sampling_strategy: FrameSamplingStrategy,
}

impl Default for VideoClassificationConfig {
    fn default() -> Self {
        let labels: Vec<String> = vec![
            "abseiling",
            "air drumming",
            "answering questions",
            "archery",
            "arm wrestling",
            "baking cookies",
            "balloon blowing",
            "bandaging",
            "barbequing",
            "bartending",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        Self {
            model_name: "MCG-NJU/videomae-base-finetuned-kinetics".to_string(),
            num_frames: 16,
            frame_height: 224,
            frame_width: 224,
            labels,
            top_k: 5,
            sampling_strategy: FrameSamplingStrategy::Uniform,
        }
    }
}

// ---------------------------------------------------------------------------
// VideoClip
// ---------------------------------------------------------------------------

/// A video represented as a sequence of raw frames.
#[derive(Debug, Clone)]
pub struct VideoClip {
    /// Flat pixel values per frame, shape `[H * W * channels]`.
    pub frames: Vec<Vec<f32>>,
    /// Frame height in pixels.
    pub frame_height: usize,
    /// Frame width in pixels.
    pub frame_width: usize,
    /// Number of colour channels (3 for RGB).
    pub channels: usize,
    /// Frame rate.
    pub fps: f32,
    /// Clip duration in seconds.
    pub duration_seconds: f32,
}

impl VideoClip {
    /// Construct a `VideoClip` from pre-decoded frames.
    ///
    /// Each frame in `frames` must have exactly `height * width * 3` elements.
    /// `fps` must be positive.
    pub fn new(
        frames: Vec<Vec<f32>>,
        height: usize,
        width: usize,
        fps: f32,
    ) -> Result<Self, VideoError> {
        if frames.is_empty() {
            return Err(VideoError::EmptyVideo);
        }
        if fps <= 0.0 || !fps.is_finite() {
            return Err(VideoError::InvalidFps(fps));
        }
        let expected = height * width * 3;
        for (idx, frame) in frames.iter().enumerate() {
            if frame.len() != expected {
                return Err(VideoError::InvalidFrame(idx));
            }
        }
        let num_frames = frames.len();
        let duration_seconds = num_frames as f32 / fps;
        Ok(Self {
            frames,
            frame_height: height,
            frame_width: width,
            channels: 3,
            fps,
            duration_seconds,
        })
    }

    /// Number of frames in the clip.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Clip duration in seconds, computed as `num_frames / fps`.
    pub fn duration_seconds(&self) -> f32 {
        self.duration_seconds
    }

    /// Sample `count` frames from the clip according to `strategy`.
    ///
    /// If `count` exceeds the clip length the clip is returned as-is (all frames).
    /// Returns owned frame data (cloned from the clip).
    pub fn sample_frames(
        &self,
        count: usize,
        strategy: &FrameSamplingStrategy,
    ) -> Vec<Vec<f32>> {
        let n = self.frames.len();
        if count == 0 || n == 0 {
            return Vec::new();
        }
        let count = count.min(n);

        let indices = match strategy {
            FrameSamplingStrategy::Uniform | FrameSamplingStrategy::MotionBased => {
                uniform_indices(n, count)
            }
            FrameSamplingStrategy::Center => center_indices(n, count),
            FrameSamplingStrategy::Random { seed } => random_indices(n, count, *seed),
        };

        indices
            .into_iter()
            .map(|i| self.frames[i].clone())
            .collect()
    }

    /// Return a reference to the frame at `index`, or `None` if out of bounds.
    pub fn frame_at(&self, index: usize) -> Option<&Vec<f32>> {
        self.frames.get(index)
    }

    /// Pixel-wise mean across all frames.
    ///
    /// Returns a `Vec<f32>` with the same length as a single frame.
    /// Returns an empty vector if the clip has no frames.
    pub fn mean_frame(&self) -> Vec<f32> {
        if self.frames.is_empty() {
            return Vec::new();
        }
        let frame_len = self.frames[0].len();
        let n = self.frames.len() as f32;
        let mut acc = vec![0.0_f32; frame_len];
        for frame in &self.frames {
            for (a, &p) in acc.iter_mut().zip(frame.iter()) {
                *a += p;
            }
        }
        acc.iter_mut().for_each(|v| *v /= n);
        acc
    }
}

// ---------------------------------------------------------------------------
// Classification result
// ---------------------------------------------------------------------------

/// Result of classifying a video clip.
#[derive(Debug, Clone)]
pub struct VideoClassificationResult {
    /// Top-1 predicted label.
    pub label: String,
    /// Top-1 confidence score.
    pub score: f32,
    /// Top-K labels sorted by descending confidence.
    pub top_labels: Vec<(String, f32)>,
    /// Number of frames that were actually processed.
    pub frames_processed: usize,
    /// Simulated inference latency in milliseconds.
    pub inference_time_ms: u64,
}

impl VideoClassificationResult {
    /// Return the top `k` entries from `top_labels` (at most).
    pub fn top_k(&self, k: usize) -> Vec<&(String, f32)> {
        self.top_labels.iter().take(k).collect()
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Video classification pipeline (VideoMAE / TimeSformer compatible).
pub struct VideoClassificationPipeline {
    config: VideoClassificationConfig,
}

impl VideoClassificationPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: VideoClassificationConfig) -> Result<Self, VideoError> {
        if config.labels.is_empty() {
            return Err(VideoError::ModelError(
                "labels list must not be empty".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Classify a [`VideoClip`].
    ///
    /// 1. Sample `num_frames` from the clip using the configured strategy.
    /// 2. Compute a mock classification from the mean pixel value.
    /// 3. Return top-K scores, sorted by descending confidence.
    pub fn classify(&self, video: &VideoClip) -> Result<VideoClassificationResult, VideoError> {
        if video.frames.is_empty() {
            return Err(VideoError::EmptyVideo);
        }
        let sampled = video.sample_frames(
            self.config.num_frames,
            &self.config.sampling_strategy,
        );
        self.classify_frames(&sampled)
    }

    /// Classify from pre-extracted frames.
    pub fn classify_frames(
        &self,
        frames: &[Vec<f32>],
    ) -> Result<VideoClassificationResult, VideoError> {
        if frames.is_empty() {
            return Err(VideoError::EmptyVideo);
        }

        let num_classes = self.config.labels.len();
        let frames_processed = frames.len();

        // Mock: derive label index from mean pixel value across all frames.
        let total_pixels: usize = frames.iter().map(|f| f.len()).sum();
        let pixel_sum: f32 = frames.iter().flat_map(|f| f.iter()).sum();
        let mean_pixel = if total_pixels > 0 {
            pixel_sum / total_pixels as f32
        } else {
            0.0
        };

        // Clamp to [0, 1] range for label indexing.
        let clamped = mean_pixel.clamp(0.0, 1.0 - f32::EPSILON);
        let top1_idx = ((clamped * num_classes as f32) as usize).min(num_classes - 1);

        // Generate deterministic softmax-like scores.
        let scores = mock_scores(top1_idx, num_classes, mean_pixel);

        // Build top-k list sorted by score descending.
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.config.top_k.min(num_classes);
        let top_labels: Vec<(String, f32)> = indexed
            .iter()
            .take(k)
            .map(|&(idx, score)| (self.config.labels[idx].clone(), score))
            .collect();

        let (label, score) = top_labels
            .first()
            .map(|(l, s)| (l.clone(), *s))
            .unwrap_or_else(|| (String::new(), 0.0));

        Ok(VideoClassificationResult {
            label,
            score,
            top_labels,
            frames_processed,
            inference_time_ms: frames_processed as u64 * 2, // 2 ms per frame mock
        })
    }

    /// Classify a batch of video clips.
    pub fn classify_batch(
        &self,
        videos: &[&VideoClip],
    ) -> Result<Vec<VideoClassificationResult>, VideoError> {
        videos.iter().map(|v| self.classify(v)).collect()
    }
}

// ---------------------------------------------------------------------------
// VideoFrame / VideoInput (new API)
// ---------------------------------------------------------------------------

/// A single decoded video frame.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Raw pixel bytes (RGB, 8-bit per channel).
    pub pixels: Vec<u8>,
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
    /// Timestamp within the video stream in milliseconds.
    pub timestamp_ms: f32,
}

impl VideoFrame {
    /// Construct a new `VideoFrame`.
    pub fn new(pixels: Vec<u8>, width: usize, height: usize, timestamp_ms: f32) -> Self {
        Self { pixels, width, height, timestamp_ms }
    }

    /// Total number of pixels (width × height).
    pub fn num_pixels(&self) -> usize {
        self.width * self.height
    }
}

/// A video represented as a sequence of decoded frames plus metadata.
#[derive(Debug, Clone)]
pub struct VideoInput {
    /// Decoded frames in chronological order.
    pub frames: Vec<VideoFrame>,
    /// Frame rate of the video stream.
    pub fps: f32,
    /// Total duration of the video in milliseconds.
    pub duration_ms: f32,
}

impl VideoInput {
    /// Construct a `VideoInput` from frames, fps, and total duration.
    pub fn new(frames: Vec<VideoFrame>, fps: f32, duration_ms: f32) -> Self {
        Self { frames, fps, duration_ms }
    }

    /// Number of frames in the input.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }
}

// ---------------------------------------------------------------------------
// TemporalPoolType
// ---------------------------------------------------------------------------

/// Pooling strategy to aggregate per-frame feature vectors into a single
/// clip-level representation.
#[derive(Debug, Clone, PartialEq)]
pub enum TemporalPoolType {
    /// Element-wise mean across frames.
    Mean,
    /// Element-wise maximum across frames.
    Max,
    /// Return the features of the last frame.
    Last,
    /// Weighted mean; weights must be the same length as the number of frames.
    WeightedMean(Vec<f32>),
}

// ---------------------------------------------------------------------------
// VideoClassificationResultNew (enhanced)
// ---------------------------------------------------------------------------

/// Enhanced per-item classification result used with the new API.
#[derive(Debug, Clone)]
pub struct VideoClassificationItem {
    /// Human-readable class label.
    pub label: String,
    /// Confidence score in `[0.0, 1.0]`.
    pub score: f32,
    /// Numeric class id.
    pub label_id: usize,
}

// ---------------------------------------------------------------------------
// VideoFeatureExtractor
// ---------------------------------------------------------------------------

/// Utility functions for video feature extraction.
pub struct VideoFeatureExtractor;

impl VideoFeatureExtractor {
    /// Aggregate `frame_features` along the temporal axis using `pool_type`.
    ///
    /// Each inner `Vec<f32>` is one frame's feature vector.
    /// All frame vectors must have the same length; the output has that length.
    ///
    /// Returns an empty `Vec` if `frame_features` is empty.
    pub fn temporal_pool(frame_features: &[Vec<f32>], pool_type: &TemporalPoolType) -> Vec<f32> {
        if frame_features.is_empty() {
            return Vec::new();
        }
        let dim = frame_features[0].len();
        let n = frame_features.len();

        match pool_type {
            TemporalPoolType::Mean => {
                let mut out = vec![0.0_f32; dim];
                for frame in frame_features {
                    for (o, &v) in out.iter_mut().zip(frame.iter()) {
                        *o += v;
                    }
                }
                out.iter_mut().for_each(|v| *v /= n as f32);
                out
            }
            TemporalPoolType::Max => {
                let mut out = vec![f32::NEG_INFINITY; dim];
                for frame in frame_features {
                    for (o, &v) in out.iter_mut().zip(frame.iter()) {
                        if v > *o {
                            *o = v;
                        }
                    }
                }
                out
            }
            TemporalPoolType::Last => frame_features.last().cloned().unwrap_or_default(),
            TemporalPoolType::WeightedMean(weights) => {
                let mut out = vec![0.0_f32; dim];
                let weight_sum: f32 = weights.iter().sum();
                let effective_sum = if weight_sum.abs() < f32::EPSILON {
                    1.0
                } else {
                    weight_sum
                };
                for (frame, weight) in frame_features.iter().zip(
                    weights
                        .iter()
                        .chain(std::iter::repeat(&0.0_f32))
                        .take(n),
                ) {
                    for (o, &v) in out.iter_mut().zip(frame.iter()) {
                        *o += v * weight;
                    }
                }
                out.iter_mut().for_each(|v| *v /= effective_sum);
                out
            }
        }
    }

    /// Return `num_samples` uniformly spaced frame indices from a clip of
    /// `total_frames` frames (0-indexed).
    ///
    /// Indices are guaranteed to be in `[0, total_frames)`.
    /// Returns an empty vector when `total_frames == 0` or `num_samples == 0`.
    pub fn sample_frames_uniform(total_frames: usize, num_samples: usize) -> Vec<usize> {
        if total_frames == 0 || num_samples == 0 {
            return Vec::new();
        }
        let n = num_samples.min(total_frames);
        if n == 1 {
            return vec![0];
        }
        (0..n)
            .map(|i| {
                let idx = (i as f64 * (total_frames - 1) as f64 / (n - 1) as f64).round() as usize;
                idx.min(total_frames - 1)
            })
            .collect()
    }

    /// Sample a centred clip of `clip_duration` frames and then uniformly pick
    /// `num_samples` from that clip.
    ///
    /// The clip is centred around the middle of `total_frames`.
    /// If `clip_duration >= total_frames` the full video is used.
    pub fn sample_frames_center_crop(
        total_frames: usize,
        num_samples: usize,
        clip_duration: usize,
    ) -> Vec<usize> {
        if total_frames == 0 || num_samples == 0 {
            return Vec::new();
        }
        let effective_clip = clip_duration.min(total_frames);
        let mid = total_frames / 2;
        let half = effective_clip / 2;
        let start = mid.saturating_sub(half);
        let end = (start + effective_clip).min(total_frames);
        let clip_len = end - start;

        let n = num_samples.min(clip_len);
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![start];
        }
        (0..n)
            .map(|i| {
                let local =
                    (i as f64 * (clip_len - 1) as f64 / (n - 1) as f64).round() as usize;
                (start + local).min(total_frames - 1)
            })
            .collect()
    }

    /// Approximate optical flow magnitude between two consecutive frames using
    /// absolute pixel difference.
    ///
    /// `frame1` and `frame2` are raw RGB byte buffers of length `w * h * 3`.
    /// Returns a `w * h` magnitude map (one value per pixel).
    pub fn optical_flow_magnitude(
        frame1: &[u8],
        frame2: &[u8],
        w: usize,
        h: usize,
    ) -> Vec<f32> {
        let num_px = w * h;
        let mut magnitudes = vec![0.0_f32; num_px];
        for px in 0..num_px {
            let base = px * 3;
            let mut sq_sum = 0.0_f32;
            for c in 0..3 {
                let idx = base + c;
                let a = frame1.get(idx).copied().unwrap_or(0) as f32;
                let b = frame2.get(idx).copied().unwrap_or(0) as f32;
                let diff = a - b;
                sq_sum += diff * diff;
            }
            magnitudes[px] = sq_sum.sqrt();
        }
        magnitudes
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Uniformly spaced frame indices.
fn uniform_indices(total: usize, count: usize) -> Vec<usize> {
    if count == 0 {
        return Vec::new();
    }
    if count >= total {
        return (0..total).collect();
    }
    (0..count)
        .map(|i| {
            let idx = (i as f64 * (total - 1) as f64 / (count - 1).max(1) as f64).round() as usize;
            idx.min(total - 1)
        })
        .collect()
}

/// Centre-aligned frame indices.
fn center_indices(total: usize, count: usize) -> Vec<usize> {
    if count == 0 {
        return Vec::new();
    }
    if count >= total {
        return (0..total).collect();
    }
    let mid = total / 2;
    let half = count / 2;
    let start = mid.saturating_sub(half);
    let start = start.min(total.saturating_sub(count));
    (start..start + count).collect()
}

/// Deterministic "random" frame indices based on `seed`.
fn random_indices(total: usize, count: usize, seed: u64) -> Vec<usize> {
    if count == 0 {
        return Vec::new();
    }
    if count >= total {
        return (0..total).collect();
    }
    // LCG to generate pseudo-random indices.
    let mut state = seed ^ (total as u64);
    let mut indices = Vec::with_capacity(count);
    for _ in 0..count {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        indices.push((state >> 33) as usize % total);
    }
    indices
}

/// Generate a mock score distribution (softmax-like) that peaks at `top_idx`.
fn mock_scores(top_idx: usize, num_classes: usize, mean_pixel: f32) -> Vec<f32> {
    // Use a simple temperature-scaled exponential centred on top_idx.
    let temperature = 2.0_f32;
    let mut logits: Vec<f32> = (0..num_classes)
        .map(|i| {
            let dist = (i as f32 - top_idx as f32).abs();
            // Add a tiny pixel-based perturbation to make scores vary.
            -(dist * temperature) + mean_pixel * 0.1
        })
        .collect();

    // Softmax.
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    logits.iter_mut().for_each(|l| *l = (*l - max_logit).exp());
    let sum: f32 = logits.iter().sum();
    if sum > 0.0 {
        logits.iter_mut().for_each(|l| *l /= sum);
    }
    logits
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip(num_frames: usize, h: usize, w: usize) -> VideoClip {
        let frame_size = h * w * 3;
        let frames: Vec<Vec<f32>> = (0..num_frames)
            .map(|i| vec![(i as f32 / num_frames as f32); frame_size])
            .collect();
        VideoClip::new(frames, h, w, 25.0).expect("valid clip")
    }

    fn default_pipeline() -> VideoClassificationPipeline {
        VideoClassificationPipeline::new(VideoClassificationConfig::default())
            .expect("valid pipeline")
    }

    // --- VideoClip::new ---

    #[test]
    fn test_video_clip_new_valid() {
        let clip = make_clip(8, 4, 4);
        assert_eq!(clip.num_frames(), 8);
        assert_eq!(clip.frame_height, 4);
        assert_eq!(clip.frame_width, 4);
    }

    #[test]
    fn test_video_clip_num_frames() {
        let clip = make_clip(16, 4, 4);
        assert_eq!(clip.num_frames(), 16);
    }

    #[test]
    fn test_video_clip_duration() {
        let clip = make_clip(25, 4, 4); // 25 frames at 25 fps = 1 second
        let dur = clip.duration_seconds();
        assert!((dur - 1.0).abs() < 0.01, "expected ~1s, got {dur}");
    }

    // --- VideoClip::sample_frames uniform ---

    #[test]
    fn test_sample_frames_uniform_count() {
        let clip = make_clip(32, 4, 4);
        let sampled = clip.sample_frames(8, &FrameSamplingStrategy::Uniform);
        assert_eq!(sampled.len(), 8);
    }

    #[test]
    fn test_sample_frames_uniform_does_not_exceed_clip() {
        let clip = make_clip(4, 4, 4);
        let sampled = clip.sample_frames(100, &FrameSamplingStrategy::Uniform);
        // Should return at most clip.num_frames().
        assert!(sampled.len() <= 4);
    }

    // --- VideoClip::sample_frames center ---

    #[test]
    fn test_sample_frames_center_count() {
        let clip = make_clip(20, 4, 4);
        let sampled = clip.sample_frames(6, &FrameSamplingStrategy::Center);
        assert_eq!(sampled.len(), 6);
    }

    #[test]
    fn test_sample_frames_center_is_centered() {
        // 20 frames, take 2: should be indices 9 and 10 (middle two).
        let clip = make_clip(20, 1, 1);
        // The frames have values i/20 so we can check the mean.
        let sampled = clip.sample_frames(2, &FrameSamplingStrategy::Center);
        assert_eq!(sampled.len(), 2);
        // Values should be near the middle of the clip.
        let mid_val = sampled[0][0];
        assert!(
            mid_val > 0.3 && mid_val < 0.7,
            "center sample should be near middle, got {mid_val}"
        );
    }

    // --- VideoClip::frame_at ---

    #[test]
    fn test_frame_at_some() {
        let clip = make_clip(4, 2, 2);
        assert!(clip.frame_at(0).is_some());
        assert!(clip.frame_at(3).is_some());
    }

    #[test]
    fn test_frame_at_none() {
        let clip = make_clip(4, 2, 2);
        assert!(clip.frame_at(10).is_none());
    }

    // --- VideoClip::mean_frame ---

    #[test]
    fn test_mean_frame_dimensions() {
        let clip = make_clip(8, 4, 4);
        let mf = clip.mean_frame();
        assert_eq!(mf.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_mean_frame_correct_values() {
        // Two frames: all 0.0 and all 1.0 → mean should be 0.5.
        let h = 2;
        let w = 2;
        let frame_size = h * w * 3;
        let frames = vec![vec![0.0f32; frame_size], vec![1.0f32; frame_size]];
        let clip = VideoClip::new(frames, h, w, 10.0).expect("ok");
        let mf = clip.mean_frame();
        for &v in &mf {
            assert!((v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
        }
    }

    // --- VideoClassificationPipeline::classify ---

    #[test]
    fn test_classify_result_has_labels() {
        let pipeline = default_pipeline();
        let clip = make_clip(20, 4, 4);
        let result = pipeline.classify(&clip).expect("ok");
        assert!(!result.label.is_empty(), "label should not be empty");
        assert!(!result.top_labels.is_empty(), "top_labels should not be empty");
    }

    // --- VideoClassificationPipeline::classify_frames ---

    #[test]
    fn test_classify_frames_basic() {
        let pipeline = default_pipeline();
        let frames = vec![vec![0.5f32; 4 * 4 * 3]; 4];
        let result = pipeline.classify_frames(&frames).expect("ok");
        assert_eq!(result.frames_processed, 4);
        assert!(result.score > 0.0);
    }

    // --- VideoClassificationPipeline::classify_batch ---

    #[test]
    fn test_classify_batch_count() {
        let pipeline = default_pipeline();
        let clip1 = make_clip(8, 4, 4);
        let clip2 = make_clip(16, 4, 4);
        let clip3 = make_clip(12, 4, 4);
        let batch: Vec<&VideoClip> = vec![&clip1, &clip2, &clip3];
        let results = pipeline.classify_batch(&batch).expect("ok");
        assert_eq!(results.len(), 3);
    }

    // --- VideoClassificationResult::top_k ---

    #[test]
    fn test_top_k_result_ordering() {
        let pipeline = default_pipeline();
        let clip = make_clip(16, 4, 4);
        let result = pipeline.classify(&clip).expect("ok");
        // top_labels should be sorted descending.
        let scores: Vec<f32> = result.top_labels.iter().map(|(_, s)| *s).collect();
        for window in scores.windows(2) {
            assert!(
                window[0] >= window[1],
                "top_labels should be sorted descending: {} < {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_top_k_limits_results() {
        let pipeline = default_pipeline();
        let clip = make_clip(16, 4, 4);
        let result = pipeline.classify(&clip).expect("ok");
        let top3 = result.top_k(3);
        assert!(top3.len() <= 3);
    }

    // --- Empty video error ---

    #[test]
    fn test_empty_video_error() {
        let pipeline = default_pipeline();
        // Can't construct an empty VideoClip through the constructor, so test classify_frames.
        let err = pipeline.classify_frames(&[]).unwrap_err();
        assert!(matches!(err, VideoError::EmptyVideo));
    }

    // --- Invalid FPS error ---

    #[test]
    fn test_invalid_fps_error() {
        let frame = vec![0.5f32; 4 * 4 * 3];
        let err = VideoClip::new(vec![frame], 4, 4, 0.0).unwrap_err();
        assert!(matches!(err, VideoError::InvalidFps(_)));
    }

    #[test]
    fn test_negative_fps_error() {
        let frame = vec![0.5f32; 4 * 4 * 3];
        let err = VideoClip::new(vec![frame], 4, 4, -5.0).unwrap_err();
        assert!(matches!(err, VideoError::InvalidFps(_)));
    }

    // --- FrameSamplingStrategy variants ---

    #[test]
    fn test_frame_sampling_strategy_variants() {
        assert_eq!(FrameSamplingStrategy::Uniform, FrameSamplingStrategy::Uniform);
        assert_ne!(FrameSamplingStrategy::Uniform, FrameSamplingStrategy::Center);
        assert_ne!(
            FrameSamplingStrategy::Random { seed: 42 },
            FrameSamplingStrategy::Random { seed: 99 }
        );
        assert_eq!(
            FrameSamplingStrategy::Random { seed: 7 },
            FrameSamplingStrategy::Random { seed: 7 }
        );
        // All four variants constructable.
        let _variants = [
            FrameSamplingStrategy::Uniform,
            FrameSamplingStrategy::Center,
            FrameSamplingStrategy::Random { seed: 0 },
            FrameSamplingStrategy::MotionBased,
        ];
    }

    // --- Scores sum to approximately 1.0 ---

    #[test]
    fn test_scores_sum_to_one() {
        let pipeline = default_pipeline();
        let clip = make_clip(16, 4, 4);
        let result = pipeline.classify(&clip).expect("ok");
        // All labels' scores (not just top-k) should sum to ~1.0.
        // We can only check top_labels here; verify they're valid probabilities.
        let sum: f32 = result.top_labels.iter().map(|(_, s)| *s).sum();
        // Sum of top-k probabilities must be in (0, 1].
        assert!(sum > 0.0 && sum <= 1.0 + 1e-4, "probability sum out of range: {sum}");
    }

    // ── VideoFrame ────────────────────────────────────────────────────────────

    #[test]
    fn test_video_frame_construction() {
        let pixels = vec![128u8; 4 * 4 * 3];
        let frame = VideoFrame::new(pixels.clone(), 4, 4, 0.0);
        assert_eq!(frame.width, 4);
        assert_eq!(frame.height, 4);
        assert_eq!(frame.timestamp_ms, 0.0);
        assert_eq!(frame.pixels, pixels);
    }

    #[test]
    fn test_video_frame_num_pixels() {
        let frame = VideoFrame::new(vec![0u8; 6 * 8 * 3], 8, 6, 100.0);
        assert_eq!(frame.num_pixels(), 48);
    }

    // ── VideoInput ────────────────────────────────────────────────────────────

    #[test]
    fn test_video_input_construction() {
        let frames: Vec<VideoFrame> = (0..5)
            .map(|i| VideoFrame::new(vec![0u8; 4 * 4 * 3], 4, 4, i as f32 * 40.0))
            .collect();
        let video = VideoInput::new(frames, 25.0, 200.0);
        assert_eq!(video.num_frames(), 5);
        assert_eq!(video.fps, 25.0);
        assert_eq!(video.duration_ms, 200.0);
    }

    #[test]
    fn test_video_input_frame_timestamps() {
        let frames: Vec<VideoFrame> = (0..3)
            .map(|i| VideoFrame::new(vec![0u8; 2 * 2 * 3], 2, 2, i as f32 * 100.0))
            .collect();
        let video = VideoInput::new(frames, 10.0, 300.0);
        assert_eq!(video.frames[0].timestamp_ms, 0.0);
        assert_eq!(video.frames[1].timestamp_ms, 100.0);
        assert_eq!(video.frames[2].timestamp_ms, 200.0);
    }

    // ── VideoFeatureExtractor::sample_frames_uniform ─────────────────────────

    #[test]
    fn test_sample_frames_uniform_basic() {
        let indices = VideoFeatureExtractor::sample_frames_uniform(100, 10);
        assert_eq!(indices.len(), 10);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[9], 99);
    }

    #[test]
    fn test_sample_frames_uniform_single() {
        let indices = VideoFeatureExtractor::sample_frames_uniform(50, 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_sample_frames_uniform_more_than_total() {
        // Requesting more frames than available → clamp to total.
        let indices = VideoFeatureExtractor::sample_frames_uniform(5, 20);
        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_sample_frames_uniform_zero_total() {
        assert!(VideoFeatureExtractor::sample_frames_uniform(0, 10).is_empty());
    }

    #[test]
    fn test_sample_frames_uniform_all_in_range() {
        let total = 30;
        let indices = VideoFeatureExtractor::sample_frames_uniform(total, 15);
        for &idx in &indices {
            assert!(idx < total, "index {idx} out of range [0, {total})");
        }
    }

    // ── VideoFeatureExtractor::sample_frames_center_crop ─────────────────────

    #[test]
    fn test_sample_frames_center_crop_basic() {
        let indices = VideoFeatureExtractor::sample_frames_center_crop(100, 8, 20);
        assert_eq!(indices.len(), 8);
    }

    #[test]
    fn test_sample_frames_center_crop_in_center() {
        // 100 frames, clip=20 centered at 50 → frames in [40, 60).
        let indices = VideoFeatureExtractor::sample_frames_center_crop(100, 4, 20);
        for &idx in &indices {
            assert!(idx >= 30 && idx < 70, "center crop index {idx} outside expected range");
        }
    }

    #[test]
    fn test_sample_frames_center_crop_zero_frames() {
        let indices = VideoFeatureExtractor::sample_frames_center_crop(0, 5, 10);
        assert!(indices.is_empty());
    }

    // ── VideoFeatureExtractor::temporal_pool ─────────────────────────────────

    #[test]
    fn test_temporal_pool_mean() {
        let frames = vec![
            vec![0.0_f32, 2.0],
            vec![2.0_f32, 4.0],
        ];
        let out = VideoFeatureExtractor::temporal_pool(&frames, &TemporalPoolType::Mean);
        assert!((out[0] - 1.0).abs() < 1e-5, "mean pool [0]: {}", out[0]);
        assert!((out[1] - 3.0).abs() < 1e-5, "mean pool [1]: {}", out[1]);
    }

    #[test]
    fn test_temporal_pool_max() {
        let frames = vec![
            vec![1.0_f32, 5.0],
            vec![3.0_f32, 2.0],
            vec![2.0_f32, 4.0],
        ];
        let out = VideoFeatureExtractor::temporal_pool(&frames, &TemporalPoolType::Max);
        assert!((out[0] - 3.0).abs() < 1e-5);
        assert!((out[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_temporal_pool_last() {
        let frames = vec![
            vec![1.0_f32, 1.0],
            vec![2.0_f32, 2.0],
            vec![9.0_f32, 8.0],
        ];
        let out = VideoFeatureExtractor::temporal_pool(&frames, &TemporalPoolType::Last);
        assert!((out[0] - 9.0).abs() < 1e-5);
        assert!((out[1] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_temporal_pool_weighted_mean() {
        let frames = vec![
            vec![0.0_f32, 0.0],
            vec![10.0_f32, 10.0],
        ];
        // Weight 0 for frame 0, weight 1 for frame 1 → output = frame 1.
        let weights = vec![0.0_f32, 1.0];
        let out = VideoFeatureExtractor::temporal_pool(
            &frames,
            &TemporalPoolType::WeightedMean(weights),
        );
        assert!((out[0] - 10.0).abs() < 1e-5);
        assert!((out[1] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_temporal_pool_empty() {
        let out = VideoFeatureExtractor::temporal_pool(&[], &TemporalPoolType::Mean);
        assert!(out.is_empty());
    }

    // ── VideoFeatureExtractor::optical_flow_magnitude ────────────────────────

    #[test]
    fn test_optical_flow_magnitude_identical_frames() {
        let frame = vec![128u8; 4 * 4 * 3];
        let mag = VideoFeatureExtractor::optical_flow_magnitude(&frame, &frame, 4, 4);
        assert_eq!(mag.len(), 16);
        assert!(mag.iter().all(|&m| m < 1e-5), "identical frames should have zero flow");
    }

    #[test]
    fn test_optical_flow_magnitude_different_frames() {
        let frame1 = vec![0u8; 4 * 4 * 3];
        let frame2 = vec![255u8; 4 * 4 * 3];
        let mag = VideoFeatureExtractor::optical_flow_magnitude(&frame1, &frame2, 4, 4);
        assert_eq!(mag.len(), 16);
        // sqrt(3 * 255^2) ≈ 441.67
        for &m in &mag {
            assert!(m > 400.0, "max-diff frames should have high flow: {m}");
        }
    }

    #[test]
    fn test_optical_flow_magnitude_output_length() {
        let f1 = vec![0u8; 6 * 8 * 3];
        let f2 = vec![100u8; 6 * 8 * 3];
        let mag = VideoFeatureExtractor::optical_flow_magnitude(&f1, &f2, 8, 6);
        assert_eq!(mag.len(), 8 * 6, "output should have w*h entries");
    }

    // ── TemporalPoolType variants ─────────────────────────────────────────────

    #[test]
    fn test_temporal_pool_type_variants() {
        assert_eq!(TemporalPoolType::Mean, TemporalPoolType::Mean);
        assert_ne!(TemporalPoolType::Mean, TemporalPoolType::Max);
        assert_ne!(TemporalPoolType::Last, TemporalPoolType::Mean);
        // WeightedMean equality requires same weights
        assert_eq!(
            TemporalPoolType::WeightedMean(vec![0.5, 0.5]),
            TemporalPoolType::WeightedMean(vec![0.5, 0.5])
        );
        assert_ne!(
            TemporalPoolType::WeightedMean(vec![1.0]),
            TemporalPoolType::WeightedMean(vec![0.5])
        );
    }
}
