//! # Real image preprocessing for the vision pipelines
//!
//! Pure-Rust decoding, resizing and normalisation. Every function computes the
//! pixel arithmetic it advertises — no zero-filled placeholders, and no
//! silently-substituted black images when a decoder is missing.
//!
//! - [`decode_image_bytes`] / [`decode_image_file`] decode Netpbm (`P5`/`P6`,
//!   binary PGM/PPM) unconditionally, and — with the `vision` feature enabled —
//!   any format the pure-Rust `image` crate supports (PNG, JPEG, BMP, TIFF, …).
//!   Without `vision`, a non-Netpbm file produces a structured
//!   [`TrustformersError::FeatureUnavailable`] naming the feature to enable.
//! - [`resize_bilinear`] performs real bilinear interpolation with
//!   `align_corners = false` half-pixel centres (the torchvision / HF
//!   `Resize` convention).
//! - [`center_crop`] and [`normalize_to_chw`] complete the standard
//!   ViT/CLIP preprocessing chain.

use crate::error::{Result, TrustformersError};
use std::path::Path;

/// An RGB image with `f32` channel values in `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct RgbImage {
    /// Row-major pixel data, `height * width * 3` values in `[0, 1]`.
    pub data: Vec<f32>,
    /// Image height in pixels.
    pub height: usize,
    /// Image width in pixels.
    pub width: usize,
}

impl RgbImage {
    /// Build an image from row-major RGB data.
    ///
    /// # Errors
    ///
    /// Returns an error when `data.len() != height * width * 3` or either
    /// dimension is zero.
    pub fn new(data: Vec<f32>, height: usize, width: usize) -> Result<Self> {
        if height == 0 || width == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "image dimensions must be non-zero".to_string(),
            ));
        }
        if data.len() != height * width * 3 {
            return Err(TrustformersError::invalid_input_simple(format!(
                "image buffer has {} values but {height}x{width}x3 = {} were expected",
                data.len(),
                height * width * 3
            )));
        }
        Ok(Self {
            data,
            height,
            width,
        })
    }

    /// Read the RGB triple at `(y, x)`.
    ///
    /// # Panics
    ///
    /// Panics if the coordinates are out of bounds; callers inside this module
    /// always bound them first.
    #[inline]
    fn pixel(&self, y: usize, x: usize) -> [f32; 3] {
        let base = (y * self.width + x) * 3;
        [self.data[base], self.data[base + 1], self.data[base + 2]]
    }
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Decode a binary Netpbm image (`P5` grayscale / `P6` RGB, 8-bit maxval).
///
/// Netpbm is trivially parseable and needs no external crate, which makes it
/// the always-available decoding path (and the fixture format for tests).
///
/// # Errors
///
/// Returns an error for malformed headers, unsupported magic numbers, a maxval
/// other than 255, or truncated pixel data.
pub fn decode_netpbm(bytes: &[u8]) -> Result<RgbImage> {
    let bad = |what: &str| TrustformersError::invalid_input_simple(format!("Netpbm: {what}"));

    if bytes.len() < 2 {
        return Err(bad("stream too short for a magic number"));
    }
    let magic = &bytes[0..2];
    let channels = match magic {
        b"P5" => 1usize,
        b"P6" => 3usize,
        _ => {
            return Err(bad(
                "unsupported magic number (only binary P5/P6 are handled here)",
            ))
        },
    };

    // Header: three ASCII integers (width, height, maxval), whitespace-separated,
    // with `#`-to-end-of-line comments allowed.
    let mut cursor = 2usize;
    let mut values = [0usize; 3];
    for slot in values.iter_mut() {
        // Skip whitespace and comments.
        loop {
            match bytes.get(cursor) {
                None => return Err(bad("truncated header")),
                Some(&b'#') => {
                    while matches!(bytes.get(cursor), Some(&c) if c != b'\n') {
                        cursor += 1;
                    }
                },
                Some(&c) if c.is_ascii_whitespace() => cursor += 1,
                Some(_) => break,
            }
        }
        let start = cursor;
        while matches!(bytes.get(cursor), Some(&c) if c.is_ascii_digit()) {
            cursor += 1;
        }
        if start == cursor {
            return Err(bad("expected an integer in the header"));
        }
        let text = std::str::from_utf8(&bytes[start..cursor])
            .map_err(|_| bad("non-UTF-8 header field"))?;
        *slot = text.parse::<usize>().map_err(|_| bad("header field is not a number"))?;
    }
    // Exactly one whitespace byte separates the header from the raster.
    cursor += 1;

    let (width, height, maxval) = (values[0], values[1], values[2]);
    if maxval != 255 {
        return Err(bad("only 8-bit images (maxval 255) are supported"));
    }
    if width == 0 || height == 0 {
        return Err(bad("zero-sized image"));
    }
    let expected = width * height * channels;
    let raster = bytes
        .get(cursor..cursor + expected)
        .ok_or_else(|| bad("truncated pixel data"))?;

    let mut data = Vec::with_capacity(width * height * 3);
    for pixel in raster.chunks_exact(channels) {
        if channels == 1 {
            let v = f32::from(pixel[0]) / 255.0;
            data.extend_from_slice(&[v, v, v]);
        } else {
            data.push(f32::from(pixel[0]) / 255.0);
            data.push(f32::from(pixel[1]) / 255.0);
            data.push(f32::from(pixel[2]) / 255.0);
        }
    }

    RgbImage::new(data, height, width)
}

/// Encode an [`RgbImage`] as a binary P6 PPM byte stream.
///
/// The exact inverse of the `P6` branch of [`decode_netpbm`] up to 8-bit
/// quantisation; used by callers that need an interchange buffer and by this
/// module's round-trip tests.
pub fn encode_ppm(image: &RgbImage) -> Vec<u8> {
    let mut out = format!("P6\n{} {}\n255\n", image.width, image.height).into_bytes();
    out.reserve(image.data.len());
    for &v in &image.data {
        out.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
    }
    out
}

/// Decode an image from an in-memory byte buffer.
///
/// Netpbm is always supported. With the `vision` feature enabled, everything
/// else is delegated to the pure-Rust `image` crate.
///
/// # Errors
///
/// Returns [`TrustformersError::FeatureUnavailable`] when the bytes are not
/// Netpbm and the `vision` feature is off, and
/// [`TrustformersError::InvalidInput`] when decoding fails. It never returns a
/// blank image.
pub fn decode_image_bytes(bytes: &[u8]) -> Result<RgbImage> {
    if bytes.starts_with(b"P5") || bytes.starts_with(b"P6") {
        return decode_netpbm(bytes);
    }

    #[cfg(feature = "vision")]
    {
        let dynamic = image::load_from_memory(bytes).map_err(|e| {
            TrustformersError::invalid_input_simple(format!("image decoding failed: {e}"))
        })?;
        dynamic_to_rgb(&dynamic)
    }

    #[cfg(not(feature = "vision"))]
    {
        Err(TrustformersError::feature_unavailable(
            "image decoding for this format requires the `vision` feature (pure-Rust `image` \
             crate); only binary Netpbm (P5/P6) can be decoded without it"
                .to_string(),
            "vision",
        ))
    }
}

#[cfg(feature = "vision")]
fn dynamic_to_rgb(dynamic: &image::DynamicImage) -> Result<RgbImage> {
    use image::GenericImageView;
    let (width, height) = dynamic.dimensions();
    let rgb = dynamic.to_rgb8();
    let data: Vec<f32> = rgb.as_raw().iter().map(|&b| f32::from(b) / 255.0).collect();
    RgbImage::new(data, height as usize, width as usize)
}

/// Decode an image file from disk.
///
/// # Errors
///
/// Returns [`TrustformersError::Io`] when the file cannot be read, and whatever
/// [`decode_image_bytes`] returns otherwise.
pub fn decode_image_file<P: AsRef<Path>>(path: P) -> Result<RgbImage> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| TrustformersError::Io {
        message: format!("failed to read image file: {e}"),
        path: Some(path.display().to_string()),
        suggestion: Some("Check that the file exists and is readable".to_string()),
    })?;
    decode_image_bytes(&bytes)
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// Resize an image with bilinear interpolation.
///
/// Uses half-pixel centres (`align_corners = false`): the source coordinate for
/// destination index `i` is `(i + 0.5) · scale - 0.5`, clamped to the source
/// range. This is the torchvision / HuggingFace `Resize` convention.
///
/// # Errors
///
/// Returns an error when either target dimension is zero.
pub fn resize_bilinear(image: &RgbImage, out_h: usize, out_w: usize) -> Result<RgbImage> {
    if out_h == 0 || out_w == 0 {
        return Err(TrustformersError::invalid_input_simple(
            "resize: target dimensions must be non-zero".to_string(),
        ));
    }
    if out_h == image.height && out_w == image.width {
        return Ok(image.clone());
    }

    let scale_y = image.height as f32 / out_h as f32;
    let scale_x = image.width as f32 / out_w as f32;
    let mut data = Vec::with_capacity(out_h * out_w * 3);

    for oy in 0..out_h {
        let src_y = ((oy as f32 + 0.5) * scale_y - 0.5).max(0.0);
        let y0 = src_y.floor() as usize;
        let y1 = (y0 + 1).min(image.height - 1);
        let wy = src_y - y0 as f32;
        let y0 = y0.min(image.height - 1);

        for ox in 0..out_w {
            let src_x = ((ox as f32 + 0.5) * scale_x - 0.5).max(0.0);
            let x0 = src_x.floor() as usize;
            let x1 = (x0 + 1).min(image.width - 1);
            let wx = src_x - x0 as f32;
            let x0 = x0.min(image.width - 1);

            let p00 = image.pixel(y0, x0);
            let p01 = image.pixel(y0, x1);
            let p10 = image.pixel(y1, x0);
            let p11 = image.pixel(y1, x1);

            for c in 0..3 {
                let top = p00[c] * (1.0 - wx) + p01[c] * wx;
                let bottom = p10[c] * (1.0 - wx) + p11[c] * wx;
                data.push(top * (1.0 - wy) + bottom * wy);
            }
        }
    }

    RgbImage::new(data, out_h, out_w)
}

/// Crop the centre `crop_h × crop_w` region of an image.
///
/// # Errors
///
/// Returns an error when the crop is larger than the image or has a zero
/// dimension.
pub fn center_crop(image: &RgbImage, crop_h: usize, crop_w: usize) -> Result<RgbImage> {
    if crop_h == 0 || crop_w == 0 {
        return Err(TrustformersError::invalid_input_simple(
            "center_crop: crop dimensions must be non-zero".to_string(),
        ));
    }
    if crop_h > image.height || crop_w > image.width {
        return Err(TrustformersError::invalid_input_simple(format!(
            "center_crop: {crop_h}x{crop_w} does not fit inside {}x{}",
            image.height, image.width
        )));
    }
    let top = (image.height - crop_h) / 2;
    let left = (image.width - crop_w) / 2;

    let mut data = Vec::with_capacity(crop_h * crop_w * 3);
    for y in 0..crop_h {
        for x in 0..crop_w {
            data.extend_from_slice(&image.pixel(top + y, left + x));
        }
    }
    RgbImage::new(data, crop_h, crop_w)
}

/// Normalise an image to a flat CHW `f32` buffer: `(pixel - mean) / std`.
///
/// Returns `3 · height · width` values ordered channel-major, ready to be
/// wrapped in a `[1, 3, H, W]` tensor.
///
/// # Errors
///
/// Returns an error when any standard deviation is zero.
pub fn normalize_to_chw(image: &RgbImage, mean: [f32; 3], std: [f32; 3]) -> Result<Vec<f32>> {
    if std.contains(&0.0) {
        return Err(TrustformersError::invalid_input_simple(
            "normalize: standard deviations must be non-zero".to_string(),
        ));
    }
    let pixels = image.height * image.width;
    let mut out = vec![0.0f32; pixels * 3];
    for c in 0..3 {
        for i in 0..pixels {
            out[c * pixels + i] = (image.data[i * 3 + c] - mean[c]) / std[c];
        }
    }
    Ok(out)
}

/// ImageNet channel means used by ViT / ResNet checkpoints.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet channel standard deviations used by ViT / ResNet checkpoints.
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
/// CLIP channel means (OpenAI CLIP preprocessing).
pub const CLIP_MEAN: [f32; 3] = [0.481_454_67, 0.457_827_5, 0.408_210_73];
/// CLIP channel standard deviations (OpenAI CLIP preprocessing).
pub const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Run the standard vision preprocessing chain: resize → centre-crop → normalise.
///
/// The image is first resized so its *shorter* side equals `size`, preserving
/// aspect ratio, then centre-cropped to `size × size` and normalised.
///
/// # Errors
///
/// Propagates errors from the individual stages.
pub fn preprocess_vision(
    image: &RgbImage,
    size: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<Vec<f32>> {
    if size == 0 {
        return Err(TrustformersError::invalid_input_simple(
            "preprocess_vision: size must be non-zero".to_string(),
        ));
    }
    let (resize_h, resize_w) = if image.height <= image.width {
        let scaled = (image.width as f64 * size as f64 / image.height as f64).round() as usize;
        (size, scaled.max(size))
    } else {
        let scaled = (image.height as f64 * size as f64 / image.width as f64).round() as usize;
        (scaled.max(size), size)
    };
    let resized = resize_bilinear(image, resize_h, resize_w)?;
    let cropped = center_crop(&resized, size, size)?;
    normalize_to_chw(&cropped, mean, std)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(h: usize, w: usize) -> RgbImage {
        let mut data = Vec::with_capacity(h * w * 3);
        for y in 0..h {
            for x in 0..w {
                data.push(x as f32 / w as f32);
                data.push(y as f32 / h as f32);
                data.push(0.25);
            }
        }
        RgbImage::new(data, h, w).expect("gradient image")
    }

    // ---- Construction ----

    #[test]
    fn rgb_image_rejects_mismatched_buffer() {
        assert!(RgbImage::new(vec![0.0; 5], 2, 2).is_err());
        assert!(RgbImage::new(vec![0.0; 12], 0, 2).is_err());
    }

    // ---- Netpbm ----

    #[test]
    fn decode_ppm_reads_real_pixels() {
        // 2x1 P6: pure red, pure green.
        let bytes = b"P6\n2 1\n255\n\xff\x00\x00\x00\xff\x00";
        let img = decode_netpbm(bytes).expect("decode");
        assert_eq!((img.height, img.width), (1, 2));
        assert!((img.data[0] - 1.0).abs() < 1e-6);
        assert!((img.data[1] - 0.0).abs() < 1e-6);
        assert!((img.data[4] - 1.0).abs() < 1e-6);
        assert!(
            img.data.iter().any(|&v| v > 0.0),
            "decoded image must not be all zeros"
        );
    }

    #[test]
    fn decode_pgm_expands_grayscale_to_rgb() {
        let bytes = b"P5\n2 1\n255\n\x00\xff";
        let img = decode_netpbm(bytes).expect("decode");
        assert_eq!(img.data.len(), 6);
        assert!((img.data[0] - 0.0).abs() < 1e-6);
        assert!((img.data[3] - 1.0).abs() < 1e-6);
        assert!((img.data[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decode_ppm_handles_comments() {
        let bytes = b"P6\n# a comment\n2 1\n255\n\xff\x00\x00\x00\xff\x00";
        let img = decode_netpbm(bytes).expect("decode with comment");
        assert_eq!(img.width, 2);
    }

    #[test]
    fn decode_ppm_rejects_truncated_raster() {
        let bytes = b"P6\n2 2\n255\n\xff\x00\x00";
        assert!(decode_netpbm(bytes).is_err());
    }

    #[test]
    fn ppm_round_trips() {
        let img = gradient(4, 5);
        let bytes = encode_ppm(&img);
        let back = decode_netpbm(&bytes).expect("round trip");
        assert_eq!((back.height, back.width), (4, 5));
        for (a, b) in back.data.iter().zip(img.data.iter()) {
            assert!((a - b).abs() < 1.0 / 255.0 + 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn decode_image_bytes_dispatches_netpbm() {
        let img = gradient(3, 3);
        let decoded = decode_image_bytes(&encode_ppm(&img)).expect("dispatch");
        assert_eq!(decoded.width, 3);
    }

    #[cfg(not(feature = "vision"))]
    #[test]
    fn decode_image_bytes_reports_missing_vision_feature() {
        // Minimal PNG signature — not Netpbm.
        let err = decode_image_bytes(b"\x89PNG\r\n\x1a\n").expect_err("needs `vision`");
        assert!(err.to_string().contains("vision"), "err: {err}");
    }

    #[test]
    fn decode_image_file_reports_missing_file() {
        let mut path = std::env::temp_dir();
        path.push("trustformers-media-does-not-exist.ppm");
        let err = decode_image_file(&path).expect_err("missing file must error");
        assert!(matches!(err, TrustformersError::Io { .. }));
    }

    #[test]
    fn decode_image_file_reads_a_real_fixture() {
        let mut path = std::env::temp_dir();
        path.push("trustformers-media-fixture.ppm");
        let img = gradient(6, 8);
        std::fs::write(&path, encode_ppm(&img)).expect("write fixture");
        let decoded = decode_image_file(&path).expect("decode fixture");
        assert_eq!((decoded.height, decoded.width), (6, 8));
        assert!(decoded.data.iter().any(|&v| v > 0.0));
        let _ = std::fs::remove_file(&path);
    }

    // ---- Resize ----

    #[test]
    fn resize_bilinear_matches_hand_computed_upsample() {
        // 2x2 single-channel-equivalent image (all channels equal):
        //   0.0 1.0
        //   2.0 3.0
        // Upsampling to 4x4 with half-pixel centres samples the source at
        // y,x = -0.25, 0.25, 0.75, 1.25 → clamped to 0, 0.25, 0.75, 1.0.
        let data: Vec<f32> = [0.0f32, 1.0, 2.0, 3.0].iter().flat_map(|&v| [v, v, v]).collect();
        let img = RgbImage::new(data, 2, 2).expect("2x2");
        let out = resize_bilinear(&img, 4, 4).expect("resize");
        assert_eq!((out.height, out.width), (4, 4));

        let coords = [0.0f32, 0.25, 0.75, 1.0];
        for (oy, &sy) in coords.iter().enumerate() {
            for (ox, &sx) in coords.iter().enumerate() {
                // Bilinear over the 2x2 grid: v = (1-sy)(1-sx)·0 + (1-sy)sx·1
                //                                + sy(1-sx)·2 + sy·sx·3
                let expected = (1.0 - sy) * sx + sy * (1.0 - sx) * 2.0 + sy * sx * 3.0;
                let got = out.data[(oy * 4 + ox) * 3];
                assert!(
                    (got - expected).abs() < 1e-5,
                    "({oy},{ox}): got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn resize_is_identity_for_same_dimensions() {
        let img = gradient(5, 7);
        let out = resize_bilinear(&img, 5, 7).expect("resize");
        assert_eq!(out, img);
    }

    #[test]
    fn resize_downsample_preserves_extremes_ordering() {
        let img = gradient(16, 16);
        let out = resize_bilinear(&img, 4, 4).expect("resize");
        assert_eq!((out.height, out.width), (4, 4));
        // Red channel increases left-to-right in the source; it must stay so.
        for y in 0..4 {
            for x in 0..3 {
                let a = out.data[(y * 4 + x) * 3];
                let b = out.data[(y * 4 + x + 1) * 3];
                assert!(b > a, "red channel must increase along x at row {y}");
            }
        }
    }

    #[test]
    fn resize_rejects_zero_target() {
        let img = gradient(4, 4);
        assert!(resize_bilinear(&img, 0, 4).is_err());
    }

    // ---- Crop / normalise ----

    #[test]
    fn center_crop_takes_the_middle() {
        let img = gradient(4, 4);
        let out = center_crop(&img, 2, 2).expect("crop");
        assert_eq!((out.height, out.width), (2, 2));
        // Top-left of the crop is source pixel (1, 1).
        let expected = img.pixel(1, 1);
        for c in 0..3 {
            assert!((out.data[c] - expected[c]).abs() < 1e-6);
        }
    }

    #[test]
    fn center_crop_rejects_oversized_crop() {
        let img = gradient(4, 4);
        assert!(center_crop(&img, 8, 8).is_err());
    }

    #[test]
    fn normalize_to_chw_reorders_and_scales() {
        let data = vec![0.5f32, 0.25, 0.75, 0.5, 0.25, 0.75];
        let img = RgbImage::new(data, 1, 2).expect("1x2");
        let out = normalize_to_chw(&img, [0.5, 0.25, 0.75], [0.5, 0.5, 0.5]).expect("norm");
        assert_eq!(out.len(), 6);
        for v in &out {
            assert!(v.abs() < 1e-6, "expected zeros after centring, got {v}");
        }
    }

    #[test]
    fn normalize_rejects_zero_std() {
        let img = gradient(2, 2);
        assert!(normalize_to_chw(&img, [0.0; 3], [0.0, 1.0, 1.0]).is_err());
    }

    #[test]
    fn preprocess_vision_produces_distinct_nonzero_vectors() {
        // Regression against the all-zero "preprocessing" the vision feature
        // extractor used to return: two different images must not map to the
        // same vector, and neither may be uniformly zero.
        let a = gradient(30, 40);
        let mut b_data = a.data.clone();
        b_data.reverse();
        let b = RgbImage::new(b_data, 30, 40).expect("reversed");

        let va = preprocess_vision(&a, 8, IMAGENET_MEAN, IMAGENET_STD).expect("a");
        let vb = preprocess_vision(&b, 8, IMAGENET_MEAN, IMAGENET_STD).expect("b");
        assert_eq!(va.len(), 3 * 8 * 8);
        assert!(
            va.iter().any(|&v| v != 0.0),
            "preprocessed vector is all zeros"
        );
        assert!(va != vb, "different images must preprocess differently");
    }

    #[test]
    fn preprocess_vision_rejects_zero_size() {
        let img = gradient(4, 4);
        assert!(preprocess_vision(&img, 0, IMAGENET_MEAN, IMAGENET_STD).is_err());
    }
}
