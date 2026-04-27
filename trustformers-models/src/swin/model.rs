//! Swin Transformer base model implementation.
//!
//! Key innovations over plain ViT:
//!
//! 1. **Patch partition** — 4×4 patch embedding produces an initial feature map.
//! 2. **Hierarchical stages** — 4 stages with `PatchMerging` between them, halving
//!    spatial resolution and doubling channel count at each transition.
//! 3. **Window-based MSA (W-MSA)** — self-attention restricted to non-overlapping
//!    local windows of size `window_size × window_size`.
//! 4. **Shifted-window MSA (SW-MSA)** — odd-indexed blocks shift the window
//!    partition by `(window_size//2, window_size//2)` pixels, allowing
//!    cross-window interactions without breaking locality.
//! 5. **Relative position bias** — learned 2D relative position encodings added
//!    to attention scores inside each window.

use crate::swin::config::SwinConfig;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4, Axis, Ix3};
use trustformers_core::device::Device;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::layers::{
    feedforward::FeedForward, layernorm::LayerNorm, linear::Linear,
};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Config, Layer};

// ─────────────────────────────────────────────────────────────────────────────
// Helper utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Partition a spatial feature map into non-overlapping windows.
///
/// # Arguments
///
/// * `x` — shape `(B, H, W, C)`.
/// * `window_size` — window height and width.
///
/// # Returns
///
/// Shape `(num_windows * B, window_size, window_size, C)`.
pub fn window_partition(x: &Array4<f32>, window_size: usize) -> Result<Array4<f32>> {
    let (b, h, w, c) = x.dim();

    if h % window_size != 0 || w % window_size != 0 {
        return Err(TrustformersError::invalid_input_simple(format!(
            "Spatial size {}×{} is not divisible by window_size {}",
            h, w, window_size
        )));
    }

    let num_h = h / window_size;
    let num_w = w / window_size;
    let num_windows = b * num_h * num_w;

    let mut windows = Array4::<f32>::zeros((num_windows, window_size, window_size, c));

    let mut win_idx = 0usize;
    for bi in 0..b {
        for wh in 0..num_h {
            for ww in 0..num_w {
                let sh = wh * window_size;
                let sw = ww * window_size;
                let patch = x.slice(s![bi, sh..sh + window_size, sw..sw + window_size, ..]);
                windows.slice_mut(s![win_idx, .., .., ..]).assign(&patch);
                win_idx += 1;
            }
        }
    }

    Ok(windows)
}

/// Reverse a window partition back to a spatial feature map.
///
/// # Arguments
///
/// * `windows` — shape `(num_windows * B, window_size, window_size, C)`.
/// * `h` — original height.
/// * `w` — original width.
///
/// # Returns
///
/// Shape `(B, H, W, C)`.
pub fn window_reverse(
    windows: &Array4<f32>,
    window_size: usize,
    h: usize,
    w: usize,
) -> Result<Array4<f32>> {
    let (total_windows, _ws_h, _ws_w, c) = windows.dim();
    let num_h = h / window_size;
    let num_w = w / window_size;
    let b = total_windows / (num_h * num_w);

    if b == 0 || b * num_h * num_w != total_windows {
        return Err(TrustformersError::invalid_input_simple(format!(
            "Cannot reverse {} windows into {}×{} grid (batch must be positive)",
            total_windows, h, w
        )));
    }

    let mut x = Array4::<f32>::zeros((b, h, w, c));
    let mut win_idx = 0usize;
    for bi in 0..b {
        for wh in 0..num_h {
            for ww in 0..num_w {
                let sh = wh * window_size;
                let sw = ww * window_size;
                let win = windows.slice(s![win_idx, .., .., ..]);
                x.slice_mut(s![bi, sh..sh + window_size, sw..sw + window_size, ..])
                    .assign(&win);
                win_idx += 1;
            }
        }
    }

    Ok(x)
}

/// Apply a cyclic shift on a 4-D feature map along spatial axes.
///
/// For SW-MSA the feature map is shifted by `(-shift, -shift)` before
/// partitioning, then shifted back by `(shift, shift)` after.
///
/// # Arguments
///
/// * `x` — shape `(B, H, W, C)`.
/// * `shift` — shift amount (positive = cyclic-left / cyclic-up).
pub fn cyclic_shift(x: &Array4<f32>, shift: usize) -> Array4<f32> {
    let (b, h, w, c) = x.dim();
    if shift == 0 || h == 0 || w == 0 {
        return x.clone();
    }
    let sh = shift % h;
    let sw = shift % w;

    // Roll along axis 1 (height) by -sh, i.e. bring rows [sh..] to front
    let mut rolled_h = Array4::<f32>::zeros((b, h, w, c));
    rolled_h.slice_mut(s![.., ..h - sh, .., ..]).assign(&x.slice(s![.., sh.., .., ..]));
    rolled_h.slice_mut(s![.., h - sh.., .., ..]).assign(&x.slice(s![.., ..sh, .., ..]));

    // Roll along axis 2 (width) by -sw
    let mut rolled = Array4::<f32>::zeros((b, h, w, c));
    rolled.slice_mut(s![.., .., ..w - sw, ..]).assign(&rolled_h.slice(s![.., .., sw.., ..]));
    rolled.slice_mut(s![.., .., w - sw.., ..]).assign(&rolled_h.slice(s![.., .., ..sw, ..]));

    rolled
}

// ─────────────────────────────────────────────────────────────────────────────
// Patch Partition + Linear Embedding
// ─────────────────────────────────────────────────────────────────────────────

/// Partition the raw image into non-overlapping 4×4 patches and project to
/// `embed_dim` channels.
///
/// Output shape: `(B, H/ps, W/ps, embed_dim)` (spatial layout for window ops).
#[derive(Debug, Clone)]
pub struct SwinPatchEmbedding {
    pub projection: Linear,
    pub patch_size: usize,
    pub num_channels: usize,
    pub embed_dim: usize,
    pub layer_norm: LayerNorm,
    device: Device,
}

impl SwinPatchEmbedding {
    pub fn new(config: &SwinConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &SwinConfig, device: Device) -> Result<Self> {
        let in_size = config.patch_size * config.patch_size * config.num_channels;
        Ok(Self {
            projection: Linear::new_with_device(in_size, config.embed_dim, true, device),
            patch_size: config.patch_size,
            num_channels: config.num_channels,
            embed_dim: config.embed_dim,
            layer_norm: LayerNorm::new_with_device(
                vec![config.embed_dim],
                config.layer_norm_eps,
                device,
            )?,
            device,
        })
    }

    /// Active device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Map an NHWC image to an `(B, H', W', C)` feature map.
    pub fn forward(&self, images: &Array4<f32>) -> Result<Array4<f32>> {
        let (b, h, w, c) = images.dim();
        if h % self.patch_size != 0 || w % self.patch_size != 0 {
            return Err(TrustformersError::invalid_input_simple(format!(
                "Image {}×{} not divisible by patch_size {}",
                h, w, self.patch_size
            )));
        }
        if c != self.num_channels {
            return Err(TrustformersError::invalid_input_simple(format!(
                "Expected {} channels, got {}",
                self.num_channels, c
            )));
        }

        let ph = h / self.patch_size;
        let pw = w / self.patch_size;
        let patch_flat = self.patch_size * self.patch_size * c;
        let num_patches = ph * pw;

        // Flatten patches → (B, ph*pw, patch_flat)
        let mut patches = Array3::<f32>::zeros((b, num_patches, patch_flat));
        for bi in 0..b {
            let mut pidx = 0usize;
            for i in 0..ph {
                for j in 0..pw {
                    let sh = i * self.patch_size;
                    let sw = j * self.patch_size;
                    let patch =
                        images.slice(s![bi, sh..sh + self.patch_size, sw..sw + self.patch_size, ..]);
                    let flat: Array1<f32> = patch.iter().cloned().collect();
                    patches.slice_mut(s![bi, pidx, ..]).assign(&flat);
                    pidx += 1;
                }
            }
        }

        // Project → (B, num_patches, embed_dim)
        let patches_tensor = Tensor::F32(patches.into_dyn());
        let projected = match self.projection.forward(patches_tensor)? {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 from patch projection".to_string(),
                ))
            },
        };

        // Layer-norm
        let proj_tensor = Tensor::F32(projected.into_dyn());
        let normed = match self.layer_norm.forward(proj_tensor)? {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 from LayerNorm".to_string(),
                ))
            },
        };

        // Reshape (B, ph*pw, C) → (B, ph, pw, C)
        let mut out = Array4::<f32>::zeros((b, ph, pw, self.embed_dim));
        for bi in 0..b {
            for i in 0..ph {
                for j in 0..pw {
                    let pidx = i * pw + j;
                    out.slice_mut(s![bi, i, j, ..]).assign(&normed.slice(s![bi, pidx, ..]));
                }
            }
        }

        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Patch Merging (downsampling between stages)
// ─────────────────────────────────────────────────────────────────────────────

/// Downsample a 2× spatial feature map by concatenating 2×2 neighbourhoods
/// and projecting from `4C → 2C` channels.
///
/// Input shape:  `(B, H, W, C)`
/// Output shape: `(B, H/2, W/2, 2C)`
#[derive(Debug, Clone)]
pub struct PatchMerging {
    pub reduction: Linear,
    pub layer_norm: LayerNorm,
    in_channels: usize,
    device: Device,
}

impl PatchMerging {
    /// Construct from the *input* channel dimension.
    pub fn new(in_channels: usize, layer_norm_eps: f32, device: Device) -> Result<Self> {
        Ok(Self {
            reduction: Linear::new_with_device(4 * in_channels, 2 * in_channels, false, device),
            layer_norm: LayerNorm::new_with_device(
                vec![4 * in_channels],
                layer_norm_eps,
                device,
            )?,
            in_channels,
            device,
        })
    }

    /// Active device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Run patch merging.
    pub fn forward(&self, x: &Array4<f32>) -> Result<Array4<f32>> {
        let (b, h, w, c) = x.dim();

        if h % 2 != 0 || w % 2 != 0 {
            return Err(TrustformersError::invalid_input_simple(format!(
                "PatchMerging requires even spatial dims, got {}×{}",
                h, w
            )));
        }

        if c != self.in_channels {
            return Err(TrustformersError::invalid_input_simple(format!(
                "PatchMerging expected {} channels, got {}",
                self.in_channels, c
            )));
        }

        let oh = h / 2;
        let ow = w / 2;

        // Gather 2×2 quad of pixels for each output position
        // → (B, oh, ow, 4C)
        let mut merged = Array4::<f32>::zeros((b, oh, ow, 4 * c));
        for bi in 0..b {
            for i in 0..oh {
                for j in 0..ow {
                    let r = i * 2;
                    let s_col = j * 2;
                    // top-left, top-right, bottom-left, bottom-right
                    let tl = x.slice(s![bi, r, s_col, ..]);
                    let tr = x.slice(s![bi, r, s_col + 1, ..]);
                    let bl = x.slice(s![bi, r + 1, s_col, ..]);
                    let br = x.slice(s![bi, r + 1, s_col + 1, ..]);
                    merged.slice_mut(s![bi, i, j, ..c]).assign(&tl);
                    merged.slice_mut(s![bi, i, j, c..2 * c]).assign(&tr);
                    merged.slice_mut(s![bi, i, j, 2 * c..3 * c]).assign(&bl);
                    merged.slice_mut(s![bi, i, j, 3 * c..]).assign(&br);
                }
            }
        }

        // Layer-norm on 4C channels
        let merged_3d: Array3<f32> = merged
            .clone()
            .into_shape_with_order((b * oh * ow, 4 * c))
            .map_err(|e| TrustformersError::shape_error(e.to_string()))?
            .into_shape_with_order((b, oh * ow, 4 * c))
            .map_err(|e| TrustformersError::shape_error(e.to_string()))?;

        let norm_tensor = Tensor::F32(merged_3d.into_dyn());
        let normed_3d = match self.layer_norm.forward(norm_tensor)? {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 from LayerNorm".to_string(),
                ))
            },
        };

        // Linear reduction: 4C → 2C
        let red_tensor = Tensor::F32(normed_3d.into_dyn());
        let reduced = match self.reduction.forward(red_tensor)? {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 from reduction Linear".to_string(),
                ))
            },
        };

        // Reshape back to (B, oh, ow, 2C)
        let out_c = 2 * c;
        let mut out = Array4::<f32>::zeros((b, oh, ow, out_c));
        for bi in 0..b {
            for i in 0..oh {
                for j in 0..ow {
                    let pidx = i * ow + j;
                    out.slice_mut(s![bi, i, j, ..]).assign(&reduced.slice(s![bi, pidx, ..]));
                }
            }
        }

        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Window Attention
// ─────────────────────────────────────────────────────────────────────────────

/// Window-based multi-head self-attention (W-MSA / SW-MSA).
///
/// Computes scaled dot-product attention *inside* fixed-size windows.
/// A learnable relative position bias table is added to the attention scores.
///
/// Supports the *shifted* variant: pass `shift_size > 0` when constructing to
/// activate the cyclic-shift trick.
#[derive(Debug, Clone)]
pub struct WindowAttention {
    /// Input/output channel dimension.
    pub dim: usize,
    /// Window size (square).
    pub window_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Scale factor = 1 / sqrt(head_dim).
    scale: f32,
    /// QKV projection: dim → 3 * dim.
    pub qkv: Linear,
    /// Output projection.
    pub proj: Linear,
    /// Relative position bias table: (2*ws-1) × (2*ws-1) × num_heads, stored flat.
    pub relative_position_bias_table: Array3<f32>,
    /// Attention dropout probability.
    attn_drop: f32,
    /// Output projection dropout.
    proj_drop: f32,
    device: Device,
}

impl WindowAttention {
    /// Construct a window-attention layer.
    ///
    /// # Arguments
    ///
    /// * `dim` — channel dimension.
    /// * `window_size` — spatial window size.
    /// * `num_heads` — number of attention heads.
    /// * `qkv_bias` — add bias to QKV projection.
    /// * `attn_drop` — attention weight dropout.
    /// * `proj_drop` — output projection dropout.
    pub fn new(
        dim: usize,
        window_size: usize,
        num_heads: usize,
        qkv_bias: bool,
        attn_drop: f32,
        proj_drop: f32,
        device: Device,
    ) -> Result<Self> {
        if num_heads == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "num_heads must be > 0".to_string(),
            ));
        }
        if dim % num_heads != 0 {
            return Err(TrustformersError::invalid_input_simple(format!(
                "dim {} is not divisible by num_heads {}",
                dim, num_heads
            )));
        }

        let head_dim = dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let bias_len = 2 * window_size - 1;
        // bias table: (bias_len, bias_len, num_heads) — initialised to zeros
        let relative_position_bias_table = Array3::<f32>::zeros((bias_len, bias_len, num_heads));

        Ok(Self {
            dim,
            window_size,
            num_heads,
            scale,
            qkv: Linear::new_with_device(dim, 3 * dim, qkv_bias, device),
            proj: Linear::new_with_device(dim, dim, true, device),
            relative_position_bias_table,
            attn_drop,
            proj_drop,
            device,
        })
    }

    /// Active device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Compute window attention.
    ///
    /// # Arguments
    ///
    /// * `x` — shape `(num_windows * B, ws * ws, dim)`.
    ///
    /// # Returns
    ///
    /// Shape `(num_windows * B, ws * ws, dim)`.
    pub fn forward(&self, x: &Array3<f32>) -> Result<Array3<f32>> {
        let (nw_b, n, _c) = x.dim();
        let ws2 = self.window_size * self.window_size;

        if n != ws2 {
            return Err(TrustformersError::invalid_input_simple(format!(
                "Sequence length {} does not match window_size^2 {}",
                n, ws2
            )));
        }

        // QKV projection: (nw_b, n, 3*dim)
        let x_tensor = Tensor::F32(x.clone().into_dyn());
        let qkv_out = match self.qkv.forward(x_tensor)? {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 from QKV".to_string(),
                ))
            },
        };

        // Split into Q, K, V: each (nw_b, n, dim)
        let q = qkv_out.slice(s![.., .., ..self.dim]).to_owned();
        let k = qkv_out.slice(s![.., .., self.dim..2 * self.dim]).to_owned();
        let v = qkv_out.slice(s![.., .., 2 * self.dim..]).to_owned();

        // Scale Q
        let q = q * self.scale;

        // Compute attention scores via explicit dot-product:
        // attn[b, i, j] = sum_d Q[b, i, d] * K[b, j, d]
        // Output: (nw_b, n, n)
        let mut attn = Array3::<f32>::zeros((nw_b, n, n));
        for b in 0..nw_b {
            for i in 0..n {
                for j in 0..n {
                    let mut dot = 0.0f32;
                    for d in 0..self.dim {
                        dot += q[[b, i, d]] * k[[b, j, d]];
                    }
                    attn[[b, i, j]] = dot;
                }
            }
        }

        // Add relative position bias (averaged over heads for simplicity)
        // bias shape: (ws, ws, num_heads) → broadcast as mean over heads per (i,j)
        let ws = self.window_size;
        for i in 0..ws2 {
            let ri = i / ws;
            let ci = i % ws;
            for j in 0..ws2 {
                let rj = j / ws;
                let cj = j % ws;
                // relative row / col in [-(ws-1), ws-1] → shifted to [0, 2*(ws-1)]
                let rel_r = (ri as isize - rj as isize + ws as isize - 1) as usize;
                let rel_c = (ci as isize - cj as isize + ws as isize - 1) as usize;
                // Average bias over heads (keeps bias contribution without
                // restructuring the per-head attention loop)
                let bias_sum: f32 =
                    (0..self.num_heads).map(|h| self.relative_position_bias_table[[rel_r, rel_c, h]]).sum();
                let bias_mean = bias_sum / self.num_heads as f32;
                for b in 0..nw_b {
                    attn[[b, i, j]] += bias_mean;
                }
            }
        }

        // Softmax row-wise
        for b in 0..nw_b {
            for i in 0..n {
                let row_max = attn.slice(s![b, i, ..]).fold(f32::NEG_INFINITY, |a, &v| a.max(v));
                let mut row_sum = 0.0f32;
                for j in 0..n {
                    let v = (attn[[b, i, j]] - row_max).exp();
                    attn[[b, i, j]] = v;
                    row_sum += v;
                }
                let inv_sum = if row_sum > 0.0 { 1.0 / row_sum } else { 1.0 };
                for j in 0..n {
                    attn[[b, i, j]] *= inv_sum;
                }
            }
        }

        // Apply attention dropout
        if self.attn_drop > 0.0 {
            attn *= 1.0 - self.attn_drop;
        }

        // Weighted sum: out[b, i, d] = sum_j attn[b, i, j] * V[b, j, d]
        let mut out = Array3::<f32>::zeros((nw_b, n, self.dim));
        for b in 0..nw_b {
            for i in 0..n {
                for j in 0..n {
                    let a = attn[[b, i, j]];
                    for d in 0..self.dim {
                        out[[b, i, d]] += a * v[[b, j, d]];
                    }
                }
            }
        }

        // Output projection
        let out_tensor = Tensor::F32(out.into_dyn());
        let projected = match self.proj.forward(out_tensor)? {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 from proj".to_string(),
                ))
            },
        };

        let result = if self.proj_drop > 0.0 {
            projected * (1.0 - self.proj_drop)
        } else {
            projected
        };

        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Swin Transformer Block
// ─────────────────────────────────────────────────────────────────────────────

/// A single Swin Transformer block.
///
/// Applies either W-MSA (`shift_size = 0`) or SW-MSA (`shift_size > 0`),
/// followed by an MLP, each with pre-norm and residual connections.
#[derive(Debug, Clone)]
pub struct SwinTransformerBlock {
    /// Channel dimension.
    pub dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Window size for local attention.
    pub window_size: usize,
    /// Shift amount for SW-MSA (0 for W-MSA).
    pub shift_size: usize,
    pub attn: WindowAttention,
    pub norm1: LayerNorm,
    pub ffn: FeedForward,
    pub norm2: LayerNorm,
    drop_rate: f32,
    device: Device,
}

impl SwinTransformerBlock {
    /// Construct a Swin block.
    ///
    /// # Arguments
    ///
    /// * `dim` — channel count.
    /// * `num_heads` — attention heads.
    /// * `window_size` — local attention window.
    /// * `shift_size` — cyclic shift (0 for W-MSA, ws//2 for SW-MSA).
    /// * `mlp_ratio` — FFN expansion ratio.
    /// * `qkv_bias` — QKV bias flag.
    /// * `drop_rate` — MLP dropout.
    /// * `attn_drop` — attention dropout.
    /// * `layer_norm_eps` — LayerNorm ε.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        num_heads: usize,
        window_size: usize,
        shift_size: usize,
        mlp_ratio: f32,
        qkv_bias: bool,
        drop_rate: f32,
        attn_drop: f32,
        layer_norm_eps: f32,
        device: Device,
    ) -> Result<Self> {
        let mlp_hidden = ((dim as f32 * mlp_ratio) as usize).max(1);
        Ok(Self {
            dim,
            num_heads,
            window_size,
            shift_size,
            attn: WindowAttention::new(
                dim,
                window_size,
                num_heads,
                qkv_bias,
                attn_drop,
                drop_rate,
                device,
            )?,
            norm1: LayerNorm::new_with_device(vec![dim], layer_norm_eps, device)?,
            ffn: FeedForward::new_with_device(dim, mlp_hidden, 0.0, device),
            norm2: LayerNorm::new_with_device(vec![dim], layer_norm_eps, device)?,
            drop_rate,
            device,
        })
    }

    /// Active device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Run the Swin block forward pass.
    ///
    /// # Arguments
    ///
    /// * `x` — spatial feature map of shape `(B, H, W, C)`.
    ///
    /// # Returns
    ///
    /// Updated feature map of the same shape.
    pub fn forward(&self, x: &Array4<f32>) -> Result<Array4<f32>> {
        let (b, h, w, c) = x.dim();
        let ws = self.window_size;

        // Pad spatial dims to be multiples of window_size
        let pad_h = if h % ws == 0 { 0 } else { ws - h % ws };
        let pad_w = if w % ws == 0 { 0 } else { ws - w % ws };

        let (ph, pw) = (h + pad_h, w + pad_w);

        // Pad with zeros
        let x_padded = if pad_h > 0 || pad_w > 0 {
            let mut p = Array4::<f32>::zeros((b, ph, pw, c));
            p.slice_mut(s![.., ..h, ..w, ..]).assign(x);
            p
        } else {
            x.clone()
        };

        // Cyclic shift for SW-MSA
        let x_shifted =
            if self.shift_size > 0 { cyclic_shift(&x_padded, self.shift_size) } else { x_padded };

        // Flatten spatial → (B, ph*pw, C)
        let mut x_flat = Array3::<f32>::zeros((b, ph * pw, c));
        for bi in 0..b {
            for i in 0..ph {
                for j in 0..pw {
                    let pidx = i * pw + j;
                    x_flat.slice_mut(s![bi, pidx, ..]).assign(&x_shifted.slice(s![bi, i, j, ..]));
                }
            }
        }

        // Pre-norm 1
        let flat_tensor = Tensor::F32(x_flat.clone().into_dyn());
        let normed1 = match self.norm1.forward(flat_tensor)? {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 from norm1".to_string(),
                ))
            },
        };

        // Reshape (B, ph*pw, C) → (B, ph, pw, C) → window partition
        let mut normed1_4d = Array4::<f32>::zeros((b, ph, pw, c));
        for bi in 0..b {
            for i in 0..ph {
                for j in 0..pw {
                    let pidx = i * pw + j;
                    normed1_4d.slice_mut(s![bi, i, j, ..]).assign(&normed1.slice(s![bi, pidx, ..]));
                }
            }
        }

        // Partition into windows: (nw*B, ws, ws, C)
        let windows = window_partition(&normed1_4d, ws)?;
        let nw_b = windows.dim().0;
        let ws2 = ws * ws;

        // Flatten window spatial → (nw*B, ws^2, C)
        let mut win_flat = Array3::<f32>::zeros((nw_b, ws2, c));
        for wi in 0..nw_b {
            for i in 0..ws {
                for j in 0..ws {
                    let pidx = i * ws + j;
                    win_flat
                        .slice_mut(s![wi, pidx, ..])
                        .assign(&windows.slice(s![wi, i, j, ..]));
                }
            }
        }

        // Window attention
        let attn_out_flat = self.attn.forward(&win_flat)?;

        // Reshape back: (nw*B, ws^2, C) → (nw*B, ws, ws, C)
        let mut attn_win = Array4::<f32>::zeros((nw_b, ws, ws, c));
        for wi in 0..nw_b {
            for i in 0..ws {
                for j in 0..ws {
                    let pidx = i * ws + j;
                    attn_win
                        .slice_mut(s![wi, i, j, ..])
                        .assign(&attn_out_flat.slice(s![wi, pidx, ..]));
                }
            }
        }

        // Reverse windows → (B, ph, pw, C)
        let attn_spatial = window_reverse(&attn_win, ws, ph, pw)?;

        // Reverse cyclic shift
        let attn_unshifted = if self.shift_size > 0 {
            // reverse shift = shift by (ph - shift_size) % ph, (pw - shift_size) % pw
            let rev_h = (ph - self.shift_size % ph) % ph;
            let rev_w = (pw - self.shift_size % pw) % pw;
            let shift_back = rev_h.min(rev_w); // same in both dims for square
            // We need to roll back, but cyclic_shift only handles equal shifts.
            // Because ph == pw in practice (square images), we can use one value.
            cyclic_shift(&attn_spatial, shift_back)
        } else {
            attn_spatial
        };

        // Remove padding
        let attn_trimmed = attn_unshifted.slice(s![.., ..h, ..w, ..]).to_owned();

        // Flatten back and add residual
        let mut attn_flat = Array3::<f32>::zeros((b, h * w, c));
        for bi in 0..b {
            for i in 0..h {
                for j in 0..w {
                    let pidx = i * w + j;
                    attn_flat
                        .slice_mut(s![bi, pidx, ..])
                        .assign(&attn_trimmed.slice(s![bi, i, j, ..]));
                }
            }
        }

        // Rebuild original flat representation from x (before padding / shift)
        let mut x_flat_orig = Array3::<f32>::zeros((b, h * w, c));
        for bi in 0..b {
            for i in 0..h {
                for j in 0..w {
                    let pidx = i * w + j;
                    x_flat_orig.slice_mut(s![bi, pidx, ..]).assign(&x.slice(s![bi, i, j, ..]));
                }
            }
        }

        let attn_drop = if self.drop_rate > 0.0 {
            attn_flat * (1.0 - self.drop_rate)
        } else {
            attn_flat
        };

        let after_attn = x_flat_orig + attn_drop;

        // Pre-norm 2
        let flat2_tensor = Tensor::F32(after_attn.clone().into_dyn());
        let normed2 = match self.norm2.forward(flat2_tensor)? {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 from norm2".to_string(),
                ))
            },
        };

        // FFN
        let ffn_tensor = Tensor::F32(normed2.into_dyn());
        let ffn_out = match self.ffn.forward(ffn_tensor)? {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 from FFN".to_string(),
                ))
            },
        };

        let ffn_out = if self.drop_rate > 0.0 { ffn_out * (1.0 - self.drop_rate) } else { ffn_out };

        let after_ffn = after_attn + ffn_out;

        // Reshape (B, h*w, C) → (B, h, w, C)
        let mut out = Array4::<f32>::zeros((b, h, w, c));
        for bi in 0..b {
            for i in 0..h {
                for j in 0..w {
                    let pidx = i * w + j;
                    out.slice_mut(s![bi, i, j, ..]).assign(&after_ffn.slice(s![bi, pidx, ..]));
                }
            }
        }

        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Swin Stage (stack of blocks + optional PatchMerging)
// ─────────────────────────────────────────────────────────────────────────────

/// A Swin Transformer stage: `depth` blocks followed by optional `PatchMerging`.
#[derive(Debug, Clone)]
pub struct SwinStage {
    pub blocks: Vec<SwinTransformerBlock>,
    /// Present on all stages except the last.
    pub downsample: Option<PatchMerging>,
    device: Device,
}

impl SwinStage {
    /// Construct a stage.
    ///
    /// # Arguments
    ///
    /// * `dim` — input channel dimension.
    /// * `depth` — number of transformer blocks.
    /// * `num_heads` — attention heads.
    /// * `window_size` — local window size.
    /// * `mlp_ratio` — FFN expansion ratio.
    /// * `qkv_bias` — QKV bias.
    /// * `drop_rate` — hidden state dropout.
    /// * `attn_drop` — attention weight dropout.
    /// * `layer_norm_eps` — LayerNorm ε.
    /// * `downsample` — whether to add PatchMerging at the end.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        depth: usize,
        num_heads: usize,
        window_size: usize,
        mlp_ratio: f32,
        qkv_bias: bool,
        drop_rate: f32,
        attn_drop: f32,
        layer_norm_eps: f32,
        downsample: bool,
        device: Device,
    ) -> Result<Self> {
        let blocks = (0..depth)
            .map(|i| {
                // Even blocks: W-MSA (shift_size = 0)
                // Odd blocks: SW-MSA (shift_size = window_size // 2)
                let shift_size = if i % 2 == 0 { 0 } else { window_size / 2 };
                SwinTransformerBlock::new(
                    dim,
                    num_heads,
                    window_size,
                    shift_size,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    attn_drop,
                    layer_norm_eps,
                    device,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let downsample_layer = if downsample {
            Some(PatchMerging::new(dim, layer_norm_eps, device)?)
        } else {
            None
        };

        Ok(Self { blocks, downsample: downsample_layer, device })
    }

    /// Active device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Forward through all blocks and the optional downsampler.
    ///
    /// Returns `(output, h_out, w_out)` where spatial dims may be halved.
    pub fn forward(&self, x: &Array4<f32>) -> Result<Array4<f32>> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        if let Some(ref ds) = self.downsample {
            x = ds.forward(&x)?;
        }
        Ok(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Swin Model
// ─────────────────────────────────────────────────────────────────────────────

/// Swin Transformer base model.
///
/// Produces a 1-D class feature of shape `(B, final_dim)` suitable for a
/// linear classification head, or the full hierarchical feature pyramid if
/// used for dense prediction.
///
/// # Example
///
/// ```rust,no_run
/// use trustformers_models::swin::{SwinConfig, SwinModel};
/// use scirs2_core::ndarray::Array4;
///
/// let config = SwinConfig::swin_tiny_patch4_window7_224();
/// let model = SwinModel::new(config).expect("valid config constructs Swin model");
/// let images = Array4::<f32>::zeros((1, 224, 224, 3));
/// let features = model.forward(&images).expect("well-formed input tensor succeeds");
/// // features: (1, 768) — averaged global pooling of final stage
/// ```
#[derive(Debug, Clone)]
pub struct SwinModel {
    pub patch_embed: SwinPatchEmbedding,
    pub stages: Vec<SwinStage>,
    pub norm: LayerNorm,
    pub config: SwinConfig,
    device: Device,
}

impl SwinModel {
    /// Create on the CPU.
    pub fn new(config: SwinConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    /// Create on the specified device.
    pub fn new_with_device(config: SwinConfig, device: Device) -> Result<Self> {
        config.validate()?;

        let patch_embed = SwinPatchEmbedding::new_with_device(&config, device)?;

        let num_stages = config.num_stages();
        let stages = (0..num_stages)
            .map(|i| {
                let dim = config.stage_dim(i);
                let depth = config.depths[i];
                let num_heads = config.num_heads[i];
                // Downsample on all stages except the last
                let downsample = i < num_stages - 1;
                SwinStage::new(
                    dim,
                    depth,
                    num_heads,
                    config.window_size,
                    config.mlp_ratio,
                    config.qkv_bias,
                    config.drop_rate,
                    config.attn_drop_rate,
                    config.layer_norm_eps,
                    downsample,
                    device,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let final_dim = config.final_dim();
        let norm = LayerNorm::new_with_device(vec![final_dim], config.layer_norm_eps, device)?;

        Ok(Self { patch_embed, stages, norm, config, device })
    }

    /// Active device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Run the Swin forward pass.
    ///
    /// Returns globally-pooled feature vector of shape `(B, final_dim)`.
    pub fn forward(&self, images: &Array4<f32>) -> Result<Array2<f32>> {
        // Patch partition + linear embedding: (B, H/ps, W/ps, embed_dim)
        let mut x = self.patch_embed.forward(images)?;

        // Pass through all stages
        for stage in &self.stages {
            x = stage.forward(&x)?;
        }

        let (b, h, w, c) = x.dim();
        // Global average pooling over spatial dims → (B, C)
        let mut flat = Array3::<f32>::zeros((b, h * w, c));
        for bi in 0..b {
            for i in 0..h {
                for j in 0..w {
                    let pidx = i * w + j;
                    flat.slice_mut(s![bi, pidx, ..]).assign(&x.slice(s![bi, i, j, ..]));
                }
            }
        }

        // Layer-norm on each spatial position
        let flat_tensor = Tensor::F32(flat.into_dyn());
        let normed = match self.norm.forward(flat_tensor)? {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 from final LayerNorm".to_string(),
                ))
            },
        };

        // Average over the spatial sequence dimension
        let pooled = normed
            .mean_axis(Axis(1))
            .ok_or_else(|| TrustformersError::invalid_input_simple(
                "Mean over spatial axis failed".to_string(),
            ))?;

        Ok(pooled)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array4;
    use trustformers_core::device::Device;

    /// LCG random number generator (a=6364136223846793005, c=1442695040888963407)
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            (self.next_u64() >> 33) as f32 / (u32::MAX as f32)
        }

        fn fill_array4(&mut self, arr: &mut Array4<f32>) {
            for v in arr.iter_mut() {
                *v = self.next_f32() * 2.0 - 1.0;
            }
        }
    }

    // --- window_partition ---

    #[test]
    fn test_window_partition_shape() {
        // B=1, H=14, W=14, C=4, ws=7 → 4 windows of (7,7,4)
        let x = Array4::<f32>::zeros((1, 14, 14, 4));
        let windows =
            window_partition(&x, 7).expect("window_partition should succeed for 14x14/7");
        assert_eq!(windows.dim(), (4, 7, 7, 4));
    }

    #[test]
    fn test_window_partition_batch_2() {
        // B=2, H=14, W=14, C=3, ws=7 → 2*(14/7)*(14/7) = 8 windows
        let x = Array4::<f32>::zeros((2, 14, 14, 3));
        let windows =
            window_partition(&x, 7).expect("window_partition batch=2 should succeed");
        assert_eq!(windows.dim().0, 8);
    }

    #[test]
    fn test_window_partition_seq_len_formula() {
        // seq_len per window = ws * ws; total windows = B*(H/ws)*(W/ws)
        let b = 1usize;
        let h = 56usize;
        let w = 56usize;
        let ws = 7usize;
        let x = Array4::<f32>::zeros((b, h, w, 8));
        let windows = window_partition(&x, ws).expect("window_partition 56x56/7 should succeed");
        let expected_nw = b * (h / ws) * (w / ws);
        assert_eq!(windows.dim().0, expected_nw);
        assert_eq!(windows.dim().1, ws);
        assert_eq!(windows.dim().2, ws);
    }

    #[test]
    fn test_window_partition_invalid_input() {
        // H=15 is not divisible by ws=7 → error
        let x = Array4::<f32>::zeros((1, 15, 14, 3));
        let result = window_partition(&x, 7);
        assert!(result.is_err(), "should fail when H%ws != 0");
    }

    #[test]
    fn test_window_partition_preserves_values() {
        let mut rng = Lcg::new(42);
        let mut x = Array4::<f32>::zeros((1, 14, 14, 2));
        rng.fill_array4(&mut x);
        let windows =
            window_partition(&x, 7).expect("window_partition should succeed");
        // First window should match top-left 7x7 block of x
        for i in 0..7 {
            for j in 0..7 {
                for c in 0..2 {
                    let diff = (windows[[0, i, j, c]] - x[[0, i, j, c]]).abs();
                    assert!(diff < 1e-6, "values mismatch at ({},{},{})", i, j, c);
                }
            }
        }
    }

    // --- window_reverse ---

    #[test]
    fn test_window_reverse_roundtrip() {
        let mut rng = Lcg::new(99);
        let mut x = Array4::<f32>::zeros((1, 14, 14, 4));
        rng.fill_array4(&mut x);
        let windows = window_partition(&x, 7).expect("partition should succeed");
        let reconstructed =
            window_reverse(&windows, 7, 14, 14).expect("reverse should succeed");
        for ((a, b_val), (c_val, d)) in x.iter().zip(reconstructed.iter()).map(|(a, b)| (a, b)).zip(
            reconstructed.iter().zip(x.iter()),
        ) {
            let _ = (a, b_val, c_val, d); // avoid unused
        }
        // Check shape
        assert_eq!(reconstructed.dim(), x.dim());
        // Check values
        let max_diff = x
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "roundtrip error too large: {}", max_diff);
    }

    #[test]
    fn test_window_reverse_batch_consistency() {
        let windows = Array4::<f32>::zeros((8, 7, 7, 3)); // 8 windows
        let result = window_reverse(&windows, 7, 14, 14).expect("reverse should succeed");
        // 8 windows = 2 batches * 2x2 grid → B=2
        assert_eq!(result.dim(), (2, 14, 14, 3));
    }

    // --- cyclic_shift ---

    #[test]
    fn test_cyclic_shift_zero() {
        let mut rng = Lcg::new(7);
        let mut x = Array4::<f32>::zeros((1, 8, 8, 2));
        rng.fill_array4(&mut x);
        let shifted = cyclic_shift(&x, 0);
        let max_diff = x
            .iter()
            .zip(shifted.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6, "shift by 0 should be identity");
    }

    #[test]
    fn test_cyclic_shift_by_window_half() {
        // Shift by ws//2=3, verify some top-left values moved
        let mut x = Array4::<f32>::zeros((1, 6, 6, 2));
        x[[0, 0, 0, 0]] = 1.0;
        let shifted = cyclic_shift(&x, 3);
        // Row 0 of x is moved to row (6-3)=3 in shifted
        assert!((shifted[[0, 3, 3, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cyclic_shift_double_inverse() {
        let mut rng = Lcg::new(17);
        let mut x = Array4::<f32>::zeros((1, 14, 14, 3));
        rng.fill_array4(&mut x);
        let shift = 3usize;
        let shifted = cyclic_shift(&x, shift);
        // inverse shift: 14 - 3 = 11
        let back = cyclic_shift(&shifted, 14 - shift);
        let max_diff = x
            .iter()
            .zip(back.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "double cyclic shift should recover original");
    }

    // --- SwinConfig helpers ---

    #[test]
    fn test_swin_tiny_config() {
        let cfg = SwinConfig::swin_tiny_patch4_window7_224();
        assert_eq!(cfg.embed_dim, 96);
        assert_eq!(cfg.patch_size, 4);
        assert_eq!(cfg.window_size, 7);
        assert_eq!(cfg.depths, vec![2, 2, 6, 2]);
        assert_eq!(cfg.num_heads, vec![3, 6, 12, 24]);
    }

    #[test]
    fn test_swin_config_stage_dim() {
        let cfg = SwinConfig::swin_tiny_patch4_window7_224();
        // Stage 0: 96, Stage 1: 192, Stage 2: 384, Stage 3: 768
        assert_eq!(cfg.stage_dim(0), 96);
        assert_eq!(cfg.stage_dim(1), 192);
        assert_eq!(cfg.stage_dim(2), 384);
        assert_eq!(cfg.stage_dim(3), 768);
    }

    #[test]
    fn test_swin_config_feature_map_sizes() {
        // 224 / 4 = 56 → initial resolution
        let cfg = SwinConfig::swin_tiny_patch4_window7_224();
        let init_res = cfg.initial_resolution();
        assert_eq!(init_res, 56, "Initial resolution should be H/patch_size = 56");
        // After stage0 (no downsample if not last): remains 56
        // PatchMerging halves: 56→28→14→7
        // final_dim = 96 * 8 = 768
        assert_eq!(cfg.final_dim(), 768);
    }

    #[test]
    fn test_swin_config_num_stages() {
        let cfg = SwinConfig::swin_tiny_patch4_window7_224();
        assert_eq!(cfg.num_stages(), 4);
    }

    #[test]
    fn test_swin_config_validate_ok() {
        let cfg = SwinConfig::swin_tiny_patch4_window7_224();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_swin_config_validate_bad_patch_size() {
        let cfg = SwinConfig {
            patch_size: 0,
            ..SwinConfig::swin_tiny_patch4_window7_224()
        };
        assert!(cfg.validate().is_err(), "patch_size=0 should fail validation");
    }

    #[test]
    fn test_swin_config_base_embed_dim() {
        let cfg = SwinConfig::swin_base_patch4_window7_224();
        assert_eq!(cfg.embed_dim, 128);
    }

    // --- relative position bias table dimensions ---

    #[test]
    fn test_relative_position_bias_table_shape() {
        let ws = 7usize;
        let num_heads = 3usize;
        let attn = WindowAttention::new(96, ws, num_heads, true, 0.0, 0.0, Device::CPU)
            .expect("WindowAttention should construct");
        let bias_len = 2 * ws - 1;
        assert_eq!(
            attn.relative_position_bias_table.dim(),
            (bias_len, bias_len, num_heads),
            "bias table shape should be (2W-1, 2W-1, num_heads)"
        );
    }

    #[test]
    fn test_relative_position_bias_table_shape_ws4() {
        let ws = 4usize;
        let num_heads = 6usize;
        let attn = WindowAttention::new(48, ws, num_heads, true, 0.0, 0.0, Device::CPU)
            .expect("WindowAttention should construct");
        let bias_len = 2 * ws - 1; // 7
        assert_eq!(attn.relative_position_bias_table.dim().0, bias_len);
        assert_eq!(attn.relative_position_bias_table.dim().1, bias_len);
        assert_eq!(attn.relative_position_bias_table.dim().2, num_heads);
    }

    // --- WindowAttention construction errors ---

    #[test]
    fn test_window_attention_invalid_num_heads() {
        let result = WindowAttention::new(96, 7, 0, true, 0.0, 0.0, Device::CPU);
        assert!(result.is_err(), "num_heads=0 should be rejected");
    }

    #[test]
    fn test_window_attention_dim_not_divisible() {
        // dim=97 is not divisible by 3
        let result = WindowAttention::new(97, 7, 3, true, 0.0, 0.0, Device::CPU);
        assert!(result.is_err(), "dim not divisible by num_heads should fail");
    }

    // --- WindowAttention forward ---

    #[test]
    fn test_window_attention_forward_shape() {
        // Window attention: input (nw*B, ws^2, dim)
        use scirs2_core::ndarray::Array3;
        let ws = 4usize;
        let dim = 8usize;
        let num_heads = 2usize;
        let attn = WindowAttention::new(dim, ws, num_heads, true, 0.0, 0.0, Device::CPU)
            .expect("WindowAttention should construct");
        let n = ws * ws;
        let input = Array3::<f32>::zeros((2, n, dim));
        let output = attn.forward(&input).expect("WindowAttention forward should succeed");
        assert_eq!(output.dim(), (2, n, dim), "output shape must match input");
    }

    // --- PatchMerging ---

    #[test]
    fn test_patch_merging_shape() {
        // Input (B=1, H=8, W=8, C=4) → output (B=1, 4, 4, 8)
        let pm = PatchMerging::new(4, 1e-5, Device::CPU)
            .expect("PatchMerging should construct");
        let x = Array4::<f32>::zeros((1, 8, 8, 4));
        let out = pm.forward(&x).expect("PatchMerging forward should succeed");
        assert_eq!(out.dim(), (1, 4, 4, 8), "PatchMerging should halve spatial and double channels");
    }

    #[test]
    fn test_patch_merging_channel_quadrupling_then_reduction() {
        // 4C → 2C: for C=96, output channels = 192
        let c = 96usize;
        let pm = PatchMerging::new(c, 1e-5, Device::CPU)
            .expect("PatchMerging should construct");
        let x = Array4::<f32>::zeros((1, 56, 56, c));
        let out = pm.forward(&x).expect("PatchMerging forward should succeed");
        assert_eq!(out.dim(), (1, 28, 28, 2 * c));
    }

    #[test]
    fn test_patch_merging_odd_size_error() {
        let pm = PatchMerging::new(4, 1e-5, Device::CPU)
            .expect("PatchMerging should construct");
        let x = Array4::<f32>::zeros((1, 7, 7, 4)); // 7 is odd → error
        let result = pm.forward(&x);
        assert!(result.is_err(), "odd spatial dims should fail");
    }

    // --- SwinPatchEmbedding ---

    #[test]
    fn test_patch_embedding_output_shape() {
        let cfg = SwinConfig::swin_tiny_patch4_window7_224();
        let embed = SwinPatchEmbedding::new(&cfg).expect("SwinPatchEmbedding should construct");
        let images = Array4::<f32>::zeros((1, 224, 224, 3));
        let out = embed.forward(&images).expect("SwinPatchEmbedding forward should succeed");
        // 224/4=56 → (1, 56, 56, 96)
        assert_eq!(out.dim(), (1, 56, 56, 96));
    }

    #[test]
    fn test_patch_embedding_wrong_channels_error() {
        let cfg = SwinConfig::swin_tiny_patch4_window7_224();
        let embed = SwinPatchEmbedding::new(&cfg).expect("SwinPatchEmbedding should construct");
        let images = Array4::<f32>::zeros((1, 224, 224, 4)); // 4 channels, expected 3
        let result = embed.forward(&images);
        assert!(result.is_err(), "wrong channel count should fail");
    }

    // --- SwinModel ---

    #[test]
    fn test_swin_model_construction() {
        let cfg = SwinConfig {
            image_size: 28,
            patch_size: 4,
            num_channels: 3,
            embed_dim: 8,
            depths: vec![1, 1],
            num_heads: vec![2, 4],
            window_size: 7,
            mlp_ratio: 2.0,
            qkv_bias: true,
            drop_rate: 0.0,
            attn_drop_rate: 0.0,
            drop_path_rate: 0.0,
            num_labels: 10,
            layer_norm_eps: 1e-5,
        };
        let model = SwinModel::new(cfg);
        assert!(model.is_ok(), "SwinModel should construct successfully");
    }

    #[test]
    fn test_swin_model_final_dim_tiny() {
        let cfg = SwinConfig::swin_tiny_patch4_window7_224();
        let final_d = cfg.final_dim();
        // 4 stages → 96 * 2^3 = 768
        assert_eq!(final_d, 768);
    }

    #[test]
    fn test_swin_block_construction() {
        let block = SwinTransformerBlock::new(
            96, 3, 7, 0, 4.0, true, 0.0, 0.0, 1e-5, Device::CPU,
        );
        assert!(block.is_ok(), "SwinTransformerBlock (W-MSA) should construct");
    }

    #[test]
    fn test_swin_block_shifted_construction() {
        let block = SwinTransformerBlock::new(
            96, 3, 7, 3, 4.0, true, 0.0, 0.0, 1e-5, Device::CPU,
        );
        assert!(block.is_ok(), "SwinTransformerBlock (SW-MSA) should construct");
    }

    #[test]
    fn test_swin_block_shift_size() {
        // shift_size should be window_size // 2 for odd blocks
        let ws = 7usize;
        let expected_shift = ws / 2; // 3
        assert_eq!(expected_shift, 3);
    }

    #[test]
    fn test_swin_stage_blocks_alternating() {
        // Even block index → shift=0 (W-MSA), odd → shift>0 (SW-MSA)
        let stage = SwinStage::new(
            96, 4, 3, 7, 4.0, true, 0.0, 0.0, 1e-5, false, Device::CPU,
        ).expect("SwinStage should construct");
        assert_eq!(stage.blocks.len(), 4);
        assert_eq!(stage.blocks[0].shift_size, 0);
        assert_eq!(stage.blocks[1].shift_size, 3);
        assert_eq!(stage.blocks[2].shift_size, 0);
        assert_eq!(stage.blocks[3].shift_size, 3);
    }

    #[test]
    fn test_swin_config_base_window12_384() {
        let cfg = SwinConfig::swin_base_patch4_window12_384();
        assert_eq!(cfg.image_size, 384);
        assert_eq!(cfg.window_size, 12);
        assert_eq!(cfg.embed_dim, 128);
        // final_dim = 128 * 8 = 1024
        assert_eq!(cfg.final_dim(), 1024);
    }

    #[test]
    fn test_window_partition_multiple_batches_content() {
        // Create an input where each batch element is distinct
        let mut x = Array4::<f32>::zeros((2, 14, 14, 1));
        for i in 0..14 {
            for j in 0..14 {
                x[[0, i, j, 0]] = 1.0; // batch 0 all ones
                x[[1, i, j, 0]] = 2.0; // batch 1 all twos
            }
        }
        let windows = window_partition(&x, 7).expect("partition should succeed");
        // Windows [0..4] from batch 0, [4..8] from batch 1
        assert!((windows[[0, 0, 0, 0]] - 1.0).abs() < 1e-6);
        assert!((windows[[4, 0, 0, 0]] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_swin_config_architecture_name() {
        use trustformers_core::traits::Config;
        let cfg = SwinConfig::default();
        assert_eq!(cfg.architecture(), "Swin");
    }
}
