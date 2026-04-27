//! Mixed precision training utilities (AMP-style).
//!
//! Maintains master weights in FP32, computes forward/backward passes in
//! BF16 or FP16 to reduce memory footprint and increase throughput on modern
//! accelerators.

use std::fmt;

// ---------------------------------------------------------------------------
// Precision mode
// ---------------------------------------------------------------------------

/// Selects the numeric format used for compute operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingPrecisionMode {
    /// Pure FP32 — no mixed precision.
    Fp32,
    /// FP16 for compute, FP32 master weights.  Uses a dynamic loss scaler.
    Fp16,
    /// BF16 for compute, FP32 master weights.  No loss scaler required.
    Bf16,
    /// Experimental 8-bit float (E4M3 variant, similar to MX FP8).
    Fp8E4M3,
}

impl fmt::Display for TrainingPrecisionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainingPrecisionMode::Fp32 => write!(f, "FP32"),
            TrainingPrecisionMode::Fp16 => write!(f, "FP16"),
            TrainingPrecisionMode::Bf16 => write!(f, "BF16"),
            TrainingPrecisionMode::Fp8E4M3 => write!(f, "FP8-E4M3"),
        }
    }
}

// ---------------------------------------------------------------------------
// BFloat16
// ---------------------------------------------------------------------------

/// Pure-Rust Brain Float 16 representation.
///
/// BF16 shares the same exponent as FP32 (8 bits) but reduces the mantissa
/// from 23 bits to 7 bits — it is literally the upper 16 bits of an FP32
/// word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BFloat16 {
    /// Raw IEEE 754 BF16 bit pattern stored as `u16`.
    pub bits: u16,
}

impl BFloat16 {
    /// Convert an `f32` value to `BFloat16`.
    ///
    /// Uses round-to-nearest-even (the same rounding mode mandated by IEEE
    /// 754-2008) and correctly propagates NaN and infinity.
    pub fn from_f32(val: f32) -> Self {
        let bits = val.to_bits();

        // Propagate NaN — preserve the quiet-NaN bit.
        if val.is_nan() {
            // Set the quiet-NaN bit (bit 22 in FP32 mantissa = bit 6 in BF16).
            return BFloat16 {
                bits: ((bits >> 16) as u16) | 0x0040,
            };
        }

        // Round to nearest even.
        // The rounding bit is bit 15 of the FP32 representation.
        let lsb = (bits >> 16) & 1; // least significant bit of BF16
        let rounding_bias = 0x7FFF_u32 + lsb; // round-to-nearest-even bias

        let rounded = bits.wrapping_add(rounding_bias);
        BFloat16 {
            bits: (rounded >> 16) as u16,
        }
    }

    /// Convert this `BFloat16` back to `f32`.
    pub fn to_f32(self) -> f32 {
        // BF16 occupies the upper 16 bits of FP32; just zero-extend.
        f32::from_bits((self.bits as u32) << 16)
    }

    /// Returns `true` if this value is NaN.
    pub fn is_nan(self) -> bool {
        // Exponent all-ones (0x7F80 mask) AND non-zero mantissa.
        (self.bits & 0x7FFF) > 0x7F80
    }

    /// Returns `true` if this value is positive or negative infinity.
    pub fn is_infinite(self) -> bool {
        (self.bits & 0x7FFF) == 0x7F80
    }

    /// Returns `true` if this value is positive or negative zero.
    pub fn is_zero(self) -> bool {
        (self.bits & 0x7FFF) == 0
    }
}

impl fmt::Display for BFloat16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

// ---------------------------------------------------------------------------
// Batch conversion helpers
// ---------------------------------------------------------------------------

/// Convert a slice of `f32` values to BF16 bit patterns (`u16`).
pub fn cast_fp32_to_bf16(fp32_slice: &[f32]) -> Vec<u16> {
    fp32_slice
        .iter()
        .map(|&v| BFloat16::from_f32(v).bits)
        .collect()
}

/// Convert a slice of BF16 bit patterns back to `f32`.
pub fn cast_bf16_to_fp32(bf16_slice: &[u16]) -> Vec<f32> {
    bf16_slice
        .iter()
        .map(|&bits| BFloat16 { bits }.to_f32())
        .collect()
}

/// Convert a slice of `f32` values to IEEE 754 FP16 bit patterns (`u16`).
///
/// Uses round-to-nearest-even and correctly handles overflow (clamps to ±Inf),
/// underflow (flush to zero for subnormals smaller than the representable
/// range), NaN, and infinity.
pub fn cast_fp32_to_fp16(fp32_slice: &[f32]) -> Vec<u16> {
    fp32_slice.iter().map(|&v| fp32_to_fp16_bits(v)).collect()
}

/// Convert a slice of IEEE 754 FP16 bit patterns to `f32`.
pub fn cast_fp16_to_fp32(fp16_slice: &[u16]) -> Vec<f32> {
    fp16_slice.iter().map(|&bits| fp16_bits_to_fp32(bits)).collect()
}

// ---------------------------------------------------------------------------
// Internal FP16 conversion (single value)
// ---------------------------------------------------------------------------

/// Convert one `f32` to the IEEE 754 FP16 bit representation.
fn fp32_to_fp16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if val.is_nan() {
        // Quiet NaN.
        return (sign << 15) | 0x7E00;
    }
    if val.is_infinite() {
        return (sign << 15) | 0x7C00;
    }

    // FP32 bias = 127; FP16 bias = 15.
    let new_exp = exp - 127 + 15;

    if new_exp >= 31 {
        // Overflow → infinity.
        return (sign << 15) | 0x7C00;
    }

    if new_exp <= 0 {
        // Underflow — try to represent as subnormal or flush to zero.
        if new_exp < -10 {
            return sign << 15; // too small even for subnormal
        }
        // Subnormal: mantissa with implicit leading 1 shifted right.
        let shift = (1 - new_exp) as u32; // 1..=10
        let mant_with_hidden = (mantissa | 0x0080_0000) >> shift;
        let round_bit = (mantissa | 0x0080_0000) >> (shift - 1) & 1;
        let fp16_mant = ((mant_with_hidden + round_bit) >> 13) as u16;
        return (sign << 15) | fp16_mant;
    }

    // Normal number — round mantissa from 23 bits to 10 bits.
    let fp16_mant = {
        let round_bit = (mantissa >> 12) & 1;
        let trunc = (mantissa >> 13) as u16;
        trunc + round_bit as u16
    };

    let fp16_exp = new_exp as u16;
    // Check for mantissa overflow after rounding.
    if fp16_mant >= 0x0400 {
        // Mantissa rounded up past 10 bits.
        let adjusted_exp = fp16_exp + 1;
        if adjusted_exp >= 31 {
            return (sign << 15) | 0x7C00; // overflow to infinity
        }
        return (sign << 15) | (adjusted_exp << 10);
    }

    (sign << 15) | (fp16_exp << 10) | fp16_mant
}

/// Convert one IEEE 754 FP16 bit pattern back to `f32`.
fn fp16_bits_to_fp32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x03FF) as u32;

    if exp == 0x1F {
        // NaN or infinity.
        let fp32_bits = (sign << 31) | 0x7F80_0000 | (mantissa << 13);
        return f32::from_bits(fp32_bits);
    }

    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31); // ±0
        }
        // Subnormal FP16 → normal FP32.
        let mant = mantissa << 1;
        // Normalize: find leading 1.
        let mut m = mant;
        let mut e = 0_u32;
        while m & 0x0400 == 0 {
            m <<= 1;
            e += 1;
        }
        m &= 0x03FF; // strip implicit leading 1
        let fp32_exp = 127 - 15 - e + 1;
        return f32::from_bits((sign << 31) | (fp32_exp << 23) | (m << 13));
    }

    // Normal.
    let fp32_exp = exp + 127 - 15;
    f32::from_bits((sign << 31) | (fp32_exp << 23) | (mantissa << 13))
}

// ---------------------------------------------------------------------------
// LossScaler (dynamic loss scaling for FP16)
// ---------------------------------------------------------------------------

/// Dynamic loss scaler for FP16 mixed-precision training.
///
/// Multiplies the loss by a large scalar before the backward pass so that
/// small gradients are not flushed to zero in FP16.  The scale is increased
/// after a run of successful steps and decreased after overflows (NaN/Inf in
/// the unscaled gradients).
#[derive(Debug, Clone)]
pub struct LossScaler {
    /// Current loss scale factor.
    pub scale: f32,
    /// Factor by which the scale is multiplied on success.
    pub growth_factor: f32,
    /// Factor by which the scale is multiplied on overflow.
    pub backoff_factor: f32,
    /// Number of consecutive overflow-free steps required before scaling up.
    pub growth_interval: usize,
    /// Steps since the last scale increase.
    pub steps_since_increase: usize,
    /// Number of consecutive overflows observed.
    pub consecutive_overflows: usize,
}

impl LossScaler {
    /// Construct a `LossScaler` with standard default hyper-parameters:
    /// initial scale 2¹⁶, growth factor 2.0, backoff 0.5, interval 2000.
    pub fn new() -> Self {
        Self {
            scale: 65536.0, // 2^16
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_increase: 0,
            consecutive_overflows: 0,
        }
    }

    /// Multiply the loss by the current scale before the backward pass.
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    /// Divide all gradients by the current scale to undo the loss scaling.
    pub fn unscale_gradients(&self, grads: &mut [Vec<f32>]) {
        let inv = 1.0 / self.scale;
        for param_grads in grads.iter_mut() {
            for g in param_grads.iter_mut() {
                *g *= inv;
            }
        }
    }

    /// Check whether *any* gradient contains a NaN or Inf value.
    ///
    /// Returns `true` if all gradients are finite (i.e., the step is safe).
    pub fn check_gradients(&self, grads: &[Vec<f32>]) -> bool {
        grads
            .iter()
            .flat_map(|g| g.iter())
            .all(|v| v.is_finite())
    }

    /// Update the scale after one optimizer step.
    ///
    /// * If `overflow` is `true`: reduce scale, reset counter.
    /// * If `overflow` is `false`: increment counter; when it reaches
    ///   `growth_interval`, multiply scale by `growth_factor`.
    pub fn update(&mut self, overflow: bool) {
        if overflow {
            self.scale *= self.backoff_factor;
            self.steps_since_increase = 0;
            self.consecutive_overflows += 1;
        } else {
            self.consecutive_overflows = 0;
            self.steps_since_increase += 1;
            if self.steps_since_increase >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_increase = 0;
            }
        }
        // Clamp scale to avoid degenerate values.
        if self.scale < 1.0 {
            self.scale = 1.0;
        }
    }

    /// Return the current loss scale.
    pub fn current_scale(&self) -> f32 {
        self.scale
    }
}

impl Default for LossScaler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AmpStats
// ---------------------------------------------------------------------------

/// Accumulated statistics for mixed-precision training.
#[derive(Debug, Clone, Default)]
pub struct AmpStats {
    /// Number of optimizer steps that were skipped due to NaN/Inf gradients.
    pub overflow_count: u64,
    /// Number of optimizer steps that completed successfully.
    pub successful_steps: u64,
    /// Most recently observed loss scale.
    pub current_loss_scale: f32,
}

impl AmpStats {
    /// Record one optimizer step outcome.
    pub fn update(&mut self, overflow: bool, scale: f32) {
        if overflow {
            self.overflow_count += 1;
        } else {
            self.successful_steps += 1;
        }
        self.current_loss_scale = scale;
    }
}

// ---------------------------------------------------------------------------
// MixedPrecisionContext
// ---------------------------------------------------------------------------

/// Container that manages master weights, optional loss scaling, and weight
/// casting for a mixed-precision training session.
pub struct MixedPrecisionContext {
    /// Which precision format is used for compute.
    pub mode: TrainingPrecisionMode,
    /// Dynamic loss scaler — populated only for FP16 mode.
    pub loss_scaler: Option<LossScaler>,
    /// FP32 master copies of all parameter tensors.
    pub master_weights: Vec<Vec<f32>>,
}

impl MixedPrecisionContext {
    /// Create a new context.
    ///
    /// `param_shapes` is a slice where each element is the number of scalar
    /// parameters in that tensor.  Master weights are initialized to zero.
    pub fn new(mode: TrainingPrecisionMode, param_shapes: &[usize]) -> Self {
        let loss_scaler = if mode == TrainingPrecisionMode::Fp16 {
            Some(LossScaler::new())
        } else {
            None
        };
        let master_weights = param_shapes.iter().map(|&n| vec![0.0_f32; n]).collect();
        Self {
            mode,
            loss_scaler,
            master_weights,
        }
    }

    /// Return the master weights cast to the compute precision (BF16 or FP16),
    /// encoded as `Vec<u16>` bit patterns.
    ///
    /// For `Fp32` and `Fp8E4M3`, returns the FP32 weights re-interpreted as
    /// `u16` pairs (the caller is expected to handle these modes appropriately;
    /// in practice `Fp32` training should not call this method for the purpose
    /// of obtaining low-precision weights).
    pub fn get_compute_weights(&self) -> Vec<Vec<u16>> {
        match self.mode {
            TrainingPrecisionMode::Bf16 => self
                .master_weights
                .iter()
                .map(|w| cast_fp32_to_bf16(w))
                .collect(),
            TrainingPrecisionMode::Fp16 => self
                .master_weights
                .iter()
                .map(|w| cast_fp32_to_fp16(w))
                .collect(),
            TrainingPrecisionMode::Fp32 | TrainingPrecisionMode::Fp8E4M3 => {
                // For FP32 mode, return the raw f32 bits split into u16 pairs
                // (high word first) so that the shape is preserved.
                self.master_weights
                    .iter()
                    .map(|w| {
                        w.iter()
                            .flat_map(|&v| {
                                let b = v.to_bits();
                                [(b >> 16) as u16, b as u16]
                            })
                            .collect()
                    })
                    .collect()
            }
        }
    }

    /// Apply a vanilla SGD update to the master weights.
    ///
    /// `grads` must already be unscaled (i.e., in FP32 magnitude, not scaled
    /// by the loss scaler).  Each grad slice must have the same length as the
    /// corresponding master-weight slice.
    pub fn update_master_weights(&mut self, grads: &[Vec<f32>], learning_rate: f32) {
        for (weights, g) in self.master_weights.iter_mut().zip(grads.iter()) {
            for (w, dw) in weights.iter_mut().zip(g.iter()) {
                *w -= learning_rate * dw;
            }
        }
    }

    /// Return `true` if *any* gradient contains NaN or Inf.
    ///
    /// Delegates to `LossScaler::check_gradients` when a scaler is present,
    /// otherwise checks directly.
    pub fn check_overflow(&self, grads: &[Vec<f32>]) -> bool {
        let all_finite = grads
            .iter()
            .flat_map(|g| g.iter())
            .all(|v| v.is_finite());
        !all_finite
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // 1. BF16 from_f32 → to_f32 round-trip (small precision error allowed)
    // ------------------------------------------------------------------
    #[test]
    fn test_bf16_round_trip() {
        let values: &[f32] = &[1.0, -1.0, 0.5, 3.14, 100.0, -0.001, 1024.0];
        for &v in values {
            let bf = BFloat16::from_f32(v);
            let back = bf.to_f32();
            // BF16 has ~7 bits of mantissa precision; relative error < 1%.
            let rel_err = ((back - v) / v).abs();
            assert!(
                rel_err < 0.01,
                "round-trip failed for {v}: got {back}, rel_err={rel_err}"
            );
        }
    }

    // ------------------------------------------------------------------
    // 2. BF16 NaN and Inf handling
    // ------------------------------------------------------------------
    #[test]
    fn test_bf16_special_values() {
        let nan = BFloat16::from_f32(f32::NAN);
        assert!(nan.is_nan(), "BF16 NaN should be NaN");

        let pos_inf = BFloat16::from_f32(f32::INFINITY);
        assert!(pos_inf.is_infinite(), "BF16 +Inf should be infinite");
        assert!(!pos_inf.is_nan());

        let neg_inf = BFloat16::from_f32(f32::NEG_INFINITY);
        assert!(neg_inf.is_infinite(), "BF16 -Inf should be infinite");

        let zero = BFloat16::from_f32(0.0);
        assert!(zero.is_zero(), "BF16 0.0 should be zero");
    }

    // ------------------------------------------------------------------
    // 3. FP32 → BF16 → FP32 batch conversion
    // ------------------------------------------------------------------
    #[test]
    fn test_fp32_bf16_fp32_batch_conversion() {
        let original: Vec<f32> = vec![1.0, 2.0, -3.5, 0.125, 1000.0];
        let bf16_bits = cast_fp32_to_bf16(&original);
        assert_eq!(bf16_bits.len(), original.len());
        let recovered = cast_bf16_to_fp32(&bf16_bits);
        for (o, r) in original.iter().zip(recovered.iter()) {
            let rel = ((r - o) / o).abs();
            assert!(rel < 0.02, "batch bf16 round-trip: {o} → {r}");
        }
    }

    // ------------------------------------------------------------------
    // 4. Loss scaling (scale_loss multiplies by current scale)
    // ------------------------------------------------------------------
    #[test]
    fn test_loss_scaling_multiplies() {
        let scaler = LossScaler::new();
        let loss = 0.5_f32;
        let scaled = scaler.scale_loss(loss);
        let expected = loss * scaler.current_scale();
        assert!(
            (scaled - expected).abs() < 1e-3,
            "scale_loss should multiply by scale"
        );
        assert!(scaled > loss, "scaled loss should be larger than original");
    }

    // ------------------------------------------------------------------
    // 5. Unscale divides gradients by the current scale
    // ------------------------------------------------------------------
    #[test]
    fn test_unscale_divides() {
        let scaler = LossScaler::new();
        let scale = scaler.current_scale();
        let mut grads = vec![vec![scale * 2.0_f32, scale * 4.0_f32]];
        scaler.unscale_gradients(&mut grads);
        assert!((grads[0][0] - 2.0).abs() < 1e-4, "unscaled[0] should be 2.0");
        assert!((grads[0][1] - 4.0).abs() < 1e-4, "unscaled[1] should be 4.0");
    }

    // ------------------------------------------------------------------
    // 6. Gradient overflow detection (Inf/NaN triggers overflow)
    // ------------------------------------------------------------------
    #[test]
    fn test_gradient_overflow_detection() {
        let scaler = LossScaler::new();

        let clean_grads = vec![vec![1.0_f32, 2.0, -3.0]];
        assert!(
            scaler.check_gradients(&clean_grads),
            "clean grads should not overflow"
        );

        let nan_grads = vec![vec![1.0_f32, f32::NAN]];
        assert!(
            !scaler.check_gradients(&nan_grads),
            "NaN grad should trigger overflow"
        );

        let inf_grads = vec![vec![f32::INFINITY, 1.0_f32]];
        assert!(
            !scaler.check_gradients(&inf_grads),
            "Inf grad should trigger overflow"
        );
    }

    // ------------------------------------------------------------------
    // 7. Loss scale decreases on overflow
    // ------------------------------------------------------------------
    #[test]
    fn test_loss_scale_decreases_on_overflow() {
        let mut scaler = LossScaler::new();
        let initial = scaler.current_scale();
        scaler.update(true); // overflow
        assert!(
            scaler.current_scale() < initial,
            "scale should decrease on overflow: {} → {}",
            initial,
            scaler.current_scale()
        );
        assert_eq!(
            scaler.current_scale(),
            initial * scaler.backoff_factor,
            "scale should be initial * backoff_factor"
        );
    }

    // ------------------------------------------------------------------
    // 8. Loss scale grows after growth_interval successful steps
    // ------------------------------------------------------------------
    #[test]
    fn test_loss_scale_grows_after_interval() {
        let mut scaler = LossScaler::new();
        // Shrink the interval so the test is fast.
        scaler.growth_interval = 5;
        let initial = scaler.current_scale();

        for _ in 0..5 {
            scaler.update(false); // no overflow
        }
        assert!(
            scaler.current_scale() > initial,
            "scale should grow after {} successful steps",
            scaler.growth_interval
        );
        assert_eq!(
            scaler.current_scale(),
            initial * scaler.growth_factor,
            "scale should be initial * growth_factor"
        );
    }

    // ------------------------------------------------------------------
    // 9. MixedPrecisionContext compute weights have the right shapes
    // ------------------------------------------------------------------
    #[test]
    fn test_mixed_precision_context_compute_weights_shape() {
        let param_shapes = &[4_usize, 6, 2];
        let ctx = MixedPrecisionContext::new(TrainingPrecisionMode::Bf16, param_shapes);
        let compute = ctx.get_compute_weights();
        assert_eq!(compute.len(), param_shapes.len());
        for (i, &n) in param_shapes.iter().enumerate() {
            assert_eq!(
                compute[i].len(),
                n,
                "compute weights for param {i} should have {n} elements"
            );
        }
    }

    // ------------------------------------------------------------------
    // 10. Master weight SGD update modifies weights correctly
    // ------------------------------------------------------------------
    #[test]
    fn test_master_weight_sgd_update() {
        let param_shapes = &[3_usize];
        let mut ctx = MixedPrecisionContext::new(TrainingPrecisionMode::Fp32, param_shapes);
        // Set initial weights.
        ctx.master_weights[0] = vec![1.0_f32, 2.0, 3.0];

        let grads = vec![vec![0.1_f32, 0.2, 0.3]];
        let lr = 0.1_f32;
        ctx.update_master_weights(&grads, lr);

        // Expected: w - lr * g.
        let expected = vec![1.0 - 0.01, 2.0 - 0.02, 3.0 - 0.03];
        for (w, e) in ctx.master_weights[0].iter().zip(expected.iter()) {
            assert!(
                (w - e).abs() < 1e-6,
                "SGD update: expected {e}, got {w}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Bonus: check_overflow returns true for NaN/Inf
    // ------------------------------------------------------------------
    #[test]
    fn test_context_check_overflow() {
        let ctx = MixedPrecisionContext::new(TrainingPrecisionMode::Fp16, &[2_usize]);
        let clean = vec![vec![0.1_f32, -0.2]];
        assert!(!ctx.check_overflow(&clean), "no overflow for clean grads");

        let inf_grads = vec![vec![f32::INFINITY, 0.5]];
        assert!(ctx.check_overflow(&inf_grads), "overflow for Inf grad");
    }

    // ------------------------------------------------------------------
    // Bonus: AmpStats update works correctly
    // ------------------------------------------------------------------
    #[test]
    fn test_amp_stats_update() {
        let mut stats = AmpStats::default();
        stats.update(false, 65536.0);
        assert_eq!(stats.successful_steps, 1);
        assert_eq!(stats.overflow_count, 0);
        assert_eq!(stats.current_loss_scale, 65536.0);

        stats.update(true, 32768.0);
        assert_eq!(stats.overflow_count, 1);
        assert_eq!(stats.successful_steps, 1);
        assert_eq!(stats.current_loss_scale, 32768.0);
    }

    // ------------------------------------------------------------------
    // Bonus: FP16 round-trip for normal values
    // ------------------------------------------------------------------
    #[test]
    fn test_fp16_round_trip() {
        let values: &[f32] = &[1.0, -1.0, 0.5, 2.0, 4.0];
        for &v in values {
            let bits = cast_fp32_to_fp16(&[v]);
            let back = cast_fp16_to_fp32(&bits);
            // FP16 has ~10 bits of mantissa; relative error < 0.2%.
            let rel = ((back[0] - v) / v).abs();
            assert!(
                rel < 0.002,
                "fp16 round-trip for {v}: got {}, rel={rel}",
                back[0]
            );
        }
    }
}
