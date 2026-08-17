//! Measured statistics and calibration math for the quantization calibration
//! toolkit.
//!
//! Everything in this module is computed from the data it is handed. There are
//! no placeholder constants: if a quantity cannot be derived from the calibration
//! samples alone (for example end-to-end accuracy retention, which needs a model
//! and an evaluation harness), it is not produced here at all.
//!
//! # Conventions
//!
//! * **Channel** means an index along the *last* axis of the calibration
//!   samples. All leading axes and all samples are pooled into that channel, so
//!   a `[batch, tokens, hidden]` sample contributes `batch * tokens` values to
//!   each of its `hidden` channels. This is the granularity per-channel
//!   quantization scales are defined at.
//! * **Percentiles** are computed on the exact sorted values with linear
//!   interpolation between neighbouring order statistics.
//! * **Entropy** is Shannon entropy in **bits**, computed over a uniform
//!   histogram of [`HISTOGRAM_BINS`] bins spanning the observed range.

use crate::errors::{invalid_input, TrustformersError};
use crate::tensor::Tensor;

use super::calibration_toolkit::{
    DatasetStatistics, DistributionAnalysis, DistributionType, DynamicRange, TensorStatistics,
};

/// Number of histogram bins used for entropy, KL divergence and mode detection.
pub(crate) const HISTOGRAM_BINS: usize = 256;

/// Percentiles reported in [`TensorStatistics::percentiles`], in order.
pub(crate) const REPORTED_PERCENTILES: [f32; 5] = [5.0, 25.0, 50.0, 75.0, 95.0];

/// Values pooled from a calibration dataset, both globally and per channel.
pub(crate) struct PooledSamples {
    /// Every value of every sample, in sample-major order.
    pub all: Vec<f32>,
    /// `per_channel[c]` holds every value that landed in channel `c`.
    pub per_channel: Vec<Vec<f32>>,
}

/// Pool the values of a calibration dataset by channel (last axis).
///
/// Returns an error for an empty dataset, for samples that carry no elements, or
/// for samples whose channel count differs from the first sample's.
pub(crate) fn pool_samples(samples: &[Tensor]) -> Result<PooledSamples, TrustformersError> {
    if samples.is_empty() {
        return Err(invalid_input(
            "Cannot calculate statistics for an empty dataset".to_string(),
        ));
    }

    let first_shape = samples[0].shape();
    let channels = *first_shape.last().unwrap_or(&1);
    if channels == 0 {
        return Err(invalid_input(
            "Calibration samples must have a non-zero last dimension".to_string(),
        ));
    }

    let mut all = Vec::new();
    let mut per_channel = vec![Vec::new(); channels];

    for (index, sample) in samples.iter().enumerate() {
        let shape = sample.shape();
        let sample_channels = *shape.last().unwrap_or(&1);
        if sample_channels != channels {
            return Err(invalid_input(format!(
                "Calibration sample {} has {} channels but sample 0 has {}",
                index, sample_channels, channels
            )));
        }

        let values = sample.to_vec_f32()?;
        if values.is_empty() {
            return Err(invalid_input(format!(
                "Calibration sample {} is empty",
                index
            )));
        }

        for (position, &value) in values.iter().enumerate() {
            per_channel[position % channels].push(value);
        }
        all.extend_from_slice(&values);
    }

    Ok(PooledSamples { all, per_channel })
}

/// Exact percentile of an already-sorted slice, with linear interpolation.
///
/// `percentile` is given in `0..=100`. Returns `0.0` for an empty slice.
pub(crate) fn percentile_of_sorted(sorted: &[f32], percentile: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let clamped = percentile.clamp(0.0, 100.0) as f64 / 100.0;
    let position = clamped * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    let weight = (position - lower as f64) as f32;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

/// Population mean, variance, skewness and (non-excess) kurtosis of `values`.
///
/// The kurtosis convention matches the "normal distribution has kurtosis 3"
/// definition (`m4 / m2^2`, not the excess form).
pub(crate) fn moments(values: &[f32]) -> (f32, f32, f32, f32) {
    let count = values.len();
    if count == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = count as f64;
    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n;

    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    let mut m4 = 0.0f64;
    for &value in values {
        let d = value as f64 - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;

    let variance = m2;
    let skewness = if m2 > 0.0 { m3 / m2.powf(1.5) } else { 0.0 };
    // A constant distribution has no shape; report the normal-equivalent value.
    let kurtosis = if m2 > 0.0 { m4 / (m2 * m2) } else { 3.0 };

    (
        mean as f32,
        variance as f32,
        skewness as f32,
        kurtosis as f32,
    )
}

/// Uniform histogram of `values` over `[min, max]` with `bins` buckets.
///
/// A degenerate range (all values identical) puts every value in the first bin.
pub(crate) fn histogram(values: &[f32], min: f32, max: f32, bins: usize) -> Vec<u64> {
    let mut counts = vec![0u64; bins.max(1)];
    if values.is_empty() || bins == 0 {
        return counts;
    }
    let range = max - min;
    if !range.is_finite() || range <= 0.0 {
        counts[0] = values.len() as u64;
        return counts;
    }
    let scale = bins as f32 / range;
    for &value in values {
        let position = ((value - min) * scale) as isize;
        let index = position.clamp(0, bins as isize - 1) as usize;
        counts[index] += 1;
    }
    counts
}

/// Shannon entropy of a histogram, in bits.
pub(crate) fn histogram_entropy_bits(counts: &[u64]) -> f32 {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    let mut entropy = 0.0f64;
    for &count in counts {
        if count > 0 {
            let p = count as f64 / total_f;
            entropy -= p * p.log2();
        }
    }
    entropy as f32
}

/// Kullback-Leibler divergence `D(p || q)` between two histograms, in nats.
///
/// Both histograms must use the same binning. Bins are Laplace-smoothed so that
/// an empty reference bin does not produce an infinite divergence.
pub(crate) fn histogram_kl_divergence(p_counts: &[u64], q_counts: &[u64]) -> f32 {
    if p_counts.len() != q_counts.len() || p_counts.is_empty() {
        return 0.0;
    }
    let bins = p_counts.len() as f64;
    let p_total: f64 = p_counts.iter().map(|&c| c as f64).sum::<f64>() + bins;
    let q_total: f64 = q_counts.iter().map(|&c| c as f64).sum::<f64>() + bins;
    if p_total <= 0.0 || q_total <= 0.0 {
        return 0.0;
    }

    let mut divergence = 0.0f64;
    for (&p_count, &q_count) in p_counts.iter().zip(q_counts.iter()) {
        let p = (p_count as f64 + 1.0) / p_total;
        let q = (q_count as f64 + 1.0) / q_total;
        divergence += p * (p / q).ln();
    }
    divergence.max(0.0) as f32
}

/// Count the modes of a histogram and report whether it looks multimodal.
///
/// The histogram is smoothed with a 3-point moving average; a bin is a mode when
/// it is a strict local maximum, holds at least 5% of the largest bin's mass,
/// and is separated from the previously accepted mode by a valley below 60% of
/// the smaller of the two peaks.
pub(crate) fn count_modes(counts: &[u64]) -> usize {
    if counts.len() < 3 {
        return usize::from(!counts.is_empty());
    }

    // 3-point moving average with edge replication, so the first and last bins
    // are not inflated by averaging over a shorter window.
    let smoothed: Vec<f64> = (0..counts.len())
        .map(|i| {
            let left = counts[i.saturating_sub(1)] as f64;
            let center = counts[i] as f64;
            let right = counts[(i + 1).min(counts.len() - 1)] as f64;
            (left + center + right) / 3.0
        })
        .collect();

    let peak_height = smoothed.iter().copied().fold(0.0f64, f64::max);
    if peak_height <= 0.0 {
        return 0;
    }
    let min_height = 0.05 * peak_height;

    // Plateau-aware local maxima, endpoints included: a run of equal values is a
    // peak when it dominates the nearest different value on each side.
    let mut candidates: Vec<(usize, f64)> = Vec::new();
    let mut index = 0usize;
    while index < smoothed.len() {
        let mut plateau_end = index;
        while plateau_end + 1 < smoothed.len()
            && (smoothed[plateau_end + 1] - smoothed[index]).abs() <= f64::EPSILON
        {
            plateau_end += 1;
        }

        let dominates_left = index == 0 || smoothed[index] > smoothed[index - 1];
        let dominates_right =
            plateau_end + 1 >= smoothed.len() || smoothed[index] > smoothed[plateau_end + 1];

        if dominates_left && dominates_right && smoothed[index] >= min_height {
            candidates.push((index, smoothed[index]));
        }
        index = plateau_end + 1;
    }

    // Merge peaks that are not separated by a deep enough valley.
    let mut modes: Vec<(usize, f64)> = Vec::new();
    for (position, height) in candidates {
        match modes.last().copied() {
            None => modes.push((position, height)),
            Some((last_position, last_height)) => {
                let valley =
                    smoothed[last_position..=position].iter().copied().fold(f64::MAX, f64::min);
                if valley < 0.6 * last_height.min(height) {
                    modes.push((position, height));
                } else if height > last_height {
                    let slot = modes.len() - 1;
                    modes[slot] = (position, height);
                }
            },
        }
    }

    modes.len().max(1)
}

/// Jarque-Bera normality test.
///
/// Returns `(statistic, p_value)`. The statistic is asymptotically chi-squared
/// with two degrees of freedom, whose survival function has the closed form
/// `exp(-x / 2)`; that is the reported p-value.
pub(crate) fn jarque_bera(values: &[f32]) -> (f32, f32) {
    let n = values.len();
    if n < 8 {
        // The asymptotic distribution is meaningless for tiny samples.
        return (0.0, 1.0);
    }
    let (_, _, skewness, kurtosis) = moments(values);
    let statistic =
        (n as f64 / 6.0) * ((skewness as f64).powi(2) + (kurtosis as f64 - 3.0).powi(2) / 4.0);
    let p_value = (-statistic / 2.0).exp();
    (statistic as f32, p_value as f32)
}

/// Per-channel statistical moments, extrema and percentiles.
pub(crate) fn tensor_statistics(pooled: &PooledSamples) -> TensorStatistics {
    let channels = pooled.per_channel.len();
    let mut mean = Vec::with_capacity(channels);
    let mut std = Vec::with_capacity(channels);
    let mut min = Vec::with_capacity(channels);
    let mut max = Vec::with_capacity(channels);
    let mut percentiles = Vec::with_capacity(channels);
    let mut skewness = Vec::with_capacity(channels);
    let mut kurtosis = Vec::with_capacity(channels);

    for channel_values in &pooled.per_channel {
        let (channel_mean, variance, channel_skew, channel_kurt) = moments(channel_values);
        mean.push(channel_mean);
        std.push(variance.max(0.0).sqrt());
        skewness.push(channel_skew);
        kurtosis.push(channel_kurt);

        let mut sorted = channel_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        min.push(sorted.first().copied().unwrap_or(0.0));
        max.push(sorted.last().copied().unwrap_or(0.0));
        percentiles
            .push(REPORTED_PERCENTILES.iter().map(|&p| percentile_of_sorted(&sorted, p)).collect());
    }

    TensorStatistics {
        mean,
        std,
        min,
        max,
        percentiles,
        skewness,
        kurtosis,
    }
}

/// Measured dynamic range, outlier ratio and suggested clipping thresholds.
///
/// `outlier_ratio` is the fraction of pooled values further than three standard
/// deviations from the pooled mean. The suggested clip range is the 0.1th /
/// 99.9th percentile of the pooled values.
pub(crate) fn dynamic_range(pooled: &PooledSamples) -> DynamicRange {
    let mut sorted = pooled.all.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let global_min = sorted.first().copied().unwrap_or(0.0);
    let global_max = sorted.last().copied().unwrap_or(0.0);

    let (mean, variance, _, _) = moments(&pooled.all);
    let deviation = variance.max(0.0).sqrt();
    let outlier_ratio = if deviation > 0.0 && !pooled.all.is_empty() {
        let threshold = 3.0 * deviation;
        let outliers = pooled.all.iter().filter(|&&value| (value - mean).abs() > threshold).count();
        outliers as f32 / pooled.all.len() as f32
    } else {
        0.0
    };

    let channel_ranges = pooled
        .per_channel
        .iter()
        .map(|values| {
            let channel_min = values.iter().copied().fold(f32::INFINITY, f32::min);
            let channel_max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if channel_min.is_finite() && channel_max.is_finite() {
                channel_max - channel_min
            } else {
                0.0
            }
        })
        .collect();

    DynamicRange {
        overall_range: global_max - global_min,
        channel_ranges,
        outlier_ratio,
        suggested_clip_min: percentile_of_sorted(&sorted, 0.1),
        suggested_clip_max: percentile_of_sorted(&sorted, 99.9),
    }
}

/// Measured distribution analysis of the pooled values.
///
/// `distribution_type` is a documented heuristic over the *measured* moments and
/// histogram (never a fixed answer): multimodality first, then the
/// skewness/kurtosis signature of the common unimodal families.
pub(crate) fn distribution_analysis(pooled: &PooledSamples) -> DistributionAnalysis {
    let values = &pooled.all;
    let (mean, variance, skewness, kurtosis) = moments(values);
    let _ = mean;
    let _ = variance;

    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let counts = histogram(values, min, max, HISTOGRAM_BINS);
    let entropy = histogram_entropy_bits(&counts);
    let max_entropy = (HISTOGRAM_BINS as f32).log2();
    let concentration = if max_entropy > 0.0 {
        (1.0 - entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let mode_count = count_modes(&counts);
    let is_multimodal = mode_count > 1;
    let (_, normality_p_value) = jarque_bera(values);

    let distribution_type = if is_multimodal {
        DistributionType::Multimodal
    } else if kurtosis < 2.2 && skewness.abs() < 0.5 {
        DistributionType::Uniform
    } else if skewness.abs() < 0.5 && (kurtosis - 3.0).abs() < 1.0 {
        DistributionType::Normal
    } else if skewness > 1.0 && min >= 0.0 {
        DistributionType::Exponential
    } else if skewness.abs() < 0.75 && kurtosis > 4.5 {
        DistributionType::Laplace
    } else {
        DistributionType::Unknown
    };

    DistributionAnalysis {
        distribution_type,
        normality_p_value,
        entropy,
        concentration,
        is_multimodal,
        mode_count: Some(mode_count),
    }
}

/// Compute the full [`DatasetStatistics`] of a calibration dataset.
pub(crate) fn dataset_statistics(
    samples: &[Tensor],
) -> Result<DatasetStatistics, TrustformersError> {
    let pooled = pool_samples(samples)?;
    let input_shapes = samples.iter().map(|sample| sample.shape().to_vec()).collect();

    Ok(DatasetStatistics {
        sample_count: samples.len(),
        input_shapes,
        statistics: tensor_statistics(&pooled),
        dynamic_range: dynamic_range(&pooled),
        distribution: distribution_analysis(&pooled),
    })
}

/// An affine quantization mapping: `q = clamp(round(x / scale) + zero_point)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AffineQuantization {
    /// Step size between adjacent quantization levels.
    pub scale: f32,
    /// Integer value that the real value `0.0` maps to.
    pub zero_point: i32,
    /// Lowest representable integer level.
    pub q_min: i32,
    /// Highest representable integer level.
    pub q_max: i32,
}

impl AffineQuantization {
    /// Derive the affine mapping that covers `[min, max]` with `bits` levels.
    ///
    /// Symmetric mode uses a zero point of 0 and the range `[-2^(bits-1),
    /// 2^(bits-1) - 1]`; asymmetric mode uses `[0, 2^bits - 1]`.
    pub fn from_range(
        min: f32,
        max: f32,
        bits: u32,
        symmetric: bool,
    ) -> Result<Self, TrustformersError> {
        if !(1..=16).contains(&bits) {
            return Err(invalid_input(format!(
                "Quantization bit width must be in 1..=16, got {}",
                bits
            )));
        }
        if !min.is_finite() || !max.is_finite() || min > max {
            return Err(invalid_input(format!(
                "Invalid quantization range [{}, {}]",
                min, max
            )));
        }

        if symmetric {
            let q_max = (1i32 << (bits - 1)) - 1;
            let q_min = -(1i32 << (bits - 1));
            let max_abs = min.abs().max(max.abs());
            // A degenerate (all-equal) range still needs a usable step size.
            let scale = if max_abs > 0.0 && q_max > 0 { max_abs / q_max as f32 } else { 1.0 };
            Ok(Self {
                scale,
                zero_point: 0,
                q_min,
                q_max,
            })
        } else {
            let q_min = 0i32;
            let q_max = (1i32 << bits) - 1;
            // The affine grid must contain the real value 0 so that the zero point
            // is representable; otherwise a range like [2, 6] produces a zero
            // point of -128, which clamping would then silently destroy.
            let range_min = min.min(0.0);
            let range_max = max.max(0.0);
            let range = range_max - range_min;
            let scale = if range > 0.0 { range / (q_max - q_min) as f32 } else { 1.0 };
            let zero_point = (q_min as f32 - range_min / scale).round() as i32;
            Ok(Self {
                scale,
                zero_point: zero_point.clamp(q_min, q_max),
                q_min,
                q_max,
            })
        }
    }

    /// Quantize a single value to its integer level.
    #[inline]
    pub fn quantize(&self, value: f32) -> i32 {
        ((value / self.scale).round() as i32 + self.zero_point).clamp(self.q_min, self.q_max)
    }

    /// Reconstruct the real value represented by an integer level.
    #[inline]
    pub fn dequantize(&self, level: i32) -> f32 {
        (level - self.zero_point) as f32 * self.scale
    }

    /// Quantize and immediately dequantize every value (the "fake quantization"
    /// round trip used to measure calibration quality).
    pub fn round_trip(&self, values: &[f32]) -> Vec<f32> {
        values.iter().map(|&value| self.dequantize(self.quantize(value))).collect()
    }
}

/// Mean squared error between two equally long sequences.
pub(crate) fn mean_squared_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    if original.is_empty() || original.len() != reconstructed.len() {
        return 0.0;
    }
    let sum: f64 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum();
    (sum / original.len() as f64) as f32
}

/// Signal-to-quantization-noise ratio in dB.
///
/// Returns `f32::INFINITY` when the reconstruction is exact and `f32::NEG_INFINITY`
/// when the signal itself is all zeros but the reconstruction is not.
pub(crate) fn sqnr_db(original: &[f32], reconstructed: &[f32]) -> f32 {
    if original.is_empty() || original.len() != reconstructed.len() {
        return 0.0;
    }
    let mut signal = 0.0f64;
    let mut noise = 0.0f64;
    for (&a, &b) in original.iter().zip(reconstructed.iter()) {
        signal += (a as f64) * (a as f64);
        let d = a as f64 - b as f64;
        noise += d * d;
    }
    if noise == 0.0 {
        return f32::INFINITY;
    }
    if signal == 0.0 {
        return f32::NEG_INFINITY;
    }
    (10.0 * (signal / noise).log10()) as f32
}

/// Pearson correlation between the original and reconstructed values.
///
/// Used as a measured "activation pattern preservation" score in `[-1, 1]`.
pub(crate) fn pearson_correlation(original: &[f32], reconstructed: &[f32]) -> f32 {
    let n = original.len();
    if n == 0 || n != reconstructed.len() {
        return 0.0;
    }
    let n_f = n as f64;
    let mean_a = original.iter().map(|&v| v as f64).sum::<f64>() / n_f;
    let mean_b = reconstructed.iter().map(|&v| v as f64).sum::<f64>() / n_f;

    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;
    for (&a, &b) in original.iter().zip(reconstructed.iter()) {
        let da = a as f64 - mean_a;
        let db = b as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if var_a <= 0.0 || var_b <= 0.0 {
        // A constant signal has no linear relationship to measure; report perfect
        // preservation only if the reconstruction is constant too.
        return if var_a <= 0.0 && var_b <= 0.0 { 1.0 } else { 0.0 };
    }
    (cov / (var_a.sqrt() * var_b.sqrt())) as f32
}

/// Objective a clipping-range search optimizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClipObjective {
    /// Minimize the mean squared reconstruction error.
    MinimizeMse,
    /// Maximize the signal-to-quantization-noise ratio.
    MaximizeSqnr,
    /// Minimize the KL divergence between the original and reconstructed
    /// value histograms (the objective of entropy calibration).
    MinimizeKlDivergence,
}

/// Search for the clipping range that optimizes `objective`.
///
/// Candidate ranges are derived from the **percentiles of the data itself**
/// (from the full observed range down to the 50th percentile, plus the extreme
/// tails 99.99 / 99.9 / 99.5). Percentile candidates matter: a fixed fraction of
/// the observed range cannot discard a single extreme outlier, because the
/// outlier *is* the range. Every candidate is scored by actually quantizing and
/// dequantizing `values`, so the returned mapping is a measurement, not an
/// estimate.
///
/// Returns `(mapping, clip_min, clip_max)`.
pub(crate) fn search_clip_range(
    values: &[f32],
    bits: u32,
    symmetric: bool,
    objective: ClipObjective,
) -> Result<(AffineQuantization, f32, f32), TrustformersError> {
    if values.is_empty() {
        return Err(invalid_input(
            "Cannot calibrate a clipping range from an empty tensor".to_string(),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid_input(
            "Calibration values contain non-finite entries".to_string(),
        ));
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let observed_min = sorted[0];
    let observed_max = sorted[sorted.len() - 1];

    let reference_hist = match objective {
        ClipObjective::MinimizeKlDivergence => Some(histogram(
            values,
            observed_min,
            observed_max,
            HISTOGRAM_BINS,
        )),
        _ => None,
    };

    // Upper percentiles to try; the lower bound is the mirrored percentile.
    let mut upper_percentiles = vec![100.0f32, 99.99, 99.9, 99.5];
    for step in 0..=24u32 {
        upper_percentiles.push(99.0 - step as f32 * 2.0);
    }

    let mut best: Option<(AffineQuantization, f32, f32, f64)> = None;

    for &upper in &upper_percentiles {
        let upper = upper.clamp(50.0, 100.0);
        let high = percentile_of_sorted(&sorted, upper);
        let low = percentile_of_sorted(&sorted, 100.0 - upper);

        let (candidate_min, candidate_max) = if symmetric {
            let half_width = low.abs().max(high.abs());
            (-half_width, half_width)
        } else {
            (low.min(high), high.max(low))
        };

        let quantization =
            AffineQuantization::from_range(candidate_min, candidate_max, bits, symmetric)?;
        let reconstructed = quantization.round_trip(values);

        let cost = match objective {
            ClipObjective::MinimizeMse => mean_squared_error(values, &reconstructed) as f64,
            ClipObjective::MaximizeSqnr => {
                let ratio_db = sqnr_db(values, &reconstructed);
                if ratio_db.is_infinite() && ratio_db.is_sign_positive() {
                    f64::NEG_INFINITY
                } else {
                    -(ratio_db as f64)
                }
            },
            ClipObjective::MinimizeKlDivergence => {
                let candidate_hist =
                    histogram(&reconstructed, observed_min, observed_max, HISTOGRAM_BINS);
                match reference_hist.as_ref() {
                    Some(reference) => histogram_kl_divergence(reference, &candidate_hist) as f64,
                    None => 0.0,
                }
            },
        };

        let better = match best {
            None => true,
            Some((_, _, _, best_cost)) => cost < best_cost,
        };
        if better {
            best = Some((quantization, candidate_min, candidate_max, cost));
        }
    }

    let (quantization, clip_min, clip_max, _) = best
        .ok_or_else(|| invalid_input("Clipping-range search produced no candidate".to_string()))?;
    Ok((quantization, clip_min, clip_max))
}

/// Percentile-based clipping range.
///
/// `percentile` is the upper percentile (for example `99.99`); the lower bound is
/// its mirror image (`100 - percentile`).
pub(crate) fn percentile_clip_range(
    values: &[f32],
    percentile: f32,
) -> Result<(f32, f32), TrustformersError> {
    if values.is_empty() {
        return Err(invalid_input(
            "Cannot calibrate a percentile range from an empty tensor".to_string(),
        ));
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let upper = percentile.clamp(50.0, 100.0);
    let lower = 100.0 - upper;
    Ok((
        percentile_of_sorted(&sorted, lower),
        percentile_of_sorted(&sorted, upper),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_hand_computed_values() {
        let sorted = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile_of_sorted(&sorted, 0.0), 1.0);
        assert_eq!(percentile_of_sorted(&sorted, 50.0), 3.0);
        assert_eq!(percentile_of_sorted(&sorted, 100.0), 5.0);
        // 25% of 4 intervals = position 1.0 exactly.
        assert_eq!(percentile_of_sorted(&sorted, 25.0), 2.0);
        // 12.5% of 4 intervals = position 0.5 -> halfway between 1 and 2.
        assert!((percentile_of_sorted(&sorted, 12.5) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn moments_match_hand_computed_values() {
        // [1, 2, 3, 4]: mean 2.5, population variance 1.25, skew 0, kurtosis 1.64
        let (mean, variance, skewness, kurtosis) = moments(&[1.0, 2.0, 3.0, 4.0]);
        assert!((mean - 2.5).abs() < 1e-6);
        assert!((variance - 1.25).abs() < 1e-6);
        assert!(skewness.abs() < 1e-6);
        assert!((kurtosis - 1.64).abs() < 1e-4);
    }

    #[test]
    fn histogram_entropy_of_uniform_data_is_log2_bins() {
        // Two equally populated bins -> 1 bit of entropy.
        let counts = vec![10u64, 10];
        assert!((histogram_entropy_bits(&counts) - 1.0).abs() < 1e-6);
        // A single populated bin carries no information.
        assert_eq!(histogram_entropy_bits(&[7, 0, 0]), 0.0);
    }

    #[test]
    fn kl_divergence_is_zero_for_identical_histograms() {
        let counts = vec![3u64, 7, 1, 0];
        assert!(histogram_kl_divergence(&counts, &counts).abs() < 1e-6);
        let other = vec![7u64, 3, 0, 1];
        assert!(histogram_kl_divergence(&counts, &other) > 0.0);
    }

    #[test]
    fn mode_detection_finds_two_peaks() {
        let counts = vec![10u64, 40, 10, 1, 0, 1, 10, 45, 8];
        assert_eq!(count_modes(&counts), 2);
        let unimodal = vec![1u64, 5, 20, 40, 20, 5, 1];
        assert_eq!(count_modes(&unimodal), 1);
    }

    #[test]
    fn affine_quantization_round_trips_within_one_step() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 8.0).collect();
        let quantization =
            AffineQuantization::from_range(-4.0, 4.0, 8, true).expect("valid symmetric range");
        for &value in &values {
            let restored = quantization.dequantize(quantization.quantize(value));
            assert!(
                (restored - value).abs() <= quantization.scale,
                "value {value} restored as {restored}, step {}",
                quantization.scale
            );
        }
    }

    #[test]
    fn asymmetric_quantization_covers_a_shifted_range() {
        let quantization =
            AffineQuantization::from_range(2.0, 6.0, 8, false).expect("valid asymmetric range");
        for value in [2.0f32, 3.5, 6.0] {
            let restored = quantization.dequantize(quantization.quantize(value));
            assert!(
                (restored - value).abs() <= quantization.scale,
                "value {value} restored as {restored}"
            );
        }
    }

    #[test]
    fn sqnr_improves_with_more_bits() {
        let values: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.017).sin()).collect();
        let coarse = AffineQuantization::from_range(-1.0, 1.0, 4, true).expect("4-bit range");
        let fine = AffineQuantization::from_range(-1.0, 1.0, 8, true).expect("8-bit range");
        let coarse_db = sqnr_db(&values, &coarse.round_trip(&values));
        let fine_db = sqnr_db(&values, &fine.round_trip(&values));
        assert!(
            fine_db > coarse_db + 10.0,
            "8-bit SQNR {fine_db} should beat 4-bit {coarse_db} by >10 dB"
        );
    }

    /// Deterministic standard-normal sample (Box-Muller over a fixed LCG).
    fn gaussian_sample(count: usize) -> Vec<f32> {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next_uniform = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Top 24 bits -> (0, 1)
            (((state >> 40) as f32) + 0.5) / 16_777_216.0
        };

        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            let u1 = next_uniform();
            let u2 = next_uniform();
            let radius = (-2.0 * u1.ln()).sqrt();
            out.push(radius * (std::f32::consts::TAU * u2).cos());
            if out.len() < count {
                out.push(radius * (std::f32::consts::TAU * u2).sin());
            }
        }
        out
    }

    /// The clipping search must actually clip when clipping lowers the error.
    ///
    /// For 4-bit quantization of Gaussian data the MSE-optimal threshold sits
    /// near 2.8 sigma, well inside the observed extremes, so an implementation
    /// that just used the full observed range (or ignored its objective) would
    /// return the untouched range and fail here.
    #[test]
    fn clip_search_clips_gaussian_tails() {
        let values = gaussian_sample(4096);
        let observed_max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let (quantization, clip_min, clip_max) =
            search_clip_range(&values, 4, true, ClipObjective::MinimizeMse)
                .expect("clip search succeeds");

        assert!(
            clip_max < observed_max * 0.95,
            "expected the Gaussian tail to be clipped: kept [{clip_min}, {clip_max}] of an \
             observed max of {observed_max}"
        );

        let clipped_error = mean_squared_error(&values, &quantization.round_trip(&values));
        let full_range = AffineQuantization::from_range(-observed_max, observed_max, 4, true)
            .expect("full-range mapping");
        let full_error = mean_squared_error(&values, &full_range.round_trip(&values));
        assert!(
            clipped_error < full_error,
            "clipped MSE {clipped_error} should beat the unclipped {full_error}"
        );
    }

    /// The search must never return a range that scores worse than the full
    /// observed range, whatever the objective -- the full range is one of the
    /// candidates, so a correct search can only improve on it.
    #[test]
    fn clip_search_never_regresses_against_full_range() {
        // Heavy outlier: here plain MSE is *dominated* by the outlier's own
        // clipping error, so the mathematically correct answer is to keep the
        // full range. The search must return that rather than a worse clip.
        let mut values: Vec<f32> = (0..511).map(|i| (i as f32 / 255.0) - 1.0).collect();
        values.push(400.0);

        let observed_min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let observed_max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let half_width = observed_min.abs().max(observed_max.abs());
        let full_range = AffineQuantization::from_range(-half_width, half_width, 4, true)
            .expect("full-range mapping");
        let full_error = mean_squared_error(&values, &full_range.round_trip(&values));

        let (quantization, _, _) = search_clip_range(&values, 4, true, ClipObjective::MinimizeMse)
            .expect("clip search succeeds");
        let chosen_error = mean_squared_error(&values, &quantization.round_trip(&values));
        assert!(
            chosen_error <= full_error * (1.0 + 1e-6),
            "search returned a worse range: {chosen_error} vs {full_error}"
        );

        // The SQNR objective must likewise not regress.
        let (sqnr_quantization, _, _) =
            search_clip_range(&values, 4, true, ClipObjective::MaximizeSqnr)
                .expect("clip search succeeds");
        let chosen_sqnr = sqnr_db(&values, &sqnr_quantization.round_trip(&values));
        let full_sqnr = sqnr_db(&values, &full_range.round_trip(&values));
        assert!(
            chosen_sqnr >= full_sqnr - 1e-3,
            "SQNR search regressed: {chosen_sqnr} dB vs {full_sqnr} dB"
        );
    }

    #[test]
    fn pearson_correlation_of_identical_signals_is_one() {
        let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
        assert!((pearson_correlation(&values, &values) - 1.0).abs() < 1e-5);
        let negated: Vec<f32> = values.iter().map(|v| -v).collect();
        assert!((pearson_correlation(&values, &negated) + 1.0).abs() < 1e-5);
    }

    #[test]
    fn statistics_track_the_input_data() {
        // Channel 0 is constant 1.0, channel 1 ramps 0..8.
        let mut data = Vec::new();
        for i in 0..8 {
            data.push(1.0f32);
            data.push(i as f32);
        }
        let sample = Tensor::from_vec(data, &[8, 2]).expect("sample tensor");
        let statistics = dataset_statistics(&[sample]).expect("statistics");

        assert_eq!(statistics.sample_count, 1);
        assert_eq!(statistics.statistics.mean.len(), 2);
        assert!((statistics.statistics.mean[0] - 1.0).abs() < 1e-6);
        assert!((statistics.statistics.mean[1] - 3.5).abs() < 1e-6);
        assert!(statistics.statistics.std[0].abs() < 1e-6);
        assert!(statistics.statistics.std[1] > 2.0);
        assert_eq!(statistics.statistics.min[1], 0.0);
        assert_eq!(statistics.statistics.max[1], 7.0);
        assert!((statistics.dynamic_range.overall_range - 7.0).abs() < 1e-6);
    }

    #[test]
    fn statistics_differ_for_different_datasets() {
        let narrow = Tensor::from_vec(vec![0.0f32; 16], &[8, 2]).expect("narrow tensor");
        let wide = Tensor::from_vec(
            (0..16).map(|i| i as f32 * 10.0).collect::<Vec<f32>>(),
            &[8, 2],
        )
        .expect("wide tensor");

        let narrow_stats = dataset_statistics(&[narrow]).expect("narrow statistics");
        let wide_stats = dataset_statistics(&[wide]).expect("wide statistics");

        assert!(narrow_stats.dynamic_range.overall_range < wide_stats.dynamic_range.overall_range);
        assert_ne!(
            narrow_stats.statistics.mean, wide_stats.statistics.mean,
            "statistics must depend on the data"
        );
    }

    #[test]
    fn empty_dataset_is_rejected() {
        assert!(dataset_statistics(&[]).is_err());
    }
}
