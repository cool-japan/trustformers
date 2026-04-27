//! ASCII-based visualization tools for loss curves, gradient histograms,
//! layer activation statistics, and attention pattern inspection.
//!
//! All types in this module are completely pure-Rust with no external plot
//! dependencies; they are usable in no-std-adjacent environments and in CI
//! pipelines that lack a display server.

// ============================================================================
// AsciiLossPlotter
// ============================================================================

/// ASCII-based loss curve renderer.
///
/// Produces a grid of `width × height` characters where the y-axis spans
/// `[min_value - pad, max_value + pad]` and the x-axis spans the step range
/// of the supplied data.
///
/// # Example
/// ```
/// use trustformers_debug::visualization::ascii_tools::AsciiLossPlotter;
/// let plotter = AsciiLossPlotter::new(60, 15);
/// let data: Vec<(u64, f32)> = (0..20).map(|i| (i as u64, 2.0 - i as f32 * 0.1)).collect();
/// let lines = plotter.render(&data);
/// assert!(!lines.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct AsciiLossPlotter {
    pub width: usize,
    pub height: usize,
    pub title: String,
}

impl AsciiLossPlotter {
    /// Create a new plotter with the given canvas dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width: width.max(10),
            height: height.max(4),
            title: String::new(),
        }
    }

    /// Attach a title that will appear above the plot.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Render `values` as an ASCII plot.
    ///
    /// Returns an empty `Vec` when fewer than 2 points are supplied.
    pub fn render(&self, values: &[(u64, f32)]) -> Vec<String> {
        if values.len() < 2 {
            return Vec::new();
        }
        self.render_curves(&[values], &["loss"])
    }

    /// Render two overlapping curves (train + val).
    ///
    /// Returns an empty `Vec` when either slice has fewer than 2 points.
    pub fn render_two(&self, train: &[(u64, f32)], val: &[(u64, f32)]) -> Vec<String> {
        if train.len() < 2 || val.len() < 2 {
            return Vec::new();
        }
        self.render_curves(&[train, val], &["train", "val"])
    }

    // ── internal ──────────────────────────────────────────────────────────────

    fn render_curves(&self, curves: &[&[(u64, f32)]], labels: &[&str]) -> Vec<String> {
        // Determine global y range across all curves.
        let all_vals: Vec<f32> = curves.iter().flat_map(|c| c.iter().map(|&(_, v)| v)).collect();
        let y_min = all_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let y_max = all_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let y_pad = ((y_max - y_min) * 0.05).max(1e-6_f32);
        let y_lo = y_min - y_pad;
        let y_hi = y_max + y_pad;
        let y_range = y_hi - y_lo;

        // Determine x range.
        let x_min = curves.iter().flat_map(|c| c.iter().map(|&(s, _)| s)).min().unwrap_or(0);
        let x_max = curves.iter().flat_map(|c| c.iter().map(|&(s, _)| s)).max().unwrap_or(1);

        // Map each step to a column index.
        let map_x = |step: u64| -> usize {
            if x_max == x_min {
                return self.width / 2;
            }
            let frac = (step - x_min) as f64 / (x_max - x_min) as f64;
            ((frac * (self.width - 1) as f64).round() as usize).min(self.width - 1)
        };

        // Map each value to a row index (row 0 = top = y_hi).
        let map_y = |v: f32| -> usize {
            let frac = (v - y_lo) / y_range;
            let row = (self.height as f32 - 1.0) * (1.0 - frac.clamp(0.0, 1.0));
            (row.round() as usize).min(self.height - 1)
        };

        // Build the grid: each cell is a char.
        let glyphs: &[char] = &['*', '+', 'o', 'x', '#'];
        let mut grid = vec![vec![' '; self.width]; self.height];

        for (ci, curve) in curves.iter().enumerate() {
            let glyph = glyphs[ci % glyphs.len()];
            for &(step, v) in *curve {
                let col = map_x(step);
                let row = map_y(v);
                grid[row][col] = glyph;
            }
        }

        // Convert grid to strings, add y-axis labels.
        let y_label_width = 9_usize;
        let separator = "─".repeat(self.width + 1);
        let mut out = Vec::<String>::new();

        if !self.title.is_empty() {
            out.push(format!(
                "{:^width$}",
                self.title,
                width = y_label_width + 1 + self.width
            ));
        }

        for row in 0..self.height {
            let y_val = y_hi - (row as f32 / (self.height - 1) as f32) * y_range;
            let y_label = format!("{:>8.4}", y_val);
            let line: String = grid[row].iter().collect();
            out.push(format!("{}|{}", y_label, line));
        }

        // x-axis
        out.push(format!("{:>width$}+{}", "", separator, width = y_label_width));
        // x labels
        let step_range = x_max - x_min;
        let left = format!("{}", x_min);
        let right = format!("{}", x_max);
        let mid_step = x_min + step_range / 2;
        let mid = format!("{}", mid_step);
        let col_pad = y_label_width + 1;
        let total = col_pad + self.width;
        let mid_pos = col_pad + self.width / 2;
        let mut x_axis_line = vec![' '; total];
        for (i, ch) in left.chars().enumerate() {
            if col_pad + i < total {
                x_axis_line[col_pad + i] = ch;
            }
        }
        let mid_start = mid_pos.saturating_sub(mid.len() / 2);
        for (i, ch) in mid.chars().enumerate() {
            if mid_start + i < total {
                x_axis_line[mid_start + i] = ch;
            }
        }
        let right_start = total.saturating_sub(right.len());
        for (i, ch) in right.chars().enumerate() {
            if right_start + i < total {
                x_axis_line[right_start + i] = ch;
            }
        }
        out.push(x_axis_line.into_iter().collect());

        // Legend
        if labels.len() > 1 || (labels.len() == 1 && !labels[0].is_empty()) {
            let legend: String = labels
                .iter()
                .zip(glyphs.iter())
                .map(|(l, g)| format!("{}={}", g, l))
                .collect::<Vec<_>>()
                .join("  ");
            out.push(format!("{:>width$} {}", "", legend, width = y_label_width));
        }

        out
    }
}

// ============================================================================
// GradientHistogram
// ============================================================================

/// Bucket-based gradient histogram builder.
///
/// The bucket edges are linearly spaced between `[min, max]`
/// (inclusive endpoints stored as the first and last element of `buckets`).
/// Values outside `[min, max]` are clamped into the first/last bucket.
#[derive(Debug, Clone)]
pub struct GradientHistogram {
    /// Bucket edges (length = `num_buckets + 1`).
    pub buckets: Vec<f32>,
    /// Count per bucket (length = `num_buckets`).
    pub counts: Vec<usize>,
    /// Total number of values added.
    pub total_values: usize,
    // Running totals for mean / variance (Welford online algorithm).
    running_mean: f64,
    running_m2: f64,
}

impl GradientHistogram {
    /// Create a new histogram with linearly spaced edges in `[min, max]`.
    ///
    /// Panics if `num_buckets == 0` or `min >= max`.
    pub fn new(num_buckets: usize, min: f32, max: f32) -> Self {
        assert!(num_buckets > 0, "num_buckets must be >= 1");
        assert!(min < max, "min must be less than max");
        let mut buckets = Vec::with_capacity(num_buckets + 1);
        for i in 0..=num_buckets {
            buckets.push(min + (max - min) * (i as f32 / num_buckets as f32));
        }
        Self {
            buckets,
            counts: vec![0; num_buckets],
            total_values: 0,
            running_mean: 0.0,
            running_m2: 0.0,
        }
    }

    /// Add a single value to the histogram.
    pub fn add_value(&mut self, val: f32) {
        let n_buckets = self.counts.len();
        let min = self.buckets[0];
        let max = *self.buckets.last().unwrap_or(&min);
        let range = max - min;
        let bucket_idx = if range <= 0.0 {
            0
        } else {
            let frac = (val - min) / range;
            let idx = (frac * n_buckets as f32).floor() as isize;
            idx.clamp(0, (n_buckets as isize) - 1) as usize
        };
        self.counts[bucket_idx] += 1;

        // Welford online update.
        self.total_values += 1;
        let delta = val as f64 - self.running_mean;
        self.running_mean += delta / self.total_values as f64;
        let delta2 = val as f64 - self.running_mean;
        self.running_m2 += delta * delta2;
    }

    /// Add multiple values.
    pub fn add_values(&mut self, vals: &[f32]) {
        for &v in vals {
            self.add_value(v);
        }
    }

    /// Compute the mean of all inserted values.
    ///
    /// Returns 0.0 if no values have been added.
    pub fn mean(&self) -> f32 {
        if self.total_values == 0 {
            return 0.0;
        }
        self.running_mean as f32
    }

    /// Compute the sample standard deviation of all inserted values.
    ///
    /// Returns 0.0 if fewer than 2 values have been added.
    pub fn std_dev(&self) -> f32 {
        if self.total_values < 2 {
            return 0.0;
        }
        (self.running_m2 / (self.total_values - 1) as f64).sqrt() as f32
    }

    /// Approximate the `p`-th percentile (0.0–100.0) via bucket interpolation.
    ///
    /// Returns the lower bucket edge when counts are insufficient for
    /// interpolation.
    pub fn percentile(&self, p: f32) -> f32 {
        if self.total_values == 0 {
            return self.buckets[0];
        }
        let target = (p.clamp(0.0, 100.0) / 100.0) * self.total_values as f32;
        let mut cum = 0.0_f32;
        for (i, &count) in self.counts.iter().enumerate() {
            let next = cum + count as f32;
            if next >= target {
                // Linear interpolation within bucket i.
                let lo = self.buckets[i];
                let hi = self.buckets[i + 1];
                let bucket_frac = if count == 0 {
                    0.0
                } else {
                    (target - cum) / count as f32
                };
                return lo + bucket_frac * (hi - lo);
            }
            cum = next;
        }
        // Clamp to max if percentile is 100.
        *self.buckets.last().unwrap_or(&self.buckets[0])
    }

    /// Render a compact horizontal bar chart (one line per bucket).
    pub fn to_ascii_bars(&self) -> String {
        if self.counts.is_empty() {
            return "(empty histogram)\n".to_string();
        }
        let max_count = self.counts.iter().copied().max().unwrap_or(0);
        let bar_width = 40_usize;
        let mut out = String::new();
        let n_buckets = self.counts.len();
        for i in 0..n_buckets {
            let lo = self.buckets[i];
            let hi = self.buckets[i + 1];
            let cnt = self.counts[i];
            let bar_len = if max_count == 0 {
                0
            } else {
                (cnt * bar_width) / max_count
            };
            let bar = "█".repeat(bar_len);
            out.push_str(&format!(
                "[{:>8.3e},{:>8.3e}) {:>6} |{}\n",
                lo, hi, cnt, bar
            ));
        }
        out
    }
}

// ============================================================================
// ActivationLayerStats
// ============================================================================

/// Per-layer activation statistics computed in a single pass over the data.
///
/// Named `ActivationLayerStats` to distinguish it from
/// `model_diagnostics::LayerActivationStats` which uses `f64` fields with
/// different names.
#[derive(Debug, Clone)]
pub struct ActivationLayerStats {
    pub layer_name: String,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    /// Fraction of values that are exactly zero.
    pub zero_fraction: f32,
    /// Fraction of values whose absolute magnitude exceeds `0.99 * |max_abs|`.
    pub saturation_fraction: f32,
}

impl ActivationLayerStats {
    /// Compute statistics from raw activation values.
    pub fn compute(layer_name: &str, activations: &[f32]) -> Self {
        if activations.is_empty() {
            return Self {
                layer_name: layer_name.to_string(),
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                zero_fraction: 1.0,
                saturation_fraction: 0.0,
            };
        }

        let n = activations.len() as f64;

        // Single-pass mean & M2 (Welford).
        let mut running_mean = 0.0_f64;
        let mut running_m2 = 0.0_f64;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut zero_count = 0usize;

        for (idx, &v) in activations.iter().enumerate() {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
            if v == 0.0 {
                zero_count += 1;
            }
            let delta = v as f64 - running_mean;
            running_mean += delta / (idx + 1) as f64;
            let delta2 = v as f64 - running_mean;
            running_m2 += delta * delta2;
        }

        let std_val = if activations.len() > 1 {
            (running_m2 / (n - 1.0)).sqrt() as f32
        } else {
            0.0
        };

        // Saturation: |v| > 0.99 * max_abs.
        let max_abs = min_val.abs().max(max_val.abs());
        let sat_threshold = 0.99 * max_abs;
        let sat_count = activations.iter().filter(|&&v| v.abs() > sat_threshold).count();

        Self {
            layer_name: layer_name.to_string(),
            mean: running_mean as f32,
            std: std_val,
            min: min_val,
            max: max_val,
            zero_fraction: zero_count as f32 / activations.len() as f32,
            saturation_fraction: sat_count as f32 / activations.len() as f32,
        }
    }

    /// Returns `true` when the zero fraction exceeds the given threshold.
    ///
    /// A layer is considered "dead" when most of its activations are zero
    /// (common symptom of a collapsed ReLU).
    pub fn is_dead(&self, threshold: f32) -> bool {
        self.zero_fraction > threshold.clamp(0.0, 1.0)
    }

    /// Return a compact one-line summary suitable for logging.
    pub fn to_summary_line(&self) -> String {
        format!(
            "[{}] mean={:.4e}  std={:.4e}  min={:.4e}  max={:.4e}  zero={:.1}%  sat={:.1}%",
            self.layer_name,
            self.mean,
            self.std,
            self.min,
            self.max,
            self.zero_fraction * 100.0,
            self.saturation_fraction * 100.0,
        )
    }
}

// ============================================================================
// AttentionVisualizer
// ============================================================================

/// Attention pattern visualizer producing ASCII heatmaps and entropy metrics.
#[derive(Debug, Clone)]
pub struct AttentionVisualizer {
    pub head_idx: usize,
    pub layer_idx: usize,
}

impl AttentionVisualizer {
    pub fn new(head_idx: usize, layer_idx: usize) -> Self {
        Self { head_idx, layer_idx }
    }

    /// Render an attention matrix as a compact ASCII heatmap.
    ///
    /// Rows = query positions, columns = key positions.
    /// Each cell is encoded with one of `' ', '░', '▒', '▓', '█'`
    /// according to the value's position in the global `[min, max]` range.
    pub fn render_ascii(attn_matrix: &[Vec<f32>]) -> Vec<String> {
        if attn_matrix.is_empty() {
            return Vec::new();
        }
        let blocks = [' ', '░', '▒', '▓', '█'];
        let n_blocks = blocks.len() as f32;

        let (global_min, global_max) = attn_matrix
            .iter()
            .flat_map(|row| row.iter().copied())
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| {
                (mn.min(v), mx.max(v))
            });
        let range = (global_max - global_min).max(1e-9_f32);

        let mut lines = Vec::with_capacity(attn_matrix.len() + 1);
        lines.push(format!(
            "Attn heatmap  min={:.3}  max={:.3}",
            global_min, global_max
        ));

        for row in attn_matrix {
            let encoded: String = row
                .iter()
                .map(|&v| {
                    let idx = ((v - global_min) / range * (n_blocks - 1.0))
                        .round()
                        .clamp(0.0, n_blocks - 1.0) as usize;
                    blocks[idx]
                })
                .collect();
            lines.push(encoded);
        }
        lines
    }

    /// Compute the (Shannon) entropy of a single attention row.
    ///
    /// `attn_row` should sum to approximately 1.0 (softmax output).
    /// Returns 0.0 for empty rows.
    pub fn entropy(attn_row: &[f32]) -> f32 {
        if attn_row.is_empty() {
            return 0.0;
        }
        // Normalise defensively to handle near-softmax vectors.
        let total: f32 = attn_row.iter().sum();
        let scale = if total > 0.0 { total } else { 1.0 };
        -attn_row
            .iter()
            .filter_map(|&p| {
                let pn = p / scale;
                if pn > 0.0 { Some(pn * pn.ln()) } else { None }
            })
            .sum::<f32>()
    }

    /// Compute per-row entropy for the full attention matrix.
    pub fn all_entropies(attn_matrix: &[Vec<f32>]) -> Vec<f32> {
        attn_matrix.iter().map(|row| Self::entropy(row)).collect()
    }

    /// Returns `true` when the range of the row is smaller than `eps`,
    /// indicating a near-uniform (or constant) attention distribution.
    pub fn is_uniform(attn_row: &[f32], eps: f32) -> bool {
        if attn_row.is_empty() {
            return true;
        }
        let max_v = attn_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_v = attn_row.iter().cloned().fold(f32::INFINITY, f32::min);
        (max_v - min_v) < eps
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── AsciiLossPlotter ────────────────────────────────────────────────────

    #[test]
    fn test_plotter_returns_empty_for_single_point() {
        let plotter = AsciiLossPlotter::new(60, 15);
        let data = vec![(0u64, 1.5_f32)];
        assert!(plotter.render(&data).is_empty());
    }

    #[test]
    fn test_plotter_returns_empty_for_zero_points() {
        let plotter = AsciiLossPlotter::new(60, 15);
        assert!(plotter.render(&[]).is_empty());
    }

    #[test]
    fn test_plotter_two_point_render() {
        let plotter = AsciiLossPlotter::new(40, 10);
        let data = vec![(0u64, 2.0_f32), (10u64, 1.0_f32)];
        let lines = plotter.render(&data);
        assert!(!lines.is_empty(), "should produce output for 2 points");
        // The plot should contain at least the axis separator.
        assert!(lines.iter().any(|l| l.contains('+')));
    }

    #[test]
    fn test_plotter_many_points_produce_canvas_height() {
        let height = 12;
        let plotter = AsciiLossPlotter::new(60, height);
        let data: Vec<(u64, f32)> = (0..30).map(|i| (i as u64, 3.0 - i as f32 * 0.1)).collect();
        let lines = plotter.render(&data);
        // height body rows + separator + x-label + optional legend ≥ height + 2
        assert!(lines.len() >= height + 2);
    }

    #[test]
    fn test_plotter_with_title() {
        let plotter = AsciiLossPlotter::new(50, 10).with_title("Training Loss");
        let data: Vec<(u64, f32)> = (0..5).map(|i| (i as u64, 1.0)).collect();
        let lines = plotter.render(&data);
        assert!(lines[0].contains("Training Loss"));
    }

    #[test]
    fn test_plotter_render_two_returns_empty_if_not_enough_points() {
        let plotter = AsciiLossPlotter::new(40, 10);
        let one_point = vec![(0u64, 1.0_f32)];
        let many = vec![(0u64, 1.0_f32), (1u64, 0.9_f32)];
        assert!(plotter.render_two(&one_point, &many).is_empty());
        assert!(plotter.render_two(&many, &one_point).is_empty());
    }

    #[test]
    fn test_plotter_render_two_produces_legend_markers() {
        let plotter = AsciiLossPlotter::new(50, 10);
        let train: Vec<(u64, f32)> = (0..5).map(|i| (i as u64, 2.0 - i as f32 * 0.3)).collect();
        let val: Vec<(u64, f32)> = (0..5).map(|i| (i as u64, 2.1 - i as f32 * 0.25)).collect();
        let lines = plotter.render_two(&train, &val);
        // Legend line should contain both labels.
        let legend = lines.last().unwrap();
        assert!(legend.contains("train") || legend.contains('*'));
        assert!(legend.contains("val") || legend.contains('+'));
    }

    // ── GradientHistogram ───────────────────────────────────────────────────

    #[test]
    fn test_histogram_bucket_assignment() {
        let mut h = GradientHistogram::new(4, 0.0, 4.0);
        // Each bucket covers 1.0 unit: [0,1), [1,2), [2,3), [3,4]
        h.add_value(0.5); // bucket 0
        h.add_value(1.5); // bucket 1
        h.add_value(2.5); // bucket 2
        h.add_value(3.5); // bucket 3
        assert_eq!(h.counts, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_histogram_clamping_below_min() {
        let mut h = GradientHistogram::new(4, 0.0, 4.0);
        h.add_value(-100.0); // should land in bucket 0
        assert_eq!(h.counts[0], 1);
    }

    #[test]
    fn test_histogram_clamping_above_max() {
        let mut h = GradientHistogram::new(4, 0.0, 4.0);
        h.add_value(999.0); // should land in bucket 3
        assert_eq!(h.counts[3], 1);
    }

    #[test]
    fn test_histogram_mean() {
        let mut h = GradientHistogram::new(10, 0.0, 10.0);
        // Add 1, 2, 3 → mean should be 2.
        h.add_values(&[1.0, 2.0, 3.0]);
        let mean = h.mean();
        assert!((mean - 2.0).abs() < 1e-4, "expected mean≈2.0, got {}", mean);
    }

    #[test]
    fn test_histogram_std_dev() {
        let mut h = GradientHistogram::new(10, -10.0, 10.0);
        // Population: 0,0,0,0 → std should be 0.
        h.add_values(&[0.0, 0.0, 0.0, 0.0]);
        assert!(h.std_dev() < 1e-4);
    }

    #[test]
    fn test_histogram_std_dev_known_value() {
        let mut h = GradientHistogram::new(20, 0.0, 10.0);
        // Values 2 and 8 → sample std dev = 3√2 ≈ 4.243.
        h.add_values(&[2.0, 8.0]);
        let std = h.std_dev();
        assert!((std - (18.0_f32).sqrt()).abs() < 0.1, "std={}", std);
    }

    #[test]
    fn test_histogram_percentile_median() {
        let mut h = GradientHistogram::new(100, 0.0, 100.0);
        // Uniform distribution 0..100 → median ≈ 50.
        let vals: Vec<f32> = (0..100).map(|i| i as f32).collect();
        h.add_values(&vals);
        let p50 = h.percentile(50.0);
        assert!((p50 - 50.0).abs() < 2.0, "p50={}", p50);
    }

    #[test]
    fn test_histogram_percentile_0_and_100() {
        let mut h = GradientHistogram::new(10, 0.0, 10.0);
        h.add_values(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let p0 = h.percentile(0.0);
        let p100 = h.percentile(100.0);
        assert!(p0 <= p100, "p0={}, p100={}", p0, p100);
    }

    #[test]
    fn test_histogram_ascii_bars_non_empty() {
        let mut h = GradientHistogram::new(5, 0.0, 5.0);
        h.add_values(&[0.5, 1.5, 2.5, 3.5, 4.5]);
        let bars = h.to_ascii_bars();
        assert!(!bars.is_empty());
        // Each bucket line contains '|'.
        assert!(bars.lines().all(|l| l.contains('|') || l.is_empty()));
    }

    #[test]
    fn test_histogram_empty_to_ascii_bars() {
        let h = GradientHistogram::new(4, 0.0, 4.0);
        let bars = h.to_ascii_bars();
        // With zero counts the histogram still has bucket lines.
        assert!(!bars.is_empty());
    }

    // ── ActivationLayerStats ────────────────────────────────────────────────

    #[test]
    fn test_activation_stats_empty() {
        let stats = ActivationLayerStats::compute("empty_layer", &[]);
        assert_eq!(stats.zero_fraction, 1.0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_activation_stats_mean() {
        let vals = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let stats = ActivationLayerStats::compute("fc1", &vals);
        assert!((stats.mean - 3.0).abs() < 1e-4, "mean={}", stats.mean);
    }

    #[test]
    fn test_activation_stats_std() {
        // All identical → std = 0.
        let vals = vec![5.0_f32; 10];
        let stats = ActivationLayerStats::compute("relu", &vals);
        assert!(stats.std < 1e-4, "std={}", stats.std);
    }

    #[test]
    fn test_activation_stats_zeros() {
        let vals = vec![0.0_f32, 0.0, 1.0, 2.0];
        let stats = ActivationLayerStats::compute("dead_relu", &vals);
        // 2 out of 4 are zero → 0.5.
        assert!((stats.zero_fraction - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_activation_stats_is_dead() {
        let vals: Vec<f32> = (0..10).map(|i| if i < 9 { 0.0 } else { 1.0 }).collect();
        let stats = ActivationLayerStats::compute("mostly_dead", &vals);
        // 9/10 = 0.9 zeros.
        assert!(stats.is_dead(0.8));
        assert!(!stats.is_dead(0.95));
    }

    #[test]
    fn test_activation_stats_summary_line_contains_name() {
        let vals = vec![0.1_f32, 0.2, 0.3];
        let stats = ActivationLayerStats::compute("encoder_0", &vals);
        let summary = stats.to_summary_line();
        assert!(summary.contains("encoder_0"));
    }

    #[test]
    fn test_activation_stats_saturation_fraction() {
        // All values at max → saturation should be 1.0.
        let vals = vec![1.0_f32; 5];
        let stats = ActivationLayerStats::compute("saturated", &vals);
        // max_abs=1.0, all vals have abs=1.0 > 0.99*1.0 → all saturated.
        assert!(
            stats.saturation_fraction >= 0.99,
            "saturation={}",
            stats.saturation_fraction
        );
    }

    // ── AttentionVisualizer ─────────────────────────────────────────────────

    #[test]
    fn test_attention_entropy_uniform() {
        // Uniform attention over 4 tokens: entropy = ln(4) ≈ 1.386.
        let row = vec![0.25_f32; 4];
        let h = AttentionVisualizer::entropy(&row);
        assert!((h - (4.0_f32).ln()).abs() < 0.01, "entropy={}", h);
    }

    #[test]
    fn test_attention_entropy_concentrated() {
        // All mass on one token → entropy = 0.
        let row = vec![0.0_f32, 0.0, 1.0, 0.0];
        let h = AttentionVisualizer::entropy(&row);
        assert!(h < 1e-4, "entropy for concentrated distribution should be ~0, got {}", h);
    }

    #[test]
    fn test_attention_entropy_empty_row() {
        assert_eq!(AttentionVisualizer::entropy(&[]), 0.0);
    }

    #[test]
    fn test_attention_all_entropies_per_row() {
        let matrix = vec![
            vec![0.25_f32; 4],  // uniform
            vec![0.0, 0.0, 1.0, 0.0], // concentrated
        ];
        let entropies = AttentionVisualizer::all_entropies(&matrix);
        assert_eq!(entropies.len(), 2);
        assert!(entropies[0] > entropies[1], "uniform > concentrated");
    }

    #[test]
    fn test_attention_is_uniform_true() {
        // All weights equal → uniform.
        let row = vec![0.25_f32; 4];
        assert!(AttentionVisualizer::is_uniform(&row, 0.01));
    }

    #[test]
    fn test_attention_is_uniform_false() {
        let row = vec![0.9_f32, 0.05, 0.03, 0.02];
        assert!(!AttentionVisualizer::is_uniform(&row, 0.01));
    }

    #[test]
    fn test_attention_render_ascii_empty() {
        let result = AttentionVisualizer::render_ascii(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_attention_render_ascii_shape() {
        let matrix = vec![vec![0.1_f32; 5]; 4]; // 4 rows × 5 cols
        let lines = AttentionVisualizer::render_ascii(&matrix);
        // 1 header + 4 rows = 5 lines.
        assert_eq!(lines.len(), 5);
        // Each data row is exactly 5 characters wide.
        for line in &lines[1..] {
            // Unicode chars; count chars not bytes.
            let char_count: usize = line.chars().count();
            assert_eq!(char_count, 5, "expected 5 chars, got {}", char_count);
        }
    }

    #[test]
    fn test_attention_visualizer_new() {
        let av = AttentionVisualizer::new(3, 1);
        assert_eq!(av.head_idx, 3);
        assert_eq!(av.layer_idx, 1);
    }
}
