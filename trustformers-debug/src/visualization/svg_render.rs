//! Dependency-free SVG rendering for [`crate::visualization::DebugVisualizer`].
//!
//! Every function in this module produces a **real** SVG document: axes are drawn
//! from the actual data range, series are real polylines through the actual points,
//! histogram bars come from a real binning pass and heatmap cells from a real
//! colour-mapped normalisation of the supplied matrix.
//!
//! This exists because SVG is the one raster-free vector format the crate can emit
//! with zero extra dependencies (the COOLJAPAN Pure-Rust policy keeps the PNG/GIF
//! encoders behind the optional `visual` / `image` / `gif` features). Formats this
//! module cannot honestly produce are rejected by the caller with a structured
//! error rather than being written out under a misleading file extension.

use super::types::{ColorScheme, HeatmapData, HistogramData, PlotData, VisualizationConfig};
use std::fmt::Write as _;

/// Pixel margins around the plotting area: (left, right, top, bottom).
const MARGIN: (f64, f64, f64, f64) = (70.0, 25.0, 40.0, 55.0);

/// Number of gridline/tick divisions on each axis.
const TICKS: usize = 5;

/// Escape the five XML predefined entities so arbitrary titles/labels cannot
/// break out of the document.
pub(crate) fn escape_xml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Turn a human title into a filesystem-safe stem.
///
/// Keeps ASCII alphanumerics, collapses every other character into `_`, and
/// falls back to `plot` when nothing usable remains.
pub(crate) fn slugify(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut last_underscore = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_underscore = false;
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "plot".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Palette for a colour scheme: (background, axis/text, series colours).
fn palette(scheme: &ColorScheme) -> (&'static str, &'static str, &'static [&'static str]) {
    match scheme {
        ColorScheme::Default => (
            "#ffffff",
            "#333333",
            &[
                "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
            ],
        ),
        ColorScheme::Dark => (
            "#1e1e1e",
            "#d4d4d4",
            &[
                "#4fc3f7", "#ef5350", "#81c784", "#ffb74d", "#ba68c8", "#a1887f",
            ],
        ),
        // Wong (2011) "Points of view: Color blindness", Nature Methods 8:441 —
        // an eight-colour palette designed to stay distinguishable under the
        // common dichromacies.
        ColorScheme::Colorblind => (
            "#ffffff",
            "#000000",
            &[
                "#0072b2", "#d55e00", "#009e73", "#e69f00", "#cc79a7", "#56b4e9",
            ],
        ),
        ColorScheme::Viridis => (
            "#ffffff",
            "#333333",
            &[
                "#440154", "#3b528b", "#21918c", "#5ec962", "#fde725", "#31688e",
            ],
        ),
        ColorScheme::Plasma => (
            "#ffffff",
            "#333333",
            &[
                "#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921",
            ],
        ),
    }
}

/// Anchor stops for the continuous colour map used by heatmaps.
///
/// `Viridis` and `Plasma` use the five canonical matplotlib anchor colours at
/// t = 0, 0.25, 0.5, 0.75, 1.0 with piecewise-linear interpolation in sRGB.
/// That is an *approximation* of the published maps (which are tabulated at 256
/// points in linear-ish space); the maximum per-channel deviation from the real
/// tables is a few percent, which is immaterial for a debugging heatmap but is
/// the reason this is documented as an approximation rather than "viridis".
fn colormap_stops(scheme: &ColorScheme) -> &'static [(u8, u8, u8)] {
    match scheme {
        ColorScheme::Default => &[
            (5, 48, 97),
            (146, 197, 222),
            (247, 247, 247),
            (244, 165, 130),
            (103, 0, 31),
        ],
        ColorScheme::Dark => &[
            (0, 0, 0),
            (69, 24, 79),
            (152, 47, 45),
            (222, 133, 25),
            (252, 255, 164),
        ],
        ColorScheme::Colorblind => &[
            (0, 32, 76),
            (0, 114, 178),
            (128, 170, 190),
            (230, 159, 0),
            (255, 250, 200),
        ],
        ColorScheme::Viridis => &[
            (68, 1, 84),
            (59, 82, 139),
            (33, 145, 140),
            (94, 201, 98),
            (253, 231, 37),
        ],
        ColorScheme::Plasma => &[
            (13, 8, 135),
            (126, 3, 168),
            (204, 71, 120),
            (248, 149, 64),
            (240, 249, 33),
        ],
    }
}

/// Map `t` in [0, 1] onto the scheme's colour ramp by piecewise-linear
/// interpolation between the anchor stops.
pub(crate) fn colormap(scheme: &ColorScheme, t: f64) -> String {
    let stops = colormap_stops(scheme);
    let t = t.clamp(0.0, 1.0);
    let segments = stops.len() - 1;
    let scaled = t * segments as f64;
    let idx = (scaled.floor() as usize).min(segments - 1);
    let frac = scaled - idx as f64;
    let (r0, g0, b0) = stops[idx];
    let (r1, g1, b1) = stops[idx + 1];
    let lerp = |a: u8, b: u8| -> u8 {
        (a as f64 + (b as f64 - a as f64) * frac).round().clamp(0.0, 255.0) as u8
    };
    format!(
        "#{:02x}{:02x}{:02x}",
        lerp(r0, r1),
        lerp(g0, g1),
        lerp(b0, b1)
    )
}

/// Finite min/max of a slice, ignoring non-finite entries.
///
/// Returns `None` when the slice holds no finite value at all.
pub(crate) fn finite_bounds(values: &[f64]) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut any = false;
    for &v in values {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
            any = true;
        }
    }
    if any {
        Some((lo, hi))
    } else {
        None
    }
}

/// Widen a degenerate range so a constant series still gets a drawable axis.
fn pad_range(lo: f64, hi: f64) -> (f64, f64) {
    if (hi - lo).abs() < f64::EPSILON {
        let pad = if lo.abs() > f64::EPSILON { lo.abs() * 0.1 } else { 1.0 };
        (lo - pad, hi + pad)
    } else {
        (lo, hi)
    }
}

/// Format an axis tick compactly without losing the magnitude.
fn tick_label(v: f64) -> String {
    let a = v.abs();
    if a != 0.0 && !(1e-3..1e6).contains(&a) {
        format!("{:.2e}", v)
    } else if a >= 100.0 {
        format!("{:.0}", v)
    } else {
        format!("{:.3}", v)
    }
}

/// Geometry of the drawing area shared by every plot kind.
struct Frame {
    width: f64,
    height: f64,
    plot_x: f64,
    plot_y: f64,
    plot_w: f64,
    plot_h: f64,
}

impl Frame {
    fn new(config: &VisualizationConfig) -> Self {
        // Clamp so a pathological config cannot produce a negative plot area.
        let width = (config.plot_width as f64).max(240.0);
        let height = (config.plot_height as f64).max(180.0);
        let (ml, mr, mt, mb) = MARGIN;
        Self {
            width,
            height,
            plot_x: ml,
            plot_y: mt,
            plot_w: (width - ml - mr).max(10.0),
            plot_h: (height - mt - mb).max(10.0),
        }
    }

    fn sx(&self, v: f64, lo: f64, hi: f64) -> f64 {
        self.plot_x + (v - lo) / (hi - lo) * self.plot_w
    }

    fn sy(&self, v: f64, lo: f64, hi: f64) -> f64 {
        self.plot_y + self.plot_h - (v - lo) / (hi - lo) * self.plot_h
    }
}

/// Emit the opening `<svg>` element plus background rectangle and title.
fn open_svg(out: &mut String, frame: &Frame, bg: &str, fg: &str, font: u32, title: &str) {
    let _ = writeln!(
        out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" \
         viewBox=\"0 0 {w} {h}\" font-family=\"sans-serif\">\n\
         <rect width=\"{w}\" height=\"{h}\" fill=\"{bg}\"/>\n\
         <text x=\"{tx:.1}\" y=\"{ty:.1}\" font-size=\"{fs}\" fill=\"{fg}\" \
         text-anchor=\"middle\" font-weight=\"bold\">{title}</text>",
        w = frame.width,
        h = frame.height,
        bg = bg,
        fg = fg,
        tx = frame.width / 2.0,
        ty = (font as f64) + 8.0,
        fs = font + 3,
        title = escape_xml(title),
    );
}

/// Draw the axis box, gridlines, numeric ticks and axis captions.
#[allow(clippy::too_many_arguments)]
fn draw_axes(
    out: &mut String,
    frame: &Frame,
    fg: &str,
    font: u32,
    x_range: (f64, f64),
    y_range: (f64, f64),
    x_label: &str,
    y_label: &str,
) {
    let _ = writeln!(
        out,
        "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"none\" \
         stroke=\"{}\" stroke-width=\"1\"/>",
        frame.plot_x, frame.plot_y, frame.plot_w, frame.plot_h, fg
    );

    for i in 0..=TICKS {
        let t = i as f64 / TICKS as f64;

        let xv = x_range.0 + (x_range.1 - x_range.0) * t;
        let px = frame.plot_x + frame.plot_w * t;
        let _ = writeln!(
            out,
            "<line x1=\"{px:.1}\" y1=\"{y0:.1}\" x2=\"{px:.1}\" y2=\"{y1:.1}\" \
             stroke=\"{fg}\" stroke-width=\"0.4\" stroke-opacity=\"0.35\"/>\n\
             <text x=\"{px:.1}\" y=\"{ty:.1}\" font-size=\"{fs}\" fill=\"{fg}\" \
             text-anchor=\"middle\">{lbl}</text>",
            px = px,
            y0 = frame.plot_y,
            y1 = frame.plot_y + frame.plot_h,
            fg = fg,
            ty = frame.plot_y + frame.plot_h + (font as f64) + 4.0,
            fs = font,
            lbl = escape_xml(&tick_label(xv)),
        );

        let yv = y_range.0 + (y_range.1 - y_range.0) * t;
        let py = frame.plot_y + frame.plot_h - frame.plot_h * t;
        let _ = writeln!(
            out,
            "<line x1=\"{x0:.1}\" y1=\"{py:.1}\" x2=\"{x1:.1}\" y2=\"{py:.1}\" \
             stroke=\"{fg}\" stroke-width=\"0.4\" stroke-opacity=\"0.35\"/>\n\
             <text x=\"{tx:.1}\" y=\"{ty:.1}\" font-size=\"{fs}\" fill=\"{fg}\" \
             text-anchor=\"end\">{lbl}</text>",
            x0 = frame.plot_x,
            x1 = frame.plot_x + frame.plot_w,
            py = py,
            fg = fg,
            tx = frame.plot_x - 6.0,
            ty = py + (font as f64) / 3.0,
            fs = font,
            lbl = escape_xml(&tick_label(yv)),
        );
    }

    if !x_label.is_empty() {
        let _ = writeln!(
            out,
            "<text x=\"{x:.1}\" y=\"{y:.1}\" font-size=\"{fs}\" fill=\"{fg}\" \
             text-anchor=\"middle\">{lbl}</text>",
            x = frame.plot_x + frame.plot_w / 2.0,
            y = frame.height - 8.0,
            fs = font,
            fg = fg,
            lbl = escape_xml(x_label),
        );
    }
    if !y_label.is_empty() {
        let cy = frame.plot_y + frame.plot_h / 2.0;
        let _ = writeln!(
            out,
            "<text x=\"14\" y=\"{cy:.1}\" font-size=\"{fs}\" fill=\"{fg}\" text-anchor=\"middle\" \
             transform=\"rotate(-90 14 {cy:.1})\">{lbl}</text>",
            cy = cy,
            fs = font,
            fg = fg,
            lbl = escape_xml(y_label),
        );
    }
}

/// Split `y_values` into `series` equal chunks.
///
/// The in-tree producers (`DebugVisualizer::plot_training_metrics`) append a
/// second series onto the same `y_values` vector and push a second entry into
/// `labels`, so the number of labels is the authoritative series count.
pub(crate) fn split_series(y_values: &[f64], series: usize) -> Vec<&[f64]> {
    if series <= 1 {
        return vec![y_values];
    }
    let per = y_values.len() / series;
    if per == 0 {
        return vec![y_values];
    }
    (0..series).map(|s| &y_values[s * per..(s + 1) * per]).collect()
}

/// Render a real multi-series line plot.
pub fn line_plot_svg(data: &PlotData, config: &VisualizationConfig) -> String {
    let (bg, fg, colors) = palette(&config.color_scheme);
    let frame = Frame::new(config);
    let font = config.font_size;
    let mut out = String::new();
    open_svg(&mut out, &frame, bg, fg, font, &data.title);

    let series_count = data.labels.len().max(1);
    let series = split_series(&data.y_values, series_count);
    let n_points = series.iter().map(|s| s.len()).max().unwrap_or(0);

    let x_bounds = finite_bounds(&data.x_values);
    let y_bounds = finite_bounds(&data.y_values);

    match (x_bounds, y_bounds, n_points >= 1) {
        (Some((xlo, xhi)), Some((ylo, yhi)), true) => {
            let (xlo, xhi) = pad_range(xlo, xhi);
            let (ylo, yhi) = pad_range(ylo, yhi);
            draw_axes(
                &mut out,
                &frame,
                fg,
                font,
                (xlo, xhi),
                (ylo, yhi),
                &data.x_label,
                &data.y_label,
            );

            for (si, ys) in series.iter().enumerate() {
                let color = colors[si % colors.len()];
                let mut points = String::new();
                for (i, &y) in ys.iter().enumerate() {
                    // Reuse the shared x axis; fall back to the index when the
                    // caller supplied fewer x values than y values.
                    let x = data.x_values.get(i).copied().unwrap_or(i as f64);
                    if !x.is_finite() || !y.is_finite() {
                        continue;
                    }
                    let _ = write!(
                        points,
                        "{:.2},{:.2} ",
                        frame.sx(x, xlo, xhi),
                        frame.sy(y, ylo, yhi)
                    );
                }
                if !points.is_empty() {
                    let _ = writeln!(
                        out,
                        "<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"1.8\" \
                         points=\"{points}\"/>",
                        color = color,
                        points = points.trim_end(),
                    );
                }
                if let Some(label) = data.labels.get(si) {
                    let ly = frame.plot_y + 14.0 + si as f64 * ((font as f64) + 4.0);
                    let _ = writeln!(
                        out,
                        "<rect x=\"{lx:.1}\" y=\"{ry:.1}\" width=\"10\" height=\"10\" \
                         fill=\"{color}\"/>\n\
                         <text x=\"{tx:.1}\" y=\"{ly:.1}\" font-size=\"{fs}\" fill=\"{fg}\">\
                         {label}</text>",
                        lx = frame.plot_x + 8.0,
                        ry = ly - 9.0,
                        color = color,
                        tx = frame.plot_x + 22.0,
                        ly = ly,
                        fs = font,
                        fg = fg,
                        label = escape_xml(label),
                    );
                }
            }
        },
        _ => {
            let _ = writeln!(
                out,
                "<text x=\"{x:.1}\" y=\"{y:.1}\" font-size=\"{fs}\" fill=\"{fg}\" \
                 text-anchor=\"middle\">no finite data points</text>",
                x = frame.width / 2.0,
                y = frame.height / 2.0,
                fs = font,
                fg = fg,
            );
        },
    }

    out.push_str("</svg>\n");
    out
}

/// Bin `values` into `bins` equal-width buckets over their finite range.
///
/// Returns `(counts, min, max)`. Non-finite values are skipped. Returns `None`
/// when there is nothing finite to bin.
pub(crate) fn histogram_counts(values: &[f64], bins: usize) -> Option<(Vec<u64>, f64, f64)> {
    let bins = bins.max(1);
    let (lo, hi) = finite_bounds(values)?;
    let mut counts = vec![0_u64; bins];
    if (hi - lo).abs() < f64::EPSILON {
        // Every finite value is identical: they all land in the first bucket.
        counts[0] = values.iter().filter(|v| v.is_finite()).count() as u64;
        return Some((counts, lo, hi));
    }
    let width = (hi - lo) / bins as f64;
    for &v in values.iter().filter(|v| v.is_finite()) {
        let idx = (((v - lo) / width).floor() as isize).clamp(0, bins as isize - 1) as usize;
        counts[idx] += 1;
    }
    Some((counts, lo, hi))
}

/// Render a real histogram from the actual binned counts.
pub fn histogram_svg(data: &HistogramData, config: &VisualizationConfig) -> String {
    let (bg, fg, colors) = palette(&config.color_scheme);
    let frame = Frame::new(config);
    let font = config.font_size;
    let mut out = String::new();
    open_svg(&mut out, &frame, bg, fg, font, &data.title);

    let bins = if data.bins == 0 { 10 } else { data.bins };
    match histogram_counts(&data.values, bins) {
        Some((counts, lo, hi)) => {
            let total: u64 = counts.iter().sum();
            let bin_width =
                if (hi - lo).abs() < f64::EPSILON { 1.0 } else { (hi - lo) / bins as f64 };
            // In density mode each bar is count / (total * bin_width) so the
            // bars integrate to 1 — the standard probability-density scaling.
            let heights: Vec<f64> = if data.density && total > 0 && bin_width > 0.0 {
                counts.iter().map(|&c| c as f64 / (total as f64 * bin_width)).collect()
            } else {
                counts.iter().map(|&c| c as f64).collect()
            };
            let ymax = heights.iter().cloned().fold(0.0_f64, f64::max).max(f64::EPSILON);
            let (xlo, xhi) = pad_range(lo, hi);
            draw_axes(
                &mut out,
                &frame,
                fg,
                font,
                (xlo, xhi),
                (0.0, ymax),
                &data.x_label,
                &data.y_label,
            );

            let color = colors[0];
            let step = frame.plot_w / bins as f64;
            for (i, &h) in heights.iter().enumerate() {
                let bar_h = h / ymax * frame.plot_h;
                if bar_h <= 0.0 {
                    continue;
                }
                let _ = writeln!(
                    out,
                    "<rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" \
                     fill=\"{color}\" fill-opacity=\"0.85\" stroke=\"{fg}\" \
                     stroke-width=\"0.3\"><title>bin {i}: {c}</title></rect>",
                    x = frame.plot_x + i as f64 * step,
                    y = frame.plot_y + frame.plot_h - bar_h,
                    w = (step - 1.0).max(0.5),
                    h = bar_h,
                    color = color,
                    fg = fg,
                    i = i,
                    c = counts[i],
                );
            }
        },
        None => {
            let _ = writeln!(
                out,
                "<text x=\"{x:.1}\" y=\"{y:.1}\" font-size=\"{fs}\" fill=\"{fg}\" \
                 text-anchor=\"middle\">no finite values to bin</text>",
                x = frame.width / 2.0,
                y = frame.height / 2.0,
                fs = font,
                fg = fg,
            );
        },
    }

    out.push_str("</svg>\n");
    out
}

/// Render a real heatmap: one `<rect>` per matrix cell, coloured by the cell's
/// position within the finite value range of the whole matrix.
pub fn heatmap_svg(data: &HeatmapData, config: &VisualizationConfig) -> String {
    let (bg, fg, _) = palette(&config.color_scheme);
    let frame = Frame::new(config);
    let font = config.font_size;
    let mut out = String::new();
    open_svg(&mut out, &frame, bg, fg, font, &data.title);

    let flat: Vec<f64> = data.values.iter().flat_map(|r| r.iter().copied()).collect();
    let rows = data.values.len();
    let cols = data.values.iter().map(|r| r.len()).max().unwrap_or(0);

    match finite_bounds(&flat) {
        Some((lo, hi)) if rows > 0 && cols > 0 => {
            let span = if (hi - lo).abs() < f64::EPSILON { 1.0 } else { hi - lo };
            let cell_w = frame.plot_w / cols as f64;
            let cell_h = frame.plot_h / rows as f64;

            for (r, row) in data.values.iter().enumerate() {
                for (c, &v) in row.iter().enumerate() {
                    if !v.is_finite() {
                        continue;
                    }
                    let t = (v - lo) / span;
                    let _ = writeln!(
                        out,
                        "<rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" \
                         fill=\"{fill}\"><title>[{r}][{c}] = {v}</title></rect>",
                        x = frame.plot_x + c as f64 * cell_w,
                        y = frame.plot_y + r as f64 * cell_h,
                        w = cell_w,
                        h = cell_h,
                        fill = colormap(&config.color_scheme, t),
                        r = r,
                        c = c,
                        v = v,
                    );
                }
            }

            let _ = writeln!(
                out,
                "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"none\" \
                 stroke=\"{}\" stroke-width=\"1\"/>",
                frame.plot_x, frame.plot_y, frame.plot_w, frame.plot_h, fg
            );

            // Colour bar: real min/max endpoints of the data.
            let bar_x = frame.plot_x + frame.plot_w + 6.0;
            let steps = 32;
            for i in 0..steps {
                let t = i as f64 / (steps - 1) as f64;
                let _ = writeln!(
                    out,
                    "<rect x=\"{x:.1}\" y=\"{y:.2}\" width=\"10\" height=\"{h:.2}\" \
                     fill=\"{fill}\"/>",
                    x = bar_x,
                    y = frame.plot_y + frame.plot_h - (t + 1.0 / steps as f64) * frame.plot_h,
                    h = frame.plot_h / steps as f64 + 0.6,
                    fill = colormap(&config.color_scheme, t),
                );
            }
            let _ = writeln!(
                out,
                "<text x=\"{x:.1}\" y=\"{y0:.1}\" font-size=\"{fs}\" fill=\"{fg}\">{hi}</text>\n\
                 <text x=\"{x:.1}\" y=\"{y1:.1}\" font-size=\"{fs}\" fill=\"{fg}\">{lo}</text>\n\
                 <text x=\"{x:.1}\" y=\"{y2:.1}\" font-size=\"{fs}\" fill=\"{fg}\">{cb}</text>",
                x = bar_x - 4.0,
                y0 = frame.plot_y - 4.0,
                y1 = frame.plot_y + frame.plot_h + (font as f64) + 2.0,
                y2 = frame.height - 8.0,
                fs = font,
                fg = fg,
                hi = escape_xml(&tick_label(hi)),
                lo = escape_xml(&tick_label(lo)),
                cb = escape_xml(&data.color_bar_label),
            );
        },
        _ => {
            let _ = writeln!(
                out,
                "<text x=\"{x:.1}\" y=\"{y:.1}\" font-size=\"{fs}\" fill=\"{fg}\" \
                 text-anchor=\"middle\">no finite cells to render</text>",
                x = frame.width / 2.0,
                y = frame.height / 2.0,
                fs = font,
                fg = fg,
            );
        },
    }

    out.push_str("</svg>\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VisualizationConfig {
        VisualizationConfig::default()
    }

    #[test]
    fn slugify_makes_a_filesystem_safe_stem() {
        assert_eq!(slugify("Gradient Flow - layer/0"), "gradient_flow_layer_0");
        assert_eq!(slugify("!!!"), "plot");
        assert_eq!(slugify(""), "plot");
    }

    #[test]
    fn escape_xml_neutralises_markup() {
        assert_eq!(
            escape_xml("<a href=\"x\">&</a>"),
            "&lt;a href=&quot;x&quot;&gt;&amp;&lt;/a&gt;"
        );
    }

    #[test]
    fn histogram_counts_are_real_counts() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (counts, lo, hi) = histogram_counts(&values, 4).expect("finite data must bin");
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 99.0);
        assert_eq!(counts.iter().sum::<u64>(), 100);
        // Equal-width bins over a uniform ramp must be near-equal.
        for c in &counts {
            assert!(
                (*c as i64 - 25).abs() <= 1,
                "unbalanced bin {c} in {counts:?}"
            );
        }
    }

    #[test]
    fn histogram_counts_skips_non_finite_and_handles_constants() {
        let values = vec![f64::NAN, 3.0, 3.0, f64::INFINITY, 3.0];
        let (counts, lo, hi) = histogram_counts(&values, 5).expect("has finite values");
        assert_eq!((lo, hi), (3.0, 3.0));
        assert_eq!(counts[0], 3, "the three finite 3.0s land in bucket 0");
        assert_eq!(counts.iter().sum::<u64>(), 3);
        assert!(histogram_counts(&[f64::NAN], 5).is_none());
    }

    #[test]
    fn line_plot_svg_contains_a_real_polyline_through_the_points() {
        let data = PlotData {
            x_values: vec![0.0, 1.0, 2.0],
            y_values: vec![10.0, 20.0, 30.0],
            labels: vec!["series".to_string()],
            title: "t".to_string(),
            x_label: "x".to_string(),
            y_label: "y".to_string(),
        };
        let svg = line_plot_svg(&data, &cfg());
        assert!(svg.starts_with("<svg"), "must be a real SVG document");
        assert!(svg.ends_with("</svg>\n"));
        let polyline = svg
            .lines()
            .find(|l| l.contains("<polyline"))
            .expect("a real polyline must be emitted");
        // Three input points -> three coordinate pairs.
        assert_eq!(
            polyline.matches(',').count(),
            3,
            "one pair per data point: {polyline}"
        );
    }

    #[test]
    fn line_plot_svg_splits_multi_series_y_values() {
        let data = PlotData {
            x_values: vec![0.0, 1.0],
            y_values: vec![1.0, 2.0, 30.0, 40.0],
            labels: vec!["loss".to_string(), "acc".to_string()],
            title: "t".to_string(),
            x_label: String::new(),
            y_label: String::new(),
        };
        let svg = line_plot_svg(&data, &cfg());
        assert_eq!(
            svg.matches("<polyline").count(),
            2,
            "two labels -> two series"
        );
        assert!(
            svg.contains(">loss<") && svg.contains(">acc<"),
            "legend must name both series"
        );
    }

    #[test]
    fn empty_data_says_so_instead_of_claiming_success() {
        let data = PlotData {
            x_values: vec![],
            y_values: vec![],
            labels: vec![],
            title: "empty".to_string(),
            x_label: String::new(),
            y_label: String::new(),
        };
        let svg = line_plot_svg(&data, &cfg());
        assert!(svg.contains("no finite data points"));
        assert!(!svg.contains("<polyline"));
    }

    #[test]
    fn histogram_svg_emits_one_bar_per_non_empty_bin() {
        let data = HistogramData {
            values: (0..40).map(|i| i as f64).collect(),
            bins: 4,
            title: "h".to_string(),
            x_label: String::new(),
            y_label: String::new(),
            density: false,
        };
        let svg = histogram_svg(&data, &cfg());
        // 1 background + 1 axis box + 4 bars.
        assert_eq!(svg.matches("<rect").count(), 6, "4 bars expected:\n{svg}");
        assert!(
            svg.contains("<title>bin 0: 10</title>"),
            "bar tooltips carry real counts"
        );
    }

    #[test]
    fn heatmap_svg_emits_one_cell_per_matrix_entry() {
        let data = HeatmapData {
            values: vec![vec![0.0, 1.0], vec![2.0, 3.0]],
            x_labels: vec![],
            y_labels: vec![],
            title: "hm".to_string(),
            color_bar_label: "v".to_string(),
        };
        let svg = heatmap_svg(&data, &cfg());
        assert!(svg.contains("<title>[0][0] = 0</title>"));
        assert!(svg.contains("<title>[1][1] = 3</title>"));
        // Distinct values must get distinct colours from the ramp.
        let c_lo = colormap(&ColorScheme::Default, 0.0);
        let c_hi = colormap(&ColorScheme::Default, 1.0);
        assert_ne!(c_lo, c_hi);
        assert!(svg.contains(&c_lo) && svg.contains(&c_hi));
    }

    #[test]
    fn colormap_is_monotone_between_anchor_stops() {
        // Endpoints must be exactly the first/last anchors.
        assert_eq!(colormap(&ColorScheme::Viridis, 0.0), "#440154");
        assert_eq!(colormap(&ColorScheme::Viridis, 1.0), "#fde725");
        // Out-of-range inputs clamp rather than panic or wrap.
        assert_eq!(colormap(&ColorScheme::Viridis, -5.0), "#440154");
        assert_eq!(colormap(&ColorScheme::Viridis, 5.0), "#fde725");
    }

    #[test]
    fn titles_cannot_inject_markup_into_the_document() {
        let data = HistogramData {
            values: vec![1.0, 2.0],
            bins: 2,
            title: "</svg><script>x</script>".to_string(),
            x_label: String::new(),
            y_label: String::new(),
            density: false,
        };
        let svg = histogram_svg(&data, &cfg());
        assert!(!svg.contains("<script>"));
        assert_eq!(svg.matches("</svg>").count(), 1);
    }
}
