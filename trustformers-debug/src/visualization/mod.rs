//! Visualization module for TrustformeRS debugging tools
//!
//! This module has been refactored into focused submodules to comply with the
//! 2000-line policy. The original visualization.rs (2843 lines) has been split into:
//!
//! - `types` - Basic visualization types, enums, and data structures
//! - Additional modules to be created as needed for terminal, video, etc.

pub mod modern_plotting;
pub mod svg_render;
pub mod types;

// Re-export main types for backward compatibility
pub use modern_plotting::*;
pub use types::*;

use anyhow::{anyhow, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A plot's source data, kept so exports and dashboards work off the real
/// numbers rather than a re-derived guess.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PlotSource {
    /// A 2-D line plot.
    Line(PlotData),
    /// A binned histogram.
    Histogram(HistogramData),
    /// A 2-D matrix heatmap.
    Heatmap(HeatmapData),
}

impl PlotSource {
    /// The plot's title, which doubles as its registry key.
    pub fn title(&self) -> &str {
        match self {
            PlotSource::Line(d) => &d.title,
            PlotSource::Histogram(d) => &d.title,
            PlotSource::Heatmap(d) => &d.title,
        }
    }
}

/// One rendered plot: its source data, the document that was actually written,
/// and where it was written to.
#[derive(Debug, Clone)]
pub struct RenderedPlot {
    /// Registry key (the plot title).
    pub name: String,
    /// File the rendered document was written to.
    pub path: PathBuf,
    /// The rendered document text (SVG, HTML or JSON depending on the config).
    pub document: String,
    /// The data the document was rendered from.
    pub source: PlotSource,
}

/// Debug visualizer that renders real plot documents to disk.
///
/// Every `create_*` / `plot_*` method renders the supplied data into a real
/// document (a genuine SVG/HTML/JSON file with axes, points and colour-mapped
/// cells derived from the caller's numbers), writes it under
/// [`VisualizationConfig::output_directory`] and returns that file's path.
///
/// Formats this crate cannot encode without an optional raster backend
/// (`ImageFormat::PNG`, `PDF`, `LaTeX`, `MP4`, `GIF`, `WebM`) are rejected with
/// a structured error naming the missing encoder — they are never written out
/// as SVG bytes under a misleading extension.
#[derive(Debug)]
pub struct DebugVisualizer {
    config: VisualizationConfig,
    /// Plots rendered so far, in creation order, keyed by title.
    plots: IndexMap<String, RenderedPlot>,
}

impl DebugVisualizer {
    pub fn new(config: VisualizationConfig) -> Self {
        Self {
            config,
            plots: IndexMap::new(),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(VisualizationConfig::default())
    }

    /// The configuration this visualizer renders with.
    pub fn config(&self) -> &VisualizationConfig {
        &self.config
    }

    /// File extension for the configured output format, or a structured error
    /// naming why this crate cannot produce that format.
    fn extension_for(format: &ImageFormat) -> Result<&'static str> {
        match format {
            ImageFormat::SVG => Ok("svg"),
            ImageFormat::HTML => Ok("html"),
            ImageFormat::JSON => Ok("json"),
            ImageFormat::PNG => Err(anyhow!(
                "ImageFormat::PNG cannot be encoded: trustformers-debug ships no raster \
                 encoder in its default (Pure-Rust) feature set. Configure \
                 VisualizationConfig::image_format = ImageFormat::SVG, or use the \
                 `visual`/`image` cargo features for the plotters raster backends."
            )),
            ImageFormat::PDF => Err(anyhow!(
                "ImageFormat::PDF cannot be encoded: no PDF writer is linked into \
                 trustformers-debug. Configure ImageFormat::SVG instead."
            )),
            ImageFormat::LaTeX => Err(anyhow!(
                "ImageFormat::LaTeX is not implemented for plots: no TikZ/PGFPlots emitter \
                 exists in trustformers-debug. Configure ImageFormat::SVG instead."
            )),
            ImageFormat::MP4 | ImageFormat::WebM => Err(anyhow!(
                "ImageFormat::{format:?} is a video container and there is no muxer or \
                 frame encoder in trustformers-debug (the `video` cargo feature was removed \
                 because nothing implemented it). Configure ImageFormat::SVG instead."
            )),
            ImageFormat::GIF => Err(anyhow!(
                "ImageFormat::GIF cannot be encoded from a static plot: the animated-GIF \
                 path lives behind the optional `gif` cargo feature and takes a frame \
                 sequence, not a single plot. Configure ImageFormat::SVG instead."
            )),
        }
    }

    /// Render `source` into the configured format and write it to disk.
    ///
    /// Returns the path of the file that was actually written.
    fn render_and_store(&mut self, source: PlotSource) -> Result<String> {
        let ext = Self::extension_for(&self.config.image_format)?;

        let svg = match &source {
            PlotSource::Line(d) => svg_render::line_plot_svg(d, &self.config),
            PlotSource::Histogram(d) => svg_render::histogram_svg(d, &self.config),
            PlotSource::Heatmap(d) => svg_render::heatmap_svg(d, &self.config),
        };

        let title = source.title().to_string();
        let document = match self.config.image_format {
            ImageFormat::SVG => svg,
            ImageFormat::HTML => format!(
                "<!doctype html>\n<html><head><meta charset=\"utf-8\">\
                 <title>{}</title></head><body>\n{}</body></html>\n",
                svg_render::escape_xml(&title),
                svg
            ),
            // JSON exports the source data itself, which is the only honest
            // JSON representation of a plot.
            ImageFormat::JSON => serde_json::to_string_pretty(&source)?,
            // Every remaining variant already returned an error from
            // `extension_for` above.
            _ => unreachable!("extension_for rejects every non-text format"),
        };

        let dir = Path::new(&self.config.output_directory);
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("{}.{}", svg_render::slugify(&title), ext));
        std::fs::write(&path, &document)?;

        let rendered = RenderedPlot {
            name: title.clone(),
            path: path.clone(),
            document,
            source,
        };
        self.plots.insert(title, rendered);
        Ok(path.to_string_lossy().to_string())
    }

    /// Render a line plot and return the path of the file written.
    pub fn create_line_plot(&mut self, data: &PlotData) -> Result<String> {
        self.render_and_store(PlotSource::Line(data.clone()))
    }

    /// Render a heatmap and return the path of the file written.
    pub fn create_heatmap(&mut self, data: &HeatmapData) -> Result<String> {
        self.render_and_store(PlotSource::Heatmap(data.clone()))
    }

    /// Render a histogram and return the path of the file written.
    pub fn create_histogram(&mut self, data: &HistogramData) -> Result<String> {
        self.render_and_store(PlotSource::Histogram(data.clone()))
    }

    /// Plot tensor distribution
    pub fn plot_tensor_distribution(
        &mut self,
        name: &str,
        values: &[f64],
        bins: usize,
    ) -> Result<String> {
        let data = HistogramData {
            values: values.to_vec(),
            bins,
            title: format!("{} Distribution", name),
            x_label: "Value".to_string(),
            y_label: "Frequency".to_string(),
            density: false,
        };
        self.create_histogram(&data)
    }

    /// Plot training metrics
    pub fn plot_training_metrics(
        &mut self,
        steps: &[f64],
        losses: &[f64],
        accuracies: Option<&[f64]>,
    ) -> Result<String> {
        let mut plot_data = PlotData {
            x_values: steps.to_vec(),
            y_values: losses.to_vec(),
            labels: vec!["Loss".to_string()],
            title: "Training Metrics".to_string(),
            x_label: "Steps".to_string(),
            y_label: "Value".to_string(),
        };

        if let Some(acc) = accuracies {
            plot_data.y_values.extend_from_slice(acc);
            plot_data.labels.push("Accuracy".to_string());
        }

        self.create_line_plot(&plot_data)
    }

    /// Plot gradient flow
    pub fn plot_gradient_flow(
        &mut self,
        layer_name: &str,
        steps: &[f64],
        gradient_norms: &[f64],
    ) -> Result<String> {
        let data = PlotData {
            x_values: steps.to_vec(),
            y_values: gradient_norms.to_vec(),
            labels: vec![format!("{} Gradient Flow", layer_name)],
            title: format!("Gradient Flow - {}", layer_name),
            x_label: "Steps".to_string(),
            y_label: "Gradient Norm".to_string(),
        };
        self.create_line_plot(&data)
    }

    /// Plot tensor heatmap
    pub fn plot_tensor_heatmap(&mut self, name: &str, values: &[Vec<f64>]) -> Result<String> {
        let data = HeatmapData {
            values: values.to_vec(),
            x_labels: (0..values.first().map_or(0, |row| row.len()))
                .map(|i| i.to_string())
                .collect(),
            y_labels: (0..values.len()).map(|i| i.to_string()).collect(),
            title: format!("{} Heatmap", name),
            color_bar_label: "Value".to_string(),
        };
        self.create_heatmap(&data)
    }

    /// Plot activation patterns
    pub fn plot_activation_patterns(
        &mut self,
        layer_name: &str,
        inputs: &[f64],
        outputs: &[f64],
    ) -> Result<String> {
        let data = PlotData {
            x_values: inputs.to_vec(),
            y_values: outputs.to_vec(),
            labels: vec![format!("{} Activation", layer_name)],
            title: format!("Activation Pattern - {}", layer_name),
            x_label: "Input".to_string(),
            y_label: "Output".to_string(),
        };
        self.create_line_plot(&data)
    }

    /// Names of the plots this visualizer has actually rendered, in creation
    /// order.
    ///
    /// Empty until something has been plotted — it is not a catalogue of what
    /// the visualizer *could* draw.
    pub fn get_plot_names(&self) -> Vec<String> {
        self.plots.keys().cloned().collect()
    }

    /// Look up a rendered plot by title.
    pub fn get_plot(&self, name: &str) -> Option<&RenderedPlot> {
        self.plots.get(name)
    }

    /// Build a dashboard page embedding the named plots.
    ///
    /// SVG/HTML renderings are inlined verbatim; a JSON rendering is linked by
    /// path (there is nothing to inline). Unknown names are rejected with a
    /// structured error listing what has actually been rendered, rather than
    /// emitting an empty card that looks like a plot.
    pub fn create_dashboard(&mut self, plot_names: &[String]) -> Result<String> {
        let unknown: Vec<&str> = plot_names
            .iter()
            .map(String::as_str)
            .filter(|n| !self.plots.contains_key(*n))
            .collect();
        if !unknown.is_empty() {
            return Err(anyhow!(
                "cannot build a dashboard for plots that were never rendered: {:?}. \
                 Rendered plots are: {:?}",
                unknown,
                self.get_plot_names()
            ));
        }

        let dashboard_path = Path::new(&self.config.output_directory).join("dashboard.html");
        std::fs::create_dir_all(&self.config.output_directory)?;

        let mut html = String::from(
            "<!doctype html>\n<html><head><meta charset=\"utf-8\">\
             <title>Debug Dashboard</title></head><body>\n",
        );
        html.push_str("<h1>TrustformeRS Debug Dashboard</h1>\n");

        for plot_name in plot_names {
            let plot = self.plots.get(plot_name).ok_or_else(|| {
                anyhow!("plot {plot_name:?} disappeared from the registry mid-render")
            })?;
            html.push_str(&format!(
                "<section><h2>{}</h2>\n",
                svg_render::escape_xml(plot_name)
            ));
            match self.config.image_format {
                ImageFormat::SVG => html.push_str(&plot.document),
                ImageFormat::HTML => {
                    // Inline just the <svg> element, not a nested document.
                    match (plot.document.find("<svg"), plot.document.rfind("</svg>")) {
                        (Some(a), Some(b)) => html.push_str(&plot.document[a..b + 6]),
                        _ => html.push_str(&format!(
                            "<p><a href=\"{}\">{}</a></p>",
                            svg_render::escape_xml(&plot.path.to_string_lossy()),
                            svg_render::escape_xml(&plot.path.to_string_lossy())
                        )),
                    }
                },
                _ => html.push_str(&format!(
                    "<p><a href=\"{}\">{}</a></p>",
                    svg_render::escape_xml(&plot.path.to_string_lossy()),
                    svg_render::escape_xml(&plot.path.to_string_lossy())
                )),
            }
            html.push_str("\n</section>\n");
        }

        html.push_str("</body></html>\n");
        std::fs::write(&dashboard_path, html)?;

        Ok(dashboard_path.to_string_lossy().to_string())
    }

    /// Export a rendered plot's **source data** as JSON to `export_path`.
    ///
    /// Errors when `plot_name` has not been rendered — the previous version
    /// wrote the literal string `"Plot data for: <name>"`, which contained no
    /// plot data at all and succeeded for names that never existed.
    pub fn export_plot_data(&self, plot_name: &str, export_path: &Path) -> Result<()> {
        let plot = self.plots.get(plot_name).ok_or_else(|| {
            anyhow!(
                "no plot named {plot_name:?} has been rendered; rendered plots are: {:?}",
                self.get_plot_names()
            )
        })?;
        if let Some(parent) = export_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(export_path, serde_json::to_string_pretty(&plot.source)?)?;
        Ok(())
    }

    /// Write the most recently rendered plot document to
    /// `output_directory/filename`.
    ///
    /// Errors when nothing has been rendered yet, instead of writing the
    /// literal placeholder text the previous implementation emitted.
    pub fn save_to_file(&self, filename: &str) -> Result<()> {
        let (_, plot) = self.plots.last().ok_or_else(|| {
            anyhow!(
                "save_to_file({filename:?}): nothing has been rendered yet, so there is no \
                 visualization to save"
            )
        })?;
        std::fs::create_dir_all(&self.config.output_directory)?;
        let output_path = Path::new(&self.config.output_directory).join(filename);
        std::fs::write(output_path, &plot.document)?;
        Ok(())
    }
}

/// Simple terminal-based visualizer
pub struct TerminalVisualizer;

impl TerminalVisualizer {
    pub fn new() -> Self {
        Self
    }

    /// Display simple text-based histogram in terminal
    pub fn display_histogram(&self, data: &HistogramData) -> Result<()> {
        println!("Terminal Histogram: {}", data.title);
        println!("Data points: {}", data.values.len());
        if data.values.is_empty() {
            return Ok(());
        }
        let bins = if data.bins == 0 { 10 } else { data.bins };
        let rendered = self.ascii_histogram(&data.values, bins);
        if !data.x_label.is_empty() || !data.y_label.is_empty() {
            println!("{} vs {}", data.y_label, data.x_label);
        }
        print!("{}", rendered);
        Ok(())
    }

    /// Display simple text-based statistics
    pub fn display_statistics(&self, label: &str, values: &[f64]) -> Result<()> {
        if values.is_empty() {
            println!("{}: No data", label);
            return Ok(());
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        println!(
            "{}: mean={:.3}, min={:.3}, max={:.3}",
            label, mean, min, max
        );
        Ok(())
    }

    /// ASCII histogram display
    pub fn ascii_histogram(&self, values: &[f64], bins: usize) -> String {
        if values.is_empty() {
            return "No data for histogram".to_string();
        }

        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if (max_val - min_val).abs() < f64::EPSILON {
            return format!("All values are {:.3}", min_val);
        }

        let mut histogram = vec![0; bins];
        let bin_width = (max_val - min_val) / bins as f64;

        for &value in values {
            let bin_index = ((value - min_val) / bin_width).floor() as usize;
            let bin_index = bin_index.min(bins - 1);
            histogram[bin_index] += 1;
        }

        let max_count = histogram.iter().max().unwrap_or(&0);
        let scale = if *max_count > 0 { 40.0 / *max_count as f64 } else { 1.0 };

        let mut result = String::new();
        for (i, &count) in histogram.iter().enumerate() {
            let bin_start = min_val + i as f64 * bin_width;
            let bin_end = bin_start + bin_width;
            let bar_length = (count as f64 * scale) as usize;
            let bar = "█".repeat(bar_length);
            result.push_str(&format!(
                "[{:.2}-{:.2}): {} ({})\n",
                bin_start, bin_end, bar, count
            ));
        }

        result
    }

    /// ASCII line plot display
    pub fn ascii_line_plot(&self, x_values: &[f64], y_values: &[f64], title: &str) -> String {
        if x_values.is_empty() || y_values.is_empty() || x_values.len() != y_values.len() {
            return "Invalid data for line plot".to_string();
        }

        let min_y = y_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_y = y_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let mut result = format!("{}\n", title);
        result.push_str("═".repeat(title.len()).as_str());
        result.push('\n');

        if (max_y - min_y).abs() < f64::EPSILON {
            result.push_str(&format!("Constant value: {:.3}\n", min_y));
            return result;
        }

        let height = 20;
        let width = x_values.len().min(80);

        // Sample data if too many points
        let step = if x_values.len() > width { x_values.len() / width } else { 1 };

        for row in (0..height).rev() {
            let y_threshold = min_y + (max_y - min_y) * row as f64 / (height - 1) as f64;
            let mut line = String::new();

            for i in (0..x_values.len()).step_by(step).take(width) {
                if y_values[i] >= y_threshold {
                    line.push('*');
                } else {
                    line.push(' ');
                }
            }
            result.push_str(&format!("{:8.2} |{}\n", y_threshold, line));
        }

        result.push_str(&format!("{:8} +{}\n", "", "─".repeat(width)));
        result
    }
}

impl Default for TerminalVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique scratch directory under the platform temp dir (never a hardcoded path).
    fn scratch(tag: &str) -> String {
        let dir = std::env::temp_dir().join(format!(
            "tfdbg_viz_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        dir.to_string_lossy().to_string()
    }

    fn viz(tag: &str) -> DebugVisualizer {
        DebugVisualizer::new(VisualizationConfig {
            output_directory: scratch(tag),
            ..Default::default()
        })
    }

    fn sample_line() -> PlotData {
        PlotData {
            x_values: vec![0.0, 1.0, 2.0, 3.0],
            y_values: vec![1.0, 4.0, 9.0, 16.0],
            labels: vec!["squares".to_string()],
            title: "Squares".to_string(),
            x_label: "n".to_string(),
            y_label: "n^2".to_string(),
        }
    }

    #[test]
    fn create_line_plot_writes_a_real_svg_document() {
        let mut v = viz("line");
        let path = v.create_line_plot(&sample_line()).expect("render must succeed");
        let written = std::fs::read_to_string(&path).expect("the returned path must exist");
        // The old implementation returned "Line plot 'Squares' created successfully"
        // and wrote nothing at all.
        assert!(!path.contains("created successfully"));
        assert!(
            written.starts_with("<svg"),
            "must be a real SVG: {written:.80}"
        );
        assert!(
            written.contains("<polyline"),
            "must contain the real data polyline"
        );
        assert!(
            written.contains(">squares<"),
            "must carry the real series label"
        );
        let _ = std::fs::remove_dir_all(v.config().output_directory.clone());
    }

    #[test]
    fn get_plot_names_reports_only_what_was_really_rendered() {
        let mut v = viz("names");
        // Previously this returned four invented names "for demonstration"
        // before anything had been plotted.
        assert!(v.get_plot_names().is_empty(), "nothing rendered yet");
        v.create_line_plot(&sample_line()).expect("render");
        v.plot_tensor_distribution("weights", &[1.0, 2.0, 3.0, 4.0], 4).expect("render");
        assert_eq!(
            v.get_plot_names(),
            vec!["Squares".to_string(), "weights Distribution".to_string()]
        );
        let _ = std::fs::remove_dir_all(v.config().output_directory.clone());
    }

    #[test]
    fn export_plot_data_exports_the_real_source_data() {
        let mut v = viz("export");
        v.create_line_plot(&sample_line()).expect("render");
        let out = std::path::PathBuf::from(v.config().output_directory.clone()).join("d.json");
        v.export_plot_data("Squares", &out).expect("export must succeed");
        let text = std::fs::read_to_string(&out).expect("export file");
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(parsed["kind"], "line");
        assert_eq!(
            parsed["y_values"][3], 16.0,
            "the real y values must round-trip"
        );
        let _ = std::fs::remove_dir_all(v.config().output_directory.clone());
    }

    #[test]
    fn export_plot_data_refuses_a_plot_that_was_never_rendered() {
        let v = viz("export_missing");
        let out = std::env::temp_dir().join("tfdbg_never_written.json");
        let err = v.export_plot_data("nope", &out).expect_err("must not fabricate an export");
        let msg = err.to_string();
        assert!(
            msg.contains("no plot named"),
            "structured error names the problem: {msg}"
        );
        assert!(
            !out.exists(),
            "must not write a file for a plot that does not exist"
        );
    }

    #[test]
    fn save_to_file_refuses_before_anything_is_rendered() {
        let v = viz("save_empty");
        let err = v.save_to_file("x.svg").expect_err("must not write placeholder content");
        assert!(err.to_string().contains("nothing has been rendered"));
    }

    #[test]
    fn save_to_file_writes_the_real_rendered_document() {
        let mut v = viz("save");
        v.create_line_plot(&sample_line()).expect("render");
        v.save_to_file("copy.svg").expect("save must succeed");
        let text = std::fs::read_to_string(
            std::path::PathBuf::from(v.config().output_directory.clone()).join("copy.svg"),
        )
        .expect("saved file");
        assert!(
            text.contains("<polyline"),
            "the saved bytes are the real rendering"
        );
        assert!(!text.contains("placeholder visualization content"));
        let _ = std::fs::remove_dir_all(v.config().output_directory.clone());
    }

    #[test]
    fn unencodable_formats_return_a_structured_error_not_mislabelled_bytes() {
        for (format, needle) in [
            (ImageFormat::PNG, "no raster encoder"),
            (ImageFormat::PDF, "no PDF writer"),
            (ImageFormat::MP4, "video container"),
            (ImageFormat::GIF, "animated-GIF"),
            (ImageFormat::LaTeX, "not implemented"),
        ] {
            let dir = scratch("fmt");
            let mut v = DebugVisualizer::new(VisualizationConfig {
                output_directory: dir.clone(),
                image_format: format.clone(),
                ..Default::default()
            });
            match v.create_line_plot(&sample_line()) {
                Ok(p) => panic!("{format:?} must be refused, but it wrote {p}"),
                Err(e) => assert!(
                    e.to_string().contains(needle),
                    "{format:?} error must name the missing encoder ({needle}): {e}"
                ),
            }
            assert!(
                !std::path::Path::new(&dir).exists(),
                "{format:?}: nothing may be written for a refused format"
            );
        }
    }

    #[test]
    fn png_is_refused_and_writes_nothing() {
        let dir = scratch("png");
        let mut v = DebugVisualizer::new(VisualizationConfig {
            output_directory: dir.clone(),
            image_format: ImageFormat::PNG,
            ..Default::default()
        });
        let err = v.create_line_plot(&sample_line()).expect_err("PNG must be refused");
        assert!(err.to_string().contains("no raster encoder"), "{err}");
        assert!(
            !std::path::Path::new(&dir).exists(),
            "nothing may be written for a refused format"
        );
    }

    #[test]
    fn create_dashboard_embeds_real_plots_and_refuses_unknown_names() {
        let mut v = viz("dash");
        v.create_line_plot(&sample_line()).expect("render");
        let err = v
            .create_dashboard(&["Squares".to_string(), "ghost".to_string()])
            .expect_err("unknown plot names must be refused");
        assert!(err.to_string().contains("never rendered"), "{err}");

        let path = v.create_dashboard(&v.get_plot_names()).expect("dashboard must build");
        let html = std::fs::read_to_string(&path).expect("dashboard file");
        assert!(
            html.contains("<polyline"),
            "the dashboard inlines the real SVG"
        );
        assert!(!html.contains("<p>Plot: "), "no fake plot cards");
        let _ = std::fs::remove_dir_all(v.config().output_directory.clone());
    }

    #[test]
    fn json_format_writes_the_real_source_data() {
        let dir = scratch("json");
        let mut v = DebugVisualizer::new(VisualizationConfig {
            output_directory: dir.clone(),
            image_format: ImageFormat::JSON,
            ..Default::default()
        });
        let path = v
            .create_histogram(&HistogramData {
                values: vec![1.0, 2.0, 3.0],
                bins: 3,
                title: "H".to_string(),
                x_label: String::new(),
                y_label: String::new(),
                density: false,
            })
            .expect("json render");
        assert!(
            path.ends_with(".json"),
            "extension must match the real content: {path}"
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("file")).expect("json");
        assert_eq!(parsed["kind"], "histogram");
        assert_eq!(parsed["values"][2], 3.0);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_display_histogram_empty_returns_ok() {
        let viz = TerminalVisualizer::new();
        let data = HistogramData {
            values: vec![],
            bins: 10,
            title: "empty".to_string(),
            x_label: String::new(),
            y_label: String::new(),
            density: false,
        };
        assert!(viz.display_histogram(&data).is_ok());
    }

    #[test]
    fn test_display_histogram_with_values_returns_ok() {
        let viz = TerminalVisualizer::new();
        let data = HistogramData {
            values: (0..50).map(|i| i as f64).collect(),
            bins: 5,
            title: "ramp".to_string(),
            x_label: "value".to_string(),
            y_label: "count".to_string(),
            density: false,
        };
        assert!(viz.display_histogram(&data).is_ok());
    }

    #[test]
    fn test_display_histogram_zero_bins_falls_back_to_default() {
        let viz = TerminalVisualizer::new();
        let data = HistogramData {
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            bins: 0,
            title: "fallback".to_string(),
            x_label: String::new(),
            y_label: String::new(),
            density: false,
        };
        // Should not panic with zero bins.
        assert!(viz.display_histogram(&data).is_ok());
    }
}
