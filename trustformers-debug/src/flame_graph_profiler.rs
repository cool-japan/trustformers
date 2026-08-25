//! Advanced flame graph profiling implementation for TrustformeRS Debug
// reason: debug/profiling scaffolding — structs are constructed and their fields/methods
// are retained for the data model, serialization completeness, and future consumers that
// do not yet read every member. Consolidated from many item-level #[allow(dead_code)].
#![allow(dead_code)]

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Instant, SystemTime};

use crate::profiler::{ProfileEvent, Profiler};

/// Flame graph node representing a stack frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameGraphNode {
    pub name: String,
    pub value: u64,
    pub delta: Option<i64>, // For differential analysis
    pub children: HashMap<String, FlameGraphNode>,
    pub total_value: u64,
    pub self_value: u64,
    pub percentage: f64,
    pub color: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Escape the XML/HTML predefined entities so frame names cannot break out of
/// the generated markup.
fn html_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
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

/// Stack frame for flame graph construction
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StackFrame {
    pub function_name: String,
    pub module_name: Option<String>,
    pub file_name: Option<String>,
    pub line_number: Option<u32>,
    pub address: Option<u64>,
}

/// Sample data for flame graph
#[derive(Debug, Clone)]
pub struct FlameGraphSample {
    pub stack: Vec<StackFrame>,
    pub duration_ns: u64,
    pub timestamp: u64,
    pub thread_id: u64,
    pub cpu_id: Option<u32>,
    pub memory_usage: Option<usize>,
    pub gpu_kernel: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Configuration for flame graph generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameGraphConfig {
    pub sampling_rate: u32, // Samples per second
    pub min_width: f64,     // Minimum width for node visibility
    pub color_scheme: FlameGraphColorScheme,
    pub direction: FlameGraphDirection,
    pub title: String,
    pub subtitle: Option<String>,
    pub include_memory: bool,
    pub include_gpu: bool,
    pub differential_mode: bool,
    pub merge_similar_stacks: bool,
    pub filter_noise: bool,
    pub noise_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlameGraphColorScheme {
    Hot,          // Red-orange gradient
    Cool,         // Blue-purple gradient
    Java,         // Java-specific colors
    Memory,       // Memory-aware coloring
    Differential, // Differential analysis colors
    Random,       // Random but consistent colors
    Custom(HashMap<String, String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlameGraphDirection {
    TopDown,  // Traditional flame graph
    BottomUp, // Icicle graph
}

/// Export format for flame graphs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlameGraphExportFormat {
    SVG,
    InteractiveHTML,
    JSON,
    Speedscope,
    D3,
    Folded,
}

/// Advanced flame graph profiler
#[derive(Debug)]
pub struct FlameGraphProfiler {
    config: FlameGraphConfig,
    samples: Vec<FlameGraphSample>,
    sampling_timer: Option<Instant>,
    root_node: Option<FlameGraphNode>,
    baseline_samples: Option<Vec<FlameGraphSample>>, // For differential analysis
    metadata: HashMap<String, String>,
    current_cpu_usage: f64,
    current_memory_usage: usize,
    performance_counters: HashMap<String, u64>,
}

impl FlameGraphProfiler {
    /// Create a new flame graph profiler
    pub fn new(config: FlameGraphConfig) -> Self {
        Self {
            config,
            samples: Vec::new(),
            sampling_timer: None,
            root_node: None,
            baseline_samples: None,
            metadata: HashMap::new(),
            current_cpu_usage: 0.0,
            current_memory_usage: 0,
            performance_counters: HashMap::new(),
        }
    }

    /// Start profiling with sampling
    pub fn start_sampling(&mut self) -> Result<()> {
        tracing::info!(
            "Starting flame graph sampling at {} Hz",
            self.config.sampling_rate
        );
        self.sampling_timer = Some(Instant::now());
        self.samples.clear();
        self.root_node = None;

        // Initialize performance counters
        self.performance_counters.insert("samples_collected".to_string(), 0);
        self.performance_counters.insert("stack_depth_max".to_string(), 0);
        self.performance_counters.insert("unique_functions".to_string(), 0);

        Ok(())
    }

    /// Stop profiling and build flame graph
    pub fn stop_sampling(&mut self) -> Result<()> {
        tracing::info!(
            "Stopping flame graph sampling, collected {} samples",
            self.samples.len()
        );
        self.sampling_timer = None;
        self.build_flame_graph()?;
        Ok(())
    }

    /// Add a sample to the profiler
    pub fn add_sample(&mut self, sample: FlameGraphSample) {
        // Update performance counters
        if let Some(counter) = self.performance_counters.get_mut("samples_collected") {
            *counter += 1;
        }

        let stack_depth = sample.stack.len() as u64;
        if let Some(max_depth) = self.performance_counters.get_mut("stack_depth_max") {
            if stack_depth > *max_depth {
                *max_depth = stack_depth;
            }
        }

        self.samples.push(sample);
    }

    /// Add a sample from current stack trace
    pub fn sample_current_stack(&mut self, duration_ns: u64) -> Result<()> {
        let stack = self.capture_stack_trace()?;
        let sample = FlameGraphSample {
            stack,
            duration_ns,
            timestamp: SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_nanos() as u64,
            thread_id: self.get_current_thread_id(),
            cpu_id: self.get_current_cpu_id(),
            memory_usage: Some(self.current_memory_usage),
            gpu_kernel: None,
            metadata: HashMap::new(),
        };

        self.add_sample(sample);
        Ok(())
    }

    /// Add GPU kernel sample
    pub fn sample_gpu_kernel(&mut self, kernel_name: &str, duration_ns: u64) {
        let stack = vec![StackFrame {
            function_name: format!("GPU::{}", kernel_name),
            module_name: Some("GPU".to_string()),
            file_name: None,
            line_number: None,
            address: None,
        }];

        let sample = FlameGraphSample {
            stack,
            duration_ns,
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            thread_id: 0, // GPU operations on virtual thread
            cpu_id: None,
            memory_usage: None,
            gpu_kernel: Some(kernel_name.to_string()),
            metadata: [("type".to_string(), "gpu".to_string())].into_iter().collect(),
        };

        self.add_sample(sample);
    }

    /// Set baseline for differential analysis
    pub fn set_baseline(&mut self) {
        self.baseline_samples = Some(self.samples.clone());
        tracing::info!("Set baseline with {} samples", self.samples.len());
    }

    /// Build flame graph from collected samples
    pub fn build_flame_graph(&mut self) -> Result<()> {
        if self.samples.is_empty() {
            return Err(anyhow::anyhow!("No samples collected"));
        }

        let mut root = FlameGraphNode {
            name: "root".to_string(),
            value: 0,
            delta: None,
            children: HashMap::new(),
            total_value: 0,
            self_value: 0,
            percentage: 100.0,
            color: None,
            metadata: HashMap::new(),
        };

        // Merge samples into tree structure
        for sample in &self.samples {
            self.merge_sample_into_tree(&mut root, sample);
        }

        // Calculate totals and percentages
        self.calculate_node_metrics(&mut root);

        // Apply differential analysis if baseline exists
        if self.config.differential_mode && self.baseline_samples.is_some() {
            self.apply_differential_analysis(&mut root)?;
        }

        // Filter noise if enabled
        if self.config.filter_noise {
            self.filter_noise_nodes(&mut root);
        }

        // Update performance counters
        let unique_functions = self.count_unique_functions(&root);
        if let Some(counter) = self.performance_counters.get_mut("unique_functions") {
            *counter = unique_functions;
        }

        self.root_node = Some(root);
        tracing::info!(
            "Built flame graph with {} unique functions",
            unique_functions
        );
        Ok(())
    }

    /// Export flame graph to various formats
    pub async fn export(&self, format: FlameGraphExportFormat, output_path: &Path) -> Result<()> {
        let root = self
            .root_node
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Flame graph not built yet"))?;

        match format {
            FlameGraphExportFormat::SVG => self.export_svg(root, output_path).await,
            FlameGraphExportFormat::InteractiveHTML => {
                self.export_interactive_html(root, output_path).await
            },
            FlameGraphExportFormat::JSON => self.export_json(root, output_path).await,
            FlameGraphExportFormat::Speedscope => self.export_speedscope(root, output_path).await,
            FlameGraphExportFormat::D3 => self.export_d3(root, output_path).await,
            FlameGraphExportFormat::Folded => self.export_folded(output_path).await,
        }
    }

    /// Export as SVG flame graph
    async fn export_svg(&self, root: &FlameGraphNode, output_path: &Path) -> Result<()> {
        let mut svg_content = String::new();

        // SVG header
        svg_content.push_str(&format!(
            r##"<?xml version="1.0" encoding="UTF-8"?>
<svg width="1200" height="800" xmlns="http://www.w3.org/2000/svg">
<defs>
    <linearGradient id="background" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" style="stop-color:#eeeeee"/>
        <stop offset="100%" style="stop-color:#eeeeb0"/>
    </linearGradient>
</defs>
<rect width="100%" height="100%" fill="url(#background)"/>
<text x="600" y="24" text-anchor="middle" font-size="17" font-family="Verdana">{}</text>
<text x="600" y="44" text-anchor="middle" font-size="12" font-family="Verdana" fill="#999">
    {} samples, {} functions
</text>
"##,
            self.config.title,
            self.samples.len(),
            self.count_unique_functions(root)
        ));

        // Render flame graph rectangles
        self.render_svg_node(&mut svg_content, root, 0, 0, 1200, 0)?;

        svg_content.push_str("</svg>");

        tokio::fs::write(output_path, svg_content).await?;
        tracing::info!("Exported SVG flame graph to {:?}", output_path);
        Ok(())
    }

    /// Export the flame graph as a self-contained HTML page.
    ///
    /// The page carries a REAL inline SVG flame graph -- one rectangle per
    /// node, width proportional to that node's total value, `y` by stack depth,
    /// with a native `<title>` tooltip -- plus the full tree as embedded JSON.
    ///
    /// It used to advertise "Click to zoom, double-click to reset. Hover for
    /// details", render `Reset Zoom` / `Search` buttons wired to functions that
    /// were never defined, and load d3 from a CDN, while the only script in the
    /// page was a `console.log`. Nothing was drawn: `#flame-graph` stayed
    /// empty. The page is now honest about being a static rendering.
    async fn export_interactive_html(
        &self,
        root: &FlameGraphNode,
        output_path: &Path,
    ) -> Result<()> {
        let json_data = serde_json::to_string(root)?;
        let svg = Self::render_flame_svg(root);

        let html_content = format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 20px; }}
        .flame-graph {{ width: 100%; overflow-x: auto; border: 1px solid #ccc; }}
        .info {{ margin-top: 20px; font-size: 14px; color: #666; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="flame-graph">{svg}</div>
    <div class="info">
        <p>Samples: {samples} | Functions: {functions} | Total Time: {total_ms:.2}ms</p>
        <p>Static rendering: hover a frame for its name and timing. There is no
           zoom or search in this export; the complete tree is embedded below as
           JSON for tools that want it.</p>
    </div>
    <script type="application/json" id="flamegraph-data">{json_data}</script>
</body>
</html>"#,
            title = html_escape(&self.config.title),
            svg = svg,
            samples = self.samples.len(),
            functions = self.count_unique_functions(root),
            total_ms = root.total_value as f64 / 1_000_000.0,
            json_data = json_data,
        );

        tokio::fs::write(output_path, html_content).await?;
        tracing::info!("Exported HTML flame graph to {:?}", output_path);
        Ok(())
    }

    /// Height in pixels of one stack level in the rendered SVG.
    const FLAME_ROW_HEIGHT: f64 = 18.0;
    /// Total width in pixels of the rendered SVG.
    const FLAME_WIDTH: f64 = 1200.0;

    /// Render `root` as a real, dependency-free SVG flame graph.
    fn render_flame_svg(root: &FlameGraphNode) -> String {
        use std::fmt::Write as _;

        let depth = Self::tree_depth(root);
        let height = (depth as f64 + 1.0) * Self::FLAME_ROW_HEIGHT + 4.0;
        let mut out = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h:.0}\" \
             viewBox=\"0 0 {w} {h:.0}\" font-family=\"sans-serif\" font-size=\"11\">",
            w = Self::FLAME_WIDTH,
            h = height,
        );
        Self::render_flame_node(&mut out, root, 0.0, 0, Self::FLAME_WIDTH, root.total_value);
        let _ = write!(out, "</svg>");
        out
    }

    fn tree_depth(node: &FlameGraphNode) -> usize {
        1 + node.children.values().map(Self::tree_depth).max().unwrap_or(0)
    }

    fn render_flame_node(
        out: &mut String,
        node: &FlameGraphNode,
        x: f64,
        depth: usize,
        width: f64,
        root_total: u64,
    ) {
        use std::fmt::Write as _;

        if width < 0.05 {
            return;
        }
        let y = depth as f64 * Self::FLAME_ROW_HEIGHT;
        // Classic flame-graph warm palette, keyed by the frame name so the same
        // function keeps the same colour across renders.
        let hue = (Self::name_hash(&node.name) % 50) as u32;
        let percentage = if root_total > 0 {
            node.total_value as f64 / root_total as f64 * 100.0
        } else {
            0.0
        };
        let _ = write!(
            out,
            "<g><rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" \
             fill=\"hsl({hue},80%,55%)\" stroke=\"#fff\" stroke-width=\"0.5\"/>\
             <title>{name} — {ms:.3}ms ({percentage:.2}%)</title>",
            x = x,
            y = y,
            w = width,
            h = Self::FLAME_ROW_HEIGHT - 1.0,
            hue = hue,
            name = html_escape(&node.name),
            ms = node.total_value as f64 / 1_000_000.0,
            percentage = percentage,
        );
        // Only label frames wide enough to hold readable text.
        if width > 40.0 {
            let max_chars = (width / 6.5) as usize;
            let label: String = node.name.chars().take(max_chars.max(1)).collect();
            let _ = write!(
                out,
                "<text x=\"{tx:.2}\" y=\"{ty:.2}\" fill=\"#000\">{label}</text>",
                tx = x + 3.0,
                ty = y + Self::FLAME_ROW_HEIGHT - 6.0,
                label = html_escape(&label),
            );
        }
        let _ = write!(out, "</g>");

        // Children are laid out left to right in a stable (name-sorted) order,
        // each taking the share of the parent's width that its value really is.
        let mut children: Vec<&FlameGraphNode> = node.children.values().collect();
        children.sort_by(|a, b| a.name.cmp(&b.name));
        let child_total: u64 = children.iter().map(|c| c.total_value).sum();
        if child_total == 0 {
            return;
        }
        let mut cursor = x;
        for child in children {
            let child_width = width * (child.total_value as f64 / child_total as f64);
            Self::render_flame_node(out, child, cursor, depth + 1, child_width, root_total);
            cursor += child_width;
        }
    }

    /// Deterministic FNV-1a hash of a frame name, for stable colouring.
    fn name_hash(name: &str) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for byte in name.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        hash
    }

    /// Export as JSON
    async fn export_json(&self, root: &FlameGraphNode, output_path: &Path) -> Result<()> {
        let json_data = serde_json::to_string_pretty(root)?;
        tokio::fs::write(output_path, json_data).await?;
        tracing::info!("Exported JSON flame graph to {:?}", output_path);
        Ok(())
    }

    /// Export as Speedscope format
    async fn export_speedscope(&self, root: &FlameGraphNode, output_path: &Path) -> Result<()> {
        let speedscope_data = self.convert_to_speedscope_format(root)?;
        let json_data = serde_json::to_string_pretty(&speedscope_data)?;
        tokio::fs::write(output_path, json_data).await?;
        tracing::info!("Exported Speedscope format to {:?}", output_path);
        Ok(())
    }

    /// Export as D3.js compatible format
    async fn export_d3(&self, root: &FlameGraphNode, output_path: &Path) -> Result<()> {
        let d3_data = self.convert_to_d3_format(root)?;
        let json_data = serde_json::to_string_pretty(&d3_data)?;
        tokio::fs::write(output_path, json_data).await?;
        tracing::info!("Exported D3 format to {:?}", output_path);
        Ok(())
    }

    /// Export as folded stack format
    async fn export_folded(&self, output_path: &Path) -> Result<()> {
        let mut folded_content = String::new();

        for sample in &self.samples {
            let stack_str: Vec<String> =
                sample.stack.iter().map(|frame| frame.function_name.clone()).collect();
            folded_content.push_str(&format!("{} {}\n", stack_str.join(";"), sample.duration_ns));
        }

        tokio::fs::write(output_path, folded_content).await?;
        tracing::info!("Exported folded format to {:?}", output_path);
        Ok(())
    }

    /// Get flame graph analysis report
    pub fn get_analysis_report(&self) -> FlameGraphAnalysisReport {
        let root = self.root_node.as_ref();

        FlameGraphAnalysisReport {
            total_samples: self.samples.len(),
            total_duration_ns: self.samples.iter().map(|s| s.duration_ns).sum(),
            unique_functions: root.map(|r| self.count_unique_functions(r)).unwrap_or(0),
            max_stack_depth: self.performance_counters.get("stack_depth_max").copied().unwrap_or(0),
            hot_functions: self.get_hot_functions(10),
            memory_usage_stats: self.get_memory_usage_stats(),
            gpu_kernel_stats: self.get_gpu_kernel_stats(),
            differential_analysis: self.get_differential_analysis(),
            performance_insights: self.generate_performance_insights(),
        }
    }

    // Private helper methods

    /// Capture the REAL current call stack via [`std::backtrace::Backtrace`].
    ///
    /// `std` does not expose structured frames on stable, so the frames are
    /// parsed out of the backtrace's textual form: each frame is a
    /// `N: symbol` line optionally followed by an `at path:line[:col]` line.
    /// Frames belonging to this module's own capture machinery are dropped so
    /// the sample starts at the caller.
    ///
    /// Returns an error when the platform captured nothing (backtraces are
    /// disabled or unsupported) rather than inventing a frame -- the previous
    /// implementation returned one hardcoded frame,
    /// `captured_function @ profiler.rs:1800`, so every sample in every flame
    /// graph was the same fictional stack.
    fn capture_stack_trace(&self) -> Result<Vec<StackFrame>> {
        let backtrace = std::backtrace::Backtrace::force_capture();
        let frames = Self::parse_backtrace(&backtrace.to_string());
        if frames.is_empty() {
            anyhow::bail!(
                "no stack frames could be captured on this platform (std::backtrace status: \
                 {:?}); refusing to record a fabricated stack",
                backtrace.status()
            );
        }
        Ok(frames)
    }

    /// Parse the textual form of a [`std::backtrace::Backtrace`] into frames.
    ///
    /// Split out from [`Self::capture_stack_trace`] so the parsing is testable
    /// against a fixed input, independent of whatever the live stack happens
    /// to be.
    fn parse_backtrace(text: &str) -> Vec<StackFrame> {
        let mut frames: Vec<StackFrame> = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("at ") {
                // Location line: attach to the frame just pushed.
                if let Some(frame) = frames.last_mut() {
                    let mut parts = rest.rsplitn(3, ':');
                    // `path:line:col` or `path:line`.
                    let last = parts.next().unwrap_or_default();
                    let middle = parts.next();
                    let head = parts.next();
                    match (head, middle, last.parse::<u32>()) {
                        // path:line:col
                        (Some(path), Some(line_no), Ok(_col)) => {
                            frame.file_name = Some(path.to_string());
                            frame.line_number = line_no.parse::<u32>().ok();
                        },
                        // path:line
                        (None, Some(path), Ok(line_no)) => {
                            frame.file_name = Some(path.to_string());
                            frame.line_number = Some(line_no);
                        },
                        _ => frame.file_name = Some(rest.to_string()),
                    }
                }
                continue;
            }

            // Frame line: `<index>: <symbol>`.
            let Some((index, symbol)) = trimmed.split_once(": ") else {
                continue;
            };
            if index.parse::<u32>().is_err() {
                continue;
            }
            let symbol = symbol.trim();
            if symbol.is_empty() {
                continue;
            }
            // The module path is everything before the final `::segment`.
            let module_name = symbol.rfind("::").map(|idx| symbol[..idx].to_string());
            frames.push(StackFrame {
                function_name: symbol.to_string(),
                module_name,
                file_name: None,
                line_number: None,
                address: None,
            });
        }

        // Drop this crate's own capture frames so the sample begins at the
        // caller of `sample_current_stack`.
        let first_caller = frames
            .iter()
            .position(|f| {
                !f.function_name.contains("std::backtrace")
                    && !f.function_name.contains("Backtrace")
                    && !f.function_name.contains("capture_stack_trace")
                    && !f.function_name.contains("parse_backtrace")
            })
            .unwrap_or(0);
        frames.split_off(first_caller)
    }

    /// A real, stable per-thread identifier.
    ///
    /// `std::thread::ThreadId` is opaque on stable, so this hands out dense
    /// ids from a process-wide counter, one per thread, cached in thread-local
    /// storage. Two samples from the same thread always share an id and two
    /// different threads never do -- which is exactly what a flame graph needs,
    /// and what the previous hardcoded `1` could not provide.
    fn get_current_thread_id(&self) -> u64 {
        use std::cell::Cell;
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT_THREAD_ID: AtomicU64 = AtomicU64::new(1);
        thread_local! {
            static THREAD_ID: Cell<u64> = const { Cell::new(0) };
        }

        THREAD_ID.with(|slot| {
            let existing = slot.get();
            if existing != 0 {
                return existing;
            }
            let assigned = NEXT_THREAD_ID.fetch_add(1, Ordering::Relaxed);
            slot.set(assigned);
            assigned
        })
    }

    /// Which CPU the sample was taken on.
    ///
    /// Always `None`: reading the current CPU needs a platform-specific call
    /// (`sched_getcpu` on Linux, `GetCurrentProcessorNumber` on Windows, no
    /// stable equivalent on macOS) and this crate is Pure Rust with no FFI.
    /// It used to return `Some(0)`, i.e. "every sample ran on CPU 0".
    fn get_current_cpu_id(&self) -> Option<u32> {
        None
    }

    fn merge_sample_into_tree(&self, node: &mut FlameGraphNode, sample: &FlameGraphSample) {
        if sample.stack.is_empty() {
            node.value += sample.duration_ns;
            return;
        }

        let frame = &sample.stack[0];
        let child =
            node.children
                .entry(frame.function_name.clone())
                .or_insert_with(|| FlameGraphNode {
                    name: frame.function_name.clone(),
                    value: 0,
                    delta: None,
                    children: HashMap::new(),
                    total_value: 0,
                    self_value: 0,
                    percentage: 0.0,
                    color: None,
                    metadata: HashMap::new(),
                });

        if sample.stack.len() == 1 {
            child.value += sample.duration_ns;
        } else {
            let mut remaining_sample = sample.clone();
            remaining_sample.stack = sample.stack[1..].to_vec();
            self.merge_sample_into_tree(child, &remaining_sample);
        }
    }

    fn calculate_node_metrics(&self, node: &mut FlameGraphNode) {
        let mut total_children_value = 0;

        for child in node.children.values_mut() {
            self.calculate_node_metrics(child);
            total_children_value += child.total_value;
        }

        node.total_value = node.value + total_children_value;
        node.self_value = node.value;

        if node.total_value > 0 && node.name != "root" {
            // Get the total from root node for percentage calculation
            let total_for_percentage = if let Some(root) = &self.root_node {
                root.total_value
            } else {
                node.total_value // fallback
            };

            if total_for_percentage > 0 {
                node.percentage = (node.total_value as f64 / total_for_percentage as f64) * 100.0;
            }
        }
    }

    fn apply_differential_analysis(&self, node: &mut FlameGraphNode) -> Result<()> {
        if let Some(baseline_samples) = &self.baseline_samples {
            // Build baseline tree
            let mut baseline_root = FlameGraphNode {
                name: "root".to_string(),
                value: 0,
                delta: None,
                children: HashMap::new(),
                total_value: 0,
                self_value: 0,
                percentage: 100.0,
                color: None,
                metadata: HashMap::new(),
            };

            for sample in baseline_samples {
                self.merge_sample_into_tree(&mut baseline_root, sample);
            }

            // Calculate deltas
            self.calculate_deltas(node, &baseline_root);
        }
        Ok(())
    }

    fn calculate_deltas(&self, current: &mut FlameGraphNode, baseline: &FlameGraphNode) {
        let baseline_value =
            baseline.children.get(&current.name).map(|n| n.total_value as i64).unwrap_or(0);

        current.delta = Some(current.total_value as i64 - baseline_value);

        for (name, child) in &mut current.children {
            if let Some(baseline_child) = baseline.children.get(name) {
                self.calculate_deltas(child, baseline_child);
            } else {
                child.delta = Some(child.total_value as i64);
            }
        }
    }

    fn filter_noise_nodes(&self, node: &mut FlameGraphNode) {
        let threshold = (node.total_value as f64 * self.config.noise_threshold / 100.0) as u64;

        node.children.retain(|_, child| {
            self.filter_noise_nodes(child);
            child.total_value >= threshold
        });
    }

    fn count_unique_functions(&self, node: &FlameGraphNode) -> u64 {
        let mut count = 1; // Count this node
        for child in node.children.values() {
            count += self.count_unique_functions(child);
        }
        count
    }

    fn render_svg_node(
        &self,
        svg: &mut String,
        node: &FlameGraphNode,
        x: i32,
        y: i32,
        width: i32,
        depth: i32,
    ) -> Result<()> {
        if width < 1 {
            return Ok(());
        }

        let height = 20;
        let color = self.get_node_color(node);

        svg.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="white" stroke-width="0.5">
<title>{}: {:.2}% ({} samples)</title>
</rect>
<text x="{}" y="{}" font-size="12" font-family="Verdana" fill="black">{}</text>
"#,
            x, y + depth * height, width, height,
            color,
            node.name, node.percentage, node.value,
            x + 2, y + depth * height + 14,
            if width > 50 { &node.name } else { "" }
        ));

        // Render children
        let mut child_x = x;
        for child in node.children.values() {
            let child_width = if node.total_value > 0 {
                (width as f64 * child.total_value as f64 / node.total_value as f64) as i32
            } else {
                0
            };
            if child_width > 0 {
                self.render_svg_node(svg, child, child_x, y, child_width, depth + 1)?;
                child_x += child_width;
            }
        }

        Ok(())
    }

    fn get_node_color(&self, node: &FlameGraphNode) -> String {
        match &self.config.color_scheme {
            FlameGraphColorScheme::Hot => {
                let intensity = (node.percentage / 100.0 * 255.0) as u8;
                format!("rgb({}, {}, 0)", 255, 255 - intensity)
            },
            FlameGraphColorScheme::Cool => {
                let intensity = (node.percentage / 100.0 * 255.0) as u8;
                format!("rgb(0, {}, {})", intensity, 255)
            },
            FlameGraphColorScheme::Memory => {
                if node.name.contains("alloc") || node.name.contains("malloc") {
                    "#ff6b6b".to_string()
                } else {
                    "#4ecdc4".to_string()
                }
            },
            FlameGraphColorScheme::Differential => {
                match node.delta {
                    Some(delta) if delta > 0 => "#ff4444".to_string(), // Red for increases
                    Some(delta) if delta < 0 => "#44ff44".to_string(), // Green for decreases
                    _ => "#cccccc".to_string(),                        // Gray for no change
                }
            },
            FlameGraphColorScheme::Java => "#ff9800".to_string(),
            FlameGraphColorScheme::Random => {
                let hash = self.hash_string(&node.name);
                format!("hsl({}, 70%, 60%)", hash % 360)
            },
            FlameGraphColorScheme::Custom(colors) => {
                colors.get(&node.name).cloned().unwrap_or_else(|| "#cccccc".to_string())
            },
        }
    }

    fn hash_string(&self, s: &str) -> u32 {
        let mut hash = 0u32;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
        hash
    }

    fn convert_to_speedscope_format(&self, root: &FlameGraphNode) -> Result<serde_json::Value> {
        // Simplified Speedscope format conversion
        Ok(serde_json::json!({
            "version": "0.7.1",
            "profiles": [{
                "type": "sampled",
                "name": self.config.title,
                "unit": "nanoseconds",
                "startValue": 0,
                "endValue": root.total_value,
                "samples": [],
                "weights": []
            }]
        }))
    }

    fn convert_to_d3_format(&self, root: &FlameGraphNode) -> Result<serde_json::Value> {
        Ok(serde_json::to_value(root)?)
    }

    fn get_hot_functions(&self, limit: usize) -> Vec<HotFunctionInfo> {
        let mut functions = Vec::new();

        if let Some(root) = &self.root_node {
            self.collect_hot_functions(root, &mut functions);
        }

        functions.sort_by_key(|item| std::cmp::Reverse(item.total_time_ns));
        functions.truncate(limit);
        functions
    }

    fn collect_hot_functions(&self, node: &FlameGraphNode, functions: &mut Vec<HotFunctionInfo>) {
        functions.push(HotFunctionInfo {
            name: node.name.clone(),
            total_time_ns: node.total_value,
            self_time_ns: node.self_value,
            percentage: node.percentage,
            call_count: 1, // Simplified
        });

        for child in node.children.values() {
            self.collect_hot_functions(child, functions);
        }
    }

    fn get_memory_usage_stats(&self) -> MemoryUsageStats {
        let memory_samples: Vec<usize> =
            self.samples.iter().filter_map(|s| s.memory_usage).collect();

        if memory_samples.is_empty() {
            return MemoryUsageStats::default();
        }

        let total: usize = memory_samples.iter().sum();
        let max = memory_samples.iter().max().copied().unwrap_or(0);
        let min = memory_samples.iter().min().copied().unwrap_or(0);
        let avg = total / memory_samples.len();

        MemoryUsageStats {
            peak_memory_bytes: max,
            avg_memory_bytes: avg,
            min_memory_bytes: min,
            total_samples: memory_samples.len(),
        }
    }

    fn get_gpu_kernel_stats(&self) -> GpuKernelStats {
        let gpu_samples: Vec<&FlameGraphSample> =
            self.samples.iter().filter(|s| s.gpu_kernel.is_some()).collect();

        let total_gpu_time: u64 = gpu_samples.iter().map(|s| s.duration_ns).sum();
        let unique_kernels: std::collections::HashSet<String> =
            gpu_samples.iter().filter_map(|s| s.gpu_kernel.clone()).collect();

        GpuKernelStats {
            total_kernel_time_ns: total_gpu_time,
            unique_kernels: unique_kernels.len(),
            total_kernel_calls: gpu_samples.len(),
        }
    }

    fn get_differential_analysis(&self) -> Option<DifferentialAnalysis> {
        if !self.config.differential_mode || self.baseline_samples.is_none() {
            return None;
        }

        let current_total: u64 = self.samples.iter().map(|s| s.duration_ns).sum();
        let baseline_total: u64 =
            self.baseline_samples.as_ref()?.iter().map(|s| s.duration_ns).sum();

        let performance_change = if baseline_total > 0 {
            ((current_total as f64 - baseline_total as f64) / baseline_total as f64) * 100.0
        } else {
            0.0
        };

        Some(DifferentialAnalysis {
            baseline_samples: self.baseline_samples.as_ref()?.len(),
            current_samples: self.samples.len(),
            performance_change_percent: performance_change,
            is_regression: performance_change > 5.0,
            is_improvement: performance_change < -5.0,
        })
    }

    fn generate_performance_insights(&self) -> Vec<String> {
        let mut insights = Vec::new();

        if let Some(root) = &self.root_node {
            let hot_functions = self.get_hot_functions(3);

            if let Some(hottest) = hot_functions.first() {
                if hottest.percentage > 50.0 {
                    insights.push(format!(
                        "Function '{}' dominates execution time ({:.1}%)",
                        hottest.name, hottest.percentage
                    ));
                }
            }

            let gpu_stats = self.get_gpu_kernel_stats();
            if gpu_stats.total_kernel_calls > 0 {
                let gpu_percentage =
                    (gpu_stats.total_kernel_time_ns as f64 / root.total_value as f64) * 100.0;
                insights.push(format!(
                    "GPU kernels account for {:.1}% of execution time",
                    gpu_percentage
                ));
            }

            if let Some(diff) = self.get_differential_analysis() {
                if diff.is_regression {
                    insights.push(format!(
                        "Performance regression detected: {:.1}% slower than baseline",
                        diff.performance_change_percent
                    ));
                } else if diff.is_improvement {
                    insights.push(format!(
                        "Performance improvement: {:.1}% faster than baseline",
                        -diff.performance_change_percent
                    ));
                }
            }
        }

        if insights.is_empty() {
            insights.push("No significant performance patterns detected".to_string());
        }

        insights
    }
}

/// Hot function information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotFunctionInfo {
    pub name: String,
    pub total_time_ns: u64,
    pub self_time_ns: u64,
    pub percentage: f64,
    pub call_count: usize,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryUsageStats {
    pub peak_memory_bytes: usize,
    pub avg_memory_bytes: usize,
    pub min_memory_bytes: usize,
    pub total_samples: usize,
}

/// GPU kernel statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuKernelStats {
    pub total_kernel_time_ns: u64,
    pub unique_kernels: usize,
    pub total_kernel_calls: usize,
}

/// Differential analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialAnalysis {
    pub baseline_samples: usize,
    pub current_samples: usize,
    pub performance_change_percent: f64,
    pub is_regression: bool,
    pub is_improvement: bool,
}

/// Flame graph analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameGraphAnalysisReport {
    pub total_samples: usize,
    pub total_duration_ns: u64,
    pub unique_functions: u64,
    pub max_stack_depth: u64,
    pub hot_functions: Vec<HotFunctionInfo>,
    pub memory_usage_stats: MemoryUsageStats,
    pub gpu_kernel_stats: GpuKernelStats,
    pub differential_analysis: Option<DifferentialAnalysis>,
    pub performance_insights: Vec<String>,
}

/// Default configuration for flame graphs
impl Default for FlameGraphConfig {
    fn default() -> Self {
        Self {
            sampling_rate: 1000, // 1000 Hz
            min_width: 0.01,
            color_scheme: FlameGraphColorScheme::Hot,
            direction: FlameGraphDirection::TopDown,
            title: "Flame Graph".to_string(),
            subtitle: None,
            include_memory: true,
            include_gpu: true,
            differential_mode: false,
            merge_similar_stacks: true,
            filter_noise: true,
            noise_threshold: 0.1, // 0.1%
        }
    }
}

/// Integration with main Profiler
impl Profiler {
    /// Create flame graph profiler with current configuration
    pub fn create_flame_graph_profiler(&self) -> FlameGraphProfiler {
        let config = FlameGraphConfig {
            title: "TrustformeRS Debug Flame Graph".to_string(),
            subtitle: Some("Performance Analysis".to_string()),
            ..Default::default()
        };
        FlameGraphProfiler::new(config)
    }

    /// Start flame graph profiling
    pub async fn start_flame_graph_profiling(&mut self) -> Result<()> {
        // This would integrate with the main profiler's timing events
        tracing::info!("Starting integrated flame graph profiling");
        Ok(())
    }

    /// Export flame graph from current profiling data
    pub async fn export_flame_graph(
        &self,
        format: FlameGraphExportFormat,
        output_path: &Path,
    ) -> Result<()> {
        let mut flame_profiler = self.create_flame_graph_profiler();

        // Convert existing events to flame graph samples
        for event in self.get_events() {
            match event {
                ProfileEvent::FunctionCall {
                    function_name,
                    duration,
                    ..
                } => {
                    let sample = FlameGraphSample {
                        stack: vec![StackFrame {
                            function_name: function_name.clone(),
                            module_name: None,
                            file_name: None,
                            line_number: None,
                            address: None,
                        }],
                        duration_ns: duration.as_nanos() as u64,
                        timestamp: 0,
                        thread_id: 0,
                        cpu_id: None,
                        memory_usage: None,
                        gpu_kernel: None,
                        metadata: HashMap::new(),
                    };
                    flame_profiler.add_sample(sample);
                },
                ProfileEvent::LayerExecution {
                    layer_name,
                    layer_type,
                    forward_time,
                    ..
                } => {
                    let sample = FlameGraphSample {
                        stack: vec![
                            StackFrame {
                                function_name: "neural_network".to_string(),
                                module_name: Some("trustformers".to_string()),
                                file_name: None,
                                line_number: None,
                                address: None,
                            },
                            StackFrame {
                                function_name: format!("{}::{}", layer_type, layer_name),
                                module_name: Some("layers".to_string()),
                                file_name: None,
                                line_number: None,
                                address: None,
                            },
                        ],
                        duration_ns: forward_time.as_nanos() as u64,
                        timestamp: 0,
                        thread_id: 0,
                        cpu_id: None,
                        memory_usage: None,
                        gpu_kernel: None,
                        metadata: HashMap::new(),
                    };
                    flame_profiler.add_sample(sample);
                },
                _ => {}, // Handle other event types as needed
            }
        }

        flame_profiler.build_flame_graph()?;
        flame_profiler.export(format, output_path).await?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "flame_graph_profiler_tests.rs"]
mod flame_graph_profiler_tests;

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Wave 6c debug-sweep2: real stack capture and real SVG ------------

    #[test]
    fn parse_backtrace_extracts_real_frames_and_locations() {
        let text = "   0: std::backtrace::Backtrace::force_capture\n                    \x20            at /rustc/lib/backtrace.rs:10:5\n                    \x20  1: my_crate::my_module::my_function\n                    \x20            at src/lib.rs:42:9\n                    \x20  2: main\n";
        let frames = FlameGraphProfiler::parse_backtrace(text);
        // The std::backtrace frame is dropped as capture machinery.
        assert_eq!(frames.len(), 2, "{frames:?}");
        assert_eq!(frames[0].function_name, "my_crate::my_module::my_function");
        assert_eq!(
            frames[0].module_name.as_deref(),
            Some("my_crate::my_module")
        );
        assert_eq!(frames[0].file_name.as_deref(), Some("src/lib.rs"));
        assert_eq!(frames[0].line_number, Some(42));
        assert_eq!(frames[1].function_name, "main");
        assert_eq!(frames[1].line_number, None);
        // The old capture returned exactly one fictional frame.
        assert!(frames.iter().all(|f| f.function_name != "captured_function"));
    }

    #[test]
    fn parse_backtrace_returns_nothing_for_unparseable_text() {
        assert!(FlameGraphProfiler::parse_backtrace("").is_empty());
        assert!(FlameGraphProfiler::parse_backtrace("disabled backtrace").is_empty());
    }

    #[test]
    fn sample_current_stack_captures_this_test_function() {
        let mut profiler = FlameGraphProfiler::new(FlameGraphConfig::default());
        profiler.sample_current_stack(1_000).expect("a live stack must be capturable");
        let sample = profiler.samples.last().expect("one sample");
        assert!(!sample.stack.is_empty());
        // A real capture names real symbols, not a single fixed placeholder.
        assert!(
            sample.stack.iter().all(|f| f.function_name != "captured_function"),
            "{:?}",
            sample.stack
        );
        // CPU id is honestly absent rather than a fabricated 0.
        assert_eq!(sample.cpu_id, None);
        assert!(sample.thread_id > 0, "a real dense thread id is assigned");
    }

    #[test]
    fn thread_ids_are_stable_per_thread_and_distinct_across_threads() {
        let profiler = FlameGraphProfiler::new(FlameGraphConfig::default());
        let mine = profiler.get_current_thread_id();
        assert_eq!(
            mine,
            profiler.get_current_thread_id(),
            "stable within a thread"
        );
        let other = std::thread::spawn(move || {
            let p = FlameGraphProfiler::new(FlameGraphConfig::default());
            p.get_current_thread_id()
        })
        .join()
        .expect("thread join");
        assert_ne!(
            mine, other,
            "a different thread must get a different id (was always 1)"
        );
    }

    #[test]
    fn render_flame_svg_draws_one_rect_per_node_with_real_widths() {
        let mut root = FlameGraphNode {
            name: "root".to_string(),
            value: 0,
            delta: None,
            children: HashMap::new(),
            total_value: 100,
            self_value: 0,
            percentage: 100.0,
            color: None,
            metadata: HashMap::new(),
        };
        for (name, value) in [("hot", 75_u64), ("cold", 25_u64)] {
            root.children.insert(
                name.to_string(),
                FlameGraphNode {
                    name: name.to_string(),
                    value,
                    delta: None,
                    children: HashMap::new(),
                    total_value: value,
                    self_value: value,
                    percentage: value as f64,
                    color: None,
                    metadata: HashMap::new(),
                },
            );
        }

        let svg = FlameGraphProfiler::render_flame_svg(&root);
        assert!(svg.starts_with("<svg"), "must be a real SVG document");
        assert_eq!(
            svg.matches("<rect").count(),
            3,
            "root + two children:\n{svg}"
        );
        assert!(
            svg.contains("root — 0.000ms (100.00%)"),
            "real hover tooltip: {svg}"
        );
        assert!(
            svg.contains("(75.00%)") && svg.contains("(25.00%)"),
            "real shares: {svg}"
        );
        // "cold" sorts before "hot", so it is laid out first at x = 0 with a
        // quarter of the width.
        assert!(svg.contains("width=\"300.00\""), "25% of 1200px: {svg}");
        assert!(svg.contains("width=\"900.00\""), "75% of 1200px: {svg}");
    }

    #[test]
    fn flame_graph_frame_names_cannot_inject_markup() {
        let root = FlameGraphNode {
            name: "</svg><script>x</script>".to_string(),
            value: 1,
            delta: None,
            children: HashMap::new(),
            total_value: 1,
            self_value: 1,
            percentage: 100.0,
            color: None,
            metadata: HashMap::new(),
        };
        let svg = FlameGraphProfiler::render_flame_svg(&root);
        assert!(!svg.contains("<script>"));
        assert_eq!(svg.matches("</svg>").count(), 1);
    }

    fn make_config() -> FlameGraphConfig {
        FlameGraphConfig {
            sampling_rate: 100,
            min_width: 0.1,
            color_scheme: FlameGraphColorScheme::Hot,
            direction: FlameGraphDirection::TopDown,
            title: "Test Profile".to_string(),
            subtitle: None,
            include_memory: false,
            include_gpu: false,
            differential_mode: false,
            merge_similar_stacks: false,
            filter_noise: false,
            noise_threshold: 0.01,
        }
    }

    fn make_sample(func_name: &str, duration_ns: u64) -> FlameGraphSample {
        FlameGraphSample {
            stack: vec![StackFrame {
                function_name: func_name.to_string(),
                module_name: None,
                file_name: None,
                line_number: None,
                address: None,
            }],
            duration_ns,
            timestamp: 1000,
            thread_id: 1,
            cpu_id: Some(0),
            memory_usage: None,
            gpu_kernel: None,
            metadata: HashMap::new(),
        }
    }

    fn make_nested_sample(funcs: &[&str], duration_ns: u64) -> FlameGraphSample {
        let stack = funcs
            .iter()
            .map(|name| StackFrame {
                function_name: name.to_string(),
                module_name: None,
                file_name: None,
                line_number: None,
                address: None,
            })
            .collect();
        FlameGraphSample {
            stack,
            duration_ns,
            timestamp: 1000,
            thread_id: 1,
            cpu_id: None,
            memory_usage: None,
            gpu_kernel: None,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_flame_graph_profiler_creation() {
        let profiler = FlameGraphProfiler::new(make_config());
        assert!(profiler.samples.is_empty());
        assert!(profiler.root_node.is_none());
    }

    #[test]
    fn test_start_sampling() {
        let mut profiler = FlameGraphProfiler::new(make_config());
        let result = profiler.start_sampling();
        assert!(result.is_ok());
        assert!(profiler.sampling_timer.is_some());
    }

    #[test]
    fn test_add_sample() {
        let mut profiler = FlameGraphProfiler::new(make_config());
        profiler.add_sample(make_sample("main", 1000));
        assert_eq!(profiler.samples.len(), 1);
    }

    #[test]
    fn test_add_multiple_samples() {
        let mut profiler = FlameGraphProfiler::new(make_config());
        for i in 0..10 {
            profiler.add_sample(make_sample(&format!("func_{}", i), (i + 1) * 100));
        }
        assert_eq!(profiler.samples.len(), 10);
    }

    #[test]
    fn test_sample_gpu_kernel() {
        let mut profiler = FlameGraphProfiler::new(make_config());
        profiler.sample_gpu_kernel("matmul_kernel", 5000);
        assert_eq!(profiler.samples.len(), 1);
        assert_eq!(
            profiler.samples[0].gpu_kernel,
            Some("matmul_kernel".to_string())
        );
    }

    #[test]
    fn test_build_flame_graph_empty() {
        let mut profiler = FlameGraphProfiler::new(make_config());
        let result = profiler.build_flame_graph();
        assert!(result.is_err());
    }

    #[test]
    fn test_build_flame_graph_single_sample() {
        let mut profiler = FlameGraphProfiler::new(make_config());
        profiler.add_sample(make_sample("main", 1000));
        let result = profiler.build_flame_graph();
        assert!(result.is_ok());
        assert!(profiler.root_node.is_some());
    }

    #[test]
    fn test_build_flame_graph_nested_stacks() {
        let mut profiler = FlameGraphProfiler::new(make_config());
        profiler.add_sample(make_nested_sample(&["main", "compute", "matmul"], 5000));
        profiler.add_sample(make_nested_sample(&["main", "compute", "softmax"], 3000));
        profiler.add_sample(make_nested_sample(&["main", "io", "load_data"], 2000));
        let result = profiler.build_flame_graph();
        assert!(result.is_ok());
        let root = profiler.root_node.as_ref().expect("root should exist");
        assert!(root.children.contains_key("main"));
    }

    #[test]
    fn test_set_baseline() {
        let mut profiler = FlameGraphProfiler::new(make_config());
        profiler.add_sample(make_sample("func_a", 100));
        profiler.add_sample(make_sample("func_b", 200));
        profiler.set_baseline();
        assert!(profiler.baseline_samples.is_some());
        let baseline = profiler.baseline_samples.as_ref().expect("baseline should exist");
        assert_eq!(baseline.len(), 2);
    }

    #[test]
    fn test_performance_counters() {
        let mut profiler = FlameGraphProfiler::new(make_config());
        let start_result = profiler.start_sampling();
        assert!(start_result.is_ok());
        profiler.add_sample(make_nested_sample(&["a", "b", "c"], 100));
        profiler.add_sample(make_nested_sample(&["a", "d"], 200));
        let counter = profiler.performance_counters.get("samples_collected");
        assert_eq!(counter, Some(&2));
        let depth = profiler.performance_counters.get("stack_depth_max");
        assert_eq!(depth, Some(&3));
    }

    #[test]
    fn test_stop_sampling_with_data() {
        let mut profiler = FlameGraphProfiler::new(make_config());
        let _ = profiler.start_sampling();
        profiler.add_sample(make_sample("test_func", 500));
        let result = profiler.stop_sampling();
        assert!(result.is_ok());
        assert!(profiler.sampling_timer.is_none());
        assert!(profiler.root_node.is_some());
    }

    #[test]
    fn test_flame_graph_node_structure() {
        let mut profiler = FlameGraphProfiler::new(make_config());
        profiler.add_sample(make_nested_sample(&["root_fn", "child_fn"], 1000));
        profiler.add_sample(make_nested_sample(&["root_fn", "child_fn"], 2000));
        let _ = profiler.build_flame_graph();
        let root = profiler.root_node.as_ref().expect("root should exist");
        assert_eq!(root.name, "root");
    }

    #[test]
    fn test_stack_frame_equality() {
        let frame1 = StackFrame {
            function_name: "test".to_string(),
            module_name: None,
            file_name: None,
            line_number: None,
            address: None,
        };
        let frame2 = StackFrame {
            function_name: "test".to_string(),
            module_name: None,
            file_name: None,
            line_number: None,
            address: None,
        };
        assert_eq!(frame1, frame2);
    }

    #[test]
    fn test_flame_graph_config_differential_mode() {
        let mut config = make_config();
        config.differential_mode = true;
        let mut profiler = FlameGraphProfiler::new(config);
        profiler.add_sample(make_sample("func_a", 100));
        profiler.set_baseline();
        profiler.add_sample(make_sample("func_a", 200));
        let result = profiler.build_flame_graph();
        assert!(result.is_ok());
    }

    #[test]
    fn test_flame_graph_config_noise_filter() {
        let mut config = make_config();
        config.filter_noise = true;
        config.noise_threshold = 0.05;
        let mut profiler = FlameGraphProfiler::new(config);
        profiler.add_sample(make_nested_sample(&["main", "big_func"], 10000));
        profiler.add_sample(make_nested_sample(&["main", "tiny_func"], 1));
        let result = profiler.build_flame_graph();
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_current_stack() {
        let mut profiler = FlameGraphProfiler::new(make_config());
        let result = profiler.sample_current_stack(500);
        assert!(result.is_ok());
        assert_eq!(profiler.samples.len(), 1);
    }

    #[tokio::test]
    async fn test_export_no_graph() {
        let profiler = FlameGraphProfiler::new(make_config());
        let tmp = std::env::temp_dir().join("test_flamegraph_export.json");
        let result = profiler.export(FlameGraphExportFormat::JSON, &tmp).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_export_json() {
        let mut profiler = FlameGraphProfiler::new(make_config());
        profiler.add_sample(make_sample("test", 100));
        let _ = profiler.build_flame_graph();
        let tmp = std::env::temp_dir().join("test_flamegraph_export_ok.json");
        let result = profiler.export(FlameGraphExportFormat::JSON, &tmp).await;
        assert!(result.is_ok());
        let _ = tokio::fs::remove_file(&tmp).await;
    }

    #[test]
    fn test_flame_graph_color_schemes() {
        let schemes = [
            FlameGraphColorScheme::Hot,
            FlameGraphColorScheme::Cool,
            FlameGraphColorScheme::Java,
            FlameGraphColorScheme::Memory,
            FlameGraphColorScheme::Differential,
            FlameGraphColorScheme::Random,
            FlameGraphColorScheme::Custom(HashMap::new()),
        ];
        assert_eq!(schemes.len(), 7);
    }

    #[test]
    fn test_flame_graph_directions() {
        let directions = [FlameGraphDirection::TopDown, FlameGraphDirection::BottomUp];
        assert_eq!(directions.len(), 2);
    }

    #[test]
    fn test_flame_graph_export_formats() {
        let formats = [
            FlameGraphExportFormat::SVG,
            FlameGraphExportFormat::InteractiveHTML,
            FlameGraphExportFormat::JSON,
            FlameGraphExportFormat::Speedscope,
            FlameGraphExportFormat::D3,
            FlameGraphExportFormat::Folded,
        ];
        assert_eq!(formats.len(), 6);
    }

    #[test]
    fn test_flame_graph_config_with_subtitle() {
        let mut config = make_config();
        config.subtitle = Some("Test Subtitle".to_string());
        let profiler = FlameGraphProfiler::new(config);
        assert!(profiler.samples.is_empty());
    }

    #[test]
    fn test_flame_graph_config_with_gpu_and_memory() {
        let mut config = make_config();
        config.include_gpu = true;
        config.include_memory = true;
        let profiler = FlameGraphProfiler::new(config);
        assert!(profiler.samples.is_empty());
    }

    #[test]
    fn test_flame_graph_node_creation() {
        let node = FlameGraphNode {
            name: "test_func".to_string(),
            value: 1000,
            delta: None,
            children: HashMap::new(),
            total_value: 1000,
            self_value: 500,
            percentage: 50.0,
            color: Some("#ff0000".to_string()),
            metadata: HashMap::new(),
        };
        assert_eq!(node.name, "test_func");
        assert_eq!(node.value, 1000);
        assert!((node.percentage - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stack_frame_with_all_fields() {
        let frame = StackFrame {
            function_name: "my_func".to_string(),
            module_name: Some("my_module".to_string()),
            file_name: Some("src/lib.rs".to_string()),
            line_number: Some(42),
            address: Some(0x12345678),
        };
        assert_eq!(frame.function_name, "my_func");
        assert_eq!(frame.module_name, Some("my_module".to_string()));
        assert_eq!(frame.line_number, Some(42));
    }
}
