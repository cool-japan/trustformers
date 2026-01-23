//! Advanced flame graph profiling implementation for TrustformeRS Debug

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
#[allow(dead_code)]
pub struct FlameGraphProfiler {
    config: FlameGraphConfig,
    samples: Vec<FlameGraphSample>,
    sampling_timer: Option<Instant>,
    root_node: Option<FlameGraphNode>,
    baseline_samples: Option<Vec<FlameGraphSample>>, // For differential analysis
    #[allow(dead_code)]
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

    /// Export as interactive HTML flame graph
    async fn export_interactive_html(
        &self,
        root: &FlameGraphNode,
        output_path: &Path,
    ) -> Result<()> {
        let json_data = serde_json::to_string(root)?;

        let html_content = format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        .flame-graph {{ width: 100%; height: 600px; border: 1px solid #ccc; }}
        .tooltip {{ position: absolute; background: rgba(0,0,0,0.8); color: white;
                   padding: 10px; border-radius: 4px; pointer-events: none; z-index: 1000; }}
        .controls {{ margin-bottom: 20px; }}
        .info {{ margin-top: 20px; font-size: 14px; color: #666; }}
    </style>
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
    <h1>{}</h1>
    <div class="controls">
        <button onclick="resetZoom()">Reset Zoom</button>
        <button onclick="searchFunction()">Search</button>
        <input type="text" id="searchInput" placeholder="Function name...">
    </div>
    <div id="flame-graph" class="flame-graph"></div>
    <div class="info">
        <p>Samples: {} | Functions: {} | Total Time: {:.2}ms</p>
        <p>Click to zoom, double-click to reset. Hover for details.</p>
    </div>
    <div id="tooltip" class="tooltip" style="display: none;"></div>

    <script>
        const data = {};
        // Interactive flame graph implementation would go here
        // This is a simplified version - full implementation would include D3.js visualization
        console.log('Flame graph data loaded:', data);
    </script>
</body>
</html>"#,
            self.config.title,
            self.config.title,
            self.samples.len(),
            self.count_unique_functions(root),
            root.total_value as f64 / 1_000_000.0, // Convert ns to ms
            json_data
        );

        tokio::fs::write(output_path, html_content).await?;
        tracing::info!("Exported interactive HTML flame graph to {:?}", output_path);
        Ok(())
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

    fn capture_stack_trace(&self) -> Result<Vec<StackFrame>> {
        // Simplified stack trace capture
        // In a real implementation, this would use platform-specific APIs
        Ok(vec![StackFrame {
            function_name: "captured_function".to_string(),
            module_name: Some("trustformers_debug".to_string()),
            file_name: Some("profiler.rs".to_string()),
            line_number: Some(1800),
            address: None,
        }])
    }

    fn get_current_thread_id(&self) -> u64 {
        // Simplified thread ID - would use thread::current().id() in practice
        1
    }

    fn get_current_cpu_id(&self) -> Option<u32> {
        // Would query current CPU ID in practice
        Some(0)
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

        functions.sort_by(|a, b| b.total_time_ns.cmp(&a.total_time_ns));
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
