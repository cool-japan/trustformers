//! Performance optimization advisor
//!
//! Analyzes model performance and provides actionable optimization suggestions
//! based on profiling data, architecture analysis, and hardware characteristics.

#![allow(unused_variables)] // Optimization advisor

use crate::errors::Result;
use crate::visualization::{GraphAnalyzer, ModelGraph};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use super::metrics::{LatencyMetrics, MemoryMetrics, ThroughputMetrics};
use super::profiler::ProfileResult;

/// Optimization category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationCategory {
    /// Model architecture optimizations
    Architecture,
    /// Memory and caching optimizations
    Memory,
    /// Compute kernel optimizations
    Compute,
    /// Quantization and compression
    Quantization,
    /// Parallelization strategies
    Parallelization,
    /// Hardware-specific optimizations
    Hardware,
    /// I/O and data loading
    DataPipeline,
}

impl fmt::Display for OptimizationCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Architecture => write!(f, "Architecture"),
            Self::Memory => write!(f, "Memory"),
            Self::Compute => write!(f, "Compute"),
            Self::Quantization => write!(f, "Quantization"),
            Self::Parallelization => write!(f, "Parallelization"),
            Self::Hardware => write!(f, "Hardware"),
            Self::DataPipeline => write!(f, "Data Pipeline"),
        }
    }
}

/// Optimization impact level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ImpactLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for ImpactLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Medium => write!(f, "Medium"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// Implementation difficulty
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Difficulty {
    Easy,   // Config change or simple modification
    Medium, // Requires some code changes
    Hard,   // Significant refactoring needed
}

impl fmt::Display for Difficulty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Easy => write!(f, "Easy"),
            Self::Medium => write!(f, "Medium"),
            Self::Hard => write!(f, "Hard"),
        }
    }
}

/// Single optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Unique identifier
    pub id: String,
    /// Category
    pub category: OptimizationCategory,
    /// Impact level
    pub impact: ImpactLevel,
    /// Implementation difficulty
    pub difficulty: Difficulty,
    /// Title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Expected performance improvement
    pub expected_improvement: PerformanceImprovement,
    /// Implementation steps
    pub implementation_steps: Vec<String>,
    /// Code examples (optional)
    pub code_examples: Option<Vec<CodeExample>>,
    /// Warnings or caveats
    pub warnings: Vec<String>,
    /// Related suggestions
    pub related_suggestions: Vec<String>,
}

/// Expected performance improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImprovement {
    /// Latency reduction percentage
    pub latency_reduction: Option<f32>,
    /// Throughput increase percentage
    pub throughput_increase: Option<f32>,
    /// Memory reduction percentage
    pub memory_reduction: Option<f32>,
    /// Additional metrics
    pub other_metrics: HashMap<String, String>,
}

/// Code example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    /// Language (rust, toml, etc)
    pub language: String,
    /// Code snippet
    pub code: String,
    /// Description
    pub description: String,
}

/// Hardware information for optimization decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    /// CPU model
    pub cpu_model: Option<String>,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// GPU model
    pub gpu_model: Option<String>,
    /// GPU memory in MB
    pub gpu_memory_mb: Option<usize>,
    /// Available system memory in MB
    pub system_memory_mb: usize,
    /// SIMD capabilities
    pub simd_capabilities: Vec<String>,
}

impl Default for HardwareInfo {
    fn default() -> Self {
        Self {
            cpu_model: None,
            cpu_cores: num_cpus::get(),
            gpu_model: None,
            gpu_memory_mb: None,
            system_memory_mb: 8192, // Default 8GB
            simd_capabilities: Vec::new(),
        }
    }
}

/// Analysis context for optimization advisor
#[derive(Debug, Clone)]
pub struct AnalysisContext {
    /// Model graph (optional)
    pub model_graph: Option<ModelGraph>,
    /// Profiling results (optional)
    pub profile_results: Option<ProfileResult>,
    /// Latency metrics
    pub latency_metrics: Option<LatencyMetrics>,
    /// Memory metrics
    pub memory_metrics: Option<MemoryMetrics>,
    /// Throughput metrics
    pub throughput_metrics: Option<ThroughputMetrics>,
    /// Hardware information
    pub hardware_info: HardwareInfo,
    /// Current configuration
    pub current_config: HashMap<String, String>,
}

/// Optimization advisor
pub struct OptimizationAdvisor {
    /// Analysis rules
    rules: Vec<Box<dyn OptimizationRule>>,
}

impl Default for OptimizationAdvisor {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationAdvisor {
    /// Create a new optimization advisor
    pub fn new() -> Self {
        let rules = Self::create_default_rules();
        Self { rules }
    }

    /// Add a custom rule
    pub fn add_rule(&mut self, rule: Box<dyn OptimizationRule>) {
        self.rules.push(rule);
    }

    /// Analyze and provide optimization suggestions
    pub fn analyze(&self, context: &AnalysisContext) -> Result<OptimizationReport> {
        let mut suggestions = Vec::new();

        // Run all rules
        for rule in &self.rules {
            if let Some(suggestion) = rule.analyze(context)? {
                suggestions.push(suggestion);
            }
        }

        // Sort by impact (highest first) and difficulty (easiest first)
        suggestions.sort_by(|a, b| b.impact.cmp(&a.impact).then(a.difficulty.cmp(&b.difficulty)));

        // Create report
        Ok(OptimizationReport {
            suggestions: suggestions.clone(),
            summary: Self::create_summary(&suggestions, context),
            hardware_info: context.hardware_info.clone(),
        })
    }

    /// Create default optimization rules
    fn create_default_rules() -> Vec<Box<dyn OptimizationRule>> {
        vec![
            Box::new(AttentionOptimizationRule),
            Box::new(MemoryOptimizationRule),
            Box::new(QuantizationRule),
            Box::new(ParallelizationRule),
            Box::new(KernelFusionRule),
            Box::new(CachingRule),
            Box::new(BatchingRule),
            Box::new(MixedPrecisionRule),
            Box::new(FlashAttentionRule),
            Box::new(GradientCheckpointingRule),
        ]
    }

    /// Create summary statistics
    fn create_summary(
        suggestions: &[OptimizationSuggestion],
        context: &AnalysisContext,
    ) -> OptimizationSummary {
        let total_suggestions = suggestions.len();

        let mut by_category = HashMap::new();
        let mut by_impact = HashMap::new();

        for suggestion in suggestions {
            *by_category.entry(suggestion.category).or_insert(0) += 1;
            *by_impact.entry(suggestion.impact).or_insert(0) += 1;
        }

        // Calculate potential improvements
        let mut total_latency_reduction = 0.0;
        let mut total_memory_reduction = 0.0;
        let mut total_throughput_increase = 0.0;

        for suggestion in suggestions {
            if let Some(reduction) = suggestion.expected_improvement.latency_reduction {
                total_latency_reduction += reduction;
            }
            if let Some(reduction) = suggestion.expected_improvement.memory_reduction {
                total_memory_reduction += reduction;
            }
            if let Some(increase) = suggestion.expected_improvement.throughput_increase {
                total_throughput_increase += increase;
            }
        }

        OptimizationSummary {
            total_suggestions,
            suggestions_by_category: by_category,
            suggestions_by_impact: by_impact,
            potential_latency_reduction: total_latency_reduction,
            potential_memory_reduction: total_memory_reduction,
            potential_throughput_increase: total_throughput_increase,
        }
    }
}

/// Optimization report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    /// Optimization suggestions
    pub suggestions: Vec<OptimizationSuggestion>,
    /// Summary statistics
    pub summary: OptimizationSummary,
    /// Hardware information used
    pub hardware_info: HardwareInfo,
}

impl OptimizationReport {
    /// Get high-impact suggestions
    pub fn high_impact_suggestions(&self) -> Vec<&OptimizationSuggestion> {
        self.suggestions.iter().filter(|s| s.impact >= ImpactLevel::High).collect()
    }

    /// Get easy-to-implement suggestions
    pub fn easy_suggestions(&self) -> Vec<&OptimizationSuggestion> {
        self.suggestions.iter().filter(|s| s.difficulty == Difficulty::Easy).collect()
    }

    /// Get suggestions by category
    pub fn suggestions_by_category(
        &self,
        category: OptimizationCategory,
    ) -> Vec<&OptimizationSuggestion> {
        self.suggestions.iter().filter(|s| s.category == category).collect()
    }

    /// Format report as markdown
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str("# Performance Optimization Report\n\n");

        // Summary
        md.push_str("## Summary\n\n");
        md.push_str(&format!(
            "Total suggestions: {}\n",
            self.summary.total_suggestions
        ));
        md.push_str(&format!(
            "Potential improvements:\n- Latency reduction: {:.1}%\n- Memory reduction: {:.1}%\n- Throughput increase: {:.1}%\n\n",
            self.summary.potential_latency_reduction,
            self.summary.potential_memory_reduction,
            self.summary.potential_throughput_increase
        ));

        // Suggestions by impact
        md.push_str("## Suggestions by Impact\n\n");
        for impact in [
            ImpactLevel::Critical,
            ImpactLevel::High,
            ImpactLevel::Medium,
            ImpactLevel::Low,
        ] {
            let suggestions: Vec<_> =
                self.suggestions.iter().filter(|s| s.impact == impact).collect();

            if !suggestions.is_empty() {
                md.push_str(&format!("### {} Impact\n\n", impact));
                for suggestion in suggestions {
                    md.push_str(&format!(
                        "- **{}** ({}, {}): {}\n",
                        suggestion.title,
                        suggestion.category,
                        suggestion.difficulty,
                        suggestion.description
                    ));
                }
                md.push('\n');
            }
        }

        // Detailed suggestions
        md.push_str("## Detailed Suggestions\n\n");
        for (i, suggestion) in self.suggestions.iter().enumerate() {
            md.push_str(&format!("### {}. {}\n\n", i + 1, suggestion.title));
            md.push_str(&format!(
                "**Category:** {} | **Impact:** {} | **Difficulty:** {}\n\n",
                suggestion.category, suggestion.impact, suggestion.difficulty
            ));
            md.push_str(&format!("{}\n\n", suggestion.description));

            // Expected improvements
            md.push_str("**Expected Improvements:**\n");
            if let Some(lat) = suggestion.expected_improvement.latency_reduction {
                md.push_str(&format!("- Latency reduction: {:.1}%\n", lat));
            }
            if let Some(mem) = suggestion.expected_improvement.memory_reduction {
                md.push_str(&format!("- Memory reduction: {:.1}%\n", mem));
            }
            if let Some(thr) = suggestion.expected_improvement.throughput_increase {
                md.push_str(&format!("- Throughput increase: {:.1}%\n", thr));
            }
            md.push('\n');

            // Implementation steps
            if !suggestion.implementation_steps.is_empty() {
                md.push_str("**Implementation Steps:**\n");
                for step in &suggestion.implementation_steps {
                    md.push_str(&format!("1. {}\n", step));
                }
                md.push('\n');
            }

            // Code examples
            if let Some(examples) = &suggestion.code_examples {
                for example in examples {
                    md.push_str(&format!("**{}:**\n", example.description));
                    md.push_str(&format!(
                        "```{}\n{}\n```\n\n",
                        example.language, example.code
                    ));
                }
            }

            // Warnings
            if !suggestion.warnings.is_empty() {
                md.push_str("**⚠️ Warnings:**\n");
                for warning in &suggestion.warnings {
                    md.push_str(&format!("- {}\n", warning));
                }
                md.push('\n');
            }
        }

        md
    }
}

/// Summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSummary {
    /// Total number of suggestions
    pub total_suggestions: usize,
    /// Suggestions by category
    pub suggestions_by_category: HashMap<OptimizationCategory, usize>,
    /// Suggestions by impact
    pub suggestions_by_impact: HashMap<ImpactLevel, usize>,
    /// Total potential latency reduction
    pub potential_latency_reduction: f32,
    /// Total potential memory reduction
    pub potential_memory_reduction: f32,
    /// Total potential throughput increase
    pub potential_throughput_increase: f32,
}

/// Trait for optimization rules
pub trait OptimizationRule: Send + Sync {
    /// Analyze context and provide suggestion if applicable
    fn analyze(&self, context: &AnalysisContext) -> Result<Option<OptimizationSuggestion>>;
}

// Built-in optimization rules

/// Attention optimization rule
struct AttentionOptimizationRule;

impl OptimizationRule for AttentionOptimizationRule {
    fn analyze(&self, context: &AnalysisContext) -> Result<Option<OptimizationSuggestion>> {
        // Check if model has attention layers
        if let Some(graph) = &context.model_graph {
            let attention_nodes: Vec<_> =
                graph.nodes.iter().filter(|n| n.node_type == "Attention").collect();

            if !attention_nodes.is_empty() {
                // Check for long sequences
                for node in &attention_nodes {
                    if let Some(shape) = &node.input_shape {
                        if shape.len() >= 2 && shape[1] > 512 {
                            return Ok(Some(OptimizationSuggestion {
                                id: "attention_optimization".to_string(),
                                category: OptimizationCategory::Architecture,
                                impact: ImpactLevel::High,
                                difficulty: Difficulty::Medium,
                                title: "Use Sparse or Linear Attention".to_string(),
                                description: format!(
                                    "Detected attention operations on sequences of length {}. \
                                    Consider using sparse attention patterns or linear attention \
                                    variants to reduce O(n²) complexity.",
                                    shape[1]
                                ),
                                expected_improvement: PerformanceImprovement {
                                    latency_reduction: Some(30.0),
                                    throughput_increase: Some(50.0),
                                    memory_reduction: Some(40.0),
                                    other_metrics: HashMap::new(),
                                },
                                implementation_steps: vec![
                                    "Identify attention layers that process long sequences".to_string(),
                                    "Replace with sparse attention (e.g., local, strided, or random patterns)".to_string(),
                                    "Consider using Linformer, Performer, or similar linear attention variants".to_string(),
                                    "Benchmark to ensure quality is maintained".to_string(),
                                ],
                                code_examples: Some(vec![CodeExample {
                                    language: "rust".to_string(),
                                    description: "Using sparse attention".to_string(),
                                    code: r#"// Replace standard attention
let attention = SparseMultiHeadAttention::new(
    config.hidden_size,
    config.num_heads,
    SparsePattern::Local { window_size: 256 },
);"#.to_string(),
                                }]),
                                warnings: vec![
                                    "May slightly reduce model quality".to_string(),
                                    "Requires retraining or fine-tuning".to_string(),
                                ],
                                related_suggestions: vec!["flash_attention".to_string()],
                            }));
                        }
                    }
                }
            }
        }

        Ok(None)
    }
}

/// Memory optimization rule
struct MemoryOptimizationRule;

impl OptimizationRule for MemoryOptimizationRule {
    fn analyze(&self, context: &AnalysisContext) -> Result<Option<OptimizationSuggestion>> {
        if let Some(memory_metrics) = &context.memory_metrics {
            let peak_mb = (memory_metrics.peak_bytes as f32) / (1024.0 * 1024.0);
            let current_mb = (memory_metrics.current_bytes as f32) / (1024.0 * 1024.0);

            // Check for high memory fragmentation
            if peak_mb > current_mb * 2.0 {
                return Ok(Some(OptimizationSuggestion {
                    id: "memory_fragmentation".to_string(),
                    category: OptimizationCategory::Memory,
                    impact: ImpactLevel::Medium,
                    difficulty: Difficulty::Easy,
                    title: "Reduce Memory Fragmentation".to_string(),
                    description: format!(
                        "High memory fragmentation detected. Peak memory ({:.1}MB) is \
                        significantly higher than current usage ({:.1}MB).",
                        peak_mb, current_mb
                    ),
                    expected_improvement: PerformanceImprovement {
                        latency_reduction: Some(5.0),
                        throughput_increase: Some(10.0),
                        memory_reduction: Some(30.0),
                        other_metrics: HashMap::new(),
                    },
                    implementation_steps: vec![
                        "Use memory pools for frequently allocated tensors".to_string(),
                        "Implement tensor recycling".to_string(),
                        "Consider using a custom allocator".to_string(),
                    ],
                    code_examples: None,
                    warnings: vec![],
                    related_suggestions: vec!["gradient_checkpointing".to_string()],
                }));
            }
        }

        Ok(None)
    }
}

/// Quantization rule
struct QuantizationRule;

impl OptimizationRule for QuantizationRule {
    fn analyze(&self, context: &AnalysisContext) -> Result<Option<OptimizationSuggestion>> {
        // Check if quantization is not already enabled
        let is_quantized =
            context.current_config.get("quantization").map(|v| v == "true").unwrap_or(false);

        if !is_quantized {
            if let Some(graph) = &context.model_graph {
                // Check model size
                let report = GraphAnalyzer::analyze(graph);
                if report.total_parameters > 100_000_000 {
                    // 100M parameters
                    return Ok(Some(OptimizationSuggestion {
                        id: "quantization".to_string(),
                        category: OptimizationCategory::Quantization,
                        impact: ImpactLevel::High,
                        difficulty: Difficulty::Medium,
                        title: "Enable Model Quantization".to_string(),
                        description: format!(
                            "Model has {:.1}M parameters. Quantization can reduce memory \
                            usage and improve inference speed with minimal quality loss.",
                            report.total_parameters as f32 / 1e6
                        ),
                        expected_improvement: PerformanceImprovement {
                            latency_reduction: Some(40.0),
                            throughput_increase: Some(60.0),
                            memory_reduction: Some(75.0),
                            other_metrics: HashMap::from([
                                ("model_size_reduction".to_string(), "75%".to_string()),
                            ]),
                        },
                        implementation_steps: vec![
                            "Choose quantization method (INT8, INT4, or dynamic)".to_string(),
                            "Calibrate quantization on representative data".to_string(),
                            "Apply quantization to weights and activations".to_string(),
                            "Validate accuracy on test set".to_string(),
                        ],
                        code_examples: Some(vec![CodeExample {
                            language: "rust".to_string(),
                            description: "Enabling INT8 quantization".to_string(),
                            code: r#"use trustformers_core::quantization::{QuantizationConfig, quantize_int8};

let config = QuantizationConfig {
    bits: 8,
    symmetric: true,
    per_channel: true,
};

let quantized_model = quantize_int8(&model, &config)?;"#.to_string(),
                        }]),
                        warnings: vec![
                            "May reduce model accuracy by 1-2%".to_string(),
                            "Requires calibration dataset".to_string(),
                        ],
                        related_suggestions: vec!["smoothquant".to_string()],
                    }));
                }
            }
        }

        Ok(None)
    }
}

/// Parallelization rule
struct ParallelizationRule;

impl OptimizationRule for ParallelizationRule {
    fn analyze(&self, context: &AnalysisContext) -> Result<Option<OptimizationSuggestion>> {
        let cpu_cores = context.hardware_info.cpu_cores;

        // Check if parallelization is underutilized
        if cpu_cores > 4 {
            let parallel_enabled = context
                .current_config
                .get("parallel_enabled")
                .map(|v| v == "true")
                .unwrap_or(false);

            if !parallel_enabled {
                return Ok(Some(OptimizationSuggestion {
                    id: "enable_parallelization".to_string(),
                    category: OptimizationCategory::Parallelization,
                    impact: ImpactLevel::High,
                    difficulty: Difficulty::Easy,
                    title: "Enable Multi-Threading".to_string(),
                    description: format!(
                        "System has {} CPU cores but parallelization is not enabled. \
                        Enable multi-threading for significant performance gains.",
                        cpu_cores
                    ),
                    expected_improvement: PerformanceImprovement {
                        latency_reduction: Some(50.0),
                        throughput_increase: Some(
                            100.0 * (cpu_cores as f32 - 1.0) / cpu_cores as f32,
                        ),
                        memory_reduction: None,
                        other_metrics: HashMap::new(),
                    },
                    implementation_steps: vec![
                        "Set parallel_enabled = true in configuration".to_string(),
                        "Configure thread pool size based on workload".to_string(),
                        "Enable parallel data loading".to_string(),
                    ],
                    code_examples: Some(vec![CodeExample {
                        language: "rust".to_string(),
                        description: "Enabling parallelization".to_string(),
                        code: r#"use trustformers_core::parallel::init_parallelism;

// Initialize with optimal thread count
init_parallelism(num_cpus::get())?;

// Or use specific configuration
let config = ParallelConfig {
    num_threads: 8,
    thread_pool_size: 16,
    enable_numa: true,
};"#
                        .to_string(),
                    }]),
                    warnings: vec![
                        "May increase memory usage".to_string(),
                        "Watch for thread contention".to_string(),
                    ],
                    related_suggestions: vec!["data_parallel".to_string()],
                }));
            }
        }

        Ok(None)
    }
}

/// Kernel fusion rule
struct KernelFusionRule;

impl OptimizationRule for KernelFusionRule {
    fn analyze(&self, context: &AnalysisContext) -> Result<Option<OptimizationSuggestion>> {
        if let Some(profile) = &context.profile_results {
            // Look for small operations (high call count, low avg time)
            if profile.call_count > 100 && profile.avg_time.as_millis() < 1 {
                return Ok(Some(OptimizationSuggestion {
                    id: "kernel_fusion".to_string(),
                    category: OptimizationCategory::Compute,
                    impact: ImpactLevel::Medium,
                    difficulty: Difficulty::Medium,
                    title: "Enable Kernel Fusion".to_string(),
                    description: format!(
                        "Found {} frequently-called small operations that could benefit \
                        from kernel fusion to reduce overhead.",
                        profile.call_count
                    ),
                    expected_improvement: PerformanceImprovement {
                        latency_reduction: Some(20.0),
                        throughput_increase: Some(25.0),
                        memory_reduction: Some(10.0),
                        other_metrics: HashMap::from([(
                            "kernel_launches_reduction".to_string(),
                            "70%".to_string(),
                        )]),
                    },
                    implementation_steps: vec![
                        "Identify fusable operation patterns".to_string(),
                        "Implement fused kernels for common patterns".to_string(),
                        "Use graph optimization pass to detect fusion opportunities".to_string(),
                    ],
                    code_examples: Some(vec![CodeExample {
                        language: "rust".to_string(),
                        description: "Using fused operations".to_string(),
                        code: r#"// Instead of separate operations
let x = linear(input)?;
let x = layer_norm(x)?;
let x = gelu(x)?;

// Use fused operation
let x = fused_linear_norm_gelu(input, weight, bias)?;"#
                            .to_string(),
                    }]),
                    warnings: vec!["Requires custom kernel implementation".to_string()],
                    related_suggestions: vec![],
                }));
            }
        }

        Ok(None)
    }
}

/// Caching rule
struct CachingRule;

impl OptimizationRule for CachingRule {
    fn analyze(&self, context: &AnalysisContext) -> Result<Option<OptimizationSuggestion>> {
        // Check for inference scenario
        let is_inference =
            context.current_config.get("mode").map(|v| v == "inference").unwrap_or(false);

        if is_inference {
            let cache_enabled = context
                .current_config
                .get("kv_cache_enabled")
                .map(|v| v == "true")
                .unwrap_or(false);

            if !cache_enabled {
                return Ok(Some(OptimizationSuggestion {
                    id: "enable_kv_cache".to_string(),
                    category: OptimizationCategory::Memory,
                    impact: ImpactLevel::High,
                    difficulty: Difficulty::Easy,
                    title: "Enable KV-Cache for Inference".to_string(),
                    description: "Key-Value caching can significantly speed up autoregressive \
                        generation by reusing attention computations."
                        .to_string(),
                    expected_improvement: PerformanceImprovement {
                        latency_reduction: Some(60.0),
                        throughput_increase: Some(150.0),
                        memory_reduction: None,
                        other_metrics: HashMap::from([(
                            "generation_speedup".to_string(),
                            "3-5x".to_string(),
                        )]),
                    },
                    implementation_steps: vec![
                        "Enable KV-cache in generation config".to_string(),
                        "Configure cache size based on max sequence length".to_string(),
                        "Implement cache management for batch inference".to_string(),
                    ],
                    code_examples: Some(vec![CodeExample {
                        language: "rust".to_string(),
                        description: "Enabling KV-cache".to_string(),
                        code: r#"let generation_config = GenerationConfig {
    use_cache: true,
    max_cache_size: 2048,
    cache_implementation: CacheType::Static,
    ..Default::default()
};"#
                        .to_string(),
                    }]),
                    warnings: vec!["Increases memory usage".to_string()],
                    related_suggestions: vec![],
                }));
            }
        }

        Ok(None)
    }
}

/// Batching rule
struct BatchingRule;

impl OptimizationRule for BatchingRule {
    fn analyze(&self, context: &AnalysisContext) -> Result<Option<OptimizationSuggestion>> {
        if let Some(latency) = &context.latency_metrics {
            // Check if running with batch size 1
            let batch_size = context
                .current_config
                .get("batch_size")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1);

            if batch_size == 1 && context.hardware_info.gpu_model.is_some() {
                return Ok(Some(OptimizationSuggestion {
                    id: "increase_batch_size".to_string(),
                    category: OptimizationCategory::DataPipeline,
                    impact: ImpactLevel::High,
                    difficulty: Difficulty::Easy,
                    title: "Increase Batch Size".to_string(),
                    description: "Running with batch size 1 underutilizes GPU. \
                        Increase batch size to improve GPU utilization and throughput."
                        .to_string(),
                    expected_improvement: PerformanceImprovement {
                        latency_reduction: None,
                        throughput_increase: Some(300.0),
                        memory_reduction: None,
                        other_metrics: HashMap::from([(
                            "gpu_utilization".to_string(),
                            "+80%".to_string(),
                        )]),
                    },
                    implementation_steps: vec![
                        "Profile memory usage with larger batch sizes".to_string(),
                        "Find optimal batch size for your GPU memory".to_string(),
                        "Implement dynamic batching for variable-length inputs".to_string(),
                    ],
                    code_examples: Some(vec![CodeExample {
                        language: "rust".to_string(),
                        description: "Dynamic batching".to_string(),
                        code: r#"let batcher = DynamicBatcher::new()
    .with_max_batch_size(32)
    .with_max_latency_ms(50)
    .with_padding_strategy(PaddingStrategy::Smart);"#
                            .to_string(),
                    }]),
                    warnings: vec![
                        "Increases memory usage proportionally".to_string(),
                        "May increase individual request latency".to_string(),
                    ],
                    related_suggestions: vec!["dynamic_batching".to_string()],
                }));
            }
        }

        Ok(None)
    }
}

/// Mixed precision rule
struct MixedPrecisionRule;

impl OptimizationRule for MixedPrecisionRule {
    fn analyze(&self, context: &AnalysisContext) -> Result<Option<OptimizationSuggestion>> {
        // Check GPU capabilities
        if let Some(gpu_model) = &context.hardware_info.gpu_model {
            let supports_mixed_precision = gpu_model.contains("V100")
                || gpu_model.contains("A100")
                || gpu_model.contains("RTX")
                || gpu_model.contains("H100");

            if supports_mixed_precision {
                let mixed_precision_enabled = context
                    .current_config
                    .get("mixed_precision")
                    .map(|v| v == "true")
                    .unwrap_or(false);

                if !mixed_precision_enabled {
                    return Ok(Some(OptimizationSuggestion {
                        id: "mixed_precision".to_string(),
                        category: OptimizationCategory::Compute,
                        impact: ImpactLevel::High,
                        difficulty: Difficulty::Easy,
                        title: "Enable Mixed Precision Training".to_string(),
                        description: format!(
                            "GPU {} supports mixed precision. Enable FP16/BF16 compute \
                            with FP32 accumulation for faster training.",
                            gpu_model
                        ),
                        expected_improvement: PerformanceImprovement {
                            latency_reduction: Some(40.0),
                            throughput_increase: Some(100.0),
                            memory_reduction: Some(50.0),
                            other_metrics: HashMap::from([(
                                "tensor_core_utilization".to_string(),
                                "Enabled".to_string(),
                            )]),
                        },
                        implementation_steps: vec![
                            "Enable automatic mixed precision (AMP)".to_string(),
                            "Add loss scaling to prevent underflow".to_string(),
                            "Monitor for numerical instabilities".to_string(),
                        ],
                        code_examples: Some(vec![CodeExample {
                            language: "rust".to_string(),
                            description: "Enabling mixed precision".to_string(),
                            code: r#"let config = TrainingConfig {
    mixed_precision: MixedPrecisionConfig {
        enabled: true,
        compute_dtype: DataType::Float16,
        accumulate_dtype: DataType::Float32,
        loss_scale: LossScale::Dynamic(1024.0),
    },
    ..Default::default()
};"#
                            .to_string(),
                        }]),
                        warnings: vec![
                            "May cause numerical instabilities in some models".to_string(),
                            "Requires careful loss scaling".to_string(),
                        ],
                        related_suggestions: vec![],
                    }));
                }
            }
        }

        Ok(None)
    }
}

/// Flash attention rule
struct FlashAttentionRule;

impl OptimizationRule for FlashAttentionRule {
    fn analyze(&self, context: &AnalysisContext) -> Result<Option<OptimizationSuggestion>> {
        if let Some(graph) = &context.model_graph {
            let has_attention = graph.nodes.iter().any(|n| n.node_type == "Attention");

            if has_attention && context.hardware_info.gpu_model.is_some() {
                let flash_attention_enabled = context
                    .current_config
                    .get("flash_attention")
                    .map(|v| v == "true")
                    .unwrap_or(false);

                if !flash_attention_enabled {
                    return Ok(Some(OptimizationSuggestion {
                        id: "flash_attention".to_string(),
                        category: OptimizationCategory::Compute,
                        impact: ImpactLevel::High,
                        difficulty: Difficulty::Medium,
                        title: "Enable Flash Attention".to_string(),
                        description: "Flash Attention provides 2-4x speedup for attention \
                            computation with reduced memory usage."
                            .to_string(),
                        expected_improvement: PerformanceImprovement {
                            latency_reduction: Some(50.0),
                            throughput_increase: Some(100.0),
                            memory_reduction: Some(60.0),
                            other_metrics: HashMap::from([(
                                "attention_flops".to_string(),
                                "-50%".to_string(),
                            )]),
                        },
                        implementation_steps: vec![
                            "Replace standard attention with Flash Attention implementation"
                                .to_string(),
                            "Verify GPU architecture compatibility (Ampere or newer)".to_string(),
                            "Adjust for block size and head dimension requirements".to_string(),
                        ],
                        code_examples: Some(vec![CodeExample {
                            language: "rust".to_string(),
                            description: "Using Flash Attention".to_string(),
                            code: r#"use trustformers_core::layers::FlashMultiHeadAttention;

let attention = FlashMultiHeadAttention::new(
    config.hidden_size,
    config.num_heads,
    FlashAttentionConfig {
        block_size: 128,
        use_causal_mask: true,
    },
)?"#
                            .to_string(),
                        }]),
                        warnings: vec![
                            "Requires GPU with compute capability >= 8.0".to_string(),
                            "Head dimension must be compatible with block size".to_string(),
                        ],
                        related_suggestions: vec!["attention_optimization".to_string()],
                    }));
                }
            }
        }

        Ok(None)
    }
}

/// Gradient checkpointing rule
struct GradientCheckpointingRule;

impl OptimizationRule for GradientCheckpointingRule {
    fn analyze(&self, context: &AnalysisContext) -> Result<Option<OptimizationSuggestion>> {
        if let Some(memory_metrics) = &context.memory_metrics {
            // Check if memory usage is high
            let memory_usage_gb = (memory_metrics.peak_bytes as f32) / (1024.0 * 1024.0 * 1024.0);
            let available_memory_gb = context.hardware_info.system_memory_mb as f32 / 1024.0;

            if memory_usage_gb > available_memory_gb * 0.8 {
                return Ok(Some(OptimizationSuggestion {
                    id: "gradient_checkpointing".to_string(),
                    category: OptimizationCategory::Memory,
                    impact: ImpactLevel::High,
                    difficulty: Difficulty::Easy,
                    title: "Enable Gradient Checkpointing".to_string(),
                    description: format!(
                        "High memory usage detected ({:.1}GB / {:.1}GB). \
                        Gradient checkpointing trades compute for memory by recomputing \
                        activations during backward pass.",
                        memory_usage_gb, available_memory_gb
                    ),
                    expected_improvement: PerformanceImprovement {
                        latency_reduction: Some(-20.0), // Negative because it increases latency
                        throughput_increase: None,
                        memory_reduction: Some(50.0),
                        other_metrics: HashMap::from([(
                            "max_batch_size".to_string(),
                            "+2x".to_string(),
                        )]),
                    },
                    implementation_steps: vec![
                        "Enable gradient checkpointing for transformer blocks".to_string(),
                        "Choose checkpointing granularity (per-layer or per-block)".to_string(),
                        "Monitor training time increase (typically 20-30%)".to_string(),
                    ],
                    code_examples: Some(vec![CodeExample {
                        language: "rust".to_string(),
                        description: "Enabling gradient checkpointing".to_string(),
                        code: r#"let model = TransformerModel::new(config)
    .with_gradient_checkpointing(true)
    .with_checkpoint_segments(4);"#
                            .to_string(),
                    }]),
                    warnings: vec![
                        "Increases training time by 20-30%".to_string(),
                        "Not beneficial for inference".to_string(),
                    ],
                    related_suggestions: vec!["memory_fragmentation".to_string()],
                }));
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_advisor() {
        let advisor = OptimizationAdvisor::new();

        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: Some(MemoryMetrics {
                current_bytes: 1000 * 1024 * 1024,
                peak_bytes: 3000 * 1024 * 1024,
                allocated_bytes: 5000 * 1024 * 1024,
                reserved_bytes: 0,
                num_allocations: 1000,
                num_deallocations: 500,
                fragmentation_percent: 10.0,
            }),
            throughput_metrics: None,
            hardware_info: HardwareInfo::default(),
            current_config: HashMap::new(),
        };

        let report = advisor.analyze(&context).unwrap();
        assert!(!report.suggestions.is_empty());
    }

    #[test]
    fn test_report_markdown() {
        let report = OptimizationReport {
            suggestions: vec![OptimizationSuggestion {
                id: "test1".to_string(),
                category: OptimizationCategory::Memory,
                impact: ImpactLevel::High,
                difficulty: Difficulty::Easy,
                title: "Test Optimization".to_string(),
                description: "Test description".to_string(),
                expected_improvement: PerformanceImprovement {
                    latency_reduction: Some(20.0),
                    throughput_increase: Some(30.0),
                    memory_reduction: Some(40.0),
                    other_metrics: HashMap::new(),
                },
                implementation_steps: vec!["Step 1".to_string()],
                code_examples: None,
                warnings: vec![],
                related_suggestions: vec![],
            }],
            summary: OptimizationSummary {
                total_suggestions: 1,
                suggestions_by_category: HashMap::from([(OptimizationCategory::Memory, 1)]),
                suggestions_by_impact: HashMap::from([(ImpactLevel::High, 1)]),
                potential_latency_reduction: 20.0,
                potential_memory_reduction: 40.0,
                potential_throughput_increase: 30.0,
            },
            hardware_info: HardwareInfo::default(),
        };

        let markdown = report.to_markdown();
        assert!(markdown.contains("Performance Optimization Report"));
        assert!(markdown.contains("Test Optimization"));
    }
}
