//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::errors::Result;
use crate::performance::metrics::{LatencyMetrics, MemoryMetrics, ThroughputMetrics};
use crate::performance::profiler::ProfileResult;
use crate::visualization::ModelGraph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::functions::OptimizationRule;

/// Attention optimization rule
pub(crate) struct AttentionOptimizationRule;
/// Mixed precision rule
pub(crate) struct MixedPrecisionRule;
/// Memory optimization rule
pub(crate) struct MemoryOptimizationRule;
/// Gradient checkpointing rule
pub(crate) struct GradientCheckpointingRule;
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
/// Kernel fusion rule
pub(crate) struct KernelFusionRule;
/// Flash attention rule
pub(crate) struct FlashAttentionRule;
/// Quantization rule
pub(crate) struct QuantizationRule;
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
/// Batching rule
pub(crate) struct BatchingRule;
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
/// Parallelization rule
pub(crate) struct ParallelizationRule;
/// Optimization advisor
pub struct OptimizationAdvisor {
    /// Analysis rules
    rules: Vec<Box<dyn OptimizationRule>>,
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
        for rule in &self.rules {
            if let Some(suggestion) = rule.analyze(context)? {
                suggestions.push(suggestion);
            }
        }
        suggestions.sort_by(|a, b| b.impact.cmp(&a.impact).then(a.difficulty.cmp(&b.difficulty)));
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
        _context: &AnalysisContext,
    ) -> OptimizationSummary {
        let total_suggestions = suggestions.len();
        let mut by_category = HashMap::new();
        let mut by_impact = HashMap::new();
        for suggestion in suggestions {
            *by_category.entry(suggestion.category).or_insert(0) += 1;
            *by_impact.entry(suggestion.impact).or_insert(0) += 1;
        }
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
/// Implementation difficulty
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}
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
/// Optimization impact level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ImpactLevel {
    Low,
    Medium,
    High,
    Critical,
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
        md.push_str("## Summary\n\n");
        md.push_str(&format!(
            "Total suggestions: {}\n",
            self.summary.total_suggestions
        ));
        md.push_str(
            &format!(
                "Potential improvements:\n- Latency reduction: {:.1}%\n- Memory reduction: {:.1}%\n- Throughput increase: {:.1}%\n\n",
                self.summary.potential_latency_reduction, self.summary
                .potential_memory_reduction, self.summary.potential_throughput_increase
            ),
        );
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
        md.push_str("## Detailed Suggestions\n\n");
        for (i, suggestion) in self.suggestions.iter().enumerate() {
            md.push_str(&format!("### {}. {}\n\n", i + 1, suggestion.title));
            md.push_str(&format!(
                "**Category:** {} | **Impact:** {} | **Difficulty:** {}\n\n",
                suggestion.category, suggestion.impact, suggestion.difficulty
            ));
            md.push_str(&format!("{}\n\n", suggestion.description));
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
            if !suggestion.implementation_steps.is_empty() {
                md.push_str("**Implementation Steps:**\n");
                for step in &suggestion.implementation_steps {
                    md.push_str(&format!("1. {}\n", step));
                }
                md.push('\n');
            }
            if let Some(examples) = &suggestion.code_examples {
                for example in examples {
                    md.push_str(&format!("**{}:**\n", example.description));
                    md.push_str(&format!(
                        "```{}\n{}\n```\n\n",
                        example.language, example.code
                    ));
                }
            }
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
/// Caching rule
pub(crate) struct CachingRule;
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
