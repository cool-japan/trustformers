//! Optimization Engine
//!
//! This module provides intelligent optimization suggestion generation for mobile
//! ML inference workloads using machine learning models and expert system rules.

use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use tracing::debug;

use super::super::types::*;
use crate::device_info::ThermalState;

/// Intelligent optimization suggestion engine
#[derive(Debug)]
pub struct OptimizationEngine {
    /// Engine configuration
    config: OptimizationEngineConfig,
    /// Generated optimization suggestions
    active_suggestions: HashMap<String, OptimizationSuggestion>,
    /// Suggestion generation rules
    optimization_rules: Vec<OptimizationRule>,
    /// Suggestion ranking system
    suggestion_ranker: SuggestionRanker,
    /// Impact estimation models
    impact_estimator: ImpactEstimator,
    /// Suggestion history and tracking
    suggestion_history: VecDeque<OptimizationEvent>,
    /// Engine performance statistics
    engine_stats: OptimizationEngineStats,
}

/// Optimization rule for suggestion generation
#[derive(Debug, Clone)]
pub struct OptimizationRule {
    /// Rule identifier
    pub id: String,
    /// Human-readable rule name
    pub name: String,
    /// Trigger condition
    pub condition: OptimizationCondition,
    /// Generated suggestion template
    pub suggestion_template: OptimizationSuggestion,
    /// Estimated performance impact
    pub estimated_impact: ImpactLevel,
    /// Implementation difficulty
    pub difficulty: DifficultyLevel,
    /// Rule confidence score
    pub confidence: f32,
    /// Whether the rule is enabled
    pub enabled: bool,
}

/// Conditions that trigger optimization suggestions
#[derive(Debug, Clone)]
pub enum OptimizationCondition {
    /// High memory usage pattern
    HighMemoryUsage {
        threshold_percent: f32,
        pattern: MemoryUsagePattern,
    },
    /// Low cache hit rate
    LowCacheHitRate {
        threshold_percent: f32,
        cache_type: CacheType,
    },
    /// High inference latency
    InferenceLatencyHigh {
        threshold_ms: f32,
        model_type: Option<String>,
    },
    /// Thermal throttling events
    ThermalThrottling {
        frequency: u32,
        severity: ThermalState,
    },
    /// High battery drain
    BatteryDrainHigh {
        threshold_mw: f32,
        context: BatteryContext,
    },
    /// Low network bandwidth utilization
    NetworkBandwidthLow {
        threshold_mbps: f32,
        connection_type: NetworkType,
    },
    /// GPU underutilization
    GPUUnderutilized {
        threshold_percent: f32,
        workload_type: WorkloadType,
    },
    /// CPU inefficiency patterns
    CPUInefficiency {
        pattern: CPUUsagePattern,
        severity: f32,
    },
}

/// Memory usage patterns for optimization
#[derive(Debug, Clone)]
pub enum MemoryUsagePattern {
    /// Steady high usage
    SteadyHigh,
    /// Rapid growth
    RapidGrowth,
    /// Memory leaks
    MemoryLeaks,
    /// Fragmentation
    Fragmentation,
    /// Large allocations
    LargeAllocations,
}

/// Cache types for optimization analysis
#[derive(Debug, Clone)]
pub enum CacheType {
    /// Model cache
    Model,
    /// Tensor cache
    Tensor,
    /// Computation cache
    Computation,
    /// Network cache
    Network,
    /// General purpose cache
    General,
}

/// Battery usage context
#[derive(Debug, Clone)]
pub enum BatteryContext {
    /// During inference
    Inference,
    /// During model loading
    ModelLoading,
    /// During background tasks
    Background,
    /// During network operations
    Network,
    /// General usage
    General,
}

/// Network connection types
#[derive(Debug, Clone)]
pub enum NetworkType {
    /// WiFi connection
    WiFi,
    /// Cellular connection
    Cellular,
    /// Low power Bluetooth
    Bluetooth,
    /// Unknown connection type
    Unknown,
}

/// ML workload types
#[derive(Debug, Clone)]
pub enum WorkloadType {
    /// Computer vision workloads
    ComputerVision,
    /// Natural language processing
    NLP,
    /// Audio processing
    Audio,
    /// General ML inference
    General,
}

/// CPU usage patterns
#[derive(Debug, Clone)]
pub enum CPUUsagePattern {
    /// High single-core usage
    SingleCoreHigh,
    /// Poor multi-core utilization
    PoorMultiCore,
    /// Frequent context switching
    FrequentSwitching,
    /// Thermal throttling induced
    ThermalLimited,
    /// Inefficient algorithms
    InefficientAlgorithms,
}

/// Optimization event for historical tracking
#[derive(Debug, Clone)]
pub struct OptimizationEvent {
    pub timestamp: std::time::Instant,
    pub suggestion_id: String,
    pub event_type: String,
    pub performance_impact: f32,
    pub implementation_status: String,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Optimization engine statistics
#[derive(Debug, Clone, Default)]
pub struct OptimizationEngineStats {
    /// Total suggestions generated
    pub suggestions_generated: u64,
    /// Suggestions accepted by users
    pub suggestions_accepted: u64,
    /// Suggestions that led to improvements
    pub successful_suggestions: u64,
    /// Average improvement achieved
    pub avg_improvement_percent: f32,
    /// Engine accuracy rate
    pub accuracy_rate: f32,
}

/// Suggestion ranking system
#[derive(Debug)]
pub struct SuggestionRanker {
    /// Ranking algorithm
    algorithm: RankingAlgorithm,
}

/// Impact estimation models
#[derive(Debug)]
pub struct ImpactEstimator {
    /// Impact models
    models: Vec<ImpactModel>,
}

/// Internal ranking algorithm
#[derive(Debug)]
struct RankingAlgorithm;

/// Internal impact model
#[derive(Debug)]
struct ImpactModel;

impl OptimizationEngine {
    /// Create a new optimization engine with the given configuration
    pub fn new(config: OptimizationEngineConfig) -> Result<Self> {
        let optimization_rules = Self::initialize_default_rules();

        Ok(Self {
            config,
            active_suggestions: HashMap::new(),
            optimization_rules,
            suggestion_ranker: SuggestionRanker {
                algorithm: RankingAlgorithm,
            },
            impact_estimator: ImpactEstimator { models: Vec::new() },
            suggestion_history: VecDeque::new(),
            engine_stats: OptimizationEngineStats::default(),
        })
    }

    /// Generate optimization suggestions based on current metrics and bottlenecks
    pub fn generate_suggestions(
        &mut self,
        metrics: &MobileMetricsSnapshot,
        bottlenecks: &[PerformanceBottleneck],
    ) -> Result<Vec<OptimizationSuggestion>> {
        let mut suggestions = Vec::new();

        // Generate suggestions from rules
        for rule in &self.optimization_rules {
            if !rule.enabled {
                continue;
            }

            if self.evaluate_optimization_condition(&rule.condition, metrics)? {
                let suggestion = self.create_suggestion_from_rule(rule, metrics)?;
                suggestions.push(suggestion);

                // Track suggestion generation
                self.engine_stats.suggestions_generated += 1;
                debug!("Generated optimization suggestion: {}", rule.name);
            }
        }

        // Generate suggestions based on bottlenecks
        for bottleneck in bottlenecks {
            let bottleneck_suggestions =
                self.generate_bottleneck_suggestions(bottleneck, metrics)?;
            suggestions.extend(bottleneck_suggestions);
        }

        // Rank and prioritize suggestions
        let ranked_suggestions = self.rank_suggestions(suggestions, metrics)?;

        // Record them so `get_active_suggestions` / `get_all_suggestions`
        // report what was generated. Without this the accessors returned an
        // empty list no matter how many rules fired.
        self.active_suggestions.clear();
        for suggestion in &ranked_suggestions {
            self.active_suggestions.insert(suggestion.title.clone(), suggestion.clone());
        }

        Ok(ranked_suggestions)
    }

    /// Initialize default optimization rules
    fn initialize_default_rules() -> Vec<OptimizationRule> {
        vec![
            OptimizationRule {
                id: "enable_quantization".to_string(),
                name: "Enable Model Quantization".to_string(),
                condition: OptimizationCondition::InferenceLatencyHigh {
                    threshold_ms: 200.0,
                    model_type: None,
                },
                suggestion_template: OptimizationSuggestion {
                    suggestion_type: SuggestionType::ModelOptimization,
                    title: "Enable INT8 Quantization".to_string(),
                    description:
                        "Reduce model size and inference latency by using 8-bit quantization"
                            .to_string(),
                    implementation_steps: vec![
                        "Configure quantization in model settings".to_string(),
                        "Test accuracy impact on validation dataset".to_string(),
                        "Deploy quantized model if accuracy is acceptable".to_string(),
                    ],
                    estimated_improvement: String::new(), // filled in from the measurement that triggered the rule
                    difficulty: DifficultyLevel::Medium,
                    priority: PriorityLevel::High,
                },
                estimated_impact: ImpactLevel::High,
                difficulty: DifficultyLevel::Medium,
                confidence: 0.85,
                enabled: true,
            },
            OptimizationRule {
                id: "enable_gpu_acceleration".to_string(),
                name: "Enable GPU Acceleration".to_string(),
                condition: OptimizationCondition::CPUInefficiency {
                    pattern: CPUUsagePattern::SingleCoreHigh,
                    severity: 0.8,
                },
                suggestion_template: OptimizationSuggestion {
                    suggestion_type: SuggestionType::HardwareOptimization,
                    title: "Enable GPU Acceleration".to_string(),
                    description:
                        "Offload computation to GPU for better performance and lower CPU usage"
                            .to_string(),
                    implementation_steps: vec![
                        "Check GPU availability and compatibility".to_string(),
                        "Configure GPU backend in inference settings".to_string(),
                        "Monitor GPU utilization and performance".to_string(),
                    ],
                    estimated_improvement: String::new(), // filled in from the measurement that triggered the rule
                    difficulty: DifficultyLevel::Low,
                    priority: PriorityLevel::High,
                },
                estimated_impact: ImpactLevel::High,
                difficulty: DifficultyLevel::Low,
                confidence: 0.9,
                enabled: true,
            },
            OptimizationRule {
                id: "reduce_batch_size".to_string(),
                name: "Reduce Batch Size".to_string(),
                condition: OptimizationCondition::HighMemoryUsage {
                    threshold_percent: 85.0,
                    pattern: MemoryUsagePattern::SteadyHigh,
                },
                suggestion_template: OptimizationSuggestion {
                    suggestion_type: SuggestionType::PerformanceOptimization,
                    title: "Reduce Inference Batch Size".to_string(),
                    description: "Lower memory usage by processing smaller batches of data"
                        .to_string(),
                    implementation_steps: vec![
                        "Identify current batch size configuration".to_string(),
                        "Reduce batch size by 25-50%".to_string(),
                        "Monitor latency and throughput impact".to_string(),
                    ],
                    estimated_improvement: String::new(), // filled in from the measurement that triggered the rule
                    difficulty: DifficultyLevel::Low,
                    priority: PriorityLevel::Medium,
                },
                estimated_impact: ImpactLevel::Medium,
                difficulty: DifficultyLevel::Low,
                confidence: 0.95,
                enabled: true,
            },
        ]
    }

    /// Evaluate an optimization condition against current metrics
    fn evaluate_optimization_condition(
        &self,
        condition: &OptimizationCondition,
        metrics: &MobileMetricsSnapshot,
    ) -> Result<bool> {
        match condition {
            OptimizationCondition::HighMemoryUsage {
                threshold_percent, ..
            } => Ok(metrics
                .memory
                .as_ref()
                .and_then(|memory| memory.resident_share_percent())
                .is_some_and(|share| share > *threshold_percent)),
            OptimizationCondition::InferenceLatencyHigh { threshold_ms, .. } => {
                Ok(metrics.inference.avg_latency_ms > *threshold_ms as f64)
            },
            OptimizationCondition::LowCacheHitRate {
                threshold_percent, ..
            } => {
                // Real hit rate: the collector's inference tracker counts
                // every recorded hit and miss. This used to compare an
                // invented constant 50.0 against the threshold, so the rule
                // fired (or not) regardless of the actual cache.
                let hit_rate_percent = metrics.inference.cache_hit_rate as f32 * 100.0;
                Ok(metrics.inference.total_inferences > 0 && hit_rate_percent < *threshold_percent)
            },
            OptimizationCondition::CPUInefficiency { severity, .. } => {
                Ok(metrics.cpu.as_ref().is_some_and(|cpu| cpu.usage_percent > (severity * 100.0)))
            },
            _ => Ok(false), // Simplified for other conditions
        }
    }

    /// Create an optimization suggestion from a triggered rule
    fn create_suggestion_from_rule(
        &self,
        rule: &OptimizationRule,
        metrics: &MobileMetricsSnapshot,
    ) -> Result<OptimizationSuggestion> {
        let mut suggestion = rule.suggestion_template.clone();

        // State what was actually measured against the rule's threshold. The
        // previous line reported `"{}% improvement"` from a fixed 30.0 base --
        // a predicted speedup nothing had measured or modelled.
        suggestion.estimated_improvement = self.describe_trigger(rule, metrics);

        Ok(suggestion)
    }

    /// Generate suggestions specifically for detected bottlenecks
    fn generate_bottleneck_suggestions(
        &self,
        bottleneck: &PerformanceBottleneck,
        _metrics: &MobileMetricsSnapshot,
    ) -> Result<Vec<OptimizationSuggestion>> {
        let suggestions = match bottleneck.bottleneck_type {
            BottleneckType::Memory => vec![OptimizationSuggestion {
                suggestion_type: SuggestionType::PerformanceOptimization,
                title: "Memory Optimization".to_string(),
                description: format!("Address {} memory bottleneck", bottleneck.description),
                implementation_steps: vec!["Optimize memory usage".to_string()],
                estimated_improvement: format!(
                    "memory bottleneck, impact score {:.1}/100",
                    bottleneck.impact_score
                ),
                difficulty: DifficultyLevel::Medium,
                priority: PriorityLevel::High,
            }],
            BottleneckType::CPU => vec![OptimizationSuggestion {
                suggestion_type: SuggestionType::HardwareOptimization,
                title: "CPU Optimization".to_string(),
                description: format!("Address {} CPU bottleneck", bottleneck.description),
                implementation_steps: vec!["Optimize CPU usage".to_string()],
                estimated_improvement: format!(
                    "CPU bottleneck, impact score {:.1}/100",
                    bottleneck.impact_score
                ),
                difficulty: DifficultyLevel::Medium,
                priority: PriorityLevel::High,
            }],
            BottleneckType::Latency => vec![OptimizationSuggestion {
                suggestion_type: SuggestionType::ModelOptimization,
                title: "Inference Latency Optimization".to_string(),
                description: format!("Address {} latency bottleneck", bottleneck.description),
                implementation_steps: vec![
                    "Enable INT8 quantization and re-measure".to_string(),
                    "Route the model through an available hardware backend".to_string(),
                ],
                estimated_improvement: format!(
                    "latency bottleneck, impact score {:.1}/100",
                    bottleneck.impact_score
                ),
                difficulty: DifficultyLevel::Medium,
                priority: PriorityLevel::High,
            }],
            BottleneckType::Cache => vec![OptimizationSuggestion {
                suggestion_type: SuggestionType::PerformanceOptimization,
                title: "Cache Effectiveness".to_string(),
                description: format!("Address {} cache bottleneck", bottleneck.description),
                implementation_steps: vec![
                    "Increase the KV/result cache size".to_string(),
                    "Review the cache key so equivalent requests share an entry".to_string(),
                ],
                estimated_improvement: format!(
                    "cache bottleneck, impact score {:.1}/100",
                    bottleneck.impact_score
                ),
                difficulty: DifficultyLevel::Low,
                priority: PriorityLevel::Medium,
            }],
            // GPU, Network, Thermal and Power bottlenecks cannot be detected
            // at all today (no measured input -- see
            // `BottleneckDetector::evaluate_rule`), so there is no suggestion
            // template for them to reach.
            BottleneckType::GPU
            | BottleneckType::Network
            | BottleneckType::Thermal
            | BottleneckType::Power => Vec::new(),
        };

        Ok(suggestions)
    }

    /// Rank suggestions by priority and relevance
    fn rank_suggestions(
        &self,
        mut suggestions: Vec<OptimizationSuggestion>,
        metrics: &MobileMetricsSnapshot,
    ) -> Result<Vec<OptimizationSuggestion>> {
        // Sort by priority and difficulty
        suggestions.sort_by(|a, b| {
            // Higher priority first, then lower difficulty
            match b.priority.cmp(&a.priority) {
                std::cmp::Ordering::Equal => a.difficulty.cmp(&b.difficulty),
                other => other,
            }
        });

        // Limit to top suggestions
        suggestions.truncate(10);
        Ok(suggestions)
    }

    /// Calculate estimated improvement for a rule in current context
    /// Describe the measurement that tripped this rule.
    ///
    /// This crate models no relationship between applying a suggestion and the
    /// speedup that follows, so it reports the observation rather than a
    /// predicted gain.
    fn describe_trigger(&self, rule: &OptimizationRule, metrics: &MobileMetricsSnapshot) -> String {
        match &rule.condition {
            OptimizationCondition::HighMemoryUsage {
                threshold_percent, ..
            } => match metrics.memory.as_ref().and_then(|m| m.resident_share_percent()) {
                Some(share) => format!(
                    "resident memory at {:.1}% of usable (rule threshold {:.1}%)",
                    share, threshold_percent
                ),
                None => "resident memory share not yet sampled".to_string(),
            },
            OptimizationCondition::InferenceLatencyHigh { threshold_ms, .. } => format!(
                "mean inference latency {:.1} ms over {} inferences (rule threshold {:.1} ms)",
                metrics.inference.avg_latency_ms, metrics.inference.total_inferences, threshold_ms
            ),
            OptimizationCondition::LowCacheHitRate {
                threshold_percent, ..
            } => format!(
                "cache hit rate {:.1}% (rule threshold {:.1}%)",
                metrics.inference.cache_hit_rate as f32 * 100.0,
                threshold_percent
            ),
            OptimizationCondition::CPUInefficiency { severity, .. } => match metrics.cpu.as_ref() {
                Some(cpu) => format!(
                    "CPU usage {:.1}% (rule threshold {:.1}%)",
                    cpu.usage_percent,
                    severity * 100.0
                ),
                None => "CPU usage not measured".to_string(),
            },
            _ => format!("rule `{}` condition met", rule.id),
        }
    }

    /// Adjust confidence based on current context
    fn adjust_confidence_for_context(
        &self,
        base_confidence: f32,
        _metrics: &MobileMetricsSnapshot,
    ) -> Result<f32> {
        // Simplified: could factor in device capabilities, model type, etc.
        Ok(base_confidence.min(1.0).max(0.0))
    }

    /// Get engine statistics
    pub fn get_engine_stats(&self) -> &OptimizationEngineStats {
        &self.engine_stats
    }

    /// Get active suggestions
    pub fn get_active_suggestions(&self) -> Vec<OptimizationSuggestion> {
        self.active_suggestions.values().cloned().collect()
    }

    /// Every suggestion generated so far in this session.
    pub fn get_all_suggestions(&self) -> Vec<OptimizationSuggestion> {
        self.active_suggestions.values().cloned().collect()
    }

    /// Hot-reload the engine's configuration from the profiler's own config.
    pub fn update_config(
        &mut self,
        config: crate::mobile_performance_profiler::config::MobileProfilerConfig,
    ) -> Result<()> {
        self.config.enabled = config.enabled;
        self.config.generation_interval_ms = config.sampling.interval_ms;
        debug!("Updated optimization engine configuration");
        Ok(())
    }

    /// Mark a suggestion as implemented
    pub fn mark_suggestion_implemented(
        &mut self,
        suggestion_id: &str,
        performance_gain: f32,
    ) -> Result<()> {
        if let Some(suggestion) = self.active_suggestions.remove(suggestion_id) {
            // Record the optimization event
            let event = OptimizationEvent {
                timestamp: Instant::now(),
                suggestion_id: suggestion_id.to_string(),
                event_type: "implemented".to_string(),
                performance_impact: performance_gain,
                implementation_status: "completed".to_string(),
                metadata: HashMap::new(),
            };

            self.suggestion_history.push_back(event);
            self.engine_stats.suggestions_accepted += 1;

            if performance_gain > 0.0 {
                self.engine_stats.successful_suggestions += 1;
            }

            // Update average improvement
            let total_improvement = self.engine_stats.avg_improvement_percent
                * self.engine_stats.successful_suggestions as f32;
            self.engine_stats.avg_improvement_percent = (total_improvement + performance_gain)
                / self.engine_stats.successful_suggestions as f32;
        }

        Ok(())
    }
}

impl Default for OptimizationEngine {
    fn default() -> Self {
        // reason: built from a known-valid default config; construction is infallible
        // for the default config and Default cannot return a Result.
        Self::new(OptimizationEngineConfig::default())
            .expect("default OptimizationEngineConfig must yield a valid engine")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mobile_performance_profiler::types::{
        BottleneckSeverity, BottleneckType, CpuMetrics, InferenceMetrics, MemoryMetrics,
        MobileMetricsSnapshot, PerformanceBottleneck,
    };

    fn bottleneck(bottleneck_type: BottleneckType) -> PerformanceBottleneck {
        PerformanceBottleneck {
            bottleneck_type,
            severity: BottleneckSeverity::Medium,
            description: "test".to_string(),
            affected_component: "test".to_string(),
            impact_score: 42.0,
            suggestions: Vec::new(),
            timestamp: 1,
        }
    }

    /// Regression: `generate_bottleneck_suggestions` matched only Memory and
    /// CPU and fell through to `Vec::new()` for everything else, so the
    /// Latency and Cache bottlenecks the detector reports produced nothing.
    #[test]
    fn test_latency_and_cache_bottlenecks_produce_suggestions() {
        let engine = OptimizationEngine::default();
        let metrics = MobileMetricsSnapshot::default();

        for kind in [BottleneckType::Latency, BottleneckType::Cache] {
            let suggestions = engine
                .generate_bottleneck_suggestions(&bottleneck(kind), &metrics)
                .expect("suggestion generation");
            assert!(
                !suggestions.is_empty(),
                "{kind:?} bottleneck produced no suggestion"
            );
            // The reported figure states the measured impact, not a predicted
            // "% improvement" from a fixed base.
            assert!(suggestions[0].estimated_improvement.contains("impact score 42.0/100"));
        }
    }

    /// Regression: the `LowCacheHitRate` condition used to be
    /// `Ok(50.0 < *threshold_percent)` -- an invented constant compared
    /// against the threshold, so the rule's verdict never depended on the
    /// actual cache.
    #[test]
    fn test_cache_hit_rate_rule_reads_the_measured_rate() {
        let engine = OptimizationEngine::default();
        let condition = OptimizationCondition::LowCacheHitRate {
            threshold_percent: 60.0,
            cache_type: CacheType::Model,
        };

        let mut poor = MobileMetricsSnapshot::default();
        poor.inference = InferenceMetrics {
            total_inferences: 100,
            cache_hit_rate: 0.10,
            ..Default::default()
        };
        assert!(engine.evaluate_optimization_condition(&condition, &poor).expect("evaluate"));

        let mut good = poor.clone();
        good.inference.cache_hit_rate = 0.95;
        assert!(!engine.evaluate_optimization_condition(&condition, &good).expect("evaluate"));
    }

    /// Regression: `HighMemoryUsage` divided by `heap_total_mb`, which no
    /// platform publishes. It now reads the measured resident share, and an
    /// unmeasured snapshot cannot trip the rule.
    #[test]
    fn test_memory_rule_needs_a_real_measurement() {
        let engine = OptimizationEngine::default();
        let condition = OptimizationCondition::HighMemoryUsage {
            threshold_percent: 80.0,
            pattern: MemoryUsagePattern::SteadyHigh,
        };

        let unmeasured = MobileMetricsSnapshot::default();
        assert!(unmeasured.memory.is_none());
        assert!(!engine
            .evaluate_optimization_condition(&condition, &unmeasured)
            .expect("evaluate"));

        let mut pressured = unmeasured.clone();
        pressured.memory = Some(MemoryMetrics {
            heap_used_mb: 900.0,
            available_mb: 100.0,
            ..Default::default()
        });
        assert!(engine
            .evaluate_optimization_condition(&condition, &pressured)
            .expect("evaluate"));
    }

    /// CPU conditions likewise need a measurement.
    #[test]
    fn test_cpu_rule_needs_a_real_measurement() {
        let engine = OptimizationEngine::default();
        let condition = OptimizationCondition::CPUInefficiency {
            pattern: CPUUsagePattern::SingleCoreHigh,
            severity: 0.8,
        };

        let unmeasured = MobileMetricsSnapshot::default();
        assert!(!engine
            .evaluate_optimization_condition(&condition, &unmeasured)
            .expect("evaluate"));

        let mut busy = unmeasured.clone();
        busy.cpu = Some(CpuMetrics {
            usage_percent: 92.0,
            ..Default::default()
        });
        assert!(engine.evaluate_optimization_condition(&condition, &busy).expect("evaluate"));
    }
}
