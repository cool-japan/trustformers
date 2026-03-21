//! Sharing Capability Analyzer
//!
//! Analyzes resource sharing capabilities and provides optimization strategies
//! for safe concurrent access patterns.

use super::super::types::*;
use crate::test_performance_monitoring::types::CachedSharingCapability;
use anyhow::Result;
use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

pub struct SharingCapabilityAnalyzer {
    /// Sharing analysis strategies
    strategies: Arc<Mutex<Vec<Box<dyn SharingAnalysisStrategy + Send + Sync>>>>,

    /// Sharing patterns database
    patterns_db: Arc<RwLock<SharingPatternsDatabase>>,

    /// Capability cache for performance
    capability_cache: Arc<Mutex<HashMap<String, CachedSharingCapability>>>,

    /// Sharing performance history
    performance_history: Arc<Mutex<SharingPerformanceHistory>>,

    /// Configuration
    config: SharingAnalysisConfig,
}

impl SharingCapabilityAnalyzer {
    /// Creates a new sharing capability analyzer
    pub async fn new(config: SharingAnalysisConfig) -> Result<Self> {
        let mut strategies: Vec<Box<dyn SharingAnalysisStrategy + Send + Sync>> = Vec::new();

        // Initialize sharing analysis strategies
        strategies.push(Box::new(ReadOnlySharingStrategy::new(true, true)));
        strategies.push(Box::new(PartitionedSharingStrategy::new()?));
        strategies.push(Box::new(TemporalSharingStrategy::new()?));
        strategies.push(Box::new(AdaptiveSharingStrategy::new()?));

        let patterns_db = SharingPatternsDatabase::new();

        Ok(Self {
            strategies: Arc::new(Mutex::new(strategies)),
            patterns_db: Arc::new(RwLock::new(patterns_db)),
            capability_cache: Arc::new(Mutex::new(HashMap::new())),
            performance_history: Arc::new(Mutex::new(SharingPerformanceHistory::new())),
            config,
        })
    }

    /// Analyzes sharing capabilities for test execution data
    pub async fn analyze_sharing_capabilities(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<SharingAnalysisResult> {
        let start_time = Utc::now();

        // Check cache first
        if let Some(cached) = self.check_cache(&test_data.test_id).await {
            if cached.is_valid() {
                // Convert cached String result to SharingCapability enum
                let capability = match cached.result.as_str() {
                    "ReadOnly" => SharingCapability::ReadOnly,
                    "ReadWrite" => SharingCapability::ReadWrite,
                    "Exclusive" => SharingCapability::Exclusive,
                    "Shared" => SharingCapability::Shared,
                    "None" => SharingCapability::None,
                    _ => SharingCapability::None, // Default
                };

                return Ok(SharingAnalysisResult {
                    sharing_requirements: SynchronizationRequirements::default(),
                    confidence: cached.confidence,
                    sharing_capabilities: vec![capability],
                    optimizations: Vec::new(),
                    performance_predictions: Vec::new(),
                    strategy_results: Vec::new(),
                    analysis_duration: Duration::from_millis(0),
                });
            }
        }

        // Run sharing analysis strategies
        // Execute synchronously to avoid lifetime issues with mutex guards
        let analysis_results: Vec<_> = {
            let strategies = self.strategies.lock();
            strategies
                .iter()
                .map(|strategy| {
                    let strategy_name = strategy.name().to_string();
                    let analysis_start = Instant::now();
                    let result = strategy.analyze_sharing_capability(
                        &test_data.test_id,
                        &test_data.resource_access_patterns,
                    );
                    let analysis_duration = analysis_start.elapsed();
                    (strategy_name, result, analysis_duration)
                })
                .collect()
        };

        // Collect results
        let mut sharing_capability_structs = Vec::new(); // For helper methods
        let mut sharing_capability_enums = Vec::new(); // For result
        let mut strategy_results = Vec::new();

        for (strategy_name, result, duration) in analysis_results {
            match result {
                Ok(capability) => {
                    // Convert ResourceSharingCapabilities to SharingCapability enum
                    let sharing_cap =
                        if capability.supports_write_sharing && capability.supports_read_sharing {
                            SharingCapability::ReadWrite
                        } else if capability.supports_read_sharing {
                            SharingCapability::ReadOnly
                        } else if capability.supports_write_sharing {
                            SharingCapability::Exclusive
                        } else {
                            SharingCapability::None
                        };

                    strategy_results.push(SharingStrategyResult {
                        strategy: strategy_name,
                        capability: sharing_cap.clone(),
                        detection_duration: duration,
                        confidence: self.calculate_strategy_confidence(&capability) as f64,
                    });
                    sharing_capability_structs.push(capability);
                    sharing_capability_enums.push(sharing_cap);
                },
                Err(e) => {
                    log::warn!("Sharing analysis strategy failed: {}", e);
                },
            }
        }

        // Synthesize sharing requirements
        let _sharing_requirements =
            self.synthesize_sharing_requirements(&sharing_capability_structs)?;

        // Generate optimization recommendations
        let optimizations =
            self.generate_sharing_optimizations(&sharing_capability_structs).await?;

        // Calculate performance predictions
        let performance_prediction =
            self.predict_sharing_performance(&sharing_capability_structs).await?;
        let performance_predictions = vec![performance_prediction];

        let result = SharingAnalysisResult {
            sharing_requirements: SynchronizationRequirements::default(),
            sharing_capabilities: sharing_capability_enums,
            optimizations,
            performance_predictions,
            strategy_results,
            analysis_duration: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
            confidence: self.calculate_overall_sharing_confidence(&sharing_capability_structs)
                as f64,
        };

        // Cache result
        self.cache_result(&test_data.test_id, &result).await?;

        Ok(result)
    }

    /// Checks cache for existing analysis results
    async fn check_cache(&self, test_id: &str) -> Option<CachedSharingCapability> {
        let cache = self.capability_cache.lock();
        cache.get(test_id).cloned()
    }

    /// Calculates confidence for a sharing strategy
    fn calculate_strategy_confidence(&self, capability: &ResourceSharingCapabilities) -> f32 {
        let mut confidence_factors = Vec::new();

        // TODO: ResourceSharingCapabilities no longer has sharing_safety_level enum
        // It has safety_assessment: f64 instead
        // Factor in sharing safety
        confidence_factors.push(capability.safety_assessment);

        // TODO: performance_overhead → sharing_overhead
        // Factor in performance impact
        confidence_factors.push(1.0 - capability.sharing_overhead.abs());

        // TODO: ResourceSharingCapabilities no longer has implementation_complexity
        // Using sharing_overhead as proxy (lower overhead = less complex)
        // Factor in complexity
        confidence_factors.push(1.0 - capability.sharing_overhead);

        confidence_factors.iter().sum::<f64>() as f32 / confidence_factors.len() as f32
    }

    /// Synthesizes sharing requirements from all capabilities
    fn synthesize_sharing_requirements(
        &self,
        capabilities: &[ResourceSharingCapabilities],
    ) -> Result<SharingRequirements> {
        if capabilities.is_empty() {
            return Ok(SharingRequirements::default());
        }

        // Find the most restrictive but safe sharing approach
        // TODO: ResourceSharingCapabilities no longer has sharing_safety_level enum
        // It has safety_assessment: f64 instead (higher = safer)
        // Also no longer has implementation_complexity
        // Safe: is_empty() check above guarantees capabilities.first() is Some
        let safest_capability = capabilities
            .iter()
            .filter(|c| c.safety_assessment >= 0.7) // Safe or conditionally safe
            .min_by(|a, b| {
                // Use sharing_overhead as complexity proxy (lower = simpler)
                a.sharing_overhead
                    .partial_cmp(&b.sharing_overhead)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .or_else(|| capabilities.first())
            .expect("capabilities is non-empty after is_empty check");

        // Convert String to SharingMode enum
        let sharing_mode = match safest_capability.sharing_mode.as_str() {
            "ReadOnly" => SharingMode::ReadOnly,
            "Write" => SharingMode::Write,
            "ReadWrite" => SharingMode::ReadWrite,
            "Exclusive" => SharingMode::Exclusive,
            "ExclusiveWrite" => SharingMode::ExclusiveWrite,
            _ => SharingMode::NoSharing,
        };

        // Convert Vec<String> to IsolationLevel enum (use first or default)
        let isolation_level = if let Some(first) = safest_capability.isolation_requirements.first()
        {
            match first.as_str() {
                "ReadUncommitted" => IsolationLevel::ReadUncommitted,
                "ReadCommitted" => IsolationLevel::ReadCommitted,
                "RepeatableRead" => IsolationLevel::RepeatableRead,
                "Serializable" => IsolationLevel::Serializable,
                _ => IsolationLevel::None,
            }
        } else {
            IsolationLevel::None
        };

        // Convert Vec<String> to SynchronizationRequirements struct
        let synchronization_requirements = SynchronizationRequirements {
            synchronization_points: Vec::new(),
            lock_usage_patterns: Vec::new(),
            coordination_requirements: safest_capability.consistency_guarantees.clone(),
            synchronization_overhead: safest_capability.sharing_overhead,
            deadlock_prevention: Vec::new(),
            optimization_opportunities: Vec::new(),
            complexity_score: 0.0,
            performance_impact: safest_capability.sharing_overhead,
            alternative_strategies: Vec::new(),
            average_wait_time: Duration::from_secs(0),
            ordered_locking: false,
            timeout_based_locking: false,
            resource_ordering: Vec::new(),
            lock_free_alternatives: Vec::new(),
            custom_requirements: Vec::new(),
        };

        Ok(SharingRequirements {
            max_concurrent_shares: safest_capability.max_concurrent_readers.unwrap_or(1),
            sharing_mode,
            isolation_level,
            synchronization_requirements,
            performance_requirements: vec![
                format!("Max overhead: {}", safest_capability.sharing_overhead),
                format!(
                    "Throughput target: {}",
                    1.0 - safest_capability.sharing_overhead
                ),
                "Latency target: 100ms".to_string(),
            ],
            required_synchronization: vec![],
            exclusive_access_needed: false,
            concurrency_limit: safest_capability.max_concurrent_readers.unwrap_or(1),
        })
    }

    /// Generates sharing optimization recommendations
    async fn generate_sharing_optimizations(
        &self,
        capabilities: &[ResourceSharingCapabilities],
    ) -> Result<Vec<SharingOptimization>> {
        let mut optimizations = Vec::new();

        for capability in capabilities {
            if capability.performance_overhead > 0.2 {
                optimizations.push(SharingOptimization {
                    optimization_type: "ReduceOverhead".to_string(),
                    description: "Consider using more efficient sharing mechanisms".to_string(),
                    expected_improvement: capability.performance_overhead * 0.5,
                    implementation_effort: "Medium".to_string(),
                    recommendations: vec![
                        "Use lock-free data structures where possible".to_string(),
                        "Implement read-copy-update patterns".to_string(),
                        "Consider message passing instead of shared state".to_string(),
                    ],
                });
            }

            if capability.implementation_complexity > 0.7 {
                optimizations.push(SharingOptimization {
                    optimization_type: "SimplifyImplementation".to_string(),
                    description: "Simplify sharing implementation for better maintainability"
                        .to_string(),
                    expected_improvement: 0.3,
                    implementation_effort: "High".to_string(),
                    recommendations: vec![
                        "Break down complex sharing patterns".to_string(),
                        "Use higher-level synchronization primitives".to_string(),
                        "Implement gradual sharing capability expansion".to_string(),
                    ],
                });
            }
        }

        Ok(optimizations)
    }

    /// Predicts sharing performance based on capabilities
    async fn predict_sharing_performance(
        &self,
        capabilities: &[ResourceSharingCapabilities],
    ) -> Result<SharingPerformancePrediction> {
        let history = self.performance_history.lock();

        // Use historical data for prediction if available
        let base_throughput = history.get_average_throughput();

        let base_latency_ms = history.get_average_latency();
        let _base_latency = Duration::from_millis(base_latency_ms as u64);

        // Calculate predictions based on capabilities
        let avg_overhead = capabilities.iter().map(|c| c.performance_overhead).sum::<f64>()
            / capabilities.len() as f64;

        let predicted_throughput = base_throughput * (1.0 - avg_overhead);
        let latency_multiplier = 1.0 + avg_overhead;
        let predicted_latency_ms = base_latency_ms * latency_multiplier;
        let predicted_latency = Duration::from_millis(predicted_latency_ms as u64);

        let scalability_factor = capabilities
            .iter()
            .map(|c| match c.sharing_mode.as_str() {
                "ReadOnly" => 0.9,
                "ReadWrite" => 0.6,
                "ExclusiveWrite" => 0.3,
                "NoSharing" => 0.1,
                _ => 0.5, // Default for unknown modes
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.5);

        let bottlenecks = self.analyze_potential_bottlenecks(capabilities);
        let bottleneck_summary = format!("Found {} potential bottlenecks", bottlenecks.len());

        Ok(SharingPerformancePrediction {
            expected_throughput: predicted_throughput,
            analysis_duration: predicted_latency,
            scalability_factor,
            confidence: self.calculate_prediction_confidence(capabilities) as f64,
            bottleneck_analysis: bottleneck_summary,
        })
    }

    /// Calculates prediction confidence
    fn calculate_prediction_confidence(&self, capabilities: &[ResourceSharingCapabilities]) -> f32 {
        if capabilities.is_empty() {
            return 0.0;
        }

        let avg_confidence = capabilities
            .iter()
            .map(|c| self.calculate_strategy_confidence(c) as f64)
            .sum::<f64>() as f32
            / capabilities.len() as f32;

        let consistency_factor = self.calculate_capability_consistency(capabilities);

        avg_confidence * consistency_factor
    }

    /// Calculates capability consistency
    fn calculate_capability_consistency(
        &self,
        capabilities: &[ResourceSharingCapabilities],
    ) -> f32 {
        if capabilities.len() < 2 {
            return 1.0;
        }

        // Measure consistency in overhead predictions
        let overheads: Vec<f32> =
            capabilities.iter().map(|c| c.performance_overhead as f32).collect();
        let mean = overheads.iter().map(|&o| o as f64).sum::<f64>() as f32 / overheads.len() as f32;
        let variance = overheads.iter().map(|&o| (o - mean).powi(2) as f64).sum::<f64>() as f32
            / overheads.len() as f32;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 1.0 };

        (1.0 - coefficient_of_variation.min(1.0)).max(0.1)
    }

    /// Analyzes potential bottlenecks
    fn analyze_potential_bottlenecks(
        &self,
        capabilities: &[ResourceSharingCapabilities],
    ) -> Vec<BottleneckAnalysis> {
        let mut bottlenecks = Vec::new();

        for capability in capabilities {
            if capability.performance_overhead > 0.3 {
                bottlenecks.push(BottleneckAnalysis {
                    bottleneck_type: "SynchronizationOverhead".to_string(),
                    severity: if capability.performance_overhead > 0.5 {
                        BottleneckSeverity::High
                    } else {
                        BottleneckSeverity::Medium
                    },
                    affected_components: vec!["synchronization".to_string()],
                    impact_score: capability.performance_overhead,
                    resolution_suggestions: vec![
                        "Consider reducing synchronization frequency".to_string(),
                        "Use more efficient synchronization primitives".to_string(),
                    ],
                });
            }

            if capability.max_concurrent_readers.unwrap_or(0) < 2 {
                bottlenecks.push(BottleneckAnalysis {
                    bottleneck_type: "ConcurrencyLimitation".to_string(),
                    severity: BottleneckSeverity::Medium,
                    affected_components: vec!["concurrency".to_string()],
                    impact_score: 0.5,
                    resolution_suggestions: vec![
                        "Investigate increasing concurrent access limits".to_string(),
                        "Consider resource partitioning strategies".to_string(),
                    ],
                });
            }
        }

        bottlenecks
    }

    /// Caches analysis result
    async fn cache_result(&self, test_id: &str, result: &SharingAnalysisResult) -> Result<()> {
        let can_share = !result.sharing_capabilities.is_empty();
        let sharing_mode = if can_share { "parallel".to_string() } else { "none".to_string() };
        let cache_timestamp = Utc::now();

        let cached = CachedSharingCapability {
            can_share,
            sharing_mode,
            cache_timestamp,
            result: format!("SharingAnalysis(confidence={})", result.confidence),
            cached_at: cache_timestamp,
            confidence: result.confidence,
        };

        let mut cache = self.capability_cache.lock();
        cache.insert(test_id.to_string(), cached);

        // Cleanup old cache entries if needed
        let cache_limit = self.config.cache_size_limit;
        if cache.len() > cache_limit {
            // Remove oldest entries (simple LRU approximation)
            let oldest_key = cache.iter().min_by_key(|(_, v)| v.cached_at).map(|(k, _)| k.clone());

            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }

        Ok(())
    }

    /// Calculates overall sharing confidence
    fn calculate_overall_sharing_confidence(
        &self,
        capabilities: &[ResourceSharingCapabilities],
    ) -> f32 {
        if capabilities.is_empty() {
            return 0.0;
        }

        let individual_confidences: Vec<f32> =
            capabilities.iter().map(|c| self.calculate_strategy_confidence(c)).collect();

        let avg_confidence = individual_confidences.iter().map(|&x| x as f64).sum::<f64>() as f32
            / individual_confidences.len() as f32;
        let consistency_factor = self.calculate_capability_consistency(capabilities);

        avg_confidence * consistency_factor
    }
}
