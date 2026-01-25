//! Intelligent Configuration Optimizer for Mobile AI
//!
//! This module provides intelligent, adaptive configuration optimization that
//! automatically tunes mobile AI settings based on device capabilities,
//! usage patterns, and performance feedback.

use crate::device_info::{MobileDeviceInfo, PerformanceTier, ThermalState};
use crate::optimization::{MetricType, PerformanceAnalyticsEngine, PerformanceInsights};
use crate::{MemoryOptimization, MobileBackend, MobileConfig, MobileQuantizationScheme};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::Duration;
use trustformers_core::error::Result;

/// Configuration optimization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Conservative optimization (stable, minimal risk)
    Conservative,
    /// Balanced optimization (performance vs stability)
    Balanced,
    /// Aggressive optimization (maximum performance)
    Aggressive,
    /// Adaptive optimization (learns from usage patterns)
    Adaptive,
    /// Custom strategy with manual parameters
    Custom,
}

/// Configuration optimization goals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationGoals {
    /// Target inference latency (milliseconds)
    pub target_latency_ms: Option<f32>,
    /// Maximum memory usage (bytes)
    pub max_memory_bytes: Option<usize>,
    /// Target power consumption (watts)
    pub target_power_watts: Option<f32>,
    /// Target thermal temperature (Celsius)
    pub max_temperature_celsius: Option<f32>,
    /// Minimum accuracy threshold (0.0-1.0)
    pub min_accuracy: Option<f32>,
    /// Optimization priorities
    pub priorities: OptimizationPriorities,
}

/// Optimization priorities (weights sum to 1.0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationPriorities {
    /// Performance priority (0.0-1.0)
    pub performance: f32,
    /// Memory efficiency priority (0.0-1.0)
    pub memory: f32,
    /// Power efficiency priority (0.0-1.0)
    pub power: f32,
    /// Thermal management priority (0.0-1.0)
    pub thermal: f32,
    /// Accuracy preservation priority (0.0-1.0)
    pub accuracy: f32,
}

impl Default for OptimizationPriorities {
    fn default() -> Self {
        Self {
            performance: 0.3,
            memory: 0.2,
            power: 0.2,
            thermal: 0.1,
            accuracy: 0.2,
        }
    }
}

/// Device capability assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilityAssessment {
    /// Overall device score (0.0-1.0)
    pub overall_score: f32,
    /// Performance tier
    pub performance_tier: PerformanceTier,
    /// Available memory (bytes)
    pub available_memory: usize,
    /// CPU core count and capabilities
    pub cpu_cores: usize,
    /// GPU availability and capabilities
    pub gpu_available: bool,
    /// Neural processing unit availability
    pub npu_available: bool,
    /// Supported quantization schemes
    pub supported_quantization: Vec<MobileQuantizationScheme>,
    /// Thermal headroom (0.0-1.0)
    pub thermal_headroom: f32,
    /// Battery level (0.0-1.0)
    pub battery_level: f32,
    /// Power constraints
    pub power_constrained: bool,
}

/// Configuration recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationRecommendation {
    /// Recommended configuration
    pub config: MobileConfig,
    /// Expected performance improvements
    pub expected_improvements: HashMap<String, f32>,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Risk assessment (0.0-1.0)
    pub risk_score: f32,
    /// Reasoning for recommendations
    pub reasoning: Vec<String>,
    /// A/B testing suggestion
    pub ab_test_candidate: bool,
}

/// Learning history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LearningEntry {
    /// Configuration used
    config: MobileConfig,
    /// Performance metrics achieved
    metrics: HashMap<MetricType, f32>,
    /// Device state at time of measurement
    device_state: DeviceState,
    /// Timestamp
    timestamp: u64,
    /// Success score (0.0-1.0)
    success_score: f32,
}

/// Device state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceState {
    /// Memory usage percentage
    memory_usage_percent: f32,
    /// CPU utilization percentage
    cpu_utilization_percent: f32,
    /// Temperature (Celsius)
    temperature_celsius: f32,
    /// Battery level percentage
    battery_level_percent: f32,
    /// Thermal state
    thermal_state: ThermalState,
}

/// Intelligent configuration optimizer
pub struct IntelligentConfigOptimizer {
    /// Current optimization strategy
    strategy: OptimizationStrategy,
    /// Optimization goals
    goals: OptimizationGoals,
    /// Device capability assessment
    device_capabilities: DeviceCapabilityAssessment,
    /// Learning history
    learning_history: Arc<Mutex<VecDeque<LearningEntry>>>,
    /// Performance analytics engine
    analytics_engine: Option<Arc<PerformanceAnalyticsEngine>>,
    /// Current best configuration
    best_config: Arc<Mutex<Option<MobileConfig>>>,
    /// Optimization iterations performed
    iterations: AtomicUsize,
    /// A/B testing results
    ab_test_results: Arc<Mutex<HashMap<String, ABTestResult>>>,
}

/// A/B testing result
#[derive(Debug, Clone)]
struct ABTestResult {
    config_a: MobileConfig,
    config_b: MobileConfig,
    performance_a: HashMap<MetricType, f32>,
    performance_b: HashMap<MetricType, f32>,
    winner: Option<String>, // "A" or "B"
    confidence: f32,
}

impl IntelligentConfigOptimizer {
    /// Create a new intelligent configuration optimizer
    pub fn new(
        strategy: OptimizationStrategy,
        goals: OptimizationGoals,
        device_info: &MobileDeviceInfo,
    ) -> Result<Self> {
        let device_capabilities = Self::assess_device_capabilities(device_info)?;

        Ok(Self {
            strategy,
            goals,
            device_capabilities,
            learning_history: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            analytics_engine: None,
            best_config: Arc::new(Mutex::new(None)),
            iterations: AtomicUsize::new(0),
            ab_test_results: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Set the performance analytics engine for feedback
    pub fn set_analytics_engine(&mut self, engine: Arc<PerformanceAnalyticsEngine>) {
        self.analytics_engine = Some(engine);
    }

    /// Generate optimized configuration recommendation
    pub fn recommend_configuration(&self) -> Result<ConfigurationRecommendation> {
        match self.strategy {
            OptimizationStrategy::Conservative => self.generate_conservative_config(),
            OptimizationStrategy::Balanced => self.generate_balanced_config(),
            OptimizationStrategy::Aggressive => self.generate_aggressive_config(),
            OptimizationStrategy::Adaptive => self.generate_adaptive_config(),
            OptimizationStrategy::Custom => self.generate_custom_config(),
        }
    }

    /// Learn from performance feedback
    pub fn learn_from_feedback(
        &self,
        config: &MobileConfig,
        performance_metrics: HashMap<MetricType, f32>,
        device_state: DeviceState,
    ) -> Result<()> {
        let success_score = self.calculate_success_score(&performance_metrics, &device_state);

        let entry = LearningEntry {
            config: config.clone(),
            metrics: performance_metrics,
            device_state,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            success_score,
        };

        // Add to learning history
        if let Ok(mut history) = self.learning_history.lock() {
            history.push_back(entry.clone());
            if history.len() > 1000 {
                history.pop_front();
            }
        }

        // Update best configuration if this is better
        if success_score > 0.8 {
            if let Ok(mut best_config) = self.best_config.lock() {
                let should_update = best_config
                    .as_ref()
                    .map(|current| {
                        // Check if new config is significantly better
                        self.compare_configurations(config, current, &entry.metrics)
                    })
                    .unwrap_or(true);

                if should_update {
                    *best_config = Some(config.clone());
                }
            }
        }

        Ok(())
    }

    /// Perform automatic A/B testing
    pub fn perform_ab_test(
        &self,
        config_a: MobileConfig,
        config_b: MobileConfig,
        test_duration: Duration,
    ) -> Result<String> {
        let test_id = format!(
            "ab_test_{}",
            self.iterations.fetch_add(1, Ordering::Relaxed)
        );

        // This would normally run actual A/B testing
        // For now, simulate based on configuration analysis
        let score_a = self.estimate_config_performance(&config_a)?;
        let score_b = self.estimate_config_performance(&config_b)?;

        let winner = if score_a > score_b { "A" } else { "B" };
        let confidence = ((score_a - score_b).abs() / score_a.max(score_b)).min(1.0);

        let result = ABTestResult {
            config_a,
            config_b,
            performance_a: HashMap::new(), // Would be filled with real metrics
            performance_b: HashMap::new(), // Would be filled with real metrics
            winner: Some(winner.to_string()),
            confidence,
        };

        if let Ok(mut results) = self.ab_test_results.lock() {
            results.insert(test_id.clone(), result);
        }

        Ok(test_id)
    }

    /// Get A/B test results
    pub fn get_ab_test_result(&self, test_id: &str) -> Option<String> {
        self.ab_test_results.lock().ok()?.get(test_id)?.winner.clone()
    }

    /// Auto-tune configuration based on current performance
    pub fn auto_tune(&self) -> Result<MobileConfig> {
        // Get current performance insights
        let insights = if let Some(ref analytics) = self.analytics_engine {
            analytics.get_cached_insights()
        } else {
            None
        };

        let mut base_config =
            self.best_config.lock().expect("Operation failed").clone().unwrap_or_default();

        // Apply tuning based on insights
        if let Some(insights) = insights {
            base_config = self.apply_insights_to_config(base_config, &insights)?;
        }

        // Apply learning-based optimizations
        base_config = self.apply_learned_optimizations(base_config)?;

        Ok(base_config)
    }

    /// Generate conservative configuration
    fn generate_conservative_config(&self) -> Result<ConfigurationRecommendation> {
        let mut config = MobileConfig::default();

        // Conservative settings
        config.memory_optimization = MemoryOptimization::Maximum;
        config.max_memory_mb = (self.device_capabilities.available_memory / (1024 * 1024)).min(256);
        config.use_fp16 = true;
        config.enable_batching = false;
        config.max_batch_size = 1;

        // Safe quantization
        config.quantization = if self
            .device_capabilities
            .supported_quantization
            .contains(&MobileQuantizationScheme::Int8)
        {
            Some(crate::MobileQuantizationConfig {
                scheme: MobileQuantizationScheme::Int8,
                dynamic: true,
                per_channel: false,
            })
        } else {
            None
        };

        Ok(ConfigurationRecommendation {
            config,
            expected_improvements: HashMap::from([
                ("stability".to_string(), 0.9),
                ("memory_efficiency".to_string(), 0.8),
            ]),
            confidence: 0.9,
            risk_score: 0.1,
            reasoning: vec![
                "Conservative settings for maximum stability".to_string(),
                "Minimal memory usage with safe optimizations".to_string(),
            ],
            ab_test_candidate: false,
        })
    }

    /// Generate balanced configuration
    fn generate_balanced_config(&self) -> Result<ConfigurationRecommendation> {
        let mut config = MobileConfig::default();

        // Balanced settings based on device capabilities
        config.memory_optimization = MemoryOptimization::Balanced;
        config.max_memory_mb = (self.device_capabilities.available_memory / (1024 * 1024)).min(512);
        config.use_fp16 = true;

        // Enable batching for higher-tier devices
        if self.device_capabilities.performance_tier >= PerformanceTier::Medium {
            config.enable_batching = true;
            config.max_batch_size = match self.device_capabilities.performance_tier {
                PerformanceTier::High | PerformanceTier::VeryHigh => 4,
                PerformanceTier::Medium => 2,
                _ => 1,
            };
        }

        // Optimize quantization based on capabilities
        config.quantization = self.select_optimal_quantization();

        // Backend selection
        config.backend = self.select_optimal_backend();

        Ok(ConfigurationRecommendation {
            config,
            expected_improvements: HashMap::from([
                ("performance".to_string(), 0.6),
                ("memory_efficiency".to_string(), 0.6),
                ("power_efficiency".to_string(), 0.5),
            ]),
            confidence: 0.8,
            risk_score: 0.3,
            reasoning: vec![
                "Balanced configuration for good overall performance".to_string(),
                "Optimized based on device capabilities".to_string(),
            ],
            ab_test_candidate: true,
        })
    }

    /// Generate aggressive configuration
    fn generate_aggressive_config(&self) -> Result<ConfigurationRecommendation> {
        let mut config = MobileConfig::default();

        // Aggressive settings for maximum performance
        config.memory_optimization = MemoryOptimization::Minimal;
        config.max_memory_mb =
            (self.device_capabilities.available_memory / (1024 * 1024)).min(1024);
        config.use_fp16 = true;
        config.enable_batching = true;
        config.max_batch_size = match self.device_capabilities.performance_tier {
            PerformanceTier::VeryHigh => 8,
            PerformanceTier::High => 6,
            PerformanceTier::Medium => 4,
            _ => 2,
        };

        // Aggressive quantization for performance
        config.quantization = if self
            .device_capabilities
            .supported_quantization
            .contains(&MobileQuantizationScheme::Int4)
        {
            Some(crate::MobileQuantizationConfig {
                scheme: MobileQuantizationScheme::Int4,
                dynamic: false,
                per_channel: true,
            })
        } else {
            self.select_optimal_quantization()
        };

        // Use best available backend
        config.backend = self.select_optimal_backend();

        Ok(ConfigurationRecommendation {
            config,
            expected_improvements: HashMap::from([
                ("performance".to_string(), 0.9),
                ("throughput".to_string(), 0.8),
            ]),
            confidence: 0.7,
            risk_score: 0.6,
            reasoning: vec![
                "Aggressive optimization for maximum performance".to_string(),
                "Higher memory usage but better throughput".to_string(),
                "May require thermal monitoring".to_string(),
            ],
            ab_test_candidate: true,
        })
    }

    /// Generate adaptive configuration based on learning
    fn generate_adaptive_config(&self) -> Result<ConfigurationRecommendation> {
        // Start with balanced config as base
        let mut recommendation = self.generate_balanced_config()?;

        // Apply learning-based optimizations
        if let Ok(history) = self.learning_history.lock() {
            if history.len() >= 10 {
                // Find patterns in successful configurations
                let successful_configs: Vec<_> =
                    history.iter().filter(|entry| entry.success_score > 0.7).collect();

                if !successful_configs.is_empty() {
                    // Average successful parameters
                    let avg_memory_mb = successful_configs
                        .iter()
                        .map(|entry| entry.config.max_memory_mb)
                        .sum::<usize>()
                        / successful_configs.len();

                    let avg_batch_size = successful_configs
                        .iter()
                        .map(|entry| entry.config.max_batch_size)
                        .sum::<usize>()
                        / successful_configs.len();

                    // Apply learned parameters
                    recommendation.config.max_memory_mb = avg_memory_mb;
                    recommendation.config.max_batch_size = avg_batch_size;

                    // Increase confidence based on learning data
                    recommendation.confidence += 0.1;
                    recommendation
                        .reasoning
                        .push("Optimized based on learned performance patterns".to_string());
                }
            }
        }

        Ok(recommendation)
    }

    /// Generate custom configuration
    fn generate_custom_config(&self) -> Result<ConfigurationRecommendation> {
        // For custom strategy, use goals to guide configuration
        let mut config = MobileConfig::default();

        // Memory optimization based on goals
        if let Some(max_memory) = self.goals.max_memory_bytes {
            config.max_memory_mb = max_memory / (1024 * 1024);
            config.memory_optimization = if max_memory < 256 * 1024 * 1024 {
                MemoryOptimization::Maximum
            } else if max_memory < 512 * 1024 * 1024 {
                MemoryOptimization::Balanced
            } else {
                MemoryOptimization::Minimal
            };
        }

        // Performance optimization based on latency goals
        if let Some(target_latency) = self.goals.target_latency_ms {
            if target_latency < 50.0 {
                // Aggressive optimization for low latency
                config.quantization = Some(crate::MobileQuantizationConfig {
                    scheme: MobileQuantizationScheme::Int4,
                    dynamic: false,
                    per_channel: true,
                });
                config.enable_batching = true;
                config.max_batch_size = 4;
            } else if target_latency < 100.0 {
                // Balanced optimization
                config.quantization = self.select_optimal_quantization();
                config.enable_batching = true;
                config.max_batch_size = 2;
            }
        }

        Ok(ConfigurationRecommendation {
            config,
            expected_improvements: HashMap::from([("goal_alignment".to_string(), 0.8)]),
            confidence: 0.6,
            risk_score: 0.4,
            reasoning: vec!["Custom configuration based on specified goals".to_string()],
            ab_test_candidate: true,
        })
    }

    /// Assess device capabilities
    fn assess_device_capabilities(
        device_info: &MobileDeviceInfo,
    ) -> Result<DeviceCapabilityAssessment> {
        // Calculate overall device score
        let performance_score = match device_info.performance_scores.tier {
            PerformanceTier::VeryLow => 0.2,
            PerformanceTier::Low => 0.4,
            PerformanceTier::Budget => 0.5,
            PerformanceTier::Medium => 0.6,
            PerformanceTier::Mid => 0.7,
            PerformanceTier::High => 0.8,
            PerformanceTier::VeryHigh => 1.0,
            PerformanceTier::Flagship => 1.0,
        };

        let memory_score = (device_info.memory_info.total_memory as f32
            / (8.0 * 1024.0 * 1024.0 * 1024.0))
            .min(1.0); // Normalize to 8GB
        let thermal_score = match device_info.thermal_info.state {
            ThermalState::Nominal => 1.0,
            ThermalState::Fair => 0.8,
            ThermalState::Serious => 0.6,
            ThermalState::Critical => 0.3,
            ThermalState::Emergency => 0.1,
            ThermalState::Shutdown => 0.0,
        };

        let overall_score = (performance_score + memory_score + thermal_score) / 3.0;

        // Determine supported quantization schemes
        let mut supported_quantization = vec![
            MobileQuantizationScheme::FP16,
            MobileQuantizationScheme::Int8,
        ];
        if device_info.performance_scores.tier >= PerformanceTier::Medium {
            supported_quantization.push(MobileQuantizationScheme::Int4);
        }
        if device_info.performance_scores.tier >= PerformanceTier::High {
            supported_quantization.push(MobileQuantizationScheme::Dynamic);
        }

        Ok(DeviceCapabilityAssessment {
            overall_score,
            performance_tier: device_info.performance_scores.tier,
            available_memory: device_info.memory_info.available_memory,
            cpu_cores: device_info.cpu_info.core_count,
            gpu_available: device_info.gpu_info.is_some(),
            npu_available: device_info.npu_info.is_some(),
            supported_quantization,
            thermal_headroom: thermal_score,
            battery_level: device_info.power_info.battery_level.unwrap_or(50) as f32 / 100.0,
            power_constrained: !device_info.power_info.is_charging
                && device_info.power_info.battery_level.unwrap_or(50) < 30,
        })
    }

    /// Select optimal quantization scheme
    fn select_optimal_quantization(&self) -> Option<crate::MobileQuantizationConfig> {
        if self.goals.priorities.performance > 0.6
            && self
                .device_capabilities
                .supported_quantization
                .contains(&MobileQuantizationScheme::Int4)
        {
            Some(crate::MobileQuantizationConfig {
                scheme: MobileQuantizationScheme::Int4,
                dynamic: false,
                per_channel: true,
            })
        } else if self
            .device_capabilities
            .supported_quantization
            .contains(&MobileQuantizationScheme::Int8)
        {
            Some(crate::MobileQuantizationConfig {
                scheme: MobileQuantizationScheme::Int8,
                dynamic: true,
                per_channel: false,
            })
        } else {
            Some(crate::MobileQuantizationConfig {
                scheme: MobileQuantizationScheme::FP16,
                dynamic: false,
                per_channel: false,
            })
        }
    }

    /// Select optimal backend
    fn select_optimal_backend(&self) -> MobileBackend {
        if self.device_capabilities.npu_available && self.goals.priorities.performance > 0.7 {
            MobileBackend::Custom // NPU backend
        } else if self.device_capabilities.gpu_available && self.goals.priorities.performance > 0.5
        {
            MobileBackend::GPU
        } else if cfg!(target_os = "ios") {
            MobileBackend::CoreML
        } else if cfg!(target_os = "android") {
            MobileBackend::NNAPI
        } else {
            MobileBackend::CPU
        }
    }

    /// Calculate success score for a configuration
    fn calculate_success_score(
        &self,
        metrics: &HashMap<MetricType, f32>,
        device_state: &DeviceState,
    ) -> f32 {
        let mut score = 1.0f32;
        let priorities = &self.goals.priorities;

        // Performance score
        if let Some(&latency) = metrics.get(&MetricType::InferenceLatency) {
            let performance_score = if let Some(target) = self.goals.target_latency_ms {
                (target / latency.max(target)).min(1.0)
            } else {
                (100.0 / latency.max(100.0)).min(1.0) // Assume 100ms as baseline
            };
            score *= 1.0 - priorities.performance * (1.0 - performance_score);
        }

        // Memory score
        if let Some(&memory_usage) = metrics.get(&MetricType::MemoryUsage) {
            let memory_score = if let Some(max_memory) = self.goals.max_memory_bytes {
                ((max_memory as f32 - memory_usage) / max_memory as f32).max(0.0)
            } else {
                (1.0 - device_state.memory_usage_percent / 100.0).max(0.0)
            };
            score *= 1.0 - priorities.memory * (1.0 - memory_score);
        }

        // Power score
        if let Some(&power) = metrics.get(&MetricType::PowerConsumption) {
            let power_score = if let Some(target) = self.goals.target_power_watts {
                ((target - power) / target).max(0.0)
            } else {
                (1.0 - power / 10.0).max(0.0) // Assume 10W as high consumption
            };
            score *= 1.0 - priorities.power * (1.0 - power_score);
        }

        // Thermal score
        let thermal_score = (1.0 - device_state.temperature_celsius / 85.0).max(0.0); // 85°C as critical
        score *= 1.0 - priorities.thermal * (1.0 - thermal_score);

        score.clamp(0.0, 1.0)
    }

    /// Compare two configurations
    fn compare_configurations(
        &self,
        config_a: &MobileConfig,
        config_b: &MobileConfig,
        metrics: &HashMap<MetricType, f32>,
    ) -> bool {
        // Simple comparison based on expected performance
        // In a real implementation, this would be more sophisticated
        let score_a = self.estimate_config_performance(config_a).unwrap_or(0.0);
        let score_b = self.estimate_config_performance(config_b).unwrap_or(0.0);

        score_a > score_b
    }

    /// Estimate configuration performance
    fn estimate_config_performance(&self, config: &MobileConfig) -> Result<f32> {
        let mut score = 0.5f32; // Base score

        // Quantization impact
        if let Some(ref quant) = config.quantization {
            score += match quant.scheme {
                MobileQuantizationScheme::Int4 => 0.3,
                MobileQuantizationScheme::Int8 => 0.2,
                MobileQuantizationScheme::FP16 => 0.1,
                MobileQuantizationScheme::Dynamic => 0.15,
            };
        }

        // Memory optimization impact
        score += match config.memory_optimization {
            MemoryOptimization::Maximum => 0.1,
            MemoryOptimization::Balanced => 0.05,
            MemoryOptimization::Minimal => 0.02,
        };

        // Backend impact
        score += match config.backend {
            MobileBackend::Custom => 0.2, // Assume NPU
            MobileBackend::GPU => 0.15,
            MobileBackend::CoreML => 0.1,
            MobileBackend::NNAPI => 0.1,
            MobileBackend::Metal => 0.15,
            MobileBackend::Vulkan => 0.14,
            MobileBackend::OpenCL => 0.13,
            MobileBackend::CPU => 0.0,
        };

        // Batching impact
        if config.enable_batching {
            score += 0.05 * (config.max_batch_size as f32 / 8.0).min(1.0);
        }

        Ok(score.clamp(0.0, 1.0))
    }

    /// Apply insights to configuration
    fn apply_insights_to_config(
        &self,
        mut config: MobileConfig,
        insights: &PerformanceInsights,
    ) -> Result<MobileConfig> {
        // Apply recommendations from insights
        for recommendation in &insights.recommendations {
            if recommendation.related_metrics.contains(&MetricType::MemoryUsage)
                && recommendation.title.contains("Memory")
            {
                config.memory_optimization = MemoryOptimization::Maximum;
                config.max_memory_mb = (config.max_memory_mb as f32 * 0.8) as usize;
            }

            if recommendation.related_metrics.contains(&MetricType::Temperature)
                && recommendation.title.contains("Thermal")
            {
                // Reduce performance to manage thermals
                config.max_batch_size = (config.max_batch_size / 2).max(1);
                config.num_threads = (config.num_threads / 2).max(1);
            }

            if recommendation.related_metrics.contains(&MetricType::InferenceLatency)
                && recommendation.title.contains("Performance")
            {
                // Enable more aggressive optimizations
                if let Some(ref mut quant) = config.quantization {
                    if quant.scheme == MobileQuantizationScheme::Int8 {
                        quant.scheme = MobileQuantizationScheme::Int4;
                    }
                }
            }
        }

        Ok(config)
    }

    /// Apply learned optimizations
    fn apply_learned_optimizations(&self, mut config: MobileConfig) -> Result<MobileConfig> {
        if let Ok(history) = self.learning_history.lock() {
            if history.len() >= 5 {
                // Find the best performing configurations
                let mut best_configs: Vec<_> =
                    history.iter().filter(|entry| entry.success_score > 0.8).collect();

                best_configs.sort_by(|a, b| {
                    b.success_score.partial_cmp(&a.success_score).expect("Operation failed")
                });

                if !best_configs.is_empty() {
                    let best_config = &best_configs[0].config;

                    // Apply learned parameters with some interpolation
                    config.max_memory_mb = (config.max_memory_mb + best_config.max_memory_mb) / 2;
                    config.max_batch_size =
                        (config.max_batch_size + best_config.max_batch_size) / 2;

                    // Copy successful optimization settings
                    config.memory_optimization = best_config.memory_optimization;
                    if best_config.quantization.is_some() {
                        config.quantization = best_config.quantization.clone();
                    }
                }
            }
        }

        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_info::*;
    use crate::MobilePlatform;

    fn create_test_device_info() -> MobileDeviceInfo {
        MobileDeviceInfo {
            basic_info: BasicDeviceInfo {
                hardware_id: "test_device".to_string(),
                model: "Test Device".to_string(),
                os_version: "1.0".to_string(),
                manufacturer: "Test".to_string(),
                platform: MobilePlatform::Generic,
                device_generation: Some(2023),
            },
            platform: MobilePlatform::Generic,
            memory_info: MemoryInfo {
                total_mb: 4096,
                available_mb: 2048,
                total_memory: 4096,
                available_memory: 2048,
                bandwidth_mbps: Some(25600),
                memory_type: "LPDDR4".to_string(),
                frequency_mhz: Some(1600),
                is_low_memory_device: false,
            },
            cpu_info: CpuInfo {
                core_count: 8,
                total_cores: 8,
                performance_cores: 6,
                efficiency_cores: 2,
                max_frequency_mhz: Some(2400),
                l1_cache_kb: Some(64),
                l2_cache_kb: Some(256),
                l3_cache_kb: Some(8192),
                architecture: "arm64".to_string(),
                features: vec!["neon".to_string(), "fp16".to_string()],
                simd_support: SimdSupport::Advanced,
            },
            gpu_info: Some(GpuInfo {
                vendor: "Test".to_string(),
                model: "Test GPU".to_string(),
                driver_version: "1.0".to_string(),
                memory_mb: Some(1024),
                compute_units: Some(16),
                supported_apis: vec![GpuApi::Vulkan, GpuApi::OpenCL],
                performance_tier: GpuPerformanceTier::High,
            }),
            npu_info: None,
            thermal_info: ThermalInfo {
                current_state: ThermalState::Nominal,
                state: ThermalState::Nominal,
                throttling_supported: true,
                temperature_sensors: Vec::new(),
                thermal_zones: Vec::new(),
            },
            power_info: PowerInfo {
                battery_capacity_mah: Some(3000),
                battery_level_percent: Some(75),
                battery_level: Some(75),
                battery_health_percent: Some(100),
                charging_status: ChargingStatus::NotCharging,
                is_charging: false,
                power_save_mode: false,
                low_power_mode_available: true,
            },
            performance_scores: PerformanceScores {
                cpu_single_core: Some(1000),
                cpu_multi_core: Some(2800),
                gpu_score: Some(1500),
                memory_score: Some(800),
                overall_tier: PerformanceTier::High,
                tier: PerformanceTier::High,
            },
            available_backends: vec![MobileBackend::CPU, MobileBackend::GPU],
        }
    }

    #[test]
    fn test_optimizer_creation() {
        let device_info = create_test_device_info();
        let goals = OptimizationGoals {
            target_latency_ms: Some(50.0),
            max_memory_bytes: Some(512 * 1024 * 1024),
            target_power_watts: Some(5.0),
            max_temperature_celsius: Some(70.0),
            min_accuracy: Some(0.95),
            priorities: OptimizationPriorities::default(),
        };

        let optimizer =
            IntelligentConfigOptimizer::new(OptimizationStrategy::Balanced, goals, &device_info);

        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_configuration_recommendation() {
        let device_info = create_test_device_info();
        let goals = OptimizationGoals {
            target_latency_ms: Some(50.0),
            max_memory_bytes: Some(512 * 1024 * 1024),
            target_power_watts: None,
            max_temperature_celsius: None,
            min_accuracy: None,
            priorities: OptimizationPriorities::default(),
        };

        let optimizer =
            IntelligentConfigOptimizer::new(OptimizationStrategy::Balanced, goals, &device_info)
                .expect("Operation failed");

        let recommendation = optimizer.recommend_configuration();
        assert!(recommendation.is_ok());

        let recommendation = recommendation.expect("Operation failed");
        assert!(recommendation.confidence > 0.0);
        assert!(recommendation.risk_score >= 0.0);
        assert!(!recommendation.reasoning.is_empty());
    }

    #[test]
    fn test_learning_from_feedback() {
        let device_info = create_test_device_info();
        let optimizer = IntelligentConfigOptimizer::new(
            OptimizationStrategy::Adaptive,
            OptimizationGoals {
                target_latency_ms: None,
                max_memory_bytes: None,
                target_power_watts: None,
                max_temperature_celsius: None,
                min_accuracy: None,
                priorities: OptimizationPriorities::default(),
            },
            &device_info,
        )
        .expect("Operation failed");

        let config = MobileConfig::default();
        let metrics = HashMap::from([
            (MetricType::InferenceLatency, 45.0),
            (MetricType::MemoryUsage, 200.0 * 1024.0 * 1024.0),
            (MetricType::PowerConsumption, 3.5),
        ]);

        let device_state = DeviceState {
            memory_usage_percent: 60.0,
            cpu_utilization_percent: 75.0,
            temperature_celsius: 40.0,
            battery_level_percent: 80.0,
            thermal_state: ThermalState::Nominal,
        };

        let result = optimizer.learn_from_feedback(&config, metrics, device_state);
        assert!(result.is_ok());
    }

    #[test]
    fn test_auto_tune() {
        let device_info = create_test_device_info();
        let optimizer = IntelligentConfigOptimizer::new(
            OptimizationStrategy::Adaptive,
            OptimizationGoals {
                target_latency_ms: Some(50.0),
                max_memory_bytes: None,
                target_power_watts: None,
                max_temperature_celsius: None,
                min_accuracy: None,
                priorities: OptimizationPriorities::default(),
            },
            &device_info,
        )
        .expect("Operation failed");

        let tuned_config = optimizer.auto_tune();
        assert!(tuned_config.is_ok());
    }
}
