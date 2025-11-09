//! Leaderboard system for tracking and comparing benchmark results

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

pub mod client;
pub mod query;
pub mod ranking;
pub mod stats;
pub mod storage;
pub mod submission;

pub use client::{ClientConfig, LeaderboardClient};
pub use query::{FilterType, LeaderboardFilter, LeaderboardQuery};
pub use ranking::{
    DefaultRankingAlgorithm, LeaderboardRanking, RankingAlgorithm, RankingCriteria, RankingMetric,
    SortOrder,
};
pub use stats::{LeaderboardStats, PerformanceTrend, TrendAnalysis};
pub use storage::{FileStorage, LeaderboardStorage, RemoteStorage};
pub use submission::{
    DefaultValidator, LeaderboardSubmission, SubmissionValidator, ValidationResult,
};

/// Leaderboard entry representing a benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    /// Unique identifier
    pub id: Uuid,
    /// Submission timestamp
    pub timestamp: DateTime<Utc>,
    /// Model name
    pub model_name: String,
    /// Model version
    pub model_version: String,
    /// Benchmark name
    pub benchmark_name: String,
    /// Category
    pub category: LeaderboardCategory,
    /// Hardware configuration
    pub hardware: HardwareInfo,
    /// Software configuration
    pub software: SoftwareInfo,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Validation status
    pub validated: bool,
    /// Submitter information
    pub submitter: SubmitterInfo,
    /// Tags
    pub tags: Vec<String>,
}

/// Leaderboard category
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum LeaderboardCategory {
    /// Inference benchmarks
    Inference,
    /// Training benchmarks
    Training,
    /// Fine-tuning benchmarks
    FineTuning,
    /// Memory efficiency
    Memory,
    /// Energy efficiency
    Energy,
    /// Accuracy benchmarks
    Accuracy,
    /// Custom category
    Custom(String),
}

impl std::fmt::Display for LeaderboardCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LeaderboardCategory::Inference => write!(f, "Inference"),
            LeaderboardCategory::Training => write!(f, "Training"),
            LeaderboardCategory::FineTuning => write!(f, "Fine-tuning"),
            LeaderboardCategory::Memory => write!(f, "Memory"),
            LeaderboardCategory::Energy => write!(f, "Energy"),
            LeaderboardCategory::Accuracy => write!(f, "Accuracy"),
            LeaderboardCategory::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Hardware information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    /// CPU model
    pub cpu: String,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// GPU model (if applicable)
    pub gpu: Option<String>,
    /// Number of GPUs
    pub gpu_count: Option<usize>,
    /// Total memory in GB
    pub memory_gb: f64,
    /// Accelerator type
    pub accelerator: Option<AcceleratorType>,
    /// Hardware platform
    pub platform: String,
}

/// Accelerator types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AcceleratorType {
    /// NVIDIA GPU
    CUDA,
    /// AMD GPU
    ROCm,
    /// Apple Silicon
    Metal,
    /// Intel GPU
    OneAPI,
    /// TPU
    TPU,
    /// Custom accelerator
    Custom(String),
}

/// Software information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftwareInfo {
    /// Framework version
    pub framework_version: String,
    /// Rust version
    pub rust_version: String,
    /// OS name and version
    pub os: String,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Precision
    pub precision: Precision,
    /// Quantization
    pub quantization: Option<String>,
    /// Compiler flags
    pub compiler_flags: Vec<String>,
}

/// Optimization level
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimizations
    None,
    /// Basic optimizations
    O1,
    /// Standard optimizations
    O2,
    /// Aggressive optimizations
    O3,
    /// Custom optimizations
    Custom(String),
}

/// Precision levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Precision {
    /// Full precision (FP32)
    FP32,
    /// Half precision (FP16)
    FP16,
    /// Brain floating point (BF16)
    BF16,
    /// 8-bit integer
    INT8,
    /// 4-bit integer
    INT4,
    /// Mixed precision
    Mixed,
    /// Custom precision
    Custom(String),
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average latency in milliseconds
    pub latency_ms: f64,
    /// Latency percentiles
    pub latency_percentiles: LatencyPercentiles,
    /// Throughput (items/second)
    pub throughput: Option<f64>,
    /// Tokens per second (for LLMs)
    pub tokens_per_second: Option<f64>,
    /// Memory usage in MB
    pub memory_mb: Option<f64>,
    /// Peak memory usage in MB
    pub peak_memory_mb: Option<f64>,
    /// GPU utilization percentage
    pub gpu_utilization: Option<f64>,
    /// Model accuracy
    pub accuracy: Option<f64>,
    /// Energy consumption in watts
    pub energy_watts: Option<f64>,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Latency percentiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub p999: f64,
}

/// Submitter information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitterInfo {
    /// Name or handle
    pub name: String,
    /// Organization (optional)
    pub organization: Option<String>,
    /// Email (optional)
    pub email: Option<String>,
    /// GitHub username (optional)
    pub github: Option<String>,
}

/// Leaderboard manager
pub struct LeaderboardManager {
    storage: Arc<dyn LeaderboardStorage>,
    validator: Arc<dyn SubmissionValidator>,
    ranking: Arc<dyn RankingAlgorithm>,
}

impl LeaderboardManager {
    /// Create new leaderboard manager
    pub fn new(
        storage: Arc<dyn LeaderboardStorage>,
        validator: Arc<dyn SubmissionValidator>,
        ranking: Arc<dyn RankingAlgorithm>,
    ) -> Self {
        Self {
            storage,
            validator,
            ranking,
        }
    }

    /// Submit a new entry
    pub async fn submit(&self, submission: LeaderboardSubmission) -> Result<Uuid> {
        // Validate submission
        let validation = self.validator.validate(&submission).await?;
        if !validation.is_valid {
            anyhow::bail!("Invalid submission: {:?}", validation.errors);
        }

        // Convert to entry
        let entry = LeaderboardEntry {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            model_name: submission.model_name,
            model_version: submission.model_version,
            benchmark_name: submission.benchmark_name,
            category: submission.category,
            hardware: submission.hardware,
            software: submission.software,
            metrics: submission.metrics,
            metadata: submission.metadata,
            validated: validation.is_valid,
            submitter: submission.submitter,
            tags: submission.tags,
        };

        // Store entry
        self.storage.store(&entry).await?;

        Ok(entry.id)
    }

    /// Query leaderboard
    pub async fn query(&self, query: LeaderboardQuery) -> Result<Vec<LeaderboardEntry>> {
        let entries = self.storage.query(&query).await?;

        // Apply ranking
        let ranked = self.ranking.rank(entries, &query.ranking_criteria)?;

        Ok(ranked)
    }

    /// Get entry by ID
    pub async fn get(&self, id: Uuid) -> Result<Option<LeaderboardEntry>> {
        self.storage.get(id).await
    }

    /// Update entry
    pub async fn update(&self, entry: LeaderboardEntry) -> Result<()> {
        self.storage.update(&entry).await
    }

    /// Delete entry
    pub async fn delete(&self, id: Uuid) -> Result<()> {
        self.storage.delete(id).await
    }

    /// Get statistics
    pub async fn get_stats(
        &self,
        category: Option<LeaderboardCategory>,
    ) -> Result<LeaderboardStats> {
        let entries = if let Some(cat) = category {
            let query = LeaderboardQuery {
                filters: vec![LeaderboardFilter {
                    filter_type: FilterType::Category,
                    value: serde_json::to_value(&cat)?,
                }],
                ..Default::default()
            };
            self.storage.query(&query).await?
        } else {
            self.storage.list_all().await?
        };

        LeaderboardStats::from_entries(&entries)
    }

    /// Get rankings for a specific category and metric
    pub async fn get_rankings(
        &self,
        category: LeaderboardCategory,
        criteria: RankingCriteria,
        limit: usize,
    ) -> Result<LeaderboardRanking> {
        let query = LeaderboardQuery {
            filters: vec![LeaderboardFilter {
                filter_type: FilterType::Category,
                value: serde_json::to_value(&category)?,
            }],
            ranking_criteria: criteria.clone(),
            limit: Some(limit),
            ..Default::default()
        };

        let entries = self.query(query).await?;

        Ok(LeaderboardRanking {
            category,
            criteria,
            entries,
            generated_at: Utc::now(),
        })
    }

    /// Compare two models
    pub async fn compare(
        &self,
        model1: &str,
        model2: &str,
        category: Option<LeaderboardCategory>,
    ) -> Result<ModelComparison> {
        let mut filters = vec![LeaderboardFilter {
            filter_type: FilterType::ModelName,
            value: serde_json::to_value(vec![model1, model2])?,
        }];

        if let Some(cat) = category {
            filters.push(LeaderboardFilter {
                filter_type: FilterType::Category,
                value: serde_json::to_value(&cat)?,
            });
        }

        let query = LeaderboardQuery {
            filters,
            ..Default::default()
        };

        let entries = self.storage.query(&query).await?;

        ModelComparison::from_entries(&entries, model1, model2)
    }

    /// Get trend analysis
    pub async fn get_trends(
        &self,
        model_name: &str,
        metric: &str,
        days: usize,
    ) -> Result<TrendAnalysis> {
        let query = LeaderboardQuery {
            filters: vec![
                LeaderboardFilter {
                    filter_type: FilterType::ModelName,
                    value: serde_json::to_value(model_name)?,
                },
                LeaderboardFilter {
                    filter_type: FilterType::DateRange,
                    value: serde_json::json!({
                        "start": Utc::now() - chrono::Duration::days(days as i64),
                        "end": Utc::now()
                    }),
                },
            ],
            ..Default::default()
        };

        let entries = self.storage.query(&query).await?;

        TrendAnalysis::analyze(&entries, metric)
    }
}

/// Model comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    pub model1: ModelStats,
    pub model2: ModelStats,
    pub relative_performance: HashMap<String, f64>,
    pub winner_by_metric: HashMap<String, String>,
}

/// Model statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    pub name: String,
    pub num_submissions: usize,
    pub best_metrics: HashMap<String, f64>,
    pub average_metrics: HashMap<String, f64>,
    pub latest_submission: DateTime<Utc>,
}

impl ModelComparison {
    /// Create comparison from entries
    pub fn from_entries(entries: &[LeaderboardEntry], model1: &str, model2: &str) -> Result<Self> {
        let model1_entries: Vec<_> = entries.iter().filter(|e| e.model_name == model1).collect();

        let model2_entries: Vec<_> = entries.iter().filter(|e| e.model_name == model2).collect();

        if model1_entries.is_empty() || model2_entries.is_empty() {
            anyhow::bail!("Insufficient data for comparison");
        }

        let model1_stats = Self::calculate_stats(model1, &model1_entries);
        let model2_stats = Self::calculate_stats(model2, &model2_entries);

        let mut relative_performance = HashMap::new();
        let mut winner_by_metric = HashMap::new();

        // Compare common metrics
        for (metric, value1) in &model1_stats.best_metrics {
            if let Some(value2) = model2_stats.best_metrics.get(metric) {
                let relative = match metric.as_str() {
                    "latency_ms" | "memory_mb" | "energy_watts" => {
                        // Lower is better
                        (value2 - value1) / value1 * 100.0
                    },
                    _ => {
                        // Higher is better
                        (value1 - value2) / value2 * 100.0
                    },
                };

                relative_performance.insert(metric.clone(), relative);

                let winner = if relative > 0.0 { model1 } else { model2 };
                winner_by_metric.insert(metric.clone(), winner.to_string());
            }
        }

        Ok(Self {
            model1: model1_stats,
            model2: model2_stats,
            relative_performance,
            winner_by_metric,
        })
    }

    fn calculate_stats(name: &str, entries: &[&LeaderboardEntry]) -> ModelStats {
        let mut best_metrics = HashMap::new();
        let mut sum_metrics: HashMap<String, f64> = HashMap::new();
        let mut count_metrics: HashMap<String, usize> = HashMap::new();

        for entry in entries {
            // Update best metrics
            Self::update_metric(
                &mut best_metrics,
                "latency_ms",
                entry.metrics.latency_ms,
                true,
            );

            if let Some(throughput) = entry.metrics.throughput {
                Self::update_metric(&mut best_metrics, "throughput", throughput, false);
            }

            if let Some(tps) = entry.metrics.tokens_per_second {
                Self::update_metric(&mut best_metrics, "tokens_per_second", tps, false);
            }

            if let Some(memory) = entry.metrics.memory_mb {
                Self::update_metric(&mut best_metrics, "memory_mb", memory, true);
            }

            if let Some(accuracy) = entry.metrics.accuracy {
                Self::update_metric(&mut best_metrics, "accuracy", accuracy, false);
            }

            // Update sums for averages
            *sum_metrics.entry("latency_ms".to_string()).or_insert(0.0) += entry.metrics.latency_ms;
            *count_metrics.entry("latency_ms".to_string()).or_insert(0) += 1;

            if let Some(throughput) = entry.metrics.throughput {
                *sum_metrics.entry("throughput".to_string()).or_insert(0.0) += throughput;
                *count_metrics.entry("throughput".to_string()).or_insert(0) += 1;
            }
        }

        // Calculate averages
        let average_metrics: HashMap<String, f64> = sum_metrics
            .iter()
            .map(|(k, v)| (k.clone(), v / count_metrics[k] as f64))
            .collect();

        let latest_submission = entries.iter().map(|e| e.timestamp).max().unwrap_or_else(Utc::now);

        ModelStats {
            name: name.to_string(),
            num_submissions: entries.len(),
            best_metrics,
            average_metrics,
            latest_submission,
        }
    }

    fn update_metric(
        metrics: &mut HashMap<String, f64>,
        name: &str,
        value: f64,
        lower_is_better: bool,
    ) {
        metrics
            .entry(name.to_string())
            .and_modify(|v| {
                if lower_is_better {
                    *v = v.min(value);
                } else {
                    *v = v.max(value);
                }
            })
            .or_insert(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaderboard_category_display() {
        assert_eq!(LeaderboardCategory::Inference.to_string(), "Inference");
        assert_eq!(
            LeaderboardCategory::Custom("ML".to_string()).to_string(),
            "ML"
        );
    }

    #[test]
    fn test_model_comparison() {
        let entries = vec![
            LeaderboardEntry {
                id: Uuid::new_v4(),
                timestamp: Utc::now(),
                model_name: "model1".to_string(),
                model_version: "1.0".to_string(),
                benchmark_name: "test".to_string(),
                category: LeaderboardCategory::Inference,
                hardware: HardwareInfo {
                    cpu: "Intel Xeon".to_string(),
                    cpu_cores: 32,
                    gpu: Some("NVIDIA A100".to_string()),
                    gpu_count: Some(1),
                    memory_gb: 256.0,
                    accelerator: Some(AcceleratorType::CUDA),
                    platform: "x86_64".to_string(),
                },
                software: SoftwareInfo {
                    framework_version: "0.1.0".to_string(),
                    rust_version: "1.75".to_string(),
                    os: "Linux".to_string(),
                    optimization_level: OptimizationLevel::O3,
                    precision: Precision::FP16,
                    quantization: None,
                    compiler_flags: vec![],
                },
                metrics: PerformanceMetrics {
                    latency_ms: 10.0,
                    latency_percentiles: LatencyPercentiles {
                        p50: 9.0,
                        p90: 12.0,
                        p95: 14.0,
                        p99: 18.0,
                        p999: 25.0,
                    },
                    throughput: Some(100.0),
                    tokens_per_second: None,
                    memory_mb: Some(1024.0),
                    peak_memory_mb: Some(1536.0),
                    gpu_utilization: Some(85.0),
                    accuracy: Some(0.95),
                    energy_watts: None,
                    custom_metrics: HashMap::new(),
                },
                metadata: HashMap::new(),
                validated: true,
                submitter: SubmitterInfo {
                    name: "Test User".to_string(),
                    organization: None,
                    email: None,
                    github: None,
                },
                tags: vec![],
            },
            LeaderboardEntry {
                id: Uuid::new_v4(),
                timestamp: Utc::now(),
                model_name: "model2".to_string(),
                model_version: "1.0".to_string(),
                benchmark_name: "test".to_string(),
                category: LeaderboardCategory::Inference,
                hardware: HardwareInfo {
                    cpu: "Intel Xeon".to_string(),
                    cpu_cores: 32,
                    gpu: Some("NVIDIA A100".to_string()),
                    gpu_count: Some(1),
                    memory_gb: 256.0,
                    accelerator: Some(AcceleratorType::CUDA),
                    platform: "x86_64".to_string(),
                },
                software: SoftwareInfo {
                    framework_version: "0.1.0".to_string(),
                    rust_version: "1.75".to_string(),
                    os: "Linux".to_string(),
                    optimization_level: OptimizationLevel::O3,
                    precision: Precision::FP16,
                    quantization: None,
                    compiler_flags: vec![],
                },
                metrics: PerformanceMetrics {
                    latency_ms: 15.0,
                    latency_percentiles: LatencyPercentiles {
                        p50: 14.0,
                        p90: 17.0,
                        p95: 19.0,
                        p99: 23.0,
                        p999: 30.0,
                    },
                    throughput: Some(66.7),
                    tokens_per_second: None,
                    memory_mb: Some(768.0),
                    peak_memory_mb: Some(1024.0),
                    gpu_utilization: Some(75.0),
                    accuracy: Some(0.92),
                    energy_watts: None,
                    custom_metrics: HashMap::new(),
                },
                metadata: HashMap::new(),
                validated: true,
                submitter: SubmitterInfo {
                    name: "Test User".to_string(),
                    organization: None,
                    email: None,
                    github: None,
                },
                tags: vec![],
            },
        ];

        let comparison = ModelComparison::from_entries(&entries, "model1", "model2").unwrap();

        assert_eq!(comparison.model1.name, "model1");
        assert_eq!(comparison.model2.name, "model2");
        assert_eq!(comparison.winner_by_metric["latency_ms"], "model1");
        assert_eq!(comparison.winner_by_metric["memory_mb"], "model2");
        assert_eq!(comparison.winner_by_metric["accuracy"], "model1");
    }
}
