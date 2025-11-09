//! Client interface for interacting with the leaderboard system

use super::{
    LeaderboardCategory, LeaderboardEntry, LeaderboardManager, LeaderboardQuery, LeaderboardStats,
    LeaderboardSubmission, ModelComparison, RankingCriteria, TrendAnalysis,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    /// API endpoint (for remote storage)
    pub endpoint: Option<String>,
    /// API key (for authentication)
    pub api_key: Option<String>,
    /// Local storage directory (for file storage)
    pub local_dir: Option<String>,
    /// Default category for queries
    pub default_category: Option<LeaderboardCategory>,
    /// Default limit for queries
    pub default_limit: usize,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            endpoint: None,
            api_key: None,
            local_dir: Some("leaderboard_data".to_string()),
            default_category: None,
            default_limit: 100,
        }
    }
}

/// High-level client for leaderboard operations
pub struct LeaderboardClient {
    manager: Arc<LeaderboardManager>,
    config: ClientConfig,
}

impl LeaderboardClient {
    /// Create new client
    pub fn new(manager: Arc<LeaderboardManager>, config: ClientConfig) -> Self {
        Self { manager, config }
    }

    /// Submit a benchmark result
    pub async fn submit(&self, submission: LeaderboardSubmission) -> Result<Uuid> {
        self.manager.submit(submission).await
    }

    /// Submit from benchmark report
    pub async fn submit_from_report(
        &self,
        report: &crate::performance::BenchmarkReport,
        model_name: String,
        model_version: String,
        submitter_name: String,
    ) -> Result<Uuid> {
        let submission =
            self.create_submission_from_report(report, model_name, model_version, submitter_name)?;

        self.submit(submission).await
    }

    /// Get top models for a category
    pub async fn get_top_models(
        &self,
        category: LeaderboardCategory,
        metric: crate::leaderboard::RankingMetric,
        limit: usize,
    ) -> Result<Vec<LeaderboardEntry>> {
        let criteria = RankingCriteria {
            primary_metric: metric,
            secondary_metric: None,
            order: crate::leaderboard::SortOrder::Ascending,
        };

        let ranking = self.manager.get_rankings(category, criteria, limit).await?;
        Ok(ranking.entries)
    }

    /// Get leaderboard for specific benchmark
    pub async fn get_benchmark_leaderboard(
        &self,
        benchmark_name: &str,
        limit: Option<usize>,
    ) -> Result<Vec<LeaderboardEntry>> {
        let query = LeaderboardQuery::builder()
            .filter(
                crate::leaderboard::FilterType::BenchmarkName,
                serde_json::json!(benchmark_name),
            )
            .limit(limit.unwrap_or(self.config.default_limit))
            .build();

        self.manager.query(query).await
    }

    /// Get model history
    pub async fn get_model_history(&self, model_name: &str) -> Result<Vec<LeaderboardEntry>> {
        let query = LeaderboardQuery::builder()
            .model_name(model_name)
            .ranking(RankingCriteria {
                primary_metric: crate::leaderboard::RankingMetric::Date,
                secondary_metric: None,
                order: crate::leaderboard::SortOrder::Descending,
            })
            .build();

        self.manager.query(query).await
    }

    /// Compare two models
    pub async fn compare_models(
        &self,
        model1: &str,
        model2: &str,
        category: Option<LeaderboardCategory>,
    ) -> Result<ModelComparison> {
        self.manager.compare(model1, model2, category).await
    }

    /// Get performance trends
    pub async fn get_trends(
        &self,
        model_name: &str,
        metric: &str,
        days: usize,
    ) -> Result<TrendAnalysis> {
        self.manager.get_trends(model_name, metric, days).await
    }

    /// Get overall statistics
    pub async fn get_stats(
        &self,
        category: Option<LeaderboardCategory>,
    ) -> Result<LeaderboardStats> {
        self.manager.get_stats(category).await
    }

    /// Search entries
    pub async fn search(&self, search_params: SearchParams) -> Result<Vec<LeaderboardEntry>> {
        let mut query_builder = LeaderboardQuery::builder();

        if let Some(models) = search_params.model_names {
            query_builder = query_builder.model_names(models);
        }

        if let Some(benchmarks) = search_params.benchmark_names {
            query_builder = query_builder.filter(
                crate::leaderboard::FilterType::BenchmarkName,
                serde_json::json!(benchmarks),
            );
        }

        if let Some(category) = search_params.category {
            query_builder = query_builder.category(category);
        }

        if let Some(tags) = search_params.tags {
            query_builder = query_builder.tags(tags);
        }

        if let Some(min_accuracy) = search_params.min_accuracy {
            query_builder = query_builder.metric_range("accuracy", Some(min_accuracy), None);
        }

        if let Some(max_latency) = search_params.max_latency_ms {
            query_builder = query_builder.metric_range("latency", None, Some(max_latency));
        }

        let query = query_builder
            .limit(search_params.limit.unwrap_or(self.config.default_limit))
            .build();

        self.manager.query(query).await
    }

    /// Get entry by ID
    pub async fn get_entry(&self, id: Uuid) -> Result<Option<LeaderboardEntry>> {
        self.manager.get(id).await
    }

    /// Get recent submissions
    pub async fn get_recent(
        &self,
        days: usize,
        limit: Option<usize>,
    ) -> Result<Vec<LeaderboardEntry>> {
        let start = chrono::Utc::now() - chrono::Duration::days(days as i64);
        let end = chrono::Utc::now();

        let query = LeaderboardQuery::builder()
            .date_range(start, end)
            .ranking(RankingCriteria {
                primary_metric: crate::leaderboard::RankingMetric::Date,
                secondary_metric: None,
                order: crate::leaderboard::SortOrder::Descending,
            })
            .limit(limit.unwrap_or(self.config.default_limit))
            .build();

        self.manager.query(query).await
    }

    /// Create submission from benchmark report
    fn create_submission_from_report(
        &self,
        report: &crate::performance::BenchmarkReport,
        model_name: String,
        model_version: String,
        submitter_name: String,
    ) -> Result<LeaderboardSubmission> {
        // Extract metrics from report
        let latency_ms = report.summary.avg_latency_ms;
        let throughput = report.summary.avg_throughput;

        let mut custom_metrics = std::collections::HashMap::new();
        for (name, stats) in &report.aggregate_metrics {
            custom_metrics.insert(name.clone(), stats.mean);
        }

        // Create latency percentiles
        let latency_percentiles = if let Some(duration_stats) = &report.duration_stats {
            crate::leaderboard::LatencyPercentiles {
                p50: duration_stats.percentiles.get("p50").copied().unwrap_or(0.0) * 1000.0,
                p90: duration_stats.percentiles.get("p90").copied().unwrap_or(0.0) * 1000.0,
                p95: duration_stats.percentiles.get("p95").copied().unwrap_or(0.0) * 1000.0,
                p99: duration_stats.percentiles.get("p99").copied().unwrap_or(0.0) * 1000.0,
                p999: duration_stats.percentiles.get("p999").copied().unwrap_or(0.0) * 1000.0,
            }
        } else {
            crate::leaderboard::LatencyPercentiles {
                p50: latency_ms * 0.9,
                p90: latency_ms * 1.1,
                p95: latency_ms * 1.2,
                p99: latency_ms * 1.5,
                p999: latency_ms * 2.0,
            }
        };

        // Determine category from benchmark name/tags
        let category = if report.tags.contains(&"inference".to_string()) {
            LeaderboardCategory::Inference
        } else if report.tags.contains(&"training".to_string()) {
            LeaderboardCategory::Training
        } else if report.tags.contains(&"memory".to_string()) {
            LeaderboardCategory::Memory
        } else {
            LeaderboardCategory::Custom(report.name.clone())
        };

        Ok(LeaderboardSubmission {
            model_name,
            model_version,
            benchmark_name: report.name.clone(),
            category,
            hardware: crate::leaderboard::HardwareInfo {
                cpu: "Unknown".to_string(), // Would need system info
                cpu_cores: num_cpus::get(),
                gpu: None, // Would need GPU detection
                gpu_count: None,
                memory_gb: 0.0, // Would need system info
                accelerator: None,
                platform: std::env::consts::ARCH.to_string(),
            },
            software: crate::leaderboard::SoftwareInfo {
                framework_version: env!("CARGO_PKG_VERSION").to_string(),
                rust_version: "1.75".to_string(), // Would need rustc version
                os: std::env::consts::OS.to_string(),
                optimization_level: crate::leaderboard::OptimizationLevel::O2,
                precision: crate::leaderboard::Precision::FP32,
                quantization: None,
                compiler_flags: vec![],
            },
            metrics: crate::leaderboard::PerformanceMetrics {
                latency_ms,
                latency_percentiles,
                throughput,
                tokens_per_second: custom_metrics.get("tokens_per_second").copied(),
                memory_mb: custom_metrics.get("memory_mb").copied(),
                peak_memory_mb: None,
                gpu_utilization: custom_metrics.get("gpu_utilization").copied(),
                accuracy: custom_metrics.get("accuracy").copied(),
                energy_watts: None,
                custom_metrics,
            },
            metadata: std::collections::HashMap::new(),
            submitter: crate::leaderboard::SubmitterInfo {
                name: submitter_name,
                organization: None,
                email: None,
                github: None,
            },
            tags: report.tags.clone(),
            benchmark_report: Some(serde_json::to_value(report)?),
        })
    }
}

/// Search parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchParams {
    /// Model names to search for
    pub model_names: Option<Vec<String>>,
    /// Benchmark names to search for
    pub benchmark_names: Option<Vec<String>>,
    /// Category filter
    pub category: Option<LeaderboardCategory>,
    /// Tags to search for
    pub tags: Option<Vec<String>>,
    /// Minimum accuracy
    pub min_accuracy: Option<f64>,
    /// Maximum latency in milliseconds
    pub max_latency_ms: Option<f64>,
    /// Result limit
    pub limit: Option<usize>,
}

/// Builder for creating leaderboard client
pub struct ClientBuilder {
    config: ClientConfig,
}

impl ClientBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: ClientConfig::default(),
        }
    }

    /// Set API endpoint
    pub fn endpoint(mut self, endpoint: String) -> Self {
        self.config.endpoint = Some(endpoint);
        self
    }

    /// Set API key
    pub fn api_key(mut self, key: String) -> Self {
        self.config.api_key = Some(key);
        self
    }

    /// Set local directory
    pub fn local_dir(mut self, dir: String) -> Self {
        self.config.local_dir = Some(dir);
        self
    }

    /// Set default category
    pub fn default_category(mut self, category: LeaderboardCategory) -> Self {
        self.config.default_category = Some(category);
        self
    }

    /// Set default limit
    pub fn default_limit(mut self, limit: usize) -> Self {
        self.config.default_limit = limit;
        self
    }

    /// Build client with file storage
    pub async fn build_with_file_storage(self) -> Result<LeaderboardClient> {
        use super::{DefaultRankingAlgorithm, DefaultValidator, FileStorage};

        let storage_dir = self
            .config
            .local_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Local directory not specified"))?;

        let storage = Arc::new(FileStorage::new(storage_dir).await?);
        let validator = Arc::new(DefaultValidator::new());
        let ranking = Arc::new(DefaultRankingAlgorithm);

        let manager = Arc::new(LeaderboardManager::new(storage, validator, ranking));

        Ok(LeaderboardClient::new(manager, self.config))
    }

    /// Build client with remote storage
    pub fn build_with_remote_storage(self) -> Result<LeaderboardClient> {
        use super::{DefaultRankingAlgorithm, DefaultValidator, RemoteStorage};

        let endpoint = self
            .config
            .endpoint
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("API endpoint not specified"))?;

        let storage = Arc::new(RemoteStorage::new(
            endpoint.clone(),
            self.config.api_key.clone(),
        ));
        let validator = Arc::new(DefaultValidator::new());
        let ranking = Arc::new(DefaultRankingAlgorithm);

        let manager = Arc::new(LeaderboardManager::new(storage, validator, ranking));

        Ok(LeaderboardClient::new(manager, self.config))
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaderboard::*;

    #[tokio::test]
    async fn test_client_builder() {
        let client = ClientBuilder::new()
            .local_dir("/tmp/test_leaderboard".to_string())
            .default_category(LeaderboardCategory::Inference)
            .default_limit(50)
            .build_with_file_storage()
            .await
            .unwrap();

        assert_eq!(client.config.default_limit, 50);
    }

    #[test]
    fn test_search_params() {
        let params = SearchParams {
            model_names: Some(vec!["bert".to_string()]),
            category: Some(LeaderboardCategory::Inference),
            min_accuracy: Some(0.9),
            max_latency_ms: Some(100.0),
            ..Default::default()
        };

        assert_eq!(params.model_names.unwrap()[0], "bert");
        assert_eq!(params.min_accuracy.unwrap(), 0.9);
    }
}
