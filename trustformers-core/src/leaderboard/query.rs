//! Query and filtering functionality for leaderboard entries

#![allow(unused_variables)] // Leaderboard query

use super::{LeaderboardCategory, LeaderboardEntry, RankingCriteria};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Query for filtering leaderboard entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardQuery {
    /// Filters to apply
    pub filters: Vec<LeaderboardFilter>,
    /// Ranking criteria
    pub ranking_criteria: RankingCriteria,
    /// Maximum number of results
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: Option<usize>,
    /// Include only validated entries
    pub validated_only: bool,
}

impl Default for LeaderboardQuery {
    fn default() -> Self {
        Self {
            filters: Vec::new(),
            ranking_criteria: RankingCriteria::default(),
            limit: None,
            offset: None,
            validated_only: true,
        }
    }
}

/// Filter for leaderboard queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardFilter {
    /// Type of filter
    pub filter_type: FilterType,
    /// Filter value
    pub value: serde_json::Value,
}

/// Types of filters available
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FilterType {
    /// Filter by model name
    ModelName,
    /// Filter by model version
    ModelVersion,
    /// Filter by benchmark name
    BenchmarkName,
    /// Filter by category
    Category,
    /// Filter by hardware
    Hardware,
    /// Filter by software
    Software,
    /// Filter by date range
    DateRange,
    /// Filter by submitter
    Submitter,
    /// Filter by tags
    Tags,
    /// Filter by metric range
    MetricRange,
    /// Filter by precision
    Precision,
    /// Filter by optimization level
    OptimizationLevel,
    /// Custom filter
    Custom(String),
}

impl LeaderboardQuery {
    /// Create a new query builder
    pub fn builder() -> QueryBuilder {
        QueryBuilder::new()
    }

    /// Apply filters to a list of entries
    pub fn apply_filters(&self, mut entries: Vec<LeaderboardEntry>) -> Vec<LeaderboardEntry> {
        // Apply validated_only filter first
        if self.validated_only {
            entries.retain(|e| e.validated);
        }

        // Apply each filter
        for filter in &self.filters {
            entries = self.apply_filter(entries, filter);
        }

        // Apply offset
        if let Some(offset) = self.offset {
            entries = entries.into_iter().skip(offset).collect();
        }

        entries
    }

    /// Apply a single filter to entries
    fn apply_filter(
        &self,
        entries: Vec<LeaderboardEntry>,
        filter: &LeaderboardFilter,
    ) -> Vec<LeaderboardEntry> {
        match &filter.filter_type {
            FilterType::ModelName => {
                if let Ok(names) = serde_json::from_value::<Vec<String>>(filter.value.clone()) {
                    let name_set: HashSet<_> = names.into_iter().collect();
                    entries.into_iter().filter(|e| name_set.contains(&e.model_name)).collect()
                } else if let Ok(name) = serde_json::from_value::<String>(filter.value.clone()) {
                    entries.into_iter().filter(|e| e.model_name == name).collect()
                } else {
                    entries
                }
            },
            FilterType::ModelVersion => {
                if let Ok(version) = serde_json::from_value::<String>(filter.value.clone()) {
                    entries.into_iter().filter(|e| e.model_version == version).collect()
                } else {
                    entries
                }
            },
            FilterType::BenchmarkName => {
                if let Ok(names) = serde_json::from_value::<Vec<String>>(filter.value.clone()) {
                    let name_set: HashSet<_> = names.into_iter().collect();
                    entries.into_iter().filter(|e| name_set.contains(&e.benchmark_name)).collect()
                } else if let Ok(name) = serde_json::from_value::<String>(filter.value.clone()) {
                    entries.into_iter().filter(|e| e.benchmark_name == name).collect()
                } else {
                    entries
                }
            },
            FilterType::Category => {
                if let Ok(category) =
                    serde_json::from_value::<LeaderboardCategory>(filter.value.clone())
                {
                    entries.into_iter().filter(|e| e.category == category).collect()
                } else {
                    entries
                }
            },
            FilterType::Hardware => {
                if let Ok(hw_filter) =
                    serde_json::from_value::<HardwareFilter>(filter.value.clone())
                {
                    entries.into_iter().filter(|e| hw_filter.matches(&e.hardware)).collect()
                } else {
                    entries
                }
            },
            FilterType::Software => {
                if let Ok(sw_filter) =
                    serde_json::from_value::<SoftwareFilter>(filter.value.clone())
                {
                    entries.into_iter().filter(|e| sw_filter.matches(&e.software)).collect()
                } else {
                    entries
                }
            },
            FilterType::DateRange => {
                if let Ok(range) = serde_json::from_value::<DateRange>(filter.value.clone()) {
                    entries.into_iter().filter(|e| range.contains(e.timestamp)).collect()
                } else {
                    entries
                }
            },
            FilterType::Submitter => {
                if let Ok(name) = serde_json::from_value::<String>(filter.value.clone()) {
                    entries.into_iter().filter(|e| e.submitter.name == name).collect()
                } else {
                    entries
                }
            },
            FilterType::Tags => {
                if let Ok(tags) = serde_json::from_value::<Vec<String>>(filter.value.clone()) {
                    let tag_set: HashSet<_> = tags.into_iter().collect();
                    entries
                        .into_iter()
                        .filter(|e| e.tags.iter().any(|t| tag_set.contains(t)))
                        .collect()
                } else {
                    entries
                }
            },
            FilterType::MetricRange => {
                if let Ok(range) = serde_json::from_value::<MetricRange>(filter.value.clone()) {
                    entries.into_iter().filter(|e| range.matches(&e.metrics)).collect()
                } else {
                    entries
                }
            },
            FilterType::Precision => {
                if let Ok(precision) =
                    serde_json::from_value::<crate::leaderboard::Precision>(filter.value.clone())
                {
                    entries.into_iter().filter(|e| e.software.precision == precision).collect()
                } else {
                    entries
                }
            },
            FilterType::OptimizationLevel => {
                if let Ok(level) = serde_json::from_value::<crate::leaderboard::OptimizationLevel>(
                    filter.value.clone(),
                ) {
                    entries.into_iter().filter(|e| e.software.optimization_level == level).collect()
                } else {
                    entries
                }
            },
            FilterType::Custom(name) => {
                // Custom filters can be implemented as needed
                entries
            },
        }
    }
}

/// Hardware filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareFilter {
    pub cpu: Option<String>,
    pub gpu: Option<String>,
    pub min_memory_gb: Option<f64>,
    pub max_memory_gb: Option<f64>,
    pub platform: Option<String>,
}

impl HardwareFilter {
    fn matches(&self, hardware: &crate::leaderboard::HardwareInfo) -> bool {
        if let Some(cpu) = &self.cpu {
            if !hardware.cpu.contains(cpu) {
                return false;
            }
        }

        if let Some(gpu) = &self.gpu {
            if hardware.gpu.as_ref().map_or(true, |g| !g.contains(gpu)) {
                return false;
            }
        }

        if let Some(min_mem) = self.min_memory_gb {
            if hardware.memory_gb < min_mem {
                return false;
            }
        }

        if let Some(max_mem) = self.max_memory_gb {
            if hardware.memory_gb > max_mem {
                return false;
            }
        }

        if let Some(platform) = &self.platform {
            if !hardware.platform.contains(platform) {
                return false;
            }
        }

        true
    }
}

/// Software filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftwareFilter {
    pub framework_version: Option<String>,
    pub rust_version: Option<String>,
    pub os: Option<String>,
    pub has_quantization: Option<bool>,
}

impl SoftwareFilter {
    fn matches(&self, software: &crate::leaderboard::SoftwareInfo) -> bool {
        if let Some(fw_version) = &self.framework_version {
            if !software.framework_version.contains(fw_version) {
                return false;
            }
        }

        if let Some(rust_version) = &self.rust_version {
            if !software.rust_version.contains(rust_version) {
                return false;
            }
        }

        if let Some(os) = &self.os {
            if !software.os.contains(os) {
                return false;
            }
        }

        if let Some(has_quant) = self.has_quantization {
            if has_quant != software.quantization.is_some() {
                return false;
            }
        }

        true
    }
}

/// Date range filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

impl DateRange {
    fn contains(&self, timestamp: DateTime<Utc>) -> bool {
        timestamp >= self.start && timestamp <= self.end
    }
}

/// Metric range filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRange {
    pub metric: String,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl MetricRange {
    fn matches(&self, metrics: &crate::leaderboard::PerformanceMetrics) -> bool {
        let value = match self.metric.as_str() {
            "latency" => Some(metrics.latency_ms),
            "throughput" => metrics.throughput,
            "tokens_per_second" => metrics.tokens_per_second,
            "memory" => metrics.memory_mb,
            "peak_memory" => metrics.peak_memory_mb,
            "gpu_utilization" => metrics.gpu_utilization,
            "accuracy" => metrics.accuracy,
            "energy" => metrics.energy_watts,
            _ => metrics.custom_metrics.get(&self.metric).copied(),
        };

        if let Some(val) = value {
            if let Some(min) = self.min {
                if val < min {
                    return false;
                }
            }

            if let Some(max) = self.max {
                if val > max {
                    return false;
                }
            }

            true
        } else {
            false
        }
    }
}

/// Query builder for constructing leaderboard queries
pub struct QueryBuilder {
    query: LeaderboardQuery,
}

impl QueryBuilder {
    /// Create new query builder
    pub fn new() -> Self {
        Self {
            query: LeaderboardQuery::default(),
        }
    }

    /// Add a filter
    pub fn filter(mut self, filter_type: FilterType, value: serde_json::Value) -> Self {
        self.query.filters.push(LeaderboardFilter { filter_type, value });
        self
    }

    /// Filter by model name
    pub fn model_name(self, name: &str) -> Self {
        self.filter(FilterType::ModelName, serde_json::json!(name))
    }

    /// Filter by multiple model names
    pub fn model_names(self, names: Vec<String>) -> Self {
        self.filter(FilterType::ModelName, serde_json::json!(names))
    }

    /// Filter by category
    pub fn category(self, category: LeaderboardCategory) -> Self {
        self.filter(
            FilterType::Category,
            serde_json::to_value(category).unwrap(),
        )
    }

    /// Filter by date range
    pub fn date_range(self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.filter(
            FilterType::DateRange,
            serde_json::json!({
                "start": start,
                "end": end
            }),
        )
    }

    /// Filter by tags
    pub fn tags(self, tags: Vec<String>) -> Self {
        self.filter(FilterType::Tags, serde_json::json!(tags))
    }

    /// Filter by metric range
    pub fn metric_range(self, metric: &str, min: Option<f64>, max: Option<f64>) -> Self {
        self.filter(
            FilterType::MetricRange,
            serde_json::json!({
                "metric": metric,
                "min": min,
                "max": max
            }),
        )
    }

    /// Set ranking criteria
    pub fn ranking(mut self, criteria: RankingCriteria) -> Self {
        self.query.ranking_criteria = criteria;
        self
    }

    /// Set result limit
    pub fn limit(mut self, limit: usize) -> Self {
        self.query.limit = Some(limit);
        self
    }

    /// Set offset for pagination
    pub fn offset(mut self, offset: usize) -> Self {
        self.query.offset = Some(offset);
        self
    }

    /// Include unvalidated entries
    pub fn include_unvalidated(mut self) -> Self {
        self.query.validated_only = false;
        self
    }

    /// Build the query
    pub fn build(self) -> LeaderboardQuery {
        self.query
    }
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaderboard::*;
    use std::collections::HashMap;

    fn create_test_entries() -> Vec<LeaderboardEntry> {
        vec![
            LeaderboardEntry {
                id: uuid::Uuid::new_v4(),
                timestamp: chrono::Utc::now(),
                model_name: "model1".to_string(),
                model_version: "1.0".to_string(),
                benchmark_name: "benchmark1".to_string(),
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
                    name: "User1".to_string(),
                    organization: None,
                    email: None,
                    github: None,
                },
                tags: vec!["fast".to_string(), "gpu".to_string()],
            },
            LeaderboardEntry {
                id: uuid::Uuid::new_v4(),
                timestamp: chrono::Utc::now(),
                model_name: "model2".to_string(),
                model_version: "2.0".to_string(),
                benchmark_name: "benchmark2".to_string(),
                category: LeaderboardCategory::Training,
                hardware: HardwareInfo {
                    cpu: "AMD EPYC".to_string(),
                    cpu_cores: 64,
                    gpu: None,
                    gpu_count: None,
                    memory_gb: 512.0,
                    accelerator: None,
                    platform: "x86_64".to_string(),
                },
                software: SoftwareInfo {
                    framework_version: "0.1.0".to_string(),
                    rust_version: "1.75".to_string(),
                    os: "Linux".to_string(),
                    optimization_level: OptimizationLevel::O2,
                    precision: Precision::FP32,
                    quantization: Some("int8".to_string()),
                    compiler_flags: vec![],
                },
                metrics: PerformanceMetrics {
                    latency_ms: 20.0,
                    latency_percentiles: LatencyPercentiles {
                        p50: 18.0,
                        p90: 24.0,
                        p95: 28.0,
                        p99: 36.0,
                        p999: 50.0,
                    },
                    throughput: Some(50.0),
                    tokens_per_second: None,
                    memory_mb: Some(512.0),
                    peak_memory_mb: Some(768.0),
                    gpu_utilization: None,
                    accuracy: Some(0.92),
                    energy_watts: None,
                    custom_metrics: HashMap::new(),
                },
                metadata: HashMap::new(),
                validated: true,
                submitter: SubmitterInfo {
                    name: "User2".to_string(),
                    organization: None,
                    email: None,
                    github: None,
                },
                tags: vec!["cpu".to_string(), "quantized".to_string()],
            },
        ]
    }

    #[test]
    fn test_filter_by_model_name() {
        let entries = create_test_entries();
        let query = LeaderboardQuery::builder().model_name("model1").build();

        let filtered = query.apply_filters(entries);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].model_name, "model1");
    }

    #[test]
    fn test_filter_by_category() {
        let entries = create_test_entries();
        let query = LeaderboardQuery::builder().category(LeaderboardCategory::Training).build();

        let filtered = query.apply_filters(entries);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].model_name, "model2");
    }

    #[test]
    fn test_filter_by_tags() {
        let entries = create_test_entries();
        let query = LeaderboardQuery::builder().tags(vec!["gpu".to_string()]).build();

        let filtered = query.apply_filters(entries);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].model_name, "model1");
    }

    #[test]
    fn test_filter_by_metric_range() {
        let entries = create_test_entries();
        let query = LeaderboardQuery::builder().metric_range("latency", None, Some(15.0)).build();

        let filtered = query.apply_filters(entries);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].model_name, "model1");
    }

    #[test]
    fn test_multiple_filters() {
        let entries = create_test_entries();
        let query = LeaderboardQuery::builder()
            .category(LeaderboardCategory::Inference)
            .tags(vec!["gpu".to_string()])
            .build();

        let filtered = query.apply_filters(entries);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].model_name, "model1");
    }
}
