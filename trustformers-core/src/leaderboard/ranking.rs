//! Ranking algorithms and criteria for leaderboard entries

use super::{LeaderboardCategory, LeaderboardEntry};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Ranking criteria for sorting entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingCriteria {
    /// Primary metric to rank by
    pub primary_metric: RankingMetric,
    /// Secondary metric for tie-breaking
    pub secondary_metric: Option<RankingMetric>,
    /// Sort order
    pub order: SortOrder,
}

impl Default for RankingCriteria {
    fn default() -> Self {
        Self {
            primary_metric: RankingMetric::Latency,
            secondary_metric: Some(RankingMetric::Throughput),
            order: SortOrder::Ascending,
        }
    }
}

/// Metrics that can be used for ranking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RankingMetric {
    /// Latency (lower is better)
    Latency,
    /// Throughput (higher is better)
    Throughput,
    /// Tokens per second (higher is better)
    TokensPerSecond,
    /// Memory usage (lower is better)
    Memory,
    /// Peak memory usage (lower is better)
    PeakMemory,
    /// GPU utilization (context-dependent)
    GPUUtilization,
    /// Accuracy (higher is better)
    Accuracy,
    /// Energy efficiency (lower is better)
    Energy,
    /// Submission date (newer first)
    Date,
    /// Custom metric by name
    Custom(String),
}

/// Sort order
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SortOrder {
    Ascending,
    Descending,
}

/// Leaderboard ranking result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardRanking {
    /// Category
    pub category: LeaderboardCategory,
    /// Ranking criteria used
    pub criteria: RankingCriteria,
    /// Ranked entries
    pub entries: Vec<LeaderboardEntry>,
    /// When the ranking was generated
    pub generated_at: DateTime<Utc>,
}

/// Trait for ranking algorithms
pub trait RankingAlgorithm: Send + Sync {
    /// Rank entries according to criteria
    fn rank(
        &self,
        entries: Vec<LeaderboardEntry>,
        criteria: &RankingCriteria,
    ) -> Result<Vec<LeaderboardEntry>>;
}

/// Default ranking algorithm
pub struct DefaultRankingAlgorithm;

impl RankingAlgorithm for DefaultRankingAlgorithm {
    fn rank(
        &self,
        mut entries: Vec<LeaderboardEntry>,
        criteria: &RankingCriteria,
    ) -> Result<Vec<LeaderboardEntry>> {
        entries.sort_by(|a, b| criteria.compare(a, b));
        Ok(entries)
    }
}

impl RankingCriteria {
    /// Compare two entries based on ranking criteria
    pub fn compare(&self, a: &LeaderboardEntry, b: &LeaderboardEntry) -> Ordering {
        let primary_cmp = self.compare_by_metric(a, b, &self.primary_metric);

        if primary_cmp != Ordering::Equal {
            return if self.order == SortOrder::Ascending {
                primary_cmp
            } else {
                primary_cmp.reverse()
            };
        }

        // Use secondary metric for tie-breaking
        if let Some(secondary) = &self.secondary_metric {
            let secondary_cmp = self.compare_by_metric(a, b, secondary);
            if self.order == SortOrder::Ascending {
                secondary_cmp
            } else {
                secondary_cmp.reverse()
            }
        } else {
            // Final tie-breaker: timestamp (newer first)
            b.timestamp.cmp(&a.timestamp)
        }
    }

    /// Compare two entries by a specific metric
    fn compare_by_metric(
        &self,
        a: &LeaderboardEntry,
        b: &LeaderboardEntry,
        metric: &RankingMetric,
    ) -> Ordering {
        match metric {
            RankingMetric::Latency => a
                .metrics
                .latency_ms
                .partial_cmp(&b.metrics.latency_ms)
                .unwrap_or(Ordering::Equal),
            RankingMetric::Throughput => {
                let a_val = a.metrics.throughput.unwrap_or(0.0);
                let b_val = b.metrics.throughput.unwrap_or(0.0);
                b_val.partial_cmp(&a_val).unwrap_or(Ordering::Equal) // Higher is better
            },
            RankingMetric::TokensPerSecond => {
                let a_val = a.metrics.tokens_per_second.unwrap_or(0.0);
                let b_val = b.metrics.tokens_per_second.unwrap_or(0.0);
                b_val.partial_cmp(&a_val).unwrap_or(Ordering::Equal) // Higher is better
            },
            RankingMetric::Memory => {
                let a_val = a.metrics.memory_mb.unwrap_or(f64::MAX);
                let b_val = b.metrics.memory_mb.unwrap_or(f64::MAX);
                a_val.partial_cmp(&b_val).unwrap_or(Ordering::Equal)
            },
            RankingMetric::PeakMemory => {
                let a_val = a.metrics.peak_memory_mb.unwrap_or(f64::MAX);
                let b_val = b.metrics.peak_memory_mb.unwrap_or(f64::MAX);
                a_val.partial_cmp(&b_val).unwrap_or(Ordering::Equal)
            },
            RankingMetric::GPUUtilization => {
                let a_val = a.metrics.gpu_utilization.unwrap_or(0.0);
                let b_val = b.metrics.gpu_utilization.unwrap_or(0.0);
                b_val.partial_cmp(&a_val).unwrap_or(Ordering::Equal) // Higher is usually better
            },
            RankingMetric::Accuracy => {
                let a_val = a.metrics.accuracy.unwrap_or(0.0);
                let b_val = b.metrics.accuracy.unwrap_or(0.0);
                b_val.partial_cmp(&a_val).unwrap_or(Ordering::Equal) // Higher is better
            },
            RankingMetric::Energy => {
                let a_val = a.metrics.energy_watts.unwrap_or(f64::MAX);
                let b_val = b.metrics.energy_watts.unwrap_or(f64::MAX);
                a_val.partial_cmp(&b_val).unwrap_or(Ordering::Equal)
            },
            RankingMetric::Date => {
                b.timestamp.cmp(&a.timestamp) // Newer first
            },
            RankingMetric::Custom(name) => {
                let a_val = a.metrics.custom_metrics.get(name).copied().unwrap_or(0.0);
                let b_val = b.metrics.custom_metrics.get(name).copied().unwrap_or(0.0);
                a_val.partial_cmp(&b_val).unwrap_or(Ordering::Equal)
            },
        }
    }
}

/// Weighted ranking algorithm that considers multiple metrics
pub struct WeightedRankingAlgorithm {
    weights: Vec<(RankingMetric, f64)>,
}

impl WeightedRankingAlgorithm {
    /// Create new weighted ranking algorithm
    pub fn new(weights: Vec<(RankingMetric, f64)>) -> Self {
        Self { weights }
    }

    /// Calculate composite score for an entry
    fn calculate_score(&self, entry: &LeaderboardEntry) -> f64 {
        let mut score = 0.0;

        for (metric, weight) in &self.weights {
            let value = match metric {
                RankingMetric::Latency => {
                    // Normalize: lower is better, so invert
                    1.0 / (1.0 + entry.metrics.latency_ms)
                },
                RankingMetric::Throughput => {
                    // Normalize: higher is better
                    entry.metrics.throughput.unwrap_or(0.0) / 1000.0
                },
                RankingMetric::TokensPerSecond => {
                    // Normalize: higher is better
                    entry.metrics.tokens_per_second.unwrap_or(0.0) / 10000.0
                },
                RankingMetric::Memory => {
                    // Normalize: lower is better
                    1.0 / (1.0 + entry.metrics.memory_mb.unwrap_or(1000.0))
                },
                RankingMetric::PeakMemory => {
                    // Normalize: lower is better
                    1.0 / (1.0 + entry.metrics.peak_memory_mb.unwrap_or(1000.0))
                },
                RankingMetric::GPUUtilization => {
                    // Normalize: context-dependent, assume higher is better
                    entry.metrics.gpu_utilization.unwrap_or(0.0) / 100.0
                },
                RankingMetric::Accuracy => {
                    // Already normalized (0-1)
                    entry.metrics.accuracy.unwrap_or(0.0)
                },
                RankingMetric::Energy => {
                    // Normalize: lower is better
                    1.0 / (1.0 + entry.metrics.energy_watts.unwrap_or(100.0))
                },
                RankingMetric::Date => {
                    // Normalize: newer is better (days since epoch)
                    entry.timestamp.timestamp() as f64 / 86400.0 / 20000.0
                },
                RankingMetric::Custom(name) => {
                    // Assume normalized
                    entry.metrics.custom_metrics.get(name).copied().unwrap_or(0.0)
                },
            };

            score += value * weight;
        }

        score
    }
}

impl RankingAlgorithm for WeightedRankingAlgorithm {
    fn rank(
        &self,
        entries: Vec<LeaderboardEntry>,
        _criteria: &RankingCriteria,
    ) -> Result<Vec<LeaderboardEntry>> {
        // Calculate scores
        let mut scored_entries: Vec<(f64, LeaderboardEntry)> = entries
            .into_iter()
            .map(|entry| {
                let score = self.calculate_score(&entry);
                (score, entry)
            })
            .collect();

        // Sort by score (descending)
        scored_entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        // Extract entries
        let ranked = scored_entries.into_iter().map(|(_, entry)| entry).collect();
        Ok(ranked)
    }
}

/// Elo-based ranking for head-to-head comparisons
pub struct EloRankingAlgorithm {
    k_factor: f64,
}

impl EloRankingAlgorithm {
    /// Create new Elo ranking algorithm
    pub fn new(k_factor: f64) -> Self {
        Self { k_factor }
    }

    /// Calculate Elo rating update
    pub fn calculate_rating_change(&self, winner_rating: f64, loser_rating: f64) -> (f64, f64) {
        let expected_winner = 1.0 / (1.0 + 10_f64.powf((loser_rating - winner_rating) / 400.0));
        let expected_loser = 1.0 - expected_winner;

        let winner_change = self.k_factor * (1.0 - expected_winner);
        let loser_change = self.k_factor * (0.0 - expected_loser);

        (winner_change, loser_change)
    }
}

/// Pareto frontier ranking for multi-objective optimization
pub struct ParetoRankingAlgorithm;

impl ParetoRankingAlgorithm {
    /// Check if entry a dominates entry b
    fn dominates(a: &LeaderboardEntry, b: &LeaderboardEntry) -> bool {
        let mut better_in_at_least_one = false;

        // Check key metrics
        let metrics = [
            (a.metrics.latency_ms, b.metrics.latency_ms, true), // Lower is better
            (
                a.metrics.throughput.unwrap_or(0.0),
                b.metrics.throughput.unwrap_or(0.0),
                false,
            ), // Higher is better
            (
                a.metrics.memory_mb.unwrap_or(f64::MAX),
                b.metrics.memory_mb.unwrap_or(f64::MAX),
                true,
            ), // Lower is better
            (
                a.metrics.accuracy.unwrap_or(0.0),
                b.metrics.accuracy.unwrap_or(0.0),
                false,
            ), // Higher is better
        ];

        for (a_val, b_val, lower_is_better) in metrics {
            let comparison = if lower_is_better { a_val <= b_val } else { a_val >= b_val };

            if !comparison {
                return false; // a is worse in this metric
            }

            if (lower_is_better && a_val < b_val) || (!lower_is_better && a_val > b_val) {
                better_in_at_least_one = true;
            }
        }

        better_in_at_least_one
    }
}

impl RankingAlgorithm for ParetoRankingAlgorithm {
    fn rank(
        &self,
        entries: Vec<LeaderboardEntry>,
        _criteria: &RankingCriteria,
    ) -> Result<Vec<LeaderboardEntry>> {
        let mut ranked = Vec::new();
        let mut remaining = entries;

        while !remaining.is_empty() {
            // Find non-dominated solutions (Pareto frontier)
            let mut frontier = Vec::new();
            let mut dominated = Vec::new();

            for entry in remaining {
                let mut is_dominated = false;

                for other in &frontier {
                    if Self::dominates(other, &entry) {
                        is_dominated = true;
                        break;
                    }
                }

                if !is_dominated {
                    // Remove any entries in frontier that are dominated by this entry
                    frontier.retain(|other| !Self::dominates(&entry, other));
                    frontier.push(entry);
                } else {
                    dominated.push(entry);
                }
            }

            // Sort frontier by primary metric
            frontier.sort_by(|a, b| {
                a.metrics
                    .latency_ms
                    .partial_cmp(&b.metrics.latency_ms)
                    .unwrap_or(Ordering::Equal)
            });

            ranked.extend(frontier);
            remaining = dominated;
        }

        Ok(ranked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaderboard::*;
    use std::collections::HashMap;

    fn create_test_entry(latency: f64, throughput: f64) -> LeaderboardEntry {
        LeaderboardEntry {
            id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            model_name: "test".to_string(),
            model_version: "1.0".to_string(),
            benchmark_name: "test".to_string(),
            category: LeaderboardCategory::Inference,
            hardware: HardwareInfo {
                cpu: "Test".to_string(),
                cpu_cores: 8,
                gpu: None,
                gpu_count: None,
                memory_gb: 16.0,
                accelerator: None,
                platform: "test".to_string(),
            },
            software: SoftwareInfo {
                framework_version: "0.1.0".to_string(),
                rust_version: "1.75".to_string(),
                os: "Test".to_string(),
                optimization_level: OptimizationLevel::O2,
                precision: Precision::FP32,
                quantization: None,
                compiler_flags: vec![],
            },
            metrics: PerformanceMetrics {
                latency_ms: latency,
                latency_percentiles: LatencyPercentiles {
                    p50: latency * 0.9,
                    p90: latency * 1.1,
                    p95: latency * 1.2,
                    p99: latency * 1.5,
                    p999: latency * 2.0,
                },
                throughput: Some(throughput),
                tokens_per_second: None,
                memory_mb: None,
                peak_memory_mb: None,
                gpu_utilization: None,
                accuracy: None,
                energy_watts: None,
                custom_metrics: HashMap::new(),
            },
            metadata: HashMap::new(),
            validated: true,
            submitter: SubmitterInfo {
                name: "Test".to_string(),
                organization: None,
                email: None,
                github: None,
            },
            tags: vec![],
        }
    }

    #[test]
    fn test_ranking_criteria_compare() {
        let criteria = RankingCriteria {
            primary_metric: RankingMetric::Latency,
            secondary_metric: Some(RankingMetric::Throughput),
            order: SortOrder::Ascending,
        };

        let entry1 = create_test_entry(10.0, 100.0);
        let entry2 = create_test_entry(20.0, 50.0);

        // Entry1 has lower latency, so it should come first
        assert_eq!(criteria.compare(&entry1, &entry2), Ordering::Less);
    }

    #[test]
    fn test_default_ranking_algorithm() {
        let algo = DefaultRankingAlgorithm;
        let criteria = RankingCriteria::default();

        let entries = vec![
            create_test_entry(20.0, 50.0),
            create_test_entry(10.0, 100.0),
            create_test_entry(30.0, 33.3),
        ];

        let ranked = algo.rank(entries, &criteria).unwrap();

        // Should be sorted by latency (ascending)
        assert_eq!(ranked[0].metrics.latency_ms, 10.0);
        assert_eq!(ranked[1].metrics.latency_ms, 20.0);
        assert_eq!(ranked[2].metrics.latency_ms, 30.0);
    }

    #[test]
    fn test_weighted_ranking() {
        let algo = WeightedRankingAlgorithm::new(vec![
            (RankingMetric::Latency, 0.6),
            (RankingMetric::Throughput, 0.4),
        ]);

        let entries = vec![
            create_test_entry(20.0, 200.0), // Moderate latency, high throughput
            create_test_entry(10.0, 50.0),  // Low latency, low throughput
            create_test_entry(30.0, 150.0), // High latency, moderate throughput
        ];

        let ranked = algo.rank(entries, &RankingCriteria::default()).unwrap();

        // The weighted score should balance both metrics
        assert!(ranked.len() == 3);
    }

    #[test]
    fn test_elo_rating_calculation() {
        let elo = EloRankingAlgorithm::new(32.0);

        let (winner_change, loser_change) = elo.calculate_rating_change(1500.0, 1500.0);

        // Equal ratings should result in symmetric changes
        assert!((winner_change - 16.0).abs() < 0.1);
        assert!((loser_change + 16.0).abs() < 0.1);
    }
}
