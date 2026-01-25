//! Main memory profiler implementation
//!
//! This module contains the core MemoryProfiler struct and its implementation
//! for real-time memory monitoring and analysis.

use anyhow::{Context, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};
use uuid::Uuid;

use super::analytics::*;
use super::types::*;

/// Main memory profiler
pub struct MemoryProfiler {
    config: ProfilerConfig,
    metrics_history: Arc<Mutex<VecDeque<MemoryMetrics>>>,
    allocations: Arc<Mutex<HashMap<Uuid, AllocationInfo>>>,
    alerts: Arc<Mutex<Vec<MemoryAlert>>>,
    patterns: Arc<Mutex<Vec<MemoryPattern>>>,
    is_monitoring: Arc<AtomicBool>,
    start_time: Option<Instant>,
    // Performance optimization: pre-allocated alert messages to reduce allocations
    cached_recommendations: AlertRecommendations,
    // Advanced analytics components
    adaptive_thresholds: Arc<Mutex<AdaptiveThresholds>>,
    leak_detection: Arc<Mutex<LeakDetectionHeuristics>>,
    memory_predictor: Arc<Mutex<MemoryPredictor>>,
    // Performance counters for monitoring overhead
    monitoring_overhead_us: Arc<AtomicU64>,
    total_collections: Arc<AtomicU64>,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new(config: ProfilerConfig) -> Result<Self> {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&config.output_dir).context("Failed to create output directory")?;

        Ok(Self {
            config,
            metrics_history: Arc::new(Mutex::new(VecDeque::new())),
            allocations: Arc::new(Mutex::new(HashMap::new())),
            alerts: Arc::new(Mutex::new(Vec::new())),
            patterns: Arc::new(Mutex::new(Vec::new())),
            is_monitoring: Arc::new(AtomicBool::new(false)),
            start_time: None,
            cached_recommendations: AlertRecommendations::new(),
            adaptive_thresholds: Arc::new(Mutex::new(AdaptiveThresholds::default())),
            leak_detection: Arc::new(Mutex::new(LeakDetectionHeuristics::default())),
            memory_predictor: Arc::new(Mutex::new(MemoryPredictor::default())),
            monitoring_overhead_us: Arc::new(AtomicU64::new(0)),
            total_collections: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Start memory monitoring
    pub async fn start_monitoring(&mut self) -> Result<()> {
        // Performance optimization: Use atomic compare-and-swap instead of mutex
        if self
            .is_monitoring
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Ok(());
        }

        self.start_time = Some(Instant::now());

        let metrics_history = self.metrics_history.clone();
        let _allocations = self.allocations.clone();
        let alerts = self.alerts.clone();
        let patterns = self.patterns.clone();
        let config = self.config.clone();
        let is_monitoring = self.is_monitoring.clone();
        let cached_recommendations = self.cached_recommendations.clone();
        let adaptive_thresholds = self.adaptive_thresholds.clone();
        let leak_detection = self.leak_detection.clone();
        let memory_predictor = self.memory_predictor.clone();
        let monitoring_overhead_us = self.monitoring_overhead_us.clone();
        let total_collections = self.total_collections.clone();

        // Spawn monitoring task
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_millis(config.collection_interval_ms));
            let mut previous_metrics: Option<MemoryMetrics> = None;

            while is_monitoring.load(Ordering::Relaxed) {
                interval.tick().await;

                // Track monitoring performance
                let monitoring_start = Instant::now();
                total_collections.fetch_add(1, Ordering::Relaxed);

                // Collect current memory metrics
                let current_metrics = match Self::collect_memory_metrics().await {
                    Ok(metrics) => metrics,
                    Err(e) => {
                        eprintln!("Failed to collect memory metrics: {}", e);
                        continue;
                    },
                };

                // Calculate growth rate
                let growth_rate = if let Some(ref prev) = previous_metrics {
                    let time_diff = current_metrics
                        .timestamp
                        .duration_since(prev.timestamp)
                        .unwrap_or_default()
                        .as_secs_f64();

                    if time_diff > 0.0 {
                        (current_metrics.total_memory_mb - prev.total_memory_mb) / time_diff
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let mut final_metrics = current_metrics;
                final_metrics.memory_growth_rate_mb_per_sec = growth_rate;

                // Store metrics
                {
                    let mut history = metrics_history.lock().expect("operation failed");
                    history.push_back(final_metrics.clone());

                    // Keep only max_data_points
                    while history.len() > config.max_data_points {
                        history.pop_front();
                    }
                }

                // Advanced analytics: Update adaptive thresholds
                Self::update_adaptive_thresholds(&final_metrics, &adaptive_thresholds).await;

                // Advanced analytics: Memory leak detection
                if config.enable_leak_detection {
                    Self::detect_memory_leaks(
                        &final_metrics,
                        &previous_metrics,
                        &leak_detection,
                        &alerts,
                        &cached_recommendations,
                    )
                    .await;
                }

                // Advanced analytics: Memory prediction
                Self::update_memory_prediction(&final_metrics, &memory_predictor, &metrics_history)
                    .await;

                // Analyze for alerts with adaptive thresholds
                Self::analyze_for_alerts_adaptive(
                    &final_metrics,
                    &previous_metrics,
                    &alerts,
                    &config,
                    &cached_recommendations,
                    &adaptive_thresholds,
                )
                .await;

                // Update patterns if enabled
                if config.enable_pattern_analysis {
                    Self::update_patterns(&final_metrics, &patterns).await;
                }

                // Track monitoring overhead
                let monitoring_duration = monitoring_start.elapsed();
                monitoring_overhead_us
                    .store(monitoring_duration.as_micros() as u64, Ordering::Relaxed);

                previous_metrics = Some(final_metrics);
            }
        });

        Ok(())
    }

    /// Stop memory monitoring
    pub async fn stop_monitoring(&self) -> Result<()> {
        self.is_monitoring.store(false, Ordering::SeqCst);
        Ok(())
    }

    /// Get current metrics snapshot
    pub async fn get_current_metrics(&self) -> Result<Option<MemoryMetrics>> {
        let history = self.metrics_history.lock().expect("operation failed");
        Ok(history.back().cloned())
    }

    /// Get all alerts
    pub async fn get_alerts(&self) -> Result<Vec<MemoryAlert>> {
        let alerts = self.alerts.lock().expect("operation failed");
        Ok(alerts.clone())
    }

    /// Get memory patterns
    pub async fn get_patterns(&self) -> Result<Vec<MemoryPattern>> {
        let patterns = self.patterns.lock().expect("operation failed");
        Ok(patterns.clone())
    }

    /// Get adaptive thresholds
    pub async fn get_adaptive_thresholds(&self) -> Result<AdaptiveThresholds> {
        let thresholds = self.adaptive_thresholds.lock().expect("operation failed");
        Ok(thresholds.clone())
    }

    /// Get monitoring statistics
    pub async fn get_monitoring_stats(&self) -> Result<MonitoringStats> {
        let total_collections = self.total_collections.load(Ordering::Relaxed);
        let overhead_us = self.monitoring_overhead_us.load(Ordering::Relaxed);

        let average_overhead_us =
            if total_collections > 0 { overhead_us / total_collections } else { 0 };

        let uptime_secs = self.start_time.map(|start| start.elapsed().as_secs()).unwrap_or(0);

        Ok(MonitoringStats {
            total_collections,
            average_overhead_us,
            uptime_secs,
        })
    }

    /// Configure leak detection parameters
    pub async fn configure_leak_detection(&self, config: LeakDetectionConfig) -> Result<()> {
        let mut detection = self.leak_detection.lock().expect("operation failed");
        *detection = LeakDetectionHeuristics {
            sustained_growth_threshold: config.growth_threshold,
            growth_duration_threshold: Duration::from_secs(config.duration_secs),
            allocation_pattern_threshold: config.allocation_threshold,
            false_positive_filter: config.confidence_threshold,
        };
        Ok(())
    }

    /// Get current leak detection configuration
    pub async fn get_leak_detection_config(&self) -> Result<LeakDetectionConfig> {
        let detection = self.leak_detection.lock().expect("operation failed");
        Ok(LeakDetectionConfig {
            growth_threshold: detection.sustained_growth_threshold,
            duration_secs: detection.growth_duration_threshold.as_secs(),
            allocation_threshold: detection.allocation_pattern_threshold,
            confidence_threshold: detection.false_positive_filter,
        })
    }

    /// Predict memory usage for given horizon
    pub async fn predict_memory_usage(
        &self,
        horizon_secs: u64,
    ) -> Result<Option<MemoryPrediction>> {
        let metrics_history = self.metrics_history.lock().expect("operation failed");
        let metrics: Vec<MemoryMetrics> = metrics_history.iter().cloned().collect();
        drop(metrics_history);

        let mut predictor = self.memory_predictor.lock().expect("operation failed");
        Ok(predictor.predict_memory_usage(&metrics, Some(horizon_secs)))
    }

    /// Get comprehensive analytics summary
    pub async fn get_analytics_summary(&self) -> Result<AnalyticsSummary> {
        let thresholds = self.get_adaptive_thresholds().await?;
        let prediction = self.predict_memory_usage(300).await?; // 5 minutes
        let monitoring_stats = self.get_monitoring_stats().await?;
        let leak_config = self.get_leak_detection_config().await?;

        Ok(AnalyticsSummary {
            adaptive_thresholds: thresholds,
            memory_prediction: prediction,
            monitoring_stats,
            leak_detection_config: leak_config,
        })
    }

    /// Get access to metrics history for reporting
    pub(crate) fn get_metrics_history(
        &self,
    ) -> &Arc<Mutex<std::collections::VecDeque<super::types::MemoryMetrics>>> {
        &self.metrics_history
    }

    /// Get access to allocations for reporting
    pub(crate) fn get_allocations(
        &self,
    ) -> &Arc<Mutex<HashMap<uuid::Uuid, super::types::AllocationInfo>>> {
        &self.allocations
    }

    /// Get access to alerts for reporting
    pub(crate) fn get_alerts_internal(&self) -> &Arc<Mutex<Vec<super::types::MemoryAlert>>> {
        &self.alerts
    }

    /// Get access to patterns for reporting
    pub(crate) fn get_patterns_internal(&self) -> &Arc<Mutex<Vec<super::types::MemoryPattern>>> {
        &self.patterns
    }

    /// Get start time for reporting
    pub(crate) fn get_start_time(&self) -> Option<std::time::Instant> {
        self.start_time
    }

    /// Get config for reporting
    pub(crate) fn get_config(&self) -> &super::types::ProfilerConfig {
        &self.config
    }

    /// Check if monitoring is active (for testing)
    #[cfg(test)]
    pub fn is_monitoring(&self) -> bool {
        self.is_monitoring.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get cached recommendations (for testing)
    #[cfg(test)]
    pub fn get_cached_recommendations(&self) -> &super::analytics::AlertRecommendations {
        &self.cached_recommendations
    }

    /// Get adaptive thresholds (for testing)
    #[cfg(test)]
    pub fn get_adaptive_thresholds_internal(
        &self,
    ) -> &Arc<Mutex<super::analytics::AdaptiveThresholds>> {
        &self.adaptive_thresholds
    }

    /// Get memory predictor (for testing)
    #[cfg(test)]
    pub fn get_memory_predictor_internal(&self) -> &Arc<Mutex<super::analytics::MemoryPredictor>> {
        &self.memory_predictor
    }
}

#[derive(Debug, Clone)]
pub struct MonitoringStats {
    pub total_collections: u64,
    pub average_overhead_us: u64,
    pub uptime_secs: u64,
}

#[derive(Debug, Clone)]
pub struct LeakDetectionConfig {
    pub growth_threshold: f64,
    pub duration_secs: u64,
    pub allocation_threshold: usize,
    pub confidence_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct AnalyticsSummary {
    pub adaptive_thresholds: AdaptiveThresholds,
    pub memory_prediction: Option<MemoryPrediction>,
    pub monitoring_stats: MonitoringStats,
    pub leak_detection_config: LeakDetectionConfig,
}
