//! System memory collection utilities
//!
//! Every figure produced here is a **real measurement** taken from the operating
//! system through the pure-Rust [`sysinfo`] crate. Nothing is simulated: when a
//! value cannot be obtained on the current platform the corresponding field is
//! `None` (or the call returns an error) rather than a plausible-looking constant.

use anyhow::{anyhow, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use sysinfo::{MemoryRefreshKind, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System};

use super::types::*;

const BYTES_PER_MB: f64 = 1024.0 * 1024.0;

/// GPU memory information.
///
/// Only produced when a GPU backend is compiled in; there is deliberately no
/// fallback that invents a device.
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub total_mb: f64,
    pub used_mb: f64,
    pub free_mb: f64,
}

/// Measured memory footprint of the current process.
#[derive(Debug, Clone)]
pub struct ProcessMemoryInfo {
    /// Resident set size (physical memory currently held), in MB.
    pub resident_mb: f64,
    /// Virtual memory size, in MB.
    pub virtual_mb: f64,
    /// Largest resident set size observed by this process since start-up, in MB.
    ///
    /// This is a running maximum over the samples this profiler has taken, not a
    /// kernel-reported high-water mark.
    pub peak_resident_mb: f64,
}

/// Counts of allocations the caller has explicitly registered with the profiler.
///
/// The profiler cannot see allocations it was not told about, so these counters
/// describe *tracked* allocations only.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrackedAllocationStats {
    /// Cumulative number of allocations registered.
    pub allocated_objects: u64,
    /// Cumulative number of deallocations registered.
    pub deallocated_objects: u64,
    /// Allocations currently registered as live.
    pub active_allocations: u64,
    /// Total size of the live allocations, in bytes.
    pub active_bytes: u64,
}

/// Running maximum of the resident set size, in bytes.
static PEAK_RESIDENT_BYTES: AtomicU64 = AtomicU64::new(0);

/// Shared `sysinfo` handle.
///
/// `System` caches per-process bookkeeping between refreshes, so a single
/// long-lived instance is both cheaper and more accurate than creating one per
/// sample.
fn system_handle() -> &'static Mutex<System> {
    static SYSTEM: std::sync::OnceLock<Mutex<System>> = std::sync::OnceLock::new();
    SYSTEM.get_or_init(|| {
        Mutex::new(System::new_with_specifics(
            RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
        ))
    })
}

impl super::profiler::MemoryProfiler {
    /// Collect current memory metrics from the operating system.
    ///
    /// `tracked` carries the allocation counters the profiler maintains for
    /// allocations that were explicitly registered by the caller.
    pub async fn collect_memory_metrics(tracked: TrackedAllocationStats) -> Result<MemoryMetrics> {
        let process_info = Self::get_process_memory_info()?;
        let gpu_info = Self::get_gpu_memory_info().await?;

        Ok(MemoryMetrics {
            timestamp: std::time::SystemTime::now(),
            total_memory_mb: process_info.resident_mb,
            virtual_memory_mb: process_info.virtual_mb,
            heap_memory_mb: None,
            stack_memory_mb: None,
            gpu_memory_mb: gpu_info.map(|info| info.used_mb),
            peak_memory_mb: process_info.peak_resident_mb,
            allocated_objects: tracked.allocated_objects,
            deallocated_objects: tracked.deallocated_objects,
            active_allocations: tracked.active_allocations,
            memory_fragmentation_ratio: Self::calculate_fragmentation_ratio(&process_info, tracked),
            memory_growth_rate_mb_per_sec: 0.0, // Filled in by the sampling loop
        })
    }

    /// Read the machine's memory statistics.
    ///
    /// `cached_memory` and `buffer_memory` are `None`: they are not exposed
    /// portably, and reporting a guess would be worse than reporting nothing.
    pub fn get_system_memory_info() -> Result<SystemMemoryInfo> {
        let mut system = system_handle()
            .lock()
            .map_err(|_| anyhow!("system information handle is poisoned"))?;
        system.refresh_memory();

        let total_memory = system.total_memory();
        if total_memory == 0 {
            return Err(anyhow!(
                "the operating system reported a total memory of 0 bytes; \
                 memory statistics are unavailable on this platform"
            ));
        }

        Ok(SystemMemoryInfo {
            total_memory,
            available_memory: system.available_memory(),
            used_memory: system.used_memory(),
            free_memory: system.free_memory(),
            cached_memory: None,
            buffer_memory: None,
        })
    }

    /// Read this process's resident and virtual memory footprint.
    pub fn get_process_memory_info() -> Result<ProcessMemoryInfo> {
        let pid = sysinfo::get_current_pid()
            .map_err(|e| anyhow!("failed to determine the current process id: {e}"))?;

        let mut system = system_handle()
            .lock()
            .map_err(|_| anyhow!("system information handle is poisoned"))?;
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_memory(),
        );

        let process = system
            .process(pid)
            .ok_or_else(|| anyhow!("the operating system reported no process with id {pid}"))?;

        let resident_bytes = process.memory();
        let virtual_bytes = process.virtual_memory();
        drop(system);

        if resident_bytes == 0 {
            return Err(anyhow!(
                "the operating system reported a resident set size of 0 bytes for this process; \
                 process memory statistics are unavailable on this platform"
            ));
        }

        let peak_bytes = PEAK_RESIDENT_BYTES.fetch_max(resident_bytes, Ordering::Relaxed);

        Ok(ProcessMemoryInfo {
            resident_mb: resident_bytes as f64 / BYTES_PER_MB,
            virtual_mb: virtual_bytes as f64 / BYTES_PER_MB,
            peak_resident_mb: peak_bytes.max(resident_bytes) as f64 / BYTES_PER_MB,
        })
    }

    /// Read GPU memory usage.
    ///
    /// Returns `Ok(None)` when no GPU backend is compiled in — this crate has no
    /// way to query a device that is not there, and a fabricated card would make
    /// every downstream report fiction.
    pub async fn get_gpu_memory_info() -> Result<Option<GpuMemoryInfo>> {
        Ok(None)
    }

    /// Share of the resident set that is *not* covered by tracked allocations.
    ///
    /// With no registered allocations there is nothing to compare the resident
    /// set against, so the ratio is `0.0` rather than an invented percentage.
    pub fn calculate_fragmentation_ratio(
        info: &ProcessMemoryInfo,
        tracked: TrackedAllocationStats,
    ) -> f64 {
        if tracked.active_bytes == 0 || info.resident_mb <= 0.0 {
            return 0.0;
        }
        let tracked_mb = tracked.active_bytes as f64 / BYTES_PER_MB;
        ((info.resident_mb - tracked_mb) / info.resident_mb).clamp(0.0, 1.0)
    }

    /// Update adaptive thresholds based on current metrics
    pub async fn update_adaptive_thresholds(
        current_metrics: &MemoryMetrics,
        adaptive_thresholds: &std::sync::Arc<
            std::sync::Mutex<super::analytics::AdaptiveThresholds>,
        >,
    ) {
        // Update thresholds every 5 minutes
        let should_update = {
            let thresholds = match adaptive_thresholds.lock() {
                Ok(guard) => guard,
                Err(_) => return, // Skip update on lock failure
            };
            current_metrics
                .timestamp
                .duration_since(thresholds.last_updated)
                .unwrap_or_default()
                .as_secs()
                > 300
        };

        if should_update {
            let mut thresholds = match adaptive_thresholds.lock() {
                Ok(guard) => guard,
                Err(_) => return, // Skip update on lock failure
            };
            // Simple adaptive update based on current usage
            let adaptation_factor = 0.1;
            thresholds.base_memory_threshold = thresholds.base_memory_threshold
                * (1.0 - adaptation_factor)
                + current_metrics.total_memory_mb * 1.2 * adaptation_factor;
            thresholds.growth_rate_threshold = thresholds.growth_rate_threshold
                * (1.0 - adaptation_factor)
                + current_metrics.memory_growth_rate_mb_per_sec.abs() * 2.0 * adaptation_factor;
            thresholds.fragmentation_threshold = thresholds.fragmentation_threshold
                * (1.0 - adaptation_factor)
                + current_metrics.memory_fragmentation_ratio * 1.5 * adaptation_factor;
            thresholds.last_updated = current_metrics.timestamp;
        }
    }

    /// Detect memory leaks using heuristics
    pub async fn detect_memory_leaks(
        current_metrics: &MemoryMetrics,
        previous_metrics: &Option<MemoryMetrics>,
        leak_detection: &std::sync::Arc<
            std::sync::Mutex<super::analytics::LeakDetectionHeuristics>,
        >,
        alerts: &std::sync::Arc<std::sync::Mutex<Vec<MemoryAlert>>>,
        cached_recommendations: &super::analytics::AlertRecommendations,
    ) {
        if let Some(_prev) = previous_metrics {
            let growth_rate = current_metrics.memory_growth_rate_mb_per_sec;
            let detection = match leak_detection.lock() {
                Ok(guard) => guard,
                Err(_) => return, // Skip detection on lock failure
            };

            // Check for sustained growth
            if growth_rate > detection.sustained_growth_threshold {
                let mut alerts_vec = match alerts.lock() {
                    Ok(guard) => guard,
                    Err(_) => return, // Skip alert on lock failure
                };
                alerts_vec.push(MemoryAlert {
                    id: uuid::Uuid::new_v4(),
                    timestamp: current_metrics.timestamp,
                    alert_type: MemoryAlertType::MemoryLeak,
                    severity: if growth_rate > detection.sustained_growth_threshold * 2.0 {
                        AlertSeverity::Critical
                    } else {
                        AlertSeverity::Warning
                    },
                    message: format!(
                        "Potential memory leak detected: {:.2} MB/sec growth",
                        growth_rate
                    ),
                    details: std::collections::HashMap::new(),
                    recommendations: cached_recommendations.memory_leak.clone(),
                });
            }
        }
    }

    /// Update memory prediction
    pub async fn update_memory_prediction(
        _current_metrics: &MemoryMetrics,
        memory_predictor: &std::sync::Arc<std::sync::Mutex<super::analytics::MemoryPredictor>>,
        metrics_history: &std::sync::Arc<
            std::sync::Mutex<std::collections::VecDeque<MemoryMetrics>>,
        >,
    ) {
        let history = match metrics_history.lock() {
            Ok(guard) => guard,
            Err(_) => return, // Skip prediction on lock failure
        };
        if history.len() >= 10 {
            let recent_metrics: Vec<MemoryMetrics> = history.iter().cloned().collect();
            drop(history);

            let mut predictor = match memory_predictor.lock() {
                Ok(guard) => guard,
                Err(_) => return, // Skip prediction on lock failure
            };
            let _prediction = predictor.predict_memory_usage(&recent_metrics, Some(300));
            // 5 minutes
            // Prediction result would be stored or used for alerts in a real implementation
        }
    }

    /// Analyze for alerts with adaptive thresholds
    pub async fn analyze_for_alerts_adaptive(
        current: &MemoryMetrics,
        _previous: &Option<MemoryMetrics>,
        alerts: &std::sync::Arc<std::sync::Mutex<Vec<MemoryAlert>>>,
        _config: &super::types::ProfilerConfig,
        cached_recommendations: &super::analytics::AlertRecommendations,
        adaptive_thresholds: &std::sync::Arc<
            std::sync::Mutex<super::analytics::AdaptiveThresholds>,
        >,
    ) {
        let thresholds = match adaptive_thresholds.lock() {
            Ok(guard) => guard,
            Err(_) => return, // Skip analysis on lock failure
        };
        let mut new_alerts = Vec::new();

        // High memory usage alert using adaptive threshold
        if current.total_memory_mb > thresholds.base_memory_threshold {
            new_alerts.push(MemoryAlert {
                id: uuid::Uuid::new_v4(),
                timestamp: current.timestamp,
                alert_type: MemoryAlertType::HighMemoryUsage,
                severity: if current.total_memory_mb > thresholds.base_memory_threshold * 1.5 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                },
                message: format!(
                    "Memory usage is {:.2}MB, exceeding adaptive threshold of {:.2}MB",
                    current.total_memory_mb, thresholds.base_memory_threshold
                ),
                details: std::collections::HashMap::new(),
                recommendations: cached_recommendations.high_memory.clone(),
            });
        }

        // Rapid growth alert
        if current.memory_growth_rate_mb_per_sec > thresholds.growth_rate_threshold {
            new_alerts.push(MemoryAlert {
                id: uuid::Uuid::new_v4(),
                timestamp: current.timestamp,
                alert_type: MemoryAlertType::RapidGrowth,
                severity: AlertSeverity::Warning,
                message: format!(
                    "Rapid memory growth detected: {:.2}MB/sec",
                    current.memory_growth_rate_mb_per_sec
                ),
                details: std::collections::HashMap::new(),
                recommendations: cached_recommendations.rapid_growth.clone(),
            });
        }

        // Fragmentation alert
        if current.memory_fragmentation_ratio > thresholds.fragmentation_threshold {
            new_alerts.push(MemoryAlert {
                id: uuid::Uuid::new_v4(),
                timestamp: current.timestamp,
                alert_type: MemoryAlertType::FragmentationHigh,
                severity: AlertSeverity::Info,
                message: format!(
                    "High memory fragmentation: {:.2}%",
                    current.memory_fragmentation_ratio * 100.0
                ),
                details: std::collections::HashMap::new(),
                recommendations: cached_recommendations.fragmentation.clone(),
            });
        }

        // Store alerts if any were generated
        if !new_alerts.is_empty() {
            let mut alerts_vec = match alerts.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    drop(thresholds);
                    return; // Skip storing alerts on lock failure
                },
            };
            alerts_vec.extend(new_alerts);

            // Keep only recent alerts (last 1000)
            if alerts_vec.len() > 1000 {
                let excess = alerts_vec.len() - 1000;
                alerts_vec.drain(0..excess);
            }
        }
        drop(thresholds);
    }

    /// Update memory patterns
    pub async fn update_patterns(
        current_metrics: &MemoryMetrics,
        patterns: &std::sync::Arc<std::sync::Mutex<Vec<MemoryPattern>>>,
    ) {
        let mut patterns_vec = match patterns.lock() {
            Ok(guard) => guard,
            Err(_) => return, // Skip pattern update on lock failure
        };

        // Simple pattern detection - in practice this would be much more sophisticated
        if current_metrics.total_memory_mb > 1000.0 {
            // Check if we already have a large allocation pattern
            let has_large_pattern = patterns_vec
                .iter()
                .any(|p| matches!(p.pattern_type, PatternType::LargeAllocations));

            if !has_large_pattern {
                patterns_vec.push(MemoryPattern {
                    pattern_type: PatternType::LargeAllocations,
                    frequency: 1,
                    average_size: current_metrics.total_memory_mb,
                    total_size: current_metrics.total_memory_mb,
                    locations: vec!["system".to_string()],
                    trend: PatternTrend::Increasing,
                });
            } else {
                // Update existing pattern
                if let Some(pattern) = patterns_vec
                    .iter_mut()
                    .find(|p| matches!(p.pattern_type, PatternType::LargeAllocations))
                {
                    pattern.frequency += 1;
                    pattern.average_size =
                        (pattern.average_size + current_metrics.total_memory_mb) / 2.0;
                    pattern.total_size += current_metrics.total_memory_mb;
                }
            }
        }

        // Keep patterns list bounded
        if patterns_vec.len() > 100 {
            let excess = patterns_vec.len() - 100;
            patterns_vec.drain(0..excess);
        }
    }
}
