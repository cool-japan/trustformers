//! Mobile Performance Metrics Collector
//!
//! This module provides comprehensive metrics collection capabilities for mobile ML inference
//! performance monitoring. It includes real-time collection of memory, CPU, GPU, network,
//! thermal, battery, and inference metrics with platform-specific implementations for iOS
//! and Android devices.
//!
//! # Features
//!
//! - **Multi-platform Support**: Optimized collection for iOS and Android platforms
//! - **Real-time Collection**: Continuous monitoring with configurable sampling rates
//! - **Comprehensive Metrics**: Memory, CPU, GPU, network, thermal, battery, and inference metrics
//! - **Thread Safety**: Atomic operations and thread-safe collection
//! - **Error Resilience**: Robust error handling and graceful degradation
//! - **Performance Optimized**: Minimal overhead collection with adaptive sampling
//! - **Historical Tracking**: Configurable history retention with memory management
//!
//! # Usage
//!
//! ```rust
//! # fn main() -> trustformers_core::error::Result<()> {
//! use trustformers_mobile::mobile_performance_profiler::collector::MobileMetricsCollector;
//! use trustformers_mobile::mobile_performance_profiler::config::MobileProfilerConfig;
//!
//! // Create collector with configuration
//! let config = MobileProfilerConfig::default();
//! let collector = MobileMetricsCollector::new(config)?;
//!
//! // Start collection
//! collector.start_collection()?;
//!
//! // Collect metrics
//! let snapshot = collector.get_current_snapshot()?;
//!
//! // Stop collection
//! collector.stop_collection()?;
//! # let _ = snapshot;
//! # Ok(())
//! # }
//! ```
//!
//! # Platform-specific Features
//!
//! ## iOS
//! - Metal GPU performance monitoring
//! - Core ML inference metrics
//! - iOS memory pressure detection
//! - Thermal state monitoring via NSProcessInfo
//! - Battery metrics via UIDevice
//!
//! ## Android
//! - NNAPI performance tracking
//! - GPU delegate statistics
//! - Android memory management metrics
//! - Doze mode status monitoring
//! - System service utilization

use super::config::MobileProfilerConfig;
use super::types::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::TrustformersError;

// Import libc for platform-specific system calls
#[cfg(any(target_os = "ios", target_os = "android"))]
extern crate libc;

/// Collection error types specific to metrics collection
#[derive(Debug, thiserror::Error)]
pub enum CollectionError {
    /// System resource unavailable
    #[error("System resource unavailable: {0}")]
    ResourceUnavailable(String),

    /// Permission denied for metric collection
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// Platform feature not supported
    #[error("Platform feature not supported: {0}")]
    PlatformNotSupported(String),

    /// Collection timeout
    #[error("Collection operation timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Collection internal error
    #[error("Internal collection error: {0}")]
    Internal(String),
}

/// Mobile metrics collector for comprehensive performance monitoring
///
/// The `MobileMetricsCollector` provides thread-safe, real-time collection of performance
/// metrics across multiple system components. It supports platform-specific optimizations
/// and handles system resource limitations gracefully.
///
/// # Thread Safety
///
/// All public methods are thread-safe. The collector uses internal locking to ensure
/// data consistency across concurrent access patterns.
///
/// # Memory Management
///
/// The collector automatically manages memory usage by:
/// - Limiting historical data based on configuration
/// - Using efficient data structures for metric storage
/// - Providing memory usage estimation and monitoring
///
/// # Error Handling
///
/// Collection operations are designed to be resilient:
/// - Individual metric collection failures don't stop other metrics
/// - Graceful degradation when system resources are unavailable
/// - Detailed error reporting for debugging and monitoring
#[derive(Debug)]
pub struct MobileMetricsCollector {
    /// Collector configuration
    config: Arc<RwLock<MobileProfilerConfig>>,

    /// Current metrics snapshot
    current_metrics: Arc<RwLock<MobileMetricsSnapshot>>,

    /// Historical metrics data with thread-safe access
    metrics_history: Arc<Mutex<VecDeque<MobileMetricsSnapshot>>>,

    /// Collection state tracking
    collection_state: Arc<RwLock<CollectionState>>,

    /// Real system metrics source
    platform_collector: Arc<SystemMetricsCollector>,

    /// Inference metrics tracker
    inference_tracker: Arc<Mutex<InferenceTracker>>,

    /// Collection statistics
    statistics: Arc<Mutex<CollectionStatistics>>,
}

/// Internal collection state
#[derive(Debug, Clone)]
struct CollectionState {
    /// Whether collection is active
    is_collecting: bool,

    /// Collection start time
    collection_start: Option<Instant>,

    /// Last collection time
    last_collection: Option<Instant>,

    /// Total samples collected
    total_samples: u64,

    /// Collection errors count
    error_count: u64,

    /// Average collection time
    avg_collection_time_ms: f64,
}

/// Inference performance tracker
#[derive(Debug)]
struct InferenceTracker {
    /// Active inference sessions
    active_inferences: HashMap<String, InferenceSession>,

    /// Completed inference metrics
    completed_inferences: VecDeque<CompletedInference>,

    /// Model load times tracking
    model_load_times: HashMap<String, f64>,

    /// Cache performance tracking
    cache_stats: CacheStats,
}

/// Individual inference session tracking
#[derive(Debug, Clone)]
struct InferenceSession {
    /// Session ID
    id: String,

    /// Model name
    model_name: String,

    /// Start time
    start_time: Instant,

    /// Initial system metrics
    initial_metrics: SystemSnapshot,
}

/// Completed inference record
#[derive(Debug, Clone)]
struct CompletedInference {
    /// Session ID
    id: String,

    /// Model name
    model_name: String,

    /// Inference duration
    duration_ms: f64,

    /// Success status
    success: bool,

    /// Memory delta
    memory_delta_mb: Option<f32>,

    /// CPU usage during inference
    cpu_usage_percent: Option<f32>,

    /// GPU usage during inference
    gpu_usage_percent: Option<f32>,

    /// Completion timestamp
    timestamp: u64,
}

/// Cache performance statistics
#[derive(Debug, Clone)]
struct CacheStats {
    /// Total cache requests
    total_requests: u64,

    /// Cache hits
    cache_hits: u64,

    /// Cache misses
    cache_misses: u64,

    /// Average hit latency
    avg_hit_latency_ms: f64,

    /// Average miss latency
    avg_miss_latency_ms: f64,
}

/// System metrics snapshot for comparison
#[derive(Debug, Clone)]
struct SystemSnapshot {
    /// Memory usage at snapshot time
    memory_usage_mb: Option<f32>,

    /// CPU usage at snapshot time
    cpu_usage_percent: Option<f32>,

    /// GPU usage at snapshot time
    gpu_usage_percent: Option<f32>,

    /// Timestamp
    timestamp: Instant,
}

/// Collection statistics and performance metrics
#[derive(Debug, Clone)]
pub struct CollectionStatistics {
    /// Total samples collected
    pub total_samples: u64,

    /// Total collection duration
    pub collection_duration: Duration,

    /// Average sampling rate (samples/second)
    pub average_sampling_rate: f64,

    /// Current history size
    pub history_size: usize,

    /// Estimated memory usage in MB
    pub current_memory_usage_mb: f32,

    /// Collection error count
    pub error_count: u64,

    /// Average collection time per sample
    pub avg_collection_time_ms: f64,

    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
}

/// Real system metrics source.
///
/// Every figure this type returns comes from a live measurement. There is a
/// single implementation rather than one per platform on purpose: the previous
/// `IOSCollector` / `AndroidCollector` / `GenericCollector` split returned
/// three different sets of invented constants (iOS "128 MB heap / 30% CPU /
/// 55% GPU", Android "96 MB / 35% / 60%", generic "64 MB / 25% / 20%") with no
/// platform call behind any of them. `sysinfo` reads the same real counters on
/// every target this crate builds for, so one implementation covers all of
/// them, and what genuinely cannot be measured is reported as absent instead
/// of being filled in.
#[derive(Debug)]
struct SystemMetricsCollector;

impl SystemMetricsCollector {
    /// Real memory figures: this process's resident set size and the system's
    /// available memory, both from `sysinfo`.
    ///
    /// The per-segment breakdown (heap/native/graphics/code/stack) has no
    /// portable source and stays `None`.
    fn collect_memory_metrics(&self) -> Result<MemoryMetrics> {
        use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

        let mut system = System::new();
        system.refresh_memory();

        // A second `System` for the process refresh, mirroring
        // `crash_reporter::collect_memory_usage`: reusing the instance that
        // already refreshed system memory yields no process on macOS.
        let pid = Pid::from_u32(std::process::id());
        let mut process_system = System::new();
        process_system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_memory(),
        );

        let bytes_to_mb = |bytes: u64| bytes as f32 / (1024.0 * 1024.0);
        let resident_mb = process_system
            .process(pid)
            .map(|process| bytes_to_mb(process.memory()))
            .ok_or_else(|| {
                CollectionError::ResourceUnavailable(
                    "sysinfo cannot see this process, so its resident memory is unreadable".into(),
                )
            })
            .map_err(|e| TrustformersError::runtime_error(e.to_string()))?;

        Ok(MemoryMetrics {
            heap_used_mb: resident_mb,
            heap_free_mb: None,
            heap_total_mb: None,
            native_used_mb: None,
            graphics_used_mb: None,
            code_used_mb: None,
            stack_used_mb: None,
            other_used_mb: None,
            available_mb: bytes_to_mb(system.available_memory()),
        })
    }

    /// Real CPU figures from `sysinfo`.
    ///
    /// Global usage needs two samples separated by at least
    /// `sysinfo::MINIMUM_CPU_UPDATE_INTERVAL`, so this call sleeps for that
    /// interval -- the cost of a real reading rather than a made-up one. Idle
    /// is derived from usage; the user/kernel split and the throttling ratio
    /// have no portable source and stay `None`.
    fn collect_cpu_metrics(&self) -> Result<CpuMetrics> {
        use sysinfo::System;

        let mut system = System::new();
        system.refresh_cpu_usage();
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        system.refresh_cpu_all();

        let usage_percent = system.global_cpu_usage();
        // `sysinfo` reports 0 MHz when it has no frequency for a CPU.
        let frequency_mhz = system
            .cpus()
            .iter()
            .map(|cpu| cpu.frequency())
            .max()
            .filter(|mhz| *mhz > 0)
            .map(|mhz| mhz as u32);

        Ok(CpuMetrics {
            usage_percent,
            user_percent: None,
            system_percent: None,
            idle_percent: (100.0 - usage_percent).clamp(0.0, 100.0),
            frequency_mhz,
            temperature_c: hottest_component_celsius(),
            throttling_level: None,
        })
    }

    /// GPU telemetry has no portable source.
    ///
    /// Utilisation, VRAM occupancy, clock and board power all need a
    /// vendor-specific driver query (Metal performance counters, the Android
    /// GPU delegate, NVML). None of those is reachable from this crate's
    /// dependency set, so there is nothing to report.
    fn collect_gpu_metrics(&self) -> Result<Option<GpuMetrics>> {
        Ok(None)
    }

    /// Thermal metrics from the system's thermal components, when it has any.
    ///
    /// `None` on hosts that expose no sensors -- Apple Silicon and most
    /// containers among them. The trend needs a history this per-sample call
    /// does not hold, so it is reported as [`TemperatureTrend::Stable`] only
    /// when a temperature is genuinely read.
    fn collect_thermal_metrics(&self) -> Result<Option<ThermalMetrics>> {
        let Some(temperature_c) = hottest_component_celsius() else {
            return Ok(None);
        };

        Ok(Some(ThermalMetrics {
            temperature_c,
            thermal_state: thermal_state_for(temperature_c),
            // A throttling ratio is not published by any portable API.
            throttling_level: None,
            temperature_trend: TemperatureTrend::Stable,
        }))
    }

    /// Battery telemetry from the kernel `power_supply` sysfs class on
    /// Android/Linux.
    ///
    /// Reuses `crate::battery::read_live_battery_reading`, the same reader
    /// `crate::battery`'s own manager calls, rather than re-implementing a
    /// second `#[cfg(target_os = ...)]`-gated sysfs walk here -- one parser
    /// for the whole crate. `None` everywhere else, and on any host where the
    /// reading carries nothing at all (no `Battery`-type supply found, e.g. a
    /// desktop Linux box with no battery).
    fn collect_battery_metrics(&self) -> Result<Option<BatteryMetrics>> {
        let reading = crate::battery::read_live_battery_reading();
        if !battery_reading_has_data(&reading) {
            return Ok(None);
        }

        let is_charging = charging_status_to_bool(reading.charging_status);

        Ok(Some(BatteryMetrics {
            level_percent: reading.level_percent,
            is_charging,
            power_consumption_mw: reading.power_consumption_mw,
            voltage_v: reading.voltage,
            estimated_life_minutes: reading.estimated_time_remaining_minutes,
        }))
    }

    /// Platform-specific metrics.
    ///
    /// The iOS and Android variants of [`PlatformMetrics`] need Metal
    /// performance counters and NNAPI/ART statistics respectively; neither is
    /// queried by this crate, so both stay absent.
    fn collect_platform_metrics(&self) -> Result<PlatformMetrics> {
        Ok(PlatformMetrics {
            #[cfg(target_os = "ios")]
            ios: None,
            #[cfg(target_os = "android")]
            android: None,
        })
    }

    /// Name of the target this collector is measuring.
    fn platform_name(&self) -> &'static str {
        std::env::consts::OS
    }

    /// Whether a metric family is actually measured on this target.
    ///
    /// Previously this answered `true` for memory, CPU, GPU, thermal *and*
    /// battery on every platform, which was only true because every one of
    /// them returned a constant.
    fn supports_metric(&self, metric_type: &str) -> bool {
        match metric_type {
            "memory" | "cpu" => true,
            "thermal" => hottest_component_celsius().is_some(),
            "battery" => battery_reading_has_data(&crate::battery::read_live_battery_reading()),
            "gpu" | "network" => false,
            _ => false,
        }
    }
}

/// Whether a battery reading actually measured anything, or is
/// [`crate::battery::BatteryReading::unavailable`] in disguise -- no
/// `Battery`-type `power_supply` node found. True on every non-Android/Linux
/// target, and on any Linux/Android host with no battery at all (a desktop, a
/// server, most CI runners).
fn battery_reading_has_data(reading: &crate::battery::BatteryReading) -> bool {
    reading.level_percent.is_some()
        || reading.voltage.is_some()
        || reading.power_consumption_mw.is_some()
        || reading.estimated_time_remaining_minutes.is_some()
        || charging_status_to_bool(reading.charging_status).is_some()
}

/// Collapse the platform's charging status to the tri-state
/// [`BatteryMetrics::is_charging`] carries: `Some(true)`/`Some(false)` when
/// the platform actually reported a status, `None` when it is
/// [`crate::device_info::ChargingStatus::Unknown`] (every target this
/// collector cannot read, plus a handful of real gauges that publish no
/// `status` node).
fn charging_status_to_bool(status: crate::device_info::ChargingStatus) -> Option<bool> {
    match status {
        crate::device_info::ChargingStatus::Charging => Some(true),
        crate::device_info::ChargingStatus::Discharging
        | crate::device_info::ChargingStatus::NotCharging
        | crate::device_info::ChargingStatus::Full => Some(false),
        crate::device_info::ChargingStatus::Unknown => None,
    }
}

/// Highest temperature reported by the system's thermal components, in
/// Celsius, or `None` when the platform exposes no sensors.
fn hottest_component_celsius() -> Option<f32> {
    sysinfo::Components::new_with_refreshed_list()
        .iter()
        .filter_map(|component| component.temperature())
        .fold(None::<f32>, |hottest, celsius| {
            Some(hottest.map_or(celsius, |best| best.max(celsius)))
        })
}

/// Bucket a measured temperature into the shared [`ThermalState`] scale.
fn thermal_state_for(celsius: f32) -> crate::device_info::ThermalState {
    use crate::device_info::ThermalState;
    match celsius {
        t if t >= 100.0 => ThermalState::Shutdown,
        t if t >= 90.0 => ThermalState::Emergency,
        t if t >= 80.0 => ThermalState::Critical,
        t if t >= 70.0 => ThermalState::Serious,
        t if t >= 55.0 => ThermalState::Fair,
        _ => ThermalState::Nominal,
    }
}

impl MobileMetricsCollector {
    /// Create new mobile metrics collector
    ///
    /// Initializes the collector with the provided configuration and sets up
    /// platform-specific collection capabilities.
    ///
    /// # Arguments
    ///
    /// * `config` - Profiler configuration including sampling rates and feature flags
    ///
    /// # Returns
    ///
    /// Returns a configured collector ready for metric collection.
    ///
    /// # Errors
    ///
    /// - `CollectionError::InvalidConfiguration` if configuration is invalid
    /// - `CollectionError::PlatformNotSupported` if platform lacks required APIs
    ///
    /// # Example
    ///
    /// ```rust
    /// # fn main() -> trustformers_core::error::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::collector::MobileMetricsCollector;
    /// # use trustformers_mobile::mobile_performance_profiler::config::MobileProfilerConfig;
    /// let config = MobileProfilerConfig::default();
    /// let collector = MobileMetricsCollector::new(config)?;
    /// # let _ = collector;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: MobileProfilerConfig) -> Result<Self> {
        // Validate configuration
        Self::validate_config(&config)?;

        let config_arc = Arc::new(RwLock::new(config.clone()));

        // Create platform-specific collector
        let platform_collector = Arc::new(SystemMetricsCollector);

        // Initialize collector state
        let collection_state = Arc::new(RwLock::new(CollectionState {
            is_collecting: false,
            collection_start: None,
            last_collection: None,
            total_samples: 0,
            error_count: 0,
            avg_collection_time_ms: 0.0,
        }));

        let inference_tracker = Arc::new(Mutex::new(InferenceTracker {
            active_inferences: HashMap::new(),
            completed_inferences: VecDeque::new(),
            model_load_times: HashMap::new(),
            cache_stats: CacheStats {
                total_requests: 0,
                cache_hits: 0,
                cache_misses: 0,
                avg_hit_latency_ms: 0.0,
                avg_miss_latency_ms: 0.0,
            },
        }));

        let statistics = Arc::new(Mutex::new(CollectionStatistics {
            total_samples: 0,
            collection_duration: Duration::new(0, 0),
            average_sampling_rate: 0.0,
            history_size: 0,
            current_memory_usage_mb: 0.0,
            error_count: 0,
            avg_collection_time_ms: 0.0,
            success_rate: 1.0,
        }));

        tracing::info!(
            "Initialized MobileMetricsCollector for platform: {}",
            platform_collector.platform_name()
        );

        Ok(Self {
            config: config_arc,
            current_metrics: Arc::new(RwLock::new(MobileMetricsSnapshot::default())),
            metrics_history: Arc::new(Mutex::new(VecDeque::new())),
            collection_state,
            platform_collector,
            inference_tracker,
            statistics,
        })
    }

    /// Start metrics collection
    ///
    /// Begins continuous metric collection according to the configured sampling rate.
    /// This method is idempotent - calling it multiple times has no additional effect.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful start, or an error if collection cannot begin.
    ///
    /// # Errors
    ///
    /// - `CollectionError::ResourceUnavailable` if system resources are unavailable
    /// - `CollectionError::PermissionDenied` if required permissions are missing
    ///
    /// # Example
    ///
    /// ```rust
    /// # fn main() -> trustformers_core::error::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::collector::MobileMetricsCollector;
    /// # use trustformers_mobile::mobile_performance_profiler::config::MobileProfilerConfig;
    /// # let config = MobileProfilerConfig::default();
    /// # let collector = MobileMetricsCollector::new(config)?;
    /// collector.start_collection()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn start_collection(&self) -> Result<()> {
        {
            let state = self
                .collection_state
                .read()
                .map_err(|e| TrustformersError::runtime_error(format!("Lock error: {}", e)))?;

            if state.is_collecting {
                tracing::warn!("Collection already active, ignoring start request");
                return Ok(());
            }
        } // Release read lock before calling collect_metrics_internal

        // Perform initial collection to verify system readiness
        let collection_start = Instant::now();
        let initial_result = self.collect_metrics_internal();

        match initial_result {
            Ok(_) => {
                // Now acquire write lock to update state
                let mut state = self
                    .collection_state
                    .write()
                    .map_err(|e| TrustformersError::runtime_error(format!("Lock error: {}", e)))?;

                state.is_collecting = true;
                state.collection_start = Some(collection_start);
                state.last_collection = Some(Instant::now());

                tracing::info!("Started mobile metrics collection");
                Ok(())
            },
            Err(e) => {
                tracing::error!("Failed to start collection: {}", e);
                Err(e)
            },
        }
    }

    /// Stop metrics collection
    ///
    /// Stops continuous metric collection and finalizes collection statistics.
    /// This method is idempotent - calling it multiple times has no additional effect.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful stop.
    ///
    /// # Example
    ///
    /// ```rust
    /// # fn main() -> trustformers_core::error::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::collector::MobileMetricsCollector;
    /// # use trustformers_mobile::mobile_performance_profiler::config::MobileProfilerConfig;
    /// # let config = MobileProfilerConfig::default();
    /// # let collector = MobileMetricsCollector::new(config)?;
    /// collector.stop_collection()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn stop_collection(&self) -> Result<()> {
        let mut state = self
            .collection_state
            .write()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock error: {}", e)))?;

        if !state.is_collecting {
            tracing::warn!("Collection not active, ignoring stop request");
            return Ok(());
        }

        state.is_collecting = false;

        // Update final statistics
        if let Ok(mut stats) = self.statistics.lock() {
            if let Some(start_time) = state.collection_start {
                stats.collection_duration = start_time.elapsed();
                if stats.collection_duration.as_secs() > 0 {
                    stats.average_sampling_rate =
                        state.total_samples as f64 / stats.collection_duration.as_secs() as f64;
                }
            }

            if state.total_samples > 0 {
                stats.success_rate = 1.0 - (state.error_count as f64 / state.total_samples as f64);
            }
        }

        tracing::info!(
            "Stopped mobile metrics collection. Collected {} samples with {} errors",
            state.total_samples,
            state.error_count
        );

        Ok(())
    }

    /// Pause metrics collection
    ///
    /// Temporarily suspends metrics collection without stopping the collector entirely.
    /// Collection can be resumed with `resume_collection()`.
    pub fn pause_collection(&self) -> Result<()> {
        let mut state = self
            .collection_state
            .write()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock error: {}", e)))?;

        if !state.is_collecting {
            tracing::warn!("Collection not active, cannot pause");
            return Ok(());
        }

        state.is_collecting = false;
        tracing::info!("Paused mobile metrics collection");
        Ok(())
    }

    /// Resume metrics collection
    ///
    /// Resumes previously paused metrics collection.
    pub fn resume_collection(&self) -> Result<()> {
        let mut state = self
            .collection_state
            .write()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock error: {}", e)))?;

        if state.is_collecting {
            tracing::warn!("Collection already active, ignoring resume request");
            return Ok(());
        }

        state.is_collecting = true;
        state.last_collection = Some(Instant::now());
        tracing::info!("Resumed mobile metrics collection");
        Ok(())
    }

    /// Get collection statistics
    ///
    /// Returns current collection statistics and performance metrics.
    pub fn get_collection_stats(&self) -> Result<CollectionStatistics> {
        let stats = self
            .statistics
            .lock()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock error: {}", e)))?;
        Ok(stats.clone())
    }

    /// Get current metrics snapshot
    ///
    /// Returns the most recently collected metrics snapshot. If collection is not
    /// active, this will return the last collected snapshot.
    ///
    /// # Returns
    ///
    /// Returns a clone of the current metrics snapshot.
    ///
    /// # Example
    ///
    /// ```rust
    /// # fn main() -> trustformers_core::error::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::collector::MobileMetricsCollector;
    /// # use trustformers_mobile::mobile_performance_profiler::config::MobileProfilerConfig;
    /// # let config = MobileProfilerConfig::default();
    /// # let collector = MobileMetricsCollector::new(config)?;
    /// let snapshot = collector.get_current_snapshot()?;
    /// // `None` when CPU profiling is disabled or the platform refused the
    /// // measurement -- never a zero standing in for one.
    /// match snapshot.cpu {
    ///     Some(cpu) => println!("Current CPU usage: {}%", cpu.usage_percent),
    ///     None => println!("CPU usage was not measured"),
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_current_snapshot(&self) -> Result<MobileMetricsSnapshot> {
        let snapshot = self
            .current_metrics
            .read()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock error: {}", e)))?;

        Ok(snapshot.clone())
    }

    /// Get all historical metrics snapshots
    ///
    /// Returns a vector containing all historical metrics snapshots currently
    /// stored in the collector's history.
    ///
    /// # Returns
    ///
    /// Returns a vector of metrics snapshots ordered by collection time.
    ///
    /// # Example
    ///
    /// ```rust
    /// # fn main() -> trustformers_core::error::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::collector::MobileMetricsCollector;
    /// # use trustformers_mobile::mobile_performance_profiler::config::MobileProfilerConfig;
    /// # let config = MobileProfilerConfig::default();
    /// # let collector = MobileMetricsCollector::new(config)?;
    /// let history = collector.get_all_snapshots();
    /// println!("Collected {} historical snapshots", history.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_all_snapshots(&self) -> Vec<MobileMetricsSnapshot> {
        self.metrics_history
            .lock()
            .map(|history| history.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Force immediate metrics collection
    ///
    /// Collects metrics immediately regardless of the configured sampling interval.
    /// This is useful for event-driven collection or debugging purposes.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful collection.
    ///
    /// # Errors
    ///
    /// - Various collection errors depending on system state and permissions
    ///
    /// # Example
    ///
    /// ```rust
    /// # fn main() -> trustformers_core::error::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::collector::MobileMetricsCollector;
    /// # use trustformers_mobile::mobile_performance_profiler::config::MobileProfilerConfig;
    /// # let config = MobileProfilerConfig::default();
    /// # let collector = MobileMetricsCollector::new(config)?;
    /// collector.collect_metrics()?;
    /// let snapshot = collector.get_current_snapshot()?;
    /// # let _ = snapshot;
    /// # Ok(())
    /// # }
    /// ```
    pub fn collect_metrics(&self) -> Result<()> {
        self.collect_metrics_internal()
    }

    /// Start tracking an inference session
    ///
    /// Begins tracking performance metrics for a specific inference operation.
    /// This enables detailed inference-specific performance analysis.
    ///
    /// # Arguments
    ///
    /// * `session_id` - Unique identifier for the inference session
    /// * `model_name` - Name of the model being used for inference
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful session start.
    ///
    /// # Example
    ///
    /// ```rust
    /// # fn main() -> trustformers_core::error::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::collector::MobileMetricsCollector;
    /// # use trustformers_mobile::mobile_performance_profiler::config::MobileProfilerConfig;
    /// # let config = MobileProfilerConfig::default();
    /// # let collector = MobileMetricsCollector::new(config)?;
    /// collector.start_inference_tracking("session_1", "my_model")?;
    /// // ... perform inference ...
    /// collector.end_inference_tracking("session_1", true)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn start_inference_tracking(&self, session_id: &str, model_name: &str) -> Result<()> {
        let mut tracker = self
            .inference_tracker
            .lock()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock error: {}", e)))?;

        // Collect current system snapshot
        let current_snapshot = self.get_current_snapshot()?;
        let system_snapshot = SystemSnapshot {
            memory_usage_mb: current_snapshot.memory.as_ref().map(|m| m.heap_used_mb),
            cpu_usage_percent: current_snapshot.cpu.as_ref().map(|c| c.usage_percent),
            gpu_usage_percent: current_snapshot.gpu.as_ref().map(|gpu| gpu.usage_percent),
            timestamp: Instant::now(),
        };

        let session = InferenceSession {
            id: session_id.to_string(),
            model_name: model_name.to_string(),
            start_time: Instant::now(),
            initial_metrics: system_snapshot,
        };

        tracker.active_inferences.insert(session_id.to_string(), session);

        tracing::debug!("Started inference tracking for session: {}", session_id);
        Ok(())
    }

    /// End inference session tracking
    ///
    /// Completes tracking for an inference session and records performance metrics.
    ///
    /// # Arguments
    ///
    /// * `session_id` - Unique identifier for the inference session
    /// * `success` - Whether the inference completed successfully
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful session completion.
    ///
    /// # Example
    ///
    /// ```rust
    /// # fn main() -> trustformers_core::error::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::collector::MobileMetricsCollector;
    /// # use trustformers_mobile::mobile_performance_profiler::config::MobileProfilerConfig;
    /// # let config = MobileProfilerConfig::default();
    /// # let collector = MobileMetricsCollector::new(config)?;
    /// collector.end_inference_tracking("session_1", true)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn end_inference_tracking(&self, session_id: &str, success: bool) -> Result<()> {
        let mut tracker = self
            .inference_tracker
            .lock()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock error: {}", e)))?;

        if let Some(session) = tracker.active_inferences.remove(session_id) {
            let end_time = Instant::now();
            let duration_ms = session.start_time.elapsed().as_millis() as f64;

            // Collect final system metrics
            let current_snapshot = self.get_current_snapshot()?;

            let completed_inference = CompletedInference {
                id: session_id.to_string(),
                model_name: session.model_name.clone(),
                duration_ms,
                success,
                memory_delta_mb: match (
                    current_snapshot.memory.as_ref().map(|m| m.heap_used_mb),
                    session.initial_metrics.memory_usage_mb,
                ) {
                    (Some(now), Some(before)) => Some(now - before),
                    _ => None,
                },
                cpu_usage_percent: current_snapshot.cpu.as_ref().map(|c| c.usage_percent),
                gpu_usage_percent: current_snapshot.gpu.as_ref().map(|gpu| gpu.usage_percent),
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            };

            tracker.completed_inferences.push_back(completed_inference);

            // Limit completed inferences history
            if tracker.completed_inferences.len() > 1000 {
                tracker.completed_inferences.pop_front();
            }

            tracing::debug!(
                "Completed inference tracking for session: {} ({}ms, success: {})",
                session_id,
                duration_ms,
                success
            );
        } else {
            tracing::warn!("No active inference session found for ID: {}", session_id);
        }

        Ok(())
    }

    /// Record model load time
    ///
    /// Records the time taken to load a specific model for inference performance analysis.
    ///
    /// # Arguments
    ///
    /// * `model_name` - Name of the model
    /// * `load_time_ms` - Time taken to load the model in milliseconds
    ///
    /// # Example
    ///
    /// ```rust
    /// # fn main() -> trustformers_core::error::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::collector::MobileMetricsCollector;
    /// # use trustformers_mobile::mobile_performance_profiler::config::MobileProfilerConfig;
    /// # let config = MobileProfilerConfig::default();
    /// # let collector = MobileMetricsCollector::new(config)?;
    /// collector.record_model_load_time("my_model", 1500.0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn record_model_load_time(&self, model_name: &str, load_time_ms: f64) -> Result<()> {
        let mut tracker = self
            .inference_tracker
            .lock()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock error: {}", e)))?;

        tracker.model_load_times.insert(model_name.to_string(), load_time_ms);

        tracing::debug!(
            "Recorded model load time: {} = {}ms",
            model_name,
            load_time_ms
        );
        Ok(())
    }

    /// Record cache hit
    ///
    /// Records a cache hit event for cache performance analysis.
    ///
    /// # Arguments
    ///
    /// * `latency_ms` - Cache hit latency in milliseconds
    ///
    /// # Example
    ///
    /// ```rust
    /// # fn main() -> trustformers_core::error::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::collector::MobileMetricsCollector;
    /// # use trustformers_mobile::mobile_performance_profiler::config::MobileProfilerConfig;
    /// # let config = MobileProfilerConfig::default();
    /// # let collector = MobileMetricsCollector::new(config)?;
    /// collector.record_cache_hit(2.5)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn record_cache_hit(&self, latency_ms: f64) -> Result<()> {
        let mut tracker = self
            .inference_tracker
            .lock()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock error: {}", e)))?;

        tracker.cache_stats.total_requests += 1;
        tracker.cache_stats.cache_hits += 1;

        // Update running average
        let total_hits = tracker.cache_stats.cache_hits as f64;
        tracker.cache_stats.avg_hit_latency_ms =
            (tracker.cache_stats.avg_hit_latency_ms * (total_hits - 1.0) + latency_ms) / total_hits;

        Ok(())
    }

    /// Record cache miss
    ///
    /// Records a cache miss event for cache performance analysis.
    ///
    /// # Arguments
    ///
    /// * `latency_ms` - Cache miss latency in milliseconds
    ///
    /// # Example
    ///
    /// ```rust
    /// # fn main() -> trustformers_core::error::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::collector::MobileMetricsCollector;
    /// # use trustformers_mobile::mobile_performance_profiler::config::MobileProfilerConfig;
    /// # let config = MobileProfilerConfig::default();
    /// # let collector = MobileMetricsCollector::new(config)?;
    /// collector.record_cache_miss(15.7)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn record_cache_miss(&self, latency_ms: f64) -> Result<()> {
        let mut tracker = self
            .inference_tracker
            .lock()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock error: {}", e)))?;

        tracker.cache_stats.total_requests += 1;
        tracker.cache_stats.cache_misses += 1;

        // Update running average
        let total_misses = tracker.cache_stats.cache_misses as f64;
        tracker.cache_stats.avg_miss_latency_ms =
            (tracker.cache_stats.avg_miss_latency_ms * (total_misses - 1.0) + latency_ms)
                / total_misses;

        Ok(())
    }

    /// Get collection statistics
    ///
    /// Returns comprehensive statistics about the collection process including
    /// performance metrics, error rates, and resource usage.
    ///
    /// # Returns
    ///
    /// Returns current collection statistics.
    ///
    /// # Example
    ///
    /// ```rust
    /// # fn main() -> trustformers_core::error::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::collector::MobileMetricsCollector;
    /// # use trustformers_mobile::mobile_performance_profiler::config::MobileProfilerConfig;
    /// # let config = MobileProfilerConfig::default();
    /// # let collector = MobileMetricsCollector::new(config)?;
    /// let stats = collector.get_collection_statistics();
    /// println!("Collected {} samples at {:.1} samples/sec",
    ///     stats.total_samples, stats.average_sampling_rate);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_collection_statistics(&self) -> CollectionStatistics {
        let mut stats = self.statistics.lock().map(|s| s.clone()).unwrap_or_default();

        // Update current statistics
        if let Ok(state) = self.collection_state.read() {
            stats.total_samples = state.total_samples;
            stats.error_count = state.error_count;
            stats.avg_collection_time_ms = state.avg_collection_time_ms;

            if let Some(start_time) = state.collection_start {
                stats.collection_duration = start_time.elapsed();

                if stats.collection_duration.as_secs() > 0 {
                    stats.average_sampling_rate =
                        state.total_samples as f64 / stats.collection_duration.as_secs() as f64;
                }
            }

            if state.total_samples > 0 {
                stats.success_rate = 1.0 - (state.error_count as f64 / state.total_samples as f64);
            }
        }

        if let Ok(history) = self.metrics_history.lock() {
            stats.history_size = history.len();
            stats.current_memory_usage_mb = self.estimate_memory_usage(&history);
        }

        stats
    }

    /// Update collector configuration
    ///
    /// Updates the collector configuration at runtime. Some changes may require
    /// restarting collection to take effect.
    ///
    /// # Arguments
    ///
    /// * `new_config` - New configuration to apply
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful configuration update.
    ///
    /// # Example
    ///
    /// ```rust
    /// # fn main() -> trustformers_core::error::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::collector::MobileMetricsCollector;
    /// # use trustformers_mobile::mobile_performance_profiler::config::MobileProfilerConfig;
    /// # let config = MobileProfilerConfig::default();
    /// # let collector = MobileMetricsCollector::new(config)?;
    /// let mut config = collector.get_config();
    /// config.sampling.interval_ms = 50;
    /// collector.update_config(config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn update_config(&self, new_config: MobileProfilerConfig) -> Result<()> {
        Self::validate_config(&new_config)?;

        let mut config = self
            .config
            .write()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock error: {}", e)))?;

        *config = new_config;

        tracing::info!("Updated collector configuration");
        Ok(())
    }

    /// Get current configuration
    ///
    /// Returns a copy of the current collector configuration.
    ///
    /// # Returns
    ///
    /// Returns the current configuration.
    pub fn get_config(&self) -> MobileProfilerConfig {
        self.config.read().map(|c| c.clone()).unwrap_or_default()
    }

    // Private implementation methods

    /// Internal metrics collection implementation
    fn collect_metrics_internal(&self) -> Result<()> {
        let collection_start = Instant::now();

        // Check if collection is enabled
        let config = self
            .config
            .read()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock error: {}", e)))?;
        if !config.enabled {
            return Err(TrustformersError::runtime_error("Collection is disabled".into()).into());
        }

        // Collect metrics with error handling
        let mut collection_errors = Vec::new();

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| TrustformersError::other(format!("Time error: {}", e)))?
            .as_millis() as u64;

        // Collect individual metric types
        // Disabled-by-configuration and failed-to-measure both yield `None`.
        // A zeroed `MemoryMetrics`/`CpuMetrics` would report an idle process
        // that nothing had looked at.
        let memory = if config.memory_profiling.enabled {
            self.platform_collector.collect_memory_metrics().map(Some).unwrap_or_else(|e| {
                collection_errors.push(e);
                None
            })
        } else {
            None
        };

        let cpu = if config.cpu_profiling.enabled {
            self.platform_collector.collect_cpu_metrics().map(Some).unwrap_or_else(|e| {
                collection_errors.push(e);
                None
            })
        } else {
            None
        };

        // Unavailable metric families stay `None`. They are deliberately not
        // replaced by a zeroed default: a zero would be read downstream as a
        // measured idle GPU / silent network rather than as "not measured".
        let gpu = if config.gpu_profiling.enabled {
            self.platform_collector.collect_gpu_metrics().unwrap_or_else(|e| {
                collection_errors.push(e);
                None
            })
        } else {
            None
        };

        let network = if config.network_profiling.enabled {
            self.collect_network_metrics().unwrap_or_else(|e| {
                collection_errors.push(e);
                None
            })
        } else {
            None
        };

        let inference = self.collect_inference_metrics().unwrap_or_else(|e| {
            collection_errors.push(e);
            InferenceMetrics::default()
        });

        let thermal = self.platform_collector.collect_thermal_metrics().unwrap_or_else(|e| {
            collection_errors.push(e);
            None
        });

        let battery = self.platform_collector.collect_battery_metrics().unwrap_or_else(|e| {
            collection_errors.push(e);
            None
        });

        let platform = self.platform_collector.collect_platform_metrics().unwrap_or_else(|e| {
            collection_errors.push(e);
            PlatformMetrics::default()
        });

        // Create metrics snapshot
        let snapshot = MobileMetricsSnapshot {
            timestamp,
            memory,
            cpu,
            gpu,
            network,
            inference,
            thermal,
            battery,
            platform,
        };

        // Update current metrics
        if let Ok(mut current) = self.current_metrics.write() {
            *current = snapshot.clone();
        }

        // Add to history
        if let Ok(mut history) = self.metrics_history.lock() {
            history.push_back(snapshot);

            // Maintain history size limit
            if history.len() > config.sampling.max_samples {
                history.pop_front();
            }
        }

        // Update collection state
        let collection_time_ms = collection_start.elapsed().as_millis() as f64;
        if let Ok(mut state) = self.collection_state.write() {
            state.total_samples += 1;
            if !collection_errors.is_empty() {
                state.error_count += 1;
            }

            // Update average collection time
            let total_samples = state.total_samples as f64;
            state.avg_collection_time_ms = (state.avg_collection_time_ms * (total_samples - 1.0)
                + collection_time_ms)
                / total_samples;

            state.last_collection = Some(Instant::now());
        }

        // Log collection errors but don't fail
        if !collection_errors.is_empty() {
            tracing::warn!(
                "Collection completed with {} errors: {:?}",
                collection_errors.len(),
                collection_errors
            );
        }

        tracing::trace!(
            "Metrics collection completed in {:.2}ms",
            collection_time_ms
        );

        Ok(())
    }

    /// Network metrics have no source in this crate's dependency set.
    ///
    /// Per-process byte/packet counters need `sysinfo`'s `network` feature
    /// (system-wide, not per-process) or a platform socket API; latency,
    /// bandwidth and error rate need active probing this profiler does not
    /// perform. The previous body returned a fixed 1 MB sent / 2 MB received /
    /// 45 ms / 25 Mbps / 2% error reading on every call, on every machine.
    fn collect_network_metrics(&self) -> Result<Option<NetworkMetrics>> {
        Ok(None)
    }

    /// Collect inference metrics from tracked sessions
    fn collect_inference_metrics(&self) -> Result<InferenceMetrics> {
        let tracker = self
            .inference_tracker
            .lock()
            .map_err(|e| TrustformersError::runtime_error(format!("Lock error: {}", e)))?;

        let total_inferences = tracker.completed_inferences.len() as u64;
        let successful_inferences =
            tracker.completed_inferences.iter().filter(|inf| inf.success).count() as u64;
        let failed_inferences = total_inferences - successful_inferences;

        let avg_latency_ms = if total_inferences > 0 {
            tracker.completed_inferences.iter().map(|inf| inf.duration_ms).sum::<f64>()
                / total_inferences as f64
        } else {
            0.0
        };

        let min_latency_ms = tracker
            .completed_inferences
            .iter()
            .map(|inf| inf.duration_ms)
            .fold(f64::INFINITY, f64::min);

        let max_latency_ms = tracker
            .completed_inferences
            .iter()
            .map(|inf| inf.duration_ms)
            .fold(0.0, f64::max);

        let throughput_per_sec = if total_inferences > 0 && avg_latency_ms > 0.0 {
            1000.0 / avg_latency_ms
        } else {
            0.0
        };

        let cache_hit_rate = if tracker.cache_stats.total_requests > 0 {
            tracker.cache_stats.cache_hits as f64 / tracker.cache_stats.total_requests as f64
        } else {
            0.0
        };

        let model_load_time_ms = tracker.model_load_times.values().copied().fold(0.0, f64::max);

        Ok(InferenceMetrics {
            total_inferences,
            successful_inferences,
            failed_inferences,
            avg_latency_ms,
            min_latency_ms: if min_latency_ms.is_infinite() { 0.0 } else { min_latency_ms },
            max_latency_ms,
            throughput_per_sec,
            cache_hit_rate,
            model_load_time_ms,
        })
    }

    /// Validate configuration
    fn validate_config(config: &MobileProfilerConfig) -> Result<()> {
        if config.sampling.interval_ms == 0 {
            return Err(TrustformersError::invalid_argument(
                "Sampling interval must be > 0".into(),
            )
            .into());
        }

        if config.sampling.max_samples == 0 {
            return Err(
                TrustformersError::invalid_argument("Max samples must be > 0".into()).into(),
            );
        }

        if config.sampling.high_freq_threshold_ms >= config.sampling.low_freq_threshold_ms {
            return Err(TrustformersError::invalid_argument(
                "High frequency threshold must be less than low frequency threshold".into(),
            )
            .into());
        }

        Ok(())
    }

    /// Estimate memory usage of collector
    fn estimate_memory_usage(&self, history: &VecDeque<MobileMetricsSnapshot>) -> f32 {
        let snapshot_size = std::mem::size_of::<MobileMetricsSnapshot>();
        let total_size = snapshot_size * history.len();
        total_size as f32 / (1024.0 * 1024.0) // Convert to MB
    }
}

impl Default for CollectionStatistics {
    fn default() -> Self {
        Self {
            total_samples: 0,
            collection_duration: Duration::new(0, 0),
            average_sampling_rate: 0.0,
            history_size: 0,
            current_memory_usage_mb: 0.0,
            error_count: 0,
            avg_collection_time_ms: 0.0,
            success_rate: 1.0,
        }
    }
}

impl Default for CollectionState {
    fn default() -> Self {
        Self {
            is_collecting: false,
            collection_start: None,
            last_collection: None,
            total_samples: 0,
            error_count: 0,
            avg_collection_time_ms: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Create a fast test config with minimal overhead for tests
    fn fast_test_config() -> MobileProfilerConfig {
        // Start with default config
        let mut config = MobileProfilerConfig::default();

        // Disable all expensive profiling
        config.memory_profiling.enabled = false;
        config.cpu_profiling.enabled = false;
        config.gpu_profiling.enabled = false;
        config.network_profiling.enabled = false;
        config.real_time_monitoring.enabled = false;

        // Slow down sampling
        config.sampling.interval_ms = 10000; // 10 seconds
        config.sampling.max_samples = 10;

        config
    }

    #[test]
    fn test_collector_creation() {
        let config = fast_test_config();
        let collector = MobileMetricsCollector::new(config);
        assert!(collector.is_ok());
    }

    #[test]
    fn test_collector_configuration_validation() {
        let mut config = MobileProfilerConfig::default();
        config.sampling.interval_ms = 0;

        let result = MobileMetricsCollector::new(config);
        assert!(result.is_err());
    }

    /// Was `#[ignore]`d with "60+ second delays (likely thread/deadlock
    /// issue)". Investigated 2026-08-24: this collector spawns no background
    /// thread at all (`grep -n 'thread::spawn' collector.rs` -- zero hits),
    /// so there is no thread to deadlock and no channel/`Condvar` anyone
    /// forgot to signal. Reliably completes in well under a second, run
    /// standalone or as part of the full suite; the stale FIXME predates this
    /// module's rewrite and the hang it described no longer exists.
    #[test]
    fn test_collection_lifecycle() {
        let config = fast_test_config();
        let collector = MobileMetricsCollector::new(config).expect("Operation failed");

        // Test start collection
        assert!(collector.start_collection().is_ok());

        // Test force collection with a small delay
        std::thread::sleep(Duration::from_millis(1));
        assert!(collector.collect_metrics().is_ok());

        // Test getting snapshot
        let snapshot = collector.get_current_snapshot();
        assert!(snapshot.is_ok());

        // Test stop collection
        assert!(collector.stop_collection().is_ok());

        // Give background tasks time to clean up
        std::thread::sleep(Duration::from_millis(1));
    }

    #[test]
    fn test_inference_tracking() {
        let config = fast_test_config();
        let collector = MobileMetricsCollector::new(config).expect("Operation failed");

        // Start inference tracking
        assert!(collector.start_inference_tracking("test_session", "test_model").is_ok());

        // Simulate some time
        std::thread::sleep(Duration::from_millis(1));

        // End inference tracking
        assert!(collector.end_inference_tracking("test_session", true).is_ok());

        // Collect metrics and check inference data
        assert!(collector.collect_metrics().is_ok());
        let snapshot = collector.get_current_snapshot().expect("Operation failed");
        assert!(snapshot.inference.total_inferences > 0);
    }

    #[test]
    fn test_cache_tracking() {
        let config = fast_test_config();
        let collector = MobileMetricsCollector::new(config).expect("Operation failed");

        // Record cache events
        assert!(collector.record_cache_hit(2.5).is_ok());
        assert!(collector.record_cache_miss(15.0).is_ok());
        assert!(collector.record_cache_hit(3.0).is_ok());

        // Collect metrics and check cache statistics
        assert!(collector.collect_metrics().is_ok());
        let snapshot = collector.get_current_snapshot().expect("Operation failed");
        assert_eq!(snapshot.inference.cache_hit_rate, 2.0 / 3.0);
    }

    #[test]
    fn test_model_load_time_tracking() {
        let config = fast_test_config();
        let collector = MobileMetricsCollector::new(config).expect("Operation failed");

        // Record model load time
        assert!(collector.record_model_load_time("test_model", 1500.0).is_ok());

        // Collect metrics and check model load time
        assert!(collector.collect_metrics().is_ok());
        let snapshot = collector.get_current_snapshot().expect("Operation failed");
        assert_eq!(snapshot.inference.model_load_time_ms, 1500.0);
    }

    #[test]
    fn test_statistics_collection() {
        let config = fast_test_config();
        let collector = MobileMetricsCollector::new(config).expect("Operation failed");

        // Start collection and collect some samples
        assert!(collector.start_collection().is_ok());
        std::thread::sleep(Duration::from_millis(1));
        for _ in 0..5 {
            assert!(collector.collect_metrics().is_ok());
        }
        assert!(collector.stop_collection().is_ok());

        // Check statistics
        let stats = collector.get_collection_statistics();
        assert_eq!(stats.total_samples, 6); // 5 manual + 1 from start_collection
        assert!(stats.success_rate > 0.0);
        assert!(stats.collection_duration.as_nanos() > 0);
    }

    #[test]
    fn test_configuration_update() {
        let config = fast_test_config();
        let collector = MobileMetricsCollector::new(config).expect("Operation failed");

        // Update configuration
        let mut new_config = collector.get_config();
        new_config.sampling.interval_ms = 200;

        assert!(collector.update_config(new_config).is_ok());

        // Verify configuration was updated
        let updated_config = collector.get_config();
        assert_eq!(updated_config.sampling.interval_ms, 200);
    }

    #[test]
    fn test_history_management() {
        let mut config = fast_test_config();
        config.sampling.max_samples = 3; // Small limit for testing

        let collector = MobileMetricsCollector::new(config).expect("Operation failed");

        // Collect more samples than the limit
        for _ in 0..5 {
            assert!(collector.collect_metrics().is_ok());
        }

        // Check that history is limited
        let history = collector.get_all_snapshots();
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_memory_usage_estimation() {
        let config = fast_test_config();
        let collector = MobileMetricsCollector::new(config).expect("Operation failed");

        // Collect some samples
        for _ in 0..10 {
            assert!(collector.collect_metrics().is_ok());
        }

        // Check memory usage estimation
        let stats = collector.get_collection_statistics();
        assert!(stats.current_memory_usage_mb > 0.0);
    }

    #[test]
    fn test_error_resilience() {
        let config = fast_test_config();
        let collector = MobileMetricsCollector::new(config).expect("Operation failed");

        // Force collection should succeed even with potential system errors
        assert!(collector.collect_metrics().is_ok());

        // Collection should provide default values on error
        let snapshot = collector.get_current_snapshot().expect("Operation failed");
        assert!(snapshot.timestamp > 0);
    }

    /// Regression: the collector used to report a fixed 64 MB resident / 25%
    /// CPU / 20% GPU / 85% battery / 35 C snapshot on every non-mobile host
    /// (and different fixed sets on iOS and Android). Memory and CPU must now
    /// be real `sysinfo` measurements, and everything without a real source
    /// must be absent rather than a constant.
    #[test]
    fn test_snapshot_carries_real_measurements_not_constants() {
        // Memory and CPU profiling explicitly enabled: this test is about what
        // a *measuring* collector reports.
        let mut config = fast_test_config();
        config.memory_profiling.enabled = true;
        config.cpu_profiling.enabled = true;
        let collector = MobileMetricsCollector::new(config).expect("collector");
        collector.collect_metrics().expect("collect");
        let snapshot = collector.get_current_snapshot().expect("snapshot");

        // Real resident memory: this test process occupies a nonzero, and not
        // exactly-64.0 MB, resident set.
        let memory = snapshot.memory.as_ref().expect("memory profiling was enabled");
        assert!(memory.heap_used_mb > 0.0);
        assert_ne!(memory.heap_used_mb, 64.0);
        assert_ne!(memory.heap_used_mb, 128.0); // old iOS constant
        assert_ne!(memory.heap_used_mb, 96.0); // old Android constant
        assert!(memory.available_mb > 0.0);

        // Real CPU: a percentage, and never one of the three old constants.
        let cpu = snapshot.cpu.as_ref().expect("cpu profiling was enabled");
        assert!((0.0..=100.0).contains(&cpu.usage_percent));
        assert_ne!(cpu.usage_percent, 25.0);
        assert_ne!(cpu.usage_percent, 30.0);
        assert_ne!(cpu.usage_percent, 35.0);

        // The unmeasurable segments are absent, not zeroed.
        assert_eq!(memory.heap_total_mb, None);
        assert_eq!(memory.graphics_used_mb, None);
        assert_eq!(cpu.user_percent, None);
        assert_eq!(cpu.throttling_level, None);

        // GPU and network telemetry have no source at all in this crate.
        // Battery does, on Android/Linux with a `power_supply` node -- absent
        // here only because this host (like most CI runners) has none; see
        // `test_battery_is_measured_or_absent` for the conditional check.
        assert!(
            snapshot.gpu.is_none(),
            "GPU telemetry has no source in this build"
        );
        assert!(
            snapshot.network.is_none(),
            "network telemetry has no source in this build"
        );
        match snapshot.battery.as_ref() {
            Some(battery) => assert!(
                battery.level_percent.is_some()
                    || battery.voltage_v.is_some()
                    || battery.power_consumption_mw.is_some()
                    || battery.estimated_life_minutes.is_some()
                    || battery.is_charging.is_some(),
                "a reported BatteryMetrics must carry at least one real reading"
            ),
            None => {
                // Absent on this host: every non-Android/Linux target, and
                // any Linux/Android host with no `power_supply` battery node
                // -- true of most CI runners and this crate's own test
                // machine.
            },
        }
    }

    /// A family switched off by configuration must be absent, not zeroed:
    /// "profiling disabled" is not "the process used no memory".
    #[test]
    fn test_disabled_profiling_reports_absence_not_zero() {
        let config = fast_test_config();
        assert!(!config.memory_profiling.enabled && !config.cpu_profiling.enabled);
        let collector = MobileMetricsCollector::new(config).expect("collector");
        collector.collect_metrics().expect("collect");
        let snapshot = collector.get_current_snapshot().expect("snapshot");

        assert!(snapshot.memory.is_none());
        assert!(snapshot.cpu.is_none());
    }

    /// `supports_metric` used to answer `true` for every family because every
    /// family returned a constant. It must now answer for what is measured.
    ///
    /// `thermal` and `battery` are genuinely host-dependent (a real sensor or
    /// `power_supply` node may or may not exist on the machine running this
    /// test), so each gets its own conditional test below instead of a fixed
    /// answer here.
    #[test]
    fn test_supports_metric_reports_the_truth() {
        let collector = SystemMetricsCollector;
        assert!(collector.supports_metric("memory"));
        assert!(collector.supports_metric("cpu"));
        assert!(!collector.supports_metric("gpu"));
        assert!(!collector.supports_metric("network"));
        assert!(!collector.supports_metric("nonsense"));
        assert_eq!(collector.platform_name(), std::env::consts::OS);
    }

    /// Battery reporting must be all-or-nothing, like thermal: either the
    /// host has a real `power_supply` battery node and a genuine reading
    /// comes back, or the family is absent. Never a fabricated level/voltage.
    #[test]
    fn test_battery_is_measured_or_absent() {
        let collector = SystemMetricsCollector;
        let live = crate::battery::read_live_battery_reading();
        let expected_has_data = battery_reading_has_data(&live);
        assert_eq!(
            collector.supports_metric("battery"),
            expected_has_data,
            "supports_metric must agree with a fresh reading of the same host"
        );

        match collector.collect_battery_metrics().expect("battery collection never errors") {
            Some(battery) => {
                assert!(expected_has_data);
                assert!(
                    battery.level_percent.is_some()
                        || battery.voltage_v.is_some()
                        || battery.power_consumption_mw.is_some()
                        || battery.estimated_life_minutes.is_some()
                        || battery.is_charging.is_some()
                );
            },
            None => assert!(!expected_has_data),
        }
    }

    /// Thermal reporting must be all-or-nothing: either the host exposes
    /// sensors and a real temperature comes back, or the family is absent.
    /// It must never be a fabricated 35.0 C / 42.0 C.
    #[test]
    fn test_thermal_is_measured_or_absent() {
        let collector = SystemMetricsCollector;
        match collector.collect_thermal_metrics().expect("thermal") {
            Some(thermal) => {
                assert!(hottest_component_celsius().is_some());
                assert_ne!(thermal.temperature_c, 35.0);
                assert_ne!(thermal.temperature_c, 42.0);
                assert_eq!(thermal.throttling_level, None);
            },
            None => assert!(hottest_component_celsius().is_none()),
        }
    }

    /// Two consecutive samples of a running process must not be byte-identical
    /// the way a constant would be. Guards against a regression back to fixed
    /// values that the value-specific assertions above would miss.
    #[test]
    fn test_repeated_collection_is_sampled_not_replayed() {
        let mut config = fast_test_config();
        config.memory_profiling.enabled = true;
        let collector = MobileMetricsCollector::new(config).expect("collector");

        collector.collect_metrics().expect("collect");
        let first = collector.get_current_snapshot().expect("snapshot");
        let mut ballast: Vec<u8> = vec![7u8; 64 * 1024 * 1024];
        // Touch every page so the allocation is actually resident.
        for page in ballast.chunks_mut(4096) {
            page[0] = 1;
        }
        collector.collect_metrics().expect("collect");
        let second = collector.get_current_snapshot().expect("snapshot");

        assert!(second.timestamp >= first.timestamp);
        let before = first.memory.as_ref().expect("memory measured").heap_used_mb;
        let after = second.memory.as_ref().expect("memory measured").heap_used_mb;
        // Touching 64 MB must move the measured resident set. A constant
        // cannot respond to this.
        assert!(
            after > before,
            "resident memory did not respond to a real 64 MB allocation: {} -> {}",
            before,
            after
        );
        drop(ballast);
    }

    /// The thermal bucketing is a pure function of a measured temperature.
    #[test]
    fn test_thermal_state_buckets() {
        use crate::device_info::ThermalState;
        assert!(matches!(thermal_state_for(20.0), ThermalState::Nominal));
        assert!(matches!(thermal_state_for(60.0), ThermalState::Fair));
        assert!(matches!(thermal_state_for(72.0), ThermalState::Serious));
        assert!(matches!(thermal_state_for(85.0), ThermalState::Critical));
        assert!(matches!(thermal_state_for(95.0), ThermalState::Emergency));
        assert!(matches!(thermal_state_for(105.0), ThermalState::Shutdown));
    }
}
