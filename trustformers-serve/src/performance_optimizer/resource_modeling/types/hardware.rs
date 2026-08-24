//! Hardware model types for resource modeling
//!
//! Main hardware modeling structures including managers, profilers, monitors,
//! analyzers, trackers, and detectors.

use super::traits_analysis::MeasurementUnavailable;
use super::{
    config::*, detection::*, enums::*, monitoring::*, profiling::*, topology::*,
    traits::VendorDetector,
};
use crate::performance_optimizer::resource_modeling::hardware_detector::{
    CpuDetectionConfig, CpuDetector, GpuDetectionConfig, GpuDetector, MemoryDetectionConfig,
    MemoryDetector,
};
use crate::performance_optimizer::resource_modeling::manager::ResourceModelingConfig;
use crate::performance_optimizer::types::{
    CacheHierarchy, GpuDeviceModel, MemoryType, NumaTopology, SystemResourceModel,
    TemperatureMetrics,
};
use parking_lot::{Mutex, RwLock};
use std::{collections::HashMap, sync::Arc};

/// Resource modeling manager for comprehensive system analysis
///
/// Main coordinator for resource modeling operations including hardware detection,
/// performance profiling, thermal monitoring, and system topology analysis.
#[derive(Debug, Clone)]
pub struct ResourceModelingManager {
    /// Current system resource model
    pub resource_model: Arc<RwLock<SystemResourceModel>>,

    /// System information provider
    pub system_info: Arc<Mutex<sysinfo::System>>,

    /// Performance profiling engine
    pub performance_profiler: Arc<PerformanceProfiler>,

    /// Temperature monitoring system
    pub temperature_monitor: Arc<TemperatureMonitor>,

    /// Topology analyzer
    pub topology_analyzer: Arc<TopologyAnalyzer>,

    /// Resource utilization tracker
    pub utilization_tracker: Arc<ResourceUtilizationTracker>,

    /// Hardware detection engine
    pub hardware_detector: Arc<HardwareDetector>,

    /// Modeling configuration
    pub config: ResourceModelingConfig,
}

/// Performance profiling engine for hardware characterization
///
/// A coordination shim: it owns the five per-subsystem profile caches that
/// [`ResourceModelingManager`] hands around, but it runs no benchmarks itself.
/// The engine that does is
/// [`resource_modeling::performance_profiler`](crate::performance_optimizer::resource_modeling::performance_profiler).
///
/// ## Changed in 0.2.1: `profile_*_performance` no longer returns a zeroed profile
///
/// Each of the five methods returned `Default::default()` for its profile
/// type — every score, bandwidth and latency `0.0` — and reported success.
/// `ComponentCoordinator::execute_performance_profiling` collected the five
/// into a `PerformanceProfileResults`, stamped it with the current time and
/// filed it as a profiling result, so downstream code could not distinguish
/// "the machine measured zero" from "nothing was measured". They now return
/// [`MeasurementUnavailable`].
#[derive(Debug, Clone)]
pub struct PerformanceProfiler {
    /// CPU profiling results
    pub cpu_profiles: Arc<Mutex<HashMap<String, CpuProfile>>>,

    /// Memory profiling results
    pub memory_profiles: Arc<Mutex<HashMap<String, MemoryProfile>>>,

    /// I/O profiling results
    pub io_profiles: Arc<Mutex<HashMap<String, IoProfile>>>,

    /// Network profiling results
    pub network_profiles: Arc<Mutex<HashMap<String, NetworkProfile>>>,

    /// GPU profiling results
    pub gpu_profiles: Arc<Mutex<HashMap<String, GpuProfile>>>,

    /// Profiling configuration
    pub config: ProfilingConfig,
}

impl PerformanceProfiler {
    /// Create a new PerformanceProfiler with default values
    pub fn new(config: ProfilingConfig) -> Self {
        Self {
            cpu_profiles: Arc::new(Mutex::new(HashMap::new())),
            memory_profiles: Arc::new(Mutex::new(HashMap::new())),
            io_profiles: Arc::new(Mutex::new(HashMap::new())),
            network_profiles: Arc::new(Mutex::new(HashMap::new())),
            gpu_profiles: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }

    /// Profile CPU performance.
    ///
    /// See the type-level note: this returns [`MeasurementUnavailable`]
    /// instead of a zeroed [`CpuProfile`].
    pub async fn profile_cpu_performance(&self) -> anyhow::Result<CpuProfile> {
        Err(Self::no_profile("a CPU performance profile"))
    }

    /// Profile memory performance.
    ///
    /// Returns [`MeasurementUnavailable`]; see the type-level note.
    pub async fn profile_memory_performance(&self) -> anyhow::Result<MemoryProfile> {
        Err(Self::no_profile("a memory performance profile"))
    }

    /// Profile I/O performance.
    ///
    /// Returns [`MeasurementUnavailable`]; see the type-level note.
    pub async fn profile_io_performance(&self) -> anyhow::Result<IoProfile> {
        Err(Self::no_profile("an I/O performance profile"))
    }

    /// Profile network performance.
    ///
    /// Returns [`MeasurementUnavailable`]; see the type-level note.
    pub async fn profile_network_performance(&self) -> anyhow::Result<NetworkProfile> {
        Err(Self::no_profile("a network performance profile"))
    }

    /// Profile GPU performance.
    ///
    /// Returns [`MeasurementUnavailable`]; see the type-level note.
    pub async fn profile_gpu_performance(&self) -> anyhow::Result<GpuProfile> {
        Err(Self::no_profile("a GPU performance profile"))
    }

    fn no_profile(what: &'static str) -> anyhow::Error {
        MeasurementUnavailable::raise(
            what,
            "this coordination shim runs no benchmarks; the profiling engine is \
             resource_modeling::performance_profiler",
        )
    }
}

/// Temperature monitoring system for thermal management
///
/// Real-time temperature monitoring system with history tracking,
/// threshold management, and thermal state analysis for system protection.
#[derive(Debug, Clone)]
pub struct TemperatureMonitor {
    /// Temperature history
    pub temperature_history: Arc<Mutex<Vec<TemperatureReading>>>,

    /// Temperature thresholds
    pub thresholds: TemperatureThresholds,

    /// Thermal management state
    pub thermal_state: Arc<Mutex<ThermalState>>,
}

impl TemperatureMonitor {
    /// Create a new TemperatureMonitor with default values
    pub fn new(thresholds: TemperatureThresholds) -> Self {
        Self {
            temperature_history: Arc::new(Mutex::new(Vec::new())),
            thresholds,
            thermal_state: Arc::new(Mutex::new(ThermalState::default())),
        }
    }

    /// Read the hottest temperature the platform's thermal components report,
    /// and append it to [`Self::temperature_history`].
    ///
    /// Returns [`MeasurementUnavailable`] when the platform exposes no
    /// components, or none of them has a reading — a common case on macOS and
    /// inside containers. It used to return the literal `45.0` °C on every
    /// call, on every machine, which
    /// [`ComponentCoordinator::execute_temperature_monitoring`] then compared
    /// against its 85 °C throttling threshold: a thermal check that could
    /// never fire.
    ///
    /// [`ComponentCoordinator::execute_temperature_monitoring`]: crate::performance_optimizer::resource_modeling::manager::ComponentCoordinator::execute_temperature_monitoring
    pub async fn get_current_temperature(&self) -> anyhow::Result<f32> {
        let components = sysinfo::Components::new_with_refreshed_list();
        let hottest = components
            .iter()
            .filter_map(|component| component.temperature())
            .fold(None::<f32>, |acc, t| Some(acc.map_or(t, |a| a.max(t))));

        match hottest {
            Some(temperature) => {
                self.temperature_history.lock().push(TemperatureReading {
                    timestamp: chrono::Utc::now(),
                    metrics: TemperatureMetrics {
                        cpu_temperature: temperature,
                        gpu_temperature: None,
                        system_temperature: temperature,
                        thermal_throttling: temperature >= self.thresholds.critical_temperature,
                    },
                });
                Ok(temperature)
            },
            None => Err(MeasurementUnavailable::raise(
                "a system temperature",
                "this platform exposes no readable thermal components",
            )),
        }
    }
}

/// Topology analyzer for hardware layout optimization
///
/// System topology analysis engine for NUMA detection, cache hierarchy analysis,
/// memory topology characterization, and I/O layout optimization.
#[derive(Debug, Clone)]
pub struct TopologyAnalyzer {
    /// NUMA topology cache
    pub numa_topology: Arc<Mutex<Option<NumaTopology>>>,

    /// Cache hierarchy analysis
    pub cache_analysis: Arc<Mutex<CacheAnalysis>>,

    /// Memory topology
    pub memory_topology: Arc<Mutex<MemoryTopology>>,

    /// I/O topology
    pub io_topology: Arc<Mutex<IoTopology>>,
}

impl Default for TopologyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl TopologyAnalyzer {
    /// Create a new TopologyAnalyzer with default values
    pub fn new() -> Self {
        Self {
            numa_topology: Arc::new(Mutex::new(None)),
            cache_analysis: Arc::new(Mutex::new(CacheAnalysis::default())),
            memory_topology: Arc::new(Mutex::new(MemoryTopology::default())),
            io_topology: Arc::new(Mutex::new(IoTopology::default())),
        }
    }

    /// Analyze complete system topology.
    ///
    /// Returns [`MeasurementUnavailable`]. This returned `Ok(())` without
    /// touching any of its four caches, after which
    /// `ComponentCoordinator::execute_topology_analysis` built a
    /// `TopologyAnalysisResults` out of `Default::default()` for every field
    /// and reported it as an analysis result — an empty NUMA topology and
    /// zeroed cache, memory and I/O topologies, presented as the machine's
    /// actual layout. The live topology analyser is
    /// [`resource_modeling::topology_analyzer`](crate::performance_optimizer::resource_modeling::topology_analyzer).
    pub async fn analyze_complete_topology(&self) -> anyhow::Result<()> {
        Err(MeasurementUnavailable::raise(
            "system topology",
            "this coordination shim performs no topology detection; the analyser is \
             resource_modeling::topology_analyzer",
        ))
    }
}

/// Resource utilization tracker for continuous monitoring
///
/// Continuous monitoring system for tracking resource utilization across
/// all system components with configurable history and sampling rates.
#[derive(Debug, Clone)]
pub struct ResourceUtilizationTracker {
    /// CPU utilization history
    pub cpu_utilization: Arc<Mutex<UtilizationHistory<f32>>>,

    /// Memory utilization history
    pub memory_utilization: Arc<Mutex<UtilizationHistory<f32>>>,

    /// I/O utilization history
    pub io_utilization: Arc<Mutex<UtilizationHistory<f32>>>,

    /// Network utilization history
    pub network_utilization: Arc<Mutex<UtilizationHistory<f32>>>,

    /// GPU utilization history
    pub gpu_utilization: Arc<Mutex<UtilizationHistory<f32>>>,

    /// Tracking configuration
    pub config: UtilizationTrackingConfig,
}

impl ResourceUtilizationTracker {
    /// Create a new ResourceUtilizationTracker with default values
    pub fn new(config: UtilizationTrackingConfig) -> Self {
        Self {
            cpu_utilization: Arc::new(Mutex::new(UtilizationHistory::new(1000))),
            memory_utilization: Arc::new(Mutex::new(UtilizationHistory::new(1000))),
            io_utilization: Arc::new(Mutex::new(UtilizationHistory::new(1000))),
            network_utilization: Arc::new(Mutex::new(UtilizationHistory::new(1000))),
            gpu_utilization: Arc::new(Mutex::new(UtilizationHistory::new(1000))),
            config,
        }
    }

    /// Start monitoring resource utilization.
    ///
    /// Returns [`MeasurementUnavailable`]. Nothing was ever started: the five
    /// utilisation histories stayed empty, and
    /// `ComponentCoordinator::execute_utilization_tracking` read the `Ok(())`
    /// as permission to emit a `UtilizationReport` whose every statistic —
    /// average, minimum, maximum, standard deviation, p95, p99 — was `0.0`.
    /// The live tracker is
    /// [`resource_modeling::utilization_tracker`](crate::performance_optimizer::resource_modeling::utilization_tracker).
    pub async fn start_monitoring(&self) -> anyhow::Result<()> {
        Err(MeasurementUnavailable::raise(
            "resource utilization tracking",
            "this coordination shim starts no sampling loop; the tracker is \
             resource_modeling::utilization_tracker",
        ))
    }
}

/// Hardware detection engine with vendor-specific capabilities
///
/// Comprehensive hardware detection system with support for multiple vendors,
/// result caching, and extensible detection algorithms.
#[derive(Debug)]
pub struct HardwareDetector {
    /// Detection cache
    pub detection_cache: Arc<Mutex<HardwareDetectionCache>>,

    /// Vendor-specific detectors
    pub vendor_detectors: Vec<Box<dyn VendorDetector + Send + Sync>>,

    /// Detection configuration
    pub config: HardwareDetectionConfig,
}

impl HardwareDetector {
    /// Create a new HardwareDetector with default values
    pub fn new(config: HardwareDetectionConfig) -> Self {
        Self {
            detection_cache: Arc::new(Mutex::new(HardwareDetectionCache::default())),
            vendor_detectors: Vec::new(),
            config,
        }
    }

    /// Detect CPU base and boost frequencies, in MHz.
    ///
    /// Delegates to
    /// [`hardware_detector::CpuDetector`](crate::performance_optimizer::resource_modeling::hardware_detector::CpuDetector),
    /// which reads them from the platform. This used to answer `(2400, 3600)`
    /// for every CPU it was ever asked about.
    pub async fn detect_cpu_frequencies(&self) -> anyhow::Result<(u32, u32)> {
        CpuDetector::new(CpuDetectionConfig::default())
            .await?
            .detect_cpu_frequencies()
            .await
    }

    /// Detect the cache hierarchy.
    ///
    /// Delegates to
    /// [`hardware_detector::CpuDetector`](crate::performance_optimizer::resource_modeling::hardware_detector::CpuDetector).
    /// This used to return a fixed 32 KiB / 256 KiB / 8 MiB hierarchy with a
    /// 64-byte line, which is a plausible x86 desktop and wrong about most
    /// other machines.
    pub async fn detect_cache_hierarchy(&self) -> anyhow::Result<CacheHierarchy> {
        CpuDetector::new(CpuDetectionConfig::default())
            .await?
            .detect_cache_hierarchy()
            .await
    }

    /// Detect memory type, speed, bandwidth and latency.
    ///
    /// Delegates to
    /// [`hardware_detector::MemoryDetector`](crate::performance_optimizer::resource_modeling::hardware_detector::MemoryDetector).
    /// This used to report DDR4 at 2400 MHz, 51.2 GB/s and 14 ns
    /// unconditionally — including on DDR5 and on unified-memory machines.
    pub async fn detect_memory_characteristics(
        &self,
    ) -> anyhow::Result<(MemoryType, u32, f32, std::time::Duration)> {
        MemoryDetector::new(MemoryDetectionConfig::default())
            .await?
            .detect_memory_characteristics()
            .await
    }

    /// Detect GPU devices.
    ///
    /// Delegates to
    /// [`hardware_detector::GpuDetector`](crate::performance_optimizer::resource_modeling::hardware_detector::GpuDetector).
    pub async fn detect_gpu_devices(&self) -> anyhow::Result<Vec<GpuDeviceModel>> {
        GpuDetector::new(GpuDetectionConfig::default())
            .await?
            .detect_gpu_devices()
            .await
    }
}
