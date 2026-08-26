//! Platform-specific profiler backends (iOS/Android/generic) implementing the `PlatformProfiler` trait.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use trustformers_core::error::Result;
use trustformers_core::TrustformersError;

use super::collector::{ExportFormat, ProfilerCapability};
use super::metrics_types::{
    CpuMetrics, MemoryMetrics, MemoryPressureLevel, NetworkMetrics, PlatformMetrics,
};

pub struct AndroidProfiler {
    pub(super) systrace_integration: bool,
    pub(super) perfetto_integration: bool,
    pub(super) capabilities: Vec<ProfilerCapability>,
}
impl AndroidProfiler {
    pub fn new() -> Result<Self> {
        Ok(Self {
            systrace_integration: Self::check_systrace_availability(),
            perfetto_integration: Self::check_perfetto_availability(),
            capabilities: vec![
                ProfilerCapability::CpuProfiling,
                ProfilerCapability::MemoryProfiling,
                ProfilerCapability::GpuProfiling,
                ProfilerCapability::NetworkProfiling,
                ProfilerCapability::SystraceIntegration,
                ProfilerCapability::PerfettoIntegration,
            ],
        })
    }

    /// Whether systrace is actually tracing this process. Checking that
    /// for real needs the `android.os.Trace` JNI API (or reading
    /// `/sys/kernel/tracing/tracing_on`, which typically needs the
    /// `adb shell` debug user, not this app's own runtime uid); this crate
    /// wires up neither, so the honest answer is always `false` -- not the
    /// unconditional `true` a previous revision returned on every Android
    /// build regardless of whether systrace was attached at all.
    pub(super) fn check_systrace_availability() -> bool {
        false
    }

    /// Same honesty policy as [`Self::check_systrace_availability`] for
    /// Perfetto: no real integration hook exists in this crate.
    pub(super) fn check_perfetto_availability() -> bool {
        false
    }
}
impl PlatformProfiler for AndroidProfiler {
    fn start_profiling(&mut self) -> Result<()> {
        // Start Android-specific profiling using systrace, Perfetto, etc.
        Ok(())
    }

    fn stop_profiling(&mut self) -> Result<()> {
        // Stop Android-specific profiling
        Ok(())
    }

    fn collect_metrics(&self) -> Result<PlatformMetrics> {
        Ok(collect_real_platform_metrics())
    }

    fn export_data(&self, format: ExportFormat) -> Result<Vec<u8>> {
        match format {
            // A real export of this profiler's own currently-collected
            // metrics.
            ExportFormat::JSON => serde_json::to_vec(&self.collect_metrics()?).map_err(|e| {
                TrustformersError::runtime_error(format!("failed to serialize metrics: {e}")).into()
            }),
            ExportFormat::Trace | ExportFormat::Perfetto => {
                // Both systrace's and Perfetto's real trace formats
                // (Perfetto's is a specific protobuf schema) are binary
                // formats this crate does not implement an encoder for,
                // and there is no local tool on this machine that could
                // verify a hand-rolled one. `Ok(vec![])`, as a previous
                // revision returned, is indistinguishable from "exported
                // an empty trace" -- an honest error is correct until a
                // real encoder exists.
                Err(TrustformersError::runtime_error(format!(
                    "{format:?} export is not implemented: this crate does not write systrace's \
                     or Perfetto's real trace formats. Use ExportFormat::JSON for a real export \
                     of the collected metrics."
                ))
                .into())
            },
            _ => Err(TrustformersError::config_error(
                "Export format not supported on Android",
                "export_android_data",
            )
            .into()),
        }
    }

    fn get_capabilities(&self) -> Vec<ProfilerCapability> {
        self.capabilities.clone()
    }
}
pub struct GenericProfiler {
    pub(super) capabilities: Vec<ProfilerCapability>,
}
impl GenericProfiler {
    pub fn new() -> Result<Self> {
        Ok(Self {
            capabilities: vec![
                ProfilerCapability::CpuProfiling,
                ProfilerCapability::MemoryProfiling,
                ProfilerCapability::NetworkProfiling,
                ProfilerCapability::CustomProfiling,
            ],
        })
    }
}
impl PlatformProfiler for GenericProfiler {
    fn start_profiling(&mut self) -> Result<()> {
        // Start generic profiling
        Ok(())
    }

    fn stop_profiling(&mut self) -> Result<()> {
        // Stop generic profiling
        Ok(())
    }

    fn collect_metrics(&self) -> Result<PlatformMetrics> {
        Ok(collect_real_platform_metrics())
    }

    fn export_data(&self, format: ExportFormat) -> Result<Vec<u8>> {
        match format {
            ExportFormat::JSON => serde_json::to_vec(&self.collect_metrics()?).map_err(|e| {
                TrustformersError::runtime_error(format!("failed to serialize metrics: {e}")).into()
            }),
            // A real (if minimal, hand-rolled for this one fixed schema)
            // CSV row of the collected metrics -- not the empty
            // placeholder a previous revision returned for both formats.
            ExportFormat::CSV => {
                let metrics = self.collect_metrics()?;
                let header = "cpu_utilization_percent,memory_used_mb,memory_available_mb,\
                               memory_pressure_level\n";
                let row = format!(
                    "{},{},{},{:?}\n",
                    metrics.cpu_metrics.utilization_percent,
                    metrics.memory_metrics.total_usage_mb,
                    metrics.memory_metrics.available_mb,
                    metrics.memory_metrics.pressure_level
                );
                Ok([header, &row].concat().into_bytes())
            },
            _ => Err(TrustformersError::config_error(
                "Export format not supported",
                "export_profiling_data",
            )
            .into()),
        }
    }

    fn get_capabilities(&self) -> Vec<ProfilerCapability> {
        self.capabilities.clone()
    }
}
pub struct IOSProfiler {
    pub(super) instruments_integration: bool,
    pub(super) capabilities: Vec<ProfilerCapability>,
}
impl IOSProfiler {
    pub fn new() -> Result<Self> {
        Ok(Self {
            instruments_integration: Self::check_instruments_availability(),
            capabilities: vec![
                ProfilerCapability::CpuProfiling,
                ProfilerCapability::MemoryProfiling,
                ProfilerCapability::GpuProfiling,
                ProfilerCapability::ThermalProfiling,
                ProfilerCapability::BatteryProfiling,
                ProfilerCapability::InstrumentsIntegration,
            ],
        })
    }

    /// Whether Apple's Instruments tooling is actually attached to this
    /// process. This crate has no `os_signpost`/DTrace-style hook to check
    /// that for real (doing so needs linking Apple's profiling frameworks,
    /// which this pure-Rust crate does not), so the honest answer is
    /// always `false` -- not the unconditional `true` a previous revision
    /// returned on every iOS build regardless of whether Instruments was
    /// running at all.
    pub(super) fn check_instruments_availability() -> bool {
        false
    }
}
impl PlatformProfiler for IOSProfiler {
    fn start_profiling(&mut self) -> Result<()> {
        // Start iOS-specific profiling using Core Animation Time Profiler, etc.
        Ok(())
    }

    fn stop_profiling(&mut self) -> Result<()> {
        // Stop iOS-specific profiling
        Ok(())
    }

    fn collect_metrics(&self) -> Result<PlatformMetrics> {
        Ok(collect_real_platform_metrics())
    }

    fn export_data(&self, format: ExportFormat) -> Result<Vec<u8>> {
        match format {
            // A real export of this profiler's own currently-collected
            // metrics (`serde_json::to_vec`, not an empty placeholder).
            ExportFormat::JSON => serde_json::to_vec(&self.collect_metrics()?).map_err(|e| {
                TrustformersError::runtime_error(format!("failed to serialize metrics: {e}")).into()
            }),
            ExportFormat::Instruments => {
                // Apple's Instruments `.trace` bundle is a proprietary
                // binary format this crate does not implement a writer
                // for (unlike `coreml_proto`'s CoreML protobuf subset,
                // there is no local Apple tool on this machine that can
                // verify a hand-rolled encoding of it). Returning `Ok(vec![])`
                // here, as a previous revision did, would be indistinguishable
                // from "successfully exported an empty trace" -- an honest
                // error is the only correct answer until a real encoder
                // exists.
                Err(TrustformersError::runtime_error(
                    "Instruments (.trace) export is not implemented: this crate does not write \
                     Apple's proprietary Instruments trace format. Use ExportFormat::JSON for a \
                     real export of the collected metrics."
                        .to_string(),
                )
                .into())
            },
            _ => Err(TrustformersError::config_error(
                "Export format not supported on iOS",
                "export_ios_data",
            )
            .into()),
        }
    }

    fn get_capabilities(&self) -> Vec<ProfilerCapability> {
        self.capabilities.clone()
    }
}
/// Platform-specific profiler trait
pub trait PlatformProfiler {
    /// Start platform-specific profiling
    fn start_profiling(&mut self) -> Result<()>;
    /// Stop platform-specific profiling
    fn stop_profiling(&mut self) -> Result<()>;
    /// Collect platform-specific metrics
    fn collect_metrics(&self) -> Result<PlatformMetrics>;
    /// Export profiling data
    fn export_data(&self, format: ExportFormat) -> Result<Vec<u8>>;
    /// Get platform capabilities
    fn get_capabilities(&self) -> Vec<ProfilerCapability>;
}

/// Real CPU/memory figures via `sysinfo`, shared by every
/// [`PlatformProfiler::collect_metrics`] implementation below. Mirrors
/// `crash_reporter`'s `collect_cpu_info`/`collect_memory_usage` (same
/// crate, same already-vetted pattern: two CPU-usage samples separated by
/// `sysinfo::MINIMUM_CPU_UPDATE_INTERVAL`, this process's RSS for the
/// "heap"-shaped figure). GPU and detailed per-connection network metrics
/// have no portable `sysinfo` source, so [`PlatformMetrics::gpu_metrics`]
/// stays `None` and [`PlatformMetrics::network_metrics`] stays at its
/// honest zero [`Default`] -- not fabricated. Previously every
/// `collect_metrics` implementation in this file was
/// `Ok(PlatformMetrics::default())`, i.e. these same zero defaults
/// presented as if they were a real collection result regardless of
/// whether profiling had even started.
pub(super) fn collect_real_platform_metrics() -> PlatformMetrics {
    use sysinfo::System;

    let mut system = System::new();
    system.refresh_cpu_usage();
    std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
    system.refresh_cpu_usage();
    system.refresh_memory();

    let per_core_utilization: Vec<f32> = system.cpus().iter().map(|c| c.cpu_usage()).collect();
    let frequency_mhz: Vec<u32> = system.cpus().iter().map(|c| c.frequency() as u32).collect();
    let utilization_percent = system.global_cpu_usage();
    let load_average = {
        let load = System::load_average();
        [load.one as f32, load.five as f32, load.fifteen as f32]
    };

    let total_mb = (system.total_memory() / (1024 * 1024)) as usize;
    let used_mb = (system.used_memory() / (1024 * 1024)) as usize;
    let available_mb = (system.available_memory() / (1024 * 1024)) as usize;
    let usage_fraction = if total_mb > 0 { used_mb as f32 / total_mb as f32 } else { 0.0 };
    let pressure_level = if usage_fraction >= 0.95 {
        MemoryPressureLevel::Critical
    } else if usage_fraction >= 0.85 {
        MemoryPressureLevel::High
    } else if usage_fraction >= 0.6 {
        MemoryPressureLevel::Medium
    } else {
        MemoryPressureLevel::Low
    };

    PlatformMetrics {
        cpu_metrics: CpuMetrics {
            utilization_percent,
            per_core_utilization,
            frequency_mhz,
            // `sysinfo` (with this crate's enabled feature set) does not
            // expose a portable context-switch counter or a user/kernel
            // time split; left at the honest `Default` zero rather than
            // invented.
            context_switches_per_sec: 0,
            load_average,
            user_time_percent: 0.0,
            kernel_time_percent: 0.0,
            idle_time_percent: (100.0 - utilization_percent).max(0.0),
        },
        gpu_metrics: None,
        memory_metrics: MemoryMetrics {
            total_usage_mb: used_mb,
            available_mb,
            pressure_level,
            page_faults_per_sec: 0,
            allocations_per_sec: 0,
            deallocations_per_sec: 0,
            gc_metrics: None,
        },
        network_metrics: NetworkMetrics::default(),
        platform_specific: HashMap::new(),
    }
}
