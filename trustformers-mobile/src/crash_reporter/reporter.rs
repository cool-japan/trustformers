//! The `MobileCrashReporter` engine: crash detection, capture, reporting and its result/statistics types.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::scirs2_compat::random::legacy;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use trustformers_core::errors::{runtime_error, Result};

use super::analysis_types::{ImpactLevel, RecoveryImpact, RiskLevel};
use super::config::{CrashReporterConfig, RecoveryStrategy};
use super::engines::{
    CrashAnalysisEngine, CrashRecoveryManager, CrashStorageManager, SignalHandler,
};
use super::report_types::{
    AppCrashInfo, AppState, BatteryCrashInfo, CpuCrashInfo, CrashAnalysis, CrashContext,
    CrashReport, CrashSeverity, CrashType, ForegroundStatus, GpuCrashInfo, MemoryDump,
    MemoryUsageInfo, NetworkCrashInfo, RecoverySuggestion, StackTrace, SystemCrashInfo,
};

/// Crash information provided when reporting a crash
#[derive(Debug, Clone)]
pub struct CrashInfo {
    pub crash_type: CrashType,
    pub stack_trace: Option<StackTrace>,
    pub memory_dump: Option<MemoryDump>,
    pub context: CrashContext,
}
/// Crash statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashStatistics {
    pub total_crashes: usize,
    pub crash_types: HashMap<CrashType, usize>,
    pub severity_counts: HashMap<CrashSeverity, usize>,
    pub crash_free_sessions: usize,
    pub mean_time_between_crashes: Duration,
}
/// Export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Json,
    Csv,
    Html,
}
/// Main crash reporter
pub struct MobileCrashReporter {
    pub(super) config: CrashReporterConfig,
    pub(super) crash_history: Arc<RwLock<VecDeque<CrashReport>>>,
    pub(super) analysis_engine: Arc<Mutex<CrashAnalysisEngine>>,
    pub(super) storage_manager: Arc<Mutex<CrashStorageManager>>,
    pub(super) recovery_manager: Arc<Mutex<CrashRecoveryManager>>,
    pub(super) signal_handler: Option<SignalHandler>,
    pub(super) is_initialized: Arc<Mutex<bool>>,
}

impl MobileCrashReporter {
    /// Create new crash reporter
    pub fn new(config: CrashReporterConfig) -> Result<Self> {
        let analysis_engine = CrashAnalysisEngine::new(&config.analysis_config)?;
        let storage_manager = CrashStorageManager::new(&config.storage_config)?;
        let recovery_manager = CrashRecoveryManager::new(&config.recovery_config)?;

        Ok(Self {
            config,
            crash_history: Arc::new(RwLock::new(VecDeque::new())),
            analysis_engine: Arc::new(Mutex::new(analysis_engine)),
            storage_manager: Arc::new(Mutex::new(storage_manager)),
            recovery_manager: Arc::new(Mutex::new(recovery_manager)),
            signal_handler: None,
            is_initialized: Arc::new(Mutex::new(false)),
        })
    }

    /// Initialize crash reporter
    pub fn initialize(&mut self) -> Result<()> {
        {
            let mut initialized = self
                .is_initialized
                .lock()
                .map_err(|_| runtime_error("Failed to acquire lock"))?;

            if *initialized {
                return Ok(());
            }

            if !self.config.enabled {
                return Ok(());
            }

            *initialized = true;
        } // Drop the lock here

        // Setup signal handlers
        self.setup_signal_handlers()?;

        // Initialize storage
        self.storage_manager
            .lock()
            .map_err(|_| runtime_error("Failed to acquire lock"))?
            .initialize()?;

        // Load existing crash reports
        self.load_crash_history()?;

        Ok(())
    }

    /// Report a crash
    pub fn report_crash(&self, crash_info: CrashInfo) -> Result<String> {
        if !self.config.enabled {
            return Err(runtime_error("Crash reporter not enabled"));
        }

        let report_id = self.generate_report_id();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        // Collect system information
        let system_info = self.collect_system_info()?;
        let app_info = self.collect_app_info()?;

        // Create crash report
        let is_privacy_compliant = self.is_privacy_compliant(&crash_info);
        let mut crash_report = CrashReport {
            report_id: report_id.clone(),
            timestamp,
            crash_type: crash_info.crash_type,
            severity: self.assess_crash_severity(&crash_info),
            system_info,
            app_info,
            stack_trace: crash_info.stack_trace,
            memory_dump: crash_info.memory_dump,
            context: crash_info.context,
            analysis: None,
            recovery_suggestions: Vec::new(),
            is_privacy_compliant,
        };

        // Analyze crash if enabled
        if self.config.analysis_config.auto_analyze {
            if let Ok(analysis) = self.analyze_crash(&crash_report) {
                crash_report.analysis = Some(analysis);
            }
        }

        // Generate recovery suggestions
        crash_report.recovery_suggestions = self.generate_recovery_suggestions(&crash_report)?;

        // Store crash report
        self.store_crash_report(&crash_report)?;

        // Add to history
        let mut history = self
            .crash_history
            .write()
            .map_err(|_| runtime_error("Failed to acquire write lock"))?;
        history.push_back(crash_report.clone());

        // Limit history size
        while history.len() > 1000 {
            history.pop_front();
        }

        // Attempt recovery if enabled
        if self.config.recovery_config.auto_recovery {
            self.attempt_recovery(&crash_report)?;
        }

        // Report remotely if enabled
        if self.config.reporting_config.remote_reporting && self.has_user_consent() {
            self.report_remote(&crash_report)?;
        }

        Ok(report_id)
    }

    /// Get crash report by ID
    pub fn get_crash_report(&self, report_id: &str) -> Result<Option<CrashReport>> {
        let history = self
            .crash_history
            .read()
            .map_err(|_| runtime_error("Failed to acquire read lock"))?;

        Ok(history.iter().find(|report| report.report_id == report_id).cloned())
    }

    /// Get recent crash reports
    pub fn get_recent_crashes(&self, limit: Option<usize>) -> Result<Vec<CrashReport>> {
        let history = self
            .crash_history
            .read()
            .map_err(|_| runtime_error("Failed to acquire read lock"))?;

        let reports: Vec<CrashReport> =
            history.iter().rev().take(limit.unwrap_or(50)).cloned().collect();

        Ok(reports)
    }

    /// Get crash statistics
    pub fn get_crash_statistics(&self) -> Result<CrashStatistics> {
        let history = self
            .crash_history
            .read()
            .map_err(|_| runtime_error("Failed to acquire read lock"))?;

        let total_crashes = history.len();
        let mut crash_types = HashMap::new();
        let mut severity_counts = HashMap::new();

        for report in history.iter() {
            *crash_types.entry(report.crash_type).or_insert(0) += 1;
            *severity_counts.entry(report.severity).or_insert(0) += 1;
        }

        Ok(CrashStatistics {
            total_crashes,
            crash_types,
            severity_counts,
            crash_free_sessions: 0, // Would be calculated from session data
            mean_time_between_crashes: Duration::from_secs(0), // Would be calculated
        })
    }

    /// Clear crash history
    pub fn clear_crash_history(&self) -> Result<()> {
        let mut history = self
            .crash_history
            .write()
            .map_err(|_| runtime_error("Failed to acquire write lock"))?;
        history.clear();

        // Clear storage
        self.storage_manager
            .lock()
            .map_err(|_| runtime_error("Failed to acquire lock"))?
            .clear_all()?;

        Ok(())
    }

    /// Export crash reports
    pub fn export_crash_reports(&self, format: ExportFormat, output_path: &Path) -> Result<()> {
        let history = self
            .crash_history
            .read()
            .map_err(|_| runtime_error("Failed to acquire read lock"))?;

        match format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(&*history)
                    .map_err(|e| runtime_error(format!("Serialization error: {}", e)))?;
                fs::write(output_path, json)
                    .map_err(|e| runtime_error(format!("IO error: {}", e)))?;
            },
            ExportFormat::Csv => {
                let data: Vec<CrashReport> = history.iter().cloned().collect();
                self.export_csv(&data, output_path)?;
            },
            ExportFormat::Html => {
                let data: Vec<CrashReport> = history.iter().cloned().collect();
                self.export_html(&data, output_path)?;
            },
        }

        Ok(())
    }

    // Private helper methods

    pub(super) fn setup_signal_handlers(&mut self) -> Result<()> {
        if !self.config.platform_config.signal_config.async_safe {
            return Ok(()); // Skip if async-safe handling not enabled
        }

        // Platform-specific signal handler setup would go here
        // This is a simplified version
        Ok(())
    }

    pub(super) fn load_crash_history(&self) -> Result<()> {
        let storage_manager = self
            .storage_manager
            .lock()
            .map_err(|_| runtime_error("Failed to acquire lock"))?;

        let reports = storage_manager.load_all_reports()?;

        let mut history = self
            .crash_history
            .write()
            .map_err(|_| runtime_error("Failed to acquire write lock"))?;

        for report in reports {
            history.push_back(report);
        }

        Ok(())
    }

    pub(super) fn generate_report_id(&self) -> String {
        format!(
            "crash_{}_{}",
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos(),
            (legacy::f64() * u32::MAX as f64) as u32
        )
    }

    pub(super) fn collect_system_info(&self) -> Result<SystemCrashInfo> {
        // Collect comprehensive system information
        Ok(SystemCrashInfo {
            device_info: crate::device_info::MobileDeviceDetector::detect()?,
            performance_metrics: None, // Would be populated from profiler
            memory_usage: self.collect_memory_usage()?,
            cpu_info: self.collect_cpu_info()?,
            gpu_info: self.collect_gpu_info(),
            thermal_state: None, // Would be populated from thermal manager
            battery_info: self.collect_battery_info(),
            network_info: self.collect_network_info(),
        })
    }

    pub(super) fn collect_app_info(&self) -> Result<AppCrashInfo> {
        Ok(AppCrashInfo {
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            build_number: "1".to_string(), // Would be from build system
            framework_version: env!("CARGO_PKG_VERSION").to_string(),
            app_state: AppState::Active, // Would be determined at runtime
            foreground_status: ForegroundStatus::Foreground,
            session_duration: Duration::from_secs(0), // Would be tracked
            model_info: None,                         // Would be populated if model is active
            recent_operations: Vec::new(),            // Would be tracked
        })
    }

    /// Real memory figures via `sysinfo` (system-wide total/used/available,
    /// and this process's RSS as the `heap_mb` estimate -- `sysinfo` reports
    /// whole-process memory, not a heap/stack split, so `heap_mb` is that
    /// process figure and `stack_mb` is left at `0.0` rather than invented).
    /// Previously every field here was a hardcoded `0.0` regardless of
    /// actual memory pressure at crash time.
    pub(super) fn collect_memory_usage(&self) -> Result<MemoryUsageInfo> {
        use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

        let mut system = System::new();
        system.refresh_memory();
        let total_mb = system.total_memory() as f32 / (1024.0 * 1024.0);
        let used_mb = system.used_memory() as f32 / (1024.0 * 1024.0);
        let available_mb = system.available_memory() as f32 / (1024.0 * 1024.0);

        let pid = Pid::from_u32(std::process::id());
        let mut proc_system = System::new();
        proc_system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_memory(),
        );
        let heap_mb = proc_system
            .process(pid)
            .map(|p| p.memory() as f32 / (1024.0 * 1024.0))
            .unwrap_or(0.0);

        Ok(MemoryUsageInfo {
            total_mb,
            used_mb,
            available_mb,
            heap_mb,
            stack_mb: 0.0,
            gpu_mb: None,
        })
    }

    /// Real CPU figures via `sysinfo`: global usage percentage (needs two
    /// samples separated by `MINIMUM_CPU_UPDATE_INTERVAL`, mirroring the
    /// pattern already used by `mlx_integration::sample_process_usage`),
    /// the first detected core's clock, and the detected core count.
    /// `report_crash` is a plain method call (this crate's
    /// `setup_signal_handlers` is a documented no-op, not a real installed
    /// OS signal handler), so a short synchronous sleep here is safe --
    /// there is no async-signal-safety constraint to honor. Previously
    /// every field here was a hardcoded constant (`0.0` usage, `0` MHz, a
    /// fabricated `1` active core) regardless of the real system state.
    pub(super) fn collect_cpu_info(&self) -> Result<CpuCrashInfo> {
        use sysinfo::System;

        let mut system = System::new();
        system.refresh_cpu_usage();
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        system.refresh_cpu_usage();

        let usage_percent = system.global_cpu_usage();
        let frequency_mhz = system.cpus().first().map(|c| c.frequency()).unwrap_or(0) as u32;
        let active_cores = system.cpus().len().max(1);

        Ok(CpuCrashInfo {
            usage_percent,
            frequency_mhz,
            // `sysinfo` (with this crate's enabled feature set) does not
            // expose a portable per-platform CPU temperature/throttling
            // signal; reporting `None`/`false` here is the honest
            // "not measurable from this crate" answer, not a fabricated
            // reading.
            temperature_c: None,
            throttling: false,
            active_cores,
        })
    }

    pub(super) fn collect_gpu_info(&self) -> Option<GpuCrashInfo> {
        None // Would be implemented with platform-specific GPU APIs
    }

    pub(super) fn collect_battery_info(&self) -> Option<BatteryCrashInfo> {
        None // Would be implemented with platform-specific battery APIs
    }

    pub(super) fn collect_network_info(&self) -> Option<NetworkCrashInfo> {
        None // Would be implemented with platform-specific network APIs
    }

    pub(super) fn assess_crash_severity(&self, crash_info: &CrashInfo) -> CrashSeverity {
        match crash_info.crash_type {
            CrashType::SegmentationFault | CrashType::StackOverflow => CrashSeverity::Critical,
            CrashType::OutOfMemory => CrashSeverity::High,
            CrashType::UncaughtException => CrashSeverity::Medium,
            CrashType::ApplicationHang => CrashSeverity::Medium,
            _ => CrashSeverity::Low,
        }
    }

    pub(super) fn is_privacy_compliant(&self, _crash_info: &CrashInfo) -> bool {
        // Check if crash report complies with privacy settings
        !self.config.privacy_config.include_user_data || self.config.privacy_config.anonymize_data
    }

    pub(super) fn analyze_crash(&self, report: &CrashReport) -> Result<CrashAnalysis> {
        let analysis_engine = self
            .analysis_engine
            .lock()
            .map_err(|_| runtime_error("Failed to acquire lock"))?;

        analysis_engine.analyze_crash(report)
    }

    pub(super) fn generate_recovery_suggestions(
        &self,
        report: &CrashReport,
    ) -> Result<Vec<RecoverySuggestion>> {
        let mut suggestions = Vec::new();

        match report.crash_type {
            CrashType::OutOfMemory => {
                suggestions.push(RecoverySuggestion {
                    suggestion_type: RecoveryStrategy::ClearCache,
                    description: "Clear application cache to free memory".to_string(),
                    steps: vec![
                        "Clear model cache".to_string(),
                        "Clear temporary files".to_string(),
                        "Restart inference engine".to_string(),
                    ],
                    success_probability: 0.8,
                    risk_level: RiskLevel::Low,
                    impact: RecoveryImpact {
                        user_experience: ImpactLevel::Low,
                        performance: ImpactLevel::Low,
                        data_loss_risk: RiskLevel::Low,
                        recovery_time_estimate: Duration::from_secs(30),
                    },
                });
            },
            CrashType::SegmentationFault => {
                suggestions.push(RecoverySuggestion {
                    suggestion_type: RecoveryStrategy::SafeMode,
                    description: "Enable safe mode with reduced functionality".to_string(),
                    steps: vec![
                        "Disable GPU acceleration".to_string(),
                        "Reduce memory usage".to_string(),
                        "Disable advanced features".to_string(),
                    ],
                    success_probability: 0.9,
                    risk_level: RiskLevel::Low,
                    impact: RecoveryImpact {
                        user_experience: ImpactLevel::Medium,
                        performance: ImpactLevel::High,
                        data_loss_risk: RiskLevel::Low,
                        recovery_time_estimate: Duration::from_secs(60),
                    },
                });
            },
            _ => {
                suggestions.push(RecoverySuggestion {
                    suggestion_type: RecoveryStrategy::RestartApp,
                    description: "Restart application to clear problematic state".to_string(),
                    steps: vec!["Restart application".to_string()],
                    success_probability: 0.7,
                    risk_level: RiskLevel::Medium,
                    impact: RecoveryImpact {
                        user_experience: ImpactLevel::High,
                        performance: ImpactLevel::Low,
                        data_loss_risk: RiskLevel::Medium,
                        recovery_time_estimate: Duration::from_secs(10),
                    },
                });
            },
        }

        Ok(suggestions)
    }

    pub(super) fn store_crash_report(&self, report: &CrashReport) -> Result<()> {
        let storage_manager = self
            .storage_manager
            .lock()
            .map_err(|_| runtime_error("Failed to acquire lock"))?;

        storage_manager.store_report(report)
    }

    pub(super) fn attempt_recovery(&self, report: &CrashReport) -> Result<()> {
        let mut recovery_manager = self
            .recovery_manager
            .lock()
            .map_err(|_| runtime_error("Failed to acquire lock"))?;

        recovery_manager.attempt_recovery(report)
    }

    pub(super) fn has_user_consent(&self) -> bool {
        if !self.config.privacy_config.require_consent {
            return true;
        }

        // Would check user consent storage
        false
    }

    pub(super) fn report_remote(&self, _report: &CrashReport) -> Result<()> {
        // Would implement remote reporting
        Ok(())
    }

    pub(super) fn export_csv(&self, reports: &[CrashReport], output_path: &Path) -> Result<()> {
        let mut csv_content = String::new();
        csv_content.push_str("Report ID,Timestamp,Crash Type,Severity,App Version\n");

        for report in reports {
            csv_content.push_str(&format!(
                "{},{},{:?},{:?},{}\n",
                report.report_id,
                report.timestamp,
                report.crash_type,
                report.severity,
                report.app_info.app_version
            ));
        }

        fs::write(output_path, csv_content)
            .map_err(|e| runtime_error(format!("IO error: {}", e)))?;

        Ok(())
    }

    pub(super) fn export_html(&self, reports: &[CrashReport], output_path: &Path) -> Result<()> {
        let mut html = String::new();
        html.push_str("<!DOCTYPE html><html><head><title>Crash Reports</title></head><body>");
        html.push_str("<h1>Crash Reports</h1>");
        html.push_str("<table border='1'>");
        html.push_str(
            "<tr><th>Report ID</th><th>Timestamp</th><th>Type</th><th>Severity</th></tr>",
        );

        for report in reports {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{:?}</td><td>{:?}</td></tr>",
                report.report_id, report.timestamp, report.crash_type, report.severity
            ));
        }

        html.push_str("</table></body></html>");

        fs::write(output_path, html).map_err(|e| runtime_error(format!("IO error: {}", e)))?;

        Ok(())
    }
}
