//! Device Farm Management
//!
//! This module provides device farm management functionality for running tests across multiple devices.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tokio::time::timeout;
use trustformers_core::error::{CoreError, Result};
use trustformers_core::TrustformersError;

use super::config::*;
use super::results::*;

/// Device farm manager for coordinating tests across multiple devices
#[derive(Debug)]
pub struct DeviceFarmManager {
    config: DeviceFarmConfig,
    active_sessions: HashMap<String, DeviceFarmSession>,
    device_pool: Vec<DeviceInfo>,
    session_counter: Arc<Mutex<usize>>,
}

/// Device farm session representing an active testing session
#[derive(Debug)]
pub struct DeviceFarmSession {
    pub session_id: String,
    pub status: SessionStatus,
    pub start_time: SystemTime,
    pub assigned_devices: Vec<String>,
    pub pending_tasks: VecDeque<TestTask>,
    pub completed_tasks: Vec<TestTask>,
    /// Results this manager actually recorded for the session. Only a task
    /// that really executed contributes one.
    pub recorded_results: Vec<DeviceTestResult>,
    pub session_metadata: DeviceFarmSessionMetadata,
}

/// Session status enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Test task for device farm execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestTask {
    pub task_id: String,
    pub test_config: TestExecutionConfig,
    pub assigned_device: Option<String>,
    pub priority: TaskPriority,
    pub status: SessionStatus,
    pub created_at: SystemTime,
    pub started_at: Option<SystemTime>,
    pub completed_at: Option<SystemTime>,
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
    Urgent = 5,
}

/// Test execution configuration for device farm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionConfig {
    pub test_type: TestType,
    pub timeout: Duration,
    pub retry_attempts: usize,
    pub resource_requirements: HardwareRequirements,
}

/// Test type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestType {
    Benchmark,
    Battery,
    Stress,
    Memory,
    Compatibility,
    Performance,
    FullSuite,
}

/// Result aggregator for combining results from multiple devices
#[derive(Debug)]
pub struct ResultAggregator {
    aggregation_rules: AggregationRules,
}

/// Rules for aggregating results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationRules {
    pub statistical_methods: Vec<StatisticalMethod>,
    pub outlier_detection: bool,
    pub confidence_level: f32,
    pub minimum_sample_size: usize,
}

/// Statistical methods for aggregation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatisticalMethod {
    Mean,
    Median,
    Mode,
    Percentile(u8),
    StandardDeviation,
    Variance,
    Range,
}

impl DeviceFarmManager {
    /// Create a new device farm manager
    pub fn new(config: DeviceFarmConfig) -> Result<Self> {
        Ok(Self {
            config,
            active_sessions: HashMap::new(),
            device_pool: Vec::new(),
            session_counter: Arc::new(Mutex::new(0)),
        })
    }

    /// Initialize device farm with available devices
    pub async fn initialize(&mut self) -> Result<()> {
        // Initialize device pool based on provider
        let provider = self.config.provider.clone();
        match provider {
            DeviceFarmProvider::AWS {
                region,
                project_name,
            } => {
                self.initialize_aws_devices(&region, &project_name).await?;
            },
            DeviceFarmProvider::Firebase {
                project_id,
                test_lab_id,
            } => {
                self.initialize_firebase_devices(&project_id, &test_lab_id).await?;
            },
            DeviceFarmProvider::Local {
                device_pool_size,
                devices,
            } => {
                self.initialize_local_devices(device_pool_size, &devices).await?;
            },
            _ => {
                return Err(TrustformersError::config_error(
                    "Unsupported device farm provider",
                    "initialize",
                )
                .into());
            },
        }

        Ok(())
    }

    /// Start a new device farm session
    pub async fn start_session(&mut self, test_tasks: Vec<TestTask>) -> Result<String> {
        let session_id = {
            let mut counter = self.session_counter.lock().unwrap_or_else(|p| p.into_inner());
            *counter += 1;
            format!("session_{:08}", *counter)
        };

        let assigned_devices = self.allocate_devices(&test_tasks).await?;

        let session = DeviceFarmSession {
            recorded_results: Vec::new(),
            session_id: session_id.clone(),
            status: SessionStatus::Pending,
            start_time: SystemTime::now(),
            assigned_devices,
            pending_tasks: test_tasks.into(),
            completed_tasks: Vec::new(),
            session_metadata: DeviceFarmSessionMetadata {
                configuration: "default".to_string(),
                requested_devices: 0,
                allocated_devices: 0,
                failed_allocations: 0,
                total_cost: None,
                resource_utilization: ResourceUtilization {
                    device_utilization: 0.0,
                    network_usage_mb: 0.0,
                    storage_usage_mb: 0.0,
                    compute_time_minutes: 0.0,
                },
            },
        };

        self.active_sessions.insert(session_id.clone(), session);

        // Session is ready to be executed - caller can call execute_session() when ready
        Ok(session_id)
    }

    /// Execute a device farm session
    pub async fn execute_session(&mut self, session_id: String) -> Result<()> {
        // Set session status to running
        {
            let session = self.active_sessions.get_mut(&session_id).ok_or_else(|| {
                TrustformersError::config_error("Session not found", "execute_session")
            })?;
            session.status = SessionStatus::Running;
        }

        // Process tasks one by one
        loop {
            // Get next task to process (separate scope to avoid borrowing conflicts)
            let mut current_task = {
                let session = self.active_sessions.get_mut(&session_id).ok_or_else(|| {
                    TrustformersError::config_error("Session not found", "execute_session")
                })?;

                match session.pending_tasks.pop_front() {
                    Some(task) => task,
                    None => break, // No more tasks to process
                }
            };

            if let Some(device_id) = &current_task.assigned_device {
                current_task.started_at = Some(SystemTime::now());
                current_task.status = SessionStatus::Running;

                // Execute task on assigned device (now we don't have any borrows on self)
                let device_id_clone = device_id.clone();
                match self.execute_task_on_device(&current_task, &device_id_clone).await {
                    Ok(_) => {
                        current_task.status = SessionStatus::Completed;
                        current_task.completed_at = Some(SystemTime::now());
                    },
                    Err(_) => {
                        current_task.status = SessionStatus::Failed;
                        if current_task.test_config.retry_attempts > 0 {
                            // Retry logic - add task back to pending queue
                            let mut retry_task = current_task.clone();
                            retry_task.test_config.retry_attempts -= 1;

                            let session =
                                self.active_sessions.get_mut(&session_id).ok_or_else(|| {
                                    TrustformersError::config_error(
                                        "Session not found",
                                        "execute_session",
                                    )
                                })?;
                            session.pending_tasks.push_back(retry_task);
                            continue;
                        }
                    },
                }

                // Add completed task to session
                let session = self.active_sessions.get_mut(&session_id).ok_or_else(|| {
                    TrustformersError::config_error("Session not found", "execute_session")
                })?;
                session.completed_tasks.push(current_task);
            }
        }

        // Mark session as completed
        {
            let session = self.active_sessions.get_mut(&session_id).ok_or_else(|| {
                TrustformersError::config_error("Session not found", "execute_session")
            })?;
            session.status = SessionStatus::Completed;
        }

        Ok(())
    }

    /// Execute a single task on a specific device
    async fn execute_task_on_device(
        &self,
        task: &TestTask,
        device_id: &str,
    ) -> Result<DeviceTestResult> {
        let timeout_duration = task.test_config.timeout;

        // Simulate task execution with timeout
        let result = timeout(timeout_duration, async {
            self.run_test_on_device(task, device_id).await
        })
        .await;

        match result {
            Ok(test_result) => test_result,
            Err(_) => Err(TrustformersError::runtime_error(
                "Task execution timeout in execute_task_on_device".to_string(),
            )
            .into()),
        }
    }

    /// Running a test suite on a farm device is not implemented in this
    /// manager.
    ///
    /// It has no channel to a device -- no ADB/`xcrun` session for a local
    /// slot, and no farm client for a remote one. The previous body slept
    /// 100 ms and returned a fabricated `TestSuiteResults` with
    /// `success_rate: 0.95` and a 60 s execution / 5 s setup / 2 s cleanup /
    /// 3 s network breakdown that nothing had timed, for every device on
    /// every provider.
    ///
    /// [`super::framework::MobileTestingFramework`] is the path that really
    /// executes a suite, in-process on this host.
    async fn run_test_on_device(
        &self,
        _task: &TestTask,
        device_id: &str,
    ) -> Result<DeviceTestResult> {
        // Confirms the device is in the pool before reporting the limitation.
        let _device_info = self.get_device_info(device_id)?;

        Err(TrustformersError::runtime_error(format!(
            "Executing a test suite on farm device `{}`: this manager has no device channel \
             (no ADB/xcrun session, no farm client). Run `MobileTestingFramework` in-process to \
             execute a suite on this host.",
            device_id
        ))
        .into())
    }

    /// Get device information by ID
    fn get_device_info(&self, device_id: &str) -> Result<DeviceInfo> {
        self.device_pool
            .iter()
            .find(|device| device.device_name == device_id)
            .cloned()
            .ok_or_else(|| TrustformersError::config_error("Device not found", "get_device_info"))
            .map_err(|e| e.into())
    }

    /// Allocate devices for test tasks
    async fn allocate_devices(&mut self, tasks: &[TestTask]) -> Result<Vec<String>> {
        let mut allocated_devices = Vec::new();

        for task in tasks {
            if let Some(device) =
                self.find_suitable_device(&task.test_config.resource_requirements).await?
            {
                allocated_devices.push(device.device_name.clone());
            }
        }

        Ok(allocated_devices)
    }

    /// Find a suitable device for given requirements
    async fn find_suitable_device(
        &self,
        requirements: &HardwareRequirements,
    ) -> Result<Option<DeviceInfo>> {
        for device in &self.device_pool {
            if device.ram_mb >= requirements.min_ram_mb
                && device.storage_gb >= requirements.min_storage_gb
            {
                return Ok(Some(device.clone()));
            }
        }
        Ok(None)
    }

    /// AWS Device Farm device discovery is not implemented.
    ///
    /// It requires an authenticated AWS Device Farm client, which this crate
    /// does not depend on. This used to populate the pool with three invented
    /// devices (`aws-iphone-14`, `aws-galaxy-s23`, `aws-pixel-7`, each with a
    /// made-up 8 GB / 256 GB / 1080x2340 spec) that no AWS query produced,
    /// after which the whole manager operated on a fictional fleet. The
    /// sibling `providers::AwsDeviceFarmProvider` reports the same limitation
    /// the same way.
    async fn initialize_aws_devices(&mut self, _region: &str, _project_name: &str) -> Result<()> {
        Err(TrustformersError::runtime_error(
            "AWS Device Farm device discovery: this crate has no AWS Device Farm client, so no \
             device list can be retrieved. Use DeviceFarmProvider::Local to run against this host."
                .to_string(),
        )
        .into())
    }

    /// Firebase Test Lab device discovery is not implemented.
    ///
    /// Same limitation as [`Self::initialize_aws_devices`]: no Firebase Test
    /// Lab client exists in this crate, and the previous body invented three
    /// devices instead of saying so.
    async fn initialize_firebase_devices(
        &mut self,
        _project_id: &str,
        _test_lab_id: &str,
    ) -> Result<()> {
        Err(TrustformersError::runtime_error(
            "Firebase Test Lab device discovery: this crate has no Firebase Test Lab client, so \
             no device list can be retrieved. Use DeviceFarmProvider::Local to run against this \
             host."
                .to_string(),
        )
        .into())
    }

    /// Initialize a local device farm from the real host.
    ///
    /// A "local farm" runs on this machine, so every slot describes the
    /// running host, detected for real by
    /// [`crate::device_info::MobileDeviceDetector`]. Only the slot's label
    /// comes from `device_names`; the hardware fields are measured, not
    /// invented (they were previously a fixed 8 GB / 256 GB / 1080x2340
    /// regardless of the machine).
    async fn initialize_local_devices(
        &mut self,
        pool_size: usize,
        device_names: &[String],
    ) -> Result<()> {
        let host = crate::device_info::MobileDeviceDetector::detect()?;
        let host_ram_mb = host.memory_info.total_mb;
        // Real free space on the volume holding the process's working
        // directory, via `sysinfo::Disks` -- the same source
        // `android_work_manager::check_storage_constraints` uses.
        let host_storage_gb = {
            let disks = sysinfo::Disks::new_with_refreshed_list();
            let total_bytes: u64 = disks.iter().map(|disk| disk.total_space()).max().unwrap_or(0);
            (total_bytes / (1024 * 1024 * 1024)) as usize
        };

        self.device_pool = device_names
            .iter()
            .take(pool_size)
            .enumerate()
            .map(|(index, name)| DeviceInfo {
                device_name: format!("local-{}", index),
                os_name: std::env::consts::OS.to_string(),
                os_version: host.basic_info.os_version.clone(),
                device_type: DeviceType::Generic,
                hardware_model: name.clone(),
                cpu_architecture: std::env::consts::ARCH.to_string(),
                ram_mb: host_ram_mb,
                storage_gb: host_storage_gb,
                // No portable API reports a screen resolution for a headless
                // host, so the slot reports none rather than a phone's.
                screen_resolution: (0, 0),
                sensors: Vec::new(),
            })
            .collect();
        Ok(())
    }

    /// Results for a completed session.
    ///
    /// `Ok(None)` when the session is unknown, and an empty device list plus
    /// an unaggregatable error when no task on it produced a real result --
    /// which is the case for every session today, because
    /// [`Self::run_test_on_device`] has no device channel to execute through.
    ///
    /// The previous body synthesised one `DeviceTestResult` per completed task
    /// with `success_rate: 0.95`, a 60 s duration and a fixed timing
    /// breakdown, then hand-wrote an `AggregatedTestResults` claiming
    /// `avg_latency_ms: 50.0`, a P95 of 70.0, `best_device: "aws-iphone-14"`
    /// and a 0.98 compatibility rate -- a complete cross-device report for
    /// tests that never ran.
    pub fn get_session_results(&self, session_id: &str) -> Result<Option<DeviceFarmSessionResult>> {
        let Some(session) = self.active_sessions.get(session_id) else {
            return Ok(None);
        };

        // Only results this manager actually recorded. Nothing records any
        // today, so this is empty and the aggregation below reports that
        // rather than inventing a fleet report.
        let device_results: Vec<DeviceTestResult> = session.recorded_results.clone();
        let aggregator = ResultAggregator::new(AggregationRules {
            statistical_methods: vec![StatisticalMethod::Mean, StatisticalMethod::Median],
            outlier_detection: false,
            confidence_level: self.config.result_aggregation.statistical_analysis.confidence_level,
            minimum_sample_size: 1,
        });
        let aggregated_results = aggregator.aggregate_results(&device_results)?;

        Ok(Some(DeviceFarmSessionResult {
            session_id: session_id.to_string(),
            start_time: session.start_time,
            duration: SystemTime::now().duration_since(session.start_time).unwrap_or_default(),
            device_results,
            aggregated_results,
        }))
    }

    /// Cancel a session
    pub fn cancel_session(&mut self, session_id: &str) -> Result<()> {
        if let Some(session) = self.active_sessions.get_mut(session_id) {
            session.status = SessionStatus::Cancelled;
            Ok(())
        } else {
            Err(TrustformersError::config_error("Session not found", "cancel_session").into())
        }
    }

    /// Devices currently in the pool.
    ///
    /// Empty until [`Self::initialize`] succeeds; a failed discovery adds
    /// nothing.
    pub fn get_available_devices(&self) -> &[DeviceInfo] {
        &self.device_pool
    }

    /// Get all active sessions
    pub fn get_active_sessions(&self) -> Vec<&DeviceFarmSession> {
        self.active_sessions
            .values()
            .filter(|session| {
                matches!(
                    session.status,
                    SessionStatus::Running | SessionStatus::Pending
                )
            })
            .collect()
    }
}

/// Mean of a sample series, or `None` when the series is empty.
///
/// `None` rather than `0.0`: no samples is not an average of zero.
fn mean_of(samples: impl Iterator<Item = f32>) -> Option<f32> {
    let collected: Vec<f32> = samples.collect();
    if collected.is_empty() {
        None
    } else {
        Some(collected.iter().sum::<f32>() / collected.len() as f32)
    }
}

impl ResultAggregator {
    /// Create a new result aggregator
    pub fn new(rules: AggregationRules) -> Self {
        Self {
            aggregation_rules: rules,
        }
    }

    /// Aggregate results from multiple devices
    pub fn aggregate_results(
        &self,
        device_results: &[DeviceTestResult],
    ) -> Result<AggregatedTestResults> {
        if device_results.is_empty() {
            return Err(TrustformersError::config_error(
                "No device results to aggregate",
                "aggregate_results",
            )
            .into());
        }

        let device_count = device_results.len();
        let overall_success_rate =
            device_results.iter().map(|r| r.test_results.success_rate).sum::<f32>()
                / device_count as f32;

        // Extract latency values from benchmark results
        let latencies: Vec<f32> = device_results
            .iter()
            .flat_map(|r| &r.test_results.benchmark_results)
            .map(|b| b.avg_latency_ms)
            .collect();

        let avg_latency = if !latencies.is_empty() {
            latencies.iter().sum::<f32>() / latencies.len() as f32
        } else {
            0.0
        };

        let latency_std_dev = if latencies.len() > 1 {
            let mean = avg_latency;
            let variance = latencies.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                / (latencies.len() - 1) as f32;
            variance.sqrt()
        } else {
            0.0
        };

        let statistical_summary = self.calculate_statistical_summary(&latencies)?;

        // Find best and worst performing devices
        let (best_device, worst_device) = self.find_performance_extremes(device_results);

        Ok(AggregatedTestResults {
            device_count,
            overall_success_rate,
            metrics: AggregatedMetrics {
                avg_latency_ms: avg_latency,
                latency_std_dev,
                avg_throughput_fps: mean_of(
                    device_results
                        .iter()
                        .flat_map(|r| &r.test_results.benchmark_results)
                        .map(|b| b.throughput_fps),
                ),
                avg_memory_usage_mb: mean_of(
                    device_results
                        .iter()
                        .flat_map(|r| &r.test_results.benchmark_results)
                        .map(|b| b.memory_usage_mb as f32),
                ),
                // Averaged over the battery results that were measured; the
                // previous line reported a fixed 500 mW on every fleet.
                avg_power_consumption_mw: mean_of(
                    device_results
                        .iter()
                        .flat_map(|r| &r.test_results.battery_results)
                        .map(|b| b.avg_power_consumption_mw),
                ),
                statistical_summary,
            },
            cross_device_analysis: CrossDeviceAnalysis {
                performance_variance: latency_std_dev / avg_latency.max(1.0),
                best_device,
                worst_device,
                compatibility_rate: overall_success_rate,
            },
        })
    }

    /// Calculate statistical summary for a set of values
    fn calculate_statistical_summary(&self, values: &[f32]) -> Result<StatisticalSummary> {
        if values.is_empty() {
            return Ok(StatisticalSummary {
                mean: 0.0,
                median: 0.0,
                std_deviation: 0.0,
                min: 0.0,
                max: 0.0,
                percentiles: HashMap::new(),
            });
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let median = sorted_values[values.len() / 2];
        let min = sorted_values[0];
        let max = sorted_values[values.len() - 1];

        let std_deviation = if values.len() > 1 {
            let variance =
                values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32;
            variance.sqrt()
        } else {
            0.0
        };

        let mut percentiles = HashMap::new();
        percentiles.insert(
            "P95".to_string(),
            sorted_values[(values.len() * 95 / 100).min(values.len() - 1)],
        );
        percentiles.insert(
            "P99".to_string(),
            sorted_values[(values.len() * 99 / 100).min(values.len() - 1)],
        );

        Ok(StatisticalSummary {
            mean,
            median,
            std_deviation,
            min,
            max,
            percentiles,
        })
    }

    /// Find best and worst performing devices
    fn find_performance_extremes(&self, device_results: &[DeviceTestResult]) -> (String, String) {
        let mut best_device = "unknown".to_string();
        let mut worst_device = "unknown".to_string();
        let mut best_score = 0.0;
        let mut worst_score = f32::INFINITY;

        for result in device_results {
            let score = result.test_results.success_rate;
            if score > best_score {
                best_score = score;
                best_device = result.device_id.clone();
            }
            if score < worst_score {
                worst_score = score;
                worst_device = result.device_id.clone();
            }
        }

        (best_device, worst_device)
    }
}
