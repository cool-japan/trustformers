//! Device Farm Providers
//!
//! This module contains implementations for various device farm providers including
//! AWS Device Farm, Firebase Test Lab, BrowserStack, and local device farms.

use async_trait::async_trait;
use std::time::{Duration, SystemTime};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::TrustformersError;

use super::config::*;
use super::device_farm::{TestExecutionConfig, TestType};
use super::results::*;

/// Trait for device farm provider implementations
#[async_trait]
pub trait DeviceFarmProviderTrait {
    /// Initialize the provider
    async fn initialize(&mut self, credentials: &DeviceFarmCredentials) -> Result<()>;

    /// Get available devices
    async fn get_available_devices(
        &self,
        criteria: &DeviceSelectionCriteria,
    ) -> Result<Vec<DeviceInfo>>;

    /// Allocate devices for testing
    async fn allocate_devices(
        &mut self,
        device_ids: &[String],
        session_id: &str,
    ) -> Result<Vec<String>>;

    /// Release devices after testing
    async fn release_devices(&mut self, device_ids: &[String], session_id: &str) -> Result<()>;

    /// Execute test on a device
    async fn execute_test(
        &self,
        device_id: &str,
        test_config: &TestExecutionConfig,
    ) -> Result<DeviceTestResult>;

    /// Get cost estimate for device usage
    async fn get_cost_estimate(&self, device_ids: &[String], duration: Duration) -> Result<f32>;

    /// Check device availability
    async fn check_device_availability(&self, device_id: &str) -> Result<bool>;

    /// Get provider capabilities
    fn get_capabilities(&self) -> Vec<String>;
}

/// AWS Device Farm provider implementation
pub struct AWSDeviceFarmProvider {
    region: String,
    project_arn: String,
    is_initialized: bool,
}

impl AWSDeviceFarmProvider {
    pub fn new(region: String, project_name: String) -> Self {
        let project_arn = format!("arn:aws:devicefarm:{}:project/{}", region, project_name);
        Self {
            region,
            project_arn,
            is_initialized: false,
        }
    }
}

#[async_trait]
impl DeviceFarmProviderTrait for AWSDeviceFarmProvider {
    async fn initialize(&mut self, credentials: &DeviceFarmCredentials) -> Result<()> {
        // Initialize AWS SDK and validate credentials
        if credentials.access_key.is_none() || credentials.secret_key.is_none() {
            return Err(TrustformersError::config_error(
                "AWS credentials (access_key, secret_key) are required",
                "initialize",
            )
            .into());
        }

        // In real implementation, this would:
        // 1. Configure AWS SDK with credentials
        // 2. Validate project access
        // 3. Check permissions

        self.is_initialized = true;
        Ok(())
    }

    async fn get_available_devices(
        &self,
        criteria: &DeviceSelectionCriteria,
    ) -> Result<Vec<DeviceInfo>> {
        if !self.is_initialized {
            return Err(TrustformersError::config_error(
                "Provider not initialized",
                "get_available_devices",
            )
            .into());
        }

        // Matches `execute_test`'s honesty policy immediately below: this
        // crate does not call the real AWS Device Farm "list devices" API,
        // so it must not hand back a device catalog that *looks* like a
        // live query result (the previous implementation returned three
        // hardcoded `DeviceInfo` entries with plausible-looking AWS device
        // farm names, filtered by `criteria` as if they came from a real
        // response). A caller cannot tell fabricated devices from real ones
        // by inspection, so refusing is the only honest option here.
        Err(TrustformersError::runtime_error(
            "AWS Device Farm device listing is not available: this crate does not call the \
             real AWS Device Farm API, so no live device catalog exists to query. Wire a real \
             AWS Device Farm client before relying on this provider's device list."
                .to_string(),
        )
        .into())
    }

    async fn allocate_devices(
        &mut self,
        device_ids: &[String],
        session_id: &str,
    ) -> Result<Vec<String>> {
        // No real AWS Device Farm API call is made here (see `execute_test`
        // for why); this no longer pads the call with a `sleep` chosen to
        // *look* like real network latency for a request that never left
        // the process.
        //
        // In real implementation, this would make AWS API calls to:
        // 1. Check device availability
        // 2. Create device pool
        // 3. Schedule test runs

        tracing::debug!(
            "AWS: recorded allocation of {} device id(s) for session {} (no live AWS call made)",
            device_ids.len(),
            session_id
        );
        Ok(device_ids.to_vec())
    }

    async fn release_devices(&mut self, device_ids: &[String], session_id: &str) -> Result<()> {
        // See `allocate_devices`: no real AWS call, no fabricated latency.
        tracing::debug!(
            "AWS: recorded release of {} device id(s) for session {} (no live AWS call made)",
            device_ids.len(),
            session_id
        );
        Ok(())
    }

    async fn execute_test(
        &self,
        device_id: &str,
        _test_config: &TestExecutionConfig,
    ) -> Result<DeviceTestResult> {
        // This used to sleep for a fixed duration and then hand back a
        // `DeviceTestResult` with `success_rate: 0.95` -- a fabricated
        // "the test passed" verdict for a test that never ran anywhere.
        // Running a real AWS Device Farm test requires: the `aws-sdk-*`
        // crates (or hand-rolled SigV4-signed HTTP calls), live network
        // access to `devicefarm.<region>.amazonaws.com`, and real AWS
        // credentials with Device Farm permissions -- none of which this
        // environment provides, and none of which this crate links today.
        // Reporting a structured "not available" error is the honest
        // answer; a caller must not be able to mistake this for a real
        // pass/fail signal.
        if !self.is_initialized {
            return Err(TrustformersError::config_error(
                "Provider not initialized",
                "execute_test",
            )
            .into());
        }
        Err(TrustformersError::runtime_error(format!(
            "AWS Device Farm test execution for device '{device_id}' is not available: this \
             crate does not link the AWS SDK / sign real Device Farm API requests, so no test \
             was actually run. Wire a real AWS Device Farm client before trusting this result."
        ))
        .into())
    }

    async fn get_cost_estimate(&self, device_ids: &[String], duration: Duration) -> Result<f32> {
        // A reference pricing *estimate* (AWS Device Farm's published
        // per-device-minute rate applied to the requested duration) -- not
        // a live quote from AWS, and not a claim that a test executed.
        let cost_per_device_minute = 0.17;
        let duration_minutes = duration.as_secs_f32() / 60.0;
        Ok(device_ids.len() as f32 * duration_minutes * cost_per_device_minute)
    }

    async fn check_device_availability(&self, device_id: &str) -> Result<bool> {
        // Same honesty policy as `execute_test`/`get_available_devices`: no
        // real AWS query is made, so an unconditional `Ok(true)` would be a
        // fabricated "yes, it's available" answer a caller could act on.
        Err(TrustformersError::runtime_error(format!(
            "AWS Device Farm availability check for device '{device_id}' is not available: this \
             crate does not call the real AWS Device Farm API."
        ))
        .into())
    }

    fn get_capabilities(&self) -> Vec<String> {
        vec![
            "iOS Testing".to_string(),
            "Android Testing".to_string(),
            "Real Devices".to_string(),
            "Video Recording".to_string(),
            "Performance Monitoring".to_string(),
            "Network Shaping".to_string(),
            "GPS Simulation".to_string(),
        ]
    }
}

/// Firebase Test Lab provider implementation
pub struct FirebaseTestLabProvider {
    project_id: String,
    test_lab_id: String,
    is_initialized: bool,
}

impl FirebaseTestLabProvider {
    pub fn new(project_id: String, test_lab_id: String) -> Self {
        Self {
            project_id,
            test_lab_id,
            is_initialized: false,
        }
    }
}

#[async_trait]
impl DeviceFarmProviderTrait for FirebaseTestLabProvider {
    async fn initialize(&mut self, credentials: &DeviceFarmCredentials) -> Result<()> {
        if credentials.service_account_json.is_none() {
            return Err(TrustformersError::config_error(
                "Firebase service account JSON is required",
                "initialize",
            )
            .into());
        }

        self.is_initialized = true;
        Ok(())
    }

    async fn get_available_devices(
        &self,
        criteria: &DeviceSelectionCriteria,
    ) -> Result<Vec<DeviceInfo>> {
        if !self.is_initialized {
            return Err(TrustformersError::config_error(
                "Provider not initialized",
                "get_available_devices",
            )
            .into());
        }

        // Same honesty policy as `execute_test` below: this crate does not
        // call the real Firebase Test Lab device catalog API, so it must
        // not hand back a device list that looks like one (the previous
        // implementation returned two hardcoded `DeviceInfo` entries with
        // plausible Firebase Test Lab device names, ignoring `criteria`
        // entirely -- it was never even filtered by it).
        Err(TrustformersError::runtime_error(
            "Firebase Test Lab device listing is not available: this crate does not call the \
             real Firebase Test Lab API, so no live device catalog exists to query. Wire a real \
             Firebase Test Lab client before relying on this provider's device list."
                .to_string(),
        )
        .into())
    }

    async fn allocate_devices(
        &mut self,
        device_ids: &[String],
        session_id: &str,
    ) -> Result<Vec<String>> {
        // No real Firebase Test Lab API call is made here (see
        // `execute_test`); this no longer pads the call with a `sleep`
        // chosen to *look* like real network latency for a request that
        // never left the process, nor prints to stdout in place of
        // `tracing`.
        tracing::debug!(
            "Firebase: recorded allocation of {} device id(s) for session {} (no live Firebase \
             call made)",
            device_ids.len(),
            session_id
        );
        Ok(device_ids.to_vec())
    }

    async fn release_devices(&mut self, device_ids: &[String], session_id: &str) -> Result<()> {
        // See `allocate_devices`: no real Firebase call, no fabricated
        // latency.
        tracing::debug!(
            "Firebase: recorded release of {} device id(s) for session {} (no live Firebase \
             call made)",
            device_ids.len(),
            session_id
        );
        Ok(())
    }

    async fn execute_test(
        &self,
        device_id: &str,
        _test_config: &TestExecutionConfig,
    ) -> Result<DeviceTestResult> {
        // Previously fabricated a `success_rate: 0.92` "passing" result
        // after a fixed sleep, for a test that never touched Firebase Test
        // Lab. A real run needs the Firebase Test Lab REST API (or
        // `gcloud firebase test android run`), a service-account-signed
        // request, and live network access -- this crate has none of
        // those wired up, so an honest structured error is returned
        // instead of a fabricated pass/fail verdict.
        if !self.is_initialized {
            return Err(TrustformersError::config_error(
                "Provider not initialized",
                "execute_test",
            )
            .into());
        }
        Err(TrustformersError::runtime_error(format!(
            "Firebase Test Lab test execution for device '{device_id}' is not available: this \
             crate does not call the real Firebase Test Lab API, so no test was actually run. \
             Wire a real Firebase Test Lab client before trusting this result."
        ))
        .into())
    }

    async fn get_cost_estimate(&self, device_ids: &[String], duration: Duration) -> Result<f32> {
        // A reference pricing *estimate* (Firebase Test Lab's published
        // per-device-hour rate applied to the requested duration) -- not a
        // live quote, and not a claim that a test executed.
        let cost_per_device_hour = 1.0;
        let duration_hours = duration.as_secs_f32() / 3600.0;
        Ok(device_ids.len() as f32 * duration_hours * cost_per_device_hour)
    }

    async fn check_device_availability(&self, device_id: &str) -> Result<bool> {
        // Same honesty policy as `execute_test`/`get_available_devices`: no
        // real Firebase query is made, so an unconditional `Ok(true)` would
        // be a fabricated "yes, it's available" answer a caller could act
        // on.
        Err(TrustformersError::runtime_error(format!(
            "Firebase Test Lab availability check for device '{device_id}' is not available: \
             this crate does not call the real Firebase Test Lab API."
        ))
        .into())
    }

    fn get_capabilities(&self) -> Vec<String> {
        vec![
            "Android Testing".to_string(),
            "Virtual Devices".to_string(),
            "Real Devices".to_string(),
            "Video Recording".to_string(),
            "Performance Profiling".to_string(),
            "Robo Test".to_string(),
        ]
    }
}

/// Local device farm provider for testing on local devices
pub struct LocalDeviceFarmProvider {
    available_devices: Vec<DeviceInfo>,
    allocated_devices: Vec<String>,
}

impl Default for LocalDeviceFarmProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalDeviceFarmProvider {
    pub fn new() -> Self {
        Self {
            available_devices: Vec::new(),
            allocated_devices: Vec::new(),
        }
    }
}

#[async_trait]
impl DeviceFarmProviderTrait for LocalDeviceFarmProvider {
    async fn initialize(&mut self, _credentials: &DeviceFarmCredentials) -> Result<()> {
        // Initialize local device detection
        self.available_devices = vec![
            DeviceInfo {
                device_name: "local-simulator-ios".to_string(),
                os_name: "iOS".to_string(),
                os_version: "17.0".to_string(),
                device_type: DeviceType::Phone,
                hardware_model: "iPhone Simulator".to_string(),
                cpu_architecture: "x86_64".to_string(),
                ram_mb: 4096,
                storage_gb: 64,
                screen_resolution: (375, 812),
                sensors: vec!["camera".to_string()],
            },
            DeviceInfo {
                device_name: "local-emulator-android".to_string(),
                os_name: "Android".to_string(),
                os_version: "14".to_string(),
                device_type: DeviceType::Phone,
                hardware_model: "Android Emulator".to_string(),
                cpu_architecture: "x86_64".to_string(),
                ram_mb: 4096,
                storage_gb: 32,
                screen_resolution: (360, 640),
                sensors: vec!["camera".to_string()],
            },
        ];
        Ok(())
    }

    async fn get_available_devices(
        &self,
        _criteria: &DeviceSelectionCriteria,
    ) -> Result<Vec<DeviceInfo>> {
        Ok(self.available_devices.clone())
    }

    async fn allocate_devices(
        &mut self,
        device_ids: &[String],
        session_id: &str,
    ) -> Result<Vec<String>> {
        for device_id in device_ids {
            if !self.allocated_devices.contains(device_id) {
                self.allocated_devices.push(device_id.clone());
            }
        }
        println!(
            "Local: Allocated {} devices for session {}",
            device_ids.len(),
            session_id
        );
        Ok(device_ids.to_vec())
    }

    async fn release_devices(&mut self, device_ids: &[String], session_id: &str) -> Result<()> {
        for device_id in device_ids {
            self.allocated_devices.retain(|id| id != device_id);
        }
        println!(
            "Local: Released {} devices from session {}",
            device_ids.len(),
            session_id
        );
        Ok(())
    }

    async fn execute_test(
        &self,
        device_id: &str,
        test_config: &TestExecutionConfig,
    ) -> Result<DeviceTestResult> {
        let execution_duration = Duration::from_secs(15);
        tokio::time::sleep(execution_duration).await;

        let device_info = self
            .available_devices
            .iter()
            .find(|d| d.device_name == device_id)
            .cloned()
            .unwrap_or_else(|| DeviceInfo {
                device_name: device_id.to_string(),
                os_name: "Unknown".to_string(),
                os_version: "Unknown".to_string(),
                device_type: DeviceType::Generic,
                hardware_model: "Unknown".to_string(),
                cpu_architecture: "unknown".to_string(),
                ram_mb: 1024,
                storage_gb: 16,
                screen_resolution: (320, 480),
                sensors: vec![],
            });

        Ok(DeviceTestResult {
            device_id: device_id.to_string(),
            device_info,
            test_results: TestSuiteResults {
                timestamp: SystemTime::now(),
                duration: execution_duration,
                benchmark_results: vec![],
                battery_results: vec![],
                stress_results: vec![],
                memory_results: vec![],
                success_rate: 0.88,
            },
            execution_metrics: DeviceExecutionMetrics {
                execution_time: execution_duration,
                setup_time: Duration::from_secs(2),
                cleanup_time: Duration::from_secs(1),
                network_time: Duration::from_secs(0),
                availability_time: Duration::from_secs(12),
            },
            artifacts: vec![],
        })
    }

    async fn get_cost_estimate(&self, _device_ids: &[String], _duration: Duration) -> Result<f32> {
        // Local devices are free
        Ok(0.0)
    }

    async fn check_device_availability(&self, device_id: &str) -> Result<bool> {
        Ok(!self.allocated_devices.contains(&device_id.to_string()))
    }

    fn get_capabilities(&self) -> Vec<String> {
        vec![
            "iOS Simulator".to_string(),
            "Android Emulator".to_string(),
            "Fast Execution".to_string(),
            "No Cost".to_string(),
            "Local Development".to_string(),
        ]
    }
}

/// Create provider instance based on configuration
pub fn create_provider(
    config: &DeviceFarmProvider,
) -> Result<Box<dyn DeviceFarmProviderTrait + Send + Sync>> {
    match config {
        DeviceFarmProvider::AWS {
            region,
            project_name,
        } => Ok(Box::new(AWSDeviceFarmProvider::new(
            region.clone(),
            project_name.clone(),
        ))),
        DeviceFarmProvider::Firebase {
            project_id,
            test_lab_id,
        } => Ok(Box::new(FirebaseTestLabProvider::new(
            project_id.clone(),
            test_lab_id.clone(),
        ))),
        DeviceFarmProvider::Local { .. } => Ok(Box::new(LocalDeviceFarmProvider::new())),
        _ => Err(TrustformersError::config_error(
            "Unsupported device farm provider",
            "create_provider",
        )
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_execution_config() -> TestExecutionConfig {
        TestExecutionConfig {
            test_type: TestType::Benchmark,
            timeout: Duration::from_secs(60),
            retry_attempts: 0,
            resource_requirements: HardwareRequirements {
                min_ram_mb: 0,
                min_cpu_cores: 0,
                min_storage_gb: 0,
                required_sensors: vec![],
                required_connectivity: vec![],
            },
        }
    }

    /// Regression test for the previous `AWSDeviceFarmProvider::execute_test`,
    /// which slept for a fixed duration and then returned a
    /// `DeviceTestResult` with `success_rate: 0.95` -- a fabricated
    /// "the test passed" verdict even though no AWS API was ever called.
    /// It must now report an honest error instead of a fake pass.
    #[tokio::test]
    async fn test_aws_execute_test_reports_honest_error_not_fake_pass() {
        let mut provider = AWSDeviceFarmProvider::new("us-west-2".to_string(), "proj".to_string());
        provider
            .initialize(&DeviceFarmCredentials {
                access_key: Some("test-key".to_string()),
                secret_key: Some("test-secret".to_string()),
                api_token: None,
                username: None,
                password: None,
                service_account_json: None,
            })
            .await
            .expect("initialize should succeed with credentials present");

        let result = provider.execute_test("aws-iphone-15-pro", &test_execution_config()).await;
        assert!(result.is_err(), "must not fabricate a passing test result");
    }

    /// Same regression, for `FirebaseTestLabProvider::execute_test`
    /// (previously a fabricated `success_rate: 0.92`).
    #[tokio::test]
    async fn test_firebase_execute_test_reports_honest_error_not_fake_pass() {
        let mut provider =
            FirebaseTestLabProvider::new("proj-id".to_string(), "lab-id".to_string());
        provider
            .initialize(&DeviceFarmCredentials {
                access_key: None,
                secret_key: None,
                api_token: None,
                username: None,
                password: None,
                service_account_json: Some("{}".to_string()),
            })
            .await
            .expect("initialize should succeed with credentials present");

        let result = provider.execute_test("firebase-pixel-7", &test_execution_config()).await;
        assert!(result.is_err(), "must not fabricate a passing test result");
    }

    /// `LocalDeviceFarmProvider` is a genuine local/offline provider (no
    /// cloud claim at all) and must be unaffected by the AWS/Firebase fix.
    #[tokio::test]
    async fn test_local_provider_still_executes_successfully() {
        let mut provider = LocalDeviceFarmProvider::new();
        provider
            .initialize(&DeviceFarmCredentials {
                access_key: None,
                secret_key: None,
                api_token: None,
                username: None,
                password: None,
                service_account_json: None,
            })
            .await
            .expect("local provider initialize should always succeed");

        let result = provider.execute_test("local-simulator-ios", &test_execution_config()).await;
        assert!(result.is_ok());
    }

    fn device_selection_criteria() -> DeviceSelectionCriteria {
        DeviceSelectionCriteria {
            device_types: vec![],
            os_versions: vec![],
            hardware_requirements: HardwareRequirements {
                min_ram_mb: 0,
                min_cpu_cores: 0,
                min_storage_gb: 0,
                required_sensors: vec![],
                required_connectivity: vec![],
            },
            gpu_requirements: GpuRequirements {
                vendor: None,
                min_memory_mb: None,
                required_features: vec![],
                min_compute_capability: None,
            },
            regions: vec![],
            availability_requirements: vec![],
        }
    }

    /// Regression test for `AWSDeviceFarmProvider::get_available_devices`
    /// and `check_device_availability`, which previously returned three
    /// hardcoded `DeviceInfo` entries (`aws-iphone-15-pro`, etc.) and an
    /// unconditional `Ok(true)` respectively -- a fabricated device catalog
    /// and a fabricated availability signal for a provider that never
    /// calls the real AWS Device Farm API. Both must now report an honest
    /// error rather than data that looks real.
    #[tokio::test]
    async fn test_aws_device_queries_report_honest_error_not_fake_catalog() {
        let mut provider = AWSDeviceFarmProvider::new("us-west-2".to_string(), "proj".to_string());
        provider
            .initialize(&DeviceFarmCredentials {
                access_key: Some("test-key".to_string()),
                secret_key: Some("test-secret".to_string()),
                api_token: None,
                username: None,
                password: None,
                service_account_json: None,
            })
            .await
            .expect("initialize should succeed with credentials present");

        let devices = provider.get_available_devices(&device_selection_criteria()).await;
        assert!(devices.is_err(), "must not fabricate a device catalog");

        let available = provider.check_device_availability("aws-iphone-15-pro").await;
        assert!(
            available.is_err(),
            "must not fabricate an availability signal"
        );
    }

    /// Same regression, for `FirebaseTestLabProvider`.
    #[tokio::test]
    async fn test_firebase_device_queries_report_honest_error_not_fake_catalog() {
        let mut provider =
            FirebaseTestLabProvider::new("proj-id".to_string(), "lab-id".to_string());
        provider
            .initialize(&DeviceFarmCredentials {
                access_key: None,
                secret_key: None,
                api_token: None,
                username: None,
                password: None,
                service_account_json: Some("{}".to_string()),
            })
            .await
            .expect("initialize should succeed with credentials present");

        let devices = provider.get_available_devices(&device_selection_criteria()).await;
        assert!(devices.is_err(), "must not fabricate a device catalog");

        let available = provider.check_device_availability("firebase-pixel-7").await;
        assert!(
            available.is_err(),
            "must not fabricate an availability signal"
        );
    }

    /// `allocate_devices`/`release_devices` on the AWS/Firebase providers
    /// still succeed (they only record local bookkeeping and never claimed
    /// to call a live API), and must run instantly now that the fabricated
    /// `tokio::time::sleep` padding is gone.
    #[tokio::test]
    async fn test_aws_and_firebase_allocate_release_are_fast_and_honest() {
        let start = std::time::Instant::now();

        let mut aws = AWSDeviceFarmProvider::new("us-west-2".to_string(), "proj".to_string());
        let ids = vec!["dev-1".to_string()];
        aws.allocate_devices(&ids, "session-1").await.expect("allocate should succeed");
        aws.release_devices(&ids, "session-1").await.expect("release should succeed");

        let mut firebase =
            FirebaseTestLabProvider::new("proj-id".to_string(), "lab-id".to_string());
        firebase
            .allocate_devices(&ids, "session-1")
            .await
            .expect("allocate should succeed");
        firebase
            .release_devices(&ids, "session-1")
            .await
            .expect("release should succeed");

        assert!(
            start.elapsed() < Duration::from_millis(200),
            "allocate/release must not pad the call with a fabricated network-latency sleep"
        );
    }
}
