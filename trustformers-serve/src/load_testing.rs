//! Load Testing Suite
//!
//! Comprehensive load testing framework for TrustformeRS Serve API with support
//! for various test patterns, metrics collection, and performance analysis.

use anyhow::Result;
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{
    sync::{Mutex, RwLock},
    time::{interval, sleep},
};

/// Load testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestConfig {
    /// Base URL of the server to test
    pub base_url: String,

    /// Test duration in seconds
    pub duration_seconds: u64,

    /// Number of concurrent users
    pub concurrent_users: usize,

    /// Requests per second (rate limiting)
    pub requests_per_second: Option<f64>,

    /// Test scenarios to run
    pub scenarios: Vec<TestScenario>,

    /// Authentication configuration
    pub auth: Option<AuthConfig>,

    /// Timeout configuration
    pub timeout_config: TimeoutConfig,

    /// Output configuration
    pub output_config: OutputConfig,

    /// Advanced configuration
    pub advanced_config: AdvancedConfig,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8080".to_string(),
            duration_seconds: 60,
            concurrent_users: 10,
            requests_per_second: None,
            scenarios: vec![TestScenario::default()],
            auth: None,
            timeout_config: TimeoutConfig::default(),
            output_config: OutputConfig::default(),
            advanced_config: AdvancedConfig::default(),
        }
    }
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Authentication type
    pub auth_type: AuthType,

    /// API key for API key authentication
    pub api_key: Option<String>,

    /// JWT token for JWT authentication
    pub jwt_token: Option<String>,

    /// OAuth2 configuration
    pub oauth2: Option<OAuth2Config>,
}

/// Authentication type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthType {
    None,
    ApiKey,
    Bearer,
    OAuth2,
}

/// OAuth2 configuration for load testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2Config {
    /// Client ID
    pub client_id: String,

    /// Client secret
    pub client_secret: String,

    /// Token endpoint
    pub token_endpoint: String,

    /// Scope
    pub scope: Option<String>,
}

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Request timeout in milliseconds
    pub request_timeout_ms: u64,

    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,

    /// Read timeout in milliseconds
    pub read_timeout_ms: u64,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            request_timeout_ms: 30000,
            connection_timeout_ms: 10000,
            read_timeout_ms: 30000,
        }
    }
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Enable real-time console output
    pub enable_realtime_output: bool,

    /// Output format
    pub format: OutputFormat,

    /// Output file path
    pub output_file: Option<String>,

    /// Enable detailed per-request logging
    pub enable_detailed_logging: bool,

    /// Percentiles to calculate
    pub percentiles: Vec<f64>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            enable_realtime_output: true,
            format: OutputFormat::Console,
            output_file: None,
            enable_detailed_logging: false,
            percentiles: vec![50.0, 75.0, 90.0, 95.0, 99.0, 99.9],
        }
    }
}

/// Output format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Console,
    Json,
    Csv,
    Html,
}

/// Advanced configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedConfig {
    /// Enable HTTP/2
    pub enable_http2: bool,

    /// Enable connection pooling
    pub enable_connection_pooling: bool,

    /// Maximum idle connections per host
    pub max_idle_per_host: usize,

    /// Keep alive timeout in seconds
    pub keep_alive_timeout_seconds: u64,

    /// Enable gzip compression
    pub enable_compression: bool,

    /// Custom headers to include in all requests
    pub custom_headers: HashMap<String, String>,

    /// Think time between requests (milliseconds)
    pub think_time_ms: Option<u64>,

    /// Ramp-up time in seconds
    pub ramp_up_seconds: u64,

    /// Ramp-down time in seconds
    pub ramp_down_seconds: u64,
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            enable_http2: true,
            enable_connection_pooling: true,
            max_idle_per_host: 32,
            keep_alive_timeout_seconds: 90,
            enable_compression: true,
            custom_headers: HashMap::new(),
            think_time_ms: None,
            ramp_up_seconds: 10,
            ramp_down_seconds: 10,
        }
    }
}

/// Test scenario configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestScenario {
    /// Scenario name
    pub name: String,

    /// Scenario weight (percentage of traffic)
    pub weight: f64,

    /// HTTP method
    pub method: HttpMethod,

    /// Endpoint path
    pub path: String,

    /// Request body template
    pub body_template: Option<String>,

    /// Query parameters
    pub query_params: HashMap<String, String>,

    /// Headers specific to this scenario
    pub headers: HashMap<String, String>,

    /// Expected response status codes
    pub expected_status_codes: Vec<u16>,

    /// Validation rules
    pub validation_rules: Vec<ValidationRule>,
}

impl Default for TestScenario {
    fn default() -> Self {
        Self {
            name: "health_check".to_string(),
            weight: 100.0,
            method: HttpMethod::Get,
            path: "/health".to_string(),
            body_template: None,
            query_params: HashMap::new(),
            headers: HashMap::new(),
            expected_status_codes: vec![200],
            validation_rules: vec![],
        }
    }
}

/// HTTP methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
    Options,
}

/// Validation rule for response validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule type
    pub rule_type: ValidationRuleType,

    /// JSON path to validate (for JSON responses)
    pub json_path: Option<String>,

    /// Expected value
    pub expected_value: Option<String>,

    /// Validation pattern (regex)
    pub pattern: Option<String>,

    /// Response header to validate
    pub header_name: Option<String>,
}

/// Validation rule types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    /// Response contains specific text
    ResponseContains,

    /// Response matches regex pattern
    ResponseMatches,

    /// JSON field equals value
    JsonFieldEquals,

    /// JSON field exists
    JsonFieldExists,

    /// Header exists
    HeaderExists,

    /// Header equals value
    HeaderEquals,

    /// Response size bounds
    ResponseSizeBounds { min: usize, max: usize },
}

/// Load test results
#[derive(Debug, Clone, Serialize)]
pub struct LoadTestResults {
    /// Test configuration
    pub config: LoadTestConfig,

    /// Test summary
    pub summary: TestSummary,

    /// Per-scenario results
    pub scenario_results: HashMap<String, ScenarioResults>,

    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,

    /// Error analysis
    pub error_analysis: ErrorAnalysis,

    /// Time series data for charts
    pub time_series: TimeSeriesData,
}

/// Test summary
#[derive(Debug, Clone, Serialize)]
pub struct TestSummary {
    /// Test start time
    pub start_time: chrono::DateTime<chrono::Utc>,

    /// Test end time
    pub end_time: chrono::DateTime<chrono::Utc>,

    /// Total test duration
    pub duration_seconds: f64,

    /// Total requests sent
    pub total_requests: u64,

    /// Total successful requests
    pub successful_requests: u64,

    /// Total failed requests
    pub failed_requests: u64,

    /// Success rate percentage
    pub success_rate: f64,

    /// Average requests per second
    pub average_rps: f64,

    /// Peak requests per second
    pub peak_rps: f64,

    /// Total data transferred (bytes)
    pub total_bytes_transferred: u64,

    /// Average response time (milliseconds)
    pub average_response_time_ms: f64,

    /// Median response time (milliseconds)
    pub median_response_time_ms: f64,
}

/// Per-scenario results
#[derive(Debug, Clone, Serialize)]
pub struct ScenarioResults {
    /// Scenario name
    pub scenario_name: String,

    /// Number of requests
    pub request_count: u64,

    /// Success count
    pub success_count: u64,

    /// Error count
    pub error_count: u64,

    /// Response time statistics
    pub response_time_stats: ResponseTimeStats,

    /// Throughput statistics
    pub throughput_stats: ThroughputStats,

    /// Error breakdown
    pub error_breakdown: HashMap<String, u64>,
}

/// Response time statistics
#[derive(Debug, Clone, Serialize)]
pub struct ResponseTimeStats {
    /// Mean response time
    pub mean_ms: f64,

    /// Median response time
    pub median_ms: f64,

    /// Minimum response time
    pub min_ms: f64,

    /// Maximum response time
    pub max_ms: f64,

    /// Standard deviation
    pub std_dev_ms: f64,

    /// Percentiles
    pub percentiles: HashMap<String, f64>,
}

/// Throughput statistics
#[derive(Debug, Clone, Serialize)]
pub struct ThroughputStats {
    /// Requests per second
    pub rps: f64,

    /// Data transfer rate (MB/s)
    pub mbps: f64,

    /// Peak RPS achieved
    pub peak_rps: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceMetrics {
    /// CPU usage during test
    pub cpu_usage_percent: Option<f64>,

    /// Memory usage during test
    pub memory_usage_mb: Option<f64>,

    /// Network utilization
    pub network_utilization: Option<NetworkUtilization>,

    /// Connection pool statistics
    pub connection_pool_stats: Option<ConnectionPoolStats>,
}

/// Network utilization
#[derive(Debug, Clone, Serialize)]
pub struct NetworkUtilization {
    /// Bytes sent
    pub bytes_sent: u64,

    /// Bytes received
    pub bytes_received: u64,

    /// Packets sent
    pub packets_sent: u64,

    /// Packets received
    pub packets_received: u64,
}

/// Connection pool statistics
#[derive(Debug, Clone, Serialize)]
pub struct ConnectionPoolStats {
    /// Active connections
    pub active_connections: u64,

    /// Idle connections
    pub idle_connections: u64,

    /// Connection reuse rate
    pub connection_reuse_rate: f64,
}

/// Error analysis
#[derive(Debug, Clone, Serialize)]
pub struct ErrorAnalysis {
    /// Total errors
    pub total_errors: u64,

    /// Error rate percentage
    pub error_rate: f64,

    /// Error breakdown by type
    pub error_types: HashMap<String, u64>,

    /// Error breakdown by status code
    pub status_code_breakdown: HashMap<u16, u64>,

    /// Most common errors
    pub top_errors: Vec<ErrorDetail>,
}

/// Error detail
#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    /// Error message
    pub message: String,

    /// Error count
    pub count: u64,

    /// Error percentage
    pub percentage: f64,

    /// Sample request that caused this error
    pub sample_request: Option<String>,
}

/// Time series data for visualization
#[derive(Debug, Clone, Serialize)]
pub struct TimeSeriesData {
    /// Time points (seconds from start)
    pub time_points: Vec<f64>,

    /// Requests per second over time
    pub rps_over_time: Vec<f64>,

    /// Response time over time
    pub response_time_over_time: Vec<f64>,

    /// Error rate over time
    pub error_rate_over_time: Vec<f64>,

    /// Active users over time
    pub active_users_over_time: Vec<u64>,
}

/// Load testing service
pub struct LoadTestService {
    config: LoadTestConfig,
    client: Client,
    results: Arc<Mutex<LoadTestResults>>,
    metrics_collector: Arc<MetricsCollector>,
}

/// Metrics collector for real-time statistics
#[derive(Debug)]
pub struct MetricsCollector {
    /// Request counter
    pub total_requests: AtomicU64,

    /// Success counter
    pub successful_requests: AtomicU64,

    /// Error counter
    pub failed_requests: AtomicU64,

    /// Response times (in milliseconds)
    pub response_times: RwLock<Vec<f64>>,

    /// Error details
    pub errors: RwLock<Vec<ErrorDetail>>,

    /// Time series data
    pub time_series: RwLock<TimeSeriesData>,

    /// Active users
    pub active_users: AtomicU64,

    /// Start time
    pub start_time: Instant,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            response_times: RwLock::new(Vec::new()),
            errors: RwLock::new(Vec::new()),
            time_series: RwLock::new(TimeSeriesData {
                time_points: Vec::new(),
                rps_over_time: Vec::new(),
                response_time_over_time: Vec::new(),
                error_rate_over_time: Vec::new(),
                active_users_over_time: Vec::new(),
            }),
            active_users: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Record a successful request
    pub async fn record_success(&self, response_time_ms: f64) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.successful_requests.fetch_add(1, Ordering::Relaxed);

        let mut response_times = self.response_times.write().await;
        response_times.push(response_time_ms);
    }

    /// Record a failed request
    pub async fn record_error(&self, error_message: String, response_time_ms: Option<f64>) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);

        if let Some(time) = response_time_ms {
            let mut response_times = self.response_times.write().await;
            response_times.push(time);
        }

        let mut errors = self.errors.write().await;
        errors.push(ErrorDetail {
            message: error_message,
            count: 1,
            percentage: 0.0, // Will be calculated later
            sample_request: None,
        });
    }

    /// Update time series data
    pub async fn update_time_series(&self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let _successful_requests = self.successful_requests.load(Ordering::Relaxed);
        let failed_requests = self.failed_requests.load(Ordering::Relaxed);
        let active_users = self.active_users.load(Ordering::Relaxed);

        let response_times = self.response_times.read().await;
        let avg_response_time = if !response_times.is_empty() {
            response_times.iter().sum::<f64>() / response_times.len() as f64
        } else {
            0.0
        };

        let rps = if elapsed > 0.0 { total_requests as f64 / elapsed } else { 0.0 };
        let error_rate = if total_requests > 0 {
            (failed_requests as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };

        let mut time_series = self.time_series.write().await;
        time_series.time_points.push(elapsed);
        time_series.rps_over_time.push(rps);
        time_series.response_time_over_time.push(avg_response_time);
        time_series.error_rate_over_time.push(error_rate);
        time_series.active_users_over_time.push(active_users);
    }

    /// Get current statistics
    pub async fn get_current_stats(&self) -> (u64, u64, u64, f64, f64) {
        let total = self.total_requests.load(Ordering::Relaxed);
        let success = self.successful_requests.load(Ordering::Relaxed);
        let failed = self.failed_requests.load(Ordering::Relaxed);

        let response_times = self.response_times.read().await;
        let avg_response_time = if !response_times.is_empty() {
            response_times.iter().sum::<f64>() / response_times.len() as f64
        } else {
            0.0
        };

        let elapsed = self.start_time.elapsed().as_secs_f64();
        let rps = if elapsed > 0.0 { total as f64 / elapsed } else { 0.0 };

        (total, success, failed, avg_response_time, rps)
    }
}

impl LoadTestService {
    /// Create a new load test service
    pub fn new(config: LoadTestConfig) -> Result<Self> {
        let timeout = Duration::from_millis(config.timeout_config.request_timeout_ms);

        let mut client_builder = Client::builder()
            .timeout(timeout)
            .connection_verbose(true)
            .pool_idle_timeout(Duration::from_secs(
                config.advanced_config.keep_alive_timeout_seconds,
            ))
            .pool_max_idle_per_host(config.advanced_config.max_idle_per_host);

        if config.advanced_config.enable_http2 {
            client_builder = client_builder.http2_prior_knowledge();
        }

        if config.advanced_config.enable_compression {
            // Compression is enabled by default in modern reqwest
            // No additional configuration needed
        }

        let client = client_builder.build()?;
        let metrics_collector = Arc::new(MetricsCollector::new());

        let results = Arc::new(Mutex::new(LoadTestResults {
            config: config.clone(),
            summary: TestSummary {
                start_time: chrono::Utc::now(),
                end_time: chrono::Utc::now(),
                duration_seconds: 0.0,
                total_requests: 0,
                successful_requests: 0,
                failed_requests: 0,
                success_rate: 0.0,
                average_rps: 0.0,
                peak_rps: 0.0,
                total_bytes_transferred: 0,
                average_response_time_ms: 0.0,
                median_response_time_ms: 0.0,
            },
            scenario_results: HashMap::new(),
            performance_metrics: PerformanceMetrics {
                cpu_usage_percent: None,
                memory_usage_mb: None,
                network_utilization: None,
                connection_pool_stats: None,
            },
            error_analysis: ErrorAnalysis {
                total_errors: 0,
                error_rate: 0.0,
                error_types: HashMap::new(),
                status_code_breakdown: HashMap::new(),
                top_errors: Vec::new(),
            },
            time_series: TimeSeriesData {
                time_points: Vec::new(),
                rps_over_time: Vec::new(),
                response_time_over_time: Vec::new(),
                error_rate_over_time: Vec::new(),
                active_users_over_time: Vec::new(),
            },
        }));

        Ok(Self {
            config,
            client,
            results,
            metrics_collector,
        })
    }

    /// Run the load test
    pub async fn run(&self) -> Result<LoadTestResults> {
        println!("🚀 Starting load test...");
        println!("📊 Configuration:");
        println!("   Base URL: {}", self.config.base_url);
        println!("   Duration: {} seconds", self.config.duration_seconds);
        println!("   Concurrent users: {}", self.config.concurrent_users);
        println!("   Scenarios: {}", self.config.scenarios.len());

        let start_time = Instant::now();

        // Start metrics collection
        let metrics_collector = self.metrics_collector.clone();
        let metrics_task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));
            loop {
                interval.tick().await;
                metrics_collector.update_time_series().await;
            }
        });

        // Start real-time reporting if enabled
        let reporting_task = if self.config.output_config.enable_realtime_output {
            let metrics_collector = self.metrics_collector.clone();
            Some(tokio::spawn(async move {
                let mut interval = interval(Duration::from_secs(5));
                loop {
                    interval.tick().await;
                    let (total, success, failed, avg_response_time, rps) =
                        metrics_collector.get_current_stats().await;

                    println!(
                        "📈 Progress: {} requests | Success: {} | Failed: {} | Avg RT: {:.2}ms | RPS: {:.2}",
                        total, success, failed, avg_response_time, rps
                    );
                }
            }))
        } else {
            None
        };

        // Run load test
        self.execute_load_test().await?;

        // Stop background tasks
        metrics_task.abort();
        if let Some(task) = reporting_task {
            task.abort();
        }

        // Calculate final results
        let results = self.calculate_results(start_time.elapsed()).await?;

        println!("✅ Load test completed!");
        self.print_summary(&results).await;

        Ok(results)
    }

    /// Execute the main load testing logic
    async fn execute_load_test(&self) -> Result<()> {
        let duration = Duration::from_secs(self.config.duration_seconds);
        let concurrent_users = self.config.concurrent_users;

        // Create tasks for concurrent users
        let mut tasks = Vec::new();

        for user_id in 0..concurrent_users {
            let client = self.client.clone();
            let config = self.config.clone();
            let metrics_collector = self.metrics_collector.clone();

            let task = tokio::spawn(async move {
                Self::simulate_user(user_id, client, config, metrics_collector, duration).await
            });

            tasks.push(task);

            // Gradual ramp-up
            if self.config.advanced_config.ramp_up_seconds > 0 {
                let ramp_delay = Duration::from_millis(
                    (self.config.advanced_config.ramp_up_seconds * 1000) / concurrent_users as u64,
                );
                sleep(ramp_delay).await;
            }
        }

        // Wait for all users to complete
        for task in tasks {
            if let Err(e) = task.await {
                eprintln!("User task failed: {}", e);
            }
        }

        Ok(())
    }

    /// Simulate a single user's behavior
    async fn simulate_user(
        _user_id: usize,
        client: Client,
        config: LoadTestConfig,
        metrics_collector: Arc<MetricsCollector>,
        duration: Duration,
    ) -> Result<()> {
        let start_time = Instant::now();
        metrics_collector.active_users.fetch_add(1, Ordering::Relaxed);

        while start_time.elapsed() < duration {
            // Select scenario based on weights
            let scenario = Self::select_scenario(&config.scenarios);

            // Execute request
            let request_start = Instant::now();
            match Self::execute_request(&client, &config, scenario).await {
                Ok(response_time) => {
                    metrics_collector.record_success(response_time).await;
                },
                Err(e) => {
                    let response_time = request_start.elapsed().as_millis() as f64;
                    metrics_collector.record_error(e.to_string(), Some(response_time)).await;
                },
            }

            // Think time
            if let Some(think_time) = config.advanced_config.think_time_ms {
                sleep(Duration::from_millis(think_time)).await;
            }

            // Rate limiting
            if let Some(rps) = config.requests_per_second {
                let request_interval =
                    Duration::from_millis((1000.0 / rps * config.concurrent_users as f64) as u64);
                sleep(request_interval).await;
            }
        }

        metrics_collector.active_users.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    /// Select a scenario based on weights
    fn select_scenario(scenarios: &[TestScenario]) -> &TestScenario {
        use scirs2_core::random::*;
        let mut rng = thread_rng();

        // Simple random selection based on weights
        let total_weight: f64 = scenarios.iter().map(|s| s.weight).sum();
        let random_value = rng.random::<f64>() * total_weight;

        let mut cumulative_weight = 0.0;
        for scenario in scenarios {
            cumulative_weight += scenario.weight;
            if random_value <= cumulative_weight {
                return scenario;
            }
        }

        // Fallback to first scenario
        &scenarios[0]
    }

    /// Execute a single request
    async fn execute_request(
        client: &Client,
        config: &LoadTestConfig,
        scenario: &TestScenario,
    ) -> Result<f64> {
        let url = format!("{}{}", config.base_url, scenario.path);
        let start_time = Instant::now();

        let mut request_builder = match scenario.method {
            HttpMethod::Get => client.get(&url),
            HttpMethod::Post => client.post(&url),
            HttpMethod::Put => client.put(&url),
            HttpMethod::Delete => client.delete(&url),
            HttpMethod::Patch => client.patch(&url),
            HttpMethod::Head => client.head(&url),
            HttpMethod::Options => client.request(reqwest::Method::OPTIONS, &url),
        };

        // Add authentication
        if let Some(auth) = &config.auth {
            request_builder = Self::add_authentication(request_builder, auth);
        }

        // Add headers
        for (key, value) in &config.advanced_config.custom_headers {
            request_builder = request_builder.header(key, value);
        }

        for (key, value) in &scenario.headers {
            request_builder = request_builder.header(key, value);
        }

        // Add query parameters
        if !scenario.query_params.is_empty() {
            request_builder = request_builder.query(&scenario.query_params);
        }

        // Add body
        if let Some(body_template) = &scenario.body_template {
            request_builder = request_builder
                .header("Content-Type", "application/json")
                .body(body_template.clone());
        }

        // Execute request
        let response = request_builder.send().await?;
        let response_time = start_time.elapsed().as_millis() as f64;

        // Validate response
        Self::validate_response(response, scenario).await?;

        Ok(response_time)
    }

    /// Add authentication to request
    fn add_authentication(
        mut request_builder: reqwest::RequestBuilder,
        auth: &AuthConfig,
    ) -> reqwest::RequestBuilder {
        match auth.auth_type {
            AuthType::None => request_builder,
            AuthType::ApiKey => {
                if let Some(api_key) = &auth.api_key {
                    request_builder = request_builder.header("X-API-Key", api_key);
                }
                request_builder
            },
            AuthType::Bearer => {
                if let Some(token) = &auth.jwt_token {
                    request_builder = request_builder.bearer_auth(token);
                }
                request_builder
            },
            AuthType::OAuth2 => {
                // OAuth2 implementation would require token management
                request_builder
            },
        }
    }

    /// Validate response according to scenario rules
    async fn validate_response(response: Response, scenario: &TestScenario) -> Result<()> {
        // Check status code
        if !scenario.expected_status_codes.contains(&response.status().as_u16()) {
            return Err(anyhow::anyhow!(
                "Unexpected status code: {} (expected: {:?})",
                response.status(),
                scenario.expected_status_codes
            ));
        }

        // Apply validation rules
        let response_text = response.text().await?;

        for rule in &scenario.validation_rules {
            match &rule.rule_type {
                ValidationRuleType::ResponseContains => {
                    if let Some(expected) = &rule.expected_value {
                        if !response_text.contains(expected) {
                            return Err(anyhow::anyhow!(
                                "Response does not contain expected text: {}",
                                expected
                            ));
                        }
                    }
                },
                ValidationRuleType::ResponseMatches => {
                    if let Some(pattern) = &rule.pattern {
                        let regex = regex::Regex::new(pattern)?;
                        if !regex.is_match(&response_text) {
                            return Err(anyhow::anyhow!(
                                "Response does not match pattern: {}",
                                pattern
                            ));
                        }
                    }
                },
                ValidationRuleType::JsonFieldExists => {
                    // JSON validation would require parsing
                    // Implementation depends on specific requirements
                },
                ValidationRuleType::ResponseSizeBounds { min, max } => {
                    let size = response_text.len();
                    if size < *min || size > *max {
                        return Err(anyhow::anyhow!(
                            "Response size {} is outside bounds [{}, {}]",
                            size,
                            min,
                            max
                        ));
                    }
                },
                _ => {
                    // Other validation rules can be implemented as needed
                },
            }
        }

        Ok(())
    }

    /// Calculate final results
    async fn calculate_results(&self, elapsed: Duration) -> Result<LoadTestResults> {
        let mut results = self.results.lock().await;

        let total_requests = self.metrics_collector.total_requests.load(Ordering::Relaxed);
        let successful_requests =
            self.metrics_collector.successful_requests.load(Ordering::Relaxed);
        let failed_requests = self.metrics_collector.failed_requests.load(Ordering::Relaxed);

        let response_times = self.metrics_collector.response_times.read().await;

        results.summary.end_time = chrono::Utc::now();
        results.summary.duration_seconds = elapsed.as_secs_f64();
        results.summary.total_requests = total_requests;
        results.summary.successful_requests = successful_requests;
        results.summary.failed_requests = failed_requests;
        results.summary.success_rate = if total_requests > 0 {
            (successful_requests as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };
        results.summary.average_rps = if elapsed.as_secs_f64() > 0.0 {
            total_requests as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        if !response_times.is_empty() {
            results.summary.average_response_time_ms =
                response_times.iter().sum::<f64>() / response_times.len() as f64;

            let mut sorted_times = response_times.clone();
            sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            results.summary.median_response_time_ms = sorted_times[sorted_times.len() / 2];
        }

        results.time_series = self.metrics_collector.time_series.read().await.clone();

        Ok(results.clone())
    }

    /// Print test summary
    async fn print_summary(&self, results: &LoadTestResults) {
        println!("\n📊 Load Test Results Summary");
        println!("═══════════════════════════════════════");
        println!(
            "⏱️  Duration: {:.2} seconds",
            results.summary.duration_seconds
        );
        println!("📊 Total Requests: {}", results.summary.total_requests);
        println!(
            "✅ Successful: {} ({:.2}%)",
            results.summary.successful_requests, results.summary.success_rate
        );
        println!(
            "❌ Failed: {} ({:.2}%)",
            results.summary.failed_requests,
            100.0 - results.summary.success_rate
        );
        println!("🚀 Average RPS: {:.2}", results.summary.average_rps);
        println!(
            "⚡ Avg Response Time: {:.2}ms",
            results.summary.average_response_time_ms
        );
        println!(
            "📈 Median Response Time: {:.2}ms",
            results.summary.median_response_time_ms
        );
        println!("═══════════════════════════════════════");
    }
}

/// Load test configuration builder
pub struct LoadTestConfigBuilder {
    config: LoadTestConfig,
}

impl Default for LoadTestConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadTestConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: LoadTestConfig::default(),
        }
    }

    pub fn base_url(mut self, url: &str) -> Self {
        self.config.base_url = url.to_string();
        self
    }

    pub fn duration(mut self, seconds: u64) -> Self {
        self.config.duration_seconds = seconds;
        self
    }

    pub fn concurrent_users(mut self, users: usize) -> Self {
        self.config.concurrent_users = users;
        self
    }

    pub fn requests_per_second(mut self, rps: f64) -> Self {
        self.config.requests_per_second = Some(rps);
        self
    }

    pub fn add_scenario(mut self, scenario: TestScenario) -> Self {
        self.config.scenarios.push(scenario);
        self
    }

    pub fn auth(mut self, auth: AuthConfig) -> Self {
        self.config.auth = Some(auth);
        self
    }

    pub fn build(self) -> LoadTestConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = LoadTestConfigBuilder::new()
            .base_url("http://localhost:8080")
            .duration(60)
            .concurrent_users(10)
            .requests_per_second(100.0)
            .build();

        assert_eq!(config.base_url, "http://localhost:8080");
        assert_eq!(config.duration_seconds, 60);
        assert_eq!(config.concurrent_users, 10);
        assert_eq!(config.requests_per_second, Some(100.0));
    }

    #[test]
    fn test_scenario_selection() {
        let scenarios = vec![
            TestScenario {
                name: "scenario1".to_string(),
                weight: 70.0,
                ..Default::default()
            },
            TestScenario {
                name: "scenario2".to_string(),
                weight: 30.0,
                ..Default::default()
            },
        ];

        let selected = LoadTestService::select_scenario(&scenarios);
        assert!(selected.name == "scenario1" || selected.name == "scenario2");
    }
}
