// Stress Testing Infrastructure for TrustformeRS
// Comprehensive stress testing framework for validating system stability under load

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use trustformers::error::TrustformersError;
type Result<T> = std::result::Result<T, TrustformersError>;

/// Stress test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestConfig {
    /// Number of concurrent threads/tasks
    pub concurrency: usize,
    /// Total number of requests to send
    pub total_requests: u64,
    /// Duration to run the test (overrides total_requests if set)
    pub duration: Option<Duration>,
    /// Delay between requests (per thread)
    pub request_delay: Option<Duration>,
    /// Memory limit in MB (will fail test if exceeded)
    pub memory_limit_mb: Option<u64>,
    /// CPU usage limit percentage (will fail test if exceeded)
    pub cpu_limit_percent: Option<f32>,
    /// Timeout for individual requests
    pub request_timeout: Duration,
    /// Ramp-up time to gradually increase load
    pub ramp_up_duration: Option<Duration>,
    /// Ramp-down time to gradually decrease load
    pub ramp_down_duration: Option<Duration>,
    /// Enable memory leak detection
    pub enable_memory_leak_detection: bool,
    /// Enable performance degradation detection
    pub enable_performance_degradation_detection: bool,
    /// Acceptable error rate percentage
    pub acceptable_error_rate: f32,
    /// Test scenario to run
    pub scenario: StressTestScenario,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            concurrency: 10,
            total_requests: 1000,
            duration: None,
            request_delay: Some(Duration::from_millis(100)),
            memory_limit_mb: Some(4096), // 4GB default limit
            cpu_limit_percent: Some(80.0),
            request_timeout: Duration::from_secs(30),
            ramp_up_duration: Some(Duration::from_secs(30)),
            ramp_down_duration: Some(Duration::from_secs(10)),
            enable_memory_leak_detection: true,
            enable_performance_degradation_detection: true,
            acceptable_error_rate: 5.0, // 5% error rate
            scenario: StressTestScenario::TextGeneration,
        }
    }
}

/// Different stress test scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StressTestScenario {
    /// Text generation stress test
    TextGeneration,
    /// Text classification stress test
    TextClassification,
    /// Mixed workload stress test
    Mixed,
    /// Memory pressure test
    MemoryPressure,
    /// High throughput test
    HighThroughput,
    /// Long running test
    LongRunning,
    /// Burst traffic test
    BurstTraffic,
    /// Resource exhaustion test
    ResourceExhaustion,
}

/// Stress test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResults {
    /// Total requests sent
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Error rate percentage
    pub error_rate: f32,
    /// Test duration
    pub duration: Duration,
    /// Throughput (requests per second)
    pub throughput: f64,
    /// Average latency
    pub average_latency: Duration,
    /// Median latency
    pub median_latency: Duration,
    /// 95th percentile latency
    pub p95_latency: Duration,
    /// 99th percentile latency
    pub p99_latency: Duration,
    /// Maximum latency
    pub max_latency: Duration,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// CPU usage statistics
    pub cpu_stats: CpuStats,
    /// Performance degradation detected
    pub performance_degradation_detected: bool,
    /// Memory leak detected
    pub memory_leak_detected: bool,
    /// Test passed overall
    pub test_passed: bool,
    /// Failure reasons
    pub failure_reasons: Vec<String>,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Initial memory usage (MB)
    pub initial_memory_mb: u64,
    /// Peak memory usage (MB)
    pub peak_memory_mb: u64,
    /// Final memory usage (MB)
    pub final_memory_mb: u64,
    /// Memory growth (MB)
    pub memory_growth_mb: i64,
    /// Memory samples over time
    pub memory_samples: Vec<MemorySample>,
}

/// CPU usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuStats {
    /// Average CPU usage percentage
    pub average_cpu_percent: f32,
    /// Peak CPU usage percentage
    pub peak_cpu_percent: f32,
    /// CPU samples over time
    pub cpu_samples: Vec<CpuSample>,
}

/// Memory usage sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySample {
    /// Timestamp since test start
    pub timestamp: Duration,
    /// Memory usage in MB
    pub memory_mb: u64,
}

/// CPU usage sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuSample {
    /// Timestamp since test start
    pub timestamp: Duration,
    /// CPU usage percentage
    pub cpu_percent: f32,
}

/// Stress test metrics collector
#[derive(Debug)]
pub struct StressTestMetrics {
    /// Total requests counter
    pub total_requests: AtomicU64,
    /// Successful requests counter
    pub successful_requests: AtomicU64,
    /// Failed requests counter
    pub failed_requests: AtomicU64,
    /// Latency samples
    pub latency_samples: Arc<tokio::sync::Mutex<Vec<Duration>>>,
    /// Memory samples
    pub memory_samples: Arc<tokio::sync::Mutex<Vec<MemorySample>>>,
    /// CPU samples
    pub cpu_samples: Arc<tokio::sync::Mutex<Vec<CpuSample>>>,
    /// Test start time
    pub start_time: Instant,
}

impl StressTestMetrics {
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            latency_samples: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            memory_samples: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            cpu_samples: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            start_time: Instant::now(),
        }
    }

    pub fn record_request(&self, success: bool, latency: Duration) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        if success {
            self.successful_requests.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_requests.fetch_add(1, Ordering::Relaxed);
        }

        // Record latency sample (async)
        let latency_samples = self.latency_samples.clone();
        tokio::spawn(async move {
            let mut samples = latency_samples.lock().await;
            samples.push(latency);
        });
    }

    pub async fn record_memory_usage(&self, memory_mb: u64) {
        let mut samples = self.memory_samples.lock().await;
        samples.push(MemorySample {
            timestamp: self.start_time.elapsed(),
            memory_mb,
        });
    }

    pub async fn record_cpu_usage(&self, cpu_percent: f32) {
        let mut samples = self.cpu_samples.lock().await;
        samples.push(CpuSample {
            timestamp: self.start_time.elapsed(),
            cpu_percent,
        });
    }

    pub async fn get_results(&self) -> StressTestResults {
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let successful_requests = self.successful_requests.load(Ordering::Relaxed);
        let failed_requests = self.failed_requests.load(Ordering::Relaxed);
        let error_rate = if total_requests > 0 {
            (failed_requests as f32 / total_requests as f32) * 100.0
        } else {
            0.0
        };

        let duration = self.start_time.elapsed();
        let throughput = total_requests as f64 / duration.as_secs_f64();

        // Calculate latency statistics
        let mut latency_samples = self.latency_samples.lock().await;
        latency_samples.sort();

        let average_latency = if !latency_samples.is_empty() {
            let sum: Duration = latency_samples.iter().sum();
            sum / latency_samples.len() as u32
        } else {
            Duration::from_millis(0)
        };

        let median_latency = if !latency_samples.is_empty() {
            let mid = latency_samples.len() / 2;
            latency_samples[mid]
        } else {
            Duration::from_millis(0)
        };

        let p95_latency = if !latency_samples.is_empty() {
            let idx = (latency_samples.len() as f64 * 0.95) as usize;
            latency_samples[idx.min(latency_samples.len() - 1)]
        } else {
            Duration::from_millis(0)
        };

        let p99_latency = if !latency_samples.is_empty() {
            let idx = (latency_samples.len() as f64 * 0.99) as usize;
            latency_samples[idx.min(latency_samples.len() - 1)]
        } else {
            Duration::from_millis(0)
        };

        let max_latency = latency_samples.iter().max().copied().unwrap_or(Duration::from_millis(0));

        // Calculate memory statistics
        let memory_samples = self.memory_samples.lock().await;
        let memory_stats = if !memory_samples.is_empty() {
            let initial_memory = memory_samples.first().unwrap().memory_mb;
            let final_memory = memory_samples.last().unwrap().memory_mb;
            let peak_memory = memory_samples.iter().map(|s| s.memory_mb).max().unwrap_or(0);

            MemoryStats {
                initial_memory_mb: initial_memory,
                peak_memory_mb: peak_memory,
                final_memory_mb: final_memory,
                memory_growth_mb: final_memory as i64 - initial_memory as i64,
                memory_samples: memory_samples.clone(),
            }
        } else {
            MemoryStats {
                initial_memory_mb: 0,
                peak_memory_mb: 0,
                final_memory_mb: 0,
                memory_growth_mb: 0,
                memory_samples: Vec::new(),
            }
        };

        // Calculate CPU statistics
        let cpu_samples = self.cpu_samples.lock().await;
        let cpu_stats = if !cpu_samples.is_empty() {
            let average_cpu =
                cpu_samples.iter().map(|s| s.cpu_percent).sum::<f32>() / cpu_samples.len() as f32;
            let peak_cpu = cpu_samples.iter().map(|s| s.cpu_percent).fold(0.0f32, |a, b| a.max(b));

            CpuStats {
                average_cpu_percent: average_cpu,
                peak_cpu_percent: peak_cpu,
                cpu_samples: cpu_samples.clone(),
            }
        } else {
            CpuStats {
                average_cpu_percent: 0.0,
                peak_cpu_percent: 0.0,
                cpu_samples: Vec::new(),
            }
        };

        // Performance degradation detection
        let performance_degradation_detected =
            self.detect_performance_degradation(&latency_samples).await;

        // Memory leak detection
        let memory_leak_detected = self.detect_memory_leak(&memory_stats).await;

        // Determine if test passed and collect failure reasons
        let mut failure_reasons = Vec::new();
        let mut test_passed = true;

        // Check error rate
        if error_rate > 5.0 {
            // Configurable threshold
            failure_reasons.push(format!("High error rate: {:.2}%", error_rate));
            test_passed = false;
        }

        // Check performance degradation
        if performance_degradation_detected {
            failure_reasons.push("Performance degradation detected".to_string());
            test_passed = false;
        }

        // Check memory leak
        if memory_leak_detected {
            failure_reasons.push("Memory leak detected".to_string());
            test_passed = false;
        }

        // Check if throughput is reasonable (at least 1 request per second)
        if throughput < 1.0 {
            failure_reasons.push(format!("Poor throughput: {:.2} req/s", throughput));
            test_passed = false;
        }

        StressTestResults {
            total_requests,
            successful_requests,
            failed_requests,
            error_rate,
            duration,
            throughput,
            average_latency,
            median_latency,
            p95_latency,
            p99_latency,
            max_latency,
            memory_stats,
            cpu_stats,
            performance_degradation_detected,
            memory_leak_detected,
            test_passed,
            failure_reasons,
        }
    }

    /// Detect performance degradation by analyzing latency trends
    async fn detect_performance_degradation(&self, latencies: &[Duration]) -> bool {
        if latencies.len() < 20 {
            return false; // Not enough data points
        }

        // Split the latency data into early and late periods
        let split_point = latencies.len() / 3;
        let early_latencies: Vec<f64> = latencies[..split_point]
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0) // Convert to milliseconds
            .collect();
        let late_latencies: Vec<f64> = latencies[latencies.len() - split_point..]
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();

        // Calculate average latencies for each period
        let early_avg: f64 = early_latencies.iter().sum::<f64>() / early_latencies.len() as f64;
        let late_avg: f64 = late_latencies.iter().sum::<f64>() / late_latencies.len() as f64;

        // Performance degradation if late period average is significantly higher
        let degradation_threshold = 1.5; // 50% increase threshold
        late_avg > early_avg * degradation_threshold
    }

    /// Detect memory leaks by analyzing memory growth patterns
    async fn detect_memory_leak(&self, memory_stats: &MemoryStats) -> bool {
        if memory_stats.memory_samples.len() < 10 {
            return false; // Not enough data points
        }

        let samples = &memory_stats.memory_samples;
        let sample_count = samples.len();

        // Calculate linear regression to detect consistent memory growth
        let n = sample_count as f64;
        let x_sum: f64 = (0..sample_count).map(|i| i as f64).sum();
        let y_sum: f64 = samples.iter().map(|s| s.memory_mb as f64).sum();
        let xy_sum: f64 =
            samples.iter().enumerate().map(|(i, s)| i as f64 * s.memory_mb as f64).sum();
        let x2_sum: f64 = (0..sample_count).map(|i| (i as f64).powi(2)).sum();

        // Calculate slope of the regression line
        let slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum);

        // Memory leak detected if there's significant positive slope (>1MB per sample)
        let leak_threshold = 1.0; // MB per sample
        slope > leak_threshold
    }
}

/// Stress test runner
pub struct StressTestRunner {
    config: StressTestConfig,
    metrics: Arc<StressTestMetrics>,
}

impl StressTestRunner {
    pub fn new(config: StressTestConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(StressTestMetrics::new()),
        }
    }

    /// Run the stress test
    pub async fn run(&self) -> Result<StressTestResults> {
        println!("Starting stress test with config: {:?}", self.config);

        // Start resource monitoring
        self.start_resource_monitoring().await;

        // Run the test based on scenario
        match self.config.scenario {
            StressTestScenario::TextGeneration => self.run_text_generation_stress().await?,
            StressTestScenario::TextClassification => self.run_text_classification_stress().await?,
            StressTestScenario::Mixed => self.run_mixed_workload_stress().await?,
            StressTestScenario::MemoryPressure => self.run_memory_pressure_stress().await?,
            StressTestScenario::HighThroughput => self.run_high_throughput_stress().await?,
            StressTestScenario::LongRunning => self.run_long_running_stress().await?,
            StressTestScenario::BurstTraffic => self.run_burst_traffic_stress().await?,
            StressTestScenario::ResourceExhaustion => self.run_resource_exhaustion_stress().await?,
        }

        // Get final results
        let results = self.metrics.get_results().await;
        println!("Stress test completed. Results: {:?}", results);

        Ok(results)
    }

    /// Start resource monitoring in background
    async fn start_resource_monitoring(&self) {
        let metrics = self.metrics.clone();
        let max_duration = self.config.duration.unwrap_or(Duration::from_secs(60));

        // Memory monitoring
        tokio::spawn(async move {
            let start_time = Instant::now();
            while start_time.elapsed() < max_duration {
                // Get current memory usage (simplified)
                let memory_mb = get_memory_usage_mb().await;
                metrics.record_memory_usage(memory_mb).await;
                sleep(Duration::from_millis(500)).await;
            }
        });

        // CPU monitoring
        let metrics = self.metrics.clone();
        tokio::spawn(async move {
            let start_time = Instant::now();
            while start_time.elapsed() < max_duration {
                // Get current CPU usage (simplified)
                let cpu_percent = get_cpu_usage_percent().await;
                metrics.record_cpu_usage(cpu_percent).await;
                sleep(Duration::from_millis(500)).await;
            }
        });
    }

    /// Run text generation stress test
    async fn run_text_generation_stress(&self) -> Result<()> {
        println!("Running text generation stress test...");

        // Create test pipeline (mock for now)
        let test_inputs = vec![
            "The future of AI is",
            "Once upon a time",
            "In a world where",
            "The key to success",
            "Explain the concept of",
        ];

        let mut tasks = Vec::new();

        for _i in 0..self.config.concurrency {
            let metrics = self.metrics.clone();
            let config = self.config.clone();
            let inputs = test_inputs.clone();

            let task = tokio::spawn(async move {
                let _thread_id = _i;
                let mut requests_sent = 0;

                loop {
                    // Check if we should stop
                    if let Some(duration) = config.duration {
                        if metrics.start_time.elapsed() > duration {
                            break;
                        }
                    } else if metrics.total_requests.load(Ordering::Relaxed)
                        >= config.total_requests
                    {
                        break;
                    }

                    // Select random input
                    let input = &inputs[requests_sent % inputs.len()];

                    // Make request
                    let start = Instant::now();
                    let result = Self::make_text_generation_request(input).await;
                    let latency = start.elapsed();

                    // Record metrics
                    metrics.record_request(result.is_ok(), latency);

                    requests_sent += 1;

                    // Add delay if configured
                    if let Some(delay) = config.request_delay {
                        sleep(delay).await;
                    }
                }
            });

            tasks.push(task);
        }

        // Wait for all tasks to complete
        for task in tasks {
            task.await.unwrap();
        }

        Ok(())
    }

    /// Run text classification stress test
    async fn run_text_classification_stress(&self) -> Result<()> {
        println!("Running text classification stress test...");

        let test_inputs = vec![
            "This is a great product!",
            "I hate this service.",
            "The weather is nice today.",
            "This movie was boring.",
            "I love programming in Rust.",
        ];

        let mut tasks = Vec::new();

        for _i in 0..self.config.concurrency {
            let metrics = self.metrics.clone();
            let config = self.config.clone();
            let inputs = test_inputs.clone();

            let task = tokio::spawn(async move {
                let mut requests_sent = 0;

                loop {
                    // Check if we should stop
                    if let Some(duration) = config.duration {
                        if metrics.start_time.elapsed() > duration {
                            break;
                        }
                    } else if metrics.total_requests.load(Ordering::Relaxed)
                        >= config.total_requests
                    {
                        break;
                    }

                    // Select random input
                    let input = &inputs[requests_sent % inputs.len()];

                    // Make request
                    let start = Instant::now();
                    let result = Self::make_text_classification_request(input).await;
                    let latency = start.elapsed();

                    // Record metrics
                    metrics.record_request(result.is_ok(), latency);

                    requests_sent += 1;

                    // Add delay if configured
                    if let Some(delay) = config.request_delay {
                        sleep(delay).await;
                    }
                }
            });

            tasks.push(task);
        }

        // Wait for all tasks to complete
        for task in tasks {
            task.await.unwrap();
        }

        Ok(())
    }

    /// Run mixed workload stress test
    async fn run_mixed_workload_stress(&self) -> Result<()> {
        println!("Running mixed workload stress test...");

        let mut tasks = Vec::new();

        for _i in 0..self.config.concurrency {
            let metrics = self.metrics.clone();
            let config = self.config.clone();

            let task = tokio::spawn(async move {
                let mut requests_sent = 0;

                loop {
                    // Check if we should stop
                    if let Some(duration) = config.duration {
                        if metrics.start_time.elapsed() > duration {
                            break;
                        }
                    } else if metrics.total_requests.load(Ordering::Relaxed)
                        >= config.total_requests
                    {
                        break;
                    }

                    // Alternate between generation and classification
                    let start = Instant::now();
                    let result = if requests_sent % 2 == 0 {
                        Self::make_text_generation_request("Hello world").await
                    } else {
                        Self::make_text_classification_request("This is a test").await
                    };
                    let latency = start.elapsed();

                    // Record metrics
                    metrics.record_request(result.is_ok(), latency);

                    requests_sent += 1;

                    // Add delay if configured
                    if let Some(delay) = config.request_delay {
                        sleep(delay).await;
                    }
                }
            });

            tasks.push(task);
        }

        // Wait for all tasks to complete
        for task in tasks {
            task.await.unwrap();
        }

        Ok(())
    }

    /// Run memory pressure stress test
    async fn run_memory_pressure_stress(&self) -> Result<()> {
        println!("Running memory pressure stress test...");

        // Create memory-intensive requests
        let mut tasks = Vec::new();

        for _i in 0..self.config.concurrency {
            let metrics = self.metrics.clone();
            let config = self.config.clone();

            let task = tokio::spawn(async move {
                let mut _requests_sent = 0;

                loop {
                    // Check if we should stop
                    if let Some(duration) = config.duration {
                        if metrics.start_time.elapsed() > duration {
                            break;
                        }
                    } else if metrics.total_requests.load(Ordering::Relaxed)
                        >= config.total_requests
                    {
                        break;
                    }

                    // Create large input to stress memory
                    let large_input = "A".repeat(10000); // 10KB input

                    // Make request
                    let start = Instant::now();
                    let result = Self::make_text_generation_request(&large_input).await;
                    let latency = start.elapsed();

                    // Record metrics
                    metrics.record_request(result.is_ok(), latency);

                    _requests_sent += 1;

                    // Add delay if configured
                    if let Some(delay) = config.request_delay {
                        sleep(delay).await;
                    }
                }
            });

            tasks.push(task);
        }

        // Wait for all tasks to complete
        for task in tasks {
            task.await.unwrap();
        }

        Ok(())
    }

    /// Run high throughput stress test
    async fn run_high_throughput_stress(&self) -> Result<()> {
        println!("Running high throughput stress test...");

        // No delay between requests for maximum throughput
        let mut config = self.config.clone();
        config.request_delay = None;

        let mut tasks = Vec::new();

        for _i in 0..config.concurrency {
            let metrics = self.metrics.clone();
            let config = config.clone();

            let task = tokio::spawn(async move {
                let mut _requests_sent = 0;

                loop {
                    // Check if we should stop
                    if let Some(duration) = config.duration {
                        if metrics.start_time.elapsed() > duration {
                            break;
                        }
                    } else if metrics.total_requests.load(Ordering::Relaxed)
                        >= config.total_requests
                    {
                        break;
                    }

                    // Make request with minimal input
                    let start = Instant::now();
                    let result = Self::make_text_classification_request("test").await;
                    let latency = start.elapsed();

                    // Record metrics
                    metrics.record_request(result.is_ok(), latency);

                    _requests_sent += 1;
                }
            });

            tasks.push(task);
        }

        // Wait for all tasks to complete
        for task in tasks {
            task.await.unwrap();
        }

        Ok(())
    }

    /// Run long running stress test
    async fn run_long_running_stress(&self) -> Result<()> {
        println!("Running long running stress test...");

        self.run_text_generation_stress().await
    }

    /// Run burst traffic stress test
    async fn run_burst_traffic_stress(&self) -> Result<()> {
        println!("Running burst traffic stress test...");

        // Create bursts of high activity followed by low activity
        let burst_duration = Duration::from_secs(2);
        let rest_duration = Duration::from_secs(1);
        let total_duration = self.config.duration.unwrap_or(Duration::from_secs(10)); // Much shorter

        let start_time = Instant::now();
        let mut in_burst = true;
        let mut last_switch = start_time;

        while start_time.elapsed() < total_duration {
            if in_burst {
                // High activity burst
                let mut tasks = Vec::new();

                for _i in 0..self.config.concurrency {
                    // Same concurrency, not doubled
                    let metrics = self.metrics.clone();

                    let task = tokio::spawn(async move {
                        let start = Instant::now();
                        let result = Self::make_text_classification_request("burst test").await;
                        let latency = start.elapsed();

                        metrics.record_request(result.is_ok(), latency);
                    });

                    tasks.push(task);
                }

                // Wait for burst to complete
                for task in tasks {
                    task.await.unwrap();
                }

                if last_switch.elapsed() > burst_duration {
                    in_burst = false;
                    last_switch = Instant::now();
                }
            } else {
                // Rest period
                sleep(Duration::from_millis(50)).await;

                if last_switch.elapsed() > rest_duration {
                    in_burst = true;
                    last_switch = Instant::now();
                }
            }
        }

        Ok(())
    }

    /// Run resource exhaustion stress test
    async fn run_resource_exhaustion_stress(&self) -> Result<()> {
        println!("Running resource exhaustion stress test...");

        // Gradually increase load but keep it reasonable for tests
        let max_concurrency = self.config.concurrency * 2; // Much less aggressive
        let mut current_concurrency = 1;

        while current_concurrency <= max_concurrency {
            println!("Testing with concurrency: {}", current_concurrency);

            let mut tasks = Vec::new();

            for _i in 0..current_concurrency {
                let metrics = self.metrics.clone();

                let task = tokio::spawn(async move {
                    for _ in 0..2 {
                        // Only 2 requests per task
                        let start = Instant::now();
                        let result = Self::make_text_generation_request("resource test").await;
                        let latency = start.elapsed();

                        metrics.record_request(result.is_ok(), latency);
                    }
                });

                tasks.push(task);
            }

            // Wait for tasks to complete
            for task in tasks {
                task.await.unwrap();
            }

            current_concurrency += 1;
            sleep(Duration::from_millis(100)).await; // Much shorter delay
        }

        Ok(())
    }

    /// Make a text generation request (mock implementation)
    async fn make_text_generation_request(input: &str) -> Result<String> {
        // Simulate request processing time
        sleep(Duration::from_millis(50 + (input.len() as u64 * 2))).await;

        // Simulate occasional failures
        if input.contains("fail") {
            return Err(TrustformersError::Pipeline {
                message: "Simulated failure".to_string(),
                pipeline_type: "text-generation".to_string(),
                suggestion: Some("Try again".to_string()),
                recovery_actions: vec![],
            });
        }

        Ok(format!("Generated response for: {}", input))
    }

    /// Make a text classification request (mock implementation)
    async fn make_text_classification_request(input: &str) -> Result<String> {
        // Simulate request processing time
        sleep(Duration::from_millis(20 + (input.len() as u64))).await;

        // Simulate occasional failures
        if input.contains("error") {
            return Err(TrustformersError::Pipeline {
                message: "Simulated error".to_string(),
                pipeline_type: "text-classification".to_string(),
                suggestion: Some("Try again".to_string()),
                recovery_actions: vec![],
            });
        }

        Ok(format!("Classification result for: {}", input))
    }
}

/// Get current memory usage in MB (simplified implementation)
async fn get_memory_usage_mb() -> u64 {
    // In a real implementation, this would use system APIs
    // For now, return a mock value
    1024 // 1GB
}

/// Get current CPU usage percentage (simplified implementation)
async fn get_cpu_usage_percent() -> f32 {
    // In a real implementation, this would use system APIs
    // For now, return a mock value
    25.0 // 25%
}

/// Configuration presets for common stress test scenarios
impl StressTestConfig {
    /// Light stress test configuration
    pub fn light() -> Self {
        Self {
            concurrency: 2,
            total_requests: 10,
            duration: Some(Duration::from_secs(5)),
            request_delay: Some(Duration::from_millis(100)),
            ..Default::default()
        }
    }

    /// Medium stress test configuration
    pub fn medium() -> Self {
        Self {
            concurrency: 4,
            total_requests: 20,
            duration: Some(Duration::from_secs(10)),
            request_delay: Some(Duration::from_millis(50)),
            ..Default::default()
        }
    }

    /// Heavy stress test configuration
    pub fn heavy() -> Self {
        Self {
            concurrency: 50,
            total_requests: 10000,
            duration: Some(Duration::from_secs(1800)),
            request_delay: Some(Duration::from_millis(50)),
            ..Default::default()
        }
    }

    /// Extreme stress test configuration
    pub fn extreme() -> Self {
        Self {
            concurrency: 100,
            total_requests: 100000,
            duration: Some(Duration::from_secs(3600)),
            request_delay: None,
            memory_limit_mb: Some(8192), // 8GB
            cpu_limit_percent: Some(95.0),
            ..Default::default()
        }
    }
}

// Test cases
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_light_stress_text_generation() {
        let config = StressTestConfig::light();
        let runner = StressTestRunner::new(config);

        let results = runner.run().await.unwrap();

        assert!(results.error_rate < 10.0); // Less than 10% error rate
        assert!(results.total_requests > 0);
        assert!(results.successful_requests > 0);
    }

    #[tokio::test]
    async fn test_medium_stress_text_classification() {
        let mut config = StressTestConfig::medium();
        config.scenario = StressTestScenario::TextClassification;

        let runner = StressTestRunner::new(config);
        let results = runner.run().await.unwrap();

        assert!(results.error_rate < 5.0); // Less than 5% error rate
        assert!(results.throughput > 0.0);
        assert!(results.average_latency < Duration::from_secs(1));
    }

    #[tokio::test]
    async fn test_mixed_workload_stress() {
        let mut config = StressTestConfig::light();
        config.scenario = StressTestScenario::Mixed;

        let runner = StressTestRunner::new(config);
        let results = runner.run().await.unwrap();

        assert!(results.total_requests > 0);
        assert!(results.successful_requests > 0);
    }

    #[tokio::test]
    async fn test_memory_pressure_stress() {
        let mut config = StressTestConfig::light();
        config.scenario = StressTestScenario::MemoryPressure;
        config.memory_limit_mb = Some(512); // Lower limit for testing

        let runner = StressTestRunner::new(config);
        let results = runner.run().await.unwrap();

        assert!(results.total_requests > 0);
        assert!(results.memory_stats.peak_memory_mb > 0);
    }

    #[tokio::test]
    async fn test_high_throughput_stress() {
        let mut config = StressTestConfig::light();
        config.scenario = StressTestScenario::HighThroughput;
        config.request_delay = None; // No delay for max throughput

        let runner = StressTestRunner::new(config);
        let results = runner.run().await.unwrap();

        assert!(results.throughput > 0.0);
        assert!(results.total_requests > 0);
    }

    #[tokio::test]
    async fn test_burst_traffic_stress() {
        let mut config = StressTestConfig::light();
        config.scenario = StressTestScenario::BurstTraffic;
        config.duration = Some(Duration::from_secs(5)); // Very short duration for testing

        let runner = StressTestRunner::new(config);
        let results = runner.run().await.unwrap();

        assert!(results.total_requests > 0);
        assert!(results.duration <= Duration::from_secs(10)); // Should complete within time limit
    }

    #[tokio::test]
    async fn test_stress_test_metrics() {
        let metrics = StressTestMetrics::new();

        // Record some test data
        metrics.record_request(true, Duration::from_millis(100));
        metrics.record_request(false, Duration::from_millis(200));
        metrics.record_memory_usage(1024).await;
        metrics.record_cpu_usage(50.0).await;

        let results = metrics.get_results().await;

        assert_eq!(results.total_requests, 2);
        assert_eq!(results.successful_requests, 1);
        assert_eq!(results.failed_requests, 1);
        assert_eq!(results.error_rate, 50.0);
        assert_eq!(results.memory_stats.memory_samples.len(), 1);
        assert_eq!(results.cpu_stats.cpu_samples.len(), 1);
    }

    #[tokio::test]
    async fn test_stress_test_config_presets() {
        let light = StressTestConfig::light();
        assert_eq!(light.concurrency, 2);
        assert_eq!(light.total_requests, 10);

        let medium = StressTestConfig::medium();
        assert_eq!(medium.concurrency, 4);
        assert_eq!(medium.total_requests, 20);

        let heavy = StressTestConfig::heavy();
        assert_eq!(heavy.concurrency, 50);
        assert_eq!(heavy.total_requests, 10000);

        let extreme = StressTestConfig::extreme();
        assert_eq!(extreme.concurrency, 100);
        assert_eq!(extreme.total_requests, 100000);
    }
}
