// Allow dead code for infrastructure under development
#![allow(dead_code)]

//! Test Utilities for Timeout Optimization
//!
//! This module provides convenient utilities and helper functions for test developers
//! to easily integrate with the timeout optimization framework.

use crate::test_timeout_optimization::{
    TestCategory, TestComplexityHints, TestExecutionContext, TestExecutionResult,
    TestProgressTracker, TestTimeoutFramework,
};
use anyhow::Result;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::OnceCell;

/// Global test timeout framework instance
static GLOBAL_FRAMEWORK: OnceCell<Arc<TestTimeoutFramework>> = OnceCell::const_new();

/// Initialize the global test timeout framework
pub async fn init_test_framework() -> Result<()> {
    let config = crate::test_timeout_optimization::TestTimeoutConfig::default();
    let mut framework = TestTimeoutFramework::new(config)?;
    framework.start().await?;

    GLOBAL_FRAMEWORK
        .set(Arc::new(framework))
        .map_err(|_| anyhow::anyhow!("Test framework already initialized"))?;

    Ok(())
}

/// Get the global test timeout framework
pub async fn get_framework() -> Result<Arc<TestTimeoutFramework>> {
    GLOBAL_FRAMEWORK.get().cloned().ok_or_else(|| {
        anyhow::anyhow!("Test framework not initialized. Call init_test_framework() first.")
    })
}

/// Execute a unit test with timeout optimization
pub async fn run_unit_test<F, Fut, T>(test_name: &str, test_fn: F) -> Result<TestExecutionResult>
where
    F: FnOnce(Arc<TestProgressTracker>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<T>> + Send,
    T: Send + 'static + std::fmt::Debug,
{
    let context = TestExecutionContext {
        test_name: test_name.to_string(),
        category: TestCategory::Unit,
        expected_duration: None,
        complexity_hints: TestComplexityHints::default(),
        environment: get_test_environment(),
        timeout_override: None,
    };

    let framework = get_framework().await?;
    framework.execute_test(context, test_fn).await
}

/// Execute an integration test with timeout optimization
pub async fn run_integration_test<F, Fut, T>(
    test_name: &str,
    test_fn: F,
) -> Result<TestExecutionResult>
where
    F: FnOnce(Arc<TestProgressTracker>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<T>> + Send,
    T: Send + 'static + std::fmt::Debug,
{
    let context = TestExecutionContext {
        test_name: test_name.to_string(),
        category: TestCategory::Integration,
        expected_duration: None,
        complexity_hints: TestComplexityHints::default(),
        environment: get_test_environment(),
        timeout_override: None,
    };

    let framework = get_framework().await?;
    framework.execute_test(context, test_fn).await
}

/// Execute a stress test with timeout optimization
pub async fn run_stress_test<F, Fut, T>(
    test_name: &str,
    concurrency: usize,
    memory_mb: u64,
    test_fn: F,
) -> Result<TestExecutionResult>
where
    F: FnOnce(Arc<TestProgressTracker>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<T>> + Send,
    T: Send + 'static + std::fmt::Debug,
{
    let context = TestExecutionContext {
        test_name: test_name.to_string(),
        category: TestCategory::Stress,
        expected_duration: None,
        complexity_hints: TestComplexityHints {
            concurrency_level: Some(concurrency),
            memory_usage: Some(memory_mb),
            network_operations: false,
            file_operations: false,
            gpu_operations: false,
            database_operations: false,
        },
        environment: get_test_environment(),
        timeout_override: None,
    };

    let framework = get_framework().await?;
    framework.execute_test(context, test_fn).await
}

/// Execute a chaos test with timeout optimization
pub async fn run_chaos_test<F, Fut, T>(test_name: &str, test_fn: F) -> Result<TestExecutionResult>
where
    F: FnOnce(Arc<TestProgressTracker>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<T>> + Send,
    T: Send + 'static + std::fmt::Debug,
{
    let context = TestExecutionContext {
        test_name: test_name.to_string(),
        category: TestCategory::Chaos,
        expected_duration: None,
        complexity_hints: TestComplexityHints {
            concurrency_level: Some(20), // Chaos tests typically use high concurrency
            network_operations: true,
            ..Default::default()
        },
        environment: get_test_environment(),
        timeout_override: None,
    };

    let framework = get_framework().await?;
    framework.execute_test(context, test_fn).await
}

/// Execute a property-based test with timeout optimization
pub async fn run_property_test<F, Fut, T>(
    test_name: &str,
    test_fn: F,
) -> Result<TestExecutionResult>
where
    F: FnOnce(Arc<TestProgressTracker>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<T>> + Send,
    T: Send + 'static + std::fmt::Debug,
{
    let context = TestExecutionContext {
        test_name: test_name.to_string(),
        category: TestCategory::Property,
        expected_duration: None,
        complexity_hints: TestComplexityHints::default(),
        environment: get_test_environment(),
        timeout_override: None,
    };

    let framework = get_framework().await?;
    framework.execute_test(context, test_fn).await
}

/// Execute a test with custom configuration
pub async fn run_custom_test<F, Fut, T>(
    context: TestExecutionContext,
    test_fn: F,
) -> Result<TestExecutionResult>
where
    F: FnOnce(Arc<TestProgressTracker>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<T>> + Send,
    T: Send + 'static + std::fmt::Debug,
{
    let framework = get_framework().await?;
    framework.execute_test(context, test_fn).await
}

/// Execute a test with timeout override
pub async fn run_test_with_timeout<F, Fut, T>(
    test_name: &str,
    category: TestCategory,
    timeout: Duration,
    test_fn: F,
) -> Result<TestExecutionResult>
where
    F: FnOnce(Arc<TestProgressTracker>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<T>> + Send,
    T: Send + 'static + std::fmt::Debug,
{
    let context = TestExecutionContext {
        test_name: test_name.to_string(),
        category,
        expected_duration: None,
        complexity_hints: TestComplexityHints::default(),
        environment: get_test_environment(),
        timeout_override: Some(timeout),
    };

    let framework = get_framework().await?;
    framework.execute_test(context, test_fn).await
}

/// Helper to get the current test environment
fn get_test_environment() -> String {
    std::env::var("TEST_ENV").unwrap_or_else(|_| {
        if cfg!(debug_assertions) {
            "debug".to_string()
        } else {
            "release".to_string()
        }
    })
}

/// Helper to create progress tracker with specific total steps
pub fn create_progress_tracker(total_steps: usize) -> Arc<TestProgressTracker> {
    Arc::new(TestProgressTracker::new(total_steps))
}

/// Macro for easy test execution with timeout optimization
#[macro_export]
macro_rules! optimized_test {
    (unit $test_name:expr, $test_body:expr) => {
        $crate::test_utilities::run_unit_test($test_name, |_progress| async move { $test_body })
            .await
    };

    (integration $test_name:expr, $test_body:expr) => {
        $crate::test_utilities::run_integration_test(
            $test_name,
            |_progress| async move { $test_body },
        )
        .await
    };

    (stress $test_name:expr, concurrency = $conc:expr, memory = $mem:expr, $test_body:expr) => {
        $crate::test_utilities::run_stress_test($test_name, $conc, $mem, |_progress| async move {
            $test_body
        })
        .await
    };

    (chaos $test_name:expr, $test_body:expr) => {
        $crate::test_utilities::run_chaos_test($test_name, |_progress| async move { $test_body })
            .await
    };

    (property $test_name:expr, $test_body:expr) => {
        $crate::test_utilities::run_property_test($test_name, |_progress| async move { $test_body })
            .await
    };

    ($test_name:expr, timeout = $timeout:expr, category = $category:expr, $test_body:expr) => {
        $crate::test_utilities::run_test_with_timeout(
            $test_name,
            $category,
            $timeout,
            |_progress| async move { $test_body },
        )
        .await
    };
}

/// Macro for tests with progress tracking
#[macro_export]
macro_rules! optimized_test_with_progress {
    ($category:ident $test_name:expr, steps = $steps:expr, $test_body:expr) => {
        paste::paste! {
            $crate::test_utilities::[<run_ $category _test>]($test_name, |progress| async move {
                let total_steps = $steps;
                progress.total_progress.store(total_steps, std::sync::atomic::Ordering::SeqCst);
                $test_body
            }).await
        }
    };
}

/// Helper trait for adding timeout optimization to existing test functions
#[allow(async_fn_in_trait)]
pub trait TimeoutOptimized<T> {
    async fn with_timeout_optimization(
        self,
        test_name: &str,
        category: TestCategory,
    ) -> Result<TestExecutionResult>;
}

impl<F, Fut, T> TimeoutOptimized<T> for F
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<T>> + Send,
    T: Send + 'static + std::fmt::Debug,
{
    async fn with_timeout_optimization(
        self,
        test_name: &str,
        category: TestCategory,
    ) -> Result<TestExecutionResult> {
        let context = TestExecutionContext {
            test_name: test_name.to_string(),
            category,
            expected_duration: None,
            complexity_hints: TestComplexityHints::default(),
            environment: get_test_environment(),
            timeout_override: None,
        };

        let framework = get_framework().await?;
        framework.execute_test(context, |_progress| async move { self().await }).await
    }
}

/// Test performance benchmarking utilities
pub mod benchmarking {
    use super::*;
    use std::time::Instant;

    /// Benchmark a test function and compare with historical data
    pub async fn benchmark_test<F, Fut, T>(
        test_name: &str,
        iterations: usize,
        test_fn: F,
    ) -> Result<BenchmarkResult>
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<T>> + Send,
        T: Send + 'static,
    {
        let mut execution_times = Vec::new();
        let start_time = Instant::now();

        for i in 0..iterations {
            let iter_start = Instant::now();
            let _ = test_fn().await?;
            let iter_time = iter_start.elapsed();
            execution_times.push(iter_time);

            // Report progress
            if i % (iterations / 10).max(1) == 0 {
                println!("Benchmark progress: {}/{} iterations", i + 1, iterations);
            }
        }

        let total_time = start_time.elapsed();
        let avg_time = execution_times.iter().sum::<Duration>() / iterations as u32;
        let min_time = execution_times.iter().min().copied().unwrap_or_default();
        let max_time = execution_times.iter().max().copied().unwrap_or_default();

        Ok(BenchmarkResult {
            test_name: test_name.to_string(),
            iterations,
            total_time,
            average_time: avg_time,
            min_time,
            max_time,
            execution_times,
        })
    }

    /// Benchmark result
    #[derive(Debug)]
    pub struct BenchmarkResult {
        pub test_name: String,
        pub iterations: usize,
        pub total_time: Duration,
        pub average_time: Duration,
        pub min_time: Duration,
        pub max_time: Duration,
        pub execution_times: Vec<Duration>,
    }

    impl BenchmarkResult {
        /// Calculate the standard deviation of execution times
        pub fn std_deviation(&self) -> Duration {
            if self.execution_times.is_empty() {
                return Duration::ZERO;
            }

            let avg_secs = self.average_time.as_secs_f64();
            let variance: f64 = self
                .execution_times
                .iter()
                .map(|d| (d.as_secs_f64() - avg_secs).powi(2))
                .sum::<f64>()
                / self.execution_times.len() as f64;

            Duration::from_secs_f64(variance.sqrt())
        }

        /// Calculate percentile execution time
        pub fn percentile(&self, percentile: f64) -> Duration {
            if self.execution_times.is_empty() {
                return Duration::ZERO;
            }

            let mut sorted_times = self.execution_times.clone();
            sorted_times.sort();

            let index = ((sorted_times.len() - 1) as f64 * percentile / 100.0) as usize;
            sorted_times[index.min(sorted_times.len() - 1)]
        }

        /// Print benchmark summary
        pub fn print_summary(&self) {
            println!("\n=== Benchmark Results for {} ===", self.test_name);
            println!("Iterations: {}", self.iterations);
            println!("Total time: {:?}", self.total_time);
            println!("Average time: {:?}", self.average_time);
            println!("Min time: {:?}", self.min_time);
            println!("Max time: {:?}", self.max_time);
            println!("Std deviation: {:?}", self.std_deviation());
            println!("95th percentile: {:?}", self.percentile(95.0));
            println!("99th percentile: {:?}", self.percentile(99.0));
            println!("=======================================\n");
        }
    }
}

/// Test grouping and batching utilities
pub mod grouping {
    use super::*;
    use tokio::task::JoinSet;

    /// Group of related tests that can be executed together
    pub struct TestGroup {
        pub name: String,
        pub tests: Vec<TestDescriptor>,
        pub parallel_execution: bool,
        pub max_concurrency: Option<usize>,
    }

    /// Test descriptor for group execution
    pub struct TestDescriptor {
        pub name: String,
        pub category: TestCategory,
        pub timeout_override: Option<Duration>,
        pub complexity_hints: TestComplexityHints,
    }

    impl TestGroup {
        /// Create a new test group
        pub fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                tests: Vec::new(),
                parallel_execution: true,
                max_concurrency: None,
            }
        }

        /// Add a test to the group
        pub fn add_test(mut self, descriptor: TestDescriptor) -> Self {
            self.tests.push(descriptor);
            self
        }

        /// Set parallel execution
        pub fn parallel(mut self, parallel: bool) -> Self {
            self.parallel_execution = parallel;
            self
        }

        /// Set maximum concurrency
        pub fn max_concurrency(mut self, max: usize) -> Self {
            self.max_concurrency = Some(max);
            self
        }

        /// Execute all tests in the group
        pub async fn execute<F>(self, test_fn: F) -> Result<Vec<TestExecutionResult>>
        where
            F: Fn(&str) -> Result<()> + Send + Sync + Clone + 'static,
        {
            if self.parallel_execution {
                self.execute_parallel(test_fn).await
            } else {
                self.execute_sequential(test_fn).await
            }
        }

        /// Execute tests in parallel
        async fn execute_parallel<F>(self, test_fn: F) -> Result<Vec<TestExecutionResult>>
        where
            F: Fn(&str) -> Result<()> + Send + Sync + Clone + 'static,
        {
            let semaphore = self
                .max_concurrency
                .map(|max_conc| Arc::new(tokio::sync::Semaphore::new(max_conc)));

            let mut join_set = JoinSet::new();

            for test_desc in self.tests {
                let test_fn_clone = test_fn.clone();
                let semaphore_clone = semaphore.clone();

                join_set.spawn(async move {
                    let _permit = if let Some(ref sem) = semaphore_clone {
                        Some(sem.acquire().await.expect("semaphore should not be closed"))
                    } else {
                        None
                    };

                    let context = TestExecutionContext {
                        test_name: test_desc.name.clone(),
                        category: test_desc.category,
                        expected_duration: None,
                        complexity_hints: test_desc.complexity_hints,
                        environment: get_test_environment(),
                        timeout_override: test_desc.timeout_override,
                    };

                    let framework = get_framework().await?;
                    framework
                        .execute_test(context, |_progress| async move {
                            test_fn_clone(&test_desc.name)
                        })
                        .await
                });
            }

            let mut results = Vec::new();
            while let Some(result) = join_set.join_next().await {
                results.push(result??);
            }

            Ok(results)
        }

        /// Execute tests sequentially
        async fn execute_sequential<F>(self, test_fn: F) -> Result<Vec<TestExecutionResult>>
        where
            F: Fn(&str) -> Result<()> + Send + Sync + Clone + 'static,
        {
            let mut results = Vec::new();

            for test_desc in self.tests {
                let context = TestExecutionContext {
                    test_name: test_desc.name.clone(),
                    category: test_desc.category,
                    expected_duration: None,
                    complexity_hints: test_desc.complexity_hints,
                    environment: get_test_environment(),
                    timeout_override: test_desc.timeout_override,
                };

                let framework = get_framework().await?;
                let test_fn_clone = test_fn.clone();
                let result = framework
                    .execute_test(context, |_progress| async move {
                        test_fn_clone(&test_desc.name)
                    })
                    .await?;

                results.push(result);
            }

            Ok(results)
        }
    }
}

/// Configuration utilities for different environments
pub mod config {
    use super::*;
    use crate::test_timeout_optimization::{EnvironmentConfig, TestTimeoutConfig};

    /// Create configuration for CI environment
    pub fn ci_config() -> TestTimeoutConfig {
        let mut config = TestTimeoutConfig::default();

        // CI environments typically need longer timeouts
        config.base_timeouts.unit_tests = Duration::from_secs(10);
        config.base_timeouts.integration_tests = Duration::from_secs(60);
        config.base_timeouts.e2e_tests = Duration::from_secs(300);
        config.base_timeouts.stress_tests = Duration::from_secs(600);

        // Enable all optimizations
        config.adaptive.enabled = true;
        config.early_termination.enabled = true;
        config.monitoring.enabled = true;

        // CI-specific environment config
        config.environment_overrides.insert(
            "ci".to_string(),
            EnvironmentConfig {
                timeout_multiplier: 1.5,
                disabled_optimizations: vec![], // Enable all optimizations in CI
                timeout_overrides: HashMap::new(),
            },
        );

        config
    }

    /// Create configuration for local development
    pub fn dev_config() -> TestTimeoutConfig {
        let mut config = TestTimeoutConfig::default();

        // Shorter timeouts for faster feedback
        config.base_timeouts.unit_tests = Duration::from_secs(3);
        config.base_timeouts.integration_tests = Duration::from_secs(15);
        config.base_timeouts.e2e_tests = Duration::from_secs(60);

        // More aggressive optimizations
        config.adaptive.learning_rate = 0.2;
        config.early_termination.min_progress_rate = 0.05;

        config
    }

    /// Create configuration for performance testing
    pub fn performance_config() -> TestTimeoutConfig {
        let mut config = TestTimeoutConfig::default();

        // Longer timeouts for performance tests
        config.base_timeouts.stress_tests = Duration::from_secs(1200); // 20 minutes
        config.base_timeouts.long_running_tests = Duration::from_secs(3600); // 1 hour

        // Disable early termination for accurate performance measurement
        config.early_termination.enabled = false;

        // Enhanced monitoring
        config.monitoring.collection_interval = Duration::from_millis(50);
        config.monitoring.tracked_percentiles = vec![50.0, 90.0, 95.0, 99.0, 99.9];

        config
    }
}
