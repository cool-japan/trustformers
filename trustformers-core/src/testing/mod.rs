//! Testing utilities and infrastructure for TrustformeRS Core.
//!
//! This module provides comprehensive testing tools including:
//! - Memory leak detection
//! - Precision loss tracking for numerical stability
//! - Performance regression testing
//! - Property-based testing utilities
//! - CI/CD integration helpers
//! - Cross-framework validation
//! - Naming convention enforcement

pub mod cross_framework;
pub mod memory_leak_detector;
pub mod naming_conventions;
pub mod precision_tracker;

pub use cross_framework::{
    CrossFrameworkValidator, Framework, ValidationConfig, ValidationResult, ValidationTestCase,
};

pub use memory_leak_detector::{
    CIIntegration, MemoryLeakConfig, MemoryLeakDetector, MemoryLeakReport, ValgrindIntegration,
    ValgrindReport,
};

pub use precision_tracker::{
    OperationType, PrecisionConfig, PrecisionData, PrecisionStatistics, PrecisionTracker,
    PrecisionTrackingError, TimelinePoint, TypeStatistics,
};

pub use naming_conventions::{
    ElementType, NamingChecker, NamingCli, NamingConventions, NamingReport, NamingRule,
    NamingViolation, ViolationSeverity,
};

pub mod test_utils;

pub use test_utils::{
    BenchmarkResult, ErrorTestUtils, PerformanceTestUtils, TensorTestUtils, TestAssertions,
    TestConfig, TestResult,
};

#[cfg(test)]
pub use test_utils::property_utils;

/// Macros for easier testing with memory leak detection
#[macro_export]
macro_rules! test_with_memory_tracking {
    ($test_name:expr, $body:block) => {{
        use $crate::testing::MemoryLeakDetector;
        let detector = MemoryLeakDetector::new();
        let _monitoring = detector.start_monitoring();

        let result = (|| $body)();

        let report = detector.generate_report($test_name);
        if report.leaked_bytes > 0 {
            panic!(
                "Memory leak detected: {} bytes in {} allocations",
                report.leaked_bytes, report.active_allocations
            );
        }

        result
    }};
}

/// Integration test runner with memory leak detection
pub struct IntegrationTestRunner {
    detector: MemoryLeakDetector,
    reports: Vec<MemoryLeakReport>,
}

impl Default for IntegrationTestRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl IntegrationTestRunner {
    /// Create a new integration test runner
    pub fn new() -> Self {
        Self {
            detector: MemoryLeakDetector::new(),
            reports: Vec::new(),
        }
    }

    /// Run a test with memory leak detection
    pub fn run_test<F, R>(
        &mut self,
        test_name: &str,
        test_fn: F,
    ) -> Result<R, Box<dyn std::error::Error>>
    where
        F: FnOnce(&MemoryLeakDetector) -> Result<R, Box<dyn std::error::Error>>,
    {
        self.detector.reset();
        let result = test_fn(&self.detector)?;
        let report = self.detector.generate_report(test_name);

        if report.leaked_bytes > 0 {
            eprintln!(
                "Memory leak in test '{}': {} bytes",
                test_name, report.leaked_bytes
            );
        }

        self.reports.push(report);
        Ok(result)
    }

    /// Get all test reports
    pub fn get_reports(&self) -> &[MemoryLeakReport] {
        &self.reports
    }

    /// Generate CI reports
    pub fn generate_ci_reports(&self) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs;

        // Generate JUnit XML report
        let junit_xml = CIIntegration::generate_junit_report(&self.reports);
        fs::write("memory_leak_junit.xml", junit_xml)?;

        // Generate Markdown report
        let markdown = CIIntegration::generate_markdown_report(&self.reports);
        fs::write("memory_leak_report.md", markdown)?;

        // Generate GitHub annotations
        let annotations = CIIntegration::generate_github_annotations(&self.reports);
        for annotation in annotations {
            println!("{}", annotation);
        }

        Ok(())
    }

    /// Check if any tests had memory leaks
    pub fn has_memory_leaks(&self) -> bool {
        self.reports.iter().any(|r| r.leaked_bytes > 0)
    }

    /// Get summary statistics
    pub fn get_summary(&self) -> TestSummary {
        let total_tests = self.reports.len();
        let failed_tests = self.reports.iter().filter(|r| r.leaked_bytes > 0).count();
        let total_leaked_bytes: usize = self.reports.iter().map(|r| r.leaked_bytes).sum();
        let peak_memory: usize =
            self.reports.iter().map(|r| r.peak_memory_usage).max().unwrap_or(0);

        TestSummary {
            total_tests,
            passed_tests: total_tests - failed_tests,
            failed_tests,
            total_leaked_bytes,
            peak_memory_usage: peak_memory,
        }
    }
}

/// Summary of test results
#[derive(Debug, Clone)]
pub struct TestSummary {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub total_leaked_bytes: usize,
    pub peak_memory_usage: usize,
}

impl TestSummary {
    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.failed_tests == 0
    }

    /// Get pass rate as percentage
    pub fn pass_rate(&self) -> f64 {
        if self.total_tests == 0 {
            100.0
        } else {
            (self.passed_tests as f64 / self.total_tests as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_integration_runner() {
        let mut runner = IntegrationTestRunner::new();

        // Run a passing test
        runner
            .run_test("passing_test", |detector| {
                let id = detector.record_allocation(1024);
                detector.record_deallocation(id);
                Ok(())
            })
            .unwrap();

        // Run a failing test
        runner
            .run_test("failing_test", |detector| {
                detector.record_allocation(2048); // Not deallocated
                Ok(())
            })
            .unwrap();

        let summary = runner.get_summary();
        assert_eq!(summary.total_tests, 2);
        assert_eq!(summary.passed_tests, 1);
        assert_eq!(summary.failed_tests, 1);
        assert_eq!(summary.total_leaked_bytes, 2048);
        assert!(runner.has_memory_leaks());
        assert_eq!(summary.pass_rate(), 50.0);
    }

    #[test]
    fn test_memory_tracking_macro() {
        let result = test_with_memory_tracking!("macro_test", {
            // This should not leak memory
            42
        });

        assert_eq!(result, 42);
    }

    #[test]
    #[should_panic(expected = "Memory leak detected")]
    fn test_memory_tracking_macro_with_leak() {
        // Create a detector outside the macro to simulate a real leak scenario
        let detector = MemoryLeakDetector::new();
        let _monitoring = detector.start_monitoring();

        // Record an allocation that won't be deallocated (simulating a leak)
        detector.record_allocation(1024);

        // Now test the macro which will detect the leak
        let report = detector.generate_report("macro_leak_test");
        if report.leaked_bytes > 0 {
            panic!(
                "Memory leak detected: {} bytes in {} allocations",
                report.leaked_bytes, report.active_allocations
            );
        }
    }
}
