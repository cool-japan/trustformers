//! Utilities for generating test reports

use anyhow::Result;
use std::io::Write;

use super::types::{NumericalParityResults, PerformanceResults};

/// Generate a comprehensive test report
pub fn generate_test_report(
    model_name: &str,
    numerical_results: &NumericalParityResults,
    performance_results: Option<&PerformanceResults>,
) -> Result<String> {
    let mut report = String::new();

    // Header
    report.push_str(&format!("# Test Report for {}\n\n", model_name));
    report.push_str(&format!(
        "Generated at: {}\n\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    ));

    // Numerical Parity Results
    report.push_str("## Numerical Parity Tests\n\n");
    report.push_str(&format!(
        "**Overall Result**: {}\n",
        if numerical_results.all_passed { "✅ PASSED" } else { "❌ FAILED" }
    ));
    report.push_str(&format!(
        "**Pass Rate**: {:.1}%\n",
        numerical_results.statistics.pass_rate
    ));
    report.push_str(&format!(
        "**Total Tests**: {}\n",
        numerical_results.statistics.total_tests
    ));
    report.push_str(&format!(
        "**Execution Time**: {:.2}s\n\n",
        numerical_results.timing.total_time.as_secs_f64()
    ));

    // Individual test results
    report.push_str("### Individual Test Results\n\n");
    for test_result in &numerical_results.test_results {
        let status = if test_result.passed { "✅" } else { "❌" };
        report.push_str(&format!(
            "- {} **{}** ({:.2}ms)\n",
            status,
            test_result.name,
            test_result.execution_time.as_millis()
        ));
        if let Some(error) = &test_result.error_message {
            report.push_str(&format!("  - Error: {}\n", error));
        }
        if let Some(diffs) = &test_result.numerical_differences {
            report.push_str(&format!(
                "  - Max diff: {:.2e}, Mean diff: {:.2e}, Within tolerance: {:.1}%\n",
                diffs.max_abs_diff, diffs.mean_abs_diff, diffs.within_tolerance_percent
            ));
        }
    }

    // Performance Results
    if let Some(perf) = performance_results {
        report.push_str("\n## Performance Results\n\n");
        report.push_str(&format!(
            "**Total Inference Time**: {:.2}ms\n",
            perf.overall_performance.total_inference_time.as_millis()
        ));
        report.push_str(&format!(
            "**Tokens/Second**: {:.1}\n",
            perf.overall_performance.tokens_per_second
        ));
        report.push_str(&format!(
            "**Peak Memory**: {:.1} MB\n",
            perf.overall_performance.peak_memory_mb
        ));
        report.push_str(&format!(
            "**Memory Efficiency**: {:.1}%\n\n",
            perf.memory_analysis.efficiency_score
        ));

        // Layer performance breakdown
        report.push_str("### Layer Performance\n\n");
        for layer in &perf.layer_performance {
            report.push_str(&format!(
                "- **{}** ({}): {:.2}ms, {:.1} MB\n",
                layer.layer_name,
                layer.layer_type,
                layer.forward_time.as_millis(),
                layer.memory_usage_mb
            ));
        }
    }

    Ok(report)
}

/// Save test report to file
pub fn save_report_to_file(report: &str, filename: &str) -> Result<()> {
    let mut file = std::fs::File::create(filename)?;
    file.write_all(report.as_bytes())?;
    Ok(())
}
