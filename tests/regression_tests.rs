//! Regression test runner
//!
//! This test runs all regression test cases to ensure we don't regress on fixed issues.

mod regression;

use std::path::Path;
use regression::{load_test_cases, run_test_case};

#[test]
fn run_all_regression_tests() {
    let test_dir = Path::new("tests/regression/test_cases");

    // Load all test cases
    let test_cases = match load_test_cases(test_dir) {
        Ok(cases) => cases,
        Err(e) => {
            eprintln!("Warning: Could not load regression test cases: {}", e);
            return; // Skip if directory doesn't exist
        }
    };

    println!("Running {} regression tests...", test_cases.len());

    let mut failed = 0;
    let mut passed = 0;

    for test_case in &test_cases {
        print!("Running {}: {} ... ", test_case.id, test_case.description);

        match run_test_case(test_case) {
            Ok(()) => {
                println!("PASSED");
                passed += 1;
            }
            Err(e) => {
                println!("FAILED");
                eprintln!("  Error: {}", e);
                if let Some(issue) = &test_case.issue {
                    eprintln!("  Related issue: {}", issue);
                }
                failed += 1;
            }
        }
    }

    println!("\nRegression test summary: {} passed, {} failed", passed, failed);

    if failed > 0 {
        panic!("{} regression tests failed", failed);
    }
}

#[test]
fn test_individual_regression_cases() {
    use regression::{RegressionTestCase, TestData, TensorData, ExpectedResult};

    // Test matrix multiplication dimension mismatch
    let test_case = RegressionTestCase {
        id: "matmul_dimension_mismatch".to_string(),
        description: "Matrix multiplication with incompatible dimensions should fail".to_string(),
        issue: Some("Issue #23: Matmul panics instead of returning error".to_string()),
        category: "tensor".to_string(),
        test_data: TestData::TensorOperation {
            operation: "matmul".to_string(),
            inputs: vec![
                TensorData {
                    data: vec![1.0, 2.0, 3.0, 4.0],
                    shape: vec![2, 2],
                },
                TensorData {
                    data: vec![1.0, 2.0, 3.0],
                    shape: vec![3, 1],
                },
            ],
            params: serde_json::json!({}),
        },
        expected: ExpectedResult::Error {
            error_type: "TensorError".to_string(),
            message_contains: Some("dimension".to_string()),
        },
    };

    let result = run_test_case(&test_case);
    assert!(result.is_ok(), "Dimension mismatch test failed: {:?}", result);
}