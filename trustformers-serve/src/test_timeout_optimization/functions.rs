//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Convenience macro for creating test execution context
#[macro_export]
macro_rules! test_context {
    ($test_name:expr, $category:expr) => {
        TestExecutionContext {
            test_name: $test_name.to_string(),
            category: $category,
            expected_duration: None,
            complexity_hints: TestComplexityHints::default(),
            environment: "test".to_string(),
            timeout_override: None,
        }
    };
    ($test_name:expr, $category:expr, timeout = $timeout:expr) => {
        TestExecutionContext {
            test_name: $test_name.to_string(),
            category: $category,
            expected_duration: None,
            complexity_hints: TestComplexityHints::default(),
            environment: "test".to_string(),
            timeout_override: Some($timeout),
        }
    };
    ($test_name:expr, $category:expr, concurrency = $concurrency:expr) => {
        TestExecutionContext {
            test_name: $test_name.to_string(),
            category: $category,
            expected_duration: None,
            complexity_hints: TestComplexityHints {
                concurrency_level: Some($concurrency),
                ..Default::default()
            },
            environment: "test".to_string(),
            timeout_override: None,
        }
    };
}
