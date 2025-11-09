use scirs2_core::random::*; // SciRS2 Integration Policy - Replaces rand
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Configuration for comprehensive testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    /// Number of random test cases to generate
    pub num_random_tests: usize,
    /// Maximum input length for testing
    pub max_input_length: usize,
    /// Timeout for individual tests (in milliseconds)
    pub timeout_ms: u64,
    /// Whether to run performance benchmarks
    pub run_benchmarks: bool,
    /// Whether to run fuzzing tests
    pub run_fuzzing: bool,
    /// Whether to run regression tests
    pub run_regression: bool,
    /// Languages to test
    pub test_languages: Vec<String>,
    /// Custom test cases to include
    pub custom_test_cases: Vec<String>,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            num_random_tests: 1000,
            max_input_length: 1000,
            timeout_ms: 5000,
            run_benchmarks: true,
            run_fuzzing: true,
            run_regression: true,
            test_languages: vec![
                "en".to_string(),
                "es".to_string(),
                "fr".to_string(),
                "de".to_string(),
                "zh".to_string(),
                "ja".to_string(),
                "ru".to_string(),
            ],
            custom_test_cases: Vec::new(),
        }
    }
}

/// Test result for a single test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test case description
    pub test_case: String,
    /// Whether the test passed
    pub passed: bool,
    /// Error message if test failed
    pub error: Option<String>,
    /// Execution time
    pub execution_time: Duration,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Comprehensive test suite results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuiteResult {
    /// Total number of tests run
    pub total_tests: usize,
    /// Number of passed tests
    pub passed_tests: usize,
    /// Number of failed tests
    pub failed_tests: usize,
    /// Individual test results
    pub test_results: Vec<TestResult>,
    /// Performance benchmark results
    pub benchmark_results: Option<BenchmarkResults>,
    /// Fuzzing test results
    pub fuzzing_results: Option<FuzzingResults>,
    /// Regression test results
    pub regression_results: Option<RegressionResults>,
    /// Cross-tokenizer validation results
    pub cross_validation_results: Option<CrossValidationResults>,
}

/// Performance benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Tokens per second for encoding
    pub encode_tokens_per_second: f64,
    /// Tokens per second for decoding
    pub decode_tokens_per_second: f64,
    /// Memory usage statistics
    pub memory_usage_mb: f64,
    /// Latency percentiles
    pub latency_percentiles: HashMap<String, Duration>,
}

/// Fuzzing test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzingResults {
    /// Number of fuzzing tests run
    pub tests_run: usize,
    /// Number of crashes/panics detected
    pub crashes_detected: usize,
    /// Unique error types found
    pub error_types: HashSet<String>,
    /// Coverage metrics
    pub coverage_metrics: HashMap<String, f64>,
}

/// Regression test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResults {
    /// Number of regression tests run
    pub tests_run: usize,
    /// Number of regressions detected
    pub regressions_detected: usize,
    /// Details of detected regressions
    pub regression_details: Vec<RegressionDetail>,
}

/// Details of a detected regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetail {
    /// Test case that regressed
    pub test_case: String,
    /// Expected result
    pub expected: String,
    /// Actual result
    pub actual: String,
    /// Difference description
    pub difference: String,
}

/// Regression test case definition
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegressionTestCase {
    /// Test case name
    name: String,
    /// Input text for tokenization
    input: String,
    /// Expected number of tokens (if known)
    expected_token_count: Option<usize>,
    /// Whether tokenization should succeed
    expected_success: bool,
    /// Maximum allowed execution time
    max_execution_time: Duration,
}

/// Cross-tokenizer validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResults {
    /// Tokenizers compared
    pub tokenizers_compared: Vec<String>,
    /// Consistency score (0.0 to 1.0)
    pub consistency_score: f64,
    /// Number of inconsistencies found
    pub inconsistencies_found: usize,
    /// Details of inconsistencies
    pub inconsistency_details: Vec<InconsistencyDetail>,
}

/// Details of a tokenization inconsistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InconsistencyDetail {
    /// Input text that caused inconsistency
    pub input: String,
    /// Results from each tokenizer
    pub tokenizer_results: HashMap<String, Vec<String>>,
    /// Severity of inconsistency
    pub severity: InconsistencySeverity,
}

/// Severity levels for inconsistencies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InconsistencySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Test case generator for fuzzing and random testing
pub struct TestCaseGenerator {
    rng: StdRng,
    config: TestConfig,
}

#[allow(deprecated)]
impl TestCaseGenerator {
    /// Create a new test case generator
    pub fn new(config: TestConfig, seed: Option<u64>) -> Self {
        let rng = if let Some(seed) = seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut thread_rng().rng_mut())
        };

        Self { rng, config }
    }

    /// Generate random text for testing
    pub fn generate_random_text(&mut self) -> String {
        let length = self.rng.gen_range(1..=self.config.max_input_length);
        let mut text = String::new();

        for _ in 0..length {
            let char_type = self.rng.gen_range(0..10);
            let ch = match char_type {
                0..=5 => self.rng.gen_range(b'a'..=b'z') as char, // Lowercase letters
                6 => self.rng.gen_range(b'A'..=b'Z') as char,     // Uppercase letters
                7 => self.rng.gen_range(b'0'..=b'9') as char,     // Digits
                8 => ' ',                                         // Space
                _ => self.generate_special_char(),                // Special characters
            };
            text.push(ch);
        }

        text
    }

    /// Generate Unicode text for testing
    pub fn generate_unicode_text(&mut self) -> String {
        let length = self.rng.gen_range(1..=self.config.max_input_length / 2);
        let mut text = String::new();

        for _ in 0..length {
            let char_type = self.rng.gen_range(0..10);
            let ch = match char_type {
                0..=3 => self.rng.gen_range('a'..='z'),
                4 => self.rng.gen_range('À'..='ÿ'), // Latin extended
                5 => self.rng.gen_range('Α'..='ω'), // Greek
                6 => self.rng.gen_range('А'..='я'), // Cyrillic
                7 => self.rng.gen_range('一'..='龯'), // CJK
                8 => self.rng.gen_range('ا'..='ي'), // Arabic
                _ => self.rng.gen_range('😀'..='🙏'), // Emoji
            };
            text.push(ch);
        }

        text
    }

    /// Generate edge case text
    pub fn generate_edge_case_text(&mut self) -> String {
        let long_token = "a".repeat(1000);
        let edge_cases = [
            "",                            // Empty string
            " ",                           // Single space
            "\n\t\r",                      // Whitespace only
            &long_token,                   // Very long token
            "123456789",                   // Numbers only
            "!@#$%^&*()",                  // Special characters only
            "\u{200B}\u{200C}\u{200D}",    // Zero-width characters
            "Test\u{0000}null",            // Null character
            "🚀🌟💫⭐",                    // Emoji sequence
            "a\u{0301}e\u{0301}i\u{0301}", // Combining characters
        ];

        edge_cases[self.rng.gen_range(0..edge_cases.len())].to_string()
    }

    /// Generate malformed input for fuzzing
    pub fn generate_malformed_input(&mut self) -> Vec<u8> {
        let length = self.rng.gen_range(1..=100);
        let mut bytes = Vec::new();

        for _ in 0..length {
            bytes.push(self.rng.gen());
        }

        bytes
    }

    fn generate_special_char(&mut self) -> char {
        let special_chars = [
            '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '=',
        ];
        special_chars[self.rng.gen_range(0..special_chars.len())]
    }
}

/// Comprehensive test runner
pub struct TestRunner {
    config: TestConfig,
    generator: TestCaseGenerator,
}

impl TestRunner {
    /// Create a new test runner
    pub fn new(config: TestConfig) -> Self {
        let generator = TestCaseGenerator::new(config.clone(), None);
        Self { config, generator }
    }

    /// Run the complete test suite
    pub fn run_complete_suite<T: Tokenizer + Clone>(
        &mut self,
        tokenizer: &T,
        test_name: &str,
    ) -> Result<TestSuiteResult> {
        let mut results = Vec::new();
        let mut total_tests = 0;
        let mut passed_tests = 0;

        // Basic functionality tests
        let basic_results = self.run_basic_tests(tokenizer, test_name)?;
        total_tests += basic_results.len();
        passed_tests += basic_results.iter().filter(|r| r.passed).count();
        results.extend(basic_results);

        // Random tests
        let random_results = self.run_random_tests(tokenizer)?;
        total_tests += random_results.len();
        passed_tests += random_results.iter().filter(|r| r.passed).count();
        results.extend(random_results);

        // Edge case tests
        let edge_results = self.run_edge_case_tests(tokenizer)?;
        total_tests += edge_results.len();
        passed_tests += edge_results.iter().filter(|r| r.passed).count();
        results.extend(edge_results);

        // Custom test cases
        if !self.config.custom_test_cases.is_empty() {
            let custom_results = self.run_custom_tests(tokenizer)?;
            total_tests += custom_results.len();
            passed_tests += custom_results.iter().filter(|r| r.passed).count();
            results.extend(custom_results);
        }

        let failed_tests = total_tests - passed_tests;

        // Optional advanced testing
        let benchmark_results = if self.config.run_benchmarks {
            Some(self.run_benchmarks(tokenizer)?)
        } else {
            None
        };

        let fuzzing_results = if self.config.run_fuzzing {
            Some(self.run_fuzzing_tests(tokenizer)?)
        } else {
            None
        };

        let regression_results = if self.config.run_regression {
            Some(self.run_regression_tests(tokenizer)?)
        } else {
            None
        };

        Ok(TestSuiteResult {
            total_tests,
            passed_tests,
            failed_tests,
            test_results: results,
            benchmark_results,
            fuzzing_results,
            regression_results,
            cross_validation_results: None, // Filled by cross-validation runner
        })
    }

    /// Run basic functionality tests
    fn run_basic_tests<T: Tokenizer>(
        &mut self,
        tokenizer: &T,
        test_name: &str,
    ) -> Result<Vec<TestResult>> {
        let mut results = Vec::new();

        // Test basic encode/decode cycle
        let test_cases = vec![
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "123456789",
            "Special chars: !@#$%^&*()",
            "",
        ];

        for (i, text) in test_cases.into_iter().enumerate() {
            let start = Instant::now();
            let test_case = format!("{}_basic_{}", test_name, i);

            match self.test_encode_decode_cycle(tokenizer, text) {
                Ok(metrics) => {
                    results.push(TestResult {
                        test_case,
                        passed: true,
                        error: None,
                        execution_time: start.elapsed(),
                        metrics,
                    });
                },
                Err(e) => {
                    results.push(TestResult {
                        test_case,
                        passed: false,
                        error: Some(e.to_string()),
                        execution_time: start.elapsed(),
                        metrics: HashMap::new(),
                    });
                },
            }
        }

        Ok(results)
    }

    /// Run random tests
    fn run_random_tests<T: Tokenizer>(&mut self, tokenizer: &T) -> Result<Vec<TestResult>> {
        let mut results = Vec::new();

        for i in 0..self.config.num_random_tests {
            let text = self.generator.generate_random_text();
            let start = Instant::now();
            let test_case = format!("random_{}", i);

            match self.test_encode_decode_cycle(tokenizer, &text) {
                Ok(metrics) => {
                    results.push(TestResult {
                        test_case,
                        passed: true,
                        error: None,
                        execution_time: start.elapsed(),
                        metrics,
                    });
                },
                Err(e) => {
                    results.push(TestResult {
                        test_case,
                        passed: false,
                        error: Some(e.to_string()),
                        execution_time: start.elapsed(),
                        metrics: HashMap::new(),
                    });
                },
            }
        }

        Ok(results)
    }

    /// Run edge case tests
    fn run_edge_case_tests<T: Tokenizer>(&mut self, tokenizer: &T) -> Result<Vec<TestResult>> {
        let mut results = Vec::new();

        for i in 0..100 {
            let text = self.generator.generate_edge_case_text();
            let start = Instant::now();
            let test_case = format!("edge_case_{}", i);

            match self.test_encode_decode_cycle(tokenizer, &text) {
                Ok(metrics) => {
                    results.push(TestResult {
                        test_case,
                        passed: true,
                        error: None,
                        execution_time: start.elapsed(),
                        metrics,
                    });
                },
                Err(e) => {
                    results.push(TestResult {
                        test_case,
                        passed: false,
                        error: Some(e.to_string()),
                        execution_time: start.elapsed(),
                        metrics: HashMap::new(),
                    });
                },
            }
        }

        Ok(results)
    }

    /// Run custom test cases
    fn run_custom_tests<T: Tokenizer>(&mut self, tokenizer: &T) -> Result<Vec<TestResult>> {
        let mut results = Vec::new();

        for (i, text) in self.config.custom_test_cases.iter().enumerate() {
            let start = Instant::now();
            let test_case = format!("custom_{}", i);

            match self.test_encode_decode_cycle(tokenizer, text) {
                Ok(metrics) => {
                    results.push(TestResult {
                        test_case,
                        passed: true,
                        error: None,
                        execution_time: start.elapsed(),
                        metrics,
                    });
                },
                Err(e) => {
                    results.push(TestResult {
                        test_case,
                        passed: false,
                        error: Some(e.to_string()),
                        execution_time: start.elapsed(),
                        metrics: HashMap::new(),
                    });
                },
            }
        }

        Ok(results)
    }

    /// Test encode/decode cycle for correctness
    fn test_encode_decode_cycle<T: Tokenizer>(
        &self,
        tokenizer: &T,
        text: &str,
    ) -> Result<HashMap<String, f64>> {
        let mut metrics = HashMap::new();

        // Encode
        let encoded = tokenizer.encode(text)?;
        metrics.insert("num_tokens".to_string(), encoded.input_ids.len() as f64);
        metrics.insert("input_length".to_string(), text.chars().count() as f64);

        // Decode
        let decoded = tokenizer.decode(&encoded.input_ids)?;

        // Verify vocabulary consistency
        for &token_id in &encoded.input_ids {
            if let Some(token) = tokenizer.id_to_token(token_id) {
                if tokenizer.token_to_id(&token).is_none() {
                    return Err(TrustformersError::runtime_error(format!(
                        "Token '{}' not found in vocabulary",
                        token
                    )));
                }
            }
        }

        // Calculate compression ratio
        if !text.is_empty() {
            let compression_ratio = encoded.input_ids.len() as f64 / text.chars().count() as f64;
            metrics.insert("compression_ratio".to_string(), compression_ratio);
        }

        // Verify round-trip if possible (not all tokenizers preserve exact text)
        if decoded.trim() != text.trim() {
            metrics.insert("exact_match".to_string(), 0.0);
        } else {
            metrics.insert("exact_match".to_string(), 1.0);
        }

        Ok(metrics)
    }

    /// Run performance benchmarks
    fn run_benchmarks<T: Tokenizer>(&mut self, tokenizer: &T) -> Result<BenchmarkResults> {
        let test_texts: Vec<String> =
            (0..1000).map(|_| self.generator.generate_random_text()).collect();

        // Benchmark encoding
        let start = Instant::now();
        let mut total_tokens = 0;

        for text in &test_texts {
            let encoded = tokenizer.encode(text)?;
            total_tokens += encoded.input_ids.len();
        }

        let encoding_time = start.elapsed();
        let encode_tokens_per_second = total_tokens as f64 / encoding_time.as_secs_f64();

        // Benchmark decoding
        let token_sequences: Vec<Vec<u32>> = test_texts
            .iter()
            .map(|text| tokenizer.encode(text).unwrap().input_ids)
            .collect();

        let start = Instant::now();
        for tokens in &token_sequences {
            let _ = tokenizer.decode(tokens)?;
        }
        let decoding_time = start.elapsed();
        let decode_tokens_per_second = total_tokens as f64 / decoding_time.as_secs_f64();

        // Memory usage (simplified)
        let vocab = tokenizer.get_vocab();
        let memory_usage_mb = (vocab.len() * 100) as f64 / 1024.0 / 1024.0; // Rough estimate

        // Latency percentiles (simplified)
        let mut latency_percentiles = HashMap::new();
        latency_percentiles.insert("p50".to_string(), encoding_time / test_texts.len() as u32);
        latency_percentiles.insert("p95".to_string(), encoding_time / test_texts.len() as u32);
        latency_percentiles.insert("p99".to_string(), encoding_time / test_texts.len() as u32);

        Ok(BenchmarkResults {
            encode_tokens_per_second,
            decode_tokens_per_second,
            memory_usage_mb,
            latency_percentiles,
        })
    }

    /// Run fuzzing tests
    fn run_fuzzing_tests<T: Tokenizer>(&mut self, tokenizer: &T) -> Result<FuzzingResults> {
        let mut tests_run = 0;
        let mut crashes_detected = 0;
        let mut error_types = HashSet::new();
        let mut coverage_metrics = HashMap::new();

        // Generate and test malformed inputs
        for _ in 0..1000 {
            tests_run += 1;

            // Try with malformed bytes converted to string (if possible)
            let malformed_bytes = self.generator.generate_malformed_input();
            if let Ok(malformed_string) = String::from_utf8(malformed_bytes) {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    tokenizer.encode(&malformed_string)
                })) {
                    Ok(result) => {
                        if let Err(e) = result {
                            error_types.insert(format!("{:?}", e));
                        }
                    },
                    Err(_) => {
                        crashes_detected += 1;
                    },
                }
            }

            // Test with extremely long inputs
            if tests_run % 100 == 0 {
                let very_long_text = "a".repeat(10000);
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    tokenizer.encode(&very_long_text)
                })) {
                    Ok(result) => {
                        if let Err(e) = result {
                            error_types.insert(format!("{:?}", e));
                        }
                    },
                    Err(_) => {
                        crashes_detected += 1;
                    },
                }
            }
        }

        // Calculate basic coverage metrics
        coverage_metrics.insert(
            "crash_rate".to_string(),
            crashes_detected as f64 / tests_run as f64,
        );
        coverage_metrics.insert("error_diversity".to_string(), error_types.len() as f64);

        Ok(FuzzingResults {
            tests_run,
            crashes_detected,
            error_types,
            coverage_metrics,
        })
    }

    /// Run regression tests against stored baseline data
    fn run_regression_tests<T: Tokenizer>(&mut self, tokenizer: &T) -> Result<RegressionResults> {
        let mut regression_details = Vec::new();
        let mut tests_run = 0;

        // Create regression test cases
        let test_cases = self.create_regression_test_cases();

        for test_case in &test_cases {
            tests_run += 1;

            // Run current tokenization
            let start_time = Instant::now();
            let current_result = match tokenizer.encode(&test_case.input) {
                Ok(result) => result,
                Err(e) => {
                    // If tokenization fails, that's a regression if baseline succeeded
                    if test_case.expected_success {
                        regression_details.push(RegressionDetail {
                            test_case: test_case.name.clone(),
                            expected: "Successful tokenization".to_string(),
                            actual: format!("Failed with error: {}", e),
                            difference: "Tokenization failed unexpectedly".to_string(),
                        });
                    }
                    continue;
                },
            };

            let execution_time = start_time.elapsed();

            // Compare against baseline
            if let Some(regression) =
                self.compare_with_baseline(test_case, &current_result, execution_time)
            {
                regression_details.push(regression);
            }
        }

        Ok(RegressionResults {
            tests_run,
            regressions_detected: regression_details.len(),
            regression_details,
        })
    }

    /// Create standard regression test cases
    fn create_regression_test_cases(&self) -> Vec<RegressionTestCase> {
        vec![
            RegressionTestCase {
                name: "basic_english".to_string(),
                input: "Hello world".to_string(),
                expected_token_count: Some(2),
                expected_success: true,
                max_execution_time: Duration::from_millis(100),
            },
            RegressionTestCase {
                name: "unicode_text".to_string(),
                input: "你好世界 🌍".to_string(),
                expected_token_count: None, // Variable depending on tokenizer
                expected_success: true,
                max_execution_time: Duration::from_millis(100),
            },
            RegressionTestCase {
                name: "long_sentence".to_string(),
                input: "The quick brown fox jumps over the lazy dog. This is a longer sentence to test tokenization performance and accuracy.".to_string(),
                expected_token_count: None, // Variable depending on tokenizer
                expected_success: true,
                max_execution_time: Duration::from_millis(200),
            },
            RegressionTestCase {
                name: "empty_string".to_string(),
                input: "".to_string(),
                expected_token_count: Some(0),
                expected_success: true,
                max_execution_time: Duration::from_millis(50),
            },
            RegressionTestCase {
                name: "special_characters".to_string(),
                input: "!@#$%^&*()_+-=[]{}|;':\",./<>?".to_string(),
                expected_token_count: None, // Variable depending on tokenizer
                expected_success: true,
                max_execution_time: Duration::from_millis(100),
            },
            RegressionTestCase {
                name: "mixed_languages".to_string(),
                input: "Hello こんにちは 你好 Hola".to_string(),
                expected_token_count: None, // Variable depending on tokenizer
                expected_success: true,
                max_execution_time: Duration::from_millis(150),
            },
        ]
    }

    /// Compare current results with baseline expectations
    fn compare_with_baseline(
        &self,
        test_case: &RegressionTestCase,
        current_result: &TokenizedInput,
        execution_time: Duration,
    ) -> Option<RegressionDetail> {
        let mut differences = Vec::new();

        // Check token count if expected
        if let Some(expected_count) = test_case.expected_token_count {
            let actual_count = current_result.input_ids.len();
            if actual_count != expected_count {
                differences.push(format!(
                    "Token count: expected {}, got {}",
                    expected_count, actual_count
                ));
            }
        }

        // Check execution time
        if execution_time > test_case.max_execution_time {
            differences.push(format!(
                "Execution time: expected <= {:?}, got {:?}",
                test_case.max_execution_time, execution_time
            ));
        }

        // Check for empty results when they shouldn't be
        if !test_case.input.is_empty() && current_result.input_ids.is_empty() {
            differences.push("Unexpected empty tokenization result".to_string());
        }

        // Check attention mask consistency
        if current_result.input_ids.len() != current_result.attention_mask.len() {
            differences.push(format!(
                "Attention mask length mismatch: input_ids={}, attention_mask={}",
                current_result.input_ids.len(),
                current_result.attention_mask.len()
            ));
        }

        if !differences.is_empty() {
            Some(RegressionDetail {
                test_case: test_case.name.clone(),
                expected: format!(
                    "Proper tokenization within {} ms",
                    test_case.max_execution_time.as_millis()
                ),
                actual: format!("Issues detected: {:?}", differences),
                difference: differences.join("; "),
            })
        } else {
            None
        }
    }
}

/// Cross-tokenizer validation runner
pub struct CrossValidationRunner {
    #[allow(dead_code)]
    config: TestConfig,
}

impl CrossValidationRunner {
    pub fn new(config: TestConfig) -> Self {
        Self { config }
    }

    /// Compare multiple tokenizers for consistency
    pub fn compare_tokenizers(
        &self,
        tokenizers: Vec<(&str, &dyn Tokenizer)>,
        test_cases: &[String],
    ) -> Result<CrossValidationResults> {
        let mut inconsistencies = Vec::new();
        let mut total_comparisons = 0;
        let mut consistent_comparisons = 0;

        for text in test_cases {
            total_comparisons += 1;
            let mut results = HashMap::new();

            // Get results from each tokenizer
            for (name, tokenizer) in &tokenizers {
                match tokenizer.encode(text) {
                    Ok(encoded) => {
                        let tokens: Vec<String> = encoded
                            .input_ids
                            .iter()
                            .filter_map(|&id| tokenizer.id_to_token(id))
                            .collect();
                        results.insert(name.to_string(), tokens);
                    },
                    Err(_) => {
                        // Skip this comparison if any tokenizer fails
                        continue;
                    },
                }
            }

            // Check for consistency
            if results.len() > 1 {
                let first_result = results.values().next().unwrap();
                let is_consistent = results.values().all(|tokens| tokens == first_result);

                if is_consistent {
                    consistent_comparisons += 1;
                } else {
                    let severity = self.determine_inconsistency_severity(&results);
                    inconsistencies.push(InconsistencyDetail {
                        input: text.clone(),
                        tokenizer_results: results,
                        severity,
                    });
                }
            }
        }

        let consistency_score = if total_comparisons > 0 {
            consistent_comparisons as f64 / total_comparisons as f64
        } else {
            0.0
        };

        Ok(CrossValidationResults {
            tokenizers_compared: tokenizers.iter().map(|(name, _)| name.to_string()).collect(),
            consistency_score,
            inconsistencies_found: inconsistencies.len(),
            inconsistency_details: inconsistencies,
        })
    }

    fn determine_inconsistency_severity(
        &self,
        results: &HashMap<String, Vec<String>>,
    ) -> InconsistencySeverity {
        // Simple heuristic: if token counts differ significantly, it's high severity
        let token_counts: Vec<usize> = results.values().map(|tokens| tokens.len()).collect();
        let min_count = *token_counts.iter().min().unwrap_or(&0);
        let max_count = *token_counts.iter().max().unwrap_or(&0);

        if max_count == 0 {
            InconsistencySeverity::Low
        } else {
            let ratio = min_count as f64 / max_count as f64;
            if ratio < 0.5 {
                InconsistencySeverity::High
            } else if ratio < 0.8 {
                InconsistencySeverity::Medium
            } else {
                InconsistencySeverity::Low
            }
        }
    }
}

/// Utilities for test reporting and analysis
pub struct TestReportUtils;

impl TestReportUtils {
    /// Generate a comprehensive test report
    pub fn generate_report(result: &TestSuiteResult) -> String {
        let mut report = String::new();

        report.push_str("=== COMPREHENSIVE TEST REPORT ===\n\n");

        // Summary
        report.push_str(&format!("Total Tests: {}\n", result.total_tests));
        report.push_str(&format!(
            "Passed: {} ({:.1}%)\n",
            result.passed_tests,
            (result.passed_tests as f64 / result.total_tests as f64) * 100.0
        ));
        report.push_str(&format!(
            "Failed: {} ({:.1}%)\n\n",
            result.failed_tests,
            (result.failed_tests as f64 / result.total_tests as f64) * 100.0
        ));

        // Failed tests
        if result.failed_tests > 0 {
            report.push_str("FAILED TESTS:\n");
            for test in &result.test_results {
                if !test.passed {
                    report.push_str(&format!(
                        "  {} - {}\n",
                        test.test_case,
                        test.error.as_ref().unwrap_or(&"Unknown error".to_string())
                    ));
                }
            }
            report.push('\n');
        }

        // Benchmark results
        if let Some(ref benchmarks) = result.benchmark_results {
            report.push_str("PERFORMANCE BENCHMARKS:\n");
            report.push_str(&format!(
                "  Encoding: {:.0} tokens/sec\n",
                benchmarks.encode_tokens_per_second
            ));
            report.push_str(&format!(
                "  Decoding: {:.0} tokens/sec\n",
                benchmarks.decode_tokens_per_second
            ));
            report.push_str(&format!(
                "  Memory Usage: {:.1} MB\n\n",
                benchmarks.memory_usage_mb
            ));
        }

        // Fuzzing results
        if let Some(ref fuzzing) = result.fuzzing_results {
            report.push_str("FUZZING RESULTS:\n");
            report.push_str(&format!("  Tests Run: {}\n", fuzzing.tests_run));
            report.push_str(&format!(
                "  Crashes Detected: {}\n",
                fuzzing.crashes_detected
            ));
            report.push_str(&format!(
                "  Unique Error Types: {}\n\n",
                fuzzing.error_types.len()
            ));
        }

        // Cross-validation results
        if let Some(ref cross_val) = result.cross_validation_results {
            report.push_str("CROSS-VALIDATION RESULTS:\n");
            report.push_str(&format!(
                "  Consistency Score: {:.3}\n",
                cross_val.consistency_score
            ));
            report.push_str(&format!(
                "  Inconsistencies Found: {}\n\n",
                cross_val.inconsistencies_found
            ));
        }

        report
    }

    /// Analyze test metrics
    pub fn analyze_metrics(results: &[TestResult]) -> HashMap<String, f64> {
        let mut analysis = HashMap::new();

        let mut total_time = Duration::new(0, 0);
        let mut total_tokens = 0.0;
        let mut compression_ratios = Vec::new();

        for result in results {
            total_time += result.execution_time;

            if let Some(&tokens) = result.metrics.get("num_tokens") {
                total_tokens += tokens;
            }

            if let Some(&ratio) = result.metrics.get("compression_ratio") {
                compression_ratios.push(ratio);
            }
        }

        analysis.insert(
            "avg_execution_time_ms".to_string(),
            total_time.as_millis() as f64 / results.len() as f64,
        );
        analysis.insert(
            "avg_tokens_per_test".to_string(),
            total_tokens / results.len() as f64,
        );

        if !compression_ratios.is_empty() {
            let avg_compression =
                compression_ratios.iter().sum::<f64>() / compression_ratios.len() as f64;
            analysis.insert("avg_compression_ratio".to_string(), avg_compression);
        }

        analysis
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Mock tokenizer for testing
    #[derive(Clone)]
    struct MockTokenizer {
        vocab: HashMap<String, u32>,
    }

    impl MockTokenizer {
        fn new() -> Self {
            let mut vocab = HashMap::new();
            vocab.insert("hello".to_string(), 1);
            vocab.insert("world".to_string(), 2);
            vocab.insert("test".to_string(), 3);
            vocab.insert("!".to_string(), 4);

            Self { vocab }
        }
    }

    impl Tokenizer for MockTokenizer {
        fn encode(&self, text: &str) -> Result<TokenizedInput> {
            let tokens: Vec<&str> = text.split_whitespace().collect();
            let mut input_ids = Vec::new();
            let mut token_strings = Vec::new();

            for token in tokens {
                if let Some(&id) = self.vocab.get(token) {
                    input_ids.push(id);
                    token_strings.push(token.to_string());
                }
            }

            Ok(TokenizedInput {
                input_ids,
                attention_mask: vec![1; token_strings.len()],
                token_type_ids: None,
                special_tokens_mask: None,
                offset_mapping: None,
                overflowing_tokens: None,
            })
        }

        fn decode(&self, token_ids: &[u32]) -> Result<String> {
            let tokens: Result<Vec<String>> = token_ids
                .iter()
                .map(|&id| {
                    self.vocab.iter().find(|(_, &v)| v == id).map(|(k, _)| k.clone()).ok_or_else(
                        || TrustformersError::other(format!("Unknown token ID: {}", id)),
                    )
                })
                .collect();

            Ok(tokens?.join(" "))
        }

        fn get_vocab(&self) -> HashMap<String, u32> {
            self.vocab.clone()
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.vocab.get(token).copied()
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            self.vocab.iter().find(|(_, &v)| v == id).map(|(k, _)| k.clone())
        }

        fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<TokenizedInput> {
            let combined = format!("{} {}", text_a, text_b);
            self.encode(&combined)
        }

        fn vocab_size(&self) -> usize {
            self.vocab.len()
        }
    }

    #[test]
    fn test_config_default() {
        let config = TestConfig::default();
        assert_eq!(config.num_random_tests, 1000);
        assert!(config.run_benchmarks);
        assert!(config.run_fuzzing);
    }

    #[test]
    fn test_case_generator() {
        let config = TestConfig::default();
        let mut generator = TestCaseGenerator::new(config, Some(42));

        let random_text = generator.generate_random_text();
        assert!(!random_text.is_empty());

        let unicode_text = generator.generate_unicode_text();
        assert!(!unicode_text.is_empty());

        let edge_case = generator.generate_edge_case_text();
        // Edge cases can be empty
        assert!(edge_case.len() <= 1000);
    }

    #[test]
    fn test_basic_functionality() {
        let config = TestConfig::default();
        let mut runner = TestRunner::new(config);
        let tokenizer = MockTokenizer::new();

        let results = runner.run_basic_tests(&tokenizer, "test").unwrap();
        assert!(!results.is_empty());

        // At least some tests should pass
        let passed_count = results.iter().filter(|r| r.passed).count();
        assert!(passed_count > 0);
    }

    #[test]
    fn test_encode_decode_cycle() {
        let config = TestConfig::default();
        let runner = TestRunner::new(config);
        let tokenizer = MockTokenizer::new();

        let metrics = runner.test_encode_decode_cycle(&tokenizer, "hello world").unwrap();
        assert!(metrics.contains_key("num_tokens"));
        assert!(metrics.contains_key("input_length"));
    }

    #[test]
    fn test_cross_validation() {
        let config = TestConfig::default();
        let validator = CrossValidationRunner::new(config);

        let tokenizer1 = MockTokenizer::new();
        let tokenizer2 = MockTokenizer::new();

        let tokenizers: Vec<(&str, &dyn Tokenizer)> =
            vec![("mock1", &tokenizer1), ("mock2", &tokenizer2)];

        let test_cases = vec!["hello world".to_string(), "test".to_string()];
        let results = validator.compare_tokenizers(tokenizers, &test_cases).unwrap();

        assert_eq!(results.tokenizers_compared.len(), 2);
        assert!(results.consistency_score >= 0.0 && results.consistency_score <= 1.0);
    }

    #[test]
    fn test_report_generation() {
        let test_result = TestSuiteResult {
            total_tests: 10,
            passed_tests: 8,
            failed_tests: 2,
            test_results: vec![
                TestResult {
                    test_case: "test1".to_string(),
                    passed: true,
                    error: None,
                    execution_time: Duration::from_millis(10),
                    metrics: HashMap::new(),
                },
                TestResult {
                    test_case: "test2".to_string(),
                    passed: false,
                    error: Some("Test failed".to_string()),
                    execution_time: Duration::from_millis(5),
                    metrics: HashMap::new(),
                },
            ],
            benchmark_results: None,
            fuzzing_results: None,
            regression_results: None,
            cross_validation_results: None,
        };

        let report = TestReportUtils::generate_report(&test_result);
        assert!(report.contains("Total Tests: 10"));
        assert!(report.contains("Passed: 8"));
        assert!(report.contains("Failed: 2"));
    }

    #[test]
    fn test_metrics_analysis() {
        let results = vec![
            TestResult {
                test_case: "test1".to_string(),
                passed: true,
                error: None,
                execution_time: Duration::from_millis(10),
                metrics: {
                    let mut m = HashMap::new();
                    m.insert("num_tokens".to_string(), 5.0);
                    m.insert("compression_ratio".to_string(), 0.8);
                    m
                },
            },
            TestResult {
                test_case: "test2".to_string(),
                passed: true,
                error: None,
                execution_time: Duration::from_millis(20),
                metrics: {
                    let mut m = HashMap::new();
                    m.insert("num_tokens".to_string(), 3.0);
                    m.insert("compression_ratio".to_string(), 1.2);
                    m
                },
            },
        ];

        let analysis = TestReportUtils::analyze_metrics(&results);
        assert!(analysis.contains_key("avg_execution_time_ms"));
        assert!(analysis.contains_key("avg_tokens_per_test"));
        assert!(analysis.contains_key("avg_compression_ratio"));
    }
}
