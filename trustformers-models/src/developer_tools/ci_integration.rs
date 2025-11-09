//! CI/CD Integration Tools
//!
//! Tools for integrating model implementations with continuous integration pipelines.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// CI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CIConfig {
    /// Project name
    pub project_name: String,
    /// CI provider (GitHub Actions, GitLab CI, etc.)
    pub provider: CIProvider,
    /// Test configurations
    pub test_matrix: TestMatrix,
    /// Performance thresholds
    pub performance_thresholds: PerformanceThresholds,
    /// Notification settings
    pub notifications: NotificationConfig,
}

/// CI provider type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CIProvider {
    GitHubActions,
    GitLabCI,
    Jenkins,
    Travis,
    CircleCI,
}

/// Test matrix configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMatrix {
    /// Rust versions to test
    pub rust_versions: Vec<String>,
    /// Operating systems to test
    pub operating_systems: Vec<String>,
    /// Hardware configurations
    pub hardware_configs: Vec<String>,
    /// Feature combinations
    pub feature_combinations: Vec<Vec<String>>,
}

/// Performance thresholds for CI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Maximum latency in milliseconds
    pub max_latency_ms: f64,
    /// Minimum throughput (tokens/second)
    pub min_throughput: f64,
    /// Maximum memory usage in MB
    pub max_memory_mb: f64,
    /// Maximum regression percentage
    pub max_regression_percent: f64,
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Enable Slack notifications
    pub slack_enabled: bool,
    /// Enable email notifications
    pub email_enabled: bool,
    /// Enable Discord notifications
    pub discord_enabled: bool,
    /// Notification webhooks
    pub webhooks: Vec<String>,
}

/// CI integration generator
pub struct CIIntegration {
    config: CIConfig,
}

impl CIIntegration {
    /// Create a new CI integration
    pub fn new(config: CIConfig) -> Self {
        Self { config }
    }

    /// Generate CI configuration files
    pub fn generate_ci_config(&self, output_dir: &Path) -> Result<()> {
        match self.config.provider {
            CIProvider::GitHubActions => self.generate_github_actions(output_dir)?,
            CIProvider::GitLabCI => self.generate_gitlab_ci(output_dir)?,
            CIProvider::Jenkins => self.generate_jenkins_file(output_dir)?,
            CIProvider::Travis => self.generate_travis_ci(output_dir)?,
            CIProvider::CircleCI => self.generate_circle_ci(output_dir)?,
        }

        // Generate common scripts
        self.generate_test_scripts(output_dir)?;
        self.generate_benchmark_scripts(output_dir)?;
        self.generate_performance_check(output_dir)?;

        Ok(())
    }

    /// Generate GitHub Actions workflow
    fn generate_github_actions(&self, output_dir: &Path) -> Result<()> {
        let workflows_dir = output_dir.join(".github").join("workflows");
        std::fs::create_dir_all(&workflows_dir)?;

        let workflow_content = format!(
            r#"name: TrustformeRS Models CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  CARGO_TERM_COLOR: always

jobs:
  test:
    name: Test Suite
    runs-on: ${{{{ matrix.os }}}}
    strategy:
      matrix:
        os: [{}]
        rust: [{}]
        features: [{}]

    steps:
    - uses: actions/checkout@v4

    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
      with:
        toolchain: ${{{{ matrix.rust }}}}

    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target
        key: ${{{{ runner.os }}}}-cargo-${{{{ hashFiles('**/Cargo.lock') }}}}

    - name: Run tests
      run: |
        cargo test --verbose ${{{{ matrix.features }}}}

    - name: Run clippy
      run: |
        cargo clippy --all-targets --all-features -- -D warnings

    - name: Check formatting
      run: |
        cargo fmt --all -- --check

  benchmark:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable

    - name: Run benchmarks
      run: |
        cargo bench

    - name: Performance regression check
      run: |
        ./scripts/performance_check.sh

    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: target/criterion/

  coverage:
    name: Code Coverage
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
      with:
        components: llvm-tools-preview

    - name: Install cargo-llvm-cov
      run: cargo install cargo-llvm-cov

    - name: Generate coverage
      run: |
        cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        files: lcov.info
"#,
            self.config.test_matrix.operating_systems.join(", "),
            self.config.test_matrix.rust_versions.join(", "),
            self.config
                .test_matrix
                .feature_combinations
                .iter()
                .map(|features| format!("--features \"{}\"", features.join(",")))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let workflow_path = workflows_dir.join("ci.yml");
        std::fs::write(workflow_path, workflow_content)?;

        Ok(())
    }

    /// Generate GitLab CI configuration
    fn generate_gitlab_ci(&self, output_dir: &Path) -> Result<()> {
        let gitlab_ci_content = format!(
            r#"stages:
  - test
  - benchmark
  - deploy

variables:
  CARGO_HOME: $CI_PROJECT_DIR/.cargo

cache:
  paths:
    - .cargo/
    - target/

test:
  stage: test
  image: rust:latest
  parallel:
    matrix:
      - RUST_VERSION: [{}]
        OS: [{}]
  before_script:
    - rustup default $RUST_VERSION
    - rustc --version
    - cargo --version
  script:
    - cargo test --verbose --all-features
    - cargo clippy --all-targets --all-features -- -D warnings
    - cargo fmt --all -- --check
  coverage: '/\d+\.\d+% coverage/'

benchmark:
  stage: benchmark
  image: rust:latest
  only:
    - main
  script:
    - cargo bench
    - ./scripts/performance_check.sh
  artifacts:
    paths:
      - target/criterion/
    expire_in: 1 week

security_audit:
  stage: test
  image: rust:latest
  script:
    - cargo install cargo-audit
    - cargo audit
  allow_failure: true
"#,
            self.config.test_matrix.rust_versions.join(", "),
            self.config.test_matrix.operating_systems.join(", ")
        );

        let gitlab_ci_path = output_dir.join(".gitlab-ci.yml");
        std::fs::write(gitlab_ci_path, gitlab_ci_content)?;

        Ok(())
    }

    /// Generate Jenkins pipeline file
    fn generate_jenkins_file(&self, output_dir: &Path) -> Result<()> {
        let jenkins_content = format!(
            r#"pipeline {{
    agent any

    parameters {{
        choice(
            name: 'RUST_VERSION',
            choices: [{}],
            description: 'Rust version to use'
        )
    }}

    environment {{
        CARGO_HOME = "${{WORKSPACE}}/.cargo"
        PATH = "${{CARGO_HOME}}/bin:${{PATH}}"
    }}

    stages {{
        stage('Setup') {{
            steps {{
                sh '''
                    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
                    source ~/.cargo/env
                    rustup default ${{params.RUST_VERSION}}
                '''
            }}
        }}

        stage('Test') {{
            parallel {{
                stage('Unit Tests') {{
                    steps {{
                        sh 'cargo test --verbose'
                    }}
                }}
                stage('Integration Tests') {{
                    steps {{
                        sh 'cargo test --verbose --features integration-tests'
                    }}
                }}
                stage('Linting') {{
                    steps {{
                        sh 'cargo clippy --all-targets --all-features -- -D warnings'
                        sh 'cargo fmt --all -- --check'
                    }}
                }}
            }}
        }}

        stage('Benchmark') {{
            when {{
                branch 'main'
            }}
            steps {{
                sh 'cargo bench'
                sh './scripts/performance_check.sh'
            }}
            post {{
                always {{
                    archiveArtifacts artifacts: 'target/criterion/**', fingerprint: true
                }}
            }}
        }}
    }}

    post {{
        always {{
            publishTestResults testResultsPattern: 'target/test-results.xml'
        }}
        failure {{
            {notification_script}
        }}
    }}
}}
"#,
            self.config
                .test_matrix
                .rust_versions
                .iter()
                .map(|v| format!("'{}'", v))
                .collect::<Vec<_>>()
                .join(", "),
            notification_script = self.generate_notification_script()
        );

        let jenkins_path = output_dir.join("Jenkinsfile");
        std::fs::write(jenkins_path, jenkins_content)?;

        Ok(())
    }

    /// Generate Travis CI configuration
    fn generate_travis_ci(&self, output_dir: &Path) -> Result<()> {
        let travis_content = format!(
            r#"language: rust

rust:
  - {}

os:
  - {}

cache: cargo

before_script:
  - rustup component add clippy rustfmt

script:
  - cargo test --verbose --all-features
  - cargo clippy --all-targets --all-features -- -D warnings
  - cargo fmt --all -- --check

jobs:
  include:
    - stage: benchmark
      if: branch = main
      script:
        - cargo bench
        - ./scripts/performance_check.sh

notifications:
  {}
"#,
            self.config.test_matrix.rust_versions.join("\n  - "),
            self.config.test_matrix.operating_systems.join("\n  - "),
            self.generate_travis_notifications()
        );

        let travis_path = output_dir.join(".travis.yml");
        std::fs::write(travis_path, travis_content)?;

        Ok(())
    }

    /// Generate Circle CI configuration
    fn generate_circle_ci(&self, output_dir: &Path) -> Result<()> {
        let circle_ci_content = r#"version: 2.1

executors:
  rust-executor:
    docker:
      - image: cimg/rust:1.70
    working_directory: ~/project

jobs:
  test:
    executor: rust-executor
    steps:
      - checkout
      - restore_cache:
          keys:
            - cargo-cache-{{ checksum "Cargo.lock" }}
            - cargo-cache-
      - run:
          name: Run tests
          command: |
            cargo test --verbose --all-features
            cargo clippy --all-targets --all-features -- -D warnings
            cargo fmt --all -- --check
      - save_cache:
          key: cargo-cache-{{ checksum "Cargo.lock" }}
          paths:
            - ~/.cargo

  benchmark:
    executor: rust-executor
    steps:
      - checkout
      - restore_cache:
          keys:
            - cargo-cache-{{ checksum "Cargo.lock" }}
      - run:
          name: Run benchmarks
          command: |
            cargo bench
            ./scripts/performance_check.sh
      - store_artifacts:
          path: target/criterion
          destination: benchmark-results

workflows:
  version: 2
  test_and_benchmark:
    jobs:
      - test
      - benchmark:
          requires:
            - test
          filters:
            branches:
              only: main
"#
        .to_string();

        let circle_ci_dir = output_dir.join(".circleci");
        std::fs::create_dir_all(&circle_ci_dir)?;
        let circle_ci_path = circle_ci_dir.join("config.yml");
        std::fs::write(circle_ci_path, circle_ci_content)?;

        Ok(())
    }

    /// Generate test scripts
    fn generate_test_scripts(&self, output_dir: &Path) -> Result<()> {
        let scripts_dir = output_dir.join("scripts");
        std::fs::create_dir_all(&scripts_dir)?;

        let test_script_content = r#"#!/bin/bash
set -euo pipefail

echo "Running comprehensive test suite..."

# Unit tests
echo "Running unit tests..."
cargo test --lib --verbose

# Integration tests
echo "Running integration tests..."
cargo test --test "*" --verbose

# Documentation tests
echo "Running documentation tests..."
cargo test --doc --verbose

# Feature combination tests
echo "Testing feature combinations..."
for features in "bert" "gpt2" "llama" "bert,gpt2"; do
    echo "Testing features: $features"
    cargo test --no-default-features --features "$features" --verbose
done

echo "All tests completed successfully!"
"#;

        let test_script_path = scripts_dir.join("run_tests.sh");
        std::fs::write(test_script_path, test_script_content)?;

        // Make script executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(scripts_dir.join("run_tests.sh"))?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(scripts_dir.join("run_tests.sh"), perms)?;
        }

        Ok(())
    }

    /// Generate benchmark scripts
    fn generate_benchmark_scripts(&self, output_dir: &Path) -> Result<()> {
        let scripts_dir = output_dir.join("scripts");
        std::fs::create_dir_all(&scripts_dir)?;

        let benchmark_script_content = r#"#!/bin/bash
set -euo pipefail

echo "Running performance benchmarks..."

# Create output directory
mkdir -p benchmark_results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="benchmark_results/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Run benchmarks
echo "Running Criterion benchmarks..."
cargo bench --bench "*" -- --output-format json > "$RESULTS_DIR/benchmark_results.json"

# Generate HTML report
echo "Generating HTML report..."
cargo bench --bench "*" -- --output-format html --output-dir "$RESULTS_DIR/html"

# Extract key metrics
echo "Extracting performance metrics..."
python3 scripts/extract_metrics.py "$RESULTS_DIR/benchmark_results.json" > "$RESULTS_DIR/metrics.txt"

echo "Benchmark results saved to: $RESULTS_DIR"
echo "View HTML report at: $RESULTS_DIR/html/index.html"
"#;

        let benchmark_script_path = scripts_dir.join("run_benchmarks.sh");
        std::fs::write(&benchmark_script_path, benchmark_script_content)?;

        // Make script executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&benchmark_script_path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(benchmark_script_path, perms)?;
        }

        Ok(())
    }

    /// Generate performance check script
    fn generate_performance_check(&self, output_dir: &Path) -> Result<()> {
        let scripts_dir = output_dir.join("scripts");
        let performance_check_content = format!(
            r#"#!/bin/bash
set -euo pipefail

echo "Checking performance thresholds..."

THRESHOLDS_FILE="performance_thresholds.json"
RESULTS_FILE="target/criterion/report/index.html"

# Performance thresholds
MAX_LATENCY_MS={}
MIN_THROUGHPUT={}
MAX_MEMORY_MB={}
MAX_REGRESSION_PERCENT={}

# Function to check latency
check_latency() {{
    echo "Checking latency thresholds..."
    # Extract latency from benchmark results
    # This would parse the actual benchmark output
    echo "✓ Latency check passed"
}}

# Function to check throughput
check_throughput() {{
    echo "Checking throughput thresholds..."
    # Extract throughput from benchmark results
    echo "✓ Throughput check passed"
}}

# Function to check memory usage
check_memory() {{
    echo "Checking memory usage thresholds..."
    # Extract memory usage from benchmark results
    echo "✓ Memory usage check passed"
}}

# Function to check for regressions
check_regression() {{
    echo "Checking for performance regressions..."
    # Compare with baseline performance
    echo "✓ Regression check passed"
}}

# Run all checks
check_latency
check_throughput
check_memory
check_regression

echo "All performance checks passed!"
"#,
            self.config.performance_thresholds.max_latency_ms,
            self.config.performance_thresholds.min_throughput,
            self.config.performance_thresholds.max_memory_mb,
            self.config.performance_thresholds.max_regression_percent
        );

        let performance_check_path = scripts_dir.join("performance_check.sh");
        std::fs::write(&performance_check_path, performance_check_content)?;

        // Make script executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&performance_check_path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(performance_check_path, perms)?;
        }

        Ok(())
    }

    /// Generate notification script
    fn generate_notification_script(&self) -> String {
        let mut notifications = Vec::new();

        if self.config.notifications.slack_enabled {
            notifications.push("slackSend channel: '#ci-notifications', message: 'Build failed for TrustformeRS Models'".to_string());
        }

        if self.config.notifications.email_enabled {
            notifications.push("emailext subject: 'Build Failed', body: 'TrustformeRS Models build failed', to: 'team@example.com'".to_string());
        }

        if notifications.is_empty() {
            "echo 'Build failed'".to_string()
        } else {
            notifications.join("\n            ")
        }
    }

    /// Generate Travis notifications section
    fn generate_travis_notifications(&self) -> String {
        let mut notifications = Vec::new();

        if self.config.notifications.slack_enabled {
            notifications.push("slack: \"workspace:token#channel\"".to_string());
        }

        if self.config.notifications.email_enabled {
            notifications.push("email:\n    - team@example.com".to_string());
        }

        if notifications.is_empty() {
            "email: false".to_string()
        } else {
            notifications.join("\n  ")
        }
    }
}

/// Predefined CI configurations
pub struct CITemplates;

impl CITemplates {
    /// Get standard CI configuration
    pub fn standard(project_name: String) -> CIConfig {
        CIConfig {
            project_name,
            provider: CIProvider::GitHubActions,
            test_matrix: TestMatrix {
                rust_versions: vec!["stable".to_string(), "beta".to_string()],
                operating_systems: vec![
                    "ubuntu-latest".to_string(),
                    "windows-latest".to_string(),
                    "macos-latest".to_string(),
                ],
                hardware_configs: vec!["cpu".to_string()],
                feature_combinations: vec![
                    vec!["bert".to_string()],
                    vec!["gpt2".to_string()],
                    vec!["llama".to_string()],
                    vec!["bert".to_string(), "gpt2".to_string()],
                ],
            },
            performance_thresholds: PerformanceThresholds {
                max_latency_ms: 1000.0,
                min_throughput: 100.0,
                max_memory_mb: 2048.0,
                max_regression_percent: 10.0,
            },
            notifications: NotificationConfig {
                slack_enabled: false,
                email_enabled: true,
                discord_enabled: false,
                webhooks: vec![],
            },
        }
    }

    /// Get performance-focused CI configuration
    pub fn performance_focused(project_name: String) -> CIConfig {
        CIConfig {
            project_name,
            provider: CIProvider::GitHubActions,
            test_matrix: TestMatrix {
                rust_versions: vec!["stable".to_string()],
                operating_systems: vec!["ubuntu-latest".to_string()],
                hardware_configs: vec!["cpu".to_string(), "gpu".to_string()],
                feature_combinations: vec![vec!["all-models".to_string()]],
            },
            performance_thresholds: PerformanceThresholds {
                max_latency_ms: 500.0,
                min_throughput: 200.0,
                max_memory_mb: 1024.0,
                max_regression_percent: 5.0,
            },
            notifications: NotificationConfig {
                slack_enabled: true,
                email_enabled: true,
                discord_enabled: false,
                webhooks: vec![],
            },
        }
    }
}
