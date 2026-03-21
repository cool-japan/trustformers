//! Load Testing CLI Tool
//!
//! Command-line interface for running comprehensive load tests against
//! TrustformeRS Serve API endpoints with configurable scenarios and reporting.

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use trustformers_serve::{
    AuthType, HttpMethod, LoadTestAuthConfig, LoadTestConfig, LoadTestConfigBuilder,
    LoadTestOutputFormat, LoadTestService, TestScenario, ValidationRule, ValidationRuleType,
};

#[derive(Parser)]
#[command(name = "load_test")]
#[command(about = "Load testing tool for TrustformeRS Serve API")]
#[command(version = "1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a load test with specified parameters
    Run {
        /// Base URL of the server to test
        #[arg(short, long, default_value = "http://localhost:8080")]
        url: String,

        /// Test duration in seconds
        #[arg(short, long, default_value = "60")]
        duration: u64,

        /// Number of concurrent users
        #[arg(short, long, default_value = "10")]
        concurrent_users: usize,

        /// Requests per second (rate limiting)
        #[arg(short, long)]
        rps: Option<f64>,

        /// Configuration file path
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Output file for results
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// API key for authentication
        #[arg(long)]
        api_key: Option<String>,

        /// JWT token for authentication
        #[arg(long)]
        jwt_token: Option<String>,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Specific endpoint to test
        #[arg(short, long, default_value = "/health")]
        endpoint: String,

        /// HTTP method
        #[arg(short, long, default_value = "GET")]
        method: String,
    },

    /// Run predefined load test scenarios
    Scenario {
        /// Scenario name to run
        #[arg(value_enum)]
        scenario: PredefinedScenario,

        /// Base URL of the server
        #[arg(short, long, default_value = "http://localhost:8080")]
        url: String,

        /// API key for authentication
        #[arg(long)]
        api_key: Option<String>,

        /// Output file for results
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Generate a sample configuration file
    GenerateConfig {
        /// Output file path
        #[arg(short, long, default_value = "load_test_config.json")]
        output: PathBuf,
    },

    /// Validate a configuration file
    ValidateConfig {
        /// Configuration file to validate
        #[arg(short, long)]
        config: PathBuf,
    },
}

#[derive(Debug, clap::ValueEnum, Clone)]
enum PredefinedScenario {
    /// Basic health check scenario
    HealthCheck,
    /// API stress test scenario
    ApiStress,
    /// Mixed workload scenario
    Mixed,
    /// Inference performance test
    InferenceTest,
    /// Authentication test
    AuthTest,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            url,
            duration,
            concurrent_users,
            rps,
            config,
            output,
            api_key,
            jwt_token,
            verbose,
            endpoint,
            method,
        } => {
            run_load_test(LoadTestParams {
                url,
                duration,
                concurrent_users,
                rps,
                config_file: config,
                output_file: output,
                api_key,
                jwt_token,
                verbose,
                endpoint,
                method,
            })
            .await?;
        },

        Commands::Scenario {
            scenario,
            url,
            api_key,
            output,
        } => {
            run_predefined_scenario(scenario, url, api_key, output).await?;
        },

        Commands::GenerateConfig { output } => {
            generate_sample_config(output).await?;
        },

        Commands::ValidateConfig { config } => {
            validate_config(config).await?;
        },
    }

    Ok(())
}

/// Load test configuration parameters
struct LoadTestParams {
    url: String,
    duration: u64,
    concurrent_users: usize,
    rps: Option<f64>,
    config_file: Option<PathBuf>,
    output_file: Option<PathBuf>,
    api_key: Option<String>,
    jwt_token: Option<String>,
    verbose: bool,
    endpoint: String,
    method: String,
}

/// Run a load test with the specified parameters
async fn run_load_test(params: LoadTestParams) -> Result<()> {
    let LoadTestParams {
        url,
        duration,
        concurrent_users,
        rps,
        config_file,
        output_file,
        api_key,
        jwt_token,
        verbose,
        endpoint,
        method,
    } = params;
    println!("🚀 Initializing load test...");

    // Load configuration from file if provided
    let mut config = if let Some(config_path) = config_file {
        println!("📄 Loading configuration from: {}", config_path.display());
        let config_content = tokio::fs::read_to_string(&config_path).await?;
        serde_json::from_str::<LoadTestConfig>(&config_content)?
    } else {
        // Build configuration from command line arguments
        let mut builder = LoadTestConfigBuilder::new()
            .base_url(&url)
            .duration(duration)
            .concurrent_users(concurrent_users);

        if let Some(rps_value) = rps {
            builder = builder.requests_per_second(rps_value);
        }

        let http_method = match method.to_uppercase().as_str() {
            "GET" => HttpMethod::Get,
            "POST" => HttpMethod::Post,
            "PUT" => HttpMethod::Put,
            "DELETE" => HttpMethod::Delete,
            "PATCH" => HttpMethod::Patch,
            "HEAD" => HttpMethod::Head,
            "OPTIONS" => HttpMethod::Options,
            _ => {
                eprintln!("❌ Unsupported HTTP method: {}", method);
                return Ok(());
            },
        };

        let scenario = TestScenario {
            name: "cli_test".to_string(),
            weight: 100.0,
            method: http_method,
            path: endpoint,
            body_template: None,
            query_params: std::collections::HashMap::new(),
            headers: std::collections::HashMap::new(),
            expected_status_codes: vec![200, 201, 204],
            validation_rules: vec![],
        };

        builder = builder.add_scenario(scenario);

        // Add authentication if provided
        if api_key.is_some() || jwt_token.is_some() {
            let auth_config = LoadTestAuthConfig {
                auth_type: if api_key.is_some() { AuthType::ApiKey } else { AuthType::Bearer },
                api_key,
                jwt_token,
                oauth2: None,
            };
            builder = builder.auth(auth_config);
        }

        builder.build()
    };

    // Override base URL if provided via CLI
    config.base_url = url;

    // Enable verbose output if requested
    if verbose {
        config.output_config.enable_realtime_output = true;
        config.output_config.enable_detailed_logging = true;
    }

    // Set output file if provided
    if let Some(output_path) = output_file {
        config.output_config.output_file = Some(output_path.to_string_lossy().to_string());
        config.output_config.format = if output_path.extension().is_some_and(|ext| ext == "json") {
            LoadTestOutputFormat::Json
        } else if output_path.extension().is_some_and(|ext| ext == "csv") {
            LoadTestOutputFormat::Csv
        } else {
            LoadTestOutputFormat::Json
        };
    }

    // Create and run load test service
    let load_test_service = LoadTestService::new(config)?;
    let results = load_test_service.run().await?;

    // Save results if output file specified
    if let Some(output_path) = &results.config.output_config.output_file {
        save_results(&results, output_path).await?;
        println!("💾 Results saved to: {}", output_path);
    }

    // Print final recommendations
    print_recommendations(&results).await;

    Ok(())
}

/// Run a predefined scenario
async fn run_predefined_scenario(
    scenario: PredefinedScenario,
    url: String,
    api_key: Option<String>,
    output_file: Option<PathBuf>,
) -> Result<()> {
    println!("🎯 Running predefined scenario: {:?}", scenario);

    let config = match scenario {
        PredefinedScenario::HealthCheck => create_health_check_scenario(url, api_key),
        PredefinedScenario::ApiStress => create_api_stress_scenario(url, api_key),
        PredefinedScenario::Mixed => create_mixed_workload_scenario(url, api_key),
        PredefinedScenario::InferenceTest => create_inference_test_scenario(url, api_key),
        PredefinedScenario::AuthTest => create_auth_test_scenario(url, api_key),
    };

    let load_test_service = LoadTestService::new(config)?;
    let results = load_test_service.run().await?;

    // Save results if output file specified
    if let Some(output_path) = output_file {
        let output_path_str = output_path.to_string_lossy().to_string();
        save_results(&results, &output_path_str).await?;
        println!("💾 Results saved to: {}", output_path_str);
    }

    print_recommendations(&results).await;

    Ok(())
}

/// Create health check scenario configuration
fn create_health_check_scenario(url: String, api_key: Option<String>) -> LoadTestConfig {
    let mut builder = LoadTestConfigBuilder::new()
        .base_url(&url)
        .duration(30)
        .concurrent_users(5)
        .requests_per_second(10.0);

    let scenario = TestScenario {
        name: "health_check".to_string(),
        weight: 100.0,
        method: HttpMethod::Get,
        path: "/health".to_string(),
        body_template: None,
        query_params: std::collections::HashMap::new(),
        headers: std::collections::HashMap::new(),
        expected_status_codes: vec![200],
        validation_rules: vec![ValidationRule {
            rule_type: ValidationRuleType::ResponseContains,
            json_path: None,
            expected_value: Some("healthy".to_string()),
            pattern: None,
            header_name: None,
        }],
    };

    builder = builder.add_scenario(scenario);

    if let Some(key) = api_key {
        let auth_config = LoadTestAuthConfig {
            auth_type: AuthType::ApiKey,
            api_key: Some(key),
            jwt_token: None,
            oauth2: None,
        };
        builder = builder.auth(auth_config);
    }

    builder.build()
}

/// Create API stress test scenario configuration
fn create_api_stress_scenario(url: String, api_key: Option<String>) -> LoadTestConfig {
    let mut builder = LoadTestConfigBuilder::new()
        .base_url(&url)
        .duration(300)  // 5 minutes
        .concurrent_users(50)
        .requests_per_second(100.0);

    // Health check scenario
    let health_scenario = TestScenario {
        name: "health_check".to_string(),
        weight: 30.0,
        method: HttpMethod::Get,
        path: "/health".to_string(),
        body_template: None,
        query_params: std::collections::HashMap::new(),
        headers: std::collections::HashMap::new(),
        expected_status_codes: vec![200],
        validation_rules: vec![],
    };

    // Stats scenario
    let stats_scenario = TestScenario {
        name: "get_stats".to_string(),
        weight: 20.0,
        method: HttpMethod::Get,
        path: "/admin/stats".to_string(),
        body_template: None,
        query_params: std::collections::HashMap::new(),
        headers: std::collections::HashMap::new(),
        expected_status_codes: vec![200, 401, 403],
        validation_rules: vec![],
    };

    // Config scenario
    let config_scenario = TestScenario {
        name: "get_config".to_string(),
        weight: 10.0,
        method: HttpMethod::Get,
        path: "/admin/config".to_string(),
        body_template: None,
        query_params: std::collections::HashMap::new(),
        headers: std::collections::HashMap::new(),
        expected_status_codes: vec![200, 401, 403],
        validation_rules: vec![],
    };

    // Metrics scenario
    let metrics_scenario = TestScenario {
        name: "get_metrics".to_string(),
        weight: 40.0,
        method: HttpMethod::Get,
        path: "/metrics".to_string(),
        body_template: None,
        query_params: std::collections::HashMap::new(),
        headers: std::collections::HashMap::new(),
        expected_status_codes: vec![200],
        validation_rules: vec![],
    };

    builder = builder
        .add_scenario(health_scenario)
        .add_scenario(stats_scenario)
        .add_scenario(config_scenario)
        .add_scenario(metrics_scenario);

    if let Some(key) = api_key {
        let auth_config = LoadTestAuthConfig {
            auth_type: AuthType::ApiKey,
            api_key: Some(key),
            jwt_token: None,
            oauth2: None,
        };
        builder = builder.auth(auth_config);
    }

    builder.build()
}

/// Create mixed workload scenario configuration
fn create_mixed_workload_scenario(url: String, api_key: Option<String>) -> LoadTestConfig {
    let mut builder = LoadTestConfigBuilder::new()
        .base_url(&url)
        .duration(180)  // 3 minutes
        .concurrent_users(25);

    // Read operations (80%)
    let health_scenario = TestScenario {
        name: "health_check".to_string(),
        weight: 40.0,
        method: HttpMethod::Get,
        path: "/health".to_string(),
        body_template: None,
        query_params: std::collections::HashMap::new(),
        headers: std::collections::HashMap::new(),
        expected_status_codes: vec![200],
        validation_rules: vec![],
    };

    let detailed_health_scenario = TestScenario {
        name: "detailed_health".to_string(),
        weight: 20.0,
        method: HttpMethod::Get,
        path: "/health/detailed".to_string(),
        body_template: None,
        query_params: std::collections::HashMap::new(),
        headers: std::collections::HashMap::new(),
        expected_status_codes: vec![200],
        validation_rules: vec![],
    };

    let readiness_scenario = TestScenario {
        name: "readiness_check".to_string(),
        weight: 10.0,
        method: HttpMethod::Get,
        path: "/health/readiness".to_string(),
        body_template: None,
        query_params: std::collections::HashMap::new(),
        headers: std::collections::HashMap::new(),
        expected_status_codes: vec![200, 503],
        validation_rules: vec![],
    };

    let liveness_scenario = TestScenario {
        name: "liveness_check".to_string(),
        weight: 10.0,
        method: HttpMethod::Get,
        path: "/health/liveness".to_string(),
        body_template: None,
        query_params: std::collections::HashMap::new(),
        headers: std::collections::HashMap::new(),
        expected_status_codes: vec![200],
        validation_rules: vec![],
    };

    // Long polling (20%)
    let poll_scenario = TestScenario {
        name: "long_poll".to_string(),
        weight: 20.0,
        method: HttpMethod::Get,
        path: "/v1/poll".to_string(),
        body_template: None,
        query_params: [
            ("event_types".to_string(), "inference,health".to_string()),
            ("timeout_seconds".to_string(), "5".to_string()),
        ]
        .iter()
        .cloned()
        .collect(),
        headers: std::collections::HashMap::new(),
        expected_status_codes: vec![200, 400],
        validation_rules: vec![],
    };

    builder = builder
        .add_scenario(health_scenario)
        .add_scenario(detailed_health_scenario)
        .add_scenario(readiness_scenario)
        .add_scenario(liveness_scenario)
        .add_scenario(poll_scenario);

    if let Some(key) = api_key {
        let auth_config = LoadTestAuthConfig {
            auth_type: AuthType::ApiKey,
            api_key: Some(key),
            jwt_token: None,
            oauth2: None,
        };
        builder = builder.auth(auth_config);
    }

    builder.build()
}

/// Create inference test scenario configuration
fn create_inference_test_scenario(url: String, api_key: Option<String>) -> LoadTestConfig {
    let mut builder = LoadTestConfigBuilder::new()
        .base_url(&url)
        .duration(120)  // 2 minutes
        .concurrent_users(20)
        .requests_per_second(10.0);

    // Single inference
    let inference_scenario = TestScenario {
        name: "single_inference".to_string(),
        weight: 70.0,
        method: HttpMethod::Post,
        path: "/v1/inference".to_string(),
        body_template: Some(
            r#"{
            "text": "What is the capital of France?",
            "max_length": 100,
            "temperature": 0.7,
            "top_p": 0.9
        }"#
            .to_string(),
        ),
        query_params: std::collections::HashMap::new(),
        headers: [("Content-Type".to_string(), "application/json".to_string())]
            .iter()
            .cloned()
            .collect(),
        expected_status_codes: vec![200, 400, 429, 503],
        validation_rules: vec![ValidationRule {
            rule_type: ValidationRuleType::JsonFieldExists,
            json_path: Some("request_id".to_string()),
            expected_value: None,
            pattern: None,
            header_name: None,
        }],
    };

    // Batch inference
    let batch_inference_scenario = TestScenario {
        name: "batch_inference".to_string(),
        weight: 30.0,
        method: HttpMethod::Post,
        path: "/v1/inference/batch".to_string(),
        body_template: Some(
            r#"{
            "requests": [
                {
                    "text": "Hello world",
                    "max_length": 50,
                    "temperature": 0.7
                },
                {
                    "text": "How are you?",
                    "max_length": 50,
                    "temperature": 0.8
                }
            ]
        }"#
            .to_string(),
        ),
        query_params: std::collections::HashMap::new(),
        headers: [("Content-Type".to_string(), "application/json".to_string())]
            .iter()
            .cloned()
            .collect(),
        expected_status_codes: vec![200, 400, 429, 503],
        validation_rules: vec![],
    };

    builder = builder.add_scenario(inference_scenario).add_scenario(batch_inference_scenario);

    if let Some(key) = api_key {
        let auth_config = LoadTestAuthConfig {
            auth_type: AuthType::ApiKey,
            api_key: Some(key),
            jwt_token: None,
            oauth2: None,
        };
        builder = builder.auth(auth_config);
    }

    builder.build()
}

/// Create authentication test scenario configuration
fn create_auth_test_scenario(url: String, api_key: Option<String>) -> LoadTestConfig {
    let mut builder = LoadTestConfigBuilder::new().base_url(&url).duration(60).concurrent_users(10);

    // Test with authentication
    let auth_stats_scenario = TestScenario {
        name: "authenticated_stats".to_string(),
        weight: 50.0,
        method: HttpMethod::Get,
        path: "/admin/stats".to_string(),
        body_template: None,
        query_params: std::collections::HashMap::new(),
        headers: std::collections::HashMap::new(),
        expected_status_codes: vec![200, 401, 403],
        validation_rules: vec![],
    };

    // Test without authentication
    let unauth_stats_scenario = TestScenario {
        name: "unauthenticated_stats".to_string(),
        weight: 30.0,
        method: HttpMethod::Get,
        path: "/admin/stats".to_string(),
        body_template: None,
        query_params: std::collections::HashMap::new(),
        headers: std::collections::HashMap::new(),
        expected_status_codes: vec![401, 403],
        validation_rules: vec![],
    };

    // Public endpoints (should work without auth)
    let public_health_scenario = TestScenario {
        name: "public_health".to_string(),
        weight: 20.0,
        method: HttpMethod::Get,
        path: "/health".to_string(),
        body_template: None,
        query_params: std::collections::HashMap::new(),
        headers: std::collections::HashMap::new(),
        expected_status_codes: vec![200],
        validation_rules: vec![],
    };

    builder = builder
        .add_scenario(auth_stats_scenario)
        .add_scenario(unauth_stats_scenario)
        .add_scenario(public_health_scenario);

    if let Some(key) = api_key {
        let auth_config = LoadTestAuthConfig {
            auth_type: AuthType::ApiKey,
            api_key: Some(key),
            jwt_token: None,
            oauth2: None,
        };
        builder = builder.auth(auth_config);
    }

    builder.build()
}

/// Generate a sample configuration file
async fn generate_sample_config(output_path: PathBuf) -> Result<()> {
    println!("📝 Generating sample configuration file...");

    let config = LoadTestConfigBuilder::new()
        .base_url("http://localhost:8080")
        .duration(60)
        .concurrent_users(10)
        .requests_per_second(50.0)
        .build();

    let config_json = serde_json::to_string_pretty(&config)?;
    tokio::fs::write(&output_path, config_json).await?;

    println!(
        "✅ Sample configuration saved to: {}",
        output_path.display()
    );
    println!("📄 You can edit this file and use it with --config flag");

    Ok(())
}

/// Validate a configuration file
async fn validate_config(config_path: PathBuf) -> Result<()> {
    println!(
        "🔍 Validating configuration file: {}",
        config_path.display()
    );

    let config_content = tokio::fs::read_to_string(&config_path).await?;
    let config: LoadTestConfig = serde_json::from_str(&config_content)?;

    // Basic validation
    let mut issues = Vec::new();

    if config.base_url.is_empty() {
        issues.push("Base URL cannot be empty");
    }

    if config.duration_seconds == 0 {
        issues.push("Duration must be greater than 0");
    }

    if config.concurrent_users == 0 {
        issues.push("Concurrent users must be greater than 0");
    }

    if config.scenarios.is_empty() {
        issues.push("At least one scenario must be defined");
    }

    for scenario in &config.scenarios {
        if scenario.name.is_empty() {
            issues.push("Scenario name cannot be empty");
        }

        if scenario.weight <= 0.0 {
            issues.push("Scenario weight must be greater than 0");
        }

        if scenario.path.is_empty() {
            issues.push("Scenario path cannot be empty");
        }
    }

    if issues.is_empty() {
        println!("✅ Configuration is valid!");
        println!("📊 Summary:");
        println!("   Base URL: {}", config.base_url);
        println!("   Duration: {} seconds", config.duration_seconds);
        println!("   Concurrent users: {}", config.concurrent_users);
        println!("   Scenarios: {}", config.scenarios.len());

        for scenario in &config.scenarios {
            println!("     - {} ({:.1}% weight)", scenario.name, scenario.weight);
        }
    } else {
        println!("❌ Configuration validation failed:");
        for issue in issues {
            println!("   - {}", issue);
        }
    }

    Ok(())
}

/// Save test results to file
async fn save_results(
    results: &trustformers_serve::LoadTestResults,
    output_path: &str,
) -> Result<()> {
    let output_format = if output_path.ends_with(".json") {
        trustformers_serve::LoadTestOutputFormat::Json
    } else if output_path.ends_with(".csv") {
        trustformers_serve::LoadTestOutputFormat::Csv
    } else {
        trustformers_serve::LoadTestOutputFormat::Json
    };

    match output_format {
        trustformers_serve::LoadTestOutputFormat::Json => {
            let json_content = serde_json::to_string_pretty(results)?;
            tokio::fs::write(output_path, json_content).await?;
        },
        trustformers_serve::LoadTestOutputFormat::Csv => {
            // Generate CSV format for key metrics
            let csv_content = format!(
                "metric,value\n\
                total_requests,{}\n\
                successful_requests,{}\n\
                failed_requests,{}\n\
                success_rate,{:.2}\n\
                average_rps,{:.2}\n\
                average_response_time_ms,{:.2}\n\
                median_response_time_ms,{:.2}\n\
                duration_seconds,{:.2}\n",
                results.summary.total_requests,
                results.summary.successful_requests,
                results.summary.failed_requests,
                results.summary.success_rate,
                results.summary.average_rps,
                results.summary.average_response_time_ms,
                results.summary.median_response_time_ms,
                results.summary.duration_seconds
            );
            tokio::fs::write(output_path, csv_content).await?;
        },
        _ => {
            let json_content = serde_json::to_string_pretty(results)?;
            tokio::fs::write(output_path, json_content).await?;
        },
    }

    Ok(())
}

/// Print performance recommendations based on test results
async fn print_recommendations(results: &trustformers_serve::LoadTestResults) {
    println!("\n💡 Performance Recommendations:");
    println!("═══════════════════════════════════════");

    // Success rate recommendations
    if results.summary.success_rate < 95.0 {
        println!("🔴 ERROR RATE HIGH ({:.2}%): Consider implementing retry logic, circuit breakers, or scaling resources", 100.0 - results.summary.success_rate);
    } else if results.summary.success_rate < 99.0 {
        println!(
            "🟡 ERROR RATE MODERATE ({:.2}%): Monitor error patterns and consider optimization",
            100.0 - results.summary.success_rate
        );
    } else {
        println!(
            "🟢 ERROR RATE EXCELLENT ({:.2}%): Great reliability!",
            100.0 - results.summary.success_rate
        );
    }

    // Response time recommendations
    if results.summary.average_response_time_ms > 1000.0 {
        println!("🔴 RESPONSE TIME HIGH ({:.2}ms): Consider caching, database optimization, or horizontal scaling", results.summary.average_response_time_ms);
    } else if results.summary.average_response_time_ms > 500.0 {
        println!(
            "🟡 RESPONSE TIME MODERATE ({:.2}ms): Look for optimization opportunities",
            results.summary.average_response_time_ms
        );
    } else {
        println!(
            "🟢 RESPONSE TIME EXCELLENT ({:.2}ms): Great performance!",
            results.summary.average_response_time_ms
        );
    }

    // Throughput recommendations
    if results.summary.average_rps < 10.0 {
        println!(
            "🔴 THROUGHPUT LOW ({:.2} RPS): Consider performance tuning and scaling",
            results.summary.average_rps
        );
    } else if results.summary.average_rps < 100.0 {
        println!(
            "🟡 THROUGHPUT MODERATE ({:.2} RPS): Room for improvement",
            results.summary.average_rps
        );
    } else {
        println!(
            "🟢 THROUGHPUT EXCELLENT ({:.2} RPS): Great capacity!",
            results.summary.average_rps
        );
    }

    // General recommendations
    println!("\n📝 General Recommendations:");

    if results.summary.success_rate < 99.0 {
        println!("   • Implement comprehensive error handling and retry mechanisms");
        println!("   • Set up proper monitoring and alerting for error rates");
        println!("   • Consider implementing circuit breakers for external dependencies");
    }

    if results.summary.average_response_time_ms > 200.0 {
        println!("   • Implement caching strategies for frequently accessed data");
        println!("   • Optimize database queries and consider read replicas");
        println!("   • Consider using CDN for static content");
    }

    if results.summary.average_rps < 50.0 {
        println!("   • Consider horizontal scaling with load balancing");
        println!("   • Optimize application performance and resource usage");
        println!("   • Implement connection pooling and keep-alive connections");
    }

    println!("   • Set up comprehensive monitoring with metrics and dashboards");
    println!("   • Implement proper logging for debugging and analysis");
    println!("   • Consider implementing rate limiting to protect against abuse");
    println!("   • Set up automated performance testing in CI/CD pipeline");

    println!("═══════════════════════════════════════");
}
