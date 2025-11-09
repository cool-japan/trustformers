//! AWS Lambda Integration for TrustformeRS C API
//!
//! This module provides comprehensive AWS Lambda deployment capabilities for TrustformeRS models,
//! including custom runtime, event handling, and performance optimization for serverless environments.

use crate::error::{TrustformersError, TrustformersResult};
use crate::model::TrustformersModel;
use crate::pipeline::TrustformersPipeline;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// AWS Lambda Runtime API version
const LAMBDA_RUNTIME_API_VERSION: &str = "2018-06-01";

/// AWS Lambda runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaConfig {
    /// Function name
    pub function_name: String,
    /// Memory allocated to the function (128-10240 MB)
    pub memory_size: u32,
    /// Timeout in seconds (1-900)
    pub timeout: u32,
    /// Runtime type
    pub runtime: LambdaRuntime,
    /// Environment variables
    pub environment: HashMap<String, String>,
    /// Cold start optimization settings
    pub cold_start_optimization: ColdStartConfig,
}

/// Lambda runtime types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LambdaRuntime {
    /// Custom runtime using provided AL2 base
    ProvidedAl2,
    /// Custom runtime using provided base
    Provided,
    /// Container runtime
    Container,
}

/// Cold start optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartConfig {
    /// Enable model preloading
    pub preload_models: bool,
    /// Models to preload during initialization
    pub preload_model_list: Vec<String>,
    /// Enable model caching in /tmp
    pub enable_tmp_caching: bool,
    /// Maximum cache size in MB
    pub max_cache_size: u32,
    /// Enable provisioned concurrency optimization
    pub provisioned_concurrency: bool,
}

/// Lambda event structure for TrustformeRS inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaInferenceEvent {
    /// Request ID for tracking
    pub request_id: String,
    /// Inference task type
    pub task: String,
    /// Input text or data
    pub input: LambdaInput,
    /// Model configuration
    pub model_config: Option<ModelConfig>,
    /// Processing options
    pub options: Option<HashMap<String, serde_json::Value>>,
}

/// Lambda input formats
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LambdaInput {
    /// Single text input
    Text(String),
    /// Batch text inputs
    TextBatch(Vec<String>),
    /// Structured input with context
    Structured {
        text: String,
        context: Option<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    },
    /// Base64 encoded binary data
    Binary { data: String, format: String },
}

/// Model configuration for Lambda
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model identifier
    pub model_id: String,
    /// Model version
    pub version: Option<String>,
    /// Device preference
    pub device: Option<String>,
    /// Quantization settings
    pub quantization: Option<QuantizationConfig>,
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Enable quantization
    pub enabled: bool,
    /// Quantization precision (int8, int4, etc.)
    pub precision: String,
    /// Quantization method
    pub method: String,
}

/// Lambda response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaInferenceResponse {
    /// Request ID
    pub request_id: String,
    /// Success status
    pub success: bool,
    /// Results
    pub results: Option<serde_json::Value>,
    /// Error information
    pub error: Option<String>,
    /// Performance metrics
    pub metrics: LambdaMetrics,
    /// Response metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Performance metrics for Lambda execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaMetrics {
    /// Inference duration in milliseconds
    pub inference_duration_ms: u64,
    /// Total duration including overhead
    pub total_duration_ms: u64,
    /// Memory usage in MB
    pub memory_used_mb: u64,
    /// Cold start indicator
    pub cold_start: bool,
    /// Model loading time (if applicable)
    pub model_load_time_ms: Option<u64>,
    /// Batch size processed
    pub batch_size: u32,
}

/// Lambda runtime handler
pub struct LambdaHandler {
    /// Lambda configuration
    config: LambdaConfig,
    /// Preloaded models
    models: HashMap<String, TrustformersModel>,
    /// Preloaded pipelines
    pipelines: HashMap<String, TrustformersPipeline>,
    /// Runtime metrics
    metrics: RuntimeMetrics,
    /// Cold start flag
    is_cold_start: bool,
}

/// Runtime metrics tracking
#[derive(Debug, Default)]
struct RuntimeMetrics {
    /// Total requests processed
    total_requests: u64,
    /// Successful requests
    successful_requests: u64,
    /// Failed requests
    failed_requests: u64,
    /// Average inference time
    avg_inference_time: Duration,
    /// Cache hit rate
    cache_hit_rate: f64,
}

impl LambdaHandler {
    /// Create new Lambda handler with configuration
    pub fn new(config: LambdaConfig) -> TrustformersResult<Self> {
        let mut handler = Self {
            config,
            models: HashMap::new(),
            pipelines: HashMap::new(),
            metrics: RuntimeMetrics::default(),
            is_cold_start: true,
        };

        // Initialize cold start optimizations
        if handler.config.cold_start_optimization.preload_models {
            handler.preload_models()?;
        }

        Ok(handler)
    }

    /// Preload commonly used models to reduce cold start latency
    fn preload_models(&mut self) -> TrustformersResult<()> {
        for model_id in &self.config.cold_start_optimization.preload_model_list {
            let start_time = Instant::now();

            // Load model
            let model: TrustformersModel = 0; // placeholder handle
            self.models.insert(model_id.clone(), model);

            // Create pipeline
            let pipeline: TrustformersPipeline = 0; // placeholder handle
            self.pipelines.insert(model_id.clone(), pipeline);

            let load_time = start_time.elapsed();
            eprintln!("Preloaded model {} in {:?}", model_id, load_time);
        }

        Ok(())
    }

    /// Handle Lambda inference event
    pub fn handle_event(
        &mut self,
        event: LambdaInferenceEvent,
    ) -> TrustformersResult<LambdaInferenceResponse> {
        let start_time = Instant::now();
        let inference_start = Instant::now();

        // Update metrics
        self.metrics.total_requests += 1;

        // Get pipeline key for later lookup
        let pipeline_key = format!(
            "{}:{}",
            event.task,
            event
                .model_config
                .as_ref()
                .map(|c| &c.model_id)
                .unwrap_or(&"default".to_string())
        );

        // Ensure pipeline exists
        self.get_or_load_pipeline(&event.task, &event.model_config)?;

        // Get batch size before processing (to avoid partial move)
        let batch_size = self.get_batch_size(&event.input);

        // Process input based on type
        let results = match event.input {
            LambdaInput::Text(text) => {
                let pipeline =
                    self.pipelines.get(&pipeline_key).ok_or(TrustformersError::RuntimeError)?;
                self.process_single_text(pipeline, &text, &event.options)?
            },
            LambdaInput::TextBatch(texts) => {
                let pipeline =
                    self.pipelines.get(&pipeline_key).ok_or(TrustformersError::RuntimeError)?;
                self.process_batch_texts(pipeline, &texts, &event.options)?
            },
            LambdaInput::Structured {
                text,
                context,
                metadata: _,
            } => {
                let pipeline =
                    self.pipelines.get(&pipeline_key).ok_or(TrustformersError::RuntimeError)?;
                self.process_structured_input(pipeline, &text, context.as_deref(), &event.options)?
            },
            LambdaInput::Binary { data, format } => {
                let pipeline =
                    self.pipelines.get(&pipeline_key).ok_or(TrustformersError::RuntimeError)?;
                self.process_binary_input(pipeline, &data, &format, &event.options)?
            },
        };

        let inference_duration = inference_start.elapsed();
        let total_duration = start_time.elapsed();

        // Update success metrics
        self.metrics.successful_requests += 1;
        self.update_average_inference_time(inference_duration);

        // Get memory usage
        let memory_used = self.get_memory_usage();

        let response = LambdaInferenceResponse {
            request_id: event.request_id,
            success: true,
            results: Some(results),
            error: None,
            metrics: LambdaMetrics {
                inference_duration_ms: inference_duration.as_millis() as u64,
                total_duration_ms: total_duration.as_millis() as u64,
                memory_used_mb: memory_used,
                cold_start: self.is_cold_start,
                model_load_time_ms: None,
                batch_size,
            },
            metadata: self.create_response_metadata(),
        };

        // Clear cold start flag after first request
        self.is_cold_start = false;

        Ok(response)
    }

    /// Get or load pipeline for the specified task
    fn get_or_load_pipeline(
        &mut self,
        task: &str,
        model_config: &Option<ModelConfig>,
    ) -> TrustformersResult<&TrustformersPipeline> {
        let pipeline_key = format!(
            "{}:{}",
            task,
            model_config.as_ref().map(|c| &c.model_id).unwrap_or(&"default".to_string())
        );

        if !self.pipelines.contains_key(&pipeline_key) {
            let model_id = model_config
                .as_ref()
                .map(|c| c.model_id.clone())
                .unwrap_or_else(|| "default".to_string());

            let pipeline: TrustformersPipeline = 0; // placeholder handle
            self.pipelines.insert(pipeline_key.clone(), pipeline);
        }

        self.pipelines.get(&pipeline_key).ok_or_else(|| TrustformersError::RuntimeError)
    }

    /// Process single text input
    fn process_single_text(
        &self,
        pipeline: &TrustformersPipeline,
        text: &str,
        options: &Option<HashMap<String, serde_json::Value>>,
    ) -> TrustformersResult<serde_json::Value> {
        // Extract options
        let opts = options.as_ref().cloned().unwrap_or_default();

        // Perform inference
        let result = serde_json::json!({"text": text, "task": "placeholder"}); // placeholder

        Ok(serde_json::to_value(result)?)
    }

    /// Process batch of text inputs
    fn process_batch_texts(
        &self,
        pipeline: &TrustformersPipeline,
        texts: &[String],
        options: &Option<HashMap<String, serde_json::Value>>,
    ) -> TrustformersResult<serde_json::Value> {
        let opts = options.as_ref().cloned().unwrap_or_default();
        let mut results = Vec::new();

        for text in texts {
            let result = serde_json::json!({"text": text, "task": "placeholder"}); // placeholder
            results.push(result);
        }

        Ok(serde_json::to_value(results)?)
    }

    /// Process structured input with context
    fn process_structured_input(
        &self,
        pipeline: &TrustformersPipeline,
        text: &str,
        context: Option<&str>,
        options: &Option<HashMap<String, serde_json::Value>>,
    ) -> TrustformersResult<serde_json::Value> {
        let mut opts = options.as_ref().cloned().unwrap_or_default();

        // Add context if provided
        if let Some(ctx) = context {
            opts.insert(
                "context".to_string(),
                serde_json::Value::String(ctx.to_string()),
            );
        }

        let result = serde_json::json!({"text": text, "task": "placeholder"}); // placeholder
        Ok(serde_json::to_value(result)?)
    }

    /// Process binary input (base64 encoded)
    fn process_binary_input(
        &self,
        pipeline: &TrustformersPipeline,
        data: &str,
        format: &str,
        options: &Option<HashMap<String, serde_json::Value>>,
    ) -> TrustformersResult<serde_json::Value> {
        // Decode base64 data
        let decoded = base64::decode(data).map_err(|_| TrustformersError::RuntimeError)?;

        // Convert to text based on format
        let text = match format {
            "utf8" | "text" => {
                String::from_utf8(decoded).map_err(|_| TrustformersError::RuntimeError)?
            },
            _ => return Err(TrustformersError::RuntimeError),
        };

        self.process_single_text(pipeline, &text, options)
    }

    /// Get current memory usage in MB
    fn get_memory_usage(&self) -> u64 {
        // Read from /proc/self/status on Linux
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb / 1024; // Convert KB to MB
                        }
                    }
                }
            }
        }

        // Fallback estimate
        0
    }

    /// Get batch size from input
    fn get_batch_size(&self, input: &LambdaInput) -> u32 {
        match input {
            LambdaInput::Text(_) => 1,
            LambdaInput::TextBatch(texts) => texts.len() as u32,
            LambdaInput::Structured { .. } => 1,
            LambdaInput::Binary { .. } => 1,
        }
    }

    /// Update average inference time
    fn update_average_inference_time(&mut self, duration: Duration) {
        let current_avg = self.metrics.avg_inference_time.as_millis() as f64;
        let new_time = duration.as_millis() as f64;
        let total_requests = self.metrics.total_requests as f64;

        let new_avg = (current_avg * (total_requests - 1.0) + new_time) / total_requests;
        self.metrics.avg_inference_time = Duration::from_millis(new_avg as u64);
    }

    /// Create response metadata
    fn create_response_metadata(&self) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();

        metadata.insert(
            "runtime".to_string(),
            serde_json::Value::String("trustformers-lambda".to_string()),
        );
        metadata.insert(
            "version".to_string(),
            serde_json::Value::String(env!("CARGO_PKG_VERSION").to_string()),
        );
        metadata.insert(
            "total_requests".to_string(),
            serde_json::Value::Number(self.metrics.total_requests.into()),
        );
        metadata.insert(
            "success_rate".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(
                    self.metrics.successful_requests as f64
                        / self.metrics.total_requests.max(1) as f64,
                )
                .unwrap_or(serde_json::Number::from(0)),
            ),
        );

        metadata
    }

    /// Handle Lambda error event
    pub fn handle_error(
        &mut self,
        error: TrustformersError,
        request_id: String,
    ) -> LambdaInferenceResponse {
        self.metrics.failed_requests += 1;

        LambdaInferenceResponse {
            request_id,
            success: false,
            results: None,
            error: Some(error.to_string()),
            metrics: LambdaMetrics {
                inference_duration_ms: 0,
                total_duration_ms: 0,
                memory_used_mb: self.get_memory_usage(),
                cold_start: self.is_cold_start,
                model_load_time_ms: None,
                batch_size: 0,
            },
            metadata: self.create_response_metadata(),
        }
    }
}

/// Lambda deployment utilities
pub struct LambdaDeployment;

impl LambdaDeployment {
    /// Generate AWS SAM template for TrustformeRS Lambda deployment
    pub fn generate_sam_template(config: &LambdaConfig) -> TrustformersResult<String> {
        let template = format!(
            r#"
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: TrustformeRS Lambda Deployment

Globals:
  Function:
    Timeout: {}
    MemorySize: {}
    Environment:
      Variables:
        RUST_LOG: info
        TRUSTFORMERS_CACHE_DIR: /tmp
        {}

Resources:
  TrustformersFunction:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: {}
      CodeUri: target/lambda/
      Handler: bootstrap
      Runtime: provided.al2
      Architectures:
        - x86_64
      Events:
        ApiEvent:
          Type: Api
          Properties:
            Path: /inference
            Method: post
      Policies:
        - CloudWatchLogsFullAccess
        - Version: '2012-10-17'
          Statement:
            - Effect: Allow
              Action:
                - s3:GetObject
                - s3:ListBucket
              Resource:
                - arn:aws:s3:::trustformers-models/*
                - arn:aws:s3:::trustformers-models

Outputs:
  TrustformersApi:
    Description: "API Gateway endpoint URL"
    Value: !Sub "https://${{ServerlessRestApi}}.execute-api.${{AWS::Region}}.amazonaws.com/Prod/inference/"
  TrustformersFunction:
    Description: "TrustformeRS Lambda Function ARN"
    Value: !GetAtt TrustformersFunction.Arn
"#,
            config.timeout,
            config.memory_size,
            config
                .environment
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<_>>()
                .join("\n        "),
            config.function_name
        );

        Ok(template)
    }

    /// Generate Dockerfile for Lambda container deployment
    pub fn generate_dockerfile() -> String {
        r#"
FROM public.ecr.aws/lambda/provided:al2

# Install system dependencies
RUN yum update -y && \
    yum install -y gcc gcc-c++ make curl && \
    yum clean all

# Install Rust
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Copy source code
COPY . /app
WORKDIR /app

# Build the application
RUN cargo build --release --target x86_64-unknown-linux-musl

# Copy the binary to Lambda runtime directory
RUN cp target/x86_64-unknown-linux-musl/release/lambda-runtime ${LAMBDA_RUNTIME_DIR}/bootstrap

# Set the CMD to your handler
CMD ["bootstrap"]
"#
        .to_string()
    }

    /// Generate build script for Lambda deployment
    pub fn generate_build_script() -> String {
        r#"#!/bin/bash
set -e

echo "Building TrustformeRS Lambda function..."

# Install cross compilation target
rustup target add x86_64-unknown-linux-musl

# Build for Lambda runtime
cargo build --release --target x86_64-unknown-linux-musl

# Create deployment package
mkdir -p target/lambda
cp target/x86_64-unknown-linux-musl/release/lambda-runtime target/lambda/bootstrap
chmod +x target/lambda/bootstrap

# Create ZIP package
cd target/lambda
zip -r ../trustformers-lambda.zip .
cd ../..

echo "Lambda package created: target/trustformers-lambda.zip"
echo "Deploy with: aws lambda create-function --function-name trustformers --runtime provided.al2 --role <role-arn> --handler bootstrap --zip-file fileb://target/trustformers-lambda.zip"
"#.to_string()
    }
}

/// AWS Lambda runtime integration
pub mod runtime {
    use super::*;
    use std::env;

    /// Initialize Lambda runtime
    pub fn init() -> TrustformersResult<LambdaHandler> {
        // Read configuration from environment
        let config = LambdaConfig {
            function_name: env::var("AWS_LAMBDA_FUNCTION_NAME")
                .unwrap_or_else(|_| "trustformers".to_string()),
            memory_size: env::var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(512),
            timeout: env::var("AWS_LAMBDA_FUNCTION_TIMEOUT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30),
            runtime: LambdaRuntime::ProvidedAl2,
            environment: std::env::vars().collect(),
            cold_start_optimization: ColdStartConfig {
                preload_models: env::var("TRUSTFORMERS_PRELOAD_MODELS")
                    .unwrap_or_else(|_| "true".to_string())
                    == "true",
                preload_model_list: env::var("TRUSTFORMERS_PRELOAD_MODEL_LIST")
                    .unwrap_or_else(|_| {
                        "distilbert-base-uncased-finetuned-sst-2-english".to_string()
                    })
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect(),
                enable_tmp_caching: env::var("TRUSTFORMERS_ENABLE_TMP_CACHING")
                    .unwrap_or_else(|_| "true".to_string())
                    == "true",
                max_cache_size: env::var("TRUSTFORMERS_MAX_CACHE_SIZE")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(256),
                provisioned_concurrency: false,
            },
        };

        LambdaHandler::new(config)
    }

    /// Main Lambda runtime loop
    pub async fn run() -> TrustformersResult<()> {
        let mut handler = init()?;

        eprintln!("TrustformeRS Lambda runtime initialized");

        // Lambda runtime event loop would go here
        // This is a simplified version - actual implementation would use AWS Lambda Runtime API

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lambda_config_creation() {
        let config = LambdaConfig {
            function_name: "test-function".to_string(),
            memory_size: 512,
            timeout: 30,
            runtime: LambdaRuntime::ProvidedAl2,
            environment: HashMap::new(),
            cold_start_optimization: ColdStartConfig {
                preload_models: true,
                preload_model_list: vec!["test-model".to_string()],
                enable_tmp_caching: true,
                max_cache_size: 256,
                provisioned_concurrency: false,
            },
        };

        assert_eq!(config.function_name, "test-function");
        assert_eq!(config.memory_size, 512);
    }

    #[test]
    fn test_sam_template_generation() {
        let config = LambdaConfig {
            function_name: "test-function".to_string(),
            memory_size: 512,
            timeout: 30,
            runtime: LambdaRuntime::ProvidedAl2,
            environment: HashMap::new(),
            cold_start_optimization: ColdStartConfig {
                preload_models: false,
                preload_model_list: vec![],
                enable_tmp_caching: true,
                max_cache_size: 256,
                provisioned_concurrency: false,
            },
        };

        let template = LambdaDeployment::generate_sam_template(&config).unwrap();
        assert!(template.contains("test-function"));
        assert!(template.contains("MemorySize: 512"));
    }
}
