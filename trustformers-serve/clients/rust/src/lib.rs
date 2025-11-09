//! # TrustformeRS Rust Client Library
//!
//! A comprehensive Rust client library for TrustformeRS serving infrastructure.
//!
//! ## Features
//!
//! - **Async/Await Support**: Built on tokio for high-performance async operations
//! - **Multiple Authentication Methods**: API Key, JWT, OAuth2, and custom authentication
//! - **Streaming Support**: Real-time streaming inference with Server-Sent Events
//! - **Batch Processing**: Efficient batch inference operations
//! - **Retry Logic**: Configurable retry logic with exponential backoff
//! - **Health Monitoring**: Health check and detailed diagnostics
//! - **Model Management**: List and inspect available models
//! - **Request/Response Logging**: Comprehensive debugging and logging support
//! - **TLS Support**: Secure connections with custom TLS configuration
//! - **Type Safety**: Strongly typed requests and responses with serde
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use trustformers_client::{TrustformersClient, InferenceRequest};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = TrustformersClient::builder("https://api.trustformers.com")
//!         .api_key("your-api-key")
//!         .build()?;
//!
//!     let request = InferenceRequest::new("What is machine learning?")
//!         .model_id("llama-3-8b-instruct");
//!
//!     let response = client.inference(request).await?;
//!     println!("Response: {}", response.choices[0].text);
//!
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use futures_util::stream::StreamExt;
use reqwest::{Client as HttpClient, Method, RequestBuilder, Response};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;
use tokio_stream::Stream;
use url::Url;
use uuid::Uuid;

/// Re-export commonly used types
pub use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
pub use reqwest::StatusCode;

/// Client configuration
#[derive(Clone, Debug)]
pub struct ClientConfig {
    /// Request timeout
    pub timeout: Duration,
    /// User agent string
    pub user_agent: String,
    /// Enable debug logging
    pub debug: bool,
    /// Maximum number of retries
    pub max_retries: u32,
    /// Initial retry delay
    pub initial_retry_delay: Duration,
    /// Maximum retry delay
    pub max_retry_delay: Duration,
    /// Backoff multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Default headers to include with requests
    pub default_headers: HeaderMap,
    /// HTTP status codes that should trigger a retry
    pub retryable_status_codes: Vec<u16>,
}

impl Default for ClientConfig {
    fn default() -> Self {
        let mut default_headers = HeaderMap::new();
        default_headers.insert("Content-Type", "application/json".parse().unwrap());
        default_headers.insert("Accept", "application/json".parse().unwrap());

        Self {
            timeout: Duration::from_secs(30),
            user_agent: "trustformers-rust-client/1.0.0".to_string(),
            debug: false,
            max_retries: 3,
            initial_retry_delay: Duration::from_millis(100),
            max_retry_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            default_headers,
            retryable_status_codes: vec![500, 502, 503, 504, 429],
        }
    }
}

/// Authentication trait for various authentication methods
#[async_trait::async_trait]
pub trait Authenticator: Send + Sync {
    /// Apply authentication to the request builder
    async fn apply(&self, request_builder: RequestBuilder) -> Result<RequestBuilder, ClientError>;
}

/// API Key authentication
#[derive(Clone)]
pub struct ApiKeyAuth {
    api_key: String,
    header: String,
    prefix: String,
}

impl ApiKeyAuth {
    /// Create a new API key authenticator
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            header: "Authorization".to_string(),
            prefix: "Bearer ".to_string(),
        }
    }

    /// Create with custom header and prefix
    pub fn with_header(api_key: impl Into<String>, header: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            header: header.into(),
            prefix: prefix.into(),
        }
    }
}

#[async_trait::async_trait]
impl Authenticator for ApiKeyAuth {
    async fn apply(&self, request_builder: RequestBuilder) -> Result<RequestBuilder, ClientError> {
        let header_value = format!("{}{}", self.prefix, self.api_key);
        Ok(request_builder.header(&self.header, header_value))
    }
}

/// JWT token authentication
#[derive(Clone)]
pub struct JwtAuth {
    token: String,
    header: String,
    prefix: String,
}

impl JwtAuth {
    /// Create a new JWT authenticator
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            header: "Authorization".to_string(),
            prefix: "Bearer ".to_string(),
        }
    }

    /// Create with custom header and prefix
    pub fn with_header(token: impl Into<String>, header: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            header: header.into(),
            prefix: prefix.into(),
        }
    }
}

#[async_trait::async_trait]
impl Authenticator for JwtAuth {
    async fn apply(&self, request_builder: RequestBuilder) -> Result<RequestBuilder, ClientError> {
        let header_value = format!("{}{}", self.prefix, self.token);
        Ok(request_builder.header(&self.header, header_value))
    }
}

/// OAuth2 authentication
#[cfg(feature = "oauth2")]
#[derive(Clone)]
pub struct OAuth2Auth {
    client: oauth2::basic::BasicClient,
    token: Arc<RwLock<Option<oauth2::StandardToken>>>,
}

#[cfg(feature = "oauth2")]
impl OAuth2Auth {
    /// Create a new OAuth2 authenticator
    pub fn new(
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
        token_url: impl AsRef<str>,
    ) -> Result<Self, ClientError> {
        let client = oauth2::basic::BasicClient::new(
            oauth2::ClientId::new(client_id.into()),
            Some(oauth2::ClientSecret::new(client_secret.into())),
            oauth2::AuthUrl::new("https://example.com/auth".to_string()).unwrap(), // Not used for client credentials
            Some(oauth2::TokenUrl::new(token_url.as_ref().to_string()).map_err(|e| ClientError::Configuration(e.to_string()))?),
        );

        Ok(Self {
            client,
            token: Arc::new(RwLock::new(None)),
        })
    }

    /// Get a valid access token
    async fn get_token(&self) -> Result<String, ClientError> {
        let token_guard = self.token.read().await;

        // Check if we have a valid token
        if let Some(token) = token_guard.as_ref() {
            if let Some(expires_at) = token.expires_in() {
                if chrono::Utc::now().timestamp() < expires_at.as_secs() as i64 {
                    return Ok(token.access_token().secret().clone());
                }
            } else {
                return Ok(token.access_token().secret().clone());
            }
        }

        drop(token_guard);

        // Need to get a new token
        let mut token_guard = self.token.write().await;

        let token_result = self
            .client
            .exchange_client_credentials()
            .request_async(oauth2::reqwest::async_http_client)
            .await
            .map_err(|e| ClientError::Authentication(format!("OAuth2 token exchange failed: {}", e)))?;

        let access_token = token_result.access_token().secret().clone();
        *token_guard = Some(token_result);

        Ok(access_token)
    }
}

#[cfg(feature = "oauth2")]
#[async_trait::async_trait]
impl Authenticator for OAuth2Auth {
    async fn apply(&self, request_builder: RequestBuilder) -> Result<RequestBuilder, ClientError> {
        let token = self.get_token().await?;
        Ok(request_builder.bearer_auth(token))
    }
}

/// Custom authentication using a closure
pub struct CustomAuth<F>
where
    F: Fn(RequestBuilder) -> Result<RequestBuilder, ClientError> + Send + Sync,
{
    apply_fn: F,
}

impl<F> CustomAuth<F>
where
    F: Fn(RequestBuilder) -> Result<RequestBuilder, ClientError> + Send + Sync,
{
    /// Create a new custom authenticator
    pub fn new(apply_fn: F) -> Self {
        Self { apply_fn }
    }
}

#[async_trait::async_trait]
impl<F> Authenticator for CustomAuth<F>
where
    F: Fn(RequestBuilder) -> Result<RequestBuilder, ClientError> + Send + Sync,
{
    async fn apply(&self, request_builder: RequestBuilder) -> Result<RequestBuilder, ClientError> {
        (self.apply_fn)(request_builder)
    }
}

/// Inference request
#[derive(Clone, Debug, Serialize)]
pub struct InferenceRequest {
    /// Input text
    pub input: String,
    /// Model ID (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Additional parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<HashMap<String, serde_json::Value>>,
    /// Inference options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<InferenceOptions>,
}

impl InferenceRequest {
    /// Create a new inference request
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            model_id: None,
            parameters: None,
            options: None,
        }
    }

    /// Set the model ID
    pub fn model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    /// Set parameters
    pub fn parameters(mut self, parameters: HashMap<String, serde_json::Value>) -> Self {
        self.parameters = Some(parameters);
        self
    }

    /// Set options
    pub fn options(mut self, options: InferenceOptions) -> Self {
        self.options = Some(options);
        self
    }

    /// Add a parameter
    pub fn parameter(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        if self.parameters.is_none() {
            self.parameters = Some(HashMap::new());
        }
        self.parameters.as_mut().unwrap().insert(key.into(), value);
        self
    }
}

/// Inference options
#[derive(Clone, Debug, Default, Serialize)]
pub struct InferenceOptions {
    /// Maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Temperature for sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Top-p for nucleus sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top-k for sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Enable streaming
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

impl InferenceOptions {
    /// Create new inference options
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max tokens
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set top-p
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set top-k
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Enable streaming
    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }
}

/// Inference response
#[derive(Clone, Debug, Deserialize)]
pub struct InferenceResponse {
    /// Response ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: i64,
    /// Model used
    pub model: String,
    /// Generated choices
    pub choices: Vec<Choice>,
    /// Token usage information
    pub usage: Usage,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
}

/// A choice in the response
#[derive(Clone, Debug, Deserialize)]
pub struct Choice {
    /// Choice index
    pub index: u32,
    /// Generated text
    pub text: String,
    /// Finish reason
    pub finish_reason: String,
    /// Confidence score
    #[serde(default)]
    pub confidence: Option<f64>,
}

/// Token usage information
#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of generated tokens
    pub completion_tokens: u32,
    /// Total tokens used
    pub total_tokens: u32,
}

/// Batch inference request
#[derive(Clone, Debug, Serialize)]
pub struct BatchInferenceRequest {
    /// List of inference requests
    pub requests: Vec<InferenceRequest>,
    /// Batch options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<BatchOptions>,
}

impl BatchInferenceRequest {
    /// Create a new batch inference request
    pub fn new(requests: Vec<InferenceRequest>) -> Self {
        Self {
            requests,
            options: None,
        }
    }

    /// Set batch options
    pub fn options(mut self, options: BatchOptions) -> Self {
        self.options = Some(options);
        self
    }
}

/// Batch processing options
#[derive(Clone, Debug, Default, Serialize)]
pub struct BatchOptions {
    /// Maximum batch size
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_batch_size: Option<u32>,
    /// Process in parallel
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel: Option<bool>,
}

impl BatchOptions {
    /// Create new batch options
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max batch size
    pub fn max_batch_size(mut self, max_batch_size: u32) -> Self {
        self.max_batch_size = Some(max_batch_size);
        self
    }

    /// Set parallel processing
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = Some(parallel);
        self
    }
}

/// Batch inference response
#[derive(Clone, Debug, Deserialize)]
pub struct BatchInferenceResponse {
    /// Response ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: i64,
    /// List of responses
    pub responses: Vec<InferenceResponse>,
    /// Batch size
    pub batch_size: u32,
}

/// Health status
#[derive(Clone, Debug, Deserialize)]
pub struct HealthStatus {
    /// Status string
    pub status: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Server version
    pub version: String,
    /// Uptime in seconds
    pub uptime: f64,
    /// Additional details
    #[serde(default)]
    pub details: HashMap<String, serde_json::Value>,
    /// Component status
    #[serde(default)]
    pub components: HashMap<String, String>,
}

/// Model information
#[derive(Clone, Debug, Deserialize)]
pub struct ModelInfo {
    /// Model ID
    pub id: String,
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Description
    pub description: String,
    /// Architecture
    pub architecture: String,
    /// Number of parameters
    pub parameters: u64,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Status
    pub status: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
}

/// Streaming chunk
#[derive(Clone, Debug, Deserialize)]
pub struct StreamingChunk {
    /// Chunk ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: i64,
    /// Model used
    pub model: String,
    /// Choices for this chunk
    pub choices: Vec<StreamingChoice>,
}

/// Streaming choice
#[derive(Clone, Debug, Deserialize)]
pub struct StreamingChoice {
    /// Choice index
    pub index: u32,
    /// Delta text
    pub delta: StreamingDelta,
    /// Finish reason (if any)
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Streaming delta
#[derive(Clone, Debug, Deserialize)]
pub struct StreamingDelta {
    /// Text content
    #[serde(default)]
    pub content: Option<String>,
}

/// Client errors
#[derive(Error, Debug)]
pub enum ClientError {
    #[error("Request failed: {0}")]
    Request(#[from] reqwest::Error),

    #[error("Authentication failed: {0}")]
    Authentication(String),

    #[error("Invalid configuration: {0}")]
    Configuration(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("HTTP {status}: {message}")]
    Http { status: StatusCode, message: String },

    #[error("Max retries exceeded")]
    MaxRetriesExceeded,

    #[error("Invalid URL: {0}")]
    InvalidUrl(#[from] url::ParseError),

    #[error("Streaming error: {0}")]
    Streaming(String),

    #[error("Timeout")]
    Timeout,
}

/// Client result type
pub type Result<T> = std::result::Result<T, ClientError>;

/// TrustformeRS client
#[derive(Clone)]
pub struct TrustformersClient {
    base_url: Url,
    http_client: HttpClient,
    config: ClientConfig,
    authenticator: Option<Arc<dyn Authenticator>>,
}

impl TrustformersClient {
    /// Create a new client builder
    pub fn builder(base_url: impl AsRef<str>) -> ClientBuilder {
        ClientBuilder::new(base_url)
    }

    /// Perform inference
    pub async fn inference(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let response = self
            .make_request(Method::POST, "/v1/inference", Some(&request))
            .await?;

        self.handle_response(response).await
    }

    /// Perform batch inference
    pub async fn batch_inference(&self, request: BatchInferenceRequest) -> Result<BatchInferenceResponse> {
        let response = self
            .make_request(Method::POST, "/v1/inference/batch", Some(&request))
            .await?;

        self.handle_response(response).await
    }

    /// Stream inference responses
    pub async fn stream_inference(
        &self,
        mut request: InferenceRequest,
    ) -> Result<impl Stream<Item = Result<StreamingChunk>>> {
        // Enable streaming
        if request.options.is_none() {
            request.options = Some(InferenceOptions::new());
        }
        request.options.as_mut().unwrap().stream = Some(true);

        let response = self
            .make_request_streaming(Method::POST, "/v1/inference/stream", Some(&request))
            .await?;

        Ok(self.handle_streaming_response(response))
    }

    /// Check health status
    pub async fn health(&self) -> Result<HealthStatus> {
        let response = self.make_request::<()>(Method::GET, "/health", None).await?;
        self.handle_response(response).await
    }

    /// Get detailed health status
    pub async fn detailed_health(&self) -> Result<HealthStatus> {
        let response = self
            .make_request::<()>(Method::GET, "/health/detailed", None)
            .await?;
        self.handle_response(response).await
    }

    /// List available models
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let response = self
            .make_request::<()>(Method::GET, "/v1/models", None)
            .await?;
        self.handle_response(response).await
    }

    /// Get model information
    pub async fn get_model(&self, model_id: &str) -> Result<ModelInfo> {
        let path = format!("/v1/models/{}", model_id);
        let response = self.make_request::<()>(Method::GET, &path, None).await?;
        self.handle_response(response).await
    }

    /// Get server metrics
    pub async fn get_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
        let response = self.make_request::<()>(Method::GET, "/metrics", None).await?;
        self.handle_response(response).await
    }

    /// Make an HTTP request
    async fn make_request<T: Serialize>(
        &self,
        method: Method,
        path: &str,
        body: Option<&T>,
    ) -> Result<Response> {
        let mut url = self.base_url.clone();
        url.set_path(path);

        let mut request_builder = self.http_client.request(method, url);

        // Add default headers
        for (name, value) in &self.config.default_headers {
            request_builder = request_builder.header(name, value);
        }

        // Add user agent
        request_builder = request_builder.header("User-Agent", &self.config.user_agent);

        // Add body if provided
        if let Some(body) = body {
            request_builder = request_builder.json(body);
        }

        // Apply authentication
        if let Some(authenticator) = &self.authenticator {
            request_builder = authenticator.apply(request_builder).await?;
        }

        // Execute with retry logic
        self.execute_with_retry(request_builder).await
    }

    /// Make a streaming HTTP request
    async fn make_request_streaming<T: Serialize>(
        &self,
        method: Method,
        path: &str,
        body: Option<&T>,
    ) -> Result<Response> {
        let mut url = self.base_url.clone();
        url.set_path(path);

        let mut request_builder = self.http_client.request(method, url);

        // Add headers for streaming
        request_builder = request_builder
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("User-Agent", &self.config.user_agent);

        // Add default headers
        for (name, value) in &self.config.default_headers {
            request_builder = request_builder.header(name, value);
        }

        // Add body if provided
        if let Some(body) = body {
            request_builder = request_builder.json(body);
        }

        // Apply authentication
        if let Some(authenticator) = &self.authenticator {
            request_builder = authenticator.apply(request_builder).await?;
        }

        // Execute request (no retry for streaming)
        let response = request_builder.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ClientError::Http {
                status,
                message: text,
            });
        }

        Ok(response)
    }

    /// Execute request with retry logic
    async fn execute_with_retry(&self, request_builder: RequestBuilder) -> Result<Response> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = std::cmp::min(
                    Duration::from_millis(
                        (self.config.initial_retry_delay.as_millis() as f64
                            * self.config.backoff_multiplier.powi(attempt as i32 - 1)) as u64,
                    ),
                    self.config.max_retry_delay,
                );

                if self.config.debug {
                    log::debug!("Retrying request after {:?} (attempt {}/{})", delay, attempt + 1, self.config.max_retries + 1);
                }

                tokio::time::sleep(delay).await;
            }

            // Clone the request builder for this attempt
            let request = match request_builder.try_clone() {
                Some(req) => req,
                None => return Err(ClientError::Configuration("Request body is not cloneable".to_string())),
            };

            match request.send().await {
                Ok(response) => {
                    let status = response.status();

                    // Check if we should retry based on status code
                    if self.config.retryable_status_codes.contains(&status.as_u16()) && attempt < self.config.max_retries {
                        last_error = Some(ClientError::Http {
                            status,
                            message: format!("Retryable error: {}", status),
                        });
                        continue;
                    }

                    return Ok(response);
                }
                Err(e) => {
                    last_error = Some(ClientError::Request(e));
                    if attempt == self.config.max_retries {
                        break;
                    }
                }
            }
        }

        Err(last_error.unwrap_or(ClientError::MaxRetriesExceeded))
    }

    /// Handle HTTP response
    async fn handle_response<T: for<'de> Deserialize<'de>>(&self, response: Response) -> Result<T> {
        let status = response.status();

        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(ClientError::Http {
                status,
                message: text,
            });
        }

        let text = response.text().await?;

        if self.config.debug {
            log::debug!("Response body: {}", text);
        }

        serde_json::from_str(&text).map_err(ClientError::Serialization)
    }

    /// Handle streaming response
    fn handle_streaming_response(&self, response: Response) -> impl Stream<Item = Result<StreamingChunk>> {
        let debug = self.config.debug;

        response.bytes_stream().map(move |chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    let text = String::from_utf8_lossy(&chunk);

                    if debug {
                        log::debug!("Streaming chunk: {}", text);
                    }

                    // Parse Server-Sent Events format
                    for line in text.lines() {
                        if line.starts_with("data: ") {
                            let data = &line[6..]; // Remove "data: " prefix
                            if data == "[DONE]" {
                                continue; // End of stream marker
                            }

                            match serde_json::from_str::<StreamingChunk>(data) {
                                Ok(chunk) => return Ok(chunk),
                                Err(e) => return Err(ClientError::Serialization(e)),
                            }
                        }
                    }

                    // If no data line found, it might be a keep-alive or other event
                    Err(ClientError::Streaming("No data in chunk".to_string()))
                }
                Err(e) => Err(ClientError::Request(e)),
            }
        })
    }
}

/// Client builder
pub struct ClientBuilder {
    base_url: String,
    config: ClientConfig,
    authenticator: Option<Arc<dyn Authenticator>>,
    http_client_builder: Option<reqwest::ClientBuilder>,
}

impl ClientBuilder {
    /// Create a new client builder
    pub fn new(base_url: impl AsRef<str>) -> Self {
        Self {
            base_url: base_url.as_ref().to_string(),
            config: ClientConfig::default(),
            authenticator: None,
            http_client_builder: None,
        }
    }

    /// Set client configuration
    pub fn config(mut self, config: ClientConfig) -> Self {
        self.config = config;
        self
    }

    /// Set timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Set user agent
    pub fn user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.config.user_agent = user_agent.into();
        self
    }

    /// Enable debug mode
    pub fn debug(mut self, debug: bool) -> Self {
        self.config.debug = debug;
        self
    }

    /// Set max retries
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.config.max_retries = max_retries;
        self
    }

    /// Set authenticator
    pub fn authenticator(mut self, authenticator: impl Authenticator + 'static) -> Self {
        self.authenticator = Some(Arc::new(authenticator));
        self
    }

    /// Set API key authentication
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.authenticator = Some(Arc::new(ApiKeyAuth::new(api_key)));
        self
    }

    /// Set JWT authentication
    pub fn jwt_token(mut self, token: impl Into<String>) -> Self {
        self.authenticator = Some(Arc::new(JwtAuth::new(token)));
        self
    }

    /// Set OAuth2 authentication
    #[cfg(feature = "oauth2")]
    pub fn oauth2(
        mut self,
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
        token_url: impl AsRef<str>,
    ) -> Result<Self> {
        let oauth2_auth = OAuth2Auth::new(client_id, client_secret, token_url)?;
        self.authenticator = Some(Arc::new(oauth2_auth));
        Ok(self)
    }

    /// Set custom HTTP client builder
    pub fn http_client_builder(mut self, builder: reqwest::ClientBuilder) -> Self {
        self.http_client_builder = Some(builder);
        self
    }

    /// Add default header
    pub fn default_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        let header_name = HeaderName::from_bytes(name.into().as_bytes())
            .map_err(|e| ClientError::Configuration(format!("Invalid header name: {}", e)))?;
        let header_value = HeaderValue::from_str(&value.into())
            .map_err(|e| ClientError::Configuration(format!("Invalid header value: {}", e)))?;

        self.config.default_headers.insert(header_name, header_value);
        Ok(self)
    }

    /// Build the client
    pub fn build(self) -> Result<TrustformersClient> {
        let base_url = Url::parse(&self.base_url)?;

        let http_client = if let Some(builder) = self.http_client_builder {
            builder
        } else {
            HttpClient::builder()
        }
        .timeout(self.config.timeout)
        .build()?;

        Ok(TrustformersClient {
            base_url,
            http_client,
            config: self.config,
            authenticator: self.authenticator,
        })
    }
}

// Async trait import
use async_trait::async_trait;