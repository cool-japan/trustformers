//! HTTP server for TrustformeRS model serving
//!
//! This module provides a built-in HTTP server for serving TrustformeRS models
//! via REST API endpoints.

use anyhow::anyhow;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::error::{TrustformersError, TrustformersResult};
use crate::{c_str_to_string, result_to_error, string_to_c_str};

/// Global HTTP server manager
static HTTP_SERVER_MANAGER: Lazy<Mutex<HttpServerManager>> =
    Lazy::new(|| Mutex::new(HttpServerManager::new()));

/// HTTP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub request_timeout_ms: u64,
    pub enable_cors: bool,
    pub cors_origins: Vec<String>,
    pub enable_metrics: bool,
    pub metrics_endpoint: String,
    pub health_endpoint: String,
    pub api_prefix: String,
    pub enable_ssl: bool,
    pub ssl_cert_path: Option<String>,
    pub ssl_key_path: Option<String>,
}

impl Default for HttpServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_connections: 100,
            request_timeout_ms: 30000,
            enable_cors: true,
            cors_origins: vec!["*".to_string()],
            enable_metrics: true,
            metrics_endpoint: "/metrics".to_string(),
            health_endpoint: "/health".to_string(),
            api_prefix: "/api/v1".to_string(),
            enable_ssl: false,
            ssl_cert_path: None,
            ssl_key_path: None,
        }
    }
}

/// HTTP server instance
pub struct HttpServer {
    config: HttpServerConfig,
    server_handle: Option<ServerHandle>,
    routes: HashMap<String, RouteHandler>,
    models: HashMap<String, ModelEndpoint>,
    middleware: Vec<MiddlewareHandler>,
    metrics: ServerMetrics,
}

impl std::fmt::Debug for HttpServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpServer")
            .field("config", &self.config)
            .field("server_handle", &self.server_handle)
            .field("routes", &format!("{} routes", self.routes.len()))
            .field("models", &self.models)
            .field(
                "middleware",
                &format!("{} middleware", self.middleware.len()),
            )
            .field("metrics", &self.metrics)
            .finish()
    }
}

/// Server handle for managing the HTTP server
#[derive(Debug)]
struct ServerHandle {
    thread_handle: thread::JoinHandle<()>,
    shutdown_signal: Arc<Mutex<bool>>,
}

/// Route handler function type
type RouteHandler = Box<dyn Fn(&HttpRequest) -> HttpResponse + Send + Sync>;

/// Middleware handler function type
type MiddlewareHandler = Box<dyn Fn(&mut HttpRequest, &mut HttpResponse) -> bool + Send + Sync>;

/// Model endpoint configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelEndpoint {
    pub name: String,
    pub model_path: String,
    pub endpoint_path: String,
    pub max_batch_size: usize,
    pub timeout_ms: u64,
    pub enable_streaming: bool,
    pub rate_limit: Option<RateLimit>,
}

/// Rate limiting configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RateLimit {
    pub requests_per_minute: u32,
    pub burst_limit: u32,
}

/// HTTP request structure
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub query_params: HashMap<String, String>,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
    pub remote_addr: String,
}

/// HTTP response structure
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status_code: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

/// Server metrics
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ServerMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_response_time_ms: f64,
    pub current_connections: u32,
    pub peak_connections: u32,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub model_requests: HashMap<String, u64>,
}

/// Text generation request
#[derive(Debug, Deserialize)]
pub struct TextGenerationRequest {
    pub prompt: String,
    pub max_length: Option<usize>,
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub do_sample: Option<bool>,
    pub num_return_sequences: Option<usize>,
    pub stream: Option<bool>,
}

/// Text generation response
#[derive(Debug, Serialize)]
pub struct TextGenerationResponse {
    pub generated_text: Vec<String>,
    pub processing_time_ms: f64,
    pub model_name: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
}

/// Text classification request
#[derive(Debug, Deserialize)]
pub struct TextClassificationRequest {
    pub text: String,
    pub return_all_scores: Option<bool>,
}

/// Text classification response
#[derive(Debug, Serialize)]
pub struct TextClassificationResponse {
    pub classifications: Vec<ClassificationResult>,
    pub processing_time_ms: f64,
    pub model_name: String,
}

/// Classification result
#[derive(Debug, Serialize)]
pub struct ClassificationResult {
    pub label: String,
    pub score: f64,
}

/// Server manager for handling multiple HTTP servers
#[derive(Debug)]
struct HttpServerManager {
    servers: HashMap<String, HttpServer>,
    next_server_id: usize,
}

impl HttpServerManager {
    fn new() -> Self {
        Self {
            servers: HashMap::new(),
            next_server_id: 1,
        }
    }

    fn create_server(&mut self, config: HttpServerConfig) -> TrustformersResult<String> {
        let server_id = format!("server_{}", self.next_server_id);
        self.next_server_id += 1;

        let server = HttpServer::new(config)?;
        self.servers.insert(server_id.clone(), server);

        Ok(server_id)
    }

    fn get_server_mut(&mut self, server_id: &str) -> Option<&mut HttpServer> {
        self.servers.get_mut(server_id)
    }

    fn remove_server(&mut self, server_id: &str) -> Option<HttpServer> {
        self.servers.remove(server_id)
    }
}

impl HttpServer {
    fn new(config: HttpServerConfig) -> TrustformersResult<Self> {
        Ok(Self {
            config,
            server_handle: None,
            routes: HashMap::new(),
            models: HashMap::new(),
            middleware: Vec::new(),
            metrics: ServerMetrics::default(),
        })
    }

    fn add_model_endpoint(&mut self, endpoint: ModelEndpoint) -> TrustformersResult<()> {
        let full_path = format!("{}{}", self.config.api_prefix, endpoint.endpoint_path);

        // Add text generation endpoint
        if endpoint.name.contains("generation") || endpoint.name.contains("gpt") {
            self.add_text_generation_route(&full_path, endpoint.clone())?;
        }

        // Add text classification endpoint
        if endpoint.name.contains("classification") || endpoint.name.contains("sentiment") {
            self.add_text_classification_route(&full_path, endpoint.clone())?;
        }

        self.models.insert(endpoint.name.clone(), endpoint);
        Ok(())
    }

    fn add_text_generation_route(
        &mut self,
        path: &str,
        endpoint: ModelEndpoint,
    ) -> TrustformersResult<()> {
        let endpoint_name = endpoint.name.clone();
        let handler: RouteHandler = Box::new(move |request: &HttpRequest| {
            Self::handle_text_generation(request, &endpoint_name)
        });

        self.routes.insert(path.to_string(), handler);
        Ok(())
    }

    fn add_text_classification_route(
        &mut self,
        path: &str,
        endpoint: ModelEndpoint,
    ) -> TrustformersResult<()> {
        let endpoint_name = endpoint.name.clone();
        let handler: RouteHandler = Box::new(move |request: &HttpRequest| {
            Self::handle_text_classification(request, &endpoint_name)
        });

        self.routes.insert(path.to_string(), handler);
        Ok(())
    }

    fn handle_text_generation(request: &HttpRequest, model_name: &str) -> HttpResponse {
        // Parse request body
        let generation_request: TextGenerationRequest = match serde_json::from_slice(&request.body)
        {
            Ok(req) => req,
            Err(e) => {
                return Self::create_error_response(400, &format!("Invalid request body: {}", e));
            },
        };

        // Simulate text generation (in real implementation, this would use actual models)
        let start_time = std::time::Instant::now();

        let generated_text = Self::simulate_text_generation(&generation_request);
        let processing_time = start_time.elapsed().as_millis() as f64;

        let response = TextGenerationResponse {
            generated_text,
            processing_time_ms: processing_time,
            model_name: model_name.to_string(),
            prompt_tokens: generation_request.prompt.split_whitespace().count(),
            generated_tokens: 50, // Simulated
        };

        Self::create_json_response(200, &response)
    }

    fn handle_text_classification(request: &HttpRequest, model_name: &str) -> HttpResponse {
        // Parse request body
        let classification_request: TextClassificationRequest =
            match serde_json::from_slice(&request.body) {
                Ok(req) => req,
                Err(e) => {
                    return Self::create_error_response(
                        400,
                        &format!("Invalid request body: {}", e),
                    );
                },
            };

        // Simulate text classification
        let start_time = std::time::Instant::now();

        let classifications = Self::simulate_text_classification(&classification_request);
        let processing_time = start_time.elapsed().as_millis() as f64;

        let response = TextClassificationResponse {
            classifications,
            processing_time_ms: processing_time,
            model_name: model_name.to_string(),
        };

        Self::create_json_response(200, &response)
    }

    fn simulate_text_generation(request: &TextGenerationRequest) -> Vec<String> {
        // Simulate text generation based on prompt
        let num_sequences = request.num_return_sequences.unwrap_or(1);
        let mut results = Vec::new();

        for i in 0..num_sequences {
            let generated = format!(
                "{} This is simulated generated text continuation #{}",
                request.prompt,
                i + 1
            );
            results.push(generated);
        }

        results
    }

    fn simulate_text_classification(
        request: &TextClassificationRequest,
    ) -> Vec<ClassificationResult> {
        // Simulate sentiment analysis
        let text_lower = request.text.to_lowercase();

        let positive_score = if text_lower.contains("good")
            || text_lower.contains("great")
            || text_lower.contains("excellent")
            || text_lower.contains("love")
        {
            0.85
        } else if text_lower.contains("bad")
            || text_lower.contains("terrible")
            || text_lower.contains("awful")
            || text_lower.contains("hate")
        {
            0.15
        } else {
            0.50
        };

        let negative_score = 1.0 - positive_score;

        vec![
            ClassificationResult {
                label: "POSITIVE".to_string(),
                score: positive_score,
            },
            ClassificationResult {
                label: "NEGATIVE".to_string(),
                score: negative_score,
            },
        ]
    }

    fn create_json_response<T: Serialize>(status_code: u16, data: &T) -> HttpResponse {
        let body = serde_json::to_vec(data).unwrap_or_default();
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("Content-Length".to_string(), body.len().to_string());

        HttpResponse {
            status_code,
            headers,
            body,
        }
    }

    fn create_error_response(status_code: u16, message: &str) -> HttpResponse {
        let error_body = serde_json::json!({
            "error": message,
            "status": status_code
        });

        Self::create_json_response(status_code, &error_body)
    }

    fn start(&mut self) -> TrustformersResult<()> {
        if self.server_handle.is_some() {
            return Err(anyhow!("Server is already running").into());
        }

        // Add default routes
        self.add_default_routes()?;

        // Simulate server startup (in real implementation, this would bind to socket)
        let shutdown_signal = Arc::new(Mutex::new(false));
        let shutdown_signal_clone = shutdown_signal.clone();
        let config = self.config.clone();

        let thread_handle = thread::spawn(move || {
            Self::run_server_loop(config, shutdown_signal_clone);
        });

        self.server_handle = Some(ServerHandle {
            thread_handle,
            shutdown_signal,
        });

        Ok(())
    }

    fn add_default_routes(&mut self) -> TrustformersResult<()> {
        // Health check endpoint
        let health_handler: RouteHandler = Box::new(|_request| {
            let health_response = serde_json::json!({
                "status": "healthy",
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "version": env!("CARGO_PKG_VERSION")
            });
            Self::create_json_response(200, &health_response)
        });
        self.routes.insert(self.config.health_endpoint.clone(), health_handler);

        // Metrics endpoint
        if self.config.enable_metrics {
            let metrics_handler: RouteHandler = Box::new(|_request| {
                let metrics_response = serde_json::json!({
                    "requests_total": 0,
                    "requests_successful": 0,
                    "requests_failed": 0,
                    "response_time_avg_ms": 0.0,
                    "connections_current": 0,
                    "connections_peak": 0
                });
                Self::create_json_response(200, &metrics_response)
            });
            self.routes.insert(self.config.metrics_endpoint.clone(), metrics_handler);
        }

        Ok(())
    }

    fn run_server_loop(config: HttpServerConfig, shutdown_signal: Arc<Mutex<bool>>) {
        // Simulate server loop (in real implementation, this would handle HTTP requests)
        eprintln!("HTTP server started on {}:{}", config.host, config.port);
        eprintln!("Health endpoint: {}", config.health_endpoint);
        eprintln!("API prefix: {}", config.api_prefix);

        loop {
            // Check shutdown signal
            if let Ok(shutdown) = shutdown_signal.lock() {
                if *shutdown {
                    break;
                }
            }

            // Simulate request processing
            thread::sleep(Duration::from_millis(100));
        }

        eprintln!("HTTP server stopped");
    }

    fn stop(&mut self) -> TrustformersResult<()> {
        if let Some(handle) = self.server_handle.take() {
            // Signal shutdown
            if let Ok(mut shutdown) = handle.shutdown_signal.lock() {
                *shutdown = true;
            }

            // Wait for server thread to finish
            handle
                .thread_handle
                .join()
                .map_err(|_| anyhow!("Failed to join server thread"))?;
        }

        Ok(())
    }

    fn get_metrics(&self) -> &ServerMetrics {
        &self.metrics
    }
}

// C API exports for HTTP server

/// Create HTTP server with default configuration
#[no_mangle]
pub extern "C" fn trustformers_http_server_create(
    server_id: *mut *mut c_char,
) -> TrustformersError {
    if server_id.is_null() {
        return TrustformersError::NullPointer;
    }

    let mut manager = match HTTP_SERVER_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let config = HttpServerConfig::default();
    let id = match manager.create_server(config) {
        Ok(id) => id,
        Err(_) => return TrustformersError::RuntimeError,
    };

    unsafe {
        *server_id = string_to_c_str(id);
    }

    TrustformersError::Success
}

/// Create HTTP server with custom configuration
#[no_mangle]
pub extern "C" fn trustformers_http_server_create_with_config(
    config_json: *const c_char,
    server_id: *mut *mut c_char,
) -> TrustformersError {
    if config_json.is_null() || server_id.is_null() {
        return TrustformersError::NullPointer;
    }

    let config_str = match c_str_to_string(config_json) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let config: HttpServerConfig = match serde_json::from_str(&config_str) {
        Ok(config) => config,
        Err(_) => return TrustformersError::SerializationError,
    };

    let mut manager = match HTTP_SERVER_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let id = match manager.create_server(config) {
        Ok(id) => id,
        Err(_) => return TrustformersError::RuntimeError,
    };

    unsafe {
        *server_id = string_to_c_str(id);
    }

    TrustformersError::Success
}

/// Add model endpoint to HTTP server
#[no_mangle]
pub extern "C" fn trustformers_http_server_add_model(
    server_id: *const c_char,
    endpoint_json: *const c_char,
) -> TrustformersError {
    if server_id.is_null() || endpoint_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let server_id_str = match c_str_to_string(server_id) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let endpoint_str = match c_str_to_string(endpoint_json) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let endpoint: ModelEndpoint = match serde_json::from_str(&endpoint_str) {
        Ok(endpoint) => endpoint,
        Err(_) => return TrustformersError::SerializationError,
    };

    let mut manager = match HTTP_SERVER_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let server = match manager.get_server_mut(&server_id_str) {
        Some(server) => server,
        None => return TrustformersError::InvalidParameter,
    };

    match server.add_model_endpoint(endpoint) {
        Ok(_) => TrustformersError::Success,
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Start HTTP server
#[no_mangle]
pub extern "C" fn trustformers_http_server_start(server_id: *const c_char) -> TrustformersError {
    if server_id.is_null() {
        return TrustformersError::NullPointer;
    }

    let server_id_str = match c_str_to_string(server_id) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let mut manager = match HTTP_SERVER_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let server = match manager.get_server_mut(&server_id_str) {
        Some(server) => server,
        None => return TrustformersError::InvalidParameter,
    };

    match server.start() {
        Ok(_) => TrustformersError::Success,
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Stop HTTP server
#[no_mangle]
pub extern "C" fn trustformers_http_server_stop(server_id: *const c_char) -> TrustformersError {
    if server_id.is_null() {
        return TrustformersError::NullPointer;
    }

    let server_id_str = match c_str_to_string(server_id) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let mut manager = match HTTP_SERVER_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let server = match manager.get_server_mut(&server_id_str) {
        Some(server) => server,
        None => return TrustformersError::InvalidParameter,
    };

    match server.stop() {
        Ok(_) => TrustformersError::Success,
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Get HTTP server metrics
#[no_mangle]
pub extern "C" fn trustformers_http_server_get_metrics(
    server_id: *const c_char,
    metrics_json: *mut *mut c_char,
) -> TrustformersError {
    if server_id.is_null() || metrics_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let server_id_str = match c_str_to_string(server_id) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let manager = match HTTP_SERVER_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let server = match manager.servers.get(&server_id_str) {
        Some(server) => server,
        None => return TrustformersError::InvalidParameter,
    };

    let metrics_json_data = match serde_json::to_string_pretty(server.get_metrics()) {
        Ok(json) => json,
        Err(_) => return TrustformersError::SerializationError,
    };

    unsafe {
        *metrics_json = string_to_c_str(metrics_json_data);
    }

    TrustformersError::Success
}

/// Destroy HTTP server
#[no_mangle]
pub extern "C" fn trustformers_http_server_destroy(server_id: *const c_char) -> TrustformersError {
    if server_id.is_null() {
        return TrustformersError::NullPointer;
    }

    let server_id_str = match c_str_to_string(server_id) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let mut manager = match HTTP_SERVER_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    if let Some(mut server) = manager.remove_server(&server_id_str) {
        let _ = server.stop(); // Ensure server is stopped before destruction
    }

    TrustformersError::Success
}
