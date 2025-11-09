//! OpenAPI/Swagger documentation for TrustformeRS Serve API
//!
//! This module provides comprehensive OpenAPI 3.0 documentation for all API endpoints,
//! including request/response schemas, authentication, and interactive Swagger UI.

use utoipa::{
    openapi::security::{ApiKey, ApiKeyValue, SecurityScheme},
    Modify, OpenApi,
};

/// OpenAPI documentation for TrustformeRS Serve API
#[derive(OpenApi)]
#[openapi(
    paths(
        crate::server::health_check,
        crate::server::detailed_health_check,
        crate::server::readiness_check,
        crate::server::liveness_check,
        crate::server::inference_endpoint,
        crate::server::batch_inference_endpoint,
        crate::server::get_stats,
        crate::server::get_config,
        crate::server::metrics_endpoint,
        crate::server::long_poll_endpoint,
        crate::server::poll_stats_endpoint,
        crate::server::shadow_stats_endpoint,
        crate::server::shadow_results_endpoint,
        crate::server::shadow_comparison_endpoint,
    ),
    components(
        schemas(
            crate::server::HealthResponse,
            crate::server::DetailedHealthResponse,
            crate::server::SystemHealthInfo,
            crate::server::ServiceHealthInfo,
            crate::server::InferenceRequest,
            crate::server::InferenceResponse,
            crate::server::BatchInferenceRequest,
            crate::server::BatchInferenceResponse,
            crate::server::StatsResponse,
            crate::server::FailoverRequest,
            crate::polling::LongPollResponse,
            crate::polling::LongPollingStats,
            crate::shadow::ShadowStats,
            crate::shadow::ShadowComparison,
            crate::ServerConfig,
            ErrorResponse,
        )
    ),
    modifiers(&SecurityAddon),
    tags(
        (name = "health", description = "Health check and monitoring endpoints"),
        (name = "inference", description = "Model inference endpoints"),
        (name = "admin", description = "Administrative endpoints"),
        (name = "monitoring", description = "Metrics and observability endpoints"),
        (name = "streaming", description = "Real-time streaming endpoints"),
        (name = "shadow", description = "Shadow testing endpoints"),
        (name = "polling", description = "Long polling endpoints"),
    ),
    info(
        title = "TrustformeRS Serve API",
        version = "1.0.0",
        description = "High-performance inference server for transformer models with advanced batching, caching, and observability features.",
        contact(
            name = "TrustformeRS Team",
            url = "https://github.com/cool-japan/trustformers",
            email = "team@trustformers.com"
        ),
        license(
            name = "MIT OR Apache-2.0",
            url = "https://github.com/cool-japan/trustformers/blob/main/LICENSE"
        )
    ),
    servers(
        (url = "http://localhost:8080", description = "Local development server"),
        (url = "https://api.trustformers.com", description = "Production server")
    )
)]
pub struct ApiDoc;

/// Security addon for API authentication
pub struct SecurityAddon;

impl Modify for SecurityAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        if let Some(components) = openapi.components.as_mut() {
            components.add_security_scheme(
                "bearer_auth",
                SecurityScheme::ApiKey(ApiKey::Header(ApiKeyValue::new("Authorization"))),
            );
            components.add_security_scheme(
                "api_key",
                SecurityScheme::ApiKey(ApiKey::Header(ApiKeyValue::new("X-API-Key"))),
            );
        }
    }
}

/// Error response
#[derive(utoipa::ToSchema, serde::Serialize)]
#[schema(example = json!({
    "error": "ValidationError",
    "message": "Invalid input parameter",
    "details": "Temperature must be between 0.0 and 1.0",
    "request_id": "550e8400-e29b-41d4-a716-446655440000"
}))]
pub struct ErrorResponse {
    /// Error type
    #[schema(example = "ValidationError")]
    pub error: String,
    /// Error message
    #[schema(example = "Invalid input parameter")]
    pub message: String,
    /// Additional error details
    #[schema(example = "Temperature must be between 0.0 and 1.0")]
    pub details: Option<String>,
    /// Request ID for debugging
    #[schema(example = "550e8400-e29b-41d4-a716-446655440000")]
    pub request_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use utoipa::OpenApi;

    #[test]
    fn test_openapi_generation() {
        let doc = ApiDoc::openapi();
        assert!(!doc.paths.paths.is_empty());
        assert!(doc.info.title == "TrustformeRS Serve API");
    }
}
