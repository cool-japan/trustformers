use axum::{
#![allow(unused_variables)]
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod models;
mod handlers;
mod error;

use models::ModelManager;
use error::{AppError, AppResult};

#[derive(Clone)]
struct AppState {
    model_manager: Arc<ModelManager>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "trustformers_server=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting TrustformeRS REST API Server");

    // Initialize model manager
    let model_manager = Arc::new(ModelManager::new().await?);

    // Load default models
    info!("Loading default models...");
    model_manager.load_model("bert-base-uncased", "bert").await?;
    info!("Models loaded successfully");

    // Create app state
    let app_state = AppState { model_manager };

    // Build router
    let app = Router::new()
        // Health check
        .route("/health", get(health_check))

        // Model management
        .route("/models", get(handlers::list_models))
        .route("/models/:model_id", get(handlers::get_model_info))
        .route("/models", post(handlers::load_model))
        .route("/models/:model_id", delete(handlers::unload_model))

        // Inference endpoints
        .route("/predict/classification", post(handlers::text_classification))
        .route("/predict/generation", post(handlers::text_generation))
        .route("/predict/qa", post(handlers::question_answering))
        .route("/predict/ner", post(handlers::token_classification))

        // Batch inference
        .route("/predict/batch", post(handlers::batch_inference))

        // Add middleware
        .layer(CompressionLayer::new())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(app_state);

    // Start server
    let addr = "0.0.0.0:8080";
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("Server listening on {}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

use axum::http::Method;

async fn delete(handler: impl Into<axum::routing::MethodRouter<AppState>>) -> axum::routing::MethodRouter<AppState> {
    axum::routing::on(Method::DELETE, handler)
}