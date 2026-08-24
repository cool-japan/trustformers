//! TrustformeRS REST API server example.
//!
//! Loads real checkpoints from local directories through `trustformers`'s
//! task-specific `AutoModelFor*` loaders and serves classification, NER,
//! question-answering and text-generation over HTTP. See `README.md` for the
//! request/response shapes and — importantly — what this server does *not* do
//! (it does not download model weights, and it refuses to serve a
//! randomly-initialised model rather than silently doing so).

use std::sync::Arc;

use axum::{
    routing::{delete, get, post},
    Router,
};
use serde::Serialize;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod error;
mod handlers;
mod models;

use models::{ModelManager, TaskKind};

#[derive(Clone)]
pub struct AppState {
    model_manager: Arc<ModelManager>,
}

const DEFAULT_PORT: u16 = 8080;
const DEFAULT_MAX_MODELS: usize = 10;

/// Loads `model_name` (when given) for `task_str` before the server starts
/// accepting requests.
///
/// Preloading is opt-in: unlike the previous version of this example, nothing
/// is loaded when `model_name` is `None`. There is no checkpoint this server
/// could assume exists on every deployment, so loading one unconditionally at
/// startup would just fail every fresh checkout.
///
/// Split out from [`preload_model`] (its thin env-var-reading wrapper) so
/// tests can exercise it directly with explicit arguments instead of mutating
/// process-global environment variables, which `cargo test`'s default
/// parallel execution makes racy.
fn preload_model_from(
    model_manager: &ModelManager,
    model_name: Option<String>,
    task_str: Option<String>,
    num_labels: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(model_name) = model_name else {
        return Ok(());
    };

    let task_str = task_str.ok_or(
        "PRELOAD_MODEL_NAME was set but PRELOAD_MODEL_TASK was not; specify the task \
         (text-classification, token-classification, question-answering, or text-generation) the \
         preloaded checkpoint should serve",
    )?;
    let task: TaskKind = serde_json::from_value(serde_json::Value::String(task_str.clone()))
        .map_err(|_| format!("PRELOAD_MODEL_TASK `{task_str}` is not a recognised task"))?;

    info!(model_name = %model_name, task = task.as_str(), "preloading model");
    model_manager.load_model(&model_name, task, num_labels, None)?;
    Ok(())
}

/// Reads `PRELOAD_MODEL_NAME`/`PRELOAD_MODEL_TASK`/`PRELOAD_NUM_LABELS` and
/// delegates to [`preload_model_from`].
fn preload_model(model_manager: &ModelManager) -> Result<(), Box<dyn std::error::Error>> {
    preload_model_from(
        model_manager,
        std::env::var("PRELOAD_MODEL_NAME").ok(),
        std::env::var("PRELOAD_MODEL_TASK").ok(),
        std::env::var("PRELOAD_NUM_LABELS").ok().and_then(|v| v.parse::<usize>().ok()),
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "trustformers_server=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("starting TrustformeRS REST API server");

    let max_models = std::env::var("MAX_MODELS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_MAX_MODELS);
    // A relative `model_name` in `POST /models` resolves against this
    // directory when set, so a deployment can mount one volume of checkpoints
    // and refer to each by name instead of a full in-container path.
    let model_manager = match std::env::var("MODEL_CACHE_DIR") {
        Ok(dir) if !dir.trim().is_empty() => {
            info!(model_cache_dir = %dir, "resolving relative model names against MODEL_CACHE_DIR");
            ModelManager::with_model_root(max_models, std::path::PathBuf::from(dir))
        },
        _ => ModelManager::new(max_models),
    };
    preload_model(&model_manager)?;
    let model_manager = Arc::new(model_manager);

    let app_state = AppState { model_manager };

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/models", get(handlers::list_models))
        .route("/models", post(handlers::load_model))
        .route("/models/{model_id}", get(handlers::get_model_info))
        .route("/models/{model_id}", delete(handlers::unload_model))
        .route("/predict/classification", post(handlers::text_classification))
        .route("/predict/generation", post(handlers::text_generation))
        .route("/predict/qa", post(handlers::question_answering))
        .route("/predict/ner", post(handlers::token_classification))
        .route("/predict/batch", post(handlers::batch_inference))
        // No response-compression layer: see the comment on `tower-http` in
        // Cargo.toml — every compression backend it offers routes through a
        // crate this repository's `deny.toml` bans (`brotli`/`flate2`/
        // `zstd`), so responses here are sent uncompressed rather than
        // through a banned dependency.
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .layer(TraceLayer::new_for_http())
        .with_state(app_state);

    let port = std::env::var("PORT").ok().and_then(|v| v.parse::<u16>().ok()).unwrap_or(DEFAULT_PORT);
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!(%addr, "server listening");

    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check() -> axum::response::Json<HealthResponse> {
    axum::response::Json(HealthResponse { status: "healthy".to_string(), version: env!("CARGO_PKG_VERSION").to_string() })
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    // These exercise `preload_model_from` with explicit arguments rather than
    // `preload_model` with real environment variables: `cargo test` runs
    // tests in parallel within one process by default, and mutating
    // process-global env vars from multiple tests is racy.

    #[test]
    fn preload_is_a_no_op_when_no_model_name_is_given() {
        let manager = ModelManager::new(4);
        preload_model_from(&manager, None, None, None).expect("no-op preload must not error");
        assert!(manager.list().is_empty());
    }

    #[test]
    fn preload_requires_a_task_when_a_name_is_given() {
        let manager = ModelManager::new(4);
        let result =
            preload_model_from(&manager, Some("/nonexistent-for-this-test".to_string()), None, None);
        assert!(result.is_err(), "a name with no task must be rejected before any load is attempted");
    }

    #[test]
    fn preload_rejects_an_unrecognised_task_before_touching_the_filesystem() {
        let manager = ModelManager::new(4);
        let result = preload_model_from(
            &manager,
            Some("/nonexistent-for-this-test".to_string()),
            Some("not-a-real-task".to_string()),
            None,
        );
        assert!(result.is_err(), "an unrecognised task string must be rejected");
    }
}
