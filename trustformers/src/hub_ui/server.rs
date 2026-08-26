//! The `HubUiServer` (axum router + listener) and its top-level start functions.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::TrustformersError;
use axum::{
    routing::{delete, get, post, put},
    Router,
};
use std::path::PathBuf;
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;

use super::handlers::{
    compare_versions, create_repository, create_version, delete_repository, delete_version,
    download_version, get_repository, get_version, list_repositories, list_versions, ui_compare,
    ui_home, ui_repository, ui_version, update_repository, update_version,
};
use super::state::{HubUiConfig, HubUiState};

/// Hub UI server
pub struct HubUiServer {
    /// Server state
    pub(super) state: HubUiState,
    /// Axum router
    pub(super) router: Router,
}
impl HubUiServer {
    /// Create a new Hub UI server
    pub fn new(config: HubUiConfig, cache_dir: PathBuf) -> Self {
        let state = HubUiState::new(config.clone(), cache_dir);
        let router = Self::create_router(state.clone());
        Self { state, router }
    }
    /// Create the Axum router with all routes
    pub(super) fn create_router(state: HubUiState) -> Router {
        let api_routes = Router::new()
            .route("/repositories", get(list_repositories))
            .route("/repositories/:model_id", get(get_repository))
            .route("/repositories/:model_id", post(create_repository))
            .route("/repositories/:model_id", put(update_repository))
            .route("/repositories/:model_id", delete(delete_repository))
            .route("/repositories/:model_id/versions", get(list_versions))
            .route(
                "/repositories/:model_id/versions/:version",
                get(get_version),
            )
            .route(
                "/repositories/:model_id/versions/:version",
                post(create_version),
            )
            .route(
                "/repositories/:model_id/versions/:version",
                put(update_version),
            )
            .route(
                "/repositories/:model_id/versions/:version",
                delete(delete_version),
            )
            .route(
                "/repositories/:model_id/compare/:from/:to",
                get(compare_versions),
            )
            .route(
                "/repositories/:model_id/download/:version",
                get(download_version),
            )
            .with_state(state.clone());
        let ui_routes = Router::new()
            .route("/", get(ui_home))
            .route("/repository/:model_id", get(ui_repository))
            .route("/repository/:model_id/version/:version", get(ui_version))
            .route("/repository/:model_id/compare/:from/:to", get(ui_compare))
            .with_state(state);
        Router::new()
            .nest("/api/v1", api_routes)
            .nest("/ui", ui_routes)
            .layer(ServiceBuilder::new().layer(CorsLayer::permissive()).into_inner())
    }
    /// Start the server
    pub async fn start(self) -> Result<(), TrustformersError> {
        let addr = format!(
            "{}:{}",
            self.state.config.bind_address, self.state.config.port
        );
        let listener = TcpListener::bind(&addr).await.map_err(|e| TrustformersError::Network {
            message: format!("Failed to bind to {}: {}", addr, e),
            url: Some(addr.clone()),
            status_code: None,
            suggestion: Some(
                "Check if the port is already in use or try a different port".to_string(),
            ),
            retry_recommended: true,
        })?;
        tracing::info!("🚀 TrustformeRS Hub UI server running on http://{}", addr);
        tracing::info!("📊 Repository management: http://{}/ui/", addr);
        tracing::info!("🔌 API endpoint: http://{}/api/v1/", addr);
        axum::serve(listener, self.router)
            .await
            .map_err(|e| TrustformersError::Network {
                message: format!("Server error: {}", e),
                url: Some(addr),
                status_code: None,
                suggestion: Some("Check server configuration and network connectivity".to_string()),
                retry_recommended: true,
            })?;
        Ok(())
    }
}
/// Start the Hub UI server with default configuration
pub async fn start_hub_ui() -> Result<(), TrustformersError> {
    start_hub_ui_with_config(HubUiConfig::default()).await
}

/// Start the Hub UI server with custom configuration
pub async fn start_hub_ui_with_config(config: HubUiConfig) -> Result<(), TrustformersError> {
    let cache_dir = crate::hub::get_cache_dir().map_err(|e| TrustformersError::AutoConfig {
        message: format!("Failed to get cache directory: {}", e),
        config_type: "cache_directory".to_string(),
        suggestion: Some(
            "Check TRUSTFORMERS_CACHE environment variable or home directory permissions"
                .to_string(),
        ),
        recovery_actions: vec![],
    })?;
    let server = HubUiServer::new(config, cache_dir);
    server.start().await
}
