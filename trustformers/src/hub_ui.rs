// Copyright (c) 2024 TrustformeRS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Model versioning UI for TrustformeRS Hub integration
//!
//! This module provides a web-based user interface for managing model versions,
//! tracking changes, comparing models, and visualizing version history.

use crate::error::TrustformersError;
use crate::hub::DownloadStats;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Html, Json},
    routing::{delete, get, post, put},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;

/// Model version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Version identifier (e.g., "v1.0.0", "main", commit hash)
    pub version: String,
    /// Human-readable version name
    pub name: Option<String>,
    /// Version description
    pub description: Option<String>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last modified timestamp
    pub modified_at: u64,
    /// Version author
    pub author: Option<String>,
    /// Version tags (e.g., "stable", "experimental", "deprecated")
    pub tags: Vec<String>,
    /// Model performance metrics
    pub metrics: Option<ModelMetrics>,
    /// File changes from previous version
    pub changes: Vec<FileChange>,
    /// Parent version (for tracking lineage)
    pub parent_version: Option<String>,
    /// Download statistics
    pub download_stats: Option<DownloadStats>,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Checksum for integrity verification
    pub checksum: Option<String>,
    /// Status of the version
    pub status: VersionStatus,
    /// Compatibility information
    pub compatibility: CompatibilityInfo,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Accuracy metrics
    pub accuracy: Option<f64>,
    /// Loss metrics
    pub loss: Option<f64>,
    /// Inference speed (tokens/second)
    pub inference_speed: Option<f64>,
    /// Memory usage (MB)
    pub memory_usage: Option<f64>,
    /// Model size (parameters)
    pub parameter_count: Option<u64>,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
    /// Benchmark results
    pub benchmarks: Vec<BenchmarkResult>,
}

/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Score
    pub score: f64,
    /// Unit of measurement
    pub unit: String,
    /// Benchmark timestamp
    pub timestamp: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// File change information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChange {
    /// File path
    pub path: String,
    /// Change type
    pub change_type: ChangeType,
    /// Old file size (for modifications)
    pub old_size: Option<u64>,
    /// New file size
    pub new_size: u64,
    /// File checksum
    pub checksum: Option<String>,
    /// Change description
    pub description: Option<String>,
}

/// Type of file change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    /// File was added
    Added,
    /// File was modified
    Modified,
    /// File was deleted
    Deleted,
    /// File was renamed
    Renamed { old_path: String },
}

/// Version status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VersionStatus {
    /// Version is in development
    Development,
    /// Version is stable and ready for use
    Stable,
    /// Version is experimental
    Experimental,
    /// Version is deprecated
    Deprecated,
    /// Version has been archived
    Archived,
}

/// Compatibility information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityInfo {
    /// Framework compatibility (e.g., "trustformers>=0.1.0")
    pub framework_version: Option<String>,
    /// Python version requirements
    pub python_version: Option<String>,
    /// CUDA version requirements
    pub cuda_version: Option<String>,
    /// Hardware requirements
    pub hardware_requirements: Vec<String>,
    /// Breaking changes from previous version
    pub breaking_changes: Vec<String>,
    /// Migration notes
    pub migration_notes: Option<String>,
}

/// Model repository for managing versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRepository {
    /// Model ID
    pub model_id: String,
    /// All versions of the model
    pub versions: HashMap<String, ModelVersion>,
    /// Version history (ordered by creation time)
    pub version_history: Vec<String>,
    /// Repository metadata
    pub metadata: RepositoryMetadata,
    /// Access control
    pub access_control: AccessControl,
}

/// Repository metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryMetadata {
    /// Repository name
    pub name: String,
    /// Repository description
    pub description: Option<String>,
    /// Repository owner
    pub owner: String,
    /// Repository visibility
    pub visibility: Visibility,
    /// Default branch/version
    pub default_version: String,
    /// Repository tags
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last update timestamp
    pub updated_at: u64,
    /// Repository statistics
    pub stats: RepositoryStats,
}

/// Repository visibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Visibility {
    /// Public repository
    Public,
    /// Private repository
    Private,
    /// Organization-only repository
    Organization,
}

/// Repository statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryStats {
    /// Total number of versions
    pub version_count: usize,
    /// Total downloads across all versions
    pub total_downloads: u64,
    /// Repository stars/likes
    pub stars: u64,
    /// Repository forks
    pub forks: u64,
    /// Active contributors
    pub contributors: u64,
    /// Last activity timestamp
    pub last_activity: u64,
}

/// Access control for repositories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControl {
    /// Repository permissions
    pub permissions: HashMap<String, Permission>,
    /// API key for access
    pub api_key: Option<String>,
    /// Access whitelist
    pub whitelist: Vec<String>,
    /// Access blacklist
    pub blacklist: Vec<String>,
}

/// Permission levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Permission {
    /// Read-only access
    Read,
    /// Read and write access
    Write,
    /// Full administrative access
    Admin,
}

/// Hub UI server state
#[derive(Debug, Clone)]
pub struct HubUiState {
    /// Model repositories
    repositories: Arc<Mutex<HashMap<String, ModelRepository>>>,
    /// UI configuration
    config: HubUiConfig,
    /// Cache directory
    cache_dir: PathBuf,
}

/// Hub UI configuration
#[derive(Debug, Clone)]
pub struct HubUiConfig {
    /// Server bind address
    pub bind_address: String,
    /// Server port
    pub port: u16,
    /// Enable authentication
    pub enable_auth: bool,
    /// Static files directory
    pub static_dir: Option<PathBuf>,
    /// Theme configuration
    pub theme: ThemeConfig,
    /// Feature flags
    pub features: FeatureFlags,
}

/// Theme configuration
#[derive(Debug, Clone)]
pub struct ThemeConfig {
    /// Primary color
    pub primary_color: String,
    /// Secondary color
    pub secondary_color: String,
    /// Dark mode support
    pub dark_mode: bool,
    /// Custom CSS
    pub custom_css: Option<String>,
}

/// Feature flags
#[derive(Debug, Clone)]
pub struct FeatureFlags {
    /// Enable model comparison
    pub enable_comparison: bool,
    /// Enable version branching
    pub enable_branching: bool,
    /// Enable performance tracking
    pub enable_performance_tracking: bool,
    /// Enable collaborative features
    pub enable_collaboration: bool,
    /// Enable CI/CD integration
    pub enable_cicd: bool,
}

impl Default for HubUiConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1".to_string(),
            port: 8080,
            enable_auth: false,
            static_dir: None,
            theme: ThemeConfig::default(),
            features: FeatureFlags::default(),
        }
    }
}

impl Default for ThemeConfig {
    fn default() -> Self {
        Self {
            primary_color: "#3b82f6".to_string(),
            secondary_color: "#64748b".to_string(),
            dark_mode: true,
            custom_css: None,
        }
    }
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            enable_comparison: true,
            enable_branching: true,
            enable_performance_tracking: true,
            enable_collaboration: true,
            enable_cicd: false,
        }
    }
}

impl HubUiState {
    /// Create new Hub UI state
    pub fn new(config: HubUiConfig, cache_dir: PathBuf) -> Self {
        Self {
            repositories: Arc::new(Mutex::new(HashMap::new())),
            config,
            cache_dir,
        }
    }

    /// Add a model repository
    pub fn add_repository(&self, repository: ModelRepository) -> Result<(), TrustformersError> {
        let mut repos = self.repositories.lock().expect("lock should not be poisoned");
        repos.insert(repository.model_id.clone(), repository);
        Ok(())
    }

    /// Get a model repository
    pub fn get_repository(&self, model_id: &str) -> Option<ModelRepository> {
        let repos = self.repositories.lock().expect("lock should not be poisoned");
        repos.get(model_id).cloned()
    }

    /// List all repositories
    pub fn list_repositories(&self) -> Vec<ModelRepository> {
        let repos = self.repositories.lock().expect("lock should not be poisoned");
        repos.values().cloned().collect()
    }

    /// Add a version to a repository
    pub fn add_version(
        &self,
        model_id: &str,
        version: ModelVersion,
    ) -> Result<(), TrustformersError> {
        let mut repos = self.repositories.lock().expect("lock should not be poisoned");
        if let Some(repo) = repos.get_mut(model_id) {
            repo.versions.insert(version.version.clone(), version.clone());
            repo.version_history.push(version.version);
            repo.version_history.sort_by(|a, b| {
                let a_version = repo.versions.get(a).expect("version in history must exist");
                let b_version = repo.versions.get(b).expect("version in history must exist");
                a_version.created_at.cmp(&b_version.created_at)
            });
            repo.metadata.updated_at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs();
            repo.metadata.stats.version_count = repo.versions.len();
            Ok(())
        } else {
            Err(TrustformersError::hub(
                format!("Model not found: {}", model_id),
                model_id.to_string(),
            ))
        }
    }
}

impl ModelRepository {
    /// Create a new model repository
    pub fn new(model_id: String, owner: String) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();
        Self {
            model_id: model_id.clone(),
            versions: HashMap::new(),
            version_history: Vec::new(),
            metadata: RepositoryMetadata {
                name: model_id.clone(),
                description: None,
                owner,
                visibility: Visibility::Public,
                default_version: "main".to_string(),
                tags: Vec::new(),
                created_at: now,
                updated_at: now,
                stats: RepositoryStats {
                    version_count: 0,
                    total_downloads: 0,
                    stars: 0,
                    forks: 0,
                    contributors: 1,
                    last_activity: now,
                },
            },
            access_control: AccessControl {
                permissions: HashMap::new(),
                api_key: None,
                whitelist: Vec::new(),
                blacklist: Vec::new(),
            },
        }
    }

    /// Get the latest version
    pub fn latest_version(&self) -> Option<&ModelVersion> {
        self.version_history.last().and_then(|v| self.versions.get(v))
    }

    /// Get version by identifier
    pub fn get_version(&self, version: &str) -> Option<&ModelVersion> {
        self.versions.get(version)
    }

    /// List all versions
    pub fn list_versions(&self) -> Vec<&ModelVersion> {
        self.version_history.iter().filter_map(|v| self.versions.get(v)).collect()
    }

    /// Compare two versions
    pub fn compare_versions(&self, from: &str, to: &str) -> Option<VersionComparison> {
        let from_version = self.versions.get(from)?;
        let to_version = self.versions.get(to)?;

        Some(VersionComparison {
            from_version: from.to_string(),
            to_version: to.to_string(),
            size_diff: to_version.size_bytes as i64 - from_version.size_bytes as i64,
            changes: self.compute_changes(from_version, to_version),
            performance_diff: self.compute_performance_diff(from_version, to_version),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
        })
    }

    fn compute_changes(&self, from: &ModelVersion, to: &ModelVersion) -> Vec<FileChange> {
        // Simplified change computation
        let mut changes = Vec::new();

        // In a real implementation, this would compare file lists and contents
        if from.checksum != to.checksum {
            changes.push(FileChange {
                path: "model.safetensors".to_string(),
                change_type: ChangeType::Modified,
                old_size: Some(from.size_bytes),
                new_size: to.size_bytes,
                checksum: to.checksum.clone(),
                description: Some("Model weights updated".to_string()),
            });
        }

        changes
    }

    fn compute_performance_diff(&self, from: &ModelVersion, to: &ModelVersion) -> PerformanceDiff {
        let from_metrics = from.metrics.as_ref();
        let to_metrics = to.metrics.as_ref();

        PerformanceDiff {
            accuracy_diff: match (from_metrics, to_metrics) {
                (Some(from), Some(to)) => to.accuracy.zip(from.accuracy).map(|(a, b)| a - b),
                _ => None,
            },
            loss_diff: match (from_metrics, to_metrics) {
                (Some(from), Some(to)) => to.loss.zip(from.loss).map(|(a, b)| a - b),
                _ => None,
            },
            speed_diff: match (from_metrics, to_metrics) {
                (Some(from), Some(to)) => {
                    to.inference_speed.zip(from.inference_speed).map(|(a, b)| a - b)
                },
                _ => None,
            },
            memory_diff: match (from_metrics, to_metrics) {
                (Some(from), Some(to)) => {
                    to.memory_usage.zip(from.memory_usage).map(|(a, b)| a - b)
                },
                _ => None,
            },
        }
    }
}

/// Version comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionComparison {
    /// Source version
    pub from_version: String,
    /// Target version
    pub to_version: String,
    /// Size difference in bytes
    pub size_diff: i64,
    /// File changes
    pub changes: Vec<FileChange>,
    /// Performance differences
    pub performance_diff: PerformanceDiff,
    /// Comparison timestamp
    pub created_at: u64,
}

/// Performance differences between versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDiff {
    /// Accuracy difference
    pub accuracy_diff: Option<f64>,
    /// Loss difference
    pub loss_diff: Option<f64>,
    /// Speed difference
    pub speed_diff: Option<f64>,
    /// Memory usage difference
    pub memory_diff: Option<f64>,
}

/// Hub UI server
pub struct HubUiServer {
    /// Server state
    state: HubUiState,
    /// Axum router
    router: Router,
}

impl HubUiServer {
    /// Create a new Hub UI server
    pub fn new(config: HubUiConfig, cache_dir: PathBuf) -> Self {
        let state = HubUiState::new(config.clone(), cache_dir);
        let router = Self::create_router(state.clone());

        Self { state, router }
    }

    /// Create the Axum router with all routes
    fn create_router(state: HubUiState) -> Router {
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

        println!("🚀 TrustformeRS Hub UI server running on http://{}", addr);
        println!("📊 Repository management: http://{}/ui/", addr);
        println!("🔌 API endpoint: http://{}/api/v1/", addr);

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

// API route handlers

#[axum::debug_handler]
async fn list_repositories(State(state): State<HubUiState>) -> Json<Vec<ModelRepository>> {
    Json(state.list_repositories())
}

async fn get_repository(
    State(state): State<HubUiState>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelRepository>, StatusCode> {
    match state.get_repository(&model_id) {
        Some(repo) => Ok(Json(repo)),
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn create_repository(
    State(state): State<HubUiState>,
    Path(model_id): Path<String>,
    Json(payload): Json<RepositoryMetadata>,
) -> Result<Json<ModelRepository>, StatusCode> {
    let repo = ModelRepository::new(model_id, payload.owner);
    state
        .add_repository(repo.clone())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(repo))
}

async fn update_repository(
    State(state): State<HubUiState>,
    Path(model_id): Path<String>,
    Json(payload): Json<RepositoryMetadata>,
) -> Result<Json<ModelRepository>, StatusCode> {
    // Implementation would update repository metadata
    Err(StatusCode::NOT_IMPLEMENTED)
}

async fn delete_repository(
    State(state): State<HubUiState>,
    Path(model_id): Path<String>,
) -> Result<StatusCode, StatusCode> {
    // Implementation would delete repository
    Err(StatusCode::NOT_IMPLEMENTED)
}

async fn list_versions(
    State(state): State<HubUiState>,
    Path(model_id): Path<String>,
) -> Result<Json<Vec<ModelVersion>>, StatusCode> {
    match state.get_repository(&model_id) {
        Some(repo) => Ok(Json(repo.list_versions().into_iter().cloned().collect())),
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn get_version(
    State(state): State<HubUiState>,
    Path((model_id, version)): Path<(String, String)>,
) -> Result<Json<ModelVersion>, StatusCode> {
    match state.get_repository(&model_id) {
        Some(repo) => match repo.get_version(&version) {
            Some(v) => Ok(Json(v.clone())),
            None => Err(StatusCode::NOT_FOUND),
        },
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn create_version(
    State(state): State<HubUiState>,
    Path((model_id, version)): Path<(String, String)>,
    Json(payload): Json<ModelVersion>,
) -> Result<Json<ModelVersion>, StatusCode> {
    state
        .add_version(&model_id, payload.clone())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(payload))
}

async fn update_version(
    State(state): State<HubUiState>,
    Path((model_id, version)): Path<(String, String)>,
    Json(payload): Json<ModelVersion>,
) -> Result<Json<ModelVersion>, StatusCode> {
    // Implementation would update version metadata
    Err(StatusCode::NOT_IMPLEMENTED)
}

async fn delete_version(
    State(state): State<HubUiState>,
    Path((model_id, version)): Path<(String, String)>,
) -> Result<StatusCode, StatusCode> {
    // Implementation would delete version
    Err(StatusCode::NOT_IMPLEMENTED)
}

async fn compare_versions(
    State(state): State<HubUiState>,
    Path((model_id, from, to)): Path<(String, String, String)>,
) -> Result<Json<VersionComparison>, StatusCode> {
    match state.get_repository(&model_id) {
        Some(repo) => match repo.compare_versions(&from, &to) {
            Some(comparison) => Ok(Json(comparison)),
            None => Err(StatusCode::NOT_FOUND),
        },
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn download_version(
    State(state): State<HubUiState>,
    Path((model_id, version)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Implementation would trigger download and return status
    Ok(Json(
        json!({"status": "download_started", "model_id": model_id, "version": version}),
    ))
}

// UI route handlers

async fn ui_home(State(state): State<HubUiState>) -> Html<String> {
    let repos = state.list_repositories();
    let html = generate_home_html(&repos, &state.config.theme);
    Html(html)
}

async fn ui_repository(
    State(state): State<HubUiState>,
    Path(model_id): Path<String>,
) -> Result<Html<String>, StatusCode> {
    match state.get_repository(&model_id) {
        Some(repo) => {
            let html = generate_repository_html(&repo, &state.config.theme);
            Ok(Html(html))
        },
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn ui_version(
    State(state): State<HubUiState>,
    Path((model_id, version)): Path<(String, String)>,
) -> Result<Html<String>, StatusCode> {
    match state.get_repository(&model_id) {
        Some(repo) => match repo.get_version(&version) {
            Some(v) => {
                let html = generate_version_html(&repo, v, &state.config.theme);
                Ok(Html(html))
            },
            None => Err(StatusCode::NOT_FOUND),
        },
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn ui_compare(
    State(state): State<HubUiState>,
    Path((model_id, from, to)): Path<(String, String, String)>,
) -> Result<Html<String>, StatusCode> {
    match state.get_repository(&model_id) {
        Some(repo) => match repo.compare_versions(&from, &to) {
            Some(comparison) => {
                let html = generate_comparison_html(&repo, &comparison, &state.config.theme);
                Ok(Html(html))
            },
            None => Err(StatusCode::NOT_FOUND),
        },
        None => Err(StatusCode::NOT_FOUND),
    }
}

// HTML generation functions

fn generate_home_html(repositories: &[ModelRepository], theme: &ThemeConfig) -> String {
    let repo_list = repositories
        .iter()
        .map(|repo| {
            format!(
                r#"<div class="repository-card">
                <h3><a href="/ui/repository/{}">{}</a></h3>
                <p>{}</p>
                <div class="stats">
                    <span>Versions: {}</span>
                    <span>Downloads: {}</span>
                    <span>Stars: {}</span>
                </div>
            </div>"#,
                repo.model_id,
                repo.metadata.name,
                repo.metadata.description.as_ref().unwrap_or(&"No description".to_string()),
                repo.metadata.stats.version_count,
                repo.metadata.stats.total_downloads,
                repo.metadata.stats.stars
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustformeRS Model Hub</title>
    <style>
        {css}
    </style>
</head>
<body>
    <header>
        <h1>🤖 TrustformeRS Model Hub</h1>
        <nav>
            <a href="/ui/">Home</a>
            <a href="/api/v1/">API</a>
        </nav>
    </header>
    <main>
        <h2>Model Repositories</h2>
        <div class="repositories">
            {repo_list}
        </div>
    </main>
</body>
</html>"#,
        css = generate_css(theme),
        repo_list = repo_list
    )
}

fn generate_repository_html(repository: &ModelRepository, theme: &ThemeConfig) -> String {
    let versions = repository.list_versions();
    let version_list = versions
        .iter()
        .map(|version| {
            format!(
                r#"<tr>
                <td><a href="/ui/repository/{}/version/{}">{}</a></td>
                <td>{}</td>
                <td>{}</td>
                <td>{:.2} MB</td>
                <td><span class="status-{}">{:?}</span></td>
            </tr>"#,
                repository.model_id,
                version.version,
                version.version,
                version.name.as_ref().unwrap_or(&version.version),
                format_timestamp(version.created_at),
                version.size_bytes as f64 / 1_000_000.0,
                format!("{:?}", version.status).to_lowercase(),
                version.status
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{} - TrustformeRS Model Hub</title>
    <style>
        {css}
    </style>
</head>
<body>
    <header>
        <h1><a href="/ui/">🤖 TrustformeRS Model Hub</a></h1>
    </header>
    <main>
        <div class="repository-header">
            <h2>{}</h2>
            <p>{}</p>
            <div class="repository-stats">
                <span>👤 {}</span>
                <span>📦 {} versions</span>
                <span>⬇️ {} downloads</span>
                <span>⭐ {} stars</span>
            </div>
        </div>

        <section>
            <h3>Versions</h3>
            <table class="versions-table">
                <thead>
                    <tr>
                        <th>Version</th>
                        <th>Name</th>
                        <th>Created</th>
                        <th>Size</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {version_list}
                </tbody>
            </table>
        </section>
    </main>
</body>
</html>"#,
        repository.metadata.name,
        repository.metadata.name,
        repository
            .metadata
            .description
            .as_ref()
            .unwrap_or(&"No description".to_string()),
        repository.metadata.owner,
        repository.metadata.stats.version_count,
        repository.metadata.stats.total_downloads,
        repository.metadata.stats.stars,
        css = generate_css(theme),
        version_list = version_list
    )
}

fn generate_version_html(
    repository: &ModelRepository,
    version: &ModelVersion,
    theme: &ThemeConfig,
) -> String {
    let metrics_html = if let Some(metrics) = &version.metrics {
        format!(
            r#"<div class="metrics">
                <h4>Performance Metrics</h4>
                <div class="metric-grid">
                    {}
                    {}
                    {}
                    {}
                </div>
            </div>"#,
            metrics
                .accuracy
                .map(|a| format!("<div>Accuracy: {:.3}</div>", a))
                .unwrap_or_default(),
            metrics.loss.map(|l| format!("<div>Loss: {:.3}</div>", l)).unwrap_or_default(),
            metrics
                .inference_speed
                .map(|s| format!("<div>Speed: {:.1} tok/s</div>", s))
                .unwrap_or_default(),
            metrics
                .memory_usage
                .map(|m| format!("<div>Memory: {:.1} MB</div>", m))
                .unwrap_or_default(),
        )
    } else {
        "".to_string()
    };

    let changes_html = version
        .changes
        .iter()
        .map(|change| {
            format!(
                r#"<div class="change-item">
                <span class="change-type-{}">{:?}</span>
                <span class="change-path">{}</span>
                <span class="change-size">{:.2} MB</span>
            </div>"#,
                format!("{:?}", change.change_type).to_lowercase(),
                change.change_type,
                change.path,
                change.new_size as f64 / 1_000_000.0
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{} {} - TrustformeRS Model Hub</title>
    <style>
        {css}
    </style>
</head>
<body>
    <header>
        <h1><a href="/ui/">🤖 TrustformeRS Model Hub</a></h1>
    </header>
    <main>
        <nav class="breadcrumb">
            <a href="/ui/repository/{}">{}</a> / {}
        </nav>

        <div class="version-header">
            <h2>{} {}</h2>
            <span class="status-{}">{:?}</span>
        </div>

        <div class="version-info">
            <p>{}</p>
            <div class="version-meta">
                <span>👤 {}</span>
                <span>📅 {}</span>
                <span>📦 {:.2} MB</span>
                <span>🏷️ {}</span>
            </div>
        </div>

        {metrics_html}

        <div class="changes">
            <h4>Changes</h4>
            <div class="changes-list">
                {changes_html}
            </div>
        </div>

        <div class="actions">
            <button onclick="downloadVersion()">⬇️ Download</button>
            <button onclick="showComparison()">🔍 Compare</button>
        </div>
    </main>

    <script>
        function downloadVersion() {{
            fetch('/api/v1/repositories/{}/download/{}', {{method: 'GET'}})
                .then(response => response.json())
                .then(data => alert('Download started: ' + data.status));
        }}

        function showComparison() {{
            // Implementation for version comparison UI
            alert('Version comparison feature coming soon!');
        }}
    </script>
</body>
</html>"#,
        repository.metadata.name,
        version.version,
        repository.model_id,
        repository.metadata.name,
        version.version,
        repository.metadata.name,
        version.name.as_ref().unwrap_or(&version.version),
        format!("{:?}", version.status).to_lowercase(),
        version.status,
        version.description.as_ref().unwrap_or(&"No description".to_string()),
        version.author.as_ref().unwrap_or(&"Unknown".to_string()),
        format_timestamp(version.created_at),
        version.size_bytes as f64 / 1_000_000.0,
        version.tags.join(", "),
        repository.model_id,
        version.version,
        css = generate_css(theme),
        metrics_html = metrics_html,
        changes_html = changes_html
    )
}

fn generate_comparison_html(
    repository: &ModelRepository,
    comparison: &VersionComparison,
    theme: &ThemeConfig,
) -> String {
    let performance_rows = vec![
        (
            "Accuracy",
            comparison.performance_diff.accuracy_diff.map(|d| format!("{:+.3}", d)),
        ),
        (
            "Loss",
            comparison.performance_diff.loss_diff.map(|d| format!("{:+.3}", d)),
        ),
        (
            "Speed",
            comparison.performance_diff.speed_diff.map(|d| format!("{:+.1} tok/s", d)),
        ),
        (
            "Memory",
            comparison.performance_diff.memory_diff.map(|d| format!("{:+.1} MB", d)),
        ),
    ]
    .into_iter()
    .map(|(metric, diff)| {
        format!(
            "<tr><td>{}</td><td>{}</td></tr>",
            metric,
            diff.unwrap_or("N/A".to_string())
        )
    })
    .collect::<Vec<_>>()
    .join("\n");

    let changes_rows = comparison
        .changes
        .iter()
        .map(|change| {
            format!(
                r#"<tr>
                <td><span class="change-type-{}">{:?}</span></td>
                <td>{}</td>
                <td>{:.2} MB</td>
                <td>{}</td>
            </tr>"#,
                format!("{:?}", change.change_type).to_lowercase(),
                change.change_type,
                change.path,
                change.new_size as f64 / 1_000_000.0,
                change.description.as_ref().unwrap_or(&"".to_string())
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Compare {} vs {} - TrustformeRS Model Hub</title>
    <style>
        {css}
    </style>
</head>
<body>
    <header>
        <h1><a href="/ui/">🤖 TrustformeRS Model Hub</a></h1>
    </header>
    <main>
        <nav class="breadcrumb">
            <a href="/ui/repository/{}">{}</a> / Compare
        </nav>

        <div class="comparison-header">
            <h2>🔍 Version Comparison</h2>
            <div class="comparison-versions">
                <span class="version-from">{}</span>
                <span class="arrow">→</span>
                <span class="version-to">{}</span>
            </div>
        </div>

        <div class="comparison-summary">
            <div class="size-diff">
                <h4>Size Change</h4>
                <span class="diff-value">{:+.2} MB</span>
            </div>
        </div>

        <section>
            <h3>Performance Changes</h3>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Change</th>
                    </tr>
                </thead>
                <tbody>
                    {performance_rows}
                </tbody>
            </table>
        </section>

        <section>
            <h3>File Changes</h3>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Type</th>
                        <th>File</th>
                        <th>Size</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    {changes_rows}
                </tbody>
            </table>
        </section>
    </main>
</body>
</html>"#,
        comparison.from_version,
        comparison.to_version,
        repository.model_id,
        repository.metadata.name,
        comparison.from_version,
        comparison.to_version,
        comparison.size_diff as f64 / 1_000_000.0,
        css = generate_css(theme),
        performance_rows = performance_rows,
        changes_rows = changes_rows
    )
}

fn generate_css(theme: &ThemeConfig) -> String {
    format!(
        r#"
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: {text_color};
            background: {bg_color};
        }}

        header {{
            background: {primary_color};
            color: white;
            padding: 1rem 2rem;
            border-bottom: 3px solid {secondary_color};
        }}

        header h1 {{
            font-size: 1.5rem;
            display: inline-block;
        }}

        header h1 a {{
            color: white;
            text-decoration: none;
        }}

        nav {{
            float: right;
            margin-top: 0.25rem;
        }}

        nav a {{
            color: white;
            text-decoration: none;
            margin-left: 1rem;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            transition: background 0.2s;
        }}

        nav a:hover {{
            background: rgba(255, 255, 255, 0.2);
        }}

        main {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}

        .breadcrumb {{
            margin-bottom: 1rem;
            color: {secondary_color};
        }}

        .breadcrumb a {{
            color: {primary_color};
            text-decoration: none;
        }}

        .repository-card {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: box-shadow 0.2s;
        }}

        .repository-card:hover {{
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }}

        .repository-card h3 {{
            margin-bottom: 0.5rem;
        }}

        .repository-card h3 a {{
            color: {primary_color};
            text-decoration: none;
        }}

        .stats {{
            margin-top: 1rem;
            display: flex;
            gap: 1rem;
            font-size: 0.9rem;
            color: {secondary_color};
        }}

        .versions-table, .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}

        .versions-table th, .versions-table td,
        .comparison-table th, .comparison-table td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid {border_color};
        }}

        .versions-table th, .comparison-table th {{
            background: {card_bg};
            font-weight: 600;
        }}

        .status-stable {{ color: #10b981; }}
        .status-development {{ color: #f59e0b; }}
        .status-experimental {{ color: #8b5cf6; }}
        .status-deprecated {{ color: #ef4444; }}
        .status-archived {{ color: {secondary_color}; }}

        .change-type-added {{ color: #10b981; }}
        .change-type-modified {{ color: #f59e0b; }}
        .change-type-deleted {{ color: #ef4444; }}
        .change-type-renamed {{ color: #3b82f6; }}

        .version-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }}

        .version-info {{
            background: {card_bg};
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}

        .version-meta {{
            margin-top: 1rem;
            display: flex;
            gap: 1rem;
            font-size: 0.9rem;
            color: {secondary_color};
        }}

        .metrics {{
            background: {card_bg};
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}

        .changes {{
            background: {card_bg};
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}

        .actions {{
            display: flex;
            gap: 1rem;
        }}

        button {{
            background: {primary_color};
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.2s;
        }}

        button:hover {{
            background: {primary_color}dd;
        }}

        .comparison-header {{
            text-align: center;
            margin-bottom: 2rem;
        }}

        .comparison-versions {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            margin-top: 1rem;
            font-size: 1.2rem;
        }}

        .version-from, .version-to {{
            background: {card_bg};
            padding: 0.5rem 1rem;
            border-radius: 6px;
            border: 1px solid {border_color};
        }}

        .arrow {{
            color: {secondary_color};
            font-size: 1.5rem;
        }}

        .comparison-summary {{
            background: {card_bg};
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            text-align: center;
        }}

        .diff-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: {primary_color};
        }}
        "#,
        primary_color = theme.primary_color,
        secondary_color = theme.secondary_color,
        text_color = if theme.dark_mode { "#e5e7eb" } else { "#111827" },
        bg_color = if theme.dark_mode { "#111827" } else { "#ffffff" },
        card_bg = if theme.dark_mode { "#1f2937" } else { "#f9fafb" },
        border_color = if theme.dark_mode { "#374151" } else { "#e5e7eb" },
    )
}

fn format_timestamp(timestamp: u64) -> String {
    // Simple timestamp formatting - in production would use proper date formatting
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("SystemTime should be after UNIX_EPOCH")
        .as_secs()
        - timestamp;

    if duration < 60 {
        "Just now".to_string()
    } else if duration < 3600 {
        format!("{} minutes ago", duration / 60)
    } else if duration < 86400 {
        format!("{} hours ago", duration / 3600)
    } else {
        format!("{} days ago", duration / 86400)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_repository_creation() {
        let repo = ModelRepository::new("test/model".to_string(), "test_user".to_string());

        assert_eq!(repo.model_id, "test/model");
        assert_eq!(repo.metadata.owner, "test_user");
        assert_eq!(repo.versions.len(), 0);
        assert_eq!(repo.version_history.len(), 0);
    }

    #[test]
    fn test_version_creation() {
        let version = ModelVersion {
            version: "v1.0.0".to_string(),
            name: Some("Initial release".to_string()),
            description: Some("First stable version".to_string()),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
            modified_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
            author: Some("test_user".to_string()),
            tags: vec!["stable".to_string()],
            metrics: None,
            changes: Vec::new(),
            parent_version: None,
            download_stats: None,
            size_bytes: 1024 * 1024 * 100, // 100MB
            checksum: Some("abc123".to_string()),
            status: VersionStatus::Stable,
            compatibility: CompatibilityInfo {
                framework_version: Some("trustformers>=0.1.0".to_string()),
                python_version: Some(">=3.8".to_string()),
                cuda_version: None,
                hardware_requirements: Vec::new(),
                breaking_changes: Vec::new(),
                migration_notes: None,
            },
        };

        assert_eq!(version.version, "v1.0.0");
        assert_eq!(version.status, VersionStatus::Stable);
        assert!(version.name.is_some());
    }

    #[test]
    fn test_hub_ui_state() {
        let config = HubUiConfig::default();
        let cache_dir = std::env::temp_dir();
        let state = HubUiState::new(config, cache_dir);

        let repo = ModelRepository::new("test/model".to_string(), "test_user".to_string());
        assert!(state.add_repository(repo).is_ok());

        let retrieved = state.get_repository("test/model");
        assert!(retrieved.is_some());

        let repos = state.list_repositories();
        assert_eq!(repos.len(), 1);
    }

    #[test]
    fn test_version_comparison() {
        let mut repo = ModelRepository::new("test/model".to_string(), "test_user".to_string());

        let v1 = ModelVersion {
            version: "v1.0.0".to_string(),
            name: Some("V1".to_string()),
            description: None,
            created_at: 1000,
            modified_at: 1000,
            author: None,
            tags: Vec::new(),
            metrics: Some(ModelMetrics {
                accuracy: Some(0.9),
                loss: Some(0.1),
                inference_speed: Some(100.0),
                memory_usage: Some(1000.0),
                parameter_count: None,
                custom_metrics: HashMap::new(),
                benchmarks: Vec::new(),
            }),
            changes: Vec::new(),
            parent_version: None,
            download_stats: None,
            size_bytes: 1000000,
            checksum: Some("abc".to_string()),
            status: VersionStatus::Stable,
            compatibility: CompatibilityInfo {
                framework_version: None,
                python_version: None,
                cuda_version: None,
                hardware_requirements: Vec::new(),
                breaking_changes: Vec::new(),
                migration_notes: None,
            },
        };

        let v2 = ModelVersion {
            version: "v2.0.0".to_string(),
            name: Some("V2".to_string()),
            description: None,
            created_at: 2000,
            modified_at: 2000,
            author: None,
            tags: Vec::new(),
            metrics: Some(ModelMetrics {
                accuracy: Some(0.95),
                loss: Some(0.05),
                inference_speed: Some(120.0),
                memory_usage: Some(1200.0),
                parameter_count: None,
                custom_metrics: HashMap::new(),
                benchmarks: Vec::new(),
            }),
            changes: Vec::new(),
            parent_version: Some("v1.0.0".to_string()),
            download_stats: None,
            size_bytes: 1200000,
            checksum: Some("def".to_string()),
            status: VersionStatus::Stable,
            compatibility: CompatibilityInfo {
                framework_version: None,
                python_version: None,
                cuda_version: None,
                hardware_requirements: Vec::new(),
                breaking_changes: Vec::new(),
                migration_notes: None,
            },
        };

        repo.versions.insert("v1.0.0".to_string(), v1);
        repo.versions.insert("v2.0.0".to_string(), v2);
        repo.version_history = vec!["v1.0.0".to_string(), "v2.0.0".to_string()];

        let comparison = repo.compare_versions("v1.0.0", "v2.0.0").unwrap();

        assert_eq!(comparison.from_version, "v1.0.0");
        assert_eq!(comparison.to_version, "v2.0.0");
        assert_eq!(comparison.size_diff, 200000);
        assert!((comparison.performance_diff.accuracy_diff.unwrap() - 0.05).abs() < 1e-10);
        assert!((comparison.performance_diff.loss_diff.unwrap() + 0.05).abs() < 1e-10);
    }
}
