//! Axum route handlers for the Hub UI's JSON API and HTML pages.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Html, Json},
};

use super::repository::ModelRepository;
use super::state::HubUiState;
use super::templates::{
    generate_comparison_html, generate_home_html, generate_repository_html, generate_version_html,
};
use super::types::{ModelVersion, RepositoryMetadata, VersionComparison};

// API route handlers

#[axum::debug_handler]
pub(super) async fn list_repositories(
    State(state): State<HubUiState>,
) -> Json<Vec<ModelRepository>> {
    Json(state.list_repositories())
}

pub(super) async fn get_repository(
    State(state): State<HubUiState>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelRepository>, StatusCode> {
    match state.get_repository(&model_id) {
        Some(repo) => Ok(Json(repo)),
        None => Err(StatusCode::NOT_FOUND),
    }
}

pub(super) async fn create_repository(
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

pub(super) async fn update_repository(
    State(state): State<HubUiState>,
    Path(model_id): Path<String>,
    Json(payload): Json<RepositoryMetadata>,
) -> Result<Json<ModelRepository>, StatusCode> {
    state
        .update_repository(&model_id, payload)
        .map(Json)
        .map_err(|_| StatusCode::NOT_FOUND)
}

pub(super) async fn delete_repository(
    State(state): State<HubUiState>,
    Path(model_id): Path<String>,
) -> Result<StatusCode, StatusCode> {
    state
        .delete_repository(&model_id)
        .map(|_| StatusCode::NO_CONTENT)
        .map_err(|_| StatusCode::NOT_FOUND)
}

pub(super) async fn list_versions(
    State(state): State<HubUiState>,
    Path(model_id): Path<String>,
) -> Result<Json<Vec<ModelVersion>>, StatusCode> {
    match state.get_repository(&model_id) {
        Some(repo) => Ok(Json(repo.list_versions().into_iter().cloned().collect())),
        None => Err(StatusCode::NOT_FOUND),
    }
}

pub(super) async fn get_version(
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

pub(super) async fn create_version(
    State(state): State<HubUiState>,
    Path((model_id, version)): Path<(String, String)>,
    Json(payload): Json<ModelVersion>,
) -> Result<Json<ModelVersion>, StatusCode> {
    // The URL's `:version` segment must match the request body's `version`
    // field: without this check, POSTing to `.../versions/v1` with a body
    // claiming to be `v2` silently created `v2` instead, ignoring the URL
    // entirely.
    if payload.version != version {
        return Err(StatusCode::BAD_REQUEST);
    }
    state
        .add_version(&model_id, payload.clone())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(payload))
}

pub(super) async fn update_version(
    State(state): State<HubUiState>,
    Path((model_id, version)): Path<(String, String)>,
    Json(payload): Json<ModelVersion>,
) -> Result<Json<ModelVersion>, StatusCode> {
    state
        .update_version(&model_id, &version, payload)
        .map(Json)
        .map_err(|_| StatusCode::NOT_FOUND)
}

pub(super) async fn delete_version(
    State(state): State<HubUiState>,
    Path((model_id, version)): Path<(String, String)>,
) -> Result<StatusCode, StatusCode> {
    state
        .delete_version(&model_id, &version)
        .map(|_| StatusCode::NO_CONTENT)
        .map_err(|_| StatusCode::NOT_FOUND)
}

pub(super) async fn compare_versions(
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

pub(super) async fn download_version(
    State(state): State<HubUiState>,
    Path((model_id, version)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Validate the model and version actually exist, using real repository
    // state, before claiming anything about them.
    let repo = state.get_repository(&model_id).ok_or(StatusCode::NOT_FOUND)?;
    if repo.get_version(&version).is_none() {
        return Err(StatusCode::NOT_FOUND);
    }

    // Actually triggering a download (fetching the real model weight files
    // via `hub::download_model` et al.) is not wired into this endpoint
    // yet; honestly report that instead of claiming a download started that
    // never did.
    Err(StatusCode::NOT_IMPLEMENTED)
}

// UI route handlers

pub(super) async fn ui_home(State(state): State<HubUiState>) -> Html<String> {
    let repos = state.list_repositories();
    let html = generate_home_html(&repos, &state.config.theme);
    Html(html)
}

pub(super) async fn ui_repository(
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

pub(super) async fn ui_version(
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

pub(super) async fn ui_compare(
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
