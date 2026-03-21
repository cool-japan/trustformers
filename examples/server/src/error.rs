use axum::{
#![allow(unused_variables)]
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

pub type AppResult<T> = Result<T, AppError>;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid request: {0}")]
    BadRequest(String),

    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Internal server error")]
    InternalError(#[from] anyhow::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AppError::ModelNotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::ModelLoadError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            AppError::InferenceError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            AppError::InternalError(err) => {
                tracing::error!("Internal error: {:?}", err);
                (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error".to_string())
            }
        };

        let body = Json(json!({
            "error": error_message,
            "status": status.as_u16(),
        }));

        (status, body).into_response()
    }
}