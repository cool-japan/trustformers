//! The REST-facing error type.
//!
//! Every variant maps to an HTTP status a client can act on, and every message
//! is built from a real failure (a propagated library error, a request the
//! server genuinely cannot honour) rather than a generic "something went
//! wrong". Nothing here invents a success response for a failed operation.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

pub type AppResult<T> = Result<T, AppError>;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("invalid request: {0}")]
    BadRequest(String),

    #[error(
        "model `{model_id}` was loaded for task `{loaded_task}`, which this endpoint does not serve \
         (it serves `{expected_task}`)"
    )]
    TaskMismatch { model_id: String, loaded_task: &'static str, expected_task: &'static str },

    #[error("{0}")]
    ResourceExhausted(String),

    #[error("failed to load model: {0}")]
    LoadError(String),

    #[error("inference failed: {0}")]
    InferenceError(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match &self {
            AppError::ModelNotFound(_) => StatusCode::NOT_FOUND,
            AppError::BadRequest(_) | AppError::TaskMismatch { .. } => StatusCode::BAD_REQUEST,
            AppError::ResourceExhausted(_) => StatusCode::SERVICE_UNAVAILABLE,
            AppError::LoadError(_) | AppError::InferenceError(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };

        if status == StatusCode::INTERNAL_SERVER_ERROR {
            tracing::error!(error = %self, "request failed");
        }

        let body = Json(json!({
            "error": self.to_string(),
            "status": status.as_u16(),
        }));

        (status, body).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_not_found_maps_to_404() {
        let err = AppError::ModelNotFound("abc".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn task_mismatch_names_both_tasks_and_maps_to_400() {
        let err = AppError::TaskMismatch {
            model_id: "abc".to_string(),
            loaded_task: "text-generation",
            expected_task: "text-classification",
        };
        let message = err.to_string();
        assert!(message.contains("text-generation"), "message was: {message}");
        assert!(message.contains("text-classification"), "message was: {message}");
        assert_eq!(err.into_response().status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn resource_exhausted_maps_to_503() {
        let err = AppError::ResourceExhausted("no free slots".to_string());
        assert_eq!(err.into_response().status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn load_error_maps_to_500() {
        let err = AppError::LoadError("no weight file".to_string());
        assert_eq!(err.into_response().status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
