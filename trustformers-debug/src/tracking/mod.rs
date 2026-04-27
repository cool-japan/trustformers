//! Experiment tracking backends for TrustformeRS debug.
//!
//! # Modules
//!
//! - [`mlflow`] — Local-file MLflow compatible tracking backend.
//! - [`tracker`] — Unified [`ExperimentTracker`] trait + [`InMemoryTracker`].

pub mod mlflow;
pub mod tracker;

pub use mlflow::{
    MetricPoint, MlflowClient, MlflowConfig, MlflowExperiment, RunInfo, RunParam, RunStatus,
};
pub use tracker::{
    ExperimentTracker,
    InMemoryTracker,
    RunId,
    TrackerRunStatus,
    TrackingError,
    TrackingSummary,
};
