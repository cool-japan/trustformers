//! WebSocket / SSE streaming dashboard.
//!
//! Provides a lightweight real-time event-streaming server that pushes
//! [`DashboardEvent`]s to connected HTTP clients via *Server-Sent Events*
//! (SSE) over plain TCP, without requiring any third-party web framework.
//!
//! # Why SSE instead of WebSockets?
//!
//! `trustformers-debug` does not list a WebSocket library in its
//! dependencies. SSE is simpler to implement with raw TCP, is supported by
//! every modern browser, and is a perfect fit for the unidirectional
//! server-to-client event stream that a training dashboard requires.
//!
//! # Modules
//!
//! - [`websocket`] — The streaming event server and supporting types.
//! - [`metrics`] — Typed training-metrics message enum, metric history, and
//!   the extended in-process dashboard server.

pub mod metrics;
pub mod websocket;

pub use websocket::{DashboardConfig, DashboardEvent, DashboardServer};
pub use metrics::{
    DashboardConfig as MetricsDashboardConfig,
    DashboardError,
    DashboardMessage,
    DashboardServerExt,
    MetricHistory,
};
