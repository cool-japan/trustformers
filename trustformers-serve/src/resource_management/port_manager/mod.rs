//! Port Manager Module for TrustformeRS
//!
//! This module provides comprehensive network port management capabilities including:
//! - Port allocation and deallocation
//! - Advanced reservation system
//! - Conflict detection and resolution
//! - Health monitoring and alerting
//! - Performance metrics collection
//!
//! The module is organized into focused submodules for better maintainability.

pub mod types;
pub mod reservation_system;
pub mod conflict_detector;
pub mod health_monitor;
pub mod performance_metrics;
pub mod manager;

// Re-export all types for convenience
pub use types::*;
pub use conflict_detector::*;
pub use health_monitor::*;
pub use performance_metrics::*;
pub use manager::NetworkPortManager;