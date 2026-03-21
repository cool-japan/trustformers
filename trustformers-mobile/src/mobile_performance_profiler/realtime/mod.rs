//! Real-Time Monitoring Module
//!
//! This module provides advanced real-time performance monitoring capabilities
//! for mobile ML inference workloads.

pub mod monitor;

pub use monitor::{
    AlertManager, AlertRecord, AlertRule, MonitoringStats, NotificationHandler, RealTimeMonitor,
    RealTimeState,
};