//! Core Profiler Module
//!
//! This module contains the main orchestrating profiler that integrates all
//! the specialized components.

pub mod profiler;

pub use profiler::{
    MobilePerformanceProfiler, ProfilingSession, ProfilingState, ProfilingStats, SessionStats,
};