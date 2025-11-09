//! Core debugging session and configuration management
//!
//! This module contains the fundamental components for TrustformeRS debugging:
//! - Main DebugSession coordinator that manages all debugging tools
//! - Configuration structures and initialization
//! - Session lifecycle management (start, stop, reporting)
//! - Core debugging coordination functionality

pub mod session;

pub use session::*;
