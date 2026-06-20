//! Test Characterization Manager Module
//!
//! This module provides comprehensive orchestration and management functionality for the test
//! characterization system. It serves as the main coordination layer that manages all specialized
//! analysis modules and provides intelligent scheduling, caching, configuration management,
//! and result synthesis.
//!
//! # Architecture
//!
//! The manager module consists of several key components:
//!
//! * `TestCharacterizationEngine` - Main orchestrator that coordinates all analysis modules
//! * `AnalysisOrchestrator` - Coordination and sequencing of different analysis phases
//! * `ComponentManager` - Management and lifecycle control of all analysis components
//! * `ResultsSynthesizer` - Integration and synthesis of results from all analysis modules
//! * `CacheCoordinator` - Coordination of caching across all modules for optimal performance
//! * `ConfigurationManager` - Centralized configuration management for all analysis components
//! * `AnalysisScheduler` - Intelligent scheduling and prioritization of analysis tasks
//! * `PerformanceCoordinator` - Coordination of performance monitoring across all modules
//! * `ErrorRecoveryManager` - Centralized error handling and recovery coordination
//! * `ReportingCoordinator` - Coordination of comprehensive reporting across all analysis results
//!
//! # Example Usage
//!
//! ```rust,no_run
//! use trustformers_serve::performance_optimizer::test_characterization::manager::TestCharacterizationEngine;
//! use trustformers_serve::performance_optimizer::test_characterization::TestExecutionData;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Characterize a test
//! let test_data = TestExecutionData::default();
//! println!("Test ID: {:?}", test_data.test_id);
//! # Ok(())
//! # }
//! ```

mod cache_coordinator;
mod component_manager;
mod config_manager;
mod engine;
mod error_recovery;
mod orchestrator;
mod performance_coordinator;
mod reporting;
mod scheduler;
mod synthesizer;

// Re-export main types
pub use cache_coordinator::*;
pub use component_manager::*;
pub use config_manager::*;
pub use engine::*;
pub use error_recovery::*;
pub use orchestrator::*;
pub use performance_coordinator::*;
pub use reporting::*;
pub use scheduler::*;
pub use synthesizer::*;

/// Maximum concurrent analysis tasks
pub const MAX_CONCURRENT_ANALYSES: usize = 16;

/// Cache entry TTL in seconds
pub const CACHE_TTL_SECONDS: u64 = 3600;

/// Configuration refresh interval in seconds
pub const CONFIG_REFRESH_INTERVAL_SECONDS: u64 = 300;

/// Performance monitoring interval in milliseconds
pub const PERFORMANCE_MONITORING_INTERVAL_MS: u64 = 1000;

/// Error recovery retry attempts
pub const ERROR_RECOVERY_MAX_RETRIES: usize = 3;

/// Analysis queue capacity
pub const ANALYSIS_QUEUE_CAPACITY: usize = 1000;
