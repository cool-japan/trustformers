//! Comprehensive Concurrency Detection Module for Test Characterization System
//!
//! This module provides sophisticated concurrency analysis and detection capabilities
//! for the TrustformeRS test framework, implementing advanced algorithms for safe
//! parallel execution, deadlock prevention, and resource conflict resolution.
//!
//! # Key Components
//!
//! 1. **ConcurrencyRequirementsDetector**: Core concurrency analysis engine
//! 2. **SafeConcurrencyEstimator**: Advanced algorithms for optimal concurrency estimation
//! 3. **ResourceConflictDetector**: Detection and analysis of resource conflicts
//! 4. **SharingCapabilityAnalyzer**: Resource sharing analysis and optimization
//! 5. **DeadlockAnalyzer**: Deadlock detection and prevention mechanisms
//! 6. **ConcurrencyRiskAssessment**: Risk assessment for concurrent execution
//! 7. **ThreadInteractionAnalyzer**: Thread interaction and synchronization analysis
//! 8. **LockContentionAnalyzer**: Lock contention detection and optimization
//! 9. **ConcurrencyPatternDetector**: Pattern recognition for concurrent behaviors
//! 10. **SafetyValidator**: Comprehensive safety validation and compliance checking
//!
//! # Features
//!
//! - **Multi-Algorithm Estimation**: Conservative, Optimistic, Adaptive, and ML-based approaches
//! - **Advanced Deadlock Detection**: Cycle detection, resource dependency analysis
//! - **Resource Conflict Resolution**: Sophisticated conflict detection and mitigation
//! - **Thread-Safe Operations**: Lock-free and wait-free concurrent data structures
//! - **Real-Time Analysis**: Low-latency analysis with minimal overhead
//! - **Comprehensive Safety**: Multi-layered safety validation and risk assessment
//!
//! # Example Usage
//!
//! ```rust,no_run
//! use trustformers_serve::performance_optimizer::test_characterization::concurrency_detector::*;
//! use trustformers_serve::performance_optimizer::test_characterization::{
//!     ConcurrencyDetectorConfig, TestExecutionData,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = ConcurrencyDetectorConfig::default();
//!     let detector = ConcurrencyRequirementsDetector::new(config).await?;
//!
//!     let test_data = TestExecutionData::default();
//!     let analysis = detector.analyze_concurrency(&test_data).await?;
//!
//!     println!("Safe concurrency level: {:?}", analysis.requirements.max_safe_concurrency);
//!     Ok(())
//! }
//! ```

// Module declarations
mod conflict_detector;
mod deadlock_analyzer;
mod detector;
mod estimator;
mod lock_analyzer;
mod pattern_detector;
mod risk_assessment;
mod safety_validator;
mod sharing_analyzer;
mod thread_analyzer;

// Re-exports
pub use conflict_detector::ResourceConflictDetector;
pub use deadlock_analyzer::DeadlockAnalyzer;
pub use detector::ConcurrencyRequirementsDetector;
pub use estimator::SafeConcurrencyEstimator;
pub use lock_analyzer::LockContentionAnalyzer;
pub use pattern_detector::ConcurrencyPatternDetector;
pub use risk_assessment::ConcurrencyRiskAssessment;
pub use safety_validator::SafetyValidator;
pub use sharing_analyzer::SharingCapabilityAnalyzer;
pub use thread_analyzer::ThreadInteractionAnalyzer;
