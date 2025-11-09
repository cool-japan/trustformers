//! Federated Learning v2.0 Module Organization
//!
//! This module organizes the federated learning v2.0 system components
//! and provides comprehensive re-exports for backward compatibility.

/// Core types and data structures
pub mod types;

/// Privacy mechanisms and differential privacy
pub mod privacy;

/// Cryptographic protocols and security primitives
pub mod crypto;

/// Secure aggregation protocols
pub mod aggregation;

/// Communication protocols and network management
pub mod communication;

/// Security monitoring and attack detection
pub mod security;

/// Training coordination and management
pub mod training;

/// Main federated learning engine
pub mod engine;

// Re-export all public types for backward compatibility
pub use self::types::*;
pub use self::privacy::*;
pub use self::crypto::*;
pub use self::aggregation::*;
pub use self::communication::*;
pub use self::security::*;
pub use self::training::*;
pub use self::engine::*;