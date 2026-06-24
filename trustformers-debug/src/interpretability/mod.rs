//! Interpretability module for TrustformeRS debug tools
//!
//! This module provides comprehensive model interpretability capabilities including:
//! - SHAP (SHapley Additive exPlanations) analysis
//! - LIME (Local Interpretable Model-agnostic Explanations) analysis
//! - Attention analysis for transformer models
//! - Feature attribution methods
//! - Counterfactual generation
//!
//! The module is organized into focused submodules for better maintainability.

pub mod analyzer;
pub mod attention;
pub mod attribution;
pub mod config;
pub mod counterfactual;
pub mod lime;
pub mod report;
pub mod shap;

// Re-export core types and functionality for convenience
pub use analyzer::*;
pub use attention::*;
pub use attribution::*;
pub use config::*;
pub use counterfactual::*;
pub use lime::*;
pub use report::*;
pub use shap::*;
