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

pub mod config;
pub mod shap;
pub mod lime;
pub mod attention;
pub mod attribution;
pub mod counterfactual;
pub mod analyzer;
pub mod report;

// Re-export core types and functionality for convenience
pub use config::*;
pub use shap::*;
pub use lime::*;
pub use attention::*;
pub use attribution::*;
pub use counterfactual::*;
pub use analyzer::*;
pub use report::*;