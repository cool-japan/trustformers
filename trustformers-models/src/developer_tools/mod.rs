//! Developer Tools and Code Generation
//!
//! This module provides comprehensive developer tools including:
//! - Model architecture templates
//! - Automatic code generation
//! - Testing framework integration
//! - Performance benchmarking
//! - CI/CD integration helpers

pub mod benchmark_generator;
pub mod ci_integration;
#[cfg(test)]
mod ci_integration_tests;
pub mod model_generator;
pub mod template_engine;
pub mod test_generator;
pub mod validation_tools;
#[cfg(test)]
mod validation_tools_tests;
