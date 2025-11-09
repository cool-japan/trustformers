//! Simulation Reporting and Summary Generation
//!
//! This module provides comprehensive reporting capabilities for simulation analysis results
//! including summary generation, trend analysis, and performance metrics.

use super::adversarial_analysis::AdversarialProbingResult;
use super::edge_case_discovery::EdgeCaseDiscoveryResult;
use super::perturbation_testing::PerturbationTestResult;
use super::types::SimulationConfig;
use super::what_if_analysis::WhatIfAnalysisResult;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive simulation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationReport {
    pub timestamp: DateTime<Utc>,
    pub config: SimulationConfig,
    pub what_if_analyses_count: usize,
    pub perturbation_tests_count: usize,
    pub adversarial_probes_count: usize,
    pub edge_case_discoveries_count: usize,
    pub recent_what_if_results: Vec<WhatIfAnalysisResult>,
    pub recent_perturbation_results: Vec<PerturbationTestResult>,
    pub recent_adversarial_results: Vec<AdversarialProbingResult>,
    pub recent_edge_case_results: Vec<EdgeCaseDiscoveryResult>,
    pub simulation_summary: HashMap<String, String>,
}
