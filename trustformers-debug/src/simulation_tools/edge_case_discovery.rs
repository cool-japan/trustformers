//! Edge Case Discovery and Risk Assessment
//!
//! This module provides comprehensive edge case discovery capabilities including
//! systematic edge case identification, classification, coverage analysis, and risk assessment.

use super::types::*;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Edge case discovery result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCaseDiscoveryResult {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// Discovered edge cases
    pub edge_cases: Vec<EdgeCase>,
    /// Edge case classification
    pub classification: EdgeCaseClassification,
    /// Coverage analysis
    pub coverage_analysis: CoverageAnalysis,
    /// Risk assessment
    pub risk_assessment: EdgeCaseRiskAssessment,
}

/// Individual edge case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCase {
    /// Edge case ID
    pub id: String,
    /// Edge case description
    pub description: String,
    /// Input that triggers edge case
    pub trigger_input: HashMap<String, f64>,
    /// Model output for edge case
    pub model_output: f64,
    /// Expected output (if known)
    pub expected_output: Option<f64>,
    /// Edge case type
    pub edge_case_type: EdgeCaseType,
    /// Severity level
    pub severity: EdgeCaseSeverity,
    /// Likelihood of occurrence
    pub likelihood: f64,
    /// Detection method
    pub detection_method: String,
}

/// Classification of discovered edge cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCaseClassification {
    /// Edge cases by type
    pub by_type: HashMap<EdgeCaseType, usize>,
    /// Edge cases by severity
    pub by_severity: HashMap<EdgeCaseSeverity, usize>,
    /// Most common edge case patterns
    pub common_patterns: Vec<EdgeCasePattern>,
    /// Systematic issues identified
    pub systematic_issues: Vec<SystematicIssue>,
}

/// Pattern in edge cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCasePattern {
    /// Pattern description
    pub pattern: String,
    /// Number of cases matching pattern
    pub frequency: usize,
    /// Features involved in pattern
    pub involved_features: Vec<String>,
    /// Pattern severity
    pub pattern_severity: f64,
}

/// Systematic issue identified
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystematicIssue {
    /// Issue description
    pub issue: String,
    /// Affected feature regions
    pub affected_regions: Vec<String>,
    /// Issue impact
    pub impact: SystematicIssueImpact,
    /// Recommended fixes
    pub recommended_fixes: Vec<String>,
}

/// Coverage analysis for edge case discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageAnalysis {
    /// Feature space coverage
    pub feature_space_coverage: f64,
    /// Boundary coverage
    pub boundary_coverage: f64,
    /// Uncovered regions
    pub uncovered_regions: Vec<UncoveredRegion>,
    /// Coverage gaps
    pub coverage_gaps: Vec<CoverageGap>,
}

/// Region not covered by edge case discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncoveredRegion {
    /// Region description
    pub region: String,
    /// Region boundaries
    pub boundaries: HashMap<String, (f64, f64)>,
    /// Estimated risk level
    pub risk_level: f64,
    /// Reason for lack of coverage
    pub coverage_reason: String,
}

/// Gap in edge case coverage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageGap {
    /// Gap description
    pub gap: String,
    /// Gap location
    pub location: HashMap<String, f64>,
    /// Gap size
    pub size: f64,
    /// Potential edge cases in gap
    pub potential_edge_cases: usize,
}

/// Risk assessment for edge cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCaseRiskAssessment {
    /// Overall risk score
    pub overall_risk: f64,
    /// Risk by edge case type
    pub risk_by_type: HashMap<EdgeCaseType, f64>,
    /// High-risk edge cases
    pub high_risk_cases: Vec<String>,
    /// Risk mitigation priorities
    pub mitigation_priorities: Vec<RiskMitigationPriority>,
}

/// Priority for risk mitigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMitigationPriority {
    /// Priority level
    pub priority: usize,
    /// Edge cases to address
    pub edge_cases: Vec<String>,
    /// Recommended actions
    pub actions: Vec<String>,
    /// Expected risk reduction
    pub risk_reduction: f64,
}
