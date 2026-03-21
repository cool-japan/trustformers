//! Simulation Tools for Model Testing and Analysis
//!
//! This module provides comprehensive simulation tools for what-if analysis,
//! perturbation testing, adversarial probing, robustness testing, and edge
//! case discovery for TrustformeRS models.
//!
//! ## Architecture
//!
//! The simulation tools system is organized into several focused modules:
//! - [`types`] - Core types, enums, and configuration for simulation tools
//! - [`what_if_analysis`] - What-if analysis for model behavior exploration
//! - [`perturbation_testing`] - Perturbation testing and robustness assessment
//! - [`adversarial_analysis`] - Adversarial analysis and attack resistance testing
//! - [`edge_case_discovery`] - Edge case discovery and risk assessment
//! - [`analyzer`] - Main SimulationAnalyzer orchestrating all components
//! - [`reporting`] - Simulation reporting and summary generation

pub mod adversarial_analysis;
pub mod analyzer;
pub mod edge_case_discovery;
pub mod perturbation_testing;
pub mod reporting;
pub mod types;
pub mod what_if_analysis;

// Re-export core types for backward compatibility
pub use analyzer::SimulationAnalyzer;
pub use types::*;

// Re-export component types for easy access
pub use what_if_analysis::{
    BoundaryComplexity, BoundaryCrossingAnalysis, BoundaryPoint, CounterfactualInsight,
    CrossingDirection, DecisionBoundaryExploration, FeatureChange, FeatureImportanceRank,
    FeatureInteractionSensitivity, FeatureSensitivityAnalysis, LocalLinearityAnalysis,
    PredictionStabilityAnalysis, Scenario, ScenarioImpactAnalysis, WhatIfAnalysisResult,
};

pub use perturbation_testing::{
    CascadingFailureAnalysis, FailureFrequencyAnalysis, FailureMode, FailureModesAnalysis,
    FailureSeverityAnalysis, MitigationStrategy, PerturbationDetail, PerturbationIntensityResult,
    PerturbationTestResult, RobustnessAssessment, SensitivityHotspot, TimeToFailureAnalysis,
    TriggeringCondition,
};

pub use adversarial_analysis::{
    AdversarialExample, AdversarialProbingResult, AdversarialRobustnessAssessment,
    AttackDifficultyAnalysis, AttackSuccessAnalysis, CertifiedRobustnessAnalysis,
    ComplexityAssessment, DefenseRecommendation, RobustnessGuarantee, VulnerabilityHotspot,
};

pub use edge_case_discovery::{
    CoverageAnalysis, CoverageGap, EdgeCase, EdgeCaseClassification, EdgeCaseDiscoveryResult,
    EdgeCasePattern, EdgeCaseRiskAssessment, RiskMitigationPriority, SystematicIssue,
    UncoveredRegion,
};

pub use reporting::SimulationReport;
