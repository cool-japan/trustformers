//! Numerical stability checking for model debugging
//!
//! This module provides tools to detect and analyze numerical stability issues in
//! neural network computations, including NaN, Inf, underflow, and overflow detection.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Numerical stability checker for detecting computational issues
#[derive(Debug)]
pub struct StabilityChecker {
    /// Detected issues by layer name
    issues: HashMap<String, Vec<StabilityIssue>>,
    /// Configuration
    config: StabilityConfig,
    /// Issue counter for tracking
    issue_counter: usize,
}

/// Configuration for stability checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityConfig {
    /// Check for NaN values
    pub check_nan: bool,
    /// Check for Inf values
    pub check_inf: bool,
    /// Check for underflow (values too close to zero)
    pub check_underflow: bool,
    /// Check for overflow (values too large)
    pub check_overflow: bool,
    /// Underflow threshold
    pub underflow_threshold: f64,
    /// Overflow threshold
    pub overflow_threshold: f64,
    /// Whether to stop on first issue
    pub stop_on_first_issue: bool,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            check_nan: true,
            check_inf: true,
            check_underflow: true,
            check_overflow: true,
            underflow_threshold: 1e-15,
            overflow_threshold: 1e15,
            stop_on_first_issue: false,
        }
    }
}

/// Type of stability issue
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IssueKind {
    /// Not a Number (NaN)
    NaN,
    /// Positive infinity
    PosInf,
    /// Negative infinity
    NegInf,
    /// Underflow (value too close to zero)
    Underflow,
    /// Overflow (value too large)
    Overflow,
    /// Precision loss
    PrecisionLoss,
}

/// Detected stability issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityIssue {
    /// Unique issue ID
    pub id: usize,
    /// Layer or operation name
    pub layer_name: String,
    /// Type of issue
    pub kind: IssueKind,
    /// Number of affected values
    pub count: usize,
    /// Position in tensor (if applicable)
    pub positions: Vec<Vec<usize>>,
    /// Sample values that triggered the issue
    pub sample_values: Vec<f64>,
    /// Timestamp when detected
    pub timestamp: u64,
    /// Additional context
    pub context: Option<String>,
}

/// Summary of all detected issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilitySummary {
    /// Total number of issues
    pub total_issues: usize,
    /// Issues grouped by kind
    pub issues_by_kind: HashMap<IssueKind, usize>,
    /// Issues grouped by layer
    pub issues_by_layer: HashMap<String, usize>,
    /// Most problematic layers
    pub problematic_layers: Vec<(String, usize)>,
}

impl StabilityChecker {
    /// Create a new stability checker
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::StabilityChecker;
    ///
    /// let checker = StabilityChecker::new();
    /// ```
    pub fn new() -> Self {
        Self {
            issues: HashMap::new(),
            config: StabilityConfig::default(),
            issue_counter: 0,
        }
    }

    /// Create a stability checker with custom configuration
    pub fn with_config(config: StabilityConfig) -> Self {
        Self {
            issues: HashMap::new(),
            config,
            issue_counter: 0,
        }
    }

    /// Check a tensor for stability issues
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Name of the layer or operation
    /// * `values` - Tensor values to check
    ///
    /// # Example
    ///
    /// ```
    /// # use trustformers_debug::StabilityChecker;
    /// # let mut checker = StabilityChecker::new();
    /// let values = vec![1.0, f64::NAN, 2.0, f64::INFINITY];
    /// let issues = checker.check_tensor("layer1", &values).unwrap();
    /// assert!(issues > 0);
    /// ```
    pub fn check_tensor(&mut self, layer_name: &str, values: &[f64]) -> Result<usize> {
        let mut issues_found = 0;

        // Check for NaN
        if self.config.check_nan {
            issues_found += self.check_nan(layer_name, values)?;
        }

        // Check for Inf
        if self.config.check_inf {
            issues_found += self.check_inf(layer_name, values)?;
        }

        // Check for underflow
        if self.config.check_underflow {
            issues_found += self.check_underflow(layer_name, values)?;
        }

        // Check for overflow
        if self.config.check_overflow {
            issues_found += self.check_overflow(layer_name, values)?;
        }

        if self.config.stop_on_first_issue && issues_found > 0 {
            anyhow::bail!("Stability issues detected in {}", layer_name);
        }

        Ok(issues_found)
    }

    /// Check for NaN values
    fn check_nan(&mut self, layer_name: &str, values: &[f64]) -> Result<usize> {
        let mut positions = Vec::new();
        let mut sample_values = Vec::new();

        for (i, &value) in values.iter().enumerate() {
            if value.is_nan() {
                positions.push(vec![i]);
                if sample_values.len() < 10 {
                    sample_values.push(value);
                }
            }
        }

        if !positions.is_empty() {
            let id = self.next_issue_id();
            self.add_issue(StabilityIssue {
                id,
                layer_name: layer_name.to_string(),
                kind: IssueKind::NaN,
                count: positions.len(),
                positions,
                sample_values,
                timestamp: current_timestamp()?,
                context: None,
            });
            Ok(1)
        } else {
            Ok(0)
        }
    }

    /// Check for Inf values
    fn check_inf(&mut self, layer_name: &str, values: &[f64]) -> Result<usize> {
        let mut pos_inf_positions = Vec::new();
        let mut neg_inf_positions = Vec::new();
        let mut pos_inf_samples = Vec::new();
        let mut neg_inf_samples = Vec::new();

        for (i, &value) in values.iter().enumerate() {
            if value.is_infinite() {
                if value.is_sign_positive() {
                    pos_inf_positions.push(vec![i]);
                    if pos_inf_samples.len() < 10 {
                        pos_inf_samples.push(value);
                    }
                } else {
                    neg_inf_positions.push(vec![i]);
                    if neg_inf_samples.len() < 10 {
                        neg_inf_samples.push(value);
                    }
                }
            }
        }

        let mut issues_count = 0;

        if !pos_inf_positions.is_empty() {
            let id = self.next_issue_id();
            self.add_issue(StabilityIssue {
                id,
                layer_name: layer_name.to_string(),
                kind: IssueKind::PosInf,
                count: pos_inf_positions.len(),
                positions: pos_inf_positions,
                sample_values: pos_inf_samples,
                timestamp: current_timestamp()?,
                context: None,
            });
            issues_count += 1;
        }

        if !neg_inf_positions.is_empty() {
            let id = self.next_issue_id();
            self.add_issue(StabilityIssue {
                id,
                layer_name: layer_name.to_string(),
                kind: IssueKind::NegInf,
                count: neg_inf_positions.len(),
                positions: neg_inf_positions,
                sample_values: neg_inf_samples,
                timestamp: current_timestamp()?,
                context: None,
            });
            issues_count += 1;
        }

        Ok(issues_count)
    }

    /// Check for underflow
    fn check_underflow(&mut self, layer_name: &str, values: &[f64]) -> Result<usize> {
        let mut positions = Vec::new();
        let mut sample_values = Vec::new();

        for (i, &value) in values.iter().enumerate() {
            if !value.is_nan()
                && !value.is_infinite()
                && value != 0.0
                && value.abs() < self.config.underflow_threshold
            {
                positions.push(vec![i]);
                if sample_values.len() < 10 {
                    sample_values.push(value);
                }
            }
        }

        if !positions.is_empty() {
            let id = self.next_issue_id();
            let threshold = self.config.underflow_threshold;
            self.add_issue(StabilityIssue {
                id,
                layer_name: layer_name.to_string(),
                kind: IssueKind::Underflow,
                count: positions.len(),
                positions,
                sample_values,
                timestamp: current_timestamp()?,
                context: Some(format!("threshold: {}", threshold)),
            });
            Ok(1)
        } else {
            Ok(0)
        }
    }

    /// Check for overflow
    fn check_overflow(&mut self, layer_name: &str, values: &[f64]) -> Result<usize> {
        let mut positions = Vec::new();
        let mut sample_values = Vec::new();

        for (i, &value) in values.iter().enumerate() {
            if !value.is_nan()
                && !value.is_infinite()
                && value.abs() > self.config.overflow_threshold
            {
                positions.push(vec![i]);
                if sample_values.len() < 10 {
                    sample_values.push(value);
                }
            }
        }

        if !positions.is_empty() {
            let id = self.next_issue_id();
            let threshold = self.config.overflow_threshold;
            self.add_issue(StabilityIssue {
                id,
                layer_name: layer_name.to_string(),
                kind: IssueKind::Overflow,
                count: positions.len(),
                positions,
                sample_values,
                timestamp: current_timestamp()?,
                context: Some(format!("threshold: {}", threshold)),
            });
            Ok(1)
        } else {
            Ok(0)
        }
    }

    /// Add an issue to the tracker
    fn add_issue(&mut self, issue: StabilityIssue) {
        let layer_name = issue.layer_name.clone();
        self.issues.entry(layer_name).or_default().push(issue);
    }

    /// Get the next issue ID
    fn next_issue_id(&mut self) -> usize {
        let id = self.issue_counter;
        self.issue_counter += 1;
        id
    }

    /// Get all issues for a specific layer
    pub fn get_issues(&self, layer_name: &str) -> Option<&Vec<StabilityIssue>> {
        self.issues.get(layer_name)
    }

    /// Get all issues
    pub fn get_all_issues(&self) -> Vec<&StabilityIssue> {
        self.issues.values().flatten().collect()
    }

    /// Get summary of all issues
    pub fn summary(&self) -> StabilitySummary {
        let mut issues_by_kind: HashMap<IssueKind, usize> = HashMap::new();
        let mut issues_by_layer: HashMap<String, usize> = HashMap::new();

        for (layer_name, layer_issues) in &self.issues {
            issues_by_layer.insert(layer_name.clone(), layer_issues.len());

            for issue in layer_issues {
                *issues_by_kind.entry(issue.kind).or_insert(0) += 1;
            }
        }

        let mut problematic_layers: Vec<_> =
            issues_by_layer.iter().map(|(k, &v)| (k.clone(), v)).collect();
        problematic_layers.sort_by(|a, b| b.1.cmp(&a.1));

        let total_issues = self.get_all_issues().len();

        StabilitySummary {
            total_issues,
            issues_by_kind,
            issues_by_layer,
            problematic_layers,
        }
    }

    /// Print a detailed report
    pub fn report(&self) -> String {
        let mut output = String::new();
        output.push_str("Numerical Stability Report\n");
        output.push_str(&"=".repeat(80));
        output.push('\n');

        let summary = self.summary();

        output.push_str(&format!("\nTotal Issues: {}\n", summary.total_issues));

        output.push_str("\nIssues by Type:\n");
        for (kind, count) in &summary.issues_by_kind {
            output.push_str(&format!("  {:?}: {}\n", kind, count));
        }

        output.push_str("\nMost Problematic Layers:\n");
        for (layer, count) in summary.problematic_layers.iter().take(10) {
            output.push_str(&format!("  {}: {} issues\n", layer, count));
        }

        output.push_str("\nDetailed Issues:\n");
        for (layer_name, layer_issues) in &self.issues {
            output.push_str(&format!("\n  Layer: {}\n", layer_name));
            for issue in layer_issues {
                output.push_str(&format!(
                    "    [{:?}] {} occurrences",
                    issue.kind, issue.count
                ));
                if let Some(ref context) = issue.context {
                    output.push_str(&format!(" ({})", context));
                }
                output.push('\n');
            }
        }

        output
    }

    /// Export issues to JSON
    pub fn export_to_json(&self, output_path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.issues)?;
        std::fs::write(output_path, json)?;
        Ok(())
    }

    /// Clear all recorded issues
    pub fn clear(&mut self) {
        self.issues.clear();
        self.issue_counter = 0;
    }

    /// Check if any issues were detected
    pub fn has_issues(&self) -> bool {
        !self.issues.is_empty()
    }

    /// Get total number of issues
    pub fn total_issues(&self) -> usize {
        self.issues.values().map(|v| v.len()).sum()
    }
}

impl Default for StabilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to get current timestamp
fn current_timestamp() -> Result<u64> {
    Ok(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stability_checker_creation() {
        let checker = StabilityChecker::new();
        assert_eq!(checker.total_issues(), 0);
    }

    #[test]
    fn test_check_nan() {
        let mut checker = StabilityChecker::new();
        let values = vec![1.0, f64::NAN, 2.0, f64::NAN];

        let issues = checker.check_tensor("layer1", &values).unwrap();
        assert!(issues > 0);
        assert!(checker.has_issues());
    }

    #[test]
    fn test_check_inf() {
        let mut checker = StabilityChecker::new();
        let values = vec![1.0, f64::INFINITY, 2.0, f64::NEG_INFINITY];

        let issues = checker.check_tensor("layer1", &values).unwrap();
        assert!(issues > 0);
        assert!(checker.has_issues());
    }

    #[test]
    fn test_check_underflow() {
        let mut checker = StabilityChecker::new();
        let values = vec![1.0, 1e-20, 2.0, 1e-18];

        let issues = checker.check_tensor("layer1", &values).unwrap();
        assert!(issues > 0);
    }

    #[test]
    fn test_check_overflow() {
        let mut config = StabilityConfig::default();
        config.overflow_threshold = 100.0;

        let mut checker = StabilityChecker::with_config(config);
        let values = vec![1.0, 200.0, 2.0, 300.0];

        let issues = checker.check_tensor("layer1", &values).unwrap();
        assert!(issues > 0);
    }

    #[test]
    fn test_summary() {
        let mut checker = StabilityChecker::new();

        checker.check_tensor("layer1", &[f64::NAN, 1.0]).unwrap();
        checker.check_tensor("layer2", &[f64::INFINITY, 2.0]).unwrap();

        let summary = checker.summary();
        assert!(summary.total_issues > 0);
        assert_eq!(summary.issues_by_layer.len(), 2);
    }

    #[test]
    fn test_report() {
        let mut checker = StabilityChecker::new();
        checker.check_tensor("layer1", &[f64::NAN, 1.0]).unwrap();

        let report = checker.report();
        assert!(report.contains("Numerical Stability Report"));
        assert!(report.contains("layer1"));
    }

    #[test]
    fn test_export_to_json() {
        use std::env;

        let temp_dir = env::temp_dir();
        let output_path = temp_dir.join("stability_issues.json");

        let mut checker = StabilityChecker::new();
        checker.check_tensor("layer1", &[f64::NAN, 1.0]).unwrap();

        checker.export_to_json(&output_path).unwrap();
        assert!(output_path.exists());

        // Clean up
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn test_clear() {
        let mut checker = StabilityChecker::new();
        checker.check_tensor("layer1", &[f64::NAN]).unwrap();

        assert!(checker.has_issues());

        checker.clear();
        assert!(!checker.has_issues());
        assert_eq!(checker.total_issues(), 0);
    }

    #[test]
    fn test_no_issues() {
        let mut checker = StabilityChecker::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let issues = checker.check_tensor("layer1", &values).unwrap();
        assert_eq!(issues, 0);
        assert!(!checker.has_issues());
    }

    #[test]
    fn test_custom_config() {
        let config = StabilityConfig {
            check_nan: true,
            check_inf: false,
            check_underflow: false,
            check_overflow: false,
            underflow_threshold: 1e-10,
            overflow_threshold: 1e10,
            stop_on_first_issue: false,
        };

        let mut checker = StabilityChecker::with_config(config);
        let values = vec![1.0, f64::INFINITY, f64::NAN];

        // Should only detect NaN, not Inf
        let issues = checker.check_tensor("layer1", &values).unwrap();
        assert_eq!(issues, 1);
    }
}
