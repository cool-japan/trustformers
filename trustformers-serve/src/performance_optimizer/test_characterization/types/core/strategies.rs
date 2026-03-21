//! Strategy types for test characterization

use async_trait::async_trait;
use std::{collections::HashMap, time::Duration};

use super::super::{
    locking::{DeadlockPreventionStrategy, DeadlockRisk},
    patterns::{SharingAnalysisStrategy, SharingStrategy},
    quality::RiskMitigationStrategy,
    resources::{ResourceAccessPattern, ResourceSharingCapabilities},
};
use super::enums::{PriorityLevel, TestCharacterizationResult, UrgencyLevel};
use super::quality::PreventionAction;

#[derive(Debug, Clone)]
pub struct BalancedStrategy {
    pub accuracy_weight: f64,
    pub performance_weight: f64,
    pub resource_weight: f64,
}

impl BalancedStrategy {
    /// Create a new BalancedStrategy with default weights
    pub fn new() -> Self {
        Self {
            accuracy_weight: 0.33,
            performance_weight: 0.33,
            resource_weight: 0.34,
        }
    }
}

impl Default for BalancedStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl super::super::performance::ProfilingStrategy for BalancedStrategy {
    fn profile(&self) -> String {
        format!(
            "Balanced Profiling Strategy (accuracy={:.2}, performance={:.2}, resource={:.2})",
            self.accuracy_weight, self.performance_weight, self.resource_weight
        )
    }

    fn name(&self) -> &str {
        "BalancedStrategy"
    }

    async fn activate(&self) -> anyhow::Result<()> {
        Ok(())
    }

    async fn deactivate(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct HighFrequencyStrategy {
    pub sample_rate_hz: f64,
    pub max_samples: usize,
}

impl HighFrequencyStrategy {
    /// Create a new HighFrequencyStrategy with default settings
    pub fn new() -> Self {
        Self {
            sample_rate_hz: 1000.0,
            max_samples: 10000,
        }
    }
}

impl Default for HighFrequencyStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl super::super::performance::ProfilingStrategy for HighFrequencyStrategy {
    fn profile(&self) -> String {
        format!(
            "High Frequency Profiling Strategy (sample_rate={:.0} Hz, max_samples={})",
            self.sample_rate_hz, self.max_samples
        )
    }

    fn name(&self) -> &str {
        "HighFrequencyStrategy"
    }

    async fn activate(&self) -> anyhow::Result<()> {
        Ok(())
    }

    async fn deactivate(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ReadOnlySharingStrategy {
    pub enabled: bool,
    pub cache_enabled: bool,
}

impl ReadOnlySharingStrategy {
    pub fn new(enabled: bool, cache_enabled: bool) -> Self {
        Self {
            enabled,
            cache_enabled,
        }
    }
}

impl SharingAnalysisStrategy for ReadOnlySharingStrategy {
    fn analyze_sharing(
        &self,
        _resource_id: &str,
        _access_patterns: &[ResourceAccessPattern],
    ) -> TestCharacterizationResult<ResourceSharingCapabilities> {
        // Read-only sharing allows unlimited concurrent readers
        Ok(ResourceSharingCapabilities {
            supports_read_sharing: self.enabled,
            supports_write_sharing: false,
            max_concurrent_readers: if self.enabled { None } else { Some(0) },
            max_concurrent_writers: Some(0),
            sharing_overhead: if self.cache_enabled { 0.05 } else { 0.1 },
            consistency_guarantees: vec!["Read consistency".to_string()],
            isolation_requirements: vec!["No writers during read".to_string()],
            recommended_strategy: SharingStrategy::ReadSharing,
            safety_assessment: 0.95,
            performance_tradeoffs: HashMap::new(),
            performance_overhead: if self.cache_enabled { 0.05 } else { 0.1 },
            implementation_complexity: 0.3,
            sharing_mode: "read-only".to_string(),
        })
    }

    fn name(&self) -> &str {
        "Read-Only Sharing Strategy"
    }

    fn accuracy(&self) -> f64 {
        0.95
    }

    fn supported_resource_types(&self) -> Vec<String> {
        vec![
            "Cache".to_string(),
            "Configuration".to_string(),
            "ReadOnlyData".to_string(),
            "Reference".to_string(),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct TimeoutBasedStrategy {
    pub timeout_ms: u64,
    pub abort_on_timeout: bool,
}

impl TimeoutBasedStrategy {
    pub fn new(timeout_ms: u64, abort_on_timeout: bool) -> Self {
        Self {
            timeout_ms,
            abort_on_timeout,
        }
    }
}

impl DeadlockPreventionStrategy for TimeoutBasedStrategy {
    fn generate_prevention(
        &self,
        _risk: &DeadlockRisk,
    ) -> TestCharacterizationResult<Vec<PreventionAction>> {
        let mut actions = Vec::new();

        // Generate timeout-based prevention actions
        actions.push(PreventionAction {
            action_id: format!("timeout_action_{}", uuid::Uuid::new_v4()),
            action_type: if self.abort_on_timeout {
                "Timeout with Abort".to_string()
            } else {
                "Timeout with Retry".to_string()
            },
            description: format!(
                "Apply timeout to all locks (timeout: {}ms)",
                self.timeout_ms
            ),
            priority: if self.abort_on_timeout {
                PriorityLevel::Critical
            } else {
                PriorityLevel::High
            },
            urgency: UrgencyLevel::High,
            estimated_effort: "Medium".to_string(),
            expected_impact: 0.75,
            implementation_steps: vec!["Set timeout on lock acquisition".to_string()],
            verification_steps: vec!["Verify timeout enforcement".to_string()],
            rollback_plan: "Remove timeout configuration".to_string(),
            dependencies: Vec::new(),
            constraints: Vec::new(),
            estimated_completion_time: Duration::from_secs(60),
            risk_mitigation_score: 0.85,
        });

        Ok(actions)
    }

    fn name(&self) -> &str {
        "Timeout-Based Strategy"
    }

    fn effectiveness(&self) -> f64 {
        if self.abort_on_timeout {
            0.75
        } else {
            0.65
        }
    }

    fn applies_to(&self, _risk: &DeadlockRisk) -> bool {
        true // Timeout applies to all deadlock risks
    }
}

#[derive(Debug, Clone)]
pub struct PreventiveMitigation {
    pub enabled: bool,
    pub strategies: Vec<String>,
}

impl PreventiveMitigation {
    pub fn new(enabled: bool, strategies: Vec<String>) -> Self {
        Self {
            enabled,
            strategies,
        }
    }
}

impl RiskMitigationStrategy for PreventiveMitigation {
    fn mitigate(&self) -> String {
        if self.enabled {
            format!(
                "Preventive mitigation with {} strategies",
                self.strategies.len()
            )
        } else {
            "Preventive mitigation disabled".to_string()
        }
    }

    fn name(&self) -> &str {
        "PreventiveMitigation"
    }

    fn is_applicable(&self) -> bool {
        self.enabled
    }
}

#[derive(Debug, Clone)]
pub struct ReactiveMitigation {
    pub enabled: bool,
    pub response_time_ms: u64,
}

impl ReactiveMitigation {
    pub fn new(enabled: bool, response_time_ms: u64) -> Self {
        Self {
            enabled,
            response_time_ms,
        }
    }
}

impl RiskMitigationStrategy for ReactiveMitigation {
    fn mitigate(&self) -> String {
        if self.enabled {
            format!(
                "Reactive mitigation with {}ms response time",
                self.response_time_ms
            )
        } else {
            "Reactive mitigation disabled".to_string()
        }
    }

    fn name(&self) -> &str {
        "ReactiveMitigation"
    }

    fn is_applicable(&self) -> bool {
        self.enabled
    }
}

#[derive(Debug, Clone)]
pub struct TimeoutResolutionStrategy {
    pub timeout_ms: u64,
    pub retry: bool,
}

impl TimeoutResolutionStrategy {
    pub fn new(timeout_ms: u64, retry: bool) -> Self {
        Self { timeout_ms, retry }
    }
}

#[derive(Debug, Clone)]
pub struct AvoidanceResolutionStrategy {
    pub enabled: bool,
    pub reserve_resources: bool,
}

impl AvoidanceResolutionStrategy {
    pub fn new(enabled: bool, reserve_resources: bool) -> Self {
        Self {
            enabled,
            reserve_resources,
        }
    }
}
