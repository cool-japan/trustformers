//! Resource-conflict detection algorithms.
//!
//! These are the [`ConflictDetectionAlgorithm`] implementations that
//! [`ResourceConflictDetector`](crate::performance_optimizer::test_characterization::concurrency_detector::ResourceConflictDetector)
//! runs over a test's recorded [`ResourceAccessPattern`]s.
//!
//! ## Changed in 0.2.1
//!
//! Every algorithm in this file used to ignore its `access_patterns` argument
//! entirely and return `Ok(Vec::new())` under a `// Simplified ... conflict
//! detection` comment. The detector took that empty vector as a measurement:
//! it fed it to `calculate_detection_confidence`, which returns `1.0` for an
//! empty slice, and published "no conflicts, confidence 1.00" for every test it
//! was ever given. Each algorithm now derives its answer from the input:
//!
//! * [`StaticConflictDetectionAlgorithm`] compares the declared access types on
//!   a resource against that resource's declared sharing capability.
//! * [`DynamicConflictDetectionAlgorithm`] reads the [`ContentionEvent`]s that
//!   were actually recorded during the run.
//! * [`PredictiveConflictDetectionAlgorithm`] projects the recorded per-access
//!   timing forward and looks for overlapping future access windows.
//!
//! A fourth implementation, `MLConflictDetectionAlgorithm`, was **deleted**
//! rather than fixed. It carried a `model: String` naming a model
//! (`"default"`, `"svm"`) that does not exist anywhere in this crate, and no
//! trained conflict classifier exists in-tree for it to call; it could only
//! ever have answered by fabricating. The one real model type nearby,
//! `PredictionModel`, holds no coefficients and reports as much (see
//! `types/core/mod.rs`).

use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

use super::super::locking::{
    ConflictDetectionAlgorithm, ConflictImpact, ConflictSeverity, ConflictType, ContentionSeverity,
};
use super::super::network_io::AccessType;
use super::super::resources::{
    ResourceAccessPattern, ResourceConflict, ResourceSharingCapabilities,
};
use super::enums::{TestCharacterizationError, TestCharacterizationResult};

/// Access modes that can modify the resource.
///
/// Everything not listed here (`ReadOnly`, `Shared`, `Execute`) observes the
/// resource without changing it.
fn is_write_access(access: AccessType) -> bool {
    matches!(
        access,
        AccessType::WriteOnly
            | AccessType::ReadWrite
            | AccessType::Exclusive
            | AccessType::Append
            | AccessType::Create
            | AccessType::Delete
            | AccessType::Modify
    )
}

/// Groups access patterns by the resource they touch, preserving input order
/// inside each group so that reported conflicts are deterministic.
fn group_by_resource(
    patterns: &[ResourceAccessPattern],
) -> BTreeMap<&str, Vec<&ResourceAccessPattern>> {
    let mut grouped: BTreeMap<&str, Vec<&ResourceAccessPattern>> = BTreeMap::new();
    for pattern in patterns {
        grouped.entry(pattern.resource_id.as_str()).or_default().push(pattern);
    }
    grouped
}

/// The most restrictive sharing capability declared by any accessor of a
/// resource.
///
/// Different accessors may declare different beliefs about the same resource;
/// the conservative intersection is the only one that is safe for all of them.
fn most_restrictive_capability(
    group: &[&ResourceAccessPattern],
) -> Option<ResourceSharingCapabilities> {
    let mut merged = group.first()?.sharing_capability.clone();
    for pattern in group.iter().skip(1) {
        let capability = &pattern.sharing_capability;
        merged.supports_read_sharing &= capability.supports_read_sharing;
        merged.supports_write_sharing &= capability.supports_write_sharing;
        merged.max_concurrent_readers = min_option(
            merged.max_concurrent_readers,
            capability.max_concurrent_readers,
        );
        merged.max_concurrent_writers = min_option(
            merged.max_concurrent_writers,
            capability.max_concurrent_writers,
        );
        merged.sharing_overhead = merged.sharing_overhead.max(capability.sharing_overhead);
        merged.safety_assessment = merged.safety_assessment.min(capability.safety_assessment);
    }
    Some(merged)
}

/// `None` means "no declared limit", so it never tightens a declared one.
fn min_option(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    match (left, right) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Maps a computed conflict probability onto the severity ladder.
///
/// The cut points are a presentation choice over a computed probability, not a
/// measurement of their own; `probability` remains the load-bearing number.
fn severity_for_probability(probability: f64) -> ConflictSeverity {
    if probability >= 0.9 {
        ConflictSeverity::Severe
    } else if probability >= 0.6 {
        ConflictSeverity::Major
    } else if probability >= 0.3 {
        ConflictSeverity::Moderate
    } else {
        ConflictSeverity::Minor
    }
}

/// Builds a conflict record whose unmeasurable impact fields stay `None`.
///
/// `ConflictImpact`'s reliability / user-experience / stability / cascade /
/// recovery-time / mitigation fields are `Option` precisely so that an
/// analysis which cannot observe them says so instead of writing a zero that
/// reads as "measured, and it is fine".
fn build_conflict(
    conflict_id: String,
    resource_id: &str,
    conflict_type: ConflictType,
    probability: f64,
    confidence: f64,
    performance_degradation: f64,
    max_safe_concurrency: usize,
) -> ResourceConflict {
    let mut resource_impact = HashMap::new();
    resource_impact.insert(resource_id.to_string(), performance_degradation);
    ResourceConflict {
        conflict_id,
        conflict_type,
        severity: severity_for_probability(probability),
        resource_id: resource_id.to_string(),
        probability,
        confidence,
        max_safe_concurrency,
        performance_impact: ConflictImpact {
            performance_degradation,
            resource_impact,
            confidence,
            ..ConflictImpact::default()
        },
        ..ResourceConflict::default()
    }
}

// ============================================================================
// STATIC ANALYSIS
// ============================================================================

/// Detects conflicts that follow from the *declared* shape of a test's resource
/// accesses, without needing any runtime observation.
///
/// A resource is in conflict when the accessors it was given cannot all be
/// admitted at once by the sharing capability declared for it: an exclusive
/// demand alongside any other accessor, more writers than the resource admits,
/// more readers than it admits, or readers alongside a writer on a resource
/// that does not support read sharing.
#[derive(Debug, Clone)]
pub struct StaticConflictDetectionAlgorithm {
    /// When `false`, `detect_conflicts` reports that it did not run rather than
    /// returning an empty result that would read as "no conflicts".
    pub enabled: bool,
    /// Lowest computed probability this algorithm will report: a conflict is
    /// emitted when `probability >= 1.0 - sensitivity`, so raising sensitivity
    /// widens what is reported.
    pub sensitivity: f64,
}

impl StaticConflictDetectionAlgorithm {
    /// Creates the algorithm.
    ///
    /// `sensitivity` is clamped into `0.0..=1.0`.
    pub fn new(enabled: bool, sensitivity: f64) -> Self {
        Self {
            enabled,
            sensitivity: sensitivity.clamp(0.0, 1.0),
        }
    }

    /// The minimum computed probability this algorithm reports.
    fn report_threshold(&self) -> f64 {
        1.0 - self.sensitivity
    }
}

impl ConflictDetectionAlgorithm for StaticConflictDetectionAlgorithm {
    fn detect_conflicts(
        &self,
        access_patterns: &[ResourceAccessPattern],
    ) -> TestCharacterizationResult<Vec<ResourceConflict>> {
        if !self.enabled {
            return Err(TestCharacterizationError::Configuration {
                message: "static conflict detection is disabled; no analysis was performed"
                    .to_string(),
                context: HashMap::new(),
            });
        }

        let mut conflicts = Vec::new();
        for (resource_id, group) in group_by_resource(access_patterns) {
            if group.len() < 2 {
                continue;
            }
            let Some(capability) = most_restrictive_capability(&group) else {
                continue;
            };
            let writers = group.iter().filter(|p| is_write_access(p.access_type)).count();
            let readers = group.len() - writers;
            let exclusives =
                group.iter().filter(|p| p.access_type == AccessType::Exclusive).count();

            let admitted_writers = if capability.supports_write_sharing {
                capability.max_concurrent_writers.unwrap_or(usize::MAX)
            } else {
                1
            };
            let admitted_readers = if capability.supports_read_sharing {
                capability.max_concurrent_readers.unwrap_or(usize::MAX)
            } else {
                1
            };

            // Each finding is (probability, conflict type, id suffix). The
            // strongest one describes the resource.
            let mut findings: Vec<(f64, ConflictType, &'static str)> = Vec::new();

            if exclusives > 0 {
                // An exclusive demand cannot coexist with any other accessor,
                // so the group size alone settles it.
                findings.push((1.0, ConflictType::ResourceAccess, "exclusive-demand"));
            }
            if writers > admitted_writers {
                let excess = (writers - admitted_writers) as f64;
                findings.push((
                    excess / writers as f64,
                    ConflictType::ReadWrite,
                    "writer-overcommit",
                ));
            }
            if readers > admitted_readers {
                let excess = (readers - admitted_readers) as f64;
                findings.push((
                    excess / readers as f64,
                    ConflictType::ResourceAccess,
                    "reader-overcommit",
                ));
            }
            if writers > 0 && readers > 0 && !capability.supports_read_sharing {
                findings.push((
                    readers as f64 / group.len() as f64,
                    ConflictType::ReadWrite,
                    "read-during-write",
                ));
            }

            let strongest = findings.into_iter().max_by(|a, b| a.0.total_cmp(&b.0));
            let Some((probability, conflict_type, reason)) = strongest else {
                continue;
            };
            if probability < self.report_threshold() {
                continue;
            }

            let degradation =
                group.iter().map(|p| p.performance_impact).sum::<f64>() / group.len() as f64;
            let admitted = if writers > 0 {
                admitted_writers.min(group.len())
            } else {
                admitted_readers.min(group.len())
            };
            conflicts.push(build_conflict(
                format!("static:{resource_id}:{reason}"),
                resource_id,
                conflict_type,
                probability,
                // The finding follows deterministically from the access types
                // and sharing capability supplied in the input; it is not a
                // statistical estimate, so there is nothing to discount.
                1.0,
                degradation,
                admitted.max(1),
            ));
        }
        Ok(conflicts)
    }

    fn name(&self) -> &str {
        "StaticConflictDetection"
    }

    fn sensitivity(&self) -> f64 {
        self.sensitivity
    }

    fn update_parameters(
        &mut self,
        params: HashMap<String, f64>,
    ) -> TestCharacterizationResult<()> {
        if let Some(&sensitivity) = params.get("sensitivity") {
            self.sensitivity = sensitivity.clamp(0.0, 1.0);
        }
        if let Some(&enabled) = params.get("enabled") {
            self.enabled = enabled != 0.0;
        }
        Ok(())
    }
}

// ============================================================================
// DYNAMIC ANALYSIS
// ============================================================================

/// Weight assigned to a recorded contention severity when an event carries no
/// measured `performance_impact` of its own.
///
/// This is a documented mapping of a recorded categorical onto a number, used
/// only as a floor under the measured impact — never in place of it.
fn contention_severity_weight(severity: ContentionSeverity) -> f64 {
    match severity {
        ContentionSeverity::Minimal => 0.05,
        ContentionSeverity::Low => 0.10,
        ContentionSeverity::Moderate | ContentionSeverity::Medium => 0.30,
        ContentionSeverity::High => 0.60,
        ContentionSeverity::Severe => 0.80,
        ContentionSeverity::Critical => 0.90,
        ContentionSeverity::Extreme => 0.95,
    }
}

/// Orders contention severities so the worst one observed can be reported.
fn contention_severity_rank(severity: ContentionSeverity) -> u8 {
    match severity {
        ContentionSeverity::Minimal => 0,
        ContentionSeverity::Low => 1,
        ContentionSeverity::Moderate | ContentionSeverity::Medium => 2,
        ContentionSeverity::High => 3,
        ContentionSeverity::Severe => 4,
        ContentionSeverity::Critical => 5,
        ContentionSeverity::Extreme => 6,
    }
}

/// Detects conflicts from the contention that was actually observed while the
/// test ran.
///
/// Contention is sampled, so what a run records is a fraction of what happened.
/// `sample_rate` is that fraction and is used to correct the estimate: the
/// probability that at least one contention occurs on a resource is
/// `1 - (∏(1 - wᵢ))^(1/sample_rate)`, where `wᵢ` is each observed event's
/// measured performance impact (floored by its recorded severity).
#[derive(Debug, Clone)]
pub struct DynamicConflictDetectionAlgorithm {
    /// When `false`, `detect_conflicts` reports that nothing was monitored
    /// rather than returning an empty result that would read as "no conflicts".
    pub runtime_monitoring: bool,
    /// Fraction of runtime events the monitor actually captured, in `0.0..=1.0`.
    pub sample_rate: f64,
}

impl DynamicConflictDetectionAlgorithm {
    /// Creates the algorithm. `sample_rate` is clamped into `0.0..=1.0`.
    pub fn new(runtime_monitoring: bool, sample_rate: f64) -> Self {
        Self {
            runtime_monitoring,
            sample_rate: sample_rate.clamp(0.0, 1.0),
        }
    }
}

impl ConflictDetectionAlgorithm for DynamicConflictDetectionAlgorithm {
    fn detect_conflicts(
        &self,
        access_patterns: &[ResourceAccessPattern],
    ) -> TestCharacterizationResult<Vec<ResourceConflict>> {
        if !self.runtime_monitoring {
            return Err(TestCharacterizationError::Configuration {
                message: "dynamic conflict detection requires runtime monitoring, which is off; \
                          no contention was observed"
                    .to_string(),
                context: HashMap::new(),
            });
        }
        if self.sample_rate <= 0.0 {
            return Err(TestCharacterizationError::Configuration {
                message: "dynamic conflict detection needs a positive sample rate; at 0.0 no \
                          runtime event can be observed"
                    .to_string(),
                context: HashMap::new(),
            });
        }

        let mut conflicts = Vec::new();
        for (resource_id, group) in group_by_resource(access_patterns) {
            let events: Vec<_> =
                group.iter().flat_map(|pattern| pattern.contention_events.iter()).collect();
            if events.is_empty() {
                // The run recorded no contention on this resource. That is an
                // observation, not an absence of analysis.
                continue;
            }

            let mut survival = 1.0_f64;
            let mut worst_rank = 0_u8;
            let mut worst_severity = ContentionSeverity::Minimal;
            let mut impact_total = 0.0_f64;
            let mut narrowest_competition = usize::MAX;
            for event in &events {
                let weight = event
                    .performance_impact
                    .max(contention_severity_weight(event.severity))
                    .clamp(0.0, 1.0);
                survival *= 1.0 - weight;
                impact_total += event.performance_impact;
                let rank = contention_severity_rank(event.severity);
                if rank >= worst_rank {
                    worst_rank = rank;
                    worst_severity = event.severity;
                }
                if !event.competing_threads.is_empty() {
                    narrowest_competition =
                        narrowest_competition.min(event.competing_threads.len());
                }
            }

            let probability = 1.0 - survival.powf(1.0 / self.sample_rate);
            if probability <= 0.0 {
                // Every recorded event carried zero impact.
                continue;
            }

            let mut conflict = build_conflict(
                format!("dynamic:{resource_id}:observed-contention"),
                resource_id,
                ConflictType::ResourceAccess,
                probability,
                // Only the sampled fraction was seen, so that fraction is
                // exactly how much of the run this conclusion rests on.
                self.sample_rate,
                impact_total / events.len() as f64,
                narrowest_competition
                    .checked_sub(1)
                    .filter(|_| narrowest_competition != usize::MAX)
                    .unwrap_or(1)
                    .max(1),
            );
            conflict.severity = match worst_severity {
                ContentionSeverity::Minimal | ContentionSeverity::Low => ConflictSeverity::Minor,
                ContentionSeverity::Moderate | ContentionSeverity::Medium => {
                    ConflictSeverity::Moderate
                },
                ContentionSeverity::High => ConflictSeverity::Major,
                ContentionSeverity::Severe => ConflictSeverity::Severe,
                ContentionSeverity::Critical => ConflictSeverity::Critical,
                ContentionSeverity::Extreme => ConflictSeverity::Blocking,
            };
            conflict.historical_count = events.len();
            conflicts.push(conflict);
        }
        Ok(conflicts)
    }

    fn name(&self) -> &str {
        "DynamicConflictDetection"
    }

    fn sensitivity(&self) -> f64 {
        self.sample_rate
    }

    fn update_parameters(
        &mut self,
        params: HashMap<String, f64>,
    ) -> TestCharacterizationResult<()> {
        if let Some(&rate) = params.get("sample_rate") {
            self.sample_rate = rate.clamp(0.0, 1.0);
        }
        if let Some(&monitoring) = params.get("runtime_monitoring") {
            self.runtime_monitoring = monitoring != 0.0;
        }
        Ok(())
    }
}

// ============================================================================
// PREDICTIVE ANALYSIS
// ============================================================================

/// One projected future access window, as an offset from the group's earliest
/// recorded access.
#[derive(Debug, Clone, Copy)]
struct ProjectedWindow {
    start: Duration,
    end: Duration,
}

/// Projects a pattern's recorded access timing forward.
///
/// Returns `None` when the pattern recorded fewer than two accesses, because a
/// single sample establishes no period to project from.
fn project_windows(
    pattern: &ResourceAccessPattern,
    base: Instant,
    horizon: usize,
) -> Option<Vec<ProjectedWindow>> {
    let mut samples: Vec<(Instant, Duration)> = pattern.timing_pattern.clone();
    samples.sort_by_key(|(start, _)| *start);
    if samples.len() < 2 {
        return None;
    }

    let mut period_total = Duration::ZERO;
    for window in samples.windows(2) {
        let (earlier, _) = window[0];
        let (later, _) = window[1];
        period_total += later.checked_duration_since(earlier).unwrap_or_default();
    }
    let period = period_total.checked_div((samples.len() - 1) as u32)?;
    if period.is_zero() {
        return None;
    }
    let mean_duration = samples
        .iter()
        .map(|(_, duration)| *duration)
        .sum::<Duration>()
        .checked_div(samples.len() as u32)
        .unwrap_or_default();

    let (last_start, _) = samples[samples.len() - 1];
    let last_offset = last_start.checked_duration_since(base).unwrap_or_default();
    Some(
        (1..=horizon)
            .map(|step| {
                let start = last_offset + period * step as u32;
                ProjectedWindow {
                    start,
                    end: start + mean_duration,
                }
            })
            .collect(),
    )
}

/// Detects conflicts that have not happened yet, by projecting each accessor's
/// recorded cadence forward and looking for overlapping future windows on the
/// same resource.
///
/// A projection is only as good as the regularity of what it extrapolates, so
/// the two accessors' recorded `predictability_score`s become the finding's
/// confidence and `accuracy_threshold` is the bar it has to clear.
#[derive(Debug, Clone)]
pub struct PredictiveConflictDetectionAlgorithm {
    /// How many future access cycles to project per accessor.
    pub prediction_horizon: usize,
    /// Minimum confidence a projected conflict must reach to be reported.
    pub accuracy_threshold: f64,
}

impl PredictiveConflictDetectionAlgorithm {
    /// Creates the algorithm. `accuracy_threshold` is clamped into `0.0..=1.0`.
    pub fn new(prediction_horizon: usize, accuracy_threshold: f64) -> Self {
        Self {
            prediction_horizon,
            accuracy_threshold: accuracy_threshold.clamp(0.0, 1.0),
        }
    }
}

impl ConflictDetectionAlgorithm for PredictiveConflictDetectionAlgorithm {
    fn detect_conflicts(
        &self,
        access_patterns: &[ResourceAccessPattern],
    ) -> TestCharacterizationResult<Vec<ResourceConflict>> {
        if access_patterns.is_empty() {
            return Ok(Vec::new());
        }
        if self.prediction_horizon == 0 {
            return Err(TestCharacterizationError::Configuration {
                message: "predictive conflict detection with a zero-cycle horizon projects nothing"
                    .to_string(),
                context: HashMap::new(),
            });
        }
        if access_patterns.iter().all(|pattern| pattern.timing_pattern.len() < 2) {
            return Err(TestCharacterizationError::ResourceAnalysis {
                message: "no accessor recorded two or more timed accesses, so there is no cadence \
                          to project forward"
                    .to_string(),
                resource_type: "access-timing".to_string(),
                context: HashMap::new(),
            });
        }

        let mut conflicts = Vec::new();
        for (resource_id, group) in group_by_resource(access_patterns) {
            if group.len() < 2 {
                continue;
            }
            let Some(base) = group
                .iter()
                .filter_map(|pattern| pattern.timing_pattern.iter().map(|(start, _)| *start).min())
                .min()
            else {
                continue;
            };

            let projected: Vec<(&ResourceAccessPattern, Vec<ProjectedWindow>)> = group
                .iter()
                .filter_map(|pattern| {
                    project_windows(pattern, base, self.prediction_horizon)
                        .map(|windows| (*pattern, windows))
                })
                .collect();
            if projected.len() < 2 {
                continue;
            }

            let mut compared = 0_usize;
            let mut overlapping = 0_usize;
            let mut confidence_total = 0.0_f64;
            let mut confidence_pairs = 0_usize;
            let mut incompatible = false;
            for (index, (left_pattern, left_windows)) in projected.iter().enumerate() {
                for (right_pattern, right_windows) in projected.iter().skip(index + 1) {
                    if !is_write_access(left_pattern.access_type)
                        && !is_write_access(right_pattern.access_type)
                    {
                        // Two readers overlapping is not a conflict.
                        continue;
                    }
                    incompatible = true;
                    confidence_total += (left_pattern.predictability_score
                        + right_pattern.predictability_score)
                        / 2.0;
                    confidence_pairs += 1;
                    for left in left_windows {
                        for right in right_windows {
                            compared += 1;
                            if left.start < right.end && right.start < left.end {
                                overlapping += 1;
                            }
                        }
                    }
                }
            }
            if !incompatible || compared == 0 || overlapping == 0 {
                continue;
            }

            let confidence = confidence_total / confidence_pairs.max(1) as f64;
            if confidence < self.accuracy_threshold {
                continue;
            }
            let probability = overlapping as f64 / compared as f64;
            let degradation =
                group.iter().map(|p| p.performance_impact).sum::<f64>() / group.len() as f64;
            conflicts.push(build_conflict(
                format!("predictive:{resource_id}:projected-overlap"),
                resource_id,
                ConflictType::Timing,
                probability,
                confidence,
                degradation,
                1,
            ));
        }
        Ok(conflicts)
    }

    fn name(&self) -> &str {
        "PredictiveConflictDetection"
    }

    fn sensitivity(&self) -> f64 {
        self.accuracy_threshold
    }

    fn update_parameters(
        &mut self,
        params: HashMap<String, f64>,
    ) -> TestCharacterizationResult<()> {
        if let Some(&horizon) = params.get("prediction_horizon") {
            self.prediction_horizon = horizon.max(0.0) as usize;
        }
        if let Some(&threshold) = params.get("accuracy_threshold") {
            self.accuracy_threshold = threshold.clamp(0.0, 1.0);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "conflict_algorithms_tests.rs"]
mod conflict_algorithms_tests;
