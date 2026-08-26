//! Concurrency-pattern detection algorithms.
//!
//! These are the [`PatternDetectionAlgorithm`] implementations that
//! [`ConcurrencyPatternDetector`](crate::performance_optimizer::test_characterization::concurrency_detector::ConcurrencyPatternDetector)
//! runs over a test's recorded [`ThreadInteraction`]s.
//!
//! ## Changed in 0.2.1
//!
//! Until 0.2.1 the trait was `fn detect(&self) -> String`: it was handed no
//! data at all, and each implementation answered from a `detected: bool` field
//! that `new()` set to `false` and nothing ever wrote. Every detector therefore
//! returned `"No <X> pattern detected"` for every test forever, and the branch
//! that would have said `"...(confidence: 0.85)"` was unreachable. The
//! detector's `detect_concurrency_patterns` ignored its `test_data` argument to
//! match.
//!
//! Detection is a function of the recorded interactions, not of detector state,
//! so the detectors are now unit structs and the trait takes the test's
//! execution data. Each returns `None` when its shape is genuinely absent from
//! the recorded interaction graph, and an error when there is nothing recorded
//! to look at.

use std::collections::{BTreeMap, BTreeSet};

use super::super::patterns::{
    ConcurrencyPattern, InteractionType, PatternDetectionAlgorithm, ThreadInteraction,
};
use super::enums::{TestCharacterizationError, TestCharacterizationResult};
use super::TestExecutionData;
use std::collections::HashMap;

/// Interaction kinds that move data from one thread to another.
///
/// These are the edges that carry a producer/consumer or pipeline relationship;
/// a `Synchronization` or `SignalHandling` edge coordinates without handing
/// work along, so it is not evidence of either shape.
fn is_data_flow(interaction: InteractionType) -> bool {
    matches!(
        interaction,
        InteractionType::MessagePassing
            | InteractionType::DataExchange
            | InteractionType::SharedMemory
    )
}

/// A directed view of the interactions between a test's threads.
struct InteractionGraph<'a> {
    edges: Vec<&'a ThreadInteraction>,
    successors: BTreeMap<u64, BTreeSet<u64>>,
    predecessors: BTreeMap<u64, BTreeSet<u64>>,
    nodes: BTreeSet<u64>,
}

impl<'a> InteractionGraph<'a> {
    /// Builds the graph from the interactions that pass `keep`.
    ///
    /// Self-interactions are dropped: a thread interacting with itself is not
    /// evidence of any inter-thread pattern.
    fn build(
        interactions: &'a [ThreadInteraction],
        keep: impl Fn(&ThreadInteraction) -> bool,
    ) -> Self {
        let mut graph = Self {
            edges: Vec::new(),
            successors: BTreeMap::new(),
            predecessors: BTreeMap::new(),
            nodes: BTreeSet::new(),
        };
        for interaction in interactions {
            if interaction.source_thread == interaction.target_thread || !keep(interaction) {
                continue;
            }
            graph.edges.push(interaction);
            graph.nodes.insert(interaction.source_thread);
            graph.nodes.insert(interaction.target_thread);
            graph
                .successors
                .entry(interaction.source_thread)
                .or_default()
                .insert(interaction.target_thread);
            graph
                .predecessors
                .entry(interaction.target_thread)
                .or_default()
                .insert(interaction.source_thread);
        }
        graph
    }

    fn out_degree(&self, node: u64) -> usize {
        self.successors.get(&node).map_or(0, BTreeSet::len)
    }

    fn in_degree(&self, node: u64) -> usize {
        self.predecessors.get(&node).map_or(0, BTreeSet::len)
    }

    fn has_edge(&self, from: u64, to: u64) -> bool {
        self.successors.get(&from).is_some_and(|targets| targets.contains(&to))
    }

    /// Number of recorded interactions with both endpoints in `participants`.
    fn edges_within(&self, participants: &BTreeSet<u64>) -> usize {
        self.edges
            .iter()
            .filter(|edge| {
                participants.contains(&edge.source_thread)
                    && participants.contains(&edge.target_thread)
            })
            .count()
    }
}

/// How many distinct threads the test recorded any interaction for.
///
/// This is the denominator for `applicability`: what fraction of the test's
/// interacting threads the detected pattern accounts for.
fn interacting_thread_count(test_data: &TestExecutionData) -> usize {
    let mut threads = BTreeSet::new();
    for interaction in &test_data.thread_interactions {
        threads.insert(interaction.source_thread);
        threads.insert(interaction.target_thread);
    }
    threads.len()
}

/// The error every detector returns when the test recorded no interactions.
fn no_interactions_recorded(pattern_type: &str) -> TestCharacterizationError {
    TestCharacterizationError::PatternRecognition {
        message: "the test recorded no thread interactions, so no concurrency pattern can be \
                  confirmed or ruled out"
            .to_string(),
        pattern_type: pattern_type.to_string(),
        context: HashMap::new(),
    }
}

/// Assembles the pattern record from measured counts.
fn build_pattern(
    pattern_type: &str,
    description: String,
    characteristics: Vec<String>,
    participants: usize,
    total_threads: usize,
    confidence: f64,
) -> ConcurrencyPattern {
    ConcurrencyPattern {
        pattern_type: pattern_type.to_string(),
        description,
        characteristics,
        // Share of the test's interacting threads this pattern accounts for.
        applicability: if total_threads == 0 {
            0.0
        } else {
            participants as f64 / total_threads as f64
        },
        confidence: confidence.clamp(0.0, 1.0),
        thread_count: participants,
    }
}

// ============================================================================
// PRODUCER / CONSUMER
// ============================================================================

/// Detects a producer/consumer split in the data-flow graph.
///
/// Producers are threads that only send data, consumers only receive it. The
/// confidence is the share of recorded data-flow interactions that run straight
/// from a producer to a consumer: a graph where data passes through
/// intermediaries is a pipeline, and scores accordingly low here.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProducerConsumerDetection;

impl ProducerConsumerDetection {
    /// Creates the detector. It holds no state; detection is a function of the
    /// execution data it is given.
    pub fn new() -> Self {
        Self
    }
}

impl PatternDetectionAlgorithm for ProducerConsumerDetection {
    fn detect(
        &self,
        test_data: &TestExecutionData,
    ) -> TestCharacterizationResult<Option<ConcurrencyPattern>> {
        if test_data.thread_interactions.is_empty() {
            return Err(no_interactions_recorded("ProducerConsumer"));
        }
        let graph = InteractionGraph::build(&test_data.thread_interactions, |i| {
            is_data_flow(i.interaction_type)
        });
        if graph.edges.is_empty() {
            return Ok(None);
        }

        let producers: BTreeSet<u64> = graph
            .nodes
            .iter()
            .copied()
            .filter(|&node| graph.out_degree(node) > 0 && graph.in_degree(node) == 0)
            .collect();
        let consumers: BTreeSet<u64> = graph
            .nodes
            .iter()
            .copied()
            .filter(|&node| graph.in_degree(node) > 0 && graph.out_degree(node) == 0)
            .collect();
        if producers.is_empty() || consumers.is_empty() {
            return Ok(None);
        }

        let direct = graph
            .edges
            .iter()
            .filter(|edge| {
                producers.contains(&edge.source_thread) && consumers.contains(&edge.target_thread)
            })
            .count();
        if direct == 0 {
            return Ok(None);
        }
        let confidence = direct as f64 / graph.edges.len() as f64;
        let participants = producers.len() + consumers.len();

        Ok(Some(build_pattern(
            "ProducerConsumer",
            format!(
                "{} producing thread(s) hand data to {} consuming thread(s)",
                producers.len(),
                consumers.len()
            ),
            vec![
                format!("{} producer thread(s)", producers.len()),
                format!("{} consumer thread(s)", consumers.len()),
                format!(
                    "{direct} of {} recorded data-flow interactions run producer to consumer",
                    graph.edges.len()
                ),
            ],
            participants,
            interacting_thread_count(test_data),
            confidence,
        )))
    }

    fn name(&self) -> &str {
        "ProducerConsumerDetection"
    }
}

// ============================================================================
// MASTER / WORKER
// ============================================================================

/// Detects a master thread coordinating a set of workers that do not talk to
/// each other.
///
/// The confidence is the share of interactions among the participating threads
/// that touch the master: a star is 1.0, and every worker-to-worker edge pulls
/// it down.
#[derive(Debug, Clone, Copy, Default)]
pub struct MasterWorkerDetection;

impl MasterWorkerDetection {
    /// Creates the detector. It holds no state.
    pub fn new() -> Self {
        Self
    }
}

impl PatternDetectionAlgorithm for MasterWorkerDetection {
    fn detect(
        &self,
        test_data: &TestExecutionData,
    ) -> TestCharacterizationResult<Option<ConcurrencyPattern>> {
        if test_data.thread_interactions.is_empty() {
            return Err(no_interactions_recorded("MasterWorker"));
        }
        let graph = InteractionGraph::build(&test_data.thread_interactions, |_| true);
        if graph.edges.is_empty() {
            return Ok(None);
        }

        let mut best: Option<(u64, BTreeSet<u64>, f64)> = None;
        for &candidate in &graph.nodes {
            let mut workers: BTreeSet<u64> = BTreeSet::new();
            if let Some(targets) = graph.successors.get(&candidate) {
                workers.extend(targets.iter().copied());
            }
            if let Some(sources) = graph.predecessors.get(&candidate) {
                workers.extend(sources.iter().copied());
            }
            workers.remove(&candidate);
            if workers.len() < 2 {
                continue;
            }
            // Workers in a master/worker shape coordinate only through the
            // master, so any worker-to-worker edge is counter-evidence.
            let mut participants = workers.clone();
            participants.insert(candidate);
            let total = graph.edges_within(&participants);
            if total == 0 {
                continue;
            }
            let through_master = graph
                .edges
                .iter()
                .filter(|edge| {
                    participants.contains(&edge.source_thread)
                        && participants.contains(&edge.target_thread)
                        && (edge.source_thread == candidate || edge.target_thread == candidate)
                })
                .count();
            let score = through_master as f64 / total as f64;
            if best.as_ref().is_none_or(|(_, current, current_score)| {
                score > *current_score || (score == *current_score && workers.len() > current.len())
            }) {
                best = Some((candidate, workers, score));
            }
        }

        let Some((master, workers, confidence)) = best else {
            return Ok(None);
        };
        Ok(Some(build_pattern(
            "MasterWorker",
            format!(
                "thread {master} coordinates {} worker thread(s)",
                workers.len()
            ),
            vec![
                format!("master thread {master}"),
                format!("{} worker thread(s)", workers.len()),
                format!(
                    "{:.0}% of interactions among the participants touch the master",
                    confidence * 100.0
                ),
            ],
            workers.len() + 1,
            interacting_thread_count(test_data),
            confidence,
        )))
    }

    fn name(&self) -> &str {
        "MasterWorkerDetection"
    }
}

// ============================================================================
// PIPELINE
// ============================================================================

/// Detects a chain of threads that hand data along, stage by stage.
///
/// A stage has exactly one downstream successor in the data-flow graph, so the
/// chain is followed deterministically from each source thread. Three stages is
/// the shortest chain that is a pipeline rather than a single hand-off.
#[derive(Debug, Clone, Copy, Default)]
pub struct PipelineDetection;

impl PipelineDetection {
    /// Creates the detector. It holds no state.
    pub fn new() -> Self {
        Self
    }
}

impl PatternDetectionAlgorithm for PipelineDetection {
    fn detect(
        &self,
        test_data: &TestExecutionData,
    ) -> TestCharacterizationResult<Option<ConcurrencyPattern>> {
        if test_data.thread_interactions.is_empty() {
            return Err(no_interactions_recorded("Pipeline"));
        }
        let graph = InteractionGraph::build(&test_data.thread_interactions, |i| {
            is_data_flow(i.interaction_type)
        });
        if graph.edges.is_empty() {
            return Ok(None);
        }

        let mut longest: Vec<u64> = Vec::new();
        for &start in &graph.nodes {
            if graph.in_degree(start) != 0 {
                continue;
            }
            let mut chain = vec![start];
            let mut seen: BTreeSet<u64> = BTreeSet::new();
            seen.insert(start);
            let mut current = start;
            while let Some(successors) = graph.successors.get(&current) {
                if successors.len() != 1 {
                    break;
                }
                let Some(&next) = successors.iter().next() else {
                    break;
                };
                if !seen.insert(next) {
                    break;
                }
                chain.push(next);
                current = next;
            }
            if chain.len() > longest.len() {
                longest = chain;
            }
        }
        if longest.len() < 3 {
            return Ok(None);
        }

        let stage_edges: Vec<&ThreadInteraction> = longest
            .windows(2)
            .filter_map(|pair| {
                graph
                    .edges
                    .iter()
                    .copied()
                    .find(|edge| edge.source_thread == pair[0] && edge.target_thread == pair[1])
            })
            .collect();
        // A pipeline runs no faster than its slowest stage, so the bottleneck
        // hand-off rate is the pipeline's measured throughput.
        let throughput =
            stage_edges.iter().map(|edge| edge.frequency).fold(f64::INFINITY, f64::min);
        let confidence = stage_edges.len() as f64 / graph.edges.len() as f64;

        Ok(Some(build_pattern(
            "Pipeline",
            format!("{} stages hand data along in sequence", longest.len()),
            vec![
                format!("{} pipeline stage(s)", longest.len()),
                format!(
                    "bottleneck stage hand-off rate {:.3} interactions/s",
                    if throughput.is_finite() { throughput } else { 0.0 }
                ),
                format!(
                    "{} of {} recorded data-flow interactions lie on the chain",
                    stage_edges.len(),
                    graph.edges.len()
                ),
            ],
            longest.len(),
            interacting_thread_count(test_data),
            confidence,
        )))
    }

    fn name(&self) -> &str {
        "PipelineDetection"
    }
}

// ============================================================================
// FORK / JOIN
// ============================================================================

/// Detects a thread that fans work out to several others which then fan back in
/// to a single collector.
///
/// The collector may be the forking thread itself, which is the common
/// scatter/gather shape.
#[derive(Debug, Clone, Copy, Default)]
pub struct ForkJoinDetection;

impl ForkJoinDetection {
    /// Creates the detector. It holds no state.
    pub fn new() -> Self {
        Self
    }
}

impl PatternDetectionAlgorithm for ForkJoinDetection {
    fn detect(
        &self,
        test_data: &TestExecutionData,
    ) -> TestCharacterizationResult<Option<ConcurrencyPattern>> {
        if test_data.thread_interactions.is_empty() {
            return Err(no_interactions_recorded("ForkJoin"));
        }
        let graph = InteractionGraph::build(&test_data.thread_interactions, |_| true);
        if graph.edges.is_empty() {
            return Ok(None);
        }

        let mut fork_points: BTreeSet<u64> = BTreeSet::new();
        let mut join_points: BTreeSet<u64> = BTreeSet::new();
        let mut participants: BTreeSet<u64> = BTreeSet::new();
        for &fork in &graph.nodes {
            let Some(branches) = graph.successors.get(&fork) else {
                continue;
            };
            if branches.len() < 2 {
                continue;
            }
            for &join in &graph.nodes {
                let rejoining: BTreeSet<u64> = branches
                    .iter()
                    .copied()
                    .filter(|&branch| branch != join && graph.has_edge(branch, join))
                    .collect();
                if rejoining.len() < 2 {
                    continue;
                }
                fork_points.insert(fork);
                join_points.insert(join);
                participants.insert(fork);
                participants.insert(join);
                participants.extend(rejoining);
            }
        }
        if fork_points.is_empty() {
            return Ok(None);
        }

        let within = graph.edges_within(&participants);
        let structural = graph
            .edges
            .iter()
            .filter(|edge| {
                (fork_points.contains(&edge.source_thread)
                    && participants.contains(&edge.target_thread))
                    || (join_points.contains(&edge.target_thread)
                        && participants.contains(&edge.source_thread))
            })
            .count();
        let confidence = if within == 0 { 0.0 } else { structural as f64 / within as f64 };

        Ok(Some(build_pattern(
            "ForkJoin",
            format!(
                "{} fork point(s) fan out to branches that rejoin at {} point(s)",
                fork_points.len(),
                join_points.len()
            ),
            vec![
                format!("{} fork point(s)", fork_points.len()),
                format!("{} join point(s)", join_points.len()),
                format!(
                    "{structural} of {within} interactions among the participants are fork or \
                     join edges"
                ),
            ],
            participants.len(),
            interacting_thread_count(test_data),
            confidence,
        )))
    }

    fn name(&self) -> &str {
        "ForkJoinDetection"
    }
}

#[cfg(test)]
#[path = "pattern_algorithms_tests.rs"]
mod pattern_algorithms_tests;
