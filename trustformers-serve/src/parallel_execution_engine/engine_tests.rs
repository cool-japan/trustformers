//! Tests for the parallel execution engine's real execution path.
//!
//! These lock in the 0.2.1 fix: before it, `execute_parallel` accepted any
//! number of tests, ran none of them, and returned an empty result set through
//! a scheduler that dropped everything -- while the (unreachable) result
//! builder manufactured constants such as `parallel_efficiency: 0.85` and
//! `speedup_factor: 2.0`.

use super::engine::{ParallelExecutionEngine, TestBody, TestBodyFuture, TestRunner};
use super::scheduling_tests::scheduled;
use crate::test_independence_analyzer::{
    AnalysisMetadata, AnalysisPerformanceMetrics, AnalysisQualityAssessment,
    TestIndependenceAnalysis,
};
use crate::test_parallelization::TestParallelizationConfig;
use crate::test_timeout_optimization::{
    TestOutcome, TestProgressTracker, TestTimeoutConfig, TestTimeoutFramework,
};
use chrono::Utc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// A runner that records how many bodies it was asked for and actually runs
/// them. Nothing here is stubbed: the bodies really execute.
struct CountingRunner {
    runs: Arc<AtomicUsize>,
    /// Names this runner refuses to supply a body for.
    unknown: Vec<String>,
}

impl TestRunner for CountingRunner {
    fn test_body(&self, test_name: &str) -> Option<TestBody> {
        if self.unknown.iter().any(|name| name == test_name) {
            return None;
        }
        let runs = Arc::clone(&self.runs);
        let body: TestBody = Box::new(move |_progress: Arc<TestProgressTracker>| {
            let future: TestBodyFuture = Box::pin(async move {
                runs.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(1)).await;
                Ok(())
            });
            future
        });
        Some(body)
    }
}

fn analysis(test_ids: &[&str]) -> TestIndependenceAnalysis {
    TestIndependenceAnalysis {
        tests: test_ids.iter().map(|id| scheduled(id, 0.5).metadata).collect(),
        dependencies: Vec::new(),
        conflicts: Vec::new(),
        groups: Vec::new(),
        analysis_metadata: AnalysisMetadata {
            started_at: Utc::now(),
            completed_at: Utc::now(),
            analysis_duration: Duration::from_millis(1),
            analyzer_version: env!("CARGO_PKG_VERSION").to_string(),
            configuration_summary: "unit test".to_string(),
            analysis_quality: 1.0,
            recommendations: Vec::new(),
        },
        performance_metrics: AnalysisPerformanceMetrics::default(),
        quality_assessment: AnalysisQualityAssessment {
            overall_score: 1.0,
            dependency_quality: 1.0,
            conflict_quality: 1.0,
            grouping_quality: 1.0,
            completeness_score: 1.0,
            confidence_level: 1.0,
            quality_issues: Vec::new(),
        },
    }
}

async fn engine() -> ParallelExecutionEngine {
    let framework = Arc::new(
        TestTimeoutFramework::new(TestTimeoutConfig::default())
            .unwrap_or_else(|e| panic!("timeout framework construction failed: {e}")),
    );
    ParallelExecutionEngine::new(TestParallelizationConfig::default(), framework)
        .await
        .unwrap_or_else(|e| panic!("engine construction failed: {e}"))
}

#[tokio::test]
async fn execute_parallel_refuses_without_a_runner() {
    let mut engine = engine().await;
    assert!(!engine.has_test_runner());
    let error = engine
        .execute_parallel(analysis(&["alpha"]))
        .await
        .err()
        .unwrap_or_else(|| panic!("engine invented results instead of refusing"));
    let message = format!("{error}");
    assert!(
        message.contains("no TestRunner installed"),
        "error must name what is missing, got: {message}"
    );
}

#[tokio::test]
async fn execute_parallel_accepts_an_empty_analysis_without_a_runner() {
    let mut engine = engine().await;
    let outcomes = engine
        .execute_parallel(analysis(&[]))
        .await
        .unwrap_or_else(|e| panic!("empty analysis should succeed: {e}"));
    assert!(outcomes.is_empty());
}

#[tokio::test]
async fn execute_parallel_really_runs_the_supplied_bodies() {
    let runs = Arc::new(AtomicUsize::new(0));
    let runner = Arc::new(CountingRunner {
        runs: Arc::clone(&runs),
        unknown: Vec::new(),
    });
    let mut engine = engine().await.with_test_runner(runner);
    assert!(engine.has_test_runner());

    let outcomes = engine
        .execute_parallel(analysis(&["alpha", "beta", "gamma"]))
        .await
        .unwrap_or_else(|e| panic!("execution failed: {e}"));

    // Before 0.2.1 this was 0: the scheduler dropped every test.
    assert_eq!(
        runs.load(Ordering::SeqCst),
        3,
        "every body must actually run"
    );
    assert_eq!(outcomes.len(), 3);

    for outcome in &outcomes {
        assert!(matches!(outcome.result.outcome, TestOutcome::Success));
        // Measured, not stamped: a body that slept must show non-zero time.
        assert!(outcome.result.execution_time > Duration::ZERO);
        assert!(outcome.completed_at >= outcome.started_at);
        assert!(outcome.started_at >= outcome.scheduled_at);
        assert!(outcome.observed_concurrency >= 1);
    }

    let stats = engine.statistics();
    assert_eq!(stats.total_tests_executed.load(Ordering::SeqCst), 3);
    assert!(stats.average_parallelism().is_some());
    assert!(stats.average_execution_time().is_some());
    assert!(stats.peak_parallelism.load(Ordering::SeqCst) >= 1);
}

#[tokio::test]
async fn execute_parallel_reports_a_test_the_runner_cannot_supply() {
    let runner = Arc::new(CountingRunner {
        runs: Arc::new(AtomicUsize::new(0)),
        unknown: vec!["orphan".to_string()],
    });
    let mut engine = engine().await.with_test_runner(runner);
    let error = engine
        .execute_parallel(analysis(&["orphan"]))
        .await
        .err()
        .unwrap_or_else(|| panic!("engine invented a result for an unrunnable test"));
    let message = format!("{error}");
    assert!(
        message.contains("supplies no body"),
        "error must name the missing body, got: {message}"
    );
}

#[tokio::test]
async fn engine_statistics_are_absent_before_any_execution() {
    let engine = engine().await;
    let stats = engine.statistics();
    assert_eq!(stats.total_tests_executed.load(Ordering::SeqCst), 0);
    assert!(stats.average_parallelism().is_none());
    assert!(stats.average_execution_time().is_none());
}
