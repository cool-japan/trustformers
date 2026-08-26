//! Async inference job queue and status store.
//!
//! `POST /inference/async` enqueues a real job; a worker task drives it through
//! the batching service and records the real outcome, which
//! `GET /jobs/{id}/status` reads back. Nothing here derives status from the shape
//! of the job id, and no result is invented.

use crate::batching::{
    aggregator::{ProcessingOutput, RequestInput},
    config::Priority,
    DynamicBatchingService, Request, RequestId,
};
use dashmap::DashMap;
use serde::Serialize;
use std::sync::Arc;
use std::time::Instant;

/// Lifecycle state of an async inference job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    /// Accepted and queued, not yet picked up by a worker.
    Pending,
    /// A worker is currently driving the job.
    Processing,
    /// The job finished and produced output.
    Completed,
    /// The job finished with an error.
    Failed,
}

impl JobState {
    /// Wire representation used by `GET /jobs/{id}/status`.
    pub fn as_str(&self) -> &'static str {
        match self {
            JobState::Pending => "pending",
            JobState::Processing => "processing",
            JobState::Completed => "completed",
            JobState::Failed => "failed",
        }
    }
}

/// A recorded async job.
#[derive(Debug, Clone)]
pub struct JobRecord {
    pub job_id: String,
    pub model: String,
    pub state: JobState,
    /// Real generated text, present only once the job completed.
    pub result_text: Option<String>,
    /// Real error message, present only when the job failed.
    pub error: Option<String>,
    /// Measured wall-clock processing time, in milliseconds.
    pub processing_time_ms: Option<f64>,
    /// Callback URL supplied by the submitter, if any.
    pub callback_url: Option<String>,
}

impl JobRecord {
    /// Render the record as the JSON payload of `GET /jobs/{id}/status`.
    pub fn result_payload(&self) -> Option<serde_json::Value> {
        match self.state {
            JobState::Completed => {
                let text = self.result_text.clone().unwrap_or_default();
                Some(serde_json::json!({
                    "text": text,
                    "tokens": text.split_whitespace().collect::<Vec<_>>(),
                    "processing_time_ms": self.processing_time_ms.unwrap_or(0.0),
                }))
            },
            JobState::Failed => Some(serde_json::json!({
                "error": self.error.clone().unwrap_or_default(),
                "processing_time_ms": self.processing_time_ms.unwrap_or(0.0),
            })),
            JobState::Pending | JobState::Processing => None,
        }
    }
}

/// In-process registry of async inference jobs.
#[derive(Debug, Default)]
pub struct JobStore {
    jobs: DashMap<String, JobRecord>,
}

impl JobStore {
    pub fn new() -> Self {
        Self {
            jobs: DashMap::new(),
        }
    }

    /// Register a newly accepted job in the `Pending` state.
    pub fn enqueue(&self, job_id: String, model: String, callback_url: Option<String>) {
        self.jobs.insert(
            job_id.clone(),
            JobRecord {
                job_id,
                model,
                state: JobState::Pending,
                result_text: None,
                error: None,
                processing_time_ms: None,
                callback_url,
            },
        );
    }

    /// Look up a job by id. `None` means the job is genuinely unknown.
    pub fn get(&self, job_id: &str) -> Option<JobRecord> {
        self.jobs.get(job_id).map(|entry| entry.clone())
    }

    /// Number of jobs currently tracked.
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Whether the store holds no jobs.
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Count of jobs in a given state.
    pub fn count_in_state(&self, state: JobState) -> usize {
        self.jobs.iter().filter(|entry| entry.state == state).count()
    }

    fn mark_processing(&self, job_id: &str) {
        if let Some(mut entry) = self.jobs.get_mut(job_id) {
            entry.state = JobState::Processing;
        }
    }

    fn mark_completed(&self, job_id: &str, text: String, elapsed_ms: f64) {
        if let Some(mut entry) = self.jobs.get_mut(job_id) {
            entry.state = JobState::Completed;
            entry.result_text = Some(text);
            entry.processing_time_ms = Some(elapsed_ms);
        }
    }

    fn mark_failed(&self, job_id: &str, error: String, elapsed_ms: f64) {
        if let Some(mut entry) = self.jobs.get_mut(job_id) {
            entry.state = JobState::Failed;
            entry.error = Some(error);
            entry.processing_time_ms = Some(elapsed_ms);
        }
    }
}

/// Drive one async job to completion against the batching service.
///
/// Runs on a spawned task; the job record is updated at every transition so a
/// concurrent `GET /jobs/{id}/status` always reports the true state.
pub async fn run_job(
    store: Arc<JobStore>,
    batching_service: Arc<DynamicBatchingService>,
    job_id: String,
    text: String,
    max_length: Option<usize>,
) {
    store.mark_processing(&job_id);
    let started = Instant::now();

    let request = Request {
        id: RequestId::new(),
        input: RequestInput::Text { text, max_length },
        priority: Priority::Normal,
        submitted_at: Instant::now(),
        deadline: None,
        metadata: std::collections::HashMap::new(),
    };

    match batching_service.submit_request(request).await {
        Ok(result) => {
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            match result.output {
                ProcessingOutput::Text(text) => store.mark_completed(&job_id, text, elapsed),
                ProcessingOutput::Tokens(tokens) => store.mark_completed(
                    &job_id,
                    tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "),
                    elapsed,
                ),
                ProcessingOutput::Error(error) => store.mark_failed(&job_id, error, elapsed),
                other => store.mark_failed(
                    &job_id,
                    format!("unsupported output type: {:?}", other),
                    elapsed,
                ),
            }
        },
        Err(e) => {
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            store.mark_failed(&job_id, e.to_string(), elapsed);
        },
    }

    // Notify the submitter, if they asked to be notified. A failed callback is
    // logged; it never changes the recorded job outcome.
    let record = store.get(&job_id);
    if let Some(record) = record {
        if let Some(url) = record.callback_url.clone() {
            let payload = serde_json::json!({
                "job_id": record.job_id,
                "status": record.state.as_str(),
                "result": record.result_payload(),
            });
            match reqwest::Client::new().post(&url).json(&payload).send().await {
                Ok(response) => {
                    tracing::debug!(
                        "async job {} callback to {} returned {}",
                        record.job_id,
                        url,
                        response.status()
                    );
                },
                Err(e) => {
                    tracing::warn!(
                        "async job {} callback to {} failed: {}",
                        record.job_id,
                        url,
                        e
                    );
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batching::config::BatchingConfig;

    /// Regression: job status must come from the store, and an unknown id must
    /// be reported as unknown rather than derived from `job_id.len() % 3`.
    #[test]
    fn unknown_job_is_unknown() {
        let store = JobStore::new();
        assert!(store.get("abc").is_none());
        assert!(store.get("abcdef").is_none());
        assert!(store.is_empty());
    }

    #[test]
    fn enqueued_job_starts_pending_with_no_result() {
        let store = JobStore::new();
        store.enqueue("job-1".to_string(), "m".to_string(), None);
        let record = store.get("job-1").expect("job recorded");
        assert_eq!(record.state, JobState::Pending);
        assert!(record.result_payload().is_none());
        assert_eq!(store.count_in_state(JobState::Pending), 1);
    }

    /// Regression: a job driven against a model-less batching service must be
    /// recorded as failed with the real reason, not "Mock async inference result".
    #[tokio::test]
    async fn job_without_model_records_a_real_failure() {
        let store = Arc::new(JobStore::new());
        let batching = Arc::new(DynamicBatchingService::new(BatchingConfig::default()));
        batching.start().await.expect("service starts");

        store.enqueue("job-2".to_string(), "m".to_string(), None);
        run_job(
            Arc::clone(&store),
            batching,
            "job-2".to_string(),
            "hello".to_string(),
            Some(4),
        )
        .await;

        let record = store.get("job-2").expect("job recorded");
        assert_eq!(record.state, JobState::Failed);
        let error = record.error.clone().unwrap_or_default();
        assert!(
            error.contains("no model configured"),
            "expected the real reason, got {error:?}"
        );
        let payload = record.result_payload().expect("failed jobs carry a payload");
        assert!(payload["error"].as_str().unwrap_or_default().contains("no model"));
    }
}
