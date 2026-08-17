// Allow dead code for infrastructure under development
#![allow(dead_code)]

//! Main GDPR compliance service implementation
//!
//! This module contains the core GdprComplianceService that orchestrates
//! all GDPR compliance functionality and provides the main API.

use anyhow::Result;
use once_cell::sync::Lazy;
use prometheus::{
    register_counter_vec, register_gauge_vec, register_histogram_vec, Counter, CounterVec, Gauge,
    GaugeVec, Histogram, HistogramVec,
};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use uuid::Uuid;

// Global metric registrations — registered exactly once, shared across all GdprComplianceService instances.
// Registration only fails on duplicate registration; fall back to an unregistered
// metric (compile-time-valid opts) so the server keeps running.
static GDPR_SUBJECT_REQUESTS_TOTAL: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "gdpr_subject_requests_total",
        "Total number of data subject requests",
        &["type"]
    )
    .unwrap_or_else(|_| {
        prometheus::CounterVec::new(
            prometheus::opts!(
                "gdpr_subject_requests_total",
                "Total number of data subject requests"
            ),
            &["type"],
        )
        .unwrap_or_else(|_| unreachable!("static prometheus opts are valid"))
    })
});

static GDPR_ACTIVE_CONSENTS: Lazy<GaugeVec> = Lazy::new(|| {
    register_gauge_vec!(
        "gdpr_active_consents",
        "Number of active consents",
        &["purpose"]
    )
    .unwrap_or_else(|_| {
        prometheus::GaugeVec::new(
            prometheus::opts!("gdpr_active_consents", "Number of active consents"),
            &["purpose"],
        )
        .unwrap_or_else(|_| unreachable!("static prometheus opts are valid"))
    })
});

static GDPR_REQUEST_PROCESSING_DURATION: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "gdpr_request_processing_duration_seconds",
        "Duration of request processing",
        &["type"]
    )
    .unwrap_or_else(|_| {
        prometheus::HistogramVec::new(
            prometheus::histogram_opts!(
                "gdpr_request_processing_duration_seconds",
                "Duration of request processing"
            ),
            &["type"],
        )
        .unwrap_or_else(|_| unreachable!("static prometheus opts are valid"))
    })
});

use super::consent_management::{ConsentEvidence, ConsentMechanism, ConsentRecord};
use super::data_subject_rights::{DataSubjectRequest, RequestDetails};
use super::types::{
    ConsentStatus, DataSubject, GdprComplianceConfig, RequestStatus, RequestType,
    VerificationStatus,
};

/// Main GDPR compliance service
#[derive(Debug)]
pub struct GdprComplianceService {
    /// Service configuration
    config: GdprComplianceConfig,
    /// Data subjects registry
    data_subjects: Arc<RwLock<HashMap<String, DataSubject>>>,
    /// Consent records
    consent_records: Arc<RwLock<HashMap<String, ConsentRecord>>>,
    /// Processing activities, keyed by `"{subject_id}:{activity_id}"`.
    processing_activities: Arc<RwLock<HashMap<String, ProcessingActivityRecord>>>,
    /// Data subject requests
    subject_requests: Arc<RwLock<HashMap<String, DataSubjectRequest>>>,
    /// Subjects whose processing is restricted under Article 18, mapped to the
    /// processing activities that are restricted (empty set = all activities).
    restricted_subjects: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    /// Recorded Article 22 requests awaiting human review, in submission order.
    human_review_queue: Arc<RwLock<Vec<String>>>,
    /// Service statistics
    stats: Arc<GdprComplianceStats>,
    /// Prometheus metrics
    prometheus_metrics: Arc<GdprPrometheusMetrics>,
}

/// A record of one processing activity performed on a data subject's data.
///
/// This is the store that access, portability, restriction, objection and
/// erasure requests actually operate on — the handlers read and mutate these
/// records rather than reporting a canned success string.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProcessingActivityRecord {
    /// Activity identifier (unique per subject).
    pub activity_id: String,
    /// Subject whose data this activity processes.
    pub subject_id: String,
    /// Purpose of the processing.
    pub purpose: String,
    /// Legal basis relied upon.
    pub legal_basis: String,
    /// Categories of data touched by the activity.
    pub data_categories: Vec<String>,
    /// The stored payload for this activity, as field/value pairs.
    pub data: HashMap<String, String>,
    /// When the activity was recorded.
    pub recorded_at: SystemTime,
    /// Whether processing is currently restricted (Article 18).
    pub restricted: bool,
}

/// Result of a data-retention sweep.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RetentionSweepReport {
    /// Consent records deleted because they exceeded the retention period.
    pub consents_deleted: usize,
    /// Processing activity records deleted because they exceeded the period.
    pub activities_deleted: usize,
    /// Data subjects deleted because no data referencing them remained.
    pub subjects_deleted: usize,
}

/// Result of a consent-renewal sweep.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConsentRenewalReport {
    /// Consents transitioned from `Given` to `Expired`.
    pub expired: usize,
    /// Consents that will expire inside the configured notice period.
    pub due_for_renewal: usize,
}

/// GDPR compliance statistics
#[derive(Debug, Default)]
pub struct GdprComplianceStats {
    /// Total number of data subjects
    pub total_data_subjects: AtomicU64,
    /// Total consent records
    pub total_consent_records: AtomicU64,
    /// Total data subject requests
    pub total_subject_requests: AtomicU64,
    /// Completed requests
    pub completed_requests: AtomicU64,
    /// Breaches detected
    pub breaches_detected: AtomicU64,
    /// Compliance violations
    pub compliance_violations: AtomicU64,
}

/// Prometheus metrics for GDPR compliance
#[derive(Debug)]
pub struct GdprPrometheusMetrics {
    /// Subject requests counter
    pub subject_requests_total: Counter,
    /// Active consents gauge
    pub active_consents: Gauge,
    /// Request processing duration histogram
    pub request_processing_duration: Histogram,
}

impl GdprPrometheusMetrics {
    /// Create new Prometheus metrics.
    ///
    /// Uses globally lazy-initialized metric families so that multiple
    /// `GdprComplianceService` instances (e.g. in tests) share the same
    /// Prometheus registration without triggering duplicate-registration errors.
    pub fn new() -> Result<Self> {
        Ok(Self {
            subject_requests_total: GDPR_SUBJECT_REQUESTS_TOTAL.with_label_values(&["all"]),
            active_consents: GDPR_ACTIVE_CONSENTS.with_label_values(&["all"]),
            request_processing_duration: GDPR_REQUEST_PROCESSING_DURATION
                .with_label_values(&["all"]),
        })
    }
}

/// What [`GdprComplianceService::start`] actually did on this invocation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StartupReport {
    /// Result of the initial retention sweep, if retention is enabled.
    pub retention: Option<RetentionSweepReport>,
    /// Result of the initial consent renewal sweep, if renewal is enabled.
    pub consent_renewal: Option<ConsentRenewalReport>,
    /// Requests awaiting processing at startup.
    pub pending_requests: u64,
    /// Requests already past their statutory response deadline at startup.
    pub overdue_requests: u64,
}

/// Request processing result
#[derive(Debug, Clone)]
pub enum RequestProcessingResult {
    /// Request processed successfully
    Success { data: String },
    /// Request partially processed
    Partial { reason: String },
    /// Request rejected
    Rejected { reason: String },
}

/// Compliance status
#[derive(Debug, Clone)]
pub struct ComplianceStatus {
    /// Total data subjects
    pub total_data_subjects: u64,
    /// Active consents count
    pub active_consents: u64,
    /// Pending requests count
    pub pending_requests: u64,
    /// Overdue requests count
    pub overdue_requests: u64,
    /// Compliance score (0.0 to 1.0)
    pub compliance_score: f64,
    /// Last assessment timestamp
    pub last_assessment: SystemTime,
}

impl GdprComplianceService {
    /// Create a new GDPR compliance service
    pub fn new(config: GdprComplianceConfig) -> Result<Self> {
        Ok(Self {
            config,
            data_subjects: Arc::new(RwLock::new(HashMap::new())),
            consent_records: Arc::new(RwLock::new(HashMap::new())),
            processing_activities: Arc::new(RwLock::new(HashMap::new())),
            subject_requests: Arc::new(RwLock::new(HashMap::new())),
            restricted_subjects: Arc::new(RwLock::new(HashMap::new())),
            human_review_queue: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(GdprComplianceStats::default()),
            prometheus_metrics: Arc::new(GdprPrometheusMetrics::new()?),
        })
    }

    /// Start the GDPR compliance service.
    ///
    /// This performs one immediate pass of every enabled maintenance job and
    /// returns what it actually did. Periodic scheduling is the caller's
    /// responsibility: invoke [`Self::run_retention_sweep`],
    /// [`Self::run_consent_renewal_sweep`] and [`Self::get_compliance_status`]
    /// on the interval that suits the deployment. Nothing here spawns a hidden
    /// background task, and nothing reports work it did not do.
    ///
    /// # Errors
    ///
    /// Propagates failures from the Prometheus metric update.
    pub async fn start(&self) -> Result<StartupReport> {
        if !self.config.enabled {
            return Ok(StartupReport::default());
        }

        let retention = if self.config.data_retention.enabled {
            Some(self.run_retention_sweep().await?)
        } else {
            None
        };

        let consent_renewal = if self.config.consent_management.renewal.enabled {
            Some(self.run_consent_renewal_sweep().await?)
        } else {
            None
        };

        let compliance = self.get_compliance_status().await;

        Ok(StartupReport {
            retention,
            consent_renewal,
            pending_requests: compliance.pending_requests,
            overdue_requests: compliance.overdue_requests,
        })
    }

    /// Register (or replace) a data subject in the compliance registry.
    ///
    /// Nothing else in this service can honour an access, erasure or
    /// portability request for a subject that was never registered, so this is
    /// the entry point for the data an operator is accountable for.
    ///
    /// # Errors
    ///
    /// Propagates failures from the Prometheus metric update.
    pub async fn register_data_subject(&self, subject: DataSubject) -> Result<()> {
        let mut subjects = self.data_subjects.write().await;
        let is_new = subjects.insert(subject.id.clone(), subject).is_none();
        drop(subjects);
        if is_new {
            self.stats.total_data_subjects.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Look up a registered data subject.
    pub async fn get_data_subject(&self, subject_id: &str) -> Option<DataSubject> {
        self.data_subjects.read().await.get(subject_id).cloned()
    }

    /// List every registered data subject id.
    pub async fn list_data_subjects(&self) -> Vec<String> {
        self.data_subjects.read().await.keys().cloned().collect()
    }

    /// Record a processing activity performed on a subject's data.
    ///
    /// Returns the composite key the record is stored under.
    ///
    /// # Errors
    ///
    /// Returns an error if the subject is not registered — recording processing
    /// against an unknown subject would make the Article 30 record of
    /// processing activities untraceable.
    pub async fn record_processing_activity(
        &self,
        activity: ProcessingActivityRecord,
    ) -> Result<String> {
        if !self.data_subjects.read().await.contains_key(&activity.subject_id) {
            return Err(anyhow::anyhow!(
                "cannot record processing for unregistered data subject: {}",
                activity.subject_id
            ));
        }
        let key = Self::activity_key(&activity.subject_id, &activity.activity_id);
        self.processing_activities.write().await.insert(key.clone(), activity);
        Ok(key)
    }

    /// List the processing activities recorded for a subject.
    pub async fn list_processing_activities(
        &self,
        subject_id: &str,
    ) -> Vec<ProcessingActivityRecord> {
        self.processing_activities
            .read()
            .await
            .values()
            .filter(|record| record.subject_id == subject_id)
            .cloned()
            .collect()
    }

    /// The Article 22 human-review queue, in submission order.
    pub async fn human_review_queue(&self) -> Vec<String> {
        self.human_review_queue.read().await.clone()
    }

    /// Delete every consent record and processing activity that has outlived
    /// the configured retention period, then delete any data subject with no
    /// remaining records.
    ///
    /// # Errors
    ///
    /// Propagates failures from the Prometheus metric update.
    pub async fn run_retention_sweep(&self) -> Result<RetentionSweepReport> {
        let retention = self.config.data_retention.default_retention_period;
        let now = SystemTime::now();
        let mut report = RetentionSweepReport::default();

        let expired = |timestamp: SystemTime| -> bool {
            now.duration_since(timestamp).map(|age| age > retention).unwrap_or(false)
        };

        {
            let mut consents = self.consent_records.write().await;
            let before = consents.len();
            consents.retain(|_, record| !expired(record.given_at));
            report.consents_deleted = before - consents.len();
        }

        {
            let mut activities = self.processing_activities.write().await;
            let before = activities.len();
            activities.retain(|_, record| !expired(record.recorded_at));
            report.activities_deleted = before - activities.len();
        }

        if self.config.data_retention.auto_deletion {
            let referenced: HashSet<String> = {
                let consents = self.consent_records.read().await;
                let activities = self.processing_activities.read().await;
                consents
                    .values()
                    .map(|record| record.subject_id.clone())
                    .chain(activities.values().map(|record| record.subject_id.clone()))
                    .collect()
            };
            let mut subjects = self.data_subjects.write().await;
            let before = subjects.len();
            subjects
                .retain(|id, subject| referenced.contains(id) || !expired(subject.last_activity));
            report.subjects_deleted = before - subjects.len();
            let remaining = subjects.len() as u64;
            drop(subjects);
            self.stats.total_data_subjects.store(remaining, Ordering::Relaxed);
        }

        self.update_prometheus_metrics().await?;
        Ok(report)
    }

    /// Mark consents whose expiry has passed as [`ConsentStatus::Expired`] and
    /// count the ones falling due inside the configured notice period.
    ///
    /// # Errors
    ///
    /// Propagates failures from the Prometheus metric update.
    pub async fn run_consent_renewal_sweep(&self) -> Result<ConsentRenewalReport> {
        let notice = self.config.consent_management.renewal.notice_period;
        let auto_expiry = self.config.consent_management.renewal.auto_expiry;
        let now = SystemTime::now();
        let mut report = ConsentRenewalReport::default();

        {
            let mut consents = self.consent_records.write().await;
            for record in consents.values_mut() {
                let Some(expires_at) = record.expires_at else {
                    continue;
                };
                if !matches!(record.status, ConsentStatus::Given) {
                    continue;
                }
                if expires_at <= now {
                    if auto_expiry {
                        record.status = ConsentStatus::Expired;
                        report.expired += 1;
                    }
                } else if expires_at
                    .duration_since(now)
                    .map(|remaining| remaining <= notice)
                    .unwrap_or(false)
                {
                    report.due_for_renewal += 1;
                }
            }
        }

        self.update_prometheus_metrics().await?;
        Ok(report)
    }

    fn activity_key(subject_id: &str, activity_id: &str) -> String {
        format!("{subject_id}:{activity_id}")
    }

    /// Record consent
    pub async fn record_consent(
        &self,
        subject_id: &str,
        purpose: &str,
        mechanism: ConsentMechanism,
    ) -> Result<String> {
        let record_id = Uuid::new_v4().to_string();
        let record = ConsentRecord {
            id: record_id.clone(),
            subject_id: subject_id.to_string(),
            purpose: purpose.to_string(),
            status: ConsentStatus::Given,
            mechanism,
            given_at: SystemTime::now(),
            withdrawn_at: None,
            // Real expiry derived from the configured renewal period, so the
            // renewal sweep and the compliance score have something to act on.
            expires_at: if self.config.consent_management.renewal.enabled {
                SystemTime::now().checked_add(self.config.consent_management.renewal.renewal_period)
            } else {
                None
            },
            evidence: ConsentEvidence {
                ip_address: None, // Would be filled from request context
                user_agent: None,
                timestamp: SystemTime::now(),
                digital_signature: None,
                witness: None,
                metadata: HashMap::new(),
            },
            version: "1.0".to_string(),
        };

        self.consent_records.write().await.insert(record_id.clone(), record);
        self.stats.total_consent_records.fetch_add(1, Ordering::Relaxed);
        self.update_prometheus_metrics().await?;

        Ok(record_id)
    }

    /// Withdraw consent
    pub async fn withdraw_consent(&self, subject_id: &str, purpose: &str) -> Result<()> {
        let mut records = self.consent_records.write().await;

        for record in records.values_mut() {
            if record.subject_id == subject_id && record.purpose == purpose {
                record.status = ConsentStatus::Withdrawn;
                record.withdrawn_at = Some(SystemTime::now());
                break;
            }
        }

        self.update_prometheus_metrics().await?;
        Ok(())
    }

    /// Submit data subject request
    pub async fn submit_data_subject_request(
        &self,
        subject_id: &str,
        request_type: RequestType,
        details: RequestDetails,
    ) -> Result<String> {
        let request_id = Uuid::new_v4().to_string();

        let request = DataSubjectRequest {
            id: request_id.clone(),
            subject_id: subject_id.to_string(),
            request_type,
            status: RequestStatus::Submitted,
            details,
            verification_status: VerificationStatus::Pending,
            submitted_at: SystemTime::now(),
            completed_at: None,
        };

        self.subject_requests.write().await.insert(request_id.clone(), request);
        self.stats.total_subject_requests.fetch_add(1, Ordering::Relaxed);
        self.prometheus_metrics.subject_requests_total.inc();

        Ok(request_id)
    }

    /// Process data subject request.
    ///
    /// The request is cloned out of the registry before the handlers run so the
    /// handlers can take their own locks (including on `subject_requests`
    /// itself, which erasure needs) without deadlocking on tokio's `RwLock`.
    ///
    /// # Errors
    ///
    /// Returns an error if `request_id` is unknown.
    pub async fn process_request(&self, request_id: &str) -> Result<RequestProcessingResult> {
        let start_time = std::time::Instant::now();

        let request = {
            let requests = self.subject_requests.read().await;
            requests
                .get(request_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Request not found: {}", request_id))?
        };

        let result = match request.request_type {
            RequestType::Access => self.process_access_request(&request).await?,
            RequestType::Rectification => self.process_rectification_request(&request).await?,
            RequestType::Erasure => self.process_erasure_request(&request).await?,
            RequestType::Restriction => self.process_restriction_request(&request).await?,
            RequestType::Portability => self.process_portability_request(&request).await?,
            RequestType::Objection => self.process_objection_request(&request).await?,
            RequestType::AutomatedDecision => {
                self.process_automated_decision_request(&request).await?
            },
        };

        // A rejected request is not a completed one: reflect the real outcome.
        let (status, completed) = match &result {
            RequestProcessingResult::Success { .. } => (RequestStatus::Completed, true),
            RequestProcessingResult::Partial { .. } => (RequestStatus::Processing, false),
            RequestProcessingResult::Rejected { .. } => (RequestStatus::Rejected, false),
        };

        {
            let mut requests = self.subject_requests.write().await;
            if let Some(stored) = requests.get_mut(request_id) {
                stored.status = status;
                if completed {
                    stored.completed_at = Some(SystemTime::now());
                }
            }
        }

        if completed {
            self.stats.completed_requests.fetch_add(1, Ordering::Relaxed);
        }

        let duration = start_time.elapsed().as_secs_f64();
        self.prometheus_metrics.request_processing_duration.observe(duration);

        Ok(result)
    }

    /// Get compliance status
    pub async fn get_compliance_status(&self) -> ComplianceStatus {
        ComplianceStatus {
            total_data_subjects: self.stats.total_data_subjects.load(Ordering::Relaxed),
            active_consents: self.count_active_consents().await,
            pending_requests: self.count_pending_requests().await,
            overdue_requests: self.count_overdue_requests().await,
            compliance_score: self.calculate_compliance_score().await,
            last_assessment: SystemTime::now(),
        }
    }

    /// Get compliance statistics
    pub async fn get_stats(&self) -> GdprComplianceStats {
        GdprComplianceStats {
            total_data_subjects: AtomicU64::new(
                self.stats.total_data_subjects.load(Ordering::Relaxed),
            ),
            total_consent_records: AtomicU64::new(
                self.stats.total_consent_records.load(Ordering::Relaxed),
            ),
            total_subject_requests: AtomicU64::new(
                self.stats.total_subject_requests.load(Ordering::Relaxed),
            ),
            completed_requests: AtomicU64::new(
                self.stats.completed_requests.load(Ordering::Relaxed),
            ),
            breaches_detected: AtomicU64::new(self.stats.breaches_detected.load(Ordering::Relaxed)),
            compliance_violations: AtomicU64::new(
                self.stats.compliance_violations.load(Ordering::Relaxed),
            ),
        }
    }

    // ── Data subject request handlers ────────────────────────────────────────
    //
    // Every handler below reads and mutates the service's real stores
    // (`data_subjects`, `consent_records`, `processing_activities`,
    // `restricted_subjects`). None of them reports work it did not do: when
    // there is nothing to act on, the request is `Rejected` with the reason,
    // and when the operation needs input or a human, it is `Partial`.

    /// Article 15: assemble everything held about the subject.
    async fn process_access_request(
        &self,
        request: &DataSubjectRequest,
    ) -> Result<RequestProcessingResult> {
        let Some(subject) = self.get_data_subject(&request.subject_id).await else {
            return Ok(Self::unknown_subject(&request.subject_id));
        };

        let consents = self.consents_for(&request.subject_id).await;
        let activities = self.list_processing_activities(&request.subject_id).await;
        let restricted = self.restricted_subjects.read().await.get(&request.subject_id).cloned();

        let payload = serde_json::json!({
            "article": "15",
            "subject": subject,
            "consent_records": consents,
            "processing_activities": activities,
            "processing_restricted": restricted.is_some(),
            "restricted_activities": restricted.unwrap_or_default(),
            "generated_at": Self::rfc3339(SystemTime::now()),
        });

        Ok(RequestProcessingResult::Success {
            data: serde_json::to_string(&payload)?,
        })
    }

    /// Article 16: apply the corrections carried in the request.
    ///
    /// The corrections are read from `details.additional_info`, which must be a
    /// JSON object whose keys are `email`, `phone`, `address`, `language` or
    /// `marketing_consent`. Without a payload there is nothing to rectify, so
    /// the request is rejected rather than reported as done.
    async fn process_rectification_request(
        &self,
        request: &DataSubjectRequest,
    ) -> Result<RequestProcessingResult> {
        let Some(payload) = request.details.additional_info.as_deref() else {
            return Ok(RequestProcessingResult::Rejected {
                reason: "rectification request carries no corrections: set \
                         details.additional_info to a JSON object of field updates"
                    .to_string(),
            });
        };

        let corrections: HashMap<String, serde_json::Value> = match serde_json::from_str(payload) {
            Ok(map) => map,
            Err(e) => {
                return Ok(RequestProcessingResult::Rejected {
                    reason: format!("rectification payload is not a JSON object: {e}"),
                })
            },
        };

        let mut subjects = self.data_subjects.write().await;
        let Some(subject) = subjects.get_mut(&request.subject_id) else {
            drop(subjects);
            return Ok(Self::unknown_subject(&request.subject_id));
        };

        let mut applied: Vec<String> = Vec::new();
        let mut ignored: Vec<String> = Vec::new();
        for (field, value) in &corrections {
            match field.as_str() {
                "email" => {
                    subject.email = value.as_str().map(str::to_string);
                    applied.push(field.clone());
                },
                "phone" => {
                    subject.phone = value.as_str().map(str::to_string);
                    applied.push(field.clone());
                },
                "address" => {
                    subject.address = value.as_str().map(str::to_string);
                    applied.push(field.clone());
                },
                "language" => match value.as_str() {
                    Some(language) => {
                        subject.preferences.language = language.to_string();
                        applied.push(field.clone());
                    },
                    None => ignored.push(field.clone()),
                },
                "marketing_consent" => match value.as_bool() {
                    Some(flag) => {
                        subject.preferences.marketing_consent = flag;
                        applied.push(field.clone());
                    },
                    None => ignored.push(field.clone()),
                },
                _ => ignored.push(field.clone()),
            }
        }
        if !applied.is_empty() {
            subject.last_activity = SystemTime::now();
        }
        drop(subjects);

        applied.sort();
        ignored.sort();

        if applied.is_empty() {
            return Ok(RequestProcessingResult::Rejected {
                reason: format!("no rectifiable fields in request; unknown fields: {ignored:?}"),
            });
        }

        let data = serde_json::to_string(&serde_json::json!({
            "article": "16",
            "subject_id": request.subject_id,
            "fields_rectified": applied,
            "fields_ignored": ignored,
        }))?;

        if ignored.is_empty() {
            Ok(RequestProcessingResult::Success { data })
        } else {
            Ok(RequestProcessingResult::Partial {
                reason: format!("rectified {applied:?}; could not rectify {ignored:?}"),
            })
        }
    }

    /// Article 17: erase the subject and every record referencing them.
    async fn process_erasure_request(
        &self,
        request: &DataSubjectRequest,
    ) -> Result<RequestProcessingResult> {
        let subject_id = &request.subject_id;

        let subject_removed = self.data_subjects.write().await.remove(subject_id).is_some();

        let consents_removed = {
            let mut consents = self.consent_records.write().await;
            let before = consents.len();
            consents.retain(|_, record| &record.subject_id != subject_id);
            before - consents.len()
        };

        let activities_removed = {
            let mut activities = self.processing_activities.write().await;
            let before = activities.len();
            activities.retain(|_, record| &record.subject_id != subject_id);
            before - activities.len()
        };

        self.restricted_subjects.write().await.remove(subject_id);
        self.human_review_queue.write().await.retain(|id| id != subject_id);

        // Other pending requests from the same subject can no longer be served
        // against data that has been erased; drop them except this one, which
        // the caller still needs a status for.
        let requests_removed = {
            let mut requests = self.subject_requests.write().await;
            let before = requests.len();
            requests.retain(|id, stored| &stored.subject_id != subject_id || id == &request.id);
            before - requests.len()
        };

        if !subject_removed && consents_removed == 0 && activities_removed == 0 {
            return Ok(Self::unknown_subject(subject_id));
        }

        let remaining = self.data_subjects.read().await.len() as u64;
        self.stats.total_data_subjects.store(remaining, Ordering::Relaxed);
        self.update_prometheus_metrics().await?;

        Ok(RequestProcessingResult::Success {
            data: serde_json::to_string(&serde_json::json!({
                "article": "17",
                "subject_id": subject_id,
                "subject_record_deleted": subject_removed,
                "consent_records_deleted": consents_removed,
                "processing_activities_deleted": activities_removed,
                "other_requests_deleted": requests_removed,
                "erased_at": Self::rfc3339(SystemTime::now()),
            }))?,
        })
    }

    /// Article 18: flag the subject's processing activities as restricted.
    async fn process_restriction_request(
        &self,
        request: &DataSubjectRequest,
    ) -> Result<RequestProcessingResult> {
        if self.get_data_subject(&request.subject_id).await.is_none() {
            return Ok(Self::unknown_subject(&request.subject_id));
        }

        let targets: HashSet<String> =
            request.details.processing_activities.iter().cloned().collect();

        let mut restricted_ids: Vec<String> = Vec::new();
        {
            let mut activities = self.processing_activities.write().await;
            for record in activities.values_mut() {
                if record.subject_id != request.subject_id {
                    continue;
                }
                if targets.is_empty() || targets.contains(&record.activity_id) {
                    record.restricted = true;
                    restricted_ids.push(record.activity_id.clone());
                }
            }
        }
        restricted_ids.sort();

        self.restricted_subjects
            .write()
            .await
            .insert(request.subject_id.clone(), targets.clone());

        if restricted_ids.is_empty() && !targets.is_empty() {
            return Ok(RequestProcessingResult::Partial {
                reason: format!(
                    "restriction recorded for subject {} but none of the named activities \
                     {targets:?} are on record",
                    request.subject_id
                ),
            });
        }

        Ok(RequestProcessingResult::Success {
            data: serde_json::to_string(&serde_json::json!({
                "article": "18",
                "subject_id": request.subject_id,
                "restricted_activities": restricted_ids,
                "scope": if targets.is_empty() { "all" } else { "named" },
            }))?,
        })
    }

    /// Article 20: export the subject's data in a machine-readable format.
    async fn process_portability_request(
        &self,
        request: &DataSubjectRequest,
    ) -> Result<RequestProcessingResult> {
        let Some(subject) = self.get_data_subject(&request.subject_id).await else {
            return Ok(Self::unknown_subject(&request.subject_id));
        };

        // Article 20 covers data the subject provided, processed on the basis
        // of consent or contract — not everything the controller holds.
        let portable_bases = ["consent", "contract"];
        let activities: Vec<ProcessingActivityRecord> = self
            .list_processing_activities(&request.subject_id)
            .await
            .into_iter()
            .filter(|record| {
                portable_bases.contains(&record.legal_basis.to_ascii_lowercase().as_str())
            })
            .collect();
        let consents = self.consents_for(&request.subject_id).await;

        let export = serde_json::json!({
            "article": "20",
            "format": "application/json",
            "subject": subject,
            "consent_records": consents,
            "portable_activities": activities,
            "exported_at": Self::rfc3339(SystemTime::now()),
        });

        Ok(RequestProcessingResult::Success {
            data: serde_json::to_string(&export)?,
        })
    }

    /// Article 21: withdraw the consents the objection targets.
    async fn process_objection_request(
        &self,
        request: &DataSubjectRequest,
    ) -> Result<RequestProcessingResult> {
        if self.get_data_subject(&request.subject_id).await.is_none() {
            return Ok(Self::unknown_subject(&request.subject_id));
        }

        let purposes: HashSet<String> = request
            .details
            .processing_activities
            .iter()
            .chain(request.details.data_categories.iter())
            .cloned()
            .collect();

        let mut withdrawn: Vec<String> = Vec::new();
        {
            let mut consents = self.consent_records.write().await;
            for record in consents.values_mut() {
                if record.subject_id != request.subject_id {
                    continue;
                }
                if !matches!(record.status, ConsentStatus::Given) {
                    continue;
                }
                if purposes.is_empty() || purposes.contains(&record.purpose) {
                    record.status = ConsentStatus::Withdrawn;
                    record.withdrawn_at = Some(SystemTime::now());
                    withdrawn.push(record.purpose.clone());
                }
            }
        }
        withdrawn.sort();

        let mut stopped: Vec<String> = Vec::new();
        {
            let mut activities = self.processing_activities.write().await;
            for record in activities.values_mut() {
                if record.subject_id != request.subject_id {
                    continue;
                }
                if purposes.is_empty() || purposes.contains(&record.purpose) {
                    record.restricted = true;
                    stopped.push(record.activity_id.clone());
                }
            }
        }
        stopped.sort();

        self.update_prometheus_metrics().await?;

        if withdrawn.is_empty() && stopped.is_empty() {
            return Ok(RequestProcessingResult::Rejected {
                reason: format!(
                    "no active consent or processing activity of subject {} matches {purposes:?}",
                    request.subject_id
                ),
            });
        }

        Ok(RequestProcessingResult::Success {
            data: serde_json::to_string(&serde_json::json!({
                "article": "21",
                "subject_id": request.subject_id,
                "consents_withdrawn": withdrawn,
                "activities_stopped": stopped,
            }))?,
        })
    }

    /// Article 22: queue the decision for human review.
    ///
    /// This service does not itself take automated decisions, so it cannot
    /// "review" one. What it can do — and does — is record the request in an
    /// auditable queue for a human to act on, which is a `Partial` outcome, not
    /// a completed one.
    async fn process_automated_decision_request(
        &self,
        request: &DataSubjectRequest,
    ) -> Result<RequestProcessingResult> {
        if self.get_data_subject(&request.subject_id).await.is_none() {
            return Ok(Self::unknown_subject(&request.subject_id));
        }

        let position = {
            let mut queue = self.human_review_queue.write().await;
            if !queue.contains(&request.subject_id) {
                queue.push(request.subject_id.clone());
            }
            queue
                .iter()
                .position(|id| id == &request.subject_id)
                .map(|i| i + 1)
                .unwrap_or(1)
        };

        Ok(RequestProcessingResult::Partial {
            reason: format!(
                "Article 22 request for subject {} queued for human review at position {position}; \
                 no automated decision was taken by this service",
                request.subject_id
            ),
        })
    }

    fn unknown_subject(subject_id: &str) -> RequestProcessingResult {
        RequestProcessingResult::Rejected {
            reason: format!("no data is held for subject {subject_id}"),
        }
    }

    async fn consents_for(&self, subject_id: &str) -> Vec<ConsentRecord> {
        self.consent_records
            .read()
            .await
            .values()
            .filter(|record| record.subject_id == subject_id)
            .cloned()
            .collect()
    }

    fn rfc3339(time: SystemTime) -> String {
        chrono::DateTime::<chrono::Utc>::from(time).to_rfc3339()
    }

    async fn count_active_consents(&self) -> u64 {
        let records = self.consent_records.read().await;
        records
            .values()
            .filter(|record| matches!(record.status, ConsentStatus::Given))
            .count() as u64
    }

    async fn count_pending_requests(&self) -> u64 {
        let requests = self.subject_requests.read().await;
        requests
            .values()
            .filter(|request| {
                matches!(
                    request.status,
                    RequestStatus::Submitted | RequestStatus::UnderReview
                )
            })
            .count() as u64
    }

    /// Count requests still open past the statutory response deadline.
    ///
    /// The deadline comes from
    /// `config.data_subject_rights.request_handling.timeframes.default_response`
    /// (30 days by default, matching Article 12(3)).
    async fn count_overdue_requests(&self) -> u64 {
        let deadline = self.config.data_subject_rights.request_handling.timeframes.default_response;
        let now = SystemTime::now();
        let requests = self.subject_requests.read().await;
        requests
            .values()
            .filter(|request| {
                matches!(
                    request.status,
                    RequestStatus::Submitted
                        | RequestStatus::UnderReview
                        | RequestStatus::VerificationRequired
                        | RequestStatus::Verified
                        | RequestStatus::Processing
                )
            })
            .filter(|request| {
                now.duration_since(request.submitted_at)
                    .map(|age| age > deadline)
                    .unwrap_or(false)
            })
            .count() as u64
    }

    /// Compute the compliance score from the service's real state.
    ///
    /// The score is the mean of three measured ratios, each in `[0, 1]`:
    ///
    /// * **Request timeliness** — the share of data subject requests that are
    ///   not past their statutory deadline.
    /// * **Consent validity** — the share of consent records that are still
    ///   `Given` rather than expired (withdrawn consents are the subject's
    ///   choice and are not counted as a compliance failure).
    /// * **Processing traceability** — the share of registered data subjects
    ///   for whom at least one processing activity is on record, as required by
    ///   Article 30.
    ///
    /// A dimension with no data contributes 1.0, because there is nothing to be
    /// non-compliant about. The score is never a hardcoded constant.
    async fn calculate_compliance_score(&self) -> f64 {
        let total_requests = self.subject_requests.read().await.len() as f64;
        let overdue = self.count_overdue_requests().await as f64;
        let timeliness = if total_requests > 0.0 { 1.0 - (overdue / total_requests) } else { 1.0 };

        let (total_consents, expired_consents) = {
            let consents = self.consent_records.read().await;
            let total = consents.len() as f64;
            let expired = consents
                .values()
                .filter(|record| matches!(record.status, ConsentStatus::Expired))
                .count() as f64;
            (total, expired)
        };
        let consent_validity = if total_consents > 0.0 {
            1.0 - (expired_consents / total_consents)
        } else {
            1.0
        };

        let traceability = {
            let subjects = self.data_subjects.read().await;
            let total = subjects.len() as f64;
            if total > 0.0 {
                let activities = self.processing_activities.read().await;
                let traced: HashSet<&String> =
                    activities.values().map(|record| &record.subject_id).collect();
                let covered = subjects.keys().filter(|id| traced.contains(id)).count() as f64;
                covered / total
            } else {
                1.0
            }
        };

        ((timeliness + consent_validity + traceability) / 3.0).clamp(0.0, 1.0)
    }

    async fn update_prometheus_metrics(&self) -> Result<()> {
        let active_consents = self.count_active_consents().await;
        self.prometheus_metrics.active_consents.set(active_consents as f64);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdpr_compliance_modules::types::DataSubjectPreferences;

    fn service() -> GdprComplianceService {
        GdprComplianceService::new(GdprComplianceConfig::default())
            .expect("service construction must succeed")
    }

    fn subject(id: &str) -> DataSubject {
        DataSubject {
            id: id.to_string(),
            email: Some(format!("{id}@example.test")),
            phone: None,
            address: None,
            registered_at: SystemTime::now(),
            last_activity: SystemTime::now(),
            preferences: DataSubjectPreferences {
                language: "en".to_string(),
                communication_channels: vec!["email".to_string()],
                data_processing_opt_ins: vec!["analytics".to_string()],
                marketing_consent: true,
            },
        }
    }

    fn activity(
        subject_id: &str,
        activity_id: &str,
        purpose: &str,
        basis: &str,
    ) -> ProcessingActivityRecord {
        ProcessingActivityRecord {
            activity_id: activity_id.to_string(),
            subject_id: subject_id.to_string(),
            purpose: purpose.to_string(),
            legal_basis: basis.to_string(),
            data_categories: vec!["PersonalData".to_string()],
            data: HashMap::from([("inference_prompt".to_string(), "hello".to_string())]),
            recorded_at: SystemTime::now(),
            restricted: false,
        }
    }

    fn details() -> RequestDetails {
        RequestDetails {
            description: "test request".to_string(),
            data_categories: vec![],
            processing_activities: vec![],
            additional_info: None,
        }
    }

    async fn submit(
        svc: &GdprComplianceService,
        subject_id: &str,
        request_type: RequestType,
        details: RequestDetails,
    ) -> String {
        svc.submit_data_subject_request(subject_id, request_type, details)
            .await
            .expect("request submission must succeed")
    }

    /// Regression test: `process_erasure_request` used to return
    /// `Success { data: "Data erased" }` without touching anything.
    #[tokio::test]
    async fn test_erasure_actually_deletes_the_stored_data() {
        let svc = service();
        svc.register_data_subject(subject("alice")).await.expect("register");
        svc.record_consent("alice", "analytics", ConsentMechanism::WebForm)
            .await
            .expect("consent");
        svc.record_processing_activity(activity("alice", "a1", "analytics", "consent"))
            .await
            .expect("activity");

        assert!(svc.get_data_subject("alice").await.is_some());
        assert_eq!(svc.list_processing_activities("alice").await.len(), 1);

        let id = submit(&svc, "alice", RequestType::Erasure, details()).await;
        let result = svc.process_request(&id).await.expect("process");

        let RequestProcessingResult::Success { data } = result else {
            panic!("erasure of existing data must succeed, got {result:?}");
        };
        let payload: serde_json::Value = serde_json::from_str(&data).expect("json");
        assert_eq!(payload["subject_record_deleted"], serde_json::json!(true));
        assert_eq!(payload["consent_records_deleted"], serde_json::json!(1));
        assert_eq!(
            payload["processing_activities_deleted"],
            serde_json::json!(1)
        );

        // The real proof: the data is gone.
        assert!(svc.get_data_subject("alice").await.is_none());
        assert!(svc.list_processing_activities("alice").await.is_empty());
        assert_eq!(svc.get_compliance_status().await.active_consents, 0);
    }

    #[tokio::test]
    async fn test_erasure_of_unknown_subject_is_rejected() {
        let svc = service();
        let id = submit(&svc, "ghost", RequestType::Erasure, details()).await;
        let result = svc.process_request(&id).await.expect("process");
        assert!(
            matches!(result, RequestProcessingResult::Rejected { .. }),
            "erasing nothing must not report success, got {result:?}"
        );
        let stored = svc.subject_requests.read().await;
        let request = stored.get(&id).expect("request still stored");
        assert!(matches!(request.status, RequestStatus::Rejected));
        assert!(request.completed_at.is_none());
    }

    /// Regression test: access used to return the literal "Access data provided".
    #[tokio::test]
    async fn test_access_returns_the_real_records() {
        let svc = service();
        svc.register_data_subject(subject("bob")).await.expect("register");
        svc.record_consent("bob", "training", ConsentMechanism::WebForm)
            .await
            .expect("consent");
        svc.record_processing_activity(activity("bob", "a1", "training", "consent"))
            .await
            .expect("activity");

        let id = submit(&svc, "bob", RequestType::Access, details()).await;
        let RequestProcessingResult::Success { data } =
            svc.process_request(&id).await.expect("process")
        else {
            panic!("access must succeed for a registered subject");
        };

        let payload: serde_json::Value = serde_json::from_str(&data).expect("json");
        assert_eq!(payload["subject"]["id"], serde_json::json!("bob"));
        assert_eq!(
            payload["subject"]["email"],
            serde_json::json!("bob@example.test")
        );
        assert_eq!(payload["consent_records"].as_array().map(Vec::len), Some(1));
        assert_eq!(
            payload["processing_activities"].as_array().map(Vec::len),
            Some(1)
        );
        assert_eq!(
            payload["processing_activities"][0]["data"]["inference_prompt"],
            serde_json::json!("hello")
        );
    }

    #[tokio::test]
    async fn test_access_for_unknown_subject_is_rejected() {
        let svc = service();
        let id = submit(&svc, "ghost", RequestType::Access, details()).await;
        let result = svc.process_request(&id).await.expect("process");
        assert!(matches!(result, RequestProcessingResult::Rejected { .. }));
    }

    #[tokio::test]
    async fn test_portability_exports_only_consent_and_contract_activities() {
        let svc = service();
        svc.register_data_subject(subject("carol")).await.expect("register");
        svc.record_processing_activity(activity("carol", "a1", "training", "consent"))
            .await
            .expect("activity");
        svc.record_processing_activity(activity("carol", "a2", "fraud", "LegitimateInterests"))
            .await
            .expect("activity");

        let id = submit(&svc, "carol", RequestType::Portability, details()).await;
        let RequestProcessingResult::Success { data } =
            svc.process_request(&id).await.expect("process")
        else {
            panic!("portability must succeed for a registered subject");
        };
        let payload: serde_json::Value = serde_json::from_str(&data).expect("json");
        let activities = payload["portable_activities"].as_array().expect("array");
        assert_eq!(
            activities.len(),
            1,
            "only consent/contract activities are portable"
        );
        assert_eq!(activities[0]["activity_id"], serde_json::json!("a1"));
    }

    #[tokio::test]
    async fn test_rectification_applies_the_corrections() {
        let svc = service();
        svc.register_data_subject(subject("dave")).await.expect("register");

        let mut d = details();
        d.additional_info = Some(
            serde_json::json!({"email": "new@example.test", "marketing_consent": false})
                .to_string(),
        );
        let id = submit(&svc, "dave", RequestType::Rectification, d).await;
        let result = svc.process_request(&id).await.expect("process");
        assert!(
            matches!(result, RequestProcessingResult::Success { .. }),
            "{result:?}"
        );

        let updated = svc.get_data_subject("dave").await.expect("subject");
        assert_eq!(updated.email.as_deref(), Some("new@example.test"));
        assert!(!updated.preferences.marketing_consent);
    }

    #[tokio::test]
    async fn test_rectification_without_payload_is_rejected() {
        let svc = service();
        svc.register_data_subject(subject("dave")).await.expect("register");
        let id = submit(&svc, "dave", RequestType::Rectification, details()).await;
        let result = svc.process_request(&id).await.expect("process");
        assert!(
            matches!(result, RequestProcessingResult::Rejected { .. }),
            "a rectification with nothing to rectify must not report success, got {result:?}"
        );
        // The stored subject is untouched.
        let unchanged = svc.get_data_subject("dave").await.expect("subject");
        assert_eq!(unchanged.email.as_deref(), Some("dave@example.test"));
    }

    #[tokio::test]
    async fn test_rectification_with_unknown_field_is_partial() {
        let svc = service();
        svc.register_data_subject(subject("dave")).await.expect("register");
        let mut d = details();
        d.additional_info =
            Some(serde_json::json!({"email": "x@example.test", "salary": 100}).to_string());
        let id = submit(&svc, "dave", RequestType::Rectification, d).await;
        let result = svc.process_request(&id).await.expect("process");
        assert!(
            matches!(result, RequestProcessingResult::Partial { .. }),
            "{result:?}"
        );
    }

    #[tokio::test]
    async fn test_restriction_flags_the_stored_activities() {
        let svc = service();
        svc.register_data_subject(subject("erin")).await.expect("register");
        svc.record_processing_activity(activity("erin", "a1", "training", "consent"))
            .await
            .expect("activity");

        let id = submit(&svc, "erin", RequestType::Restriction, details()).await;
        let result = svc.process_request(&id).await.expect("process");
        assert!(
            matches!(result, RequestProcessingResult::Success { .. }),
            "{result:?}"
        );

        let activities = svc.list_processing_activities("erin").await;
        assert!(
            activities.iter().all(|a| a.restricted),
            "activities must be marked restricted"
        );
    }

    #[tokio::test]
    async fn test_objection_withdraws_the_matching_consent() {
        let svc = service();
        svc.register_data_subject(subject("frank")).await.expect("register");
        svc.record_consent("frank", "marketing", ConsentMechanism::WebForm)
            .await
            .expect("consent");
        svc.record_consent("frank", "training", ConsentMechanism::WebForm)
            .await
            .expect("consent");
        assert_eq!(svc.get_compliance_status().await.active_consents, 2);

        let mut d = details();
        d.processing_activities = vec!["marketing".to_string()];
        let id = submit(&svc, "frank", RequestType::Objection, d).await;
        let RequestProcessingResult::Success { data } =
            svc.process_request(&id).await.expect("process")
        else {
            panic!("objection against a live consent must succeed");
        };
        let payload: serde_json::Value = serde_json::from_str(&data).expect("json");
        assert_eq!(
            payload["consents_withdrawn"],
            serde_json::json!(["marketing"])
        );
        assert_eq!(
            svc.get_compliance_status().await.active_consents,
            1,
            "only the objected-to consent may be withdrawn"
        );
    }

    #[tokio::test]
    async fn test_objection_with_no_match_is_rejected() {
        let svc = service();
        svc.register_data_subject(subject("frank")).await.expect("register");
        let mut d = details();
        d.processing_activities = vec!["nonexistent-purpose".to_string()];
        let id = submit(&svc, "frank", RequestType::Objection, d).await;
        let result = svc.process_request(&id).await.expect("process");
        assert!(
            matches!(result, RequestProcessingResult::Rejected { .. }),
            "{result:?}"
        );
    }

    /// Regression test: this used to claim "Automated decision reviewed".
    #[tokio::test]
    async fn test_automated_decision_is_queued_for_human_review() {
        let svc = service();
        svc.register_data_subject(subject("grace")).await.expect("register");
        let id = submit(&svc, "grace", RequestType::AutomatedDecision, details()).await;
        let result = svc.process_request(&id).await.expect("process");
        assert!(
            matches!(result, RequestProcessingResult::Partial { .. }),
            "no automated decision is taken here, so this cannot be a completed review: {result:?}"
        );
        assert_eq!(svc.human_review_queue().await, vec!["grace".to_string()]);
    }

    #[tokio::test]
    async fn test_process_request_unknown_id_errors() {
        let svc = service();
        assert!(svc.process_request("no-such-request").await.is_err());
    }

    /// Regression test: `process_request` used to hold a write lock on
    /// `subject_requests` while running the handler, so any handler that
    /// touched that map (erasure does) would deadlock.
    #[tokio::test]
    async fn test_erasure_does_not_deadlock_on_the_request_map() {
        let svc = service();
        svc.register_data_subject(subject("heidi")).await.expect("register");
        // A second, unrelated request from the same subject that erasure must
        // clean up while `process_request` is running.
        let _other = submit(&svc, "heidi", RequestType::Access, details()).await;
        let id = submit(&svc, "heidi", RequestType::Erasure, details()).await;

        let outcome =
            tokio::time::timeout(std::time::Duration::from_secs(5), svc.process_request(&id))
                .await
                .expect("process_request must not deadlock")
                .expect("process");
        assert!(matches!(outcome, RequestProcessingResult::Success { .. }));
        assert_eq!(svc.subject_requests.read().await.len(), 1);
    }

    /// Regression test: `count_overdue_requests()` returned a hardcoded 0.
    #[tokio::test]
    async fn test_overdue_requests_are_counted_from_real_timestamps() {
        let svc = service();
        svc.register_data_subject(subject("ivan")).await.expect("register");
        let id = submit(&svc, "ivan", RequestType::Access, details()).await;
        assert_eq!(svc.get_compliance_status().await.overdue_requests, 0);

        // Back-date the submission past the 30-day statutory deadline.
        {
            let mut requests = svc.subject_requests.write().await;
            let request = requests.get_mut(&id).expect("request");
            request.submitted_at = SystemTime::now() - std::time::Duration::from_secs(40 * 86_400);
        }
        assert_eq!(
            svc.get_compliance_status().await.overdue_requests,
            1,
            "a 40-day-old open request is past the 30-day deadline"
        );
    }

    /// Regression test: `calculate_compliance_score()` returned the constant 0.85.
    #[tokio::test]
    async fn test_compliance_score_reflects_real_state() {
        let svc = service();
        // Empty service: nothing to be non-compliant about.
        assert!((svc.get_compliance_status().await.compliance_score - 1.0).abs() < 1e-9);

        svc.register_data_subject(subject("judy")).await.expect("register");
        // A registered subject with no processing record on file lowers the
        // Article 30 traceability dimension.
        let with_gap = svc.get_compliance_status().await.compliance_score;
        assert!(
            with_gap < 1.0,
            "an untraced subject must lower the score, got {with_gap}"
        );
        assert_ne!(
            with_gap, 0.85,
            "score must not be the old hardcoded constant"
        );

        svc.record_processing_activity(activity("judy", "a1", "training", "consent"))
            .await
            .expect("activity");
        let restored = svc.get_compliance_status().await.compliance_score;
        assert!(
            restored > with_gap,
            "recording the processing activity must raise the score: {with_gap} -> {restored}"
        );
    }

    #[tokio::test]
    async fn test_record_processing_activity_requires_a_registered_subject() {
        let svc = service();
        let err = svc
            .record_processing_activity(activity("ghost", "a1", "training", "consent"))
            .await
            .expect_err("recording against an unknown subject must fail");
        assert!(err.to_string().contains("unregistered data subject"));
    }

    /// Regression test: `start()` was four `Ok(())` no-ops.
    #[tokio::test]
    async fn test_start_runs_real_sweeps_and_reports_what_it_did() {
        let mut config = GdprComplianceConfig::default();
        // Retain nothing, so the sweep has real work to do.
        config.data_retention.default_retention_period = std::time::Duration::from_secs(0);
        let svc = GdprComplianceService::new(config).expect("service");

        svc.register_data_subject(subject("ken")).await.expect("register");
        svc.record_consent("ken", "analytics", ConsentMechanism::WebForm)
            .await
            .expect("consent");

        // Back-date so the record is unambiguously older than the zero-length
        // retention period.
        {
            let mut consents = svc.consent_records.write().await;
            for record in consents.values_mut() {
                record.given_at = SystemTime::now() - std::time::Duration::from_secs(3_600);
            }
        }

        let report = svc.start().await.expect("start");
        let retention = report.retention.expect("retention sweep must have run");
        assert_eq!(
            retention.consents_deleted, 1,
            "the expired consent must be deleted"
        );
        assert!(svc.consent_records.read().await.is_empty());
    }

    #[tokio::test]
    async fn test_start_is_a_noop_when_disabled() {
        let mut config = GdprComplianceConfig::default();
        config.enabled = false;
        let svc = GdprComplianceService::new(config).expect("service");
        assert_eq!(svc.start().await.expect("start"), StartupReport::default());
    }

    #[tokio::test]
    async fn test_consent_renewal_sweep_expires_stale_consents() {
        let svc = service();
        svc.register_data_subject(subject("laura")).await.expect("register");
        svc.record_consent("laura", "analytics", ConsentMechanism::WebForm)
            .await
            .expect("consent");
        {
            let mut consents = svc.consent_records.write().await;
            for record in consents.values_mut() {
                record.expires_at = Some(SystemTime::now() - std::time::Duration::from_secs(60));
            }
        }
        let report = svc.run_consent_renewal_sweep().await.expect("sweep");
        assert_eq!(report.expired, 1);
        assert_eq!(svc.get_compliance_status().await.active_consents, 0);
    }

    #[tokio::test]
    async fn test_recorded_consent_has_a_real_expiry() {
        let svc = service();
        svc.register_data_subject(subject("mike")).await.expect("register");
        let id = svc
            .record_consent("mike", "analytics", ConsentMechanism::WebForm)
            .await
            .expect("consent");
        let consents = svc.consent_records.read().await;
        let record = consents.get(&id).expect("record");
        assert!(
            record.expires_at.is_some(),
            "consent must carry the configured expiry, not None"
        );
    }
}
