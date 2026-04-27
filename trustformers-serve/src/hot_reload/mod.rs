//! Zero-Downtime Hot-Reload for TrustformeRS Inference Server
//!
//! Implements atomic model swapping with no downtime. In-flight requests complete
//! with the old model; new requests use the new model atomically after the swap.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use thiserror::Error;

// ── Error type ─────────────────────────────────────────────────────────────────

/// Errors that can occur during hot-reload operations.
#[derive(Debug, Error)]
pub enum HotReloadError {
    #[error("Drain timeout exceeded after {timeout_ms} ms with {remaining} in-flight requests")]
    DrainTimeout { timeout_ms: u64, remaining: usize },

    #[error("Version {version_id} not found in history")]
    VersionNotFound { version_id: u64 },

    #[error("Version {version_id} is already active")]
    AlreadyActive { version_id: u64 },

    #[error("Health check failed after {retries} retries")]
    HealthCheckFailed { retries: u32 },

    #[error("A drain is already in progress")]
    DrainInProgress,

    #[error("No active version to roll back from")]
    NoActiveVersion,

    #[error("RwLock poisoned: {msg}")]
    LockPoisoned { msg: String },
}

// ── Configuration ──────────────────────────────────────────────────────────────

/// Configuration for the hot-reload manager.
#[derive(Debug, Clone)]
pub struct HotReloadConfig {
    /// Maximum time (ms) to wait for in-flight requests to drain before aborting.
    pub drain_timeout_ms: u64,
    /// Number of health-check retries before declaring a new version invalid.
    pub health_check_retries: u32,
    /// If true, automatically roll back to the previous version when health
    /// checks fail.
    pub rollback_on_failure: bool,
    /// Maximum number of old versions retained in the history map.
    pub version_history_limit: usize,
}

impl Default for HotReloadConfig {
    fn default() -> Self {
        Self {
            drain_timeout_ms: 5000,
            health_check_retries: 3,
            rollback_on_failure: true,
            version_history_limit: 3,
        }
    }
}

// ── ModelVersion ───────────────────────────────────────────────────────────────

/// Metadata for a single loaded model version.
#[derive(Debug, Clone)]
pub struct ModelVersion {
    /// Monotonically-increasing identifier.
    pub version_id: u64,
    /// Wall-clock instant at which this version was loaded.
    pub loaded_at: Instant,
    /// Total number of requests that have been dispatched to this version.
    pub request_count: u64,
    /// Whether this is the currently-active version.
    pub is_active: bool,
}

// ── RequestGuard ───────────────────────────────────────────────────────────────

/// RAII guard that tracks a single in-flight request.
///
/// Constructed by [`HotReloadManager::begin_request`]; automatically decrements
/// the in-flight counter when dropped.
pub struct RequestGuard {
    in_flight: Arc<AtomicUsize>,
    version_id: u64,
    versions: Arc<RwLock<HashMap<u64, ModelVersion>>>,
}

impl RequestGuard {
    /// Returns the model version this request was dispatched against.
    pub fn version_id(&self) -> u64 {
        self.version_id
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.in_flight.fetch_sub(1, Ordering::Release);
        // Increment request_count for the version this guard was bound to.
        if let Ok(mut versions) = self.versions.write() {
            if let Some(v) = versions.get_mut(&self.version_id) {
                v.request_count += 1;
            }
        }
    }
}

// ── SwapResult ─────────────────────────────────────────────────────────────────

/// Outcome of a successful atomic model swap.
#[derive(Debug, Clone)]
pub struct SwapResult {
    /// Version that was active before the swap.
    pub old_version: u64,
    /// Version that is active after the swap.
    pub new_version: u64,
    /// Wall-clock milliseconds spent draining in-flight requests.
    pub drain_duration_ms: u64,
    /// Number of requests that were allowed to complete before the swap.
    pub requests_drained: usize,
}

// ── HotReloadManager ──────────────────────────────────────────────────────────

/// Manages zero-downtime atomic model swapping.
///
/// # Thread Safety
///
/// All fields are wrapped in `Arc`-based primitives so that the manager may be
/// freely cloned and shared across threads.
#[derive(Clone)]
pub struct HotReloadManager {
    config: HotReloadConfig,
    current_version: Arc<AtomicU64>,
    versions: Arc<RwLock<HashMap<u64, ModelVersion>>>,
    in_flight: Arc<AtomicUsize>,
    pending_drain: Arc<AtomicBool>,
}

impl HotReloadManager {
    /// Construct a new manager with the given configuration.
    ///
    /// An initial synthetic version `0` is registered as active.
    pub fn new(config: HotReloadConfig) -> Self {
        let initial_version = ModelVersion {
            version_id: 0,
            loaded_at: Instant::now(),
            request_count: 0,
            is_active: true,
        };
        let mut map = HashMap::new();
        map.insert(0u64, initial_version);

        Self {
            config,
            current_version: Arc::new(AtomicU64::new(0)),
            versions: Arc::new(RwLock::new(map)),
            in_flight: Arc::new(AtomicUsize::new(0)),
            pending_drain: Arc::new(AtomicBool::new(false)),
        }
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    /// Increment the in-flight counter and return an RAII guard.
    ///
    /// The guard, when dropped, decrements the counter and increments the
    /// request count for the version that was active at the time of the call.
    pub fn begin_request(&self) -> RequestGuard {
        let version_id = self.current_version.load(Ordering::Acquire);
        self.in_flight.fetch_add(1, Ordering::AcqRel);
        RequestGuard {
            in_flight: self.in_flight.clone(),
            version_id,
            versions: self.versions.clone(),
        }
    }

    /// Register a new version in the history map without making it active.
    ///
    /// Call this before [`Self::commit_swap`] to pre-warm the new version's metadata.
    pub fn prepare_new_version(&self, version_id: u64) -> Result<(), HotReloadError> {
        let mut versions = self
            .versions
            .write()
            .map_err(|e| HotReloadError::LockPoisoned { msg: e.to_string() })?;

        let current = self.current_version.load(Ordering::Acquire);
        if version_id == current {
            return Err(HotReloadError::AlreadyActive { version_id });
        }

        versions.insert(
            version_id,
            ModelVersion {
                version_id,
                loaded_at: Instant::now(),
                request_count: 0,
                is_active: false,
            },
        );

        // Evict old versions beyond history limit (keep current + new + history).
        self.evict_old_versions_locked(&mut versions, current);

        Ok(())
    }

    /// Drain in-flight requests and atomically swap to `new_version_id`.
    ///
    /// The drain loop polls every 10 ms until all in-flight requests finish or
    /// `drain_timeout_ms` elapses.  After a successful drain the current-version
    /// atomic is updated and the old version is marked inactive.
    pub fn commit_swap(&self, new_version_id: u64) -> Result<SwapResult, HotReloadError> {
        // Guard against concurrent swaps.
        if self
            .pending_drain
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(HotReloadError::DrainInProgress);
        }

        let _drain_guard = DrainGuard {
            flag: self.pending_drain.clone(),
        };

        // Ensure the new version exists.
        {
            let versions = self
                .versions
                .read()
                .map_err(|e| HotReloadError::LockPoisoned { msg: e.to_string() })?;
            if !versions.contains_key(&new_version_id) {
                return Err(HotReloadError::VersionNotFound {
                    version_id: new_version_id,
                });
            }
        }

        // Run health checks before draining traffic.
        self.run_health_checks(new_version_id)?; // validates version exists

        let old_version = self.current_version.load(Ordering::Acquire);
        let drain_start = Instant::now();
        let timeout = Duration::from_millis(self.config.drain_timeout_ms);

        // Drain: wait for all in-flight requests started on the old version to
        // finish.  New requests will still increment in_flight; we need to stop
        // dispatching them to the old version first by swapping atomically.
        //
        // Strategy: do a soft swap first so that new requests go to new_version,
        // then wait for in_flight to reach 0 (guards from old requests will
        // decrement it naturally).
        self.current_version.store(new_version_id, Ordering::Release);

        // Spin-wait for existing in-flight to drain.
        let mut requests_drained = 0usize;
        loop {
            let count = self.in_flight.load(Ordering::Acquire);
            if count == 0 {
                break;
            }
            if drain_start.elapsed() >= timeout {
                // Timeout: roll back the atomic if configured.
                self.current_version.store(old_version, Ordering::Release);
                if self.config.rollback_on_failure {
                    let _ = self.mark_version_active_locked(old_version);
                }
                return Err(HotReloadError::DrainTimeout {
                    timeout_ms: self.config.drain_timeout_ms,
                    remaining: count,
                });
            }
            requests_drained += 1;
            std::thread::sleep(Duration::from_millis(10));
        }

        let drain_duration_ms = drain_start.elapsed().as_millis() as u64;

        // Commit: update version metadata.
        {
            let mut versions = self
                .versions
                .write()
                .map_err(|e| HotReloadError::LockPoisoned { msg: e.to_string() })?;
            if let Some(old) = versions.get_mut(&old_version) {
                old.is_active = false;
            }
            if let Some(new_v) = versions.get_mut(&new_version_id) {
                new_v.is_active = true;
            }
        }

        Ok(SwapResult {
            old_version,
            new_version: new_version_id,
            drain_duration_ms,
            requests_drained,
        })
    }

    /// Roll back the active version to a previously-loaded version.
    pub fn rollback(&self, to_version_id: u64) -> Result<(), HotReloadError> {
        let current = self.current_version.load(Ordering::Acquire);
        if to_version_id == current {
            return Err(HotReloadError::AlreadyActive {
                version_id: to_version_id,
            });
        }

        {
            let versions = self
                .versions
                .read()
                .map_err(|e| HotReloadError::LockPoisoned { msg: e.to_string() })?;
            if !versions.contains_key(&to_version_id) {
                return Err(HotReloadError::VersionNotFound {
                    version_id: to_version_id,
                });
            }
        }

        self.current_version.store(to_version_id, Ordering::Release);

        let mut versions = self
            .versions
            .write()
            .map_err(|e| HotReloadError::LockPoisoned { msg: e.to_string() })?;
        if let Some(old) = versions.get_mut(&current) {
            old.is_active = false;
        }
        if let Some(new_v) = versions.get_mut(&to_version_id) {
            new_v.is_active = true;
        }

        Ok(())
    }

    /// Return the currently-active version id.
    pub fn current_version(&self) -> u64 {
        self.current_version.load(Ordering::Acquire)
    }

    /// Return a snapshot of all known model versions.
    pub fn version_history(&self) -> Vec<ModelVersion> {
        self.versions.read().map(|v| v.values().cloned().collect()).unwrap_or_default()
    }

    /// Return the number of in-flight requests at this instant.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.load(Ordering::Acquire)
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    fn run_health_checks(&self, version_id: u64) -> Result<(), HotReloadError> {
        // In production this would load-test the new version endpoint.  Here we
        // simulate a deterministic pass (tests can override by using the error
        // path directly).  All retries succeed unless the caller has specifically
        // injected a broken version — the contract is: if the version exists in
        // the map, it is considered healthy.
        for attempt in 0..self.config.health_check_retries {
            let versions = self
                .versions
                .read()
                .map_err(|e| HotReloadError::LockPoisoned { msg: e.to_string() })?;
            if versions.contains_key(&version_id) {
                return Ok(());
            }
            drop(versions);
            if attempt + 1 == self.config.health_check_retries {
                return Err(HotReloadError::HealthCheckFailed {
                    retries: self.config.health_check_retries,
                });
            }
        }
        Ok(())
    }

    fn mark_version_active_locked(&self, version_id: u64) -> Result<(), HotReloadError> {
        let mut versions = self
            .versions
            .write()
            .map_err(|e| HotReloadError::LockPoisoned { msg: e.to_string() })?;
        if let Some(v) = versions.get_mut(&version_id) {
            v.is_active = true;
        }
        Ok(())
    }

    /// Evict versions beyond `version_history_limit`.
    ///
    /// Keeps: current version, plus the N most-recently-loaded non-current versions.
    fn evict_old_versions_locked(
        &self,
        versions: &mut HashMap<u64, ModelVersion>,
        current_id: u64,
    ) {
        let limit = self.config.version_history_limit;
        // Collect non-current ids sorted by loaded_at descending.
        let mut non_current: Vec<u64> = versions
            .iter()
            .filter(|(id, _)| **id != current_id)
            .map(|(id, _)| *id)
            .collect();
        non_current.sort_by(|a, b| {
            let ta = versions[a].loaded_at;
            let tb = versions[b].loaded_at;
            tb.cmp(&ta)
        });
        // Remove entries beyond limit (keep at most `limit` non-current versions).
        for id in non_current.into_iter().skip(limit) {
            versions.remove(&id);
        }
    }
}

/// RAII helper to clear the `pending_drain` flag when the swap completes or
/// errors out.
struct DrainGuard {
    flag: Arc<AtomicBool>,
}

impl Drop for DrainGuard {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::Release);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::thread;

    fn default_manager() -> HotReloadManager {
        HotReloadManager::new(HotReloadConfig::default())
    }

    // 1. Initial state
    #[test]
    fn test_initial_version_is_zero() {
        let mgr = default_manager();
        assert_eq!(mgr.current_version(), 0);
    }

    // 2. in_flight starts at zero
    #[test]
    fn test_initial_in_flight_is_zero() {
        let mgr = default_manager();
        assert_eq!(mgr.in_flight_count(), 0);
    }

    // 3. begin_request increments in_flight
    #[test]
    fn test_begin_request_increments_in_flight() {
        let mgr = default_manager();
        let _g = mgr.begin_request();
        assert_eq!(mgr.in_flight_count(), 1);
    }

    // 4. dropping RequestGuard decrements in_flight
    #[test]
    fn test_drop_guard_decrements_in_flight() {
        let mgr = default_manager();
        {
            let _g = mgr.begin_request();
            assert_eq!(mgr.in_flight_count(), 1);
        }
        assert_eq!(mgr.in_flight_count(), 0);
    }

    // 5. multiple guards
    #[test]
    fn test_multiple_guards() {
        let mgr = default_manager();
        let g1 = mgr.begin_request();
        let g2 = mgr.begin_request();
        let g3 = mgr.begin_request();
        assert_eq!(mgr.in_flight_count(), 3);
        drop(g1);
        assert_eq!(mgr.in_flight_count(), 2);
        drop(g2);
        drop(g3);
        assert_eq!(mgr.in_flight_count(), 0);
    }

    // 6. prepare_new_version registers it as inactive
    #[test]
    fn test_prepare_new_version_registers_inactive() {
        let mgr = default_manager();
        mgr.prepare_new_version(1).expect("prepare should succeed");
        let history = mgr.version_history();
        let v1 = history.iter().find(|v| v.version_id == 1).expect("v1 missing");
        assert!(!v1.is_active);
    }

    // 7. commit_swap advances active version
    #[test]
    fn test_commit_swap_advances_version() {
        let mgr = default_manager();
        mgr.prepare_new_version(1).expect("prepare");
        let result = mgr.commit_swap(1).expect("swap");
        assert_eq!(result.old_version, 0);
        assert_eq!(result.new_version, 1);
        assert_eq!(mgr.current_version(), 1);
    }

    // 8. old version is marked inactive after swap
    #[test]
    fn test_old_version_marked_inactive_after_swap() {
        let mgr = default_manager();
        mgr.prepare_new_version(1).expect("prepare");
        mgr.commit_swap(1).expect("swap");
        let history = mgr.version_history();
        let v0 = history.iter().find(|v| v.version_id == 0).expect("v0 missing");
        assert!(!v0.is_active);
        let v1 = history.iter().find(|v| v.version_id == 1).expect("v1 missing");
        assert!(v1.is_active);
    }

    // 9. swap unknown version returns error
    #[test]
    fn test_swap_unknown_version_returns_error() {
        let mgr = default_manager();
        let err = mgr.commit_swap(99).unwrap_err();
        assert!(matches!(
            err,
            HotReloadError::VersionNotFound { version_id: 99 }
        ));
    }

    // 10. rollback restores previous version
    #[test]
    fn test_rollback_restores_previous_version() {
        let mgr = default_manager();
        mgr.prepare_new_version(1).expect("prepare");
        mgr.commit_swap(1).expect("swap");
        mgr.rollback(0).expect("rollback");
        assert_eq!(mgr.current_version(), 0);
    }

    // 11. rollback to unknown version returns error
    #[test]
    fn test_rollback_to_unknown_version() {
        let mgr = default_manager();
        let err = mgr.rollback(42).unwrap_err();
        assert!(matches!(
            err,
            HotReloadError::VersionNotFound { version_id: 42 }
        ));
    }

    // 12. rollback to already-active version returns AlreadyActive
    #[test]
    fn test_rollback_to_active_version_returns_error() {
        let mgr = default_manager();
        let err = mgr.rollback(0).unwrap_err();
        assert!(matches!(
            err,
            HotReloadError::AlreadyActive { version_id: 0 }
        ));
    }

    // 13. version history is returned with all known versions
    #[test]
    fn test_version_history_contains_all_versions() {
        let mgr = default_manager();
        mgr.prepare_new_version(1).expect("prepare 1");
        mgr.prepare_new_version(2).expect("prepare 2");
        let history = mgr.version_history();
        assert!(history.iter().any(|v| v.version_id == 0));
        assert!(history.iter().any(|v| v.version_id == 1));
        assert!(history.iter().any(|v| v.version_id == 2));
    }

    // 14. version_history_limit evicts oldest versions
    #[test]
    fn test_version_history_limit_evicts_old() {
        let config = HotReloadConfig {
            version_history_limit: 2,
            ..Default::default()
        };
        let mgr = HotReloadManager::new(config);
        // Prepare 3 non-current versions; with limit=2 the oldest should be
        // evicted.
        mgr.prepare_new_version(1).expect("p1");
        mgr.prepare_new_version(2).expect("p2");
        mgr.prepare_new_version(3).expect("p3");
        let history = mgr.version_history();
        // Should have current (0) + 2 most recent non-current (2, 3); version 1
        // should have been evicted.
        assert!(
            history.iter().any(|v| v.version_id == 0),
            "current must survive"
        );
        let non_current_count = history.iter().filter(|v| v.version_id != 0).count();
        assert!(
            non_current_count <= 2,
            "expected at most 2 non-current versions, got {non_current_count}"
        );
    }

    // 15. concurrent begin/end does not corrupt in_flight counter
    #[test]
    fn test_concurrent_begin_end_request() {
        let mgr = Arc::new(default_manager());
        let barrier = Arc::new(Barrier::new(8));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let mgr2 = mgr.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                let _g = mgr2.begin_request();
                // Simulate some work.
                std::hint::black_box(42u64);
            }));
        }
        for h in handles {
            h.join().expect("thread panicked");
        }
        assert_eq!(mgr.in_flight_count(), 0);
    }

    // 16. drain timeout returns DrainTimeout error
    #[test]
    fn test_drain_timeout_returns_error() {
        let config = HotReloadConfig {
            drain_timeout_ms: 50, // very short timeout
            rollback_on_failure: false,
            ..Default::default()
        };
        let mgr = Arc::new(HotReloadManager::new(config));
        // Hold a guard so in_flight never drains.
        let _guard = mgr.begin_request();
        mgr.prepare_new_version(1).expect("prepare");
        let err = mgr.commit_swap(1).unwrap_err();
        assert!(
            matches!(err, HotReloadError::DrainTimeout { .. }),
            "expected DrainTimeout, got: {err}"
        );
    }

    // 17. after drain timeout current version is restored (rollback_on_failure=true)
    #[test]
    fn test_drain_timeout_rolls_back_version() {
        let config = HotReloadConfig {
            drain_timeout_ms: 50,
            rollback_on_failure: true,
            ..Default::default()
        };
        let mgr = Arc::new(HotReloadManager::new(config));
        let _guard = mgr.begin_request();
        mgr.prepare_new_version(1).expect("prepare");
        let _ = mgr.commit_swap(1);
        // After timeout with rollback the current_version should be 0.
        assert_eq!(mgr.current_version(), 0);
    }

    // 18. guard carries correct version_id
    #[test]
    fn test_guard_carries_correct_version_id() {
        let mgr = default_manager();
        mgr.prepare_new_version(5).expect("prepare");
        mgr.commit_swap(5).expect("swap");
        let guard = mgr.begin_request();
        assert_eq!(guard.version_id(), 5);
    }

    // 19. prepare already-active version returns AlreadyActive
    #[test]
    fn test_prepare_active_version_returns_error() {
        let mgr = default_manager();
        let err = mgr.prepare_new_version(0).unwrap_err();
        assert!(matches!(
            err,
            HotReloadError::AlreadyActive { version_id: 0 }
        ));
    }

    // 20. SwapResult drain_duration_ms is reasonable
    #[test]
    fn test_swap_result_drain_duration_reasonable() {
        let mgr = default_manager();
        mgr.prepare_new_version(1).expect("prepare");
        let result = mgr.commit_swap(1).expect("swap");
        // With no in-flight requests the drain should complete near-instantly.
        assert!(result.drain_duration_ms < 1000);
        assert_eq!(result.old_version, 0);
        assert_eq!(result.new_version, 1);
    }
}

// ─── ModelVersionMetrics ─────────────────────────────────────────────────────

/// Runtime performance metrics for a single named model version.
#[derive(Debug, Clone, Default)]
pub struct ModelVersionMetrics {
    /// Total number of requests successfully served by this version.
    pub requests_served: u64,
    /// Exponential moving average of per-request latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Total number of errors attributed to this version.
    pub error_count: u64,
}

impl ModelVersionMetrics {
    /// Record a successfully completed request with the given latency.
    ///
    /// Updates `requests_served` and blends `latency_ms` into the EMA:
    /// `avg = 0.9 * avg + 0.1 * latency_ms`.
    pub fn record_request(&mut self, latency_ms: f64) {
        self.requests_served += 1;
        self.avg_latency_ms = 0.9 * self.avg_latency_ms + 0.1 * latency_ms;
    }

    /// Record a failed request, incrementing the error counter.
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }
}

// ─── NamedModelVersion ────────────────────────────────────────────────────────

/// Metadata for a named model version managed by [`HotReloadController`].
///
/// Unlike [`ModelVersion`] which uses `u64` identifiers, this variant uses a
/// human-readable `String` version ID and tracks the path to the model weights.
#[derive(Debug, Clone)]
pub struct NamedModelVersion {
    /// Human-readable version identifier, e.g. `"v1"`, `"v2"`.
    pub version_id: String,
    /// Filesystem path (or URI) to the model weights file.
    pub model_path: String,
    /// Wall-clock time at which this version was loaded.
    pub loaded_at: std::time::SystemTime,
    /// Accumulated performance metrics for this version.
    pub metrics: ModelVersionMetrics,
}

// ─── HotReloadControllerConfig ───────────────────────────────────────────────

/// Configuration for [`HotReloadController`].
#[derive(Debug, Clone)]
pub struct HotReloadControllerConfig {
    /// Interval between automatic file-change checks in milliseconds.
    pub check_interval_ms: u64,
    /// Maximum number of retries when attempting a model reload.
    pub max_reload_retries: u32,
    /// When true the controller reverts to the previous version if a reload
    /// fails after all retries are exhausted.
    pub rollback_on_error: bool,
}

impl Default for HotReloadControllerConfig {
    fn default() -> Self {
        Self {
            check_interval_ms: 1000,
            max_reload_retries: 3,
            rollback_on_error: true,
        }
    }
}

// ─── HotReloadController internals ───────────────────────────────────────────

/// Internal bookkeeping for a single named model.
struct ModelEntry {
    current: NamedModelVersion,
    /// Previously-active versions, oldest-first.
    history: Vec<NamedModelVersion>,
}

// ─── HotReloadController ─────────────────────────────────────────────────────

/// High-level controller for hot-swapping named models without downtime.
///
/// Models are identified by a string name (e.g. `"bert-qa"`).  Each swap
/// atomically replaces the current version and pushes the old version into a
/// bounded history.  Rolling back restores the most recent historical version.
pub struct HotReloadController {
    config: HotReloadControllerConfig,
    models: std::sync::RwLock<HashMap<String, ModelEntry>>,
}

impl HotReloadController {
    /// Create a new controller with the given configuration.
    pub fn new(config: HotReloadControllerConfig) -> Self {
        Self {
            config,
            models: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Register a new model under `name` pointing to `path`.
    ///
    /// Returns [`HotReloadError::AlreadyActive`] if the name is already
    /// registered (use [`Self::swap_model`] to update an existing registration).
    pub fn register_model(&self, name: &str, path: &str) -> Result<(), HotReloadError> {
        let mut models = self
            .models
            .write()
            .map_err(|e| HotReloadError::LockPoisoned { msg: e.to_string() })?;
        if models.contains_key(name) {
            return Err(HotReloadError::AlreadyActive { version_id: 0 });
        }
        let version = NamedModelVersion {
            version_id: "v1".to_string(),
            model_path: path.to_string(),
            loaded_at: std::time::SystemTime::now(),
            metrics: ModelVersionMetrics::default(),
        };
        models.insert(
            name.to_string(),
            ModelEntry {
                current: version,
                history: Vec::new(),
            },
        );
        Ok(())
    }

    /// Atomically swap the active model version for `name` to one at `new_path`.
    ///
    /// The current version is pushed into the history (bounded to 10 entries;
    /// oldest entries are evicted first).  Returns a clone of the new current
    /// version.
    ///
    /// Returns [`HotReloadError::VersionNotFound`] if `name` is not registered.
    pub fn swap_model(
        &self,
        name: &str,
        new_path: &str,
    ) -> Result<NamedModelVersion, HotReloadError> {
        let mut models = self
            .models
            .write()
            .map_err(|e| HotReloadError::LockPoisoned { msg: e.to_string() })?;
        let entry =
            models.get_mut(name).ok_or(HotReloadError::VersionNotFound { version_id: 0 })?;

        // Determine the next version id based on history depth.
        let new_version_id = format!("v{}", entry.history.len() + 2);

        // Push current into history, evicting the oldest if necessary.
        let old = entry.current.clone();
        entry.history.push(old);
        const HISTORY_LIMIT: usize = 10;
        if entry.history.len() > HISTORY_LIMIT {
            entry.history.remove(0);
        }

        entry.current = NamedModelVersion {
            version_id: new_version_id,
            model_path: new_path.to_string(),
            loaded_at: std::time::SystemTime::now(),
            metrics: ModelVersionMetrics::default(),
        };

        Ok(entry.current.clone())
    }

    /// Roll back `name` to the previous version in the history.
    ///
    /// Returns [`HotReloadError::NoActiveVersion`] if the history is empty.
    /// Returns [`HotReloadError::VersionNotFound`] if `name` is not registered.
    pub fn rollback_model(&self, name: &str) -> Result<NamedModelVersion, HotReloadError> {
        let mut models = self
            .models
            .write()
            .map_err(|e| HotReloadError::LockPoisoned { msg: e.to_string() })?;
        let entry =
            models.get_mut(name).ok_or(HotReloadError::VersionNotFound { version_id: 0 })?;
        let previous = entry.history.pop().ok_or(HotReloadError::NoActiveVersion)?;
        entry.current = previous;
        Ok(entry.current.clone())
    }

    /// Return a clone of the currently-active version for `name`, if registered.
    pub fn get_current_version(&self, name: &str) -> Option<NamedModelVersion> {
        self.models.read().ok()?.get(name).map(|e| e.current.clone())
    }

    /// Return a cloned snapshot of the version history for `name`.
    ///
    /// The returned vector is ordered oldest-first.  Returns an empty vec if
    /// the name is unknown.
    pub fn get_version_history(&self, name: &str) -> Vec<NamedModelVersion> {
        self.models
            .read()
            .ok()
            .and_then(|guard| guard.get(name).map(|e| e.history.clone()))
            .unwrap_or_default()
    }

    /// Record a successfully-completed request against the current version of `name`.
    pub fn record_request_metrics(
        &self,
        name: &str,
        latency_ms: f64,
    ) -> Result<(), HotReloadError> {
        let mut models = self
            .models
            .write()
            .map_err(|e| HotReloadError::LockPoisoned { msg: e.to_string() })?;
        let entry =
            models.get_mut(name).ok_or(HotReloadError::VersionNotFound { version_id: 0 })?;
        entry.current.metrics.record_request(latency_ms);
        Ok(())
    }

    /// Record an error against the current version of `name`.
    pub fn record_error_metrics(&self, name: &str) -> Result<(), HotReloadError> {
        let mut models = self
            .models
            .write()
            .map_err(|e| HotReloadError::LockPoisoned { msg: e.to_string() })?;
        let entry =
            models.get_mut(name).ok_or(HotReloadError::VersionNotFound { version_id: 0 })?;
        entry.current.metrics.record_error();
        Ok(())
    }

    /// Return a reference to the controller's configuration.
    pub fn config(&self) -> &HotReloadControllerConfig {
        &self.config
    }
}

// ─── HotReloadController tests ───────────────────────────────────────────────

#[cfg(test)]
mod controller_tests {
    use super::*;

    fn make_controller() -> HotReloadController {
        HotReloadController::new(HotReloadControllerConfig::default())
    }

    // 1. ModelVersionMetrics::record_request increments requests_served
    #[test]
    fn test_metrics_record_request_increments_count() {
        let mut m = ModelVersionMetrics::default();
        m.record_request(10.0);
        assert_eq!(m.requests_served, 1);
        m.record_request(20.0);
        assert_eq!(m.requests_served, 2);
    }

    // 2. ModelVersionMetrics::record_error increments error_count
    #[test]
    fn test_metrics_record_error_increments_count() {
        let mut m = ModelVersionMetrics::default();
        m.record_error();
        m.record_error();
        assert_eq!(m.error_count, 2);
    }

    // 3. ModelVersionMetrics avg_latency_ms updates with EMA
    #[test]
    fn test_metrics_avg_latency_ema() {
        let mut m = ModelVersionMetrics::default();
        // After first call: avg = 0.9 * 0 + 0.1 * 100 = 10.0
        m.record_request(100.0);
        assert!((m.avg_latency_ms - 10.0).abs() < 1e-9);
        // After second call: avg = 0.9 * 10.0 + 0.1 * 100 = 9.0 + 10.0 = 19.0
        m.record_request(100.0);
        assert!((m.avg_latency_ms - 19.0).abs() < 1e-9);
    }

    // 4. HotReloadController::register_model succeeds on first registration
    #[test]
    fn test_register_model_succeeds() {
        let ctrl = make_controller();
        ctrl.register_model("bert", "/models/bert.bin").expect("register");
        let v = ctrl.get_current_version("bert").expect("get_current");
        assert_eq!(v.version_id, "v1");
        assert_eq!(v.model_path, "/models/bert.bin");
    }

    // 5. HotReloadController::register_model fails if model already registered
    #[test]
    fn test_register_model_duplicate_fails() {
        let ctrl = make_controller();
        ctrl.register_model("gpt2", "/models/gpt2.bin").expect("first register");
        let err = ctrl.register_model("gpt2", "/models/gpt2-v2.bin").unwrap_err();
        assert!(matches!(err, HotReloadError::AlreadyActive { .. }));
    }

    // 6. HotReloadController::get_current_version returns None for unknown model
    #[test]
    fn test_get_current_version_unknown() {
        let ctrl = make_controller();
        assert!(ctrl.get_current_version("nonexistent").is_none());
    }

    // 7. HotReloadController::swap_model returns new version with incremented id
    #[test]
    fn test_swap_model_increments_version_id() {
        let ctrl = make_controller();
        ctrl.register_model("llama", "/models/llama-v1.bin").expect("register");
        let new_v = ctrl.swap_model("llama", "/models/llama-v2.bin").expect("swap");
        assert_eq!(new_v.version_id, "v2");
        assert_eq!(new_v.model_path, "/models/llama-v2.bin");
    }

    // 8. HotReloadController::swap_model moves old version to history
    #[test]
    fn test_swap_model_moves_old_to_history() {
        let ctrl = make_controller();
        ctrl.register_model("phi3", "/models/phi3-v1.bin").expect("register");
        ctrl.swap_model("phi3", "/models/phi3-v2.bin").expect("swap");
        let history = ctrl.get_version_history("phi3");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].version_id, "v1");
    }

    // 9. HotReloadController::rollback_model restores previous version
    #[test]
    fn test_rollback_model_restores_previous() {
        let ctrl = make_controller();
        ctrl.register_model("mistral", "/models/mistral-v1.bin").expect("register");
        ctrl.swap_model("mistral", "/models/mistral-v2.bin").expect("swap");
        let restored = ctrl.rollback_model("mistral").expect("rollback");
        assert_eq!(restored.version_id, "v1");
        assert_eq!(restored.model_path, "/models/mistral-v1.bin");
    }

    // 10. HotReloadController::rollback_model on empty history returns error
    #[test]
    fn test_rollback_model_empty_history_error() {
        let ctrl = make_controller();
        ctrl.register_model("falcon", "/models/falcon.bin").expect("register");
        let err = ctrl.rollback_model("falcon").unwrap_err();
        assert!(matches!(err, HotReloadError::NoActiveVersion));
    }

    // 11. HotReloadController::get_version_history grows after swaps
    #[test]
    fn test_version_history_grows_after_swaps() {
        let ctrl = make_controller();
        ctrl.register_model("qwen", "/models/qwen-v1.bin").expect("register");
        ctrl.swap_model("qwen", "/models/qwen-v2.bin").expect("swap 1");
        ctrl.swap_model("qwen", "/models/qwen-v3.bin").expect("swap 2");
        let history = ctrl.get_version_history("qwen");
        assert_eq!(history.len(), 2);
    }

    // 12. HotReloadController::record_request_metrics updates metrics
    #[test]
    fn test_record_request_metrics_updates() {
        let ctrl = make_controller();
        ctrl.register_model("gemma", "/models/gemma.bin").expect("register");
        ctrl.record_request_metrics("gemma", 50.0).expect("record");
        ctrl.record_request_metrics("gemma", 50.0).expect("record");
        let v = ctrl.get_current_version("gemma").expect("get");
        assert_eq!(v.metrics.requests_served, 2);
    }

    // 13. HotReloadController::record_error_metrics updates error count
    #[test]
    fn test_record_error_metrics_updates() {
        let ctrl = make_controller();
        ctrl.register_model("starcoder", "/models/starcoder.bin").expect("register");
        ctrl.record_error_metrics("starcoder").expect("record error");
        let v = ctrl.get_current_version("starcoder").expect("get");
        assert_eq!(v.metrics.error_count, 1);
    }

    // 14. HotReloadControllerConfig default values
    #[test]
    fn test_controller_config_defaults() {
        let cfg = HotReloadControllerConfig::default();
        assert_eq!(cfg.check_interval_ms, 1000);
        assert_eq!(cfg.max_reload_retries, 3);
        assert!(cfg.rollback_on_error);
    }

    // ── Extended tests ─────────────────────────────────────────────────────

    // 15. HotReloadController::swap_model — current version changes
    #[test]
    fn test_swap_model_current_version_updates() {
        let ctrl = make_controller();
        ctrl.register_model("phi2", "/models/phi2-v1.bin").expect("register");
        ctrl.swap_model("phi2", "/models/phi2-v2.bin").expect("swap");
        let current = ctrl.get_current_version("phi2").expect("get current");
        assert_eq!(
            current.version_id, "v2",
            "current version must be v2 after swap"
        );
    }

    // 16. HotReloadController::rollback_model — history shrinks by one
    #[test]
    fn test_rollback_model_history_shrinks() {
        let ctrl = make_controller();
        ctrl.register_model("gpt3", "/models/gpt3-v1.bin").expect("register");
        ctrl.swap_model("gpt3", "/models/gpt3-v2.bin").expect("swap");
        let before_len = ctrl.get_version_history("gpt3").len();
        ctrl.rollback_model("gpt3").expect("rollback");
        let after_len = ctrl.get_version_history("gpt3").len();
        assert!(after_len < before_len, "history must shrink after rollback");
    }

    // 17. HotReloadController::record_request_metrics — avg_latency changes
    #[test]
    fn test_record_request_metrics_latency_changes() {
        let ctrl = make_controller();
        ctrl.register_model("bloom", "/models/bloom.bin").expect("register");
        ctrl.record_request_metrics("bloom", 200.0).expect("record");
        let v = ctrl.get_current_version("bloom").expect("get");
        assert!(
            v.metrics.avg_latency_ms > 0.0,
            "avg_latency must be > 0 after recording"
        );
    }

    // 18. HotReloadController::record_error_metrics — error_count increments
    #[test]
    fn test_record_error_metrics_increments_error_count() {
        let ctrl = make_controller();
        ctrl.register_model("t5", "/models/t5.bin").expect("register");
        ctrl.record_error_metrics("t5").expect("err 1");
        ctrl.record_error_metrics("t5").expect("err 2");
        ctrl.record_error_metrics("t5").expect("err 3");
        let v = ctrl.get_current_version("t5").expect("get");
        assert_eq!(v.metrics.error_count, 3);
    }

    // 19. HotReloadController::swap_model on unknown model returns error
    #[test]
    fn test_swap_unknown_model_returns_error() {
        let ctrl = make_controller();
        let err = ctrl.swap_model("no-such-model", "/models/x.bin").unwrap_err();
        assert!(matches!(err, HotReloadError::VersionNotFound { .. }));
    }

    // 20. HotReloadController::config returns the stored config
    #[test]
    fn test_controller_config_accessor() {
        let cfg = HotReloadControllerConfig {
            check_interval_ms: 500,
            max_reload_retries: 5,
            rollback_on_error: false,
        };
        let ctrl = HotReloadController::new(cfg.clone());
        assert_eq!(ctrl.config().check_interval_ms, 500);
        assert_eq!(ctrl.config().max_reload_retries, 5);
        assert!(!ctrl.config().rollback_on_error);
    }

    // 21. ModelVersionMetrics::default — all zeros
    #[test]
    fn test_model_version_metrics_default_zeros() {
        let m = ModelVersionMetrics::default();
        assert_eq!(m.requests_served, 0);
        assert_eq!(m.error_count, 0);
        assert!((m.avg_latency_ms - 0.0).abs() < 1e-9);
    }

    // 22. HotReloadError::DrainTimeout — display contains timeout_ms
    #[test]
    fn test_hot_reload_error_drain_timeout_display() {
        let e = HotReloadError::DrainTimeout {
            timeout_ms: 5000,
            remaining: 3,
        };
        let s = e.to_string();
        assert!(s.contains("5000"), "display must contain timeout_ms");
        assert!(s.contains("3"), "display must contain remaining count");
    }

    // 23. HotReloadError::VersionNotFound — display contains version_id
    #[test]
    fn test_hot_reload_error_version_not_found_display() {
        let e = HotReloadError::VersionNotFound { version_id: 42 };
        assert!(e.to_string().contains("42"));
    }

    // 24. HotReloadConfig::default — rollback_on_failure is true
    #[test]
    fn test_hot_reload_config_default_rollback_on_failure() {
        let cfg = HotReloadConfig::default();
        assert!(cfg.rollback_on_failure);
    }

    // 25. HotReloadConfig::default — drain_timeout_ms is 5000
    #[test]
    fn test_hot_reload_config_default_drain_timeout() {
        let cfg = HotReloadConfig::default();
        assert_eq!(cfg.drain_timeout_ms, 5000);
    }

    // 26. HotReloadManager::begin_request increments in_flight counter
    #[test]
    fn test_hot_reload_manager_begin_request_increments_inflight() {
        let mgr = HotReloadManager::new(HotReloadConfig::default());
        let initial = mgr.in_flight.load(std::sync::atomic::Ordering::Acquire);
        let _guard = mgr.begin_request();
        let after = mgr.in_flight.load(std::sync::atomic::Ordering::Acquire);
        assert_eq!(after, initial + 1, "begin_request must increment in_flight");
    }

    // 27. RequestGuard::version_id returns active version
    #[test]
    fn test_request_guard_version_id() {
        let mgr = HotReloadManager::new(HotReloadConfig::default());
        let guard = mgr.begin_request();
        // Initial version is 0
        assert_eq!(guard.version_id(), 0);
    }

    // 28. RequestGuard drop — decrements in_flight counter
    #[test]
    fn test_request_guard_drop_decrements_inflight() {
        let mgr = HotReloadManager::new(HotReloadConfig::default());
        {
            let _guard = mgr.begin_request();
            assert_eq!(mgr.in_flight.load(std::sync::atomic::Ordering::Acquire), 1);
        }
        // After drop
        assert_eq!(mgr.in_flight.load(std::sync::atomic::Ordering::Acquire), 0);
    }

    // 29. Multiple versions accumulate in history correctly
    #[test]
    fn test_multiple_swaps_history_length() {
        let ctrl = make_controller();
        ctrl.register_model("roberta", "/models/r-v1.bin").expect("register");
        for i in 2..=5 {
            ctrl.swap_model("roberta", &format!("/models/r-v{i}.bin")).expect("swap");
        }
        let history = ctrl.get_version_history("roberta");
        assert_eq!(history.len(), 4, "4 swaps should produce 4 history entries");
    }

    // 30. HotReloadError::AlreadyActive display contains version_id
    #[test]
    fn test_hot_reload_error_already_active_display() {
        let e = HotReloadError::AlreadyActive { version_id: 7 };
        assert!(e.to_string().contains("7"));
    }
}
