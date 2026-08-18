//! # Cleanup Handler Implementations
//!
//! This module contains concrete implementations of cleanup handlers for
//! various memory cleanup strategies. Each handler specializes in a specific
//! type of memory cleanup operation.
//!
//! ## Available Handlers
//!
//! - **CacheEvictionHandler**: Manages cache eviction using a pluggable [`CacheManager`]
//! - **ModelUnloadingHandler**: Unloads resident models through a [`ModelRegistry`]
//! - **RequestRejectionHandler**: Implements backpressure by rejecting new requests
//!
//! Every handler here delegates the actual release to a component that owns the
//! memory, and reports back exactly what that component released. A handler
//! that cannot delegate cannot honestly report a byte count, which is why the
//! garbage-collection and buffer-compaction handlers were removed in 0.2.1 —
//! see the note below.
//!
//! ## Handler Characteristics
//!
//! | Handler | Speed | Effectiveness | Risk | Priority |
//! |---------|-------|---------------|------|----------|
//! | Cache   | Fast  | High          | Low  | Medium   |
//! | Model   | Slow  | Very High     | Medium| High    |
//! | Request | Fast  | None (backpressure only) | High | Low |

use super::{CacheManager, CleanupHandler, ModelRegistry};
use crate::memory_pressure::config::*;
use anyhow::Result;
use std::{sync::Arc, time::Duration};
use tracing::{debug, info, warn};

// =============================================================================
// REMOVED IN 0.2.1: GarbageCollectionHandler and BufferCompactionHandler
// =============================================================================
//
// Both handlers reported memory they never freed.
//
// `GarbageCollectionHandler::cleanup` slept for 10 or 50 ms and then returned
// one of five hard-coded byte counts (5 MiB to 30 MiB) chosen by a `match` on
// the pressure level. Rust has no runtime garbage collector to force, this
// crate installs no instrumented `#[global_allocator]`, and the handler
// touched no allocation: every byte it reported as reclaimed was invented, and
// `MemoryPressureHandler` added those bytes to its cleanup totals.
//
// `BufferCompactionHandler` was the same shape: `get_buffer_memory` returned a
// constant 50 MiB, `compact_buffers` slept for 20 ms and multiplied that
// constant by an assumed 15% fragmentation ratio. There is no buffer pool
// behind it to compact.
//
// `CacheEvictionHandler` below shows what a real handler looks like here: it
// takes a `CacheManager` and returns whatever that manager reports having
// evicted. `ModelUnloadingHandler` now follows the same shape with
// `ModelRegistry`. A handler for genuine heap compaction would need the same
// thing — an owner of the memory to delegate to.

// =============================================================================
// Cache Eviction Handler
// =============================================================================

/// Cache eviction cleanup handler
///
/// This handler uses a pluggable cache manager to evict cached data.
/// It's highly effective and fast, making it one of the best cleanup
/// strategies for applications with significant caching.
///
/// ## Characteristics
///
/// - **Speed**: Fast (typically completes in 5-20ms)
/// - **Effectiveness**: High (can free large amounts of memory quickly)
/// - **Risk**: Low (cached data can be recomputed)
/// - **Best Used**: For applications with large caches
#[derive(Debug)]
pub struct CacheEvictionHandler {
    /// Cache manager for handling cache operations
    cache_manager: Arc<dyn CacheManager>,

    /// Eviction strategy parameters
    max_eviction_percentage: f32,
    min_cache_retention: f32,
}

impl CacheEvictionHandler {
    /// Create a new cache eviction handler
    pub fn new(cache_manager: Arc<dyn CacheManager>) -> Self {
        Self {
            cache_manager,
            max_eviction_percentage: 0.8, // Maximum 80% eviction
            min_cache_retention: 0.1,     // Minimum 10% retention
        }
    }

    /// Create a new aggressive cache eviction handler
    pub fn new_aggressive(cache_manager: Arc<dyn CacheManager>) -> Self {
        Self {
            cache_manager,
            max_eviction_percentage: 0.95, // Maximum 95% eviction
            min_cache_retention: 0.05,     // Minimum 5% retention
        }
    }

    /// Eviction fraction for `pressure_level`, bounded by both configured
    /// limits.
    ///
    /// `min_cache_retention` used to be accepted, stored and never consulted:
    /// a handler built with `new_aggressive` advertised "minimum 5% retention"
    /// while `calculate_eviction_percentage` was free to return 0.9. Both
    /// bounds are enforced now.
    fn calculate_eviction_percentage(&self, pressure_level: MemoryPressureLevel) -> f32 {
        let base_percentage = match pressure_level {
            MemoryPressureLevel::Normal => 0.0_f32,
            MemoryPressureLevel::Low => 0.1_f32,
            MemoryPressureLevel::Medium => 0.3_f32,
            MemoryPressureLevel::High => 0.6_f32,
            MemoryPressureLevel::Critical => 0.8_f32,
            MemoryPressureLevel::Emergency => 0.9_f32,
        };

        base_percentage
            .min(self.max_eviction_percentage)
            .min(1.0_f32 - self.min_cache_retention)
            .max(0.0_f32)
    }
}

impl CleanupHandler for CacheEvictionHandler {
    fn cleanup(&self, pressure_level: MemoryPressureLevel) -> Result<u64> {
        let eviction_percentage = self.calculate_eviction_percentage(pressure_level);

        if eviction_percentage <= 0.0 {
            return Ok(0);
        }

        // Perform cache eviction
        let memory_freed = self.cache_manager.evict_percentage(eviction_percentage)?;

        debug!(
            "Cache eviction freed {} bytes ({:.1}% of cache)",
            memory_freed,
            eviction_percentage * 100.0
        );

        Ok(memory_freed)
    }

    fn estimate_memory_freed(&self) -> u64 {
        // Estimate based on evictable cache size
        let evictable_size = self.cache_manager.get_evictable_size();
        (evictable_size as f64 * 0.5) as u64 // Conservative 50% estimate
    }

    fn get_priority(&self) -> u32 {
        110 // Medium-high priority - cache eviction is safe and effective
    }

    fn name(&self) -> &'static str {
        "CacheEviction"
    }

    fn should_execute(&self, pressure_level: MemoryPressureLevel) -> bool {
        // Only execute if there's evictable cache data
        pressure_level > MemoryPressureLevel::Normal && self.cache_manager.get_evictable_size() > 0
    }
}

// =============================================================================
// Model Unloading Handler
// =============================================================================

/// Model unloading cleanup handler
///
/// Releases resident models through a [`ModelRegistry`], which is the component
/// that actually owns them. The handler contributes the policy — how many
/// models to release at a given pressure level, which ones are protected, how
/// small is too small to bother with — and the registry contributes the bytes.
///
/// Before 0.2.1 this handler carried a `find_unloadable_models` that returned
/// three invented models of 200, 150 and 300 MiB and an `unload_model` that
/// slept in proportion to the invented size and reported it as freed. Nothing
/// was ever unloaded, and the fabricated totals flowed straight into the
/// cleanup statistics. It now cannot be constructed without a registry.
///
/// ## Characteristics
///
/// - **Speed**: bounded by the registry's own unload cost
/// - **Effectiveness**: Very High (models can be very large)
/// - **Risk**: Medium (models need to be reloaded)
/// - **Best Used**: For ML applications with multiple models
#[derive(Debug, Clone)]
pub struct ModelUnloadingHandler {
    /// Registry that owns the resident models.
    registry: Arc<dyn ModelRegistry>,

    /// Minimum model size to consider for unloading (in bytes)
    min_model_size: u64,

    /// Maximum models to unload in one cleanup operation
    max_models_per_cleanup: usize,

    /// Models that should never be unloaded (critical models)
    protected_models: Vec<String>,
}

impl ModelUnloadingHandler {
    /// Create a handler that unloads through `registry`.
    pub fn new(registry: Arc<dyn ModelRegistry>) -> Self {
        Self {
            registry,
            min_model_size: 100 * 1024 * 1024, // 100MB minimum
            max_models_per_cleanup: 3,
            protected_models: Vec::new(),
        }
    }

    /// Create a handler that will never unload the named models.
    pub fn with_protected_models(
        registry: Arc<dyn ModelRegistry>,
        protected_models: Vec<String>,
    ) -> Self {
        Self {
            registry,
            min_model_size: 100 * 1024 * 1024,
            max_models_per_cleanup: 3,
            protected_models,
        }
    }

    /// Override the smallest model worth unloading.
    pub fn with_min_model_size(mut self, min_model_size: u64) -> Self {
        self.min_model_size = min_model_size;
        self
    }

    /// Models the registry holds that this handler is willing to unload, in the
    /// registry's own preference order.
    fn unloadable_models(&self) -> Vec<(String, u64)> {
        self.registry
            .resident_models()
            .into_iter()
            .filter(|(name, size)| {
                *size >= self.min_model_size && !self.protected_models.contains(name)
            })
            .collect()
    }

    /// How many models to release at `pressure_level`.
    fn unload_budget(&self, pressure_level: MemoryPressureLevel) -> usize {
        match pressure_level {
            MemoryPressureLevel::Emergency => self.max_models_per_cleanup,
            MemoryPressureLevel::Critical => self.max_models_per_cleanup.saturating_sub(1),
            MemoryPressureLevel::High => 2,
            MemoryPressureLevel::Medium => 1,
            _ => 0,
        }
    }
}

impl CleanupHandler for ModelUnloadingHandler {
    fn cleanup(&self, pressure_level: MemoryPressureLevel) -> Result<u64> {
        let budget = self.unload_budget(pressure_level);
        if budget == 0 {
            return Ok(0);
        }

        let mut total_freed = 0u64;
        for (model_name, _) in self.unloadable_models().into_iter().take(budget) {
            match self.registry.unload_model(&model_name) {
                Ok(0) => debug!("Model '{model_name}' released no memory"),
                Ok(freed) => {
                    info!("Unloaded model '{model_name}' freeing {freed} bytes");
                    total_freed += freed;
                },
                Err(error) => warn!("Failed to unload model '{model_name}': {error}"),
            }
        }

        Ok(total_freed)
    }

    fn estimate_memory_freed(&self) -> u64 {
        // The registry's reported resident sizes are the only evidence
        // available; take the two the registry would release first.
        self.unloadable_models().into_iter().take(2).map(|(_, size)| size).sum()
    }

    fn get_priority(&self) -> u32 {
        80 // Lower priority due to performance impact
    }

    fn name(&self) -> &'static str {
        "ModelUnloading"
    }

    fn should_execute(&self, pressure_level: MemoryPressureLevel) -> bool {
        // Only execute for medium pressure or higher, and only when the
        // registry actually holds something worth unloading.
        pressure_level >= MemoryPressureLevel::Medium && !self.unloadable_models().is_empty()
    }
}

// =============================================================================
// Request Rejection Handler
// =============================================================================

/// Request rejection cleanup handler
///
/// This handler implements backpressure by temporarily rejecting new requests.
/// It doesn't free existing memory but prevents new allocations.
///
/// ## Characteristics
///
/// - **Speed**: Very Fast (no actual cleanup work)
/// - **Effectiveness**: Low (prevents new allocations only)
/// - **Risk**: High (impacts service availability)
/// - **Best Used**: As a last resort for critical memory situations
#[derive(Debug, Clone)]
pub struct RequestRejectionHandler {
    /// Whether request rejection is currently active
    rejection_active: Arc<std::sync::atomic::AtomicBool>,

    /// Rejection rate (0.0-1.0)
    rejection_rate: f32,

    /// Maximum duration to maintain rejection (in seconds)
    max_rejection_duration_secs: u64,

    /// Requests seen since rejection was last activated, used to interleave
    /// rejections at exactly `rejection_rate`.
    requests_seen: Arc<std::sync::atomic::AtomicU64>,
}

impl Default for RequestRejectionHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestRejectionHandler {
    /// Create a new request rejection handler
    pub fn new() -> Self {
        Self {
            rejection_active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            rejection_rate: 0.5,             // Reject 50% of requests
            max_rejection_duration_secs: 60, // Maximum 1 minute
            requests_seen: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Create a new aggressive request rejection handler
    pub fn new_aggressive() -> Self {
        Self {
            rejection_active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            rejection_rate: 0.9,             // Reject 90% of requests
            max_rejection_duration_secs: 30, // Maximum 30 seconds
            requests_seen: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Whether backpressure is currently engaged.
    ///
    /// Distinct from [`RequestRejectionHandler::should_reject_request`], which
    /// answers for one specific request and sheds only `rejection_rate` of
    /// them.
    pub fn is_rejecting(&self) -> bool {
        self.rejection_active.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether the caller's request should be rejected.
    ///
    /// Backpressure sheds `rejection_rate` of the requests that arrive while it
    /// is active, deterministically interleaved: the handler counts calls and
    /// rejects whenever the count crosses the next multiple of `1 /
    /// rejection_rate`. Deterministic rather than sampled, so a load test sees
    /// the configured ratio exactly instead of a distribution around it.
    ///
    /// Before 0.2.1 `rejection_rate` was stored and never read, so a handler
    /// configured to shed 50% shed 100%.
    pub fn should_reject_request(&self) -> bool {
        use std::sync::atomic::Ordering;

        if !self.rejection_active.load(Ordering::Relaxed) {
            return false;
        }
        // Integer permille arithmetic, so the observed ratio is exact rather
        // than drifting by one on a float rounding error.
        let permille = (self.rejection_rate.clamp(0.0, 1.0) * 1000.0).round() as u64;
        if permille == 0 {
            return false;
        }
        if permille >= 1000 {
            return true;
        }

        // `n` counts requests seen while rejection is active. Rejecting when
        // floor(n * rate) advances sheds exactly `rate` of them.
        let n = self.requests_seen.fetch_add(1, Ordering::Relaxed);
        (n + 1) * permille / 1000 > n * permille / 1000
    }

    /// Activate request rejection
    fn activate_rejection(&self) {
        self.requests_seen.store(0, std::sync::atomic::Ordering::Relaxed);
        self.rejection_active.store(true, std::sync::atomic::Ordering::Relaxed);

        // Schedule deactivation after maximum duration
        let rejection_active = self.rejection_active.clone();
        let duration = self.max_rejection_duration_secs;

        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(duration));
            rejection_active.store(false, std::sync::atomic::Ordering::Relaxed);
        });
    }
}

impl CleanupHandler for RequestRejectionHandler {
    /// Turn on backpressure. Always reports zero bytes reclaimed, because
    /// rejecting requests frees nothing that is already allocated — it only
    /// stops new allocations. Reporting an invented 10 MiB here (as this
    /// handler did before 0.2.1) inflated the cleanup engine's totals with
    /// memory that was never released.
    fn cleanup(&self, pressure_level: MemoryPressureLevel) -> Result<u64> {
        match pressure_level {
            MemoryPressureLevel::Critical | MemoryPressureLevel::Emergency => {
                self.activate_rejection();
                warn!("Activated request rejection due to critical memory pressure");
                Ok(0)
            },
            _ => {
                // Don't activate rejection for lower pressure levels
                Ok(0)
            },
        }
    }

    /// Zero: this handler reclaims nothing. See [`RequestRejectionHandler`].
    fn estimate_memory_freed(&self) -> u64 {
        0
    }

    fn get_priority(&self) -> u32 {
        50 // Low priority - use as last resort
    }

    fn name(&self) -> &'static str {
        "RequestRejection"
    }

    fn should_execute(&self, pressure_level: MemoryPressureLevel) -> bool {
        // Only use for critical situations
        pressure_level >= MemoryPressureLevel::Critical
            && !self.rejection_active.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Registry that records what was unloaded, so the handler's reported byte
    /// counts can be checked against real releases.
    #[derive(Debug)]
    struct RecordingRegistry {
        resident: Mutex<Vec<(String, u64)>>,
        unloaded: Mutex<Vec<String>>,
    }

    impl RecordingRegistry {
        fn new(models: Vec<(&str, u64)>) -> Arc<Self> {
            Arc::new(Self {
                resident: Mutex::new(models.into_iter().map(|(n, s)| (n.to_string(), s)).collect()),
                unloaded: Mutex::new(Vec::new()),
            })
        }

        fn unloaded(&self) -> Vec<String> {
            self.unloaded.lock().unwrap_or_else(|p| p.into_inner()).clone()
        }
    }

    impl ModelRegistry for RecordingRegistry {
        fn resident_models(&self) -> Vec<(String, u64)> {
            self.resident.lock().unwrap_or_else(|p| p.into_inner()).clone()
        }

        fn unload_model(&self, model_name: &str) -> Result<u64> {
            let mut resident = self.resident.lock().unwrap_or_else(|p| p.into_inner());
            let Some(position) = resident.iter().position(|(n, _)| n == model_name) else {
                return Ok(0);
            };
            let (_, size) = resident.remove(position);
            self.unloaded
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(model_name.to_string());
            Ok(size)
        }
    }

    const MIB: u64 = 1024 * 1024;

    #[test]
    fn model_unloading_reports_only_what_the_registry_released() {
        let registry = RecordingRegistry::new(vec![
            ("model_a", 200 * MIB),
            ("model_b", 150 * MIB),
            ("model_c", 300 * MIB),
        ]);
        let handler = ModelUnloadingHandler::new(Arc::clone(&registry) as Arc<dyn ModelRegistry>);

        // High pressure unloads two models: 200 MiB + 150 MiB.
        let freed = handler.cleanup(MemoryPressureLevel::High).expect("cleanup should succeed");

        assert_eq!(freed, 350 * MIB);
        assert_eq!(registry.unloaded(), vec!["model_a", "model_b"]);
    }

    #[test]
    fn model_unloading_frees_nothing_when_the_registry_is_empty() {
        let registry = RecordingRegistry::new(Vec::new());
        let handler = ModelUnloadingHandler::new(Arc::clone(&registry) as Arc<dyn ModelRegistry>);

        assert_eq!(handler.estimate_memory_freed(), 0);
        assert!(!handler.should_execute(MemoryPressureLevel::Emergency));
        assert_eq!(
            handler.cleanup(MemoryPressureLevel::Emergency).expect("cleanup should succeed"),
            0,
            "with no resident models there is nothing to free"
        );
        assert!(registry.unloaded().is_empty());
    }

    #[test]
    fn model_unloading_skips_protected_and_undersized_models() {
        let registry = RecordingRegistry::new(vec![
            ("critical", 500 * MIB),
            ("tiny", MIB),
            ("evictable", 400 * MIB),
        ]);
        let handler = ModelUnloadingHandler::with_protected_models(
            Arc::clone(&registry) as Arc<dyn ModelRegistry>,
            vec!["critical".to_string()],
        );

        let freed =
            handler.cleanup(MemoryPressureLevel::Emergency).expect("cleanup should succeed");

        assert_eq!(freed, 400 * MIB);
        assert_eq!(registry.unloaded(), vec!["evictable"]);
    }

    #[test]
    fn model_unloading_estimate_comes_from_the_registry() {
        let registry = RecordingRegistry::new(vec![
            ("model_a", 200 * MIB),
            ("model_b", 150 * MIB),
            ("model_c", 300 * MIB),
        ]);
        let handler = ModelUnloadingHandler::new(registry as Arc<dyn ModelRegistry>);

        assert_eq!(handler.estimate_memory_freed(), 350 * MIB);
    }

    #[test]
    fn model_unloading_does_not_run_below_medium_pressure() {
        let registry = RecordingRegistry::new(vec![("model_a", 200 * MIB)]);
        let handler = ModelUnloadingHandler::new(Arc::clone(&registry) as Arc<dyn ModelRegistry>);

        assert!(!handler.should_execute(MemoryPressureLevel::Low));
        assert!(handler.should_execute(MemoryPressureLevel::High));
        assert_eq!(
            handler.cleanup(MemoryPressureLevel::Low).expect("cleanup should succeed"),
            0
        );
        assert!(registry.unloaded().is_empty());
    }

    #[tokio::test]
    async fn request_rejection_activates_without_claiming_reclaimed_memory() {
        let handler = RequestRejectionHandler::new();

        assert_eq!(handler.name(), "RequestRejection");
        assert_eq!(handler.get_priority(), 50);
        assert!(!handler.should_execute(MemoryPressureLevel::Medium));
        assert!(handler.should_execute(MemoryPressureLevel::Critical));
        assert!(!handler.is_rejecting());

        let freed = handler.cleanup(MemoryPressureLevel::Critical).expect("cleanup should succeed");

        assert!(
            handler.is_rejecting(),
            "critical pressure must engage backpressure"
        );
        assert_eq!(
            freed, 0,
            "rejecting requests releases nothing that is already allocated"
        );
        assert_eq!(handler.estimate_memory_freed(), 0);
    }

    /// Cache manager that reports a fixed evictable size and records the
    /// fraction it was asked to evict.
    #[derive(Debug, Default)]
    struct RecordingCache {
        requested: Mutex<Vec<f32>>,
    }

    impl CacheManager for RecordingCache {
        fn evict_cache(&self, _pressure_level: MemoryPressureLevel) -> Result<u64> {
            Ok(0)
        }

        fn get_cache_size(&self) -> u64 {
            1000
        }

        fn get_evictable_size(&self) -> u64 {
            1000
        }

        fn evict_percentage(&self, percentage: f32) -> Result<u64> {
            self.requested.lock().unwrap_or_else(|p| p.into_inner()).push(percentage);
            Ok((1000.0 * percentage) as u64)
        }
    }

    #[test]
    fn cache_eviction_honours_the_configured_minimum_retention() {
        let cache = Arc::new(RecordingCache::default());
        // `new_aggressive` advertises a 5% minimum retention, so an emergency
        // eviction may ask for at most 95%.
        let handler =
            CacheEvictionHandler::new_aggressive(Arc::clone(&cache) as Arc<dyn CacheManager>);

        handler.cleanup(MemoryPressureLevel::Emergency).expect("cleanup should succeed");

        let requested = cache.requested.lock().unwrap_or_else(|p| p.into_inner()).clone();
        assert_eq!(requested.len(), 1);
        assert!(
            requested[0] <= 0.95,
            "eviction must respect the 5% retention floor, asked for {}",
            requested[0]
        );
    }

    #[test]
    fn cache_eviction_retention_floor_can_bind_before_the_ceiling() {
        let cache = Arc::new(RecordingCache::default());
        // The default handler retains 10%, so 0.9 is the hard ceiling even
        // though `max_eviction_percentage` is 0.8 — the tighter of the two wins.
        let handler = CacheEvictionHandler::new(Arc::clone(&cache) as Arc<dyn CacheManager>);

        handler.cleanup(MemoryPressureLevel::Emergency).expect("cleanup should succeed");

        let requested = cache.requested.lock().unwrap_or_else(|p| p.into_inner()).clone();
        assert!(requested[0] <= 0.8);
        assert!(requested[0] <= 0.9);
    }

    #[test]
    fn request_rejection_sheds_the_configured_fraction() {
        let handler = RequestRejectionHandler::new(); // 50%
        handler.cleanup(MemoryPressureLevel::Emergency).expect("cleanup should succeed");

        let rejected = (0..100).filter(|_| handler.should_reject_request()).count();
        assert_eq!(
            rejected, 50,
            "a handler configured to shed 50% must shed 50%, not everything"
        );
    }

    #[test]
    fn aggressive_request_rejection_sheds_more() {
        let handler = RequestRejectionHandler::new_aggressive(); // 90%
        handler.cleanup(MemoryPressureLevel::Critical).expect("cleanup should succeed");

        let rejected = (0..100).filter(|_| handler.should_reject_request()).count();
        assert_eq!(rejected, 90);
    }

    #[test]
    fn request_rejection_sheds_nothing_while_inactive() {
        let handler = RequestRejectionHandler::new();
        assert!(
            (0..50).all(|_| !handler.should_reject_request()),
            "no request may be shed before backpressure is engaged"
        );
    }
}
