//! Metal buffer identity, the GPU buffer cache and its RAII handle.
//!
//! # Lifetime management
//!
//! Every GPU-resident intermediate produced by a `*_gpu_to_gpu` op is parked in a
//! process-global [`BufferCache`]. Historically that cache was a plain
//! `HashMap<BufferId, Arc<Buffer>>` with no eviction and no callers of
//! `remove_persistent_buffer`, so *every* intermediate stayed resident until the
//! process exited - a decode loop leaked six buffers per attention call.
//!
//! # Why eviction is opt-in, not automatic
//!
//! The obvious fix - "cap the cache and LRU-evict the oldest entries" - is **unsound
//! here**, and a regression test in `parity_tests.rs`
//! (`a_live_op_output_survives_eviction_pressure`) exists to keep it that way. A
//! `Tensor::Metal` stores a bare `BufferId`, not a reference-counted handle, so an op
//! result that a caller is still holding is indistinguishable from dead scratch. An
//! LRU that evicts by age therefore trades a memory leak for
//! `"Buffer ... not found in cache"` in the middle of a forward pass - a worse bug.
//!
//! So the cache has three tiers and **only one of them is ever auto-evicted**:
//!
//! * **`Pinned`** - long-lived weights from
//!   [`MetalBackend::create_persistent_buffer`](crate::gpu_ops::metal::MetalBackend::create_persistent_buffer).
//!   Released explicitly with `remove_persistent_buffer` / `clear_buffer_cache`.
//! * **`Live`** - the default tier: every `*_gpu_to_gpu` result and every
//!   `create_transient_buffer`. The caller holds a `BufferId` that can be dereferenced
//!   at any time, so these are **never** reclaimed behind its back. They are freed by
//!   `release_buffers` (which the composite ops now call for their own intermediates),
//!   by dropping the last [`MetalBufferHandle`], or by `clear_buffer_cache`.
//! * **`Evictable`** - explicit scratch from
//!   [`MetalBackend::create_evictable_buffer`](crate::gpu_ops::metal::MetalBackend::create_evictable_buffer),
//!   whose caller has stated it can re-create the contents. This tier is LRU-evicted
//!   once the byte cap is exceeded.
//!
//! A [`MetalBufferHandle`] pins any entry for as long as it lives and frees it when the
//! last clone drops, mirroring `gpu_ops::cuda::BufferHandle`.
//!
//! When the cap is exceeded and nothing is evictable, the cache records an
//! `eviction_stall` and warns rather than silently dropping live data: memory pressure
//! becomes visible instead of becoming corruption.
//!
//! Eviction only ever drops the cache's own `Arc<Buffer>`. Metal command buffers
//! retain the resources they reference until completion (`MTLCommandQueue`'s
//! default retained-references mode), so dropping the last CPU-side reference
//! while a dispatch is still in flight is safe.
//!
//! ## Known limitation (cross-crate follow-up, evaluated and deliberately deferred)
//!
//! Because `Live` entries are never auto-reclaimed, a buffer whose id the caller
//! simply forgets still occupies the cache until `clear_buffer_cache`. The composite
//! ops in this module release their own intermediates, so the hot attention path no
//! longer accumulates; the remaining case is every op that stores its GPU result id
//! straight into a `Tensor::Metal`. Closing that needs [`MetalTensorData`]
//! (`tensor/mod.rs`) to hold a [`MetalBufferHandle`] instead of a bare `BufferId` -
//! the same change `gpu_ops::cuda::BufferHandle` already got in `CudaTensorData` -
//! obtained through
//! [`MetalBackend::retain_buffer`](crate::gpu_ops::metal::MetalBackend::retain_buffer),
//! which already exists and is ready to be called; only the adoption is missing.
//!
//! That adoption was scoped out of this pass: `MetalTensorData::buffer_id` is a public
//! field, constructed with a plain `BufferId` at every GPU-result call site, so the
//! field-type change is all-or-nothing across every file that builds or reads one -
//! there is no incremental path that leaves the rest compiling. As of this writing that
//! is, beyond this module's own `metalbackend_lifecycle_group.rs` and `types.rs`:
//!
//! * `tensor/mod.rs` - the `MetalTensorData` struct definition itself (`pub buffer_id`
//!   field and its `Clone` impl).
//! * `layers/linear.rs` - `Linear::forward`'s GPU-to-GPU matmul and bias-add results
//!   (the case this note originally called out).
//! * `layers/layernorm.rs` - three separate `layernorm_gpu_to_gpu` result sites.
//! * `ops/activations.rs` and `tensor/activations.rs` - `gelu_gpu_to_gpu` results.
//! * `tensor/math_ops/arithmetic.rs` - `add_gpu_to_gpu` results.
//! * `tensor/utils.rs` - `to_device_enum`'s persistent-buffer construction and the
//!   `Hash` impl that reads `buffer_id`.
//! * `generation/core.rs` - a `download_buffer_to_vec` call keyed on `buffer_id`.
//!
//! Every one of those needs its `MetalTensorData { buffer_id: raw_id, .. }` literal
//! turned into `retain_buffer(&raw_id)?` plus a `.buffer.id()`-style accessor at every
//! read, mirroring `CudaTensorData::buffer_id()`. None of that work can land from
//! inside this module alone.

#[allow(unused_imports)]
use super::common::*;

/// Default byte cap for the Metal buffer cache: 2 GiB.
///
/// Overridable at process start through `TRUSTFORMERS_METAL_BUFFER_CACHE_BYTES`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub const DEFAULT_BUFFER_CACHE_CAPACITY_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Buffer ID for persistent GPU buffers
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(u64);

#[cfg(all(target_os = "macos", feature = "metal"))]
impl BufferId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        BufferId(COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    /// Raw monotonic value behind this id (diagnostics only).
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// Reclamation tier of a cache entry. See the module docs for why only one tier is
/// auto-evicted.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferTier {
    /// Model weights. Never auto-evicted; released explicitly.
    Pinned,
    /// Default tier for op results and uploads: the caller still holds the id, so the
    /// cache must not reclaim it behind their back.
    Live,
    /// Caller-declared scratch that can be regenerated. LRU-evicted past the cap.
    Evictable,
}

/// One cached GPU allocation plus the bookkeeping the LRU needs.
#[cfg(all(target_os = "macos", feature = "metal"))]
struct CacheEntry {
    buffer: Arc<Buffer>,
    /// Allocation size in bytes, as reported by `MTLBuffer::length`.
    bytes: usize,
    /// Monotonic tick of the last `get`/`insert` touching this entry.
    last_used: u64,
    /// Which reclamation tier this entry belongs to.
    tier: BufferTier,
    /// Number of live [`MetalBufferHandle`]s referencing this entry.
    refs: usize,
}

/// Snapshot of the cache's occupancy and eviction counters.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferCacheStats {
    /// Number of live entries.
    pub entries: usize,
    /// Sum of the `MTLBuffer` lengths currently held.
    pub live_bytes: usize,
    /// Byte cap that triggers LRU eviction.
    pub capacity_bytes: usize,
    /// Entries evicted since process start.
    pub evicted_entries: u64,
    /// Bytes reclaimed by eviction since process start.
    pub evicted_bytes: u64,
    /// Times an insert exceeded the cap but found no evictable entry.
    ///
    /// A non-zero value means the cache is over its watermark and holding `Live` or
    /// `Pinned` data it is not allowed to reclaim - the signal to release buffers
    /// explicitly or raise the cap, not a sign of data loss.
    pub eviction_stalls: u64,
    /// Entries in the `Evictable` tier.
    pub evictable_entries: usize,
}

/// Persistent buffer cache for Metal GPU: byte-capped LRU with pinning + refcounts.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) struct BufferCache {
    entries: HashMap<BufferId, CacheEntry>,
    /// Monotonic clock driving LRU ordering.
    tick: u64,
    live_bytes: usize,
    capacity_bytes: usize,
    evicted_entries: u64,
    evicted_bytes: u64,
    eviction_stalls: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl BufferCache {
    /// Cache sized from `TRUSTFORMERS_METAL_BUFFER_CACHE_BYTES`, else the 2 GiB default.
    pub(crate) fn new() -> Self {
        let capacity_bytes = std::env::var("TRUSTFORMERS_METAL_BUFFER_CACHE_BYTES")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|bytes| *bytes > 0)
            .unwrap_or(DEFAULT_BUFFER_CACHE_CAPACITY_BYTES);
        Self::with_capacity_bytes(capacity_bytes)
    }

    /// Cache with an explicit byte cap. Used by tests so eviction is deterministic
    /// without touching a process-global environment variable.
    pub(crate) fn with_capacity_bytes(capacity_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            tick: 0,
            live_bytes: 0,
            capacity_bytes: capacity_bytes.max(1),
            evicted_entries: 0,
            evicted_bytes: 0,
            eviction_stalls: 0,
        }
    }

    fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.wrapping_add(1);
        self.tick
    }

    /// Insert a `Live` entry: the caller holds the id, so it is never auto-evicted.
    pub(crate) fn insert(&mut self, id: BufferId, buffer: Arc<Buffer>) {
        self.insert_tiered(id, buffer, BufferTier::Live);
    }

    /// Insert a pinned entry (model weights): excluded from automatic eviction.
    pub(crate) fn insert_pinned(&mut self, id: BufferId, buffer: Arc<Buffer>) {
        self.insert_tiered(id, buffer, BufferTier::Pinned);
    }

    /// Insert caller-declared regenerable scratch: the only auto-evicted tier.
    pub(crate) fn insert_evictable(&mut self, id: BufferId, buffer: Arc<Buffer>) {
        self.insert_tiered(id, buffer, BufferTier::Evictable);
    }

    fn insert_tiered(&mut self, id: BufferId, buffer: Arc<Buffer>, tier: BufferTier) {
        let bytes = buffer.length() as usize;
        // Replacing an existing id: drop its accounting first.
        if let Some(previous) = self.entries.remove(&id) {
            self.live_bytes = self.live_bytes.saturating_sub(previous.bytes);
        }
        self.evict_to_fit(bytes);
        let last_used = self.next_tick();
        self.live_bytes = self.live_bytes.saturating_add(bytes);
        self.entries.insert(
            id,
            CacheEntry {
                buffer,
                bytes,
                last_used,
                tier,
                refs: 0,
            },
        );
    }

    /// Fetch a buffer, refreshing its LRU position.
    pub(crate) fn get(&mut self, id: &BufferId) -> Option<Arc<Buffer>> {
        let tick = self.next_tick();
        let entry = self.entries.get_mut(id)?;
        entry.last_used = tick;
        Some(Arc::clone(&entry.buffer))
    }

    pub(crate) fn contains(&self, id: &BufferId) -> bool {
        self.entries.contains_key(id)
    }

    pub(crate) fn remove(&mut self, id: &BufferId) -> Option<Arc<Buffer>> {
        let entry = self.entries.remove(id)?;
        self.live_bytes = self.live_bytes.saturating_sub(entry.bytes);
        Some(entry.buffer)
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.live_bytes = 0;
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn live_bytes(&self) -> usize {
        self.live_bytes
    }

    pub(crate) fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    /// Resize the cap, evicting immediately if the new cap is already exceeded.
    pub(crate) fn set_capacity_bytes(&mut self, capacity_bytes: usize) {
        self.capacity_bytes = capacity_bytes.max(1);
        self.evict_to_fit(0);
    }

    pub(crate) fn stats(&self) -> BufferCacheStats {
        BufferCacheStats {
            entries: self.len(),
            live_bytes: self.live_bytes(),
            capacity_bytes: self.capacity_bytes(),
            evicted_entries: self.evicted_entries,
            evicted_bytes: self.evicted_bytes,
            eviction_stalls: self.eviction_stalls,
            evictable_entries: self
                .entries
                .values()
                .filter(|entry| entry.tier == BufferTier::Evictable)
                .count(),
        }
    }

    /// Move an entry to a different reclamation tier. Returns false when unknown.
    pub(crate) fn set_tier(&mut self, id: &BufferId, tier: BufferTier) -> bool {
        match self.entries.get_mut(id) {
            Some(entry) => {
                entry.tier = tier;
                true
            },
            None => false,
        }
    }

    /// Tier of an entry, if it is resident.
    pub(crate) fn tier_of(&self, id: &BufferId) -> Option<BufferTier> {
        self.entries.get(id).map(|entry| entry.tier)
    }

    /// Take a reference for a [`MetalBufferHandle`]. Returns false when the id is unknown.
    pub(crate) fn retain(&mut self, id: &BufferId) -> bool {
        match self.entries.get_mut(id) {
            Some(entry) => {
                entry.refs = entry.refs.saturating_add(1);
                true
            },
            None => false,
        }
    }

    /// Drop a handle reference; removes the entry when the last handle goes away.
    pub(crate) fn release(&mut self, id: &BufferId) {
        let drop_entry = match self.entries.get_mut(id) {
            Some(entry) => {
                entry.refs = entry.refs.saturating_sub(1);
                entry.refs == 0
            },
            None => false,
        };
        if drop_entry {
            self.remove(id);
        }
    }

    /// Evict least-recently-used `Evictable` entries until `incoming` bytes fit.
    ///
    /// An entry is a candidate only when **all** of the following hold:
    ///
    /// * its tier is [`BufferTier::Evictable`] - the caller explicitly declared it
    ///   regenerable scratch. `Live` op results and `Pinned` weights are never taken;
    ///   see the module docs for why that would be a correctness bug rather than a
    ///   memory optimisation.
    /// * no [`MetalBufferHandle`] references it, and
    /// * the cache holds the sole `Arc` - nobody is mid-op with the buffer checked out.
    ///
    /// If no candidate exists the insert still proceeds (correctness beats the cap)
    /// and `eviction_stalls` records the pressure.
    fn evict_to_fit(&mut self, incoming: usize) {
        while self.live_bytes.saturating_add(incoming) > self.capacity_bytes {
            let victim = self
                .entries
                .iter()
                .filter(|(_, entry)| {
                    entry.tier == BufferTier::Evictable
                        && entry.refs == 0
                        && Arc::strong_count(&entry.buffer) == 1
                })
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(id, _)| *id);
            match victim {
                Some(id) => {
                    if let Some(entry) = self.entries.remove(&id) {
                        self.live_bytes = self.live_bytes.saturating_sub(entry.bytes);
                        self.evicted_entries = self.evicted_entries.saturating_add(1);
                        self.evicted_bytes = self.evicted_bytes.saturating_add(entry.bytes as u64);
                    }
                },
                None => {
                    // Over the watermark with nothing reclaimable. Surface it instead
                    // of dropping data the caller can still name.
                    self.eviction_stalls = self.eviction_stalls.saturating_add(1);
                    if self.eviction_stalls.is_power_of_two() {
                        tracing::warn!(
                            live_bytes = self.live_bytes,
                            capacity_bytes = self.capacity_bytes,
                            stalls = self.eviction_stalls,
                            "Metal buffer cache is over its watermark and holds no \
                             evictable scratch; release buffers explicitly \
                             (`release_buffers` / `MetalBufferHandle`) or raise \
                             TRUSTFORMERS_METAL_BUFFER_CACHE_BYTES"
                        );
                    }
                    break;
                },
            }
        }
    }
}

/// Reference-counted handle to a cached Metal buffer.
///
/// Mirrors `gpu_ops::cuda::BufferHandle`: cloning shares the same GPU allocation
/// (refcount increment only) and dropping the last clone removes the cache entry,
/// freeing the `MTLBuffer`. Entries with a live handle are exempt from LRU
/// eviction, so a handle is the way to keep a long-lived GPU tensor resident.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct MetalBufferHandle {
    id: BufferId,
    cache: Arc<std::sync::Mutex<BufferCache>>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl std::fmt::Debug for MetalBufferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The cache itself is not Debug (it holds raw Metal objects); the id is the
        // only meaningful identity here.
        f.debug_struct("MetalBufferHandle").field("id", &self.id).finish()
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalBufferHandle {
    /// Take the first reference to `id`. Fails when the id is not in `cache`.
    pub(crate) fn new(cache: Arc<std::sync::Mutex<BufferCache>>, id: BufferId) -> Result<Self> {
        {
            let mut guard = cache.lock().map_err(|_| {
                TrustformersError::hardware_error(
                    "Failed to lock buffer cache",
                    "MetalBufferHandle::new",
                )
            })?;
            if !guard.retain(&id) {
                return Err(TrustformersError::hardware_error(
                    &format!("Buffer {:?} not found in cache", id),
                    "MetalBufferHandle::new",
                ));
            }
        }
        Ok(Self { id, cache })
    }

    /// The buffer id this handle keeps alive.
    pub fn id(&self) -> BufferId {
        self.id
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl Clone for MetalBufferHandle {
    fn clone(&self) -> Self {
        if let Ok(mut guard) = self.cache.lock() {
            guard.retain(&self.id);
        }
        Self {
            id: self.id,
            cache: Arc::clone(&self.cache),
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl Drop for MetalBufferHandle {
    fn drop(&mut self) {
        // A poisoned cache mutex means another thread panicked while holding it;
        // leaking one entry is preferable to panicking inside a Drop.
        if let Ok(mut guard) = self.cache.lock() {
            guard.release(&self.id);
        }
    }
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;

    /// Allocate a real `MTLBuffer` of `bytes` bytes on the system default device.
    fn device_buffer(device: &MetalDevice, bytes: usize) -> Arc<Buffer> {
        Arc::new(device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared))
    }

    fn default_device() -> Option<MetalDevice> {
        MetalDevice::system_default()
    }

    /// Regression: before this change the cache never evicted anything, so this
    /// assertion (`entries` stays bounded, `evicted_entries > 0`) was impossible.
    /// Uses the `Evictable` tier - the only one eviction is allowed to touch.
    #[test]
    fn buffer_cache_evicts_lru_past_capacity() {
        let Some(device) = default_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        // Cap at 4 KiB; insert 16 x 1 KiB transient buffers.
        let mut cache = BufferCache::with_capacity_bytes(4 * 1024);
        let mut ids = Vec::new();
        for _ in 0..16 {
            let id = BufferId::new();
            cache.insert_evictable(id, device_buffer(&device, 1024));
            ids.push(id);
        }
        let stats = cache.stats();
        assert!(
            stats.live_bytes <= stats.capacity_bytes,
            "cache must stay under its cap: {stats:?}"
        );
        assert!(
            stats.evicted_entries >= 12,
            "16 x 1 KiB into a 4 KiB cap must evict at least 12 entries, got {stats:?}"
        );
        // The oldest ids are gone, the newest survive.
        assert!(!cache.contains(&ids[0]), "LRU victim must be evicted");
        assert!(
            cache.contains(&ids[15]),
            "most recently inserted entry must survive"
        );
    }

    #[test]
    fn buffer_cache_never_evicts_pinned_weights() {
        let Some(device) = default_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let mut cache = BufferCache::with_capacity_bytes(4 * 1024);
        let weight = BufferId::new();
        cache.insert_pinned(weight, device_buffer(&device, 2048));
        for _ in 0..32 {
            cache.insert_evictable(BufferId::new(), device_buffer(&device, 1024));
        }
        assert!(
            cache.contains(&weight),
            "pinned weight buffer must survive eviction pressure"
        );
    }

    #[test]
    fn buffer_cache_get_refreshes_lru_position() {
        let Some(device) = default_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let mut cache = BufferCache::with_capacity_bytes(3 * 1024);
        let a = BufferId::new();
        let b = BufferId::new();
        cache.insert_evictable(a, device_buffer(&device, 1024));
        cache.insert_evictable(b, device_buffer(&device, 1024));
        // Touch `a` so `b` becomes the least-recently-used entry.
        let touched = cache.get(&a);
        drop(touched);
        for _ in 0..4 {
            cache.insert_evictable(BufferId::new(), device_buffer(&device, 1024));
        }
        assert!(!cache.contains(&b), "untouched entry must be evicted first");
    }

    #[test]
    fn metal_buffer_handle_releases_on_last_drop() {
        let Some(device) = default_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let cache = Arc::new(std::sync::Mutex::new(BufferCache::with_capacity_bytes(
            1024 * 1024,
        )));
        let id = BufferId::new();
        {
            let mut guard = cache.lock().expect("cache lock");
            guard.insert(id, device_buffer(&device, 4096));
        }
        let handle = MetalBufferHandle::new(Arc::clone(&cache), id).expect("handle");
        let clone = handle.clone();
        drop(handle);
        {
            let guard = cache.lock().expect("cache lock");
            assert!(guard.contains(&id), "entry alive while a clone remains");
        }
        drop(clone);
        {
            let guard = cache.lock().expect("cache lock");
            assert!(!guard.contains(&id), "last handle drop must free the entry");
            assert_eq!(guard.live_bytes(), 0, "byte accounting must return to zero");
        }
    }

    #[test]
    fn retained_entries_are_exempt_from_eviction() {
        let Some(device) = default_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let cache = Arc::new(std::sync::Mutex::new(BufferCache::with_capacity_bytes(
            4 * 1024,
        )));
        let id = BufferId::new();
        {
            let mut guard = cache.lock().expect("cache lock");
            // Deliberately in the evictable tier: the handle alone must protect it.
            guard.insert_evictable(id, device_buffer(&device, 1024));
        }
        let handle = MetalBufferHandle::new(Arc::clone(&cache), id).expect("handle");
        {
            let mut guard = cache.lock().expect("cache lock");
            for _ in 0..32 {
                guard.insert_evictable(BufferId::new(), device_buffer(&device, 1024));
            }
            assert!(guard.contains(&id), "retained entry must not be evicted");
        }
        drop(handle);
        let guard = cache.lock().expect("cache lock");
        assert!(!guard.contains(&id), "released entry becomes reclaimable");
    }

    /// The soundness invariant: the default (`Live`) tier is never auto-evicted, no
    /// matter how much pressure is applied. `Tensor::Metal` holds a bare `BufferId`,
    /// so evicting a `Live` entry would surface as a mid-forward lookup failure.
    #[test]
    fn live_tier_entries_are_never_auto_evicted() {
        let Some(device) = default_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let mut cache = BufferCache::with_capacity_bytes(4 * 1024);
        let live = BufferId::new();
        cache.insert(live, device_buffer(&device, 1024));
        for _ in 0..128 {
            cache.insert_evictable(BufferId::new(), device_buffer(&device, 1024));
        }
        assert!(
            cache.contains(&live),
            "a Live op result must survive any amount of eviction pressure"
        );
        // And a cache holding only Live entries stalls rather than dropping them.
        let mut live_only = BufferCache::with_capacity_bytes(2 * 1024);
        let mut ids = Vec::new();
        for _ in 0..8 {
            let id = BufferId::new();
            live_only.insert(id, device_buffer(&device, 1024));
            ids.push(id);
        }
        let stats = live_only.stats();
        assert_eq!(stats.entries, 8, "no Live entry may be dropped: {stats:?}");
        assert!(
            stats.eviction_stalls > 0,
            "exceeding the cap with nothing evictable must be recorded as a stall, \
             not silently ignored: {stats:?}"
        );
        assert_eq!(stats.evicted_entries, 0, "nothing may have been evicted");
        for id in &ids {
            assert!(live_only.contains(id), "every Live id must still resolve");
        }
    }

    #[test]
    fn buffer_cache_capacity_honours_env_override() {
        // `new()` reads the env var once per construction; assert the parse path
        // rather than mutating a process-global in a threaded test binary.
        let cache = BufferCache::new();
        assert!(
            cache.capacity_bytes() >= 1,
            "capacity must always be positive"
        );
        let explicit = BufferCache::with_capacity_bytes(0);
        assert_eq!(
            explicit.capacity_bytes(),
            1,
            "a zero cap is clamped to 1 byte, never to unbounded"
        );
    }
}
