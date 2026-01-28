//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[allow(unused_imports)]
use super::common::*;

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
}
/// Persistent buffer cache for Metal GPU
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) struct BufferCache {
    buffers: HashMap<BufferId, Arc<Buffer>>,
}
#[cfg(all(target_os = "macos", feature = "metal"))]
impl BufferCache {
    pub(crate) fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }
    pub(crate) fn insert(&mut self, id: BufferId, buffer: Arc<Buffer>) {
        self.buffers.insert(id, buffer);
    }
    pub(crate) fn get(&self, id: &BufferId) -> Option<Arc<Buffer>> {
        self.buffers.get(id).cloned()
    }
    pub(crate) fn remove(&mut self, id: &BufferId) -> Option<Arc<Buffer>> {
        self.buffers.remove(id)
    }
    pub(crate) fn clear(&mut self) {
        self.buffers.clear();
    }
    pub(crate) fn len(&self) -> usize {
        self.buffers.len()
    }
}
