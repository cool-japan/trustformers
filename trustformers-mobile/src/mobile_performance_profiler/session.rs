//! Session management and profiling lifecycle.

pub use super::profiler::ProfilingSession;
pub use super::types::{EventData, EventType, ProfilingEvent, SessionMetadata};

/// Session information
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub session_id: String,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub duration_ms: Option<u64>,
    pub total_events: usize,
}
