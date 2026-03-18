//! # HistoricalDataManager - Trait Implementations
//!
//! This module contains trait implementations for `HistoricalDataManager`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::HistoricalDataManager;

impl fmt::Debug for HistoricalDataManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HistoricalDataManager")
            .field("config", &self.config)
            .field("time_series_store", &self.time_series_store)
            .field("retention_manager", &self.retention_manager)
            .field("compression_engine", &self.compression_engine)
            .field("archival_system", &self.archival_system)
            .field("data_lifecycle_manager", &self.data_lifecycle_manager)
            .field("query_engine", &self.query_engine)
            .field("data_statistics", &self.data_statistics)
            .field("storage_backend_count", &self.storage_backends.len())
            .finish()
    }
}
