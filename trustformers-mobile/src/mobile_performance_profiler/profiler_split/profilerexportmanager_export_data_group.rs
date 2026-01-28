//! # ProfilerExportManager - export_data_group Methods
//!
//! This module contains method implementations for `ProfilerExportManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, HashSet};
use super::profilerexportmanager_type::ProfilerExportManager;

impl ProfilerExportManager {
    fn export_data(&self, data: &ProfilingData) -> Result<String> {
        let timestamp = chrono::Utc::now().timestamp();
        let export_path = format!("/tmp/claude/profiling_export_{}.json", timestamp);
        let json_data = serde_json::to_string_pretty(data)
            .context("Failed to serialize profiling data")?;
        std::fs::create_dir_all("/tmp/claude")
            .context("Failed to create export directory")?;
        std::fs::write(&export_path, json_data).context("Failed to write export file")?;
        info!("Profiling data exported to: {}", export_path);
        Ok(export_path)
    }
}
