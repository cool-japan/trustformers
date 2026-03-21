//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{VecDeque, HashMap};

/// Comprehensive data export and visualization management
///
/// Handles exporting profiling data in multiple formats with advanced
/// visualization capabilities and customizable reporting.
#[derive(Debug)]
pub struct ProfilerExportManager {
    /// Export configuration
    config: ExportManagerConfig,
    /// Export format handlers
    formatters: HashMap<ExportFormat, Box<dyn DataFormatter + Send + Sync>>,
    /// Export history tracking
    export_history: VecDeque<ExportRecord>,
    /// Pending export tasks
    pending_exports: VecDeque<ExportTask>,
    /// Visualization engine
    visualization_engine: VisualizationEngine,
    /// Export statistics
    export_stats: ExportManagerStats,
}
