//! Export utilities for profiling and trace data.
//!
//! # Modules
//!
//! - [`perfetto`] — Chrome/Perfetto trace-event JSON format.
//! - [`tracy`] — Tracy offline CSV format.
//! - [`unified`] — Unified exporter, `CsvExporter`, `JsonExporter`, and shared
//!   types (`TimingEvent`, `ExportFormat`, `ExportConfig`, `ProfilingTrace`).

pub mod perfetto;
pub mod tracy;
pub mod unified;

pub use perfetto::{PerfettoEvent, PerfettoExporter, PerfettoPhase, PerfettoTrace};
pub use tracy::{TracyExporter, TracyTrace, TracyZone};
pub use unified::{
    CsvExporter,
    ExportConfig,
    ExportError,
    ExportFormat,
    JsonExporter,
    ProfilingTrace,
    TimingEvent,
    TraceExporter,
};
