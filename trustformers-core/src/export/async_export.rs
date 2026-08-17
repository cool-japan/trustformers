//! Async export driver for large models.
//!
//! Exports run on a blocking task so the caller's runtime stays responsive, with
//! step-level progress, cancellation before the write begins, and a result that
//! reports the real size of the file produced.
//!
//! # Progress is step-level, not byte-level
//!
//! [`ModelExporter`] has no progress callback: an export is one blocking call.
//! This module therefore reports the step it is on and two measured byte figures —
//! the model's real parameter footprint before the write, and the real file size
//! after it. It does not interpolate a per-layer byte counter, because nothing here
//! can observe one.

use crate::export::*;
use crate::traits::Model;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;

/// Progress information for async export operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportProgress {
    /// Current step in the export process
    pub current_step: ExportStep,
    /// Overall progress percentage (0-100)
    pub progress_percentage: f64,
    /// Current operation being performed
    pub current_operation: String,
    /// Estimated time remaining in seconds
    pub estimated_time_remaining_secs: Option<u64>,
    /// Number of bytes processed
    pub bytes_processed: u64,
    /// Total bytes to process (if known)
    pub total_bytes: Option<u64>,
    /// Export speed in bytes per second
    pub speed_bytes_per_sec: Option<f64>,
    /// Elapsed time since export started
    pub elapsed_time_secs: u64,
}

/// Steps in the export process
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportStep {
    Initializing,
    ValidatingModel,
    OptimizingModel,
    ConvertingWeights,
    ApplyingQuantization,
    GeneratingMetadata,
    WritingOutput,
    Finalizing,
    Completed,
    Failed,
}

/// Async export handle that allows monitoring and controlling export operations
pub struct AsyncExportHandle {
    task_handle: JoinHandle<Result<ExportResult>>,
    progress_receiver: mpsc::Receiver<ExportProgress>,
    cancel_sender: mpsc::Sender<()>,
    export_id: String,
}

/// Async export manager for handling multiple concurrent exports
pub struct AsyncExportManager {
    active_exports: Arc<RwLock<std::collections::HashMap<String, AsyncExportInfo>>>,
    max_concurrent_exports: usize,
}

/// Information about an active export
#[derive(Debug, Clone)]
struct AsyncExportInfo {
    #[allow(dead_code)]
    export_id: String,
    #[allow(dead_code)]
    config: ExportConfig,
    #[allow(dead_code)]
    start_time: Instant,
    current_progress: ExportProgress,
    cancel_sender: mpsc::Sender<()>,
}

/// Controller for managing the export process
struct ExportController {
    progress_sender: mpsc::Sender<ExportProgress>,
    cancel_receiver: mpsc::Receiver<()>,
    start_time: Instant,
    bytes_processed: Arc<AtomicU64>,
    is_cancelled: Arc<AtomicBool>,
}

impl AsyncExportHandle {
    /// Wait for the export to complete
    pub async fn wait(self) -> Result<ExportResult> {
        self.task_handle.await?
    }

    /// Get the current progress without waiting for completion
    pub async fn get_progress(&mut self) -> Option<ExportProgress> {
        self.progress_receiver.try_recv().ok()
    }

    /// Cancel the export operation
    pub async fn cancel(&self) -> Result<()> {
        self.cancel_sender
            .send(())
            .await
            .map_err(|_| anyhow!("Failed to send cancel signal"))?;
        Ok(())
    }

    /// Get the export ID
    pub fn export_id(&self) -> &str {
        &self.export_id
    }
}

impl AsyncExportManager {
    /// Create a new async export manager
    pub fn new(max_concurrent_exports: usize) -> Self {
        Self {
            active_exports: Arc::new(RwLock::new(std::collections::HashMap::new())),
            max_concurrent_exports,
        }
    }

    /// Start an async export operation
    pub async fn export_async<M: Model + Send + Sync + 'static>(
        &self,
        model: Arc<M>,
        config: ExportConfig,
        exporter: ConcreteExporter,
    ) -> Result<AsyncExportHandle> {
        // Check if we've reached the maximum number of concurrent exports
        let active_count = self.active_exports.read().await.len();
        if active_count >= self.max_concurrent_exports {
            return Err(anyhow!(
                "Maximum number of concurrent exports ({}) reached",
                self.max_concurrent_exports
            ));
        }

        let export_id = format!("export_{}", uuid::Uuid::new_v4());
        let (progress_tx, progress_rx) = mpsc::channel(100);
        let (cancel_tx, cancel_rx) = mpsc::channel(1);

        let controller = ExportController {
            progress_sender: progress_tx.clone(),
            cancel_receiver: cancel_rx,
            start_time: Instant::now(),
            bytes_processed: Arc::new(AtomicU64::new(0)),
            is_cancelled: Arc::new(AtomicBool::new(false)),
        };

        // Add to active exports
        let export_info = AsyncExportInfo {
            export_id: export_id.clone(),
            config: config.clone(),
            start_time: Instant::now(),
            current_progress: ExportProgress {
                current_step: ExportStep::Initializing,
                progress_percentage: 0.0,
                current_operation: "Starting export".to_string(),
                estimated_time_remaining_secs: None,
                bytes_processed: 0,
                total_bytes: None,
                speed_bytes_per_sec: None,
                elapsed_time_secs: 0,
            },
            cancel_sender: cancel_tx.clone(),
        };

        self.active_exports.write().await.insert(export_id.clone(), export_info);

        // Spawn the export task
        let active_exports = self.active_exports.clone();
        let export_id_for_task = export_id.clone();

        let task_handle = tokio::spawn(async move {
            let result = Self::run_export_with_progress(model, config, exporter, controller).await;

            // Remove from active exports when done
            active_exports.write().await.remove(&export_id_for_task);

            result
        });

        Ok(AsyncExportHandle {
            task_handle,
            progress_receiver: progress_rx,
            cancel_sender: cancel_tx,
            export_id,
        })
    }

    /// Run the export, reporting progress that reflects work actually done.
    ///
    /// The steps below correspond to real operations. There is deliberately no
    /// per-layer byte counter: [`ModelExporter`] is a single blocking call with no
    /// progress callback, so the only byte figures reported are the model's real
    /// parameter footprint (known before the write) and the real size of the file
    /// on disk afterwards.
    ///
    /// A previous revision slept for four seconds while emitting
    /// `"Converting layer i/100"` and `bytes_processed = (i + 1) * 100_000` for
    /// every model, whatever its size.
    async fn run_export_with_progress<M: Model + Send + Sync + 'static>(
        model: Arc<M>,
        config: ExportConfig,
        exporter: ConcreteExporter,
        mut controller: ExportController,
    ) -> Result<ExportResult> {
        let start_time = Instant::now();

        // Step 1: validation — the exporter's own check, plus the model's real
        // parameter footprint, which is what the export has to write.
        controller
            .update_progress(
                ExportStep::ValidatingModel,
                10.0,
                "Validating model compatibility",
                None,
            )
            .await?;

        if controller.check_cancelled().await {
            return Err(anyhow!("Export cancelled during validation"));
        }

        exporter.validate_model(model.as_ref(), config.format)?;

        let parameter_bytes: u64 =
            model.named_tensors().iter().map(|(_, tensor)| tensor.size_bytes() as u64).sum();
        let total_bytes = (parameter_bytes > 0).then_some(parameter_bytes);

        // Step 2: the export itself. It is a single blocking call, so it cannot be
        // interrupted once started; the last chance to cancel is here.
        controller
            .update_progress(
                ExportStep::WritingOutput,
                20.0,
                "Writing output file",
                total_bytes,
            )
            .await?;

        if controller.check_cancelled().await {
            return Err(anyhow!("Export cancelled before writing"));
        }

        let format = config.format;
        let output_path = config.output_path.clone();
        let export_config = config.clone();

        let export_outcome =
            tokio::task::spawn_blocking(move || exporter.export(model.as_ref(), &export_config))
                .await?;

        if let Err(error) = export_outcome {
            let _ = controller
                .update_progress(
                    ExportStep::Failed,
                    100.0,
                    &format!("Export failed: {error}"),
                    total_bytes,
                )
                .await;
            return Err(error);
        }

        // Step 3: measure what was actually written.
        let output_size_bytes = measure_output_bytes(&config)?;
        controller.bytes_processed.store(output_size_bytes, Ordering::Relaxed);

        controller
            .update_progress(
                ExportStep::Completed,
                100.0,
                "Export completed",
                total_bytes,
            )
            .await?;

        let elapsed = start_time.elapsed();

        Ok(ExportResult {
            format,
            output_path,
            // This wrapper applies no optimization passes of its own; whatever the
            // exporter did is the exporter's business to report.
            optimizations_applied: Vec::new(),
            export_time_ms: elapsed.as_millis() as u64,
            output_size_bytes,
        })
    }

    /// Get progress for a specific export
    pub async fn get_export_progress(&self, export_id: &str) -> Option<ExportProgress> {
        self.active_exports
            .read()
            .await
            .get(export_id)
            .map(|info| info.current_progress.clone())
    }

    /// Get all active exports
    pub async fn get_active_exports(&self) -> Vec<String> {
        self.active_exports.read().await.keys().cloned().collect()
    }

    /// Cancel an export by ID
    pub async fn cancel_export(&self, export_id: &str) -> Result<()> {
        let exports = self.active_exports.read().await;
        if let Some(export_info) = exports.get(export_id) {
            export_info
                .cancel_sender
                .send(())
                .await
                .map_err(|_| anyhow!("Failed to send cancel signal"))?;
            Ok(())
        } else {
            Err(anyhow!("Export with ID {} not found", export_id))
        }
    }

    /// Cancel all active exports
    pub async fn cancel_all_exports(&self) -> Result<()> {
        let exports = self.active_exports.read().await;
        for export_info in exports.values() {
            let _ = export_info.cancel_sender.send(()).await;
        }
        Ok(())
    }
}

impl ExportController {
    /// Update progress and send to receiver
    async fn update_progress(
        &self,
        step: ExportStep,
        percentage: f64,
        operation: &str,
        total_bytes: Option<u64>,
    ) -> Result<()> {
        let elapsed = self.start_time.elapsed();
        let bytes_processed = self.bytes_processed.load(Ordering::Relaxed);

        let speed = if elapsed.as_secs() > 0 {
            Some(bytes_processed as f64 / elapsed.as_secs_f64())
        } else {
            None
        };

        let eta = if let (Some(total), Some(speed_val)) = (total_bytes, speed) {
            if speed_val > 0.0 {
                let remaining_bytes = total.saturating_sub(bytes_processed) as f64;
                Some((remaining_bytes / speed_val) as u64)
            } else {
                None
            }
        } else {
            None
        };

        let progress = ExportProgress {
            current_step: step,
            progress_percentage: percentage,
            current_operation: operation.to_string(),
            estimated_time_remaining_secs: eta,
            bytes_processed,
            total_bytes,
            speed_bytes_per_sec: speed,
            elapsed_time_secs: elapsed.as_secs(),
        };

        self.progress_sender
            .send(progress)
            .await
            .map_err(|_| anyhow!("Failed to send progress update"))?;

        Ok(())
    }

    /// Check if export has been cancelled
    async fn check_cancelled(&mut self) -> bool {
        if self.is_cancelled.load(Ordering::Relaxed) {
            return true;
        }

        if self.cancel_receiver.try_recv().is_ok() {
            self.is_cancelled.store(true, Ordering::Relaxed);
            return true;
        }

        false
    }
}

/// Locate the file an exporter wrote and return its real size.
///
/// Exporters append their format's extension to `output_path`, so both the bare
/// path and the extended one are checked. When neither exists the export claimed a
/// success it did not deliver, and that is reported rather than papered over with
/// a made-up byte count.
fn measure_output_bytes(config: &ExportConfig) -> Result<u64> {
    let extension = match config.format {
        ExportFormat::ONNX => "onnx",
        ExportFormat::GGML => "ggml",
        ExportFormat::GGUF => "gguf",
        ExportFormat::NNEF => "nnef",
        ExportFormat::OpenVINO => "xml",
        ExportFormat::TensorRT => "plan",
        ExportFormat::TVM => "so",
        ExportFormat::CoreML => "mlmodel",
    };

    let candidates = [
        std::path::PathBuf::from(&config.output_path),
        std::path::PathBuf::from(format!("{}.{}", config.output_path, extension)),
    ];

    for candidate in &candidates {
        if let Ok(metadata) = std::fs::metadata(candidate) {
            if metadata.is_file() {
                return Ok(metadata.len());
            }
        }
    }

    Err(anyhow!(
        "the {:?} exporter reported success but no output file exists at {} or {}.{}",
        config.format,
        config.output_path,
        config.output_path,
        extension
    ))
}

impl ExportStep {
    /// Get a human-readable description of the export step
    pub fn description(&self) -> &'static str {
        match self {
            ExportStep::Initializing => "Initializing export process",
            ExportStep::ValidatingModel => "Validating model compatibility",
            ExportStep::OptimizingModel => "Optimizing model structure",
            ExportStep::ConvertingWeights => "Converting model weights",
            ExportStep::ApplyingQuantization => "Applying quantization",
            ExportStep::GeneratingMetadata => "Generating metadata",
            ExportStep::WritingOutput => "Writing output file",
            ExportStep::Finalizing => "Finalizing export",
            ExportStep::Completed => "Export completed",
            ExportStep::Failed => "Export failed",
        }
    }

    /// Get the expected duration range for this step (in seconds)
    pub fn expected_duration_range(&self) -> (u64, u64) {
        match self {
            ExportStep::Initializing => (1, 5),
            ExportStep::ValidatingModel => (2, 10),
            ExportStep::OptimizingModel => (5, 30),
            ExportStep::ConvertingWeights => (10, 300),
            ExportStep::ApplyingQuantization => (20, 120),
            ExportStep::GeneratingMetadata => (1, 10),
            ExportStep::WritingOutput => (5, 60),
            ExportStep::Finalizing => (1, 5),
            ExportStep::Completed => (0, 0),
            ExportStep::Failed => (0, 0),
        }
    }
}

/// Convenience function to export a model asynchronously
pub async fn export_model_async<M: Model + Send + Sync + 'static>(
    model: Arc<M>,
    config: ExportConfig,
    exporter: ConcreteExporter,
) -> Result<AsyncExportHandle> {
    let manager = AsyncExportManager::new(1);
    manager.export_async(model, config, exporter).await
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::export::test_support::TestModel;

    #[tokio::test]
    async fn test_async_export_manager_creation() {
        let manager = AsyncExportManager::new(3);
        assert_eq!(manager.max_concurrent_exports, 3);

        let active = manager.get_active_exports().await;
        assert!(active.is_empty());
    }

    #[tokio::test]
    async fn test_export_steps() {
        assert_eq!(
            ExportStep::Initializing.description(),
            "Initializing export process"
        );
        assert_eq!(ExportStep::Completed.description(), "Export completed");

        let (min, max) = ExportStep::ConvertingWeights.expected_duration_range();
        assert!(min <= max);
        assert!(min > 0);
    }

    #[test]
    fn test_export_progress_serialization() {
        let progress = ExportProgress {
            current_step: ExportStep::ConvertingWeights,
            progress_percentage: 50.0,
            current_operation: "Test operation".to_string(),
            estimated_time_remaining_secs: Some(120),
            bytes_processed: 1000000,
            total_bytes: Some(2000000),
            speed_bytes_per_sec: Some(8333.33),
            elapsed_time_secs: 120,
        };

        let serialized = serde_json::to_string(&progress).expect("JSON serialization failed");
        let deserialized: ExportProgress =
            serde_json::from_str(&serialized).expect("JSON deserialization failed");

        assert_eq!(deserialized.progress_percentage, 50.0);
        assert_eq!(deserialized.bytes_processed, 1000000);
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// Regression test: progress used to report `(i + 1) * 100_000` bytes towards a
    /// hard-coded 10 MB total for every model, and slept four seconds doing it.
    #[tokio::test]
    async fn progress_reports_measured_bytes_not_invented_ones() {
        let dir = temp_dir("trustformers_async_export_real");
        let output = dir.join("model");

        let model = Arc::new(TestModel::with_seed(1.0));
        let expected_parameter_bytes: u64 =
            model.named_tensors().iter().map(|(_, tensor)| tensor.size_bytes() as u64).sum();

        let config = ExportConfig {
            format: ExportFormat::GGUF,
            output_path: output.to_string_lossy().to_string(),
            ..Default::default()
        };

        let started = Instant::now();
        let mut handle = export_model_async(
            model,
            config,
            ConcreteExporter::GGUF(crate::export::gguf::GGUFExporter::new()),
        )
        .await
        .expect("start export");

        let mut updates = Vec::new();
        while let Some(progress) = handle.get_progress().await {
            updates.push(progress);
        }

        let result = handle.wait().await.expect("export");
        let wall_clock = started.elapsed();

        // The old implementation slept for at least 2.2 seconds unconditionally.
        assert!(
            wall_clock < std::time::Duration::from_secs(2),
            "a tiny model must not take {wall_clock:?} to export"
        );

        let real_size =
            std::fs::metadata(output.with_extension("gguf")).expect("output file").len();
        assert_eq!(result.output_size_bytes, real_size);
        assert!(result.optimizations_applied.is_empty());

        for progress in &updates {
            assert_ne!(
                progress.bytes_processed, 100_000,
                "byte counts must not come from a fixed schedule"
            );
            if let Some(total) = progress.total_bytes {
                assert_eq!(
                    total, expected_parameter_bytes,
                    "the declared total must be the model's real parameter footprint"
                );
            }
            assert!(
                !progress.current_operation.contains("/100"),
                "no invented per-layer counter: {}",
                progress.current_operation
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A failing exporter must surface its error rather than report a completed export.
    #[tokio::test]
    async fn a_failing_export_is_reported_as_a_failure() {
        let dir = temp_dir("trustformers_async_export_fail");
        let output = dir.join("model");

        let config = ExportConfig {
            format: ExportFormat::TensorRT,
            output_path: output.to_string_lossy().to_string(),
            ..Default::default()
        };

        let handle = export_model_async(
            Arc::new(TestModel::with_seed(1.0)),
            config,
            ConcreteExporter::TensorRT(crate::export::tensorrt::TensorRTExporter::new()),
        )
        .await
        .expect("start export");

        let err = handle.wait().await.expect_err("TensorRT export cannot succeed");
        assert!(err.to_string().contains("Unsupported operation"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A model with no weights must fail validation before anything is written.
    #[tokio::test]
    async fn a_model_without_weights_fails_validation() {
        let dir = temp_dir("trustformers_async_export_empty");
        let output = dir.join("model");

        let config = ExportConfig {
            format: ExportFormat::GGUF,
            output_path: output.to_string_lossy().to_string(),
            ..Default::default()
        };

        let handle = export_model_async(
            Arc::new(TestModel::empty()),
            config,
            ConcreteExporter::GGUF(crate::export::gguf::GGUFExporter::new()),
        )
        .await
        .expect("start export");

        let err = handle.wait().await.expect_err("no weights, no export");
        assert!(err.to_string().contains("named_tensors"), "{err}");
        assert!(!output.with_extension("gguf").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
