//! Async Checkpointing
//!
//! Non-blocking model checkpoint saving — training continues while checkpoint
//! serializes in background thread.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

// ─── Data structures ────────────────────────────────────────────────────────

/// Bytes occupied by one stored weight value (`f32`).
const WEIGHT_VALUE_BYTES: usize = std::mem::size_of::<f32>();

/// Leading bytes of a [`SerializeFormat::Binary`] checkpoint file.
///
/// [`AsyncCheckpointer::load`] sniffs this so a checkpoint can be read back without the
/// caller having to remember which format it was written in.
const BINARY_MAGIC: [u8; 8] = *b"TFCKPT01";

/// What to checkpoint
///
/// Weights and optimizer state are stored as `f32` — the precision model parameters
/// actually have. Widening them to `f64` on the way to disk doubled the checkpoint size
/// without adding a single bit of information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointData {
    pub step: u64,
    pub epoch: u32,
    /// layer_name -> flat weight vector
    pub weights: HashMap<String, Vec<f32>>,
    pub optimizer_state: HashMap<String, Vec<f32>>,
    /// loss, accuracy, etc.
    pub metrics: HashMap<String, f64>,
    pub config: serde_json::Value,
}

impl CheckpointData {
    pub fn new(step: u64, epoch: u32) -> Self {
        Self {
            step,
            epoch,
            weights: HashMap::new(),
            optimizer_state: HashMap::new(),
            metrics: HashMap::new(),
            config: serde_json::Value::Null,
        }
    }

    pub fn add_weight(&mut self, name: impl Into<String>, weights: Vec<f32>) {
        self.weights.insert(name.into(), weights);
    }

    pub fn add_metric(&mut self, name: impl Into<String>, value: f64) {
        self.metrics.insert(name.into(), value);
    }

    /// Estimate the bulk footprint in bytes (4 bytes per stored `f32`).
    pub fn estimated_size_bytes(&self) -> usize {
        let weight_bytes: usize = self.weights.values().map(|v| v.len() * WEIGHT_VALUE_BYTES).sum();
        let opt_bytes: usize =
            self.optimizer_state.values().map(|v| v.len() * WEIGHT_VALUE_BYTES).sum();
        weight_bytes + opt_bytes
    }
}

/// Wire form of [`CheckpointData`] for the non-self-describing binary format.
///
/// `oxicode` (like every bincode-shaped codec) cannot decode a `serde_json::Value`: the
/// `Value` deserializer calls `deserialize_any`, which a format without type tags cannot
/// answer. The free-form `config` is therefore carried as its JSON text and re-parsed on
/// load; everything else round-trips as native binary.
#[derive(Debug, Serialize, Deserialize)]
struct BinaryCheckpoint {
    step: u64,
    epoch: u32,
    weights: HashMap<String, Vec<f32>>,
    optimizer_state: HashMap<String, Vec<f32>>,
    metrics: HashMap<String, f64>,
    config_json: String,
}

impl BinaryCheckpoint {
    fn from_data(data: &CheckpointData) -> Result<Self, CheckpointError> {
        Ok(Self {
            step: data.step,
            epoch: data.epoch,
            weights: data.weights.clone(),
            optimizer_state: data.optimizer_state.clone(),
            metrics: data.metrics.clone(),
            config_json: serde_json::to_string(&data.config)
                .map_err(|e| CheckpointError::Serialization(e.to_string()))?,
        })
    }

    fn into_data(self) -> Result<CheckpointData, CheckpointError> {
        let config: serde_json::Value = serde_json::from_str(&self.config_json)
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;
        Ok(CheckpointData {
            step: self.step,
            epoch: self.epoch,
            weights: self.weights,
            optimizer_state: self.optimizer_state,
            metrics: self.metrics,
            config,
        })
    }
}

// ─── Configuration ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum MetricMode {
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SerializeFormat {
    /// serde_json human-readable (pretty-printed). Debug format only — a real checkpoint
    /// written this way is roughly an order of magnitude larger than the model.
    Json,
    /// serde_json compact. Still decimal text for every weight.
    Compact,
    /// `oxicode` binary — 4 bytes per weight, streamed straight to the file. Default.
    Binary,
}

impl SerializeFormat {
    /// File extension used for checkpoints written in this format.
    fn extension(self) -> &'static str {
        match self {
            SerializeFormat::Json | SerializeFormat::Compact => "json",
            SerializeFormat::Binary => "ckpt",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AsyncCheckpointConfig {
    pub checkpoint_dir: PathBuf,
    /// Delete old checkpoints, keeping at most this many
    pub max_checkpoints_to_keep: usize,
    /// Checkpoint every N steps
    pub save_interval_steps: u64,
    /// Only checkpoint when metric improves
    pub save_best_only: bool,
    /// e.g. "val_loss"
    pub best_metric_name: String,
    pub best_metric_mode: MetricMode,
    pub serialize_format: SerializeFormat,
}

impl Default for AsyncCheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: std::env::temp_dir().join("trustformers_checkpoints"),
            max_checkpoints_to_keep: 3,
            save_interval_steps: 100,
            save_best_only: false,
            best_metric_name: "val_loss".to_string(),
            best_metric_mode: MetricMode::Min,
            serialize_format: SerializeFormat::Binary,
        }
    }
}

// ─── Handle ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CheckpointHandle {
    pub step: u64,
    pub path: PathBuf,
    pub is_complete: bool,
    pub error: Option<String>,
}

// ─── Checkpointer ───────────────────────────────────────────────────────────

pub struct AsyncCheckpointer {
    config: AsyncCheckpointConfig,
    /// Handles for in-flight background saves
    pending_handles: Arc<Mutex<Vec<Arc<Mutex<CheckpointHandle>>>>>,
    /// (step, path) ordered history of completed saves
    saved_paths: Arc<Mutex<Vec<(u64, PathBuf)>>>,
    /// Best tracked metric together with the step that produced it, so
    /// [`AsyncCheckpointer::best_checkpoint`] can name the right file instead of guessing.
    best_metric: Arc<Mutex<Option<BestMetric>>>,
}

/// The best value seen for [`AsyncCheckpointConfig::best_metric_name`] and its step.
#[derive(Debug, Clone, Copy, PartialEq)]
struct BestMetric {
    value: f64,
    step: u64,
}

impl AsyncCheckpointer {
    pub fn new(config: AsyncCheckpointConfig) -> Result<Self, CheckpointError> {
        std::fs::create_dir_all(&config.checkpoint_dir)?;
        Ok(Self {
            config,
            pending_handles: Arc::new(Mutex::new(Vec::new())),
            saved_paths: Arc::new(Mutex::new(Vec::new())),
            best_metric: Arc::new(Mutex::new(None)),
        })
    }

    /// Non-blocking: spawn a background thread to write the checkpoint.
    /// Returns immediately with a handle the caller can poll.
    pub fn save_async(
        &self,
        data: CheckpointData,
    ) -> Result<Arc<Mutex<CheckpointHandle>>, CheckpointError> {
        let path = self.checkpoint_path(data.step);
        let handle = Arc::new(Mutex::new(CheckpointHandle {
            step: data.step,
            path: path.clone(),
            is_complete: false,
            error: None,
        }));

        let handle_clone = Arc::clone(&handle);
        let saved_paths_clone = Arc::clone(&self.saved_paths);
        let best_metric_clone = Arc::clone(&self.best_metric);
        let format = self.config.serialize_format;
        let metric_name = self.config.best_metric_name.clone();
        let metric_mode = self.config.best_metric_mode.clone();

        thread::spawn(move || {
            let result = Self::write_to_disk(&data, &path, format);
            let mut h = match handle_clone.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            h.is_complete = true;
            match result {
                Ok(()) => {
                    if let Ok(mut sp) = saved_paths_clone.lock() {
                        sp.push((data.step, path));
                    }
                    // Background saves must feed the best-metric tracker too, otherwise
                    // `should_checkpoint(.., save_best_only)` would only ever see the
                    // checkpoints written synchronously.
                    if let Some(&value) = data.metrics.get(&metric_name) {
                        if let Ok(mut best) = best_metric_clone.lock() {
                            update_best(&mut best, value, data.step, &metric_mode);
                        }
                    }
                },
                Err(e) => {
                    h.error = Some(e.to_string());
                },
            }
        });

        {
            let mut pending = self
                .pending_handles
                .lock()
                .map_err(|e| CheckpointError::Thread(e.to_string()))?;
            pending.push(Arc::clone(&handle));
        }

        Ok(handle)
    }

    /// Blocking: wait for all pending checkpoints to complete.
    pub fn wait_all(&self) -> Result<(), CheckpointError> {
        // Spin-wait with short sleeps — adequate for testing; production would
        // use a condvar.
        loop {
            let all_done = {
                let pending = self
                    .pending_handles
                    .lock()
                    .map_err(|e| CheckpointError::Thread(e.to_string()))?;
                pending.iter().all(|h| h.lock().map(|g| g.is_complete).unwrap_or(false))
            };
            if all_done {
                break;
            }
            thread::sleep(std::time::Duration::from_millis(5));
        }

        // Propagate any errors from background threads
        let pending = self
            .pending_handles
            .lock()
            .map_err(|e| CheckpointError::Thread(e.to_string()))?;
        for h in pending.iter() {
            if let Ok(guard) = h.lock() {
                if let Some(ref err) = guard.error {
                    return Err(CheckpointError::Thread(err.clone()));
                }
            }
        }

        // Clean up old checkpoints now that all pending writes finished
        drop(pending);
        self.cleanup_old_checkpoints()?;
        Ok(())
    }

    /// Synchronous save (blocks until complete).
    pub fn save_sync(&self, data: &CheckpointData) -> Result<PathBuf, CheckpointError> {
        let path = self.checkpoint_path(data.step);
        Self::write_to_disk(data, &path, self.config.serialize_format)?;
        {
            let mut sp =
                self.saved_paths.lock().map_err(|e| CheckpointError::Thread(e.to_string()))?;
            sp.push((data.step, path.clone()));
        }
        // Update best metric tracking
        if let Some(&metric_value) = data.metrics.get(&self.config.best_metric_name) {
            let mut best =
                self.best_metric.lock().map_err(|e| CheckpointError::Thread(e.to_string()))?;
            update_best(
                &mut best,
                metric_value,
                data.step,
                &self.config.best_metric_mode,
            );
        }
        self.cleanup_old_checkpoints()?;
        Ok(path)
    }

    /// Load a checkpoint from `path`, in whichever format it was written.
    ///
    /// The format is detected from the file's leading bytes ([`BINARY_MAGIC`]), not from the
    /// extension or from any configuration, so a checkpointer configured for one format can
    /// still read files produced by another. The file is streamed through a `BufReader`
    /// rather than materialised as one `String`.
    pub fn load(path: &Path) -> Result<CheckpointData, CheckpointError> {
        if !path.exists() {
            return Err(CheckpointError::NotFound {
                path: path.to_path_buf(),
            });
        }
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; BINARY_MAGIC.len()];
        let read = read_up_to(&mut reader, &mut magic)?;
        if read == BINARY_MAGIC.len() && magic == BINARY_MAGIC {
            let (wire, _): (BinaryCheckpoint, usize) =
                oxicode::serde::decode_from_std_read(&mut reader, oxicode::config::standard())
                    .map_err(|e| CheckpointError::Serialization(e.to_string()))?;
            return wire.into_data();
        }

        // Not a binary checkpoint — rewind and parse it as JSON.
        reader.seek(SeekFrom::Start(0))?;
        serde_json::from_reader(reader).map_err(|e| CheckpointError::Serialization(e.to_string()))
    }

    /// Adopt the checkpoints already present in `checkpoint_dir`.
    ///
    /// `saved_paths` is in-memory, so a fresh process starts out believing no checkpoint
    /// exists. This scans the directory for `checkpoint_step_<step>.{json,ckpt}` — **both**
    /// extensions, so a directory written before the binary format became the default is
    /// still found — and merges what it finds into the tracked history, replacing any entry
    /// for the same step. Returns the discovered `(step, path)` pairs in step order.
    ///
    /// Best-metric tracking is *not* reconstructed: that would mean reading every
    /// checkpoint in full to recover one scalar. [`AsyncCheckpointer::best_checkpoint`]
    /// therefore reports `None` until this process saves a checkpoint of its own, rather
    /// than naming a file it has not actually compared.
    pub fn discover_existing(&self) -> Result<Vec<(u64, PathBuf)>, CheckpointError> {
        let mut discovered = Vec::new();
        for entry in std::fs::read_dir(&self.config.checkpoint_dir)? {
            let path = entry?.path();
            if !path.is_file() {
                continue;
            }
            let Some(step) = checkpoint_step_from_path(&path) else {
                continue;
            };
            discovered.push((step, path));
        }
        discovered.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut saved =
            self.saved_paths.lock().map_err(|e| CheckpointError::Thread(e.to_string()))?;
        for (step, path) in &discovered {
            if let Some(existing) = saved.iter_mut().find(|(s, _)| s == step) {
                existing.1 = path.clone();
            } else {
                saved.push((*step, path.clone()));
            }
        }
        saved.sort_by_key(|(step, _)| *step);
        Ok(discovered)
    }

    /// List all saved checkpoints in step order.
    pub fn list_checkpoints(&self) -> Vec<(u64, PathBuf)> {
        let mut result = self.saved_paths.lock().map(|g| g.clone()).unwrap_or_default();
        result.sort_by_key(|(step, _)| *step);
        result
    }

    /// Path of the checkpoint that produced the best tracked metric.
    ///
    /// Returns `None` when no metric has been observed yet, or when the checkpoint that
    /// produced it has since been rotated away by `max_checkpoints_to_keep` — a stale path
    /// is worse than an honest `None`.
    pub fn best_checkpoint(&self) -> Option<PathBuf> {
        let best = (*self.best_metric.lock().ok()?)?;
        if best.value.is_nan() {
            return None;
        }
        let saved = self.saved_paths.lock().ok()?;
        saved.iter().find(|(step, _)| *step == best.step).map(|(_, path)| path.clone())
    }

    /// Best value observed so far for the configured metric, if any.
    pub fn best_metric_value(&self) -> Option<f64> {
        self.best_metric.lock().ok().and_then(|guard| guard.map(|best| best.value))
    }

    /// Should we checkpoint at this step?
    pub fn should_checkpoint(&self, step: u64, current_metric: Option<f64>) -> bool {
        // Interval check
        let interval_hit = step > 0 && step.is_multiple_of(self.config.save_interval_steps);
        if !interval_hit {
            return false;
        }

        if !self.config.save_best_only {
            return true;
        }

        // save_best_only: only save when metric improves
        let current = match current_metric {
            Some(v) => v,
            None => return true, // no metric supplied — save anyway
        };

        let best = self.best_metric.lock().ok().and_then(|g| *g);
        match best {
            None => true, // first checkpoint
            Some(prev) => match self.config.best_metric_mode {
                MetricMode::Min => current < prev.value,
                MetricMode::Max => current > prev.value,
            },
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    fn cleanup_old_checkpoints(&self) -> Result<(), CheckpointError> {
        let mut saved =
            self.saved_paths.lock().map_err(|e| CheckpointError::Thread(e.to_string()))?;
        saved.sort_by_key(|(s, _)| *s);
        while saved.len() > self.config.max_checkpoints_to_keep {
            let (_, old_path) = saved.remove(0);
            if old_path.exists() {
                std::fs::remove_file(&old_path)?;
            }
        }
        Ok(())
    }

    fn checkpoint_path(&self, step: u64) -> PathBuf {
        let extension = self.config.serialize_format.extension();
        self.config
            .checkpoint_dir
            .join(format!("checkpoint_step_{step:010}.{extension}"))
    }

    /// Stream `data` to `path` in `format`.
    ///
    /// Every format writes through a `BufWriter` — the checkpoint is never materialised as a
    /// second in-memory copy (which for JSON meant a `String` several times the size of the
    /// model itself).
    fn write_to_disk(
        data: &CheckpointData,
        path: &Path,
        format: SerializeFormat,
    ) -> Result<(), CheckpointError> {
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        match format {
            SerializeFormat::Json => serde_json::to_writer_pretty(&mut writer, data)
                .map_err(|e| CheckpointError::Serialization(e.to_string()))?,
            SerializeFormat::Compact => serde_json::to_writer(&mut writer, data)
                .map_err(|e| CheckpointError::Serialization(e.to_string()))?,
            SerializeFormat::Binary => {
                let wire = BinaryCheckpoint::from_data(data)?;
                writer.write_all(&BINARY_MAGIC)?;
                oxicode::serde::encode_into_std_write(
                    &wire,
                    &mut writer,
                    oxicode::config::standard(),
                )
                .map_err(|e| CheckpointError::Serialization(e.to_string()))?;
            },
        }
        writer.flush()?;
        Ok(())
    }
}

/// Step encoded in a `checkpoint_step_<step>.{json,ckpt}` filename, if it is one.
fn checkpoint_step_from_path(path: &Path) -> Option<u64> {
    let extension = path.extension()?.to_str()?;
    if extension != "json" && extension != "ckpt" {
        return None;
    }
    let stem = path.file_stem()?.to_str()?;
    stem.strip_prefix("checkpoint_step_")?.parse::<u64>().ok()
}

/// Fold `value` (observed at `step`) into the running best under `mode`.
fn update_best(best: &mut Option<BestMetric>, value: f64, step: u64, mode: &MetricMode) {
    if value.is_nan() {
        return;
    }
    let improved = match best {
        None => true,
        Some(prev) => match mode {
            MetricMode::Min => value < prev.value,
            MetricMode::Max => value > prev.value,
        },
    };
    if improved {
        *best = Some(BestMetric { value, step });
    }
}

/// Read up to `buf.len()` bytes, tolerating a file shorter than the buffer.
///
/// `Read::read` may return fewer bytes than requested even when more are available, so a
/// single call is not enough to decide whether the magic header is present.
fn read_up_to<R: Read>(reader: &mut R, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0usize;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

// ─── CheckpointState ────────────────────────────────────────────────────────

/// Full training state for checkpointing (lightweight, f32-based).
#[derive(Debug, Clone)]
pub struct CheckpointState {
    /// Current training epoch (zero-based).
    pub epoch: usize,
    /// Current global step.
    pub step: usize,
    /// Flat model parameter vector.
    pub model_params: Vec<f32>,
    /// Flat optimizer state vector (e.g., Adam m and v concatenated).
    pub optimizer_state: Vec<f32>,
    /// Latest training / validation loss.
    pub loss: f32,
    /// Arbitrary named scalar metrics.
    pub metrics: HashMap<String, f32>,
}

impl CheckpointState {
    /// Create a new `CheckpointState` with empty param/optimizer vectors.
    pub fn new(epoch: usize, step: usize, loss: f32) -> Self {
        Self {
            epoch,
            step,
            model_params: Vec::new(),
            optimizer_state: Vec::new(),
            loss,
            metrics: HashMap::new(),
        }
    }

    /// Builder helper — attach model parameters.
    pub fn with_params(mut self, params: Vec<f32>) -> Self {
        self.model_params = params;
        self
    }

    /// Builder helper — attach optimizer state.
    pub fn with_optimizer_state(mut self, state: Vec<f32>) -> Self {
        self.optimizer_state = state;
        self
    }

    /// Add a named scalar metric.
    pub fn add_metric(&mut self, name: impl Into<String>, value: f32) {
        self.metrics.insert(name.into(), value);
    }
}

// ─── CheckpointMetadata ──────────────────────────────────────────────────────

/// Lightweight metadata about one saved checkpoint — no bulk data.
#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    /// Path to the checkpoint file on disk.
    pub path: String,
    /// Global step at which this checkpoint was saved.
    pub step: usize,
    /// Epoch at which this checkpoint was saved.
    pub epoch: usize,
    /// Loss value at checkpoint time.
    pub loss: f32,
    /// Wall-clock time when the checkpoint was registered.
    pub timestamp: std::time::SystemTime,
}

impl CheckpointMetadata {
    /// Create a new `CheckpointMetadata`, timestamping with `SystemTime::now()`.
    pub fn new(path: impl Into<String>, step: usize, epoch: usize, loss: f32) -> Self {
        Self {
            path: path.into(),
            step,
            epoch,
            loss,
            timestamp: std::time::SystemTime::now(),
        }
    }
}

// ─── CheckpointManager ───────────────────────────────────────────────────────

/// Manages checkpoint lifecycle: registration, best-checkpoint tracking, and rotation.
///
/// `CheckpointManager` is purely in-memory — it does not perform I/O itself.
/// Callers are responsible for writing the actual checkpoint files; they then
/// register the result via [`CheckpointManager::register_checkpoint`].
pub struct CheckpointManager {
    /// Directory where checkpoints are (or will be) stored.
    pub save_dir: std::path::PathBuf,
    /// Maximum number of checkpoints to retain in the tracked list.
    pub max_checkpoints: usize,
    /// Queue of registered checkpoints (oldest first).
    pub saved_checkpoints: std::collections::VecDeque<CheckpointMetadata>,
}

impl CheckpointManager {
    /// Create a new manager. `max_checkpoints` is clamped to a minimum of 1.
    pub fn new(save_dir: &str, max_checkpoints: usize) -> Self {
        Self {
            save_dir: std::path::PathBuf::from(save_dir),
            max_checkpoints: max_checkpoints.max(1),
            saved_checkpoints: std::collections::VecDeque::new(),
        }
    }

    /// Returns `true` when a checkpoint should be saved at `step`.
    ///
    /// A checkpoint is due when:
    /// - `save_every_n_steps > 0`
    /// - `step > 0`
    /// - `step % save_every_n_steps == 0`
    pub fn should_save(&self, step: usize, save_every_n_steps: usize) -> bool {
        save_every_n_steps > 0 && step > 0 && step.is_multiple_of(save_every_n_steps)
    }

    /// Register a checkpoint. Pushes `meta` to the back of the queue.
    ///
    /// Call [`cleanup_old_checkpoints`](Self::cleanup_old_checkpoints) after
    /// registration to enforce the `max_checkpoints` limit.
    pub fn register_checkpoint(&mut self, meta: CheckpointMetadata) {
        self.saved_checkpoints.push_back(meta);
    }

    /// Return a reference to the checkpoint with the lowest `loss`.
    ///
    /// Returns `None` if no checkpoints have been registered.
    pub fn get_best_checkpoint(&self) -> Option<&CheckpointMetadata> {
        self.saved_checkpoints
            .iter()
            .min_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Return a reference to the most recently registered checkpoint.
    ///
    /// Returns `None` if no checkpoints have been registered.
    pub fn get_latest_checkpoint(&self) -> Option<&CheckpointMetadata> {
        self.saved_checkpoints.back()
    }

    /// Remove the oldest checkpoints until the queue is within `max_checkpoints`.
    ///
    /// Returns the paths of the removed checkpoints so the caller can delete
    /// the corresponding files from disk.
    pub fn cleanup_old_checkpoints(&mut self) -> Vec<String> {
        let mut removed = Vec::new();
        while self.saved_checkpoints.len() > self.max_checkpoints {
            if let Some(old) = self.saved_checkpoints.pop_front() {
                removed.push(old.path);
            }
        }
        removed
    }

    /// Number of checkpoints currently tracked.
    pub fn checkpoint_count(&self) -> usize {
        self.saved_checkpoints.len()
    }
}

// ─── Error ──────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Checkpoint not found: {path}")]
    NotFound { path: PathBuf },
    #[error("Thread error: {0}")]
    Thread(String),
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn make_config(dir: PathBuf) -> AsyncCheckpointConfig {
        AsyncCheckpointConfig {
            checkpoint_dir: dir,
            max_checkpoints_to_keep: 3,
            save_interval_steps: 10,
            save_best_only: false,
            best_metric_name: "val_loss".to_string(),
            best_metric_mode: MetricMode::Min,
            serialize_format: SerializeFormat::Compact,
        }
    }

    fn make_data(step: u64) -> CheckpointData {
        let mut d = CheckpointData::new(step, (step / 100) as u32);
        d.add_weight("layer0.weight", vec![1.0, 2.0, 3.0]);
        d.add_metric("val_loss", 1.0 / (step as f64 + 1.0));
        d
    }

    // 1. New checkpointer creates directory
    #[test]
    fn test_new_creates_dir() {
        let dir = std::env::temp_dir().join(format!("ckpt_new_{}", fastrand::u64(..)));
        assert!(!dir.exists());
        let _ckpt = AsyncCheckpointer::new(make_config(dir.clone())).unwrap();
        assert!(dir.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 2. save_async returns a handle immediately (does not block)
    #[test]
    fn test_async_save_returns_immediately() {
        let dir = std::env::temp_dir().join(format!("ckpt_async_{}", fastrand::u64(..)));
        let ckpt = AsyncCheckpointer::new(make_config(dir.clone())).unwrap();
        let data = make_data(100);
        let start = Instant::now();
        let handle = ckpt.save_async(data).unwrap();
        let elapsed = start.elapsed();
        // The call itself should be very fast (< 500 ms) — the write happens async
        assert!(
            elapsed < Duration::from_millis(500),
            "save_async took too long: {elapsed:?}"
        );
        // Handle should exist
        let step = handle.lock().unwrap_or_else(|e| e.into_inner()).step;
        assert_eq!(step, 100);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 3. wait_all completes after async save
    #[test]
    fn test_wait_all_completes() {
        let dir = std::env::temp_dir().join(format!("ckpt_wait_{}", fastrand::u64(..)));
        let ckpt = AsyncCheckpointer::new(make_config(dir.clone())).unwrap();
        ckpt.save_async(make_data(10)).unwrap();
        ckpt.save_async(make_data(20)).unwrap();
        ckpt.wait_all().unwrap();
        // After wait_all all handles are complete
        let pending = ckpt.pending_handles.lock().unwrap_or_else(|e| e.into_inner());
        for h in pending.iter() {
            assert!(h.lock().unwrap_or_else(|e| e.into_inner()).is_complete);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 4. Files are actually written after wait_all
    #[test]
    fn test_files_written_after_wait_all() {
        let dir = std::env::temp_dir().join(format!("ckpt_files_{}", fastrand::u64(..)));
        let ckpt = AsyncCheckpointer::new(make_config(dir.clone())).unwrap();
        let _h = ckpt.save_async(make_data(50)).unwrap();
        ckpt.wait_all().unwrap();
        let expected = dir.join("checkpoint_step_0000000050.json");
        assert!(
            expected.exists(),
            "checkpoint file should exist at {expected:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 5. save_sync + load roundtrip
    #[test]
    fn test_sync_save_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("ckpt_rt_{}", fastrand::u64(..)));
        let ckpt = AsyncCheckpointer::new(make_config(dir.clone())).unwrap();
        let mut data = make_data(200);
        data.add_weight("layer1.bias", vec![0.1, 0.2]);
        data.config = serde_json::json!({"hidden": 256});

        let path = ckpt.save_sync(&data).unwrap();
        let loaded = AsyncCheckpointer::load(&path).unwrap();

        assert_eq!(loaded.step, 200);
        assert_eq!(loaded.epoch, data.epoch);
        assert_eq!(loaded.weights["layer0.weight"], vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded.weights["layer1.bias"], vec![0.1, 0.2]);
        assert!((loaded.metrics["val_loss"] - data.metrics["val_loss"]).abs() < 1e-12);
        assert_eq!(loaded.config["hidden"], 256);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 6. load returns NotFound for missing path
    #[test]
    fn test_load_not_found() {
        let path = std::env::temp_dir().join("does_not_exist_12345678.json");
        let result = AsyncCheckpointer::load(&path);
        assert!(matches!(result, Err(CheckpointError::NotFound { .. })));
    }

    // 7. max_checkpoints_to_keep cleanup via sync saves
    #[test]
    fn test_max_checkpoints_cleanup() {
        let dir = std::env::temp_dir().join(format!("ckpt_clean_{}", fastrand::u64(..)));
        let mut cfg = make_config(dir.clone());
        cfg.max_checkpoints_to_keep = 2;
        let ckpt = AsyncCheckpointer::new(cfg).unwrap();

        for step in [10u64, 20, 30, 40] {
            ckpt.save_sync(&make_data(step)).unwrap();
        }

        let saved = ckpt.list_checkpoints();
        // Should keep only the 2 most recent
        assert_eq!(
            saved.len(),
            2,
            "should keep 2 checkpoints, got {}",
            saved.len()
        );
        assert_eq!(saved[0].0, 30);
        assert_eq!(saved[1].0, 40);

        // Old files deleted from disk
        assert!(!dir.join("checkpoint_step_0000000010.json").exists());
        assert!(!dir.join("checkpoint_step_0000000020.json").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 8. max_checkpoints_to_keep cleanup via async saves
    #[test]
    fn test_max_checkpoints_async_cleanup() {
        let dir = std::env::temp_dir().join(format!("ckpt_aclean_{}", fastrand::u64(..)));
        let mut cfg = make_config(dir.clone());
        cfg.max_checkpoints_to_keep = 2;
        let ckpt = AsyncCheckpointer::new(cfg).unwrap();

        for step in [100u64, 200, 300, 400] {
            ckpt.save_async(make_data(step)).unwrap();
        }
        ckpt.wait_all().unwrap();

        let saved = ckpt.list_checkpoints();
        assert!(saved.len() <= 2, "should keep at most 2 checkpoints");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 9. save_best_only with MetricMode::Min
    #[test]
    fn test_save_best_only_min() {
        let dir = std::env::temp_dir().join(format!("ckpt_best_{}", fastrand::u64(..)));
        let mut cfg = make_config(dir.clone());
        cfg.save_best_only = true;
        cfg.best_metric_mode = MetricMode::Min;
        cfg.best_metric_name = "val_loss".to_string();
        cfg.save_interval_steps = 10;
        let ckpt = AsyncCheckpointer::new(cfg).unwrap();

        // step=10, val_loss=0.5 — first, should save
        assert!(ckpt.should_checkpoint(10, Some(0.5)));
        // Simulate saving to update best metric
        let mut d = make_data(10);
        d.metrics.insert("val_loss".to_string(), 0.5);
        ckpt.save_sync(&d).unwrap();
        // Update internal best (save_sync updates best_metric)
        // step=20, val_loss=0.6 — worse, should not save
        assert!(!ckpt.should_checkpoint(20, Some(0.6)));
        // step=30, val_loss=0.3 — better, should save
        assert!(ckpt.should_checkpoint(30, Some(0.3)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 10. save_best_only with MetricMode::Max
    #[test]
    fn test_save_best_only_max() {
        let dir = std::env::temp_dir().join(format!("ckpt_bestmax_{}", fastrand::u64(..)));
        let mut cfg = make_config(dir.clone());
        cfg.save_best_only = true;
        cfg.best_metric_mode = MetricMode::Max;
        cfg.best_metric_name = "accuracy".to_string();
        cfg.save_interval_steps = 10;
        let ckpt = AsyncCheckpointer::new(cfg).unwrap();

        // First checkpoint always saves
        assert!(ckpt.should_checkpoint(10, Some(0.7)));
        let mut d = make_data(10);
        d.metrics.insert("accuracy".to_string(), 0.7);
        ckpt.save_sync(&d).unwrap();
        // 0.6 is worse for Max
        assert!(!ckpt.should_checkpoint(20, Some(0.6)));
        // 0.9 is better for Max
        assert!(ckpt.should_checkpoint(30, Some(0.9)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 11. should_checkpoint respects interval
    #[test]
    fn test_should_checkpoint_interval() {
        let dir = std::env::temp_dir().join(format!("ckpt_intv_{}", fastrand::u64(..)));
        let mut cfg = make_config(dir.clone());
        cfg.save_interval_steps = 50;
        let ckpt = AsyncCheckpointer::new(cfg).unwrap();

        assert!(!ckpt.should_checkpoint(0, None));
        assert!(!ckpt.should_checkpoint(49, None));
        assert!(ckpt.should_checkpoint(50, None));
        assert!(!ckpt.should_checkpoint(51, None));
        assert!(ckpt.should_checkpoint(100, None));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 12. list_checkpoints returns ordered results
    #[test]
    fn test_list_checkpoints_ordering() {
        let dir = std::env::temp_dir().join(format!("ckpt_order_{}", fastrand::u64(..)));
        let ckpt = AsyncCheckpointer::new(make_config(dir.clone())).unwrap();
        // Save in non-sequential order
        for &step in &[30u64, 10, 20] {
            ckpt.save_sync(&make_data(step)).unwrap();
        }
        let saved = ckpt.list_checkpoints();
        let steps: Vec<u64> = saved.iter().map(|(s, _)| *s).collect();
        assert_eq!(steps, vec![10, 20, 30]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 13. estimated_size_bytes
    #[test]
    fn test_estimated_size_bytes() {
        let mut d = CheckpointData::new(0, 0);
        d.add_weight("w", vec![1.0; 1000]);
        d.optimizer_state.insert("m".to_string(), vec![0.0; 500]);
        // 1000 + 500 = 1500 values * 4 bytes (f32, not a widened f64)
        assert_eq!(d.estimated_size_bytes(), 6_000);
    }

    // 14. Pretty Json format roundtrip
    #[test]
    fn test_json_format_roundtrip() {
        let dir = std::env::temp_dir().join(format!("ckpt_json_{}", fastrand::u64(..)));
        let mut cfg = make_config(dir.clone());
        cfg.serialize_format = SerializeFormat::Json;
        let ckpt = AsyncCheckpointer::new(cfg).unwrap();
        let data = make_data(5);
        let path = ckpt.save_sync(&data).unwrap();
        // Pretty-printed JSON should contain newlines
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains('\n'), "pretty JSON should contain newlines");
        let loaded = AsyncCheckpointer::load(&path).unwrap();
        assert_eq!(loaded.step, 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 15. Multiple async saves all complete without error
    #[test]
    fn test_multiple_async_saves_no_error() {
        let dir = std::env::temp_dir().join(format!("ckpt_multi_{}", fastrand::u64(..)));
        let mut cfg = make_config(dir.clone());
        cfg.max_checkpoints_to_keep = 20;
        let ckpt = AsyncCheckpointer::new(cfg).unwrap();
        for i in 0..10u64 {
            ckpt.save_async(make_data(i * 10 + 10)).unwrap();
        }
        ckpt.wait_all().unwrap();
        let saved = ckpt.list_checkpoints();
        assert_eq!(saved.len(), 10);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ─── Binary checkpoint format ────────────────────────────────────────────

    /// A checkpoint whose weights are full-precision-looking f32s — the decimal expansion
    /// of these in JSON is long, which is exactly the cost the binary format removes.
    fn bulky_data(step: u64, values: usize) -> CheckpointData {
        let mut d = CheckpointData::new(step, 0);
        let weights: Vec<f32> = (0..values)
            .map(|i| (i as f32) * std::f32::consts::PI / 7.0 - 1.234_567_9)
            .collect();
        let opt: Vec<f32> = weights.iter().map(|w| w * 0.5).collect();
        d.add_weight("layer0.weight", weights);
        d.optimizer_state.insert("layer0.weight.m".to_string(), opt);
        d.add_metric("val_loss", 0.25);
        d.config = serde_json::json!({"hidden": 256, "name": "tiny"});
        d
    }

    // 33. Binary round-trip is bit-exact and preserves the free-form config.
    #[test]
    fn test_binary_format_roundtrip_is_bit_exact() {
        let dir = std::env::temp_dir().join(format!("ckpt_bin_{}", fastrand::u64(..)));
        let mut cfg = make_config(dir.clone());
        cfg.serialize_format = SerializeFormat::Binary;
        let ckpt = AsyncCheckpointer::new(cfg).expect("checkpointer");

        let mut data = bulky_data(70, 64);
        // Values that a lossy path would round or clamp.
        data.add_weight(
            "edge",
            vec![f32::MIN_POSITIVE, -f32::MAX, 1e-8, 0.1, 65_504.0, 70_000.0],
        );
        let path = ckpt.save_sync(&data).expect("save");
        assert_eq!(
            path.extension().and_then(|e| e.to_str()),
            Some("ckpt"),
            "binary checkpoints must not pretend to be .json"
        );

        let loaded = AsyncCheckpointer::load(&path).expect("load");
        assert_eq!(loaded.step, 70);
        for (name, original) in &data.weights {
            let round_tripped = loaded.weights.get(name).expect("weight survives the round trip");
            let original_bits: Vec<u32> = original.iter().map(|v| v.to_bits()).collect();
            let loaded_bits: Vec<u32> = round_tripped.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                original_bits, loaded_bits,
                "weights '{name}' must be bit-exact"
            );
        }
        assert_eq!(loaded.optimizer_state, data.optimizer_state);
        assert_eq!(loaded.config["hidden"], 256);
        assert_eq!(loaded.config["name"], "tiny");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 34. Regression: weights used to be pretty-printed f64 decimal text. The binary
    //     format must be dramatically smaller than either JSON encoding.
    #[test]
    fn test_binary_format_is_far_smaller_than_json() {
        let dir = std::env::temp_dir().join(format!("ckpt_size_{}", fastrand::u64(..)));
        let data = bulky_data(10, 4096);

        let mut sizes = HashMap::new();
        for format in [
            SerializeFormat::Json,
            SerializeFormat::Compact,
            SerializeFormat::Binary,
        ] {
            let mut cfg = make_config(dir.clone());
            cfg.serialize_format = format;
            let ckpt = AsyncCheckpointer::new(cfg).expect("checkpointer");
            let path = ckpt.save_sync(&data).expect("save");
            let len = std::fs::metadata(&path).expect("metadata").len();
            sizes.insert(format, len);
            // Every format must still round-trip through the sniffing loader.
            let loaded = AsyncCheckpointer::load(&path).expect("load");
            assert_eq!(
                loaded.weights["layer0.weight"],
                data.weights["layer0.weight"]
            );
        }

        let binary = sizes[&SerializeFormat::Binary];
        let compact = sizes[&SerializeFormat::Compact];
        let pretty = sizes[&SerializeFormat::Json];
        // 8192 f32 values -> ~32 KiB of payload; JSON needs >10 bytes per value.
        assert!(
            binary * 2 < compact,
            "binary ({binary} B) must be far smaller than compact JSON ({compact} B)"
        );
        assert!(
            compact <= pretty,
            "pretty JSON ({pretty} B) cannot be smaller than compact JSON ({compact} B)"
        );
        let expected_payload = (8192 * WEIGHT_VALUE_BYTES) as u64;
        assert!(
            binary < expected_payload * 2,
            "binary ({binary} B) should be within 2x of the raw f32 payload \
             ({expected_payload} B)"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 35. The loader detects the format from the file, not from the configuration.
    #[test]
    fn test_load_detects_format_independently_of_config() {
        let dir = std::env::temp_dir().join(format!("ckpt_sniff_{}", fastrand::u64(..)));
        let data = bulky_data(20, 8);

        let mut binary_cfg = make_config(dir.clone());
        binary_cfg.serialize_format = SerializeFormat::Binary;
        let binary_path = AsyncCheckpointer::new(binary_cfg)
            .expect("checkpointer")
            .save_sync(&data)
            .expect("save");

        let mut json_cfg = make_config(dir.clone());
        json_cfg.serialize_format = SerializeFormat::Compact;
        let json_path = AsyncCheckpointer::new(json_cfg)
            .expect("checkpointer")
            .save_sync(&data)
            .expect("save");

        // One static loader reads both.
        assert_eq!(
            AsyncCheckpointer::load(&binary_path).expect("binary").step,
            20
        );
        assert_eq!(AsyncCheckpointer::load(&json_path).expect("json").step, 20);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 36. An empty (or truncated) file is an error, never a silently empty checkpoint.
    #[test]
    fn test_load_rejects_a_truncated_file() {
        let dir = std::env::temp_dir().join(format!("ckpt_trunc_{}", fastrand::u64(..)));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("empty.ckpt");
        std::fs::write(&path, b"").expect("write");
        assert!(matches!(
            AsyncCheckpointer::load(&path),
            Err(CheckpointError::Serialization(_))
        ));

        let short = dir.join("short.ckpt");
        std::fs::write(&short, BINARY_MAGIC).expect("write");
        assert!(matches!(
            AsyncCheckpointer::load(&short),
            Err(CheckpointError::Serialization(_))
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 37. Regression: `best_checkpoint` used to return the *last* saved path regardless of
    //     which step actually produced the best metric.
    #[test]
    fn test_best_checkpoint_names_the_step_with_the_best_metric() {
        let dir = std::env::temp_dir().join(format!("ckpt_bestpath_{}", fastrand::u64(..)));
        let mut cfg = make_config(dir.clone());
        cfg.max_checkpoints_to_keep = 10;
        cfg.best_metric_name = "val_loss".to_string();
        cfg.best_metric_mode = MetricMode::Min;
        let ckpt = AsyncCheckpointer::new(cfg).expect("checkpointer");

        for (step, loss) in [(10u64, 0.9f64), (20, 0.2), (30, 0.7)] {
            let mut d = CheckpointData::new(step, 0);
            d.add_weight("w", vec![step as f32]);
            d.add_metric("val_loss", loss);
            ckpt.save_sync(&d).expect("save");
        }

        let best = ckpt.best_checkpoint().expect("a best checkpoint must be known");
        assert!(
            best.to_string_lossy().contains("0000000020"),
            "step 20 had the lowest val_loss, got {best:?}"
        );
        assert_eq!(ckpt.best_metric_value(), Some(0.2));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 39. A restarted checkpointer finds the checkpoints already on disk, in either format.
    #[test]
    fn test_discover_existing_finds_both_formats() {
        let dir = std::env::temp_dir().join(format!("ckpt_discover_{}", fastrand::u64(..)));
        let data = bulky_data(10, 8);

        let mut binary_cfg = make_config(dir.clone());
        binary_cfg.serialize_format = SerializeFormat::Binary;
        binary_cfg.max_checkpoints_to_keep = 10;
        let first = AsyncCheckpointer::new(binary_cfg).expect("checkpointer");
        first.save_sync(&data).expect("save");

        let mut json_cfg = make_config(dir.clone());
        json_cfg.serialize_format = SerializeFormat::Compact;
        json_cfg.max_checkpoints_to_keep = 10;
        let second = AsyncCheckpointer::new(json_cfg).expect("checkpointer");
        second.save_sync(&bulky_data(20, 8)).expect("save");

        // A brand-new process: nothing tracked in memory yet.
        let mut fresh_cfg = make_config(dir.clone());
        fresh_cfg.max_checkpoints_to_keep = 10;
        let restarted = AsyncCheckpointer::new(fresh_cfg).expect("checkpointer");
        assert!(restarted.list_checkpoints().is_empty());

        let found = restarted.discover_existing().expect("scan");
        let steps: Vec<u64> = found.iter().map(|(step, _)| *step).collect();
        assert_eq!(
            steps,
            vec![10, 20],
            "both the .ckpt and the .json must be found"
        );
        assert_eq!(restarted.list_checkpoints().len(), 2);
        for (_, path) in &found {
            AsyncCheckpointer::load(path).expect("every discovered checkpoint must load");
        }
        // Nothing was compared, so no checkpoint may be claimed as "best".
        assert!(restarted.best_checkpoint().is_none());

        // Unrelated files in the directory are ignored, and rescanning is idempotent.
        std::fs::write(dir.join("notes.txt"), b"hello").expect("write");
        std::fs::write(dir.join("checkpoint_step_abc.ckpt"), b"x").expect("write");
        assert_eq!(restarted.discover_existing().expect("rescan").len(), 2);
        assert_eq!(restarted.list_checkpoints().len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 38. Background saves feed the best-metric tracker as well as synchronous ones.
    #[test]
    fn test_async_save_updates_best_metric() {
        let dir = std::env::temp_dir().join(format!("ckpt_asyncbest_{}", fastrand::u64(..)));
        let mut cfg = make_config(dir.clone());
        cfg.max_checkpoints_to_keep = 10;
        cfg.save_best_only = true;
        let ckpt = AsyncCheckpointer::new(cfg).expect("checkpointer");

        let mut d = CheckpointData::new(10, 0);
        d.add_metric("val_loss", 0.4);
        ckpt.save_async(d).expect("save_async");
        ckpt.wait_all().expect("wait");

        assert_eq!(ckpt.best_metric_value(), Some(0.4));
        assert!(
            !ckpt.should_checkpoint(20, Some(0.5)),
            "0.5 is worse than 0.4"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ─── CheckpointState tests ───────────────────────────────────────────────

    // 16. CheckpointState::new sets fields correctly
    #[test]
    fn test_checkpoint_state_new() {
        let state = CheckpointState::new(1, 100, 0.5);
        assert_eq!(state.epoch, 1);
        assert_eq!(state.step, 100);
        assert!((state.loss - 0.5).abs() < 1e-6);
        assert!(state.model_params.is_empty());
        assert!(state.optimizer_state.is_empty());
        assert!(state.metrics.is_empty());
    }

    // 17. with_params builder populates model_params
    #[test]
    fn test_checkpoint_state_with_params() {
        let params = vec![1.0_f32, 2.0, 3.0];
        let state = CheckpointState::new(0, 0, 0.0).with_params(params.clone());
        assert_eq!(state.model_params, params);
    }

    // 18. with_optimizer_state builder populates optimizer_state
    #[test]
    fn test_checkpoint_state_with_optimizer_state() {
        let opt_state: Vec<f32> = vec![0.1_f32; 10];
        let state = CheckpointState::new(0, 0, 0.0).with_optimizer_state(opt_state.clone());
        assert_eq!(state.optimizer_state, opt_state);
    }

    // 19. add_metric inserts into metrics map
    #[test]
    fn test_checkpoint_state_add_metric() {
        let mut state = CheckpointState::new(2, 200, 0.3);
        state.add_metric("accuracy", 0.95);
        state.add_metric("f1", 0.88);
        assert!((state.metrics["accuracy"] - 0.95).abs() < 1e-6);
        assert!((state.metrics["f1"] - 0.88).abs() < 1e-6);
    }

    // ─── CheckpointMetadata tests ────────────────────────────────────────────

    // 20. CheckpointMetadata::new sets all fields
    #[test]
    fn test_checkpoint_metadata_new() {
        let ckpt_path =
            std::env::temp_dir().join("ckpt_step_100.json").to_string_lossy().into_owned();
        let before = std::time::SystemTime::now();
        let meta = CheckpointMetadata::new(ckpt_path.clone(), 100, 1, 0.42);
        let after = std::time::SystemTime::now();
        assert_eq!(meta.path, ckpt_path);
        assert_eq!(meta.step, 100);
        assert_eq!(meta.epoch, 1);
        assert!((meta.loss - 0.42).abs() < 1e-6);
        assert!(meta.timestamp >= before && meta.timestamp <= after);
    }

    // ─── CheckpointManager tests ─────────────────────────────────────────────

    // 21. CheckpointManager::new sets save_dir and max_checkpoints
    #[test]
    fn test_checkpoint_manager_new() {
        let save_dir = std::env::temp_dir().join("test_ckpts");
        let mgr = CheckpointManager::new(&save_dir.to_string_lossy(), 5);
        assert_eq!(mgr.save_dir, save_dir);
        assert_eq!(mgr.max_checkpoints, 5);
        assert_eq!(mgr.checkpoint_count(), 0);
    }

    // 22. should_save returns true when step is a multiple of save_every_n_steps
    #[test]
    fn test_checkpoint_manager_should_save_true() {
        let mgr = CheckpointManager::new(&std::env::temp_dir().to_string_lossy(), 3);
        assert!(mgr.should_save(100, 50));
        assert!(mgr.should_save(50, 50));
    }

    // 23. should_save returns false when step is not a multiple
    #[test]
    fn test_checkpoint_manager_should_save_false_not_multiple() {
        let mgr = CheckpointManager::new(&std::env::temp_dir().to_string_lossy(), 3);
        assert!(!mgr.should_save(51, 50));
        assert!(!mgr.should_save(99, 50));
    }

    // 24. should_save returns false when step == 0
    #[test]
    fn test_checkpoint_manager_should_save_false_step_zero() {
        let mgr = CheckpointManager::new(&std::env::temp_dir().to_string_lossy(), 3);
        assert!(!mgr.should_save(0, 50));
    }

    // 25. register_checkpoint increases count
    #[test]
    fn test_checkpoint_manager_register_checkpoint() {
        let dir = std::env::temp_dir();
        let mut mgr = CheckpointManager::new(&dir.to_string_lossy(), 5);
        mgr.register_checkpoint(CheckpointMetadata::new(
            dir.join("c1.json").to_string_lossy().into_owned(),
            10,
            0,
            0.5,
        ));
        assert_eq!(mgr.checkpoint_count(), 1);
    }

    // 26. get_best_checkpoint returns the one with lowest loss
    #[test]
    fn test_checkpoint_manager_get_best() {
        let dir = std::env::temp_dir();
        let path_b = dir.join("b.json").to_string_lossy().into_owned();
        let mut mgr = CheckpointManager::new(&dir.to_string_lossy(), 5);
        mgr.register_checkpoint(CheckpointMetadata::new(
            dir.join("a.json").to_string_lossy().into_owned(),
            10,
            0,
            0.5,
        ));
        mgr.register_checkpoint(CheckpointMetadata::new(path_b.clone(), 20, 0, 0.3));
        mgr.register_checkpoint(CheckpointMetadata::new(
            dir.join("c.json").to_string_lossy().into_owned(),
            30,
            0,
            0.8,
        ));
        let best = mgr.get_best_checkpoint().expect("should have best");
        assert!(
            (best.loss - 0.3).abs() < 1e-6,
            "best loss should be 0.3, got {}",
            best.loss
        );
        assert_eq!(best.path, path_b);
    }

    // 27. get_latest_checkpoint returns the most recently registered
    #[test]
    fn test_checkpoint_manager_get_latest() {
        let dir = std::env::temp_dir();
        let path_third = dir.join("third.json").to_string_lossy().into_owned();
        let mut mgr = CheckpointManager::new(&dir.to_string_lossy(), 5);
        mgr.register_checkpoint(CheckpointMetadata::new(
            dir.join("first.json").to_string_lossy().into_owned(),
            10,
            0,
            0.5,
        ));
        mgr.register_checkpoint(CheckpointMetadata::new(
            dir.join("second.json").to_string_lossy().into_owned(),
            20,
            0,
            0.4,
        ));
        mgr.register_checkpoint(CheckpointMetadata::new(path_third.clone(), 30, 0, 0.3));
        let latest = mgr.get_latest_checkpoint().expect("should have latest");
        assert_eq!(latest.path, path_third);
    }

    // 28. cleanup_old_checkpoints removes excess, returns paths, count drops
    #[test]
    fn test_checkpoint_manager_cleanup() {
        let dir = std::env::temp_dir();
        let mut mgr = CheckpointManager::new(&dir.to_string_lossy(), 2);
        for i in 1..=4 {
            mgr.register_checkpoint(CheckpointMetadata::new(
                dir.join(format!("ck{i}.json")).to_string_lossy().into_owned(),
                i * 10,
                0,
                i as f32 * 0.1,
            ));
        }
        assert_eq!(mgr.checkpoint_count(), 4);
        let removed = mgr.cleanup_old_checkpoints();
        assert_eq!(removed.len(), 2, "should remove 2 checkpoints");
        assert_eq!(mgr.checkpoint_count(), 2);
        assert!(removed.contains(&dir.join("ck1.json").to_string_lossy().into_owned()));
        assert!(removed.contains(&dir.join("ck2.json").to_string_lossy().into_owned()));
    }

    // 29. cleanup_old_checkpoints is a no-op when under limit
    #[test]
    fn test_checkpoint_manager_cleanup_no_op() {
        let dir = std::env::temp_dir();
        let mut mgr = CheckpointManager::new(&dir.to_string_lossy(), 5);
        for i in 1..=3 {
            mgr.register_checkpoint(CheckpointMetadata::new(
                dir.join(format!("ck{i}.json")).to_string_lossy().into_owned(),
                i,
                0,
                0.1,
            ));
        }
        let removed = mgr.cleanup_old_checkpoints();
        assert!(removed.is_empty(), "should not remove anything");
        assert_eq!(mgr.checkpoint_count(), 3);
    }

    // 30. get_best_checkpoint returns None when empty
    #[test]
    fn test_checkpoint_manager_best_empty() {
        let mgr = CheckpointManager::new(&std::env::temp_dir().to_string_lossy(), 3);
        assert!(mgr.get_best_checkpoint().is_none());
    }

    // 31. get_latest_checkpoint returns None when empty
    #[test]
    fn test_checkpoint_manager_latest_empty() {
        let mgr = CheckpointManager::new(&std::env::temp_dir().to_string_lossy(), 3);
        assert!(mgr.get_latest_checkpoint().is_none());
    }

    // 32. max_checkpoints=1: after cleanup only 1 remains
    #[test]
    fn test_checkpoint_manager_max_1() {
        let dir = std::env::temp_dir();
        let mut mgr = CheckpointManager::new(&dir.to_string_lossy(), 1);
        for i in 1..=3 {
            mgr.register_checkpoint(CheckpointMetadata::new(
                dir.join(format!("ck{i}.json")).to_string_lossy().into_owned(),
                i,
                0,
                0.5,
            ));
        }
        let removed = mgr.cleanup_old_checkpoints();
        assert_eq!(removed.len(), 2);
        assert_eq!(mgr.checkpoint_count(), 1);
    }
}
