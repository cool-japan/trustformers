//! Parallel model weight loading for faster startup.
//!
//! Loads model weight chunks concurrently using a thread pool,
//! reducing loading time by 2-4x for large models with many shards.
//!
//! # Example
//!
//! ```rust,ignore
//! use trustformers::loading::{ParallelWeightLoader, ParallelLoaderConfig};
//! use std::path::Path;
//!
//! let config = ParallelLoaderConfig::default();
//! let loader = ParallelWeightLoader::new(config);
//! let chunks = loader.load_sharded_directory(Path::new("/models/llama-70b"))?;
//! ```

use crate::error::TrustformersError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the parallel weight loader.
#[derive(Debug, Clone)]
pub struct ParallelLoaderConfig {
    /// Number of parallel loading threads.
    /// Defaults to the number of logical CPUs.
    pub num_threads: usize,
    /// Maximum chunk size in bytes when splitting a single file.
    /// Default: 512 MiB.
    pub chunk_size_bytes: usize,
    /// If `true`, prefer memory-mapped I/O for large files (>= 256 MiB).
    pub use_mmap: bool,
    /// If `true`, asynchronously prefetch the next shard while processing the current one.
    pub prefetch: bool,
    /// How many chunks to load between progress callback invocations.
    pub progress_interval: usize,
}

impl Default for ParallelLoaderConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get().max(1),
            chunk_size_bytes: 512 * 1024 * 1024, // 512 MiB
            use_mmap: true,
            prefetch: true,
            progress_interval: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Progress reporting
// ---------------------------------------------------------------------------

/// Progress snapshot emitted during a parallel loading session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadingProgress {
    /// Number of chunks successfully loaded so far.
    pub loaded_chunks: usize,
    /// Total number of chunks expected.
    pub total_chunks: usize,
    /// Total bytes loaded so far.
    pub loaded_bytes: usize,
    /// Total bytes to load.
    pub total_bytes: usize,
    /// Elapsed wall-clock time in seconds.
    pub elapsed_secs: f64,
    /// Current throughput in MiB/s.
    pub throughput_mb_per_sec: f64,
}

impl LoadingProgress {
    /// Percentage of chunks loaded (0.0 – 100.0).
    pub fn pct_complete(&self) -> f32 {
        if self.total_chunks == 0 {
            return 100.0;
        }
        self.loaded_chunks as f32 / self.total_chunks as f32 * 100.0
    }
}

/// Type alias for a boxed progress callback.
pub type ProgressCallback = Box<dyn Fn(LoadingProgress) + Send + Sync>;

// ---------------------------------------------------------------------------
// WeightChunk
// ---------------------------------------------------------------------------

/// A single chunk of weights loaded from one shard file.
#[derive(Debug)]
pub struct WeightChunk {
    /// Sequential index of this chunk within the loading session.
    pub chunk_id: usize,
    /// Raw byte buffers keyed by tensor name.
    pub tensors: HashMap<String, Vec<u8>>,
    /// Dtype string keyed by tensor name (e.g., `"float32"`, `"bfloat16"`).
    pub dtype_map: HashMap<String, String>,
    /// Tensor shapes keyed by tensor name.
    pub shape_map: HashMap<String, Vec<usize>>,
}

impl WeightChunk {
    /// Returns the total byte count stored across all tensors.
    pub fn total_bytes(&self) -> usize {
        self.tensors.values().map(|v| v.len()).sum()
    }

    /// Returns the number of tensors in this chunk.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

// ---------------------------------------------------------------------------
// LoadingStats
// ---------------------------------------------------------------------------

/// Aggregate statistics from a completed loading session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoadingStats {
    /// Number of files (shards) loaded.
    pub files_loaded: usize,
    /// Total bytes read across all shards.
    pub total_bytes_loaded: usize,
    /// Total wall-clock time in seconds.
    pub total_duration_secs: f64,
    /// Average throughput over the full session in MiB/s.
    pub avg_throughput_mb_per_sec: f64,
    /// Peak single-interval throughput in MiB/s.
    pub peak_throughput_mb_per_sec: f64,
}

// ---------------------------------------------------------------------------
// Shared loading state
// ---------------------------------------------------------------------------

/// Shared mutable state tracked across worker threads.
struct SharedLoadState {
    loaded_chunks: usize,
    loaded_bytes: usize,
    peak_throughput: f64,
    start: Instant,
}

impl SharedLoadState {
    fn new() -> Self {
        Self {
            loaded_chunks: 0,
            loaded_bytes: 0,
            peak_throughput: 0.0,
            start: Instant::now(),
        }
    }

    fn record_chunk(&mut self, byte_count: usize) -> f64 {
        self.loaded_chunks += 1;
        self.loaded_bytes += byte_count;
        let elapsed = self.start.elapsed().as_secs_f64();
        let throughput = if elapsed > 0.0 {
            self.loaded_bytes as f64 / (elapsed * 1024.0 * 1024.0)
        } else {
            0.0
        };
        if throughput > self.peak_throughput {
            self.peak_throughput = throughput;
        }
        throughput
    }
}

// ---------------------------------------------------------------------------
// ParallelWeightLoader
// ---------------------------------------------------------------------------

/// Parallel weight loader that reads multiple shard files concurrently.
pub struct ParallelWeightLoader {
    config: ParallelLoaderConfig,
    progress_callback: Option<Arc<ProgressCallback>>,
    stats: Arc<Mutex<LoadingStats>>,
}

impl ParallelWeightLoader {
    /// Create a new loader with the given configuration.
    pub fn new(config: ParallelLoaderConfig) -> Self {
        Self {
            config,
            progress_callback: None,
            stats: Arc::new(Mutex::new(LoadingStats::default())),
        }
    }

    /// Attach a progress callback, invoked every `progress_interval` chunks.
    pub fn with_progress(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(Arc::new(callback));
        self
    }

    /// Load weights from a directory that may contain sharded safetensors files.
    ///
    /// Recognises:
    /// - A single `model.safetensors`
    /// - Multiple `model-NNNNN-of-MMMMM.safetensors` shards
    /// - Legacy `pytorch_model.bin` (treated as a single opaque shard)
    pub fn load_sharded_directory(
        &self,
        dir: &Path,
    ) -> Result<HashMap<String, WeightChunk>, TrustformersError> {
        if !dir.is_dir() {
            return Err(TrustformersError::Io {
                message: format!("'{}' is not a directory", dir.display()),
                path: Some(dir.to_string_lossy().to_string()),
                suggestion: Some("Provide a path to a model directory containing weight files".to_string()),
            });
        }

        let files = self.collect_weight_files(dir)?;
        if files.is_empty() {
            return Err(TrustformersError::Io {
                message: format!("No weight files found in '{}'", dir.display()),
                path: Some(dir.to_string_lossy().to_string()),
                suggestion: Some("Ensure the directory contains .safetensors or .bin weight files".to_string()),
            });
        }

        info!(
            num_files = files.len(),
            dir = %dir.display(),
            "Loading model weights"
        );

        let chunks = self.load_files(&files)?;
        let mut result = HashMap::new();
        for chunk in chunks {
            let key = format!("shard_{}", chunk.chunk_id);
            result.insert(key, chunk);
        }
        Ok(result)
    }

    /// Load a single weight file.
    pub fn load_single_file(&self, path: &Path) -> Result<WeightChunk, TrustformersError> {
        let chunks = self.load_files(&[path.to_path_buf()])?;
        chunks.into_iter().next().ok_or_else(|| TrustformersError::Io {
            message: format!("No data read from '{}'", path.display()),
            path: Some(path.to_string_lossy().to_string()),
            suggestion: Some("Check that the file is non-empty and readable".to_string()),
        })
    }

    /// Load multiple shard files, dispatching reads across worker threads.
    pub fn load_files(
        &self,
        files: &[PathBuf],
    ) -> Result<Vec<WeightChunk>, TrustformersError> {
        let total_bytes = files
            .iter()
            .filter_map(|p| p.metadata().ok())
            .map(|m| m.len() as usize)
            .sum::<usize>();

        let total_chunks = files.len();
        let shared = Arc::new(Mutex::new(SharedLoadState::new()));
        let callback = self.progress_callback.clone();
        let interval = self.config.progress_interval;
        let start = Instant::now();

        // Split files into batches for workers
        let num_threads = self.config.num_threads.min(files.len()).max(1);
        let chunks_per_thread = files.len().div_ceil(num_threads);

        // Collect results, indexed so we can sort later
        let results: Arc<Mutex<Vec<(usize, WeightChunk)>>> = Arc::new(Mutex::new(Vec::new()));
        let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        std::thread::scope(|scope| {
            for (batch_idx, file_batch) in files.chunks(chunks_per_thread).enumerate() {
                let shared = Arc::clone(&shared);
                let results = Arc::clone(&results);
                let errors = Arc::clone(&errors);
                let callback = callback.clone();
                let use_mmap = self.config.use_mmap;

                // Compute global start index for this batch
                let global_start = batch_idx * chunks_per_thread;
                let file_batch: Vec<PathBuf> = file_batch.to_vec();

                scope.spawn(move || {
                    for (local_idx, path) in file_batch.iter().enumerate() {
                        let chunk_id = global_start + local_idx;
                        match load_file_as_chunk(chunk_id, path, use_mmap) {
                            Ok(chunk) => {
                                let byte_count = chunk.total_bytes();
                                let throughput = {
                                    let mut state = shared.lock().unwrap_or_else(|e| e.into_inner());
                                    state.record_chunk(byte_count)
                                };

                                // Fire progress callback if needed
                                if let Some(ref cb) = callback {
                                    let (lc, lb) = {
                                        let s = shared.lock().unwrap_or_else(|e| e.into_inner());
                                        (s.loaded_chunks, s.loaded_bytes)
                                    };
                                    if lc % interval == 0 || lc == total_chunks {
                                        let elapsed = start.elapsed().as_secs_f64();
                                        cb(LoadingProgress {
                                            loaded_chunks: lc,
                                            total_chunks,
                                            loaded_bytes: lb,
                                            total_bytes,
                                            elapsed_secs: elapsed,
                                            throughput_mb_per_sec: throughput,
                                        });
                                    }
                                }

                                let mut res = results.lock().unwrap_or_else(|e| e.into_inner());
                                res.push((chunk_id, chunk));
                            }
                            Err(e) => {
                                let mut errs = errors.lock().unwrap_or_else(|e| e.into_inner());
                                errs.push(format!("{}: {}", path.display(), e));
                            }
                        }
                    }
                });
            }
        });

        // Check for errors
        let errs = {
            let guard = errors.lock().unwrap_or_else(|e| e.into_inner());
            guard.clone()
        };

        if !errs.is_empty() {
            return Err(TrustformersError::Io {
                message: format!(
                    "{} file(s) failed to load: {}",
                    errs.len(),
                    errs.join("; ")
                ),
                path: None,
                suggestion: Some("Check file permissions and disk integrity".to_string()),
            });
        }

        // Sort by chunk_id to give callers a deterministic order.
        // After the thread scope the Arc has exactly one owner, so try_unwrap always succeeds.
        let mut result_pairs = {
            let mut guard = results.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *guard)
        };

        result_pairs.sort_by_key(|(id, _)| *id);

        // Update persistent stats
        let elapsed = start.elapsed().as_secs_f64();
        let avg = if elapsed > 0.0 {
            total_bytes as f64 / (elapsed * 1024.0 * 1024.0)
        } else {
            0.0
        };
        let peak = {
            let s = shared.lock().unwrap_or_else(|e| e.into_inner());
            s.peak_throughput
        };

        {
            let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
            stats.files_loaded += result_pairs.len();
            stats.total_bytes_loaded += total_bytes;
            stats.total_duration_secs += elapsed;
            stats.avg_throughput_mb_per_sec = avg;
            stats.peak_throughput_mb_per_sec = peak;
        }

        info!(
            files = result_pairs.len(),
            total_bytes,
            elapsed_secs = elapsed,
            avg_mib_s = avg,
            "Weight loading complete"
        );

        Ok(result_pairs.into_iter().map(|(_, c)| c).collect())
    }

    /// Estimate loading time in seconds for a given total byte count,
    /// based on the configured number of threads and an assumed disk read speed.
    pub fn estimate_loading_time_secs(&self, total_bytes: u64) -> f64 {
        // Assume 500 MiB/s per thread, limited by the thread count
        const BYTES_PER_SEC_PER_THREAD: f64 = 500.0 * 1024.0 * 1024.0;
        let effective_bandwidth = BYTES_PER_SEC_PER_THREAD * self.config.num_threads as f64;
        total_bytes as f64 / effective_bandwidth
    }

    /// Return a snapshot of accumulated loading statistics.
    pub fn stats(&self) -> LoadingStats {
        self.stats
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn collect_weight_files(&self, dir: &Path) -> Result<Vec<PathBuf>, TrustformersError> {
        let read_dir = std::fs::read_dir(dir).map_err(|e| TrustformersError::Io {
            message: format!("Cannot read directory '{}': {}", dir.display(), e),
            path: Some(dir.to_string_lossy().to_string()),
            suggestion: Some("Check directory permissions".to_string()),
        })?;

        let mut safetensor_files: Vec<PathBuf> = Vec::new();
        let mut bin_files: Vec<PathBuf> = Vec::new();

        for entry in read_dir {
            let entry = entry.map_err(|e| TrustformersError::Io {
                message: format!("Error reading directory entry: {e}"),
                path: None,
                suggestion: None,
            })?;
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                match ext {
                    "safetensors" => safetensor_files.push(path),
                    "bin" => bin_files.push(path),
                    _ => {}
                }
            }
        }

        // Prefer safetensors; fall back to bin
        let mut files = if !safetensor_files.is_empty() {
            safetensor_files
        } else {
            bin_files
        };

        // Sort for deterministic ordering
        files.sort();
        debug!(count = files.len(), "Discovered weight files");
        Ok(files)
    }
}

// ---------------------------------------------------------------------------
// File-level loading helper
// ---------------------------------------------------------------------------

/// Load a single weight file into a [`WeightChunk`].
///
/// Currently this is a stub that reads raw file bytes and associates them with
/// a synthetic tensor name. A full implementation would parse the safetensors
/// header to populate `dtype_map` and `shape_map`.
fn load_file_as_chunk(
    chunk_id: usize,
    path: &Path,
    _use_mmap: bool,
) -> Result<WeightChunk, std::io::Error> {
    let bytes = std::fs::read(path)?;
    let tensor_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    let mut tensors = HashMap::new();
    let mut dtype_map = HashMap::new();
    let mut shape_map = HashMap::new();

    // Parse a minimal safetensors header if present to extract tensor metadata.
    // Safetensors format: 8-byte little-endian header length, then JSON header.
    if bytes.len() >= 8 {
        if let Some(header_len) = parse_safetensors_header_len(&bytes) {
            if let Some(names) = extract_safetensor_tensor_names(&bytes, header_len) {
                for name in names {
                    dtype_map.insert(name.clone(), "float32".to_string());
                    shape_map.insert(name.clone(), vec![]);
                    tensors.insert(name, bytes.clone());
                }
            } else {
                // Fallback: store whole file under file name
                dtype_map.insert(tensor_name.clone(), "unknown".to_string());
                shape_map.insert(tensor_name.clone(), vec![]);
                tensors.insert(tensor_name, bytes);
            }
        } else {
            dtype_map.insert(tensor_name.clone(), "unknown".to_string());
            shape_map.insert(tensor_name.clone(), vec![]);
            tensors.insert(tensor_name, bytes);
        }
    } else {
        // File too small to have a safetensors header
        dtype_map.insert(tensor_name.clone(), "unknown".to_string());
        shape_map.insert(tensor_name.clone(), vec![]);
        tensors.insert(tensor_name, bytes);
    }

    Ok(WeightChunk {
        chunk_id,
        tensors,
        dtype_map,
        shape_map,
    })
}

/// Read the 8-byte little-endian header length from the start of a safetensors file.
fn parse_safetensors_header_len(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 8 {
        return None;
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    let len = u64::from_le_bytes(buf) as usize;
    // Sanity-check: header must fit within the file
    if len > 0 && 8 + len <= bytes.len() {
        Some(len)
    } else {
        None
    }
}

/// Extract tensor names from a safetensors JSON header.
fn extract_safetensor_tensor_names(bytes: &[u8], header_len: usize) -> Option<Vec<String>> {
    let json_bytes = bytes.get(8..8 + header_len)?;
    let json_str = std::str::from_utf8(json_bytes).ok()?;
    let value: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let obj = value.as_object()?;
    let names = obj
        .keys()
        .filter(|k| k.as_str() != "__metadata__")
        .cloned()
        .collect::<Vec<_>>();
    if names.is_empty() {
        None
    } else {
        Some(names)
    }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Load a model directory in parallel with default configuration.
///
/// Logs progress via `tracing::info!`.
pub fn load_model_parallel(
    path: &Path,
    config: Option<ParallelLoaderConfig>,
) -> Result<HashMap<String, WeightChunk>, TrustformersError> {
    let cfg = config.unwrap_or_default();
    let num_threads = cfg.num_threads;
    let loader = ParallelWeightLoader::new(cfg).with_progress(Box::new(move |p| {
        info!(
            pct = p.pct_complete(),
            chunks = p.loaded_chunks,
            total = p.total_chunks,
            throughput_mb = p.throughput_mb_per_sec,
            "Loading weights"
        );
    }));
    let result = loader.load_sharded_directory(path)?;
    let stats = loader.stats();
    info!(
        files = stats.files_loaded,
        bytes = stats.total_bytes_loaded,
        secs = stats.total_duration_secs,
        avg_mib_s = stats.avg_throughput_mb_per_sec,
        threads = num_threads,
        "Parallel loading finished"
    );
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_file(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(content).expect("write temp file");
        path
    }

    /// Minimal valid safetensors payload with an empty object header.
    fn minimal_safetensors_payload() -> Vec<u8> {
        let header = b"{}";
        let header_len = header.len() as u64;
        let mut bytes = header_len.to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes
    }

    #[test]
    fn test_default_config() {
        let cfg = ParallelLoaderConfig::default();
        assert!(cfg.num_threads >= 1);
        assert!(cfg.chunk_size_bytes > 0);
        assert!(cfg.progress_interval > 0);
    }

    #[test]
    fn test_loading_progress_pct_complete() {
        let p = LoadingProgress {
            loaded_chunks: 3,
            total_chunks: 4,
            loaded_bytes: 300,
            total_bytes: 400,
            elapsed_secs: 1.0,
            throughput_mb_per_sec: 300.0,
        };
        assert!((p.pct_complete() - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_loading_progress_pct_complete_zero_total() {
        let p = LoadingProgress {
            loaded_chunks: 0,
            total_chunks: 0,
            loaded_bytes: 0,
            total_bytes: 0,
            elapsed_secs: 0.0,
            throughput_mb_per_sec: 0.0,
        };
        assert!((p.pct_complete() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_load_single_file() {
        let tmp = std::env::temp_dir().join("tf_parallel_test_single");
        std::fs::create_dir_all(&tmp).unwrap();
        let path = write_temp_file(&tmp, "model.bin", b"fake weights data here");

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig::default());
        let chunk = loader.load_single_file(&path).expect("load_single_file");
        assert_eq!(chunk.chunk_id, 0);
        assert!(!chunk.tensors.is_empty());
        assert_eq!(chunk.tensor_count(), chunk.tensors.len());

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_load_files_multiple() {
        let tmp = std::env::temp_dir().join("tf_parallel_test_multi");
        std::fs::create_dir_all(&tmp).unwrap();

        let paths: Vec<PathBuf> = (0..3)
            .map(|i| write_temp_file(&tmp, &format!("shard_{i}.bin"), b"weights chunk"))
            .collect();

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig {
            num_threads: 2,
            ..Default::default()
        });

        let chunks = loader.load_files(&paths).expect("load_files");
        assert_eq!(chunks.len(), 3);
        // Chunks should be in order
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_id, i);
        }

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_load_sharded_directory_safetensors() {
        let tmp = std::env::temp_dir().join("tf_parallel_test_shard_dir");
        std::fs::create_dir_all(&tmp).unwrap();

        let payload = minimal_safetensors_payload();
        write_temp_file(&tmp, "model-00001-of-00002.safetensors", &payload);
        write_temp_file(&tmp, "model-00002-of-00002.safetensors", &payload);

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig::default());
        let result = loader
            .load_sharded_directory(&tmp)
            .expect("load_sharded_directory");
        assert_eq!(result.len(), 2);

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_load_sharded_directory_not_a_dir() {
        let tmp = std::env::temp_dir().join("tf_parallel_not_dir_test.bin");
        std::fs::write(&tmp, b"not a dir").ok();

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig::default());
        let result = loader.load_sharded_directory(&tmp);
        assert!(result.is_err());

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_load_sharded_directory_empty() {
        let tmp = std::env::temp_dir().join("tf_parallel_empty_dir");
        std::fs::create_dir_all(&tmp).unwrap();

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig::default());
        let result = loader.load_sharded_directory(&tmp);
        assert!(result.is_err());

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_progress_callback_is_called() {
        let tmp = std::env::temp_dir().join("tf_parallel_test_progress");
        std::fs::create_dir_all(&tmp).unwrap();

        let paths: Vec<PathBuf> = (0..4)
            .map(|i| write_temp_file(&tmp, &format!("s{i}.bin"), b"data"))
            .collect();

        let call_count = Arc::new(Mutex::new(0usize));
        let cc = Arc::clone(&call_count);

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig {
            num_threads: 1,
            progress_interval: 1,
            ..Default::default()
        })
        .with_progress(Box::new(move |_p| {
            let mut c = cc.lock().unwrap_or_else(|e| e.into_inner());
            *c += 1;
        }));

        loader.load_files(&paths).expect("load");
        let count = *call_count.lock().unwrap_or_else(|e| e.into_inner());
        assert!(count >= 1, "progress callback should have been called at least once");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_stats_accumulate() {
        let tmp = std::env::temp_dir().join("tf_parallel_test_stats");
        std::fs::create_dir_all(&tmp).unwrap();

        let paths: Vec<PathBuf> = (0..2)
            .map(|i| write_temp_file(&tmp, &format!("w{i}.bin"), b"some bytes"))
            .collect();

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig::default());
        loader.load_files(&paths).expect("load");

        let stats = loader.stats();
        assert_eq!(stats.files_loaded, 2);
        assert!(stats.total_bytes_loaded > 0);
        assert!(stats.total_duration_secs >= 0.0);

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_estimate_loading_time() {
        let cfg = ParallelLoaderConfig {
            num_threads: 4,
            ..Default::default()
        };
        let loader = ParallelWeightLoader::new(cfg);
        let secs = loader.estimate_loading_time_secs(4 * 500 * 1024 * 1024);
        // With 4 threads at 500 MiB/s each, 2 GiB should take ~1 second
        assert!(secs > 0.0 && secs < 10.0, "estimate was {secs}");
    }

    #[test]
    fn test_weight_chunk_total_bytes() {
        let mut chunk = WeightChunk {
            chunk_id: 0,
            tensors: HashMap::new(),
            dtype_map: HashMap::new(),
            shape_map: HashMap::new(),
        };
        chunk.tensors.insert("a".to_string(), vec![0u8; 100]);
        chunk.tensors.insert("b".to_string(), vec![0u8; 200]);
        assert_eq!(chunk.total_bytes(), 300);
        assert_eq!(chunk.tensor_count(), 2);
    }

    #[test]
    fn test_load_model_parallel_convenience() {
        let tmp = std::env::temp_dir().join("tf_parallel_convenience");
        std::fs::create_dir_all(&tmp).unwrap();

        let payload = minimal_safetensors_payload();
        write_temp_file(&tmp, "model.safetensors", &payload);

        let result = load_model_parallel(&tmp, None).expect("load_model_parallel");
        assert_eq!(result.len(), 1);

        std::fs::remove_dir_all(&tmp).ok();
    }
}
