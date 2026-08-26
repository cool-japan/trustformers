//! Parallel model weight loading for faster startup.
//!
//! Loads model weight chunks concurrently using a thread pool, reducing
//! loading time for large models with many shards. Each shard is parsed for
//! real: `.safetensors` headers are read and every tensor is sliced out of
//! its own `data_offsets` range (memory-mapped for files at or above
//! `MMAP_THRESHOLD_BYTES` when [`ParallelLoaderConfig::use_mmap`] is set);
//! `.gguf` files go through `trustformers-models`' real `GGUFLoader`; legacy
//! `.bin`/`.pt`/`.pth` PyTorch checkpoints go through `trustformers-core`'s
//! real ZIP + pickle `PyTorchReader`. A file that doesn't parse as a real
//! checkpoint in its format is a [`std::io::Error`], never a fabricated chunk.
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
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{debug, info};
use trustformers_core::traits::WeightReader as _;
use trustformers_core::utils::weight_loading::PyTorchReader;
use trustformers_models::{GGUFLoader, WeightLoader as _};

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
                suggestion: Some(
                    "Provide a path to a model directory containing weight files".to_string(),
                ),
            });
        }

        let files = self.collect_weight_files(dir)?;
        if files.is_empty() {
            return Err(TrustformersError::Io {
                message: format!("No weight files found in '{}'", dir.display()),
                path: Some(dir.to_string_lossy().to_string()),
                suggestion: Some(
                    "Ensure the directory contains .safetensors or .bin weight files".to_string(),
                ),
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
    pub fn load_files(&self, files: &[PathBuf]) -> Result<Vec<WeightChunk>, TrustformersError> {
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
                                    let mut state =
                                        shared.lock().unwrap_or_else(|e| e.into_inner());
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
                            },
                            Err(e) => {
                                let mut errs = errors.lock().unwrap_or_else(|e| e.into_inner());
                                errs.push(format!("{}: {}", path.display(), e));
                            },
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
                message: format!("{} file(s) failed to load: {}", errs.len(), errs.join("; ")),
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
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
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
        let mut gguf_files: Vec<PathBuf> = Vec::new();
        let mut bin_files: Vec<PathBuf> = Vec::new();

        for entry in read_dir {
            let entry = entry.map_err(|e| TrustformersError::Io {
                message: format!("Error reading directory entry: {e}"),
                path: None,
                suggestion: None,
            })?;
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                match ext.to_ascii_lowercase().as_str() {
                    "safetensors" => safetensor_files.push(path),
                    "gguf" => gguf_files.push(path),
                    "bin" | "pt" | "pth" => bin_files.push(path),
                    _ => {},
                }
            }
        }

        // Prefer safetensors, then GGUF, then legacy PyTorch checkpoints.
        let mut files = if !safetensor_files.is_empty() {
            safetensor_files
        } else if !gguf_files.is_empty() {
            gguf_files
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

/// Threshold above which `use_mmap` switches a safetensors file from a plain
/// `std::fs::read` to a memory-mapped read, matching [`ParallelLoaderConfig::use_mmap`]'s
/// documented "large files (>= 256 MiB)" behavior.
const MMAP_THRESHOLD_BYTES: u64 = 256 * 1024 * 1024;

/// Owned or memory-mapped file bytes, so the safetensors header/tensor parsing
/// below can work against a single `&[u8]` view regardless of which backing
/// storage was used to read the file.
enum FileBytes {
    Mapped(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for FileBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            FileBytes::Mapped(mapping) => &mapping[..],
            FileBytes::Owned(bytes) => &bytes[..],
        }
    }
}

/// Read a file's bytes, memory-mapping it when `use_mmap` is set and the file
/// is at least [`MMAP_THRESHOLD_BYTES`] large.
fn read_file_bytes(path: &Path, use_mmap: bool) -> io::Result<FileBytes> {
    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();

    if use_mmap && len >= MMAP_THRESHOLD_BYTES {
        // SAFETY: mirrors the same pattern used by
        // `trustformers-models`' `MemoryMappedLoader` — the file is a
        // checkpoint on local disk that is not expected to be mutated by
        // another process while it is being loaded. The kernel guarantees the
        // mapping stays valid for `file`'s lifetime; we drop `file` only after
        // the mapping outlives the need for the raw handle.
        let mapping = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| io::Error::new(e.kind(), format!("failed to mmap the file: {e}")))?;
        Ok(FileBytes::Mapped(mapping))
    } else {
        drop(file);
        Ok(FileBytes::Owned(std::fs::read(path)?))
    }
}

fn invalid_data(path: &Path, message: impl std::fmt::Display) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("'{}': {message}", path.display()),
    )
}

/// Load a single weight file into a [`WeightChunk`], dispatching on extension
/// to a real parser for each supported format. Every tensor's `dtype_map` and
/// `shape_map` entry reflects what was actually parsed from the file — never a
/// placeholder — and `tensors` holds exactly that tensor's bytes, not the
/// whole file.
fn load_file_as_chunk(
    chunk_id: usize,
    path: &Path,
    use_mmap: bool,
) -> Result<WeightChunk, io::Error> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match extension.as_str() {
        "safetensors" => load_safetensors_chunk(chunk_id, path, use_mmap),
        "gguf" => load_gguf_chunk(chunk_id, path),
        "bin" | "pt" | "pth" => load_pytorch_chunk(chunk_id, path),
        other => Err(invalid_data(
            path,
            format!(
                "unsupported weight file extension '{other}'; expected .safetensors, .gguf, \
                 .bin, .pt, or .pth"
            ),
        )),
    }
}

/// One tensor's metadata as declared by a safetensors header, with `start`/`end`
/// byte offsets relative to the start of the tensor-data section (i.e. as the
/// header itself expresses `data_offsets`).
struct SafetensorsEntry {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// Byte size of one element of a safetensors dtype token, when known.
///
/// Returns `None` for unrecognized tokens so callers can skip the size
/// cross-check rather than reject a file using a dtype newer than this list.
fn safetensors_dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" => Some(1),
        "I16" | "U16" | "F16" | "BF16" => Some(2),
        "I32" | "U32" | "F32" => Some(4),
        "I64" | "U64" | "F64" => Some(8),
        _ => None,
    }
}

/// Parse a safetensors file's header into `(data_start, entries)`, validating
/// every tensor's declared byte range as it goes:
///
/// * `data_offsets` must be in-bounds for the file and `start <= end`;
/// * when the dtype is recognized, `end - start` must equal
///   `product(shape) * dtype_size` — a mismatch means the header is
///   internally inconsistent (corrupt or hand-edited), not just unusual;
/// * no two tensors may claim overlapping byte ranges.
///
/// Any violation is a parse error, never a truncated or best-effort chunk.
fn parse_safetensors_entries(
    bytes: &[u8],
) -> Result<(usize, HashMap<String, SafetensorsEntry>), String> {
    let header_len =
        parse_safetensors_header_len(bytes).ok_or("missing or malformed safetensors header")?;
    let data_start = 8 + header_len;
    let data_len = bytes.len().saturating_sub(data_start);

    let json_bytes = &bytes[8..data_start];
    let json_str = std::str::from_utf8(json_bytes)
        .map_err(|e| format!("safetensors header is not valid UTF-8: {e}"))?;
    let value: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| format!("safetensors header is not valid JSON: {e}"))?;
    let obj = value
        .as_object()
        .ok_or_else(|| "safetensors header is not a JSON object".to_string())?;

    let mut entries = HashMap::with_capacity(obj.len());
    for (name, entry) in obj {
        if name == "__metadata__" {
            continue;
        }
        let entry_obj = entry
            .as_object()
            .ok_or_else(|| format!("tensor '{name}' header entry is not a JSON object"))?;

        let dtype = entry_obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("tensor '{name}' is missing a string 'dtype' field"))?
            .to_string();

        let shape_values = entry_obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("tensor '{name}' is missing an array 'shape' field"))?;
        let mut shape = Vec::with_capacity(shape_values.len());
        for v in shape_values {
            let dim = v
                .as_u64()
                .ok_or_else(|| format!("tensor '{name}' has a non-integer shape entry"))?;
            shape.push(dim as usize);
        }

        let offsets = entry_obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("tensor '{name}' is missing a 'data_offsets' field"))?;
        if offsets.len() != 2 {
            return Err(format!(
                "tensor '{name}' has a 'data_offsets' field with {} elements, expected 2",
                offsets.len()
            ));
        }
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| format!("tensor '{name}' has a non-integer data_offsets[0]"))?
            as usize;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| format!("tensor '{name}' has a non-integer data_offsets[1]"))?
            as usize;

        if start > end {
            return Err(format!(
                "tensor '{name}' has data_offsets start ({start}) after end ({end})"
            ));
        }
        if end > data_len {
            return Err(format!(
                "tensor '{name}' data_offsets end ({end}) exceeds the tensor data region \
                 ({data_len} bytes)"
            ));
        }

        if let Some(dtype_size) = safetensors_dtype_size(&dtype) {
            let element_count: usize = shape.iter().product();
            let expected_bytes = element_count * dtype_size;
            let actual_bytes = end - start;
            if expected_bytes != actual_bytes {
                return Err(format!(
                    "tensor '{name}' declares shape {shape:?} and dtype {dtype} \
                     ({expected_bytes} bytes) but data_offsets span {actual_bytes} bytes"
                ));
            }
        }

        entries.insert(
            name.clone(),
            SafetensorsEntry {
                dtype,
                shape,
                start,
                end,
            },
        );
    }

    // Reject overlapping tensor regions: a well-formed writer never emits
    // these, so an overlap means the header was corrupted or tampered with.
    let mut spans: Vec<(usize, usize, &str)> =
        entries.iter().map(|(n, e)| (e.start, e.end, n.as_str())).collect();
    spans.sort_by_key(|(start, _, _)| *start);
    for pair in spans.windows(2) {
        let (_, prev_end, prev_name) = pair[0];
        let (next_start, _, next_name) = pair[1];
        if next_start < prev_end {
            return Err(format!(
                "tensors '{prev_name}' and '{next_name}' have overlapping data_offsets"
            ));
        }
    }

    Ok((data_start, entries))
}

/// Load a `.safetensors` file, slicing each tensor's own byte range out of the
/// (possibly memory-mapped) file rather than cloning the whole file per tensor.
fn load_safetensors_chunk(
    chunk_id: usize,
    path: &Path,
    use_mmap: bool,
) -> Result<WeightChunk, io::Error> {
    let file_bytes = read_file_bytes(path, use_mmap)?;
    let bytes: &[u8] = &file_bytes;

    let (data_start, entries) =
        parse_safetensors_entries(bytes).map_err(|msg| invalid_data(path, msg))?;
    if entries.is_empty() {
        return Err(invalid_data(path, "file contains no tensors"));
    }

    let mut tensors = HashMap::with_capacity(entries.len());
    let mut dtype_map = HashMap::with_capacity(entries.len());
    let mut shape_map = HashMap::with_capacity(entries.len());
    for (name, entry) in entries {
        let slice = &bytes[data_start + entry.start..data_start + entry.end];
        tensors.insert(name.clone(), slice.to_vec());
        dtype_map.insert(name.clone(), entry.dtype);
        shape_map.insert(name, entry.shape);
    }

    Ok(WeightChunk {
        chunk_id,
        tensors,
        dtype_map,
        shape_map,
    })
}

/// Pack an f32 tensor's values into little-endian raw bytes.
fn f32_values_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Load a `.gguf` file via `trustformers-models`' real `GGUFLoader`, which
/// parses the actual header/tensor-info table and dequantizes each tensor
/// (F32, F16, or a ggml k-quant format) to f32. `dtype_map` records "F32"
/// because that dequantized representation — not the on-disk quantization —
/// is what ends up in `tensors`.
fn load_gguf_chunk(chunk_id: usize, path: &Path) -> Result<WeightChunk, io::Error> {
    let mut loader = GGUFLoader::new(path)
        .map_err(|e| invalid_data(path, format!("failed to open GGUF file: {e}")))?;
    let names = loader
        .list_tensors()
        .map_err(|e| invalid_data(path, format!("failed to list GGUF tensors: {e}")))?;
    if names.is_empty() {
        return Err(invalid_data(path, "file contains no tensors"));
    }

    let mut tensors = HashMap::with_capacity(names.len());
    let mut dtype_map = HashMap::with_capacity(names.len());
    let mut shape_map = HashMap::with_capacity(names.len());
    for name in names {
        let tensor = loader.load_tensor(&name).map_err(|e| {
            invalid_data(path, format!("failed to dequantize tensor '{name}': {e}"))
        })?;
        let shape = tensor.shape();
        let values = tensor.to_vec_f32().map_err(|e| {
            invalid_data(
                path,
                format!("tensor '{name}' could not be read as f32: {e}"),
            )
        })?;
        tensors.insert(name.clone(), f32_values_to_le_bytes(&values));
        dtype_map.insert(name.clone(), "F32".to_string());
        shape_map.insert(name, shape);
    }

    Ok(WeightChunk {
        chunk_id,
        tensors,
        dtype_map,
        shape_map,
    })
}

/// Load a legacy PyTorch checkpoint (`.bin`, `.pt`, `.pth`) via
/// `trustformers-core`'s real ZIP + pickle `PyTorchReader`. A file that is not
/// an actual PyTorch checkpoint (e.g. an arbitrary blob with a `.bin`
/// extension) is rejected with a structured error rather than being wrapped
/// up as a fabricated single-tensor chunk.
fn load_pytorch_chunk(chunk_id: usize, path: &Path) -> Result<WeightChunk, io::Error> {
    let mut reader = PyTorchReader::from_file(path)
        .map_err(|e| invalid_data(path, format!("failed to parse PyTorch checkpoint: {e}")))?;
    let names = reader.list_tensors();
    if names.is_empty() {
        return Err(invalid_data(path, "checkpoint contains no tensors"));
    }

    let mut tensors = HashMap::with_capacity(names.len());
    let mut dtype_map = HashMap::with_capacity(names.len());
    let mut shape_map = HashMap::with_capacity(names.len());
    for name in names {
        let tensor = reader
            .read_tensor(&name)
            .map_err(|e| invalid_data(path, format!("failed to read tensor '{name}': {e}")))?;
        let shape = tensor.shape();
        let values = tensor.to_vec_f32().map_err(|e| {
            invalid_data(
                path,
                format!("tensor '{name}' could not be read as f32: {e}"),
            )
        })?;
        tensors.insert(name.clone(), f32_values_to_le_bytes(&values));
        dtype_map.insert(name.clone(), "F32".to_string());
        shape_map.insert(name, shape);
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

    /// Little-endian byte encoding of a slice of f32 values.
    fn f32_le(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Build a real safetensors payload (8-byte header length + JSON header +
    /// tensor data) from `(name, dtype, shape, raw_le_bytes)` entries, laying
    /// each tensor's bytes back to back and recording the resulting
    /// `data_offsets` — i.e. exactly what a real safetensors writer produces.
    fn build_safetensors_payload(entries: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for &(name, dtype, shape, bytes) in entries {
            let start = data.len();
            data.extend_from_slice(bytes);
            let end = data.len();
            header.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }
        let header_bytes =
            serde_json::to_vec(&serde_json::Value::Object(header)).expect("serialize header");
        let mut out = (header_bytes.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        out
    }

    /// Valid, non-empty single-tensor safetensors payload: one F32 tensor
    /// named "weight" with shape `[2]` and values `[1.0, 2.0]`.
    fn minimal_safetensors_payload() -> Vec<u8> {
        let bytes = f32_le(&[1.0, 2.0]);
        build_safetensors_payload(&[("weight", "F32", &[2], &bytes)])
    }

    /// Build a real, minimal GGUF file (magic, header, tensor-info table, then
    /// an alignment-padded tensor-data section) with the given
    /// `(name, shape, f32 values)` tensors, all stored as ggml type 0 (F32) so
    /// no dequantization arithmetic is needed to predict the expected output.
    /// Byte layout mirrors `GGUFLoader::{read_header,read_tensor_info}` in
    /// `trustformers-models` exactly (see that module for the reader side).
    fn build_minimal_gguf(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes()); // version
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); // tensor_count
        out.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count (0 => default 32-byte alignment)

        let mut running_offset: u64 = 0;
        let mut data_section = Vec::new();
        for &(name, shape, values) in tensors {
            let name_bytes = name.as_bytes();
            out.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(name_bytes);
            out.extend_from_slice(&(shape.len() as u32).to_le_bytes()); // n_dims
            for dim in shape {
                out.extend_from_slice(&(*dim as u64).to_le_bytes());
            }
            out.extend_from_slice(&0u32.to_le_bytes()); // ggml_type = 0 (F32)
            out.extend_from_slice(&running_offset.to_le_bytes()); // offset into data section

            let value_bytes = f32_le(values);
            running_offset += value_bytes.len() as u64;
            data_section.extend_from_slice(&value_bytes);
        }

        // Tensor data starts at the next 32-byte boundary after the info table.
        let info_end = out.len() as u64;
        let alignment: u64 = 32;
        let data_start = info_end.div_ceil(alignment) * alignment;
        out.resize(data_start as usize, 0u8);
        out.extend_from_slice(&data_section);
        out
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
        let path = write_temp_file(&tmp, "model.safetensors", &minimal_safetensors_payload());

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig::default());
        let chunk = loader.load_single_file(&path).expect("load_single_file");
        assert_eq!(chunk.chunk_id, 0);
        assert!(!chunk.tensors.is_empty());
        assert_eq!(chunk.tensor_count(), chunk.tensors.len());
        // Real shape/dtype from the header, not the empty/hardcoded placeholders
        // the old implementation always produced.
        assert_eq!(chunk.shape_map.get("weight"), Some(&vec![2usize]));
        assert_eq!(chunk.dtype_map.get("weight"), Some(&"F32".to_string()));

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_load_files_multiple() {
        let tmp = std::env::temp_dir().join("tf_parallel_test_multi");
        std::fs::create_dir_all(&tmp).unwrap();

        let paths: Vec<PathBuf> = (0..3)
            .map(|i| {
                let bytes = f32_le(&[i as f32]);
                let payload = build_safetensors_payload(&[("w", "F32", &[1], &bytes)]);
                write_temp_file(&tmp, &format!("shard_{i}.safetensors"), &payload)
            })
            .collect();

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig {
            num_threads: 2,
            ..Default::default()
        });

        let chunks = loader.load_files(&paths).expect("load_files");
        assert_eq!(chunks.len(), 3);
        // Chunks should be in order, and each should carry its own file's
        // tensor value rather than a shared whole-file clone.
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_id, i);
            let bytes = chunk.tensors.get("w").expect("tensor 'w'");
            assert_eq!(bytes, &f32_le(&[i as f32]));
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
        let result = loader.load_sharded_directory(&tmp).expect("load_sharded_directory");
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
            .map(|i| {
                let bytes = f32_le(&[i as f32]);
                let payload = build_safetensors_payload(&[("w", "F32", &[1], &bytes)]);
                write_temp_file(&tmp, &format!("s{i}.safetensors"), &payload)
            })
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
        assert!(
            count >= 1,
            "progress callback should have been called at least once"
        );

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_stats_accumulate() {
        let tmp = std::env::temp_dir().join("tf_parallel_test_stats");
        std::fs::create_dir_all(&tmp).unwrap();

        let paths: Vec<PathBuf> = (0..2)
            .map(|i| {
                let bytes = f32_le(&[i as f32, i as f32 + 1.0]);
                let payload = build_safetensors_payload(&[("w", "F32", &[2], &bytes)]);
                write_temp_file(&tmp, &format!("w{i}.safetensors"), &payload)
            })
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

    // -----------------------------------------------------------------------
    // Regression tests: real per-tensor slicing, shapes and dtypes.
    //
    // The old `load_file_as_chunk` cloned the *entire file* into every tensor
    // name it found, always reported dtype "float32", and always reported an
    // empty shape. These tests would all have failed against that code: the
    // shape/dtype assertions because they were always `vec![]`/"float32", and
    // the "distinct tensors" / "shorter than the file" assertions because
    // every tensor's bytes were byte-identical to the whole file.
    // -----------------------------------------------------------------------

    #[test]
    fn test_safetensors_multi_tensor_real_shapes_dtypes_and_slices() {
        let tmp = std::env::temp_dir().join("tf_parallel_test_multi_tensor_real");
        std::fs::create_dir_all(&tmp).unwrap();

        let a_bytes = f32_le(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // F32, shape [2, 3] -> 24 bytes
                                                               // IEEE-754 half-precision bit patterns for 1.0 and -2.0, little-endian.
        let b_bytes: Vec<u8> = vec![0x00, 0x3C, 0x00, 0xC0]; // F16, shape [2] -> 4 bytes
        let payload = build_safetensors_payload(&[
            ("a", "F32", &[2, 3], &a_bytes),
            ("b", "F16", &[2], &b_bytes),
        ]);
        let path = write_temp_file(&tmp, "multi.safetensors", &payload);

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig::default());
        let chunk = loader.load_single_file(&path).expect("load_single_file");

        assert_eq!(chunk.shape_map.get("a"), Some(&vec![2usize, 3usize]));
        assert_eq!(chunk.dtype_map.get("a"), Some(&"F32".to_string()));
        assert_eq!(chunk.tensors.get("a"), Some(&a_bytes));

        assert_eq!(chunk.shape_map.get("b"), Some(&vec![2usize]));
        assert_eq!(chunk.dtype_map.get("b"), Some(&"F16".to_string()));
        assert_eq!(chunk.tensors.get("b"), Some(&b_bytes));

        // The old implementation cloned the *whole file* for every tensor
        // name, so "a" and "b" would have been byte-identical and each as
        // long as the file itself.
        assert_ne!(chunk.tensors["a"], chunk.tensors["b"]);
        assert_eq!(chunk.tensors["a"].len(), 24);
        assert_eq!(chunk.tensors["b"].len(), 4);
        let file_len = std::fs::metadata(&path).expect("metadata").len() as usize;
        assert!(chunk.tensors["a"].len() < file_len);
        assert!(chunk.tensors["b"].len() < file_len);

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_safetensors_rejects_shape_dtype_size_mismatch() {
        let tmp = std::env::temp_dir().join("tf_parallel_test_size_mismatch");
        std::fs::create_dir_all(&tmp).unwrap();

        // Declares shape [4] (16 bytes of F32) but the data section only holds
        // the 8 bytes actually written below.
        let bytes = f32_le(&[1.0, 2.0]);
        let payload = build_safetensors_payload(&[("bad", "F32", &[4], &bytes)]);
        let path = write_temp_file(&tmp, "bad.safetensors", &payload);

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig::default());
        let err = loader.load_single_file(&path).expect_err("size mismatch must be rejected");
        assert!(err.to_string().contains("bad"), "{err}");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_safetensors_rejects_out_of_bounds_offsets() {
        let tmp = std::env::temp_dir().join("tf_parallel_test_oob");
        std::fs::create_dir_all(&tmp).unwrap();

        let bytes = f32_le(&[1.0, 2.0]);
        let mut payload = build_safetensors_payload(&[("t", "F32", &[2], &bytes)]);
        // Truncate the data section after building a structurally valid file,
        // so the header's data_offsets point past the end of the file.
        let new_len = payload.len() - 4;
        payload.truncate(new_len);
        let path = write_temp_file(&tmp, "oob.safetensors", &payload);

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig::default());
        let err = loader
            .load_single_file(&path)
            .expect_err("out-of-bounds offsets must be rejected");
        assert!(err.to_string().contains("exceeds"), "{err}");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_safetensors_rejects_overlapping_offsets() {
        let tmp = std::env::temp_dir().join("tf_parallel_test_overlap");
        std::fs::create_dir_all(&tmp).unwrap();

        let data = f32_le(&[1.0, 2.0, 3.0, 4.0]); // 16 bytes total
                                                  // Two tensors both individually size-consistent (8 bytes each for a
                                                  // 2-element F32 tensor), but claiming overlapping byte ranges: [0,8)
                                                  // and [4,12).
        let header = serde_json::json!({
            "x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "y": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
        });
        let header_bytes = serde_json::to_vec(&header).expect("serialize header");
        let mut payload = (header_bytes.len() as u64).to_le_bytes().to_vec();
        payload.extend_from_slice(&header_bytes);
        payload.extend_from_slice(&data);
        let path = write_temp_file(&tmp, "overlap.safetensors", &payload);

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig::default());
        let err = loader
            .load_single_file(&path)
            .expect_err("overlapping offsets must be rejected");
        assert!(err.to_string().contains("overlap"), "{err}");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_safetensors_rejects_empty_header() {
        let tmp = std::env::temp_dir().join("tf_parallel_test_empty_header");
        std::fs::create_dir_all(&tmp).unwrap();

        let header = b"{}";
        let mut payload = (header.len() as u64).to_le_bytes().to_vec();
        payload.extend_from_slice(header);
        let path = write_temp_file(&tmp, "empty.safetensors", &payload);

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig::default());
        let err = loader
            .load_single_file(&path)
            .expect_err("a header with no tensors must be rejected");
        assert!(err.to_string().contains("no tensors"), "{err}");

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// Regression test: the old implementation treated *any* readable file as
    /// a weight file, wrapping arbitrary bytes up as a fabricated single
    /// "unknown"-dtype, empty-shape tensor and reporting success. A `.bin`
    /// file that is not really a PyTorch checkpoint must now fail with a
    /// structured error instead of a silently-invented chunk.
    #[test]
    fn test_bin_garbage_content_is_rejected_not_fabricated() {
        let tmp = std::env::temp_dir().join("tf_parallel_test_bin_garbage");
        std::fs::create_dir_all(&tmp).unwrap();
        let path = write_temp_file(&tmp, "model.bin", b"this is not a pytorch checkpoint");

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig::default());
        let err = loader
            .load_single_file(&path)
            .expect_err("garbage .bin content must not be silently accepted");
        assert!(err.to_string().contains("PyTorch"), "{err}");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_unsupported_extension_is_rejected() {
        let tmp = std::env::temp_dir().join("tf_parallel_test_unsupported_ext");
        std::fs::create_dir_all(&tmp).unwrap();
        let path = write_temp_file(&tmp, "model.weights", b"whatever");

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig::default());
        let err = loader
            .load_single_file(&path)
            .expect_err("unrecognized extension must be rejected");
        assert!(err.to_string().contains("unsupported"), "{err}");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_gguf_real_tensor_round_trip() {
        let tmp = std::env::temp_dir().join("tf_parallel_test_gguf");
        std::fs::create_dir_all(&tmp).unwrap();

        let payload = build_minimal_gguf(&[
            ("alpha", &[3], &[1.0, 2.0, 3.0]),
            ("beta", &[2], &[-4.5, 8.25]),
        ]);
        let path = write_temp_file(&tmp, "model.gguf", &payload);

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig::default());
        let chunk = loader.load_single_file(&path).expect("load_single_file (gguf)");

        assert_eq!(chunk.shape_map.get("alpha"), Some(&vec![3usize]));
        assert_eq!(chunk.dtype_map.get("alpha"), Some(&"F32".to_string()));
        assert_eq!(chunk.tensors.get("alpha"), Some(&f32_le(&[1.0, 2.0, 3.0])));

        assert_eq!(chunk.shape_map.get("beta"), Some(&vec![2usize]));
        assert_eq!(chunk.tensors.get("beta"), Some(&f32_le(&[-4.5, 8.25])));

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_collect_weight_files_prefers_safetensors_over_gguf_and_bin() {
        let tmp = std::env::temp_dir().join("tf_parallel_test_format_priority");
        std::fs::create_dir_all(&tmp).unwrap();
        write_temp_file(&tmp, "model.safetensors", &minimal_safetensors_payload());
        write_temp_file(
            &tmp,
            "model.gguf",
            &build_minimal_gguf(&[("t", &[1], &[1.0])]),
        );
        write_temp_file(&tmp, "model.bin", b"irrelevant");

        let loader = ParallelWeightLoader::new(ParallelLoaderConfig::default());
        let files = loader.collect_weight_files(&tmp).expect("collect_weight_files");
        assert_eq!(files.len(), 1);
        assert_eq!(
            files[0].extension().and_then(|e| e.to_str()),
            Some("safetensors")
        );

        std::fs::remove_dir_all(&tmp).ok();
    }
}
