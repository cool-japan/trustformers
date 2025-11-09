//! String Interning System for Memory Optimization
//!
//! This module provides a comprehensive string interning system that reduces memory usage
//! and improves performance for repeated string operations by storing strings once and
//! referencing them by ID.

use super::types::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once};

/// String interning system for frequently used strings
/// This reduces memory usage and improves performance for repeated string operations
#[derive(Debug, Clone)]
pub struct StringInterner {
    /// Map from string content to interned ID
    string_to_id: Arc<Mutex<HashMap<String, u32>>>,
    /// Map from interned ID to string content
    id_to_string: Arc<Mutex<HashMap<u32, Arc<String>>>>,
    /// Next available ID
    next_id: Arc<Mutex<u32>>,
    /// Usage statistics for optimization
    usage_stats: Arc<Mutex<HashMap<u32, StringUsageStats>>>,
}

impl StringInterner {
    /// Create a new string interner
    pub fn new() -> Self {
        Self {
            string_to_id: Arc::new(Mutex::new(HashMap::new())),
            id_to_string: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(1)), // Start from 1, reserve 0 for null
            usage_stats: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Intern a string and return its ID
    pub fn intern(&self, s: &str) -> u32 {
        let mut string_to_id = self.string_to_id.lock().unwrap();
        let mut id_to_string = self.id_to_string.lock().unwrap();
        let mut usage_stats = self.usage_stats.lock().unwrap();

        if let Some(&id) = string_to_id.get(s) {
            // String already interned, update usage stats
            if let Some(stats) = usage_stats.get_mut(&id) {
                stats.update_access();
            }
            return id;
        }

        // Intern new string
        let mut next_id = self.next_id.lock().unwrap();
        let id = *next_id;
        *next_id += 1;
        drop(next_id);

        let string_arc = Arc::new(s.to_string());
        string_to_id.insert(s.to_string(), id);
        id_to_string.insert(id, string_arc);

        // Initialize usage stats
        usage_stats.insert(id, StringUsageStats::new(s.len()));

        id
    }

    /// Get string by ID
    pub fn get(&self, id: u32) -> Option<Arc<String>> {
        let id_to_string = self.id_to_string.lock().unwrap();
        let mut usage_stats = self.usage_stats.lock().unwrap();

        if let Some(string_arc) = id_to_string.get(&id) {
            // Update usage stats
            if let Some(stats) = usage_stats.get_mut(&id) {
                stats.update_access();
            }
            Some(string_arc.clone())
        } else {
            None
        }
    }

    /// Get ID for a string without interning it
    pub fn get_id(&self, s: &str) -> Option<u32> {
        let string_to_id = self.string_to_id.lock().unwrap();
        string_to_id.get(s).copied()
    }

    /// Check if a string is interned
    pub fn contains(&self, s: &str) -> bool {
        let string_to_id = self.string_to_id.lock().unwrap();
        string_to_id.contains_key(s)
    }

    /// Get statistics about interned strings
    pub fn get_statistics(&self) -> StringInternerStats {
        let string_to_id = self.string_to_id.lock().unwrap();
        let usage_stats = self.usage_stats.lock().unwrap();

        let total_strings = string_to_id.len();
        let total_memory = usage_stats.values().map(|stats| stats.size_bytes).sum();
        let total_accesses = usage_stats.values().map(|stats| stats.access_count).sum();

        // Find most frequently used strings
        let mut frequent_strings = usage_stats
            .iter()
            .map(|(&id, stats)| (id, stats.access_count))
            .collect::<Vec<_>>();
        frequent_strings.sort_by(|a, b| b.1.cmp(&a.1));
        frequent_strings.truncate(10); // Top 10

        StringInternerStats {
            total_strings,
            total_memory_bytes: total_memory,
            total_accesses,
            avg_accesses_per_string: if total_strings > 0 {
                total_accesses as f64 / total_strings as f64
            } else {
                0.0
            },
            most_frequent_strings: frequent_strings,
        }
    }

    /// Clean up rarely used strings to free memory
    pub fn cleanup_unused(&self, min_access_count: u64, max_age_seconds: u64) -> usize {
        let mut string_to_id = self.string_to_id.lock().unwrap();
        let mut id_to_string = self.id_to_string.lock().unwrap();
        let mut usage_stats = self.usage_stats.lock().unwrap();

        let now = std::time::Instant::now();
        let mut cleaned_count = 0;

        // Collect IDs to remove
        let mut ids_to_remove = Vec::new();
        for (&id, stats) in usage_stats.iter() {
            let age_seconds = now.duration_since(stats.last_access).as_secs();
            if stats.access_count < min_access_count && age_seconds > max_age_seconds {
                ids_to_remove.push(id);
            }
        }

        // Remove identified strings
        for id in ids_to_remove {
            if let Some(string_arc) = id_to_string.remove(&id) {
                string_to_id.remove(string_arc.as_ref());
                usage_stats.remove(&id);
                cleaned_count += 1;
            }
        }

        cleaned_count
    }

    /// Get memory usage breakdown
    pub fn get_memory_breakdown(&self) -> MemoryBreakdown {
        let usage_stats = self.usage_stats.lock().unwrap();

        let mut total_memory = 0;
        let mut small_strings = 0;
        let mut medium_strings = 0;
        let mut large_strings = 0;

        for stats in usage_stats.values() {
            total_memory += stats.size_bytes;
            if stats.size_bytes < memory_categories::SMALL_STRING_THRESHOLD {
                small_strings += 1;
            } else if stats.size_bytes < memory_categories::MEDIUM_STRING_THRESHOLD {
                medium_strings += 1;
            } else {
                large_strings += 1;
            }
        }

        MemoryBreakdown {
            total_memory_bytes: total_memory,
            small_strings_count: small_strings,
            medium_strings_count: medium_strings,
            large_strings_count: large_strings,
        }
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

// Global string interner instance
static mut GLOBAL_INTERNER: Option<StringInterner> = None;
static INTERNER_INIT: Once = Once::new();

/// Get the global string interner
pub fn get_global_interner() -> &'static StringInterner {
    unsafe {
        INTERNER_INIT.call_once(|| {
            GLOBAL_INTERNER = Some(StringInterner::new());
        });
        GLOBAL_INTERNER.as_ref().unwrap()
    }
}

/// Initialize the global string interner with common strings
pub fn initialize_global_interner() {
    // This call ensures the global interner is initialized
    let _interner = get_global_interner();

    // Initialize with common strings
    common_strings::init_common_strings();
}

/// Cleanup the global string interner (note: due to static lifetime, actual cleanup is limited)
pub fn cleanup_global_interner() {
    // Due to the static lifetime and Once initialization pattern,
    // we can't actually free the global interner memory.
    // This function exists for API compatibility but is essentially a no-op.
    // In a real implementation, you might want to clear the internal HashMaps
    // but keep the interner structure intact.
}

/// Common strings module for frequently used ML/AI strings
pub mod common_strings {
    use super::*;

    /// Initialize common strings that are frequently used
    pub fn init_common_strings() {
        let interner = get_global_interner();

        // Common model names
        intern_common_model_names(interner);

        // Common tensor operations
        intern_common_operations(interner);

        // Common data types
        intern_common_data_types(interner);

        // Common error messages
        intern_common_error_messages(interner);

        // Common device names
        intern_common_device_names(interner);

        // Common configuration keys
        intern_common_config_keys(interner);
    }

    fn intern_common_model_names(interner: &StringInterner) {
        let model_names = [
            "bert",
            "gpt2",
            "gpt3",
            "gpt4",
            "t5",
            "roberta",
            "distilbert",
            "electra",
            "albert",
            "xlnet",
            "transformer",
            "llama",
            "mistral",
            "claude",
            "gemini",
            "palm",
        ];

        for name in &model_names {
            interner.intern(name);
        }
    }

    fn intern_common_operations(interner: &StringInterner) {
        let operations = [
            "matmul",
            "add",
            "mul",
            "div",
            "sub",
            "softmax",
            "relu",
            "gelu",
            "tanh",
            "sigmoid",
            "layer_norm",
            "batch_norm",
            "dropout",
            "attention",
            "embedding",
            "linear",
            "conv2d",
            "pool",
            "flatten",
            "reshape",
        ];

        for op in &operations {
            interner.intern(op);
        }
    }

    fn intern_common_data_types(interner: &StringInterner) {
        let data_types = [
            "float32",
            "float16",
            "bfloat16",
            "int32",
            "int64",
            "int8",
            "uint8",
            "bool",
            "complex64",
            "complex128",
            "string",
            "tensor",
        ];

        for dtype in &data_types {
            interner.intern(dtype);
        }
    }

    fn intern_common_error_messages(interner: &StringInterner) {
        let error_messages = [
            "null_pointer",
            "out_of_memory",
            "invalid_argument",
            "model_not_found",
            "tokenizer_error",
            "cuda_error",
            "allocation_failed",
            "dimension_mismatch",
            "type_error",
            "runtime_error",
            "validation_error",
            "timeout_error",
        ];

        for error in &error_messages {
            interner.intern(error);
        }
    }

    fn intern_common_device_names(interner: &StringInterner) {
        let devices = [
            "cpu", "cuda", "gpu", "metal", "rocm", "tpu", "npu", "xpu", "cuda:0", "cuda:1",
            "cuda:2", "cuda:3", "mps", "vulkan",
        ];

        for device in &devices {
            interner.intern(device);
        }
    }

    fn intern_common_config_keys(interner: &StringInterner) {
        let config_keys = [
            "max_length",
            "batch_size",
            "num_heads",
            "hidden_size",
            "vocab_size",
            "num_layers",
            "dropout_rate",
            "learning_rate",
            "temperature",
            "top_k",
            "top_p",
            "seq_length",
            "embed_dim",
            "num_classes",
            "padding_idx",
        ];

        for key in &config_keys {
            interner.intern(key);
        }
    }

    /// Get interned ID for common strings (optimized lookup)
    pub fn get_common_string_id(name: &str) -> Option<u32> {
        get_global_interner().get_id(name)
    }

    /// Check if a string is a pre-interned common string
    pub fn is_common_string(name: &str) -> bool {
        get_global_interner().contains(name)
    }
}
