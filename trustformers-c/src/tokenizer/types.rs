//! Tokenizer type definitions for TrustformeRS C API

use std::os::raw::{c_char, c_int, c_uint};
use std::ptr;

/// C-compatible tokenizer handle
pub type TrustformersTokenizer = usize;

/// C-compatible token IDs array
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersTokenIds {
    /// Array of token IDs
    pub ids: *mut c_uint,
    /// Length of the array
    pub length: usize,
    /// Capacity of the allocated array
    pub capacity: usize,
}

impl Default for TrustformersTokenIds {
    fn default() -> Self {
        Self {
            ids: ptr::null_mut(),
            length: 0,
            capacity: 0,
        }
    }
}

/// C-compatible encoding result
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersEncoding {
    /// Token IDs
    pub input_ids: TrustformersTokenIds,
    /// Attention mask
    pub attention_mask: TrustformersTokenIds,
    /// Token type IDs (optional)
    pub token_type_ids: TrustformersTokenIds,
    /// Special tokens mask (optional)
    pub special_tokens_mask: TrustformersTokenIds,
}

impl Default for TrustformersEncoding {
    fn default() -> Self {
        Self {
            input_ids: TrustformersTokenIds::default(),
            attention_mask: TrustformersTokenIds::default(),
            token_type_ids: TrustformersTokenIds::default(),
            special_tokens_mask: TrustformersTokenIds::default(),
        }
    }
}

/// C-compatible batch encoding result
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersBatchEncoding {
    /// Array of encodings
    pub encodings: *mut TrustformersEncoding,
    /// Number of encodings
    pub num_encodings: usize,
}

impl Default for TrustformersBatchEncoding {
    fn default() -> Self {
        Self {
            encodings: ptr::null_mut(),
            num_encodings: 0,
        }
    }
}

/// C-compatible tokenizer configuration
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersTokenizerConfig {
    /// Maximum sequence length
    pub max_length: c_int,
    /// Whether to pad sequences
    pub padding: c_int,
    /// Whether to truncate sequences
    pub truncation: c_int,
    /// Whether to return attention mask
    pub return_attention_mask: c_int,
    /// Whether to return token type IDs
    pub return_token_type_ids: c_int,
    /// Whether to return special tokens mask
    pub return_special_tokens_mask: c_int,
    /// Whether to add special tokens
    pub add_special_tokens: c_int,
}

impl Default for TrustformersTokenizerConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            padding: 1,                    // True
            truncation: 1,                 // True
            return_attention_mask: 1,      // True
            return_token_type_ids: 0,      // False
            return_special_tokens_mask: 0, // False
            add_special_tokens: 1,         // True
        }
    }
}

/// Tokenizer training configuration
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersTokenizerTrainingConfig {
    /// Vocabulary size
    pub vocab_size: c_uint,
    /// Minimum frequency for tokens
    pub min_frequency: c_uint,
    /// Whether to use byte-level BPE
    pub byte_level: c_int,
    /// Whether to include special tokens
    pub include_special_tokens: c_int,
    /// Path to save the trained tokenizer (optional)
    pub save_path: *const c_char,
    /// Special tokens to add
    pub pad_token: *const c_char,
    pub unk_token: *const c_char,
    pub cls_token: *const c_char,
    pub sep_token: *const c_char,
    pub mask_token: *const c_char,
    pub bos_token: *const c_char,
    pub eos_token: *const c_char,
}

impl Default for TrustformersTokenizerTrainingConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30000,
            min_frequency: 2,
            byte_level: 1,
            include_special_tokens: 1,
            save_path: std::ptr::null(),
            pad_token: std::ptr::null(),
            unk_token: std::ptr::null(),
            cls_token: std::ptr::null(),
            sep_token: std::ptr::null(),
            mask_token: std::ptr::null(),
            bos_token: std::ptr::null(),
            eos_token: std::ptr::null(),
        }
    }
}
