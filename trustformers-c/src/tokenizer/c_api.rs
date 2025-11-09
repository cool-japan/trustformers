//! C API functions for tokenizer operations

use anyhow::{anyhow, Result};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint, c_ulong};
use std::ptr;

use super::types::*;
use crate::error::*;
use crate::{
    c_str_to_string, core_result_to_error, core_tensor_result_to_error, result_to_error,
    string_to_c_str, trustformers_result_to_error, RESOURCE_REGISTRY,
};

use trustformers::{AutoTokenizer, Tokenizer};

/// Load a tokenizer from a pretrained model
#[no_mangle]
pub extern "C" fn trustformers_tokenizer_from_pretrained(
    model_name: *const c_char,
    tokenizer_handle: *mut TrustformersTokenizer,
) -> TrustformersError {
    if model_name.is_null() || tokenizer_handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let model_name_str = c_try!(c_str_to_string(model_name));

    // Load tokenizer
    let tokenizer_result = AutoTokenizer::from_pretrained(&model_name_str);
    let (error, tokenizer_opt) = core_tensor_result_to_error(tokenizer_result);

    if error != TrustformersError::Success {
        return error;
    }

    let tokenizer = tokenizer_opt.unwrap();

    // Register tokenizer and return handle
    let mut registry = RESOURCE_REGISTRY.write();
    let handle = registry.register_tokenizer(tokenizer);

    unsafe {
        *tokenizer_handle = handle;
    }

    TrustformersError::Success
}

/// Load a tokenizer from a local path
#[no_mangle]
pub extern "C" fn trustformers_tokenizer_from_path(
    path: *const c_char,
    tokenizer_handle: *mut TrustformersTokenizer,
) -> TrustformersError {
    if path.is_null() || tokenizer_handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let path_str = c_try!(c_str_to_string(path));

    // Load tokenizer from path
    let tokenizer_result = AutoTokenizer::from_pretrained(&path_str);
    let (error, tokenizer_opt) = core_tensor_result_to_error(tokenizer_result);

    if error != TrustformersError::Success {
        return error;
    }

    let tokenizer = tokenizer_opt.unwrap();

    // Register tokenizer and return handle
    let mut registry = RESOURCE_REGISTRY.write();
    let handle = registry.register_tokenizer(tokenizer);

    unsafe {
        *tokenizer_handle = handle;
    }

    TrustformersError::Success
}

/// Encode a single text using the tokenizer
#[no_mangle]
pub extern "C" fn trustformers_tokenizer_encode(
    tokenizer_handle: TrustformersTokenizer,
    text: *const c_char,
    config: *const TrustformersTokenizerConfig,
    encoding: *mut TrustformersEncoding,
) -> TrustformersError {
    if text.is_null() || encoding.is_null() {
        return TrustformersError::NullPointer;
    }

    let text_str = c_try!(c_str_to_string(text));

    let config = if config.is_null() {
        TrustformersTokenizerConfig::default()
    } else {
        unsafe { std::ptr::read(config) }
    };

    // Get tokenizer from registry
    let registry = RESOURCE_REGISTRY.read();
    let tokenizer_arc = match registry.get_tokenizer(tokenizer_handle) {
        Some(t) => t,
        None => return TrustformersError::InvalidParameter,
    };

    // tokenizer_arc is &Arc<AutoTokenizer>, get reference to inner AutoTokenizer
    let tokenizer = tokenizer_arc.as_ref();

    // Encode text
    let encode_result = tokenizer.encode(&text_str);

    let (error, token_output_opt) = core_tensor_result_to_error(encode_result);

    if error != TrustformersError::Success {
        return error;
    }

    let token_output = token_output_opt.unwrap();

    // Convert to C-compatible format
    let input_ids = token_output.input_ids;
    let attention_mask = if token_output.attention_mask.is_empty() {
        vec![1; input_ids.len()]
    } else {
        token_output.attention_mask
    };

    // Allocate and fill input_ids
    let input_ids_len = input_ids.len();
    let input_ids_ptr = unsafe {
        let layout = std::alloc::Layout::array::<c_uint>(input_ids_len).unwrap();
        let ptr = std::alloc::alloc(layout) as *mut c_uint;
        if ptr.is_null() {
            return TrustformersError::OutOfMemory;
        }
        for (i, &id) in input_ids.iter().enumerate() {
            *ptr.add(i) = id as c_uint;
        }
        ptr
    };

    // Allocate and fill attention_mask
    let attention_mask_len = attention_mask.len();
    let attention_mask_ptr = unsafe {
        let layout = std::alloc::Layout::array::<c_uint>(attention_mask_len).unwrap();
        let ptr = std::alloc::alloc(layout) as *mut c_uint;
        if ptr.is_null() {
            // Clean up input_ids before returning
            let layout = std::alloc::Layout::array::<c_uint>(input_ids_len).unwrap();
            std::alloc::dealloc(input_ids_ptr as *mut u8, layout);
            return TrustformersError::OutOfMemory;
        }
        for (i, &mask) in attention_mask.iter().enumerate() {
            *ptr.add(i) = mask as c_uint;
        }
        ptr
    };

    unsafe {
        (*encoding).input_ids = TrustformersTokenIds {
            ids: input_ids_ptr,
            length: input_ids_len,
            capacity: input_ids_len,
        };

        (*encoding).attention_mask = TrustformersTokenIds {
            ids: attention_mask_ptr,
            length: attention_mask_len,
            capacity: attention_mask_len,
        };

        // Initialize optional fields as empty
        (*encoding).token_type_ids = TrustformersTokenIds::default();
        (*encoding).special_tokens_mask = TrustformersTokenIds::default();
    }

    TrustformersError::Success
}

/// Encode multiple texts using the tokenizer
#[no_mangle]
pub extern "C" fn trustformers_tokenizer_encode_batch(
    tokenizer_handle: TrustformersTokenizer,
    texts: *const *const c_char,
    num_texts: usize,
    config: *const TrustformersTokenizerConfig,
    batch_encoding: *mut TrustformersBatchEncoding,
) -> TrustformersError {
    if texts.is_null() || batch_encoding.is_null() || num_texts == 0 {
        return TrustformersError::InvalidParameter;
    }

    let config = if config.is_null() {
        TrustformersTokenizerConfig::default()
    } else {
        unsafe { std::ptr::read(config) }
    };

    // Convert C string array to Rust Vec
    let mut text_strings = Vec::with_capacity(num_texts);
    for i in 0..num_texts {
        unsafe {
            let text_ptr = *texts.add(i);
            if text_ptr.is_null() {
                return TrustformersError::NullPointer;
            }
            let text_str = c_try!(c_str_to_string(text_ptr));
            text_strings.push(text_str);
        }
    }

    // Get tokenizer from registry
    let registry = RESOURCE_REGISTRY.read();
    let tokenizer_arc = match registry.get_tokenizer(tokenizer_handle) {
        Some(t) => t,
        None => return TrustformersError::InvalidParameter,
    };

    // tokenizer_arc is &Arc<AutoTokenizer>, get reference to inner AutoTokenizer
    let tokenizer = tokenizer_arc.as_ref();

    // Allocate encodings array
    let encodings_ptr = unsafe {
        let layout = std::alloc::Layout::array::<TrustformersEncoding>(num_texts).unwrap();
        let ptr = std::alloc::alloc(layout) as *mut TrustformersEncoding;
        if ptr.is_null() {
            return TrustformersError::OutOfMemory;
        }

        // Initialize all encodings to default
        for i in 0..num_texts {
            *ptr.add(i) = TrustformersEncoding::default();
        }
        ptr
    };

    // Process each text
    for (idx, text_str) in text_strings.iter().enumerate() {
        let encode_result = tokenizer.encode(text_str);

        let (error, token_output_opt) = core_tensor_result_to_error(encode_result);

        if error != TrustformersError::Success {
            // Clean up any allocated encodings before returning error
            unsafe {
                for i in 0..idx {
                    let encoding = &mut *encodings_ptr.add(i);
                    trustformers_encoding_free_contents(encoding);
                }
                let layout = std::alloc::Layout::array::<TrustformersEncoding>(num_texts).unwrap();
                std::alloc::dealloc(encodings_ptr as *mut u8, layout);
            }
            return error;
        }

        let token_output = token_output_opt.unwrap();

        // Fill encoding at current index
        unsafe {
            let encoding = &mut *encodings_ptr.add(idx);
            let fill_result = fill_encoding_from_token_output(encoding, &token_output);
            if fill_result != TrustformersError::Success {
                // Clean up and return error
                for i in 0..idx {
                    let enc = &mut *encodings_ptr.add(i);
                    trustformers_encoding_free_contents(enc);
                }
                let layout = std::alloc::Layout::array::<TrustformersEncoding>(num_texts).unwrap();
                std::alloc::dealloc(encodings_ptr as *mut u8, layout);
                return fill_result;
            }
        }
    }

    unsafe {
        (*batch_encoding).encodings = encodings_ptr;
        (*batch_encoding).num_encodings = num_texts;
    }

    TrustformersError::Success
}

/// Decode token IDs back to text
#[no_mangle]
pub extern "C" fn trustformers_tokenizer_decode(
    tokenizer_handle: TrustformersTokenizer,
    token_ids: *const c_uint,
    num_tokens: usize,
    skip_special_tokens: c_int,
    decoded_text: *mut *mut c_char,
) -> TrustformersError {
    if token_ids.is_null() || decoded_text.is_null() || num_tokens == 0 {
        return TrustformersError::InvalidParameter;
    }

    // Convert C array to Rust Vec
    let mut ids = Vec::with_capacity(num_tokens);
    unsafe {
        for i in 0..num_tokens {
            ids.push(*token_ids.add(i) as u32);
        }
    }

    // Get tokenizer from registry
    let registry = RESOURCE_REGISTRY.read();
    let tokenizer_arc = match registry.get_tokenizer(tokenizer_handle) {
        Some(t) => t,
        None => return TrustformersError::InvalidParameter,
    };

    // tokenizer_arc is &Arc<AutoTokenizer>, get reference to inner AutoTokenizer
    let tokenizer = tokenizer_arc.as_ref();

    let decode_result = tokenizer.decode(&ids);
    let (error, text_opt) = core_tensor_result_to_error(decode_result);

    if error != TrustformersError::Success {
        return error;
    }

    let text = text_opt.unwrap();
    let c_text = string_to_c_str(text);

    unsafe {
        *decoded_text = c_text;
    }

    TrustformersError::Success
}

/// Get vocabulary size of the tokenizer
#[no_mangle]
pub extern "C" fn trustformers_tokenizer_vocab_size(
    tokenizer_handle: TrustformersTokenizer,
    vocab_size: *mut c_ulong,
) -> TrustformersError {
    if vocab_size.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = RESOURCE_REGISTRY.read();
    let tokenizer_arc = match registry.get_tokenizer(tokenizer_handle) {
        Some(t) => t,
        None => return TrustformersError::InvalidParameter,
    };

    // tokenizer_arc is &Arc<AutoTokenizer>, get reference to inner AutoTokenizer
    let tokenizer = tokenizer_arc.as_ref();

    unsafe {
        *vocab_size = tokenizer.vocab_size() as c_ulong;
    }
    TrustformersError::Success
}

/// Decode multiple token ID arrays back to texts (batch decoding)
#[no_mangle]
pub extern "C" fn trustformers_tokenizer_decode_batch(
    tokenizer_handle: TrustformersTokenizer,
    token_ids_batch: *const *const c_uint,
    num_tokens_batch: *const usize,
    num_sequences: usize,
    skip_special_tokens: c_int,
    decoded_texts: *mut *mut *mut c_char,
) -> TrustformersError {
    if token_ids_batch.is_null()
        || num_tokens_batch.is_null()
        || decoded_texts.is_null()
        || num_sequences == 0
    {
        return TrustformersError::InvalidParameter;
    }

    // Get tokenizer from registry
    let registry = RESOURCE_REGISTRY.read();
    let tokenizer_arc = match registry.get_tokenizer(tokenizer_handle) {
        Some(t) => t,
        None => return TrustformersError::InvalidParameter,
    };

    // tokenizer_arc is &Arc<AutoTokenizer>, get reference to inner AutoTokenizer
    let tokenizer = tokenizer_arc.as_ref();

    // Allocate array for decoded text pointers
    let texts_ptr = unsafe {
        let layout = std::alloc::Layout::array::<*mut c_char>(num_sequences).unwrap();
        let ptr = std::alloc::alloc(layout) as *mut *mut c_char;
        if ptr.is_null() {
            return TrustformersError::OutOfMemory;
        }
        ptr
    };

    // Process each sequence
    for i in 0..num_sequences {
        unsafe {
            let token_ids_ptr = *token_ids_batch.add(i);
            let num_tokens = *num_tokens_batch.add(i);

            if token_ids_ptr.is_null() || num_tokens == 0 {
                // Clean up allocated texts and return error
                for j in 0..i {
                    let text_ptr = *texts_ptr.add(j);
                    if !text_ptr.is_null() {
                        let _ = CString::from_raw(text_ptr);
                    }
                }
                let layout = std::alloc::Layout::array::<*mut c_char>(num_sequences).unwrap();
                std::alloc::dealloc(texts_ptr as *mut u8, layout);
                return TrustformersError::InvalidParameter;
            }

            // Convert C array to Rust Vec
            let mut ids = Vec::with_capacity(num_tokens);
            for k in 0..num_tokens {
                ids.push(*token_ids_ptr.add(k) as u32);
            }

            // Decode the sequence
            let decode_result = tokenizer.decode(&ids);
            let (error, text_opt) = core_tensor_result_to_error(decode_result);

            if error != TrustformersError::Success {
                // Clean up and return error
                for j in 0..i {
                    let text_ptr = *texts_ptr.add(j);
                    if !text_ptr.is_null() {
                        let _ = CString::from_raw(text_ptr);
                    }
                }
                let layout = std::alloc::Layout::array::<*mut c_char>(num_sequences).unwrap();
                std::alloc::dealloc(texts_ptr as *mut u8, layout);
                return error;
            }

            let text = text_opt.unwrap();
            let c_text = string_to_c_str(text);
            *texts_ptr.add(i) = c_text;
        }
    }

    unsafe {
        *decoded_texts = texts_ptr;
    }

    TrustformersError::Success
}

/// Free batch decoded texts
#[no_mangle]
pub extern "C" fn trustformers_batch_decoded_texts_free(
    decoded_texts: *mut *mut c_char,
    num_sequences: usize,
) {
    if decoded_texts.is_null() || num_sequences == 0 {
        return;
    }

    unsafe {
        for i in 0..num_sequences {
            let text_ptr = *decoded_texts.add(i);
            if !text_ptr.is_null() {
                let _ = CString::from_raw(text_ptr);
            }
        }

        let layout = std::alloc::Layout::array::<*mut c_char>(num_sequences).unwrap();
        std::alloc::dealloc(decoded_texts as *mut u8, layout);
    }
}

/// Get the tokenizer's vocabulary as a list of tokens
#[no_mangle]
pub extern "C" fn trustformers_tokenizer_get_vocab(
    tokenizer_handle: TrustformersTokenizer,
    vocab_tokens: *mut *mut *mut c_char,
    vocab_size: *mut usize,
) -> TrustformersError {
    if vocab_tokens.is_null() || vocab_size.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = RESOURCE_REGISTRY.read();
    let tokenizer_arc = match registry.get_tokenizer(tokenizer_handle) {
        Some(t) => t,
        None => return TrustformersError::InvalidParameter,
    };

    // tokenizer_arc is &Arc<AutoTokenizer>, get reference to inner AutoTokenizer
    let tokenizer = tokenizer_arc.as_ref();

    let vocab = tokenizer.get_vocab();
    let size = vocab.len();

    // Allocate array for token pointers
    let tokens_ptr = unsafe {
        let layout = std::alloc::Layout::array::<*mut c_char>(size).unwrap();
        let ptr = std::alloc::alloc(layout) as *mut *mut c_char;
        if ptr.is_null() {
            return TrustformersError::OutOfMemory;
        }
        ptr
    };

    // Convert tokens to C strings
    let mut processed = 0;
    for (token, _) in vocab.iter() {
        let c_token = string_to_c_str(token.clone());
        if c_token.is_null() {
            // Clean up on failure
            unsafe {
                for i in 0..processed {
                    let ptr = *tokens_ptr.add(i);
                    if !ptr.is_null() {
                        let _ = CString::from_raw(ptr);
                    }
                }
                let layout = std::alloc::Layout::array::<*mut c_char>(size).unwrap();
                std::alloc::dealloc(tokens_ptr as *mut u8, layout);
            }
            return TrustformersError::OutOfMemory;
        }

        unsafe {
            *tokens_ptr.add(processed) = c_token;
        }
        processed += 1;
    }

    unsafe {
        *vocab_tokens = tokens_ptr;
        *vocab_size = size;
    }

    TrustformersError::Success
}

/// Free vocabulary tokens array
#[no_mangle]
pub extern "C" fn trustformers_vocab_tokens_free(
    vocab_tokens: *mut *mut c_char,
    vocab_size: usize,
) {
    if vocab_tokens.is_null() || vocab_size == 0 {
        return;
    }

    unsafe {
        for i in 0..vocab_size {
            let token_ptr = *vocab_tokens.add(i);
            if !token_ptr.is_null() {
                let _ = CString::from_raw(token_ptr);
            }
        }

        let layout = std::alloc::Layout::array::<*mut c_char>(vocab_size).unwrap();
        std::alloc::dealloc(vocab_tokens as *mut u8, layout);
    }
}

/// Convert tokens to IDs
#[no_mangle]
pub extern "C" fn trustformers_tokenizer_convert_tokens_to_ids(
    tokenizer_handle: TrustformersTokenizer,
    tokens: *const *const c_char,
    num_tokens: usize,
    token_ids: *mut *mut c_uint,
    num_ids: *mut usize,
) -> TrustformersError {
    if tokens.is_null() || token_ids.is_null() || num_ids.is_null() || num_tokens == 0 {
        return TrustformersError::InvalidParameter;
    }

    let registry = RESOURCE_REGISTRY.read();
    let tokenizer_arc = match registry.get_tokenizer(tokenizer_handle) {
        Some(t) => t,
        None => return TrustformersError::InvalidParameter,
    };

    // tokenizer_arc is &Arc<AutoTokenizer>, get reference to inner AutoTokenizer
    let tokenizer = tokenizer_arc.as_ref();

    // Convert C string array to Rust strings
    let mut token_strings = Vec::with_capacity(num_tokens);
    for i in 0..num_tokens {
        unsafe {
            let token_ptr = *tokens.add(i);
            if token_ptr.is_null() {
                return TrustformersError::NullPointer;
            }
            let token_str = c_try!(c_str_to_string(token_ptr));
            token_strings.push(token_str);
        }
    }

    // Convert tokens to IDs
    let mut ids = Vec::with_capacity(num_tokens);
    for token in &token_strings {
        if let Some(id) = tokenizer.token_to_id(token) {
            ids.push(id);
        } else {
            // Token not found, use unknown token ID if available
            if let Some(unk_id) = tokenizer.token_to_id("[UNK]") {
                ids.push(unk_id);
            } else {
                return TrustformersError::TokenizerError;
            }
        }
    }

    // Allocate C array for IDs
    let ids_ptr = unsafe {
        let layout = std::alloc::Layout::array::<c_uint>(ids.len()).unwrap();
        let ptr = std::alloc::alloc(layout) as *mut c_uint;
        if ptr.is_null() {
            return TrustformersError::OutOfMemory;
        }
        for (i, &id) in ids.iter().enumerate() {
            *ptr.add(i) = id as c_uint;
        }
        ptr
    };

    unsafe {
        *token_ids = ids_ptr;
        *num_ids = ids.len();
    }

    TrustformersError::Success
}

/// Convert IDs to tokens
#[no_mangle]
pub extern "C" fn trustformers_tokenizer_convert_ids_to_tokens(
    tokenizer_handle: TrustformersTokenizer,
    token_ids: *const c_uint,
    num_ids: usize,
    tokens: *mut *mut *mut c_char,
    num_tokens: *mut usize,
) -> TrustformersError {
    if token_ids.is_null() || tokens.is_null() || num_tokens.is_null() || num_ids == 0 {
        return TrustformersError::InvalidParameter;
    }

    let registry = RESOURCE_REGISTRY.read();
    let tokenizer_arc = match registry.get_tokenizer(tokenizer_handle) {
        Some(t) => t,
        None => return TrustformersError::InvalidParameter,
    };

    // tokenizer_arc is &Arc<AutoTokenizer>, get reference to inner AutoTokenizer
    let tokenizer = tokenizer_arc.as_ref();

    // Convert C array to Rust Vec
    let mut ids = Vec::with_capacity(num_ids);
    unsafe {
        for i in 0..num_ids {
            ids.push(*token_ids.add(i) as u32);
        }
    }

    // Allocate array for token pointers
    let tokens_ptr = unsafe {
        let layout = std::alloc::Layout::array::<*mut c_char>(num_ids).unwrap();
        let ptr = std::alloc::alloc(layout) as *mut *mut c_char;
        if ptr.is_null() {
            return TrustformersError::OutOfMemory;
        }
        ptr
    };

    // Convert IDs to tokens
    let mut processed = 0;
    for &id in &ids {
        let token = tokenizer.id_to_token(id).unwrap_or_else(|| "[UNK]".to_string());
        let c_token = string_to_c_str(token);

        if c_token.is_null() {
            // Clean up on failure
            unsafe {
                for i in 0..processed {
                    let ptr = *tokens_ptr.add(i);
                    if !ptr.is_null() {
                        let _ = CString::from_raw(ptr);
                    }
                }
                let layout = std::alloc::Layout::array::<*mut c_char>(num_ids).unwrap();
                std::alloc::dealloc(tokens_ptr as *mut u8, layout);
            }
            return TrustformersError::OutOfMemory;
        }

        unsafe {
            *tokens_ptr.add(processed) = c_token;
        }
        processed += 1;
    }

    unsafe {
        *tokens = tokens_ptr;
        *num_tokens = num_ids;
    }

    TrustformersError::Success
}

/// Get special token IDs
#[no_mangle]
pub extern "C" fn trustformers_tokenizer_get_special_tokens(
    tokenizer_handle: TrustformersTokenizer,
    pad_token_id: *mut c_uint,
    unk_token_id: *mut c_uint,
    cls_token_id: *mut c_uint,
    sep_token_id: *mut c_uint,
    mask_token_id: *mut c_uint,
    bos_token_id: *mut c_uint,
    eos_token_id: *mut c_uint,
) -> TrustformersError {
    let registry = RESOURCE_REGISTRY.read();
    let tokenizer_arc = match registry.get_tokenizer(tokenizer_handle) {
        Some(t) => t,
        None => return TrustformersError::InvalidParameter,
    };

    // tokenizer_arc is &Arc<AutoTokenizer>, get reference to inner AutoTokenizer
    let tokenizer = tokenizer_arc.as_ref();

    unsafe {
        // Set special token IDs (using common token names)
        if !pad_token_id.is_null() {
            *pad_token_id = tokenizer.token_to_id("[PAD]").unwrap_or(0);
        }
        if !unk_token_id.is_null() {
            *unk_token_id = tokenizer.token_to_id("[UNK]").unwrap_or(100);
        }
        if !cls_token_id.is_null() {
            *cls_token_id = tokenizer.token_to_id("[CLS]").unwrap_or(101);
        }
        if !sep_token_id.is_null() {
            *sep_token_id = tokenizer.token_to_id("[SEP]").unwrap_or(102);
        }
        if !mask_token_id.is_null() {
            *mask_token_id = tokenizer.token_to_id("[MASK]").unwrap_or(103);
        }
        if !bos_token_id.is_null() {
            *bos_token_id = tokenizer
                .token_to_id("<s>")
                .or_else(|| tokenizer.token_to_id("[CLS]"))
                .unwrap_or(101);
        }
        if !eos_token_id.is_null() {
            *eos_token_id = tokenizer
                .token_to_id("</s>")
                .or_else(|| tokenizer.token_to_id("[SEP]"))
                .unwrap_or(102);
        }
    }

    TrustformersError::Success
}

/// Train a new tokenizer from text files
#[no_mangle]
pub extern "C" fn trustformers_tokenizer_train_from_files(
    training_files: *const *const c_char,
    num_files: usize,
    config: *const TrustformersTokenizerTrainingConfig,
    tokenizer_handle: *mut TrustformersTokenizer,
) -> TrustformersError {
    if training_files.is_null() || tokenizer_handle.is_null() || num_files == 0 {
        return TrustformersError::InvalidParameter;
    }

    let config = if config.is_null() {
        TrustformersTokenizerTrainingConfig::default()
    } else {
        unsafe { std::ptr::read(config) }
    };

    // Convert file paths from C strings
    let mut file_paths = Vec::with_capacity(num_files);
    for i in 0..num_files {
        unsafe {
            let file_ptr = *training_files.add(i);
            if file_ptr.is_null() {
                return TrustformersError::NullPointer;
            }
            let file_path = c_try!(c_str_to_string(file_ptr));
            file_paths.push(file_path);
        }
    }

    // Create a simple BPE tokenizer for training
    // Note: This is a simplified implementation. In a real scenario,
    // you would use the tokenizers library's training capabilities.
    let tokenizer_result = create_trained_tokenizer(&file_paths, &config);
    let tokenizer = match tokenizer_result {
        Ok(t) => t,
        Err(e) => return e,
    };

    // Register tokenizer and return handle
    let mut registry = RESOURCE_REGISTRY.write();
    let handle = registry.register_tokenizer(tokenizer);

    unsafe {
        *tokenizer_handle = handle;
    }

    TrustformersError::Success
}

/// Train a tokenizer from text strings in memory
#[no_mangle]
pub extern "C" fn trustformers_tokenizer_train_from_texts(
    training_texts: *const *const c_char,
    num_texts: usize,
    config: *const TrustformersTokenizerTrainingConfig,
    tokenizer_handle: *mut TrustformersTokenizer,
) -> TrustformersError {
    if training_texts.is_null() || tokenizer_handle.is_null() || num_texts == 0 {
        return TrustformersError::InvalidParameter;
    }

    let config = if config.is_null() {
        TrustformersTokenizerTrainingConfig::default()
    } else {
        unsafe { std::ptr::read(config) }
    };

    // Convert texts from C strings
    let mut texts = Vec::with_capacity(num_texts);
    for i in 0..num_texts {
        unsafe {
            let text_ptr = *training_texts.add(i);
            if text_ptr.is_null() {
                return TrustformersError::NullPointer;
            }
            let text = c_try!(c_str_to_string(text_ptr));
            texts.push(text);
        }
    }

    // Create tokenizer from texts
    let tokenizer_result = create_trained_tokenizer_from_texts(&texts, &config);
    let tokenizer = match tokenizer_result {
        Ok(t) => t,
        Err(e) => return e,
    };

    // Register tokenizer and return handle
    let mut registry = RESOURCE_REGISTRY.write();
    let handle = registry.register_tokenizer(tokenizer);

    unsafe {
        *tokenizer_handle = handle;
    }

    TrustformersError::Success
}

/// Save a tokenizer to disk
#[no_mangle]
pub extern "C" fn trustformers_tokenizer_save(
    tokenizer_handle: TrustformersTokenizer,
    save_path: *const c_char,
) -> TrustformersError {
    if save_path.is_null() {
        return TrustformersError::NullPointer;
    }

    let path_str = c_try!(c_str_to_string(save_path));

    let registry = RESOURCE_REGISTRY.read();
    let tokenizer_arc = match registry.get_tokenizer(tokenizer_handle) {
        Some(t) => t,
        None => return TrustformersError::InvalidParameter,
    };

    // tokenizer_arc is &Arc<AutoTokenizer>, get reference to inner AutoTokenizer
    let tokenizer = tokenizer_arc.as_ref();

    // Call save_pretrained on the AutoTokenizer (TokenizerWrapper)
    let save_result = tokenizer.save_pretrained(&path_str);
    let (error, _) = core_tensor_result_to_error(save_result);
    error
}

/// Add special tokens to existing tokenizer
#[no_mangle]
pub extern "C" fn trustformers_tokenizer_add_special_tokens(
    tokenizer_handle: TrustformersTokenizer,
    special_tokens: *const *const c_char,
    num_tokens: usize,
) -> TrustformersError {
    if special_tokens.is_null() || num_tokens == 0 {
        return TrustformersError::InvalidParameter;
    }

    // Convert special tokens from C strings
    let mut tokens = Vec::with_capacity(num_tokens);
    for i in 0..num_tokens {
        unsafe {
            let token_ptr = *special_tokens.add(i);
            if token_ptr.is_null() {
                return TrustformersError::NullPointer;
            }
            let token = c_try!(c_str_to_string(token_ptr));
            tokens.push(token);
        }
    }

    let registry = RESOURCE_REGISTRY.read();
    let tokenizer_arc = match registry.get_tokenizer(tokenizer_handle) {
        Some(t) => t,
        None => return TrustformersError::InvalidParameter,
    };

    // tokenizer_arc is &Arc<AutoTokenizer>, get reference to inner AutoTokenizer
    let _tokenizer = tokenizer_arc.as_ref();

    // Note: Adding special tokens to an existing tokenizer is complex
    // and would require modification of the tokenizer's vocabulary.
    // This is a placeholder implementation.
    eprintln!("Warning: Adding special tokens to existing tokenizer not fully implemented");
    TrustformersError::FeatureNotAvailable
}

/// Destroy a tokenizer and free its resources
#[no_mangle]
pub extern "C" fn trustformers_tokenizer_destroy(
    tokenizer_handle: TrustformersTokenizer,
) -> TrustformersError {
    let mut registry = RESOURCE_REGISTRY.write();
    if registry.remove_tokenizer(tokenizer_handle) {
        TrustformersError::Success
    } else {
        TrustformersError::InvalidParameter
    }
}

/// Free encoding memory
#[no_mangle]
pub extern "C" fn trustformers_encoding_free(encoding: *mut TrustformersEncoding) {
    trustformers_encoding_free_contents(encoding);
}

/// Free batch encoding memory
#[no_mangle]
pub extern "C" fn trustformers_batch_encoding_free(batch_encoding: *mut TrustformersBatchEncoding) {
    if batch_encoding.is_null() {
        return;
    }

    unsafe {
        let batch_enc = &mut *batch_encoding;

        if !batch_enc.encodings.is_null() && batch_enc.num_encodings > 0 {
            // Free each encoding
            for i in 0..batch_enc.num_encodings {
                let encoding = &mut *batch_enc.encodings.add(i);
                trustformers_encoding_free_contents(encoding);
            }

            // Free the encodings array
            let layout =
                std::alloc::Layout::array::<TrustformersEncoding>(batch_enc.num_encodings).unwrap();
            std::alloc::dealloc(batch_enc.encodings as *mut u8, layout);
            batch_enc.encodings = ptr::null_mut();
        }

        batch_enc.num_encodings = 0;
    }
}

/// Free encoding contents
pub fn trustformers_encoding_free_contents(encoding: *mut TrustformersEncoding) {
    if encoding.is_null() {
        return;
    }

    unsafe {
        let enc = &mut *encoding;

        // Free input_ids
        if !enc.input_ids.ids.is_null() && enc.input_ids.capacity > 0 {
            let layout = std::alloc::Layout::array::<c_uint>(enc.input_ids.capacity).unwrap();
            std::alloc::dealloc(enc.input_ids.ids as *mut u8, layout);
            enc.input_ids.ids = ptr::null_mut();
        }

        // Free attention_mask
        if !enc.attention_mask.ids.is_null() && enc.attention_mask.capacity > 0 {
            let layout = std::alloc::Layout::array::<c_uint>(enc.attention_mask.capacity).unwrap();
            std::alloc::dealloc(enc.attention_mask.ids as *mut u8, layout);
            enc.attention_mask.ids = ptr::null_mut();
        }

        // Free token_type_ids if allocated
        if !enc.token_type_ids.ids.is_null() && enc.token_type_ids.capacity > 0 {
            let layout = std::alloc::Layout::array::<c_uint>(enc.token_type_ids.capacity).unwrap();
            std::alloc::dealloc(enc.token_type_ids.ids as *mut u8, layout);
            enc.token_type_ids.ids = ptr::null_mut();
        }

        // Free special_tokens_mask if allocated
        if !enc.special_tokens_mask.ids.is_null() && enc.special_tokens_mask.capacity > 0 {
            let layout =
                std::alloc::Layout::array::<c_uint>(enc.special_tokens_mask.capacity).unwrap();
            std::alloc::dealloc(enc.special_tokens_mask.ids as *mut u8, layout);
            enc.special_tokens_mask.ids = ptr::null_mut();
        }
    }
}

/// Fill encoding from token output
pub fn fill_encoding_from_token_output(
    encoding: &mut TrustformersEncoding,
    token_output: &trustformers::TokenizedInput,
) -> TrustformersError {
    let input_ids = &token_output.input_ids;
    let len = input_ids.len();
    let attention_mask: Vec<u32> = token_output.attention_mask.iter().map(|&x| x as u32).collect();

    // Allocate and fill input_ids
    let input_ids_len = input_ids.len();
    let input_ids_ptr = unsafe {
        let layout = std::alloc::Layout::array::<c_uint>(input_ids_len).unwrap();
        let ptr = std::alloc::alloc(layout) as *mut c_uint;
        if ptr.is_null() {
            return TrustformersError::OutOfMemory;
        }
        for (i, &id) in input_ids.iter().enumerate() {
            *ptr.add(i) = id as c_uint;
        }
        ptr
    };

    // Allocate and fill attention_mask
    let attention_mask_len = attention_mask.len();
    let attention_mask_ptr = unsafe {
        let layout = std::alloc::Layout::array::<c_uint>(attention_mask_len).unwrap();
        let ptr = std::alloc::alloc(layout) as *mut c_uint;
        if ptr.is_null() {
            // Clean up input_ids
            let layout = std::alloc::Layout::array::<c_uint>(input_ids_len).unwrap();
            std::alloc::dealloc(input_ids_ptr as *mut u8, layout);
            return TrustformersError::OutOfMemory;
        }
        for (i, &mask) in attention_mask.iter().enumerate() {
            *ptr.add(i) = mask as c_uint;
        }
        ptr
    };

    encoding.input_ids = TrustformersTokenIds {
        ids: input_ids_ptr,
        length: input_ids_len,
        capacity: input_ids_len,
    };

    encoding.attention_mask = TrustformersTokenIds {
        ids: attention_mask_ptr,
        length: attention_mask_len,
        capacity: attention_mask_len,
    };

    TrustformersError::Success
}

/// Helper function to create a trained tokenizer from files
fn create_trained_tokenizer(
    _file_paths: &[String],
    _config: &TrustformersTokenizerTrainingConfig,
) -> TrustformersResult<AutoTokenizer> {
    // Note: This is a placeholder implementation.
    // In a real implementation, you would use the tokenizers library's
    // training capabilities to train a custom tokenizer.
    Err(TrustformersError::FeatureNotAvailable)
}

/// Helper function to create a trained tokenizer from texts
fn create_trained_tokenizer_from_texts(
    _texts: &[String],
    _config: &TrustformersTokenizerTrainingConfig,
) -> TrustformersResult<AutoTokenizer> {
    // Note: This is a placeholder implementation.
    // In a real implementation, you would use the tokenizers library's
    // training capabilities to train a custom tokenizer.
    Err(TrustformersError::FeatureNotAvailable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_config_default() {
        let config = TrustformersTokenizerConfig::default();
        assert_eq!(config.max_length, 512);
        assert_eq!(config.padding, 1);
        assert_eq!(config.truncation, 1);
        assert_eq!(config.return_attention_mask, 1);
    }

    #[test]
    fn test_token_ids_default() {
        let token_ids = TrustformersTokenIds::default();
        assert!(token_ids.ids.is_null());
        assert_eq!(token_ids.length, 0);
        assert_eq!(token_ids.capacity, 0);
    }

    #[test]
    fn test_encoding_default() {
        let encoding = TrustformersEncoding::default();
        assert!(encoding.input_ids.ids.is_null());
        assert!(encoding.attention_mask.ids.is_null());
        assert!(encoding.token_type_ids.ids.is_null());
        assert!(encoding.special_tokens_mask.ids.is_null());
    }
}
