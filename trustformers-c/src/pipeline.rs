//! Pipeline C API for TrustformeRS

use anyhow::anyhow;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_float, c_int};
use std::path::PathBuf;
use std::ptr;

use crate::error::*;
use crate::{
    c_str_to_string, core_result_to_error, core_tensor_result_to_error, result_to_error,
    string_to_c_str, trustformers_result_to_error, RESOURCE_REGISTRY,
};

use trustformers::pipeline::{
    enhanced_pipeline, onnx_text_classification_pipeline, onnx_text_generation_pipeline, Backend,
    Device, ONNXBackendConfig, ONNXPipelineOptions, Pipeline, PipelineOptions,
};
use trustformers::AutoTokenizer;

/// C-compatible pipeline handle
pub type TrustformersPipeline = usize;

/// C-compatible pipeline configuration
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersPipelineConfig {
    /// Pipeline task type (e.g., "text-classification", "text-generation")
    pub task: *const c_char,
    /// Model name or path
    pub model: *const c_char,
    /// Backend type: 0=Native, 1=ONNX
    pub backend_type: c_int,
    /// ONNX model path (if using ONNX backend)
    pub onnx_model_path: *const c_char,
    /// Device type: 0=CPU, 1=CUDA
    pub device_type: c_int,
    /// Device ID for multi-GPU setups
    pub device_id: c_int,
    /// Batch size for inference
    pub batch_size: c_int,
    /// Maximum sequence length
    pub max_length: c_int,
    /// Whether to enable profiling
    pub enable_profiling: c_int,
    /// Number of threads for CPU inference
    pub num_threads: c_int,
}

impl Default for TrustformersPipelineConfig {
    fn default() -> Self {
        Self {
            task: ptr::null(),
            model: ptr::null(),
            backend_type: 0, // Native
            onnx_model_path: ptr::null(),
            device_type: 0, // CPU
            device_id: 0,
            batch_size: 1,
            max_length: 512,
            enable_profiling: 0,
            num_threads: 0, // Auto-detect
        }
    }
}

/// C-compatible inference result
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersInferenceResult {
    /// JSON string containing the inference result
    pub result_json: *mut c_char,
    /// Confidence score (if applicable)
    pub confidence: c_float,
    /// Inference time in milliseconds
    pub inference_time_ms: c_double,
    /// Memory used during inference (bytes)
    pub memory_used_bytes: u64,
}

impl Default for TrustformersInferenceResult {
    fn default() -> Self {
        Self {
            result_json: ptr::null_mut(),
            confidence: 0.0,
            inference_time_ms: 0.0,
            memory_used_bytes: 0,
        }
    }
}

/// C-compatible batch inference result
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersBatchResult {
    /// Array of result JSON strings
    pub results: *mut *mut c_char,
    /// Number of results
    pub num_results: usize,
    /// Array of confidence scores
    pub confidences: *mut c_float,
    /// Total batch inference time in milliseconds
    pub total_time_ms: c_double,
    /// Average time per item in milliseconds
    pub avg_time_per_item_ms: c_double,
}

/// C-compatible streaming configuration
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersStreamingConfig {
    /// Maximum tokens to generate
    pub max_new_tokens: c_int,
    /// Temperature for sampling
    pub temperature: c_float,
    /// Top-p for nucleus sampling
    pub top_p: c_float,
    /// Top-k for top-k sampling
    pub top_k: c_int,
    /// Repetition penalty
    pub repetition_penalty: c_float,
    /// Whether to skip special tokens in output
    pub skip_special_tokens: c_int,
    /// Stream buffer size
    pub buffer_size: c_int,
}

/// C-compatible conversation turn
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersConversationTurn {
    /// Speaker: 0=User, 1=Assistant, 2=System
    pub speaker: c_int,
    /// Message content
    pub message: *const c_char,
    /// Timestamp (Unix timestamp)
    pub timestamp: u64,
    /// Turn metadata (JSON string)
    pub metadata: *const c_char,
}

/// C-compatible conversation history
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersConversationHistory {
    /// Array of conversation turns
    pub turns: *mut TrustformersConversationTurn,
    /// Number of turns
    pub num_turns: usize,
    /// Conversation ID
    pub conversation_id: *mut c_char,
    /// Maximum context length
    pub max_context_length: c_int,
}

/// C-compatible multimodal input
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersMultimodalInput {
    /// Text input
    pub text: *const c_char,
    /// Image paths (array of strings)
    pub image_paths: *const *const c_char,
    /// Number of images
    pub num_images: usize,
    /// Audio path
    pub audio_path: *const c_char,
    /// Video path
    pub video_path: *const c_char,
    /// Additional metadata (JSON string)
    pub metadata: *const c_char,
}

/// C-compatible streaming callback function type
/// Parameters: token (UTF-8 string), is_final (0/1), user_data
pub type TrustformersStreamingCallback = extern "C" fn(
    token: *const c_char,
    is_final: c_int,
    user_data: *mut std::os::raw::c_void,
) -> c_int;

impl Default for TrustformersBatchResult {
    fn default() -> Self {
        Self {
            results: ptr::null_mut(),
            num_results: 0,
            confidences: ptr::null_mut(),
            total_time_ms: 0.0,
            avg_time_per_item_ms: 0.0,
        }
    }
}

impl Default for TrustformersStreamingConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 0.8,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
            skip_special_tokens: 1,
            buffer_size: 256,
        }
    }
}

impl Default for TrustformersConversationTurn {
    fn default() -> Self {
        Self {
            speaker: 0,
            message: ptr::null(),
            timestamp: 0,
            metadata: ptr::null(),
        }
    }
}

impl Default for TrustformersConversationHistory {
    fn default() -> Self {
        Self {
            turns: ptr::null_mut(),
            num_turns: 0,
            conversation_id: ptr::null_mut(),
            max_context_length: 2048,
        }
    }
}

impl Default for TrustformersMultimodalInput {
    fn default() -> Self {
        Self {
            text: ptr::null(),
            image_paths: ptr::null(),
            num_images: 0,
            audio_path: ptr::null(),
            video_path: ptr::null(),
            metadata: ptr::null(),
        }
    }
}

/// Create a new pipeline with the given configuration
#[no_mangle]
pub extern "C" fn trustformers_pipeline_create(
    config: *const TrustformersPipelineConfig,
    pipeline_handle: *mut TrustformersPipeline,
) -> TrustformersError {
    if config.is_null() || pipeline_handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let config = unsafe { &*config };

    let task = match c_str_to_string(config.task) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let model_name = if !config.model.is_null() {
        Some(c_str_to_string(config.model).unwrap_or_default())
    } else {
        None
    };

    // Create device configuration
    let device = match config.device_type {
        0 => Some(Device::Cpu),
        1 => {
            // CUDA support - use CUDA device if available
            if is_cuda_available() {
                // Use specified CUDA device ID or default to 0
                let cuda_id = if config.device_id >= 0 { config.device_id as usize } else { 0 };

                // Check if the requested CUDA device is available
                if cuda_id < get_cuda_device_count() {
                    eprintln!("Using CUDA device: {}", cuda_id);
                    Some(Device::Gpu(cuda_id))
                } else {
                    eprintln!("CUDA device {} not available, falling back to CPU", cuda_id);
                    Some(Device::Cpu)
                }
            } else {
                eprintln!("CUDA not available, using CPU");
                Some(Device::Cpu)
            }
        },
        2 => {
            // Auto device selection - choose best available device
            select_best_device(config.device_id as usize)
        },
        _ => Some(Device::Cpu), // Default to CPU for unknown device types
    };

    // Create backend configuration
    let backend = match config.backend_type {
        0 => Some(Backend::Native),
        1 => {
            if config.onnx_model_path.is_null() {
                return TrustformersError::InvalidParameter;
            }
            let onnx_path = match c_str_to_string(config.onnx_model_path) {
                Ok(s) => PathBuf::from(s),
                Err(_) => return TrustformersError::InvalidParameter,
            };
            Some(Backend::ONNX {
                model_path: onnx_path,
            })
        },
        _ => None,
    };

    // Create pipeline options
    let options = PipelineOptions {
        backend,
        device,
        batch_size: if config.batch_size > 0 { Some(config.batch_size as usize) } else { None },
        max_length: if config.max_length > 0 { Some(config.max_length as usize) } else { None },
        num_threads: if config.num_threads > 0 { Some(config.num_threads as usize) } else { None },
        ..Default::default()
    };

    // Create the pipeline
    let pipeline_result = enhanced_pipeline(&task, model_name.as_deref(), Some(options));
    let (error, pipeline_opt) = trustformers_result_to_error(pipeline_result);

    if error != TrustformersError::Success {
        return error;
    }

    let pipeline = pipeline_opt.unwrap();

    // Register pipeline and return handle (pipeline is already Box<dyn Pipeline>)
    let mut registry = RESOURCE_REGISTRY.write();
    let handle = registry.register_pipeline(pipeline);

    unsafe {
        *pipeline_handle = handle;
    }

    TrustformersError::Success
}

/// Create an ONNX text classification pipeline
#[no_mangle]
pub extern "C" fn trustformers_onnx_text_classification_pipeline_create(
    model_path: *const c_char,
    tokenizer_name: *const c_char,
    pipeline_handle: *mut TrustformersPipeline,
) -> TrustformersError {
    if model_path.is_null() || tokenizer_name.is_null() || pipeline_handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let model_path_str = c_try!(c_str_to_string(model_path));
    let tokenizer_name_str = c_try!(c_str_to_string(tokenizer_name));

    let model_path_buf = PathBuf::from(model_path_str);

    // Load tokenizer
    let tokenizer_result = AutoTokenizer::from_pretrained(&tokenizer_name_str);
    let (error, tokenizer_opt) = core_tensor_result_to_error(tokenizer_result);

    if error != TrustformersError::Success {
        return error;
    }

    let tokenizer = tokenizer_opt.unwrap();

    // Create ONNX config
    let config = ONNXBackendConfig::cpu_optimized(model_path_buf.clone());

    // Create pipeline
    let pipeline_result =
        onnx_text_classification_pipeline(&model_path_buf, tokenizer, Some(config));
    let (error, pipeline_opt) = core_tensor_result_to_error(pipeline_result);

    if error != TrustformersError::Success {
        return error;
    }

    let pipeline = pipeline_opt.unwrap();

    // Register pipeline and return handle
    let mut registry = RESOURCE_REGISTRY.write();
    let handle = registry.register_pipeline(Box::new(pipeline));

    unsafe {
        *pipeline_handle = handle;
    }

    TrustformersError::Success
}

/// Create an ONNX text generation pipeline
#[no_mangle]
pub extern "C" fn trustformers_onnx_text_generation_pipeline_create(
    model_path: *const c_char,
    tokenizer_name: *const c_char,
    pipeline_handle: *mut TrustformersPipeline,
) -> TrustformersError {
    if model_path.is_null() || tokenizer_name.is_null() || pipeline_handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let model_path_str = c_try!(c_str_to_string(model_path));
    let tokenizer_name_str = c_try!(c_str_to_string(tokenizer_name));

    let model_path_buf = PathBuf::from(model_path_str);

    // Load tokenizer
    let tokenizer_result = AutoTokenizer::from_pretrained(&tokenizer_name_str);
    let (error, tokenizer_opt) = core_tensor_result_to_error(tokenizer_result);

    if error != TrustformersError::Success {
        return error;
    }

    let tokenizer = tokenizer_opt.unwrap();

    // Create ONNX config
    let config = ONNXBackendConfig::cpu_optimized(model_path_buf.clone());

    // Create pipeline
    let pipeline_result = onnx_text_generation_pipeline(&model_path_buf, tokenizer, Some(config));
    let (error, pipeline_opt) = core_tensor_result_to_error(pipeline_result);

    if error != TrustformersError::Success {
        return error;
    }

    let pipeline = pipeline_opt.unwrap();

    // Register pipeline and return handle
    let mut registry = RESOURCE_REGISTRY.write();
    let handle = registry.register_pipeline(Box::new(pipeline));

    unsafe {
        *pipeline_handle = handle;
    }

    TrustformersError::Success
}

/// Perform inference on a single input
#[no_mangle]
pub extern "C" fn trustformers_pipeline_infer(
    pipeline_handle: TrustformersPipeline,
    input: *const c_char,
    result: *mut TrustformersInferenceResult,
) -> TrustformersError {
    if input.is_null() || result.is_null() {
        return TrustformersError::NullPointer;
    }

    let input_str = c_try!(c_str_to_string(input));

    // Get pipeline from registry
    let registry = RESOURCE_REGISTRY.read();
    let pipeline_arc = match registry.get_pipeline(pipeline_handle) {
        Some(p) => p,
        None => return TrustformersError::InvalidParameter,
    };

    // pipeline_arc is &Arc<dyn Pipeline<Input = String, Output = PipelineOutput>>
    let pipeline = pipeline_arc.as_ref();

    let start_time = std::time::Instant::now();

    // Perform inference
    let inference_result = pipeline.__call__(input_str);
    let (error, output_opt) = trustformers_result_to_error(inference_result);

    if error != TrustformersError::Success {
        return error;
    }

    let output = output_opt.unwrap();
    let inference_time = start_time.elapsed().as_secs_f64() * 1000.0;

    // Convert output to JSON
    let result_json = match serde_json::to_string(&output) {
        Ok(json) => string_to_c_str(json),
        Err(_) => return TrustformersError::SerializationError,
    };

    unsafe {
        (*result).result_json = result_json;
        (*result).confidence = 0.0; // Would extract from output if available
        (*result).inference_time_ms = inference_time;
        (*result).memory_used_bytes = 0; // Would calculate actual memory usage
    }

    TrustformersError::Success
}

/// Perform batch inference on multiple inputs
#[no_mangle]
pub extern "C" fn trustformers_pipeline_batch_infer(
    pipeline_handle: TrustformersPipeline,
    inputs: *const *const c_char,
    num_inputs: usize,
    result: *mut TrustformersBatchResult,
) -> TrustformersError {
    if inputs.is_null() || result.is_null() || num_inputs == 0 {
        return TrustformersError::InvalidParameter;
    }

    // Convert C string array to Rust Vec
    let mut input_strings = Vec::with_capacity(num_inputs);
    for i in 0..num_inputs {
        unsafe {
            let input_ptr = *inputs.add(i);
            if input_ptr.is_null() {
                return TrustformersError::NullPointer;
            }
            let input_str = c_try!(c_str_to_string(input_ptr));
            input_strings.push(input_str);
        }
    }

    // Get pipeline from registry
    let registry = RESOURCE_REGISTRY.read();
    let pipeline_arc = match registry.get_pipeline(pipeline_handle) {
        Some(p) => p,
        None => return TrustformersError::InvalidParameter,
    };

    // pipeline_arc is &Arc<dyn Pipeline<Input = String, Output = PipelineOutput>>
    let pipeline = pipeline_arc.as_ref();

    let start_time = std::time::Instant::now();

    // Perform batch inference
    let mut results: Vec<*mut c_char> = Vec::with_capacity(num_inputs);
    let mut confidences = Vec::with_capacity(num_inputs);

    for input in input_strings {
        let inference_result = pipeline.__call__(input);
        let (error, output_opt) = trustformers_result_to_error(inference_result);

        if error != TrustformersError::Success {
            // Clean up any allocated results before returning error
            for result_ptr in &results {
                unsafe {
                    if !result_ptr.is_null() {
                        let _ = CString::from_raw(*result_ptr);
                    }
                }
            }
            return error;
        }

        let output = output_opt.unwrap();
        let result_json = match serde_json::to_string(&output) {
            Ok(json) => string_to_c_str(json),
            Err(_) => return TrustformersError::SerializationError,
        };

        results.push(result_json);
        confidences.push(0.0); // Would extract from output if available
    }

    let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
    let avg_time = total_time / num_inputs as f64;

    // Allocate C arrays
    let results_array = results.into_boxed_slice();
    let confidences_array = confidences.into_boxed_slice();

    unsafe {
        (*result).results = Box::into_raw(results_array) as *mut *mut c_char;
        (*result).num_results = num_inputs;
        (*result).confidences = Box::into_raw(confidences_array) as *mut c_float;
        (*result).total_time_ms = total_time;
        (*result).avg_time_per_item_ms = avg_time;
    }

    TrustformersError::Success
}

/// Perform streaming text generation with callback
#[no_mangle]
pub extern "C" fn trustformers_pipeline_stream_generate(
    pipeline_handle: TrustformersPipeline,
    input: *const c_char,
    config: *const TrustformersStreamingConfig,
    callback: TrustformersStreamingCallback,
    user_data: *mut std::os::raw::c_void,
) -> TrustformersError {
    if input.is_null() {
        return TrustformersError::NullPointer;
    }

    let input_str = c_try!(c_str_to_string(input));

    let config = if config.is_null() {
        TrustformersStreamingConfig::default()
    } else {
        unsafe { std::ptr::read(config) }
    };

    // Get pipeline from registry
    let registry = RESOURCE_REGISTRY.read();
    let pipeline_arc = match registry.get_pipeline(pipeline_handle) {
        Some(p) => p,
        None => return TrustformersError::InvalidParameter,
    };

    // pipeline_arc is &Arc<dyn Pipeline<Input = String, Output = PipelineOutput>>
    let _pipeline = pipeline_arc.as_ref();

    // Simulate streaming by generating tokens one by one
    // In a real implementation, this would integrate with the model's generation loop
    let simulated_tokens = simulate_streaming_generation(&input_str, &config);

    for (i, token) in simulated_tokens.iter().enumerate() {
        let is_final = if i == simulated_tokens.len() - 1 { 1 } else { 0 };
        let c_token = CString::new(token.as_str())
            .expect("token text should not contain null bytes");

        let should_continue = callback(c_token.as_ptr(), is_final, user_data);
        if should_continue == 0 {
            break; // User requested to stop streaming
        }
    }

    TrustformersError::Success
}

/// Create a conversation pipeline
#[no_mangle]
pub extern "C" fn trustformers_conversation_pipeline_create(
    model_name: *const c_char,
    system_prompt: *const c_char,
    pipeline_handle: *mut TrustformersPipeline,
) -> TrustformersError {
    if model_name.is_null() || pipeline_handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let model_name_str = c_try!(c_str_to_string(model_name));
    let system_prompt_str = if system_prompt.is_null() {
        "You are a helpful assistant.".to_string()
    } else {
        c_try!(c_str_to_string(system_prompt))
    };

    // Create conversation-specific pipeline configuration
    let options = PipelineOptions {
        max_length: Some(512),
        batch_size: Some(1),
        ..Default::default()
    };

    // Create the pipeline
    let pipeline_result = enhanced_pipeline("conversational", Some(&model_name_str), Some(options));
    let (error, pipeline_opt) = trustformers_result_to_error(pipeline_result);

    if error != TrustformersError::Success {
        return error;
    }

    let pipeline = pipeline_opt.unwrap();

    // Register pipeline and return handle (pipeline is already Box<dyn Pipeline>)
    let mut registry = RESOURCE_REGISTRY.write();
    let handle = registry.register_pipeline(pipeline);

    unsafe {
        *pipeline_handle = handle;
    }

    TrustformersError::Success
}

/// Continue a conversation with history
#[no_mangle]
pub extern "C" fn trustformers_conversation_continue(
    pipeline_handle: TrustformersPipeline,
    history: *const TrustformersConversationHistory,
    new_message: *const c_char,
    result: *mut TrustformersInferenceResult,
) -> TrustformersError {
    if history.is_null() || new_message.is_null() || result.is_null() {
        return TrustformersError::NullPointer;
    }

    let new_message_str = c_try!(c_str_to_string(new_message));
    let history_ref = unsafe { &*history };

    // Build conversation context
    let context = match build_conversation_context(history_ref, &new_message_str) {
        Ok(ctx) => ctx,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    // Get pipeline from registry
    let registry = RESOURCE_REGISTRY.read();
    let pipeline_arc = match registry.get_pipeline(pipeline_handle) {
        Some(p) => p,
        None => return TrustformersError::InvalidParameter,
    };

    // pipeline_arc is &Arc<dyn Pipeline<Input = String, Output = PipelineOutput>>
    let pipeline = pipeline_arc.as_ref();

    let start_time = std::time::Instant::now();

    // Perform inference with conversation context
    let inference_result = pipeline.__call__(context);
    let (error, output_opt) = trustformers_result_to_error(inference_result);

    if error != TrustformersError::Success {
        return error;
    }

    let output = output_opt.unwrap();
    let inference_time = start_time.elapsed().as_secs_f64() * 1000.0;

    // Convert output to JSON
    let result_json = match serde_json::to_string(&output) {
        Ok(json) => string_to_c_str(json),
        Err(_) => return TrustformersError::SerializationError,
    };

    unsafe {
        (*result).result_json = result_json;
        (*result).confidence = 0.0;
        (*result).inference_time_ms = inference_time;
        (*result).memory_used_bytes = 0;
    }

    TrustformersError::Success
}

/// Create a multimodal pipeline
#[no_mangle]
pub extern "C" fn trustformers_multimodal_pipeline_create(
    model_name: *const c_char,
    modalities: *const c_char, // JSON string: ["text", "image", "audio"]
    pipeline_handle: *mut TrustformersPipeline,
) -> TrustformersError {
    if model_name.is_null() || modalities.is_null() || pipeline_handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let model_name_str = c_try!(c_str_to_string(model_name));
    let modalities_str = c_try!(c_str_to_string(modalities));

    // Parse modalities JSON
    let _modalities_list: Vec<String> = match serde_json::from_str(&modalities_str) {
        Ok(list) => list,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    // Parse modality configuration
    let modality_list: Vec<String> = match serde_json::from_str(&modalities_str) {
        Ok(list) => list,
        Err(_) => {
            // Fallback to default multimodal setup
            vec!["text".to_string(), "image".to_string()]
        },
    };

    // Create multimodal-specific pipeline configuration
    let options = PipelineOptions {
        max_length: Some(256),
        batch_size: Some(1),
        ..Default::default()
    };

    // Create the pipeline (placeholder - would need actual multimodal implementation)
    let pipeline_result = enhanced_pipeline("multimodal", Some(&model_name_str), Some(options));
    let (error, pipeline_opt) = trustformers_result_to_error(pipeline_result);

    if error != TrustformersError::Success {
        // For now, return feature not available since multimodal is not fully implemented
        return TrustformersError::FeatureNotAvailable;
    }

    let pipeline = pipeline_opt.unwrap();

    let mut registry = RESOURCE_REGISTRY.write();
    let handle = registry.register_pipeline(pipeline);

    unsafe {
        *pipeline_handle = handle;
    }

    TrustformersError::Success
}

/// Perform multimodal inference
#[no_mangle]
pub extern "C" fn trustformers_multimodal_infer(
    pipeline_handle: TrustformersPipeline,
    input: *const TrustformersMultimodalInput,
    result: *mut TrustformersInferenceResult,
) -> TrustformersError {
    if input.is_null() || result.is_null() {
        return TrustformersError::NullPointer;
    }

    let input_ref = unsafe { &*input };

    // Extract multimodal inputs
    let text_input = if input_ref.text.is_null() {
        String::new()
    } else {
        c_try!(c_str_to_string(input_ref.text))
    };

    let mut image_paths = Vec::new();
    if !input_ref.image_paths.is_null() && input_ref.num_images > 0 {
        for i in 0..input_ref.num_images {
            unsafe {
                let path_ptr = *input_ref.image_paths.add(i);
                if !path_ptr.is_null() {
                    let path = c_try!(c_str_to_string(path_ptr));
                    image_paths.push(path);
                }
            }
        }
    }

    // Get pipeline from registry
    let registry = RESOURCE_REGISTRY.read();
    let pipeline_arc = match registry.get_pipeline(pipeline_handle) {
        Some(p) => p,
        None => return TrustformersError::InvalidParameter,
    };

    // pipeline_arc is &Arc<dyn Pipeline<Input = String, Output = PipelineOutput>>
    let _pipeline = pipeline_arc.as_ref();

    // For now, return feature not available since multimodal is not fully implemented
    // In a real implementation, this would process the multimodal inputs
    let _multimodal_context = format!("Text: {} Images: {:?}", text_input, image_paths);

    TrustformersError::FeatureNotAvailable
}

/// Free conversation history memory
#[no_mangle]
pub extern "C" fn trustformers_conversation_history_free(
    history: *mut TrustformersConversationHistory,
) {
    if history.is_null() {
        return;
    }

    unsafe {
        let hist = &mut *history;

        if !hist.turns.is_null() && hist.num_turns > 0 {
            for i in 0..hist.num_turns {
                let turn = &mut *hist.turns.add(i);
                if !turn.message.is_null() {
                    // Note: We don't free these since they're owned by the caller
                    // In a real implementation, you'd track ownership properly
                }
                if !turn.metadata.is_null() {
                    // Same note as above
                }
            }

            let layout =
                std::alloc::Layout::array::<TrustformersConversationTurn>(hist.num_turns).unwrap();
            std::alloc::dealloc(hist.turns as *mut u8, layout);
            hist.turns = ptr::null_mut();
        }

        if !hist.conversation_id.is_null() {
            let _ = CString::from_raw(hist.conversation_id);
            hist.conversation_id = ptr::null_mut();
        }

        hist.num_turns = 0;
    }
}

/// Free inference result memory
#[no_mangle]
pub extern "C" fn trustformers_inference_result_free(result: *mut TrustformersInferenceResult) {
    if result.is_null() {
        return;
    }

    unsafe {
        let result_ref = &mut *result;
        if !result_ref.result_json.is_null() {
            let _ = CString::from_raw(result_ref.result_json);
            result_ref.result_json = ptr::null_mut();
        }
    }
}

/// Free batch inference result memory
#[no_mangle]
pub extern "C" fn trustformers_batch_result_free(result: *mut TrustformersBatchResult) {
    if result.is_null() {
        return;
    }

    unsafe {
        let result_ref = &mut *result;

        if !result_ref.results.is_null() && result_ref.num_results > 0 {
            // Free individual result strings
            for i in 0..result_ref.num_results {
                let result_ptr = *result_ref.results.add(i);
                if !result_ptr.is_null() {
                    let _ = CString::from_raw(result_ptr);
                }
            }

            // Free the results array
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(
                result_ref.results,
                result_ref.num_results,
            ));
            result_ref.results = ptr::null_mut();
        }

        if !result_ref.confidences.is_null() && result_ref.num_results > 0 {
            // Free confidences array
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(
                result_ref.confidences,
                result_ref.num_results,
            ));
            result_ref.confidences = ptr::null_mut();
        }

        result_ref.num_results = 0;
    }
}

/// Destroy a pipeline and free its resources
#[no_mangle]
pub extern "C" fn trustformers_pipeline_destroy(
    pipeline_handle: TrustformersPipeline,
) -> TrustformersError {
    let mut registry = RESOURCE_REGISTRY.write();
    if registry.remove_pipeline(pipeline_handle) {
        TrustformersError::Success
    } else {
        TrustformersError::InvalidParameter
    }
}

/// Get pipeline information
#[no_mangle]
pub extern "C" fn trustformers_pipeline_info(
    pipeline_handle: TrustformersPipeline,
    info_json: *mut *mut c_char,
) -> TrustformersError {
    if info_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = RESOURCE_REGISTRY.read();
    let _pipeline_arc = match registry.get_pipeline(pipeline_handle) {
        Some(p) => p,
        None => return TrustformersError::InvalidParameter,
    };

    // Create pipeline info JSON
    let mut info = Map::new();
    info.insert("type".to_string(), Value::String("pipeline".to_string()));
    info.insert("handle".to_string(), Value::Number(pipeline_handle.into()));
    // Would add more specific info based on pipeline type

    let info_str = match serde_json::to_string(&info) {
        Ok(json) => string_to_c_str(json),
        Err(_) => return TrustformersError::SerializationError,
    };

    unsafe {
        *info_json = info_str;
    }

    TrustformersError::Success
}

/// Helper function to simulate streaming generation (placeholder implementation)
fn simulate_streaming_generation(input: &str, config: &TrustformersStreamingConfig) -> Vec<String> {
    // In a real implementation, this would integrate with the model's generation loop
    // For now, simulate by splitting a response into tokens
    let simulated_response = format!(
        "This is a simulated response to: '{}'. Temperature: {}, Max tokens: {}",
        input, config.temperature, config.max_new_tokens
    );

    // Split into word tokens for simulation
    let words: Vec<String> = simulated_response
        .split_whitespace()
        .take(config.max_new_tokens as usize)
        .map(|s| format!("{} ", s))
        .collect();

    words
}

/// Helper function to build conversation context from history
fn build_conversation_context(
    history: &TrustformersConversationHistory,
    new_message: &str,
) -> TrustformersResult<String> {
    let mut context = String::new();

    // Add conversation history
    if !history.turns.is_null() && history.num_turns > 0 {
        for i in 0..history.num_turns {
            unsafe {
                let turn = &*history.turns.add(i);
                if !turn.message.is_null() {
                    let message = c_str_to_string(turn.message)?;
                    let speaker = match turn.speaker {
                        0 => "User",
                        1 => "Assistant",
                        2 => "System",
                        _ => "Unknown",
                    };
                    context.push_str(&format!("{}: {}\n", speaker, message));
                }
            }
        }
    }

    // Add new user message
    context.push_str(&format!("User: {}\nAssistant: ", new_message));

    // Truncate to max context length if needed
    if context.len() > history.max_context_length as usize {
        let start = context.len() - history.max_context_length as usize;
        context = context[start..].to_string();
    }

    Ok(context)
}

/// Check if CUDA is available on the system
fn is_cuda_available() -> bool {
    // Check for CUDA availability through various means
    // 1. Try to initialize CUDA context
    // 2. Check for CUDA drivers
    // 3. Check for CUDA-capable devices

    // For now, we'll use a simple heuristic approach
    // In a real implementation, this would use proper CUDA APIs

    #[cfg(feature = "cuda")]
    {
        // Check if CUDA runtime is available
        // This would typically use cudaGetDeviceCount() or similar
        match std::env::var("CUDA_VISIBLE_DEVICES") {
            Ok(devices) => !devices.trim().is_empty() && devices != "-1",
            Err(_) => {
                // Check if nvidia-smi is available as a fallback
                std::process::Command::new("nvidia-smi")
                    .output()
                    .map(|output| output.status.success())
                    .unwrap_or(false)
            },
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Select the best available device automatically
fn select_best_device(preferred_id: usize) -> Option<Device> {
    // Auto device selection priority:
    // 1. CUDA (if available and requested ID is valid)
    // 2. CPU (fallback)

    if is_cuda_available() {
        // Query available CUDA devices and select the best one
        let cuda_device_count = get_cuda_device_count();

        if cuda_device_count > 0 {
            // Use the preferred device ID if it's valid, otherwise use device 0
            let device_id = if preferred_id < cuda_device_count {
                preferred_id
            } else {
                // If preferred ID is out of range, use device 0
                // In production, you might want to log this fallback
                0
            };

            Some(Device::Gpu(device_id))
        } else {
            // CUDA is available but no devices found, fallback to CPU
            Some(Device::Cpu)
        }
    } else {
        Some(Device::Cpu)
    }
}

// Duplicate is_cuda_available function removed

/// Get the number of available CUDA devices
fn get_cuda_device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        // In a real implementation, this would query CUDA device count
        // For now, simulate based on environment or common configurations
        if let Ok(devices) = std::env::var("CUDA_VISIBLE_DEVICES") {
            if devices == "-1" || devices.is_empty() {
                return 0;
            }
            // Count comma-separated device IDs
            devices.split(',').count()
        } else if is_cuda_available() {
            // Default assumption: single GPU if CUDA is available
            1
        } else {
            0
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_pipeline_config_default() {
        let config = TrustformersPipelineConfig::default();
        assert_eq!(config.batch_size, 1);
        assert_eq!(config.max_length, 512);
        assert_eq!(config.backend_type, 0);
        assert_eq!(config.device_type, 0);
    }

    #[test]
    fn test_inference_result_default() {
        let result = TrustformersInferenceResult::default();
        assert!(result.result_json.is_null());
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.inference_time_ms, 0.0);
    }
}
