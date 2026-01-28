//! Plugin system for TrustformeRS C API
//!
//! This module provides a plugin system for dynamically loading and managing
//! custom backends, operations, and extensions at runtime.

use crate::error::TrustformersError;
use crate::utils::{c_str_to_string, string_to_c_str};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;

/// Plugin handle type
pub type TrustformersPluginHandle = usize;

/// Plugin type enumeration
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PluginType {
    /// Backend plugin (custom hardware acceleration)
    Backend = 0,
    /// Operation plugin (custom tensor operations)
    Operation = 1,
    /// Optimizer plugin (custom optimization strategies)
    Optimizer = 2,
    /// Tokenizer plugin (custom tokenization algorithms)
    Tokenizer = 3,
    /// Model plugin (custom model architectures)
    Model = 4,
    /// Custom/unknown plugin type
    Custom = 999,
}

/// Plugin capability flags
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PluginCapabilities {
    /// Supports GPU acceleration
    pub supports_gpu: bool,
    /// Supports multi-threading
    pub supports_multithreading: bool,
    /// Supports quantization
    pub supports_quantization: bool,
    /// Supports dynamic batching
    pub supports_dynamic_batching: bool,
    /// Supports mixed precision
    pub supports_mixed_precision: bool,
    /// Maximum batch size (0 = unlimited)
    pub max_batch_size: c_int,
}

impl Default for PluginCapabilities {
    fn default() -> Self {
        Self {
            supports_gpu: false,
            supports_multithreading: true,
            supports_quantization: false,
            supports_dynamic_batching: false,
            supports_mixed_precision: false,
            max_batch_size: 0,
        }
    }
}

/// Plugin metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin author
    pub author: String,
    /// Plugin description
    pub description: String,
    /// Plugin type
    pub plugin_type: PluginType,
    /// Required API version
    pub required_api_version: String,
}

/// Plugin initialization function type
///
/// Called when the plugin is loaded. Should return 0 on success, non-zero on error.
pub type PluginInitFn = extern "C" fn(config: *const c_void) -> c_int;

/// Plugin cleanup function type
///
/// Called when the plugin is unloaded.
pub type PluginCleanupFn = extern "C" fn();

/// Plugin operation function type
///
/// Generic function pointer for plugin operations.
pub type PluginOperationFn = extern "C" fn(
    input: *const c_void,
    input_size: usize,
    output: *mut c_void,
    output_size: *mut usize,
) -> c_int;

/// Internal plugin structure
struct Plugin {
    handle: TrustformersPluginHandle,
    metadata: PluginMetadata,
    capabilities: PluginCapabilities,
    library_path: PathBuf,
    init_fn: Option<PluginInitFn>,
    cleanup_fn: Option<PluginCleanupFn>,
    operations: HashMap<String, PluginOperationFn>,
    is_initialized: bool,
}

impl Plugin {
    fn new(
        handle: TrustformersPluginHandle,
        metadata: PluginMetadata,
        capabilities: PluginCapabilities,
        library_path: PathBuf,
    ) -> Self {
        Self {
            handle,
            metadata,
            capabilities,
            library_path,
            init_fn: None,
            cleanup_fn: None,
            operations: HashMap::new(),
            is_initialized: false,
        }
    }

    fn initialize(&mut self, config: *const c_void) -> Result<(), String> {
        if self.is_initialized {
            return Ok(());
        }

        if let Some(init_fn) = self.init_fn {
            let result = init_fn(config);
            if result != 0 {
                return Err(format!("Plugin initialization failed with code {}", result));
            }
        }

        self.is_initialized = true;
        Ok(())
    }

    fn cleanup(&mut self) {
        if !self.is_initialized {
            return;
        }

        if let Some(cleanup_fn) = self.cleanup_fn {
            cleanup_fn();
        }

        self.is_initialized = false;
    }

    fn register_operation(&mut self, name: String, operation: PluginOperationFn) {
        self.operations.insert(name, operation);
    }

    fn get_operation(&self, name: &str) -> Option<PluginOperationFn> {
        self.operations.get(name).copied()
    }
}

/// Plugin registry
struct PluginRegistry {
    plugins: HashMap<TrustformersPluginHandle, Arc<RwLock<Plugin>>>,
    plugin_by_name: HashMap<String, TrustformersPluginHandle>,
    next_handle: TrustformersPluginHandle,
    plugin_search_paths: Vec<PathBuf>,
}

impl PluginRegistry {
    fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            plugin_by_name: HashMap::new(),
            next_handle: 1,
            plugin_search_paths: Vec::new(),
        }
    }

    fn register_plugin(
        &mut self,
        metadata: PluginMetadata,
        capabilities: PluginCapabilities,
        library_path: PathBuf,
    ) -> TrustformersPluginHandle {
        let handle = self.next_handle;
        self.next_handle += 1;

        let plugin = Arc::new(RwLock::new(Plugin::new(
            handle,
            metadata.clone(),
            capabilities,
            library_path,
        )));

        self.plugins.insert(handle, plugin);
        self.plugin_by_name.insert(metadata.name, handle);

        handle
    }

    fn get_plugin(&self, handle: TrustformersPluginHandle) -> Option<Arc<RwLock<Plugin>>> {
        self.plugins.get(&handle).cloned()
    }

    fn get_plugin_by_name(&self, name: &str) -> Option<Arc<RwLock<Plugin>>> {
        self.plugin_by_name
            .get(name)
            .and_then(|handle| self.plugins.get(handle).cloned())
    }

    fn unregister_plugin(&mut self, handle: TrustformersPluginHandle) -> bool {
        if let Some(plugin_arc) = self.plugins.remove(&handle) {
            let mut plugin = plugin_arc.write();
            plugin.cleanup();
            self.plugin_by_name.remove(&plugin.metadata.name);
            true
        } else {
            false
        }
    }

    fn add_search_path(&mut self, path: PathBuf) {
        if !self.plugin_search_paths.contains(&path) {
            self.plugin_search_paths.push(path);
        }
    }

    fn list_plugins(&self) -> Vec<TrustformersPluginHandle> {
        self.plugins.keys().copied().collect()
    }
}

/// Global plugin registry
static PLUGIN_REGISTRY: Lazy<RwLock<PluginRegistry>> =
    Lazy::new(|| RwLock::new(PluginRegistry::new()));

/// Register a new plugin
///
/// # Parameters
/// - `name`: Plugin name
/// - `version`: Plugin version
/// - `plugin_type`: Type of plugin
/// - `library_path`: Path to the plugin library
/// - `capabilities`: Plugin capabilities
/// - `handle`: Output parameter for plugin handle
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_plugin_register(
    name: *const c_char,
    version: *const c_char,
    plugin_type: PluginType,
    library_path: *const c_char,
    capabilities: *const PluginCapabilities,
    handle: *mut TrustformersPluginHandle,
) -> TrustformersError {
    if name.is_null() || version.is_null() || library_path.is_null() || handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let name_str = match c_str_to_string(name) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let version_str = match c_str_to_string(version) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let path_str = match c_str_to_string(library_path) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let caps = if capabilities.is_null() {
        PluginCapabilities::default()
    } else {
        unsafe { *capabilities }
    };

    let metadata = PluginMetadata {
        name: name_str,
        version: version_str,
        author: String::from("Unknown"),
        description: String::from("Plugin"),
        plugin_type,
        required_api_version: env!("CARGO_PKG_VERSION").to_string(),
    };

    let plugin_handle =
        PLUGIN_REGISTRY.write().register_plugin(metadata, caps, PathBuf::from(path_str));

    unsafe {
        *handle = plugin_handle;
    }

    TrustformersError::Success
}

/// Initialize a registered plugin
///
/// # Parameters
/// - `handle`: Plugin handle
/// - `config`: Optional configuration data (can be null)
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_plugin_initialize(
    handle: TrustformersPluginHandle,
    config: *const c_void,
) -> TrustformersError {
    let registry = PLUGIN_REGISTRY.read();
    if let Some(plugin_arc) = registry.get_plugin(handle) {
        let mut plugin = plugin_arc.write();
        match plugin.initialize(config) {
            Ok(_) => TrustformersError::Success,
            Err(_) => TrustformersError::PluginInitError,
        }
    } else {
        TrustformersError::InvalidHandle
    }
}

/// Unregister and cleanup a plugin
///
/// # Parameters
/// - `handle`: Plugin handle to unregister
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_plugin_unregister(
    handle: TrustformersPluginHandle,
) -> TrustformersError {
    let removed = PLUGIN_REGISTRY.write().unregister_plugin(handle);
    if removed {
        TrustformersError::Success
    } else {
        TrustformersError::InvalidHandle
    }
}

/// Get plugin metadata
///
/// # Parameters
/// - `handle`: Plugin handle
/// - `name_out`: Output buffer for plugin name
/// - `name_len`: Size of name buffer
/// - `version_out`: Output buffer for version
/// - `version_len`: Size of version buffer
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_plugin_get_metadata(
    handle: TrustformersPluginHandle,
    name_out: *mut c_char,
    name_len: usize,
    version_out: *mut c_char,
    version_len: usize,
) -> TrustformersError {
    if name_out.is_null() || version_out.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = PLUGIN_REGISTRY.read();
    if let Some(plugin_arc) = registry.get_plugin(handle) {
        let plugin = plugin_arc.read();

        // Copy name
        let name_bytes = plugin.metadata.name.as_bytes();
        let copy_len = std::cmp::min(name_bytes.len(), name_len - 1);
        unsafe {
            ptr::copy_nonoverlapping(name_bytes.as_ptr(), name_out as *mut u8, copy_len);
            *name_out.add(copy_len) = 0;
        }

        // Copy version
        let version_bytes = plugin.metadata.version.as_bytes();
        let copy_len = std::cmp::min(version_bytes.len(), version_len - 1);
        unsafe {
            ptr::copy_nonoverlapping(version_bytes.as_ptr(), version_out as *mut u8, copy_len);
            *version_out.add(copy_len) = 0;
        }

        TrustformersError::Success
    } else {
        TrustformersError::InvalidHandle
    }
}

/// Get plugin capabilities
///
/// # Parameters
/// - `handle`: Plugin handle
/// - `capabilities`: Output parameter for capabilities
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_plugin_get_capabilities(
    handle: TrustformersPluginHandle,
    capabilities: *mut PluginCapabilities,
) -> TrustformersError {
    if capabilities.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = PLUGIN_REGISTRY.read();
    if let Some(plugin_arc) = registry.get_plugin(handle) {
        let plugin = plugin_arc.read();
        unsafe {
            *capabilities = plugin.capabilities;
        }
        TrustformersError::Success
    } else {
        TrustformersError::InvalidHandle
    }
}

/// Register an operation with a plugin
///
/// # Parameters
/// - `handle`: Plugin handle
/// - `operation_name`: Name of the operation
/// - `operation_fn`: Function pointer for the operation
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_plugin_register_operation(
    handle: TrustformersPluginHandle,
    operation_name: *const c_char,
    operation_fn: PluginOperationFn,
) -> TrustformersError {
    if operation_name.is_null() {
        return TrustformersError::NullPointer;
    }

    let op_name = match c_str_to_string(operation_name) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let registry = PLUGIN_REGISTRY.read();
    if let Some(plugin_arc) = registry.get_plugin(handle) {
        let mut plugin = plugin_arc.write();
        plugin.register_operation(op_name, operation_fn);
        TrustformersError::Success
    } else {
        TrustformersError::InvalidHandle
    }
}

/// Execute a plugin operation
///
/// # Parameters
/// - `handle`: Plugin handle
/// - `operation_name`: Name of the operation to execute
/// - `input`: Input data
/// - `input_size`: Size of input data
/// - `output`: Output buffer
/// - `output_size`: Output buffer size (input/output parameter)
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_plugin_execute_operation(
    handle: TrustformersPluginHandle,
    operation_name: *const c_char,
    input: *const c_void,
    input_size: usize,
    output: *mut c_void,
    output_size: *mut usize,
) -> TrustformersError {
    if operation_name.is_null() || input.is_null() || output.is_null() || output_size.is_null() {
        return TrustformersError::NullPointer;
    }

    let op_name = match c_str_to_string(operation_name) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let registry = PLUGIN_REGISTRY.read();
    if let Some(plugin_arc) = registry.get_plugin(handle) {
        let plugin = plugin_arc.read();

        if let Some(operation) = plugin.get_operation(&op_name) {
            let result = operation(input, input_size, output, output_size);
            if result == 0 {
                TrustformersError::Success
            } else {
                TrustformersError::RuntimeError
            }
        } else {
            TrustformersError::OperationNotFound
        }
    } else {
        TrustformersError::InvalidHandle
    }
}

/// Add a plugin search path
///
/// # Parameters
/// - `path`: Directory path to search for plugins
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_plugin_add_search_path(path: *const c_char) -> TrustformersError {
    if path.is_null() {
        return TrustformersError::NullPointer;
    }

    let path_str = match c_str_to_string(path) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    PLUGIN_REGISTRY.write().add_search_path(PathBuf::from(path_str));
    TrustformersError::Success
}

/// List all registered plugins
///
/// # Parameters
/// - `handles`: Output array for plugin handles
/// - `max_count`: Maximum number of handles to return
/// - `actual_count`: Output parameter for actual number of plugins
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_plugin_list(
    handles: *mut TrustformersPluginHandle,
    max_count: usize,
    actual_count: *mut usize,
) -> TrustformersError {
    if handles.is_null() || actual_count.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = PLUGIN_REGISTRY.read();
    let plugin_handles = registry.list_plugins();

    let count = std::cmp::min(plugin_handles.len(), max_count);
    unsafe {
        ptr::copy_nonoverlapping(plugin_handles.as_ptr(), handles, count);
        *actual_count = plugin_handles.len();
    }

    TrustformersError::Success
}

/// Find a plugin by name
///
/// # Parameters
/// - `name`: Plugin name to search for
/// - `handle`: Output parameter for plugin handle
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_plugin_find_by_name(
    name: *const c_char,
    handle: *mut TrustformersPluginHandle,
) -> TrustformersError {
    if name.is_null() || handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let name_str = match c_str_to_string(name) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let registry = PLUGIN_REGISTRY.read();
    if let Some(plugin_arc) = registry.get_plugin_by_name(&name_str) {
        let plugin = plugin_arc.read();
        unsafe {
            *handle = plugin.handle;
        }
        TrustformersError::Success
    } else {
        TrustformersError::PluginNotFound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_registration() {
        let name = CString::new("test-plugin").unwrap();
        let version = CString::new("1.0.0").unwrap();
        let path = CString::new("/path/to/plugin.so").unwrap();
        let mut handle: TrustformersPluginHandle = 0;

        let caps = PluginCapabilities {
            supports_gpu: true,
            supports_multithreading: true,
            ..Default::default()
        };

        let err = trustformers_plugin_register(
            name.as_ptr(),
            version.as_ptr(),
            PluginType::Backend,
            path.as_ptr(),
            &caps,
            &mut handle,
        );

        assert_eq!(err, TrustformersError::Success);
        assert_ne!(handle, 0);

        // Cleanup
        let err = trustformers_plugin_unregister(handle);
        assert_eq!(err, TrustformersError::Success);
    }

    #[test]
    fn test_plugin_find_by_name() {
        let name = CString::new("findme-plugin").unwrap();
        let version = CString::new("1.0.0").unwrap();
        let path = CString::new("/path/to/plugin.so").unwrap();
        let mut handle: TrustformersPluginHandle = 0;

        let caps = PluginCapabilities::default();

        trustformers_plugin_register(
            name.as_ptr(),
            version.as_ptr(),
            PluginType::Backend,
            path.as_ptr(),
            &caps,
            &mut handle,
        );

        // Find by name
        let mut found_handle: TrustformersPluginHandle = 0;
        let err = trustformers_plugin_find_by_name(name.as_ptr(), &mut found_handle);

        assert_eq!(err, TrustformersError::Success);
        assert_eq!(handle, found_handle);

        // Cleanup
        trustformers_plugin_unregister(handle);
    }

    #[test]
    fn test_plugin_metadata() {
        let name = CString::new("metadata-plugin").unwrap();
        let version = CString::new("2.5.0").unwrap();
        let path = CString::new("/path/to/plugin.so").unwrap();
        let mut handle: TrustformersPluginHandle = 0;

        trustformers_plugin_register(
            name.as_ptr(),
            version.as_ptr(),
            PluginType::Model,
            path.as_ptr(),
            ptr::null(),
            &mut handle,
        );

        // Get metadata
        let mut name_buf = vec![0u8; 128];
        let mut version_buf = vec![0u8; 128];

        let err = trustformers_plugin_get_metadata(
            handle,
            name_buf.as_mut_ptr() as *mut c_char,
            name_buf.len(),
            version_buf.as_mut_ptr() as *mut c_char,
            version_buf.len(),
        );

        assert_eq!(err, TrustformersError::Success);

        // Cleanup
        trustformers_plugin_unregister(handle);
    }
}
