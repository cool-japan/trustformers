//! Common type mapping utilities for code generators

use crate::codegen::ast::FfiType;
use crate::codegen::{CodeGenConfig, TargetLanguage};
use std::collections::HashMap;

/// Base type mapper trait for language-specific type mappings
pub trait TypeMapper {
    /// Map an FFI type to the target language type
    fn map_type(&self, ffi_type: &FfiType) -> String;

    /// Map a base type name to the target language
    fn map_base_type(&self, type_name: &str) -> String;

    /// Get the target language
    fn target_language(&self) -> TargetLanguage;
}

/// Create default type mappings for C types
pub fn create_c_type_mappings() -> HashMap<String, String> {
    let mut mappings = HashMap::new();

    // Basic integer types
    mappings.insert("c_int".to_string(), "c_int".to_string());
    mappings.insert("c_uint".to_string(), "c_uint".to_string());
    mappings.insert("c_short".to_string(), "c_short".to_string());
    mappings.insert("c_ushort".to_string(), "c_ushort".to_string());
    mappings.insert("c_long".to_string(), "c_long".to_string());
    mappings.insert("c_ulong".to_string(), "c_ulong".to_string());
    mappings.insert("c_longlong".to_string(), "c_longlong".to_string());
    mappings.insert("c_ulonglong".to_string(), "c_ulonglong".to_string());

    // Floating point types
    mappings.insert("c_float".to_string(), "c_float".to_string());
    mappings.insert("c_double".to_string(), "c_double".to_string());

    // Character and boolean types
    mappings.insert("c_char".to_string(), "c_char".to_string());
    mappings.insert("c_uchar".to_string(), "c_uchar".to_string());
    mappings.insert("c_bool".to_string(), "c_bool".to_string());

    // Void and size types
    mappings.insert("c_void".to_string(), "c_void".to_string());
    mappings.insert("isize".to_string(), "isize".to_string());
    mappings.insert("usize".to_string(), "usize".to_string());

    // Pointer types
    mappings.insert("*const c_char".to_string(), "*const c_char".to_string());
    mappings.insert("*mut c_char".to_string(), "*mut c_char".to_string());
    mappings.insert("*const c_void".to_string(), "*const c_void".to_string());
    mappings.insert("*mut c_void".to_string(), "*mut c_void".to_string());

    mappings
}

/// Common helper function to check if a type is a pointer type
pub fn is_pointer_type(type_name: &str) -> bool {
    type_name.starts_with('*') || type_name.contains("*const") || type_name.contains("*mut")
}

/// Common helper function to check if a type is a string type
pub fn is_string_type(ffi_type: &FfiType) -> bool {
    ffi_type.name == "*const c_char" || ffi_type.name == "*mut c_char" || ffi_type.name == "String"
}

/// Common helper function to extract base type from pointer type
pub fn extract_base_type(type_name: &str) -> String {
    if type_name.starts_with("*const ") {
        type_name[7..].to_string()
    } else if type_name.starts_with("*mut ") {
        type_name[5..].to_string()
    } else if type_name.starts_with('*') {
        type_name[1..].to_string()
    } else {
        type_name.to_string()
    }
}

/// Common helper function to check if a function can fail (returns Result)
pub fn function_can_fail(return_type: &FfiType) -> bool {
    return_type.name.starts_with("Result<") || return_type.name.contains("Error")
}

/// Common helper to generate documentation comment prefix for different languages
pub fn get_doc_comment_prefix(language: TargetLanguage) -> &'static str {
    match language {
        TargetLanguage::Python => "#",
        TargetLanguage::Java | TargetLanguage::JavaScript | TargetLanguage::TypeScript => "//",
        TargetLanguage::Go => "//",
        TargetLanguage::CSharp => "//",
        TargetLanguage::Kotlin => "//",
        TargetLanguage::Swift => "//",
        TargetLanguage::Ruby => "#",
        TargetLanguage::Php => "//",
        _ => "//",
    }
}
