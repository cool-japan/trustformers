//! Abstract Syntax Tree for FFI interface representation
//!
//! Defines the data structures used to represent the C FFI interface
//! parsed from Rust source code, which can then be used to generate
//! language bindings.

use super::TargetLanguage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete FFI interface representation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FfiInterface {
    /// All exported functions
    pub functions: Vec<FfiFunction>,
    /// All exported structs/types
    pub structs: Vec<FfiStruct>,
    /// All exported enums
    pub enums: Vec<FfiEnum>,
    /// All exported constants
    pub constants: Vec<FfiConstant>,
    /// All exported type aliases
    pub type_aliases: Vec<FfiTypeAlias>,
    /// Global configuration/metadata
    pub metadata: InterfaceMetadata,
}

/// Metadata about the FFI interface
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InterfaceMetadata {
    /// Library name
    pub library_name: String,
    /// Library version
    pub version: String,
    /// Minimum required library version
    pub min_version: Option<String>,
    /// Feature flags this interface depends on
    pub required_features: Vec<String>,
    /// Optional features
    pub optional_features: Vec<String>,
    /// Platform-specific information
    pub platforms: Vec<Platform>,
}

/// Platform-specific information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Platform {
    /// Platform identifier (e.g., "windows", "linux", "macos")
    pub name: String,
    /// Architecture (e.g., "x86_64", "aarch64")
    pub arch: Option<String>,
    /// Platform-specific functions
    pub functions: Vec<String>,
    /// Platform-specific types
    pub types: Vec<String>,
}

/// Exported FFI function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiFunction {
    /// Function name (without prefix)
    pub name: String,
    /// C symbol name (with prefix)
    pub c_name: String,
    /// Function documentation
    pub documentation: Vec<String>,
    /// Function parameters
    pub parameters: Vec<FfiParameter>,
    /// Return type
    pub return_type: FfiType,
    /// Whether the function is unsafe
    pub is_unsafe: bool,
    /// Whether the function can fail (returns error)
    pub can_fail: bool,
    /// Feature flags required for this function
    pub required_features: Vec<String>,
    /// Platform availability
    pub platforms: Vec<String>,
    /// Deprecation information
    pub deprecation: Option<DeprecationInfo>,
    /// Function attributes
    pub attributes: Vec<FfiAttribute>,
}

/// Function parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub type_info: FfiType,
    /// Parameter documentation
    pub documentation: Vec<String>,
    /// Whether the parameter is optional
    pub is_optional: bool,
    /// Default value (if any)
    pub default_value: Option<String>,
    /// Parameter attributes
    pub attributes: Vec<FfiAttribute>,
}

/// FFI type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiType {
    /// Type name
    pub name: String,
    /// Whether the type is a pointer
    pub is_pointer: bool,
    /// Whether the pointer is const
    pub is_const: bool,
    /// Whether the type is mutable
    pub is_mutable: bool,
    /// Pointer level (0 = not pointer, 1 = *, 2 = **, etc.)
    pub pointer_level: u32,
    /// Array size (if fixed-size array)
    pub array_size: Option<u32>,
    /// Generic type parameters
    pub generic_params: Vec<FfiType>,
    /// Underlying primitive type
    pub primitive_type: Option<PrimitiveType>,
    /// Whether this is a callback/function pointer
    pub is_callback: bool,
    /// Callback signature (if applicable)
    pub callback_signature: Option<Box<FfiFunction>>,
}

/// Primitive C types
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrimitiveType {
    // Integer types
    Int8,
    Int16,
    #[default]
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    IntPtr,
    UIntPtr,

    // Floating point
    Float32,
    Float64,

    // Other types
    Bool,
    Char,
    Void,
    CString,

    // Handle types
    Handle,
    OpaquePointer,
}

/// Exported struct/record
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FfiStruct {
    /// Struct name
    pub name: String,
    /// C struct name
    pub c_name: String,
    /// Struct documentation
    pub documentation: Vec<String>,
    /// Struct fields
    pub fields: Vec<FfiField>,
    /// Whether the struct is opaque (no field access)
    pub is_opaque: bool,
    /// Whether the struct is packed
    pub is_packed: bool,
    /// Struct alignment
    pub alignment: Option<u32>,
    /// Feature flags required for this struct
    pub required_features: Vec<String>,
    /// Platform availability
    pub platforms: Vec<String>,
    /// Deprecation information
    pub deprecation: Option<DeprecationInfo>,
}

/// Struct field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiField {
    /// Field name
    pub name: String,
    /// Field type
    pub type_info: FfiType,
    /// Field documentation
    pub documentation: Vec<String>,
    /// Field offset (if known)
    pub offset: Option<u32>,
    /// Whether the field is private
    pub is_private: bool,
    /// Field attributes
    pub attributes: Vec<FfiAttribute>,
}

/// Exported enum
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FfiEnum {
    /// Enum name
    pub name: String,
    /// C enum name
    pub c_name: String,
    /// Enum documentation
    pub documentation: Vec<String>,
    /// Enum variants
    pub variants: Vec<FfiEnumVariant>,
    /// Underlying integer type
    pub underlying_type: PrimitiveType,
    /// Whether this is a flags enum (bitfield)
    pub is_flags: bool,
    /// Feature flags required for this enum
    pub required_features: Vec<String>,
    /// Platform availability
    pub platforms: Vec<String>,
    /// Deprecation information
    pub deprecation: Option<DeprecationInfo>,
}

/// Enum variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiEnumVariant {
    /// Variant name
    pub name: String,
    /// C constant name
    pub c_name: String,
    /// Variant value
    pub value: i64,
    /// Variant documentation
    pub documentation: Vec<String>,
    /// Deprecation information
    pub deprecation: Option<DeprecationInfo>,
}

/// Exported constant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiConstant {
    /// Constant name
    pub name: String,
    /// C constant name
    pub c_name: String,
    /// Constant type
    pub type_info: FfiType,
    /// Constant value
    pub value: ConstantValue,
    /// Constant documentation
    pub documentation: Vec<String>,
    /// Feature flags required for this constant
    pub required_features: Vec<String>,
    /// Platform availability
    pub platforms: Vec<String>,
    /// Deprecation information
    pub deprecation: Option<DeprecationInfo>,
}

/// Constant value representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstantValue {
    Integer(i64),
    UInteger(u64),
    Float(f64),
    String(String),
    Boolean(bool),
    Null,
}

/// Type alias
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiTypeAlias {
    /// Alias name
    pub name: String,
    /// C type name
    pub c_name: String,
    /// Target type
    pub target_type: FfiType,
    /// Type alias documentation
    pub documentation: Vec<String>,
    /// Feature flags required for this type alias
    pub required_features: Vec<String>,
    /// Platform availability
    pub platforms: Vec<String>,
    /// Deprecation information
    pub deprecation: Option<DeprecationInfo>,
}

/// Deprecation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeprecationInfo {
    /// Deprecation message
    pub message: String,
    /// Version when deprecated
    pub since_version: Option<String>,
    /// Replacement function/type
    pub replacement: Option<String>,
    /// Version when it will be removed
    pub removal_version: Option<String>,
}

/// FFI attribute (for additional metadata)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiAttribute {
    /// Attribute name
    pub name: String,
    /// Attribute value
    pub value: Option<String>,
}

impl FfiType {
    /// Check if this type is a pointer
    pub fn is_pointer(&self) -> bool {
        self.is_pointer || self.pointer_level > 0
    }

    /// Check if this type is const
    pub fn is_const(&self) -> bool {
        self.is_const
    }

    /// Check if this type represents an error type
    pub fn is_error_type(&self) -> bool {
        self.name == "TrustformersError" || self.name == "c_int" && self.name.contains("error")
    }

    /// Check if this type is a handle type
    pub fn is_handle(&self) -> bool {
        self.primitive_type == Some(PrimitiveType::Handle)
            || self.name.ends_with("Handle")
            || self.primitive_type == Some(PrimitiveType::OpaquePointer)
    }

    /// Check if this type is a string type
    pub fn is_string(&self) -> bool {
        self.primitive_type == Some(PrimitiveType::CString)
            || (self.is_pointer() && self.name == "c_char")
    }

    /// Check if this type is nullable
    pub fn is_nullable(&self) -> bool {
        self.is_pointer()
    }

    /// Get the base type (removing pointer indirection)
    pub fn base_type(&self) -> String {
        if self.is_pointer() {
            // Remove pointer syntax
            self.name
                .trim_start_matches("*const ")
                .trim_start_matches("*mut ")
                .trim_start_matches("*")
                .to_string()
        } else {
            self.name.clone()
        }
    }

    /// Get a language-specific type mapping
    pub fn map_to_language(&self, language: &TargetLanguage) -> String {
        match language {
            TargetLanguage::Python => self.map_to_python(),
            TargetLanguage::Java => self.map_to_java(),
            TargetLanguage::Go => self.map_to_go(),
            TargetLanguage::CSharp => self.map_to_csharp(),
            TargetLanguage::TypeScript => self.map_to_typescript(),
            _ => self.name.clone(), // Default to original name
        }
    }

    fn map_to_python(&self) -> String {
        if self.is_string() {
            return "str".to_string();
        }
        if self.is_handle() {
            return "int".to_string(); // Handle as integer ID
        }

        match self.primitive_type {
            Some(PrimitiveType::Int8) => "int",
            Some(PrimitiveType::Int16) => "int",
            Some(PrimitiveType::Int32) => "int",
            Some(PrimitiveType::Int64) => "int",
            Some(PrimitiveType::UInt8) => "int",
            Some(PrimitiveType::UInt16) => "int",
            Some(PrimitiveType::UInt32) => "int",
            Some(PrimitiveType::UInt64) => "int",
            Some(PrimitiveType::Float32) => "float",
            Some(PrimitiveType::Float64) => "float",
            Some(PrimitiveType::Bool) => "bool",
            Some(PrimitiveType::Void) => "None",
            _ => "Any",
        }
        .to_string()
    }

    fn map_to_java(&self) -> String {
        if self.is_string() {
            return "String".to_string();
        }
        if self.is_handle() {
            return "long".to_string(); // Handle as long integer
        }

        match self.primitive_type {
            Some(PrimitiveType::Int8) => "byte",
            Some(PrimitiveType::Int16) => "short",
            Some(PrimitiveType::Int32) => "int",
            Some(PrimitiveType::Int64) => "long",
            Some(PrimitiveType::UInt8) => "byte", // Note: Java doesn't have unsigned
            Some(PrimitiveType::UInt16) => "short",
            Some(PrimitiveType::UInt32) => "int",
            Some(PrimitiveType::UInt64) => "long",
            Some(PrimitiveType::Float32) => "float",
            Some(PrimitiveType::Float64) => "double",
            Some(PrimitiveType::Bool) => "boolean",
            Some(PrimitiveType::Void) => "void",
            _ => "Object",
        }
        .to_string()
    }

    fn map_to_go(&self) -> String {
        if self.is_string() {
            return "string".to_string();
        }
        if self.is_handle() {
            return "uintptr".to_string();
        }

        match self.primitive_type {
            Some(PrimitiveType::Int8) => "int8",
            Some(PrimitiveType::Int16) => "int16",
            Some(PrimitiveType::Int32) => "int32",
            Some(PrimitiveType::Int64) => "int64",
            Some(PrimitiveType::UInt8) => "uint8",
            Some(PrimitiveType::UInt16) => "uint16",
            Some(PrimitiveType::UInt32) => "uint32",
            Some(PrimitiveType::UInt64) => "uint64",
            Some(PrimitiveType::Float32) => "float32",
            Some(PrimitiveType::Float64) => "float64",
            Some(PrimitiveType::Bool) => "bool",
            Some(PrimitiveType::Void) => "",
            _ => "interface{}",
        }
        .to_string()
    }

    fn map_to_csharp(&self) -> String {
        if self.is_string() {
            return "string".to_string();
        }
        if self.is_handle() {
            return "IntPtr".to_string();
        }

        match self.primitive_type {
            Some(PrimitiveType::Int8) => "sbyte",
            Some(PrimitiveType::Int16) => "short",
            Some(PrimitiveType::Int32) => "int",
            Some(PrimitiveType::Int64) => "long",
            Some(PrimitiveType::UInt8) => "byte",
            Some(PrimitiveType::UInt16) => "ushort",
            Some(PrimitiveType::UInt32) => "uint",
            Some(PrimitiveType::UInt64) => "ulong",
            Some(PrimitiveType::Float32) => "float",
            Some(PrimitiveType::Float64) => "double",
            Some(PrimitiveType::Bool) => "bool",
            Some(PrimitiveType::Void) => "void",
            _ => "object",
        }
        .to_string()
    }

    fn map_to_typescript(&self) -> String {
        if self.is_string() {
            return "string".to_string();
        }
        if self.is_handle() {
            return "number".to_string();
        }

        match self.primitive_type {
            Some(PrimitiveType::Bool) => "boolean",
            Some(PrimitiveType::Void) => "void",
            Some(_) => "number", // All numeric types map to number in TypeScript
            None => "any",
        }
        .to_string()
    }
}

impl FfiFunction {
    /// Check if this function can fail (returns an error)
    pub fn can_fail(&self) -> bool {
        self.can_fail || self.return_type.is_error_type()
    }

    /// Check if this function is deprecated
    pub fn is_deprecated(&self) -> bool {
        self.deprecation.is_some()
    }

    /// Get the function signature as a string
    pub fn signature(&self) -> String {
        let params: Vec<String> = self
            .parameters
            .iter()
            .map(|p| format!("{}: {}", p.name, p.type_info.name))
            .collect();

        format!(
            "{}({}) -> {}",
            self.name,
            params.join(", "),
            self.return_type.name
        )
    }

    /// Check if this function requires a specific feature
    pub fn requires_feature(&self, feature: &str) -> bool {
        self.required_features.contains(&feature.to_string())
    }

    /// Check if this function is available on a specific platform
    pub fn available_on_platform(&self, platform: &str) -> bool {
        self.platforms.is_empty() || self.platforms.contains(&platform.to_string())
    }
}

impl FfiStruct {
    /// Check if this struct is deprecated
    pub fn is_deprecated(&self) -> bool {
        self.deprecation.is_some()
    }

    /// Get all public fields
    pub fn public_fields(&self) -> Vec<&FfiField> {
        self.fields.iter().filter(|f| !f.is_private).collect()
    }

    /// Get the struct size estimation
    pub fn estimated_size(&self) -> u32 {
        if self.is_opaque {
            return 8; // Assume pointer size for opaque structs
        }

        self.fields.iter().map(|f| f.estimated_size()).sum()
    }
}

impl FfiField {
    /// Estimate the size of this field in bytes
    pub fn estimated_size(&self) -> u32 {
        match self.type_info.primitive_type {
            Some(PrimitiveType::Int8) | Some(PrimitiveType::UInt8) | Some(PrimitiveType::Bool) => 1,
            Some(PrimitiveType::Int16) | Some(PrimitiveType::UInt16) => 2,
            Some(PrimitiveType::Int32)
            | Some(PrimitiveType::UInt32)
            | Some(PrimitiveType::Float32) => 4,
            Some(PrimitiveType::Int64)
            | Some(PrimitiveType::UInt64)
            | Some(PrimitiveType::Float64) => 8,
            Some(PrimitiveType::IntPtr) | Some(PrimitiveType::UIntPtr) => 8, // Assume 64-bit
            _ if self.type_info.is_pointer() => 8,                           // Pointer size
            _ => 8,                                                          // Default assumption
        }
    }
}

impl FfiEnum {
    /// Check if this enum is deprecated
    pub fn is_deprecated(&self) -> bool {
        self.deprecation.is_some()
    }

    /// Get all non-deprecated variants
    pub fn active_variants(&self) -> Vec<&FfiEnumVariant> {
        self.variants.iter().filter(|v| v.deprecation.is_none()).collect()
    }

    /// Get the range of values in this enum
    pub fn value_range(&self) -> (i64, i64) {
        let values: Vec<i64> = self.variants.iter().map(|v| v.value).collect();
        let min = values.iter().min().copied().unwrap_or(0);
        let max = values.iter().max().copied().unwrap_or(0);
        (min, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_type_predicates() {
        let pointer_type = FfiType {
            name: "*const c_char".to_string(),
            is_pointer: true,
            is_const: true,
            pointer_level: 1,
            primitive_type: Some(PrimitiveType::CString),
            ..Default::default()
        };

        assert!(pointer_type.is_pointer());
        assert!(pointer_type.is_const());
        assert!(pointer_type.is_string());
        assert!(pointer_type.is_nullable());
    }

    #[test]
    fn test_language_type_mapping() {
        let int_type = FfiType {
            name: "c_int".to_string(),
            primitive_type: Some(PrimitiveType::Int32),
            ..Default::default()
        };

        assert_eq!(int_type.map_to_python(), "int");
        assert_eq!(int_type.map_to_java(), "int");
        assert_eq!(int_type.map_to_go(), "int32");
        assert_eq!(int_type.map_to_csharp(), "int");
        assert_eq!(int_type.map_to_typescript(), "number");
    }

    #[test]
    fn test_function_signature() {
        let function = FfiFunction {
            name: "test_function".to_string(),
            parameters: vec![FfiParameter {
                name: "param1".to_string(),
                type_info: FfiType {
                    name: "c_int".to_string(),
                    primitive_type: Some(PrimitiveType::Int32),
                    ..Default::default()
                },
                ..Default::default()
            }],
            return_type: FfiType {
                name: "c_int".to_string(),
                primitive_type: Some(PrimitiveType::Int32),
                ..Default::default()
            },
            ..Default::default()
        };

        assert_eq!(
            function.signature(),
            "test_function(param1: c_int) -> c_int"
        );
    }
}

// Default implementations for easier construction in tests and parsers
impl Default for FfiFunction {
    fn default() -> Self {
        Self {
            name: String::new(),
            c_name: String::new(),
            documentation: Vec::new(),
            parameters: Vec::new(),
            return_type: FfiType::default(),
            is_unsafe: false,
            can_fail: false,
            required_features: Vec::new(),
            platforms: Vec::new(),
            deprecation: None,
            attributes: Vec::new(),
        }
    }
}

impl Default for FfiType {
    fn default() -> Self {
        Self {
            name: "void".to_string(),
            is_pointer: false,
            is_const: false,
            is_mutable: false,
            pointer_level: 0,
            array_size: None,
            generic_params: Vec::new(),
            primitive_type: Some(PrimitiveType::Void),
            is_callback: false,
            callback_signature: None,
        }
    }
}

impl Default for FfiParameter {
    fn default() -> Self {
        Self {
            name: String::new(),
            type_info: FfiType::default(),
            documentation: Vec::new(),
            is_optional: false,
            default_value: None,
            attributes: Vec::new(),
        }
    }
}
