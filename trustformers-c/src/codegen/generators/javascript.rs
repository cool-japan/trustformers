//! JavaScript bindings generator for FFI interfaces

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::codegen::ast::{FfiEnum, FfiFunction, FfiInterface, FfiStruct, FfiType};
use crate::codegen::templates::TemplateEngine;
use crate::codegen::{CodeGenConfig, TargetLanguage};
use crate::error::TrustformersResult;

use super::common::{function_can_fail, TypeMapper};
use super::LanguageGenerator;

/// JavaScript bindings generator
pub struct JavaScriptGenerator {
    config: CodeGenConfig,
    type_mappings: HashMap<String, String>,
}

impl JavaScriptGenerator {
    pub fn new(config: &CodeGenConfig) -> TrustformersResult<Self> {
        let mut type_mappings = HashMap::new();

        // FFI type mappings for ffi-napi (JavaScript FFI library)
        // These are the type strings used in ffi.Library() function declarations
        type_mappings.insert("c_int".to_string(), "int".to_string());
        type_mappings.insert("c_uint".to_string(), "uint".to_string());
        type_mappings.insert("c_short".to_string(), "short".to_string());
        type_mappings.insert("c_ushort".to_string(), "ushort".to_string());
        type_mappings.insert("c_long".to_string(), "long".to_string());
        type_mappings.insert("c_ulong".to_string(), "ulong".to_string());
        type_mappings.insert("c_longlong".to_string(), "longlong".to_string());
        type_mappings.insert("c_ulonglong".to_string(), "ulonglong".to_string());
        type_mappings.insert("c_float".to_string(), "float".to_string());
        type_mappings.insert("c_double".to_string(), "double".to_string());
        type_mappings.insert("c_char".to_string(), "char".to_string());
        type_mappings.insert("c_uchar".to_string(), "uchar".to_string());
        type_mappings.insert("c_bool".to_string(), "bool".to_string());
        type_mappings.insert("c_void".to_string(), "void".to_string());
        type_mappings.insert("*const c_char".to_string(), "string".to_string());
        type_mappings.insert("*mut c_char".to_string(), "string".to_string());
        type_mappings.insert("*const c_void".to_string(), "pointer".to_string());
        type_mappings.insert("*mut c_void".to_string(), "pointer".to_string());
        type_mappings.insert("isize".to_string(), "long".to_string());
        type_mappings.insert("usize".to_string(), "ulong".to_string());

        // Add custom type mappings from config
        for (k, v) in &config.type_mappings {
            type_mappings.insert(k.clone(), v.target_type.clone());
        }

        Ok(Self {
            config: config.clone(),
            type_mappings,
        })
    }

    fn map_base_type(&self, type_name: &str) -> String {
        match type_name {
            "c_int" | "i32" => "int",
            "c_uint" | "u32" => "uint",
            "c_short" | "i16" => "short",
            "c_ushort" | "u16" => "ushort",
            "c_long" | "i64" => "long",
            "c_ulong" | "u64" => "ulong",
            "c_longlong" => "longlong",
            "c_ulonglong" => "ulonglong",
            "c_float" | "f32" => "float",
            "c_double" | "f64" => "double",
            "c_char" | "i8" => "char",
            "c_uchar" | "u8" => "uchar",
            "c_bool" => "bool",
            "c_void" | "()" => "void",
            "isize" => "long",
            "usize" => "ulong",
            name if name.ends_with("Handle") => "pointer",
            _ => "pointer",
        }
        .to_string()
    }

    fn generate_function_binding(&self, func: &FfiFunction) -> String {
        let mut lines = Vec::new();

        // Generate ffi-napi function signature
        let params: Vec<String> = func
            .parameters
            .iter()
            .map(|p| format!("'{}'", self.map_type(&p.type_info)))
            .collect();

        let return_type = self.map_type(&func.return_type);

        // Format: 'function_name': ['return_type', ['param1_type', 'param2_type', ...]]
        lines.push(format!(
            "  '{}': ['{}', [{}]],",
            func.c_name,
            return_type,
            params.join(", ")
        ));

        lines.join("\n")
    }

    fn generate_struct_class(&self, struct_def: &FfiStruct) -> String {
        let mut lines = Vec::new();

        // JSDoc documentation
        if !struct_def.documentation.is_empty() {
            lines.push("/**".to_string());
            for doc_line in &struct_def.documentation {
                lines.push(format!(" * {}", doc_line));
            }
            lines.push(" */".to_string());
        }

        if struct_def.is_opaque {
            // Opaque struct - just a wrapper around a pointer
            lines.push(format!("class {} {{", struct_def.name));
            lines.push("  /**".to_string());
            lines.push("   * @param {Buffer} handle - Native handle".to_string());
            lines.push("   */".to_string());
            lines.push("  constructor(handle) {".to_string());
            lines.push("    this.handle = handle;".to_string());
            lines.push("  }".to_string());
            lines.push("}".to_string());
        } else {
            // Regular struct with fields
            lines.push(format!("class {} {{", struct_def.name));

            // Constructor with all fields
            lines.push("  /**".to_string());
            for field in &struct_def.fields {
                if !field.is_private {
                    lines.push(format!(
                        "   * @param {{{}}} {}",
                        self.map_type_to_jsdoc(&field.type_info),
                        field.name
                    ));
                }
            }
            lines.push("   */".to_string());

            let field_names: Vec<String> = struct_def
                .fields
                .iter()
                .filter(|f| !f.is_private)
                .map(|f| f.name.clone())
                .collect();

            lines.push(format!("  constructor({}) {{", field_names.join(", ")));
            for field in &struct_def.fields {
                if !field.is_private {
                    lines.push(format!("    this.{} = {};", field.name, field.name));
                }
            }
            lines.push("  }".to_string());
            lines.push("}".to_string());
        }

        lines.join("\n")
    }

    fn generate_enum_object(&self, enum_def: &FfiEnum) -> String {
        let mut lines = Vec::new();

        // JSDoc documentation
        if !enum_def.documentation.is_empty() {
            lines.push("/**".to_string());
            for doc_line in &enum_def.documentation {
                lines.push(format!(" * {}", doc_line));
            }
            if enum_def.is_flags {
                lines.push(" * @enum {number}".to_string());
                lines.push(" * @readonly".to_string());
            } else {
                lines.push(" * @enum {number}".to_string());
                lines.push(" * @readonly".to_string());
            }
            lines.push(" */".to_string());
        }

        lines.push(format!("const {} = {{", enum_def.name));

        for variant in &enum_def.variants {
            if let Some(deprecation) = &variant.deprecation {
                lines.push(format!("  /** @deprecated {} */", deprecation.message));
            }
            lines.push(format!("  {}: {},", variant.name, variant.value));
        }

        lines.push("};".to_string());

        // Make the enum object immutable
        lines.push(format!("Object.freeze({});", enum_def.name));

        lines.join("\n")
    }

    fn map_type_to_jsdoc(&self, ffi_type: &FfiType) -> String {
        if ffi_type.is_string() {
            return "string".to_string();
        }
        if ffi_type.is_pointer() {
            if ffi_type.base_type() == "c_void" {
                return "Buffer".to_string();
            } else {
                return format!(
                    "Array<{}>",
                    self.map_type_to_jsdoc_base(&ffi_type.base_type())
                );
            }
        }
        if ffi_type.array_size.is_some() {
            return format!(
                "Array<{}>",
                self.map_type_to_jsdoc_base(&ffi_type.base_type())
            );
        }
        self.map_type_to_jsdoc_base(&ffi_type.name)
    }

    fn map_type_to_jsdoc_base(&self, type_name: &str) -> String {
        match type_name {
            "c_int" | "i32" | "c_uint" | "u32" => "number",
            "c_short" | "i16" | "c_ushort" | "u16" => "number",
            "c_long" | "i64" | "c_ulong" | "u64" => "number",
            "c_longlong" | "c_ulonglong" => "number",
            "c_float" | "f32" | "c_double" | "f64" => "number",
            "c_char" | "i8" | "c_uchar" | "u8" => "number",
            "c_bool" => "boolean",
            "c_void" | "()" => "void",
            "isize" | "usize" => "number",
            name if name.ends_with("Handle") => "Buffer",
            _ => "any",
        }
        .to_string()
    }
}

impl TypeMapper for JavaScriptGenerator {
    fn map_type(&self, ffi_type: &FfiType) -> String {
        if let Some(mapped) = self.type_mappings.get(&ffi_type.name) {
            return mapped.clone();
        }

        if ffi_type.is_pointer() {
            if ffi_type.is_string() {
                return "string".to_string();
            } else if ffi_type.base_type() == "c_void" {
                return "pointer".to_string();
            } else {
                // Array pointer
                return "pointer".to_string();
            }
        }

        if ffi_type.array_size.is_some() {
            return "pointer".to_string();
        }

        self.map_base_type(&ffi_type.name)
    }

    fn map_base_type(&self, type_name: &str) -> String {
        self.map_base_type(type_name)
    }

    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::JavaScript
    }
}

impl LanguageGenerator for JavaScriptGenerator {
    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::JavaScript
    }

    fn file_extension(&self) -> &'static str {
        "js"
    }

    fn generate(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        let mut main_content = Vec::new();

        // Module header with JSDoc
        main_content.push("/**".to_string());
        main_content.push(format!(
            " * JavaScript bindings for {}",
            &interface.metadata.library_name
        ));
        main_content.push(" *".to_string());
        main_content.push(" * TrustformeRS C API bindings".to_string());
        main_content.push(format!(" * Version: {}", &interface.metadata.version));
        main_content.push(" *".to_string());
        main_content.push(" * @module trustformers".to_string());
        main_content.push(" */".to_string());
        main_content.push("".to_string());

        // Strict mode
        main_content.push("'use strict';".to_string());
        main_content.push("".to_string());

        // Require statements
        main_content.push("const ffi = require('ffi-napi');".to_string());
        main_content.push("const ref = require('ref-napi');".to_string());
        main_content.push("const path = require('path');".to_string());
        main_content.push("".to_string());

        // Error class
        main_content.push("/**".to_string());
        main_content.push(" * TrustformeRS error class".to_string());
        main_content.push(" * @extends Error".to_string());
        main_content.push(" */".to_string());
        main_content.push("class TrustformersError extends Error {".to_string());
        main_content.push("  /**".to_string());
        main_content.push("   * @param {string} message - Error message".to_string());
        main_content.push("   */".to_string());
        main_content.push("  constructor(message) {".to_string());
        main_content.push("    super(message);".to_string());
        main_content.push("    this.name = 'TrustformersError';".to_string());
        main_content.push("  }".to_string());
        main_content.push("}".to_string());
        main_content.push("".to_string());

        // Generate enums
        for enum_def in &interface.enums {
            main_content.push(self.generate_enum_object(enum_def));
            main_content.push("".to_string());
        }

        // Generate structs
        for struct_def in &interface.structs {
            main_content.push(self.generate_struct_class(struct_def));
            main_content.push("".to_string());
        }

        // Library path detection
        main_content.push("/**".to_string());
        main_content
            .push(" * Detect the correct library path for the current platform".to_string());
        main_content.push(" * @returns {string} Library path".to_string());
        main_content.push(" */".to_string());
        main_content.push("function getLibraryPath() {".to_string());
        main_content.push("  const platform = process.platform;".to_string());
        main_content.push("  let libName;".to_string());
        main_content.push("".to_string());
        main_content.push("  if (platform === 'win32') {".to_string());
        main_content.push("    libName = 'trustformers_c.dll';".to_string());
        main_content.push("  } else if (platform === 'darwin') {".to_string());
        main_content.push("    libName = 'libtrusformers_c.dylib';".to_string());
        main_content.push("  } else {".to_string());
        main_content.push("    libName = 'libtrusformers_c.so';".to_string());
        main_content.push("  }".to_string());
        main_content.push("".to_string());
        main_content.push("  // Try to find library in standard locations".to_string());
        main_content.push("  const searchPaths = [".to_string());
        main_content.push("    path.join(__dirname, '..', 'native'),".to_string());
        main_content.push("    path.join(__dirname, 'native'),".to_string());
        main_content.push("    path.join(__dirname),".to_string());
        main_content.push("    libName // System library path".to_string());
        main_content.push("  ];".to_string());
        main_content.push("".to_string());
        main_content.push("  for (const searchPath of searchPaths) {".to_string());
        main_content.push("    const libPath = path.join(searchPath, libName);".to_string());
        main_content.push("    try {".to_string());
        main_content.push("      if (require('fs').existsSync(libPath)) {".to_string());
        main_content.push("        return libPath;".to_string());
        main_content.push("      }".to_string());
        main_content.push("    } catch (e) {".to_string());
        main_content.push("      // Continue searching".to_string());
        main_content.push("    }".to_string());
        main_content.push("  }".to_string());
        main_content.push("".to_string());
        main_content.push(
            "  // Default to library name (system will search in standard paths)".to_string(),
        );
        main_content.push("  return libName;".to_string());
        main_content.push("}".to_string());
        main_content.push("".to_string());

        // Library loading with function bindings
        main_content.push("/**".to_string());
        main_content.push(" * Native library bindings".to_string());
        main_content.push(" * @type {Object}".to_string());
        main_content.push(" */".to_string());
        main_content.push("const lib = ffi.Library(getLibraryPath(), {".to_string());

        for func in &interface.functions {
            main_content.push(self.generate_function_binding(func));
        }

        main_content.push("});".to_string());
        main_content.push("".to_string());

        // Module exports
        main_content.push("// Export all bindings".to_string());
        main_content.push("module.exports = {".to_string());
        main_content.push("  lib,".to_string());
        main_content.push("  TrustformersError,".to_string());

        // Export enums
        for enum_def in &interface.enums {
            main_content.push(format!("  {},", enum_def.name));
        }

        // Export structs
        for struct_def in &interface.structs {
            main_content.push(format!("  {},", struct_def.name));
        }

        main_content.push("};".to_string());

        // Write output file
        let output_path = output_dir.join("trustformers.js");
        fs::write(output_path, main_content.join("\n"))?;

        Ok(())
    }
}
