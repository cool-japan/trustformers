//! TypeScript bindings generator for FFI interfaces

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::codegen::ast::{FfiEnum, FfiFunction, FfiInterface, FfiStruct, FfiType};
use crate::codegen::templates::TemplateEngine;
use crate::codegen::{CodeGenConfig, TargetLanguage};
use crate::error::TrustformersResult;

use super::common::{function_can_fail, TypeMapper};
use super::LanguageGenerator;

/// TypeScript bindings generator
pub struct TypeScriptGenerator {
    config: CodeGenConfig,
    type_mappings: HashMap<String, String>,
}

impl TypeScriptGenerator {
    pub fn new(config: &CodeGenConfig) -> TrustformersResult<Self> {
        let mut type_mappings = HashMap::new();

        // Basic type mappings for TypeScript FFI
        type_mappings.insert("c_int".to_string(), "number".to_string());
        type_mappings.insert("c_uint".to_string(), "number".to_string());
        type_mappings.insert("c_long".to_string(), "number".to_string());
        type_mappings.insert("c_ulong".to_string(), "number".to_string());
        type_mappings.insert("c_float".to_string(), "number".to_string());
        type_mappings.insert("c_double".to_string(), "number".to_string());
        type_mappings.insert("c_char".to_string(), "string".to_string());
        type_mappings.insert("c_bool".to_string(), "boolean".to_string());
        type_mappings.insert("c_void".to_string(), "void".to_string());
        type_mappings.insert("*const c_char".to_string(), "string".to_string());
        type_mappings.insert("*mut c_char".to_string(), "string".to_string());
        type_mappings.insert("*const c_void".to_string(), "Buffer".to_string());
        type_mappings.insert("*mut c_void".to_string(), "Buffer".to_string());

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

    fn generate_function_binding(&self, func: &FfiFunction) -> String {
        let mut lines = Vec::new();

        // Function documentation
        if !func.documentation.is_empty() {
            lines.push("  /**".to_string());
            for doc_line in &func.documentation {
                lines.push(format!("   * {}", doc_line));
            }
            lines.push("   */".to_string());
        }

        // Function signature
        let params: Vec<String> = func
            .parameters
            .iter()
            .map(|p| format!("{}: {}", p.name, self.map_type(&p.type_info)))
            .collect();

        let return_type = self.map_type(&func.return_type);

        lines.push(format!(
            "  {}({}): {};",
            func.name,
            params.join(", "),
            return_type
        ));

        lines.join("\n")
    }

    fn generate_struct_binding(&self, struct_def: &FfiStruct) -> String {
        let mut lines = Vec::new();

        // Struct documentation
        if !struct_def.documentation.is_empty() {
            lines.push("/**".to_string());
            for doc_line in &struct_def.documentation {
                lines.push(format!(" * {}", doc_line));
            }
            lines.push(" */".to_string());
        }

        if struct_def.is_opaque {
            // Opaque struct - just a type alias to Buffer
            lines.push(format!("export type {} = Buffer;", struct_def.name));
        } else {
            // Regular struct with fields
            lines.push(format!("export interface {} {{", struct_def.name));

            for field in &struct_def.fields {
                if !field.is_private {
                    let field_type = self.map_type(&field.type_info);
                    lines.push(format!("  {}: {};", field.name, field_type));
                }
            }

            lines.push("}".to_string());
        }

        lines.join("\n")
    }

    fn generate_enum_binding(&self, enum_def: &FfiEnum) -> String {
        let mut lines = Vec::new();

        // Enum documentation
        if !enum_def.documentation.is_empty() {
            lines.push("/**".to_string());
            for doc_line in &enum_def.documentation {
                lines.push(format!(" * {}", doc_line));
            }
            lines.push(" */".to_string());
        }

        lines.push(format!("export enum {} {{", enum_def.name));

        for variant in &enum_def.variants {
            if let Some(deprecation) = &variant.deprecation {
                lines.push(format!("  /** @deprecated {} */", deprecation.message));
            }
            lines.push(format!("  {} = {},", variant.name, variant.value));
        }

        lines.push("}".to_string());

        lines.join("\n")
    }
}

impl TypeMapper for TypeScriptGenerator {
    fn map_type(&self, ffi_type: &FfiType) -> String {
        if let Some(mapped) = self.type_mappings.get(&ffi_type.name) {
            return mapped.clone();
        }

        if ffi_type.is_pointer() {
            if ffi_type.is_string() {
                return "string".to_string();
            } else if ffi_type.base_type() == "c_void" {
                return "Buffer".to_string();
            } else {
                return format!("{}[]", self.map_base_type(&ffi_type.base_type()));
            }
        }

        if let Some(size) = ffi_type.array_size {
            let base_type = self.map_base_type(&ffi_type.base_type());
            return format!("{}[]", base_type);
        }

        self.map_base_type(&ffi_type.name)
    }

    fn map_base_type(&self, type_name: &str) -> String {
        self.map_base_type(type_name)
    }

    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::TypeScript
    }
}

impl LanguageGenerator for TypeScriptGenerator {
    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::TypeScript
    }

    fn file_extension(&self) -> &'static str {
        "ts"
    }

    fn generate(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        let mut main_content = Vec::new();

        // Module header
        main_content.push("/**".to_string());
        main_content.push(format!(
            " * TypeScript bindings for {}",
            &interface.metadata.library_name
        ));
        main_content.push(" *".to_string());
        main_content.push(" * TrustformeRS C API bindings".to_string());
        main_content.push(format!(" * Version: {}", &interface.metadata.version));
        main_content.push(" */".to_string());
        main_content.push("".to_string());

        // Imports
        main_content.push("import * as ffi from 'ffi-napi';".to_string());
        main_content.push("import * as ref from 'ref-napi';".to_string());
        main_content.push("".to_string());

        // Error class
        main_content.push("export class TrustformersError extends Error {".to_string());
        main_content.push("  constructor(message: string) {".to_string());
        main_content.push("    super(message);".to_string());
        main_content.push("    this.name = 'TrustformersError';".to_string());
        main_content.push("  }".to_string());
        main_content.push("}".to_string());
        main_content.push("".to_string());

        // Generate enums
        for enum_def in &interface.enums {
            main_content.push(self.generate_enum_binding(enum_def));
            main_content.push("".to_string());
        }

        // Generate structs
        for struct_def in &interface.structs {
            main_content.push(self.generate_struct_binding(struct_def));
            main_content.push("".to_string());
        }

        // Library interface
        main_content.push("interface TrustformersLib {".to_string());
        for func in &interface.functions {
            main_content.push(self.generate_function_binding(func));
        }
        main_content.push("}".to_string());
        main_content.push("".to_string());

        // Library loading
        main_content.push("const libPath = process.platform === 'win32'".to_string());
        main_content.push("  ? 'trustformers_c.dll'".to_string());
        main_content.push("  : process.platform === 'darwin'".to_string());
        main_content.push("  ? 'libtrusformers_c.dylib'".to_string());
        main_content.push("  : 'libtrusformers_c.so';".to_string());
        main_content.push("".to_string());
        main_content
            .push("export const lib = ffi.Library(libPath, {}) as TrustformersLib;".to_string());

        // Write output file
        let output_path = output_dir.join("trustformers.ts");
        fs::write(output_path, main_content.join("\n"))?;

        Ok(())
    }
}
