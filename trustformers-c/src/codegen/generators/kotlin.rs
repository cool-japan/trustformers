//! Kotlin bindings generator for FFI interfaces
//!
//! Generates JNA (Java Native Access) bindings for the TrustformeRS C API in Kotlin.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::codegen::ast::{FfiEnum, FfiFunction, FfiInterface, FfiStruct, FfiType};
use crate::codegen::templates::TemplateEngine;
use crate::codegen::{CodeGenConfig, TargetLanguage};
use crate::error::TrustformersResult;

use super::common::{function_can_fail, TypeMapper};
use super::LanguageGenerator;

/// Kotlin bindings generator using JNA
pub struct KotlinGenerator {
    config: CodeGenConfig,
    type_mappings: HashMap<String, String>,
}

impl KotlinGenerator {
    pub fn new(config: &CodeGenConfig) -> TrustformersResult<Self> {
        let mut type_mappings = HashMap::new();

        // Basic type mappings for Kotlin JNA
        type_mappings.insert("c_int".to_string(), "Int".to_string());
        type_mappings.insert("c_uint".to_string(), "Int".to_string());
        type_mappings.insert("c_short".to_string(), "Short".to_string());
        type_mappings.insert("c_ushort".to_string(), "Short".to_string());
        type_mappings.insert("c_long".to_string(), "Long".to_string());
        type_mappings.insert("c_ulong".to_string(), "Long".to_string());
        type_mappings.insert("c_longlong".to_string(), "Long".to_string());
        type_mappings.insert("c_ulonglong".to_string(), "Long".to_string());
        type_mappings.insert("c_float".to_string(), "Float".to_string());
        type_mappings.insert("c_double".to_string(), "Double".to_string());
        type_mappings.insert("c_char".to_string(), "Byte".to_string());
        type_mappings.insert("c_uchar".to_string(), "Byte".to_string());
        type_mappings.insert("c_bool".to_string(), "Boolean".to_string());
        type_mappings.insert("c_void".to_string(), "Unit".to_string());
        type_mappings.insert("isize".to_string(), "Long".to_string());
        type_mappings.insert("usize".to_string(), "Long".to_string());

        // Pointer type mappings
        type_mappings.insert("*const c_char".to_string(), "String".to_string());
        type_mappings.insert("*mut c_char".to_string(), "String".to_string());
        type_mappings.insert("*const c_void".to_string(), "Pointer".to_string());
        type_mappings.insert("*mut c_void".to_string(), "Pointer".to_string());

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
            "c_int" | "i32" | "c_uint" | "u32" => "Int",
            "c_short" | "i16" | "c_ushort" | "u16" => "Short",
            "c_long" | "i64" | "c_ulong" | "u64" => "Long",
            "c_longlong" | "c_ulonglong" => "Long",
            "c_float" | "f32" => "Float",
            "c_double" | "f64" => "Double",
            "c_char" | "i8" | "c_uchar" | "u8" => "Byte",
            "c_bool" => "Boolean",
            "c_void" | "()" => "Unit",
            "isize" | "usize" => "Long",
            name if name.ends_with("Handle") => "Pointer",
            _ => "Pointer", // Default for unknown types
        }
        .to_string()
    }

    fn generate_data_class_for_struct(&self, struct_def: &FfiStruct) -> String {
        let mut lines = Vec::new();

        // Class documentation
        if !struct_def.documentation.is_empty() {
            lines.push("  /**".to_string());
            for doc_line in &struct_def.documentation {
                lines.push(format!("   * {}", doc_line));
            }
            lines.push("   */".to_string());
        }

        if struct_def.is_opaque {
            // Opaque struct - use PointerType
            lines.push(format!(
                "  class {}(p: Pointer? = null) : PointerType(p)",
                struct_def.name
            ));
        } else {
            // Regular struct extending JNA Structure
            lines.push(format!("  open class {} : Structure {{", struct_def.name));

            // Fields - using JvmField annotation for public access
            for field in &struct_def.fields {
                if !field.is_private {
                    if !field.documentation.is_empty() {
                        lines.push("    /**".to_string());
                        for doc_line in &field.documentation {
                            lines.push(format!("     * {}", doc_line));
                        }
                        lines.push("     */".to_string());
                    }
                    let field_type = self.map_type(&field.type_info);
                    lines.push(format!(
                        "    @JvmField var {}: {} = {}",
                        self.to_camel_case(&field.name),
                        field_type,
                        self.get_default_value(&field_type)
                    ));
                }
            }

            lines.push("".to_string());

            // getFieldOrder method required by JNA
            lines.push("    override fun getFieldOrder(): List<String> {".to_string());
            lines.push("      return listOf(".to_string());

            let visible_fields: Vec<String> = struct_def
                .fields
                .iter()
                .filter(|f| !f.is_private)
                .map(|f| format!("\"{}\"", self.to_camel_case(&f.name)))
                .collect();

            if !visible_fields.is_empty() {
                for (i, field_name) in visible_fields.iter().enumerate() {
                    if i == visible_fields.len() - 1 {
                        lines.push(format!("        {}", field_name));
                    } else {
                        lines.push(format!("        {},", field_name));
                    }
                }
            }

            lines.push("      )".to_string());
            lines.push("    }".to_string());

            // Constructors
            lines.push("".to_string());
            lines.push("    constructor() : super()".to_string());
            lines.push("".to_string());
            lines.push("    constructor(p: Pointer) : super(p) {".to_string());
            lines.push("      read()".to_string());
            lines.push("    }".to_string());

            lines.push("  }".to_string());
        }

        lines.join("\n")
    }

    fn generate_enum_class(&self, enum_def: &FfiEnum) -> String {
        let mut lines = Vec::new();

        // Enum documentation
        if !enum_def.documentation.is_empty() {
            lines.push("  /**".to_string());
            for doc_line in &enum_def.documentation {
                lines.push(format!("   * {}", doc_line));
            }
            lines.push("   */".to_string());
        }

        if enum_def.is_flags {
            // For flags/bitfield enums, use object with constants
            lines.push(format!("  object {} {{", enum_def.name));
            for variant in &enum_def.variants {
                if let Some(deprecation) = &variant.deprecation {
                    lines.push(format!("    @Deprecated(\"{}\")", deprecation.message));
                }
                if !variant.documentation.is_empty() {
                    lines.push("    /**".to_string());
                    for doc_line in &variant.documentation {
                        lines.push(format!("     * {}", doc_line));
                    }
                    lines.push("     */".to_string());
                }
                lines.push(format!(
                    "    const val {}: Int = {}",
                    variant.name, variant.value
                ));
            }
            lines.push("  }".to_string());
        } else {
            // For regular enums, use Kotlin enum class
            lines.push(format!("  enum class {}(val value: Int) {{", enum_def.name));

            for (i, variant) in enum_def.variants.iter().enumerate() {
                if let Some(deprecation) = &variant.deprecation {
                    lines.push(format!("    @Deprecated(\"{}\")", deprecation.message));
                }
                if !variant.documentation.is_empty() {
                    lines.push("    /**".to_string());
                    for doc_line in &variant.documentation {
                        lines.push(format!("     * {}", doc_line));
                    }
                    lines.push("     */".to_string());
                }

                let terminator = if i == enum_def.variants.len() - 1 { ";" } else { "," };
                lines.push(format!(
                    "    {}({}){}",
                    variant.name, variant.value, terminator
                ));
            }

            lines.push("".to_string());

            // Companion object with fromValue method
            lines.push("    companion object {".to_string());
            lines.push("      @JvmStatic".to_string());
            lines.push(format!(
                "      fun fromValue(value: Int): {} {{",
                enum_def.name
            ));
            lines.push("        return values().find { it.value == value }".to_string());
            lines.push(
                "          ?: throw IllegalArgumentException(\"Unknown enum value: $value\")"
                    .to_string(),
            );
            lines.push("      }".to_string());
            lines.push("    }".to_string());

            lines.push("  }".to_string());
        }

        lines.join("\n")
    }

    fn generate_interface_for_functions(&self, functions: &[FfiFunction]) -> String {
        let mut lines = Vec::new();

        lines.push("  /**".to_string());
        lines.push("   * JNA Library interface for TrustformeRS C API".to_string());
        lines.push("   */".to_string());
        lines.push("  interface TrustformersLibrary : Library {".to_string());

        for func in functions {
            // Function documentation
            if !func.documentation.is_empty() {
                lines.push("".to_string());
                lines.push("    /**".to_string());
                for doc_line in &func.documentation {
                    lines.push(format!("     * {}", doc_line));
                }

                // Add parameter documentation
                if !func.parameters.is_empty() {
                    lines.push("     *".to_string());
                    for param in &func.parameters {
                        if !param.documentation.is_empty() {
                            lines.push(format!(
                                "     * @param {} {}",
                                self.to_camel_case(&param.name),
                                param.documentation.join(" ")
                            ));
                        } else {
                            lines.push(format!(
                                "     * @param {} parameter",
                                self.to_camel_case(&param.name)
                            ));
                        }
                    }
                }

                // Add return documentation
                if func.return_type.name != "c_void" && func.return_type.name != "()" {
                    lines.push("     * @return return value".to_string());
                }

                lines.push("     */".to_string());
            }

            // Deprecation annotation
            if let Some(deprecation) = &func.deprecation {
                lines.push(format!("    @Deprecated(\"{}\")", deprecation.message));
            }

            // Function signature with @JvmName annotation for C name mapping
            if func.name != func.c_name {
                lines.push(format!("    @JvmName(\"{}\")", func.c_name));
            }

            let params: Vec<String> = func
                .parameters
                .iter()
                .map(|p| {
                    format!(
                        "{}: {}",
                        self.to_camel_case(&p.name),
                        self.map_type(&p.type_info)
                    )
                })
                .collect();

            let return_type = self.map_type(&func.return_type);

            lines.push(format!(
                "    fun {}({}): {}",
                self.to_camel_case(&func.c_name),
                params.join(", "),
                return_type
            ));
        }

        lines.push("  }".to_string());

        lines.join("\n")
    }

    fn to_camel_case(&self, snake_case: &str) -> String {
        let mut result = String::new();
        let mut capitalize_next = false;

        for ch in snake_case.chars() {
            if ch == '_' {
                capitalize_next = true;
            } else if capitalize_next {
                result.push(ch.to_ascii_uppercase());
                capitalize_next = false;
            } else {
                result.push(ch);
            }
        }

        result
    }

    fn to_pascal_case(&self, snake_case: &str) -> String {
        let camel = self.to_camel_case(snake_case);
        if camel.is_empty() {
            return camel;
        }

        let mut chars = camel.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        }
    }

    fn get_default_value(&self, type_name: &str) -> &str {
        match type_name {
            "Int" | "Short" | "Long" | "Byte" => "0",
            "Float" => "0.0f",
            "Double" => "0.0",
            "Boolean" => "false",
            "String" => "\"\"",
            _ => "Pointer.NULL",
        }
    }
}

impl TypeMapper for KotlinGenerator {
    fn map_type(&self, ffi_type: &FfiType) -> String {
        // Check for explicit mapping first
        if let Some(mapped) = self.type_mappings.get(&ffi_type.name) {
            return mapped.clone();
        }

        // Handle pointer types
        if ffi_type.is_pointer() {
            if ffi_type.is_string() {
                return "String?".to_string();
            } else if ffi_type.base_type() == "c_void" {
                return "Pointer?".to_string();
            } else {
                // For typed pointers, use Pointer or array
                let base_type = self.map_base_type(&ffi_type.base_type());
                if base_type == "Byte" || base_type == "Int" || base_type == "Float" {
                    return format!("{}Array?", base_type);
                } else {
                    return "Pointer?".to_string();
                }
            }
        }

        // Handle arrays
        if let Some(_size) = ffi_type.array_size {
            let base_type = self.map_base_type(&ffi_type.base_type());
            return format!("{}Array", base_type);
        }

        // Handle callbacks
        if ffi_type.is_callback {
            return "Callback?".to_string();
        }

        // Default to base type mapping
        self.map_base_type(&ffi_type.name)
    }

    fn map_base_type(&self, type_name: &str) -> String {
        self.map_base_type(type_name)
    }

    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::Kotlin
    }
}

impl LanguageGenerator for KotlinGenerator {
    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::Kotlin
    }

    fn file_extension(&self) -> &'static str {
        "kt"
    }

    fn generate(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        let mut main_content = Vec::new();

        // Package declaration
        let package_name = self.config.package_name.as_deref().unwrap_or("com.trustformers.ffi");
        main_content.push(format!("package {}", package_name));
        main_content.push("".to_string());

        // File header documentation
        main_content.push("/**".to_string());
        main_content.push(format!(
            " * Kotlin bindings for {}",
            &interface.metadata.library_name
        ));
        main_content.push(" *".to_string());
        main_content
            .push(" * TrustformeRS C API bindings using JNA (Java Native Access)".to_string());
        main_content.push(format!(" * Version: {}", &interface.metadata.version));
        main_content.push(" *".to_string());
        main_content.push(" * This file is auto-generated. Do not edit manually.".to_string());
        main_content.push(" */".to_string());
        main_content.push("".to_string());

        // Imports
        main_content.push("import com.sun.jna.Library".to_string());
        main_content.push("import com.sun.jna.Native".to_string());
        main_content.push("import com.sun.jna.Pointer".to_string());
        main_content.push("import com.sun.jna.PointerType".to_string());
        main_content.push("import com.sun.jna.Structure".to_string());
        main_content.push("import com.sun.jna.Callback".to_string());
        main_content.push("import java.nio.ByteBuffer".to_string());
        main_content.push("".to_string());

        // Suppress warnings for JNA
        main_content.push("@Suppress(\"unused\", \"MemberVisibilityCanBePrivate\")".to_string());

        // Main wrapper object (Kotlin singleton)
        main_content.push("/**".to_string());
        main_content.push(" * Main TrustformeRS JNA wrapper object".to_string());
        main_content.push(" */".to_string());
        main_content.push("object Trustformers {".to_string());
        main_content.push("".to_string());

        // Error class
        main_content.push("  /**".to_string());
        main_content.push("   * TrustformeRS exception class".to_string());
        main_content.push("   */".to_string());
        main_content.push("  class TrustformersException : Exception {".to_string());
        main_content.push("    constructor(message: String) : super(message)".to_string());
        main_content.push("".to_string());
        main_content.push(
            "    constructor(message: String, cause: Throwable) : super(message, cause)"
                .to_string(),
        );
        main_content.push("  }".to_string());
        main_content.push("".to_string());

        // Generate enums
        for enum_def in &interface.enums {
            main_content.push(self.generate_enum_class(enum_def));
            main_content.push("".to_string());
        }

        // Generate structs
        for struct_def in &interface.structs {
            main_content.push(self.generate_data_class_for_struct(struct_def));
            main_content.push("".to_string());
        }

        // Generate library interface
        main_content.push(self.generate_interface_for_functions(&interface.functions));
        main_content.push("".to_string());

        // Library instance and loading
        main_content.push("  /**".to_string());
        main_content.push("   * Native library instance (lazy-initialized)".to_string());
        main_content.push("   */".to_string());
        main_content.push("  private val instance: TrustformersLibrary by lazy {".to_string());
        main_content.push("    val libName = \"trustformers_c\"".to_string());
        main_content.push("    try {".to_string());
        main_content
            .push("      Native.load(libName, TrustformersLibrary::class.java)".to_string());
        main_content.push("    } catch (e: UnsatisfiedLinkError) {".to_string());
        main_content.push("      throw TrustformersException(".to_string());
        main_content
            .push("        \"Failed to load TrustformeRS native library: $libName\",".to_string());
        main_content.push("        e".to_string());
        main_content.push("      )".to_string());
        main_content.push("    }".to_string());
        main_content.push("  }".to_string());
        main_content.push("".to_string());

        // Public getter for library instance
        main_content.push("  /**".to_string());
        main_content.push("   * Get library instance, loading if necessary".to_string());
        main_content.push("   */".to_string());
        main_content.push("  @JvmStatic".to_string());
        main_content.push("  fun getInstance(): TrustformersLibrary = instance".to_string());
        main_content.push("".to_string());

        // Static initializer block
        main_content.push("  init {".to_string());
        main_content.push("    // Configure JNA options".to_string());
        main_content.push("    Native.setProtected(true)".to_string());
        main_content.push("  }".to_string());

        // Close main object
        main_content.push("}".to_string());

        // Write output file
        let output_path = output_dir.join("Trustformers.kt");
        fs::write(output_path, main_content.join("\n"))?;

        Ok(())
    }
}
