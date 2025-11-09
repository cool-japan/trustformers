//! Go bindings generator for FFI interfaces
//!
//! Generates CGO bindings for the TrustformeRS C API.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::codegen::ast::{FfiEnum, FfiFunction, FfiInterface, FfiStruct, FfiType, PrimitiveType};
use crate::codegen::templates::TemplateEngine;
use crate::codegen::{CodeGenConfig, TargetLanguage};
use crate::error::TrustformersResult;

use super::common::{function_can_fail, TypeMapper};
use super::LanguageGenerator;

/// Go bindings generator using CGO
pub struct GoGenerator {
    config: CodeGenConfig,
    type_mappings: HashMap<String, String>,
}

impl GoGenerator {
    pub fn new(config: &CodeGenConfig) -> TrustformersResult<Self> {
        let mut type_mappings = HashMap::new();

        // Basic type mappings for Go CGO
        type_mappings.insert("c_int".to_string(), "C.int".to_string());
        type_mappings.insert("c_uint".to_string(), "C.uint".to_string());
        type_mappings.insert("c_short".to_string(), "C.short".to_string());
        type_mappings.insert("c_ushort".to_string(), "C.ushort".to_string());
        type_mappings.insert("c_long".to_string(), "C.long".to_string());
        type_mappings.insert("c_ulong".to_string(), "C.ulong".to_string());
        type_mappings.insert("c_longlong".to_string(), "C.longlong".to_string());
        type_mappings.insert("c_ulonglong".to_string(), "C.ulonglong".to_string());
        type_mappings.insert("c_float".to_string(), "C.float".to_string());
        type_mappings.insert("c_double".to_string(), "C.double".to_string());
        type_mappings.insert("c_char".to_string(), "C.char".to_string());
        type_mappings.insert("c_uchar".to_string(), "C.uchar".to_string());
        type_mappings.insert("c_bool".to_string(), "C.bool".to_string());
        type_mappings.insert("isize".to_string(), "C.long".to_string());
        type_mappings.insert("usize".to_string(), "C.ulong".to_string());

        // Pointer type mappings
        type_mappings.insert("*const c_char".to_string(), "*C.char".to_string());
        type_mappings.insert("*mut c_char".to_string(), "*C.char".to_string());
        type_mappings.insert("*const c_void".to_string(), "unsafe.Pointer".to_string());
        type_mappings.insert("*mut c_void".to_string(), "unsafe.Pointer".to_string());

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
            "c_int" | "i32" => "int32",
            "c_uint" | "u32" => "uint32",
            "c_short" | "i16" => "int16",
            "c_ushort" | "u16" => "uint16",
            "c_long" | "i64" => "int64",
            "c_ulong" | "u64" => "uint64",
            "c_longlong" => "int64",
            "c_ulonglong" => "uint64",
            "c_float" | "f32" => "float32",
            "c_double" | "f64" => "float64",
            "c_char" | "i8" => "int8",
            "c_uchar" | "u8" => "uint8",
            "c_bool" => "bool",
            "c_void" | "()" => "",
            "isize" => "int64",
            "usize" => "uint64",
            name if name.ends_with("Handle") => "unsafe.Pointer",
            _ => "unsafe.Pointer", // Default for unknown types
        }
        .to_string()
    }

    fn map_c_type(&self, type_name: &str) -> String {
        match type_name {
            "c_int" | "i32" => "C.int",
            "c_uint" | "u32" => "C.uint",
            "c_short" | "i16" => "C.short",
            "c_ushort" | "u16" => "C.ushort",
            "c_long" | "i64" => "C.long",
            "c_ulong" | "u64" => "C.ulong",
            "c_longlong" => "C.longlong",
            "c_ulonglong" => "C.ulonglong",
            "c_float" | "f32" => "C.float",
            "c_double" | "f64" => "C.double",
            "c_char" | "i8" => "C.char",
            "c_uchar" | "u8" => "C.uchar",
            "c_bool" => "C.bool",
            "c_void" | "()" => "",
            "isize" => "C.long",
            "usize" => "C.ulong",
            name if name.ends_with("Handle") => "unsafe.Pointer",
            _ => "unsafe.Pointer",
        }
        .to_string()
    }

    fn generate_struct_type(&self, struct_def: &FfiStruct) -> String {
        let mut lines = Vec::new();

        // Struct documentation
        if !struct_def.documentation.is_empty() {
            for doc_line in &struct_def.documentation {
                lines.push(format!("// {}", doc_line));
            }
        }

        if struct_def.is_opaque {
            // Opaque struct - just a type alias to unsafe.Pointer
            lines.push(format!("type {} unsafe.Pointer", struct_def.name));
        } else {
            // Regular struct with fields
            lines.push(format!("type {} struct {{", struct_def.name));

            for field in &struct_def.fields {
                if !field.is_private {
                    // Field documentation
                    if !field.documentation.is_empty() {
                        for doc_line in &field.documentation {
                            lines.push(format!("\t// {}", doc_line));
                        }
                    }

                    let field_type = self.map_type(&field.type_info);
                    // Convert field name to PascalCase for Go export
                    let field_name = self.to_pascal_case(&field.name);
                    lines.push(format!("\t{} {}", field_name, field_type));
                }
            }

            lines.push("}".to_string());
        }

        lines.join("\n")
    }

    fn generate_enum_constants(&self, enum_def: &FfiEnum) -> String {
        let mut lines = Vec::new();

        // Enum documentation
        if !enum_def.documentation.is_empty() {
            for doc_line in &enum_def.documentation {
                lines.push(format!("// {}", doc_line));
            }
        }

        // Type definition
        let underlying_type = match enum_def.underlying_type {
            PrimitiveType::Int32 => "int32",
            PrimitiveType::UInt32 => "uint32",
            PrimitiveType::Int64 => "int64",
            PrimitiveType::UInt64 => "uint64",
            _ => "int32",
        };

        lines.push(format!("type {} {}", enum_def.name, underlying_type));
        lines.push("".to_string());

        // Constants with iota or explicit values
        if enum_def.is_flags {
            // For flags, use explicit bit values
            lines.push("const (".to_string());
            for variant in &enum_def.variants {
                if let Some(deprecation) = &variant.deprecation {
                    lines.push(format!("\t// DEPRECATED: {}", deprecation.message));
                }
                if !variant.documentation.is_empty() {
                    for doc_line in &variant.documentation {
                        lines.push(format!("\t// {}", doc_line));
                    }
                }
                lines.push(format!(
                    "\t{}{} {} = 1 << {}",
                    enum_def.name,
                    variant.name,
                    enum_def.name,
                    variant.value.trailing_zeros()
                ));
            }
            lines.push(")".to_string());
        } else {
            // For regular enums, check if we can use iota
            let can_use_iota =
                enum_def.variants.iter().enumerate().all(|(i, v)| v.value == i as i64);

            lines.push("const (".to_string());
            for (i, variant) in enum_def.variants.iter().enumerate() {
                if let Some(deprecation) = &variant.deprecation {
                    lines.push(format!("\t// DEPRECATED: {}", deprecation.message));
                }
                if !variant.documentation.is_empty() {
                    for doc_line in &variant.documentation {
                        lines.push(format!("\t// {}", doc_line));
                    }
                }

                if can_use_iota {
                    if i == 0 {
                        lines.push(format!(
                            "\t{}{} {} = iota",
                            enum_def.name, variant.name, enum_def.name
                        ));
                    } else {
                        lines.push(format!("\t{}{}", enum_def.name, variant.name));
                    }
                } else {
                    lines.push(format!(
                        "\t{}{} {} = {}",
                        enum_def.name, variant.name, enum_def.name, variant.value
                    ));
                }
            }
            lines.push(")".to_string());
        }

        lines.join("\n")
    }

    fn generate_cgo_declarations(&self, interface: &FfiInterface) -> String {
        let mut lines = Vec::new();

        // CGO directives
        lines.push(format!(
            "// #cgo LDFLAGS: -l{}",
            self.config.package_info.name
        ));
        lines.push(format!("// #include <{}.h>", self.config.package_info.name));

        lines.join("\n")
    }

    fn generate_function_wrapper(&self, func: &FfiFunction) -> String {
        let mut lines = Vec::new();

        // Function documentation
        if !func.documentation.is_empty() {
            for doc_line in &func.documentation {
                lines.push(format!("// {}", doc_line));
            }
        }

        // Deprecation comment
        if let Some(deprecation) = &func.deprecation {
            lines.push(format!("// Deprecated: {}", deprecation.message));
        }

        // Build Go function signature
        let func_name = self.to_pascal_case(&func.name);
        let mut go_params = Vec::new();
        let mut c_call_params = Vec::new();
        let mut conversions = Vec::new();
        let mut defer_statements = Vec::new();

        for (i, param) in func.parameters.iter().enumerate() {
            let go_type = self.map_type(&param.type_info);
            let param_name =
                if param.name.is_empty() { format!("param{}", i) } else { param.name.clone() };

            go_params.push(format!("{} {}", param_name, go_type));

            // Handle type conversions
            if param.type_info.is_string() {
                let c_param_name = format!("c{}", self.to_pascal_case(&param_name));
                conversions.push(format!("\t{} := C.CString({})", c_param_name, param_name));
                defer_statements.push(format!("\tdefer C.free(unsafe.Pointer({}))", c_param_name));
                c_call_params.push(c_param_name);
            } else if go_type.starts_with("*") || go_type == "unsafe.Pointer" {
                c_call_params.push(format!(
                    "(*C.{})({})",
                    self.map_c_type(&param.type_info.base_type()),
                    param_name
                ));
            } else {
                let c_type = self.map_c_type(&param.type_info.name);
                c_call_params.push(format!("{}({})", c_type, param_name));
            }
        }

        let params_str = go_params.join(", ");
        let return_type = if func.return_type.name == "c_void" || func.return_type.name == "()" {
            "error".to_string()
        } else if function_can_fail(&func.return_type) {
            format!("({}, error)", self.map_type(&func.return_type))
        } else {
            self.map_type(&func.return_type)
        };

        lines.push(format!(
            "func {}({}) {} {{",
            func_name, params_str, return_type
        ));

        // Add conversions
        for conversion in &conversions {
            lines.push(conversion.clone());
        }

        // Add defer statements
        for defer_stmt in &defer_statements {
            lines.push(defer_stmt.clone());
        }

        // Make C function call
        let c_call = format!("C.{}({})", func.c_name, c_call_params.join(", "));

        if function_can_fail(&func.return_type) {
            lines.push(format!("\tresult := {}", c_call));
            lines.push("\tif result != 0 {".to_string());
            lines.push(format!(
                "\t\treturn nil, &TrustformersError{{Code: int(result), Message: \"{} failed\"}}",
                func.name
            ));
            lines.push("\t}".to_string());

            if func.return_type.name != "c_void" && func.return_type.name != "()" {
                lines.push(format!(
                    "\treturn {}(result), nil",
                    self.map_type(&func.return_type)
                ));
            } else {
                lines.push("\treturn nil".to_string());
            }
        } else if func.return_type.name == "c_void" || func.return_type.name == "()" {
            lines.push(format!("\t{}", c_call));
            lines.push("\treturn nil".to_string());
        } else {
            let go_return_type = self.map_type(&func.return_type);
            if func.return_type.is_string() {
                lines.push(format!("\tcResult := {}", c_call));
                lines.push(format!("\tdefer C.free(unsafe.Pointer(cResult))"));
                lines.push(format!("\treturn C.GoString(cResult)"));
            } else {
                lines.push(format!("\treturn {}({})", go_return_type, c_call));
            }
        }

        lines.push("}".to_string());

        lines.join("\n")
    }

    fn to_pascal_case(&self, s: &str) -> String {
        s.split('_')
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                }
            })
            .collect()
    }
}

impl TypeMapper for GoGenerator {
    fn map_type(&self, ffi_type: &FfiType) -> String {
        // Check for custom mappings first
        if let Some(mapped) = self.type_mappings.get(&ffi_type.name) {
            return mapped.clone();
        }

        // Handle pointer types
        if ffi_type.is_pointer() {
            if ffi_type.is_string() {
                return "string".to_string();
            } else if ffi_type.base_type() == "c_void" {
                return "unsafe.Pointer".to_string();
            } else {
                return format!("*{}", self.map_base_type(&ffi_type.base_type()));
            }
        }

        // Handle array types
        if let Some(size) = ffi_type.array_size {
            let base_type = self.map_base_type(&ffi_type.base_type());
            return format!("[{}]{}", size, base_type);
        }

        // Handle regular types
        self.map_base_type(&ffi_type.name)
    }

    fn map_base_type(&self, type_name: &str) -> String {
        self.map_base_type(type_name)
    }

    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::Go
    }
}

impl LanguageGenerator for GoGenerator {
    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::Go
    }

    fn file_extension(&self) -> &'static str {
        "go"
    }

    fn generate(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        // Generate main Go file
        let mut main_content = Vec::new();

        // Package declaration
        main_content.push(format!("package {}", self.config.package_info.name));
        main_content.push("".to_string());

        // Package documentation
        main_content.push("/*".to_string());
        main_content.push(format!(
            "Package {} provides Go bindings for the TrustformeRS C API.",
            self.config.package_info.name
        ));
        main_content.push("".to_string());
        main_content.push(self.config.package_info.description.clone());
        main_content.push(format!("Version: {}", self.config.package_info.version));
        main_content.push("*/".to_string());
        main_content.push("".to_string());

        // CGO declarations
        main_content.push(self.generate_cgo_declarations(interface));
        main_content.push("import \"C\"".to_string());
        main_content.push("".to_string());

        // Imports
        main_content.push("import (".to_string());
        main_content.push("\t\"errors\"".to_string());
        main_content.push("\t\"unsafe\"".to_string());
        main_content.push(")".to_string());
        main_content.push("".to_string());

        // Error type
        main_content
            .push("// TrustformersError represents an error from the C library".to_string());
        main_content.push("type TrustformersError struct {".to_string());
        main_content.push("\tCode    int".to_string());
        main_content.push("\tMessage string".to_string());
        main_content.push("}".to_string());
        main_content.push("".to_string());

        main_content.push("// Error implements the error interface".to_string());
        main_content.push("func (e *TrustformersError) Error() string {".to_string());
        main_content.push("\treturn e.Message".to_string());
        main_content.push("}".to_string());
        main_content.push("".to_string());

        // Variable for common errors
        main_content.push("var (".to_string());
        main_content.push(
            "\t// ErrInvalidArgument is returned when an invalid argument is passed".to_string(),
        );
        main_content.push("\tErrInvalidArgument = errors.New(\"invalid argument\")".to_string());
        main_content
            .push("\t// ErrOutOfMemory is returned when memory allocation fails".to_string());
        main_content.push("\tErrOutOfMemory = errors.New(\"out of memory\")".to_string());
        main_content.push(")".to_string());
        main_content.push("".to_string());

        // Generate enums
        for enum_def in &interface.enums {
            main_content.push(self.generate_enum_constants(enum_def));
            main_content.push("".to_string());
        }

        // Generate structs
        for struct_def in &interface.structs {
            main_content.push(self.generate_struct_type(struct_def));
            main_content.push("".to_string());
        }

        // Generate function bindings
        main_content.push("// Function bindings".to_string());
        main_content.push("".to_string());
        for func in &interface.functions {
            main_content.push(self.generate_function_wrapper(func));
            main_content.push("".to_string());
        }

        // Write main module file
        let main_file = output_dir.join(format!("{}.go", self.config.package_info.name));
        fs::write(&main_file, main_content.join("\n"))?;

        // Generate package files
        self.generate_package_files(interface, output_dir, _templates)?;

        // Generate examples
        self.generate_examples(interface, output_dir, _templates)?;

        Ok(())
    }

    fn generate_package_files(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        // Generate go.mod
        let go_mod_content = format!(
            "module github.com/{}/{}

go 1.21

// Generated bindings for TrustformeRS C API
// Version: {}
",
            self.config.package_info.author.to_lowercase().replace(' ', "-"),
            self.config.package_info.name,
            self.config.package_info.version
        );

        fs::write(output_dir.join("go.mod"), go_mod_content)?;

        // Generate README.md
        let readme_content = format!(
            "# {} Go Bindings

{}

## Installation

```bash
go get github.com/{}/{}
```

## Usage

```go
import \"github.com/{}/{}\"

// Example usage
func main() {{
    // Your code here
}}
```

## License

{}

## Version

{}
",
            self.config.package_info.name,
            self.config.package_info.description,
            self.config.package_info.author.to_lowercase().replace(' ', "-"),
            self.config.package_info.name,
            self.config.package_info.author.to_lowercase().replace(' ', "-"),
            self.config.package_info.name,
            self.config.package_info.license,
            self.config.package_info.version
        );

        fs::write(output_dir.join("README.md"), readme_content)?;

        Ok(())
    }

    fn generate_examples(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        let examples_dir = output_dir.join("examples");
        fs::create_dir_all(&examples_dir)?;

        // Generate basic usage example
        let mut example_content = Vec::new();
        example_content.push("package main".to_string());
        example_content.push("".to_string());
        example_content.push("import (".to_string());
        example_content.push("\t\"fmt\"".to_string());
        example_content.push(format!(
            "\t\"github.com/{}/{}\"",
            self.config.package_info.author.to_lowercase().replace(' ', "-"),
            self.config.package_info.name
        ));
        example_content.push(")".to_string());
        example_content.push("".to_string());
        example_content.push("func main() {".to_string());
        example_content.push("\tfmt.Println(\"TrustformeRS Go Bindings Example\")".to_string());
        example_content.push("".to_string());

        // Add example function calls if available
        if let Some(first_func) = interface.functions.first() {
            if first_func.parameters.is_empty() {
                let func_name = self.to_pascal_case(&first_func.name);
                example_content.push(format!("\t// Call {}", func_name));
                example_content.push(format!(
                    "\tresult := {}.{}()",
                    self.config.package_info.name, func_name
                ));
                example_content.push("\tif err := result; err != nil {".to_string());
                example_content.push("\t\tfmt.Printf(\"Error: %v\\n\", err)".to_string());
                example_content.push("\t\treturn".to_string());
                example_content.push("\t}".to_string());
                example_content.push("\tfmt.Printf(\"Success!\\n\")".to_string());
            }
        }

        example_content.push("}".to_string());

        fs::write(
            examples_dir.join("basic_usage.go"),
            example_content.join("\n"),
        )?;

        Ok(())
    }

    fn generate_tests(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        // Generate test file
        let mut test_content = Vec::new();
        test_content.push(format!("package {}_test", self.config.package_info.name));
        test_content.push("".to_string());
        test_content.push("import (".to_string());
        test_content.push("\t\"testing\"".to_string());
        test_content.push(format!(
            "\t\"github.com/{}/{}\"",
            self.config.package_info.author.to_lowercase().replace(' ', "-"),
            self.config.package_info.name
        ));
        test_content.push(")".to_string());
        test_content.push("".to_string());

        test_content.push("func TestImport(t *testing.T) {".to_string());
        test_content.push("\t// If we get here, import worked".to_string());
        test_content.push("\tt.Log(\"Package imported successfully\")".to_string());
        test_content.push("}".to_string());
        test_content.push("".to_string());

        // Add tests for available functions
        for func in interface.functions.iter().take(3) {
            let func_name = self.to_pascal_case(&func.name);
            test_content.push(format!("func Test{}(t *testing.T) {{", func_name));
            test_content.push(format!("\t// Test {} function", func_name));
            test_content.push("\t// Add appropriate test here".to_string());
            test_content.push("\tt.Skip(\"Not implemented\")".to_string());
            test_content.push("}".to_string());
            test_content.push("".to_string());
        }

        fs::write(
            output_dir.join(format!("{}_test.go", self.config.package_info.name)),
            test_content.join("\n"),
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_go_generator_creation() {
        let config = CodeGenConfig::default();
        let generator = GoGenerator::new(&config);
        assert!(generator.is_ok());
    }

    #[test]
    fn test_type_mapping() {
        let config = CodeGenConfig::default();
        let generator = GoGenerator::new(&config).unwrap();

        assert_eq!(generator.map_base_type("c_int"), "int32");
        assert_eq!(generator.map_base_type("c_uint"), "uint32");
        assert_eq!(generator.map_base_type("c_float"), "float32");
        assert_eq!(generator.map_base_type("c_double"), "float64");
        assert_eq!(generator.map_base_type("c_bool"), "bool");
    }

    #[test]
    fn test_pascal_case_conversion() {
        let config = CodeGenConfig::default();
        let generator = GoGenerator::new(&config).unwrap();

        assert_eq!(generator.to_pascal_case("load_model"), "LoadModel");
        assert_eq!(generator.to_pascal_case("get_config"), "GetConfig");
        assert_eq!(generator.to_pascal_case("create_tensor"), "CreateTensor");
    }
}
