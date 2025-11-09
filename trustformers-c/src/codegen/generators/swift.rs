//! Swift bindings generator for FFI interfaces
//!
//! Generates Swift bindings with C interop for the TrustformeRS C API.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::codegen::ast::{FfiEnum, FfiFunction, FfiInterface, FfiStruct, FfiType};
use crate::codegen::templates::TemplateEngine;
use crate::codegen::{CodeGenConfig, TargetLanguage};
use crate::error::TrustformersResult;

use super::common::{function_can_fail, TypeMapper};
use super::LanguageGenerator;

/// Swift bindings generator
pub struct SwiftGenerator {
    config: CodeGenConfig,
    type_mappings: HashMap<String, String>,
}

impl SwiftGenerator {
    pub fn new(config: &CodeGenConfig) -> TrustformersResult<Self> {
        let mut type_mappings = HashMap::new();

        // Basic type mappings for Swift C interop
        type_mappings.insert("c_int".to_string(), "Int32".to_string());
        type_mappings.insert("c_uint".to_string(), "UInt32".to_string());
        type_mappings.insert("c_short".to_string(), "Int16".to_string());
        type_mappings.insert("c_ushort".to_string(), "UInt16".to_string());
        type_mappings.insert("c_long".to_string(), "Int".to_string());
        type_mappings.insert("c_ulong".to_string(), "UInt".to_string());
        type_mappings.insert("c_longlong".to_string(), "Int64".to_string());
        type_mappings.insert("c_ulonglong".to_string(), "UInt64".to_string());
        type_mappings.insert("c_float".to_string(), "Float".to_string());
        type_mappings.insert("c_double".to_string(), "Double".to_string());
        type_mappings.insert("c_char".to_string(), "Int8".to_string());
        type_mappings.insert("c_uchar".to_string(), "UInt8".to_string());
        type_mappings.insert("c_bool".to_string(), "Bool".to_string());
        type_mappings.insert("c_void".to_string(), "Void".to_string());
        type_mappings.insert("*const c_char".to_string(), "String".to_string());
        type_mappings.insert(
            "*mut c_char".to_string(),
            "UnsafeMutablePointer<Int8>".to_string(),
        );
        type_mappings.insert("*const c_void".to_string(), "UnsafeRawPointer".to_string());
        type_mappings.insert(
            "*mut c_void".to_string(),
            "UnsafeMutableRawPointer".to_string(),
        );

        // Rust primitive type mappings
        type_mappings.insert("i8".to_string(), "Int8".to_string());
        type_mappings.insert("i16".to_string(), "Int16".to_string());
        type_mappings.insert("i32".to_string(), "Int32".to_string());
        type_mappings.insert("i64".to_string(), "Int64".to_string());
        type_mappings.insert("u8".to_string(), "UInt8".to_string());
        type_mappings.insert("u16".to_string(), "UInt16".to_string());
        type_mappings.insert("u32".to_string(), "UInt32".to_string());
        type_mappings.insert("u64".to_string(), "UInt64".to_string());
        type_mappings.insert("f32".to_string(), "Float".to_string());
        type_mappings.insert("f64".to_string(), "Double".to_string());
        type_mappings.insert("usize".to_string(), "UInt".to_string());
        type_mappings.insert("isize".to_string(), "Int".to_string());
        type_mappings.insert("()".to_string(), "Void".to_string());

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
            "c_int" | "i32" => "Int32",
            "c_uint" | "u32" => "UInt32",
            "c_short" | "i16" => "Int16",
            "c_ushort" | "u16" => "UInt16",
            "c_long" | "i64" => "Int64",
            "c_ulong" | "u64" => "UInt64",
            "c_longlong" => "Int64",
            "c_ulonglong" => "UInt64",
            "c_float" | "f32" => "Float",
            "c_double" | "f64" => "Double",
            "c_char" | "i8" => "Int8",
            "c_uchar" | "u8" => "UInt8",
            "c_bool" => "Bool",
            "c_void" | "()" => "Void",
            "isize" => "Int",
            "usize" => "UInt",
            name if name.ends_with("Handle") => "OpaquePointer",
            _ => "OpaquePointer", // Default for unknown types
        }
        .to_string()
    }

    fn generate_struct_definition(&self, struct_def: &FfiStruct) -> String {
        let mut lines = Vec::new();

        // Struct documentation
        if !struct_def.documentation.is_empty() {
            for doc_line in &struct_def.documentation {
                lines.push(format!("/// {}", doc_line));
            }
        }

        if struct_def.is_opaque {
            // Opaque struct - just a typealias to OpaquePointer
            lines.push(format!(
                "public typealias {} = OpaquePointer",
                struct_def.name
            ));
        } else {
            // Regular struct with fields
            lines.push(format!("public struct {} {{", struct_def.name));

            // Field definitions
            for field in &struct_def.fields {
                if !field.is_private {
                    if !field.documentation.is_empty() {
                        for doc_line in &field.documentation {
                            lines.push(format!("    /// {}", doc_line));
                        }
                    }

                    let field_type = self.map_type(&field.type_info);
                    let field_name = Self::to_camel_case(&field.name);
                    lines.push(format!("    public var {}: {}", field_name, field_type));
                }
            }

            // Default initializer
            lines.push("".to_string());
            lines.push("    public init(".to_string());

            let visible_fields: Vec<_> =
                struct_def.fields.iter().filter(|f| !f.is_private).collect();

            for (i, field) in visible_fields.iter().enumerate() {
                let field_type = self.map_type(&field.type_info);
                let field_name = Self::to_camel_case(&field.name);
                let comma = if i < visible_fields.len() - 1 { "," } else { "" };
                lines.push(format!("        {}: {}{}", field_name, field_type, comma));
            }

            lines.push("    ) {".to_string());

            for field in &visible_fields {
                let field_name = Self::to_camel_case(&field.name);
                lines.push(format!("        self.{0} = {0}", field_name));
            }

            lines.push("    }".to_string());

            lines.push("}".to_string());
        }

        lines.join("\n")
    }

    fn generate_enum_definition(&self, enum_def: &FfiEnum) -> String {
        let mut lines = Vec::new();

        // Enum documentation
        if !enum_def.documentation.is_empty() {
            for doc_line in &enum_def.documentation {
                lines.push(format!("/// {}", doc_line));
            }
        }

        if enum_def.is_flags {
            // For flags/bitfield enums, use OptionSet
            lines.push(format!("public struct {}: OptionSet {{", enum_def.name));
            lines.push("    public let rawValue: Int".to_string());
            lines.push("".to_string());
            lines.push("    public init(rawValue: Int) {".to_string());
            lines.push("        self.rawValue = rawValue".to_string());
            lines.push("    }".to_string());
            lines.push("".to_string());

            // Generate static constants for each flag
            for variant in &enum_def.variants {
                if !variant.documentation.is_empty() {
                    for doc_line in &variant.documentation {
                        lines.push(format!("    /// {}", doc_line));
                    }
                }

                if let Some(deprecation) = &variant.deprecation {
                    lines.push(format!(
                        "    @available(*, deprecated, message: \"{}\")",
                        deprecation.message
                    ));
                }

                let variant_name = Self::to_camel_case(&variant.name);
                lines.push(format!(
                    "    public static let {} = {}(rawValue: {})",
                    variant_name, enum_def.name, variant.value
                ));
            }

            lines.push("}".to_string());
        } else {
            // Regular enum with raw values
            lines.push(format!("public enum {}: Int {{", enum_def.name));

            for variant in &enum_def.variants {
                if !variant.documentation.is_empty() {
                    for doc_line in &variant.documentation {
                        lines.push(format!("    /// {}", doc_line));
                    }
                }

                if let Some(deprecation) = &variant.deprecation {
                    lines.push(format!(
                        "    @available(*, deprecated, message: \"{}\")",
                        deprecation.message
                    ));
                }

                let variant_name = Self::to_camel_case(&variant.name);
                lines.push(format!("    case {} = {}", variant_name, variant.value));
            }

            lines.push("}".to_string());
        }

        lines.join("\n")
    }

    fn generate_function_declarations(&self, func: &FfiFunction) -> String {
        let mut lines = Vec::new();

        // Function documentation
        if !func.documentation.is_empty() {
            for doc_line in &func.documentation {
                lines.push(format!("    /// {}", doc_line));
            }

            // Add parameter documentation
            for param in &func.parameters {
                if !param.documentation.is_empty() {
                    lines.push(format!(
                        "    /// - Parameter {}: {}",
                        Self::to_camel_case(&param.name),
                        param.documentation.join(" ")
                    ));
                }
            }

            // Add return documentation
            if func.return_type.name != "c_void" && func.return_type.name != "()" {
                lines.push("    /// - Returns: Return value from C function".to_string());
            }
        }

        // Add deprecation attribute if needed
        if let Some(deprecation) = &func.deprecation {
            lines.push(format!(
                "    @available(*, deprecated, message: \"{}\")",
                deprecation.message
            ));
        }

        // Build parameter list
        let params: Vec<String> = func
            .parameters
            .iter()
            .map(|p| {
                let param_type = self.map_type(&p.type_info);
                let param_name = Self::to_camel_case(&p.name);
                format!("{}: {}", param_name, param_type)
            })
            .collect();

        let return_type = self.map_type(&func.return_type);
        let func_name = Self::to_camel_case(&func.name);

        // Generate wrapper function
        let return_annotation = if return_type == "Void" {
            String::new()
        } else {
            format!(" -> {}", return_type)
        };

        lines.push(format!(
            "    public static func {}({}){} {{",
            func_name,
            params.join(", "),
            return_annotation
        ));

        // Convert Swift strings to C strings if needed
        let mut conversions = Vec::new();
        let mut param_names_for_call = Vec::new();

        for param in &func.parameters {
            let param_name = Self::to_camel_case(&param.name);
            if param.type_info.is_string() {
                let c_string_var = format!(
                    "c{}",
                    param_name.chars().next().unwrap().to_uppercase().collect::<String>()
                        + &param_name[1..]
                );
                conversions.push(format!(
                    "        let {} = {}.utf8CString",
                    c_string_var, param_name
                ));
                param_names_for_call.push(format!(
                    "{}.withUnsafeBufferPointer {{ $0.baseAddress }}",
                    c_string_var
                ));
            } else {
                param_names_for_call.push(param_name);
            }
        }

        // Add string conversions
        for conversion in conversions {
            lines.push(conversion);
        }

        // Call C function
        if function_can_fail(&func.return_type) {
            lines.push(format!(
                "        let result = {}({})",
                func.c_name,
                param_names_for_call.join(", ")
            ));
            lines.push("        if result != 0 {".to_string());
            lines.push(format!(
                "            fatalError(\"Function {} failed with error code \\(result)\")",
                func.name
            ));
            lines.push("        }".to_string());
            if return_type != "Void" {
                lines.push("        return result".to_string());
            }
        } else if return_type == "Void" {
            lines.push(format!(
                "        {}({})",
                func.c_name,
                param_names_for_call.join(", ")
            ));
        } else {
            // Handle string return types
            if func.return_type.is_string() {
                lines.push(format!(
                    "        guard let cString = {}({}) else {{",
                    func.c_name,
                    param_names_for_call.join(", ")
                ));
                lines.push("            return \"\"".to_string());
                lines.push("        }".to_string());
                lines.push("        return String(cString: cString)".to_string());
            } else {
                lines.push(format!(
                    "        return {}({})",
                    func.c_name,
                    param_names_for_call.join(", ")
                ));
            }
        }

        lines.push("    }".to_string());

        lines.join("\n")
    }

    fn to_pascal_case(s: &str) -> String {
        s.split('_')
            .filter(|part| !part.is_empty())
            .map(|part| {
                let mut chars = part.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => {
                        first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
                    },
                }
            })
            .collect()
    }

    fn to_camel_case(s: &str) -> String {
        let pascal = Self::to_pascal_case(s);
        if pascal.is_empty() {
            return pascal;
        }
        let mut chars = pascal.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_lowercase().collect::<String>() + chars.as_str(),
        }
    }
}

impl TypeMapper for SwiftGenerator {
    fn map_type(&self, ffi_type: &FfiType) -> String {
        // Check for custom mappings first
        if let Some(mapped) = self.type_mappings.get(&ffi_type.name) {
            return mapped.clone();
        }

        // Handle pointer types
        if ffi_type.is_pointer() {
            if ffi_type.is_string() {
                return "String".to_string();
            } else if ffi_type.base_type() == "c_void" {
                if ffi_type.is_const {
                    return "UnsafeRawPointer".to_string();
                } else {
                    return "UnsafeMutableRawPointer".to_string();
                }
            } else if ffi_type.is_handle() {
                return "OpaquePointer".to_string();
            } else {
                // For typed pointers
                let base_type = self.map_base_type(&ffi_type.base_type());
                if ffi_type.is_const {
                    return format!("UnsafePointer<{}>", base_type);
                } else {
                    return format!("UnsafeMutablePointer<{}>", base_type);
                }
            }
        }

        // Handle array types
        if let Some(size) = ffi_type.array_size {
            let base_type = self.map_base_type(&ffi_type.base_type());
            return format!("[{}]", base_type);
        }

        // Handle regular types
        self.map_base_type(&ffi_type.name)
    }

    fn map_base_type(&self, type_name: &str) -> String {
        self.map_base_type(type_name)
    }

    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::Swift
    }
}

impl LanguageGenerator for SwiftGenerator {
    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::Swift
    }

    fn file_extension(&self) -> &'static str {
        "swift"
    }

    fn generate(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        // Generate main Swift file
        let mut main_content = Vec::new();

        // File header
        main_content.push("//".to_string());
        main_content.push(format!(
            "// Swift bindings for {}",
            &interface.metadata.library_name
        ));
        main_content.push("//".to_string());
        main_content.push("// TrustformeRS C API bindings".to_string());
        main_content.push(format!("// Version: {}", &interface.metadata.version));
        main_content.push("//".to_string());
        main_content.push("// This file is auto-generated. Do not modify manually.".to_string());
        main_content.push("//".to_string());
        main_content.push("".to_string());

        // Imports
        main_content.push("import Foundation".to_string());
        main_content.push("".to_string());

        // TrustformersError struct
        main_content.push("/// TrustformeRS error type".to_string());
        main_content
            .push("public struct TrustformersError: Error, CustomStringConvertible {".to_string());
        main_content.push("    public let message: String".to_string());
        main_content.push("    public let code: Int?".to_string());
        main_content.push("".to_string());
        main_content.push("    public init(message: String, code: Int? = nil) {".to_string());
        main_content.push("        self.message = message".to_string());
        main_content.push("        self.code = code".to_string());
        main_content.push("    }".to_string());
        main_content.push("".to_string());
        main_content.push("    public var description: String {".to_string());
        main_content.push("        if let code = code {".to_string());
        main_content
            .push("            return \"TrustformersError(\\(code)): \\(message)\"".to_string());
        main_content.push("        } else {".to_string());
        main_content.push("            return \"TrustformersError: \\(message)\"".to_string());
        main_content.push("        }".to_string());
        main_content.push("    }".to_string());
        main_content.push("}".to_string());
        main_content.push("".to_string());

        // Generate enums
        for enum_def in &interface.enums {
            main_content.push(self.generate_enum_definition(enum_def));
            main_content.push("".to_string());
        }

        // Generate structs
        for struct_def in &interface.structs {
            main_content.push(self.generate_struct_definition(struct_def));
            main_content.push("".to_string());
        }

        // Generate main Trustformers class
        main_content.push("/// Main TrustformeRS Swift wrapper class".to_string());
        main_content.push("public class Trustformers {".to_string());
        main_content.push("".to_string());

        // Library loading
        main_content.push("    /// Shared library handle".to_string());
        main_content
            .push("    private static var libraryHandle: UnsafeMutableRawPointer?".to_string());
        main_content.push("".to_string());

        main_content.push("    /// Load the TrustformeRS native library".to_string());
        main_content.push("    public static func loadLibrary() throws {".to_string());
        main_content.push("        #if os(macOS)".to_string());
        main_content.push("        let libName = \"libtrusformers_c.dylib\"".to_string());
        main_content.push("        #elseif os(Linux)".to_string());
        main_content.push("        let libName = \"libtrusformers_c.so\"".to_string());
        main_content.push("        #elseif os(Windows)".to_string());
        main_content.push("        let libName = \"trustformers_c.dll\"".to_string());
        main_content.push("        #else".to_string());
        main_content.push("        let libName = \"libtrusformers_c.so\"".to_string());
        main_content.push("        #endif".to_string());
        main_content.push("".to_string());
        main_content
            .push("        guard let handle = dlopen(libName, RTLD_NOW) else {".to_string());
        main_content.push("            if let error = dlerror() {".to_string());
        main_content.push("                throw TrustformersError(".to_string());
        main_content.push(
            "                    message: \"Failed to load library: \\(String(cString: error))\""
                .to_string(),
        );
        main_content.push("                )".to_string());
        main_content.push("            }".to_string());
        main_content.push(
            "            throw TrustformersError(message: \"Failed to load library: \\(libName)\")"
                .to_string(),
        );
        main_content.push("        }".to_string());
        main_content.push("".to_string());
        main_content.push("        libraryHandle = handle".to_string());
        main_content.push("    }".to_string());
        main_content.push("".to_string());

        main_content.push("    /// Unload the TrustformeRS native library".to_string());
        main_content.push("    public static func unloadLibrary() {".to_string());
        main_content.push("        if let handle = libraryHandle {".to_string());
        main_content.push("            dlclose(handle)".to_string());
        main_content.push("            libraryHandle = nil".to_string());
        main_content.push("        }".to_string());
        main_content.push("    }".to_string());
        main_content.push("".to_string());

        // Generate C function declarations
        for func in &interface.functions {
            // Generate C function import
            let params: Vec<String> = func
                .parameters
                .iter()
                .map(|p| {
                    let param_type = self.map_type(&p.type_info);
                    format!("_ {}: {}", Self::to_camel_case(&p.name), param_type)
                })
                .collect();

            let return_type = self.map_type(&func.return_type);
            let return_annotation = if return_type == "Void" {
                String::new()
            } else {
                format!(" -> {}", return_type)
            };

            main_content.push(format!(
                "    private static let {}: @convention(c) ({}){} = unsafeBitCast(dlsym(libraryHandle, \"{}\"), to: (@convention(c) ({}){}).self)",
                func.c_name,
                params.join(", "),
                return_annotation,
                func.c_name,
                params.join(", "),
                return_annotation
            ));
        }

        main_content.push("".to_string());

        // Generate function bindings
        for func in &interface.functions {
            main_content.push(self.generate_function_declarations(func));
            main_content.push("".to_string());
        }

        // Close class
        main_content.push("}".to_string());

        // Write main file
        let main_file = output_dir.join(format!("{}.swift", self.config.package_info.name));
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
        // Generate Package.swift
        let package_content = format!(
            r#"// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "{}",
    platforms: [
        .macOS(.v10_15),
        .iOS(.v13),
        .tvOS(.v13),
        .watchOS(.v6)
    ],
    products: [
        .library(
            name: "{}",
            targets: ["{}"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "{}",
            dependencies: []),
        .testTarget(
            name: "{}Tests",
            dependencies: ["{}"]),
    ]
)
"#,
            self.config.package_info.name,
            self.config.package_info.name,
            self.config.package_info.name,
            self.config.package_info.name,
            self.config.package_info.name,
            self.config.package_info.name
        );

        fs::write(output_dir.join("Package.swift"), package_content)?;

        // Generate README
        let readme_content = format!(
            "# {} - Swift Bindings

{}

## Installation

### Swift Package Manager

Add the following to your `Package.swift`:

```swift
dependencies: [
    .package(url: \"https://github.com/yourusername/{}.git\", from: \"{}\")
]
```

## Usage

```swift
import {}

// Load the library
try Trustformers.loadLibrary()

// Use the bindings
// ...

// Cleanup
Trustformers.unloadLibrary()
```

## Version

Version: {}

## License

{}
",
            self.config.package_info.name,
            self.config.package_info.description,
            self.config.package_info.name,
            self.config.package_info.version,
            self.config.package_info.name,
            self.config.package_info.version,
            self.config.package_info.license
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
        let examples_dir = output_dir.join("Examples");
        fs::create_dir_all(&examples_dir)?;

        // Generate basic usage example
        let mut example_content = Vec::new();
        example_content.push("import Foundation".to_string());
        example_content.push(format!("import {}", self.config.package_info.name));
        example_content.push("".to_string());
        example_content.push("/// Basic usage example for TrustformeRS Swift bindings".to_string());
        example_content.push("func main() {".to_string());
        example_content.push("    do {".to_string());
        example_content.push("        // Load the library".to_string());
        example_content.push("        try Trustformers.loadLibrary()".to_string());
        example_content.push("".to_string());
        example_content.push("        print(\"TrustformeRS Swift Bindings Example\")".to_string());
        example_content.push("".to_string());

        // Add example function calls if available
        if let Some(first_func) = interface.functions.first() {
            if first_func.parameters.is_empty() {
                let func_name = Self::to_camel_case(&first_func.name);
                example_content.push(format!("        // Call {}", first_func.name));
                example_content.push(format!("        let result = Trustformers.{}()", func_name));
                example_content.push("        print(\"Result: \\(result)\")".to_string());
            }
        }

        example_content.push("".to_string());
        example_content.push("        // Cleanup".to_string());
        example_content.push("        Trustformers.unloadLibrary()".to_string());
        example_content.push("    } catch {".to_string());
        example_content.push("        print(\"Error: \\(error)\")".to_string());
        example_content.push("    }".to_string());
        example_content.push("}".to_string());
        example_content.push("".to_string());
        example_content.push("main()".to_string());

        fs::write(
            examples_dir.join("BasicUsage.swift"),
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
        let tests_dir =
            output_dir.join("Tests").join(format!("{}Tests", self.config.package_info.name));
        fs::create_dir_all(&tests_dir)?;

        // Generate basic test file
        let mut test_content = Vec::new();
        test_content.push("import XCTest".to_string());
        test_content.push(format!(
            "@testable import {}",
            self.config.package_info.name
        ));
        test_content.push("".to_string());
        test_content.push(format!(
            "final class {}Tests: XCTestCase {{",
            self.config.package_info.name
        ));
        test_content.push("".to_string());
        test_content.push("    override func setUpWithError() throws {".to_string());
        test_content.push("        try Trustformers.loadLibrary()".to_string());
        test_content.push("    }".to_string());
        test_content.push("".to_string());
        test_content.push("    override func tearDownWithError() throws {".to_string());
        test_content.push("        Trustformers.unloadLibrary()".to_string());
        test_content.push("    }".to_string());
        test_content.push("".to_string());

        test_content.push("    func testLibraryLoading() throws {".to_string());
        test_content.push("        // If we get here, library loaded successfully".to_string());
        test_content.push("        XCTAssertTrue(true)".to_string());
        test_content.push("    }".to_string());

        // Add tests for available functions
        for func in interface.functions.iter().take(3) {
            let test_name = format!("test{}", Self::to_pascal_case(&func.name));
            test_content.push("".to_string());
            test_content.push(format!("    func {}() throws {{", test_name));
            test_content.push(format!("        // Test {} function", func.name));
            test_content.push("        // Add appropriate test here".to_string());
            test_content.push("    }".to_string());
        }

        test_content.push("}".to_string());

        fs::write(
            tests_dir.join(format!("{}Tests.swift", self.config.package_info.name)),
            test_content.join("\n"),
        )?;

        Ok(())
    }
}
