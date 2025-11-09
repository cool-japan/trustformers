//! Python bindings generator for FFI interfaces

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::codegen::ast::{FfiEnum, FfiFunction, FfiInterface, FfiStruct, FfiType};
use crate::codegen::templates::TemplateEngine;
use crate::codegen::{CodeGenConfig, TargetLanguage};
use crate::error::TrustformersResult;

use super::common::{function_can_fail, TypeMapper};
use super::LanguageGenerator;

/// Python bindings generator
pub struct PythonGenerator {
    config: CodeGenConfig,
    type_mappings: HashMap<String, String>,
}

impl PythonGenerator {
    pub fn new(config: &CodeGenConfig) -> TrustformersResult<Self> {
        let mut type_mappings = HashMap::new();

        // Basic type mappings
        type_mappings.insert("c_int".to_string(), "ctypes.c_int".to_string());
        type_mappings.insert("c_uint".to_string(), "ctypes.c_uint".to_string());
        type_mappings.insert("c_long".to_string(), "ctypes.c_long".to_string());
        type_mappings.insert("c_ulong".to_string(), "ctypes.c_ulong".to_string());
        type_mappings.insert("c_longlong".to_string(), "ctypes.c_longlong".to_string());
        type_mappings.insert("c_ulonglong".to_string(), "ctypes.c_ulonglong".to_string());
        type_mappings.insert("c_float".to_string(), "ctypes.c_float".to_string());
        type_mappings.insert("c_double".to_string(), "ctypes.c_double".to_string());
        type_mappings.insert("c_char".to_string(), "ctypes.c_char".to_string());
        type_mappings.insert("c_bool".to_string(), "ctypes.c_bool".to_string());
        type_mappings.insert("c_void".to_string(), "None".to_string());
        type_mappings.insert("*const c_char".to_string(), "ctypes.c_char_p".to_string());
        type_mappings.insert("*mut c_char".to_string(), "ctypes.c_char_p".to_string());
        type_mappings.insert("*const c_void".to_string(), "ctypes.c_void_p".to_string());
        type_mappings.insert("*mut c_void".to_string(), "ctypes.c_void_p".to_string());

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
            "c_int" | "i32" => "ctypes.c_int",
            "c_uint" | "u32" => "ctypes.c_uint",
            "c_short" | "i16" => "ctypes.c_short",
            "c_ushort" | "u16" => "ctypes.c_ushort",
            "c_long" | "i64" => "ctypes.c_long",
            "c_ulong" | "u64" => "ctypes.c_ulong",
            "c_longlong" => "ctypes.c_longlong",
            "c_ulonglong" => "ctypes.c_ulonglong",
            "c_float" | "f32" => "ctypes.c_float",
            "c_double" | "f64" => "ctypes.c_double",
            "c_char" | "i8" => "ctypes.c_char",
            "c_uchar" | "u8" => "ctypes.c_ubyte",
            "c_bool" => "ctypes.c_bool",
            "c_void" | "()" => "None",
            "isize" => "ctypes.c_ssize_t",
            "usize" => "ctypes.c_size_t",
            name if name.ends_with("Handle") => "ctypes.c_void_p",
            _ => "ctypes.c_void_p", // Default for unknown types
        }
        .to_string()
    }

    fn generate_function_binding(&self, func: &FfiFunction) -> String {
        let mut lines = Vec::new();

        // Function documentation
        if !func.documentation.is_empty() {
            lines.push("".to_string());
            for doc_line in &func.documentation {
                lines.push(format!("# {}", doc_line));
            }
        }

        // Function signature
        let param_types: Vec<String> =
            func.parameters.iter().map(|p| self.map_type(&p.type_info)).collect();

        let return_type = self.map_type(&func.return_type);

        lines.push(format!(
            "_lib.{}.argtypes = [{}]",
            func.c_name,
            param_types.join(", ")
        ));
        lines.push(format!("_lib.{}.restype = {}", func.c_name, return_type));

        // Python wrapper function
        let param_names: Vec<String> = func.parameters.iter().map(|p| p.name.clone()).collect();

        lines.push("".to_string());
        lines.push(format!("def {}({}):", func.name, param_names.join(", ")));

        // Add docstring
        if !func.documentation.is_empty() {
            lines.push("    \"\"\"".to_string());
            for doc_line in &func.documentation {
                lines.push(format!("    {}", doc_line));
            }

            // Add parameter documentation
            if !func.parameters.is_empty() {
                lines.push("".to_string());
                lines.push("    Parameters:".to_string());
                for param in &func.parameters {
                    let param_type = param.type_info.map_to_language(&TargetLanguage::Python);
                    lines.push(format!(
                        "        {} ({}): Parameter description",
                        param.name, param_type
                    ));
                }
            }

            // Add return documentation
            if func.return_type.name != "void" {
                lines.push("".to_string());
                lines.push("    Returns:".to_string());
                let return_type = func.return_type.map_to_language(&TargetLanguage::Python);
                lines.push(format!("        {}: Return value description", return_type));
            }

            lines.push("    \"\"\"".to_string());
        }

        // Function implementation
        if function_can_fail(&func.return_type) {
            lines.push(format!(
                "    result = _lib.{}({})",
                func.c_name,
                param_names.join(", ")
            ));
            lines.push("    if result != 0:".to_string());
            lines.push(
                "        raise TrustformersError(f\"Function {} failed with error code {result}\")"
                    .to_string(),
            );
            lines.push("    return result".to_string());
        } else {
            lines.push(format!(
                "    return _lib.{}({})",
                func.c_name,
                param_names.join(", ")
            ));
        }

        lines.join("\n")
    }

    fn generate_struct_binding(&self, struct_def: &FfiStruct) -> String {
        let mut lines = Vec::new();

        // Struct documentation
        if !struct_def.documentation.is_empty() {
            lines.push("".to_string());
            for doc_line in &struct_def.documentation {
                lines.push(format!("# {}", doc_line));
            }
        }

        if struct_def.is_opaque {
            // Opaque struct - just a handle
            lines.push(format!("class {}(ctypes.Structure):", struct_def.name));
            lines.push(
                "    \"\"\"Opaque structure - internal implementation hidden\"\"\"".to_string(),
            );
            lines.push("    pass".to_string());
        } else {
            // Regular struct with fields
            lines.push(format!("class {}(ctypes.Structure):", struct_def.name));

            if !struct_def.documentation.is_empty() {
                lines.push("    \"\"\"".to_string());
                for doc_line in &struct_def.documentation {
                    lines.push(format!("    {}", doc_line));
                }
                lines.push("    \"\"\"".to_string());
            }

            // Field definitions
            lines.push("    _fields_ = [".to_string());
            for field in &struct_def.fields {
                if !field.is_private {
                    let field_type = self.map_type(&field.type_info);
                    lines.push(format!("        ('{}', {}),", field.name, field_type));
                }
            }
            lines.push("    ]".to_string());

            // Property accessors for better Python integration
            for field in &struct_def.fields {
                if !field.is_private {
                    lines.push("".to_string());
                    lines.push("    @property".to_string());
                    lines.push(format!("    def {}(self):", field.name));
                    lines.push(format!("        return self.{}", field.name));

                    lines.push("".to_string());
                    lines.push(format!("    @{}.setter", field.name));
                    lines.push(format!("    def {}(self, value):", field.name));
                    lines.push(format!("        self.{} = value", field.name));
                }
            }
        }

        lines.join("\n")
    }

    fn generate_enum_binding(&self, enum_def: &FfiEnum) -> String {
        let mut lines = Vec::new();

        // Enum documentation
        if !enum_def.documentation.is_empty() {
            lines.push("".to_string());
            for doc_line in &enum_def.documentation {
                lines.push(format!("# {}", doc_line));
            }
        }

        lines.push(format!("class {}(IntEnum):", enum_def.name));

        if !enum_def.documentation.is_empty() {
            lines.push("    \"\"\"".to_string());
            for doc_line in &enum_def.documentation {
                lines.push(format!("    {}", doc_line));
            }
            lines.push("    \"\"\"".to_string());
        }

        // Enum variants
        for variant in &enum_def.variants {
            if let Some(deprecation) = &variant.deprecation {
                lines.push(format!("    # DEPRECATED: {}", deprecation.message));
            }
            lines.push(format!("    {} = {}", variant.name, variant.value));
        }

        lines.join("\n")
    }
}

impl TypeMapper for PythonGenerator {
    fn map_type(&self, ffi_type: &FfiType) -> String {
        // Check for custom mappings first
        if let Some(mapped) = self.type_mappings.get(&ffi_type.name) {
            return mapped.clone();
        }

        // Handle pointer types
        if ffi_type.is_pointer() {
            if ffi_type.is_string() {
                return "ctypes.c_char_p".to_string();
            } else if ffi_type.base_type() == "c_void" {
                return "ctypes.c_void_p".to_string();
            } else {
                return format!(
                    "ctypes.POINTER({})",
                    self.map_base_type(&ffi_type.base_type())
                );
            }
        }

        // Handle array types
        if let Some(size) = ffi_type.array_size {
            let base_type = self.map_base_type(&ffi_type.base_type());
            return format!("({} * {})", base_type, size);
        }

        // Handle regular types
        self.map_base_type(&ffi_type.name)
    }

    fn map_base_type(&self, type_name: &str) -> String {
        self.map_base_type(type_name)
    }

    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::Python
    }
}

impl LanguageGenerator for PythonGenerator {
    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::Python
    }

    fn file_extension(&self) -> &'static str {
        "py"
    }

    fn generate(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        // Generate main module file
        let mut main_content = Vec::new();

        // Module header
        main_content.push("\"\"\"".to_string());
        main_content.push(format!(
            "Python bindings for {}",
            &interface.metadata.library_name
        ));
        main_content.push("".to_string());
        main_content.push("TrustformeRS C API bindings".to_string());
        main_content.push(format!("Version: {}", &interface.metadata.version));
        main_content.push("\"\"\"".to_string());
        main_content.push("".to_string());

        // Imports
        main_content.push("import ctypes".to_string());
        main_content.push("import os".to_string());
        main_content.push("import sys".to_string());
        main_content.push("import platform".to_string());
        main_content.push("from enum import IntEnum".to_string());
        main_content.push("from typing import Optional, List, Union".to_string());
        main_content.push("".to_string());

        // Error handling
        main_content.push("class TrustformersError(Exception):".to_string());
        main_content.push("    \"\"\"Base exception for TrustformeRS errors\"\"\"".to_string());
        main_content.push("    pass".to_string());
        main_content.push("".to_string());

        // Library loading
        main_content.push("def _load_library():".to_string());
        main_content.push("    \"\"\"Load the native library\"\"\"".to_string());
        main_content.push("    system = platform.system()".to_string());
        main_content.push("    if system == 'Windows':".to_string());
        main_content.push("        lib_name = 'trustformers_c.dll'".to_string());
        main_content.push("    elif system == 'Darwin':".to_string());
        main_content.push("        lib_name = 'libtrusformers_c.dylib'".to_string());
        main_content.push("    else:".to_string());
        main_content.push("        lib_name = 'libtrusformers_c.so'".to_string());
        main_content.push("    ".to_string());
        main_content.push("    # Try to find the library".to_string());
        main_content
            .push("    lib_path = os.path.join(os.path.dirname(__file__), lib_name)".to_string());
        main_content.push("    if not os.path.exists(lib_path):".to_string());
        main_content.push("        lib_path = lib_name  # Try system path".to_string());
        main_content.push("    ".to_string());
        main_content.push("    return ctypes.CDLL(lib_path)".to_string());
        main_content.push("".to_string());
        main_content.push("_lib = _load_library()".to_string());
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

        // Generate function bindings
        main_content.push("# Function bindings".to_string());
        for func in &interface.functions {
            main_content.push(self.generate_function_binding(func));
            main_content.push("".to_string());
        }

        // Write main module file
        let main_file = output_dir.join(format!("{}.py", self.config.package_info.name));
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
        // Generate __init__.py
        let init_content = format!(
            "\"\"\"
{} - {}

{}
\"\"\"

from .{} import *

__version__ = \"{}\"
__author__ = \"{}\"
__license__ = \"{}\"
",
            self.config.package_info.name,
            self.config.package_info.description,
            self.config.package_info.description,
            self.config.package_info.name,
            self.config.package_info.version,
            self.config.package_info.author,
            self.config.package_info.license
        );

        fs::write(output_dir.join("__init__.py"), init_content)?;

        // Generate setup.py
        let setup_content = format!(
            "from setuptools import setup, find_packages

setup(
    name=\"{}\",
    version=\"{}\",
    description=\"{}\",
    author=\"{}\",
    license=\"{}\",
    packages=find_packages(),
    python_requires=\">=3.7\",
    install_requires=[],
    classifiers=[
        \"Development Status :: 4 - Beta\",
        \"Intended Audience :: Developers\",
        \"License :: OSI Approved :: MIT License\",
        \"Programming Language :: Python :: 3\",
        \"Programming Language :: Python :: 3.7\",
        \"Programming Language :: Python :: 3.8\",
        \"Programming Language :: Python :: 3.9\",
        \"Programming Language :: Python :: 3.10\",
        \"Programming Language :: Python :: 3.11\",
    ],
)
",
            self.config.package_info.name,
            self.config.package_info.version,
            self.config.package_info.description,
            self.config.package_info.author,
            self.config.package_info.license
        );

        fs::write(output_dir.join("setup.py"), setup_content)?;

        // Generate requirements.txt
        fs::write(
            output_dir.join("requirements.txt"),
            "# No additional requirements\n",
        )?;

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
        example_content.push("#!/usr/bin/env python3".to_string());
        example_content.push("\"\"\"".to_string());
        example_content.push("Basic usage example for TrustformeRS Python bindings".to_string());
        example_content.push("\"\"\"".to_string());
        example_content.push("".to_string());
        example_content.push(format!("import {}", self.config.package_info.name));
        example_content.push("".to_string());
        example_content.push("def main():".to_string());
        example_content.push("    print(\"TrustformeRS Python Bindings Example\")".to_string());
        example_content.push("    ".to_string());

        // Add example function calls if available
        if let Some(first_func) = interface.functions.first() {
            if first_func.parameters.is_empty() {
                example_content.push(format!("    # Call {}", first_func.name));
                example_content.push(format!(
                    "    result = {}.{}()",
                    self.config.package_info.name, first_func.name
                ));
                example_content.push("    print(f\"Result: {result}\")".to_string());
            }
        }

        example_content.push("".to_string());
        example_content.push("if __name__ == '__main__':".to_string());
        example_content.push("    main()".to_string());

        fs::write(
            examples_dir.join("basic_usage.py"),
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
        let tests_dir = output_dir.join("tests");
        fs::create_dir_all(&tests_dir)?;

        // Generate basic test file
        let mut test_content = Vec::new();
        test_content.push("import unittest".to_string());
        test_content.push(format!("import {}", self.config.package_info.name));
        test_content.push("".to_string());
        test_content.push("class TestTrustformers(unittest.TestCase):".to_string());
        test_content.push("    def test_import(self):".to_string());
        test_content.push("        \"\"\"Test that the module can be imported\"\"\"".to_string());
        test_content.push("        # If we get here, import worked".to_string());
        test_content.push("        self.assertTrue(True)".to_string());
        test_content.push("".to_string());

        // Add tests for available functions
        for func in interface.functions.iter().take(3) {
            // Limit to first 3 functions
            test_content.push(format!("    def test_{}(self):", func.name));
            test_content.push(format!("        \"\"\"Test {} function\"\"\"", func.name));
            test_content.push("        # Add appropriate test here".to_string());
            test_content.push("        pass".to_string());
            test_content.push("".to_string());
        }

        test_content.push("if __name__ == '__main__':".to_string());
        test_content.push("    unittest.main()".to_string());

        fs::write(tests_dir.join("test_basic.py"), test_content.join("\n"))?;

        // Generate __init__.py for tests
        fs::write(tests_dir.join("__init__.py"), "")?;

        Ok(())
    }
}
