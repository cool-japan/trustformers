//! PHP bindings generator for FFI interfaces
//!
//! Generates PHP 7.4+ FFI bindings for the C API.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::codegen::ast::{FfiEnum, FfiFunction, FfiInterface, FfiStruct, FfiType};
use crate::codegen::templates::TemplateEngine;
use crate::codegen::{CodeGenConfig, TargetLanguage};
use crate::error::TrustformersResult;

use super::common::{function_can_fail, TypeMapper};
use super::LanguageGenerator;

/// PHP bindings generator using FFI extension
pub struct PhpGenerator {
    config: CodeGenConfig,
    type_mappings: HashMap<String, String>,
}

impl PhpGenerator {
    pub fn new(config: &CodeGenConfig) -> TrustformersResult<Self> {
        let mut type_mappings = HashMap::new();

        // Basic type mappings for PHP FFI
        // Reference: https://www.php.net/manual/en/ffi.cdef.php
        type_mappings.insert("c_int".to_string(), "int".to_string());
        type_mappings.insert("c_uint".to_string(), "unsigned int".to_string());
        type_mappings.insert("c_short".to_string(), "short".to_string());
        type_mappings.insert("c_ushort".to_string(), "unsigned short".to_string());
        type_mappings.insert("c_long".to_string(), "long".to_string());
        type_mappings.insert("c_ulong".to_string(), "unsigned long".to_string());
        type_mappings.insert("c_longlong".to_string(), "long long".to_string());
        type_mappings.insert("c_ulonglong".to_string(), "unsigned long long".to_string());
        type_mappings.insert("c_float".to_string(), "float".to_string());
        type_mappings.insert("c_double".to_string(), "double".to_string());
        type_mappings.insert("c_char".to_string(), "char".to_string());
        type_mappings.insert("c_uchar".to_string(), "unsigned char".to_string());
        type_mappings.insert("c_bool".to_string(), "bool".to_string());
        type_mappings.insert("c_void".to_string(), "void".to_string());
        type_mappings.insert("*const c_char".to_string(), "char*".to_string());
        type_mappings.insert("*mut c_char".to_string(), "char*".to_string());
        type_mappings.insert("*const c_void".to_string(), "void*".to_string());
        type_mappings.insert("*mut c_void".to_string(), "void*".to_string());

        // Rust primitive type mappings
        type_mappings.insert("i8".to_string(), "int8_t".to_string());
        type_mappings.insert("i16".to_string(), "int16_t".to_string());
        type_mappings.insert("i32".to_string(), "int32_t".to_string());
        type_mappings.insert("i64".to_string(), "int64_t".to_string());
        type_mappings.insert("u8".to_string(), "uint8_t".to_string());
        type_mappings.insert("u16".to_string(), "uint16_t".to_string());
        type_mappings.insert("u32".to_string(), "uint32_t".to_string());
        type_mappings.insert("u64".to_string(), "uint64_t".to_string());
        type_mappings.insert("f32".to_string(), "float".to_string());
        type_mappings.insert("f64".to_string(), "double".to_string());
        type_mappings.insert("isize".to_string(), "intptr_t".to_string());
        type_mappings.insert("usize".to_string(), "size_t".to_string());

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
            "c_uint" | "u32" => "unsigned int",
            "c_short" | "i16" => "short",
            "c_ushort" | "u16" => "unsigned short",
            "c_long" | "i64" => "long",
            "c_ulong" | "u64" => "unsigned long",
            "c_longlong" => "long long",
            "c_ulonglong" => "unsigned long long",
            "c_float" | "f32" => "float",
            "c_double" | "f64" => "double",
            "c_char" | "i8" => "char",
            "c_uchar" | "u8" => "unsigned char",
            "c_bool" => "bool",
            "c_void" | "()" => "void",
            "isize" => "intptr_t",
            "usize" => "size_t",
            name if name.ends_with("Handle") => "void*",
            _ => "void*", // Default for unknown types
        }
        .to_string()
    }

    fn generate_class_for_struct(&self, struct_def: &FfiStruct) -> String {
        let mut lines = Vec::new();

        // Class documentation
        if !struct_def.documentation.is_empty() {
            lines.push("/**".to_string());
            for doc_line in &struct_def.documentation {
                lines.push(format!(" * {}", doc_line));
            }
            lines.push(" */".to_string());
        }

        if struct_def.is_opaque {
            // Opaque struct - simple wrapper around FFI CData
            lines.push(format!("class {} {{", struct_def.name));
            lines.push("    /** @var \\FFI\\CData */".to_string());
            lines.push("    private $cdata;".to_string());
            lines.push("".to_string());
            lines.push("    /**".to_string());
            lines.push("     * @param \\FFI\\CData $cdata".to_string());
            lines.push("     */".to_string());
            lines.push("    public function __construct($cdata) {".to_string());
            lines.push("        $this->cdata = $cdata;".to_string());
            lines.push("    }".to_string());
            lines.push("".to_string());
            lines.push("    /**".to_string());
            lines.push("     * @return \\FFI\\CData".to_string());
            lines.push("     */".to_string());
            lines.push("    public function getCData() {".to_string());
            lines.push("        return $this->cdata;".to_string());
            lines.push("    }".to_string());
            lines.push("}".to_string());
        } else {
            // Regular struct with fields
            lines.push(format!("class {} {{", struct_def.name));
            lines.push("    /** @var \\FFI\\CData */".to_string());
            lines.push("    private $cdata;".to_string());
            lines.push("".to_string());

            // Constructor
            lines.push("    /**".to_string());
            lines.push("     * @param \\FFI\\CData|null $cdata".to_string());
            lines.push("     */".to_string());
            lines.push("    public function __construct($cdata = null) {".to_string());
            lines.push(
                "        $this->cdata = $cdata ?? Trustformers::$ffi->new('{}');".to_string(),
            );
            lines.push("    }".to_string());
            lines.push("".to_string());

            lines.push("    /**".to_string());
            lines.push("     * @return \\FFI\\CData".to_string());
            lines.push("     */".to_string());
            lines.push("    public function getCData() {".to_string());
            lines.push("        return $this->cdata;".to_string());
            lines.push("    }".to_string());

            // Generate getters and setters for each field
            for field in &struct_def.fields {
                if !field.is_private {
                    lines.push("".to_string());

                    // Field documentation
                    if !field.documentation.is_empty() {
                        lines.push("    /**".to_string());
                        for doc_line in &field.documentation {
                            lines.push(format!("     * {}", doc_line));
                        }
                        lines.push("     */".to_string());
                    }

                    // Convert snake_case to camelCase for PHP
                    let method_name = Self::to_camel_case(&field.name);

                    // Getter
                    lines.push(format!("    public function get{}() {{", method_name));
                    lines.push(format!("        return $this->cdata->{};", field.name));
                    lines.push("    }".to_string());
                    lines.push("".to_string());

                    // Setter
                    lines.push(format!("    public function set{}($value) {{", method_name));
                    lines.push(format!("        $this->cdata->{} = $value;", field.name));
                    lines.push("    }".to_string());
                }
            }

            lines.push("}".to_string());
        }

        lines.join("\n")
    }

    fn generate_enum_class(&self, enum_def: &FfiEnum) -> String {
        let mut lines = Vec::new();

        // Enum documentation
        if !enum_def.documentation.is_empty() {
            lines.push("/**".to_string());
            for doc_line in &enum_def.documentation {
                lines.push(format!(" * {}", doc_line));
            }
            lines.push(" */".to_string());
        }

        lines.push(format!("class {} {{", enum_def.name));

        // Enum constants
        for variant in &enum_def.variants {
            lines.push("".to_string());
            if !variant.documentation.is_empty() {
                lines.push("    /**".to_string());
                for doc_line in &variant.documentation {
                    lines.push(format!("     * {}", doc_line));
                }
                lines.push("     */".to_string());
            }
            if let Some(deprecation) = &variant.deprecation {
                lines.push(format!("    /** @deprecated {} */", deprecation.message));
            }
            lines.push(format!(
                "    public const {} = {};",
                variant.name, variant.value
            ));
        }

        lines.push("}".to_string());
        lines.join("\n")
    }

    fn generate_ffi_bindings(&self, func: &FfiFunction) -> String {
        let mut lines = Vec::new();

        // Generate FFI C declaration
        let param_types: Vec<String> = func
            .parameters
            .iter()
            .map(|p| {
                let type_str = self.map_type(&p.type_info);
                format!("{} {}", type_str, p.name)
            })
            .collect();

        let return_type = self.map_type(&func.return_type);

        lines.push(format!(
            "    {} {}({});",
            return_type,
            func.c_name,
            param_types.join(", ")
        ));

        lines.join("\n")
    }

    fn generate_function_wrapper(&self, func: &FfiFunction) -> String {
        let mut lines = Vec::new();

        lines.push("".to_string());

        // Function documentation
        if !func.documentation.is_empty() {
            lines.push("    /**".to_string());
            for doc_line in &func.documentation {
                lines.push(format!("     * {}", doc_line));
            }
            lines.push("     *".to_string());

            // Add parameter documentation
            for param in &func.parameters {
                lines.push(format!("     * @param mixed ${}", param.name));
            }

            // Add return documentation
            if func.return_type.name != "void" {
                lines.push("     * @return mixed".to_string());
            }

            if func.can_fail {
                lines.push("     * @throws TrustformersException".to_string());
            }

            lines.push("     */".to_string());
        }

        // Convert to camelCase for PHP method name
        let method_name = Self::to_camel_case(&func.name);

        let param_names: Vec<String> =
            func.parameters.iter().map(|p| format!("${}", p.name)).collect();

        lines.push(format!(
            "    public static function {}({}) {{",
            method_name,
            param_names.join(", ")
        ));

        // Function implementation
        if function_can_fail(&func.return_type) {
            lines.push(format!(
                "        $result = self::$ffi->{}({});",
                func.c_name,
                param_names.join(", ")
            ));
            lines.push("        if ($result !== 0) {".to_string());
            lines.push(format!(
                "            throw new TrustformersException('Function {} failed with error code ' . $result);",
                func.name
            ));
            lines.push("        }".to_string());
            lines.push("        return $result;".to_string());
        } else {
            lines.push(format!(
                "        return self::$ffi->{}({});",
                func.c_name,
                param_names.join(", ")
            ));
        }

        lines.push("    }".to_string());

        lines.join("\n")
    }

    /// Convert snake_case to camelCase
    fn to_camel_case(s: &str) -> String {
        let mut result = String::new();
        let mut capitalize_next = false;

        for (i, c) in s.chars().enumerate() {
            if c == '_' {
                capitalize_next = true;
            } else if capitalize_next || i == 0 {
                result.push(c.to_ascii_uppercase());
                capitalize_next = false;
            } else {
                result.push(c);
            }
        }

        result
    }
}

impl TypeMapper for PhpGenerator {
    fn map_type(&self, ffi_type: &FfiType) -> String {
        // Check for custom mappings first
        if let Some(mapped) = self.type_mappings.get(&ffi_type.name) {
            return mapped.clone();
        }

        // Handle pointer types
        if ffi_type.is_pointer() {
            if ffi_type.is_string() {
                return "char*".to_string();
            } else if ffi_type.base_type() == "c_void" {
                return "void*".to_string();
            } else {
                return format!("{}*", self.map_base_type(&ffi_type.base_type()));
            }
        }

        // Handle array types
        if let Some(size) = ffi_type.array_size {
            let base_type = self.map_base_type(&ffi_type.base_type());
            return format!("{}[{}]", base_type, size);
        }

        // Handle regular types
        self.map_base_type(&ffi_type.name)
    }

    fn map_base_type(&self, type_name: &str) -> String {
        self.map_base_type(type_name)
    }

    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::Php
    }
}

impl LanguageGenerator for PhpGenerator {
    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::Php
    }

    fn file_extension(&self) -> &'static str {
        "php"
    }

    fn generate(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        // Generate main module file
        let mut main_content = Vec::new();

        // PHP opening tag
        main_content.push("<?php".to_string());
        main_content.push("".to_string());

        // File documentation
        main_content.push("/**".to_string());
        main_content.push(format!(
            " * PHP FFI bindings for {}",
            &interface.metadata.library_name
        ));
        main_content.push(" *".to_string());
        main_content.push(" * TrustformeRS C API bindings using PHP FFI".to_string());
        main_content.push(format!(" * Version: {}", &interface.metadata.version));
        main_content.push(" *".to_string());
        main_content.push(" * Requires PHP 7.4+ with FFI extension enabled".to_string());
        main_content.push(" */".to_string());
        main_content.push("".to_string());

        // Namespace
        main_content.push(format!(
            "namespace {};",
            Self::to_camel_case(&self.config.package_info.name)
        ));
        main_content.push("".to_string());

        // Use statements
        main_content.push("use FFI;".to_string());
        main_content.push("use Exception;".to_string());
        main_content.push("".to_string());

        // Exception class
        main_content.push("/**".to_string());
        main_content.push(" * TrustformeRS exception class".to_string());
        main_content.push(" */".to_string());
        main_content.push("class TrustformersException extends Exception {".to_string());
        main_content.push("}".to_string());
        main_content.push("".to_string());

        // Generate enum classes
        for enum_def in &interface.enums {
            main_content.push(self.generate_enum_class(enum_def));
            main_content.push("".to_string());
        }

        // Generate struct classes
        for struct_def in &interface.structs {
            main_content.push(self.generate_class_for_struct(struct_def));
            main_content.push("".to_string());
        }

        // Generate main Trustformers class with FFI bindings
        main_content.push("/**".to_string());
        main_content.push(" * Main TrustformeRS FFI interface".to_string());
        main_content.push(" */".to_string());
        main_content.push("class Trustformers {".to_string());
        main_content.push("    /** @var \\FFI */".to_string());
        main_content.push("    public static $ffi;".to_string());
        main_content.push("".to_string());

        // Static initialization
        main_content.push("    /**".to_string());
        main_content.push("     * Initialize FFI bindings".to_string());
        main_content.push("     *".to_string());
        main_content.push("     * @param string|null $libraryPath".to_string());
        main_content.push("     * @throws TrustformersException".to_string());
        main_content.push("     */".to_string());
        main_content.push("    public static function init($libraryPath = null) {".to_string());
        main_content.push("        if (self::$ffi !== null) {".to_string());
        main_content.push("            return; // Already initialized".to_string());
        main_content.push("        }".to_string());
        main_content.push("".to_string());
        main_content.push("        if (!extension_loaded('ffi')) {".to_string());
        main_content.push(
            "            throw new TrustformersException('FFI extension is not loaded. Please enable it in php.ini');"
                .to_string(),
        );
        main_content.push("        }".to_string());
        main_content.push("".to_string());

        // Library path detection
        main_content.push("        if ($libraryPath === null) {".to_string());
        main_content.push("            if (PHP_OS_FAMILY === 'Windows') {".to_string());
        main_content.push("                $libName = 'trustformers_c.dll';".to_string());
        main_content.push("            } elseif (PHP_OS_FAMILY === 'Darwin') {".to_string());
        main_content.push("                $libName = 'libtrusformers_c.dylib';".to_string());
        main_content.push("            } else {".to_string());
        main_content.push("                $libName = 'libtrusformers_c.so';".to_string());
        main_content.push("            }".to_string());
        main_content.push(
            "            $libraryPath = __DIR__ . DIRECTORY_SEPARATOR . $libName;".to_string(),
        );
        main_content.push("        }".to_string());
        main_content.push("".to_string());
        main_content.push("        if (!file_exists($libraryPath)) {".to_string());
        main_content.push(
            "            throw new TrustformersException('Library not found: ' . $libraryPath);"
                .to_string(),
        );
        main_content.push("        }".to_string());
        main_content.push("".to_string());

        // FFI C definitions
        main_content.push("        $cdef = <<<'CDEF'".to_string());

        // Add struct definitions
        for struct_def in &interface.structs {
            if !struct_def.is_opaque {
                main_content.push(format!("typedef struct {} {{", struct_def.c_name));
                for field in &struct_def.fields {
                    if !field.is_private {
                        let field_type = self.map_type(&field.type_info);
                        main_content.push(format!("    {} {};", field_type, field.name));
                    }
                }
                main_content.push(format!("}} {};", struct_def.c_name));
            } else {
                main_content.push(format!(
                    "typedef struct {} {};",
                    struct_def.c_name, struct_def.c_name
                ));
            }
        }

        main_content.push("".to_string());

        // Add function declarations
        for func in &interface.functions {
            main_content.push(self.generate_ffi_bindings(func));
        }

        main_content.push("CDEF;".to_string());
        main_content.push("".to_string());
        main_content.push("        self::$ffi = FFI::cdef($cdef, $libraryPath);".to_string());
        main_content.push("    }".to_string());

        // Generate function wrappers
        for func in &interface.functions {
            main_content.push(self.generate_function_wrapper(func));
        }

        main_content.push("}".to_string());
        main_content.push("".to_string());

        // Auto-initialize
        main_content.push("// Auto-initialize FFI".to_string());
        main_content.push("Trustformers::init();".to_string());

        // Write main module file
        let class_name = Self::to_camel_case(&self.config.package_info.name);
        let main_file = output_dir.join(format!("{}.php", class_name));
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
        // Generate composer.json
        let composer_content = format!(
            r#"{{
    "name": "{}/{}",
    "description": "{}",
    "type": "library",
    "license": "{}",
    "version": "{}",
    "authors": [
        {{
            "name": "{}",
            "email": "example@example.com"
        }}
    ],
    "require": {{
        "php": ">=7.4",
        "ext-ffi": "*"
    }},
    "require-dev": {{
        "phpunit/phpunit": "^9.0"
    }},
    "autoload": {{
        "psr-4": {{
            "{}\\": "src/"
        }}
    }},
    "autoload-dev": {{
        "psr-4": {{
            "{}\\Tests\\": "tests/"
        }}
    }}
}}
"#,
            self.config.package_info.author.to_lowercase().replace(' ', "-"),
            self.config.package_info.name.to_lowercase().replace('_', "-"),
            self.config.package_info.description,
            self.config.package_info.license,
            self.config.package_info.version,
            self.config.package_info.author,
            Self::to_camel_case(&self.config.package_info.name),
            Self::to_camel_case(&self.config.package_info.name)
        );

        fs::write(output_dir.join("composer.json"), composer_content)?;

        // Generate README.md
        let readme_content = format!(
            r#"# {} PHP Bindings

{}

## Requirements

- PHP 7.4 or higher
- FFI extension enabled

## Installation

```bash
composer require {}/{}
```

## Usage

```php
<?php

require_once 'vendor/autoload.php';

use {}\\Trustformers;

// FFI is auto-initialized
// Use the bindings...
```

## Documentation

See the official documentation for more information.

## License

{}
"#,
            self.config.package_info.name,
            self.config.package_info.description,
            self.config.package_info.author.to_lowercase().replace(' ', "-"),
            self.config.package_info.name.to_lowercase().replace('_', "-"),
            Self::to_camel_case(&self.config.package_info.name),
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
        let examples_dir = output_dir.join("examples");
        fs::create_dir_all(&examples_dir)?;

        // Generate basic usage example
        let mut example_content = Vec::new();
        example_content.push("<?php".to_string());
        example_content.push("".to_string());
        example_content.push("/**".to_string());
        example_content.push(" * Basic usage example for TrustformeRS PHP bindings".to_string());
        example_content.push(" */".to_string());
        example_content.push("".to_string());
        example_content.push("require_once __DIR__ . '/../vendor/autoload.php';".to_string());
        example_content.push("".to_string());
        example_content.push(format!(
            "use {}\\Trustformers;",
            Self::to_camel_case(&self.config.package_info.name)
        ));
        example_content.push("".to_string());
        example_content.push("try {".to_string());
        example_content.push("    echo \"TrustformeRS PHP Bindings Example\\n\";".to_string());
        example_content.push("".to_string());

        // Add example function calls if available
        if let Some(first_func) = interface.functions.first() {
            if first_func.parameters.is_empty() {
                let method_name = Self::to_camel_case(&first_func.name);
                example_content.push(format!("    // Call {}", first_func.name));
                example_content.push(format!("    $result = Trustformers::{}();", method_name));
                example_content.push("    echo \"Result: $result\\n\";".to_string());
            }
        }

        example_content.push("".to_string());
        example_content.push("} catch (\\Exception $e) {".to_string());
        example_content.push("    echo \"Error: \" . $e->getMessage() . \"\\n\";".to_string());
        example_content.push("}".to_string());

        fs::write(
            examples_dir.join("basic_usage.php"),
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

        // Generate PHPUnit test
        let mut test_content = Vec::new();
        test_content.push("<?php".to_string());
        test_content.push("".to_string());
        test_content.push(format!(
            "namespace {}\\Tests;",
            Self::to_camel_case(&self.config.package_info.name)
        ));
        test_content.push("".to_string());
        test_content.push("use PHPUnit\\Framework\\TestCase;".to_string());
        test_content.push(format!(
            "use {}\\Trustformers;",
            Self::to_camel_case(&self.config.package_info.name)
        ));
        test_content.push("".to_string());
        test_content.push("class TrustformersTest extends TestCase {".to_string());
        test_content.push("    public function testFfiLoaded() {".to_string());
        test_content.push("        $this->assertTrue(extension_loaded('ffi'));".to_string());
        test_content.push("    }".to_string());
        test_content.push("".to_string());

        // Add tests for available functions
        for func in interface.functions.iter().take(3) {
            // Limit to first 3 functions
            let method_name = Self::to_camel_case(&func.name);
            test_content.push(format!("    public function test{}() {{", method_name));
            test_content.push(format!("        // Test {} function", func.name));
            test_content.push(
                "        $this->markTestIncomplete('Add appropriate test here');".to_string(),
            );
            test_content.push("    }".to_string());
            test_content.push("".to_string());
        }

        test_content.push("}".to_string());

        fs::write(
            tests_dir.join("TrustformersTest.php"),
            test_content.join("\n"),
        )?;

        // Generate phpunit.xml
        let phpunit_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<phpunit xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:noNamespaceSchemaLocation="https://schema.phpunit.de/9.5/phpunit.xsd"
         bootstrap="vendor/autoload.php"
         colors="true">
    <testsuites>
        <testsuite name="TrustformeRS Test Suite">
            <directory>tests</directory>
        </testsuite>
    </testsuites>
    <coverage>
        <include>
            <directory suffix=".php">src</directory>
        </include>
    </coverage>
</phpunit>
"#;

        fs::write(output_dir.join("phpunit.xml"), phpunit_xml)?;

        Ok(())
    }
}
