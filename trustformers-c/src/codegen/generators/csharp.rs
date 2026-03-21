//! C# bindings generator for FFI interfaces

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::codegen::ast::{FfiEnum, FfiFunction, FfiInterface, FfiStruct, FfiType};
use crate::codegen::templates::TemplateEngine;
use crate::codegen::{CodeGenConfig, TargetLanguage};
use crate::error::TrustformersResult;

use super::common::{function_can_fail, TypeMapper};
use super::LanguageGenerator;

/// C# bindings generator
pub struct CSharpGenerator {
    config: CodeGenConfig,
    type_mappings: HashMap<String, String>,
}

impl CSharpGenerator {
    pub fn new(config: &CodeGenConfig) -> TrustformersResult<Self> {
        let mut type_mappings = HashMap::new();

        // Basic type mappings for C# P/Invoke
        type_mappings.insert("c_int".to_string(), "int".to_string());
        type_mappings.insert("c_uint".to_string(), "uint".to_string());
        type_mappings.insert("c_short".to_string(), "short".to_string());
        type_mappings.insert("c_ushort".to_string(), "ushort".to_string());
        type_mappings.insert("c_long".to_string(), "long".to_string());
        type_mappings.insert("c_ulong".to_string(), "ulong".to_string());
        type_mappings.insert("c_longlong".to_string(), "long".to_string());
        type_mappings.insert("c_ulonglong".to_string(), "ulong".to_string());
        type_mappings.insert("c_float".to_string(), "float".to_string());
        type_mappings.insert("c_double".to_string(), "double".to_string());
        type_mappings.insert("c_char".to_string(), "sbyte".to_string());
        type_mappings.insert("c_uchar".to_string(), "byte".to_string());
        type_mappings.insert("c_bool".to_string(), "bool".to_string());
        type_mappings.insert("c_void".to_string(), "void".to_string());
        type_mappings.insert("*const c_char".to_string(), "string".to_string());
        type_mappings.insert("*mut c_char".to_string(), "IntPtr".to_string());
        type_mappings.insert("*const c_void".to_string(), "IntPtr".to_string());
        type_mappings.insert("*mut c_void".to_string(), "IntPtr".to_string());

        // Rust primitive type mappings
        type_mappings.insert("i8".to_string(), "sbyte".to_string());
        type_mappings.insert("i16".to_string(), "short".to_string());
        type_mappings.insert("i32".to_string(), "int".to_string());
        type_mappings.insert("i64".to_string(), "long".to_string());
        type_mappings.insert("u8".to_string(), "byte".to_string());
        type_mappings.insert("u16".to_string(), "ushort".to_string());
        type_mappings.insert("u32".to_string(), "uint".to_string());
        type_mappings.insert("u64".to_string(), "ulong".to_string());
        type_mappings.insert("f32".to_string(), "float".to_string());
        type_mappings.insert("f64".to_string(), "double".to_string());
        type_mappings.insert("usize".to_string(), "UIntPtr".to_string());
        type_mappings.insert("isize".to_string(), "IntPtr".to_string());
        type_mappings.insert("()".to_string(), "void".to_string());

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
            "c_longlong" => "long",
            "c_ulonglong" => "ulong",
            "c_float" | "f32" => "float",
            "c_double" | "f64" => "double",
            "c_char" | "i8" => "sbyte",
            "c_uchar" | "u8" => "byte",
            "c_bool" => "bool",
            "c_void" | "()" => "void",
            "isize" => "IntPtr",
            "usize" => "UIntPtr",
            name if name.ends_with("Handle") => "IntPtr",
            _ => "IntPtr", // Default for unknown types
        }
        .to_string()
    }

    fn generate_struct_class(&self, struct_def: &FfiStruct) -> String {
        let mut lines = Vec::new();

        // Struct documentation
        if !struct_def.documentation.is_empty() {
            lines.push("    /// <summary>".to_string());
            for doc_line in &struct_def.documentation {
                lines.push(format!("    /// {}", doc_line));
            }
            lines.push("    /// </summary>".to_string());
        }

        if struct_def.is_opaque {
            // Opaque struct - just a handle wrapper
            lines.push("    [StructLayout(LayoutKind.Sequential)]".to_string());
            lines.push(format!("    public struct {}", struct_def.name));
            lines.push("    {".to_string());
            lines.push("        private IntPtr handle;".to_string());
            lines.push("".to_string());
            lines.push("        public IntPtr Handle => handle;".to_string());
            lines.push("    }".to_string());
        } else {
            // Regular struct with fields
            lines.push("    [StructLayout(LayoutKind.Sequential)]".to_string());
            lines.push(format!("    public struct {}", struct_def.name));
            lines.push("    {".to_string());

            // Field definitions
            for field in &struct_def.fields {
                if !field.is_private {
                    if !field.documentation.is_empty() {
                        lines.push("        /// <summary>".to_string());
                        for doc_line in &field.documentation {
                            lines.push(format!("        /// {}", doc_line));
                        }
                        lines.push("        /// </summary>".to_string());
                    }

                    let field_type = self.map_type(&field.type_info);
                    // Use MarshalAs for string fields
                    if field.type_info.is_string() {
                        lines.push("        [MarshalAs(UnmanagedType.LPStr)]".to_string());
                    }
                    lines.push(format!(
                        "        public {} {};",
                        field_type,
                        Self::to_pascal_case(&field.name)
                    ));
                    lines.push("".to_string());
                }
            }

            lines.push("    }".to_string());
        }

        lines.join("\n")
    }

    fn generate_enum(&self, enum_def: &FfiEnum) -> String {
        let mut lines = Vec::new();

        // Enum documentation
        if !enum_def.documentation.is_empty() {
            lines.push("    /// <summary>".to_string());
            for doc_line in &enum_def.documentation {
                lines.push(format!("    /// {}", doc_line));
            }
            lines.push("    /// </summary>".to_string());
        }

        // Use Flags attribute for bitfield enums
        if enum_def.is_flags {
            lines.push("    [Flags]".to_string());
        }

        lines.push(format!("    public enum {}", enum_def.name));
        lines.push("    {".to_string());

        // Enum variants
        for (i, variant) in enum_def.variants.iter().enumerate() {
            if !variant.documentation.is_empty() {
                lines.push("        /// <summary>".to_string());
                for doc_line in &variant.documentation {
                    lines.push(format!("        /// {}", doc_line));
                }
                lines.push("        /// </summary>".to_string());
            }

            if let Some(deprecation) = &variant.deprecation {
                lines.push(format!("        [Obsolete(\"{}\")]", deprecation.message));
            }

            let comma = if i < enum_def.variants.len() - 1 { "," } else { "" };
            lines.push(format!(
                "        {} = {}{}",
                Self::to_pascal_case(&variant.name),
                variant.value,
                comma
            ));
        }

        lines.push("    }".to_string());

        lines.join("\n")
    }

    fn generate_pinvoke_methods(&self, func: &FfiFunction) -> String {
        let mut lines = Vec::new();

        // Function documentation
        if !func.documentation.is_empty() {
            lines.push("        /// <summary>".to_string());
            for doc_line in &func.documentation {
                lines.push(format!("        /// {}", doc_line));
            }
            lines.push("        /// </summary>".to_string());

            // Add parameter documentation
            for param in &func.parameters {
                let param_type = param.type_info.map_to_language(&TargetLanguage::CSharp);
                lines.push(format!(
                    "        /// <param name=\"{}\">{}</param>",
                    Self::to_camel_case(&param.name),
                    if param.documentation.is_empty() {
                        format!("Parameter of type {}", param_type)
                    } else {
                        param.documentation.join(" ")
                    }
                ));
            }

            // Add return documentation
            if func.return_type.name != "void" {
                let return_type = func.return_type.map_to_language(&TargetLanguage::CSharp);
                lines.push(format!(
                    "        /// <returns>Return value of type {}</returns>",
                    return_type
                ));
            }
        }

        // Add deprecation attribute if needed
        if let Some(deprecation) = &func.deprecation {
            lines.push(format!("        [Obsolete(\"{}\")]", deprecation.message));
        }

        // P/Invoke declaration
        let return_type = self.map_type(&func.return_type);

        // Build DllImport attribute
        let mut dll_import_parts = vec![format!("\"{}\"", "trustformers_c")];
        dll_import_parts.push("CallingConvention = CallingConvention.Cdecl".to_string());
        dll_import_parts.push(format!("EntryPoint = \"{}\"", func.c_name));

        lines.push(format!(
            "        [DllImport({})]",
            dll_import_parts.join(", ")
        ));

        // Build parameter list
        let params: Vec<String> = func
            .parameters
            .iter()
            .map(|p| {
                let param_type = self.map_type(&p.type_info);
                let param_name = Self::to_camel_case(&p.name);

                // Add MarshalAs for string parameters
                if p.type_info.is_string() && param_type == "string" {
                    format!(
                        "[MarshalAs(UnmanagedType.LPStr)] {} {}",
                        param_type, param_name
                    )
                } else {
                    format!("{} {}", param_type, param_name)
                }
            })
            .collect();

        lines.push(format!(
            "        private static extern {} {}({});",
            return_type,
            func.c_name,
            params.join(", ")
        ));

        // Generate C# wrapper method
        lines.push("".to_string());

        // Wrapper documentation
        lines.push("        /// <summary>".to_string());
        lines.push(format!("        /// Wrapper for {}", func.name));
        lines.push("        /// </summary>".to_string());

        let wrapper_params: Vec<String> = func
            .parameters
            .iter()
            .map(|p| {
                format!(
                    "{} {}",
                    self.map_type(&p.type_info),
                    Self::to_camel_case(&p.name)
                )
            })
            .collect();

        let wrapper_name = Self::to_pascal_case(&func.name);

        lines.push(format!(
            "        public static {} {}({})",
            return_type,
            wrapper_name,
            wrapper_params.join(", ")
        ));
        lines.push("        {".to_string());

        // Call native function
        let param_names: Vec<String> =
            func.parameters.iter().map(|p| Self::to_camel_case(&p.name)).collect();

        if function_can_fail(&func.return_type) {
            lines.push(format!(
                "            var result = {}({});",
                func.c_name,
                param_names.join(", ")
            ));
            lines.push("            if (result != 0)".to_string());
            lines.push("            {".to_string());
            lines.push(format!(
                "                throw new TrustformersException($\"Function {} failed with error code {{result}}\");",
                func.name
            ));
            lines.push("            }".to_string());
            lines.push("            return result;".to_string());
        } else if return_type == "void" {
            lines.push(format!(
                "            {}({});",
                func.c_name,
                param_names.join(", ")
            ));
        } else {
            lines.push(format!(
                "            return {}({});",
                func.c_name,
                param_names.join(", ")
            ));
        }

        lines.push("        }".to_string());

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

impl TypeMapper for CSharpGenerator {
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
                return "IntPtr".to_string();
            } else if ffi_type.is_handle() {
                return "IntPtr".to_string();
            } else {
                // For other pointer types, use IntPtr
                return "IntPtr".to_string();
            }
        }

        // Handle array types
        if let Some(size) = ffi_type.array_size {
            let base_type = self.map_base_type(&ffi_type.base_type());
            return format!(
                "[MarshalAs(UnmanagedType.ByValArray, SizeConst = {})] {}[]",
                size, base_type
            );
        }

        // Handle regular types
        self.map_base_type(&ffi_type.name)
    }

    fn map_base_type(&self, type_name: &str) -> String {
        self.map_base_type(type_name)
    }

    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::CSharp
    }
}

impl LanguageGenerator for CSharpGenerator {
    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::CSharp
    }

    fn file_extension(&self) -> &'static str {
        "cs"
    }

    fn generate(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        // Generate main C# file
        let mut main_content = Vec::new();

        // File header
        main_content.push("//".to_string());
        main_content.push(format!(
            "// C# bindings for {}",
            &interface.metadata.library_name
        ));
        main_content.push("//".to_string());
        main_content.push("// TrustformeRS C API bindings".to_string());
        main_content.push(format!("// Version: {}", &interface.metadata.version));
        main_content.push("//".to_string());
        main_content.push("// This file is auto-generated. Do not modify manually.".to_string());
        main_content.push("//".to_string());
        main_content.push("".to_string());

        // Using statements
        main_content.push("using System;".to_string());
        main_content.push("using System.Runtime.InteropServices;".to_string());
        main_content.push("".to_string());

        // Namespace declaration
        let namespace = format!("TrustformeRS.Bindings");
        main_content.push(format!("namespace {}", namespace));
        main_content.push("{".to_string());

        // TrustformersException class
        main_content.push("    /// <summary>".to_string());
        main_content.push("    /// Base exception for TrustformeRS errors".to_string());
        main_content.push("    /// </summary>".to_string());
        main_content.push("    public class TrustformersException : Exception".to_string());
        main_content.push("    {".to_string());
        main_content.push(
            "        public TrustformersException(string message) : base(message) { }".to_string(),
        );
        main_content.push("        public TrustformersException(string message, Exception inner) : base(message, inner) { }".to_string());
        main_content.push("    }".to_string());
        main_content.push("".to_string());

        // Generate enums
        for enum_def in &interface.enums {
            main_content.push(self.generate_enum(enum_def));
            main_content.push("".to_string());
        }

        // Generate structs
        for struct_def in &interface.structs {
            main_content.push(self.generate_struct_class(struct_def));
            main_content.push("".to_string());
        }

        // Generate native methods class
        main_content.push("    /// <summary>".to_string());
        main_content.push("    /// Native method bindings for TrustformeRS".to_string());
        main_content.push("    /// </summary>".to_string());
        main_content.push("    public static class TrustformersNative".to_string());
        main_content.push("    {".to_string());

        // Generate function bindings
        for func in &interface.functions {
            main_content.push(self.generate_pinvoke_methods(func));
            main_content.push("".to_string());
        }

        main_content.push("    }".to_string());

        // Close namespace
        main_content.push("}".to_string());

        // Write main file
        let main_file = output_dir.join(format!("{}.cs", self.config.package_info.name));
        fs::write(&main_file, main_content.join("\n"))?;

        // Generate project file
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
        // Generate .csproj file
        let csproj_content = format!(
            r#"<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>netstandard2.0</TargetFramework>
    <LangVersion>latest</LangVersion>
    <AssemblyName>{}</AssemblyName>
    <RootNamespace>TrustformeRS.Bindings</RootNamespace>
    <Version>{}</Version>
    <Authors>{}</Authors>
    <Description>{}</Description>
    <PackageLicenseExpression>{}</PackageLicenseExpression>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
  </PropertyGroup>

  <ItemGroup>
    <None Include="../native/**/*" Pack="true" PackagePath="runtimes/" />
  </ItemGroup>

</Project>
"#,
            self.config.package_info.name,
            self.config.package_info.version,
            self.config.package_info.author,
            self.config.package_info.description,
            self.config.package_info.license
        );

        fs::write(
            output_dir.join(format!("{}.csproj", self.config.package_info.name)),
            csproj_content,
        )?;

        // Generate NuGet.config
        let nuget_config = r#"<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" protocolVersion="3" />
  </packageSources>
</configuration>
"#;
        fs::write(output_dir.join("NuGet.config"), nuget_config)?;

        // Generate README
        let readme_content = format!(
            "# {} - C# Bindings

{}

## Installation

```bash
dotnet add package {}
```

## Usage

```csharp
using TrustformeRS.Bindings;

// Use the bindings
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
        example_content.push("using System;".to_string());
        example_content.push("using TrustformeRS.Bindings;".to_string());
        example_content.push("".to_string());
        example_content.push("namespace Examples".to_string());
        example_content.push("{".to_string());
        example_content.push("    /// <summary>".to_string());
        example_content
            .push("    /// Basic usage example for TrustformeRS C# bindings".to_string());
        example_content.push("    /// </summary>".to_string());
        example_content.push("    class Program".to_string());
        example_content.push("    {".to_string());
        example_content.push("        static void Main(string[] args)".to_string());
        example_content.push("        {".to_string());
        example_content.push(
            "            Console.WriteLine(\"TrustformeRS C# Bindings Example\");".to_string(),
        );
        example_content.push("".to_string());

        // Add example function calls if available
        if let Some(first_func) = interface.functions.first() {
            if first_func.parameters.is_empty() {
                let wrapper_name = Self::to_pascal_case(&first_func.name);
                example_content.push(format!("            // Call {}", first_func.name));
                example_content.push("            try".to_string());
                example_content.push("            {".to_string());
                example_content.push(format!(
                    "                var result = TrustformersNative.{}();",
                    wrapper_name
                ));
                example_content.push(
                    "                Console.WriteLine($\"Result: {{result}}\");".to_string(),
                );
                example_content.push("            }".to_string());
                example_content.push("            catch (TrustformersException ex)".to_string());
                example_content.push("            {".to_string());
                example_content.push(
                    "                Console.WriteLine($\"Error: {{ex.Message}}\");".to_string(),
                );
                example_content.push("            }".to_string());
            }
        }

        example_content.push("        }".to_string());
        example_content.push("    }".to_string());
        example_content.push("}".to_string());

        fs::write(
            examples_dir.join("BasicUsage.cs"),
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
        let tests_dir = output_dir.join("Tests");
        fs::create_dir_all(&tests_dir)?;

        // Generate basic test file
        let mut test_content = Vec::new();
        test_content.push("using System;".to_string());
        test_content.push("using Xunit;".to_string());
        test_content.push("using TrustformeRS.Bindings;".to_string());
        test_content.push("".to_string());
        test_content.push("namespace TrustformeRS.Tests".to_string());
        test_content.push("{".to_string());
        test_content.push("    public class BasicTests".to_string());
        test_content.push("    {".to_string());
        test_content.push("        [Fact]".to_string());
        test_content.push("        public void TestImport()".to_string());
        test_content.push("        {".to_string());
        test_content.push("            // If we get here, import worked".to_string());
        test_content.push("            Assert.True(true);".to_string());
        test_content.push("        }".to_string());

        // Add tests for available functions
        for func in interface.functions.iter().take(3) {
            let test_name = Self::to_pascal_case(&func.name);
            test_content.push("".to_string());
            test_content.push("        [Fact]".to_string());
            test_content.push(format!("        public void Test{}()", test_name));
            test_content.push("        {".to_string());
            test_content.push(format!("            // Test {} function", func.name));
            test_content.push("            // Add appropriate test here".to_string());
            test_content.push("        }".to_string());
        }

        test_content.push("    }".to_string());
        test_content.push("}".to_string());

        fs::write(tests_dir.join("BasicTests.cs"), test_content.join("\n"))?;

        // Generate test project file
        let test_csproj = format!(
            r#"<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net6.0</TargetFramework>
    <IsPackable>false</IsPackable>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.5.0" />
    <PackageReference Include="xunit" Version="2.4.2" />
    <PackageReference Include="xunit.runner.visualstudio" Version="2.4.5">
      <IncludeAssets>runtime; build; native; contentfiles; analyzers; buildtransitive</IncludeAssets>
      <PrivateAssets>all</PrivateAssets>
    </PackageReference>
  </ItemGroup>

  <ItemGroup>
    <ProjectReference Include="../{}.csproj" />
  </ItemGroup>

</Project>
"#,
            self.config.package_info.name
        );

        fs::write(tests_dir.join("TrustformeRS.Tests.csproj"), test_csproj)?;

        Ok(())
    }
}
