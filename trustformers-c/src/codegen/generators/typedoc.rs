//! TypeDoc bindings generator for FFI interfaces
//!
//! Generates TypeDoc-compatible JSON documentation for FFI interfaces.
//! TypeDoc is a documentation generator for TypeScript projects.

use std::fs;
use std::path::Path;

use serde_json::{json, Value};

use crate::codegen::ast::{
    ConstantValue, FfiConstant, FfiEnum, FfiEnumVariant, FfiField, FfiFunction, FfiInterface,
    FfiParameter, FfiStruct, FfiType, FfiTypeAlias, PrimitiveType,
};
use crate::codegen::templates::TemplateEngine;
use crate::codegen::{CodeGenConfig, TargetLanguage};
use crate::error::TrustformersResult;

use super::LanguageGenerator;

/// TypeDoc bindings generator
pub struct TypeDocGenerator {
    config: CodeGenConfig,
    next_id: std::cell::Cell<u32>,
}

/// TypeDoc reflection kinds
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum TypeDocKind {
    Project = 1,
    Module = 2,
    Namespace = 4,
    Enum = 8,
    EnumMember = 16,
    Variable = 32,
    Function = 64,
    Class = 128,
    Interface = 256,
    Constructor = 512,
    Property = 1024,
    Method = 2048,
    CallSignature = 4096,
    IndexSignature = 8192,
    ConstructorSignature = 16384,
    Parameter = 32768,
    TypeLiteral = 65536,
    TypeParameter = 131072,
    Accessor = 262144,
    GetSignature = 524288,
    SetSignature = 1048576,
    TypeAlias = 2097152,
    Reference = 16777216,
}

impl TypeDocGenerator {
    pub fn new(config: &CodeGenConfig) -> TrustformersResult<Self> {
        Ok(Self {
            config: config.clone(),
            next_id: std::cell::Cell::new(0),
        })
    }

    /// Get next unique ID for TypeDoc nodes
    fn next_id(&self) -> u32 {
        let id = self.next_id.get();
        self.next_id.set(id + 1);
        id
    }

    /// Generate a TypeDoc JSON node
    fn generate_json_node(
        &self,
        name: &str,
        kind: TypeDocKind,
        comment: Option<&Vec<String>>,
    ) -> Value {
        let mut node = json!({
            "id": self.next_id(),
            "name": name,
            "kind": kind as u32,
            "flags": {}
        });

        if let Some(doc) = comment {
            if !doc.is_empty() {
                node["comment"] = json!({
                    "summary": doc.iter()
                        .map(|line| json!({"kind": "text", "text": line}))
                        .collect::<Vec<_>>()
                });
            }
        }

        node
    }

    /// Map FFI type to TypeScript type string
    fn map_type_to_typescript(&self, ffi_type: &FfiType) -> String {
        if ffi_type.is_string() {
            return "string".to_string();
        }

        if ffi_type.is_handle() {
            return "number".to_string();
        }

        if ffi_type.is_callback {
            return "Function".to_string();
        }

        match &ffi_type.primitive_type {
            Some(PrimitiveType::Bool) => "boolean".to_string(),
            Some(PrimitiveType::Void) => "void".to_string(),
            Some(PrimitiveType::Int8)
            | Some(PrimitiveType::Int16)
            | Some(PrimitiveType::Int32)
            | Some(PrimitiveType::Int64)
            | Some(PrimitiveType::UInt8)
            | Some(PrimitiveType::UInt16)
            | Some(PrimitiveType::UInt32)
            | Some(PrimitiveType::UInt64)
            | Some(PrimitiveType::IntPtr)
            | Some(PrimitiveType::UIntPtr)
            | Some(PrimitiveType::Float32)
            | Some(PrimitiveType::Float64) => "number".to_string(),
            Some(PrimitiveType::Char) => "string".to_string(),
            Some(PrimitiveType::CString) => "string".to_string(),
            Some(PrimitiveType::Handle) | Some(PrimitiveType::OpaquePointer) => {
                "number".to_string()
            },
            None => {
                if ffi_type.is_pointer() {
                    "number".to_string() // Pointer as number
                } else {
                    ffi_type.name.clone()
                }
            },
        }
    }

    /// Generate TypeDoc type node
    fn generate_type_node(&self, ffi_type: &FfiType) -> Value {
        let type_name = self.map_type_to_typescript(ffi_type);

        if ffi_type.array_size.is_some() {
            json!({
                "type": "array",
                "elementType": {
                    "type": "intrinsic",
                    "name": type_name
                }
            })
        } else {
            json!({
                "type": "intrinsic",
                "name": type_name
            })
        }
    }

    /// Generate documentation for a function parameter
    fn generate_parameter_docs(&self, param: &FfiParameter) -> Value {
        let mut param_node = self.generate_json_node(
            &param.name,
            TypeDocKind::Parameter,
            Some(&param.documentation),
        );

        param_node["type"] = self.generate_type_node(&param.type_info);

        if param.is_optional {
            param_node["flags"]["isOptional"] = json!(true);
        }

        if let Some(default) = &param.default_value {
            param_node["defaultValue"] = json!(default);
        }

        param_node
    }

    /// Generate documentation for an FFI function
    fn generate_function_docs(&self, func: &FfiFunction) -> Value {
        let mut func_node =
            self.generate_json_node(&func.name, TypeDocKind::Function, Some(&func.documentation));

        // Generate signature
        let sig_id = self.next_id();
        let mut signature = json!({
            "id": sig_id,
            "name": func.name,
            "kind": TypeDocKind::CallSignature as u32,
            "type": self.generate_type_node(&func.return_type)
        });

        // Add parameters
        if !func.parameters.is_empty() {
            signature["parameters"] = json!(func
                .parameters
                .iter()
                .map(|p| self.generate_parameter_docs(p))
                .collect::<Vec<_>>());
        }

        // Add parameter documentation to comment
        if !func.parameters.is_empty() || !func.return_type.name.is_empty() {
            let mut block_tags = Vec::new();

            for param in &func.parameters {
                let param_type = self.map_type_to_typescript(&param.type_info);
                let param_desc = if param.documentation.is_empty() {
                    "Parameter description".to_string()
                } else {
                    param.documentation.join(" ")
                };

                block_tags.push(json!({
                    "tag": "@param",
                    "content": [{
                        "kind": "text",
                        "text": format!("{} {{{}}} {}", param.name, param_type, param_desc)
                    }]
                }));
            }

            if func.return_type.name != "void" {
                let return_type = self.map_type_to_typescript(&func.return_type);
                block_tags.push(json!({
                    "tag": "@returns",
                    "content": [{
                        "kind": "text",
                        "text": format!("{{{}}}", return_type)
                    }]
                }));
            }

            if !block_tags.is_empty() {
                signature["comment"] = json!({
                    "blockTags": block_tags
                });
            }
        }

        func_node["signatures"] = json!([signature]);

        // Add flags
        if func.is_unsafe {
            func_node["flags"]["isUnsafe"] = json!(true);
        }

        if let Some(deprecation) = &func.deprecation {
            func_node["flags"]["isDeprecated"] = json!(true);
            if func_node.get("comment").is_none() {
                func_node["comment"] = json!({});
            }
            func_node["comment"]["blockTags"] = json!([{
                "tag": "@deprecated",
                "content": [{
                    "kind": "text",
                    "text": deprecation.message.clone()
                }]
            }]);
        }

        func_node
    }

    /// Generate documentation for an FFI struct field
    fn generate_field_docs(&self, field: &FfiField) -> Value {
        let mut field_node = self.generate_json_node(
            &field.name,
            TypeDocKind::Property,
            Some(&field.documentation),
        );

        field_node["type"] = self.generate_type_node(&field.type_info);

        if field.is_private {
            field_node["flags"]["isPrivate"] = json!(true);
        }

        field_node
    }

    /// Generate documentation for an FFI struct/interface
    fn generate_interface_docs(&self, struct_def: &FfiStruct) -> Value {
        let mut interface_node = self.generate_json_node(
            &struct_def.name,
            TypeDocKind::Interface,
            Some(&struct_def.documentation),
        );

        if struct_def.is_opaque {
            interface_node["comment"] = json!({
                "summary": [{
                    "kind": "text",
                    "text": "Opaque structure - internal implementation hidden"
                }]
            });
        } else {
            // Add fields
            let children: Vec<Value> = struct_def
                .fields
                .iter()
                .filter(|f| !f.is_private)
                .map(|f| self.generate_field_docs(f))
                .collect();

            if !children.is_empty() {
                interface_node["children"] = json!(children);
            }
        }

        if let Some(deprecation) = &struct_def.deprecation {
            interface_node["flags"]["isDeprecated"] = json!(true);
            if interface_node.get("comment").is_none() {
                interface_node["comment"] = json!({});
            }
            interface_node["comment"]["blockTags"] = json!([{
                "tag": "@deprecated",
                "content": [{
                    "kind": "text",
                    "text": deprecation.message.clone()
                }]
            }]);
        }

        interface_node
    }

    /// Generate documentation for an enum variant
    fn generate_enum_member_docs(&self, variant: &FfiEnumVariant) -> Value {
        let mut member_node = self.generate_json_node(
            &variant.name,
            TypeDocKind::EnumMember,
            Some(&variant.documentation),
        );

        member_node["defaultValue"] = json!(variant.value.to_string());

        if let Some(deprecation) = &variant.deprecation {
            member_node["flags"]["isDeprecated"] = json!(true);
            if member_node.get("comment").is_none() {
                member_node["comment"] = json!({});
            }
            member_node["comment"]["blockTags"] = json!([{
                "tag": "@deprecated",
                "content": [{
                    "kind": "text",
                    "text": deprecation.message.clone()
                }]
            }]);
        }

        member_node
    }

    /// Generate documentation for an FFI enum
    fn generate_enum_docs(&self, enum_def: &FfiEnum) -> Value {
        let mut enum_node = self.generate_json_node(
            &enum_def.name,
            TypeDocKind::Enum,
            Some(&enum_def.documentation),
        );

        let children: Vec<Value> =
            enum_def.variants.iter().map(|v| self.generate_enum_member_docs(v)).collect();

        if !children.is_empty() {
            enum_node["children"] = json!(children);
        }

        if enum_def.is_flags {
            if enum_node.get("comment").is_none() {
                enum_node["comment"] = json!({});
            }
            enum_node["comment"]["blockTags"] = json!([{
                "tag": "@flags",
                "content": [{
                    "kind": "text",
                    "text": "This is a flags enum (bitfield)"
                }]
            }]);
        }

        if let Some(deprecation) = &enum_def.deprecation {
            enum_node["flags"]["isDeprecated"] = json!(true);
            if enum_node.get("comment").is_none() {
                enum_node["comment"] = json!({});
            }
            enum_node["comment"]["blockTags"] = json!([{
                "tag": "@deprecated",
                "content": [{
                    "kind": "text",
                    "text": deprecation.message.clone()
                }]
            }]);
        }

        enum_node
    }

    /// Generate documentation for a constant
    fn generate_constant_docs(&self, constant: &FfiConstant) -> Value {
        let mut const_node = self.generate_json_node(
            &constant.name,
            TypeDocKind::Variable,
            Some(&constant.documentation),
        );

        const_node["type"] = self.generate_type_node(&constant.type_info);
        const_node["flags"]["isConst"] = json!(true);

        let value_str = match &constant.value {
            ConstantValue::Integer(v) => v.to_string(),
            ConstantValue::UInteger(v) => v.to_string(),
            ConstantValue::Float(v) => v.to_string(),
            ConstantValue::String(v) => format!("\"{}\"", v),
            ConstantValue::Boolean(v) => v.to_string(),
            ConstantValue::Null => "null".to_string(),
        };

        const_node["defaultValue"] = json!(value_str);

        if let Some(deprecation) = &constant.deprecation {
            const_node["flags"]["isDeprecated"] = json!(true);
            if const_node.get("comment").is_none() {
                const_node["comment"] = json!({});
            }
            const_node["comment"]["blockTags"] = json!([{
                "tag": "@deprecated",
                "content": [{
                    "kind": "text",
                    "text": deprecation.message.clone()
                }]
            }]);
        }

        const_node
    }

    /// Generate documentation for a type alias
    fn generate_type_alias_docs(&self, type_alias: &FfiTypeAlias) -> Value {
        let mut alias_node = self.generate_json_node(
            &type_alias.name,
            TypeDocKind::TypeAlias,
            Some(&type_alias.documentation),
        );

        alias_node["type"] = self.generate_type_node(&type_alias.target_type);

        if let Some(deprecation) = &type_alias.deprecation {
            alias_node["flags"]["isDeprecated"] = json!(true);
            if alias_node.get("comment").is_none() {
                alias_node["comment"] = json!({});
            }
            alias_node["comment"]["blockTags"] = json!([{
                "tag": "@deprecated",
                "content": [{
                    "kind": "text",
                    "text": deprecation.message.clone()
                }]
            }]);
        }

        alias_node
    }

    /// Generate complete TypeDoc JSON structure
    fn generate_typedoc_json(&self, interface: &FfiInterface) -> Value {
        // Reset ID counter
        self.next_id.set(0);

        let mut children = Vec::new();

        // Add all functions
        for func in &interface.functions {
            children.push(self.generate_function_docs(func));
        }

        // Add all structs/interfaces
        for struct_def in &interface.structs {
            children.push(self.generate_interface_docs(struct_def));
        }

        // Add all enums
        for enum_def in &interface.enums {
            children.push(self.generate_enum_docs(enum_def));
        }

        // Add all constants
        for constant in &interface.constants {
            children.push(self.generate_constant_docs(constant));
        }

        // Add all type aliases
        for type_alias in &interface.type_aliases {
            children.push(self.generate_type_alias_docs(type_alias));
        }

        // Create module node
        json!({
            "id": 0,
            "name": interface.metadata.library_name.clone(),
            "kind": TypeDocKind::Module as u32,
            "flags": {},
            "children": children,
            "groups": [
                {
                    "title": "Functions",
                    "children": interface.functions.iter().enumerate().map(|(i, _)| i + 1).collect::<Vec<_>>()
                },
                {
                    "title": "Interfaces",
                    "children": interface.structs.iter().enumerate()
                        .map(|(i, _)| interface.functions.len() + i + 1)
                        .collect::<Vec<_>>()
                },
                {
                    "title": "Enumerations",
                    "children": interface.enums.iter().enumerate()
                        .map(|(i, _)| interface.functions.len() + interface.structs.len() + i + 1)
                        .collect::<Vec<_>>()
                }
            ]
        })
    }

    /// Generate typedoc.json configuration file
    fn generate_typedoc_config(&self, interface: &FfiInterface) -> String {
        let config = json!({
            "$schema": "https://typedoc.org/schema.json",
            "name": interface.metadata.library_name,
            "entryPoints": ["./index.ts"],
            "out": "./docs",
            "json": "./docs/documentation.json",
            "exclude": [
                "**/node_modules/**"
            ],
            "readme": "README.md",
            "plugin": [],
            "includeVersion": true,
            "categorizeByGroup": true,
            "defaultCategory": "Other",
            "categoryOrder": [
                "Functions",
                "Interfaces",
                "Enumerations",
                "Type Aliases",
                "*"
            ]
        });

        serde_json::to_string_pretty(&config).unwrap()
    }

    /// Generate package.json for TypeDoc project
    fn generate_package_json(&self, interface: &FfiInterface) -> String {
        let package = json!({
            "name": format!("{}-docs", interface.metadata.library_name),
            "version": interface.metadata.version,
            "description": format!("TypeDoc documentation for {}", interface.metadata.library_name),
            "scripts": {
                "docs": "typedoc",
                "docs:json": "typedoc --json docs/documentation.json"
            },
            "devDependencies": {
                "typedoc": "^0.25.0",
                "typescript": "^5.3.0"
            }
        });

        serde_json::to_string_pretty(&package).unwrap()
    }

    /// Generate README.md with build instructions
    fn generate_readme(&self, interface: &FfiInterface) -> String {
        format!(
            r#"# {} Documentation

TypeDoc-compatible documentation for the {} FFI interface.

## Building Documentation

### Prerequisites

- Node.js (v16 or later)
- npm or yarn

### Installation

```bash
npm install
```

or

```bash
yarn install
```

### Generate Documentation

To generate HTML documentation:

```bash
npm run docs
```

To generate JSON documentation only:

```bash
npm run docs:json
```

The generated documentation will be available in the `docs/` directory.

### View Documentation

Open `docs/index.html` in your web browser to view the documentation.

## Project Structure

- `typedoc.json` - TypeDoc configuration
- `package.json` - npm package configuration
- `documentation.json` - Raw TypeDoc JSON output
- `README.md` - This file

## Library Information

- **Name**: {}
- **Version**: {}
- **License**: MIT

## Generated Content

This documentation includes:

- {} functions
- {} interfaces/structs
- {} enumerations
- {} constants
- {} type aliases

## TypeDoc Resources

- [TypeDoc Documentation](https://typedoc.org/)
- [TypeDoc GitHub](https://github.com/TypeStrong/typedoc)

---

*Generated by TrustformeRS FFI Code Generator*
"#,
            interface.metadata.library_name,
            interface.metadata.library_name,
            interface.metadata.library_name,
            interface.metadata.version,
            interface.functions.len(),
            interface.structs.len(),
            interface.enums.len(),
            interface.constants.len(),
            interface.type_aliases.len()
        )
    }
}

impl LanguageGenerator for TypeDocGenerator {
    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::TypeDoc
    }

    fn file_extension(&self) -> &'static str {
        "json"
    }

    fn generate(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        // Create output directory
        fs::create_dir_all(output_dir)?;

        // Generate TypeDoc JSON documentation
        let typedoc_json = self.generate_typedoc_json(interface);
        let json_path = output_dir.join("documentation.json");
        fs::write(&json_path, serde_json::to_string_pretty(&typedoc_json)?)?;

        // Generate typedoc.json configuration
        let config = self.generate_typedoc_config(interface);
        let config_path = output_dir.join("typedoc.json");
        fs::write(&config_path, config)?;

        // Generate package.json
        let package_json = self.generate_package_json(interface);
        let package_path = output_dir.join("package.json");
        fs::write(&package_path, package_json)?;

        // Generate README.md
        let readme = self.generate_readme(interface);
        let readme_path = output_dir.join("README.md");
        fs::write(&readme_path, readme)?;

        // Create a simple index.ts placeholder
        let index_ts = format!(
            r#"/**
 * {} FFI Interface
 *
 * This file is a placeholder for TypeDoc generation.
 * The actual FFI bindings are provided through the C library.
 *
 * @module {}
 */

// This is a placeholder TypeScript file for TypeDoc documentation generation
export {{}};
"#,
            interface.metadata.library_name, interface.metadata.library_name
        );
        let index_path = output_dir.join("index.ts");
        fs::write(&index_path, index_ts)?;

        println!(
            "Generated TypeDoc documentation in {}",
            output_dir.display()
        );
        println!("  - documentation.json: Raw TypeDoc JSON");
        println!("  - typedoc.json: TypeDoc configuration");
        println!("  - package.json: npm package file");
        println!("  - README.md: Build instructions");
        println!("  - index.ts: TypeScript placeholder");
        println!();
        println!("To build HTML documentation:");
        println!("  cd {}", output_dir.display());
        println!("  npm install");
        println!("  npm run docs");

        Ok(())
    }

    fn generate_package_files(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        // Package files are already generated in generate() method
        let _ = (interface, output_dir);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typedoc_generator_creation() {
        let config = CodeGenConfig::default();
        let generator = TypeDocGenerator::new(&config);
        assert!(generator.is_ok());
    }

    #[test]
    fn test_type_mapping() {
        let config = CodeGenConfig::default();
        let generator = TypeDocGenerator::new(&config).unwrap();

        let int_type = FfiType {
            name: "c_int".to_string(),
            primitive_type: Some(PrimitiveType::Int32),
            ..Default::default()
        };
        assert_eq!(generator.map_type_to_typescript(&int_type), "number");

        let string_type = FfiType {
            name: "*const c_char".to_string(),
            primitive_type: Some(PrimitiveType::CString),
            is_pointer: true,
            ..Default::default()
        };
        assert_eq!(generator.map_type_to_typescript(&string_type), "string");

        let bool_type = FfiType {
            name: "bool".to_string(),
            primitive_type: Some(PrimitiveType::Bool),
            ..Default::default()
        };
        assert_eq!(generator.map_type_to_typescript(&bool_type), "boolean");
    }

    #[test]
    fn test_generate_function_docs() {
        let config = CodeGenConfig::default();
        let generator = TypeDocGenerator::new(&config).unwrap();

        let func = FfiFunction {
            name: "test_function".to_string(),
            c_name: "trustformers_test_function".to_string(),
            documentation: vec!["Test function".to_string()],
            parameters: vec![FfiParameter {
                name: "param1".to_string(),
                type_info: FfiType {
                    name: "c_int".to_string(),
                    primitive_type: Some(PrimitiveType::Int32),
                    ..Default::default()
                },
                documentation: vec!["Parameter 1".to_string()],
                ..Default::default()
            }],
            return_type: FfiType {
                name: "c_int".to_string(),
                primitive_type: Some(PrimitiveType::Int32),
                ..Default::default()
            },
            ..Default::default()
        };

        let doc = generator.generate_function_docs(&func);
        assert_eq!(doc["name"], "test_function");
        assert_eq!(doc["kind"], TypeDocKind::Function as u32);
        assert!(doc["signatures"].is_array());
    }

    #[test]
    fn test_generate_enum_docs() {
        let config = CodeGenConfig::default();
        let generator = TypeDocGenerator::new(&config).unwrap();

        let enum_def = FfiEnum {
            name: "TestEnum".to_string(),
            c_name: "TestEnum".to_string(),
            documentation: vec!["Test enumeration".to_string()],
            variants: vec![
                FfiEnumVariant {
                    name: "Variant1".to_string(),
                    c_name: "TEST_ENUM_VARIANT1".to_string(),
                    value: 0,
                    documentation: vec!["First variant".to_string()],
                    deprecation: None,
                },
                FfiEnumVariant {
                    name: "Variant2".to_string(),
                    c_name: "TEST_ENUM_VARIANT2".to_string(),
                    value: 1,
                    documentation: vec!["Second variant".to_string()],
                    deprecation: None,
                },
            ],
            ..Default::default()
        };

        let doc = generator.generate_enum_docs(&enum_def);
        assert_eq!(doc["name"], "TestEnum");
        assert_eq!(doc["kind"], TypeDocKind::Enum as u32);
        assert!(doc["children"].is_array());
        assert_eq!(doc["children"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_generate_interface_docs() {
        let config = CodeGenConfig::default();
        let generator = TypeDocGenerator::new(&config).unwrap();

        let struct_def = FfiStruct {
            name: "TestStruct".to_string(),
            c_name: "TestStruct".to_string(),
            documentation: vec!["Test structure".to_string()],
            fields: vec![FfiField {
                name: "field1".to_string(),
                type_info: FfiType {
                    name: "c_int".to_string(),
                    primitive_type: Some(PrimitiveType::Int32),
                    ..Default::default()
                },
                documentation: vec!["Field 1".to_string()],
                is_private: false,
                offset: None,
                attributes: vec![],
            }],
            is_opaque: false,
            ..Default::default()
        };

        let doc = generator.generate_interface_docs(&struct_def);
        assert_eq!(doc["name"], "TestStruct");
        assert_eq!(doc["kind"], TypeDocKind::Interface as u32);
        assert!(doc["children"].is_array());
    }
}
