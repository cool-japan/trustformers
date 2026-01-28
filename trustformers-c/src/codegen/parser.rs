//! FFI interface parser for extracting binding information from Rust source
//!
//! Parses Rust source files to extract FFI function definitions, structs, enums,
//! and other interface elements that can be used to generate language bindings.

use anyhow::anyhow;
use quote::ToTokens;
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use syn::{File, ForeignItemFn, Item, ItemConst, ItemEnum, ItemFn, ItemStruct, ItemType};

use super::ast::*;
use crate::error::TrustformersResult;

/// Parser for extracting FFI interface from Rust source code
pub struct FfiParser {
    /// Regular expressions for parsing comments and attributes
    regex_cache: RegexCache,
    /// Configuration for parsing
    config: ParserConfig,
}

/// Cached regular expressions for parsing
struct RegexCache {
    /// Matches `#[no_mangle]` attribute
    no_mangle: Regex,
    /// Matches `pub extern "C"` functions
    extern_c: Regex,
    /// Matches documentation comments
    doc_comment: Regex,
    /// Matches feature attributes
    feature_attr: Regex,
    /// Matches deprecation attributes
    deprecated_attr: Regex,
    /// Matches repr(C) attributes
    repr_c: Regex,
    /// Matches cbindgen attributes
    cbindgen_attr: Regex,
}

/// Configuration for the parser
#[derive(Debug, Clone)]
pub struct ParserConfig {
    /// Function name prefix to include (e.g., "trustformers_")
    pub function_prefix: String,
    /// Whether to include private items
    pub include_private: bool,
    /// Whether to parse documentation
    pub parse_documentation: bool,
    /// Whether to extract feature requirements
    pub extract_features: bool,
    /// List of files to exclude from parsing
    pub exclude_files: Vec<String>,
    /// List of functions to exclude
    pub exclude_functions: Vec<String>,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            function_prefix: "trustformers_".to_string(),
            include_private: false,
            parse_documentation: true,
            extract_features: true,
            exclude_files: vec!["tests".to_string(), "benches".to_string()],
            exclude_functions: Vec::new(),
        }
    }
}

impl Default for FfiParser {
    fn default() -> Self {
        Self::new()
    }
}

impl FfiParser {
    /// Create a new FFI parser with default configuration
    pub fn new() -> Self {
        Self::with_config(ParserConfig::default())
    }

    /// Create a new FFI parser with custom configuration
    pub fn with_config(config: ParserConfig) -> Self {
        Self {
            regex_cache: RegexCache::new(),
            config,
        }
    }

    /// Parse FFI interface from a directory containing Rust source files
    pub fn parse_directory(&self, dir: &Path) -> TrustformersResult<FfiInterface> {
        let mut interface = FfiInterface::default();

        // Set basic metadata
        interface.metadata.library_name = "trustformers".to_string();
        interface.metadata.version = self.extract_version_from_cargo_toml(dir)?;

        // Find all Rust source files
        let rust_files = self.find_rust_files(dir)?;

        // Parse each file
        for file_path in rust_files {
            if self.should_exclude_file(&file_path) {
                continue;
            }

            let file_interface = self.parse_file(&file_path)?;
            self.merge_interface(&mut interface, file_interface);
        }

        // Post-process the interface
        self.post_process_interface(&mut interface)?;

        Ok(interface)
    }

    /// Parse FFI interface from a single Rust source file
    pub fn parse_file(&self, file_path: &Path) -> TrustformersResult<FfiInterface> {
        let content = fs::read_to_string(file_path)?;
        self.parse_source(&content, file_path)
    }

    /// Parse FFI interface from Rust source code
    pub fn parse_source(&self, source: &str, file_path: &Path) -> TrustformersResult<FfiInterface> {
        let syntax_tree: File = syn::parse_str(source)
            .map_err(|e| anyhow!("Failed to parse {}: {}", file_path.display(), e))?;

        let mut interface = FfiInterface::default();

        // Extract file-level attributes and documentation
        if self.config.parse_documentation {
            interface
                .metadata
                .required_features
                .extend(self.extract_file_features(&syntax_tree.attrs)?);
        }

        // Parse all items in the file
        for item in &syntax_tree.items {
            match item {
                Item::Fn(func) => {
                    if let Some(ffi_func) = self.parse_function(func)? {
                        interface.functions.push(ffi_func);
                    }
                },
                Item::Struct(struct_item) => {
                    if let Some(ffi_struct) = self.parse_struct(struct_item)? {
                        interface.structs.push(ffi_struct);
                    }
                },
                Item::Enum(enum_item) => {
                    if let Some(ffi_enum) = self.parse_enum(enum_item)? {
                        interface.enums.push(ffi_enum);
                    }
                },
                Item::Const(const_item) => {
                    if let Some(ffi_const) = self.parse_constant(const_item)? {
                        interface.constants.push(ffi_const);
                    }
                },
                Item::Type(type_item) => {
                    if let Some(ffi_type_alias) = self.parse_type_alias(type_item)? {
                        interface.type_aliases.push(ffi_type_alias);
                    }
                },
                Item::ForeignMod(foreign_mod) => {
                    // Parse foreign function declarations
                    for foreign_item in &foreign_mod.items {
                        if let syn::ForeignItem::Fn(foreign_fn) = foreign_item {
                            if let Some(ffi_func) = self.parse_foreign_function(foreign_fn)? {
                                interface.functions.push(ffi_func);
                            }
                        }
                    }
                },
                _ => {}, // Ignore other items
            }
        }

        Ok(interface)
    }

    // Private helper methods

    fn find_rust_files(&self, dir: &Path) -> TrustformersResult<Vec<std::path::PathBuf>> {
        let mut rust_files = Vec::new();

        fn visit_dir(
            dir: &Path,
            rust_files: &mut Vec<std::path::PathBuf>,
        ) -> TrustformersResult<()> {
            if dir.is_dir() {
                for entry in fs::read_dir(dir)? {
                    let entry = entry?;
                    let path = entry.path();

                    if path.is_dir() {
                        // Skip target and hidden directories
                        if let Some(name) = path.file_name() {
                            if name == "target" || name.to_string_lossy().starts_with('.') {
                                continue;
                            }
                        }
                        visit_dir(&path, rust_files)?;
                    } else if path.extension().is_some_and(|ext| ext == "rs") {
                        rust_files.push(path);
                    }
                }
            }
            Ok(())
        }

        visit_dir(dir, &mut rust_files)?;
        Ok(rust_files)
    }

    fn should_exclude_file(&self, file_path: &Path) -> bool {
        let file_str = file_path.to_string_lossy();
        self.config.exclude_files.iter().any(|exclude| file_str.contains(exclude))
    }

    fn parse_function(&self, func: &ItemFn) -> TrustformersResult<Option<FfiFunction>> {
        // Check if this is an FFI function
        if !self.is_ffi_function(func) {
            return Ok(None);
        }

        let func_name = func.sig.ident.to_string();

        // Check exclusion list
        if self.config.exclude_functions.contains(&func_name) {
            return Ok(None);
        }

        let mut ffi_func = FfiFunction {
            name: self.strip_function_prefix(&func_name),
            c_name: func_name.clone(),
            is_unsafe: func.sig.unsafety.is_some(),
            ..Default::default()
        };

        // Parse documentation
        if self.config.parse_documentation {
            ffi_func.documentation = self.extract_documentation(&func.attrs)?;
        }

        // Parse parameters
        for input in &func.sig.inputs {
            if let syn::FnArg::Typed(pat_type) = input {
                let param = self.parse_parameter(pat_type)?;
                ffi_func.parameters.push(param);
            }
        }

        // Parse return type
        ffi_func.return_type = self.parse_return_type(&func.sig.output)?;
        ffi_func.can_fail = ffi_func.return_type.is_error_type();

        // Parse attributes
        ffi_func.required_features = self.extract_function_features(&func.attrs)?;
        ffi_func.deprecation = self.extract_deprecation(&func.attrs)?;
        ffi_func.attributes = self.extract_attributes(&func.attrs)?;

        Ok(Some(ffi_func))
    }

    fn is_ffi_function(&self, func: &ItemFn) -> bool {
        // Check for #[no_mangle] and pub extern "C"
        let has_no_mangle = func.attrs.iter().any(|attr| attr.path().is_ident("no_mangle"));

        let is_extern_c = matches!(func.sig.abi, Some(ref abi) if abi.name.as_ref().map(|lit| lit.value()) == Some("C".to_string()));

        let is_public = matches!(func.vis, syn::Visibility::Public(_));

        has_no_mangle && is_extern_c && is_public
    }

    fn strip_function_prefix(&self, name: &str) -> String {
        if name.starts_with(&self.config.function_prefix) {
            name[self.config.function_prefix.len()..].to_string()
        } else {
            name.to_string()
        }
    }

    fn parse_parameter(&self, pat_type: &syn::PatType) -> TrustformersResult<FfiParameter> {
        let name = match &*pat_type.pat {
            syn::Pat::Ident(ident) => ident.ident.to_string(),
            _ => "unnamed".to_string(),
        };

        let type_info = self.parse_type(&pat_type.ty)?;

        Ok(FfiParameter {
            name,
            type_info,
            documentation: Vec::new(),
            is_optional: false,
            default_value: None,
            attributes: Vec::new(),
        })
    }

    fn parse_return_type(&self, output: &syn::ReturnType) -> TrustformersResult<FfiType> {
        match output {
            syn::ReturnType::Default => Ok(FfiType {
                name: "void".to_string(),
                primitive_type: Some(PrimitiveType::Void),
                ..Default::default()
            }),
            syn::ReturnType::Type(_, ty) => self.parse_type(ty),
        }
    }

    fn parse_type(&self, ty: &syn::Type) -> TrustformersResult<FfiType> {
        match ty {
            syn::Type::Path(type_path) => {
                let type_name = type_path
                    .path
                    .segments
                    .last()
                    .map(|seg| seg.ident.to_string())
                    .unwrap_or_else(|| "unknown".to_string());

                let primitive = self.map_to_primitive_type(&type_name);

                Ok(FfiType {
                    name: type_name,
                    primitive_type: primitive,
                    ..Default::default()
                })
            },
            syn::Type::Ptr(type_ptr) => {
                let inner_type = self.parse_type(&type_ptr.elem)?;
                Ok(FfiType {
                    name: format!(
                        "*{}{}",
                        if type_ptr.const_token.is_some() { "const " } else { "mut " },
                        inner_type.name
                    ),
                    is_pointer: true,
                    is_const: type_ptr.const_token.is_some(),
                    is_mutable: type_ptr.mutability.is_some(),
                    pointer_level: 1,
                    primitive_type: inner_type.primitive_type,
                    ..Default::default()
                })
            },
            syn::Type::Reference(type_ref) => {
                let inner_type = self.parse_type(&type_ref.elem)?;
                Ok(FfiType {
                    name: format!(
                        "&{}{}",
                        if type_ref.mutability.is_some() { "mut " } else { "" },
                        inner_type.name
                    ),
                    is_pointer: true,
                    is_const: type_ref.mutability.is_none(),
                    is_mutable: type_ref.mutability.is_some(),
                    pointer_level: 1,
                    primitive_type: inner_type.primitive_type,
                    ..Default::default()
                })
            },
            syn::Type::Array(type_array) => {
                let inner_type = self.parse_type(&type_array.elem)?;
                // Try to extract array size
                let array_size = if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Int(int_lit),
                    ..
                }) = &type_array.len
                {
                    int_lit.base10_parse().ok()
                } else {
                    None
                };

                Ok(FfiType {
                    name: format!(
                        "[{}; {}]",
                        inner_type.name,
                        type_array.len.to_token_stream()
                    ),
                    array_size,
                    primitive_type: inner_type.primitive_type,
                    ..Default::default()
                })
            },
            _ => {
                // Fallback for unknown types
                Ok(FfiType {
                    name: ty.to_token_stream().to_string(),
                    ..Default::default()
                })
            },
        }
    }

    fn map_to_primitive_type(&self, type_name: &str) -> Option<PrimitiveType> {
        match type_name {
            "i8" | "c_char" => Some(PrimitiveType::Int8),
            "i16" | "c_short" => Some(PrimitiveType::Int16),
            "i32" | "c_int" => Some(PrimitiveType::Int32),
            "i64" | "c_long" | "c_longlong" => Some(PrimitiveType::Int64),
            "u8" | "c_uchar" => Some(PrimitiveType::UInt8),
            "u16" | "c_ushort" => Some(PrimitiveType::UInt16),
            "u32" | "c_uint" => Some(PrimitiveType::UInt32),
            "u64" | "c_ulong" | "c_ulonglong" => Some(PrimitiveType::UInt64),
            "isize" => Some(PrimitiveType::IntPtr),
            "usize" => Some(PrimitiveType::UIntPtr),
            "f32" | "c_float" => Some(PrimitiveType::Float32),
            "f64" | "c_double" => Some(PrimitiveType::Float64),
            "bool" | "c_bool" => Some(PrimitiveType::Bool),
            "c_void" | "()" => Some(PrimitiveType::Void),
            "CString" | "CStr" => Some(PrimitiveType::CString),
            name if name.ends_with("Handle") => Some(PrimitiveType::Handle),
            _ => None,
        }
    }

    fn parse_struct(&self, struct_item: &ItemStruct) -> TrustformersResult<Option<FfiStruct>> {
        // Check if this is a C-compatible struct
        if !self.is_c_struct(struct_item) {
            return Ok(None);
        }

        let struct_name = struct_item.ident.to_string();

        let mut ffi_struct = FfiStruct {
            name: struct_name.clone(),
            c_name: struct_name,
            is_opaque: matches!(struct_item.fields, syn::Fields::Unit),
            ..Default::default()
        };

        // Parse documentation
        if self.config.parse_documentation {
            ffi_struct.documentation = self.extract_documentation(&struct_item.attrs)?;
        }

        // Parse fields
        match &struct_item.fields {
            syn::Fields::Named(fields) => {
                for field in &fields.named {
                    let ffi_field = self.parse_struct_field(field)?;
                    ffi_struct.fields.push(ffi_field);
                }
            },
            syn::Fields::Unnamed(fields) => {
                for (i, field) in fields.unnamed.iter().enumerate() {
                    let mut ffi_field = self.parse_struct_field(field)?;
                    ffi_field.name = format!("field_{}", i);
                    ffi_struct.fields.push(ffi_field);
                }
            },
            syn::Fields::Unit => {
                // Unit struct - mark as opaque
                ffi_struct.is_opaque = true;
            },
        }

        // Parse attributes
        ffi_struct.required_features = self.extract_struct_features(&struct_item.attrs)?;
        ffi_struct.deprecation = self.extract_deprecation(&struct_item.attrs)?;
        ffi_struct.is_packed = self.extract_packed_attribute(&struct_item.attrs)?;

        Ok(Some(ffi_struct))
    }

    fn is_c_struct(&self, struct_item: &ItemStruct) -> bool {
        // Check for #[repr(C)] attribute
        struct_item.attrs.iter().any(|attr| {
            if attr.path().is_ident("repr") {
                if let syn::Meta::List(meta_list) = &attr.meta {
                    return meta_list.tokens.to_string().contains("C");
                }
            }
            false
        })
    }

    fn parse_struct_field(&self, field: &syn::Field) -> TrustformersResult<FfiField> {
        let name = field
            .ident
            .as_ref()
            .map(|i| i.to_string())
            .unwrap_or_else(|| "unnamed".to_string());

        let type_info = self.parse_type(&field.ty)?;
        let is_private = !matches!(field.vis, syn::Visibility::Public(_));

        Ok(FfiField {
            name,
            type_info,
            documentation: if self.config.parse_documentation {
                self.extract_documentation(&field.attrs)?
            } else {
                Vec::new()
            },
            offset: None,
            is_private,
            attributes: self.extract_attributes(&field.attrs)?,
        })
    }

    fn parse_enum(&self, enum_item: &ItemEnum) -> TrustformersResult<Option<FfiEnum>> {
        // Check if this is a C-compatible enum
        if !self.is_c_enum(enum_item) {
            return Ok(None);
        }

        let enum_name = enum_item.ident.to_string();

        let mut ffi_enum = FfiEnum {
            name: enum_name.clone(),
            c_name: enum_name,
            underlying_type: PrimitiveType::Int32, // Default to int
            ..Default::default()
        };

        // Parse documentation
        if self.config.parse_documentation {
            ffi_enum.documentation = self.extract_documentation(&enum_item.attrs)?;
        }

        // Parse variants
        for (i, variant) in enum_item.variants.iter().enumerate() {
            let variant_name = variant.ident.to_string();

            // Try to extract explicit discriminant value
            let value = if let Some((_, expr)) = &variant.discriminant {
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Int(int_lit),
                    ..
                }) = expr
                {
                    int_lit.base10_parse().unwrap_or(i as i64)
                } else {
                    i as i64
                }
            } else {
                i as i64
            };

            let ffi_variant = FfiEnumVariant {
                name: variant_name.clone(),
                c_name: format!("{}_{}", ffi_enum.c_name, variant_name),
                value,
                documentation: if self.config.parse_documentation {
                    self.extract_documentation(&variant.attrs)?
                } else {
                    Vec::new()
                },
                deprecation: self.extract_deprecation(&variant.attrs)?,
            };

            ffi_enum.variants.push(ffi_variant);
        }

        // Parse attributes
        ffi_enum.required_features = self.extract_enum_features(&enum_item.attrs)?;
        ffi_enum.deprecation = self.extract_deprecation(&enum_item.attrs)?;
        ffi_enum.is_flags = self.extract_flags_attribute(&enum_item.attrs)?;

        Ok(Some(ffi_enum))
    }

    fn is_c_enum(&self, enum_item: &ItemEnum) -> bool {
        // Check for #[repr(C)] or #[repr(u32)] etc.
        enum_item.attrs.iter().any(|attr| {
            if attr.path().is_ident("repr") {
                if let syn::Meta::List(meta_list) = &attr.meta {
                    let tokens = meta_list.tokens.to_string();
                    return tokens.contains("C")
                        || tokens.contains("u8")
                        || tokens.contains("u16")
                        || tokens.contains("u32")
                        || tokens.contains("u64")
                        || tokens.contains("i8")
                        || tokens.contains("i16")
                        || tokens.contains("i32")
                        || tokens.contains("i64");
                }
            }
            false
        })
    }

    fn parse_constant(&self, const_item: &ItemConst) -> TrustformersResult<Option<FfiConstant>> {
        // Only include public constants
        if !matches!(const_item.vis, syn::Visibility::Public(_)) {
            return Ok(None);
        }

        let const_name = const_item.ident.to_string();
        let type_info = self.parse_type(&const_item.ty)?;

        // Try to extract the constant value
        let value = self.extract_constant_value(&const_item.expr)?;

        let ffi_const = FfiConstant {
            name: const_name.clone(),
            c_name: const_name,
            type_info,
            value,
            documentation: if self.config.parse_documentation {
                self.extract_documentation(&const_item.attrs)?
            } else {
                Vec::new()
            },
            required_features: self.extract_const_features(&const_item.attrs)?,
            platforms: Vec::new(),
            deprecation: self.extract_deprecation(&const_item.attrs)?,
        };

        Ok(Some(ffi_const))
    }

    fn parse_type_alias(&self, type_item: &ItemType) -> TrustformersResult<Option<FfiTypeAlias>> {
        // Only include public type aliases
        if !matches!(type_item.vis, syn::Visibility::Public(_)) {
            return Ok(None);
        }

        let alias_name = type_item.ident.to_string();
        let target_type = self.parse_type(&type_item.ty)?;

        let ffi_type_alias = FfiTypeAlias {
            name: alias_name.clone(),
            c_name: alias_name,
            target_type,
            documentation: if self.config.parse_documentation {
                self.extract_documentation(&type_item.attrs)?
            } else {
                Vec::new()
            },
            required_features: self.extract_type_features(&type_item.attrs)?,
            platforms: Vec::new(),
            deprecation: self.extract_deprecation(&type_item.attrs)?,
        };

        Ok(Some(ffi_type_alias))
    }

    fn parse_foreign_function(
        &self,
        foreign_fn: &ForeignItemFn,
    ) -> TrustformersResult<Option<FfiFunction>> {
        // Parse foreign function declarations (from extern blocks)
        let func_name = foreign_fn.sig.ident.to_string();

        if self.config.exclude_functions.contains(&func_name) {
            return Ok(None);
        }

        let mut ffi_func = FfiFunction {
            name: self.strip_function_prefix(&func_name),
            c_name: func_name.clone(),
            is_unsafe: true, // Foreign functions are inherently unsafe
            ..Default::default()
        };

        // Parse documentation
        if self.config.parse_documentation {
            ffi_func.documentation = self.extract_documentation(&foreign_fn.attrs)?;
        }

        // Parse parameters
        for input in &foreign_fn.sig.inputs {
            if let syn::FnArg::Typed(pat_type) = input {
                let param = self.parse_parameter(pat_type)?;
                ffi_func.parameters.push(param);
            }
        }

        // Parse return type
        ffi_func.return_type = self.parse_return_type(&foreign_fn.sig.output)?;
        ffi_func.can_fail = ffi_func.return_type.is_error_type();

        // Parse attributes
        ffi_func.required_features = self.extract_function_features(&foreign_fn.attrs)?;
        ffi_func.deprecation = self.extract_deprecation(&foreign_fn.attrs)?;
        ffi_func.attributes = self.extract_attributes(&foreign_fn.attrs)?;

        Ok(Some(ffi_func))
    }

    // Attribute extraction methods

    fn extract_documentation(&self, attrs: &[syn::Attribute]) -> TrustformersResult<Vec<String>> {
        let mut docs = Vec::new();

        for attr in attrs {
            if attr.path().is_ident("doc") {
                if let syn::Meta::NameValue(meta_name_value) = &attr.meta {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(lit_str),
                        ..
                    }) = &meta_name_value.value
                    {
                        let doc_line = lit_str.value();
                        docs.push(doc_line.trim().to_string());
                    }
                }
            }
        }

        Ok(docs)
    }

    fn extract_function_features(
        &self,
        attrs: &[syn::Attribute],
    ) -> TrustformersResult<Vec<String>> {
        self.extract_features_from_attrs(attrs)
    }

    fn extract_struct_features(&self, attrs: &[syn::Attribute]) -> TrustformersResult<Vec<String>> {
        self.extract_features_from_attrs(attrs)
    }

    fn extract_enum_features(&self, attrs: &[syn::Attribute]) -> TrustformersResult<Vec<String>> {
        self.extract_features_from_attrs(attrs)
    }

    fn extract_const_features(&self, attrs: &[syn::Attribute]) -> TrustformersResult<Vec<String>> {
        self.extract_features_from_attrs(attrs)
    }

    fn extract_type_features(&self, attrs: &[syn::Attribute]) -> TrustformersResult<Vec<String>> {
        self.extract_features_from_attrs(attrs)
    }

    fn extract_file_features(&self, attrs: &[syn::Attribute]) -> TrustformersResult<Vec<String>> {
        self.extract_features_from_attrs(attrs)
    }

    fn extract_features_from_attrs(
        &self,
        attrs: &[syn::Attribute],
    ) -> TrustformersResult<Vec<String>> {
        let mut features = Vec::new();

        for attr in attrs {
            if attr.path().is_ident("cfg") {
                if let syn::Meta::List(meta_list) = &attr.meta {
                    let cfg_str = meta_list.tokens.to_string();
                    if cfg_str.contains("feature") {
                        // Extract feature name from cfg(feature = "name")
                        if let Some(start) = cfg_str.find('"') {
                            if let Some(end) = cfg_str[start + 1..].find('"') {
                                let feature_name = &cfg_str[start + 1..start + 1 + end];
                                features.push(feature_name.to_string());
                            }
                        }
                    }
                }
            }
        }

        Ok(features)
    }

    fn extract_deprecation(
        &self,
        attrs: &[syn::Attribute],
    ) -> TrustformersResult<Option<DeprecationInfo>> {
        for attr in attrs {
            if attr.path().is_ident("deprecated") {
                let mut deprecation = DeprecationInfo {
                    message: "This item is deprecated".to_string(),
                    since_version: None,
                    replacement: None,
                    removal_version: None,
                };

                if let syn::Meta::List(meta_list) = &attr.meta {
                    let tokens_str = meta_list.tokens.to_string();

                    // Parse since = "version"
                    if let Some(since_start) = tokens_str.find("since = \"") {
                        let since_content = &tokens_str[since_start + 9..];
                        if let Some(since_end) = since_content.find('"') {
                            deprecation.since_version =
                                Some(since_content[..since_end].to_string());
                        }
                    }

                    // Parse note = "message"
                    if let Some(note_start) = tokens_str.find("note = \"") {
                        let note_content = &tokens_str[note_start + 8..];
                        if let Some(note_end) = note_content.find('"') {
                            deprecation.message = note_content[..note_end].to_string();
                        }
                    }
                }

                return Ok(Some(deprecation));
            }
        }

        Ok(None)
    }

    fn extract_attributes(
        &self,
        attrs: &[syn::Attribute],
    ) -> TrustformersResult<Vec<FfiAttribute>> {
        let mut ffi_attrs = Vec::new();

        for attr in attrs {
            let attr_name = attr
                .path()
                .segments
                .last()
                .map(|seg| seg.ident.to_string())
                .unwrap_or_else(|| "unknown".to_string());

            let value = match &attr.meta {
                syn::Meta::NameValue(name_value) => {
                    Some(name_value.value.to_token_stream().to_string())
                },
                syn::Meta::List(list) => Some(list.tokens.to_string()),
                syn::Meta::Path(_) => None,
            };

            ffi_attrs.push(FfiAttribute {
                name: attr_name,
                value,
            });
        }

        Ok(ffi_attrs)
    }

    fn extract_packed_attribute(&self, attrs: &[syn::Attribute]) -> TrustformersResult<bool> {
        for attr in attrs {
            if attr.path().is_ident("repr") {
                if let syn::Meta::List(meta_list) = &attr.meta {
                    if meta_list.tokens.to_string().contains("packed") {
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }

    fn extract_flags_attribute(&self, attrs: &[syn::Attribute]) -> TrustformersResult<bool> {
        // Check for bitflags! macro or similar patterns
        for attr in attrs {
            if attr.path().is_ident("bitflags") {
                return Ok(true);
            }
            // Could also check for specific derive attributes like BitOr, BitAnd, etc.
        }
        Ok(false)
    }

    fn extract_constant_value(&self, expr: &syn::Expr) -> TrustformersResult<ConstantValue> {
        match expr {
            syn::Expr::Lit(syn::ExprLit { lit, .. }) => match lit {
                syn::Lit::Int(int_lit) => Ok(ConstantValue::Integer(int_lit.base10_parse()?)),
                syn::Lit::Float(float_lit) => Ok(ConstantValue::Float(float_lit.base10_parse()?)),
                syn::Lit::Str(str_lit) => Ok(ConstantValue::String(str_lit.value())),
                syn::Lit::Bool(bool_lit) => Ok(ConstantValue::Boolean(bool_lit.value)),
                _ => Ok(ConstantValue::String(lit.to_token_stream().to_string())),
            },
            _ => {
                // For complex expressions, just store as string
                Ok(ConstantValue::String(expr.to_token_stream().to_string()))
            },
        }
    }

    fn extract_version_from_cargo_toml(&self, dir: &Path) -> TrustformersResult<String> {
        let cargo_toml_path = dir.join("Cargo.toml");
        if cargo_toml_path.exists() {
            let content = fs::read_to_string(cargo_toml_path)?;

            // Simple regex to extract version
            let version_regex = Regex::new(r#"version\s*=\s*"([^"]+)""#)?;
            if let Some(captures) = version_regex.captures(&content) {
                if let Some(version) = captures.get(1) {
                    return Ok(version.as_str().to_string());
                }
            }
        }

        Ok("0.1.0".to_string()) // Default version
    }

    fn merge_interface(&self, target: &mut FfiInterface, source: FfiInterface) {
        target.functions.extend(source.functions);
        target.structs.extend(source.structs);
        target.enums.extend(source.enums);
        target.constants.extend(source.constants);
        target.type_aliases.extend(source.type_aliases);

        // Merge metadata
        target.metadata.required_features.extend(source.metadata.required_features);
        target.metadata.optional_features.extend(source.metadata.optional_features);
    }

    fn post_process_interface(&self, interface: &mut FfiInterface) -> TrustformersResult<()> {
        // Remove duplicates
        interface.functions.sort_by(|a, b| a.name.cmp(&b.name));
        interface.functions.dedup_by(|a, b| a.name == b.name);

        interface.structs.sort_by(|a, b| a.name.cmp(&b.name));
        interface.structs.dedup_by(|a, b| a.name == b.name);

        interface.enums.sort_by(|a, b| a.name.cmp(&b.name));
        interface.enums.dedup_by(|a, b| a.name == b.name);

        // Remove duplicate features
        interface.metadata.required_features.sort();
        interface.metadata.required_features.dedup();

        interface.metadata.optional_features.sort();
        interface.metadata.optional_features.dedup();

        Ok(())
    }
}

impl RegexCache {
    fn new() -> Self {
        Self {
            no_mangle: Regex::new(r"#\[no_mangle\]").unwrap(),
            extern_c: Regex::new(r#"pub extern "C""#).unwrap(),
            doc_comment: Regex::new(r"///\s*(.*)").unwrap(),
            feature_attr: Regex::new(r#"#\[cfg\(feature = "([^"]+)"\)\]"#).unwrap(),
            deprecated_attr: Regex::new(r"#\[deprecated").unwrap(),
            repr_c: Regex::new(r"#\[repr\(C\)\]").unwrap(),
            cbindgen_attr: Regex::new(r"#\[cbindgen::").unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_parse_simple_function() {
        let source = r#"
            #[no_mangle]
            pub extern "C" fn trustformers_test_function(param: c_int) -> c_int {
                param + 1
            }
        "#;

        let parser = FfiParser::new();
        let interface = parser.parse_source(source, Path::new("test.rs")).unwrap();

        assert_eq!(interface.functions.len(), 1);
        assert_eq!(interface.functions[0].name, "test_function");
        assert_eq!(interface.functions[0].parameters.len(), 1);
        assert_eq!(interface.functions[0].parameters[0].name, "param");
    }

    #[test]
    fn test_parse_struct() {
        let source = r#"
            #[repr(C)]
            pub struct TestStruct {
                pub field1: c_int,
                pub field2: *const c_char,
            }
        "#;

        let parser = FfiParser::new();
        let interface = parser.parse_source(source, Path::new("test.rs")).unwrap();

        assert_eq!(interface.structs.len(), 1);
        assert_eq!(interface.structs[0].name, "TestStruct");
        assert_eq!(interface.structs[0].fields.len(), 2);
        assert_eq!(interface.structs[0].fields[0].name, "field1");
        assert_eq!(interface.structs[0].fields[1].name, "field2");
    }

    #[test]
    fn test_parse_enum() {
        let source = r#"
            #[repr(C)]
            pub enum TestEnum {
                Variant1 = 0,
                Variant2 = 1,
                Variant3 = 2,
            }
        "#;

        let parser = FfiParser::new();
        let interface = parser.parse_source(source, Path::new("test.rs")).unwrap();

        assert_eq!(interface.enums.len(), 1);
        assert_eq!(interface.enums[0].name, "TestEnum");
        assert_eq!(interface.enums[0].variants.len(), 3);
        assert_eq!(interface.enums[0].variants[0].name, "Variant1");
        assert_eq!(interface.enums[0].variants[0].value, 0);
    }
}
