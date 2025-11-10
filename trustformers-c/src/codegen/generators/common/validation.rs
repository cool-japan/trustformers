//! Validation utilities for code generators

use crate::codegen::ast::{FfiField, FfiFunction, FfiParameter, FfiStruct, FfiType};
use crate::error::{TrustformersError, TrustformersResult};

/// Validate that a function definition is complete and consistent
pub fn validate_function(func: &FfiFunction) -> TrustformersResult<()> {
    // Check that function name is valid
    if func.name.is_empty() {
        return Err(TrustformersError::InvalidParameter);
    }

    if func.c_name.is_empty() {
        return Err(TrustformersError::InvalidParameter);
    }

    // Validate function name contains only valid characters
    if !is_valid_identifier(&func.name) {
        return Err(TrustformersError::InvalidParameter);
    }

    // Validate parameters
    for param in &func.parameters {
        validate_parameter(param)?;
    }

    // Validate return type
    validate_type(&func.return_type)?;

    Ok(())
}

/// Validate that a struct definition is complete and consistent
pub fn validate_struct(struct_def: &FfiStruct) -> TrustformersResult<()> {
    // Check that struct name is valid
    if struct_def.name.is_empty() {
        return Err(TrustformersError::InvalidParameter);
    }

    if !is_valid_identifier(&struct_def.name) {
        return Err(TrustformersError::InvalidParameter);
    }

    // Validate fields (unless it's an opaque struct)
    if !struct_def.is_opaque {
        if struct_def.fields.is_empty() {
            return Err(TrustformersError::InvalidParameter);
        }

        for field in &struct_def.fields {
            validate_field(field)?;
        }
    }

    Ok(())
}

/// Validate that a parameter definition is complete and consistent
pub fn validate_parameter(param: &FfiParameter) -> TrustformersResult<()> {
    if param.name.is_empty() {
        return Err(TrustformersError::InvalidParameter);
    }

    if !is_valid_identifier(&param.name) {
        return Err(TrustformersError::InvalidParameter);
    }

    validate_type(&param.type_info)?;

    Ok(())
}

/// Validate that a field definition is complete and consistent
pub fn validate_field(field: &FfiField) -> TrustformersResult<()> {
    if field.name.is_empty() {
        return Err(TrustformersError::InvalidParameter);
    }

    if !is_valid_identifier(&field.name) {
        return Err(TrustformersError::InvalidParameter);
    }

    validate_type(&field.type_info)?;

    Ok(())
}

/// Validate that a type definition is complete and consistent
pub fn validate_type(type_info: &FfiType) -> TrustformersResult<()> {
    if type_info.name.is_empty() {
        return Err(TrustformersError::InvalidParameter);
    }

    // Check for valid type names
    if !is_valid_type_name(&type_info.name) {
        return Err(TrustformersError::InvalidParameter);
    }

    Ok(())
}

/// Check if a string is a valid identifier (alphanumeric + underscore, starts with letter or underscore)
pub fn is_valid_identifier(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    // Must start with letter or underscore
    let first_char = name.chars().next().unwrap();
    if !first_char.is_alphabetic() && first_char != '_' {
        return false;
    }

    // Rest must be alphanumeric or underscore
    name.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Check if a string is a valid type name (allows some special characters for C types)
pub fn is_valid_type_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    // Allow basic type names and pointer syntax
    if name.starts_with('*') {
        // Pointer type - validate the base type
        let base_type = if name.starts_with("*const ") {
            &name[7..]
        } else if name.starts_with("*mut ") {
            &name[5..]
        } else {
            &name[1..]
        };
        return is_valid_type_name(base_type);
    }

    // Allow array syntax
    if name.contains('[') && name.ends_with(']') {
        // Parse array type
        if let Some(bracket_pos) = name.find('[') {
            let base_type = &name[..bracket_pos];
            return is_valid_type_name(base_type);
        }
    }

    // Allow generic types
    if name.contains('<') && name.ends_with('>') {
        if let Some(angle_pos) = name.find('<') {
            let base_type = &name[..angle_pos];
            return is_valid_identifier(base_type);
        }
    }

    // Regular identifier validation
    is_valid_identifier(name)
}

/// Check if a function name is reserved in common programming languages
pub fn is_reserved_keyword(name: &str, languages: &[crate::codegen::TargetLanguage]) -> bool {
    for language in languages {
        if is_reserved_in_language(name, language.clone()) {
            return true;
        }
    }
    false
}

/// Check if a name is reserved in a specific language
pub fn is_reserved_in_language(name: &str, language: crate::codegen::TargetLanguage) -> bool {
    use crate::codegen::TargetLanguage;

    let reserved_words: &[&str] = match language {
        TargetLanguage::Python => &[
            "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
            "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
            "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
            "return", "try", "while", "with", "yield",
        ],
        TargetLanguage::Java => &[
            "abstract",
            "assert",
            "boolean",
            "break",
            "byte",
            "case",
            "catch",
            "char",
            "class",
            "const",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extends",
            "final",
            "finally",
            "float",
            "for",
            "goto",
            "if",
            "implements",
            "import",
            "instanceof",
            "int",
            "interface",
            "long",
            "native",
            "new",
            "package",
            "private",
            "protected",
            "public",
            "return",
            "short",
            "static",
            "strictfp",
            "super",
            "switch",
            "synchronized",
            "this",
            "throw",
            "throws",
            "transient",
            "try",
            "void",
            "volatile",
            "while",
        ],
        TargetLanguage::JavaScript | TargetLanguage::TypeScript => &[
            "break",
            "case",
            "catch",
            "class",
            "const",
            "continue",
            "debugger",
            "default",
            "delete",
            "do",
            "else",
            "export",
            "extends",
            "finally",
            "for",
            "function",
            "if",
            "import",
            "in",
            "instanceof",
            "let",
            "new",
            "return",
            "super",
            "switch",
            "this",
            "throw",
            "try",
            "typeof",
            "var",
            "void",
            "while",
            "with",
            "yield",
        ],
        TargetLanguage::CSharp => &[
            "abstract",
            "as",
            "base",
            "bool",
            "break",
            "byte",
            "case",
            "catch",
            "char",
            "checked",
            "class",
            "const",
            "continue",
            "decimal",
            "default",
            "delegate",
            "do",
            "double",
            "else",
            "enum",
            "event",
            "explicit",
            "extern",
            "false",
            "finally",
            "fixed",
            "float",
            "for",
            "foreach",
            "goto",
            "if",
            "implicit",
            "in",
            "int",
            "interface",
            "internal",
            "is",
            "lock",
            "long",
            "namespace",
            "new",
            "null",
            "object",
            "operator",
            "out",
            "override",
            "params",
            "private",
            "protected",
            "public",
            "readonly",
            "ref",
            "return",
            "sbyte",
            "sealed",
            "short",
            "sizeof",
            "stackalloc",
            "static",
            "string",
            "struct",
            "switch",
            "this",
            "throw",
            "true",
            "try",
            "typeof",
            "uint",
            "ulong",
            "unchecked",
            "unsafe",
            "ushort",
            "using",
            "virtual",
            "void",
            "volatile",
            "while",
        ],
        TargetLanguage::Go => &[
            "break",
            "case",
            "chan",
            "const",
            "continue",
            "default",
            "defer",
            "else",
            "fallthrough",
            "for",
            "func",
            "go",
            "goto",
            "if",
            "import",
            "interface",
            "map",
            "package",
            "range",
            "return",
            "select",
            "struct",
            "switch",
            "type",
            "var",
        ],
        _ => &[], // Empty for other languages
    };

    reserved_words.contains(&name)
}

/// Suggest an alternative name if the given name is reserved
pub fn suggest_alternative_name(name: &str, language: crate::codegen::TargetLanguage) -> String {
    if is_reserved_in_language(name, language) {
        format!("{}_", name) // Simple approach: append underscore
    } else {
        name.to_string()
    }
}
