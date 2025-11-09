//! Template helper utilities for code generators

use crate::codegen::ast::{FfiFunction, FfiParameter, FfiStruct};
use crate::codegen::TargetLanguage;

/// Generate function parameter list for a target language
pub fn generate_parameter_list(
    parameters: &[FfiParameter],
    language: TargetLanguage,
    include_types: bool,
) -> String {
    let mut params = Vec::new();

    for param in parameters {
        if include_types {
            match language {
                TargetLanguage::Python => {
                    // Python doesn't require type annotations in parameter list for basic bindings
                    params.push(param.name.clone());
                },
                TargetLanguage::TypeScript => {
                    params.push(format!("{}: {}", param.name, param.type_info.name));
                },
                TargetLanguage::Java => {
                    params.push(format!("{} {}", param.type_info.name, param.name));
                },
                TargetLanguage::CSharp => {
                    params.push(format!("{} {}", param.type_info.name, param.name));
                },
                TargetLanguage::Go => {
                    params.push(format!("{} {}", param.name, param.type_info.name));
                },
                TargetLanguage::Swift => {
                    params.push(format!("{}: {}", param.name, param.type_info.name));
                },
                TargetLanguage::Kotlin => {
                    params.push(format!("{}: {}", param.name, param.type_info.name));
                },
                _ => {
                    params.push(format!("{} {}", param.type_info.name, param.name));
                },
            }
        } else {
            params.push(param.name.clone());
        }
    }

    params.join(", ")
}

/// Generate function call with proper syntax for target language
pub fn generate_function_call(
    function_name: &str,
    parameters: &[FfiParameter],
    language: TargetLanguage,
) -> String {
    let param_names: Vec<String> = parameters.iter().map(|p| p.name.clone()).collect();

    match language {
        TargetLanguage::Python => {
            format!("{}({})", function_name, param_names.join(", "))
        },
        TargetLanguage::Java | TargetLanguage::CSharp => {
            format!("{}({})", function_name, param_names.join(", "))
        },
        TargetLanguage::JavaScript | TargetLanguage::TypeScript => {
            format!("{}({})", function_name, param_names.join(", "))
        },
        TargetLanguage::Go => {
            format!("{}({})", function_name, param_names.join(", "))
        },
        TargetLanguage::Swift => {
            format!("{}({})", function_name, param_names.join(", "))
        },
        TargetLanguage::Kotlin => {
            format!("{}({})", function_name, param_names.join(", "))
        },
        TargetLanguage::Ruby => {
            format!("{}({})", function_name, param_names.join(", "))
        },
        TargetLanguage::Php => {
            format!("{}({})", function_name, param_names.join(", "))
        },
        _ => {
            format!("{}({})", function_name, param_names.join(", "))
        },
    }
}

/// Generate class/struct declaration syntax for target language
pub fn generate_class_declaration(
    class_name: &str,
    language: TargetLanguage,
    base_class: Option<&str>,
) -> String {
    match language {
        TargetLanguage::Python => {
            if let Some(base) = base_class {
                format!("class {}({}):", class_name, base)
            } else {
                format!("class {}:", class_name)
            }
        },
        TargetLanguage::Java => {
            if let Some(base) = base_class {
                format!("public class {} extends {} {{", class_name, base)
            } else {
                format!("public class {} {{", class_name)
            }
        },
        TargetLanguage::CSharp => {
            if let Some(base) = base_class {
                format!("public class {} : {} {{", class_name, base)
            } else {
                format!("public class {} {{", class_name)
            }
        },
        TargetLanguage::TypeScript => {
            if let Some(base) = base_class {
                format!("export class {} extends {} {{", class_name, base)
            } else {
                format!("export class {} {{", class_name)
            }
        },
        TargetLanguage::JavaScript => {
            if let Some(base) = base_class {
                format!("class {} extends {} {{", class_name, base)
            } else {
                format!("class {} {{", class_name)
            }
        },
        TargetLanguage::Go => {
            format!("type {} struct {{", class_name)
        },
        TargetLanguage::Swift => {
            if let Some(base) = base_class {
                format!("class {}: {} {{", class_name, base)
            } else {
                format!("class {} {{", class_name)
            }
        },
        TargetLanguage::Kotlin => {
            if let Some(base) = base_class {
                format!("class {} : {} {{", class_name, base)
            } else {
                format!("class {} {{", class_name)
            }
        },
        TargetLanguage::Ruby => {
            if let Some(base) = base_class {
                format!("class {} < {}", class_name, base)
            } else {
                format!("class {}", class_name)
            }
        },
        TargetLanguage::Php => {
            if let Some(base) = base_class {
                format!("class {} extends {} {{", class_name, base)
            } else {
                format!("class {} {{", class_name)
            }
        },
        _ => {
            format!("class {} {{", class_name)
        },
    }
}

/// Generate method/function declaration syntax for target language
pub fn generate_method_declaration(
    method_name: &str,
    parameters: &[FfiParameter],
    return_type: Option<&str>,
    language: TargetLanguage,
    is_static: bool,
) -> String {
    let param_list = generate_parameter_list(parameters, language, true);

    match language {
        TargetLanguage::Python => {
            if is_static {
                format!("@staticmethod\ndef {}({}):", method_name, param_list)
            } else {
                let params = if param_list.is_empty() {
                    "self".to_string()
                } else {
                    format!("self, {}", param_list)
                };
                format!("def {}({}):", method_name, params)
            }
        },
        TargetLanguage::Java => {
            let static_keyword = if is_static { "static " } else { "" };
            let ret_type = return_type.unwrap_or("void");
            format!(
                "public {}{} {}({})",
                static_keyword, ret_type, method_name, param_list
            )
        },
        TargetLanguage::CSharp => {
            let static_keyword = if is_static { "static " } else { "" };
            let ret_type = return_type.unwrap_or("void");
            format!(
                "public {}{} {}({})",
                static_keyword, ret_type, method_name, param_list
            )
        },
        TargetLanguage::TypeScript => {
            let static_keyword = if is_static { "static " } else { "" };
            let ret_type = return_type.map(|t| format!(": {}", t)).unwrap_or_default();
            format!(
                "{}{}({}){}",
                static_keyword, method_name, param_list, ret_type
            )
        },
        TargetLanguage::JavaScript => {
            let static_keyword = if is_static { "static " } else { "" };
            format!("{}{}({})", static_keyword, method_name, param_list)
        },
        TargetLanguage::Go => {
            if let Some(ret_type) = return_type {
                format!("func {}({}) {}", method_name, param_list, ret_type)
            } else {
                format!("func {}({})", method_name, param_list)
            }
        },
        TargetLanguage::Swift => {
            let static_keyword = if is_static { "static " } else { "" };
            let ret_type = return_type.map(|t| format!(" -> {}", t)).unwrap_or_default();
            format!(
                "{}func {}({}){}",
                static_keyword, method_name, param_list, ret_type
            )
        },
        TargetLanguage::Kotlin => {
            let ret_type = return_type.map(|t| format!(": {}", t)).unwrap_or_default();
            format!("fun {}({}){}", method_name, param_list, ret_type)
        },
        TargetLanguage::Ruby => {
            format!("def {}({})", method_name, param_list)
        },
        TargetLanguage::Php => {
            let static_keyword = if is_static { "static " } else { "" };
            format!(
                "public {}function {}({})",
                static_keyword, method_name, param_list
            )
        },
        _ => {
            format!("{}({})", method_name, param_list)
        },
    }
}

/// Generate proper indentation for target language
pub fn get_indentation(language: TargetLanguage, level: usize) -> String {
    let indent_char = match language {
        TargetLanguage::Python => "    ", // 4 spaces
        TargetLanguage::Go => "\t",       // Tab
        _ => "    ",                      // 4 spaces for most languages
    };

    indent_char.repeat(level)
}
