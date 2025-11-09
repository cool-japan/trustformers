//! OpenAPI/Swagger documentation generator for FFI interfaces
//!
//! Generates OpenAPI 3.0 specifications from FFI interface definitions,
//! mapping FFI functions to REST API endpoints with proper schemas.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use serde_json::{json, Value};

use crate::codegen::ast::{
    ConstantValue, FfiEnum, FfiFunction, FfiInterface, FfiParameter, FfiStruct, FfiType,
    PrimitiveType,
};
use crate::codegen::templates::TemplateEngine;
use crate::codegen::{CodeGenConfig, TargetLanguage};
use crate::error::TrustformersResult;

use super::LanguageGenerator;

/// OpenAPI 3.0 specification generator for FFI interfaces
///
/// Maps FFI functions to REST API endpoints and generates:
/// - OpenAPI 3.0 JSON/YAML specification
/// - Swagger UI integration HTML
/// - README.md with API documentation
/// - Example API client code
pub struct OpenApiGenerator {
    config: CodeGenConfig,
}

impl OpenApiGenerator {
    pub fn new(config: &CodeGenConfig) -> TrustformersResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Generate complete OpenAPI 3.0 specification
    fn generate_openapi_spec(&self, interface: &FfiInterface) -> Value {
        let mut spec = json!({
            "openapi": "3.0.3",
            "info": {
                "title": interface.metadata.library_name.clone().or_else(||
                    Some(self.config.package_info.name.clone())).unwrap_or_else(|| "TrustformeRS API".to_string()),
                "description": self.config.package_info.description.clone(),
                "version": interface.metadata.version.clone().or_else(||
                    Some(self.config.package_info.version.clone())).unwrap_or_else(|| "1.0.0".to_string()),
                "contact": {
                    "name": self.config.package_info.author.clone(),
                    "url": self.config.package_info.repository.clone()
                },
                "license": {
                    "name": self.config.package_info.license.clone(),
                    "url": format!("https://opensource.org/licenses/{}", self.config.package_info.license)
                }
            },
            "servers": [
                {
                    "url": "http://localhost:8080/api/v1",
                    "description": "Local development server"
                },
                {
                    "url": "https://api.trustformers.ai/v1",
                    "description": "Production server"
                }
            ],
            "tags": self.generate_tags(interface),
            "paths": self.generate_paths(interface),
            "components": {
                "schemas": self.generate_schemas(interface),
                "responses": self.generate_responses(),
                "securitySchemes": {
                    "ApiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key"
                    },
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    }
                },
                "parameters": {
                    "VersionParam": {
                        "name": "version",
                        "in": "query",
                        "description": "API version",
                        "schema": {
                            "type": "string",
                            "default": "v1"
                        }
                    }
                }
            },
            "security": [
                {
                    "ApiKeyAuth": []
                },
                {
                    "BearerAuth": []
                }
            ]
        });

        // Add external documentation if available
        if !self.config.package_info.repository.is_empty() {
            spec["externalDocs"] = json!({
                "description": "Find more info here",
                "url": format!("{}/blob/main/README.md", self.config.package_info.repository)
            });
        }

        spec
    }

    /// Generate API tags for grouping operations
    fn generate_tags(&self, interface: &FfiInterface) -> Value {
        let mut tags = HashSet::new();
        let mut tag_descriptions = HashMap::new();

        // Extract tags from function names (prefix before first underscore)
        for func in &interface.functions {
            let tag = self.extract_tag_from_function(&func.name);
            tags.insert(tag.clone());

            if !tag_descriptions.contains_key(&tag) {
                tag_descriptions
                    .insert(tag.clone(), self.generate_tag_description(&tag, interface));
            }
        }

        // Convert to JSON array
        let mut tag_array: Vec<Value> = tags
            .iter()
            .map(|tag| {
                json!({
                    "name": tag,
                    "description": tag_descriptions.get(tag).unwrap_or(&format!("{} operations", tag))
                })
            })
            .collect();

        // Sort tags alphabetically
        tag_array.sort_by(|a, b| a["name"].as_str().unwrap().cmp(b["name"].as_str().unwrap()));

        json!(tag_array)
    }

    /// Extract tag from function name (e.g., "model_load" -> "model")
    fn extract_tag_from_function(&self, name: &str) -> String {
        // Remove common prefixes
        let name = name.trim_start_matches("trustformers_").trim_start_matches("tf_");

        // Get first component
        name.split('_').next().unwrap_or("general").to_string()
    }

    /// Generate description for a tag based on related functions
    fn generate_tag_description(&self, tag: &str, interface: &FfiInterface) -> String {
        match tag {
            "model" => "Model loading, configuration, and management operations".to_string(),
            "tokenizer" => "Tokenization and text processing operations".to_string(),
            "pipeline" => "End-to-end inference pipeline operations".to_string(),
            "tensor" => "Tensor creation and manipulation operations".to_string(),
            "inference" => "Model inference and prediction operations".to_string(),
            "training" => "Model training and optimization operations".to_string(),
            "config" => "Configuration management operations".to_string(),
            "error" => "Error handling and status operations".to_string(),
            "memory" => "Memory management operations".to_string(),
            "device" => "Device and hardware management operations".to_string(),
            _ => {
                // Generate generic description
                let count = interface
                    .functions
                    .iter()
                    .filter(|f| self.extract_tag_from_function(&f.name) == tag)
                    .count();
                format!("{} related operations ({} functions)", tag, count)
            },
        }
    }

    /// Generate OpenAPI paths from FFI functions
    fn generate_paths(&self, interface: &FfiInterface) -> Value {
        let mut paths = serde_json::Map::new();

        for func in &interface.functions {
            let (path, method) = self.function_to_endpoint(func);
            let operation = self.generate_operation(func, interface);

            if !paths.contains_key(&path) {
                paths.insert(path.clone(), json!({}));
            }

            let path_item = paths.get_mut(&path).unwrap();
            if let Some(obj) = path_item.as_object_mut() {
                obj.insert(method, operation);
            }
        }

        json!(paths)
    }

    /// Convert FFI function to REST endpoint (path and method)
    fn function_to_endpoint(&self, func: &FfiFunction) -> (String, String) {
        let name = func.name.trim_start_matches("trustformers_").trim_start_matches("tf_");

        // Determine HTTP method based on function name
        let method =
            if name.starts_with("create_") || name.starts_with("new_") || name.ends_with("_create")
            {
                "post".to_string()
            } else if name.starts_with("get_")
                || name.starts_with("list_")
                || name.starts_with("read_")
                || name.ends_with("_info")
            {
                "get".to_string()
            } else if name.starts_with("update_")
                || name.starts_with("set_")
                || name.ends_with("_update")
            {
                "put".to_string()
            } else if name.starts_with("delete_")
                || name.starts_with("destroy_")
                || name.starts_with("free_")
                || name.ends_with("_destroy")
                || name.ends_with("_free")
            {
                "delete".to_string()
            } else if name.contains("_partial_") || name.starts_with("patch_") {
                "patch".to_string()
            } else {
                "post".to_string() // Default to POST for actions
            };

        // Convert function name to REST path
        let path = self.function_name_to_path(name);

        (path, method)
    }

    /// Convert function name to REST path
    fn function_name_to_path(&self, name: &str) -> String {
        // Split by underscore and build path
        let parts: Vec<&str> = name.split('_').collect();

        if parts.is_empty() {
            return format!("/{}", name);
        }

        // Handle common patterns
        let path = if parts.len() >= 2 {
            match (parts[0], parts[1]) {
                ("model", "load") => "/models/load",
                ("model", "unload") => "/models/unload",
                ("model", "info") => "/models/info",
                ("model", "list") => "/models",
                ("tokenizer", "create") => "/tokenizers",
                ("tokenizer", "encode") => "/tokenizers/encode",
                ("tokenizer", "decode") => "/tokenizers/decode",
                ("pipeline", "create") => "/pipelines",
                ("pipeline", "run") => "/pipelines/run",
                ("tensor", "create") => "/tensors",
                ("tensor", "from") if parts.len() > 2 => {
                    format!("/tensors/from/{}", parts[2..].join("/"))
                },
                ("inference", "run") => "/inference",
                ("config", "load") => "/config",
                ("config", "save") => "/config",
                _ => {
                    // Generic conversion: first part is resource, rest is action
                    if parts.len() == 2 {
                        format!("/{}/{}", parts[0], parts[1])
                    } else {
                        format!("/{}/{}", parts[0], parts[1..].join("/"))
                    }
                },
            }
        } else {
            format!("/{}", name)
        };

        path.to_string()
    }

    /// Generate operation object for a function
    fn generate_operation(&self, func: &FfiFunction, interface: &FfiInterface) -> Value {
        let tag = self.extract_tag_from_function(&func.name);
        let operation_id = func.name.clone();
        let summary = self.generate_summary(func);
        let description = if func.documentation.is_empty() {
            summary.clone()
        } else {
            func.documentation.join("\n")
        };

        let mut operation = json!({
            "tags": [tag],
            "operationId": operation_id,
            "summary": summary,
            "description": description,
            "parameters": self.generate_parameters(func),
            "responses": self.generate_operation_responses(func, interface)
        });

        // Add request body if function has non-path parameters
        if self.should_have_request_body(func) {
            operation["requestBody"] = self.generate_request_body(func, interface);
        }

        // Add deprecation notice if applicable
        if func.is_deprecated() {
            operation["deprecated"] = json!(true);
            if let Some(ref deprecation) = func.deprecation {
                operation["x-deprecation-message"] = json!(deprecation.message);
                if let Some(ref replacement) = deprecation.replacement {
                    operation["x-deprecated-replacement"] = json!(replacement);
                }
            }
        }

        // Add security requirements if needed
        if self.requires_authentication(func) {
            operation["security"] = json!([
                {"ApiKeyAuth": []},
                {"BearerAuth": []}
            ]);
        }

        operation
    }

    /// Generate summary for a function
    fn generate_summary(&self, func: &FfiFunction) -> String {
        if !func.documentation.is_empty() {
            // Use first line of documentation
            return func.documentation[0].clone();
        }

        // Generate from function name
        let name = func.name.trim_start_matches("trustformers_").trim_start_matches("tf_");

        // Convert snake_case to Title Case
        name.split('_')
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().chain(chars).collect(),
                }
            })
            .collect::<Vec<String>>()
            .join(" ")
    }

    /// Generate parameters for operation
    fn generate_parameters(&self, func: &FfiFunction) -> Value {
        let mut params = Vec::new();

        // Only include path and query parameters
        // Body parameters go in requestBody
        for param in &func.parameters {
            if self.is_path_parameter(param) {
                params.push(self.generate_parameter_schema(param, "path"));
            } else if self.is_query_parameter(param) {
                params.push(self.generate_parameter_schema(param, "query"));
            }
        }

        json!(params)
    }

    /// Check if parameter should be in path
    fn is_path_parameter(&self, param: &FfiParameter) -> bool {
        // Handles and IDs are typically path parameters
        param.type_info.is_handle()
            || param.name.ends_with("_id")
            || param.name.ends_with("_handle")
    }

    /// Check if parameter should be in query
    fn is_query_parameter(&self, param: &FfiParameter) -> bool {
        // Simple types that are optional can be query params
        param.is_optional
            && !param.type_info.is_handle()
            && !param.type_info.is_pointer()
            && param.type_info.primitive_type.is_some()
    }

    /// Generate parameter schema
    fn generate_parameter_schema(&self, param: &FfiParameter, location: &str) -> Value {
        let mut schema = json!({
            "name": param.name,
            "in": location,
            "required": !param.is_optional,
            "schema": self.type_to_openapi_schema(&param.type_info)
        });

        if !param.documentation.is_empty() {
            schema["description"] = json!(param.documentation.join("\n"));
        }

        if let Some(ref default) = param.default_value {
            schema["schema"]["default"] = json!(default);
        }

        schema
    }

    /// Check if function should have request body
    fn should_have_request_body(&self, func: &FfiFunction) -> bool {
        // Functions with non-simple parameters should have request body
        func.parameters
            .iter()
            .any(|p| !self.is_path_parameter(p) && !self.is_query_parameter(p))
    }

    /// Generate request body schema
    fn generate_request_body(&self, func: &FfiFunction, interface: &FfiInterface) -> Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in &func.parameters {
            if !self.is_path_parameter(param) && !self.is_query_parameter(param) {
                properties.insert(
                    param.name.clone(),
                    self.type_to_openapi_schema(&param.type_info),
                );

                if !param.is_optional {
                    required.push(param.name.clone());
                }
            }
        }

        json!({
            "required": true,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }
        })
    }

    /// Generate response schemas for operation
    fn generate_operation_responses(&self, func: &FfiFunction, interface: &FfiInterface) -> Value {
        let mut responses = serde_json::Map::new();

        // Success response
        let success_response = if func.return_type.name == "void"
            || func.return_type.primitive_type == Some(PrimitiveType::Void)
        {
            json!({
                "description": "Operation completed successfully",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {
                                    "type": "boolean",
                                    "example": true
                                },
                                "message": {
                                    "type": "string",
                                    "example": "Operation completed successfully"
                                }
                            }
                        }
                    }
                }
            })
        } else {
            json!({
                "description": "Successful operation",
                "content": {
                    "application/json": {
                        "schema": self.type_to_openapi_schema(&func.return_type)
                    }
                }
            })
        };

        responses.insert("200".to_string(), success_response);

        // Error responses
        if func.can_fail() {
            responses.insert(
                "400".to_string(),
                json!({"$ref": "#/components/responses/BadRequest"}),
            );
            responses.insert(
                "404".to_string(),
                json!({"$ref": "#/components/responses/NotFound"}),
            );
            responses.insert(
                "500".to_string(),
                json!({"$ref": "#/components/responses/InternalError"}),
            );
        }

        json!(responses)
    }

    /// Convert FFI type to OpenAPI schema
    fn type_to_openapi_schema(&self, ffi_type: &FfiType) -> Value {
        // Handle arrays
        if let Some(array_size) = ffi_type.array_size {
            return json!({
                "type": "array",
                "items": self.primitive_to_openapi_type(ffi_type.primitive_type.as_ref()),
                "minItems": array_size,
                "maxItems": array_size
            });
        }

        // Handle pointers to arrays (dynamic arrays)
        if ffi_type.is_pointer() && !ffi_type.is_string() && !ffi_type.is_handle() {
            return json!({
                "type": "array",
                "items": self.primitive_to_openapi_type(ffi_type.primitive_type.as_ref())
            });
        }

        // Handle strings
        if ffi_type.is_string() {
            return json!({
                "type": "string"
            });
        }

        // Handle handles (opaque pointers)
        if ffi_type.is_handle() {
            return json!({
                "type": "string",
                "format": "uuid",
                "description": "Resource handle/identifier"
            });
        }

        // Handle primitives
        if let Some(ref primitive) = ffi_type.primitive_type {
            return self.primitive_to_openapi_type(Some(primitive));
        }

        // Handle references to other schemas
        if !ffi_type.is_pointer() && !ffi_type.is_const {
            return json!({
                "$ref": format!("#/components/schemas/{}", ffi_type.name)
            });
        }

        // Default to object
        json!({
            "type": "object"
        })
    }

    /// Convert primitive type to OpenAPI type
    fn primitive_to_openapi_type(&self, primitive: Option<&PrimitiveType>) -> Value {
        match primitive {
            Some(PrimitiveType::Bool) => json!({"type": "boolean"}),
            Some(PrimitiveType::Int8) | Some(PrimitiveType::Int16) | Some(PrimitiveType::Int32) => {
                json!({"type": "integer", "format": "int32"})
            },
            Some(PrimitiveType::Int64) | Some(PrimitiveType::IntPtr) => {
                json!({"type": "integer", "format": "int64"})
            },
            Some(PrimitiveType::UInt8)
            | Some(PrimitiveType::UInt16)
            | Some(PrimitiveType::UInt32) => {
                json!({"type": "integer", "format": "int32", "minimum": 0})
            },
            Some(PrimitiveType::UInt64) | Some(PrimitiveType::UIntPtr) => {
                json!({"type": "integer", "format": "int64", "minimum": 0})
            },
            Some(PrimitiveType::Float32) => json!({"type": "number", "format": "float"}),
            Some(PrimitiveType::Float64) => json!({"type": "number", "format": "double"}),
            Some(PrimitiveType::Char) | Some(PrimitiveType::CString) => json!({"type": "string"}),
            Some(PrimitiveType::Void) => json!({"type": "null"}),
            Some(PrimitiveType::Handle) | Some(PrimitiveType::OpaquePointer) => {
                json!({"type": "string", "format": "uuid"})
            },
            None => json!({"type": "object"}),
        }
    }

    /// Generate component schemas from structs and enums
    fn generate_schemas(&self, interface: &FfiInterface) -> Value {
        let mut schemas = serde_json::Map::new();

        // Generate schemas for structs
        for struct_def in &interface.structs {
            let schema = self.struct_to_schema(struct_def);
            schemas.insert(struct_def.name.clone(), schema);
        }

        // Generate schemas for enums
        for enum_def in &interface.enums {
            let schema = self.enum_to_schema(enum_def);
            schemas.insert(enum_def.name.clone(), schema);
        }

        // Add common error schema
        schemas.insert(
            "Error".to_string(),
            json!({
                "type": "object",
                "required": ["code", "message"],
                "properties": {
                    "code": {
                        "type": "integer",
                        "description": "Error code"
                    },
                    "message": {
                        "type": "string",
                        "description": "Error message"
                    },
                    "details": {
                        "type": "object",
                        "description": "Additional error details"
                    }
                }
            }),
        );

        json!(schemas)
    }

    /// Convert struct to OpenAPI schema
    fn struct_to_schema(&self, struct_def: &FfiStruct) -> Value {
        if struct_def.is_opaque {
            return json!({
                "type": "object",
                "description": struct_def.documentation.join("\n"),
                "x-opaque": true,
                "additionalProperties": false
            });
        }

        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for field in &struct_def.fields {
            if !field.is_private {
                properties.insert(
                    field.name.clone(),
                    self.type_to_openapi_schema(&field.type_info),
                );
                required.push(field.name.clone());
            }
        }

        let mut schema = json!({
            "type": "object",
            "properties": properties,
            "required": required
        });

        if !struct_def.documentation.is_empty() {
            schema["description"] = json!(struct_def.documentation.join("\n"));
        }

        schema
    }

    /// Convert enum to OpenAPI schema
    fn enum_to_schema(&self, enum_def: &FfiEnum) -> Value {
        let values: Vec<i64> = enum_def.variants.iter().map(|v| v.value).collect();
        let names: Vec<String> = enum_def.variants.iter().map(|v| v.name.clone()).collect();

        let mut schema = if enum_def.is_flags {
            // Bitfield enum
            json!({
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": names
                },
                "uniqueItems": true
            })
        } else {
            // Regular enum
            json!({
                "type": "integer",
                "enum": values,
                "x-enum-names": names
            })
        };

        if !enum_def.documentation.is_empty() {
            schema["description"] = json!(enum_def.documentation.join("\n"));
        }

        // Add variant descriptions
        let mut variants_info = Vec::new();
        for variant in &enum_def.variants {
            variants_info.push(json!({
                "name": variant.name,
                "value": variant.value,
                "description": variant.documentation.join("\n")
            }));
        }
        schema["x-enum-variants"] = json!(variants_info);

        schema
    }

    /// Generate common response definitions
    fn generate_responses(&self) -> Value {
        json!({
            "BadRequest": {
                "description": "Bad request - invalid parameters",
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": "#/components/schemas/Error"
                        },
                        "example": {
                            "code": -2,
                            "message": "Invalid parameter provided",
                            "details": {
                                "parameter": "model_path",
                                "reason": "File not found"
                            }
                        }
                    }
                }
            },
            "NotFound": {
                "description": "Resource not found",
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": "#/components/schemas/Error"
                        },
                        "example": {
                            "code": -4,
                            "message": "File not found",
                            "details": {
                                "path": "/path/to/model"
                            }
                        }
                    }
                }
            },
            "InternalError": {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": "#/components/schemas/Error"
                        },
                        "example": {
                            "code": -13,
                            "message": "Runtime error occurred",
                            "details": {
                                "trace": "Stack trace information"
                            }
                        }
                    }
                }
            },
            "Unauthorized": {
                "description": "Authentication required",
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": "#/components/schemas/Error"
                        },
                        "example": {
                            "code": 401,
                            "message": "Authentication required"
                        }
                    }
                }
            }
        })
    }

    /// Check if function requires authentication
    fn requires_authentication(&self, func: &FfiFunction) -> bool {
        // Functions that modify state or access sensitive data require auth
        let name = func.name.as_str();
        name.contains("create")
            || name.contains("update")
            || name.contains("delete")
            || name.contains("set")
            || name.contains("configure")
    }

    /// Generate Swagger UI HTML
    fn generate_swagger_ui_html(&self) -> String {
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{} - API Documentation</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.10.0/swagger-ui.css">
    <style>
        body {{
            margin: 0;
            padding: 0;
        }}
        .topbar {{
            background-color: #1a1a1a !important;
        }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5.10.0/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5.10.0/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            const ui = SwaggerUIBundle({{
                url: "openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                defaultModelsExpandDepth: 1,
                defaultModelExpandDepth: 1,
                docExpansion: "list",
                filter: true,
                showExtensions: true,
                showCommonExtensions: true,
                syntaxHighlight: {{
                    activate: true,
                    theme: "monokai"
                }}
            }});
            window.ui = ui;
        }};
    </script>
</body>
</html>
"#,
            self.config.package_info.name
        )
    }

    /// Generate README.md with API documentation
    fn generate_readme(&self, interface: &FfiInterface) -> String {
        let mut readme = String::new();

        readme.push_str(&format!(
            "# {} API Documentation\n\n",
            self.config.package_info.name
        ));
        readme.push_str(&format!("{}\n\n", self.config.package_info.description));

        readme.push_str("## Overview\n\n");
        readme.push_str("This API provides access to the TrustformeRS transformer library functionality through a RESTful interface.\n\n");

        readme.push_str("## OpenAPI Specification\n\n");
        readme.push_str(
            "- **Specification**: `openapi.json` (JSON format) or `openapi.yaml` (YAML format)\n",
        );
        readme
            .push_str("- **Interactive Documentation**: Open `swagger-ui.html` in a web browser\n");
        readme.push_str("- **Version**: OpenAPI 3.0.3\n\n");

        readme.push_str("## Authentication\n\n");
        readme.push_str("The API supports two authentication methods:\n\n");
        readme.push_str("1. **API Key**: Include `X-API-Key` header in requests\n");
        readme.push_str("   ```bash\n");
        readme.push_str(
            "   curl -H \"X-API-Key: your-api-key\" https://api.trustformers.ai/v1/models\n",
        );
        readme.push_str("   ```\n\n");
        readme.push_str("2. **Bearer Token**: Include `Authorization: Bearer <token>` header\n");
        readme.push_str("   ```bash\n");
        readme.push_str("   curl -H \"Authorization: Bearer your-jwt-token\" https://api.trustformers.ai/v1/models\n");
        readme.push_str("   ```\n\n");

        readme.push_str("## Base URLs\n\n");
        readme.push_str("- Development: `http://localhost:8080/api/v1`\n");
        readme.push_str("- Production: `https://api.trustformers.ai/v1`\n\n");

        readme.push_str("## API Endpoints\n\n");

        // Group functions by tag
        let mut functions_by_tag: HashMap<String, Vec<&FfiFunction>> = HashMap::new();
        for func in &interface.functions {
            let tag = self.extract_tag_from_function(&func.name);
            functions_by_tag.entry(tag).or_insert_with(Vec::new).push(func);
        }

        for (tag, functions) in &functions_by_tag {
            readme.push_str(&format!("### {} Operations\n\n", tag.to_uppercase()));

            for func in functions {
                let (path, method) = self.function_to_endpoint(func);
                readme.push_str(&format!(
                    "- **{} {}**: {}\n",
                    method.to_uppercase(),
                    path,
                    self.generate_summary(func)
                ));
            }

            readme.push_str("\n");
        }

        readme.push_str("## Example Requests\n\n");
        readme.push_str("### Load a Model\n\n");
        readme.push_str("```bash\n");
        readme.push_str("curl -X POST https://api.trustformers.ai/v1/models/load \\\n");
        readme.push_str("  -H \"Content-Type: application/json\" \\\n");
        readme.push_str("  -H \"X-API-Key: your-api-key\" \\\n");
        readme.push_str("  -d '{\n");
        readme.push_str("    \"model_path\": \"/path/to/model\",\n");
        readme.push_str("    \"device\": \"cuda\"\n");
        readme.push_str("  }'\n");
        readme.push_str("```\n\n");

        readme.push_str("### Run Inference\n\n");
        readme.push_str("```bash\n");
        readme.push_str("curl -X POST https://api.trustformers.ai/v1/inference \\\n");
        readme.push_str("  -H \"Content-Type: application/json\" \\\n");
        readme.push_str("  -H \"X-API-Key: your-api-key\" \\\n");
        readme.push_str("  -d '{\n");
        readme.push_str("    \"model_id\": \"bert-base-uncased\",\n");
        readme.push_str("    \"input\": \"Hello, world!\"\n");
        readme.push_str("  }'\n");
        readme.push_str("```\n\n");

        readme.push_str("## Error Handling\n\n");
        readme.push_str("All errors follow a consistent format:\n\n");
        readme.push_str("```json\n");
        readme.push_str("{\n");
        readme.push_str("  \"code\": -2,\n");
        readme.push_str("  \"message\": \"Invalid parameter provided\",\n");
        readme.push_str("  \"details\": {\n");
        readme.push_str("    \"parameter\": \"model_path\",\n");
        readme.push_str("    \"reason\": \"File not found\"\n");
        readme.push_str("  }\n");
        readme.push_str("}\n");
        readme.push_str("```\n\n");

        readme.push_str("### Common Error Codes\n\n");
        readme.push_str("- `-1`: Null pointer\n");
        readme.push_str("- `-2`: Invalid parameter\n");
        readme.push_str("- `-3`: Out of memory\n");
        readme.push_str("- `-4`: File not found\n");
        readme.push_str("- `-6`: Model loading error\n");
        readme.push_str("- `-9`: Inference error\n\n");

        readme.push_str("## Rate Limiting\n\n");
        readme.push_str("API requests are rate-limited to prevent abuse:\n\n");
        readme.push_str("- Free tier: 100 requests/minute\n");
        readme.push_str("- Pro tier: 1000 requests/minute\n");
        readme.push_str("- Enterprise: Custom limits\n\n");

        readme.push_str("## SDK Support\n\n");
        readme.push_str("Official SDKs are available for:\n\n");
        readme.push_str("- Python (`pip install trustformers-api`)\n");
        readme.push_str("- JavaScript/TypeScript (`npm install @trustformers/api`)\n");
        readme.push_str("- Go (`go get github.com/trustformers/api-go`)\n");
        readme.push_str("- Java (Maven/Gradle)\n\n");

        readme.push_str("## Support\n\n");
        readme.push_str(&format!(
            "- Documentation: {}\n",
            self.config.package_info.repository
        ));
        readme.push_str("- Issues: Report bugs and feature requests on GitHub\n");
        readme.push_str(&format!(
            "- License: {}\n",
            self.config.package_info.license
        ));

        readme
    }

    /// Generate example Python client code
    fn generate_python_client_example(&self) -> String {
        format!(
            r#"""
{} Python API Client Example

Usage:
    pip install requests
    python client_example.py
"""

import requests
import json
from typing import Dict, Any, Optional

class {}Client:
    """Python client for {} API"""

    def __init__(self, api_key: str, base_url: str = "https://api.trustformers.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({{
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }})

    def _request(self, method: str, path: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make API request"""
        url = f"{{self.base_url}}{{path}}"

        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=data)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data)
            elif method.upper() == "DELETE":
                response = self.session.delete(url)
            else:
                raise ValueError(f"Unsupported method: {{method}}")

            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_data = e.response.json() if e.response.text else {{}};
            raise Exception(f"API error: {{error_data.get('message', str(e))}}")

    def load_model(self, model_path: str, device: str = "cpu") -> Dict[str, Any]:
        """Load a model"""
        return self._request("POST", "/models/load", {{
            "model_path": model_path,
            "device": device
        }})

    def run_inference(self, model_id: str, input_text: str) -> Dict[str, Any]:
        """Run inference on input text"""
        return self._request("POST", "/inference", {{
            "model_id": model_id,
            "input": input_text
        }})

    def tokenize(self, text: str) -> Dict[str, Any]:
        """Tokenize text"""
        return self._request("POST", "/tokenizers/encode", {{
            "text": text
        }})

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = {}Client(api_key="your-api-key-here")

    # Load a model
    print("Loading model...")
    model = client.load_model(
        model_path="bert-base-uncased",
        device="cuda"
    )
    print(f"Model loaded: {{model}}")

    # Run inference
    print("\\nRunning inference...")
    result = client.run_inference(
        model_id="bert-base-uncased",
        input_text="Hello, world!"
    )
    print(f"Inference result: {{result}}")

    # Tokenize text
    print("\\nTokenizing text...")
    tokens = client.tokenize("This is a test")
    print(f"Tokens: {{tokens}}")
"#,
            self.config.package_info.name,
            self.config.package_info.name.replace("-", "").to_uppercase(),
            self.config.package_info.name,
            self.config.package_info.name.replace("-", "").to_uppercase()
        )
    }

    /// Generate example TypeScript client code
    fn generate_typescript_client_example(&self) -> String {
        format!(
            r#"/**
 * {} TypeScript API Client Example
 *
 * Usage:
 *   npm install axios
 *   ts-node client_example.ts
 */

import axios, {{ AxiosInstance, AxiosResponse }} from 'axios';

interface ApiError {{
    code: number;
    message: string;
    details?: any;
}}

interface LoadModelRequest {{
    model_path: string;
    device?: string;
}}

interface InferenceRequest {{
    model_id: string;
    input: string;
}}

interface TokenizeRequest {{
    text: string;
}}

export class {}Client {{
    private client: AxiosInstance;

    constructor(
        private apiKey: string,
        private baseUrl: string = 'https://api.trustformers.ai/v1'
    ) {{
        this.client = axios.create({{
            baseURL: this.baseUrl,
            headers: {{
                'X-API-Key': this.apiKey,
                'Content-Type': 'application/json'
            }}
        }});

        // Add error interceptor
        this.client.interceptors.response.use(
            response => response,
            error => {{
                if (error.response?.data) {{
                    const apiError: ApiError = error.response.data;
                    throw new Error(`API Error (${{apiError.code}}): ${{apiError.message}}`);
                }}
                throw error;
            }}
        );
    }}

    async loadModel(request: LoadModelRequest): Promise<any> {{
        const response = await this.client.post('/models/load', request);
        return response.data;
    }}

    async runInference(request: InferenceRequest): Promise<any> {{
        const response = await this.client.post('/inference', request);
        return response.data;
    }}

    async tokenize(request: TokenizeRequest): Promise<any> {{
        const response = await this.client.post('/tokenizers/encode', request);
        return response.data;
    }}
}}

// Example usage
async function main() {{
    const client = new {}Client('your-api-key-here');

    try {{
        // Load model
        console.log('Loading model...');
        const model = await client.loadModel({{
            model_path: 'bert-base-uncased',
            device: 'cuda'
        }});
        console.log('Model loaded:', model);

        // Run inference
        console.log('\\nRunning inference...');
        const result = await client.runInference({{
            model_id: 'bert-base-uncased',
            input: 'Hello, world!'
        }});
        console.log('Inference result:', result);

        // Tokenize
        console.log('\\nTokenizing text...');
        const tokens = await client.tokenize({{
            text: 'This is a test'
        }});
        console.log('Tokens:', tokens);
    }} catch (error) {{
        console.error('Error:', error);
    }}
}}

if (require.main === module) {{
    main();
}}
"#,
            self.config.package_info.name,
            self.config.package_info.name.replace("-", ""),
            self.config.package_info.name.replace("-", "")
        )
    }
}

impl LanguageGenerator for OpenApiGenerator {
    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::OpenApi
    }

    fn file_extension(&self) -> &'static str {
        "yaml"
    }

    fn generate(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        // Create output directory
        fs::create_dir_all(output_dir)?;

        // Generate OpenAPI specification
        let spec = self.generate_openapi_spec(interface);

        // Write JSON spec
        let json_path = output_dir.join("openapi.json");
        let json_content = serde_json::to_string_pretty(&spec)?;
        fs::write(&json_path, json_content)?;
        println!("Generated OpenAPI JSON: {:?}", json_path);

        // Write YAML spec (using JSON as source, convert to YAML)
        let yaml_path = output_dir.join("openapi.yaml");
        let yaml_content = serde_yaml::to_string(&spec)
            .map_err(|e| crate::error::TrustformersError::SerializationError)?;
        fs::write(&yaml_path, yaml_content)?;
        println!("Generated OpenAPI YAML: {:?}", yaml_path);

        // Generate Swagger UI HTML
        let swagger_html = self.generate_swagger_ui_html();
        let html_path = output_dir.join("swagger-ui.html");
        fs::write(&html_path, swagger_html)?;
        println!("Generated Swagger UI: {:?}", html_path);

        // Generate README
        let readme_content = self.generate_readme(interface);
        let readme_path = output_dir.join("README.md");
        fs::write(&readme_path, readme_content)?;
        println!("Generated README: {:?}", readme_path);

        // Generate example client code
        let examples_dir = output_dir.join("examples");
        fs::create_dir_all(&examples_dir)?;

        // Python example
        let python_example = self.generate_python_client_example();
        let python_path = examples_dir.join("client_example.py");
        fs::write(&python_path, python_example)?;
        println!("Generated Python example: {:?}", python_path);

        // TypeScript example
        let typescript_example = self.generate_typescript_client_example();
        let typescript_path = examples_dir.join("client_example.ts");
        fs::write(&typescript_path, typescript_example)?;
        println!("Generated TypeScript example: {:?}", typescript_path);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::PackageInfo;

    fn create_test_config() -> CodeGenConfig {
        CodeGenConfig {
            output_dir: std::path::PathBuf::from("/tmp/openapi_test"),
            target_languages: vec![TargetLanguage::OpenApi],
            package_info: PackageInfo {
                name: "trustformers".to_string(),
                version: "1.0.0".to_string(),
                description: "Test API".to_string(),
                author: "Test Author".to_string(),
                license: "MIT".to_string(),
                repository: "https://github.com/test/repo".to_string(),
            },
            features: HashMap::new(),
            type_mappings: HashMap::new(),
        }
    }

    #[test]
    fn test_openapi_generator_creation() {
        let config = create_test_config();
        let generator = OpenApiGenerator::new(&config);
        assert!(generator.is_ok());
    }

    #[test]
    fn test_extract_tag_from_function() {
        let config = create_test_config();
        let generator = OpenApiGenerator::new(&config).unwrap();

        assert_eq!(
            generator.extract_tag_from_function("trustformers_model_load"),
            "model"
        );
        assert_eq!(
            generator.extract_tag_from_function("tf_tokenizer_encode"),
            "tokenizer"
        );
        assert_eq!(
            generator.extract_tag_from_function("pipeline_create"),
            "pipeline"
        );
    }

    #[test]
    fn test_function_to_endpoint() {
        let config = create_test_config();
        let generator = OpenApiGenerator::new(&config).unwrap();

        let mut func = FfiFunction::default();
        func.name = "model_load".to_string();

        let (path, method) = generator.function_to_endpoint(&func);
        assert_eq!(path, "/models/load");
        assert_eq!(method, "post");

        func.name = "model_info".to_string();
        let (path, method) = generator.function_to_endpoint(&func);
        assert_eq!(path, "/models/info");
        assert_eq!(method, "get");
    }

    #[test]
    fn test_type_to_openapi_schema() {
        let config = create_test_config();
        let generator = OpenApiGenerator::new(&config).unwrap();

        let int_type = FfiType {
            name: "c_int".to_string(),
            primitive_type: Some(PrimitiveType::Int32),
            ..Default::default()
        };

        let schema = generator.type_to_openapi_schema(&int_type);
        assert_eq!(schema["type"], "integer");
        assert_eq!(schema["format"], "int32");

        let string_type = FfiType {
            name: "*const c_char".to_string(),
            is_pointer: true,
            is_const: true,
            primitive_type: Some(PrimitiveType::CString),
            ..Default::default()
        };

        let schema = generator.type_to_openapi_schema(&string_type);
        assert_eq!(schema["type"], "string");
    }

    #[test]
    fn test_generate_swagger_ui_html() {
        let config = create_test_config();
        let generator = OpenApiGenerator::new(&config).unwrap();

        let html = generator.generate_swagger_ui_html();
        assert!(html.contains("swagger-ui"));
        assert!(html.contains("openapi.json"));
        assert!(html.contains(&config.package_info.name));
    }
}
