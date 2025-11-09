//! ABI Compatibility Checker for TrustformeRS Mobile
//!
//! This module provides comprehensive ABI (Application Binary Interface) compatibility
//! checking capabilities to ensure API stability across versions and prevent breaking
//! changes from being introduced unintentionally.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use trustformers_core::{errors::runtime_error, Result};

/// ABI version information
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AbiVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl AbiVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Check if this version is compatible with another version
    pub fn is_compatible_with(&self, other: &AbiVersion) -> bool {
        // Major version must match for ABI compatibility
        if self.major != other.major {
            return false;
        }

        // Minor version can be higher (backward compatible)
        if self.minor > other.minor {
            return true;
        }

        // Same minor version, patch can be different
        self.minor == other.minor
    }
}

/// Function signature information for ABI checking
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FunctionSignature {
    pub name: String,
    pub return_type: String,
    pub parameters: Vec<Parameter>,
    pub is_exported: bool,
    pub version_introduced: AbiVersion,
}

/// Parameter information
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub param_type: String,
    pub is_optional: bool,
}

/// Struct/type definition for ABI checking
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeDefinition {
    pub name: String,
    pub fields: Vec<Field>,
    pub type_kind: TypeKind,
    pub version_introduced: AbiVersion,
    pub is_exported: bool,
}

/// Field information for structs/enums
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub field_type: String,
    pub offset: Option<usize>,
    pub is_optional: bool,
}

/// Type categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeKind {
    Struct,
    Enum,
    Union,
    Alias,
}

/// ABI compatibility check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityResult {
    pub is_compatible: bool,
    pub breaking_changes: Vec<BreakingChange>,
    pub warnings: Vec<CompatibilityWarning>,
    pub added_functions: Vec<String>,
    pub added_types: Vec<String>,
}

/// Breaking change information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakingChange {
    pub change_type: ChangeType,
    pub description: String,
    pub affected_symbol: String,
    pub severity: Severity,
}

/// Type of ABI change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    FunctionRemoved,
    FunctionSignatureChanged,
    TypeRemoved,
    TypeLayoutChanged,
    FieldRemoved,
    FieldTypeChanged,
    EnumVariantRemoved,
}

/// Compatibility warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityWarning {
    pub message: String,
    pub symbol: String,
    pub recommendation: String,
}

/// Severity of breaking change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Critical, // Will definitely break existing code
    High,     // Likely to break existing code
    Medium,   // May break some code
    Low,      // Unlikely to break code but not recommended
}

/// ABI specification containing all exported symbols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbiSpecification {
    pub version: AbiVersion,
    pub functions: HashMap<String, FunctionSignature>,
    pub types: HashMap<String, TypeDefinition>,
    pub constants: HashMap<String, String>,
    pub generated_at: String,
}

impl AbiSpecification {
    /// Create a new ABI specification
    pub fn new(version: AbiVersion) -> Self {
        Self {
            version,
            functions: HashMap::new(),
            types: HashMap::new(),
            constants: HashMap::new(),
            generated_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Add a function to the specification
    pub fn add_function(&mut self, function: FunctionSignature) {
        self.functions.insert(function.name.clone(), function);
    }

    /// Add a type to the specification
    pub fn add_type(&mut self, type_def: TypeDefinition) {
        self.types.insert(type_def.name.clone(), type_def);
    }

    /// Add a constant to the specification
    pub fn add_constant(&mut self, name: String, value: String) {
        self.constants.insert(name, value);
    }
}

/// ABI compatibility checker
pub struct AbiChecker {
    baseline_spec: Option<AbiSpecification>,
}

impl AbiChecker {
    /// Create a new ABI checker
    pub fn new() -> Self {
        Self {
            baseline_spec: None,
        }
    }

    /// Set the baseline ABI specification
    pub fn set_baseline(&mut self, spec: AbiSpecification) {
        self.baseline_spec = Some(spec);
    }

    /// Load baseline specification from file
    pub fn load_baseline_from_file(&mut self, path: &str) -> Result<()> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| runtime_error(format!("Failed to read ABI spec: {}", e)))?;

        let spec: AbiSpecification = serde_json::from_str(&content)
            .map_err(|e| runtime_error(format!("Failed to parse ABI spec: {}", e)))?;

        self.baseline_spec = Some(spec);
        Ok(())
    }

    /// Save ABI specification to file
    pub fn save_specification(&self, spec: &AbiSpecification, path: &str) -> Result<()> {
        let content = serde_json::to_string_pretty(spec)
            .map_err(|e| runtime_error(format!("Failed to serialize ABI spec: {}", e)))?;

        std::fs::write(path, content)
            .map_err(|e| runtime_error(format!("Failed to write ABI spec: {}", e)))?;

        Ok(())
    }

    /// Check compatibility between baseline and current specification
    pub fn check_compatibility(
        &self,
        current_spec: &AbiSpecification,
    ) -> Result<CompatibilityResult> {
        let baseline = self
            .baseline_spec
            .as_ref()
            .ok_or_else(|| runtime_error("No baseline specification set"))?;

        let mut result = CompatibilityResult {
            is_compatible: true,
            breaking_changes: Vec::new(),
            warnings: Vec::new(),
            added_functions: Vec::new(),
            added_types: Vec::new(),
        };

        // Check function compatibility
        self.check_function_compatibility(baseline, current_spec, &mut result);

        // Check type compatibility
        self.check_type_compatibility(baseline, current_spec, &mut result);

        // Check for additions
        self.check_additions(baseline, current_spec, &mut result);

        // Determine overall compatibility
        result.is_compatible = result.breaking_changes.is_empty();

        Ok(result)
    }

    /// Check function compatibility
    fn check_function_compatibility(
        &self,
        baseline: &AbiSpecification,
        current: &AbiSpecification,
        result: &mut CompatibilityResult,
    ) {
        // Check for removed functions
        for (name, baseline_func) in &baseline.functions {
            if !current.functions.contains_key(name) {
                result.breaking_changes.push(BreakingChange {
                    change_type: ChangeType::FunctionRemoved,
                    description: format!("Function '{}' was removed", name),
                    affected_symbol: name.clone(),
                    severity: Severity::Critical,
                });
                continue;
            }

            let current_func = &current.functions[name];

            // Check signature compatibility
            if !self.is_function_signature_compatible(baseline_func, current_func) {
                result.breaking_changes.push(BreakingChange {
                    change_type: ChangeType::FunctionSignatureChanged,
                    description: format!("Function '{}' signature changed", name),
                    affected_symbol: name.clone(),
                    severity: Severity::High,
                });
            }
        }
    }

    /// Check type compatibility
    fn check_type_compatibility(
        &self,
        baseline: &AbiSpecification,
        current: &AbiSpecification,
        result: &mut CompatibilityResult,
    ) {
        // Check for removed types
        for (name, baseline_type) in &baseline.types {
            if !current.types.contains_key(name) {
                result.breaking_changes.push(BreakingChange {
                    change_type: ChangeType::TypeRemoved,
                    description: format!("Type '{}' was removed", name),
                    affected_symbol: name.clone(),
                    severity: Severity::Critical,
                });
                continue;
            }

            let current_type = &current.types[name];

            // Check layout compatibility
            if !self.is_type_layout_compatible(baseline_type, current_type) {
                result.breaking_changes.push(BreakingChange {
                    change_type: ChangeType::TypeLayoutChanged,
                    description: format!("Type '{}' layout changed", name),
                    affected_symbol: name.clone(),
                    severity: Severity::High,
                });
            }
        }
    }

    /// Check for additions (non-breaking but worth noting)
    fn check_additions(
        &self,
        baseline: &AbiSpecification,
        current: &AbiSpecification,
        result: &mut CompatibilityResult,
    ) {
        // Check for added functions
        for name in current.functions.keys() {
            if !baseline.functions.contains_key(name) {
                result.added_functions.push(name.clone());
            }
        }

        // Check for added types
        for name in current.types.keys() {
            if !baseline.types.contains_key(name) {
                result.added_types.push(name.clone());
            }
        }
    }

    /// Check if function signatures are compatible
    fn is_function_signature_compatible(
        &self,
        baseline: &FunctionSignature,
        current: &FunctionSignature,
    ) -> bool {
        // Return type must match exactly
        if baseline.return_type != current.return_type {
            return false;
        }

        // Parameter count must match
        if baseline.parameters.len() != current.parameters.len() {
            return false;
        }

        // Parameter types must match
        for (baseline_param, current_param) in baseline.parameters.iter().zip(&current.parameters) {
            if baseline_param.param_type != current_param.param_type {
                return false;
            }
        }

        true
    }

    /// Check if type layouts are compatible
    fn is_type_layout_compatible(
        &self,
        baseline: &TypeDefinition,
        current: &TypeDefinition,
    ) -> bool {
        // Type kind must match
        if baseline.type_kind != current.type_kind {
            return false;
        }

        match baseline.type_kind {
            TypeKind::Struct => self.is_struct_layout_compatible(baseline, current),
            TypeKind::Enum => self.is_enum_layout_compatible(baseline, current),
            TypeKind::Union => self.is_union_layout_compatible(baseline, current),
            TypeKind::Alias => baseline.fields == current.fields,
        }
    }

    /// Check struct layout compatibility
    fn is_struct_layout_compatible(
        &self,
        baseline: &TypeDefinition,
        current: &TypeDefinition,
    ) -> bool {
        // All baseline fields must be present in current
        for baseline_field in &baseline.fields {
            if let Some(current_field) =
                current.fields.iter().find(|f| f.name == baseline_field.name)
            {
                // Field type must match
                if baseline_field.field_type != current_field.field_type {
                    return false;
                }
                // Offset must match if specified
                if let (Some(baseline_offset), Some(current_offset)) =
                    (baseline_field.offset, current_field.offset)
                {
                    if baseline_offset != current_offset {
                        return false;
                    }
                }
            } else if !baseline_field.is_optional {
                // Required field is missing
                return false;
            }
        }

        true
    }

    /// Check enum layout compatibility
    fn is_enum_layout_compatible(
        &self,
        baseline: &TypeDefinition,
        current: &TypeDefinition,
    ) -> bool {
        // All baseline variants must be present
        let baseline_variants: HashSet<_> = baseline.fields.iter().map(|f| &f.name).collect();
        let current_variants: HashSet<_> = current.fields.iter().map(|f| &f.name).collect();

        baseline_variants.is_subset(&current_variants)
    }

    /// Check union layout compatibility
    fn is_union_layout_compatible(
        &self,
        baseline: &TypeDefinition,
        current: &TypeDefinition,
    ) -> bool {
        // Union compatibility is more strict - all fields must match exactly
        baseline.fields == current.fields
    }

    /// Generate current ABI specification from analysis
    pub fn generate_current_specification(&self) -> Result<AbiSpecification> {
        let version = AbiVersion::new(0, 1, 0); // Current version
        let mut spec = AbiSpecification::new(version);

        // Add mobile inference functions
        self.add_mobile_inference_functions(&mut spec);

        // Add mobile optimization types
        self.add_mobile_optimization_types(&mut spec);

        // Add mobile configuration types
        self.add_mobile_config_types(&mut spec);

        Ok(spec)
    }

    /// Add mobile inference functions to specification
    fn add_mobile_inference_functions(&self, spec: &mut AbiSpecification) {
        // Mobile inference engine functions
        spec.add_function(FunctionSignature {
            name: "trustformers_mobile_engine_new".to_string(),
            return_type: "*mut MobileInferenceEngine".to_string(),
            parameters: vec![Parameter {
                name: "config".to_string(),
                param_type: "*const MobileConfig".to_string(),
                is_optional: false,
            }],
            is_exported: true,
            version_introduced: AbiVersion::new(0, 1, 0),
        });

        spec.add_function(FunctionSignature {
            name: "trustformers_mobile_engine_free".to_string(),
            return_type: "void".to_string(),
            parameters: vec![Parameter {
                name: "engine".to_string(),
                param_type: "*mut MobileInferenceEngine".to_string(),
                is_optional: false,
            }],
            is_exported: true,
            version_introduced: AbiVersion::new(0, 1, 0),
        });

        spec.add_function(FunctionSignature {
            name: "trustformers_mobile_inference".to_string(),
            return_type: "i32".to_string(),
            parameters: vec![
                Parameter {
                    name: "engine".to_string(),
                    param_type: "*mut MobileInferenceEngine".to_string(),
                    is_optional: false,
                },
                Parameter {
                    name: "input".to_string(),
                    param_type: "*const f32".to_string(),
                    is_optional: false,
                },
                Parameter {
                    name: "input_len".to_string(),
                    param_type: "usize".to_string(),
                    is_optional: false,
                },
                Parameter {
                    name: "output".to_string(),
                    param_type: "*mut f32".to_string(),
                    is_optional: false,
                },
                Parameter {
                    name: "output_len".to_string(),
                    param_type: "usize".to_string(),
                    is_optional: false,
                },
            ],
            is_exported: true,
            version_introduced: AbiVersion::new(0, 1, 0),
        });
    }

    /// Add mobile optimization types to specification
    fn add_mobile_optimization_types(&self, spec: &mut AbiSpecification) {
        // MobilePrecision enum
        spec.add_type(TypeDefinition {
            name: "MobilePrecision".to_string(),
            fields: vec![
                Field {
                    name: "INT4".to_string(),
                    field_type: "u32".to_string(),
                    offset: Some(0),
                    is_optional: false,
                },
                Field {
                    name: "INT8".to_string(),
                    field_type: "u32".to_string(),
                    offset: Some(1),
                    is_optional: false,
                },
                Field {
                    name: "FP16".to_string(),
                    field_type: "u32".to_string(),
                    offset: Some(2),
                    is_optional: false,
                },
                Field {
                    name: "Mixed4_8".to_string(),
                    field_type: "u32".to_string(),
                    offset: Some(3),
                    is_optional: false,
                },
                Field {
                    name: "Mixed8_16".to_string(),
                    field_type: "u32".to_string(),
                    offset: Some(4),
                    is_optional: false,
                },
                Field {
                    name: "DYNAMIC".to_string(),
                    field_type: "u32".to_string(),
                    offset: Some(5),
                    is_optional: false,
                },
            ],
            type_kind: TypeKind::Enum,
            version_introduced: AbiVersion::new(0, 1, 0),
            is_exported: true,
        });

        // MobilePlatform enum
        spec.add_type(TypeDefinition {
            name: "MobilePlatform".to_string(),
            fields: vec![
                Field {
                    name: "Ios".to_string(),
                    field_type: "u32".to_string(),
                    offset: Some(0),
                    is_optional: false,
                },
                Field {
                    name: "Android".to_string(),
                    field_type: "u32".to_string(),
                    offset: Some(1),
                    is_optional: false,
                },
                Field {
                    name: "Generic".to_string(),
                    field_type: "u32".to_string(),
                    offset: Some(2),
                    is_optional: false,
                },
            ],
            type_kind: TypeKind::Enum,
            version_introduced: AbiVersion::new(0, 1, 0),
            is_exported: true,
        });
    }

    /// Add mobile configuration types to specification
    fn add_mobile_config_types(&self, spec: &mut AbiSpecification) {
        // MobileConfig struct
        spec.add_type(TypeDefinition {
            name: "MobileConfig".to_string(),
            fields: vec![
                Field {
                    name: "platform".to_string(),
                    field_type: "MobilePlatform".to_string(),
                    offset: Some(0),
                    is_optional: false,
                },
                Field {
                    name: "backend".to_string(),
                    field_type: "MobileBackend".to_string(),
                    offset: Some(8),
                    is_optional: false,
                },
                Field {
                    name: "precision".to_string(),
                    field_type: "MobilePrecision".to_string(),
                    offset: Some(16),
                    is_optional: false,
                },
                Field {
                    name: "batch_size".to_string(),
                    field_type: "usize".to_string(),
                    offset: Some(24),
                    is_optional: false,
                },
                Field {
                    name: "max_memory_mb".to_string(),
                    field_type: "usize".to_string(),
                    offset: Some(32),
                    is_optional: false,
                },
                Field {
                    name: "enable_caching".to_string(),
                    field_type: "bool".to_string(),
                    offset: Some(40),
                    is_optional: false,
                },
            ],
            type_kind: TypeKind::Struct,
            version_introduced: AbiVersion::new(0, 1, 0),
            is_exported: true,
        });
    }
}

impl Default for AbiChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abi_version_compatibility() {
        let v1_0_0 = AbiVersion::new(1, 0, 0);
        let v1_1_0 = AbiVersion::new(1, 1, 0);
        let v2_0_0 = AbiVersion::new(2, 0, 0);

        assert!(v1_1_0.is_compatible_with(&v1_0_0));
        assert!(!v1_0_0.is_compatible_with(&v2_0_0));
        assert!(!v2_0_0.is_compatible_with(&v1_0_0));
    }

    #[test]
    fn test_abi_checker_creation() {
        let checker = AbiChecker::new();
        assert!(checker.baseline_spec.is_none());
    }

    #[test]
    fn test_function_signature_compatibility() {
        let checker = AbiChecker::new();

        let baseline = FunctionSignature {
            name: "test_func".to_string(),
            return_type: "i32".to_string(),
            parameters: vec![Parameter {
                name: "param1".to_string(),
                param_type: "i32".to_string(),
                is_optional: false,
            }],
            is_exported: true,
            version_introduced: AbiVersion::new(0, 1, 0),
        };

        let compatible = FunctionSignature {
            name: "test_func".to_string(),
            return_type: "i32".to_string(),
            parameters: vec![Parameter {
                name: "param1".to_string(),
                param_type: "i32".to_string(),
                is_optional: false,
            }],
            is_exported: true,
            version_introduced: AbiVersion::new(0, 1, 0),
        };

        let incompatible = FunctionSignature {
            name: "test_func".to_string(),
            return_type: "f32".to_string(), // Different return type
            parameters: vec![Parameter {
                name: "param1".to_string(),
                param_type: "i32".to_string(),
                is_optional: false,
            }],
            is_exported: true,
            version_introduced: AbiVersion::new(0, 1, 0),
        };

        assert!(checker.is_function_signature_compatible(&baseline, &compatible));
        assert!(!checker.is_function_signature_compatible(&baseline, &incompatible));
    }

    #[test]
    fn test_specification_generation() {
        let checker = AbiChecker::new();
        let spec = checker.generate_current_specification().unwrap();

        assert!(!spec.functions.is_empty());
        assert!(!spec.types.is_empty());
        assert!(spec.functions.contains_key("trustformers_mobile_engine_new"));
        assert!(spec.types.contains_key("MobilePlatform"));
    }

    #[test]
    fn test_struct_layout_compatibility() {
        let checker = AbiChecker::new();

        let baseline = TypeDefinition {
            name: "TestStruct".to_string(),
            fields: vec![
                Field {
                    name: "field1".to_string(),
                    field_type: "i32".to_string(),
                    offset: Some(0),
                    is_optional: false,
                },
                Field {
                    name: "field2".to_string(),
                    field_type: "f32".to_string(),
                    offset: Some(4),
                    is_optional: false,
                },
            ],
            type_kind: TypeKind::Struct,
            version_introduced: AbiVersion::new(0, 1, 0),
            is_exported: true,
        };

        let compatible = TypeDefinition {
            name: "TestStruct".to_string(),
            fields: vec![
                Field {
                    name: "field1".to_string(),
                    field_type: "i32".to_string(),
                    offset: Some(0),
                    is_optional: false,
                },
                Field {
                    name: "field2".to_string(),
                    field_type: "f32".to_string(),
                    offset: Some(4),
                    is_optional: false,
                },
                Field {
                    name: "field3".to_string(),
                    field_type: "bool".to_string(),
                    offset: Some(8),
                    is_optional: true,
                }, // Added optional field
            ],
            type_kind: TypeKind::Struct,
            version_introduced: AbiVersion::new(0, 1, 0),
            is_exported: true,
        };

        let incompatible = TypeDefinition {
            name: "TestStruct".to_string(),
            fields: vec![
                Field {
                    name: "field1".to_string(),
                    field_type: "f64".to_string(),
                    offset: Some(0),
                    is_optional: false,
                }, // Changed type
                Field {
                    name: "field2".to_string(),
                    field_type: "f32".to_string(),
                    offset: Some(8),
                    is_optional: false,
                }, // Changed offset
            ],
            type_kind: TypeKind::Struct,
            version_introduced: AbiVersion::new(0, 1, 0),
            is_exported: true,
        };

        assert!(checker.is_struct_layout_compatible(&baseline, &compatible));
        assert!(!checker.is_struct_layout_compatible(&baseline, &incompatible));
    }
}
