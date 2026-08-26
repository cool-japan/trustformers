//! JSON Schema validation for schema-constrained generation.
//!
//! Implements the structural and value keywords of JSON Schema draft 7 that
//! matter for constrained decoding:
//!
//! * type checks (`type`, including a list of alternatives)
//! * value checks (`enum`, `const`)
//! * number checks (`minimum`, `maximum`, `exclusiveMinimum`,
//!   `exclusiveMaximum`, `multipleOf`)
//! * string checks (`minLength`, `maxLength`, `pattern`)
//! * array checks (`items`, `minItems`, `maxItems`, `uniqueItems`)
//! * object checks (`properties`, `required`, `additionalProperties`,
//!   `minProperties`, `maxProperties`)
//! * combinators (`allOf`, `anyOf`, `oneOf`, `not`)
//!
//! Keywords outside that set are ignored rather than silently failing, which
//! matches the JSON Schema rule that unknown keywords are annotations.
//! `$ref` is **not** resolved; a schema containing one is reported as
//! [`JsonSchemaError::UnsupportedKeyword`] at construction time instead of
//! being quietly treated as "accept everything".

use std::fmt;

use regex::Regex;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Reasons a document can fail schema validation, or a schema can fail to
/// load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonSchemaError {
    /// The schema text itself is not valid JSON.
    InvalidSchema(String),
    /// The schema uses a keyword this validator deliberately does not fake.
    UnsupportedKeyword(String),
    /// A `pattern` was not a valid regular expression.
    InvalidPattern(String),
    /// The document violated a constraint.
    Violation { path: String, reason: String },
}

impl fmt::Display for JsonSchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonSchemaError::InvalidSchema(reason) => write!(f, "invalid JSON schema: {reason}"),
            JsonSchemaError::UnsupportedKeyword(keyword) => {
                write!(f, "unsupported JSON schema keyword: {keyword}")
            },
            JsonSchemaError::InvalidPattern(pattern) => {
                write!(f, "invalid `pattern` regular expression: {pattern}")
            },
            JsonSchemaError::Violation { path, reason } => {
                write!(f, "{path}: {reason}")
            },
        }
    }
}

impl std::error::Error for JsonSchemaError {}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

/// A compiled JSON schema.
#[derive(Debug, Clone)]
pub struct JsonSchema {
    root: Value,
}

impl JsonSchema {
    /// Compile a schema from its JSON text.
    pub fn parse(text: &str) -> Result<Self, JsonSchemaError> {
        let root: Value = serde_json::from_str(text)
            .map_err(|error| JsonSchemaError::InvalidSchema(error.to_string()))?;
        if !root.is_object() && !root.is_boolean() {
            return Err(JsonSchemaError::InvalidSchema(
                "a schema must be an object or a boolean".to_string(),
            ));
        }
        reject_unsupported(&root)?;
        Ok(Self { root })
    }

    /// The schema as a JSON value.
    pub fn as_value(&self) -> &Value {
        &self.root
    }

    /// Validate a JSON document against the schema.
    pub fn validate(&self, document: &Value) -> Result<(), JsonSchemaError> {
        validate_against(&self.root, document, "$")
    }

    /// Validate the JSON text of a document.
    pub fn validate_text(&self, text: &str) -> Result<(), JsonSchemaError> {
        let document: Value = serde_json::from_str(text)
            .map_err(|error| JsonSchemaError::InvalidSchema(error.to_string()))?;
        self.validate(&document)
    }
}

/// Walk the schema and reject `$ref`, which this validator cannot resolve.
fn reject_unsupported(schema: &Value) -> Result<(), JsonSchemaError> {
    match schema {
        Value::Object(map) => {
            if map.contains_key("$ref") {
                return Err(JsonSchemaError::UnsupportedKeyword("$ref".to_string()));
            }
            for (key, value) in map {
                // Only recurse into positions that hold sub-schemas.
                match key.as_str() {
                    "properties" | "patternProperties" | "definitions" | "$defs" => {
                        if let Value::Object(children) = value {
                            for child in children.values() {
                                reject_unsupported(child)?;
                            }
                        }
                    },
                    "items" | "additionalProperties" | "not" | "contains" | "propertyNames" => {
                        reject_unsupported(value)?;
                    },
                    "allOf" | "anyOf" | "oneOf" => {
                        if let Value::Array(children) = value {
                            for child in children {
                                reject_unsupported(child)?;
                            }
                        }
                    },
                    _ => {},
                }
            }
            Ok(())
        },
        _ => Ok(()),
    }
}

fn violation(path: &str, reason: impl Into<String>) -> JsonSchemaError {
    JsonSchemaError::Violation {
        path: path.to_string(),
        reason: reason.into(),
    }
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn matches_type(expected: &str, value: &Value) -> bool {
    match expected {
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        other => other == type_name(value),
    }
}

fn validate_against(schema: &Value, value: &Value, path: &str) -> Result<(), JsonSchemaError> {
    let map = match schema {
        // `true` accepts everything, `false` rejects everything.
        Value::Bool(true) => return Ok(()),
        Value::Bool(false) => return Err(violation(path, "schema `false` rejects all documents")),
        Value::Object(map) => map,
        _ => return Ok(()),
    };

    if let Some(expected) = map.get("type") {
        validate_type(expected, value, path)?;
    }

    if let Some(Value::Array(allowed)) = map.get("enum") {
        if !allowed.contains(value) {
            return Err(violation(
                path,
                format!("value {value} is not one of the allowed enum entries"),
            ));
        }
    }

    if let Some(expected) = map.get("const") {
        if expected != value {
            return Err(violation(
                path,
                format!("value {value} does not equal the required const {expected}"),
            ));
        }
    }

    validate_number_keywords(map, value, path)?;
    validate_string_keywords(map, value, path)?;
    validate_array_keywords(map, value, path)?;
    validate_object_keywords(map, value, path)?;
    validate_combinators(map, value, path)?;

    Ok(())
}

fn validate_type(expected: &Value, value: &Value, path: &str) -> Result<(), JsonSchemaError> {
    let acceptable = match expected {
        Value::String(name) => matches_type(name, value),
        Value::Array(names) => names
            .iter()
            .filter_map(|name| name.as_str())
            .any(|name| matches_type(name, value)),
        _ => true,
    };
    if acceptable {
        Ok(())
    } else {
        Err(violation(
            path,
            format!("expected type {expected}, found {}", type_name(value)),
        ))
    }
}

fn validate_number_keywords(
    map: &serde_json::Map<String, Value>,
    value: &Value,
    path: &str,
) -> Result<(), JsonSchemaError> {
    let Some(number) = value.as_f64() else {
        return Ok(());
    };

    if let Some(minimum) = map.get("minimum").and_then(Value::as_f64) {
        if number < minimum {
            return Err(violation(
                path,
                format!("{number} is below minimum {minimum}"),
            ));
        }
    }
    if let Some(maximum) = map.get("maximum").and_then(Value::as_f64) {
        if number > maximum {
            return Err(violation(
                path,
                format!("{number} is above maximum {maximum}"),
            ));
        }
    }
    if let Some(bound) = map.get("exclusiveMinimum").and_then(Value::as_f64) {
        if number <= bound {
            return Err(violation(
                path,
                format!("{number} must be strictly greater than {bound}"),
            ));
        }
    }
    if let Some(bound) = map.get("exclusiveMaximum").and_then(Value::as_f64) {
        if number >= bound {
            return Err(violation(
                path,
                format!("{number} must be strictly less than {bound}"),
            ));
        }
    }
    if let Some(step) = map.get("multipleOf").and_then(Value::as_f64) {
        if step > 0.0 {
            let quotient = number / step;
            if (quotient - quotient.round()).abs() > 1e-9 {
                return Err(violation(
                    path,
                    format!("{number} is not a multiple of {step}"),
                ));
            }
        }
    }
    Ok(())
}

fn validate_string_keywords(
    map: &serde_json::Map<String, Value>,
    value: &Value,
    path: &str,
) -> Result<(), JsonSchemaError> {
    let Some(text) = value.as_str() else {
        return Ok(());
    };
    let length = text.chars().count();

    if let Some(minimum) = map.get("minLength").and_then(Value::as_u64) {
        if (length as u64) < minimum {
            return Err(violation(
                path,
                format!("string of length {length} is shorter than minLength {minimum}"),
            ));
        }
    }
    if let Some(maximum) = map.get("maxLength").and_then(Value::as_u64) {
        if (length as u64) > maximum {
            return Err(violation(
                path,
                format!("string of length {length} is longer than maxLength {maximum}"),
            ));
        }
    }
    if let Some(pattern) = map.get("pattern").and_then(Value::as_str) {
        let regex = Regex::new(pattern)
            .map_err(|_| JsonSchemaError::InvalidPattern(pattern.to_string()))?;
        if !regex.is_match(text) {
            return Err(violation(
                path,
                format!("string {text:?} does not match pattern {pattern:?}"),
            ));
        }
    }
    Ok(())
}

fn validate_array_keywords(
    map: &serde_json::Map<String, Value>,
    value: &Value,
    path: &str,
) -> Result<(), JsonSchemaError> {
    let Some(items) = value.as_array() else {
        return Ok(());
    };

    if let Some(minimum) = map.get("minItems").and_then(Value::as_u64) {
        if (items.len() as u64) < minimum {
            return Err(violation(
                path,
                format!(
                    "array of {} items is shorter than minItems {minimum}",
                    items.len()
                ),
            ));
        }
    }
    if let Some(maximum) = map.get("maxItems").and_then(Value::as_u64) {
        if (items.len() as u64) > maximum {
            return Err(violation(
                path,
                format!(
                    "array of {} items is longer than maxItems {maximum}",
                    items.len()
                ),
            ));
        }
    }
    if map.get("uniqueItems").and_then(Value::as_bool) == Some(true) {
        for (index, item) in items.iter().enumerate() {
            if items[..index].contains(item) {
                return Err(violation(path, format!("duplicate item at index {index}")));
            }
        }
    }

    match map.get("items") {
        Some(Value::Array(schemas)) => {
            for (index, (item, schema)) in items.iter().zip(schemas.iter()).enumerate() {
                validate_against(schema, item, &format!("{path}[{index}]"))?;
            }
        },
        Some(schema) => {
            for (index, item) in items.iter().enumerate() {
                validate_against(schema, item, &format!("{path}[{index}]"))?;
            }
        },
        None => {},
    }
    Ok(())
}

fn validate_object_keywords(
    map: &serde_json::Map<String, Value>,
    value: &Value,
    path: &str,
) -> Result<(), JsonSchemaError> {
    let Some(object) = value.as_object() else {
        return Ok(());
    };

    if let Some(Value::Array(required)) = map.get("required") {
        for name in required.iter().filter_map(Value::as_str) {
            if !object.contains_key(name) {
                return Err(violation(
                    path,
                    format!("missing required property `{name}`"),
                ));
            }
        }
    }
    if let Some(minimum) = map.get("minProperties").and_then(Value::as_u64) {
        if (object.len() as u64) < minimum {
            return Err(violation(
                path,
                format!(
                    "object has {} properties, minProperties is {minimum}",
                    object.len()
                ),
            ));
        }
    }
    if let Some(maximum) = map.get("maxProperties").and_then(Value::as_u64) {
        if (object.len() as u64) > maximum {
            return Err(violation(
                path,
                format!(
                    "object has {} properties, maxProperties is {maximum}",
                    object.len()
                ),
            ));
        }
    }

    let properties = map.get("properties").and_then(Value::as_object);
    if let Some(properties) = properties {
        for (name, child) in object {
            if let Some(schema) = properties.get(name) {
                validate_against(schema, child, &format!("{path}.{name}"))?;
            }
        }
    }

    match map.get("additionalProperties") {
        Some(Value::Bool(false)) => {
            for name in object.keys() {
                let declared = properties.map(|p| p.contains_key(name)).unwrap_or(false);
                if !declared {
                    return Err(violation(
                        path,
                        format!("additional property `{name}` is not allowed"),
                    ));
                }
            }
        },
        Some(schema @ Value::Object(_)) => {
            for (name, child) in object {
                let declared = properties.map(|p| p.contains_key(name)).unwrap_or(false);
                if !declared {
                    validate_against(schema, child, &format!("{path}.{name}"))?;
                }
            }
        },
        _ => {},
    }

    Ok(())
}

fn validate_combinators(
    map: &serde_json::Map<String, Value>,
    value: &Value,
    path: &str,
) -> Result<(), JsonSchemaError> {
    if let Some(Value::Array(schemas)) = map.get("allOf") {
        for schema in schemas {
            validate_against(schema, value, path)?;
        }
    }
    if let Some(Value::Array(schemas)) = map.get("anyOf") {
        let any = schemas.iter().any(|schema| validate_against(schema, value, path).is_ok());
        if !any {
            return Err(violation(
                path,
                "value satisfies none of the `anyOf` schemas",
            ));
        }
    }
    if let Some(Value::Array(schemas)) = map.get("oneOf") {
        let matches = schemas
            .iter()
            .filter(|schema| validate_against(schema, value, path).is_ok())
            .count();
        if matches != 1 {
            return Err(violation(
                path,
                format!("value satisfies {matches} of the `oneOf` schemas, expected exactly 1"),
            ));
        }
    }
    if let Some(schema) = map.get("not") {
        if validate_against(schema, value, path).is_ok() {
            return Err(violation(path, "value matches the `not` schema"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(text: &str) -> JsonSchema {
        JsonSchema::parse(text).expect("schema compiles")
    }

    #[test]
    fn test_empty_schema_accepts_everything() {
        let s = schema("{}");
        assert!(s.validate_text("42").is_ok());
        assert!(s.validate_text(r#"{"a": 1}"#).is_ok());
        assert!(s.validate_text(r#""hello""#).is_ok());
    }

    #[test]
    fn test_type_keyword_is_enforced() {
        // Regression: the old validator only checked JSON well-formedness and
        // ignored the schema entirely, so this used to pass.
        let s = schema(r#"{"type": "object"}"#);
        assert!(s.validate_text(r#"{"a": 1}"#).is_ok());
        assert!(s.validate_text("42").is_err());
        assert!(s.validate_text(r#""str""#).is_err());
    }

    #[test]
    fn test_integer_versus_number() {
        let integers = schema(r#"{"type": "integer"}"#);
        assert!(integers.validate_text("7").is_ok());
        assert!(integers.validate_text("7.5").is_err());

        let numbers = schema(r#"{"type": "number"}"#);
        assert!(numbers.validate_text("7.5").is_ok());
    }

    #[test]
    fn test_type_union() {
        let s = schema(r#"{"type": ["string", "null"]}"#);
        assert!(s.validate_text(r#""x""#).is_ok());
        assert!(s.validate_text("null").is_ok());
        assert!(s.validate_text("1").is_err());
    }

    #[test]
    fn test_required_properties() {
        let s = schema(r#"{"type":"object","required":["name","age"]}"#);
        assert!(s.validate_text(r#"{"name":"a","age":1}"#).is_ok());
        assert!(s.validate_text(r#"{"name":"a"}"#).is_err());
    }

    #[test]
    fn test_nested_property_schemas() {
        let s = schema(r#"{"type":"object","properties":{"age":{"type":"integer","minimum":0}}}"#);
        assert!(s.validate_text(r#"{"age": 30}"#).is_ok());
        assert!(s.validate_text(r#"{"age": -1}"#).is_err());
        assert!(s.validate_text(r#"{"age": "old"}"#).is_err());
    }

    #[test]
    fn test_additional_properties_false() {
        let s = schema(
            r#"{"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":false}"#,
        );
        assert!(s.validate_text(r#"{"a": 1}"#).is_ok());
        assert!(s.validate_text(r#"{"a": 1, "b": 2}"#).is_err());
    }

    #[test]
    fn test_array_items_and_bounds() {
        let s = schema(r#"{"type":"array","items":{"type":"string"},"minItems":2}"#);
        assert!(s.validate_text(r#"["a","b"]"#).is_ok());
        assert!(s.validate_text(r#"["a"]"#).is_err());
        assert!(s.validate_text(r#"["a", 2]"#).is_err());
    }

    #[test]
    fn test_unique_items() {
        let s = schema(r#"{"type":"array","uniqueItems":true}"#);
        assert!(s.validate_text("[1,2,3]").is_ok());
        assert!(s.validate_text("[1,2,1]").is_err());
    }

    #[test]
    fn test_string_length_and_pattern() {
        let s = schema(r#"{"type":"string","minLength":2,"maxLength":4,"pattern":"^a"}"#);
        assert!(s.validate_text(r#""abc""#).is_ok());
        assert!(s.validate_text(r#""a""#).is_err());
        assert!(s.validate_text(r#""abcde""#).is_err());
        assert!(s.validate_text(r#""bbc""#).is_err());
    }

    #[test]
    fn test_enum_and_const() {
        let enums = schema(r#"{"enum":["red","green"]}"#);
        assert!(enums.validate_text(r#""red""#).is_ok());
        assert!(enums.validate_text(r#""blue""#).is_err());

        let constant = schema(r#"{"const": 5}"#);
        assert!(constant.validate_text("5").is_ok());
        assert!(constant.validate_text("6").is_err());
    }

    #[test]
    fn test_numeric_bounds() {
        let s = schema(r#"{"type":"number","exclusiveMinimum":0,"maximum":10,"multipleOf":2}"#);
        assert!(s.validate_text("2").is_ok());
        assert!(s.validate_text("0").is_err());
        assert!(s.validate_text("12").is_err());
        assert!(s.validate_text("3").is_err());
    }

    #[test]
    fn test_combinators() {
        let any_of = schema(r#"{"anyOf":[{"type":"string"},{"type":"integer"}]}"#);
        assert!(any_of.validate_text(r#""x""#).is_ok());
        assert!(any_of.validate_text("1").is_ok());
        assert!(any_of.validate_text("true").is_err());

        let one_of = schema(r#"{"oneOf":[{"type":"number"},{"type":"integer"}]}"#);
        assert!(one_of.validate_text("1").is_err(), "matches both branches");

        let not = schema(r#"{"not":{"type":"string"}}"#);
        assert!(not.validate_text("1").is_ok());
        assert!(not.validate_text(r#""x""#).is_err());

        let all_of = schema(r#"{"allOf":[{"type":"integer"},{"minimum":5}]}"#);
        assert!(all_of.validate_text("6").is_ok());
        assert!(all_of.validate_text("4").is_err());
    }

    #[test]
    fn test_boolean_schemas() {
        assert!(schema("true").validate_text("1").is_ok());
        assert!(schema("false").validate_text("1").is_err());
    }

    #[test]
    fn test_invalid_schema_text_is_an_error() {
        assert!(matches!(
            JsonSchema::parse("{not json}"),
            Err(JsonSchemaError::InvalidSchema(_))
        ));
        assert!(matches!(
            JsonSchema::parse("[1,2]"),
            Err(JsonSchemaError::InvalidSchema(_))
        ));
    }

    #[test]
    fn test_ref_is_reported_rather_than_ignored() {
        assert_eq!(
            JsonSchema::parse(r##"{"$ref": "#/definitions/x"}"##).err(),
            Some(JsonSchemaError::UnsupportedKeyword("$ref".to_string()))
        );
        assert!(matches!(
            JsonSchema::parse(r##"{"properties":{"a":{"$ref":"#/x"}}}"##),
            Err(JsonSchemaError::UnsupportedKeyword(_))
        ));
    }

    #[test]
    fn test_violation_message_carries_a_path() {
        let s = schema(r#"{"properties":{"a":{"type":"integer"}}}"#);
        let error = s.validate_text(r#"{"a": "x"}"#).expect_err("must fail");
        assert!(error.to_string().contains("$.a"), "{error}");
    }

    #[test]
    fn test_error_display_is_non_empty() {
        assert!(!JsonSchemaError::InvalidSchema("x".into()).to_string().is_empty());
        assert!(!JsonSchemaError::UnsupportedKeyword("x".into()).to_string().is_empty());
        assert!(!JsonSchemaError::InvalidPattern("x".into()).to_string().is_empty());
    }
}
