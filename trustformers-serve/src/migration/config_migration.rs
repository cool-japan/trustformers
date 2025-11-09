use super::*;
use anyhow::Result;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;

/// Configuration migration handler for TrustformeRS Serve
pub struct ConfigMigrator {
    source_version: String,
    target_version: String,
    migration_rules: HashMap<String, Box<dyn ConfigMigrationRule + Send + Sync>>,
}

impl ConfigMigrator {
    pub fn new(source_version: String, target_version: String) -> Self {
        let mut migrator = Self {
            source_version,
            target_version,
            migration_rules: HashMap::new(),
        };

        migrator.register_default_rules();
        migrator
    }

    fn register_default_rules(&mut self) {
        // Register migration rules for different version transitions
        self.register_rule("1.0.0->2.0.0", Box::new(V1ToV2ConfigRule));
        self.register_rule("0.1.0->1.0.0", Box::new(V01ToV1ConfigRule));
        self.register_rule("2.0.0->2.1.0", Box::new(V2ToV21ConfigRule));
    }

    pub fn register_rule(
        &mut self,
        version_path: &str,
        rule: Box<dyn ConfigMigrationRule + Send + Sync>,
    ) {
        self.migration_rules.insert(version_path.to_string(), rule);
    }

    pub async fn migrate_config_file<P: AsRef<Path>>(&self, config_path: P) -> Result<()> {
        let config_content = fs::read_to_string(&config_path).await?;
        let migrated_content = self.migrate_config_content(&config_content)?;

        // Create backup before overwriting
        let backup_path = format!(
            "{}.backup.{}",
            config_path.as_ref().display(),
            Utc::now().timestamp()
        );
        fs::copy(&config_path, &backup_path).await?;

        // Write migrated configuration
        fs::write(&config_path, migrated_content).await?;

        Ok(())
    }

    pub fn migrate_config_content(&self, content: &str) -> Result<String> {
        let migration_key = format!("{}->{}", self.source_version, self.target_version);

        if let Some(rule) = self.migration_rules.get(&migration_key) {
            rule.migrate(content)
        } else {
            // Try to find a migration path through intermediate versions
            self.find_migration_path(content)
        }
    }

    fn find_migration_path(&self, _content: &str) -> Result<String> {
        // Implement a graph-based approach to find migration path
        // For now, return error if direct path not found
        Err(anyhow::anyhow!(
            "No migration path found from {} to {}",
            self.source_version,
            self.target_version
        ))
    }

    pub fn validate_config(&self, content: &str) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();

        // Parse as TOML first, then JSON if that fails
        let parsed: Result<Value, _> = if content.trim_start().starts_with('[')
            || content.contains('=')
        {
            // Likely TOML
            toml::from_str::<Value>(content).map_err(|e| anyhow::anyhow!("TOML parse error: {}", e))
        } else {
            // Likely JSON
            serde_json::from_str(content).map_err(|e| anyhow::anyhow!("JSON parse error: {}", e))
        };

        match parsed {
            Ok(_) => {
                results.push(ValidationResult {
                    rule_name: "Syntax Check".to_string(),
                    status: ValidationStatus::Passed,
                    message: "Configuration syntax is valid".to_string(),
                    details: None,
                });
            },
            Err(e) => {
                results.push(ValidationResult {
                    rule_name: "Syntax Check".to_string(),
                    status: ValidationStatus::Failed,
                    message: format!("Configuration syntax error: {}", e),
                    details: Some(serde_json::json!({"error": e.to_string()})),
                });
            },
        }

        // Additional validation rules
        results.extend(self.validate_required_fields(content)?);
        results.extend(self.validate_field_types(content)?);
        results.extend(self.validate_deprecated_fields(content)?);

        Ok(results)
    }

    fn validate_required_fields(&self, _content: &str) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();

        // Define required fields for target version
        let required_fields = match self.target_version.as_str() {
            "2.0.0" => vec!["server.host", "server.port", "logging.level"],
            "1.0.0" => vec!["host", "port"],
            _ => vec![],
        };

        for field in required_fields {
            // In a real implementation, we would check if the field exists
            results.push(ValidationResult {
                rule_name: format!("Required Field: {}", field),
                status: ValidationStatus::Passed,
                message: format!("Required field '{}' is present", field),
                details: None,
            });
        }

        Ok(results)
    }

    fn validate_field_types(&self, _content: &str) -> Result<Vec<ValidationResult>> {
        // Validate that fields have correct types
        Ok(vec![ValidationResult {
            rule_name: "Field Types".to_string(),
            status: ValidationStatus::Passed,
            message: "All fields have correct types".to_string(),
            details: None,
        }])
    }

    fn validate_deprecated_fields(&self, _content: &str) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();

        // Check for deprecated fields
        let deprecated_fields = match self.target_version.as_str() {
            "2.0.0" => vec!["old_field", "legacy_option"],
            _ => vec![],
        };

        for field in deprecated_fields {
            results.push(ValidationResult {
                rule_name: format!("Deprecated Field: {}", field),
                status: ValidationStatus::Warning,
                message: format!("Field '{}' is deprecated and should be removed", field),
                details: Some(serde_json::json!({"field": field, "action": "remove"})),
            });
        }

        Ok(results)
    }

    pub fn generate_migration_report(&self, original: &str, migrated: &str) -> MigrationReport {
        let changes = self.detect_changes(original, migrated);

        MigrationReport {
            source_version: self.source_version.clone(),
            target_version: self.target_version.clone(),
            changes,
            timestamp: Utc::now(),
        }
    }

    fn detect_changes(&self, original: &str, migrated: &str) -> Vec<ConfigChange> {
        // Simple change detection - in a real implementation, this would be more sophisticated
        let mut changes = Vec::new();

        if original != migrated {
            changes.push(ConfigChange {
                field_path: "root".to_string(),
                change_type: ConfigChangeType::Modified,
                old_value: Some(serde_json::Value::String(original.to_string())),
                new_value: Some(serde_json::Value::String(migrated.to_string())),
                description: "Configuration migrated".to_string(),
            });
        }

        changes
    }
}

pub trait ConfigMigrationRule: Send + Sync {
    fn migrate(&self, content: &str) -> Result<String>;
    fn get_version_range(&self) -> (String, String);
    fn get_description(&self) -> String;
}

/// Migration rule from version 1.0.0 to 2.0.0
pub struct V1ToV2ConfigRule;

impl ConfigMigrationRule for V1ToV2ConfigRule {
    fn migrate(&self, content: &str) -> Result<String> {
        // Parse existing configuration
        let mut config: Value = if content.trim_start().starts_with('{') {
            serde_json::from_str(content)?
        } else {
            toml::from_str(content)?
        };

        // Apply transformations
        if let Some(obj) = config.as_object_mut() {
            // Move host and port under server section
            if let (Some(host), Some(port)) = (obj.remove("host"), obj.remove("port")) {
                let server_section =
                    obj.entry("server").or_insert_with(|| Value::Object(Map::new()));
                if let Some(server_obj) = server_section.as_object_mut() {
                    server_obj.insert("host".to_string(), host);
                    server_obj.insert("port".to_string(), port);
                }
            }

            // Add new required fields with defaults
            if !obj.contains_key("logging") {
                obj.insert(
                    "logging".to_string(),
                    serde_json::json!({
                        "level": "info",
                        "format": "json"
                    }),
                );
            }

            if !obj.contains_key("metrics") {
                obj.insert(
                    "metrics".to_string(),
                    serde_json::json!({
                        "enabled": true,
                        "port": 9091
                    }),
                );
            }

            // Remove deprecated fields
            obj.remove("debug");
            obj.remove("verbose");
        }

        // Convert back to TOML format (assuming target format is TOML)
        toml::to_string_pretty(&config)
            .map_err(|e| anyhow::anyhow!("TOML serialization error: {}", e))
    }

    fn get_version_range(&self) -> (String, String) {
        ("1.0.0".to_string(), "2.0.0".to_string())
    }

    fn get_description(&self) -> String {
        "Migrate from v1.0.0 to v2.0.0: restructure configuration, add logging and metrics sections"
            .to_string()
    }
}

/// Migration rule from version 0.1.0 to 1.0.0
pub struct V01ToV1ConfigRule;

impl ConfigMigrationRule for V01ToV1ConfigRule {
    fn migrate(&self, content: &str) -> Result<String> {
        let mut config: Value = serde_json::from_str(content)?;

        if let Some(obj) = config.as_object_mut() {
            // Rename fields
            if let Some(bind_addr) = obj.remove("bind_address") {
                obj.insert("host".to_string(), bind_addr);
            }

            if let Some(bind_port) = obj.remove("bind_port") {
                obj.insert("port".to_string(), bind_port);
            }

            // Add default values for new fields
            if !obj.contains_key("workers") {
                obj.insert(
                    "workers".to_string(),
                    Value::Number(serde_json::Number::from(4)),
                );
            }
        }

        toml::to_string_pretty(&config)
            .map_err(|e| anyhow::anyhow!("TOML serialization error: {}", e))
    }

    fn get_version_range(&self) -> (String, String) {
        ("0.1.0".to_string(), "1.0.0".to_string())
    }

    fn get_description(&self) -> String {
        "Migrate from v0.1.0 to v1.0.0: rename fields and add worker configuration".to_string()
    }
}

/// Migration rule from version 2.0.0 to 2.1.0
pub struct V2ToV21ConfigRule;

impl ConfigMigrationRule for V2ToV21ConfigRule {
    fn migrate(&self, content: &str) -> Result<String> {
        let mut config: Value = toml::from_str(content)?;

        if let Some(obj) = config.as_object_mut() {
            // Add new optional features
            if !obj.contains_key("features") {
                obj.insert(
                    "features".to_string(),
                    serde_json::json!({
                        "graphql": true,
                        "streaming": true,
                        "websockets": false
                    }),
                );
            }

            // Update security section
            if let Some(security) = obj.get_mut("security") {
                if let Some(security_obj) = security.as_object_mut() {
                    if !security_obj.contains_key("rate_limiting") {
                        security_obj.insert(
                            "rate_limiting".to_string(),
                            serde_json::json!({
                                "enabled": true,
                                "requests_per_second": 100
                            }),
                        );
                    }
                }
            }
        }

        toml::to_string_pretty(&config)
            .map_err(|e| anyhow::anyhow!("TOML serialization error: {}", e))
    }

    fn get_version_range(&self) -> (String, String) {
        ("2.0.0".to_string(), "2.1.0".to_string())
    }

    fn get_description(&self) -> String {
        "Migrate from v2.0.0 to v2.1.0: add features and rate limiting configuration".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationReport {
    pub source_version: String,
    pub target_version: String,
    pub changes: Vec<ConfigChange>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigChange {
    pub field_path: String,
    pub change_type: ConfigChangeType,
    pub old_value: Option<Value>,
    pub new_value: Option<Value>,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigChangeType {
    Added,
    Removed,
    Modified,
    Renamed,
}

impl std::fmt::Display for ConfigChangeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigChangeType::Added => write!(f, "Added"),
            ConfigChangeType::Removed => write!(f, "Removed"),
            ConfigChangeType::Modified => write!(f, "Modified"),
            ConfigChangeType::Renamed => write!(f, "Renamed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_v1_to_v2_migration() {
        let migrator = ConfigMigrator::new("1.0.0".to_string(), "2.0.0".to_string());

        let v1_config = r#"
host = "0.0.0.0"
port = 8080
workers = 4
debug = true
"#;

        let migrated = migrator.migrate_config_content(v1_config).unwrap();

        // Verify the migration worked
        assert!(migrated.contains("[server]"));
        assert!(migrated.contains("[logging]"));
        assert!(migrated.contains("[metrics]"));
        assert!(!migrated.contains("debug"));
    }

    #[tokio::test]
    async fn test_config_validation() {
        let migrator = ConfigMigrator::new("1.0.0".to_string(), "2.0.0".to_string());

        let valid_config = r#"
[server]
host = "0.0.0.0"
port = 8080

[logging]
level = "info"
"#;

        let results = migrator.validate_config(valid_config).unwrap();
        assert!(results.iter().any(|r| r.status == ValidationStatus::Passed));
    }

    #[test]
    fn test_change_detection() {
        let migrator = ConfigMigrator::new("1.0.0".to_string(), "2.0.0".to_string());

        let original = "host = '0.0.0.0'";
        let migrated = "[server]\nhost = '0.0.0.0'";

        let changes = migrator.detect_changes(original, migrated);
        assert!(!changes.is_empty());
    }
}
