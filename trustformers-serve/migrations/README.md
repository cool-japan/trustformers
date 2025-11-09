# TrustformeRS Migration System

The TrustformeRS Migration System provides comprehensive tools for migrating between different versions of TrustformeRS Serve, including configuration files, data files, and model files.

## Features

- **Full Version Migration**: Comprehensive migration between versions
- **Configuration Migration**: Migrate configuration files with validation
- **Data Migration**: Migrate data files with integrity checks
- **Model Migration**: Migrate model files with optimization options
- **Backup & Rollback**: Automatic backup and rollback capabilities
- **Validation**: Comprehensive validation before and after migration
- **Progress Tracking**: Real-time progress monitoring
- **CLI Tool**: Easy-to-use command-line interface

## Architecture

The migration system consists of several key components:

### Core Components

1. **MigrationManager**: Main coordinator for all migration operations
2. **ConfigMigrator**: Handles configuration file migrations
3. **DataMigrator**: Manages data file migrations
4. **ModelMigrator**: Handles model file migrations
5. **VersionMigrator**: Coordinates full version migrations

### Migration Types

- **Configuration Migration**: Updates configuration files to new formats
- **Data Migration**: Migrates data files with schema updates
- **Model Migration**: Updates model files and metadata
- **Version Migration**: Full system migration between versions

## Usage

### Command Line Interface

The migration system provides a CLI tool for easy migration operations:

```bash
# Full version migration
./target/release/migrate version --from 1.0.0 --to 2.0.0 --path /path/to/trustformers

# Configuration migration
./target/release/migrate config --from 1.0.0 --to 2.0.0 --config /path/to/config.toml

# Data migration
./target/release/migrate data --from 1.0.0 --to 2.0.0 --data-path /path/to/data

# Model migration
./target/release/migrate models --from 1.0.0 --to 2.0.0 --models-path /path/to/models --optimize

# Show migration information
./target/release/migrate info --from 1.0.0 --to 2.0.0

# Validate migration
./target/release/migrate validate --path /path/to/trustformers --version 2.0.0
```

### Programmatic Usage

```rust
use trustformers_serve::migration::{
    MigrationConfig, MigrationType, MigrationOptions,
    version_migration::VersionMigrator,
};

// Create version migrator
let migrator = VersionMigrator::new("1.0.0".to_string(), "2.0.0".to_string())?;

// Get migration information
let info = migrator.get_migration_info();
println!("Migration type: {:?}", info.migration_type);
println!("Estimated duration: {:?}", info.estimated_duration);

// Execute migration
let result = migrator.execute_full_migration(&base_path).await?;
match result.overall_status {
    MigrationStatus::Completed => println!("Migration successful!"),
    MigrationStatus::Failed => println!("Migration failed!"),
    MigrationStatus::RolledBack => println!("Migration rolled back!"),
    _ => println!("Migration status: {:?}", result.overall_status),
}
```

## Migration Versions

### Supported Versions

- **0.1.0**: Initial release
- **1.0.0**: First stable release
- **2.0.0**: Major architecture update
- **2.1.0**: Feature enhancements

### Version Migration Paths

The system supports the following migration paths:

```
0.1.0 -> 1.0.0 -> 2.0.0 -> 2.1.0
```

#### 0.1.0 to 1.0.0

- **Configuration**: Rename fields, add worker configuration
- **Data**: Add version fields to data files
- **Models**: Create metadata files for models
- **Breaking Changes**: None

#### 1.0.0 to 2.0.0

- **Configuration**: Restructure configuration format
- **Data**: Update schemas, convert timestamps
- **Models**: Add optimization settings, update metadata
- **Breaking Changes**: 
  - API endpoints restructured
  - Configuration format changed
  - Database schema updated

#### 2.0.0 to 2.1.0

- **Configuration**: Add streaming and rate limiting features
- **Data**: Add performance metrics and security enhancements
- **Models**: Add streaming and caching features
- **Breaking Changes**: None

## Configuration

### Migration Configuration File

The migration system uses a configuration file (`migration_config.toml`) to define migration rules and settings:

```toml
[global]
backup_before_migration = true
validate_after_migration = true
rollback_on_failure = true
timeout_seconds = 3600

[config_migration]
enabled = true
backup_original = true
format_validation = true

[data_migration]
enabled = true
backup_data = true
validate_integrity = true

[model_migration]
enabled = true
backup_models = true
validate_models = true
optimize_models = false
```

### Environment Variables

```bash
# Set migration configuration file
export TRUSTFORMERS_MIGRATION_CONFIG=/path/to/migration_config.toml

# Set log level
export RUST_LOG=info

# Set backup directory
export TRUSTFORMERS_BACKUP_DIR=/path/to/backups
```

## Migration Process

### 1. Pre-Migration

1. **Validation**: Check source version and target version compatibility
2. **Backup**: Create full backup of current installation
3. **Planning**: Generate migration plan with steps and dependencies
4. **Estimation**: Estimate migration duration and resource requirements

### 2. Migration Execution

1. **Configuration Migration**: Update configuration files
2. **Data Migration**: Migrate data files with schema updates
3. **Model Migration**: Update model files and metadata
4. **Version-Specific Changes**: Apply version-specific transformations
5. **Validation**: Validate each step after execution

### 3. Post-Migration

1. **Final Validation**: Comprehensive validation of migrated system
2. **Version Update**: Update version files and metadata
3. **Cleanup**: Remove temporary files and old backups
4. **Reporting**: Generate migration report

## Backup and Rollback

### Automatic Backup

The system automatically creates backups before migration:

```
backups/
├── full_backup_1640995200/
│   ├── config/
│   ├── data/
│   ├── models/
│   └── metadata.json
└── data_backup_1640995200/
    └── ...
```

### Rollback Process

If migration fails, the system can automatically rollback:

1. **Stop Current Services**: Gracefully stop running services
2. **Restore Backup**: Restore from backup directory
3. **Validate Rollback**: Ensure system is in working state
4. **Restart Services**: Restart services with original configuration

### Manual Rollback

```bash
# Manual rollback using CLI
./target/release/migrate rollback --backup-path /path/to/backup

# Or using the migration system
let migrator = VersionMigrator::new("2.0.0".to_string(), "1.0.0".to_string())?;
migrator.rollback_migration(&base_path, &backup_path).await?;
```

## Validation

### Pre-Migration Validation

- **Version Compatibility**: Check if migration path exists
- **System Requirements**: Verify system meets requirements
- **File Integrity**: Validate existing files
- **Configuration Syntax**: Check configuration file syntax

### Post-Migration Validation

- **File Existence**: Verify all required files exist
- **Configuration Validity**: Validate migrated configuration
- **Data Integrity**: Check data file integrity
- **Model Compatibility**: Verify model files are compatible
- **System Functionality**: Basic functionality tests

### Validation Rules

```rust
// Configuration validation
ValidationRule {
    name: "Configuration Syntax".to_string(),
    rule_type: ValidationRuleType::ConfigurationSyntax,
    severity: ValidationSeverity::Error,
    parameters: HashMap::new(),
}

// Data validation
ValidationRule {
    name: "Data Integrity".to_string(),
    rule_type: ValidationRuleType::DataIntegrity,
    severity: ValidationSeverity::Error,
    parameters: HashMap::new(),
}

// Model validation
ValidationRule {
    name: "Model Compatibility".to_string(),
    rule_type: ValidationRuleType::ModelCompatibility,
    severity: ValidationSeverity::Warning,
    parameters: HashMap::new(),
}
```

## Error Handling

### Common Errors

1. **Migration Path Not Found**: No migration path between versions
2. **Configuration Validation Failed**: Invalid configuration syntax
3. **Data Integrity Check Failed**: Corrupted data files
4. **Model Compatibility Issues**: Incompatible model format
5. **Insufficient Disk Space**: Not enough space for backup/migration
6. **Permission Denied**: Insufficient permissions for file operations

### Error Recovery

- **Automatic Rollback**: On critical failures
- **Partial Recovery**: Fix issues and resume migration
- **Manual Intervention**: For complex issues requiring user input

## Performance Considerations

### Memory Usage

- **Streaming Processing**: Process large files in chunks
- **Memory Limits**: Configurable memory limits
- **Garbage Collection**: Efficient memory management

### Disk Usage

- **Backup Storage**: Compressed backups to save space
- **Temporary Files**: Cleanup temporary files after migration
- **Space Estimation**: Pre-migration disk space checks

### Network Usage

- **Model Downloads**: Efficient model downloading
- **Progress Reporting**: Minimal network overhead
- **Remote Backups**: Optional remote backup support

## Security

### Access Control

- **File Permissions**: Preserve file permissions during migration
- **User Authentication**: Validate user permissions
- **Audit Logging**: Log all migration activities

### Data Protection

- **Encryption**: Optional encryption for sensitive data
- **Backup Security**: Secure backup storage
- **Data Validation**: Integrity checks for all data

## Testing

### Unit Tests

```bash
# Run migration tests
cargo test migration

# Run specific test
cargo test test_v1_to_v2_migration
```

### Integration Tests

```bash
# Run integration tests
cargo test --test integration_migration

# Test specific migration path
cargo test test_full_migration_1_to_2
```

### End-to-End Tests

```bash
# Run E2E tests
cargo test --test e2e_migration

# Test with real data
cargo test test_production_migration
```

## Monitoring and Observability

### Metrics

- **Migration Duration**: Time taken for each step
- **Success Rate**: Percentage of successful migrations
- **Error Rate**: Frequency of migration errors
- **Resource Usage**: CPU, memory, and disk usage

### Logging

- **Structured Logging**: JSON-formatted logs
- **Log Levels**: Debug, info, warn, error
- **Audit Trail**: Complete audit trail of migration activities

### Progress Reporting

- **Real-time Progress**: Live progress updates
- **Step-by-step Status**: Status of each migration step
- **ETA Estimation**: Estimated time to completion

## Troubleshooting

### Common Issues

1. **Migration Stuck**: Check logs for blocked operations
2. **Validation Failures**: Review validation error messages
3. **Backup Failures**: Check disk space and permissions
4. **Rollback Issues**: Verify backup integrity

### Debug Commands

```bash
# Enable debug logging
RUST_LOG=debug ./target/release/migrate ...

# Validate without migrating
./target/release/migrate validate --path /path --version 2.0.0

# Show detailed migration info
./target/release/migrate info --from 1.0.0 --to 2.0.0

# Dry run migration
./target/release/migrate version --from 1.0.0 --to 2.0.0 --path /path --dry-run
```

## Best Practices

### Before Migration

1. **Full Backup**: Always create a full backup
2. **Test Environment**: Test migration in staging environment
3. **Resource Check**: Ensure sufficient disk space and memory
4. **Maintenance Window**: Schedule migration during maintenance window
5. **Documentation**: Document current configuration and customizations

### During Migration

1. **Monitor Progress**: Watch migration progress and logs
2. **Resource Usage**: Monitor CPU, memory, and disk usage
3. **Error Handling**: Be prepared to handle errors
4. **Communication**: Keep stakeholders informed

### After Migration

1. **Validation**: Thoroughly validate migrated system
2. **Functionality Testing**: Test all critical functionality
3. **Performance Testing**: Verify performance is acceptable
4. **Cleanup**: Remove unnecessary backup files
5. **Documentation**: Update documentation with new version

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/cool-japan/trustformers
cd trustformers/trustformers-serve

# Install dependencies
cargo build

# Run tests
cargo test migration
```

### Adding New Migration Rules

1. **Create Migration Rule**: Implement migration rule trait
2. **Add Tests**: Write comprehensive tests
3. **Update Documentation**: Document new migration
4. **Version Graph**: Update migration graph

### Example Migration Rule

```rust
pub struct V2ToV3ConfigRule;

impl ConfigMigrationRule for V2ToV3ConfigRule {
    fn migrate(&self, content: &str) -> Result<String> {
        // Implementation here
    }
    
    fn get_version_range(&self) -> (String, String) {
        ("2.0.0".to_string(), "3.0.0".to_string())
    }
    
    fn get_description(&self) -> String {
        "Migrate from v2.0.0 to v3.0.0".to_string()
    }
}
```

## License

This project is licensed under the MIT OR Apache-2.0 license.

## Support

For support and questions:

- **Issues**: https://github.com/cool-japan/trustformers/issues
- **Discussions**: https://github.com/cool-japan/trustformers/discussions
- **Documentation**: https://docs.trustformers.dev

## Changelog

### v2.1.0
- Added streaming support migration
- Enhanced model optimization during migration
- Improved validation framework

### v2.0.0
- Complete migration system redesign
- Added comprehensive backup and rollback
- Introduced version migration coordinator

### v1.0.0
- Initial migration system implementation
- Basic configuration and data migration
- Command-line interface