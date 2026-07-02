use anyhow::Result;
use clap::{Arg, Command};
use std::path::PathBuf;
use trustformers_serve::migration::{
    config_migration::ConfigMigrator, data_migration::DataMigrator, model_migration::ModelMigrator,
    version_migration::VersionMigrator,
};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let matches = Command::new("trustformers-migrate")
        .version("1.0.0")
        .author("TrustformeRS Team")
        .about("TrustformeRS Migration Tool")
        .subcommand(
            Command::new("version")
                .about("Perform full version migration")
                .arg(
                    Arg::new("from")
                        .short('f')
                        .long("from")
                        .value_name("VERSION")
                        .help("Source version")
                        .required(true),
                )
                .arg(
                    Arg::new("to")
                        .short('t')
                        .long("to")
                        .value_name("VERSION")
                        .help("Target version")
                        .required(true),
                )
                .arg(
                    Arg::new("path")
                        .short('p')
                        .long("path")
                        .value_name("PATH")
                        .help("Base path for migration")
                        .required(true),
                )
                .arg(
                    Arg::new("dry-run")
                        .long("dry-run")
                        .help("Show what would be migrated without making changes")
                        .action(clap::ArgAction::SetTrue),
                ),
        )
        .subcommand(
            Command::new("config")
                .about("Migrate configuration files")
                .arg(
                    Arg::new("from")
                        .short('f')
                        .long("from")
                        .value_name("VERSION")
                        .help("Source version")
                        .required(true),
                )
                .arg(
                    Arg::new("to")
                        .short('t')
                        .long("to")
                        .value_name("VERSION")
                        .help("Target version")
                        .required(true),
                )
                .arg(
                    Arg::new("config")
                        .short('c')
                        .long("config")
                        .value_name("FILE")
                        .help("Configuration file path")
                        .required(true),
                ),
        )
        .subcommand(
            Command::new("data")
                .about("Migrate data files")
                .arg(
                    Arg::new("from")
                        .short('f')
                        .long("from")
                        .value_name("VERSION")
                        .help("Source version")
                        .required(true),
                )
                .arg(
                    Arg::new("to")
                        .short('t')
                        .long("to")
                        .value_name("VERSION")
                        .help("Target version")
                        .required(true),
                )
                .arg(
                    Arg::new("data-path")
                        .short('d')
                        .long("data-path")
                        .value_name("PATH")
                        .help("Data directory path")
                        .required(true),
                ),
        )
        .subcommand(
            Command::new("models")
                .about("Migrate model files")
                .arg(
                    Arg::new("from")
                        .short('f')
                        .long("from")
                        .value_name("VERSION")
                        .help("Source version")
                        .required(true),
                )
                .arg(
                    Arg::new("to")
                        .short('t')
                        .long("to")
                        .value_name("VERSION")
                        .help("Target version")
                        .required(true),
                )
                .arg(
                    Arg::new("models-path")
                        .short('m')
                        .long("models-path")
                        .value_name("PATH")
                        .help("Models directory path")
                        .required(true),
                )
                .arg(
                    Arg::new("optimize")
                        .long("optimize")
                        .help("Optimize models during migration")
                        .action(clap::ArgAction::SetTrue),
                ),
        )
        .subcommand(
            Command::new("info")
                .about("Show migration information")
                .arg(
                    Arg::new("from")
                        .short('f')
                        .long("from")
                        .value_name("VERSION")
                        .help("Source version")
                        .required(true),
                )
                .arg(
                    Arg::new("to")
                        .short('t')
                        .long("to")
                        .value_name("VERSION")
                        .help("Target version")
                        .required(true),
                ),
        )
        .subcommand(
            Command::new("validate")
                .about("Validate migration without performing it")
                .arg(
                    Arg::new("path")
                        .short('p')
                        .long("path")
                        .value_name("PATH")
                        .help("Path to validate")
                        .required(true),
                )
                .arg(
                    Arg::new("version")
                        .short('v')
                        .long("version")
                        .value_name("VERSION")
                        .help("Expected version")
                        .required(true),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("version", sub_matches)) => {
            let from = sub_matches
                .get_one::<String>("from")
                .ok_or_else(|| anyhow::anyhow!("from argument is required"))?;
            let to = sub_matches
                .get_one::<String>("to")
                .ok_or_else(|| anyhow::anyhow!("to argument is required"))?;
            let path = sub_matches
                .get_one::<String>("path")
                .ok_or_else(|| anyhow::anyhow!("path argument is required"))?;
            let dry_run = sub_matches.get_flag("dry-run");

            migrate_version(from, to, path, dry_run).await?;
        },
        Some(("config", sub_matches)) => {
            let from = sub_matches
                .get_one::<String>("from")
                .ok_or_else(|| anyhow::anyhow!("from argument is required"))?;
            let to = sub_matches
                .get_one::<String>("to")
                .ok_or_else(|| anyhow::anyhow!("to argument is required"))?;
            let config_file = sub_matches
                .get_one::<String>("config")
                .ok_or_else(|| anyhow::anyhow!("config argument is required"))?;

            migrate_config(from, to, config_file).await?;
        },
        Some(("data", sub_matches)) => {
            let from = sub_matches
                .get_one::<String>("from")
                .ok_or_else(|| anyhow::anyhow!("from argument is required"))?;
            let to = sub_matches
                .get_one::<String>("to")
                .ok_or_else(|| anyhow::anyhow!("to argument is required"))?;
            let data_path = sub_matches
                .get_one::<String>("data-path")
                .ok_or_else(|| anyhow::anyhow!("data-path argument is required"))?;

            migrate_data(from, to, data_path).await?;
        },
        Some(("models", sub_matches)) => {
            let from = sub_matches
                .get_one::<String>("from")
                .ok_or_else(|| anyhow::anyhow!("from argument is required"))?;
            let to = sub_matches
                .get_one::<String>("to")
                .ok_or_else(|| anyhow::anyhow!("to argument is required"))?;
            let models_path = sub_matches
                .get_one::<String>("models-path")
                .ok_or_else(|| anyhow::anyhow!("models-path argument is required"))?;
            let optimize = sub_matches.get_flag("optimize");

            migrate_models(from, to, models_path, optimize).await?;
        },
        Some(("info", sub_matches)) => {
            let from = sub_matches
                .get_one::<String>("from")
                .ok_or_else(|| anyhow::anyhow!("from argument is required"))?;
            let to = sub_matches
                .get_one::<String>("to")
                .ok_or_else(|| anyhow::anyhow!("to argument is required"))?;

            show_migration_info(from, to).await?;
        },
        Some(("validate", sub_matches)) => {
            let path = sub_matches
                .get_one::<String>("path")
                .ok_or_else(|| anyhow::anyhow!("path argument is required"))?;
            let version = sub_matches
                .get_one::<String>("version")
                .ok_or_else(|| anyhow::anyhow!("version argument is required"))?;

            validate_migration(path, version).await?;
        },
        _ => {
            println!("No subcommand provided. Use --help for usage information.");
        },
    }

    Ok(())
}

async fn migrate_version(from: &str, to: &str, path: &str, dry_run: bool) -> Result<()> {
    println!("🔄 Starting full version migration from {} to {}", from, to);

    if dry_run {
        println!("🧪 DRY RUN MODE - No changes will be made");
    }

    let migrator = VersionMigrator::new(from.to_string(), to.to_string())?;
    let migration_info = migrator.get_migration_info();

    println!("📋 Migration Information:");
    println!("   Type: {:?}", migration_info.migration_type);
    println!(
        "   Estimated Duration: {:?}",
        migration_info.estimated_duration
    );

    if !migration_info.breaking_changes.is_empty() {
        println!("⚠️  Breaking Changes:");
        for change in &migration_info.breaking_changes {
            println!("   - {}", change);
        }
    }

    if !migration_info.required_actions.is_empty() {
        println!("📝 Required Actions:");
        for action in &migration_info.required_actions {
            println!("   - {}", action);
        }
    }

    if dry_run {
        println!("✅ Dry run completed. No changes were made.");
        return Ok(());
    }

    // Ask for confirmation
    println!("\n❓ Do you want to proceed with the migration? (y/N):");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    if input.trim().to_lowercase() != "y" {
        println!("❌ Migration cancelled.");
        return Ok(());
    }

    let path = PathBuf::from(path);
    let result = migrator.execute_full_migration(&path).await?;

    println!("\n📊 Migration Results:");
    println!("   Status: {:?}", result.overall_status);
    println!(
        "   Started: {}",
        result.started_at.format("%Y-%m-%d %H:%M:%S")
    );
    if let Some(completed) = result.completed_at {
        println!("   Completed: {}", completed.format("%Y-%m-%d %H:%M:%S"));
        let duration = completed.signed_duration_since(result.started_at);
        println!("   Duration: {} seconds", duration.num_seconds());
    }

    println!("\n🔧 Migration Steps:");
    for step in &result.step_results {
        println!(
            "   Step {}: {} -> {} ({:?})",
            step.step_number, step.from_version, step.to_version, step.status
        );
        if let Some(error) = &step.error {
            println!("      Error: {}", error);
        }
        if !step.changes_applied.is_empty() {
            println!("      Changes: {} applied", step.changes_applied.len());
        }
    }

    if let Some(rollback_info) = &result.rollback_info {
        println!("\n💾 Backup Information:");
        println!("   Location: {}", rollback_info.backup_path);
        println!("   Size: {} bytes", rollback_info.backup_size);
        println!(
            "   Created: {}",
            rollback_info.backup_timestamp.format("%Y-%m-%d %H:%M:%S")
        );
    }

    match result.overall_status {
        trustformers_serve::migration::MigrationStatus::Completed => {
            println!("\n✅ Migration completed successfully!");
        },
        trustformers_serve::migration::MigrationStatus::Failed => {
            println!("\n❌ Migration failed. Check the logs for details.");
        },
        trustformers_serve::migration::MigrationStatus::RolledBack => {
            println!("\n⏪ Migration failed and was rolled back.");
        },
        _ => {
            println!(
                "\n❓ Migration ended with status: {:?}",
                result.overall_status
            );
        },
    }

    Ok(())
}

async fn migrate_config(from: &str, to: &str, config_file: &str) -> Result<()> {
    println!("🔧 Migrating configuration from {} to {}", from, to);

    let migrator = ConfigMigrator::new(from.to_string(), to.to_string());
    let config_path = PathBuf::from(config_file);

    // Validate configuration before migration
    let content = tokio::fs::read_to_string(&config_path).await?;
    let validation_results = migrator.validate_config(&content)?;

    println!("📋 Configuration Validation:");
    for result in &validation_results {
        let status_icon = match result.status {
            trustformers_serve::migration::ValidationStatus::Passed => "✅",
            trustformers_serve::migration::ValidationStatus::Failed => "❌",
            trustformers_serve::migration::ValidationStatus::Warning => "⚠️",
            trustformers_serve::migration::ValidationStatus::Skipped => "⏭️",
        };
        println!(
            "   {} {}: {}",
            status_icon, result.rule_name, result.message
        );
    }

    // Check if validation passed
    if validation_results
        .iter()
        .any(|r| r.status == trustformers_serve::migration::ValidationStatus::Failed)
    {
        println!("❌ Configuration validation failed. Please fix the issues and try again.");
        return Ok(());
    }

    // Perform migration
    migrator.migrate_config_file(&config_path).await?;

    // Generate migration report
    let migrated_content = tokio::fs::read_to_string(&config_path).await?;
    let report = migrator.generate_migration_report(&content, &migrated_content);

    println!("\n📊 Migration Report:");
    println!("   Changes Applied: {}", report.changes.len());
    for change in &report.changes {
        println!("   - {}: {}", change.change_type, change.description);
    }

    println!("\n✅ Configuration migration completed successfully!");

    Ok(())
}

async fn migrate_data(from: &str, to: &str, data_path: &str) -> Result<()> {
    println!("💾 Migrating data from {} to {}", from, to);

    let migrator = DataMigrator::new(from.to_string(), to.to_string());
    let data_path = PathBuf::from(data_path);

    let result = migrator.migrate_data_directory(&data_path).await?;

    println!("\n📊 Data Migration Results:");
    println!("   Total Files: {}", result.statistics.total_files);
    println!("   Migrated Files: {}", result.statistics.migrated_files);
    println!("   Failed Files: {}", result.statistics.failed_files);

    if !result.errors.is_empty() {
        println!("\n❌ Errors:");
        for error in &result.errors {
            println!("   - {}: {}", error.file_path, error.error);
        }
    }

    if !result.migrated_files.is_empty() {
        println!("\n📁 Migrated Files:");
        for file in &result.migrated_files {
            println!(
                "   - {}: {} -> {} bytes",
                file.file_path, file.original_size, file.migrated_size
            );
            if !file.changes.is_empty() {
                for change in &file.changes {
                    println!("      - {}: {}", change.change_type, change.description);
                }
            }
        }
    }

    println!("\n✅ Data migration completed!");

    Ok(())
}

async fn migrate_models(from: &str, to: &str, models_path: &str, optimize: bool) -> Result<()> {
    println!("🤖 Migrating models from {} to {}", from, to);

    let migrator = ModelMigrator::new(from.to_string(), to.to_string());
    let models_path = PathBuf::from(models_path);

    let result = migrator.migrate_models_directory(&models_path).await?;

    println!("\n📊 Model Migration Results:");
    println!("   Total Models: {}", result.statistics.total_models);
    println!("   Migrated Models: {}", result.statistics.migrated_models);
    println!("   Failed Models: {}", result.statistics.failed_models);

    if !result.errors.is_empty() {
        println!("\n❌ Errors:");
        for error in &result.errors {
            println!("   - {}: {}", error.model_path, error.error);
        }
    }

    if optimize {
        println!("\n🔧 Optimizing models...");
        let optimization_result = migrator.optimize_models(&models_path).await?;

        println!("📈 Optimization Results:");
        println!(
            "   Total Size Before: {} bytes",
            optimization_result.total_size_before
        );
        println!(
            "   Total Size After: {} bytes",
            optimization_result.total_size_after
        );
        let savings = optimization_result.total_size_before - optimization_result.total_size_after;
        let savings_percent =
            (savings as f64 / optimization_result.total_size_before as f64) * 100.0;
        println!(
            "   Space Saved: {} bytes ({:.1}%)",
            savings, savings_percent
        );
        println!(
            "   Optimization Time: {:?}",
            optimization_result.optimization_time
        );
    }

    println!("\n✅ Model migration completed!");

    Ok(())
}

async fn show_migration_info(from: &str, to: &str) -> Result<()> {
    println!("📋 Migration Information: {} -> {}", from, to);

    let migrator = VersionMigrator::new(from.to_string(), to.to_string())?;
    let info = migrator.get_migration_info();

    println!("\n🔍 Migration Details:");
    println!("   Type: {:?}", info.migration_type);
    println!("   Estimated Duration: {:?}", info.estimated_duration);

    if !info.breaking_changes.is_empty() {
        println!("\n⚠️  Breaking Changes:");
        for change in &info.breaking_changes {
            println!("   - {}", change);
        }
    }

    if !info.required_actions.is_empty() {
        println!("\n📝 Required Actions:");
        for action in &info.required_actions {
            println!("   - {}", action);
        }
    }

    // Show supported migration paths
    println!("\n🛣️  Migration Path:");
    let migration_graph = trustformers_serve::migration::version_migration::MigrationGraph::new();
    match migration_graph.find_migration_path(from, to) {
        Ok(path) => {
            for (i, version) in path.iter().enumerate() {
                if i > 0 {
                    print!(" -> ");
                }
                print!("{}", version);
            }
            println!();
        },
        Err(e) => {
            println!("   ❌ No migration path found: {}", e);
        },
    }

    Ok(())
}

async fn validate_migration(path: &str, version: &str) -> Result<()> {
    println!(
        "🔍 Validating migration at {} for version {}",
        path, version
    );

    let migrator = VersionMigrator::new(version.to_string(), version.to_string())?;
    let path = PathBuf::from(path);

    let validation_results = migrator.validate_migration_step(&path, version).await?;

    println!("\n📊 Validation Results:");
    let mut passed = 0;
    let mut failed = 0;
    let mut warnings = 0;

    for result in &validation_results {
        let status_icon = match result.status {
            trustformers_serve::migration::ValidationStatus::Passed => {
                passed += 1;
                "✅"
            },
            trustformers_serve::migration::ValidationStatus::Failed => {
                failed += 1;
                "❌"
            },
            trustformers_serve::migration::ValidationStatus::Warning => {
                warnings += 1;
                "⚠️"
            },
            trustformers_serve::migration::ValidationStatus::Skipped => "⏭️",
        };
        println!(
            "   {} {}: {}",
            status_icon, result.rule_name, result.message
        );
    }

    println!("\n📈 Summary:");
    println!("   ✅ Passed: {}", passed);
    println!("   ❌ Failed: {}", failed);
    println!("   ⚠️  Warnings: {}", warnings);

    if failed > 0 {
        println!("\n❌ Validation failed! Please fix the issues before proceeding.");
    } else if warnings > 0 {
        println!("\n⚠️  Validation passed with warnings. Consider addressing the warnings.");
    } else {
        println!("\n✅ Validation passed successfully!");
    }

    Ok(())
}
