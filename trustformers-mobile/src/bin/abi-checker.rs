//! ABI Compatibility Checker CLI Tool
//!
//! This tool provides command-line interface for checking ABI compatibility
//! between different versions of TrustformeRS Mobile API.

use clap::{Arg, Command};
use std::process;
use trustformers_mobile::abi_checker::{AbiChecker, AbiVersion};

fn main() {
    let matches = Command::new("TrustformeRS Mobile ABI Checker")
        .version("0.1.0")
        .author("TrustformeRS Team")
        .about("Checks ABI compatibility between TrustformeRS Mobile API versions")
        .subcommand(
            Command::new("generate")
                .about("Generate ABI specification from current codebase")
                .arg(
                    Arg::new("output")
                        .short('o')
                        .long("output")
                        .value_name("FILE")
                        .help("Output file for ABI specification")
                        .required(true),
                )
                .arg(
                    Arg::new("version")
                        .short('v')
                        .long("version")
                        .value_name("VERSION")
                        .help("Version string (e.g., 1.0.0)")
                        .default_value("0.1.0"),
                ),
        )
        .subcommand(
            Command::new("check")
                .about("Check compatibility between baseline and current ABI")
                .arg(
                    Arg::new("baseline")
                        .short('b')
                        .long("baseline")
                        .value_name("FILE")
                        .help("Baseline ABI specification file")
                        .required(true),
                )
                .arg(
                    Arg::new("current").short('c').long("current").value_name("FILE").help(
                        "Current ABI specification file (optional, will generate if not provided)",
                    ),
                )
                .arg(
                    Arg::new("format")
                        .short('f')
                        .long("format")
                        .value_name("FORMAT")
                        .help("Output format: json, text")
                        .default_value("text"),
                )
                .arg(
                    Arg::new("output")
                        .short('o')
                        .long("output")
                        .value_name("FILE")
                        .help("Output file for compatibility report (default: stdout)"),
                ),
        )
        .subcommand(
            Command::new("diff")
                .about("Show detailed differences between two ABI specifications")
                .arg(
                    Arg::new("old")
                        .value_name("OLD_SPEC")
                        .help("Path to old ABI specification")
                        .required(true)
                        .index(1),
                )
                .arg(
                    Arg::new("new")
                        .value_name("NEW_SPEC")
                        .help("Path to new ABI specification")
                        .required(true)
                        .index(2),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("generate", sub_matches)) => {
            if let Err(e) = generate_specification(sub_matches) {
                eprintln!("Error generating ABI specification: {}", e);
                process::exit(1);
            }
        },
        Some(("check", sub_matches)) => {
            if let Err(e) = check_compatibility(sub_matches) {
                eprintln!("Error checking ABI compatibility: {}", e);
                process::exit(1);
            }
        },
        Some(("diff", sub_matches)) => {
            if let Err(e) = show_diff(sub_matches) {
                eprintln!("Error showing ABI diff: {}", e);
                process::exit(1);
            }
        },
        _ => {
            eprintln!("No subcommand provided. Use --help for usage information.");
            process::exit(1);
        },
    }
}

fn generate_specification(matches: &clap::ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let output_file = matches.get_one::<String>("output").unwrap();
    let version_str = matches.get_one::<String>("version").unwrap();

    // Parse version string
    let version = parse_version(version_str)?;

    println!(
        "Generating ABI specification for version {}.{}.{}",
        version.major, version.minor, version.patch
    );

    let checker = AbiChecker::new();
    let mut spec = checker.generate_current_specification()?;
    spec.version = version;

    checker.save_specification(&spec, output_file)?;

    println!("ABI specification saved to: {}", output_file);
    println!("Functions: {}", spec.functions.len());
    println!("Types: {}", spec.types.len());
    println!("Constants: {}", spec.constants.len());

    Ok(())
}

fn check_compatibility(matches: &clap::ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let baseline_file = matches.get_one::<String>("baseline").unwrap();
    let format = matches.get_one::<String>("format").unwrap();

    let mut checker = AbiChecker::new();
    checker.load_baseline_from_file(baseline_file)?;

    // Get current specification
    let current_spec = if let Some(current_file) = matches.get_one::<String>("current") {
        // Load from file
        let content = std::fs::read_to_string(current_file)?;
        serde_json::from_str(&content)?
    } else {
        // Generate from current codebase
        checker.generate_current_specification()?
    };

    let result = checker.check_compatibility(&current_spec)?;

    // Format and output result
    let output = match format.as_str() {
        "json" => serde_json::to_string_pretty(&result)?,
        "text" => format_text_report(&result),
        _ => return Err("Invalid format. Use 'json' or 'text'.".into()),
    };

    if let Some(output_file) = matches.get_one::<String>("output") {
        std::fs::write(output_file, output)?;
        println!("Compatibility report written to: {}", output_file);
    } else {
        println!("{}", output);
    }

    // Exit with error code if incompatible
    if !result.is_compatible {
        process::exit(1);
    }

    Ok(())
}

fn show_diff(matches: &clap::ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let old_file = matches.get_one::<String>("old").unwrap();
    let new_file = matches.get_one::<String>("new").unwrap();

    // Load specifications
    let old_content = std::fs::read_to_string(old_file)?;
    let old_spec: trustformers_mobile::abi_checker::AbiSpecification =
        serde_json::from_str(&old_content)?;

    let new_content = std::fs::read_to_string(new_file)?;
    let new_spec: trustformers_mobile::abi_checker::AbiSpecification =
        serde_json::from_str(&new_content)?;

    // Perform comparison
    let mut checker = AbiChecker::new();
    checker.set_baseline(old_spec.clone());
    let result = checker.check_compatibility(&new_spec)?;

    // Show detailed diff
    println!("ABI Specification Comparison");
    println!("============================");
    println!(
        "Old version: {}.{}.{}",
        old_spec.version.major, old_spec.version.minor, old_spec.version.patch
    );
    println!(
        "New version: {}.{}.{}",
        new_spec.version.major, new_spec.version.minor, new_spec.version.patch
    );
    println!();

    if result.is_compatible {
        println!("✅ ABI COMPATIBLE");
    } else {
        println!("❌ ABI INCOMPATIBLE");
    }
    println!();

    if !result.breaking_changes.is_empty() {
        println!("Breaking Changes ({}):", result.breaking_changes.len());
        println!("-------------------");
        for change in &result.breaking_changes {
            println!(
                "• {} [{}]",
                change.description,
                format_severity(&change.severity)
            );
            println!("  Symbol: {}", change.affected_symbol);
            println!();
        }
    }

    if !result.warnings.is_empty() {
        println!("Warnings ({}):", result.warnings.len());
        println!("----------");
        for warning in &result.warnings {
            println!("• {}", warning.message);
            println!("  Symbol: {}", warning.symbol);
            println!("  Recommendation: {}", warning.recommendation);
            println!();
        }
    }

    if !result.added_functions.is_empty() {
        println!("Added Functions ({}):", result.added_functions.len());
        println!("----------------");
        for func in &result.added_functions {
            println!("• {}", func);
        }
        println!();
    }

    if !result.added_types.is_empty() {
        println!("Added Types ({}):", result.added_types.len());
        println!("------------");
        for type_name in &result.added_types {
            println!("• {}", type_name);
        }
        println!();
    }

    Ok(())
}

fn parse_version(version_str: &str) -> Result<AbiVersion, Box<dyn std::error::Error>> {
    let parts: Vec<&str> = version_str.split('.').collect();
    if parts.len() != 3 {
        return Err("Version must be in format major.minor.patch (e.g., 1.0.0)".into());
    }

    let major = parts[0].parse::<u32>()?;
    let minor = parts[1].parse::<u32>()?;
    let patch = parts[2].parse::<u32>()?;

    Ok(AbiVersion::new(major, minor, patch))
}

fn format_text_report(result: &trustformers_mobile::abi_checker::CompatibilityResult) -> String {
    let mut output = String::new();

    output.push_str("ABI Compatibility Report\n");
    output.push_str("========================\n\n");

    if result.is_compatible {
        output.push_str("✅ Status: COMPATIBLE\n\n");
    } else {
        output.push_str("❌ Status: INCOMPATIBLE\n\n");
    }

    if !result.breaking_changes.is_empty() {
        output.push_str(&format!(
            "Breaking Changes ({}):\n",
            result.breaking_changes.len()
        ));
        output.push_str("----------------------\n");
        for (i, change) in result.breaking_changes.iter().enumerate() {
            output.push_str(&format!(
                "{}. {} [{}]\n",
                i + 1,
                change.description,
                format_severity(&change.severity)
            ));
            output.push_str(&format!("   Symbol: {}\n", change.affected_symbol));
            output.push_str(&format!("   Type: {:?}\n\n", change.change_type));
        }
    }

    if !result.warnings.is_empty() {
        output.push_str(&format!("Warnings ({}):\n", result.warnings.len()));
        output.push_str("-------------\n");
        for (i, warning) in result.warnings.iter().enumerate() {
            output.push_str(&format!("{}. {}\n", i + 1, warning.message));
            output.push_str(&format!("   Symbol: {}\n", warning.symbol));
            output.push_str(&format!(
                "   Recommendation: {}\n\n",
                warning.recommendation
            ));
        }
    }

    if !result.added_functions.is_empty() {
        output.push_str(&format!(
            "Added Functions ({}):\n",
            result.added_functions.len()
        ));
        output.push_str("-------------------\n");
        for func in &result.added_functions {
            output.push_str(&format!("• {}\n", func));
        }
        output.push('\n');
    }

    if !result.added_types.is_empty() {
        output.push_str(&format!("Added Types ({}):\n", result.added_types.len()));
        output.push_str("-------------\n");
        for type_name in &result.added_types {
            output.push_str(&format!("• {}\n", type_name));
        }
        output.push('\n');
    }

    output.push_str("Summary:\n");
    output.push_str("--------\n");
    output.push_str(&format!(
        "Breaking changes: {}\n",
        result.breaking_changes.len()
    ));
    output.push_str(&format!("Warnings: {}\n", result.warnings.len()));
    output.push_str(&format!(
        "Added functions: {}\n",
        result.added_functions.len()
    ));
    output.push_str(&format!("Added types: {}\n", result.added_types.len()));

    output
}

fn format_severity(severity: &trustformers_mobile::abi_checker::Severity) -> &'static str {
    match severity {
        trustformers_mobile::abi_checker::Severity::Critical => "CRITICAL",
        trustformers_mobile::abi_checker::Severity::High => "HIGH",
        trustformers_mobile::abi_checker::Severity::Medium => "MEDIUM",
        trustformers_mobile::abi_checker::Severity::Low => "LOW",
    }
}
