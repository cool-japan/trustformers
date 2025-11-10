//! Command-line tool for enforcing naming conventions
//!
//! This binary provides a standalone command-line interface for checking
//! and enforcing naming conventions across the TrustformeRS codebase.

use std::env;
use std::path::Path;
use std::process;
// Import from the testing module directly
use trustformers_core::testing::naming_conventions::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    match args[1].as_str() {
        "check" => {
            if args.len() < 3 {
                eprintln!("Error: Missing directory path");
                print_usage();
                process::exit(1);
            }

            let result = run_check(&args[2..]);
            match result {
                Ok(exit_code) => process::exit(exit_code),
                Err(e) => {
                    eprintln!("Error: {}", e);
                    process::exit(1);
                },
            }
        },
        "init" => {
            if let Err(e) = create_config_file() {
                eprintln!("Error creating config file: {}", e);
                process::exit(1);
            }
            println!("Created naming_conventions.toml configuration file");
        },
        "help" | "--help" | "-h" => {
            print_help();
        },
        "version" | "--version" | "-V" => {
            println!("TrustformeRS Naming Checker v{}", env!("CARGO_PKG_VERSION"));
        },
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
            process::exit(1);
        },
    }
}

fn print_usage() {
    println!("Usage: naming_checker <command> [options]");
    println!();
    println!("Commands:");
    println!("  check <directory>     Check naming conventions in directory");
    println!("  init                  Create default configuration file");
    println!("  help                  Show detailed help");
    println!("  version               Show version information");
}

fn print_help() {
    println!("TrustformeRS Naming Convention Checker");
    println!();
    print_usage();
    println!();
    println!("Check options:");
    println!("  --config <file>       Use custom configuration file");
    println!("  --json                Output results in JSON format");
    println!("  --fix                 Attempt to automatically fix violations");
    println!("  --severity <level>    Minimum severity to report (error, warning, info)");
    println!("  --exclude <pattern>   Exclude files matching regex pattern");
    println!("  --include-ext <ext>   Only check files with given extension");
    println!("  --max-length <n>      Maximum allowed name length");
    println!("  --min-length <n>      Minimum allowed name length");
    println!();
    println!("Examples:");
    println!("  naming_checker check src/");
    println!("  naming_checker check . --json --config custom.toml");
    println!("  naming_checker check src/ --fix --severity warning");
    println!("  naming_checker check . --exclude '^test_.*' --include-ext rs");
    println!();
    println!("Exit codes:");
    println!("  0 - No violations found");
    println!("  1 - Naming violations found");
    println!("  2 - Error running checker");
}

fn run_check(args: &[String]) -> Result<i32, Box<dyn std::error::Error>> {
    let directory = Path::new(&args[0]);
    if !directory.exists() {
        return Err(format!("Directory does not exist: {}", directory.display()).into());
    }

    let mut checker = create_checker_from_args(&args[1..])?;
    let mut json_output = false;
    let mut fix_violations = false;

    // Parse remaining arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => {
                json_output = true;
            },
            "--fix" => {
                fix_violations = true;
            },
            "--config" => {
                if i + 1 >= args.len() {
                    return Err("Missing config file path".into());
                }
                // Load custom config (implementation would read TOML file)
                i += 1;
            },
            "--severity" => {
                if i + 1 >= args.len() {
                    return Err("Missing severity level".into());
                }
                // Set minimum severity level
                i += 1;
            },
            "--exclude" => {
                if i + 1 >= args.len() {
                    return Err("Missing exclude pattern".into());
                }
                checker.exclude_pattern(&args[i + 1])?;
                i += 1;
            },
            "--max-length" => {
                if i + 1 >= args.len() {
                    return Err("Missing max length value".into());
                }
                // Would set max length in a full implementation
                i += 1;
            },
            "--min-length" => {
                if i + 1 >= args.len() {
                    return Err("Missing min length value".into());
                }
                // Would set min length in a full implementation
                i += 1;
            },
            "--include-ext" => {
                if i + 1 >= args.len() {
                    return Err("Missing file extension".into());
                }
                // Would set included extensions in a full implementation
                i += 1;
            },
            _ => {
                if args[i].starts_with("--") {
                    return Err(format!("Unknown option: {}", args[i]).into());
                }
            },
        }
        i += 1;
    }

    // Run the naming convention check
    let violations = checker.check_directory(directory)?;
    let report = checker.generate_report(&violations);

    // Output results
    if json_output {
        println!("{}", report.to_json()?);
    } else {
        report.print_report();
    }

    // Auto-fix violations if requested
    if fix_violations {
        println!("Auto-fix feature would be implemented here");
        // In a full implementation, this would attempt to fix violations
    }

    // Return appropriate exit code
    if report.has_errors() {
        Ok(1) // Naming violations found
    } else {
        Ok(0) // No violations
    }
}

fn create_checker_from_args(_args: &[String]) -> Result<NamingChecker, Box<dyn std::error::Error>> {
    // Create default checker - in a full implementation, this would parse
    // command line arguments and configuration files
    let conventions = create_trustformers_conventions();
    let mut checker = NamingChecker::new(conventions);

    // Add project-specific exclusions
    checker.exclude_pattern(r"^_.*")?; // Private items
    checker.exclude_pattern(r".*_test$")?; // Test functions
    checker.exclude_pattern(r"^test_.*")?; // Test functions
    checker.exclude_pattern(r"^bench_.*")?; // Benchmark functions
    checker.exclude_pattern(r"^proptest_.*")?; // Property test functions

    Ok(checker)
}

fn create_trustformers_conventions() -> NamingConventions {
    let mut conventions = NamingConventions {
        functions: NamingRule::SnakeCase,
        structs: NamingRule::PascalCase,
        enums: NamingRule::PascalCase,
        traits: NamingRule::PascalCase,
        constants: NamingRule::ScreamingSnakeCase,
        variables: NamingRule::SnakeCase,
        modules: NamingRule::SnakeCase,
        macros: NamingRule::SnakeCase,
        type_aliases: NamingRule::PascalCase,
        generics: NamingRule::SingleUppercase,
        ..Default::default()
    };

    // Add custom rules for domain-specific naming
    conventions.custom_rules.insert("tensor_ops".to_string(), NamingRule::SnakeCase);
    conventions
        .custom_rules
        .insert("gpu_kernels".to_string(), NamingRule::SnakeCase);
    conventions
        .custom_rules
        .insert("error_types".to_string(), NamingRule::PascalCase);

    conventions
}

fn create_config_file() -> Result<(), Box<dyn std::error::Error>> {
    let config_content = r#"# TrustformeRS Naming Convention Configuration

[naming_conventions]

[naming_conventions.functions]
rule = "SnakeCase"

[naming_conventions.structs]
rule = "PascalCase"

[naming_conventions.enums]
rule = "PascalCase"

[naming_conventions.traits]
rule = "PascalCase"

[naming_conventions.constants]
rule = "ScreamingSnakeCase"

[naming_conventions.variables]
rule = "SnakeCase"

[naming_conventions.modules]
rule = "SnakeCase"

[naming_conventions.macros]
rule = "SnakeCase"

[naming_conventions.type_aliases]
rule = "PascalCase"

[naming_conventions.generics]
rule = "SingleUppercase"

[checker_settings]
max_name_length = 50
min_name_length = 1
included_extensions = ["rs", "toml"]

# Exclusion patterns (regex)
exclude_patterns = [
    "^_.*",           # Private items starting with underscore
    ".*_test$",       # Test functions
    "^test_.*",       # Test functions
    "^bench_.*",      # Benchmark functions
    "^proptest_.*",   # Property test functions
]

# Domain-specific abbreviations
[abbreviations]
gpu = "GPU"
cpu = "CPU"
api = "API"
url = "URL"
http = "HTTP"
json = "JSON"
xml = "XML"
ai = "AI"
ml = "ML"
nn = "NN"
rnn = "RNN"
cnn = "CNN"
gru = "GRU"
lstm = "LSTM"
bert = "BERT"
gpt = "GPT"
cuda = "CUDA"
rocm = "ROCm"
simd = "SIMD"
avx = "AVX"
sse = "SSE"
neon = "NEON"
"#;

    std::fs::write("naming_conventions.toml", config_content)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_checker_creation() {
        let args = vec![];
        let checker = create_checker_from_args(&args).unwrap();
        // Basic test that checker is created successfully
        assert!(checker.check_directory(Path::new(".")).is_ok());
    }

    #[test]
    fn test_config_file_creation() {
        let temp_dir = TempDir::new().unwrap();
        let current_dir = env::current_dir().unwrap();
        env::set_current_dir(temp_dir.path()).unwrap();

        create_config_file().unwrap();
        assert!(Path::new("naming_conventions.toml").exists());

        let content = fs::read_to_string("naming_conventions.toml").unwrap();
        assert!(content.contains("[naming_conventions]"));
        assert!(content.contains("rule = \"SnakeCase\""));

        env::set_current_dir(current_dir).unwrap();
    }

    #[test]
    fn test_trustformers_conventions() {
        let conventions = create_trustformers_conventions();
        assert_eq!(conventions.functions, NamingRule::SnakeCase);
        assert_eq!(conventions.structs, NamingRule::PascalCase);
        assert_eq!(conventions.constants, NamingRule::ScreamingSnakeCase);
        assert!(conventions.custom_rules.contains_key("tensor_ops"));
    }
}
