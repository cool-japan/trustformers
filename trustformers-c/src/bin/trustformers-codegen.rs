//! CLI tool for generating language bindings for TrustformeRS
//!
//! This tool analyzes the Rust FFI interface and generates language bindings
//! for various programming languages including Python, Java, Go, TypeScript, etc.

#[cfg(feature = "codegen")]
use clap::{Arg, ArgAction, Command, ValueEnum};
#[cfg(feature = "codegen")]
use std::collections::HashMap;
#[cfg(feature = "codegen")]
use std::path::PathBuf;
#[cfg(feature = "codegen")]
use trustformers_c::codegen::*;

#[cfg(feature = "codegen")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Language {
    Python,
    Java,
    Go,
    #[value(name = "csharp")]
    CSharp,
    #[value(name = "typescript")]
    TypeScript,
    #[value(name = "javascript")]
    JavaScript,
    Kotlin,
    Swift,
    Ruby,
    #[value(name = "php")]
    PHP,
    All,
}

#[cfg(feature = "codegen")]
impl From<Language> for TargetLanguage {
    fn from(lang: Language) -> Self {
        match lang {
            Language::Python => TargetLanguage::Python,
            Language::Java => TargetLanguage::Java,
            Language::Go => TargetLanguage::Go,
            Language::CSharp => TargetLanguage::CSharp,
            Language::TypeScript => TargetLanguage::TypeScript,
            Language::JavaScript => TargetLanguage::JavaScript,
            Language::Kotlin => TargetLanguage::Kotlin,
            Language::Swift => TargetLanguage::Swift,
            Language::Ruby => TargetLanguage::Ruby,
            Language::PHP => TargetLanguage::PHP,
            Language::All => unreachable!("All should be handled separately"),
        }
    }
}

#[cfg(feature = "codegen")]
fn main() -> anyhow::Result<()> {
    let matches = Command::new("trustformers-codegen")
        .version("0.1.0")
        .author("TrustformeRS Team <team@trustformers.ai>")
        .about("Generates language bindings for TrustformeRS from Rust FFI interfaces")
        .arg(
            Arg::new("source")
                .long("source")
                .short('s')
                .value_name("DIR")
                .help("Source directory to scan for FFI interfaces")
                .default_value("src"),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .short('o')
                .value_name("DIR")
                .help("Output directory for generated bindings")
                .default_value("generated"),
        )
        .arg(
            Arg::new("languages")
                .long("languages")
                .short('l')
                .value_name("LANG")
                .help("Target languages to generate bindings for")
                .action(ArgAction::Append)
                .default_values(["python", "typescript"]),
        )
        .arg(
            Arg::new("package-name")
                .long("package-name")
                .value_name("NAME")
                .help("Package name for generated bindings")
                .default_value("trustformers"),
        )
        .arg(
            Arg::new("package-version")
                .long("package-version")
                .value_name("VERSION")
                .help("Package version for generated bindings")
                .default_value("0.1.0"),
        )
        .arg(
            Arg::new("validate")
                .long("validate")
                .help("Validate FFI interface before generation")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("docs")
                .long("docs")
                .help("Generate API documentation")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("incremental")
                .long("incremental")
                .help("Only regenerate changed files")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("verbose")
                .long("verbose")
                .short('v')
                .help("Enable verbose output")
                .action(ArgAction::SetTrue),
        )
        .subcommand(
            Command::new("list")
                .about("List available languages, templates, and features")
                .arg(
                    Arg::new("type")
                        .value_parser(["languages", "templates", "features"])
                        .help("What to list")
                        .default_value("languages"),
                ),
        )
        .subcommand(
            Command::new("validate")
                .about("Validate FFI interface without generating bindings")
                .arg(
                    Arg::new("source")
                        .long("source")
                        .short('s')
                        .value_name("DIR")
                        .help("Source directory to validate")
                        .default_value("src"),
                ),
        )
        .get_matches();

    // Initialize logging if verbose
    if matches.get_flag("verbose") {
        tracing_subscriber::fmt::init();
    }

    match matches.subcommand() {
        Some(("list", sub_matches)) => {
            let list_type = sub_matches.get_one::<String>("type").expect("'type' argument is required by clap");
            match list_type.as_str() {
                "languages" => list_languages(),
                "templates" => list_templates(),
                "features" => list_features(),
                _ => unreachable!(),
            }
            return Ok(());
        },
        Some(("validate", sub_matches)) => {
            let source_dir = PathBuf::from(sub_matches.get_one::<String>("source").expect("'source' argument is required by clap"));
            return validate_interface(&source_dir);
        },
        _ => {},
    }

    // Main generation logic
    let source_dir = PathBuf::from(matches.get_one::<String>("source").expect("'source' argument is required by clap"));
    let output_dir = PathBuf::from(matches.get_one::<String>("output").expect("'output' argument is required by clap"));
    let package_name = matches.get_one::<String>("package-name").expect("'package-name' argument is required by clap").clone();
    let package_version = matches.get_one::<String>("package-version").expect("'package-version' argument is required by clap").clone();
    let generate_docs = matches.get_flag("docs");
    let validate = matches.get_flag("validate");
    let incremental = matches.get_flag("incremental");
    let verbose = matches.get_flag("verbose");

    // Parse target languages
    let language_values: Vec<&Language> =
        matches.get_many::<Language>("languages").unwrap_or_default().collect();

    let target_languages = if language_values.contains(&&Language::All) {
        vec![
            TargetLanguage::Python,
            TargetLanguage::Java,
            TargetLanguage::Go,
            TargetLanguage::CSharp,
            TargetLanguage::TypeScript,
            TargetLanguage::JavaScript,
            TargetLanguage::Kotlin,
            TargetLanguage::Swift,
            TargetLanguage::Ruby,
            TargetLanguage::PHP,
        ]
    } else {
        language_values
            .into_iter()
            .filter(|&&lang| !matches!(lang, Language::All))
            .map(|&lang| lang.into())
            .collect()
    };

    if verbose {
        println!("🚀 TrustformeRS Code Generator v0.1.0");
        println!("📁 Source directory: {}", source_dir.display());
        println!("📂 Output directory: {}", output_dir.display());
        println!("🎯 Target languages: {:?}", target_languages);
        println!("📦 Package: {} v{}", package_name, package_version);
        println!();
    }

    // Create configuration
    let config = CodeGenConfig {
        output_dir,
        target_languages,
        package_info: PackageInfo {
            name: package_name,
            version: package_version,
            description: "TrustformeRS - High-performance transformer library".to_string(),
            author: "TrustformeRS Team".to_string(),
            license: "MIT".to_string(),
            repository: "https://github.com/trustformers/trustformers".to_string(),
        },
        package_name: None,
        features: HashMap::new(),
        type_mappings: HashMap::new(),
    };

    // Create and configure code generator
    let mut generator = CodeGenerator::new(config)?;

    // Parse FFI interface
    if verbose {
        println!("🔍 Parsing FFI interface from {}", source_dir.display());
    }
    generator.parse_interface(&source_dir)?;

    // Validate interface if requested
    if validate {
        if verbose {
            println!("✅ Validating FFI interface...");
        }
        let warnings = generator.validate_interface()?;
        if !warnings.is_empty() {
            println!("⚠️  Validation warnings:");
            for warning in warnings {
                println!("  - {}", warning);
            }
        } else if verbose {
            println!("✅ Interface validation passed!");
        }
    }

    // Generate bindings
    if verbose {
        println!("🔧 Generating language bindings...");
    }

    if incremental {
        generator.update_bindings(true)?;
    } else {
        generator.generate_all()?;
    }

    // Generate documentation if requested
    if generate_docs {
        if verbose {
            println!("📚 Generating API documentation...");
        }
        generator.generate_documentation()?;
    }

    if verbose {
        println!("✨ Code generation completed successfully!");
    } else {
        println!(
            "Generated bindings for {} languages",
            generator.config().target_languages.len()
        );
    }

    Ok(())
}

#[cfg(feature = "codegen")]
fn list_languages() {
    println!("Available target languages:");
    println!("  🐍 python      - Python bindings using ctypes/cffi");
    println!("  ☕ java        - Java bindings using JNI");
    println!("  🐹 go          - Go bindings using cgo");
    println!("  🔷 csharp      - C# bindings using P/Invoke");
    println!("  📘 typescript  - TypeScript bindings using ffi-napi");
    println!("  📒 javascript  - JavaScript bindings using ffi-napi");
    println!("  🟣 kotlin      - Kotlin/JVM bindings using JNI");
    println!("  🍎 swift       - Swift bindings for iOS/macOS");
    println!("  💎 ruby        - Ruby bindings using FFI gem");
    println!("  🐘 php         - PHP bindings using FFI extension");
    println!("  🌟 all         - Generate bindings for all languages");
}

#[cfg(feature = "codegen")]
fn list_templates() {
    println!("Available templates:");
    println!("  📄 function    - Function binding templates");
    println!("  🏗️  struct      - Structure/class templates");
    println!("  🔢 enum        - Enumeration templates");
    println!("  📦 package     - Package/project templates");
    println!("  📚 docs        - Documentation templates");
    println!("  🧪 test        - Test file templates");
    println!("  📋 example     - Example usage templates");
}

#[cfg(feature = "codegen")]
fn list_features() {
    println!("Available features:");
    println!("  ✅ validation  - Interface validation and linting");
    println!("  📚 docs        - API documentation generation");
    println!("  🔄 incremental - Incremental code generation");
    println!("  🧪 testing     - Test file generation");
    println!("  📋 examples    - Example code generation");
    println!("  🔧 customization - Custom type mappings");
    println!("  🎯 templates   - Template-based generation");
}

#[cfg(feature = "codegen")]
fn validate_interface(source_dir: &PathBuf) -> anyhow::Result<()> {
    println!("🔍 Validating FFI interface in {}", source_dir.display());

    let config = CodeGenConfig::default();
    let mut generator = CodeGenerator::new(config)?;

    generator.parse_interface(source_dir)?;
    let warnings = generator.validate_interface()?;

    if warnings.is_empty() {
        println!("✅ Interface validation passed - no issues found!");
    } else {
        println!("⚠️  Found {} validation warnings:", warnings.len());
        for (i, warning) in warnings.iter().enumerate() {
            println!("  {}. {}", i + 1, warning);
        }
    }

    Ok(())
}

#[cfg(not(feature = "codegen"))]
fn main() {
    eprintln!("❌ Error: trustformers-codegen requires the 'codegen' feature to be enabled.");
    eprintln!("Please build with: cargo build --features codegen");
    eprintln!();
    eprintln!("Available features:");
    eprintln!("  --features codegen        Enable code generation");
    eprintln!("  --features codegen,cuda   Enable code generation with CUDA support");
    std::process::exit(1);
}
