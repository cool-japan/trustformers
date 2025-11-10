//! Ruby bindings generator for FFI interfaces

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::codegen::ast::{FfiEnum, FfiFunction, FfiInterface, FfiStruct, FfiType};
use crate::codegen::templates::TemplateEngine;
use crate::codegen::{CodeGenConfig, TargetLanguage};
use crate::error::TrustformersResult;

use super::common::{function_can_fail, TypeMapper};
use super::LanguageGenerator;

/// Ruby bindings generator
pub struct RubyGenerator {
    config: CodeGenConfig,
    type_mappings: HashMap<String, String>,
}

impl RubyGenerator {
    pub fn new(config: &CodeGenConfig) -> TrustformersResult<Self> {
        let mut type_mappings = HashMap::new();

        // Basic type mappings for Ruby FFI
        type_mappings.insert("c_int".to_string(), ":int".to_string());
        type_mappings.insert("c_uint".to_string(), ":uint".to_string());
        type_mappings.insert("c_long".to_string(), ":long".to_string());
        type_mappings.insert("c_ulong".to_string(), ":ulong".to_string());
        type_mappings.insert("c_longlong".to_string(), ":long_long".to_string());
        type_mappings.insert("c_ulonglong".to_string(), ":ulong_long".to_string());
        type_mappings.insert("c_short".to_string(), ":short".to_string());
        type_mappings.insert("c_ushort".to_string(), ":ushort".to_string());
        type_mappings.insert("c_float".to_string(), ":float".to_string());
        type_mappings.insert("c_double".to_string(), ":double".to_string());
        type_mappings.insert("c_char".to_string(), ":char".to_string());
        type_mappings.insert("c_uchar".to_string(), ":uchar".to_string());
        type_mappings.insert("c_bool".to_string(), ":bool".to_string());
        type_mappings.insert("c_void".to_string(), ":void".to_string());
        type_mappings.insert("*const c_char".to_string(), ":string".to_string());
        type_mappings.insert("*mut c_char".to_string(), ":string".to_string());
        type_mappings.insert("*const c_void".to_string(), ":pointer".to_string());
        type_mappings.insert("*mut c_void".to_string(), ":pointer".to_string());

        // Add custom type mappings from config
        for (k, v) in &config.type_mappings {
            type_mappings.insert(k.clone(), v.target_type.clone());
        }

        Ok(Self {
            config: config.clone(),
            type_mappings,
        })
    }

    fn map_base_type(&self, type_name: &str) -> String {
        match type_name {
            "c_int" | "i32" => ":int",
            "c_uint" | "u32" => ":uint",
            "c_short" | "i16" => ":short",
            "c_ushort" | "u16" => ":ushort",
            "c_long" | "i64" => ":long",
            "c_ulong" | "u64" => ":ulong",
            "c_longlong" => ":long_long",
            "c_ulonglong" => ":ulong_long",
            "c_float" | "f32" => ":float",
            "c_double" | "f64" => ":double",
            "c_char" | "i8" => ":char",
            "c_uchar" | "u8" => ":uchar",
            "c_bool" => ":bool",
            "c_void" | "()" => ":void",
            "isize" => ":ssize_t",
            "usize" => ":size_t",
            name if name.ends_with("Handle") => ":pointer",
            _ => ":pointer", // Default for unknown types
        }
        .to_string()
    }

    fn generate_struct_class(&self, struct_def: &FfiStruct) -> String {
        let mut lines = Vec::new();

        // Struct documentation
        if !struct_def.documentation.is_empty() {
            lines.push("".to_string());
            for doc_line in &struct_def.documentation {
                lines.push(format!("  # {}", doc_line));
            }
        }

        // Convert to PascalCase for Ruby class name
        let class_name = to_pascal_case(&struct_def.name);

        if struct_def.is_opaque {
            // Opaque struct - just a handle
            lines.push(format!("  class {} < FFI::Struct", class_name));
            lines.push("    # Opaque structure - internal implementation hidden".to_string());
            lines.push("  end".to_string());
        } else {
            // Regular struct with fields
            lines.push(format!("  class {} < FFI::Struct", class_name));

            // Field definitions
            lines.push("    layout(".to_string());
            let public_fields: Vec<_> =
                struct_def.fields.iter().filter(|f| !f.is_private).collect();

            for (i, field) in public_fields.iter().enumerate() {
                let field_type = self.map_type(&field.type_info);
                let field_name = to_snake_case(&field.name);
                let comma = if i < public_fields.len() - 1 { "," } else { "" };
                lines.push(format!("      :{}, {}{}", field_name, field_type, comma));
            }

            lines.push("    )".to_string());

            // Generate getter and setter methods for better Ruby integration
            for field in &public_fields {
                let field_name = to_snake_case(&field.name);
                lines.push("".to_string());
                lines.push(format!("    # Get {}", field.name));
                lines.push(format!(
                    "    # @return [Object] the value of {}",
                    field.name
                ));
                lines.push(format!("    def {}", field_name));
                lines.push(format!(
                    "      self[:{field_name}]",
                    field_name = field_name
                ));
                lines.push("    end".to_string());

                lines.push("".to_string());
                lines.push(format!("    # Set {}", field.name));
                lines.push(format!(
                    "    # @param value [Object] the value to set for {}",
                    field.name
                ));
                lines.push(format!("    def {}=(value)", field_name));
                lines.push(format!(
                    "      self[:{field_name}] = value",
                    field_name = field_name
                ));
                lines.push("    end".to_string());
            }

            lines.push("  end".to_string());
        }

        lines.join("\n")
    }

    fn generate_enum_module(&self, enum_def: &FfiEnum) -> String {
        let mut lines = Vec::new();

        // Enum documentation
        if !enum_def.documentation.is_empty() {
            lines.push("".to_string());
            for doc_line in &enum_def.documentation {
                lines.push(format!("  # {}", doc_line));
            }
        }

        // Convert to PascalCase for Ruby module name
        let module_name = to_pascal_case(&enum_def.name);
        lines.push(format!("  module {}", module_name));

        // Enum variants as constants
        for variant in &enum_def.variants {
            if let Some(deprecation) = &variant.deprecation {
                lines.push(format!("    # DEPRECATED: {}", deprecation.message));
            }
            let constant_name = to_screaming_snake_case(&variant.name);
            lines.push(format!("    {} = {}", constant_name, variant.value));
        }

        lines.push("  end".to_string());
        lines.join("\n")
    }

    fn generate_function_bindings(&self, func: &FfiFunction) -> String {
        let mut lines = Vec::new();

        // Function documentation
        if !func.documentation.is_empty() {
            lines.push("".to_string());
            for doc_line in &func.documentation {
                lines.push(format!("  # {}", doc_line));
            }
        }

        // Generate parameter documentation
        if !func.parameters.is_empty() {
            if func.documentation.is_empty() {
                lines.push("  #".to_string());
            }
            for param in &func.parameters {
                let param_type = param.type_info.map_to_language(&TargetLanguage::Ruby);
                lines.push(format!(
                    "  # @param {} [{}] parameter description",
                    to_snake_case(&param.name),
                    param_type
                ));
            }
        }

        // Generate return documentation
        if func.return_type.name != "void" {
            let return_type = func.return_type.map_to_language(&TargetLanguage::Ruby);
            lines.push(format!(
                "  # @return [{}] return value description",
                return_type
            ));
        }

        // attach_function declaration
        let param_types: Vec<String> =
            func.parameters.iter().map(|p| self.map_type(&p.type_info)).collect();

        let return_type = self.map_type(&func.return_type);
        let ruby_name = to_snake_case(&func.name);

        lines.push(format!(
            "  attach_function :{}, :{}, [{}], {}",
            ruby_name,
            func.c_name,
            param_types.join(", "),
            return_type
        ));

        lines.join("\n")
    }
}

impl TypeMapper for RubyGenerator {
    fn map_type(&self, ffi_type: &FfiType) -> String {
        // Check for custom mappings first
        if let Some(mapped) = self.type_mappings.get(&ffi_type.name) {
            return mapped.clone();
        }

        // Handle pointer types
        if ffi_type.is_pointer() {
            if ffi_type.is_string() {
                return ":string".to_string();
            } else if ffi_type.base_type() == "c_void" {
                return ":pointer".to_string();
            } else {
                // For typed pointers, use :pointer
                return ":pointer".to_string();
            }
        }

        // Handle array types
        if let Some(size) = ffi_type.array_size {
            let base_type = self.map_base_type(&ffi_type.base_type());
            return format!("[{}, {}]", base_type, size);
        }

        // Handle regular types
        self.map_base_type(&ffi_type.name)
    }

    fn map_base_type(&self, type_name: &str) -> String {
        self.map_base_type(type_name)
    }

    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::Ruby
    }
}

impl LanguageGenerator for RubyGenerator {
    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::Ruby
    }

    fn file_extension(&self) -> &'static str {
        "rb"
    }

    fn generate(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        // Generate main module file
        let mut main_content = Vec::new();

        // Module header
        main_content.push("# frozen_string_literal: true".to_string());
        main_content.push("".to_string());
        main_content.push("#".to_string());
        main_content.push(format!(
            "# Ruby bindings for {}",
            &interface.metadata.library_name
        ));
        main_content.push("#".to_string());
        main_content.push("# TrustformeRS C API bindings".to_string());
        main_content.push(format!("# Version: {}", &interface.metadata.version));
        main_content.push("#".to_string());
        main_content.push("".to_string());

        // Require FFI
        main_content.push("require 'ffi'".to_string());
        main_content.push("".to_string());

        // Main module
        let module_name = to_pascal_case(&self.config.package_info.name);
        main_content.push(format!("module {}", module_name));
        main_content.push("".to_string());

        // Error class
        main_content.push("  # Base error class for TrustformeRS errors".to_string());
        main_content.push("  class TrustformersError < StandardError; end".to_string());
        main_content.push("".to_string());

        // Generate enums
        for enum_def in &interface.enums {
            main_content.push(self.generate_enum_module(enum_def));
            main_content.push("".to_string());
        }

        // Generate structs
        for struct_def in &interface.structs {
            main_content.push(self.generate_struct_class(struct_def));
            main_content.push("".to_string());
        }

        // Native module with FFI bindings
        main_content.push("  # Native FFI bindings".to_string());
        main_content.push("  module Native".to_string());
        main_content.push("    extend FFI::Library".to_string());
        main_content.push("".to_string());

        // Library loading
        main_content.push("    # Load the native library".to_string());
        main_content.push("    def self.lib_path".to_string());
        main_content.push("      case RbConfig::CONFIG['host_os']".to_string());
        main_content.push("      when /mswin|msys|mingw|cygwin|bccwin|wince|emc/".to_string());
        main_content.push("        'trustformers_c.dll'".to_string());
        main_content.push("      when /darwin|mac os/".to_string());
        main_content.push("        'libtrusformers_c.dylib'".to_string());
        main_content.push("      else".to_string());
        main_content.push("        'libtrusformers_c.so'".to_string());
        main_content.push("      end".to_string());
        main_content.push("    end".to_string());
        main_content.push("".to_string());
        main_content.push("    ffi_lib lib_path".to_string());
        main_content.push("".to_string());

        // Generate function bindings
        main_content.push("    # Function bindings".to_string());
        for func in &interface.functions {
            main_content.push(self.generate_function_bindings(func));
        }

        main_content.push("  end".to_string());
        main_content.push("".to_string());

        // Generate convenience wrapper methods
        main_content.push("  # Convenience wrapper methods".to_string());
        main_content.push("  class << self".to_string());
        for func in &interface.functions {
            let ruby_name = to_snake_case(&func.name);
            let param_names: Vec<String> =
                func.parameters.iter().map(|p| to_snake_case(&p.name)).collect();

            main_content.push("".to_string());
            if !func.documentation.is_empty() {
                for doc_line in &func.documentation {
                    main_content.push(format!("    # {}", doc_line));
                }
            }

            main_content.push(format!("    def {}({})", ruby_name, param_names.join(", ")));

            if function_can_fail(&func.return_type) {
                main_content.push(format!(
                    "      result = Native.{}({})",
                    ruby_name,
                    param_names.join(", ")
                ));
                main_content.push("      raise TrustformersError, \"Function #{__method__} failed with error code #{result}\" if result != 0".to_string());
                main_content.push("      result".to_string());
            } else {
                main_content.push(format!(
                    "      Native.{}({})",
                    ruby_name,
                    param_names.join(", ")
                ));
            }

            main_content.push("    end".to_string());
        }
        main_content.push("  end".to_string());

        main_content.push("end".to_string());

        // Write main module file
        let main_file = output_dir.join(format!(
            "{}.rb",
            to_snake_case(&self.config.package_info.name)
        ));
        fs::write(&main_file, main_content.join("\n"))?;

        // Generate package files
        self.generate_package_files(interface, output_dir, _templates)?;

        // Generate examples
        self.generate_examples(interface, output_dir, _templates)?;

        Ok(())
    }

    fn generate_package_files(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        let module_name = to_pascal_case(&self.config.package_info.name);
        let gem_name = to_snake_case(&self.config.package_info.name);

        // Generate gemspec
        let gemspec_content = format!(
            "# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name          = \"{}\"
  spec.version       = \"{}\"
  spec.authors       = [\"{}\"]
  spec.email         = [\"noreply@example.com\"]

  spec.summary       = \"{}\"
  spec.description   = \"{}\"
  spec.homepage      = \"https://github.com/cool-japan/trustformers\"
  spec.license       = \"{}\"
  spec.required_ruby_version = \">= 2.7.0\"

  spec.metadata[\"homepage_uri\"] = spec.homepage
  spec.metadata[\"source_code_uri\"] = spec.homepage
  spec.metadata[\"changelog_uri\"] = \"#{{spec.homepage}}/blob/main/CHANGELOG.md\"

  # Specify which files should be added to the gem when it is released.
  spec.files = Dir.glob(\"{{lib,ext}}/**/*\") + %w[README.md LICENSE]
  spec.bindir        = \"exe\"
  spec.executables   = spec.files.grep(%r{{\\Aexe/}}) {{ |f| File.basename(f) }}
  spec.require_paths = [\"lib\"]

  # Runtime dependencies
  spec.add_dependency \"ffi\", \"~> 1.15\"

  # Development dependencies
  spec.add_development_dependency \"rake\", \"~> 13.0\"
  spec.add_development_dependency \"rspec\", \"~> 3.0\"
  spec.add_development_dependency \"rubocop\", \"~> 1.21\"
end
",
            gem_name,
            self.config.package_info.version,
            self.config.package_info.author,
            self.config.package_info.description,
            self.config.package_info.description,
            self.config.package_info.license
        );

        fs::write(
            output_dir.join(format!("{}.gemspec", gem_name)),
            gemspec_content,
        )?;

        // Generate Gemfile
        let gemfile_content = format!(
            "# frozen_string_literal: true

source \"https://rubygems.org\"

# Specify your gem's dependencies in {}.gemspec
gemspec

gem \"rake\", \"~> 13.0\"
gem \"rspec\", \"~> 3.0\"
gem \"rubocop\", \"~> 1.21\"
",
            gem_name
        );

        fs::write(output_dir.join("Gemfile"), gemfile_content)?;

        // Generate Rakefile
        let rakefile_content = "# frozen_string_literal: true

require \"bundler/gem_tasks\"
require \"rspec/core/rake_task\"
require \"rubocop/rake_task\"

RSpec::Core::RakeTask.new(:spec)
RuboCop::RakeTask.new

task default: %i[spec rubocop]
";

        fs::write(output_dir.join("Rakefile"), rakefile_content)?;

        // Generate lib directory and loader
        let lib_dir = output_dir.join("lib");
        fs::create_dir_all(&lib_dir)?;

        let loader_content = format!(
            "# frozen_string_literal: true

require_relative \"{}/version\"
require_relative \"../{}\"

module {}
  class Error < StandardError; end
  # Your code goes here...
end
",
            gem_name, gem_name, module_name
        );

        fs::write(lib_dir.join(format!("{}.rb", gem_name)), loader_content)?;

        // Generate version file
        let version_content = format!(
            "# frozen_string_literal: true

module {}
  VERSION = \"{}\"
end
",
            module_name, self.config.package_info.version
        );

        let version_dir = lib_dir.join(&gem_name);
        fs::create_dir_all(&version_dir)?;
        fs::write(version_dir.join("version.rb"), version_content)?;

        Ok(())
    }

    fn generate_examples(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        let examples_dir = output_dir.join("examples");
        fs::create_dir_all(&examples_dir)?;

        let gem_name = to_snake_case(&self.config.package_info.name);
        let module_name = to_pascal_case(&self.config.package_info.name);

        // Generate basic usage example
        let mut example_content = Vec::new();
        example_content.push("#!/usr/bin/env ruby".to_string());
        example_content.push("# frozen_string_literal: true".to_string());
        example_content.push("".to_string());
        example_content.push("#".to_string());
        example_content.push("# Basic usage example for TrustformeRS Ruby bindings".to_string());
        example_content.push("#".to_string());
        example_content.push("".to_string());
        example_content.push(format!("require '{}'", gem_name));
        example_content.push("".to_string());
        example_content.push("def main".to_string());
        example_content.push("  puts 'TrustformeRS Ruby Bindings Example'".to_string());
        example_content.push("".to_string());

        // Add example function calls if available
        if let Some(first_func) = interface.functions.first() {
            if first_func.parameters.is_empty() {
                let ruby_name = to_snake_case(&first_func.name);
                example_content.push(format!("  # Call {}", first_func.name));
                example_content.push(format!("  result = {}.{}", module_name, ruby_name));
                example_content.push("  puts \"Result: #{result}\"".to_string());
            }
        }

        example_content.push("rescue => e".to_string());
        example_content.push("  puts \"Error: #{e.message}\"".to_string());
        example_content.push("  exit 1".to_string());
        example_content.push("end".to_string());
        example_content.push("".to_string());
        example_content.push("main if __FILE__ == $PROGRAM_NAME".to_string());

        fs::write(
            examples_dir.join("basic_usage.rb"),
            example_content.join("\n"),
        )?;

        Ok(())
    }

    fn generate_tests(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        let spec_dir = output_dir.join("spec");
        fs::create_dir_all(&spec_dir)?;

        let gem_name = to_snake_case(&self.config.package_info.name);
        let module_name = to_pascal_case(&self.config.package_info.name);

        // Generate spec_helper.rb
        let spec_helper = format!(
            "# frozen_string_literal: true

require \"{}\"

RSpec.configure do |config|
  # Enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = \".rspec_status\"

  # Disable RSpec exposing methods globally on `Module` and `main`
  config.disable_monkey_patching!

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end
end
",
            gem_name
        );

        fs::write(spec_dir.join("spec_helper.rb"), spec_helper)?;

        // Generate basic test file
        let mut test_content = Vec::new();
        test_content.push("# frozen_string_literal: true".to_string());
        test_content.push("".to_string());
        test_content.push("require 'spec_helper'".to_string());
        test_content.push("".to_string());
        test_content.push(format!("RSpec.describe {} do", module_name));
        test_content.push("  it 'has a version number' do".to_string());
        test_content.push(format!(
            "    expect({}::VERSION).not_to be nil",
            module_name
        ));
        test_content.push("  end".to_string());
        test_content.push("".to_string());

        // Add tests for available functions
        for func in interface.functions.iter().take(3) {
            // Limit to first 3 functions
            let ruby_name = to_snake_case(&func.name);
            test_content.push(format!("  describe '.{}' do", ruby_name));
            test_content.push(format!("    it 'responds to {}' do", ruby_name));
            test_content.push(format!(
                "      expect({}).to respond_to(:{})",
                module_name, ruby_name
            ));
            test_content.push("    end".to_string());
            test_content.push("  end".to_string());
            test_content.push("".to_string());
        }

        test_content.push("end".to_string());

        fs::write(
            spec_dir.join(format!("{}_spec.rb", gem_name)),
            test_content.join("\n"),
        )?;

        Ok(())
    }
}

// Helper functions for naming conventions

/// Convert snake_case or kebab-case to PascalCase
fn to_pascal_case(s: &str) -> String {
    s.split(|c| c == '_' || c == '-')
        .filter(|s| !s.is_empty())
        .map(|s| {
            let mut chars = s.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
                },
            }
        })
        .collect()
}

/// Convert to snake_case
/// Handles acronyms by inserting underscores between each capital letter
fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();

    for (i, &ch) in chars.iter().enumerate() {
        if ch.is_uppercase() {
            // Add underscore if not at start
            if i > 0 {
                result.push('_');
            }
            result.push(ch.to_lowercase().next().unwrap());
        } else if ch == '-' {
            result.push('_');
        } else {
            result.push(ch);
        }
    }

    result
}

/// Convert to SCREAMING_SNAKE_CASE
fn to_screaming_snake_case(s: &str) -> String {
    to_snake_case(s).to_uppercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(to_pascal_case("hello_world"), "HelloWorld");
        assert_eq!(to_pascal_case("test_case"), "TestCase");
        assert_eq!(to_pascal_case("ffi_interface"), "FfiInterface");
        assert_eq!(to_pascal_case("trustformers-c"), "TrustformersC");
    }

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("HelloWorld"), "hello_world");
        assert_eq!(to_snake_case("TestCase"), "test_case");
        assert_eq!(to_snake_case("FFIInterface"), "f_f_i_interface");
        assert_eq!(to_snake_case("already_snake"), "already_snake");
    }

    #[test]
    fn test_to_screaming_snake_case() {
        assert_eq!(to_screaming_snake_case("HelloWorld"), "HELLO_WORLD");
        assert_eq!(to_screaming_snake_case("test_case"), "TEST_CASE");
        assert_eq!(to_screaming_snake_case("value"), "VALUE");
    }

    #[test]
    fn test_type_mapping() {
        let config = CodeGenConfig::default();
        let generator = RubyGenerator::new(&config).unwrap();

        assert_eq!(generator.map_base_type("c_int"), ":int");
        assert_eq!(generator.map_base_type("c_float"), ":float");
        assert_eq!(generator.map_base_type("c_bool"), ":bool");
        assert_eq!(generator.map_base_type("c_void"), ":void");
    }
}
