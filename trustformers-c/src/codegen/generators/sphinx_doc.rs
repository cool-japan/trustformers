//! Sphinx documentation generator for FFI interfaces
//!
//! Generates Sphinx-compatible reStructuredText documentation for the C FFI interface.
//! This includes conf.py, index.rst, API reference pages, and all supporting files.

use std::fs;
use std::path::Path;

use crate::codegen::ast::{
    FfiEnum, FfiEnumVariant, FfiField, FfiFunction, FfiInterface, FfiParameter, FfiStruct, FfiType,
};
use crate::codegen::templates::TemplateEngine;
use crate::codegen::{CodeGenConfig, TargetLanguage};
use crate::error::TrustformersResult;

use super::LanguageGenerator;

/// Sphinx documentation generator configuration
#[derive(Debug, Clone)]
pub struct SphinxConfig {
    /// Project name for Sphinx
    pub project_name: String,
    /// Project version
    pub version: String,
    /// Author name
    pub author: String,
    /// Copyright notice
    pub copyright: String,
    /// Sphinx theme to use
    pub theme: String,
    /// Language for documentation
    pub language: String,
    /// Enable autodoc extension
    pub enable_autodoc: bool,
    /// Enable intersphinx
    pub enable_intersphinx: bool,
}

impl Default for SphinxConfig {
    fn default() -> Self {
        Self {
            project_name: "TrustformeRS C API".to_string(),
            version: "0.1.0".to_string(),
            author: "TrustformeRS Team".to_string(),
            copyright: "2024, TrustformeRS Team".to_string(),
            theme: "sphinx_rtd_theme".to_string(),
            language: "en".to_string(),
            enable_autodoc: true,
            enable_intersphinx: true,
        }
    }
}

/// Sphinx documentation generator
pub struct SphinxDocGenerator {
    config: CodeGenConfig,
    sphinx_config: SphinxConfig,
}

impl SphinxDocGenerator {
    pub fn new(config: &CodeGenConfig) -> TrustformersResult<Self> {
        let mut sphinx_config = SphinxConfig::default();

        // Override with config values if available
        sphinx_config.project_name = config.package_info.name.clone();
        sphinx_config.version = config.package_info.version.clone();
        sphinx_config.author = config.package_info.author.clone();

        Ok(Self {
            config: config.clone(),
            sphinx_config,
        })
    }

    /// Generate the main conf.py file for Sphinx
    fn generate_conf_py(&self, output_dir: &Path) -> TrustformersResult<()> {
        let conf_py_content = format!(
            r#"# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = '{}'
copyright = '{}'
author = '{}'
version = '{}'
release = '{}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = '{}'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = '{}'
html_static_path = ['_static']
html_theme_options = {{
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}}

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {{
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}}

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {{
    'python': ('https://docs.python.org/3', None),
}}

# -- Options for napoleon ----------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True
"#,
            self.sphinx_config.project_name,
            self.sphinx_config.copyright,
            self.sphinx_config.author,
            self.sphinx_config.version,
            self.sphinx_config.version,
            self.sphinx_config.language,
            self.sphinx_config.theme,
        );

        fs::write(output_dir.join("conf.py"), conf_py_content)?;
        Ok(())
    }

    /// Generate the main index.rst file with toctree
    fn generate_index_rst(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
    ) -> TrustformersResult<()> {
        let mut content = Vec::new();

        // Title
        content.push(format!("{}", self.sphinx_config.project_name));
        content.push("=".repeat(self.sphinx_config.project_name.len()));
        content.push(String::new());

        // Project description
        content.push(format!("Version: {}", self.sphinx_config.version));
        content.push(String::new());
        content.push(format!("{}", self.config.package_info.description));
        content.push(String::new());

        // Overview
        content.push("Overview".to_string());
        content.push("---------".to_string());
        content.push(String::new());
        content.push(
            "The TrustformeRS C API provides a comprehensive FFI interface for interacting with"
                .to_string(),
        );
        content.push(
            "TrustformeRS transformer models from C and other languages that support C FFI."
                .to_string(),
        );
        content.push(String::new());

        // Statistics
        content.push("API Statistics".to_string());
        content.push("~~~~~~~~~~~~~~".to_string());
        content.push(String::new());
        content.push(format!(
            "* **Functions**: {} exported functions",
            interface.functions.len()
        ));
        content.push(format!(
            "* **Structures**: {} data structures",
            interface.structs.len()
        ));
        content.push(format!(
            "* **Enumerations**: {} enumerations",
            interface.enums.len()
        ));
        content.push(format!(
            "* **Constants**: {} constants",
            interface.constants.len()
        ));
        content.push(String::new());

        // Features
        content.push("Features".to_string());
        content.push("--------".to_string());
        content.push(String::new());
        content.push("* Memory-safe C interface with comprehensive error handling".to_string());
        content.push(
            "* Support for all major transformer architectures (BERT, GPT, T5, LLaMA, etc.)"
                .to_string(),
        );
        content.push("* Hardware acceleration (CUDA, Metal, ROCm)".to_string());
        content.push("* Quantization support (INT8, INT4, FP16)".to_string());
        content.push("* Tokenizer integration".to_string());
        content.push("* Pipeline API for common tasks".to_string());
        content.push(String::new());

        // Table of contents
        content.push("Table of Contents".to_string());
        content.push("----------------".to_string());
        content.push(String::new());
        content.push(".. toctree::".to_string());
        content.push("   :maxdepth: 2".to_string());
        content.push("   :caption: Contents:".to_string());
        content.push(String::new());
        content.push("   installation".to_string());
        content.push("   quickstart".to_string());
        content.push("   api".to_string());
        content.push("   functions".to_string());
        content.push("   structures".to_string());
        content.push("   enumerations".to_string());
        content.push("   constants".to_string());
        content.push("   examples".to_string());
        content.push("   error_handling".to_string());
        content.push(String::new());

        // Indices
        content.push("Indices and tables".to_string());
        content.push("==================".to_string());
        content.push(String::new());
        content.push("* :ref:`genindex`".to_string());
        content.push("* :ref:`modindex`".to_string());
        content.push("* :ref:`search`".to_string());
        content.push(String::new());

        fs::write(output_dir.join("index.rst"), content.join("\n"))?;
        Ok(())
    }

    /// Generate the API reference page
    fn generate_api_rst(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
    ) -> TrustformersResult<()> {
        let mut content = Vec::new();

        content.push("API Reference".to_string());
        content.push("=============".to_string());
        content.push(String::new());

        content.push("This section provides detailed documentation for all exported symbols in the TrustformeRS C API.".to_string());
        content.push(String::new());

        // Functions section
        if !interface.functions.is_empty() {
            content.push("Functions".to_string());
            content.push("---------".to_string());
            content.push(String::new());
            content.push("See :doc:`functions` for detailed function documentation.".to_string());
            content.push(String::new());
        }

        // Structures section
        if !interface.structs.is_empty() {
            content.push("Data Structures".to_string());
            content.push("---------------".to_string());
            content.push(String::new());
            content.push("See :doc:`structures` for detailed structure documentation.".to_string());
            content.push(String::new());
        }

        // Enumerations section
        if !interface.enums.is_empty() {
            content.push("Enumerations".to_string());
            content.push("------------".to_string());
            content.push(String::new());
            content.push(
                "See :doc:`enumerations` for detailed enumeration documentation.".to_string(),
            );
            content.push(String::new());
        }

        // Constants section
        if !interface.constants.is_empty() {
            content.push("Constants".to_string());
            content.push("---------".to_string());
            content.push(String::new());
            content.push("See :doc:`constants` for detailed constant documentation.".to_string());
            content.push(String::new());
        }

        fs::write(output_dir.join("api.rst"), content.join("\n"))?;
        Ok(())
    }

    /// Generate documentation for all functions
    fn generate_functions_rst(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
    ) -> TrustformersResult<()> {
        let mut content = Vec::new();

        content.push("Functions".to_string());
        content.push("=========".to_string());
        content.push(String::new());

        content.push(
            "This section documents all exported C functions in the TrustformeRS API.".to_string(),
        );
        content.push(String::new());

        // Group functions by category (based on prefix or module)
        let mut categories: std::collections::HashMap<String, Vec<&FfiFunction>> =
            std::collections::HashMap::new();

        for func in &interface.functions {
            let category = self.categorize_function(&func.name);
            categories.entry(category).or_insert_with(Vec::new).push(func);
        }

        // Sort categories
        let mut category_names: Vec<_> = categories.keys().cloned().collect();
        category_names.sort();

        for category in category_names {
            let funcs = categories.get(&category).unwrap();

            content.push(String::new());
            content.push(format!("{}", category));
            content.push("-".repeat(category.len()));
            content.push(String::new());

            for func in funcs {
                content.extend(self.generate_function_docs(func));
                content.push(String::new());
            }
        }

        fs::write(output_dir.join("functions.rst"), content.join("\n"))?;
        Ok(())
    }

    /// Generate documentation for a single function
    fn generate_function_docs(&self, func: &FfiFunction) -> Vec<String> {
        let mut lines = Vec::new();

        // Function name as section header
        lines.push(format!(
            ".. c:function:: {} {}({})",
            self.format_type(&func.return_type),
            func.c_name,
            self.format_parameters(&func.parameters)
        ));
        lines.push(String::new());

        // Indented content
        let indent = "   ";

        // Documentation
        if !func.documentation.is_empty() {
            for doc_line in &func.documentation {
                lines.push(format!("{}{}", indent, doc_line));
            }
            lines.push(String::new());
        }

        // Parameters
        if !func.parameters.is_empty() {
            for param in &func.parameters {
                let param_type = self.format_type(&param.type_info);
                lines.push(format!(
                    "{}:param {} {}: {}",
                    indent,
                    param_type,
                    param.name,
                    if param.documentation.is_empty() {
                        "Parameter description".to_string()
                    } else {
                        param.documentation.join(" ")
                    }
                ));
            }
            lines.push(String::new());
        }

        // Return value
        if func.return_type.name != "void" && func.return_type.name != "()" {
            lines.push(format!(
                "{}:returns: {}",
                indent,
                if func.can_fail {
                    "Error code indicating success or failure".to_string()
                } else {
                    "Return value".to_string()
                }
            ));
            lines.push(format!(
                "{}:rtype: {}",
                indent,
                self.format_type(&func.return_type)
            ));
            lines.push(String::new());
        }

        // Additional metadata
        if !func.required_features.is_empty() {
            lines.push(format!("{}.. note::", indent));
            lines.push(format!(
                "{}   Required features: {}",
                indent,
                func.required_features.join(", ")
            ));
            lines.push(String::new());
        }

        if !func.platforms.is_empty() {
            lines.push(format!("{}.. note::", indent));
            lines.push(format!(
                "{}   Platform availability: {}",
                indent,
                func.platforms.join(", ")
            ));
            lines.push(String::new());
        }

        if let Some(deprecation) = &func.deprecation {
            lines.push(format!(
                "{}.. deprecated:: {}",
                indent,
                deprecation.since_version.as_ref().unwrap_or(&"".to_string())
            ));
            lines.push(format!("{}   {}", indent, deprecation.message));
            if let Some(replacement) = &deprecation.replacement {
                lines.push(format!(
                    "{}   Use :c:func:`{}` instead.",
                    indent, replacement
                ));
            }
            lines.push(String::new());
        }

        // Example usage
        lines.push(format!("{}**Example:**", indent));
        lines.push(String::new());
        lines.push(format!("{}.. code-block:: c", indent));
        lines.push(String::new());
        lines.push(format!("{}   // Example usage of {}", indent, func.c_name));
        let example_params: Vec<String> =
            func.parameters.iter().map(|p| self.get_example_value(&p.type_info)).collect();

        if func.return_type.name != "void" && func.return_type.name != "()" {
            lines.push(format!(
                "{}   {} result = {}({});",
                indent,
                self.format_type(&func.return_type),
                func.c_name,
                example_params.join(", ")
            ));
            if func.can_fail {
                lines.push(format!("{}   if (result != 0) {{", indent));
                lines.push(format!("{}       // Handle error", indent));
                lines.push(format!("{}   }}", indent));
            }
        } else {
            lines.push(format!(
                "{}   {}({});",
                indent,
                func.c_name,
                example_params.join(", ")
            ));
        }

        lines
    }

    /// Generate documentation for all structures
    fn generate_structures_rst(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
    ) -> TrustformersResult<()> {
        let mut content = Vec::new();

        content.push("Data Structures".to_string());
        content.push("===============".to_string());
        content.push(String::new());

        content.push(
            "This section documents all data structures exposed in the TrustformeRS C API."
                .to_string(),
        );
        content.push(String::new());

        for struct_def in &interface.structs {
            content.extend(self.generate_struct_docs(struct_def));
            content.push(String::new());
        }

        fs::write(output_dir.join("structures.rst"), content.join("\n"))?;
        Ok(())
    }

    /// Generate documentation for a single structure
    fn generate_struct_docs(&self, struct_def: &FfiStruct) -> Vec<String> {
        let mut lines = Vec::new();

        // Structure name as section header
        lines.push(format!(".. c:struct:: {}", struct_def.c_name));
        lines.push(String::new());

        let indent = "   ";

        // Documentation
        if !struct_def.documentation.is_empty() {
            for doc_line in &struct_def.documentation {
                lines.push(format!("{}{}", indent, doc_line));
            }
            lines.push(String::new());
        }

        if struct_def.is_opaque {
            lines.push(format!("{}.. note::", indent));
            lines.push(format!(
                "{}   This is an opaque structure. Its internal implementation is hidden.",
                indent
            ));
            lines.push(format!(
                "{}   Use the provided API functions to interact with instances of this type.",
                indent
            ));
            lines.push(String::new());
        } else {
            // Fields
            if !struct_def.fields.is_empty() {
                lines.push(format!("{}**Fields:**", indent));
                lines.push(String::new());

                for field in &struct_def.fields {
                    if !field.is_private {
                        lines.extend(self.generate_field_docs(field, indent));
                    }
                }
                lines.push(String::new());
            }
        }

        // Additional metadata
        if struct_def.is_packed {
            lines.push(format!("{}.. note::", indent));
            lines.push(format!(
                "{}   This structure is packed (no padding between fields).",
                indent
            ));
            lines.push(String::new());
        }

        if let Some(alignment) = struct_def.alignment {
            lines.push(format!("{}.. note::", indent));
            lines.push(format!(
                "{}   This structure has explicit alignment: {} bytes",
                indent, alignment
            ));
            lines.push(String::new());
        }

        if let Some(deprecation) = &struct_def.deprecation {
            lines.push(format!(
                "{}.. deprecated:: {}",
                indent,
                deprecation.since_version.as_ref().unwrap_or(&"".to_string())
            ));
            lines.push(format!("{}   {}", indent, deprecation.message));
            if let Some(replacement) = &deprecation.replacement {
                lines.push(format!(
                    "{}   Use :c:type:`{}` instead.",
                    indent, replacement
                ));
            }
            lines.push(String::new());
        }

        lines
    }

    /// Generate documentation for a structure field
    fn generate_field_docs(&self, field: &FfiField, base_indent: &str) -> Vec<String> {
        let mut lines = Vec::new();
        let indent = format!("{}   ", base_indent);

        lines.push(format!(
            "{}.. c:member:: {} {}",
            base_indent,
            self.format_type(&field.type_info),
            field.name
        ));
        lines.push(String::new());

        if !field.documentation.is_empty() {
            for doc_line in &field.documentation {
                lines.push(format!("{}{}", indent, doc_line));
            }
            lines.push(String::new());
        }

        lines.push(format!(
            "{}:type: {}",
            indent,
            self.format_type(&field.type_info)
        ));
        lines.push(String::new());

        lines
    }

    /// Generate documentation for all enumerations
    fn generate_enumerations_rst(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
    ) -> TrustformersResult<()> {
        let mut content = Vec::new();

        content.push("Enumerations".to_string());
        content.push("============".to_string());
        content.push(String::new());

        content.push(
            "This section documents all enumeration types in the TrustformeRS C API.".to_string(),
        );
        content.push(String::new());

        for enum_def in &interface.enums {
            content.extend(self.generate_enum_docs(enum_def));
            content.push(String::new());
        }

        fs::write(output_dir.join("enumerations.rst"), content.join("\n"))?;
        Ok(())
    }

    /// Generate documentation for a single enumeration
    fn generate_enum_docs(&self, enum_def: &FfiEnum) -> Vec<String> {
        let mut lines = Vec::new();

        // Enum name as section header
        lines.push(format!(".. c:enum:: {}", enum_def.c_name));
        lines.push(String::new());

        let indent = "   ";

        // Documentation
        if !enum_def.documentation.is_empty() {
            for doc_line in &enum_def.documentation {
                lines.push(format!("{}{}", indent, doc_line));
            }
            lines.push(String::new());
        }

        if enum_def.is_flags {
            lines.push(format!("{}.. note::", indent));
            lines.push(format!(
                "{}   This is a flags enumeration. Values can be combined using bitwise OR.",
                indent
            ));
            lines.push(String::new());
        }

        // Variants
        if !enum_def.variants.is_empty() {
            lines.push(format!("{}**Values:**", indent));
            lines.push(String::new());

            for variant in &enum_def.variants {
                lines.extend(self.generate_enum_variant_docs(variant, indent));
            }
            lines.push(String::new());
        }

        // Value range
        let (min_val, max_val) = enum_def.value_range();
        lines.push(format!("{}.. note::", indent));
        lines.push(format!(
            "{}   Value range: {} to {}",
            indent, min_val, max_val
        ));
        lines.push(String::new());

        if let Some(deprecation) = &enum_def.deprecation {
            lines.push(format!(
                "{}.. deprecated:: {}",
                indent,
                deprecation.since_version.as_ref().unwrap_or(&"".to_string())
            ));
            lines.push(format!("{}   {}", indent, deprecation.message));
            if let Some(replacement) = &deprecation.replacement {
                lines.push(format!(
                    "{}   Use :c:type:`{}` instead.",
                    indent, replacement
                ));
            }
            lines.push(String::new());
        }

        lines
    }

    /// Generate documentation for an enumeration variant
    fn generate_enum_variant_docs(
        &self,
        variant: &FfiEnumVariant,
        base_indent: &str,
    ) -> Vec<String> {
        let mut lines = Vec::new();
        let indent = format!("{}   ", base_indent);

        lines.push(format!(
            "{}.. c:enumerator:: {}",
            base_indent, variant.c_name
        ));
        lines.push(String::new());

        lines.push(format!("{}Value: {}", indent, variant.value));
        lines.push(String::new());

        if !variant.documentation.is_empty() {
            for doc_line in &variant.documentation {
                lines.push(format!("{}{}", indent, doc_line));
            }
            lines.push(String::new());
        }

        if let Some(deprecation) = &variant.deprecation {
            lines.push(format!(
                "{}.. deprecated:: {}",
                indent,
                deprecation.since_version.as_ref().unwrap_or(&"".to_string())
            ));
            lines.push(format!("{}   {}", indent, deprecation.message));
            lines.push(String::new());
        }

        lines
    }

    /// Generate constants documentation
    fn generate_constants_rst(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
    ) -> TrustformersResult<()> {
        let mut content = Vec::new();

        content.push("Constants".to_string());
        content.push("=========".to_string());
        content.push(String::new());

        content.push(
            "This section documents all constants defined in the TrustformeRS C API.".to_string(),
        );
        content.push(String::new());

        for constant in &interface.constants {
            content.push(format!(".. c:macro:: {}", constant.c_name));
            content.push(String::new());

            let indent = "   ";

            if !constant.documentation.is_empty() {
                for doc_line in &constant.documentation {
                    content.push(format!("{}{}", indent, doc_line));
                }
                content.push(String::new());
            }

            content.push(format!(
                "{}:type: {}",
                indent,
                self.format_type(&constant.type_info)
            ));
            content.push(format!("{}:value: {:?}", indent, constant.value));
            content.push(String::new());
        }

        fs::write(output_dir.join("constants.rst"), content.join("\n"))?;
        Ok(())
    }

    /// Generate installation guide
    fn generate_installation_rst(&self, output_dir: &Path) -> TrustformersResult<()> {
        let content = r#"Installation
============

This guide explains how to install and set up the TrustformeRS C API.

Prerequisites
-------------

* C compiler (GCC, Clang, or MSVC)
* CMake 3.15 or later
* Rust toolchain (if building from source)

Building from Source
--------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/trustformers/trustformers.git
      cd trustformers

2. Build the C library:

   .. code-block:: bash

      cargo build --release -p trustformers-c

3. The compiled library will be available at:

   * Linux: ``target/release/libtrustformers_c.so``
   * macOS: ``target/release/libtrustformers_c.dylib``
   * Windows: ``target/release/trustformers_c.dll``

Using CMake
-----------

Add the following to your ``CMakeLists.txt``:

.. code-block:: cmake

   find_library(TRUSTFORMERS_C trustformers_c PATHS /path/to/lib)
   target_link_libraries(your_target ${TRUSTFORMERS_C})

Using pkg-config
----------------

.. code-block:: bash

   gcc your_program.c $(pkg-config --cflags --libs trustformers-c) -o your_program

Verification
------------

To verify the installation, compile and run this simple program:

.. code-block:: c

   #include <trustformers_c.h>
   #include <stdio.h>

   int main() {
       printf("TrustformeRS C API version: %s\n", trustformers_version());
       return 0;
   }
"#;

        fs::write(output_dir.join("installation.rst"), content)?;
        Ok(())
    }

    /// Generate quickstart guide
    fn generate_quickstart_rst(&self, output_dir: &Path) -> TrustformersResult<()> {
        let content = r#"Quick Start
===========

This guide will help you get started with the TrustformeRS C API.

Basic Example
-------------

Here's a simple example that loads a model and performs inference:

.. code-block:: c

   #include <trustformers_c.h>
   #include <stdio.h>
   #include <stdlib.h>

   int main() {
       TrustformersError error;
       TrustformersModelHandle model = NULL;
       TrustformersTokenizerHandle tokenizer = NULL;

       // Initialize the library
       error = trustformers_init();
       if (error != TRUSTFORMERS_SUCCESS) {
           fprintf(stderr, "Failed to initialize: %s\n",
                   trustformers_error_message(error));
           return 1;
       }

       // Load model
       error = trustformers_model_load("bert-base-uncased", &model);
       if (error != TRUSTFORMERS_SUCCESS) {
           fprintf(stderr, "Failed to load model: %s\n",
                   trustformers_error_message(error));
           return 1;
       }

       // Load tokenizer
       error = trustformers_tokenizer_load("bert-base-uncased", &tokenizer);
       if (error != TRUSTFORMERS_SUCCESS) {
           fprintf(stderr, "Failed to load tokenizer: %s\n",
                   trustformers_error_message(error));
           trustformers_model_free(model);
           return 1;
       }

       // Tokenize input
       const char* text = "Hello, world!";
       TrustformersTokenIds* token_ids = NULL;
       error = trustformers_tokenizer_encode(tokenizer, text, &token_ids);
       if (error != TRUSTFORMERS_SUCCESS) {
           fprintf(stderr, "Failed to tokenize: %s\n",
                   trustformers_error_message(error));
           goto cleanup;
       }

       // Run inference
       TrustformersTensor* output = NULL;
       error = trustformers_model_forward(model, token_ids, &output);
       if (error != TRUSTFORMERS_SUCCESS) {
           fprintf(stderr, "Failed to run inference: %s\n",
                   trustformers_error_message(error));
           goto cleanup;
       }

       printf("Inference completed successfully!\n");

   cleanup:
       trustformers_tensor_free(output);
       trustformers_token_ids_free(token_ids);
       trustformers_tokenizer_free(tokenizer);
       trustformers_model_free(model);
       trustformers_shutdown();

       return 0;
   }

Pipeline API
------------

For common tasks, you can use the simplified pipeline API:

.. code-block:: c

   #include <trustformers_c.h>
   #include <stdio.h>

   int main() {
       TrustformersError error;
       TrustformersPipelineHandle pipeline = NULL;

       // Create a text generation pipeline
       error = trustformers_pipeline_create("text-generation",
                                           "gpt2",
                                           &pipeline);
       if (error != TRUSTFORMERS_SUCCESS) {
           fprintf(stderr, "Failed to create pipeline: %s\n",
                   trustformers_error_message(error));
           return 1;
       }

       // Generate text
       const char* prompt = "Once upon a time";
       char* generated_text = NULL;
       error = trustformers_pipeline_generate(pipeline, prompt, &generated_text);
       if (error != TRUSTFORMERS_SUCCESS) {
           fprintf(stderr, "Failed to generate text: %s\n",
                   trustformers_error_message(error));
           trustformers_pipeline_free(pipeline);
           return 1;
       }

       printf("Generated text: %s\n", generated_text);

       // Cleanup
       trustformers_string_free(generated_text);
       trustformers_pipeline_free(pipeline);

       return 0;
   }

Next Steps
----------

* Read the :doc:`api` documentation for detailed API reference
* Check out the :doc:`examples` for more usage examples
* Learn about :doc:`error_handling` best practices
"#;

        fs::write(output_dir.join("quickstart.rst"), content)?;
        Ok(())
    }

    /// Generate examples documentation
    fn generate_examples_rst(&self, output_dir: &Path) -> TrustformersResult<()> {
        let content = r#"Examples
========

This section provides comprehensive examples for using the TrustformeRS C API.

Text Classification
-------------------

.. code-block:: c

   #include <trustformers_c.h>
   #include <stdio.h>

   int main() {
       TrustformersPipelineHandle pipeline = NULL;
       TrustformersError error;

       error = trustformers_pipeline_create("text-classification",
                                           "distilbert-base-uncased-finetuned-sst-2-english",
                                           &pipeline);

       const char* text = "I love this product!";
       TrustformersClassificationResult* result = NULL;

       error = trustformers_pipeline_classify(pipeline, text, &result);

       printf("Label: %s, Score: %.4f\n", result->label, result->score);

       trustformers_classification_result_free(result);
       trustformers_pipeline_free(pipeline);
       return 0;
   }

Question Answering
------------------

.. code-block:: c

   #include <trustformers_c.h>
   #include <stdio.h>

   int main() {
       TrustformersPipelineHandle pipeline = NULL;
       TrustformersError error;

       error = trustformers_pipeline_create("question-answering",
                                           "distilbert-base-cased-distilled-squad",
                                           &pipeline);

       const char* context = "Paris is the capital of France.";
       const char* question = "What is the capital of France?";

       TrustformersQAResult* result = NULL;
       error = trustformers_pipeline_qa(pipeline, question, context, &result);

       printf("Answer: %s (score: %.4f)\n", result->answer, result->score);

       trustformers_qa_result_free(result);
       trustformers_pipeline_free(pipeline);
       return 0;
   }

Batch Processing
----------------

.. code-block:: c

   #include <trustformers_c.h>
   #include <stdio.h>

   int main() {
       TrustformersModelHandle model = NULL;
       TrustformersTokenizerHandle tokenizer = NULL;
       TrustformersError error;

       error = trustformers_model_load("bert-base-uncased", &model);
       error = trustformers_tokenizer_load("bert-base-uncased", &tokenizer);

       // Prepare batch of texts
       const char* texts[] = {
           "First text",
           "Second text",
           "Third text"
       };
       int batch_size = 3;

       TrustformersBatch* batch = NULL;
       error = trustformers_batch_create(batch_size, &batch);

       for (int i = 0; i < batch_size; i++) {
           TrustformersTokenIds* token_ids = NULL;
           error = trustformers_tokenizer_encode(tokenizer, texts[i], &token_ids);
           error = trustformers_batch_add(batch, token_ids);
           trustformers_token_ids_free(token_ids);
       }

       TrustformersTensor* output = NULL;
       error = trustformers_model_forward_batch(model, batch, &output);

       printf("Batch inference completed!\n");

       trustformers_tensor_free(output);
       trustformers_batch_free(batch);
       trustformers_tokenizer_free(tokenizer);
       trustformers_model_free(model);
       return 0;
   }
"#;

        fs::write(output_dir.join("examples.rst"), content)?;
        Ok(())
    }

    /// Generate error handling documentation
    fn generate_error_handling_rst(&self, output_dir: &Path) -> TrustformersResult<()> {
        let content = r#"Error Handling
==============

Proper error handling is essential for robust applications using the TrustformeRS C API.

Error Codes
-----------

All API functions return a ``TrustformersError`` code. A value of ``TRUSTFORMERS_SUCCESS`` (0)
indicates success, while negative values indicate errors.

Common Error Codes
~~~~~~~~~~~~~~~~~~

* ``TRUSTFORMERS_SUCCESS`` (0) - Operation completed successfully
* ``TRUSTFORMERS_NULL_POINTER`` (-1) - Null pointer passed where valid pointer expected
* ``TRUSTFORMERS_INVALID_PARAMETER`` (-2) - Invalid parameter provided
* ``TRUSTFORMERS_OUT_OF_MEMORY`` (-3) - Memory allocation failed
* ``TRUSTFORMERS_FILE_NOT_FOUND`` (-4) - File not found or inaccessible
* ``TRUSTFORMERS_MODEL_LOAD_ERROR`` (-6) - Model loading failed
* ``TRUSTFORMERS_INFERENCE_ERROR`` (-9) - Inference execution failed

Error Messages
--------------

To get a human-readable error message, use ``trustformers_error_message()``:

.. code-block:: c

   TrustformersError error = trustformers_model_load("invalid-model", &model);
   if (error != TRUSTFORMERS_SUCCESS) {
       const char* msg = trustformers_error_message(error);
       fprintf(stderr, "Error: %s\n", msg);
   }

Best Practices
--------------

1. Always Check Return Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

   TrustformersError error;

   error = trustformers_model_load(model_name, &model);
   if (error != TRUSTFORMERS_SUCCESS) {
       // Handle error
       return error;
   }

2. Use Error Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

   TrustformersErrorDiagnostics diagnostics = {0};
   error = trustformers_error_get_diagnostics(&diagnostics);

   printf("Total errors: %lu\n", diagnostics.total_errors);
   printf("Error rate: %.2f/min\n", diagnostics.error_rate);
   printf("Health score: %.2f\n", diagnostics.health_score);

   trustformers_error_diagnostics_free(&diagnostics);

3. Cleanup on Error
~~~~~~~~~~~~~~~~~~~

Always ensure proper cleanup even when errors occur:

.. code-block:: c

   TrustformersModelHandle model = NULL;
   TrustformersTokenizerHandle tokenizer = NULL;
   TrustformersError error;

   error = trustformers_model_load(model_name, &model);
   if (error != TRUSTFORMERS_SUCCESS) {
       goto cleanup;
   }

   error = trustformers_tokenizer_load(tokenizer_name, &tokenizer);
   if (error != TRUSTFORMERS_SUCCESS) {
       goto cleanup;
   }

   // ... use model and tokenizer ...

cleanup:
   if (tokenizer) trustformers_tokenizer_free(tokenizer);
   if (model) trustformers_model_free(model);
   return error;

Error Recovery
--------------

The API provides automatic error recovery capabilities:

.. code-block:: c

   TrustformersErrorRecovery recovery_config = {
       .strategy = 0,  // Retry
       .max_retries = 3,
       .retry_delay_ms = 1000,
       .auto_recovery = 1,
       .fallback_config = NULL
   };

   int success = 0;
   error = trustformers_error_attempt_recovery(
       error_code,
       &recovery_config,
       &success
   );

   if (success) {
       printf("Error recovered successfully\n");
   }
"#;

        fs::write(output_dir.join("error_handling.rst"), content)?;
        Ok(())
    }

    /// Generate Makefile for building documentation
    fn generate_makefile(&self, output_dir: &Path) -> TrustformersResult<()> {
        let content = r#"# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
"#;

        fs::write(output_dir.join("Makefile"), content)?;
        Ok(())
    }

    /// Generate requirements.txt for Sphinx dependencies
    fn generate_requirements_txt(&self, output_dir: &Path) -> TrustformersResult<()> {
        let content = format!(
            r#"# Sphinx documentation dependencies
sphinx>=7.0.0
sphinx-rtd-theme>=2.0.0
sphinx-autodoc-typehints>=1.24.0
sphinxcontrib-napoleon>=0.7
sphinxcontrib-bibtex>=2.6.0
myst-parser>=2.0.0
"#
        );

        fs::write(output_dir.join("requirements.txt"), content)?;
        Ok(())
    }

    /// Categorize a function by its name prefix
    fn categorize_function(&self, func_name: &str) -> String {
        if func_name.starts_with("trustformers_model_") {
            "Model Operations".to_string()
        } else if func_name.starts_with("trustformers_tokenizer_") {
            "Tokenizer Operations".to_string()
        } else if func_name.starts_with("trustformers_pipeline_") {
            "Pipeline Operations".to_string()
        } else if func_name.starts_with("trustformers_tensor_") {
            "Tensor Operations".to_string()
        } else if func_name.starts_with("trustformers_error_") {
            "Error Handling".to_string()
        } else if func_name.starts_with("trustformers_config_") {
            "Configuration".to_string()
        } else if func_name.starts_with("trustformers_device_") {
            "Device Management".to_string()
        } else {
            "Core Functions".to_string()
        }
    }

    /// Format a type for reStructuredText display
    fn format_type(&self, type_info: &FfiType) -> String {
        if type_info.is_const && type_info.is_pointer {
            format!("const {}*", type_info.base_type())
        } else if type_info.is_pointer {
            format!("{}*", type_info.base_type())
        } else {
            type_info.name.clone()
        }
    }

    /// Format function parameters for reStructuredText display
    fn format_parameters(&self, params: &[FfiParameter]) -> String {
        if params.is_empty() {
            return "void".to_string();
        }

        params
            .iter()
            .map(|p| format!("{} {}", self.format_type(&p.type_info), p.name))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Get an example value for a type
    fn get_example_value(&self, type_info: &FfiType) -> String {
        if type_info.is_pointer() {
            if type_info.name.contains("char") {
                "\"example\"".to_string()
            } else {
                "&variable".to_string()
            }
        } else if type_info.name.contains("int") {
            "0".to_string()
        } else if type_info.name.contains("float") || type_info.name.contains("double") {
            "0.0".to_string()
        } else if type_info.name.contains("bool") {
            "true".to_string()
        } else {
            "value".to_string()
        }
    }
}

impl LanguageGenerator for SphinxDocGenerator {
    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::Sphinx
    }

    fn file_extension(&self) -> &'static str {
        "rst"
    }

    fn generate(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        // Create output directory structure
        fs::create_dir_all(output_dir)?;
        fs::create_dir_all(output_dir.join("_static"))?;
        fs::create_dir_all(output_dir.join("_templates"))?;

        println!("Generating Sphinx documentation in {:?}", output_dir);

        // Generate main configuration and structure files
        self.generate_conf_py(output_dir)?;
        self.generate_index_rst(interface, output_dir)?;
        self.generate_api_rst(interface, output_dir)?;

        // Generate documentation for each category
        self.generate_functions_rst(interface, output_dir)?;
        self.generate_structures_rst(interface, output_dir)?;
        self.generate_enumerations_rst(interface, output_dir)?;
        self.generate_constants_rst(interface, output_dir)?;

        // Generate guide pages
        self.generate_installation_rst(output_dir)?;
        self.generate_quickstart_rst(output_dir)?;
        self.generate_examples_rst(output_dir)?;
        self.generate_error_handling_rst(output_dir)?;

        // Generate build files
        self.generate_makefile(output_dir)?;
        self.generate_requirements_txt(output_dir)?;

        println!("Sphinx documentation generated successfully!");
        println!("To build the documentation, run:");
        println!("  cd {:?}", output_dir);
        println!("  pip install -r requirements.txt");
        println!("  make html");
        println!(
            "The built documentation will be in {:?}",
            output_dir.join("_build/html")
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::PackageInfo;
    use std::collections::HashMap;

    #[test]
    fn test_sphinx_generator_creation() {
        let config = CodeGenConfig {
            output_dir: std::path::PathBuf::from("test_output"),
            target_languages: vec![TargetLanguage::Sphinx],
            package_info: PackageInfo {
                name: "TestProject".to_string(),
                version: "1.0.0".to_string(),
                description: "Test description".to_string(),
                author: "Test Author".to_string(),
                license: "MIT".to_string(),
                repository: "https://github.com/test/test".to_string(),
            },
            features: HashMap::new(),
            type_mappings: HashMap::new(),
        };

        let generator = SphinxDocGenerator::new(&config);
        assert!(generator.is_ok());
    }

    #[test]
    fn test_categorize_function() {
        let config = CodeGenConfig::default();
        let generator = SphinxDocGenerator::new(&config).unwrap();

        assert_eq!(
            generator.categorize_function("trustformers_model_load"),
            "Model Operations"
        );
        assert_eq!(
            generator.categorize_function("trustformers_tokenizer_encode"),
            "Tokenizer Operations"
        );
        assert_eq!(
            generator.categorize_function("trustformers_pipeline_create"),
            "Pipeline Operations"
        );
    }

    #[test]
    fn test_format_type() {
        let config = CodeGenConfig::default();
        let generator = SphinxDocGenerator::new(&config).unwrap();

        let int_type = FfiType {
            name: "int".to_string(),
            is_pointer: false,
            is_const: false,
            ..Default::default()
        };
        assert_eq!(generator.format_type(&int_type), "int");

        let ptr_type = FfiType {
            name: "char".to_string(),
            is_pointer: true,
            is_const: true,
            ..Default::default()
        };
        assert_eq!(generator.format_type(&ptr_type), "const char*");
    }
}
