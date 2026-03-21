//! Language-specific code generators for FFI bindings
//!
//! Contains generators for different target languages that convert the
//! parsed FFI interface into language-specific binding code.

use anyhow::Result;
use std::path::Path;

use super::ast::*;
use super::templates::TemplateEngine;
use super::{CodeGenConfig, TargetLanguage};
use crate::error::TrustformersResult;

// Common utilities
pub mod common;

// Language-specific generators
pub mod csharp;
pub mod go;
pub mod java;
pub mod javascript;
pub mod kotlin;
pub mod php;
pub mod python;
pub mod ruby;
pub mod swift;
pub mod typescript;

// Documentation generators
pub mod javadoc;
pub mod markdown_doc;
pub mod openapi;
pub mod sphinx_doc;
pub mod typedoc;

// Re-export generators for convenience
pub use csharp::CSharpGenerator;
pub use go::GoGenerator;
pub use java::JavaGenerator;
pub use javadoc::JavadocGenerator;
pub use javascript::JavaScriptGenerator;
pub use kotlin::KotlinGenerator;
pub use markdown_doc::MarkdownDocGenerator;
pub use openapi::OpenApiGenerator;
pub use php::PhpGenerator;
pub use python::PythonGenerator;
pub use ruby::RubyGenerator;
pub use sphinx_doc::SphinxDocGenerator;
pub use swift::SwiftGenerator;
pub use typedoc::TypeDocGenerator;
pub use typescript::TypeScriptGenerator;

/// Trait for language-specific code generators
pub trait LanguageGenerator {
    /// Generate bindings for the given FFI interface
    fn generate(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        templates: &TemplateEngine,
    ) -> TrustformersResult<()>;

    /// Get the language this generator targets
    fn target_language(&self) -> TargetLanguage;

    /// Get file extension for generated files
    fn file_extension(&self) -> &'static str;

    /// Generate package/project files (setup.py, package.json, etc.)
    fn generate_package_files(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        // Default implementation - do nothing
        let _ = (interface, output_dir, templates);
        Ok(())
    }

    /// Generate example/demo files
    fn generate_examples(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        // Default implementation - do nothing
        let _ = (interface, output_dir, templates);
        Ok(())
    }

    /// Generate test files
    fn generate_tests(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        // Default implementation - do nothing
        let _ = (interface, output_dir, templates);
        Ok(())
    }
}
