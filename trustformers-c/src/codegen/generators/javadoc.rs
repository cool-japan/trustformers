//! Javadoc HTML documentation generator for FFI interfaces
//!
//! Generates JavaDoc-style HTML documentation for the FFI interface,
//! including index pages, class documentation, and navigation.

use std::fs;
use std::path::Path;

use crate::codegen::ast::{
    FfiConstant, FfiEnum, FfiEnumVariant, FfiField, FfiFunction, FfiInterface, FfiParameter,
    FfiStruct, FfiType,
};
use crate::codegen::templates::TemplateEngine;
use crate::codegen::{CodeGenConfig, TargetLanguage};
use crate::error::TrustformersResult;

use super::LanguageGenerator;

/// Javadoc HTML documentation generator
pub struct JavadocGenerator {
    config: CodeGenConfig,
}

impl JavadocGenerator {
    pub fn new(config: &CodeGenConfig) -> TrustformersResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Generate the main index.html overview page
    fn generate_overview_html(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
    ) -> TrustformersResult<()> {
        let mut html = String::new();

        // HTML header
        html.push_str(&self.generate_html_header("Overview", &interface.metadata.library_name));

        // Body start
        html.push_str("<body>\n");
        html.push_str(&self.generate_top_navbar(&interface.metadata.library_name));

        html.push_str("<div class=\"flex-container\">\n");
        html.push_str(&self.generate_navigation(interface));

        // Main content
        html.push_str("<main class=\"main-content\">\n");
        html.push_str("<div class=\"header\">\n");
        html.push_str(&format!(
            "<h1 class=\"title\">{}</h1>\n",
            interface.metadata.library_name
        ));
        html.push_str(&format!(
            "<div class=\"subtitle\">Version {}</div>\n",
            interface.metadata.version
        ));
        html.push_str("</div>\n");

        // Package summary
        html.push_str("<section class=\"summary\">\n");
        html.push_str("<h2>Package Summary</h2>\n");

        // Interfaces table
        if !interface.structs.is_empty() {
            html.push_str("<table class=\"summary-table\">\n");
            html.push_str("<thead><tr><th>Interface</th><th>Description</th></tr></thead>\n");
            html.push_str("<tbody>\n");
            for struct_def in &interface.structs {
                let desc = struct_def
                    .documentation
                    .first()
                    .map(|s| self.escape_html(s))
                    .unwrap_or_else(|| "No description available.".to_string());
                html.push_str("<tr>");
                html.push_str(&format!(
                    "<td><a href=\"{}.html\" class=\"type-link\">{}</a></td>",
                    struct_def.name, struct_def.name
                ));
                html.push_str(&format!("<td>{}</td>", desc));
                html.push_str("</tr>\n");
            }
            html.push_str("</tbody>\n</table>\n");
        }

        // Enums table
        if !interface.enums.is_empty() {
            html.push_str("<h2>Enums</h2>\n");
            html.push_str("<table class=\"summary-table\">\n");
            html.push_str("<thead><tr><th>Enum</th><th>Description</th></tr></thead>\n");
            html.push_str("<tbody>\n");
            for enum_def in &interface.enums {
                let desc = enum_def
                    .documentation
                    .first()
                    .map(|s| self.escape_html(s))
                    .unwrap_or_else(|| "No description available.".to_string());
                html.push_str("<tr>");
                html.push_str(&format!(
                    "<td><a href=\"{}.html\" class=\"type-link\">{}</a></td>",
                    enum_def.name, enum_def.name
                ));
                html.push_str(&format!("<td>{}</td>", desc));
                html.push_str("</tr>\n");
            }
            html.push_str("</tbody>\n</table>\n");
        }

        html.push_str("</section>\n");
        html.push_str("</main>\n");
        html.push_str("</div>\n");

        // Footer
        html.push_str(&self.generate_footer());
        html.push_str("</body>\n</html>");

        fs::write(output_dir.join("index.html"), html)?;
        Ok(())
    }

    /// Generate HTML documentation for a struct
    fn generate_class_html(
        &self,
        struct_def: &FfiStruct,
        interface: &FfiInterface,
        output_dir: &Path,
    ) -> TrustformersResult<()> {
        let mut html = String::new();

        html.push_str(
            &self.generate_html_header(&struct_def.name, &interface.metadata.library_name),
        );
        html.push_str("<body>\n");
        html.push_str(&self.generate_top_navbar(&interface.metadata.library_name));

        html.push_str("<div class=\"flex-container\">\n");
        html.push_str(&self.generate_navigation(interface));

        html.push_str("<main class=\"main-content\">\n");

        // Class header
        html.push_str("<div class=\"header\">\n");
        html.push_str("<div class=\"package-name\">Package</div>\n");
        html.push_str(&format!(
            "<h1 class=\"title\">Struct {}</h1>\n",
            struct_def.name
        ));
        html.push_str("</div>\n");

        // Class description
        if !struct_def.documentation.is_empty() {
            html.push_str("<div class=\"description\">\n");
            for line in &struct_def.documentation {
                html.push_str(&format!("<p>{}</p>\n", self.escape_html(line)));
            }
            html.push_str("</div>\n");
        }

        // Deprecation warning
        if let Some(deprecation) = &struct_def.deprecation {
            html.push_str("<div class=\"deprecation-warning\">\n");
            html.push_str("<strong>Deprecated.</strong> ");
            html.push_str(&self.escape_html(&deprecation.message));
            if let Some(replacement) = &deprecation.replacement {
                html.push_str(&format!(
                    " Use <code>{}</code> instead.",
                    self.escape_html(replacement)
                ));
            }
            html.push_str("</div>\n");
        }

        // Field summary
        if !struct_def.is_opaque && !struct_def.fields.is_empty() {
            html.push_str("<section class=\"summary\">\n");
            html.push_str("<h2>Field Summary</h2>\n");
            html.push_str("<table class=\"summary-table\">\n");
            html.push_str("<thead><tr><th>Modifier and Type</th><th>Field</th><th>Description</th></tr></thead>\n");
            html.push_str("<tbody>\n");

            for field in &struct_def.fields {
                if !field.is_private {
                    html.push_str("<tr>");
                    html.push_str(&format!(
                        "<td><code>{}</code></td>",
                        self.format_type(&field.type_info)
                    ));
                    html.push_str(&format!(
                        "<td><code><a href=\"#{}\">{}</a></code></td>",
                        field.name, field.name
                    ));
                    let desc = field
                        .documentation
                        .first()
                        .map(|s| self.escape_html(s))
                        .unwrap_or_else(|| "".to_string());
                    html.push_str(&format!("<td>{}</td>", desc));
                    html.push_str("</tr>\n");
                }
            }

            html.push_str("</tbody>\n</table>\n");
            html.push_str("</section>\n");

            // Field details
            html.push_str("<section class=\"details\">\n");
            html.push_str("<h2>Field Details</h2>\n");

            for field in &struct_def.fields {
                if !field.is_private {
                    html.push_str(&format!("<div class=\"member\" id=\"{}\">\n", field.name));
                    html.push_str(&format!("<h3>{}</h3>\n", field.name));
                    html.push_str("<pre class=\"signature\">");
                    html.push_str(&format!(
                        "public {} {}",
                        self.format_type(&field.type_info),
                        field.name
                    ));
                    html.push_str("</pre>\n");

                    if !field.documentation.is_empty() {
                        html.push_str("<div class=\"description\">\n");
                        for line in &field.documentation {
                            html.push_str(&format!("<p>{}</p>\n", self.escape_html(line)));
                        }
                        html.push_str("</div>\n");
                    }

                    html.push_str("</div>\n");
                }
            }

            html.push_str("</section>\n");
        }

        // Related functions
        let related_functions: Vec<&FfiFunction> = interface
            .functions
            .iter()
            .filter(|f| {
                f.parameters.iter().any(|p| p.type_info.name.contains(&struct_def.name))
                    || f.return_type.name.contains(&struct_def.name)
            })
            .collect();

        if !related_functions.is_empty() {
            html.push_str("<section class=\"summary\">\n");
            html.push_str("<h2>Related Functions</h2>\n");
            html.push_str("<table class=\"summary-table\">\n");
            html.push_str("<thead><tr><th>Function</th><th>Description</th></tr></thead>\n");
            html.push_str("<tbody>\n");

            for func in related_functions {
                html.push_str("<tr>");
                html.push_str(&format!(
                    "<td><a href=\"functions.html#{}\" class=\"type-link\"><code>{}</code></a></td>",
                    func.name, func.name
                ));
                let desc = func
                    .documentation
                    .first()
                    .map(|s| self.escape_html(s))
                    .unwrap_or_else(|| "".to_string());
                html.push_str(&format!("<td>{}</td>", desc));
                html.push_str("</tr>\n");
            }

            html.push_str("</tbody>\n</table>\n");
            html.push_str("</section>\n");
        }

        html.push_str("</main>\n");
        html.push_str("</div>\n");
        html.push_str(&self.generate_footer());
        html.push_str("</body>\n</html>");

        fs::write(output_dir.join(format!("{}.html", struct_def.name)), html)?;
        Ok(())
    }

    /// Generate HTML documentation for an enum
    fn generate_enum_html(
        &self,
        enum_def: &FfiEnum,
        interface: &FfiInterface,
        output_dir: &Path,
    ) -> TrustformersResult<()> {
        let mut html = String::new();

        html.push_str(&self.generate_html_header(&enum_def.name, &interface.metadata.library_name));
        html.push_str("<body>\n");
        html.push_str(&self.generate_top_navbar(&interface.metadata.library_name));

        html.push_str("<div class=\"flex-container\">\n");
        html.push_str(&self.generate_navigation(interface));

        html.push_str("<main class=\"main-content\">\n");

        // Enum header
        html.push_str("<div class=\"header\">\n");
        html.push_str("<div class=\"package-name\">Package</div>\n");
        html.push_str(&format!(
            "<h1 class=\"title\">Enum {}</h1>\n",
            enum_def.name
        ));
        html.push_str("</div>\n");

        // Enum description
        if !enum_def.documentation.is_empty() {
            html.push_str("<div class=\"description\">\n");
            for line in &enum_def.documentation {
                html.push_str(&format!("<p>{}</p>\n", self.escape_html(line)));
            }
            html.push_str("</div>\n");
        }

        // Deprecation warning
        if let Some(deprecation) = &enum_def.deprecation {
            html.push_str("<div class=\"deprecation-warning\">\n");
            html.push_str("<strong>Deprecated.</strong> ");
            html.push_str(&self.escape_html(&deprecation.message));
            if let Some(replacement) = &deprecation.replacement {
                html.push_str(&format!(
                    " Use <code>{}</code> instead.",
                    self.escape_html(replacement)
                ));
            }
            html.push_str("</div>\n");
        }

        // Enum constants summary
        if !enum_def.variants.is_empty() {
            html.push_str("<section class=\"summary\">\n");
            html.push_str("<h2>Enum Constants</h2>\n");
            html.push_str("<table class=\"summary-table\">\n");
            html.push_str(
                "<thead><tr><th>Constant</th><th>Value</th><th>Description</th></tr></thead>\n",
            );
            html.push_str("<tbody>\n");

            for variant in &enum_def.variants {
                html.push_str("<tr>");
                html.push_str(&format!(
                    "<td><code><a href=\"#{}\">{}</a></code></td>",
                    variant.name, variant.name
                ));
                html.push_str(&format!("<td><code>{}</code></td>", variant.value));
                let desc = variant
                    .documentation
                    .first()
                    .map(|s| self.escape_html(s))
                    .unwrap_or_else(|| "".to_string());
                html.push_str(&format!("<td>{}</td>", desc));
                html.push_str("</tr>\n");
            }

            html.push_str("</tbody>\n</table>\n");
            html.push_str("</section>\n");

            // Enum constant details
            html.push_str("<section class=\"details\">\n");
            html.push_str("<h2>Enum Constant Details</h2>\n");

            for variant in &enum_def.variants {
                html.push_str(&format!("<div class=\"member\" id=\"{}\">\n", variant.name));
                html.push_str(&format!("<h3>{}</h3>\n", variant.name));
                html.push_str("<pre class=\"signature\">");
                html.push_str(&format!(
                    "public static final int {} = {}",
                    variant.name, variant.value
                ));
                html.push_str("</pre>\n");

                if !variant.documentation.is_empty() {
                    html.push_str("<div class=\"description\">\n");
                    for line in &variant.documentation {
                        html.push_str(&format!("<p>{}</p>\n", self.escape_html(line)));
                    }
                    html.push_str("</div>\n");
                }

                if let Some(deprecation) = &variant.deprecation {
                    html.push_str("<div class=\"deprecation-warning\">\n");
                    html.push_str("<strong>Deprecated.</strong> ");
                    html.push_str(&self.escape_html(&deprecation.message));
                    html.push_str("</div>\n");
                }

                html.push_str("</div>\n");
            }

            html.push_str("</section>\n");
        }

        html.push_str("</main>\n");
        html.push_str("</div>\n");
        html.push_str(&self.generate_footer());
        html.push_str("</body>\n</html>");

        fs::write(output_dir.join(format!("{}.html", enum_def.name)), html)?;
        Ok(())
    }

    /// Generate function documentation HTML
    fn generate_function_html(&self, func: &FfiFunction) -> String {
        let mut html = String::new();

        html.push_str(&format!("<div class=\"member\" id=\"{}\">\n", func.name));
        html.push_str(&format!("<h3>{}</h3>\n", func.name));

        // Function signature
        html.push_str("<pre class=\"signature\">");
        html.push_str(&format!(
            "{} {}",
            self.format_type(&func.return_type),
            func.name
        ));
        html.push_str("(");

        for (i, param) in func.parameters.iter().enumerate() {
            if i > 0 {
                html.push_str(",\n    ");
            }
            html.push_str(&format!(
                "{} {}",
                self.format_type(&param.type_info),
                param.name
            ));
        }

        html.push_str(")");
        html.push_str("</pre>\n");

        // Function description
        if !func.documentation.is_empty() {
            html.push_str("<div class=\"description\">\n");
            for line in &func.documentation {
                html.push_str(&format!("<p>{}</p>\n", self.escape_html(line)));
            }
            html.push_str("</div>\n");
        }

        // Parameters
        if !func.parameters.is_empty() {
            html.push_str("<dl class=\"param-list\">\n");
            html.push_str("<dt>Parameters:</dt>\n");
            for param in &func.parameters {
                html.push_str("<dd>");
                html.push_str(&format!("<code>{}</code> - ", param.name));
                if !param.documentation.is_empty() {
                    html.push_str(&self.escape_html(&param.documentation.join(" ")));
                } else {
                    html.push_str(&format!(
                        "parameter of type {}",
                        self.format_type(&param.type_info)
                    ));
                }
                html.push_str("</dd>\n");
            }
            html.push_str("</dl>\n");
        }

        // Return value
        if func.return_type.name != "void" {
            html.push_str("<dl class=\"param-list\">\n");
            html.push_str("<dt>Returns:</dt>\n");
            html.push_str(&format!(
                "<dd>{}</dd>\n",
                self.format_type(&func.return_type)
            ));
            html.push_str("</dl>\n");
        }

        // Deprecation warning
        if let Some(deprecation) = &func.deprecation {
            html.push_str("<div class=\"deprecation-warning\">\n");
            html.push_str("<strong>Deprecated.</strong> ");
            html.push_str(&self.escape_html(&deprecation.message));
            if let Some(replacement) = &deprecation.replacement {
                html.push_str(&format!(
                    " Use <code>{}</code> instead.",
                    self.escape_html(replacement)
                ));
            }
            html.push_str("</div>\n");
        }

        html.push_str("</div>\n");

        html
    }

    /// Generate all functions page
    fn generate_functions_page(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
    ) -> TrustformersResult<()> {
        let mut html = String::new();

        html.push_str(&self.generate_html_header("Functions", &interface.metadata.library_name));
        html.push_str("<body>\n");
        html.push_str(&self.generate_top_navbar(&interface.metadata.library_name));

        html.push_str("<div class=\"flex-container\">\n");
        html.push_str(&self.generate_navigation(interface));

        html.push_str("<main class=\"main-content\">\n");
        html.push_str("<div class=\"header\">\n");
        html.push_str("<h1 class=\"title\">Functions</h1>\n");
        html.push_str("</div>\n");

        // Function summary
        html.push_str("<section class=\"summary\">\n");
        html.push_str("<h2>Function Summary</h2>\n");
        html.push_str("<table class=\"summary-table\">\n");
        html.push_str("<thead><tr><th>Function</th><th>Description</th></tr></thead>\n");
        html.push_str("<tbody>\n");

        for func in &interface.functions {
            html.push_str("<tr>");
            html.push_str(&format!(
                "<td><a href=\"#{}\" class=\"type-link\"><code>{}</code></a></td>",
                func.name, func.name
            ));
            let desc = func
                .documentation
                .first()
                .map(|s| self.escape_html(s))
                .unwrap_or_else(|| "".to_string());
            html.push_str(&format!("<td>{}</td>", desc));
            html.push_str("</tr>\n");
        }

        html.push_str("</tbody>\n</table>\n");
        html.push_str("</section>\n");

        // Function details
        html.push_str("<section class=\"details\">\n");
        html.push_str("<h2>Function Details</h2>\n");

        for func in &interface.functions {
            html.push_str(&self.generate_function_html(func));
        }

        html.push_str("</section>\n");
        html.push_str("</main>\n");
        html.push_str("</div>\n");
        html.push_str(&self.generate_footer());
        html.push_str("</body>\n</html>");

        fs::write(output_dir.join("functions.html"), html)?;
        Ok(())
    }

    /// Generate navigation sidebar
    fn generate_navigation(&self, interface: &FfiInterface) -> String {
        let mut nav = String::new();

        nav.push_str("<nav class=\"sidebar\">\n");
        nav.push_str("<div class=\"nav-header\">Navigation</div>\n");
        nav.push_str("<ul class=\"nav-list\">\n");

        // Overview link
        nav.push_str("<li><a href=\"index.html\">Overview</a></li>\n");

        // Interfaces/Structs
        if !interface.structs.is_empty() {
            nav.push_str("<li class=\"nav-section\">Interfaces</li>\n");
            nav.push_str("<ul class=\"nav-sublist\">\n");
            for struct_def in &interface.structs {
                nav.push_str(&format!(
                    "<li><a href=\"{}.html\">{}</a></li>\n",
                    struct_def.name, struct_def.name
                ));
            }
            nav.push_str("</ul>\n");
        }

        // Enums
        if !interface.enums.is_empty() {
            nav.push_str("<li class=\"nav-section\">Enums</li>\n");
            nav.push_str("<ul class=\"nav-sublist\">\n");
            for enum_def in &interface.enums {
                nav.push_str(&format!(
                    "<li><a href=\"{}.html\">{}</a></li>\n",
                    enum_def.name, enum_def.name
                ));
            }
            nav.push_str("</ul>\n");
        }

        // Functions
        if !interface.functions.is_empty() {
            nav.push_str("<li><a href=\"functions.html\">Functions</a></li>\n");
        }

        nav.push_str("</ul>\n");
        nav.push_str("</nav>\n");

        nav
    }

    /// Generate allclasses-frame.html for class listing
    fn generate_allclasses_frame(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
    ) -> TrustformersResult<()> {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<meta charset=\"UTF-8\">\n");
        html.push_str("<title>All Classes</title>\n");
        html.push_str("<link rel=\"stylesheet\" href=\"stylesheet.css\">\n");
        html.push_str("</head>\n<body>\n");

        html.push_str("<h1 class=\"bar\">All Classes</h1>\n");
        html.push_str("<div class=\"index-container\">\n");
        html.push_str("<ul>\n");

        // List all structs
        for struct_def in &interface.structs {
            html.push_str(&format!(
                "<li><a href=\"{}.html\" target=\"classFrame\">{}</a></li>\n",
                struct_def.name, struct_def.name
            ));
        }

        // List all enums
        for enum_def in &interface.enums {
            html.push_str(&format!(
                "<li><a href=\"{}.html\" target=\"classFrame\">{}</a></li>\n",
                enum_def.name, enum_def.name
            ));
        }

        html.push_str("</ul>\n");
        html.push_str("</div>\n");
        html.push_str("</body>\n</html>");

        fs::write(output_dir.join("allclasses-frame.html"), html)?;
        Ok(())
    }

    /// Generate overview-tree.html for type hierarchy
    fn generate_overview_tree(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
    ) -> TrustformersResult<()> {
        let mut html = String::new();

        html.push_str(
            &self.generate_html_header("Class Hierarchy", &interface.metadata.library_name),
        );
        html.push_str("<body>\n");
        html.push_str(&self.generate_top_navbar(&interface.metadata.library_name));

        html.push_str("<div class=\"flex-container\">\n");
        html.push_str(&self.generate_navigation(interface));

        html.push_str("<main class=\"main-content\">\n");
        html.push_str("<div class=\"header\">\n");
        html.push_str("<h1 class=\"title\">Hierarchy For All Packages</h1>\n");
        html.push_str("</div>\n");

        html.push_str("<section class=\"hierarchy\">\n");
        html.push_str("<h2>Struct Hierarchy</h2>\n");
        html.push_str("<ul>\n");

        for struct_def in &interface.structs {
            html.push_str(&format!(
                "<li><a href=\"{}.html\">{}</a></li>\n",
                struct_def.name, struct_def.name
            ));
        }

        html.push_str("</ul>\n");

        html.push_str("<h2>Enum Hierarchy</h2>\n");
        html.push_str("<ul>\n");

        for enum_def in &interface.enums {
            html.push_str(&format!(
                "<li><a href=\"{}.html\">{}</a></li>\n",
                enum_def.name, enum_def.name
            ));
        }

        html.push_str("</ul>\n");
        html.push_str("</section>\n");

        html.push_str("</main>\n");
        html.push_str("</div>\n");
        html.push_str(&self.generate_footer());
        html.push_str("</body>\n</html>");

        fs::write(output_dir.join("overview-tree.html"), html)?;
        Ok(())
    }

    /// Generate CSS stylesheet
    fn generate_stylesheet(&self, output_dir: &Path) -> TrustformersResult<()> {
        let css = r#"/* Javadoc Stylesheet */

:root {
    --primary-color: #353833;
    --secondary-color: #4a5568;
    --accent-color: #3182ce;
    --background: #ffffff;
    --sidebar-bg: #f7fafc;
    --border-color: #e2e8f0;
    --code-bg: #f5f5f5;
    --deprecated-bg: #fff5e1;
    --link-color: #3182ce;
    --link-hover: #2c5aa0;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Arial', 'Helvetica Neue', sans-serif;
    font-size: 14px;
    line-height: 1.6;
    color: var(--primary-color);
    background: var(--background);
}

a {
    color: var(--link-color);
    text-decoration: none;
}

a:hover {
    color: var(--link-hover);
    text-decoration: underline;
}

/* Top Navigation Bar */
.top-navbar {
    background: var(--primary-color);
    color: white;
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.top-navbar .brand {
    font-size: 18px;
    font-weight: bold;
}

.top-navbar a {
    color: white;
    margin-left: 20px;
}

/* Layout */
.flex-container {
    display: flex;
    min-height: calc(100vh - 100px);
}

.sidebar {
    width: 250px;
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border-color);
    padding: 20px;
    overflow-y: auto;
}

.main-content {
    flex: 1;
    padding: 30px;
    max-width: 1200px;
}

/* Navigation */
.nav-header {
    font-size: 16px;
    font-weight: bold;
    margin-bottom: 15px;
    color: var(--primary-color);
}

.nav-list {
    list-style: none;
}

.nav-list > li {
    margin-bottom: 8px;
}

.nav-section {
    font-weight: bold;
    margin-top: 15px;
    margin-bottom: 5px;
    color: var(--secondary-color);
}

.nav-sublist {
    list-style: none;
    margin-left: 15px;
}

.nav-sublist li {
    margin-bottom: 5px;
}

/* Header */
.header {
    margin-bottom: 30px;
    border-bottom: 2px solid var(--border-color);
    padding-bottom: 20px;
}

.header .title {
    font-size: 28px;
    font-weight: 300;
    color: var(--primary-color);
}

.header .subtitle {
    font-size: 14px;
    color: var(--secondary-color);
    margin-top: 5px;
}

.header .package-name {
    font-size: 12px;
    color: var(--secondary-color);
    text-transform: uppercase;
    margin-bottom: 5px;
}

/* Description */
.description {
    margin: 20px 0;
    line-height: 1.8;
}

.description p {
    margin-bottom: 10px;
}

/* Tables */
.summary-table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    background: white;
    border: 1px solid var(--border-color);
}

.summary-table thead {
    background: var(--sidebar-bg);
}

.summary-table th {
    text-align: left;
    padding: 12px;
    font-weight: 600;
    border-bottom: 2px solid var(--border-color);
}

.summary-table td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border-color);
}

.summary-table tbody tr:hover {
    background: #f8f9fa;
}

/* Code and Signatures */
code {
    background: var(--code-bg);
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
}

pre.signature {
    background: var(--code-bg);
    padding: 15px;
    border-left: 3px solid var(--accent-color);
    border-radius: 4px;
    overflow-x: auto;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    margin: 15px 0;
}

/* Sections */
.summary {
    margin: 30px 0;
}

.summary h2 {
    font-size: 20px;
    font-weight: 500;
    margin-bottom: 15px;
    color: var(--primary-color);
}

.details {
    margin: 30px 0;
}

.details h2 {
    font-size: 20px;
    font-weight: 500;
    margin-bottom: 20px;
    color: var(--primary-color);
}

/* Member Details */
.member {
    margin: 30px 0;
    padding: 20px;
    background: white;
    border: 1px solid var(--border-color);
    border-radius: 4px;
}

.member h3 {
    font-size: 18px;
    font-weight: 500;
    margin-bottom: 15px;
    color: var(--primary-color);
}

/* Parameter Lists */
.param-list {
    margin: 15px 0;
}

.param-list dt {
    font-weight: 600;
    margin-bottom: 5px;
}

.param-list dd {
    margin-left: 20px;
    margin-bottom: 8px;
}

/* Deprecation Warning */
.deprecation-warning {
    background: var(--deprecated-bg);
    border-left: 4px solid #ff9800;
    padding: 15px;
    margin: 15px 0;
    border-radius: 4px;
}

.deprecation-warning strong {
    color: #e65100;
}

/* Footer */
.footer {
    background: var(--sidebar-bg);
    padding: 20px;
    text-align: center;
    border-top: 1px solid var(--border-color);
    color: var(--secondary-color);
    font-size: 12px;
}

/* Type Links */
.type-link {
    font-weight: 500;
}

/* Hierarchy */
.hierarchy ul {
    list-style: none;
    margin-left: 20px;
}

.hierarchy li {
    margin: 8px 0;
}

/* Index Container */
.index-container {
    padding: 20px;
}

.index-container ul {
    list-style: none;
}

.index-container li {
    margin: 5px 0;
}

/* Responsive Design */
@media (max-width: 768px) {
    .flex-container {
        flex-direction: column;
    }

    .sidebar {
        width: 100%;
        border-right: none;
        border-bottom: 1px solid var(--border-color);
    }

    .main-content {
        padding: 20px;
    }

    .summary-table {
        font-size: 12px;
    }

    .summary-table th,
    .summary-table td {
        padding: 8px;
    }
}
"#;

        fs::write(output_dir.join("stylesheet.css"), css)?;
        Ok(())
    }

    /// Generate JavaScript for navigation and search
    fn generate_javascript(&self, output_dir: &Path) -> TrustformersResult<()> {
        let js = r##"// Javadoc JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Highlight current page in navigation
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-list a');

    navLinks.forEach(link => {
        if (link.getAttribute('href') && currentPath.includes(link.getAttribute('href'))) {
            link.style.fontWeight = 'bold';
            link.style.color = '#3182ce';
        }
    });

    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add copy button to code blocks
    document.querySelectorAll('pre.signature').forEach(pre => {
        const button = document.createElement('button');
        button.textContent = 'Copy';
        button.className = 'copy-button';
        button.style.cssText = 'position: absolute; top: 5px; right: 5px; padding: 5px 10px; background: #3182ce; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 12px;';

        pre.style.position = 'relative';
        pre.appendChild(button);

        button.addEventListener('click', function() {
            const text = pre.textContent.replace('Copy', '').trim();
            navigator.clipboard.writeText(text).then(() => {
                button.textContent = 'Copied!';
                setTimeout(() => {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
    });
});
"##;

        fs::write(output_dir.join("script.js"), js)?;
        Ok(())
    }

    /// Generate HTML header with meta tags
    fn generate_html_header(&self, title: &str, library_name: &str) -> String {
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{} - {} Documentation</title>
    <link rel="stylesheet" href="stylesheet.css">
    <script src="script.js"></script>
</head>
"#,
            title, library_name
        )
    }

    /// Generate top navigation bar
    fn generate_top_navbar(&self, library_name: &str) -> String {
        format!(
            r#"<div class="top-navbar">
    <div class="brand">{}</div>
    <div>
        <a href="index.html">Overview</a>
        <a href="overview-tree.html">Tree</a>
        <a href="functions.html">Functions</a>
    </div>
</div>
"#,
            library_name
        )
    }

    /// Generate footer
    fn generate_footer(&self) -> String {
        format!(
            r#"<div class="footer">
    <p>Generated by TrustformeRS Javadoc Generator</p>
    <p>Copyright © {} {}. All rights reserved.</p>
</div>
"#,
            chrono::Utc::now().format("%Y"),
            self.config.package_info.author
        )
    }

    /// Format FFI type for display
    fn format_type(&self, type_info: &FfiType) -> String {
        let base = type_info.map_to_language(&TargetLanguage::Java);

        if type_info.is_pointer() && !type_info.is_string() {
            if type_info.is_const() {
                format!("final {}", base)
            } else {
                base
            }
        } else {
            base
        }
    }

    /// Escape HTML special characters
    fn escape_html(&self, text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#39;")
    }
}

impl LanguageGenerator for JavadocGenerator {
    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::Javadoc
    }

    fn file_extension(&self) -> &'static str {
        "html"
    }

    fn generate(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        // Create output directory if it doesn't exist
        fs::create_dir_all(output_dir)?;

        // Generate CSS and JavaScript
        self.generate_stylesheet(output_dir)?;
        self.generate_javascript(output_dir)?;

        // Generate main overview page
        self.generate_overview_html(interface, output_dir)?;

        // Generate individual struct pages
        for struct_def in &interface.structs {
            self.generate_class_html(struct_def, interface, output_dir)?;
        }

        // Generate individual enum pages
        for enum_def in &interface.enums {
            self.generate_enum_html(enum_def, interface, output_dir)?;
        }

        // Generate functions page
        if !interface.functions.is_empty() {
            self.generate_functions_page(interface, output_dir)?;
        }

        // Generate auxiliary pages
        self.generate_allclasses_frame(interface, output_dir)?;
        self.generate_overview_tree(interface, output_dir)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::CodeGenConfig;
    use tempfile::TempDir;

    #[test]
    fn test_javadoc_generator_creation() {
        let config = CodeGenConfig::default();
        let generator = JavadocGenerator::new(&config);
        assert!(generator.is_ok());
    }

    #[test]
    fn test_target_language() {
        let config = CodeGenConfig::default();
        let generator = JavadocGenerator::new(&config).unwrap();
        assert_eq!(generator.target_language(), TargetLanguage::Javadoc);
    }

    #[test]
    fn test_file_extension() {
        let config = CodeGenConfig::default();
        let generator = JavadocGenerator::new(&config).unwrap();
        assert_eq!(generator.file_extension(), "html");
    }

    #[test]
    fn test_escape_html() {
        let config = CodeGenConfig::default();
        let generator = JavadocGenerator::new(&config).unwrap();

        assert_eq!(generator.escape_html("<div>"), "&lt;div&gt;");
        assert_eq!(generator.escape_html("a & b"), "a &amp; b");
        assert_eq!(generator.escape_html("\"quote\""), "&quot;quote&quot;");
    }

    #[test]
    fn test_generate_empty_interface() {
        let config = CodeGenConfig::default();
        let generator = JavadocGenerator::new(&config).unwrap();
        let temp_dir = TempDir::new().unwrap();
        let interface = FfiInterface::default();
        let templates = TemplateEngine::new().unwrap();

        let result = generator.generate(&interface, temp_dir.path(), &templates);
        assert!(result.is_ok());

        // Check that files were created
        assert!(temp_dir.path().join("index.html").exists());
        assert!(temp_dir.path().join("stylesheet.css").exists());
        assert!(temp_dir.path().join("script.js").exists());
    }

    #[test]
    fn test_generate_with_structs() {
        let config = CodeGenConfig::default();
        let generator = JavadocGenerator::new(&config).unwrap();
        let temp_dir = TempDir::new().unwrap();

        let mut interface = FfiInterface::default();
        interface.structs.push(FfiStruct {
            name: "TestStruct".to_string(),
            c_name: "test_struct".to_string(),
            documentation: vec!["A test struct".to_string()],
            fields: vec![],
            is_opaque: false,
            is_packed: false,
            alignment: None,
            required_features: vec![],
            platforms: vec![],
            deprecation: None,
        });

        let templates = TemplateEngine::new().unwrap();
        let result = generator.generate(&interface, temp_dir.path(), &templates);
        assert!(result.is_ok());

        // Check that struct page was created
        assert!(temp_dir.path().join("TestStruct.html").exists());
    }
}
