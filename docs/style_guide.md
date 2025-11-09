# TrustformeRS Code Style Guide

This document outlines the code style conventions and automation tools used in the TrustformeRS project.

## Table of Contents

1. [Overview](#overview)
2. [Rust Code Style](#rust-code-style)
3. [Documentation Style](#documentation-style)
4. [Commit Message Format](#commit-message-format)
5. [File Organization](#file-organization)
6. [Automation Tools](#automation-tools)
7. [Pre-commit Hooks](#pre-commit-hooks)
8. [CI/CD Integration](#cicd-integration)
9. [Editor Configuration](#editor-configuration)

## Overview

TrustformeRS enforces consistent code style through automated tooling. All code must pass formatting and linting checks before being merged.

### Quick Start

```bash
# Install development tools
make install-tools

# Set up git hooks
./scripts/setup-hooks.sh

# Run all checks
make check

# Fix formatting issues
make fmt
```

## Rust Code Style

### Formatting (rustfmt)

We use `rustfmt` with custom configuration (see `rustfmt.toml`):

- **Indentation**: 4 spaces
- **Max line width**: 100 characters
- **Import grouping**: Std, External, Crate
- **Trailing commas**: Always in vertical layouts
- **Brace style**: Same line where possible

Run formatting:
```bash
cargo fmt --all
```

### Linting (clippy)

We use `clippy` with strict settings (see `clippy.toml`):

- **All clippy lints**: Enabled as warnings
- **Pedantic lints**: Enabled
- **Nursery lints**: Enabled (experimental lints)
- **Cognitive complexity**: Max 30 per function
- **Function length**: Max 100 lines

Run linting:
```bash
cargo clippy --all-targets --all-features -- -D warnings
```

### Naming Conventions

- **Modules**: `snake_case`
- **Types**: `PascalCase`
- **Functions/Methods**: `snake_case`
- **Constants**: `SCREAMING_SNAKE_CASE`
- **Type parameters**: Single capital letter or `PascalCase`

### Code Organization

1. **Imports** (grouped and alphabetized):
   ```rust
   // Standard library
   use std::collections::HashMap;
   use std::sync::Arc;
   
   // External crates
   use serde::{Deserialize, Serialize};
   use tokio::sync::RwLock;
   
   // Internal crates
   use crate::tensor::Tensor;
   use crate::utils::config::Config;
   ```

2. **Module structure**:
   ```rust
   //! Module documentation
   
   // Imports
   
   // Constants
   
   // Type definitions
   
   // Trait definitions
   
   // Struct/Enum definitions
   
   // Implementations
   
   // Functions
   
   // Tests
   #[cfg(test)]
   mod tests {
       use super::*;
       // Test code
   }
   ```

## Documentation Style

### Rust Documentation

- Use `///` for public item documentation
- Use `//!` for module-level documentation
- Include examples in doc comments:

```rust
/// Performs matrix multiplication on two tensors.
///
/// # Arguments
///
/// * `other` - The right-hand side tensor
///
/// # Returns
///
/// A new tensor containing the result
///
/// # Example
///
/// ```
/// use trustformers_core::Tensor;
///
/// let a = Tensor::ones(&[2, 3]);
/// let b = Tensor::ones(&[3, 4]);
/// let c = a.matmul(&b);
/// assert_eq!(c.shape(), &[2, 4]);
/// ```
pub fn matmul(&self, other: &Tensor) -> Tensor {
    // Implementation
}
```

### Markdown Documentation

We use markdownlint with custom rules (see `.markdownlint.yaml`):

- **Line length**: 100 characters (except tables and code blocks)
- **Headings**: ATX style (`#` not underlines)
- **Lists**: Dash (`-`) for unordered lists
- **Code blocks**: Fenced with language tags

## Commit Message Format

We follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test changes
- `chore`: Maintenance tasks
- `build`: Build system changes
- `ci`: CI/CD changes
- `revert`: Revert previous commit

### Examples

```
feat(models): add flash attention support

Implement Flash Attention v2 for improved memory efficiency
in transformer models. This reduces memory usage by 4x for
long sequences.

Closes #123
```

## File Organization

### Directory Structure

```
trustformers/
├── trustformers-core/      # Core tensor operations
├── trustformers-models/    # Model implementations
├── trustformers-optim/     # Optimizers
├── trustformers-data/      # Data loaders
├── trustformers-tokenizers/# Tokenizers
├── trustformers-training/  # Training utilities
├── trustformers-py/        # Python bindings
├── trustformers-mobile/    # Mobile framework
├── docs/                   # Documentation
├── examples/              # Example code
├── benches/              # Benchmarks
└── scripts/              # Utility scripts
```

### File Naming

- Rust source files: `snake_case.rs`
- Test files: `*_test.rs` or in `tests/` directory
- Benchmark files: `*_bench.rs` or in `benches/` directory
- Documentation: `snake_case.md`

## Automation Tools

### Available Tools

1. **rustfmt** - Code formatting
2. **clippy** - Linting
3. **cargo-audit** - Security vulnerability scanning
4. **cargo-deny** - Dependency license checking
5. **typos** - Spell checking
6. **cargo-nextest** - Fast test runner
7. **pre-commit** - Git hook management

### Makefile Commands

```bash
make help          # Show all available commands
make check         # Run all checks
make fmt          # Format code
make lint         # Run clippy
make test         # Run tests
make audit        # Security audit
make deny         # License check
make typos        # Spell check
make coverage     # Generate coverage report
```

## Pre-commit Hooks

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

Hooks run automatically on `git commit`. To run manually:
```bash
pre-commit run --all-files
```

### Hook Configuration

See `.pre-commit-config.yaml` for full configuration. Key hooks:

- File formatting checks
- Rust formatting and linting
- Security audit
- Spell checking
- YAML/TOML validation

## CI/CD Integration

### GitHub Actions Workflows

1. **code-quality.yml** - Runs on every PR:
   - Format checking
   - Clippy analysis
   - Nextest
   - Documentation linting
   - Spell checking
   - MSRV checking

2. **pr-tests.yml** - Comprehensive PR testing:
   - Multi-platform tests
   - Feature flag combinations
   - Integration tests
   - Code coverage

3. **ci.yml** - Main CI pipeline:
   - Full test matrix
   - Security audit
   - Benchmarks
   - License compliance

### Required Checks

All PRs must pass:
- [ ] rustfmt
- [ ] clippy (no warnings)
- [ ] All tests
- [ ] Documentation builds
- [ ] Security audit
- [ ] License check

## Editor Configuration

### VS Code

Install extensions:
- `rust-analyzer`
- `EditorConfig`
- `markdownlint`
- `typos-spell-checker`

Settings (`.vscode/settings.json`):
```json
{
    "editor.formatOnSave": true,
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.rustfmt.extraArgs": ["--config-path", "rustfmt.toml"],
    "[rust]": {
        "editor.defaultFormatter": "rust-lang.rust-analyzer"
    }
}
```

### IntelliJ IDEA / CLion

1. Install Rust plugin
2. Enable rustfmt on save
3. Configure clippy as external linter
4. Import `.editorconfig`

### Vim/Neovim

```vim
" Install rust.vim
Plug 'rust-lang/rust.vim'

" Format on save
let g:rustfmt_autosave = 1

" Use project rustfmt.toml
let g:rustfmt_options = '--config-path rustfmt.toml'
```

## Best Practices

### Error Handling

- Use `Result<T, E>` for fallible operations
- Create custom error types with `thiserror`
- Provide context with `anyhow` in applications
- Document error conditions

### Testing

- Write unit tests in the same file
- Integration tests in `tests/` directory
- Use property-based testing for complex logic
- Aim for >80% code coverage

### Performance

- Profile before optimizing
- Document performance characteristics
- Benchmark critical paths
- Use `#[inline]` judiciously

### Dependencies

- Minimize external dependencies
- Check licenses with `cargo deny`
- Audit for vulnerabilities regularly
- Document why each dependency is needed

## Enforcement

Code style is enforced at multiple levels:

1. **Local**: Git hooks prevent commits with style violations
2. **PR**: CI checks block merging of non-compliant code
3. **IDE**: Editor integrations provide real-time feedback

To temporarily bypass (not recommended):
```bash
git commit --no-verify
```

## Questions?

For questions about code style:
1. Check existing code for examples
2. Refer to Rust API guidelines
3. Ask in PR reviews
4. Open a discussion issue