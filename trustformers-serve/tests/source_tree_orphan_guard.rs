//! Guard against orphan source files in `trustformers-serve/src`.
//!
//! An *orphan* is a `.rs` file that no `mod` declaration reaches from
//! `src/lib.rs` (or from a `src/bin/*.rs` binary root). Rust never compiles such
//! a file, so `cargo check`, `cargo clippy` and `cargo nextest` are all silent
//! about it: it can call APIs that no longer exist, assert behaviour that was
//! deleted, or — worst of all — implement a feature that reads as supported
//! while being unreachable from every construction path.
//!
//! Version 0.2.1 removed several such clusters from this crate (see the module
//! headers of [`trustformers_serve::auth`],
//! [`trustformers_serve::encryption`],
//! [`trustformers_serve::performance_optimizer`] and
//! [`trustformers_serve::resource_management`] for what went and why) and
//! mounted the ones worth keeping. This test exists so the tree cannot drift
//! back: it re-derives the module graph from the source itself and fails with
//! the offending paths listed.
//!
//! Against the pre-0.2.1 tree this test failed with 38 orphan files totalling
//! roughly 12,500 lines.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Read a file, returning an empty string when it cannot be read.
fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_default()
}

/// Extract the quoted value of a `#[path = "…"]` attribute line.
fn path_attribute_value(line: &str) -> Option<String> {
    let rest = line.trim().strip_prefix("#[path")?;
    let start = rest.find('"')? + 1;
    let end = rest[start..].find('"')? + start;
    Some(rest[start..end].to_string())
}

/// Extract the module name from a `mod name;` declaration line.
///
/// Returns `None` for `mod name { … }` blocks, comments, and anything that is
/// not a file-backed module declaration.
fn module_declaration_name(line: &str) -> Option<String> {
    // Trailing line comments are common on module declarations
    // (`pub mod threshold; // re-enabled`), so strip them before parsing.
    let without_comment = match line.split("//").next() {
        Some(head) => head,
        None => line,
    };
    let mut rest = without_comment.trim();
    if rest.is_empty() || rest.contains('{') {
        return None;
    }
    for visibility in ["pub(crate) ", "pub(super) ", "pub(in crate) ", "pub "] {
        if let Some(stripped) = rest.strip_prefix(visibility) {
            rest = stripped.trim_start();
            break;
        }
    }
    let rest = rest.strip_prefix("mod ")?;
    let name = rest.strip_suffix(';')?.trim();
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    Some(name.to_string())
}

/// One file-backed child module declared by a source file.
struct ChildModule {
    /// Module name as written in the `mod` declaration.
    name: String,
    /// Explicit `#[path = "…"]` override, when the declaration carried one.
    explicit_path: Option<String>,
}

/// Collect the file-backed child modules declared by `source`.
fn declared_children(source: &str) -> Vec<ChildModule> {
    let mut children = Vec::new();
    let mut pending_path: Option<String> = None;
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if let Some(value) = path_attribute_value(trimmed) {
            pending_path = Some(value);
            continue;
        }
        if let Some(name) = module_declaration_name(trimmed) {
            children.push(ChildModule {
                name,
                explicit_path: pending_path.take(),
            });
            continue;
        }
        if !trimmed.starts_with("#[") {
            pending_path = None;
        }
    }
    children
}

/// Walk the module graph from `file`, marking every reachable source file.
///
/// `module_dir` is the directory a plain `mod name;` inside `file` resolves
/// against: `src` for `src/lib.rs`, `src/foo` for both `src/foo.rs` and
/// `src/foo/mod.rs`.
fn visit(file: &Path, module_dir: &Path, reached: &mut HashSet<PathBuf>) {
    let Ok(canonical) = file.canonicalize() else {
        return;
    };
    if !reached.insert(canonical.clone()) {
        return;
    }
    let source = read(&canonical);
    let parent = canonical.parent().map(Path::to_path_buf).unwrap_or_default();
    for child in declared_children(&source) {
        if let Some(explicit) = child.explicit_path {
            visit(&parent.join(explicit), module_dir, reached);
            continue;
        }
        let as_file = module_dir.join(format!("{}.rs", child.name));
        let as_directory = module_dir.join(&child.name).join("mod.rs");
        if as_file.exists() {
            visit(&as_file, &module_dir.join(&child.name), reached);
        }
        if as_directory.exists() {
            visit(&as_directory, &module_dir.join(&child.name), reached);
        }
    }
}

/// Every `.rs` file under `dir`, recursively.
fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            if let Ok(canonical) = path.canonicalize() {
                out.push(canonical);
            }
        }
    }
}

/// The crate's `src` directory.
fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Files reachable from `src/lib.rs` and every `src/bin/*.rs` root.
fn reachable_sources() -> HashSet<PathBuf> {
    let src = src_dir();
    let mut reached = HashSet::new();
    visit(&src.join("lib.rs"), &src, &mut reached);

    let bin_dir = src.join("bin");
    if let Ok(entries) = fs::read_dir(&bin_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "rs") {
                let stem = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
                visit(&path, &bin_dir.join(stem), &mut reached);
            }
        }
    }
    reached
}

/// Every `.rs` file under `src/` must be reachable from a crate root.
#[test]
fn no_orphan_source_files_under_src() {
    let src = src_dir();
    assert!(
        src.is_dir(),
        "src directory must exist at {}",
        src.display()
    );

    let reached = reachable_sources();
    let mut all = Vec::new();
    rust_sources(&src, &mut all);

    let mut orphans: Vec<String> = all
        .into_iter()
        .filter(|path| !reached.contains(path))
        .map(|path| path.strip_prefix(&src).unwrap_or(&path).to_string_lossy().to_string())
        .collect();
    orphans.sort();

    assert!(
        orphans.is_empty(),
        "{} source file(s) under src/ are declared by no `mod` and are therefore \
         never compiled. Either declare them (and let their tests run) or delete \
         them — shipping unreachable code in a published crate is not an option:\n  {}",
        orphans.len(),
        orphans.join("\n  ")
    );
}

/// No editor/refactoring backups may sit under `src/`.
///
/// `cargo package` copies everything under `src/`, so a `types.rs.backup_v2`
/// left behind by a refactor is published to crates.io as part of the crate.
#[test]
fn no_backup_artifacts_under_src() {
    let src = src_dir();
    let mut suspicious = Vec::new();
    let mut stack = vec![src.clone()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            let is_backup = name.contains(".rs.")
                || name.ends_with(".bak")
                || name.ends_with(".orig")
                || name.ends_with(".rej")
                || name.ends_with('~');
            if is_backup {
                suspicious
                    .push(path.strip_prefix(&src).unwrap_or(&path).to_string_lossy().to_string());
            }
        }
    }
    suspicious.sort();
    assert!(
        suspicious.is_empty(),
        "backup artifact(s) under src/ would be published with the crate:\n  {}",
        suspicious.join("\n  ")
    );
}
