//! Workspace-wide hygiene gates (COOLJAPAN policy), executed as a normal test.
//!
//! These checks used to live only in tooling (`cargo deny`, code review, a
//! `find` invocation), which meant nothing failed when the tooling itself was
//! broken. Every assertion below fails against the pre-0.2.1 tree:
//!
//! * [`deny_toml_license_exceptions_are_complete`] — the old `deny.toml` carried
//!   an `[[licenses.exceptions]]` entry whose only content was two commented-out
//!   example lines. cargo-deny requires `crate` + `allow` on every entry, so the
//!   whole file failed to deserialize and *every* gate in it, including the
//!   COOLJAPAN ban list, silently did nothing.
//! * [`deny_toml_uses_the_current_cargo_deny_schema`] — the old file also used
//!   the 0.13-era keys `[bans] deprecated` / `denied`, removed upstream, which is
//!   the second half of that same deserialization failure.
//! * [`deny_toml_bans_the_cooljapan_crate_list`] — the old `[bans] deny` list was
//!   `[]` (one commented-out example), i.e. no crate was banned at all.
//! * [`member_manifests_do_not_duplicate_workspace_versions`] — ~50 dependency
//!   version pins were duplicated in member manifests instead of being taken from
//!   `[workspace.dependencies]`.
//! * [`no_backup_artifacts_in_the_repository`] — `*.bak2`, `*.rs.conflict_backup`
//!   and friends were tracked in git and would have been published by
//!   `cargo package`.
//! * [`no_compiled_binaries_outside_target`] — two ~500 KB Mach-O executables
//!   were tracked under `trustformers-tokenizers/`.
//! * [`virtual_manifest_root_has_no_source_tree`] — an 899-line `src/` tree sat at
//!   the repository root, belonging to no package and compiled by nothing.
//!
//! The crate that hosts the file is incidental: this is the smallest workspace
//! member, so the gate is cheap to run. When the crate is consumed standalone
//! (unpacked from its `.crate` archive) there is no workspace root above it; the
//! tests then report that they are inapplicable instead of failing.

use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use toml::Value;

/// Crates the COOLJAPAN Pure-Rust / oxi* policy forbids, which `[bans] deny` in
/// `deny.toml` must therefore list. Kept deliberately short: these are the names
/// named by the policy itself, not the full transitive `-sys` closure that
/// `deny.toml` additionally covers.
const REQUIRED_BANS: &[&str] = &[
    "openblas-src",
    "blas-src",
    "bincode",
    "rustfft",
    "rusqlite",
    "zip",
    "flate2",
    "zstd",
    "bzip2",
    "lz4",
    "tar",
    "snap",
    "brotli",
    "miniz_oxide",
];

/// `(member, dependency, why)` triples that still carry an inline version pin for
/// a crate `[workspace.dependencies]` also declares.
///
/// Every entry is a known, owned follow-up — not an amnesty. Anything *not*
/// listed here fails [`member_manifests_do_not_duplicate_workspace_versions`].
const KNOWN_INLINE_PINS: &[(&str, &str, &str)] = &[
    (
        "trustformers-wasm",
        "getrandom",
        "wasm needs the `wasm_js` backend feature set; hoisting it is owned by the \
         trustformers-wasm work package",
    ),
    (
        "trustformers-wasm",
        "console_error_panic_hook",
        "same manifest as the `getrandom` pin above; switched together by the \
         trustformers-wasm work package",
    ),
];

/// Workspace members allowed to omit `rust-version.workspace = true`, with the
/// reason. Everything else fails [`member_manifests_inherit_the_workspace_msrv`].
const KNOWN_MISSING_MSRV: &[(&str, &str)] = &[(
    "trustformers-wasm",
    "manifest is owned by the trustformers-wasm work package and is being edited \
     concurrently; the one-line inheritance is added there",
)];

/// File extensions produced by editors, refactoring tools and merge conflicts.
///
/// Matched against the final `.`-separated component of a file name, so
/// `types.rs.bak_refactored` matches on `bak_refactored` while the legitimate
/// `training_script.rs.template` does not match.
const BACKUP_EXTENSIONS: &[&str] = &[
    "old",
    "orig",
    "rej",
    "backup",
    "bk",
    "save",
    "conflict_backup",
    "orig_backup",
    "rej_backup",
    "prelude_fix",
    "policy_fix",
];

/// Directories never walked: build output, VCS metadata, agent scratch space and
/// third-party dependency trees.
const SKIPPED_DIRECTORIES: &[&str] = &[
    "target",
    ".git",
    ".claude",
    ".claude-scratch",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
];

/// Executable-image magic numbers: Mach-O 64/32 (both endiannesses), Mach-O
/// universal ("fat") archives, and ELF.
const EXECUTABLE_MAGICS: &[[u8; 4]] = &[
    [0xcf, 0xfa, 0xed, 0xfe],
    [0xfe, 0xed, 0xfa, 0xcf],
    [0xce, 0xfa, 0xed, 0xfe],
    [0xfe, 0xed, 0xfa, 0xce],
    [0xca, 0xfe, 0xba, 0xbe],
    *b"\x7fELF",
];

/// Nearest ancestor directory holding a `Cargo.toml` with a `[workspace]` table.
///
/// `None` when this crate is being tested outside its workspace (an unpacked
/// `.crate` archive), in which case there is nothing workspace-wide to check.
fn workspace_root() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest_dir.ancestors() {
        let manifest = ancestor.join("Cargo.toml");
        let Ok(text) = fs::read_to_string(&manifest) else {
            continue;
        };
        // `toml::from_str`, not `str::parse::<Value>()`: since toml 1.0 the
        // `FromStr` impl parses a single TOML *value*, so a whole document is
        // rejected with "unexpected content, expected nothing". Getting this
        // wrong turns every test below into a silent no-op.
        let Ok(value) = toml::from_str::<Value>(&text) else {
            continue;
        };
        if value.get("workspace").is_some() {
            return Some(ancestor.to_path_buf());
        }
    }
    None
}

/// Is this crate being tested from an unpacked `.crate` archive?
///
/// `cargo package` writes the pristine manifest next to the rewritten one, so
/// `Cargo.toml.orig` is present in a published tarball and absent in the git
/// checkout. It is the only situation in which a missing workspace root is
/// legitimate.
fn is_packaged_crate() -> bool {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml.orig").is_file()
}

/// Workspace root, or `None` when — and only when — the crate is unpacked from a
/// published archive.
///
/// Inside a checkout a missing root means the *detector* is broken, and a gate
/// that quietly skips is indistinguishable from a gate that passes, so that case
/// fails loudly instead.
fn workspace_root_or_skip(test: &str) -> Option<PathBuf> {
    match workspace_root() {
        Some(root) => Some(root),
        None => {
            assert!(
                is_packaged_crate(),
                "{test}: no ancestor of {} declares [workspace], and this is not an unpacked \
                 .crate (no Cargo.toml.orig). The workspace-root probe is broken, which would \
                 silently turn every hygiene gate in this file into a no-op.",
                env!("CARGO_MANIFEST_DIR"),
            );
            eprintln!(
                "{test}: running from an unpacked .crate archive — there is no workspace above \
                 this crate, so there is nothing workspace-wide to check."
            );
            None
        },
    }
}

/// Parse a TOML file, failing the test with the path on any error.
fn parse_toml(path: &Path) -> Value {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    toml::from_str::<Value>(&text)
        .unwrap_or_else(|error| panic!("{} is not valid TOML: {error}", path.display()))
}

/// `[workspace] members` of the root manifest, as directory names.
fn workspace_members(root: &Path) -> Vec<String> {
    parse_toml(&root.join("Cargo.toml"))
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(Value::as_array)
        .map(|members| members.iter().filter_map(Value::as_str).map(str::to_string).collect())
        .unwrap_or_default()
}

/// Crate names declared in `[workspace.dependencies]`.
fn workspace_dependency_names(root: &Path) -> BTreeSet<String> {
    parse_toml(&root.join("Cargo.toml"))
        .get("workspace")
        .and_then(|workspace| workspace.get("dependencies"))
        .and_then(Value::as_table)
        .map(|table| table.keys().cloned().collect())
        .unwrap_or_default()
}

/// External crate names declared in `[workspace.dependencies]`.
///
/// Entries carrying `path` are intra-workspace crates: they are declared once at
/// the root so the version and the path live in a single place, and the leaves of
/// the graph (`trustformers`, `trustformers-serve`, ...) legitimately have no
/// consumer inside the workspace. Only third-party entries have to be used.
fn external_workspace_dependencies(root: &Path) -> BTreeSet<String> {
    parse_toml(&root.join("Cargo.toml"))
        .get("workspace")
        .and_then(|workspace| workspace.get("dependencies"))
        .and_then(Value::as_table)
        .map(|table| {
            table
                .iter()
                .filter(|(_, spec)| {
                    !spec.as_table().is_some_and(|entry| entry.contains_key("path"))
                })
                .map(|(name, _)| name.clone())
                .collect()
        })
        .unwrap_or_default()
}

/// Every dependency key a member manifest declares, across `[dependencies]`,
/// `[dev-dependencies]`, `[build-dependencies]` and any `[target.'cfg(…)']`
/// variant. Keys match `[workspace.dependencies]` keys even when the root entry
/// renames the crate with `package = "…"`, because inheritance is by key.
fn declared_dependency_keys(manifest: &Value, out: &mut BTreeSet<String>) {
    let Some(table) = manifest.as_table() else {
        return;
    };
    for (key, value) in table {
        match key.as_str() {
            "dependencies" | "dev-dependencies" | "build-dependencies" => {
                if let Some(dependencies) = value.as_table() {
                    out.extend(dependencies.keys().cloned());
                }
            },
            "target" => {
                if let Some(targets) = value.as_table() {
                    for sub in targets.values() {
                        declared_dependency_keys(sub, out);
                    }
                }
            },
            _ => {},
        }
    }
}

/// Does this dependency entry pin a version locally instead of deferring to the
/// workspace?
///
/// `path` dependencies (intra-workspace crates) and `workspace = true` entries
/// are fine; a bare `"1.2.3"` string or a table carrying `version` is not.
fn declares_inline_version(spec: &Value) -> bool {
    match spec {
        Value::String(_) => true,
        Value::Table(table) => {
            !table.contains_key("workspace")
                && !table.contains_key("path")
                && table.contains_key("version")
        },
        _ => false,
    }
}

/// Collect `(dependency_name, table_path)` for every locally pinned dependency of
/// a member manifest, including `[target.'cfg(…)'.dependencies]` blocks.
fn inline_pins(manifest: &Value, section: &str, out: &mut Vec<(String, String)>) {
    let Some(table) = manifest.as_table() else {
        return;
    };
    for (key, value) in table {
        match key.as_str() {
            "dependencies" | "dev-dependencies" | "build-dependencies" => {
                let Some(dependencies) = value.as_table() else {
                    continue;
                };
                for (name, spec) in dependencies {
                    if declares_inline_version(spec) {
                        out.push((name.clone(), format!("{section}[{key}]")));
                    }
                }
            },
            "target" => {
                let Some(targets) = value.as_table() else {
                    continue;
                };
                for (triple, sub) in targets {
                    inline_pins(sub, &format!("{section}[target.{triple}]."), out);
                }
            },
            _ => {},
        }
    }
}

/// Every file under `dir`, skipping [`SKIPPED_DIRECTORIES`].
fn walk_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            if !SKIPPED_DIRECTORIES.contains(&name.as_str()) {
                walk_files(&path, out);
            }
        } else {
            out.push(path);
        }
    }
}

/// Path relative to the workspace root, for readable assertion messages.
fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).unwrap_or(path).to_string_lossy().to_string()
}

/// Every `[[licenses.exceptions]]` entry must carry the fields cargo-deny needs.
///
/// A partial entry does not merely disable the license gate: cargo-deny refuses
/// the whole configuration file, so `cargo deny check bans` cannot run either.
#[test]
fn deny_toml_license_exceptions_are_complete() {
    let Some(root) = workspace_root_or_skip("deny_toml_license_exceptions_are_complete") else {
        return;
    };
    let deny = parse_toml(&root.join("deny.toml"));
    let exceptions = deny
        .get("licenses")
        .and_then(|licenses| licenses.get("exceptions"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let mut broken = Vec::new();
    for (index, exception) in exceptions.iter().enumerate() {
        let has_crate = exception
            .get("crate")
            .and_then(Value::as_str)
            .is_some_and(|name| !name.is_empty());
        let has_allow = exception.get("allow").and_then(Value::as_array).is_some_and(|allowed| {
            !allowed.is_empty() && allowed.iter().all(|license| license.as_str().is_some())
        });
        if !has_crate || !has_allow {
            broken.push(format!(
                "  [[licenses.exceptions]] #{index}: crate={has_crate}, allow={has_allow} ({exception})"
            ));
        }
    }

    assert!(
        broken.is_empty(),
        "deny.toml has {} incomplete [[licenses.exceptions]] entr(ies); cargo-deny refuses to \
         deserialize the file, which disables every gate in it (including the COOLJAPAN ban \
         list):\n{}",
        broken.len(),
        broken.join("\n")
    );
}

/// `deny.toml` must not carry keys removed from cargo-deny before 0.14.
///
/// These keys are what made the original file undeserializable; TOML syntax was
/// never the problem, so a plain parse check would not have caught them.
#[test]
fn deny_toml_uses_the_current_cargo_deny_schema() {
    let Some(root) = workspace_root_or_skip("deny_toml_uses_the_current_cargo_deny_schema") else {
        return;
    };
    let deny = parse_toml(&root.join("deny.toml"));

    let removed_keys: &[(&str, &[&str])] = &[
        ("bans", &["deprecated", "denied"]),
        (
            "licenses",
            &["unlicensed", "default", "allow-osi-fsf-free", "copyleft"],
        ),
        ("advisories", &["vulnerability", "notice", "unsound"]),
    ];

    let mut stale = Vec::new();
    for (section, keys) in removed_keys {
        let Some(table) = deny.get(*section) else {
            continue;
        };
        for key in *keys {
            if table.get(*key).is_some() {
                stale.push(format!("  [{section}] {key}"));
            }
        }
    }

    assert!(
        stale.is_empty(),
        "deny.toml uses {} cargo-deny key(s) that were removed upstream; the file will fail to \
         deserialize and every gate in it becomes a no-op:\n{}",
        stale.len(),
        stale.join("\n")
    );
}

/// `[bans] deny` must list every crate the COOLJAPAN policy forbids.
#[test]
fn deny_toml_bans_the_cooljapan_crate_list() {
    let Some(root) = workspace_root_or_skip("deny_toml_bans_the_cooljapan_crate_list") else {
        return;
    };
    let deny = parse_toml(&root.join("deny.toml"));
    let banned: BTreeSet<String> = deny
        .get("bans")
        .and_then(|bans| bans.get("deny"))
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| match entry {
                    Value::String(name) => Some(name.clone()),
                    other => other
                        .get("crate")
                        .or_else(|| other.get("name"))
                        .and_then(Value::as_str)
                        .map(str::to_string),
                })
                .collect()
        })
        .unwrap_or_default();

    let missing: Vec<&str> = REQUIRED_BANS
        .iter()
        .copied()
        .filter(|required| !banned.contains(*required))
        .collect();

    assert!(
        missing.is_empty(),
        "deny.toml's [bans] deny list is missing {} COOLJAPAN-banned crate(s): {}. \
         `cargo deny check bans` cannot enforce a policy it was never told about.",
        missing.len(),
        missing.join(", ")
    );
}

/// Member manifests must take versions from `[workspace.dependencies]`.
#[test]
fn member_manifests_do_not_duplicate_workspace_versions() {
    let Some(root) = workspace_root_or_skip("member_manifests_do_not_duplicate_workspace_versions")
    else {
        return;
    };
    let workspace_dependencies = workspace_dependency_names(&root);
    let known: BTreeSet<(&str, &str)> = KNOWN_INLINE_PINS
        .iter()
        .map(|(member, dependency, _)| (*member, *dependency))
        .collect();

    let mut offenders = Vec::new();
    for member in workspace_members(&root) {
        let manifest_path = root.join(&member).join("Cargo.toml");
        if !manifest_path.is_file() {
            continue;
        }
        let manifest = parse_toml(&manifest_path);
        let mut pins = Vec::new();
        inline_pins(&manifest, "", &mut pins);
        for (dependency, section) in pins {
            if !workspace_dependencies.contains(&dependency) {
                continue;
            }
            if known.contains(&(member.as_str(), dependency.as_str())) {
                continue;
            }
            offenders.push(format!("  {member}/Cargo.toml {section} {dependency}"));
        }
    }
    offenders.sort();

    assert!(
        offenders.is_empty(),
        "{} member dependenc(ies) pin a version that [workspace.dependencies] already declares. \
         Use `<crate>.workspace = true` (adding only `optional` / extra `features`), or hoist a \
         deliberate exception into KNOWN_INLINE_PINS with the reason:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}

/// No editor/refactoring backup may sit anywhere in the repository.
///
/// `cargo package` copies whatever is next to the live sources, so a leftover
/// `types.rs.bak_refactored` ships to crates.io.
#[test]
fn no_backup_artifacts_in_the_repository() {
    let Some(root) = workspace_root_or_skip("no_backup_artifacts_in_the_repository") else {
        return;
    };
    let mut files = Vec::new();
    walk_files(&root, &mut files);

    let mut offenders: Vec<String> = files
        .iter()
        .filter(|path| {
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            if name.ends_with('~') {
                return true;
            }
            match name.rsplit_once('.') {
                Some((_, extension)) => {
                    extension.starts_with("bak") || BACKUP_EXTENSIONS.contains(&extension)
                },
                None => false,
            }
        })
        .map(|path| relative(&root, path))
        .collect();
    offenders.sort();

    assert!(
        offenders.is_empty(),
        "{} backup artifact(s) are sitting next to live sources and would be published with the \
         crate:\n  {}",
        offenders.len(),
        offenders.join("\n  ")
    );
}

/// No compiled executable may live outside `target/`.
///
/// Two ~500 KB Mach-O test binaries used to be tracked under
/// `trustformers-tokenizers/`; they are not portable, not reproducible, and are
/// copied verbatim into the published `.crate`.
#[test]
fn no_compiled_binaries_outside_target() {
    let Some(root) = workspace_root_or_skip("no_compiled_binaries_outside_target") else {
        return;
    };
    let mut files = Vec::new();
    walk_files(&root, &mut files);

    let mut offenders = Vec::new();
    for path in files {
        // Shared libraries built in place by maturin/`cargo build` are covered by
        // the per-directory .gitignore files; only *tracked-shaped* artifacts are
        // interesting here, so ignore the well-known dynamic-library suffixes.
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        if name.ends_with(".so") || name.ends_with(".dylib") || name.ends_with(".dll") {
            continue;
        }
        let Ok(metadata) = fs::metadata(&path) else {
            continue;
        };
        if metadata.len() < 4096 {
            continue;
        }
        let Ok(mut file) = fs::File::open(&path) else {
            continue;
        };
        let mut magic = [0_u8; 4];
        if file.read_exact(&mut magic).is_err() {
            continue;
        }
        if EXECUTABLE_MAGICS.contains(&magic) {
            offenders.push(relative(&root, &path));
        }
    }
    offenders.sort();

    assert!(
        offenders.is_empty(),
        "{} compiled executable(s) live outside target/:\n  {}",
        offenders.len(),
        offenders.join("\n  ")
    );
}

/// The workspace root is a virtual manifest, so it must not contain `src/`.
///
/// A `src/` tree there is compiled by nothing: no `cargo check`, no clippy, no
/// test run ever looks at it, and it rots silently. The check turns itself off if
/// the root ever gains a real `[package]` section.
#[test]
fn virtual_manifest_root_has_no_source_tree() {
    let Some(root) = workspace_root_or_skip("virtual_manifest_root_has_no_source_tree") else {
        return;
    };
    let manifest = parse_toml(&root.join("Cargo.toml"));
    if manifest.get("package").is_some() {
        return;
    }

    let orphan_src = root.join("src");
    let mut files = Vec::new();
    if orphan_src.is_dir() {
        walk_files(&orphan_src, &mut files);
    }
    let mut listing: Vec<String> = files.iter().map(|path| relative(&root, path)).collect();
    listing.sort();

    assert!(
        listing.is_empty(),
        "the workspace root has no [package] section, so the {} file(s) under its src/ belong to \
         no crate and are compiled by nothing:\n  {}",
        listing.len(),
        listing.join("\n  ")
    );
}

/// Every third-party `[workspace.dependencies]` entry must have a consumer.
///
/// An orphan entry is not inert: `cargo update` keeps resolving it, `cargo deny`
/// audits it, and the next person to need that crate copies a version nobody has
/// ever built against. The pre-0.2.1 root manifest declared `numpy = "0.29"` with
/// no member referencing it — the crate the Python bindings actually use is
/// `scirs2-numpy`, and it lives in the excluded `trustformers-py` manifest.
#[test]
fn workspace_dependency_table_has_no_unused_entries() {
    let Some(root) = workspace_root_or_skip("workspace_dependency_table_has_no_unused_entries")
    else {
        return;
    };

    let mut referenced = BTreeSet::new();
    for member in workspace_members(&root) {
        let manifest_path = root.join(&member).join("Cargo.toml");
        if !manifest_path.is_file() {
            continue;
        }
        declared_dependency_keys(&parse_toml(&manifest_path), &mut referenced);
    }

    let orphans: Vec<String> = external_workspace_dependencies(&root)
        .into_iter()
        .filter(|name| !referenced.contains(name))
        .collect();

    assert!(
        orphans.is_empty(),
        "{} entr(ies) in [workspace.dependencies] are referenced by no workspace member: {}. \
         Delete them, or move the declaration next to the excluded package that actually uses \
         it.",
        orphans.len(),
        orphans.join(", ")
    );
}

/// Every member must inherit the MSRV from `[workspace.package] rust-version`.
///
/// `rust-version` is published metadata: crates.io and `cargo add` use it to warn
/// a consumer before a build fails with a syntax error from a newer edition. When
/// nine of ten manifests omit it, the workspace ships an MSRV promise for one
/// crate and silence for the rest — and the silent ones are the entry points
/// (`trustformers`, `trustformers-serve`) a consumer actually depends on.
#[test]
fn member_manifests_inherit_the_workspace_msrv() {
    let Some(root) = workspace_root_or_skip("member_manifests_inherit_the_workspace_msrv") else {
        return;
    };

    let root_manifest = parse_toml(&root.join("Cargo.toml"));
    let workspace_msrv = root_manifest
        .get("workspace")
        .and_then(|workspace| workspace.get("package"))
        .and_then(|package| package.get("rust-version"))
        .and_then(Value::as_str);
    assert!(
        workspace_msrv.is_some(),
        "[workspace.package] declares no rust-version, so members have nothing to inherit"
    );

    let allowed: BTreeSet<&str> = KNOWN_MISSING_MSRV.iter().map(|(member, _)| *member).collect();

    let mut offenders = Vec::new();
    for member in workspace_members(&root) {
        if allowed.contains(member.as_str()) {
            continue;
        }
        let manifest_path = root.join(&member).join("Cargo.toml");
        if !manifest_path.is_file() {
            continue;
        }
        let inherits = parse_toml(&manifest_path)
            .get("package")
            .and_then(|package| package.get("rust-version"))
            .and_then(Value::as_table)
            .and_then(|entry| entry.get("workspace"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !inherits {
            offenders.push(format!("  {member}/Cargo.toml"));
        }
    }
    offenders.sort();

    assert!(
        offenders.is_empty(),
        "{} member manifest(s) do not declare `rust-version.workspace = true`, so they publish \
         no MSRV at all:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}
