//! Loading real benchmark datasets from disk.
//!
//! Every evaluator in [`crate::evaluation::benchmarks`] reads its examples
//! through this module. There is deliberately no synthetic fallback: a
//! benchmark score is only meaningful if it was produced on that benchmark's
//! data, so a missing dataset is an error, never a template-generated stand-in.
//!
//! Supported on-disk formats:
//!
//! * **TSV** — the layout the GLUE distribution ships (`dev.tsv` and friends),
//!   with or without a header row.
//! * **JSONL** — one JSON object per line, the layout SuperGLUE, HellaSwag and
//!   HumanEval ship.
//! * **CSV** — headerless comma-separated rows, the layout MMLU ships.

use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};

/// One labelled evaluation example.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelledExample {
    /// Model input, already assembled (multi-field inputs are joined with
    /// ` [SEP] `, matching the convention the evaluators use).
    pub input: String,
    /// Gold label as it appears in the dataset.
    pub target: String,
}

/// Where a benchmark's files live and how to read them.
#[derive(Debug, Clone)]
pub struct DatasetSchema<'a> {
    /// Sub-directory of the dataset root holding this task's files, if any.
    pub subdirectory: Option<&'a str>,
    /// File names to try, in order.
    pub file_candidates: &'a [&'a str],
    /// For each input field, the column/property names to try, in order.
    pub input_fields: &'a [&'a [&'a str]],
    /// Column/property names to try for the label, in order.
    pub label_fields: &'a [&'a str],
}

/// Locate a task's data file under `root`.
///
/// Returns a `FileNotFound`-style error listing everything that was tried, so
/// the caller can see exactly what the dataset directory is missing.
pub fn resolve_file(root: &Path, schema: &DatasetSchema<'_>) -> Result<PathBuf> {
    let directory = match schema.subdirectory {
        Some(subdirectory) => root.join(subdirectory),
        None => root.to_path_buf(),
    };

    for candidate in schema.file_candidates {
        let path = directory.join(candidate);
        if path.is_file() {
            return Ok(path);
        }
    }

    Err(anyhow!(
        "no dataset file found in {}: tried {}",
        directory.display(),
        schema.file_candidates.join(", ")
    ))
}

/// Load labelled examples for a task from the dataset root.
pub fn load_examples(
    root: &Path,
    schema: &DatasetSchema<'_>,
    limit: Option<usize>,
) -> Result<Vec<LabelledExample>> {
    let path = resolve_file(root, schema)?;
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let examples = match extension.as_str() {
        "jsonl" | "json" => load_jsonl(&path, schema, limit)?,
        "tsv" => load_delimited(&path, schema, b'\t', limit)?,
        "csv" => load_delimited(&path, schema, b',', limit)?,
        other => {
            return Err(anyhow!(
                "unsupported dataset file extension '{}' for {}",
                other,
                path.display()
            ))
        },
    };

    if examples.is_empty() {
        return Err(anyhow!("{} contained no usable examples", path.display()));
    }

    Ok(examples)
}

/// Split a delimited line, honouring simple double-quoted fields (as used by
/// MMLU's CSV distribution).
fn split_delimited(line: &str, delimiter: u8) -> Vec<String> {
    let delimiter = delimiter as char;
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut characters = line.chars().peekable();

    while let Some(character) = characters.next() {
        match character {
            '"' if in_quotes => {
                if characters.peek() == Some(&'"') {
                    current.push('"');
                    characters.next();
                } else {
                    in_quotes = false;
                }
            },
            '"' if current.is_empty() => in_quotes = true,
            c if c == delimiter && !in_quotes => {
                fields.push(std::mem::take(&mut current));
            },
            c => current.push(c),
        }
    }
    fields.push(current);
    fields
}

/// Read a TSV/CSV file.
///
/// If the first row contains every requested field name it is treated as a
/// header and columns are matched by name; otherwise the file is read
/// positionally using the numeric field names in the schema (for example
/// `"3"` for the fourth column, the layout of GLUE's headerless CoLA files).
pub fn load_delimited(
    path: &Path,
    schema: &DatasetSchema<'_>,
    delimiter: u8,
    limit: Option<usize>,
) -> Result<Vec<LabelledExample>> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;

    let mut lines = contents.lines().filter(|line| !line.trim().is_empty());
    let Some(first_line) = lines.next() else {
        return Err(anyhow!("{} is empty", path.display()));
    };

    let header = split_delimited(first_line, delimiter);
    let header_lookup: Vec<String> =
        header.iter().map(|field| field.trim().to_ascii_lowercase()).collect();

    // Purely numeric candidates are positional indices, not header names; they
    // must never match a header cell (a headerless CoLA row whose label column
    // literally contains "1" would otherwise look like a header).
    let find_named = |candidates: &[&str]| -> Option<usize> {
        candidates
            .iter()
            .filter(|candidate| candidate.parse::<usize>().is_err())
            .find_map(|candidate| {
                let needle = candidate.to_ascii_lowercase();
                header_lookup.iter().position(|field| *field == needle)
            })
    };

    // The first row is a header only if every requested field is named in it.
    let label_index = find_named(schema.label_fields);
    let has_header = label_index.is_some()
        && schema.input_fields.iter().all(|candidates| find_named(candidates).is_some());

    // Resolve every column index up front.
    let (input_indices, label_index) = if has_header {
        let mut input_indices = Vec::new();
        for candidates in schema.input_fields {
            let index = find_named(candidates).ok_or_else(|| {
                anyhow!(
                    "{} has no column matching any of {:?}; header is {:?}",
                    path.display(),
                    candidates,
                    header
                )
            })?;
            input_indices.push(index);
        }
        (
            input_indices,
            label_index.ok_or_else(|| anyhow!("unreachable: label index was checked"))?,
        )
    } else {
        // Positional layout: the schema names must be column indices.
        let parse_positional = |candidates: &[&str], what: &str| -> Result<usize> {
            candidates
                .iter()
                .find_map(|candidate| candidate.parse::<usize>().ok())
                .ok_or_else(|| {
                    anyhow!(
                        "{} has no header row and the schema gives no positional {} column \
                         (candidates were {:?})",
                        path.display(),
                        what,
                        candidates
                    )
                })
        };
        let mut input_indices = Vec::new();
        for candidates in schema.input_fields {
            input_indices.push(parse_positional(candidates, "input")?);
        }
        (
            input_indices,
            parse_positional(schema.label_fields, "label")?,
        )
    };

    let mut examples = Vec::new();
    let rows: Box<dyn Iterator<Item = &str>> = if has_header {
        Box::new(lines)
    } else {
        Box::new(std::iter::once(first_line).chain(lines))
    };

    for row in rows {
        if let Some(limit) = limit {
            if examples.len() >= limit {
                break;
            }
        }

        let fields = split_delimited(row, delimiter);
        let mut inputs = Vec::with_capacity(input_indices.len());
        let mut usable = true;
        for index in &input_indices {
            match fields.get(*index) {
                Some(value) => inputs.push(value.trim().to_string()),
                None => {
                    usable = false;
                    break;
                },
            }
        }
        let Some(label) = fields.get(label_index) else {
            continue;
        };
        if !usable {
            continue;
        }

        examples.push(LabelledExample {
            input: inputs.join(" [SEP] "),
            target: label.trim().to_string(),
        });
    }

    Ok(examples)
}

/// Read a JSONL file, matching fields by property name.
///
/// Array-valued properties (HellaSwag's `endings`, SuperGLUE COPA's choices)
/// are flattened into the input with ` | ` separators so the model sees the
/// full candidate set.
pub fn load_jsonl(
    path: &Path,
    schema: &DatasetSchema<'_>,
    limit: Option<usize>,
) -> Result<Vec<LabelledExample>> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;

    let mut examples = Vec::new();

    for (line_number, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(limit) = limit {
            if examples.len() >= limit {
                break;
            }
        }

        let value: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("{}:{} is not valid JSON", path.display(), line_number + 1))?;

        let mut inputs = Vec::with_capacity(schema.input_fields.len());
        let mut usable = true;
        for candidates in schema.input_fields {
            match candidates.iter().find_map(|candidate| value.get(*candidate)) {
                Some(field) => inputs.push(json_field_to_string(field)),
                None => {
                    usable = false;
                    break;
                },
            }
        }
        if !usable {
            continue;
        }

        let Some(label) = schema.label_fields.iter().find_map(|candidate| value.get(*candidate))
        else {
            continue;
        };

        examples.push(LabelledExample {
            input: inputs.join(" [SEP] "),
            target: json_field_to_string(label),
        });
    }

    Ok(examples)
}

/// Render a JSON value as the string the evaluators compare against.
fn json_field_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Array(items) => {
            items.iter().map(json_field_to_string).collect::<Vec<_>>().join(" | ")
        },
        serde_json::Value::Bool(flag) => flag.to_string(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Read every JSON object of a JSONL file, for callers that need more than an
/// `(input, target)` pair (HumanEval needs the prompt, the tests and the entry
/// point).
pub fn load_jsonl_records(path: &Path, limit: Option<usize>) -> Result<Vec<serde_json::Value>> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;

    let mut records = Vec::new();
    for (line_number, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(limit) = limit {
            if records.len() >= limit {
                break;
            }
        }
        records.push(serde_json::from_str(line).with_context(|| {
            format!("{}:{} is not valid JSON", path.display(), line_number + 1)
        })?);
    }

    if records.is_empty() {
        return Err(anyhow!("{} contained no records", path.display()));
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "trustformers_dataset_{}_{}",
            name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create_dir_all failed");
        path
    }

    #[test]
    fn test_tsv_with_header() {
        let root = temp_dir("tsv_header");
        std::fs::create_dir_all(root.join("sst2")).expect("mkdir failed");
        std::fs::write(
            root.join("sst2/dev.tsv"),
            "sentence\tlabel\nhide new secretions\t0\ncontains no wit\t0\nthat loves its characters\t1\n",
        )
        .expect("write failed");

        let schema = DatasetSchema {
            subdirectory: Some("sst2"),
            file_candidates: &["dev.tsv"],
            input_fields: &[&["sentence"]],
            label_fields: &["label"],
        };

        let examples = load_examples(&root, &schema, None).expect("load failed");
        assert_eq!(examples.len(), 3);
        assert_eq!(examples[0].input, "hide new secretions");
        assert_eq!(examples[0].target, "0");
        assert_eq!(examples[2].target, "1");

        let limited = load_examples(&root, &schema, Some(2)).expect("load failed");
        assert_eq!(limited.len(), 2);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_tsv_without_header_uses_positional_columns() {
        let root = temp_dir("tsv_positional");
        std::fs::create_dir_all(root.join("cola")).expect("mkdir failed");
        // GLUE CoLA layout: source, label, notes, sentence — no header row.
        std::fs::write(
            root.join("cola/dev.tsv"),
            "gj04\t1\t\tOur friends won't buy this analysis.\ngj04\t0\t*\tThey drank the pub.\n",
        )
        .expect("write failed");

        let schema = DatasetSchema {
            subdirectory: Some("cola"),
            file_candidates: &["dev.tsv"],
            input_fields: &[&["sentence", "3"]],
            label_fields: &["label", "1"],
        };

        let examples = load_examples(&root, &schema, None).expect("load failed");
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0].input, "Our friends won't buy this analysis.");
        assert_eq!(examples[0].target, "1");
        assert_eq!(examples[1].target, "0");

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_jsonl_with_multiple_input_fields() {
        let root = temp_dir("jsonl");
        std::fs::write(
            root.join("val.jsonl"),
            "{\"premise\":\"a man eats\",\"hypothesis\":\"a person eats\",\"label\":0}\n\
             {\"premise\":\"a man eats\",\"hypothesis\":\"a person sleeps\",\"label\":2}\n",
        )
        .expect("write failed");

        let schema = DatasetSchema {
            subdirectory: None,
            file_candidates: &["val.jsonl"],
            input_fields: &[&["premise"], &["hypothesis"]],
            label_fields: &["label"],
        };

        let examples = load_examples(&root, &schema, None).expect("load failed");
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0].input, "a man eats [SEP] a person eats");
        assert_eq!(examples[0].target, "0");
        assert_eq!(examples[1].target, "2");

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_csv_positional_for_mmlu_layout() {
        let root = temp_dir("csv");
        std::fs::write(
            root.join("anatomy_test.csv"),
            "\"What is the largest organ?\",Skin,Liver,Heart,Lung,A\n",
        )
        .expect("write failed");

        let schema = DatasetSchema {
            subdirectory: None,
            file_candidates: &["anatomy_test.csv"],
            input_fields: &[&["0"], &["1"], &["2"], &["3"], &["4"]],
            label_fields: &["5"],
        };

        let examples = load_examples(&root, &schema, None).expect("load failed");
        assert_eq!(examples.len(), 1);
        assert!(examples[0].input.starts_with("What is the largest organ?"));
        assert_eq!(examples[0].target, "A");

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_missing_file_is_an_error() {
        let root = temp_dir("missing");
        let schema = DatasetSchema {
            subdirectory: Some("nope"),
            file_candidates: &["dev.tsv", "validation.tsv"],
            input_fields: &[&["sentence"]],
            label_fields: &["label"],
        };

        let error = load_examples(&root, &schema, None).expect_err("must not invent data");
        let message = error.to_string();
        assert!(message.contains("dev.tsv"), "{message}");
        assert!(message.contains("validation.tsv"), "{message}");

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_empty_file_is_an_error() {
        let root = temp_dir("empty");
        std::fs::write(root.join("dev.tsv"), "sentence\tlabel\n").expect("write failed");
        let schema = DatasetSchema {
            subdirectory: None,
            file_candidates: &["dev.tsv"],
            input_fields: &[&["sentence"]],
            label_fields: &["label"],
        };
        assert!(load_examples(&root, &schema, None).is_err());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_quoted_field_splitting() {
        let fields = split_delimited("\"a, b\",c,\"d\"\"e\"", b',');
        assert_eq!(fields, vec!["a, b", "c", "d\"e"]);
    }
}
