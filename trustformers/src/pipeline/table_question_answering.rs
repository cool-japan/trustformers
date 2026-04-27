//! # Table Question Answering Pipeline
//!
//! Answer natural-language questions about structured tabular data (CSV-like tables).
//!
//! ## Supported models
//! - **TAPAS** (google/tapas-base-finetuned-wtq)
//! - **TaPEx** (microsoft/tapex-base)
//!
//! ## Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::table_question_answering::{
//!     Table, TableQaConfig, TableQaPipeline,
//! };
//!
//! let table = Table::from_csv("name,age\nAlice,30\nBob,25").unwrap();
//! let config = TableQaConfig::default();
//! let pipeline = TableQaPipeline::new(config)?;
//! let answer = pipeline.answer("What is the age of Alice?", &table)?;
//! println!("{}", answer.answer);
//! ```

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the table module.
#[derive(Debug, Error)]
pub enum TableError {
    /// CSV or structural parse failure.
    #[error("Parse error: {0}")]
    ParseError(String),
    /// Named column not found.
    #[error("Column not found: {0}")]
    ColumnNotFound(String),
    /// Column contains non-numeric values.
    #[error("Non-numeric values in column: {0}")]
    NonNumeric(String),
    /// Row/column index out of bounds.
    #[error("Index out of bounds")]
    OutOfBounds,
}

/// Errors produced by the table QA pipeline.
#[derive(Debug, Error)]
pub enum TableQaError {
    /// Table has no rows or no columns.
    #[error("Empty table")]
    EmptyTable,
    /// Question string was empty.
    #[error("Empty question")]
    EmptyQuestion,
    /// Underlying model error.
    #[error("Model error: {0}")]
    ModelError(String),
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/// Numeric aggregation operations supported over a table column.
#[derive(Debug, Clone, PartialEq)]
pub enum Aggregation {
    Sum,
    Average,
    Count,
    Min,
    Max,
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

/// A single row in a [`Table`].
#[derive(Debug, Clone)]
pub struct TableRow {
    /// Cell values in column order.
    pub cells: Vec<String>,
}

/// An in-memory table with named columns.
#[derive(Debug, Clone)]
pub struct TableQaTable {
    /// Column header names.
    pub headers: Vec<String>,
    /// Data rows.
    pub rows: Vec<TableRow>,
}

impl TableQaTable {
    /// Construct a table from separate header and data vectors.
    pub fn new(headers: Vec<String>, rows: Vec<Vec<String>>) -> Self {
        let rows = rows
            .into_iter()
            .map(|cells| TableRow { cells })
            .collect();
        Self { headers, rows }
    }

    /// Parse a CSV string: first line = headers, remaining lines = rows.
    pub fn from_csv(csv: &str) -> Result<Self, TableError> {
        let mut lines = csv.lines();
        let header_line = lines.next().ok_or_else(|| {
            TableError::ParseError("CSV is empty".to_string())
        })?;
        let headers: Vec<String> = header_line
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        if headers.is_empty() || headers.iter().all(|h| h.is_empty()) {
            return Err(TableError::ParseError("No headers found".to_string()));
        }

        let num_cols = headers.len();
        let mut rows = Vec::new();
        for (line_no, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let cells: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
            if cells.len() != num_cols {
                return Err(TableError::ParseError(format!(
                    "Row {} has {} cells; expected {}",
                    line_no + 2,
                    cells.len(),
                    num_cols
                )));
            }
            rows.push(TableRow { cells });
        }
        Ok(Self { headers, rows })
    }

    /// Number of data rows.
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Number of columns.
    pub fn num_cols(&self) -> usize {
        self.headers.len()
    }

    /// Access a single cell by zero-based (row, col) indices.
    pub fn cell(&self, row: usize, col: usize) -> Option<&str> {
        self.rows.get(row)?.cells.get(col).map(|s| s.as_str())
    }

    /// Return all values in the named column.
    pub fn column(&self, col_name: &str) -> Option<Vec<&str>> {
        let col_idx = self.headers.iter().position(|h| h == col_name)?;
        Some(
            self.rows
                .iter()
                .map(|r| r.cells[col_idx].as_str())
                .collect(),
        )
    }

    /// Linearise the table into TAPAS-style text.
    ///
    /// Format: `col: h1 | h2 ... row: v1 | v2 ... row: v1 | v2 ...`
    pub fn to_linear_form(&self) -> String {
        let header_part = format!("col: {}", self.headers.join(" | "));
        let row_parts: Vec<String> = self
            .rows
            .iter()
            .map(|r| format!("row: {}", r.cells.join(" | ")))
            .collect();
        std::iter::once(header_part)
            .chain(row_parts)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Return a new table containing only rows for which `predicate(cell)` is true
    /// for the specified column.
    pub fn filter_rows(&self, col_name: &str, predicate: impl Fn(&str) -> bool) -> Self {
        let col_idx = match self.headers.iter().position(|h| h == col_name) {
            Some(idx) => idx,
            None => return Self::new(self.headers.clone(), vec![]),
        };
        let rows: Vec<Vec<String>> = self
            .rows
            .iter()
            .filter(|r| predicate(&r.cells[col_idx]))
            .map(|r| r.cells.clone())
            .collect();
        Self::new(self.headers.clone(), rows)
    }

    /// Apply a numeric aggregation over the named column.
    pub fn aggregate_column(
        &self,
        col_name: &str,
        agg: Aggregation,
    ) -> Result<f64, TableError> {
        let values = self
            .column(col_name)
            .ok_or_else(|| TableError::ColumnNotFound(col_name.to_string()))?;

        if agg == Aggregation::Count {
            return Ok(values.len() as f64);
        }

        let nums: Vec<f64> = values
            .iter()
            .map(|v| {
                v.parse::<f64>().map_err(|_| {
                    TableError::NonNumeric(col_name.to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        if nums.is_empty() {
            return Ok(0.0);
        }

        let result = match agg {
            Aggregation::Sum => nums.iter().sum(),
            Aggregation::Average => nums.iter().sum::<f64>() / nums.len() as f64,
            Aggregation::Count => nums.len() as f64,
            Aggregation::Min => nums
                .iter()
                .cloned()
                .reduce(f64::min)
                .unwrap_or(0.0),
            Aggregation::Max => nums
                .iter()
                .cloned()
                .reduce(f64::max)
                .unwrap_or(0.0),
        };
        Ok(result)
    }
}

/// Public type alias — `Table` maps to [`TableQaTable`] for ergonomics.
pub type Table = TableQaTable;

// ---------------------------------------------------------------------------
// TableQaConfig
// ---------------------------------------------------------------------------

/// Configuration for [`TableQaPipeline`].
#[derive(Debug, Clone)]
pub struct TableQaConfig {
    /// HuggingFace model identifier or local path.
    pub model_name: String,
    /// Maximum number of tokens in the question.
    pub max_question_length: usize,
    /// Maximum number of table cells to encode.
    pub max_table_cells: usize,
    /// Enable numeric aggregation answers.
    pub aggregation: bool,
}

impl Default for TableQaConfig {
    fn default() -> Self {
        Self {
            model_name: "google/tapas-base-finetuned-wtq".to_string(),
            max_question_length: 64,
            max_table_cells: 512,
            aggregation: true,
        }
    }
}

// ---------------------------------------------------------------------------
// TableQaAnswer
// ---------------------------------------------------------------------------

/// Answer to a table question.
#[derive(Debug, Clone)]
pub struct TableQaAnswer {
    /// The answer as a string.
    pub answer: String,
    /// (row, col) coordinates of cells contributing to the answer.
    pub cells: Vec<(usize, usize)>,
    /// Aggregation function applied, if any.
    pub aggregation: Option<Aggregation>,
    /// Confidence score in `[0, 1]`.
    pub confidence: f32,
    /// Alias for `cells` — kept for API compatibility.
    pub coordinates: Vec<(usize, usize)>,
}

// ---------------------------------------------------------------------------
// TableQaPipeline
// ---------------------------------------------------------------------------

/// TAPAS-compatible table question answering pipeline.
pub struct TableQaPipeline {
    config: TableQaConfig,
}

impl TableQaPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: TableQaConfig) -> Result<Self, TableQaError> {
        Ok(Self { config })
    }

    /// Answer a single question about `table`.
    ///
    /// Mock implementation:
    /// 1. Linearise the table.
    /// 2. For numeric questions (sum / average / count / total / how many) find
    ///    a numeric column and apply the corresponding aggregation.
    /// 3. For value questions find the cell that best matches a keyword from the
    ///    question.
    pub fn answer(&self, question: &str, table: &Table) -> Result<TableQaAnswer, TableQaError> {
        if question.trim().is_empty() {
            return Err(TableQaError::EmptyQuestion);
        }
        if table.num_rows() == 0 || table.num_cols() == 0 {
            return Err(TableQaError::EmptyTable);
        }

        let q_lower = question.to_lowercase();

        // Detect numeric question type.
        let numeric_keyword = detect_numeric_keyword(&q_lower);

        if let Some(agg) = numeric_keyword {
            // Find first numeric column.
            if let Some(col_name) = find_numeric_column(table) {
                if let Ok(value) = table.aggregate_column(&col_name, agg.clone()) {
                    let answer = format_numeric(value, &agg);
                    let col_idx = table
                        .headers
                        .iter()
                        .position(|h| h == &col_name)
                        .unwrap_or(0);
                    let coords: Vec<(usize, usize)> = (0..table.num_rows())
                        .map(|r| (r, col_idx))
                        .collect();
                    return Ok(TableQaAnswer {
                        answer,
                        cells: coords.clone(),
                        aggregation: Some(agg),
                        confidence: 0.85,
                        coordinates: coords,
                    });
                }
            }
        }

        // Value-lookup: extract notable words from question and scan cells.
        let keywords = extract_question_keywords(question);
        let (row, col, cell_val) = find_best_matching_cell(table, &keywords);
        let coords = vec![(row, col)];
        Ok(TableQaAnswer {
            answer: cell_val,
            cells: coords.clone(),
            aggregation: None,
            confidence: 0.70,
            coordinates: coords,
        })
    }

    /// Answer multiple questions about the same table.
    pub fn answer_batch(
        &self,
        questions: &[&str],
        table: &Table,
    ) -> Result<Vec<TableQaAnswer>, TableQaError> {
        questions.iter().map(|q| self.answer(q, table)).collect()
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Detect whether the question asks for a numeric aggregation.
fn detect_numeric_keyword(q_lower: &str) -> Option<Aggregation> {
    if q_lower.contains("sum") || q_lower.contains("total") {
        Some(Aggregation::Sum)
    } else if q_lower.contains("average") || q_lower.contains("mean") {
        Some(Aggregation::Average)
    } else if q_lower.contains("how many") || q_lower.contains("count") {
        Some(Aggregation::Count)
    } else if q_lower.contains("minimum") || q_lower.contains("min ") {
        Some(Aggregation::Min)
    } else if q_lower.contains("maximum") || q_lower.contains("max ") {
        Some(Aggregation::Max)
    } else {
        None
    }
}

/// Return the name of the first column whose values are all parseable as `f64`.
fn find_numeric_column(table: &Table) -> Option<String> {
    for header in &table.headers {
        if let Some(vals) = table.column(header) {
            if !vals.is_empty() && vals.iter().all(|v| v.parse::<f64>().is_ok()) {
                return Some(header.clone());
            }
        }
    }
    None
}

/// Format a numeric aggregation result as a string.
fn format_numeric(value: f64, agg: &Aggregation) -> String {
    match agg {
        Aggregation::Count => format!("{}", value as usize),
        _ => {
            // Use integer display if the fractional part is negligible.
            if (value - value.round()).abs() < 1e-9 {
                format!("{}", value as i64)
            } else {
                format!("{:.2}", value)
            }
        }
    }
}

/// Extract simple non-stopword keywords from a question.
fn extract_question_keywords(question: &str) -> Vec<String> {
    let stopwords = [
        "what", "is", "the", "of", "a", "an", "in", "for", "how", "many", "which",
        "where", "when", "who", "does", "do", "are", "was", "were", "has", "have",
    ];
    question
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty() && !stopwords.contains(&w.as_str()))
        .collect()
}

/// Find the cell whose value best matches any of the given keywords.
/// Returns (row, col, value). Falls back to (0, 0, cell(0,0)).
fn find_best_matching_cell(table: &Table, keywords: &[String]) -> (usize, usize, String) {
    for row in 0..table.num_rows() {
        for col in 0..table.num_cols() {
            if let Some(val) = table.cell(row, col) {
                let val_lower = val.to_lowercase();
                let header_lower = table.headers[col].to_lowercase();
                for kw in keywords {
                    if val_lower.contains(kw.as_str())
                        || header_lower.contains(kw.as_str())
                    {
                        return (row, col, val.to_string());
                    }
                }
            }
        }
    }
    // Default: return first cell value.
    let default_val = table.cell(0, 0).unwrap_or("").to_string();
    (0, 0, default_val)
}

// ---------------------------------------------------------------------------
// Extended types
// ---------------------------------------------------------------------------

/// A single cell in a table with its row/column coordinates.
#[derive(Debug, Clone)]
pub struct TableCell {
    pub value: String,
    pub row: usize,
    pub col: usize,
}

/// Aggregation operation for table QA results.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationType {
    Select,
    Count,
    Sum,
    Average,
    Min,
    Max,
}

/// A table QA result with answer, cell coordinates, aggregation type, and score.
#[derive(Debug, Clone)]
pub struct TableQaResult {
    pub answer: String,
    pub coordinates: Vec<(usize, usize)>,
    pub aggregation: AggregationType,
    pub score: f32,
}

/// A wrapper around `TableQaTable` that provides additional parsing utilities.
pub struct TableParser {
    pub table: TableQaTable,
}

impl TableParser {
    /// Parse a CSV string into a `TableQaTable`.
    pub fn from_csv(csv: &str) -> Result<TableQaTable, TableError> {
        TableQaTable::from_csv(csv)
    }

    /// Build a table from headers and row data directly.
    pub fn from_rows(headers: Vec<String>, rows: Vec<Vec<String>>) -> TableQaTable {
        TableQaTable::new(headers, rows)
    }

    /// Access a cell by (row, col) zero-based indices.
    pub fn cell_at(&self, row: usize, col: usize) -> Option<&str> {
        self.table.cell(row, col)
    }

    /// Return all string values in the column at `col_idx`.
    pub fn column_values(&self, col_idx: usize) -> Vec<&str> {
        self.table
            .rows
            .iter()
            .filter_map(|r| r.cells.get(col_idx).map(|s| s.as_str()))
            .collect()
    }

    /// Parse string values in the column at `col_idx` into `f64`.
    /// Non-parseable values are silently skipped.
    pub fn numeric_column_values(&self, col_idx: usize) -> Vec<f64> {
        self.column_values(col_idx)
            .iter()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Standalone aggregation helpers
// ---------------------------------------------------------------------------

/// Sum a slice of f64 values.
pub fn aggregate_sum(values: &[f64]) -> f64 {
    values.iter().sum()
}

/// Average a slice of f64 values. Returns NaN for an empty slice.
pub fn aggregate_average(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Count the number of values.
pub fn aggregate_count(values: &[f64]) -> usize {
    values.len()
}

/// Minimum value. Returns `f64::INFINITY` for an empty slice.
pub fn aggregate_min(values: &[f64]) -> f64 {
    values.iter().cloned().fold(f64::INFINITY, f64::min)
}

/// Maximum value. Returns `f64::NEG_INFINITY` for an empty slice.
pub fn aggregate_max(values: &[f64]) -> f64 {
    values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_table() -> Table {
        Table::from_csv("name,age,score\nAlice,30,95\nBob,25,80\nCarol,35,90").unwrap()
    }

    fn default_pipeline() -> TableQaPipeline {
        TableQaPipeline::new(TableQaConfig::default()).unwrap()
    }

    // 1. from_csv parses headers and rows correctly.
    #[test]
    fn table_from_csv() {
        let t = sample_table();
        assert_eq!(t.headers, vec!["name", "age", "score"]);
        assert_eq!(t.num_rows(), 3);
    }

    // 2. num_rows and num_cols are correct.
    #[test]
    fn table_num_rows_num_cols() {
        let t = sample_table();
        assert_eq!(t.num_rows(), 3);
        assert_eq!(t.num_cols(), 3);
    }

    // 3. cell access returns the correct value.
    #[test]
    fn table_cell_access() {
        let t = sample_table();
        assert_eq!(t.cell(0, 0), Some("Alice"));
        assert_eq!(t.cell(1, 1), Some("25"));
        assert_eq!(t.cell(10, 0), None);
    }

    // 4. column retrieval returns all values.
    #[test]
    fn table_column_retrieval() {
        let t = sample_table();
        let ages = t.column("age").unwrap();
        assert_eq!(ages, vec!["30", "25", "35"]);
    }

    // 5. to_linear_form contains headers and row data.
    #[test]
    fn table_to_linear_form() {
        let t = sample_table();
        let linear = t.to_linear_form();
        assert!(linear.contains("col:"));
        assert!(linear.contains("name"));
        assert!(linear.contains("row:"));
        assert!(linear.contains("Alice"));
    }

    // 6. filter_rows returns only matching rows.
    #[test]
    fn table_filter_rows() {
        let t = sample_table();
        let filtered = t.filter_rows("age", |v| v.parse::<u32>().map(|n| n >= 30).unwrap_or(false));
        assert_eq!(filtered.num_rows(), 2); // Alice(30) and Carol(35)
    }

    // 7. aggregate Sum.
    #[test]
    fn aggregate_sum() {
        let t = sample_table();
        let sum = t.aggregate_column("age", Aggregation::Sum).unwrap();
        assert!((sum - 90.0).abs() < 1e-9);
    }

    // 8. aggregate Average.
    #[test]
    fn aggregate_average() {
        let t = sample_table();
        let avg = t.aggregate_column("age", Aggregation::Average).unwrap();
        assert!((avg - 30.0).abs() < 1e-9);
    }

    // 9. aggregate Count.
    #[test]
    fn aggregate_count() {
        let t = sample_table();
        let count = t.aggregate_column("age", Aggregation::Count).unwrap();
        assert!((count - 3.0).abs() < 1e-9);
    }

    // 10. aggregate Min.
    #[test]
    fn aggregate_min() {
        let t = sample_table();
        let min = t.aggregate_column("age", Aggregation::Min).unwrap();
        assert!((min - 25.0).abs() < 1e-9);
    }

    // 11. aggregate Max.
    #[test]
    fn aggregate_max() {
        let t = sample_table();
        let max = t.aggregate_column("age", Aggregation::Max).unwrap();
        assert!((max - 35.0).abs() < 1e-9);
    }

    // 12. Numeric question (sum) returns an aggregated answer.
    #[test]
    fn answer_numeric_question() {
        let pipe = default_pipeline();
        let t = sample_table();
        let ans = pipe.answer("What is the sum of age?", &t).unwrap();
        assert!(!ans.answer.is_empty());
        assert_eq!(ans.aggregation, Some(Aggregation::Sum));
    }

    // 13. Value question matches a cell.
    #[test]
    fn answer_value_question() {
        let pipe = default_pipeline();
        let t = sample_table();
        let ans = pipe.answer("What is the score of Alice?", &t).unwrap();
        assert!(!ans.answer.is_empty());
        assert!(!ans.coordinates.is_empty());
    }

    // 14. Empty table returns TableQaError::EmptyTable.
    #[test]
    fn answer_empty_table_error() {
        let pipe = default_pipeline();
        let empty = Table::new(vec![], vec![]);
        let result = pipe.answer("Any question?", &empty);
        assert!(matches!(result, Err(TableQaError::EmptyTable)));
    }

    // 15. answer_batch returns one answer per question.
    #[test]
    fn answer_batch_count() {
        let pipe = default_pipeline();
        let t = sample_table();
        let questions = vec!["What is the total age?", "Who has the highest score?"];
        let answers = pipe.answer_batch(&questions, &t).unwrap();
        assert_eq!(answers.len(), 2);
    }

    // -----------------------------------------------------------------------
    // New extended types and functions
    // -----------------------------------------------------------------------

    // 16. TableCell construction and fields
    #[test]
    fn table_cell_construction() {
        let cell = TableCell {
            value: "Alice".to_string(),
            row: 0,
            col: 1,
        };
        assert_eq!(cell.value, "Alice");
        assert_eq!(cell.row, 0);
        assert_eq!(cell.col, 1);
    }

    // 17. Table (alias) via from_rows
    #[test]
    fn table_from_rows() {
        let t = TableParser::from_rows(
            vec!["city".to_string(), "pop".to_string()],
            vec![
                vec!["Paris".to_string(), "2161000".to_string()],
                vec!["Rome".to_string(), "2873000".to_string()],
            ],
        );
        assert_eq!(t.num_rows(), 2);
        assert_eq!(t.num_cols(), 2);
        assert_eq!(t.cell(1, 0), Some("Rome"));
    }

    // 18. TableParser::from_csv round-trip
    #[test]
    fn table_parser_from_csv() {
        let t = TableParser::from_csv("x,y,z\n1,2,3\n4,5,6").expect("parse ok");
        assert_eq!(t.headers, vec!["x", "y", "z"]);
        assert_eq!(t.num_rows(), 2);
    }

    // 19. TableParser::from_csv empty string returns error
    #[test]
    fn table_parser_from_csv_empty_error() {
        let result = TableParser::from_csv("");
        assert!(result.is_err());
    }

    // 20. TableParser::from_csv mismatched column count returns error
    #[test]
    fn table_parser_from_csv_mismatched_cols() {
        let result = TableParser::from_csv("a,b\n1,2,3");
        assert!(result.is_err());
    }

    // 21. TableParser::cell_at with valid index
    #[test]
    fn table_parser_cell_at_valid() {
        let t = TableParser::from_csv("a,b\n10,20\n30,40").expect("ok");
        let parser = TableParser { table: t };
        assert_eq!(parser.cell_at(0, 0), Some("10"));
        assert_eq!(parser.cell_at(1, 1), Some("40"));
    }

    // 22. TableParser::cell_at out of bounds returns None
    #[test]
    fn table_parser_cell_at_oob() {
        let t = TableParser::from_csv("a,b\n1,2").expect("ok");
        let parser = TableParser { table: t };
        assert_eq!(parser.cell_at(10, 0), None);
    }

    // 23. TableParser::column_values
    #[test]
    fn table_parser_column_values() {
        let t = TableParser::from_csv("name,score\nAlice,95\nBob,80").expect("ok");
        let parser = TableParser { table: t };
        let vals = parser.column_values(1);
        assert_eq!(vals, vec!["95", "80"]);
    }

    // 24. TableParser::numeric_column_values
    #[test]
    fn table_parser_numeric_column_values() {
        let t = TableParser::from_csv("name,val\nA,1.5\nB,2.5\nC,3.0").expect("ok");
        let parser = TableParser { table: t };
        let nums = parser.numeric_column_values(1);
        assert_eq!(nums.len(), 3);
        assert!((nums[0] - 1.5).abs() < 1e-9);
        assert!((nums[1] - 2.5).abs() < 1e-9);
    }

    // 25. aggregate_sum
    #[test]
    fn standalone_aggregate_sum() {
        let vals = vec![1.0_f64, 2.0, 3.0, 4.0];
        assert!((aggregate_sum(&vals) - 10.0).abs() < 1e-9);
    }

    // 26. aggregate_average
    #[test]
    fn standalone_aggregate_average() {
        let vals = vec![2.0_f64, 4.0, 6.0];
        assert!((aggregate_average(&vals) - 4.0).abs() < 1e-9);
    }

    // 27. aggregate_average empty slice
    #[test]
    fn standalone_aggregate_average_empty() {
        assert!(aggregate_average(&[]).is_nan());
    }

    // 28. aggregate_count
    #[test]
    fn standalone_aggregate_count() {
        let vals = vec![1.0_f64; 7];
        assert_eq!(aggregate_count(&vals), 7);
    }

    // 29. aggregate_min
    #[test]
    fn standalone_aggregate_min() {
        let vals = vec![3.0_f64, 1.0, 4.0, 1.0, 5.0];
        assert!((aggregate_min(&vals) - 1.0).abs() < 1e-9);
    }

    // 30. aggregate_max
    #[test]
    fn standalone_aggregate_max() {
        let vals = vec![3.0_f64, 1.0, 9.0, 2.0];
        assert!((aggregate_max(&vals) - 9.0).abs() < 1e-9);
    }

    // 31. AggregationType variants all exist
    #[test]
    fn aggregation_type_variants() {
        let _ = AggregationType::Select;
        let _ = AggregationType::Count;
        let _ = AggregationType::Sum;
        let _ = AggregationType::Average;
        let _ = AggregationType::Min;
        let _ = AggregationType::Max;
    }

    // 32. TableQaResult construction
    #[test]
    fn table_qa_result_construction() {
        let r = TableQaResult {
            answer: "42".to_string(),
            coordinates: vec![(1, 2)],
            aggregation: AggregationType::Sum,
            score: 0.9,
        };
        assert_eq!(r.answer, "42");
        assert_eq!(r.score, 0.9);
        assert!(matches!(r.aggregation, AggregationType::Sum));
    }

    // 33. TableQaPipeline::answer with average question
    #[test]
    fn answer_average_question() {
        let pipe = default_pipeline();
        let t = sample_table();
        let ans = pipe.answer("What is the average score?", &t).unwrap();
        assert!(ans.aggregation == Some(Aggregation::Average));
    }

    // 34. TableQaPipeline::answer with min question
    #[test]
    fn answer_min_question() {
        let pipe = default_pipeline();
        let t = sample_table();
        let ans = pipe.answer("What is the minimum age?", &t).unwrap();
        assert!(ans.aggregation == Some(Aggregation::Min));
    }

    // 35. TableQaPipeline::answer with max question
    #[test]
    fn answer_max_question() {
        let pipe = default_pipeline();
        let t = sample_table();
        let ans = pipe.answer("What is the maximum score?", &t).unwrap();
        assert!(ans.aggregation == Some(Aggregation::Max));
    }
}
