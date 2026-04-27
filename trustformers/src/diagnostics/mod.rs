//! Rich error diagnostics with troubleshooting suggestions.
//!
//! Wraps TrustformersError with contextual information and actionable suggestions,
//! providing structured diagnostic output and aggregation for inference sessions.

use crate::error::TrustformersError;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Severity level for a diagnostic context.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DiagnosticSeverity {
    /// Informational message, not an error.
    Info,
    /// Warning — execution continues but something is sub-optimal.
    Warning,
    /// Error — operation failed but the process can continue.
    Error,
    /// Fatal — unrecoverable; further execution is unsafe.
    Fatal,
}

impl fmt::Display for DiagnosticSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
            Self::Fatal => write!(f, "FATAL"),
        }
    }
}

// ---------------------------------------------------------------------------
// DiagnosticContext
// ---------------------------------------------------------------------------

/// A diagnostic context attached to an error providing rich troubleshooting information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticContext {
    /// Human-readable explanation of what went wrong.
    pub what: String,
    /// Why this typically happens.
    pub why: Vec<String>,
    /// Actionable steps to resolve.
    pub how_to_fix: Vec<String>,
    /// Related documentation links (relative paths or URLs).
    pub docs: Vec<String>,
    /// Severity level.
    pub severity: DiagnosticSeverity,
    /// Error code for programmatic handling.
    pub code: &'static str,
}

impl DiagnosticContext {
    /// Create a new diagnostic context.
    pub fn new(what: impl Into<String>, severity: DiagnosticSeverity, code: &'static str) -> Self {
        Self {
            what: what.into(),
            why: Vec::new(),
            how_to_fix: Vec::new(),
            docs: Vec::new(),
            severity,
            code,
        }
    }

    /// Add a "why" entry.
    pub fn with_why(mut self, reason: impl Into<String>) -> Self {
        self.why.push(reason.into());
        self
    }

    /// Add a "how to fix" entry.
    pub fn with_fix(mut self, step: impl Into<String>) -> Self {
        self.how_to_fix.push(step.into());
        self
    }

    /// Add a documentation reference.
    pub fn with_doc(mut self, doc: impl Into<String>) -> Self {
        self.docs.push(doc.into());
        self
    }
}

impl fmt::Display for DiagnosticContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} (code: {})", self.severity, self.what, self.code)
    }
}

// ---------------------------------------------------------------------------
// Diagnosable trait
// ---------------------------------------------------------------------------

/// Trait for errors that can provide diagnostic context.
pub trait Diagnosable {
    /// Return a diagnostic context for this error, if available.
    fn diagnostic(&self) -> Option<DiagnosticContext>;

    /// Return a short one-line suggestion for fixing this error.
    fn suggestion(&self) -> Option<String> {
        self.diagnostic().and_then(|d| d.how_to_fix.into_iter().next())
    }
}

// ---------------------------------------------------------------------------
// ErrorSpan
// ---------------------------------------------------------------------------

/// Location and model context for an error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSpan {
    /// Source file where the error originated.
    pub file: String,
    /// Line number in the source file.
    pub line: u32,
    /// Optional model name in scope when the error occurred.
    pub model_name: Option<String>,
    /// Optional layer name in scope when the error occurred.
    pub layer_name: Option<String>,
}

impl ErrorSpan {
    /// Create a minimal span from file and line.
    pub fn new(file: impl Into<String>, line: u32) -> Self {
        Self {
            file: file.into(),
            line,
            model_name: None,
            layer_name: None,
        }
    }

    /// Attach a model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model_name = Some(model.into());
        self
    }

    /// Attach a layer name.
    pub fn with_layer(mut self, layer: impl Into<String>) -> Self {
        self.layer_name = Some(layer.into());
        self
    }
}

impl fmt::Display for ErrorSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.line)?;
        if let Some(m) = &self.model_name {
            write!(f, " (model: {m})")?;
        }
        if let Some(l) = &self.layer_name {
            write!(f, " [layer: {l}]")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RichError
// ---------------------------------------------------------------------------

/// Rich error wrapper combining a source error, diagnostic context, and optional span.
#[derive(Debug)]
pub struct RichError {
    /// The underlying error.
    pub inner: Box<dyn std::error::Error + Send + Sync>,
    /// Structured diagnostic context.
    pub context: DiagnosticContext,
    /// Optional source location / model context.
    pub span: Option<ErrorSpan>,
}

impl RichError {
    /// Wrap an error with a diagnostic context.
    pub fn new(
        error: impl std::error::Error + Send + Sync + 'static,
        context: DiagnosticContext,
    ) -> Self {
        Self {
            inner: Box::new(error),
            context,
            span: None,
        }
    }

    /// Attach source location information.
    pub fn with_span(mut self, span: ErrorSpan) -> Self {
        self.span = Some(span);
        self
    }

    /// Format as a rich multi-line message with all suggestions.
    pub fn display_rich(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!(
            "[{}] {} (code: {})\n",
            self.context.severity, self.context.what, self.context.code
        ));

        if let Some(ref span) = self.span {
            out.push_str(&format!("  at {span}\n"));
        }

        out.push_str(&format!("  caused by: {}\n", self.inner));

        if !self.context.why.is_empty() {
            out.push_str("  possible causes:\n");
            for why in &self.context.why {
                out.push_str(&format!("    - {why}\n"));
            }
        }

        if !self.context.how_to_fix.is_empty() {
            out.push_str("  how to fix:\n");
            for fix in &self.context.how_to_fix {
                out.push_str(&format!("    - {fix}\n"));
            }
        }

        if !self.context.docs.is_empty() {
            out.push_str("  see also:\n");
            for doc in &self.context.docs {
                out.push_str(&format!("    - {doc}\n"));
            }
        }

        out
    }

    /// Format as a compact single-line plain-text message suitable for logs.
    pub fn display_plain(&self) -> String {
        let span_str = self.span.as_ref().map(|s| format!(" at {s}")).unwrap_or_default();
        format!(
            "[{}]{} {} — {}",
            self.context.severity, span_str, self.context.what, self.inner
        )
    }
}

impl fmt::Display for RichError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_rich())
    }
}

impl std::error::Error for RichError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.inner.as_ref())
    }
}

// ---------------------------------------------------------------------------
// CommonDiagnostics
// ---------------------------------------------------------------------------

/// Pre-built diagnostic contexts for common TrustformeRS error scenarios.
pub struct CommonDiagnostics;

impl CommonDiagnostics {
    /// Diagnostic for a model that could not be found locally or remotely.
    pub fn model_not_found(model_name: &str) -> DiagnosticContext {
        DiagnosticContext::new(
            format!("Model '{model_name}' was not found"),
            DiagnosticSeverity::Error,
            "E001",
        )
        .with_why("The model ID may be misspelled")
        .with_why("The model files are not present in the local cache")
        .with_why("Network access to the model hub is unavailable")
        .with_fix(format!("Verify the model identifier: '{model_name}'"))
        .with_fix("Run `trust-hub download <model_id>` to pre-download the model")
        .with_fix("Set TRUSTFORMERS_OFFLINE=1 and point to a local directory")
        .with_doc("https://docs.trustformers.dev/models/loading")
    }

    /// Diagnostic for failed weight loading.
    pub fn weight_loading_failed(model_name: &str, format: &str) -> DiagnosticContext {
        DiagnosticContext::new(
            format!("Failed to load weights for '{model_name}' in {format} format"),
            DiagnosticSeverity::Error,
            "E002",
        )
        .with_why("The weight file may be corrupted or truncated")
        .with_why(format!(
            "The {format} format is not supported by the current build"
        ))
        .with_why("Insufficient disk space or memory to load the model")
        .with_fix("Re-download the model weights to ensure file integrity")
        .with_fix("Check available disk space and memory")
        .with_fix(format!(
            "Try converting weights to safetensors format from {format}"
        ))
        .with_doc("https://docs.trustformers.dev/models/formats")
    }

    /// Diagnostic for a tokenizer that doesn't match the model architecture.
    pub fn tokenizer_mismatch(model_type: &str, tokenizer_type: &str) -> DiagnosticContext {
        DiagnosticContext::new(
            format!("Tokenizer type '{tokenizer_type}' does not match model type '{model_type}'"),
            DiagnosticSeverity::Error,
            "E003",
        )
        .with_why("The tokenizer was initialized for a different model family")
        .with_why("The tokenizer config file specifies a mismatched tokenizer_class")
        .with_fix(format!(
            "Use AutoTokenizer::from_pretrained('{model_type}') to get the correct tokenizer"
        ))
        .with_fix("Ensure tokenizer_config.json in the model directory is correct")
        .with_doc("https://docs.trustformers.dev/tokenizers/auto")
    }

    /// Diagnostic for CUDA out-of-memory errors.
    pub fn cuda_out_of_memory(required_mb: u64, available_mb: u64) -> DiagnosticContext {
        DiagnosticContext::new(
            format!("CUDA out of memory: need {required_mb} MB, only {available_mb} MB available"),
            DiagnosticSeverity::Fatal,
            "E004",
        )
        .with_why("The model is too large to fit in GPU memory at this batch size")
        .with_why("Other CUDA processes are consuming available memory")
        .with_fix("Reduce the batch size (e.g., set batch_size = 1)")
        .with_fix("Enable mixed-precision inference (fp16 or bf16)")
        .with_fix("Use model parallelism across multiple GPUs")
        .with_fix("Run `nvidia-smi` to inspect other CUDA consumers and free memory")
        .with_doc("https://docs.trustformers.dev/deployment/gpu-memory")
    }

    /// Diagnostic for an unsupported precision type on the target hardware.
    pub fn unsupported_precision(requested: &str, hardware: &str) -> DiagnosticContext {
        DiagnosticContext::new(
            format!("Precision '{requested}' is not supported on hardware '{hardware}'"),
            DiagnosticSeverity::Warning,
            "W001",
        )
        .with_why(format!(
            "{hardware} does not have native {requested} compute units"
        ))
        .with_why("The required kernel has not been compiled for this precision")
        .with_fix(format!("Fall back to f32 precision on {hardware}"))
        .with_fix("Use a device that supports the requested precision (e.g., NVIDIA A100 for bf16)")
        .with_doc("https://docs.trustformers.dev/precision/hardware-matrix")
    }

    /// Diagnostic for a failed hub download.
    pub fn hub_download_failed(model_id: &str, reason: &str) -> DiagnosticContext {
        DiagnosticContext::new(
            format!("Hub download failed for model '{model_id}': {reason}"),
            DiagnosticSeverity::Error,
            "E005",
        )
        .with_why("Network connectivity issue or DNS resolution failure")
        .with_why("The model repository requires authentication (private model)")
        .with_why("The hub endpoint is temporarily unavailable")
        .with_fix("Check your network connection and try again")
        .with_fix("Set HUGGINGFACE_HUB_TOKEN for private models")
        .with_fix("Use `--offline` mode with a locally cached copy")
        .with_doc("https://docs.trustformers.dev/hub/authentication")
    }

    /// Diagnostic for a pipeline that received an incompatible model.
    pub fn pipeline_type_mismatch(pipeline: &str, model: &str) -> DiagnosticContext {
        DiagnosticContext::new(
            format!("Pipeline '{pipeline}' is not compatible with model '{model}'"),
            DiagnosticSeverity::Error,
            "E006",
        )
        .with_why("The model was not fine-tuned for the requested task")
        .with_why("The model architecture does not expose the required output head")
        .with_fix(format!("Use a model with a '{pipeline}' head, e.g., AutoModelFor{pipeline}::from_pretrained(...)"))
        .with_fix("Check the model card to verify supported tasks")
        .with_doc("https://docs.trustformers.dev/pipelines/compatibility")
    }

    /// Diagnostic for sequences that exceed the model's maximum length.
    pub fn sequence_too_long(length: usize, max_length: usize) -> DiagnosticContext {
        DiagnosticContext::new(
            format!("Input sequence length {length} exceeds model maximum {max_length}"),
            DiagnosticSeverity::Warning,
            "W002",
        )
        .with_why("The tokenized input is longer than the model's positional embedding capacity")
        .with_fix(format!("Truncate input to {max_length} tokens"))
        .with_fix("Use sliding window / chunked inference for long documents")
        .with_fix("Use a model with a larger context window (e.g., Longformer, BigBird)")
        .with_doc("https://docs.trustformers.dev/models/long-context")
    }
}

// ---------------------------------------------------------------------------
// DiagnosticReport
// ---------------------------------------------------------------------------

/// Aggregates diagnostics emitted during a single training or inference session.
#[derive(Debug, Default)]
pub struct DiagnosticReport {
    issues: Vec<(String, DiagnosticContext)>,
}

impl DiagnosticReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an issue attributed to a named component.
    pub fn add(&mut self, component: &str, context: DiagnosticContext) {
        self.issues.push((component.to_string(), context));
    }

    /// Returns `true` if any issue has severity `Error` or `Fatal`.
    pub fn has_errors(&self) -> bool {
        self.issues.iter().any(|(_, ctx)| {
            ctx.severity == DiagnosticSeverity::Error || ctx.severity == DiagnosticSeverity::Fatal
        })
    }

    /// Returns `true` if any issue has severity `Warning`.
    pub fn has_warnings(&self) -> bool {
        self.issues.iter().any(|(_, ctx)| ctx.severity == DiagnosticSeverity::Warning)
    }

    /// Filter issues by exact severity.
    pub fn issues_by_severity(&self, severity: &DiagnosticSeverity) -> Vec<&DiagnosticContext> {
        self.issues
            .iter()
            .filter(|(_, ctx)| &ctx.severity == severity)
            .map(|(_, ctx)| ctx)
            .collect()
    }

    /// Render the full report as a human-readable string.
    pub fn to_string_report(&self) -> String {
        if self.issues.is_empty() {
            return "No issues found.".to_string();
        }

        let mut out = format!(
            "Diagnostic Report — {} issue(s)\n{}\n",
            self.issues.len(),
            "=".repeat(60)
        );

        for (component, ctx) in &self.issues {
            out.push_str(&format!(
                "\n[{severity}] <{component}> {what} (code: {code})\n",
                severity = ctx.severity,
                what = ctx.what,
                code = ctx.code,
            ));
            if !ctx.why.is_empty() {
                out.push_str("  Causes:\n");
                for w in &ctx.why {
                    out.push_str(&format!("    - {w}\n"));
                }
            }
            if !ctx.how_to_fix.is_empty() {
                out.push_str("  Fixes:\n");
                for fix in &ctx.how_to_fix {
                    out.push_str(&format!("    - {fix}\n"));
                }
            }
        }

        out
    }

    /// Serialize the report to JSON.
    pub fn to_json(&self) -> Result<String, TrustformersError> {
        #[derive(Serialize)]
        struct ReportEntry<'a> {
            component: &'a str,
            code: &'a str,
            severity: &'a DiagnosticSeverity,
            what: &'a str,
            why: &'a [String],
            how_to_fix: &'a [String],
            docs: &'a [String],
        }

        let entries: Vec<ReportEntry<'_>> = self
            .issues
            .iter()
            .map(|(comp, ctx)| ReportEntry {
                component: comp,
                code: ctx.code,
                severity: &ctx.severity,
                what: &ctx.what,
                why: &ctx.why,
                how_to_fix: &ctx.how_to_fix,
                docs: &ctx.docs,
            })
            .collect();

        serde_json::to_string_pretty(&entries).map_err(|e| {
            TrustformersError::pipeline(
                format!("Failed to serialize diagnostic report: {e}"),
                "diagnostics",
            )
        })
    }

    /// Number of issues in the report.
    pub fn len(&self) -> usize {
        self.issues.len()
    }

    /// Returns `true` when there are no issues.
    pub fn is_empty(&self) -> bool {
        self.issues.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Model numerical diagnostics
// ---------------------------------------------------------------------------

/// Pass / Warning / Fail / Info status for a single diagnostic check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagStatus {
    /// The check passed within the expected range.
    Pass,
    /// A potential issue was detected but execution can continue.
    Warning,
    /// The check failed and the issue should be addressed.
    Fail,
    /// Informational result — not a pass/fail verdict.
    Info,
}

impl fmt::Display for DiagStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagStatus::Pass => write!(f, "PASS"),
            DiagStatus::Warning => write!(f, "WARN"),
            DiagStatus::Fail => write!(f, "FAIL"),
            DiagStatus::Info => write!(f, "INFO"),
        }
    }
}

/// A single numerical diagnostic check result.
#[derive(Debug, Clone)]
pub struct DiagnosticResult {
    /// Name of the layer or component being tested.
    pub layer_name: String,
    /// Metric being measured (e.g. "weight_norm", "grad_norm").
    pub metric_name: String,
    /// Computed value.
    pub value: f32,
    /// Optional threshold used to determine pass/fail.
    pub threshold: Option<f32>,
    /// Pass / Warning / Fail / Info status.
    pub status: DiagStatus,
}

impl DiagnosticResult {
    fn new(
        layer_name: impl Into<String>,
        metric_name: impl Into<String>,
        value: f32,
        threshold: Option<f32>,
        status: DiagStatus,
    ) -> Self {
        Self {
            layer_name: layer_name.into(),
            metric_name: metric_name.into(),
            value,
            threshold,
            status,
        }
    }
}

impl fmt::Display for DiagnosticResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{status}] {layer}/{metric} = {value:.6}",
            status = self.status,
            layer = self.layer_name,
            metric = self.metric_name,
            value = self.value,
        )?;
        if let Some(t) = self.threshold {
            write!(f, " (threshold: {t:.6})")?;
        }
        Ok(())
    }
}

/// Summary of a collection of [`DiagnosticResult`]s.
#[derive(Debug, Clone)]
pub struct DiagnosticSummary {
    /// Total number of checks.
    pub total: usize,
    /// Number of checks with status `Pass`.
    pub passed: usize,
    /// Number of checks with status `Warning`.
    pub warnings: usize,
    /// Number of checks with status `Fail`.
    pub failed: usize,
    /// Human-readable descriptions of critical failing checks.
    pub critical_issues: Vec<String>,
}

impl DiagnosticSummary {
    /// Returns `true` if there are any failures.
    pub fn has_failures(&self) -> bool {
        self.failed > 0
    }
}

impl fmt::Display for DiagnosticSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DiagnosticSummary {{ total: {}, passed: {}, warnings: {}, failed: {} }}",
            self.total, self.passed, self.warnings, self.failed
        )
    }
}

/// Collection of numerical model diagnostics.
///
/// All methods are pure functions operating on slices — no GPU or external
/// runtime is required.
pub struct ModelDiagnostics;

impl ModelDiagnostics {
    // ── Helpers ─────────────────────────────────────────────────────────────

    fn mean(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f32>() / data.len() as f32
    }

    fn std_dev(data: &[f32]) -> f32 {
        if data.len() < 2 {
            return 0.0;
        }
        let m = Self::mean(data);
        let variance = data.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / data.len() as f32;
        variance.sqrt()
    }

    fn l2_norm(data: &[f32]) -> f32 {
        data.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    fn max_abs(data: &[f32]) -> f32 {
        data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max)
    }

    fn percent_zeros(data: &[f32], threshold: f32) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        let zeros = data.iter().filter(|&&x| x.abs() <= threshold).count();
        zeros as f32 / data.len() as f32
    }

    /// Compute the entropy of a probability distribution (in nats).
    fn entropy(probs: &[f32]) -> f32 {
        probs.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum()
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Check whether the L2 norm of `weights` lies within `expected_range`.
    pub fn check_weight_norms(
        weights: &[f32],
        layer_name: &str,
        expected_range: (f32, f32),
    ) -> DiagnosticResult {
        let norm = Self::l2_norm(weights);
        let (lo, hi) = expected_range;
        let status = if norm >= lo && norm <= hi {
            DiagStatus::Pass
        } else if norm < lo * 0.1 || norm > hi * 10.0 {
            DiagStatus::Fail
        } else {
            DiagStatus::Warning
        };
        DiagnosticResult::new(layer_name, "weight_l2_norm", norm, Some(hi), status)
    }

    /// Compute activation statistics for a layer.
    ///
    /// Returns four results: mean, std, percent-zeros, max-abs.
    pub fn check_activation_stats(activations: &[f32], layer_name: &str) -> Vec<DiagnosticResult> {
        if activations.is_empty() {
            return vec![DiagnosticResult::new(
                layer_name,
                "activations",
                0.0,
                None,
                DiagStatus::Info,
            )];
        }

        let mean = Self::mean(activations);
        let std = Self::std_dev(activations);
        let pct_zeros = Self::percent_zeros(activations, 1e-6);
        let max_abs = Self::max_abs(activations);

        let std_status = if std < 1e-7 {
            DiagStatus::Warning // Collapsed activations
        } else if std > 1e4 {
            DiagStatus::Fail // Exploding activations
        } else {
            DiagStatus::Pass
        };

        let zero_status = if pct_zeros > 0.99 {
            DiagStatus::Fail // Almost all dead
        } else if pct_zeros > 0.50 {
            DiagStatus::Warning
        } else {
            DiagStatus::Pass
        };

        let max_status = if max_abs > 1e6 {
            DiagStatus::Fail
        } else if max_abs > 1e4 {
            DiagStatus::Warning
        } else {
            DiagStatus::Pass
        };

        vec![
            DiagnosticResult::new(layer_name, "activation_mean", mean, None, DiagStatus::Info),
            DiagnosticResult::new(layer_name, "activation_std", std, None, std_status),
            DiagnosticResult::new(
                layer_name,
                "activation_pct_zeros",
                pct_zeros,
                Some(0.5),
                zero_status,
            ),
            DiagnosticResult::new(
                layer_name,
                "activation_max_abs",
                max_abs,
                Some(1e4),
                max_status,
            ),
        ]
    }

    /// Check gradient flow across layers, detecting vanishing and exploding gradients.
    ///
    /// `layer_gradients` is a slice of (layer_name, gradient_l2_norm) pairs.
    pub fn check_gradient_flow(layer_gradients: &[(String, f32)]) -> Vec<DiagnosticResult> {
        layer_gradients
            .iter()
            .map(|(name, grad_norm)| {
                let grad_norm = *grad_norm;
                let status = if !(1e-7_f32..=1e4_f32).contains(&grad_norm) {
                    DiagStatus::Fail // Vanishing or exploding gradient
                } else if !(1e-4..=1e2).contains(&grad_norm) {
                    DiagStatus::Warning
                } else {
                    DiagStatus::Pass
                };
                DiagnosticResult::new(name, "gradient_norm", grad_norm, None, status)
            })
            .collect()
    }

    /// Check attention entropy per head.
    ///
    /// `attn_probs` is a flat slice of probabilities laid out as
    /// `[head_0_row_0, head_0_row_1, ..., head_1_row_0, ...]`.
    /// Each row must sum to ~1.0 and each head occupies `len / num_heads` values.
    ///
    /// Returns one result per head.
    pub fn check_attention_entropy(attn_probs: &[f32], num_heads: usize) -> Vec<DiagnosticResult> {
        if num_heads == 0 || attn_probs.is_empty() {
            return Vec::new();
        }

        let per_head = attn_probs.len() / num_heads;
        (0..num_heads)
            .map(|h| {
                let start = h * per_head;
                let end = start + per_head;
                let head_probs = &attn_probs[start..end.min(attn_probs.len())];
                let ent = Self::entropy(head_probs);
                // Maximum possible entropy for a uniform distribution over `per_head` tokens.
                let max_ent = if per_head > 1 { (per_head as f32).ln() } else { 0.0 };

                let status = if ent < 1e-6 {
                    DiagStatus::Fail // Peaked: attending to single token
                } else if max_ent > 0.0 && ent / max_ent > 0.95 {
                    DiagStatus::Warning // Too uniform: not focusing
                } else {
                    DiagStatus::Pass
                };
                DiagnosticResult::new(
                    format!("head_{h}"),
                    "attention_entropy",
                    ent,
                    Some(max_ent),
                    status,
                )
            })
            .collect()
    }

    /// Detect dead neurons in an activation tensor.
    ///
    /// A neuron is considered "dead" when its activation magnitude never exceeds
    /// `threshold` across all provided samples.  Here `activations` is treated as
    /// a single sample and the zero-fraction is computed per-call.
    pub fn detect_dead_neurons(
        activations: &[f32],
        layer_name: &str,
        threshold: f32,
    ) -> DiagnosticResult {
        if activations.is_empty() {
            return DiagnosticResult::new(
                layer_name,
                "dead_neuron_fraction",
                0.0,
                Some(threshold),
                DiagStatus::Info,
            );
        }
        let dead_fraction = Self::percent_zeros(activations, threshold);
        let status = if dead_fraction > 0.99 {
            DiagStatus::Fail
        } else if dead_fraction > 0.50 {
            DiagStatus::Warning
        } else {
            DiagStatus::Pass
        };
        DiagnosticResult::new(
            layer_name,
            "dead_neuron_fraction",
            dead_fraction,
            Some(threshold),
            status,
        )
    }

    /// Detect weight collapse (rank collapse) by examining the ratio of the
    /// maximum absolute value to the standard deviation of the weight tensor.
    ///
    /// A very small std relative to the max-abs suggests weights have collapsed
    /// to a small neighbourhood — a symptom of rank collapse.
    pub fn detect_weight_collapse(weights: &[f32], layer_name: &str) -> DiagnosticResult {
        if weights.is_empty() {
            return DiagnosticResult::new(
                layer_name,
                "weight_collapse_ratio",
                0.0,
                None,
                DiagStatus::Info,
            );
        }
        let std = Self::std_dev(weights);
        let max_abs = Self::max_abs(weights);

        // Collapse ratio: how much the maximum deviates from the std.
        // If std is near zero and max_abs is also near zero → complete collapse.
        let collapse_ratio = if std < 1e-12 {
            if max_abs < 1e-12 {
                0.0 // All zeros
            } else {
                f32::INFINITY
            }
        } else {
            max_abs / std
        };

        let status = if std < 1e-7 {
            DiagStatus::Fail // Collapsed to near-constant
        } else if collapse_ratio > 1000.0 {
            DiagStatus::Warning // High dynamic range — potential collapse
        } else {
            DiagStatus::Pass
        };

        DiagnosticResult::new(
            layer_name,
            "weight_collapse_ratio",
            collapse_ratio,
            Some(1000.0),
            status,
        )
    }

    /// Aggregate a slice of [`DiagnosticResult`]s into a [`DiagnosticSummary`].
    pub fn report_summary(results: &[DiagnosticResult]) -> DiagnosticSummary {
        let total = results.len();
        let mut passed = 0;
        let mut warnings = 0;
        let mut failed = 0;
        let mut critical_issues = Vec::new();

        for r in results {
            match r.status {
                DiagStatus::Pass => passed += 1,
                DiagStatus::Warning => warnings += 1,
                DiagStatus::Fail => {
                    failed += 1;
                    critical_issues.push(format!(
                        "{}/{}: {:.6} (threshold: {})",
                        r.layer_name,
                        r.metric_name,
                        r.value,
                        r.threshold
                            .map(|t| format!("{t:.6}"))
                            .unwrap_or_else(|| "none".to_string()),
                    ));
                },
                DiagStatus::Info => {},
            }
        }

        DiagnosticSummary {
            total,
            passed,
            warnings,
            failed,
            critical_issues,
        }
    }
}

// ─── DiagnosticReport::to_text ────────────────────────────────────────────────

impl DiagnosticReport {
    /// Render the report as a plain-text multi-line string.
    ///
    /// This is an alias for [`DiagnosticReport::to_string_report`] with a
    /// consistent name that matches the task specification.
    pub fn to_text(&self) -> String {
        self.to_string_report()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_error_ctx() -> DiagnosticContext {
        DiagnosticContext::new("Test error", DiagnosticSeverity::Error, "E999")
            .with_why("Because of testing")
            .with_fix("Fix your tests")
            .with_doc("https://example.com/docs")
    }

    #[test]
    fn test_diagnostic_context_builder() {
        let ctx = make_error_ctx();
        assert_eq!(ctx.what, "Test error");
        assert_eq!(ctx.severity, DiagnosticSeverity::Error);
        assert_eq!(ctx.code, "E999");
        assert_eq!(ctx.why.len(), 1);
        assert_eq!(ctx.how_to_fix.len(), 1);
        assert_eq!(ctx.docs.len(), 1);
    }

    #[test]
    fn test_diagnostic_severity_ordering() {
        assert!(DiagnosticSeverity::Fatal > DiagnosticSeverity::Error);
        assert!(DiagnosticSeverity::Error > DiagnosticSeverity::Warning);
        assert!(DiagnosticSeverity::Warning > DiagnosticSeverity::Info);
    }

    #[test]
    fn test_diagnostic_severity_display() {
        assert_eq!(DiagnosticSeverity::Fatal.to_string(), "FATAL");
        assert_eq!(DiagnosticSeverity::Error.to_string(), "ERROR");
        assert_eq!(DiagnosticSeverity::Warning.to_string(), "WARNING");
        assert_eq!(DiagnosticSeverity::Info.to_string(), "INFO");
    }

    #[test]
    fn test_rich_error_display_plain() {
        let ctx = make_error_ctx();
        let err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let rich = RichError::new(err, ctx);
        let plain = rich.display_plain();
        assert!(plain.contains("[ERROR]"));
        assert!(plain.contains("Test error"));
        assert!(plain.contains("file missing"));
    }

    #[test]
    fn test_rich_error_display_rich() {
        let ctx = make_error_ctx();
        let err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let span = ErrorSpan::new("src/lib.rs", 42).with_model("bert-base").with_layer("encoder.0");
        let rich = RichError::new(err, ctx).with_span(span);
        let display = rich.display_rich();
        assert!(display.contains("how to fix"));
        assert!(display.contains("Fix your tests"));
        assert!(display.contains("src/lib.rs:42"));
        assert!(display.contains("bert-base"));
    }

    #[test]
    fn test_rich_error_source() {
        use std::error::Error;
        let ctx = make_error_ctx();
        let err = std::io::Error::other("source error");
        let rich = RichError::new(err, ctx);
        assert!(rich.source().is_some());
    }

    #[test]
    fn test_common_diagnostics_model_not_found() {
        let ctx = CommonDiagnostics::model_not_found("bert-base-uncased");
        assert_eq!(ctx.severity, DiagnosticSeverity::Error);
        assert_eq!(ctx.code, "E001");
        assert!(ctx.what.contains("bert-base-uncased"));
        assert!(!ctx.how_to_fix.is_empty());
    }

    #[test]
    fn test_common_diagnostics_cuda_oom() {
        let ctx = CommonDiagnostics::cuda_out_of_memory(8192, 4096);
        assert_eq!(ctx.severity, DiagnosticSeverity::Fatal);
        assert_eq!(ctx.code, "E004");
        assert!(ctx.what.contains("8192"));
        assert!(ctx.what.contains("4096"));
    }

    #[test]
    fn test_common_diagnostics_sequence_too_long() {
        let ctx = CommonDiagnostics::sequence_too_long(2048, 512);
        assert_eq!(ctx.severity, DiagnosticSeverity::Warning);
        assert_eq!(ctx.code, "W002");
    }

    #[test]
    fn test_diagnostic_report_empty() {
        let report = DiagnosticReport::new();
        assert!(report.is_empty());
        assert_eq!(report.len(), 0);
        assert!(!report.has_errors());
        assert!(!report.has_warnings());
        let s = report.to_string_report();
        assert!(s.contains("No issues"));
    }

    #[test]
    fn test_diagnostic_report_add_and_filter() {
        let mut report = DiagnosticReport::new();
        report.add(
            "pipeline",
            DiagnosticContext::new("warn msg", DiagnosticSeverity::Warning, "W010"),
        );
        report.add(
            "loader",
            DiagnosticContext::new("err msg", DiagnosticSeverity::Error, "E010"),
        );

        assert_eq!(report.len(), 2);
        assert!(report.has_errors());
        assert!(report.has_warnings());

        let errors = report.issues_by_severity(&DiagnosticSeverity::Error);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].code, "E010");

        let warnings = report.issues_by_severity(&DiagnosticSeverity::Warning);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].code, "W010");
    }

    #[test]
    fn test_diagnostic_report_to_string() {
        let mut report = DiagnosticReport::new();
        report.add("model", CommonDiagnostics::model_not_found("my-model"));
        let s = report.to_string_report();
        assert!(s.contains("model"));
        assert!(s.contains("E001"));
    }

    #[test]
    fn test_diagnostic_report_to_json() {
        let mut report = DiagnosticReport::new();
        report.add(
            "hub",
            CommonDiagnostics::hub_download_failed("gpt2", "timeout"),
        );
        let json = report.to_json().expect("JSON serialization must succeed");
        assert!(json.contains("hub"));
        assert!(json.contains("E005"));
    }

    #[test]
    fn test_diagnosable_trait() {
        #[derive(Debug)]
        struct MyError;
        impl fmt::Display for MyError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "my error")
            }
        }
        impl std::error::Error for MyError {}
        impl Diagnosable for MyError {
            fn diagnostic(&self) -> Option<DiagnosticContext> {
                Some(
                    DiagnosticContext::new("My specific error", DiagnosticSeverity::Info, "I001")
                        .with_fix("Do something"),
                )
            }
        }

        let e = MyError;
        let diag = e.diagnostic().expect("should have diagnostic");
        assert_eq!(diag.code, "I001");
        let suggestion = e.suggestion().expect("should have suggestion");
        assert_eq!(suggestion, "Do something");
    }

    #[test]
    fn test_error_span_display() {
        let span = ErrorSpan::new("src/models/bert.rs", 100)
            .with_model("bert-large")
            .with_layer("layer.11.attention");
        let s = span.to_string();
        assert!(s.contains("src/models/bert.rs:100"));
        assert!(s.contains("bert-large"));
        assert!(s.contains("layer.11.attention"));
    }

    // ── ModelDiagnostics tests ─────────────────────────────────────────────────

    #[test]
    fn test_diag_status_display() {
        assert_eq!(DiagStatus::Pass.to_string(), "PASS");
        assert_eq!(DiagStatus::Warning.to_string(), "WARN");
        assert_eq!(DiagStatus::Fail.to_string(), "FAIL");
        assert_eq!(DiagStatus::Info.to_string(), "INFO");
    }

    #[test]
    fn test_check_weight_norms_pass() {
        let weights: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let result = ModelDiagnostics::check_weight_norms(&weights, "layer0", (0.0, 100.0));
        assert_eq!(result.status, DiagStatus::Pass);
        assert!(result.value > 0.0);
    }

    #[test]
    fn test_check_weight_norms_fail_too_large() {
        // Norm will be huge.
        let weights: Vec<f32> = vec![1e8; 64];
        let result = ModelDiagnostics::check_weight_norms(&weights, "layer1", (0.0, 1.0));
        assert_eq!(result.status, DiagStatus::Fail);
    }

    #[test]
    fn test_check_activation_stats_normal() {
        let acts: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
        let results = ModelDiagnostics::check_activation_stats(&acts, "relu_out");
        // Should have four metrics.
        assert_eq!(results.len(), 4);
        let names: Vec<_> = results.iter().map(|r| r.metric_name.as_str()).collect();
        assert!(names.contains(&"activation_mean"));
        assert!(names.contains(&"activation_std"));
        assert!(names.contains(&"activation_pct_zeros"));
        assert!(names.contains(&"activation_max_abs"));
    }

    #[test]
    fn test_check_activation_stats_dead() {
        // All zeros → high dead-neuron fraction.
        let acts: Vec<f32> = vec![0.0; 200];
        let results = ModelDiagnostics::check_activation_stats(&acts, "dead_layer");
        let pct_zeros = results.iter().find(|r| r.metric_name == "activation_pct_zeros").unwrap();
        assert_eq!(pct_zeros.status, DiagStatus::Fail);
    }

    #[test]
    fn test_check_activation_stats_collapsed_std() {
        // All the same value → std ≈ 0 → warning.
        let acts: Vec<f32> = vec![1.0; 100];
        let results = ModelDiagnostics::check_activation_stats(&acts, "collapsed");
        let std_result = results.iter().find(|r| r.metric_name == "activation_std").unwrap();
        assert_eq!(std_result.status, DiagStatus::Warning);
    }

    #[test]
    fn test_check_activation_stats_empty() {
        let results = ModelDiagnostics::check_activation_stats(&[], "empty_layer");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, DiagStatus::Info);
    }

    #[test]
    fn test_check_gradient_flow_healthy() {
        let grads = vec![
            ("layer0".to_string(), 0.01),
            ("layer1".to_string(), 0.05),
            ("layer2".to_string(), 0.03),
        ];
        let results = ModelDiagnostics::check_gradient_flow(&grads);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.status, DiagStatus::Pass);
        }
    }

    #[test]
    fn test_check_gradient_flow_vanishing() {
        let grads = vec![("deep_layer".to_string(), 1e-10)];
        let results = ModelDiagnostics::check_gradient_flow(&grads);
        assert_eq!(results[0].status, DiagStatus::Fail);
    }

    #[test]
    fn test_check_gradient_flow_exploding() {
        let grads = vec![("embed".to_string(), 1e6)];
        let results = ModelDiagnostics::check_gradient_flow(&grads);
        assert_eq!(results[0].status, DiagStatus::Fail);
    }

    #[test]
    fn test_check_attention_entropy_uniform() {
        // Uniform distribution → high entropy → warning.
        let seq_len = 10;
        let num_heads = 2;
        let prob = 1.0 / seq_len as f32;
        let attn: Vec<f32> = vec![prob; seq_len * num_heads];
        let results = ModelDiagnostics::check_attention_entropy(&attn, num_heads);
        assert_eq!(results.len(), num_heads);
        // Entropy should be near maximum → warning status.
        assert!(results
            .iter()
            .any(|r| r.status == DiagStatus::Warning || r.status == DiagStatus::Pass));
    }

    #[test]
    fn test_check_attention_entropy_peaked() {
        // Fully peaked: one token gets all attention → fail.
        let num_heads = 1;
        let seq_len = 10;
        let mut attn = vec![0.0f32; seq_len];
        attn[0] = 1.0; // All attention on token 0.
        let results = ModelDiagnostics::check_attention_entropy(&attn, num_heads);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, DiagStatus::Fail);
    }

    #[test]
    fn test_check_attention_entropy_empty() {
        let results = ModelDiagnostics::check_attention_entropy(&[], 4);
        assert!(results.is_empty());
    }

    #[test]
    fn test_detect_dead_neurons_none() {
        let acts: Vec<f32> = (1..=100).map(|i| i as f32 * 0.01).collect();
        let result = ModelDiagnostics::detect_dead_neurons(&acts, "relu", 1e-6);
        assert_eq!(result.status, DiagStatus::Pass);
    }

    #[test]
    fn test_detect_dead_neurons_all_dead() {
        let acts: Vec<f32> = vec![0.0; 100];
        let result = ModelDiagnostics::detect_dead_neurons(&acts, "dead_relu", 1e-6);
        assert_eq!(result.status, DiagStatus::Fail);
        assert!((result.value - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_detect_dead_neurons_half_dead() {
        // 70 zeros and 30 live neurons → fraction = 0.70 > 0.50 → Warning.
        let mut acts = vec![0.0f32; 70];
        acts.extend((1..=30).map(|i| i as f32 * 0.1));
        let result = ModelDiagnostics::detect_dead_neurons(&acts, "half_dead", 1e-6);
        assert_eq!(result.status, DiagStatus::Warning);
    }

    #[test]
    fn test_detect_weight_collapse_normal() {
        let weights: Vec<f32> = (0..256)
            .map(|i| {
                ((i as f32 * std::f32::consts::TAU) % std::f32::consts::PI)
                    - std::f32::consts::FRAC_PI_2
            })
            .collect();
        let result = ModelDiagnostics::detect_weight_collapse(&weights, "fc1");
        assert_eq!(result.status, DiagStatus::Pass);
    }

    #[test]
    fn test_detect_weight_collapse_all_zeros() {
        let weights = vec![0.0f32; 64];
        let result = ModelDiagnostics::detect_weight_collapse(&weights, "collapsed");
        // std ≈ 0 → Fail.
        assert_eq!(result.status, DiagStatus::Fail);
    }

    #[test]
    fn test_detect_weight_collapse_empty() {
        let result = ModelDiagnostics::detect_weight_collapse(&[], "empty");
        assert_eq!(result.status, DiagStatus::Info);
    }

    #[test]
    fn test_report_summary() {
        let results = vec![
            DiagnosticResult::new("l0", "weight_norm", 1.0, None, DiagStatus::Pass),
            DiagnosticResult::new("l1", "grad_norm", 1e-10, None, DiagStatus::Fail),
            DiagnosticResult::new("l2", "activation_std", 0.1, None, DiagStatus::Warning),
            DiagnosticResult::new("l3", "info_metric", 0.5, None, DiagStatus::Info),
        ];
        let summary = ModelDiagnostics::report_summary(&results);
        assert_eq!(summary.total, 4);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.warnings, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.critical_issues.len(), 1);
        assert!(summary.has_failures());
    }

    #[test]
    fn test_report_summary_all_pass() {
        let results = vec![
            DiagnosticResult::new("l0", "m0", 1.0, None, DiagStatus::Pass),
            DiagnosticResult::new("l1", "m1", 2.0, None, DiagStatus::Pass),
        ];
        let summary = ModelDiagnostics::report_summary(&results);
        assert!(!summary.has_failures());
        assert_eq!(summary.failed, 0);
        assert!(summary.critical_issues.is_empty());
    }

    #[test]
    fn test_report_summary_empty() {
        let summary = ModelDiagnostics::report_summary(&[]);
        assert_eq!(summary.total, 0);
        assert!(!summary.has_failures());
    }

    #[test]
    fn test_diagnostic_result_display() {
        let r = DiagnosticResult::new(
            "encoder.0",
            "weight_norm",
            std::f32::consts::PI,
            Some(10.0),
            DiagStatus::Pass,
        );
        let s = r.to_string();
        assert!(s.contains("PASS"));
        assert!(s.contains("encoder.0"));
        assert!(s.contains("weight_norm"));
    }

    #[test]
    fn test_diagnostic_report_to_text() {
        let mut report = DiagnosticReport::new();
        report.add("model", CommonDiagnostics::model_not_found("bert-tiny"));
        let text = report.to_text();
        assert!(!text.is_empty());
        assert!(text.contains("E001"));
    }

    #[test]
    fn test_diag_summary_display() {
        let summary = DiagnosticSummary {
            total: 5,
            passed: 3,
            warnings: 1,
            failed: 1,
            critical_issues: vec!["layer0/grad_norm".to_string()],
        };
        let s = summary.to_string();
        assert!(s.contains("5"));
        assert!(s.contains("3"));
    }
}
