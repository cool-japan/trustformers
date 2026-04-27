//! Animated gradient flow visualization.
//!
//! Records per-layer gradient statistics across training steps and produces
//! exportable frame sequences (JSON, CSV) and ASCII heatmap animations.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write as _;
use std::path::Path;

// ============================================================================
// Data types
// ============================================================================

/// Statistics for a single layer captured at one training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerGradientInfo {
    /// Layer / parameter name.
    pub name: String,
    /// Mean of absolute gradient values.
    pub mean_abs_grad: f64,
    /// Maximum absolute gradient value.
    pub max_abs_grad: f64,
    /// L2 norm of the gradient vector.
    pub grad_norm: f64,
    /// `true` when the mean absolute gradient is below the vanishing threshold.
    pub is_vanishing: bool,
    /// `true` when the maximum absolute gradient exceeds the exploding threshold.
    pub is_exploding: bool,
    /// Normalised flow intensity in `[0.0, 1.0]` used for visual colour encoding.
    pub flow_intensity: f64,
}

/// One frame of the gradient animation — all layers at a single training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFrame {
    /// The global training step index.
    pub step: u64,
    /// Per-layer information for this frame.
    pub layers: Vec<LayerGradientInfo>,
}

// ============================================================================
// Health classification
// ============================================================================

/// Overall gradient health of the training run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GradientHealth {
    Healthy,
    MinorIssues,
    ProblemsDetected,
    Critical,
}

/// Summary report produced by `GradientFlowAnimator::summary_report()`.
#[derive(Debug, Clone)]
pub struct GradientFlowSummary {
    /// Number of steps recorded.
    pub total_steps: u64,
    /// Layers that exhibited vanishing gradients in at least one frame.
    pub layers_with_vanishing_grads: Vec<String>,
    /// Layers that exhibited exploding gradients in at least one frame.
    pub layers_with_exploding_grads: Vec<String>,
    /// Overall health classification for the run.
    pub overall_health: GradientHealth,
    /// Actionable recommendations derived from the observed gradient behaviour.
    pub recommendations: Vec<String>,
}

// ============================================================================
// Animator
// ============================================================================

/// Collects per-step gradient data and provides export / analysis utilities.
pub struct GradientFlowAnimator {
    frames: Vec<GradientFrame>,
    /// Maximum number of frames to retain (rolling window, oldest dropped first).
    max_frames: usize,
    /// Mean absolute gradient below this value is flagged as vanishing.
    vanishing_threshold: f64,
    /// Maximum absolute gradient above this value is flagged as exploding.
    exploding_threshold: f64,
}

impl GradientFlowAnimator {
    /// Create a new animator.
    ///
    /// * `max_frames` — rolling window size (0 means unlimited).
    pub fn new(max_frames: usize) -> Self {
        Self {
            frames: Vec::new(),
            max_frames,
            vanishing_threshold: 1e-7,
            exploding_threshold: 1e3,
        }
    }

    /// Set a custom vanishing-gradient threshold (default: 1e-7).
    pub fn with_vanishing_threshold(mut self, threshold: f64) -> Self {
        self.vanishing_threshold = threshold;
        self
    }

    /// Set a custom exploding-gradient threshold (default: 1e3).
    pub fn with_exploding_threshold(mut self, threshold: f64) -> Self {
        self.exploding_threshold = threshold;
        self
    }

    /// Record gradient tensors for one training step.
    ///
    /// `gradients` maps a layer name to the flat gradient vector for that layer.
    pub fn record_step(&mut self, step: u64, gradients: &HashMap<String, Vec<f64>>) {
        // Compute global max norm across all layers to normalise `flow_intensity`.
        let global_max_norm: f64 = gradients
            .values()
            .map(|g| l2_norm(g))
            .fold(0.0_f64, f64::max);

        let mut layers: Vec<LayerGradientInfo> = gradients
            .iter()
            .map(|(name, grad)| {
                let mean_abs = mean_abs(grad);
                let max_abs = max_abs(grad);
                let norm = l2_norm(grad);
                let is_vanishing = mean_abs < self.vanishing_threshold;
                let is_exploding = max_abs > self.exploding_threshold;
                let flow_intensity = if global_max_norm > 0.0 {
                    (norm / global_max_norm).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                LayerGradientInfo {
                    name: name.clone(),
                    mean_abs_grad: mean_abs,
                    max_abs_grad: max_abs,
                    grad_norm: norm,
                    is_vanishing,
                    is_exploding,
                    flow_intensity,
                }
            })
            .collect();

        // Stable ordering for deterministic output.
        layers.sort_by(|a, b| a.name.cmp(&b.name));

        let frame = GradientFrame { step, layers };
        self.frames.push(frame);

        // Enforce rolling window.
        if self.max_frames > 0 && self.frames.len() > self.max_frames {
            self.frames.remove(0);
        }
    }

    /// All retained frames (possibly a rolling subset).
    pub fn frames(&self) -> &[GradientFrame] {
        &self.frames
    }

    /// Export all frames to a JSON file.
    pub fn export_json(&self, path: &Path) -> Result<()> {
        let json =
            serde_json::to_string_pretty(&self.frames).context("failed to serialise gradient frames")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create output directory: {}", parent.display()))?;
        }
        std::fs::write(path, json)
            .with_context(|| format!("failed to write gradient animation JSON: {}", path.display()))?;
        Ok(())
    }

    /// Export a CSV timeline: `step,layer,mean_abs_grad,max_abs_grad,grad_norm`.
    pub fn export_csv(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create output directory: {}", parent.display()))?;
        }

        let mut file = std::fs::File::create(path)
            .with_context(|| format!("failed to create CSV file: {}", path.display()))?;

        writeln!(file, "step,layer,mean_abs_grad,max_abs_grad,grad_norm,is_vanishing,is_exploding")
            .context("failed to write CSV header")?;

        for frame in &self.frames {
            for layer in &frame.layers {
                writeln!(
                    file,
                    "{},{},{:.8e},{:.8e},{:.8e},{},{}",
                    frame.step,
                    layer.name,
                    layer.mean_abs_grad,
                    layer.max_abs_grad,
                    layer.grad_norm,
                    layer.is_vanishing as u8,
                    layer.is_exploding as u8,
                )
                .context("failed to write CSV row")?;
            }
        }
        Ok(())
    }

    /// Render an ASCII "heatmap" animation string suitable for terminal display.
    ///
    /// Each row represents one layer; each column one recorded step.
    /// Intensity is encoded with the characters `' ', '░', '▒', '▓', '█'`.
    pub fn to_ascii_animation(&self) -> String {
        if self.frames.is_empty() {
            return "(no gradient frames recorded)\n".to_string();
        }

        // Collect all unique layer names in stable order.
        let layer_names: Vec<String> = {
            let mut seen: HashMap<&str, ()> = HashMap::new();
            let mut names: Vec<String> = Vec::new();
            for frame in &self.frames {
                for layer in &frame.layers {
                    if seen.insert(layer.name.as_str(), ()).is_none() {
                        names.push(layer.name.clone());
                    }
                }
            }
            names.sort();
            names
        };

        let blocks = [' ', '░', '▒', '▓', '█'];
        let max_name_len = layer_names.iter().map(|n| n.len()).max().unwrap_or(8);

        let mut out = String::new();
        out.push_str("Gradient Flow Animation (step → right, layer ↓)\n");
        out.push_str(&format!("{:>width$}  ", "layer", width = max_name_len));
        for (i, _) in self.frames.iter().enumerate() {
            out.push_str(&format!("{}", i % 10));
        }
        out.push('\n');
        out.push_str(&"─".repeat(max_name_len + 2 + self.frames.len()));
        out.push('\n');

        for layer_name in &layer_names {
            out.push_str(&format!("{:>width$}  ", layer_name, width = max_name_len));
            for frame in &self.frames {
                let intensity = frame
                    .layers
                    .iter()
                    .find(|l| l.name == *layer_name)
                    .map(|l| l.flow_intensity)
                    .unwrap_or(0.0);
                let idx = ((intensity * (blocks.len() - 1) as f64).round() as usize)
                    .min(blocks.len() - 1);
                out.push(blocks[idx]);
            }
            out.push('\n');
        }

        out
    }

    /// Generate a summary report for the recorded gradient history.
    pub fn summary_report(&self) -> GradientFlowSummary {
        let total_steps = self.frames.last().map(|f| f.step + 1).unwrap_or(0);

        let mut vanishing: HashMap<String, ()> = HashMap::new();
        let mut exploding: HashMap<String, ()> = HashMap::new();

        for frame in &self.frames {
            for layer in &frame.layers {
                if layer.is_vanishing {
                    vanishing.insert(layer.name.clone(), ());
                }
                if layer.is_exploding {
                    exploding.insert(layer.name.clone(), ());
                }
            }
        }

        let mut layers_with_vanishing_grads: Vec<String> = vanishing.into_keys().collect();
        layers_with_vanishing_grads.sort();
        let mut layers_with_exploding_grads: Vec<String> = exploding.into_keys().collect();
        layers_with_exploding_grads.sort();

        let overall_health = classify_health(
            &layers_with_vanishing_grads,
            &layers_with_exploding_grads,
            &self.frames,
        );

        let recommendations = build_recommendations(
            &overall_health,
            &layers_with_vanishing_grads,
            &layers_with_exploding_grads,
        );

        GradientFlowSummary {
            total_steps,
            layers_with_vanishing_grads,
            layers_with_exploding_grads,
            overall_health,
            recommendations,
        }
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

fn mean_abs(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|v| v.abs()).sum::<f64>() / values.len() as f64
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn classify_health(
    vanishing: &[String],
    exploding: &[String],
    frames: &[GradientFrame],
) -> GradientHealth {
    // Count frames with any issue.
    let issue_frames = frames
        .iter()
        .filter(|f| f.layers.iter().any(|l| l.is_vanishing || l.is_exploding))
        .count();
    let total = frames.len().max(1);
    let issue_ratio = issue_frames as f64 / total as f64;

    if !exploding.is_empty() && issue_ratio > 0.5 {
        return GradientHealth::Critical;
    }
    if !exploding.is_empty() || issue_ratio > 0.3 {
        return GradientHealth::ProblemsDetected;
    }
    if !vanishing.is_empty() || issue_ratio > 0.1 {
        return GradientHealth::MinorIssues;
    }
    GradientHealth::Healthy
}

fn build_recommendations(
    health: &GradientHealth,
    vanishing: &[String],
    exploding: &[String],
) -> Vec<String> {
    let mut recs = Vec::new();

    if !vanishing.is_empty() {
        recs.push(format!(
            "Vanishing gradients detected in: {}. Consider residual connections, layer normalisation, or a larger learning rate.",
            vanishing.join(", ")
        ));
        recs.push("Investigate weight initialisation — Xavier or Kaiming init can prevent early vanishing.".to_string());
    }
    if !exploding.is_empty() {
        recs.push(format!(
            "Exploding gradients detected in: {}. Apply gradient clipping (clip_grad_norm).",
            exploding.join(", ")
        ));
        recs.push("Consider reducing the learning rate or switching to a gradient-friendly optimiser (e.g. AdamW with weight decay).".to_string());
    }
    match health {
        GradientHealth::Critical => {
            recs.push("CRITICAL: Training stability is severely compromised — halt training and diagnose before continuing.".to_string());
        }
        GradientHealth::ProblemsDetected => {
            recs.push("Significant gradient issues detected. Review architecture depth and learning rate schedule.".to_string());
        }
        GradientHealth::MinorIssues => {
            recs.push("Minor gradient issues detected. Monitor closely; intervention may not be required immediately.".to_string());
        }
        GradientHealth::Healthy => {
            recs.push("Gradients appear healthy — no immediate action required.".to_string());
        }
    }

    recs
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn simple_grads(layer_names: &[&str], value: f64) -> HashMap<String, Vec<f64>> {
        layer_names
            .iter()
            .map(|&n| (n.to_string(), vec![value, value, value, value]))
            .collect()
    }

    #[test]
    fn test_record_step_basic() {
        let mut animator = GradientFlowAnimator::new(0);
        let grads = simple_grads(&["attn", "ffn"], 0.01);
        animator.record_step(0, &grads);
        assert_eq!(animator.frames().len(), 1);
        let frame = &animator.frames()[0];
        assert_eq!(frame.layers.len(), 2);
    }

    #[test]
    fn test_rolling_window() {
        let mut animator = GradientFlowAnimator::new(3);
        for i in 0..10u64 {
            let grads = simple_grads(&["layer_a"], 0.1);
            animator.record_step(i, &grads);
        }
        assert_eq!(animator.frames().len(), 3, "rolling window should cap at max_frames");
    }

    #[test]
    fn test_vanishing_detection() {
        let mut animator = GradientFlowAnimator::new(0).with_vanishing_threshold(1e-6);
        let mut grads = HashMap::new();
        grads.insert("shallow".to_string(), vec![1e-8, 1e-8]);
        grads.insert("deep".to_string(), vec![0.01, 0.01]);
        animator.record_step(0, &grads);
        let frame = &animator.frames()[0];
        let shallow = frame.layers.iter().find(|l| l.name == "shallow").unwrap();
        assert!(shallow.is_vanishing);
        let deep = frame.layers.iter().find(|l| l.name == "deep").unwrap();
        assert!(!deep.is_vanishing);
    }

    #[test]
    fn test_exploding_detection() {
        let mut animator = GradientFlowAnimator::new(0).with_exploding_threshold(100.0);
        let mut grads = HashMap::new();
        grads.insert("bad_layer".to_string(), vec![500.0, 200.0]);
        grads.insert("ok_layer".to_string(), vec![0.1, 0.1]);
        animator.record_step(0, &grads);
        let frame = &animator.frames()[0];
        let bad = frame.layers.iter().find(|l| l.name == "bad_layer").unwrap();
        assert!(bad.is_exploding);
        let ok = frame.layers.iter().find(|l| l.name == "ok_layer").unwrap();
        assert!(!ok.is_exploding);
    }

    #[test]
    fn test_flow_intensity_normalised() {
        let mut animator = GradientFlowAnimator::new(0);
        let mut grads = HashMap::new();
        grads.insert("large".to_string(), vec![1.0; 10]);
        grads.insert("small".to_string(), vec![0.001; 10]);
        animator.record_step(0, &grads);
        let frame = &animator.frames()[0];
        for layer in &frame.layers {
            assert!(layer.flow_intensity >= 0.0 && layer.flow_intensity <= 1.0);
        }
    }

    #[test]
    fn test_export_json() -> Result<()> {
        let mut animator = GradientFlowAnimator::new(0);
        animator.record_step(0, &simple_grads(&["a", "b"], 0.1));
        animator.record_step(1, &simple_grads(&["a", "b"], 0.05));

        let path = temp_dir().join(format!("grad_anim_{}.json", uuid::Uuid::new_v4()));
        animator.export_json(&path)?;
        assert!(path.exists());
        let content = std::fs::read_to_string(&path)?;
        let frames: Vec<GradientFrame> = serde_json::from_str(&content)?;
        assert_eq!(frames.len(), 2);
        Ok(())
    }

    #[test]
    fn test_export_csv() -> Result<()> {
        let mut animator = GradientFlowAnimator::new(0);
        animator.record_step(0, &simple_grads(&["encoder"], 0.2));
        animator.record_step(1, &simple_grads(&["encoder"], 0.18));

        let path = temp_dir().join(format!("grad_anim_{}.csv", uuid::Uuid::new_v4()));
        animator.export_csv(&path)?;
        assert!(path.exists());
        let content = std::fs::read_to_string(&path)?;
        // Header + 2 data rows
        assert!(content.lines().count() >= 3);
        assert!(content.contains("step,layer,mean_abs_grad"));
        Ok(())
    }

    #[test]
    fn test_to_ascii_animation_empty() {
        let animator = GradientFlowAnimator::new(0);
        let out = animator.to_ascii_animation();
        assert!(out.contains("no gradient frames"));
    }

    #[test]
    fn test_to_ascii_animation_nonempty() {
        let mut animator = GradientFlowAnimator::new(0);
        for i in 0..5u64 {
            animator.record_step(i, &simple_grads(&["embed", "attn"], 0.1 * i as f64));
        }
        let out = animator.to_ascii_animation();
        assert!(out.contains("embed"));
        assert!(out.contains("attn"));
    }

    #[test]
    fn test_summary_report_healthy() {
        let mut animator = GradientFlowAnimator::new(0);
        for i in 0..5u64 {
            animator.record_step(i, &simple_grads(&["layer"], 0.1));
        }
        let summary = animator.summary_report();
        assert_eq!(summary.overall_health, GradientHealth::Healthy);
        assert!(summary.layers_with_vanishing_grads.is_empty());
        assert!(summary.layers_with_exploding_grads.is_empty());
    }
}
