//! Unified debugging session manager
//!
//! This module provides a high-level API for managing debugging sessions that
//! integrate multiple debugging tools (TensorBoard, visualizers, stability checkers, etc.)

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{
    ActivationVisualizer, AttentionVisualizer, GraphVisualizer, NetronExporter, StabilityChecker,
    TensorBoardWriter,
};

/// Unified debugging session that manages multiple debugging tools
#[derive(Debug)]
pub struct UnifiedDebugSession {
    /// Session ID
    session_id: String,
    /// Session directory for outputs
    session_dir: PathBuf,
    /// TensorBoard writer (optional)
    tensorboard: Option<TensorBoardWriter>,
    /// Activation visualizer
    activation_viz: ActivationVisualizer,
    /// Attention visualizer
    attention_viz: AttentionVisualizer,
    /// Stability checker
    stability_checker: StabilityChecker,
    /// Graph visualizer (optional)
    graph_viz: Option<GraphVisualizer>,
    /// Session configuration
    config: UnifiedDebugSessionConfig,
    /// Current step counter
    step: u64,
}

/// Configuration for debug session
#[derive(Debug, Clone)]
pub struct UnifiedDebugSessionConfig {
    /// Enable TensorBoard logging
    pub enable_tensorboard: bool,
    /// Enable activation visualization
    pub enable_activation_viz: bool,
    /// Enable attention visualization
    pub enable_attention_viz: bool,
    /// Enable stability checking
    pub enable_stability_check: bool,
    /// Enable graph visualization
    pub enable_graph_viz: bool,
    /// Auto-save interval (in steps, 0 = disabled)
    pub auto_save_interval: u64,
    /// Session name prefix
    pub session_name: Option<String>,
}

impl Default for UnifiedDebugSessionConfig {
    fn default() -> Self {
        Self {
            enable_tensorboard: true,
            enable_activation_viz: true,
            enable_attention_viz: true,
            enable_stability_check: true,
            enable_graph_viz: false,
            auto_save_interval: 100,
            session_name: None,
        }
    }
}

/// Summary of debugging session results
#[derive(Debug, Clone)]
pub struct SessionSummary {
    /// Session ID
    pub session_id: String,
    /// Total steps recorded
    pub total_steps: u64,
    /// Number of activations captured
    pub num_activations: usize,
    /// Number of attention patterns captured
    pub num_attention_patterns: usize,
    /// Number of stability issues detected
    pub num_stability_issues: usize,
    /// Session directory
    pub output_directory: PathBuf,
}

impl UnifiedDebugSession {
    /// Create a new debugging session
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Directory where debugging outputs will be saved
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::UnifiedDebugSession;
    ///
    /// let session = UnifiedDebugSession::new("debug_outputs").unwrap();
    /// ```
    pub fn new<P: AsRef<Path>>(output_dir: P) -> Result<Self> {
        Self::with_config(output_dir, UnifiedDebugSessionConfig::default())
    }

    /// Create a debugging session with custom configuration
    pub fn with_config<P: AsRef<Path>>(
        output_dir: P,
        config: UnifiedDebugSessionConfig,
    ) -> Result<Self> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        let session_id = if let Some(ref name) = config.session_name {
            format!("{}_{}", name, timestamp)
        } else {
            format!("debug_session_{}", timestamp)
        };

        let session_dir = output_dir.as_ref().join(&session_id);
        std::fs::create_dir_all(&session_dir)?;

        let tensorboard = if config.enable_tensorboard {
            let tb_dir = session_dir.join("tensorboard");
            Some(TensorBoardWriter::new(&tb_dir)?)
        } else {
            None
        };

        let graph_viz = if config.enable_graph_viz {
            Some(GraphVisualizer::new(&session_id))
        } else {
            None
        };

        Ok(Self {
            session_id,
            session_dir,
            tensorboard,
            activation_viz: ActivationVisualizer::new(),
            attention_viz: AttentionVisualizer::new(),
            stability_checker: StabilityChecker::new(),
            graph_viz,
            config,
            step: 0,
        })
    }

    /// Log a scalar metric
    pub fn log_scalar(&mut self, tag: &str, value: f64) -> Result<()> {
        if let Some(ref mut tb) = self.tensorboard {
            tb.add_scalar(tag, value, self.step)?;
        }
        Ok(())
    }

    /// Log multiple scalars at once
    pub fn log_scalars(&mut self, tag: &str, values: &[(&str, f64)]) -> Result<()> {
        if let Some(tb) = &mut self.tensorboard {
            for (name, value) in values {
                tb.add_scalar(&format!("{}/{}", tag, name), *value, self.step)?;
            }
        }
        Ok(())
    }

    /// Log histogram (e.g., weight distribution)
    pub fn log_histogram(&mut self, tag: &str, values: &[f64]) -> Result<()> {
        if let Some(tb) = &mut self.tensorboard {
            tb.add_histogram(tag, values, self.step)?;
        }
        Ok(())
    }

    /// Register layer activations
    pub fn register_activations(
        &mut self,
        layer_name: &str,
        values: Vec<f32>,
        shape: Vec<usize>,
    ) -> Result<()> {
        if self.config.enable_activation_viz {
            self.activation_viz.register(layer_name, values, shape)?;
        }
        Ok(())
    }

    /// Register attention weights
    pub fn register_attention(
        &mut self,
        layer_name: &str,
        weights: Vec<Vec<Vec<f64>>>,
        tokens: Vec<String>,
    ) -> Result<()> {
        if self.config.enable_attention_viz {
            self.attention_viz.register(
                layer_name,
                weights,
                tokens.clone(),
                tokens,
                crate::AttentionType::SelfAttention,
            )?;
        }
        Ok(())
    }

    /// Check tensor for numerical stability
    pub fn check_stability(&mut self, layer_name: &str, values: &[f64]) -> Result<usize> {
        if self.config.enable_stability_check {
            self.stability_checker.check_tensor(layer_name, values)
        } else {
            Ok(0)
        }
    }

    /// Increment step counter
    pub fn step(&mut self) {
        self.step += 1;

        // Auto-save if enabled
        if self.config.auto_save_interval > 0 && self.step % self.config.auto_save_interval == 0 {
            let _ = self.save();
        }
    }

    /// Get current step
    pub fn current_step(&self) -> u64 {
        self.step
    }

    /// Save all accumulated data to disk
    pub fn save(&mut self) -> Result<()> {
        // Flush TensorBoard
        if let Some(tb) = &mut self.tensorboard {
            tb.flush()?;
        }

        // Export activation summaries
        if self.config.enable_activation_viz && self.activation_viz.num_layers() > 0 {
            let act_summary = self.activation_viz.print_summary()?;
            let act_path = self.session_dir.join("activation_summary.txt");
            std::fs::write(act_path, act_summary)?;
        }

        // Export attention summaries
        if self.config.enable_attention_viz && self.attention_viz.num_layers() > 0 {
            let att_summary = self.attention_viz.summary();
            let att_path = self.session_dir.join("attention_summary.txt");
            std::fs::write(att_path, att_summary)?;
        }

        // Export stability report
        if self.config.enable_stability_check && self.stability_checker.has_issues() {
            let stability_report = self.stability_checker.report();
            let stability_path = self.session_dir.join("stability_report.txt");
            std::fs::write(stability_path, stability_report)?;

            let issues_json = self.session_dir.join("stability_issues.json");
            self.stability_checker.export_to_json(&issues_json)?;
        }

        Ok(())
    }

    /// Export activation visualization for a specific layer
    pub fn export_activation_viz(&self, layer_name: &str, filename: &str) -> Result<()> {
        let path = self.session_dir.join(filename);
        self.activation_viz.export_statistics(layer_name, &path)
    }

    /// Export attention visualization for a specific layer
    pub fn export_attention_viz(&self, layer_name: &str, filename: &str) -> Result<()> {
        let path = self.session_dir.join(filename);
        self.attention_viz.export_to_json(layer_name, &path)
    }

    /// Export attention as BertViz HTML
    pub fn export_attention_bertviz(&self, layer_name: &str, filename: &str) -> Result<()> {
        let path = self.session_dir.join(filename);
        self.attention_viz.export_to_bertviz(layer_name, &path)
    }

    /// Get session summary
    pub fn summary(&self) -> SessionSummary {
        SessionSummary {
            session_id: self.session_id.clone(),
            total_steps: self.step,
            num_activations: self.activation_viz.num_layers(),
            num_attention_patterns: self.attention_viz.num_layers(),
            num_stability_issues: self.stability_checker.total_issues(),
            output_directory: self.session_dir.clone(),
        }
    }

    /// Get session directory path
    pub fn session_dir(&self) -> &Path {
        &self.session_dir
    }

    /// Print summary to string
    pub fn print_summary(&self) -> String {
        let summary = self.summary();
        format!(
            r#"Debug Session Summary
=====================
Session ID: {}
Total Steps: {}
Activations Captured: {}
Attention Patterns: {}
Stability Issues: {}
Output Directory: {}

TensorBoard: {}
Use: tensorboard --logdir={}
"#,
            summary.session_id,
            summary.total_steps,
            summary.num_activations,
            summary.num_attention_patterns,
            summary.num_stability_issues,
            summary.output_directory.display(),
            if self.tensorboard.is_some() { "Enabled" } else { "Disabled" },
            self.session_dir.join("tensorboard").display(),
        )
    }

    /// Close session and perform final save
    pub fn close(mut self) -> Result<SessionSummary> {
        self.save()?;
        Ok(self.summary())
    }

    /// Get reference to activation visualizer
    pub fn activation_visualizer(&self) -> &ActivationVisualizer {
        &self.activation_viz
    }

    /// Get reference to attention visualizer
    pub fn attention_visualizer(&self) -> &AttentionVisualizer {
        &self.attention_viz
    }

    /// Get reference to stability checker
    pub fn stability_checker(&self) -> &StabilityChecker {
        &self.stability_checker
    }

    /// Get mutable reference to graph visualizer (if enabled)
    pub fn graph_visualizer_mut(&mut self) -> Option<&mut GraphVisualizer> {
        self.graph_viz.as_mut()
    }

    /// Export model architecture to Netron format
    pub fn export_model_netron(&self, model_name: &str, description: &str) -> Result<PathBuf> {
        let exporter = NetronExporter::new(model_name, description);
        let path = self.session_dir.join(format!("{}.json", model_name));
        exporter.export(&path)?;
        Ok(path)
    }
}

impl Drop for UnifiedDebugSession {
    fn drop(&mut self) {
        // Auto-save on drop
        let _ = self.save();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_debug_session_creation() {
        let temp_dir = env::temp_dir().join("debug_session_test");
        let session = UnifiedDebugSession::new(&temp_dir).unwrap();

        assert_eq!(session.current_step(), 0);
        assert!(session.session_dir().exists());
    }

    #[test]
    fn test_log_scalar() {
        let temp_dir = env::temp_dir().join("debug_session_scalar_test");
        let mut session = UnifiedDebugSession::new(&temp_dir).unwrap();

        session.log_scalar("test/loss", 0.5).unwrap();
        session.log_scalar("test/accuracy", 0.95).unwrap();
        session.step();

        assert_eq!(session.current_step(), 1);
    }

    #[test]
    fn test_register_activations() {
        let temp_dir = env::temp_dir().join("debug_session_activations_test");
        let mut session = UnifiedDebugSession::new(&temp_dir).unwrap();

        let activations = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        session.register_activations("layer1", activations, vec![5]).unwrap();

        assert_eq!(session.activation_visualizer().num_layers(), 1);
    }

    #[test]
    fn test_check_stability() {
        let temp_dir = env::temp_dir().join("debug_session_stability_test");
        let mut session = UnifiedDebugSession::new(&temp_dir).unwrap();

        let values = vec![1.0, f64::NAN, 2.0];
        let issues = session.check_stability("layer1", &values).unwrap();

        assert!(issues > 0);
        assert!(session.stability_checker().has_issues());
    }

    #[test]
    fn test_session_save() {
        let temp_dir = env::temp_dir().join("debug_session_save_test");
        let mut session = UnifiedDebugSession::new(&temp_dir).unwrap();

        session.log_scalar("test/metric", 1.0).unwrap();
        session.register_activations("layer1", vec![1.0, 2.0, 3.0], vec![3]).unwrap();

        session.save().unwrap();

        // Check that files were created
        let session_dir = session.session_dir();
        assert!(session_dir.exists());
    }

    #[test]
    fn test_session_summary() {
        let temp_dir = env::temp_dir().join("debug_session_summary_test");
        let mut session = UnifiedDebugSession::new(&temp_dir).unwrap();

        session.register_activations("layer1", vec![1.0], vec![1]).unwrap();
        session.step();
        session.step();

        let summary = session.summary();
        assert_eq!(summary.total_steps, 2);
        assert_eq!(summary.num_activations, 1);
    }

    #[test]
    fn test_custom_config() {
        let temp_dir = env::temp_dir().join("debug_session_config_test");

        let config = UnifiedDebugSessionConfig {
            enable_tensorboard: false,
            enable_activation_viz: true,
            enable_attention_viz: false,
            enable_stability_check: true,
            enable_graph_viz: false,
            auto_save_interval: 0,
            session_name: Some("test_session".to_string()),
        };

        let session = UnifiedDebugSession::with_config(&temp_dir, config).unwrap();
        assert!(session.tensorboard.is_none());
        assert!(session.session_id.starts_with("test_session"));
    }

    #[test]
    fn test_auto_save() {
        let temp_dir = env::temp_dir().join("debug_session_autosave_test");

        let config = UnifiedDebugSessionConfig {
            auto_save_interval: 2,
            ..Default::default()
        };

        let mut session = UnifiedDebugSession::with_config(&temp_dir, config).unwrap();

        session.log_scalar("test/value", 1.0).unwrap();
        session.step(); // step 1
        session.step(); // step 2 - should trigger auto-save

        // Session should have saved automatically
        assert_eq!(session.current_step(), 2);
    }

    #[test]
    fn test_print_summary() {
        let temp_dir = env::temp_dir().join("debug_session_print_test");
        let session = UnifiedDebugSession::new(&temp_dir).unwrap();

        let summary_str = session.print_summary();
        assert!(summary_str.contains("Debug Session Summary"));
        assert!(summary_str.contains("Session ID"));
    }
}
