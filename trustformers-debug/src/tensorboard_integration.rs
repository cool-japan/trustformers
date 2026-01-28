//! TensorBoard integration for exporting training metrics and visualizations
//!
//! This module provides functionality to export TrustformeRS debugging data to TensorBoard format,
//! enabling integration with TensorBoard's powerful visualization tools.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use scirs2_core::ndarray::ArrayD;

/// TensorBoard event writer for logging scalars, histograms, and embeddings
#[derive(Debug)]
pub struct TensorBoardWriter {
    log_dir: PathBuf,
    run_name: String,
    step_counter: u64,
    scalar_logs: Vec<ScalarEvent>,
    histogram_logs: Vec<HistogramEvent>,
    text_logs: Vec<TextEvent>,
    embedding_logs: Vec<EmbeddingEvent>,
}

/// Scalar event for logging single values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarEvent {
    pub tag: String,
    pub value: f64,
    pub step: u64,
    pub timestamp: u64,
}

/// Histogram event for logging distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramEvent {
    pub tag: String,
    pub values: Vec<f64>,
    pub step: u64,
    pub timestamp: u64,
    pub min: f64,
    pub max: f64,
    pub num: usize,
    pub sum: f64,
    pub sum_squares: f64,
}

/// Text event for logging text data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEvent {
    pub tag: String,
    pub text: String,
    pub step: u64,
    pub timestamp: u64,
}

/// Embedding event for projector visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingEvent {
    pub tag: String,
    pub embeddings: Vec<Vec<f64>>,
    pub labels: Option<Vec<String>>,
    pub step: u64,
    pub timestamp: u64,
}

/// Graph node for model architecture visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub name: String,
    pub op_type: String,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub attributes: HashMap<String, String>,
}

/// Graph definition for model structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDef {
    pub nodes: Vec<GraphNode>,
    pub metadata: HashMap<String, String>,
}

impl TensorBoardWriter {
    /// Create a new TensorBoard writer with the specified log directory
    ///
    /// # Arguments
    ///
    /// * `log_dir` - Directory where TensorBoard logs will be written
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_debug::TensorBoardWriter;
    ///
    /// let writer = TensorBoardWriter::new("runs/experiment1").unwrap();
    /// ```
    pub fn new<P: AsRef<Path>>(log_dir: P) -> Result<Self> {
        let log_dir = log_dir.as_ref().to_path_buf();
        let run_name = format!(
            "run_{}",
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
        );

        // Create log directory if it doesn't exist
        fs::create_dir_all(&log_dir)?;

        Ok(Self {
            log_dir,
            run_name,
            step_counter: 0,
            scalar_logs: Vec::new(),
            histogram_logs: Vec::new(),
            text_logs: Vec::new(),
            embedding_logs: Vec::new(),
        })
    }

    /// Create a new TensorBoard writer with a custom run name
    pub fn with_run_name<P: AsRef<Path>>(log_dir: P, run_name: String) -> Result<Self> {
        let log_dir = log_dir.as_ref().to_path_buf();

        // Create log directory if it doesn't exist
        fs::create_dir_all(&log_dir)?;

        Ok(Self {
            log_dir,
            run_name,
            step_counter: 0,
            scalar_logs: Vec::new(),
            histogram_logs: Vec::new(),
            text_logs: Vec::new(),
            embedding_logs: Vec::new(),
        })
    }

    /// Add a scalar value to the log
    ///
    /// # Arguments
    ///
    /// * `tag` - Name/identifier for this scalar
    /// * `value` - Scalar value to log
    /// * `step` - Training step number
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use trustformers_debug::TensorBoardWriter;
    /// # let mut writer = TensorBoardWriter::new("runs/test").unwrap();
    /// writer.add_scalar("loss/train", 0.5, 100).unwrap();
    /// writer.add_scalar("accuracy/val", 0.95, 100).unwrap();
    /// ```
    pub fn add_scalar(&mut self, tag: &str, value: f64, step: u64) -> Result<()> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        self.scalar_logs.push(ScalarEvent {
            tag: tag.to_string(),
            value,
            step,
            timestamp,
        });

        Ok(())
    }

    /// Add multiple scalars at once
    pub fn add_scalars(
        &mut self,
        main_tag: &str,
        tag_scalar_dict: HashMap<String, f64>,
        step: u64,
    ) -> Result<()> {
        for (tag, value) in tag_scalar_dict {
            let full_tag = format!("{}/{}", main_tag, tag);
            self.add_scalar(&full_tag, value, step)?;
        }
        Ok(())
    }

    /// Add a histogram of values
    ///
    /// # Arguments
    ///
    /// * `tag` - Name/identifier for this histogram
    /// * `values` - Array of values to create histogram from
    /// * `step` - Training step number
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use trustformers_debug::TensorBoardWriter;
    /// # let mut writer = TensorBoardWriter::new("runs/test").unwrap();
    /// let weights = vec![0.1, 0.2, 0.15, 0.3, 0.25];
    /// writer.add_histogram("layer.0.weight", &weights, 100).unwrap();
    /// ```
    pub fn add_histogram(&mut self, tag: &str, values: &[f64], step: u64) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f64 = values.iter().sum();
        let sum_squares: f64 = values.iter().map(|x| x * x).sum();

        self.histogram_logs.push(HistogramEvent {
            tag: tag.to_string(),
            values: values.to_vec(),
            step,
            timestamp,
            min,
            max,
            num: values.len(),
            sum,
            sum_squares,
        });

        Ok(())
    }

    /// Add text data for logging
    pub fn add_text(&mut self, tag: &str, text: &str, step: u64) -> Result<()> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        self.text_logs.push(TextEvent {
            tag: tag.to_string(),
            text: text.to_string(),
            step,
            timestamp,
        });

        Ok(())
    }

    /// Add embeddings for projector visualization
    ///
    /// # Arguments
    ///
    /// * `tag` - Name for this embedding
    /// * `embeddings` - 2D array of embedding vectors
    /// * `labels` - Optional labels for each embedding
    /// * `step` - Training step number
    pub fn add_embedding(
        &mut self,
        tag: &str,
        embeddings: Vec<Vec<f64>>,
        labels: Option<Vec<String>>,
        step: u64,
    ) -> Result<()> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        self.embedding_logs.push(EmbeddingEvent {
            tag: tag.to_string(),
            embeddings,
            labels,
            step,
            timestamp,
        });

        Ok(())
    }

    /// Add a graph definition for model architecture visualization
    pub fn add_graph(&mut self, graph: &GraphDef) -> Result<()> {
        let graph_path = self.log_dir.join(&self.run_name).join("graph.json");
        fs::create_dir_all(graph_path.parent().unwrap())?;

        let graph_json = serde_json::to_string_pretty(graph)?;
        fs::write(graph_path, graph_json)?;

        Ok(())
    }

    /// Flush all pending logs to disk
    pub fn flush(&mut self) -> Result<()> {
        let run_dir = self.log_dir.join(&self.run_name);
        fs::create_dir_all(&run_dir)?;

        // Write scalar logs
        if !self.scalar_logs.is_empty() {
            let scalars_path = run_dir.join("scalars.jsonl");
            let mut file = File::create(scalars_path)?;
            for event in &self.scalar_logs {
                let line = serde_json::to_string(event)?;
                writeln!(file, "{}", line)?;
            }
        }

        // Write histogram logs
        if !self.histogram_logs.is_empty() {
            let histograms_path = run_dir.join("histograms.jsonl");
            let mut file = File::create(histograms_path)?;
            for event in &self.histogram_logs {
                let line = serde_json::to_string(event)?;
                writeln!(file, "{}", line)?;
            }
        }

        // Write text logs
        if !self.text_logs.is_empty() {
            let text_path = run_dir.join("text.jsonl");
            let mut file = File::create(text_path)?;
            for event in &self.text_logs {
                let line = serde_json::to_string(event)?;
                writeln!(file, "{}", line)?;
            }
        }

        // Write embedding logs
        if !self.embedding_logs.is_empty() {
            let embeddings_path = run_dir.join("embeddings.jsonl");
            let mut file = File::create(embeddings_path)?;
            for event in &self.embedding_logs {
                let line = serde_json::to_string(event)?;
                writeln!(file, "{}", line)?;
            }
        }

        Ok(())
    }

    /// Get the path to the log directory
    pub fn log_dir(&self) -> &Path {
        &self.log_dir
    }

    /// Get the current run name
    pub fn run_name(&self) -> &str {
        &self.run_name
    }

    /// Increment the internal step counter
    pub fn increment_step(&mut self) -> u64 {
        self.step_counter += 1;
        self.step_counter
    }

    /// Get the current step counter value
    pub fn current_step(&self) -> u64 {
        self.step_counter
    }

    /// Close the writer and flush all remaining data
    pub fn close(mut self) -> Result<()> {
        self.flush()
    }
}

impl Drop for TensorBoardWriter {
    fn drop(&mut self) {
        // Auto-flush on drop
        let _ = self.flush();
    }
}

/// Helper function to create a graph node
pub fn create_graph_node(
    name: String,
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
) -> GraphNode {
    GraphNode {
        name,
        op_type,
        input_names: inputs,
        output_names: outputs,
        attributes: HashMap::new(),
    }
}

/// Helper to convert tensor statistics to histogram-compatible format
pub fn tensor_to_histogram_values(tensor: &ArrayD<f32>) -> Vec<f64> {
    tensor.iter().map(|&x| x as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_tensorboard_writer_creation() {
        let temp_dir = env::temp_dir().join("tensorboard_test");
        let writer = TensorBoardWriter::new(&temp_dir).unwrap();
        assert!(writer.log_dir().exists());
    }

    #[test]
    fn test_add_scalar() {
        let temp_dir = env::temp_dir().join("tensorboard_scalar_test");
        let mut writer = TensorBoardWriter::new(&temp_dir).unwrap();

        writer.add_scalar("test/loss", 0.5, 0).unwrap();
        writer.add_scalar("test/loss", 0.4, 1).unwrap();
        writer.add_scalar("test/loss", 0.3, 2).unwrap();

        assert_eq!(writer.scalar_logs.len(), 3);
        assert_eq!(writer.scalar_logs[0].value, 0.5);
        assert_eq!(writer.scalar_logs[1].value, 0.4);
    }

    #[test]
    fn test_add_histogram() {
        let temp_dir = env::temp_dir().join("tensorboard_histogram_test");
        let mut writer = TensorBoardWriter::new(&temp_dir).unwrap();

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        writer.add_histogram("test/weights", &values, 0).unwrap();

        assert_eq!(writer.histogram_logs.len(), 1);
        assert_eq!(writer.histogram_logs[0].min, 1.0);
        assert_eq!(writer.histogram_logs[0].max, 5.0);
        assert_eq!(writer.histogram_logs[0].num, 5);
    }

    #[test]
    fn test_add_text() {
        let temp_dir = env::temp_dir().join("tensorboard_text_test");
        let mut writer = TensorBoardWriter::new(&temp_dir).unwrap();

        writer.add_text("test/note", "This is a test", 0).unwrap();
        assert_eq!(writer.text_logs.len(), 1);
        assert_eq!(writer.text_logs[0].text, "This is a test");
    }

    #[test]
    fn test_flush() {
        let temp_dir = env::temp_dir().join("tensorboard_flush_test");
        let mut writer = TensorBoardWriter::new(&temp_dir).unwrap();

        writer.add_scalar("test/metric", 1.0, 0).unwrap();
        writer.flush().unwrap();

        let scalars_path = temp_dir.join(writer.run_name()).join("scalars.jsonl");
        assert!(scalars_path.exists());
    }

    #[test]
    fn test_add_scalars() {
        let temp_dir = env::temp_dir().join("tensorboard_scalars_test");
        let mut writer = TensorBoardWriter::new(&temp_dir).unwrap();

        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), 0.5);
        metrics.insert("accuracy".to_string(), 0.95);

        writer.add_scalars("train", metrics, 0).unwrap();
        assert_eq!(writer.scalar_logs.len(), 2);
    }

    #[test]
    fn test_add_embedding() {
        let temp_dir = env::temp_dir().join("tensorboard_embedding_test");
        let mut writer = TensorBoardWriter::new(&temp_dir).unwrap();

        let embeddings = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let labels = vec!["class1".to_string(), "class2".to_string()];

        writer.add_embedding("test/emb", embeddings, Some(labels), 0).unwrap();
        assert_eq!(writer.embedding_logs.len(), 1);
    }

    #[test]
    fn test_graph_node_creation() {
        let node = create_graph_node(
            "layer1".to_string(),
            "Linear".to_string(),
            vec!["input".to_string()],
            vec!["output".to_string()],
        );

        assert_eq!(node.name, "layer1");
        assert_eq!(node.op_type, "Linear");
        assert_eq!(node.input_names.len(), 1);
        assert_eq!(node.output_names.len(), 1);
    }
}
