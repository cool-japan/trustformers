// TensorBoard logging integration for training metrics and visualizations
use crate::tensor::Tensor;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// TensorBoard event writer for logging training metrics and visualizations
pub struct TensorBoardLogger {
    log_dir: PathBuf,
    event_file: Option<BufWriter<File>>,
    step: u64,
    session_id: String,
}

impl TensorBoardLogger {
    /// Create a new TensorBoard logger
    pub fn new<P: AsRef<Path>>(log_dir: P) -> Result<Self> {
        let log_dir = log_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&log_dir)?;

        let session_id = format!(
            "trustformers_{}",
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
        );

        Ok(Self {
            log_dir,
            event_file: None,
            step: 0,
            session_id,
        })
    }

    /// Initialize the event file for writing
    fn init_event_file(&mut self) -> Result<()> {
        if self.event_file.is_none() {
            let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
            let filename = format!("events.out.tfevents.{}.{}", timestamp, self.session_id);
            let filepath = self.log_dir.join(filename);

            let file = OpenOptions::new().create(true).append(true).open(filepath)?;

            self.event_file = Some(BufWriter::new(file));
        }
        Ok(())
    }

    /// Log a scalar value
    pub fn log_scalar(&mut self, tag: &str, value: f32, step: Option<u64>) -> Result<()> {
        self.init_event_file()?;
        let step = step.unwrap_or(self.step);

        let event = TensorBoardEvent::scalar(tag, value, step)?;
        self.write_event(&event)?;

        if step >= self.step {
            self.step = step + 1;
        }

        Ok(())
    }

    /// Log multiple scalar values at once
    pub fn log_scalars(&mut self, scalars: HashMap<String, f32>, step: Option<u64>) -> Result<()> {
        let step = step.unwrap_or(self.step);

        for (tag, value) in scalars {
            self.log_scalar(&tag, value, Some(step))?;
        }

        Ok(())
    }

    /// Log a histogram of values
    pub fn log_histogram(&mut self, tag: &str, values: &[f32], step: Option<u64>) -> Result<()> {
        self.init_event_file()?;
        let step = step.unwrap_or(self.step);

        let event = TensorBoardEvent::histogram(tag, values, step)?;
        self.write_event(&event)?;

        if step >= self.step {
            self.step = step + 1;
        }

        Ok(())
    }

    /// Log tensor values as histogram
    pub fn log_tensor_histogram(
        &mut self,
        tag: &str,
        tensor: &Tensor,
        step: Option<u64>,
    ) -> Result<()> {
        let values = tensor.data()?;
        self.log_histogram(tag, &values, step)
    }

    /// Log attention weights as heatmap
    pub fn log_attention_heatmap(
        &mut self,
        tag: &str,
        attention_weights: &Tensor,
        step: Option<u64>,
    ) -> Result<()> {
        self.init_event_file()?;
        let step = step.unwrap_or(self.step);

        // For now, log as histogram until we implement image logging
        self.log_tensor_histogram(&format!("{}/histogram", tag), attention_weights, Some(step))?;

        // Also log attention statistics
        let weights = attention_weights.data()?;
        let max_attention = weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_attention = weights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let mean_attention = weights.iter().sum::<f32>() / weights.len() as f32;

        self.log_scalar(&format!("{}/max", tag), max_attention, Some(step))?;
        self.log_scalar(&format!("{}/min", tag), min_attention, Some(step))?;
        self.log_scalar(&format!("{}/mean", tag), mean_attention, Some(step))?;

        Ok(())
    }

    /// Log training metrics
    pub fn log_training_metrics(
        &mut self,
        metrics: &TrainingMetrics,
        step: Option<u64>,
    ) -> Result<()> {
        let step = step.unwrap_or(self.step);

        if let Some(loss) = metrics.loss {
            self.log_scalar("loss/train", loss, Some(step))?;
        }

        if let Some(accuracy) = metrics.accuracy {
            self.log_scalar("accuracy/train", accuracy, Some(step))?;
        }

        if let Some(learning_rate) = metrics.learning_rate {
            self.log_scalar("learning_rate", learning_rate, Some(step))?;
        }

        if let Some(grad_norm) = metrics.grad_norm {
            self.log_scalar("grad_norm", grad_norm, Some(step))?;
        }

        Ok(())
    }

    /// Write an event to the log file
    fn write_event(&mut self, event: &TensorBoardEvent) -> Result<()> {
        if let Some(ref mut writer) = self.event_file {
            // Write length-prefixed record
            let serialized = event.serialize()?;
            let length = serialized.len() as u64;

            // TensorBoard format: [length][crc][data][crc]
            writer.write_all(&length.to_le_bytes())?;
            writer.write_all(&Self::crc32(&length.to_le_bytes()).to_le_bytes())?;
            writer.write_all(&serialized)?;
            writer.write_all(&Self::crc32(&serialized).to_le_bytes())?;
            writer.flush()?;
        }
        Ok(())
    }

    /// Simple CRC32 implementation for TensorBoard format
    fn crc32(data: &[u8]) -> u32 {
        // Simplified CRC32 - in production should use proper CRC32 implementation
        let mut crc = 0xffffffffu32;
        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xedb88320;
                } else {
                    crc >>= 1;
                }
            }
        }
        !crc
    }

    /// Flush and close the logger
    pub fn close(&mut self) -> Result<()> {
        if let Some(ref mut writer) = self.event_file {
            writer.flush()?;
        }
        self.event_file = None;
        Ok(())
    }
}

impl Drop for TensorBoardLogger {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// TensorBoard event representation
struct TensorBoardEvent {
    timestamp: f64,
    step: u64,
    tag: String,
    value: EventValue,
}

enum EventValue {
    Scalar(f32),
    Histogram {
        min: f32,
        max: f32,
        num: i64,
        sum: f64,
        sum_squares: f64,
        buckets: Vec<HistogramBucket>,
    },
}

struct HistogramBucket {
    edge: f64,
    count: i64,
}

impl TensorBoardEvent {
    fn scalar(tag: &str, value: f32, step: u64) -> Result<Self> {
        Ok(Self {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64(),
            step,
            tag: tag.to_string(),
            value: EventValue::Scalar(value),
        })
    }

    fn histogram(tag: &str, values: &[f32], step: u64) -> Result<Self> {
        if values.is_empty() {
            return Err(anyhow!("Cannot create histogram from empty values"));
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted_values[0];
        let max = sorted_values[sorted_values.len() - 1];
        let num = values.len() as i64;
        let sum = values.iter().sum::<f32>() as f64;
        let sum_squares = values.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();

        // Create histogram buckets (simplified version)
        let num_buckets = 30.min(values.len());
        let mut buckets = Vec::with_capacity(num_buckets);

        if min != max {
            let bucket_width = (max - min) / num_buckets as f32;
            let mut current_edge = min as f64;
            let mut value_idx = 0;

            for _ in 0..num_buckets {
                current_edge += bucket_width as f64;
                let mut count = 0;

                while value_idx < sorted_values.len()
                    && (sorted_values[value_idx] as f64) <= current_edge
                {
                    count += 1;
                    value_idx += 1;
                }

                buckets.push(HistogramBucket {
                    edge: current_edge,
                    count,
                });
            }
        } else {
            // All values are the same
            buckets.push(HistogramBucket {
                edge: max as f64,
                count: num,
            });
        }

        Ok(Self {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64(),
            step,
            tag: tag.to_string(),
            value: EventValue::Histogram {
                min,
                max,
                num,
                sum,
                sum_squares,
                buckets,
            },
        })
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        // Simplified serialization - in production should use proper Protocol Buffers
        let mut data = Vec::new();

        // Write timestamp
        data.extend_from_slice(&self.timestamp.to_le_bytes());

        // Write step
        data.extend_from_slice(&self.step.to_le_bytes());

        // Write tag length and tag
        let tag_bytes = self.tag.as_bytes();
        data.extend_from_slice(&(tag_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(tag_bytes);

        // Write value based on type
        match &self.value {
            EventValue::Scalar(value) => {
                data.push(0); // Scalar type marker
                data.extend_from_slice(&value.to_le_bytes());
            },
            EventValue::Histogram {
                min,
                max,
                num,
                sum,
                sum_squares,
                buckets,
            } => {
                data.push(1); // Histogram type marker
                data.extend_from_slice(&min.to_le_bytes());
                data.extend_from_slice(&max.to_le_bytes());
                data.extend_from_slice(&num.to_le_bytes());
                data.extend_from_slice(&sum.to_le_bytes());
                data.extend_from_slice(&sum_squares.to_le_bytes());

                // Write buckets
                data.extend_from_slice(&(buckets.len() as u32).to_le_bytes());
                for bucket in buckets {
                    data.extend_from_slice(&bucket.edge.to_le_bytes());
                    data.extend_from_slice(&bucket.count.to_le_bytes());
                }
            },
        }

        Ok(data)
    }
}

/// Training metrics for TensorBoard logging
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    pub loss: Option<f32>,
    pub accuracy: Option<f32>,
    pub learning_rate: Option<f32>,
    pub grad_norm: Option<f32>,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_loss(mut self, loss: f32) -> Self {
        self.loss = Some(loss);
        self
    }

    pub fn with_accuracy(mut self, accuracy: f32) -> Self {
        self.accuracy = Some(accuracy);
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }

    pub fn with_grad_norm(mut self, grad_norm: f32) -> Self {
        self.grad_norm = Some(grad_norm);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_tensorboard_logger_creation() {
        let temp_dir = tempdir().unwrap();
        let _logger = TensorBoardLogger::new(temp_dir.path()).unwrap();
        assert!(temp_dir.path().exists());
    }

    #[test]
    fn test_scalar_logging() -> Result<()> {
        let temp_dir = tempdir().unwrap();
        let mut logger = TensorBoardLogger::new(temp_dir.path())?;

        logger.log_scalar("test/loss", 0.5, Some(0))?;
        logger.log_scalar("test/accuracy", 0.95, Some(1))?;

        Ok(())
    }

    #[test]
    fn test_histogram_logging() -> Result<()> {
        let temp_dir = tempdir().unwrap();
        let mut logger = TensorBoardLogger::new(temp_dir.path())?;

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        logger.log_histogram("test/weights", &values, Some(0))?;

        Ok(())
    }

    #[test]
    fn test_training_metrics_logging() -> Result<()> {
        let temp_dir = tempdir().unwrap();
        let mut logger = TensorBoardLogger::new(temp_dir.path())?;

        let metrics = TrainingMetrics::new()
            .with_loss(0.5)
            .with_accuracy(0.95)
            .with_learning_rate(0.001);

        logger.log_training_metrics(&metrics, Some(0))?;

        Ok(())
    }

    #[test]
    fn test_attention_heatmap_logging() -> Result<()> {
        let temp_dir = tempdir().unwrap();
        let mut logger = TensorBoardLogger::new(temp_dir.path())?;

        // Create mock attention weights
        let attention_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let attention_tensor = Tensor::from_vec(attention_data, &[2, 4])?;

        logger.log_attention_heatmap("attention/layer_0", &attention_tensor, Some(0))?;

        Ok(())
    }

    #[test]
    fn test_multiple_scalars_logging() -> Result<()> {
        let temp_dir = tempdir().unwrap();
        let mut logger = TensorBoardLogger::new(temp_dir.path())?;

        let mut scalars = HashMap::new();
        scalars.insert("train/loss".to_string(), 0.5);
        scalars.insert("train/accuracy".to_string(), 0.95);
        scalars.insert("val/loss".to_string(), 0.6);
        scalars.insert("val/accuracy".to_string(), 0.92);

        logger.log_scalars(scalars, Some(0))?;

        Ok(())
    }
}
