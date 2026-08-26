//! Utility functions for checkpoint handling

#![allow(unused_variables)] // Checkpoint utilities

use anyhow::{anyhow, Result};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::checkpoint::formats::{
    Checkpoint, CheckpointFormat, JaxCheckpoint, PyTorchCheckpoint, TensorFlowCheckpoint,
    TrustformersCheckpoint,
};

/// Detect checkpoint format from file content
pub fn detect_format(path: &Path) -> Result<CheckpointFormat> {
    // First try by extension
    if let Some(format) = CheckpointFormat::from_path(path) {
        return Ok(format);
    }

    // Then try by file content
    let mut file = BufReader::new(File::open(path)?);
    let mut magic_bytes = [0u8; 16];
    file.read_exact(&mut magic_bytes)?;

    // Check for known file signatures
    if magic_bytes.starts_with(b"\x80\x02\x8a") {
        // PyTorch pickle format
        Ok(CheckpointFormat::PyTorch)
    } else if magic_bytes.starts_with(b"\x89HDF") {
        // HDF5 format (TensorFlow)
        Ok(CheckpointFormat::TensorFlow)
    } else if magic_bytes.starts_with(b"\x82\xa4") {
        // MessagePack format (JAX)
        Ok(CheckpointFormat::JAX)
    } else if magic_bytes.starts_with(b"TRUST") {
        // Custom TrustformeRS format
        Ok(CheckpointFormat::Trustformers)
    } else if magic_bytes.starts_with(b"{\"") {
        // JSON format (SafeTensors)
        Ok(CheckpointFormat::SafeTensors)
    } else {
        Err(anyhow!(
            "Unable to detect checkpoint format from file content"
        ))
    }
}

/// Load checkpoint with automatic format detection
pub fn load_checkpoint(path: &Path) -> Result<Box<dyn Checkpoint>> {
    let format = detect_format(path)?;

    match format {
        CheckpointFormat::PyTorch => Ok(Box::new(PyTorchCheckpoint::load(path)?)),
        CheckpointFormat::TensorFlow => Ok(Box::new(TensorFlowCheckpoint::load(path)?)),
        CheckpointFormat::JAX => Ok(Box::new(JaxCheckpoint::load(path)?)),
        CheckpointFormat::Trustformers => Ok(Box::new(TrustformersCheckpoint::load(path)?)),
        _ => Err(anyhow!("Unsupported checkpoint format: {:?}", format)),
    }
}

/// Save checkpoint in specified format
pub fn save_checkpoint(
    checkpoint: &dyn Checkpoint,
    path: &Path,
    format: CheckpointFormat,
) -> Result<()> {
    // If saving in a different format, need to convert first
    if checkpoint.format() != format {
        return Err(anyhow!(
            "Checkpoint format mismatch: {:?} != {:?}",
            checkpoint.format(),
            format
        ));
    }

    checkpoint.save(path)
}

/// Read a checkpoint's metadata without loading its weights.
///
/// For safetensors this parses the JSON header only, so a multi-gigabyte file
/// costs one small read. Formats whose header this crate cannot parse report
/// `weight_count: None` and an empty metadata map — never a guess.
pub fn get_checkpoint_info(path: &Path) -> Result<CheckpointInfo> {
    let format = detect_format(path)?;
    let file_size_bytes = std::fs::metadata(path)?.len();

    let (weight_count, metadata) = match format {
        CheckpointFormat::SafeTensors => read_safetensors_header(path)?,
        // GGUF/PyTorch/TensorFlow/JAX headers are not parsed here; reporting a
        // count for them would be a guess.
        _ => (None, Default::default()),
    };

    Ok(CheckpointInfo {
        format,
        file_size_bytes,
        weight_count,
        metadata,
    })
}

/// Parse a safetensors header: `[u64 header_len][header_len JSON bytes]`.
///
/// Returns the number of weight tensors and the `__metadata__` map.
fn read_safetensors_header(
    path: &Path,
) -> Result<(Option<usize>, std::collections::HashMap<String, String>)> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = std::fs::File::open(path)?;

    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes)?;
    let header_len = u64::from_le_bytes(length_bytes);

    // Guard against a corrupt length claiming a header larger than the file.
    let file_len = file.seek(SeekFrom::End(0))?;
    if header_len == 0 || header_len.saturating_add(8) > file_len {
        return Err(anyhow!(
            "safetensors header length {} is invalid for a {}-byte file",
            header_len,
            file_len
        ));
    }
    file.seek(SeekFrom::Start(8))?;

    let header_len = usize::try_from(header_len)
        .map_err(|_| anyhow!("safetensors header length {} exceeds usize", header_len))?;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;

    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|error| anyhow!("safetensors header is not valid JSON: {}", error))?;
    let object = header
        .as_object()
        .ok_or_else(|| anyhow!("safetensors header is not a JSON object"))?;

    let mut metadata = std::collections::HashMap::new();
    if let Some(serde_json::Value::Object(entries)) = object.get("__metadata__") {
        for (key, value) in entries {
            let rendered = match value {
                serde_json::Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            metadata.insert(key.clone(), rendered);
        }
    }

    // Every key except `__metadata__` names a tensor.
    let weight_count = object.keys().filter(|key| key.as_str() != "__metadata__").count();

    Ok((Some(weight_count), metadata))
}

#[derive(Debug)]
pub struct CheckpointInfo {
    pub format: CheckpointFormat,
    pub file_size_bytes: u64,
    pub weight_count: Option<usize>,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Validate checkpoint integrity
pub fn validate_checkpoint(path: &Path) -> Result<bool> {
    let checkpoint = load_checkpoint(path)?;

    // Basic validation: check that all weights can be loaded
    for name in checkpoint.weight_names() {
        checkpoint.get_weight(&name)?;
    }

    Ok(true)
}

/// Compare two checkpoints for compatibility
pub fn compare_checkpoints(path1: &Path, path2: &Path) -> Result<CheckpointComparison> {
    let checkpoint1 = load_checkpoint(path1)?;
    let checkpoint2 = load_checkpoint(path2)?;

    let names1: std::collections::HashSet<_> = checkpoint1.weight_names().into_iter().collect();
    let names2: std::collections::HashSet<_> = checkpoint2.weight_names().into_iter().collect();

    let common_weights: Vec<_> = names1.intersection(&names2).cloned().collect();
    let only_in_first: Vec<_> = names1.difference(&names2).cloned().collect();
    let only_in_second: Vec<_> = names2.difference(&names1).cloned().collect();

    // Check shape compatibility for common weights
    let mut shape_mismatches = Vec::new();
    for name in &common_weights {
        let weight1 = checkpoint1.get_weight(name)?;
        let weight2 = checkpoint2.get_weight(name)?;

        if weight1.shape != weight2.shape {
            shape_mismatches.push(ShapeMismatch {
                weight_name: name.clone(),
                shape1: weight1.shape,
                shape2: weight2.shape,
            });
        }
    }

    Ok(CheckpointComparison {
        format1: checkpoint1.format(),
        format2: checkpoint2.format(),
        common_weights,
        only_in_first,
        only_in_second,
        shape_mismatches,
    })
}

#[derive(Debug)]
pub struct CheckpointComparison {
    pub format1: CheckpointFormat,
    pub format2: CheckpointFormat,
    pub common_weights: Vec<String>,
    pub only_in_first: Vec<String>,
    pub only_in_second: Vec<String>,
    pub shape_mismatches: Vec<ShapeMismatch>,
}

#[derive(Debug)]
pub struct ShapeMismatch {
    pub weight_name: String,
    pub shape1: Vec<usize>,
    pub shape2: Vec<usize>,
}

impl CheckpointComparison {
    pub fn is_compatible(&self) -> bool {
        self.only_in_first.is_empty()
            && self.only_in_second.is_empty()
            && self.shape_mismatches.is_empty()
    }

    pub fn summary(&self) -> String {
        format!(
            "Checkpoint Comparison:\n\
             - Formats: {:?} vs {:?}\n\
             - Common weights: {}\n\
             - Only in first: {}\n\
             - Only in second: {}\n\
             - Shape mismatches: {}",
            self.format1,
            self.format2,
            self.common_weights.len(),
            self.only_in_first.len(),
            self.only_in_second.len(),
            self.shape_mismatches.len()
        )
    }
}

/// Merge multiple checkpoints (e.g., for sharded models)
pub fn merge_checkpoints(
    paths: &[&Path],
    output_path: &Path,
    format: CheckpointFormat,
) -> Result<()> {
    if paths.is_empty() {
        return Err(anyhow!("No checkpoints to merge"));
    }

    // Load first checkpoint as base
    let mut merged = load_checkpoint(paths[0])?;

    // Merge additional checkpoints
    for path in &paths[1..] {
        let checkpoint = load_checkpoint(path)?;

        // Add weights from this checkpoint
        for name in checkpoint.weight_names() {
            let weight = checkpoint.get_weight(&name)?;
            merged.set_weight(&name, weight)?;
        }
    }

    // Save merged checkpoint
    merged.save(output_path)
}

/// Split checkpoint into shards
pub fn shard_checkpoint(
    path: &Path,
    output_dir: &Path,
    max_shard_size_mb: usize,
) -> Result<Vec<String>> {
    let checkpoint = load_checkpoint(path)?;
    let weight_names = checkpoint.weight_names();

    let mut shards = Vec::new();
    let mut current_shard = TrustformersCheckpoint::new();
    let mut current_size = 0usize;
    let max_size = max_shard_size_mb * 1024 * 1024;

    for name in weight_names {
        let weight = checkpoint.get_weight(&name)?;
        let weight_size = weight.data.len() * std::mem::size_of::<f32>();

        if current_size + weight_size > max_size && current_size > 0 {
            // Save current shard
            let shard_path = output_dir.join(format!("shard_{}.trust", shards.len()));
            current_shard.save(&shard_path)?;
            shards.push(shard_path.to_string_lossy().to_string());

            // Start new shard
            current_shard = TrustformersCheckpoint::new();
            current_size = 0;
        }

        current_shard.set_weight(&name, weight)?;
        current_size += weight_size;
    }

    // Save final shard
    if current_size > 0 {
        let shard_path = output_dir.join(format!("shard_{}.trust", shards.len()));
        current_shard.save(&shard_path)?;
        shards.push(shard_path.to_string_lossy().to_string());
    }

    Ok(shards)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: `get_checkpoint_info` returned `weight_count: None`
    /// and empty metadata for every format, including safetensors, whose
    /// header carries both.
    #[test]
    fn test_safetensors_info_reports_the_real_weight_count() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "trustformers_ckpt_info_{}.safetensors",
            std::process::id()
        ));

        // Header: two tensors plus a metadata block.
        let header = serde_json::json!({
            "__metadata__": {"format": "pt", "producer": "trustformers-test"},
            "encoder.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
            "encoder.bias": {"dtype": "F32", "shape": [2], "data_offsets": [16, 24]},
        });
        let header_bytes = serde_json::to_vec(&header)?;
        let mut file_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
        file_bytes.extend_from_slice(&header_bytes);
        // 24 bytes of tensor payload.
        file_bytes.extend_from_slice(&[0u8; 24]);
        std::fs::write(&path, &file_bytes)?;

        let info = get_checkpoint_info(&path)?;
        assert_eq!(info.format, CheckpointFormat::SafeTensors);
        assert_eq!(
            info.weight_count,
            Some(2),
            "the two tensors in the header must be counted"
        );
        assert_eq!(info.metadata.get("format").map(String::as_str), Some("pt"));
        assert_eq!(info.file_size_bytes, file_bytes.len() as u64);

        std::fs::remove_file(&path).ok();
        Ok(())
    }

    /// A corrupt header length must be an error, not a silent zero count.
    #[test]
    fn test_safetensors_info_rejects_a_corrupt_header() {
        let path = std::env::temp_dir().join(format!(
            "trustformers_ckpt_bad_{}.safetensors",
            std::process::id()
        ));
        // Claims a 1 GiB header inside a 16-byte file.
        let mut bytes = (1_073_741_824u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0u8; 8]);
        std::fs::write(&path, &bytes).expect("write failed");

        assert!(get_checkpoint_info(&path).is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_format_detection_by_extension() {
        assert_eq!(
            CheckpointFormat::from_path(Path::new("model.pt")),
            Some(CheckpointFormat::PyTorch)
        );
        assert_eq!(
            CheckpointFormat::from_path(Path::new("model.ckpt")),
            Some(CheckpointFormat::TensorFlow)
        );
    }

    #[test]
    fn test_checkpoint_comparison() {
        // This would need actual checkpoint files to test properly
        // For now, just test the data structures
        let comparison = CheckpointComparison {
            format1: CheckpointFormat::PyTorch,
            format2: CheckpointFormat::TensorFlow,
            common_weights: vec!["weight1".to_string(), "weight2".to_string()],
            only_in_first: vec!["extra1".to_string()],
            only_in_second: vec!["extra2".to_string()],
            shape_mismatches: vec![ShapeMismatch {
                weight_name: "weight1".to_string(),
                shape1: vec![512, 768],
                shape2: vec![768, 512],
            }],
        };

        assert!(!comparison.is_compatible());
        let summary = comparison.summary();
        assert!(summary.contains("Common weights: 2"));
        assert!(summary.contains("Shape mismatches: 1"));
    }
}
