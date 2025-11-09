use crate::core::error::Result;
use crate::error::TrustformersError;
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Enhanced differential update system for TrustformeRS Hub integration
/// Provides efficient model updates using binary diff algorithms and version tracking

/// Version metadata for model tracking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelVersion {
    pub id: String,
    pub model_id: String,
    pub version: String,
    pub parent_version: Option<String>,
    pub created_at: SystemTime,
    pub file_hash: String,
    pub file_size: u64,
    pub compressed_size: Option<u64>,
    pub description: Option<String>,
    pub changes: Vec<ChangeDescription>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChangeDescription {
    pub change_type: ChangeType,
    pub component: String,
    pub description: String,
    pub impact_score: f64, // 0.0 to 1.0, higher means more significant change
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChangeType {
    WeightUpdate,
    ArchitectureChange,
    ConfigUpdate,
    TokenizerUpdate,
    NewLayer,
    LayerRemoval,
    QuantizationChange,
    Other(String),
}

/// Enhanced differential information with advanced compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedDeltaInfo {
    pub base_version: ModelVersion,
    pub target_version: ModelVersion,
    pub delta_url: String,
    pub delta_algorithm: DeltaAlgorithm,
    pub compression_ratio: f64,
    pub delta_size: u64,
    pub full_size: u64,
    pub patch_verification: PatchVerification,
    pub estimated_apply_time: Duration,
    pub bandwidth_savings: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeltaAlgorithm {
    XDelta3,
    BSDiff,
    Custom(String),
    LayerWise, // TrustformeRS-specific algorithm for neural network layers
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchVerification {
    pub pre_patch_hash: String,
    pub post_patch_hash: String,
    pub delta_hash: String,
    pub integrity_checks: Vec<IntegrityCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityCheck {
    pub check_type: CheckType,
    pub expected_value: String,
    pub tolerance: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckType {
    SHA256Hash,
    ModelLayerCount,
    ParameterCount,
    ConfigChecksum,
    TokenizerVocabSize,
}

/// Advanced binary diff engine using multiple algorithms
pub struct BinaryDiffEngine {
    algorithm: DeltaAlgorithm,
    compression_level: u32,
    chunk_size: usize,
    enable_layer_wise: bool,
}

impl Default for BinaryDiffEngine {
    fn default() -> Self {
        Self {
            algorithm: DeltaAlgorithm::LayerWise,
            compression_level: 6,
            chunk_size: 64 * 1024, // 64KB chunks
            enable_layer_wise: true,
        }
    }
}

impl BinaryDiffEngine {
    pub fn new(algorithm: DeltaAlgorithm) -> Self {
        Self {
            algorithm,
            ..Default::default()
        }
    }

    /// Create a binary diff between two model files
    pub fn create_diff(
        &self,
        base_path: &Path,
        target_path: &Path,
        output_path: &Path,
    ) -> Result<EnhancedDeltaInfo> {
        let base_data = fs::read(base_path).map_err(|e| TrustformersError::Io {
            message: format!("Failed to read base file: {}", e),
            path: Some(base_path.display().to_string()),
            suggestion: Some("Check if the file exists and is readable".to_string()),
        })?;
        let target_data = fs::read(target_path).map_err(|e| TrustformersError::Io {
            message: format!("Failed to read target file: {}", e),
            path: Some(target_path.display().to_string()),
            suggestion: Some("Check if the file exists and is readable".to_string()),
        })?;

        let start_time = std::time::Instant::now();

        let delta_data = match &self.algorithm {
            DeltaAlgorithm::XDelta3 => self.create_xdelta3_diff(&base_data, &target_data)?,
            DeltaAlgorithm::BSDiff => self.create_bsdiff(&base_data, &target_data)?,
            DeltaAlgorithm::LayerWise => self.create_layer_wise_diff(&base_data, &target_data)?,
            DeltaAlgorithm::Custom(name) => {
                return Err(TrustformersError::FeatureUnavailable {
                    message: format!("Custom algorithm '{}' not implemented", name),
                    feature: name.clone(),
                    suggestion: Some(
                        "Use one of the built-in algorithms: XDelta3, BSDiff, or LayerWise"
                            .to_string(),
                    ),
                    alternatives: vec![
                        "XDelta3".to_string(),
                        "BSDiff".to_string(),
                        "LayerWise".to_string(),
                    ],
                }
                .into());
            },
        };

        // Compress the delta
        let compressed_delta = self.compress_delta(&delta_data)?;

        // Write compressed delta to output
        fs::write(output_path, &compressed_delta).map_err(|e| TrustformersError::Io {
            message: format!("Failed to write delta file: {}", e),
            path: Some(output_path.display().to_string()),
            suggestion: Some("Check if the directory exists and is writable".to_string()),
        })?;

        let apply_time = start_time.elapsed();
        let compression_ratio = compressed_delta.len() as f64 / target_data.len() as f64;
        let bandwidth_savings = 1.0 - compression_ratio;

        // Create verification info
        let patch_verification = PatchVerification {
            pre_patch_hash: self.calculate_hash(&base_data),
            post_patch_hash: self.calculate_hash(&target_data),
            delta_hash: self.calculate_hash(&compressed_delta),
            integrity_checks: self.create_integrity_checks(&base_data, &target_data)?,
        };

        Ok(EnhancedDeltaInfo {
            base_version: self.create_version_from_data("base", &base_data)?,
            target_version: self.create_version_from_data("target", &target_data)?,
            delta_url: format!("file://{}", output_path.display()),
            delta_algorithm: self.algorithm.clone(),
            compression_ratio,
            delta_size: compressed_delta.len() as u64,
            full_size: target_data.len() as u64,
            patch_verification,
            estimated_apply_time: apply_time,
            bandwidth_savings,
        })
    }

    /// Apply a binary diff to reconstruct the target file
    pub fn apply_diff(
        &self,
        delta_path: &Path,
        base_path: &Path,
        output_path: &Path,
    ) -> Result<()> {
        let compressed_delta = fs::read(delta_path).map_err(|e| TrustformersError::Io {
            message: format!("Failed to read delta file: {}", e),
            path: None,
            suggestion: Some("Check if the file exists and is readable".to_string()),
        })?;

        let base_data = fs::read(base_path).map_err(|e| TrustformersError::Io {
            message: format!("Failed to read base file: {}", e),
            path: None,
            suggestion: Some("Check if the file exists and is readable".to_string()),
        })?;

        // Decompress delta
        let delta_data = self.decompress_delta(&compressed_delta)?;

        let target_data = match &self.algorithm {
            DeltaAlgorithm::XDelta3 => self.apply_xdelta3_diff(&base_data, &delta_data)?,
            DeltaAlgorithm::BSDiff => self.apply_bsdiff(&base_data, &delta_data)?,
            DeltaAlgorithm::LayerWise => self.apply_layer_wise_diff(&base_data, &delta_data)?,
            DeltaAlgorithm::Custom(name) => {
                return Err(TrustformersError::FeatureUnavailable {
                    message: format!("Custom algorithm '{}' not implemented", name),
                    feature: name.clone(),
                    suggestion: Some(
                        "Use one of the built-in algorithms: XDelta3, BSDiff, or LayerWise"
                            .to_string(),
                    ),
                    alternatives: vec![
                        "XDelta3".to_string(),
                        "BSDiff".to_string(),
                        "LayerWise".to_string(),
                    ],
                }
                .into());
            },
        };

        fs::write(output_path, target_data).map_err(|e| TrustformersError::Io {
            message: format!("Failed to write output file: {}", e),
            path: None,
            suggestion: Some("Check if the directory exists and is writable".to_string()),
        })?;

        Ok(())
    }

    /// XDelta3-inspired diff algorithm
    fn create_xdelta3_diff(&self, base: &[u8], target: &[u8]) -> Result<Vec<u8>> {
        let mut diff = Vec::new();
        let mut base_pos = 0;
        let mut target_pos = 0;

        while target_pos < target.len() {
            let window_size = std::cmp::min(self.chunk_size, target.len() - target_pos);
            let target_window = &target[target_pos..target_pos + window_size];

            // Find the best match in base
            if let Some(match_pos) = self.find_best_match(base, target_window, base_pos) {
                // Copy instruction
                diff.extend_from_slice(&[0x01]); // Copy opcode
                diff.extend_from_slice(&(match_pos as u64).to_le_bytes());
                diff.extend_from_slice(&(window_size as u64).to_le_bytes());
                base_pos = match_pos + window_size;
            } else {
                // Add instruction
                diff.extend_from_slice(&[0x02]); // Add opcode
                diff.extend_from_slice(&(window_size as u64).to_le_bytes());
                diff.extend_from_slice(target_window);
            }

            target_pos += window_size;
        }

        Ok(diff)
    }

    /// Apply XDelta3-inspired diff
    fn apply_xdelta3_diff(&self, base: &[u8], delta: &[u8]) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        let mut delta_pos = 0;

        while delta_pos < delta.len() {
            let opcode = delta[delta_pos];
            delta_pos += 1;

            match opcode {
                0x01 => {
                    // Copy instruction
                    let match_pos = u64::from_le_bytes([
                        delta[delta_pos],
                        delta[delta_pos + 1],
                        delta[delta_pos + 2],
                        delta[delta_pos + 3],
                        delta[delta_pos + 4],
                        delta[delta_pos + 5],
                        delta[delta_pos + 6],
                        delta[delta_pos + 7],
                    ]) as usize;
                    delta_pos += 8;

                    let length = u64::from_le_bytes([
                        delta[delta_pos],
                        delta[delta_pos + 1],
                        delta[delta_pos + 2],
                        delta[delta_pos + 3],
                        delta[delta_pos + 4],
                        delta[delta_pos + 5],
                        delta[delta_pos + 6],
                        delta[delta_pos + 7],
                    ]) as usize;
                    delta_pos += 8;

                    if match_pos + length <= base.len() {
                        result.extend_from_slice(&base[match_pos..match_pos + length]);
                    } else {
                        return Err(TrustformersError::InvalidInput {
                            message: "Invalid copy instruction in delta".to_string(),
                            parameter: Some("match_pos".to_string()),
                            expected: Some("valid position and length".to_string()),
                            received: Some(format!("pos: {}, len: {}", match_pos, length)),
                            suggestion: Some("Check if the delta file is corrupted".to_string()),
                        }
                        .into());
                    }
                },
                0x02 => {
                    // Add instruction
                    let length = u64::from_le_bytes([
                        delta[delta_pos],
                        delta[delta_pos + 1],
                        delta[delta_pos + 2],
                        delta[delta_pos + 3],
                        delta[delta_pos + 4],
                        delta[delta_pos + 5],
                        delta[delta_pos + 6],
                        delta[delta_pos + 7],
                    ]) as usize;
                    delta_pos += 8;

                    if delta_pos + length <= delta.len() {
                        result.extend_from_slice(&delta[delta_pos..delta_pos + length]);
                        delta_pos += length;
                    } else {
                        return Err(TrustformersError::InvalidInput {
                            message: "Invalid add instruction in delta".to_string(),
                            parameter: Some("delta_pos".to_string()),
                            expected: Some(format!("length <= {}", delta.len() - delta_pos)),
                            received: Some(format!("length: {}", length)),
                            suggestion: Some("Check if the delta file is corrupted".to_string()),
                        }
                        .into());
                    }
                },
                _ => {
                    return Err(TrustformersError::InvalidInput {
                        message: format!("Unknown opcode in delta: {}", opcode),
                        parameter: Some("opcode".to_string()),
                        expected: Some("0x01 or 0x02".to_string()),
                        received: Some(format!("0x{:02x}", opcode)),
                        suggestion: Some("Check if the delta file is corrupted".to_string()),
                    }
                    .into());
                },
            }
        }

        Ok(result)
    }

    /// Layer-wise diff for neural network models (TrustformeRS-specific)
    fn create_layer_wise_diff(&self, base: &[u8], target: &[u8]) -> Result<Vec<u8>> {
        // This is a simplified implementation of layer-wise diffing
        // In a real implementation, this would parse the model format and diff individual layers

        let mut diff = Vec::new();

        // Header indicating layer-wise format
        diff.extend_from_slice(b"TFRS_LAYER_DIFF_V1");

        // For now, fall back to block-based diffing with model-aware chunking
        let layer_size = 1024 * 1024; // 1MB per "layer" chunk
        let mut pos = 0;

        while pos < std::cmp::max(base.len(), target.len()) {
            let base_chunk = if pos < base.len() {
                let end = std::cmp::min(pos + layer_size, base.len());
                &base[pos..end]
            } else {
                &[]
            };

            let target_chunk = if pos < target.len() {
                let end = std::cmp::min(pos + layer_size, target.len());
                &target[pos..end]
            } else {
                &[]
            };

            if base_chunk != target_chunk {
                // Layer changed
                diff.push(0x03); // Layer change opcode
                diff.extend_from_slice(&(pos as u64).to_le_bytes());
                diff.extend_from_slice(&(target_chunk.len() as u64).to_le_bytes());
                diff.extend_from_slice(target_chunk);
            }

            pos += layer_size;
        }

        Ok(diff)
    }

    /// Apply layer-wise diff
    fn apply_layer_wise_diff(&self, base: &[u8], delta: &[u8]) -> Result<Vec<u8>> {
        if !delta.starts_with(b"TFRS_LAYER_DIFF_V1") {
            return Err(TrustformersError::InvalidInput {
                message: "Invalid layer-wise diff format".to_string(),
                parameter: Some("delta".to_string()),
                expected: Some("TFRS_LAYER_DIFF_V1 header".to_string()),
                received: Some("unknown format".to_string()),
                suggestion: Some(
                    "Ensure the delta file is a valid TrustformeRS layer-wise diff format"
                        .to_string(),
                ),
            }
            .into());
        }

        let mut result = base.to_vec();
        let mut delta_pos = 18; // Skip header

        while delta_pos < delta.len() {
            if delta[delta_pos] != 0x03 {
                return Err(TrustformersError::InvalidInput {
                    message: format!("Invalid layer diff opcode: 0x{:02x}", delta[delta_pos]),
                    parameter: Some("opcode".to_string()),
                    expected: Some("0x03".to_string()),
                    received: Some(format!("0x{:02x}", delta[delta_pos])),
                    suggestion: Some("Check if the delta file is corrupted".to_string()),
                }
                .into());
            }
            delta_pos += 1;

            let offset = u64::from_le_bytes([
                delta[delta_pos],
                delta[delta_pos + 1],
                delta[delta_pos + 2],
                delta[delta_pos + 3],
                delta[delta_pos + 4],
                delta[delta_pos + 5],
                delta[delta_pos + 6],
                delta[delta_pos + 7],
            ]) as usize;
            delta_pos += 8;

            let length = u64::from_le_bytes([
                delta[delta_pos],
                delta[delta_pos + 1],
                delta[delta_pos + 2],
                delta[delta_pos + 3],
                delta[delta_pos + 4],
                delta[delta_pos + 5],
                delta[delta_pos + 6],
                delta[delta_pos + 7],
            ]) as usize;
            delta_pos += 8;

            // Ensure result is large enough
            if offset + length > result.len() {
                result.resize(offset + length, 0);
            }

            // Copy new layer data
            if delta_pos + length <= delta.len() {
                result[offset..offset + length]
                    .copy_from_slice(&delta[delta_pos..delta_pos + length]);
                delta_pos += length;
            } else {
                return Err(TrustformersError::InvalidInput {
                    message: "Invalid layer data in delta".to_string(),
                    parameter: Some("layer_data".to_string()),
                    expected: Some(format!("length <= {}", delta.len() - delta_pos)),
                    received: Some(format!("length: {}", length)),
                    suggestion: Some("Check if the delta file is corrupted".to_string()),
                }
                .into());
            }
        }

        Ok(result)
    }

    /// Simple BSDiff-inspired algorithm
    fn create_bsdiff(&self, base: &[u8], target: &[u8]) -> Result<Vec<u8>> {
        // Simplified BSDiff implementation
        let mut diff = Vec::new();
        diff.extend_from_slice(b"BSDIFF40");

        // This is a basic implementation - real BSDiff uses suffix arrays
        let mut base_pos = 0;
        let mut target_pos = 0;

        while target_pos < target.len() {
            let match_info = self.find_longest_match(base, target, base_pos, target_pos);

            if match_info.length > 8 {
                // Copy instruction
                diff.push(0x01);
                diff.extend_from_slice(&(match_info.base_pos as u64).to_le_bytes());
                diff.extend_from_slice(&(match_info.length as u64).to_le_bytes());
                target_pos += match_info.length;
                base_pos = match_info.base_pos + match_info.length;
            } else {
                // Insert instruction
                diff.push(0x02);
                diff.push(target[target_pos]);
                target_pos += 1;
            }
        }

        Ok(diff)
    }

    /// Apply BSDiff
    fn apply_bsdiff(&self, base: &[u8], delta: &[u8]) -> Result<Vec<u8>> {
        if !delta.starts_with(b"BSDIFF40") {
            return Err(TrustformersError::InvalidInput {
                message: "Invalid BSDiff format".to_string(),
                parameter: Some("delta".to_string()),
                expected: Some("BSDiff40 header".to_string()),
                received: Some("unknown format".to_string()),
                suggestion: Some("Ensure the delta file is a valid BSDiff format".to_string()),
            }
            .into());
        }

        let mut result = Vec::new();
        let mut delta_pos = 8; // Skip header

        while delta_pos < delta.len() {
            match delta[delta_pos] {
                0x01 => {
                    // Copy instruction
                    delta_pos += 1;
                    let base_pos = u64::from_le_bytes([
                        delta[delta_pos],
                        delta[delta_pos + 1],
                        delta[delta_pos + 2],
                        delta[delta_pos + 3],
                        delta[delta_pos + 4],
                        delta[delta_pos + 5],
                        delta[delta_pos + 6],
                        delta[delta_pos + 7],
                    ]) as usize;
                    delta_pos += 8;

                    let length = u64::from_le_bytes([
                        delta[delta_pos],
                        delta[delta_pos + 1],
                        delta[delta_pos + 2],
                        delta[delta_pos + 3],
                        delta[delta_pos + 4],
                        delta[delta_pos + 5],
                        delta[delta_pos + 6],
                        delta[delta_pos + 7],
                    ]) as usize;
                    delta_pos += 8;

                    if base_pos + length <= base.len() {
                        result.extend_from_slice(&base[base_pos..base_pos + length]);
                    } else {
                        return Err(TrustformersError::InvalidInput {
                            message: "Invalid copy in BSDiff".to_string(),
                            parameter: Some("base_pos".to_string()),
                            expected: Some("valid position and length".to_string()),
                            received: Some(format!("pos: {}, len: {}", base_pos, length)),
                            suggestion: Some("Check if the delta file is corrupted".to_string()),
                        }
                        .into());
                    }
                },
                0x02 => {
                    // Insert instruction
                    delta_pos += 1;
                    result.push(delta[delta_pos]);
                    delta_pos += 1;
                },
                _ => {
                    return Err(TrustformersError::InvalidInput {
                        message: format!("Unknown BSDiff opcode: 0x{:02x}", delta[delta_pos]),
                        parameter: Some("opcode".to_string()),
                        expected: Some("valid BSDiff opcode (0x01 or 0x02)".to_string()),
                        received: Some(format!("0x{:02x}", delta[delta_pos])),
                        suggestion: Some("Check if the delta file is corrupted".to_string()),
                    }
                    .into());
                },
            }
        }

        Ok(result)
    }

    /// Compress delta using gzip
    fn compress_delta(&self, delta: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(self.compression_level));
        encoder.write_all(delta).map_err(|e| TrustformersError::InvalidInput {
            message: format!("Failed to compress delta: {}", e),
            parameter: Some("delta_data".to_string()),
            expected: Some("compressible data".to_string()),
            received: None,
            suggestion: Some("Check if the data is valid and not corrupted".to_string()),
        })?;
        encoder.finish().map_err(|e| {
            TrustformersError::InvalidInput {
                message: format!("Failed to finish compression: {}", e),
                parameter: Some("compression".to_string()),
                expected: Some("valid compression state".to_string()),
                received: None,
                suggestion: Some("Check if the compression stream is valid".to_string()),
            }
            .into()
        })
    }

    /// Decompress delta
    fn decompress_delta(&self, compressed: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = GzDecoder::new(compressed);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .map_err(|e| TrustformersError::InvalidInput {
                message: format!("Failed to decompress delta: {}", e),
                parameter: Some("compressed_data".to_string()),
                expected: Some("valid compressed data".to_string()),
                received: None,
                suggestion: Some("Check if the compressed data is not corrupted".to_string()),
            })?;
        Ok(decompressed)
    }

    /// Find best match for a pattern in base data
    fn find_best_match(&self, base: &[u8], pattern: &[u8], start_pos: usize) -> Option<usize> {
        let search_end = if base.len() >= pattern.len() {
            base.len() - pattern.len() + 1
        } else {
            return None;
        };

        (start_pos..search_end).find(|&pos| base[pos..pos + pattern.len()] == *pattern)
    }

    /// Find longest common substring
    fn find_longest_match(
        &self,
        base: &[u8],
        target: &[u8],
        base_start: usize,
        target_start: usize,
    ) -> MatchInfo {
        let mut best_match = MatchInfo {
            base_pos: base_start,
            target_pos: target_start,
            length: 0,
        };

        for base_pos in base_start..base.len() {
            for target_pos in target_start..target.len() {
                let mut length = 0;
                while base_pos + length < base.len()
                    && target_pos + length < target.len()
                    && base[base_pos + length] == target[target_pos + length]
                {
                    length += 1;
                }

                if length > best_match.length {
                    best_match = MatchInfo {
                        base_pos,
                        target_pos,
                        length,
                    };
                }
            }
        }

        best_match
    }

    fn calculate_hash(&self, data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }

    fn create_version_from_data(&self, id: &str, data: &[u8]) -> Result<ModelVersion> {
        Ok(ModelVersion {
            id: id.to_string(),
            model_id: "unknown".to_string(),
            version: "unknown".to_string(),
            parent_version: None,
            created_at: SystemTime::now(),
            file_hash: self.calculate_hash(data),
            file_size: data.len() as u64,
            compressed_size: None,
            description: None,
            changes: Vec::new(),
            metadata: HashMap::new(),
        })
    }

    fn create_integrity_checks(&self, base: &[u8], target: &[u8]) -> Result<Vec<IntegrityCheck>> {
        Ok(vec![
            IntegrityCheck {
                check_type: CheckType::SHA256Hash,
                expected_value: self.calculate_hash(target),
                tolerance: None,
            },
            IntegrityCheck {
                check_type: CheckType::ParameterCount,
                expected_value: target.len().to_string(),
                tolerance: Some(0.01), // 1% tolerance
            },
        ])
    }
}

#[derive(Debug, Clone)]
struct MatchInfo {
    base_pos: usize,
    target_pos: usize,
    length: usize,
}

/// Version manager for tracking model versions and their relationships
pub struct ModelVersionManager {
    versions: HashMap<String, ModelVersion>,
    version_graph: HashMap<String, Vec<String>>, // parent -> children
    storage_path: PathBuf,
}

impl ModelVersionManager {
    pub fn new(storage_path: PathBuf) -> Result<Self> {
        fs::create_dir_all(&storage_path).map_err(|e| TrustformersError::Io {
            message: format!("Failed to create storage directory: {}", e),
            path: Some(storage_path.display().to_string()),
            suggestion: Some(
                "Check if you have write permissions in the parent directory".to_string(),
            ),
        })?;

        let mut manager = Self {
            versions: HashMap::new(),
            version_graph: HashMap::new(),
            storage_path,
        };

        manager.load_versions()?;
        Ok(manager)
    }

    pub fn add_version(&mut self, version: ModelVersion) -> Result<()> {
        // Update parent-child relationships
        if let Some(parent_id) = &version.parent_version {
            self.version_graph
                .entry(parent_id.clone())
                .or_default()
                .push(version.id.clone());
        }

        self.versions.insert(version.id.clone(), version);
        self.save_versions()
    }

    pub fn get_version(&self, version_id: &str) -> Option<&ModelVersion> {
        self.versions.get(version_id)
    }

    pub fn get_version_path(
        &self,
        base_version: &str,
        target_version: &str,
    ) -> Option<Vec<String>> {
        // Find shortest path between versions using BFS
        let mut queue = std::collections::VecDeque::new();
        let mut visited = std::collections::HashSet::new();
        let mut parent_map: HashMap<String, String> = HashMap::new();

        queue.push_back(base_version.to_string());
        visited.insert(base_version.to_string());

        while let Some(current) = queue.pop_front() {
            if current == target_version {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = target_version.to_string();

                while node != base_version {
                    path.push(node.clone());
                    if let Some(parent) = parent_map.get(&node) {
                        node = parent.clone();
                    } else {
                        break;
                    }
                }
                path.push(base_version.to_string());
                path.reverse();
                return Some(path);
            }

            // Add children
            if let Some(children) = self.version_graph.get(&current) {
                for child in children {
                    if !visited.contains(child) {
                        visited.insert(child.clone());
                        parent_map.insert(child.clone(), current.clone());
                        queue.push_back(child.clone());
                    }
                }
            }

            // Add parent (for bidirectional search)
            if let Some(version) = self.versions.get(&current) {
                if let Some(parent) = &version.parent_version {
                    if !visited.contains(parent) {
                        visited.insert(parent.clone());
                        parent_map.insert(parent.clone(), current.clone());
                        queue.push_back(parent.clone());
                    }
                }
            }
        }

        None
    }

    pub fn list_versions(&self) -> Vec<&ModelVersion> {
        self.versions.values().collect()
    }

    pub fn find_optimal_delta_path(&self, base: &str, target: &str) -> Option<Vec<String>> {
        // Find path that minimizes total delta size
        let path = self.get_version_path(base, target)?;

        // For now, return the direct path
        // In a more sophisticated implementation, this would calculate
        // the optimal path considering delta sizes
        Some(path)
    }

    fn load_versions(&mut self) -> Result<()> {
        let versions_file = self.storage_path.join("versions.json");
        if versions_file.exists() {
            let content =
                fs::read_to_string(&versions_file).map_err(|e| TrustformersError::Io {
                    message: format!("Failed to read versions file: {}", e),
                    path: Some(versions_file.display().to_string()),
                    suggestion: Some("Check if the file exists and is readable".to_string()),
                })?;

            let versions: HashMap<String, ModelVersion> =
                serde_json::from_str(&content).map_err(|e| TrustformersError::InvalidInput {
                    message: format!("Failed to parse versions: {}", e),
                    parameter: Some("versions_file".to_string()),
                    expected: Some("valid JSON".to_string()),
                    received: Some("invalid JSON".to_string()),
                    suggestion: Some("Check if the versions file is corrupted".to_string()),
                })?;

            self.versions = versions;

            // Rebuild version graph
            self.version_graph.clear();
            for version in self.versions.values() {
                if let Some(parent_id) = &version.parent_version {
                    self.version_graph
                        .entry(parent_id.clone())
                        .or_default()
                        .push(version.id.clone());
                }
            }
        }
        Ok(())
    }

    fn save_versions(&self) -> Result<()> {
        let versions_file = self.storage_path.join("versions.json");
        let content = serde_json::to_string_pretty(&self.versions).map_err(|e| {
            TrustformersError::InvalidInput {
                message: format!("Failed to serialize versions: {}", e),
                parameter: Some("versions".to_string()),
                expected: Some("serializable data".to_string()),
                received: None,
                suggestion: Some("Check if the version data is valid".to_string()),
            }
        })?;

        fs::write(&versions_file, content).map_err(|e| TrustformersError::Io {
            message: format!("Failed to write versions file: {}", e),
            path: Some(versions_file.display().to_string()),
            suggestion: Some("Check if you have write permissions".to_string()),
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_binary_diff_engine() {
        let engine = BinaryDiffEngine::default();

        let base_data = b"Hello, World!";
        let target_data = b"Hello, Rust!";

        let diff = engine.create_xdelta3_diff(base_data, target_data).unwrap();
        let reconstructed = engine.apply_xdelta3_diff(base_data, &diff).unwrap();

        assert_eq!(target_data, reconstructed.as_slice());
    }

    #[test]
    fn test_layer_wise_diff() {
        let engine = BinaryDiffEngine::new(DeltaAlgorithm::LayerWise);

        let base_data = vec![0u8; 2048]; // 2KB base
        let mut target_data = base_data.clone();
        target_data[1024..1536].fill(255); // Change middle 512 bytes

        let diff = engine.create_layer_wise_diff(&base_data, &target_data).unwrap();
        let reconstructed = engine.apply_layer_wise_diff(&base_data, &diff).unwrap();

        assert_eq!(target_data, reconstructed);
    }

    #[test]
    fn test_version_manager() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = ModelVersionManager::new(temp_dir.path().to_path_buf()).unwrap();

        let version1 = ModelVersion {
            id: "v1".to_string(),
            model_id: "test-model".to_string(),
            version: "1.0".to_string(),
            parent_version: None,
            created_at: SystemTime::now(),
            file_hash: "hash1".to_string(),
            file_size: 1024,
            compressed_size: None,
            description: Some("Initial version".to_string()),
            changes: Vec::new(),
            metadata: HashMap::new(),
        };

        let version2 = ModelVersion {
            id: "v2".to_string(),
            model_id: "test-model".to_string(),
            version: "2.0".to_string(),
            parent_version: Some("v1".to_string()),
            created_at: SystemTime::now(),
            file_hash: "hash2".to_string(),
            file_size: 1024,
            compressed_size: None,
            description: Some("Updated version".to_string()),
            changes: Vec::new(),
            metadata: HashMap::new(),
        };

        manager.add_version(version1).unwrap();
        manager.add_version(version2).unwrap();

        let path = manager.get_version_path("v1", "v2");
        assert!(path.is_some());
        assert_eq!(path.unwrap(), vec!["v1", "v2"]);
    }

    #[test]
    fn test_compression() {
        let engine = BinaryDiffEngine::default();
        let data = b"This is test data for compression. ".repeat(100);

        let compressed = engine.compress_delta(&data).unwrap();
        let decompressed = engine.decompress_delta(&compressed).unwrap();

        assert_eq!(data, decompressed);
        assert!(compressed.len() < data.len()); // Should be compressed
    }
}
