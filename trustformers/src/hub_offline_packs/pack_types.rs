//! Offline Model Pack System for TrustformeRS: metadata and configuration types for packaging and distributing model collections for offline deployment.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// Offline Model Pack System for TrustformeRS
/// Enables packaging and distribution of model collections for offline deployment
/// Metadata for an offline model pack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPackMetadata {
    pub pack_id: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub created_at: SystemTime,
    pub created_by: String,
    pub total_size: u64,
    pub models: Vec<PackedModelInfo>,
    pub dependencies: Vec<String>,
    pub target_platforms: Vec<String>,
    pub checksum: String,
    pub compression_ratio: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    TextGeneration,
    TextClassification,
    ImageClassification,
    SpeechRecognition,
    Translation,
    Summarization,
    QuestionAnswering,
    Multimodal,
}
/// Configuration for creating model packs
#[derive(Debug, Clone)]
pub struct PackCreationConfig {
    pub compression_level: u8,
    pub include_cache: bool,
    pub include_examples: bool,
    pub include_documentation: bool,
    pub target_platforms: Vec<String>,
    pub max_pack_size: Option<u64>,
    pub split_large_packs: bool,
}
impl Default for PackCreationConfig {
    fn default() -> Self {
        Self {
            compression_level: 6,
            include_cache: false,
            include_examples: true,
            include_documentation: true,
            target_platforms: vec![
                "linux".to_string(),
                "windows".to_string(),
                "macos".to_string(),
            ],
            max_pack_size: Some(2 * 1024 * 1024 * 1024),
            split_large_packs: true,
        }
    }
}
/// Information about a model within a pack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedModelInfo {
    pub model_id: String,
    pub name: String,
    pub version: String,
    pub original_size: u64,
    pub compressed_size: u64,
    pub model_type: ModelType,
    pub framework: String,
    pub precision: PrecisionType,
    pub metadata: HashMap<String, String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrecisionType {
    FP32,
    FP16,
    INT8,
    INT4,
    Mixed,
}
