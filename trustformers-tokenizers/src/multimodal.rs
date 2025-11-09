//! Multimodal tokenization for TrustformeRS
//!
//! This module provides tokenization support for multimodal inputs including
//! text + images, text + audio, and other cross-modal combinations.

use crate::{TokenizedInput, Tokenizer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use trustformers_core::errors::Result;

/// Configuration for multimodal tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalConfig {
    /// Maximum sequence length for text
    pub max_text_length: Option<usize>,
    /// Maximum number of image patches
    pub max_image_patches: Option<usize>,
    /// Maximum number of audio frames
    pub max_audio_frames: Option<usize>,
    /// Image patch size
    pub image_patch_size: usize,
    /// Audio frame size
    pub audio_frame_size: usize,
    /// Whether to include special multimodal tokens
    pub include_special_tokens: bool,
    /// Whether to use cross-modal attention
    pub use_cross_modal_attention: bool,
    /// Modality fusion strategy
    pub fusion_strategy: FusionStrategy,
    /// Text tokenizer configuration
    pub text_tokenizer_config: Option<HashMap<String, String>>,
    /// Vision tokenizer configuration
    pub vision_tokenizer_config: Option<HashMap<String, String>>,
    /// Audio tokenizer configuration
    pub audio_tokenizer_config: Option<HashMap<String, String>>,
}

impl Default for MultimodalConfig {
    fn default() -> Self {
        Self {
            max_text_length: Some(512),
            max_image_patches: Some(196), // 14x14 patches for 224x224 image
            max_audio_frames: Some(1000),
            image_patch_size: 16,
            audio_frame_size: 256,
            include_special_tokens: true,
            use_cross_modal_attention: true,
            fusion_strategy: FusionStrategy::Concatenation,
            text_tokenizer_config: None,
            vision_tokenizer_config: None,
            audio_tokenizer_config: None,
        }
    }
}

/// Fusion strategies for multimodal data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Simple concatenation of modalities
    Concatenation,
    /// Interleaved modalities
    Interleaved,
    /// Cross-attention between modalities
    CrossAttention,
    /// Hierarchical fusion
    Hierarchical,
    /// Gate-based fusion
    Gated,
}

/// Types of modalities
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModalityType {
    Text,
    Image,
    Audio,
    Video,
    Depth,
    PointCloud,
    Graph,
    Table,
    Code,
    Custom(String),
}

/// Multimodal token with modality information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalToken {
    /// Token ID
    pub token_id: u32,
    /// Modality type
    pub modality: ModalityType,
    /// Position within modality
    pub modality_position: usize,
    /// Global position in sequence
    pub global_position: usize,
    /// Additional metadata
    pub metadata: Option<MultimodalTokenMetadata>,
}

/// Metadata for multimodal tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalTokenMetadata {
    /// Spatial coordinates (for images/video)
    pub spatial_coords: Option<(usize, usize)>,
    /// Temporal coordinates (for audio/video)
    pub temporal_coords: Option<f64>,
    /// Channel information
    pub channel: Option<usize>,
    /// Confidence score
    pub confidence: Option<f64>,
    /// Feature vector
    pub features: Option<Vec<f32>>,
    /// Attention weights
    pub attention_weights: Option<Vec<f32>>,
}

/// Image patch representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagePatch {
    /// Patch coordinates
    pub x: usize,
    pub y: usize,
    /// Patch size
    pub width: usize,
    pub height: usize,
    /// Flattened pixel values
    pub pixels: Vec<f32>,
    /// Patch embedding
    pub embedding: Option<Vec<f32>>,
}

/// Audio frame representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFrame {
    /// Frame timestamp
    pub timestamp: f64,
    /// Frame duration
    pub duration: f64,
    /// Audio samples
    pub samples: Vec<f32>,
    /// Spectral features
    pub features: Option<Vec<f32>>,
}

/// Video frame representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoFrame {
    /// Frame number
    pub frame_number: usize,
    /// Timestamp
    pub timestamp: f64,
    /// Image patches
    pub patches: Vec<ImagePatch>,
    /// Motion vectors
    pub motion_vectors: Option<Vec<(f32, f32)>>,
}

/// Multimodal input containing different modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalInput {
    /// Text content
    pub text: Option<String>,
    /// Image patches
    pub image_patches: Option<Vec<ImagePatch>>,
    /// Audio frames
    pub audio_frames: Option<Vec<AudioFrame>>,
    /// Video frames
    pub video_frames: Option<Vec<VideoFrame>>,
    /// Table data
    pub table_data: Option<TableData>,
    /// Graph structure
    pub graph_data: Option<GraphData>,
    /// Custom modality data
    pub custom_data: Option<HashMap<String, Vec<u8>>>,
}

/// Table data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableData {
    /// Column headers
    pub headers: Vec<String>,
    /// Table rows
    pub rows: Vec<Vec<String>>,
    /// Column types
    pub column_types: Option<Vec<String>>,
}

/// Graph data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphData {
    /// Node features
    pub nodes: Vec<Vec<f32>>,
    /// Edge list
    pub edges: Vec<(usize, usize)>,
    /// Edge features
    pub edge_features: Option<Vec<Vec<f32>>>,
    /// Node labels
    pub node_labels: Option<Vec<String>>,
}

/// Tokenized multimodal output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalTokenizedInput {
    /// All token IDs in sequence
    pub input_ids: Vec<u32>,
    /// Attention mask
    pub attention_mask: Option<Vec<u32>>,
    /// Token type IDs (modality indicators)
    pub token_type_ids: Option<Vec<u32>>,
    /// Modality tokens with metadata
    pub modality_tokens: Vec<MultimodalToken>,
    /// Modality boundaries
    pub modality_boundaries: HashMap<ModalityType, (usize, usize)>,
    /// Cross-modal attention matrix
    pub cross_modal_attention: Option<Vec<Vec<f32>>>,
}

/// Multimodal tokenizer implementation
pub struct MultimodalTokenizer<T: Tokenizer> {
    text_tokenizer: Arc<T>,
    config: MultimodalConfig,
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    next_id: u32,
    modality_token_ids: HashMap<ModalityType, u32>,
}

impl<T: Tokenizer> MultimodalTokenizer<T> {
    /// Create a new multimodal tokenizer
    pub fn new(text_tokenizer: T, config: MultimodalConfig) -> Self {
        let mut tokenizer = Self {
            text_tokenizer: Arc::new(text_tokenizer),
            config,
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            next_id: 0,
            modality_token_ids: HashMap::new(),
        };

        tokenizer.initialize_vocab();
        tokenizer
    }

    /// Create with default configuration
    pub fn from_text_tokenizer(text_tokenizer: T) -> Self {
        Self::new(text_tokenizer, MultimodalConfig::default())
    }

    /// Initialize vocabulary with multimodal tokens
    fn initialize_vocab(&mut self) {
        // Add special tokens
        if self.config.include_special_tokens {
            self.add_token("[CLS]");
            self.add_token("[SEP]");
            self.add_token("[PAD]");
            self.add_token("[UNK]");
            self.add_token("[MASK]");
        }

        // Add modality-specific tokens
        let modality_tokens = vec![
            (ModalityType::Text, "[TEXT]"),
            (ModalityType::Image, "[IMG]"),
            (ModalityType::Audio, "[AUD]"),
            (ModalityType::Video, "[VID]"),
            (ModalityType::Table, "[TAB]"),
            (ModalityType::Graph, "[GRF]"),
            (ModalityType::Code, "[COD]"),
        ];

        for (modality, token) in modality_tokens {
            let token_id = self.add_token(token);
            self.modality_token_ids.insert(modality, token_id);
        }

        // Add patch and frame tokens
        for i in 0..self.config.max_image_patches.unwrap_or(196) {
            self.add_token(&format!("[PATCH_{}]", i));
        }

        for i in 0..self.config.max_audio_frames.unwrap_or(1000) {
            self.add_token(&format!("[FRAME_{}]", i));
        }

        // Add fusion tokens
        self.add_token("[FUSE]");
        self.add_token("[CROSS_ATTN]");
        self.add_token("[MODAL_SEP]");
    }

    /// Add token to vocabulary
    fn add_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.vocab.get(token) {
            return id;
        }

        let id = self.next_id;
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        self.next_id += 1;
        id
    }

    /// Tokenize multimodal input
    pub fn tokenize_multimodal(&self, input: &MultimodalInput) -> Result<MultimodalTokenizedInput> {
        let mut all_tokens = Vec::new();
        let mut modality_boundaries = HashMap::new();
        let mut current_position = 0;

        // Tokenize text
        if let Some(ref text) = input.text {
            let text_tokens = self.tokenize_text(text, current_position)?;
            let start_pos = current_position;
            all_tokens.extend(text_tokens);
            let end_pos = all_tokens.len();
            modality_boundaries.insert(ModalityType::Text, (start_pos, end_pos));
            current_position = end_pos;
        }

        // Tokenize image patches
        if let Some(ref patches) = input.image_patches {
            let image_tokens = self.tokenize_image_patches(patches, current_position)?;
            let start_pos = current_position;
            all_tokens.extend(image_tokens);
            let end_pos = all_tokens.len();
            modality_boundaries.insert(ModalityType::Image, (start_pos, end_pos));
            current_position = end_pos;
        }

        // Tokenize audio frames
        if let Some(ref frames) = input.audio_frames {
            let audio_tokens = self.tokenize_audio_frames(frames, current_position)?;
            let start_pos = current_position;
            all_tokens.extend(audio_tokens);
            let end_pos = all_tokens.len();
            modality_boundaries.insert(ModalityType::Audio, (start_pos, end_pos));
            current_position = end_pos;
        }

        // Tokenize video frames
        if let Some(ref frames) = input.video_frames {
            let video_tokens = self.tokenize_video_frames(frames, current_position)?;
            let start_pos = current_position;
            all_tokens.extend(video_tokens);
            let end_pos = all_tokens.len();
            modality_boundaries.insert(ModalityType::Video, (start_pos, end_pos));
            current_position = end_pos;
        }

        // Tokenize table data
        if let Some(ref table) = input.table_data {
            let table_tokens = self.tokenize_table(table, current_position)?;
            let start_pos = current_position;
            all_tokens.extend(table_tokens);
            let end_pos = all_tokens.len();
            modality_boundaries.insert(ModalityType::Table, (start_pos, end_pos));
            let _ = end_pos; // Track position for potential future use
        }

        // Apply fusion strategy
        let fused_tokens = self.apply_fusion_strategy(&all_tokens)?;

        // Create input IDs and other outputs
        let input_ids: Vec<u32> = fused_tokens.iter().map(|t| t.token_id).collect();
        let attention_mask = Some(vec![1u32; input_ids.len()]);
        let token_type_ids =
            Some(fused_tokens.iter().map(|t| self.get_modality_type_id(&t.modality)).collect());

        Ok(MultimodalTokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids,
            modality_tokens: fused_tokens,
            modality_boundaries,
            cross_modal_attention: None, // Would be computed during model forward pass
        })
    }

    /// Tokenize text using the underlying text tokenizer
    fn tokenize_text(&self, text: &str, start_position: usize) -> Result<Vec<MultimodalToken>> {
        let text_tokenized = self.text_tokenizer.encode(text)?;
        let mut tokens = Vec::new();

        for (i, &token_id) in text_tokenized.input_ids.iter().enumerate() {
            tokens.push(MultimodalToken {
                token_id,
                modality: ModalityType::Text,
                modality_position: i,
                global_position: start_position + i,
                metadata: None,
            });
        }

        Ok(tokens)
    }

    /// Tokenize image patches
    fn tokenize_image_patches(
        &self,
        patches: &[ImagePatch],
        start_position: usize,
    ) -> Result<Vec<MultimodalToken>> {
        let mut tokens = Vec::new();

        // Add image start token
        if let Some(&img_token_id) = self.modality_token_ids.get(&ModalityType::Image) {
            tokens.push(MultimodalToken {
                token_id: img_token_id,
                modality: ModalityType::Image,
                modality_position: 0,
                global_position: start_position,
                metadata: None,
            });
        }

        // Add patch tokens
        for (i, patch) in patches.iter().enumerate() {
            let patch_token = format!("[PATCH_{}]", i);
            if let Some(&token_id) = self.vocab.get(&patch_token) {
                let metadata = Some(MultimodalTokenMetadata {
                    spatial_coords: Some((patch.x, patch.y)),
                    temporal_coords: None,
                    channel: None,
                    confidence: None,
                    features: patch.embedding.clone(),
                    attention_weights: None,
                });

                tokens.push(MultimodalToken {
                    token_id,
                    modality: ModalityType::Image,
                    modality_position: i + 1,
                    global_position: start_position + tokens.len(),
                    metadata,
                });
            }

            // Stop if we reach max patches
            if tokens.len() >= self.config.max_image_patches.unwrap_or(196) {
                break;
            }
        }

        Ok(tokens)
    }

    /// Tokenize audio frames
    fn tokenize_audio_frames(
        &self,
        frames: &[AudioFrame],
        start_position: usize,
    ) -> Result<Vec<MultimodalToken>> {
        let mut tokens = Vec::new();

        // Add audio start token
        if let Some(&aud_token_id) = self.modality_token_ids.get(&ModalityType::Audio) {
            tokens.push(MultimodalToken {
                token_id: aud_token_id,
                modality: ModalityType::Audio,
                modality_position: 0,
                global_position: start_position,
                metadata: None,
            });
        }

        // Add frame tokens
        for (i, frame) in frames.iter().enumerate() {
            let frame_token = format!("[FRAME_{}]", i);
            if let Some(&token_id) = self.vocab.get(&frame_token) {
                let metadata = Some(MultimodalTokenMetadata {
                    spatial_coords: None,
                    temporal_coords: Some(frame.timestamp),
                    channel: None,
                    confidence: None,
                    features: frame.features.clone(),
                    attention_weights: None,
                });

                tokens.push(MultimodalToken {
                    token_id,
                    modality: ModalityType::Audio,
                    modality_position: i + 1,
                    global_position: start_position + tokens.len(),
                    metadata,
                });
            }

            // Stop if we reach max frames
            if tokens.len() >= self.config.max_audio_frames.unwrap_or(1000) {
                break;
            }
        }

        Ok(tokens)
    }

    /// Tokenize video frames
    fn tokenize_video_frames(
        &self,
        frames: &[VideoFrame],
        start_position: usize,
    ) -> Result<Vec<MultimodalToken>> {
        let mut tokens = Vec::new();

        // Add video start token
        if let Some(&vid_token_id) = self.modality_token_ids.get(&ModalityType::Video) {
            tokens.push(MultimodalToken {
                token_id: vid_token_id,
                modality: ModalityType::Video,
                modality_position: 0,
                global_position: start_position,
                metadata: None,
            });
        }

        // Tokenize each frame as image patches
        for (frame_idx, frame) in frames.iter().enumerate() {
            for (patch_idx, patch) in frame.patches.iter().enumerate() {
                let patch_token = format!("[PATCH_{}]", patch_idx);
                if let Some(&token_id) = self.vocab.get(&patch_token) {
                    let metadata = Some(MultimodalTokenMetadata {
                        spatial_coords: Some((patch.x, patch.y)),
                        temporal_coords: Some(frame.timestamp),
                        channel: Some(frame_idx),
                        confidence: None,
                        features: patch.embedding.clone(),
                        attention_weights: None,
                    });

                    tokens.push(MultimodalToken {
                        token_id,
                        modality: ModalityType::Video,
                        modality_position: tokens.len(),
                        global_position: start_position + tokens.len(),
                        metadata,
                    });
                }
            }
        }

        Ok(tokens)
    }

    /// Tokenize table data
    fn tokenize_table(
        &self,
        table: &TableData,
        start_position: usize,
    ) -> Result<Vec<MultimodalToken>> {
        let mut tokens = Vec::new();

        // Add table start token
        if let Some(&tab_token_id) = self.modality_token_ids.get(&ModalityType::Table) {
            tokens.push(MultimodalToken {
                token_id: tab_token_id,
                modality: ModalityType::Table,
                modality_position: 0,
                global_position: start_position,
                metadata: None,
            });
        }

        // Tokenize headers and rows as text
        let mut table_text = table.headers.join(" | ");
        for row in &table.rows {
            table_text.push_str(" | ");
            table_text.push_str(&row.join(" | "));
        }

        let text_tokens = self.text_tokenizer.encode(&table_text)?;
        for (i, &token_id) in text_tokens.input_ids.iter().enumerate() {
            tokens.push(MultimodalToken {
                token_id,
                modality: ModalityType::Table,
                modality_position: i + 1,
                global_position: start_position + tokens.len(),
                metadata: None,
            });
        }

        Ok(tokens)
    }

    /// Apply fusion strategy to tokens
    fn apply_fusion_strategy(&self, tokens: &[MultimodalToken]) -> Result<Vec<MultimodalToken>> {
        match self.config.fusion_strategy {
            FusionStrategy::Concatenation => Ok(tokens.to_vec()),
            FusionStrategy::Interleaved => self.apply_interleaved_fusion(tokens),
            FusionStrategy::CrossAttention => self.apply_cross_attention_fusion(tokens),
            FusionStrategy::Hierarchical => self.apply_hierarchical_fusion(tokens),
            FusionStrategy::Gated => self.apply_gated_fusion(tokens),
        }
    }

    /// Apply interleaved fusion
    fn apply_interleaved_fusion(&self, tokens: &[MultimodalToken]) -> Result<Vec<MultimodalToken>> {
        // Group tokens by modality
        let mut modality_groups: HashMap<ModalityType, Vec<&MultimodalToken>> = HashMap::new();
        for token in tokens {
            modality_groups.entry(token.modality.clone()).or_default().push(token);
        }

        // Interleave tokens from different modalities
        let mut result = Vec::new();
        let max_len = modality_groups.values().map(|v| v.len()).max().unwrap_or(0);

        for i in 0..max_len {
            for group in modality_groups.values() {
                if let Some(token) = group.get(i) {
                    result.push((*token).clone());
                }
            }
        }

        Ok(result)
    }

    /// Apply cross-attention fusion
    fn apply_cross_attention_fusion(
        &self,
        tokens: &[MultimodalToken],
    ) -> Result<Vec<MultimodalToken>> {
        // Group tokens by modality
        let mut modality_groups: HashMap<ModalityType, Vec<&MultimodalToken>> = HashMap::new();
        for token in tokens {
            modality_groups.entry(token.modality.clone()).or_default().push(token);
        }

        // If we have less than 2 modalities, no cross-attention needed
        if modality_groups.len() < 2 {
            return Ok(tokens.to_vec());
        }

        let mut result = Vec::new();
        let modalities: Vec<_> = modality_groups.keys().cloned().collect();

        // Add cross-attention token between modality groups
        if let Some(&cross_attn_token_id) = self.vocab.get("[CROSS_ATTN]") {
            for (i, (modality, group)) in modality_groups.iter().enumerate() {
                // Add original tokens from this modality
                for token in group {
                    let mut enhanced_token = (*token).clone();

                    // Calculate attention weights with other modalities
                    let mut attention_weights = Vec::new();
                    for (j, other_modality) in modalities.iter().enumerate() {
                        if i != j {
                            // Simple attention score based on position and modality compatibility
                            let attention_score = self.calculate_cross_modal_attention_score(
                                modality,
                                other_modality,
                                token.modality_position,
                            );
                            attention_weights.push(attention_score);
                        }
                    }

                    // Update token metadata with attention weights
                    if let Some(ref mut metadata) = enhanced_token.metadata {
                        metadata.attention_weights = Some(attention_weights);
                    } else {
                        enhanced_token.metadata = Some(MultimodalTokenMetadata {
                            spatial_coords: None,
                            temporal_coords: None,
                            channel: None,
                            confidence: None,
                            features: None,
                            attention_weights: Some(attention_weights),
                        });
                    }

                    result.push(enhanced_token);
                }

                // Add cross-attention separator between modalities (except last)
                if i < modality_groups.len() - 1 {
                    result.push(MultimodalToken {
                        token_id: cross_attn_token_id,
                        modality: ModalityType::Custom("cross_attention".to_string()),
                        modality_position: 0,
                        global_position: result.len(),
                        metadata: None,
                    });
                }
            }
        } else {
            // If cross-attention token not available, just return original tokens
            result = tokens.to_vec();
        }

        Ok(result)
    }

    /// Calculate cross-modal attention score between two modalities
    fn calculate_cross_modal_attention_score(
        &self,
        source_modality: &ModalityType,
        target_modality: &ModalityType,
        position: usize,
    ) -> f32 {
        // Base attention score based on modality compatibility
        let base_score = match (source_modality, target_modality) {
            // Text-Image interactions are typically strong
            (ModalityType::Text, ModalityType::Image)
            | (ModalityType::Image, ModalityType::Text) => 0.8,
            // Text-Audio interactions
            (ModalityType::Text, ModalityType::Audio)
            | (ModalityType::Audio, ModalityType::Text) => 0.7,
            // Image-Video interactions are very strong
            (ModalityType::Image, ModalityType::Video)
            | (ModalityType::Video, ModalityType::Image) => 0.9,
            // Audio-Video interactions for multimedia content
            (ModalityType::Audio, ModalityType::Video)
            | (ModalityType::Video, ModalityType::Audio) => 0.75,
            // Table-Text interactions for structured data
            (ModalityType::Table, ModalityType::Text)
            | (ModalityType::Text, ModalityType::Table) => 0.6,
            // Code-Text interactions
            (ModalityType::Code, ModalityType::Text) | (ModalityType::Text, ModalityType::Code) => {
                0.65
            },
            // Graph-Table interactions for structured data
            (ModalityType::Graph, ModalityType::Table)
            | (ModalityType::Table, ModalityType::Graph) => 0.7,
            // Same modality gets moderate attention
            (a, b) if a == b => 0.5,
            // Default for other combinations
            _ => 0.4,
        };

        // Position-based attention decay (closer positions get higher attention)
        let position_factor = 1.0 / (1.0 + (position as f32 * 0.1));

        // Combine base score with position factor
        base_score * position_factor
    }

    /// Apply hierarchical fusion
    fn apply_hierarchical_fusion(
        &self,
        tokens: &[MultimodalToken],
    ) -> Result<Vec<MultimodalToken>> {
        // Group by modality and add fusion tokens between groups
        let mut result = Vec::new();
        let mut current_modality = None;

        if let Some(&fuse_token_id) = self.vocab.get("[FUSE]") {
            for token in tokens {
                if current_modality.is_some() && current_modality.as_ref() != Some(&token.modality)
                {
                    // Add fusion token between modalities
                    result.push(MultimodalToken {
                        token_id: fuse_token_id,
                        modality: ModalityType::Custom("fusion".to_string()),
                        modality_position: 0,
                        global_position: result.len(),
                        metadata: None,
                    });
                }
                result.push(token.clone());
                current_modality = Some(token.modality.clone());
            }
        }

        Ok(result)
    }

    /// Apply gated fusion
    fn apply_gated_fusion(&self, tokens: &[MultimodalToken]) -> Result<Vec<MultimodalToken>> {
        // Group tokens by modality
        let mut modality_groups: HashMap<ModalityType, Vec<&MultimodalToken>> = HashMap::new();
        for token in tokens {
            modality_groups.entry(token.modality.clone()).or_default().push(token);
        }

        // If we have less than 2 modalities, no gating needed
        if modality_groups.len() < 2 {
            return Ok(tokens.to_vec());
        }

        let mut result = Vec::new();

        // Calculate gate weights for each modality based on content characteristics
        let mut modality_gates: HashMap<ModalityType, f32> = HashMap::new();
        for (modality, group) in &modality_groups {
            let gate_weight = self.calculate_modality_gate_weight(modality, group);
            modality_gates.insert(modality.clone(), gate_weight);
        }

        // Normalize gate weights so they sum to 1.0
        let total_weight: f32 = modality_gates.values().sum();
        if total_weight > 0.0 {
            for weight in modality_gates.values_mut() {
                *weight /= total_weight;
            }
        }

        // Apply gated fusion by weighting tokens based on modality gates
        for (modality, group) in &modality_groups {
            let gate_weight = modality_gates.get(modality).copied().unwrap_or(0.0);

            for token in group {
                let mut gated_token = (*token).clone();

                // Calculate confidence based on gate weight and token characteristics
                let token_confidence = self.calculate_token_confidence(token, gate_weight);

                // Update token metadata with gate information
                if let Some(ref mut metadata) = gated_token.metadata {
                    metadata.confidence = Some(token_confidence as f64);
                } else {
                    gated_token.metadata = Some(MultimodalTokenMetadata {
                        spatial_coords: None,
                        temporal_coords: None,
                        channel: None,
                        confidence: Some(token_confidence as f64),
                        features: None,
                        attention_weights: None,
                    });
                }

                // Only include tokens that pass the gating threshold
                if token_confidence > 0.1 {
                    result.push(gated_token);
                }
            }
        }

        // Sort tokens by confidence (highest first) to prioritize important content
        result.sort_by(|a, b| {
            let conf_a = a.metadata.as_ref().and_then(|m| m.confidence).unwrap_or(0.0);
            let conf_b = b.metadata.as_ref().and_then(|m| m.confidence).unwrap_or(0.0);
            conf_b.partial_cmp(&conf_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update global positions after sorting
        for (i, token) in result.iter_mut().enumerate() {
            token.global_position = i;
        }

        Ok(result)
    }

    /// Calculate gate weight for a modality based on its tokens
    fn calculate_modality_gate_weight(
        &self,
        modality: &ModalityType,
        tokens: &[&MultimodalToken],
    ) -> f32 {
        if tokens.is_empty() {
            return 0.0;
        }

        // Base weight based on modality importance
        let base_weight = match modality {
            ModalityType::Text => 1.0,      // Text is usually most important
            ModalityType::Image => 0.8,     // Images are highly informative
            ModalityType::Video => 0.9,     // Video combines visual and temporal info
            ModalityType::Audio => 0.7,     // Audio adds complementary information
            ModalityType::Table => 0.6,     // Structured data is valuable but less dynamic
            ModalityType::Graph => 0.65,    // Graph data is structured and informative
            ModalityType::Code => 0.75,     // Code has high semantic value
            ModalityType::Custom(_) => 0.5, // Custom modalities get moderate weight
            _ => 0.4,                       // Other modalities get lower weight
        };

        // Factor in the number of tokens (more tokens might indicate more importance)
        let token_count_factor = (tokens.len() as f32).sqrt() / 10.0;

        // Factor in feature richness (tokens with more metadata are more informative)
        let feature_richness = tokens
            .iter()
            .map(|token| {
                if let Some(metadata) = &token.metadata {
                    let mut richness = 0.0;
                    if metadata.spatial_coords.is_some() {
                        richness += 0.2;
                    }
                    if metadata.temporal_coords.is_some() {
                        richness += 0.2;
                    }
                    if metadata.features.is_some() {
                        richness += 0.4;
                    }
                    if metadata.confidence.is_some() {
                        richness += 0.2;
                    }
                    richness
                } else {
                    0.1 // Base richness for tokens without metadata
                }
            })
            .sum::<f32>()
            / tokens.len() as f32;

        // Combine factors
        base_weight * (1.0 + token_count_factor) * (1.0 + feature_richness)
    }

    /// Calculate confidence for a token based on gate weight and token characteristics
    fn calculate_token_confidence(&self, token: &MultimodalToken, gate_weight: f32) -> f32 {
        // Start with the gate weight as base confidence
        let mut confidence = gate_weight;

        // Factor in token-specific characteristics
        if let Some(metadata) = &token.metadata {
            // Spatial information increases confidence for visual modalities
            if metadata.spatial_coords.is_some()
                && matches!(token.modality, ModalityType::Image | ModalityType::Video)
            {
                confidence *= 1.2;
            }

            // Temporal information increases confidence for temporal modalities
            if metadata.temporal_coords.is_some()
                && matches!(token.modality, ModalityType::Audio | ModalityType::Video)
            {
                confidence *= 1.15;
            }

            // Feature vectors indicate processed/meaningful content
            if let Some(features) = &metadata.features {
                let feature_magnitude =
                    features.iter().map(|f| f.abs()).sum::<f32>() / features.len() as f32;
                confidence *= 1.0 + (feature_magnitude * 0.1);
            }

            // Existing confidence scores should be respected
            if let Some(existing_confidence) = metadata.confidence {
                confidence = (confidence + existing_confidence as f32) / 2.0;
            }
        }

        // Position-based confidence (earlier tokens might be more important)
        let position_factor = 1.0 / (1.0 + (token.modality_position as f32 * 0.05));
        confidence *= position_factor;

        // Ensure confidence is in valid range [0, 1]
        confidence.clamp(0.0, 1.0)
    }

    /// Get modality type ID for token type IDs
    fn get_modality_type_id(&self, modality: &ModalityType) -> u32 {
        match modality {
            ModalityType::Text => 0,
            ModalityType::Image => 1,
            ModalityType::Audio => 2,
            ModalityType::Video => 3,
            ModalityType::Table => 4,
            ModalityType::Graph => 5,
            ModalityType::Code => 6,
            ModalityType::Custom(_) => 7,
            _ => 0,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &MultimodalConfig {
        &self.config
    }

    /// Get vocabulary
    pub fn get_vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    /// Get underlying text tokenizer
    pub fn text_tokenizer(&self) -> &T {
        &self.text_tokenizer
    }
}

impl<T: Tokenizer> Tokenizer for MultimodalTokenizer<T> {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        // For plain text, just use the underlying text tokenizer
        self.text_tokenizer.encode(text)
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        // Filter out special multimodal tokens and decode text tokens
        let text_tokens: Vec<u32> = token_ids
            .iter()
            .copied()
            .filter(|&id| {
                if let Some(token) = self.id_to_token.get(&id) {
                    !token.starts_with('[') || !token.ends_with(']')
                } else {
                    true
                }
            })
            .collect();

        self.text_tokenizer.decode(&text_tokens)
    }

    fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<TokenizedInput> {
        // For text pairs, use the underlying text tokenizer
        self.text_tokenizer.encode_pair(text_a, text_b)
    }

    fn vocab_size(&self) -> usize {
        self.text_tokenizer.vocab_size() + self.vocab.len()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        let mut vocab = self.text_tokenizer.get_vocab();
        for (token, &id) in &self.vocab {
            vocab.insert(token.clone(), id);
        }
        vocab
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab
            .get(token)
            .copied()
            .or_else(|| self.text_tokenizer.token_to_id(token))
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.id_to_token
            .get(&id)
            .cloned()
            .or_else(|| self.text_tokenizer.id_to_token(id))
    }
}

/// Utilities for multimodal tokenization
pub struct MultimodalUtils;

impl MultimodalUtils {
    /// Create image patches from image dimensions
    pub fn create_image_patches(
        image_width: usize,
        image_height: usize,
        patch_size: usize,
    ) -> Vec<ImagePatch> {
        let mut patches = Vec::new();

        for y in (0..image_height).step_by(patch_size) {
            for x in (0..image_width).step_by(patch_size) {
                let width = (patch_size).min(image_width - x);
                let height = (patch_size).min(image_height - y);

                patches.push(ImagePatch {
                    x,
                    y,
                    width,
                    height,
                    pixels: vec![0.0; width * height * 3], // RGB
                    embedding: None,
                });
            }
        }

        patches
    }

    /// Create audio frames from audio parameters
    pub fn create_audio_frames(
        sample_rate: f64,
        duration: f64,
        frame_size: usize,
        hop_size: usize,
    ) -> Vec<AudioFrame> {
        let mut frames = Vec::new();
        let total_samples = (sample_rate * duration) as usize;

        for start in (0..total_samples).step_by(hop_size) {
            let end = (start + frame_size).min(total_samples);
            let timestamp = start as f64 / sample_rate;
            let frame_duration = (end - start) as f64 / sample_rate;

            frames.push(AudioFrame {
                timestamp,
                duration: frame_duration,
                samples: vec![0.0; end - start],
                features: None,
            });
        }

        frames
    }

    /// Convert tokenized input to multimodal format
    pub fn convert_to_multimodal(
        tokenized: TokenizedInput,
        modality: ModalityType,
    ) -> MultimodalTokenizedInput {
        let modality_tokens: Vec<MultimodalToken> = tokenized
            .input_ids
            .into_iter()
            .enumerate()
            .map(|(i, token_id)| MultimodalToken {
                token_id,
                modality: modality.clone(),
                modality_position: i,
                global_position: i,
                metadata: None,
            })
            .collect();

        let mut boundaries = HashMap::new();
        boundaries.insert(modality, (0, modality_tokens.len()));

        MultimodalTokenizedInput {
            input_ids: modality_tokens.iter().map(|t| t.token_id).collect(),
            attention_mask: Some(tokenized.attention_mask.into_iter().map(|x| x as u32).collect()),
            token_type_ids: tokenized.token_type_ids,
            modality_tokens,
            modality_boundaries: boundaries,
            cross_modal_attention: None,
        }
    }

    /// Calculate cross-modal attention matrix
    pub fn calculate_cross_modal_attention(
        tokens: &[MultimodalToken],
        query_modality: &ModalityType,
        key_modality: &ModalityType,
    ) -> Vec<Vec<f32>> {
        let query_tokens: Vec<_> =
            tokens.iter().filter(|t| &t.modality == query_modality).collect();
        let key_tokens: Vec<_> = tokens.iter().filter(|t| &t.modality == key_modality).collect();

        // Placeholder attention calculation
        let mut attention = vec![vec![0.0; key_tokens.len()]; query_tokens.len()];

        for (i, _) in query_tokens.iter().enumerate() {
            for (j, _) in key_tokens.iter().enumerate() {
                // Simplified attention score
                attention[i][j] = 1.0 / (key_tokens.len() as f32);
            }
        }

        attention
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::char::CharTokenizer;
    use std::collections::HashMap;

    fn create_test_char_tokenizer() -> CharTokenizer {
        let mut vocab = HashMap::new();
        vocab.insert("[PAD]".to_string(), 0);
        vocab.insert("[UNK]".to_string(), 1);
        vocab.insert("[CLS]".to_string(), 2);
        vocab.insert("[SEP]".to_string(), 3);
        vocab.insert("h".to_string(), 4);
        vocab.insert("e".to_string(), 5);
        vocab.insert("l".to_string(), 6);
        vocab.insert("o".to_string(), 7);
        vocab.insert("w".to_string(), 8);
        vocab.insert("r".to_string(), 9);
        vocab.insert("d".to_string(), 10);
        vocab.insert(" ".to_string(), 11);
        vocab.insert("t".to_string(), 12);
        vocab.insert("s".to_string(), 13);
        CharTokenizer::new(vocab)
    }

    #[test]
    fn test_multimodal_config() {
        let config = MultimodalConfig::default();
        assert_eq!(config.max_text_length, Some(512));
        assert_eq!(config.max_image_patches, Some(196));
        assert!(config.include_special_tokens);
    }

    #[test]
    fn test_multimodal_tokenizer_creation() {
        let text_tokenizer = create_test_char_tokenizer();
        let multimodal_tokenizer = MultimodalTokenizer::from_text_tokenizer(text_tokenizer);

        assert!(multimodal_tokenizer.get_vocab().contains_key("[IMG]"));
        assert!(multimodal_tokenizer.get_vocab().contains_key("[AUD]"));
    }

    #[test]
    fn test_text_only_tokenization() {
        let text_tokenizer = create_test_char_tokenizer();
        let multimodal_tokenizer = MultimodalTokenizer::from_text_tokenizer(text_tokenizer);

        let input = MultimodalInput {
            text: Some("hello world".to_string()),
            image_patches: None,
            audio_frames: None,
            video_frames: None,
            table_data: None,
            graph_data: None,
            custom_data: None,
        };

        let result = multimodal_tokenizer.tokenize_multimodal(&input);
        assert!(result.is_ok());
        let tokenized = result.unwrap();
        assert!(!tokenized.input_ids.is_empty());
        assert!(tokenized.modality_boundaries.contains_key(&ModalityType::Text));
    }

    #[test]
    fn test_image_patch_creation() {
        let patches = MultimodalUtils::create_image_patches(224, 224, 16);
        assert_eq!(patches.len(), 14 * 14); // 196 patches

        let first_patch = &patches[0];
        assert_eq!(first_patch.x, 0);
        assert_eq!(first_patch.y, 0);
        assert_eq!(first_patch.width, 16);
        assert_eq!(first_patch.height, 16);
    }

    #[test]
    fn test_audio_frame_creation() {
        let frames = MultimodalUtils::create_audio_frames(44100.0, 1.0, 1024, 512);
        assert!(!frames.is_empty());

        let first_frame = &frames[0];
        assert_eq!(first_frame.timestamp, 0.0);
        assert_eq!(first_frame.samples.len(), 1024);
    }

    #[test]
    fn test_multimodal_input_with_images() {
        let text_tokenizer = create_test_char_tokenizer();
        let multimodal_tokenizer = MultimodalTokenizer::from_text_tokenizer(text_tokenizer);

        let patches = vec![ImagePatch {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
            pixels: vec![0.0; 16 * 16 * 3],
            embedding: Some(vec![1.0, 2.0, 3.0]),
        }];

        let input = MultimodalInput {
            text: Some("An image".to_string()),
            image_patches: Some(patches),
            audio_frames: None,
            video_frames: None,
            table_data: None,
            graph_data: None,
            custom_data: None,
        };

        let result = multimodal_tokenizer.tokenize_multimodal(&input);
        assert!(result.is_ok());
        let tokenized = result.unwrap();
        assert!(tokenized.modality_boundaries.contains_key(&ModalityType::Text));
        assert!(tokenized.modality_boundaries.contains_key(&ModalityType::Image));
    }

    #[test]
    fn test_table_tokenization() {
        let text_tokenizer = create_test_char_tokenizer();
        let multimodal_tokenizer = MultimodalTokenizer::from_text_tokenizer(text_tokenizer);

        let table = TableData {
            headers: vec!["Name".to_string(), "Age".to_string()],
            rows: vec![
                vec!["Alice".to_string(), "25".to_string()],
                vec!["Bob".to_string(), "30".to_string()],
            ],
            column_types: Some(vec!["string".to_string(), "int".to_string()]),
        };

        let input = MultimodalInput {
            text: None,
            image_patches: None,
            audio_frames: None,
            video_frames: None,
            table_data: Some(table),
            graph_data: None,
            custom_data: None,
        };

        let result = multimodal_tokenizer.tokenize_multimodal(&input);
        assert!(result.is_ok());
        let tokenized = result.unwrap();
        assert!(tokenized.modality_boundaries.contains_key(&ModalityType::Table));
    }

    #[test]
    fn test_fusion_strategies() {
        let text_tokenizer = create_test_char_tokenizer();
        let mut config = MultimodalConfig::default();
        config.fusion_strategy = FusionStrategy::Interleaved;
        let multimodal_tokenizer = MultimodalTokenizer::new(text_tokenizer, config);

        let tokens = vec![
            MultimodalToken {
                token_id: 1,
                modality: ModalityType::Text,
                modality_position: 0,
                global_position: 0,
                metadata: None,
            },
            MultimodalToken {
                token_id: 2,
                modality: ModalityType::Image,
                modality_position: 0,
                global_position: 1,
                metadata: None,
            },
        ];

        let result = multimodal_tokenizer.apply_fusion_strategy(&tokens);
        assert!(result.is_ok());
    }

    #[test]
    fn test_convert_to_multimodal() {
        let tokenized = TokenizedInput {
            input_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let multimodal = MultimodalUtils::convert_to_multimodal(tokenized, ModalityType::Text);
        assert_eq!(multimodal.input_ids, vec![1, 2, 3]);
        assert_eq!(multimodal.modality_tokens.len(), 3);
        assert!(multimodal.modality_boundaries.contains_key(&ModalityType::Text));
    }

    #[test]
    fn test_cross_modal_attention() {
        let tokens = vec![
            MultimodalToken {
                token_id: 1,
                modality: ModalityType::Text,
                modality_position: 0,
                global_position: 0,
                metadata: None,
            },
            MultimodalToken {
                token_id: 2,
                modality: ModalityType::Image,
                modality_position: 0,
                global_position: 1,
                metadata: None,
            },
        ];

        let attention = MultimodalUtils::calculate_cross_modal_attention(
            &tokens,
            &ModalityType::Text,
            &ModalityType::Image,
        );

        assert_eq!(attention.len(), 1); // 1 text token
        assert_eq!(attention[0].len(), 1); // 1 image token
        assert_eq!(attention[0][0], 1.0);
    }
}
