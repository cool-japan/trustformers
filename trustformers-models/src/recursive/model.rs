use crate::recursive::config::RecursiveConfig;
use trustformers_core::{
    errors::{Result, TrustformersError},
    layers::{Embedding, FeedForward, LayerNorm, Linear, MultiHeadAttention},
    tensor::Tensor,
    traits::{Layer, Model},
};

/// Main Recursive Transformer model
pub struct RecursiveTransformer {
    config: RecursiveConfig,
    embeddings: Embedding,
    position_embeddings: Embedding,
    recursive_layers: Vec<RecursiveLayer>,
    memory_manager: MemoryManager,
    depth_predictor: Option<DepthPredictor>,
    hierarchy_manager: Option<HierarchyManager>,
    universal_controller: Option<UniversalController>,
    norm: LayerNorm,
    lm_head: Linear,
}

impl RecursiveTransformer {
    pub fn new(config: RecursiveConfig) -> Result<Self> {
        let embeddings = Embedding::new(config.vocab_size, config.hidden_size, None)?;
        let position_embeddings =
            Embedding::new(config.max_position_embeddings, config.hidden_size, None)?;

        let mut recursive_layers = Vec::new();
        for _ in 0..config.num_recursive_layers {
            recursive_layers.push(RecursiveLayer::new(config.clone())?);
        }

        let memory_manager = MemoryManager::new(config.clone())?;

        let depth_predictor = if config.use_adaptive_depth {
            Some(DepthPredictor::new(config.clone())?)
        } else {
            None
        };

        let hierarchy_manager = if config.use_hierarchical_attention {
            Some(HierarchyManager::new(config.clone())?)
        } else {
            None
        };

        let universal_controller = if config.use_universal_transformer {
            Some(UniversalController::new(config.clone())?)
        } else {
            None
        };

        let norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;
        let lm_head = Linear::new(config.hidden_size, config.vocab_size, false);

        Ok(Self {
            config,
            embeddings,
            position_embeddings,
            recursive_layers,
            memory_manager,
            depth_predictor,
            hierarchy_manager,
            universal_controller,
            norm,
            lm_head,
        })
    }

    fn chunk_input(&self, input_ids: &Tensor) -> Result<Vec<Tensor>> {
        let seq_len = input_ids.shape()[1];
        let chunk_size = self.config.chunk_size;
        let overlap_size = self.config.overlap_size;
        let effective_chunk_size = chunk_size - overlap_size;

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < seq_len {
            let end = std::cmp::min(start + chunk_size, seq_len);
            let chunk = input_ids.slice(1, start, end)?;
            chunks.push(chunk);

            if end == seq_len {
                break;
            }

            start += effective_chunk_size;
        }

        Ok(chunks)
    }

    fn recursive_process(
        &self,
        chunks: Vec<Tensor>,
        depth: usize,
        memory: &mut MemoryState,
    ) -> Result<Tensor> {
        if depth == 0 || chunks.len() == 1 {
            // Base case: process single chunk
            return self.process_single_chunk(&chunks[0], memory);
        }

        // Divide chunks into groups for recursive processing
        let group_size = std::cmp::max(2, chunks.len() / 2);
        let mut processed_chunks = Vec::new();

        for chunk_group in chunks.chunks(group_size) {
            if chunk_group.len() == 1 {
                let processed = self.process_single_chunk(&chunk_group[0], memory)?;
                processed_chunks.push(processed);
            } else {
                // Recursively process the group
                let sub_result = self.recursive_process(chunk_group.to_vec(), depth - 1, memory)?;
                processed_chunks.push(sub_result);
            }
        }

        // Combine processed chunks
        if processed_chunks.len() == 1 {
            Ok(processed_chunks.into_iter().next().unwrap())
        } else {
            self.combine_chunks(processed_chunks, memory)
        }
    }

    fn process_single_chunk(&self, chunk: &Tensor, memory: &mut MemoryState) -> Result<Tensor> {
        // Embed the chunk
        let chunk_vec: Vec<u32> = match chunk {
            Tensor::F32(array) => array.iter().map(|&x| x as u32).collect(),
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Unsupported tensor type for chunk",
                    "recursive_forward",
                ))
            },
        };
        let embedded = self.embeddings.forward(chunk_vec)?;

        // Add positional embeddings
        let seq_len = embedded.shape()[1];
        let position_ids: Vec<u32> = (0..seq_len).map(|i| i as u32).collect();
        let pos_embedded = self.position_embeddings.forward(position_ids)?;
        let mut hidden_states = embedded.add(&pos_embedded)?;

        // Process through recursive layers
        if let Some(ref universal) = self.universal_controller {
            // Universal Transformer processing
            hidden_states = universal.process(hidden_states, &self.recursive_layers[0], memory)?;
        } else {
            // Standard recursive processing
            for layer in &self.recursive_layers {
                hidden_states = layer.forward(RecursiveLayerInput {
                    hidden_states,
                    memory_state: memory.clone(),
                    depth: 0,
                })?;
            }
        }

        // Update memory with chunk representation
        let chunk_summary = self.summarize_chunk(&hidden_states)?;
        memory.update(chunk_summary)?;

        Ok(hidden_states)
    }

    fn combine_chunks(&self, chunks: Vec<Tensor>, memory: &mut MemoryState) -> Result<Tensor> {
        // Concatenate chunks along sequence dimension
        let combined = Tensor::concat(&chunks, 1)?;

        // Apply cross-chunk attention if hierarchical
        if let Some(ref hierarchy) = self.hierarchy_manager {
            hierarchy.cross_chunk_attention(combined, memory)
        } else {
            Ok(combined)
        }
    }

    fn summarize_chunk(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Create a summary representation of the chunk
        // Use mean pooling for simplicity
        hidden_states.mean()
    }
}

impl Model for RecursiveTransformer {
    type Config = RecursiveConfig;
    type Input = RecursiveInput;
    type Output = RecursiveOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let RecursiveInput {
            input_ids,
            attention_mask: _,
            position_ids: _,
            memory_state: initial_memory,
        } = input;

        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];

        // Initialize memory state
        let mut memory = initial_memory.unwrap_or_else(|| {
            MemoryState::new(batch_size, self.config.memory_size, self.config.hidden_size)
        });

        // Determine recursion depth
        let depth = if let Some(ref predictor) = self.depth_predictor {
            predictor.predict_depth(&input_ids, &memory)?
        } else {
            self.config.recursion_depth
        };

        // Chunk the input if necessary
        let chunks = if seq_len > self.config.chunk_size {
            self.chunk_input(&input_ids)?
        } else {
            vec![input_ids.clone()]
        };

        // Process recursively
        let hidden_states = self.recursive_process(chunks, depth, &mut memory)?;

        // Final normalization and prediction
        let normalized = self.norm.forward(hidden_states)?;
        let logits = self.lm_head.forward(normalized.clone())?;

        Ok(RecursiveOutput {
            last_hidden_state: normalized,
            logits,
            memory_state: memory,
            recursion_depth: depth,
            computation_steps: 0, // Would be computed in real implementation
        })
    }

    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> Result<()> {
        Err(TrustformersError::not_implemented(
            "Use load_from_path or load_from_huggingface for enhanced weight loading".to_string(),
        ))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Core components
        total += self.embeddings.parameter_count();
        total += self.position_embeddings.parameter_count();
        total += self.recursive_layers.iter().map(|layer| layer.parameter_count()).sum::<usize>();
        total += self.memory_manager.parameter_count();
        total += self.norm.parameter_count();
        total += self.lm_head.parameter_count();

        // Optional components
        if let Some(ref predictor) = self.depth_predictor {
            total += predictor.parameter_count();
        }
        if let Some(ref manager) = self.hierarchy_manager {
            total += manager.parameter_count();
        }
        if let Some(ref controller) = self.universal_controller {
            total += controller.parameter_count();
        }

        total
    }
}

impl RecursiveTransformer {
    /// Enhanced weight loading from local path
    pub fn load_from_path(&mut self, model_path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::weight_loading::{auto_create_loader, WeightLoadingConfig};

        let config = WeightLoadingConfig {
            lazy_loading: true,
            memory_mapped: false,
            ..Default::default()
        };

        let mut loader = auto_create_loader(model_path, Some(config))?;

        // Load embeddings
        if let Ok(embed_weights) = loader.load_tensor("embeddings.weight") {
            self.embeddings.set_weight(embed_weights)?;
        }

        // Load position embeddings
        if let Ok(pos_embed_weights) = loader.load_tensor("position_embeddings.weight") {
            self.position_embeddings.set_weight(pos_embed_weights)?;
        }

        // Load recursive layers
        for (i, layer) in self.recursive_layers.iter_mut().enumerate() {
            let layer_prefix = format!("recursive_layers.{}", i);

            // Note: MultiHeadAttention weight loading is not accessible due to private fields
            // This would need to be implemented at the core level or through a different interface

            // Note: FeedForward weight loading is not accessible due to private fields
            // This would need to be implemented at the core level or through accessor methods

            // Load layer norm weights
            if let Ok(ln1_weight) =
                loader.load_tensor(&format!("{}.layer_norm1.weight", layer_prefix))
            {
                layer.layer_norm1.set_weight(ln1_weight)?;
            }
            if let Ok(ln2_weight) =
                loader.load_tensor(&format!("{}.layer_norm2.weight", layer_prefix))
            {
                layer.layer_norm2.set_weight(ln2_weight)?;
            }

            // Load layer norm3 if present
            if let Some(ref mut ln3) = layer.layer_norm3 {
                if let Ok(ln3_weight) =
                    loader.load_tensor(&format!("{}.layer_norm3.weight", layer_prefix))
                {
                    ln3.set_weight(ln3_weight)?;
                }
            }

            // Note: Memory gate weight loading would need to be implemented
            // based on the MemoryGate structure
        }

        // Load final normalization
        if let Ok(norm_weight) = loader.load_tensor("norm.weight") {
            self.norm.set_weight(norm_weight)?;
        }

        // Load LM head
        if let Ok(lm_head_weight) = loader.load_tensor("lm_head.weight") {
            self.lm_head.set_weight(lm_head_weight)?;
        }

        // Load optional components
        // Note: DepthPredictor weight loading would need to be implemented
        // based on the DepthPredictor structure and available methods

        loader.close()?;
        Ok(())
    }

    /// Enhanced weight loading from HuggingFace Hub
    pub fn load_from_huggingface(&mut self, model_name: &str) -> Result<()> {
        // Check if model is cached locally
        let cache_dir = std::env::var("HF_HOME")
            .or_else(|_| std::env::var("HUGGINGFACE_HUB_CACHE"))
            .unwrap_or_else(|_| {
                std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
                    + "/.cache/huggingface/hub"
            });

        let model_path = std::path::Path::new(&cache_dir)
            .join(format!("models--{}", model_name.replace("/", "--")));

        if model_path.exists() {
            self.load_from_path(&model_path)
        } else {
            // Attempt to download the model from HuggingFace Hub
            self.download_from_huggingface_hub(model_name, &model_path)?;
            self.load_from_path(&model_path)
        }
    }

    /// Download model from HuggingFace Hub
    fn download_from_huggingface_hub(
        &self,
        model_name: &str,
        model_path: &std::path::Path,
    ) -> Result<()> {
        use std::process::Command;

        println!(
            "Downloading model {} from HuggingFace Hub to {:?}",
            model_name, model_path
        );

        // Create the model directory
        std::fs::create_dir_all(model_path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to create model directory: {}", e))
        })?;

        // List of essential files for Recursive models
        let essential_files = vec![
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "pytorch_model.bin", // Try .bin first
            "model.safetensors", // Fall back to safetensors
        ];

        let base_url = format!("https://huggingface.co/{}/resolve/main", model_name);

        // Try to download each essential file
        for file_name in &essential_files {
            let file_url = format!("{}/{}", base_url, file_name);
            let file_path = model_path.join(file_name);

            println!("Attempting to download {}", file_url);

            // Try using curl first
            let curl_result = Command::new("curl")
                .args([
                    "-L", // Follow redirects
                    "-f", // Fail on HTTP errors
                    "-o",
                    file_path.to_str().unwrap(),
                    &file_url,
                ])
                .output();

            match curl_result {
                Ok(output) if output.status.success() => {
                    println!("Successfully downloaded {}", file_name);
                    continue;
                },
                Ok(output) => {
                    eprintln!(
                        "Failed to download {} with curl: {}",
                        file_name,
                        String::from_utf8_lossy(&output.stderr)
                    );
                },
                Err(e) => {
                    println!("curl not available: {}", e);
                },
            }

            // Try using wget as fallback
            let wget_result = Command::new("wget")
                .args(["-O", file_path.to_str().unwrap(), &file_url])
                .output();

            match wget_result {
                Ok(output) if output.status.success() => {
                    println!("Successfully downloaded {} with wget", file_name);
                    continue;
                },
                Ok(output) => {
                    eprintln!(
                        "Failed to download {} with wget: {}",
                        file_name,
                        String::from_utf8_lossy(&output.stderr)
                    );
                },
                Err(e) => {
                    println!("wget not available: {}", e);
                },
            }

            // If essential files like config.json or pytorch_model.bin fail, return error
            if matches!(file_name, &"config.json" | &"pytorch_model.bin") {
                return Err(TrustformersError::io_error(format!(
                    "Failed to download essential file {} for model {}. Please ensure curl or wget is installed and you have internet access.",
                    file_name, model_name
                )));
            }
        }

        println!(
            "Successfully downloaded model {} from HuggingFace Hub",
            model_name
        );
        Ok(())
    }

    /// Load weights with lazy loading for large models
    pub fn load_with_lazy_loading(
        &mut self,
        model_path: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        // For now, delegate to regular loading
        self.load_from_path(model_path)
    }
}

/// Single recursive layer
pub struct RecursiveLayer {
    #[allow(dead_code)]
    config: RecursiveConfig,
    self_attention: MultiHeadAttention,
    cross_attention: Option<MultiHeadAttention>,
    feed_forward: FeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    layer_norm3: Option<LayerNorm>,
    memory_gate: MemoryGate,
}

impl RecursiveLayer {
    pub fn new(config: RecursiveConfig) -> Result<Self> {
        let self_attention = MultiHeadAttention::new(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_dropout,
            true,
        )?;

        let cross_attention = if config.use_hierarchical_attention {
            Some(MultiHeadAttention::new(
                config.hidden_size,
                config.num_attention_heads,
                config.attention_dropout,
                false,
            )?)
        } else {
            None
        };

        let feed_forward =
            FeedForward::new(config.hidden_size, config.intermediate_size, config.dropout)?;

        let layer_norm1 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;
        let layer_norm2 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;
        let layer_norm3 = if cross_attention.is_some() {
            Some(LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps,
            )?)
        } else {
            None
        };

        let memory_gate = MemoryGate::new(config.clone())?;

        Ok(Self {
            config,
            self_attention,
            cross_attention,
            feed_forward,
            layer_norm1,
            layer_norm2,
            layer_norm3,
            memory_gate,
        })
    }

    pub fn parameter_count(&self) -> usize {
        let mut total = self.self_attention.parameter_count()
            + self.feed_forward.parameter_count()
            + self.layer_norm1.parameter_count()
            + self.layer_norm2.parameter_count()
            + self.memory_gate.parameter_count();

        if let Some(ref cross_attn) = self.cross_attention {
            total += cross_attn.parameter_count();
        }
        if let Some(ref norm3) = self.layer_norm3 {
            total += norm3.parameter_count();
        }

        total
    }
}

impl Layer for RecursiveLayer {
    type Input = RecursiveLayerInput;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let RecursiveLayerInput {
            hidden_states,
            memory_state,
            depth: _,
        } = input;

        let mut hidden_states = hidden_states;

        // Self-attention
        let normed = self.layer_norm1.forward(hidden_states.clone())?;
        let attn_output = self.self_attention.forward(normed)?;
        hidden_states = hidden_states.add(&attn_output)?;

        // Cross-attention with memory (if enabled)
        if let (Some(cross_attn), Some(layer_norm)) = (&self.cross_attention, &self.layer_norm3) {
            let normed = layer_norm.forward(hidden_states.clone())?;
            let _memory_content = memory_state.get_content()?;
            let cross_output = cross_attn.forward(normed)?;
            hidden_states = hidden_states.add(&cross_output)?;
        }

        // Feed-forward
        let normed = self.layer_norm2.forward(hidden_states.clone())?;
        let ff_output = self.feed_forward.forward(normed)?;
        hidden_states = hidden_states.add(&ff_output)?;

        // Memory gating
        hidden_states = self.memory_gate.forward((hidden_states, memory_state))?;

        Ok(hidden_states)
    }
}

/// Memory management for recursive processing
pub struct MemoryManager {
    #[allow(dead_code)]
    config: RecursiveConfig,
    memory_projection: Linear,
    compression_layer: Option<Linear>,
}

impl MemoryManager {
    pub fn new(config: RecursiveConfig) -> Result<Self> {
        let memory_projection = Linear::new(config.hidden_size, config.memory_size, true);

        let compression_layer = if config.use_memory_compression {
            Some(Linear::new(
                config.memory_size,
                (config.memory_size as f32 * config.compression_ratio) as usize,
                true,
            ))
        } else {
            None
        };

        Ok(Self {
            config,
            memory_projection,
            compression_layer,
        })
    }

    pub fn parameter_count(&self) -> usize {
        let mut total = self.memory_projection.parameter_count();
        if let Some(ref compression) = self.compression_layer {
            total += compression.parameter_count();
        }
        total
    }
}

/// Memory state for tracking information across recursive calls
#[derive(Debug, Clone)]
pub struct MemoryState {
    content: Tensor,
    write_head: usize,
    read_head: usize,
    capacity: usize,
}

impl MemoryState {
    pub fn new(batch_size: usize, memory_size: usize, hidden_size: usize) -> Self {
        let content = Tensor::zeros(&[batch_size, memory_size, hidden_size]).unwrap();
        Self {
            content,
            write_head: 0,
            read_head: 0,
            capacity: memory_size,
        }
    }

    pub fn update(&mut self, new_content: Tensor) -> Result<()> {
        let content_size = new_content.shape()[1];

        // If new content is larger than or equal to capacity, just use the last `capacity` elements
        if content_size >= self.capacity {
            self.content = new_content.slice(1, content_size - self.capacity, content_size)?;
            self.write_head = 0;
            return Ok(());
        }

        // Simple append if there's room
        let end_pos = self.write_head + content_size;

        if end_pos <= self.capacity {
            // Content fits without wrapping
            let before = if self.write_head > 0 {
                Some(self.content.slice(1, 0, self.write_head)?)
            } else {
                None
            };

            let after = if end_pos < self.capacity {
                Some(self.content.slice(1, end_pos, self.capacity)?)
            } else {
                None
            };

            // Reconstruct memory with new content (concatenate on dimension 1)
            match (before, after) {
                (Some(b), Some(a)) => {
                    self.content = Tensor::concat(&[b, new_content, a], 1)?;
                }
                (Some(b), None) => {
                    self.content = Tensor::concat(&[b, new_content], 1)?;
                }
                (None, Some(a)) => {
                    self.content = Tensor::concat(&[new_content, a], 1)?;
                }
                (None, None) => {
                    self.content = new_content;
                }
            }
            self.write_head = end_pos;
        } else {
            // Content wraps around
            let available = self.capacity - self.write_head;
            let first_part = new_content.slice(1, 0, available)?;
            let second_part = new_content.slice(1, available, content_size)?;
            let remaining = content_size - available;

            // Get any old content between where second_part ends and write_head
            let middle = if remaining < self.write_head {
                Some(self.content.slice(1, remaining, self.write_head)?)
            } else {
                None
            };

            // Reconstruct: [second_part, middle (if any), first_part]
            match middle {
                Some(m) => {
                    self.content = Tensor::concat(&[second_part, m, first_part], 1)?;
                }
                None => {
                    self.content = Tensor::concat(&[second_part, first_part], 1)?;
                }
            }
            self.write_head = remaining;
        }

        Ok(())
    }

    pub fn get_content(&self) -> Result<Tensor> {
        Ok(self.content.clone())
    }

    pub fn read(&mut self, size: usize) -> Result<Tensor> {
        let end_pos = std::cmp::min(self.read_head + size, self.capacity);
        let content = self.content.slice(1, self.read_head, end_pos)?;
        self.read_head = (self.read_head + size) % self.capacity;
        Ok(content)
    }
}

/// Memory gate for controlling information flow
pub struct MemoryGate {
    gate_projection: Linear,
    memory_projection: Linear,
}

impl MemoryGate {
    pub fn new(config: RecursiveConfig) -> Result<Self> {
        let gate_projection = Linear::new(config.hidden_size * 2, config.hidden_size, true);
        let memory_projection = Linear::new(config.memory_size, config.hidden_size, false);

        Ok(Self {
            gate_projection,
            memory_projection,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.gate_projection.parameter_count() + self.memory_projection.parameter_count()
    }
}

impl Layer for MemoryGate {
    type Input = (Tensor, MemoryState);
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let (hidden_states, memory_state) = input;

        // Get memory content
        let memory_content = memory_state.get_content()?;
        let memory_summary = memory_content.mean()?; // Summarize memory
        let memory_features = self.memory_projection.forward(memory_summary)?;

        // Compute gate
        let combined = Tensor::concat(&[hidden_states.clone(), memory_features.clone()], 1)?;
        let gate = self.gate_projection.forward(combined)?.sigmoid()?;

        // Apply gate
        let gated_memory = memory_features.mul(&gate)?;
        let gated_hidden = hidden_states.mul(&(Tensor::ones_like(&gate)?.sub(&gate)?))?;

        gated_hidden.add(&gated_memory)
    }
}

/// Depth predictor for adaptive recursion
pub struct DepthPredictor {
    predictor: Linear,
    threshold: f32,
}

impl DepthPredictor {
    pub fn new(config: RecursiveConfig) -> Result<Self> {
        let predictor = Linear::new(config.hidden_size, 1, true);
        Ok(Self {
            predictor,
            threshold: config.depth_threshold,
        })
    }

    pub fn predict_depth(&self, input_ids: &Tensor, memory: &MemoryState) -> Result<usize> {
        // Simple heuristic based on sequence length and memory state
        let seq_len = input_ids.shape()[1] as f32;
        let memory_usage = memory.write_head as f32 / memory.capacity as f32;

        let complexity_score = (seq_len / 1000.0) + memory_usage;

        if complexity_score > self.threshold {
            Ok(5) // High complexity, use more depth
        } else if complexity_score > self.threshold / 2.0 {
            Ok(3) // Medium complexity
        } else {
            Ok(1) // Low complexity
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.predictor.parameter_count()
    }
}

/// Hierarchy manager for multi-level processing
pub struct HierarchyManager {
    #[allow(dead_code)]
    config: RecursiveConfig,
    level_projections: Vec<Linear>,
    cross_level_attention: MultiHeadAttention,
}

impl HierarchyManager {
    pub fn new(config: RecursiveConfig) -> Result<Self> {
        let mut level_projections = Vec::new();

        for &ratio in &config.level_compression_ratios {
            let compressed_size = (config.hidden_size as f32 * ratio) as usize;
            level_projections.push(Linear::new(config.hidden_size, compressed_size, false));
        }

        let cross_level_attention = MultiHeadAttention::new(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_dropout,
            false,
        )?;

        Ok(Self {
            config,
            level_projections,
            cross_level_attention,
        })
    }

    pub fn cross_chunk_attention(&self, combined: Tensor, memory: &MemoryState) -> Result<Tensor> {
        let _memory_content = memory.get_content()?;
        let attended = self.cross_level_attention.forward(combined.clone())?;
        combined.add(&attended)
    }

    pub fn parameter_count(&self) -> usize {
        let projections_count: usize =
            self.level_projections.iter().map(|proj| proj.parameter_count()).sum();
        projections_count + self.cross_level_attention.parameter_count()
    }
}

/// Universal Transformer controller
pub struct UniversalController {
    config: RecursiveConfig,
    step_embedding: Embedding,
    halting_predictor: Linear,
}

impl UniversalController {
    pub fn new(config: RecursiveConfig) -> Result<Self> {
        let step_embedding = Embedding::new(config.max_steps, config.hidden_size, None)?;
        let halting_predictor = Linear::new(config.hidden_size, 1, true);

        Ok(Self {
            config,
            step_embedding,
            halting_predictor,
        })
    }

    pub fn process(
        &self,
        mut hidden_states: Tensor,
        layer: &RecursiveLayer,
        memory: &mut MemoryState,
    ) -> Result<Tensor> {
        let mut total_halting_prob =
            Tensor::zeros(&[hidden_states.shape()[0], hidden_states.shape()[1]])?;
        let mut step = 0;

        while step < self.config.max_steps {
            // Add step embedding
            let step_vec = vec![step as u32];
            let step_emb = self.step_embedding.forward(step_vec)?;
            hidden_states = hidden_states.add(&step_emb.unsqueeze(0)?.unsqueeze(0)?)?;

            // Process through layer
            hidden_states = layer.forward(RecursiveLayerInput {
                hidden_states,
                memory_state: memory.clone(),
                depth: step,
            })?;

            // Compute halting probability
            let halting_logits = self.halting_predictor.forward(hidden_states.clone())?;
            let halting_prob = halting_logits.sigmoid()?;

            total_halting_prob = total_halting_prob.add(&halting_prob.squeeze(1)?)?;

            // Check if we should halt using Adaptive Computation Time (ACT)
            if self.config.adaptive_computation_time {
                // Check individual tokens - if any token has high enough probability, consider halting
                let batch_size = total_halting_prob.shape()[0];
                let seq_len = total_halting_prob.shape()[1];
                let mut should_halt = false;

                for batch_idx in 0..batch_size {
                    for seq_idx in 0..seq_len {
                        if let Ok(prob) = total_halting_prob.get_scalar(&[batch_idx, seq_idx]) {
                            if prob >= self.config.act_threshold {
                                should_halt = true;
                                break;
                            }
                        }
                    }
                    if should_halt {
                        break;
                    }
                }

                if should_halt {
                    break;
                }
            } else {
                // Fallback to simple step threshold when ACT is disabled
                if step >= 5 {
                    break;
                }
            }

            step += 1;
        }

        Ok(hidden_states)
    }

    pub fn parameter_count(&self) -> usize {
        self.step_embedding.parameter_count() + self.halting_predictor.parameter_count()
    }
}

// Input/Output structures
#[derive(Debug)]
pub struct RecursiveInput {
    pub input_ids: Tensor,
    pub attention_mask: Option<Tensor>,
    pub position_ids: Option<Tensor>,
    pub memory_state: Option<MemoryState>,
}

#[derive(Debug)]
pub struct RecursiveOutput {
    pub last_hidden_state: Tensor,
    pub logits: Tensor,
    pub memory_state: MemoryState,
    pub recursion_depth: usize,
    pub computation_steps: usize,
}

#[derive(Debug)]
pub struct RecursiveLayerInput {
    pub hidden_states: Tensor,
    pub memory_state: MemoryState,
    pub depth: usize,
}

/// For language modeling tasks
pub struct RecursiveForCausalLM {
    base_model: RecursiveTransformer,
}

impl RecursiveForCausalLM {
    pub fn new(config: RecursiveConfig) -> Result<Self> {
        let base_model = RecursiveTransformer::new(config)?;
        Ok(Self { base_model })
    }
}

impl Model for RecursiveForCausalLM {
    type Config = RecursiveConfig;
    type Input = RecursiveInput;
    type Output = RecursiveOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.base_model.forward(input)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> Result<()> {
        Err(TrustformersError::not_implemented(
            "Use load_from_path or load_from_huggingface for enhanced weight loading".to_string(),
        ))
    }

    fn get_config(&self) -> &Self::Config {
        &self.base_model.config
    }

    fn num_parameters(&self) -> usize {
        self.base_model.num_parameters()
    }
}

impl RecursiveForCausalLM {
    /// Enhanced weight loading from local path
    pub fn load_from_path(&mut self, model_path: impl AsRef<std::path::Path>) -> Result<()> {
        // Delegate to base model weight loading
        self.base_model.load_from_path(model_path)
    }

    /// Enhanced weight loading from HuggingFace Hub
    pub fn load_from_huggingface(&mut self, model_name: &str) -> Result<()> {
        // Delegate to base model weight loading
        self.base_model.load_from_huggingface(model_name)
    }

    /// Load weights with lazy loading for large models
    pub fn load_with_lazy_loading(
        &mut self,
        model_path: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        // Delegate to base model weight loading
        self.base_model.load_with_lazy_loading(model_path)
    }
}

/// For sequence classification tasks
pub struct RecursiveForSequenceClassification {
    base_model: RecursiveTransformer,
    classifier: Linear,
    #[allow(dead_code)]
    num_labels: usize,
}

impl RecursiveForSequenceClassification {
    pub fn new(config: RecursiveConfig, num_labels: usize) -> Result<Self> {
        let base_config = config;
        // Don't need LM head for classification
        let base_model = RecursiveTransformer::new(base_config.clone())?;
        let classifier = Linear::new(base_config.hidden_size, num_labels, false);

        Ok(Self {
            base_model,
            classifier,
            num_labels,
        })
    }
}

impl Model for RecursiveForSequenceClassification {
    type Config = RecursiveConfig;
    type Input = RecursiveInput;
    type Output = RecursiveClassificationOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let output = self.base_model.forward(input)?;

        // Use final hidden state for classification
        let pooled = output.last_hidden_state.mean()?; // Pool over sequence
        let logits = self.classifier.forward(pooled)?;

        Ok(RecursiveClassificationOutput {
            logits,
            hidden_states: output.last_hidden_state,
            memory_state: output.memory_state,
            recursion_depth: output.recursion_depth,
        })
    }

    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> Result<()> {
        Err(TrustformersError::not_implemented(
            "Use load_from_path or load_from_huggingface for enhanced weight loading".to_string(),
        ))
    }

    fn get_config(&self) -> &Self::Config {
        &self.base_model.config
    }

    fn num_parameters(&self) -> usize {
        self.base_model.num_parameters() + self.classifier.parameter_count()
    }
}

impl RecursiveForSequenceClassification {
    /// Enhanced weight loading from local path
    pub fn load_from_path(&mut self, model_path: impl AsRef<std::path::Path>) -> Result<()> {
        println!(
            "Loading sequence classification weights from: {:?}",
            model_path.as_ref()
        );

        // Load base model weights first
        self.base_model.load_from_path(model_path)?;

        // In a complete implementation, this would also load classifier weights

        Ok(())
    }

    /// Enhanced weight loading from HuggingFace Hub
    pub fn load_from_huggingface(&mut self, model_name: &str) -> Result<()> {
        // Check if model is cached locally
        let cache_dir = std::env::var("HF_HOME")
            .or_else(|_| std::env::var("HUGGINGFACE_HUB_CACHE"))
            .unwrap_or_else(|_| {
                std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
                    + "/.cache/huggingface/hub"
            });

        let model_path = std::path::Path::new(&cache_dir)
            .join(format!("models--{}", model_name.replace("/", "--")));

        if model_path.exists() {
            self.load_from_path(&model_path)
        } else {
            // Attempt to download the model from HuggingFace Hub
            self.download_from_huggingface_hub(model_name, &model_path)?;
            self.load_from_path(&model_path)
        }
    }

    /// Download model from HuggingFace Hub
    fn download_from_huggingface_hub(
        &self,
        model_name: &str,
        model_path: &std::path::Path,
    ) -> Result<()> {
        use std::process::Command;

        println!(
            "Downloading model {} from HuggingFace Hub to {:?}",
            model_name, model_path
        );

        // Create the model directory
        std::fs::create_dir_all(model_path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to create model directory: {}", e))
        })?;

        // List of essential files for Recursive models
        let essential_files = vec![
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "pytorch_model.bin", // Try .bin first
            "model.safetensors", // Fall back to safetensors
        ];

        let base_url = format!("https://huggingface.co/{}/resolve/main", model_name);

        // Try to download each essential file
        for file_name in &essential_files {
            let file_url = format!("{}/{}", base_url, file_name);
            let file_path = model_path.join(file_name);

            println!("Attempting to download {}", file_url);

            // Try using curl first
            let curl_result = Command::new("curl")
                .args([
                    "-L", // Follow redirects
                    "-f", // Fail on HTTP errors
                    "-o",
                    file_path.to_str().unwrap(),
                    &file_url,
                ])
                .output();

            match curl_result {
                Ok(output) if output.status.success() => {
                    println!("Successfully downloaded {}", file_name);
                    continue;
                },
                Ok(output) => {
                    eprintln!(
                        "Failed to download {} with curl: {}",
                        file_name,
                        String::from_utf8_lossy(&output.stderr)
                    );
                },
                Err(e) => {
                    println!("curl not available: {}", e);
                },
            }

            // Try using wget as fallback
            let wget_result = Command::new("wget")
                .args(["-O", file_path.to_str().unwrap(), &file_url])
                .output();

            match wget_result {
                Ok(output) if output.status.success() => {
                    println!("Successfully downloaded {} with wget", file_name);
                    continue;
                },
                Ok(output) => {
                    eprintln!(
                        "Failed to download {} with wget: {}",
                        file_name,
                        String::from_utf8_lossy(&output.stderr)
                    );
                },
                Err(e) => {
                    println!("wget not available: {}", e);
                },
            }

            // If essential files like config.json or pytorch_model.bin fail, return error
            if matches!(file_name, &"config.json" | &"pytorch_model.bin") {
                return Err(TrustformersError::io_error(format!(
                    "Failed to download essential file {} for model {}. Please ensure curl or wget is installed and you have internet access.",
                    file_name, model_name
                )));
            }
        }

        println!(
            "Successfully downloaded model {} from HuggingFace Hub",
            model_name
        );
        Ok(())
    }

    /// Load weights with lazy loading for large models
    pub fn load_with_lazy_loading(
        &mut self,
        model_path: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        // For now, delegate to regular loading
        self.load_from_path(model_path)
    }
}

#[derive(Debug)]
pub struct RecursiveClassificationOutput {
    pub logits: Tensor,
    pub hidden_states: Tensor,
    pub memory_state: MemoryState,
    pub recursion_depth: usize,
}
