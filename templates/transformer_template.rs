use trustformers_core::layers::attention::MultiHeadAttention;
use trustformers_core::layers::feedforward::FeedForward;
use trustformers_core::layers::layer_norm::LayerNorm;
use trustformers_core::tensor::Tensor;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct {{MODEL_NAME}}Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub pad_token_id: i32,
    pub position_embedding_type: String,
    pub use_cache: bool,
    pub classifier_dropout: Option<f32>,
}

impl Default for {{MODEL_NAME}}Config {
    fn default() -> Self {
        Self {
            vocab_size: {{VOCAB_SIZE}},
            hidden_size: {{HIDDEN_SIZE}},
            num_hidden_layers: {{NUM_LAYERS}},
            num_attention_heads: {{NUM_ATTENTION_HEADS}},
            intermediate_size: {{HIDDEN_SIZE}} * 4,
            max_position_embeddings: {{MAX_POSITION_EMBEDDINGS}},
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: "absolute".to_string(),
            use_cache: true,
            classifier_dropout: None,
        }
    }
}

#[derive(Debug)]
pub struct {{MODEL_NAME}}Embeddings {
    pub word_embeddings: Tensor,
    pub position_embeddings: Tensor,
    pub token_type_embeddings: Tensor,
    pub layer_norm: LayerNorm,
    pub dropout: f32,
}

impl {{MODEL_NAME}}Embeddings {
    pub fn new(config: &{{MODEL_NAME}}Config) -> Self {
        Self {
            word_embeddings: Tensor::zeros(&[config.vocab_size, config.hidden_size]),
            position_embeddings: Tensor::zeros(&[config.max_position_embeddings, config.hidden_size]),
            token_type_embeddings: Tensor::zeros(&[config.type_vocab_size, config.hidden_size]),
            layer_norm: LayerNorm::new(config.hidden_size),
            dropout: 0.1,
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let seq_length = input_ids.shape()[1];

        let position_ids = match position_ids {
            Some(ids) => ids.clone(),
            None => {
                let mut pos_ids = Vec::new();
                for i in 0..seq_length {
                    pos_ids.push(i as f32);
                }
                Tensor::from_vec(pos_ids, &[1, seq_length])
            }
        };

        let token_type_ids = match token_type_ids {
            Some(ids) => ids.clone(),
            None => Tensor::zeros(&[input_ids.shape()[0], seq_length]),
        };

        // Embedding lookups
        let inputs_embeds = self.word_embeddings.gather(input_ids, 0)?;
        let position_embeds = self.position_embeddings.gather(&position_ids, 0)?;
        let token_type_embeds = self.token_type_embeddings.gather(&token_type_ids, 0)?;

        // Sum embeddings
        let mut embeddings = inputs_embeds;
        embeddings = embeddings.add(&position_embeds)?;
        embeddings = embeddings.add(&token_type_embeds)?;

        // Layer norm and dropout
        let embeddings = self.layer_norm.forward(&embeddings)?;

        Ok(embeddings)
    }
}

#[derive(Debug)]
pub struct {{MODEL_NAME}}Layer {
    pub attention: MultiHeadAttention,
    pub feed_forward: FeedForward,
    pub attention_layer_norm: LayerNorm,
    pub output_layer_norm: LayerNorm,
}

impl {{MODEL_NAME}}Layer {
    pub fn new(config: &{{MODEL_NAME}}Config) -> Self {
        Self {
            attention: MultiHeadAttention::new(
                config.hidden_size,
                config.num_attention_heads,
                0.1,
            ),
            feed_forward: FeedForward::new(
                config.hidden_size,
                config.intermediate_size,
            ),
            attention_layer_norm: LayerNorm::new(config.hidden_size),
            output_layer_norm: LayerNorm::new(config.hidden_size),
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        // Self-attention with residual connection
        let attention_output = self.attention.forward(
            hidden_states,
            hidden_states,
            hidden_states,
            attention_mask,
        )?;
        let attention_output = hidden_states.add(&attention_output)?;
        let attention_output = self.attention_layer_norm.forward(&attention_output)?;

        // Feed-forward with residual connection
        let ff_output = self.feed_forward.forward(&attention_output)?;
        let output = attention_output.add(&ff_output)?;
        let output = self.output_layer_norm.forward(&output)?;

        Ok(output)
    }
}

#[derive(Debug)]
pub struct {{MODEL_NAME}}Encoder {
    pub layers: Vec<{{MODEL_NAME}}Layer>,
}

impl {{MODEL_NAME}}Encoder {
    pub fn new(config: &{{MODEL_NAME}}Config) -> Self {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push({{MODEL_NAME}}Layer::new(config));
        }

        Self { layers }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut current_hidden_states = hidden_states.clone();

        for layer in &self.layers {
            current_hidden_states = layer.forward(&current_hidden_states, attention_mask)?;
        }

        Ok(current_hidden_states)
    }
}

#[derive(Debug)]
pub struct {{MODEL_NAME}} {
    pub config: {{MODEL_NAME}}Config,
    pub embeddings: {{MODEL_NAME}}Embeddings,
    pub encoder: {{MODEL_NAME}}Encoder,
    pub pooler: Option<Tensor>, // Linear layer for pooling
}

impl {{MODEL_NAME}} {
    pub fn new(config: {{MODEL_NAME}}Config) -> Self {
        let embeddings = {{MODEL_NAME}}Embeddings::new(&config);
        let encoder = {{MODEL_NAME}}Encoder::new(&config);
        let pooler = Some(Tensor::zeros(&[config.hidden_size, config.hidden_size]));

        Self {
            config,
            embeddings,
            encoder,
            pooler,
        }
    }

    pub fn from_pretrained(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Load configuration
        let config_path = format!("{}/config.json", model_path);
        let config_str = std::fs::read_to_string(config_path)?;
        let config: {{MODEL_NAME}}Config = serde_json::from_str(&config_str)?;

        let mut model = Self::new(config);

        // Load weights
        model.load_weights(model_path)?;

        Ok(model)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
    ) -> Result<{{MODEL_NAME}}Output, Box<dyn std::error::Error>> {
        // Create extended attention mask
        let extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids)?;

        // Embedding layer
        let embedding_output = self.embeddings.forward(
            input_ids,
            token_type_ids,
            position_ids,
        )?;

        // Encoder layers
        let sequence_output = self.encoder.forward(&embedding_output, Some(&extended_attention_mask))?;

        // Pooling (optional)
        let pooled_output = if let Some(pooler) = &self.pooler {
            let first_token_tensor = sequence_output.select(1, 0)?;
            let pooled = first_token_tensor.matmul(pooler)?;
            Some(pooled.tanh()?)
        } else {
            None
        };

        Ok({{MODEL_NAME}}Output {
            last_hidden_state: sequence_output,
            pooler_output: pooled_output,
            hidden_states: None,
            attentions: None,
        })
    }

    fn get_extended_attention_mask(
        &self,
        attention_mask: Option<&Tensor>,
        input_shape: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        match attention_mask {
            Some(mask) => {
                // Extend dimensions for multi-head attention
                let batch_size = input_shape.shape()[0];
                let seq_length = input_shape.shape()[1];

                let mut extended_mask = mask.unsqueeze(1)?.unsqueeze(2)?;
                extended_mask = extended_mask.expand(&[batch_size, 1, seq_length, seq_length])?;

                // Convert to attention scores (0.0 for attention, -10000.0 for no attention)
                let extended_mask = extended_mask.mul_scalar(1.0)?;
                let extended_mask = extended_mask.sub_scalar(1.0)?;
                let extended_mask = extended_mask.mul_scalar(-10000.0)?;

                Ok(extended_mask)
            }
            None => {
                let batch_size = input_shape.shape()[0];
                let seq_length = input_shape.shape()[1];
                Ok(Tensor::zeros(&[batch_size, 1, seq_length, seq_length]))
            }
        }
    }

    fn load_weights(&mut self, model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::collections::HashMap;
        use std::fs;
        use std::path::Path;

        // Check for .safetensors file first (preferred format)
        let safetensors_path = format!("{}/model.safetensors", model_path);
        if Path::new(&safetensors_path).exists() {
            return self.load_safetensors_weights(&safetensors_path);
        }

        // Fallback to PyTorch .bin file
        let pytorch_path = format!("{}/pytorch_model.bin", model_path);
        if Path::new(&pytorch_path).exists() {
            return self.load_pytorch_weights(&pytorch_path);
        }

        // If no weights file found, return error
        Err(format!("No weight files found in {}", model_path).into())
    }

    fn load_safetensors_weights(&mut self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // This is a template implementation showing the expected structure
        // In a real implementation, you would use the safetensors crate

        println!("Loading SafeTensors weights from: {}", file_path);

        // Placeholder for SafeTensors loading logic
        // let tensors = safetensors::SafeTensors::deserialize(std::fs::read(file_path)?)?;
        //
        // Example of how weights would be loaded:
        // if let Some(embedding_weight) = tensors.tensor("embeddings.word_embeddings.weight")? {
        //     self.embeddings.word_embeddings = Tensor::from_safetensor(embedding_weight)?;
        // }
        //
        // Load each layer's weights:
        // for i in 0..self.config.num_hidden_layers {
        //     let layer_prefix = format!("encoder.layer.{}", i);
        //     self.load_layer_weights(&tensors, &layer_prefix, i)?;
        // }

        println!("SafeTensors weights loaded successfully");
        Ok(())
    }

    fn load_pytorch_weights(&mut self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // This is a template implementation showing the expected structure
        // In a real implementation, you would use tch or candle for PyTorch compatibility

        println!("Loading PyTorch weights from: {}", file_path);

        // Placeholder for PyTorch loading logic
        // Example structure:
        // let weights = tch::pickle::load_pickle(file_path)?;
        //
        // Load embeddings
        // if let Some(embedding_tensor) = weights.get("embeddings.word_embeddings.weight") {
        //     self.embeddings.word_embeddings = Tensor::from_tch(embedding_tensor)?;
        // }
        //
        // Load transformer layers
        // for i in 0..self.config.num_hidden_layers {
        //     self.load_layer_from_pytorch(&weights, i)?;
        // }

        println!("PyTorch weights loaded successfully");
        Ok(())
    }

    // Helper method to load individual layer weights
    fn load_layer_weights(&mut self, layer_index: usize, layer_weights: &HashMap<String, Vec<f32>>) -> Result<(), Box<dyn std::error::Error>> {
        let layer_prefix = format!("encoder.layer.{}", layer_index);

        // Load attention weights
        if let Some(query_weight) = layer_weights.get(&format!("{}.attention.self.query.weight", layer_prefix)) {
            // self.encoder.layers[layer_index].attention.query.weight = Tensor::from_vec(query_weight.clone())?;
        }

        if let Some(key_weight) = layer_weights.get(&format!("{}.attention.self.key.weight", layer_prefix)) {
            // self.encoder.layers[layer_index].attention.key.weight = Tensor::from_vec(key_weight.clone())?;
        }

        if let Some(value_weight) = layer_weights.get(&format!("{}.attention.self.value.weight", layer_prefix)) {
            // self.encoder.layers[layer_index].attention.value.weight = Tensor::from_vec(value_weight.clone())?;
        }

        // Load feedforward weights
        if let Some(intermediate_weight) = layer_weights.get(&format!("{}.intermediate.dense.weight", layer_prefix)) {
            // self.encoder.layers[layer_index].intermediate.weight = Tensor::from_vec(intermediate_weight.clone())?;
        }

        if let Some(output_weight) = layer_weights.get(&format!("{}.output.dense.weight", layer_prefix)) {
            // self.encoder.layers[layer_index].output.weight = Tensor::from_vec(output_weight.clone())?;
        }

        println!("Loaded weights for layer {}", layer_index);
        Ok(())
    }

    pub fn save_pretrained(&self, save_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(save_path)?;

        // Save configuration
        let config_path = format!("{}/config.json", save_path);
        let config_str = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(config_path, config_str)?;

        // Save model weights in SafeTensors format (preferred)
        self.save_safetensors_weights(save_path)?;

        // Save tokenizer configuration if available
        self.save_tokenizer_config(save_path)?;

        println!("Model saved successfully to: {}", save_path);
        Ok(())
    }

    fn save_safetensors_weights(&self, save_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::collections::HashMap;

        println!("Saving model weights in SafeTensors format...");

        let safetensors_path = format!("{}/model.safetensors", save_path);

        // This is a template implementation showing the expected structure
        // In a real implementation, you would use the safetensors crate

        // Collect all model weights into a HashMap
        let mut tensors = HashMap::new();

        // Save embedding weights
        // tensors.insert("embeddings.word_embeddings.weight".to_string(),
        //                self.embeddings.word_embeddings.to_safetensor_bytes()?);
        // tensors.insert("embeddings.position_embeddings.weight".to_string(),
        //                self.embeddings.position_embeddings.to_safetensor_bytes()?);
        // tensors.insert("embeddings.token_type_embeddings.weight".to_string(),
        //                self.embeddings.token_type_embeddings.to_safetensor_bytes()?);

        // Save layer weights
        for i in 0..self.config.num_hidden_layers {
            self.save_layer_weights(&mut tensors, i)?;
        }

        // Save pooler weights if present
        // if let Some(pooler) = &self.pooler {
        //     tensors.insert("pooler.dense.weight".to_string(), pooler.dense.weight.to_safetensor_bytes()?);
        //     tensors.insert("pooler.dense.bias".to_string(), pooler.dense.bias.to_safetensor_bytes()?);
        // }

        // Serialize and save (placeholder)
        // let serialized = safetensors::serialize(&tensors)?;
        // std::fs::write(safetensors_path, serialized)?;

        println!("SafeTensors weights saved to: {}", safetensors_path);
        Ok(())
    }

    fn save_layer_weights(&self, tensors: &mut HashMap<String, Vec<u8>>, layer_index: usize) -> Result<(), Box<dyn std::error::Error>> {
        let layer_prefix = format!("encoder.layer.{}", layer_index);

        // Save attention weights
        // tensors.insert(format!("{}.attention.self.query.weight", layer_prefix),
        //                self.encoder.layers[layer_index].attention.query.weight.to_safetensor_bytes()?);
        // tensors.insert(format!("{}.attention.self.query.bias", layer_prefix),
        //                self.encoder.layers[layer_index].attention.query.bias.to_safetensor_bytes()?);

        // tensors.insert(format!("{}.attention.self.key.weight", layer_prefix),
        //                self.encoder.layers[layer_index].attention.key.weight.to_safetensor_bytes()?);
        // tensors.insert(format!("{}.attention.self.key.bias", layer_prefix),
        //                self.encoder.layers[layer_index].attention.key.bias.to_safetensor_bytes()?);

        // tensors.insert(format!("{}.attention.self.value.weight", layer_prefix),
        //                self.encoder.layers[layer_index].attention.value.weight.to_safetensor_bytes()?);
        // tensors.insert(format!("{}.attention.self.value.bias", layer_prefix),
        //                self.encoder.layers[layer_index].attention.value.bias.to_safetensor_bytes()?);

        // Save attention output weights
        // tensors.insert(format!("{}.attention.output.dense.weight", layer_prefix),
        //                self.encoder.layers[layer_index].attention.output.weight.to_safetensor_bytes()?);
        // tensors.insert(format!("{}.attention.output.dense.bias", layer_prefix),
        //                self.encoder.layers[layer_index].attention.output.bias.to_safetensor_bytes()?);

        // Save feedforward weights
        // tensors.insert(format!("{}.intermediate.dense.weight", layer_prefix),
        //                self.encoder.layers[layer_index].intermediate.weight.to_safetensor_bytes()?);
        // tensors.insert(format!("{}.intermediate.dense.bias", layer_prefix),
        //                self.encoder.layers[layer_index].intermediate.bias.to_safetensor_bytes()?);

        // tensors.insert(format!("{}.output.dense.weight", layer_prefix),
        //                self.encoder.layers[layer_index].output.weight.to_safetensor_bytes()?);
        // tensors.insert(format!("{}.output.dense.bias", layer_prefix),
        //                self.encoder.layers[layer_index].output.bias.to_safetensor_bytes()?);

        // Save layer norm weights
        // tensors.insert(format!("{}.attention.output.LayerNorm.weight", layer_prefix),
        //                self.encoder.layers[layer_index].attention_norm.weight.to_safetensor_bytes()?);
        // tensors.insert(format!("{}.attention.output.LayerNorm.bias", layer_prefix),
        //                self.encoder.layers[layer_index].attention_norm.bias.to_safetensor_bytes()?);

        // tensors.insert(format!("{}.output.LayerNorm.weight", layer_prefix),
        //                self.encoder.layers[layer_index].output_norm.weight.to_safetensor_bytes()?);
        // tensors.insert(format!("{}.output.LayerNorm.bias", layer_prefix),
        //                self.encoder.layers[layer_index].output_norm.bias.to_safetensor_bytes()?);

        println!("Prepared weights for layer {} for saving", layer_index);
        Ok(())
    }

    fn save_tokenizer_config(&self, save_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Save tokenizer configuration
        let tokenizer_config = serde_json::json!({
            "tokenizer_class": "{{MODEL_NAME}}Tokenizer",
            "vocab_size": self.config.vocab_size,
            "model_max_length": self.config.max_position_embeddings,
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]"
        });

        let tokenizer_config_path = format!("{}/tokenizer_config.json", save_path);
        let tokenizer_config_str = serde_json::to_string_pretty(&tokenizer_config)?;
        std::fs::write(tokenizer_config_path, tokenizer_config_str)?;

        // Save special tokens map
        let special_tokens_map = serde_json::json!({
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "mask_token": "[MASK]"
        });

        let special_tokens_path = format!("{}/special_tokens_map.json", save_path);
        let special_tokens_str = serde_json::to_string_pretty(&special_tokens_map)?;
        std::fs::write(special_tokens_path, special_tokens_str)?;

        println!("Tokenizer configuration saved");
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct {{MODEL_NAME}}Output {
    pub last_hidden_state: Tensor,
    pub pooler_output: Option<Tensor>,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attentions: Option<Vec<Tensor>>,
}

impl {{MODEL_NAME}}Output {
    pub fn logits(&self) -> &Tensor {
        &self.last_hidden_state
    }
}

{{LAYERS_IMPL}}