use trustformers_core::layers::layer_norm::LayerNorm;
use trustformers_core::tensor::Tensor;
use std::collections::HashMap;

/// Extended LSTM (xLSTM) Configuration
/// Implementation of the revolutionary Extended LSTM architecture (2024)
/// that revives LSTM with modern enhancements for transformer-competitive performance
#[derive(Debug, Clone)]
pub struct {{MODEL_NAME}}Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,

    // xLSTM-specific parameters
    pub slstm_layers: Vec<usize>,  // Layer indices for scalar memory LSTM (sLSTM)
    pub mlstm_layers: Vec<usize>,  // Layer indices for matrix memory LSTM (mLSTM)
    pub exponential_gating: bool,   // Enable exponential gating mechanism
    pub memory_dimension: usize,    // Memory dimension for mLSTM blocks
    pub num_heads: usize,          // Number of heads for mLSTM (like attention)

    // Training and regularization
    pub dropout: f32,
    pub layer_norm_eps: f32,
    pub initializer_range: f32,
    pub pad_token_id: i32,
    pub use_cache: bool,
}

impl Default for {{MODEL_NAME}}Config {
    fn default() -> Self {
        Self {
            vocab_size: {{VOCAB_SIZE}},
            hidden_size: {{HIDDEN_SIZE}},
            num_layers: {{NUM_LAYERS}},
            intermediate_size: {{HIDDEN_SIZE}} * 4,
            max_position_embeddings: {{MAX_POSITION_EMBEDDINGS}},

            // Default xLSTM configuration with mixed sLSTM/mLSTM blocks
            slstm_layers: (0..{{NUM_LAYERS}}).step_by(2).collect(), // Even layers: sLSTM
            mlstm_layers: (1..{{NUM_LAYERS}}).step_by(2).collect(), // Odd layers: mLSTM
            exponential_gating: true,
            memory_dimension: {{HIDDEN_SIZE}},
            num_heads: {{NUM_ATTENTION_HEADS}},

            dropout: 0.1,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
            pad_token_id: 0,
            use_cache: true,
        }
    }
}

/// Scalar Memory LSTM (sLSTM) Block
/// Enhanced LSTM with exponential gating for improved memory capacity
#[derive(Debug)]
pub struct SLSTMBlock {
    pub input_gate: Tensor,      // Input gate weights
    pub forget_gate: Tensor,     // Forget gate weights
    pub output_gate: Tensor,     // Output gate weights
    pub cell_gate: Tensor,       // Cell state candidate weights
    pub layer_norm: LayerNorm,

    // Exponential gating enhancements
    pub exp_forget_bias: f32,    // Exponential forget gate bias
    pub exp_input_bias: f32,     // Exponential input gate bias
    pub stabilizer_eps: f32,     // Numerical stability epsilon
}

impl SLSTMBlock {
    pub fn new(hidden_size: usize, config: &{{MODEL_NAME}}Config) -> Self {
        Self {
            input_gate: Tensor::zeros(&[hidden_size, hidden_size * 4]),
            forget_gate: Tensor::zeros(&[hidden_size, hidden_size * 4]),
            output_gate: Tensor::zeros(&[hidden_size, hidden_size * 4]),
            cell_gate: Tensor::zeros(&[hidden_size, hidden_size * 4]),
            layer_norm: LayerNorm::new(hidden_size, config.layer_norm_eps),

            exp_forget_bias: 3.0,    // Bias for exponential forget gate
            exp_input_bias: -1.0,    // Bias for exponential input gate
            stabilizer_eps: 1e-8,
        }
    }

    /// Forward pass with exponential gating mechanism
    /// Implements the core sLSTM computation with enhanced memory capacity
    pub fn forward(&self, input: &Tensor, hidden_state: &Tensor, cell_state: &Tensor) -> (Tensor, Tensor, Tensor) {
        let batch_size = input.size()[0];
        let seq_len = input.size()[1];
        let hidden_size = hidden_state.size()[2];

        // Concatenate input and previous hidden state
        let combined = Tensor::cat(&[input, hidden_state], 2);

        // Compute gates with exponential enhancement
        let gates = combined.matmul(&self.input_gate);
        let gate_chunks = gates.chunk(4, 2);

        let input_gate = if self.exp_input_bias != 0.0 {
            (&gate_chunks[0] + self.exp_input_bias).exp()  // Exponential input gate
        } else {
            gate_chunks[0].sigmoid()
        };

        let forget_gate = if self.exp_forget_bias != 0.0 {
            (&gate_chunks[1] + self.exp_forget_bias).exp()  // Exponential forget gate
        } else {
            gate_chunks[1].sigmoid()
        };

        let output_gate = gate_chunks[2].sigmoid();
        let cell_candidate = gate_chunks[3].tanh();

        // Enhanced cell state computation with exponential gating
        let new_cell_state = &forget_gate * cell_state + &input_gate * &cell_candidate;

        // Stabilization for exponential gating
        let stabilized_cell = if self.stabilizer_eps > 0.0 {
            &new_cell_state / (new_cell_state.abs().mean(-1, true) + self.stabilizer_eps)
        } else {
            new_cell_state.clone()
        };

        // Output computation with layer normalization
        let cell_tanh = stabilized_cell.tanh();
        let raw_output = &output_gate * &cell_tanh;
        let output = self.layer_norm.forward(&raw_output);

        (output, raw_output, stabilized_cell)
    }
}

/// Matrix Memory LSTM (mLSTM) Block
/// Fully parallelizable LSTM variant with matrix memory for transformer-like efficiency
#[derive(Debug)]
pub struct MLSTMBlock {
    pub query_projection: Tensor,    // Query projection for matrix memory
    pub key_projection: Tensor,      // Key projection for matrix memory
    pub value_projection: Tensor,    // Value projection for matrix memory
    pub forget_gate: Tensor,         // Forget gate weights
    pub input_gate: Tensor,          // Input gate weights
    pub output_projection: Tensor,   // Output projection
    pub layer_norm: LayerNorm,

    pub num_heads: usize,
    pub head_dim: usize,
    pub memory_dimension: usize,
}

impl MLSTMBlock {
    pub fn new(hidden_size: usize, config: &{{MODEL_NAME}}Config) -> Self {
        let head_dim = hidden_size / config.num_heads;

        Self {
            query_projection: Tensor::zeros(&[hidden_size, hidden_size]),
            key_projection: Tensor::zeros(&[hidden_size, config.memory_dimension]),
            value_projection: Tensor::zeros(&[hidden_size, hidden_size]),
            forget_gate: Tensor::zeros(&[hidden_size, hidden_size]),
            input_gate: Tensor::zeros(&[hidden_size, hidden_size]),
            output_projection: Tensor::zeros(&[hidden_size, hidden_size]),
            layer_norm: LayerNorm::new(hidden_size, config.layer_norm_eps),

            num_heads: config.num_heads,
            head_dim,
            memory_dimension: config.memory_dimension,
        }
    }

    /// Forward pass with matrix memory mechanism
    /// Implements parallelizable LSTM with matrix-based memory storage
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let batch_size = input.size()[0];
        let seq_len = input.size()[1];
        let hidden_size = input.size()[2];

        // Project to query, key, value
        let queries = input.matmul(&self.query_projection);
        let keys = input.matmul(&self.key_projection);
        let values = input.matmul(&self.value_projection);

        // Reshape for multi-head processing
        let q = queries.view(&[batch_size, seq_len, self.num_heads, self.head_dim]).transpose(1, 2);
        let k = keys.view(&[batch_size, seq_len, self.num_heads, self.memory_dimension / self.num_heads]).transpose(1, 2);
        let v = values.view(&[batch_size, seq_len, self.num_heads, self.head_dim]).transpose(1, 2);

        // Compute gates
        let forget_gates = input.matmul(&self.forget_gate).sigmoid();
        let input_gates = input.matmul(&self.input_gate).sigmoid();

        // Matrix memory computation (parallelizable across sequence)
        let mut memory_states = Vec::new();
        let mut current_memory = Tensor::zeros(&[batch_size, self.num_heads, self.memory_dimension / self.num_heads, self.head_dim]);

        for t in 0..seq_len {
            let q_t = q.select(2, t as i64);  // [batch, heads, head_dim]
            let k_t = k.select(2, t as i64);  // [batch, heads, mem_dim/heads]
            let v_t = v.select(2, t as i64);  // [batch, heads, head_dim]
            let f_t = forget_gates.select(1, t as i64);  // [batch, hidden]
            let i_t = input_gates.select(1, t as i64);   // [batch, hidden]

            // Update matrix memory: M_t = f_t * M_{t-1} + i_t * (k_t^T @ v_t)
            let f_reshaped = f_t.view(&[batch_size, self.num_heads, self.head_dim]).unsqueeze(2);
            let i_reshaped = i_t.view(&[batch_size, self.num_heads, self.head_dim]).unsqueeze(2);

            let outer_product = k_t.unsqueeze(3).matmul(&v_t.unsqueeze(2));
            current_memory = &f_reshaped * &current_memory + &i_reshaped * &outer_product;

            // Compute output: o_t = q_t^T @ M_t
            let output_t = q_t.unsqueeze(2).matmul(&current_memory).squeeze(2);
            memory_states.push(output_t);
        }

        // Concatenate all outputs
        let stacked_outputs = Tensor::stack(&memory_states, 2);
        let reshaped_output = stacked_outputs.transpose(1, 2).contiguous()
            .view(&[batch_size, seq_len, hidden_size]);

        // Final projection and layer norm
        let projected = reshaped_output.matmul(&self.output_projection);
        self.layer_norm.forward(&projected)
    }
}

/// Main xLSTM Model
/// Combines sLSTM and mLSTM blocks in configurable patterns
#[derive(Debug)]
pub struct {{MODEL_NAME}} {
    pub embeddings: {{MODEL_NAME}}Embeddings,
    pub slstm_blocks: Vec<SLSTMBlock>,
    pub mlstm_blocks: Vec<MLSTMBlock>,
    pub layer_mapping: Vec<String>,  // "slstm" or "mlstm" for each layer
    pub final_layer_norm: LayerNorm,
    pub config: {{MODEL_NAME}}Config,
}

impl {{MODEL_NAME}} {
    pub fn new(config: {{MODEL_NAME}}Config) -> Self {
        let mut slstm_blocks = Vec::new();
        let mut mlstm_blocks = Vec::new();
        let mut layer_mapping = Vec::new();

        // Create blocks according to configuration
        for layer_idx in 0..config.num_layers {
            if config.slstm_layers.contains(&layer_idx) {
                slstm_blocks.push(SLSTMBlock::new(config.hidden_size, &config));
                layer_mapping.push("slstm".to_string());
            } else if config.mlstm_layers.contains(&layer_idx) {
                mlstm_blocks.push(MLSTMBlock::new(config.hidden_size, &config));
                layer_mapping.push("mlstm".to_string());
            } else {
                // Default to sLSTM if not specified
                slstm_blocks.push(SLSTMBlock::new(config.hidden_size, &config));
                layer_mapping.push("slstm".to_string());
            }
        }

        Self {
            embeddings: {{MODEL_NAME}}Embeddings::new(&config),
            slstm_blocks,
            mlstm_blocks,
            layer_mapping,
            final_layer_norm: LayerNorm::new(config.hidden_size, config.layer_norm_eps),
            config,
        }
    }

    /// Forward pass through the xLSTM model
    /// Processes input through the configured sequence of sLSTM and mLSTM blocks
    pub fn forward(&self, input_ids: &Tensor) -> Tensor {
        let mut hidden_states = self.embeddings.forward(input_ids);

        let batch_size = hidden_states.size()[0];
        let seq_len = hidden_states.size()[1];
        let hidden_size = self.config.hidden_size;

        // Initialize LSTM states for sLSTM blocks
        let mut lstm_hidden = Tensor::zeros(&[batch_size, 1, hidden_size]);
        let mut lstm_cell = Tensor::zeros(&[batch_size, 1, hidden_size]);

        let mut slstm_idx = 0;
        let mut mlstm_idx = 0;

        // Process through each layer according to the configured pattern
        for (layer_idx, layer_type) in self.layer_mapping.iter().enumerate() {
            match layer_type.as_str() {
                "slstm" => {
                    let (output, new_hidden, new_cell) = self.slstm_blocks[slstm_idx]
                        .forward(&hidden_states, &lstm_hidden, &lstm_cell);
                    hidden_states = output;
                    lstm_hidden = new_hidden;
                    lstm_cell = new_cell;
                    slstm_idx += 1;
                }
                "mlstm" => {
                    hidden_states = self.mlstm_blocks[mlstm_idx].forward(&hidden_states);
                    mlstm_idx += 1;
                }
                _ => unreachable!("Unknown layer type in xLSTM"),
            }
        }

        self.final_layer_norm.forward(&hidden_states)
    }

    /// Get the number of parameters in the model
    pub fn num_parameters(&self) -> usize {
        let mut total_params = 0;

        // Embeddings parameters
        total_params += self.config.vocab_size * self.config.hidden_size; // word embeddings
        total_params += self.config.max_position_embeddings * self.config.hidden_size; // position embeddings

        // sLSTM block parameters
        for _ in &self.slstm_blocks {
            total_params += self.config.hidden_size * self.config.hidden_size * 4; // gates
            total_params += self.config.hidden_size * 2; // layer norm
        }

        // mLSTM block parameters
        for _ in &self.mlstm_blocks {
            total_params += self.config.hidden_size * self.config.hidden_size; // query proj
            total_params += self.config.hidden_size * self.config.memory_dimension; // key proj
            total_params += self.config.hidden_size * self.config.hidden_size; // value proj
            total_params += self.config.hidden_size * self.config.hidden_size; // forget gate
            total_params += self.config.hidden_size * self.config.hidden_size; // input gate
            total_params += self.config.hidden_size * self.config.hidden_size; // output proj
            total_params += self.config.hidden_size * 2; // layer norm
        }

        // Final layer norm
        total_params += self.config.hidden_size * 2;

        total_params
    }
}

/// Embeddings for xLSTM model
#[derive(Debug)]
pub struct {{MODEL_NAME}}Embeddings {
    pub word_embeddings: Tensor,
    pub position_embeddings: Tensor,
    pub layer_norm: LayerNorm,
    pub dropout: f32,
}

impl {{MODEL_NAME}}Embeddings {
    pub fn new(config: &{{MODEL_NAME}}Config) -> Self {
        Self {
            word_embeddings: Tensor::zeros(&[config.vocab_size, config.hidden_size]),
            position_embeddings: Tensor::zeros(&[config.max_position_embeddings, config.hidden_size]),
            layer_norm: LayerNorm::new(config.hidden_size, config.layer_norm_eps),
            dropout: config.dropout,
        }
    }

    pub fn forward(&self, input_ids: &Tensor) -> Tensor {
        let seq_length = input_ids.size()[1];
        let position_ids = Tensor::arange(seq_length, input_ids.device())
            .unsqueeze(0)
            .expand_as(input_ids);

        let inputs_embeds = input_ids.embedding(&self.word_embeddings);
        let position_embeds = position_ids.embedding(&self.position_embeddings);

        let embeddings = inputs_embeds + position_embeds;
        self.layer_norm.forward(&embeddings)
    }
}

{{LAYERS_IMPL}}