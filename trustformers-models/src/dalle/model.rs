use crate::dalle::config::{
    DalleConfig, DalleDiffusionConfig, DalleImageConfig, DalleTextConfig, DalleVisionConfig,
};
use trustformers_core::{
    device::Device,
    kernels::fused_ops::ActivationType,
    layers::{
        attention::{AttentionConfig, MultiHeadAttention},
        conv2d::Conv2d,
        embedding::Embedding,
        layernorm::LayerNorm,
        linear::Linear,
    },
    tensor::{DType, Tensor, TensorType},
    traits::Layer,
};

/// DALL-E model for text-to-image generation
#[derive(Debug, Clone)]
pub struct DalleModel {
    /// Configuration
    pub config: DalleConfig,
    /// Text encoder (CLIP or T5)
    pub text_encoder: DalleTextEncoder,
    /// Image encoder for CLIP alignment
    pub image_encoder: DalleImageEncoder,
    /// U-Net diffusion model
    pub unet: DalleUNet,
    /// VAE encoder/decoder
    pub vae: DalleVAE,
    /// Text projection layer
    pub text_projection: Linear,
    /// Image projection layer
    pub image_projection: Linear,
    /// Temperature parameter for CLIP loss
    pub logit_scale: Tensor,
    /// Device for computation
    device: Device,
}

impl DalleModel {
    /// Create a new DALL-E model on CPU
    pub fn new(config: DalleConfig) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_device(config, Device::CPU)
    }

    /// Create a new DALL-E model with specified device
    pub fn new_with_device(
        config: DalleConfig,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let text_encoder = DalleTextEncoder::new_with_device(config.text_config.clone(), device)?;
        let image_encoder =
            DalleImageEncoder::new_with_device(config.vision_config.clone(), device)?;
        let unet = DalleUNet::new_with_device(
            config.image_config.clone(),
            config.diffusion_config.clone(),
            device,
        )?;
        let vae = DalleVAE::new_with_device(config.image_config.clone(), device)?;

        let text_projection = Linear::new(
            config.text_config.hidden_size,
            config.vision_config.hidden_size,
            false,
        );

        let image_projection = Linear::new(
            config.vision_config.hidden_size,
            config.vision_config.hidden_size,
            false,
        );

        let logit_scale = Tensor::from_scalar(2.6592, TensorType::F32)?; // ln(1/0.07)

        Ok(Self {
            config,
            text_encoder,
            image_encoder,
            unet,
            vae,
            text_projection,
            image_projection,
            logit_scale,
            device,
        })
    }

    /// Get the device this model is on
    pub fn device(&self) -> Device {
        self.device
    }

    /// Forward pass for training (CLIP alignment + diffusion loss)
    pub fn forward_train(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        pixel_values: &Tensor,
        noise: Option<&Tensor>,
        timesteps: Option<&Tensor>,
    ) -> Result<DalleModelOutput, Box<dyn std::error::Error>> {
        // Encode text
        let text_features = self.text_encoder.forward(input_ids, attention_mask)?;
        let text_embeds = self.text_projection.forward(text_features.clone())?;
        // L2 normalization along last dimension
        let text_norm = text_embeds.norm_dim(2, Some(vec![-1]), true)?;
        let text_embeds = text_embeds.div(&text_norm)?;

        // Encode images with CLIP vision encoder
        let image_features = self.image_encoder.forward(pixel_values)?;
        let image_embeds = self.image_projection.forward(image_features.clone())?;
        // L2 normalization along last dimension
        let image_norm = image_embeds.norm_dim(2, Some(vec![-1]), true)?;
        let image_embeds = image_embeds.div(&image_norm)?;

        // CLIP contrastive loss
        let logit_scale = self.logit_scale.exp()?;
        let logits_per_image =
            (&image_embeds.matmul(&text_embeds.transpose(1, 2)?)? * &logit_scale)?;
        let logits_per_text = logits_per_image.transpose(1, 2)?;

        // VAE encoding
        let latents = self.vae.encode(pixel_values)?;

        // Add noise for diffusion training
        let (noisy_latents, noise_pred_target) =
            if let (Some(noise), Some(timesteps)) = (noise, timesteps) {
                let noisy_latents = self.add_noise(&latents, noise, timesteps)?;
                (noisy_latents, noise.clone())
            } else {
                // Generate random noise and timesteps for training
                let noise = Tensor::randn_like(&latents)?;
                let timesteps = Tensor::randint(
                    0,
                    self.config.num_diffusion_steps as i64,
                    &[latents.shape()[0]],
                    TensorType::I64,
                )?;
                let noisy_latents = self.add_noise(&latents, &noise, &timesteps)?;
                (noisy_latents, noise)
            };

        // U-Net noise prediction
        let noise_pred = self.unet.forward(
            &noisy_latents,
            timesteps.as_ref().expect("operation failed"),
            &text_embeds,
        )?;

        Ok(DalleModelOutput {
            text_embeds: Some(text_embeds),
            image_embeds: Some(image_embeds),
            logits_per_image: Some(logits_per_image),
            logits_per_text: Some(logits_per_text),
            latents: Some(latents),
            noise_pred: Some(noise_pred),
            noise_pred_target: Some(noise_pred_target),
        })
    }

    /// Generate images from text descriptions
    pub fn generate(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        num_inference_steps: Option<usize>,
        guidance_scale: Option<f64>,
        _generator: Option<u64>,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let num_steps =
            num_inference_steps.unwrap_or(self.config.diffusion_config.num_inference_steps);
        let guidance = guidance_scale.unwrap_or(self.config.guidance_scale);

        // Encode text prompt
        let text_features = self.text_encoder.forward(input_ids, attention_mask)?;
        let text_embeds = self.text_projection.forward(text_features.clone())?;
        // L2 normalization along last dimension
        let text_norm = text_embeds.norm_dim(2, Some(vec![-1]), true)?;
        let text_embeds = text_embeds.div(&text_norm)?;

        // Create unconditioned embeddings for classifier-free guidance
        let batch_size = input_ids.shape()[0];
        let uncond_input_ids =
            Tensor::zeros_dtype(TensorType::I64, &[batch_size, input_ids.shape()[1]])?;
        let uncond_attention_mask =
            Tensor::zeros_dtype(TensorType::F32, &[batch_size, attention_mask.shape()[1]])?;
        let uncond_features =
            self.text_encoder.forward(&uncond_input_ids, &uncond_attention_mask)?;
        let uncond_embeds = self.text_projection.forward(uncond_features.clone())?;
        // L2 normalization along last dimension
        let uncond_norm = uncond_embeds.norm_dim(2, Some(vec![-1]), true)?;
        let uncond_embeds = uncond_embeds.div(&uncond_norm)?;

        // Initialize random latents
        let latent_shape = vec![
            batch_size,
            self.config.image_config.latent_channels,
            self.config.image_config.latent_size(),
            self.config.image_config.latent_size(),
        ];
        let mut latents = Tensor::randn(&latent_shape)?;

        // DDIM sampling schedule
        let timesteps = self.get_timesteps(num_steps)?;

        for (i, &t) in timesteps.iter().enumerate() {
            let timestep_tensor =
                Tensor::full_with_dtype(&[batch_size], t as f64, TensorType::I64)?;

            // Classifier-free guidance
            let latent_model_input = Tensor::concat(&[latents.clone(), latents.clone()], 0)?;
            let text_embeds_input =
                Tensor::concat(&[text_embeds.clone(), uncond_embeds.clone()], 0)?;

            // U-Net prediction
            // Repeat timestep for both conditional and unconditional
            let timestep_doubled = timestep_tensor.repeat(&[2])?;
            let noise_pred =
                self.unet.forward(&latent_model_input, &timestep_doubled, &text_embeds_input)?;

            // Split conditional and unconditional predictions
            let noise_pred_cond = noise_pred.slice(0, 0, batch_size)?;
            let noise_pred_uncond = noise_pred.slice(0, batch_size, 2 * batch_size)?;

            // Apply classifier-free guidance
            let noise_pred =
                (&noise_pred_uncond + &((&noise_pred_cond - &noise_pred_uncond)? * guidance)?)?;

            // DDIM step
            latents = self.ddim_step(&latents, &noise_pred, t, i, num_steps)?;
        }

        // Decode latents to images
        let images = self.vae.decode(&latents)?;

        Ok(images)
    }

    /// Add noise to latents for diffusion training
    fn add_noise(
        &self,
        latents: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        // Get noise schedule parameters
        let alphas_cumprod = self.get_alphas_cumprod()?;

        let mut sqrt_alpha_prod = Vec::new();
        let mut sqrt_one_minus_alpha_prod = Vec::new();

        for i in 0..timesteps.shape()[0] {
            let t = timesteps.select(0, i as i64)?.get_scalar_i64()? as usize;
            sqrt_alpha_prod.push(alphas_cumprod[t].sqrt());
            sqrt_one_minus_alpha_prod.push((1.0 - alphas_cumprod[t]).sqrt());
        }

        let sqrt_alpha_prod_tensor =
            Tensor::from_vec_with_dtype(sqrt_alpha_prod, &[timesteps.shape()[0]], TensorType::F32)?;
        let sqrt_one_minus_alpha_prod_tensor = Tensor::from_vec_with_dtype(
            sqrt_one_minus_alpha_prod,
            &[timesteps.shape()[0]],
            TensorType::F32,
        )?;

        // Expand to match latents shape
        let shape = vec![timesteps.shape()[0], 1, 1, 1];
        let sqrt_alpha_prod_tensor = sqrt_alpha_prod_tensor.reshape(&shape)?;
        let sqrt_one_minus_alpha_prod_tensor = sqrt_one_minus_alpha_prod_tensor.reshape(&shape)?;

        let noisy_latents = (&(&sqrt_alpha_prod_tensor * latents)?
            + &(&sqrt_one_minus_alpha_prod_tensor * noise)?)?;

        Ok(noisy_latents)
    }

    /// Get alphas_cumprod for noise schedule
    fn get_alphas_cumprod(&self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let num_timesteps = self.config.diffusion_config.num_timesteps;
        let beta_start = self.config.diffusion_config.beta_start;
        let beta_end = self.config.diffusion_config.beta_end;

        let betas: Vec<f64> = match self.config.diffusion_config.beta_schedule.as_str() {
            "linear" => (0..num_timesteps)
                .map(|i| {
                    beta_start + (beta_end - beta_start) * i as f64 / (num_timesteps - 1) as f64
                })
                .collect(),
            "scaled_linear" => (0..num_timesteps)
                .map(|i| {
                    let linear = beta_start
                        + (beta_end - beta_start) * i as f64 / (num_timesteps - 1) as f64;
                    linear.sqrt()
                })
                .collect(),
            _ => return Err("Unsupported beta schedule".into()),
        };

        let alphas: Vec<f64> = betas.iter().map(|beta| 1.0 - beta).collect();
        let mut alphas_cumprod = Vec::with_capacity(num_timesteps);
        let mut cumprod = 1.0;

        for alpha in alphas {
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }

        Ok(alphas_cumprod)
    }

    /// Get timesteps for DDIM sampling
    fn get_timesteps(
        &self,
        num_inference_steps: usize,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        let num_train_timesteps = self.config.diffusion_config.num_timesteps;
        let step = num_train_timesteps / num_inference_steps;

        let timesteps: Vec<usize> =
            (0..num_inference_steps).map(|i| num_train_timesteps - 1 - i * step).collect();

        Ok(timesteps)
    }

    /// DDIM sampling step
    fn ddim_step(
        &self,
        sample: &Tensor,
        model_output: &Tensor,
        timestep: usize,
        step_index: usize,
        num_inference_steps: usize,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let alphas_cumprod = self.get_alphas_cumprod()?;
        let alpha_prod_t = alphas_cumprod[timestep];

        let alpha_prod_t_prev = if step_index == num_inference_steps - 1 {
            1.0
        } else {
            let timesteps = self.get_timesteps(num_inference_steps)?;
            let prev_timestep = timesteps[step_index + 1];
            alphas_cumprod[prev_timestep]
        };

        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

        let eta = self.config.diffusion_config.eta;
        let variance =
            eta * (beta_prod_t_prev / beta_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev);

        let pred_original_sample =
            (&(sample - &(model_output * (beta_prod_t.sqrt()))?)? / alpha_prod_t.sqrt())?;
        let pred_sample_direction = (model_output * (beta_prod_t_prev - variance).sqrt())?;

        let prev_sample =
            (&(&pred_original_sample * alpha_prod_t_prev.sqrt())? + &pred_sample_direction)?;

        if variance > 0.0 {
            let noise = Tensor::randn_like(sample)?;
            let prev_sample = (&prev_sample + &(&noise * variance.sqrt())?)?;
            Ok(prev_sample)
        } else {
            Ok(prev_sample)
        }
    }
}

/// Text encoder for DALL-E (CLIP or T5)
#[derive(Debug, Clone)]
pub struct DalleTextEncoder {
    pub config: DalleTextConfig,
    pub embeddings: Embedding,
    pub layers: Vec<DalleTextLayer>,
    pub final_layer_norm: LayerNorm,
    device: Device,
}

impl DalleTextEncoder {
    pub fn new(config: DalleTextConfig) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(
        config: DalleTextConfig,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let embeddings = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(DalleTextLayer::new_with_device(&config, device)?);
        }

        let final_layer_norm =
            LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;

        Ok(Self {
            config,
            embeddings,
            layers,
            final_layer_norm,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let input_ids_vec: Vec<u32> =
            input_ids.to_vec_f32()?.into_iter().map(|x| x as u32).collect();
        let mut hidden_states = self.embeddings.forward(input_ids_vec)?;

        // Add positional embeddings
        let seq_len = input_ids.shape()[1];
        let position_ids = Tensor::range(0, seq_len as i64, DType::I64)?.unsqueeze(0)?;
        let position_ids_vec: Vec<u32> =
            position_ids.to_vec_f32()?.into_iter().map(|x| x as u32).collect();
        let position_embeddings = self.embeddings.forward(position_ids_vec)?;
        hidden_states = (&hidden_states + &position_embeddings)?;

        // Apply transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }

        hidden_states = self.final_layer_norm.forward(hidden_states)?;

        // Pool by taking the last token (for CLIP) or mean pooling
        if self.config.hidden_act == "quick_gelu" {
            // CLIP-style: take the last token
            let sum_result = attention_mask.sum_dim(-1, false)?;
            let sum_i64 = sum_result.to_i64()?;
            let last_token_idx = sum_i64.sub_scalar(1.0)?;
            let pooled = hidden_states
                .gather(
                    -2,
                    &last_token_idx
                        .unsqueeze_i64(-1)?
                        .unsqueeze_i64(-1)?
                        .broadcast_to(&hidden_states.shape())?,
                )?
                .squeeze_i64(-2)?;
            Ok(pooled)
        } else {
            // T5-style: mean pooling
            let attention_mask_expanded =
                attention_mask.unsqueeze_i64(-1)?.broadcast_to(&hidden_states.shape())?;
            let product = (&hidden_states * &attention_mask_expanded)?;
            let sum_embeddings = product.sum_dim(-2, false)?;
            let sum_mask = attention_mask.sum_dim(-1, false)?;
            let pooled = sum_embeddings.div(&sum_mask.unsqueeze_i64(-1)?)?;
            Ok(pooled)
        }
    }
}

/// Single text transformer layer
#[derive(Debug, Clone)]
pub struct DalleTextLayer {
    pub self_attention: MultiHeadAttention,
    pub mlp: DalleMLP,
    pub layer_norm1: LayerNorm,
    pub layer_norm2: LayerNorm,
    device: Device,
}

impl DalleTextLayer {
    pub fn new(config: &DalleTextConfig) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(
        config: &DalleTextConfig,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let attention_config = AttentionConfig {
            hidden_size: config.hidden_size,
            num_heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
            dropout_prob: config.attention_dropout as f32,
            bias: true,
            max_seq_len: None,
        };

        let self_attention = MultiHeadAttention::new(
            attention_config.hidden_size,
            attention_config.num_heads,
            attention_config.dropout_prob,
            attention_config.bias,
        )?;
        let mlp = DalleMLP::new_with_device(
            config.hidden_size,
            config.intermediate_size,
            &config.hidden_act,
            device,
        )?;
        let layer_norm1 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;
        let layer_norm2 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;

        Ok(Self {
            self_attention,
            mlp,
            layer_norm1,
            layer_norm2,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        // Self-attention with residual connection
        let normed_states = self.layer_norm1.forward(hidden_states.clone())?;
        let attention_output = self.self_attention.forward_self_attention(
            &normed_states,
            Some(attention_mask),
            false, // bidirectional
        )?;
        let hidden_states = (hidden_states + &attention_output)?;

        // MLP with residual connection
        let normed_states = self.layer_norm2.forward(hidden_states.clone())?;
        let mlp_output = self.mlp.forward(&normed_states)?;
        let hidden_states = (&hidden_states + &mlp_output)?;

        Ok(hidden_states)
    }
}

/// Image encoder for CLIP alignment
#[derive(Debug, Clone)]
pub struct DalleImageEncoder {
    pub config: DalleVisionConfig,
    pub patch_embedding: Conv2d,
    pub class_embedding: Tensor,
    pub position_embedding: Tensor,
    pub pre_layer_norm: LayerNorm,
    pub layers: Vec<DalleVisionLayer>,
    pub post_layer_norm: LayerNorm,
    device: Device,
}

impl DalleImageEncoder {
    pub fn new(config: DalleVisionConfig) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(
        config: DalleVisionConfig,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let patch_embedding = Conv2d::new(
            config.num_channels,
            config.hidden_size,
            (config.patch_size, config.patch_size),
            (config.patch_size, config.patch_size),
            (0, 0),
            false,
        )?;

        let class_embedding = Tensor::randn(&[config.hidden_size])?;
        let position_embedding = Tensor::randn(&[config.seq_len(), config.hidden_size])?;

        let pre_layer_norm =
            LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;
        let post_layer_norm =
            LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(DalleVisionLayer::new_with_device(&config, device)?);
        }

        Ok(Self {
            config,
            patch_embedding,
            class_embedding,
            position_embedding,
            pre_layer_norm,
            layers,
            post_layer_norm,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let batch_size = pixel_values.shape()[0];

        // Patch embedding
        let patch_embeds = self.patch_embedding.forward(pixel_values.clone())?;
        let patch_embeds = patch_embeds.flatten(2, -1)?.transpose_i64(-1, -2)?; // [B, N_patches, D]

        // Add class token
        let class_embeds = self.class_embedding.unsqueeze(0)?.unsqueeze(0)?.broadcast_to(&[
            batch_size,
            1,
            self.config.hidden_size,
        ])?;
        let embeddings = Tensor::concat(&[class_embeds, patch_embeds], 1)?;

        // Add position embeddings
        let embeddings = (&embeddings + &self.position_embedding.unsqueeze(0)?)?;

        let mut hidden_states = self.pre_layer_norm.forward(embeddings)?;

        // Apply vision transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        hidden_states = self.post_layer_norm.forward(hidden_states)?;

        // Extract class token
        let pooled_output = hidden_states.slice(1, 0, 1)?.squeeze(1)?;

        Ok(pooled_output)
    }
}

/// Single vision transformer layer
#[derive(Debug, Clone)]
pub struct DalleVisionLayer {
    pub self_attention: MultiHeadAttention,
    pub mlp: DalleMLP,
    pub layer_norm1: LayerNorm,
    pub layer_norm2: LayerNorm,
    device: Device,
}

impl DalleVisionLayer {
    pub fn new(config: &DalleVisionConfig) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(
        config: &DalleVisionConfig,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let attention_config = AttentionConfig {
            hidden_size: config.hidden_size,
            num_heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
            dropout_prob: config.attention_dropout as f32,
            bias: true,
            max_seq_len: None,
        };

        let self_attention = MultiHeadAttention::new(
            attention_config.hidden_size,
            attention_config.num_heads,
            attention_config.dropout_prob,
            attention_config.bias,
        )?;
        let mlp = DalleMLP::new_with_device(
            config.hidden_size,
            config.intermediate_size,
            &config.hidden_act,
            device,
        )?;
        let layer_norm1 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;
        let layer_norm2 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;

        Ok(Self {
            self_attention,
            mlp,
            layer_norm1,
            layer_norm2,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        // Self-attention with residual connection
        let normed_states = self.layer_norm1.forward(hidden_states.clone())?;
        let attention_output = self.self_attention.forward(normed_states)?;
        let hidden_states = (hidden_states + &attention_output)?;

        // MLP with residual connection
        let normed_states = self.layer_norm2.forward(hidden_states.clone())?;
        let mlp_output = self.mlp.forward(&normed_states)?;
        let hidden_states = (&hidden_states + &mlp_output)?;

        Ok(hidden_states)
    }
}

/// U-Net model for diffusion
#[derive(Debug, Clone)]
pub struct DalleUNet {
    pub config: DalleImageConfig,
    pub time_embedding: DalleTimeEmbedding,
    pub text_projection: Linear,
    pub conv_in: Conv2d,
    pub down_blocks: Vec<DalleUNetBlock>,
    pub mid_block: DalleUNetBlock,
    pub up_blocks: Vec<DalleUNetBlock>,
    pub conv_out: Conv2d,
    device: Device,
}

impl DalleUNet {
    pub fn new(
        image_config: DalleImageConfig,
        diffusion_config: DalleDiffusionConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_device(image_config, diffusion_config, Device::CPU)
    }

    pub fn new_with_device(
        image_config: DalleImageConfig,
        _diffusion_config: DalleDiffusionConfig,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let time_embedding = DalleTimeEmbedding::new_with_device(image_config.hidden_size, device)?;
        let text_projection =
            Linear::new(image_config.hidden_size, image_config.hidden_size, false);

        let conv_in = Conv2d::new(
            image_config.latent_channels,
            image_config.hidden_size,
            (3, 3),
            (1, 1),
            (1, 1),
            false,
        )?;

        let conv_out = Conv2d::new(
            image_config.hidden_size,
            image_config.latent_channels,
            (3, 3),
            (1, 1),
            (1, 1),
            false,
        )?;

        // Simplified U-Net structure
        let mut down_blocks = Vec::new();
        let mut up_blocks = Vec::new();

        for i in 0..3 {
            let in_channels = if i == 0 {
                image_config.hidden_size
            } else {
                image_config.hidden_size * (2_usize.pow(i as u32))
            };
            let out_channels = image_config.hidden_size * (2_usize.pow((i + 1) as u32));
            down_blocks.push(DalleUNetBlock::new_with_device(
                in_channels,
                out_channels,
                true,
                device,
            )?);
        }

        let mid_block = DalleUNetBlock::new_with_device(
            image_config.hidden_size * 8,
            image_config.hidden_size * 8,
            false,
            device,
        )?;

        for i in (0..3).rev() {
            let in_channels = image_config.hidden_size * (2_usize.pow((i + 1) as u32))
                + image_config.hidden_size * (2_usize.pow((i + 1) as u32)); // Skip connection
            let out_channels = if i == 0 {
                image_config.hidden_size
            } else {
                image_config.hidden_size * (2_usize.pow(i as u32))
            };
            up_blocks.push(DalleUNetBlock::new_with_device(
                in_channels,
                out_channels,
                true,
                device,
            )?);
        }

        Ok(Self {
            config: image_config,
            time_embedding,
            text_projection,
            conv_in,
            down_blocks,
            mid_block,
            up_blocks,
            conv_out,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        // Time embedding
        let time_emb = self.time_embedding.forward(timestep)?;

        // Project text features
        let text_emb = self.text_projection.forward(encoder_hidden_states.clone())?;

        // Initial convolution
        let mut sample = self.conv_in.forward(sample.clone())?;

        // Store skip connections
        let mut skip_connections = Vec::new();

        // Down blocks
        for down_block in &self.down_blocks {
            skip_connections.push(sample.clone());
            sample = down_block.forward(&sample, &time_emb, &text_emb)?;
        }

        // Middle block
        sample = self.mid_block.forward(&sample, &time_emb, &text_emb)?;

        // Up blocks
        for (up_block, skip) in self.up_blocks.iter().zip(skip_connections.iter().rev()) {
            sample = Tensor::concat(&[sample.clone(), skip.clone()], 1)?; // Concatenate skip connection
            sample = up_block.forward(&sample, &time_emb, &text_emb)?;
        }

        // Final convolution
        sample = self.conv_out.forward(sample)?;

        Ok(sample)
    }
}

/// U-Net block with attention and time/text conditioning
#[derive(Debug, Clone)]
pub struct DalleUNetBlock {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
    pub time_mlp: Linear,
    pub text_mlp: Linear,
    pub attention: Option<MultiHeadAttention>,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub downsample: Option<Conv2d>,
    device: Device,
}

impl DalleUNetBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        downsample: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_device(in_channels, out_channels, downsample, Device::CPU)
    }

    pub fn new_with_device(
        in_channels: usize,
        out_channels: usize,
        downsample: bool,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let conv1 = Conv2d::new(in_channels, out_channels, (3, 3), (1, 1), (1, 1), false)?;
        let conv2 = Conv2d::new(out_channels, out_channels, (3, 3), (1, 1), (1, 1), false)?;

        let time_mlp = Linear::new(out_channels, out_channels, true);
        let text_mlp = Linear::new(out_channels, out_channels, true);

        let norm1 = LayerNorm::new(vec![out_channels], 1e-5)?;
        let norm2 = LayerNorm::new(vec![out_channels], 1e-5)?;

        let attention = if out_channels >= 512 {
            let attention_config = AttentionConfig {
                hidden_size: out_channels,
                num_heads: 8,
                head_dim: out_channels / 8,
                dropout_prob: 0.0,
                bias: true,
                max_seq_len: None,
            };
            Some(MultiHeadAttention::new(
                attention_config.hidden_size,
                attention_config.num_heads,
                attention_config.dropout_prob,
                attention_config.bias,
            )?)
        } else {
            None
        };

        let downsample = if downsample {
            Some(Conv2d::new(
                out_channels,
                out_channels,
                (3, 3),
                (2, 2),
                (1, 1),
                false,
            )?)
        } else {
            None
        };

        Ok(Self {
            conv1,
            conv2,
            time_mlp,
            text_mlp,
            attention,
            norm1,
            norm2,
            downsample,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(
        &self,
        x: &Tensor,
        time_emb: &Tensor,
        text_emb: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut h = self.conv1.forward(x.clone())?;

        // Add time conditioning
        let time_proj = self.time_mlp.forward(time_emb.clone())?;
        h = (&h + &time_proj.unsqueeze_i64(-1)?.unsqueeze_i64(-1)?)?;

        // Add text conditioning
        let text_proj = self.text_mlp.forward(text_emb.clone())?;
        h = (&h + &text_proj.unsqueeze_i64(-1)?.unsqueeze_i64(-1)?)?;

        h = self.norm1.forward(h)?;
        h = h.gelu()?;

        h = self.conv2.forward(h)?;
        h = self.norm2.forward(h)?;

        // Self-attention if available
        if let Some(attention) = &self.attention {
            let batch_size = h.shape()[0];
            let channels = h.shape()[1];
            let height = h.shape()[2];
            let width = h.shape()[3];

            // Reshape for attention: [B, C, H, W] -> [B, H*W, C]
            let h_attn =
                h.reshape(&[batch_size, channels, height * width])?.transpose_i64(-1, -2)?;
            let attn_out = attention.forward(h_attn)?;
            let h_attn = attn_out
                .transpose_i64(-1, -2)?
                .reshape(&[batch_size, channels, height, width])?;
            h = (&h + &h_attn)?;
        }

        // Residual connection
        h = (&h + x)?;
        h = h.gelu()?;

        // Downsample if needed
        if let Some(downsample) = &self.downsample {
            h = downsample.forward(h)?;
        }

        Ok(h)
    }
}

/// Time embedding for diffusion timesteps
#[derive(Debug, Clone)]
pub struct DalleTimeEmbedding {
    pub linear1: Linear,
    pub linear2: Linear,
    pub embedding_dim: usize,
    device: Device,
}

impl DalleTimeEmbedding {
    pub fn new(embedding_dim: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_device(embedding_dim, Device::CPU)
    }

    pub fn new_with_device(
        embedding_dim: usize,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let linear1 = Linear::new(embedding_dim, embedding_dim * 4, true);
        let linear2 = Linear::new(embedding_dim * 4, embedding_dim, true);

        Ok(Self {
            linear1,
            linear2,
            embedding_dim,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, timestep: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let emb = self.get_timestep_embedding(timestep)?;
        let emb = self.linear1.forward(emb)?;
        let emb = emb.silu()?;
        let emb = self.linear2.forward(emb)?;
        Ok(emb)
    }

    fn get_timestep_embedding(
        &self,
        timestep: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let half_dim = self.embedding_dim / 2;
        let emb = (10000.0_f64).ln() / (half_dim - 1) as f64;

        let mut freqs = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            freqs.push((-emb * i as f64).exp());
        }

        let freqs_tensor = Tensor::from_vec_with_dtype(freqs, &[half_dim], TensorType::F32)?;
        let timestep_f32 = timestep.to_f32()?;
        let args = (&timestep_f32.unsqueeze_i64(-1)? * &freqs_tensor.unsqueeze(0)?)?;

        let sin_emb = args.sin()?;
        let cos_emb = args.cos()?;

        // Concatenate along last axis (axis -1 = axis 1 for 2D tensors)
        let axis = sin_emb.shape().len() - 1;
        let emb = Tensor::concat(&[sin_emb, cos_emb], axis)?;
        Ok(emb)
    }
}

/// VAE for encoding/decoding images to latent space
#[derive(Debug, Clone)]
pub struct DalleVAE {
    pub encoder: DalleVAEEncoder,
    pub decoder: DalleVAEDecoder,
    pub quant_conv: Conv2d,
    pub post_quant_conv: Conv2d,
    device: Device,
}

impl DalleVAE {
    pub fn new(config: DalleImageConfig) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(
        config: DalleImageConfig,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let encoder = DalleVAEEncoder::new_with_device(&config, device)?;
        let decoder = DalleVAEDecoder::new_with_device(&config, device)?;
        let quant_conv = Conv2d::new(
            config.hidden_size,
            config.latent_channels * 2,
            (1, 1),
            (1, 1),
            (0, 0),
            false,
        )?;
        let post_quant_conv = Conv2d::new(
            config.latent_channels,
            config.hidden_size,
            (1, 1),
            (1, 1),
            (0, 0),
            false,
        )?;

        Ok(Self {
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn encode(&self, x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let h = self.encoder.forward(x)?;
        let moments = self.quant_conv.forward(h)?;

        let mean = moments.slice(1, 0, moments.shape()[1] / 2)?;
        let logvar = moments.slice(1, moments.shape()[1] / 2, moments.shape()[1])?;

        // Reparameterization trick
        let half_logvar: Tensor = (&logvar * 0.5)?;
        let std = half_logvar.exp()?;
        let noise = Tensor::randn_like(&mean)?;
        let std_noise = (&std * &noise)?;
        let z = (&mean + &std_noise)?;

        Ok(z)
    }

    pub fn decode(&self, z: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let h = self.post_quant_conv.forward(z.clone())?;
        let x = self.decoder.forward(&h)?;
        Ok(x)
    }
}

/// VAE encoder
#[derive(Debug, Clone)]
pub struct DalleVAEEncoder {
    pub conv_in: Conv2d,
    pub down_blocks: Vec<Conv2d>,
    pub mid_block: Conv2d,
    pub norm_out: LayerNorm,
    pub conv_out: Conv2d,
    device: Device,
}

impl DalleVAEEncoder {
    pub fn new(config: &DalleImageConfig) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(
        config: &DalleImageConfig,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let conv_in = Conv2d::new(config.num_channels, 128, (3, 3), (1, 1), (1, 1), false)?;

        let mut down_blocks = Vec::new();
        let channels = [128, 256, 512, 512];
        for i in 0..channels.len() - 1 {
            down_blocks.push(Conv2d::new(
                channels[i],
                channels[i + 1],
                (3, 3),
                (2, 2),
                (1, 1),
                false,
            )?);
        }

        let mid_block = Conv2d::new(512, config.hidden_size, (3, 3), (1, 1), (1, 1), false)?;
        let norm_out = LayerNorm::new(vec![config.hidden_size], 1e-5)?;
        let conv_out = Conv2d::new(
            config.hidden_size,
            config.hidden_size,
            (3, 3),
            (1, 1),
            (1, 1),
            false,
        )?;

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            norm_out,
            conv_out,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut h = self.conv_in.forward(x.clone())?;
        h = h.relu()?;

        for down_block in &self.down_blocks {
            h = down_block.forward(h)?;
            h = h.relu()?;
        }

        h = self.mid_block.forward(h)?;
        h = self.norm_out.forward(h)?;
        h = h.swish()?;
        h = self.conv_out.forward(h)?;

        Ok(h)
    }
}

/// VAE decoder
#[derive(Debug, Clone)]
pub struct DalleVAEDecoder {
    pub conv_in: Conv2d,
    pub mid_block: Conv2d,
    pub up_blocks: Vec<Conv2d>,
    pub norm_out: LayerNorm,
    pub conv_out: Conv2d,
    device: Device,
}

impl DalleVAEDecoder {
    pub fn new(config: &DalleImageConfig) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(
        config: &DalleImageConfig,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let conv_in = Conv2d::new(
            config.hidden_size,
            config.hidden_size,
            (3, 3),
            (1, 1),
            (1, 1),
            false,
        )?;
        let mid_block = Conv2d::new(config.hidden_size, 512, (3, 3), (1, 1), (1, 1), false)?;

        let mut up_blocks = Vec::new();
        let channels = [512, 512, 256, 128];
        for i in 0..channels.len() - 1 {
            // Note: This is simplified - real implementation would use transpose convolution
            up_blocks.push(Conv2d::new(
                channels[i],
                channels[i + 1],
                (3, 3),
                (1, 1),
                (1, 1),
                false,
            )?);
        }

        let norm_out = LayerNorm::new(vec![128], 1e-5)?;
        let conv_out = Conv2d::new(128, config.num_channels, (3, 3), (1, 1), (1, 1), false)?;

        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            norm_out,
            conv_out,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, z: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut h = self.conv_in.forward(z.clone())?;
        h = h.swish()?;
        h = self.mid_block.forward(h)?;
        h = h.relu()?;

        for up_block in &self.up_blocks {
            // Upsample using bilinear interpolation
            let h_shape = h.shape();
            let new_h = h_shape[2] * 2;
            let new_w = h_shape[3] * 2;
            h = h.interpolate((new_h, new_w))?;
            h = up_block.forward(h)?;
            h = h.relu()?;
        }

        h = self.norm_out.forward(h)?;
        h = h.swish()?;
        h = self.conv_out.forward(h)?;
        h = h.tanh()?; // Map to [-1, 1]

        Ok(h)
    }
}

/// MLP layer used in various components
#[derive(Debug, Clone)]
pub struct DalleMLP {
    pub fc1: Linear,
    pub fc2: Linear,
    pub activation: ActivationType,
    device: Device,
}

impl DalleMLP {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        activation: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_device(hidden_size, intermediate_size, activation, Device::CPU)
    }

    pub fn new_with_device(
        hidden_size: usize,
        intermediate_size: usize,
        activation: &str,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let fc1 = Linear::new(hidden_size, intermediate_size, true);
        let fc2 = Linear::new(intermediate_size, hidden_size, true);
        let activation = match activation {
            "relu" => ActivationType::ReLU,
            "gelu" | "quick_gelu" => ActivationType::GELU,
            "silu" | "swish" => ActivationType::SiLU,
            "tanh" => ActivationType::Tanh,
            "sigmoid" => ActivationType::Sigmoid,
            _ => ActivationType::GELU, // Default
        };

        Ok(Self {
            fc1,
            fc2,
            activation,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let x = self.fc1.forward(x.clone())?;
        let x = match self.activation {
            ActivationType::ReLU => x.relu()?,
            ActivationType::GELU => x.gelu()?,
            ActivationType::SiLU => x.silu()?,
            ActivationType::Tanh => x.tanh()?,
            ActivationType::Sigmoid => x.sigmoid()?,
        };
        let x = self.fc2.forward(x)?;
        Ok(x)
    }
}

/// Output of DALL-E model
#[derive(Debug, Clone)]
pub struct DalleModelOutput {
    pub text_embeds: Option<Tensor>,
    pub image_embeds: Option<Tensor>,
    pub logits_per_image: Option<Tensor>,
    pub logits_per_text: Option<Tensor>,
    pub latents: Option<Tensor>,
    pub noise_pred: Option<Tensor>,
    pub noise_pred_target: Option<Tensor>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Heavy test - large model creation, run with --ignored
    fn test_dalle_model_creation() {
        let config = DalleConfig::dalle_mini();
        let model = DalleModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    #[ignore] // Heavy test - large encoder, run with --ignored
    fn test_dalle_text_encoder() {
        let config = DalleTextConfig::clip_base();
        let encoder = DalleTextEncoder::new(config).expect("operation failed");

        let batch_size = 2;
        let seq_len = 77;
        let input_ids = Tensor::randint(0, 1000, &[batch_size, seq_len], TensorType::I64)
            .expect("operation failed");
        let attention_mask = Tensor::ones(&[batch_size, seq_len]).expect("operation failed");

        let output = encoder.forward(&input_ids, &attention_mask);
        assert!(output.is_ok());

        let output = output.expect("operation failed");
        assert_eq!(output.shape(), &[batch_size, encoder.config.hidden_size]);
    }

    #[test]
    #[ignore] // Heavy test - large encoder, run with --ignored
    fn test_dalle_image_encoder() {
        let config = DalleVisionConfig::clip_vit_b();
        let encoder = DalleImageEncoder::new(config).expect("operation failed");

        let batch_size = 2;
        let pixel_values = Tensor::randn(&[batch_size, 3, 224, 224]).expect("operation failed");

        let output = encoder.forward(&pixel_values);
        assert!(output.is_ok());

        let output = output.expect("operation failed");
        assert_eq!(output.shape(), &[batch_size, encoder.config.hidden_size]);
    }

    #[test]
    #[ignore] // Heavy test - VAE requires significant memory, run with --ignored
    fn test_dalle_vae() {
        let config = DalleImageConfig::dalle_mini();
        let vae = DalleVAE::new(config.clone()).expect("operation failed");

        let batch_size = 1;
        let images = Tensor::randn(&[batch_size, 3, 256, 256]).expect("operation failed");

        // Test encoding
        let latents = vae.encode(&images);
        assert!(latents.is_ok());
        let latents = latents.expect("operation failed");
        assert_eq!(latents.shape()[0], batch_size);
        assert_eq!(latents.shape()[1], config.latent_channels);

        // Test decoding
        let reconstructed = vae.decode(&latents);
        assert!(reconstructed.is_ok());
        let reconstructed = reconstructed.expect("operation failed");
        assert_eq!(reconstructed.shape(), images.shape());
    }

    #[test]
    #[ignore] // Heavy test - UNet requires significant memory, run with --ignored
    fn test_dalle_unet() {
        let image_config = DalleImageConfig::dalle_mini();
        let diffusion_config = DalleDiffusionConfig::dalle_mini();
        let unet =
            DalleUNet::new(image_config.clone(), diffusion_config).expect("operation failed");

        let batch_size = 1;
        let latents = Tensor::randn(&[
            batch_size,
            image_config.latent_channels,
            image_config.latent_size(),
            image_config.latent_size(),
        ])
        .expect("operation failed");
        let timestep =
            Tensor::randint(0, 1000, &[batch_size], TensorType::I64).expect("operation failed");
        let text_embeds =
            Tensor::randn(&[batch_size, image_config.hidden_size]).expect("operation failed");

        let output = unet.forward(&latents, &timestep, &text_embeds);
        assert!(output.is_ok());

        let output = output.expect("operation failed");
        assert_eq!(output.shape(), latents.shape());
    }

    #[test]
    fn test_time_embedding() {
        let embedding_dim = 512;
        let time_emb = DalleTimeEmbedding::new(embedding_dim).expect("operation failed");

        let batch_size = 2;
        let timestep =
            Tensor::randint(0, 1000, &[batch_size], TensorType::I64).expect("operation failed");

        let output = time_emb.forward(&timestep);
        assert!(output.is_ok());

        let output = output.expect("operation failed");
        assert_eq!(output.shape(), &[batch_size, embedding_dim]);
    }

    #[test]
    #[ignore] // Very heavy test - full generation pipeline, run with --ignored
    fn test_dalle_generation_pipeline() {
        let config = DalleConfig::dalle_mini();
        let model = DalleModel::new(config.clone()).expect("operation failed");

        let batch_size = 1;
        let seq_len = 77;
        let input_ids = Tensor::randint(0, 1000, &[batch_size, seq_len], TensorType::I64)
            .expect("operation failed");
        let attention_mask = Tensor::ones(&[batch_size, seq_len]).expect("operation failed");

        // Test generation (simplified)
        let result = model.generate(&input_ids, &attention_mask, Some(10), Some(5.0), Some(42));
        assert!(result.is_ok());

        let images = result.expect("operation failed");
        assert_eq!(images.shape()[0], batch_size);
        assert_eq!(images.shape()[1], 3); // RGB channels
        assert_eq!(images.shape()[2], config.image_size);
        assert_eq!(images.shape()[3], config.image_size);
    }

    #[test]
    #[ignore] // Very heavy test - training forward pass, run with --ignored
    fn test_dalle_training_forward() {
        let config = DalleConfig::dalle_mini();
        let model = DalleModel::new(config.clone()).expect("operation failed");

        let batch_size = 1;
        let seq_len = 77;
        let input_ids = Tensor::randint(0, 1000, &[batch_size, seq_len], TensorType::I64)
            .expect("operation failed");
        let attention_mask = Tensor::ones(&[batch_size, seq_len]).expect("operation failed");
        let pixel_values = Tensor::randn(&[batch_size, 3, config.image_size, config.image_size])
            .expect("operation failed");

        let output = model.forward_train(&input_ids, &attention_mask, &pixel_values, None, None);
        assert!(output.is_ok());

        let output = output.expect("operation failed");
        assert!(output.text_embeds.is_some());
        assert!(output.image_embeds.is_some());
        assert!(output.logits_per_image.is_some());
        assert!(output.logits_per_text.is_some());
        assert!(output.latents.is_some());
        assert!(output.noise_pred.is_some());
        assert!(output.noise_pred_target.is_some());
    }
}
