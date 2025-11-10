/// DALL-E model configurations
pub mod config;

/// DALL-E model implementations
pub mod model;

// Re-export main types for convenience
pub use config::{
    DalleConfig, DalleDiffusionConfig, DalleImageConfig, DalleTextConfig, DalleVisionConfig,
};

pub use model::{
    DalleImageEncoder, DalleMLP, DalleModel, DalleModelOutput, DalleTextEncoder,
    DalleTimeEmbedding, DalleUNet, DalleVAE,
};

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::{Tensor, TensorType};

    #[test]
    fn test_dalle_module_imports() {
        // Test that all main types can be imported
        let _config = DalleConfig::default();
        let _text_config = DalleTextConfig::default();
        let _image_config = DalleImageConfig::default();
        let _vision_config = DalleVisionConfig::default();
        let _diffusion_config = DalleDiffusionConfig::default();

        // Test model creation
        let config = DalleConfig::dalle_mini();
        let model = DalleModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_dalle_mini_end_to_end() {
        let config = DalleConfig::dalle_mini();
        let model = DalleModel::new(config.clone()).unwrap();

        // Create sample inputs
        let batch_size = 1;
        let seq_len = 77;
        let input_ids = Tensor::randint(
            0,
            config.text_vocab_size as i64,
            &[batch_size, seq_len],
            TensorType::I64,
        )
        .unwrap();
        let attention_mask = Tensor::ones(&[batch_size, seq_len]).unwrap();
        let pixel_values =
            Tensor::randn(&[batch_size, 3, config.image_size, config.image_size]).unwrap();

        // Test training forward pass
        let train_output =
            model.forward_train(&input_ids, &attention_mask, &pixel_values, None, None);
        assert!(train_output.is_ok());

        let train_output = train_output.unwrap();

        // Verify outputs have correct shapes
        assert!(train_output.text_embeds.is_some());
        assert!(train_output.image_embeds.is_some());
        assert!(train_output.logits_per_image.is_some());
        assert!(train_output.logits_per_text.is_some());
        assert!(train_output.latents.is_some());
        assert!(train_output.noise_pred.is_some());
        assert!(train_output.noise_pred_target.is_some());

        let text_embeds = train_output.text_embeds.unwrap();
        let image_embeds = train_output.image_embeds.unwrap();
        let logits_per_image = train_output.logits_per_image.unwrap();
        let latents = train_output.latents.unwrap();
        let noise_pred = train_output.noise_pred.unwrap();

        // Check tensor shapes
        assert_eq!(text_embeds.shape()[0], batch_size);
        assert_eq!(image_embeds.shape()[0], batch_size);
        assert_eq!(logits_per_image.shape()[0], batch_size);
        assert_eq!(logits_per_image.shape()[1], batch_size);
        assert_eq!(latents.shape()[0], batch_size);
        assert_eq!(latents.shape()[1], config.image_config.latent_channels);
        assert_eq!(noise_pred.shape(), latents.shape());

        // Test inference generation
        let generated_images =
            model.generate(&input_ids, &attention_mask, Some(5), Some(3.0), Some(42));
        assert!(generated_images.is_ok());

        let generated_images = generated_images.unwrap();
        assert_eq!(
            generated_images.shape(),
            &[batch_size, 3, config.image_size, config.image_size]
        );
    }

    #[test]
    fn test_dalle_2_configuration() {
        let config = DalleConfig::dalle_2();
        let model = DalleModel::new(config.clone()).unwrap();

        // Verify configuration parameters
        assert_eq!(config.text_vocab_size, 49408); // CLIP vocab
        assert_eq!(config.image_size, 512);
        assert_eq!(config.text_config.context_length, 77);
        assert_eq!(config.image_config.latent_channels, 4);
        assert_eq!(config.diffusion_config.num_timesteps, 1000);
        assert!(config.use_cross_attention);
        assert!(config.use_clip_loss);

        // Test with larger inputs
        let batch_size = 2;
        let seq_len = config.text_config.context_length;
        let input_ids = Tensor::randint(
            0,
            config.text_vocab_size as i64,
            &[batch_size, seq_len],
            TensorType::I64,
        )
        .unwrap();
        let attention_mask = Tensor::ones(&[batch_size, seq_len]).unwrap();

        // Test text encoding
        let text_features = model.text_encoder.forward(&input_ids, &attention_mask);
        assert!(text_features.is_ok());

        let text_features = text_features.unwrap();
        assert_eq!(text_features.shape()[0], batch_size);
        assert_eq!(text_features.shape()[1], config.text_config.hidden_size);
    }

    #[test]
    fn test_dalle_3_configuration() {
        let config = DalleConfig::dalle_3();

        // Verify DALL-E 3 specific configurations
        assert_eq!(config.text_vocab_size, 32128); // T5 vocab
        assert_eq!(config.image_size, 1024);
        assert_eq!(config.text_config.vocab_size, 32128);
        assert_eq!(config.text_config.hidden_size, 4096);
        assert_eq!(config.vision_config.hidden_size, 1664); // ViT-G
        assert_eq!(config.image_config.hidden_size, 1536);
        assert_eq!(config.guidance_scale, 10.0);
        assert!(config.diffusion_config.learned_variance);
        assert!(config.diffusion_config.v_parameterization);

        // Test model creation (might be computationally expensive)
        let model = DalleModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_dalle_components_separately() {
        let config = DalleConfig::dalle_mini();

        // Test text encoder
        let text_encoder = DalleTextEncoder::new(config.text_config.clone());
        assert!(text_encoder.is_ok());

        // Test image encoder
        let image_encoder = DalleImageEncoder::new(config.vision_config.clone());
        assert!(image_encoder.is_ok());

        // Test U-Net
        let unet = DalleUNet::new(config.image_config.clone(), config.diffusion_config.clone());
        assert!(unet.is_ok());

        // Test VAE
        let vae = DalleVAE::new(config.image_config.clone());
        assert!(vae.is_ok());

        // Test time embedding
        let time_emb = DalleTimeEmbedding::new(512);
        assert!(time_emb.is_ok());
    }

    #[test]
    fn test_dalle_config_serialization() {
        let configs = vec![
            DalleConfig::dalle_mini(),
            DalleConfig::dalle_2(),
            DalleConfig::dalle_3(),
        ];

        for config in configs {
            // Test JSON serialization
            let json = serde_json::to_string(&config);
            assert!(json.is_ok());

            let json_str = json.unwrap();
            let deserialized: Result<DalleConfig, _> = serde_json::from_str(&json_str);
            assert!(deserialized.is_ok());

            let deserialized = deserialized.unwrap();

            // Verify key fields are preserved
            assert_eq!(config.text_vocab_size, deserialized.text_vocab_size);
            assert_eq!(config.image_size, deserialized.image_size);
            assert_eq!(config.guidance_scale, deserialized.guidance_scale);
            assert_eq!(config.use_cross_attention, deserialized.use_cross_attention);
            assert_eq!(
                config.text_config.vocab_size,
                deserialized.text_config.vocab_size
            );
            assert_eq!(
                config.image_config.latent_channels,
                deserialized.image_config.latent_channels
            );
            assert_eq!(
                config.diffusion_config.num_timesteps,
                deserialized.diffusion_config.num_timesteps
            );
        }
    }

    #[test]
    fn test_dalle_model_output_structure() {
        let config = DalleConfig::dalle_mini();
        let model = DalleModel::new(config.clone()).unwrap();

        let batch_size = 1;
        let seq_len = 77;
        let input_ids = Tensor::randint(
            0,
            config.text_vocab_size as i64,
            &[batch_size, seq_len],
            TensorType::I64,
        )
        .unwrap();
        let attention_mask = Tensor::ones(&[batch_size, seq_len]).unwrap();
        let pixel_values =
            Tensor::randn(&[batch_size, 3, config.image_size, config.image_size]).unwrap();

        let output = model
            .forward_train(&input_ids, &attention_mask, &pixel_values, None, None)
            .unwrap();

        // Verify all expected outputs are present
        assert!(output.text_embeds.is_some());
        assert!(output.image_embeds.is_some());
        assert!(output.logits_per_image.is_some());
        assert!(output.logits_per_text.is_some());
        assert!(output.latents.is_some());
        assert!(output.noise_pred.is_some());
        assert!(output.noise_pred_target.is_some());

        // Verify contrastive logits are symmetric
        let logits_per_image = output.logits_per_image.unwrap();
        let logits_per_text = output.logits_per_text.unwrap();

        assert_eq!(logits_per_image.shape()[0], batch_size);
        assert_eq!(logits_per_image.shape()[1], batch_size);
        assert_eq!(logits_per_text.shape()[0], batch_size);
        assert_eq!(logits_per_text.shape()[1], batch_size);
    }
}
