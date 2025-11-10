pub mod config;
pub mod model;

pub use config::*;
pub use model::*;

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::Tensor;

    #[test]
    fn test_blip2_model_creation() {
        let config = Blip2Config::default();
        let model = Blip2Model::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_blip2_for_conditional_generation() {
        let config = Blip2Config::default();
        let model = Blip2ForConditionalGeneration::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_blip2_vision_model() {
        let config = Blip2VisionConfig::default();
        let model = Blip2VisionModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_blip2_qformer() {
        let config = Blip2QFormerConfig::default();
        let model = Blip2QFormerModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_blip2_forward() {
        let config = Blip2Config::default();
        let model = Blip2ForConditionalGeneration::new(config).unwrap();

        // Create dummy inputs
        let batch_size = 1;
        let seq_len = 10;
        let image_size = 224;
        let channels = 3;

        let input_ids =
            Tensor::from_vec(vec![1.0; batch_size * seq_len], &[batch_size, seq_len]).unwrap();

        let pixel_values = Tensor::from_vec(
            vec![0.5; batch_size * channels * image_size * image_size],
            &[batch_size, channels, image_size, image_size],
        )
        .unwrap();

        let result = model.forward(&input_ids, Some(&pixel_values), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_blip2_text_generation() {
        let config = Blip2Config::default();
        let model = Blip2ForConditionalGeneration::new(config).unwrap();

        let pixel_values = Tensor::from_vec(vec![0.5; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();

        let result = model.generate(&pixel_values, None, 50, 1.0, 0.9);
        assert!(result.is_ok());
    }

    #[test]
    fn test_blip2_vision_text_similarity() {
        let config = Blip2Config::default();
        let model = Blip2Model::new(config).unwrap();

        let pixel_values = Tensor::from_vec(vec![0.5; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();

        let input_ids = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5]).unwrap();

        let result = model.get_text_features(&input_ids);
        assert!(result.is_ok());

        let result = model.get_image_features(&pixel_values);
        assert!(result.is_ok());
    }

    #[test]
    fn test_blip2_config_serialization() {
        let config = Blip2Config::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: Blip2Config = serde_json::from_str(&serialized).unwrap();
        assert_eq!(
            config.vision_config.image_size,
            deserialized.vision_config.image_size
        );
        assert_eq!(
            config.qformer_config.hidden_size,
            deserialized.qformer_config.hidden_size
        );
    }

    #[test]
    fn test_blip2_model_sizes() {
        let configs = vec![
            Blip2Config::opt_2_7b(),
            Blip2Config::opt_6_7b(),
            Blip2Config::flan_t5_xl(),
            Blip2Config::flan_t5_xxl(),
        ];

        for config in configs {
            let model = Blip2ForConditionalGeneration::new(config);
            assert!(model.is_ok());
        }
    }

    #[test]
    fn test_blip2_attention_mechanism() {
        let config = Blip2Config::default();
        let model = Blip2Model::new(config).unwrap();

        let pixel_values = Tensor::from_vec(vec![0.5; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();

        let input_ids = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5]).unwrap();

        let result = model.forward(&input_ids, Some(&pixel_values), None);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.logits.shape().len() >= 2);
    }

    #[test]
    fn test_blip2_vision_transformer() {
        let config = Blip2VisionConfig::default();
        let model = Blip2VisionModel::new(config).unwrap();

        let pixel_values = Tensor::from_vec(vec![0.5; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();

        let result = model.forward(&pixel_values);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.shape().len() >= 2);
    }

    #[test]
    fn test_blip2_qformer_model() {
        let config = Blip2QFormerConfig::default();
        let model = Blip2QFormerModel::new(config).unwrap();

        let input_ids = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5]).unwrap();

        // Vision model outputs 1408-dim (not 768), so encoder states should match
        let encoder_hidden_states = Tensor::from_vec(
            vec![0.5; 197 * 1408], // 197 = 1 + 14*14 (CLS + patches), 1408 = vision dim
            &[1, 197, 1408],
        )
        .unwrap();

        let result = model.forward(&input_ids, Some(&encoder_hidden_states), None, None);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.last_hidden_state.shape().len() >= 2);
    }
}
