#[cfg(test)]
mod tests {
    use crate::vit::config::ViTConfig;
    use crate::vit::model::{PatchEmbedding, ViTForImageClassification, ViTModel};
    use ndarray::Array4;
    use trustformers_core::traits::Config;

    #[test]
    fn test_vit_config() {
        let config = ViTConfig::base();
        assert_eq!(config.image_size, 224);
        assert_eq!(config.patch_size, 16);
        assert_eq!(config.num_patches(), 196); // (224/16)^2
        assert_eq!(config.seq_length(), 197); // 196 patches + 1 class token

        config.validate().unwrap();
    }

    #[test]
    fn test_vit_config_variants() {
        let tiny = ViTConfig::tiny();
        assert_eq!(tiny.hidden_size, 192);
        assert_eq!(tiny.num_attention_heads, 3);

        let small = ViTConfig::small();
        assert_eq!(small.hidden_size, 384);
        assert_eq!(small.num_attention_heads, 6);

        let large = ViTConfig::large();
        assert_eq!(large.hidden_size, 1024);
        assert_eq!(large.num_attention_heads, 16);
    }

    #[test]
    fn test_patch_embedding() {
        // Use minimal config to reduce memory usage
        let config = ViTConfig {
            image_size: 32,
            patch_size: 16,
            hidden_size: 64,
            num_attention_heads: 4,
            intermediate_size: 256,
            num_hidden_layers: 2,
            ..ViTConfig::default()
        };
        let patch_embedding = PatchEmbedding::new(&config);

        // Test with a small 32x32x3 image to reduce memory usage
        let image = Array4::zeros((1, 32, 32, 3));
        let result = patch_embedding.forward(&image);

        assert!(result.is_ok());
        let patches = result.unwrap();
        assert_eq!(patches.shape(), &[1, 4, 64]); // 1 batch, 4 patches (32/16)^2, 64 hidden

        // Explicit cleanup
        drop(patches);
        drop(patch_embedding);
        std::hint::black_box(());
    }

    #[test]
    fn test_patch_embedding_different_sizes() {
        // Use minimal config to reduce memory usage
        let mut config = ViTConfig {
            image_size: 32,
            patch_size: 16,
            hidden_size: 64,
            num_attention_heads: 4,
            intermediate_size: 256,
            num_hidden_layers: 2,
            ..ViTConfig::default()
        };
        config.patch_size = 16;
        config.image_size = 32;

        let patch_embedding = PatchEmbedding::new(&config);

        // Test with small 32x32x3 image and 16x16 patches to reduce memory usage
        let image = Array4::zeros((1, 32, 32, 3));
        let result = patch_embedding.forward(&image).unwrap();

        let expected_patches = (32 / 16) * (32 / 16); // 2x2 = 4 patches
        assert_eq!(result.shape(), &[1, expected_patches, 64]);

        // Explicit cleanup
        drop(result);
        drop(patch_embedding);
        std::hint::black_box(());
    }

    #[test]
    fn test_vit_model() {
        // Use even smaller config to reduce memory usage
        let config = ViTConfig {
            image_size: 32,
            patch_size: 16,
            hidden_size: 64,
            num_attention_heads: 4,
            intermediate_size: 256,
            num_hidden_layers: 2,
            ..ViTConfig::default()
        };
        let model = ViTModel::new(config).unwrap();

        // Test with a single small image to reduce memory usage
        let images = Array4::zeros((1, 32, 32, 3));
        let result = model.forward(&images);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), &[1, 5, 64]); // batch, seq_len (4 patches + 1 class token), hidden_size

        // Explicit cleanup
        drop(output);
        drop(model);
        std::hint::black_box(());
    }

    #[test]
    fn test_vit_classification() {
        // Use smaller config and batch size to reduce memory usage
        let config = ViTConfig {
            image_size: 32,
            patch_size: 16,
            hidden_size: 64,
            num_attention_heads: 4,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_labels: 10, // Reduced from 1000
            ..ViTConfig::default()
        };
        let model = ViTForImageClassification::new(config).unwrap();

        let images = Array4::zeros((1, 32, 32, 3));
        let result = model.forward(&images);

        assert!(result.is_ok());
        let logits = result.unwrap();
        assert_eq!(logits.shape(), &[1, 10]); // batch_size, num_classes

        // Explicit cleanup
        drop(logits);
        drop(model);
        std::hint::black_box(());
    }

    #[test]
    fn test_vit_class_token_output() {
        // Use smaller config to reduce memory usage
        let config = ViTConfig {
            image_size: 32,
            patch_size: 16,
            hidden_size: 64,
            num_attention_heads: 4,
            intermediate_size: 256,
            num_hidden_layers: 2,
            ..ViTConfig::default()
        };
        let model = ViTModel::new(config).unwrap();

        let images = Array4::zeros((1, 32, 32, 3));
        let result = model.get_class_token_output(&images);

        assert!(result.is_ok());
        let class_output = result.unwrap();
        assert_eq!(class_output.shape(), &[1, 64]); // batch_size, hidden_size

        // Explicit cleanup
        drop(class_output);
        drop(model);
        std::hint::black_box(());
    }

    #[test]
    fn test_vit_without_class_token() {
        // Use smaller config to reduce memory usage
        let mut config = ViTConfig {
            image_size: 32,
            patch_size: 16,
            hidden_size: 64,
            num_attention_heads: 4,
            intermediate_size: 256,
            num_hidden_layers: 2,
            ..ViTConfig::default()
        };
        config.use_class_token = false;

        let model = ViTModel::new(config).unwrap();

        let images = Array4::zeros((1, 32, 32, 3));
        let output = model.forward(&images).unwrap();

        // Should have 4 patches (no class token) (32/16)^2
        assert_eq!(output.shape(), &[1, 4, 64]);

        // Class token output should be mean of patches
        let class_output = model.get_class_token_output(&images).unwrap();
        assert_eq!(class_output.shape(), &[1, 64]);

        // Explicit cleanup
        drop(output);
        drop(class_output);
        drop(model);
        std::hint::black_box(());
    }

    #[test]
    fn test_from_pretrained_name() {
        let base = ViTConfig::from_pretrained_name("vit-base-patch16-224");
        assert_eq!(base.hidden_size, 768);

        let large = ViTConfig::from_pretrained_name("vit-large-patch16-224");
        assert_eq!(large.hidden_size, 1024);

        let tiny = ViTConfig::from_pretrained_name("vit-tiny-patch16-224");
        assert_eq!(tiny.hidden_size, 192);
    }

    #[test]
    fn test_config_validation_errors() {
        let mut config = ViTConfig::base();

        // Test invalid hidden_size / num_attention_heads
        config.hidden_size = 100;
        config.num_attention_heads = 12;
        assert!(config.validate().is_err());

        // Test invalid image_size / patch_size
        config = ViTConfig::base();
        config.image_size = 225; // Not divisible by 16
        assert!(config.validate().is_err());

        // Test zero patch_size
        config = ViTConfig::base();
        config.patch_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_with_different_patch_sizes() {
        let base = ViTConfig::base();
        let patch32 = base.with_patch_size(32);

        assert_eq!(patch32.patch_size, 32);
        assert_eq!(patch32.encoder_stride, 32);
        assert_eq!(patch32.num_patches(), 49); // (224/32)^2
    }

    #[test]
    fn test_config_with_different_image_sizes() {
        let base = ViTConfig::base();
        let img384 = base.with_image_size(384);

        assert_eq!(img384.image_size, 384);
        assert_eq!(img384.num_patches(), 576); // (384/16)^2
    }
}
