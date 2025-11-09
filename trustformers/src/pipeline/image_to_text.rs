use crate::core::traits::Tokenizer;
use crate::error::{Result, TrustformersError};
use crate::pipeline::{BasePipeline, Pipeline, PipelineOutput};
use crate::AutoModel;
use crate::AutoTokenizer;

#[cfg(feature = "vision")]
use image::DynamicImage;
#[cfg(feature = "vision")]
use std::path::Path;
use trustformers_core::cache::CacheKeyBuilder;
use trustformers_core::tensor::Tensor;

#[cfg(feature = "vision")]
/// Pipeline for image-to-text tasks (image captioning, VQA)
#[derive(Clone)]
pub struct ImageToTextPipeline {
    base: BasePipeline<AutoModel, AutoTokenizer>,
    max_new_tokens: usize,
    temperature: f32,
    do_sample: bool,
}

#[cfg(feature = "vision")]
impl ImageToTextPipeline {
    pub fn new(model: AutoModel, tokenizer: AutoTokenizer) -> Result<Self> {
        Ok(Self {
            base: BasePipeline::new(model, tokenizer),
            max_new_tokens: 50,
            temperature: 1.0,
            do_sample: true,
        })
    }

    pub fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_sampling(mut self, do_sample: bool) -> Self {
        self.do_sample = do_sample;
        self
    }

    /// Generate text from image
    fn generate_from_image(&self, input: &ImageToTextInput) -> Result<ImageToTextOutput> {
        // Check cache if enabled
        if let Some(cache) = &self.base.cache {
            let cache_key = CacheKeyBuilder::new("image-to-text", "generate")
                .with_param("max_new_tokens", &self.max_new_tokens)
                .with_param("temperature", &self.temperature.to_string())
                .with_param("do_sample", &self.do_sample)
                .build();

            if let Some(cached_data) = cache.get(&cache_key) {
                if let Ok(result) = serde_json::from_slice::<ImageToTextOutput>(&cached_data) {
                    return Ok(result);
                }
            }
        }

        // Process image
        let image_features = self.process_image(&input.image)?;

        // Generate text based on the task
        let generated_text = match &input.text_prompt {
            Some(prompt) => {
                // VQA or prompted generation
                self.generate_with_prompt(&image_features, prompt)?
            },
            None => {
                // Image captioning
                self.generate_caption(&image_features)?
            },
        };

        let result = ImageToTextOutput {
            generated_text: generated_text.clone(),
            image_features: Some(image_features),
            confidence: 0.95, // Placeholder confidence score
        };

        // Cache result if enabled
        if let Some(cache) = &self.base.cache {
            let cache_key = CacheKeyBuilder::new("image-to-text", "generate")
                .with_param("max_new_tokens", &self.max_new_tokens)
                .with_param("temperature", &self.temperature.to_string())
                .with_param("do_sample", &self.do_sample)
                .build();

            if let Ok(serialized) = serde_json::to_vec(&result) {
                cache.insert(cache_key, serialized);
            }
        }

        Ok(result)
    }

    /// Process image and extract features
    fn process_image(&self, image: &DynamicImage) -> Result<Tensor> {
        // Resize image to model input size (typically 224x224 for vision models)
        let target_size = 224;
        let resized = image.resize_exact(
            target_size,
            target_size,
            image::imageops::FilterType::Lanczos3,
        );

        // Convert to RGB if needed
        let rgb_image = resized.to_rgb8();

        // Normalize pixel values to [0, 1] range
        let mut pixel_values = Vec::new();
        for pixel in rgb_image.pixels() {
            pixel_values.push(pixel[0] as f32 / 255.0); // R
            pixel_values.push(pixel[1] as f32 / 255.0); // G
            pixel_values.push(pixel[2] as f32 / 255.0); // B
        }

        // Create tensor with shape [1, 3, 224, 224] (batch, channels, height, width)
        let tensor = Tensor::from_vec(
            pixel_values,
            &[1, 3, target_size as usize, target_size as usize],
        )?;

        Ok(tensor)
    }

    /// Generate caption for image
    fn generate_caption(&self, image_features: &Tensor) -> Result<String> {
        // In a real implementation, this would:
        // 1. Pass image features through a vision encoder
        // 2. Use a decoder model to generate text
        // 3. Apply beam search or sampling for generation

        // For now, return a placeholder caption
        Ok("A photo showing various objects and scenes.".to_string())
    }

    /// Generate text with a prompt (for VQA or guided generation)
    fn generate_with_prompt(&self, image_features: &Tensor, prompt: &str) -> Result<String> {
        // Tokenize the text prompt
        let prompt_tokens = self.base.tokenizer.encode(prompt)?;

        // In a real implementation:
        // 1. Combine image features with text prompt
        // 2. Use multimodal model to generate response
        // 3. Apply appropriate decoding strategy

        // For now, return a contextual response
        let response = if prompt.to_lowercase().contains("what") {
            "This appears to be an image containing various visual elements."
        } else if prompt.to_lowercase().contains("where") {
            "This scene appears to be taken in an indoor/outdoor setting."
        } else if prompt.to_lowercase().contains("how many") {
            "There appear to be several items in the image."
        } else {
            "Based on the image content, this appears to be a relevant response to your question."
        };

        Ok(response.to_string())
    }
}

#[cfg(feature = "vision")]
impl Pipeline for ImageToTextPipeline {
    type Input = ImageToTextInput;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        let result = self.generate_from_image(&input)?;
        Ok(PipelineOutput::ImageToText(result))
    }
}

#[cfg(feature = "vision")]
/// Input for image-to-text pipeline
#[derive(Debug, Clone)]
pub struct ImageToTextInput {
    pub image: DynamicImage,
    pub text_prompt: Option<String>, // For VQA or guided generation
}

#[cfg(feature = "vision")]
impl ImageToTextInput {
    /// Create from image file path
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let image =
            image::open(path).map_err(|e| TrustformersError::pipeline(e.to_string(), "runtime"))?;

        Ok(Self {
            image,
            text_prompt: None,
        })
    }

    /// Create from image file path with text prompt (for VQA)
    pub fn from_path_with_prompt<P: AsRef<Path>>(path: P, prompt: String) -> Result<Self> {
        let image =
            image::open(path).map_err(|e| TrustformersError::pipeline(e.to_string(), "runtime"))?;

        Ok(Self {
            image,
            text_prompt: Some(prompt),
        })
    }

    /// Create from image data
    pub fn from_image(image: DynamicImage) -> Self {
        Self {
            image,
            text_prompt: None,
        }
    }

    /// Create from image data with text prompt
    pub fn from_image_with_prompt(image: DynamicImage, prompt: String) -> Self {
        Self {
            image,
            text_prompt: Some(prompt),
        }
    }
}

#[cfg(feature = "vision")]
/// Output for image-to-text pipeline
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImageToTextOutput {
    pub generated_text: String,
    #[serde(skip)] // Skip serialization for cache
    pub image_features: Option<Tensor>,
    pub confidence: f32,
}

#[cfg(all(test, feature = "vision"))]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn test_image_to_text_input_creation() {
        // Create a simple test image
        let img = RgbImage::new(100, 100);
        let dynamic_img = DynamicImage::ImageRgb8(img);

        let input = ImageToTextInput::from_image(dynamic_img);
        assert!(input.text_prompt.is_none());

        let input_with_prompt = ImageToTextInput::from_image_with_prompt(
            input.image.clone(),
            "What is in this image?".to_string(),
        );
        assert!(input_with_prompt.text_prompt.is_some());
    }

    #[test]
    fn test_image_processing() {
        // This test would require actual model instantiation
        // For now, just test that the struct can be created
        // In a real test, you would mock the model and tokenizer
    }
}
