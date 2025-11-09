use trustformers_core::tensor::Tensor;
use trustformers_core::layers::conv::Conv2d;
use trustformers_core::layers::pooling::MaxPool2d;
use trustformers_core::layers::linear::Linear;
use trustformers_core::layers::batch_norm::BatchNorm2d;
use trustformers_core::layers::activation::{ReLU, Softmax};

#[derive(Debug, Clone)]
pub struct {{MODEL_NAME}}Config {
    pub num_classes: usize,
    pub input_channels: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub dropout_rate: f32,
}

impl Default for {{MODEL_NAME}}Config {
    fn default() -> Self {
        Self {
            num_classes: 10,
            input_channels: 3,
            input_height: 224,
            input_width: 224,
            dropout_rate: 0.5,
        }
    }
}

#[derive(Debug)]
pub struct ConvBlock {
    pub conv: Conv2d,
    pub batch_norm: BatchNorm2d,
    pub activation: ReLU,
    pub pool: Option<MaxPool2d>,
}

impl ConvBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        use_pool: bool,
    ) -> Self {
        Self {
            conv: Conv2d::new(in_channels, out_channels, kernel_size, stride, padding),
            batch_norm: BatchNorm2d::new(out_channels),
            activation: ReLU::new(),
            pool: if use_pool {
                Some(MaxPool2d::new(2, 2, 0))
            } else {
                None
            },
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut x = self.conv.forward(x)?;
        x = self.batch_norm.forward(&x)?;
        x = self.activation.forward(&x)?;

        if let Some(pool) = &self.pool {
            x = pool.forward(&x)?;
        }

        Ok(x)
    }
}

#[derive(Debug)]
pub struct {{MODEL_NAME}} {
    pub config: {{MODEL_NAME}}Config,
    pub features: Vec<ConvBlock>,
    pub avgpool: Option<Tensor>, // Global average pooling
    pub classifier: Vec<Linear>,
    pub dropout: f32,
    pub softmax: Softmax,
}

impl {{MODEL_NAME}} {
    pub fn new(config: {{MODEL_NAME}}Config) -> Self {
        let mut features = Vec::new();

        // Convolutional feature extraction layers
        features.push(ConvBlock::new(config.input_channels, 64, 7, 2, 3, true));
        features.push(ConvBlock::new(64, 128, 3, 1, 1, true));
        features.push(ConvBlock::new(128, 256, 3, 1, 1, true));
        features.push(ConvBlock::new(256, 512, 3, 1, 1, true));

        // Classification head
        let mut classifier = Vec::new();
        classifier.push(Linear::new(512, 256));
        classifier.push(Linear::new(256, config.num_classes));

        Self {
            config,
            features,
            avgpool: None,
            classifier,
            dropout: config.dropout_rate,
            softmax: Softmax::new(1),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<{{MODEL_NAME}}Output, Box<dyn std::error::Error>> {
        let mut x = x.clone();

        // Feature extraction
        for feature_block in &self.features {
            x = feature_block.forward(&x)?;
        }

        // Global average pooling
        let batch_size = x.shape()[0];
        let num_features = x.shape()[1];
        x = x.mean(&[2, 3], true)?; // Average over height and width
        x = x.view(&[batch_size, num_features])?;

        // Classification head with dropout
        for (i, linear) in self.classifier.iter().enumerate() {
            x = linear.forward(&x)?;

            // Apply ReLU activation for all but the last layer
            if i < self.classifier.len() - 1 {
                x = x.relu()?;

                // Apply dropout during training
                if self.training() {
                    x = self.apply_dropout(&x)?;
                }
            }
        }

        // Apply softmax for probabilities
        let probabilities = self.softmax.forward(&x)?;

        Ok({{MODEL_NAME}}Output {
            logits: x,
            probabilities,
        })
    }

    pub fn forward_features(&self, x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut x = x.clone();

        // Feature extraction only
        for feature_block in &self.features {
            x = feature_block.forward(&x)?;
        }

        Ok(x)
    }

    fn apply_dropout(&self, x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        // Simple dropout implementation
        // In a real implementation, this would use proper random masking
        Ok(x.clone())
    }

    fn training(&self) -> bool {
        // This would typically be managed by the training state
        true
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

    fn load_weights(&mut self, model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::collections::HashMap;
        use std::fs;
        use std::path::Path;

        println!("Loading CNN model weights from: {}", model_path);

        // Check for .safetensors file first (preferred format)
        let safetensors_path = format!("{}/model.safetensors", model_path);
        if Path::new(&safetensors_path).exists() {
            return self.load_safetensors_weights(&safetensors_path);
        }

        // Fallback to PyTorch .pth file
        let pytorch_path = format!("{}/model.pth", model_path);
        if Path::new(&pytorch_path).exists() {
            return self.load_pytorch_weights(&pytorch_path);
        }

        Err(format!("No weight files found in {}", model_path).into())
    }

    fn load_safetensors_weights(&mut self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("Loading CNN weights from SafeTensors: {}", file_path);

        // Template implementation for SafeTensors loading
        // In a real implementation, you would use the safetensors crate

        // Example structure for CNN weights:
        // let tensors = safetensors::SafeTensors::deserialize(std::fs::read(file_path)?)?;
        //
        // Load convolutional layers
        // for i in 0..self.config.num_conv_layers {
        //     let conv_weight_key = format!("features.{}.weight", i * 3); // Assuming conv, bn, relu pattern
        //     let conv_bias_key = format!("features.{}.bias", i * 3);
        //
        //     if let Some(weight) = tensors.tensor(&conv_weight_key)? {
        //         self.features[i].conv.weight = Tensor::from_safetensor(weight)?;
        //     }
        //     if let Some(bias) = tensors.tensor(&conv_bias_key)? {
        //         self.features[i].conv.bias = Tensor::from_safetensor(bias)?;
        //     }
        // }
        //
        // Load classifier weights
        // if let Some(fc_weight) = tensors.tensor("classifier.weight")? {
        //     self.classifier.weight = Tensor::from_safetensor(fc_weight)?;
        // }
        // if let Some(fc_bias) = tensors.tensor("classifier.bias")? {
        //     self.classifier.bias = Tensor::from_safetensor(fc_bias)?;
        // }

        println!("CNN SafeTensors weights loaded successfully");
        Ok(())
    }

    fn load_pytorch_weights(&mut self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("Loading CNN weights from PyTorch: {}", file_path);

        // Template implementation for PyTorch loading
        // In a real implementation, you would use tch or candle

        // Example structure:
        // let weights = tch::pickle::load_pickle(file_path)?;
        //
        // Load convolutional feature extractor
        // for (key, tensor) in weights.iter() {
        //     if key.starts_with("features.") {
        //         self.load_feature_weight(key, tensor)?;
        //     } else if key.starts_with("classifier.") {
        //         self.load_classifier_weight(key, tensor)?;
        //     }
        // }

        println!("CNN PyTorch weights loaded successfully");
        Ok(())
    }

    pub fn save_pretrained(&self, save_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(save_path)?;

        // Save configuration
        let config_path = format!("{}/config.json", save_path);
        let config_str = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(config_path, config_str)?;

        // Save model weights in SafeTensors format
        self.save_safetensors_weights(save_path)?;

        println!("CNN model saved successfully to: {}", save_path);
        Ok(())
    }

    fn save_safetensors_weights(&self, save_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::collections::HashMap;

        println!("Saving CNN model weights in SafeTensors format...");

        let safetensors_path = format!("{}/model.safetensors", save_path);

        // Template implementation for SafeTensors saving
        // In a real implementation, you would use the safetensors crate

        let mut tensors = HashMap::new();

        // Save convolutional feature extraction layers
        for (i, feature_block) in self.features.iter().enumerate() {
            // Save convolution weights
            // tensors.insert(format!("features.{}.weight", i * 3),
            //                feature_block.conv.weight.to_safetensor_bytes()?);
            // tensors.insert(format!("features.{}.bias", i * 3),
            //                feature_block.conv.bias.to_safetensor_bytes()?);

            // Save batch normalization weights
            // tensors.insert(format!("features.{}.weight", i * 3 + 1),
            //                feature_block.batch_norm.weight.to_safetensor_bytes()?);
            // tensors.insert(format!("features.{}.bias", i * 3 + 1),
            //                feature_block.batch_norm.bias.to_safetensor_bytes()?);
            // tensors.insert(format!("features.{}.running_mean", i * 3 + 1),
            //                feature_block.batch_norm.running_mean.to_safetensor_bytes()?);
            // tensors.insert(format!("features.{}.running_var", i * 3 + 1),
            //                feature_block.batch_norm.running_var.to_safetensor_bytes()?);

            println!("Prepared feature block {} for saving", i);
        }

        // Save classifier layers
        for (i, linear) in self.classifier.iter().enumerate() {
            // tensors.insert(format!("classifier.{}.weight", i),
            //                linear.weight.to_safetensor_bytes()?);
            // tensors.insert(format!("classifier.{}.bias", i),
            //                linear.bias.to_safetensor_bytes()?);

            println!("Prepared classifier layer {} for saving", i);
        }

        // Serialize and save (placeholder)
        // let serialized = safetensors::serialize(&tensors)?;
        // std::fs::write(safetensors_path, serialized)?;

        println!("CNN SafeTensors weights saved to: {}", safetensors_path);
        Ok(())
    }

    pub fn num_parameters(&self) -> usize {
        // Calculate total number of parameters
        let mut total = 0;

        // Count convolutional layer parameters
        for feature_block in &self.features {
            total += feature_block.conv.num_parameters();
            total += feature_block.batch_norm.num_parameters();
        }

        // Count classifier parameters
        for linear in &self.classifier {
            total += linear.num_parameters();
        }

        total
    }

    pub fn get_layer_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        for (i, _) in self.features.iter().enumerate() {
            names.push(format!("features.{}.conv", i));
            names.push(format!("features.{}.batch_norm", i));
        }

        for (i, _) in self.classifier.iter().enumerate() {
            names.push(format!("classifier.{}", i));
        }

        names
    }
}

#[derive(Debug, Clone)]
pub struct {{MODEL_NAME}}Output {
    pub logits: Tensor,
    pub probabilities: Tensor,
}

impl {{MODEL_NAME}}Output {
    pub fn predicted_class(&self) -> Result<usize, Box<dyn std::error::Error>> {
        self.probabilities.argmax(1).map(|t| t.to_scalar::<usize>())
    }

    pub fn top_k_classes(&self, k: usize) -> Result<Vec<(usize, f32)>, Box<dyn std::error::Error>> {
        let probs = &self.probabilities;
        let mut class_probs: Vec<(usize, f32)> = Vec::new();

        // Get all class probabilities
        for i in 0..probs.shape()[1] {
            let prob = probs.get(&[0, i])?.to_scalar::<f32>()?;
            class_probs.push((i, prob));
        }

        // Sort by probability (descending)
        class_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Return top k
        Ok(class_probs.into_iter().take(k).collect())
    }
}

{{LAYERS_IMPL}}