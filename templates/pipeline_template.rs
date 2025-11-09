use trustformers::pipeline::{Pipeline, PipelineOutput};
use trustformers_core::tensor::Tensor;
use trustformers_tokenizers::tokenizer::Tokenizer;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct {{PIPELINE_NAME}}Config {
    pub model_path: String,
    pub tokenizer_path: String,
    pub device: String,
    pub batch_size: usize,
    pub max_length: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl Default for {{PIPELINE_NAME}}Config {
    fn default() -> Self {
        Self {
            model_path: "./models/{{MODEL_TYPE}}".to_string(),
            tokenizer_path: "./tokenizers/{{MODEL_TYPE}}".to_string(),
            device: "cpu".to_string(),
            batch_size: 1,
            max_length: 512,
            temperature: 1.0,
            top_p: 0.9,
            top_k: 50,
        }
    }
}

#[derive(Debug)]
pub struct {{PIPELINE_NAME}} {
    pub config: {{PIPELINE_NAME}}Config,
    pub model: Box<dyn Pipeline>,
    pub tokenizer: Tokenizer,
    pub device: String,
}

impl {{PIPELINE_NAME}} {
    pub fn new(config: {{PIPELINE_NAME}}Config) -> Result<Self, Box<dyn std::error::Error>> {
        // Load tokenizer
        let tokenizer = Tokenizer::from_pretrained(&config.tokenizer_path)?;

        // Create model pipeline based on task type
        let model = Self::create_pipeline(&config)?;

        Ok(Self {
            device: config.device.clone(),
            config,
            model,
            tokenizer,
        })
    }

    fn create_pipeline(config: &{{PIPELINE_NAME}}Config) -> Result<Box<dyn Pipeline>, Box<dyn std::error::Error>> {
        match "{{TASK_TYPE}}" {
            "text-generation" => {
                use trustformers::pipeline::text_generation::TextGenerationPipeline;
                let pipeline = TextGenerationPipeline::new(&config.model_path, &config.device)?;
                Ok(Box::new(pipeline))
            }
            "text-classification" => {
                use trustformers::pipeline::text_classification::TextClassificationPipeline;
                let pipeline = TextClassificationPipeline::new(&config.model_path, &config.device)?;
                Ok(Box::new(pipeline))
            }
            "question-answering" => {
                use trustformers::pipeline::question_answering::QuestionAnsweringPipeline;
                let pipeline = QuestionAnsweringPipeline::new(&config.model_path, &config.device)?;
                Ok(Box::new(pipeline))
            }
            "summarization" => {
                use trustformers::pipeline::summarization::SummarizationPipeline;
                let pipeline = SummarizationPipeline::new(&config.model_path, &config.device)?;
                Ok(Box::new(pipeline))
            }
            "translation" => {
                use trustformers::pipeline::translation::TranslationPipeline;
                let pipeline = TranslationPipeline::new(&config.model_path, &config.device)?;
                Ok(Box::new(pipeline))
            }
            "fill-mask" => {
                use trustformers::pipeline::fill_mask::FillMaskPipeline;
                let pipeline = FillMaskPipeline::new(&config.model_path, &config.device)?;
                Ok(Box::new(pipeline))
            }
            _ => {
                Err(format!("Unsupported task type: {{TASK_TYPE}}").into())
            }
        }
    }

    pub async fn process(&self, input: &str) -> Result<{{PIPELINE_NAME}}Output, Box<dyn std::error::Error>> {
        // Tokenize input
        let tokens = self.tokenizer.encode(input, Some(self.config.max_length))?;
        let input_ids = Tensor::from_slice(&tokens.ids, &[1, tokens.ids.len()]);
        let attention_mask = Tensor::from_slice(&tokens.attention_mask, &[1, tokens.attention_mask.len()]);

        // Create input for pipeline
        let mut pipeline_input = HashMap::new();
        pipeline_input.insert("input_ids".to_string(), input_ids);
        pipeline_input.insert("attention_mask".to_string(), attention_mask);

        // Add task-specific parameters
        match "{{TASK_TYPE}}" {
            "text-generation" => {
                pipeline_input.insert("max_length".to_string(),
                    Tensor::from_scalar(self.config.max_length as f32));
                pipeline_input.insert("temperature".to_string(),
                    Tensor::from_scalar(self.config.temperature));
                pipeline_input.insert("top_p".to_string(),
                    Tensor::from_scalar(self.config.top_p));
                pipeline_input.insert("top_k".to_string(),
                    Tensor::from_scalar(self.config.top_k as f32));
            }
            "text-classification" => {
                // Classification-specific parameters
            }
            "question-answering" => {
                // QA-specific parameters would be added here
            }
            _ => {}
        }

        // Run inference
        let output = self.model.forward(&pipeline_input).await?;

        // Process output based on task type
        let processed_output = self.process_output(&output, input)?;

        Ok(processed_output)
    }

    fn process_output(
        &self,
        output: &PipelineOutput,
        original_input: &str,
    ) -> Result<{{PIPELINE_NAME}}Output, Box<dyn std::error::Error>> {
        match "{{TASK_TYPE}}" {
            "text-generation" => {
                let generated_ids = output.get_tensor("generated_ids")
                    .ok_or("Missing generated_ids in output")?;

                let generated_text = self.tokenizer.decode(&generated_ids.to_vec::<i32>()?, true)?;

                Ok({{PIPELINE_NAME}}Output::TextGeneration {
                    generated_text,
                    input_text: original_input.to_string(),
                })
            }
            "text-classification" => {
                let logits = output.get_tensor("logits")
                    .ok_or("Missing logits in output")?;

                let probabilities = logits.softmax(-1)?;
                let predicted_class = probabilities.argmax(-1)?.to_scalar::<i32>()?;
                let confidence = probabilities.max(-1)?.to_scalar::<f32>()?;

                Ok({{PIPELINE_NAME}}Output::Classification {
                    predicted_class: predicted_class as usize,
                    confidence,
                    probabilities: probabilities.to_vec::<f32>()?,
                })
            }
            "question-answering" => {
                let start_logits = output.get_tensor("start_logits")
                    .ok_or("Missing start_logits in output")?;
                let end_logits = output.get_tensor("end_logits")
                    .ok_or("Missing end_logits in output")?;

                let start_idx = start_logits.argmax(-1)?.to_scalar::<i32>()? as usize;
                let end_idx = end_logits.argmax(-1)?.to_scalar::<i32>()? as usize;

                // Extract answer span
                let tokens = self.tokenizer.encode(original_input, Some(self.config.max_length))?;
                let answer_tokens = &tokens.ids[start_idx..=end_idx];
                let answer = self.tokenizer.decode(answer_tokens, true)?;

                let confidence = (start_logits.max(-1)?.to_scalar::<f32>()? +
                                end_logits.max(-1)?.to_scalar::<f32>()?) / 2.0;

                Ok({{PIPELINE_NAME}}Output::QuestionAnswering {
                    answer,
                    confidence,
                    start_idx,
                    end_idx,
                })
            }
            "summarization" => {
                let summary_ids = output.get_tensor("summary_ids")
                    .ok_or("Missing summary_ids in output")?;

                let summary = self.tokenizer.decode(&summary_ids.to_vec::<i32>()?, true)?;

                Ok({{PIPELINE_NAME}}Output::Summarization {
                    summary,
                    original_text: original_input.to_string(),
                })
            }
            _ => {
                Err(format!("Unsupported task type for output processing: {{TASK_TYPE}}").into())
            }
        }
    }

    pub async fn batch_process(&self, inputs: &[String]) -> Result<Vec<{{PIPELINE_NAME}}Output>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();

        // Process in batches
        for chunk in inputs.chunks(self.config.batch_size) {
            let mut batch_results = Vec::new();

            // Process each item in the batch
            for input in chunk {
                let result = self.process(input).await?;
                batch_results.push(result);
            }

            results.extend(batch_results);
        }

        Ok(results)
    }

    pub fn get_model_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("model_path".to_string(), self.config.model_path.clone());
        info.insert("tokenizer_path".to_string(), self.config.tokenizer_path.clone());
        info.insert("task_type".to_string(), "{{TASK_TYPE}}".to_string());
        info.insert("device".to_string(), self.device.clone());
        info.insert("max_length".to_string(), self.config.max_length.to_string());
        info
    }

    pub fn update_config(&mut self, new_config: {{PIPELINE_NAME}}Config) -> Result<(), Box<dyn std::error::Error>> {
        // Check if model needs to be reloaded
        let reload_model = self.config.model_path != new_config.model_path ||
                          self.config.device != new_config.device;

        let reload_tokenizer = self.config.tokenizer_path != new_config.tokenizer_path;

        if reload_model {
            self.model = Self::create_pipeline(&new_config)?;
            self.device = new_config.device.clone();
        }

        if reload_tokenizer {
            self.tokenizer = Tokenizer::from_pretrained(&new_config.tokenizer_path)?;
        }

        self.config = new_config;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum {{PIPELINE_NAME}}Output {
    TextGeneration {
        generated_text: String,
        input_text: String,
    },
    Classification {
        predicted_class: usize,
        confidence: f32,
        probabilities: Vec<f32>,
    },
    QuestionAnswering {
        answer: String,
        confidence: f32,
        start_idx: usize,
        end_idx: usize,
    },
    Summarization {
        summary: String,
        original_text: String,
    },
    Translation {
        translated_text: String,
        source_language: String,
        target_language: String,
    },
}

impl {{PIPELINE_NAME}}Output {
    pub fn to_string(&self) -> String {
        match self {
            Self::TextGeneration { generated_text, .. } => generated_text.clone(),
            Self::Classification { predicted_class, confidence, .. } => {
                format!("Class: {}, Confidence: {:.4}", predicted_class, confidence)
            }
            Self::QuestionAnswering { answer, confidence, .. } => {
                format!("Answer: {} (confidence: {:.4})", answer, confidence)
            }
            Self::Summarization { summary, .. } => summary.clone(),
            Self::Translation { translated_text, .. } => translated_text.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_creation() {
        let config = {{PIPELINE_NAME}}Config::default();
        // Note: This test would require actual model files to pass
        // In practice, you'd use mock models for unit testing
    }

    #[test]
    fn test_config_serialization() {
        let config = {{PIPELINE_NAME}}Config::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: {{PIPELINE_NAME}}Config = serde_json::from_str(&json).unwrap();

        assert_eq!(config.max_length, deserialized.max_length);
        assert_eq!(config.temperature, deserialized.temperature);
    }

    #[test]
    fn test_output_to_string() {
        let output = {{PIPELINE_NAME}}Output::Classification {
            predicted_class: 1,
            confidence: 0.95,
            probabilities: vec![0.05, 0.95],
        };

        let output_str = output.to_string();
        assert!(output_str.contains("Class: 1"));
        assert!(output_str.contains("0.95"));
    }
}