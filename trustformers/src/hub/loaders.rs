//! Loading model configs, weights, and README-derived model cards from the Hub.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use trustformers_core::errors::TrustformersError as CoreTrustformersError;

use super::download::download_file_from_hub;
use super::types::HubOptions;

/// Model card information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    pub license: Option<String>,
    pub language: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub datasets: Option<Vec<String>>,
    pub metrics: Option<Vec<String>>,
    pub widget: Option<Vec<serde_json::Value>>,
    pub model_index: Option<Vec<serde_json::Value>>,
    pub thumbnail: Option<String>,
    pub pipeline_tag: Option<String>,
    pub inference: Option<bool>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Load a model configuration from the Hub
pub fn load_config_from_hub(
    model_id: &str,
    options: Option<HubOptions>,
) -> Result<serde_json::Value> {
    let config_path = download_file_from_hub(model_id, "config.json", options)?;
    let config_str = std::fs::read_to_string(&config_path).map_err(|e| TrustformersError::Io {
        message: format!("Failed to read config file: {}", e),
        path: Some(config_path.to_string_lossy().to_string()),
        suggestion: Some("Check file existence and permissions".to_string()),
    })?;
    serde_json::from_str(&config_str).map_err(|e| {
        TrustformersError::invalid_input(
            format!("Failed to parse config: {}", e),
            Some("config_json"),
            Some("valid JSON format"),
            Some("invalid JSON"),
        )
    })
}

/// Load model weights from the Hub (supports SafeTensors format)
pub fn load_weights_from_hub(
    model_id: &str,
    options: Option<HubOptions>,
) -> Result<Box<dyn crate::core::traits::WeightReader>> {
    // Try SafeTensors first
    let safetensors_path = download_file_from_hub(model_id, "model.safetensors", options.clone());

    if let Ok(path) = safetensors_path {
        let reader = crate::core::utils::weight_loading::SafeTensorsReader::from_file(&path)?;
        return Ok(Box::new(reader));
    }

    // Fall back to PyTorch format if SafeTensors not available
    let pytorch_formats = ["pytorch_model.bin", "model.pt", "pytorch_model.pt"];

    for pytorch_file in &pytorch_formats {
        if let Ok(path) = download_file_from_hub(model_id, pytorch_file, options.clone()) {
            let reader = crate::core::utils::weight_loading::PyTorchReader::from_file(&path)?;
            return Ok(Box::new(reader));
        }
    }

    // If neither SafeTensors nor PyTorch formats are found
    Err(TrustformersError::Core(CoreTrustformersError::other(
        format!("No supported weight format found for model {}: Tried SafeTensors (.safetensors), PyTorch (.bin, .pt)", model_id)
    )))
}

/// Parse model card from README.md
fn parse_model_card_from_readme(content: &str) -> Result<ModelCard> {
    // Look for YAML frontmatter in the README
    if let Some(yaml_start) = content.find("---\n") {
        if let Some(yaml_end) = content[yaml_start + 4..].find("\n---") {
            let yaml_content = &content[yaml_start + 4..yaml_start + 4 + yaml_end];

            // Parse YAML frontmatter
            let yaml_value: serde_yaml_ng::Value =
                serde_yaml_ng::from_str(yaml_content).map_err(|e| {
                    TrustformersError::invalid_input(
                        format!("Failed to parse YAML frontmatter: {}", e),
                        Some("yaml_frontmatter"),
                        Some("valid YAML format"),
                        Some("invalid YAML"),
                    )
                })?;

            // Convert YAML to JSON for easier handling with serde_json
            let json_value = serde_json::to_value(yaml_value).map_err(|e| {
                TrustformersError::invalid_input(
                    format!("Failed to convert YAML to JSON: {}", e),
                    Some("yaml_content"),
                    Some("YAML convertible to JSON"),
                    Some("incompatible YAML structure"),
                )
            })?;

            // Parse as ModelCard
            let model_card: ModelCard = serde_json::from_value(json_value).map_err(|e| {
                TrustformersError::invalid_input(
                    format!("Failed to parse model card: {}", e),
                    Some("model_card_json"),
                    Some("valid ModelCard structure"),
                    Some("invalid model card format"),
                )
            })?;

            return Ok(model_card);
        }
    }

    // If no YAML frontmatter found, return empty model card
    Ok(ModelCard {
        license: None,
        language: None,
        tags: None,
        datasets: None,
        metrics: None,
        widget: None,
        model_index: None,
        thumbnail: None,
        pipeline_tag: None,
        inference: None,
        extra: serde_json::Map::new(),
    })
}

/// Load a model card from the Hub
pub fn load_model_card_from_hub(model_id: &str, options: Option<HubOptions>) -> Result<ModelCard> {
    // Try to download README.md
    let readme_path = download_file_from_hub(model_id, "README.md", options);

    if let Ok(path) = readme_path {
        let readme_content = std::fs::read_to_string(&path).map_err(|e| TrustformersError::Io {
            message: format!("Failed to read README.md: {}", e),
            path: Some(path.to_string_lossy().to_string()),
            suggestion: Some("Check file existence and permissions".to_string()),
        })?;

        parse_model_card_from_readme(&readme_content)
    } else {
        // If README.md not found, return empty model card
        Ok(ModelCard {
            license: None,
            language: None,
            tags: None,
            datasets: None,
            metrics: None,
            widget: None,
            model_index: None,
            thumbnail: None,
            pipeline_tag: None,
            inference: None,
            extra: serde_json::Map::new(),
        })
    }
}
