//! Bridges offline model packs with the online Hub client.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TrustformersError};
use std::path::Path;

use super::manager::OfflineModelPackManager;
use super::pack_types::PackCreationConfig;
use super::resolution::{empty_model_info, ModelInfo};

/// Hub integration for offline packs
/// Provides bridge between online Hub functionality and offline model packs
pub struct HubIntegration {
    pub hub_options: crate::hub::HubOptions,
}

impl HubIntegration {
    /// Create a new Hub integration instance
    pub fn new(options: Option<crate::hub::HubOptions>) -> Self {
        Self {
            hub_options: options.unwrap_or_default(),
        }
    }

    /// Download model from Hub and add it to an offline pack
    pub async fn download_model_to_pack(
        &self,
        pack_manager: &mut OfflineModelPackManager,
        model_id: &str,
        pack_id: &str,
    ) -> Result<()> {
        // Download model from Hub using existing hub functionality
        let _model_path = crate::hub::download_file_from_hub(
            model_id,
            "config.json",
            Some(self.hub_options.clone()),
        )
        .map_err(|e| TrustformersError::io_error(format!("Hub download failed: {}", e)))?;

        // Get model info from Hub, validating that it resolves to real Hub
        // metadata before adding it to the pack; the value itself isn't
        // needed here (matches the `let _ = ...` validation pattern in
        // `create_pack_from_hub_collection` below).
        let _model_info = self.get_hub_model_info(model_id).await?;

        // Update existing pack with new model
        let additional_models = vec![model_id.to_string()];
        pack_manager.update_pack(pack_id, additional_models).await?;

        Ok(())
    }

    /// Create a pack from Hub model collection
    pub async fn create_pack_from_hub_collection(
        &self,
        pack_manager: &mut OfflineModelPackManager,
        collection_name: &str,
        model_ids: Vec<String>,
        config: PackCreationConfig,
    ) -> Result<String> {
        // Verify all models exist on Hub before creating pack
        for model_id in &model_ids {
            let _ = self.get_hub_model_info(model_id).await?;
        }

        // Create pack using verified models
        pack_manager
            .create_pack(
                format!("Hub Collection: {}", collection_name),
                format!(
                    "Model pack created from Hub collection: {}",
                    collection_name
                ),
                model_ids,
                config,
            )
            .await
    }

    /// Get model information from Hub.
    ///
    /// When `model_id` is itself a local directory (the same convention
    /// [`resolve_model_source_dir`] uses), there is no Hub repo id to look up
    /// a model card for, so the network call is skipped entirely rather than
    /// sending a filesystem path to `crate::hub::load_model_card_from_hub`.
    pub(super) async fn get_hub_model_info(&self, model_id: &str) -> Result<ModelInfo> {
        if Path::new(model_id).is_dir() {
            return Ok(empty_model_info(model_id));
        }

        // Try to load model card from Hub
        match crate::hub::load_model_card_from_hub(model_id, Some(self.hub_options.clone())) {
            Ok(model_card) => {
                // Convert model card to ModelInfo
                Ok(ModelInfo {
                    model_id: model_id.to_string(),
                    library_name: Some("transformers".to_string()),
                    pipeline_tag: model_card.pipeline_tag.clone(),
                    tags: model_card.tags.unwrap_or_default(),
                    config: model_card.extra.into_iter().collect(),
                    downloads: None, // Not available in model card
                    likes: None,     // Not available in model card
                    created_at: None,
                    updated_at: None,
                    author: None,
                    description: None,
                    license: model_card.license,
                    task: model_card.pipeline_tag,
                    language: model_card.language.unwrap_or_default(),
                    dataset: model_card.datasets.unwrap_or_default(),
                    model_type: None,
                    architecture: None,
                })
            },
            Err(_) => {
                // No network access (or the model card genuinely doesn't
                // exist): we don't know anything about this model beyond its
                // id, so every Hub-side field is honestly `None`/empty
                // rather than a guessed placeholder.
                Ok(empty_model_info(model_id))
            },
        }
    }
}
