//! Resolving a Hub model id to real on-disk info and a source directory, without ever fabricating one.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Model information structure for Hub integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_id: String,
    pub library_name: Option<String>,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
    pub config: HashMap<String, serde_json::Value>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub author: Option<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    pub task: Option<String>,
    pub language: Vec<String>,
    pub dataset: Vec<String>,
    pub model_type: Option<String>,
    pub architecture: Option<String>,
}
/// Parse a Hugging Face Hub `/api/models/{id}` JSON response into a [`ModelInfo`].
///
/// This is a pure, network-free mapping function so the field-extraction logic
/// (including the nested `cardData`/`config` lookups) can be unit-tested
/// without making any HTTP calls. Only the `hub`-feature body of
/// `OfflineModelPackManager::get_model_info` performs the actual request;
/// this function just maps the resulting JSON.
///
/// Field mapping mirrors `hub.rs::get_download_stats`'s manual-field-pull
/// pattern for this exact same endpoint:
/// - `downloads`, `likes`, `pipeline_tag`, `tags`, `library_name` map directly
/// - `createdAt` -> `created_at`, `lastModified` -> `updated_at`
/// - `cardData.license` / `cardData.language` / `cardData.datasets` -> `license` / `language` / `dataset`
/// - `config.model_type` -> `model_type`, `config.architectures[0]` -> `architecture`
#[cfg(feature = "hub")]
pub(super) fn model_info_from_hub_json(model_id: &str, json: &serde_json::Value) -> ModelInfo {
    // `cardData.language`/`cardData.datasets` may be a single string or an
    // array of strings depending on the model's metadata; normalize both.
    fn as_string_vec(value: Option<&serde_json::Value>) -> Vec<String> {
        match value {
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            Some(serde_json::Value::Array(items)) => {
                items.iter().filter_map(|item| item.as_str().map(str::to_string)).collect()
            },
            _ => Vec::new(),
        }
    }
    let pipeline_tag = json.get("pipeline_tag").and_then(|v| v.as_str()).map(str::to_string);
    let card_data = json.get("cardData");
    let config_obj = json.get("config");
    ModelInfo {
        model_id: model_id.to_string(),
        library_name: json.get("library_name").and_then(|v| v.as_str()).map(str::to_string),
        pipeline_tag: pipeline_tag.clone(),
        tags: json
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|items| items.iter().filter_map(|t| t.as_str().map(str::to_string)).collect())
            .unwrap_or_default(),
        config: config_obj
            .and_then(|c| c.as_object())
            .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default(),
        downloads: json.get("downloads").and_then(|v| v.as_u64()),
        likes: json.get("likes").and_then(|v| v.as_u64()),
        created_at: json.get("createdAt").and_then(|v| v.as_str()).map(str::to_string),
        updated_at: json.get("lastModified").and_then(|v| v.as_str()).map(str::to_string),
        author: json.get("author").and_then(|v| v.as_str()).map(str::to_string),
        description: None,
        license: card_data
            .and_then(|c| c.get("license"))
            .and_then(|v| v.as_str())
            .map(str::to_string),
        task: pipeline_tag,
        language: as_string_vec(card_data.and_then(|c| c.get("language"))),
        dataset: as_string_vec(card_data.and_then(|c| c.get("datasets"))),
        model_type: config_obj
            .and_then(|c| c.get("model_type"))
            .and_then(|v| v.as_str())
            .map(str::to_string),
        architecture: config_obj
            .and_then(|c| c.get("architectures"))
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .map(str::to_string),
    }
}

/// Build a "we don't know anything about this model" [`ModelInfo`]: every
/// Hub-side field is `None`/empty rather than a guessed placeholder. Shared
/// by every path that genuinely has no Hub metadata to report — `model_id`
/// being a local directory (no Hub repo id to query at all), the `hub`
/// feature being disabled, or a failed Hub lookup.
pub(super) fn empty_model_info(model_id: &str) -> ModelInfo {
    ModelInfo {
        model_id: model_id.to_string(),
        library_name: None,
        pipeline_tag: None,
        tags: vec![],
        config: HashMap::new(),
        downloads: None,
        likes: None,
        created_at: None,
        updated_at: None,
        author: None,
        description: None,
        license: None,
        task: None,
        language: vec![],
        dataset: vec![],
        model_type: None,
        architecture: None,
    }
}

/// Resolve `model_id` to a real, on-disk directory containing that model's
/// files, without ever fabricating one:
///
/// 1. If `model_id` is itself an existing local directory, use it directly —
///    this is how a caller points at a model that was never downloaded from
///    the Hub at all.
/// 2. Otherwise, check the Hub's on-disk cache (the same layout
///    [`crate::hub::download_model_enhanced`] writes to).
/// 3. With the `hub` feature enabled and no local cache hit, download the
///    model's essential files into that cache, then use it.
///
/// Returns an error — never an empty or synthetic directory — if none of the
/// above produces a real directory containing at least one file.
pub(super) async fn resolve_model_source_dir(model_id: &str) -> Result<PathBuf> {
    let explicit = Path::new(model_id);
    if explicit.is_dir() {
        return Ok(explicit.to_path_buf());
    }
    let cache_dir = crate::hub::get_cache_dir()?;
    let cached_model_dir = cache_dir.join("models").join(model_id.replace('/', "--")).join("main");
    if cached_model_dir.is_dir() && has_any_files(&cached_model_dir) {
        return Ok(cached_model_dir);
    }
    #[cfg(feature = "hub")]
    {
        let (downloaded_dir, _stats) =
            crate::hub::download_model_enhanced(model_id, None).await.map_err(|e| {
                TrustformersError::io_error(format!(
                    "Failed to download model '{model_id}' to build an offline pack: {e}"
                ))
            })?;
        if has_any_files(&downloaded_dir) {
            return Ok(downloaded_dir);
        }
        Err(TrustformersError::file_not_found(format!(
            "Downloaded model directory for '{model_id}' contains no files"
        )))
    }
    #[cfg(not(feature = "hub"))]
    {
        Err(TrustformersError::invalid_input_simple(format!(
            "Cannot build an offline pack for '{model_id}': it is not a local directory, it is \
             not present in the local cache ({}), and the `hub` feature is disabled so it cannot \
             be downloaded. Provide a local model directory, pre-populate the cache, or rebuild \
             with `--features hub`.",
            cached_model_dir.display()
        )))
    }
}

/// Whether `dir` contains at least one regular file directly inside it.
pub(super) fn has_any_files(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.filter_map(|e| e.ok()).any(|e| e.path().is_file())
}

/// Lexically normalize a tar entry's path and confirm it cannot escape
/// `base`, without touching the filesystem (the destination doesn't exist yet
/// during extraction, so canonicalization isn't an option). Any `..`
/// component or absolute-path component is rejected outright rather than
/// "resolved" — the simplest policy that is unambiguously safe against a
/// crafted archive with entries like `../../etc/passwd`.
pub(super) fn safe_relative_path(base: &Path, entry_name: &str) -> Option<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in Path::new(entry_name).components() {
        match component {
            std::path::Component::Normal(part) => normalized.push(part),
            std::path::Component::CurDir => {},
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => return None,
        }
    }
    if normalized.as_os_str().is_empty() {
        return None;
    }
    Some(base.join(normalized))
}
