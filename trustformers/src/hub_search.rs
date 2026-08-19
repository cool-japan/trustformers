//! Hugging Face Hub model search (`GET /api/models`).
//!
//! `trustformers/TODO.md` used to self-correct, in three separate places,
//! that an earlier revision of the documentation claimed a `search_models()`
//! function existed when it never did (`rg 'search_models'` returned zero
//! matches). This module is the real implementation: search by free-text
//! query, filter by task/library/author/language, and sort by downloads or
//! likes, against the actual Hub search endpoint.
//!
//! Mirrors `hub_offline_packs.rs`'s `model_info_from_hub_json` split: the
//! JSON -> struct mapping is a pure, network-free function that is fully
//! unit-tested on its own, and only the socket itself is gated behind
//! `#[cfg(feature = "hub")]`. Without the `hub` feature, [`search_models`]
//! returns a structured error rather than fabricated results.

use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};

#[cfg(any(test, feature = "hub"))]
const HF_HUB_URL: &str = "https://huggingface.co";

/// One row of a Hub model search result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSearchResult {
    pub model_id: String,
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub tags: Vec<String>,
    pub downloads: u64,
    pub likes: u64,
    pub private: bool,
}

/// Sort order for [`search_models`] results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSearchSort {
    Downloads,
    Likes,
    LastModified,
}

impl ModelSearchSort {
    #[cfg(any(test, feature = "hub"))]
    fn as_query_value(self) -> &'static str {
        match self {
            ModelSearchSort::Downloads => "downloads",
            ModelSearchSort::Likes => "likes",
            ModelSearchSort::LastModified => "lastModified",
        }
    }
}

/// Query parameters for [`search_models`], built with the fluent `with_*` /
/// `sorted_by` setters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelSearchQuery {
    pub search: Option<String>,
    pub author: Option<String>,
    /// Task / pipeline tag filter (HF's `filter` query param), e.g. `"text-classification"`.
    pub filter_task: Option<String>,
    /// Library filter (HF's `library` query param), e.g. `"transformers"`.
    pub filter_library: Option<String>,
    pub language: Option<String>,
    pub sort: Option<ModelSearchSort>,
    pub limit: Option<u32>,
}

impl ModelSearchQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_search(mut self, search: impl Into<String>) -> Self {
        self.search = Some(search.into());
        self
    }

    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    pub fn with_task(mut self, task: impl Into<String>) -> Self {
        self.filter_task = Some(task.into());
        self
    }

    pub fn with_library(mut self, library: impl Into<String>) -> Self {
        self.filter_library = Some(library.into());
        self
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    pub fn sorted_by(mut self, sort: ModelSearchSort) -> Self {
        self.sort = Some(sort);
        self
    }

    pub fn with_limit(mut self, limit: u32) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Build the full `GET /api/models` request URL for `query`, percent-encoding
/// every value through `url::form_urlencoded` rather than hand-rolled string
/// concatenation.
#[cfg(any(test, feature = "hub"))]
fn build_search_url(query: &ModelSearchQuery) -> String {
    let mut serializer = url::form_urlencoded::Serializer::new(String::new());
    if let Some(search) = &query.search {
        serializer.append_pair("search", search);
    }
    if let Some(author) = &query.author {
        serializer.append_pair("author", author);
    }
    if let Some(task) = &query.filter_task {
        serializer.append_pair("filter", task);
    }
    if let Some(library) = &query.filter_library {
        serializer.append_pair("library", library);
    }
    if let Some(language) = &query.language {
        serializer.append_pair("language", language);
    }
    if let Some(sort) = query.sort {
        serializer.append_pair("sort", sort.as_query_value());
        serializer.append_pair("direction", "-1");
    }
    if let Some(limit) = query.limit {
        serializer.append_pair("limit", &limit.to_string());
    }

    let query_string = serializer.finish();
    if query_string.is_empty() {
        format!("{HF_HUB_URL}/api/models")
    } else {
        format!("{HF_HUB_URL}/api/models?{query_string}")
    }
}

/// Pure JSON -> struct mapping for one entry of the `/api/models` search
/// response, kept separate from the HTTP call so it can be unit-tested
/// without a network connection.
#[cfg(any(test, feature = "hub"))]
fn model_search_result_from_json(json: &serde_json::Value) -> ModelSearchResult {
    let model_id = json
        .get("id")
        .or_else(|| json.get("modelId"))
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    ModelSearchResult {
        model_id,
        pipeline_tag: json.get("pipeline_tag").and_then(|v| v.as_str()).map(str::to_string),
        library_name: json.get("library_name").and_then(|v| v.as_str()).map(str::to_string),
        tags: json
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|items| items.iter().filter_map(|t| t.as_str().map(str::to_string)).collect())
            .unwrap_or_default(),
        downloads: json.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0),
        likes: json.get("likes").and_then(|v| v.as_u64()).unwrap_or(0),
        private: json.get("private").and_then(|v| v.as_bool()).unwrap_or(false),
    }
}

/// Search Hub models by task/language/library/author, sorted by downloads,
/// likes, or last-modified date, against the real `GET /api/models` endpoint.
#[cfg(feature = "hub")]
pub async fn search_models(query: &ModelSearchQuery) -> Result<Vec<ModelSearchResult>> {
    let url = build_search_url(query);
    let client = reqwest::Client::new();

    let response = client.get(&url).send().await.map_err(|e| TrustformersError::Hub {
        message: format!("Failed to search models: {}", e),
        model_id: String::new(),
        endpoint: Some(url.clone()),
        suggestion: Some("Check network connectivity".to_string()),
        recovery_actions: vec![],
    })?;

    if !response.status().is_success() {
        return Err(TrustformersError::Hub {
            message: format!("Model search failed: HTTP {}", response.status()),
            model_id: String::new(),
            endpoint: Some(url.clone()),
            suggestion: Some("Check the search query and filter parameters are valid".to_string()),
            recovery_actions: vec![],
        });
    }

    let results: Vec<serde_json::Value> = response.json().await.map_err(|e| {
        TrustformersError::invalid_input(
            format!("Failed to parse model search response: {}", e),
            Some("api_response"),
            Some("valid JSON array of model objects"),
            Some("invalid JSON format"),
        )
    })?;

    Ok(results.iter().map(model_search_result_from_json).collect())
}

/// Without the `hub` feature (no networking), model search cannot reach the
/// Hub — return a clean, structured error rather than fabricated results.
#[cfg(not(feature = "hub"))]
pub async fn search_models(_query: &ModelSearchQuery) -> Result<Vec<ModelSearchResult>> {
    Err(TrustformersError::Hub {
        message: "Hub model search is disabled: the `hub` feature is not enabled".to_string(),
        model_id: String::new(),
        endpoint: None,
        suggestion: Some(
            "Rebuild with the `hub` feature (e.g. `--features hub`) to search the Hugging Face \
             Hub"
            .to_string(),
        ),
        recovery_actions: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_search_url_empty_query() {
        let url = build_search_url(&ModelSearchQuery::new());
        assert_eq!(url, "https://huggingface.co/api/models");
    }

    #[test]
    fn test_build_search_url_encodes_all_filters() {
        let query = ModelSearchQuery::new()
            .with_search("sentiment analysis")
            .with_author("some-org")
            .with_task("text-classification")
            .with_library("transformers")
            .with_language("en")
            .sorted_by(ModelSearchSort::Downloads)
            .with_limit(25);
        let url = build_search_url(&query);

        assert!(url.starts_with("https://huggingface.co/api/models?"));
        assert!(url.contains("search=sentiment+analysis"));
        assert!(url.contains("author=some-org"));
        assert!(url.contains("filter=text-classification"));
        assert!(url.contains("library=transformers"));
        assert!(url.contains("language=en"));
        assert!(url.contains("sort=downloads"));
        assert!(url.contains("direction=-1"));
        assert!(url.contains("limit=25"));
    }

    #[test]
    fn test_build_search_url_sort_variants() {
        assert!(
            build_search_url(&ModelSearchQuery::new().sorted_by(ModelSearchSort::Likes))
                .contains("sort=likes")
        );
        assert!(build_search_url(
            &ModelSearchQuery::new().sorted_by(ModelSearchSort::LastModified)
        )
        .contains("sort=lastModified"));
    }

    #[test]
    fn test_model_search_result_from_json_full() {
        let json = serde_json::json!({
            "id": "bert-base-uncased",
            "pipeline_tag": "fill-mask",
            "library_name": "transformers",
            "tags": ["pytorch", "bert", "en"],
            "downloads": 5_000_000,
            "likes": 1200,
            "private": false,
        });
        let result = model_search_result_from_json(&json);
        assert_eq!(result.model_id, "bert-base-uncased");
        assert_eq!(result.pipeline_tag.as_deref(), Some("fill-mask"));
        assert_eq!(result.library_name.as_deref(), Some("transformers"));
        assert_eq!(result.tags, vec!["pytorch", "bert", "en"]);
        assert_eq!(result.downloads, 5_000_000);
        assert_eq!(result.likes, 1200);
        assert!(!result.private);
    }

    #[test]
    fn test_model_search_result_from_json_uses_model_id_fallback() {
        let json = serde_json::json!({ "modelId": "org/model-name" });
        let result = model_search_result_from_json(&json);
        assert_eq!(result.model_id, "org/model-name");
    }

    #[test]
    fn test_model_search_result_from_json_missing_fields_default() {
        let json = serde_json::json!({});
        let result = model_search_result_from_json(&json);
        assert_eq!(result.model_id, "");
        assert_eq!(result.pipeline_tag, None);
        assert_eq!(result.downloads, 0);
        assert_eq!(result.likes, 0);
        assert!(!result.private);
        assert!(result.tags.is_empty());
    }

    #[test]
    fn test_model_search_query_builder() {
        let query = ModelSearchQuery::new().with_search("gpt").with_limit(10);
        assert_eq!(query.search.as_deref(), Some("gpt"));
        assert_eq!(query.limit, Some(10));
        assert!(query.author.is_none());
    }

    /// Regression test: without the `hub` feature, `search_models` must
    /// return a structured error, never a fabricated result list.
    #[cfg(not(feature = "hub"))]
    #[tokio::test]
    async fn test_search_models_without_hub_feature_errors() {
        let result = search_models(&ModelSearchQuery::new().with_search("bert")).await;
        assert!(
            result.is_err(),
            "search_models must fail cleanly without the `hub` feature"
        );
    }
}
