//! Storage backends for leaderboard data

use super::{LeaderboardEntry, LeaderboardQuery};
use anyhow::Result;
use async_trait::async_trait;
use serde_json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Trait for leaderboard storage backends
#[async_trait]
pub trait LeaderboardStorage: Send + Sync {
    /// Store a new entry
    async fn store(&self, entry: &LeaderboardEntry) -> Result<()>;

    /// Get entry by ID
    async fn get(&self, id: Uuid) -> Result<Option<LeaderboardEntry>>;

    /// Query entries
    async fn query(&self, query: &LeaderboardQuery) -> Result<Vec<LeaderboardEntry>>;

    /// Update an existing entry
    async fn update(&self, entry: &LeaderboardEntry) -> Result<()>;

    /// Delete an entry
    async fn delete(&self, id: Uuid) -> Result<()>;

    /// List all entries
    async fn list_all(&self) -> Result<Vec<LeaderboardEntry>>;

    /// Clear all entries (use with caution)
    async fn clear(&self) -> Result<()>;
}

/// File-based storage implementation
pub struct FileStorage {
    base_dir: PathBuf,
    index: Arc<RwLock<HashMap<Uuid, PathBuf>>>,
}

impl FileStorage {
    /// Create new file storage
    pub async fn new<P: AsRef<Path>>(base_dir: P) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        fs::create_dir_all(&base_dir).await?;

        let storage = Self {
            base_dir,
            index: Arc::new(RwLock::new(HashMap::new())),
        };

        // Build index
        storage.rebuild_index().await?;

        Ok(storage)
    }

    /// Rebuild the index from disk
    async fn rebuild_index(&self) -> Result<()> {
        let mut index = self.index.write().await;
        index.clear();

        let mut entries = fs::read_dir(&self.base_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(filename) = path.file_stem() {
                    if let Ok(id) = Uuid::parse_str(filename.to_str().unwrap_or("")) {
                        index.insert(id, path);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get file path for an entry
    fn get_path(&self, id: Uuid) -> PathBuf {
        self.base_dir.join(format!("{}.json", id))
    }
}

#[async_trait]
impl LeaderboardStorage for FileStorage {
    async fn store(&self, entry: &LeaderboardEntry) -> Result<()> {
        let path = self.get_path(entry.id);
        let json = serde_json::to_string_pretty(entry)?;
        fs::write(&path, json).await?;

        let mut index = self.index.write().await;
        index.insert(entry.id, path);

        Ok(())
    }

    async fn get(&self, id: Uuid) -> Result<Option<LeaderboardEntry>> {
        let index = self.index.read().await;
        if let Some(path) = index.get(&id) {
            let data = fs::read_to_string(path).await?;
            let entry: LeaderboardEntry = serde_json::from_str(&data)?;
            Ok(Some(entry))
        } else {
            Ok(None)
        }
    }

    async fn query(&self, query: &LeaderboardQuery) -> Result<Vec<LeaderboardEntry>> {
        let mut entries = self.list_all().await?;

        // Apply filters
        entries = query.apply_filters(entries);

        // Apply sorting before limiting
        entries.sort_by(|a, b| query.ranking_criteria.compare(a, b));

        // Apply limit
        if let Some(limit) = query.limit {
            entries.truncate(limit);
        }

        Ok(entries)
    }

    async fn update(&self, entry: &LeaderboardEntry) -> Result<()> {
        self.store(entry).await
    }

    async fn delete(&self, id: Uuid) -> Result<()> {
        let mut index = self.index.write().await;
        if let Some(path) = index.remove(&id) {
            fs::remove_file(path).await?;
        }
        Ok(())
    }

    async fn list_all(&self) -> Result<Vec<LeaderboardEntry>> {
        let index = self.index.read().await;
        let mut entries = Vec::new();

        for path in index.values() {
            let data = fs::read_to_string(path).await?;
            let entry: LeaderboardEntry = serde_json::from_str(&data)?;
            entries.push(entry);
        }

        Ok(entries)
    }

    async fn clear(&self) -> Result<()> {
        let mut index = self.index.write().await;

        for path in index.values() {
            fs::remove_file(path).await?;
        }

        index.clear();
        Ok(())
    }
}

/// Remote storage implementation (e.g., REST API, database)
pub struct RemoteStorage {
    endpoint: String,
    client: reqwest::Client,
    api_key: Option<String>,
}

impl RemoteStorage {
    /// Create new remote storage
    pub fn new(endpoint: String, api_key: Option<String>) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::new(),
            api_key,
        }
    }

    /// Build request with authentication
    fn build_request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.endpoint, path);
        let mut req = self.client.request(method, url);

        if let Some(api_key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }

        req
    }
}

#[async_trait]
impl LeaderboardStorage for RemoteStorage {
    async fn store(&self, entry: &LeaderboardEntry) -> Result<()> {
        let response =
            self.build_request(reqwest::Method::POST, "entries").json(entry).send().await?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to store entry: {}", response.status());
        }

        Ok(())
    }

    async fn get(&self, id: Uuid) -> Result<Option<LeaderboardEntry>> {
        let response = self
            .build_request(reqwest::Method::GET, &format!("entries/{}", id))
            .send()
            .await?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }

        if !response.status().is_success() {
            anyhow::bail!("Failed to get entry: {}", response.status());
        }

        let entry = response.json().await?;
        Ok(Some(entry))
    }

    async fn query(&self, query: &LeaderboardQuery) -> Result<Vec<LeaderboardEntry>> {
        let response =
            self.build_request(reqwest::Method::POST, "query").json(query).send().await?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to query entries: {}", response.status());
        }

        let entries = response.json().await?;
        Ok(entries)
    }

    async fn update(&self, entry: &LeaderboardEntry) -> Result<()> {
        let response = self
            .build_request(reqwest::Method::PUT, &format!("entries/{}", entry.id))
            .json(entry)
            .send()
            .await?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to update entry: {}", response.status());
        }

        Ok(())
    }

    async fn delete(&self, id: Uuid) -> Result<()> {
        let response = self
            .build_request(reqwest::Method::DELETE, &format!("entries/{}", id))
            .send()
            .await?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete entry: {}", response.status());
        }

        Ok(())
    }

    async fn list_all(&self) -> Result<Vec<LeaderboardEntry>> {
        let response = self.build_request(reqwest::Method::GET, "entries").send().await?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list entries: {}", response.status());
        }

        let entries = response.json().await?;
        Ok(entries)
    }

    async fn clear(&self) -> Result<()> {
        let response = self.build_request(reqwest::Method::DELETE, "entries").send().await?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to clear entries: {}", response.status());
        }

        Ok(())
    }
}

/// In-memory storage for testing
pub struct MemoryStorage {
    entries: Arc<RwLock<HashMap<Uuid, LeaderboardEntry>>>,
}

impl MemoryStorage {
    /// Create new memory storage
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LeaderboardStorage for MemoryStorage {
    async fn store(&self, entry: &LeaderboardEntry) -> Result<()> {
        let mut entries = self.entries.write().await;
        entries.insert(entry.id, entry.clone());
        Ok(())
    }

    async fn get(&self, id: Uuid) -> Result<Option<LeaderboardEntry>> {
        let entries = self.entries.read().await;
        Ok(entries.get(&id).cloned())
    }

    async fn query(&self, query: &LeaderboardQuery) -> Result<Vec<LeaderboardEntry>> {
        let entries = self.entries.read().await;
        let mut results: Vec<_> = entries.values().cloned().collect();

        // Apply filters
        results = query.apply_filters(results);

        // Apply sorting
        results.sort_by(|a, b| query.ranking_criteria.compare(a, b));

        // Apply limit
        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    async fn update(&self, entry: &LeaderboardEntry) -> Result<()> {
        let mut entries = self.entries.write().await;
        entries.insert(entry.id, entry.clone());
        Ok(())
    }

    async fn delete(&self, id: Uuid) -> Result<()> {
        let mut entries = self.entries.write().await;
        entries.remove(&id);
        Ok(())
    }

    async fn list_all(&self) -> Result<Vec<LeaderboardEntry>> {
        let entries = self.entries.read().await;
        Ok(entries.values().cloned().collect())
    }

    async fn clear(&self) -> Result<()> {
        let mut entries = self.entries.write().await;
        entries.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaderboard::*;
    use tempfile::TempDir;

    async fn create_test_entry() -> LeaderboardEntry {
        LeaderboardEntry {
            id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            model_name: "test_model".to_string(),
            model_version: "1.0".to_string(),
            benchmark_name: "test_benchmark".to_string(),
            category: LeaderboardCategory::Inference,
            hardware: HardwareInfo {
                cpu: "Test CPU".to_string(),
                cpu_cores: 8,
                gpu: None,
                gpu_count: None,
                memory_gb: 16.0,
                accelerator: None,
                platform: "test".to_string(),
            },
            software: SoftwareInfo {
                framework_version: "0.1.0".to_string(),
                rust_version: "1.75".to_string(),
                os: "Test OS".to_string(),
                optimization_level: OptimizationLevel::O2,
                precision: Precision::FP32,
                quantization: None,
                compiler_flags: vec![],
            },
            metrics: PerformanceMetrics {
                latency_ms: 50.0,
                latency_percentiles: LatencyPercentiles {
                    p50: 45.0,
                    p90: 60.0,
                    p95: 70.0,
                    p99: 85.0,
                    p999: 100.0,
                },
                throughput: Some(20.0),
                tokens_per_second: None,
                memory_mb: Some(512.0),
                peak_memory_mb: Some(768.0),
                gpu_utilization: None,
                accuracy: None,
                energy_watts: None,
                custom_metrics: HashMap::new(),
            },
            metadata: HashMap::new(),
            validated: true,
            submitter: SubmitterInfo {
                name: "Test User".to_string(),
                organization: None,
                email: None,
                github: None,
            },
            tags: vec!["test".to_string()],
        }
    }

    #[tokio::test]
    async fn test_file_storage() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileStorage::new(temp_dir.path()).await.unwrap();

        let entry = create_test_entry().await;

        // Test store
        storage.store(&entry).await.unwrap();

        // Test get
        let retrieved = storage.get(entry.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, entry.id);

        // Test list_all
        let all = storage.list_all().await.unwrap();
        assert_eq!(all.len(), 1);

        // Test delete
        storage.delete(entry.id).await.unwrap();
        let deleted = storage.get(entry.id).await.unwrap();
        assert!(deleted.is_none());
    }

    #[tokio::test]
    async fn test_memory_storage() {
        let storage = MemoryStorage::new();

        let entry = create_test_entry().await;

        // Test store
        storage.store(&entry).await.unwrap();

        // Test get
        let retrieved = storage.get(entry.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, entry.id);

        // Test query
        let query = LeaderboardQuery::default();
        let results = storage.query(&query).await.unwrap();
        assert_eq!(results.len(), 1);

        // Test clear
        storage.clear().await.unwrap();
        let all = storage.list_all().await.unwrap();
        assert_eq!(all.len(), 0);
    }
}
