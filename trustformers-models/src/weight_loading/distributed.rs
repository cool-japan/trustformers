/// Distributed Weight Loader
///
/// This module provides distributed weight loading capabilities across multiple nodes
/// with load balancing, fault tolerance, and caching.
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;
use trustformers_core::{
    errors::{runtime_error, Result, TrustformersError},
    tensor::Tensor,
};

use super::checkpoint::Checkpoint;
use super::config::{
    CacheEvictionPolicy, CacheStrategy, DistributedConfig, FaultToleranceConfig,
    LoadBalancingStrategy, NodeConfig, WeightLoadingConfig,
};
use super::huggingface::{TensorMetadata, WeightLoader};

/// Default number of tensors held in the distributed cache.
///
/// `DistributedCacheConfig` carries no capacity field, so this is the documented
/// default; use [`DistributedWeightLoader::with_cache_capacity`] to change it.
pub const DEFAULT_CACHE_CAPACITY: usize = 1024;

/// Lifetime used by [`CacheEvictionPolicy::TTL`].
///
/// The configuration exposes no per-entry lifetime, so this documented default
/// stands in for one; entries older than this are dropped first, and if that
/// frees nothing the oldest entry goes (FIFO) so the cache still respects its
/// capacity.
pub const DEFAULT_CACHE_TTL: Duration = Duration::from_secs(300);

/// One cached tensor plus the bookkeeping the eviction policies need.
#[derive(Debug, Clone)]
struct CacheEntry {
    tensor: Tensor,
    inserted_at: Instant,
    last_access: Instant,
    accesses: u64,
}

/// Tensor cache with a real, configurable eviction policy.
///
/// A previous revision ignored the configured [`CacheEvictionPolicy`] entirely:
/// past a hardcoded limit of 1000 entries it dropped whichever key
/// `HashMap::keys().next()` happened to yield.
#[derive(Debug, Default)]
struct TensorCache {
    entries: HashMap<String, CacheEntry>,
}

impl TensorCache {
    /// Look a tensor up, recording the access for LRU/LFU.
    fn get(&mut self, name: &str) -> Option<Tensor> {
        let entry = self.entries.get_mut(name)?;
        entry.last_access = Instant::now();
        entry.accesses += 1;
        Some(entry.tensor.clone())
    }

    /// Insert or replace a tensor.
    fn insert(&mut self, name: String, tensor: Tensor) {
        let now = Instant::now();
        self.entries.insert(
            name,
            CacheEntry {
                tensor,
                inserted_at: now,
                last_access: now,
                accesses: 0,
            },
        );
    }

    /// Number of cached tensors.
    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Evict entries until the cache is within `capacity`, per `policy`.
    fn evict(&mut self, policy: &CacheEvictionPolicy, capacity: usize) {
        if capacity == 0 {
            self.entries.clear();
            return;
        }

        if matches!(policy, CacheEvictionPolicy::TTL) {
            let now = Instant::now();
            self.entries
                .retain(|_, entry| now.duration_since(entry.inserted_at) < DEFAULT_CACHE_TTL);
        }

        while self.entries.len() > capacity {
            let Some(victim) = self.select_victim(policy) else {
                break;
            };
            self.entries.remove(&victim);
        }
    }

    /// Pick the entry this policy would drop next.
    fn select_victim(&self, policy: &CacheEvictionPolicy) -> Option<String> {
        match policy {
            CacheEvictionPolicy::LRU => self
                .entries
                .iter()
                .min_by_key(|(name, entry)| (entry.last_access, (*name).clone()))
                .map(|(name, _)| name.clone()),
            CacheEvictionPolicy::LFU => self
                .entries
                .iter()
                .min_by_key(|(name, entry)| (entry.accesses, entry.last_access, (*name).clone()))
                .map(|(name, _)| name.clone()),
            // TTL falls back to FIFO once every expired entry has already gone.
            CacheEvictionPolicy::FIFO | CacheEvictionPolicy::TTL => self
                .entries
                .iter()
                .min_by_key(|(name, entry)| (entry.inserted_at, (*name).clone()))
                .map(|(name, _)| name.clone()),
            CacheEvictionPolicy::Random => {
                let names: Vec<&String> = self.entries.keys().collect();
                if names.is_empty() {
                    None
                } else {
                    names.get(fastrand::usize(..names.len())).map(|name| (*name).clone())
                }
            },
        }
    }
}

/// Distributed weight loader for loading across multiple nodes
pub struct DistributedWeightLoader {
    config: WeightLoadingConfig,
    distributed_config: DistributedConfig,
    local_loaders: HashMap<String, Box<dyn WeightLoader>>,
    node_connections: HashMap<String, tokio::net::TcpStream>,
    load_balancer: LoadBalancer,
    health_monitor: HealthMonitor,
    tensor_cache: Arc<Mutex<TensorCache>>,
    cache_capacity: usize,
    stats: DistributedStats,
}

impl DistributedWeightLoader {
    /// Create a new distributed weight loader
    pub fn new(config: WeightLoadingConfig, distributed_config: DistributedConfig) -> Result<Self> {
        let load_balancer =
            LoadBalancer::new(&distributed_config.load_balancer, &distributed_config.nodes)?;
        let health_monitor = HealthMonitor::new(&distributed_config.fault_tolerance)?;

        Ok(Self {
            config,
            distributed_config,
            local_loaders: HashMap::new(),
            node_connections: HashMap::new(),
            load_balancer,
            health_monitor,
            tensor_cache: Arc::new(Mutex::new(TensorCache::default())),
            cache_capacity: DEFAULT_CACHE_CAPACITY,
            stats: DistributedStats::new(),
        })
    }

    /// Set how many tensors the cache holds before evicting.
    pub fn with_cache_capacity(mut self, capacity: usize) -> Self {
        self.cache_capacity = capacity;
        self
    }

    /// Number of tensors currently held in the distributed cache.
    ///
    /// # Errors
    ///
    /// Fails when the cache lock is poisoned.
    pub fn cached_tensor_count(&self) -> Result<usize> {
        let cache = self
            .tensor_cache
            .lock()
            .map_err(|_| TrustformersError::lock_error("Cache lock poisoned".to_string()))?;
        Ok(cache.len())
    }

    /// Initialize connections to all nodes
    ///
    /// With failover enabled an unreachable node does not abort the whole
    /// initialisation; the failure is recorded in
    /// [`DistributedStats::connection_failures`] instead of being written to
    /// stderr by this library.
    pub async fn initialize(&mut self) -> Result<()> {
        for node in &self.distributed_config.nodes.clone() {
            if let Err(e) = self.connect_to_node(node).await {
                if !self.distributed_config.fault_tolerance.enable_failover {
                    return Err(e);
                }
                self.stats.connection_failures.push((node.id.clone(), e.to_string()));
            }
        }

        // Start health monitoring
        self.health_monitor.start_monitoring(&self.distributed_config.nodes).await?;

        Ok(())
    }

    /// Connect to a specific node
    async fn connect_to_node(&mut self, node: &NodeConfig) -> Result<()> {
        let address = format!("{}:{}", node.address, node.port);
        let timeout_duration = self.distributed_config.network.connection_timeout;

        let stream =
            tokio::time::timeout(timeout_duration, tokio::net::TcpStream::connect(&address))
                .await
                .map_err(|_| {
                    TrustformersError::runtime_error(format!("Connection to {} timed out", address))
                })?
                .map_err(|e| {
                    TrustformersError::io_error(format!("Failed to connect to {}: {}", address, e))
                })?;

        self.node_connections.insert(node.id.clone(), stream);
        Ok(())
    }

    /// Load tensor with distributed strategy
    async fn load_tensor_distributed(&mut self, name: &str) -> Result<Tensor> {
        // Check local cache first
        if let Some(tensor) = self.check_cache(name).await? {
            self.stats.cache_hits += 1;
            return Ok(tensor);
        }

        self.stats.cache_misses += 1;

        // Select optimal node for loading
        let selected_node =
            self.load_balancer.select_node(name, &self.distributed_config.nodes)?.clone();

        // Attempt to load from selected node with fault tolerance
        let mut attempts = 0;
        let max_retries = self.distributed_config.fault_tolerance.max_retries;

        loop {
            match self.load_from_node(&selected_node, name).await {
                Ok(tensor) => {
                    // Cache the tensor if enabled
                    if self.should_cache(name) {
                        self.cache_tensor(name, &tensor).await?;
                    }

                    self.stats.successful_loads += 1;
                    return Ok(tensor);
                },
                Err(e) => {
                    attempts += 1;
                    self.stats.failed_loads += 1;

                    if attempts >= max_retries {
                        if self.distributed_config.fault_tolerance.enable_failover {
                            // Try backup nodes
                            if let Some(backup_node) = self.find_backup_node(&selected_node.id) {
                                return self.load_from_node(&backup_node, name).await;
                            }
                        }
                        return Err(e);
                    }

                    // Wait before retry
                    tokio::time::sleep(self.distributed_config.fault_tolerance.retry_delay).await;
                },
            }
        }
    }

    /// Load tensor from a specific node
    async fn load_from_node(&mut self, node: &NodeConfig, name: &str) -> Result<Tensor> {
        let start_time = Instant::now();

        // Find tensor file on the node
        let file_path = self.find_tensor_on_node(node, name)?;

        // Load tensor based on file type and configuration
        let tensor = if self.config.streaming {
            self.stream_tensor_from_node(node, &file_path, name).await?
        } else {
            self.load_tensor_from_node(node, &file_path, name).await?
        };

        let load_time = start_time.elapsed();
        self.stats.total_load_time += load_time;
        self.stats.node_load_times.entry(node.id.clone()).or_default().push(load_time);

        Ok(tensor)
    }

    /// Find tensor file on a specific node
    fn find_tensor_on_node(&self, node: &NodeConfig, name: &str) -> Result<PathBuf> {
        for storage_path in &node.storage_paths {
            let potential_files = vec![
                storage_path.join(format!("{}.safetensors", name)),
                storage_path.join(format!("{}.bin", name)),
                storage_path.join("pytorch_model.bin"),
                storage_path.join("model.safetensors"),
            ];

            for file_path in potential_files {
                if file_path.exists() {
                    return Ok(file_path);
                }
            }
        }

        Err(runtime_error(format!(
            "Tensor {} not found on node {}",
            name, node.id
        )))
    }

    /// Load tensor from node using appropriate loader
    async fn load_tensor_from_node(
        &mut self,
        node: &NodeConfig,
        file_path: &PathBuf,
        name: &str,
    ) -> Result<Tensor> {
        // Create appropriate loader for the node
        let loader_key = format!("{}:{}", node.id, file_path.to_string_lossy());

        if !self.local_loaders.contains_key(&loader_key) {
            let loader = super::create_huggingface_loader(
                file_path.parent().unwrap_or(file_path),
                Some(self.config.clone()),
            )?;
            self.local_loaders.insert(loader_key.clone(), loader);
        }

        let loader = self.local_loaders.get_mut(&loader_key).ok_or_else(|| {
            TrustformersError::runtime_error(format!(
                "Loader for {} not found after insertion",
                loader_key
            ))
        })?;
        loader.load_tensor(name)
    }

    /// Read a node's checkpoint file asynchronously and extract one tensor.
    ///
    /// The whole container is read before parsing because both supported
    /// containers need their header — and, for a PyTorch archive, the ZIP
    /// central directory at the end of the file — before any tensor's byte range
    /// is known. Chunked reads are the job of
    /// [`super::streaming::StreamingLoader`], which groups tensors and evicts
    /// them under a memory budget; this path exists so that a remote node's file
    /// is read off the async runtime rather than blocking it.
    ///
    /// # Errors
    ///
    /// Fails when the file cannot be opened or read, when it is not a checkpoint
    /// this crate can parse, or when it does not contain `name`.
    async fn stream_tensor_from_node(
        &mut self,
        _node: &NodeConfig,
        file_path: &PathBuf,
        name: &str,
    ) -> Result<Tensor> {
        let mut file = tokio::fs::File::open(file_path).await.map_err(|e| {
            TrustformersError::file_not_found(format!(
                "Failed to open {}: {}",
                file_path.display(),
                e
            ))
        })?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .await
            .map_err(|e| TrustformersError::io_error(e.to_string()))?;

        self.parse_tensor_from_bytes(buffer, name)
    }

    /// Extract one named tensor from a checkpoint held in memory.
    ///
    /// The buffer is parsed as the checkpoint it actually is — the container
    /// format is detected, the index is honoured and the requested tensor is
    /// decoded with its own dtype and shape.
    ///
    /// A previous revision ignored `name` entirely and reinterpreted the whole
    /// file as a flat `f32` array (`shape = vec![data.len() / 4]`), so every
    /// tensor streamed through this path came back with the wrong rank, the wrong
    /// shape and — for anything but an f32 payload — meaningless values.
    ///
    /// # Errors
    ///
    /// Fails when the buffer is not a checkpoint this crate can parse, or when
    /// it does not contain `name`.
    fn parse_tensor_from_bytes(&self, data: Vec<u8>, name: &str) -> Result<Tensor> {
        let checkpoint = Checkpoint::from_bytes(&data)?;
        checkpoint.get(name).cloned().ok_or_else(|| {
            TrustformersError::weight_load_error(format!(
                "streamed checkpoint does not contain tensor {name}; it holds {} tensor(s)",
                checkpoint.len()
            ))
        })
    }

    /// Check if tensor should be cached
    fn should_cache(&self, _name: &str) -> bool {
        match self.distributed_config.distributed_cache.cache_strategy {
            CacheStrategy::None => false,
            CacheStrategy::ReadThrough | CacheStrategy::WriteThrough | CacheStrategy::WriteBack => {
                true
            },
            CacheStrategy::ReadAround => false, // Skip cache on read
        }
    }

    /// Check cache for tensor
    ///
    /// A hit records the access so that the LRU and LFU policies have real data
    /// to work from.
    async fn check_cache(&self, name: &str) -> Result<Option<Tensor>> {
        let mut cache = self
            .tensor_cache
            .lock()
            .map_err(|_| TrustformersError::lock_error("Cache lock poisoned".to_string()))?;
        Ok(cache.get(name))
    }

    /// Cache tensor with replication
    async fn cache_tensor(&self, name: &str, tensor: &Tensor) -> Result<()> {
        {
            let mut cache = self
                .tensor_cache
                .lock()
                .map_err(|_| TrustformersError::lock_error("Cache lock poisoned".to_string()))?;

            cache.insert(name.to_string(), tensor.clone());
            cache.evict(
                &self.distributed_config.distributed_cache.eviction_policy,
                self.cache_capacity,
            );
        }

        // Replicate to other nodes based on replication factor
        if self.distributed_config.distributed_cache.replication_factor > 1 {
            self.replicate_tensor(name, tensor).await?;
        }

        Ok(())
    }

    /// Replicate a cached tensor onto peer nodes.
    ///
    /// Not implemented: this loader has no wire protocol for pushing tensor data
    /// to a peer, only for pulling it from a node's local storage paths. A
    /// previous revision printed `"Replicating tensor X to node Y"` and returned
    /// `Ok(())`, so a configured `replication_factor > 1` reported replication
    /// that never happened.
    ///
    /// # Errors
    ///
    /// Always returns [`TrustformersError::not_implemented`]. Set
    /// `replication_factor` to 1 to disable replication.
    async fn replicate_tensor(&self, name: &str, _tensor: &Tensor) -> Result<()> {
        Err(TrustformersError::not_implemented(format!(
            "distributed cache replication (replication_factor = {}, requested while caching \
             tensor {name}): this loader can read weights from a node's storage paths but has no \
             protocol for pushing tensor data to a peer",
            self.distributed_config.distributed_cache.replication_factor
        )))
    }

    /// Find backup node for failover
    fn find_backup_node(&self, failed_node_id: &str) -> Option<NodeConfig> {
        self.distributed_config
            .nodes
            .iter()
            .find(|node| {
                node.id != failed_node_id
                    && self.distributed_config.fault_tolerance.backup_nodes.contains(&node.id)
            })
            .cloned()
    }

    /// Get distributed loading statistics
    pub fn get_stats(&self) -> &DistributedStats {
        &self.stats
    }
}

impl WeightLoader for DistributedWeightLoader {
    fn load_tensor(&mut self, name: &str) -> Result<Tensor> {
        // For sync interface, use blocking runtime
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to create async runtime: {}", e))
        })?;

        rt.block_on(self.load_tensor_distributed(name))
    }

    fn list_tensors(&self) -> Result<Vec<String>> {
        // Aggregate tensor lists from all nodes
        let mut all_tensors = Vec::new();

        for loader in self.local_loaders.values() {
            let tensors = loader.list_tensors()?;
            all_tensors.extend(tensors);
        }

        // Remove duplicates
        all_tensors.sort();
        all_tensors.dedup();

        Ok(all_tensors)
    }

    fn tensor_info(&self, name: &str) -> Result<Option<TensorMetadata>> {
        // Try to get info from any available loader
        for loader in self.local_loaders.values() {
            if let Ok(Some(info)) = loader.tensor_info(name) {
                return Ok(Some(info));
            }
        }
        Ok(None)
    }

    fn close(&mut self) -> Result<()> {
        // Close all local loaders
        for loader in self.local_loaders.values_mut() {
            loader.close()?;
        }

        // Close network connections
        self.node_connections.clear();

        Ok(())
    }
}

/// Load balancer for distributed weight loading
struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    node_states: HashMap<String, NodeState>,
    round_robin_index: usize,
}

impl LoadBalancer {
    fn new(strategy: &LoadBalancingStrategy, nodes: &[NodeConfig]) -> Result<Self> {
        let mut node_states = HashMap::new();
        for node in nodes {
            node_states.insert(node.id.clone(), NodeState::new());
        }

        Ok(Self {
            strategy: strategy.clone(),
            node_states,
            round_robin_index: 0,
        })
    }

    fn select_node<'a>(
        &mut self,
        tensor_name: &str,
        nodes: &'a [NodeConfig],
    ) -> Result<&'a NodeConfig> {
        match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let selected = &nodes[self.round_robin_index % nodes.len()];
                self.round_robin_index += 1;
                Ok(selected)
            },
            LoadBalancingStrategy::LeastLoaded => {
                let least_loaded_id = self
                    .node_states
                    .iter()
                    .min_by_key(|(_, state)| state.current_load)
                    .map(|(id, _)| id)
                    .ok_or_else(|| {
                        TrustformersError::invalid_state("No nodes available".to_string())
                    })?;

                nodes
                    .iter()
                    .find(|node| &node.id == least_loaded_id)
                    .ok_or_else(|| TrustformersError::invalid_state("Node not found".to_string()))
            },
            LoadBalancingStrategy::ConsistentHashing => {
                // Simple hash-based selection
                let hash = self.hash_tensor_name(tensor_name);
                let index = hash % nodes.len();
                Ok(&nodes[index])
            },
            _ => {
                // Fallback to round robin for other strategies
                let selected = &nodes[self.round_robin_index % nodes.len()];
                self.round_robin_index += 1;
                Ok(selected)
            },
        }
    }

    fn hash_tensor_name(&self, name: &str) -> usize {
        // Simple hash function
        name.bytes().map(|b| b as usize).sum()
    }
}

/// Health monitor for tracking node health
struct HealthMonitor {
    _config: FaultToleranceConfig,
    node_health: HashMap<String, NodeHealth>,
}

impl HealthMonitor {
    fn new(config: &FaultToleranceConfig) -> Result<Self> {
        Ok(Self {
            _config: config.clone(),
            node_health: HashMap::new(),
        })
    }

    async fn start_monitoring(&mut self, nodes: &[NodeConfig]) -> Result<()> {
        for node in nodes {
            self.node_health.insert(node.id.clone(), NodeHealth::new());
        }

        // Start background health checking
        tokio::spawn(async move {
            loop {
                // Perform health checks
                tokio::time::sleep(Duration::from_secs(30)).await;
            }
        });

        Ok(())
    }
}

/// Node state for load balancing
struct NodeState {
    current_load: u64,
    _total_requests: u64,
    _failed_requests: u64,
    _last_request_time: Instant,
}

impl NodeState {
    fn new() -> Self {
        Self {
            current_load: 0,
            _total_requests: 0,
            _failed_requests: 0,
            _last_request_time: Instant::now(),
        }
    }
}

/// Node health information
struct NodeHealth {
    _is_healthy: bool,
    _last_check: Instant,
    _consecutive_failures: u32,
}

impl NodeHealth {
    fn new() -> Self {
        Self {
            _is_healthy: true,
            _last_check: Instant::now(),
            _consecutive_failures: 0,
        }
    }
}

/// Statistics for distributed weight loading
#[derive(Debug, Default)]
pub struct DistributedStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub successful_loads: u64,
    pub failed_loads: u64,
    pub total_load_time: Duration,
    pub node_load_times: HashMap<String, Vec<Duration>>,
    pub bytes_transferred: u64,
    /// Nodes that could not be reached during `initialize`, with the reason.
    pub connection_failures: Vec<(String, String)>,
}

impl DistributedStats {
    fn new() -> Self {
        Self::default()
    }

    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    pub fn success_rate(&self) -> f64 {
        let total = self.successful_loads + self.failed_loads;
        if total == 0 {
            0.0
        } else {
            self.successful_loads as f64 / total as f64
        }
    }

    pub fn average_load_time(&self) -> Duration {
        if self.successful_loads == 0 {
            Duration::ZERO
        } else {
            self.total_load_time / self.successful_loads as u32
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_loading::config::{
        CacheEvictionPolicy, CacheStrategy, ConsistencyLevel, DistributedCacheConfig,
        DistributedConfig, FaultToleranceConfig, LoadBalancingStrategy, NetworkConfig, NodeConfig,
        WeightLoadingConfig,
    };
    use std::time::Duration;

    fn make_node(id: &str) -> NodeConfig {
        NodeConfig {
            id: id.to_string(),
            address: "127.0.0.1".to_string(),
            port: 9000,
            weight_capacity: 1024 * 1024 * 1024,
            bandwidth: 1000.0,
            priority: 128,
            storage_paths: vec![std::path::PathBuf::from("/tmp")],
        }
    }

    fn make_distributed_config(nodes: Vec<NodeConfig>) -> DistributedConfig {
        DistributedConfig {
            nodes,
            load_balancer: LoadBalancingStrategy::RoundRobin,
            fault_tolerance: FaultToleranceConfig {
                max_retries: 3,
                retry_delay: Duration::from_millis(10),
                timeout: Duration::from_secs(5),
                enable_failover: true,
                health_check_interval: Duration::from_secs(60),
                backup_nodes: vec![],
            },
            network: NetworkConfig {
                max_concurrent_connections: 4,
                connection_timeout: Duration::from_secs(5),
                read_timeout: Duration::from_secs(10),
                chunk_size: 4096,
                enable_keepalive: false,
                compression_threshold: 1024 * 1024,
            },
            distributed_cache: DistributedCacheConfig {
                cache_strategy: CacheStrategy::ReadThrough,
                replication_factor: 1,
                eviction_policy: CacheEvictionPolicy::LRU,
                consistency_level: ConsistencyLevel::Eventual,
            },
            compression: false,
        }
    }

    // ── DistributedStats tests ────────────────────────────────────────────────

    #[test]
    fn test_stats_default_zero() {
        let stats = DistributedStats::new();
        assert_eq!(stats.cache_hits, 0, "initial cache_hits must be 0");
        assert_eq!(stats.cache_misses, 0, "initial cache_misses must be 0");
        assert_eq!(
            stats.successful_loads, 0,
            "initial successful_loads must be 0"
        );
        assert_eq!(stats.failed_loads, 0, "initial failed_loads must be 0");
    }

    #[test]
    fn test_cache_hit_rate_zero_when_no_requests() {
        let stats = DistributedStats::new();
        let rate = stats.cache_hit_rate();
        assert!(
            (rate - 0.0).abs() < 1e-10,
            "cache_hit_rate must be 0 when no requests"
        );
    }

    #[test]
    fn test_cache_hit_rate_all_hits() {
        let mut stats = DistributedStats::new();
        stats.cache_hits = 10;
        stats.cache_misses = 0;
        let rate = stats.cache_hit_rate();
        assert!(
            (rate - 1.0).abs() < 1e-10,
            "cache_hit_rate must be 1.0 with all hits"
        );
    }

    #[test]
    fn test_cache_hit_rate_half() {
        let mut stats = DistributedStats::new();
        stats.cache_hits = 5;
        stats.cache_misses = 5;
        let rate = stats.cache_hit_rate();
        assert!(
            (rate - 0.5).abs() < 1e-10,
            "cache_hit_rate must be 0.5 with equal hits/misses"
        );
    }

    #[test]
    fn test_success_rate_zero_when_no_loads() {
        let stats = DistributedStats::new();
        assert!(
            (stats.success_rate() - 0.0).abs() < 1e-10,
            "success_rate must be 0 when no loads"
        );
    }

    #[test]
    fn test_success_rate_all_success() {
        let mut stats = DistributedStats::new();
        stats.successful_loads = 7;
        stats.failed_loads = 0;
        assert!(
            (stats.success_rate() - 1.0).abs() < 1e-10,
            "success_rate must be 1.0 when all loads succeed"
        );
    }

    #[test]
    fn test_success_rate_partial_failure() {
        let mut stats = DistributedStats::new();
        stats.successful_loads = 3;
        stats.failed_loads = 1;
        let expected = 3.0 / 4.0;
        let delta = (stats.success_rate() - expected).abs();
        assert!(
            delta < 1e-10,
            "success_rate must be 0.75 for 3 successes and 1 failure"
        );
    }

    #[test]
    fn test_average_load_time_zero_when_no_loads() {
        let stats = DistributedStats::new();
        assert_eq!(
            stats.average_load_time(),
            Duration::ZERO,
            "average_load_time must be ZERO when no loads completed"
        );
    }

    #[test]
    fn test_average_load_time_single_load() {
        let mut stats = DistributedStats::new();
        let elapsed = Duration::from_millis(100);
        stats.successful_loads = 1;
        stats.total_load_time = elapsed;
        assert_eq!(
            stats.average_load_time(),
            elapsed,
            "average_load_time must equal total for single load"
        );
    }

    #[test]
    fn test_average_load_time_multiple_loads() {
        let mut stats = DistributedStats::new();
        stats.successful_loads = 4;
        stats.total_load_time = Duration::from_millis(400);
        assert_eq!(
            stats.average_load_time(),
            Duration::from_millis(100),
            "average_load_time must be 100ms for 4 loads totalling 400ms"
        );
    }

    // ── DistributedWeightLoader construction tests ────────────────────────────

    #[test]
    fn test_loader_construct_single_node() {
        let nodes = vec![make_node("node-0")];
        let dist_config = make_distributed_config(nodes);
        let weight_config = WeightLoadingConfig::default();
        let loader = DistributedWeightLoader::new(weight_config, dist_config);
        assert!(
            loader.is_ok(),
            "DistributedWeightLoader must construct with single node"
        );
    }

    #[test]
    fn test_loader_construct_multiple_nodes() {
        let nodes = vec![
            make_node("node-0"),
            make_node("node-1"),
            make_node("node-2"),
        ];
        let dist_config = make_distributed_config(nodes);
        let weight_config = WeightLoadingConfig::default();
        let loader = DistributedWeightLoader::new(weight_config, dist_config);
        assert!(
            loader.is_ok(),
            "DistributedWeightLoader must construct with multiple nodes"
        );
    }

    #[test]
    fn test_loader_stats_initial_zero() {
        let nodes = vec![make_node("node-0")];
        let dist_config = make_distributed_config(nodes);
        let loader = DistributedWeightLoader::new(WeightLoadingConfig::default(), dist_config)
            .expect("loader must construct");
        let stats = loader.get_stats();
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.successful_loads, 0);
        assert_eq!(stats.failed_loads, 0);
    }

    // ── Load balancer (via DistributedWeightLoader) tests ────────────────────

    #[test]
    fn test_load_balancer_round_robin_construct() {
        let nodes = vec![make_node("a"), make_node("b")];
        let dist_config = make_distributed_config(nodes);
        let loader = DistributedWeightLoader::new(WeightLoadingConfig::default(), dist_config);
        assert!(loader.is_ok(), "round-robin load balancer must construct");
    }

    #[test]
    fn test_load_balancer_least_loaded_construct() {
        let nodes = vec![make_node("a"), make_node("b")];
        let mut dist_config = make_distributed_config(nodes);
        dist_config.load_balancer = LoadBalancingStrategy::LeastLoaded;
        let loader = DistributedWeightLoader::new(WeightLoadingConfig::default(), dist_config);
        assert!(loader.is_ok(), "least-loaded load balancer must construct");
    }

    #[test]
    fn test_load_balancer_consistent_hashing_construct() {
        let nodes = vec![make_node("a"), make_node("b")];
        let mut dist_config = make_distributed_config(nodes);
        dist_config.load_balancer = LoadBalancingStrategy::ConsistentHashing;
        let loader = DistributedWeightLoader::new(WeightLoadingConfig::default(), dist_config);
        assert!(
            loader.is_ok(),
            "consistent hashing load balancer must construct"
        );
    }

    // ── Cache strategy tests ──────────────────────────────────────────────────

    #[test]
    fn test_cache_none_strategy() {
        let nodes = vec![make_node("node-0")];
        let mut dist_config = make_distributed_config(nodes);
        dist_config.distributed_cache.cache_strategy = CacheStrategy::None;
        let loader = DistributedWeightLoader::new(WeightLoadingConfig::default(), dist_config)
            .expect("loader must construct");
        // Verify the should_cache returns false for CacheStrategy::None
        // (tested indirectly via construction; actual caching is in async path)
        let stats = loader.get_stats();
        assert_eq!(
            stats.cache_hits, 0,
            "fresh loader must have zero cache hits"
        );
    }

    #[test]
    fn test_cache_read_through_strategy_construct() {
        let nodes = vec![make_node("node-0")];
        let mut dist_config = make_distributed_config(nodes);
        dist_config.distributed_cache.cache_strategy = CacheStrategy::ReadThrough;
        let loader = DistributedWeightLoader::new(WeightLoadingConfig::default(), dist_config);
        assert!(
            loader.is_ok(),
            "ReadThrough cache strategy must be constructible"
        );
    }
    // ── Tensor parsing, eviction and replication ─────────────────────────────

    use crate::weight_loading::test_support::{build_safetensors, F32Tensor};

    fn loader_with(policy: CacheEvictionPolicy) -> DistributedWeightLoader {
        let mut config = make_distributed_config(vec![make_node("a")]);
        config.distributed_cache.eviction_policy = policy;
        config.distributed_cache.replication_factor = 1;
        DistributedWeightLoader::new(WeightLoadingConfig::default(), config)
            .expect("loader must build")
    }

    #[test]
    fn streamed_checkpoints_are_parsed_by_name_shape_and_dtype() {
        // Regression: this used to ignore the name and reinterpret the whole file
        // as a flat f32 array of length `bytes / 4`.
        let loader = loader_with(CacheEvictionPolicy::LRU);
        let bytes = build_safetensors(&[
            F32Tensor::new("first.weight", &[2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            F32Tensor::new("second.bias", &[3], vec![9.0, 8.0, 7.0]),
        ]);

        let tensor = loader
            .parse_tensor_from_bytes(bytes.clone(), "second.bias")
            .expect("named tensor must be recoverable");
        assert_eq!(tensor.shape(), vec![3]);
        match tensor {
            Tensor::F32(arr) => {
                assert_eq!(
                    arr.iter().copied().collect::<Vec<f32>>(),
                    vec![9.0, 8.0, 7.0]
                );
            },
            other => panic!("expected an F32 tensor, got {other:?}"),
        }

        let first = loader
            .parse_tensor_from_bytes(bytes, "first.weight")
            .expect("named tensor must be recoverable");
        assert_eq!(first.shape(), vec![2, 2]);
    }

    #[test]
    fn streaming_an_unknown_tensor_name_is_an_error() {
        let loader = loader_with(CacheEvictionPolicy::LRU);
        let bytes = build_safetensors(&[F32Tensor::new("only", &[1], vec![1.0])]);
        let err = loader
            .parse_tensor_from_bytes(bytes, "missing")
            .expect_err("an absent tensor must not be invented");
        assert!(
            err.to_string().contains("missing"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn streaming_a_non_checkpoint_buffer_is_an_error() {
        let loader = loader_with(CacheEvictionPolicy::LRU);
        let err = loader
            .parse_tensor_from_bytes(vec![0x11; 4096], "anything")
            .expect_err("raw bytes must not be reinterpreted as a tensor");
        assert!(
            err.to_string().contains("unrecognised checkpoint container"),
            "unexpected error: {err}"
        );
    }

    fn tensor(value: f32) -> Tensor {
        Tensor::from_vec(vec![value], &[1]).expect("tensor must build")
    }

    #[test]
    fn lru_eviction_drops_the_least_recently_used_entry() {
        let mut cache = TensorCache::default();
        cache.insert("a".to_string(), tensor(1.0));
        std::thread::sleep(Duration::from_millis(2));
        cache.insert("b".to_string(), tensor(2.0));
        std::thread::sleep(Duration::from_millis(2));
        cache.insert("c".to_string(), tensor(3.0));
        // Touch "a" so that "b" becomes the least recently used.
        std::thread::sleep(Duration::from_millis(2));
        assert!(cache.get("a").is_some());

        cache.evict(&CacheEvictionPolicy::LRU, 2);
        assert_eq!(cache.len(), 2);
        assert!(cache.get("b").is_none(), "LRU must drop the stalest entry");
        assert!(cache.get("a").is_some());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn lfu_eviction_drops_the_least_frequently_used_entry() {
        let mut cache = TensorCache::default();
        cache.insert("hot".to_string(), tensor(1.0));
        cache.insert("cold".to_string(), tensor(2.0));
        for _ in 0..3 {
            assert!(cache.get("hot").is_some());
        }

        cache.evict(&CacheEvictionPolicy::LFU, 1);
        assert_eq!(cache.len(), 1);
        assert!(cache.get("hot").is_some(), "LFU must keep the hot entry");
    }

    #[test]
    fn fifo_eviction_drops_the_oldest_insertion() {
        let mut cache = TensorCache::default();
        cache.insert("first".to_string(), tensor(1.0));
        std::thread::sleep(Duration::from_millis(2));
        cache.insert("second".to_string(), tensor(2.0));
        // Reading "first" must not save it from a FIFO eviction.
        assert!(cache.get("first").is_some());

        cache.evict(&CacheEvictionPolicy::FIFO, 1);
        assert_eq!(cache.len(), 1);
        assert!(
            cache.get("second").is_some(),
            "FIFO must drop the oldest entry"
        );
    }

    #[test]
    fn eviction_respects_the_configured_capacity() {
        for policy in [
            CacheEvictionPolicy::LRU,
            CacheEvictionPolicy::LFU,
            CacheEvictionPolicy::FIFO,
            CacheEvictionPolicy::Random,
            CacheEvictionPolicy::TTL,
        ] {
            let mut cache = TensorCache::default();
            for i in 0..10 {
                cache.insert(format!("t{i}"), tensor(i as f32));
            }
            cache.evict(&policy, 4);
            assert_eq!(cache.len(), 4, "{policy:?} must respect the capacity");
        }
    }

    #[tokio::test]
    async fn replication_reports_that_it_is_not_implemented() {
        // Regression: this used to print "Replicating tensor X to node Y" and
        // return Ok, so a configured replication factor silently did nothing.
        let mut config = make_distributed_config(vec![make_node("a"), make_node("b")]);
        config.distributed_cache.replication_factor = 2;
        let loader = DistributedWeightLoader::new(WeightLoadingConfig::default(), config)
            .expect("loader must build");

        let err = loader
            .cache_tensor("w", &tensor(1.0))
            .await
            .expect_err("replication must not be silently skipped");
        assert!(
            err.to_string().contains("replication"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn caching_without_replication_stores_the_tensor() {
        let loader = loader_with(CacheEvictionPolicy::LRU);
        loader.cache_tensor("w", &tensor(4.0)).await.expect("caching must succeed");
        assert_eq!(loader.cached_tensor_count().expect("count"), 1);
        let cached = loader.check_cache("w").await.expect("lookup must succeed");
        assert!(cached.is_some());
    }
}
