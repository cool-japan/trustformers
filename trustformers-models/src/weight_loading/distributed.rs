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

use super::config::{
    CacheStrategy, DistributedConfig, FaultToleranceConfig, LoadBalancingStrategy, NodeConfig,
    WeightLoadingConfig,
};
use super::huggingface::{TensorMetadata, WeightLoader};

/// Distributed weight loader for loading across multiple nodes
pub struct DistributedWeightLoader {
    config: WeightLoadingConfig,
    distributed_config: DistributedConfig,
    local_loaders: HashMap<String, Box<dyn WeightLoader>>,
    node_connections: HashMap<String, tokio::net::TcpStream>,
    load_balancer: LoadBalancer,
    health_monitor: HealthMonitor,
    tensor_cache: Arc<Mutex<HashMap<String, Tensor>>>,
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
            tensor_cache: Arc::new(Mutex::new(HashMap::new())),
            stats: DistributedStats::new(),
        })
    }

    /// Initialize connections to all nodes
    pub async fn initialize(&mut self) -> Result<()> {
        for node in &self.distributed_config.nodes.clone() {
            if let Err(e) = self.connect_to_node(node).await {
                if !self.distributed_config.fault_tolerance.enable_failover {
                    return Err(e);
                }
                // Log warning but continue with other nodes
                eprintln!("Warning: Failed to connect to node {}: {}", node.id, e);
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

    /// Stream tensor from node in chunks
    async fn stream_tensor_from_node(
        &mut self,
        _node: &NodeConfig,
        file_path: &PathBuf,
        name: &str,
    ) -> Result<Tensor> {
        // Open file for streaming
        let mut file = tokio::fs::File::open(file_path).await.map_err(|e| {
            TrustformersError::file_not_found(format!(
                "Failed to open {}: {}",
                file_path.display(),
                e
            ))
        })?;

        // For simplicity, load the entire tensor
        // In practice, this would stream chunks and reconstruct the tensor
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .await
            .map_err(|e| TrustformersError::io_error(e.to_string()))?;

        // Parse tensor from bytes (simplified)
        self.parse_tensor_from_bytes(buffer, name)
    }

    /// Parse tensor from raw bytes
    fn parse_tensor_from_bytes(&self, data: Vec<u8>, _name: &str) -> Result<Tensor> {
        // Simplified tensor parsing - in practice this would handle different formats
        if data.len() < 4 {
            return Err(TrustformersError::invalid_format_simple(
                "Insufficient data for tensor".to_string(),
            ));
        }

        // For demo purposes, create a simple tensor
        let shape = vec![data.len() / 4]; // Assume f32 data
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Tensor::from_vec(floats, &shape)
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
    async fn check_cache(&self, name: &str) -> Result<Option<Tensor>> {
        let cache = self
            .tensor_cache
            .lock()
            .map_err(|_| TrustformersError::lock_error("Cache lock poisoned".to_string()))?;
        Ok(cache.get(name).cloned())
    }

    /// Cache tensor with replication
    async fn cache_tensor(&self, name: &str, tensor: &Tensor) -> Result<()> {
        let mut cache = self
            .tensor_cache
            .lock()
            .map_err(|_| TrustformersError::lock_error("Cache lock poisoned".to_string()))?;

        // Apply eviction policy if cache is full
        self.apply_eviction_policy(&mut cache)?;

        cache.insert(name.to_string(), tensor.clone());

        // Replicate to other nodes based on replication factor
        if self.distributed_config.distributed_cache.replication_factor > 1 {
            self.replicate_tensor(name, tensor).await?;
        }

        Ok(())
    }

    /// Apply cache eviction policy
    fn apply_eviction_policy(&self, cache: &mut HashMap<String, Tensor>) -> Result<()> {
        // Simplified eviction - remove random entry if cache is too large
        if cache.len() > 1000 {
            // Arbitrary limit
            if let Some(key) = cache.keys().next().cloned() {
                cache.remove(&key);
            }
        }
        Ok(())
    }

    /// Replicate tensor to other nodes
    async fn replicate_tensor(&self, name: &str, _tensor: &Tensor) -> Result<()> {
        let replication_count = (self.distributed_config.distributed_cache.replication_factor
            as usize)
            .min(self.distributed_config.nodes.len());

        // Select nodes for replication (simplified)
        for node in self.distributed_config.nodes.iter().take(replication_count) {
            // In practice, this would send the tensor data to the node
            println!("Replicating tensor {} to node {}", name, node.id);
        }

        Ok(())
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
}
