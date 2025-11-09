//! Distributed Cache Implementation
//!
//! Provides distributed caching capabilities with consistency guarantees.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, trace, warn};

use super::config::{ConsistencyLevel, DistributedConfig};

/// Cache node in the distributed system
#[derive(Debug, Clone)]
pub struct CacheNode {
    pub id: String,
    pub address: String,
    pub is_healthy: bool,
    pub last_health_check: u64,
}

/// Consistent hashing ring for node selection
pub struct ConsistentHashing {
    nodes: Vec<CacheNode>,
    ring: HashMap<u64, String>, // hash -> node_id
}

impl ConsistentHashing {
    pub fn new(nodes: Vec<CacheNode>) -> Self {
        let mut ring = HashMap::new();

        // Simple implementation - each node gets multiple positions
        for node in &nodes {
            for i in 0..100 {
                let hash = self::hash_key(&format!("{}-{}", node.id, i));
                ring.insert(hash, node.id.clone());
            }
        }

        Self { nodes, ring }
    }

    /// Get node for a given key
    pub fn get_node(&self, key: &str) -> Option<&CacheNode> {
        let key_hash = hash_key(key);

        // Find the first node with hash >= key_hash
        let mut min_hash = u64::MAX;
        let mut selected_node_id = None;

        for (&node_hash, node_id) in &self.ring {
            if node_hash >= key_hash && node_hash < min_hash {
                min_hash = node_hash;
                selected_node_id = Some(node_id);
            }
        }

        // If no node found, wrap around to the smallest hash
        if selected_node_id.is_none() {
            if let Some((&_, node_id)) = self.ring.iter().min_by_key(|(&hash, _)| hash) {
                selected_node_id = Some(node_id);
            }
        }

        selected_node_id.and_then(|id| self.nodes.iter().find(|n| &n.id == id))
    }
}

/// Replication strategy
#[derive(Debug, Clone)]
pub enum ReplicationStrategy {
    None,
    Master { replicas: usize },
    MultiMaster,
}

/// Distributed cache cluster
pub struct CacheCluster {
    nodes: Vec<CacheNode>,
    consistent_hash: ConsistentHashing,
    replication: ReplicationStrategy,
    config: DistributedConfig,
    // In-memory storage simulating distributed nodes
    node_storage: Arc<RwLock<HashMap<String, HashMap<String, Vec<u8>>>>>,
}

impl CacheCluster {
    pub fn new(config: DistributedConfig) -> Self {
        let nodes: Vec<CacheNode> = config
            .nodes
            .iter()
            .enumerate()
            .map(|(i, addr)| CacheNode {
                id: format!("node-{}", i),
                address: addr.clone(),
                is_healthy: true,
                last_health_check: 0,
            })
            .collect();

        let consistent_hash = ConsistentHashing::new(nodes.clone());

        // Initialize storage for each node
        let mut storage = HashMap::new();
        for node in &nodes {
            storage.insert(node.id.clone(), HashMap::new());
        }

        Self {
            nodes,
            consistent_hash,
            replication: ReplicationStrategy::Master {
                replicas: config.replication_factor,
            },
            config,
            node_storage: Arc::new(RwLock::new(storage)),
        }
    }

    /// Get value from distributed cache
    pub async fn get(&self, key: &str) -> Option<Vec<u8>> {
        // Find the primary node for this key
        let primary_node = self.consistent_hash.get_node(key)?;

        // Get storage and try to read from primary node
        let storage = self.node_storage.read().await;
        if let Some(node_cache) = storage.get(&primary_node.id) {
            if let Some(value) = node_cache.get(key) {
                return Some(value.clone());
            }
        }

        // If primary node doesn't have the value, try replica nodes
        let replica_nodes = self.get_replica_nodes(key);
        for node in replica_nodes {
            if let Some(node_cache) = storage.get(&node.id) {
                if let Some(value) = node_cache.get(key) {
                    return Some(value.clone());
                }
            }
        }

        None
    }

    /// Put value in distributed cache
    pub async fn put(&self, key: &str, value: Vec<u8>) -> Result<()> {
        // Find the primary node for this key
        let primary_node = self
            .consistent_hash
            .get_node(key)
            .ok_or_else(|| anyhow::anyhow!("No available nodes for key: {}", key))?;

        let mut storage = self.node_storage.write().await;

        // Store on primary node
        if let Some(node_cache) = storage.get_mut(&primary_node.id) {
            node_cache.insert(key.to_string(), value.clone());
        }

        // Store on replica nodes based on replication strategy
        let replica_nodes = self.get_replica_nodes(key);
        let max_replicas = match &self.replication {
            ReplicationStrategy::None => 0,
            ReplicationStrategy::Master { replicas } => *replicas,
            ReplicationStrategy::MultiMaster => self.nodes.len().saturating_sub(1),
        };

        for (i, node) in replica_nodes.iter().enumerate() {
            if i >= max_replicas {
                break;
            }
            if let Some(node_cache) = storage.get_mut(&node.id) {
                node_cache.insert(key.to_string(), value.clone());
            }
        }

        Ok(())
    }

    /// Remove value from distributed cache
    pub async fn remove(&self, key: &str) -> Result<()> {
        // Find the primary node for this key
        let primary_node = self
            .consistent_hash
            .get_node(key)
            .ok_or_else(|| anyhow::anyhow!("No available nodes for key: {}", key))?;

        let mut storage = self.node_storage.write().await;

        // Remove from primary node
        if let Some(node_cache) = storage.get_mut(&primary_node.id) {
            node_cache.remove(key);
        }

        // Remove from replica nodes
        let replica_nodes = self.get_replica_nodes(key);
        for node in replica_nodes {
            if let Some(node_cache) = storage.get_mut(&node.id) {
                node_cache.remove(key);
            }
        }

        Ok(())
    }

    /// Get replica nodes for a given key
    fn get_replica_nodes(&self, key: &str) -> Vec<&CacheNode> {
        let key_hash = hash_key(key);
        let mut replica_nodes = Vec::new();

        // Sort nodes by their distance from the key hash
        let mut node_distances: Vec<(u64, &CacheNode)> = self
            .nodes
            .iter()
            .map(|node| {
                let node_hash = hash_key(&node.id);
                let distance = if node_hash >= key_hash {
                    node_hash - key_hash
                } else {
                    (u64::MAX - key_hash) + node_hash
                };
                (distance, node)
            })
            .collect();

        node_distances.sort_by_key(|(distance, _)| *distance);

        // Skip the first node (primary) and return the rest as replicas
        for (_, node) in node_distances.iter().skip(1) {
            if node.is_healthy {
                replica_nodes.push(*node);
            }
        }

        replica_nodes
    }

    /// Clear all data from all nodes in the cluster
    pub async fn clear_all(&self) -> Result<()> {
        let mut storage = self.node_storage.write().await;

        // Clear all data from all nodes
        for (node_id, node_cache) in storage.iter_mut() {
            node_cache.clear();
            trace!("Cleared cache for node: {}", node_id);
        }

        info!("Cleared all data from {} cache nodes", self.nodes.len());
        Ok(())
    }

    /// Get the number of nodes in the cluster
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Main distributed cache service
pub struct DistributedCache {
    cluster: Arc<RwLock<CacheCluster>>,
    config: DistributedConfig,
}

impl DistributedCache {
    pub fn new(config: DistributedConfig) -> Self {
        let cluster = CacheCluster::new(config.clone());

        Self {
            cluster: Arc::new(RwLock::new(cluster)),
            config,
        }
    }

    /// Check health of a cache node
    async fn check_node_health(address: &str) -> bool {
        // Simple health check implementation
        // In a real distributed system, this would make an HTTP request or TCP connection
        // to the actual node to verify it's responsive

        // For simulation purposes, we'll use a simple heuristic:
        // - Consider localhost addresses as always healthy
        // - For other addresses, simulate occasional failures (5% failure rate)

        if address.starts_with("127.0.0.1") || address.starts_with("localhost") {
            // Localhost nodes are considered always healthy
            true
        } else {
            // Simulate network health checks with occasional failures
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            address.hash(&mut hasher);
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            current_time.hash(&mut hasher);

            // Use hash to create pseudo-random health status
            // This simulates real network conditions where nodes might occasionally be unhealthy
            let health_score = hasher.finish() % 100;
            health_score > 5 // 95% chance of being healthy
        }
    }

    /// Start the distributed cache service
    pub async fn start(&self) -> Result<()> {
        info!(
            "Starting distributed cache service with {} nodes",
            self.config.nodes.len()
        );

        // Start health checking background task
        let cluster_clone = Arc::clone(&self.cluster);
        let health_check_interval = std::time::Duration::from_secs(30); // Check every 30 seconds

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(health_check_interval);
            loop {
                interval.tick().await;

                // Perform health check on all nodes
                {
                    let mut cluster = cluster_clone.write().await;
                    let current_time = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();

                    for node in &mut cluster.nodes {
                        // Simple health check - in a real implementation this would
                        // ping the actual node endpoints
                        let was_healthy = node.is_healthy;

                        // Simulate health check (in production this would be an actual HTTP/TCP check)
                        node.is_healthy = Self::check_node_health(&node.address).await;
                        node.last_health_check = current_time;

                        if was_healthy != node.is_healthy {
                            if node.is_healthy {
                                info!(
                                    "Node {} ({}) recovered and is now healthy",
                                    node.id, node.address
                                );
                            } else {
                                warn!("Node {} ({}) is now unhealthy", node.id, node.address);
                            }
                        }
                    }

                    // Update consistent hashing ring if needed
                    let healthy_nodes: Vec<CacheNode> =
                        cluster.nodes.iter().filter(|n| n.is_healthy).cloned().collect();

                    if !healthy_nodes.is_empty() {
                        cluster.consistent_hash = ConsistentHashing::new(healthy_nodes);
                    }
                }

                trace!("Completed health check for all cache nodes");
            }
        });

        info!("Distributed cache service started successfully");
        Ok(())
    }

    /// Get value from cache
    pub async fn get(&self, key: &str) -> Option<Vec<u8>> {
        let cluster = self.cluster.read().await;
        cluster.get(key).await
    }

    /// Put value in cache
    pub async fn put(&self, key: &str, value: Vec<u8>) -> Result<()> {
        let cluster = self.cluster.read().await;
        cluster.put(key, value).await
    }

    /// Invalidate all caches
    pub async fn invalidate_all(&self) -> Result<()> {
        let cluster = self.cluster.write().await;
        cluster.clear_all().await?;
        info!(
            "Invalidated all caches across {} nodes",
            cluster.node_count()
        );
        Ok(())
    }

    /// Update configuration
    pub async fn update_config(&self, config: DistributedConfig) -> Result<()> {
        info!("Updating distributed cache configuration");

        let mut cluster = self.cluster.write().await;

        // Update cluster configuration
        cluster.config = config.clone();

        // Rebuild nodes if the node list changed
        let new_nodes: Vec<CacheNode> = config
            .nodes
            .iter()
            .enumerate()
            .map(|(i, addr)| {
                // Try to preserve health status for existing nodes
                let existing_node = cluster.nodes.iter().find(|n| n.address == *addr);

                CacheNode {
                    id: format!("node-{}", i),
                    address: addr.clone(),
                    is_healthy: existing_node.map(|n| n.is_healthy).unwrap_or(true),
                    last_health_check: existing_node.map(|n| n.last_health_check).unwrap_or(0),
                }
            })
            .collect();

        // Update nodes and rebuild consistent hashing
        cluster.nodes = new_nodes.clone();
        cluster.consistent_hash = ConsistentHashing::new(new_nodes);

        // Update replication strategy based on config
        cluster.replication = match config.consistency_level {
            ConsistencyLevel::Eventual => ReplicationStrategy::Master { replicas: 1 },
            ConsistencyLevel::Strong => ReplicationStrategy::Master {
                replicas: (cluster.nodes.len() / 2).max(1),
            },
            ConsistencyLevel::Weak => ReplicationStrategy::None,
            ConsistencyLevel::Session => ReplicationStrategy::Master {
                replicas: 2, // Session consistency requires at least 2 replicas for session affinity
            },
        };

        // Initialize storage for any new nodes
        let mut storage = cluster.node_storage.write().await;
        for node in &cluster.nodes {
            storage.entry(node.id.clone()).or_default();
        }

        // Remove storage for nodes that are no longer present
        let current_node_ids: std::collections::HashSet<String> =
            cluster.nodes.iter().map(|n| n.id.clone()).collect();

        storage.retain(|node_id, _| current_node_ids.contains(node_id));

        info!(
            "Successfully updated distributed cache configuration with {} nodes",
            cluster.nodes.len()
        );
        Ok(())
    }

    /// Get distributed cache statistics
    pub async fn get_stats(&self) -> Result<DistributedCacheStats> {
        let cluster = self.cluster.read().await;
        let healthy_nodes = cluster.nodes.iter().filter(|n| n.is_healthy).count();

        let replication_factor = match cluster.replication {
            ReplicationStrategy::None => 1,
            ReplicationStrategy::Master { replicas } => replicas + 1,
            ReplicationStrategy::MultiMaster => cluster.nodes.len(),
        };

        Ok(DistributedCacheStats {
            node_count: cluster.nodes.len(),
            healthy_nodes,
            replication_factor,
            consistency_level: cluster.config.consistency_level.clone(),
        })
    }
}

/// Distributed cache statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct DistributedCacheStats {
    pub node_count: usize,
    pub healthy_nodes: usize,
    pub replication_factor: usize,
    pub consistency_level: ConsistencyLevel,
}

/// Simple hash function
fn hash_key(key: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}
