//! Failover Management and Load Balancing
//!
//! Provides failover capabilities, load balancing, and node health management
//! for high availability deployments of the inference server.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::RwLock;

/// Failover manager for handling node failures and load balancing
#[derive(Clone)]
pub struct FailoverManager {
    config: FailoverConfig,
    nodes: Arc<RwLock<HashMap<String, NodeHealth>>>,
    load_balancer: Arc<LoadBalancer>,
    stats: Arc<RwLock<FailoverStats>>,
    current_primary: Arc<RwLock<Option<String>>>,
}

impl FailoverManager {
    /// Create a new failover manager
    pub fn new(config: FailoverConfig) -> Self {
        Self {
            config: config.clone(),
            nodes: Arc::new(RwLock::new(HashMap::new())),
            load_balancer: Arc::new(LoadBalancer::new(config.load_balancer_strategy)),
            stats: Arc::new(RwLock::new(FailoverStats::default())),
            current_primary: Arc::new(RwLock::new(None)),
        }
    }

    /// Start monitoring for failover conditions
    pub async fn start_monitoring(&self) -> Result<()> {
        let nodes = self.nodes.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        let primary = self.current_primary.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.health_check_interval);

            loop {
                interval.tick().await;

                // Check node health
                let mut nodes_guard = nodes.write().await;
                let mut failed_nodes = Vec::new();

                for (node_id, node_health) in nodes_guard.iter_mut() {
                    if let Err(_) = Self::check_node_health(node_health).await {
                        node_health.status = NodeStatus::Unhealthy;
                        node_health.last_failure = Some(
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .expect("SystemTime should be after UNIX_EPOCH")
                                .as_secs(),
                        );
                        failed_nodes.push(node_id.clone());
                    }
                }

                // Handle failovers for failed nodes
                for node_id in failed_nodes {
                    if let Some(primary_node) = primary.read().await.as_ref() {
                        if *primary_node == node_id {
                            // Primary node failed, initiate failover
                            if let Some(backup) = Self::find_healthy_backup(&*nodes_guard).await {
                                *primary.write().await = Some(backup.clone());
                                stats.write().await.total_failovers += 1;
                                tracing::warn!("Failover initiated: {} -> {}", node_id, backup);
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Register a new node
    pub async fn register_node(&self, node_id: String, endpoint: String) -> Result<()> {
        let node_health = NodeHealth {
            node_id: node_id.clone(),
            endpoint,
            status: NodeStatus::Healthy,
            last_health_check: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
            last_failure: None,
            response_time_ms: 0.0,
            request_count: 0,
            error_count: 0,
            cpu_usage: 0.0,
            memory_usage: 0.0,
            active_connections: 0,
        };

        self.nodes.write().await.insert(node_id.clone(), node_health);

        // Set as primary if no primary exists
        if self.current_primary.read().await.is_none() {
            *self.current_primary.write().await = Some(node_id);
        }

        Ok(())
    }

    /// Unregister a node
    pub async fn unregister_node(&self, node_id: &str) -> Result<()> {
        self.nodes.write().await.remove(node_id);

        // If this was the primary, elect a new one
        if let Some(current) = self.current_primary.read().await.as_ref() {
            if current == node_id {
                let nodes = self.nodes.read().await;
                if let Some(backup) = Self::find_healthy_backup(&*nodes).await {
                    *self.current_primary.write().await = Some(backup);
                } else {
                    *self.current_primary.write().await = None;
                }
            }
        }

        Ok(())
    }

    /// Get the current primary node
    pub async fn get_primary_node(&self) -> Option<String> {
        self.current_primary.read().await.clone()
    }

    /// Get a node for load balancing
    pub async fn get_load_balanced_node(&self) -> Option<String> {
        let nodes = self.nodes.read().await;
        let healthy_nodes: Vec<_> = nodes
            .iter()
            .filter(|(_, health)| health.status == NodeStatus::Healthy)
            .collect();

        if healthy_nodes.is_empty() {
            return None;
        }

        self.load_balancer.select_node(&healthy_nodes).await
    }

    /// Force failover to a specific node
    pub async fn force_failover(&self, target_node: String) -> Result<()> {
        let nodes = self.nodes.read().await;

        if let Some(node) = nodes.get(&target_node) {
            if node.status == NodeStatus::Healthy {
                *self.current_primary.write().await = Some(target_node.clone());
                self.stats.write().await.total_failovers += 1;
                self.stats.write().await.manual_failovers += 1;
                tracing::info!("Manual failover to node: {}", target_node);
                return Ok(());
            }
        }

        Err(anyhow::anyhow!(
            "Target node not healthy or not found: {}",
            target_node
        ))
    }

    /// Get failover statistics
    pub async fn get_stats(&self) -> FailoverStats {
        self.stats.read().await.clone()
    }

    /// Update node metrics
    pub async fn update_node_metrics(
        &self,
        node_id: &str,
        response_time_ms: f64,
        cpu_usage: f64,
        memory_usage: f64,
        active_connections: usize,
    ) -> Result<()> {
        if let Some(node) = self.nodes.write().await.get_mut(node_id) {
            node.response_time_ms = response_time_ms;
            node.cpu_usage = cpu_usage;
            node.memory_usage = memory_usage;
            node.active_connections = active_connections;
            node.last_health_check = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs();
        }

        Ok(())
    }

    /// Check if a node is healthy
    async fn check_node_health(node: &NodeHealth) -> Result<()> {
        // Simple health check based on last update time and metrics
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();
        let time_since_last_check = Duration::from_secs(now.saturating_sub(node.last_health_check));

        if time_since_last_check > Duration::from_secs(60) {
            return Err(anyhow::anyhow!("Node health check timeout"));
        }

        if node.cpu_usage > 95.0 || node.memory_usage > 95.0 {
            return Err(anyhow::anyhow!("Node resource usage too high"));
        }

        if node.response_time_ms > 5000.0 {
            return Err(anyhow::anyhow!("Node response time too high"));
        }

        Ok(())
    }

    /// Find a healthy backup node
    async fn find_healthy_backup(nodes: &HashMap<String, NodeHealth>) -> Option<String> {
        nodes
            .iter()
            .filter(|(_, health)| health.status == NodeStatus::Healthy)
            .min_by(|(_, a), (_, b)| {
                a.response_time_ms
                    .partial_cmp(&b.response_time_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| id.clone())
    }
}

/// Failover configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverConfig {
    /// Health check interval
    pub health_check_interval: Duration,

    /// Failover timeout
    pub failover_timeout: Duration,

    /// Maximum retry attempts before marking node as failed
    pub max_retry_attempts: u32,

    /// Load balancer strategy
    pub load_balancer_strategy: LoadBalancerStrategy,

    /// Enable automatic failover
    pub enable_auto_failover: bool,

    /// Enable load balancing
    pub enable_load_balancing: bool,

    /// Minimum healthy nodes required
    pub min_healthy_nodes: usize,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            health_check_interval: Duration::from_secs(30),
            failover_timeout: Duration::from_secs(10),
            max_retry_attempts: 3,
            load_balancer_strategy: LoadBalancerStrategy::RoundRobin,
            enable_auto_failover: true,
            enable_load_balancing: true,
            min_healthy_nodes: 1,
        }
    }
}

/// Load balancer for distributing requests across healthy nodes
#[derive(Debug)]
pub struct LoadBalancer {
    strategy: LoadBalancerStrategy,
    round_robin_counter: Arc<RwLock<usize>>,
}

impl LoadBalancer {
    pub fn new(strategy: LoadBalancerStrategy) -> Self {
        Self {
            strategy,
            round_robin_counter: Arc::new(RwLock::new(0)),
        }
    }

    /// Select a node based on the load balancing strategy
    pub async fn select_node(&self, healthy_nodes: &[(&String, &NodeHealth)]) -> Option<String> {
        if healthy_nodes.is_empty() {
            return None;
        }

        match self.strategy {
            LoadBalancerStrategy::RoundRobin => {
                let mut counter = self.round_robin_counter.write().await;
                let index = *counter % healthy_nodes.len();
                *counter = (*counter + 1) % healthy_nodes.len();
                Some(healthy_nodes[index].0.clone())
            },

            LoadBalancerStrategy::LeastConnections => healthy_nodes
                .iter()
                .min_by_key(|(_, health)| health.active_connections)
                .map(|(id, _)| id.to_string()),

            LoadBalancerStrategy::WeightedResponseTime => healthy_nodes
                .iter()
                .min_by(|(_, a), (_, b)| {
                    a.response_time_ms
                        .partial_cmp(&b.response_time_ms)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(id, _)| id.to_string()),

            LoadBalancerStrategy::ResourceBased => healthy_nodes
                .iter()
                .min_by(|(_, a), (_, b)| {
                    let score_a = (a.cpu_usage + a.memory_usage) / 2.0;
                    let score_b = (b.cpu_usage + b.memory_usage) / 2.0;
                    score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(id, _)| id.to_string()),
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LoadBalancerStrategy {
    RoundRobin,
    LeastConnections,
    WeightedResponseTime,
    ResourceBased,
}

/// Failover strategies (alias for LoadBalancerStrategy for backward compatibility)
pub type FailoverStrategy = LoadBalancerStrategy;

/// Node health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeHealth {
    pub node_id: String,
    pub endpoint: String,
    pub status: NodeStatus,
    pub last_health_check: u64,    // Unix timestamp in seconds
    pub last_failure: Option<u64>, // Unix timestamp in seconds
    pub response_time_ms: f64,
    pub request_count: u64,
    pub error_count: u64,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub active_connections: usize,
}

/// Node status enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum NodeStatus {
    Healthy,
    Unhealthy,
    Recovering,
    Maintenance,
}

/// Failover statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FailoverStats {
    pub total_failovers: u64,
    pub manual_failovers: u64,
    pub automatic_failovers: u64,
    pub avg_failover_time_ms: f64,
    pub current_primary_node: Option<String>,
    pub healthy_nodes_count: usize,
    pub total_nodes_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_failover_manager() {
        let config = FailoverConfig::default();
        let manager = FailoverManager::new(config);

        // Register nodes
        manager
            .register_node("node1".to_string(), "http://node1:8080".to_string())
            .await
            .unwrap();
        manager
            .register_node("node2".to_string(), "http://node2:8080".to_string())
            .await
            .unwrap();

        // Check primary node
        let primary = manager.get_primary_node().await;
        assert!(primary.is_some());

        // Test load balancing
        let balanced_node = manager.get_load_balanced_node().await;
        assert!(balanced_node.is_some());
    }

    #[tokio::test]
    async fn test_load_balancer() {
        let lb = LoadBalancer::new(LoadBalancerStrategy::RoundRobin);

        let node1 = NodeHealth {
            node_id: "node1".to_string(),
            endpoint: "http://node1:8080".to_string(),
            status: NodeStatus::Healthy,
            last_health_check: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
            last_failure: None,
            response_time_ms: 100.0,
            request_count: 0,
            error_count: 0,
            cpu_usage: 50.0,
            memory_usage: 60.0,
            active_connections: 10,
        };

        let node2 = NodeHealth {
            node_id: "node2".to_string(),
            endpoint: "http://node2:8080".to_string(),
            status: NodeStatus::Healthy,
            last_health_check: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
            last_failure: None,
            response_time_ms: 150.0,
            request_count: 0,
            error_count: 0,
            cpu_usage: 30.0,
            memory_usage: 40.0,
            active_connections: 5,
        };

        let node1_id = "node1".to_string();
        let node2_id = "node2".to_string();

        let nodes = vec![(&node1_id, &node1), (&node2_id, &node2)];

        // Test round robin
        let selected1 = lb.select_node(&nodes).await;
        let selected2 = lb.select_node(&nodes).await;

        assert!(selected1.is_some());
        assert!(selected2.is_some());
        assert_ne!(selected1, selected2); // Should alternate
    }

    #[test]
    fn test_node_status() {
        let status = NodeStatus::Healthy;
        assert_eq!(status, NodeStatus::Healthy);
    }
}
