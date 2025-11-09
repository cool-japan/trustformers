//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
/// Convenience functions for distributed debugging
/// Create a distributed debugger for a master node
pub fn create_master_debugger(rank: u32, hostname: String) -> DistributedDebugger {
    let node_id = NodeId::new(rank, hostname);
    DistributedDebugger::new(DistributedDebugConfig::default(), node_id)
}
/// Create a distributed debugger for a worker node
pub fn create_worker_debugger(rank: u32, hostname: String) -> DistributedDebugger {
    let node_id = NodeId::new(rank, hostname);
    let mut config = DistributedDebugConfig::default();
    config.enable_auto_recovery = false;
    DistributedDebugger::new(config, node_id)
}
/// Macro for monitoring gradient synchronization
#[macro_export]
macro_rules! monitor_gradient_sync {
    ($debugger:expr, $sync_round:expr, $nodes:expr, $sync_time:expr) => {{
        let sync_event = GradientSyncEvent {
            timestamp: std::time::SystemTime::now(),
            sync_round: $sync_round,
            participating_nodes: $nodes,
            total_sync_time: $sync_time,
            gradient_sizes: HashMap::new(),
            compression_ratio: 1.0,
            sync_algorithm: SyncAlgorithm::AllReduce,
        };
        $debugger.monitor_gradient_sync(sync_event).await
    }};
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::time::Duration;

    #[tokio::test]
    async fn test_distributed_debugger_creation() {
        let node_id = NodeId::new(0, "test-node".to_string());
        let debugger = DistributedDebugger::new(DistributedDebugConfig::default(), node_id);
        assert_eq!(debugger.node_id.rank, 0);
    }
    #[tokio::test]
    async fn test_node_info_creation() {
        let node_id = NodeId::new(1, "worker-1".to_string());
        let debugger = DistributedDebugger::new(DistributedDebugConfig::default(), node_id);
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let node_info = debugger.create_node_info(addr).await.unwrap();
        assert_eq!(node_info.node_id.rank, 1);
        assert_eq!(node_info.status, NodeStatus::Healthy);
        assert_eq!(node_info.address, addr);
    }
    #[tokio::test]
    async fn test_gradient_sync_monitoring() {
        let node_id = NodeId::new(0, "test-node".to_string());
        let debugger = DistributedDebugger::new(DistributedDebugConfig::default(), node_id.clone());
        let sync_event = GradientSyncEvent {
            timestamp: std::time::SystemTime::now(),
            sync_round: 1,
            participating_nodes: vec![node_id],
            total_sync_time: Duration::from_millis(100),
            gradient_sizes: HashMap::new(),
            compression_ratio: 0.8,
            sync_algorithm: SyncAlgorithm::AllReduce,
        };
        let result = debugger.monitor_gradient_sync(sync_event).await;
        assert!(result.is_ok());
    }
    #[tokio::test]
    async fn test_cluster_analysis() {
        let node_id = NodeId::new(0, "test-node".to_string());
        let debugger = DistributedDebugger::new(DistributedDebugConfig::default(), node_id);
        let _report = debugger.analyze_cluster_performance().await.unwrap();
        // Successfully generated cluster analysis report
    }
    #[tokio::test]
    async fn test_fault_detection() {
        let node_id = NodeId::new(0, "test-node".to_string());
        let debugger = DistributedDebugger::new(DistributedDebugConfig::default(), node_id);
        let _faults = debugger.detect_faults().await.unwrap();
        // Successfully detected faults
    }
    #[tokio::test]
    async fn test_distributed_debug_report() {
        let node_id = NodeId::new(0, "test-node".to_string());
        let debugger = DistributedDebugger::new(DistributedDebugConfig::default(), node_id);
        let report = debugger.generate_distributed_debug_report().await.unwrap();
        assert!(!report.recommendations.is_empty());
    }
    #[test]
    fn test_convenience_functions() {
        let master = create_master_debugger(0, "master".to_string());
        let worker = create_worker_debugger(1, "worker-1".to_string());
        assert_eq!(master.node_id.rank, 0);
        assert_eq!(worker.node_id.rank, 1);
        assert!(!worker.config.enable_auto_recovery);
    }
}
