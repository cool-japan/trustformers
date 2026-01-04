//! Distributed Training Profiling Demo
//!
//! This example demonstrates how to use the distributed profiling system
//! to analyze and optimize distributed training performance.

use anyhow::Result;
use std::time::Duration;
use trustformers_debug::distributed_profiling::{
    CommunicationEvent, CommunicationType, DistributedProfiler, DistributedProfilerConfig,
    NodeInfo, NodePerformanceSnapshot, NodeRole, NodeStatus, SyncType, SynchronizationEvent,
};

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("=== Distributed Training Profiling Demo ===\n");

    // Demo 1: Basic profiling setup
    demo_basic_profiling()?;

    // Demo 2: Communication analysis
    demo_communication_analysis()?;

    // Demo 3: Load balance analysis
    demo_load_balance_analysis()?;

    // Demo 4: Bottleneck detection
    demo_bottleneck_detection()?;

    println!("\n=== Demo Complete ===");

    Ok(())
}

fn demo_basic_profiling() -> Result<()> {
    println!("--- Demo 1: Basic Distributed Profiling ---\n");

    let config = DistributedProfilerConfig::default();
    let profiler = DistributedProfiler::new(config);

    println!("✓ Created distributed profiler");

    // Register 4 nodes in the cluster
    for i in 0..4 {
        let node = NodeInfo {
            node_id: format!("node-{}", i),
            rank: i,
            world_size: 4,
            host: format!("worker-{}.cluster.local", i),
            gpu_count: 8,
            role: if i == 0 { NodeRole::Master } else { NodeRole::Worker },
            status: NodeStatus::Active,
        };

        profiler.register_node(node)?;
    }

    println!("✓ Registered 4 nodes (1 master + 3 workers)");

    // Get real-time stats
    let stats = profiler.get_realtime_stats()?;
    println!("\n📊 Real-time Statistics:");
    println!(
        "  Active Nodes: {}/{}",
        stats.active_nodes, stats.total_nodes
    );
    println!("  Elapsed Time: {:.2}s", stats.elapsed_time_secs);
    println!();

    Ok(())
}

fn demo_communication_analysis() -> Result<()> {
    println!("--- Demo 2: Communication Analysis ---\n");

    let config = DistributedProfilerConfig::default();
    let profiler = DistributedProfiler::new(config);

    // Register nodes
    for i in 0..4 {
        let node = NodeInfo {
            node_id: format!("node-{}", i),
            rank: i,
            world_size: 4,
            host: format!("worker-{}.cluster.local", i),
            gpu_count: 8,
            role: if i == 0 { NodeRole::Master } else { NodeRole::Worker },
            status: NodeStatus::Active,
        };
        profiler.register_node(node)?;
    }

    println!("Simulating distributed training communications...\n");

    // Simulate all-reduce operations (common in data-parallel training)
    for step in 0..20 {
        for src_node in 0..4 {
            for dest_node in 0..4 {
                if src_node != dest_node {
                    let event = CommunicationEvent {
                        event_id: step * 16 + src_node * 4 + dest_node,
                        timestamp: Duration::from_millis((step * 100) as u64),
                        source_node: format!("node-{}", src_node),
                        dest_node: format!("node-{}", dest_node),
                        comm_type: CommunicationType::AllReduce,
                        data_size_bytes: 1024 * 1024 * 100, // 100 MB gradients
                        duration_ms: 15.0 + (step as f64 * 0.5), // Increasing latency
                        bandwidth_mbps: 1000.0 - (step as f64 * 10.0), // Decreasing bandwidth
                    };

                    profiler.record_communication(event)?;
                }
            }
        }
    }

    println!("✓ Recorded 240 communication events (20 steps × 12 node pairs)");

    // Simulate gradient synchronization
    for step in 0..20 {
        let sync_event = SynchronizationEvent {
            event_id: step,
            timestamp: Duration::from_millis((step * 100) as u64),
            nodes: vec![
                "node-0".to_string(),
                "node-1".to_string(),
                "node-2".to_string(),
                "node-3".to_string(),
            ],
            sync_type: SyncType::DataParallel,
            gradient_size_bytes: 1024 * 1024 * 400, // 400 MB total gradients
            duration_ms: 50.0 + (step as f64 * 2.0), // Increasing sync time
            success: true,
            error: None,
        };

        profiler.record_synchronization(sync_event)?;
    }

    println!("✓ Recorded 20 gradient synchronization events");

    // Generate report
    let report = profiler.generate_report()?;

    println!("\n📊 Communication Analysis:");
    println!(
        "  Total Communications: {}",
        report.communication_summary.total_events
    );
    println!(
        "  Total Data Transferred: {:.2} GB",
        report.communication_summary.total_data_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "  Average Bandwidth: {:.1} MB/s",
        report.communication_summary.avg_bandwidth_mbps
    );
    println!(
        "  Peak Bandwidth: {:.1} MB/s",
        report.communication_summary.peak_bandwidth_mbps
    );
    println!(
        "  Communication Overhead: {:.1}%",
        report.communication_summary.overhead_pct
    );

    println!("\n📊 Synchronization Analysis:");
    println!(
        "  Total Synchronizations: {}",
        report.synchronization_summary.total_syncs
    );
    println!(
        "  Successful: {}",
        report.synchronization_summary.successful_syncs
    );
    println!(
        "  Average Duration: {:.1} ms",
        report.synchronization_summary.avg_sync_duration_ms
    );
    println!(
        "  Max Duration: {:.1} ms",
        report.synchronization_summary.max_sync_duration_ms
    );
    println!(
        "  Sync Efficiency: {:.1}%",
        report.synchronization_summary.sync_efficiency * 100.0
    );

    println!();

    Ok(())
}

fn demo_load_balance_analysis() -> Result<()> {
    println!("--- Demo 3: Load Balance Analysis ---\n");

    let config = DistributedProfilerConfig {
        enable_load_balance_profiling: true,
        ..Default::default()
    };
    let profiler = DistributedProfiler::new(config);

    // Register nodes
    for i in 0..4 {
        let node = NodeInfo {
            node_id: format!("node-{}", i),
            rank: i,
            world_size: 4,
            host: format!("worker-{}.cluster.local", i),
            gpu_count: 8,
            role: if i == 0 { NodeRole::Master } else { NodeRole::Worker },
            status: NodeStatus::Active,
        };
        profiler.register_node(node)?;
    }

    println!("Simulating unbalanced workload distribution...\n");

    // Simulate performance snapshots with load imbalance
    // Node 2 is a straggler with lower performance
    for step in 0..50 {
        for node_id in 0..4 {
            let (compute_util, throughput) = if node_id == 2 {
                // Node 2 is struggling
                (45.0 + (step as f64 * 0.2), 50.0)
            } else {
                // Other nodes are healthy
                (85.0 + (step as f64 * 0.1), 100.0)
            };

            let snapshot = NodePerformanceSnapshot {
                timestamp: Duration::from_millis((step * 100) as u64),
                node_id: format!("node-{}", node_id),
                compute_utilization_pct: compute_util,
                memory_utilization_pct: 70.0,
                network_utilization_pct: 60.0,
                throughput,
                active_communications: if node_id == 2 { 8 } else { 3 },
                pending_operations: if node_id == 2 { 10 } else { 2 },
            };

            profiler.record_snapshot(snapshot)?;
        }
    }

    println!("✓ Recorded 200 performance snapshots (50 steps × 4 nodes)");

    // Generate report
    let report = profiler.generate_report()?;

    println!("\n📊 Load Balance Analysis:");
    println!(
        "  Imbalance Score: {:.3} (lower is better)",
        report.load_balance.imbalance_score
    );
    println!("\n  Compute Utilization by Node:");
    for (node_id, util) in &report.load_balance.compute_utilization {
        println!("    {}: {:.1}%", node_id, util);
    }

    println!("\n  Throughput by Node:");
    for (node_id, tput) in &report.load_balance.throughput {
        println!("    {}: {:.1} samples/sec", node_id, tput);
    }

    if !report.load_balance.stragglers.is_empty() {
        println!("\n  ⚠️  Straggler Nodes Detected:");
        for straggler in &report.load_balance.stragglers {
            println!("    - {}", straggler);
            if let Some(idle) = report.load_balance.idle_time.get(straggler) {
                println!("      Idle Time: {:.1}s", idle);
            }
        }
    }

    println!();

    Ok(())
}

fn demo_bottleneck_detection() -> Result<()> {
    println!("--- Demo 4: Bottleneck Detection & Recommendations ---\n");

    let config = DistributedProfilerConfig {
        enable_bottleneck_detection: true,
        bottleneck_threshold_pct: 60.0, // Lower threshold for demo
        ..Default::default()
    };
    let profiler = DistributedProfiler::new(config);

    // Register nodes
    for i in 0..8 {
        // 8-node cluster
        let node = NodeInfo {
            node_id: format!("node-{}", i),
            rank: i,
            world_size: 8,
            host: format!("worker-{}.cluster.local", i),
            gpu_count: 8,
            role: if i == 0 { NodeRole::Master } else { NodeRole::Worker },
            status: NodeStatus::Active,
        };
        profiler.register_node(node)?;
    }

    println!("Simulating training with multiple bottlenecks...\n");

    // Simulate high communication overhead
    for step in 0..30 {
        for src_node in 0..8 {
            for dest_node in 0..8 {
                if src_node != dest_node {
                    let event = CommunicationEvent {
                        event_id: step * 64 + src_node * 8 + dest_node,
                        timestamp: Duration::from_millis((step * 200) as u64),
                        source_node: format!("node-{}", src_node),
                        dest_node: format!("node-{}", dest_node),
                        comm_type: CommunicationType::AllReduce,
                        data_size_bytes: 1024 * 1024 * 200, // Large gradients
                        duration_ms: 100.0 + (step as f64 * 3.0), // High latency!
                        bandwidth_mbps: 500.0,              // Low bandwidth
                    };

                    profiler.record_communication(event)?;
                }
            }
        }
    }

    // Simulate inefficient synchronization
    for step in 0..30 {
        let sync_event = SynchronizationEvent {
            event_id: step,
            timestamp: Duration::from_millis((step * 200) as u64),
            nodes: (0..8).map(|i| format!("node-{}", i)).collect(),
            sync_type: SyncType::DataParallel,
            gradient_size_bytes: 1024 * 1024 * 1600, // 1.6 GB gradients
            duration_ms: 200.0 + (step as f64 * 5.0), // Very slow!
            success: true,
            error: None,
        };

        profiler.record_synchronization(sync_event)?;
    }

    // Simulate load imbalance
    for step in 0..50 {
        for node_id in 0..8 {
            let (compute_util, throughput) = if node_id >= 6 {
                // Nodes 6 and 7 are stragglers
                (40.0, 30.0)
            } else {
                (90.0, 100.0)
            };

            let snapshot = NodePerformanceSnapshot {
                timestamp: Duration::from_millis((step * 100) as u64),
                node_id: format!("node-{}", node_id),
                compute_utilization_pct: compute_util,
                memory_utilization_pct: 75.0,
                network_utilization_pct: 80.0,
                throughput,
                active_communications: 5,
                pending_operations: if node_id >= 6 { 20 } else { 3 },
            };

            profiler.record_snapshot(snapshot)?;
        }
    }

    println!("✓ Simulated 30 training steps with bottlenecks");

    // Generate comprehensive report
    let report = profiler.generate_report()?;

    println!("\n📊 Overall Statistics:");
    println!("  Cluster Size: {} nodes", report.num_nodes);
    println!("  Profiling Duration: {:.1}s", report.total_duration_secs);
    println!(
        "  Communication Overhead: {:.1}%",
        report.communication_summary.overhead_pct
    );
    println!(
        "  Sync Efficiency: {:.1}%",
        report.synchronization_summary.sync_efficiency * 100.0
    );
    println!(
        "  Load Imbalance: {:.2}",
        report.load_balance.imbalance_score
    );

    println!("\n🔍 Detected Bottlenecks ({}):", report.bottlenecks.len());
    for (i, bottleneck) in report.bottlenecks.iter().enumerate() {
        println!(
            "\n  {}. {:?} (Severity: {:.0}/100)",
            i + 1,
            bottleneck.bottleneck_type,
            bottleneck.severity
        );
        println!("     Description: {}", bottleneck.description);
        println!("     Affected Nodes: {:?}", bottleneck.affected_nodes);
        println!("     💡 Suggestion: {}", bottleneck.suggestion);
    }

    println!(
        "\n💡 Optimization Recommendations ({}):",
        report.recommendations.len()
    );
    for (i, rec) in report.recommendations.iter().enumerate() {
        println!("  {}. {}", i + 1, rec);
    }

    // Export report to JSON
    let json_path = std::env::temp_dir().join("distributed_profiling_report.json");
    profiler.export_json(&json_path)?;
    println!("\n✓ Exported detailed report to: {}", json_path.display());

    println!();

    Ok(())
}
