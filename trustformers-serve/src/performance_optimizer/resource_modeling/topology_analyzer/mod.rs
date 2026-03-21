//! Hardware Topology Analysis Module for Performance Optimizer
//!
//! This module provides comprehensive hardware topology analysis capabilities for optimal
//! performance configuration and resource allocation strategies. It includes detection and
//! analysis of NUMA topology, cache hierarchy, memory subsystem layout, I/O infrastructure,
//! and system interconnects to enable intelligent optimization decisions.
//!
//! # Features
//!
//! * **NUMA Topology Detection**: Complete NUMA domain discovery and cross-domain optimization
//! * **Cache Hierarchy Analysis**: Multi-level cache analysis with optimization recommendations
//! * **Memory Topology Mapping**: Memory subsystem topology for optimal allocation strategies
//! * **I/O Topology Analysis**: PCIe layout optimization and device placement strategies
//! * **Interconnect Analysis**: Bandwidth analysis and bottleneck identification
//! * **Topology Optimization**: Comprehensive optimization recommendations based on analysis
//! * **Affinity Management**: CPU and memory affinity optimization for performance
//! * **Bandwidth Analysis**: System-wide interconnect bandwidth monitoring and analysis
//! * **Topology Validation**: Verification and validation of topology detection results
//! * **Advanced Hardware Support**: Vendor-specific optimizations and enterprise features

pub mod analyzer;
pub mod cache;
pub mod config;
pub mod interconnect;
pub mod io_topology;
pub mod memory;
pub mod numa;
pub mod optimization;
pub mod types;
pub mod validation;
pub mod vendor_specific;

// Re-export public API
pub use analyzer::*;
pub use cache::*;
pub use config::*;
pub use interconnect::*;
pub use io_topology::*;
pub use memory::*;
pub use numa::*;
pub use optimization::*;
pub use types::*;
pub use validation::*;
pub use vendor_specific::*;
