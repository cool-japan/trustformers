//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::ResourceRequirement;

impl Default for ResourceRequirement {
    fn default() -> Self {
        Self {
            resource_type: "mixed".to_string(),
            min_amount: 1.0,
            cpu_cores: 1.0,
            memory_mb: 512,
            gpu_devices: vec![],
            network_ports: 0,
            temp_directories: 0,
            database_connections: 0,
            custom_resources: HashMap::new(),
        }
    }
}
