//! # ConnectionEncryptionConfig - Trait Implementations
//!
//! This module contains trait implementations for `ConnectionEncryptionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ConnectionAuthentication, ConnectionEncryptionConfig, TLSConfig};

impl Default for ConnectionEncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tls_config: TLSConfig::default(),
            authentication: ConnectionAuthentication::default(),
            timeout: std::time::Duration::from_secs(30),
        }
    }
}

