//! # TLSConfig - Trait Implementations
//!
//! This module contains trait implementations for `TLSConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{TLSConfig, TLSVersion};

impl Default for TLSConfig {
    fn default() -> Self {
        Self {
            version: TLSVersion::TLS13,
            cipher_suites: vec![
                "TLS_AES_256_GCM_SHA384".to_string(), "TLS_CHACHA20_POLY1305_SHA256"
                .to_string(),
            ],
            certificate_validation: true,
            client_certificate: None,
        }
    }
}

