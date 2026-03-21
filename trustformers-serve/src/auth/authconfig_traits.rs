//! # AuthConfig - Trait Implementations
//!
//! This module contains trait implementations for `AuthConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use jsonwebtoken::Algorithm;
use std::collections::HashMap;

use super::types::AuthConfig;

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            secret_key: "your-secret-key".to_string(),
            issuer: "trustformers-serve".to_string(),
            audience: "trustformers-api".to_string(),
            algorithm: Algorithm::HS256,
            leeway: 60,
            oauth2_providers: HashMap::new(),
            oauth2_state_max_age: 600,
        }
    }
}
