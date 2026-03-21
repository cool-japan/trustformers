//! # AuthenticationCredentials - Trait Implementations
//!
//! This module contains trait implementations for `AuthenticationCredentials`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AuthenticationCredentials;

impl Default for AuthenticationCredentials {
    fn default() -> Self {
        Self {
            username: None,
            password: None,
            certificate_path: None,
            token: None,
        }
    }
}

