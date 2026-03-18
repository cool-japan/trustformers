//! # ConnectionAuthentication - Trait Implementations
//!
//! This module contains trait implementations for `ConnectionAuthentication`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{AuthenticationCredentials, AuthenticationMethod, ConnectionAuthentication};

impl Default for ConnectionAuthentication {
    fn default() -> Self {
        Self {
            method: AuthenticationMethod::UsernamePassword,
            credentials: AuthenticationCredentials::default(),
        }
    }
}

