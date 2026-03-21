//! # ApiKeyInfo - Trait Implementations
//!
//! This module contains trait implementations for `ApiKeyInfo`.
//!
//! ## Implemented Traits
//!
//! - `From`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ApiKey, ApiKeyInfo};

impl From<ApiKey> for ApiKeyInfo {
    fn from(api_key: ApiKey) -> Self {
        Self {
            id: api_key.id,
            name: api_key.name,
            scopes: api_key.scopes,
            created_at: api_key.created_at,
            expires_at: api_key.expires_at,
            last_used_at: api_key.last_used_at,
            is_active: api_key.is_active,
            usage_count: api_key.usage_count,
            ip_whitelist: api_key.ip_whitelist,
            allowed_endpoints: api_key.allowed_endpoints,
        }
    }
}
