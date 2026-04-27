//! Cache module for TrustformeRS.
//!
//! Provides versioned, TTL-aware caching for model weights and inference results.

pub mod versioned_cache;

pub use versioned_cache::{
    CacheError, CacheEvictionPolicy, VersionedCache, VersionedCacheConfig, VersionedCacheStats,
};
