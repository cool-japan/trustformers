// Cache module for inference optimization
pub mod cache_key;
pub mod eviction;
pub mod inference_cache;
pub mod metrics;

pub use cache_key::{CacheKey, CacheKeyBuilder};
pub use eviction::{EvictionPolicy, LRUEviction, SizeBasedEviction, TTLEviction};
pub use inference_cache::{CacheConfig, CacheEntry, InferenceCache};
pub use metrics::CacheMetrics;
