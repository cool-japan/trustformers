//! Offline Model Pack System for TrustformeRS
//!
//! Enables packaging and distribution of model collections for offline deployment.
//!
//! Split into cohesive submodules: [`resolution`] (resolving a Hub model id to a
//! real on-disk directory), [`pack_types`] (pack metadata/config types),
//! [`manager`] (the `OfflineModelPackManager` engine) and [`hub_integration`]
//! (bridging packs with the online Hub client).

pub mod hub_integration;
pub mod manager;
pub mod pack_types;
pub mod resolution;
#[cfg(test)]
mod tests;

// Re-export all types
pub use hub_integration::*;
pub use manager::*;
pub use pack_types::*;
pub use resolution::*;
