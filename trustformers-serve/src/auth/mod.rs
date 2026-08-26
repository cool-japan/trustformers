//! Authentication and authorization.
//!
//! [`types`] holds the live stack: JWT issuing and verification through the
//! `jsonwebtoken` crate, API keys digested with SHA-256, and the axum
//! middleware the server actually installs. [`functions`] holds the free
//! functions those handlers call.
//!
//! # Removed in 0.2.1: the orphan authentication cluster
//!
//! `oidc.rs` (987 lines) and `middleware.rs` (554 lines) were deleted rather
//! than mounted. Neither was ever declared as a module, so neither had ever
//! been compiled, and neither was reachable from any public construction path.
//! Both re-implemented — badly — what [`types`] already does correctly:
//!
//! * `oidc.rs` advertised `HS256`, `RS256` and `ES256` JWT validation on top of
//!   a hand-rolled SHA-256/HMAC implementation. Its `JwtValidator::validate`
//!   verified the signature only in the `HS256` arm; for `RS256` and `ES256` it
//!   skipped signature verification entirely and still returned
//!   `AuthResult::Authenticated`. That is an authentication bypass: any forged
//!   RS256 token would have been accepted. The live validator in [`types`]
//!   delegates to `jsonwebtoken`, which verifies every algorithm it accepts.
//! * `middleware.rs` stored API keys as an FNV-1a digest — a non-cryptographic
//!   hash, trivially invertible over a realistic key space — while documenting
//!   it as "never stored in plain text". It also defined a second, structurally
//!   different `ApiKeyInfo`/`AuthMiddleware` pair that would have collided with
//!   the `pub use types::*` glob below. The live path hashes keys with SHA-256
//!   (see [`functions`]).
//!
//! Shipping either file dormant in a published crate was the worst of the
//! options available: dead code that reads like a supported security feature,
//! that no test could exercise, and that a reader cannot distinguish from the
//! implementation that is actually wired in.

pub mod apikeyinfo_traits;
pub mod authconfig_traits;
pub mod functions;
pub mod types;
#[cfg(test)]
mod types_tests;

// Re-export all types
pub use functions::*;
pub use types::*;
