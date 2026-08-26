// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Model versioning UI for TrustformeRS Hub integration
//!
//! This module provides a web-based user interface for managing model versions,
//! tracking changes, comparing models, and visualizing version history.
//!
//! Split into cohesive submodules: [`types`] (version/repository/access-control data
//! model), [`repository`] (the `ModelRepository` domain type), [`state`] (server-wide
//! UI state and configuration), [`server`] (the `HubUiServer` axum router and start
//! functions), [`handlers`] (axum route handlers) and [`templates`] (HTML/CSS view
//! generation).

pub mod handlers;
pub mod repository;
pub mod server;
pub mod state;
pub mod templates;
#[cfg(test)]
mod tests;
pub mod types;

// Re-export all types
pub use repository::*;
pub use server::*;
pub use state::*;
pub use types::*;
// `handlers` and `templates` hold functions that were module-private in the
// original single-file `hub_ui.rs` (only ever called from within this same
// module, e.g. wired into the axum `Router` in `server.rs`); splitting them
// out required upgrading them to `pub(super)` for cross-submodule access, but
// they are still not part of this crate's public API. The `tests` submodule
// reaches them unqualified via its own `use super::*;`, exactly as it could
// when they all lived in one file, so the re-export is only needed there.
#[cfg(test)]
use handlers::*;
#[cfg(test)]
use templates::*;
