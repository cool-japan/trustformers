//! Binary delta encoding for incremental model-file updates.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TrustformersError};
#[cfg(feature = "hub")]
use reqwest::Client as AsyncClient;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use trustformers_core::errors::TrustformersError as CoreTrustformersError;

#[cfg(feature = "hub")]
use super::types::HF_HUB_URL;

/// Delta compression info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaInfo {
    pub base_version: String,
    pub target_version: String,
    pub delta_url: String,
    pub compression_ratio: f64,
    pub delta_size: u64,
    pub full_size: u64,
    /// SHA-256 (hex) of the delta file's raw bytes, if the server provided
    /// one. When present, the delta download itself is checksum-verified in
    /// addition to the TFDELTA1 format's own embedded base/target hashes.
    #[serde(default)]
    pub delta_checksum: Option<String>,
}

// ─── Binary delta codec (TFDELTA1) ─────────────────────────────────────────
//
// Real implementation lives in `crate::hub_delta_codec` so this file stays
// under the workspace's 2000-line-per-file limit; these are the public,
// bytes-level entry points. Neither needs the `hub` feature — both are pure
// local computation — so they work even in builds with networking disabled.

/// Reconstruct a target file's bytes from a base file's bytes and a TFDELTA1
/// binary delta (as produced by [`create_binary_delta`]).
///
/// Both the base and the reconstructed target are checksum-verified against
/// the SHA-256 hashes embedded in the delta itself. A delta that doesn't
/// match the supplied base, that is truncated or malformed, or that doesn't
/// replay to the bytes it claims to, is a [`TrustformersError`] — never a
/// silently corrupted result. This is what makes
/// [`DownloadManager::apply_binary_delta`] safe: it is the only way this
/// module ever produces reconstructed bytes.
pub fn reconstruct_from_delta(base_data: &[u8], delta_data: &[u8]) -> Result<Vec<u8>> {
    crate::hub_delta_codec::apply_delta(base_data, delta_data).map_err(|message| {
        TrustformersError::Core(CoreTrustformersError::other(format!(
            "failed to apply binary delta: {message}"
        )))
    })
}

/// Encode a TFDELTA1 binary delta that reconstructs `target_path` from
/// `base_path`.
///
/// Hosting the result at a [`DeltaInfo::delta_url`] lets a future download of
/// `target_path`, starting from a peer that already has `base_path`, transfer
/// only the delta; [`reconstruct_from_delta`] is the inverse operation.
pub fn create_binary_delta(base_path: &Path, target_path: &Path) -> Result<Vec<u8>> {
    let base_data = fs::read(base_path).map_err(|e| TrustformersError::Io {
        message: format!("Failed to read base file: {}", e),
        path: Some(base_path.to_string_lossy().to_string()),
        suggestion: Some("Check file existence and permissions".to_string()),
    })?;
    let target_data = fs::read(target_path).map_err(|e| TrustformersError::Io {
        message: format!("Failed to read target file: {}", e),
        path: Some(target_path.to_string_lossy().to_string()),
        suggestion: Some("Check file existence and permissions".to_string()),
    })?;
    Ok(crate::hub_delta_codec::encode_delta(
        &base_data,
        &target_data,
    ))
}

/// Check if delta compression is available for a model update
#[cfg(feature = "hub")]
pub async fn check_delta_availability(
    model_id: &str,
    from_revision: &str,
    to_revision: &str,
) -> Result<Option<DeltaInfo>> {
    // A real GET against a hypothetical delta-serving endpoint. The public
    // Hugging Face Hub does not currently expose `/api/models/*/deltas/*`, so
    // in practice this will honestly resolve to `Ok(None)` (a 404) rather
    // than fabricating delta availability; a Hub-compatible server that does
    // implement the endpoint would be answered for real, with no code change
    // needed here.
    let delta_url = format!(
        "{}/api/models/{}/deltas/{}/{}",
        HF_HUB_URL, model_id, from_revision, to_revision
    );

    let client = AsyncClient::new();
    let response = client.get(&delta_url).send().await;

    match response {
        Ok(resp) if resp.status().is_success() => {
            let delta_info: DeltaInfo = resp.json().await.map_err(|e| {
                TrustformersError::invalid_input(
                    format!("Failed to parse delta info: {}", e),
                    Some("delta_response"),
                    Some("valid DeltaInfo JSON object"),
                    Some("invalid JSON format"),
                )
            })?;
            Ok(Some(delta_info))
        },
        _ => Ok(None), // Delta not available
    }
}
