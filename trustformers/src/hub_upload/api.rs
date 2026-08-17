//! The real HuggingFace Hub upload wire protocol.
//!
//! Three real HTTP exchanges, matching the actual Hub API (cross-checked
//! against the `hf-hub` crate's implementation of the same endpoints):
//!
//! * **Repo existence**: `GET {base_url}/api/{repo_type}s/{repo_id}`.
//! * **Repo creation**: `POST {base_url}/api/repos/create` with a JSON body
//!   `{"name", "type", "private", "organization"?}`.
//! * **Commit** (upload/delete): `POST {base_url}/api/{repo_type}s/{repo_id}/commit/{revision}`
//!   with an `application/x-ndjson` body — one `{"key":"header",...}` line,
//!   then one `{"key":"file",...}` (base64 content) or `{"key":"deletedFile",...}`
//!   line per operation.
//!
//! Large files that would need real Git-LFS object storage (a separate
//! preupload + batch-upload exchange this module does not implement) are
//! refused with [`HubError::LfsRequired`] *before* any request is sent,
//! rather than silently included as an oversized base64 blob or a dangling
//! `lfsFile` pointer to bytes that were never actually uploaded anywhere —
//! either of those would be exactly the kind of silent-fabrication bug this
//! module replaces.
//!
//! Every function here runs its own single-threaded Tokio runtime via
//! [`run_blocking`], mirroring `hub.rs`'s `download_file` sync wrapper, so
//! the public `HubUploader` API stays fully synchronous.

use super::{HubError, RepoType};
use serde::Deserialize;
use std::time::Duration;

/// Files at or above this size need real Git-LFS object storage, which this
/// module does not implement (see the module doc comment).
pub(super) const LFS_INLINE_THRESHOLD_BYTES: u64 = 10 * 1024 * 1024;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Run `future` to completion on a fresh, single-threaded Tokio runtime.
///
/// Must only be called from a thread with no *already-running* Tokio runtime
/// (i.e. from ordinary synchronous code, not from inside `#[tokio::test]` or
/// another `.block_on` call) — like `hub.rs`'s synchronous `download_file`,
/// which uses the exact same pattern.
pub(super) fn run_blocking<T>(
    future: impl std::future::Future<Output = std::result::Result<T, HubError>>,
) -> std::result::Result<T, HubError> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| HubError::Network {
            message: format!("Failed to start async runtime: {e}"),
        })?;
    rt.block_on(future)
}

#[cfg(feature = "hub")]
fn client() -> std::result::Result<reqwest::Client, HubError> {
    reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| HubError::Network {
            message: format!("Failed to build HTTP client: {e}"),
        })
}

/// `GET {base_url}/api/{repo_type}s/{repo_id}` — `true` on HTTP 200, `false`
/// on HTTP 404, an error for anything else (including transport failures).
#[cfg(feature = "hub")]
pub(super) async fn repo_exists(
    base_url: &str,
    repo_type: RepoType,
    repo_id: &str,
    token: &str,
) -> std::result::Result<bool, HubError> {
    let url = format!("{base_url}/api/{}s/{repo_id}", repo_type.as_str());
    let response =
        client()?
            .get(&url)
            .bearer_auth(token)
            .send()
            .await
            .map_err(|e| HubError::Network {
                message: format!("Failed to check repository existence: {e}"),
            })?;

    match response.status().as_u16() {
        200 => Ok(true),
        404 => Ok(false),
        401 | 403 => Err(HubError::Unauthorized {
            message: format!(
                "Hub rejected the token while checking '{repo_id}': HTTP {}",
                response.status()
            ),
        }),
        status => Err(HubError::RequestFailed {
            status_code: status,
            message: format!("Unexpected response checking repository existence for '{repo_id}'"),
        }),
    }
}

#[cfg(not(feature = "hub"))]
pub(super) async fn repo_exists(
    _base_url: &str,
    _repo_type: RepoType,
    _repo_id: &str,
    _token: &str,
) -> std::result::Result<bool, HubError> {
    Err(feature_unavailable())
}

#[derive(Debug, Deserialize)]
struct RepoUrlResponse {
    url: Option<String>,
}

/// `POST {base_url}/api/repos/create` with body
/// `{"name", "type", "private", "organization"?}`, splitting `repo_id` into
/// an optional `organization` namespace and a bare `name` — matching how the
/// real Hub API distinguishes the two.
#[cfg(feature = "hub")]
pub(super) async fn create_repo(
    base_url: &str,
    repo_type: RepoType,
    repo_id: &str,
    private: bool,
    token: &str,
) -> std::result::Result<String, HubError> {
    let url = format!("{base_url}/api/repos/create");
    let (organization, name) = split_repo_id(repo_id);

    let mut body = serde_json::json!({
        "name": name,
        "type": repo_type.as_str(),
        "private": private,
    });
    if let Some(org) = organization {
        body["organization"] = serde_json::Value::String(org.to_string());
    }

    let response =
        client()?.post(&url).bearer_auth(token).json(&body).send().await.map_err(|e| {
            HubError::Network {
                message: format!("Failed to create repository '{repo_id}': {e}"),
            }
        })?;

    let status = response.status();
    if status.as_u16() == 401 || status.as_u16() == 403 {
        return Err(HubError::Unauthorized {
            message: format!("Hub rejected the token while creating '{repo_id}': HTTP {status}"),
        });
    }
    if !status.is_success() {
        let body_text = response.text().await.unwrap_or_default();
        return Err(HubError::RequestFailed {
            status_code: status.as_u16(),
            message: format!("Failed to create repository '{repo_id}': {body_text}"),
        });
    }

    let parsed: RepoUrlResponse = response.json().await.map_err(|e| HubError::RequestFailed {
        status_code: status.as_u16(),
        message: format!("Failed to parse repo-create response: {e}"),
    })?;

    Ok(parsed.url.unwrap_or_else(|| format!("{base_url}/{repo_id}")))
}

#[cfg(not(feature = "hub"))]
pub(super) async fn create_repo(
    _base_url: &str,
    _repo_type: RepoType,
    _repo_id: &str,
    _private: bool,
    _token: &str,
) -> std::result::Result<String, HubError> {
    Err(feature_unavailable())
}

/// One file to include in a commit, already read into memory.
pub(super) struct CommitFile {
    pub repo_path: String,
    pub content: Vec<u8>,
}

/// Result of a successful commit.
pub(super) struct CommitOutcome {
    pub commit_url: Option<String>,
    pub commit_oid: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct CommitResponse {
    #[serde(rename = "commitUrl")]
    commit_url: Option<String>,
    #[serde(rename = "commitOid")]
    commit_oid: Option<String>,
}

/// `POST {base_url}/api/{repo_type}s/{repo_id}/commit/{revision}` with an
/// `application/x-ndjson` body: one `header` line, then one `file` (base64)
/// or `deletedFile` line per operation.
///
/// Every `file` in `files` must already be under [`LFS_INLINE_THRESHOLD_BYTES`]
/// — callers check that before reaching this function, since it's a decision
/// that shouldn't cost a network round-trip to discover.
#[cfg(feature = "hub")]
pub(super) async fn commit(
    base_url: &str,
    repo_type: RepoType,
    repo_id: &str,
    revision: &str,
    commit_message: &str,
    files: &[CommitFile],
    deletions: &[String],
    token: &str,
) -> std::result::Result<CommitOutcome, HubError> {
    use base64::Engine as _;

    let url = format!(
        "{base_url}/api/{}s/{repo_id}/commit/{}",
        repo_type.as_str(),
        urlencode_ref(revision)
    );

    let mut body = Vec::new();
    let header = serde_json::json!({
        "key": "header",
        "value": { "summary": commit_message, "description": "" }
    });
    serde_json::to_writer(&mut body, &header).map_err(|e| HubError::InvalidInput {
        message: format!("Failed to encode commit header: {e}"),
    })?;
    body.push(b'\n');

    for file in files {
        let encoded = base64::engine::general_purpose::STANDARD.encode(&file.content);
        let line = serde_json::json!({
            "key": "file",
            "value": { "content": encoded, "path": file.repo_path, "encoding": "base64" }
        });
        serde_json::to_writer(&mut body, &line).map_err(|e| HubError::InvalidInput {
            message: format!(
                "Failed to encode commit file entry for '{}': {e}",
                file.repo_path
            ),
        })?;
        body.push(b'\n');
    }

    for path in deletions {
        let line = serde_json::json!({
            "key": "deletedFile",
            "value": { "path": path }
        });
        serde_json::to_writer(&mut body, &line).map_err(|e| HubError::InvalidInput {
            message: format!("Failed to encode commit deletion entry for '{path}': {e}"),
        })?;
        body.push(b'\n');
    }

    let response = client()?
        .post(&url)
        .bearer_auth(token)
        .header(reqwest::header::CONTENT_TYPE, "application/x-ndjson")
        .body(body)
        .send()
        .await
        .map_err(|e| HubError::Network {
            message: format!("Failed to commit to '{repo_id}': {e}"),
        })?;

    let status = response.status();
    if status.as_u16() == 401 || status.as_u16() == 403 {
        return Err(HubError::Unauthorized {
            message: format!(
                "Hub rejected the token while committing to '{repo_id}': HTTP {status}"
            ),
        });
    }
    if !status.is_success() {
        let body_text = response.text().await.unwrap_or_default();
        return Err(HubError::RequestFailed {
            status_code: status.as_u16(),
            message: format!("Commit to '{repo_id}' failed: {body_text}"),
        });
    }

    let parsed: CommitResponse = response.json().await.unwrap_or_default();
    Ok(CommitOutcome {
        commit_url: parsed.commit_url,
        commit_oid: parsed.commit_oid,
    })
}

#[cfg(not(feature = "hub"))]
pub(super) async fn commit(
    _base_url: &str,
    _repo_type: RepoType,
    _repo_id: &str,
    _revision: &str,
    _commit_message: &str,
    _files: &[CommitFile],
    _deletions: &[String],
    _token: &str,
) -> std::result::Result<CommitOutcome, HubError> {
    Err(feature_unavailable())
}

fn feature_unavailable() -> HubError {
    HubError::FeatureUnavailable {
        message: "Hub networking is disabled: rebuild with the `hub` feature (e.g. `--features \
                   hub`) to upload to the Hugging Face Hub"
            .to_string(),
    }
}

/// Split `"owner/name"` into `(Some("owner"), "name")`, or `"name"` into
/// `(None, "name")` — matching how the real Hub `/api/repos/create` endpoint
/// distinguishes an explicit namespace from "the token owner's namespace".
fn split_repo_id(repo_id: &str) -> (Option<&str>, &str) {
    match repo_id.split_once('/') {
        Some((namespace, name)) => (Some(namespace), name),
        None => (None, repo_id),
    }
}

/// Minimal RFC 3986 percent-encoding for a URL *path segment* (a git
/// ref/revision name). Deliberately not `url::form_urlencoded`, which encodes
/// space as `+` — correct for a query string or form body, wrong in a path,
/// where a literal `+` would not be decoded back to space by the server.
fn urlencode_ref(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            },
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_repo_id_with_namespace() {
        assert_eq!(
            split_repo_id("owner/model-name"),
            (Some("owner"), "model-name")
        );
    }

    #[test]
    fn test_split_repo_id_without_namespace() {
        assert_eq!(split_repo_id("model-name"), (None, "model-name"));
    }

    #[test]
    fn test_urlencode_ref_plain() {
        assert_eq!(urlencode_ref("main"), "main");
    }

    #[test]
    fn test_urlencode_ref_special_chars() {
        assert_eq!(urlencode_ref("feature branch"), "feature%20branch");
        assert_eq!(urlencode_ref("a/b"), "a%2Fb");
    }

    #[test]
    fn test_lfs_threshold_is_ten_mebibytes() {
        assert_eq!(LFS_INLINE_THRESHOLD_BYTES, 10 * 1024 * 1024);
    }

    #[test]
    fn test_run_blocking_returns_ok_value() {
        let result: std::result::Result<i32, HubError> = run_blocking(async { Ok(42) });
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_run_blocking_propagates_err() {
        let result: std::result::Result<i32, HubError> = run_blocking(async {
            Err(HubError::InvalidInput {
                message: "boom".to_string(),
            })
        });
        assert!(result.is_err());
    }
}
