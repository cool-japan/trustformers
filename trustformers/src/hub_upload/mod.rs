//! Upload models and datasets to the HuggingFace Hub.
//!
//! `HubUploader` speaks the real Hub upload protocol (repo existence check,
//! repo creation, and the NDJSON commit API) via `reqwest`, behind the
//! `hub` feature — see `api` for the wire-level detail. Without a token,
//! every operation fails fast with [`HubError::MissingCredentials`] /
//! [`TrustformersError::Hub`] instead of proceeding. Without the `hub`
//! feature (no networking compiled in), every operation fails with
//! [`HubError::FeatureUnavailable`] instead of a fabricated success.
//!
//! An earlier revision of this module never contacted the Hub at all: every
//! upload/create/delete method validated its inputs, logged
//! `"(simulated)"`, and returned a synthetic [`UploadResult`] with a
//! commit URL built from the all-zeros SHA `"0000...0000"` — indistinguishable
//! from a real success to a caller that didn't read the log line. If a dry
//! run — validate everything, touch no network — is what's wanted, set
//! [`UploadConfig::dry_run`] explicitly; [`UploadResult::dry_run`] on the
//! returned value says which one happened.

mod api;

use crate::error::{Result, TrustformersError};
use api::{CommitFile, LFS_INLINE_THRESHOLD_BYTES};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

const HF_HUB_URL: &str = "https://huggingface.co";

/// Repository type on HuggingFace Hub
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RepoType {
    /// A model repository (default)
    #[default]
    Model,
    /// A dataset repository
    Dataset,
    /// A Spaces application
    Space,
}

impl RepoType {
    /// Returns the string representation used in API calls
    pub fn as_str(&self) -> &'static str {
        match self {
            RepoType::Model => "model",
            RepoType::Dataset => "dataset",
            RepoType::Space => "space",
        }
    }
}

/// Configuration for uploading to HuggingFace Hub
#[derive(Debug, Clone)]
pub struct UploadConfig {
    /// HuggingFace API token (required for upload)
    pub token: String,
    /// Repository ID in the format "username/model-name"
    pub repo_id: String,
    /// Repository type: Model, Dataset, or Space
    pub repo_type: RepoType,
    /// Branch/revision to upload to
    pub revision: String,
    /// Commit message for the upload
    pub commit_message: String,
    /// Whether to create the repository if it doesn't exist
    pub create_if_missing: bool,
    /// Whether the repository should be private
    pub private: bool,
    /// Base API endpoint. Defaults to the real Hugging Face Hub
    /// (`https://huggingface.co`); override to point at a local mock server
    /// in tests.
    pub base_url: String,
    /// When `true`, validate the token/repo id/files and report what *would*
    /// be uploaded without making any network request.
    /// [`UploadResult::dry_run`] is `true` on the result this produces.
    pub dry_run: bool,
}

impl Default for UploadConfig {
    fn default() -> Self {
        Self {
            token: String::new(),
            repo_id: String::new(),
            repo_type: RepoType::Model,
            revision: "main".to_string(),
            commit_message: "Upload via TrustformeRS".to_string(),
            create_if_missing: true,
            private: false,
            base_url: HF_HUB_URL.to_string(),
            dry_run: false,
        }
    }
}

/// Represents a single file to be uploaded
#[derive(Debug, Clone)]
pub struct UploadFile {
    /// Local path to the file on disk
    pub local_path: PathBuf,
    /// Destination path within the repository (relative to repo root)
    pub repo_path: String,
}

impl UploadFile {
    /// Create a new UploadFile
    pub fn new(local_path: impl Into<PathBuf>, repo_path: impl Into<String>) -> Self {
        Self {
            local_path: local_path.into(),
            repo_path: repo_path.into(),
        }
    }
}

/// Result of an upload operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UploadResult {
    /// Repository ID where files were (or, in a dry run, would be) uploaded
    pub repo_id: String,
    /// Revision/branch that was (or would be) updated
    pub revision: String,
    /// URL to the commit on the Hub. `None` in a dry run, or if the server's
    /// response didn't include one.
    pub commit_url: Option<String>,
    /// Commit SHA, when the server returned one.
    pub commit_oid: Option<String>,
    /// List of repo paths that were (or, in a dry run, would be) uploaded
    pub files_uploaded: Vec<String>,
    /// `true` if this result came from [`UploadConfig::dry_run`] rather than
    /// an actual upload.
    pub dry_run: bool,
}

/// Upload client for HuggingFace Hub
pub struct HubUploader {
    config: UploadConfig,
}

impl HubUploader {
    /// Create a new uploader from config
    pub fn new(config: UploadConfig) -> Self {
        Self { config }
    }

    /// Validate the upload configuration.
    ///
    /// An empty token is a [`TrustformersError::Hub`] "missing credentials"
    /// error, not a generic `InvalidInput` — callers can distinguish "you
    /// never gave me a token" from "the server rejected your token" (the
    /// latter surfaces as a `Hub` error from the network call itself).
    pub fn validate(&self) -> Result<()> {
        if self.config.token.is_empty() {
            return Err(missing_credentials_error(&self.config.repo_id));
        }
        if self.config.repo_id.is_empty() {
            return Err(TrustformersError::InvalidInput {
                message: "Repository ID cannot be empty".to_string(),
                parameter: Some("repo_id".to_string()),
                expected: None,
                received: None,
                suggestion: None,
            });
        }
        if !self.config.repo_id.contains('/') {
            return Err(TrustformersError::InvalidInput {
                message: "Repository ID must be in format 'username/repo-name'".to_string(),
                parameter: Some("repo_id".to_string()),
                expected: Some("username/repo-name".to_string()),
                received: Some(self.config.repo_id.clone()),
                suggestion: None,
            });
        }
        if self.config.revision.is_empty() {
            return Err(TrustformersError::InvalidInput {
                message: "Revision/branch name cannot be empty".to_string(),
                parameter: Some("revision".to_string()),
                expected: None,
                received: None,
                suggestion: None,
            });
        }
        Ok(())
    }

    /// Check whether the repository exists on the Hub
    /// (`GET /api/{repo_type}s/{repo_id}`).
    pub fn repo_exists(&self) -> Result<bool> {
        self.validate()?;
        if self.config.dry_run {
            debug!(repo_id = %self.config.repo_id, "repo_exists: dry run, skipping network call");
            return Ok(false);
        }
        api::run_blocking(api::repo_exists(
            &self.config.base_url,
            self.config.repo_type,
            &self.config.repo_id,
            &self.config.token,
        ))
        .map_err(|e| hub_error_to_trustformers(e, &self.config.repo_id))
    }

    /// Create a repository on the Hub (`POST /api/repos/create`).
    ///
    /// Returns the repository URL.
    pub fn create_repo(&self) -> Result<String> {
        self.validate()?;
        if self.config.dry_run {
            let url = format!("{}/{}", self.config.base_url, self.config.repo_id);
            info!(repo_id = %self.config.repo_id, "create_repo: dry run, not contacting the Hub");
            return Ok(url);
        }
        api::run_blocking(api::create_repo(
            &self.config.base_url,
            self.config.repo_type,
            &self.config.repo_id,
            self.config.private,
            &self.config.token,
        ))
        .map_err(|e| hub_error_to_trustformers(e, &self.config.repo_id))
    }

    /// Upload a single file to the Hub as a one-file commit.
    pub fn upload_file(&self, file: &UploadFile) -> Result<UploadResult> {
        self.upload_files(std::slice::from_ref(file))
    }

    /// Upload multiple files in a single commit.
    ///
    /// Creates the repository first if [`UploadConfig::create_if_missing`]
    /// is set and it doesn't already exist. Any file at or above
    /// `api::LFS_INLINE_THRESHOLD_BYTES` is refused *before* any network
    /// request — this module doesn't implement the real Git-LFS object
    /// upload (a separate preupload + batch-upload exchange), so silently
    /// either truncating it, inlining a huge base64 blob, or fabricating an
    /// `lfsFile` pointer to bytes that were never actually sent anywhere are
    /// all worse than a clear, immediate error.
    pub fn upload_files(&self, files: &[UploadFile]) -> Result<UploadResult> {
        self.validate()?;

        if files.is_empty() {
            return Err(TrustformersError::InvalidInput {
                message: "File list cannot be empty".to_string(),
                parameter: Some("files".to_string()),
                expected: None,
                received: None,
                suggestion: None,
            });
        }

        let mut repo_paths = Vec::with_capacity(files.len());
        let mut commit_files = Vec::with_capacity(files.len());

        for file in files {
            if !file.local_path.exists() {
                return Err(TrustformersError::Io {
                    message: format!("File not found: {}", file.local_path.display()),
                    path: Some(file.local_path.display().to_string()),
                    suggestion: Some("Ensure all files exist before uploading".to_string()),
                });
            }
            if file.repo_path.is_empty() {
                return Err(TrustformersError::InvalidInput {
                    message: "Repository path cannot be empty for one of the files".to_string(),
                    parameter: Some("repo_path".to_string()),
                    expected: None,
                    received: None,
                    suggestion: None,
                });
            }

            let content = std::fs::read(&file.local_path).map_err(|e| TrustformersError::Io {
                message: format!("Cannot read file: {e}"),
                path: Some(file.local_path.display().to_string()),
                suggestion: None,
            })?;
            if content.len() as u64 >= LFS_INLINE_THRESHOLD_BYTES {
                return Err(hub_error_to_trustformers(
                    HubError::LfsRequired {
                        path: file.repo_path.clone(),
                        size: content.len() as u64,
                    },
                    &self.config.repo_id,
                ));
            }

            repo_paths.push(file.repo_path.clone());
            commit_files.push(CommitFile {
                repo_path: file.repo_path.clone(),
                content,
            });
        }

        if self.config.dry_run {
            info!(
                file_count = files.len(),
                repo_id = %self.config.repo_id,
                "upload_files: dry run, not contacting the Hub"
            );
            return Ok(UploadResult {
                repo_id: self.config.repo_id.clone(),
                revision: self.config.revision.clone(),
                commit_url: None,
                commit_oid: None,
                files_uploaded: repo_paths,
                dry_run: true,
            });
        }

        if self.config.create_if_missing && !self.repo_exists()? {
            info!(repo_id = %self.config.repo_id, "Repository does not exist yet; creating it");
            self.create_repo()?;
        }

        let outcome = api::run_blocking(api::commit(
            &self.config.base_url,
            self.config.repo_type,
            &self.config.repo_id,
            &self.config.revision,
            &self.config.commit_message,
            &commit_files,
            &[],
            &self.config.token,
        ))
        .map_err(|e| hub_error_to_trustformers(e, &self.config.repo_id))?;

        info!(
            file_count = files.len(),
            repo_id = %self.config.repo_id,
            commit_url = ?outcome.commit_url,
            "Uploaded files to Hub"
        );

        Ok(UploadResult {
            repo_id: self.config.repo_id.clone(),
            revision: self.config.revision.clone(),
            commit_url: outcome.commit_url,
            commit_oid: outcome.commit_oid,
            files_uploaded: repo_paths,
            dry_run: false,
        })
    }

    /// Upload an entire directory to the Hub.
    ///
    /// All files under `local_dir` are recursively collected and uploaded.
    /// `repo_prefix` is prepended to each file's relative path in the repository.
    pub fn upload_directory(&self, local_dir: &Path, repo_prefix: &str) -> Result<UploadResult> {
        self.validate()?;

        if !local_dir.is_dir() {
            return Err(TrustformersError::Io {
                message: format!("Not a directory: {}", local_dir.display()),
                path: Some(local_dir.display().to_string()),
                suggestion: Some("Provide a path to an existing directory".to_string()),
            });
        }

        let files = collect_files_recursive(local_dir, local_dir, repo_prefix)?;

        if files.is_empty() {
            warn!(
                dir = %local_dir.display(),
                "Directory is empty; nothing to upload"
            );
            return Ok(UploadResult {
                repo_id: self.config.repo_id.clone(),
                revision: self.config.revision.clone(),
                commit_url: None,
                commit_oid: None,
                files_uploaded: vec![],
                dry_run: self.config.dry_run,
            });
        }

        self.upload_files(&files)
    }

    /// Delete a file from the repository (a commit with one `deletedFile` op).
    pub fn delete_file(&self, repo_path: &str) -> Result<()> {
        self.validate()?;

        if repo_path.is_empty() {
            return Err(TrustformersError::InvalidInput {
                message: "Repository path cannot be empty".to_string(),
                parameter: Some("repo_path".to_string()),
                expected: None,
                received: None,
                suggestion: None,
            });
        }

        if self.config.dry_run {
            info!(repo_path = %repo_path, repo_id = %self.config.repo_id, "delete_file: dry run, not contacting the Hub");
            return Ok(());
        }

        api::run_blocking(api::commit(
            &self.config.base_url,
            self.config.repo_type,
            &self.config.repo_id,
            &self.config.revision,
            &format!("Delete {repo_path}"),
            &[],
            std::slice::from_ref(&repo_path.to_string()),
            &self.config.token,
        ))
        .map_err(|e| hub_error_to_trustformers(e, &self.config.repo_id))?;

        info!(repo_path = %repo_path, repo_id = %self.config.repo_id, "Deleted file from Hub");
        Ok(())
    }
}

/// Recursively collect all files under `base_dir`, building UploadFile entries.
fn collect_files_recursive(
    base_dir: &Path,
    current_dir: &Path,
    repo_prefix: &str,
) -> Result<Vec<UploadFile>> {
    let mut files = Vec::new();

    let entries = std::fs::read_dir(current_dir).map_err(|e| TrustformersError::Io {
        message: format!("Cannot read directory: {e}"),
        path: Some(current_dir.display().to_string()),
        suggestion: None,
    })?;

    for entry_result in entries {
        let entry = entry_result.map_err(|e| TrustformersError::Io {
            message: format!("Cannot read directory entry: {e}"),
            path: Some(current_dir.display().to_string()),
            suggestion: None,
        })?;

        let path = entry.path();

        if path.is_dir() {
            let mut sub_files = collect_files_recursive(base_dir, &path, repo_prefix)?;
            files.append(&mut sub_files);
        } else {
            let relative = path.strip_prefix(base_dir).map_err(|e| TrustformersError::Io {
                message: format!("Path prefix stripping failed: {e}"),
                path: Some(path.display().to_string()),
                suggestion: None,
            })?;

            let repo_path = if repo_prefix.is_empty() {
                relative.display().to_string()
            } else {
                format!("{}/{}", repo_prefix, relative.display())
            };

            // Normalise OS-specific path separators to forward slashes
            let repo_path = repo_path.replace('\\', "/");

            files.push(UploadFile {
                local_path: path.clone(),
                repo_path,
            });
        }
    }

    Ok(files)
}

/// Builder pattern for constructing a `HubUploader`
pub struct HubUploaderBuilder {
    config: UploadConfig,
}

impl HubUploaderBuilder {
    /// Start building with required fields: token and repo_id
    pub fn new(token: impl Into<String>, repo_id: impl Into<String>) -> Self {
        let config = UploadConfig {
            token: token.into(),
            repo_id: repo_id.into(),
            ..Default::default()
        };
        Self { config }
    }

    /// Set the repository type
    pub fn repo_type(mut self, repo_type: RepoType) -> Self {
        self.config.repo_type = repo_type;
        self
    }

    /// Set the branch/revision to upload to
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.config.revision = revision.into();
        self
    }

    /// Set the commit message
    pub fn commit_message(mut self, msg: impl Into<String>) -> Self {
        self.config.commit_message = msg.into();
        self
    }

    /// Set whether the repository should be private
    pub fn private(mut self, private: bool) -> Self {
        self.config.private = private;
        self
    }

    /// Set whether to create the repository if it doesn't exist
    pub fn create_if_missing(mut self, create: bool) -> Self {
        self.config.create_if_missing = create;
        self
    }

    /// Override the base API endpoint (for pointing at a local mock server).
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.config.base_url = base_url.into();
        self
    }

    /// Set dry-run mode: validate everything, touch no network.
    pub fn dry_run(mut self, dry_run: bool) -> Self {
        self.config.dry_run = dry_run;
        self
    }

    /// Build the `HubUploader`, validating the configuration first
    pub fn build(self) -> Result<HubUploader> {
        let uploader = HubUploader::new(self.config);
        uploader.validate()?;
        Ok(uploader)
    }
}

// ─── HubError ─────────────────────────────────────────────────────────────────

/// Dedicated error type for Hub upload/download operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HubError {
    /// The Hub rejected the credentials that were sent (HTTP 401/403).
    Unauthorized { message: String },
    /// No API token was supplied at all — distinct from `Unauthorized`
    /// (which means a token *was* sent and the server rejected it).
    MissingCredentials { message: String },
    /// A requested resource was not found on the Hub.
    NotFound {
        repo_id: String,
        path: Option<String>,
    },
    /// The request was rejected by the Hub (e.g., quota exceeded).
    RequestFailed { status_code: u16, message: String },
    /// A local file system operation failed.
    Io {
        message: String,
        path: Option<String>,
    },
    /// Input validation failed.
    InvalidInput { message: String },
    /// Network connectivity issue.
    Network { message: String },
    /// The `hub` feature (networking) is not compiled in.
    FeatureUnavailable { message: String },
    /// A file is too large to inline as base64 in a commit and would need
    /// real Git-LFS object storage, which this module does not implement.
    LfsRequired { path: String, size: u64 },
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HubError::Unauthorized { message } => write!(f, "Unauthorized: {message}"),
            HubError::MissingCredentials { message } => write!(f, "Missing credentials: {message}"),
            HubError::NotFound { repo_id, path } => {
                if let Some(p) = path {
                    write!(f, "Not found: {repo_id}/{p}")
                } else {
                    write!(f, "Not found: {repo_id}")
                }
            },
            HubError::RequestFailed {
                status_code,
                message,
            } => {
                write!(f, "Request failed (HTTP {status_code}): {message}")
            },
            HubError::Io { message, path } => {
                if let Some(p) = path {
                    write!(f, "IO error at {p}: {message}")
                } else {
                    write!(f, "IO error: {message}")
                }
            },
            HubError::InvalidInput { message } => write!(f, "Invalid input: {message}"),
            HubError::Network { message } => write!(f, "Network error: {message}"),
            HubError::FeatureUnavailable { message } => write!(f, "Feature unavailable: {message}"),
            HubError::LfsRequired { path, size } => {
                write!(
                    f,
                    "'{path}' is {size} bytes, at or above the {}-byte inline-upload threshold, \
                     and would require real Git-LFS object storage, which is not implemented",
                    LFS_INLINE_THRESHOLD_BYTES
                )
            },
        }
    }
}

impl std::error::Error for HubError {}

impl From<TrustformersError> for HubError {
    fn from(e: TrustformersError) -> Self {
        HubError::RequestFailed {
            status_code: 0,
            message: e.to_string(),
        }
    }
}

/// Map a [`HubError`] to the crate-wide [`TrustformersError`], carrying
/// `repo_id`/`model_id` context along for the `Hub` variants.
fn hub_error_to_trustformers(error: HubError, repo_id: &str) -> TrustformersError {
    match error {
        HubError::MissingCredentials { .. } => missing_credentials_error(repo_id),
        HubError::Unauthorized { message } => TrustformersError::Hub {
            message,
            model_id: repo_id.to_string(),
            endpoint: None,
            suggestion: Some("Check that the API token is valid and has write access".to_string()),
            recovery_actions: vec![],
        },
        HubError::FeatureUnavailable { message } => TrustformersError::Hub {
            message,
            model_id: repo_id.to_string(),
            endpoint: None,
            suggestion: Some("Rebuild with `--features hub`".to_string()),
            recovery_actions: vec![],
        },
        HubError::LfsRequired { path, size } => TrustformersError::Hub {
            message: format!(
                "'{path}' is {size} bytes and would require real Git-LFS object storage, which \
                 is not implemented"
            ),
            model_id: repo_id.to_string(),
            endpoint: None,
            suggestion: Some("Upload large files through the Hub web UI or the CLI's `huggingface-cli upload-large-folder` until LFS object upload is implemented here".to_string()),
            recovery_actions: vec![],
        },
        HubError::NotFound { repo_id, path } => TrustformersError::Hub {
            message: format!("Not found: {repo_id}{}", path.map(|p| format!("/{p}")).unwrap_or_default()),
            model_id: repo_id,
            endpoint: None,
            suggestion: None,
            recovery_actions: vec![],
        },
        HubError::RequestFailed { status_code, message } => TrustformersError::Hub {
            message: format!("HTTP {status_code}: {message}"),
            model_id: repo_id.to_string(),
            endpoint: None,
            suggestion: None,
            recovery_actions: vec![],
        },
        HubError::Network { message } => TrustformersError::Hub {
            message,
            model_id: repo_id.to_string(),
            endpoint: None,
            suggestion: Some("Check network connectivity".to_string()),
            recovery_actions: vec![],
        },
        HubError::Io { message, path } => TrustformersError::Io {
            message,
            path,
            suggestion: None,
        },
        HubError::InvalidInput { message } => TrustformersError::InvalidInput {
            message,
            parameter: None,
            expected: None,
            received: None,
            suggestion: None,
        },
    }
}

fn missing_credentials_error(repo_id: &str) -> TrustformersError {
    TrustformersError::Hub {
        message: "Missing credentials: a Hub API token is required to upload".to_string(),
        model_id: repo_id.to_string(),
        endpoint: None,
        suggestion: Some(
            "Set `UploadConfig::token` (e.g. from the `HF_TOKEN` environment variable)".to_string(),
        ),
        recovery_actions: vec![],
    }
}

// ─── HubUploadConfig ──────────────────────────────────────────────────────────

/// Simplified upload configuration with named fields that mirror the HF Hub API.
#[derive(Debug, Clone)]
pub struct HubUploadConfig {
    /// Repository ID in "username/repo-name" format.
    pub repo_id: String,
    /// HuggingFace API token.
    pub token: String,
    /// Commit message to use when uploading.
    pub commit_message: String,
    /// Whether the repository is private.
    pub private: bool,
    /// Branch/revision to upload to. `None` defaults to "main".
    pub revision: Option<String>,
    /// Base API endpoint override (for tests).
    pub base_url: Option<String>,
    /// Dry-run mode (see [`UploadConfig::dry_run`]).
    pub dry_run: bool,
}

impl HubUploadConfig {
    /// Create a new config with required fields.
    pub fn new(
        repo_id: impl Into<String>,
        token: impl Into<String>,
        commit_message: impl Into<String>,
    ) -> Self {
        Self {
            repo_id: repo_id.into(),
            token: token.into(),
            commit_message: commit_message.into(),
            private: false,
            revision: None,
            base_url: None,
            dry_run: false,
        }
    }

    /// Set the private flag.
    pub fn with_private(mut self, private: bool) -> Self {
        self.private = private;
        self
    }

    /// Set the target revision/branch.
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// Override the base API endpoint (for pointing at a local mock server).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Enable dry-run mode.
    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Effective revision (defaults to "main").
    pub fn effective_revision(&self) -> &str {
        self.revision.as_deref().unwrap_or("main")
    }

    fn validate(&self) -> std::result::Result<(), HubError> {
        if self.token.is_empty() {
            return Err(HubError::MissingCredentials {
                message: "API token cannot be empty".to_string(),
            });
        }
        if self.repo_id.is_empty() {
            return Err(HubError::InvalidInput {
                message: "repo_id cannot be empty".to_string(),
            });
        }
        if !self.repo_id.contains('/') {
            return Err(HubError::InvalidInput {
                message: format!(
                    "repo_id must be in 'username/repo-name' format, got '{}'",
                    self.repo_id
                ),
            });
        }
        Ok(())
    }
}

// ─── HubUploadProgress ────────────────────────────────────────────────────────

/// Tracks progress of a multi-file upload operation.
#[derive(Debug, Clone, Default)]
pub struct HubUploadProgress {
    /// Total number of files to upload.
    pub total_files: usize,
    /// Number of files that have been uploaded so far.
    pub uploaded_files: usize,
    /// Total bytes across all files.
    pub total_bytes: u64,
    /// Bytes uploaded so far.
    pub uploaded_bytes: u64,
}

impl HubUploadProgress {
    /// Create a new progress tracker.
    pub fn new(total_files: usize, total_bytes: u64) -> Self {
        Self {
            total_files,
            uploaded_files: 0,
            total_bytes,
            uploaded_bytes: 0,
        }
    }

    /// Mark a file as uploaded.
    pub fn record_file(&mut self, bytes: u64) {
        self.uploaded_files += 1;
        self.uploaded_bytes += bytes;
    }

    /// Returns upload completion as a value in `[0.0, 1.0]`.
    pub fn fraction(&self) -> f64 {
        if self.total_bytes == 0 {
            if self.total_files == 0 {
                1.0
            } else {
                self.uploaded_files as f64 / self.total_files as f64
            }
        } else {
            self.uploaded_bytes as f64 / self.total_bytes as f64
        }
    }

    /// Returns `true` when all files are uploaded.
    pub fn is_complete(&self) -> bool {
        self.uploaded_files >= self.total_files
    }
}

// ─── SHA-256 ──────────────────────────────────────────────────────────────────

/// Compute the real SHA-256 digest of `data`, returned as 64 lowercase hex chars.
pub fn sha256(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Compute the SHA-256 hash of a file on disk.
pub fn sha256_file(path: &Path) -> std::result::Result<String, HubError> {
    let data = std::fs::read(path).map_err(|e| HubError::Io {
        message: format!("Cannot read file for hashing: {e}"),
        path: Some(path.display().to_string()),
    })?;
    Ok(sha256(&data))
}

// ─── SingleFileUploadResult ───────────────────────────────────────────────────

/// Result of uploading a single file to the Hub.
#[derive(Debug, Clone)]
pub struct SingleFileUploadResult {
    /// Remote URL where the file can be accessed.
    pub remote_url: String,
    /// Commit URL on the Hub, when the server returned one.
    pub commit_url: Option<String>,
    /// Commit SHA, when the server returned one.
    pub commit_oid: Option<String>,
    /// Size of the uploaded file in bytes.
    pub file_size: u64,
    /// SHA-256 hash of the file content.
    pub sha256: String,
}

// ─── Extensions on HubUploader ────────────────────────────────────────────────

impl HubUploader {
    /// Create a `HubUploader` from a `HubUploadConfig`.
    pub fn from_hub_config(cfg: HubUploadConfig) -> std::result::Result<Self, HubError> {
        cfg.validate()?;
        let revision = cfg.effective_revision().to_string();
        let upload_config = UploadConfig {
            token: cfg.token,
            repo_id: cfg.repo_id,
            repo_type: RepoType::Model,
            revision,
            commit_message: cfg.commit_message,
            create_if_missing: true,
            private: cfg.private,
            base_url: cfg.base_url.unwrap_or_else(|| HF_HUB_URL.to_string()),
            dry_run: cfg.dry_run,
        };
        Ok(Self::new(upload_config))
    }

    /// Upload a single local file by path, returning a rich `SingleFileUploadResult`.
    pub fn upload_file_path(
        &self,
        local_path: &str,
        remote_path: &str,
    ) -> std::result::Result<SingleFileUploadResult, HubError> {
        let path = Path::new(local_path);
        if !path.exists() {
            return Err(HubError::Io {
                message: format!("File not found: {local_path}"),
                path: Some(local_path.to_string()),
            });
        }
        if remote_path.is_empty() {
            return Err(HubError::InvalidInput {
                message: "remote_path cannot be empty".to_string(),
            });
        }

        let metadata = path.metadata().map_err(|e| HubError::Io {
            message: format!("Cannot read file metadata: {e}"),
            path: Some(local_path.to_string()),
        })?;
        let file_size = metadata.len();
        let sha256 = sha256_file(path)?;

        let result =
            self.upload_file(&UploadFile::new(path, remote_path)).map_err(HubError::from)?;

        let remote_url = format!(
            "{}/{}/blob/{}/{}",
            self.config.base_url, self.config.repo_id, self.config.revision, remote_path
        );

        Ok(SingleFileUploadResult {
            remote_url,
            commit_url: result.commit_url,
            commit_oid: result.commit_oid,
            file_size,
            sha256,
        })
    }

    /// Upload all files in a model directory (config.json, *.safetensors, tokenizer files, etc.).
    ///
    /// Returns one `SingleFileUploadResult` per file found.
    pub fn upload_model(
        &self,
        model_dir: &str,
    ) -> std::result::Result<Vec<SingleFileUploadResult>, HubError> {
        let base = Path::new(model_dir);
        if !base.is_dir() {
            return Err(HubError::Io {
                message: format!("Not a directory: {model_dir}"),
                path: Some(model_dir.to_string()),
            });
        }
        self.upload_dir_filtered(base, |name| {
            // Upload model-relevant files: config, weights, generation config, etc.
            name.ends_with(".json")
                || name.ends_with(".safetensors")
                || name.ends_with(".bin")
                || name.ends_with(".pt")
                || name.ends_with(".ckpt")
                || name.ends_with(".msgpack")
                || name.ends_with(".model")
                || name == "README.md"
        })
    }

    /// Upload tokenizer files from a directory (tokenizer.json, vocab.txt, merges.txt, etc.).
    ///
    /// Returns one `SingleFileUploadResult` per file found.
    pub fn upload_tokenizer(
        &self,
        tokenizer_dir: &str,
    ) -> std::result::Result<Vec<SingleFileUploadResult>, HubError> {
        let base = Path::new(tokenizer_dir);
        if !base.is_dir() {
            return Err(HubError::Io {
                message: format!("Not a directory: {tokenizer_dir}"),
                path: Some(tokenizer_dir.to_string()),
            });
        }
        self.upload_dir_filtered(base, |name| {
            name.ends_with("tokenizer.json")
                || name.ends_with("tokenizer_config.json")
                || name.ends_with("vocab.json")
                || name.ends_with("vocab.txt")
                || name.ends_with("merges.txt")
                || name.ends_with("special_tokens_map.json")
                || name.ends_with("added_tokens.json")
                || name.ends_with(".model")
                || name.ends_with("spiece.model")
        })
    }

    /// Create a repository on the Hub for the given `repo_type`.
    ///
    /// Returns the new repository URL.
    pub fn create_repo_typed(&self, repo_type: RepoType) -> std::result::Result<String, HubError> {
        let mut cfg = self.config.clone();
        cfg.repo_type = repo_type;
        let tmp = HubUploader::new(cfg);
        tmp.create_repo().map_err(HubError::from)
    }

    /// Delete a file from the Hub repository.
    pub fn delete_remote_file(&self, remote_path: &str) -> std::result::Result<(), HubError> {
        self.delete_file(remote_path).map_err(HubError::from)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn upload_dir_filtered<F>(
        &self,
        base: &Path,
        filter: F,
    ) -> std::result::Result<Vec<SingleFileUploadResult>, HubError>
    where
        F: Fn(&str) -> bool,
    {
        let entries = collect_files_recursive_hub(base, base)?;
        let mut results = Vec::new();
        for (local_path, repo_path) in entries {
            let file_name = local_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !filter(file_name) {
                continue;
            }
            let local_str = local_path.display().to_string();
            let result = self.upload_file_path(&local_str, &repo_path)?;
            results.push(result);
        }
        Ok(results)
    }
}

/// Recursively collect all files under `base`, returning (local_path, repo_relative_path) pairs.
fn collect_files_recursive_hub(
    base: &Path,
    current: &Path,
) -> std::result::Result<Vec<(PathBuf, String)>, HubError> {
    let mut files = Vec::new();
    let entries = std::fs::read_dir(current).map_err(|e| HubError::Io {
        message: format!("Cannot read directory: {e}"),
        path: Some(current.display().to_string()),
    })?;
    for entry_result in entries {
        let entry = entry_result.map_err(|e| HubError::Io {
            message: format!("Cannot read directory entry: {e}"),
            path: Some(current.display().to_string()),
        })?;
        let path = entry.path();
        if path.is_dir() {
            let mut sub = collect_files_recursive_hub(base, &path)?;
            files.append(&mut sub);
        } else {
            let relative = path.strip_prefix(base).map_err(|e| HubError::Io {
                message: format!("Path strip prefix failed: {e}"),
                path: Some(path.display().to_string()),
            })?;
            let repo_path = relative.display().to_string().replace('\\', "/");
            files.push((path, repo_path));
        }
    }
    Ok(files)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
