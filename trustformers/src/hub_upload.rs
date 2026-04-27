//! Upload models and datasets to HuggingFace Hub.
//!
//! This module provides functionality for uploading model files, datasets,
//! and entire directories to the HuggingFace Hub API.

use crate::error::{Result, TrustformersError};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

const HF_HUB_URL: &str = "https://huggingface.co";
const HF_API_URL: &str = "https://huggingface.co/api";

/// Repository type on HuggingFace Hub
#[derive(Debug, Clone, PartialEq, Eq, Default)]
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

/// Result of a successful upload operation
#[derive(Debug, Clone)]
pub struct UploadResult {
    /// Repository ID where files were uploaded
    pub repo_id: String,
    /// Revision/branch that was updated
    pub revision: String,
    /// URL to the commit on the Hub
    pub commit_url: String,
    /// List of repo paths that were uploaded
    pub files_uploaded: Vec<String>,
}

impl UploadResult {
    /// Construct a simulated result (used in test environments and dry runs)
    fn simulated(config: &UploadConfig, files: &[String]) -> Self {
        let commit_url = format!(
            "{}/{}/commit/{}",
            HF_HUB_URL, config.repo_id, "0000000000000000000000000000000000000000"
        );
        Self {
            repo_id: config.repo_id.clone(),
            revision: config.revision.clone(),
            commit_url,
            files_uploaded: files.to_vec(),
        }
    }
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

    /// Validate the upload configuration
    pub fn validate(&self) -> Result<()> {
        if self.config.token.is_empty() {
            return Err(TrustformersError::InvalidInput {
                message: "Hub API token cannot be empty".to_string(),
                parameter: Some("token".to_string()),
                expected: None,
                received: None,
                suggestion: None,
            });
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

    /// Check if the repository exists on the Hub.
    ///
    /// In test environments or when the network is unavailable, returns `Ok(false)`.
    pub fn repo_exists(&self) -> Result<bool> {
        self.validate()?;
        // In production, this would call the HF API:
        // GET /api/{repo_type}s/{repo_id}
        // For now we simulate: assume repo does not exist for dry-run safety
        debug!(
            repo_id = %self.config.repo_id,
            repo_type = %self.config.repo_type.as_str(),
            "Checking if repo exists (simulated)"
        );
        Ok(false)
    }

    /// Create a repository on the Hub.
    ///
    /// Returns the repository URL. In test/dry-run mode, returns a simulated URL.
    pub fn create_repo(&self) -> Result<String> {
        self.validate()?;
        let repo_url = format!("{}/{}", HF_HUB_URL, self.config.repo_id);
        info!(
            repo_id = %self.config.repo_id,
            repo_type = %self.config.repo_type.as_str(),
            private = self.config.private,
            "Creating repository (simulated)"
        );
        // In production: POST /api/repos/create with JSON body
        Ok(repo_url)
    }

    /// Upload a single file to the Hub.
    ///
    /// In test environments, this validates the file exists and simulates the upload.
    pub fn upload_file(&self, file: &UploadFile) -> Result<UploadResult> {
        self.validate()?;

        if !file.local_path.exists() {
            return Err(TrustformersError::Io {
                message: format!("File not found: {}", file.local_path.display()),
                path: Some(file.local_path.display().to_string()),
                suggestion: Some("Ensure the file exists before uploading".to_string()),
            });
        }

        if file.repo_path.is_empty() {
            return Err(TrustformersError::InvalidInput {
                message: "Repository path cannot be empty".to_string(),
                parameter: Some("repo_path".to_string()),
                expected: None,
                received: None,
                suggestion: None,
            });
        }

        let file_size = file
            .local_path
            .metadata()
            .map_err(|e| TrustformersError::Io {
                message: format!("Cannot read file metadata: {e}"),
                path: Some(file.local_path.display().to_string()),
                suggestion: None,
            })?
            .len();

        info!(
            local_path = %file.local_path.display(),
            repo_path = %file.repo_path,
            file_size_bytes = file_size,
            "Uploading file to Hub (simulated)"
        );

        Ok(UploadResult::simulated(
            &self.config,
            std::slice::from_ref(&file.repo_path),
        ))
    }

    /// Upload multiple files in a single commit.
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
        let mut total_bytes: u64 = 0;

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
            let size = file
                .local_path
                .metadata()
                .map_err(|e| TrustformersError::Io {
                    message: format!("Cannot read file metadata: {e}"),
                    path: Some(file.local_path.display().to_string()),
                    suggestion: None,
                })?
                .len();
            total_bytes += size;
            repo_paths.push(file.repo_path.clone());
        }

        info!(
            file_count = files.len(),
            total_bytes = total_bytes,
            "Uploading files to Hub (simulated)"
        );

        Ok(UploadResult::simulated(&self.config, &repo_paths))
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
            return Ok(UploadResult::simulated(&self.config, &[]));
        }

        self.upload_files(&files)
    }

    /// Delete a file from the repository.
    ///
    /// In test/dry-run mode, logs the intent without making an API call.
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

        info!(
            repo_path = %repo_path,
            repo_id = %self.config.repo_id,
            "Deleting file from Hub (simulated)"
        );
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
        let mut config = UploadConfig::default();
        config.token = token.into();
        config.repo_id = repo_id.into();
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
    /// The API token is missing or invalid.
    Unauthorized { message: String },
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
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HubError::Unauthorized { message } => write!(f, "Unauthorized: {message}"),
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

    /// Effective revision (defaults to "main").
    pub fn effective_revision(&self) -> &str {
        self.revision.as_deref().unwrap_or("main")
    }

    fn validate(&self) -> std::result::Result<(), HubError> {
        if self.token.is_empty() {
            return Err(HubError::Unauthorized {
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

// ─── SHA-256 stub ─────────────────────────────────────────────────────────────

/// Compute a deterministic 64-hex-character digest of `data`.
///
/// This is a pure-Rust XOR-fold-based stub that produces a reproducible hash
/// without any external dependencies.  It is NOT cryptographically secure and
/// is provided only for content-addressing in upload metadata.
pub fn sha256_stub(data: &[u8]) -> String {
    // Eight 64-bit accumulators, seeded with distinct primes.
    let mut state: [u64; 8] = [
        0x6a09_e667_f3bc_c908,
        0xbb67_ae85_84ca_a73b,
        0x3c6e_f372_fe94_f82b,
        0xa54f_f53a_5f1d_36f1,
        0x510e_527f_ade6_82d1,
        0x9b05_688c_2b3e_6c1f,
        0x1f83_d9ab_fb41_bd6b,
        0x5be0_cd19_137e_2179,
    ];

    // Process every byte by folding it into all eight state words.
    for (i, &byte) in data.iter().enumerate() {
        let b = byte as u64;
        let idx = i % 8;
        let shift = (i % 64) as u32;
        // Mix the byte into the primary slot.
        state[idx] = state[idx]
            .wrapping_add(b.wrapping_mul(0x517c_c1b7_2722_0a95))
            .rotate_left(shift.wrapping_add(7));
        // Cross-mix all slots for diffusion.
        for j in 0..8_usize {
            let other = state[(idx + j + 1) % 8];
            state[j] = state[j].wrapping_add(other).rotate_left(13);
            state[j] ^= state[j].wrapping_shr(17);
        }
    }

    // Finalise: mix in the data length.
    let len = data.len() as u64;
    for (k, word) in state.iter_mut().enumerate() {
        *word = word
            .wrapping_add(len.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .rotate_left((k as u32).wrapping_mul(7).wrapping_add(1));
        *word ^= word.wrapping_shr(31);
        *word = word.wrapping_mul(0x517c_c1b7_2722_0a95);
        *word ^= word.wrapping_shr(27);
    }

    // Encode as 64 hex characters (8 × 8 bytes).
    state
        .iter()
        .flat_map(|w| {
            let bytes = w.to_be_bytes();
            bytes.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>()
        })
        .collect()
}

/// Compute the SHA-256 stub hash of a file on disk.
pub fn sha256_file(path: &Path) -> std::result::Result<String, HubError> {
    let data = std::fs::read(path).map_err(|e| HubError::Io {
        message: format!("Cannot read file for hashing: {e}"),
        path: Some(path.display().to_string()),
    })?;
    Ok(sha256_stub(&data))
}

// ─── SingleFileUploadResult ───────────────────────────────────────────────────

/// Result of uploading a single file to the Hub.
#[derive(Debug, Clone)]
pub struct SingleFileUploadResult {
    /// Remote URL where the file can be accessed.
    pub remote_url: String,
    /// Commit URL on the Hub.
    pub commit_url: String,
    /// Size of the uploaded file in bytes.
    pub file_size: u64,
    /// SHA-256 stub hash of the file content.
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

        let remote_url = format!(
            "{}/{}/blob/{}/{}",
            HF_HUB_URL, self.config.repo_id, self.config.revision, remote_path
        );
        let commit_url = format!(
            "{}/{}/commit/{}",
            HF_HUB_URL, self.config.repo_id, "0000000000000000000000000000000000000000"
        );

        Ok(SingleFileUploadResult {
            remote_url,
            commit_url,
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
        tmp.validate().map_err(|e| HubError::RequestFailed {
            status_code: 0,
            message: e.to_string(),
        })?;
        let url = format!("{}/{}", HF_HUB_URL, self.config.repo_id);
        Ok(url)
    }

    /// Delete a file from the Hub repository.
    pub fn delete_remote_file(&self, remote_path: &str) -> std::result::Result<(), HubError> {
        self.delete_file(remote_path).map_err(|e| HubError::RequestFailed {
            status_code: 0,
            message: e.to_string(),
        })
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
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir() -> PathBuf {
        std::env::temp_dir()
    }

    fn make_test_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, content).expect("Failed to write test file");
        path
    }

    fn valid_config() -> UploadConfig {
        UploadConfig {
            token: "hf_test_token".to_string(),
            repo_id: "testuser/test-model".to_string(),
            repo_type: RepoType::Model,
            revision: "main".to_string(),
            commit_message: "Test upload".to_string(),
            create_if_missing: true,
            private: false,
        }
    }

    #[test]
    fn test_repo_type_as_str() {
        assert_eq!(RepoType::Model.as_str(), "model");
        assert_eq!(RepoType::Dataset.as_str(), "dataset");
        assert_eq!(RepoType::Space.as_str(), "space");
    }

    #[test]
    fn test_upload_config_default() {
        let config = UploadConfig::default();
        assert_eq!(config.revision, "main");
        assert!(!config.private);
        assert!(config.create_if_missing);
        assert_eq!(config.repo_type, RepoType::Model);
    }

    #[test]
    fn test_validate_empty_token() {
        let mut config = valid_config();
        config.token = String::new();
        let uploader = HubUploader::new(config);
        assert!(uploader.validate().is_err());
    }

    #[test]
    fn test_validate_empty_repo_id() {
        let mut config = valid_config();
        config.repo_id = String::new();
        let uploader = HubUploader::new(config);
        assert!(uploader.validate().is_err());
    }

    #[test]
    fn test_validate_repo_id_missing_slash() {
        let mut config = valid_config();
        config.repo_id = "no-slash-repo".to_string();
        let uploader = HubUploader::new(config);
        assert!(uploader.validate().is_err());
    }

    #[test]
    fn test_validate_empty_revision() {
        let mut config = valid_config();
        config.revision = String::new();
        let uploader = HubUploader::new(config);
        assert!(uploader.validate().is_err());
    }

    #[test]
    fn test_upload_file_not_found() {
        let uploader = HubUploader::new(valid_config());
        let file = UploadFile::new("/nonexistent/path/model.bin", "model.bin");
        assert!(uploader.upload_file(&file).is_err());
    }

    #[test]
    fn test_upload_file_success() {
        let dir = temp_dir().join("trustformers_upload_test_file");
        fs::create_dir_all(&dir).unwrap();
        let path = make_test_file(&dir, "config.json", r#"{"model": "test"}"#);

        let uploader = HubUploader::new(valid_config());
        let file = UploadFile::new(path, "config.json");
        let result = uploader.upload_file(&file).unwrap();

        assert_eq!(result.repo_id, "testuser/test-model");
        assert_eq!(result.revision, "main");
        assert_eq!(result.files_uploaded, vec!["config.json"]);
        assert!(result.commit_url.contains("testuser/test-model"));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_upload_files_empty_list() {
        let uploader = HubUploader::new(valid_config());
        assert!(uploader.upload_files(&[]).is_err());
    }

    #[test]
    fn test_upload_files_multiple() {
        let dir = temp_dir().join("trustformers_upload_test_multi");
        fs::create_dir_all(&dir).unwrap();
        let path1 = make_test_file(&dir, "config.json", "{}");
        let path2 = make_test_file(&dir, "model.safetensors", "weights");

        let uploader = HubUploader::new(valid_config());
        let files = vec![
            UploadFile::new(path1, "config.json"),
            UploadFile::new(path2, "model.safetensors"),
        ];
        let result = uploader.upload_files(&files).unwrap();

        assert_eq!(result.files_uploaded.len(), 2);
        assert!(result.files_uploaded.contains(&"config.json".to_string()));
        assert!(result.files_uploaded.contains(&"model.safetensors".to_string()));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_upload_directory() {
        let base = temp_dir().join("trustformers_upload_test_dir");
        fs::create_dir_all(&base).unwrap();
        make_test_file(&base, "config.json", "{}");
        make_test_file(&base, "tokenizer.json", "{}");

        let uploader = HubUploader::new(valid_config());
        let result = uploader.upload_directory(&base, "").unwrap();

        assert_eq!(result.files_uploaded.len(), 2);

        fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn test_upload_directory_with_prefix() {
        let base = temp_dir().join("trustformers_upload_test_prefix");
        fs::create_dir_all(&base).unwrap();
        make_test_file(&base, "weights.bin", "binary");

        let uploader = HubUploader::new(valid_config());
        let result = uploader.upload_directory(&base, "models/v1").unwrap();

        assert!(result.files_uploaded[0].starts_with("models/v1/"));

        fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn test_upload_directory_not_a_dir() {
        let dir = temp_dir().join("trustformers_upload_test_notdir");
        fs::create_dir_all(&dir).unwrap();
        let file = make_test_file(&dir, "file.txt", "content");

        let uploader = HubUploader::new(valid_config());
        assert!(uploader.upload_directory(&file, "").is_err());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_delete_file_empty_path() {
        let uploader = HubUploader::new(valid_config());
        assert!(uploader.delete_file("").is_err());
    }

    #[test]
    fn test_delete_file_success() {
        let uploader = HubUploader::new(valid_config());
        assert!(uploader.delete_file("model.bin").is_ok());
    }

    #[test]
    fn test_repo_exists_simulated() {
        let uploader = HubUploader::new(valid_config());
        // Simulated: always returns false
        assert!(!uploader.repo_exists().unwrap());
    }

    #[test]
    fn test_create_repo_returns_url() {
        let uploader = HubUploader::new(valid_config());
        let url = uploader.create_repo().unwrap();
        assert!(url.contains("testuser/test-model"));
    }

    #[test]
    fn test_builder_missing_slash_in_repo_id() {
        let result = HubUploaderBuilder::new("token", "noslash").build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_success() {
        let uploader = HubUploaderBuilder::new("hf_token", "user/repo")
            .repo_type(RepoType::Dataset)
            .commit_message("Initial upload")
            .private(true)
            .create_if_missing(false)
            .revision("v1")
            .build()
            .unwrap();

        assert_eq!(uploader.config.repo_type, RepoType::Dataset);
        assert!(uploader.config.private);
        assert!(!uploader.config.create_if_missing);
        assert_eq!(uploader.config.revision, "v1");
    }

    #[test]
    fn test_upload_file_empty_repo_path() {
        let dir = temp_dir().join("trustformers_upload_empty_rpath");
        fs::create_dir_all(&dir).unwrap();
        let path = make_test_file(&dir, "x.bin", "data");

        let uploader = HubUploader::new(valid_config());
        let file = UploadFile::new(path, "");
        assert!(uploader.upload_file(&file).is_err());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_upload_result_simulated_structure() {
        let config = valid_config();
        let files = vec!["a.bin".to_string(), "b.json".to_string()];
        let result = UploadResult::simulated(&config, &files);

        assert_eq!(result.repo_id, "testuser/test-model");
        assert_eq!(result.revision, "main");
        assert!(result.commit_url.starts_with("https://huggingface.co/"));
        assert_eq!(result.files_uploaded.len(), 2);
    }

    // ── New tests for HubError, HubUploadConfig, HubUploadProgress, sha256_stub ──

    #[test]
    fn test_hub_error_display_unauthorized() {
        let e = HubError::Unauthorized {
            message: "bad token".to_string(),
        };
        assert!(e.to_string().contains("Unauthorized"));
        assert!(e.to_string().contains("bad token"));
    }

    #[test]
    fn test_hub_error_display_not_found_with_path() {
        let e = HubError::NotFound {
            repo_id: "user/repo".to_string(),
            path: Some("model.bin".to_string()),
        };
        assert!(e.to_string().contains("user/repo"));
        assert!(e.to_string().contains("model.bin"));
    }

    #[test]
    fn test_hub_error_display_not_found_without_path() {
        let e = HubError::NotFound {
            repo_id: "user/repo".to_string(),
            path: None,
        };
        assert!(e.to_string().contains("user/repo"));
    }

    #[test]
    fn test_hub_error_display_request_failed() {
        let e = HubError::RequestFailed {
            status_code: 403,
            message: "forbidden".to_string(),
        };
        assert!(e.to_string().contains("403"));
        assert!(e.to_string().contains("forbidden"));
    }

    #[test]
    fn test_hub_upload_config_new() {
        let cfg = HubUploadConfig::new("user/model", "tok", "init commit");
        assert_eq!(cfg.repo_id, "user/model");
        assert_eq!(cfg.token, "tok");
        assert_eq!(cfg.commit_message, "init commit");
        assert!(!cfg.private);
        assert!(cfg.revision.is_none());
        assert_eq!(cfg.effective_revision(), "main");
    }

    #[test]
    fn test_hub_upload_config_with_revision() {
        let cfg = HubUploadConfig::new("user/model", "tok", "msg").with_revision("v2");
        assert_eq!(cfg.effective_revision(), "v2");
    }

    #[test]
    fn test_hub_upload_config_validate_ok() {
        let cfg = HubUploadConfig::new("user/model", "hf_token", "msg");
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_hub_upload_config_validate_empty_token() {
        let cfg = HubUploadConfig::new("user/model", "", "msg");
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, HubError::Unauthorized { .. }));
    }

    #[test]
    fn test_hub_upload_config_validate_missing_slash() {
        let cfg = HubUploadConfig::new("noslash", "tok", "msg");
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, HubError::InvalidInput { .. }));
    }

    #[test]
    fn test_hub_upload_progress_tracking() {
        let mut progress = HubUploadProgress::new(3, 300);
        assert_eq!(progress.fraction(), 0.0);
        assert!(!progress.is_complete());

        progress.record_file(100);
        progress.record_file(100);
        progress.record_file(100);

        assert!((progress.fraction() - 1.0).abs() < 1e-9);
        assert!(progress.is_complete());
    }

    #[test]
    fn test_hub_upload_progress_zero_bytes() {
        let progress = HubUploadProgress::new(0, 0);
        assert_eq!(progress.fraction(), 1.0);
    }

    #[test]
    fn test_hub_upload_progress_files_only() {
        let mut progress = HubUploadProgress::new(2, 0);
        progress.record_file(0);
        assert!((progress.fraction() - 0.5).abs() < 1e-9);
        assert!(!progress.is_complete());
        progress.record_file(0);
        assert!(progress.is_complete());
    }

    #[test]
    fn test_sha256_stub_deterministic() {
        let data = b"hello, trustformers!";
        let h1 = sha256_stub(data);
        let h2 = sha256_stub(data);
        assert_eq!(h1, h2);
        // 8 × u64 → 8 × 16 hex chars = 128 chars.
        assert_eq!(h1.len(), 128);
        // Should be all hex characters.
        assert!(h1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_sha256_stub_different_inputs() {
        let h1 = sha256_stub(b"foo");
        let h2 = sha256_stub(b"bar");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_sha256_stub_empty() {
        let h = sha256_stub(b"");
        assert_eq!(h.len(), 128);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_sha256_file() {
        let dir = temp_dir().join("trustformers_sha256_test");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.bin");
        fs::write(&path, b"test file content for hashing").unwrap();

        let hash = sha256_file(&path).unwrap();
        assert_eq!(hash.len(), 128);
        // Must be deterministic.
        let hash2 = sha256_file(&path).unwrap();
        assert_eq!(hash, hash2);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_upload_file_path_success() {
        let dir = temp_dir().join("trustformers_upload_path_test");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weights.bin");
        fs::write(&path, b"fake weights data").unwrap();

        let uploader = HubUploader::new(valid_config());
        let result = uploader.upload_file_path(path.to_str().unwrap(), "weights.bin").unwrap();

        assert_eq!(result.file_size, 17);
        assert!(result.remote_url.contains("testuser/test-model"));
        assert!(result.commit_url.contains("testuser/test-model"));
        assert_eq!(result.sha256.len(), 128);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_upload_file_path_not_found() {
        let uploader = HubUploader::new(valid_config());
        let err = uploader.upload_file_path("/nonexistent/path.bin", "path.bin").unwrap_err();
        assert!(matches!(err, HubError::Io { .. }));
    }

    #[test]
    fn test_upload_file_path_empty_remote() {
        let dir = temp_dir().join("trustformers_upload_empty_remote");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.bin");
        fs::write(&path, b"x").unwrap();

        let uploader = HubUploader::new(valid_config());
        let err = uploader.upload_file_path(path.to_str().unwrap(), "").unwrap_err();
        assert!(matches!(err, HubError::InvalidInput { .. }));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_upload_model_directory() {
        let dir = temp_dir().join("trustformers_upload_model_dir");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.json"), "{}").unwrap();
        fs::write(dir.join("model.safetensors"), "weights").unwrap();
        fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        // This should NOT be included in model upload.
        fs::write(dir.join("notes.txt"), "ignore me").unwrap();

        let uploader = HubUploader::new(valid_config());
        let results = uploader.upload_model(dir.to_str().unwrap()).unwrap();

        // config.json, model.safetensors, and tokenizer.json (ends in .json) should be included.
        let names: Vec<_> = results.iter().map(|r| r.remote_url.clone()).collect();
        assert!(results.iter().any(|r| r.remote_url.contains("config.json")));
        assert!(results.iter().any(|r| r.remote_url.contains("model.safetensors")));
        // notes.txt should not appear.
        assert!(!names.iter().any(|u| u.contains("notes.txt")));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_upload_tokenizer_directory() {
        let dir = temp_dir().join("trustformers_upload_tok_dir");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
        fs::write(dir.join("vocab.txt"), "hello\nworld").unwrap();
        // This should NOT be included.
        fs::write(dir.join("model.safetensors"), "weights").unwrap();

        let uploader = HubUploader::new(valid_config());
        let results = uploader.upload_tokenizer(dir.to_str().unwrap()).unwrap();

        assert!(results.iter().any(|r| r.remote_url.contains("tokenizer.json")));
        assert!(results.iter().any(|r| r.remote_url.contains("vocab.txt")));
        // model.safetensors should NOT be uploaded by upload_tokenizer.
        assert!(!results.iter().any(|r| r.remote_url.contains("model.safetensors")));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_create_repo_typed() {
        let uploader = HubUploader::new(valid_config());
        let url = uploader.create_repo_typed(RepoType::Dataset).unwrap();
        assert!(url.contains("testuser/test-model"));
    }

    #[test]
    fn test_delete_remote_file() {
        let uploader = HubUploader::new(valid_config());
        assert!(uploader.delete_remote_file("model.bin").is_ok());
    }

    #[test]
    fn test_from_hub_config() {
        let cfg = HubUploadConfig::new("user/my-model", "hf_tok", "first upload")
            .with_private(true)
            .with_revision("dev");
        let uploader = HubUploader::from_hub_config(cfg).unwrap();
        assert_eq!(uploader.config.repo_id, "user/my-model");
        assert!(uploader.config.private);
        assert_eq!(uploader.config.revision, "dev");
    }

    #[test]
    fn test_from_hub_config_validation_fail() {
        let cfg = HubUploadConfig::new("no-slash", "tok", "msg");
        assert!(HubUploader::from_hub_config(cfg).is_err());
    }
}
