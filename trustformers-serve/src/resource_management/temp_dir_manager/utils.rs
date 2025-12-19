//! Utility Functions for Temporary Directory Manager
//!
//! This module contains helper functions and utilities used throughout the temporary
//! directory management system, including file system operations, permissions management,
//! and disk space calculations.

use super::types::*;
use crate::resource_management::types::*;

use std::{
    fs,
    io::{self, ErrorKind},
    path::Path,
};
use tracing::{debug, warn};

// ================================================================================================
// File System Utilities
// ================================================================================================

/// Calculate the total size and file count of a directory recursively
pub fn calculate_directory_size(path: &Path) -> Result<(u64, usize), io::Error> {
    let mut total_size = 0u64;
    let mut file_count = 0usize;

    calculate_directory_size_recursive(path, &mut total_size, &mut file_count)?;

    Ok((total_size, file_count))
}

/// Calculate detailed directory usage statistics
pub fn calculate_directory_usage(path: &Path) -> Result<DirectorySpaceUsage, io::Error> {
    let mut usage = DirectorySpaceUsage::new();

    calculate_directory_usage_recursive(path, &mut usage)?;
    usage.update_average_file_size();

    Ok(usage)
}

/// Recursively calculate directory size
fn calculate_directory_size_recursive(
    path: &Path,
    total_size: &mut u64,
    file_count: &mut usize,
) -> Result<(), io::Error> {
    if !path.exists() {
        return Ok(());
    }

    let entries = fs::read_dir(path)?;

    for entry in entries {
        let entry = entry?;
        let metadata = entry.metadata()?;

        if metadata.is_file() {
            *total_size += metadata.len();
            *file_count += 1;
        } else if metadata.is_dir() {
            calculate_directory_size_recursive(&entry.path(), total_size, file_count)?;
        }
    }

    Ok(())
}

/// Recursively calculate detailed directory usage
fn calculate_directory_usage_recursive(
    path: &Path,
    usage: &mut DirectorySpaceUsage,
) -> Result<(), io::Error> {
    if !path.exists() {
        return Ok(());
    }

    let entries = fs::read_dir(path)?;

    for entry in entries {
        let entry = entry?;
        let metadata = entry.metadata()?;

        if metadata.is_file() {
            let size = metadata.len();
            usage.total_size += size;
            usage.file_count += 1;
            usage.largest_file_size = usage.largest_file_size.max(size);

            // Track usage by file extension
            if let Some(extension) = entry.path().extension() {
                let ext = extension.to_string_lossy().to_lowercase();
                *usage.usage_by_type.entry(ext).or_insert(0) += size;
            }
        } else if metadata.is_dir() {
            usage.subdirectory_count += 1;
            calculate_directory_usage_recursive(&entry.path(), usage)?;
        }
    }

    Ok(())
}

/// Get available disk space for a given path
pub fn get_available_disk_space(path: &Path) -> TempDirResult<u64> {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::mem;
        use std::os::unix::ffi::OsStrExt;

        let path_cstring =
            CString::new(path.as_os_str().as_bytes()).map_err(|_| TempDirError::InternalError {
                message: "Invalid path for disk space calculation".to_string(),
            })?;

        let mut statvfs_buf = unsafe { mem::zeroed() };
        let result = unsafe { libc::statvfs(path_cstring.as_ptr(), &mut statvfs_buf) };

        if result == 0 {
            let available_bytes = statvfs_buf.f_bavail as u64 * statvfs_buf.f_frsize;
            Ok(available_bytes)
        } else {
            // Fallback to a simple check
            Ok(u64::MAX) // Assume unlimited space if we can't determine
        }
    }

    #[cfg(not(unix))]
    {
        // Simplified implementation for non-Unix systems
        // In a real implementation, you'd use GetDiskFreeSpaceEx on Windows
        match fs::metadata(path) {
            Ok(_) => Ok(u64::MAX), // Assume unlimited space
            Err(e) => Err(TempDirError::IoError {
                path: path.display().to_string(),
                source: e,
            }),
        }
    }
}

/// Check if a path has sufficient free space
pub fn has_sufficient_space(path: &Path, required_bytes: u64) -> TempDirResult<bool> {
    let available = get_available_disk_space(path)?;
    Ok(available >= required_bytes)
}

/// Clean empty directories recursively
pub fn clean_empty_directories(path: &Path) -> Result<usize, io::Error> {
    let mut cleaned_count = 0;

    if !path.exists() || !path.is_dir() {
        return Ok(0);
    }

    let entries = fs::read_dir(path)?;
    let mut subdirectories = Vec::new();

    // First, process all entries and collect subdirectories
    for entry in entries {
        let entry = entry?;
        let entry_path = entry.path();

        if entry_path.is_dir() {
            subdirectories.push(entry_path);
        }
    }

    // Recursively clean subdirectories first
    for subdir in subdirectories {
        cleaned_count += clean_empty_directories(&subdir)?;

        // After cleaning subdirectory, check if it's now empty
        if is_directory_empty(&subdir)? {
            match fs::remove_dir(&subdir) {
                Ok(()) => {
                    cleaned_count += 1;
                    debug!(path = %subdir.display(), "Removed empty directory");
                },
                Err(e) if e.kind() == ErrorKind::NotFound => {
                    // Directory was already removed, that's fine
                },
                Err(e) => {
                    warn!(path = %subdir.display(), error = %e, "Failed to remove empty directory");
                },
            }
        }
    }

    Ok(cleaned_count)
}

/// Check if a directory is empty
pub fn is_directory_empty(path: &Path) -> Result<bool, io::Error> {
    if !path.exists() {
        return Ok(true);
    }

    if !path.is_dir() {
        return Ok(false);
    }

    let mut entries = fs::read_dir(path)?;
    Ok(entries.next().is_none())
}

/// Create directory with parents if they don't exist
pub fn create_directory_recursive(path: &Path) -> TempDirResult<()> {
    fs::create_dir_all(path).map_err(|e| TempDirError::IoError {
        path: path.display().to_string(),
        source: e,
    })
}

/// Remove directory and all its contents safely
pub fn remove_directory_recursive(path: &Path) -> TempDirResult<()> {
    if !path.exists() {
        return Ok(());
    }

    fs::remove_dir_all(path).map_err(|e| TempDirError::CleanupFailed {
        path: path.display().to_string(),
        message: format!("Failed to remove directory: {}", e),
    })
}

// ================================================================================================
// Permission Management Utilities
// ================================================================================================

/// Set directory permissions based on the platform
pub fn set_directory_permissions(
    path: &Path,
    permissions: &DirectoryPermissions,
) -> TempDirResult<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut mode = 0o000;

        if permissions.owner_read {
            mode |= 0o400;
        }
        if permissions.owner_write {
            mode |= 0o200;
        }
        if permissions.owner_execute {
            mode |= 0o100;
        }

        mode |= (permissions.group_permissions as u32) << 3;
        mode |= permissions.other_permissions as u32;

        let metadata = fs::metadata(path).map_err(|e| TempDirError::IoError {
            path: path.display().to_string(),
            source: e,
        })?;

        let mut perms = metadata.permissions();
        perms.set_mode(mode);
        fs::set_permissions(path, perms).map_err(|e| TempDirError::IoError {
            path: path.display().to_string(),
            source: e,
        })?;

        debug!(path = %path.display(), mode = %format!("{:o}", mode), "Set Unix permissions");
    }

    #[cfg(not(unix))]
    {
        // On non-Unix systems, just set read-only based on write permission
        let metadata = fs::metadata(path).map_err(|e| TempDirError::IoError {
            path: path.display().to_string(),
            source: e,
        })?;

        let mut perms = metadata.permissions();
        perms.set_readonly(!permissions.owner_write);
        fs::set_permissions(path, perms).map_err(|e| TempDirError::IoError {
            path: path.display().to_string(),
            source: e,
        })?;

        debug!(path = %path.display(), readonly = %!permissions.owner_write, "Set Windows permissions");
    }

    Ok(())
}

/// Get current directory permissions
pub fn get_directory_permissions(path: &Path) -> TempDirResult<DirectoryPermissions> {
    let metadata = fs::metadata(path).map_err(|e| TempDirError::IoError {
        path: path.display().to_string(),
        source: e,
    })?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mode = metadata.permissions().mode();

        Ok(DirectoryPermissions {
            owner_read: (mode & 0o400) != 0,
            owner_write: (mode & 0o200) != 0,
            owner_execute: (mode & 0o100) != 0,
            group_permissions: ((mode >> 3) & 0o7) as u8,
            other_permissions: (mode & 0o7) as u8,
        })
    }

    #[cfg(not(unix))]
    {
        let readonly = metadata.permissions().readonly();

        Ok(DirectoryPermissions {
            owner_read: true,
            owner_write: !readonly,
            owner_execute: true,
            group_permissions: if readonly { 0o5 } else { 0o7 },
            other_permissions: if readonly { 0o5 } else { 0o7 },
        })
    }
}

/// Check if the current process has write permissions for a path
pub fn has_write_permission(path: &Path) -> bool {
    if !path.exists() {
        // Check parent directory if path doesn't exist
        if let Some(parent) = path.parent() {
            return has_write_permission(parent);
        }
        return false;
    }

    // Try to create a temporary file to test write permissions
    let test_file = path.join(".write_test");
    match fs::write(&test_file, b"test") {
        Ok(()) => {
            let _ = fs::remove_file(&test_file);
            true
        },
        Err(_) => false,
    }
}

// ================================================================================================
// Path and Naming Utilities
// ================================================================================================

/// Generate a unique directory name
pub fn generate_unique_directory_name(test_id: &str, index: usize) -> String {
    let timestamp = chrono::Utc::now().timestamp_millis();
    let random_suffix = uuid::Uuid::new_v4().to_string()[..8].to_string();
    format!("test_{}_{}_{}_{}", test_id, timestamp, index, random_suffix)
}

/// Sanitize a string for use in file paths
pub fn sanitize_path_component(input: &str) -> String {
    input
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect()
}

/// Check if a path is safe (within allowed boundaries)
pub fn is_path_safe(path: &Path, base_path: &Path) -> bool {
    // Ensure the path is absolute
    if !path.is_absolute() {
        return false;
    }

    // Ensure the path is within the base directory
    match path.strip_prefix(base_path) {
        Ok(relative_path) => {
            // Check for path traversal attempts
            for component in relative_path.components() {
                match component {
                    std::path::Component::ParentDir => return false,
                    std::path::Component::CurDir => return false,
                    _ => {},
                }
            }
            true
        },
        Err(_) => false,
    }
}

/// Resolve symlinks and get canonical path
pub fn get_canonical_path(path: &Path) -> TempDirResult<std::path::PathBuf> {
    path.canonicalize().map_err(|e| TempDirError::IoError {
        path: path.display().to_string(),
        source: e,
    })
}

// ================================================================================================
// File Operation Utilities
// ================================================================================================

/// Copy directory contents recursively
pub fn copy_directory_contents(src: &Path, dst: &Path) -> TempDirResult<()> {
    if !src.exists() {
        return Err(TempDirError::DirectoryNotFound {
            path: src.display().to_string(),
        });
    }

    create_directory_recursive(dst)?;

    let entries = fs::read_dir(src).map_err(|e| TempDirError::IoError {
        path: src.display().to_string(),
        source: e,
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| TempDirError::IoError {
            path: src.display().to_string(),
            source: e,
        })?;

        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if src_path.is_file() {
            fs::copy(&src_path, &dst_path).map_err(|e| TempDirError::IoError {
                path: src_path.display().to_string(),
                source: e,
            })?;
        } else if src_path.is_dir() {
            copy_directory_contents(&src_path, &dst_path)?;
        }
    }

    Ok(())
}

/// Move directory to a new location
pub fn move_directory(src: &Path, dst: &Path) -> TempDirResult<()> {
    if !src.exists() {
        return Err(TempDirError::DirectoryNotFound {
            path: src.display().to_string(),
        });
    }

    // Try rename first (most efficient)
    match fs::rename(src, dst) {
        Ok(()) => return Ok(()),
        Err(e) => {
            // If rename fails, try copy + remove
            debug!(
                src = %src.display(),
                dst = %dst.display(),
                error = %e,
                "Rename failed, falling back to copy + remove"
            );
        },
    }

    // Fallback: copy then remove
    copy_directory_contents(src, dst)?;
    remove_directory_recursive(src)?;

    Ok(())
}

// ================================================================================================
// Validation Utilities
// ================================================================================================

/// Validate directory name
pub fn validate_directory_name(name: &str) -> TempDirResult<()> {
    if name.is_empty() {
        return Err(TempDirError::ConfigurationError {
            message: "Directory name cannot be empty".to_string(),
        });
    }

    if name.len() > 255 {
        return Err(TempDirError::ConfigurationError {
            message: "Directory name too long".to_string(),
        });
    }

    // Check for invalid characters
    for c in name.chars() {
        match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => {
                return Err(TempDirError::ConfigurationError {
                    message: format!("Directory name contains invalid character: {}", c),
                });
            },
            c if c.is_control() => {
                return Err(TempDirError::ConfigurationError {
                    message: "Directory name contains control characters".to_string(),
                });
            },
            _ => {},
        }
    }

    Ok(())
}

/// Validate test ID
pub fn validate_test_id(test_id: &str) -> TempDirResult<()> {
    if test_id.is_empty() {
        return Err(TempDirError::ConfigurationError {
            message: "Test ID cannot be empty".to_string(),
        });
    }

    if test_id.len() > 128 {
        return Err(TempDirError::ConfigurationError {
            message: "Test ID too long".to_string(),
        });
    }

    // Check for valid characters (alphanumeric, underscore, hyphen)
    for c in test_id.chars() {
        if !c.is_alphanumeric() && c != '_' && c != '-' {
            return Err(TempDirError::ConfigurationError {
                message: format!("Test ID contains invalid character: {}", c),
            });
        }
    }

    Ok(())
}

// ================================================================================================
// Format and Display Utilities
// ================================================================================================

/// Format byte size in human-readable format
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    const THRESHOLD: u64 = 1024;

    if bytes == 0 {
        return "0 B".to_string();
    }

    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= THRESHOLD as f64 && unit_index < UNITS.len() - 1 {
        size /= THRESHOLD as f64;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

/// Format duration in human-readable format
pub fn format_duration(duration: std::time::Duration) -> String {
    let secs = duration.as_secs();

    if secs == 0 {
        return format!("{}ms", duration.as_millis());
    }

    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;

    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}

// ================================================================================================
// Test Utilities (for unit tests)
// ================================================================================================

#[cfg(test)]
pub mod test_utils {
    use super::*;
    use std::env;

    /// Create a temporary test directory
    pub fn create_test_directory() -> TempDirResult<std::path::PathBuf> {
        let test_dir =
            env::temp_dir().join(format!("test_temp_dir_manager_{}", uuid::Uuid::new_v4()));
        create_directory_recursive(&test_dir)?;
        Ok(test_dir)
    }

    /// Clean up test directory
    pub fn cleanup_test_directory(path: &Path) -> TempDirResult<()> {
        if path.exists() {
            remove_directory_recursive(path)?;
        }
        Ok(())
    }

    /// Create test files in a directory
    pub fn create_test_files(dir: &Path, file_count: usize, file_size: usize) -> TempDirResult<()> {
        for i in 0..file_count {
            let file_path = dir.join(format!("test_file_{}.txt", i));
            let content = "x".repeat(file_size);
            fs::write(&file_path, content).map_err(|e| TempDirError::IoError {
                path: file_path.display().to_string(),
                source: e,
            })?;
        }
        Ok(())
    }
}
