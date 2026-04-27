//! Core Types and Configuration Structures
//!
//! This module contains all the fundamental types, enums, and configuration structures
//! used throughout the encryption system for TrustformeRS.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Main encryption configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Enable encryption at rest
    pub enabled: bool,
    /// Default encryption algorithm
    pub default_algorithm: EncryptionAlgorithm,
    /// Key management configuration
    pub key_management: KeyManagementConfig,
    /// Database encryption configuration
    pub database_encryption: DatabaseEncryptionConfig,
    /// File system encryption configuration
    pub filesystem_encryption: FilesystemEncryptionConfig,
    /// Memory encryption configuration
    pub memory_encryption: MemoryEncryptionConfig,
    /// Key rotation configuration
    pub key_rotation: KeyRotationConfig,
    /// Compliance configuration
    pub compliance: ComplianceConfig,
    /// Performance configuration
    pub performance: PerformanceConfig,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_algorithm: EncryptionAlgorithm::AES256GCM,
            key_management: KeyManagementConfig::default(),
            database_encryption: DatabaseEncryptionConfig::default(),
            filesystem_encryption: FilesystemEncryptionConfig::default(),
            memory_encryption: MemoryEncryptionConfig::default(),
            key_rotation: KeyRotationConfig::default(),
            compliance: ComplianceConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

/// Supported encryption algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EncryptionAlgorithm {
    /// AES 256-bit in GCM mode (recommended)
    AES256GCM,
    /// AES 256-bit in CBC mode
    AES256CBC,
    /// AES 128-bit in GCM mode
    AES128GCM,
    /// ChaCha20-Poly1305
    ChaCha20Poly1305,
    /// XChaCha20-Poly1305
    XChaCha20Poly1305,
    /// Salsa20
    Salsa20,
}

impl EncryptionAlgorithm {
    /// Get the key size in bytes for this algorithm
    pub fn key_size(&self) -> usize {
        match self {
            Self::AES256GCM | Self::AES256CBC => 32, // 256 bits
            Self::AES128GCM => 16,                   // 128 bits
            Self::ChaCha20Poly1305 | Self::XChaCha20Poly1305 => 32, // 256 bits
            Self::Salsa20 => 32,                     // 256 bits
        }
    }

    /// Get the nonce/IV size in bytes for this algorithm
    pub fn nonce_size(&self) -> usize {
        match self {
            Self::AES256GCM | Self::AES128GCM => 12, // 96 bits for GCM
            Self::AES256CBC => 16,                   // 128 bits for CBC
            Self::ChaCha20Poly1305 => 12,            // 96 bits
            Self::XChaCha20Poly1305 => 24,           // 192 bits
            Self::Salsa20 => 8,                      // 64 bits
        }
    }

    /// Check if this algorithm provides authenticated encryption
    pub fn is_authenticated(&self) -> bool {
        match self {
            Self::AES256GCM | Self::AES128GCM => true,
            Self::ChaCha20Poly1305 | Self::XChaCha20Poly1305 => true,
            Self::AES256CBC | Self::Salsa20 => false,
        }
    }
}

/// Key management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementConfig {
    /// Key management system type
    pub kms_type: KeyManagementSystem,
    /// Master key configuration
    pub master_key: MasterKeyConfig,
    /// Data encryption key (DEK) configuration
    pub data_encryption_keys: DEKConfig,
    /// Key derivation configuration
    pub key_derivation: KeyDerivationConfig,
    /// Hardware security module configuration
    pub hsm: Option<HSMConfig>,
}

impl Default for KeyManagementConfig {
    fn default() -> Self {
        Self {
            kms_type: KeyManagementSystem::Internal,
            master_key: MasterKeyConfig::default(),
            data_encryption_keys: DEKConfig::default(),
            key_derivation: KeyDerivationConfig::default(),
            hsm: None,
        }
    }
}

/// Key management systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyManagementSystem {
    /// Internal key management
    Internal,
    /// AWS Key Management Service
    AWS { region: String, key_id: String },
    /// Google Cloud KMS
    GCP {
        project_id: String,
        location: String,
        key_ring: String,
        key_name: String,
    },
    /// Azure Key Vault
    Azure { vault_url: String, key_name: String },
    /// HashiCorp Vault
    Vault { url: String, mount_path: String },
    /// External KMS
    External {
        endpoint: String,
        auth_config: HashMap<String, String>,
    },
}

/// Rotation triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RotationTrigger {
    /// Time-based rotation
    TimeBasedRotation { interval: Duration },
    /// Usage-based rotation
    UsageBasedRotation { threshold: u64 },
    /// Security event triggered rotation
    SecurityEvent { event_type: String },
    /// Manual rotation
    Manual,
    /// Compliance-driven rotation
    Compliance { policy: String },
}

/// Database encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseEncryptionConfig {
    /// Enable database encryption
    pub enabled: bool,
    /// Encryption scope
    pub scope: DatabaseEncryptionScope,
    /// Column encryption configuration
    pub column_encryption: ColumnEncryptionConfig,
    /// Table encryption configuration
    pub table_encryption: TableEncryptionConfig,
    /// Sensitive data detection
    pub sensitive_data_detection: SensitiveDataDetection,
}

impl Default for DatabaseEncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scope: DatabaseEncryptionScope::ColumnLevel,
            column_encryption: ColumnEncryptionConfig::default(),
            table_encryption: TableEncryptionConfig::default(),
            sensitive_data_detection: SensitiveDataDetection::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatabaseEncryptionScope {
    /// Encrypt specific columns
    ColumnLevel,
    /// Encrypt entire tables
    TableLevel,
    /// Encrypt entire database
    DatabaseLevel,
    /// Application-level encryption
    ApplicationLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnEncryptionConfig {
    /// Enable column encryption
    pub enabled: bool,
    /// Default encryption for sensitive columns
    pub auto_encrypt_sensitive: bool,
    /// Column patterns to encrypt
    pub encryption_patterns: Vec<String>,
    /// Columns to exclude from encryption
    pub exclusion_patterns: Vec<String>,
    /// Search capability for encrypted columns
    pub enable_search: bool,
}

impl Default for ColumnEncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_encrypt_sensitive: true,
            encryption_patterns: vec![
                "*password*".to_string(),
                "*ssn*".to_string(),
                "*credit_card*".to_string(),
                "*email*".to_string(),
            ],
            exclusion_patterns: vec![],
            enable_search: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableEncryptionConfig {
    /// Tables to encrypt
    pub encrypted_tables: Vec<String>,
    /// Encryption algorithm for tables
    pub table_algorithm: Option<EncryptionAlgorithm>,
    /// Key per table
    pub key_per_table: bool,
}

impl Default for TableEncryptionConfig {
    fn default() -> Self {
        Self {
            encrypted_tables: vec![],
            table_algorithm: None, // Use default
            key_per_table: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitiveDataDetection {
    /// Enable automatic detection
    pub enabled: bool,
    /// Detection patterns
    pub patterns: Vec<SensitiveDataPattern>,
    /// Action when sensitive data is detected
    pub detection_action: DetectionAction,
    /// Confidence threshold
    pub confidence_threshold: f64,
}

impl Default for SensitiveDataDetection {
    fn default() -> Self {
        Self {
            enabled: true,
            patterns: vec![
                SensitiveDataPattern::CreditCard,
                SensitiveDataPattern::SSN,
                SensitiveDataPattern::Email,
                SensitiveDataPattern::PhoneNumber,
                SensitiveDataPattern::Custom {
                    name: "API Key".to_string(),
                    regex: r#"[Aa][Pp][Ii]_?[Kk][Ee][Yy]\s*[:=]\s*['\"]?[A-Za-z0-9]{20,}['\"]?"#
                        .to_string(),
                },
            ],
            detection_action: DetectionAction::AutoEncrypt,
            confidence_threshold: 0.8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensitiveDataPattern {
    /// Credit card numbers
    CreditCard,
    /// Social Security Numbers
    SSN,
    /// Email addresses
    Email,
    /// Phone numbers
    PhoneNumber,
    /// Custom pattern
    Custom { name: String, regex: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectionAction {
    /// Automatically encrypt
    AutoEncrypt,
    /// Log and alert
    AlertOnly,
    /// Block operation
    Block,
    /// Request user confirmation
    RequestConfirmation,
}

/// Filesystem encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemEncryptionConfig {
    pub enabled: bool,
    pub default_algorithm: EncryptionAlgorithm,
    pub encrypted_paths: Vec<String>,
}

impl Default for FilesystemEncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_algorithm: EncryptionAlgorithm::AES256GCM,
            encrypted_paths: vec![],
        }
    }
}

/// Memory encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEncryptionConfig {
    pub enabled: bool,
    pub memory_wiping: MemoryWipingConfig,
}

impl Default for MemoryEncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            memory_wiping: MemoryWipingConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryWipingConfig {
    pub enabled: bool,
    pub wiping_method: WipingMethod,
}

impl Default for MemoryWipingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            wiping_method: WipingMethod::ZeroFill,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WipingMethod {
    ZeroFill,
    RandomFill,
    MultiPass,
}

/// Key rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRotationConfig {
    pub enabled: bool,
    pub rotation_schedule: RotationSchedule,
}

impl Default for KeyRotationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rotation_schedule: RotationSchedule::Daily,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RotationSchedule {
    Daily,
    Weekly,
    Monthly,
    Custom { interval: Duration },
}

/// Compliance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    pub enabled: bool,
    pub standards: Vec<ComplianceStandard>,
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            standards: vec![ComplianceStandard::FIPS140_2],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceStandard {
    FIPS140_2,
    CommonCriteria,
    SOX,
    HIPAA,
    GDPR,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub enabled: bool,
    pub hardware_acceleration: HardwareAcceleration,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hardware_acceleration: HardwareAcceleration::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareAcceleration {
    pub enabled: bool,
    pub use_aes_ni: bool,
}

impl Default for HardwareAcceleration {
    fn default() -> Self {
        Self {
            enabled: true,
            use_aes_ni: true,
        }
    }
}

// Master key, DEK, and other supporting types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterKeyConfig {
    pub key_size: u32,
    pub generation_method: KeyGenerationMethod,
    pub backup: KeyBackupConfig,
    pub escrow: Option<KeyEscrowConfig>,
}

impl Default for MasterKeyConfig {
    fn default() -> Self {
        Self {
            key_size: 256,
            generation_method: KeyGenerationMethod::SecureRandom,
            backup: KeyBackupConfig::default(),
            escrow: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyGenerationMethod {
    SecureRandom,
    HardwareRandom,
    PasswordBased { salt: Vec<u8>, iterations: u32 },
    MultiSource { sources: Vec<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyBackupConfig {
    pub enabled: bool,
    pub storage_locations: Vec<BackupLocation>,
    pub backup_encryption: bool,
    pub backup_frequency: Duration,
}

impl Default for KeyBackupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            storage_locations: vec![BackupLocation::Local {
                path: "/secure/keys/backup".to_string(),
            }],
            backup_encryption: true,
            backup_frequency: Duration::from_secs(24 * 3600),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupLocation {
    Local {
        path: String,
    },
    S3 {
        bucket: String,
        prefix: String,
    },
    GCS {
        bucket: String,
        prefix: String,
    },
    Azure {
        container: String,
        prefix: String,
    },
    Network {
        url: String,
        credentials: HashMap<String, String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyEscrowConfig {
    pub enabled: bool,
    pub agents: Vec<EscrowAgent>,
    pub threshold: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscrowAgent {
    pub agent_id: String,
    pub public_key: Vec<u8>,
    pub contact_info: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DEKConfig {
    pub generation_method: DEKGenerationMethod,
    pub caching: DEKCachingConfig,
    pub lifecycle: DEKLifecycleConfig,
}

impl Default for DEKConfig {
    fn default() -> Self {
        Self {
            generation_method: DEKGenerationMethod::OnDemand,
            caching: DEKCachingConfig::default(),
            lifecycle: DEKLifecycleConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DEKGenerationMethod {
    OnDemand,
    PreGenerated { pool_size: u32 },
    Derived { context: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DEKCachingConfig {
    pub enabled: bool,
    pub cache_size: u32,
    pub ttl: Duration,
    pub eviction_policy: EvictionPolicy,
}

impl Default for DEKCachingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_size: 1000,
            ttl: Duration::from_secs(3600),
            eviction_policy: EvictionPolicy::LRU,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    TTL,
    Random,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DEKLifecycleConfig {
    pub max_usage_count: Option<u64>,
    pub max_lifetime: Option<Duration>,
    pub auto_rotation: bool,
    pub rotation_triggers: Vec<RotationTrigger>,
}

impl Default for DEKLifecycleConfig {
    fn default() -> Self {
        Self {
            max_usage_count: Some(1_000_000),
            max_lifetime: Some(Duration::from_secs(30 * 24 * 3600)),
            auto_rotation: true,
            rotation_triggers: vec![
                RotationTrigger::TimeBasedRotation {
                    interval: Duration::from_secs(7 * 24 * 3600),
                },
                RotationTrigger::UsageBasedRotation { threshold: 500_000 },
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyDerivationConfig {
    pub kdf: KeyDerivationFunction,
    pub salt: SaltConfig,
    pub iterations: u32,
    pub memory_cost: Option<u32>,
    pub time_cost: Option<u32>,
}

impl Default for KeyDerivationConfig {
    fn default() -> Self {
        Self {
            kdf: KeyDerivationFunction::PBKDF2SHA256,
            salt: SaltConfig::default(),
            iterations: 100_000,
            memory_cost: Some(64 * 1024),
            time_cost: Some(3),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyDerivationFunction {
    PBKDF2SHA256,
    PBKDF2SHA512,
    Argon2i,
    Argon2d,
    Argon2id,
    Scrypt,
    HKDFSHA256,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaltConfig {
    pub generation_method: SaltGenerationMethod,
    pub size: u32,
    pub storage: SaltStorage,
}

impl Default for SaltConfig {
    fn default() -> Self {
        Self {
            generation_method: SaltGenerationMethod::SecureRandom,
            size: 32,
            storage: SaltStorage::WithData,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SaltGenerationMethod {
    SecureRandom,
    Deterministic { context: String },
    UserProvided,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SaltStorage {
    WithData,
    Separate { location: String },
    Derived { context: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HSMConfig {
    pub enabled: bool,
    pub hsm_type: HSMType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HSMType {
    PKCS11 { library_path: String },
    CloudHSM { endpoint: String },
    Network { url: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- EncryptionAlgorithm tests ---

    #[test]
    fn test_aes256gcm_key_size() {
        assert_eq!(EncryptionAlgorithm::AES256GCM.key_size(), 32);
    }

    #[test]
    fn test_aes256cbc_key_size() {
        assert_eq!(EncryptionAlgorithm::AES256CBC.key_size(), 32);
    }

    #[test]
    fn test_aes128gcm_key_size() {
        assert_eq!(EncryptionAlgorithm::AES128GCM.key_size(), 16);
    }

    #[test]
    fn test_chacha20_key_size() {
        assert_eq!(EncryptionAlgorithm::ChaCha20Poly1305.key_size(), 32);
    }

    #[test]
    fn test_xchacha20_key_size() {
        assert_eq!(EncryptionAlgorithm::XChaCha20Poly1305.key_size(), 32);
    }

    #[test]
    fn test_salsa20_key_size() {
        assert_eq!(EncryptionAlgorithm::Salsa20.key_size(), 32);
    }

    #[test]
    fn test_aes256gcm_nonce_size() {
        assert_eq!(EncryptionAlgorithm::AES256GCM.nonce_size(), 12);
    }

    #[test]
    fn test_aes128gcm_nonce_size() {
        assert_eq!(EncryptionAlgorithm::AES128GCM.nonce_size(), 12);
    }

    #[test]
    fn test_aes256cbc_nonce_size() {
        assert_eq!(EncryptionAlgorithm::AES256CBC.nonce_size(), 16);
    }

    #[test]
    fn test_chacha20_nonce_size() {
        assert_eq!(EncryptionAlgorithm::ChaCha20Poly1305.nonce_size(), 12);
    }

    #[test]
    fn test_xchacha20_nonce_size() {
        assert_eq!(EncryptionAlgorithm::XChaCha20Poly1305.nonce_size(), 24);
    }

    #[test]
    fn test_salsa20_nonce_size() {
        assert_eq!(EncryptionAlgorithm::Salsa20.nonce_size(), 8);
    }

    #[test]
    fn test_aes256gcm_is_authenticated() {
        assert!(EncryptionAlgorithm::AES256GCM.is_authenticated());
    }

    #[test]
    fn test_aes128gcm_is_authenticated() {
        assert!(EncryptionAlgorithm::AES128GCM.is_authenticated());
    }

    #[test]
    fn test_chacha20_is_authenticated() {
        assert!(EncryptionAlgorithm::ChaCha20Poly1305.is_authenticated());
    }

    #[test]
    fn test_xchacha20_is_authenticated() {
        assert!(EncryptionAlgorithm::XChaCha20Poly1305.is_authenticated());
    }

    #[test]
    fn test_aes256cbc_is_not_authenticated() {
        assert!(!EncryptionAlgorithm::AES256CBC.is_authenticated());
    }

    #[test]
    fn test_salsa20_is_not_authenticated() {
        assert!(!EncryptionAlgorithm::Salsa20.is_authenticated());
    }

    // --- Default config tests ---

    #[test]
    fn test_encryption_config_default() {
        let config = EncryptionConfig::default();
        assert!(config.enabled);
        assert_eq!(config.default_algorithm, EncryptionAlgorithm::AES256GCM);
    }

    #[test]
    fn test_key_management_config_default() {
        let config = KeyManagementConfig::default();
        assert!(matches!(config.kms_type, KeyManagementSystem::Internal));
        assert!(config.hsm.is_none());
    }

    #[test]
    fn test_database_encryption_config_default() {
        let config = DatabaseEncryptionConfig::default();
        assert!(config.enabled);
        assert!(matches!(config.scope, DatabaseEncryptionScope::ColumnLevel));
    }

    #[test]
    fn test_column_encryption_config_default() {
        let config = ColumnEncryptionConfig::default();
        assert!(config.enabled);
        assert!(config.auto_encrypt_sensitive);
        assert!(!config.encryption_patterns.is_empty());
        assert!(config.exclusion_patterns.is_empty());
    }

    #[test]
    fn test_table_encryption_config_default() {
        let config = TableEncryptionConfig::default();
        assert!(config.encrypted_tables.is_empty());
        assert!(config.table_algorithm.is_none());
        assert!(config.key_per_table);
    }

    #[test]
    fn test_sensitive_data_detection_default() {
        let config = SensitiveDataDetection::default();
        assert!(config.enabled);
        assert_eq!(config.patterns.len(), 5);
        assert!((config.confidence_threshold - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_filesystem_encryption_config_default() {
        let config = FilesystemEncryptionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.default_algorithm, EncryptionAlgorithm::AES256GCM);
    }

    #[test]
    fn test_memory_encryption_config_default() {
        let config = MemoryEncryptionConfig::default();
        assert!(!config.enabled);
    }

    #[test]
    fn test_memory_wiping_config_default() {
        let config = MemoryWipingConfig::default();
        assert!(config.enabled);
        assert!(matches!(config.wiping_method, WipingMethod::ZeroFill));
    }

    #[test]
    fn test_key_rotation_config_default() {
        let config = KeyRotationConfig::default();
        assert!(config.enabled);
        assert!(matches!(config.rotation_schedule, RotationSchedule::Daily));
    }

    #[test]
    fn test_compliance_config_default() {
        let config = ComplianceConfig::default();
        assert!(config.enabled);
        assert_eq!(config.standards.len(), 1);
    }

    #[test]
    fn test_performance_config_default() {
        let config = PerformanceConfig::default();
        assert!(config.enabled);
    }

    #[test]
    fn test_hardware_acceleration_default() {
        let accel = HardwareAcceleration::default();
        assert!(accel.enabled);
        assert!(accel.use_aes_ni);
    }

    #[test]
    fn test_master_key_config_default() {
        let config = MasterKeyConfig::default();
        assert_eq!(config.key_size, 256);
        assert!(matches!(
            config.generation_method,
            KeyGenerationMethod::SecureRandom
        ));
        assert!(config.escrow.is_none());
    }

    #[test]
    fn test_key_backup_config_default() {
        let config = KeyBackupConfig::default();
        assert!(config.enabled);
        assert!(config.backup_encryption);
        assert_eq!(config.storage_locations.len(), 1);
    }

    #[test]
    fn test_dek_config_default() {
        let config = DEKConfig::default();
        assert!(matches!(
            config.generation_method,
            DEKGenerationMethod::OnDemand
        ));
    }

    #[test]
    fn test_dek_caching_config_default() {
        let config = DEKCachingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.cache_size, 1000);
        assert!(matches!(config.eviction_policy, EvictionPolicy::LRU));
    }

    #[test]
    fn test_dek_lifecycle_config_default() {
        let config = DEKLifecycleConfig::default();
        assert!(config.auto_rotation);
        assert_eq!(config.max_usage_count, Some(1_000_000));
        assert_eq!(config.rotation_triggers.len(), 2);
    }

    #[test]
    fn test_key_derivation_config_default() {
        let config = KeyDerivationConfig::default();
        assert!(matches!(config.kdf, KeyDerivationFunction::PBKDF2SHA256));
        assert_eq!(config.iterations, 100_000);
    }

    #[test]
    fn test_salt_config_default() {
        let config = SaltConfig::default();
        assert_eq!(config.size, 32);
        assert!(matches!(
            config.generation_method,
            SaltGenerationMethod::SecureRandom
        ));
        assert!(matches!(config.storage, SaltStorage::WithData));
    }
}
