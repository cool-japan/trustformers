//! Security configuration and policies

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

/// Security policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    /// Policy name
    pub name: String,

    /// Policy type
    pub policy_type: SecurityPolicyType,

    /// Policy rules
    pub rules: Vec<SecurityRule>,

    /// Enforcement level
    pub enforcement_level: EnforcementLevel,

    /// Policy enabled
    pub enabled: bool,
}

/// Security policy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityPolicyType {
    /// Data protection
    DataProtection,

    /// Access control
    AccessControl,

    /// Network security
    NetworkSecurity,

    /// Compliance
    Compliance,

    /// Custom policy
    Custom(String),
}

/// Security rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRule {
    /// Rule name
    pub name: String,

    /// Rule description
    pub description: String,

    /// Rule condition
    pub condition: String,

    /// Rule action
    pub action: SecurityAction,

    /// Rule severity
    pub severity: SecuritySeverity,
}

/// Security actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityAction {
    /// Allow operation
    Allow,

    /// Deny operation
    Deny,

    /// Log and allow
    LogAndAllow,

    /// Log and deny
    LogAndDeny,

    /// Encrypt data
    Encrypt,

    /// Audit operation
    Audit,

    /// Custom action
    Custom(String),
}

/// Security severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecuritySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Enforcement levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementLevel {
    /// Advisory only
    Advisory,

    /// Warning
    Warning,

    /// Blocking
    Blocking,

    /// Strict blocking
    Strict,
}

/// Compliance standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceStandard {
    /// GDPR compliance
    Gdpr,

    /// HIPAA compliance
    Hipaa,

    /// SOC2 compliance
    Soc2,

    /// ISO27001 compliance
    Iso27001,

    /// Custom compliance
    Custom(String),
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Security enabled
    pub enabled: bool,

    /// Access control configuration
    pub access_control: AccessControlConfig,

    /// API key management
    pub api_key_management: ApiKeyManagement,

    /// Encryption configuration
    pub encryption: EncryptionConfig,

    /// Key management
    pub key_management: KeyManagementConfig,

    /// Audit configuration
    pub audit: AuditConfig,

    /// Compliance configuration
    pub compliance: ComplianceConfig,
}

/// Access control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlConfig {
    /// Authorization scheme
    pub authorization_scheme: AuthorizationScheme,

    /// RBAC configuration
    pub rbac: Option<RbacConfig>,

    /// Session timeout
    pub session_timeout: Duration,

    /// Multi-factor authentication required
    pub mfa_required: bool,

    /// IP restrictions
    pub ip_restrictions: Vec<String>,
}

/// Authorization schemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthorizationScheme {
    /// No authorization
    None,

    /// API key based
    ApiKey,

    /// JWT based
    Jwt,

    /// OAuth2
    OAuth2,

    /// RBAC (Role-Based Access Control)
    Rbac,

    /// ABAC (Attribute-Based Access Control)
    Abac,

    /// Custom scheme
    Custom(String),
}

/// RBAC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbacConfig {
    /// Roles
    pub roles: Vec<Role>,

    /// Role assignments
    pub role_assignments: Vec<RoleAssignment>,

    /// Default role
    pub default_role: Option<String>,
}

/// Role definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role name
    pub name: String,

    /// Role description
    pub description: String,

    /// Permissions
    pub permissions: Vec<Permission>,

    /// Parent roles
    pub parent_roles: Vec<String>,
}

/// Permission definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    /// Permission name
    pub name: String,

    /// Resource
    pub resource: String,

    /// Actions
    pub actions: Vec<String>,

    /// Conditions
    pub conditions: Vec<String>,
}

/// Role assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleAssignment {
    /// User/principal identifier
    pub principal: String,

    /// Role name
    pub role: String,

    /// Scope
    pub scope: Option<String>,

    /// Expiration
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// API key management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyManagement {
    /// Key generation settings
    pub generation: KeyGenerationSettings,

    /// Key rotation settings
    pub rotation: KeyRotationSettings,

    /// Key validation settings
    pub validation: KeyValidationSettings,

    /// Key storage settings
    pub storage: KeyStorageSettings,
}

/// Key generation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyGenerationSettings {
    /// Key algorithm
    pub algorithm: KeyAlgorithm,

    /// Key length (bits)
    pub key_length: usize,

    /// Entropy source
    pub entropy_source: EntropySource,

    /// Key prefix
    pub key_prefix: Option<String>,

    /// Key suffix
    pub key_suffix: Option<String>,
}

/// Key algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyAlgorithm {
    /// Random bytes
    Random,

    /// HMAC-SHA256
    HmacSha256,

    /// HMAC-SHA512
    HmacSha512,

    /// Ed25519
    Ed25519,

    /// Custom algorithm
    Custom(String),
}

/// Entropy sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntropySource {
    /// System random
    SystemRandom,

    /// Hardware random
    HardwareRandom,

    /// Custom source
    Custom(String),
}

/// Key rotation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRotationSettings {
    /// Rotation enabled
    pub enabled: bool,

    /// Rotation strategy
    pub strategy: RotationStrategy,

    /// Rotation interval
    pub interval: Duration,

    /// Grace period for old keys
    pub grace_period: Duration,

    /// Maximum key age
    pub max_key_age: Duration,
}

/// Rotation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RotationStrategy {
    /// Time-based rotation
    TimeBased,

    /// Usage-based rotation
    UsageBased { max_uses: u64 },

    /// Manual rotation
    Manual,

    /// Custom strategy
    Custom(String),
}

/// Key validation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyValidationSettings {
    /// Validation enabled
    pub enabled: bool,

    /// Cache validation results
    pub cache_validation: ValidationCaching,

    /// Rate limiting
    pub rate_limiting: bool,

    /// Maximum validation attempts
    pub max_attempts: usize,

    /// Lockout duration
    pub lockout_duration: Duration,
}

/// Validation caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCaching {
    /// Cache enabled
    pub enabled: bool,

    /// Cache TTL
    pub ttl: Duration,

    /// Maximum cache size
    pub max_size: usize,
}

/// Key storage settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyStorageSettings {
    /// Storage type
    pub storage_type: KeyStorageType,

    /// Encryption at rest
    pub encryption_at_rest: bool,

    /// Backup settings
    pub backup: BackupSettings,

    /// Access logging
    pub access_logging: bool,
}

/// Key storage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyStorageType {
    /// In-memory storage
    Memory,

    /// File-based storage
    File { path: String },

    /// Database storage
    Database { connection_string: String },

    /// Redis storage
    Redis { url: String },

    /// HashiCorp Vault
    Vault { url: String, token: String },

    /// AWS KMS
    AwsKms { region: String },

    /// Custom storage
    Custom(String),
}

/// Backup settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupSettings {
    /// Backup enabled
    pub enabled: bool,

    /// Backup interval
    pub interval: Duration,

    /// Backup location
    pub location: String,

    /// Encryption enabled
    pub encryption: bool,

    /// Retention period
    pub retention_period: Duration,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Data at rest encryption
    pub data_at_rest: DataAtRestEncryption,

    /// Data in transit encryption
    pub data_in_transit: DataInTransitEncryption,
}

/// Data at rest encryption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAtRestEncryption {
    /// Encryption enabled
    pub enabled: bool,

    /// Encryption algorithm
    pub algorithm: String,

    /// Key derivation
    pub key_derivation: KeyDerivation,
}

/// Key derivation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyDerivation {
    /// Derivation function
    pub function: DerivationFunction,

    /// Salt generation
    pub salt_generation: SaltGeneration,

    /// Iteration count
    pub iterations: u32,
}

/// Derivation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DerivationFunction {
    Pbkdf2,
    Scrypt,
    Argon2,
}

/// Salt generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaltGeneration {
    /// Salt length
    pub length: usize,

    /// Per-key salt
    pub per_key: bool,
}

/// Data in transit encryption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataInTransitEncryption {
    /// TLS required
    pub tls_required: bool,

    /// Minimum TLS version
    pub min_tls_version: String,

    /// Certificate management
    pub certificate_management: CertificateManagement,
}

/// Certificate management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateManagement {
    /// Certificate source
    pub source: CertificateSource,

    /// Certificate validation
    pub validation: CertificateValidation,

    /// Certificate rotation
    pub rotation: CertificateRotation,
}

/// Certificate sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CertificateSource {
    /// Self-signed certificates
    SelfSigned,

    /// Let's Encrypt
    LetsEncrypt,

    /// External CA
    ExternalCa { ca_url: String },

    /// File-based certificates
    File { cert_path: String, key_path: String },

    /// Custom source
    Custom(String),
}

/// Certificate validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateValidation {
    /// Validation enabled
    pub enabled: bool,

    /// Validation rules
    pub rules: Vec<CertificateValidationRule>,

    /// OCSP checking
    pub ocsp_checking: bool,

    /// CRL checking
    pub crl_checking: bool,
}

/// Certificate validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateValidationRule {
    /// Rule name
    pub name: String,

    /// Validation type
    pub validation_type: CertificateValidationType,

    /// Expected value
    pub expected_value: String,

    /// Required
    pub required: bool,
}

/// Certificate validation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CertificateValidationType {
    /// Subject validation
    Subject,

    /// Subject Alternative Name
    SubjectAltName,

    /// Issuer validation
    Issuer,

    /// Key usage validation
    KeyUsage,

    /// Extended key usage
    ExtendedKeyUsage,

    /// Custom validation
    Custom(String),
}

/// Certificate rotation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateRotation {
    /// Rotation enabled
    pub enabled: bool,

    /// Rotation strategy
    pub strategy: CertificateRotationStrategy,

    /// Renewal threshold (days before expiration)
    pub renewal_threshold_days: u32,

    /// Grace period
    pub grace_period: Duration,
}

/// Certificate rotation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CertificateRotationStrategy {
    /// Automatic rotation
    Automatic,

    /// Manual rotation
    Manual,

    /// Scheduled rotation
    Scheduled { schedule: String },

    /// Custom strategy
    Custom(String),
}

/// Key management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementConfig {
    /// Key storage
    pub storage: KeyStorageConfig,

    /// Key lifecycle
    pub lifecycle: KeyLifecycleConfig,

    /// Key access control
    pub access_control: KeyAccessControl,
}

/// Key storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyStorageConfig {
    /// Storage backend
    pub backend: KeyStorageBackend,

    /// Backup configuration
    pub backup: BackupConfiguration,

    /// Encryption at rest
    pub encryption_at_rest: bool,

    /// Access logging
    pub access_logging: bool,
}

/// Key storage backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyStorageBackend {
    /// HashiCorp Vault
    Vault {
        url: String,
        token: String,
        mount_path: String,
    },

    /// AWS KMS
    AwsKms { region: String, key_spec: String },

    /// Azure Key Vault
    AzureKeyVault {
        vault_url: String,
        tenant_id: String,
    },

    /// Google Cloud KMS
    GoogleCloudKms {
        project_id: String,
        location: String,
        key_ring: String,
    },

    /// File-based storage
    File {
        path: String,
        encryption_key: Option<String>,
    },

    /// Database storage
    Database {
        connection_string: String,
        table_name: String,
    },

    /// Custom backend
    Custom(String),
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfiguration {
    /// Backup enabled
    pub enabled: bool,

    /// Backup strategy
    pub strategy: BackupStrategy,

    /// Backup retention
    pub retention: BackupRetention,

    /// Archive settings
    pub archive: ArchiveSettings,
}

/// Backup strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupStrategy {
    /// Full backup
    Full,

    /// Incremental backup
    Incremental,

    /// Differential backup
    Differential,

    /// Custom strategy
    Custom(String),
}

/// Backup retention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRetention {
    /// Daily backups to keep
    pub daily_backups: u32,

    /// Weekly backups to keep
    pub weekly_backups: u32,

    /// Monthly backups to keep
    pub monthly_backups: u32,

    /// Yearly backups to keep
    pub yearly_backups: u32,
}

/// Archive settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveSettings {
    /// Archive format
    pub format: ArchiveFormat,

    /// Compression enabled
    pub compression: bool,

    /// Encryption enabled
    pub encryption: bool,
}

/// Archive formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchiveFormat {
    Tar,
    Zip,
    SevenZip,
    Custom(String),
}

/// Key lifecycle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyLifecycleConfig {
    /// Key generation
    pub generation: KeyGenerationConfig,

    /// Key rotation
    pub rotation: KeyRotationConfig,

    /// Key retirement
    pub retirement: KeyRetirementConfig,
}

/// Key generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyGenerationConfig {
    /// Generation algorithm
    pub algorithm: KeyGenerationAlgorithm,

    /// Key size (bits)
    pub key_size: usize,

    /// Entropy requirements
    pub entropy_requirements: String,
}

/// Key generation algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyGenerationAlgorithm {
    Rsa,
    Ecdsa,
    Ed25519,
    Aes,
    Custom(String),
}

/// Key rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRotationConfig {
    /// Rotation schedule
    pub schedule: RotationSchedule,

    /// Rotation triggers
    pub triggers: Vec<RotationTrigger>,

    /// Overlap period
    pub overlap_period: Duration,
}

/// Rotation schedules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RotationSchedule {
    /// Daily rotation
    Daily,

    /// Weekly rotation
    Weekly,

    /// Monthly rotation
    Monthly,

    /// Yearly rotation
    Yearly,

    /// Custom schedule
    Custom(String),
}

/// Rotation triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RotationTrigger {
    /// Time-based trigger
    Time(Duration),

    /// Usage-based trigger
    Usage(u64),

    /// Event-based trigger
    Event(String),

    /// Manual trigger
    Manual,

    /// Custom trigger
    Custom(String),
}

/// Key retirement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRetirementConfig {
    /// Retirement policy
    pub policy: KeyRetirementPolicy,

    /// Grace period
    pub grace_period: Duration,

    /// Archive retired keys
    pub archive: bool,
}

/// Key retirement policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyRetirementPolicy {
    /// Immediate retirement
    Immediate,

    /// Graceful retirement
    Graceful(Duration),

    /// Usage-based retirement
    UsageBased(u64),

    /// Custom policy
    Custom(String),
}

/// Key access control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyAccessControl {
    /// Access policies
    pub policies: Vec<KeyAccessPolicy>,

    /// Permission model
    pub permission_model: KeyPermissionModel,

    /// Audit key access
    pub audit_access: bool,
}

/// Key access policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyAccessPolicy {
    /// Policy name
    pub name: String,

    /// Key patterns
    pub key_patterns: Vec<String>,

    /// Permissions
    pub permissions: Vec<KeyPermission>,

    /// Access conditions
    pub conditions: Vec<AccessCondition>,

    /// Policy enabled
    pub enabled: bool,
}

/// Key permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyPermission {
    /// Read permission
    Read,

    /// Write permission
    Write,

    /// Delete permission
    Delete,

    /// Use permission (for cryptographic operations)
    Use,

    /// Rotate permission
    Rotate,

    /// Export permission
    Export,

    /// Import permission
    Import,

    /// Backup permission
    Backup,

    /// Restore permission
    Restore,

    /// Custom permission
    Custom(String),
}

/// Access condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessCondition {
    /// Condition type
    pub condition_type: AccessConditionType,

    /// Operator
    pub operator: ConditionOperator,

    /// Value
    pub value: String,
}

/// Access condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessConditionType {
    /// IP address condition
    IpAddress,

    /// Time of day condition
    TimeOfDay,

    /// Day of week condition
    DayOfWeek,

    /// User agent condition
    UserAgent,

    /// Custom condition
    Custom(String),
}

/// Condition operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    Contains,
    NotContains,
    In,
    NotIn,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Key permission models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyPermissionModel {
    /// Whitelist model (explicit allow)
    Whitelist,

    /// Blacklist model (explicit deny)
    Blacklist,

    /// RBAC model
    Rbac,

    /// ABAC model
    Abac,

    /// Custom model
    Custom(String),
}

/// Audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Audit enabled
    pub enabled: bool,

    /// Audit targets
    pub targets: Vec<AuditTarget>,

    /// Storage configuration
    pub storage: AuditStorageConfig,

    /// Retention configuration
    pub retention: AuditRetentionConfig,

    /// Archive configuration
    pub archive: AuditArchiveConfig,

    /// Notification configuration
    pub notifications: AuditNotificationConfig,
}

/// Audit targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditTarget {
    /// Authentication events
    Authentication,

    /// Authorization events
    Authorization,

    /// Key operations
    KeyOperations,

    /// Configuration changes
    ConfigurationChanges,

    /// Data access
    DataAccess,

    /// Administrative actions
    AdministrativeActions,

    /// Custom target
    Custom(String),
}

/// Audit storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditStorageConfig {
    /// Storage backend
    pub backend: AuditStorageBackend,

    /// Storage format
    pub format: AuditStorageFormat,

    /// Encryption enabled
    pub encryption: bool,

    /// Integrity protection
    pub integrity_protection: bool,
}

/// Audit storage backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditStorageBackend {
    /// File-based storage
    File { path: String },

    /// Database storage
    Database { connection_string: String },

    /// Syslog
    Syslog { server: String },

    /// ElasticSearch
    ElasticSearch { url: String },

    /// Splunk
    Splunk { url: String },

    /// Custom backend
    Custom(String),
}

/// Audit storage formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditStorageFormat {
    Json,
    Xml,
    Csv,
    Syslog,
    Cef,
    Custom(String),
}

/// Audit retention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRetentionConfig {
    /// Retention policy
    pub policy: AuditRetentionPolicy,

    /// Retention period
    pub retention_period: Duration,

    /// Archive before deletion
    pub archive_before_deletion: bool,
}

/// Audit retention policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditRetentionPolicy {
    /// Time-based retention
    TimeBased,

    /// Size-based retention
    SizeBased { max_size_mb: u64 },

    /// Count-based retention
    CountBased { max_records: u64 },

    /// Custom policy
    Custom(String),
}

/// Audit archive configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditArchiveConfig {
    /// Archive enabled
    pub enabled: bool,

    /// Archive location
    pub location: String,

    /// Archive format
    pub format: ArchiveFormat,

    /// Compression enabled
    pub compression: bool,

    /// Encryption enabled
    pub encryption: bool,
}

/// Audit notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditNotificationConfig {
    /// Notifications enabled
    pub enabled: bool,

    /// Notification rules
    pub rules: Vec<AuditNotificationRule>,

    /// Alert configuration
    pub alerts: AuditAlertConfig,
}

/// Audit notification rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditNotificationRule {
    /// Rule name
    pub name: String,

    /// Trigger condition
    pub condition: String,

    /// Notification channels
    pub channels: Vec<String>,

    /// Rule enabled
    pub enabled: bool,
}

/// Audit alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditAlertConfig {
    /// Alerts enabled
    pub enabled: bool,

    /// Alert templates
    pub templates: Vec<AlertTemplate>,

    /// Escalation configuration
    pub escalation: AlertEscalationConfig,
}

/// Compliance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    /// Compliance enabled
    pub enabled: bool,

    /// Compliance frameworks
    pub frameworks: Vec<ComplianceFramework>,

    /// Reporting configuration
    pub reporting: ComplianceReportingConfig,

    /// Monitoring configuration
    pub monitoring: ComplianceMonitoringConfig,

    /// Dashboard configuration
    pub dashboard: ComplianceDashboardConfig,
}

/// Compliance framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceFramework {
    /// Framework name
    pub name: String,

    /// Framework version
    pub version: String,

    /// Requirements
    pub requirements: Vec<ComplianceRequirement>,

    /// Framework enabled
    pub enabled: bool,
}

/// Compliance requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRequirement {
    /// Requirement ID
    pub id: String,

    /// Requirement name
    pub name: String,

    /// Requirement description
    pub description: String,

    /// Compliance status
    pub status: ComplianceStatus,

    /// Evidence
    pub evidence: Vec<ComplianceEvidence>,
}

/// Compliance status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceStatus {
    Compliant,
    NonCompliant,
    PartiallyCompliant,
    NotApplicable,
    Unknown,
}

/// Compliance evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceEvidence {
    /// Evidence type
    pub evidence_type: EvidenceType,

    /// Evidence data
    pub data: String,

    /// Evidence timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Evidence source
    pub source: String,
}

/// Evidence types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Configuration evidence
    Configuration,

    /// Log evidence
    Log,

    /// Audit evidence
    Audit,

    /// Documentation evidence
    Documentation,

    /// Testing evidence
    Testing,

    /// Certification evidence
    Certification,

    /// Custom evidence
    Custom(String),
}

/// Compliance check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheck {
    /// Check name
    pub name: String,

    /// Check type
    pub check_type: ComplianceCheckType,

    /// Check schedule
    pub schedule: String,

    /// Expected result
    pub expected_result: String,

    /// Check enabled
    pub enabled: bool,
}

/// Compliance check types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceCheckType {
    /// Configuration check
    Configuration { key: String },

    /// Policy check
    Policy { policy_name: String },

    /// Audit check
    Audit { event_type: String },

    /// Security check
    Security { security_control: String },

    /// Custom check
    Custom(String),
}

/// Compliance reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReportingConfig {
    /// Report generation
    pub generation: ComplianceReportGeneration,

    /// Report distribution
    pub distribution: ComplianceReportDistribution,

    /// Report templates
    pub templates: Vec<ComplianceReportTemplate>,
}

/// Compliance report generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReportGeneration {
    /// Generation enabled
    pub enabled: bool,

    /// Generation schedule
    pub schedule: String,

    /// Report formats
    pub formats: Vec<String>,

    /// Include evidence
    pub include_evidence: bool,
}

/// Compliance report distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReportDistribution {
    /// Distribution enabled
    pub enabled: bool,

    /// Distribution channels
    pub channels: Vec<String>,

    /// Recipients
    pub recipients: Vec<String>,

    /// Encryption required
    pub encryption_required: bool,
}

/// Compliance report template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReportTemplate {
    /// Template name
    pub name: String,

    /// Template content
    pub content: String,

    /// Template variables
    pub variables: HashMap<String, String>,
}

/// Compliance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceMonitoringConfig {
    /// Monitoring enabled
    pub enabled: bool,

    /// Monitoring rules
    pub rules: Vec<ComplianceMonitoringRule>,

    /// Alert configuration
    pub alerts: ComplianceAlertConfig,
}

/// Compliance monitoring rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceMonitoringRule {
    /// Rule name
    pub name: String,

    /// Trigger condition
    pub condition: String,

    /// Actions
    pub actions: Vec<ComplianceAction>,

    /// Rule enabled
    pub enabled: bool,
}

/// Compliance actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceAction {
    /// Generate alert
    Alert { severity: String },

    /// Send notification
    Notify { channel: String },

    /// Create incident
    CreateIncident,

    /// Execute remediation
    Remediate { action: String },

    /// Generate report
    GenerateReport,

    /// Custom action
    Custom(String),
}

/// Compliance alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceAlertConfig {
    /// Alerts enabled
    pub enabled: bool,

    /// Alert templates
    pub templates: Vec<AlertTemplate>,

    /// Escalation configuration
    pub escalation: AlertEscalationConfig,
}

/// Alert template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertTemplate {
    /// Template name
    pub name: String,

    /// Subject template
    pub subject: String,

    /// Body template
    pub body: String,

    /// Severity
    pub severity: String,
}

/// Alert escalation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEscalationConfig {
    /// Escalation enabled
    pub enabled: bool,

    /// Escalation levels
    pub levels: Vec<EscalationLevel>,
}

/// Escalation level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLevel {
    /// Level number
    pub level: u32,

    /// Time threshold
    pub time_threshold: Duration,

    /// Recipients
    pub recipients: Vec<String>,

    /// Actions
    pub actions: Vec<String>,
}

/// Compliance dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceDashboardConfig {
    /// Dashboard enabled
    pub enabled: bool,

    /// Widgets
    pub widgets: Vec<ComplianceWidget>,

    /// Access control
    pub access_control: DashboardAccessControl,
}

/// Compliance widget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceWidget {
    /// Widget name
    pub name: String,

    /// Widget type
    pub widget_type: ComplianceWidgetType,

    /// Position
    pub position: WidgetPosition,

    /// Configuration
    pub config: HashMap<String, serde_json::Value>,
}

/// Compliance widget types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceWidgetType {
    /// Compliance status chart
    StatusChart,

    /// Compliance trend graph
    TrendGraph,

    /// Requirements checklist
    RequirementsChecklist,

    /// Evidence summary
    EvidenceSummary,

    /// Alert summary
    AlertSummary,

    /// Report links
    ReportLinks,

    /// Custom widget
    Custom(String),
}

/// Widget position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetPosition {
    /// Row
    pub row: u32,

    /// Column
    pub column: u32,

    /// Width (in grid units)
    pub width: u32,

    /// Height (in grid units)
    pub height: u32,
}

/// Dashboard access control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardAccessControl {
    /// Authentication required
    pub authentication_required: bool,

    /// Access restrictions
    pub restrictions: Vec<AccessRestriction>,

    /// Session timeout
    pub session_timeout: Duration,
}

/// Access restriction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessRestriction {
    /// Restriction type
    pub restriction_type: AccessRestrictionType,

    /// Allowed values
    pub allowed_values: Vec<String>,

    /// Restriction enabled
    pub enabled: bool,
}

/// Access restriction types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessRestrictionType {
    /// IP address restriction
    IpAddress,

    /// User restriction
    User,

    /// Role restriction
    Role,

    /// Time restriction
    Time,

    /// Custom restriction
    Custom(String),
}
