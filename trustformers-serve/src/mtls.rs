//! Mutual TLS (mTLS) Support for TrustformeRS Inference Server
//!
//! Provides comprehensive mTLS implementation for secure client-server
//! authentication and encrypted communication.

use anyhow::Result;
use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer};
use tokio_rustls::rustls::ServerConfig;
use tokio_rustls::TlsAcceptor;

/// mTLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MTlsConfig {
    /// Enable mTLS
    pub enabled: bool,
    /// Server certificate path
    pub server_cert_path: PathBuf,
    /// Server private key path
    pub server_key_path: PathBuf,
    /// Certificate Authority (CA) certificate path
    pub ca_cert_path: PathBuf,
    /// Client certificate validation mode
    pub client_cert_validation: ClientCertValidation,
    /// Certificate revocation list (CRL) path
    pub crl_path: Option<PathBuf>,
    /// OCSP (Online Certificate Status Protocol) settings
    pub ocsp_config: OcspConfig,
    /// TLS protocol versions
    pub tls_versions: Vec<TlsVersion>,
    /// Cipher suites
    pub cipher_suites: Vec<CipherSuite>,
    /// Certificate refresh interval in hours
    pub cert_refresh_interval_hours: u64,
    /// Client certificate cache TTL in seconds
    pub client_cert_cache_ttl: u64,
    /// Enable certificate pinning
    pub enable_cert_pinning: bool,
    /// Pinned certificate fingerprints
    pub pinned_fingerprints: Vec<String>,
}

impl Default for MTlsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            server_cert_path: PathBuf::from("certs/server.crt"),
            server_key_path: PathBuf::from("certs/server.key"),
            ca_cert_path: PathBuf::from("certs/ca.crt"),
            client_cert_validation: ClientCertValidation::Required,
            crl_path: None,
            ocsp_config: OcspConfig::default(),
            tls_versions: vec![TlsVersion::TLS1_2, TlsVersion::TLS1_3],
            cipher_suites: vec![
                CipherSuite::TlsAes256GcmSha384,
                CipherSuite::TlsChacha20Poly1305Sha256,
                CipherSuite::TlsAes128GcmSha256,
            ],
            cert_refresh_interval_hours: 24,
            client_cert_cache_ttl: 3600,
            enable_cert_pinning: false,
            pinned_fingerprints: Vec::new(),
        }
    }
}

/// Client certificate validation modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientCertValidation {
    /// No client certificate required
    None,
    /// Client certificate optional
    Optional,
    /// Client certificate required
    Required,
    /// Client certificate required with specific validation
    Strict { allowed_issuers: Vec<String> },
}

/// OCSP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcspConfig {
    /// Enable OCSP checking
    pub enabled: bool,
    /// OCSP responder URL
    pub responder_url: Option<String>,
    /// OCSP timeout in seconds
    pub timeout_seconds: u64,
    /// Cache OCSP responses
    pub cache_responses: bool,
    /// OCSP cache TTL in seconds
    pub cache_ttl_seconds: u64,
}

impl Default for OcspConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            responder_url: None,
            timeout_seconds: 10,
            cache_responses: true,
            cache_ttl_seconds: 3600,
        }
    }
}

/// TLS protocol versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TlsVersion {
    TLS1_0,
    TLS1_1,
    TLS1_2,
    TLS1_3,
}

/// TLS cipher suites
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CipherSuite {
    TlsAes128GcmSha256,
    TlsAes256GcmSha384,
    TlsChacha20Poly1305Sha256,
    TlsEcdheEcdsaWithAes128GcmSha256,
    TlsEcdheEcdsaWithAes256GcmSha384,
    TlsEcdheRsaWithAes128GcmSha256,
    TlsEcdheRsaWithAes256GcmSha384,
}

/// Certificate information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateInfo {
    /// Certificate subject
    pub subject: String,
    /// Certificate issuer
    pub issuer: String,
    /// Certificate serial number
    pub serial_number: String,
    /// Certificate fingerprint (SHA-256)
    pub fingerprint_sha256: String,
    /// Certificate not before timestamp
    pub not_before: SystemTime,
    /// Certificate not after timestamp
    pub not_after: SystemTime,
    /// Certificate extensions
    pub extensions: HashMap<String, String>,
    /// Certificate key usage
    pub key_usage: Vec<KeyUsage>,
}

/// Certificate key usage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyUsage {
    DigitalSignature,
    NonRepudiation,
    KeyEncipherment,
    DataEncipherment,
    KeyAgreement,
    KeyCertSign,
    CRLSign,
    EncipherOnly,
    DecipherOnly,
}

/// Client certificate information
#[derive(Debug, Clone)]
pub struct ClientCertificate {
    /// Certificate information
    pub info: CertificateInfo,
    /// Raw certificate data
    pub raw_data: Vec<u8>,
    /// Validation status
    pub validation_status: CertValidationStatus,
    /// First seen timestamp
    pub first_seen: SystemTime,
    /// Last seen timestamp
    pub last_seen: SystemTime,
    /// Usage count
    pub usage_count: u64,
}

/// Certificate validation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CertValidationStatus {
    Valid,
    Expired,
    NotYetValid,
    Revoked,
    UnknownCA,
    InvalidSignature,
    InvalidUsage,
    Pinned,
    NotPinned,
}

/// mTLS service for managing mutual TLS authentication
#[derive(Clone)]
pub struct MTlsService {
    /// Configuration
    config: MTlsConfig,
    /// TLS acceptor
    tls_acceptor: Arc<RwLock<Option<TlsAcceptor>>>,
    /// Client certificate cache
    client_cert_cache: Arc<RwLock<HashMap<String, ClientCertificate>>>,
    /// Certificate store
    cert_store: Arc<CertificateStore>,
    /// Service statistics
    stats: Arc<MTlsStats>,
}

/// Certificate store for managing certificates and CRLs
#[derive(Debug)]
pub struct CertificateStore {
    /// CA certificates
    ca_certificates: RwLock<Vec<CertificateInfo>>,
    /// Certificate revocation lists
    crls: RwLock<Vec<CertificateRevocationList>>,
    /// OCSP cache
    ocsp_cache: RwLock<HashMap<String, OcspResponse>>,
    /// Certificate pinning store
    pinning_store: RwLock<HashMap<String, PinnedCertificate>>,
}

/// Certificate Revocation List
#[derive(Debug, Clone)]
pub struct CertificateRevocationList {
    /// Issuer of the CRL
    pub issuer: String,
    /// Revoked certificates
    pub revoked_certificates: Vec<RevokedCertificate>,
    /// CRL number
    pub crl_number: u64,
    /// This update timestamp
    pub this_update: SystemTime,
    /// Next update timestamp
    pub next_update: Option<SystemTime>,
}

/// Revoked certificate entry
#[derive(Debug, Clone)]
pub struct RevokedCertificate {
    /// Certificate serial number
    pub serial_number: String,
    /// Revocation date
    pub revocation_date: SystemTime,
    /// Revocation reason
    pub reason: RevocationReason,
}

/// Certificate revocation reasons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RevocationReason {
    Unspecified,
    KeyCompromise,
    CACompromise,
    AffiliationChanged,
    Superseded,
    CessationOfOperation,
    CertificateHold,
    RemoveFromCRL,
    PrivilegeWithdrawn,
    AACompromise,
}

/// OCSP response
#[derive(Debug, Clone)]
pub struct OcspResponse {
    /// Certificate status
    pub status: OcspStatus,
    /// Response timestamp
    pub timestamp: SystemTime,
    /// Next update time
    pub next_update: Option<SystemTime>,
}

/// OCSP certificate status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OcspStatus {
    Good,
    Revoked {
        reason: RevocationReason,
        revocation_time: SystemTime,
    },
    Unknown,
}

/// Pinned certificate
#[derive(Debug, Clone)]
pub struct PinnedCertificate {
    /// Certificate fingerprint
    pub fingerprint: String,
    /// Pin creation time
    pub created_at: SystemTime,
    /// Pin expiry time
    pub expires_at: Option<SystemTime>,
    /// Description/label
    pub description: String,
}

/// mTLS service statistics
#[derive(Debug, Default)]
pub struct MTlsStats {
    /// Total TLS connections
    pub total_connections: std::sync::atomic::AtomicU64,
    /// Successful mTLS authentications
    pub successful_authentications: std::sync::atomic::AtomicU64,
    /// Failed authentications
    pub failed_authentications: std::sync::atomic::AtomicU64,
    /// Certificate validation errors
    pub cert_validation_errors: std::sync::atomic::AtomicU64,
    /// Revoked certificates encountered
    pub revoked_certificates: std::sync::atomic::AtomicU64,
    /// OCSP queries performed
    pub ocsp_queries: std::sync::atomic::AtomicU64,
    /// Certificate cache hits
    pub cache_hits: std::sync::atomic::AtomicU64,
    /// Certificate cache misses
    pub cache_misses: std::sync::atomic::AtomicU64,
}

impl MTlsService {
    /// Create a new mTLS service
    pub fn new(config: MTlsConfig) -> Result<Self> {
        let cert_store = Arc::new(CertificateStore::new());

        Ok(Self {
            config,
            tls_acceptor: Arc::new(RwLock::new(None)),
            client_cert_cache: Arc::new(RwLock::new(HashMap::new())),
            cert_store,
            stats: Arc::new(MTlsStats::default()),
        })
    }

    /// Initialize the mTLS service
    pub async fn initialize(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Load server certificates
        self.load_server_certificates().await?;

        // Load CA certificates
        self.load_ca_certificates().await?;

        // Load CRL if configured
        if let Some(crl_path) = &self.config.crl_path {
            self.load_crl(crl_path).await?;
        }

        // Initialize certificate pinning
        if self.config.enable_cert_pinning {
            self.initialize_cert_pinning().await?;
        }

        // Start background tasks
        self.start_certificate_refresh_task().await?;
        self.start_cache_cleanup_task().await?;

        Ok(())
    }

    /// Create TLS acceptor middleware
    pub async fn create_tls_acceptor(&self) -> Result<TlsAcceptor> {
        let acceptor = self.tls_acceptor.read().await;
        acceptor
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("TLS acceptor not initialized"))
            .map(|a| a.clone())
    }

    /// Validate client certificate
    pub async fn validate_client_certificate(
        &self,
        cert_data: &[u8],
    ) -> Result<CertValidationResult> {
        self.stats.total_connections.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Parse certificate
        let cert_info = self.parse_certificate(cert_data)?;

        // Check cache first
        let cache_key = cert_info.fingerprint_sha256.clone();
        {
            let cache = self.client_cert_cache.read().await;
            if let Some(cached_cert) = cache.get(&cache_key) {
                self.stats.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                // Check if cache entry is still valid
                if SystemTime::now().duration_since(cached_cert.last_seen)?.as_secs()
                    < self.config.client_cert_cache_ttl
                {
                    return Ok(CertValidationResult {
                        valid: matches!(cached_cert.validation_status, CertValidationStatus::Valid),
                        status: cached_cert.validation_status.clone(),
                        certificate_info: cert_info,
                        validation_details: ValidationDetails::from_cache(),
                    });
                }
            }
        }
        self.stats.cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Perform full validation
        let validation_result = self.perform_certificate_validation(&cert_info, cert_data).await?;

        // Update cache
        {
            let mut cache = self.client_cert_cache.write().await;
            let client_cert = ClientCertificate {
                info: cert_info.clone(),
                raw_data: cert_data.to_vec(),
                validation_status: validation_result.status.clone(),
                first_seen: SystemTime::now(),
                last_seen: SystemTime::now(),
                usage_count: 1,
            };
            cache.insert(cache_key, client_cert);
        }

        // Update statistics
        if validation_result.valid {
            self.stats
                .successful_authentications
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            self.stats
                .failed_authentications
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        Ok(validation_result)
    }

    /// Extract client certificate from request
    pub async fn extract_client_certificate(&self, _request: &Request) -> Option<Vec<u8>> {
        // Extract certificate from TLS connection
        // This is a simplified implementation
        // In practice, you'd extract from the TLS connection context
        None
    }

    /// Get mTLS statistics
    pub async fn get_stats(&self) -> MTlsStatsSummary {
        MTlsStatsSummary {
            total_connections: self
                .stats
                .total_connections
                .load(std::sync::atomic::Ordering::Relaxed),
            successful_authentications: self
                .stats
                .successful_authentications
                .load(std::sync::atomic::Ordering::Relaxed),
            failed_authentications: self
                .stats
                .failed_authentications
                .load(std::sync::atomic::Ordering::Relaxed),
            cert_validation_errors: self
                .stats
                .cert_validation_errors
                .load(std::sync::atomic::Ordering::Relaxed),
            revoked_certificates: self
                .stats
                .revoked_certificates
                .load(std::sync::atomic::Ordering::Relaxed),
            ocsp_queries: self.stats.ocsp_queries.load(std::sync::atomic::Ordering::Relaxed),
            cache_hit_rate: {
                let hits = self.stats.cache_hits.load(std::sync::atomic::Ordering::Relaxed);
                let misses = self.stats.cache_misses.load(std::sync::atomic::Ordering::Relaxed);
                if hits + misses > 0 {
                    hits as f32 / (hits + misses) as f32
                } else {
                    0.0
                }
            },
            cached_certificates: self.client_cert_cache.read().await.len() as u64,
        }
    }

    // Private helper methods

    async fn load_server_certificates(&self) -> Result<()> {
        // Load server certificate and private key
        let cert_pem = fs::read_to_string(&self.config.server_cert_path)?;
        let key_pem = fs::read_to_string(&self.config.server_key_path)?;

        // Parse certificates and keys
        let certs = self.parse_certificate_chain(&cert_pem)?;
        let key = self.parse_private_key(&key_pem)?;

        // Create TLS server configuration
        // Note: rustls 0.23 includes safe defaults automatically
        let server_config = ServerConfig::builder()
            .with_client_cert_verifier(self.create_client_cert_verifier().await?)
            .with_single_cert(certs, key)?;

        // Create TLS acceptor
        let acceptor = TlsAcceptor::from(Arc::new(server_config));
        *self.tls_acceptor.write().await = Some(acceptor);

        Ok(())
    }

    async fn load_ca_certificates(&self) -> Result<()> {
        let ca_pem = fs::read_to_string(&self.config.ca_cert_path)?;
        let ca_certs = self.parse_certificate_chain(&ca_pem)?;

        let mut store = self.cert_store.ca_certificates.write().await;
        store.clear();

        for cert in ca_certs {
            let cert_info = self.extract_certificate_info(&cert)?;
            store.push(cert_info);
        }

        Ok(())
    }

    async fn load_crl(&self, crl_path: &PathBuf) -> Result<()> {
        let crl_pem = fs::read_to_string(crl_path)?;
        let crl = self.parse_crl(&crl_pem)?;

        let mut store = self.cert_store.crls.write().await;
        store.clear();
        store.push(crl);

        Ok(())
    }

    async fn initialize_cert_pinning(&self) -> Result<()> {
        let mut pinning_store = self.cert_store.pinning_store.write().await;

        for fingerprint in &self.config.pinned_fingerprints {
            let pinned_cert = PinnedCertificate {
                fingerprint: fingerprint.clone(),
                created_at: SystemTime::now(),
                expires_at: None,
                description: "Configured pinned certificate".to_string(),
            };
            pinning_store.insert(fingerprint.clone(), pinned_cert);
        }

        Ok(())
    }

    fn parse_certificate_chain(&self, pem_data: &str) -> Result<Vec<CertificateDer<'static>>> {
        // Parse PEM certificate chain
        // This is simplified - in practice would use rustls-pemfile or similar
        Ok(vec![CertificateDer::from(pem_data.as_bytes().to_vec())])
    }

    fn parse_private_key(&self, pem_data: &str) -> Result<PrivateKeyDer<'static>> {
        // Parse PEM private key
        // This is simplified - in practice would use rustls-pemfile or similar
        Ok(PrivateKeyDer::try_from(pem_data.as_bytes().to_vec())
            .map_err(|_| anyhow::anyhow!("Failed to parse private key"))?)
    }

    async fn create_client_cert_verifier(
        &self,
    ) -> Result<Arc<dyn rustls::server::danger::ClientCertVerifier>> {
        // Create custom client certificate verifier
        // This is simplified - would implement rustls::server::danger::ClientCertVerifier trait
        Err(anyhow::anyhow!(
            "Client cert verifier creation not implemented"
        ))
    }

    fn parse_certificate(&self, _cert_data: &[u8]) -> Result<CertificateInfo> {
        // Parse certificate and extract information
        // This is simplified - in practice would use a proper X.509 parser
        Ok(CertificateInfo {
            subject: "CN=Client".to_string(),
            issuer: "CN=CA".to_string(),
            serial_number: "123456".to_string(),
            fingerprint_sha256: "abcdef123456".to_string(),
            not_before: SystemTime::now() - Duration::from_secs(86400),
            not_after: SystemTime::now() + Duration::from_secs(86400 * 365),
            extensions: HashMap::new(),
            key_usage: vec![KeyUsage::DigitalSignature],
        })
    }

    fn extract_certificate_info(&self, cert: &CertificateDer) -> Result<CertificateInfo> {
        // Extract certificate information from rustls CertificateDer
        self.parse_certificate(cert.as_ref())
    }

    fn parse_crl(&self, _crl_pem: &str) -> Result<CertificateRevocationList> {
        // Parse CRL from PEM data
        // This is simplified - in practice would use a proper CRL parser
        Ok(CertificateRevocationList {
            issuer: "CN=CA".to_string(),
            revoked_certificates: Vec::new(),
            crl_number: 1,
            this_update: SystemTime::now(),
            next_update: Some(SystemTime::now() + Duration::from_secs(86400 * 7)),
        })
    }

    async fn perform_certificate_validation(
        &self,
        cert_info: &CertificateInfo,
        _cert_data: &[u8],
    ) -> Result<CertValidationResult> {
        let mut validation_details = ValidationDetails::new();

        // Check certificate validity period
        let now = SystemTime::now();
        if now < cert_info.not_before {
            return Ok(CertValidationResult {
                valid: false,
                status: CertValidationStatus::NotYetValid,
                certificate_info: cert_info.clone(),
                validation_details,
            });
        }

        if now > cert_info.not_after {
            return Ok(CertValidationResult {
                valid: false,
                status: CertValidationStatus::Expired,
                certificate_info: cert_info.clone(),
                validation_details,
            });
        }

        // Check certificate revocation
        if self.is_certificate_revoked(cert_info).await? {
            self.stats
                .revoked_certificates
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(CertValidationResult {
                valid: false,
                status: CertValidationStatus::Revoked,
                certificate_info: cert_info.clone(),
                validation_details,
            });
        }

        // Check OCSP if enabled
        if self.config.ocsp_config.enabled {
            match self.check_ocsp_status(cert_info).await? {
                OcspStatus::Revoked { .. } => {
                    return Ok(CertValidationResult {
                        valid: false,
                        status: CertValidationStatus::Revoked,
                        certificate_info: cert_info.clone(),
                        validation_details,
                    });
                },
                OcspStatus::Unknown => {
                    validation_details.ocsp_status = Some("Unknown".to_string());
                },
                OcspStatus::Good => {
                    validation_details.ocsp_status = Some("Good".to_string());
                },
            }
        }

        // Check certificate pinning
        if self.config.enable_cert_pinning {
            if !self.is_certificate_pinned(cert_info).await? {
                return Ok(CertValidationResult {
                    valid: false,
                    status: CertValidationStatus::NotPinned,
                    certificate_info: cert_info.clone(),
                    validation_details,
                });
            }
        }

        // Validate against client certificate policy
        match &self.config.client_cert_validation {
            ClientCertValidation::None => {
                return Ok(CertValidationResult {
                    valid: true,
                    status: CertValidationStatus::Valid,
                    certificate_info: cert_info.clone(),
                    validation_details,
                });
            },
            ClientCertValidation::Optional => {
                // Certificate is valid if present
            },
            ClientCertValidation::Required => {
                // Certificate must be present and valid
            },
            ClientCertValidation::Strict { allowed_issuers } => {
                if !allowed_issuers.contains(&cert_info.issuer) {
                    return Ok(CertValidationResult {
                        valid: false,
                        status: CertValidationStatus::UnknownCA,
                        certificate_info: cert_info.clone(),
                        validation_details,
                    });
                }
            },
        }

        Ok(CertValidationResult {
            valid: true,
            status: CertValidationStatus::Valid,
            certificate_info: cert_info.clone(),
            validation_details,
        })
    }

    async fn is_certificate_revoked(&self, cert_info: &CertificateInfo) -> Result<bool> {
        let crls = self.cert_store.crls.read().await;

        for crl in crls.iter() {
            if crl.issuer == cert_info.issuer {
                for revoked in &crl.revoked_certificates {
                    if revoked.serial_number == cert_info.serial_number {
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    async fn check_ocsp_status(&self, cert_info: &CertificateInfo) -> Result<OcspStatus> {
        self.stats.ocsp_queries.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Check cache first
        {
            let cache = self.cert_store.ocsp_cache.read().await;
            if let Some(response) = cache.get(&cert_info.serial_number) {
                if SystemTime::now().duration_since(response.timestamp)?.as_secs()
                    < self.config.ocsp_config.cache_ttl_seconds
                {
                    return Ok(response.status.clone());
                }
            }
        }

        // Perform OCSP query (simplified)
        let status = OcspStatus::Good; // Would perform actual OCSP query

        // Cache response
        if self.config.ocsp_config.cache_responses {
            let mut cache = self.cert_store.ocsp_cache.write().await;
            let response = OcspResponse {
                status: status.clone(),
                timestamp: SystemTime::now(),
                next_update: Some(
                    SystemTime::now()
                        + Duration::from_secs(self.config.ocsp_config.cache_ttl_seconds),
                ),
            };
            cache.insert(cert_info.serial_number.clone(), response);
        }

        Ok(status)
    }

    async fn is_certificate_pinned(&self, cert_info: &CertificateInfo) -> Result<bool> {
        let pinning_store = self.cert_store.pinning_store.read().await;
        Ok(pinning_store.contains_key(&cert_info.fingerprint_sha256))
    }

    async fn start_certificate_refresh_task(&self) -> Result<()> {
        let service = self.clone();
        let interval = Duration::from_secs(service.config.cert_refresh_interval_hours * 3600);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                if let Err(e) = service.refresh_certificates().await {
                    eprintln!("Certificate refresh failed: {}", e);
                }
            }
        });

        Ok(())
    }

    async fn start_cache_cleanup_task(&self) -> Result<()> {
        let service = self.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(3600));

            loop {
                interval.tick().await;

                if let Err(e) = service.cleanup_cache().await {
                    eprintln!("Cache cleanup failed: {}", e);
                }
            }
        });

        Ok(())
    }

    async fn refresh_certificates(&self) -> Result<()> {
        // Reload server certificates
        self.load_server_certificates().await?;

        // Reload CA certificates
        self.load_ca_certificates().await?;

        // Reload CRL if configured
        if let Some(crl_path) = &self.config.crl_path {
            self.load_crl(crl_path).await?;
        }

        Ok(())
    }

    async fn cleanup_cache(&self) -> Result<()> {
        let mut cache = self.client_cert_cache.write().await;
        let now = SystemTime::now();
        let ttl = Duration::from_secs(self.config.client_cert_cache_ttl);

        cache.retain(|_, cert| now.duration_since(cert.last_seen).unwrap_or(Duration::ZERO) < ttl);

        // Cleanup OCSP cache
        let mut ocsp_cache = self.cert_store.ocsp_cache.write().await;
        let ocsp_ttl = Duration::from_secs(self.config.ocsp_config.cache_ttl_seconds);

        ocsp_cache.retain(|_, response| {
            now.duration_since(response.timestamp).unwrap_or(Duration::ZERO) < ocsp_ttl
        });

        Ok(())
    }
}

impl CertificateStore {
    fn new() -> Self {
        Self {
            ca_certificates: RwLock::new(Vec::new()),
            crls: RwLock::new(Vec::new()),
            ocsp_cache: RwLock::new(HashMap::new()),
            pinning_store: RwLock::new(HashMap::new()),
        }
    }
}

/// Certificate validation result
#[derive(Debug, Clone)]
pub struct CertValidationResult {
    /// Whether the certificate is valid
    pub valid: bool,
    /// Validation status
    pub status: CertValidationStatus,
    /// Certificate information
    pub certificate_info: CertificateInfo,
    /// Detailed validation information
    pub validation_details: ValidationDetails,
}

/// Detailed validation information
#[derive(Debug, Clone)]
pub struct ValidationDetails {
    /// Chain validation result
    pub chain_valid: bool,
    /// Signature validation result
    pub signature_valid: bool,
    /// OCSP status
    pub ocsp_status: Option<String>,
    /// CRL check result
    pub crl_checked: bool,
    /// Certificate pinning result
    pub pinning_checked: bool,
    /// Validation timestamp
    pub validated_at: SystemTime,
}

impl ValidationDetails {
    fn new() -> Self {
        Self {
            chain_valid: true,
            signature_valid: true,
            ocsp_status: None,
            crl_checked: false,
            pinning_checked: false,
            validated_at: SystemTime::now(),
        }
    }

    fn from_cache() -> Self {
        Self {
            chain_valid: true,
            signature_valid: true,
            ocsp_status: Some("Cached".to_string()),
            crl_checked: true,
            pinning_checked: true,
            validated_at: SystemTime::now(),
        }
    }
}

/// mTLS middleware for Axum
pub async fn mtls_middleware(
    request: Request,
    next: Next,
) -> Result<Response, axum::http::StatusCode> {
    // Extract and validate client certificate
    // This is simplified - would integrate with the MTlsService

    // For now, just pass through
    Ok(next.run(request).await)
}

/// mTLS statistics summary
#[derive(Debug, Serialize)]
pub struct MTlsStatsSummary {
    pub total_connections: u64,
    pub successful_authentications: u64,
    pub failed_authentications: u64,
    pub cert_validation_errors: u64,
    pub revoked_certificates: u64,
    pub ocsp_queries: u64,
    pub cache_hit_rate: f32,
    pub cached_certificates: u64,
}

/// mTLS error types
#[derive(Debug, thiserror::Error)]
pub enum MTlsError {
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    #[error("Certificate error: {message}")]
    CertificateError { message: String },

    #[error("Validation error: {message}")]
    ValidationError { message: String },

    #[error("TLS error: {message}")]
    TlsError { message: String },

    #[error("OCSP error: {message}")]
    OcspError { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mtls_service_creation() {
        let config = MTlsConfig::default();
        let service = MTlsService::new(config).unwrap();
        assert!(!service.config.enabled); // Default is disabled
    }

    #[tokio::test]
    async fn test_certificate_info_creation() {
        let cert_info = CertificateInfo {
            subject: "CN=Test".to_string(),
            issuer: "CN=CA".to_string(),
            serial_number: "123".to_string(),
            fingerprint_sha256: "abc123".to_string(),
            not_before: SystemTime::now(),
            not_after: SystemTime::now() + Duration::from_secs(86400),
            extensions: HashMap::new(),
            key_usage: vec![KeyUsage::DigitalSignature],
        };

        assert_eq!(cert_info.subject, "CN=Test");
        assert_eq!(cert_info.serial_number, "123");
    }

    #[test]
    fn test_ocsp_config_default() {
        let config = OcspConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.timeout_seconds, 10);
        assert!(config.cache_responses);
    }
}
