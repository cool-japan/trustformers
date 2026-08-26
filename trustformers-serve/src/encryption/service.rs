//! Core Encryption Service Implementation

use super::cipher;
use super::errors::*;
use super::types::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Main encryption service
#[derive(Debug)]
pub struct EncryptionService {
    pub config: EncryptionConfig,
    key_store: Arc<RwLock<HashMap<String, EncryptionKey>>>,
    // 0.2.1: a `dek_cache: Arc<Mutex<HashMap<String, DataEncryptionKey>>>`
    // field lived here. This service has no envelope-encryption path at all --
    // `DataEncryptionKey` is constructed nowhere outside a unit test -- so the
    // cache was created empty and never read or written. It is deleted rather
    // than kept as a cache that can never hit. `DataEncryptionKey` stays: it is
    // the public shape an envelope-encryption implementation would use.
    stats: Arc<EncryptionStats>,
}

#[derive(Debug)]
pub struct EncryptionKey {
    pub key_id: String,
    pub key_material: Vec<u8>,
    pub algorithm: EncryptionAlgorithm,
    pub status: KeyStatus,
    pub created_at: SystemTime,
    pub expires_at: Option<SystemTime>,
    pub usage_count: AtomicU64,
    pub metadata: HashMap<String, String>,
}

impl Clone for EncryptionKey {
    fn clone(&self) -> Self {
        Self {
            key_id: self.key_id.clone(),
            key_material: self.key_material.clone(),
            algorithm: self.algorithm.clone(),
            status: self.status.clone(),
            created_at: self.created_at,
            expires_at: self.expires_at,
            usage_count: AtomicU64::new(self.usage_count.load(Ordering::Relaxed)),
            metadata: self.metadata.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyStatus {
    Active,
    Pending,
    Deprecated,
    Revoked,
    Expired,
}

#[derive(Debug)]
pub struct DataEncryptionKey {
    pub dek_id: String,
    pub master_key_id: String,
    pub encrypted_key_material: Vec<u8>,
    pub plaintext_key_material: Option<Vec<u8>>,
    pub algorithm: EncryptionAlgorithm,
    pub created_at: SystemTime,
    pub last_used_at: SystemTime,
    pub usage_count: AtomicU64,
}

impl Clone for DataEncryptionKey {
    fn clone(&self) -> Self {
        Self {
            dek_id: self.dek_id.clone(),
            master_key_id: self.master_key_id.clone(),
            encrypted_key_material: self.encrypted_key_material.clone(),
            plaintext_key_material: self.plaintext_key_material.clone(),
            algorithm: self.algorithm.clone(),
            created_at: self.created_at,
            last_used_at: self.last_used_at,
            usage_count: AtomicU64::new(self.usage_count.load(Ordering::Relaxed)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EncryptedData {
    pub ciphertext: Vec<u8>,
    pub iv: Vec<u8>,
    pub tag: Option<Vec<u8>>,
    pub key_id: String,
    pub algorithm: EncryptionAlgorithm,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct DecryptedData {
    pub plaintext: Vec<u8>,
    pub key_id: String,
    pub algorithm: EncryptionAlgorithm,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug)]
pub struct EncryptionStats {
    pub total_encryptions: AtomicU64,
    pub total_decryptions: AtomicU64,
    pub total_key_operations: AtomicU64,
    pub failed_operations: AtomicU64,
    pub avg_encryption_time_us: AtomicU64,
    pub avg_decryption_time_us: AtomicU64,
    pub total_bytes_encrypted: AtomicU64,
    pub total_bytes_decrypted: AtomicU64,
}

impl EncryptionService {
    pub fn new(config: EncryptionConfig) -> EncryptionResult<Self> {
        super::validate_encryption_config(&config)?;

        let service = Self {
            config,
            key_store: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(EncryptionStats::new()),
        };

        Ok(service)
    }

    pub async fn encrypt(
        &self,
        data: &[u8],
        key_id: Option<&str>,
    ) -> EncryptionResult<EncryptedData> {
        let start_time = std::time::Instant::now();

        let key = match key_id {
            Some(id) => self.get_key(id).await?,
            None => self.get_default_key().await?,
        };

        let iv = self.generate_iv(&key.algorithm)?;
        let (ciphertext, tag) = self.encrypt_with_key(data, &key, &iv).await?;

        self.update_encryption_stats(data.len(), start_time.elapsed());

        Ok(EncryptedData {
            ciphertext,
            iv: iv.to_vec(),
            tag,
            key_id: key.key_id,
            algorithm: key.algorithm,
            metadata: HashMap::new(),
        })
    }

    pub async fn decrypt(
        &self,
        ciphertext: &[u8],
        iv: &[u8],
        tag: Option<&[u8]>,
        key_id: &str,
    ) -> EncryptionResult<DecryptedData> {
        let start_time = std::time::Instant::now();

        let key = self.get_key(key_id).await?;
        let plaintext = self.decrypt_with_key(ciphertext, &key, iv, tag).await?;

        self.update_decryption_stats(plaintext.len(), start_time.elapsed());

        Ok(DecryptedData {
            plaintext,
            key_id: key.key_id,
            algorithm: key.algorithm,
            metadata: HashMap::new(),
        })
    }

    pub async fn generate_key(
        &self,
        algorithm: Option<EncryptionAlgorithm>,
    ) -> EncryptionResult<EncryptionKey> {
        let algorithm = algorithm.unwrap_or(self.config.default_algorithm.clone());
        let key_id = Uuid::new_v4().to_string();

        let key_material = self.generate_key_material(&algorithm)?;

        let key = EncryptionKey {
            key_id: key_id.clone(),
            key_material,
            algorithm,
            status: KeyStatus::Active,
            created_at: SystemTime::now(),
            expires_at: None,
            usage_count: AtomicU64::new(0),
            metadata: HashMap::new(),
        };

        {
            let mut store = self.key_store.write().await;
            store.insert(key_id, key.clone());
        }

        self.stats.total_key_operations.fetch_add(1, Ordering::Relaxed);
        Ok(key)
    }

    /// Derive a key from input keying material using HKDF-SHA256 (RFC 5869)
    /// and register it in the key store.
    ///
    /// This is the supported way to obtain a reproducible key from a shared
    /// secret; it is *not* a password KDF — use
    /// [`Self::derive_key_from_password`] for human-chosen secrets.
    ///
    /// # Errors
    ///
    /// Returns [`EncryptionError::KeyDerivationFailed`] if HKDF rejects the
    /// requested output length.
    pub async fn derive_key(
        &self,
        input_key_material: &[u8],
        salt: &[u8],
        info: &[u8],
        algorithm: Option<EncryptionAlgorithm>,
    ) -> EncryptionResult<EncryptionKey> {
        let algorithm = algorithm.unwrap_or(self.config.default_algorithm.clone());
        let key_material =
            cipher::hkdf_sha256(input_key_material, salt, info, algorithm.key_size())?;
        self.store_key(key_material, algorithm).await
    }

    /// Derive a key from a password using PBKDF2-HMAC-SHA256 and register it in
    /// the key store.
    ///
    /// # Errors
    ///
    /// Returns [`EncryptionError::KeyDerivationFailed`] if `iterations` is zero.
    pub async fn derive_key_from_password(
        &self,
        password: &str,
        salt: &[u8],
        iterations: u32,
        algorithm: Option<EncryptionAlgorithm>,
    ) -> EncryptionResult<EncryptionKey> {
        let algorithm = algorithm.unwrap_or(self.config.default_algorithm.clone());
        let key_material =
            cipher::pbkdf2_sha256(password.as_bytes(), salt, iterations, algorithm.key_size())?;
        self.store_key(key_material, algorithm).await
    }

    /// Compare two secrets in constant time.
    ///
    /// Exposed so callers verifying derived keys or MACs do not reach for `==`,
    /// which short-circuits on the first differing byte.
    pub fn verify_secret(expected: &[u8], actual: &[u8]) -> bool {
        cipher::constant_time_eq(expected, actual)
    }

    async fn store_key(
        &self,
        key_material: Vec<u8>,
        algorithm: EncryptionAlgorithm,
    ) -> EncryptionResult<EncryptionKey> {
        let key_id = Uuid::new_v4().to_string();
        let key = EncryptionKey {
            key_id: key_id.clone(),
            key_material,
            algorithm,
            status: KeyStatus::Active,
            created_at: SystemTime::now(),
            expires_at: None,
            usage_count: AtomicU64::new(0),
            metadata: HashMap::new(),
        };
        {
            let mut store = self.key_store.write().await;
            store.insert(key_id, key.clone());
        }
        self.stats.total_key_operations.fetch_add(1, Ordering::Relaxed);
        Ok(key)
    }

    pub fn get_stats(&self) -> EncryptionStats {
        EncryptionStats {
            total_encryptions: AtomicU64::new(self.stats.total_encryptions.load(Ordering::Relaxed)),
            total_decryptions: AtomicU64::new(self.stats.total_decryptions.load(Ordering::Relaxed)),
            total_key_operations: AtomicU64::new(
                self.stats.total_key_operations.load(Ordering::Relaxed),
            ),
            failed_operations: AtomicU64::new(self.stats.failed_operations.load(Ordering::Relaxed)),
            avg_encryption_time_us: AtomicU64::new(
                self.stats.avg_encryption_time_us.load(Ordering::Relaxed),
            ),
            avg_decryption_time_us: AtomicU64::new(
                self.stats.avg_decryption_time_us.load(Ordering::Relaxed),
            ),
            total_bytes_encrypted: AtomicU64::new(
                self.stats.total_bytes_encrypted.load(Ordering::Relaxed),
            ),
            total_bytes_decrypted: AtomicU64::new(
                self.stats.total_bytes_decrypted.load(Ordering::Relaxed),
            ),
        }
    }

    // Private helper methods
    async fn get_key(&self, key_id: &str) -> EncryptionResult<EncryptionKey> {
        let store = self.key_store.read().await;
        store.get(key_id).cloned().ok_or_else(|| EncryptionError::key_not_found(key_id))
    }

    async fn get_default_key(&self) -> EncryptionResult<EncryptionKey> {
        self.generate_key(None).await
    }

    /// Draw a fresh nonce/IV of the algorithm's required size from the OS CSPRNG.
    ///
    /// GCM and ChaCha20-Poly1305 lose all confidentiality guarantees if a nonce
    /// repeats under the same key, so this must never be derived from a clock
    /// reading or any other low-entropy source.
    fn generate_iv(&self, algorithm: &EncryptionAlgorithm) -> EncryptionResult<Vec<u8>> {
        cipher::random_bytes(algorithm.nonce_size())
    }

    /// Draw fresh key material of the algorithm's key size from the OS CSPRNG.
    fn generate_key_material(&self, algorithm: &EncryptionAlgorithm) -> EncryptionResult<Vec<u8>> {
        cipher::random_bytes(algorithm.key_size())
    }

    /// The additional authenticated data bound to every ciphertext.
    ///
    /// Binding the key id means a ciphertext produced under one key cannot be
    /// replayed as if it had been produced under another, even if an attacker
    /// controls the stored key id.
    fn associated_data(key_id: &str) -> Vec<u8> {
        key_id.as_bytes().to_vec()
    }

    async fn encrypt_with_key(
        &self,
        data: &[u8],
        key: &EncryptionKey,
        iv: &[u8],
    ) -> EncryptionResult<(Vec<u8>, Option<Vec<u8>>)> {
        let aad = Self::associated_data(&key.key_id);
        let result = cipher::seal(&key.algorithm, &key.key_material, iv, &aad, data);
        if result.is_err() {
            self.stats.failed_operations.fetch_add(1, Ordering::Relaxed);
        }
        let (ciphertext, tag) = result?;
        key.usage_count.fetch_add(1, Ordering::Relaxed);
        Ok((ciphertext, tag))
    }

    async fn decrypt_with_key(
        &self,
        ciphertext: &[u8],
        key: &EncryptionKey,
        iv: &[u8],
        tag: Option<&[u8]>,
    ) -> EncryptionResult<Vec<u8>> {
        let aad = Self::associated_data(&key.key_id);
        let result = cipher::open(&key.algorithm, &key.key_material, iv, &aad, ciphertext, tag);
        if result.is_err() {
            self.stats.failed_operations.fetch_add(1, Ordering::Relaxed);
        }
        let plaintext = result?;
        key.usage_count.fetch_add(1, Ordering::Relaxed);
        Ok(plaintext)
    }

    fn update_encryption_stats(&self, bytes_encrypted: usize, duration: Duration) {
        self.stats.total_encryptions.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_bytes_encrypted
            .fetch_add(bytes_encrypted as u64, Ordering::Relaxed);
        self.stats
            .avg_encryption_time_us
            .store(duration.as_micros() as u64, Ordering::Relaxed);
    }

    fn update_decryption_stats(&self, bytes_decrypted: usize, duration: Duration) {
        self.stats.total_decryptions.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_bytes_decrypted
            .fetch_add(bytes_decrypted as u64, Ordering::Relaxed);
        self.stats
            .avg_decryption_time_us
            .store(duration.as_micros() as u64, Ordering::Relaxed);
    }
}

impl EncryptionStats {
    fn new() -> Self {
        Self {
            total_encryptions: AtomicU64::new(0),
            total_decryptions: AtomicU64::new(0),
            total_key_operations: AtomicU64::new(0),
            failed_operations: AtomicU64::new(0),
            avg_encryption_time_us: AtomicU64::new(0),
            avg_decryption_time_us: AtomicU64::new(0),
            total_bytes_encrypted: AtomicU64::new(0),
            total_bytes_decrypted: AtomicU64::new(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    /// Simple LCG for deterministic pseudo-random bytes
    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }
        fn next_byte(&mut self) -> u8 {
            (self.next() & 0xFF) as u8
        }
        fn fill_bytes(&mut self, buf: &mut [u8]) {
            for b in buf.iter_mut() {
                *b = self.next_byte();
            }
        }
    }

    fn make_service() -> EncryptionService {
        let config = EncryptionConfig::default();
        EncryptionService::new(config).expect("service creation should succeed")
    }

    fn make_service_with_algo(algo: EncryptionAlgorithm) -> EncryptionService {
        let mut config = EncryptionConfig::default();
        config.default_algorithm = algo;
        EncryptionService::new(config).expect("service creation should succeed")
    }

    #[tokio::test]
    async fn test_service_creation_succeeds() {
        let svc = make_service();
        assert!(svc.config.enabled);
    }

    #[tokio::test]
    async fn test_generate_key_returns_active_key() {
        let svc = make_service();
        let key = svc.generate_key(None).await.expect("key generation should succeed");
        assert_eq!(key.status, KeyStatus::Active);
        assert!(!key.key_id.is_empty());
    }

    #[tokio::test]
    async fn test_generate_key_with_specific_algorithm() {
        let svc = make_service();
        let key = svc
            .generate_key(Some(EncryptionAlgorithm::ChaCha20Poly1305))
            .await
            .expect("key generation should succeed");
        assert_eq!(key.algorithm, EncryptionAlgorithm::ChaCha20Poly1305);
    }

    #[tokio::test]
    async fn test_key_material_has_correct_length_aes256() {
        let svc = make_service();
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES256GCM)).await.expect("key gen");
        assert_eq!(
            key.key_material.len(),
            EncryptionAlgorithm::AES256GCM.key_size()
        );
    }

    #[tokio::test]
    async fn test_key_material_has_correct_length_aes128() {
        let svc = make_service();
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES128GCM)).await.expect("key gen");
        assert_eq!(
            key.key_material.len(),
            EncryptionAlgorithm::AES128GCM.key_size()
        );
    }

    #[tokio::test]
    async fn test_encrypt_returns_correct_length() {
        let svc = make_service();
        let plaintext = b"Hello, world! This is a test message.";
        let result = svc.encrypt(plaintext, None).await.expect("encrypt should succeed");
        assert_eq!(result.ciphertext.len(), plaintext.len());
    }

    #[tokio::test]
    async fn test_encrypt_produces_different_output_from_input() {
        let svc = make_service();
        let plaintext = b"sensitive data that should be transformed";
        let result = svc.encrypt(plaintext, None).await.expect("encrypt should succeed");
        assert_ne!(
            result.ciphertext.as_slice(),
            plaintext.as_slice(),
            "encryption must transform data"
        );
    }

    /// Regression test for the XOR "demo" cipher: a repeating-key XOR leaks the
    /// plaintext structure, so two identical plaintext blocks produce two
    /// identical ciphertext blocks whenever the block offset is a multiple of
    /// the key length. A real AEAD does not.
    #[tokio::test]
    async fn test_repeating_plaintext_does_not_produce_repeating_ciphertext() {
        let svc = make_service();
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES256GCM)).await.expect("key");
        // 32-byte key => the old XOR cipher repeated its keystream every 32 bytes
        // (and the IV every 12), so lcm(32, 12) = 96 bytes apart the keystream
        // repeated exactly.
        let plaintext = vec![0x41u8; 192];
        let encrypted = svc.encrypt(&plaintext, Some(&key.key_id)).await.expect("encrypt");
        assert_ne!(
            &encrypted.ciphertext[0..96],
            &encrypted.ciphertext[96..192],
            "ciphertext must not repeat for repeating plaintext"
        );
    }

    /// Regression test: the previous implementation derived keys from a
    /// `DefaultHasher` over the current nanosecond timestamp, so two keys
    /// generated back to back shared most of their bytes.
    #[tokio::test]
    async fn test_generated_keys_are_independent() {
        let svc = make_service();
        let a = svc.generate_key(Some(EncryptionAlgorithm::AES256GCM)).await.expect("key a");
        let b = svc.generate_key(Some(EncryptionAlgorithm::AES256GCM)).await.expect("key b");
        assert_ne!(a.key_material, b.key_material);
        let shared =
            a.key_material.iter().zip(b.key_material.iter()).filter(|(x, y)| x == y).count();
        assert!(
            shared < 16,
            "two independent 32-byte keys shared {shared} bytes; key generation is not random"
        );
    }

    /// Regression test: nonces must never repeat under the same key.
    #[tokio::test]
    async fn test_ivs_are_unique_across_encryptions() {
        let svc = make_service();
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES256GCM)).await.expect("key");
        let mut seen = std::collections::HashSet::new();
        for _ in 0..64 {
            let out = svc.encrypt(b"same plaintext", Some(&key.key_id)).await.expect("encrypt");
            assert!(
                seen.insert(out.iv.clone()),
                "IV reuse detected: {:?}",
                out.iv
            );
        }
    }

    /// Regression test for the fabricated tag: the old implementation computed
    /// `tag[i] = ciphertext[i] ^ key[i]` and never verified it, so any tampered
    /// ciphertext decrypted "successfully".
    #[tokio::test]
    async fn test_tampered_ciphertext_fails_authentication() {
        let svc = make_service();
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES256GCM)).await.expect("key");
        let encrypted = svc.encrypt(b"transfer 10 USD", Some(&key.key_id)).await.expect("encrypt");
        let mut tampered = encrypted.ciphertext.clone();
        tampered[0] ^= 0x01;
        let err = svc
            .decrypt(
                &tampered,
                &encrypted.iv,
                encrypted.tag.as_deref(),
                &encrypted.key_id,
            )
            .await
            .expect_err("tampered ciphertext must not decrypt");
        assert!(matches!(err, EncryptionError::AuthenticationFailed));
    }

    #[tokio::test]
    async fn test_tampered_tag_fails_authentication() {
        let svc = make_service();
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES256GCM)).await.expect("key");
        let encrypted = svc.encrypt(b"transfer 10 USD", Some(&key.key_id)).await.expect("encrypt");
        let mut tag = encrypted.tag.clone().expect("authenticated algorithm must emit a tag");
        tag[0] ^= 0x01;
        let err = svc
            .decrypt(
                &encrypted.ciphertext,
                &encrypted.iv,
                Some(&tag),
                &encrypted.key_id,
            )
            .await
            .expect_err("tampered tag must not verify");
        assert!(matches!(err, EncryptionError::AuthenticationFailed));
    }

    #[tokio::test]
    async fn test_tampered_iv_fails_authentication() {
        let svc = make_service();
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES256GCM)).await.expect("key");
        let encrypted = svc.encrypt(b"transfer 10 USD", Some(&key.key_id)).await.expect("encrypt");
        let mut iv = encrypted.iv.clone();
        iv[0] ^= 0x01;
        let err = svc
            .decrypt(
                &encrypted.ciphertext,
                &iv,
                encrypted.tag.as_deref(),
                &encrypted.key_id,
            )
            .await
            .expect_err("tampered IV must not verify");
        assert!(matches!(err, EncryptionError::AuthenticationFailed));
    }

    #[tokio::test]
    async fn test_authenticated_roundtrip_for_every_aead_algorithm() {
        for algorithm in [
            EncryptionAlgorithm::AES256GCM,
            EncryptionAlgorithm::AES128GCM,
            EncryptionAlgorithm::ChaCha20Poly1305,
            EncryptionAlgorithm::XChaCha20Poly1305,
        ] {
            let svc = make_service_with_algo(algorithm.clone());
            let key = svc.generate_key(Some(algorithm.clone())).await.expect("key");
            let plaintext = b"authenticated payload";
            let encrypted = svc.encrypt(plaintext, Some(&key.key_id)).await.expect("encrypt");
            assert_eq!(encrypted.ciphertext.len(), plaintext.len());
            let decrypted = svc
                .decrypt(
                    &encrypted.ciphertext,
                    &encrypted.iv,
                    encrypted.tag.as_deref(),
                    &encrypted.key_id,
                )
                .await
                .expect("decrypt");
            assert_eq!(decrypted.plaintext, plaintext.to_vec(), "{algorithm:?}");
        }
    }

    #[tokio::test]
    async fn test_derive_key_is_reproducible_and_usable() {
        let svc = make_service();
        let a = svc
            .derive_key(b"shared-secret", b"salt", b"context", None)
            .await
            .expect("derive a");
        let b = svc
            .derive_key(b"shared-secret", b"salt", b"context", None)
            .await
            .expect("derive b");
        assert_eq!(
            a.key_material, b.key_material,
            "HKDF must be deterministic for identical inputs"
        );
        let c = svc
            .derive_key(b"shared-secret", b"other-salt", b"context", None)
            .await
            .expect("derive c");
        assert_ne!(
            a.key_material, c.key_material,
            "salt must change the output"
        );

        let encrypted = svc.encrypt(b"derived-key payload", Some(&a.key_id)).await.expect("enc");
        let decrypted = svc
            .decrypt(
                &encrypted.ciphertext,
                &encrypted.iv,
                encrypted.tag.as_deref(),
                &encrypted.key_id,
            )
            .await
            .expect("dec");
        assert_eq!(decrypted.plaintext, b"derived-key payload".to_vec());
    }

    #[tokio::test]
    async fn test_derive_key_from_password_rejects_zero_iterations() {
        let svc = make_service();
        let err = svc
            .derive_key_from_password("hunter2", b"salt", 0, None)
            .await
            .expect_err("zero iterations must be rejected");
        assert!(matches!(err, EncryptionError::KeyDerivationFailed { .. }));
    }

    #[tokio::test]
    async fn test_derive_key_from_password_is_reproducible() {
        let svc = make_service();
        let a = svc
            .derive_key_from_password("hunter2", b"salt", 1000, None)
            .await
            .expect("derive a");
        let b = svc
            .derive_key_from_password("hunter2", b"salt", 1000, None)
            .await
            .expect("derive b");
        assert_eq!(a.key_material, b.key_material);
        assert!(EncryptionService::verify_secret(
            &a.key_material,
            &b.key_material
        ));
        let c = svc
            .derive_key_from_password("hunter3", b"salt", 1000, None)
            .await
            .expect("derive c");
        assert!(!EncryptionService::verify_secret(
            &a.key_material,
            &c.key_material
        ));
    }

    #[tokio::test]
    async fn test_ciphertext_is_bound_to_its_key_id() {
        let svc = make_service();
        let key_a = svc.generate_key(Some(EncryptionAlgorithm::AES256GCM)).await.expect("key a");
        let encrypted = svc.encrypt(b"bound payload", Some(&key_a.key_id)).await.expect("encrypt");

        // Register a second key that holds *the same* key material under a
        // different id; the AAD binding must still make decryption fail.
        let cloned = svc
            .store_key(key_a.key_material.clone(), EncryptionAlgorithm::AES256GCM)
            .await
            .expect("store cloned key");
        let err = svc
            .decrypt(
                &encrypted.ciphertext,
                &encrypted.iv,
                encrypted.tag.as_deref(),
                &cloned.key_id,
            )
            .await
            .expect_err("key-id rebinding must fail authentication");
        assert!(matches!(err, EncryptionError::AuthenticationFailed));
    }

    #[tokio::test]
    async fn test_encrypt_with_explicit_key_id() {
        let svc = make_service();
        let key = svc.generate_key(None).await.expect("key gen");
        let plaintext = b"test data for encryption";
        let result = svc
            .encrypt(plaintext, Some(&key.key_id))
            .await
            .expect("encrypt with explicit key should succeed");
        assert_eq!(result.key_id, key.key_id);
    }

    #[tokio::test]
    async fn test_encrypt_then_decrypt_roundtrip() {
        let svc = make_service_with_algo(EncryptionAlgorithm::AES256CBC);
        let plaintext = b"roundtrip test data";
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES256CBC)).await.expect("key");
        let encrypted = svc.encrypt(plaintext, Some(&key.key_id)).await.expect("encrypt");
        let decrypted = svc
            .decrypt(
                &encrypted.ciphertext,
                &encrypted.iv,
                encrypted.tag.as_deref(),
                &encrypted.key_id,
            )
            .await
            .expect("decrypt");
        assert_eq!(decrypted.plaintext, plaintext.to_vec());
    }

    #[tokio::test]
    async fn test_decrypt_missing_tag_on_authenticated_algo_fails() {
        let svc = make_service();
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES256GCM)).await.expect("key");
        let plaintext = b"test";
        let encrypted = svc.encrypt(plaintext, Some(&key.key_id)).await.expect("encrypt");
        // Pass None as tag to trigger auth failure
        let result = svc.decrypt(&encrypted.ciphertext, &encrypted.iv, None, &key.key_id).await;
        assert!(
            result.is_err(),
            "decryption without tag for authenticated algorithm must fail"
        );
    }

    #[tokio::test]
    async fn test_decrypt_with_unknown_key_id_fails() {
        let svc = make_service();
        let plaintext = b"test data";
        let result = svc.decrypt(plaintext, &[0u8; 12], None, "nonexistent-key-id").await;
        assert!(result.is_err(), "decryption with unknown key id must fail");
    }

    #[tokio::test]
    async fn test_stats_track_encryptions() {
        let svc = make_service();
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES256CBC)).await.expect("key");
        for _ in 0..3 {
            svc.encrypt(b"data", Some(&key.key_id)).await.expect("encrypt");
        }
        let stats = svc.get_stats();
        assert_eq!(stats.total_encryptions.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn test_stats_track_bytes_encrypted() {
        let svc = make_service();
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES256CBC)).await.expect("key");
        let data = b"twelve bytes";
        svc.encrypt(data, Some(&key.key_id)).await.expect("encrypt");
        let stats = svc.get_stats();
        assert_eq!(
            stats.total_bytes_encrypted.load(Ordering::Relaxed),
            data.len() as u64
        );
    }

    #[tokio::test]
    async fn test_stats_track_decryptions() {
        let svc = make_service_with_algo(EncryptionAlgorithm::AES256CBC);
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES256CBC)).await.expect("key");
        let encrypted = svc.encrypt(b"hello world", Some(&key.key_id)).await.expect("encrypt");
        svc.decrypt(
            &encrypted.ciphertext,
            &encrypted.iv,
            encrypted.tag.as_deref(),
            &encrypted.key_id,
        )
        .await
        .expect("decrypt");
        let stats = svc.get_stats();
        assert_eq!(stats.total_decryptions.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_stats_initial_values_are_zero() {
        let svc = make_service();
        let stats = svc.get_stats();
        assert_eq!(stats.total_encryptions.load(Ordering::Relaxed), 0);
        assert_eq!(stats.total_decryptions.load(Ordering::Relaxed), 0);
        assert_eq!(stats.failed_operations.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_encrypt_empty_data() {
        let svc = make_service_with_algo(EncryptionAlgorithm::AES256CBC);
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES256CBC)).await.expect("key");
        let result = svc.encrypt(b"", Some(&key.key_id)).await.expect("encrypt empty");
        // Real AES-256-CBC pads with PKCS#7, so an empty message becomes exactly
        // one full block of padding — it is never a zero-length ciphertext.
        assert_eq!(result.ciphertext.len(), 16);
        let decrypted = svc
            .decrypt(&result.ciphertext, &result.iv, None, &result.key_id)
            .await
            .expect("decrypt empty");
        assert!(decrypted.plaintext.is_empty());
    }

    #[tokio::test]
    async fn test_encrypt_empty_data_authenticated() {
        let svc = make_service();
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES256GCM)).await.expect("key");
        let result = svc.encrypt(b"", Some(&key.key_id)).await.expect("encrypt empty");
        assert_eq!(result.ciphertext.len(), 0);
        assert_eq!(
            result.tag.as_ref().map(|t| t.len()),
            Some(16),
            "GCM must still authenticate an empty message"
        );
    }

    #[tokio::test]
    async fn test_key_status_active_on_creation() {
        let svc = make_service();
        let key = svc.generate_key(None).await.expect("key gen");
        assert_eq!(key.status, KeyStatus::Active);
    }

    #[tokio::test]
    async fn test_key_has_created_at_timestamp() {
        let svc = make_service();
        let before = SystemTime::now();
        let key = svc.generate_key(None).await.expect("key gen");
        let after = SystemTime::now();
        assert!(key.created_at >= before);
        assert!(key.created_at <= after);
    }

    #[tokio::test]
    async fn test_encryption_key_clone() {
        let svc = make_service();
        let key = svc.generate_key(None).await.expect("key gen");
        let cloned = key.clone();
        assert_eq!(key.key_id, cloned.key_id);
        assert_eq!(key.key_material, cloned.key_material);
    }

    #[tokio::test]
    async fn test_encrypt_with_aes128_algorithm() {
        let svc = make_service();
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES128GCM)).await.expect("key");
        let result = svc.encrypt(b"test data", Some(&key.key_id)).await.expect("encrypt");
        assert_eq!(result.algorithm, EncryptionAlgorithm::AES128GCM);
    }

    #[tokio::test]
    async fn test_authenticated_algorithm_produces_tag() {
        let svc = make_service();
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES256GCM)).await.expect("key");
        let result = svc.encrypt(b"auth test", Some(&key.key_id)).await.expect("encrypt");
        assert!(
            result.tag.is_some(),
            "authenticated algorithm should produce a tag"
        );
    }

    #[tokio::test]
    async fn test_non_authenticated_algorithm_no_tag() {
        let svc = make_service();
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES256CBC)).await.expect("key");
        let result = svc.encrypt(b"no-auth test", Some(&key.key_id)).await.expect("encrypt");
        assert!(
            result.tag.is_none(),
            "non-authenticated algorithm should not produce a tag"
        );
    }

    #[tokio::test]
    async fn test_multiple_keys_stored_independently() {
        let svc = make_service();
        let key1 = svc.generate_key(Some(EncryptionAlgorithm::AES256GCM)).await.expect("key1");
        let key2 = svc.generate_key(Some(EncryptionAlgorithm::AES256CBC)).await.expect("key2");
        assert_ne!(key1.key_id, key2.key_id);
        assert_ne!(key1.algorithm, key2.algorithm);
    }

    #[tokio::test]
    async fn test_stats_key_operations_tracked() {
        let svc = make_service();
        svc.generate_key(None).await.expect("key1");
        svc.generate_key(None).await.expect("key2");
        let stats = svc.get_stats();
        assert!(stats.total_key_operations.load(Ordering::Relaxed) >= 2);
    }

    #[tokio::test]
    async fn test_large_data_encryption() {
        let svc = make_service_with_algo(EncryptionAlgorithm::AES256CBC);
        let key = svc.generate_key(Some(EncryptionAlgorithm::AES256CBC)).await.expect("key");
        let mut lcg = Lcg::new(99887766);
        let mut large_data = vec![0u8; 4096];
        lcg.fill_bytes(&mut large_data);
        let result = svc.encrypt(&large_data, Some(&key.key_id)).await.expect("encrypt large");
        // 4096 bytes is an exact multiple of the 16-byte block size, so PKCS#7
        // appends one whole extra block.
        assert_eq!(result.ciphertext.len(), large_data.len() + 16);
        let decrypted = svc
            .decrypt(&result.ciphertext, &result.iv, None, &result.key_id)
            .await
            .expect("decrypt large");
        assert_eq!(decrypted.plaintext, large_data);
    }

    #[tokio::test]
    async fn test_key_id_format_is_uuid_like() {
        let svc = make_service();
        let key = svc.generate_key(None).await.expect("key gen");
        // UUID v4 has 36 chars with hyphens
        assert_eq!(key.key_id.len(), 36);
        let parts: Vec<&str> = key.key_id.split('-').collect();
        assert_eq!(parts.len(), 5);
    }

    #[tokio::test]
    async fn test_data_encryption_key_clone() {
        let dek = DataEncryptionKey {
            dek_id: "test-dek-id".to_string(),
            master_key_id: "master-key-id".to_string(),
            encrypted_key_material: vec![1, 2, 3, 4],
            plaintext_key_material: Some(vec![5, 6, 7, 8]),
            algorithm: EncryptionAlgorithm::AES256GCM,
            created_at: SystemTime::now(),
            last_used_at: SystemTime::now(),
            usage_count: AtomicU64::new(42),
        };
        let cloned = dek.clone();
        assert_eq!(cloned.dek_id, dek.dek_id);
        assert_eq!(cloned.master_key_id, dek.master_key_id);
        assert_eq!(cloned.usage_count.load(Ordering::Relaxed), 42);
    }

    #[test]
    fn test_key_status_equality() {
        assert_eq!(KeyStatus::Active, KeyStatus::Active);
        assert_ne!(KeyStatus::Active, KeyStatus::Deprecated);
        assert_ne!(KeyStatus::Revoked, KeyStatus::Expired);
    }

    #[tokio::test]
    async fn test_encrypt_creates_non_empty_iv() {
        let svc = make_service();
        let result = svc.encrypt(b"iv test data", None).await.expect("encrypt");
        assert!(!result.iv.is_empty(), "IV should not be empty");
    }
}
