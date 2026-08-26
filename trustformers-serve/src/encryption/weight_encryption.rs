//! Model Weight Encryption using ChaCha20-Poly1305
//!
//! Pure-Rust weight encryption for secure model deployment and serving.
//!
//! Layer payloads are protected with the RFC 8439 ChaCha20-Poly1305 AEAD from
//! the RustCrypto [`chacha20poly1305`] crate: the Poly1305 tag is a real
//! message authentication code, so a modified ciphertext, a modified nonce, a
//! modified layer id or a wrong key all fail authentication. Keys are derived
//! from the password with PBKDF2-HMAC-SHA256, which — unlike a non-cryptographic
//! hash — cannot be inverted or brute-forced cheaply and does not collapse the
//! password's entropy into a single 64-bit accumulator.
//!
//! The hand-written ChaCha20 core below (quarter round, block function, keystream
//! XOR) is retained as a verified low-level building block and is covered by the
//! RFC 7539 test vectors, but it is **not** used on its own for weight
//! protection: a bare stream cipher provides no integrity.

use chacha20poly1305::aead::{AeadInOut, KeyInit, Nonce, Tag};
use chacha20poly1305::ChaCha20Poly1305;
use hmac::Hmac;
use sha2::Sha256;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error Types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by weight encryption operations.
#[derive(Debug, Clone, PartialEq)]
pub enum WeightEncryptionError {
    /// Poly1305 tag verification failed — the blob was tampered with, the layer
    /// id does not match, or the key/password is wrong.
    AuthenticationFailed,
    /// Key derivation rejected its parameters (for example a zero iteration
    /// count).
    KeyDerivationFailed(String),
    /// The AEAD implementation rejected the input (message too long).
    CipherRejectedInput(String),
    /// Ciphertext length is not a valid multiple for decryption.
    InvalidCiphertextLength(usize),
    /// Byte slice length is not a multiple of 4 and cannot be converted to f32 slice.
    InvalidF32Alignment(usize),
    /// The model layer index is out of range for the configured nonce scheme.
    LayerIndexOutOfRange(u32),
    /// The blobs slice length does not match the expected layer count.
    LayerCountMismatch { expected: usize, actual: usize },
}

impl fmt::Display for WeightEncryptionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AuthenticationFailed => write!(
                f,
                "Poly1305 authentication failed: the encrypted weights are tampered, \
                 mislabelled or were produced with a different key"
            ),
            Self::KeyDerivationFailed(msg) => write!(f, "Key derivation failed: {msg}"),
            Self::CipherRejectedInput(msg) => write!(f, "Cipher rejected input: {msg}"),
            Self::InvalidCiphertextLength(n) => {
                write!(f, "Invalid ciphertext length: {} bytes", n)
            },
            Self::InvalidF32Alignment(n) => write!(
                f,
                "Byte buffer length {} is not a multiple of 4; cannot interpret as f32",
                n
            ),
            Self::LayerIndexOutOfRange(idx) => {
                write!(
                    f,
                    "Layer index {} is out of range for nonce construction",
                    idx
                )
            },
            Self::LayerCountMismatch { expected, actual } => write!(
                f,
                "Layer count mismatch: expected {} layers, got {}",
                expected, actual
            ),
        }
    }
}

impl std::error::Error for WeightEncryptionError {}

// ─────────────────────────────────────────────────────────────────────────────
// ChaCha20 Core — pure arithmetic, no lookup tables
// ─────────────────────────────────────────────────────────────────────────────

/// The four constants that appear in every ChaCha20 initial state.
const CHACHA20_CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

/// Perform one ChaCha20 quarter-round update on four state words.
///
/// # Arguments
/// * `a`, `b`, `c`, `d` — four 32-bit words of the ChaCha20 state
///
/// # Returns
/// Updated `(a, b, c, d)` after one quarter-round.
#[inline(always)]
pub fn chacha20_quarter_round(a: u32, b: u32, c: u32, d: u32) -> (u32, u32, u32, u32) {
    let a = a.wrapping_add(b);
    let d = (d ^ a).rotate_left(16);
    let c = c.wrapping_add(d);
    let b = (b ^ c).rotate_left(12);
    let a = a.wrapping_add(b);
    let d = (d ^ a).rotate_left(8);
    let c = c.wrapping_add(d);
    let b = (b ^ c).rotate_left(7);
    (a, b, c, d)
}

/// Read a little-endian u32 from a byte slice at the given offset.
#[inline(always)]
fn read_le_u32(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
    ])
}

/// Produce one 64-byte ChaCha20 keystream block.
///
/// # Arguments
/// * `key` — 32-byte secret key
/// * `counter` — block counter (monotonically increasing)
/// * `nonce` — 12-byte nonce (must be unique per (key, plaintext) pair)
///
/// # Returns
/// 64 bytes of keystream.
pub fn chacha20_block(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u8; 64] {
    // Build the 16-word initial state.
    let mut state = [0u32; 16];

    // Words 0-3: constants
    state[0] = CHACHA20_CONSTANTS[0];
    state[1] = CHACHA20_CONSTANTS[1];
    state[2] = CHACHA20_CONSTANTS[2];
    state[3] = CHACHA20_CONSTANTS[3];

    // Words 4-11: key (8 × little-endian u32)
    for i in 0..8 {
        state[4 + i] = read_le_u32(key, i * 4);
    }

    // Word 12: block counter
    state[12] = counter;

    // Words 13-15: nonce (3 × little-endian u32)
    state[13] = read_le_u32(nonce, 0);
    state[14] = read_le_u32(nonce, 4);
    state[15] = read_le_u32(nonce, 8);

    let initial = state;

    // 20 rounds = 10 double-rounds
    for _ in 0..10 {
        // Column rounds
        (state[0], state[4], state[8], state[12]) =
            chacha20_quarter_round(state[0], state[4], state[8], state[12]);
        (state[1], state[5], state[9], state[13]) =
            chacha20_quarter_round(state[1], state[5], state[9], state[13]);
        (state[2], state[6], state[10], state[14]) =
            chacha20_quarter_round(state[2], state[6], state[10], state[14]);
        (state[3], state[7], state[11], state[15]) =
            chacha20_quarter_round(state[3], state[7], state[11], state[15]);

        // Diagonal rounds
        (state[0], state[5], state[10], state[15]) =
            chacha20_quarter_round(state[0], state[5], state[10], state[15]);
        (state[1], state[6], state[11], state[12]) =
            chacha20_quarter_round(state[1], state[6], state[11], state[12]);
        (state[2], state[7], state[8], state[13]) =
            chacha20_quarter_round(state[2], state[7], state[8], state[13]);
        (state[3], state[4], state[9], state[14]) =
            chacha20_quarter_round(state[3], state[4], state[9], state[14]);
    }

    // Add initial state to final state.
    for i in 0..16 {
        state[i] = state[i].wrapping_add(initial[i]);
    }

    // Serialize as little-endian bytes.
    let mut output = [0u8; 64];
    for i in 0..16 {
        let bytes = state[i].to_le_bytes();
        output[i * 4..i * 4 + 4].copy_from_slice(&bytes);
    }
    output
}

/// Encrypt (or decrypt) a byte slice using ChaCha20 stream cipher.
///
/// ChaCha20 is a symmetric stream cipher: `decrypt == encrypt`.
///
/// # Arguments
/// * `plaintext` — input bytes
/// * `key` — 32-byte secret key
/// * `nonce` — 12-byte nonce; **must be unique** for each (key, message) pair
/// * `initial_counter` — starting block counter (typically 0)
///
/// # Returns
/// Ciphertext with the same length as the input.
pub fn chacha20_encrypt(
    plaintext: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 12],
    initial_counter: u32,
) -> Vec<u8> {
    let mut output = Vec::with_capacity(plaintext.len());
    let mut block_idx = 0u32;
    let mut offset = 0usize;

    while offset < plaintext.len() {
        let counter = initial_counter.wrapping_add(block_idx);
        let keystream = chacha20_block(key, counter, nonce);

        let remaining = plaintext.len() - offset;
        let take = remaining.min(64);

        for i in 0..take {
            output.push(plaintext[offset + i] ^ keystream[i]);
        }

        offset += take;
        block_idx += 1;
    }

    output
}

/// Decrypt a byte slice using ChaCha20 stream cipher.
///
/// Identical to [`chacha20_encrypt`] because XOR is self-inverse.
pub fn chacha20_decrypt(
    ciphertext: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 12],
    initial_counter: u32,
) -> Vec<u8> {
    // XOR stream cipher — decrypt == encrypt
    chacha20_encrypt(ciphertext, key, nonce, initial_counter)
}

// ─────────────────────────────────────────────────────────────────────────────
// Adler-32 checksum
// ─────────────────────────────────────────────────────────────────────────────

const ADLER32_MOD: u32 = 65521;

/// Compute the Adler-32 checksum of a byte slice.
///
/// # Warning
///
/// Adler-32 is a **non-cryptographic** checksum: an attacker who changes the
/// plaintext can trivially adjust other bytes so the checksum still matches. It
/// detects accidental corruption only, and is never used for the tamper
/// detection performed by [`WeightEncryptor`] — that relies on the Poly1305 tag.
pub fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;

    for &byte in data {
        a = (a + u32::from(byte)) % ADLER32_MOD;
        b = (b + a) % ADLER32_MOD;
    }

    (b << 16) | a
}

// ─────────────────────────────────────────────────────────────────────────────
// Key Derivation (PBKDF2-HMAC-SHA256, RFC 8018)
// ─────────────────────────────────────────────────────────────────────────────

/// Default PBKDF2-HMAC-SHA256 iteration count.
///
/// Matches the OWASP Password Storage Cheat Sheet recommendation for
/// PBKDF2-HMAC-SHA256. Callers who need a different work factor set
/// [`WeightEncryptionConfig::kdf_iterations`].
pub const DEFAULT_KDF_ITERATIONS: u32 = 210_000;

/// Derive a 32-byte key from a password and a 16-byte salt using
/// PBKDF2-HMAC-SHA256 (RFC 8018).
///
/// Unlike a plain hash chain, PBKDF2 keeps the full entropy of the password in
/// the derived key and imposes `iterations` HMAC evaluations on every guess.
///
/// # Errors
///
/// Returns [`WeightEncryptionError::KeyDerivationFailed`] when `iterations` is
/// zero, which would make the KDF a single HMAC evaluation.
pub fn derive_key_from_password(
    password: &str,
    salt: &[u8; 16],
    iterations: u32,
) -> Result<[u8; 32], WeightEncryptionError> {
    if iterations == 0 {
        return Err(WeightEncryptionError::KeyDerivationFailed(
            "PBKDF2 iteration count must be greater than zero".to_string(),
        ));
    }
    let mut key = [0u8; 32];
    pbkdf2::pbkdf2::<Hmac<Sha256>>(password.as_bytes(), salt, iterations, &mut key).map_err(
        |e| WeightEncryptionError::KeyDerivationFailed(format!("PBKDF2-HMAC-SHA256: {e}")),
    )?;
    Ok(key)
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration and Data Types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the weight encryption subsystem.
#[derive(Debug, Clone)]
pub struct WeightEncryptionConfig {
    /// 16-byte salt used during key derivation.
    pub key_derivation_salt: [u8; 16],
    /// First 8 bytes of the 12-byte nonce; the last 4 bytes are the layer index.
    pub nonce_prefix: [u8; 8],
    /// When `true`, each layer receives a distinct nonce derived from its index.
    pub use_per_layer_nonce: bool,
    /// When `true`, compress plaintext before encrypting (not yet implemented).
    pub compress_before_encrypt: bool,
    /// PBKDF2-HMAC-SHA256 iteration count used to turn the password into a key.
    pub kdf_iterations: u32,
}

impl Default for WeightEncryptionConfig {
    fn default() -> Self {
        Self {
            key_derivation_salt: [
                0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
                0xcd, 0xef,
            ],
            nonce_prefix: [0x54, 0x52, 0x53, 0x54, 0x46, 0x4d, 0x52, 0x53], // "TRSTFMRS"
            use_per_layer_nonce: true,
            compress_before_encrypt: false,
            kdf_iterations: DEFAULT_KDF_ITERATIONS,
        }
    }
}

/// A self-contained encrypted representation of one model layer's weights.
#[derive(Debug, Clone)]
pub struct EncryptedWeightBlob {
    /// Raw ciphertext bytes.
    pub ciphertext: Vec<u8>,
    /// 12-byte nonce that was used during encryption.
    pub nonce: [u8; 12],
    /// Index of the layer this blob belongs to.
    pub layer_id: u32,
    /// Original plaintext size in bytes (before encryption).
    pub original_size: usize,
    /// Poly1305 authentication tag over the ciphertext and the layer id.
    ///
    /// Verified in constant time by [`WeightEncryptor::decrypt_layer`]; any
    /// modification of `ciphertext`, `nonce`, `layer_id` or the key makes
    /// decryption fail with [`WeightEncryptionError::AuthenticationFailed`].
    pub tag: [u8; 16],
}

// ─────────────────────────────────────────────────────────────────────────────
// WeightEncryptor
// ─────────────────────────────────────────────────────────────────────────────

/// Encrypts and decrypts model weight tensors using ChaCha20.
///
/// A single `WeightEncryptor` holds a derived 32-byte key and can process an
/// entire model's layer stack.
pub struct WeightEncryptor {
    /// Configuration used to build this encryptor.
    pub config: WeightEncryptionConfig,
    /// 32-byte derived key (never exposed in `Debug` output).
    key: [u8; 32],
}

impl fmt::Debug for WeightEncryptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeightEncryptor")
            .field("config", &self.config)
            .field("key", &"[redacted]")
            .finish()
    }
}

impl WeightEncryptor {
    /// Create a new `WeightEncryptor` by deriving a key from `password` and
    /// the salt stored in `config` with PBKDF2-HMAC-SHA256.
    ///
    /// # Errors
    ///
    /// Returns [`WeightEncryptionError::KeyDerivationFailed`] if
    /// `config.kdf_iterations` is zero.
    pub fn new(
        password: &str,
        config: WeightEncryptionConfig,
    ) -> Result<Self, WeightEncryptionError> {
        let key =
            derive_key_from_password(password, &config.key_derivation_salt, config.kdf_iterations)?;
        Ok(Self { config, key })
    }

    /// Create a `WeightEncryptor` from an already-derived 32-byte key.
    ///
    /// Use this when the key comes from a key-management system rather than a
    /// password, so no KDF work is repeated.
    pub fn from_key(key: [u8; 32], config: WeightEncryptionConfig) -> Self {
        Self { config, key }
    }

    /// Build the 12-byte nonce for a given layer.
    ///
    /// If `use_per_layer_nonce` is disabled every layer uses the same nonce
    /// (the prefix padded with zeros), which is weaker but still correct for
    /// single-layer models.
    fn nonce_for_layer(&self, layer_id: u32) -> [u8; 12] {
        let mut nonce = [0u8; 12];
        nonce[..8].copy_from_slice(&self.config.nonce_prefix);
        if self.config.use_per_layer_nonce {
            nonce[8..12].copy_from_slice(&layer_id.to_le_bytes());
        }
        nonce
    }

    /// Encrypt a single layer's weight slice with ChaCha20-Poly1305.
    ///
    /// The `f32` slice is reinterpreted as little-endian bytes and sealed under
    /// the layer's nonce, with the layer id authenticated as associated data so
    /// blobs cannot be swapped between layers. The resulting
    /// [`EncryptedWeightBlob`] is self-contained: all information needed for
    /// decryption is stored inside it (except the key / password).
    ///
    /// # Errors
    ///
    /// Returns [`WeightEncryptionError::CipherRejectedInput`] if the AEAD
    /// rejects the message (only possible for absurdly large layers).
    pub fn encrypt_layer(
        &self,
        weights: &[f32],
        layer_id: u32,
    ) -> Result<EncryptedWeightBlob, WeightEncryptionError> {
        // Serialize f32 values to little-endian bytes.
        let mut buffer: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
        let original_size = buffer.len();

        let nonce_bytes = self.nonce_for_layer(layer_id);
        let cipher = self.aead()?;
        let nonce = Nonce::<ChaCha20Poly1305>::from(nonce_bytes);
        let tag = cipher
            .encrypt_inout_detached(
                &nonce,
                &layer_id.to_le_bytes(),
                buffer.as_mut_slice().into(),
            )
            .map_err(|e| WeightEncryptionError::CipherRejectedInput(e.to_string()))?;

        Ok(EncryptedWeightBlob {
            ciphertext: buffer,
            nonce: nonce_bytes,
            layer_id,
            original_size,
            tag: tag.into(),
        })
    }

    /// Decrypt a single layer blob back into an `f32` weight slice.
    ///
    /// # Errors
    /// - [`WeightEncryptionError::AuthenticationFailed`] if the Poly1305 tag
    ///   does not verify (tampering, wrong layer id, or wrong key).
    /// - [`WeightEncryptionError::InvalidF32Alignment`] if byte count is not
    ///   divisible by 4.
    pub fn decrypt_layer(
        &self,
        blob: &EncryptedWeightBlob,
    ) -> Result<Vec<f32>, WeightEncryptionError> {
        let mut plaintext = blob.ciphertext.clone();
        let cipher = self.aead()?;
        let nonce = Nonce::<ChaCha20Poly1305>::from(blob.nonce);
        let tag = Tag::<ChaCha20Poly1305>::from(blob.tag);
        cipher
            .decrypt_inout_detached(
                &nonce,
                &blob.layer_id.to_le_bytes(),
                plaintext.as_mut_slice().into(),
                &tag,
            )
            .map_err(|_| WeightEncryptionError::AuthenticationFailed)?;

        // Convert bytes → f32.
        if !plaintext.len().is_multiple_of(4) {
            return Err(WeightEncryptionError::InvalidF32Alignment(plaintext.len()));
        }

        let weights = plaintext
            .chunks_exact(4)
            .map(|chunk| {
                let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32::from_le_bytes(arr)
            })
            .collect();

        Ok(weights)
    }

    /// Build the AEAD instance for this encryptor's key.
    fn aead(&self) -> Result<ChaCha20Poly1305, WeightEncryptionError> {
        ChaCha20Poly1305::new_from_slice(&self.key).map_err(|e| {
            WeightEncryptionError::KeyDerivationFailed(format!("invalid ChaCha20 key: {e}"))
        })
    }

    /// Encrypt an entire model represented as a slice of weight layers.
    ///
    /// Layer `i` receives `layer_id = i as u32`.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`Self::encrypt_layer`].
    pub fn encrypt_model(
        &self,
        layers: &[Vec<f32>],
    ) -> Result<Vec<EncryptedWeightBlob>, WeightEncryptionError> {
        layers
            .iter()
            .enumerate()
            .map(|(i, weights)| self.encrypt_layer(weights, i as u32))
            .collect()
    }

    /// Decrypt an entire model represented as a slice of encrypted blobs.
    ///
    /// # Errors
    /// Propagates any [`WeightEncryptionError`] from individual layer
    /// decryption; processing stops at the first error.
    pub fn decrypt_model(
        &self,
        blobs: &[EncryptedWeightBlob],
    ) -> Result<Vec<Vec<f32>>, WeightEncryptionError> {
        blobs.iter().map(|blob| self.decrypt_layer(blob)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. ChaCha20 quarter-round: verify a known test vector ────────────────
    #[test]
    fn test_chacha20_quarter_round_known_values() {
        // Values taken from RFC 7539 §2.1.1 test vector.
        let (a, b, c, d) = chacha20_quarter_round(0x11111111, 0x01020304, 0x9b8d6f43, 0x01234567);
        assert_eq!(a, 0xea2a92f4);
        assert_eq!(b, 0xcb1cf8ce);
        assert_eq!(c, 0x4581472e);
        assert_eq!(d, 0x5881c4bb);
    }

    // ── 2. chacha20_block produces exactly 64 bytes ───────────────────────────
    #[test]
    fn test_chacha20_block_produces_64_bytes() {
        let key = [0u8; 32];
        let nonce = [0u8; 12];
        let block = chacha20_block(&key, 0, &nonce);
        assert_eq!(block.len(), 64);
    }

    // ── 3. chacha20_block matches a known constant-key/zero-nonce vector ──────
    //     First 4 bytes of the RFC 7539 §2.3.2 test vector for key=0, nonce=0, counter=0.
    #[test]
    fn test_chacha20_block_rfc_vector() {
        let key = [0u8; 32];
        let nonce = [0u8; 12];
        let block = chacha20_block(&key, 0, &nonce);
        // First four bytes from RFC 7539 §2.3.2 test vector
        assert_eq!(block[0], 0x76);
        assert_eq!(block[1], 0xb8);
        assert_eq!(block[2], 0xe0);
        assert_eq!(block[3], 0xad);
    }

    // ── 4. Encrypt / decrypt round-trip ──────────────────────────────────────
    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = [0x42u8; 32];
        let nonce = [0x55u8; 12];
        let plaintext = b"Hello, TrustformeRS model weights!";

        let ciphertext = chacha20_encrypt(plaintext, &key, &nonce, 0);
        let recovered = chacha20_decrypt(&ciphertext, &key, &nonce, 0);

        assert_eq!(recovered, plaintext.as_ref());
    }

    // ── 5. Different keys produce different ciphertext ────────────────────────
    #[test]
    fn test_different_keys_different_ciphertext() {
        let key_a = [0xAAu8; 32];
        let key_b = [0xBBu8; 32];
        let nonce = [0u8; 12];
        let plaintext = b"same plaintext, different keys";

        let ct_a = chacha20_encrypt(plaintext, &key_a, &nonce, 0);
        let ct_b = chacha20_encrypt(plaintext, &key_b, &nonce, 0);

        assert_ne!(ct_a, ct_b);
    }

    // ── 6. Adler-32 known values ──────────────────────────────────────────────
    #[test]
    fn test_adler32_known_values() {
        // "Wikipedia" → 0x11E60398 (widely cited reference value)
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        // Empty input → 1 (per Adler-32 specification)
        assert_eq!(adler32(b""), 1);
        // Single zero byte: a=1, b=1 → (1 << 16) | 1
        assert_eq!(adler32(&[0u8]), (1u32 << 16) | 1u32);
    }

    /// Config with a small KDF work factor so the test suite stays fast; the
    /// production default is [`DEFAULT_KDF_ITERATIONS`].
    fn fast_config() -> WeightEncryptionConfig {
        WeightEncryptionConfig {
            kdf_iterations: 1_000,
            ..Default::default()
        }
    }

    fn encryptor(password: &str) -> WeightEncryptor {
        WeightEncryptor::new(password, fast_config()).expect("key derivation must succeed")
    }

    // ── 7. Key derivation is deterministic ───────────────────────────────────
    #[test]
    fn test_key_derivation_is_deterministic() {
        let salt = [0x11u8; 16];
        let key_a = derive_key_from_password("my-secret-password", &salt, 1_000).expect("kdf");
        let key_b = derive_key_from_password("my-secret-password", &salt, 1_000).expect("kdf");
        assert_eq!(key_a, key_b);
    }

    // ── 8. Different salts / passwords produce different keys ─────────────────
    #[test]
    fn test_key_derivation_different_inputs() {
        let salt1 = [0x11u8; 16];
        let salt2 = [0x22u8; 16];
        let key1 = derive_key_from_password("password", &salt1, 1_000).expect("kdf");
        let key2 = derive_key_from_password("password", &salt2, 1_000).expect("kdf");
        assert_ne!(key1, key2);

        let key3 = derive_key_from_password("other-password", &salt1, 1_000).expect("kdf");
        assert_ne!(key1, key3);
    }

    /// Regression test for the FNV-1a KDF: it funnelled every password through a
    /// single 64-bit accumulator, so the derived key never carried more than
    /// 64 bits of entropy and the iteration count changed nothing observable.
    /// PBKDF2 keeps the work factor meaningful — a different iteration count
    /// must produce a different key.
    #[test]
    fn test_iteration_count_changes_derived_key() {
        let salt = [0x33u8; 16];
        let k1 = derive_key_from_password("password", &salt, 1_000).expect("kdf");
        let k2 = derive_key_from_password("password", &salt, 2_000).expect("kdf");
        assert_ne!(k1, k2, "PBKDF2 output must depend on the iteration count");
    }

    /// PBKDF2-HMAC-SHA256 known answer for ("password", "salt", c = 2, dkLen = 32).
    #[test]
    fn test_key_derivation_known_answer() {
        // The published vector uses a 4-byte salt, so call PBKDF2 directly
        // rather than through the 16-byte-salt wrapper.
        let mut key = [0u8; 32];
        pbkdf2::pbkdf2::<Hmac<Sha256>>(b"password", b"salt", 2, &mut key).expect("pbkdf2");
        assert_eq!(
            hex::encode(key),
            "ae4d0c95af6b46d32d0adff928f06dd02a303f8ef3c251dfd6e2d85a95474c43"
        );
    }

    #[test]
    fn test_zero_iterations_is_rejected() {
        let err = derive_key_from_password("password", &[0u8; 16], 0)
            .expect_err("zero iterations must be rejected");
        assert!(matches!(err, WeightEncryptionError::KeyDerivationFailed(_)));
    }

    // ── 9. Per-layer nonce produces different nonces for different layers ──────
    #[test]
    fn test_per_layer_nonce_differs() {
        let encryptor = encryptor("pass");
        let nonce0 = encryptor.nonce_for_layer(0);
        let nonce1 = encryptor.nonce_for_layer(1);
        assert_ne!(nonce0, nonce1);
    }

    // ── 10. Weight encrypt / decrypt round-trip (f32 slice) ───────────────────
    #[test]
    fn test_weight_encrypt_decrypt_roundtrip() {
        let encryptor = encryptor("secret-model-key");

        let weights: Vec<f32> = vec![1.0, -2.5, 3.14, 0.0, f32::MAX, f32::MIN_POSITIVE];
        let blob = encryptor.encrypt_layer(&weights, 0).expect("encrypt");

        let recovered = encryptor.decrypt_layer(&blob).expect("decryption should succeed");
        assert_eq!(recovered.len(), weights.len());
        for (orig, dec) in weights.iter().zip(recovered.iter()) {
            assert_eq!(orig.to_bits(), dec.to_bits(), "f32 bit patterns must match");
        }
    }

    // ── 11. Tampering with the ciphertext is detected ─────────────────────────
    //
    // Regression test for the Adler-32 "tamper detection" claim: with a stream
    // cipher an attacker could flip plaintext bits and repair the checksum.
    // Poly1305 makes that computationally infeasible.
    #[test]
    fn test_tampered_ciphertext_is_detected() {
        let encryptor = encryptor("secret");

        let weights = vec![1.0f32, 2.0, 3.0];
        let mut blob = encryptor.encrypt_layer(&weights, 0).expect("encrypt");
        blob.ciphertext[0] ^= 0x01;

        let result = encryptor.decrypt_layer(&blob);
        assert_eq!(result, Err(WeightEncryptionError::AuthenticationFailed));
    }

    #[test]
    fn test_tampered_tag_is_detected() {
        let encryptor = encryptor("secret");
        let mut blob = encryptor.encrypt_layer(&[1.0f32, 2.0], 0).expect("encrypt");
        blob.tag[15] ^= 0x80;
        assert_eq!(
            encryptor.decrypt_layer(&blob),
            Err(WeightEncryptionError::AuthenticationFailed)
        );
    }

    #[test]
    fn test_relabelled_layer_is_detected() {
        let encryptor = encryptor("secret");
        let mut blob = encryptor.encrypt_layer(&[1.0f32, 2.0], 3).expect("encrypt");
        // The layer id is authenticated as associated data, so relabelling the
        // blob must fail even though the ciphertext bytes are untouched.
        blob.layer_id = 4;
        assert_eq!(
            encryptor.decrypt_layer(&blob),
            Err(WeightEncryptionError::AuthenticationFailed)
        );
    }

    /// A stream cipher alone lets an attacker flip any plaintext bit by flipping
    /// the corresponding ciphertext bit. With Poly1305 the forgery is rejected
    /// rather than silently decoded into different weights.
    #[test]
    fn test_bit_flip_forgery_is_rejected() {
        let encryptor = encryptor("secret");
        let weights = vec![1.0f32];
        let mut blob = encryptor.encrypt_layer(&weights, 0).expect("encrypt");
        // Flip the sign bit of the single f32 (byte 3, bit 7 little-endian).
        blob.ciphertext[3] ^= 0x80;
        assert_eq!(
            encryptor.decrypt_layer(&blob),
            Err(WeightEncryptionError::AuthenticationFailed),
            "a targeted bit flip must not silently change the decrypted weight"
        );
    }

    // ── 12. encrypt_model / decrypt_model full pipeline ───────────────────────
    #[test]
    fn test_encrypt_decrypt_model_pipeline() {
        let encryptor = encryptor("model-password");

        let model: Vec<Vec<f32>> = vec![
            vec![0.1, 0.2, 0.3],
            vec![-1.0, 0.0, 1.0, 2.0],
            vec![f32::NAN], // NaN preserved bit-for-bit
        ];

        let blobs = encryptor.encrypt_model(&model).expect("encrypt model");
        assert_eq!(blobs.len(), 3);
        assert_eq!(blobs[0].layer_id, 0);
        assert_eq!(blobs[1].layer_id, 1);
        assert_eq!(blobs[2].layer_id, 2);

        let recovered = encryptor.decrypt_model(&blobs).expect("model decryption should succeed");
        assert_eq!(recovered.len(), 3);

        assert_eq!(
            model[2][0].to_bits(),
            recovered[2][0].to_bits(),
            "NaN bit pattern must be preserved"
        );

        for (orig_layer, rec_layer) in model.iter().zip(recovered.iter()) {
            assert_eq!(orig_layer.len(), rec_layer.len());
            for (o, r) in orig_layer.iter().zip(rec_layer.iter()) {
                assert_eq!(o.to_bits(), r.to_bits());
            }
        }
    }

    // ── 13. Large weight buffer round-trip (stress) ───────────────────────────
    #[test]
    fn test_large_weight_buffer_roundtrip() {
        let encryptor = encryptor("stress-test-key");

        let weights: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.001).collect();
        let blob = encryptor.encrypt_layer(&weights, 7).expect("encrypt");
        let recovered = encryptor.decrypt_layer(&blob).expect("large buffer decryption");
        assert_eq!(weights, recovered);
    }

    // ── 14. Custom nonce_prefix changes the nonce ─────────────────────────────
    #[test]
    fn test_custom_nonce_prefix_changes_nonce() {
        let config_a = WeightEncryptionConfig {
            nonce_prefix: [0x11u8; 8],
            ..fast_config()
        };
        let config_b = WeightEncryptionConfig {
            nonce_prefix: [0x22u8; 8],
            ..fast_config()
        };
        let enc_a = WeightEncryptor::new("pass", config_a).expect("kdf");
        let enc_b = WeightEncryptor::new("pass", config_b).expect("kdf");

        assert_ne!(
            enc_a.nonce_for_layer(0),
            enc_b.nonce_for_layer(0),
            "different nonce_prefix must produce different nonces"
        );
    }

    // ── 15. use_per_layer_nonce=false: all layers share the same nonce ────────
    #[test]
    fn test_no_per_layer_nonce_same_for_all_layers() {
        let config = WeightEncryptionConfig {
            use_per_layer_nonce: false,
            ..fast_config()
        };
        let encryptor = WeightEncryptor::new("pass", config).expect("kdf");

        let nonce0 = encryptor.nonce_for_layer(0);
        let nonce1 = encryptor.nonce_for_layer(1);
        let nonce255 = encryptor.nonce_for_layer(255);

        assert_eq!(
            nonce0, nonce1,
            "all layers must share the same nonce when per-layer is off"
        );
        assert_eq!(
            nonce0, nonce255,
            "all layers must share the same nonce when per-layer is off"
        );
    }

    // ── 16. Empty weights layer produces a valid zero-element round-trip ──────
    #[test]
    fn test_empty_layer_roundtrip() {
        let encryptor = encryptor("key");
        let empty: Vec<f32> = vec![];
        let blob = encryptor.encrypt_layer(&empty, 0).expect("encrypt");

        assert_eq!(blob.original_size, 0);
        let recovered = encryptor.decrypt_layer(&blob).expect("empty layer must decrypt");
        assert!(recovered.is_empty());
    }

    // ── 17. Wrong key cannot decrypt ──────────────────────────────────────────
    #[test]
    fn test_wrong_key_fails_authentication() {
        let correct_enc = encryptor("correct-password");
        let wrong_enc = encryptor("wrong-password");

        let weights = vec![1.0f32, 2.0, 3.0, 4.0];
        let blob = correct_enc.encrypt_layer(&weights, 0).expect("encrypt");

        assert_eq!(
            wrong_enc.decrypt_layer(&blob),
            Err(WeightEncryptionError::AuthenticationFailed),
            "decrypting with the wrong key must fail authentication"
        );
    }

    // ── 18. WeightEncryptionConfig fields have expected defaults ──────────────
    #[test]
    fn test_config_default_values() {
        let config = WeightEncryptionConfig::default();
        assert!(
            config.use_per_layer_nonce,
            "per-layer nonce should be on by default"
        );
        assert!(
            !config.compress_before_encrypt,
            "compression should be off by default"
        );
        assert_ne!(
            config.key_derivation_salt, [0u8; 16],
            "salt must not be all-zero"
        );
        assert_ne!(
            config.nonce_prefix, [0u8; 8],
            "nonce prefix must not be all-zero"
        );
        assert_eq!(config.kdf_iterations, DEFAULT_KDF_ITERATIONS);
        assert!(
            config.kdf_iterations >= 100_000,
            "the default PBKDF2 work factor must be meaningful"
        );
    }

    // ── 19. Nonce construction embeds layer_id in last four bytes ─────────────
    #[test]
    fn test_nonce_contains_layer_id_in_last_four_bytes() {
        let config = WeightEncryptionConfig {
            nonce_prefix: [0xAAu8; 8],
            use_per_layer_nonce: true,
            ..fast_config()
        };
        let encryptor = WeightEncryptor::new("key", config).expect("kdf");

        let layer_id: u32 = 0xDEAD_BEEFu32;
        let nonce = encryptor.nonce_for_layer(layer_id);

        assert_eq!(
            &nonce[8..12],
            &layer_id.to_le_bytes(),
            "last 4 bytes of nonce must be layer_id in little-endian"
        );
        assert_eq!(
            &nonce[..8],
            &[0xAAu8; 8],
            "first 8 bytes must be the nonce_prefix"
        );
    }

    // ── 20. Many-layer model: each blob carries the correct layer_id ──────────
    #[test]
    fn test_many_layer_model_blob_ids() {
        let encryptor = encryptor("model-key");

        let mut lcg: u64 = 0xACE1_ACE1_ACE1_ACE1;
        let model: Vec<Vec<f32>> = (0..16u32)
            .map(|_| {
                lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let v = f32::from_bits((lcg & 0x7FFF_FFFF) as u32 | 0x3F80_0000);
                vec![v, v * 2.0, v * 3.0]
            })
            .collect();

        let blobs = encryptor.encrypt_model(&model).expect("encrypt model");
        assert_eq!(blobs.len(), 16);
        for (i, blob) in blobs.iter().enumerate() {
            assert_eq!(
                blob.layer_id, i as u32,
                "blob at index {i} must have layer_id = {i}"
            );
        }
    }

    // ── 21. decrypt_model propagates first error ───────────────────────────────
    #[test]
    fn test_decrypt_model_propagates_error() {
        let encryptor = encryptor("pass");

        let weights = vec![1.0f32, 2.0, 3.0];
        let mut blobs =
            encryptor.encrypt_model(&[weights.clone(), weights.clone()]).expect("encrypt");
        blobs[1].tag[0] ^= 0xFF;

        let result = encryptor.decrypt_model(&blobs);
        assert!(
            result.is_err(),
            "decrypt_model must propagate the authentication error"
        );
    }

    // ── 22. EncryptedWeightBlob stores exact original_size ────────────────────
    #[test]
    fn test_blob_stores_correct_original_size() {
        let encryptor = encryptor("k");
        let weights: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let blob = encryptor.encrypt_layer(&weights, 0).expect("encrypt");
        assert_eq!(blob.original_size, weights.len() * 4);
    }

    // ── 23. Detached tag keeps the ciphertext length equal to the plaintext ───
    #[test]
    fn test_ciphertext_length_equals_original_size() {
        let encryptor = encryptor("k");
        let weights: Vec<f32> = vec![0.0f32; 37]; // non-power-of-two size
        let blob = encryptor.encrypt_layer(&weights, 0).expect("encrypt");
        assert_eq!(
            blob.ciphertext.len(),
            blob.original_size,
            "the detached Poly1305 tag must not inflate the ciphertext"
        );
        assert_eq!(blob.tag.len(), 16);
    }

    // ── 24. from_key skips the KDF but produces an interoperable encryptor ────
    #[test]
    fn test_from_key_matches_password_derived_encryptor() {
        let config = fast_config();
        let key =
            derive_key_from_password("pw", &config.key_derivation_salt, config.kdf_iterations)
                .expect("kdf");
        let via_password = WeightEncryptor::new("pw", config.clone()).expect("kdf");
        let via_key = WeightEncryptor::from_key(key, config);

        let blob = via_password.encrypt_layer(&[1.0f32, 2.0], 0).expect("encrypt");
        let recovered = via_key.decrypt_layer(&blob).expect("decrypt");
        assert_eq!(recovered, vec![1.0f32, 2.0]);
    }
}
