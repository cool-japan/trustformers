//! Advanced Security Features for Mobile AI
//!
//! Real, pure-Rust cryptography for mobile AI: additively-homomorphic
//! encryption, secret sharing, zero-knowledge proofs and post-quantum
//! primitives. Every engine here computes the algorithm it is named after; the
//! placeholder byte-tricks that previously stood in for them have been removed
//! and are covered by regression tests.
//!
//! # What is actually implemented
//!
//! | Engine | Real algorithm | Module |
//! |---|---|---|
//! | [`HomomorphicEncryptionEngine`] | Paillier (additively homomorphic) | [`paillier`] |
//! | [`SecureMultipartyEngine`] | Shamir secret sharing over GF(2^8) | [`shamir`] |
//! | [`ZeroKnowledgeProofEngine`] | Schnorr sigma protocol + Fiat-Shamir | [`zkp`] |
//! | [`QuantumResistantEngine`] | ML-KEM-768, ML-DSA-65, SLH-DSA-SHAKE-128f | [`pqc`] |
//!
//! # What is deliberately *not* implemented
//!
//! These return a structured
//! [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation)
//! error naming what *is* available, rather than a placeholder that looks like
//! it works:
//!
//! - **Fully homomorphic encryption** (BGV / BFV / CKKS / TFHE). Paillier gives
//!   ciphertext addition and ciphertext×plaintext multiplication; ciphertext ×
//!   ciphertext needs an FHE scheme this crate does not ship.
//! - **Classic McEliece** and **Falcon**: no vetted pure-Rust implementation
//!   exists on crates.io.
//! - **Circuit proof systems** (Groth16 / PLONK / STARKs / Bulletproofs): the
//!   sigma protocol here proves knowledge of a discrete log bound to a context
//!   string, not arbitrary circuit satisfiability.
//! - **Garbled circuits, BGW, GMW**: only secret sharing is implemented.
//!
//! # Security caveats
//!
//! The post-quantum primitives delegate to RustCrypto crates (`ml-kem`,
//! `ml-dsa`, `slh-dsa`), which state that they have not been independently
//! audited. The Paillier and Schnorr implementations in this crate are built on
//! `num-bigint`, whose `modpow` is not constant-time, so they are not hardened
//! against a local timing attacker. These limitations are documented rather
//! than hidden.

pub mod paillier;
pub mod pqc;
pub mod shamir;
pub mod zkp;

#[cfg(test)]
pub(crate) mod test_rng;

use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::{invalid_input, tensor_op_error, Result};
use trustformers_core::Tensor;

/// Advanced security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedSecurityConfig {
    /// Enable homomorphic encryption for private inference
    pub homomorphic_encryption: HomomorphicConfig,
    /// Secure multi-party computation settings
    pub secure_multiparty: SecureMultipartyConfig,
    /// Zero-knowledge proof configuration
    pub zero_knowledge_proofs: ZKProofConfig,
    /// Quantum-resistant cryptography settings
    pub quantum_resistant: QuantumResistantConfig,
}

/// Homomorphic encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomomorphicConfig {
    /// Enable homomorphic encryption
    pub enabled: bool,
    /// Encryption scheme to use
    pub scheme: HomomorphicScheme,
    /// Security level (key size)
    pub security_level: SecurityLevel,
    /// Optimization settings
    pub optimization: EncryptionOptimization,
}

/// Homomorphic encryption schemes.
///
/// Only [`HomomorphicScheme::Paillier`] is implemented. The remaining variants
/// name fully-homomorphic lattice schemes that this crate does not ship;
/// selecting one makes the engine return a structured
/// [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation)
/// error instead of pretending to encrypt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HomomorphicScheme {
    /// Paillier: additively homomorphic. **This is the implemented scheme.**
    Paillier,
    /// Brakerski-Gentry-Vaikuntanathan (BGV) scheme. Not implemented.
    BGV,
    /// Brakerski/Fan-Vercauteren (BFV) scheme. Not implemented.
    BFV,
    /// Cheon-Kim-Kim-Song (CKKS) scheme for approximate computation. Not implemented.
    CKKS,
    /// Torus Fully Homomorphic Encryption. Not implemented.
    TFHE,
}

/// Security levels for encryption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// 128-bit security (fastest)
    Bit128,
    /// 192-bit security (balanced)
    Bit192,
    /// 256-bit security (most secure)
    Bit256,
}

/// Encryption optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionOptimization {
    /// Use batching for efficiency
    pub enable_batching: bool,
    /// Use bootstrapping for depth optimization
    pub enable_bootstrapping: bool,
    /// Relinearization threshold
    pub relinearization_threshold: usize,
    /// Memory vs computation tradeoff
    pub memory_optimization: bool,
}

/// Secure multi-party computation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureMultipartyConfig {
    /// Enable secure multi-party computation
    pub enabled: bool,
    /// Number of parties
    pub num_parties: usize,
    /// Threshold for secret sharing
    pub threshold: usize,
    /// MPC protocol to use
    pub protocol: MPCProtocol,
    /// Communication settings
    pub communication: MPCCommunication,
}

/// Multi-party computation protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MPCProtocol {
    /// Shamir's Secret Sharing
    ShamirSecretSharing,
    /// Garbled Circuits
    GarbledCircuits,
    /// BGW Protocol
    BGW,
    /// GMW Protocol
    GMW,
}

/// MPC communication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MPCCommunication {
    /// Use secure channels
    pub secure_channels: bool,
    /// Timeout for operations (seconds)
    pub timeout_seconds: u64,
    /// Maximum message size
    pub max_message_size: usize,
    /// Compression settings
    pub enable_compression: bool,
}

/// Zero-knowledge proof configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZKProofConfig {
    /// Enable zero-knowledge proofs
    pub enabled: bool,
    /// Proof system to use
    pub proof_system: ZKProofSystem,
    /// Verification settings
    pub verification: ZKVerificationConfig,
}

/// Zero-knowledge proof systems.
///
/// Only [`ZKProofSystem::SchnorrSigma`] is implemented. The remaining variants
/// name general-purpose circuit proof systems that this crate does not ship;
/// selecting one makes the engine return a structured
/// [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation)
/// error instead of emitting a proof that anything would verify.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZKProofSystem {
    /// Schnorr sigma protocol with Fiat-Shamir. **This is the implemented system.**
    SchnorrSigma,
    /// zk-SNARKs (Zero-Knowledge Succinct Non-Interactive Arguments of Knowledge). Not implemented.
    ZkSNARKs,
    /// zk-STARKs (Zero-Knowledge Scalable Transparent Arguments of Knowledge). Not implemented.
    ZkSTARKs,
    /// Bulletproofs. Not implemented.
    Bulletproofs,
    /// Plonk. Not implemented.
    Plonk,
}

/// Zero-knowledge verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZKVerificationConfig {
    /// Enable batch verification
    pub batch_verification: bool,
    /// Verification timeout (seconds)
    pub timeout_seconds: u64,
    /// Cache verification results
    pub cache_results: bool,
    /// Maximum proof size
    pub max_proof_size: usize,
}

/// Quantum-resistant cryptography configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResistantConfig {
    /// Enable quantum-resistant algorithms
    pub enabled: bool,
    /// Primary encryption algorithm
    pub encryption_algorithm: QuantumResistantAlgorithm,
    /// Digital signature algorithm
    pub signature_algorithm: QuantumResistantSignature,
    /// Key exchange mechanism
    pub key_exchange: QuantumResistantKeyExchange,
}

/// Quantum-resistant encryption algorithms.
///
/// Only [`QuantumResistantAlgorithm::MlKem768`] is implemented; the rest have
/// no vetted pure-Rust implementation and are reported as unsupported.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantumResistantAlgorithm {
    /// FIPS 203 ML-KEM-768 (formerly CRYSTALS-Kyber). **Implemented.**
    MlKem768,
    /// Code-based encryption (Classic McEliece). Not implemented.
    ClassicMcEliece,
    /// Multivariate encryption. Not implemented.
    Multivariate,
    /// Hash-based encryption. Not implemented.
    HashBased,
}

/// Quantum-resistant digital signatures.
///
/// [`QuantumResistantSignature::MlDsa65`] and
/// [`QuantumResistantSignature::SlhDsaShake128f`] are implemented; Falcon has no
/// vetted pure-Rust implementation and is reported as unsupported.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantumResistantSignature {
    /// FIPS 204 ML-DSA-65 (formerly CRYSTALS-Dilithium). **Implemented.**
    MlDsa65,
    /// Falcon. Not implemented.
    Falcon,
    /// FIPS 205 SLH-DSA-SHAKE-128f (formerly SPHINCS+). **Implemented.**
    SlhDsaShake128f,
}

/// Quantum-resistant key exchange.
///
/// Only [`QuantumResistantKeyExchange::MlKem768`] is implemented. SIKE is
/// additionally *cryptographically broken* (Castryck-Decru, 2022) and is
/// rejected outright rather than merely unimplemented.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantumResistantKeyExchange {
    /// FIPS 203 ML-KEM-768 KEM. **Implemented.**
    MlKem768,
    /// SIKE (Supersingular Isogeny Key Encapsulation). Broken; always rejected.
    SIKE,
    /// NTRU. Not implemented.
    NTRU,
}

/// Additively-homomorphic encryption engine, backed by real Paillier.
///
/// # Scheme
///
/// This engine implements **Paillier**, and only Paillier. The
/// [`HomomorphicScheme`] enum still names BGV / BFV / CKKS / TFHE because it is
/// part of the published API, but those are fully-homomorphic lattice schemes
/// that this crate does not implement: selecting one makes [`Self::new`] return
/// a structured
/// [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation)
/// error. Use [`HomomorphicScheme::Paillier`].
///
/// # Capabilities
///
/// - [`Self::encrypt`] / [`Self::decrypt`] — real public-key encryption. The
///   ciphertext is a vector of `Z*_{n²}` residues, one per tensor element; it
///   does not contain the plaintext and is randomized, so encrypting the same
///   tensor twice yields different bytes.
/// - [`Self::add_encrypted`] — real homomorphic addition: the decryption of the
///   sum equals the sum of the plaintexts.
/// - [`Self::multiply_encrypted`] — returns `UnsupportedOperation`. Paillier
///   cannot multiply two ciphertexts; see [`Self::multiply_encrypted_by_plaintext`]
///   for what it *can* do.
///
/// # Fixed-point encoding
///
/// Paillier plaintexts are integers in `[0, n)`. Tensor elements are `f32`, so
/// each value is encoded as `round(v * SCALE) + BIAS` with a fixed scale, which
/// makes addition exact up to the scale. The bias keeps negative values
/// representable; it is subtracted after decryption, and the engine tracks how
/// many ciphertexts have been summed so the accumulated bias can be removed.
pub struct HomomorphicEncryptionEngine {
    config: HomomorphicConfig,
    keypair: paillier::PaillierKeypair,
}

/// Fixed-point scale for encoding `f32` tensor values as Paillier plaintexts.
///
/// 2^20 keeps roughly six decimal digits of fraction while leaving ample
/// headroom below the modulus for summing many ciphertexts.
const FIXED_POINT_SCALE: f64 = 1_048_576.0;

/// Additive bias so that negative values map into `[0, n)`.
///
/// Encoded values live in `[0, 2 * BIAS)`, far below any supported modulus.
const FIXED_POINT_BIAS: i64 = 1_i64 << 50;

impl HomomorphicEncryptionEngine {
    /// Create a new homomorphic encryption engine, generating a real Paillier
    /// keypair at the configured security level.
    ///
    /// Key generation samples two random primes and is therefore *slow* —
    /// hundreds of milliseconds to seconds depending on the security level.
    /// Callers should create the engine once and reuse it.
    ///
    /// # Errors
    /// Returns [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation)
    /// when the configured scheme is one of the fully-homomorphic schemes this
    /// crate does not implement.
    pub fn new(config: HomomorphicConfig) -> Result<Self> {
        let modulus_bits = Self::modulus_bits(&config)?;
        let keypair = paillier::generate_keypair(modulus_bits)?;
        Ok(Self { config, keypair })
    }

    /// Create an engine with a caller-supplied CSPRNG. Used by tests to keep
    /// key generation deterministic and fast.
    ///
    /// # Errors
    /// See [`Self::new`].
    pub fn new_with_rng<R: rand_core::CryptoRng>(
        config: HomomorphicConfig,
        rng: &mut R,
    ) -> Result<Self> {
        let modulus_bits = Self::modulus_bits(&config)?;
        let keypair = paillier::generate_keypair_with_rng(modulus_bits, rng)?;
        Ok(Self { config, keypair })
    }

    /// Build an engine around an already-generated keypair.
    ///
    /// Test-only: real callers go through [`Self::new`], which picks the
    /// modulus size from the configured security level. Exposing this would let
    /// a caller install an undersized key.
    #[cfg(test)]
    fn from_parts(config: HomomorphicConfig, keypair: paillier::PaillierKeypair) -> Self {
        Self { config, keypair }
    }

    /// Map the configured scheme and security level onto a Paillier modulus
    /// size, rejecting the schemes that are not implemented.
    fn modulus_bits(config: &HomomorphicConfig) -> Result<u64> {
        match config.scheme {
            HomomorphicScheme::Paillier => {},
            HomomorphicScheme::BGV
            | HomomorphicScheme::BFV
            | HomomorphicScheme::CKKS
            | HomomorphicScheme::TFHE => {
                return Err(paillier::unsupported_paillier_operation(&format!(
                    "fully homomorphic scheme {:?}",
                    config.scheme
                )));
            },
        }
        // NIST-equivalent RSA/Paillier modulus sizes for the requested symmetric
        // security level.
        Ok(match config.security_level {
            SecurityLevel::Bit128 => 3072,
            SecurityLevel::Bit192 => 7680,
            SecurityLevel::Bit256 => 15360,
        })
    }

    /// The engine's configuration.
    pub fn config(&self) -> &HomomorphicConfig {
        &self.config
    }

    /// The Paillier public key. Safe to publish; needed by any party that
    /// wants to encrypt to this engine or add its ciphertexts.
    pub fn public_key(&self) -> &paillier::PaillierPublicKey {
        self.keypair.public_key_ref()
    }

    /// Encrypt a tensor element-wise.
    ///
    /// # Errors
    /// Returns an error if the tensor cannot be read or a value is not finite.
    pub fn encrypt(&self, tensor: &Tensor) -> Result<EncryptedTensor> {
        let mut rng = rand_core::UnwrapErr(getrandom::SysRng);
        self.encrypt_with_rng(tensor, &mut rng)
    }

    /// Encrypt a tensor with a caller-supplied CSPRNG.
    ///
    /// # Errors
    /// Returns an error if the tensor cannot be read or a value is not finite.
    pub fn encrypt_with_rng<R: rand_core::CryptoRng>(
        &self,
        tensor: &Tensor,
        rng: &mut R,
    ) -> Result<EncryptedTensor> {
        let values = tensor.to_vec_f32()?;
        let public = self.keypair.public_key_ref();
        let mut ciphertexts = Vec::with_capacity(values.len());
        for value in &values {
            let encoded = encode_fixed_point(*value)?;
            ciphertexts.push(public.encrypt_with_rng(&encoded, rng)?.to_bytes_be());
        }
        Ok(EncryptedTensor {
            ciphertexts,
            shape: tensor.shape().to_vec(),
            scheme: HomomorphicScheme::Paillier,
            addend_count: 1,
        })
    }

    /// Decrypt an encrypted tensor back to plaintext.
    ///
    /// # Errors
    /// Returns an error if the ciphertext was produced under a different
    /// scheme, or if a residue is malformed.
    pub fn decrypt(&self, encrypted: &EncryptedTensor) -> Result<Tensor> {
        if encrypted.scheme != HomomorphicScheme::Paillier {
            return Err(paillier::unsupported_paillier_operation(&format!(
                "decryption under scheme {:?}",
                encrypted.scheme
            )));
        }
        let mut values = Vec::with_capacity(encrypted.ciphertexts.len());
        for bytes in &encrypted.ciphertexts {
            let ciphertext = paillier::PaillierCiphertext::from_bytes_be(bytes);
            let plain = self.keypair.private_key_ref().decrypt(&ciphertext)?;
            values.push(decode_fixed_point(&plain, encrypted.addend_count)?);
        }
        Tensor::from_vec(values, &encrypted.shape)
    }

    /// Real homomorphic addition: `D(E(a) ⊞ E(b)) == a + b`, element-wise.
    ///
    /// No decryption happens here — the operation is performed entirely on
    /// ciphertexts.
    ///
    /// # Errors
    /// Returns an error if the operands disagree on scheme or shape.
    pub fn add_encrypted(
        &self,
        a: &EncryptedTensor,
        b: &EncryptedTensor,
    ) -> Result<EncryptedTensor> {
        if a.scheme != b.scheme || a.shape != b.shape {
            return Err(invalid_input("Incompatible encrypted tensors for addition"));
        }
        if a.scheme != HomomorphicScheme::Paillier {
            return Err(paillier::unsupported_paillier_operation(&format!(
                "homomorphic addition under scheme {:?}",
                a.scheme
            )));
        }
        if a.ciphertexts.len() != b.ciphertexts.len() {
            return Err(invalid_input(
                "Encrypted tensors have mismatched element counts",
            ));
        }

        let public = self.keypair.public_key_ref();
        let mut ciphertexts = Vec::with_capacity(a.ciphertexts.len());
        for (lhs, rhs) in a.ciphertexts.iter().zip(b.ciphertexts.iter()) {
            let sum = public.add(
                &paillier::PaillierCiphertext::from_bytes_be(lhs),
                &paillier::PaillierCiphertext::from_bytes_be(rhs),
            );
            ciphertexts.push(sum.to_bytes_be());
        }
        Ok(EncryptedTensor {
            ciphertexts,
            shape: a.shape.clone(),
            scheme: HomomorphicScheme::Paillier,
            // Each operand carries its own encoding bias; the sum carries both.
            addend_count: a.addend_count.saturating_add(b.addend_count),
        })
    }

    /// Multiplication of two ciphertexts is **not** supported by Paillier.
    ///
    /// # Errors
    /// Always returns
    /// [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation).
    /// Use [`Self::multiply_encrypted_by_plaintext`] for the ciphertext ×
    /// plaintext product, which Paillier *does* support, or a fully homomorphic
    /// scheme (not implemented here) for ciphertext × ciphertext.
    pub fn multiply_encrypted(
        &self,
        _a: &EncryptedTensor,
        _b: &EncryptedTensor,
    ) -> Result<EncryptedTensor> {
        Err(paillier::unsupported_paillier_operation(
            "homomorphic multiplication of two ciphertexts",
        ))
    }

    /// Real homomorphic scalar multiplication: `D(E(a) ⊠ k) == k · a`.
    ///
    /// `scalar` must be a non-negative integer; Paillier's exponentiation
    /// homomorphism is defined for plaintext exponents in `Z_n`.
    ///
    /// # Errors
    /// Returns an error for a non-Paillier ciphertext.
    pub fn multiply_encrypted_by_plaintext(
        &self,
        a: &EncryptedTensor,
        scalar: u32,
    ) -> Result<EncryptedTensor> {
        if a.scheme != HomomorphicScheme::Paillier {
            return Err(paillier::unsupported_paillier_operation(&format!(
                "scalar multiplication under scheme {:?}",
                a.scheme
            )));
        }
        let public = self.keypair.public_key_ref();
        let big_scalar = BigUint::from(scalar);
        let mut ciphertexts = Vec::with_capacity(a.ciphertexts.len());
        for bytes in &a.ciphertexts {
            let scaled = public.multiply_by_plaintext(
                &paillier::PaillierCiphertext::from_bytes_be(bytes),
                &big_scalar,
            );
            ciphertexts.push(scaled.to_bytes_be());
        }
        Ok(EncryptedTensor {
            ciphertexts,
            shape: a.shape.clone(),
            scheme: HomomorphicScheme::Paillier,
            // Scaling multiplies the encoded bias by the same factor.
            addend_count: a.addend_count.saturating_mul(u64::from(scalar)),
        })
    }

    /// Run `model_fn` over encrypted data without decrypting it.
    ///
    /// The closure receives ciphertexts and must return ciphertexts; the engine
    /// never decrypts. Only the operations this engine actually supports —
    /// [`Self::add_encrypted`] and [`Self::multiply_encrypted_by_plaintext`] —
    /// are available inside it, which is exactly the class of linear functions
    /// Paillier can evaluate.
    ///
    /// # Errors
    /// Propagates whatever `model_fn` returns; also errors if the input is not
    /// a Paillier ciphertext.
    pub fn private_inference<F>(
        &self,
        encrypted_input: &EncryptedTensor,
        model_fn: F,
    ) -> Result<EncryptedTensor>
    where
        F: Fn(&EncryptedTensor) -> Result<EncryptedTensor>,
    {
        if encrypted_input.scheme != HomomorphicScheme::Paillier {
            return Err(paillier::unsupported_paillier_operation(&format!(
                "private inference under scheme {:?}",
                encrypted_input.scheme
            )));
        }
        if encrypted_input.ciphertexts.is_empty() {
            return Err(tensor_op_error(
                "Cannot run private inference on an empty ciphertext",
                "homomorphic_inference",
            ));
        }
        let result = model_fn(encrypted_input)?;
        if result.scheme != HomomorphicScheme::Paillier {
            return Err(tensor_op_error(
                "Private inference produced a ciphertext under a different scheme",
                "homomorphic_inference",
            ));
        }
        Ok(result)
    }
}

/// Encode an `f32` as a Paillier plaintext using fixed point plus a bias.
///
/// # Errors
/// Returns an error for a non-finite value or one that overflows the encoding
/// range, rather than wrapping it into a wrong number.
fn encode_fixed_point(value: f32) -> Result<BigUint> {
    if !value.is_finite() {
        return Err(invalid_input(format!(
            "Cannot homomorphically encrypt a non-finite value: {value}"
        )));
    }
    let scaled = f64::from(value) * FIXED_POINT_SCALE;
    if scaled.abs() >= FIXED_POINT_BIAS as f64 {
        return Err(invalid_input(format!(
            "Value {value} is out of range for the fixed-point homomorphic encoding"
        )));
    }
    // `scaled.abs() < FIXED_POINT_BIAS` was just checked, so the rounded value
    // fits an i64 and the sum with the bias is non-negative.
    let rounded = scaled.round() as i64;
    let biased = rounded.saturating_add(FIXED_POINT_BIAS);
    // `biased` is non-negative by construction.
    Ok(BigUint::from(biased.unsigned_abs()))
}

/// Invert [`encode_fixed_point`], removing `addend_count` copies of the bias.
///
/// # Errors
/// Returns an error if the recovered integer is outside the representable
/// range, which indicates a corrupt ciphertext rather than a valid result.
fn decode_fixed_point(plain: &BigUint, addend_count: u64) -> Result<f32> {
    let digits = plain.to_u64_digits();
    if digits.len() > 2 {
        return Err(invalid_input(
            "Decrypted homomorphic value is out of the fixed-point range".to_string(),
        ));
    }
    let magnitude: u128 = digits.iter().enumerate().map(|(i, d)| u128::from(*d) << (64 * i)).sum();
    let bias = u128::from(addend_count) * (FIXED_POINT_BIAS as u128);
    let signed = i128::try_from(magnitude)
        .map_err(|_| invalid_input("Decrypted homomorphic value overflowed".to_string()))?
        - i128::try_from(bias)
            .map_err(|_| invalid_input("Homomorphic bias overflowed".to_string()))?;
    Ok((signed as f64 / FIXED_POINT_SCALE) as f32)
}

/// An encrypted tensor: one Paillier ciphertext per element.
///
/// The old version of this type held a single `data: Vec<u8>` that was the
/// tensor's raw `f32` bytes. It now holds genuine ciphertexts, and the
/// plaintext is not recoverable from it without the private key.
#[derive(Debug, Clone)]
pub struct EncryptedTensor {
    /// One big-endian Paillier ciphertext per tensor element.
    pub ciphertexts: Vec<Vec<u8>>,
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Encryption scheme used.
    pub scheme: HomomorphicScheme,
    /// How many freshly-encrypted tensors have been summed into this one.
    ///
    /// Needed to strip the accumulated fixed-point bias at decryption time.
    pub addend_count: u64,
}

impl EncryptedTensor {
    /// The number of encrypted elements.
    pub fn len(&self) -> usize {
        self.ciphertexts.len()
    }

    /// Whether the ciphertext holds no elements.
    pub fn is_empty(&self) -> bool {
        self.ciphertexts.is_empty()
    }

    /// Total ciphertext size in bytes, for transport accounting.
    pub fn byte_len(&self) -> usize {
        self.ciphertexts.iter().map(Vec::len).sum()
    }
}

/// Secure multi-party computation engine backed by real Shamir secret sharing.
///
/// # What is implemented
///
/// [`MPCProtocol::ShamirSecretSharing`] only, using the GF(2^8) implementation
/// in [`shamir`]. Splitting a tensor produces `num_parties` shares of which any
/// `threshold` reconstruct it exactly, and any `threshold - 1` are
/// information-theoretically independent of the secret.
///
/// Garbled circuits, BGW and GMW are protocols for evaluating *functions* under
/// MPC, which this crate does not implement; selecting one returns a structured
/// [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation)
/// error rather than the zero-filled placeholder shares that were here before.
pub struct SecureMultipartyEngine {
    config: SecureMultipartyConfig,
    party_id: usize,
    /// This party's own share of each secret it has helped create.
    shares: HashMap<String, shamir::Share>,
}

impl SecureMultipartyEngine {
    /// Create a new secure multi-party computation engine.
    ///
    /// # Errors
    /// Returns an error for an out-of-range `party_id`, a party count outside
    /// `[2, 255]`, or a threshold outside `[2, num_parties]`.
    pub fn new(config: SecureMultipartyConfig, party_id: usize) -> Result<Self> {
        if party_id >= config.num_parties {
            return Err(invalid_input("Party ID exceeds number of parties"));
        }
        if config.num_parties < 2 {
            return Err(invalid_input(format!(
                "Secure multi-party computation needs at least 2 parties, got {}",
                config.num_parties
            )));
        }
        if config.num_parties > shamir::MAX_SHARES {
            return Err(invalid_input(format!(
                "Shamir over GF(2^8) supports at most {} parties, got {}",
                shamir::MAX_SHARES,
                config.num_parties
            )));
        }
        if config.threshold < 2 || config.threshold > config.num_parties {
            return Err(invalid_input(format!(
                "Threshold must be in [2, {}], got {}",
                config.num_parties, config.threshold
            )));
        }

        Ok(Self {
            config,
            party_id,
            shares: HashMap::new(),
        })
    }

    /// This engine's party index.
    pub fn party_id(&self) -> usize {
        self.party_id
    }

    /// The engine configuration.
    pub fn config(&self) -> &SecureMultipartyConfig {
        &self.config
    }

    /// Split `tensor` into real Shamir shares, one per party.
    ///
    /// The tensor is serialized little-endian and split byte-wise. This party's
    /// own share is retained under `secret_id` so that
    /// [`Self::secure_computation`] can use it.
    ///
    /// # Errors
    /// Returns an error for a non-Shamir protocol, or if the tensor cannot be
    /// read.
    pub fn create_shares(
        &mut self,
        tensor: &Tensor,
        secret_id: String,
    ) -> Result<Vec<shamir::Share>> {
        let mut rng = rand_core::UnwrapErr(getrandom::SysRng);
        self.create_shares_with_rng(tensor, secret_id, &mut rng)
    }

    /// Split `tensor` using a caller-supplied CSPRNG.
    ///
    /// # Errors
    /// See [`Self::create_shares`].
    pub fn create_shares_with_rng<R: rand_core::CryptoRng>(
        &mut self,
        tensor: &Tensor,
        secret_id: String,
        rng: &mut R,
    ) -> Result<Vec<shamir::Share>> {
        self.require_shamir("secret sharing")?;

        let payload = serialize_tensor(tensor)?;
        let shares = shamir::split_with_rng(
            &payload,
            self.config.num_parties,
            self.config.threshold,
            rng,
        )?;

        if let Some(our_share) = shares.get(self.party_id) {
            self.shares.insert(secret_id, our_share.clone());
        }
        Ok(shares)
    }

    /// Reconstruct a tensor from at least `threshold` shares.
    ///
    /// # Errors
    /// Returns an error for a non-Shamir protocol, for fewer than `threshold`
    /// shares (rather than returning a wrong tensor, as the old placeholder
    /// did), or for a corrupt payload.
    pub fn reconstruct_secret(&self, shares: &[shamir::Share], _secret_id: &str) -> Result<Tensor> {
        self.require_shamir("secret reconstruction")?;
        let payload = shamir::reconstruct(shares, self.config.threshold)?;
        deserialize_tensor(&payload)
    }

    /// Run `operation` over the shares this party currently holds.
    ///
    /// # Errors
    /// Propagates whatever `operation` returns.
    pub fn secure_computation<F>(&self, operation: F) -> Result<Vec<u8>>
    where
        F: Fn(&[shamir::Share]) -> Result<Vec<u8>>,
    {
        let held: Vec<shamir::Share> = self.shares.values().cloned().collect();
        operation(&held)
    }

    /// This party's stored share for `secret_id`, if any.
    pub fn stored_share(&self, secret_id: &str) -> Option<&shamir::Share> {
        self.shares.get(secret_id)
    }

    /// Reject the protocols that are named in the config but not implemented.
    fn require_shamir(&self, operation: &str) -> Result<()> {
        match self.config.protocol {
            MPCProtocol::ShamirSecretSharing => Ok(()),
            MPCProtocol::GarbledCircuits | MPCProtocol::BGW | MPCProtocol::GMW => {
                Err(trustformers_core::errors::unsupported_operation(
                    format!("{operation} with protocol {:?}", self.config.protocol),
                    "trustformers-mobile (only MPCProtocol::ShamirSecretSharing is implemented)"
                        .to_string(),
                ))
            },
        }
    }
}

/// Serialize a tensor as `rank ‖ dims ‖ little-endian f32 values`.
///
/// A self-describing encoding is needed because Shamir operates on opaque
/// bytes: the shape must survive the round trip.
fn serialize_tensor(tensor: &Tensor) -> Result<Vec<u8>> {
    let shape = tensor.shape().to_vec();
    let values = tensor.to_vec_f32()?;
    let rank = u32::try_from(shape.len())
        .map_err(|_| invalid_input("Tensor rank does not fit in u32".to_string()))?;

    let mut out = Vec::with_capacity(4 + shape.len() * 8 + values.len() * 4);
    out.extend_from_slice(&rank.to_le_bytes());
    for dim in &shape {
        let dim64 = u64::try_from(*dim)
            .map_err(|_| invalid_input("Tensor dimension does not fit in u64".to_string()))?;
        out.extend_from_slice(&dim64.to_le_bytes());
    }
    for value in &values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

/// Inverse of [`serialize_tensor`].
///
/// # Errors
/// Returns an error for a truncated or inconsistent payload, rather than
/// silently producing a differently-shaped tensor.
fn deserialize_tensor(payload: &[u8]) -> Result<Tensor> {
    if payload.len() < 4 {
        return Err(invalid_input(
            "Reconstructed tensor payload is truncated".to_string(),
        ));
    }
    let mut rank_bytes = [0u8; 4];
    rank_bytes.copy_from_slice(&payload[..4]);
    let rank = u32::from_le_bytes(rank_bytes) as usize;

    let dims_end = 4 + rank * 8;
    if payload.len() < dims_end {
        return Err(invalid_input(
            "Reconstructed tensor payload is truncated in the shape header".to_string(),
        ));
    }
    let mut shape = Vec::with_capacity(rank);
    for chunk in payload[4..dims_end].chunks_exact(8) {
        let mut dim_bytes = [0u8; 8];
        dim_bytes.copy_from_slice(chunk);
        shape
            .push(usize::try_from(u64::from_le_bytes(dim_bytes)).map_err(|_| {
                invalid_input("Tensor dimension does not fit in usize".to_string())
            })?);
    }

    let value_bytes = &payload[dims_end..];
    if !value_bytes.len().is_multiple_of(4) {
        return Err(invalid_input(
            "Reconstructed tensor payload has a partial f32".to_string(),
        ));
    }
    let mut values = Vec::with_capacity(value_bytes.len() / 4);
    for chunk in value_bytes.chunks_exact(4) {
        let mut value_array = [0u8; 4];
        value_array.copy_from_slice(chunk);
        values.push(f32::from_le_bytes(value_array));
    }

    let expected: usize = shape.iter().product();
    if expected != values.len() {
        return Err(invalid_input(format!(
            "Reconstructed tensor shape {shape:?} implies {expected} elements but {} were present",
            values.len()
        )));
    }
    Tensor::from_vec(values, &shape)
}

/// Zero-knowledge proof engine backed by a real Schnorr sigma protocol.
///
/// # What is implemented
///
/// A non-interactive proof of knowledge of a discrete logarithm over a
/// 2048-bit safe-prime group, made non-interactive with Fiat-Shamir over
/// SHA-256 (see [`zkp`]). The proof is bound to a caller-supplied statement
/// (here the model hash), so it cannot be replayed for a different model.
///
/// The previous implementation appended the *witness in cleartext* to the proof
/// and "verified" by checking `proof.len() > 32`. Both behaviours are covered by
/// regression tests: the witness must not appear in the proof bytes, and
/// arbitrary blobs must not verify.
///
/// # What is not implemented
///
/// [`ZKProofSystem::ZkSNARKs`], [`ZKProofSystem::ZkSTARKs`],
/// [`ZKProofSystem::Bulletproofs`] and [`ZKProofSystem::Plonk`] are
/// general-purpose circuit proof systems. Selecting one makes [`Self::new`]
/// return a structured
/// [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation)
/// error naming what is available. Use [`ZKProofSystem::SchnorrSigma`].
pub struct ZeroKnowledgeProofEngine {
    config: ZKProofConfig,
    keypair: zkp::SchnorrKeypair,
}

impl ZeroKnowledgeProofEngine {
    /// Create an engine, generating a fresh Schnorr keypair.
    ///
    /// # Errors
    /// Returns [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation)
    /// for a circuit proof system this crate does not implement.
    pub fn new(config: ZKProofConfig) -> Result<Self> {
        Self::require_supported(&config.proof_system)?;
        Ok(Self {
            config,
            keypair: zkp::SchnorrKeypair::generate(),
        })
    }

    /// Create an engine with a caller-supplied CSPRNG (deterministic tests).
    ///
    /// # Errors
    /// See [`Self::new`].
    pub fn new_with_rng<R: rand_core::CryptoRng>(
        config: ZKProofConfig,
        rng: &mut R,
    ) -> Result<Self> {
        Self::require_supported(&config.proof_system)?;
        Ok(Self {
            config,
            keypair: zkp::SchnorrKeypair::generate_with_rng(zkp::SchnorrGroup::default(), rng),
        })
    }

    fn require_supported(system: &ZKProofSystem) -> Result<()> {
        match system {
            ZKProofSystem::SchnorrSigma => Ok(()),
            ZKProofSystem::ZkSNARKs
            | ZKProofSystem::ZkSTARKs
            | ZKProofSystem::Bulletproofs
            | ZKProofSystem::Plonk => Err(zkp::unsupported_proof_system(&format!("{system:?}"))),
        }
    }

    /// The engine configuration.
    pub fn config(&self) -> &ZKProofConfig {
        &self.config
    }

    /// The public verification data. Publish this alongside proofs.
    pub fn verifier(&self) -> zkp::SchnorrVerifier {
        self.keypair.verifier()
    }

    /// Prove knowledge of the engine's secret witness, bound to `model_hash`.
    ///
    /// Unlike the old API this takes no `witness` argument: the witness is the
    /// engine's own secret and is never accepted from — nor revealed to — the
    /// caller. `model_hash` is public and becomes the Fiat-Shamir context, so a
    /// proof for one model does not verify for another.
    ///
    /// # Errors
    /// Returns an error if the generated proof would exceed the configured
    /// `max_proof_size`.
    pub fn prove_model_integrity(&self, model_hash: &[u8]) -> Result<ZKProof> {
        let mut rng = rand_core::UnwrapErr(getrandom::SysRng);
        self.prove_model_integrity_with_rng(model_hash, &mut rng)
    }

    /// Prove with a caller-supplied CSPRNG.
    ///
    /// # Errors
    /// See [`Self::prove_model_integrity`].
    pub fn prove_model_integrity_with_rng<R: rand_core::CryptoRng>(
        &self,
        model_hash: &[u8],
        rng: &mut R,
    ) -> Result<ZKProof> {
        let proof = self.keypair.prove_with_rng(model_hash, rng);
        let data = proof.to_bytes();
        if data.len() > self.config.verification.max_proof_size {
            return Err(invalid_input(format!(
                "Schnorr proof is {} bytes, exceeding the configured maximum of {}",
                data.len(),
                self.config.verification.max_proof_size
            )));
        }
        Ok(ZKProof {
            data,
            system: ZKProofSystem::SchnorrSigma,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }

    /// Verify a proof against `public_inputs` (the model hash it was bound to).
    ///
    /// Returns `Ok(false)` for a forged, malformed, replayed or oversized
    /// proof. It never returns `true` for anything the algebraic check rejects.
    ///
    /// # Errors
    /// Returns an error only for a proof declaring an unsupported system.
    pub fn verify_proof(&self, proof: &ZKProof, public_inputs: &[u8]) -> Result<bool> {
        if proof.data.len() > self.config.verification.max_proof_size {
            return Ok(false);
        }
        if proof.system != ZKProofSystem::SchnorrSigma {
            return Err(zkp::unsupported_proof_system(&format!(
                "{:?}",
                proof.system
            )));
        }
        let Ok(parsed) = zkp::SchnorrProof::from_bytes(&proof.data) else {
            return Ok(false);
        };
        Ok(self.keypair.verifier().verify(&parsed, public_inputs))
    }
}

/// A zero-knowledge proof.
///
/// `data` is the serialized Schnorr transcript `(commitment, response)`. It
/// does **not** contain the witness — that is the point of the scheme, and is
/// asserted by the module's regression tests.
#[derive(Debug, Clone)]
pub struct ZKProof {
    /// Serialized proof.
    pub data: Vec<u8>,
    /// Proof system used.
    pub system: ZKProofSystem,
    /// Unix timestamp at which the proof was generated.
    pub timestamp: u64,
}

/// Post-quantum cryptography engine backed by the real NIST standards.
///
/// # What is implemented
///
/// | Config value | Real algorithm | Crate |
/// |---|---|---|
/// | [`QuantumResistantAlgorithm::MlKem768`] | FIPS 203 ML-KEM-768 (formerly Kyber) | `ml-kem` |
/// | [`QuantumResistantSignature::MlDsa65`] | FIPS 204 ML-DSA-65 (formerly Dilithium) | `ml-dsa` |
/// | [`QuantumResistantSignature::SlhDsaShake128f`] | FIPS 205 SLH-DSA (formerly SPHINCS+) | `slh-dsa` |
///
/// [`Self::encrypt`] is a KEM-DEM construction: a real ML-KEM encapsulation
/// carrying a ChaCha20-Poly1305 AEAD, so a tampered ciphertext is rejected
/// rather than mis-decrypted. The previous implementation returned
/// `public_key || plaintext` and called that Kyber encryption.
///
/// # What is not implemented
///
/// [`QuantumResistantAlgorithm::ClassicMcEliece`],
/// [`QuantumResistantAlgorithm::Multivariate`],
/// [`QuantumResistantAlgorithm::HashBased`] and
/// [`QuantumResistantSignature::Falcon`] have no vetted pure-Rust
/// implementation on crates.io. Selecting one makes [`Self::new`] return a
/// structured
/// [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation)
/// error naming the algorithms that are real.
pub struct QuantumResistantEngine {
    config: QuantumResistantConfig,
    kem: pqc::KyberKem,
    signer: PostQuantumSigner,
}

impl std::fmt::Debug for QuantumResistantEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Key material is deliberately not printed.
        f.debug_struct("QuantumResistantEngine")
            .field("encryption", &self.config.encryption_algorithm)
            .field("signature", &self.config.signature_algorithm)
            .finish()
    }
}

/// The signature backend actually in use.
enum PostQuantumSigner {
    /// FIPS 204 ML-DSA-65.
    MlDsa(pqc::DilithiumSigner),
    /// FIPS 205 SLH-DSA-SHAKE-128f.
    SlhDsa(pqc::SphincsSigner),
}

impl QuantumResistantEngine {
    /// Create an engine, generating real keypairs for the configured
    /// algorithms.
    ///
    /// # Errors
    /// Returns [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation)
    /// for an algorithm with no vetted pure-Rust implementation.
    pub fn new(config: QuantumResistantConfig) -> Result<Self> {
        Self::check_algorithms(&config)?;
        let signer = match config.signature_algorithm {
            QuantumResistantSignature::MlDsa65 => {
                let mut rng = rand_core::UnwrapErr(getrandom::SysRng);
                PostQuantumSigner::MlDsa(pqc::DilithiumSigner::generate_from_rng(&mut rng))
            },
            QuantumResistantSignature::SlhDsaShake128f => {
                let mut rng = rand_core::UnwrapErr(getrandom::SysRng);
                PostQuantumSigner::SlhDsa(pqc::SphincsSigner::generate_from_rng(&mut rng))
            },
            QuantumResistantSignature::Falcon => {
                return Err(pqc::unsupported("Falcon"));
            },
        };
        Ok(Self {
            config,
            kem: pqc::KyberKem::generate(),
            signer,
        })
    }

    /// Create an engine with a caller-supplied CSPRNG (deterministic tests).
    ///
    /// # Errors
    /// See [`Self::new`].
    pub fn new_with_rng<R: rand_core::CryptoRng>(
        config: QuantumResistantConfig,
        rng: &mut R,
    ) -> Result<Self> {
        Self::check_algorithms(&config)?;
        let signer = match config.signature_algorithm {
            QuantumResistantSignature::MlDsa65 => {
                PostQuantumSigner::MlDsa(pqc::DilithiumSigner::generate_from_rng(rng))
            },
            QuantumResistantSignature::SlhDsaShake128f => {
                PostQuantumSigner::SlhDsa(pqc::SphincsSigner::generate_from_rng(rng))
            },
            QuantumResistantSignature::Falcon => return Err(pqc::unsupported("Falcon")),
        };
        Ok(Self {
            config,
            kem: pqc::KyberKem::generate_from_rng(rng),
            signer,
        })
    }

    /// Reject the algorithms with no real implementation before any key
    /// material is generated.
    fn check_algorithms(config: &QuantumResistantConfig) -> Result<()> {
        match config.encryption_algorithm {
            QuantumResistantAlgorithm::MlKem768 => {},
            QuantumResistantAlgorithm::ClassicMcEliece => {
                return Err(pqc::unsupported("Classic McEliece"))
            },
            QuantumResistantAlgorithm::Multivariate => {
                return Err(pqc::unsupported("multivariate encryption"))
            },
            QuantumResistantAlgorithm::HashBased => {
                return Err(pqc::unsupported("hash-based encryption"))
            },
        }
        match config.key_exchange {
            QuantumResistantKeyExchange::MlKem768 => {},
            QuantumResistantKeyExchange::SIKE => {
                // SIKE was broken by a classical attack in 2022 and must not be
                // offered at all.
                return Err(pqc::unsupported(
                    "SIKE (cryptographically broken by the Castryck-Decru attack, 2022)",
                ));
            },
            QuantumResistantKeyExchange::NTRU => return Err(pqc::unsupported("NTRU")),
        }
        match config.signature_algorithm {
            QuantumResistantSignature::MlDsa65 | QuantumResistantSignature::SlhDsaShake128f => {},
            QuantumResistantSignature::Falcon => return Err(pqc::unsupported("Falcon")),
        }
        Ok(())
    }

    /// The engine configuration.
    pub fn config(&self) -> &QuantumResistantConfig {
        &self.config
    }

    /// The ML-KEM encapsulation key ("public key"). Safe to publish.
    pub fn encapsulation_key_bytes(&self) -> Vec<u8> {
        self.kem.encapsulation_key_bytes()
    }

    /// The signature verifying key ("public key"). Safe to publish.
    pub fn verifying_key_bytes(&self) -> Vec<u8> {
        match &self.signer {
            PostQuantumSigner::MlDsa(signer) => signer.verifying_key_bytes(),
            PostQuantumSigner::SlhDsa(signer) => signer.verifying_key_bytes(),
        }
    }

    /// Encrypt `data` with real ML-KEM-768 + ChaCha20-Poly1305.
    ///
    /// # Errors
    /// Returns an error if the AEAD layer fails.
    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.kem.encrypt(data, b"")
    }

    /// Decrypt a ciphertext produced by [`Self::encrypt`].
    ///
    /// # Errors
    /// Returns an error — never a wrong plaintext — for a tampered ciphertext
    /// or a mismatched key.
    pub fn decrypt(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        self.kem.decrypt(encrypted_data, b"")
    }

    /// Sign `data` with the configured post-quantum signature algorithm.
    ///
    /// # Errors
    /// Infallible for the supported algorithms; the `Result` is kept for API
    /// stability.
    pub fn sign(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(match &self.signer {
            PostQuantumSigner::MlDsa(signer) => signer.sign(data),
            PostQuantumSigner::SlhDsa(signer) => signer.sign(data),
        })
    }

    /// Verify a signature over `data`.
    ///
    /// Returns `Ok(false)` for a forged, tampered or malformed signature.
    ///
    /// # Errors
    /// Infallible for the supported algorithms; the `Result` is kept for API
    /// stability.
    pub fn verify(&self, data: &[u8], signature: &[u8]) -> Result<bool> {
        Ok(match &self.signer {
            PostQuantumSigner::MlDsa(signer) => signer.verify(data, signature),
            PostQuantumSigner::SlhDsa(signer) => signer.verify(data, signature),
        })
    }
}

/// Advanced security manager that combines the real security engines.
///
/// Each engine is constructed only when its config section is `enabled`; a
/// disabled section means the corresponding protection is genuinely absent,
/// which [`SecureInferenceResult`] reports honestly.
pub struct AdvancedSecurityManager {
    config: AdvancedSecurityConfig,
    homomorphic_engine: Option<HomomorphicEncryptionEngine>,
    mpc_engine: Option<SecureMultipartyEngine>,
    zk_engine: Option<ZeroKnowledgeProofEngine>,
    quantum_engine: Option<QuantumResistantEngine>,
}

impl AdvancedSecurityManager {
    /// Create a new advanced security manager.
    ///
    /// Enabling homomorphic encryption triggers real Paillier key generation,
    /// which takes seconds at the default 3072-bit modulus. Construct once and
    /// reuse.
    ///
    /// # Errors
    /// Propagates any engine's
    /// [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation)
    /// for an algorithm this crate does not implement.
    pub fn new(config: AdvancedSecurityConfig) -> Result<Self> {
        let homomorphic_engine = if config.homomorphic_encryption.enabled {
            Some(HomomorphicEncryptionEngine::new(
                config.homomorphic_encryption.clone(),
            )?)
        } else {
            None
        };

        let mpc_engine = if config.secure_multiparty.enabled {
            Some(SecureMultipartyEngine::new(
                config.secure_multiparty.clone(),
                0,
            )?)
        } else {
            None
        };

        let zk_engine = if config.zero_knowledge_proofs.enabled {
            Some(ZeroKnowledgeProofEngine::new(
                config.zero_knowledge_proofs.clone(),
            )?)
        } else {
            None
        };

        let quantum_engine = if config.quantum_resistant.enabled {
            Some(QuantumResistantEngine::new(
                config.quantum_resistant.clone(),
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            homomorphic_engine,
            mpc_engine,
            zk_engine,
            quantum_engine,
        })
    }

    /// The manager configuration.
    pub fn config(&self) -> &AdvancedSecurityConfig {
        &self.config
    }

    /// The homomorphic engine, if enabled.
    pub fn homomorphic_engine(&self) -> Option<&HomomorphicEncryptionEngine> {
        self.homomorphic_engine.as_ref()
    }

    /// The zero-knowledge engine, if enabled.
    pub fn zk_engine(&self) -> Option<&ZeroKnowledgeProofEngine> {
        self.zk_engine.as_ref()
    }

    /// The post-quantum engine, if enabled.
    pub fn quantum_engine(&self) -> Option<&QuantumResistantEngine> {
        self.quantum_engine.as_ref()
    }

    /// Run inference, applying whichever protections are enabled.
    ///
    /// `model_fn` operates on plaintext. If homomorphic encryption is enabled,
    /// `linear_model_fn` is used instead and runs entirely on ciphertexts — the
    /// manager never decrypts to apply it, which is the whole point. Because
    /// Paillier only supports linear functions, that closure is restricted to
    /// the homomorphic operations the engine exposes.
    ///
    /// `model_hash` binds any generated zero-knowledge proof to this particular
    /// model, so the proof cannot be replayed for another.
    ///
    /// # Errors
    /// Propagates errors from the model function and from the engines.
    pub fn secure_inference<F, G>(
        &self,
        input: &Tensor,
        model_hash: &[u8],
        model_fn: F,
        linear_model_fn: G,
    ) -> Result<SecureInferenceResult>
    where
        F: Fn(&Tensor) -> Result<Tensor>,
        G: Fn(&HomomorphicEncryptionEngine, &EncryptedTensor) -> Result<EncryptedTensor>,
    {
        let start_time = std::time::Instant::now();

        let (result, homomorphic_used) = if let Some(he_engine) = &self.homomorphic_engine {
            // The model runs on ciphertexts; no plaintext is exposed inside.
            let encrypted_input = he_engine.encrypt(input)?;
            let encrypted_result = he_engine
                .private_inference(&encrypted_input, |ct| linear_model_fn(he_engine, ct))?;
            (he_engine.decrypt(&encrypted_result)?, true)
        } else {
            (model_fn(input)?, false)
        };

        let computation_time = start_time.elapsed();

        let proof = match &self.zk_engine {
            Some(zk_engine) => Some(zk_engine.prove_model_integrity(model_hash)?),
            None => None,
        };

        Ok(SecureInferenceResult {
            result,
            computation_time,
            security_level: self.estimate_security_level(),
            proof,
            homomorphic_used,
            // `secure_inference` does not itself split shares or wrap the
            // payload post-quantum: it reports availability, not use. Naming
            // these `*_available` keeps the claim honest.
            mpc_available: self.mpc_engine.is_some(),
            quantum_resistant_available: self.quantum_engine.is_some(),
        })
    }

    /// Fraction of the manager's protections that are active, in `[0, 1]`.
    ///
    /// This is a coverage indicator, not a cryptographic strength estimate —
    /// the weights are a stated convention, not a measurement.
    fn estimate_security_level(&self) -> f32 {
        let mut score = 0.0;

        if self.homomorphic_engine.is_some() {
            score += 0.3;
        }
        if self.mpc_engine.is_some() {
            score += 0.2;
        }
        if self.zk_engine.is_some() {
            score += 0.2;
        }
        if self.quantum_engine.is_some() {
            score += 0.3;
        }

        score
    }
}

/// Result of secure inference.
#[derive(Debug)]
pub struct SecureInferenceResult {
    /// The inference result.
    pub result: Tensor,
    /// Measured wall-clock time for the computation.
    pub computation_time: std::time::Duration,
    /// Fraction of the available protections that were active, in `[0, 1]`.
    ///
    /// A coverage indicator, not a bit-strength claim.
    pub security_level: f32,
    /// Zero-knowledge proof, when the ZK engine is enabled.
    pub proof: Option<ZKProof>,
    /// Whether the computation actually ran on ciphertexts.
    ///
    /// Unlike the two fields below this is a statement about what happened:
    /// `true` means `model_fn` never saw plaintext.
    pub homomorphic_used: bool,
    /// Whether a multi-party engine is *available* on this manager.
    ///
    /// `secure_inference` does not perform secret sharing, so this is
    /// deliberately not called `mpc_used`: sharing is a separate call the
    /// caller makes through [`SecureMultipartyEngine`].
    pub mpc_available: bool,
    /// Whether a post-quantum engine is *available* on this manager.
    ///
    /// `secure_inference` does not wrap its result post-quantum, so this
    /// reports availability rather than use. Call [`QuantumResistantEngine`]
    /// directly to encrypt or sign a payload.
    pub quantum_resistant_available: bool,
}

impl Default for AdvancedSecurityConfig {
    fn default() -> Self {
        Self {
            homomorphic_encryption: HomomorphicConfig {
                enabled: false,
                scheme: HomomorphicScheme::Paillier,
                security_level: SecurityLevel::Bit128,
                optimization: EncryptionOptimization {
                    enable_batching: true,
                    enable_bootstrapping: false,
                    relinearization_threshold: 2,
                    memory_optimization: true,
                },
            },
            secure_multiparty: SecureMultipartyConfig {
                enabled: false,
                num_parties: 3,
                threshold: 2,
                protocol: MPCProtocol::ShamirSecretSharing,
                communication: MPCCommunication {
                    secure_channels: true,
                    timeout_seconds: 30,
                    max_message_size: 1024 * 1024, // 1MB
                    enable_compression: true,
                },
            },
            zero_knowledge_proofs: ZKProofConfig {
                enabled: false,
                proof_system: ZKProofSystem::SchnorrSigma,
                verification: ZKVerificationConfig {
                    batch_verification: true,
                    timeout_seconds: 10,
                    cache_results: true,
                    max_proof_size: 1024 * 1024, // 1MB
                },
            },
            quantum_resistant: QuantumResistantConfig {
                enabled: false,
                encryption_algorithm: QuantumResistantAlgorithm::MlKem768,
                signature_algorithm: QuantumResistantSignature::MlDsa65,
                key_exchange: QuantumResistantKeyExchange::MlKem768,
            },
        }
    }
}

#[cfg(test)]
mod tests;
