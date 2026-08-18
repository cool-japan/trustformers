//! Real post-quantum cryptography backing [`QuantumResistantEngine`].
//!
//! [`QuantumResistantEngine`]: super::QuantumResistantEngine
//!
//! Everything here delegates to audited-in-the-open, pure-Rust RustCrypto
//! implementations of the finalized NIST standards:
//!
//! | This module | Crate | Standard |
//! |---|---|---|
//! | [`KyberKem`] | [`ml_kem`] | FIPS 203 ML-KEM (formerly CRYSTALS-Kyber) |
//! | [`DilithiumSigner`] | [`ml_dsa`] | FIPS 204 ML-DSA (formerly CRYSTALS-Dilithium) |
//! | [`SphincsSigner`] | [`slh_dsa`] | FIPS 205 SLH-DSA (formerly SPHINCS+) |
//!
//! The previous implementation of this file's callers concatenated the public
//! key with the plaintext and called it "Kyber encryption", and signed by
//! copying 64 bytes of a constant "private key". Both are gone: key material
//! below is a real keypair produced by the real crate, ciphertexts are real
//! ML-KEM encapsulations wrapped around a real ChaCha20-Poly1305 AEAD (a KEM
//! transports a symmetric key, it does not encrypt a message directly), and
//! signatures are real ML-DSA / SLH-DSA signatures that fail verification when
//! the message, the key, or the signature is altered.
//!
//! Classic McEliece and Falcon (FN-DSA, FIPS 206) are **not implemented here**.
//! Pure-Rust crates for both do exist on crates.io — `classic-mceliece-rust`
//! and the FN-DSA/`falcon-rust` family — but none is a RustCrypto-maintained
//! implementation of a finalized FIPS standard on the footing of the three
//! above (FIPS 206 was still draft at the time of writing, and the McEliece
//! crate is a single-maintainer port), so this crate does not take on the
//! security responsibility of shipping them under an `encrypt`/`sign` API.
//! Selecting one yields a structured
//! [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation)
//! error naming the algorithms that are real, rather than pretending. Adding
//! them is a deliberate future decision, not an impossibility.

use chacha20poly1305::aead::{Aead, KeyInit as AeadKeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key as AeadKey, Nonce};
use ml_dsa::{
    signature::{Keypair as MlDsaKeypair, SignatureEncoding, Signer as MlDsaSigner, Verifier},
    KeyExport as MlDsaKeyExport, MlDsa65, Signature as MlDsaSignature,
    SigningKey as MlDsaSigningKey,
};
use ml_kem::kem::{Decapsulate, Encapsulate};
use ml_kem::{Ciphertext as KemCiphertext, KeyExport as MlKemKeyExport, MlKem768};
use rand_core::CryptoRng;
use slh_dsa::{
    signature::{
        Keypair as SlhKeypair, RandomizedSigner, SignatureEncoding as SlhSignatureEncoding,
        Signer as SlhSigner, Verifier as SlhVerifier,
    },
    Shake128f, Signature as SlhSignature, SigningKey as SlhSigningKey,
};
use trustformers_core::errors::{invalid_input, unsupported_operation, Result, TrustformersError};

/// The post-quantum algorithms this crate actually implements, in the form
/// used by the [`unsupported`] error message.
pub const SUPPORTED_ALGORITHMS: &str =
    "ML-KEM-768 (FIPS 203, formerly Kyber), ML-DSA-65 (FIPS 204, formerly Dilithium), \
     SLH-DSA-SHAKE-128f (FIPS 205, formerly SPHINCS+)";

/// Nonce length of ChaCha20-Poly1305, in bytes.
const AEAD_NONCE_LEN: usize = 12;

/// Structured error for a post-quantum algorithm this crate does not implement.
///
/// Used instead of a placeholder so that a caller asking for Classic McEliece
/// or Falcon learns immediately that it is not available and what is.
pub fn unsupported(algorithm: &str) -> TrustformersError {
    unsupported_operation(
        format!("post-quantum algorithm {algorithm}"),
        format!("trustformers-mobile (not implemented here); supported: {SUPPORTED_ALGORITHMS}"),
    )
}

/// A real ML-KEM-768 keypair plus the AEAD that turns the KEM into an
/// encryption scheme.
///
/// ML-KEM on its own only transports a 32-byte shared secret. To encrypt an
/// arbitrary message we use the standard KEM-DEM construction: encapsulate a
/// shared secret to the recipient's encapsulation key, use that secret as a
/// ChaCha20-Poly1305 key, and ship `ciphertext || nonce || aead_ciphertext`.
/// Decapsulation recovers the same secret and authenticates the AEAD tag, so a
/// tampered ciphertext is rejected rather than silently mis-decrypted.
pub struct KyberKem {
    decapsulation_key: ml_kem::DecapsulationKey<MlKem768>,
    encapsulation_key: ml_kem::EncapsulationKey<MlKem768>,
}

impl std::fmt::Debug for KyberKem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print key material.
        f.debug_struct("KyberKem").field("algorithm", &"ML-KEM-768").finish()
    }
}

impl KyberKem {
    /// Generate a fresh ML-KEM-768 keypair from the supplied CSPRNG.
    pub fn generate_from_rng<R: CryptoRng>(rng: &mut R) -> Self {
        let (decapsulation_key, encapsulation_key) =
            <MlKem768 as ml_kem::Kem>::generate_keypair_from_rng(rng);
        Self {
            decapsulation_key,
            encapsulation_key,
        }
    }

    /// Generate a fresh ML-KEM-768 keypair from the operating system CSPRNG.
    pub fn generate() -> Self {
        let (decapsulation_key, encapsulation_key) = <MlKem768 as ml_kem::Kem>::generate_keypair();
        Self {
            decapsulation_key,
            encapsulation_key,
        }
    }

    /// The serialized encapsulation key ("public key"), 1184 bytes for ML-KEM-768.
    pub fn encapsulation_key_bytes(&self) -> Vec<u8> {
        self.encapsulation_key.to_bytes().to_vec()
    }

    /// Encrypt `plaintext` to this keypair's encapsulation key.
    ///
    /// Layout: `kem_ciphertext (1088 B) || nonce (12 B) || aead_ciphertext`.
    pub fn encrypt(&self, plaintext: &[u8], associated_data: &[u8]) -> Result<Vec<u8>> {
        let (kem_ciphertext, shared_secret) = self.encapsulation_key.encapsulate();

        // The KEM shared secret is a uniformly random 32-byte value, which is
        // exactly a ChaCha20-Poly1305 key. Derive the nonce from the KEM
        // ciphertext transcript so it is unique per encapsulation without
        // needing extra randomness.
        let aead = ChaCha20Poly1305::new(AeadKey::from_slice(shared_secret.as_slice()));
        let nonce_bytes = derive_nonce(shared_secret.as_slice(), kem_ciphertext.as_slice());
        let nonce = Nonce::from_slice(&nonce_bytes);

        let sealed = aead
            .encrypt(
                nonce,
                Payload {
                    msg: plaintext,
                    aad: associated_data,
                },
            )
            .map_err(|_| invalid_input("ML-KEM AEAD encryption failed"))?;

        let mut out = Vec::with_capacity(kem_ciphertext.len() + AEAD_NONCE_LEN + sealed.len());
        out.extend_from_slice(kem_ciphertext.as_slice());
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&sealed);
        Ok(out)
    }

    /// Decrypt a ciphertext produced by [`KyberKem::encrypt`].
    ///
    /// Returns an error — never a wrong plaintext — when the KEM ciphertext, the
    /// nonce or the AEAD tag has been tampered with, or when the decapsulation
    /// key does not match the one used for encryption.
    pub fn decrypt(&self, ciphertext: &[u8], associated_data: &[u8]) -> Result<Vec<u8>> {
        let kem_len = kem_ciphertext_len();
        if ciphertext.len() < kem_len + AEAD_NONCE_LEN {
            return Err(invalid_input(format!(
                "ML-KEM ciphertext too short: {} bytes, need at least {}",
                ciphertext.len(),
                kem_len + AEAD_NONCE_LEN
            )));
        }
        let (kem_part, rest) = ciphertext.split_at(kem_len);
        let (nonce_bytes, sealed) = rest.split_at(AEAD_NONCE_LEN);

        let kem_ciphertext = KemCiphertext::<MlKem768>::try_from(kem_part)
            .map_err(|_| invalid_input("malformed ML-KEM ciphertext"))?;
        let shared_secret = self.decapsulation_key.decapsulate(&kem_ciphertext);

        // ML-KEM decapsulation is implicit-rejection: a corrupted KEM ciphertext
        // yields a *different* pseudorandom secret rather than an error. The
        // AEAD tag below is what turns that into a hard failure.
        let aead = ChaCha20Poly1305::new(AeadKey::from_slice(shared_secret.as_slice()));
        let nonce = Nonce::from_slice(nonce_bytes);
        aead.decrypt(
            nonce,
            Payload {
                msg: sealed,
                aad: associated_data,
            },
        )
        .map_err(|_| {
            invalid_input(
                "ML-KEM authenticated decryption failed (wrong key or tampered ciphertext)",
            )
        })
    }
}

/// Length in bytes of a raw ML-KEM-768 ciphertext (1088).
fn kem_ciphertext_len() -> usize {
    KemCiphertext::<MlKem768>::default().len()
}

/// Derive a deterministic, per-encapsulation AEAD nonce.
///
/// The shared secret is fresh for every encapsulation, so hashing it together
/// with the KEM ciphertext yields a nonce that never repeats for a given key.
fn derive_nonce(shared_secret: &[u8], kem_ciphertext: &[u8]) -> [u8; AEAD_NONCE_LEN] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"trustformers-mobile/ml-kem-768/nonce/v1");
    hasher.update(shared_secret);
    hasher.update(kem_ciphertext);
    let digest = hasher.finalize();
    let mut nonce = [0u8; AEAD_NONCE_LEN];
    nonce.copy_from_slice(&digest[..AEAD_NONCE_LEN]);
    nonce
}

/// A real ML-DSA-65 (FIPS 204) signing keypair.
pub struct DilithiumSigner {
    signing_key: MlDsaSigningKey<MlDsa65>,
}

impl std::fmt::Debug for DilithiumSigner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DilithiumSigner").field("algorithm", &"ML-DSA-65").finish()
    }
}

impl DilithiumSigner {
    /// Generate a fresh ML-DSA-65 keypair from the supplied CSPRNG.
    pub fn generate_from_rng<R: CryptoRng + ?Sized>(rng: &mut R) -> Self {
        Self {
            signing_key: <MlDsaSigningKey<MlDsa65> as ml_dsa::Generate>::generate_from_rng(rng),
        }
    }

    /// The serialized verifying key ("public key"), 1952 bytes for ML-DSA-65.
    pub fn verifying_key_bytes(&self) -> Vec<u8> {
        self.signing_key.verifying_key().to_bytes().to_vec()
    }

    /// Sign `message`, returning the 3309-byte ML-DSA-65 signature.
    pub fn sign(&self, message: &[u8]) -> Vec<u8> {
        self.signing_key.sign(message).encode().to_vec()
    }

    /// Verify `signature` over `message` against this keypair's verifying key.
    ///
    /// Returns `Ok(false)` for a well-formed but invalid signature and
    /// `Ok(false)` for a malformed one; it never returns `true` for anything the
    /// real verifier rejects.
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> bool {
        let Ok(encoded) = ml_dsa::EncodedSignature::<MlDsa65>::try_from(signature) else {
            return false;
        };
        let Some(parsed) = MlDsaSignature::<MlDsa65>::decode(&encoded) else {
            return false;
        };
        self.signing_key.verifying_key().verify(message, &parsed).is_ok()
    }
}

/// A real SLH-DSA-SHAKE-128f (FIPS 205) signing keypair.
///
/// SLH-DSA is stateless hash-based: the security assumption is only the hash
/// function, at the cost of a ~17 KiB signature.
pub struct SphincsSigner {
    signing_key: SlhSigningKey<Shake128f>,
}

impl std::fmt::Debug for SphincsSigner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SphincsSigner")
            .field("algorithm", &"SLH-DSA-SHAKE-128f")
            .finish()
    }
}

impl SphincsSigner {
    /// Generate a fresh SLH-DSA-SHAKE-128f keypair from the supplied CSPRNG.
    pub fn generate_from_rng<R: CryptoRng + ?Sized>(rng: &mut R) -> Self {
        Self {
            signing_key: SlhSigningKey::<Shake128f>::new(rng),
        }
    }

    /// The serialized verifying key ("public key"), 32 bytes for SHAKE-128f.
    pub fn verifying_key_bytes(&self) -> Vec<u8> {
        self.signing_key.verifying_key().to_bytes().to_vec()
    }

    /// Sign `message` deterministically.
    pub fn sign(&self, message: &[u8]) -> Vec<u8> {
        self.signing_key.sign(message).to_bytes().to_vec()
    }

    /// Sign `message` with fresh randomness (hedged signing).
    pub fn sign_with_rng<R: CryptoRng + ?Sized>(&self, rng: &mut R, message: &[u8]) -> Vec<u8> {
        self.signing_key.sign_with_rng(rng, message).to_bytes().to_vec()
    }

    /// Verify `signature` over `message`.
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> bool {
        let Ok(parsed) = SlhSignature::<Shake128f>::try_from(signature) else {
            return false;
        };
        self.signing_key.verifying_key().verify(message, &parsed).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::advanced_security::test_rng::TestRng;

    /// Regression for the old `kyber_encrypt` = `public_key || plaintext`.
    /// The ciphertext must not contain the plaintext anywhere.
    #[test]
    fn ml_kem_ciphertext_does_not_contain_plaintext() {
        let mut rng = TestRng::seeded(1);
        let kem = KyberKem::generate_from_rng(&mut rng);
        let plaintext = b"attack at dawn, this exact byte string must not appear in the ciphertext";
        let ciphertext = kem.encrypt(plaintext, b"").expect("encrypt");

        assert!(
            !ciphertext.windows(plaintext.len()).any(|w| w == plaintext),
            "ciphertext leaks the plaintext verbatim"
        );
        // The old placeholder also prefixed the public key.
        let public = kem.encapsulation_key_bytes();
        assert!(
            !ciphertext
                .windows(public.len().min(ciphertext.len()))
                .any(|w| w == public.as_slice()),
            "ciphertext leaks the public key verbatim"
        );
    }

    #[test]
    fn ml_kem_round_trip() {
        let mut rng = TestRng::seeded(2);
        let kem = KyberKem::generate_from_rng(&mut rng);
        let plaintext = b"private inference payload";
        let ciphertext = kem.encrypt(plaintext, b"aad").expect("encrypt");
        let recovered = kem.decrypt(&ciphertext, b"aad").expect("decrypt");
        assert_eq!(&recovered[..], plaintext);
    }

    #[test]
    fn ml_kem_rejects_wrong_key() {
        let mut rng_a = TestRng::seeded(3);
        let mut rng_b = TestRng::seeded(4);
        let alice = KyberKem::generate_from_rng(&mut rng_a);
        let mallory = KyberKem::generate_from_rng(&mut rng_b);

        let ciphertext = alice.encrypt(b"secret", b"").expect("encrypt");
        assert!(
            mallory.decrypt(&ciphertext, b"").is_err(),
            "decryption under a foreign key must fail"
        );
    }

    #[test]
    fn ml_kem_rejects_tampered_ciphertext() {
        let mut rng = TestRng::seeded(5);
        let kem = KyberKem::generate_from_rng(&mut rng);
        let mut ciphertext = kem.encrypt(b"secret", b"").expect("encrypt");
        let last = ciphertext.len() - 1;
        ciphertext[last] ^= 0x01;
        assert!(
            kem.decrypt(&ciphertext, b"").is_err(),
            "tampered AEAD tag must fail"
        );

        let mut ciphertext = kem.encrypt(b"secret", b"").expect("encrypt");
        ciphertext[0] ^= 0x01; // corrupt the KEM part
        assert!(
            kem.decrypt(&ciphertext, b"").is_err(),
            "tampered KEM ciphertext must fail"
        );
    }

    #[test]
    fn ml_kem_rejects_wrong_associated_data() {
        let mut rng = TestRng::seeded(6);
        let kem = KyberKem::generate_from_rng(&mut rng);
        let ciphertext = kem.encrypt(b"secret", b"context-a").expect("encrypt");
        assert!(kem.decrypt(&ciphertext, b"context-b").is_err());
    }

    #[test]
    fn ml_dsa_sign_verify_round_trip() {
        let mut rng = TestRng::seeded(7);
        let signer = DilithiumSigner::generate_from_rng(&mut rng);
        let message = b"model weights digest";
        let signature = signer.sign(message);
        assert!(signer.verify(message, &signature));
    }

    /// Regression for the old `dilithium_verify`, which compared a prefix of a
    /// constant key and so accepted a forgery anyone could construct.
    #[test]
    fn ml_dsa_rejects_forgery() {
        let mut rng = TestRng::seeded(8);
        let signer = DilithiumSigner::generate_from_rng(&mut rng);
        let message = b"model weights digest";
        let signature = signer.sign(message);

        // Wrong message.
        assert!(!signer.verify(b"different message", &signature));

        // Bit-flipped signature.
        let mut tampered = signature.clone();
        tampered[0] ^= 0x01;
        assert!(!signer.verify(message, &tampered));

        // The old forgery recipe: 64 constant bytes followed by the message prefix.
        let mut old_style_forgery = vec![4u8; 64];
        old_style_forgery.extend_from_slice(&message[..message.len().min(32)]);
        assert!(!signer.verify(message, &old_style_forgery));

        // Signature from a different key.
        let mut rng2 = TestRng::seeded(9);
        let other = DilithiumSigner::generate_from_rng(&mut rng2);
        assert!(!signer.verify(message, &other.sign(message)));
    }

    #[test]
    fn slh_dsa_sign_verify_round_trip() {
        let mut rng = TestRng::seeded(10);
        let signer = SphincsSigner::generate_from_rng(&mut rng);
        let message = b"checkpoint attestation";
        let signature = signer.sign(message);
        assert!(signer.verify(message, &signature));
    }

    #[test]
    fn slh_dsa_rejects_forgery() {
        let mut rng = TestRng::seeded(11);
        let signer = SphincsSigner::generate_from_rng(&mut rng);
        let message = b"checkpoint attestation";
        let mut signature = signer.sign(message);
        signature[0] ^= 0x01;
        assert!(!signer.verify(message, &signature));

        // The old forgery recipe.
        let mut old_style_forgery = vec![4u8; 96];
        old_style_forgery.extend_from_slice(&message[..message.len().min(32)]);
        assert!(!signer.verify(message, &old_style_forgery));
    }

    #[test]
    fn slh_dsa_randomized_signing_verifies() {
        let mut rng = TestRng::seeded(12);
        let signer = SphincsSigner::generate_from_rng(&mut rng);
        let message = b"hedged";
        let mut sign_rng = TestRng::seeded(13);
        let signature = signer.sign_with_rng(&mut sign_rng, message);
        assert!(signer.verify(message, &signature));
    }

    #[test]
    fn unsupported_error_lists_the_real_algorithms() {
        let err = unsupported("Classic McEliece");
        let text = err.to_string();
        assert!(text.contains("Classic McEliece"), "{text}");
        let full = format!("{err:?}");
        assert!(
            full.contains("ML-KEM-768") || text.contains("ML-KEM-768"),
            "error should name the supported algorithms: {full}"
        );
    }
}
