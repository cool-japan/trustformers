//! Real cryptographic primitives backing the encryption subsystem.
//!
//! Every routine in this module performs the algorithm it advertises. There is
//! no XOR "demo" path, no fabricated authentication tag and no pseudo-random
//! key material derived from a clock reading:
//!
//! * AEAD sealing/opening uses the RustCrypto [`aes_gcm`] and
//!   [`chacha20poly1305`] implementations with **detached** 128-bit tags, so
//!   ciphertext length equals plaintext length and the tag is carried
//!   separately (matching [`super::service::EncryptedData`]).
//! * Tag verification is performed by the AEAD implementations themselves and
//!   is constant time; a mismatch surfaces as
//!   [`EncryptionError::AuthenticationFailed`].
//! * Unauthenticated modes (AES-256-CBC, Salsa20) are the real ciphers as
//!   well — CBC with PKCS#7 padding, Salsa20 as a keystream XOR.
//! * Random material comes from the operating system CSPRNG via
//!   [`getrandom::fill`].
//! * Key derivation uses HKDF-SHA256 and PBKDF2-HMAC-SHA256.

use aes::cipher::block_padding::Pkcs7;
use aes::cipher::{BlockModeDecrypt, BlockModeEncrypt, KeyIvInit};
use aes_gcm::aead::{AeadCore, AeadInOut, KeyInit, Nonce};
use aes_gcm::{Aes128Gcm, Aes256Gcm};
use chacha20poly1305::{ChaCha20Poly1305, XChaCha20Poly1305};
use hkdf::Hkdf;
use hmac::Hmac;
use salsa20::cipher::StreamCipher;
use salsa20::Salsa20;
use sha2::{Sha256, Sha512};
use subtle::ConstantTimeEq;

use super::errors::{EncryptionError, EncryptionResult};
use super::types::EncryptionAlgorithm;

/// AES-256-CBC encryptor/decryptor aliases.
type Aes256CbcEnc = cbc::Encryptor<aes::Aes256>;
type Aes256CbcDec = cbc::Decryptor<aes::Aes256>;

/// Length of the authentication tag produced by every supported AEAD mode.
pub const AEAD_TAG_LEN: usize = 16;

/// Fill `dest` with cryptographically secure random bytes from the OS CSPRNG.
///
/// # Errors
///
/// Returns [`EncryptionError::KeyGenerationFailed`] if the operating system
/// entropy source is unavailable. The buffer is left untouched in that case;
/// callers must not fall back to a deterministic value.
pub fn fill_random(dest: &mut [u8]) -> EncryptionResult<()> {
    getrandom::fill(dest).map_err(|e| EncryptionError::KeyGenerationFailed {
        message: format!("operating system CSPRNG unavailable: {e}"),
    })
}

/// Allocate `len` cryptographically secure random bytes.
///
/// # Errors
///
/// Propagates any failure from [`fill_random`].
pub fn random_bytes(len: usize) -> EncryptionResult<Vec<u8>> {
    let mut out = vec![0u8; len];
    fill_random(&mut out)?;
    Ok(out)
}

/// Encrypt `plaintext` under `algorithm`.
///
/// Returns `(ciphertext, tag)`. For authenticated algorithms the tag is
/// `Some(16 bytes)` and `ciphertext.len() == plaintext.len()`. For
/// unauthenticated algorithms the tag is `None`; AES-256-CBC applies PKCS#7
/// padding, so its ciphertext is longer than the plaintext.
///
/// `aad` is authenticated but not encrypted (AEAD modes only).
///
/// # Errors
///
/// Returns an error when the key or nonce length does not match the algorithm,
/// or when the underlying primitive rejects the input.
pub fn seal(
    algorithm: &EncryptionAlgorithm,
    key: &[u8],
    nonce: &[u8],
    aad: &[u8],
    plaintext: &[u8],
) -> EncryptionResult<(Vec<u8>, Option<Vec<u8>>)> {
    check_lengths(algorithm, key, nonce)?;

    match algorithm {
        EncryptionAlgorithm::AES256GCM => {
            let mut buf = plaintext.to_vec();
            let tag = aead_seal::<Aes256Gcm>(key, nonce, aad, &mut buf)?;
            Ok((buf, Some(tag)))
        },
        EncryptionAlgorithm::AES128GCM => {
            let mut buf = plaintext.to_vec();
            let tag = aead_seal::<Aes128Gcm>(key, nonce, aad, &mut buf)?;
            Ok((buf, Some(tag)))
        },
        EncryptionAlgorithm::ChaCha20Poly1305 => {
            let mut buf = plaintext.to_vec();
            let tag = aead_seal::<ChaCha20Poly1305>(key, nonce, aad, &mut buf)?;
            Ok((buf, Some(tag)))
        },
        EncryptionAlgorithm::XChaCha20Poly1305 => {
            let mut buf = plaintext.to_vec();
            let tag = aead_seal::<XChaCha20Poly1305>(key, nonce, aad, &mut buf)?;
            Ok((buf, Some(tag)))
        },
        EncryptionAlgorithm::AES256CBC => {
            let encryptor = Aes256CbcEnc::new_from_slices(key, nonce).map_err(|_| {
                EncryptionError::EncryptionFailed {
                    message: "AES-256-CBC key/IV length mismatch".to_string(),
                }
            })?;
            Ok((encryptor.encrypt_padded_vec::<Pkcs7>(plaintext), None))
        },
        EncryptionAlgorithm::Salsa20 => {
            let mut buf = plaintext.to_vec();
            let mut stream = Salsa20::new_from_slices(key, nonce).map_err(|_| {
                EncryptionError::EncryptionFailed {
                    message: "Salsa20 key/nonce length mismatch".to_string(),
                }
            })?;
            stream.apply_keystream(&mut buf);
            Ok((buf, None))
        },
    }
}

/// Decrypt `ciphertext` under `algorithm`, verifying the authentication tag
/// when the algorithm is authenticated.
///
/// # Errors
///
/// * [`EncryptionError::AuthenticationFailed`] when the tag is missing, has the
///   wrong length, or does not verify (constant-time check inside the AEAD).
/// * [`EncryptionError::DecryptionFailed`] for malformed CBC padding or a
///   length mismatch.
pub fn open(
    algorithm: &EncryptionAlgorithm,
    key: &[u8],
    nonce: &[u8],
    aad: &[u8],
    ciphertext: &[u8],
    tag: Option<&[u8]>,
) -> EncryptionResult<Vec<u8>> {
    check_lengths(algorithm, key, nonce)?;

    if algorithm.is_authenticated() {
        let tag = tag.ok_or(EncryptionError::AuthenticationFailed)?;
        if tag.len() != AEAD_TAG_LEN {
            return Err(EncryptionError::AuthenticationFailed);
        }
        let mut buf = ciphertext.to_vec();
        match algorithm {
            EncryptionAlgorithm::AES256GCM => {
                aead_open::<Aes256Gcm>(key, nonce, aad, &mut buf, tag)
            },
            EncryptionAlgorithm::AES128GCM => {
                aead_open::<Aes128Gcm>(key, nonce, aad, &mut buf, tag)
            },
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                aead_open::<ChaCha20Poly1305>(key, nonce, aad, &mut buf, tag)
            },
            EncryptionAlgorithm::XChaCha20Poly1305 => {
                aead_open::<XChaCha20Poly1305>(key, nonce, aad, &mut buf, tag)
            },
            // Unreachable: `is_authenticated()` is false for the remaining variants.
            EncryptionAlgorithm::AES256CBC | EncryptionAlgorithm::Salsa20 => {
                Err(EncryptionError::UnsupportedAlgorithm {
                    algorithm: format!("{algorithm:?}"),
                })
            },
        }?;
        return Ok(buf);
    }

    match algorithm {
        EncryptionAlgorithm::AES256CBC => {
            let decryptor = Aes256CbcDec::new_from_slices(key, nonce).map_err(|_| {
                EncryptionError::DecryptionFailed {
                    message: "AES-256-CBC key/IV length mismatch".to_string(),
                }
            })?;
            decryptor.decrypt_padded_vec::<Pkcs7>(ciphertext).map_err(|_| {
                EncryptionError::DecryptionFailed {
                    message: "AES-256-CBC PKCS#7 padding is invalid".to_string(),
                }
            })
        },
        EncryptionAlgorithm::Salsa20 => {
            let mut buf = ciphertext.to_vec();
            let mut stream = Salsa20::new_from_slices(key, nonce).map_err(|_| {
                EncryptionError::DecryptionFailed {
                    message: "Salsa20 key/nonce length mismatch".to_string(),
                }
            })?;
            stream.apply_keystream(&mut buf);
            Ok(buf)
        },
        other => Err(EncryptionError::UnsupportedAlgorithm {
            algorithm: format!("{other:?}"),
        }),
    }
}

/// Derive `out_len` bytes of key material from input keying material using
/// HKDF-SHA256 (RFC 5869).
///
/// # Errors
///
/// Returns [`EncryptionError::KeyDerivationFailed`] if `out_len` exceeds the
/// HKDF-SHA256 maximum of 255 × 32 bytes.
pub fn hkdf_sha256(
    ikm: &[u8],
    salt: &[u8],
    info: &[u8],
    out_len: usize,
) -> EncryptionResult<Vec<u8>> {
    let hkdf = Hkdf::<Sha256>::new(Some(salt), ikm);
    let mut okm = vec![0u8; out_len];
    hkdf.expand(info, &mut okm).map_err(|e| EncryptionError::KeyDerivationFailed {
        message: format!("HKDF-SHA256 expand failed: {e}"),
    })?;
    Ok(okm)
}

/// Derive `out_len` bytes from a password using PBKDF2-HMAC-SHA256.
///
/// # Errors
///
/// Returns [`EncryptionError::KeyDerivationFailed`] when `iterations` is zero
/// or the underlying PBKDF2 implementation rejects the parameters.
pub fn pbkdf2_sha256(
    password: &[u8],
    salt: &[u8],
    iterations: u32,
    out_len: usize,
) -> EncryptionResult<Vec<u8>> {
    if iterations == 0 {
        return Err(EncryptionError::KeyDerivationFailed {
            message: "PBKDF2 iteration count must be greater than zero".to_string(),
        });
    }
    let mut okm = vec![0u8; out_len];
    pbkdf2::pbkdf2::<Hmac<Sha256>>(password, salt, iterations, &mut okm).map_err(|e| {
        EncryptionError::KeyDerivationFailed {
            message: format!("PBKDF2-HMAC-SHA256 failed: {e}"),
        }
    })?;
    Ok(okm)
}

/// Derive `out_len` bytes from a password using PBKDF2-HMAC-SHA512.
///
/// # Errors
///
/// Returns [`EncryptionError::KeyDerivationFailed`] when `iterations` is zero
/// or the underlying PBKDF2 implementation rejects the parameters.
pub fn pbkdf2_sha512(
    password: &[u8],
    salt: &[u8],
    iterations: u32,
    out_len: usize,
) -> EncryptionResult<Vec<u8>> {
    if iterations == 0 {
        return Err(EncryptionError::KeyDerivationFailed {
            message: "PBKDF2 iteration count must be greater than zero".to_string(),
        });
    }
    let mut okm = vec![0u8; out_len];
    pbkdf2::pbkdf2::<Hmac<Sha512>>(password, salt, iterations, &mut okm).map_err(|e| {
        EncryptionError::KeyDerivationFailed {
            message: format!("PBKDF2-HMAC-SHA512 failed: {e}"),
        }
    })?;
    Ok(okm)
}

/// Constant-time equality check for secret material.
///
/// Returns `true` only when both slices have the same length and contents. The
/// comparison does not short-circuit on the first differing byte.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.ct_eq(b).into()
}

fn check_lengths(
    algorithm: &EncryptionAlgorithm,
    key: &[u8],
    nonce: &[u8],
) -> EncryptionResult<()> {
    if key.len() != algorithm.key_size() {
        return Err(EncryptionError::KeyValidationFailed {
            message: format!(
                "{algorithm:?} requires a {}-byte key, got {}",
                algorithm.key_size(),
                key.len()
            ),
        });
    }
    if nonce.len() != algorithm.nonce_size() {
        return Err(EncryptionError::InvalidNonce {
            message: format!(
                "{algorithm:?} requires a {}-byte nonce/IV, got {}",
                algorithm.nonce_size(),
                nonce.len()
            ),
        });
    }
    Ok(())
}

fn aead_seal<A>(key: &[u8], nonce: &[u8], aad: &[u8], buf: &mut [u8]) -> EncryptionResult<Vec<u8>>
where
    A: KeyInit + AeadInOut + AeadCore,
{
    let cipher = A::new_from_slice(key).map_err(|_| EncryptionError::KeyValidationFailed {
        message: "AEAD key length mismatch".to_string(),
    })?;
    let nonce = Nonce::<A>::try_from(nonce).map_err(|_| EncryptionError::InvalidNonce {
        message: "AEAD nonce length mismatch".to_string(),
    })?;
    let tag = cipher.encrypt_inout_detached(&nonce, aad, buf.into()).map_err(|_| {
        EncryptionError::EncryptionFailed {
            message: "AEAD encryption rejected the input".to_string(),
        }
    })?;
    Ok(tag.to_vec())
}

fn aead_open<A>(
    key: &[u8],
    nonce: &[u8],
    aad: &[u8],
    buf: &mut [u8],
    tag: &[u8],
) -> EncryptionResult<()>
where
    A: KeyInit + AeadInOut + AeadCore,
{
    let cipher = A::new_from_slice(key).map_err(|_| EncryptionError::KeyValidationFailed {
        message: "AEAD key length mismatch".to_string(),
    })?;
    let nonce = Nonce::<A>::try_from(nonce).map_err(|_| EncryptionError::InvalidNonce {
        message: "AEAD nonce length mismatch".to_string(),
    })?;
    let tag = aes_gcm::aead::Tag::<A>::try_from(tag)
        .map_err(|_| EncryptionError::AuthenticationFailed)?;
    cipher
        .decrypt_inout_detached(&nonce, aad, buf.into(), &tag)
        .map_err(|_| EncryptionError::AuthenticationFailed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex_to_bytes(s: &str) -> Vec<u8> {
        hex::decode(s).expect("test vector must be valid hex")
    }

    // ── NIST GCM specification, Test Case 13 (AES-256, empty plaintext) ──────
    #[test]
    fn test_aes256gcm_nist_test_case_13() {
        let key = [0u8; 32];
        let nonce = [0u8; 12];
        let (ct, tag) = seal(&EncryptionAlgorithm::AES256GCM, &key, &nonce, b"", b"")
            .expect("seal must succeed");
        assert!(ct.is_empty(), "empty plaintext must give empty ciphertext");
        assert_eq!(
            tag.expect("GCM must produce a tag"),
            hex_to_bytes("530f8afbc74536b9a963b4f1c4cb738b"),
            "AES-256-GCM tag must match NIST test case 13"
        );
    }

    // ── NIST GCM specification, Test Case 14 (AES-256, one zero block) ───────
    #[test]
    fn test_aes256gcm_nist_test_case_14() {
        let key = [0u8; 32];
        let nonce = [0u8; 12];
        let (ct, tag) = seal(
            &EncryptionAlgorithm::AES256GCM,
            &key,
            &nonce,
            b"",
            &[0u8; 16],
        )
        .expect("seal must succeed");
        assert_eq!(ct, hex_to_bytes("cea7403d4d606b6e074ec5d3baf39d18"));
        assert_eq!(
            tag.expect("GCM must produce a tag"),
            hex_to_bytes("d0d1c8a799996bf0265b98b5d48ab919")
        );
    }

    // ── RFC 8439 §2.8.2 ChaCha20-Poly1305 AEAD test vector ──────────────────
    #[test]
    fn test_chacha20poly1305_rfc8439_vector() {
        let key: Vec<u8> = (0x80u8..=0x9fu8).collect();
        let nonce = hex_to_bytes("070000004041424344454647");
        let aad = hex_to_bytes("50515253c0c1c2c3c4c5c6c7");
        let plaintext = b"Ladies and Gentlemen of the class of '99: If I could offer you \
only one tip for the future, sunscreen would be it.";
        let expected_ct = hex_to_bytes(concat!(
            "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d6",
            "3dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36",
            "92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc",
            "3ff4def08e4b7a9de576d26586cec64b6116"
        ));
        let expected_tag = hex_to_bytes("1ae10b594f09e26a7e902ecbd0600691");

        let (ct, tag) = seal(
            &EncryptionAlgorithm::ChaCha20Poly1305,
            &key,
            &nonce,
            &aad,
            plaintext,
        )
        .expect("seal must succeed");
        assert_eq!(ct, expected_ct, "RFC 8439 ciphertext mismatch");
        assert_eq!(
            tag.expect("ChaCha20-Poly1305 must produce a tag"),
            expected_tag,
            "RFC 8439 tag mismatch"
        );
    }

    // ── NIST SP 800-38A F.2.5, CBC-AES256.Encrypt first block ───────────────
    #[test]
    fn test_aes256cbc_nist_sp800_38a_first_block() {
        let key = hex_to_bytes("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4");
        let iv = hex_to_bytes("000102030405060708090a0b0c0d0e0f");
        let plaintext = hex_to_bytes("6bc1bee22e409f96e93d7e117393172a");
        let (ct, tag) = seal(&EncryptionAlgorithm::AES256CBC, &key, &iv, b"", &plaintext)
            .expect("seal must succeed");
        assert!(tag.is_none(), "CBC is unauthenticated: no tag");
        assert_eq!(
            &ct[..16],
            &hex_to_bytes("f58c4c04d6e5f1ba779eabfb5f7bfbd6")[..],
            "AES-256-CBC first block must match NIST SP 800-38A"
        );
        assert_eq!(ct.len(), 32, "PKCS#7 appends a full padding block");
    }

    #[test]
    fn test_aead_roundtrip_all_authenticated_algorithms() {
        for algorithm in [
            EncryptionAlgorithm::AES256GCM,
            EncryptionAlgorithm::AES128GCM,
            EncryptionAlgorithm::ChaCha20Poly1305,
            EncryptionAlgorithm::XChaCha20Poly1305,
        ] {
            let key = random_bytes(algorithm.key_size()).expect("csprng");
            let nonce = random_bytes(algorithm.nonce_size()).expect("csprng");
            let plaintext = b"the quick brown fox jumps over the lazy dog";
            let (ct, tag) =
                seal(&algorithm, &key, &nonce, b"aad", plaintext).expect("seal must succeed");
            assert_eq!(
                ct.len(),
                plaintext.len(),
                "{algorithm:?} must be length preserving"
            );
            let pt = open(&algorithm, &key, &nonce, b"aad", &ct, tag.as_deref())
                .expect("open must succeed");
            assert_eq!(pt, plaintext.to_vec());
        }
    }

    #[test]
    fn test_tampered_ciphertext_is_rejected() {
        let algorithm = EncryptionAlgorithm::AES256GCM;
        let key = random_bytes(algorithm.key_size()).expect("csprng");
        let nonce = random_bytes(algorithm.nonce_size()).expect("csprng");
        let (mut ct, tag) =
            seal(&algorithm, &key, &nonce, b"", b"authenticate me").expect("seal must succeed");
        ct[0] ^= 0x01;
        let err = open(&algorithm, &key, &nonce, b"", &ct, tag.as_deref())
            .expect_err("tampered ciphertext must not decrypt");
        assert!(matches!(err, EncryptionError::AuthenticationFailed));
    }

    #[test]
    fn test_tampered_tag_is_rejected() {
        let algorithm = EncryptionAlgorithm::ChaCha20Poly1305;
        let key = random_bytes(algorithm.key_size()).expect("csprng");
        let nonce = random_bytes(algorithm.nonce_size()).expect("csprng");
        let (ct, tag) = seal(&algorithm, &key, &nonce, b"", b"authenticate me").expect("seal");
        let mut tag = tag.expect("tag");
        tag[15] ^= 0x80;
        let err = open(&algorithm, &key, &nonce, b"", &ct, Some(&tag))
            .expect_err("tampered tag must not verify");
        assert!(matches!(err, EncryptionError::AuthenticationFailed));
    }

    #[test]
    fn test_wrong_nonce_is_rejected() {
        let algorithm = EncryptionAlgorithm::AES256GCM;
        let key = random_bytes(algorithm.key_size()).expect("csprng");
        let nonce = random_bytes(algorithm.nonce_size()).expect("csprng");
        let (ct, tag) = seal(&algorithm, &key, &nonce, b"", b"authenticate me").expect("seal");
        let mut other_nonce = nonce.clone();
        other_nonce[0] ^= 0xff;
        let err = open(&algorithm, &key, &other_nonce, b"", &ct, tag.as_deref())
            .expect_err("wrong nonce must not verify");
        assert!(matches!(err, EncryptionError::AuthenticationFailed));
    }

    #[test]
    fn test_wrong_aad_is_rejected() {
        let algorithm = EncryptionAlgorithm::AES256GCM;
        let key = random_bytes(algorithm.key_size()).expect("csprng");
        let nonce = random_bytes(algorithm.nonce_size()).expect("csprng");
        let (ct, tag) = seal(&algorithm, &key, &nonce, b"key-a", b"payload").expect("seal");
        let err = open(&algorithm, &key, &nonce, b"key-b", &ct, tag.as_deref())
            .expect_err("AAD mismatch must not verify");
        assert!(matches!(err, EncryptionError::AuthenticationFailed));
    }

    #[test]
    fn test_missing_tag_is_rejected_for_aead() {
        let algorithm = EncryptionAlgorithm::AES256GCM;
        let key = random_bytes(algorithm.key_size()).expect("csprng");
        let nonce = random_bytes(algorithm.nonce_size()).expect("csprng");
        let (ct, _tag) = seal(&algorithm, &key, &nonce, b"", b"payload").expect("seal");
        let err = open(&algorithm, &key, &nonce, b"", &ct, None)
            .expect_err("missing tag must be rejected");
        assert!(matches!(err, EncryptionError::AuthenticationFailed));
    }

    #[test]
    fn test_salsa20_roundtrip_and_is_not_identity() {
        let algorithm = EncryptionAlgorithm::Salsa20;
        let key = random_bytes(algorithm.key_size()).expect("csprng");
        let nonce = random_bytes(algorithm.nonce_size()).expect("csprng");
        let plaintext = vec![0u8; 64];
        let (ct, tag) = seal(&algorithm, &key, &nonce, b"", &plaintext).expect("seal");
        assert!(tag.is_none());
        assert_ne!(ct, plaintext, "keystream must not be all zeros");
        let pt = open(&algorithm, &key, &nonce, b"", &ct, None).expect("open");
        assert_eq!(pt, plaintext);
    }

    #[test]
    fn test_key_length_mismatch_is_rejected() {
        let err = seal(
            &EncryptionAlgorithm::AES256GCM,
            &[0u8; 16],
            &[0u8; 12],
            b"",
            b"x",
        )
        .expect_err("short key must be rejected");
        assert!(matches!(err, EncryptionError::KeyValidationFailed { .. }));
    }

    #[test]
    fn test_nonce_length_mismatch_is_rejected() {
        let err = seal(
            &EncryptionAlgorithm::AES256GCM,
            &[0u8; 32],
            &[0u8; 8],
            b"",
            b"x",
        )
        .expect_err("short nonce must be rejected");
        assert!(matches!(err, EncryptionError::InvalidNonce { .. }));
    }

    #[test]
    fn test_random_bytes_are_not_constant() {
        let a = random_bytes(32).expect("csprng");
        let b = random_bytes(32).expect("csprng");
        assert_ne!(a, b, "CSPRNG must not repeat 32-byte draws");
        assert_ne!(a, vec![0u8; 32], "CSPRNG must not return all zeros");
    }

    // ── RFC 5869 Test Case 1 (HKDF-SHA256) ──────────────────────────────────
    #[test]
    fn test_hkdf_sha256_rfc5869_test_case_1() {
        let ikm = vec![0x0bu8; 22];
        let salt = hex_to_bytes("000102030405060708090a0b0c");
        let info = hex_to_bytes("f0f1f2f3f4f5f6f7f8f9");
        let okm = hkdf_sha256(&ikm, &salt, &info, 42).expect("hkdf");
        assert_eq!(
            okm,
            hex_to_bytes(concat!(
                "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf",
                "34007208d5b887185865"
            ))
        );
    }

    // ── RFC 6070 Test Case 2 (PBKDF2-HMAC-SHA1 semantics, SHA-256 variant) ──
    // Uses the widely published PBKDF2-HMAC-SHA256 vector for
    // ("password", "salt", c = 2, dkLen = 32).
    #[test]
    fn test_pbkdf2_sha256_known_answer() {
        let okm = pbkdf2_sha256(b"password", b"salt", 2, 32).expect("pbkdf2");
        assert_eq!(
            okm,
            hex_to_bytes("ae4d0c95af6b46d32d0adff928f06dd02a303f8ef3c251dfd6e2d85a95474c43")
        );
    }

    #[test]
    fn test_pbkdf2_rejects_zero_iterations() {
        let err = pbkdf2_sha256(b"password", b"salt", 0, 32)
            .expect_err("zero iterations must be rejected");
        assert!(matches!(err, EncryptionError::KeyDerivationFailed { .. }));
    }

    #[test]
    fn test_constant_time_eq_behaviour() {
        assert!(constant_time_eq(b"abcdef", b"abcdef"));
        assert!(!constant_time_eq(b"abcdef", b"abcdeg"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
    }
}
