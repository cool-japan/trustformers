//! The privacy engines, backed by the real primitives in
//! [`advanced_security`](crate::advanced_security).
//!
//! Split out of `mod.rs` to keep each file under the 2000-line limit. Every
//! engine here replaces a placeholder that returned zeros or constants; the doc
//! comment on each names what it used to do.

use super::*;

/// Secure multiparty computation via real Shamir secret sharing.
///
/// The previous implementation returned a single
/// `SecretShare { share_data: vec![0u8; 32] }` regardless of input. It now
/// splits the actual tensor into `num_parties` real shares.
pub struct SecureMultipartyComputation {
    config: SecureMultipartyConfig,
}

impl SecureMultipartyComputation {
    /// Create the engine.
    ///
    /// # Errors
    /// Returns an error for a party count outside `[2, 255]` or a threshold
    /// outside `[2, num_parties]`.
    pub fn new(config: SecureMultipartyConfig) -> Result<Self> {
        if config.num_parties < 2 || config.num_parties > shamir::MAX_SHARES {
            return Err(CoreError::InvalidInput(format!(
                "Shamir secret sharing supports 2..={} parties, got {}",
                shamir::MAX_SHARES,
                config.num_parties
            )));
        }
        if config.threshold < 2 || config.threshold > config.num_parties {
            return Err(CoreError::InvalidInput(format!(
                "Threshold must be in [2, {}], got {}",
                config.num_parties, config.threshold
            )));
        }
        Ok(Self { config })
    }

    /// The engine configuration.
    pub fn config(&self) -> &SecureMultipartyConfig {
        &self.config
    }

    /// Split `data` into real Shamir shares.
    ///
    /// # Errors
    /// Returns an error for a non-Shamir scheme, or if the tensor cannot be
    /// read.
    pub async fn create_secret_shares(
        &self,
        data: &Tensor,
        client_id: &str,
    ) -> Result<Vec<SecretShare>> {
        let mut rng = rand_core::UnwrapErr(getrandom::SysRng);
        self.create_secret_shares_with_rng(data, client_id, &mut rng)
    }

    /// Split with a caller-supplied CSPRNG.
    ///
    /// # Errors
    /// See [`Self::create_secret_shares`].
    pub fn create_secret_shares_with_rng<R: rand_core::CryptoRng>(
        &self,
        data: &Tensor,
        client_id: &str,
        rng: &mut R,
    ) -> Result<Vec<SecretShare>> {
        match self.config.secret_sharing {
            SecretSharingScheme::Shamir => {},
            SecretSharingScheme::Additive
            | SecretSharingScheme::Replicated
            | SecretSharingScheme::Packed => {
                return Err(unsupported(
                    format!("secret sharing scheme {:?}", self.config.secret_sharing),
                    "trustformers-mobile (only SecretSharingScheme::Shamir is implemented)"
                        .to_string(),
                ));
            },
        }

        let payload = serialize_tensor_for_sharing(data)?;
        let shares = shamir::split_with_rng(
            &payload,
            self.config.num_parties,
            self.config.threshold,
            rng,
        )?;

        Ok(shares
            .into_iter()
            .map(|share| SecretShare {
                share_id: usize::from(share.index()),
                share_data: share.to_bytes(),
                client_id: client_id.to_string(),
            })
            .collect())
    }

    /// Reconstruct a tensor from at least `threshold` shares.
    ///
    /// # Errors
    /// Returns an error for fewer than `threshold` shares (rather than a wrong
    /// tensor), or for a corrupt payload.
    pub fn reconstruct(&self, shares: &[SecretShare]) -> Result<Tensor> {
        let parsed = shares
            .iter()
            .map(|s| shamir::Share::from_bytes(&s.share_data).map_err(from_security_error))
            .collect::<Result<Vec<shamir::Share>>>()?;
        let payload =
            shamir::reconstruct(&parsed, self.config.threshold).map_err(from_security_error)?;
        deserialize_tensor_from_sharing(&payload)
    }
}

/// Serialize a tensor as `rank ‖ dims ‖ little-endian f32 values` for sharing.
fn serialize_tensor_for_sharing(tensor: &Tensor) -> Result<Vec<u8>> {
    let shape = tensor.shape().to_vec();
    let values = tensor.to_vec_f32()?;
    let rank = u32::try_from(shape.len())
        .map_err(|_| CoreError::InvalidInput("Tensor rank does not fit in u32".to_string()))?;

    let mut out = Vec::with_capacity(4 + shape.len() * 8 + values.len() * 4);
    out.extend_from_slice(&rank.to_le_bytes());
    for dim in &shape {
        let dim64 = u64::try_from(*dim).map_err(|_| {
            CoreError::InvalidInput("Tensor dimension does not fit in u64".to_string())
        })?;
        out.extend_from_slice(&dim64.to_le_bytes());
    }
    for value in &values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

/// Inverse of [`serialize_tensor_for_sharing`].
fn deserialize_tensor_from_sharing(payload: &[u8]) -> Result<Tensor> {
    fn invalid_input(message: String) -> CoreError {
        CoreError::InvalidInput(message)
    }

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
    Tensor::from_vec(values, &shape).map_err(from_security_error)
}

/// Homomorphic encryption of secret shares, backed by real Paillier.
///
/// The previous implementation returned `vec![0u8; 64]` regardless of input.
pub struct HomomorphicEncryption {
    config: HomomorphicEncryptionConfig,
    keypair: paillier::PaillierKeypair,
}

impl HomomorphicEncryption {
    /// Create the engine, generating a real Paillier keypair.
    ///
    /// Key generation is slow (seconds at production modulus sizes); construct
    /// once and reuse.
    ///
    /// # Errors
    /// Returns [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation)
    /// for a fully-homomorphic scheme this crate does not implement.
    pub fn new(config: HomomorphicEncryptionConfig) -> Result<Self> {
        let modulus_bits = Self::modulus_bits(&config)?;
        Ok(Self {
            keypair: paillier::generate_keypair(modulus_bits)?,
            config,
        })
    }

    /// Create with a caller-supplied CSPRNG (deterministic, fast tests).
    ///
    /// # Errors
    /// See [`Self::new`].
    pub fn new_with_rng<R: rand_core::CryptoRng>(
        config: HomomorphicEncryptionConfig,
        rng: &mut R,
    ) -> Result<Self> {
        let modulus_bits = Self::modulus_bits(&config)?;
        Ok(Self {
            keypair: paillier::generate_keypair_with_rng(modulus_bits, rng)?,
            config,
        })
    }

    /// Build an engine around an already-generated keypair.
    ///
    /// Test-only: real callers go through [`Self::new`], which derives the
    /// modulus size from the configured security level.
    #[cfg(test)]
    pub(crate) fn from_parts(
        config: HomomorphicEncryptionConfig,
        keypair: paillier::PaillierKeypair,
    ) -> Self {
        Self { keypair, config }
    }

    /// Reject the schemes with no implementation and map the security level
    /// onto a Paillier modulus size.
    fn modulus_bits(config: &HomomorphicEncryptionConfig) -> Result<u64> {
        match config.scheme {
            HomomorphicScheme::Paillier => {},
            HomomorphicScheme::BFV
            | HomomorphicScheme::CKKS
            | HomomorphicScheme::BGV
            | HomomorphicScheme::TFHE => {
                return Err(from_security_error(
                    paillier::unsupported_paillier_operation(&format!(
                        "fully homomorphic scheme {:?}",
                        config.scheme
                    )),
                ));
            },
        }
        Ok(match config.security_level {
            0..=128 => 3072,
            129..=192 => 7680,
            _ => 15360,
        })
    }

    /// The engine configuration.
    pub fn config(&self) -> &HomomorphicEncryptionConfig {
        &self.config
    }

    /// The Paillier public key. Safe to publish.
    pub fn public_key(&self) -> &paillier::PaillierPublicKey {
        self.keypair.public_key_ref()
    }

    /// Encrypt each share's bytes under Paillier.
    ///
    /// A Paillier plaintext must be smaller than the modulus `n`, but a share
    /// of a large tensor is not, so each share is split into fixed-size blocks
    /// (one byte short of the modulus, so every block is guaranteed to be in
    /// range) and each block is encrypted separately. The result is a vector of
    /// genuine ciphertexts, not a zero-filled placeholder.
    ///
    /// # Errors
    /// Returns an error if the modulus is too small to hold even one block.
    pub async fn encrypt_shares(&self, shares: &[SecretShare]) -> Result<Vec<EncryptedShare>> {
        let mut rng = rand_core::UnwrapErr(getrandom::SysRng);
        self.encrypt_shares_with_rng(shares, &mut rng)
    }

    /// Encrypt with a caller-supplied CSPRNG.
    ///
    /// # Errors
    /// See [`Self::encrypt_shares`].
    pub fn encrypt_shares_with_rng<R: rand_core::CryptoRng>(
        &self,
        shares: &[SecretShare],
        rng: &mut R,
    ) -> Result<Vec<EncryptedShare>> {
        let public = self.keypair.public_key_ref();
        let block_len = Self::block_len(public)?;

        shares
            .iter()
            .map(|share| {
                let mut blocks = Vec::new();
                for chunk in share.share_data.chunks(block_len) {
                    // A `block_len`-byte big-endian integer is strictly below
                    // `n` because `block_len` is one byte short of `n`'s size.
                    let message = num_bigint::BigUint::from_bytes_be(chunk);
                    let ciphertext =
                        public.encrypt_with_rng(&message, rng).map_err(from_security_error)?;
                    blocks.push(ciphertext.to_bytes_be());
                }
                Ok(EncryptedShare {
                    ciphertext_blocks: blocks,
                    plaintext_len: share.share_data.len(),
                    public_key_id: format!("paillier-{}", public.bits()),
                    share_id: share.share_id,
                })
            })
            .collect()
    }

    /// Bytes of plaintext per Paillier block: one byte short of the modulus, so
    /// every block is guaranteed to be strictly less than `n`.
    fn block_len(public: &paillier::PaillierPublicKey) -> Result<usize> {
        let modulus_bytes = (public.bits() as usize).div_ceil(8);
        modulus_bytes.checked_sub(1).filter(|len| *len > 0).ok_or_else(|| {
            CoreError::InvalidInput(format!(
                "Paillier modulus of {} bits is too small to encrypt any data",
                public.bits()
            ))
        })
    }

    /// Decrypt an encrypted share back to its raw bytes.
    ///
    /// Reverses the block split performed by [`Self::encrypt_shares_with_rng`],
    /// restoring the leading zero bytes that big-integer encoding drops.
    ///
    /// # Errors
    /// Returns an error for a malformed ciphertext or an inconsistent length.
    pub fn decrypt_share(&self, share: &EncryptedShare) -> Result<Vec<u8>> {
        let public = self.keypair.public_key_ref();
        let block_len = Self::block_len(public)?;
        let block_count = share.ciphertext_blocks.len();

        let mut plaintext = Vec::with_capacity(share.plaintext_len);
        for (index, block) in share.ciphertext_blocks.iter().enumerate() {
            let ciphertext = paillier::PaillierCiphertext::from_bytes_be(block);
            let decrypted = self
                .keypair
                .private_key_ref()
                .decrypt(&ciphertext)
                .map_err(from_security_error)?
                .to_bytes_be();

            // Every block but the last was exactly `block_len` bytes of
            // plaintext; the big-integer encoding drops leading zeros, so pad
            // them back.
            let expected = if index + 1 == block_count {
                share.plaintext_len - block_len * (block_count - 1)
            } else {
                block_len
            };
            if decrypted.len() > expected {
                return Err(CoreError::InvalidInput(format!(
                    "Paillier block {index} decrypted to {} bytes, expected at most {expected}",
                    decrypted.len()
                )));
            }
            plaintext.resize(plaintext.len() + (expected - decrypted.len()), 0);
            plaintext.extend_from_slice(&decrypted);
        }

        if plaintext.len() != share.plaintext_len {
            return Err(CoreError::InvalidInput(format!(
                "Decrypted share is {} bytes, expected {}",
                plaintext.len(),
                share.plaintext_len
            )));
        }
        Ok(plaintext)
    }
}

/// Zero-knowledge proofs of correct privatization, backed by a real Schnorr
/// sigma protocol.
///
/// The previous implementation returned `proof_data: vec![0u8; 128]`.
pub struct ZeroKnowledgeProofs {
    config: ZeroKnowledgeConfig,
    keypair: zkp::SchnorrKeypair,
}

impl ZeroKnowledgeProofs {
    /// Create the engine, generating a real Schnorr keypair.
    ///
    /// # Errors
    /// Returns [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation)
    /// for a circuit proof system this crate does not implement.
    pub fn new(config: ZeroKnowledgeConfig) -> Result<Self> {
        Self::require_supported(&config.proof_system)?;
        Ok(Self {
            config,
            keypair: zkp::SchnorrKeypair::generate(),
        })
    }

    /// Create with a caller-supplied CSPRNG.
    ///
    /// # Errors
    /// See [`Self::new`].
    pub fn new_with_rng<R: rand_core::CryptoRng>(
        config: ZeroKnowledgeConfig,
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
            ZKProofSystem::Groth16
            | ZKProofSystem::PLONK
            | ZKProofSystem::Bulletproofs
            | ZKProofSystem::STARK
            | ZKProofSystem::Marlin => Err(from_security_error(zkp::unsupported_proof_system(
                &format!("{system:?}"),
            ))),
        }
    }

    /// The engine configuration.
    pub fn config(&self) -> &ZeroKnowledgeConfig {
        &self.config
    }

    /// Prove that this client performed the privatization step, binding the
    /// proof to a transcript of the original and privatized updates plus the
    /// budget actually spent.
    ///
    /// The proof is a real Schnorr transcript; it does not contain the witness,
    /// and it does not verify against a different transcript.
    ///
    /// # Errors
    /// Returns an error if either tensor cannot be read.
    pub async fn generate_correctness_proof(
        &self,
        original: &Tensor,
        privatized: &Tensor,
        budget: &BudgetAllocation,
    ) -> Result<ZKProof> {
        let mut rng = rand_core::UnwrapErr(getrandom::SysRng);
        self.generate_correctness_proof_with_rng(original, privatized, budget, &mut rng)
    }

    /// Prove with a caller-supplied CSPRNG.
    ///
    /// # Errors
    /// See [`Self::generate_correctness_proof`].
    pub fn generate_correctness_proof_with_rng<R: rand_core::CryptoRng>(
        &self,
        original: &Tensor,
        privatized: &Tensor,
        budget: &BudgetAllocation,
        rng: &mut R,
    ) -> Result<ZKProof> {
        let context = Self::transcript(original, privatized, budget)?;
        let proof = self.keypair.prove_with_rng(&context, rng);
        Ok(ZKProof {
            proof_data: proof.to_bytes(),
            verification_key: self.keypair.public_value().to_bytes_be(),
        })
    }

    /// Verify a proof against the same transcript.
    ///
    /// # Errors
    /// Returns an error if either tensor cannot be read.
    pub fn verify_correctness_proof(
        &self,
        proof: &ZKProof,
        original: &Tensor,
        privatized: &Tensor,
        budget: &BudgetAllocation,
    ) -> Result<bool> {
        let context = Self::transcript(original, privatized, budget)?;
        let Ok(parsed) = zkp::SchnorrProof::from_bytes(&proof.proof_data) else {
            return Ok(false);
        };
        Ok(self.keypair.verifier().verify(&parsed, &context))
    }

    /// Build the public transcript the proof is bound to.
    ///
    /// Hashing (rather than embedding) the tensors keeps the context small and
    /// avoids publishing the updates themselves.
    fn transcript(
        original: &Tensor,
        privatized: &Tensor,
        budget: &BudgetAllocation,
    ) -> Result<Vec<u8>> {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"trustformers-mobile/privacy/correctness/v1");
        for tensor in [original, privatized] {
            for value in tensor.to_vec_f32()? {
                hasher.update(value.to_le_bytes());
            }
            hasher.update(b"|");
        }
        hasher.update(budget.differential_privacy.epsilon.to_le_bytes());
        hasher.update(budget.differential_privacy.delta.to_le_bytes());
        Ok(hasher.finalize().to_vec())
    }
}

/// Private information retrieval.
///
/// The previous implementation returned `Tensor::zeros(&[1, 1])` together with
/// invented privacy costs. No real PIR scheme is implemented in this crate, so
/// this now reports that honestly instead.
pub struct PrivateInformationRetrieval {
    config: PrivateRetrievalConfig,
}

impl PrivateInformationRetrieval {
    /// Create the engine.
    ///
    /// # Errors
    /// Currently infallible; the `Result` is kept for API stability.
    pub fn new(config: PrivateRetrievalConfig) -> Result<Self> {
        Ok(Self { config })
    }

    /// The engine configuration.
    pub fn config(&self) -> &PrivateRetrievalConfig {
        &self.config
    }

    /// Retrieve a model update without revealing which one was requested.
    ///
    /// # Errors
    /// Always returns
    /// [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation):
    /// no PIR scheme is implemented. Returning a zero tensor and a fabricated
    /// privacy cost — what this used to do — is worse than failing, because a
    /// caller would believe its query pattern was hidden when it was not.
    pub async fn retrieve_model_update(
        &self,
        query: &ModelQuery,
    ) -> Result<PrivateRetrievalResult> {
        Err(unsupported(
            format!(
                "private information retrieval for model {} (scheme {:?})",
                query.model_id, self.config.scheme
            ),
            "trustformers-mobile (no PIR scheme is implemented; use the homomorphic, secret \
             sharing or differential privacy engines instead)"
                .to_string(),
        ))
    }
}

/// Private federated analytics: real statistics under real differential
/// privacy.
///
/// The previous implementation returned the constants `mean: 0.5,
/// variance: 0.1` with hardcoded confidence intervals regardless of input.
pub struct PrivateFederatedAnalytics {
    config: FederatedAnalyticsConfig,
}

impl PrivateFederatedAnalytics {
    /// Create the engine.
    ///
    /// # Errors
    /// Currently infallible; the `Result` is kept for API stability.
    pub fn new(config: FederatedAnalyticsConfig) -> Result<Self> {
        Ok(Self { config })
    }

    /// The engine configuration.
    pub fn config(&self) -> &FederatedAnalyticsConfig {
        &self.config
    }

    /// Compute differentially-private statistics over `data`.
    ///
    /// Each requested statistic is computed from the real samples and then
    /// perturbed by a real Laplace mechanism calibrated to the configured
    /// per-statistic budget. The reported privacy cost is the sum of what was
    /// actually spent, and the confidence intervals are derived from the
    /// mechanism's own noise distribution rather than invented.
    ///
    /// # Errors
    /// Returns an error for empty input, an unsupported statistic, or an
    /// invalid budget.
    pub async fn compute_analytics(
        &self,
        data: &[Tensor],
        analytics: &[AnalyticsType],
    ) -> Result<PrivateAnalyticsResult> {
        let mut rng = rand_core::UnwrapErr(getrandom::SysRng);
        self.compute_analytics_with_rng(data, analytics, &mut rng)
    }

    /// Compute with a caller-supplied CSPRNG.
    ///
    /// # Errors
    /// See [`Self::compute_analytics`].
    pub fn compute_analytics_with_rng<R: rand_core::CryptoRng>(
        &self,
        data: &[Tensor],
        analytics: &[AnalyticsType],
        rng: &mut R,
    ) -> Result<PrivateAnalyticsResult> {
        let invalid_input = CoreError::InvalidInput;

        if data.is_empty() {
            return Err(invalid_input(
                "Cannot compute analytics over an empty sample set".to_string(),
            ));
        }

        // Flatten every sample into one population.
        let mut population = Vec::new();
        for tensor in data {
            population.extend(tensor.to_vec_f32()?);
        }
        if population.is_empty() {
            return Err(invalid_input(
                "Cannot compute analytics over empty tensors".to_string(),
            ));
        }

        let clipping = self.config.statistics.clipping_bounds.clone();
        let range = clipping.upper_bound - clipping.lower_bound;
        if range <= 0.0 || !range.is_finite() {
            return Err(invalid_input(format!(
                "Analytics clipping bounds must satisfy lower < upper, got [{}, {}]",
                clipping.lower_bound, clipping.upper_bound
            )));
        }
        // Clipping to a known range is what bounds the sensitivity.
        let clipped: Vec<f64> = population
            .iter()
            .map(|v| f64::from(*v).clamp(clipping.lower_bound, clipping.upper_bound))
            .collect();
        let n = clipped.len() as f64;

        let epsilon_each = self.config.statistics.privacy_budget_per_statistic;
        let mut statistics = HashMap::new();
        let mut confidence_intervals = HashMap::new();
        let mut accountant = dp::PrivacyAccountant::new();

        for analytic in analytics {
            // `BasicStatistics` expands to the individual statistics; the other
            // analytic families are not implemented and are reported as such
            // rather than faked.
            let requested: Vec<(&str, f64, f64)> = match analytic {
                AnalyticsType::BasicStatistics => {
                    let mean = clipped.iter().sum::<f64>() / n;
                    let variance = clipped.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                    vec![
                        // Changing one record moves the mean by at most range/n.
                        ("mean", mean, range / n),
                        // ...the variance by at most range^2/n.
                        ("variance", variance, range.powi(2) / n),
                        // Under *unbounded* DP (add/remove a record) the count
                        // changes by exactly 1. Note this is the add/remove
                        // neighbouring relation; under bounded DP (replace one
                        // record) the count is invariant and its sensitivity
                        // would be 0. We use the conservative value.
                        ("count", n, 1.0),
                        // ...and the sum by at most the clipping range.
                        ("sum", clipped.iter().sum::<f64>(), range),
                    ]
                },
                other => {
                    return Err(unsupported(
                        format!("private analytic {other:?}"),
                        "trustformers-mobile (only AnalyticsType::BasicStatistics is implemented)",
                    ));
                },
            };

            for (name, true_value, sensitivity) in requested {
                let mechanism = dp::LaplaceMechanism::new(sensitivity, epsilon_each)
                    .map_err(from_security_error)?;
                let noised = mechanism
                    .privatize_with_rng(&[true_value as f32], rng)
                    .map_err(from_security_error)?;
                let noised_value = f64::from(noised[0]);
                statistics.insert(name.to_string(), noised_value);

                // A 95% interval for Laplace(b) noise: +/- b * ln(1/0.05).
                let half_width = mechanism.scale() * (1.0f64 / 0.05).ln();
                confidence_intervals.insert(
                    name.to_string(),
                    (noised_value - half_width, noised_value + half_width),
                );
                accountant.record(mechanism.budget());
            }
        }

        let (spent, _kind) = accountant.total_spent(1e-6);
        let statistic_count = statistics.len();
        Ok(PrivateAnalyticsResult {
            statistics,
            privacy_cost: PrivacyCost {
                differential_privacy_cost: spent.epsilon,
                // Not measured by this call; the caller measures transport and
                // compute cost where it can, rather than this inventing one.
                computational_cost: 0,
                communication_cost: 0,
                sample_count: clipped.len(),
                statistic_count,
            },
            confidence_intervals,
        })
    }
}

/// Post-quantum transport security using real ML-KEM-768 and ML-DSA-65.
///
/// The previous implementation returned `encrypted_payload: vec![0u8; 256],
/// quantum_signature: vec![0u8; 64]` regardless of input.
pub struct PostQuantumCryptography {
    config: PostQuantumConfig,
    kem: crate::advanced_security::pqc::KyberKem,
    signer: crate::advanced_security::pqc::DilithiumSigner,
}

impl PostQuantumCryptography {
    /// Create the engine, generating real ML-KEM and ML-DSA keypairs.
    ///
    /// # Errors
    /// Returns [`UnsupportedOperation`](trustformers_core::errors::ErrorKind::UnsupportedOperation)
    /// for an algorithm with no vetted pure-Rust implementation.
    pub fn new(config: PostQuantumConfig) -> Result<Self> {
        let mut rng = rand_core::UnwrapErr(getrandom::SysRng);
        Self::new_with_rng(config, &mut rng)
    }

    /// Create with a caller-supplied CSPRNG.
    ///
    /// # Errors
    /// See [`Self::new`].
    pub fn new_with_rng<R: rand_core::CryptoRng>(
        config: PostQuantumConfig,
        rng: &mut R,
    ) -> Result<Self> {
        use crate::advanced_security::pqc;
        match config.kem {
            KEMAlgorithm::MlKem768 => {},
            KEMAlgorithm::NTRU => return Err(from_security_error(pqc::unsupported("NTRU"))),
            KEMAlgorithm::SABER => return Err(from_security_error(pqc::unsupported("SABER"))),
            KEMAlgorithm::FrodoKEM => {
                return Err(from_security_error(pqc::unsupported("FrodoKEM")))
            },
        }
        match config.signature {
            SignatureAlgorithm::MlDsa65 => {},
            SignatureAlgorithm::Falcon => {
                return Err(from_security_error(pqc::unsupported("Falcon")))
            },
            SignatureAlgorithm::Rainbow => {
                return Err(from_security_error(pqc::unsupported(
                    "Rainbow (cryptographically broken by Beullens' attack, 2022)",
                )))
            },
        }
        Ok(Self {
            config,
            kem: pqc::KyberKem::generate_from_rng(rng),
            signer: pqc::DilithiumSigner::generate_from_rng(rng),
        })
    }

    /// The engine configuration.
    pub fn config(&self) -> &PostQuantumConfig {
        &self.config
    }

    /// Encrypt and sign a private payload for transmission.
    ///
    /// The payload is serialized, encrypted under real ML-KEM-768 + AEAD, and
    /// the *ciphertext* is signed with real ML-DSA-65 (sign-then-encrypt would
    /// leak the signature).
    ///
    /// # Errors
    /// Returns an error if encryption fails.
    pub async fn secure_transmission(&self, data: &PrivateData) -> Result<PostQuantumSecuredData> {
        let payload = Self::serialize(data);
        let encrypted_payload = self.kem.encrypt(&payload, b"trustformers-mobile/pq-transport")?;
        let quantum_signature = self.signer.sign(&encrypted_payload);
        Ok(PostQuantumSecuredData {
            encrypted_payload,
            quantum_signature,
        })
    }

    /// Verify and decrypt a payload produced by [`Self::secure_transmission`].
    ///
    /// # Errors
    /// Returns an error for an invalid signature or a tampered ciphertext.
    pub fn open_transmission(&self, secured: &PostQuantumSecuredData) -> Result<Vec<u8>> {
        if !self.signer.verify(&secured.encrypted_payload, &secured.quantum_signature) {
            return Err(CoreError::InvalidInput(
                "Post-quantum signature verification failed".to_string(),
            ));
        }
        self.kem
            .decrypt(
                &secured.encrypted_payload,
                b"trustformers-mobile/pq-transport",
            )
            .map_err(from_security_error)
    }

    /// Length-prefixed serialization of the private payload.
    pub(crate) fn serialize(data: &PrivateData) -> Vec<u8> {
        let mut out = Vec::new();
        let share_count = u32::try_from(data.mpc_shares.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&share_count.to_le_bytes());
        for share in &data.mpc_shares {
            let len = u32::try_from(share.share_data.len()).unwrap_or(u32::MAX);
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&share.share_data);
        }
        let proof_len = u32::try_from(data.zk_proof.proof_data.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&proof_len.to_le_bytes());
        out.extend_from_slice(&data.zk_proof.proof_data);
        out
    }
}

/// Adaptive privacy budgeting with real per-client accounting.
///
/// The previous implementation returned a constant `epsilon: 0.1,
/// delta: 1e-5` allocation regardless of client or requested cost.
pub struct AdaptivePrivacyBudgeting {
    config: AdaptiveBudgetingConfig,
    /// Remaining budget per client, in units of epsilon.
    remaining: Mutex<HashMap<String, f64>>,
}

impl AdaptivePrivacyBudgeting {
    /// Create the engine.
    ///
    /// # Errors
    /// Returns an error for a non-positive initial budget.
    pub fn new(config: AdaptiveBudgetingConfig) -> Result<Self> {
        if !config.initial_budget.is_finite() || config.initial_budget <= 0.0 {
            return Err(CoreError::InvalidInput(format!(
                "Initial privacy budget must be finite and positive, got {}",
                config.initial_budget
            )));
        }
        Ok(Self {
            config,
            remaining: Mutex::new(HashMap::new()),
        })
    }

    /// The engine configuration.
    pub fn config(&self) -> &AdaptiveBudgetingConfig {
        &self.config
    }

    /// Remaining epsilon for `client_id`.
    pub fn remaining_budget(&self, client_id: &str) -> f64 {
        let guard = self.remaining.lock().unwrap_or_else(|p| p.into_inner());
        guard.get(client_id).copied().unwrap_or(self.config.initial_budget)
    }

    /// Allocate budget for one round, deducting it from the client's remaining
    /// allowance.
    ///
    /// # Errors
    /// Returns an error when the client has exhausted its budget — the old code
    /// handed out a fixed allocation forever, which meant the accounting was
    /// decorative.
    pub async fn allocate_budget(
        &self,
        client_id: &str,
        cost: &PrivacyCost,
    ) -> Result<BudgetAllocation> {
        let requested = cost.differential_privacy_cost;
        if !requested.is_finite() || requested <= 0.0 {
            return Err(CoreError::InvalidInput(format!(
                "Requested privacy cost must be finite and positive, got {requested}"
            )));
        }

        let mut guard = self.remaining.lock().unwrap_or_else(|p| p.into_inner());
        let remaining = guard.entry(client_id.to_string()).or_insert(self.config.initial_budget);

        // Fairness cap: never hand out more than the configured per-client
        // maximum in a single round.
        let capped = requested.min(self.config.fairness_constraints.max_budget_per_client);

        if *remaining < capped {
            return Err(CoreError::ResourceExhausted(format!(
                "Client {client_id} has {remaining:.6} epsilon remaining but {capped:.6} was \
                 requested; the privacy budget is exhausted"
            )));
        }
        *remaining -= capped;

        Ok(BudgetAllocation {
            differential_privacy: DifferentialPrivacyBudget {
                epsilon: capped,
                delta: self.config.renewal.amount.min(1e-5),
                renyi_alpha: 2.0,
            },
            computational_budget: cost.computational_cost as f64,
            communication_budget: cost.communication_cost as f64,
        })
    }
}

/// Measured performance of the privacy pipeline.
///
/// The previous implementation returned the constants
/// `average_execution_time: 500ms, throughput: 100.0, privacy_efficiency: 0.85,
/// resource_utilization: 0.70`. This one records real durations and derives the
/// metrics from them, and reports [`PrivacyPerformanceMetrics::NotAvailable`]
/// when nothing has been measured yet.
#[derive(Debug, Default)]
pub struct PrivacyPerformanceMonitor {
    samples: Mutex<Vec<Duration>>,
}

impl PrivacyPerformanceMonitor {
    /// Create a monitor with no samples.
    ///
    /// # Errors
    /// Currently infallible; the `Result` is kept for API stability.
    pub fn new() -> Result<Self> {
        Ok(Self::default())
    }

    /// Record one measured operation duration.
    pub fn record(&self, elapsed: Duration) {
        self.samples.lock().unwrap_or_else(|p| p.into_inner()).push(elapsed);
    }

    /// How many measurements have been recorded.
    pub fn sample_count(&self) -> usize {
        self.samples.lock().unwrap_or_else(|p| p.into_inner()).len()
    }

    /// Metrics derived from the recorded measurements.
    ///
    /// # Errors
    /// Currently infallible; the `Result` is kept for API stability.
    pub async fn get_metrics(&self) -> Result<PrivacyPerformanceMetrics> {
        let guard = self.samples.lock().unwrap_or_else(|p| p.into_inner());
        if guard.is_empty() {
            return Ok(PrivacyPerformanceMetrics::NotAvailable {
                reason: "no privacy operations have been measured yet".to_string(),
            });
        }
        let total: Duration = guard.iter().sum();
        let count = guard.len() as u32;
        let average = total / count;
        let average_secs = average.as_secs_f64();
        Ok(PrivacyPerformanceMetrics::Measured {
            average_execution_time: average,
            slowest: guard.iter().copied().max().unwrap_or(average),
            fastest: guard.iter().copied().min().unwrap_or(average),
            throughput_per_second: if average_secs > 0.0 {
                1.0 / average_secs
            } else {
                f64::INFINITY
            },
            sample_count: guard.len(),
        })
    }
}
