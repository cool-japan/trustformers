//! Core types and data structures for federated learning v2.0
//!
//! This module contains all the fundamental types, enums, and data structures
//! used throughout the federated learning system, including privacy mechanisms,
//! protocols, device capabilities, and participant information.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::{Tensor};

/// Privacy mechanisms for differential privacy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyMechanism {
    /// Gaussian mechanism
    Gaussian,
    /// Laplace mechanism
    Laplace,
    /// Exponential mechanism
    Exponential,
    /// Above threshold mechanism
    AboveThreshold,
    /// Sparse vector technique
    SparseVector,
    /// Private aggregation of teacher ensembles (PATE)
    PATE,
    /// Renyi differential privacy
    RenyiDP,
    /// Zero-concentrated differential privacy
    ZCDP,
}

/// Privacy models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyModel {
    /// Local differential privacy
    LocalDP,
    /// Central differential privacy
    CentralDP,
    /// Shuffled model
    ShuffledModel,
    /// Hybrid model
    HybridModel,
}

/// Advanced composition methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositionMethod {
    /// Basic composition
    Basic,
    /// Advanced composition
    Advanced,
    /// Optimal composition
    Optimal,
    /// Renyi differential privacy composition
    RenyiComposition,
    /// Zero-concentrated differential privacy composition
    ZCDPComposition,
    /// Privacy loss distribution tracking
    PLDTracking,
}

/// Sampling methods for privacy amplification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingMethod {
    /// Uniform random sampling
    UniformRandom,
    /// Poisson sampling
    Poisson,
    /// Systematic sampling
    Systematic,
    /// Stratified sampling
    Stratified,
}

/// Secure aggregation protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecureAggregationProtocol {
    /// Basic secure aggregation
    BasicSecureAggregation,
    /// Federated secure aggregation
    FederatedSecureAggregation,
    /// Private federated learning
    PrivateFederatedLearning,
    /// SecAgg+ protocol
    SecAggPlus,
    /// Flamingo protocol
    Flamingo,
    /// FATE protocol
    FATE,
}

/// Homomorphic encryption schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HomomorphicScheme {
    /// BFV scheme
    BFV,
    /// CKKS scheme
    CKKS,
    /// BGV scheme
    BGV,
    /// TFHE scheme
    TFHE,
}

/// Optimization levels for homomorphic encryption
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Advanced optimization
    Advanced,
    /// Maximum optimization
    Maximum,
}

/// MPC protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MPCProtocol {
    /// Shamir's secret sharing
    ShamirSecretSharing,
    /// BGW protocol
    BGW,
    /// GMW protocol
    GMW,
    /// SPDZ protocol
    SPDZ,
    /// ABY protocol
    ABY,
    /// CrypTFlow protocol
    CrypTFlow,
}

/// Digital signature schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DigitalSignatureScheme {
    /// ECDSA
    ECDSA,
    /// EdDSA
    EdDSA,
    /// RSA-PSS
    RSAPSS,
    /// BLS signatures
    BLS,
    /// Ring signatures
    RingSignature,
}

/// Key exchange protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyExchangeProtocol {
    /// ECDH
    ECDH,
    /// X25519
    X25519,
    /// CRYSTALS-Kyber (post-quantum)
    Kyber,
    /// NTRU (post-quantum)
    NTRU,
}

/// Zero-knowledge proof systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZKProofSystem {
    /// zk-SNARKs
    ZkSNARKs,
    /// zk-STARKs
    ZkSTARKs,
    /// Bulletproofs
    Bulletproofs,
    /// PLONK
    PLONK,
}

/// Communication protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommunicationProtocol {
    /// HTTP/HTTPS
    HTTP,
    /// gRPC
    GRPC,
    /// WebRTC
    WebRTC,
    /// Custom TCP
    CustomTCP,
    /// Message queue (MQTT/RabbitMQ)
    MessageQueue,
}

/// Compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// GZIP compression
    GZIP,
    /// LZ4 compression
    LZ4,
    /// Brotli compression
    Brotli,
    /// Custom compression
    Custom,
}

/// Transport security protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransportSecurity {
    /// TLS 1.3
    TLS13,
    /// DTLS
    DTLS,
    /// Custom encryption
    CustomEncryption,
}

/// Privacy accounting methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyAccountingMethod {
    /// Moments accountant
    MomentsAccountant,
    /// Renyi DP accountant
    RenyiDPAccountant,
    /// Privacy loss distribution accountant
    PLDAccountant,
    /// Gaussian DP accountant
    GaussianDPAccountant,
}

/// Attack types for detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttackType {
    /// Model poisoning attack
    ModelPoisoning,
    /// Byzantine attack
    Byzantine,
    /// Gradient inversion attack
    GradientInversion,
    /// Membership inference attack
    MembershipInference,
    /// Property inference attack
    PropertyInference,
    /// Backdoor attack
    Backdoor,
}

/// Countermeasures against attacks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Countermeasure {
    /// Reject the update
    UpdateRejection,
    /// Apply additional noise
    AdditionalNoise,
    /// Reduce participant weight
    WeightReduction,
    /// Temporary participant exclusion
    TemporaryExclusion,
    /// Permanent participant ban
    PermanentBan,
}

/// Device compute capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeCapability {
    /// Low compute capability
    Low,
    /// Medium compute capability
    Medium,
    /// High compute capability
    High,
    /// Very high compute capability
    VeryHigh,
}

/// Training progress states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingProgress {
    /// Initializing
    Initializing,
    /// Training in progress
    Training,
    /// Converged
    Converged,
    /// Failed
    Failed,
    /// Stopped
    Stopped,
}

/// Bandwidth adaptation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BandwidthAdaptationStrategy {
    /// Conservative adaptation
    Conservative,
    /// Aggressive adaptation
    Aggressive,
    /// Hybrid adaptation
    Hybrid,
}

/// Network congestion states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CongestionState {
    /// No congestion
    NoCongestion,
    /// Light congestion
    LightCongestion,
    /// Moderate congestion
    ModerateCongestion,
    /// Heavy congestion
    HeavyCongestion,
}

/// Device capabilities for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Compute capability level
    pub compute_capability: ComputeCapability,
    /// Memory capacity in MB
    pub memory_capacity_mb: u64,
    /// Network bandwidth in Mbps
    pub network_bandwidth_mbps: f64,
    /// Battery level (0.0 to 1.0)
    pub battery_level: f64,
    /// Available storage in MB
    pub available_storage_mb: u64,
}

/// Participant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantInfo {
    /// Unique participant identifier
    pub id: String,
    /// Public key for cryptographic operations
    pub public_key: Vec<u8>,
    /// Trust score (0.0 to 1.0)
    pub trust_score: f64,
    /// Participation history
    pub participation_history: Vec<ParticipationRecord>,
    /// Device capabilities
    pub device_capabilities: DeviceCapabilities,
}

/// Participation record for history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipationRecord {
    /// Training round number
    pub round: u32,
    /// Participation timestamp
    pub timestamp: u64,
    /// Update quality score
    pub update_quality: f64,
    /// Communication latency in ms
    pub latency_ms: u64,
}

/// Attack detection event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackDetectionEvent {
    /// Timestamp of detection
    pub timestamp: u64,
    /// Participant ID involved
    pub participant_id: String,
    /// Type of attack detected
    pub attack_type: AttackType,
    /// Confidence score (0.0 to 1.0)
    pub confidence_score: f64,
    /// Applied countermeasures
    pub countermeasures: Vec<Countermeasure>,
    /// Additional details
    pub details: HashMap<String, String>,
}

/// Convergence metrics for training monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetrics {
    /// Current model accuracy
    pub current_accuracy: f64,
    /// Best accuracy achieved
    pub best_accuracy: f64,
    /// Current loss value
    pub current_loss: f64,
    /// Loss reduction rate
    pub loss_reduction_rate: f64,
    /// Gradient norm
    pub gradient_norm: f64,
    /// Stability score
    pub stability_score: f64,
}

/// Training round statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundStatistics {
    /// Round number
    pub round: u32,
    /// Number of participating clients
    pub participants: u32,
    /// Round duration in seconds
    pub duration_seconds: u64,
    /// Average update quality
    pub avg_update_quality: f64,
    /// Communication overhead in MB
    pub communication_overhead_mb: f64,
    /// Privacy budget consumed this round
    pub privacy_budget_consumed: f64,
}

/// Bandwidth monitoring information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthMonitor {
    /// Current bandwidth in Mbps
    pub current_bandwidth_mbps: f64,
    /// Bandwidth history
    pub bandwidth_history: std::collections::VecDeque<f64>,
    /// Adaptation strategy
    pub adaptation_strategy: BandwidthAdaptationStrategy,
    /// Current congestion state
    pub congestion_state: CongestionState,
}

/// Default implementations for easy configuration
impl Default for PrivacyMechanism {
    fn default() -> Self {
        Self::Gaussian
    }
}

impl Default for PrivacyModel {
    fn default() -> Self {
        Self::CentralDP
    }
}

impl Default for CompositionMethod {
    fn default() -> Self {
        Self::Advanced
    }
}

impl Default for SecureAggregationProtocol {
    fn default() -> Self {
        Self::FederatedSecureAggregation
    }
}

impl Default for HomomorphicScheme {
    fn default() -> Self {
        Self::CKKS
    }
}

impl Default for MPCProtocol {
    fn default() -> Self {
        Self::SPDZ
    }
}

impl Default for DigitalSignatureScheme {
    fn default() -> Self {
        Self::EdDSA
    }
}

impl Default for KeyExchangeProtocol {
    fn default() -> Self {
        Self::X25519
    }
}

impl Default for ZKProofSystem {
    fn default() -> Self {
        Self::ZkSNARKs
    }
}

impl Default for CommunicationProtocol {
    fn default() -> Self {
        Self::GRPC
    }
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        Self::LZ4
    }
}

impl Default for TransportSecurity {
    fn default() -> Self {
        Self::TLS13
    }
}

impl Default for PrivacyAccountingMethod {
    fn default() -> Self {
        Self::RenyiDPAccountant
    }
}

impl Default for ComputeCapability {
    fn default() -> Self {
        Self::Medium
    }
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            compute_capability: ComputeCapability::Medium,
            memory_capacity_mb: 4096,
            network_bandwidth_mbps: 50.0,
            battery_level: 0.8,
            available_storage_mb: 5120,
        }
    }
}

impl Default for TrainingProgress {
    fn default() -> Self {
        Self::Initializing
    }
}

impl Default for BandwidthAdaptationStrategy {
    fn default() -> Self {
        Self::Hybrid
    }
}

impl Default for CongestionState {
    fn default() -> Self {
        Self::NoCongestion
    }
}