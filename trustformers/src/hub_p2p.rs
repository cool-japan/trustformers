use crate::core::error::Result;
use crate::error::TrustformersError;
use crate::hub_differential::{EnhancedDeltaInfo, ModelVersion};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::sync::{broadcast, RwLock};
use tokio::time::{interval, timeout};

/// Peer-to-Peer model sharing infrastructure for TrustformeRS
/// Enables decentralized model distribution and collaborative model development

/// P2P network node information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeerInfo {
    pub peer_id: String,
    pub address: SocketAddr,
    pub public_key: String,
    pub last_seen: SystemTime,
    pub reputation: f64, // 0.0 to 1.0
    pub capabilities: PeerCapabilities,
    pub bandwidth: BandwidthInfo,
    pub models: Vec<ModelAdvertisement>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeerCapabilities {
    pub can_serve_models: bool,
    pub can_compute_diffs: bool,
    pub supported_algorithms: Vec<String>,
    pub max_model_size: u64,
    pub storage_capacity: u64,
    pub compute_power: f64, // Relative score
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BandwidthInfo {
    pub upload_mbps: f64,
    pub download_mbps: f64,
    pub latency_ms: u32,
    pub measured_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelAdvertisement {
    pub model_id: String,
    pub version: String,
    pub size: u64,
    pub checksum: String,
    pub availability: f64, // 0.0 to 1.0
    pub last_updated: SystemTime,
    pub metadata: HashMap<String, String>,
}

/// P2P protocol messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum P2PMessage {
    // Discovery
    PeerAnnouncement(PeerInfo),
    PeerDiscovery,
    PeerList(Vec<PeerInfo>),

    // Model sharing
    ModelRequest {
        model_id: String,
        version: String,
        requestor_id: String,
    },
    ModelResponse {
        model_id: String,
        version: String,
        available: bool,
        estimated_time: Duration,
        chunk_info: ChunkInfo,
    },
    ModelChunk {
        model_id: String,
        version: String,
        chunk_id: u32,
        data: Vec<u8>,
        checksum: String,
    },

    // Differential updates
    DeltaRequest {
        model_id: String,
        from_version: String,
        to_version: String,
        requestor_id: String,
    },
    DeltaResponse {
        model_id: String,
        from_version: String,
        to_version: String,
        delta_info: Option<EnhancedDeltaInfo>,
        available: bool,
    },

    // Network maintenance
    Heartbeat {
        peer_id: String,
        timestamp: SystemTime,
    },
    PingRequest {
        peer_id: String,
        timestamp: SystemTime,
    },
    PingResponse {
        peer_id: String,
        timestamp: SystemTime,
        latency: Duration,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkInfo {
    pub total_chunks: u32,
    pub chunk_size: u32,
    pub total_size: u64,
    pub checksums: Vec<String>,
}

/// P2P network configuration
#[derive(Debug, Clone)]
pub struct P2PConfig {
    pub listen_address: SocketAddr,
    pub discovery_port: u16,
    pub max_peers: usize,
    pub heartbeat_interval: Duration,
    pub peer_timeout: Duration,
    pub chunk_size: u32,
    pub max_concurrent_transfers: usize,
    pub enable_dht: bool,
    pub bootstrap_peers: Vec<SocketAddr>,
    pub reputation_threshold: f64,
    pub storage_path: PathBuf,
}

impl Default for P2PConfig {
    fn default() -> Self {
        Self {
            listen_address: "127.0.0.1:8080".parse().unwrap(),
            discovery_port: 8081,
            max_peers: 100,
            heartbeat_interval: Duration::from_secs(30),
            peer_timeout: Duration::from_secs(300),
            chunk_size: 1024 * 1024, // 1MB chunks
            max_concurrent_transfers: 5,
            enable_dht: true,
            bootstrap_peers: Vec::new(),
            reputation_threshold: 0.5,
            storage_path: PathBuf::from("./p2p_storage"),
        }
    }
}

/// Peer reputation tracker
#[derive(Debug)]
pub struct ReputationTracker {
    reputation_scores: HashMap<String, f64>,
    interaction_history: HashMap<String, Vec<ReputationEvent>>,
    decay_factor: f64,
}

#[derive(Debug, Clone)]
pub struct ReputationEvent {
    event_type: ReputationEventType,
    timestamp: SystemTime,
    score_delta: f64,
}

#[derive(Debug, Clone)]
pub enum ReputationEventType {
    SuccessfulTransfer,
    FailedTransfer,
    InvalidData,
    SlowResponse,
    FastResponse,
    Availability,
}

impl Default for ReputationTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ReputationTracker {
    pub fn new() -> Self {
        Self {
            reputation_scores: HashMap::new(),
            interaction_history: HashMap::new(),
            decay_factor: 0.95, // 5% decay over time
        }
    }

    pub fn get_reputation(&self, peer_id: &str) -> f64 {
        self.reputation_scores.get(peer_id).copied().unwrap_or(0.5)
    }

    pub fn update_reputation(&mut self, peer_id: &str, event: ReputationEvent) {
        let current_score = self.get_reputation(peer_id);
        let new_score = (current_score + event.score_delta).clamp(0.0, 1.0);

        self.reputation_scores.insert(peer_id.to_string(), new_score);
        self.interaction_history.entry(peer_id.to_string()).or_default().push(event);
    }

    pub fn decay_reputations(&mut self) {
        for score in self.reputation_scores.values_mut() {
            *score = (*score * self.decay_factor).max(0.0);
        }
    }
}

/// P2P node implementation
pub struct P2PNode {
    config: P2PConfig,
    peer_id: String,
    peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
    reputation: Arc<Mutex<ReputationTracker>>,
    message_sender: broadcast::Sender<P2PMessage>,
    message_receiver: broadcast::Receiver<P2PMessage>,
    models: Arc<RwLock<HashMap<String, ModelVersion>>>,
    active_transfers: Arc<RwLock<HashMap<String, TransferState>>>,
}

#[derive(Debug, Clone)]
pub struct TransferState {
    pub model_id: String,
    pub version: String,
    pub peer_id: String,
    pub progress: f64,
    pub start_time: SystemTime,
    pub chunks_received: HashSet<u32>,
    pub total_chunks: u32,
}

impl P2PNode {
    pub async fn new(config: P2PConfig) -> Result<Self> {
        let peer_id = Self::generate_peer_id();
        let (message_sender, message_receiver) = broadcast::channel(1000);

        tokio::fs::create_dir_all(&config.storage_path).await.map_err(|e| {
            TrustformersError::Io {
                message: format!("Failed to create storage directory: {}", e),
                path: None,
                suggestion: Some(
                    "Check if you have write permissions in the parent directory".to_string(),
                ),
            }
        })?;

        Ok(Self {
            config,
            peer_id,
            peers: Arc::new(RwLock::new(HashMap::new())),
            reputation: Arc::new(Mutex::new(ReputationTracker::new())),
            message_sender,
            message_receiver,
            models: Arc::new(RwLock::new(HashMap::new())),
            active_transfers: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn start(&mut self) -> Result<()> {
        // Start TCP listener for peer connections
        let tcp_listener = TcpListener::bind(&self.config.listen_address).await.map_err(|e| {
            TrustformersError::Network {
                message: format!("Failed to bind TCP listener: {}", e),
                url: None,
                status_code: None,
                suggestion: Some("Check if the port is already in use".to_string()),
                retry_recommended: false,
            }
        })?;

        // Start UDP socket for discovery
        let discovery_addr =
            SocketAddr::new(self.config.listen_address.ip(), self.config.discovery_port);
        let udp_socket =
            UdpSocket::bind(discovery_addr).await.map_err(|e| TrustformersError::Network {
                message: format!("Failed to bind UDP socket: {}", e),
                url: None,
                status_code: None,
                suggestion: Some("Check if the port is already in use".to_string()),
                retry_recommended: false,
            })?;

        // Spawn background tasks
        self.spawn_tcp_handler(tcp_listener).await;
        self.spawn_discovery_handler(udp_socket).await;
        self.spawn_heartbeat_task().await;
        self.spawn_reputation_decay_task().await;

        // Connect to bootstrap peers
        self.connect_to_bootstrap_peers().await?;

        println!(
            "P2P node {} started on {}",
            self.peer_id, self.config.listen_address
        );
        Ok(())
    }

    async fn spawn_tcp_handler(&self, listener: TcpListener) {
        let peers = self.peers.clone();
        let reputation = self.reputation.clone();
        let message_sender = self.message_sender.clone();
        let models = self.models.clone();
        let active_transfers = self.active_transfers.clone();
        let peer_id = self.peer_id.clone();

        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, addr)) => {
                        let peers = peers.clone();
                        let reputation = reputation.clone();
                        let message_sender = message_sender.clone();
                        let models = models.clone();
                        let active_transfers = active_transfers.clone();
                        let peer_id = peer_id.clone();

                        tokio::spawn(async move {
                            if let Err(e) = Self::handle_peer_connection(
                                stream,
                                addr,
                                peers,
                                reputation,
                                message_sender,
                                models,
                                active_transfers,
                                peer_id,
                            )
                            .await
                            {
                                eprintln!("Error handling peer connection: {}", e);
                            }
                        });
                    },
                    Err(e) => {
                        eprintln!("Failed to accept connection: {}", e);
                    },
                }
            }
        });
    }

    async fn spawn_discovery_handler(&self, socket: UdpSocket) {
        let peers = self.peers.clone();
        let peer_id = self.peer_id.clone();
        let message_sender = self.message_sender.clone();

        tokio::spawn(async move {
            let mut buf = [0u8; 4096];

            loop {
                match socket.recv_from(&mut buf).await {
                    Ok((len, addr)) => {
                        if let Ok(message) = serde_json::from_slice::<P2PMessage>(&buf[..len]) {
                            match message {
                                P2PMessage::PeerDiscovery => {
                                    // Respond with our peer info
                                    let our_info = Self::create_peer_info(&peer_id, addr);
                                    let response = P2PMessage::PeerAnnouncement(our_info);

                                    if let Ok(response_data) = serde_json::to_vec(&response) {
                                        let _ = socket.send_to(&response_data, addr).await;
                                    }
                                },
                                P2PMessage::PeerAnnouncement(peer_info) => {
                                    // Add peer to our list
                                    let mut peers_lock = peers.write().await;
                                    peers_lock.insert(peer_info.peer_id.clone(), peer_info);
                                },
                                _ => {},
                            }
                        }
                    },
                    Err(e) => {
                        eprintln!("Discovery error: {}", e);
                    },
                }
            }
        });
    }

    async fn spawn_heartbeat_task(&self) {
        let peers = self.peers.clone();
        let reputation = self.reputation.clone();
        let heartbeat_interval = self.config.heartbeat_interval;
        let peer_timeout = self.config.peer_timeout;
        let peer_id = self.peer_id.clone();

        tokio::spawn(async move {
            let mut interval = interval(heartbeat_interval);

            loop {
                interval.tick().await;

                // Send heartbeats to all peers
                let heartbeat = P2PMessage::Heartbeat {
                    peer_id: peer_id.clone(),
                    timestamp: SystemTime::now(),
                };

                // Clean up timed-out peers
                let mut peers_lock = peers.write().await;
                let now = SystemTime::now();

                peers_lock.retain(|_, peer| {
                    let time_since_seen =
                        now.duration_since(peer.last_seen).unwrap_or(Duration::from_secs(0));
                    time_since_seen < peer_timeout
                });

                drop(peers_lock);

                // Send heartbeat to remaining peers
                // (In a real implementation, this would send via TCP/UDP)
            }
        });
    }

    async fn spawn_reputation_decay_task(&self) {
        let reputation = self.reputation.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(3600)); // Hourly decay

            loop {
                interval.tick().await;
                let mut reputation_lock = reputation.lock().unwrap();
                reputation_lock.decay_reputations();
            }
        });
    }

    async fn handle_peer_connection(
        mut stream: TcpStream,
        addr: SocketAddr,
        peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
        reputation: Arc<Mutex<ReputationTracker>>,
        message_sender: broadcast::Sender<P2PMessage>,
        models: Arc<RwLock<HashMap<String, ModelVersion>>>,
        active_transfers: Arc<RwLock<HashMap<String, TransferState>>>,
        peer_id: String,
    ) -> Result<()> {
        let mut buffer = [0u8; 4096];

        loop {
            match timeout(Duration::from_secs(30), stream.read(&mut buffer)).await {
                Ok(Ok(0)) => break, // Connection closed
                Ok(Ok(n)) => {
                    if let Ok(message) = serde_json::from_slice::<P2PMessage>(&buffer[..n]) {
                        Self::handle_message(
                            message,
                            &mut stream,
                            &peers,
                            &reputation,
                            &message_sender,
                            &models,
                            &active_transfers,
                            &peer_id,
                        )
                        .await?;
                    }
                },
                Ok(Err(e)) => {
                    eprintln!("Read error: {}", e);
                    break;
                },
                Err(_) => {
                    // Timeout
                    break;
                },
            }
        }

        Ok(())
    }

    async fn handle_message(
        message: P2PMessage,
        stream: &mut TcpStream,
        peers: &Arc<RwLock<HashMap<String, PeerInfo>>>,
        reputation: &Arc<Mutex<ReputationTracker>>,
        message_sender: &broadcast::Sender<P2PMessage>,
        models: &Arc<RwLock<HashMap<String, ModelVersion>>>,
        active_transfers: &Arc<RwLock<HashMap<String, TransferState>>>,
        peer_id: &str,
    ) -> Result<()> {
        match message {
            P2PMessage::ModelRequest {
                model_id,
                version,
                requestor_id,
            } => {
                let models_lock = models.read().await;
                let model_key = format!("{}:{}", model_id, version);

                if let Some(model) = models_lock.get(&model_key) {
                    let chunk_info = ChunkInfo {
                        total_chunks: (model.file_size / 1024 / 1024) as u32 + 1, // 1MB chunks
                        chunk_size: 1024 * 1024,
                        total_size: model.file_size,
                        checksums: vec!["dummy".to_string()], // Would calculate real checksums
                    };

                    let response = P2PMessage::ModelResponse {
                        model_id: model_id.clone(),
                        version: version.clone(),
                        available: true,
                        estimated_time: Duration::from_secs(60),
                        chunk_info,
                    };

                    let response_data = serde_json::to_vec(&response).map_err(|e| {
                        TrustformersError::InvalidInput {
                            message: format!("Failed to serialize response: {}", e),
                            parameter: Some("response".to_string()),
                            expected: Some("serializable data".to_string()),
                            received: None,
                            suggestion: Some("Check if the response data is valid".to_string()),
                        }
                    })?;

                    stream.write_all(&response_data).await.map_err(|e| {
                        TrustformersError::Network {
                            message: format!("Failed to send response: {}", e),
                            url: None,
                            status_code: None,
                            suggestion: Some("Check network connection".to_string()),
                            retry_recommended: true,
                        }
                    })?;
                }
            },

            P2PMessage::DeltaRequest {
                model_id,
                from_version,
                to_version,
                requestor_id,
            } => {
                // Check if we can provide the delta
                let models_lock = models.read().await;
                let from_key = format!("{}:{}", model_id, from_version);
                let to_key = format!("{}:{}", model_id, to_version);

                let available =
                    models_lock.contains_key(&from_key) && models_lock.contains_key(&to_key);

                let response = P2PMessage::DeltaResponse {
                    model_id,
                    from_version,
                    to_version,
                    delta_info: None, // Would compute actual delta info
                    available,
                };

                let response_data =
                    serde_json::to_vec(&response).map_err(|e| TrustformersError::InvalidInput {
                        message: format!("Failed to serialize delta response: {}", e),
                        parameter: Some("delta_response".to_string()),
                        expected: Some("serializable data".to_string()),
                        received: None,
                        suggestion: Some("Check if the delta response data is valid".to_string()),
                    })?;

                stream.write_all(&response_data).await.map_err(|e| TrustformersError::Network {
                    message: format!("Failed to send delta response: {}", e),
                    url: None,
                    status_code: None,
                    suggestion: Some("Check network connection".to_string()),
                    retry_recommended: true,
                })?;
            },

            P2PMessage::PingRequest {
                peer_id: requester_id,
                timestamp,
            } => {
                let response = P2PMessage::PingResponse {
                    peer_id: peer_id.to_string(),
                    timestamp: SystemTime::now(),
                    latency: SystemTime::now().duration_since(timestamp).unwrap_or(Duration::ZERO),
                };

                let response_data =
                    serde_json::to_vec(&response).map_err(|e| TrustformersError::InvalidInput {
                        message: format!("Failed to serialize ping response: {}", e),
                        parameter: Some("ping_response".to_string()),
                        expected: Some("serializable data".to_string()),
                        received: None,
                        suggestion: Some("Check if the ping response data is valid".to_string()),
                    })?;

                stream.write_all(&response_data).await.map_err(|e| TrustformersError::Network {
                    message: format!("Failed to send ping response: {}", e),
                    url: None,
                    status_code: None,
                    suggestion: Some("Check network connection".to_string()),
                    retry_recommended: true,
                })?;
            },

            _ => {
                // Handle other message types
            },
        }

        Ok(())
    }

    async fn connect_to_bootstrap_peers(&self) -> Result<()> {
        for peer_addr in &self.config.bootstrap_peers {
            match TcpStream::connect(peer_addr).await {
                Ok(mut stream) => {
                    // Send peer discovery message
                    let discovery = P2PMessage::PeerDiscovery;
                    let data = serde_json::to_vec(&discovery).map_err(|e| {
                        TrustformersError::InvalidInput {
                            message: format!("Failed to serialize discovery: {}", e),
                            parameter: Some("discovery".to_string()),
                            expected: Some("serializable data".to_string()),
                            received: None,
                            suggestion: Some("Check if the discovery data is valid".to_string()),
                        }
                    })?;

                    if let Err(e) = stream.write_all(&data).await {
                        eprintln!("Failed to send discovery to {}: {}", peer_addr, e);
                    }
                },
                Err(e) => {
                    eprintln!("Failed to connect to bootstrap peer {}: {}", peer_addr, e);
                },
            }
        }
        Ok(())
    }

    pub async fn request_model(&self, model_id: &str, version: &str) -> Result<PathBuf> {
        // Find best peers that have the model
        let peers = self.find_peers_with_model(model_id, version).await;
        if peers.is_empty() {
            return Err(TrustformersError::Hub {
                message: format!("No peers found with model {}:{}", model_id, version),
                model_id: model_id.to_string(),
                endpoint: None,
                suggestion: Some(
                    "Try connecting to more bootstrap peers or wait for peer discovery".to_string(),
                ),
                recovery_actions: vec![],
            }
            .into());
        }

        // Select best peer based on reputation and bandwidth
        let best_peer = self.select_best_peer(&peers).await;

        // Start download
        self.download_model_from_peer(&best_peer, model_id, version).await
    }

    pub async fn share_model(
        &self,
        model_path: &Path,
        model_id: &str,
        version: &str,
    ) -> Result<()> {
        // Load model and add to our catalog
        let model_data = fs::read(model_path).await.map_err(|e| TrustformersError::Io {
            message: format!("Failed to read model file: {}", e),
            path: None,
            suggestion: Some("Check if the file exists and is readable".to_string()),
        })?;

        let model_version = ModelVersion {
            id: format!("{}:{}", model_id, version),
            model_id: model_id.to_string(),
            version: version.to_string(),
            parent_version: None,
            created_at: SystemTime::now(),
            file_hash: Self::calculate_hash(&model_data),
            file_size: model_data.len() as u64,
            compressed_size: None,
            description: None,
            changes: Vec::new(),
            metadata: HashMap::new(),
        };

        // Store model
        let storage_path = self.config.storage_path.join(format!("{}_{}.model", model_id, version));
        fs::write(&storage_path, model_data).await.map_err(|e| TrustformersError::Io {
            message: format!("Failed to store model: {}", e),
            path: None,
            suggestion: Some("Check if you have write permissions".to_string()),
        })?;

        // Add to catalog
        let mut models_lock = self.models.write().await;
        models_lock.insert(model_version.id.clone(), model_version);

        println!("Model {}:{} is now being shared", model_id, version);
        Ok(())
    }

    async fn find_peers_with_model(&self, model_id: &str, version: &str) -> Vec<PeerInfo> {
        let peers_lock = self.peers.read().await;
        peers_lock
            .values()
            .filter(|peer| {
                peer.models.iter().any(|ad| ad.model_id == model_id && ad.version == version)
            })
            .cloned()
            .collect()
    }

    async fn select_best_peer(&self, peers: &[PeerInfo]) -> PeerInfo {
        // Simple selection based on reputation and bandwidth
        let reputation_lock = self.reputation.lock().unwrap();

        peers
            .iter()
            .max_by(|a, b| {
                let score_a =
                    reputation_lock.get_reputation(&a.peer_id) * a.bandwidth.download_mbps;
                let score_b =
                    reputation_lock.get_reputation(&b.peer_id) * b.bandwidth.download_mbps;
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .unwrap_or_else(|| peers[0].clone())
    }

    async fn download_model_from_peer(
        &self,
        peer: &PeerInfo,
        model_id: &str,
        version: &str,
    ) -> Result<PathBuf> {
        // Connect to peer
        let mut stream =
            TcpStream::connect(&peer.address)
                .await
                .map_err(|e| TrustformersError::Network {
                    message: format!("Failed to connect to peer: {}", e),
                    url: None,
                    status_code: None,
                    suggestion: Some("Check network connection and peer availability".to_string()),
                    retry_recommended: true,
                })?;

        // Send model request
        let request = P2PMessage::ModelRequest {
            model_id: model_id.to_string(),
            version: version.to_string(),
            requestor_id: self.peer_id.clone(),
        };

        let request_data =
            serde_json::to_vec(&request).map_err(|e| TrustformersError::InvalidInput {
                message: format!("Failed to serialize request: {}", e),
                parameter: Some("request".to_string()),
                expected: Some("serializable data".to_string()),
                received: None,
                suggestion: Some("Check if the request data is valid".to_string()),
            })?;

        stream.write_all(&request_data).await.map_err(|e| TrustformersError::Network {
            message: format!("Failed to send request: {}", e),
            url: None,
            status_code: None,
            suggestion: Some("Check network connection".to_string()),
            retry_recommended: true,
        })?;

        // Receive response
        let mut buffer = [0u8; 4096];
        let n = stream.read(&mut buffer).await.map_err(|e| TrustformersError::Network {
            message: format!("Failed to read response: {}", e),
            url: None,
            status_code: None,
            suggestion: Some("Check network connection".to_string()),
            retry_recommended: true,
        })?;

        let response: P2PMessage =
            serde_json::from_slice(&buffer[..n]).map_err(|e| TrustformersError::InvalidInput {
                message: format!("Failed to parse response: {}", e),
                parameter: Some("response".to_string()),
                expected: Some("valid JSON".to_string()),
                received: None,
                suggestion: Some("Check if the response is corrupted".to_string()),
            })?;

        match response {
            P2PMessage::ModelResponse {
                available: true,
                chunk_info,
                ..
            } => {
                // Download chunks
                let output_path =
                    self.config.storage_path.join(format!("{}_{}.model", model_id, version));
                self.download_chunks(&mut stream, &chunk_info, &output_path).await?;
                Ok(output_path)
            },
            _ => Err(TrustformersError::Hub {
                message: "Model not available from peer".to_string(),
                model_id: model_id.to_string(),
                endpoint: None,
                suggestion: Some("Try requesting from a different peer".to_string()),
                recovery_actions: vec![],
            }
            .into()),
        }
    }

    async fn download_chunks(
        &self,
        stream: &mut TcpStream,
        chunk_info: &ChunkInfo,
        output_path: &Path,
    ) -> Result<()> {
        let mut file_data = Vec::with_capacity(chunk_info.total_size as usize);

        for chunk_id in 0..chunk_info.total_chunks {
            // Request chunk (simplified - would send chunk request message)
            let mut chunk_buffer = vec![0u8; chunk_info.chunk_size as usize];
            let n =
                stream.read(&mut chunk_buffer).await.map_err(|e| TrustformersError::Network {
                    message: format!("Failed to read chunk: {}", e),
                    url: None,
                    status_code: None,
                    suggestion: Some("Check network connection".to_string()),
                    retry_recommended: true,
                })?;

            chunk_buffer.truncate(n);
            file_data.extend_from_slice(&chunk_buffer);
        }

        fs::write(output_path, file_data).await.map_err(|e| TrustformersError::Io {
            message: format!("Failed to write model file: {}", e),
            path: None,
            suggestion: Some("Check if you have write permissions".to_string()),
        })?;

        Ok(())
    }

    fn generate_peer_id() -> String {
        let mut hasher = Sha256::new();
        hasher
            .update(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos().to_be_bytes());
        hex::encode(hasher.finalize())[..16].to_string()
    }

    fn create_peer_info(peer_id: &str, addr: SocketAddr) -> PeerInfo {
        PeerInfo {
            peer_id: peer_id.to_string(),
            address: addr,
            public_key: "dummy_key".to_string(),
            last_seen: SystemTime::now(),
            reputation: 0.5,
            capabilities: PeerCapabilities {
                can_serve_models: true,
                can_compute_diffs: true,
                supported_algorithms: vec!["xdelta3".to_string(), "layerwise".to_string()],
                max_model_size: 10 * 1024 * 1024 * 1024, // 10GB
                storage_capacity: 100 * 1024 * 1024 * 1024, // 100GB
                compute_power: 1.0,
            },
            bandwidth: BandwidthInfo {
                upload_mbps: 100.0,
                download_mbps: 100.0,
                latency_ms: 50,
                measured_at: SystemTime::now(),
            },
            models: Vec::new(),
        }
    }

    fn calculate_hash(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }

    pub async fn get_peer_count(&self) -> usize {
        let peers_lock = self.peers.read().await;
        peers_lock.len()
    }

    pub async fn get_model_count(&self) -> usize {
        let models_lock = self.models.read().await;
        models_lock.len()
    }

    pub fn get_peer_id(&self) -> &str {
        &self.peer_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_peer_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = P2PConfig {
            storage_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let node = P2PNode::new(config).await.unwrap();
        assert!(!node.peer_id.is_empty());
    }

    #[test]
    fn test_reputation_tracker() {
        let mut tracker = ReputationTracker::new();

        let event = ReputationEvent {
            event_type: ReputationEventType::SuccessfulTransfer,
            timestamp: SystemTime::now(),
            score_delta: 0.1,
        };

        tracker.update_reputation("peer1", event);
        assert_eq!(tracker.get_reputation("peer1"), 0.6);

        tracker.decay_reputations();
        assert!(tracker.get_reputation("peer1") < 0.6);
    }

    #[test]
    fn test_peer_info_serialization() {
        let peer_info = PeerInfo {
            peer_id: "test_peer".to_string(),
            address: "127.0.0.1:8080".parse().unwrap(),
            public_key: "test_key".to_string(),
            last_seen: SystemTime::now(),
            reputation: 0.8,
            capabilities: PeerCapabilities {
                can_serve_models: true,
                can_compute_diffs: true,
                supported_algorithms: vec!["xdelta3".to_string()],
                max_model_size: 1024,
                storage_capacity: 2048,
                compute_power: 1.0,
            },
            bandwidth: BandwidthInfo {
                upload_mbps: 100.0,
                download_mbps: 100.0,
                latency_ms: 50,
                measured_at: SystemTime::now(),
            },
            models: Vec::new(),
        };

        let serialized = serde_json::to_string(&peer_info).unwrap();
        let deserialized: PeerInfo = serde_json::from_str(&serialized).unwrap();

        assert_eq!(peer_info.peer_id, deserialized.peer_id);
        assert_eq!(peer_info.reputation, deserialized.reputation);
    }
}
