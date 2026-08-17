//! Point-to-point transports that back the collective algorithms in
//! [`super::collective`].
//!
//! A [`Transport`] moves opaque byte payloads between ranks. Two
//! implementations are provided, both pure Rust:
//!
//! * [`InProcessTransport`] — ranks are threads of a single process and
//!   rendezvous through shared memory. Used by the test-suite and by
//!   single-machine simulations.
//! * [`TcpTransport`] — ranks are separate processes (optionally on separate
//!   hosts) connected by a full mesh of TCP sockets.
//!
//! Both guarantee ordered, reliable delivery per `(source, tag)` pair, which is
//! all the collective algorithms require.

use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

/// Default time a `recv` waits before declaring the peer unresponsive.
///
/// A finite timeout turns a mis-ordered collective (which would otherwise
/// deadlock forever) into a diagnosable error.
pub const DEFAULT_RECV_TIMEOUT: Duration = Duration::from_secs(30);

/// Default time spent retrying an outgoing TCP connection before giving up.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Errors raised by the transport layer.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    /// `rank` is not a valid participant.
    #[error("rank {rank} is out of range for world size {world_size}")]
    RankOutOfRange {
        /// Offending rank.
        rank: usize,
        /// Configured world size.
        world_size: usize,
    },

    /// A rank sent a message to itself; collectives never do this.
    #[error("rank {rank} attempted to send to itself")]
    SelfSend {
        /// Offending rank.
        rank: usize,
    },

    /// No message arrived in time.
    #[error(
        "rank {rank} timed out after {timeout:?} waiting for a message from rank {peer} \
         (tag {tag}); the peer is not running the same collective sequence"
    )]
    RecvTimeout {
        /// Waiting rank.
        rank: usize,
        /// Expected sender.
        peer: usize,
        /// Expected tag.
        tag: u64,
        /// Timeout that elapsed.
        timeout: Duration,
    },

    /// Two ranks disagree about the size of an in-process session.
    #[error(
        "in-process session `{session}` was created with world size {expected}, \
         but rank {rank} requested world size {actual}"
    )]
    WorldSizeMismatch {
        /// Session key.
        session: String,
        /// World size the session was created with.
        expected: usize,
        /// World size requested now.
        actual: usize,
        /// Requesting rank.
        rank: usize,
    },

    /// The same rank joined a session twice.
    #[error("rank {rank} has already joined in-process session `{session}`")]
    DuplicateRank {
        /// Session key.
        session: String,
        /// Offending rank.
        rank: usize,
    },

    /// The peer address table is inconsistent.
    #[error("invalid peer table: {0}")]
    InvalidPeers(String),

    /// A frame could not be parsed.
    #[error("malformed frame from rank {peer}: {reason}")]
    MalformedFrame {
        /// Sender rank as reported by the frame.
        peer: usize,
        /// What was wrong.
        reason: String,
    },

    /// Underlying socket failure.
    #[error("transport I/O failure: {0}")]
    Io(#[from] std::io::Error),
}

/// Reliable, ordered point-to-point byte transport between ranks.
///
/// Implementations must be safe to share across threads, but the collective
/// algorithms assume a single logical caller per rank (SPMD).
pub trait Transport: Send + Sync + std::fmt::Debug {
    /// Rank of this participant.
    fn rank(&self) -> usize;

    /// Total number of participants.
    fn world_size(&self) -> usize;

    /// Deliver `payload` to `peer`, tagged with `tag`.
    ///
    /// Must not block on the peer having posted a matching `recv`.
    fn send(&self, peer: usize, tag: u64, payload: &[u8]) -> Result<()>;

    /// Block until a payload from `peer` with `tag` is available.
    fn recv(&self, peer: usize, tag: u64) -> Result<Vec<u8>>;
}

impl<T: Transport + ?Sized> Transport for Arc<T> {
    fn rank(&self) -> usize {
        (**self).rank()
    }
    fn world_size(&self) -> usize {
        (**self).world_size()
    }
    fn send(&self, peer: usize, tag: u64, payload: &[u8]) -> Result<()> {
        (**self).send(peer, tag, payload)
    }
    fn recv(&self, peer: usize, tag: u64) -> Result<Vec<u8>> {
        (**self).recv(peer, tag)
    }
}

/// Per-rank inbox keyed by `(source rank, tag)`.
#[derive(Debug, Default)]
struct Mailbox {
    slots: Mutex<HashMap<(usize, u64), VecDeque<Vec<u8>>>>,
    arrival: Condvar,
}

impl Mailbox {
    fn deposit(&self, from: usize, tag: u64, payload: Vec<u8>) {
        let mut slots = self.slots.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        slots.entry((from, tag)).or_default().push_back(payload);
        drop(slots);
        self.arrival.notify_all();
    }

    fn collect(&self, rank: usize, from: usize, tag: u64, timeout: Duration) -> Result<Vec<u8>> {
        let deadline = Instant::now() + timeout;
        let mut slots = self.slots.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        loop {
            if let Some(queue) = slots.get_mut(&(from, tag)) {
                if let Some(payload) = queue.pop_front() {
                    if queue.is_empty() {
                        slots.remove(&(from, tag));
                    }
                    return Ok(payload);
                }
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(TransportError::RecvTimeout {
                    rank,
                    peer: from,
                    tag,
                    timeout,
                }
                .into());
            }

            let (guard, _) = self
                .arrival
                .wait_timeout(slots, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            slots = guard;
        }
    }
}

// ─── In-process transport ────────────────────────────────────────────────────

/// Shared rendezvous point for ranks living in one process.
#[derive(Debug)]
pub struct InProcessSession {
    world_size: usize,
    mailboxes: Vec<Arc<Mailbox>>,
    joined: Mutex<Vec<bool>>,
    key: String,
}

impl InProcessSession {
    /// Create an anonymous session for `world_size` ranks.
    pub fn new(world_size: usize) -> Result<Arc<Self>> {
        Self::with_key("<anonymous>", world_size)
    }

    fn with_key(key: &str, world_size: usize) -> Result<Arc<Self>> {
        if world_size == 0 {
            return Err(TransportError::InvalidPeers("world_size must be >= 1".to_string()).into());
        }
        Ok(Arc::new(Self {
            world_size,
            mailboxes: (0..world_size).map(|_| Arc::new(Mailbox::default())).collect(),
            joined: Mutex::new(vec![false; world_size]),
            key: key.to_string(),
        }))
    }

    /// Number of participants.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Claim `rank` within this session.
    ///
    /// Each rank may only be claimed once; a second attempt yields
    /// [`TransportError::DuplicateRank`].
    pub fn transport(self: &Arc<Self>, rank: usize) -> Result<InProcessTransport> {
        if rank >= self.world_size {
            return Err(TransportError::RankOutOfRange {
                rank,
                world_size: self.world_size,
            }
            .into());
        }

        let mut joined = self.joined.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if joined[rank] {
            return Err(TransportError::DuplicateRank {
                session: self.key.clone(),
                rank,
            }
            .into());
        }
        joined[rank] = true;
        drop(joined);

        Ok(InProcessTransport {
            session: Arc::clone(self),
            rank,
            timeout: DEFAULT_RECV_TIMEOUT,
        })
    }

    /// Convenience constructor returning one transport per rank.
    pub fn transports(world_size: usize) -> Result<Vec<InProcessTransport>> {
        let session = Self::new(world_size)?;
        (0..world_size).map(|rank| session.transport(rank)).collect()
    }
}

fn session_registry() -> &'static Mutex<HashMap<String, Weak<InProcessSession>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Weak<InProcessSession>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Join the named in-process session, creating it if this is the first rank.
///
/// All ranks of a session must pass the same `world_size`. The session is
/// dropped automatically once every transport derived from it has been
/// released, so a subsequent join with the same key starts fresh.
pub fn join_in_process_session(key: &str, world_size: usize) -> Result<Arc<InProcessSession>> {
    let mut registry = session_registry().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    registry.retain(|_, weak| weak.strong_count() > 0);

    if let Some(existing) = registry.get(key).and_then(Weak::upgrade) {
        if existing.world_size != world_size {
            return Err(TransportError::WorldSizeMismatch {
                session: key.to_string(),
                expected: existing.world_size,
                actual: world_size,
                rank: usize::MAX,
            }
            .into());
        }
        return Ok(existing);
    }

    let session = InProcessSession::with_key(key, world_size)?;
    registry.insert(key.to_string(), Arc::downgrade(&session));
    Ok(session)
}

/// A single rank's handle on an [`InProcessSession`].
#[derive(Debug, Clone)]
pub struct InProcessTransport {
    session: Arc<InProcessSession>,
    rank: usize,
    timeout: Duration,
}

impl InProcessTransport {
    /// Override the receive timeout (default [`DEFAULT_RECV_TIMEOUT`]).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// The session this transport belongs to.
    pub fn session(&self) -> &Arc<InProcessSession> {
        &self.session
    }
}

impl Transport for InProcessTransport {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.session.world_size
    }

    fn send(&self, peer: usize, tag: u64, payload: &[u8]) -> Result<()> {
        if peer >= self.session.world_size {
            return Err(TransportError::RankOutOfRange {
                rank: peer,
                world_size: self.session.world_size,
            }
            .into());
        }
        if peer == self.rank {
            return Err(TransportError::SelfSend { rank: self.rank }.into());
        }
        self.session.mailboxes[peer].deposit(self.rank, tag, payload.to_vec());
        Ok(())
    }

    fn recv(&self, peer: usize, tag: u64) -> Result<Vec<u8>> {
        if peer >= self.session.world_size {
            return Err(TransportError::RankOutOfRange {
                rank: peer,
                world_size: self.session.world_size,
            }
            .into());
        }
        self.session.mailboxes[self.rank].collect(self.rank, peer, tag, self.timeout)
    }
}

// ─── TCP transport ───────────────────────────────────────────────────────────

const FRAME_HEADER_LEN: usize = 4 + 8 + 4; // src: u32, tag: u64, len: u32

fn write_frame(stream: &mut TcpStream, src: usize, tag: u64, payload: &[u8]) -> Result<()> {
    let mut header = Vec::with_capacity(FRAME_HEADER_LEN + payload.len());
    header.extend_from_slice(&(src as u32).to_le_bytes());
    header.extend_from_slice(&tag.to_le_bytes());
    header.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    header.extend_from_slice(payload);
    stream.write_all(&header).map_err(TransportError::Io)?;
    stream.flush().map_err(TransportError::Io)?;
    Ok(())
}

/// Reads frames from `stream` into `mailbox` until EOF or shutdown.
fn reader_loop(mut stream: TcpStream, mailbox: Arc<Mailbox>, shutdown: Arc<AtomicBool>) {
    let mut header = [0u8; FRAME_HEADER_LEN];
    loop {
        if shutdown.load(Ordering::Relaxed) {
            return;
        }
        match stream.read_exact(&mut header) {
            Ok(()) => {},
            Err(_) => return, // EOF or peer went away
        }

        let mut src_bytes = [0u8; 4];
        src_bytes.copy_from_slice(&header[0..4]);
        let mut tag_bytes = [0u8; 8];
        tag_bytes.copy_from_slice(&header[4..12]);
        let mut len_bytes = [0u8; 4];
        len_bytes.copy_from_slice(&header[12..16]);

        let src = u32::from_le_bytes(src_bytes) as usize;
        let tag = u64::from_le_bytes(tag_bytes);
        let len = u32::from_le_bytes(len_bytes) as usize;

        let mut payload = vec![0u8; len];
        if len > 0 && stream.read_exact(&mut payload).is_err() {
            return;
        }
        mailbox.deposit(src, tag, payload);
    }
}

/// Full-mesh TCP transport.
///
/// Rank `i` listens on `peers[i]` and dials `peers[j]` on first use. Incoming
/// connections are drained by dedicated reader threads into a local mailbox, so
/// a `send` never blocks waiting for the peer to post a matching `recv`.
#[derive(Debug)]
pub struct TcpTransport {
    rank: usize,
    peers: Vec<SocketAddr>,
    mailbox: Arc<Mailbox>,
    outgoing: Mutex<HashMap<usize, TcpStream>>,
    shutdown: Arc<AtomicBool>,
    timeout: Duration,
    connect_timeout: Duration,
}

impl TcpTransport {
    /// Bind `peers[rank]` and prepare the mesh.
    pub fn bind(rank: usize, peers: Vec<SocketAddr>) -> Result<Self> {
        if rank >= peers.len() {
            return Err(TransportError::RankOutOfRange {
                rank,
                world_size: peers.len(),
            }
            .into());
        }
        let listener = TcpListener::bind(peers[rank]).map_err(TransportError::Io)?;
        Self::with_listener(rank, peers, listener)
    }

    /// Derive the peer table from a master address, assigning rank `i` the port
    /// `master_port + i`, then bind and prepare the mesh.
    pub fn from_master(
        rank: usize,
        world_size: usize,
        master_addr: &str,
        master_port: u16,
    ) -> Result<Self> {
        let peers = Self::peer_table(world_size, master_addr, master_port)?;
        Self::bind(rank, peers)
    }

    /// Build the `master_addr:(master_port + i)` peer table.
    pub fn peer_table(
        world_size: usize,
        master_addr: &str,
        master_port: u16,
    ) -> Result<Vec<SocketAddr>> {
        use std::net::ToSocketAddrs;

        if world_size == 0 {
            return Err(TransportError::InvalidPeers("world_size must be >= 1".to_string()).into());
        }

        let mut peers = Vec::with_capacity(world_size);
        for index in 0..world_size {
            let port = master_port.checked_add(index as u16).ok_or_else(|| {
                TransportError::InvalidPeers(format!(
                    "master_port {master_port} + rank {index} overflows a u16"
                ))
            })?;
            let addr = format!("{master_addr}:{port}")
                .to_socket_addrs()
                .map_err(TransportError::Io)?
                .next()
                .ok_or_else(|| {
                    TransportError::InvalidPeers(format!(
                        "`{master_addr}:{port}` did not resolve to any socket address"
                    ))
                })?;
            peers.push(addr);
        }
        Ok(peers)
    }

    /// Use an already-bound listener. Useful for tests, which can bind every
    /// rank's listener on port 0 first and thereby avoid port races.
    pub fn with_listener(
        rank: usize,
        mut peers: Vec<SocketAddr>,
        listener: TcpListener,
    ) -> Result<Self> {
        if rank >= peers.len() {
            return Err(TransportError::RankOutOfRange {
                rank,
                world_size: peers.len(),
            }
            .into());
        }
        peers[rank] = listener.local_addr().map_err(TransportError::Io)?;

        let mailbox = Arc::new(Mailbox::default());
        let shutdown = Arc::new(AtomicBool::new(false));

        listener.set_nonblocking(true).map_err(TransportError::Io)?;
        let accept_mailbox = Arc::clone(&mailbox);
        let accept_shutdown = Arc::clone(&shutdown);
        std::thread::Builder::new()
            .name(format!("tf-tcp-accept-{rank}"))
            .spawn(move || {
                while !accept_shutdown.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            if stream.set_nonblocking(false).is_err() {
                                continue;
                            }
                            let reader_mailbox = Arc::clone(&accept_mailbox);
                            let reader_shutdown = Arc::clone(&accept_shutdown);
                            let _ = std::thread::Builder::new()
                                .name(format!("tf-tcp-read-{rank}"))
                                .spawn(move || {
                                    reader_loop(stream, reader_mailbox, reader_shutdown)
                                });
                        },
                        Err(ref err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(2));
                        },
                        Err(_) => return,
                    }
                }
            })
            .map_err(TransportError::Io)?;

        Ok(Self {
            rank,
            peers,
            mailbox,
            outgoing: Mutex::new(HashMap::new()),
            shutdown,
            timeout: DEFAULT_RECV_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        })
    }

    /// Override the receive timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Override how long an outgoing connection is retried.
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Address this rank listens on.
    pub fn local_addr(&self) -> SocketAddr {
        self.peers[self.rank]
    }

    fn dial(&self, peer: usize) -> Result<TcpStream> {
        let addr = self.peers[peer];
        let deadline = Instant::now() + self.connect_timeout;
        let mut last_error: Option<std::io::Error> = None;

        while Instant::now() < deadline {
            match TcpStream::connect(addr) {
                Ok(stream) => {
                    stream.set_nodelay(true).map_err(TransportError::Io)?;
                    return Ok(stream);
                },
                Err(err) => {
                    last_error = Some(err);
                    std::thread::sleep(Duration::from_millis(5));
                },
            }
        }

        Err(TransportError::Io(last_error.unwrap_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!("could not connect to rank {peer} at {addr}"),
            )
        }))
        .into())
    }
}

impl Drop for TcpTransport {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        let mut outgoing = self.outgoing.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        for (_, stream) in outgoing.drain() {
            let _ = stream.shutdown(std::net::Shutdown::Both);
        }
    }
}

impl Transport for TcpTransport {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.peers.len()
    }

    fn send(&self, peer: usize, tag: u64, payload: &[u8]) -> Result<()> {
        if peer >= self.peers.len() {
            return Err(TransportError::RankOutOfRange {
                rank: peer,
                world_size: self.peers.len(),
            }
            .into());
        }
        if peer == self.rank {
            return Err(TransportError::SelfSend { rank: self.rank }.into());
        }

        let mut outgoing = self.outgoing.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let stream = match outgoing.entry(peer) {
            std::collections::hash_map::Entry::Occupied(slot) => slot.into_mut(),
            std::collections::hash_map::Entry::Vacant(slot) => slot.insert(self.dial(peer)?),
        };
        write_frame(stream, self.rank, tag, payload)
    }

    fn recv(&self, peer: usize, tag: u64) -> Result<Vec<u8>> {
        if peer >= self.peers.len() {
            return Err(TransportError::RankOutOfRange {
                rank: peer,
                world_size: self.peers.len(),
            }
            .into());
        }
        self.mailbox.collect(self.rank, peer, tag, self.timeout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_process_transport_moves_bytes_between_ranks() {
        let transports =
            InProcessSession::transports(2).expect("session creation must succeed in test");
        let sender = transports[0].clone();
        let receiver = transports[1].clone();

        let handle = std::thread::spawn(move || receiver.recv(0, 7));
        sender.send(1, 7, b"hello").expect("send must succeed in test");

        let received = handle
            .join()
            .expect("receiver thread must not panic")
            .expect("recv must succeed in test");
        assert_eq!(received, b"hello");
    }

    #[test]
    fn in_process_transport_preserves_per_tag_order() {
        let transports =
            InProcessSession::transports(2).expect("session creation must succeed in test");
        transports[0].send(1, 1, b"first").expect("send must succeed in test");
        transports[0].send(1, 1, b"second").expect("send must succeed in test");

        assert_eq!(
            transports[1].recv(0, 1).expect("recv must succeed in test"),
            b"first"
        );
        assert_eq!(
            transports[1].recv(0, 1).expect("recv must succeed in test"),
            b"second"
        );
    }

    #[test]
    fn recv_times_out_instead_of_hanging() {
        let transports =
            InProcessSession::transports(2).expect("session creation must succeed in test");
        let receiver = transports[1].clone().with_timeout(Duration::from_millis(50));

        let err = receiver.recv(0, 99).expect_err("recv must time out in test");
        let transport_err = err.downcast_ref::<TransportError>().expect("must be a TransportError");
        assert!(matches!(transport_err, TransportError::RecvTimeout { .. }));
    }

    #[test]
    fn duplicate_rank_is_rejected() {
        let session = InProcessSession::new(2).expect("session creation must succeed in test");
        let _first = session.transport(0).expect("first join must succeed in test");
        let err = session.transport(0).expect_err("second join must fail in test");
        assert!(matches!(
            err.downcast_ref::<TransportError>(),
            Some(TransportError::DuplicateRank { .. })
        ));
    }

    #[test]
    fn named_sessions_rendezvous_and_reject_size_mismatch() {
        let key = "transport-test-session";
        let session = join_in_process_session(key, 3).expect("join must succeed in test");
        let again = join_in_process_session(key, 3).expect("second join must succeed in test");
        assert!(Arc::ptr_eq(&session, &again));

        let err = join_in_process_session(key, 4).expect_err("mismatch must fail in test");
        assert!(matches!(
            err.downcast_ref::<TransportError>(),
            Some(TransportError::WorldSizeMismatch { .. })
        ));
    }

    #[test]
    fn self_send_is_rejected() {
        let transports =
            InProcessSession::transports(2).expect("session creation must succeed in test");
        let err = transports[0].send(0, 0, b"x").expect_err("self-send must fail in test");
        assert!(matches!(
            err.downcast_ref::<TransportError>(),
            Some(TransportError::SelfSend { .. })
        ));
    }

    #[test]
    fn tcp_transport_moves_bytes_between_ranks() {
        // Bind both listeners up-front so no port can be stolen between
        // discovery and use.
        let listeners: Vec<TcpListener> = (0..2)
            .map(|_| TcpListener::bind("127.0.0.1:0").expect("bind must succeed in test"))
            .collect();
        let addrs: Vec<SocketAddr> = listeners
            .iter()
            .map(|l| l.local_addr().expect("local_addr must succeed in test"))
            .collect();

        let mut listeners = listeners.into_iter();
        let first = listeners.next().expect("two listeners in test");
        let second = listeners.next().expect("two listeners in test");

        let rank0 = TcpTransport::with_listener(0, addrs.clone(), first)
            .expect("rank 0 transport must build in test")
            .with_timeout(Duration::from_secs(10));
        let rank1 = TcpTransport::with_listener(1, addrs, second)
            .expect("rank 1 transport must build in test")
            .with_timeout(Duration::from_secs(10));

        let handle = std::thread::spawn(move || {
            let payload = rank1.recv(0, 42)?;
            rank1.send(0, 43, b"pong")?;
            anyhow::Ok(payload)
        });

        rank0.send(1, 42, b"ping").expect("send must succeed in test");
        let echoed = rank0.recv(1, 43).expect("recv must succeed in test");

        let received = handle
            .join()
            .expect("peer thread must not panic")
            .expect("peer exchange must succeed in test");
        assert_eq!(received, b"ping");
        assert_eq!(echoed, b"pong");
    }
}
