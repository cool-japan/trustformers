//! Deterministic, test-only CSPRNG.
//!
//! Production code in this crate always draws from the operating system CSPRNG
//! (`getrandom::SysRng`). Tests need reproducibility, so this module provides a
//! SHA-256-in-counter-mode generator that implements the `rand_core` traits the
//! RustCrypto key generators require.
//!
//! It is `#[cfg(test)]`-only and must never be reachable from the public API.

use sha2::{Digest, Sha256};

/// SHA-256 counter-mode deterministic RNG for tests.
pub(crate) struct TestRng {
    seed: [u8; 32],
    counter: u64,
    buffer: [u8; 32],
    used: usize,
}

impl TestRng {
    /// Create a generator from a one-byte seed label.
    pub(crate) fn seeded(label: u8) -> Self {
        Self {
            seed: [label; 32],
            counter: 0,
            buffer: [0u8; 32],
            used: 32,
        }
    }

    fn refill(&mut self) {
        let mut hasher = Sha256::new();
        hasher.update(b"trustformers-mobile/test-rng/v1");
        hasher.update(self.seed);
        hasher.update(self.counter.to_le_bytes());
        self.buffer.copy_from_slice(&hasher.finalize());
        self.counter = self.counter.wrapping_add(1);
        self.used = 0;
    }

    fn next_byte(&mut self) -> u8 {
        if self.used >= self.buffer.len() {
            self.refill();
        }
        let byte = self.buffer[self.used];
        self.used += 1;
        byte
    }
}

impl rand_core::TryRng for TestRng {
    type Error = core::convert::Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        let mut bytes = [0u8; 4];
        for byte in &mut bytes {
            *byte = self.next_byte();
        }
        Ok(u32::from_le_bytes(bytes))
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let mut bytes = [0u8; 8];
        for byte in &mut bytes {
            *byte = self.next_byte();
        }
        Ok(u64::from_le_bytes(bytes))
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        for byte in dst.iter_mut() {
            *byte = self.next_byte();
        }
        Ok(())
    }
}

impl rand_core::TryCryptoRng for TestRng {}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::Rng;

    #[test]
    fn same_seed_gives_same_stream() {
        let mut a = TestRng::seeded(7);
        let mut b = TestRng::seeded(7);
        let mut buf_a = [0u8; 64];
        let mut buf_b = [0u8; 64];
        a.fill_bytes(&mut buf_a);
        b.fill_bytes(&mut buf_b);
        assert_eq!(buf_a, buf_b);
    }

    #[test]
    fn different_seeds_give_different_streams() {
        let mut a = TestRng::seeded(1);
        let mut b = TestRng::seeded(2);
        let mut buf_a = [0u8; 64];
        let mut buf_b = [0u8; 64];
        a.fill_bytes(&mut buf_a);
        b.fill_bytes(&mut buf_b);
        assert_ne!(buf_a, buf_b);
    }

    #[test]
    fn stream_is_not_constant() {
        let mut rng = TestRng::seeded(3);
        let mut buf = [0u8; 128];
        rng.fill_bytes(&mut buf);
        assert!(
            buf.windows(2).any(|w| w[0] != w[1]),
            "output must not be constant"
        );
    }
}
