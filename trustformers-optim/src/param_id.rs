//! Stable parameter identity for optimizer state.
//!
//! # Why this module exists
//!
//! Optimizers keep per-parameter state (momentum, variance, slow weights, …) in
//! `HashMap<String, _>` keyed by a *parameter id*. Historically that id was the
//! parameter's heap address (`format!("{:p}", array.as_ptr())`) or a hash of the
//! parameter's current values. Both are unusable as a durable identity:
//!
//! * A heap address is different in every process, so a checkpoint written in one run
//!   never matches the parameters of the next run — `load_state_dict` appeared to
//!   succeed while restoring nothing, and training silently resumed from zeroed
//!   moments.
//! * A value hash changes the instant the parameter moves, so every single step
//!   allocated a *fresh* state entry: the optimizer degenerated to its first step
//!   forever and the state map grew without bound.
//!
//! [`ParamRegistry`] replaces both with a dense, registration-ordered [`ParamId`].
//!
//! # Identity contract
//!
//! A [`ParamId`] is an index assigned **the first time the registry sees a
//! parameter**, and it never changes for the lifetime of the optimizer. Two
//! resolution paths exist:
//!
//! 1. **Named** — [`ParamRegistry::key_for_named_tensor`]. The caller supplies a
//!    stable name (`"encoder.layer.0.weight"`). This is the preferred path: names
//!    are written into the checkpoint keys, so resume is *order independent*.
//!    Frameworks whose API already carries names (`HashMap<String, Tensor>` of
//!    gradients, PyTorch/TensorFlow/JAX compatibility layers) should always use it.
//!
//! 2. **Anonymous** — [`ParamRegistry::key_for_tensor`] /
//!    [`ParamRegistry::key_for_addr`]. Used by the bare
//!    [`Optimizer::update`](trustformers_core::traits::Optimizer::update) signature,
//!    which carries no name. Identity within a process comes from the tensor's data
//!    address; identity *across* processes comes from registration order. The
//!    contract is therefore:
//!
//!    > **A run that resumes from a checkpoint must present its parameters to
//!    > `update()` in the same order as the run that wrote the checkpoint.**
//!
//!    This is the same contract PyTorch imposes on `param_groups` ordering. A
//!    violation is *detected*, not ignored: binding a restored slot to a parameter
//!    of a different element count returns an error rather than silently starting
//!    from zero.
//!
//! # Checkpoint round trip
//!
//! State keys are `"n:<name>"` for named parameters and `"p:<index>"` for anonymous
//! ones. Because the key embeds the identity, the registry can be rebuilt purely
//! from the keys found in a checkpoint — see [`ParamRegistry::restore_key`]. After
//! restoring, entries are *unbound* (they have no address yet); the first `update()`
//! calls of the new run bind them in registration order.

use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;

/// Prefix marking a state key that identifies a parameter by name.
pub const NAMED_KEY_PREFIX: &str = "n:";
/// Prefix marking a state key that identifies a parameter by registration index.
pub const INDEXED_KEY_PREFIX: &str = "p:";

/// A stable, dense identifier for one parameter tensor within a single optimizer.
///
/// Ids are assigned in registration order starting at zero and are never reused or
/// renumbered. See the [module documentation](self) for the identity contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamId(usize);

impl ParamId {
    /// The dense registration index behind this id.
    pub fn index(self) -> usize {
        self.0
    }
}

/// One registry slot: the durable identity of a single parameter.
#[derive(Debug, Clone)]
struct ParamEntry {
    /// Stable caller-supplied name, when the caller had one.
    name: Option<String>,
    /// Number of elements, used to validate bindings after a checkpoint restore.
    numel: usize,
    /// Canonical state-map key (`"n:<name>"` or `"p:<index>"`).
    key: String,
    /// Data address of the tensor currently bound to this slot, if any.
    ///
    /// This is an in-process identity cache only; it is never persisted.
    addr: Option<usize>,
}

/// Assigns and remembers a stable [`ParamId`] for every parameter an optimizer sees.
///
/// Cheap to clone and free of interior mutability, so optimizers embedding it stay
/// `Clone + Send + Sync`.
#[derive(Debug, Clone, Default)]
pub struct ParamRegistry {
    /// Registration-ordered slots; `ParamId(i)` indexes `entries[i]`.
    entries: Vec<ParamEntry>,
    /// Name → registration index.
    by_name: HashMap<String, usize>,
    /// Data address → registration index (in-process cache, never persisted).
    by_addr: HashMap<usize, usize>,
    /// Lowest index that may still be waiting for an address binding.
    bind_cursor: usize,
}

impl ParamRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of parameters registered so far.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no parameter has been registered yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Forgets every registration. Call this alongside clearing optimizer state.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.by_name.clear();
        self.by_addr.clear();
        self.bind_cursor = 0;
    }

    /// The canonical state-map key for `id`, if it has been registered.
    pub fn key(&self, id: ParamId) -> Option<&str> {
        self.entries.get(id.0).map(|e| e.key.as_str())
    }

    /// The stable name of `id`, if it was registered through the named path.
    pub fn name(&self, id: ParamId) -> Option<&str> {
        self.entries.get(id.0).and_then(|e| e.name.as_deref())
    }

    /// The element count recorded for `id`, if it has been registered.
    pub fn numel(&self, id: ParamId) -> Option<usize> {
        self.entries.get(id.0).map(|e| e.numel)
    }

    /// Resolves the id of a parameter identified by a stable caller-supplied name.
    ///
    /// The first call for a given name registers it; later calls return the same id
    /// regardless of where the tensor lives in memory.
    pub fn id_for_named_tensor(&mut self, name: &str, tensor: &Tensor) -> Result<ParamId> {
        let (addr, numel) = tensor_identity(tensor)?;
        Ok(self.id_for_named_addr(name, addr, numel))
    }

    /// Resolves the id for a name plus an already-extracted address / element count.
    pub fn id_for_named_addr(&mut self, name: &str, addr: usize, numel: usize) -> ParamId {
        if let Some(&index) = self.by_name.get(name) {
            // Rebind: the tensor may have been reallocated between steps.
            if let Some(entry) = self.entries.get_mut(index) {
                if let Some(old) = entry.addr.replace(addr) {
                    if old != addr {
                        self.by_addr.remove(&old);
                    }
                }
                // A restored slot has numel 0 until the first real binding.
                if entry.numel == 0 {
                    entry.numel = numel;
                }
            }
            self.by_addr.insert(addr, index);
            self.advance_bind_cursor();
            return ParamId(index);
        }

        let index = self.entries.len();
        self.entries.push(ParamEntry {
            name: Some(name.to_string()),
            numel,
            key: format!("{NAMED_KEY_PREFIX}{name}"),
            addr: Some(addr),
        });
        self.by_name.insert(name.to_string(), index);
        self.by_addr.insert(addr, index);
        self.advance_bind_cursor();
        ParamId(index)
    }

    /// Resolves the id of an anonymous parameter tensor.
    ///
    /// See the [module documentation](self) for the ordering contract this implies
    /// for checkpoint resume.
    ///
    /// # Errors
    ///
    /// Returns an error when a slot restored from a checkpoint is bound to a
    /// parameter whose element count does not match, which means the caller is
    /// presenting parameters in a different order than the checkpointed run.
    pub fn id_for_tensor(&mut self, tensor: &Tensor) -> Result<ParamId> {
        let (addr, numel) = tensor_identity(tensor)?;
        self.id_for_addr(addr, numel)
    }

    /// Resolves the id for an already-extracted address / element count.
    ///
    /// # Errors
    ///
    /// See [`ParamRegistry::id_for_tensor`].
    pub fn id_for_addr(&mut self, addr: usize, numel: usize) -> Result<ParamId> {
        // Fast path: this exact buffer was seen before.
        if let Some(&index) = self.by_addr.get(&addr) {
            if self.entries.get(index).map(|e| e.numel) == Some(numel) {
                return Ok(ParamId(index));
            }
            // The address was reused by a differently-sized tensor: drop the stale
            // binding rather than corrupting another parameter's state.
            self.by_addr.remove(&addr);
            if let Some(entry) = self.entries.get_mut(index) {
                entry.addr = None;
            }
        }

        // Adopt the next slot that is still waiting for a binding. This is what makes
        // checkpoint resume work: `restore_key` creates unbound slots in registration
        // order, and the first updates of the new run claim them in the same order.
        self.advance_bind_cursor();
        if let Some(entry) = self.entries.get_mut(self.bind_cursor) {
            if entry.name.is_some() {
                // A named slot must be claimed through the named path, otherwise an
                // anonymous update would hijack a named parameter's state.
                return Err(TrustformersError::invalid_input(format!(
                    "optimizer state slot {} was checkpointed under the name '{}' but is \
                     being resumed through the anonymous update path; use `update_named` \
                     so the name can be matched",
                    self.bind_cursor,
                    entry.name.as_deref().unwrap_or("<unknown>")
                )));
            }
            if entry.numel != 0 && entry.numel != numel {
                return Err(TrustformersError::invalid_input(format!(
                    "optimizer state slot {} holds {} elements but the parameter being \
                     bound to it has {}; parameters must be passed to `update()` in the \
                     same order as the run that wrote the checkpoint (or use \
                     `update_named`)",
                    self.bind_cursor, entry.numel, numel
                )));
            }
            entry.numel = numel;
            entry.addr = Some(addr);
            let index = self.bind_cursor;
            self.by_addr.insert(addr, index);
            self.advance_bind_cursor();
            return Ok(ParamId(index));
        }

        // Genuinely new parameter.
        let index = self.entries.len();
        self.entries.push(ParamEntry {
            name: None,
            numel,
            key: format!("{INDEXED_KEY_PREFIX}{index}"),
            addr: Some(addr),
        });
        self.by_addr.insert(addr, index);
        self.advance_bind_cursor();
        Ok(ParamId(index))
    }

    /// Re-points an existing slot at a new data address.
    ///
    /// Optimizers that write results back with
    /// [`Tensor::set_data_f32`](trustformers_core::tensor::Tensor::set_data_f32) — which
    /// replaces the underlying buffer rather than mutating it — must call this after the
    /// write so the anonymous identity cache keeps tracking the parameter. Optimizers
    /// that mutate through `iter_mut()` keep their address and need not call it.
    ///
    /// # Errors
    ///
    /// Returns an error when `id` was never registered or the tensor dtype is unsupported.
    pub fn rebind(&mut self, id: ParamId, tensor: &Tensor) -> Result<()> {
        let (addr, numel) = tensor_identity(tensor)?;
        let entry = self.entries.get_mut(id.0).ok_or_else(|| {
            TrustformersError::invalid_input(format!(
                "cannot rebind unregistered parameter id {}",
                id.0
            ))
        })?;
        if let Some(old) = entry.addr.replace(addr) {
            if old != addr {
                self.by_addr.remove(&old);
            }
        }
        entry.numel = numel;
        self.by_addr.insert(addr, id.0);
        self.advance_bind_cursor();
        Ok(())
    }

    /// Convenience wrapper returning the canonical state-map key for a named tensor.
    ///
    /// # Errors
    ///
    /// Returns an error for tensor dtypes whose data address cannot be taken.
    pub fn key_for_named_tensor(&mut self, name: &str, tensor: &Tensor) -> Result<String> {
        let id = self.id_for_named_tensor(name, tensor)?;
        Ok(self.key_string(id))
    }

    /// Convenience wrapper returning the canonical state-map key for a named
    /// parameter given its address and element count.
    pub fn key_for_named_addr(&mut self, name: &str, addr: usize, numel: usize) -> String {
        let id = self.id_for_named_addr(name, addr, numel);
        self.key_string(id)
    }

    /// Convenience wrapper returning the canonical state-map key for a tensor.
    ///
    /// This is the direct replacement for the old `format!("{:p}", …)` idiom.
    ///
    /// # Errors
    ///
    /// See [`ParamRegistry::id_for_tensor`].
    pub fn key_for_tensor(&mut self, tensor: &Tensor) -> Result<String> {
        let id = self.id_for_tensor(tensor)?;
        Ok(self.key_string(id))
    }

    /// Convenience wrapper returning the canonical state-map key for an anonymous
    /// parameter given its data address and element count.
    ///
    /// # Errors
    ///
    /// See [`ParamRegistry::id_for_tensor`].
    pub fn key_for_addr(&mut self, addr: usize, numel: usize) -> Result<String> {
        let id = self.id_for_addr(addr, numel)?;
        Ok(self.key_string(id))
    }

    /// Re-creates the registry slot described by a checkpointed state key.
    ///
    /// `numel` is the length of the restored buffer and is used to validate later
    /// bindings. The recreated slot has no address, so the first matching `update()`
    /// of the resuming run claims it.
    ///
    /// # Errors
    ///
    /// Returns an error when `key` does not use a recognised identity prefix.
    pub fn restore_key(&mut self, key: &str, numel: usize) -> Result<ParamId> {
        if let Some(name) = key.strip_prefix(NAMED_KEY_PREFIX) {
            if let Some(&index) = self.by_name.get(name) {
                if let Some(entry) = self.entries.get_mut(index) {
                    if entry.numel == 0 {
                        entry.numel = numel;
                    }
                }
                return Ok(ParamId(index));
            }
            let index = self.entries.len();
            self.entries.push(ParamEntry {
                name: Some(name.to_string()),
                numel,
                key: key.to_string(),
                addr: None,
            });
            self.by_name.insert(name.to_string(), index);
            self.reset_bind_cursor();
            return Ok(ParamId(index));
        }

        if let Some(raw_index) = key.strip_prefix(INDEXED_KEY_PREFIX) {
            let index: usize = raw_index.parse().map_err(|_| {
                TrustformersError::invalid_input(format!(
                    "malformed optimizer state key '{key}': '{raw_index}' is not an index"
                ))
            })?;
            while self.entries.len() <= index {
                let placeholder = self.entries.len();
                self.entries.push(ParamEntry {
                    name: None,
                    numel: 0,
                    key: format!("{INDEXED_KEY_PREFIX}{placeholder}"),
                    addr: None,
                });
            }
            if let Some(entry) = self.entries.get_mut(index) {
                if entry.name.is_none() {
                    entry.numel = numel;
                }
            }
            self.reset_bind_cursor();
            return Ok(ParamId(index));
        }

        Err(TrustformersError::invalid_input(format!(
            "unrecognised optimizer state key '{key}': expected a '{NAMED_KEY_PREFIX}' or \
             '{INDEXED_KEY_PREFIX}' identity prefix"
        )))
    }

    /// All canonical keys in registration order.
    pub fn keys(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.key.clone()).collect()
    }

    fn key_string(&self, id: ParamId) -> String {
        self.entries
            .get(id.0)
            .map(|e| e.key.clone())
            .unwrap_or_else(|| format!("{INDEXED_KEY_PREFIX}{}", id.0))
    }

    fn advance_bind_cursor(&mut self) {
        while self.entries.get(self.bind_cursor).is_some_and(|e| e.addr.is_some()) {
            self.bind_cursor += 1;
        }
    }

    fn reset_bind_cursor(&mut self) {
        self.bind_cursor = 0;
        self.advance_bind_cursor();
    }
}

/// Extracts a tensor's data address and element count.
///
/// The address is used only as an in-process identity cache; it is never persisted.
fn tensor_identity(tensor: &Tensor) -> Result<(usize, usize)> {
    let numel: usize = tensor.shape().iter().product();
    let addr = match tensor {
        Tensor::F32(a) => a.as_ptr() as usize,
        Tensor::F64(a) => a.as_ptr() as usize,
        Tensor::F16(a) => a.as_ptr() as usize,
        Tensor::BF16(a) => a.as_ptr() as usize,
        Tensor::I64(a) => a.as_ptr() as usize,
        Tensor::C32(a) => a.as_ptr() as usize,
        Tensor::C64(a) => a.as_ptr() as usize,
        Tensor::CF16(a) => a.as_ptr() as usize,
        Tensor::CBF16(a) => a.as_ptr() as usize,
        other => {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "cannot derive a parameter identity for tensor dtype {:?}",
                    other.dtype()
                ),
                "ParamRegistry::id_for_tensor",
            ))
        },
    };
    Ok((addr, numel))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(len: usize) -> Tensor {
        Tensor::from_vec(vec![0.0_f32; len], &[len]).expect("tensor")
    }

    #[test]
    fn same_tensor_resolves_to_one_id() {
        let mut registry = ParamRegistry::new();
        let t = tensor(4);
        let a = registry.id_for_tensor(&t).expect("first");
        let b = registry.id_for_tensor(&t).expect("second");
        assert_eq!(a, b);
        assert_eq!(registry.len(), 1, "state must not grow per call");
    }

    #[test]
    fn distinct_tensors_get_distinct_ids() {
        let mut registry = ParamRegistry::new();
        let t1 = tensor(4);
        let t2 = tensor(8);
        let a = registry.id_for_tensor(&t1).expect("t1");
        let b = registry.id_for_tensor(&t2).expect("t2");
        assert_ne!(a, b);
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn mutating_a_tensor_in_place_does_not_change_its_id() {
        // Regression for value-hash keying (adafisher_simple): moving the parameter
        // used to allocate a brand-new state entry every step. Optimizers mutate
        // parameters through `iter_mut()`, which keeps the buffer address.
        let mut registry = ParamRegistry::new();
        let mut t = tensor(4);
        let before = registry.id_for_tensor(&t).expect("before");
        match &mut t {
            Tensor::F32(array) => {
                for value in array.iter_mut() {
                    *value = 9.0;
                }
            },
            _ => panic!("expected F32"),
        }
        let after = registry.id_for_tensor(&t).expect("after");
        assert_eq!(before, after);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn rebind_tracks_a_reallocated_buffer() {
        // `set_data_f32` replaces the buffer, so optimizers using it must rebind.
        let mut registry = ParamRegistry::new();
        let mut t = tensor(4);
        let id = registry.id_for_tensor(&t).expect("register");
        t.set_data_f32(&[9.0, 9.0, 9.0, 9.0]).expect("reallocate");
        registry.rebind(id, &t).expect("rebind");
        assert_eq!(registry.id_for_tensor(&t).expect("after"), id);
        assert_eq!(registry.len(), 1, "rebinding must not append a slot");
    }

    #[test]
    fn named_ids_are_address_independent() {
        let mut registry = ParamRegistry::new();
        let first = tensor(4);
        let id1 = registry.id_for_named_tensor("w", &first).expect("first");
        drop(first);
        let second = tensor(4);
        let id2 = registry.id_for_named_tensor("w", &second).expect("second");
        assert_eq!(id1, id2, "a name must outlive the tensor allocation");
        assert_eq!(registry.key(id1), Some("n:w"));
    }

    #[test]
    fn keys_are_stable_and_prefixed() {
        let mut registry = ParamRegistry::new();
        let t = tensor(2);
        assert_eq!(registry.key_for_tensor(&t).expect("key"), "p:0");
        assert_eq!(
            registry.key_for_named_tensor("bias", &t).expect("key"),
            "n:bias"
        );
    }

    #[test]
    fn restored_anonymous_slots_are_claimed_in_order() {
        // Regression for address keying: a new process has different addresses, so a
        // restored checkpoint used to match nothing.
        let mut registry = ParamRegistry::new();
        registry.restore_key("p:0", 4).expect("restore 0");
        registry.restore_key("p:1", 8).expect("restore 1");

        let t1 = tensor(4);
        let t2 = tensor(8);
        assert_eq!(registry.key_for_tensor(&t1).expect("bind 0"), "p:0");
        assert_eq!(registry.key_for_tensor(&t2).expect("bind 1"), "p:1");
        assert_eq!(registry.len(), 2, "resume must not append new slots");
    }

    #[test]
    fn restored_named_slots_match_by_name_in_any_order() {
        let mut registry = ParamRegistry::new();
        registry.restore_key("n:a", 4).expect("restore a");
        registry.restore_key("n:b", 8).expect("restore b");

        let tb = tensor(8);
        let ta = tensor(4);
        // Deliberately reversed relative to registration order.
        assert_eq!(registry.key_for_named_tensor("b", &tb).expect("b"), "n:b");
        assert_eq!(registry.key_for_named_tensor("a", &ta).expect("a"), "n:a");
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn order_mismatch_on_resume_is_an_error_not_silent_reset() {
        let mut registry = ParamRegistry::new();
        registry.restore_key("p:0", 4).expect("restore 0");
        let wrong = tensor(9);
        let err = registry.id_for_tensor(&wrong);
        assert!(
            err.is_err(),
            "binding a 9-element tensor to a 4-element slot must be reported"
        );
    }

    #[test]
    fn anonymous_path_refuses_to_hijack_a_named_slot() {
        let mut registry = ParamRegistry::new();
        registry.restore_key("n:w", 4).expect("restore");
        let t = tensor(4);
        assert!(registry.id_for_tensor(&t).is_err());
    }

    #[test]
    fn restore_rejects_unprefixed_keys() {
        let mut registry = ParamRegistry::new();
        assert!(registry.restore_key("0x7f9c2a001234", 4).is_err());
    }

    #[test]
    fn clear_resets_everything() {
        let mut registry = ParamRegistry::new();
        let t = tensor(4);
        registry.id_for_tensor(&t).expect("register");
        registry.clear();
        assert!(registry.is_empty());
        assert_eq!(registry.key_for_tensor(&t).expect("re-register"), "p:0");
    }
}
