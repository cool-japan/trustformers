use super::*;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

/// Build a handle whose release callback increments a shared counter and records
/// the `(device_id, buffer_id)` pair it was invoked with.
#[allow(clippy::type_complexity)]
fn counting_handle(
    device_id: usize,
) -> (
    OxiCudaBufferHandle,
    Arc<AtomicUsize>,
    Arc<Mutex<Option<(usize, OxiCudaBufferId)>>>,
) {
    let releases = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(Mutex::new(None));
    let releases_hook = Arc::clone(&releases);
    let observed_hook = Arc::clone(&observed);
    let handle = OxiCudaBufferHandle::with_release(
        OxiCudaBufferId::new(),
        device_id,
        Box::new(move |dev, id| {
            releases_hook.fetch_add(1, AtomicOrdering::SeqCst);
            if let Ok(mut slot) = observed_hook.lock() {
                *slot = Some((dev, id));
            }
        }),
    );
    (handle, releases, observed)
}

#[test]
fn buffer_handle_releases_exactly_once_after_last_clone() {
    let (handle, releases, _) = counting_handle(0);
    let clone_a = handle.clone();
    let clone_b = clone_a.clone();
    assert_eq!(handle.ref_count(), 3);

    // Drop in an order different from creation: no release until the last one.
    drop(clone_a);
    assert_eq!(releases.load(AtomicOrdering::SeqCst), 0);
    drop(handle);
    assert_eq!(releases.load(AtomicOrdering::SeqCst), 0);
    assert_eq!(clone_b.ref_count(), 1);

    drop(clone_b);
    assert_eq!(
        releases.load(AtomicOrdering::SeqCst),
        1,
        "release must fire exactly once, on the last drop"
    );
}

#[test]
fn buffer_handle_clones_share_identity() {
    let (handle, _, _) = counting_handle(2);
    let clone = handle.clone();
    assert_eq!(handle.id(), clone.id());
    assert_eq!(handle.device_id(), clone.device_id());
    assert_eq!(clone.device_id(), 2);
}

#[test]
fn buffer_handle_release_receives_device_and_buffer_id() {
    let (handle, releases, observed) = counting_handle(7);
    let expected_id = handle.id();
    drop(handle);

    assert_eq!(releases.load(AtomicOrdering::SeqCst), 1);
    let slot = observed.lock().unwrap_or_else(|poison| poison.into_inner());
    assert_eq!(
        *slot,
        Some((7, expected_id)),
        "release must be invoked with the handle's device ordinal and buffer id"
    );
}

#[test]
fn cuda_tensor_data_clone_and_drop_frees_exactly_once() {
    // End-to-end over the tensor wrapper: cloning `Tensor::CUDA` bumps the refcount,
    // and only the final drop of the last clone triggers the (mocked) device free.
    let (handle, releases, _) = counting_handle(0);
    let data =
        crate::tensor::CudaTensorData::from_handle(handle, vec![2, 2], crate::tensor::DType::F32);
    let t1 = crate::tensor::Tensor::CUDA(data);
    let t2 = t1.clone();
    let t3 = t2.clone();

    drop(t1);
    drop(t3);
    assert_eq!(
        releases.load(AtomicOrdering::SeqCst),
        0,
        "buffer must stay alive while any tensor clone remains"
    );

    drop(t2);
    assert_eq!(
        releases.load(AtomicOrdering::SeqCst),
        1,
        "buffer must be freed exactly once when the last tensor clone drops"
    );
}

#[test]
fn release_without_backend_is_a_noop_and_never_constructs_one() {
    // A device ordinal no test ever instantiates a backend for.
    let device_id = usize::MAX;
    // Must neither panic nor lazily construct a backend (construction would require
    // real CUDA hardware and must never happen from a Drop path).
    release_persistent_buffer(device_id, OxiCudaBufferId::new());

    let registry = OXICUDA_BACKENDS.lock().unwrap_or_else(|poison| poison.into_inner());
    assert!(
        !registry.contains_key(&device_id),
        "release path must not create backend registry entries"
    );
}

#[test]
fn default_cuda_device_id_is_zero_without_override() {
    // The environment override is parsed once per process; the test environment does
    // not set it, so the default ordinal must be 0.
    if std::env::var("TRUSTFORMERS_CUDA_DEVICE").is_err() {
        assert_eq!(default_cuda_device_id(), 0);
    }
}
