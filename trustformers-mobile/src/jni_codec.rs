//! Pure byte<->[`Tensor`] codecs used at JNI/FFI boundaries.
//!
//! `android::jni` (and any other platform bridge that needs the same flat
//! little-endian `f32` wire format) is `#[cfg(target_os = "android")]`, so
//! its own tests never compile or run on a non-Android host -- including
//! every host this workspace's CI/development happens on. The actual
//! encode/decode logic has no JNI dependency at all, so it lives here,
//! unconditionally compiled, where its tests are real, always-running
//! regression coverage rather than code that merely looks tested.
//!
//! This is the direct fix for the P0 finding against the previous
//! `android::jni::Java_..._inference`: it computed a real output tensor and
//! then discarded it, always returning `vec![0; 4]` to the JVM caller.

use trustformers_core::errors::Result;
use trustformers_core::Tensor;

/// Decode a flat little-endian `f32` byte buffer (the layout a Java caller
/// builds via `ByteBuffer.order(ByteOrder.LITTLE_ENDIAN)`) into a rank-1
/// [`Tensor`].
///
/// # Errors
///
/// Fails when `bytes.len()` is not a multiple of 4 -- silently dropping a
/// trailing 1-3 byte remainder (as `chunks(4)`, rather than
/// `chunks_exact(4)`, would do) would decode a different value than the
/// caller actually sent.
pub fn bytes_to_f32_tensor(bytes: &[u8]) -> Result<Tensor> {
    if !bytes.len().is_multiple_of(4) {
        return Err(trustformers_core::errors::invalid_format(
            "a byte length that is a multiple of 4 (a flat little-endian f32 buffer)",
            format!("{} bytes", bytes.len()),
        ));
    }
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Tensor::from_vec(floats, &[bytes.len() / 4])
}

/// Encode a [`Tensor`]'s real values as a flat little-endian `f32` byte
/// buffer -- the exact inverse of [`bytes_to_f32_tensor`], so a Java caller
/// decodes the real computed output with the same `ByteBuffer` layout it
/// used to send the input.
pub fn tensor_to_le_bytes(tensor: &Tensor) -> Result<Vec<u8>> {
    let data = tensor.data()?;
    Ok(data.iter().flat_map(|value| value.to_le_bytes()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `bytes_to_f32_tensor` must decode the exact little-endian values a
    /// Java `ByteBuffer.order(LITTLE_ENDIAN)` caller would have written --
    /// not a shape-plausible but wrong stand-in.
    #[test]
    fn test_bytes_to_f32_tensor_decodes_real_values() {
        let values = [1.0f32, -2.5, 3.25, 0.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let tensor = bytes_to_f32_tensor(&bytes).expect("well-formed 16-byte buffer must decode");
        assert_eq!(tensor.shape(), vec![4]);
        assert_eq!(tensor.data().expect("data"), values.to_vec());
    }

    /// A byte length that is not a multiple of 4 cannot be a flat f32
    /// buffer and must error -- a `chunks(4)`-based decoder would have
    /// silently dropped the trailing remainder instead of rejecting it.
    #[test]
    fn test_bytes_to_f32_tensor_rejects_misaligned_length() {
        let bytes = vec![0u8; 6];
        assert!(bytes_to_f32_tensor(&bytes).is_err());
    }

    /// Regression test for the P0 finding: the previous JNI `inference`
    /// entry point computed `output_tensor` and then discarded it,
    /// returning a hardcoded `vec![0; 4]` regardless of what inference
    /// produced. `tensor_to_le_bytes` is the piece that must now carry the
    /// real values through.
    #[test]
    fn test_tensor_to_le_bytes_round_trips_real_computed_output() {
        let output = Tensor::from_vec(vec![7.5f32, -1.0, 42.0], &[3]).expect("tensor");
        let bytes = tensor_to_le_bytes(&output).expect("encode");

        assert_eq!(bytes.len(), 12);
        assert_ne!(
            bytes,
            vec![0u8; 12],
            "encoded bytes must carry the real tensor values, not a hardcoded zero placeholder"
        );

        let decoded = bytes_to_f32_tensor(&bytes).expect("decode back");
        assert_eq!(decoded.data().expect("data"), vec![7.5, -1.0, 42.0]);
    }

    /// End-to-end: encode -> decode must be the identity for any real
    /// tensor values, confirming the two functions agree on byte order and
    /// element width.
    #[test]
    fn test_bytes_and_tensor_round_trip_is_identity() {
        let original = vec![0.1f32, 123.456, -9999.0, 0.0, -0.0, f32::MIN, f32::MAX];
        let tensor = Tensor::from_vec(original.clone(), &[original.len()]).expect("tensor");

        let bytes = tensor_to_le_bytes(&tensor).expect("encode");
        let round_tripped = bytes_to_f32_tensor(&bytes).expect("decode");

        assert_eq!(round_tripped.data().expect("data"), original);
    }

    /// An empty buffer is technically a multiple of 4 (zero times) and
    /// should decode to a valid, empty rank-1 tensor rather than erroring.
    #[test]
    fn test_empty_buffer_decodes_to_empty_tensor() {
        let tensor = bytes_to_f32_tensor(&[]).expect("empty buffer is a valid zero-length tensor");
        assert_eq!(tensor.shape(), vec![0]);
    }
}
