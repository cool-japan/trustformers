//! # MetalBackend - rope_f32_group Methods
//!
//! This module contains method implementations for `MetalBackend`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
use super::common::*;

use super::metalbackend_type::MetalBackend;

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalBackend {
    /// Execute RoPE (Rotary Position Embedding) on GPU
    /// Rotates Q and K tensors for position encoding
    /// Input/output shape: [seq_len, num_heads, head_dim]
    ///
    /// # Synchronisation
    ///
    /// This is a host-in / host-out entry point: the result is read straight out
    /// of the output `MTLBuffer` before returning, so the command buffer must be
    /// **waited on**, not merely committed. It previously used
    /// [`commit_async`](Self::commit_async) and then read `contents()` on the
    /// very next line, which raced the GPU and — because a freshly allocated
    /// `StorageModeShared` buffer starts zeroed — returned an all-zero vector
    /// essentially every time. This is the same defect
    /// [`layernorm_f32`](Self::layernorm_f32) was fixed for; see that method's
    /// note for why `commit_async` remains correct for the `*_gpu_to_gpu`
    /// methods and is never correct here.
    ///
    /// # Degenerate parameters
    ///
    /// A zero-element request returns an empty vector without dispatching:
    /// Metal rejects zero-length buffers.
    ///
    /// `rotary_ndims == 0` rotates nothing, so the answer is the input itself.
    /// It is returned directly rather than dispatched, because the kernel
    /// derives its grid width from `rotary_ndims / 2` — a zero-threadgroup
    /// dispatch writes nothing at all, and the untouched output buffer would
    /// again read back as zeros. That is the same fabrication as the race
    /// above, reached through a different parameter.
    ///
    /// # Errors
    ///
    /// Returns a shape error when `input.len()` does not match
    /// `seq_len * num_heads * head_dim`, when `rotary_ndims` is odd (the kernel
    /// rotates the pairs `(i, i + rotary_ndims / 2)`, which is undefined for an
    /// odd count), or when `rotary_ndims > head_dim` (the partner index would
    /// then run past the end of the head's slice and corrupt the neighbouring
    /// head — or, for the final head of the final position, run past the end of
    /// the buffer). Returns a hardware error when the output buffer's contents
    /// pointer is null.
    pub fn rope_f32(
        &self,
        input: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_ndims: usize,
        base: f32,
    ) -> Result<Vec<f32>> {
        let total_size = seq_len * num_heads * head_dim;
        if input.len() != total_size {
            return Err(TrustformersError::shape_error(format!(
                "Input size {} doesn't match seq_len {} * num_heads {} * head_dim {}",
                input.len(),
                seq_len,
                num_heads,
                head_dim
            )));
        }
        if !rotary_ndims.is_multiple_of(2) {
            return Err(TrustformersError::shape_error(format!(
                "rotary_ndims {} must be even: RoPE rotates the pairs (i, i + rotary_ndims / 2)",
                rotary_ndims
            )));
        }
        if rotary_ndims > head_dim {
            return Err(TrustformersError::shape_error(format!(
                "rotary_ndims {} exceeds head_dim {}: the rotation partner index would run past \
                 the end of the head",
                rotary_ndims, head_dim
            )));
        }
        if total_size == 0 {
            // Nothing to rotate: Metal rejects zero-length buffers and a
            // zero-threadgroup dispatch, so answer directly.
            return Ok(Vec::new());
        }
        if rotary_ndims == 0 {
            // Rotating no dimensions is the identity. See "Degenerate
            // parameters" above: dispatching here would write nothing and read
            // back zeros.
            return Ok(input.to_vec());
        }
        let input_buffer = self.create_buffer(input)?;
        let output_buffer = self.device.new_buffer(
            (total_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rope_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        let seq_len_u32 = seq_len as u32;
        let num_heads_u32 = num_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let rotary_ndims_u32 = rotary_ndims as u32;
        encoder.set_bytes(
            2,
            mem::size_of::<u32>() as u64,
            &seq_len_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &num_heads_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &head_dim_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &rotary_ndims_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            mem::size_of::<f32>() as u64,
            &base as *const f32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 8,
            height: 4,
            depth: 4,
        };
        let threadgroups = metal::MTLSize {
            width: ((rotary_ndims / 2) as u64).div_ceil(8),
            height: (num_heads as u64).div_ceil(4),
            depth: (seq_len as u64).div_ceil(4),
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        // CRITICAL: the readback below dereferences the output buffer directly,
        // so the kernel must have finished writing it first. See the
        // "Synchronisation" note on this method.
        command_buffer.wait_until_completed();

        let result_ptr = output_buffer.contents();
        if result_ptr.is_null() {
            return Err(TrustformersError::hardware_error(
                "GPU buffer contents pointer is null",
                "MetalBackend::rope_f32",
            ));
        }
        let result_ptr = result_ptr as *const f32;
        let result = unsafe { std::slice::from_raw_parts(result_ptr, total_size) }.to_vec();
        Ok(result)
    }
}

#[cfg(all(target_os = "macos", feature = "metal", test))]
mod tests {
    use super::super::functions::get_metal_backend;

    /// Reference CPU RoPE, matching the `rope` kernel in
    /// `metalbackend_new_group.rs`: the pair `(i, i + rotary_ndims / 2)` is
    /// rotated by `pos / base^(2i / rotary_ndims)`, and every dimension at or
    /// above `rotary_ndims` is copied through unchanged.
    fn cpu_rope(
        input: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_ndims: usize,
        base: f32,
    ) -> Vec<f32> {
        // Starting from a copy gives the untouched trailing dimensions for free.
        let mut out = input.to_vec();
        let half = rotary_ndims / 2;
        for pos in 0..seq_len {
            for head in 0..num_heads {
                let row = pos * num_heads * head_dim + head * head_dim;
                for i in 0..half {
                    let j = i + half;
                    let freq = 1.0_f32 / base.powf(2.0 * i as f32 / rotary_ndims as f32);
                    let angle = pos as f32 * freq;
                    let (sin_val, cos_val) = angle.sin_cos();
                    let x_i = input[row + i];
                    let x_j = input[row + j];
                    out[row + i] = x_i * cos_val - x_j * sin_val;
                    out[row + j] = x_i * sin_val + x_j * cos_val;
                }
            }
        }
        out
    }

    /// The host-in / host-out kernel must return the *computed* values, not the
    /// zeroed contents of a buffer the GPU has not written yet.
    ///
    /// This is the regression test for the read-after-async-commit race: before
    /// the fix this method committed the command buffer asynchronously and
    /// dereferenced `output_buffer.contents()` on the next line, so a freshly
    /// allocated (zeroed) `StorageModeShared` buffer was read back essentially
    /// every time.
    #[test]
    fn rope_f32_matches_the_cpu_reference_instead_of_returning_zeros() {
        let Ok(backend) = get_metal_backend() else {
            eprintln!("no Metal device; skipping rope_f32 parity test");
            return;
        };

        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 6;
        // Deliberately smaller than head_dim so the kernel's "copy the
        // non-rotated dimensions" branch is exercised too.
        let rotary_ndims = 4;
        let base = 10_000.0_f32;
        let input: Vec<f32> = (0..seq_len * num_heads * head_dim)
            .map(|i| ((i % 7) as f32) * 0.3 - 1.0)
            .collect();

        let gpu = backend
            .rope_f32(&input, seq_len, num_heads, head_dim, rotary_ndims, base)
            .expect("rope_f32 must run on a Metal device");
        let expected = cpu_rope(&input, seq_len, num_heads, head_dim, rotary_ndims, base);

        assert_eq!(gpu.len(), expected.len(), "output length must match");
        assert!(
            gpu.iter().any(|v| v.abs() > 1e-6),
            "the GPU result is all zeros, which is what the read-after-commit race produced"
        );
        for (index, (got, want)) in gpu.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "element {index}: GPU {got} vs CPU reference {want}"
            );
        }

        // The result must actually depend on the rotation, not merely be a copy
        // of the input: positions beyond 0 rotate by a non-zero angle.
        let rotated_changed = gpu
            .iter()
            .zip(input.iter())
            .enumerate()
            .filter(|(index, _)| index % head_dim < rotary_ndims)
            .any(|(_, (got, original))| (got - original).abs() > 1e-4);
        assert!(
            rotated_changed,
            "no rotated element changed; the kernel would be an identity function"
        );

        // ...while the dimensions at or above `rotary_ndims` are copied verbatim.
        for (index, (got, original)) in gpu.iter().zip(input.iter()).enumerate() {
            if index % head_dim >= rotary_ndims {
                assert!(
                    (got - original).abs() < 1e-6,
                    "element {index} is outside the rotary slice and must be copied unchanged: \
                     got {got}, input {original}"
                );
            }
        }
    }

    /// `rotary_ndims == 0` rotates nothing. The kernel derives its grid width
    /// from `rotary_ndims / 2`, so dispatching would write nothing and read back
    /// as zeros — the identity must be answered directly.
    #[test]
    fn rope_f32_with_no_rotary_dimensions_is_the_identity() {
        let Ok(backend) = get_metal_backend() else {
            eprintln!("no Metal device; skipping rope_f32 zero-rotary test");
            return;
        };
        let input: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 2.0).collect();
        let result = backend
            .rope_f32(&input, 2, 2, 3, 0, 10_000.0)
            .expect("a zero-rotary request must not fail");
        assert_eq!(
            result, input,
            "rotating zero dimensions must return the input, not a zeroed buffer"
        );
    }

    /// A zero-element request is answered directly rather than by dispatching a
    /// zero-threadgroup kernel over zero-length Metal buffers.
    #[test]
    fn rope_f32_handles_an_empty_request() {
        let Ok(backend) = get_metal_backend() else {
            eprintln!("no Metal device; skipping rope_f32 empty-request test");
            return;
        };
        let result = backend
            .rope_f32(&[], 0, 2, 4, 4, 10_000.0)
            .expect("an empty request must not fail");
        assert!(result.is_empty(), "an empty request must return no values");
    }

    /// `rotary_ndims > head_dim` makes the kernel's partner index run past the
    /// end of the head. It must be refused, not dispatched.
    #[test]
    fn rope_f32_rejects_a_rotary_slice_wider_than_the_head() {
        let Ok(backend) = get_metal_backend() else {
            eprintln!("no Metal device; skipping rope_f32 oversized-rotary test");
            return;
        };
        let input = vec![0.25_f32; 8];
        let error = backend
            .rope_f32(&input, 2, 1, 4, 6, 10_000.0)
            .expect_err("rotary_ndims wider than head_dim must be refused");
        let message = error.to_string();
        assert!(
            message.contains("head_dim"),
            "the error must name the offending dimension, got: {message}"
        );
    }

    /// An odd `rotary_ndims` has no well-defined pairing.
    #[test]
    fn rope_f32_rejects_an_odd_rotary_width() {
        let Ok(backend) = get_metal_backend() else {
            eprintln!("no Metal device; skipping rope_f32 odd-rotary test");
            return;
        };
        let input = vec![0.25_f32; 8];
        let error = backend
            .rope_f32(&input, 2, 1, 4, 3, 10_000.0)
            .expect_err("an odd rotary_ndims must be refused");
        let message = error.to_string();
        assert!(
            message.contains("even"),
            "the error must say the width has to be even, got: {message}"
        );
    }
}
