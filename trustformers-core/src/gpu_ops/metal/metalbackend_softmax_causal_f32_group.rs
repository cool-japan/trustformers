//! # MetalBackend - softmax_causal_f32_group Methods
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
    /// Execute Softmax with causal mask on GPU
    /// Applies causal mask: position i can only attend to j <= i
    /// Input/output shape: [seq_len, seq_len]
    ///
    /// # Synchronisation
    ///
    /// This is a host-in / host-out entry point: the result is read straight out
    /// of the output `MTLBuffer` before returning, so the command buffer must be
    /// **waited on**, not merely committed. It previously used
    /// [`commit_async`](Self::commit_async) and then read `contents()` on the
    /// very next line, which raced the GPU and — because a freshly allocated
    /// `StorageModeShared` buffer starts zeroed — returned an all-zero attention
    /// weight matrix essentially every time. That is not merely a wrong number:
    /// an all-zero softmax row is not a probability distribution at all, so
    /// every consumer downstream (see `gpt_neox`'s attention block, which calls
    /// this method) silently produced a zero context vector. This is the same
    /// defect [`layernorm_f32`](Self::layernorm_f32) was fixed for; see that
    /// method's note for why `commit_async` remains correct for the
    /// `*_gpu_to_gpu` methods and is never correct here.
    ///
    /// # Errors
    ///
    /// Returns a shape error when `input.len()` does not equal `seq_len * seq_len`,
    /// and a hardware error when the output buffer's contents pointer is null.
    pub fn softmax_causal_f32(&self, input: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        let total_size = seq_len * seq_len;
        if input.len() != total_size {
            return Err(TrustformersError::shape_error(format!(
                "Input size {} doesn't match seq_len^2 {}",
                input.len(),
                total_size
            )));
        }
        if total_size == 0 {
            // Nothing to normalise: Metal rejects zero-length buffers and a
            // zero-threadgroup dispatch, so answer directly.
            return Ok(Vec::new());
        }
        let input_buffer = self.create_buffer(input)?;
        let output_buffer = self.device.new_buffer(
            (total_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.softmax_causal_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        let seq_len_u32 = seq_len as u32;
        encoder.set_bytes(
            2,
            mem::size_of::<u32>() as u64,
            &seq_len_u32 as *const u32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (seq_len as u64).div_ceil(64),
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        // CRITICAL: the readback below dereferences the output buffer directly,
        // so the kernel must have finished writing it first. See the
        // "Synchronisation" note on this method.
        //
        // Committing here rather than through `commit_async` deliberately keeps
        // this buffer out of `pending_command_buffers`: that list exists so
        // `flush()` can wait on work that is still in flight, and this command
        // buffer is already complete by the time the next line runs. Recording
        // it would only leave a finished entry for the next `commit_async` to
        // retain-scan away.
        command_buffer.wait_until_completed();

        let result_ptr = output_buffer.contents();
        if result_ptr.is_null() {
            return Err(TrustformersError::hardware_error(
                "GPU buffer contents pointer is null",
                "MetalBackend::softmax_causal_f32",
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

    /// Reference CPU causal softmax, matching the `softmax_causal` kernel in
    /// `metalbackend_new_group.rs` for inputs in a normal numeric range.
    ///
    /// The kernel additionally carries two degenerate branches (an all `-inf`
    /// row, and a row whose exponent sum underflows) that both emit
    /// `[1, 0, 0, …]`. They are deliberately *not* reproduced here: the test
    /// inputs stay in a range where neither can fire, so this reference is a
    /// plain causal softmax and the comparison actually checks the arithmetic
    /// rather than a shared fallback.
    fn cpu_softmax_causal(input: &[f32], seq_len: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; seq_len * seq_len];
        for row in 0..seq_len {
            let offset = row * seq_len;
            let visible = &input[offset..=offset + row];
            let max_val = visible.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = visible.iter().map(|v| (v - max_val).exp()).sum();
            for (col, value) in visible.iter().enumerate() {
                out[offset + col] = (value - max_val).exp() / sum;
            }
        }
        out
    }

    /// The host-in / host-out kernel must return the *computed* weights, not the
    /// zeroed contents of a buffer the GPU has not written yet.
    ///
    /// This is the regression test for the read-after-async-commit race: before
    /// the fix this method committed the command buffer asynchronously and
    /// dereferenced `output_buffer.contents()` on the next line, so a freshly
    /// allocated (zeroed) `StorageModeShared` buffer was read back essentially
    /// every time — an "attention weight" matrix whose rows summed to zero.
    #[test]
    fn softmax_causal_f32_matches_the_cpu_reference_instead_of_returning_zeros() {
        let Ok(backend) = get_metal_backend() else {
            eprintln!("no Metal device; skipping softmax_causal_f32 parity test");
            return;
        };

        let seq_len = 6;
        // Ordinary logit magnitudes: neither of the kernel's degenerate
        // branches can fire, so the comparison is against real arithmetic.
        let input: Vec<f32> =
            (0..seq_len * seq_len).map(|i| ((i % 5) as f32) * 0.5 - 1.0).collect();

        let gpu = backend
            .softmax_causal_f32(&input, seq_len)
            .expect("softmax_causal_f32 must run on a Metal device");
        let expected = cpu_softmax_causal(&input, seq_len);

        assert_eq!(gpu.len(), expected.len(), "output length must match");
        assert!(
            gpu.iter().any(|v| v.abs() > 1e-6),
            "the GPU result is all zeros, which is what the read-after-commit race produced"
        );
        for (index, (got, want)) in gpu.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "element {index}: GPU {got} vs CPU reference {want}"
            );
        }

        // Every row is a real probability distribution over the visible prefix.
        for row in 0..seq_len {
            let offset = row * seq_len;
            let row_sum: f32 = gpu[offset..offset + seq_len].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "row {row} sums to {row_sum}, not 1.0"
            );
            for (col, value) in gpu[offset..offset + seq_len].iter().enumerate() {
                if col > row {
                    assert_eq!(
                        *value, 0.0,
                        "row {row} column {col} is above the diagonal and must be masked to zero"
                    );
                }
            }
        }
    }

    /// The output must depend on the input: two different score matrices may not
    /// produce the same weights.
    #[test]
    fn softmax_causal_f32_is_input_dependent() {
        let Ok(backend) = get_metal_backend() else {
            eprintln!("no Metal device; skipping softmax_causal_f32 input-dependence test");
            return;
        };
        let seq_len = 4;
        let flat: Vec<f32> = vec![0.0_f32; seq_len * seq_len];
        let peaked: Vec<f32> = (0..seq_len * seq_len)
            .map(|i| if i % seq_len == 0 { 4.0 } else { 0.0 })
            .collect();

        let flat_weights =
            backend.softmax_causal_f32(&flat, seq_len).expect("uniform scores must encode");
        let peaked_weights =
            backend.softmax_causal_f32(&peaked, seq_len).expect("peaked scores must encode");
        assert_ne!(
            flat_weights, peaked_weights,
            "different attention scores must produce different weights"
        );
        // The uniform case has a closed form: row r spreads 1/(r+1) over its
        // visible prefix.
        for row in 0..seq_len {
            let expected = 1.0_f32 / (row + 1) as f32;
            for col in 0..=row {
                let got = flat_weights[row * seq_len + col];
                assert!(
                    (got - expected).abs() < 1e-5,
                    "uniform row {row} column {col}: got {got}, expected {expected}"
                );
            }
        }
    }

    /// A zero-element request is answered directly rather than by dispatching a
    /// zero-threadgroup kernel over zero-length Metal buffers.
    #[test]
    fn softmax_causal_f32_handles_an_empty_request() {
        let Ok(backend) = get_metal_backend() else {
            eprintln!("no Metal device; skipping softmax_causal_f32 empty-request test");
            return;
        };
        let result = backend.softmax_causal_f32(&[], 0).expect("an empty request must not fail");
        assert!(result.is_empty(), "an empty request must return no values");
    }
}
