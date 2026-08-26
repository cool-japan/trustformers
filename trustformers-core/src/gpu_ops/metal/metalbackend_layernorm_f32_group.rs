//! # MetalBackend - layernorm_f32_group Methods
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
    /// Execute LayerNorm on GPU
    /// LayerNorm: output = (x - mean) / sqrt(var + eps) * weight + bias
    /// Optimized for transformer models (normalize over hidden dimension)
    ///
    /// # Synchronisation
    ///
    /// This is a host-in / host-out entry point: the result is read straight out
    /// of the output `MTLBuffer` before returning, so the command buffer must be
    /// **waited on**, not merely committed. It previously used
    /// `commit_async` and then read `contents()`
    /// immediately, which raced the GPU and — because a freshly allocated
    /// `StorageModeShared` buffer starts zeroed — returned an all-zero vector
    /// essentially every time. Any model whose `LayerNorm` took the 2-D
    /// `Tensor::F32` Metal fast path therefore produced an all-zero forward pass
    /// whenever the `metal` feature was compiled in, even on `Device::CPU`. Keep
    /// the wait here.
    ///
    /// `commit_async` is correct for the `*_gpu_to_gpu` methods, which hand a
    /// buffer id to the next kernel on the same queue and let
    /// `download_buffer_to_vec` take the barrier — every `commit_async` caller in
    /// `metalbackend_initialize_mps_group.rs` is of that kind. It is **not**
    /// correct for a host-out entry point. Two further host-out entry points
    /// carried the identical defect when this note was first written —
    /// [`rope_f32`](Self::rope_f32) and
    /// [`softmax_causal_f32`](Self::softmax_causal_f32), which both
    /// `commit_async`-ed and then dereferenced `output_buffer.contents()` on the
    /// very next line. Both have since been fixed the same way (commit, wait,
    /// null-check), each with its own GPU-versus-CPU-reference regression test,
    /// and each carries its own copy of this note. That closes the class: every
    /// remaining `commit_async` call site in this module hands its result on as
    /// a buffer id rather than reading it back on the host.
    pub fn layernorm_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let total_size = seq_len * hidden_size;
        if total_size == 0 {
            // Nothing to normalise: Metal rejects zero-length buffers and a
            // zero-threadgroup dispatch, so answer directly.
            return Ok(Vec::new());
        }
        if input.len() != total_size {
            return Err(TrustformersError::shape_error(format!(
                "Input size {} doesn't match seq_len {} * hidden_size {}",
                input.len(),
                seq_len,
                hidden_size
            )));
        }
        if weight.len() != hidden_size || bias.len() != hidden_size {
            return Err(TrustformersError::shape_error(
                "Weight/bias size must match hidden_size".to_string(),
            ));
        }
        let input_buffer = self.create_buffer(input)?;
        let weight_buffer = self.create_buffer(weight)?;
        let bias_buffer = self.create_buffer(bias)?;
        let output_buffer = self.device.new_buffer(
            (total_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.layernorm_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&bias_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        let seq_len_u32 = seq_len as u32;
        let hidden_size_u32 = hidden_size as u32;
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &seq_len_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &hidden_size_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            mem::size_of::<f32>() as u64,
            &eps as *const f32 as *const _,
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
        command_buffer.wait_until_completed();

        let result_ptr = output_buffer.contents();
        if result_ptr.is_null() {
            return Err(TrustformersError::hardware_error(
                "GPU buffer contents pointer is null",
                "MetalBackend::layernorm_f32",
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

    /// Reference CPU LayerNorm over the trailing dimension.
    fn cpu_layernorm(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0_f32; seq_len * hidden_size];
        for row in 0..seq_len {
            let start = row * hidden_size;
            let slice = &input[start..start + hidden_size];
            let n = hidden_size as f32;
            let mean = slice.iter().sum::<f32>() / n;
            let var = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let inv = 1.0 / (var + eps).sqrt();
            for col in 0..hidden_size {
                out[start + col] = (slice[col] - mean) * inv * weight[col] + bias[col];
            }
        }
        out
    }

    /// The host-in / host-out kernel must return the *computed* values, not the
    /// zeroed contents of a buffer the GPU has not written yet.
    ///
    /// This is the regression test for the read-after-async-commit race that made
    /// every `metal`-feature build return an all-zero LayerNorm (and therefore an
    /// all-zero BERT / xLSTM forward pass) on a pure-CPU `Device::CPU` model.
    #[test]
    fn layernorm_f32_matches_the_cpu_reference_instead_of_returning_zeros() {
        let Ok(backend) = get_metal_backend() else {
            eprintln!("no Metal device; skipping layernorm_f32 parity test");
            return;
        };

        let seq_len = 5;
        let hidden_size = 8;
        let eps = 1e-5_f32;
        let input: Vec<f32> =
            (0..seq_len * hidden_size).map(|i| ((i % 13) as f32) * 0.25 - 1.5).collect();
        let weight: Vec<f32> = (0..hidden_size).map(|i| 1.0 + i as f32 * 0.1).collect();
        let bias: Vec<f32> = (0..hidden_size).map(|i| -0.5 + i as f32 * 0.05).collect();

        let gpu = backend
            .layernorm_f32(&input, &weight, &bias, seq_len, hidden_size, eps)
            .expect("layernorm_f32 must run on a Metal device");
        let expected = cpu_layernorm(&input, &weight, &bias, seq_len, hidden_size, eps);

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
    }

    /// A zero-element request is answered directly rather than by dispatching a
    /// zero-threadgroup kernel over zero-length Metal buffers.
    #[test]
    fn layernorm_f32_handles_an_empty_request() {
        let Ok(backend) = get_metal_backend() else {
            eprintln!("no Metal device; skipping layernorm_f32 empty-request test");
            return;
        };
        let result = backend
            .layernorm_f32(&[], &[], &[], 0, 8, 1e-5)
            .expect("an empty request must not fail");
        assert!(result.is_empty(), "an empty request must return no values");
    }
}
