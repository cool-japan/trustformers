use super::*;

#[test]
fn oxicuda_cuda_matmul_parity() -> crate::errors::Result<()> {
    // A is [m=2, k=3], B is [k=3, n=2], row-major.
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let m = 2usize;
    let k = 3usize;
    let n = 2usize;

    // Naive CPU reference, row-major.
    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                expected[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }

    let backend = match OxicudaCudaBackend::new(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping oxicuda CUDA test: no CUDA device available");
            return Ok(());
        },
    };

    let result = backend.matmul_f32(&a, &b, m, k, n)?;

    for idx in 0..(m * n) {
        assert!(
            (result[idx] - expected[idx]).abs() < 1e-3,
            "mismatch at {}: got {} expected {}",
            idx,
            result[idx],
            expected[idx]
        );
    }

    Ok(())
}

#[test]
fn oxicuda_cuda_gelu_parity() -> crate::errors::Result<()> {
    // Moderate values only (oxicuda GELU lacks the cudarc ±10 clamps).
    let input = vec![-1.0f32, -0.5, 0.0, 0.5, 1.0, 2.0];

    // CPU reference: tanh-approximation GELU.
    let mut expected = vec![0.0f32; input.len()];
    for (idx, &x) in input.iter().enumerate() {
        let inner = 0.7978845608f32 * (x + 0.044715f32 * x * x * x);
        expected[idx] = 0.5f32 * x * (1.0f32 + inner.tanh());
    }

    let backend = match OxicudaCudaBackend::new(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping oxicuda CUDA GELU test: no CUDA device available");
            return Ok(());
        },
    };

    let result = backend.gelu_f32(&input)?;

    for idx in 0..input.len() {
        assert!(
            (result[idx] - expected[idx]).abs() < 1e-3,
            "mismatch at {}: got {} expected {}",
            idx,
            result[idx],
            expected[idx]
        );
    }

    Ok(())
}

#[test]
fn oxicuda_cuda_layernorm_parity() -> crate::errors::Result<()> {
    let seq_len = 2usize;
    let hidden_size = 4usize;
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
    let weight = vec![1.0f32; hidden_size];
    let bias = vec![0.0f32; hidden_size];
    let eps = 1e-5f32;

    // CPU reference: per-row population-variance layer norm.
    let mut expected = vec![0.0f32; seq_len * hidden_size];
    for row in 0..seq_len {
        let offset = row * hidden_size;
        let mut sum = 0.0f32;
        for i in 0..hidden_size {
            sum += input[offset + i];
        }
        let mean = sum / hidden_size as f32;
        let mut var_sum = 0.0f32;
        for i in 0..hidden_size {
            let diff = input[offset + i] - mean;
            var_sum += diff * diff;
        }
        let variance = var_sum / hidden_size as f32;
        let std_dev = (variance + eps).sqrt();
        for i in 0..hidden_size {
            let normalized = (input[offset + i] - mean) / std_dev;
            expected[offset + i] = normalized * weight[i] + bias[i];
        }
    }

    let backend = match OxicudaCudaBackend::new(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping oxicuda CUDA LayerNorm test: no CUDA device available");
            return Ok(());
        },
    };

    let result = backend.layernorm_f32(&input, &weight, &bias, seq_len, hidden_size, eps)?;

    for idx in 0..(seq_len * hidden_size) {
        assert!(
            (result[idx] - expected[idx]).abs() < 1e-3,
            "mismatch at {}: got {} expected {}",
            idx,
            result[idx],
            expected[idx]
        );
    }

    Ok(())
}

#[test]
fn oxicuda_cuda_softmax_causal_parity() -> crate::errors::Result<()> {
    let seq_len = 4usize;
    let total = seq_len * seq_len;

    // Deterministic varied input.
    let mut input = vec![0.0f32; total];
    for (i, slot) in input.iter_mut().enumerate() {
        *slot = (i as f32 * 0.37 - 2.0).sin();
    }

    // CPU reference: causal (lower-triangular) softmax, no scaling.
    let mut expected = vec![0.0f32; total];
    for r in 0..seq_len {
        let offset = r * seq_len;
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..=r {
            let v = input[offset + j];
            if v > max_val {
                max_val = v;
            }
        }
        let mut sum = 0.0f32;
        for j in 0..=r {
            sum += (input[offset + j] - max_val).exp();
        }
        for j in 0..seq_len {
            if j <= r {
                expected[offset + j] = (input[offset + j] - max_val).exp() / sum;
            } else {
                expected[offset + j] = 0.0f32;
            }
        }
    }

    let backend = match OxicudaCudaBackend::new(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping oxicuda CUDA causal-softmax test: no CUDA device available");
            return Ok(());
        },
    };

    let result = backend.softmax_causal_f32(&input, seq_len)?;

    for idx in 0..total {
        assert!(
            (result[idx] - expected[idx]).abs() < 1e-3,
            "mismatch at {}: got {} expected {}",
            idx,
            result[idx],
            expected[idx]
        );
    }

    Ok(())
}

#[test]
fn oxicuda_cuda_rope_parity() -> crate::errors::Result<()> {
    // Partial-rotary half-split case: head_dim=6, rotary_ndims=4 (half=2),
    // so dims 0<->2 and 1<->3 rotate, dims 4,5 pass through.
    let seq_len = 2usize;
    let num_heads = 1usize;
    let head_dim = 6usize;
    let rotary_ndims = 4usize;
    let base = 10000.0f32;
    let total = seq_len * num_heads * head_dim;

    // Deterministic varied input.
    let mut input = vec![0.0f32; total];
    for (idx, slot) in input.iter_mut().enumerate() {
        *slot = (idx as f32) * 0.5 + 0.1;
    }

    // CPU reference: GPT-NeoX half-split partial RoPE.
    let mut expected = vec![0.0f32; total];
    let half = rotary_ndims / 2;
    for pos in 0..seq_len {
        for h in 0..num_heads {
            let base_off = (pos * num_heads + h) * head_dim;
            for i in 0..half {
                let freq = base.powf(-2.0 * (i as f32) / (rotary_ndims as f32));
                let angle = (pos as f32) * freq;
                let c = angle.cos();
                let s = angle.sin();
                let x_i = input[base_off + i];
                let x_j = input[base_off + i + half];
                expected[base_off + i] = x_i * c - x_j * s;
                expected[base_off + i + half] = x_i * s + x_j * c;
            }
            expected[(base_off + rotary_ndims)..(base_off + head_dim)]
                .copy_from_slice(&input[(base_off + rotary_ndims)..(base_off + head_dim)]);
        }
    }

    let backend = match OxicudaCudaBackend::new(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping oxicuda CUDA RoPE test: no CUDA device available");
            return Ok(());
        },
    };

    let result = backend.rope_f32(&input, seq_len, num_heads, head_dim, rotary_ndims, base)?;

    for idx in 0..total {
        assert!(
            (result[idx] - expected[idx]).abs() < 1e-3,
            "mismatch at {}: got {} expected {}",
            idx,
            result[idx],
            expected[idx]
        );
    }

    Ok(())
}

#[test]
fn oxicuda_cuda_resident_matmul_parity() -> crate::errors::Result<()> {
    // A is [m=2, k=3], B is [k=3, n=2], row-major.
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let m = 2usize;
    let k = 3usize;
    let n = 2usize;

    // Naive CPU reference (triple loop), row-major.
    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                expected[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }

    let backend = match OxicudaCudaBackend::new(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping oxicuda CUDA resident-matmul test: no CUDA device available");
            return Ok(());
        },
    };

    // Empty cache to start.
    assert_eq!(backend.buffer_cache_size()?, 0);

    // Upload both operands as resident persistent buffers.
    let a_id = backend.create_persistent_buffer(&a)?;
    let b_id = backend.create_persistent_buffer(&b)?;
    assert_eq!(backend.buffer_cache_size()?, 2);
    assert_eq!(backend.get_persistent_buffer(&a_id)?, m * k);
    assert_eq!(backend.get_persistent_buffer(&b_id)?, k * n);

    // GPU-to-GPU matmul: result stays resident.
    let c_id = backend.matmul_gpu_to_gpu(&a_id, &b_id, m, k, n)?;
    assert_eq!(backend.buffer_cache_size()?, 3);

    // Download and compare to the CPU reference.
    let result = backend.download_buffer(&c_id)?;
    assert_eq!(result.len(), m * n);
    for idx in 0..(m * n) {
        assert!(
            (result[idx] - expected[idx]).abs() < 1e-3,
            "mismatch at {}: got {} expected {}",
            idx,
            result[idx],
            expected[idx]
        );
    }

    // `buffer_to_cpu` is an alias of `download_buffer`.
    let result_alias = backend.buffer_to_cpu(&c_id, m * n)?;
    assert_eq!(result_alias, result);

    // Exercise remove + clear + size bookkeeping.
    backend.remove_persistent_buffer(&a_id)?;
    assert_eq!(backend.buffer_cache_size()?, 2);
    // Removing an absent id is a no-op.
    backend.remove_persistent_buffer(&a_id)?;
    assert_eq!(backend.buffer_cache_size()?, 2);

    backend.clear_buffer_cache()?;
    assert_eq!(backend.buffer_cache_size()?, 0);

    // A zeroed resident allocation reads back as all zeros.
    let z_id = backend.create_persistent_buffer_zeroed(4)?;
    let zeros = backend.download_buffer(&z_id)?;
    assert_eq!(zeros, vec![0.0f32; 4]);

    Ok(())
}

#[test]
fn oxicuda_cuda_resident_gelu_parity() -> crate::errors::Result<()> {
    // Moderate values only (oxicuda GELU lacks the cudarc ±10 clamps).
    let input = vec![-1.0f32, -0.5, 0.0, 0.5, 1.0, 2.0];
    let size = input.len();

    // CPU reference: tanh-approximation GELU.
    let mut expected = vec![0.0f32; size];
    for (idx, &x) in input.iter().enumerate() {
        let inner = 0.7978845608f32 * (x + 0.044715f32 * x * x * x);
        expected[idx] = 0.5f32 * x * (1.0f32 + inner.tanh());
    }

    let backend = match OxicudaCudaBackend::new(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping oxicuda CUDA resident-GELU test: no CUDA device available");
            return Ok(());
        },
    };

    let in_id = backend.create_persistent_buffer(&input)?;
    let out_id = backend.gelu_gpu_to_gpu(&in_id, size)?;
    let result = backend.download_buffer(&out_id)?;
    assert_eq!(result.len(), size);

    for idx in 0..size {
        assert!(
            (result[idx] - expected[idx]).abs() < 1e-3,
            "mismatch at {}: got {} expected {}",
            idx,
            result[idx],
            expected[idx]
        );
    }

    Ok(())
}

#[test]
fn oxicuda_cuda_resident_add_bias_parity() -> crate::errors::Result<()> {
    // Input is [m=3, n=4] row-major; bias is length n=4, broadcast down each row.
    let m = 3usize;
    let n = 4usize;
    let input: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.5 - 2.0).collect();
    let bias = vec![10.0f32, 20.0, 30.0, 40.0];

    // CPU reference: out[i, j] = input[i, j] + bias[j].
    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            expected[i * n + j] = input[i * n + j] + bias[j];
        }
    }

    let backend = match OxicudaCudaBackend::new(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping oxicuda CUDA resident-add-bias test: no CUDA device available");
            return Ok(());
        },
    };

    let in_id = backend.create_persistent_buffer(&input)?;
    let bias_id = backend.create_persistent_buffer(&bias)?;
    let out_id = backend.add_bias_gpu_to_gpu(&in_id, &bias_id, m, n)?;
    let result = backend.download_buffer(&out_id)?;
    assert_eq!(result.len(), m * n);

    for idx in 0..(m * n) {
        assert!(
            (result[idx] - expected[idx]).abs() < 1e-3,
            "mismatch at {}: got {} expected {}",
            idx,
            result[idx],
            expected[idx]
        );
    }

    Ok(())
}

#[test]
fn oxicuda_cuda_resident_layernorm_parity() -> crate::errors::Result<()> {
    let seq_len = 2usize;
    let hidden_size = 4usize;
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
    let weight = vec![1.0f32; hidden_size];
    let bias = vec![0.0f32; hidden_size];
    let eps = 1e-5f32;

    // CPU reference: per-row population-variance layer norm.
    let mut expected = vec![0.0f32; seq_len * hidden_size];
    for row in 0..seq_len {
        let offset = row * hidden_size;
        let mut sum = 0.0f32;
        for i in 0..hidden_size {
            sum += input[offset + i];
        }
        let mean = sum / hidden_size as f32;
        let mut var_sum = 0.0f32;
        for i in 0..hidden_size {
            let diff = input[offset + i] - mean;
            var_sum += diff * diff;
        }
        let variance = var_sum / hidden_size as f32;
        let std_dev = (variance + eps).sqrt();
        for i in 0..hidden_size {
            let normalized = (input[offset + i] - mean) / std_dev;
            expected[offset + i] = normalized * weight[i] + bias[i];
        }
    }

    let backend = match OxicudaCudaBackend::new(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping oxicuda CUDA resident-LayerNorm test: no CUDA device available");
            return Ok(());
        },
    };

    let in_id = backend.create_persistent_buffer(&input)?;
    let weight_id = backend.create_persistent_buffer(&weight)?;
    let bias_id = backend.create_persistent_buffer(&bias)?;

    let out_id =
        backend.layernorm_gpu_to_gpu(&in_id, &weight_id, &bias_id, seq_len, hidden_size, eps)?;
    let result = backend.download_buffer(&out_id)?;
    assert_eq!(result.len(), seq_len * hidden_size);

    for idx in 0..(seq_len * hidden_size) {
        assert!(
            (result[idx] - expected[idx]).abs() < 1e-3,
            "mismatch at {}: got {} expected {}",
            idx,
            result[idx],
            expected[idx]
        );
    }

    Ok(())
}

#[test]
fn oxicuda_cuda_matmul_with_cached_weight_parity() -> crate::errors::Result<()> {
    // A is [m=2, k=3] (host activations), B is [k=3, n=2] (cached weight), row-major.
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let m = 2usize;
    let k = 3usize;
    let n = 2usize;

    // Naive CPU reference (triple loop), row-major.
    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                expected[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }

    let backend = match OxicudaCudaBackend::new(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping oxicuda CUDA cached-weight matmul test: no CUDA device available");
            return Ok(());
        },
    };

    // Park the weight on the device, then multiply host activations against it twice
    // (the cached weight must survive repeated forward passes).
    let weight_id = backend.create_persistent_buffer(&b)?;
    for _ in 0..2 {
        let result = backend.matmul_with_cached_weight(&a, &weight_id, m, k, n)?;
        assert_eq!(result.len(), m * n);
        for idx in 0..(m * n) {
            assert!(
                (result[idx] - expected[idx]).abs() < 1e-3,
                "mismatch at {}: got {} expected {}",
                idx,
                result[idx],
                expected[idx]
            );
        }
    }

    Ok(())
}

#[test]
fn oxicuda_cuda_device_info_reports_ordinal() -> crate::errors::Result<()> {
    let backend = match OxicudaCudaBackend::new(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping oxicuda CUDA device-info test: no CUDA device available");
            return Ok(());
        },
    };

    let info = backend.device_info();
    assert!(
        info.contains("ordinal: 0"),
        "device_info should report device ordinal 0, got {:?}",
        info
    );

    Ok(())
}

#[test]
fn oxicuda_backend_singleton_is_shared_and_resident() -> crate::errors::Result<()> {
    // Probe whether a CUDA device exists; skip gracefully if not (CI without a GPU).
    if OxicudaCudaBackend::new(0).is_err() {
        eprintln!("Skipping oxicuda backend singleton test: no CUDA device available");
        return Ok(());
    }

    // Two get-or-create calls for the same device must return the *same* backend Arc.
    let b1 = oxicuda_backend(0)?;
    let b2 = oxicuda_backend(0)?;
    assert!(
        Arc::ptr_eq(&b1, &b2),
        "oxicuda_backend(0) must return the same Arc on repeated calls"
    );

    // A resident buffer created through one handle is visible through a later handle
    // obtained from the singleton — proving the backend (and its cache) persists.
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let id = b1.create_persistent_buffer(&data)?;
    let b3 = oxicuda_backend(0)?;
    assert_eq!(
        b3.get_persistent_buffer(&id)?,
        data.len(),
        "resident buffer minted via the singleton must be visible to a later handle"
    );
    let round_trip = b3.download_buffer(&id)?;
    assert_eq!(round_trip, data);

    // Clean up so this test leaves no resident state behind for sibling tests.
    b3.remove_persistent_buffer(&id)?;

    Ok(())
}
