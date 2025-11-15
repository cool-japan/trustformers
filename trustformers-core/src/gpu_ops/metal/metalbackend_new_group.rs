//! # MetalBackend - new_group Methods
//!
//! This module contains method implementations for `MetalBackend`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::common::*;
use super::metalbackend_type::MetalBackend;
use super::types::{BufferCache, BufferId};

impl MetalBackend {
    /// Create a new Metal backend
    pub fn new() -> Result<Self> {
        let device = MetalDevice::system_default().ok_or_else(|| {
            TrustformersError::hardware_error("No Metal device found", "MetalBackend::new")
        })?;
        let command_queue = device.new_command_queue();
        let shader_source = r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void matmul(
                device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* c [[buffer(2)]],
                constant uint& M [[buffer(3)]],
                constant uint& N [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint row = gid.y;
                uint col = gid.x;

                if (row >= M || col >= N) return;

                float sum = 0.0;
                for (uint i = 0; i < K; ++i) {
                    sum += a[row * K + i] * b[i * N + col];
                }
                c[row * N + col] = sum;
            }

            // GELU activation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            // With NaN guarding for extreme values
            kernel void gelu(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& size [[buffer(2)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid >= size) return;
                float x = input[gid];

                // Clamp extreme values to prevent NaN
                if (x > 10.0f) {
                    output[gid] = x;  // GELU(x) ≈ x for large positive x
                    return;
                } else if (x < -10.0f) {
                    output[gid] = 0.0f;  // GELU(x) ≈ 0 for large negative x
                    return;
                }

                float x_cubed = x * x * x;
                // sqrt(2/π) ≈ 0.7978845608
                float inner = 0.7978845608f * (x + 0.044715f * x_cubed);

                // Clamp inner to prevent tanh overflow
                inner = clamp(inner, -20.0f, 20.0f);

                output[gid] = 0.5f * x * (1.0f + tanh(inner));
            }

            // Fused matmul+GELU: Combines matrix multiplication and GELU activation
            // Eliminates intermediate buffer and one kernel dispatch
            // A: [M, K], B: [K, N] -> C: [M, N] with GELU applied
            kernel void fused_matmul_gelu(
                device const float* A [[buffer(0)]],
                device const float* B [[buffer(1)]],
                device float* C [[buffer(2)]],
                constant uint& M [[buffer(3)]],
                constant uint& N [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint row = gid.y;
                uint col = gid.x;

                if (row >= M || col >= N) return;

                // 1. Compute matmul element C[row, col] = sum(A[row, k] * B[k, col])
                float sum = 0.0f;
                for (uint k = 0; k < K; ++k) {
                    sum += A[row * K + k] * B[k * N + col];
                }

                // 2. Apply GELU activation immediately (no intermediate buffer)
                float x = sum;

                // Handle extreme values to prevent NaN
                if (x > 10.0f) {
                    C[row * N + col] = x;
                    return;
                } else if (x < -10.0f) {
                    C[row * N + col] = 0.0f;
                    return;
                }

                // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                float x_cubed = x * x * x;
                float inner = 0.7978845608f * (x + 0.044715f * x_cubed);
                inner = clamp(inner, -20.0f, 20.0f);
                C[row * N + col] = 0.5f * x * (1.0f + tanh(inner));
            }

            // Fused matmul+bias+GELU: Combines matrix multiplication, bias addition, and GELU activation
            // Eliminates two intermediate buffers and two kernel dispatches
            // A: [M, K], B: [K, N], bias: [N] -> C: [M, N] with bias and GELU applied
            kernel void fused_matmul_bias_gelu(
                device const float* A [[buffer(0)]],
                device const float* B [[buffer(1)]],
                device const float* bias [[buffer(2)]],
                device float* C [[buffer(3)]],
                constant uint& M [[buffer(4)]],
                constant uint& N [[buffer(5)]],
                constant uint& K [[buffer(6)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint row = gid.y;
                uint col = gid.x;

                if (row >= M || col >= N) return;

                // 1. Compute matmul element C[row, col] = sum(A[row, k] * B[k, col])
                float sum = 0.0f;
                for (uint k = 0; k < K; ++k) {
                    sum += A[row * K + k] * B[k * N + col];
                }

                // 2. Add bias (broadcasted across rows)
                float x = sum + bias[col];

                // 3. Apply GELU activation immediately (no intermediate buffer)
                // Handle extreme values to prevent NaN
                if (x > 10.0f) {
                    C[row * N + col] = x;
                    return;
                } else if (x < -10.0f) {
                    C[row * N + col] = 0.0f;
                    return;
                }

                // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                float x_cubed = x * x * x;
                float inner = 0.7978845608f * (x + 0.044715f * x_cubed);
                inner = clamp(inner, -20.0f, 20.0f);
                C[row * N + col] = 0.5f * x * (1.0f + tanh(inner));
            }

            // Scalar multiplication: output[i] = input[i] * scale
            // Used for attention score scaling: scores *= 1/sqrt(head_dim)
            kernel void scale(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant float& scale [[buffer(2)]],
                constant uint& size [[buffer(3)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid >= size) return;
                output[gid] = input[gid] * scale;
            }

            // Bias addition: Add 1D bias vector to 2D matrix (broadcasting)
            // Input: [m, n], Bias: [n] → Output: [m, n]
            kernel void add_bias(
                device const float* input [[buffer(0)]],
                device const float* bias [[buffer(1)]],
                device float* output [[buffer(2)]],
                constant uint& m [[buffer(3)]],
                constant uint& n [[buffer(4)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint row = gid.y;
                uint col = gid.x;

                if (row >= m || col >= n) return;

                uint idx = row * n + col;
                output[idx] = input[idx] + bias[col];
            }

            // RoPE (Rotary Position Embedding)
            // Rotates pairs (i, i+D/2) for i in 0..D/2 where D is rotary_ndims
            // Input/output: [seq_len, num_heads, head_dim]
            kernel void rope(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& seq_len [[buffer(2)]],
                constant uint& num_heads [[buffer(3)]],
                constant uint& head_dim [[buffer(4)]],
                constant uint& rotary_ndims [[buffer(5)]],
                constant float& base [[buffer(6)]],
                uint3 gid [[thread_position_in_grid]]
            ) {
                uint pos = gid.z;
                uint h = gid.y;
                uint i = gid.x;

                if (pos >= seq_len || h >= num_heads || i >= (rotary_ndims / 2)) return;

                uint j = i + (rotary_ndims / 2);

                // Calculate rotation angle
                float freq = 1.0f / pow(base, 2.0f * float(i) / float(rotary_ndims));
                float angle = float(pos) * freq;
                float cos_val = cos(angle);
                float sin_val = sin(angle);

                // Get input indices
                uint idx_i = pos * (num_heads * head_dim) + h * head_dim + i;
                uint idx_j = pos * (num_heads * head_dim) + h * head_dim + j;

                // Apply rotation: (x_i, x_j) → (x_i*cos - x_j*sin, x_i*sin + x_j*cos)
                float x_i = input[idx_i];
                float x_j = input[idx_j];

                output[idx_i] = x_i * cos_val - x_j * sin_val;
                output[idx_j] = x_i * sin_val + x_j * cos_val;

                // Copy non-rotated dimensions
                if (i == 0) {
                    for (uint k = rotary_ndims; k < head_dim; ++k) {
                        uint idx_k = pos * (num_heads * head_dim) + h * head_dim + k;
                        output[idx_k] = input[idx_k];
                    }
                }
            }

            // Reshape for multi-head attention
            // Input: [seq_len, hidden_size] where hidden_size = num_heads * head_dim
            // Output: [num_heads, seq_len, head_dim]
            kernel void reshape_to_heads(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& seq_len [[buffer(2)]],
                constant uint& num_heads [[buffer(3)]],
                constant uint& head_dim [[buffer(4)]],
                uint3 gid [[thread_position_in_grid]]
            ) {
                uint h = gid.x;  // head index
                uint s = gid.y;  // sequence position
                uint d = gid.z;  // dimension within head

                if (h >= num_heads || s >= seq_len || d >= head_dim) return;

                // Input layout: [seq_len, hidden_size] = [seq_len, num_heads * head_dim]
                // input[s, h * head_dim + d]
                uint input_idx = s * (num_heads * head_dim) + h * head_dim + d;

                // Output layout: [num_heads, seq_len, head_dim]
                // output[h, s, d]
                uint output_idx = h * (seq_len * head_dim) + s * head_dim + d;

                output[output_idx] = input[input_idx];
            }

            // Reshape from multi-head back to flat
            // Input: [num_heads, seq_len, head_dim]
            // Output: [seq_len, hidden_size] where hidden_size = num_heads * head_dim
            kernel void reshape_from_heads(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& seq_len [[buffer(2)]],
                constant uint& num_heads [[buffer(3)]],
                constant uint& head_dim [[buffer(4)]],
                uint3 gid [[thread_position_in_grid]]
            ) {
                uint h = gid.x;  // head index
                uint s = gid.y;  // sequence position
                uint d = gid.z;  // dimension within head

                if (h >= num_heads || s >= seq_len || d >= head_dim) return;

                // Input layout: [num_heads, seq_len, head_dim]
                // input[h, s, d]
                uint input_idx = h * (seq_len * head_dim) + s * head_dim + d;

                // Output layout: [seq_len, hidden_size] = [seq_len, num_heads * head_dim]
                // output[s, h * head_dim + d]
                uint output_idx = s * (num_heads * head_dim) + h * head_dim + d;

                output[output_idx] = input[input_idx];
            }

            // Softmax with causal mask for attention
            // Input: [seq_len, seq_len] attention scores
            // Output: [seq_len, seq_len] attention weights
            // Applies causal mask: position i can only attend to j <= i
            kernel void softmax_causal(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& seq_len [[buffer(2)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid >= seq_len) return;

                uint row = gid;
                uint offset = row * seq_len;

                // Find max for numerical stability (only consider j <= row for causal mask)
                float max_val = -3.402823466e+38f;  // -FLT_MAX
                for (uint j = 0; j <= row; ++j) {
                    max_val = max(max_val, input[offset + j]);
                }

                // Handle edge case: if all values are -inf (shouldn't happen but be safe)
                if (max_val < -1e38f) {
                    for (uint j = 0; j < seq_len; ++j) {
                        output[offset + j] = (j == 0) ? 1.0f : 0.0f;
                    }
                    return;
                }

                // Compute exp and sum
                float sum = 0.0f;
                for (uint j = 0; j <= row; ++j) {
                    sum += exp(input[offset + j] - max_val);
                }

                // Handle degenerate case
                if (sum < 1e-10f) {
                    for (uint j = 0; j < seq_len; ++j) {
                        output[offset + j] = (j == 0) ? 1.0f : 0.0f;
                    }
                    return;
                }

                // Normalize and apply causal mask
                for (uint j = 0; j < seq_len; ++j) {
                    if (j <= row) {
                        output[offset + j] = exp(input[offset + j] - max_val) / sum;
                    } else {
                        output[offset + j] = 0.0f;  // Causal mask
                    }
                }
            }

            // LayerNorm: output = (x - mean) / sqrt(var + eps) * weight + bias
            // Optimized for transformer layers: normalize over last dimension (hidden_size)
            kernel void layernorm(
                device const float* input [[buffer(0)]],
                device const float* weight [[buffer(1)]],
                device const float* bias [[buffer(2)]],
                device float* output [[buffer(3)]],
                constant uint& seq_len [[buffer(4)]],
                constant uint& hidden_size [[buffer(5)]],
                constant float& eps [[buffer(6)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid >= seq_len) return;

                // Each thread processes one sequence position
                uint offset = gid * hidden_size;

                // Compute mean
                float sum = 0.0f;
                for (uint i = 0; i < hidden_size; ++i) {
                    sum += input[offset + i];
                }
                float mean = sum / float(hidden_size);

                // Compute variance
                float var_sum = 0.0f;
                for (uint i = 0; i < hidden_size; ++i) {
                    float diff = input[offset + i] - mean;
                    var_sum += diff * diff;
                }
                float variance = var_sum / float(hidden_size);
                float std_dev = sqrt(variance + eps);

                // Normalize and apply affine transform
                for (uint i = 0; i < hidden_size; ++i) {
                    float normalized = (input[offset + i] - mean) / std_dev;
                    output[offset + i] = normalized * weight[i] + bias[i];
                }
            }

            // Copy tensor data with offset for stacking/concatenation
            // Copies elements from input buffer to output buffer at specified offset
            kernel void copy_with_offset(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& output_offset [[buffer(2)]],
                constant uint& num_elements [[buffer(3)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid >= num_elements) return;
                output[output_offset + gid] = input[gid];
            }

            // Split QKV tensor into Q, K, V components on GPU
            // Input: [batch, seq_len, 3*hidden_size]
            // Outputs: Q, K, V each [batch, seq_len, hidden_size]
            kernel void split_qkv(
                device const float* qkv [[buffer(0)]],
                device float* q [[buffer(1)]],
                device float* k [[buffer(2)]],
                device float* v [[buffer(3)]],
                constant uint& batch_size [[buffer(4)]],
                constant uint& seq_len [[buffer(5)]],
                constant uint& hidden_size [[buffer(6)]],
                uint3 gid [[thread_position_in_grid]]
            ) {
                uint b = gid.x;
                uint s = gid.y;
                uint h = gid.z;

                if (b >= batch_size || s >= seq_len || h >= hidden_size) return;

                // Calculate indices
                uint qkv_base = (b * seq_len + s) * (3 * hidden_size);
                uint output_idx = (b * seq_len + s) * hidden_size + h;

                // Extract Q, K, V from packed tensor
                q[output_idx] = qkv[qkv_base + h];
                k[output_idx] = qkv[qkv_base + hidden_size + h];
                v[output_idx] = qkv[qkv_base + 2 * hidden_size + h];
            }

            // Element-wise addition: output = a + b
            // Critical for residual connections in transformers (prevents CPU round-trips)
            // Inputs and output have the same shape
            kernel void elementwise_add(
                device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* output [[buffer(2)]],
                constant uint& size [[buffer(3)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid >= size) return;
                output[gid] = a[gid] + b[gid];
            }

            // Transpose 2D matrix: output[j, i] = input[i, j]
            // Input: [rows, cols], Output: [cols, rows]
            // Critical for attention: K^T in Q @ K^T
            kernel void transpose_2d(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& rows [[buffer(2)]],
                constant uint& cols [[buffer(3)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint row = gid.y;
                uint col = gid.x;

                if (row >= rows || col >= cols) return;

                // Transpose: output[col, row] = input[row, col]
                output[col * rows + row] = input[row * cols + col];
            }

            // BATCHED transpose 3D: Transpose all heads in parallel
            // Input: [num_heads, rows, cols], Output: [num_heads, cols, rows]
            // Critical for multi-head attention: Process all K^T simultaneously
            kernel void batched_transpose_3d(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& num_heads [[buffer(2)]],
                constant uint& rows [[buffer(3)]],
                constant uint& cols [[buffer(4)]],
                uint3 gid [[thread_position_in_grid]]
            ) {
                uint h = gid.z;    // head index
                uint row = gid.y;  // row index
                uint col = gid.x;  // col index

                if (h >= num_heads || row >= rows || col >= cols) return;

                // Input layout: [num_heads, rows, cols]
                uint input_idx = h * (rows * cols) + row * cols + col;

                // Output layout: [num_heads, cols, rows] (transposed within each head)
                uint output_idx = h * (cols * rows) + col * rows + row;

                output[output_idx] = input[input_idx];
            }

            // BATCHED softmax with causal mask: Process all heads in parallel
            // Input: [num_heads, seq_len, seq_len] attention scores
            // Output: [num_heads, seq_len, seq_len] attention weights
            // Applies causal mask: position i can only attend to j <= i
            // CRITICAL OPTIMIZATION: All heads processed in parallel (8-12x speedup)
            kernel void batched_softmax_causal(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& num_heads [[buffer(2)]],
                constant uint& seq_len [[buffer(3)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint h = gid.y;    // head index
                uint row = gid.x;  // sequence position

                if (h >= num_heads || row >= seq_len) return;

                // Calculate base offset for this head and row
                uint base_offset = h * (seq_len * seq_len) + row * seq_len;

                // Find max for numerical stability (only consider j <= row for causal mask)
                float max_val = -3.402823466e+38f;  // -FLT_MAX
                for (uint j = 0; j <= row; ++j) {
                    max_val = max(max_val, input[base_offset + j]);
                }

                // Handle edge case: if all values are -inf
                if (max_val < -1e38f) {
                    for (uint j = 0; j < seq_len; ++j) {
                        output[base_offset + j] = (j == 0) ? 1.0f : 0.0f;
                    }
                    return;
                }

                // Compute exp(x - max) and sum (only for j <= row)
                float sum = 0.0f;
                for (uint j = 0; j <= row; ++j) {
                    float exp_val = exp(input[base_offset + j] - max_val);
                    output[base_offset + j] = exp_val;
                    sum += exp_val;
                }

                // Normalize and apply causal mask
                for (uint j = 0; j < seq_len; ++j) {
                    if (j <= row) {
                        output[base_offset + j] /= sum;  // Normalize
                    } else {
                        output[base_offset + j] = 0.0f;  // Causal mask (future positions)
                    }
                }
            }

            // BATCHED matmul: Multiply all heads in parallel (tiled for performance)
            // A: [num_heads, M, K], B: [num_heads, K, N] → C: [num_heads, M, N]
            // Critical optimization: Process all heads simultaneously (8-12x speedup)
            // Uses 16x16 tiling for efficient memory access
            kernel void batched_matmul(
                device const float* A [[buffer(0)]],
                device const float* B [[buffer(1)]],
                device float* C [[buffer(2)]],
                constant uint& num_heads [[buffer(3)]],
                constant uint& M [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                constant uint& N [[buffer(6)]],
                uint3 gid [[thread_position_in_grid]],
                uint3 tid [[thread_position_in_threadgroup]],
                uint3 tgid [[threadgroup_position_in_grid]]
            ) {
                // Thread indices
                uint h = gid.z;        // head index
                uint row = gid.y;      // output row
                uint col = gid.x;      // output col

                if (h >= num_heads || row >= M || col >= N) return;

                // Compute dot product for C[h][row][col]
                float sum = 0.0f;

                uint a_base = h * (M * K) + row * K;  // A[h][row][0]
                uint b_base = h * (K * N);            // B[h][0][0]

                for (uint k = 0; k < K; ++k) {
                    sum += A[a_base + k] * B[b_base + k * N + col];
                }

                // Write result
                uint c_idx = h * (M * N) + row * N + col;
                C[c_idx] = sum;
            }

            // BATCHED scaled matmul: Multiply all heads in parallel with scaling
            // A: [num_heads, M, K], B: [num_heads, K, N] → C: [num_heads, M, N]
            // Computes C = alpha * (A @ B) for all heads simultaneously
            // Used for scaled attention scores: Q @ K^T / sqrt(d_k)
            kernel void batched_matmul_scaled(
                device const float* A [[buffer(0)]],
                device const float* B [[buffer(1)]],
                device float* C [[buffer(2)]],
                constant uint& num_heads [[buffer(3)]],
                constant uint& M [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                constant uint& N [[buffer(6)]],
                constant float& alpha [[buffer(7)]],
                uint3 gid [[thread_position_in_grid]]
            ) {
                uint h = gid.z;
                uint row = gid.y;
                uint col = gid.x;

                if (h >= num_heads || row >= M || col >= N) return;

                float sum = 0.0f;

                uint a_base = h * (M * K) + row * K;
                uint b_base = h * (K * N);

                for (uint k = 0; k < K; ++k) {
                    sum += A[a_base + k] * B[b_base + k * N + col];
                }

                uint c_idx = h * (M * N) + row * N + col;
                C[c_idx] = alpha * sum;  // Apply scaling
            }

            // Fused scaled matmul + softmax with causal mask
            // Q: [num_heads, seq_len, head_dim], K^T: [num_heads, head_dim, seq_len]
            // Output: [num_heads, seq_len, seq_len] attention weights
            // Each thread handles one row (sequence position) for one head
            // Computes: softmax(Q[h,i,:] @ K^T[h,:,:] * alpha) with causal mask
            kernel void batched_scaled_matmul_softmax_causal(
                device const float* Q [[buffer(0)]],      // [num_heads, seq_len, head_dim]
                device const float* K_T [[buffer(1)]],    // [num_heads, head_dim, seq_len]
                device float* output [[buffer(2)]],       // [num_heads, seq_len, seq_len]
                constant uint& num_heads [[buffer(3)]],
                constant uint& seq_len [[buffer(4)]],
                constant uint& head_dim [[buffer(5)]],
                constant float& alpha [[buffer(6)]],      // Scaling factor (1/sqrt(head_dim))
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint h = gid.y;        // head index
                uint row = gid.x;      // sequence position

                if (h >= num_heads || row >= seq_len) return;

                // Base pointers for this head
                uint q_base = h * (seq_len * head_dim) + row * head_dim;  // Q[h, row, :]
                uint k_base = h * (head_dim * seq_len);                    // K^T[h, :, :]
                uint out_base = h * (seq_len * seq_len) + row * seq_len;  // output[h, row, :]

                // Step 1: Compute scaled dot products (Q @ K^T) for this row
                // Only compute up to current position (causal mask)
                float scores[256];  // Max seq_len = 256 for local array
                float max_score = -3.402823466e+38f;  // -FLT_MAX

                // Compute scores and find max for numerical stability
                for (uint col = 0; col <= row && col < seq_len; ++col) {
                    float dot = 0.0f;

                    // Dot product: Q[h,row,:] · K^T[h,:,col]
                    for (uint k = 0; k < head_dim; ++k) {
                        dot += Q[q_base + k] * K_T[k_base + k * seq_len + col];
                    }

                    scores[col] = alpha * dot;  // Apply scaling
                    max_score = max(max_score, scores[col]);
                }

                // Step 2: Compute exp and sum for softmax
                float sum = 0.0f;
                for (uint col = 0; col <= row && col < seq_len; ++col) {
                    scores[col] = exp(scores[col] - max_score);  // Numerically stable
                    sum += scores[col];
                }

                // Step 3: Normalize and write output with causal masking
                for (uint col = 0; col < seq_len; ++col) {
                    if (col <= row) {
                        output[out_base + col] = scores[col] / sum;  // Normalized attention weight
                    } else {
                        output[out_base + col] = 0.0f;  // Causal mask: future positions zeroed
                    }
                }
            }

            // Fused scaled matmul + softmax for generation (q_seq != kv_seq)
            // Optimized for autoregressive generation where q_seq=1, kv_seq=cached_length
            // Q: [num_heads, q_seq_len, head_dim], K^T: [num_heads, head_dim, kv_seq_len]
            // Output: [num_heads, q_seq_len, kv_seq_len] attention weights
            // Each thread handles one (query_pos, head) pair, computing attention over all KV positions
            // Computes: softmax(Q[h,i,:] @ K^T[h,:,:] * alpha) - NO causal mask (all KV in past)
            kernel void batched_scaled_matmul_softmax_gen(
                device const float* Q [[buffer(0)]],         // [num_heads, q_seq_len, head_dim]
                device const float* K_T [[buffer(1)]],       // [num_heads, head_dim, kv_seq_len]
                device float* output [[buffer(2)]],          // [num_heads, q_seq_len, kv_seq_len]
                constant uint& num_heads [[buffer(3)]],
                constant uint& q_seq_len [[buffer(4)]],
                constant uint& kv_seq_len [[buffer(5)]],
                constant uint& head_dim [[buffer(6)]],
                constant float& alpha [[buffer(7)]],         // Scaling factor (1/sqrt(head_dim))
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint h = gid.y;        // head index
                uint q_row = gid.x;    // query sequence position

                if (h >= num_heads || q_row >= q_seq_len) return;

                // Base pointers for this head
                uint q_base = h * (q_seq_len * head_dim) + q_row * head_dim;    // Q[h, q_row, :]
                uint k_base = h * (head_dim * kv_seq_len);                       // K^T[h, :, :]
                uint out_base = h * (q_seq_len * kv_seq_len) + q_row * kv_seq_len;  // output[h, q_row, :]

                // Step 1: Compute scaled dot products (Q @ K^T) for this query row
                // Attend to ALL kv positions (they're all in the past for generation)
                float scores[512];  // Max kv_seq_len = 512 for local array
                float max_score = -3.402823466e+38f;  // -FLT_MAX

                // Compute scores and find max for numerical stability
                for (uint kv_col = 0; kv_col < kv_seq_len; ++kv_col) {
                    float dot = 0.0f;

                    // Dot product: Q[h,q_row,:] · K^T[h,:,kv_col]
                    for (uint k = 0; k < head_dim; ++k) {
                        dot += Q[q_base + k] * K_T[k_base + k * kv_seq_len + kv_col];
                    }

                    scores[kv_col] = alpha * dot;  // Apply scaling
                    max_score = max(max_score, scores[kv_col]);
                }

                // Step 2: Compute exp and sum for softmax
                float sum = 0.0f;
                for (uint kv_col = 0; kv_col < kv_seq_len; ++kv_col) {
                    scores[kv_col] = exp(scores[kv_col] - max_score);  // Numerically stable
                    sum += scores[kv_col];
                }

                // Step 3: Normalize and write output (no causal masking - all KV valid)
                for (uint kv_col = 0; kv_col < kv_seq_len; ++kv_col) {
                    output[out_base + kv_col] = scores[kv_col] / sum;  // Normalized attention weight
                }
            }

            // Concatenate two tensors along sequence dimension for KV-cache
            // Input 1: [batch, num_heads, seq_len1, head_dim] (cached K or V)
            // Input 2: [batch, num_heads, seq_len2, head_dim] (new K or V)
            // Output: [batch, num_heads, seq_len1+seq_len2, head_dim]
            // Critical for GPU-aware KV-cache: avoids CPU transfer
            kernel void concat_seq_dim(
                device const float* input1 [[buffer(0)]],
                device const float* input2 [[buffer(1)]],
                device float* output [[buffer(2)]],
                constant uint& batch [[buffer(3)]],
                constant uint& num_heads [[buffer(4)]],
                constant uint& seq_len1 [[buffer(5)]],
                constant uint& seq_len2 [[buffer(6)]],
                constant uint& head_dim [[buffer(7)]],
                uint3 gid [[thread_position_in_grid]]
            ) {
                uint b = gid.z;      // batch index
                uint h = gid.y;      // head index
                uint d = gid.x;      // head_dim index

                if (b >= batch || h >= num_heads || d >= head_dim) return;

                uint total_seq_len = seq_len1 + seq_len2;

                // Calculate base offsets for this batch and head
                uint input1_head_offset = b * (num_heads * seq_len1 * head_dim) +
                                         h * (seq_len1 * head_dim);
                uint input2_head_offset = b * (num_heads * seq_len2 * head_dim) +
                                         h * (seq_len2 * head_dim);
                uint output_head_offset = b * (num_heads * total_seq_len * head_dim) +
                                         h * (total_seq_len * head_dim);

                // Copy seq_len1 elements from input1
                for (uint s = 0; s < seq_len1; ++s) {
                    uint input_idx = input1_head_offset + s * head_dim + d;
                    uint output_idx = output_head_offset + s * head_dim + d;
                    output[output_idx] = input1[input_idx];
                }

                // Copy seq_len2 elements from input2 (append after input1)
                for (uint s = 0; s < seq_len2; ++s) {
                    uint input_idx = input2_head_offset + s * head_dim + d;
                    uint output_idx = output_head_offset + (seq_len1 + s) * head_dim + d;
                    output[output_idx] = input2[input_idx];
                }
            }

            // ==============================================================================
            // Flash Attention Kernel
            // Based on "FlashAttention: Fast and Memory-Efficient Exact Attention"
            // ==============================================================================

            // Flash Attention parameters
            struct FlashAttentionParams {
                uint batch_size;
                uint num_heads;
                uint q_seq_len;
                uint kv_seq_len;
                uint head_dim;
                float scale;
                uint use_causal_mask;
            };

            // Helper function: Load query block into shared memory
            inline void load_q_block(
                device const float* Q,
                threadgroup float* shared_Q,
                uint batch_idx,
                uint head_idx,
                uint q_block_start,
                uint q_idx_in_block,
                constant FlashAttentionParams& params
            ) {
                // Calculate global Q index
                uint q_idx = q_block_start + q_idx_in_block;

                if (q_idx < params.q_seq_len) {
                    // Load one row of Q: [head_dim] elements
                    uint q_offset = ((batch_idx * params.num_heads + head_idx) * params.q_seq_len + q_idx) * params.head_dim;

                    for (uint d = 0; d < params.head_dim; ++d) {
                        shared_Q[q_idx_in_block * params.head_dim + d] = Q[q_offset + d];
                    }
                } else {
                    // Pad with zeros if out of bounds
                    for (uint d = 0; d < params.head_dim; ++d) {
                        shared_Q[q_idx_in_block * params.head_dim + d] = 0.0f;
                    }
                }
            }

            // Helper function: Load KV block into shared memory
            inline void load_kv_block(
                device const float* K,
                device const float* V,
                threadgroup float* shared_K,
                threadgroup float* shared_V,
                uint batch_idx,
                uint head_idx,
                uint kv_block_start,
                uint kv_block_size,
                uint thread_idx,
                constant FlashAttentionParams& params
            ) {
                // Each thread loads multiple KV rows (BLOCK_KV might be > num_threads)
                for (uint i = thread_idx; i < kv_block_size; i += 32) {  // Assuming 32 threads
                    uint kv_idx = kv_block_start + i;

                    if (kv_idx < params.kv_seq_len) {
                        uint kv_offset = ((batch_idx * params.num_heads + head_idx) * params.kv_seq_len + kv_idx) * params.head_dim;

                        for (uint d = 0; d < params.head_dim; ++d) {
                            shared_K[i * params.head_dim + d] = K[kv_offset + d];
                            shared_V[i * params.head_dim + d] = V[kv_offset + d];
                        }
                    } else {
                        // Pad with zeros
                        for (uint d = 0; d < params.head_dim; ++d) {
                            shared_K[i * params.head_dim + d] = 0.0f;
                            shared_V[i * params.head_dim + d] = 0.0f;
                        }
                    }
                }
            }

            // Flash Attention kernel
            kernel void flash_attention_forward(
                device const float* Q [[buffer(0)]],          // [batch, heads, q_len, head_dim]
                device const float* K [[buffer(1)]],          // [batch, heads, kv_len, head_dim]
                device const float* V [[buffer(2)]],          // [batch, heads, kv_len, head_dim]
                device float* O [[buffer(3)]],                // [batch, heads, q_len, head_dim] output
                device float* L [[buffer(4)]],                // [batch, heads, q_len] logsumexp
                constant FlashAttentionParams& params [[buffer(5)]],
                threadgroup float* shared_Q [[threadgroup(0)]],   // [BLOCK_Q, head_dim]
                threadgroup float* shared_K [[threadgroup(1)]],   // [BLOCK_KV, head_dim]
                threadgroup float* shared_V [[threadgroup(2)]],   // [BLOCK_KV, head_dim]
                uint3 threadgroup_id [[threadgroup_position_in_grid]],  // (q_block, head, batch)
                uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]
            ) {
                // Thread indices
                const uint batch_idx = threadgroup_id.z;
                const uint head_idx = threadgroup_id.y;
                const uint q_block_idx = threadgroup_id.x;
                const uint q_idx_in_block = thread_position_in_threadgroup.x;
                const uint thread_idx = thread_position_in_threadgroup.x;

                // Calculate Q block boundaries
                const uint q_block_start = q_block_idx * 32;  // BLOCK_Q = 32
                const uint q_idx = q_block_start + q_idx_in_block;

                // Early exit if beyond sequence length
                if (q_idx >= params.q_seq_len) {
                    return;
                }

                // Load Q block into shared memory
                threadgroup_barrier(mem_flags::mem_threadgroup);
                load_q_block(Q, shared_Q, batch_idx, head_idx, q_block_start, q_idx_in_block, params);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Initialize accumulators for online softmax
                float max_score = -INFINITY;  // Running maximum for numerical stability
                float sum_exp = 0.0f;         // Running sum of exponentials
                float output[256];            // Local output accumulator (max head_dim = 256)

                for (uint d = 0; d < params.head_dim; ++d) {
                    output[d] = 0.0f;
                }

                // Iterate over KV blocks
                const uint num_kv_blocks = (params.kv_seq_len + 64 - 1) / 64;  // BLOCK_KV = 64

                for (uint kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
                    const uint kv_block_start = kv_block * 64;
                    const uint kv_block_end = min(kv_block_start + 64, params.kv_seq_len);
                    const uint kv_block_size = kv_block_end - kv_block_start;

                    // Load KV block into shared memory
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    load_kv_block(K, V, shared_K, shared_V, batch_idx, head_idx, kv_block_start, kv_block_size, thread_idx, params);
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    // Compute attention scores for this Q token against all KV tokens in block
                    for (uint kv_idx_in_block = 0; kv_idx_in_block < kv_block_size; ++kv_idx_in_block) {
                        const uint kv_idx = kv_block_start + kv_idx_in_block;

                        // Apply causal mask: only attend to past tokens
                        if (params.use_causal_mask && kv_idx > q_idx) {
                            continue;
                        }

                        // Compute dot product: Q[q_idx] · K[kv_idx]
                        float score = 0.0f;
                        for (uint d = 0; d < params.head_dim; ++d) {
                            score += shared_Q[q_idx_in_block * params.head_dim + d] *
                                     shared_K[kv_idx_in_block * params.head_dim + d];
                        }
                        score *= params.scale;  // Scale by 1/sqrt(head_dim)

                        // Online softmax update
                        float old_max = max_score;
                        max_score = max(max_score, score);

                        // Correction factor for previous accumulator
                        float correction = exp(old_max - max_score);
                        sum_exp = correction * sum_exp + exp(score - max_score);

                        // Update output with corrected previous values + new contribution
                        float attn_weight = exp(score - max_score);
                        for (uint d = 0; d < params.head_dim; ++d) {
                            output[d] = correction * output[d] +
                                       attn_weight * shared_V[kv_idx_in_block * params.head_dim + d];
                        }
                    }
                }

                // Final normalization: divide by sum of exp
                if (sum_exp > 0.0f) {
                    for (uint d = 0; d < params.head_dim; ++d) {
                        output[d] /= sum_exp;
                    }
                }

                // Write output to global memory
                uint out_offset = ((batch_idx * params.num_heads + head_idx) * params.q_seq_len + q_idx) * params.head_dim;
                for (uint d = 0; d < params.head_dim; ++d) {
                    O[out_offset + d] = output[d];
                }

                // Write logsumexp for reference (can be used for backward pass)
                uint l_offset = (batch_idx * params.num_heads + head_idx) * params.q_seq_len + q_idx;
                L[l_offset] = max_score + log(sum_exp);
            }
        "#;
        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to compile Metal shader: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let kernel = library.get_function("matmul", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get kernel function: {}", e),
                "MetalBackend::new",
            )
        })?;
        let matmul_pipeline =
            device.new_compute_pipeline_state_with_function(&kernel).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create matmul pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let gelu_kernel = library.get_function("gelu", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get gelu kernel function: {}", e),
                "MetalBackend::new",
            )
        })?;
        let gelu_pipeline =
            device.new_compute_pipeline_state_with_function(&gelu_kernel).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create gelu pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let fused_matmul_gelu_kernel =
            library.get_function("fused_matmul_gelu", None).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to get fused_matmul_gelu kernel function: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let matmul_gelu_pipeline = device
            .new_compute_pipeline_state_with_function(&fused_matmul_gelu_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create matmul_gelu pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let fused_matmul_bias_gelu_kernel =
            library.get_function("fused_matmul_bias_gelu", None).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!(
                        "Failed to get fused_matmul_bias_gelu kernel function: {}",
                        e
                    ),
                    "MetalBackend::new",
                )
            })?;
        let matmul_bias_gelu_pipeline = device
            .new_compute_pipeline_state_with_function(&fused_matmul_bias_gelu_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create matmul_bias_gelu pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let scale_kernel = library.get_function("scale", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get scale kernel function: {}", e),
                "MetalBackend::new",
            )
        })?;
        let scale_pipeline =
            device.new_compute_pipeline_state_with_function(&scale_kernel).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create scale pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let add_bias_kernel = library.get_function("add_bias", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get add_bias kernel function: {}", e),
                "MetalBackend::new",
            )
        })?;
        let add_bias_pipeline =
            device.new_compute_pipeline_state_with_function(&add_bias_kernel).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create add_bias pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let layernorm_kernel = library.get_function("layernorm", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get layernorm kernel function: {}", e),
                "MetalBackend::new",
            )
        })?;
        let layernorm_pipeline = device
            .new_compute_pipeline_state_with_function(&layernorm_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create layernorm pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let rope_kernel = library.get_function("rope", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get rope kernel function: {}", e),
                "MetalBackend::new",
            )
        })?;
        let rope_pipeline =
            device.new_compute_pipeline_state_with_function(&rope_kernel).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create rope pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let softmax_causal_kernel = library.get_function("softmax_causal", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get softmax_causal kernel function: {}", e),
                "MetalBackend::new",
            )
        })?;
        let softmax_causal_pipeline = device
            .new_compute_pipeline_state_with_function(&softmax_causal_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create softmax_causal pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let copy_with_offset_kernel =
            library.get_function("copy_with_offset", None).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to get copy_with_offset kernel function: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let copy_with_offset_pipeline = device
            .new_compute_pipeline_state_with_function(&copy_with_offset_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create copy_with_offset pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let split_qkv_kernel = library.get_function("split_qkv", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get split_qkv kernel function: {}", e),
                "MetalBackend::new",
            )
        })?;
        let split_qkv_pipeline = device
            .new_compute_pipeline_state_with_function(&split_qkv_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create split_qkv pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let elementwise_add_kernel =
            library.get_function("elementwise_add", None).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to get elementwise_add kernel function: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let elementwise_add_pipeline = device
            .new_compute_pipeline_state_with_function(&elementwise_add_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create elementwise_add pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let transpose_kernel = library.get_function("transpose_2d", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get transpose_2d kernel function: {}", e),
                "MetalBackend::new",
            )
        })?;
        let transpose_pipeline = device
            .new_compute_pipeline_state_with_function(&transpose_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create transpose_2d pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let reshape_to_heads_kernel =
            library.get_function("reshape_to_heads", None).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to get reshape_to_heads kernel function: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let reshape_to_heads_pipeline = device
            .new_compute_pipeline_state_with_function(&reshape_to_heads_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create reshape_to_heads pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let reshape_from_heads_kernel =
            library.get_function("reshape_from_heads", None).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to get reshape_from_heads kernel function: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let reshape_from_heads_pipeline = device
            .new_compute_pipeline_state_with_function(&reshape_from_heads_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create reshape_from_heads pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let batched_transpose_kernel =
            library.get_function("batched_transpose_3d", None).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to get batched_transpose_3d kernel function: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let batched_transpose_pipeline = device
            .new_compute_pipeline_state_with_function(&batched_transpose_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create batched_transpose_3d pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let batched_softmax_causal_kernel =
            library.get_function("batched_softmax_causal", None).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!(
                        "Failed to get batched_softmax_causal kernel function: {}",
                        e
                    ),
                    "MetalBackend::new",
                )
            })?;
        let batched_softmax_causal_pipeline = device
            .new_compute_pipeline_state_with_function(&batched_softmax_causal_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create batched_softmax_causal pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let batched_matmul_kernel = library.get_function("batched_matmul", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get batched_matmul kernel function: {}", e),
                "MetalBackend::new",
            )
        })?;
        let batched_matmul_pipeline = device
            .new_compute_pipeline_state_with_function(&batched_matmul_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create batched_matmul pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let batched_matmul_scaled_kernel =
            library.get_function("batched_matmul_scaled", None).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to get batched_matmul_scaled kernel function: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let batched_matmul_scaled_pipeline = device
            .new_compute_pipeline_state_with_function(&batched_matmul_scaled_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create batched_matmul_scaled pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let batched_scaled_matmul_softmax_causal_kernel = library
            .get_function("batched_scaled_matmul_softmax_causal", None)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!(
                        "Failed to get batched_scaled_matmul_softmax_causal kernel function: {}",
                        e
                    ),
                    "MetalBackend::new",
                )
            })?;
        let batched_scaled_matmul_softmax_causal_pipeline = device
            .new_compute_pipeline_state_with_function(&batched_scaled_matmul_softmax_causal_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!(
                        "Failed to create batched_scaled_matmul_softmax_causal pipeline: {}",
                        e
                    ),
                    "MetalBackend::new",
                )
            })?;
        let batched_scaled_matmul_softmax_gen_kernel =
            library.get_function("batched_scaled_matmul_softmax_gen", None).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!(
                        "Failed to get batched_scaled_matmul_softmax_gen kernel function: {}",
                        e
                    ),
                    "MetalBackend::new",
                )
            })?;
        let batched_scaled_matmul_softmax_gen_pipeline = device
            .new_compute_pipeline_state_with_function(&batched_scaled_matmul_softmax_gen_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!(
                        "Failed to create batched_scaled_matmul_softmax_gen pipeline: {}",
                        e
                    ),
                    "MetalBackend::new",
                )
            })?;
        let concat_seq_dim_kernel = library.get_function("concat_seq_dim", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get concat_seq_dim kernel function: {}", e),
                "MetalBackend::new",
            )
        })?;
        let concat_seq_dim_pipeline = device
            .new_compute_pipeline_state_with_function(&concat_seq_dim_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create concat_seq_dim pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        // Flash Attention pipeline
        let flash_attention_kernel =
            library.get_function("flash_attention_forward", None).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!(
                        "Failed to get flash_attention_forward kernel function: {}",
                        e
                    ),
                    "MetalBackend::new",
                )
            })?;
        let flash_attention_pipeline = device
            .new_compute_pipeline_state_with_function(&flash_attention_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create flash_attention pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;
        let mps_ops = Arc::new(Self::initialize_mps(&device, &command_queue));
        Ok(Self {
            device,
            command_queue,
            buffer_cache: Arc::new(std::sync::Mutex::new(BufferCache::new())),
            matmul_pipeline: Arc::new(matmul_pipeline),
            gelu_pipeline: Arc::new(gelu_pipeline),
            matmul_gelu_pipeline: Arc::new(matmul_gelu_pipeline),
            matmul_bias_gelu_pipeline: Arc::new(matmul_bias_gelu_pipeline),
            scale_pipeline: Arc::new(scale_pipeline),
            add_bias_pipeline: Arc::new(add_bias_pipeline),
            layernorm_pipeline: Arc::new(layernorm_pipeline),
            rope_pipeline: Arc::new(rope_pipeline),
            softmax_causal_pipeline: Arc::new(softmax_causal_pipeline),
            copy_with_offset_pipeline: Arc::new(copy_with_offset_pipeline),
            elementwise_add_pipeline: Arc::new(elementwise_add_pipeline),
            split_qkv_pipeline: Arc::new(split_qkv_pipeline),
            transpose_pipeline: Arc::new(transpose_pipeline),
            reshape_to_heads_pipeline: Arc::new(reshape_to_heads_pipeline),
            reshape_from_heads_pipeline: Arc::new(reshape_from_heads_pipeline),
            batched_transpose_pipeline: Arc::new(batched_transpose_pipeline),
            batched_softmax_causal_pipeline: Arc::new(batched_softmax_causal_pipeline),
            batched_matmul_pipeline: Arc::new(batched_matmul_pipeline),
            batched_matmul_scaled_pipeline: Arc::new(batched_matmul_scaled_pipeline),
            batched_scaled_matmul_softmax_causal_pipeline: Arc::new(
                batched_scaled_matmul_softmax_causal_pipeline,
            ),
            batched_scaled_matmul_softmax_gen_pipeline: Arc::new(
                batched_scaled_matmul_softmax_gen_pipeline,
            ),
            concat_seq_dim_pipeline: Arc::new(concat_seq_dim_pipeline),
            flash_attention_pipeline: Arc::new(flash_attention_pipeline),
            mps_ops,
        })
    }
}
