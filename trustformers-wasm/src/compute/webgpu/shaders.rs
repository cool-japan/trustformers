//! WGSL compute shaders for tensor operations

/// Matrix multiplication shader (basic version)
pub const MATMUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec3<u32>; // M, N, K

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let M = dims.x;
    let N = dims.y;
    let K = dims.z;

    if (row >= M || col >= N) {
        return;
    }

    var sum = 0.0;
    for (var k = 0u; k < K; k = k + 1u) {
        let a_idx = row * K + k;
        let b_idx = k * N + col;
        sum = sum + a[a_idx] * b[b_idx];
    }

    let result_idx = row * N + col;
    result[result_idx] = sum;
}
"#;

/// Optimized matrix multiplication shader with shared memory tiling
pub const MATMUL_SHARED_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec3<u32>; // M, N, K

const TILE_SIZE: u32 = 16u;
var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    let local_row = local_id.x;
    let local_col = local_id.y;

    let M = dims.x;
    let N = dims.y;
    let K = dims.z;

    var sum = 0.0;

    // Iterate over tiles
    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;
    for (var tile = 0u; tile < num_tiles; tile = tile + 1u) {
        // Load tiles into shared memory
        let a_row = workgroup_id.x * TILE_SIZE + local_row;
        let a_col = tile * TILE_SIZE + local_col;
        let b_row = tile * TILE_SIZE + local_row;
        let b_col = workgroup_id.y * TILE_SIZE + local_col;

        if (a_row < M && a_col < K) {
            tile_a[local_row][local_col] = a[a_row * K + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }

        if (b_row < K && b_col < N) {
            tile_b[local_row][local_col] = b[b_row * N + b_col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }

        // Synchronize workgroup
        workgroupBarrier();

        // Compute partial dot product using shared memory
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[local_row][k] * tile_b[k][local_col];
        }

        // Synchronize before loading next tile
        workgroupBarrier();
    }

    // Write result
    if (row < M && col < N) {
        result[row * N + col] = sum;
    }
}
"#;

/// Element-wise addition shader
pub const ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }
    result[idx] = a[idx] + b[idx];
}
"#;

/// Element-wise multiplication shader
pub const MUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }
    result[idx] = a[idx] * b[idx];
}
"#;

/// ReLU activation shader
pub const RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }
    output[idx] = max(0.0, input[idx]);
}
"#;

/// GELU activation shader
pub const GELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

fn gelu(x: f32) -> f32 {
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let c1 = 0.7978845608; // sqrt(2/π)
    let c2 = 0.044715;
    let inner = c1 * (x + c2 * x * x * x);
    let tanh_inner = tanh(inner);
    return 0.5 * x * (1.0 + tanh_inner);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }
    output[idx] = gelu(input[idx]);
}
"#;

/// Softmax shader (for last dimension)
pub const SOFTMAX_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec2<u32>; // batch_size, feature_size

@compute @workgroup_size(1, 256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let feature_idx = global_id.y;
    let batch_size = dims.x;
    let feature_size = dims.y;

    if (batch_idx >= batch_size) {
        return;
    }

    let base_idx = batch_idx * feature_size;

    // Find max for numerical stability
    var max_val = input[base_idx];
    for (var i = 1u; i < feature_size; i = i + 1u) {
        max_val = max(max_val, input[base_idx + i]);
    }

    // Compute exp and sum
    var sum = 0.0;
    for (var i = 0u; i < feature_size; i = i + 1u) {
        let exp_val = exp(input[base_idx + i] - max_val);
        output[base_idx + i] = exp_val;
        sum = sum + exp_val;
    }

    // Normalize
    for (var i = 0u; i < feature_size; i = i + 1u) {
        output[base_idx + i] = output[base_idx + i] / sum;
    }
}
"#;

/// Optimized softmax with shared memory for large feature dimensions
pub const SOFTMAX_SHARED_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec2<u32>; // batch_size, feature_size

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> shared_data: array<f32, WORKGROUP_SIZE>;
var<workgroup> max_val: f32;
var<workgroup> sum_val: f32;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch_idx = workgroup_id.x;
    let thread_id = local_id.x;
    let batch_size = dims.x;
    let feature_size = dims.y;

    if (batch_idx >= batch_size) {
        return;
    }

    let base_idx = batch_idx * feature_size;

    // Phase 1: Find maximum using shared memory reduction
    var local_max = -3.4028235e+38; // -FLT_MAX
    for (var i = thread_id; i < feature_size; i = i + WORKGROUP_SIZE) {
        local_max = max(local_max, input[base_idx + i]);
    }
    shared_data[thread_id] = local_max;
    workgroupBarrier();

    // Reduce to find global max
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (thread_id < stride) {
            shared_data[thread_id] = max(shared_data[thread_id], shared_data[thread_id + stride]);
        }
        workgroupBarrier();
    }

    if (thread_id == 0u) {
        max_val = shared_data[0];
    }
    workgroupBarrier();

    // Phase 2: Compute exp and sum using shared memory reduction
    var local_sum = 0.0;
    for (var i = thread_id; i < feature_size; i = i + WORKGROUP_SIZE) {
        let exp_val = exp(input[base_idx + i] - max_val);
        output[base_idx + i] = exp_val;
        local_sum = local_sum + exp_val;
    }
    shared_data[thread_id] = local_sum;
    workgroupBarrier();

    // Reduce to find global sum
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (thread_id < stride) {
            shared_data[thread_id] = shared_data[thread_id] + shared_data[thread_id + stride];
        }
        workgroupBarrier();
    }

    if (thread_id == 0u) {
        sum_val = shared_data[0];
    }
    workgroupBarrier();

    // Phase 3: Normalize
    for (var i = thread_id; i < feature_size; i = i + WORKGROUP_SIZE) {
        output[base_idx + i] = output[base_idx + i] / sum_val;
    }
}
"#;

/// Layer normalization shader
pub const LAYER_NORM_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> dims: vec2<u32>; // batch_size, feature_size

@compute @workgroup_size(1, 256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let batch_size = dims.x;
    let feature_size = dims.y;

    if (batch_idx >= batch_size) {
        return;
    }

    let base_idx = batch_idx * feature_size;

    // Compute mean
    var mean = 0.0;
    for (var i = 0u; i < feature_size; i = i + 1u) {
        mean = mean + input[base_idx + i];
    }
    mean = mean / f32(feature_size);

    // Compute variance
    var variance = 0.0;
    for (var i = 0u; i < feature_size; i = i + 1u) {
        let diff = input[base_idx + i] - mean;
        variance = variance + diff * diff;
    }
    variance = variance / f32(feature_size);

    // Normalize and apply affine transform
    let eps = 1e-5;
    let std_inv = 1.0 / sqrt(variance + eps);

    for (var i = 0u; i < feature_size; i = i + 1u) {
        let normalized = (input[base_idx + i] - mean) * std_inv;
        output[base_idx + i] = normalized * gamma[i] + beta[i];
    }
}
"#;

/// Optimized layer normalization with shared memory reductions
pub const LAYER_NORM_SHARED_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> dims: vec2<u32>; // batch_size, feature_size

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> shared_data: array<f32, WORKGROUP_SIZE>;
var<workgroup> mean_val: f32;
var<workgroup> variance_val: f32;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch_idx = workgroup_id.x;
    let thread_id = local_id.x;
    let batch_size = dims.x;
    let feature_size = dims.y;

    if (batch_idx >= batch_size) {
        return;
    }

    let base_idx = batch_idx * feature_size;

    // Phase 1: Compute mean using shared memory reduction
    var local_sum = 0.0;
    for (var i = thread_id; i < feature_size; i = i + WORKGROUP_SIZE) {
        local_sum = local_sum + input[base_idx + i];
    }
    shared_data[thread_id] = local_sum;
    workgroupBarrier();

    // Reduce to find total sum
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (thread_id < stride) {
            shared_data[thread_id] = shared_data[thread_id] + shared_data[thread_id + stride];
        }
        workgroupBarrier();
    }

    if (thread_id == 0u) {
        mean_val = shared_data[0] / f32(feature_size);
    }
    workgroupBarrier();

    // Phase 2: Compute variance using shared memory reduction
    var local_var_sum = 0.0;
    for (var i = thread_id; i < feature_size; i = i + WORKGROUP_SIZE) {
        let diff = input[base_idx + i] - mean_val;
        local_var_sum = local_var_sum + diff * diff;
    }
    shared_data[thread_id] = local_var_sum;
    workgroupBarrier();

    // Reduce to find total variance sum
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (thread_id < stride) {
            shared_data[thread_id] = shared_data[thread_id] + shared_data[thread_id + stride];
        }
        workgroupBarrier();
    }

    if (thread_id == 0u) {
        variance_val = shared_data[0] / f32(feature_size);
    }
    workgroupBarrier();

    // Phase 3: Normalize and apply affine transform
    let eps = 1e-5;
    let std_inv = 1.0 / sqrt(variance_val + eps);

    for (var i = thread_id; i < feature_size; i = i + WORKGROUP_SIZE) {
        let normalized = (input[base_idx + i] - mean_val) * std_inv;
        output[base_idx + i] = normalized * gamma[i] + beta[i];
    }
}
"#;

/// Attention shader (simplified for demonstration)
pub const ATTENTION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> dims: vec4<u32>; // batch, heads, seq_len, head_dim

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch = global_id.x;
    let head = global_id.y;
    let seq_pos = global_id.z;

    let batch_size = dims.x;
    let num_heads = dims.y;
    let seq_len = dims.z;
    let head_dim = dims.w;

    if (batch >= batch_size || head >= num_heads || seq_pos >= seq_len) {
        return;
    }

    // Compute attention scores for this position
    let q_offset = (batch * num_heads + head) * seq_len * head_dim + seq_pos * head_dim;
    let k_offset = (batch * num_heads + head) * seq_len * head_dim;

    var max_score = -1e10;
    for (var i = 0u; i < seq_len; i = i + 1u) {
        var score = 0.0;
        for (var d = 0u; d < head_dim; d = d + 1u) {
            score = score + query[q_offset + d] * key[k_offset + i * head_dim + d];
        }
        score = score / sqrt(f32(head_dim));
        max_score = max(max_score, score);
    }

    // Compute softmax
    var sum = 0.0;
    var scores: array<f32, 512>; // Assuming max seq_len of 512
    for (var i = 0u; i < seq_len; i = i + 1u) {
        var score = 0.0;
        for (var d = 0u; d < head_dim; d = d + 1u) {
            score = score + query[q_offset + d] * key[k_offset + i * head_dim + d];
        }
        score = score / sqrt(f32(head_dim));
        scores[i] = exp(score - max_score);
        sum = sum + scores[i];
    }

    // Apply attention to values
    let v_offset = (batch * num_heads + head) * seq_len * head_dim;
    let out_offset = (batch * num_heads + head) * seq_len * head_dim + seq_pos * head_dim;

    for (var d = 0u; d < head_dim; d = d + 1u) {
        var result = 0.0;
        for (var i = 0u; i < seq_len; i = i + 1u) {
            result = result + (scores[i] / sum) * value[v_offset + i * head_dim + d];
        }
        output[out_offset + d] = result;
    }
}
"#;

/// Optimized attention shader with shared memory for queries and keys caching
pub const ATTENTION_SHARED_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> dims: vec4<u32>; // batch, heads, seq_len, head_dim

const TILE_SIZE: u32 = 16u;
const MAX_SEQ_LEN: u32 = 512u;
var<workgroup> query_tile: array<f32, TILE_SIZE>;
var<workgroup> key_tile: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> attention_scores: array<f32, MAX_SEQ_LEN>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch = workgroup_id.x;
    let head = workgroup_id.y;
    let seq_pos = global_id.z;
    let local_thread = local_id.x;

    let batch_size = dims.x;
    let num_heads = dims.y;
    let seq_len = dims.z;
    let head_dim = dims.w;

    if (batch >= batch_size || head >= num_heads || seq_pos >= seq_len) {
        return;
    }

    let head_offset = (batch * num_heads + head) * seq_len * head_dim;
    let q_offset = head_offset + seq_pos * head_dim;

    // Load query into shared memory
    if (local_thread < head_dim) {
        query_tile[local_thread] = query[q_offset + local_thread];
    }
    workgroupBarrier();

    // Compute attention scores using shared memory
    var max_score = -1e10;

    // Process keys in tiles
    let num_key_tiles = (seq_len + TILE_SIZE - 1u) / TILE_SIZE;
    for (var tile = 0u; tile < num_key_tiles; tile = tile + 1u) {
        // Load key tile into shared memory
        let key_start = tile * TILE_SIZE;
        let key_end = min(key_start + TILE_SIZE, seq_len);

        for (var k = key_start; k < key_end; k = k + 1u) {
            if (local_thread < head_dim) {
                let k_offset = head_offset + k * head_dim + local_thread;
                key_tile[k - key_start][local_thread] = key[k_offset];
            }
        }
        workgroupBarrier();

        // Compute scores for this tile
        for (var k = key_start; k < key_end; k = k + 1u) {
            var score = 0.0;
            for (var d = 0u; d < head_dim; d = d + 1u) {
                score = score + query_tile[d] * key_tile[k - key_start][d];
            }
            score = score / sqrt(f32(head_dim));
            attention_scores[k] = score;
            max_score = max(max_score, score);
        }
        workgroupBarrier();
    }

    // Compute softmax
    var sum = 0.0;
    for (var i = 0u; i < seq_len; i = i + 1u) {
        attention_scores[i] = exp(attention_scores[i] - max_score);
        sum = sum + attention_scores[i];
    }

    // Apply attention to values
    let out_offset = head_offset + seq_pos * head_dim;
    for (var d = 0u; d < head_dim; d = d + 1u) {
        var result = 0.0;
        for (var i = 0u; i < seq_len; i = i + 1u) {
            let v_offset = head_offset + i * head_dim + d;
            result = result + (attention_scores[i] / sum) * value[v_offset];
        }
        output[out_offset + d] = result;
    }
}
"#;

/// Batch normalization shader
pub const BATCH_NORM_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read> running_mean: array<f32>;
@group(0) @binding(4) var<storage, read> running_var: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> dims: vec3<u32>; // batch_size, channels, spatial_size

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let batch_size = dims.x;
    let channels = dims.y;
    let spatial_size = dims.z;
    let total_size = batch_size * channels * spatial_size;

    if (idx >= total_size) {
        return;
    }

    let channel = (idx / spatial_size) % channels;
    let eps = 1e-5;

    let normalized = (input[idx] - running_mean[channel]) / sqrt(running_var[channel] + eps);
    output[idx] = normalized * gamma[channel] + beta[channel];
}
"#;

/// Dropout shader with random seed support
pub const DROPOUT_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read_write> mask: array<u32>;
@group(0) @binding(3) var<uniform> params: vec3<f32>; // dropout_prob, scale, seed

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn random_float(seed: u32, idx: u32) -> f32 {
    let hash = pcg_hash(seed + idx);
    return f32(hash) / f32(0xFFFFFFFFu);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let size = arrayLength(&input);

    if (idx >= size) {
        return;
    }

    let dropout_prob = params.x;
    let scale = params.y;
    let seed = u32(params.z);

    let rand_val = random_float(seed, idx);
    let keep = rand_val > dropout_prob;

    mask[idx] = u32(keep);
    output[idx] = select(0.0, input[idx] * scale, keep);
}
"#;

/// Embedding lookup shader
pub const EMBEDDING_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read> embeddings: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec3<u32>; // seq_len, vocab_size, embed_dim

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let seq_pos = global_id.x;
    let embed_dim_idx = global_id.y;
    let seq_len = dims.x;
    let vocab_size = dims.y;
    let embed_dim = dims.z;

    if (seq_pos >= seq_len || embed_dim_idx >= embed_dim) {
        return;
    }

    let token_id = indices[seq_pos];
    if (token_id >= vocab_size) {
        output[seq_pos * embed_dim + embed_dim_idx] = 0.0;
        return;
    }

    let embedding_idx = token_id * embed_dim + embed_dim_idx;
    output[seq_pos * embed_dim + embed_dim_idx] = embeddings[embedding_idx];
}
"#;

/// Positional encoding shader (sinusoidal)
pub const POSITIONAL_ENCODING_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> embeddings: array<f32>;
@group(0) @binding(1) var<uniform> dims: vec2<u32>; // seq_len, embed_dim

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pos = global_id.x;
    let dim_idx = global_id.y;
    let seq_len = dims.x;
    let embed_dim = dims.y;

    if (pos >= seq_len || dim_idx >= embed_dim) {
        return;
    }

    let pos_f = f32(pos);
    let dim_f = f32(dim_idx);
    let embed_dim_f = f32(embed_dim);

    let angle = pos_f / pow(10000.0, 2.0 * floor(dim_f / 2.0) / embed_dim_f);

    let pos_encoding = select(cos(angle), sin(angle), dim_idx % 2u == 0u);

    let idx = pos * embed_dim + dim_idx;
    embeddings[idx] = embeddings[idx] + pos_encoding;
}
"#;

/// Fused matrix multiplication with bias and ReLU activation
pub const MATMUL_BIAS_RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;
@group(0) @binding(4) var<uniform> dims: vec3<u32>; // M, N, K

const TILE_SIZE: u32 = 16u;
var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    let local_row = local_id.x;
    let local_col = local_id.y;

    let M = dims.x;
    let N = dims.y;
    let K = dims.z;

    var sum = 0.0;

    // Iterate over tiles for matrix multiplication
    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;
    for (var tile = 0u; tile < num_tiles; tile = tile + 1u) {
        // Load tiles into shared memory
        let a_row = workgroup_id.x * TILE_SIZE + local_row;
        let a_col = tile * TILE_SIZE + local_col;
        let b_row = tile * TILE_SIZE + local_row;
        let b_col = workgroup_id.y * TILE_SIZE + local_col;

        if (a_row < M && a_col < K) {
            tile_a[local_row][local_col] = a[a_row * K + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }

        if (b_row < K && b_col < N) {
            tile_b[local_row][local_col] = b[b_row * N + b_col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product using shared memory
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[local_row][k] * tile_b[k][local_col];
        }

        workgroupBarrier();
    }

    // Apply bias and ReLU activation
    if (row < M && col < N) {
        let biased = sum + bias[col];
        result[row * N + col] = max(0.0, biased);
    }
}
"#;
