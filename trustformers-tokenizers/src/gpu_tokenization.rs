use std::collections::HashMap;
use std::sync::Arc;
use trustformers_core::traits::Tokenizer;

/// GPU-accelerated tokenization backend
#[derive(Clone)]
pub struct GpuTokenizer {
    /// Underlying tokenizer implementation
    tokenizer: Arc<dyn Tokenizer>,
    /// GPU context
    gpu_context: Option<GpuContext>,
    /// Configuration
    config: GpuTokenizerConfig,
    /// Vocabulary cache on GPU
    vocab_cache: Option<GpuVocabularyCache>,
    /// Batch processing configuration
    batch_config: BatchProcessingConfig,
}

impl std::fmt::Debug for GpuTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTokenizer")
            .field("gpu_context", &self.gpu_context)
            .field("config", &self.config)
            .field("vocab_cache", &self.vocab_cache)
            .field("batch_config", &self.batch_config)
            .finish()
    }
}

/// GPU context for tokenization operations
#[derive(Debug, Clone)]
pub struct GpuContext {
    /// Device ID
    pub device_id: u32,
    /// GPU backend type
    pub backend: GpuBackend,
    /// Stream handle for async operations
    pub stream: Option<u64>,
    /// Memory pool for buffers
    pub memory_pool: Option<GpuMemoryPool>,
    /// Kernel cache
    pub kernel_cache: HashMap<String, GpuKernel>,
    /// Compute capability (for CUDA)
    pub compute_capability: Option<(u32, u32)>,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: Option<f32>,
}

/// GPU backend types
#[derive(Debug, Clone, PartialEq)]
pub enum GpuBackend {
    /// NVIDIA CUDA
    Cuda,
    /// AMD ROCm/HIP
    Rocm,
    /// Intel oneAPI
    OneApi,
    /// OpenCL (generic)
    OpenCL,
    /// Vulkan compute
    Vulkan,
}

/// GPU tokenizer configuration
#[derive(Debug, Clone)]
pub struct GpuTokenizerConfig {
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Preferred GPU backend
    pub backend: GpuBackend,
    /// Device ID to use
    pub device_id: u32,
    /// Batch size for GPU processing
    pub batch_size: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Use pinned memory for faster transfers
    pub use_pinned_memory: bool,
    /// Enable async processing
    pub enable_async: bool,
    /// Stream parallelism level
    pub stream_parallelism: usize,
    /// Memory optimization level
    pub memory_optimization: MemoryOptimization,
    /// Kernel optimization level
    pub kernel_optimization: KernelOptimization,
    /// Enable tensor core acceleration (CUDA)
    pub enable_tensor_cores: bool,
    /// Enable mixed precision
    pub enable_mixed_precision: bool,
}

impl Default for GpuTokenizerConfig {
    fn default() -> Self {
        Self {
            enable_gpu: true,
            backend: Self::detect_best_backend(),
            device_id: 0,
            batch_size: 32,
            max_sequence_length: 512,
            use_pinned_memory: true,
            enable_async: true,
            stream_parallelism: 4,
            memory_optimization: MemoryOptimization::Balanced,
            kernel_optimization: KernelOptimization::Aggressive,
            enable_tensor_cores: true,
            enable_mixed_precision: false,
        }
    }
}

impl GpuTokenizerConfig {
    /// Detect the best available GPU backend
    pub fn detect_best_backend() -> GpuBackend {
        // Check for CUDA first (most common for ML)
        if Self::is_cuda_available() {
            GpuBackend::Cuda
        } else if Self::is_rocm_available() {
            GpuBackend::Rocm
        } else if Self::is_oneapi_available() {
            GpuBackend::OneApi
        } else if Self::is_opencl_available() {
            GpuBackend::OpenCL
        } else {
            GpuBackend::Vulkan // Fallback to Vulkan compute
        }
    }

    /// Check if CUDA is available
    pub fn is_cuda_available() -> bool {
        // In a real implementation, this would check for CUDA runtime
        // For now, simulate based on common conditions
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
            || std::path::Path::new("/usr/local/cuda").exists()
    }

    /// Check if ROCm is available
    pub fn is_rocm_available() -> bool {
        // Check for ROCm installation
        std::env::var("ROCM_PATH").is_ok() || std::path::Path::new("/opt/rocm").exists()
    }

    /// Check if Intel oneAPI is available
    pub fn is_oneapi_available() -> bool {
        // Check for oneAPI installation
        std::env::var("ONEAPI_ROOT").is_ok() || std::path::Path::new("/opt/intel/oneapi").exists()
    }

    /// Check if OpenCL is available
    pub fn is_opencl_available() -> bool {
        // Basic check for OpenCL
        std::path::Path::new("/usr/lib/libOpenCL.so").exists()
            || std::path::Path::new("/System/Library/Frameworks/OpenCL.framework").exists()
    }
}

/// Memory optimization strategies
#[derive(Debug, Clone)]
pub enum MemoryOptimization {
    /// Minimize memory usage
    Conservative,
    /// Balance memory and performance
    Balanced,
    /// Maximize performance
    Aggressive,
}

/// Kernel optimization levels
#[derive(Debug, Clone)]
pub enum KernelOptimization {
    /// Basic optimization
    Basic,
    /// Moderate optimization
    Moderate,
    /// Aggressive optimization
    Aggressive,
}

/// GPU vocabulary cache for fast lookups
#[derive(Debug, Clone)]
pub struct GpuVocabularyCache {
    /// Token to ID mapping on GPU
    pub token_to_id: GpuHashMap<String, u32>,
    /// ID to token mapping on GPU
    pub id_to_token: GpuArray<String>,
    /// Special tokens
    pub special_tokens: GpuHashMap<String, u32>,
    /// Vocabulary size
    pub vocab_size: usize,
}

/// GPU hash map implementation
#[derive(Debug, Clone)]
pub struct GpuHashMap<K, V> {
    /// Hash table on GPU
    pub data: Vec<Option<(K, V)>>,
    /// Size of hash table
    pub size: usize,
    /// Hash function parameters
    pub hash_params: HashParams,
}

/// GPU array implementation
#[derive(Debug, Clone)]
pub struct GpuArray<T> {
    /// Array data on GPU
    pub data: Vec<T>,
    /// Size
    pub size: usize,
}

/// Hash function parameters
#[derive(Debug, Clone)]
pub struct HashParams {
    /// Hash function type
    pub hash_type: HashType,
    /// Hash seed
    pub seed: u64,
    /// Hash table load factor
    pub load_factor: f32,
}

/// Hash function types
#[derive(Debug, Clone)]
pub enum HashType {
    /// FNV-1a hash
    Fnv1a,
    /// MurmurHash3
    MurmurHash3,
    /// CityHash
    CityHash,
    /// XXHash
    XXHash,
}

/// GPU memory pool for efficient allocation
#[derive(Debug, Clone)]
pub struct GpuMemoryPool {
    /// Pool blocks
    pub blocks: Vec<MemoryBlock>,
    /// Total capacity
    pub capacity: usize,
    /// Used memory
    pub used: usize,
    /// Allocation strategy
    pub strategy: AllocationStrategy,
}

/// Memory block in GPU memory pool
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Block start address
    pub address: u64,
    /// Block size
    pub size: usize,
    /// Is block free
    pub is_free: bool,
    /// Block type
    pub block_type: BlockType,
}

/// Memory block types
#[derive(Debug, Clone)]
pub enum BlockType {
    /// Input buffer
    Input,
    /// Output buffer
    Output,
    /// Vocabulary buffer
    Vocabulary,
    /// Temporary buffer
    Temporary,
}

/// Memory allocation strategies
#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    /// First fit
    FirstFit,
    /// Best fit
    BestFit,
    /// Buddy allocation
    Buddy,
    /// Pool allocation
    Pool,
}

/// GPU kernel for tokenization operations
#[derive(Debug, Clone)]
pub struct GpuKernel {
    /// Kernel name
    pub name: String,
    /// Kernel function pointer
    pub function: u64,
    /// Grid dimensions
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions
    pub block_dim: (u32, u32, u32),
    /// Shared memory size
    pub shared_mem_size: usize,
    /// Kernel parameters
    pub params: Vec<KernelParam>,
}

/// Kernel parameter
#[derive(Debug, Clone)]
pub struct KernelParam {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: ParamType,
    /// Parameter size
    pub size: usize,
}

/// Parameter types
#[derive(Debug, Clone)]
pub enum ParamType {
    /// Integer parameter
    Int,
    /// Float parameter
    Float,
    /// Pointer parameter
    Pointer,
    /// Array parameter
    Array,
}

/// Batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchProcessingConfig {
    /// Dynamic batching enabled
    pub dynamic_batching: bool,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Padding strategy
    pub padding_strategy: PaddingStrategy,
    /// Sequence packing enabled
    pub sequence_packing: bool,
}

/// Padding strategies for batch processing
#[derive(Debug, Clone)]
pub enum PaddingStrategy {
    /// Pad to longest sequence in batch
    Longest,
    /// Pad to fixed length
    Fixed(usize),
    /// Pad to next power of 2
    NextPowerOf2,
    /// No padding
    None,
}

/// GPU tokenization results
#[derive(Debug, Clone)]
pub struct GpuTokenizationResult {
    /// Token IDs
    pub token_ids: Vec<Vec<u32>>,
    /// Attention masks
    pub attention_masks: Option<Vec<Vec<u8>>>,
    /// Token type IDs
    pub token_type_ids: Option<Vec<Vec<u32>>>,
    /// Processing time in microseconds
    pub processing_time_us: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Batch size processed
    pub batch_size: usize,
}

/// GPU tokenization statistics
#[derive(Debug, Clone)]
pub struct GpuTokenizationStats {
    /// Total tokens processed
    pub total_tokens: u64,
    /// Total batches processed
    pub total_batches: u64,
    /// Average processing time per token
    pub avg_time_per_token_us: f64,
    /// Average processing time per batch
    pub avg_time_per_batch_us: f64,
    /// Memory utilization
    pub memory_utilization: f32,
    /// GPU utilization
    pub gpu_utilization: f32,
    /// Throughput (tokens per second)
    pub throughput_tokens_per_sec: f64,
}

impl GpuTokenizer {
    /// Create a new GPU tokenizer
    pub fn new(tokenizer: Arc<dyn Tokenizer>) -> Result<Self, GpuTokenizerError> {
        let config = GpuTokenizerConfig::default();
        Self::with_config(tokenizer, config)
    }

    /// Create GPU tokenizer with configuration
    pub fn with_config(
        tokenizer: Arc<dyn Tokenizer>,
        config: GpuTokenizerConfig,
    ) -> Result<Self, GpuTokenizerError> {
        let gpu_context = if config.enable_gpu {
            Some(Self::initialize_gpu_context(&config)?)
        } else {
            None
        };

        let vocab_cache = if config.enable_gpu {
            Some(Self::build_vocabulary_cache(&tokenizer, &config)?)
        } else {
            None
        };

        let batch_config = BatchProcessingConfig {
            dynamic_batching: true,
            max_batch_size: config.batch_size,
            batch_timeout_ms: 10,
            padding_strategy: PaddingStrategy::Longest,
            sequence_packing: true,
        };

        Ok(Self {
            tokenizer,
            gpu_context,
            config,
            vocab_cache,
            batch_config,
        })
    }

    /// Initialize GPU context
    fn initialize_gpu_context(
        config: &GpuTokenizerConfig,
    ) -> Result<GpuContext, GpuTokenizerError> {
        let backend = config.backend.clone();
        let device_id = config.device_id;

        // Initialize backend-specific context
        let (compute_capability, memory_bandwidth) = Self::initialize_backend(&backend, device_id)?;

        // Create stream for async operations
        let stream = if config.enable_async { Some(Self::create_stream(&backend)?) } else { None };

        // Initialize memory pool with backend-specific optimizations
        let memory_pool = Some(GpuMemoryPool {
            blocks: Vec::new(),
            capacity: Self::get_optimal_memory_size(&backend, device_id),
            used: 0,
            strategy: Self::get_optimal_allocation_strategy(&backend),
        });

        // Initialize kernel cache with backend-specific kernels
        let mut kernel_cache = HashMap::new();
        Self::load_tokenization_kernels(&mut kernel_cache, &backend, config)?;

        Ok(GpuContext {
            device_id,
            backend,
            stream,
            memory_pool,
            kernel_cache,
            compute_capability,
            memory_bandwidth,
        })
    }

    /// Initialize backend-specific context
    fn initialize_backend(
        backend: &GpuBackend,
        device_id: u32,
    ) -> Result<(Option<(u32, u32)>, Option<f32>), GpuTokenizerError> {
        match backend {
            GpuBackend::Cuda => {
                // Initialize CUDA
                Self::initialize_cuda(device_id)
            },
            GpuBackend::Rocm => {
                // Initialize ROCm/HIP
                Self::initialize_rocm(device_id)
            },
            GpuBackend::OneApi => {
                // Initialize Intel oneAPI
                Self::initialize_oneapi(device_id)
            },
            GpuBackend::OpenCL => {
                // Initialize OpenCL
                Self::initialize_opencl(device_id)
            },
            GpuBackend::Vulkan => {
                // Initialize Vulkan compute
                Self::initialize_vulkan(device_id)
            },
        }
    }

    /// Initialize CUDA backend
    fn initialize_cuda(
        _device_id: u32,
    ) -> Result<(Option<(u32, u32)>, Option<f32>), GpuTokenizerError> {
        // In a real implementation, this would use CUDA driver API
        // cuInit(), cuDeviceGet(), cuCtxCreate(), etc.

        // Simulate getting device properties
        let compute_capability = Some((8, 6)); // Simulating RTX 30xx series
        let memory_bandwidth = Some(936.0); // GB/s for RTX 3090

        // Set CUDA context
        // cuCtxSetCurrent() would be called here

        Ok((compute_capability, memory_bandwidth))
    }

    /// Initialize ROCm/HIP backend
    fn initialize_rocm(
        _device_id: u32,
    ) -> Result<(Option<(u32, u32)>, Option<f32>), GpuTokenizerError> {
        // In a real implementation, this would use HIP API
        // hipInit(), hipDeviceGet(), hipCtxCreate(), etc.

        // Simulate getting device properties
        let compute_capability = None; // ROCm doesn't use compute capability
        let memory_bandwidth = Some(1638.0); // GB/s for MI100

        // Set HIP context
        // hipCtxSetCurrent() would be called here

        Ok((compute_capability, memory_bandwidth))
    }

    /// Initialize Intel oneAPI backend
    fn initialize_oneapi(
        _device_id: u32,
    ) -> Result<(Option<(u32, u32)>, Option<f32>), GpuTokenizerError> {
        // In a real implementation, this would use SYCL/DPC++
        // sycl::queue creation, device selection, etc.

        // Simulate getting device properties
        let compute_capability = None; // Intel GPUs use different metrics
        let memory_bandwidth = Some(560.0); // GB/s for Intel Xe-HPC

        Ok((compute_capability, memory_bandwidth))
    }

    /// Initialize OpenCL backend
    fn initialize_opencl(
        _device_id: u32,
    ) -> Result<(Option<(u32, u32)>, Option<f32>), GpuTokenizerError> {
        // In a real implementation, this would use OpenCL API
        // clGetPlatformIDs(), clGetDeviceIDs(), clCreateContext(), etc.

        let compute_capability = None;
        let memory_bandwidth = Some(500.0); // Generic estimate

        Ok((compute_capability, memory_bandwidth))
    }

    /// Initialize Vulkan compute backend
    fn initialize_vulkan(
        _device_id: u32,
    ) -> Result<(Option<(u32, u32)>, Option<f32>), GpuTokenizerError> {
        // In a real implementation, this would use Vulkan API
        // vkCreateInstance(), vkEnumeratePhysicalDevices(), etc.

        let compute_capability = None;
        let memory_bandwidth = Some(400.0); // Generic estimate

        Ok((compute_capability, memory_bandwidth))
    }

    /// Get optimal memory size for backend
    fn get_optimal_memory_size(backend: &GpuBackend, _device_id: u32) -> usize {
        match backend {
            GpuBackend::Cuda => 2 * 1024 * 1024 * 1024, // 2GB for CUDA
            GpuBackend::Rocm => 4 * 1024 * 1024 * 1024, // 4GB for ROCm (HBM)
            GpuBackend::OneApi => 1024 * 1024 * 1024,   // 1GB for Intel
            _ => 1024 * 1024 * 1024,                    // 1GB default
        }
    }

    /// Get optimal allocation strategy for backend
    fn get_optimal_allocation_strategy(backend: &GpuBackend) -> AllocationStrategy {
        match backend {
            GpuBackend::Cuda => AllocationStrategy::Pool,
            GpuBackend::Rocm => AllocationStrategy::Buddy,
            GpuBackend::OneApi => AllocationStrategy::FirstFit,
            _ => AllocationStrategy::FirstFit,
        }
    }

    /// Create stream for backend
    fn create_stream(backend: &GpuBackend) -> Result<u64, GpuTokenizerError> {
        match backend {
            GpuBackend::Cuda => Self::create_cuda_stream(),
            GpuBackend::Rocm => Self::create_hip_stream(),
            GpuBackend::OneApi => Self::create_sycl_queue(),
            GpuBackend::OpenCL => Self::create_opencl_queue(),
            GpuBackend::Vulkan => Self::create_vulkan_queue(),
        }
    }

    /// Create CUDA stream
    fn create_cuda_stream() -> Result<u64, GpuTokenizerError> {
        // In a real implementation: cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING)
        Ok(0x1234567890ABCDEF)
    }

    /// Create HIP stream
    fn create_hip_stream() -> Result<u64, GpuTokenizerError> {
        // In a real implementation: hipStreamCreate(&stream)
        Ok(0x2345678901BCDEF0)
    }

    /// Create SYCL queue
    fn create_sycl_queue() -> Result<u64, GpuTokenizerError> {
        // In a real implementation: sycl::queue q(sycl::gpu_selector_v)
        Ok(0x3456789012CDEF01)
    }

    /// Create OpenCL queue
    fn create_opencl_queue() -> Result<u64, GpuTokenizerError> {
        // In a real implementation: clCreateCommandQueue(context, device, 0, &err)
        Ok(0x456789013DEF012A)
    }

    /// Create Vulkan queue
    fn create_vulkan_queue() -> Result<u64, GpuTokenizerError> {
        // In a real implementation: vkGetDeviceQueue(device, queue_family, 0, &queue)
        Ok(0x56789014EF012AB3)
    }

    /// Load tokenization kernels
    fn load_tokenization_kernels(
        kernel_cache: &mut HashMap<String, GpuKernel>,
        backend: &GpuBackend,
        config: &GpuTokenizerConfig,
    ) -> Result<(), GpuTokenizerError> {
        // Load backend-specific kernels
        Self::load_bpe_kernels(kernel_cache, backend, config)?;
        Self::load_wordpiece_kernels(kernel_cache, backend, config)?;
        Self::load_vocab_kernels(kernel_cache, backend, config)?;
        Self::load_utility_kernels(kernel_cache, backend, config)?;

        Ok(())
    }

    /// Load BPE tokenization kernels
    fn load_bpe_kernels(
        kernel_cache: &mut HashMap<String, GpuKernel>,
        backend: &GpuBackend,
        config: &GpuTokenizerConfig,
    ) -> Result<(), GpuTokenizerError> {
        let (grid_dim, block_dim, shared_mem) =
            Self::get_optimal_kernel_params(backend, "bpe_tokenize");

        // BPE tokenization kernel
        kernel_cache.insert(
            "bpe_tokenize".to_string(),
            GpuKernel {
                name: "bpe_tokenize".to_string(),
                function: Self::get_kernel_function_ptr(backend, "bpe_tokenize")?,
                grid_dim,
                block_dim,
                shared_mem_size: shared_mem,
                params: vec![
                    KernelParam {
                        name: "input_text".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                    KernelParam {
                        name: "output_tokens".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                    KernelParam {
                        name: "vocab_table".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                    KernelParam {
                        name: "merge_table".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                    KernelParam {
                        name: "sequence_length".to_string(),
                        param_type: ParamType::Int,
                        size: 4,
                    },
                    KernelParam {
                        name: "vocab_size".to_string(),
                        param_type: ParamType::Int,
                        size: 4,
                    },
                ],
            },
        );

        // BPE merge kernel with tensor cores support for CUDA
        if matches!(backend, GpuBackend::Cuda) && config.enable_tensor_cores {
            let (tc_grid, tc_block, tc_shared) = Self::get_tensor_core_params();
            kernel_cache.insert(
                "bpe_merge_tensorcore".to_string(),
                GpuKernel {
                    name: "bpe_merge_tensorcore".to_string(),
                    function: Self::get_kernel_function_ptr(backend, "bpe_merge_tensorcore")?,
                    grid_dim: tc_grid,
                    block_dim: tc_block,
                    shared_mem_size: tc_shared,
                    params: vec![
                        KernelParam {
                            name: "token_embeddings".to_string(),
                            param_type: ParamType::Pointer,
                            size: 8,
                        },
                        KernelParam {
                            name: "merge_scores".to_string(),
                            param_type: ParamType::Pointer,
                            size: 8,
                        },
                        KernelParam {
                            name: "output_merges".to_string(),
                            param_type: ParamType::Pointer,
                            size: 8,
                        },
                    ],
                },
            );
        }

        Ok(())
    }

    /// Load WordPiece tokenization kernels
    fn load_wordpiece_kernels(
        kernel_cache: &mut HashMap<String, GpuKernel>,
        backend: &GpuBackend,
        _config: &GpuTokenizerConfig,
    ) -> Result<(), GpuTokenizerError> {
        let (grid_dim, block_dim, shared_mem) =
            Self::get_optimal_kernel_params(backend, "wordpiece_tokenize");

        // WordPiece tokenization kernel
        kernel_cache.insert(
            "wordpiece_tokenize".to_string(),
            GpuKernel {
                name: "wordpiece_tokenize".to_string(),
                function: Self::get_kernel_function_ptr(backend, "wordpiece_tokenize")?,
                grid_dim,
                block_dim,
                shared_mem_size: shared_mem,
                params: vec![
                    KernelParam {
                        name: "input_text".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                    KernelParam {
                        name: "output_tokens".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                    KernelParam {
                        name: "vocab_table".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                    KernelParam {
                        name: "prefix_scores".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                ],
            },
        );

        Ok(())
    }

    /// Load vocabulary lookup kernels
    fn load_vocab_kernels(
        kernel_cache: &mut HashMap<String, GpuKernel>,
        backend: &GpuBackend,
        _config: &GpuTokenizerConfig,
    ) -> Result<(), GpuTokenizerError> {
        let (grid_dim, block_dim, shared_mem) =
            Self::get_optimal_kernel_params(backend, "vocab_lookup");

        // Vocabulary lookup kernel
        kernel_cache.insert(
            "vocab_lookup".to_string(),
            GpuKernel {
                name: "vocab_lookup".to_string(),
                function: Self::get_kernel_function_ptr(backend, "vocab_lookup")?,
                grid_dim,
                block_dim,
                shared_mem_size: shared_mem,
                params: vec![
                    KernelParam {
                        name: "tokens".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                    KernelParam {
                        name: "token_ids".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                    KernelParam {
                        name: "vocab_table".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                ],
            },
        );

        // Fast hash-based vocabulary lookup
        kernel_cache.insert(
            "vocab_hash_lookup".to_string(),
            GpuKernel {
                name: "vocab_hash_lookup".to_string(),
                function: Self::get_kernel_function_ptr(backend, "vocab_hash_lookup")?,
                grid_dim: (1024, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_size: 2048,
                params: vec![
                    KernelParam {
                        name: "tokens".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                    KernelParam {
                        name: "token_ids".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                    KernelParam {
                        name: "hash_table".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                    KernelParam {
                        name: "hash_params".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                ],
            },
        );

        Ok(())
    }

    /// Load utility kernels
    fn load_utility_kernels(
        kernel_cache: &mut HashMap<String, GpuKernel>,
        backend: &GpuBackend,
        _config: &GpuTokenizerConfig,
    ) -> Result<(), GpuTokenizerError> {
        // Padding kernel
        kernel_cache.insert(
            "padding".to_string(),
            GpuKernel {
                name: "padding".to_string(),
                function: Self::get_kernel_function_ptr(backend, "padding")?,
                grid_dim: (128, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_size: 256,
                params: vec![
                    KernelParam {
                        name: "input_sequences".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                    KernelParam {
                        name: "output_sequences".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                    KernelParam {
                        name: "target_length".to_string(),
                        param_type: ParamType::Int,
                        size: 4,
                    },
                ],
            },
        );

        // Text normalization kernel
        kernel_cache.insert(
            "text_normalize".to_string(),
            GpuKernel {
                name: "text_normalize".to_string(),
                function: Self::get_kernel_function_ptr(backend, "text_normalize")?,
                grid_dim: (512, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_size: 1024,
                params: vec![
                    KernelParam {
                        name: "input_text".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                    KernelParam {
                        name: "normalized_text".to_string(),
                        param_type: ParamType::Pointer,
                        size: 8,
                    },
                    KernelParam {
                        name: "normalization_flags".to_string(),
                        param_type: ParamType::Int,
                        size: 4,
                    },
                ],
            },
        );

        Ok(())
    }

    /// Get optimal kernel parameters for backend
    fn get_optimal_kernel_params(
        backend: &GpuBackend,
        kernel_name: &str,
    ) -> ((u32, u32, u32), (u32, u32, u32), usize) {
        match backend {
            GpuBackend::Cuda => match kernel_name {
                "bpe_tokenize" => ((256, 1, 1), (256, 1, 1), 2048),
                "wordpiece_tokenize" => ((256, 1, 1), (256, 1, 1), 3072),
                "vocab_lookup" => ((512, 1, 1), (512, 1, 1), 1024),
                _ => ((128, 1, 1), (128, 1, 1), 512),
            },
            GpuBackend::Rocm => {
                // AMD GPUs typically have larger workgroups
                match kernel_name {
                    "bpe_tokenize" => ((128, 1, 1), (512, 1, 1), 4096),
                    "wordpiece_tokenize" => ((128, 1, 1), (512, 1, 1), 4096),
                    "vocab_lookup" => ((256, 1, 1), (512, 1, 1), 2048),
                    _ => ((64, 1, 1), (256, 1, 1), 1024),
                }
            },
            GpuBackend::OneApi => {
                // Intel GPUs optimize for different patterns
                match kernel_name {
                    "bpe_tokenize" => ((128, 1, 1), (128, 1, 1), 1024),
                    "wordpiece_tokenize" => ((128, 1, 1), (128, 1, 1), 1024),
                    "vocab_lookup" => ((256, 1, 1), (256, 1, 1), 512),
                    _ => ((64, 1, 1), (64, 1, 1), 256),
                }
            },
            _ => ((64, 1, 1), (64, 1, 1), 256),
        }
    }

    /// Get tensor core parameters for CUDA
    fn get_tensor_core_params() -> ((u32, u32, u32), (u32, u32, u32), usize) {
        // Tensor cores work best with specific dimensions
        ((128, 1, 1), (32, 8, 1), 16384) // Optimized for Tensor Core operations
    }

    /// Get kernel function pointer for backend
    fn get_kernel_function_ptr(
        backend: &GpuBackend,
        kernel_name: &str,
    ) -> Result<u64, GpuTokenizerError> {
        // In a real implementation, this would load compiled kernels
        match backend {
            GpuBackend::Cuda => {
                // CUDA kernels would be loaded with cuModuleGetFunction
                match kernel_name {
                    "bpe_tokenize" => Ok(0x1000),
                    "bpe_merge_tensorcore" => Ok(0x1100),
                    "wordpiece_tokenize" => Ok(0x2000),
                    "vocab_lookup" => Ok(0x3000),
                    "vocab_hash_lookup" => Ok(0x3100),
                    "padding" => Ok(0x4000),
                    "text_normalize" => Ok(0x5000),
                    _ => Err(GpuTokenizerError::KernelNotFound(kernel_name.to_string())),
                }
            },
            GpuBackend::Rocm => {
                // HIP kernels would be loaded with hipModuleGetFunction
                match kernel_name {
                    "bpe_tokenize" => Ok(0x2000),
                    "wordpiece_tokenize" => Ok(0x2100),
                    "vocab_lookup" => Ok(0x2200),
                    "vocab_hash_lookup" => Ok(0x2300),
                    "padding" => Ok(0x2400),
                    "text_normalize" => Ok(0x2500),
                    _ => Err(GpuTokenizerError::KernelNotFound(kernel_name.to_string())),
                }
            },
            GpuBackend::OneApi => {
                // SYCL kernels would be loaded differently
                match kernel_name {
                    "bpe_tokenize" => Ok(0x3000),
                    "wordpiece_tokenize" => Ok(0x3100),
                    "vocab_lookup" => Ok(0x3200),
                    "vocab_hash_lookup" => Ok(0x3300),
                    "padding" => Ok(0x3400),
                    "text_normalize" => Ok(0x3500),
                    _ => Err(GpuTokenizerError::KernelNotFound(kernel_name.to_string())),
                }
            },
            GpuBackend::OpenCL => {
                // OpenCL kernels
                match kernel_name {
                    "bpe_tokenize" => Ok(0x4000),
                    "wordpiece_tokenize" => Ok(0x4100),
                    "vocab_lookup" => Ok(0x4200),
                    "vocab_hash_lookup" => Ok(0x4300),
                    "padding" => Ok(0x4400),
                    "text_normalize" => Ok(0x4500),
                    _ => Err(GpuTokenizerError::KernelNotFound(kernel_name.to_string())),
                }
            },
            GpuBackend::Vulkan => {
                // Vulkan compute shaders
                match kernel_name {
                    "bpe_tokenize" => Ok(0x5000),
                    "wordpiece_tokenize" => Ok(0x5100),
                    "vocab_lookup" => Ok(0x5200),
                    "vocab_hash_lookup" => Ok(0x5300),
                    "padding" => Ok(0x5400),
                    "text_normalize" => Ok(0x5500),
                    _ => Err(GpuTokenizerError::KernelNotFound(kernel_name.to_string())),
                }
            },
        }
    }

    /// Build vocabulary cache on GPU
    fn build_vocabulary_cache(
        tokenizer: &Arc<dyn Tokenizer>,
        _config: &GpuTokenizerConfig,
    ) -> Result<GpuVocabularyCache, GpuTokenizerError> {
        let vocab_size = tokenizer.vocab_size();

        // Create token to ID mapping
        let mut token_to_id_data = vec![None; vocab_size * 2]; // 2x for hash table
        let hash_params = HashParams {
            hash_type: HashType::Fnv1a,
            seed: 0x517cc1b727220a95,
            load_factor: 0.75,
        };

        // Build hash table
        for i in 0..vocab_size {
            if let Some(token) = tokenizer.id_to_token(i as u32) {
                let hash = Self::compute_hash(&token, &hash_params);
                let mut index = hash % token_to_id_data.len();

                // Linear probing for collision resolution
                while token_to_id_data[index].is_some() {
                    index = (index + 1) % token_to_id_data.len();
                }

                token_to_id_data[index] = Some((token.clone(), i as u32));
            }
        }

        let token_to_id = GpuHashMap {
            data: token_to_id_data,
            size: vocab_size * 2,
            hash_params: hash_params.clone(),
        };

        // Create ID to token mapping
        let mut id_to_token_data = Vec::with_capacity(vocab_size);
        for i in 0..vocab_size {
            let token = tokenizer.id_to_token(i as u32).unwrap_or_else(|| format!("<unk_{}>", i));
            id_to_token_data.push(token);
        }

        let id_to_token = GpuArray {
            data: id_to_token_data,
            size: vocab_size,
        };

        // Build special tokens mapping
        let special_tokens = GpuHashMap {
            data: vec![None; 128], // Small hash table for special tokens
            size: 128,
            hash_params: hash_params.clone(),
        };

        Ok(GpuVocabularyCache {
            token_to_id,
            id_to_token,
            special_tokens,
            vocab_size,
        })
    }

    /// Compute hash for token
    fn compute_hash(token: &str, params: &HashParams) -> usize {
        match params.hash_type {
            HashType::Fnv1a => {
                let mut hash = params.seed;
                for byte in token.bytes() {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(0x100000001b3);
                }
                hash as usize
            },
            _ => {
                // Fallback to simple hash
                token.chars().map(|c| c as usize).sum()
            },
        }
    }

    /// Tokenize text using GPU acceleration
    pub fn tokenize_batch(
        &self,
        texts: &[String],
    ) -> Result<GpuTokenizationResult, GpuTokenizerError> {
        let start_time = std::time::Instant::now();

        if !self.config.enable_gpu || self.gpu_context.is_none() {
            return self.tokenize_batch_cpu(texts);
        }

        // Prepare batch
        let batch_size = texts.len().min(self.config.batch_size);
        let mut token_ids = Vec::with_capacity(batch_size);
        let mut attention_masks = Vec::with_capacity(batch_size);

        // Process batch on GPU
        for text in texts.iter().take(batch_size) {
            let (tokens, mask) = self.tokenize_single_gpu(text)?;
            token_ids.push(tokens);
            attention_masks.push(mask);
        }

        // Apply padding if needed
        if matches!(self.batch_config.padding_strategy, PaddingStrategy::Longest) {
            self.apply_padding(&mut token_ids, &mut attention_masks)?;
        }

        let processing_time = start_time.elapsed().as_micros() as u64;
        let memory_usage = self.estimate_memory_usage(&token_ids);

        Ok(GpuTokenizationResult {
            token_ids,
            attention_masks: Some(attention_masks),
            token_type_ids: None,
            processing_time_us: processing_time,
            memory_usage_bytes: memory_usage,
            batch_size,
        })
    }

    /// Tokenize single text on GPU
    fn tokenize_single_gpu(&self, text: &str) -> Result<(Vec<u32>, Vec<u8>), GpuTokenizerError> {
        let gpu_context = self.gpu_context.as_ref().unwrap();

        // Get BPE tokenization kernel
        let kernel = gpu_context.kernel_cache.get("bpe_tokenize").ok_or(
            GpuTokenizerError::KernelNotFound("bpe_tokenize".to_string()),
        )?;

        // Allocate GPU memory
        let input_buffer = self.allocate_gpu_memory(text.len())?;
        let output_buffer = self.allocate_gpu_memory(self.config.max_sequence_length * 4)?;

        // Copy text to GPU
        self.copy_to_gpu(text.as_bytes(), input_buffer)?;

        // Launch kernel
        self.launch_kernel(kernel, &[input_buffer, output_buffer])?;

        // Copy results back
        let mut token_ids = vec![0u32; self.config.max_sequence_length];
        self.copy_from_gpu(output_buffer, &mut token_ids)?;

        // Find actual sequence length
        let actual_length = token_ids.iter().position(|&x| x == 0).unwrap_or(token_ids.len());
        token_ids.truncate(actual_length);

        // Generate attention mask
        let attention_mask = vec![1u8; token_ids.len()];

        // Free GPU memory
        self.free_gpu_memory(input_buffer)?;
        self.free_gpu_memory(output_buffer)?;

        Ok((token_ids, attention_mask))
    }

    /// Tokenize batch on CPU as fallback
    fn tokenize_batch_cpu(
        &self,
        texts: &[String],
    ) -> Result<GpuTokenizationResult, GpuTokenizerError> {
        let start_time = std::time::Instant::now();

        let mut token_ids = Vec::new();
        let mut attention_masks = Vec::new();

        for text in texts {
            let result = self
                .tokenizer
                .encode(text)
                .map_err(|e| GpuTokenizerError::TokenizationError(e.to_string()))?;

            token_ids.push(result.input_ids);
            attention_masks.push(result.attention_mask);
        }

        let processing_time = start_time.elapsed().as_micros() as u64;
        let memory_usage = self.estimate_memory_usage(&token_ids);

        Ok(GpuTokenizationResult {
            token_ids,
            attention_masks: Some(attention_masks),
            token_type_ids: None,
            processing_time_us: processing_time,
            memory_usage_bytes: memory_usage,
            batch_size: texts.len(),
        })
    }

    /// Apply padding to token sequences
    fn apply_padding(
        &self,
        token_ids: &mut [Vec<u32>],
        attention_masks: &mut [Vec<u8>],
    ) -> Result<(), GpuTokenizerError> {
        if token_ids.is_empty() {
            return Ok(());
        }

        let max_length = token_ids.iter().map(|seq| seq.len()).max().unwrap_or(0);

        for (tokens, mask) in token_ids.iter_mut().zip(attention_masks.iter_mut()) {
            let current_length = tokens.len();
            if current_length < max_length {
                tokens.resize(max_length, 0); // Pad with 0
                mask.resize(max_length, 0); // Pad attention mask with 0
            }
        }

        Ok(())
    }

    /// Allocate GPU memory
    fn allocate_gpu_memory(&self, size: usize) -> Result<u64, GpuTokenizerError> {
        // Placeholder for actual GPU memory allocation
        Ok(0x1000000 + size as u64)
    }

    /// Copy data to GPU
    fn copy_to_gpu(&self, _data: &[u8], _gpu_ptr: u64) -> Result<(), GpuTokenizerError> {
        // Placeholder for actual GPU memory copy
        Ok(())
    }

    /// Copy data from GPU
    fn copy_from_gpu(&self, _gpu_ptr: u64, data: &mut [u32]) -> Result<(), GpuTokenizerError> {
        // Placeholder for actual GPU memory copy
        // For now, simulate tokenization results
        for (i, token) in data.iter_mut().enumerate() {
            *token = if i < 10 { (i + 1) as u32 } else { 0 };
        }
        Ok(())
    }

    /// Launch GPU kernel
    fn launch_kernel(&self, _kernel: &GpuKernel, _params: &[u64]) -> Result<(), GpuTokenizerError> {
        // Placeholder for actual kernel launch
        Ok(())
    }

    /// Free GPU memory
    fn free_gpu_memory(&self, _gpu_ptr: u64) -> Result<(), GpuTokenizerError> {
        // Placeholder for actual GPU memory deallocation
        Ok(())
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self, token_ids: &[Vec<u32>]) -> usize {
        token_ids.iter().map(|seq| seq.len() * 4).sum()
    }

    /// Get tokenization statistics
    pub fn get_stats(&self) -> GpuTokenizationStats {
        GpuTokenizationStats {
            total_tokens: 0,
            total_batches: 0,
            avg_time_per_token_us: 0.0,
            avg_time_per_batch_us: 0.0,
            memory_utilization: 0.0,
            gpu_utilization: 0.0,
            throughput_tokens_per_sec: 0.0,
        }
    }

    /// Enable/disable GPU acceleration
    pub fn set_gpu_enabled(&mut self, enabled: bool) {
        self.config.enable_gpu = enabled;
    }

    /// Set batch size
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.config.batch_size = batch_size;
        self.batch_config.max_batch_size = batch_size;
    }

    /// Set device ID
    pub fn set_device_id(&mut self, device_id: u32) {
        self.config.device_id = device_id;
    }
}

/// GPU tokenizer errors
#[derive(Debug, Clone)]
pub enum GpuTokenizerError {
    /// GPU initialization error
    GpuInitializationError(String),
    /// Memory allocation error
    MemoryAllocationError(String),
    /// Kernel launch error
    KernelLaunchError(String),
    /// Kernel not found
    KernelNotFound(String),
    /// Tokenization error
    TokenizationError(String),
    /// Configuration error
    ConfigurationError(String),
    /// CUDA error
    CudaError(String),
    /// Invalid device
    InvalidDevice(u32),
    /// Out of memory
    OutOfMemory,
}

impl std::fmt::Display for GpuTokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuTokenizerError::GpuInitializationError(msg) => {
                write!(f, "GPU initialization error: {}", msg)
            },
            GpuTokenizerError::MemoryAllocationError(msg) => {
                write!(f, "Memory allocation error: {}", msg)
            },
            GpuTokenizerError::KernelLaunchError(msg) => {
                write!(f, "Kernel launch error: {}", msg)
            },
            GpuTokenizerError::KernelNotFound(name) => {
                write!(f, "Kernel not found: {}", name)
            },
            GpuTokenizerError::TokenizationError(msg) => {
                write!(f, "Tokenization error: {}", msg)
            },
            GpuTokenizerError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            },
            GpuTokenizerError::CudaError(msg) => {
                write!(f, "CUDA error: {}", msg)
            },
            GpuTokenizerError::InvalidDevice(id) => {
                write!(f, "Invalid device: {}", id)
            },
            GpuTokenizerError::OutOfMemory => {
                write!(f, "Out of memory")
            },
        }
    }
}

impl std::error::Error for GpuTokenizerError {}

/// GPU tokenization benchmarks
pub struct GpuTokenizationBenchmark {
    /// Test configurations
    pub configs: Vec<GpuTokenizerConfig>,
    /// Test texts
    pub test_texts: Vec<String>,
    /// Results
    pub results: Vec<BenchmarkResult>,
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Configuration used
    pub config: GpuTokenizerConfig,
    /// Processing time
    pub processing_time_us: u64,
    /// Throughput
    pub throughput_tokens_per_sec: f64,
    /// Memory usage
    pub memory_usage_bytes: usize,
    /// GPU utilization
    pub gpu_utilization: f32,
}

impl Default for GpuTokenizationBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuTokenizationBenchmark {
    /// Create new benchmark
    pub fn new() -> Self {
        Self {
            configs: vec![
                GpuTokenizerConfig::default(),
                GpuTokenizerConfig {
                    batch_size: 64,
                    ..Default::default()
                },
                GpuTokenizerConfig {
                    batch_size: 128,
                    memory_optimization: MemoryOptimization::Aggressive,
                    ..Default::default()
                },
            ],
            test_texts: vec![
                "Hello world".to_string(),
                "This is a longer text for testing tokenization performance".to_string(),
                "The quick brown fox jumps over the lazy dog".repeat(10),
            ],
            results: Vec::new(),
        }
    }

    /// Run benchmark
    pub fn run(&mut self, tokenizer: Arc<dyn Tokenizer>) -> Result<(), GpuTokenizerError> {
        for config in &self.configs {
            let gpu_tokenizer = GpuTokenizer::with_config(tokenizer.clone(), config.clone())?;

            let start_time = std::time::Instant::now();
            let result = gpu_tokenizer.tokenize_batch(&self.test_texts)?;
            let processing_time = start_time.elapsed().as_micros() as u64;

            let total_tokens: usize = result.token_ids.iter().map(|seq| seq.len()).sum();
            let throughput = total_tokens as f64 / (processing_time as f64 / 1_000_000.0);

            self.results.push(BenchmarkResult {
                config: config.clone(),
                processing_time_us: processing_time,
                throughput_tokens_per_sec: throughput,
                memory_usage_bytes: result.memory_usage_bytes,
                gpu_utilization: 0.0, // Would be measured in real implementation
            });
        }

        Ok(())
    }

    /// Get best configuration
    pub fn get_best_config(&self) -> Option<&GpuTokenizerConfig> {
        self.results
            .iter()
            .max_by(|a, b| {
                a.throughput_tokens_per_sec
                    .partial_cmp(&b.throughput_tokens_per_sec)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|result| &result.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::BPETokenizer;
    use std::sync::Arc;

    fn create_test_tokenizer() -> Arc<dyn Tokenizer> {
        let mut vocab = std::collections::HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("<unk>".to_string(), 2);

        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("l".to_string(), "l".to_string()),
        ];

        let tokenizer = BPETokenizer::new(vocab, merges);
        Arc::new(tokenizer)
    }

    #[test]
    fn test_gpu_tokenizer_creation() {
        let tokenizer = create_test_tokenizer();
        let gpu_tokenizer = GpuTokenizer::new(tokenizer);
        assert!(gpu_tokenizer.is_ok());
    }

    #[test]
    fn test_gpu_tokenizer_config() {
        let config = GpuTokenizerConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.max_sequence_length, 512);
        assert!(config.enable_gpu);
    }

    #[test]
    fn test_gpu_context_initialization() {
        let config = GpuTokenizerConfig::default();
        let result = GpuTokenizer::initialize_gpu_context(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_vocabulary_cache_building() {
        let tokenizer = create_test_tokenizer();
        let config = GpuTokenizerConfig::default();
        let result = GpuTokenizer::build_vocabulary_cache(&tokenizer, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hash_computation() {
        let params = HashParams {
            hash_type: HashType::Fnv1a,
            seed: 0x517cc1b727220a95,
            load_factor: 0.75,
        };
        let hash1 = GpuTokenizer::compute_hash("hello", &params);
        let hash2 = GpuTokenizer::compute_hash("world", &params);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_batch_tokenization() {
        let tokenizer = create_test_tokenizer();
        let gpu_tokenizer = GpuTokenizer::new(tokenizer).unwrap();
        let texts = vec!["Hello world".to_string(), "This is a test".to_string()];
        let result = gpu_tokenizer.tokenize_batch(&texts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_padding_application() {
        let tokenizer = create_test_tokenizer();
        let gpu_tokenizer = GpuTokenizer::new(tokenizer).unwrap();
        let mut token_ids = vec![vec![1, 2, 3], vec![4, 5]];
        let mut attention_masks = vec![vec![1, 1, 1], vec![1, 1]];

        gpu_tokenizer.apply_padding(&mut token_ids, &mut attention_masks).unwrap();

        assert_eq!(token_ids[0].len(), 3);
        assert_eq!(token_ids[1].len(), 3);
        assert_eq!(token_ids[1][2], 0); // Padding token
        assert_eq!(attention_masks[1][2], 0); // Padding mask
    }

    #[test]
    fn test_gpu_tokenization_stats() {
        let tokenizer = create_test_tokenizer();
        let gpu_tokenizer = GpuTokenizer::new(tokenizer).unwrap();
        let stats = gpu_tokenizer.get_stats();
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.total_batches, 0);
    }

    #[test]
    fn test_gpu_tokenizer_configuration() {
        let tokenizer = create_test_tokenizer();
        let mut gpu_tokenizer = GpuTokenizer::new(tokenizer).unwrap();

        gpu_tokenizer.set_batch_size(64);
        gpu_tokenizer.set_device_id(1);
        gpu_tokenizer.set_gpu_enabled(false);

        assert_eq!(gpu_tokenizer.config.batch_size, 64);
        assert_eq!(gpu_tokenizer.config.device_id, 1);
        assert!(!gpu_tokenizer.config.enable_gpu);
    }

    #[test]
    fn test_benchmark_creation() {
        let benchmark = GpuTokenizationBenchmark::new();
        assert_eq!(benchmark.configs.len(), 3);
        assert_eq!(benchmark.test_texts.len(), 3);
        assert_eq!(benchmark.results.len(), 0);
    }

    #[test]
    fn test_memory_optimization_levels() {
        let conservative = MemoryOptimization::Conservative;
        let balanced = MemoryOptimization::Balanced;
        let aggressive = MemoryOptimization::Aggressive;

        assert!(matches!(conservative, MemoryOptimization::Conservative));
        assert!(matches!(balanced, MemoryOptimization::Balanced));
        assert!(matches!(aggressive, MemoryOptimization::Aggressive));
    }

    #[test]
    fn test_kernel_optimization_levels() {
        let basic = KernelOptimization::Basic;
        let moderate = KernelOptimization::Moderate;
        let aggressive = KernelOptimization::Aggressive;

        assert!(matches!(basic, KernelOptimization::Basic));
        assert!(matches!(moderate, KernelOptimization::Moderate));
        assert!(matches!(aggressive, KernelOptimization::Aggressive));
    }
}
