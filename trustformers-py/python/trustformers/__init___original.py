"""
TrustformeRS: High-performance transformer library for Python
===============================================================

A drop-in replacement for Hugging Face Transformers with Rust performance.

TrustformeRS provides blazing-fast transformer models implemented in Rust with a 
Python API that's fully compatible with Hugging Face Transformers. Get the same 
functionality with significantly better performance.

Key Features:
- **High Performance**: Rust-based implementation for maximum speed
- **Full Compatibility**: Drop-in replacement for Hugging Face Transformers
- **Zero-Copy Operations**: Efficient memory usage with numpy/PyTorch interop
- **Async Support**: Non-blocking operations for production workloads
- **Advanced Caching**: Intelligent model and tokenizer caching
- **Comprehensive Debugging**: Built-in profiling and monitoring tools
- **Framework Integration**: Native support for PyTorch, JAX, and NumPy
- **MLOps Integration**: MLflow, Weights & Biases, TensorBoard support
- **Data Science Tools**: Pandas, Scikit-learn, visualization libraries
- **Advanced Training**: Mixed precision, gradient flow analysis, memory optimization

Quick Start:
-----------

.. code-block:: python

    import trustformers as tf
    
    # Load a model (just like Hugging Face)
    model = tf.AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = tf.AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize and run inference
    inputs = tokenizer("Hello, world!", return_tensors="np")
    outputs = model(**inputs)
    
    # Use pipelines for common tasks
    classifier = tf.pipeline("text-classification", 
                           model="bert-base-uncased")
    result = classifier("I love this library!")

Async Operations:
----------------

.. code-block:: python

    import asyncio
    import trustformers as tf
    
    async def main():
        # Load models asynchronously
        model = await tf.AsyncAutoModel.from_pretrained('gpt2')
        tokenizer = await tf.AsyncAutoTokenizer.from_pretrained('gpt2')
        
        # Create async pipeline
        pipeline = await tf.async_pipeline("text-generation", model=model)
        
        # Run inference asynchronously
        result = await pipeline("The future of AI is")
        print(result)
    
    asyncio.run(main())

Performance Optimization:
------------------------

.. code-block:: python

    import trustformers as tf
    
    # Enable performance monitoring
    with tf.performance_profiler.monitor():
        # Use caching for repeated operations
        model = tf.get_cached_model('bert-base-uncased')
        tokenizer = tf.get_cached_tokenizer('bert-base-uncased')
        
        # Batch processing for efficiency
        texts = ["Text 1", "Text 2", "Text 3"]
        results = tf.batch_process(texts, pipeline)
        
    # Get performance statistics
    stats = tf.get_performance_stats()
    print(f"Average inference time: {stats['avg_time']:.2f}ms")

Framework Integration:
---------------------

.. code-block:: python

    import torch
    import jax.numpy as jnp
    import numpy as np
    import trustformers as tf
    
    # NumPy integration (zero-copy when possible)
    numpy_array = np.random.randn(2, 512)
    tensor = tf.Tensor.from_numpy_zero_copy(numpy_array)
    
    # PyTorch integration
    torch_tensor = torch.randn(2, 512)
    tensor = tf.torch_to_tensor(torch_tensor)
    
    # JAX integration
    jax_array = jnp.ones((2, 512))
    tensor = tf.jax_to_tensor(jax_array)
    
    # Use with PyTorch nn.Module
    torch_model = tf.TorchModuleWrapper(model)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Create PyTorch training loop
    training_loop = tf.create_torch_training_loop(
        model=torch_model,
        loss_fn=loss_fn,
        optimizer=torch.optim.Adam(torch_model.parameters())
    )

Advanced Features:
-----------------

.. code-block:: python

    import trustformers as tf
    
    # Advanced tokenization with offset mapping
    from trustformers.tokenizer_advanced import FastTokenizer
    
    tokenizer = FastTokenizer.from_pretrained('bert-base-uncased')
    tokens, offsets = tokenizer.tokenize_with_offsets("Hello world!")
    
    # Training custom tokenizers
    from trustformers.tokenizer_advanced import create_fast_tokenizer_from_files
    
    custom_tokenizer = create_fast_tokenizer_from_files(
        training_files=['corpus.txt'],
        vocab_size=30000,
        algorithm='wordpiece'
    )
    
    # Configuration management
    config = tf.AutoConfig.from_pretrained('bert-base-uncased')
    migrated_config = tf.migrate_config(config, target_version='2.0')
    
    # Memory and debugging
    with tf.tensor_tracker.track():
        result = model(inputs)
        memory_usage = tf.tensor_tracker.get_memory_usage()
    
    # Jupyter integration (if in notebook)
    tf.display_model(model)  # Rich HTML display
    tf.display_tensor(tensor)  # Interactive tensor visualization

MLOps Integration:
-----------------

.. code-block:: python

    import trustformers as tf
    
    # Start experiment tracking
    experiment_id = tf.start_experiment(
        name="bert-classification",
        project="nlp-experiments",
        parameters={"learning_rate": 1e-5, "batch_size": 32}
    )
    
    # Log training metrics
    tf.log_metrics({"accuracy": 0.95, "loss": 0.1}, step=100)
    
    # Save model to registry
    model_uri, version = tf.save_model(
        model, "bert-classifier-v1",
        description="Fine-tuned BERT for sentiment analysis"
    )
    
    # Use different tracking backends
    manager = tf.ExperimentManager()
    manager.create_tracker("mlflow", tracking_uri="http://localhost:5000")
    manager.create_tracker("wandb", project="my-project")

Data Science Integration:
------------------------

.. code-block:: python

    import trustformers as tf
    import pandas as pd
    
    # Convert tensors to/from pandas DataFrames
    df = tf.to_dataframe(tensor, columns=['feature1', 'feature2'])
    tensor = tf.from_dataframe(df)
    
    # Analyze data with comprehensive EDA
    analysis = tf.analyze_data(df)
    print(analysis['correlation_heatmap'])
    
    # Visualization
    tf.plot_tensor(tensor, plot_type="distribution")
    tf.plot_tensor(attention_weights, plot_type="heatmap")
    
    # Scikit-learn integration
    pipeline = tf.create_sklearn_pipeline(model, preprocessing_steps=['scaler'])
    pipeline.fit(X_train, y_train)
    
    # Pandas integration
    feature_df = tf.PandasIntegration.create_feature_dataframe({
        'text_features': text_tensor,
        'numeric_features': numeric_tensor
    })

Advanced PyTorch Training:
-------------------------

.. code-block:: python

    import torch
    import trustformers as tf
    
    # Mixed precision training
    mp_config = tf.MixedPrecisionConfig(enabled=True, dtype="float16")
    mp_trainer = tf.create_mixed_precision_trainer(model, optimizer, mp_config)
    
    # Gradient flow analysis
    grad_analyzer = tf.analyze_gradient_flow(model, log_dir="./logs")
    
    # Advanced training manager with all optimizations
    training_manager = tf.create_advanced_training_manager(
        model=model,
        optimizer=optimizer,
        mixed_precision_config=mp_config
    )
    
    # Training step with all optimizations
    step_stats = training_manager.train_step(batch, loss_fn)
    
    # Get comprehensive training report
    report = training_manager.get_comprehensive_report()
    print(f"Gradient flow issues: {report['gradient_flow']['potential_issues']}")
    print(f"Memory usage: {report['memory']['gpu_memory']['allocated']:.2f} GB")

Supported Models:
----------------

- **BERT**: BertModel, BertForSequenceClassification, BertForTokenClassification
- **GPT-2**: GPT2Model, GPT2LMHeadModel, GPT2ForSequenceClassification  
- **T5**: T5Model, T5ForConditionalGeneration
- **LLaMA**: LlamaModel, LlamaForCausalLM
- **AutoModels**: Automatic model detection and loading

Pipeline Tasks:
--------------

- text-classification
- text-generation  
- fill-mask
- question-answering
- summarization
- token-classification
- translation

Installation:
------------

.. code-block:: bash

    pip install trustformers
    
    # With PyTorch support
    pip install trustformers[torch]
    
    # With JAX support  
    pip install trustformers[jax]
    
    # With all optional dependencies
    pip install trustformers[all]

For more information, visit: https://github.com/trustformers/trustformers
"""

__version__ = "0.1.0"

# Import the Rust extension module
try:
    from ._trustformers import (
        # Core classes
        Tensor,
        
        # Models
        BertModel,
        GPT2Model,
        T5Model,
        LlamaModel,
        PreTrainedModel,
        
        # Task-specific models (now implemented in Rust)
        BertForSequenceClassification,
        GPT2LMHeadModel,
        
        # Tokenizers
        WordPieceTokenizer,
        BPETokenizer,
        
        # Auto classes
        AutoModel,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoModelForQuestionAnswering,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        
        # Pipelines
        TextGenerationPipeline,
        TextClassificationPipeline,
        pipeline,
        
        # Training
        Trainer,
        TrainingArguments,
        
        # Utilities
        get_device,
        set_seed,
        enable_grad,
        no_grad,
    )
except ImportError as e:
    raise ImportError(
        "Could not import TrustformeRS C extension. "
        "Make sure the package is properly installed. "
        f"Original error: {e}"
    )

# Re-export commonly used classes at package level
from .modeling_utils import (
    PretrainedConfig,
    ModelOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    QuestionAnsweringModelOutput,
)

from .tokenization_utils import (
    PreTrainedTokenizer,
    BatchEncoding,
    tokenizer_from_pretrained,
)

from .utils import (
    logging,
    cached_path,
    download_file,
    is_torch_available,
    is_tf_available,
    requires_backends,
)

from .numpy_utils import (
    numpy_to_tensor,
    tensor_to_numpy,
    stack_numpy_arrays,
    pad_sequence,
    batch_convert_to_tensors,
    ensure_numpy_array,
    create_attention_mask_from_lengths,
    apply_numpy_function,
    tensor_stats,
    NumpyTensorWrapper,
    concatenate,
    split,
    where,
)

# Import torch utilities if available
try:
    from .torch_utils import (
        torch_to_tensor,
        tensor_to_torch,
        batch_torch_to_tensor,
        batch_tensor_to_torch,
        ensure_torch_tensor,
        pad_sequence_torch,
        TorchTensorWrapper,
        create_torch_dataloader,
        torch_optimizer_step,
    )
except ImportError:
    # PyTorch not available
    pass

# Import JAX utilities if available
try:
    from .jax_utils import (
        jax_available,
        jax_to_tensor,
        tensor_to_jax,
        batch_jax_to_tensor,
        batch_tensor_to_jax,
        ensure_jax_array,
        JAXTensorWrapper,
        create_jax_dataloader,
        jax_model_apply,
        jax_grad_fn,
        jax_vmap_fn,
        jax_jit_fn,
        create_jax_pipeline,
        setup_jax_config,
    )
except ImportError:
    # JAX not available
    pass

# Import advanced JAX integration if available
try:
    from .jax_integration import (
        JAXDevice,
        JAXGradientTransforms,
        JAXCompilation,
        JAXOptimization,
        JAXModelWrapper,
        JAXTrainingLoop,
        JAXCheckpointing,
        JAXEcosystem,
        jax_device_manager,
        jax_gradient_transforms,
        jax_compilation,
        jax_optimization,
        jax_checkpointing,
        jax_ecosystem,
    )
except ImportError:
    # Advanced JAX integration not available
    pass

# Import new advanced features (2025-07-15)
# Import PyTorch module compatibility
try:
    from .torch_module_compat import (
        TorchModuleWrapper,
        TorchModuleBert,
        TorchModuleGPT2,
        TorchLoss,
        create_torch_training_loop,
        wrap_bert_model,
        wrap_gpt2_model,
        wrap_model,
    )
except ImportError:
    # PyTorch not available
    pass

# Import batch encoding utilities
from .batch_encoding import (
    BatchEncoding as AdvancedBatchEncoding,
    BatchTokenizer,
    create_batch_encoding,
)

# Import generation utilities
from .generation_utils import (
    GenerationConfig,
    GenerationMixin,
    batch_generate,
)

# Import advanced numpy utilities
from .advanced_numpy import (
    AdvancedArray,
    StructuredArrays,
    BroadcastingUtils,
    memory_efficient_operation,
    advanced_concatenate,
    advanced_stack,
)

# Import configuration classes
from .configuration_bert import BertConfig
from .configuration_gpt2 import GPT2Config
from .configuration_t5 import T5Config
from .configuration_llama import LlamaConfig

# Import configuration management system
from .config_manager import (
    ConfigManager,
    ConfigMetadata,
    MigrationRule,
    ConfigValidationError,
    ConfigMigrationError,
    config_manager,
)

from .auto_config import (
    AutoConfig,
    load_config,
    detect_model_type,
    validate_config,
    migrate_config,
    convert_from_huggingface,
    create_custom_config,
    list_supported_models,
    get_model_info,
)

# Import caching system
from .caching import (
    ModelCache,
    TokenizerCache,
    ResultCache,
    get_model_cache,
    get_tokenizer_cache,
    get_result_cache,
    memoize,
    cache_model,
    get_cached_model,
    cache_tokenizer,
    get_cached_tokenizer,
    clear_all_caches,
    get_cache_stats,
)

# Import debugging tools
from .debugging import (
    TensorTracker,
    MemoryProfiler,
    PerformanceProfiler,
    TensorVisualizer,
    tensor_tracker,
    memory_profiler,
    performance_profiler,
    tensor_visualizer,
    debug_tensor,
    profile_operation,
    track_tensor_creation,
    get_debug_report,
    export_debug_report,
    start_memory_monitoring,
    stop_memory_monitoring,
    clear_debug_data,
)

# Import performance optimization
from .performance import (
    BatchProcessor,
    ObjectPool,
    LazyLoader,
    MemoryPool,
    CallBatcher,
    PrewarmingCache,
    PerformanceMonitor,
    batched,
    lazy_init,
    pooled,
    monitored,
    batch_process,
    get_from_pool,
    return_to_pool,
    prewarm_cache,
    get_performance_stats,
    optimize_memory,
    cleanup_performance_resources,
)

# Import Jupyter support (optional)
try:
    from .jupyter_support import (
        check_jupyter_availability,
        is_notebook_environment,
        TensorDisplay,
        ModelDisplay,
        TrainingDisplay,
        InteractiveVisualization,
        ProgressBar,
        display_tensor,
        display_model,
        create_training_display,
        create_progress_bar,
        tensor_heatmap,
        attention_visualization,
        setup_jupyter_environment,
    )
except ImportError:
    # Jupyter support not available
    pass

# Import async utilities
from .async_utils import (
    AsyncAutoModel,
    AsyncAutoTokenizer,
    AsyncPipeline,
    AsyncModelWrapper,
    AsyncTokenizerWrapper,
    AsyncBatchProcessor,
    AsyncResourceManager,
    AsyncStreamProcessor,
    async_download_file,
    async_cached_path,
    async_pipeline,
    wrap_model_async,
    wrap_tokenizer_async,
    safe_async_call,
    async_wrapper,
    get_async_executor,
    cleanup_async_executor,
    AsyncOperationError,
)

# Import MLOps integration (2025-07-15)
try:
    from .mlops_integration import (
        ExperimentConfig,
        ModelMetadata,
        ExperimentTracker,
        MLflowTracker,
        WandBTracker,
        TensorBoardTracker,
        CompositeTracker,
        ModelRegistry,
        ExperimentManager,
        get_experiment_manager,
        start_experiment,
        log_metrics,
        log_parameters,
        save_model,
        load_model,
        end_experiment,
    )
except ImportError:
    # MLOps dependencies not available
    pass

# Import Data Science Tools integration (2025-07-15)
try:
    from .data_science_tools import (
        DataScienceConfig,
        PandasIntegration,
        SklearnIntegration,
        VisualizationTools,
        DataScienceWorkflow,
        get_data_science_workflow,
        to_dataframe,
        from_dataframe,
        plot_tensor,
        analyze_data,
        create_sklearn_pipeline,
    )
except ImportError:
    # Data science dependencies not available
    pass

# Import Advanced PyTorch integration (2025-07-15)
try:
    from .pytorch_advanced import (
        GradientStats,
        MixedPrecisionConfig,
        GradientFlowAnalyzer,
        MixedPrecisionTrainer,
        AdvancedOptimizer,
        MemoryOptimizedTrainer,
        AdvancedTrainingManager,
        create_mixed_precision_trainer,
        analyze_gradient_flow,
        create_advanced_training_manager,
    )
except ImportError:
    # Advanced PyTorch features not available
    pass

# Import Distributed Training support (2025-07-15)
try:
    from .distributed_training import (
        DistributedConfig,
        DistributedManager,
        DistributedTrainer,
        DDPModelWrapper,
        HorovodModelWrapper,
        FaultToleranceManager,
        ElasticTrainingManager,
        DistributedBackend,
        get_free_port,
        setup_logging_for_distributed,
        all_gather_object,
        broadcast_object,
        barrier,
        reduce_tensor,
        get_default_distributed_manager,
        is_distributed,
        get_rank,
        get_world_size,
        is_main_process,
    )
except ImportError:
    # Distributed training dependencies not available
    pass

# Import Model Serving infrastructure (2025-07-15)
try:
    from .serving import (
        ModelConfig,
        ModelInstance,
        InferenceRequest,
        InferenceResponse,
        HealthResponse,
        ABTestConfig,
        ServingManager,
        ModelVersionManager,
        LoadBalancer,
        BatchProcessor,
        ABTestManager,
        MetricsCollector,
        ModelStatus,
        LoadBalancingStrategy,
        create_app,
        serve,
        serve_model,
    )
except ImportError:
    # Serving dependencies not available (FastAPI, uvicorn)
    pass

# Import Custom Extensions System (2025-07-16)
from .extensions import (
    # Core extension classes
    CallbackManager,
    HookRegistry,
    EventBus,
    PluginManager,
    CustomLayer,
    CustomLayerRegistry,
    
    # Configuration classes
    CallbackConfig,
    EventConfig,
    ExtensionConfig,
    
    # Event and result classes
    Event,
    CallbackResult,
    
    # Exception classes
    CallbackError,
    HookError,
    EventError,
    PluginError,
    
    # Plugin interface
    PluginInterface,
    PluginMetadata,
    
    # Built-in layer implementations
    LinearLayer,
    ActivationLayer,
    LayerProtocol,
    
    # Decorators and utilities
    callback,
    hook,
    event_handler,
    callback_context,
    hook_context,
    create_custom_layer,
    register_layer_type,
    configure_extensions,
    
    # Global instances
    callback_manager,
    hook_registry,
    event_bus,
    custom_layer_registry,
    plugin_manager,
)

# Import model-specific classes (composition pattern implemented)
from .modeling_bert import (
    BertForSequenceClassification,
    BertForTokenClassification,
    BertForQuestionAnswering,
    BertForMaskedLM,
)

from .modeling_gpt2 import (
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
)

# from .modeling_t5 import (
#     T5ForConditionalGeneration,
# )

# from .modeling_llama import (
#     LlamaForCausalLM,
# )

# Import pipeline classes
from .pipelines import (
    FillMaskPipeline,
    QuestionAnsweringPipeline,
    SummarizationPipeline,
    TokenClassificationPipeline,
    TranslationPipeline,
)

# Import data collators
from .data_collator import (
    DataCollator,
    DefaultDataCollator,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification,
    get_data_collator,
)

# Import evaluation metrics
from .evaluation import (
    EvaluationMetric,
    AccuracyMetric,
    F1Metric,
    PrecisionRecallMetric,
    MatthewsCorrCoefMetric,
    PerplexityMetric,
    BLEUMetric,
    ROUGEMetric,
    ExactMatchMetric,
    SquadMetric,
    MetricCollection,
    get_classification_metrics,
    get_generation_metrics,
    get_qa_metrics,
    compute_metric,
)

# Import trainer utilities
from .trainer_utils import (
    TrainerState,
    TrainerControl,
    TrainerCallback,
    EarlyStoppingCallback,
    ProgressCallback,
    MetricsCallback,
    TensorBoardCallback,
    WandbCallback,
    ModelCheckpointCallback,
    CallbackHandler,
    LearningRateScheduler,
    LinearScheduler,
    CosineScheduler,
    get_default_callbacks,
    setup_logging_callbacks,
)

# Define what gets imported with "from trustformers import *"
__all__ = [
    # Version
    "__version__",
    
    # Core
    "Tensor",
    "PreTrainedModel",
    "PretrainedConfig",
    
    # Models
    "BertModel",
    "BertConfig",
    "BertForSequenceClassification", # Now implemented in Rust
    
    "GPT2Model",
    "GPT2Config",
    "GPT2LMHeadModel", # Now implemented in Rust
    
    "T5Model",
    "T5Config",
    # "T5ForConditionalGeneration",
    
    "LlamaModel",
    "LlamaConfig",
    # "LlamaForCausalLM",
    
    # Configuration management system
    "ConfigManager",
    "ConfigMetadata",
    "MigrationRule",
    "ConfigValidationError",
    "ConfigMigrationError",
    "config_manager",
    "AutoConfig",
    "load_config",
    "detect_model_type",
    "validate_config",
    "migrate_config",
    "convert_from_huggingface",
    "create_custom_config",
    "list_supported_models",
    "get_model_info",
    
    # Tokenizers
    "PreTrainedTokenizer",
    "WordPieceTokenizer",
    "BPETokenizer",
    "BatchEncoding",
    
    # Auto classes
    "AutoModel",
    "AutoTokenizer",
    "AutoModelForSequenceClassification",
    "AutoModelForTokenClassification",
    "AutoModelForQuestionAnswering", 
    "AutoModelForCausalLM",
    "AutoModelForMaskedLM",
    
    # Pipelines
    "pipeline",
    "TextGenerationPipeline",
    "TextClassificationPipeline",
    "FillMaskPipeline",
    "QuestionAnsweringPipeline",
    "SummarizationPipeline",
    "TokenClassificationPipeline", 
    "TranslationPipeline",
    
    # Training
    "Trainer",
    "TrainingArguments",
    "TrainerState",
    "TrainerControl", 
    "TrainerCallback",
    "EarlyStoppingCallback",
    "ProgressCallback",
    "MetricsCallback",
    "TensorBoardCallback",
    "WandbCallback",
    "ModelCheckpointCallback",
    "CallbackHandler",
    
    # Data collators
    "DataCollator",
    "DefaultDataCollator",
    "DataCollatorWithPadding",
    "DataCollatorForLanguageModeling",
    "DataCollatorForSeq2Seq", 
    "DataCollatorForTokenClassification",
    "get_data_collator",
    
    # Evaluation metrics
    "EvaluationMetric",
    "AccuracyMetric",
    "F1Metric",
    "PrecisionRecallMetric",
    "MatthewsCorrCoefMetric", 
    "PerplexityMetric",
    "BLEUMetric",
    "ROUGEMetric",
    "ExactMatchMetric",
    "SquadMetric",
    "MetricCollection",
    "get_classification_metrics",
    "get_generation_metrics",
    "get_qa_metrics",
    "compute_metric",
    
    # Learning rate schedulers
    "LearningRateScheduler",
    "LinearScheduler",
    "CosineScheduler",
    
    # Model outputs
    "ModelOutput",
    "BaseModelOutput",
    "BaseModelOutputWithPooling",
    "CausalLMOutput",
    "MaskedLMOutput",
    "SequenceClassifierOutput",
    "TokenClassifierOutput",
    "QuestionAnsweringModelOutput",
    
    # Utilities
    "logging",
    "set_seed",
    "get_device",
    "enable_grad",
    "no_grad",
    
    # JAX integration (if available)
    "jax_available",
    "jax_to_tensor",
    "tensor_to_jax",
    "batch_jax_to_tensor",
    "batch_tensor_to_jax",
    "ensure_jax_array",
    "JAXTensorWrapper",
    "create_jax_dataloader",
    "jax_model_apply",
    "jax_grad_fn",
    "jax_vmap_fn", 
    "jax_jit_fn",
    "create_jax_pipeline",
    "setup_jax_config",
    
    # Advanced JAX integration (if available)
    "JAXDevice",
    "JAXGradientTransforms",
    "JAXCompilation",
    "JAXOptimization",
    "JAXModelWrapper",
    "JAXTrainingLoop",
    "JAXCheckpointing",
    "JAXEcosystem",
    "jax_device_manager",
    "jax_gradient_transforms",
    "jax_compilation",
    "jax_optimization",
    "jax_checkpointing",
    "jax_ecosystem",
    
    # Advanced features (2025-07-15)
    # PyTorch module compatibility
    "TorchModuleWrapper",
    "TorchModuleBert",
    "TorchModuleGPT2",
    "TorchLoss",
    "create_torch_training_loop",
    "wrap_bert_model",
    "wrap_gpt2_model",
    "wrap_model",
    
    # Batch encoding utilities
    "AdvancedBatchEncoding",
    "BatchTokenizer",
    "create_batch_encoding",
    
    # Generation utilities
    "GenerationConfig",
    "GenerationMixin",
    "batch_generate",
    
    # Advanced numpy utilities
    "AdvancedArray",
    "StructuredArrays",
    "BroadcastingUtils",
    "memory_efficient_operation",
    "advanced_concatenate",
    "advanced_stack",
    
    # Caching system
    "ModelCache",
    "TokenizerCache",
    "ResultCache",
    "get_model_cache",
    "get_tokenizer_cache",
    "get_result_cache",
    "memoize",
    "cache_model",
    "get_cached_model",
    "cache_tokenizer",
    "get_cached_tokenizer",
    "clear_all_caches",
    "get_cache_stats",
    
    # Debugging tools
    "TensorTracker",
    "MemoryProfiler",
    "PerformanceProfiler",
    "TensorVisualizer",
    "tensor_tracker",
    "memory_profiler",
    "performance_profiler",
    "tensor_visualizer",
    "debug_tensor",
    "profile_operation",
    "track_tensor_creation",
    "get_debug_report",
    "export_debug_report",
    "start_memory_monitoring",
    "stop_memory_monitoring",
    "clear_debug_data",
    
    # Performance optimization
    "BatchProcessor",
    "ObjectPool",
    "LazyLoader",
    "MemoryPool",
    "CallBatcher",
    "PrewarmingCache",
    "PerformanceMonitor",
    "batched",
    "lazy_init",
    "pooled",
    "monitored",
    "batch_process",
    "get_from_pool",
    "return_to_pool",
    "prewarm_cache",
    "get_performance_stats",
    "optimize_memory",
    "cleanup_performance_resources",
    
    # Jupyter support (if available)
    "check_jupyter_availability",
    "is_notebook_environment",
    "TensorDisplay",
    "ModelDisplay",
    "TrainingDisplay",
    "InteractiveVisualization",
    "ProgressBar",
    "display_tensor",
    "display_model",
    "create_training_display",
    "create_progress_bar",
    "tensor_heatmap",
    "attention_visualization",
    "setup_jupyter_environment",
    
    # Async utilities
    "AsyncAutoModel",
    "AsyncAutoTokenizer",
    "AsyncPipeline",
    "AsyncModelWrapper",
    "AsyncTokenizerWrapper",
    "AsyncBatchProcessor",
    "AsyncResourceManager",
    "AsyncStreamProcessor",
    "async_download_file",
    "async_cached_path",
    "async_pipeline",
    "wrap_model_async",
    "wrap_tokenizer_async",
    "safe_async_call",
    "async_wrapper",
    "get_async_executor",
    "cleanup_async_executor",
    "AsyncOperationError",
    
    # MLOps integration (if available)
    "ExperimentConfig",
    "ModelMetadata",
    "ExperimentTracker",
    "MLflowTracker",
    "WandBTracker",
    "TensorBoardTracker",
    "CompositeTracker",
    "ModelRegistry",
    "ExperimentManager",
    "get_experiment_manager",
    "start_experiment",
    "log_metrics",
    "log_parameters",
    "save_model",
    "load_model",
    "end_experiment",
    
    # Data Science Tools integration (if available)
    "DataScienceConfig",
    "PandasIntegration",
    "SklearnIntegration",
    "VisualizationTools",
    "DataScienceWorkflow",
    "get_data_science_workflow",
    "to_dataframe",
    "from_dataframe",
    "plot_tensor",
    "analyze_data",
    "create_sklearn_pipeline",
    
    # Advanced PyTorch integration (if available)
    "GradientStats",
    "MixedPrecisionConfig",
    "GradientFlowAnalyzer",
    "MixedPrecisionTrainer",
    "AdvancedOptimizer",
    "MemoryOptimizedTrainer",
    "AdvancedTrainingManager",
    "create_mixed_precision_trainer",
    "analyze_gradient_flow",
    "create_advanced_training_manager",
    
    # Distributed Training (if available)
    "DistributedConfig",
    "DistributedManager",
    "DistributedTrainer",
    "DDPModelWrapper",
    "HorovodModelWrapper",
    "FaultToleranceManager",
    "ElasticTrainingManager",
    "DistributedBackend",
    "get_free_port",
    "setup_logging_for_distributed",
    "all_gather_object",
    "broadcast_object",
    "barrier",
    "reduce_tensor",
    "get_default_distributed_manager",
    "is_distributed",
    "get_rank",
    "get_world_size",
    "is_main_process",
    
    # Model Serving (if available)
    "ModelConfig",
    "ModelInstance",
    "InferenceRequest",
    "InferenceResponse",
    "HealthResponse",
    "ABTestConfig",
    "ServingManager",
    "ModelVersionManager",
    "LoadBalancer",
    "BatchProcessor",
    "ABTestManager",
    "MetricsCollector",
    "ModelStatus",
    "LoadBalancingStrategy",
    "create_app",
    "serve",
    "serve_model",
    
    # Custom Extensions System (2025-07-16)
    "CallbackManager",
    "HookRegistry",
    "EventBus",
    "PluginManager",
    "CustomLayer",
    "CustomLayerRegistry",
    "CallbackConfig",
    "EventConfig",
    "ExtensionConfig",
    "Event",
    "CallbackResult",
    "CallbackError",
    "HookError",
    "EventError",
    "PluginError",
    "PluginInterface",
    "PluginMetadata",
    "LinearLayer",
    "ActivationLayer",
    "LayerProtocol",
    "callback",
    "hook",
    "event_handler",
    "callback_context",
    "hook_context",
    "create_custom_layer",
    "register_layer_type",
    "configure_extensions",
    "callback_manager",
    "hook_registry",
    "event_bus",
    "custom_layer_registry",
    "plugin_manager",
]