---
name: Model Implementation Request
about: Request support for a new model architecture or improvement to existing models
title: "[MODEL] "
labels: ["model-request", "enhancement"]
assignees: ''

---

## Model Information
Provide details about the model you'd like to see implemented or improved.

### Model Name
**Model**: [e.g. Llama-3, GPT-4, Claude-3, Mistral-7B, Phi-3, Gemma]
**Architecture**: [e.g. Transformer, Mamba, RetNet, RWKV, Mamba2]
**Model Size**: [e.g. 7B, 13B, 70B, 8x7B MoE parameters]

### Official Information
- **Paper**: [link to arXiv/paper if available]
- **Official Repository**: [link to official implementation]
- **HuggingFace Model**: [link to HF model hub if available]
- **License**: [e.g. Apache 2.0, MIT, Commercial, Custom]

### Model Details
**Architecture Type**:
- [ ] Decoder-only (GPT-style)
- [ ] Encoder-only (BERT-style) 
- [ ] Encoder-Decoder (T5-style)
- [ ] Mixture of Experts (MoE)
- [ ] State Space Model (Mamba-style)
- [ ] Retrieval Augmented (RAG)
- [ ] Multimodal (Vision + Language)
- [ ] Other: [specify]

**Key Features**:
- [ ] Grouped Query Attention (GQA)
- [ ] Sliding Window Attention
- [ ] Rotary Position Embedding (RoPE)
- [ ] SwiGLU Activation
- [ ] RMSNorm
- [ ] Custom attention mechanisms
- [ ] Novel training techniques
- [ ] Specialized tokenization
- [ ] Other: [specify]

## Use Case Justification
Why is this model important to implement?

### Community Demand
- [ ] **High community interest** (popular model with many users)
- [ ] **Production usage** (needed for business/production deployments)
- [ ] **Research importance** (significant advancement in the field)
- [ ] **Benchmark performance** (SOTA results on important benchmarks)
- [ ] **Unique capabilities** (enables new types of applications)

### Performance Characteristics
What makes this model special?
- [ ] **Better accuracy** than existing models
- [ ] **Higher efficiency** (faster inference, less memory)
- [ ] **Longer context** (supports longer sequences)
- [ ] **Multimodal capabilities** (text + vision/audio)
- [ ] **Better instruction following**
- [ ] **Domain specialization** (code, math, reasoning, etc.)
- [ ] **Novel architecture** (new paradigm worth supporting)

### Specific Benefits
- **Accuracy improvements**: [describe vs existing models]
- **Efficiency gains**: [memory/speed benefits]
- **New capabilities**: [what can this model do that others can't]
- **Production value**: [why businesses would want this]

## Implementation Priority
How urgent/important is this implementation?
- [ ] **Critical** (blocking major use cases)
- [ ] **High** (significant user demand)
- [ ] **Medium** (nice to have, good for completeness)
- [ ] **Low** (interesting but not urgent)

### Timeline
When would you need this implementation?
- [ ] ASAP (within 1 month)
- [ ] Soon (within 3 months)  
- [ ] Eventually (within 6 months)
- [ ] No specific timeline

## Implementation Details
Technical information to help with implementation.

### Reference Implementations
Provide links to existing implementations:
- **Official**: [link to official code]
- **HuggingFace**: [link to HF implementation]
- **PyTorch**: [link to PyTorch implementation] 
- **Other**: [other reference implementations]

### Model Weights
Are pre-trained weights available?
- [ ] **Official weights** available at: [link/location]
- [ ] **Community weights** available at: [link/location]
- [ ] **No weights** (implementation only)
- [ ] **Training from scratch** required

### Technical Challenges
Are there any known implementation challenges?
- [ ] **Novel attention mechanisms** requiring custom kernels
- [ ] **Large model size** requiring memory optimization
- [ ] **Complex tokenization** not supported by standard tokenizers
- [ ] **Custom training procedures** needed
- [ ] **Hardware requirements** (specific GPU/TPU needs)
- [ ] **License restrictions** limiting distribution
- [ ] **Dependency requirements** (special libraries needed)

### Compatibility Requirements
What compatibility is needed?
- [ ] **Drop-in replacement** for existing models
- [ ] **Same API** as other transformer models
- [ ] **Custom API** may be acceptable
- [ ] **Quantization support** (INT8, INT4, etc.)
- [ ] **Hardware acceleration** (CUDA, ROCm, Metal)
- [ ] **Mobile deployment** compatibility

## Implementation Contribution
Are you willing to help with implementation?

### Your Contribution
- [ ] **Full implementation** (I can implement the complete model)
- [ ] **Partial implementation** (I can help with specific parts)
- [ ] **Testing/validation** (I can test and validate the implementation)
- [ ] **Documentation** (I can write docs and examples)
- [ ] **Requirements only** (I can provide detailed specs but not code)
- [ ] **No contribution** (requesting implementation only)

### Your Expertise
What's your background with this model?
- [ ] **Model author/contributor** (I worked on the original model)
- [ ] **Production user** (I use this model in production)
- [ ] **Research expert** (I have deep understanding of the architecture)
- [ ] **Framework developer** (I have experience implementing models)
- [ ] **End user** (I want to use this model for my applications)

## Additional Context
Provide any other relevant information:

### Related Models
Are there similar models already implemented in TrustformeRS?
- **Similar architecture**: [existing models with similar design]
- **Code reuse potential**: [which existing components could be reused]

### Benchmarks
What benchmarks should this model be evaluated on?
- [ ] **Language modeling** (perplexity, next token prediction)
- [ ] **Downstream tasks** (GLUE, SuperGLUE, etc.)
- [ ] **Code generation** (HumanEval, MBPP)
- [ ] **Math reasoning** (GSM8K, MATH)
- [ ] **Instruction following** (Alpaca, Vicuna)
- [ ] **Safety/alignment** (various safety benchmarks)
- [ ] **Performance** (inference speed, memory usage)

### Resources
Any additional resources that would help:
- Links to model cards or technical reports
- Performance comparisons with other models
- Known issues or limitations
- Community discussions about the model

## Checklist
- [ ] I have searched existing issues to avoid duplicates
- [ ] I have provided links to official model information
- [ ] I have explained why this model is important to implement
- [ ] I have identified potential implementation challenges
- [ ] I have specified what help I can provide with implementation