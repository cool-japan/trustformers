use crate::codellama::config::CodeLlamaConfig;
use crate::codellama::model::CodeLlamaForCausalLM;
use trustformers_core::errors::Result;
use trustformers_core::tensor::Tensor;

/// Output of a CodeLlama causal LM forward pass
pub struct CodeLMOutput {
    /// Logits of shape `[seq_len, vocab_size]`
    pub logits: Tensor,
}

/// Code completion wrapper around CodeLlama
pub struct CodeLlamaCompletion {
    inner: CodeLlamaForCausalLM,
}

impl CodeLlamaCompletion {
    pub fn new(config: CodeLlamaConfig) -> Result<Self> {
        let inner = CodeLlamaForCausalLM::new(config)?;
        Ok(Self { inner })
    }

    pub fn config(&self) -> &CodeLlamaConfig {
        self.inner.config()
    }

    /// Run a code completion forward pass
    pub fn complete(&self, input_ids: Vec<u32>) -> Result<CodeLMOutput> {
        let logits = self.inner.forward(input_ids)?;
        Ok(CodeLMOutput { logits })
    }
}

/// Fill-in-the-Middle (FIM / infilling) wrapper
///
/// CodeLlama infilling uses a special `<FILL_ME>` sentinel token to request
/// that the model complete a span between provided prefix and suffix context.
pub struct CodeLlamaInfilling {
    inner: CodeLlamaForCausalLM,
    /// Whether the underlying model config declares infilling support
    pub infilling_enabled: bool,
}

impl CodeLlamaInfilling {
    pub fn new(config: CodeLlamaConfig) -> Result<Self> {
        let infilling_enabled = config.infilling;
        let inner = CodeLlamaForCausalLM::new(config)?;
        Ok(Self {
            inner,
            infilling_enabled,
        })
    }

    pub fn config(&self) -> &CodeLlamaConfig {
        self.inner.config()
    }

    /// Fill-in-the-Middle forward pass
    ///
    /// In production this method would prepend/append the FIM sentinel tokens
    /// and run structured generation.  Here we forward the input IDs directly.
    pub fn infill(
        &self,
        _prefix_ids: &[u32],
        _suffix_ids: &[u32],
        merged_ids: Vec<u32>,
    ) -> Result<CodeLMOutput> {
        let logits = self.inner.forward(merged_ids)?;
        Ok(CodeLMOutput { logits })
    }
}

/// Repository-level code understanding task
pub struct CodeLlamaRepoLevel {
    inner: CodeLlamaForCausalLM,
    /// Maximum tokens available for repository context
    pub repo_context_limit: usize,
}

impl CodeLlamaRepoLevel {
    pub fn new(config: CodeLlamaConfig) -> Result<Self> {
        let repo_context_limit = config.effective_max_context();
        let inner = CodeLlamaForCausalLM::new(config)?;
        Ok(Self {
            inner,
            repo_context_limit,
        })
    }

    pub fn config(&self) -> &CodeLlamaConfig {
        self.inner.config()
    }

    /// Run a repository-level context forward pass
    pub fn forward_with_repo_context(&self, input_ids: Vec<u32>) -> Result<CodeLMOutput> {
        let logits = self.inner.forward(input_ids)?;
        Ok(CodeLMOutput { logits })
    }
}
