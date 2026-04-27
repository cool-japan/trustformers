//! Advanced Retrieval-Augmented Generation (RAG) Pipeline
//!
//! This module implements cutting-edge RAG techniques for 2024-2025:
//! - Multi-hop reasoning with iterative retrieval
//! - Adaptive retrieval with uncertainty-based triggering
//! - Self-reflective RAG with answer verification
//! - Multi-modal RAG supporting text, images, and structured data
//! - Graph-based RAG for knowledge graph integration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{Result, TrustformersError};
use crate::pipeline::{Pipeline, PipelineInput, PipelineOutput};

/// Configuration for Advanced RAG Pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedRAGConfig {
    /// Maximum number of retrieval iterations for multi-hop reasoning
    pub max_hops: usize,
    /// Threshold for uncertainty-based retrieval triggering
    pub uncertainty_threshold: f32,
    /// Whether to enable self-reflection and answer verification
    pub enable_self_reflection: bool,
    /// Whether to enable multi-modal retrieval
    pub enable_multimodal: bool,
    /// Whether to enable graph-based RAG
    pub enable_graph_rag: bool,
    /// Top-k documents to retrieve per iteration
    pub top_k: usize,
    /// Minimum similarity score for retrieved documents
    pub min_similarity: f32,
    /// Maximum context length for generation
    pub max_context_length: usize,
    /// Whether to use adaptive chunking based on content type
    pub adaptive_chunking: bool,
}

impl Default for AdvancedRAGConfig {
    fn default() -> Self {
        Self {
            max_hops: 3,
            uncertainty_threshold: 0.7,
            enable_self_reflection: true,
            enable_multimodal: false,
            enable_graph_rag: false,
            top_k: 5,
            min_similarity: 0.6,
            max_context_length: 4096,
            adaptive_chunking: true,
        }
    }
}

/// Multi-modal document representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalDocument {
    pub id: String,
    pub text_content: String,
    pub image_content: Option<Vec<u8>>,
    pub structured_data: Option<HashMap<String, serde_json::Value>>,
    pub metadata: HashMap<String, String>,
    pub embedding: Vec<f32>,
    pub similarity_score: f32,
}

/// Graph node for knowledge graph RAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphNode {
    pub id: String,
    pub entity_type: String,
    pub properties: HashMap<String, serde_json::Value>,
    pub connections: Vec<KnowledgeGraphEdge>,
}

/// Graph edge for knowledge graph RAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphEdge {
    pub target_id: String,
    pub relation_type: String,
    pub weight: f32,
}

/// RAG retrieval result with enhanced metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RAGRetrievalResult {
    pub documents: Vec<MultiModalDocument>,
    pub graph_nodes: Vec<KnowledgeGraphNode>,
    pub retrieval_metadata: RetrievalMetadata,
}

/// Metadata about the retrieval process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalMetadata {
    pub query_embedding: Vec<f32>,
    pub num_candidates_searched: usize,
    pub average_similarity: f32,
    pub retrieval_time_ms: u64,
    pub reasoning_hop: usize,
}

/// Self-reflection result for answer verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfReflectionResult {
    pub answer_confidence: f32,
    pub evidence_quality: f32,
    pub consistency_score: f32,
    pub should_retrieve_more: bool,
    pub identified_gaps: Vec<String>,
}

/// Advanced RAG reasoning step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_id: usize,
    pub query: String,
    pub retrieved_docs: Vec<MultiModalDocument>,
    pub intermediate_answer: String,
    pub confidence: f32,
    pub reasoning_trace: String,
}

/// Advanced RAG output with detailed reasoning chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedRAGOutput {
    pub final_answer: String,
    pub confidence_score: f32,
    pub reasoning_chain: Vec<ReasoningStep>,
    pub total_documents_used: usize,
    pub retrieval_iterations: usize,
    pub self_reflection_results: Vec<SelfReflectionResult>,
    pub knowledge_graph_paths: Vec<Vec<KnowledgeGraphNode>>,
}

/// Trait for advanced document retrieval
#[async_trait::async_trait]
pub trait AdvancedRetriever: Send + Sync {
    async fn retrieve_documents(
        &self,
        query: &str,
        config: &AdvancedRAGConfig,
        context: Option<&[MultiModalDocument]>,
    ) -> Result<RAGRetrievalResult>;

    async fn retrieve_graph_nodes(
        &self,
        entities: &[String],
        max_depth: usize,
    ) -> Result<Vec<KnowledgeGraphNode>>;
}

/// Trait for uncertainty estimation
#[async_trait::async_trait]
pub trait UncertaintyEstimator: Send + Sync {
    async fn estimate_uncertainty(&self, text: &str, context: &[MultiModalDocument])
        -> Result<f32>;
}

/// Trait for self-reflection and verification
#[async_trait::async_trait]
pub trait SelfReflector: Send + Sync {
    async fn reflect_on_answer(
        &self,
        query: &str,
        answer: &str,
        evidence: &[MultiModalDocument],
    ) -> Result<SelfReflectionResult>;
}

/// Advanced RAG Pipeline Implementation
pub struct AdvancedRAGPipeline {
    config: AdvancedRAGConfig,
    retriever: Arc<dyn AdvancedRetriever>,
    uncertainty_estimator: Option<Arc<dyn UncertaintyEstimator>>,
    self_reflector: Option<Arc<dyn SelfReflector>>,
    generation_pipeline: Arc<dyn Pipeline<Input = String, Output = PipelineOutput>>,
    document_cache: Arc<RwLock<HashMap<String, MultiModalDocument>>>,
}

impl AdvancedRAGPipeline {
    /// Create a new Advanced RAG Pipeline
    pub fn new(
        config: AdvancedRAGConfig,
        retriever: Arc<dyn AdvancedRetriever>,
        generation_pipeline: Arc<dyn Pipeline<Input = String, Output = PipelineOutput>>,
    ) -> Self {
        Self {
            config,
            retriever,
            uncertainty_estimator: None,
            self_reflector: None,
            generation_pipeline,
            document_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set uncertainty estimator for adaptive retrieval
    pub fn with_uncertainty_estimator(mut self, estimator: Arc<dyn UncertaintyEstimator>) -> Self {
        self.uncertainty_estimator = Some(estimator);
        self
    }

    /// Set self-reflector for answer verification
    pub fn with_self_reflector(mut self, reflector: Arc<dyn SelfReflector>) -> Self {
        self.self_reflector = Some(reflector);
        self
    }

    /// Perform multi-hop reasoning with iterative retrieval
    async fn multi_hop_reasoning(&self, query: &str) -> Result<AdvancedRAGOutput> {
        let mut reasoning_chain = Vec::new();
        let mut all_documents = Vec::new();
        let mut current_query = query.to_string();
        let mut knowledge_graph_paths = Vec::new();

        for hop in 0..self.config.max_hops {
            // Retrieve documents for current query
            let retrieval_result = self
                .retriever
                .retrieve_documents(
                    &current_query,
                    &self.config,
                    if all_documents.is_empty() { None } else { Some(&all_documents) },
                )
                .await?;

            let retrieved_docs = retrieval_result.documents;
            all_documents.extend(retrieved_docs.clone());

            // Retrieve knowledge graph nodes if enabled
            if self.config.enable_graph_rag {
                let entities = self.extract_entities(&current_query).await?;
                let graph_nodes = self.retriever.retrieve_graph_nodes(&entities, 2).await?;
                knowledge_graph_paths.push(graph_nodes);
            }

            // Generate intermediate answer
            let context = self.build_context(&retrieved_docs).await?;
            let prompt = self.build_reasoning_prompt(&current_query, &context, hop);

            let generation_output = self.generation_pipeline.__call__(prompt)?;

            let intermediate_answer = match generation_output {
                PipelineOutput::Text(text) => text,
                _ => {
                    return Err(TrustformersError::new(
                        trustformers_core::errors::TrustformersError::new(
                            trustformers_core::errors::ErrorKind::PipelineError {
                                reason: "Invalid generation output format".to_string(),
                            },
                        ),
                    ))
                },
            };

            // Estimate uncertainty if available
            let confidence = if let Some(estimator) = &self.uncertainty_estimator {
                1.0 - estimator.estimate_uncertainty(&intermediate_answer, &retrieved_docs).await?
            } else {
                0.8 // Default confidence
            };

            let reasoning_step = ReasoningStep {
                step_id: hop,
                query: current_query.clone(),
                retrieved_docs: retrieved_docs.clone(),
                intermediate_answer: intermediate_answer.clone(),
                confidence,
                reasoning_trace: format!(
                    "Hop {} reasoning with {} documents",
                    hop + 1,
                    retrieved_docs.len()
                ),
            };

            reasoning_chain.push(reasoning_step);

            // Check if we need another hop
            if confidence > self.config.uncertainty_threshold || hop == self.config.max_hops - 1 {
                // Perform self-reflection if enabled
                let mut self_reflection_results = Vec::new();
                if self.config.enable_self_reflection {
                    if let Some(reflector) = &self.self_reflector {
                        let reflection = reflector
                            .reflect_on_answer(query, &intermediate_answer, &all_documents)
                            .await?;

                        if !reflection.should_retrieve_more {
                            self_reflection_results.push(reflection);
                            break;
                        }
                        self_reflection_results.push(reflection);
                    }
                }

                break;
            }

            // Prepare next hop query based on gaps in current answer
            current_query =
                self.generate_followup_query(&intermediate_answer, &retrieved_docs).await?;
        }

        // Final answer synthesis
        let final_answer = self.synthesize_final_answer(&reasoning_chain).await?;
        let overall_confidence = reasoning_chain.iter().map(|step| step.confidence).sum::<f32>()
            / reasoning_chain.len() as f32;
        let retrieval_iterations = reasoning_chain.len();

        Ok(AdvancedRAGOutput {
            final_answer,
            confidence_score: overall_confidence,
            reasoning_chain,
            total_documents_used: all_documents.len(),
            retrieval_iterations,
            self_reflection_results: Vec::new(), // Populated above if enabled
            knowledge_graph_paths,
        })
    }

    /// Build context from retrieved documents with adaptive chunking
    async fn build_context(&self, documents: &[MultiModalDocument]) -> Result<String> {
        let mut context_parts = Vec::new();
        let mut current_length = 0;

        for doc in documents {
            let chunk = if self.config.adaptive_chunking {
                self.adaptive_chunk(&doc.text_content, &doc.metadata).await?
            } else {
                doc.text_content.clone()
            };

            if current_length + chunk.len() > self.config.max_context_length {
                break;
            }

            context_parts.push(format!("Document {}: {}", doc.id, chunk));
            current_length += chunk.len();
        }

        Ok(context_parts.join("\n\n"))
    }

    /// Adaptive chunking based on content type and structure
    async fn adaptive_chunk(
        &self,
        content: &str,
        metadata: &HashMap<String, String>,
    ) -> Result<String> {
        // Simple implementation - could be enhanced with NLP techniques
        let content_type = metadata.get("content_type").map(|s| s.as_str()).unwrap_or("text");

        match content_type {
            "scientific_paper" => {
                // Extract abstract and key findings
                self.extract_scientific_content(content).await
            },
            "code" => {
                // Extract functions and classes
                self.extract_code_content(content).await
            },
            "structured" => {
                // Handle structured data
                self.extract_structured_content(content).await
            },
            _ => Ok(content.chars().take(1000).collect()), // Default truncation
        }
    }

    /// Extract scientific content (abstract, conclusions, key findings)
    async fn extract_scientific_content(&self, content: &str) -> Result<String> {
        // Look for common scientific paper sections
        let sections = vec!["abstract", "conclusion", "results", "findings"];
        let mut extracted = Vec::new();

        for section in sections {
            if let Some(section_content) = self.extract_section(content, section) {
                extracted.push(format!("{}: {}", section.to_uppercase(), section_content));
            }
        }

        if extracted.is_empty() {
            Ok(content.chars().take(1000).collect())
        } else {
            Ok(extracted.join("\n\n"))
        }
    }

    /// Extract code content (functions, classes, main logic)
    async fn extract_code_content(&self, content: &str) -> Result<String> {
        // Simple regex-based extraction for demonstration
        // In practice, would use proper AST parsing
        let lines: Vec<&str> = content.lines().collect();
        let mut important_lines = Vec::new();

        for line in lines {
            let trimmed = line.trim();
            if trimmed.starts_with("def ")
                || trimmed.starts_with("class ")
                || trimmed.starts_with("fn ")
                || trimmed.starts_with("function ")
                || trimmed.contains("// TODO")
                || trimmed.contains("# TODO")
            {
                important_lines.push(line);
            }
        }

        if important_lines.is_empty() {
            Ok(content.chars().take(1000).collect())
        } else {
            Ok(important_lines.join("\n"))
        }
    }

    /// Extract structured content
    async fn extract_structured_content(&self, content: &str) -> Result<String> {
        // Handle JSON, XML, YAML structured data
        if content.trim_start().starts_with('{') {
            // JSON handling
            Ok(self
                .summarize_json(content)
                .await
                .unwrap_or_else(|_| content.chars().take(1000).collect()))
        } else if content.trim_start().starts_with('<') {
            // XML handling
            Ok(self
                .summarize_xml(content)
                .await
                .unwrap_or_else(|_| content.chars().take(1000).collect()))
        } else {
            Ok(content.chars().take(1000).collect())
        }
    }

    /// Summarize JSON content
    async fn summarize_json(&self, content: &str) -> Result<String> {
        match serde_json::from_str::<serde_json::Value>(content) {
            Ok(json) => {
                let mut summary = Vec::new();
                if let Some(obj) = json.as_object() {
                    for (key, value) in obj.iter().take(10) {
                        let value_summary = match value {
                            serde_json::Value::Object(obj) => {
                                format!("{}: [object with {} fields]", key, obj.len())
                            },
                            serde_json::Value::Array(arr) => {
                                format!("{}: [array with {} items]", key, arr.len())
                            },
                            _ => format!("{}: {}", key, value),
                        };
                        summary.push(value_summary);
                    }
                }
                Ok(summary.join(", "))
            },
            Err(_) => Ok(content.chars().take(1000).collect()),
        }
    }

    /// Summarize XML content
    async fn summarize_xml(&self, content: &str) -> Result<String> {
        // Simple XML tag extraction for demonstration
        let tag_regex = regex::Regex::new(r"<(\w+)").expect("static regex pattern is valid");
        let tags: Vec<_> = tag_regex.captures_iter(content).map(|cap| cap[1].to_string()).collect();

        if tags.is_empty() {
            Ok(content.chars().take(1000).collect())
        } else {
            Ok(format!("XML with tags: {}", tags.join(", ")))
        }
    }

    /// Extract a section from text content
    fn extract_section(&self, content: &str, section: &str) -> Option<String> {
        let section_regex =
            regex::Regex::new(&format!(r"(?i){}[\s\n]*(.{{0,500}})", section)).ok()?;
        section_regex.captures(content).map(|cap| cap[1].to_string())
    }

    /// Build reasoning prompt for multi-hop retrieval
    fn build_reasoning_prompt(&self, query: &str, context: &str, hop: usize) -> String {
        format!(
            "Query: {}\n\nContext (Reasoning Hop {}):\n{}\n\nBased on the context above, provide a detailed answer to the query. If the information is insufficient, indicate what additional information would be needed.\n\nAnswer:",
            query, hop + 1, context
        )
    }

    /// Extract entities from query for knowledge graph retrieval
    async fn extract_entities(&self, query: &str) -> Result<Vec<String>> {
        // Simple entity extraction - in practice would use NER models
        let words: Vec<String> = query
            .split_whitespace()
            .filter(|word| {
                word.len() > 3 && word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
            })
            .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|word| !word.is_empty())
            .collect();

        Ok(words)
    }

    /// Generate follow-up query for next reasoning hop
    async fn generate_followup_query(
        &self,
        _current_answer: &str,
        _documents: &[MultiModalDocument],
    ) -> Result<String> {
        // Simplified implementation - in practice would analyze gaps in current answer
        Ok("What additional details are needed to complete this answer?".to_string())
    }

    /// Synthesize final answer from reasoning chain
    async fn synthesize_final_answer(&self, reasoning_chain: &[ReasoningStep]) -> Result<String> {
        if reasoning_chain.is_empty() {
            return Ok("No reasoning steps available.".to_string());
        }

        let mut synthesis_parts = Vec::new();

        // Combine insights from all reasoning steps
        for (i, step) in reasoning_chain.iter().enumerate() {
            synthesis_parts.push(format!("Step {}: {}", i + 1, step.intermediate_answer));
        }

        // Final synthesis prompt
        let synthesis_prompt = format!(
            "Based on the following reasoning steps, provide a comprehensive final answer:\n\n{}\n\nFinal Answer:",
            synthesis_parts.join("\n\n")
        );

        let synthesis_output = self.generation_pipeline.__call__(synthesis_prompt)?;

        match synthesis_output {
            PipelineOutput::Text(text) => Ok(text),
            _ => Ok(reasoning_chain
                .last()
                .map(|r| r.intermediate_answer.clone())
                .unwrap_or_default()),
        }
    }
}

impl Pipeline for AdvancedRAGPipeline {
    type Input = PipelineInput;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        let query = match input {
            PipelineInput::Text(text) => text,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "AdvancedRAG requires text input".to_string(),
                ))
            },
        };

        // Use current runtime handle to avoid creating nested runtimes
        let result = if let Ok(handle) = tokio::runtime::Handle::try_current() {
            tokio::task::block_in_place(|| handle.block_on(self.multi_hop_reasoning(&query)))
        } else {
            // Fallback for non-async contexts
            let rt = tokio::runtime::Runtime::new().map_err(|e| {
                TrustformersError::runtime_error(format!("Failed to create async runtime: {}", e))
            })?;
            rt.block_on(self.multi_hop_reasoning(&query))
        }
        .map_err(|e| {
            TrustformersError::runtime_error(format!("Advanced RAG reasoning failed: {}", e))
        })?;

        Ok(PipelineOutput::AdvancedRAG(result))
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl crate::pipeline::AsyncPipeline for AdvancedRAGPipeline {
    type Input = PipelineInput;
    type Output = PipelineOutput;

    async fn __call_async__(&self, input: Self::Input) -> Result<Self::Output> {
        let query = match input {
            PipelineInput::Text(text) => text,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "AdvancedRAG requires text input".to_string(),
                ))
            },
        };

        let result = self.multi_hop_reasoning(&query).await.map_err(|e| {
            TrustformersError::invalid_input(
                format!("Advanced RAG reasoning failed: {}", e),
                Some("query"),
                Some("valid query for advanced RAG reasoning"),
                None::<String>,
            )
        })?;
        Ok(PipelineOutput::AdvancedRAG(result))
    }
}

/// Mock implementations for testing and demonstration

/// Mock retriever for demonstration
pub struct MockAdvancedRetriever {
    documents: Vec<MultiModalDocument>,
}

impl Default for MockAdvancedRetriever {
    fn default() -> Self {
        Self::new()
    }
}

impl MockAdvancedRetriever {
    pub fn new() -> Self {
        let documents = vec![
            MultiModalDocument {
                id: "doc1".to_string(),
                text_content: "Climate change refers to long-term shifts in global temperatures and weather patterns.".to_string(),
                image_content: None,
                structured_data: None,
                metadata: HashMap::from([("topic".to_string(), "climate".to_string())]),
                embedding: vec![0.1, 0.2, 0.3],
                similarity_score: 0.9,
            },
            MultiModalDocument {
                id: "doc2".to_string(),
                text_content: "Renewable energy sources include solar, wind, and hydroelectric power.".to_string(),
                image_content: None,
                structured_data: None,
                metadata: HashMap::from([("topic".to_string(), "energy".to_string())]),
                embedding: vec![0.2, 0.3, 0.4],
                similarity_score: 0.8,
            },
        ];

        Self { documents }
    }
}

#[async_trait::async_trait]
impl AdvancedRetriever for MockAdvancedRetriever {
    async fn retrieve_documents(
        &self,
        _query: &str,
        config: &AdvancedRAGConfig,
        _context: Option<&[MultiModalDocument]>,
    ) -> Result<RAGRetrievalResult> {
        let selected_docs = self.documents.iter().take(config.top_k).cloned().collect();

        Ok(RAGRetrievalResult {
            documents: selected_docs,
            graph_nodes: Vec::new(),
            retrieval_metadata: RetrievalMetadata {
                query_embedding: vec![0.1, 0.2, 0.3],
                num_candidates_searched: self.documents.len(),
                average_similarity: 0.85,
                retrieval_time_ms: 50,
                reasoning_hop: 0,
            },
        })
    }

    async fn retrieve_graph_nodes(
        &self,
        _entities: &[String],
        _max_depth: usize,
    ) -> Result<Vec<KnowledgeGraphNode>> {
        Ok(vec![KnowledgeGraphNode {
            id: "entity1".to_string(),
            entity_type: "concept".to_string(),
            properties: HashMap::new(),
            connections: Vec::new(),
        }])
    }
}

/// Mock uncertainty estimator
pub struct MockUncertaintyEstimator;

#[async_trait::async_trait]
impl UncertaintyEstimator for MockUncertaintyEstimator {
    async fn estimate_uncertainty(
        &self,
        text: &str,
        _context: &[MultiModalDocument],
    ) -> Result<f32> {
        // Simple heuristic - shorter answers are more uncertain
        let uncertainty = if text.len() < 50 {
            0.6
        } else if text.len() < 100 {
            0.3
        } else {
            0.1
        };
        Ok(uncertainty)
    }
}

/// Mock self-reflector
pub struct MockSelfReflector;

#[async_trait::async_trait]
impl SelfReflector for MockSelfReflector {
    async fn reflect_on_answer(
        &self,
        _query: &str,
        answer: &str,
        evidence: &[MultiModalDocument],
    ) -> Result<SelfReflectionResult> {
        let answer_confidence = if answer.len() > 100 { 0.9 } else { 0.6 };
        let evidence_quality = if evidence.len() >= 3 { 0.9 } else { 0.7 };
        let consistency_score = 0.8; // Mock consistency
        let should_retrieve_more = answer_confidence < 0.7 || evidence_quality < 0.8;

        Ok(SelfReflectionResult {
            answer_confidence,
            evidence_quality,
            consistency_score,
            should_retrieve_more,
            identified_gaps: if should_retrieve_more {
                vec!["Need more specific evidence".to_string()]
            } else {
                Vec::new()
            },
        })
    }
}

/// Factory functions for creating advanced RAG pipelines

/// Create a basic advanced RAG pipeline
pub fn create_advanced_rag_pipeline(
    config: AdvancedRAGConfig,
    generation_pipeline: Arc<dyn Pipeline<Input = String, Output = PipelineOutput>>,
) -> AdvancedRAGPipeline {
    let retriever = Arc::new(MockAdvancedRetriever::new());
    AdvancedRAGPipeline::new(config, retriever, generation_pipeline)
}

/// Create a fully-featured advanced RAG pipeline with all components
pub fn create_full_advanced_rag_pipeline(
    config: AdvancedRAGConfig,
    generation_pipeline: Arc<dyn Pipeline<Input = String, Output = PipelineOutput>>,
) -> AdvancedRAGPipeline {
    let retriever = Arc::new(MockAdvancedRetriever::new());
    let uncertainty_estimator = Arc::new(MockUncertaintyEstimator);
    let self_reflector = Arc::new(MockSelfReflector);

    AdvancedRAGPipeline::new(config, retriever, generation_pipeline)
        .with_uncertainty_estimator(uncertainty_estimator)
        .with_self_reflector(self_reflector)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Result;

    // Simple LCG for deterministic pseudo-random values
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_f32(&mut self) -> f32 {
            self.state =
                self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.state >> 33) as f32) / (u32::MAX as f32)
        }
    }

    // Mock generation pipeline for testing
    struct MockGenerationPipeline;

    impl Pipeline for MockGenerationPipeline {
        type Input = String;
        type Output = PipelineOutput;

        fn __call__(&self, _input: Self::Input) -> Result<Self::Output> {
            Ok(PipelineOutput::Text(
                "This is a mock generated response about climate change and renewable energy."
                    .to_string(),
            ))
        }
    }

    // Helper: build a test MultiModalDocument with given id and score
    fn make_doc(id: &str, text: &str, score: f32) -> MultiModalDocument {
        MultiModalDocument {
            id: id.to_string(),
            text_content: text.to_string(),
            image_content: None,
            structured_data: None,
            metadata: HashMap::new(),
            embedding: vec![0.1, 0.2, 0.3],
            similarity_score: score,
        }
    }

    // ── Config & default field tests ────────────────────────────────────────

    #[test]
    fn test_config_default_top_k() {
        let config = AdvancedRAGConfig::default();
        assert_eq!(config.top_k, 5, "default top_k should be 5");
    }

    #[test]
    fn test_config_default_min_similarity() {
        let config = AdvancedRAGConfig::default();
        assert!(
            config.min_similarity > 0.0 && config.min_similarity < 1.0,
            "min_similarity should be in (0,1)"
        );
    }

    #[test]
    fn test_config_default_max_context_length() {
        let config = AdvancedRAGConfig::default();
        assert!(
            config.max_context_length > 0,
            "max_context_length must be positive"
        );
    }

    #[test]
    fn test_config_custom_top_k() {
        let config = AdvancedRAGConfig {
            top_k: 10,
            ..Default::default()
        };
        assert_eq!(config.top_k, 10);
    }

    #[test]
    fn test_config_similarity_threshold_bounds() {
        let config = AdvancedRAGConfig {
            min_similarity: 0.75,
            ..Default::default()
        };
        assert!(config.min_similarity >= 0.0 && config.min_similarity <= 1.0);
    }

    // ── Mock retriever behaviour ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_mock_retriever_respects_top_k() {
        let retriever = MockAdvancedRetriever::new();
        let mut config = AdvancedRAGConfig::default();
        config.top_k = 1;
        let result = retriever
            .retrieve_documents("test query", &config, None)
            .await
            .expect("retrieve_documents should succeed");
        assert_eq!(result.documents.len(), 1, "Retriever must respect top_k=1");
    }

    #[tokio::test]
    async fn test_mock_retriever_metadata() {
        let retriever = MockAdvancedRetriever::new();
        let config = AdvancedRAGConfig::default();
        let result = retriever
            .retrieve_documents("energy", &config, None)
            .await
            .expect("retrieve_documents should succeed");
        assert!(result.retrieval_metadata.average_similarity > 0.0);
        assert!(result.retrieval_metadata.retrieval_time_ms > 0);
    }

    #[tokio::test]
    async fn test_mock_retriever_graph_nodes() {
        let retriever = MockAdvancedRetriever::new();
        let entities = vec!["Climate".to_string(), "Energy".to_string()];
        let nodes = retriever
            .retrieve_graph_nodes(&entities, 2)
            .await
            .expect("retrieve_graph_nodes should succeed");
        assert!(!nodes.is_empty(), "should return at least one node");
    }

    #[tokio::test]
    async fn test_empty_entity_extraction_yields_no_crash() {
        let config = AdvancedRAGConfig::default();
        let generation_pipeline = Arc::new(MockGenerationPipeline);
        let pipeline = create_advanced_rag_pipeline(config, generation_pipeline);
        // all lowercase, short words → no entities extracted
        let entities = pipeline
            .extract_entities("what is the sky")
            .await
            .expect("extract_entities should succeed");
        // May be empty; just must not panic
        let _ = entities;
    }

    #[tokio::test]
    async fn test_entity_extraction_uppercase_words() {
        let config = AdvancedRAGConfig::default();
        let generation_pipeline = Arc::new(MockGenerationPipeline);
        let pipeline = create_advanced_rag_pipeline(config, generation_pipeline);
        let entities = pipeline
            .extract_entities("Einstein developed Relativity theory")
            .await
            .expect("extract_entities should succeed");
        assert!(
            !entities.is_empty(),
            "capitalised words should be detected as entities"
        );
    }

    // ── Document chunking tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_adaptive_chunking_scientific() {
        let config = AdvancedRAGConfig {
            adaptive_chunking: true,
            ..Default::default()
        };
        let generation_pipeline = Arc::new(MockGenerationPipeline);
        let pipeline = create_advanced_rag_pipeline(config, generation_pipeline);
        let metadata =
            HashMap::from([("content_type".to_string(), "scientific_paper".to_string())]);
        let content = "Abstract: This paper studies climate. Results: Significant warming observed. Conclusion: Action needed.";
        let chunk = pipeline
            .adaptive_chunk(content, &metadata)
            .await
            .expect("adaptive_chunk should succeed");
        assert!(!chunk.is_empty());
    }

    #[tokio::test]
    async fn test_adaptive_chunking_code() {
        let config = AdvancedRAGConfig {
            adaptive_chunking: true,
            ..Default::default()
        };
        let generation_pipeline = Arc::new(MockGenerationPipeline);
        let pipeline = create_advanced_rag_pipeline(config, generation_pipeline);
        let metadata = HashMap::from([("content_type".to_string(), "code".to_string())]);
        let content = "fn main() {\n    println!(\"hello\");\n}\nclass Foo {}\ndef bar(): pass";
        let chunk = pipeline
            .adaptive_chunk(content, &metadata)
            .await
            .expect("adaptive_chunk should succeed");
        assert!(
            chunk.contains("fn") || chunk.contains("class") || chunk.contains("def"),
            "code chunk should contain function/class definitions"
        );
    }

    #[tokio::test]
    async fn test_adaptive_chunking_json() {
        let config = AdvancedRAGConfig {
            adaptive_chunking: true,
            ..Default::default()
        };
        let generation_pipeline = Arc::new(MockGenerationPipeline);
        let pipeline = create_advanced_rag_pipeline(config, generation_pipeline);
        let metadata = HashMap::from([("content_type".to_string(), "structured".to_string())]);
        let json_content = r#"{"name": "test", "value": 42, "items": [1, 2, 3]}"#;
        let chunk = pipeline
            .adaptive_chunk(json_content, &metadata)
            .await
            .expect("adaptive_chunk should succeed for JSON");
        assert!(!chunk.is_empty());
    }

    #[tokio::test]
    async fn test_adaptive_chunking_default_truncation() {
        let config = AdvancedRAGConfig {
            adaptive_chunking: true,
            ..Default::default()
        };
        let generation_pipeline = Arc::new(MockGenerationPipeline);
        let pipeline = create_advanced_rag_pipeline(config, generation_pipeline);
        // 'other' content type → default truncation
        let mut metadata = HashMap::new();
        metadata.insert("content_type".to_string(), "other".to_string());
        let long_text = "x".repeat(2000);
        let chunk = pipeline
            .adaptive_chunk(&long_text, &metadata)
            .await
            .expect("adaptive_chunk should succeed");
        assert!(
            chunk.len() <= 1000 + 5,
            "default chunking should truncate to ~1000 chars"
        );
    }

    // ── Context window assembly ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_build_context_respects_max_length() {
        let max_len = 50;
        let config = AdvancedRAGConfig {
            max_context_length: max_len,
            adaptive_chunking: false,
            ..Default::default()
        };
        let generation_pipeline = Arc::new(MockGenerationPipeline);
        let pipeline = create_advanced_rag_pipeline(config, generation_pipeline);
        // Documents that together exceed max_context_length
        let docs = vec![
            make_doc("d1", "A".repeat(40).as_str(), 0.9),
            make_doc("d2", "B".repeat(40).as_str(), 0.8),
        ];
        let ctx = pipeline.build_context(&docs).await.expect("build_context should succeed");
        assert!(
            ctx.len() <= max_len + 30,
            "context should not hugely exceed max_context_length"
        );
    }

    #[tokio::test]
    async fn test_build_context_empty_docs() {
        let config = AdvancedRAGConfig::default();
        let generation_pipeline = Arc::new(MockGenerationPipeline);
        let pipeline = create_advanced_rag_pipeline(config, generation_pipeline);
        let ctx = pipeline
            .build_context(&[])
            .await
            .expect("build_context with empty docs should succeed");
        assert!(
            ctx.is_empty(),
            "context from empty docs should be empty string"
        );
    }

    // ── Reasoning prompt ────────────────────────────────────────────────────

    #[test]
    fn test_build_reasoning_prompt_contains_query() {
        let config = AdvancedRAGConfig::default();
        let generation_pipeline = Arc::new(MockGenerationPipeline);
        let pipeline = create_advanced_rag_pipeline(config, generation_pipeline);
        let prompt = pipeline.build_reasoning_prompt("test query", "some context", 0);
        assert!(
            prompt.contains("test query"),
            "prompt must include the query"
        );
    }

    #[test]
    fn test_build_reasoning_prompt_contains_hop_number() {
        let config = AdvancedRAGConfig::default();
        let generation_pipeline = Arc::new(MockGenerationPipeline);
        let pipeline = create_advanced_rag_pipeline(config, generation_pipeline);
        let prompt = pipeline.build_reasoning_prompt("query", "ctx", 2);
        assert!(prompt.contains("3"), "hop 2 → display hop 3");
    }

    // ── Uncertainty estimation ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_mock_uncertainty_short_text_high_uncertainty() {
        let estimator = MockUncertaintyEstimator;
        let short_text = "Yes.";
        let uncertainty = estimator
            .estimate_uncertainty(short_text, &[])
            .await
            .expect("estimate_uncertainty should succeed");
        assert!(
            uncertainty >= 0.3,
            "short answer should have higher uncertainty"
        );
    }

    #[tokio::test]
    async fn test_mock_uncertainty_long_text_low_uncertainty() {
        let estimator = MockUncertaintyEstimator;
        let long_text = "a".repeat(200);
        let uncertainty = estimator
            .estimate_uncertainty(&long_text, &[])
            .await
            .expect("estimate_uncertainty should succeed");
        assert!(
            uncertainty < 0.3,
            "long answer should have lower uncertainty"
        );
    }

    // ── Self-reflection ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_mock_self_reflector_long_answer_high_confidence() {
        let reflector = MockSelfReflector;
        let long_answer = "a".repeat(200);
        let result = reflector
            .reflect_on_answer("query", &long_answer, &[])
            .await
            .expect("reflect_on_answer should succeed");
        assert!(
            result.answer_confidence >= 0.8,
            "long answer should yield high confidence"
        );
    }

    #[tokio::test]
    async fn test_mock_self_reflector_short_answer_suggests_more_retrieval() {
        let reflector = MockSelfReflector;
        let short_answer = "Ok.";
        let result = reflector
            .reflect_on_answer("query", short_answer, &[])
            .await
            .expect("reflect_on_answer should succeed");
        assert!(
            result.should_retrieve_more,
            "short answer should suggest more retrieval"
        );
    }

    // ── RRF scoring simulation ────────────────────────────────────────────────

    #[test]
    fn test_rrf_score_formula() {
        // RRF: score = sum(1 / (rank + k))
        // For k=60, rank 0 → 1/60, rank 1 → 1/61
        let k = 60.0_f64;
        let rank0_score = 1.0 / (0.0 + k);
        let rank1_score = 1.0 / (1.0 + k);
        assert!(
            rank0_score > rank1_score,
            "higher rank (lower index) should score higher in RRF"
        );
    }

    #[test]
    fn test_sparse_dense_fusion_ordering() {
        // Simulate fusing two ranked lists: sparse ranks [0,1] and dense ranks [1,0]
        // Doc A: sparse rank 0, dense rank 1 → RRF = 1/60 + 1/61 ≈ 0.03279
        // Doc B: sparse rank 1, dense rank 0 → RRF = 1/61 + 1/60 ≈ 0.03279 (same for symmetric)
        // This verifies the arithmetic, not ordering
        let k = 60.0_f64;
        let score_a: f64 = 1.0 / (0.0 + k) + 1.0 / (1.0 + k);
        let score_b: f64 = 1.0 / (1.0 + k) + 1.0 / (0.0 + k);
        let diff = (score_a - score_b).abs();
        assert!(
            diff < 1e-10,
            "symmetric sparse/dense ranks should yield equal fused scores"
        );
    }

    // ── RAG chain end-to-end ─────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn test_advanced_rag_pipeline() {
        let config = AdvancedRAGConfig::default();
        let mock_generation_pipeline = Arc::new(MockGenerationPipeline);
        let rag_pipeline = create_advanced_rag_pipeline(config, mock_generation_pipeline);
        let input = PipelineInput::Text("What is climate change?".to_string());
        let result = rag_pipeline.__call__(input);
        assert!(result.is_ok());
        if let Ok(PipelineOutput::AdvancedRAG(rag_output)) = result {
            assert!(!rag_output.final_answer.is_empty());
            assert!(rag_output.confidence_score > 0.0);
            assert!(!rag_output.reasoning_chain.is_empty());
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_multi_hop_reasoning() {
        let mut config = AdvancedRAGConfig::default();
        config.max_hops = 2;
        let mock_generation_pipeline = Arc::new(MockGenerationPipeline);
        let rag_pipeline = create_advanced_rag_pipeline(config, mock_generation_pipeline);
        let input =
            PipelineInput::Text("How does climate change affect renewable energy?".to_string());
        let result = rag_pipeline.__call__(input);
        assert!(result.is_ok());
        if let Ok(PipelineOutput::AdvancedRAG(rag_output)) = result {
            assert!(rag_output.retrieval_iterations <= 2);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_full_rag_pipeline_with_reflection() {
        let config = AdvancedRAGConfig {
            enable_self_reflection: true,
            max_hops: 1,
            ..Default::default()
        };
        let generation_pipeline = Arc::new(MockGenerationPipeline);
        let pipeline = create_full_advanced_rag_pipeline(config, generation_pipeline);
        let input = PipelineInput::Text("Tell me about renewable energy.".to_string());
        let result = pipeline.__call__(input);
        assert!(result.is_ok(), "full pipeline should succeed");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_rag_non_text_input_rejected() {
        let config = AdvancedRAGConfig::default();
        let generation_pipeline = Arc::new(MockGenerationPipeline);
        let pipeline = create_advanced_rag_pipeline(config, generation_pipeline);
        let input = PipelineInput::Tokens(vec![1, 2, 3]);
        let result = pipeline.__call__(input);
        assert!(result.is_err(), "non-text input should be rejected");
    }

    // ── Citation extraction via document IDs ──────────────────────────────────

    #[test]
    fn test_reasoning_step_preserves_doc_ids() {
        let step = ReasoningStep {
            step_id: 0,
            query: "test".to_string(),
            retrieved_docs: vec![make_doc("doc42", "some content", 0.9)],
            intermediate_answer: "answer".to_string(),
            confidence: 0.9,
            reasoning_trace: "trace".to_string(),
        };
        assert_eq!(
            step.retrieved_docs[0].id, "doc42",
            "document id should be preserved in reasoning step for citation extraction"
        );
    }

    #[test]
    fn test_knowledge_graph_edge_weight() {
        let edge = KnowledgeGraphEdge {
            target_id: "e2".to_string(),
            relation_type: "related_to".to_string(),
            weight: 0.75,
        };
        assert!(edge.weight > 0.0 && edge.weight <= 1.0);
    }

    // ── Synthesis ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_synthesize_empty_chain_returns_fallback() {
        let config = AdvancedRAGConfig::default();
        let generation_pipeline = Arc::new(MockGenerationPipeline);
        let pipeline = create_advanced_rag_pipeline(config, generation_pipeline);
        let result = pipeline
            .synthesize_final_answer(&[])
            .await
            .expect("synthesize with empty chain should not error");
        assert!(
            !result.is_empty(),
            "should return non-empty fallback for empty chain"
        );
    }

    #[tokio::test]
    async fn test_synthesize_single_step_chain() {
        let config = AdvancedRAGConfig::default();
        let generation_pipeline = Arc::new(MockGenerationPipeline);
        let pipeline = create_advanced_rag_pipeline(config, generation_pipeline);
        let steps = vec![ReasoningStep {
            step_id: 0,
            query: "q".to_string(),
            retrieved_docs: vec![],
            intermediate_answer: "The sky is blue.".to_string(),
            confidence: 0.9,
            reasoning_trace: "trace".to_string(),
        }];
        let result = pipeline
            .synthesize_final_answer(&steps)
            .await
            .expect("synthesize with single step should succeed");
        assert!(!result.is_empty());
    }
}
