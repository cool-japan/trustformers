use async_trait::async_trait;
use futures::future::{BoxFuture, FutureExt};
use futures::stream::Stream;
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};
use tokio::task;
use trustformers_core::errors::Result;
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Async version of the Tokenizer trait
#[async_trait]
pub trait AsyncTokenizer: Send + Sync {
    /// Asynchronously encode a single text
    async fn encode_async(&self, text: &str) -> Result<TokenizedInput>;

    /// Asynchronously encode text pairs
    async fn encode_pair_async(&self, text: &str, text2: &str) -> Result<TokenizedInput>;

    /// Asynchronously decode token IDs to text
    async fn decode_async(&self, ids: &[u32]) -> Result<String>;

    /// Asynchronously encode multiple texts in parallel
    async fn encode_batch_async(&self, texts: &[&str]) -> Result<Vec<TokenizedInput>>;

    /// Asynchronously encode text pairs in parallel
    async fn encode_pair_batch_async(
        &self,
        text_pairs: &[(&str, &str)],
    ) -> Result<Vec<TokenizedInput>>;

    /// Stream-based encoding for large datasets
    fn encode_stream<'a>(
        &'a self,
        texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Box<dyn Stream<Item = Result<TokenizedInput>> + Send + Unpin>>>;
}

/// Wrapper that adds async capabilities to any synchronous tokenizer
pub struct AsyncTokenizerWrapper<T> {
    tokenizer: Arc<T>,
    max_concurrent_tasks: usize,
    task_semaphore: Arc<Semaphore>,
}

impl<T> AsyncTokenizerWrapper<T>
where
    T: Tokenizer + Send + Sync + 'static,
{
    /// Create a new async wrapper around a synchronous tokenizer
    pub fn new(tokenizer: T, max_concurrent_tasks: Option<usize>) -> Self {
        let max_tasks = max_concurrent_tasks.unwrap_or(num_cpus::get() * 2);
        Self {
            tokenizer: Arc::new(tokenizer),
            max_concurrent_tasks: max_tasks,
            task_semaphore: Arc::new(Semaphore::new(max_tasks)),
        }
    }

    /// Set the maximum number of concurrent tasks
    pub fn with_max_concurrent_tasks(mut self, max_tasks: usize) -> Self {
        self.max_concurrent_tasks = max_tasks;
        self.task_semaphore = Arc::new(Semaphore::new(max_tasks));
        self
    }

    /// Get the underlying synchronous tokenizer
    pub fn inner(&self) -> &Arc<T> {
        &self.tokenizer
    }
}

#[async_trait]
impl<T> AsyncTokenizer for AsyncTokenizerWrapper<T>
where
    T: Tokenizer + Send + Sync + 'static,
{
    async fn encode_async(&self, text: &str) -> Result<TokenizedInput> {
        let tokenizer = Arc::clone(&self.tokenizer);
        let text = text.to_string();
        let _permit = self.task_semaphore.acquire().await.map_err(|_| {
            trustformers_core::errors::TrustformersError::other(
                anyhow::anyhow!("Failed to acquire semaphore permit").to_string(),
            )
        })?;

        task::spawn_blocking(move || tokenizer.encode(&text)).await.map_err(|e| {
            trustformers_core::errors::TrustformersError::other(
                anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
            )
        })?
    }

    async fn encode_pair_async(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        let tokenizer = Arc::clone(&self.tokenizer);
        let text = text.to_string();
        let text2 = text2.to_string();
        let _permit = self.task_semaphore.acquire().await.map_err(|_| {
            trustformers_core::errors::TrustformersError::other(
                anyhow::anyhow!("Failed to acquire semaphore permit").to_string(),
            )
        })?;

        task::spawn_blocking(move || tokenizer.encode_pair(&text, &text2))
            .await
            .map_err(|e| {
                trustformers_core::errors::TrustformersError::other(
                    anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                )
            })?
    }

    async fn decode_async(&self, ids: &[u32]) -> Result<String> {
        let tokenizer = Arc::clone(&self.tokenizer);
        let ids = ids.to_vec();
        let _permit = self.task_semaphore.acquire().await.map_err(|_| {
            trustformers_core::errors::TrustformersError::other(
                anyhow::anyhow!("Failed to acquire semaphore permit").to_string(),
            )
        })?;

        task::spawn_blocking(move || tokenizer.decode(&ids)).await.map_err(|e| {
            trustformers_core::errors::TrustformersError::other(
                anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
            )
        })?
    }

    async fn encode_batch_async(&self, texts: &[&str]) -> Result<Vec<TokenizedInput>> {
        let mut tasks = Vec::new();

        for text in texts {
            let tokenizer = Arc::clone(&self.tokenizer);
            let text = text.to_string();
            let semaphore = Arc::clone(&self.task_semaphore);

            let task = task::spawn(async move {
                let _permit = semaphore.acquire().await.map_err(|_| {
                    trustformers_core::errors::TrustformersError::other(
                        anyhow::anyhow!("Failed to acquire semaphore permit").to_string(),
                    )
                })?;

                task::spawn_blocking(move || tokenizer.encode(&text)).await.map_err(|e| {
                    trustformers_core::errors::TrustformersError::other(
                        anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                    )
                })?
            });

            tasks.push(task);
        }

        let mut results = Vec::with_capacity(texts.len());
        for task in tasks {
            let result = task.await.map_err(|e| {
                trustformers_core::errors::TrustformersError::other(
                    anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                )
            })??;
            results.push(result);
        }

        Ok(results)
    }

    async fn encode_pair_batch_async(
        &self,
        text_pairs: &[(&str, &str)],
    ) -> Result<Vec<TokenizedInput>> {
        let mut tasks = Vec::new();

        for (text1, text2) in text_pairs {
            let tokenizer = Arc::clone(&self.tokenizer);
            let text1 = text1.to_string();
            let text2 = text2.to_string();
            let semaphore = Arc::clone(&self.task_semaphore);

            let task = task::spawn(async move {
                let _permit = semaphore.acquire().await.map_err(|_| {
                    trustformers_core::errors::TrustformersError::other(
                        anyhow::anyhow!("Failed to acquire semaphore permit").to_string(),
                    )
                })?;

                task::spawn_blocking(move || tokenizer.encode_pair(&text1, &text2))
                    .await
                    .map_err(|e| {
                        trustformers_core::errors::TrustformersError::other(
                            anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                        )
                    })?
            });

            tasks.push(task);
        }

        let mut results = Vec::with_capacity(text_pairs.len());
        for task in tasks {
            let result = task.await.map_err(|e| {
                trustformers_core::errors::TrustformersError::other(
                    anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                )
            })??;
            results.push(result);
        }

        Ok(results)
    }

    fn encode_stream<'a>(
        &'a self,
        texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Box<dyn Stream<Item = Result<TokenizedInput>> + Send + Unpin>>> {
        async move {
            let (tx, rx) = mpsc::unbounded_channel();
            let tokenizer = Arc::clone(&self.tokenizer);
            let semaphore = Arc::clone(&self.task_semaphore);

            // Spawn a task to process all texts
            task::spawn(async move {
                for text in texts {
                    let tokenizer = Arc::clone(&tokenizer);
                    let semaphore = Arc::clone(&semaphore);
                    let tx = tx.clone();

                    task::spawn(async move {
                        let result = async {
                            let _permit = semaphore.acquire().await.map_err(|_| {
                                trustformers_core::errors::TrustformersError::other(
                                    anyhow::anyhow!("Failed to acquire semaphore permit")
                                        .to_string(),
                                )
                            })?;

                            task::spawn_blocking(move || tokenizer.encode(&text)).await.map_err(
                                |e| {
                                    trustformers_core::errors::TrustformersError::other(
                                        anyhow::anyhow!(format!("Task join error: {}", e))
                                            .to_string(),
                                    )
                                },
                            )?
                        }
                        .await;

                        let _ = tx.send(result);
                    });
                }
            });

            let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
            Ok(Box::new(stream)
                as Box<
                    dyn Stream<Item = Result<TokenizedInput>> + Send + Unpin,
                >)
        }
        .boxed()
    }
}

/// Configuration for async tokenization operations
#[derive(Debug, Clone)]
pub struct AsyncTokenizerConfig {
    /// Maximum number of concurrent tokenization tasks
    pub max_concurrent_tasks: usize,

    /// Buffer size for streaming operations
    pub stream_buffer_size: usize,

    /// Timeout for individual tokenization operations (in milliseconds)
    pub task_timeout_ms: Option<u64>,

    /// Enable task cancellation on timeout
    pub enable_cancellation: bool,
}

impl Default for AsyncTokenizerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: num_cpus::get() * 2,
            stream_buffer_size: 1000,
            task_timeout_ms: None,
            enable_cancellation: false,
        }
    }
}

/// Advanced async tokenizer with configurable behavior
pub struct ConfigurableAsyncTokenizer<T> {
    tokenizer: Arc<T>,
    config: AsyncTokenizerConfig,
    task_semaphore: Arc<Semaphore>,
}

impl<T> ConfigurableAsyncTokenizer<T>
where
    T: Tokenizer + Send + Sync + 'static,
{
    /// Create a new configurable async tokenizer
    pub fn new(tokenizer: T, config: AsyncTokenizerConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_tasks));
        Self {
            tokenizer: Arc::new(tokenizer),
            config,
            task_semaphore: semaphore,
        }
    }

    /// Update configuration
    pub fn update_config(&mut self, config: AsyncTokenizerConfig) {
        self.task_semaphore = Arc::new(Semaphore::new(config.max_concurrent_tasks));
        self.config = config;
    }

    /// Get current configuration
    pub fn config(&self) -> &AsyncTokenizerConfig {
        &self.config
    }

    /// Process a large batch with progress reporting
    pub async fn encode_large_batch_with_progress<F>(
        &self,
        texts: &[&str],
        mut progress_callback: F,
    ) -> Result<Vec<TokenizedInput>>
    where
        F: FnMut(usize, usize) + Send + 'static,
    {
        let total = texts.len();
        let mut completed = 0;
        let mut results = Vec::with_capacity(total);

        // Process in chunks to avoid overwhelming the system
        let chunk_size = (self.config.max_concurrent_tasks).max(1);

        for chunk in texts.chunks(chunk_size) {
            let chunk_results = self.encode_batch_async(chunk).await?;
            results.extend(chunk_results);

            completed += chunk.len();
            progress_callback(completed, total);
        }

        Ok(results)
    }
}

#[async_trait]
impl<T> AsyncTokenizer for ConfigurableAsyncTokenizer<T>
where
    T: Tokenizer + Send + Sync + 'static,
{
    async fn encode_async(&self, text: &str) -> Result<TokenizedInput> {
        let tokenizer = Arc::clone(&self.tokenizer);
        let text = text.to_string();
        let _permit = self.task_semaphore.acquire().await.map_err(|_| {
            trustformers_core::errors::TrustformersError::other(
                anyhow::anyhow!("Failed to acquire semaphore permit").to_string(),
            )
        })?;

        let encoding_task = task::spawn_blocking(move || tokenizer.encode(&text));

        if let Some(timeout_ms) = self.config.task_timeout_ms {
            match tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), encoding_task)
                .await
            {
                Ok(result) => result.map_err(|e| {
                    trustformers_core::errors::TrustformersError::other(
                        anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                    )
                })?,
                Err(_) => Err(trustformers_core::errors::TrustformersError::other(
                    anyhow::anyhow!("Tokenization timeout".to_string()).to_string(),
                )),
            }
        } else {
            encoding_task.await.map_err(|e| {
                trustformers_core::errors::TrustformersError::other(
                    anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                )
            })?
        }
    }

    async fn encode_pair_async(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        let tokenizer = Arc::clone(&self.tokenizer);
        let text = text.to_string();
        let text2 = text2.to_string();
        let _permit = self.task_semaphore.acquire().await.map_err(|_| {
            trustformers_core::errors::TrustformersError::other(
                anyhow::anyhow!("Failed to acquire semaphore permit").to_string(),
            )
        })?;

        let encoding_task = task::spawn_blocking(move || tokenizer.encode_pair(&text, &text2));

        if let Some(timeout_ms) = self.config.task_timeout_ms {
            match tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), encoding_task)
                .await
            {
                Ok(result) => result.map_err(|e| {
                    trustformers_core::errors::TrustformersError::other(
                        anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                    )
                })?,
                Err(_) => Err(trustformers_core::errors::TrustformersError::other(
                    anyhow::anyhow!("Tokenization timeout".to_string()).to_string(),
                )),
            }
        } else {
            encoding_task.await.map_err(|e| {
                trustformers_core::errors::TrustformersError::other(
                    anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                )
            })?
        }
    }

    async fn decode_async(&self, ids: &[u32]) -> Result<String> {
        let tokenizer = Arc::clone(&self.tokenizer);
        let ids = ids.to_vec();
        let _permit = self.task_semaphore.acquire().await.map_err(|_| {
            trustformers_core::errors::TrustformersError::other(
                anyhow::anyhow!("Failed to acquire semaphore permit").to_string(),
            )
        })?;

        let decoding_task = task::spawn_blocking(move || tokenizer.decode(&ids));

        if let Some(timeout_ms) = self.config.task_timeout_ms {
            match tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), decoding_task)
                .await
            {
                Ok(result) => result.map_err(|e| {
                    trustformers_core::errors::TrustformersError::other(
                        anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                    )
                })?,
                Err(_) => Err(trustformers_core::errors::TrustformersError::other(
                    anyhow::anyhow!("Decoding timeout".to_string()).to_string(),
                )),
            }
        } else {
            decoding_task.await.map_err(|e| {
                trustformers_core::errors::TrustformersError::other(
                    anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                )
            })?
        }
    }

    async fn encode_batch_async(&self, texts: &[&str]) -> Result<Vec<TokenizedInput>> {
        let mut tasks = Vec::new();

        for text in texts {
            let tokenizer = Arc::clone(&self.tokenizer);
            let text = text.to_string();
            let semaphore = Arc::clone(&self.task_semaphore);
            let timeout_ms = self.config.task_timeout_ms;

            let task = task::spawn(async move {
                let _permit = semaphore.acquire().await.map_err(|_| {
                    trustformers_core::errors::TrustformersError::other(
                        anyhow::anyhow!("Failed to acquire semaphore permit").to_string(),
                    )
                })?;

                let encoding_task = task::spawn_blocking(move || tokenizer.encode(&text));

                if let Some(timeout_ms) = timeout_ms {
                    match tokio::time::timeout(
                        std::time::Duration::from_millis(timeout_ms),
                        encoding_task,
                    )
                    .await
                    {
                        Ok(result) => result.map_err(|e| {
                            trustformers_core::errors::TrustformersError::other(
                                anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                            )
                        })?,
                        Err(_) => Err(trustformers_core::errors::TrustformersError::other(
                            anyhow::anyhow!("Tokenization timeout").to_string(),
                        )),
                    }
                } else {
                    encoding_task.await.map_err(|e| {
                        trustformers_core::errors::TrustformersError::other(
                            anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                        )
                    })?
                }
            });

            tasks.push(task);
        }

        let mut results = Vec::with_capacity(texts.len());
        for task in tasks {
            let result = task.await.map_err(|e| {
                trustformers_core::errors::TrustformersError::other(
                    anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                )
            })??;
            results.push(result);
        }

        Ok(results)
    }

    async fn encode_pair_batch_async(
        &self,
        text_pairs: &[(&str, &str)],
    ) -> Result<Vec<TokenizedInput>> {
        let mut tasks = Vec::new();

        for (text1, text2) in text_pairs {
            let tokenizer = Arc::clone(&self.tokenizer);
            let text1 = text1.to_string();
            let text2 = text2.to_string();
            let semaphore = Arc::clone(&self.task_semaphore);
            let timeout_ms = self.config.task_timeout_ms;

            let task = task::spawn(async move {
                let _permit = semaphore.acquire().await.map_err(|_| {
                    trustformers_core::errors::TrustformersError::other(
                        anyhow::anyhow!("Failed to acquire semaphore permit").to_string(),
                    )
                })?;

                let encoding_task =
                    task::spawn_blocking(move || tokenizer.encode_pair(&text1, &text2));

                if let Some(timeout_ms) = timeout_ms {
                    match tokio::time::timeout(
                        std::time::Duration::from_millis(timeout_ms),
                        encoding_task,
                    )
                    .await
                    {
                        Ok(result) => result.map_err(|e| {
                            trustformers_core::errors::TrustformersError::other(
                                anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                            )
                        })?,
                        Err(_) => Err(trustformers_core::errors::TrustformersError::other(
                            anyhow::anyhow!("Tokenization timeout").to_string(),
                        )),
                    }
                } else {
                    encoding_task.await.map_err(|e| {
                        trustformers_core::errors::TrustformersError::other(
                            anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                        )
                    })?
                }
            });

            tasks.push(task);
        }

        let mut results = Vec::with_capacity(text_pairs.len());
        for task in tasks {
            let result = task.await.map_err(|e| {
                trustformers_core::errors::TrustformersError::other(
                    anyhow::anyhow!(format!("Task join error: {}", e)).to_string(),
                )
            })??;
            results.push(result);
        }

        Ok(results)
    }

    fn encode_stream<'a>(
        &'a self,
        texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Box<dyn Stream<Item = Result<TokenizedInput>> + Send + Unpin>>> {
        async move {
            let (tx, rx) = mpsc::channel(self.config.stream_buffer_size);
            let tokenizer = Arc::clone(&self.tokenizer);
            let semaphore = Arc::clone(&self.task_semaphore);
            let timeout_ms = self.config.task_timeout_ms;

            // Spawn a task to process all texts
            task::spawn(async move {
                for text in texts {
                    let tokenizer = Arc::clone(&tokenizer);
                    let semaphore = Arc::clone(&semaphore);
                    let tx = tx.clone();

                    task::spawn(async move {
                        let result = async {
                            let _permit = semaphore.acquire().await.map_err(|_| {
                                trustformers_core::errors::TrustformersError::other(
                                    anyhow::anyhow!("Failed to acquire semaphore permit")
                                        .to_string(),
                                )
                            })?;

                            let encoding_task =
                                task::spawn_blocking(move || tokenizer.encode(&text));

                            if let Some(timeout_ms) = timeout_ms {
                                match tokio::time::timeout(
                                    std::time::Duration::from_millis(timeout_ms),
                                    encoding_task,
                                )
                                .await
                                {
                                    Ok(result) => result.map_err(|e| {
                                        trustformers_core::errors::TrustformersError::other(
                                            anyhow::anyhow!(format!("Task join error: {}", e))
                                                .to_string(),
                                        )
                                    })?,
                                    Err(_) => {
                                        Err(trustformers_core::errors::TrustformersError::other(
                                            anyhow::anyhow!("Tokenization timeout").to_string(),
                                        ))
                                    },
                                }
                            } else {
                                encoding_task.await.map_err(|e| {
                                    trustformers_core::errors::TrustformersError::other(
                                        anyhow::anyhow!(format!("Task join error: {}", e))
                                            .to_string(),
                                    )
                                })?
                            }
                        }
                        .await;

                        let _ = tx.send(result).await;
                    });
                }
            });

            let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
            Ok(Box::new(stream)
                as Box<
                    dyn Stream<Item = Result<TokenizedInput>> + Send + Unpin,
                >)
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wordpiece::WordPieceTokenizer;
    use futures::StreamExt;
    use std::time::Instant;

    #[tokio::test]
    async fn test_async_tokenizer_wrapper() {
        let mut vocab = std::collections::HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("[CLS]".to_string(), 1);
        vocab.insert("[SEP]".to_string(), 2);
        vocab.insert("[PAD]".to_string(), 3);
        vocab.insert("[MASK]".to_string(), 4);
        vocab.insert("hello".to_string(), 5);
        vocab.insert("world".to_string(), 6);

        let tokenizer = WordPieceTokenizer::new(vocab, true);
        let async_tokenizer = AsyncTokenizerWrapper::new(tokenizer, Some(4));

        let result = async_tokenizer.encode_async("Hello world").await.unwrap();
        assert!(!result.input_ids.is_empty());
    }

    #[tokio::test]
    async fn test_batch_async_encoding() {
        let tokenizer = WordPieceTokenizer::from_pretrained("bert-base-uncased").unwrap();
        let async_tokenizer = AsyncTokenizerWrapper::new(tokenizer, Some(4));

        let texts = vec!["Hello world", "This is a test", "Async tokenization"];
        let results = async_tokenizer.encode_batch_async(&texts).await.unwrap();

        assert_eq!(results.len(), texts.len());
        for result in &results {
            assert!(!result.input_ids.is_empty());
        }
    }

    #[tokio::test]
    async fn test_configurable_async_tokenizer() {
        let tokenizer = WordPieceTokenizer::from_pretrained("bert-base-uncased").unwrap();
        let config = AsyncTokenizerConfig {
            max_concurrent_tasks: 2,
            stream_buffer_size: 100,
            task_timeout_ms: Some(5000),
            enable_cancellation: true,
        };
        let async_tokenizer = ConfigurableAsyncTokenizer::new(tokenizer, config);

        let result = async_tokenizer.encode_async("Hello world").await.unwrap();
        assert!(!result.input_ids.is_empty());
    }

    #[tokio::test]
    async fn test_async_decode() {
        let mut vocab = std::collections::HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("[CLS]".to_string(), 1);
        vocab.insert("[SEP]".to_string(), 2);
        vocab.insert("[PAD]".to_string(), 3);
        vocab.insert("[MASK]".to_string(), 4);
        vocab.insert("hello".to_string(), 5);
        vocab.insert("world".to_string(), 6);

        let tokenizer = WordPieceTokenizer::new(vocab, true);
        let async_tokenizer = AsyncTokenizerWrapper::new(tokenizer, Some(4));

        let encoded = async_tokenizer.encode_async("Hello world").await.unwrap();
        let decoded = async_tokenizer.decode_async(&encoded.input_ids).await.unwrap();

        assert!(!decoded.is_empty());
        assert!(
            decoded.to_lowercase().contains("hello") || decoded.to_lowercase().contains("world")
        );
    }

    #[tokio::test]
    async fn test_stream_encoding() {
        let mut vocab = std::collections::HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("[CLS]".to_string(), 1);
        vocab.insert("[SEP]".to_string(), 2);
        vocab.insert("[PAD]".to_string(), 3);
        vocab.insert("[MASK]".to_string(), 4);
        vocab.insert("hello".to_string(), 5);
        vocab.insert("world".to_string(), 6);
        vocab.insert("this".to_string(), 7);
        vocab.insert("is".to_string(), 8);
        vocab.insert("a".to_string(), 9);
        vocab.insert("test".to_string(), 10);
        vocab.insert("async".to_string(), 11);
        vocab.insert("tokenization".to_string(), 12);

        let tokenizer = WordPieceTokenizer::new(vocab, true);
        let async_tokenizer = AsyncTokenizerWrapper::new(tokenizer, Some(4));

        let texts = vec![
            "Hello world".to_string(),
            "This is a test".to_string(),
            "Async tokenization".to_string(),
        ];

        let mut stream = async_tokenizer.encode_stream(texts.clone()).await.unwrap();
        let mut results = Vec::new();

        while let Some(result) = stream.next().await {
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), texts.len());
    }

    #[tokio::test]
    async fn test_large_batch_with_progress() {
        let tokenizer = WordPieceTokenizer::from_pretrained("bert-base-uncased").unwrap();
        let config = AsyncTokenizerConfig::default();
        let async_tokenizer = ConfigurableAsyncTokenizer::new(tokenizer, config);

        let texts: Vec<&str> = (0..100)
            .map(
                |i| {
                    if i % 2 == 0 {
                        "Hello world"
                    } else {
                        "This is a test"
                    }
                },
            )
            .collect();

        let progress_updates = Arc::new(std::sync::Mutex::new(Vec::new()));
        let progress_updates_clone = Arc::clone(&progress_updates);

        let results = async_tokenizer
            .encode_large_batch_with_progress(&texts, move |completed, total| {
                progress_updates_clone
                    .lock()
                    .expect("lock should not be poisoned")
                    .push((completed, total));
            })
            .await
            .unwrap();

        assert_eq!(results.len(), texts.len());

        let updates = progress_updates.lock().expect("lock should not be poisoned");
        assert!(!updates.is_empty());
        assert_eq!(updates.last().unwrap().0, texts.len());
        assert_eq!(updates.last().unwrap().1, texts.len());
    }

    #[tokio::test]
    async fn test_concurrent_performance() {
        let tokenizer = WordPieceTokenizer::from_pretrained("bert-base-uncased").unwrap();
        let async_tokenizer = AsyncTokenizerWrapper::new(tokenizer, Some(8));

        let texts: Vec<&str> = (0..50)
            .map(|i| {
                if i % 2 == 0 {
                    "Hello world from async tokenization"
                } else {
                    "This is a performance test"
                }
            })
            .collect();

        let start = Instant::now();
        let results = async_tokenizer.encode_batch_async(&texts).await.unwrap();
        let duration = start.elapsed();

        assert_eq!(results.len(), texts.len());
        println!("Encoded {} texts in {:?}", texts.len(), duration);

        // Verify all results are valid
        for result in &results {
            assert!(!result.input_ids.is_empty());
            assert_eq!(result.input_ids.len(), result.attention_mask.len());
        }
    }
}
