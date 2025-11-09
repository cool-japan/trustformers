use anyhow::Result as AnyhowResult;
use std::io::{BufRead, BufReader, Read};
use trustformers_core::errors::Result;
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Streaming tokenizer for processing large texts efficiently
pub struct StreamingTokenizer<T: Tokenizer> {
    tokenizer: T,
    buffer_size: usize,
    overlap_size: usize,
    max_chunk_length: Option<usize>,
}

impl<T: Tokenizer> StreamingTokenizer<T> {
    /// Create a new streaming tokenizer
    pub fn new(tokenizer: T) -> Self {
        Self {
            tokenizer,
            buffer_size: 8192, // 8KB buffer
            overlap_size: 256, // 256 chars overlap between chunks
            max_chunk_length: None,
        }
    }

    /// Set the buffer size for reading from stream
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }

    /// Set the overlap size between chunks
    pub fn with_overlap_size(mut self, overlap_size: usize) -> Self {
        self.overlap_size = overlap_size;
        self
    }

    /// Set maximum chunk length for tokenization
    pub fn with_max_chunk_length(mut self, max_length: usize) -> Self {
        self.max_chunk_length = Some(max_length);
        self
    }

    /// Process a stream of text and return tokenized chunks
    pub fn process_stream<R: Read>(&self, reader: R) -> Result<Vec<TokenizedInput>> {
        let mut buf_reader = BufReader::with_capacity(self.buffer_size, reader);
        let mut chunks = Vec::new();
        let mut buffer = String::new();
        let mut previous_overlap = String::new();

        loop {
            buffer.clear();
            let bytes_read = buf_reader.read_line(&mut buffer).map_err(|e| {
                trustformers_core::errors::TrustformersError::other(format!("I/O error: {}", e))
            })?;

            if bytes_read == 0 {
                break; // End of stream
            }

            // Combine with previous overlap
            let full_text = if previous_overlap.is_empty() {
                buffer.clone()
            } else {
                format!("{}{}", previous_overlap, buffer)
            };

            // Tokenize the chunk
            let tokenized = self.tokenize_chunk(&full_text)?;
            chunks.push(tokenized);

            // Prepare overlap for next chunk
            if full_text.len() > self.overlap_size {
                previous_overlap = full_text[full_text.len() - self.overlap_size..].to_string();
            } else {
                previous_overlap.clear();
            }
        }

        Ok(chunks)
    }

    /// Process text from a string in streaming fashion
    pub fn process_text(&self, text: &str) -> Result<Vec<TokenizedInput>> {
        let mut chunks = Vec::new();
        let mut start = 0;
        let chunk_size = self.buffer_size;

        // Handle empty text case
        if text.is_empty() {
            let empty_chunk = self.tokenize_chunk("")?;
            chunks.push(empty_chunk);
            return Ok(chunks);
        }

        while start < text.len() {
            let end = std::cmp::min(start + chunk_size, text.len());
            let mut chunk_end = end;

            // Try to end at a word boundary if possible
            if end < text.len() {
                if let Some(last_space) = text[start..end].rfind(' ') {
                    chunk_end = start + last_space;
                }
            }

            // Ensure we always make progress to avoid infinite loops
            if chunk_end <= start {
                chunk_end = std::cmp::min(start + 1, text.len());
            }

            let chunk_text = &text[start..chunk_end];
            let tokenized = self.tokenize_chunk(chunk_text)?;
            chunks.push(tokenized);

            // Move start with overlap, ensuring we always advance
            let next_start = if chunk_end > self.overlap_size {
                chunk_end - self.overlap_size
            } else {
                chunk_end
            };

            // Ensure we always advance at least one position to avoid infinite loop
            start = std::cmp::max(next_start, start + 1);
        }

        Ok(chunks)
    }

    /// Process an iterator of text lines
    pub fn process_lines<I>(&self, lines: I) -> Result<Vec<TokenizedInput>>
    where
        I: Iterator<Item = String>,
    {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        for line in lines {
            // Add line to current chunk
            if !current_chunk.is_empty() {
                current_chunk.push('\n');
            }
            current_chunk.push_str(&line);

            // If chunk is large enough, tokenize it
            if current_chunk.len() >= self.buffer_size {
                let tokenized = self.tokenize_chunk(&current_chunk)?;
                chunks.push(tokenized);

                // Keep overlap
                if current_chunk.len() > self.overlap_size {
                    current_chunk =
                        current_chunk[current_chunk.len() - self.overlap_size..].to_string();
                } else {
                    current_chunk.clear();
                }
            }
        }

        // Process remaining chunk
        if !current_chunk.is_empty() {
            let tokenized = self.tokenize_chunk(&current_chunk)?;
            chunks.push(tokenized);
        }

        Ok(chunks)
    }

    /// Tokenize a single chunk with length constraints
    fn tokenize_chunk(&self, text: &str) -> Result<TokenizedInput> {
        let mut tokenized = self.tokenizer.encode(text)?;

        // Apply max chunk length if specified
        if let Some(max_len) = self.max_chunk_length {
            if tokenized.input_ids.len() > max_len {
                tokenized.input_ids.truncate(max_len);
                tokenized.attention_mask.truncate(max_len);
                if let Some(ref mut token_type_ids) = tokenized.token_type_ids {
                    token_type_ids.truncate(max_len);
                }
            }
        }

        Ok(tokenized)
    }

    /// Get the underlying tokenizer
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Get overlap size
    pub fn overlap_size(&self) -> usize {
        self.overlap_size
    }

    /// Get max chunk length
    pub fn max_chunk_length(&self) -> Option<usize> {
        self.max_chunk_length
    }
}

/// Batched streaming tokenizer for processing multiple streams
pub struct BatchedStreamingTokenizer<T: Tokenizer> {
    streaming_tokenizer: StreamingTokenizer<T>,
    batch_size: usize,
}

impl<T: Tokenizer> BatchedStreamingTokenizer<T> {
    /// Create a new batched streaming tokenizer
    pub fn new(tokenizer: T, batch_size: usize) -> Self {
        Self {
            streaming_tokenizer: StreamingTokenizer::new(tokenizer),
            batch_size,
        }
    }

    /// Set streaming parameters
    pub fn with_streaming_params(mut self, buffer_size: usize, overlap_size: usize) -> Self {
        self.streaming_tokenizer = self
            .streaming_tokenizer
            .with_buffer_size(buffer_size)
            .with_overlap_size(overlap_size);
        self
    }

    /// Set max chunk length
    pub fn with_max_chunk_length(mut self, max_length: usize) -> Self {
        self.streaming_tokenizer = self.streaming_tokenizer.with_max_chunk_length(max_length);
        self
    }

    /// Process multiple text streams in batches
    pub fn process_text_batch(&self, texts: &[String]) -> Result<Vec<Vec<TokenizedInput>>> {
        let mut results = Vec::new();

        for batch in texts.chunks(self.batch_size) {
            let mut batch_results = Vec::new();
            for text in batch {
                let tokenized_chunks = self.streaming_tokenizer.process_text(text)?;
                batch_results.push(tokenized_chunks);
            }
            results.extend(batch_results);
        }

        Ok(results)
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get the underlying streaming tokenizer
    pub fn streaming_tokenizer(&self) -> &StreamingTokenizer<T> {
        &self.streaming_tokenizer
    }
}

/// Memory-efficient text iterator for large files
pub struct TextFileIterator<R: BufRead> {
    reader: R,
    buffer: String,
    chunk_size: usize,
    #[allow(dead_code)]
    overlap_size: usize,
    eof: bool,
}

impl<R: BufRead> TextFileIterator<R> {
    /// Create a new text file iterator
    pub fn new(reader: R, chunk_size: usize, overlap_size: usize) -> Self {
        Self {
            reader,
            buffer: String::new(),
            chunk_size,
            overlap_size,
            eof: false,
        }
    }

    /// Read next chunk from the file
    pub fn next_chunk(&mut self) -> AnyhowResult<Option<String>> {
        if self.eof {
            return Ok(None);
        }

        self.buffer.clear();

        // Read chunk_size bytes
        let mut bytes_read = 0;
        let mut temp_buf = String::new();

        while bytes_read < self.chunk_size {
            temp_buf.clear();
            let n = self.reader.read_line(&mut temp_buf)?;
            if n == 0 {
                self.eof = true;
                break;
            }
            self.buffer.push_str(&temp_buf);
            bytes_read += n;
        }

        if self.buffer.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.buffer.clone()))
        }
    }
}

impl<R: BufRead> Iterator for TextFileIterator<R> {
    type Item = AnyhowResult<String>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_chunk() {
            Ok(Some(chunk)) => Some(Ok(chunk)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::char::CharTokenizer;
    use std::io::Cursor;

    fn create_test_tokenizer() -> CharTokenizer {
        let mut vocab = std::collections::HashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 1);
        vocab.insert("c".to_string(), 2);
        vocab.insert(" ".to_string(), 3);
        CharTokenizer::new(vocab)
    }

    #[test]
    fn test_streaming_tokenizer_basic() {
        let tokenizer = create_test_tokenizer();
        let streaming = StreamingTokenizer::new(tokenizer);

        let text = "Hello world! This is a test of streaming tokenization.";
        let chunks = streaming.process_text(text).unwrap();

        assert!(!chunks.is_empty());
        // Each chunk should have tokenized content
        for chunk in chunks {
            assert!(!chunk.input_ids.is_empty());
            assert!(!chunk.attention_mask.is_empty());
        }
    }

    #[test]
    fn test_streaming_tokenizer_with_params() {
        let tokenizer = create_test_tokenizer();
        let streaming = StreamingTokenizer::new(tokenizer)
            .with_buffer_size(50)
            .with_overlap_size(10)
            .with_max_chunk_length(20);

        let text = "This is a longer text that should be split into multiple chunks based on the buffer size.";
        let chunks = streaming.process_text(text).unwrap();

        assert!(chunks.len() > 1);

        // Check max chunk length constraint
        for chunk in chunks {
            assert!(chunk.input_ids.len() <= 20);
        }
    }

    #[test]
    fn test_streaming_tokenizer_from_reader() {
        let tokenizer = create_test_tokenizer();
        let streaming = StreamingTokenizer::new(tokenizer);

        let text = "Line 1\nLine 2\nLine 3\n";
        let cursor = Cursor::new(text.as_bytes());
        let chunks = streaming.process_stream(cursor).unwrap();

        assert!(!chunks.is_empty());
        for chunk in chunks {
            assert!(!chunk.input_ids.is_empty());
        }
    }

    #[test]
    fn test_streaming_tokenizer_lines() {
        let tokenizer = create_test_tokenizer();
        let streaming = StreamingTokenizer::new(tokenizer).with_buffer_size(20);

        let lines = vec![
            "First line".to_string(),
            "Second line".to_string(),
            "Third line".to_string(),
        ];

        let chunks = streaming.process_lines(lines.into_iter()).unwrap();
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_batched_streaming_tokenizer() {
        let tokenizer = create_test_tokenizer();
        let batched = BatchedStreamingTokenizer::new(tokenizer, 2).with_streaming_params(50, 10);

        let texts = vec![
            "First text to tokenize".to_string(),
            "Second text to tokenize".to_string(),
            "Third text to tokenize".to_string(),
        ];

        let results = batched.process_text_batch(&texts).unwrap();
        assert_eq!(results.len(), 3);

        for result in results {
            assert!(!result.is_empty());
            for chunk in result {
                assert!(!chunk.input_ids.is_empty());
            }
        }
    }

    #[test]
    fn test_text_file_iterator() {
        let text = "Line 1\nLine 2\nLine 3\nLine 4\n";
        let cursor = Cursor::new(text.as_bytes());
        let buf_reader = BufReader::new(cursor);

        let iterator = TextFileIterator::new(buf_reader, 10, 2);

        let chunks: std::result::Result<Vec<_>, _> = iterator.collect();
        let chunks = chunks.unwrap();

        assert!(!chunks.is_empty());
        for chunk in chunks {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn test_streaming_empty_text() {
        let tokenizer = create_test_tokenizer();
        let streaming = StreamingTokenizer::new(tokenizer);

        let chunks = streaming.process_text("").unwrap();
        assert_eq!(chunks.len(), 1); // Should have one empty chunk
        assert!(chunks[0].input_ids.is_empty() || chunks[0].input_ids.len() == 1);
        // Might have just padding
    }

    #[test]
    fn test_streaming_configuration() {
        let tokenizer = create_test_tokenizer();
        let streaming = StreamingTokenizer::new(tokenizer)
            .with_buffer_size(1024)
            .with_overlap_size(128)
            .with_max_chunk_length(512);

        assert_eq!(streaming.buffer_size(), 1024);
        assert_eq!(streaming.overlap_size(), 128);
        assert_eq!(streaming.max_chunk_length(), Some(512));
    }
}
