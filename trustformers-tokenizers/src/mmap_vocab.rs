use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use trustformers_core::errors::{Result, TrustformersError};

/// Memory-mapped vocabulary for efficient large vocabulary handling
#[derive(Debug)]
pub struct MmapVocab {
    /// Memory-mapped file data
    mmap: Mmap,
    /// Offset table for fast token lookup
    token_offsets: Vec<(u32, u32)>, // (offset, length) pairs
    /// Token ID to index mapping
    id_to_index: Vec<u32>,
    /// Token string to ID mapping (for encoding)
    token_to_id: HashMap<String, u32>,
    /// Vocabulary size
    vocab_size: usize,
}

impl MmapVocab {
    /// Create a new memory-mapped vocabulary from a file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to open vocab file: {}", e))
        })?;

        let mmap = unsafe {
            MmapOptions::new().map(&file).map_err(|e| {
                TrustformersError::io_error(format!("Failed to memory map file: {}", e))
            })?
        };

        Self::from_mmap_data(mmap)
    }

    /// Create vocabulary from memory-mapped data
    fn from_mmap_data(mmap: Mmap) -> Result<Self> {
        let mut token_offsets = Vec::new();
        let mut token_to_id = HashMap::new();
        let mut id_to_index = Vec::new();

        let data = &mmap[..];
        let mut offset = 0;
        let mut token_id = 0u32;

        // Parse the memory-mapped vocabulary file
        // Expected format: one token per line, optionally with frequency
        while offset < data.len() {
            let line_start = offset;

            // Find end of line
            while offset < data.len() && data[offset] != b'\n' {
                offset += 1;
            }

            if offset > line_start {
                let line_data = &data[line_start..offset];

                // Parse token (everything before first whitespace or tab)
                let token_end = line_data
                    .iter()
                    .position(|&b| b == b' ' || b == b'\t')
                    .unwrap_or(line_data.len());

                if token_end > 0 {
                    let token_bytes = &line_data[..token_end];
                    let token = String::from_utf8_lossy(token_bytes).into_owned();

                    // Store offset and length for this token
                    token_offsets.push((line_start as u32, token_end as u32));
                    token_to_id.insert(token, token_id);
                    id_to_index.push(token_offsets.len() as u32 - 1);

                    token_id += 1;
                }
            }

            // Skip newline
            if offset < data.len() {
                offset += 1;
            }
        }

        Ok(Self {
            mmap,
            token_offsets,
            id_to_index,
            token_to_id,
            vocab_size: token_id as usize,
        })
    }

    /// Create a memory-mapped vocabulary from a regular vocabulary and save to file
    pub fn create_from_vocab<P: AsRef<Path>>(
        vocab: &HashMap<String, u32>,
        output_path: P,
    ) -> Result<Self> {
        let file = File::create(&output_path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to create vocab file: {}", e))
        })?;

        let mut writer = BufWriter::new(file);

        // Sort tokens by ID for consistent ordering
        let mut sorted_tokens: Vec<_> = vocab.iter().collect();
        sorted_tokens.sort_by_key(|(_, &id)| id);

        // Write tokens to file, one per line
        for (token, _) in sorted_tokens {
            writeln!(writer, "{}", token).map_err(|e| {
                TrustformersError::io_error(format!("Failed to write token: {}", e))
            })?;
        }

        writer
            .flush()
            .map_err(|e| TrustformersError::io_error(format!("Failed to flush writer: {}", e)))?;

        drop(writer);

        // Load the created file as memory-mapped vocabulary
        Self::from_file(output_path)
    }

    /// Get token ID by token string
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token string by ID
    pub fn get_token(&self, id: u32) -> Option<String> {
        if id as usize >= self.vocab_size {
            return None;
        }

        let index = self.id_to_index[id as usize] as usize;
        if index >= self.token_offsets.len() {
            return None;
        }

        let (offset, length) = self.token_offsets[index];
        let token_data = &self.mmap[offset as usize..(offset + length) as usize];
        Some(String::from_utf8_lossy(token_data).into_owned())
    }

    /// Get vocabulary size
    pub fn size(&self) -> usize {
        self.vocab_size
    }

    /// Check if token exists in vocabulary
    pub fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    /// Get all tokens as an iterator
    pub fn tokens(&self) -> TokenIterator<'_> {
        TokenIterator {
            vocab: self,
            current_id: 0,
        }
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            mmap_size: self.mmap.len(),
            token_offsets_size: self.token_offsets.len() * std::mem::size_of::<(u32, u32)>(),
            id_to_index_size: self.id_to_index.len() * std::mem::size_of::<u32>(),
            token_to_id_size: self.token_to_id.len()
                * (std::mem::size_of::<String>() + std::mem::size_of::<u32>())
                + self.token_to_id.keys().map(|k| k.len()).sum::<usize>(),
            total_tokens: self.vocab_size,
        }
    }

    /// Compact the vocabulary by rebuilding internal structures
    pub fn compact(&mut self) -> Result<()> {
        // Rebuild token_to_id with exact capacity
        let mut new_token_to_id = HashMap::with_capacity(self.vocab_size);

        for id in 0..self.vocab_size as u32 {
            if let Some(token) = self.get_token(id) {
                new_token_to_id.insert(token, id);
            }
        }

        self.token_to_id = new_token_to_id;
        Ok(())
    }

    /// Search for tokens with a given prefix
    pub fn find_tokens_with_prefix(&self, prefix: &str) -> Vec<(String, u32)> {
        self.token_to_id
            .iter()
            .filter_map(
                |(token, &id)| {
                    if token.starts_with(prefix) {
                        Some((token.clone(), id))
                    } else {
                        None
                    }
                },
            )
            .collect()
    }

    /// Get the most frequent tokens (assumes tokens are ordered by frequency)
    pub fn get_most_frequent(&self, count: usize) -> Vec<(String, u32)> {
        (0..count.min(self.vocab_size))
            .filter_map(|id| self.get_token(id as u32).map(|token| (token, id as u32)))
            .collect()
    }
}

/// Iterator over vocabulary tokens
pub struct TokenIterator<'a> {
    vocab: &'a MmapVocab,
    current_id: u32,
}

impl<'a> Iterator for TokenIterator<'a> {
    type Item = (String, u32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_id as usize >= self.vocab.vocab_size {
            return None;
        }

        let token = self.vocab.get_token(self.current_id)?;
        let id = self.current_id;
        self.current_id += 1;

        Some((token, id))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vocab.vocab_size - self.current_id as usize;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for TokenIterator<'a> {}

/// Memory usage statistics for the vocabulary
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Size of memory-mapped file in bytes
    pub mmap_size: usize,
    /// Size of token offset table in bytes
    pub token_offsets_size: usize,
    /// Size of ID to index mapping in bytes
    pub id_to_index_size: usize,
    /// Size of token to ID hash map in bytes (approximate)
    pub token_to_id_size: usize,
    /// Total number of tokens
    pub total_tokens: usize,
}

impl MemoryStats {
    /// Get total memory usage in bytes
    pub fn total_memory(&self) -> usize {
        self.mmap_size + self.token_offsets_size + self.id_to_index_size + self.token_to_id_size
    }

    /// Get memory usage per token in bytes
    pub fn memory_per_token(&self) -> f64 {
        if self.total_tokens == 0 {
            0.0
        } else {
            self.total_memory() as f64 / self.total_tokens as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::NamedTempFile;

    fn create_test_vocab() -> HashMap<String, u32> {
        let mut vocab = HashMap::new();
        vocab.insert("[PAD]".to_string(), 0);
        vocab.insert("[UNK]".to_string(), 1);
        vocab.insert("[CLS]".to_string(), 2);
        vocab.insert("[SEP]".to_string(), 3);
        vocab.insert("hello".to_string(), 4);
        vocab.insert("world".to_string(), 5);
        vocab.insert("test".to_string(), 6);
        vocab
    }

    #[test]
    fn test_mmap_vocab_creation() {
        let vocab = create_test_vocab();
        let temp_file = NamedTempFile::new().unwrap();

        let mmap_vocab = MmapVocab::create_from_vocab(&vocab, temp_file.path()).unwrap();

        assert_eq!(mmap_vocab.size(), vocab.len());

        // Test token lookups
        assert_eq!(mmap_vocab.get_id("hello"), Some(4));
        assert_eq!(mmap_vocab.get_id("world"), Some(5));
        assert_eq!(mmap_vocab.get_id("nonexistent"), None);

        // Test reverse lookups
        assert_eq!(mmap_vocab.get_token(4), Some("hello".to_string()));
        assert_eq!(mmap_vocab.get_token(5), Some("world".to_string()));
        assert_eq!(mmap_vocab.get_token(999), None);
    }

    #[test]
    fn test_mmap_vocab_contains() {
        let vocab = create_test_vocab();
        let temp_file = NamedTempFile::new().unwrap();

        let mmap_vocab = MmapVocab::create_from_vocab(&vocab, temp_file.path()).unwrap();

        assert!(mmap_vocab.contains("hello"));
        assert!(mmap_vocab.contains("[PAD]"));
        assert!(!mmap_vocab.contains("nonexistent"));
    }

    #[test]
    fn test_mmap_vocab_iterator() {
        let vocab = create_test_vocab();
        let temp_file = NamedTempFile::new().unwrap();

        let mmap_vocab = MmapVocab::create_from_vocab(&vocab, temp_file.path()).unwrap();

        let tokens: Vec<_> = mmap_vocab.tokens().collect();
        assert_eq!(tokens.len(), vocab.len());

        // Check that all original tokens are present
        let token_set: std::collections::HashSet<_> =
            tokens.iter().map(|(token, _)| token).collect();
        for original_token in vocab.keys() {
            assert!(token_set.contains(original_token));
        }
    }

    #[test]
    fn test_mmap_vocab_memory_stats() {
        let vocab = create_test_vocab();
        let temp_file = NamedTempFile::new().unwrap();

        let mmap_vocab = MmapVocab::create_from_vocab(&vocab, temp_file.path()).unwrap();

        let stats = mmap_vocab.memory_stats();
        assert!(stats.mmap_size > 0);
        assert!(stats.total_memory() > 0);
        assert!(stats.memory_per_token() > 0.0);
        assert_eq!(stats.total_tokens, vocab.len());
    }

    #[test]
    fn test_mmap_vocab_prefix_search() {
        let vocab = create_test_vocab();
        let temp_file = NamedTempFile::new().unwrap();

        let mmap_vocab = MmapVocab::create_from_vocab(&vocab, temp_file.path()).unwrap();

        let cls_tokens = mmap_vocab.find_tokens_with_prefix("[");
        assert!(cls_tokens.len() >= 3); // [PAD], [UNK], [CLS], [SEP]

        let h_tokens = mmap_vocab.find_tokens_with_prefix("h");
        assert_eq!(h_tokens.len(), 1); // Only "hello"
        assert_eq!(h_tokens[0].0, "hello");
    }

    #[test]
    fn test_mmap_vocab_most_frequent() {
        let vocab = create_test_vocab();
        let temp_file = NamedTempFile::new().unwrap();

        let mmap_vocab = MmapVocab::create_from_vocab(&vocab, temp_file.path()).unwrap();

        let top_3 = mmap_vocab.get_most_frequent(3);
        assert_eq!(top_3.len(), 3);

        // Should be ordered by ID (which corresponds to original frequency order)
        assert_eq!(top_3[0].1, 0); // [PAD]
        assert_eq!(top_3[1].1, 1); // [UNK]
        assert_eq!(top_3[2].1, 2); // [CLS]
    }

    #[test]
    fn test_mmap_vocab_compact() {
        let vocab = create_test_vocab();
        let temp_file = NamedTempFile::new().unwrap();

        let mut mmap_vocab = MmapVocab::create_from_vocab(&vocab, temp_file.path()).unwrap();

        // Test compaction
        assert!(mmap_vocab.compact().is_ok());

        // Functionality should remain the same after compaction
        assert_eq!(mmap_vocab.get_id("hello"), Some(4));
        assert_eq!(mmap_vocab.get_token(4), Some("hello".to_string()));
    }

    #[test]
    fn test_mmap_vocab_large_file() {
        // Test with a larger vocabulary
        let mut large_vocab = HashMap::new();
        for i in 0..10000 {
            large_vocab.insert(format!("token_{}", i), i as u32);
        }

        let temp_file = NamedTempFile::new().unwrap();
        let mmap_vocab = MmapVocab::create_from_vocab(&large_vocab, temp_file.path()).unwrap();

        assert_eq!(mmap_vocab.size(), 10000);
        assert_eq!(mmap_vocab.get_id("token_5000"), Some(5000));
        assert_eq!(mmap_vocab.get_token(5000), Some("token_5000".to_string()));

        let stats = mmap_vocab.memory_stats();
        assert!(stats.memory_per_token() < 100.0); // Should be efficient
    }
}
