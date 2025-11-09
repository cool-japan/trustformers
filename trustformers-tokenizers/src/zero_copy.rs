use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::slice;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Zero-copy header for tokenizer files
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct ZeroCopyHeader {
    /// Magic bytes to identify format
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// Size of the header
    pub header_size: u32,
    /// Offset to vocabulary section
    pub vocab_offset: u64,
    /// Size of vocabulary section
    pub vocab_size: u64,
    /// Offset to metadata section
    pub metadata_offset: u64,
    /// Size of metadata section
    pub metadata_size: u64,
    /// Offset to special tokens section
    pub special_tokens_offset: u64,
    /// Size of special tokens section
    pub special_tokens_size: u64,
    /// Checksum of the entire file
    pub checksum: u64,
    /// Padding for alignment
    pub padding: [u8; 8],
}

impl ZeroCopyHeader {
    const MAGIC: [u8; 4] = *b"TFZC"; // TrustFormeR Zero Copy
    const VERSION: u32 = 1;
    const SIZE: usize = std::mem::size_of::<Self>();

    /// Create a new header with the given parameters
    pub fn new(
        vocab_offset: u64,
        vocab_size: u64,
        metadata_offset: u64,
        metadata_size: u64,
        special_tokens_offset: u64,
        special_tokens_size: u64,
        checksum: u64,
    ) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            header_size: Self::SIZE as u32,
            vocab_offset,
            vocab_size,
            metadata_offset,
            metadata_size,
            special_tokens_offset,
            special_tokens_size,
            checksum,
            padding: [0; 8],
        }
    }

    /// Validate the header
    pub fn validate(&self) -> Result<()> {
        if self.magic != Self::MAGIC {
            return Err(TrustformersError::serialization_error(
                "Invalid magic bytes in zero-copy header".to_string(),
            ));
        }

        let version = self.version;
        if version != Self::VERSION {
            return Err(TrustformersError::serialization_error(format!(
                "Unsupported version: {}, expected: {}",
                version,
                Self::VERSION
            )));
        }

        let header_size = self.header_size;
        if header_size != Self::SIZE as u32 {
            return Err(TrustformersError::serialization_error(
                "Invalid header size".to_string(),
            ));
        }

        Ok(())
    }
}

/// Vocabulary entry for zero-copy access
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct ZeroCopyVocabEntry {
    /// Token ID
    pub id: u32,
    /// Offset to token string in the file
    pub token_offset: u64,
    /// Length of token string
    pub token_length: u32,
    /// Token frequency
    pub frequency: f32,
    /// Flags (special token, etc.)
    pub flags: u32,
    /// Padding for alignment
    pub padding: [u8; 4],
}

impl ZeroCopyVocabEntry {
    /// Check if this is a special token
    pub fn is_special(&self) -> bool {
        (self.flags & 0x01) != 0
    }

    /// Set special token flag
    pub fn set_special(&mut self, is_special: bool) {
        if is_special {
            self.flags |= 0x01;
        } else {
            self.flags &= !0x01;
        }
    }
}

/// Zero-copy tokenizer implementation
pub struct ZeroCopyTokenizer {
    /// Memory-mapped file
    mmap: Mmap,
    /// Header information
    header: ZeroCopyHeader,
    /// Vocabulary entries
    vocab_entries: &'static [ZeroCopyVocabEntry],
    /// Token-to-ID mapping for fast lookup
    token_to_id: HashMap<String, u32>,
    /// ID-to-token mapping for fast lookup
    id_to_token: HashMap<u32, String>,
}

impl ZeroCopyTokenizer {
    /// Load a tokenizer from a zero-copy file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Read and validate header
        if mmap.len() < ZeroCopyHeader::SIZE {
            return Err(TrustformersError::serialization_error(
                "File too small to contain header".to_string(),
            ));
        }

        let header_bytes = &mmap[0..ZeroCopyHeader::SIZE];
        let header: ZeroCopyHeader =
            unsafe { std::ptr::read(header_bytes.as_ptr() as *const ZeroCopyHeader) };

        header.validate()?;

        // Read vocabulary entries
        let vocab_start = header.vocab_offset as usize;
        let vocab_end = vocab_start + header.vocab_size as usize;

        if vocab_end > mmap.len() {
            return Err(TrustformersError::serialization_error(
                "Vocabulary section extends beyond file".to_string(),
            ));
        }

        let entry_size = std::mem::size_of::<ZeroCopyVocabEntry>();
        let num_entries = header.vocab_size as usize / entry_size;

        let vocab_entries = unsafe {
            slice::from_raw_parts(
                mmap[vocab_start..].as_ptr() as *const ZeroCopyVocabEntry,
                num_entries,
            )
        };

        // Build lookup maps
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for entry in vocab_entries {
            let token_start = entry.token_offset as usize;
            let token_end = token_start + entry.token_length as usize;

            if token_end > mmap.len() {
                return Err(TrustformersError::serialization_error(
                    "Token string extends beyond file".to_string(),
                ));
            }

            let token_bytes = &mmap[token_start..token_end];
            let token = String::from_utf8(token_bytes.to_vec()).map_err(|e| {
                TrustformersError::serialization_error(format!("Invalid UTF-8 in token: {}", e))
            })?;

            token_to_id.insert(token.clone(), entry.id);
            id_to_token.insert(entry.id, token);
        }

        Ok(Self {
            mmap,
            header,
            vocab_entries,
            token_to_id,
            id_to_token,
        })
    }

    /// Get the header information
    pub fn header(&self) -> &ZeroCopyHeader {
        &self.header
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_entries.len()
    }

    /// Get a token by ID without copying
    pub fn get_token_unchecked(&self, id: u32) -> Option<&str> {
        self.vocab_entries.iter().find(|entry| entry.id == id).and_then(|entry| {
            let token_start = entry.token_offset as usize;
            let token_end = token_start + entry.token_length as usize;

            if token_end <= self.mmap.len() {
                std::str::from_utf8(&self.mmap[token_start..token_end]).ok()
            } else {
                None
            }
        })
    }

    /// Get token ID without copying
    pub fn get_id_unchecked(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get vocabulary entry by index
    pub fn get_vocab_entry(&self, index: usize) -> Option<&ZeroCopyVocabEntry> {
        self.vocab_entries.get(index)
    }

    /// Iterate over all vocabulary entries
    pub fn vocab_entries(&self) -> impl Iterator<Item = &ZeroCopyVocabEntry> {
        self.vocab_entries.iter()
    }

    /// Get metadata section as bytes
    pub fn metadata_bytes(&self) -> &[u8] {
        let start = self.header.metadata_offset as usize;
        let end = start + self.header.metadata_size as usize;
        &self.mmap[start..end]
    }

    /// Get special tokens section as bytes
    pub fn special_tokens_bytes(&self) -> &[u8] {
        let start = self.header.special_tokens_offset as usize;
        let end = start + self.header.special_tokens_size as usize;
        &self.mmap[start..end]
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> ZeroCopyMemoryStats {
        ZeroCopyMemoryStats {
            file_size: self.mmap.len(),
            header_size: ZeroCopyHeader::SIZE,
            vocab_size: self.header.vocab_size as usize,
            metadata_size: self.header.metadata_size as usize,
            special_tokens_size: self.header.special_tokens_size as usize,
            lookup_table_size: self.token_to_id.len()
                * (std::mem::size_of::<String>() + std::mem::size_of::<u32>())
                + self.id_to_token.len()
                    * (std::mem::size_of::<u32>() + std::mem::size_of::<String>()),
        }
    }

    /// Verify file integrity using checksum
    pub fn verify_integrity(&self) -> Result<bool> {
        // Calculate checksum of the data sections
        let mut hasher = crc32fast::Hasher::new();

        // Hash vocabulary section
        let vocab_start = self.header.vocab_offset as usize;
        let vocab_end = vocab_start + self.header.vocab_size as usize;
        hasher.update(&self.mmap[vocab_start..vocab_end]);

        // Hash metadata section
        let metadata_start = self.header.metadata_offset as usize;
        let metadata_end = metadata_start + self.header.metadata_size as usize;
        hasher.update(&self.mmap[metadata_start..metadata_end]);

        // Hash special tokens section
        let special_start = self.header.special_tokens_offset as usize;
        let special_end = special_start + self.header.special_tokens_size as usize;
        hasher.update(&self.mmap[special_start..special_end]);

        let calculated_checksum = hasher.finalize() as u64;
        Ok(calculated_checksum == self.header.checksum)
    }
}

impl Tokenizer for ZeroCopyTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        // Simple word-based tokenization for demonstration
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let mut input_ids = Vec::new();
        let mut tokens_out = Vec::new();

        for token in tokens {
            if let Some(id) = self.get_id_unchecked(token) {
                input_ids.push(id);
                tokens_out.push(token.to_string());
            }
        }

        Ok(TokenizedInput {
            input_ids,
            attention_mask: vec![1; tokens_out.len()],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let tokens: std::result::Result<Vec<&str>, _> = token_ids
            .iter()
            .map(|&id| {
                self.get_token_unchecked(id)
                    .ok_or_else(|| TrustformersError::other(format!("Unknown token ID: {}", id)))
            })
            .collect();

        Ok(tokens?.join(" "))
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.token_to_id.clone()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.get_id_unchecked(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.get_token_unchecked(id).map(|s| s.to_string())
    }

    fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<TokenizedInput> {
        // Simple concatenation with space separator
        let combined = format!("{} {}", text_a, text_b);
        self.encode(&combined)
    }

    fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }
}

/// Memory usage statistics for zero-copy tokenizer
#[derive(Debug, Clone)]
pub struct ZeroCopyMemoryStats {
    pub file_size: usize,
    pub header_size: usize,
    pub vocab_size: usize,
    pub metadata_size: usize,
    pub special_tokens_size: usize,
    pub lookup_table_size: usize,
}

impl ZeroCopyMemoryStats {
    /// Calculate total memory usage
    pub fn total_memory(&self) -> usize {
        self.lookup_table_size // Only lookup tables are in memory
    }

    /// Calculate memory efficiency ratio
    pub fn efficiency_ratio(&self) -> f64 {
        if self.file_size == 0 {
            0.0
        } else {
            self.total_memory() as f64 / self.file_size as f64
        }
    }
}

/// Builder for creating zero-copy tokenizer files
pub struct ZeroCopyBuilder {
    vocabulary: Vec<(String, u32, f32, bool)>, // token, id, frequency, is_special
    metadata: Vec<u8>,
    special_tokens: Vec<u8>,
}

impl ZeroCopyBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            vocabulary: Vec::new(),
            metadata: Vec::new(),
            special_tokens: Vec::new(),
        }
    }

    /// Add a token to the vocabulary
    pub fn add_token(
        &mut self,
        token: String,
        id: u32,
        frequency: f32,
        is_special: bool,
    ) -> &mut Self {
        self.vocabulary.push((token, id, frequency, is_special));
        self
    }

    /// Add multiple tokens from a HashMap
    pub fn add_tokens_from_map(&mut self, vocab: &HashMap<String, u32>) -> &mut Self {
        for (token, &id) in vocab {
            self.add_token(token.clone(), id, 1.0, false);
        }
        self
    }

    /// Set metadata
    pub fn set_metadata(&mut self, metadata: Vec<u8>) -> &mut Self {
        self.metadata = metadata;
        self
    }

    /// Set special tokens data
    pub fn set_special_tokens(&mut self, special_tokens: Vec<u8>) -> &mut Self {
        self.special_tokens = special_tokens;
        self
    }

    /// Build and save the zero-copy tokenizer file
    pub fn build_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        use std::fs::OpenOptions;
        use std::io::Write;

        let mut file = OpenOptions::new().create(true).write(true).truncate(true).open(path)?;

        // Calculate section offsets and sizes
        let header_size = ZeroCopyHeader::SIZE as u64;
        let vocab_entry_size = std::mem::size_of::<ZeroCopyVocabEntry>() as u64;
        let vocab_entries_size = self.vocabulary.len() as u64 * vocab_entry_size;

        // Calculate string data size
        let string_data_size: u64 =
            self.vocabulary.iter().map(|(token, _, _, _)| token.len() as u64).sum();

        let vocab_offset = header_size;
        let vocab_size = vocab_entries_size + string_data_size;
        let metadata_offset = vocab_offset + vocab_size;
        let metadata_size = self.metadata.len() as u64;
        let special_tokens_offset = metadata_offset + metadata_size;
        let special_tokens_size = self.special_tokens.len() as u64;

        // Prepare vocabulary entries and string data
        let mut vocab_entries = Vec::new();
        let mut string_data = Vec::new();
        let mut current_string_offset = vocab_offset + vocab_entries_size;

        for (token, id, frequency, is_special) in &self.vocabulary {
            let token_bytes = token.as_bytes();
            let mut entry = ZeroCopyVocabEntry {
                id: *id,
                token_offset: current_string_offset,
                token_length: token_bytes.len() as u32,
                frequency: *frequency,
                flags: 0,
                padding: [0; 4],
            };

            entry.set_special(*is_special);
            vocab_entries.push(entry);

            string_data.extend_from_slice(token_bytes);
            current_string_offset += token_bytes.len() as u64;
        }

        // Calculate checksum
        let mut hasher = crc32fast::Hasher::new();

        // Hash vocabulary entries
        let entries_bytes = unsafe {
            slice::from_raw_parts(
                vocab_entries.as_ptr() as *const u8,
                vocab_entries.len() * vocab_entry_size as usize,
            )
        };
        hasher.update(entries_bytes);
        hasher.update(&string_data);
        hasher.update(&self.metadata);
        hasher.update(&self.special_tokens);

        let checksum = hasher.finalize() as u64;

        // Create header
        let header = ZeroCopyHeader::new(
            vocab_offset,
            vocab_size,
            metadata_offset,
            metadata_size,
            special_tokens_offset,
            special_tokens_size,
            checksum,
        );

        // Write header
        let header_bytes = unsafe {
            slice::from_raw_parts(
                &header as *const ZeroCopyHeader as *const u8,
                ZeroCopyHeader::SIZE,
            )
        };
        file.write_all(header_bytes)?;

        // Write vocabulary entries
        file.write_all(entries_bytes)?;

        // Write string data
        file.write_all(&string_data)?;

        // Write metadata
        file.write_all(&self.metadata)?;

        // Write special tokens
        file.write_all(&self.special_tokens)?;

        file.flush()?;
        Ok(())
    }
}

impl Default for ZeroCopyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utilities for working with zero-copy tokenizers
pub struct ZeroCopyUtils;

impl ZeroCopyUtils {
    /// Convert a regular tokenizer to zero-copy format
    pub fn convert_tokenizer_to_zero_copy<T: Tokenizer, P: AsRef<Path>>(
        tokenizer: &T,
        path: P,
        metadata: Option<&[u8]>,
        special_tokens: Option<&[u8]>,
    ) -> Result<()> {
        let mut builder = ZeroCopyBuilder::new();

        let vocab = tokenizer.get_vocab();
        builder.add_tokens_from_map(&vocab);

        if let Some(meta) = metadata {
            builder.set_metadata(meta.to_vec());
        }

        if let Some(special) = special_tokens {
            builder.set_special_tokens(special.to_vec());
        }

        builder.build_to_file(path)
    }

    /// Validate a zero-copy tokenizer file
    pub fn validate_file<P: AsRef<Path>>(path: P) -> Result<bool> {
        let tokenizer = ZeroCopyTokenizer::from_file(path)?;
        tokenizer.verify_integrity()
    }

    /// Get file information without loading the full tokenizer
    pub fn get_file_info<P: AsRef<Path>>(path: P) -> Result<HashMap<String, String>> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        if mmap.len() < ZeroCopyHeader::SIZE {
            return Err(TrustformersError::serialization_error(
                "File too small to contain header".to_string(),
            ));
        }

        let header_bytes = &mmap[0..ZeroCopyHeader::SIZE];
        let header: ZeroCopyHeader =
            unsafe { std::ptr::read(header_bytes.as_ptr() as *const ZeroCopyHeader) };

        header.validate()?;

        let mut info = HashMap::new();
        // Copy packed struct fields to local variables to avoid alignment issues
        let version = header.version;
        let vocab_size = header.vocab_size;
        let metadata_size = header.metadata_size;
        let special_tokens_size = header.special_tokens_size;

        info.insert("format".to_string(), "ZeroCopy".to_string());
        info.insert("version".to_string(), version.to_string());
        info.insert("file_size".to_string(), mmap.len().to_string());
        info.insert(
            "vocab_size".to_string(),
            (vocab_size / std::mem::size_of::<ZeroCopyVocabEntry>() as u64).to_string(),
        );
        info.insert("metadata_size".to_string(), metadata_size.to_string());
        info.insert(
            "special_tokens_size".to_string(),
            special_tokens_size.to_string(),
        );

        Ok(info)
    }

    /// Compare two zero-copy files
    pub fn compare_files<P1: AsRef<Path>, P2: AsRef<Path>>(
        path1: P1,
        path2: P2,
    ) -> Result<HashMap<String, String>> {
        let info1 = Self::get_file_info(path1)?;
        let info2 = Self::get_file_info(path2)?;

        let mut comparison = HashMap::new();

        let default_value = "0".to_string();
        for key in &[
            "file_size",
            "vocab_size",
            "metadata_size",
            "special_tokens_size",
        ] {
            let val1 = info1.get(*key).unwrap_or(&default_value);
            let val2 = info2.get(*key).unwrap_or(&default_value);

            comparison.insert(format!("{}_1", key), val1.clone());
            comparison.insert(format!("{}_2", key), val2.clone());
            comparison.insert(format!("{}_equal", key), (val1 == val2).to_string());
        }

        Ok(comparison)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as TestHashMap;
    use tempfile::tempdir;

    #[test]
    fn test_zero_copy_header() {
        let header = ZeroCopyHeader::new(100, 200, 300, 50, 350, 25, 0x12345678);

        assert_eq!(header.magic, ZeroCopyHeader::MAGIC);
        // Copy fields to local variables to avoid unaligned reference errors
        let version = header.version;
        let vocab_offset = header.vocab_offset;
        let vocab_size = header.vocab_size;
        let checksum = header.checksum;
        assert_eq!(version, ZeroCopyHeader::VERSION);
        assert_eq!(vocab_offset, 100);
        assert_eq!(vocab_size, 200);
        assert_eq!(checksum, 0x12345678);

        assert!(header.validate().is_ok());
    }

    #[test]
    fn test_vocab_entry_flags() {
        let mut entry = ZeroCopyVocabEntry {
            id: 1,
            token_offset: 0,
            token_length: 5,
            frequency: 1.0,
            flags: 0,
            padding: [0; 4],
        };

        assert!(!entry.is_special());

        entry.set_special(true);
        assert!(entry.is_special());

        entry.set_special(false);
        assert!(!entry.is_special());
    }

    #[test]
    fn test_zero_copy_builder() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_tokenizer.zc");

        let mut builder = ZeroCopyBuilder::new();
        builder
            .add_token("hello".to_string(), 1, 1.0, false)
            .add_token("world".to_string(), 2, 1.0, false)
            .add_token("<pad>".to_string(), 0, 1.0, true);

        assert!(builder.build_to_file(&file_path).is_ok());
        assert!(file_path.exists());
    }

    #[test]
    fn test_zero_copy_tokenizer_loading() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_load.zc");

        // Build a test file
        let mut builder = ZeroCopyBuilder::new();
        builder
            .add_token("test".to_string(), 1, 1.0, false)
            .add_token("token".to_string(), 2, 1.0, false)
            .add_token("[CLS]".to_string(), 0, 1.0, true);

        builder.build_to_file(&file_path).unwrap();

        // Load the tokenizer
        let tokenizer = ZeroCopyTokenizer::from_file(&file_path).unwrap();

        assert_eq!(tokenizer.vocab_size(), 3);
        assert_eq!(tokenizer.get_id_unchecked("test"), Some(1));
        assert_eq!(tokenizer.get_id_unchecked("token"), Some(2));
        assert_eq!(tokenizer.get_id_unchecked("[CLS]"), Some(0));

        assert_eq!(tokenizer.get_token_unchecked(1), Some("test"));
        assert_eq!(tokenizer.get_token_unchecked(2), Some("token"));
        assert_eq!(tokenizer.get_token_unchecked(0), Some("[CLS]"));
    }

    #[test]
    fn test_tokenizer_interface() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_interface.zc");

        // Build a test file
        let mut builder = ZeroCopyBuilder::new();
        builder.add_token("hello".to_string(), 1, 1.0, false).add_token(
            "world".to_string(),
            2,
            1.0,
            false,
        );

        builder.build_to_file(&file_path).unwrap();

        // Load and test tokenizer interface
        let tokenizer = ZeroCopyTokenizer::from_file(&file_path).unwrap();

        let encoded = tokenizer.encode("hello world").unwrap();
        assert_eq!(encoded.input_ids, vec![1, 2]);
        // Test that we can get the tokens back using id_to_token
        let tokens: Vec<String> =
            encoded.input_ids.iter().map(|&id| tokenizer.id_to_token(id).unwrap()).collect();
        assert_eq!(tokens, vec!["hello", "world"]);

        let decoded = tokenizer.decode(&[1, 2]).unwrap();
        assert_eq!(decoded, "hello world");

        let vocab = tokenizer.get_vocab();
        assert_eq!(vocab.len(), 2);
        assert_eq!(vocab.get("hello"), Some(&1));
        assert_eq!(vocab.get("world"), Some(&2));
    }

    #[test]
    fn test_memory_stats() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_stats.zc");

        let mut builder = ZeroCopyBuilder::new();
        builder
            .add_token("test".to_string(), 1, 1.0, false)
            .add_token("memory".to_string(), 2, 1.0, false)
            .set_metadata(b"test metadata".to_vec())
            .set_special_tokens(b"special".to_vec());

        builder.build_to_file(&file_path).unwrap();

        let tokenizer = ZeroCopyTokenizer::from_file(&file_path).unwrap();
        let stats = tokenizer.memory_stats();

        assert!(stats.file_size > 0);
        assert!(stats.vocab_size > 0);
        assert_eq!(stats.metadata_size, 13); // "test metadata".len()
        assert_eq!(stats.special_tokens_size, 7); // "special".len()
        assert!(stats.lookup_table_size > 0);
    }

    #[test]
    fn test_integrity_verification() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_integrity.zc");

        let mut builder = ZeroCopyBuilder::new();
        builder.add_token("integrity".to_string(), 1, 1.0, false).add_token(
            "check".to_string(),
            2,
            1.0,
            false,
        );

        builder.build_to_file(&file_path).unwrap();

        let tokenizer = ZeroCopyTokenizer::from_file(&file_path).unwrap();
        assert!(tokenizer.verify_integrity().unwrap());
    }

    #[test]
    fn test_utils_file_info() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_info.zc");

        let mut builder = ZeroCopyBuilder::new();
        builder.add_token("info".to_string(), 1, 1.0, false).add_token(
            "test".to_string(),
            2,
            1.0,
            false,
        );

        builder.build_to_file(&file_path).unwrap();

        let info = ZeroCopyUtils::get_file_info(&file_path).unwrap();

        assert_eq!(info.get("format").unwrap(), "ZeroCopy");
        assert_eq!(info.get("vocab_size").unwrap(), "2");
        assert!(info.contains_key("file_size"));
    }

    #[test]
    fn test_utils_validation() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_validation.zc");

        let mut builder = ZeroCopyBuilder::new();
        builder.add_token("validate".to_string(), 1, 1.0, false);

        builder.build_to_file(&file_path).unwrap();

        assert!(ZeroCopyUtils::validate_file(&file_path).unwrap());
    }

    #[test]
    fn test_builder_from_map() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_from_map.zc");

        let mut vocab = TestHashMap::new();
        vocab.insert("from".to_string(), 1);
        vocab.insert("map".to_string(), 2);
        vocab.insert("test".to_string(), 3);

        let mut builder = ZeroCopyBuilder::new();
        builder.add_tokens_from_map(&vocab);

        builder.build_to_file(&file_path).unwrap();

        let tokenizer = ZeroCopyTokenizer::from_file(&file_path).unwrap();
        assert_eq!(tokenizer.vocab_size(), 3);

        for (token, &expected_id) in &vocab {
            assert_eq!(tokenizer.get_id_unchecked(token), Some(expected_id));
        }
    }
}
