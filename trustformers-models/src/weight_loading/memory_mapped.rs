/// Memory-Mapped Weight Loader
///
/// This module provides memory-mapped file support for efficient weight loading without
/// loading entire files into memory.
use std::fs::File;
use std::path::Path;
use trustformers_core::{
    errors::{invalid_format, Result, TrustformersError},
    tensor::Tensor,
};

use super::config::WeightDataType;
use super::huggingface::{SafeTensorsHeader, TensorMetadata, WeightLoader};

/// Memory-mapped weight loader for efficient access to large weight files
pub struct MemoryMappedLoader {
    #[allow(dead_code)]
    file: File,
    mapping: Option<memmap2::Mmap>,
    header: SafeTensorsHeader,
}

impl MemoryMappedLoader {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let mapping = unsafe { memmap2::Mmap::map(&file)? };

        // Parse header from memory map
        let header = Self::parse_header_from_mmap(&mapping)?;

        Ok(Self {
            file,
            mapping: Some(mapping),
            header,
        })
    }

    fn parse_header_from_mmap(mmap: &[u8]) -> Result<SafeTensorsHeader> {
        // Read header length
        let header_len = u64::from_le_bytes([
            mmap[0], mmap[1], mmap[2], mmap[3], mmap[4], mmap[5], mmap[6], mmap[7],
        ]);

        // Parse header JSON
        let header_bytes = &mmap[8..8 + header_len as usize];
        let header_str = std::str::from_utf8(header_bytes).map_err(|e| {
            TrustformersError::weight_load_error(format!(
                "Invalid UTF-8 in SafeTensors header: {}",
                e
            ))
        })?;

        serde_json::from_str(header_str).map_err(|e| {
            TrustformersError::serialization_error(format!(
                "Failed to parse SafeTensors header: {}",
                e
            ))
        })
    }

    fn mmap_bytes_to_tensor(&self, data: &[u8], dtype: &str, shape: &[usize]) -> Result<Tensor> {
        match dtype {
            "F32" => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_vec(floats, shape)
            },
            "F16" => {
                let floats: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect();
                Tensor::from_vec(floats, shape)
            },
            _ => Err(invalid_format(
                "data type",
                format!("Unsupported dtype: {}", dtype),
            )),
        }
    }
}

impl WeightLoader for MemoryMappedLoader {
    fn load_tensor(&mut self, name: &str) -> Result<Tensor> {
        if let Some(tensor_info) = self.header.tensors.get(name) {
            if let Some(ref mapping) = self.mapping {
                let start = tensor_info.data_offsets[0] as usize;
                let end = tensor_info.data_offsets[1] as usize;
                let data = &mapping[start..end];

                self.mmap_bytes_to_tensor(data, &tensor_info.dtype, &tensor_info.shape)
            } else {
                Err(TrustformersError::invalid_state(
                    "No memory mapping".to_string(),
                ))
            }
        } else {
            Err(TrustformersError::runtime_error(format!(
                "Tensor not found: {}",
                name
            )))
        }
    }

    fn list_tensors(&self) -> Result<Vec<String>> {
        Ok(self.header.tensors.keys().cloned().collect())
    }

    fn tensor_info(&self, name: &str) -> Result<Option<TensorMetadata>> {
        if let Some(tensor_info) = self.header.tensors.get(name) {
            let dtype = match tensor_info.dtype.as_str() {
                "F32" => WeightDataType::Float32,
                "F16" => WeightDataType::Float16,
                "I8" => WeightDataType::Int8,
                _ => WeightDataType::Float32,
            };

            Ok(Some(TensorMetadata {
                shape: tensor_info.shape.clone(),
                dtype,
                size_bytes: tensor_info.data_offsets[1] - tensor_info.data_offsets[0],
                offset: tensor_info.data_offsets[0],
            }))
        } else {
            Ok(None)
        }
    }

    fn close(&mut self) -> Result<()> {
        self.mapping.take();
        Ok(())
    }
}
