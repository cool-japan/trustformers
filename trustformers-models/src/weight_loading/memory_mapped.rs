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
    /// File offset of the tensor data section: `8 + header_len`.
    ///
    /// safetensors `data_offsets` are relative to this point, not to the start
    /// of the file. A previous revision sliced the mapping with the raw offsets,
    /// so every tensor was read `8 + header_len` bytes too early — that is, from
    /// inside the JSON header — and the values were silently wrong.
    data_start: usize,
}

impl MemoryMappedLoader {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let mapping = unsafe { memmap2::Mmap::map(&file)? };

        // Parse header from memory map
        let (header, data_start) = Self::parse_header_from_mmap(&mapping)?;

        Ok(Self {
            file,
            mapping: Some(mapping),
            header,
            data_start,
        })
    }

    /// Parse the safetensors header, returning it with the data-section offset.
    fn parse_header_from_mmap(mmap: &[u8]) -> Result<(SafeTensorsHeader, usize)> {
        if mmap.len() < 8 {
            return Err(TrustformersError::weight_load_error(format!(
                "file is {} byte(s) long; too small to hold a SafeTensors header",
                mmap.len()
            )));
        }

        // Read header length
        let header_len = u64::from_le_bytes([
            mmap[0], mmap[1], mmap[2], mmap[3], mmap[4], mmap[5], mmap[6], mmap[7],
        ]) as usize;

        let data_start =
            8usize.checked_add(header_len).filter(|end| *end <= mmap.len()).ok_or_else(|| {
                TrustformersError::weight_load_error(format!(
                    "SafeTensors header claims {header_len} bytes but the file holds only {}",
                    mmap.len()
                ))
            })?;

        // Parse header JSON
        let header_bytes = &mmap[8..data_start];
        let header_str = std::str::from_utf8(header_bytes).map_err(|e| {
            TrustformersError::weight_load_error(format!(
                "Invalid UTF-8 in SafeTensors header: {}",
                e
            ))
        })?;

        let header = serde_json::from_str(header_str).map_err(|e| {
            TrustformersError::serialization_error(format!(
                "Failed to parse SafeTensors header: {}",
                e
            ))
        })?;
        Ok((header, data_start))
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
            "BF16" => {
                let floats: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::bf16::from_bits(bits).to_f32()
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
                let start = self.data_start + tensor_info.data_offsets[0] as usize;
                let end = self.data_start + tensor_info.data_offsets[1] as usize;
                if end > mapping.len() || start > end {
                    return Err(TrustformersError::weight_load_error(format!(
                        "tensor {name} declares bytes {start}..{end} but the file is {} bytes long",
                        mapping.len()
                    )));
                }
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
                "BF16" => WeightDataType::BFloat16,
                "I8" | "U8" => WeightDataType::Int8,
                // Reporting an unknown dtype as Float32 would understate the
                // element width and mislead every size calculation downstream.
                other => {
                    return Err(invalid_format(
                        "data type",
                        format!("Unsupported dtype {other} for tensor {name}"),
                    ))
                },
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_loading::test_support::{build_safetensors, write_temp_file, F32Tensor};

    struct TempFile {
        path: std::path::PathBuf,
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[test]
    fn memory_mapped_loader_reads_tensors_from_the_data_section() {
        // Regression: `data_offsets` are relative to the start of the tensor data
        // section, so slicing the raw mapping with them read the JSON header
        // instead of the weights and returned plausible-looking garbage.
        let tensors = vec![
            F32Tensor::new("a", &[2, 2], vec![1.5, -2.5, 3.5, -4.5]),
            F32Tensor::new("b", &[3], vec![10.0, 20.0, 30.0]),
        ];
        let file = TempFile {
            path: write_temp_file("mmap", "safetensors", &build_safetensors(&tensors)),
        };

        let mut loader = MemoryMappedLoader::new(&file.path).expect("mapping must open");
        match loader.load_tensor("a").expect("tensor must load") {
            Tensor::F32(arr) => {
                assert_eq!(arr.shape(), &[2, 2]);
                assert_eq!(
                    arr.iter().copied().collect::<Vec<f32>>(),
                    vec![1.5, -2.5, 3.5, -4.5]
                );
            },
            other => panic!("expected an F32 tensor, got {other:?}"),
        }
        match loader.load_tensor("b").expect("tensor must load") {
            Tensor::F32(arr) => {
                assert_eq!(
                    arr.iter().copied().collect::<Vec<f32>>(),
                    vec![10.0, 20.0, 30.0]
                );
            },
            other => panic!("expected an F32 tensor, got {other:?}"),
        }
    }

    #[test]
    fn memory_mapped_loader_rejects_a_truncated_file() {
        let file = TempFile {
            path: write_temp_file("mmap_short", "safetensors", &[0u8; 4]),
        };
        let result = MemoryMappedLoader::new(&file.path);
        assert!(
            result.is_err(),
            "a 4-byte file cannot hold a SafeTensors header"
        );
    }

    #[test]
    fn memory_mapped_loader_reports_real_metadata() {
        let tensors = vec![F32Tensor::new(
            "w",
            &[4, 2],
            (0..8).map(|i| i as f32).collect(),
        )];
        let file = TempFile {
            path: write_temp_file("mmap_meta", "safetensors", &build_safetensors(&tensors)),
        };
        let loader = MemoryMappedLoader::new(&file.path).expect("mapping must open");
        let info = loader
            .tensor_info("w")
            .expect("metadata lookup must succeed")
            .expect("tensor must exist");
        assert_eq!(info.shape, vec![4, 2]);
        assert_eq!(info.size_bytes, 32);
    }
}
