//! Reading pretrained weights from checkpoint files.
//!
//! Two formats are supported end to end:
//!
//! * **safetensors** via [`SafeTensorsReader`];
//! * **PyTorch** (`.bin`, `.pt`, `.pth`) via [`PyTorchReader`], which reads the real
//!   ZIP + pickle checkpoint with the pure-Rust implementation in
//!   [`zip_archive`], [`pickle`] and [`torch`].
//!
//! An earlier revision of `PyTorchReader` never parsed anything: it ran
//! `String::from_utf8_lossy` over the file, looked for fifteen hard-coded substrings
//! such as `"attention.self.query.weight"`, and handed back all-zero tensors with
//! guessed BERT-base dimensions — or, when even that failed, a fabricated ten-tensor
//! BERT state dict. Loading a checkpoint silently produced a zeroed model. All of
//! that is gone.

#[path = "weight_loading/pickle.rs"]
pub mod pickle;
#[path = "weight_loading/torch.rs"]
pub mod torch;
#[path = "weight_loading/zip_archive.rs"]
pub mod zip_archive;

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use crate::traits::WeightReader;
use safetensors::{SafeTensors, View};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Convert IEEE 754 half-precision (F16) to single-precision (F32)
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x3ff;

    // Handle special cases
    if exponent == 0 {
        if fraction == 0 {
            // Zero
            return if sign == 1 { -0.0 } else { 0.0 };
        } else {
            // Subnormal number
            let f = (fraction as f32) / 1024.0;
            let result = f * 2.0f32.powi(-14);
            return if sign == 1 { -result } else { result };
        }
    } else if exponent == 31 {
        // Infinity or NaN
        return if fraction == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }

    // Normal number
    let f = 1.0 + (fraction as f32) / 1024.0;
    let result = f * 2.0f32.powi((exponent as i32) - 15);
    if sign == 1 {
        -result
    } else {
        result
    }
}

pub struct SafeTensorsReader {
    data: Vec<u8>,
    tensors: HashMap<String, TensorInfo>,
}

#[derive(Debug)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    #[allow(dead_code)] // Reserved for future streaming/partial loading features
    data_offsets: (usize, usize),
}

impl SafeTensorsReader {
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| TrustformersError::safe_tensors_error(e.to_string()))?;
        let mut tensor_map = HashMap::new();

        for (name, tensor_view) in tensors.tensors() {
            let info = TensorInfo {
                dtype: format!("{:?}", tensor_view.dtype()),
                shape: tensor_view.shape().to_vec(),
                data_offsets: (0, tensor_view.data_len()),
            };
            tensor_map.insert(name.to_string(), info);
        }

        Ok(Self {
            data,
            tensors: tensor_map,
        })
    }
}

impl WeightReader for SafeTensorsReader {
    fn read_tensor(&mut self, name: &str) -> Result<Tensor> {
        let info = self.tensors.get(name).ok_or_else(|| {
            TrustformersError::weight_load_error(format!("Tensor {} not found", name))
        })?;

        let tensors = SafeTensors::deserialize(&self.data)
            .map_err(|e| TrustformersError::safe_tensors_error(e.to_string()))?;
        let tensor_view = tensors
            .tensor(name)
            .map_err(|e| TrustformersError::safe_tensors_error(e.to_string()))?;

        match info.dtype.as_str() {
            "F32" => {
                let data = tensor_view.data();
                let values: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                let arr = ArrayD::from_shape_vec(IxDyn(&info.shape), values)
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(Tensor::F32(arr))
            },
            "F16" => {
                // Convert F16 to F32
                let data = tensor_view.data();
                let values: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f16_to_f32(bits)
                    })
                    .collect();

                let arr = ArrayD::from_shape_vec(IxDyn(&info.shape), values)
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(Tensor::F32(arr))
            },
            _ => Err(TrustformersError::weight_load_error(format!(
                "Unsupported dtype: {}",
                info.dtype
            ))),
        }
    }

    fn list_tensors(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
}

/// Reader for PyTorch checkpoints (`.pt`, `.bin`, `.pth`).
///
/// The file is parsed for real: the ZIP central directory is walked, `data.pkl` is
/// interpreted by a pickle virtual machine that records rather than executes the
/// `torch._utils._rebuild_tensor_v2` calls it finds, and each tensor is
/// reconstructed from its storage record with the checkpoint's own dtype, shape and
/// bytes.
///
/// # Errors
///
/// Construction fails, with an explanatory message, when the file is not a PyTorch
/// checkpoint, uses the legacy pre-1.6 non-ZIP layout, has deflate-compressed
/// members, contains no tensors, or holds a non-contiguous tensor view.
#[derive(Debug)]
pub struct PyTorchReader {
    state_dict: torch::TorchStateDict,
}

impl PyTorchReader {
    /// Parse a PyTorch checkpoint from disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        let state_dict = torch::read_torch_file(path).map_err(|e| {
            TrustformersError::weight_load_error(format!("failed to read PyTorch file: {e}"))
        })?;
        Ok(Self { state_dict })
    }

    /// Parse a PyTorch checkpoint held in memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let state_dict = torch::read_torch_bytes(bytes).map_err(|e| {
            TrustformersError::weight_load_error(format!("failed to read PyTorch data: {e}"))
        })?;
        Ok(Self { state_dict })
    }

    /// The parsed state dict.
    pub fn state_dict(&self) -> &torch::TorchStateDict {
        &self.state_dict
    }

    /// The element type a tensor had in the checkpoint.
    pub fn dtype_of(&self, name: &str) -> Option<torch::TorchDType> {
        self.state_dict.get(name).map(|tensor| tensor.dtype)
    }
}

impl WeightReader for PyTorchReader {
    fn read_tensor(&mut self, name: &str) -> Result<Tensor> {
        self.state_dict.get(name).map(|record| record.tensor.clone()).ok_or_else(|| {
            TrustformersError::weight_load_error(format!(
                "Tensor {name} not found in the checkpoint"
            ))
        })
    }

    fn list_tensors(&self) -> Vec<String> {
        self.state_dict.names()
    }
}

/// Report from [`WeightLoader::load_weights_into_model`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WeightLoadReport {
    /// Parameters whose checkpoint tensor was found and copied in.
    pub loaded: Vec<String>,
    /// Parameters the checkpoint had no tensor for.
    pub missing: Vec<String>,
    /// Checkpoint tensors that matched no model parameter.
    pub unused: Vec<String>,
}

impl WeightLoadReport {
    /// Whether every model parameter was filled from the checkpoint.
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
    }
}

/// Helpers for loading checkpoints into models.
pub struct WeightLoader;

impl WeightLoader {
    /// Copy a checkpoint's tensors into a model's live parameters.
    ///
    /// Parameters are matched by the names the model reports through
    /// [`Model::named_tensors_mut`](crate::traits::Model::named_tensors_mut).
    /// Every copy is shape-checked, so a mismatched checkpoint fails instead of
    /// leaving the model half-initialised.
    ///
    /// A previous revision serialised the tensors into an invented JSON envelope
    /// and handed that to `Model::load_pretrained`, which no model implementation
    /// could parse; the weights never reached the model.
    ///
    /// # Errors
    ///
    /// * The model exposes no named parameters (`named_tensors_mut` not implemented).
    /// * A checkpoint tensor's shape differs from the parameter it matches.
    /// * A tensor cannot be read from the checkpoint.
    pub fn load_weights_into_model<M>(
        model: &mut M,
        reader: &mut dyn WeightReader,
    ) -> Result<WeightLoadReport>
    where
        M: crate::traits::Model,
    {
        let available: Vec<String> = reader.list_tensors();
        let mut loaded_tensors: HashMap<String, Tensor> = HashMap::with_capacity(available.len());
        for name in &available {
            loaded_tensors.insert(name.clone(), reader.read_tensor(name)?);
        }

        let parameter_names: Vec<String> =
            model.named_tensors().into_iter().map(|(name, _)| name).collect();
        if parameter_names.is_empty() {
            return Err(TrustformersError::weight_load_error(
                "the model exposes no named parameters: `Model::named_tensors_mut` is not \
                 implemented for this type, so there is nowhere to put the checkpoint's tensors"
                    .to_string(),
            ));
        }

        let mut report = WeightLoadReport::default();
        for (name, parameter) in model.named_tensors_mut() {
            match loaded_tensors.get(&name) {
                Some(source) => {
                    if source.shape() != parameter.shape() {
                        return Err(TrustformersError::weight_load_error(format!(
                            "checkpoint tensor '{name}' has shape {:?} but the model parameter \
                             has shape {:?}",
                            source.shape(),
                            parameter.shape()
                        )));
                    }
                    *parameter = source.clone();
                    report.loaded.push(name);
                },
                None => report.missing.push(name),
            }
        }

        let matched: std::collections::HashSet<&String> = report.loaded.iter().collect();
        report.unused = available.into_iter().filter(|name| !matched.contains(name)).collect();

        report.loaded.sort();
        report.missing.sort();
        report.unused.sort();
        Ok(report)
    }

    /// Load weights from a SafeTensors file
    pub fn load_from_safetensors<P: AsRef<Path>>(path: P) -> Result<SafeTensorsReader> {
        SafeTensorsReader::from_file(path.as_ref())
    }

    /// List all available tensors in a SafeTensors file
    pub fn list_tensors_in_file<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
        let reader = SafeTensorsReader::from_file(path.as_ref())?;
        Ok(reader.list_tensors())
    }

    /// Load a specific tensor from a SafeTensors file
    pub fn load_tensor_from_file<P: AsRef<Path>>(path: P, tensor_name: &str) -> Result<Tensor> {
        let mut reader = SafeTensorsReader::from_file(path.as_ref())?;
        reader.read_tensor(tensor_name)
    }

    /// Load weights from a PyTorch file
    pub fn load_from_pytorch<P: AsRef<Path>>(path: P) -> Result<PyTorchReader> {
        PyTorchReader::from_file(path.as_ref())
    }

    /// List all available tensors in a PyTorch file
    pub fn list_tensors_in_pytorch_file<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
        let reader = PyTorchReader::from_file(path.as_ref())?;
        Ok(reader.list_tensors())
    }

    /// Load a specific tensor from a PyTorch file
    pub fn load_tensor_from_pytorch_file<P: AsRef<Path>>(
        path: P,
        tensor_name: &str,
    ) -> Result<Tensor> {
        let mut reader = PyTorchReader::from_file(path.as_ref())?;
        reader.read_tensor(tensor_name)
    }

    /// Auto-detect format and load weights
    pub fn load_weights_auto<P: AsRef<Path>>(path: P) -> Result<Box<dyn WeightReader>> {
        let path = path.as_ref();

        // Determine format by file extension
        if let Some(extension) = path.extension() {
            match extension.to_str().unwrap_or("").to_lowercase().as_str() {
                "safetensors" => {
                    let reader = SafeTensorsReader::from_file(path)?;
                    Ok(Box::new(reader))
                },
                "pt" | "pth" | "bin" => {
                    let reader = PyTorchReader::from_file(path)?;
                    Ok(Box::new(reader))
                },
                _ => Err(TrustformersError::weight_load_error(format!(
                    "Unsupported file format: {}",
                    extension.to_string_lossy()
                ))),
            }
        } else {
            Err(TrustformersError::weight_load_error(
                "Unable to determine file format from extension".into(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::torch::fixture::{build_checkpoint, FixtureTensor};
    use super::*;
    use crate::traits::{Config, Model};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    struct TinyConfig;

    impl Config for TinyConfig {
        fn architecture(&self) -> &'static str {
            "tiny"
        }
    }

    /// A model with two real parameters, so weight loading has somewhere to write.
    struct TinyModel {
        config: TinyConfig,
        weight: Tensor,
        bias: Tensor,
    }

    impl TinyModel {
        fn zeros() -> Self {
            Self {
                config: TinyConfig,
                weight: Tensor::zeros(&[2, 3]).expect("weight"),
                bias: Tensor::zeros(&[2]).expect("bias"),
            }
        }
    }

    impl Model for TinyModel {
        type Config = TinyConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, input: Self::Input) -> Result<Self::Output> {
            Ok(input)
        }

        fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
            Ok(())
        }

        fn get_config(&self) -> &Self::Config {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            self.weight.len() + self.bias.len()
        }

        fn named_tensors(&self) -> Vec<(String, &Tensor)> {
            vec![
                ("weight".to_string(), &self.weight),
                ("bias".to_string(), &self.bias),
            ]
        }

        fn named_tensors_mut(&mut self) -> Vec<(String, &mut Tensor)> {
            vec![
                ("weight".to_string(), &mut self.weight),
                ("bias".to_string(), &mut self.bias),
            ]
        }
    }

    /// A model that does not implement the enumeration API.
    struct OpaqueModel {
        config: TinyConfig,
    }

    impl Model for OpaqueModel {
        type Config = TinyConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, input: Self::Input) -> Result<Self::Output> {
            Ok(input)
        }

        fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
            Ok(())
        }

        fn get_config(&self) -> &Self::Config {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            0
        }
    }

    fn checkpoint_bytes(weight: &[f32], bias: &[f32]) -> Vec<u8> {
        build_checkpoint(
            "archive",
            &[
                FixtureTensor {
                    name: "weight".to_string(),
                    storage_class: "FloatStorage",
                    storage_key: "0".to_string(),
                    bytes: weight.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    element_count: weight.len(),
                    storage_offset: 0,
                    shape: vec![2, 3],
                    stride: vec![3, 1],
                },
                FixtureTensor {
                    name: "bias".to_string(),
                    storage_class: "FloatStorage",
                    storage_key: "1".to_string(),
                    bytes: bias.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    element_count: bias.len(),
                    storage_offset: 0,
                    shape: vec![2],
                    stride: vec![1],
                },
            ],
        )
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("trustformers_weight_loading_tests");
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir.join(name)
    }

    /// Regression test: the old reader returned zeros with guessed BERT shapes.
    #[test]
    fn pytorch_reader_returns_the_checkpoints_real_values() {
        let weight: Vec<f32> = (0..6).map(|i| i as f32 * 0.5 - 1.0).collect();
        let bias = vec![2.0f32, -3.0];
        let path = temp_path("real_values.bin");
        std::fs::write(&path, checkpoint_bytes(&weight, &bias)).expect("write");

        let mut reader = PyTorchReader::from_file(&path).expect("read");
        let mut names = reader.list_tensors();
        names.sort();
        assert_eq!(names, vec!["bias".to_string(), "weight".to_string()]);

        let loaded = reader.read_tensor("weight").expect("weight");
        assert_eq!(loaded.shape(), vec![2, 3]);
        assert_eq!(loaded.to_vec_f32().expect("f32"), weight);
        assert!(
            loaded.to_vec_f32().expect("f32").iter().any(|v| *v != 0.0),
            "the old reader returned all zeros"
        );

        assert!(reader.read_tensor("nonexistent").is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn pytorch_reader_output_depends_on_the_file() {
        let path_a = temp_path("varies_a.bin");
        let path_b = temp_path("varies_b.bin");
        std::fs::write(&path_a, checkpoint_bytes(&[1.0; 6], &[1.0; 2])).expect("write");
        std::fs::write(&path_b, checkpoint_bytes(&[9.0; 6], &[9.0; 2])).expect("write");

        let mut a = PyTorchReader::from_file(&path_a).expect("read");
        let mut b = PyTorchReader::from_file(&path_b).expect("read");
        assert_ne!(
            a.read_tensor("weight").expect("a").to_vec_f32().expect("f32"),
            b.read_tensor("weight").expect("b").to_vec_f32().expect("f32")
        );

        let _ = std::fs::remove_file(&path_a);
        let _ = std::fs::remove_file(&path_b);
    }

    /// Regression test: the old reader invented a BERT state dict for any file.
    #[test]
    fn pytorch_reader_refuses_files_that_are_not_checkpoints() {
        let path = temp_path("not_a_checkpoint.bin");
        std::fs::write(&path, b"this file mentions state_dict and weight and bias").expect("write");

        let err = PyTorchReader::from_file(&path).expect_err("must not invent a state dict");
        let message = err.to_string();
        assert!(
            !message.is_empty() && message.contains("PyTorch"),
            "expected an explanatory error, got: {message}"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn pytorch_reader_records_the_source_dtype() {
        let path = temp_path("dtype.bin");
        std::fs::write(&path, checkpoint_bytes(&[0.0; 6], &[0.0; 2])).expect("write");

        let reader = PyTorchReader::from_file(&path).expect("read");
        assert_eq!(reader.dtype_of("weight"), Some(torch::TorchDType::F32));
        assert_eq!(reader.dtype_of("absent"), None);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_weights_into_model_actually_writes_the_parameters() {
        let weight: Vec<f32> = (0..6).map(|i| i as f32 + 1.0).collect();
        let bias = vec![7.0f32, 8.0];
        let path = temp_path("into_model.bin");
        std::fs::write(&path, checkpoint_bytes(&weight, &bias)).expect("write");

        let mut model = TinyModel::zeros();
        assert!(model.weight.to_vec_f32().expect("f32").iter().all(|v| *v == 0.0));

        let mut reader = PyTorchReader::from_file(&path).expect("read");
        let report = WeightLoader::load_weights_into_model(&mut model, &mut reader).expect("load");

        assert!(report.is_complete(), "missing: {:?}", report.missing);
        assert_eq!(
            report.loaded,
            vec!["bias".to_string(), "weight".to_string()]
        );
        assert!(report.unused.is_empty());
        assert_eq!(model.weight.to_vec_f32().expect("f32"), weight);
        assert_eq!(model.bias.to_vec_f32().expect("f32"), bias);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_weights_into_model_rejects_a_model_without_named_parameters() {
        let path = temp_path("opaque.bin");
        std::fs::write(&path, checkpoint_bytes(&[0.0; 6], &[0.0; 2])).expect("write");

        let mut model = OpaqueModel { config: TinyConfig };
        let mut reader = PyTorchReader::from_file(&path).expect("read");
        let err = WeightLoader::load_weights_into_model(&mut model, &mut reader)
            .expect_err("nowhere to load into");
        assert!(err.to_string().contains("named_tensors_mut"), "{err}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_weights_into_model_rejects_shape_mismatches() {
        // A checkpoint whose "weight" is [3, 2] rather than the model's [2, 3].
        let values: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let bytes = build_checkpoint(
            "archive",
            &[FixtureTensor {
                name: "weight".to_string(),
                storage_class: "FloatStorage",
                storage_key: "0".to_string(),
                bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                element_count: 6,
                storage_offset: 0,
                shape: vec![3, 2],
                stride: vec![2, 1],
            }],
        );
        let path = temp_path("shape_mismatch.bin");
        std::fs::write(&path, bytes).expect("write");

        let mut model = TinyModel::zeros();
        let mut reader = PyTorchReader::from_file(&path).expect("read");
        let err = WeightLoader::load_weights_into_model(&mut model, &mut reader)
            .expect_err("shape mismatch");
        assert!(err.to_string().contains("shape"), "{err}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_weights_into_model_reports_missing_and_unused_tensors() {
        let values: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let bytes = build_checkpoint(
            "archive",
            &[
                FixtureTensor {
                    name: "weight".to_string(),
                    storage_class: "FloatStorage",
                    storage_key: "0".to_string(),
                    bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    element_count: 6,
                    storage_offset: 0,
                    shape: vec![2, 3],
                    stride: vec![3, 1],
                },
                FixtureTensor {
                    name: "extra".to_string(),
                    storage_class: "FloatStorage",
                    storage_key: "1".to_string(),
                    bytes: vec![0u8; 4],
                    element_count: 1,
                    storage_offset: 0,
                    shape: vec![1],
                    stride: vec![1],
                },
            ],
        );
        let path = temp_path("partial.bin");
        std::fs::write(&path, bytes).expect("write");

        let mut model = TinyModel::zeros();
        let mut reader = PyTorchReader::from_file(&path).expect("read");
        let report = WeightLoader::load_weights_into_model(&mut model, &mut reader).expect("load");

        assert_eq!(report.loaded, vec!["weight".to_string()]);
        assert_eq!(report.missing, vec!["bias".to_string()]);
        assert_eq!(report.unused, vec!["extra".to_string()]);
        assert!(!report.is_complete());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_weights_auto_dispatches_on_the_extension() {
        let path = temp_path("auto.bin");
        std::fs::write(&path, checkpoint_bytes(&[1.0; 6], &[1.0; 2])).expect("write");

        let reader = WeightLoader::load_weights_auto(&path).expect("auto");
        assert_eq!(reader.list_tensors().len(), 2);

        let unknown = temp_path("auto.unknown");
        std::fs::write(&unknown, b"x").expect("write");
        assert!(WeightLoader::load_weights_auto(&unknown).is_err());

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&unknown);
    }
}
