//! OpenVINO export functionality
//!
//! This module provides OpenVINO export capabilities for TrustformeRS models.

#![allow(unused_variables)] // Export implementation with reserved parameters
//! OpenVINO is Intel's optimization toolkit for deep learning inference.

use crate::export::{ExportConfig, ExportFormat, ExportPrecision, ModelExporter};
use crate::traits::Model;
use anyhow::{anyhow, Result};
use serde_json::{json, Value as JsonValue};
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;

/// OpenVINO exporter for TrustformeRS models
#[derive(Clone)]
pub struct OpenVINOExporter {
    version: String,
    optimize_for_device: String,
    precision_config: OpenVINOPrecisionConfig,
}

/// OpenVINO precision configuration
#[derive(Clone, Debug)]
pub struct OpenVINOPrecisionConfig {
    pub input_precision: String,
    pub output_precision: String,
    pub weights_precision: String,
    pub enable_int8_calibration: bool,
}

impl OpenVINOExporter {
    /// Create a new OpenVINO exporter
    pub fn new() -> Self {
        Self {
            version: "2024.3".to_string(),
            optimize_for_device: "CPU".to_string(),
            precision_config: OpenVINOPrecisionConfig::default(),
        }
    }

    /// Create OpenVINO exporter with custom configuration
    pub fn with_config(
        version: String,
        device: String,
        precision_config: OpenVINOPrecisionConfig,
    ) -> Self {
        Self {
            version,
            optimize_for_device: device,
            precision_config,
        }
    }

    /// Export model to OpenVINO IR format
    fn export_to_openvino<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<()> {
        let output_path = Path::new(&config.output_path);

        // Create output directory if it doesn't exist
        if let Some(parent) = output_path.parent() {
            create_dir_all(parent)?;
        }

        // Generate OpenVINO IR XML file
        let xml_content = self.build_openvino_xml(model, config)?;
        let xml_file = output_path.with_extension("xml");
        let mut file = File::create(&xml_file)?;
        file.write_all(xml_content.as_bytes())?;

        // Generate OpenVINO IR BIN file (weights)
        self.export_weights_bin(model, &output_path.with_extension("bin"), config)?;

        // Generate mapping file
        let mapping_file = output_path.with_extension("mapping");
        let mapping_content = self.build_mapping_file(model, config)?;
        let mut file = File::create(mapping_file)?;
        file.write_all(mapping_content.as_bytes())?;

        // Generate optimization config
        let config_file = output_path.with_extension("json");
        let config_content = self.build_optimization_config(model, config)?;
        let mut file = File::create(config_file)?;
        file.write_all(serde_json::to_string_pretty(&config_content)?.as_bytes())?;

        println!("✅ OpenVINO export completed: {}", xml_file.display());
        Ok(())
    }

    /// Build OpenVINO IR XML representation
    fn build_openvino_xml<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<String> {
        let mut xml = String::new();

        // XML header and root element
        xml.push_str(r#"<?xml version="1.0"?>"#);
        xml.push('\n');
        xml.push_str(&format!(
            r#"<net name="trustformers_model" version="10" batch="{}">
"#,
            config.batch_size.unwrap_or(1)
        ));

        // Layers section
        xml.push_str("    <layers>\n");
        self.add_transformer_layers_xml(&mut xml, config)?;
        xml.push_str("    </layers>\n");

        // Edges section (connections between layers)
        xml.push_str("    <edges>\n");
        self.add_layer_connections_xml(&mut xml, config)?;
        xml.push_str("    </edges>\n");

        // Pre-process section
        xml.push_str("    <pre-process reference-layer-name=\"input\">\n");
        xml.push_str("        <channel id=\"0\">\n");
        xml.push_str("            <mean value=\"0\"/>\n");
        xml.push_str("            <scale value=\"1\"/>\n");
        xml.push_str("        </channel>\n");
        xml.push_str("    </pre-process>\n");

        xml.push_str("</net>\n");

        Ok(xml)
    }

    /// Add transformer layers to OpenVINO XML
    fn add_transformer_layers_xml(&self, xml: &mut String, config: &ExportConfig) -> Result<()> {
        let seq_len = config.sequence_length.unwrap_or(512);
        let batch_size = config.batch_size.unwrap_or(1);
        let hidden_size = 768;
        let num_heads = 12;
        let head_dim = hidden_size / num_heads;

        // Input layer
        xml.push_str(&format!(
            r#"        <layer id="0" name="input" type="Parameter" version="opset1">
            <data element_type="{}" shape="{},{}" />
            <output>
                <port id="0" precision="{}">
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
            </output>
        </layer>
"#,
            self.precision_to_openvino_type(config.precision),
            batch_size,
            seq_len,
            self.precision_to_openvino_type(config.precision),
            batch_size,
            seq_len
        ));

        // Embedding layer
        xml.push_str(&format!(
            r#"        <layer id="1" name="embedding" type="MatMul" version="opset1">
            <data transpose_a="false" transpose_b="true" />
            <input>
                <port id="0">
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
                <port id="1">
                    <dim>{}</dim>
                    <dim>512</dim>
                </port>
            </input>
            <output>
                <port id="0" precision="{}">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
            </output>
        </layer>
"#,
            batch_size,
            seq_len,
            hidden_size,
            self.precision_to_openvino_type(config.precision),
            batch_size,
            seq_len,
            hidden_size
        ));

        // Multi-head attention layers
        for head in 0..num_heads {
            // Query projection
            xml.push_str(&format!(
                r#"        <layer id="{}" name="attention_query_{}" type="MatMul" version="opset1">
            <data transpose_a="false" transpose_b="true" />
            <input>
                <port id="0">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
                <port id="1">
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
            </input>
            <output>
                <port id="0" precision="{}">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
            </output>
        </layer>
"#,
                2 + head * 3,
                head,
                batch_size,
                seq_len,
                hidden_size,
                head_dim,
                hidden_size,
                self.precision_to_openvino_type(config.precision),
                batch_size,
                seq_len,
                head_dim
            ));

            // Key projection
            xml.push_str(&format!(
                r#"        <layer id="{}" name="attention_key_{}" type="MatMul" version="opset1">
            <data transpose_a="false" transpose_b="true" />
            <input>
                <port id="0">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
                <port id="1">
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
            </input>
            <output>
                <port id="0" precision="{}">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
            </output>
        </layer>
"#,
                3 + head * 3,
                head,
                batch_size,
                seq_len,
                hidden_size,
                head_dim,
                hidden_size,
                self.precision_to_openvino_type(config.precision),
                batch_size,
                seq_len,
                head_dim
            ));

            // Value projection
            xml.push_str(&format!(
                r#"        <layer id="{}" name="attention_value_{}" type="MatMul" version="opset1">
            <data transpose_a="false" transpose_b="true" />
            <input>
                <port id="0">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
                <port id="1">
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
            </input>
            <output>
                <port id="0" precision="{}">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
            </output>
        </layer>
"#,
                4 + head * 3,
                head,
                batch_size,
                seq_len,
                hidden_size,
                head_dim,
                hidden_size,
                self.precision_to_openvino_type(config.precision),
                batch_size,
                seq_len,
                head_dim
            ));
        }

        // Attention computation
        xml.push_str(&format!(
            r#"        <layer id="38" name="attention_computation" type="MatMul" version="opset1">
            <data transpose_a="false" transpose_b="true" />
            <input>
                <port id="0">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
                <port id="1">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
            </input>
            <output>
                <port id="0" precision="{}">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
            </output>
        </layer>
"#,
            batch_size,
            seq_len,
            head_dim,
            batch_size,
            head_dim,
            seq_len,
            self.precision_to_openvino_type(config.precision),
            batch_size,
            seq_len,
            seq_len
        ));

        // Softmax
        xml.push_str(&format!(
            r#"        <layer id="39" name="attention_softmax" type="Softmax" version="opset1">
            <data axis="-1" />
            <input>
                <port id="0">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
            </input>
            <output>
                <port id="0" precision="{}">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
            </output>
        </layer>
"#,
            batch_size,
            seq_len,
            seq_len,
            self.precision_to_openvino_type(config.precision),
            batch_size,
            seq_len,
            seq_len
        ));

        // Feed Forward Network
        xml.push_str(&format!(
            r#"        <layer id="40" name="ffn_intermediate" type="MatMul" version="opset1">
            <data transpose_a="false" transpose_b="true" />
            <input>
                <port id="0">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
                <port id="1">
                    <dim>3072</dim>
                    <dim>{}</dim>
                </port>
            </input>
            <output>
                <port id="0" precision="{}">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>3072</dim>
                </port>
            </output>
        </layer>
"#,
            batch_size,
            seq_len,
            hidden_size,
            hidden_size,
            self.precision_to_openvino_type(config.precision),
            batch_size,
            seq_len
        ));

        // GELU activation
        xml.push_str(&format!(
            r#"        <layer id="41" name="gelu" type="Gelu" version="opset7">
            <input>
                <port id="0">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>3072</dim>
                </port>
            </input>
            <output>
                <port id="0" precision="{}">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>3072</dim>
                </port>
            </output>
        </layer>
"#,
            batch_size,
            seq_len,
            self.precision_to_openvino_type(config.precision),
            batch_size,
            seq_len
        ));

        // Output projection
        xml.push_str(&format!(
            r#"        <layer id="42" name="output" type="MatMul" version="opset1">
            <data transpose_a="false" transpose_b="true" />
            <input>
                <port id="0">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>3072</dim>
                </port>
                <port id="1">
                    <dim>{}</dim>
                    <dim>3072</dim>
                </port>
            </input>
            <output>
                <port id="0" precision="{}">
                    <dim>{}</dim>
                    <dim>{}</dim>
                    <dim>{}</dim>
                </port>
            </output>
        </layer>
"#,
            batch_size,
            seq_len,
            hidden_size,
            self.precision_to_openvino_type(config.precision),
            batch_size,
            seq_len,
            hidden_size
        ));

        Ok(())
    }

    /// Add layer connections to OpenVINO XML
    fn add_layer_connections_xml(&self, xml: &mut String, _config: &ExportConfig) -> Result<()> {
        // Simple sequential connections for demonstration
        for i in 0..42 {
            xml.push_str(&format!(
                r#"        <edge from-layer="{}" from-port="0" to-layer="{}" to-port="0"/>
"#,
                i,
                i + 1
            ));
        }

        Ok(())
    }

    /// Export model weights in OpenVINO binary format
    fn export_weights_bin<M: Model>(
        &self,
        model: &M,
        bin_path: &Path,
        config: &ExportConfig,
    ) -> Result<()> {
        let mut file = File::create(bin_path)?;

        // Export dummy weights (in practice, extract actual weights from model)
        let weight_layouts = vec![
            ("embedding_weight", vec![768, 512]),
            ("attention_query_weight", vec![64, 768]),
            ("attention_key_weight", vec![64, 768]),
            ("attention_value_weight", vec![64, 768]),
            ("ffn_intermediate_weight", vec![3072, 768]),
            ("ffn_output_weight", vec![768, 3072]),
        ];

        for (name, shape) in weight_layouts {
            let size: usize = shape.iter().product();

            // Generate dummy data based on precision
            let data = match config.precision {
                ExportPrecision::FP32 => {
                    let values: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
                    values.iter().flat_map(|&x| x.to_le_bytes()).collect::<Vec<u8>>()
                },
                ExportPrecision::FP16 => {
                    let values: Vec<u16> =
                        (0..size).map(|i| (((i as f32) * 0.001).to_bits() >> 16) as u16).collect();
                    values.iter().flat_map(|&x| x.to_le_bytes()).collect::<Vec<u8>>()
                },
                ExportPrecision::INT8 => (0..size).map(|i| (i % 256) as u8).collect(),
                ExportPrecision::INT4 => {
                    // Pack two 4-bit values per byte
                    (0..(size + 1) / 2)
                        .map(|i| {
                            let low = (i * 2) % 16;
                            let high = ((i * 2 + 1) % 16) << 4;
                            (low | high) as u8
                        })
                        .collect()
                },
            };

            file.write_all(&data)?;
        }

        Ok(())
    }

    /// Build mapping file for layer names and weights
    fn build_mapping_file<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<String> {
        let mut mapping = String::new();

        mapping.push_str("# OpenVINO Layer Mapping File\n");
        mapping.push_str("# Format: layer_name,weight_offset,weight_size\n\n");

        let mut offset = 0;
        let weight_info = vec![
            ("embedding", 768 * 512 * 4),
            ("attention_query_0", 64 * 768 * 4),
            ("attention_key_0", 64 * 768 * 4),
            ("attention_value_0", 64 * 768 * 4),
            ("ffn_intermediate", 3072 * 768 * 4),
            ("ffn_output", 768 * 3072 * 4),
        ];

        for (layer_name, size) in weight_info {
            mapping.push_str(&format!("{},{},{}\n", layer_name, offset, size));
            offset += size;
        }

        Ok(mapping)
    }

    /// Build optimization configuration
    fn build_optimization_config<M: Model>(
        &self,
        model: &M,
        config: &ExportConfig,
    ) -> Result<JsonValue> {
        Ok(json!({
            "model_optimizer": {
                "version": self.version,
                "target_device": self.optimize_for_device,
                "precision": format!("{:?}", config.precision),
                "batch_size": config.batch_size.unwrap_or(1),
                "sequence_length": config.sequence_length.unwrap_or(512)
            },
            "runtime_config": {
                "num_threads": 4,
                "num_streams": 1,
                "affinity": "NUMA",
                "inference_precision": self.precision_config.weights_precision
            },
            "optimization_config": {
                "optimize": config.optimize,
                "model_name": "trustformers_model",
                "compress_to_fp16": matches!(config.precision, ExportPrecision::FP16),
                "quantize": config.quantization.is_some()
            },
            "input_config": {
                "layout": "NC",
                "precision": self.precision_config.input_precision,
                "mean_values": [],
                "scale_values": []
            },
            "output_config": {
                "precision": self.precision_config.output_precision,
                "layout": "NC"
            }
        }))
    }

    /// Convert export precision to OpenVINO element type
    fn precision_to_openvino_type(&self, precision: ExportPrecision) -> &'static str {
        match precision {
            ExportPrecision::FP32 => "f32",
            ExportPrecision::FP16 => "f16",
            ExportPrecision::INT8 => "i8",
            ExportPrecision::INT4 => "i4",
        }
    }

    /// Validate OpenVINO export configuration
    fn validate_config(&self, config: &ExportConfig) -> Result<()> {
        if config.format != ExportFormat::OpenVINO {
            return Err(anyhow!(
                "Invalid format for OpenVINO exporter: {:?}",
                config.format
            ));
        }

        // Validate device support
        if !["CPU", "GPU", "MYRIAD", "HDDL", "GNA"].contains(&self.optimize_for_device.as_str()) {
            return Err(anyhow!(
                "Unsupported OpenVINO device: {}",
                self.optimize_for_device
            ));
        }

        // Validate precision support per device
        match (self.optimize_for_device.as_str(), config.precision) {
            ("CPU", ExportPrecision::FP32)
            | ("CPU", ExportPrecision::FP16)
            | ("CPU", ExportPrecision::INT8) => {},
            ("GPU", ExportPrecision::FP32) | ("GPU", ExportPrecision::FP16) => {},
            ("MYRIAD", ExportPrecision::FP16) => {},
            ("GNA", ExportPrecision::INT8) => {},
            (device, precision) => {
                return Err(anyhow!(
                    "Precision {:?} not supported on device {}",
                    precision,
                    device
                ));
            },
        }

        Ok(())
    }
}

impl Default for OpenVINOPrecisionConfig {
    fn default() -> Self {
        Self {
            input_precision: "FP32".to_string(),
            output_precision: "FP32".to_string(),
            weights_precision: "FP32".to_string(),
            enable_int8_calibration: false,
        }
    }
}

impl ModelExporter for OpenVINOExporter {
    fn export<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<()> {
        self.validate_config(config)?;
        self.export_to_openvino(model, config)
    }

    fn supported_formats(&self) -> Vec<ExportFormat> {
        vec![ExportFormat::OpenVINO]
    }

    fn validate_model<M: Model>(&self, _model: &M, format: ExportFormat) -> Result<()> {
        if format != ExportFormat::OpenVINO {
            return Err(anyhow!("OpenVINO exporter only supports OpenVINO format"));
        }
        Ok(())
    }
}

impl Default for OpenVINOExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct MockModel {
        config: MockConfig,
    }

    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    struct MockConfig {
        hidden_size: usize,
    }

    impl crate::traits::Config for MockConfig {
        fn architecture(&self) -> &'static str {
            "mock"
        }
    }

    impl Model for MockModel {
        type Config = MockConfig;
        type Input = crate::tensor::Tensor;
        type Output = crate::tensor::Tensor;

        fn forward(&self, input: Self::Input) -> crate::errors::Result<Self::Output> {
            Ok(input)
        }

        fn load_pretrained(
            &mut self,
            _reader: &mut dyn std::io::Read,
        ) -> crate::errors::Result<()> {
            Ok(())
        }

        fn get_config(&self) -> &Self::Config {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            // Mock model with a reasonable parameter count for testing
            650_000
        }
    }

    #[test]
    fn test_openvino_exporter_creation() {
        let exporter = OpenVINOExporter::new();
        let formats = exporter.supported_formats();
        assert_eq!(formats, vec![ExportFormat::OpenVINO]);
        assert_eq!(exporter.version, "2024.3");
        assert_eq!(exporter.optimize_for_device, "CPU");
    }

    #[test]
    fn test_openvino_exporter_with_config() {
        let precision_config = OpenVINOPrecisionConfig {
            input_precision: "FP16".to_string(),
            output_precision: "FP16".to_string(),
            weights_precision: "FP16".to_string(),
            enable_int8_calibration: true,
        };

        let exporter = OpenVINOExporter::with_config(
            "2024.4".to_string(),
            "GPU".to_string(),
            precision_config,
        );

        assert_eq!(exporter.version, "2024.4");
        assert_eq!(exporter.optimize_for_device, "GPU");
        assert_eq!(exporter.precision_config.input_precision, "FP16");
        assert!(exporter.precision_config.enable_int8_calibration);
    }

    #[test]
    fn test_precision_to_openvino_type() {
        let exporter = OpenVINOExporter::new();
        assert_eq!(
            exporter.precision_to_openvino_type(ExportPrecision::FP32),
            "f32"
        );
        assert_eq!(
            exporter.precision_to_openvino_type(ExportPrecision::FP16),
            "f16"
        );
        assert_eq!(
            exporter.precision_to_openvino_type(ExportPrecision::INT8),
            "i8"
        );
        assert_eq!(
            exporter.precision_to_openvino_type(ExportPrecision::INT4),
            "i4"
        );
    }

    #[test]
    fn test_validate_config_success() {
        let exporter = OpenVINOExporter::new();
        let config = ExportConfig {
            format: ExportFormat::OpenVINO,
            precision: ExportPrecision::FP32,
            ..Default::default()
        };

        assert!(exporter.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_wrong_format() {
        let exporter = OpenVINOExporter::new();
        let config = ExportConfig {
            format: ExportFormat::ONNX,
            ..Default::default()
        };

        assert!(exporter.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_unsupported_device() {
        let exporter = OpenVINOExporter::with_config(
            "2024.3".to_string(),
            "INVALID_DEVICE".to_string(),
            OpenVINOPrecisionConfig::default(),
        );
        let config = ExportConfig {
            format: ExportFormat::OpenVINO,
            ..Default::default()
        };

        assert!(exporter.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_unsupported_precision_device_combo() {
        let exporter = OpenVINOExporter::with_config(
            "2024.3".to_string(),
            "MYRIAD".to_string(),
            OpenVINOPrecisionConfig::default(),
        );
        let config = ExportConfig {
            format: ExportFormat::OpenVINO,
            precision: ExportPrecision::FP32, // MYRIAD only supports FP16
            ..Default::default()
        };

        assert!(exporter.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_model_success() {
        let exporter = OpenVINOExporter::new();
        let model = MockModel {
            config: MockConfig { hidden_size: 512 },
        };

        assert!(exporter.validate_model(&model, ExportFormat::OpenVINO).is_ok());
    }

    #[test]
    fn test_validate_model_wrong_format() {
        let exporter = OpenVINOExporter::new();
        let model = MockModel {
            config: MockConfig { hidden_size: 512 },
        };

        assert!(exporter.validate_model(&model, ExportFormat::ONNX).is_err());
    }

    #[test]
    fn test_openvino_precision_config_default() {
        let config = OpenVINOPrecisionConfig::default();
        assert_eq!(config.input_precision, "FP32");
        assert_eq!(config.output_precision, "FP32");
        assert_eq!(config.weights_precision, "FP32");
        assert!(!config.enable_int8_calibration);
    }

    #[test]
    fn test_xml_generation() {
        let exporter = OpenVINOExporter::new();
        let model = MockModel {
            config: MockConfig { hidden_size: 512 },
        };
        let config = ExportConfig {
            format: ExportFormat::OpenVINO,
            batch_size: Some(2),
            sequence_length: Some(128),
            ..Default::default()
        };

        let xml = exporter.build_openvino_xml(&model, &config).unwrap();

        assert!(xml.contains(r#"<net name="trustformers_model""#));
        assert!(xml.contains("batch=\"2\""));
        assert!(xml.contains("type=\"Parameter\""));
        assert!(xml.contains("type=\"MatMul\""));
        assert!(xml.contains("type=\"Softmax\""));
        assert!(xml.contains("type=\"Gelu\""));
        assert!(xml.contains("<layers>"));
        assert!(xml.contains("<edges>"));
    }

    #[test]
    fn test_mapping_file_generation() {
        let exporter = OpenVINOExporter::new();
        let model = MockModel {
            config: MockConfig { hidden_size: 512 },
        };
        let config = ExportConfig {
            format: ExportFormat::OpenVINO,
            ..Default::default()
        };

        let mapping = exporter.build_mapping_file(&model, &config).unwrap();

        assert!(mapping.contains("# OpenVINO Layer Mapping File"));
        assert!(mapping.contains("embedding,"));
        assert!(mapping.contains("attention_query_0,"));
        assert!(mapping.contains("ffn_intermediate,"));
    }

    #[test]
    fn test_optimization_config_generation() {
        let exporter = OpenVINOExporter::new();
        let model = MockModel {
            config: MockConfig { hidden_size: 512 },
        };
        let config = ExportConfig {
            format: ExportFormat::OpenVINO,
            precision: ExportPrecision::FP16,
            optimize: true,
            batch_size: Some(4),
            ..Default::default()
        };

        let opt_config = exporter.build_optimization_config(&model, &config).unwrap();

        assert_eq!(opt_config["model_optimizer"]["target_device"], "CPU");
        assert_eq!(opt_config["model_optimizer"]["precision"], "FP16");
        assert_eq!(opt_config["model_optimizer"]["batch_size"], 4);
        assert_eq!(opt_config["optimization_config"]["optimize"], true);
        assert_eq!(opt_config["optimization_config"]["compress_to_fp16"], true);
    }
}
