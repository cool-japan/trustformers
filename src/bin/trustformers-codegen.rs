use clap::{App, Arg, SubCommand};
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use trustformers::codegen::{CodeGenerator, ModelTemplate, LayerConfig, ModelConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = App::new("TrustformeRS Code Generator")
        .version("1.0")
        .author("TrustformeRS Team")
        .about("Generate model implementations, pipelines, and training code")
        .subcommand(
            SubCommand::with_name("model")
                .about("Generate a model implementation")
                .arg(
                    Arg::with_name("config")
                        .short("c")
                        .long("config")
                        .value_name("FILE")
                        .help("JSON configuration file for the model")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("output")
                        .short("o")
                        .long("output")
                        .value_name("DIR")
                        .help("Output directory for generated code")
                        .default_value("./generated")
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("pipeline")
                .about("Generate a pipeline implementation")
                .arg(
                    Arg::with_name("name")
                        .short("n")
                        .long("name")
                        .value_name("NAME")
                        .help("Name of the pipeline")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("model-type")
                        .short("m")
                        .long("model-type")
                        .value_name("TYPE")
                        .help("Type of model (e.g., BERT, GPT, T5)")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("task")
                        .short("t")
                        .long("task")
                        .value_name("TASK")
                        .help("Task type (e.g., text-generation, text-classification)")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("output")
                        .short("o")
                        .long("output")
                        .value_name("DIR")
                        .help("Output directory for generated code")
                        .default_value("./generated")
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("training")
                .about("Generate a training loop implementation")
                .arg(
                    Arg::with_name("model")
                        .short("m")
                        .long("model")
                        .value_name("MODEL")
                        .help("Model name")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("optimizer")
                        .short("opt")
                        .long("optimizer")
                        .value_name("OPTIMIZER")
                        .help("Optimizer type (e.g., Adam, SGD, AdamW)")
                        .default_value("Adam")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("loss")
                        .short("l")
                        .long("loss")
                        .value_name("LOSS")
                        .help("Loss function (e.g., CrossEntropy, MSE)")
                        .default_value("CrossEntropy")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("output")
                        .short("o")
                        .long("output")
                        .value_name("DIR")
                        .help("Output directory for generated code")
                        .default_value("./generated")
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("init")
                .about("Initialize a new model project with template structure")
                .arg(
                    Arg::with_name("name")
                        .short("n")
                        .long("name")
                        .value_name("NAME")
                        .help("Project name")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("template")
                        .short("t")
                        .long("template")
                        .value_name("TEMPLATE")
                        .help("Template type (transformer, cnn, rnn)")
                        .default_value("transformer")
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("list")
                .about("List available templates and examples"),
        )
        .get_matches();

    match matches.subcommand() {
        ("model", Some(model_matches)) => {
            let config_file = model_matches.value_of("config").unwrap();
            let output_dir = model_matches.value_of("output").unwrap();
            generate_model(config_file, output_dir)?;
        }
        ("pipeline", Some(pipeline_matches)) => {
            let name = pipeline_matches.value_of("name").unwrap();
            let model_type = pipeline_matches.value_of("model-type").unwrap();
            let task = pipeline_matches.value_of("task").unwrap();
            let output_dir = pipeline_matches.value_of("output").unwrap();
            generate_pipeline(name, model_type, task, output_dir)?;
        }
        ("training", Some(training_matches)) => {
            let model = training_matches.value_of("model").unwrap();
            let optimizer = training_matches.value_of("optimizer").unwrap();
            let loss = training_matches.value_of("loss").unwrap();
            let output_dir = training_matches.value_of("output").unwrap();
            generate_training(model, optimizer, loss, output_dir)?;
        }
        ("init", Some(init_matches)) => {
            let name = init_matches.value_of("name").unwrap();
            let template = init_matches.value_of("template").unwrap();
            initialize_project(name, template)?;
        }
        ("list", Some(_)) => {
            list_templates();
        }
        _ => {
            println!("No subcommand specified. Use --help for usage information.");
        }
    }

    Ok(())
}

fn generate_model(config_file: &str, output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating model from config: {}", config_file);

    let config_content = fs::read_to_string(config_file)?;
    let template: ModelTemplate = serde_json::from_str(&config_content)?;

    let generator = CodeGenerator::new(output_dir);
    let generated_code = generator.generate_model(&template)?;

    let filename = format!("{}_model.rs", template.name.to_lowercase());
    generator.generate_to_file(&generated_code, &filename)?;

    println!("✅ Model generated successfully!");
    println!("📁 Output: {}/{}", output_dir, filename);

    Ok(())
}

fn generate_pipeline(
    name: &str,
    model_type: &str,
    task: &str,
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating pipeline: {} ({} for {})", name, model_type, task);

    let generator = CodeGenerator::new(output_dir);
    let generated_code = generator.generate_pipeline(name, model_type, task)?;

    let filename = format!("{}_pipeline.rs", name.to_lowercase());
    generator.generate_to_file(&generated_code, &filename)?;

    println!("✅ Pipeline generated successfully!");
    println!("📁 Output: {}/{}", output_dir, filename);

    Ok(())
}

fn generate_training(
    model: &str,
    optimizer: &str,
    loss: &str,
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating training loop for: {} (optimizer: {}, loss: {})", model, optimizer, loss);

    let generator = CodeGenerator::new(output_dir);
    let generated_code = generator.generate_training_loop(model, optimizer, loss)?;

    let filename = format!("{}_training.rs", model.to_lowercase());
    generator.generate_to_file(&generated_code, &filename)?;

    println!("✅ Training loop generated successfully!");
    println!("📁 Output: {}/{}", output_dir, filename);

    Ok(())
}

fn initialize_project(name: &str, template: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing project: {} (template: {})", name, template);

    let project_dir = format!("./{}", name);
    fs::create_dir_all(&project_dir)?;

    // Create project structure
    create_project_structure(&project_dir)?;

    // Generate sample configuration
    generate_sample_config(&project_dir, template)?;

    // Generate README
    generate_readme(&project_dir, name, template)?;

    println!("✅ Project initialized successfully!");
    println!("📁 Project directory: {}", project_dir);
    println!("📝 Next steps:");
    println!("   1. cd {}", name);
    println!("   2. Edit config.json to customize your model");
    println!("   3. Run: trustformers-codegen model -c config.json");

    Ok(())
}

fn create_project_structure(project_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let dirs = ["src", "examples", "tests", "configs"];

    for dir in &dirs {
        let full_path = Path::new(project_dir).join(dir);
        fs::create_dir_all(full_path)?;
    }

    // Create Cargo.toml
    let cargo_toml = format!(r#"[package]
name = "{}"
version = "0.1.0"
edition = "2021"

[dependencies]
trustformers = {{ path = "../trustformers" }}
trustformers-core = {{ path = "../trustformers-core" }}
trustformers-optim = {{ path = "../trustformers-optim" }}
trustformers-training = {{ path = "../trustformers-training" }}
serde = {{ version = "1.0", features = ["derive"] }}
serde_json = "1.0"
tokio = {{ version = "1.0", features = ["full"] }}

[lib]
name = "custom_model"
path = "src/lib.rs"
"#, Path::new(project_dir).file_name().unwrap().to_str().unwrap());

    fs::write(Path::new(project_dir).join("Cargo.toml"), cargo_toml)?;

    Ok(())
}

fn generate_sample_config(project_dir: &str, template: &str) -> Result<(), Box<dyn std::error::Error>> {
    let config = match template {
        "transformer" => {
            let model_config = ModelConfig {
                hidden_size: 768,
                num_layers: 12,
                num_attention_heads: 12,
                vocab_size: 30000,
                max_position_embeddings: 512,
            };

            let layers = vec![
                LayerConfig {
                    layer_type: "attention".to_string(),
                    params: {
                        let mut params = HashMap::new();
                        params.insert("hidden_size".to_string(), serde_json::Value::Number(serde_json::Number::from(768)));
                        params.insert("num_heads".to_string(), serde_json::Value::Number(serde_json::Number::from(12)));
                        params
                    },
                },
                LayerConfig {
                    layer_type: "feedforward".to_string(),
                    params: {
                        let mut params = HashMap::new();
                        params.insert("hidden_size".to_string(), serde_json::Value::Number(serde_json::Number::from(768)));
                        params.insert("intermediate_size".to_string(), serde_json::Value::Number(serde_json::Number::from(3072)));
                        params
                    },
                },
            ];

            ModelTemplate {
                name: "CustomTransformer".to_string(),
                architecture: "transformer".to_string(),
                layers,
                config: model_config,
            }
        }
        "xlstm" => {
            let model_config = ModelConfig {
                hidden_size: 768,
                num_layers: 16, // xLSTM typically uses more layers
                num_attention_heads: 12, // Used for mLSTM heads
                vocab_size: 30000,
                max_position_embeddings: 2048, // xLSTM can handle longer sequences
            };

            let layers = vec![
                LayerConfig {
                    layer_type: "slstm".to_string(),
                    params: {
                        let mut params = HashMap::new();
                        params.insert("hidden_size".to_string(), serde_json::Value::Number(serde_json::Number::from(768)));
                        params.insert("exponential_gating".to_string(), serde_json::Value::Bool(true));
                        params
                    },
                },
                LayerConfig {
                    layer_type: "mlstm".to_string(),
                    params: {
                        let mut params = HashMap::new();
                        params.insert("hidden_size".to_string(), serde_json::Value::Number(serde_json::Number::from(768)));
                        params.insert("memory_dimension".to_string(), serde_json::Value::Number(serde_json::Number::from(768)));
                        params.insert("num_heads".to_string(), serde_json::Value::Number(serde_json::Number::from(12)));
                        params
                    },
                },
                LayerConfig {
                    layer_type: "rms_norm".to_string(),
                    params: {
                        let mut params = HashMap::new();
                        params.insert("hidden_size".to_string(), serde_json::Value::Number(serde_json::Number::from(768)));
                        params.insert("eps".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(1e-5).unwrap()));
                        params
                    },
                },
            ];

            ModelTemplate {
                name: "CustomxLSTM".to_string(),
                architecture: "xlstm".to_string(),
                layers,
                config: model_config,
            }
        }
        "cnn" => {
            let model_config = ModelConfig {
                hidden_size: 512,
                num_layers: 4,
                num_attention_heads: 1,
                vocab_size: 10, // num_classes for CNN
                max_position_embeddings: 224, // image_size
            };

            ModelTemplate {
                name: "CustomCNN".to_string(),
                architecture: "cnn".to_string(),
                layers: vec![],
                config: model_config,
            }
        }
        _ => {
            return Err(format!("Unknown template: {}. Supported templates: transformer, xlstm, cnn", template).into());
        }
    };

    let config_json = serde_json::to_string_pretty(&config)?;
    fs::write(Path::new(project_dir).join("config.json"), config_json)?;

    Ok(())
}

fn generate_readme(project_dir: &str, name: &str, template: &str) -> Result<(), Box<dyn std::error::Error>> {
    let readme_content = format!(r#"# {}

A custom {} model implementation using TrustformeRS.

## Getting Started

1. **Customize the model configuration:**
   Edit `config.json` to match your requirements.

2. **Generate the model code:**
   ```bash
   trustformers-codegen model -c config.json -o src/
   ```

3. **Generate a pipeline:**
   ```bash
   trustformers-codegen pipeline -n {} -m {} -t text-classification -o src/
   ```

4. **Generate training code:**
   ```bash
   trustformers-codegen training -m {} -o src/
   ```

5. **Build and test:**
   ```bash
   cargo build
   cargo test
   ```

## Project Structure

- `src/` - Generated model implementations
- `config.json` - Model configuration
- `examples/` - Usage examples
- `tests/` - Unit tests

## Configuration

The `config.json` file controls model architecture:

- `name`: Model class name
- `architecture`: Template type ({})
- `config.hidden_size`: Hidden dimension size
- `config.num_layers`: Number of layers
- `config.num_attention_heads`: Number of attention heads (for transformers)
- `config.vocab_size`: Vocabulary size
- `config.max_position_embeddings`: Maximum sequence length

## Usage

```rust
use custom_model::{{}}Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {{
    let config = {}Config::default();
    let model = {}::new(config);

    // Your code here

    Ok(())
}}
```

## Contributing

1. Edit the model configuration
2. Regenerate code using the codegen tool
3. Add tests and examples
4. Submit a pull request
"#,
    name,
    template,
    name.to_lowercase(),
    template,
    name,
    template,
    name,
    name,
    name
);

    fs::write(Path::new(project_dir).join("README.md"), readme_content)?;

    Ok(())
}

fn list_templates() {
    println!("📋 Available Templates:");
    println!();
    println!("🏗️  Model Templates:");
    println!("   • transformer    - Transformer-based models (BERT, GPT, T5 style)");
    println!("   • xlstm          - 🔥 Extended LSTM - cutting-edge LSTM revival (2024)");
    println!("   • cnn            - Convolutional neural networks");
    println!("   • pipeline       - End-to-end inference pipelines");
    println!("   • training       - Training loop implementations");
    println!();

    println!("🧠 Modern Layer Types:");
    println!("   Classic:");
    println!("   • attention      - Multi-head self-attention mechanism");
    println!("   • feedforward    - Standard feed-forward networks");
    println!("   • layer_norm     - Layer normalization");
    println!();
    println!("   Modern (2023-2024):");
    println!("   • rms_norm       - 🔥 RMS normalization (LLaMA, PaLM style)");
    println!("   • swiglu         - 🔥 SwiGLU activation (GPT-4, PaLM style)");
    println!("   • rope           - 🔥 Rotary Position Embedding (RoPE)");
    println!("   • group_norm     - Group normalization for stable training");
    println!("   • slstm          - 🔥 Scalar memory LSTM (xLSTM)");
    println!("   • mlstm          - 🔥 Matrix memory LSTM (xLSTM)");
    println!("   • ring_attention - 🔥 Ring attention for ultra-long sequences");
    println!();

    println!("🎯 Supported Tasks:");
    println!("   • text-generation        - Autoregressive text generation");
    println!("   • text-classification    - Text classification tasks");
    println!("   • question-answering     - Extractive question answering");
    println!("   • summarization          - Text summarization");
    println!("   • translation            - Machine translation");
    println!("   • fill-mask              - Masked language modeling");
    println!("   • long-sequence          - Ultra-long sequence processing");
    println!();

    println!("🔧 Supported Optimizers:");
    println!("   • Adam, AdamW, SGD, Adafactor, LAMB, Lion");
    println!();

    println!("📊 Supported Loss Functions:");
    println!("   • CrossEntropy, MSE, Huber, FocalLoss");
    println!();

    println!("🔥 New Features:");
    println!("   • xLSTM models with exponential gating and matrix memory");
    println!("   • Ring attention for sequences up to millions of tokens");
    println!("   • Modern normalization and activation functions");
    println!("   • State-of-the-art architectural components");
    println!();

    println!("💡 Examples:");
    println!("   # Create a classic transformer project");
    println!("   trustformers-codegen init -n my_project -t transformer");
    println!();
    println!("   # Create a cutting-edge xLSTM project");
    println!("   trustformers-codegen init -n lstm_revival -t xlstm");
    println!();
    println!("   # Generate models from config");
    println!("   trustformers-codegen model -c config.json -o src/");
    println!();
    println!("   # Create specialized pipelines");
    println!("   trustformers-codegen pipeline -n MyPipeline -m xLSTM -t text-generation");
    println!("   trustformers-codegen pipeline -n UltraLong -m transformer -t long-sequence");
    println!();
    println!("   # Generate training code");
    println!("   trustformers-codegen training -m MyModel -opt AdamW -l CrossEntropy");
}