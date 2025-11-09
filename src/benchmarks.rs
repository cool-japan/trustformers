use tch::{nn, Tensor, Device, Kind};
use std::time::{Duration, Instant};
use std::collections::HashMap;

pub struct BenchmarkSuite {
    results: HashMap<String, BenchmarkResult>,
    device: Device,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub duration: Duration,
    pub memory_used: Option<usize>,
    pub throughput: Option<f64>, // tokens/second or operations/second
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub batch_sizes: Vec<i64>,
    pub sequence_lengths: Vec<i64>,
    pub num_iterations: usize,
    pub warmup_iterations: usize,
    pub measure_memory: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            batch_sizes: vec![1, 4, 8, 16],
            sequence_lengths: vec![128, 256, 512, 1024],
            num_iterations: 100,
            warmup_iterations: 10,
            measure_memory: true,
        }
    }
}

impl BenchmarkSuite {
    pub fn new(device: Device) -> Self {
        Self {
            results: HashMap::new(),
            device,
        }
    }

    pub fn benchmark_attention(&mut self, config: &BenchmarkConfig) {
        println!("Benchmarking Multi-Head Attention...");

        for &batch_size in &config.batch_sizes {
            for &seq_len in &config.sequence_lengths {
                let benchmark_name = format!("attention_b{}_s{}", batch_size, seq_len);

                if let Ok(result) = self.run_attention_benchmark(batch_size, seq_len, config) {
                    self.results.insert(benchmark_name, result);
                }
            }
        }
    }

    fn run_attention_benchmark(&self, batch_size: i64, seq_len: i64, config: &BenchmarkConfig) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let d_model = 768i64;
        let num_heads = 12i64;

        // Create dummy variables
        let vs = nn::VarStore::new(self.device);
        let attention = crate::attention::MultiHeadAttention::new(&vs.root(), d_model, num_heads, 0.1);

        // Prepare input tensors
        let input = Tensor::randn(&[batch_size, seq_len, d_model])?;

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _ = attention.forward(&input, &input, &input, None);
        }

        // Benchmark
        let start_time = Instant::now();
        for _ in 0..config.num_iterations {
            let _ = attention.forward(&input, &input, &input, None);
        }
        let duration = start_time.elapsed() / config.num_iterations as u32;

        // Calculate throughput (tokens processed per second)
        let total_tokens = batch_size * seq_len;
        let throughput = total_tokens as f64 / duration.as_secs_f64();

        let mut parameters = HashMap::new();
        parameters.insert("batch_size".to_string(), batch_size.to_string());
        parameters.insert("seq_len".to_string(), seq_len.to_string());
        parameters.insert("d_model".to_string(), d_model.to_string());
        parameters.insert("num_heads".to_string(), num_heads.to_string());

        Ok(BenchmarkResult {
            name: format!("MultiHeadAttention_{}x{}", batch_size, seq_len),
            duration,
            memory_used: self.estimate_memory_usage(batch_size, seq_len, d_model),
            throughput: Some(throughput),
            parameters,
        })
    }

    pub fn benchmark_feedforward(&mut self, config: &BenchmarkConfig) {
        println!("Benchmarking Feed-Forward Networks...");

        for &batch_size in &config.batch_sizes {
            for &seq_len in &config.sequence_lengths {
                let benchmark_name = format!("feedforward_b{}_s{}", batch_size, seq_len);

                if let Ok(result) = self.run_feedforward_benchmark(batch_size, seq_len, config) {
                    self.results.insert(benchmark_name, result);
                }
            }
        }
    }

    fn run_feedforward_benchmark(&self, batch_size: i64, seq_len: i64, config: &BenchmarkConfig) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let d_model = 768i64;
        let d_ff = 3072i64;

        // Create dummy variables
        let vs = nn::VarStore::new(self.device);
        let feedforward = crate::layers::FeedForward::new(
            &vs.root(),
            d_model,
            d_ff,
            0.1,
            crate::layers::ActivationType::GELU,
        );

        // Prepare input tensors
        let input = Tensor::randn(&[batch_size, seq_len, d_model])?;

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _ = feedforward.forward(&input);
        }

        // Benchmark
        let start_time = Instant::now();
        for _ in 0..config.num_iterations {
            let _ = feedforward.forward(&input);
        }
        let duration = start_time.elapsed() / config.num_iterations as u32;

        // Calculate throughput
        let total_tokens = batch_size * seq_len;
        let throughput = total_tokens as f64 / duration.as_secs_f64();

        let mut parameters = HashMap::new();
        parameters.insert("batch_size".to_string(), batch_size.to_string());
        parameters.insert("seq_len".to_string(), seq_len.to_string());
        parameters.insert("d_model".to_string(), d_model.to_string());
        parameters.insert("d_ff".to_string(), d_ff.to_string());

        Ok(BenchmarkResult {
            name: format!("FeedForward_{}x{}", batch_size, seq_len),
            duration,
            memory_used: self.estimate_memory_usage(batch_size, seq_len, d_model),
            throughput: Some(throughput),
            parameters,
        })
    }

    pub fn benchmark_layer_norm(&mut self, config: &BenchmarkConfig) {
        println!("Benchmarking Layer Normalization...");

        for &batch_size in &config.batch_sizes {
            for &seq_len in &config.sequence_lengths {
                let benchmark_name = format!("layernorm_b{}_s{}", batch_size, seq_len);

                if let Ok(result) = self.run_layernorm_benchmark(batch_size, seq_len, config) {
                    self.results.insert(benchmark_name, result);
                }
            }
        }
    }

    fn run_layernorm_benchmark(&self, batch_size: i64, seq_len: i64, config: &BenchmarkConfig) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let d_model = 768i64;

        // Create dummy variables
        let vs = nn::VarStore::new(self.device);
        let layer_norm = crate::layers::LayerNorm::new(&vs.root(), vec![d_model], 1e-5, true);

        // Prepare input tensors
        let input = Tensor::randn(&[batch_size, seq_len, d_model])?;

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _ = layer_norm.forward(&input);
        }

        // Benchmark
        let start_time = Instant::now();
        for _ in 0..config.num_iterations {
            let _ = layer_norm.forward(&input);
        }
        let duration = start_time.elapsed() / config.num_iterations as u32;

        // Calculate throughput
        let total_tokens = batch_size * seq_len;
        let throughput = total_tokens as f64 / duration.as_secs_f64();

        let mut parameters = HashMap::new();
        parameters.insert("batch_size".to_string(), batch_size.to_string());
        parameters.insert("seq_len".to_string(), seq_len.to_string());
        parameters.insert("d_model".to_string(), d_model.to_string());

        Ok(BenchmarkResult {
            name: format!("LayerNorm_{}x{}", batch_size, seq_len),
            duration,
            memory_used: self.estimate_memory_usage(batch_size, seq_len, d_model),
            throughput: Some(throughput),
            parameters,
        })
    }

    pub fn benchmark_full_transformer(&mut self, config: &BenchmarkConfig) {
        println!("Benchmarking Full Transformer...");

        for &batch_size in &config.batch_sizes {
            for &seq_len in &config.sequence_lengths {
                let benchmark_name = format!("transformer_b{}_s{}", batch_size, seq_len);

                let result = self.run_transformer_benchmark(batch_size, seq_len, config);
                self.results.insert(benchmark_name, result);
            }
        }
    }

    fn run_transformer_benchmark(&self, batch_size: i64, seq_len: i64, config: &BenchmarkConfig) -> BenchmarkResult {
        let transformer_config = crate::models::TransformerConfig {
            vocab_size: 30000,
            d_model: 768,
            num_heads: 12,
            num_layers: 12,
            d_ff: 3072,
            max_seq_len: 1024,
            dropout: 0.1,
            ..Default::default()
        };

        // Create dummy variables
        let vs = nn::VarStore::new(self.device);
        let transformer = crate::models::Transformer::new(&vs.root(), transformer_config);

        // Prepare input tensors
        let input_ids = Tensor::randint(transformer_config.vocab_size, &[batch_size, seq_len], (Kind::Int64, self.device));

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _ = transformer.forward(&input_ids, None);
        }

        // Benchmark
        let start_time = Instant::now();
        for _ in 0..config.num_iterations {
            let _ = transformer.forward(&input_ids, None);
        }
        let duration = start_time.elapsed() / config.num_iterations as u32;

        // Calculate throughput
        let total_tokens = batch_size * seq_len;
        let throughput = total_tokens as f64 / duration.as_secs_f64();

        let mut parameters = HashMap::new();
        parameters.insert("batch_size".to_string(), batch_size.to_string());
        parameters.insert("seq_len".to_string(), seq_len.to_string());
        parameters.insert("num_layers".to_string(), "12".to_string());
        parameters.insert("d_model".to_string(), "768".to_string());

        BenchmarkResult {
            name: format!("Transformer_{}x{}", batch_size, seq_len),
            duration,
            memory_used: self.estimate_memory_usage(batch_size, seq_len, 768),
            throughput: Some(throughput),
            parameters,
        }
    }

    pub fn benchmark_bert_forward(&mut self, config: &BenchmarkConfig) {
        println!("Benchmarking BERT Forward Pass...");

        for &batch_size in &config.batch_sizes {
            for &seq_len in &config.sequence_lengths {
                let benchmark_name = format!("bert_b{}_s{}", batch_size, seq_len);

                let result = self.run_bert_benchmark(batch_size, seq_len, config);
                self.results.insert(benchmark_name, result);
            }
        }
    }

    fn run_bert_benchmark(&self, batch_size: i64, seq_len: i64, config: &BenchmarkConfig) -> BenchmarkResult {
        let bert_config = crate::bert::BertConfig::default();

        // Create dummy variables
        let vs = nn::VarStore::new(self.device);
        let bert = crate::bert::BertModel::new(&vs.root(), bert_config.clone(), false);

        // Prepare input tensors
        let input_ids = Tensor::randint(bert_config.vocab_size, &[batch_size, seq_len], (Kind::Int64, self.device));

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _ = bert.forward(&input_ids, None, None);
        }

        // Benchmark
        let start_time = Instant::now();
        for _ in 0..config.num_iterations {
            let _ = bert.forward(&input_ids, None, None);
        }
        let duration = start_time.elapsed() / config.num_iterations as u32;

        // Calculate throughput
        let total_tokens = batch_size * seq_len;
        let throughput = total_tokens as f64 / duration.as_secs_f64();

        let mut parameters = HashMap::new();
        parameters.insert("batch_size".to_string(), batch_size.to_string());
        parameters.insert("seq_len".to_string(), seq_len.to_string());
        parameters.insert("model".to_string(), "BERT".to_string());

        BenchmarkResult {
            name: format!("BERT_{}x{}", batch_size, seq_len),
            duration,
            memory_used: self.estimate_memory_usage(batch_size, seq_len, bert_config.d_model),
            throughput: Some(throughput),
            parameters,
        }
    }

    pub fn benchmark_gpt_generation(&mut self, config: &BenchmarkConfig) {
        println!("Benchmarking GPT Generation...");

        for &batch_size in &config.batch_sizes {
            let seq_len = 128i64; // Starting sequence length for generation
            let max_length = 256i64; // Target generation length

            let benchmark_name = format!("gpt_gen_b{}_s{}_max{}", batch_size, seq_len, max_length);

            let result = self.run_gpt_generation_benchmark(batch_size, seq_len, max_length, config);
            self.results.insert(benchmark_name, result);
        }
    }

    fn run_gpt_generation_benchmark(&self, batch_size: i64, seq_len: i64, max_length: i64, config: &BenchmarkConfig) -> BenchmarkResult {
        let gpt_config = crate::gpt::GPTConfig::default();

        // Create dummy variables
        let vs = nn::VarStore::new(self.device);
        let gpt = crate::gpt::GPTModel::new(&vs.root(), gpt_config.clone());

        // Prepare input tensors
        let input_ids = Tensor::randint(gpt_config.vocab_size, &[batch_size, seq_len], (Kind::Int64, self.device));

        // Warmup (fewer iterations for generation)
        for _ in 0..(config.warmup_iterations / 10).max(1) {
            let _ = gpt.generate(&input_ids, max_length, 1.0, None, None, true, None, None);
        }

        // Benchmark
        let start_time = Instant::now();
        for _ in 0..(config.num_iterations / 10).max(1) {
            let _ = gpt.generate(&input_ids, max_length, 1.0, None, None, true, None, None);
        }
        let duration = start_time.elapsed() / ((config.num_iterations / 10).max(1) as u32);

        // Calculate throughput (generated tokens per second)
        let generated_tokens = batch_size * (max_length - seq_len);
        let throughput = generated_tokens as f64 / duration.as_secs_f64();

        let mut parameters = HashMap::new();
        parameters.insert("batch_size".to_string(), batch_size.to_string());
        parameters.insert("input_seq_len".to_string(), seq_len.to_string());
        parameters.insert("max_length".to_string(), max_length.to_string());
        parameters.insert("model".to_string(), "GPT".to_string());

        BenchmarkResult {
            name: format!("GPT_Generation_{}x{}", batch_size, max_length - seq_len),
            duration,
            memory_used: self.estimate_memory_usage(batch_size, max_length, gpt_config.d_model),
            throughput: Some(throughput),
            parameters,
        }
    }

    fn estimate_memory_usage(&self, batch_size: i64, seq_len: i64, d_model: i64) -> Option<usize> {
        // Rough estimation of memory usage in bytes
        let float_size = 4; // 4 bytes per float32

        // Basic tensor memory: batch_size * seq_len * d_model * float_size
        let basic_memory = (batch_size * seq_len * d_model * float_size as i64) as usize;

        // Attention memory: roughly batch_size * num_heads * seq_len^2 * float_size
        let num_heads = 12i64;
        let attention_memory = (batch_size * num_heads * seq_len * seq_len * float_size as i64) as usize;

        // Total rough estimate
        Some(basic_memory + attention_memory)
    }

    pub fn run_all_benchmarks(&mut self, config: &BenchmarkConfig) {
        println!("Running comprehensive benchmark suite...");

        self.benchmark_attention(config);
        self.benchmark_feedforward(config);
        self.benchmark_layer_norm(config);
        self.benchmark_full_transformer(config);
        self.benchmark_bert_forward(config);
        self.benchmark_gpt_generation(config);

        println!("All benchmarks completed!");
    }

    pub fn get_results(&self) -> &HashMap<String, BenchmarkResult> {
        &self.results
    }

    pub fn print_summary(&self) {
        println!("\n=== Benchmark Results Summary ===");

        let mut results: Vec<_> = self.results.values().collect();
        results.sort_by(|a, b| a.name.cmp(&b.name));

        for result in results {
            println!("\n{}", result.name);
            println!("  Duration: {:.2}ms", result.duration.as_secs_f64() * 1000.0);

            if let Some(throughput) = result.throughput {
                println!("  Throughput: {:.2} tokens/sec", throughput);
            }

            if let Some(memory) = result.memory_used {
                println!("  Memory: {:.2}MB", memory as f64 / 1_000_000.0);
            }

            for (key, value) in &result.parameters {
                println!("  {}: {}", key, value);
            }
        }
    }

    pub fn export_csv(&self, filename: &str) -> Result<(), std::io::Error> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(filename)?;

        // Write header
        writeln!(file, "Name,Duration_ms,Throughput_tokens_per_sec,Memory_MB,Batch_Size,Seq_Len")?;

        // Write data
        for result in self.results.values() {
            let duration_ms = result.duration.as_secs_f64() * 1000.0;
            let throughput = result.throughput.unwrap_or(0.0);
            let memory_mb = result.memory_used.map(|m| m as f64 / 1_000_000.0).unwrap_or(0.0);
            let batch_size = result.parameters.get("batch_size").unwrap_or(&"0".to_string());
            let seq_len = result.parameters.get("seq_len").unwrap_or(&"0".to_string());

            writeln!(file, "{},{:.2},{:.2},{:.2},{},{}",
                    result.name, duration_ms, throughput, memory_mb, batch_size, seq_len)?;
        }

        Ok(())
    }

    pub fn compare_with_baseline(&self, baseline_results: &HashMap<String, BenchmarkResult>) {
        println!("\n=== Performance Comparison ===");

        for (name, current) in &self.results {
            if let Some(baseline) = baseline_results.get(name) {
                let speedup = baseline.duration.as_secs_f64() / current.duration.as_secs_f64();
                let throughput_improvement = if let (Some(current_tp), Some(baseline_tp)) =
                    (current.throughput, baseline.throughput) {
                    current_tp / baseline_tp
                } else {
                    1.0
                };

                println!("{}: {:.2}x speedup, {:.2}x throughput improvement",
                        name, speedup, throughput_improvement);
            }
        }
    }
}

pub fn run_quick_benchmark() -> BenchmarkSuite {
    let device = Device::Cpu; // Change to Cuda(0) if CUDA is available
    let mut suite = BenchmarkSuite::new(device);

    let config = BenchmarkConfig {
        batch_sizes: vec![1, 4],
        sequence_lengths: vec![128, 512],
        num_iterations: 10,
        warmup_iterations: 2,
        measure_memory: true,
    };

    suite.run_all_benchmarks(&config);
    suite.print_summary();

    suite
}

pub fn run_comprehensive_benchmark() -> BenchmarkSuite {
    let device = Device::Cpu; // Change to Cuda(0) if CUDA is available
    let mut suite = BenchmarkSuite::new(device);

    let config = BenchmarkConfig::default();

    suite.run_all_benchmarks(&config);
    suite.print_summary();

    // Export results
    if let Err(e) = suite.export_csv("benchmark_results.csv") {
        println!("Failed to export CSV: {}", e);
    }

    suite
}