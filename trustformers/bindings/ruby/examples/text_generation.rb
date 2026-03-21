#!/usr/bin/env ruby
# frozen_string_literal: true

# Example: Text Generation with TrustFormeRS Ruby bindings
#
# This example demonstrates how to use TrustFormeRS for text generation tasks.
# It shows both basic usage and advanced configuration options.

require "bundler/setup"
require "trustformers"

def main
  puts "🚀 TrustFormeRS Ruby Text Generation Example"
  puts "=" * 50

  # Display system information
  puts "\n📊 System Information:"
  info = TrustFormeRS.system_info
  puts "  Version: #{info[:version]}"
  puts "  Available devices: #{info[:devices].join(', ')}"
  puts "  GPU available: #{info[:gpu_available]}"
  puts "  Platform: #{info[:platform]}"

  # Initialize TrustFormeRS with optimized configuration
  puts "\n🔧 Initializing TrustFormeRS..."
  config = if TrustFormeRS.gpu_available?
             TrustFormeRS::Configuration.gpu("auto")
           else
             TrustFormeRS::Configuration.cpu
           end

  config.enable_logging! if ENV["DEBUG"]

  trustformers = TrustFormeRS.new(config)
  puts "  Device: #{trustformers.current_device}"
  puts "  Configuration: #{config.summary}"

  # Create a text generation pipeline
  puts "\n🔮 Creating text generation pipeline..."
  pipeline = trustformers.create_pipeline(
    task: :text_generation,
    model_id: "gpt2", # You can change this to other models
    max_new_tokens: 100,
    temperature: 0.7,
    top_p: 0.9,
    do_sample: true
  )

  puts "  Pipeline created successfully!"
  puts "  Model: #{pipeline.model_id}"
  puts "  Task: #{pipeline.task}"

  # Example prompts for text generation
  prompts = [
    "The future of artificial intelligence is",
    "In a world where technology and nature coexist,",
    "The most important discovery in science was",
    "Once upon a time, in a distant galaxy,",
    "The secret to happiness lies in"
  ]

  puts "\n📝 Generating text for sample prompts..."
  puts "-" * 40

  prompts.each_with_index do |prompt, index|
    puts "\n#{index + 1}. Prompt: \"#{prompt}\""
    puts "   Generating..."

    begin
      # Generate text
      start_time = Time.now
      result = pipeline.generate(prompt)
      end_time = Time.now

      # Display results
      generated_text = result.is_a?(TrustFormeRS::Pipeline::TextGenerationResult) ? 
                      result.generated_text : result
      
      puts "   Result: #{generated_text}"
      puts "   Time: #{((end_time - start_time) * 1000).round(2)}ms"
      
    rescue => e
      puts "   Error: #{e.message}"
    end
  end

  # Batch generation example
  puts "\n\n🔄 Batch Generation Example"
  puts "-" * 30

  batch_prompts = [
    "Technology will change",
    "The environment needs",
    "Education should focus on"
  ]

  puts "Generating text for #{batch_prompts.size} prompts simultaneously..."

  begin
    start_time = Time.now
    batch_results = pipeline.process_batch(batch_prompts, batch_size: 2)
    end_time = Time.now

    batch_results.each_with_index do |result, index|
      generated_text = result.is_a?(TrustFormeRS::Pipeline::TextGenerationResult) ? 
                      result.generated_text : result
      puts "#{index + 1}. #{batch_prompts[index]} → #{generated_text[0..100]}..."
    end

    puts "Batch processing time: #{((end_time - start_time) * 1000).round(2)}ms"

  rescue => e
    puts "Batch generation error: #{e.message}"
  end

  # Advanced configuration example
  puts "\n\n⚙️  Advanced Configuration Example"
  puts "-" * 35

  advanced_config = {
    max_new_tokens: 150,
    temperature: 0.8,
    top_p: 0.95,
    top_k: 40,
    repetition_penalty: 1.1,
    do_sample: true
  }

  puts "Configuration: #{advanced_config}"

  creative_prompt = "Write a short story about a robot who discovers emotions:"

  begin
    puts "\nGenerating creative text with advanced settings..."
    result = pipeline.generate(creative_prompt, **advanced_config)
    generated_text = result.is_a?(TrustFormeRS::Pipeline::TextGenerationResult) ? 
                    result.generated_text : result
    
    puts "\nResult:"
    puts word_wrap(generated_text, 70)

  rescue => e
    puts "Advanced generation error: #{e.message}"
  end

  # Memory usage statistics
  puts "\n\n📊 Memory Usage Statistics"
  puts "-" * 25
  
  stats = trustformers.memory_stats
  if stats[:total_allocated]
    puts "Total allocated: #{format_bytes(stats[:total_allocated])}"
    puts "Currently allocated: #{format_bytes(stats[:currently_allocated])}"
    puts "Peak allocated: #{format_bytes(stats[:peak_allocated])}"
  else
    puts "Memory statistics: #{stats}"
  end

rescue => e
  puts "\n❌ Error: #{e.message}"
  puts "   #{e.class}"
  puts "\nThis might happen if:"
  puts "   • TrustFormeRS native library is not installed"
  puts "   • The specified model is not available"
  puts "   • Insufficient system resources"
  puts "\nFor installation instructions, see the README.md file."

ensure
  # Clean up resources
  trustformers&.close
  puts "\n✅ Example completed!"
end

# Helper method to wrap text at specified width
def word_wrap(text, width = 80)
  text.gsub(/(.{1,#{width}})(\s+|\Z)/) { "#{$1}\n" }.strip
end

# Helper method to format bytes in human-readable format
def format_bytes(bytes)
  return "#{bytes} bytes" if bytes < 1024
  
  units = %w[KB MB GB TB]
  unit_index = 0
  size = bytes.to_f
  
  while size >= 1024 && unit_index < units.length - 1
    size /= 1024
    unit_index += 1
  end
  
  "#{size.round(2)} #{units[unit_index]}"
end

# Run the example if this file is executed directly
if __FILE__ == $0
  main
end