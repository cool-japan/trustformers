//
//  TrustformersKitExample.swift
//  Example usage of TrustformersKit
//
//  Copyright (c) 2024 TrustformeRS. All rights reserved.
//

import UIKit
import TrustformersKit
import CoreML

class TrustformersKitExample {
    
    private let engine: TFKInferenceEngine
    private var model: TFKModel?
    
    init() {
        // Initialize with optimized configuration
        let config = TFKModelConfig.optimizedConfig()
        engine = TFKInferenceEngine(config: config)
        
        // Set up logging
        TFKLogger.setLogLevel(.debug)
        TFKLogger.enableFileLogging()
    }
    
    // MARK: - Basic Inference Example
    
    func basicInferenceExample() {
        do {
            // Load model
            let model = try engine.loadModel(
                bundleResource: "bert_model",
                type: "tfm",
                config: .defaultConfig()
            )
            
            // Create input tensor
            let inputData: [Float] = Array(repeating: 0.5, count: 768)
            let inputTensor = TFKTensor(floats: inputData, shape: [1, 768])
            
            // Perform inference
            let result = engine.performInference(model, input: inputTensor)
            
            if result.success {
                print("Inference successful!")
                print("Time: \(result.inferenceTimeMs) ms")
                print("Memory: \(result.memoryUsed) MB")
                print("Output shape: \(result.output.shape)")
            } else if let error = result.error {
                print("Inference failed: \(error)")
            }
            
        } catch {
            print("Error: \(error)")
        }
    }
    
    // MARK: - Text Classification Example
    
    func textClassificationExample(text: String) {
        do {
            // Load classification model
            let config = TFKModelConfig.textGenerationConfig()
            let model = try engine.loadModel(
                bundleResource: "text_classifier",
                type: "tfm",
                config: config
            )
            
            // Tokenize text (simplified - would use real tokenizer)
            let tokens = tokenizeText(text)
            let inputTensor = TFKTensor(floats: tokens, shape: [1, tokens.count])
            
            // Perform inference
            let result = engine.performInference(model, input: inputTensor)
            
            if result.success {
                // Apply softmax to get probabilities
                let probResult = try result.applySoftmax()
                
                // Get top predictions
                let labels = ["positive", "negative", "neutral"]
                let topPredictions = probResult.topKPredictions(k: 3, labels: labels)
                
                print("Text Classification Results:")
                for prediction in topPredictions {
                    let percentage = prediction.score * 100
                    print("- \(prediction.label ?? "Unknown"): \(String(format: "%.2f", percentage))%")
                }
            }
            
        } catch {
            TFKLogger.error(error)
        }
    }
    
    // MARK: - Image Classification Example
    
    func imageClassificationExample(image: UIImage) {
        do {
            // Load image classification model
            let config = TFKModelConfig.imageClassificationConfig()
            let model = try engine.loadModel(
                bundleResource: "mobilenet",
                type: "tfm",
                config: config
            )
            
            // Preprocess image
            let inputTensor = preprocessImage(image, size: CGSize(width: 224, height: 224))
            
            // Perform inference
            let result = engine.performInference(model, input: inputTensor)
            
            if result.success {
                // Get classification
                let labels = loadImageNetLabels()
                let classification = result.getClassification(labels: labels)
                
                print("Image Classification:")
                print("Label: \(classification.label ?? "Unknown")")
                print("Confidence: \(String(format: "%.2f", classification.score * 100))%")
                
                // Get performance stats
                let stats = engine.performanceStats
                print("\nPerformance:")
                print(stats.performanceSummary())
            }
            
        } catch {
            TFKLogger.error(error)
        }
    }
    
    // MARK: - Core ML Integration Example
    
    @available(iOS 11.0, *)
    func coreMLIntegrationExample() {
        do {
            // Use Core ML backend
            let config = TFKModelConfig()
            config.backend = .coreML
            config.useFP16 = true
            
            // Load Core ML model
            let model = try engine.loadModel(
                bundleResource: "model",
                type: "mlmodelc",
                config: config
            )
            
            // Create MLMultiArray input
            let mlArray = try MLMultiArray(shape: [1, 3, 224, 224], dataType: .float32)
            
            // Fill with data...
            for i in 0..<mlArray.count {
                mlArray[i] = NSNumber(value: Float.random(in: 0...1))
            }
            
            // Convert to TFKTensor
            let inputTensor = try TFKTensor(mlMultiArray: mlArray)
            
            // Perform inference
            let result = engine.performInference(model, input: inputTensor)
            
            if result.success {
                // Convert output back to MLMultiArray if needed
                let outputArray = try result.output.toMLMultiArray()
                print("Core ML output shape: \(outputArray.shape)")
            }
            
        } catch {
            TFKLogger.error(error)
        }
    }
    
    // MARK: - Batch Processing Example
    
    func batchProcessingExample(images: [UIImage]) {
        do {
            // Configure for batch processing
            let config = TFKModelConfig()
            config.enableBatching = true
            config.maxBatchSize = 4
            
            let model = try engine.loadModel(
                bundleResource: "batch_model",
                type: "tfm",
                config: config
            )
            
            // Preprocess all images
            let inputTensors = images.map { image in
                preprocessImage(image, size: CGSize(width: 224, height: 224))
            }
            
            // Perform batch inference
            let startTime = CFAbsoluteTimeGetCurrent()
            let results = engine.performBatchInference(model, inputs: inputTensors)
            let totalTime = CFAbsoluteTimeGetCurrent() - startTime
            
            print("Batch Processing Results:")
            print("Processed \(images.count) images in \(String(format: "%.2f", totalTime * 1000)) ms")
            print("Average time per image: \(String(format: "%.2f", totalTime * 1000 / Double(images.count))) ms")
            
            // Aggregate results
            let aggregated = TFKInferenceResult.aggregate(results)
            print(aggregated.description)
            
        } catch {
            TFKLogger.error(error)
        }
    }
    
    // MARK: - Memory Management Example
    
    func memoryManagementExample() {
        // Set memory pressure handler
        engine.setMemoryPressureHandler { optimization in
            print("Memory pressure detected! Switching to: \(optimization)")
            
            // You could unload models or reduce batch size here
            switch optimization {
            case .maximum:
                // Unload non-essential models
                self.model?.unload()
            case .balanced:
                // Reduce batch size
                break
            case .minimal:
                // Normal operation
                break
            }
        }
        
        // Enable thermal throttling
        engine.setThermalThrottling(true)
        
        // Optimize for battery life when needed
        if !TFKDeviceInfo.currentDevice().isCharging {
            engine.optimizeForBatteryLife(true)
        }
    }
    
    // MARK: - Device Capabilities Example
    
    func checkDeviceCapabilities() {
        let deviceInfo = TFKDeviceInfo.currentDevice()
        
        print("Device Information:")
        print(deviceInfo.deviceSummary())
        
        // Get recommended configuration
        let recommendedConfig = deviceInfo.recommendedConfig()
        print("\nRecommended Configuration:")
        print(recommendedConfig.configurationSummary())
        
        // Check specific features
        if deviceInfo.supportsFeature("neuralengine") {
            print("\n✅ Neural Engine available - using optimized models")
        }
        
        if deviceInfo.supportsFeature("coreml") {
            print("✅ Core ML available - can use .mlmodel files")
        }
        
        // Check performance tier
        switch deviceInfo.performanceTier() {
        case .flagship:
            print("\n🚀 Flagship device - enabling all optimizations")
        case .high:
            print("\n⚡ High-end device - good performance expected")
        case .medium:
            print("\n📱 Mid-range device - balanced settings recommended")
        case .low:
            print("\n🔋 Entry-level device - using conservative settings")
        }
    }
    
    // MARK: - Helper Methods
    
    private func tokenizeText(_ text: String) -> [Float] {
        // Simplified tokenization - in practice, use proper tokenizer
        let words = text.lowercased().components(separatedBy: .whitespacesAndNewlines)
        return words.prefix(512).map { _ in Float.random(in: 0...1) }
    }
    
    private func preprocessImage(_ image: UIImage, size: CGSize) -> TFKTensor {
        // Simplified image preprocessing
        // In practice, would resize and normalize properly
        let pixelCount = Int(size.width * size.height) * 3
        let pixels = [Float](repeating: 0.5, count: pixelCount)
        return TFKTensor(floats: pixels, shape: [1, 3, Int(size.height), Int(size.width)])
    }
    
    private func loadImageNetLabels() -> [String] {
        // In practice, load from file
        return ["cat", "dog", "bird", "car", "person"]
    }
}

// MARK: - SwiftUI Example View

import SwiftUI

@available(iOS 13.0, *)
struct TrustformersKitDemoView: View {
    @State private var selectedImage: UIImage?
    @State private var predictionResult: String = "Select an image to classify"
    @State private var isProcessing: Bool = false
    @State private var performanceStats: String = ""
    
    private let example = TrustformersKitExample()
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Device Info
                DeviceInfoCard()
                
                // Image Selection
                if let image = selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(height: 200)
                        .cornerRadius(10)
                } else {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.gray.opacity(0.3))
                        .frame(height: 200)
                        .overlay(
                            Text("Tap to select image")
                                .foregroundColor(.gray)
                        )
                }
                
                // Results
                VStack(alignment: .leading, spacing: 10) {
                    Text("Prediction")
                        .font(.headline)
                    
                    Text(predictionResult)
                        .font(.body)
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(8)
                    
                    if !performanceStats.isEmpty {
                        Text("Performance")
                            .font(.headline)
                            .padding(.top)
                        
                        Text(performanceStats)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                            .background(Color.gray.opacity(0.1))
                            .cornerRadius(8)
                    }
                }
                
                Spacer()
                
                // Action Buttons
                HStack(spacing: 20) {
                    Button(action: selectImage) {
                        Label("Select Image", systemImage: "photo")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    
                    Button(action: runInference) {
                        if isProcessing {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle())
                        } else {
                            Label("Classify", systemImage: "cpu")
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .buttonStyle(.borderedProminent)
                    .disabled(selectedImage == nil || isProcessing)
                }
            }
            .padding()
            .navigationTitle("TrustformersKit Demo")
        }
    }
    
    private func selectImage() {
        // In a real app, would use image picker
        selectedImage = UIImage(systemName: "photo")
        predictionResult = "Ready to classify"
        performanceStats = ""
    }
    
    private func runInference() {
        guard let image = selectedImage else { return }
        
        isProcessing = true
        predictionResult = "Processing..."
        
        DispatchQueue.global(qos: .userInitiated).async {
            // Run inference
            example.imageClassificationExample(image: image)
            
            // Get results (simplified)
            let result = "Cat (confidence: 95.3%)"
            let stats = """
            Inference Time: 45.2 ms
            Memory Used: 128 MB
            Backend: Core ML
            """
            
            DispatchQueue.main.async {
                self.predictionResult = result
                self.performanceStats = stats
                self.isProcessing = false
            }
        }
    }
}

@available(iOS 13.0, *)
struct DeviceInfoCard: View {
    private let deviceInfo = TFKDeviceInfo.currentDevice()
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Device Capabilities")
                .font(.headline)
            
            HStack {
                Image(systemName: "cpu")
                Text("\(deviceInfo.cpuCores) CPU cores")
                Spacer()
                
                if deviceInfo.hasNeuralEngine {
                    Image(systemName: "brain")
                    Text("Neural Engine")
                }
            }
            .font(.caption)
            
            HStack {
                Image(systemName: "memorychip")
                Text("\(deviceInfo.totalMemoryMB) MB RAM")
                Spacer()
                
                Text("Tier: \(deviceInfo.performanceTier().debugDescription)")
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(Color.blue.opacity(0.2))
                    .cornerRadius(4)
            }
            .font(.caption)
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(10)
    }
}