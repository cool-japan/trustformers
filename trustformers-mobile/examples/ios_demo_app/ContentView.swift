//
//  ContentView.swift
//  TrustformersDemo
//
//  SwiftUI demo application showcasing TrustformeRS mobile inference capabilities
//

import SwiftUI
import CoreML
import ARKit
import Combine
import TrustformersKit

struct ContentView: View {
    @StateObject private var inferenceEngine = InferenceEngineViewModel()
    @StateObject private var arkitEngine = ARKitDemoViewModel()
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // Text Classification Demo
            TextClassificationView(engine: inferenceEngine)
                .tabItem {
                    Image(systemName: "text.bubble")
                    Text("Text AI")
                }
                .tag(0)
            
            // Object Detection Demo
            ObjectDetectionView(engine: inferenceEngine)
                .tabItem {
                    Image(systemName: "camera.viewfinder")
                    Text("Vision AI")
                }
                .tag(1)
            
            // ARKit Integration Demo
            ARKitDemoView(arkitEngine: arkitEngine)
                .tabItem {
                    Image(systemName: "arkit")
                    Text("AR AI")
                }
                .tag(2)
            
            // Performance Monitoring
            PerformanceView(engine: inferenceEngine)
                .tabItem {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                    Text("Performance")
                }
                .tag(3)
        }
        .onAppear {
            setupTrustformers()
        }
    }
    
    private func setupTrustformers() {
        inferenceEngine.initialize()
        arkitEngine.initialize()
    }
}

// MARK: - Text Classification View
struct TextClassificationView: View {
    @ObservedObject var engine: InferenceEngineViewModel
    @State private var inputText = ""
    @State private var isAnalyzing = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Input Section
                VStack(alignment: .leading, spacing: 10) {
                    Text("Enter text to analyze:")
                        .font(.headline)
                    
                    TextEditor(text: $inputText)
                        .frame(height: 120)
                        .padding(8)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.blue, lineWidth: 1)
                        )
                }
                
                // Analysis Button
                Button(action: {
                    analyzeText()
                }) {
                    HStack {
                        if isAnalyzing {
                            ProgressView()
                                .scaleEffect(0.8)
                                .foregroundColor(.white)
                        }
                        Text(isAnalyzing ? "Analyzing..." : "Analyze Text")
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(inputText.isEmpty ? Color.gray : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(inputText.isEmpty || isAnalyzing)
                
                // Results Section
                if let result = engine.textClassificationResult {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Analysis Results:")
                            .font(.headline)
                        
                        ForEach(result.predictions, id: \.label) { prediction in
                            HStack {
                                Text(prediction.label)
                                    .fontWeight(.medium)
                                Spacer()
                                Text(String(format: "%.1f%%", prediction.confidence * 100))
                                    .foregroundColor(.blue)
                                    .fontWeight(.bold)
                            }
                            .padding(.vertical, 4)
                        }
                        
                        HStack {
                            Text("Processing Time:")
                                .fontWeight(.medium)
                            Spacer()
                            Text(String(format: "%.2f ms", result.processingTimeMs))
                                .foregroundColor(.green)
                        }
                    }
                    .padding()
                    .background(Color.green.opacity(0.1))
                    .cornerRadius(10)
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("Text Classification")
        }
    }
    
    private func analyzeText() {
        isAnalyzing = true
        engine.classifyText(inputText) { result in
            DispatchQueue.main.async {
                isAnalyzing = false
            }
        }
    }
}

// MARK: - Object Detection View
struct ObjectDetectionView: View {
    @ObservedObject var engine: InferenceEngineViewModel
    @State private var showingImagePicker = false
    @State private var selectedImage: UIImage?
    @State private var isDetecting = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Image Selection
                Button(action: {
                    showingImagePicker = true
                }) {
                    VStack {
                        if let image = selectedImage {
                            Image(uiImage: image)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(height: 200)
                                .cornerRadius(10)
                        } else {
                            Image(systemName: "photo.badge.plus")
                                .font(.system(size: 50))
                                .foregroundColor(.gray)
                            Text("Tap to select image")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                    }
                    .frame(height: 220)
                    .frame(maxWidth: .infinity)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(10)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(Color.blue, lineWidth: 1)
                    )
                }
                
                // Detection Button
                Button(action: {
                    detectObjects()
                }) {
                    HStack {
                        if isDetecting {
                            ProgressView()
                                .scaleEffect(0.8)
                                .foregroundColor(.white)
                        }
                        Text(isDetecting ? "Detecting..." : "Detect Objects")
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(selectedImage == nil ? Color.gray : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(selectedImage == nil || isDetecting)
                
                // Detection Results
                if let result = engine.objectDetectionResult {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Detected Objects:")
                            .font(.headline)
                        
                        ForEach(result.detections, id: \.id) { detection in
                            HStack {
                                VStack(alignment: .leading) {
                                    Text(detection.className)
                                        .fontWeight(.medium)
                                    Text("Confidence: \(String(format: "%.1f%%", detection.confidence * 100))")
                                        .font(.caption)
                                        .foregroundColor(.blue)
                                }
                                Spacer()
                                Text("(\(Int(detection.bbox.x)), \(Int(detection.bbox.y)))")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                            }
                            .padding(.vertical, 4)
                        }
                        
                        HStack {
                            Text("Processing Time:")
                                .fontWeight(.medium)
                            Spacer()
                            Text(String(format: "%.2f ms", result.processingTimeMs))
                                .foregroundColor(.green)
                        }
                    }
                    .padding()
                    .background(Color.green.opacity(0.1))
                    .cornerRadius(10)
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("Object Detection")
            .sheet(isPresented: $showingImagePicker) {
                ImagePicker(image: $selectedImage)
            }
        }
    }
    
    private func detectObjects() {
        guard let image = selectedImage else { return }
        
        isDetecting = true
        engine.detectObjects(in: image) { result in
            DispatchQueue.main.async {
                isDetecting = false
            }
        }
    }
}

// MARK: - ARKit Demo View
struct ARKitDemoView: View {
    @ObservedObject var arkitEngine: ARKitDemoViewModel
    @State private var isARSessionActive = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // AR Session Status
                HStack {
                    Circle()
                        .fill(isARSessionActive ? Color.green : Color.red)
                        .frame(width: 12, height: 12)
                    Text(isARSessionActive ? "AR Session Active" : "AR Session Inactive")
                        .fontWeight(.medium)
                    Spacer()
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(10)
                
                // AR Controls
                VStack(spacing: 15) {
                    Button(action: {
                        toggleARSession()
                    }) {
                        Text(isARSessionActive ? "Stop AR Session" : "Start AR Session")
                            .fontWeight(.semibold)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(isARSessionActive ? Color.red : Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    
                    if isARSessionActive {
                        Button(action: {
                            arkitEngine.captureWorldMap()
                        }) {
                            Text("Save AR World Map")
                                .fontWeight(.semibold)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                    }
                }
                
                // AR Statistics
                if isARSessionActive, let stats = arkitEngine.sessionStats {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("AR Session Statistics:")
                            .font(.headline)
                        
                        StatRow(label: "Tracking Quality", value: stats.trackingQuality.rawValue)
                        StatRow(label: "Detected Planes", value: "\(stats.detectedPlanesCount)")
                        StatRow(label: "Active Detections", value: "\(stats.activeDetectionsCount)")
                        StatRow(label: "Avg Processing Time", value: String(format: "%.2f ms", stats.averageProcessingTimeMs))
                        StatRow(label: "Session Duration", value: formatDuration(stats.sessionDuration))
                    }
                    .padding()
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(10)
                }
                
                Spacer()
                
                // AR Preview Placeholder
                Text("AR Camera View")
                    .frame(maxWidth: .infinity, minHeight: 200)
                    .background(Color.black.opacity(0.1))
                    .cornerRadius(10)
                    .overlay(
                        Text(isARSessionActive ? "🎥 AR Active" : "📷 Camera Ready")
                            .font(.title2)
                            .foregroundColor(.gray)
                    )
            }
            .padding()
            .navigationTitle("AR Object Detection")
            .onDisappear {
                if isARSessionActive {
                    arkitEngine.stopARSession()
                    isARSessionActive = false
                }
            }
        }
    }
    
    private func toggleARSession() {
        if isARSessionActive {
            arkitEngine.stopARSession()
        } else {
            arkitEngine.startARSession()
        }
        isARSessionActive.toggle()
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }
}

// MARK: - Performance View
struct PerformanceView: View {
    @ObservedObject var engine: InferenceEngineViewModel
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Device Information
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Device Information")
                            .font(.headline)
                        
                        if let deviceInfo = engine.deviceInfo {
                            StatRow(label: "Device", value: deviceInfo.deviceModel)
                            StatRow(label: "iOS Version", value: deviceInfo.osVersion)
                            StatRow(label: "Performance Tier", value: deviceInfo.performanceTier.rawValue)
                            StatRow(label: "Neural Engine", value: deviceInfo.hasNeuralEngine ? "Available" : "Not Available")
                            StatRow(label: "Metal Support", value: deviceInfo.hasMetalSupport ? "Yes" : "No")
                        }
                    }
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(10)
                    
                    // Performance Metrics
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Performance Metrics")
                            .font(.headline)
                        
                        if let metrics = engine.performanceMetrics {
                            StatRow(label: "CPU Usage", value: String(format: "%.1f%%", metrics.cpuUsage))
                            StatRow(label: "Memory Usage", value: String(format: "%.1f MB", metrics.memoryUsageMB))
                            StatRow(label: "GPU Usage", value: String(format: "%.1f%%", metrics.gpuUsage))
                            StatRow(label: "Battery Level", value: String(format: "%.0f%%", metrics.batteryLevel * 100))
                            StatRow(label: "Thermal State", value: metrics.thermalState.rawValue)
                        }
                    }
                    .padding()
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(10)
                    
                    // Model Information
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Loaded Models")
                            .font(.headline)
                        
                        ForEach(engine.loadedModels, id: \.name) { model in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text(model.name)
                                        .fontWeight(.medium)
                                    Spacer()
                                    Text(String(format: "%.1f MB", model.sizeGB * 1000))
                                        .font(.caption)
                                        .foregroundColor(.blue)
                                }
                                Text("Backend: \(model.backend.rawValue)")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                    .padding()
                    .background(Color.green.opacity(0.1))
                    .cornerRadius(10)
                    
                    Spacer()
                }
                .padding()
            }
            .navigationTitle("Performance Monitor")
            .onAppear {
                engine.startPerformanceMonitoring()
            }
            .onDisappear {
                engine.stopPerformanceMonitoring()
            }
        }
    }
}

// MARK: - Supporting Views
struct StatRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .fontWeight(.medium)
            Spacer()
            Text(value)
                .foregroundColor(.blue)
        }
    }
}

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Environment(\.presentationMode) var presentationMode
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .photoLibrary
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
            }
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

#Preview {
    ContentView()
}