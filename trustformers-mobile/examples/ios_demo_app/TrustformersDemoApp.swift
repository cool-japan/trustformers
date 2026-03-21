//
//  TrustformersDemoApp.swift
//  TrustformersDemo
//
//  Main app entry point for TrustformeRS mobile demo
//

import SwiftUI

@main
struct TrustformersDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .onAppear {
                    setupApp()
                }
        }
    }
    
    private func setupApp() {
        // Configure TrustformersKit logging
        TrustformersKit.setLogLevel(.info)
        
        // Initialize performance monitoring
        TrustformersKit.enablePerformanceMonitoring(true)
        
        // Configure memory management
        TrustformersKit.setMemoryWarningThreshold(0.8) // 80% memory usage
        
        print("🚀 TrustformersRS Demo App Initialized")
    }
}

// Mock TrustformersKit configuration
struct TrustformersKit {
    enum LogLevel {
        case debug, info, warning, error
    }
    
    static func setLogLevel(_ level: LogLevel) {
        print("📝 Log level set to: \(level)")
    }
    
    static func enablePerformanceMonitoring(_ enabled: Bool) {
        print("📊 Performance monitoring: \(enabled ? "enabled" : "disabled")")
    }
    
    static func setMemoryWarningThreshold(_ threshold: Double) {
        print("⚠️ Memory warning threshold set to: \(Int(threshold * 100))%")
    }
}