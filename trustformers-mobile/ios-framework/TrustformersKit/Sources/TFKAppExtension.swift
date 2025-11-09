//
//  TFKAppExtension.swift
//  TrustformersKit
//
//  App Extension support for background ML inference
//  Copyright (c) 2024 TrustformeRS. All rights reserved.
//

import Foundation
import CoreML
import Metal
import UserNotifications
import BackgroundTasks
import CryptoKit

/// App Extension support for background ML inference
@available(iOS 13.0, *)
public class TFKAppExtension: NSObject {
    
    // MARK: - Background Task Identifiers
    
    /// Background app refresh identifier
    public static let backgroundAppRefreshIdentifier = "com.trustformers.backgroundInference"
    
    /// Background processing identifier
    public static let backgroundProcessingIdentifier = "com.trustformers.backgroundProcessing"
    
    // MARK: - Properties
    
    private static let shared = TFKAppExtension()
    private var backgroundTasks: [String: BGTask] = [:]
    private var backgroundQueue = DispatchQueue(label: "com.trustformers.backgroundExtension", qos: .background)
    
    // Model cache for background processing
    private var cachedModels: [String: TFKModel] = [:]
    private var modelCache: NSCache<NSString, TFKModel> = NSCache()
    
    // Background processing configuration
    private var backgroundConfig = TFKBackgroundConfig()
    
    // MARK: - Initialization
    
    override init() {
        super.init()
        
        // Configure model cache
        modelCache.countLimit = 3
        modelCache.totalCostLimit = 100 * 1024 * 1024 // 100MB
        
        // Setup background task handlers
        setupBackgroundTaskHandlers()
    }
    
    // MARK: - Public Interface
    
    /// Register background tasks
    public static func registerBackgroundTasks() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: backgroundAppRefreshIdentifier,
            using: nil
        ) { task in
            shared.handleBackgroundAppRefresh(task as! BGAppRefreshTask)
        }
        
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: backgroundProcessingIdentifier,
            using: nil
        ) { task in
            shared.handleBackgroundProcessing(task as! BGProcessingTask)
        }
    }
    
    /// Schedule background model update
    public static func scheduleBackgroundModelUpdate() {
        let request = BGAppRefreshTaskRequest(identifier: backgroundAppRefreshIdentifier)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60) // 15 minutes
        
        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            TFKLogger.log(level: .error, message: "Failed to schedule background model update: \(error)")
        }
    }
    
    /// Schedule background processing
    public static func scheduleBackgroundProcessing() {
        let request = BGProcessingTaskRequest(identifier: backgroundProcessingIdentifier)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 60 * 60) // 1 hour
        request.requiresNetworkConnectivity = true
        request.requiresExternalPower = false
        
        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            TFKLogger.log(level: .error, message: "Failed to schedule background processing: \(error)")
        }
    }
    
    /// Configure background inference settings
    public static func configureBackgroundInference(_ config: TFKBackgroundConfig) {
        shared.backgroundConfig = config
    }
    
    /// Perform background inference
    public static func performBackgroundInference(
        modelPath: String,
        inputData: Data,
        completion: @escaping (Result<TFKInferenceResult, Error>) -> Void
    ) {
        shared.backgroundQueue.async {
            shared.executeBackgroundInference(modelPath: modelPath, inputData: inputData, completion: completion)
        }
    }
    
    // MARK: - Background Task Handlers
    
    private func setupBackgroundTaskHandlers() {
        // Handle app entering background
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appDidEnterBackground),
            name: UIApplication.didEnterBackgroundNotification,
            object: nil
        )
        
        // Handle app becoming active
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appDidBecomeActive),
            name: UIApplication.didBecomeActiveNotification,
            object: nil
        )
    }
    
    @objc private func appDidEnterBackground() {
        // Cache frequently used models
        cacheFrequentModels()
        
        // Schedule background tasks
        TFKAppExtension.scheduleBackgroundModelUpdate()
        TFKAppExtension.scheduleBackgroundProcessing()
    }
    
    @objc private func appDidBecomeActive() {
        // Clear expired cache
        clearExpiredCache()
    }
    
    private func handleBackgroundAppRefresh(_ task: BGAppRefreshTask) {
        TFKLogger.log(level: .info, message: "Background app refresh started")
        
        // Set expiration handler
        task.expirationHandler = {
            TFKLogger.log(level: .warning, message: "Background app refresh expired")
            task.setTaskCompleted(success: false)
        }
        
        // Perform background model update
        backgroundQueue.async { [weak self] in
            self?.performBackgroundModelUpdate { success in
                DispatchQueue.main.async {
                    task.setTaskCompleted(success: success)
                    
                    // Schedule next refresh
                    TFKAppExtension.scheduleBackgroundModelUpdate()
                }
            }
        }
    }
    
    private func handleBackgroundProcessing(_ task: BGProcessingTask) {
        TFKLogger.log(level: .info, message: "Background processing started")
        
        // Set expiration handler
        task.expirationHandler = {
            TFKLogger.log(level: .warning, message: "Background processing expired")
            task.setTaskCompleted(success: false)
        }
        
        // Perform background processing
        backgroundQueue.async { [weak self] in
            self?.performBackgroundProcessing { success in
                DispatchQueue.main.async {
                    task.setTaskCompleted(success: success)
                    
                    // Schedule next processing
                    TFKAppExtension.scheduleBackgroundProcessing()
                }
            }
        }
    }
    
    // MARK: - Background Operations
    
    private func performBackgroundModelUpdate(completion: @escaping (Bool) -> Void) {
        guard backgroundConfig.enableModelUpdates else {
            completion(false)
            return
        }
        
        // Check for model updates
        checkForModelUpdates { [weak self] updates in
            guard let self = self, !updates.isEmpty else {
                completion(true)
                return
            }
            
            // Download and cache updated models
            self.downloadModelUpdates(updates) { success in
                if success {
                    self.sendBackgroundNotification(title: "Models Updated", body: "New model versions available")
                }
                completion(success)
            }
        }
    }
    
    private func performBackgroundProcessing(completion: @escaping (Bool) -> Void) {
        guard backgroundConfig.enableBackgroundProcessing else {
            completion(false)
            return
        }
        
        // Process queued inference requests
        processQueuedInferences { [weak self] processed in
            guard let self = self else {
                completion(false)
                return
            }
            
            // Perform maintenance tasks
            self.performMaintenanceTasks()
            
            completion(processed > 0)
        }
    }
    
    private func executeBackgroundInference(
        modelPath: String,
        inputData: Data,
        completion: @escaping (Result<TFKInferenceResult, Error>) -> Void
    ) {
        // Load model from cache or disk
        loadModelForBackground(path: modelPath) { [weak self] result in
            switch result {
            case .success(let model):
                self?.performBackgroundInferenceWithModel(model: model, inputData: inputData, completion: completion)
            case .failure(let error):
                completion(.failure(error))
            }
        }
    }
    
    private func performBackgroundInferenceWithModel(
        model: TFKModel,
        inputData: Data,
        completion: @escaping (Result<TFKInferenceResult, Error>) -> Void
    ) {
        do {
            // Convert input data to tensor
            let inputTensor = try createTensorFromData(inputData)
            
            // Configure for background execution
            var backgroundModelConfig = model.config
            backgroundModelConfig.numThreads = 1
            backgroundModelConfig.memoryOptimization = .balanced
            backgroundModelConfig.enableBatching = false
            
            model.updateConfig(backgroundModelConfig)
            
            // Perform inference with timeout
            let timeoutInterval: TimeInterval = 30.0 // 30 seconds timeout
            let timeoutTimer = DispatchSource.makeTimerSource(queue: backgroundQueue)
            
            var isCompleted = false
            
            timeoutTimer.schedule(deadline: .now() + timeoutInterval)
            timeoutTimer.setEventHandler {
                if !isCompleted {
                    isCompleted = true
                    completion(.failure(TFKError.inferenceFailed(reason: "Background inference timeout")))
                }
            }
            
            timeoutTimer.resume()
            
            // Execute inference
            backgroundQueue.async {
                let result = TFKInferenceEngine.shared.performInference(model, input: inputTensor)
                
                timeoutTimer.cancel()
                
                if !isCompleted {
                    isCompleted = true
                    completion(.success(result))
                }
            }
            
        } catch {
            completion(.failure(error))
        }
    }
    
    // MARK: - Model Management
    
    private func loadModelForBackground(path: String, completion: @escaping (Result<TFKModel, Error>) -> Void) {
        // Check cache first
        if let cachedModel = modelCache.object(forKey: path as NSString) {
            completion(.success(cachedModel))
            return
        }
        
        // Load from disk
        backgroundQueue.async {
            do {
                let config = TFKModelConfig.backgroundOptimizedConfig()
                let model = try TFKInferenceEngine.shared.loadModel(at: path, config: config)
                
                // Cache the model
                self.modelCache.setObject(model, forKey: path as NSString)
                
                completion(.success(model))
            } catch {
                completion(.failure(error))
            }
        }
    }
    
    private func cacheFrequentModels() {
        // Cache frequently used models for background access
        let frequentModels = backgroundConfig.frequentModelPaths
        
        for modelPath in frequentModels {
            loadModelForBackground(path: modelPath) { result in
                switch result {
                case .success:
                    TFKLogger.log(level: .info, message: "Cached model for background: \(modelPath)")
                case .failure(let error):
                    TFKLogger.log(level: .error, message: "Failed to cache model: \(error)")
                }
            }
        }
    }
    
    private func clearExpiredCache() {
        // Remove expired models from cache
        modelCache.removeAllObjects()
        cachedModels.removeAll()
    }
    
    // MARK: - Update Management
    
    private func checkForModelUpdates(completion: @escaping ([TFKModelUpdate]) -> Void) {
        // Check for model updates from server
        // This would typically involve checking version numbers or checksums
        
        guard let updateURL = backgroundConfig.updateServerURL else {
            completion([])
            return
        }
        
        let task = URLSession.shared.dataTask(with: updateURL) { data, response, error in
            guard let data = data, error == nil else {
                completion([])
                return
            }
            
            do {
                let updates = try JSONDecoder().decode([TFKModelUpdate].self, from: data)
                completion(updates)
            } catch {
                TFKLogger.log(level: .error, message: "Failed to decode model updates: \(error)")
                completion([])
            }
        }
        
        task.resume()
    }
    
    private func downloadModelUpdates(_ updates: [TFKModelUpdate], completion: @escaping (Bool) -> Void) {
        let group = DispatchGroup()
        var downloadErrors: [Error] = []
        
        for update in updates {
            group.enter()
            
            downloadModelUpdate(update) { error in
                if let error = error {
                    downloadErrors.append(error)
                }
                group.leave()
            }
        }
        
        group.notify(queue: backgroundQueue) {
            completion(downloadErrors.isEmpty)
        }
    }
    
    private func downloadModelUpdate(_ update: TFKModelUpdate, completion: @escaping (Error?) -> Void) {
        let task = URLSession.shared.downloadTask(with: update.downloadURL) { tempURL, response, error in
            if let error = error {
                completion(error)
                return
            }
            
            guard let tempURL = tempURL else {
                completion(TFKError.modelLoadFailed(reason: "Invalid download URL"))
                return
            }
            
            do {
                // Move downloaded file to final location
                let finalURL = URL(fileURLWithPath: update.localPath)
                try FileManager.default.moveItem(at: tempURL, to: finalURL)
                
                // Verify model integrity
                try self.verifyModelIntegrity(at: finalURL, expectedChecksum: update.checksum)
                
                completion(nil)
            } catch {
                completion(error)
            }
        }
        
        task.resume()
    }
    
    private func verifyModelIntegrity(at url: URL, expectedChecksum: String) throws {
        let data = try Data(contentsOf: url)
        let actualChecksum = data.sha256
        
        guard actualChecksum == expectedChecksum else {
            throw TFKError.modelLoadFailed(reason: "Model integrity verification failed")
        }
    }
    
    // MARK: - Background Processing
    
    private func processQueuedInferences(completion: @escaping (Int) -> Void) {
        // Process queued inference requests
        let queuedRequests = loadQueuedInferences()
        
        guard !queuedRequests.isEmpty else {
            completion(0)
            return
        }
        
        let group = DispatchGroup()
        var processedCount = 0
        
        for request in queuedRequests {
            group.enter()
            
            executeBackgroundInference(
                modelPath: request.modelPath,
                inputData: request.inputData
            ) { result in
                // Save result
                self.saveInferenceResult(request.id, result: result)
                processedCount += 1
                group.leave()
            }
        }
        
        group.notify(queue: backgroundQueue) {
            // Clear processed requests
            self.clearProcessedInferences(queuedRequests.map { $0.id })
            completion(processedCount)
        }
    }
    
    private func performMaintenanceTasks() {
        // Clean up temporary files
        cleanupTemporaryFiles()
        
        // Optimize model cache
        optimizeModelCache()
        
        // Update performance statistics
        updatePerformanceStatistics()
    }
    
    // MARK: - Helper Methods
    
    private func createTensorFromData(_ data: Data) throws -> TFKTensor {
        // Convert data to float array
        let floatCount = data.count / MemoryLayout<Float>.size
        let floats = data.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: Float.self))
        }
        
        return TFKTensor(floats: floats, shape: [floatCount])
    }
    
    private func sendBackgroundNotification(title: String, body: String) {
        guard backgroundConfig.enableNotifications else { return }
        
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = UNNotificationSound.default
        
        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil
        )
        
        UNUserNotificationCenter.current().add(request)
    }
    
    private func loadQueuedInferences() -> [TFKQueuedInference] {
        // Load queued inference requests from persistent storage
        guard let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return []
        }
        
        let queueURL = documentsPath.appendingPathComponent("inference_queue.json")
        
        do {
            let data = try Data(contentsOf: queueURL)
            return try JSONDecoder().decode([TFKQueuedInference].self, from: data)
        } catch {
            return []
        }
    }
    
    private func saveInferenceResult(_ id: String, result: Result<TFKInferenceResult, Error>) {
        // Save inference result to persistent storage
        guard let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return
        }
        
        let resultURL = documentsPath.appendingPathComponent("inference_results.json")
        var results: [String: TFKStoredInferenceResult] = [:]
        
        // Load existing results
        if let data = try? Data(contentsOf: resultURL),
           let existingResults = try? JSONDecoder().decode([String: TFKStoredInferenceResult].self, from: data) {
            results = existingResults
        }
        
        // Add new result
        results[id] = TFKStoredInferenceResult(id: id, result: result, timestamp: Date())
        
        // Save updated results
        do {
            let data = try JSONEncoder().encode(results)
            try data.write(to: resultURL)
        } catch {
            TFKLogger.log(level: .error, message: "Failed to save inference result: \(error)")
        }
    }
    
    private func clearProcessedInferences(_ ids: [String]) {
        // Remove processed inference requests from queue
        guard let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return
        }
        
        let queueURL = documentsPath.appendingPathComponent("inference_queue.json")
        
        do {
            let data = try Data(contentsOf: queueURL)
            var queue = try JSONDecoder().decode([TFKQueuedInference].self, from: data)
            
            queue.removeAll { ids.contains($0.id) }
            
            let updatedData = try JSONEncoder().encode(queue)
            try updatedData.write(to: queueURL)
        } catch {
            TFKLogger.log(level: .error, message: "Failed to clear processed inferences: \(error)")
        }
    }
    
    private func cleanupTemporaryFiles() {
        // Clean up temporary files older than 24 hours
        let tempDirectory = FileManager.default.temporaryDirectory
        let oneDayAgo = Date().addingTimeInterval(-24 * 60 * 60)
        
        do {
            let contents = try FileManager.default.contentsOfDirectory(at: tempDirectory, includingPropertiesForKeys: [.creationDateKey])
            
            for url in contents {
                let attributes = try url.resourceValues(forKeys: [.creationDateKey])
                if let creationDate = attributes.creationDate, creationDate < oneDayAgo {
                    try FileManager.default.removeItem(at: url)
                }
            }
        } catch {
            TFKLogger.log(level: .error, message: "Failed to clean up temporary files: \(error)")
        }
    }
    
    private func optimizeModelCache() {
        // Optimize model cache by removing least recently used models
        let maxCacheSize = 100 * 1024 * 1024 // 100MB
        let currentSize = modelCache.totalCostLimit
        
        if currentSize > maxCacheSize {
            modelCache.removeAllObjects()
            TFKLogger.log(level: .info, message: "Model cache optimized")
        }
    }
    
    private func updatePerformanceStatistics() {
        // Update performance statistics
        let stats = TFKInferenceEngine.shared.performanceStats
        
        // Log performance metrics
        TFKLogger.log(level: .info, message: "Performance stats updated: \(stats)")
    }
}

// MARK: - Supporting Types

/// Background configuration
public struct TFKBackgroundConfig {
    public var enableModelUpdates: Bool = true
    public var enableBackgroundProcessing: Bool = true
    public var enableNotifications: Bool = true
    public var frequentModelPaths: [String] = []
    public var updateServerURL: URL? = nil
    public var maxBackgroundTime: TimeInterval = 30.0
    
    public init() {}
}

/// Model update information
public struct TFKModelUpdate: Codable {
    public let modelId: String
    public let version: String
    public let downloadURL: URL
    public let localPath: String
    public let checksum: String
    public let size: Int64
    
    public init(modelId: String, version: String, downloadURL: URL, localPath: String, checksum: String, size: Int64) {
        self.modelId = modelId
        self.version = version
        self.downloadURL = downloadURL
        self.localPath = localPath
        self.checksum = checksum
        self.size = size
    }
}

/// Queued inference request
public struct TFKQueuedInference: Codable {
    public let id: String
    public let modelPath: String
    public let inputData: Data
    public let timestamp: Date
    
    public init(id: String, modelPath: String, inputData: Data, timestamp: Date) {
        self.id = id
        self.modelPath = modelPath
        self.inputData = inputData
        self.timestamp = timestamp
    }
}

/// Stored inference result
public struct TFKStoredInferenceResult: Codable {
    public let id: String
    public let success: Bool
    public let timestamp: Date
    public let error: String?
    
    public init(id: String, result: Result<TFKInferenceResult, Error>, timestamp: Date) {
        self.id = id
        self.timestamp = timestamp
        
        switch result {
        case .success:
            self.success = true
            self.error = nil
        case .failure(let error):
            self.success = false
            self.error = error.localizedDescription
        }
    }
}

// MARK: - Extensions

extension TFKModelConfig {
    /// Configuration optimized for background execution
    public static func backgroundOptimizedConfig() -> TFKModelConfig {
        var config = TFKModelConfig()
        config.numThreads = 1
        config.memoryOptimization = .balanced
        config.enableBatching = false
        config.computeBackend = .auto
        return config
    }
}

extension Data {
    /// SHA256 checksum
    var sha256: String {
        let digest = SHA256.hash(data: self)
        return digest.compactMap { String(format: "%02x", $0) }.joined()
    }
}