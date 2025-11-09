//
//  TFKInferenceEngine+Combine.swift
//  TrustformersKit
//
//  Combine reactive programming extensions for TFKInferenceEngine
//  Copyright (c) 2024 TrustformeRS. All rights reserved.
//

import Foundation
import Combine
import CoreML
import Metal

@available(iOS 13.0, macOS 10.15, tvOS 13.0, watchOS 6.0, *)
extension TFKInferenceEngine {
    
    // MARK: - Publishers
    
    /// Publisher for inference results
    public func inferencePublisher(model: TFKModel, input: TFKTensor) -> AnyPublisher<TFKInferenceResult, Never> {
        return Future<TFKInferenceResult, Never> { [weak self] promise in
            guard let self = self else {
                promise(.success(TFKInferenceResult(
                    output: TFKTensor(floats: [], shape: [0]),
                    inferenceTime: 0,
                    memoryUsed: 0,
                    success: false,
                    error: TFKError.engineNotInitialized
                )))
                return
            }
            
            DispatchQueue.global(qos: .userInitiated).async {
                let result = self.performInference(model, input: input)
                promise(.success(result))
            }
        }
        .eraseToAnyPublisher()
    }
    
    /// Publisher for batch inference results
    public func batchInferencePublisher(model: TFKModel, inputs: [TFKTensor]) -> AnyPublisher<[TFKInferenceResult], Never> {
        return Future<[TFKInferenceResult], Never> { [weak self] promise in
            guard let self = self else {
                promise(.success([]))
                return
            }
            
            DispatchQueue.global(qos: .userInitiated).async {
                let results = self.performBatchInference(model, inputs: inputs)
                promise(.success(results))
            }
        }
        .eraseToAnyPublisher()
    }
    
    /// Publisher for model loading
    public func loadModelPublisher(at path: String, config: TFKModelConfig) -> AnyPublisher<TFKModel, TFKError> {
        return Future<TFKModel, TFKError> { [weak self] promise in
            guard let self = self else {
                promise(.failure(.engineNotInitialized))
                return
            }
            
            DispatchQueue.global(qos: .utility).async {
                do {
                    let model = try self.loadModel(at: path, config: config)
                    promise(.success(model))
                } catch let error as TFKError {
                    promise(.failure(error))
                } catch {
                    promise(.failure(.modelLoadFailed(reason: error.localizedDescription)))
                }
            }
        }
        .eraseToAnyPublisher()
    }
    
    /// Publisher for bundle model loading
    public func loadModelPublisher(bundleResource name: String, type: String, config: TFKModelConfig) -> AnyPublisher<TFKModel, TFKError> {
        return Future<TFKModel, TFKError> { [weak self] promise in
            guard let self = self else {
                promise(.failure(.engineNotInitialized))
                return
            }
            
            DispatchQueue.global(qos: .utility).async {
                do {
                    let model = try self.loadModel(bundleResource: name, type: type, config: config)
                    promise(.success(model))
                } catch let error as TFKError {
                    promise(.failure(error))
                } catch {
                    promise(.failure(.modelLoadFailed(reason: error.localizedDescription)))
                }
            }
        }
        .eraseToAnyPublisher()
    }
    
    // MARK: - Reactive State Publishers
    
    /// Publisher for performance statistics updates
    public var performanceStatsPublisher: AnyPublisher<TFKPerformanceStats, Never> {
        return performanceStatsSubject.eraseToAnyPublisher()
    }
    
    /// Publisher for thermal state changes
    public var thermalStatePublisher: AnyPublisher<TFKThermalState, Never> {
        return thermalStateSubject.eraseToAnyPublisher()
    }
    
    /// Publisher for memory pressure notifications
    public var memoryPressurePublisher: AnyPublisher<TFKMemoryOptimization, Never> {
        return memoryPressureSubject.eraseToAnyPublisher()
    }
    
    /// Publisher for device info changes
    public var deviceInfoPublisher: AnyPublisher<TFKDeviceInfo, Never> {
        return deviceInfoSubject.eraseToAnyPublisher()
    }
    
    // MARK: - Streaming Inference
    
    /// Publisher for continuous inference streaming
    public func streamingInferencePublisher(model: TFKModel) -> (AnyPublisher<TFKInferenceResult, Never>, (TFKTensor) -> Void) {
        let inputSubject = PassthroughSubject<TFKTensor, Never>()
        
        let outputPublisher = inputSubject
            .flatMap { [weak self] input -> AnyPublisher<TFKInferenceResult, Never> in
                guard let self = self else {
                    return Just(TFKInferenceResult(
                        output: TFKTensor(floats: [], shape: [0]),
                        inferenceTime: 0,
                        memoryUsed: 0,
                        success: false,
                        error: TFKError.engineNotInitialized
                    )).eraseToAnyPublisher()
                }
                
                return self.inferencePublisher(model: model, input: input)
            }
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
        
        let inputHandler: (TFKTensor) -> Void = { tensor in
            inputSubject.send(tensor)
        }
        
        return (outputPublisher, inputHandler)
    }
    
    // MARK: - Advanced Reactive Patterns
    
    /// Debounced inference publisher (useful for real-time input)
    public func debouncedInferencePublisher(model: TFKModel, debounceInterval: DispatchQueue.SchedulerTimeType.Stride = .milliseconds(100)) -> (AnyPublisher<TFKInferenceResult, Never>, (TFKTensor) -> Void) {
        let inputSubject = PassthroughSubject<TFKTensor, Never>()
        
        let outputPublisher = inputSubject
            .debounce(for: debounceInterval, scheduler: DispatchQueue.main)
            .flatMap { [weak self] input -> AnyPublisher<TFKInferenceResult, Never> in
                guard let self = self else {
                    return Just(TFKInferenceResult(
                        output: TFKTensor(floats: [], shape: [0]),
                        inferenceTime: 0,
                        memoryUsed: 0,
                        success: false,
                        error: TFKError.engineNotInitialized
                    )).eraseToAnyPublisher()
                }
                
                return self.inferencePublisher(model: model, input: input)
            }
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
        
        let inputHandler: (TFKTensor) -> Void = { tensor in
            inputSubject.send(tensor)
        }
        
        return (outputPublisher, inputHandler)
    }
    
    /// Throttled inference publisher (rate limiting)
    public func throttledInferencePublisher(model: TFKModel, throttleInterval: DispatchQueue.SchedulerTimeType.Stride = .milliseconds(50)) -> (AnyPublisher<TFKInferenceResult, Never>, (TFKTensor) -> Void) {
        let inputSubject = PassthroughSubject<TFKTensor, Never>()
        
        let outputPublisher = inputSubject
            .throttle(for: throttleInterval, scheduler: DispatchQueue.main, latest: true)
            .flatMap { [weak self] input -> AnyPublisher<TFKInferenceResult, Never> in
                guard let self = self else {
                    return Just(TFKInferenceResult(
                        output: TFKTensor(floats: [], shape: [0]),
                        inferenceTime: 0,
                        memoryUsed: 0,
                        success: false,
                        error: TFKError.engineNotInitialized
                    )).eraseToAnyPublisher()
                }
                
                return self.inferencePublisher(model: model, input: input)
            }
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
        
        let inputHandler: (TFKTensor) -> Void = { tensor in
            inputSubject.send(tensor)
        }
        
        return (outputPublisher, inputHandler)
    }
    
    /// Combined inference with timeout and retry
    public func robustInferencePublisher(model: TFKModel, input: TFKTensor, timeout: TimeInterval = 30.0, retryCount: Int = 3) -> AnyPublisher<TFKInferenceResult, TFKError> {
        return inferencePublisher(model: model, input: input)
            .timeout(.seconds(timeout), scheduler: DispatchQueue.main, customError: { TFKError.inferenceFailed(reason: "Timeout") })
            .compactMap { result in
                return result.success ? result : nil
            }
            .catch { error -> AnyPublisher<TFKInferenceResult, TFKError> in
                if retryCount > 0 {
                    return self.robustInferencePublisher(model: model, input: input, timeout: timeout, retryCount: retryCount - 1)
                } else {
                    return Fail(error: error as? TFKError ?? TFKError.inferenceFailed(reason: error.localizedDescription))
                        .eraseToAnyPublisher()
                }
            }
            .eraseToAnyPublisher()
    }
    
    // MARK: - Combine Pipeline Utilities
    
    /// Chain multiple inference operations
    public func chainedInferencePublisher(models: [TFKModel], initialInput: TFKTensor) -> AnyPublisher<TFKInferenceResult, Never> {
        return models.enumerated().reduce(
            Just(TFKInferenceResult(
                output: initialInput,
                inferenceTime: 0,
                memoryUsed: 0,
                success: true,
                error: nil
            )).eraseToAnyPublisher()
        ) { publisher, indexedModel in
            let (index, model) = indexedModel
            return publisher.flatMap { [weak self] previousResult -> AnyPublisher<TFKInferenceResult, Never> in
                guard let self = self, previousResult.success else {
                    return Just(previousResult).eraseToAnyPublisher()
                }
                
                return self.inferencePublisher(model: model, input: previousResult.output)
                    .map { result in
                        // Accumulate timing and memory usage
                        return TFKInferenceResult(
                            output: result.output,
                            inferenceTime: previousResult.inferenceTime + result.inferenceTime,
                            memoryUsed: max(previousResult.memoryUsed, result.memoryUsed),
                            success: result.success,
                            error: result.error
                        )
                    }
                    .eraseToAnyPublisher()
            }
            .eraseToAnyPublisher()
        }
    }
}

// MARK: - Internal Subjects for Reactive State

@available(iOS 13.0, macOS 10.15, tvOS 13.0, watchOS 6.0, *)
extension TFKInferenceEngine {
    private static var performanceStatsSubject = PassthroughSubject<TFKPerformanceStats, Never>()
    private static var thermalStateSubject = PassthroughSubject<TFKThermalState, Never>()
    private static var memoryPressureSubject = PassthroughSubject<TFKMemoryOptimization, Never>()
    private static var deviceInfoSubject = PassthroughSubject<TFKDeviceInfo, Never>()
    
    internal func publishPerformanceStats(_ stats: TFKPerformanceStats) {
        Self.performanceStatsSubject.send(stats)
    }
    
    internal func publishThermalState(_ state: TFKThermalState) {
        Self.thermalStateSubject.send(state)
    }
    
    internal func publishMemoryPressure(_ level: TFKMemoryOptimization) {
        Self.memoryPressureSubject.send(level)
    }
    
    internal func publishDeviceInfo(_ info: TFKDeviceInfo) {
        Self.deviceInfoSubject.send(info)
    }
}