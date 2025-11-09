//
//  TFKLogger.swift
//  TrustformersKit
//
//  Logging configuration and utilities
//  Copyright (c) 2024 TrustformeRS. All rights reserved.
//

import Foundation
import os.log

/// Logging levels
public enum TFKLogLevel: Int {
    case verbose = 0
    case debug = 1
    case info = 2
    case warning = 3
    case error = 4
    
    var name: String {
        switch self {
        case .verbose: return "VERBOSE"
        case .debug: return "DEBUG"
        case .info: return "INFO"
        case .warning: return "WARNING"
        case .error: return "ERROR"
        }
    }
    
    var emoji: String {
        switch self {
        case .verbose: return "🔍"
        case .debug: return "🐛"
        case .info: return "ℹ️"
        case .warning: return "⚠️"
        case .error: return "❌"
        }
    }
}

/// Logging configuration
public class TFKLogger: NSObject {
    
    // MARK: - Properties
    
    /// Current log level
    private static var logLevel: TFKLogLevel = .info
    
    /// Custom log handler
    private static var customHandler: ((TFKLogLevel, String) -> Void)?
    
    /// OS Log subsystem
    private static let subsystem = "com.trustformers.kit"
    
    /// OS Log categories
    private static let loggers: [TFKLogLevel: OSLog] = [
        .verbose: OSLog(subsystem: subsystem, category: "verbose"),
        .debug: OSLog(subsystem: subsystem, category: "debug"),
        .info: OSLog(subsystem: subsystem, category: "info"),
        .warning: OSLog(subsystem: subsystem, category: "warning"),
        .error: OSLog(subsystem: subsystem, category: "error")
    ]
    
    /// Log file URL (if file logging is enabled)
    private static var logFileURL: URL?
    
    /// Date formatter for log timestamps
    private static let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return formatter
    }()
    
    // MARK: - Configuration
    
    /// Set log level
    public static func setLogLevel(_ level: TFKLogLevel) {
        logLevel = level
        
        // Also update Rust logging level
        tfk_set_log_level(Int32(level.rawValue))
    }
    
    /// Set custom log handler
    public static func setLogHandler(_ handler: @escaping (TFKLogLevel, String) -> Void) {
        customHandler = handler
    }
    
    /// Enable file logging
    public static func enableFileLogging(to url: URL? = nil) {
        if let url = url {
            logFileURL = url
        } else {
            // Default log file in documents directory
            if let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
                let logsDirectory = documentsPath.appendingPathComponent("TrustformersKit/Logs")
                try? FileManager.default.createDirectory(at: logsDirectory, withIntermediateDirectories: true)
                
                let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .short, timeStyle: .none)
                    .replacingOccurrences(of: "/", with: "-")
                logFileURL = logsDirectory.appendingPathComponent("tfk-\(timestamp).log")
            }
        }
        
        // Create log file if it doesn't exist
        if let logFileURL = logFileURL, !FileManager.default.fileExists(atPath: logFileURL.path) {
            FileManager.default.createFile(atPath: logFileURL.path, contents: nil)
        }
    }
    
    /// Disable file logging
    public static func disableFileLogging() {
        logFileURL = nil
    }
    
    // MARK: - Logging Methods
    
    /// Log a message
    public static func log(level: TFKLogLevel, message: String, file: String = #file, function: String = #function, line: Int = #line) {
        // Check if we should log this level
        guard level.rawValue >= logLevel.rawValue else { return }
        
        // Format message with metadata
        let fileName = URL(fileURLWithPath: file).lastPathComponent
        let formattedMessage = "[\(fileName):\(line)] \(function) - \(message)"
        
        // Call custom handler if set
        if let handler = customHandler {
            handler(level, formattedMessage)
        }
        
        // Log to OS Log
        if #available(iOS 14.0, *) {
            let logger = loggers[level] ?? loggers[.info]!
            
            switch level {
            case .verbose, .debug:
                os_log(.debug, log: logger, "%{public}@", formattedMessage)
            case .info:
                os_log(.info, log: logger, "%{public}@", formattedMessage)
            case .warning:
                os_log(.default, log: logger, "%{public}@", formattedMessage)
            case .error:
                os_log(.error, log: logger, "%{public}@", formattedMessage)
            }
        } else {
            // Fallback for older iOS versions
            let prefix = "\(level.emoji) [\(level.name)]"
            print("\(prefix) \(formattedMessage)")
        }
        
        // Log to file if enabled
        if let logFileURL = logFileURL {
            let timestamp = dateFormatter.string(from: Date())
            let logLine = "\(timestamp) [\(level.name)] \(formattedMessage)\n"
            
            if let data = logLine.data(using: .utf8) {
                try? data.append(to: logFileURL)
            }
        }
    }
    
    /// Log verbose message
    public static func verbose(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        log(level: .verbose, message: message, file: file, function: function, line: line)
    }
    
    /// Log debug message
    public static func debug(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        log(level: .debug, message: message, file: file, function: function, line: line)
    }
    
    /// Log info message
    public static func info(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        log(level: .info, message: message, file: file, function: function, line: line)
    }
    
    /// Log warning message
    public static func warning(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        log(level: .warning, message: message, file: file, function: function, line: line)
    }
    
    /// Log error message
    public static func error(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        log(level: .error, message: message, file: file, function: function, line: line)
    }
    
    /// Log error with Error object
    public static func error(_ error: Error, message: String? = nil, file: String = #file, function: String = #function, line: Int = #line) {
        let errorMessage = message ?? "Error occurred"
        let fullMessage = "\(errorMessage): \(error.localizedDescription)"
        log(level: .error, message: fullMessage, file: file, function: function, line: line)
    }
}

// MARK: - Performance Logging

extension TFKLogger {
    
    /// Log performance metrics
    public static func logPerformance(operation: String, time: TimeInterval, memory: Int? = nil) {
        var message = "\(operation) completed in \(String(format: "%.2f", time * 1000)) ms"
        
        if let memory = memory {
            message += " (Memory: \(memory) MB)"
        }
        
        log(level: .info, message: message)
    }
    
    /// Measure and log execution time
    public static func measure<T>(operation: String, block: () throws -> T) rethrows -> T {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        do {
            let result = try block()
            let endTime = CFAbsoluteTimeGetCurrent()
            logPerformance(operation: operation, time: endTime - startTime)
            return result
        } catch {
            let endTime = CFAbsoluteTimeGetCurrent()
            logPerformance(operation: "\(operation) (failed)", time: endTime - startTime)
            throw error
        }
    }
}

// MARK: - Log Export

extension TFKLogger {
    
    /// Get log file contents
    public static func getLogFileContents() -> String? {
        guard let logFileURL = logFileURL else { return nil }
        
        do {
            return try String(contentsOf: logFileURL, encoding: .utf8)
        } catch {
            self.error(error, message: "Failed to read log file")
            return nil
        }
    }
    
    /// Clear log file
    public static func clearLogFile() {
        guard let logFileURL = logFileURL else { return }
        
        do {
            try "".write(to: logFileURL, atomically: true, encoding: .utf8)
            info("Log file cleared")
        } catch {
            self.error(error, message: "Failed to clear log file")
        }
    }
    
    /// Export logs to URL
    public static func exportLogs(to url: URL) throws {
        guard let contents = getLogFileContents() else {
            throw TFKError.inferenceFailed(reason: "No log file available")
        }
        
        try contents.write(to: url, atomically: true, encoding: .utf8)
    }
}

// MARK: - Data Extension

private extension Data {
    
    /// Append data to file
    func append(to url: URL) throws {
        if let fileHandle = FileHandle(forWritingAtPath: url.path) {
            defer { fileHandle.closeFile() }
            fileHandle.seekToEndOfFile()
            fileHandle.write(self)
        } else {
            try write(to: url)
        }
    }
}