using System;

namespace TrustformeRS
{
    /// <summary>
    /// Error codes returned by TrustformersRS native library
    /// </summary>
    public enum TrustformersError : int
    {
        /// <summary>Operation completed successfully</summary>
        Success = 0,
        
        /// <summary>Generic runtime error</summary>
        RuntimeError = 1,
        
        /// <summary>Invalid parameter provided</summary>
        InvalidParameter = 2,
        
        /// <summary>Null pointer passed where non-null expected</summary>
        NullPointer = 3,
        
        /// <summary>Memory allocation failed</summary>
        OutOfMemory = 4,
        
        /// <summary>File not found or inaccessible</summary>
        FileNotFound = 5,
        
        /// <summary>Invalid file format</summary>
        InvalidFormat = 6,
        
        /// <summary>Network operation failed</summary>
        NetworkError = 7,
        
        /// <summary>Operation timed out</summary>
        Timeout = 8,
        
        /// <summary>Resource is busy or locked</summary>
        Busy = 9,
        
        /// <summary>Operation not supported</summary>
        NotSupported = 10,
        
        /// <summary>Permission denied</summary>
        PermissionDenied = 11,
        
        /// <summary>Already initialized</summary>
        AlreadyInitialized = 12,
        
        /// <summary>Not initialized</summary>
        NotInitialized = 13,
        
        /// <summary>Serialization/deserialization error</summary>
        SerializationError = 14,
        
        /// <summary>Memory corruption detected</summary>
        MemoryCorruption = 15,
        
        /// <summary>Buffer overflow detected</summary>
        BufferOverflow = 16,
        
        /// <summary>Use after free detected</summary>
        UseAfterFree = 17,
        
        /// <summary>Double free detected</summary>
        DoubleFree = 18,
        
        /// <summary>Hardware acceleration not available</summary>
        HardwareNotAvailable = 19,
        
        /// <summary>CUDA operation failed</summary>
        CudaError = 20,
        
        /// <summary>ROCm operation failed</summary>
        RocmError = 21,
        
        /// <summary>Metal operation failed</summary>
        MetalError = 22,
        
        /// <summary>Model loading failed</summary>
        ModelLoadError = 23,
        
        /// <summary>Tokenization failed</summary>
        TokenizationError = 24,
        
        /// <summary>Pipeline operation failed</summary>
        PipelineError = 25,
        
        /// <summary>Configuration error</summary>
        ConfigError = 26,
        
        /// <summary>Version mismatch</summary>
        VersionMismatch = 27,
        
        /// <summary>License check failed</summary>
        LicenseError = 28,
        
        /// <summary>Authentication failed</summary>
        AuthenticationError = 29,
        
        /// <summary>Rate limit exceeded</summary>
        RateLimitExceeded = 30
    }

    /// <summary>
    /// Exception thrown by TrustformersRS operations
    /// </summary>
    public class TrustformersException : Exception
    {
        /// <summary>
        /// The TrustformersRS error code that caused this exception
        /// </summary>
        public TrustformersError ErrorCode { get; }

        /// <summary>
        /// Creates a new TrustformersException with the specified error code
        /// </summary>
        /// <param name="errorCode">The error code</param>
        public TrustformersException(TrustformersError errorCode)
            : base(GetErrorMessage(errorCode))
        {
            ErrorCode = errorCode;
        }

        /// <summary>
        /// Creates a new TrustformersException with the specified error code and message
        /// </summary>
        /// <param name="errorCode">The error code</param>
        /// <param name="message">The error message</param>
        public TrustformersException(TrustformersError errorCode, string message)
            : base(string.IsNullOrEmpty(message) ? GetErrorMessage(errorCode) : $"{GetErrorMessage(errorCode)}: {message}")
        {
            ErrorCode = errorCode;
        }

        /// <summary>
        /// Creates a new TrustformersException with the specified error code, message, and inner exception
        /// </summary>
        /// <param name="errorCode">The error code</param>
        /// <param name="message">The error message</param>
        /// <param name="innerException">The inner exception</param>
        public TrustformersException(TrustformersError errorCode, string message, Exception innerException)
            : base(string.IsNullOrEmpty(message) ? GetErrorMessage(errorCode) : $"{GetErrorMessage(errorCode)}: {message}", innerException)
        {
            ErrorCode = errorCode;
        }

        /// <summary>
        /// Gets a human-readable error message for the specified error code
        /// </summary>
        /// <param name="errorCode">The error code</param>
        /// <returns>Error message</returns>
        public static string GetErrorMessage(TrustformersError errorCode)
        {
            return errorCode switch
            {
                TrustformersError.Success => "Operation completed successfully",
                TrustformersError.RuntimeError => "Runtime error occurred",
                TrustformersError.InvalidParameter => "Invalid parameter provided",
                TrustformersError.NullPointer => "Null pointer passed where non-null expected",
                TrustformersError.OutOfMemory => "Memory allocation failed",
                TrustformersError.FileNotFound => "File not found or inaccessible",
                TrustformersError.InvalidFormat => "Invalid file format",
                TrustformersError.NetworkError => "Network operation failed",
                TrustformersError.Timeout => "Operation timed out",
                TrustformersError.Busy => "Resource is busy or locked",
                TrustformersError.NotSupported => "Operation not supported",
                TrustformersError.PermissionDenied => "Permission denied",
                TrustformersError.AlreadyInitialized => "Already initialized",
                TrustformersError.NotInitialized => "Not initialized",
                TrustformersError.SerializationError => "Serialization/deserialization error",
                TrustformersError.MemoryCorruption => "Memory corruption detected",
                TrustformersError.BufferOverflow => "Buffer overflow detected",
                TrustformersError.UseAfterFree => "Use after free detected",
                TrustformersError.DoubleFree => "Double free detected",
                TrustformersError.HardwareNotAvailable => "Hardware acceleration not available",
                TrustformersError.CudaError => "CUDA operation failed",
                TrustformersError.RocmError => "ROCm operation failed",
                TrustformersError.MetalError => "Metal operation failed",
                TrustformersError.ModelLoadError => "Model loading failed",
                TrustformersError.TokenizationError => "Tokenization failed",
                TrustformersError.PipelineError => "Pipeline operation failed",
                TrustformersError.ConfigError => "Configuration error",
                TrustformersError.VersionMismatch => "Version mismatch",
                TrustformersError.LicenseError => "License check failed",
                TrustformersError.AuthenticationError => "Authentication failed",
                TrustformersError.RateLimitExceeded => "Rate limit exceeded",
                _ => $"Unknown error ({(int)errorCode})"
            };
        }

        /// <summary>
        /// Determines if the error code represents a memory safety violation
        /// </summary>
        /// <param name="errorCode">The error code to check</param>
        /// <returns>True if it's a memory safety error</returns>
        public static bool IsMemoryError(TrustformersError errorCode)
        {
            return errorCode switch
            {
                TrustformersError.OutOfMemory or
                TrustformersError.MemoryCorruption or
                TrustformersError.BufferOverflow or
                TrustformersError.UseAfterFree or
                TrustformersError.DoubleFree => true,
                _ => false
            };
        }

        /// <summary>
        /// Determines if the error code represents a hardware-related error
        /// </summary>
        /// <param name="errorCode">The error code to check</param>
        /// <returns>True if it's a hardware error</returns>
        public static bool IsHardwareError(TrustformersError errorCode)
        {
            return errorCode switch
            {
                TrustformersError.HardwareNotAvailable or
                TrustformersError.CudaError or
                TrustformersError.RocmError or
                TrustformersError.MetalError => true,
                _ => false
            };
        }

        /// <summary>
        /// Determines if the error is recoverable (operation can be retried)
        /// </summary>
        /// <param name="errorCode">The error code to check</param>
        /// <returns>True if the error is potentially recoverable</returns>
        public static bool IsRecoverable(TrustformersError errorCode)
        {
            return errorCode switch
            {
                TrustformersError.NetworkError or
                TrustformersError.Timeout or
                TrustformersError.Busy or
                TrustformersError.OutOfMemory or
                TrustformersError.RateLimitExceeded => true,
                _ => false
            };
        }

        /// <summary>
        /// Gets the severity level of the error
        /// </summary>
        /// <param name="errorCode">The error code to check</param>
        /// <returns>Severity level (0=Info, 1=Warning, 2=Error, 3=Critical)</returns>
        public static int GetSeverity(TrustformersError errorCode)
        {
            return errorCode switch
            {
                TrustformersError.Success => 0,
                
                TrustformersError.Timeout or
                TrustformersError.Busy or
                TrustformersError.RateLimitExceeded => 1,
                
                TrustformersError.InvalidParameter or
                TrustformersError.FileNotFound or
                TrustformersError.InvalidFormat or
                TrustformersError.NetworkError or
                TrustformersError.NotSupported or
                TrustformersError.PermissionDenied or
                TrustformersError.AlreadyInitialized or
                TrustformersError.NotInitialized or
                TrustformersError.SerializationError or
                TrustformersError.HardwareNotAvailable or
                TrustformersError.ModelLoadError or
                TrustformersError.TokenizationError or
                TrustformersError.PipelineError or
                TrustformersError.ConfigError or
                TrustformersError.VersionMismatch or
                TrustformersError.LicenseError or
                TrustformersError.AuthenticationError => 2,
                
                TrustformersError.RuntimeError or
                TrustformersError.NullPointer or
                TrustformersError.OutOfMemory or
                TrustformersError.MemoryCorruption or
                TrustformersError.BufferOverflow or
                TrustformersError.UseAfterFree or
                TrustformersError.DoubleFree or
                TrustformersError.CudaError or
                TrustformersError.RocmError or
                TrustformersError.MetalError => 3,
                
                _ => 2
            };
        }
    }
}