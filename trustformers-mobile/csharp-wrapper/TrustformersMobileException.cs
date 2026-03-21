using System;
using TrustformersMobile.Native;

namespace TrustformersMobile
{
    /// <summary>
    /// Exception thrown by TrustformeRS Mobile operations
    /// </summary>
    public class TrustformersMobileException : Exception
    {
        /// <summary>
        /// The native error code that caused this exception
        /// </summary>
        public TrustformersMobileError ErrorCode { get; }

        /// <summary>
        /// Creates a new exception with the specified error code
        /// </summary>
        /// <param name="errorCode">Native error code</param>
        public TrustformersMobileException(TrustformersMobileError errorCode)
            : base(GetErrorMessage(errorCode))
        {
            ErrorCode = errorCode;
        }

        /// <summary>
        /// Creates a new exception with the specified error code and message
        /// </summary>
        /// <param name="errorCode">Native error code</param>
        /// <param name="message">Error message</param>
        public TrustformersMobileException(TrustformersMobileError errorCode, string message)
            : base(message)
        {
            ErrorCode = errorCode;
        }

        /// <summary>
        /// Creates a new exception with the specified error code, message, and inner exception
        /// </summary>
        /// <param name="errorCode">Native error code</param>
        /// <param name="message">Error message</param>
        /// <param name="innerException">Inner exception</param>
        public TrustformersMobileException(TrustformersMobileError errorCode, string message, Exception innerException)
            : base(message, innerException)
        {
            ErrorCode = errorCode;
        }

        private static string GetErrorMessage(TrustformersMobileError errorCode)
        {
            return errorCode switch
            {
                TrustformersMobileError.Success => "Operation completed successfully",
                TrustformersMobileError.InvalidParameter => "Invalid parameter provided",
                TrustformersMobileError.OutOfMemory => "Out of memory",
                TrustformersMobileError.ModelLoadError => "Failed to load model",
                TrustformersMobileError.InferenceError => "Inference operation failed",
                TrustformersMobileError.ConfigurationError => "Configuration error",
                TrustformersMobileError.PlatformNotSupported => "Platform not supported",
                TrustformersMobileError.RuntimeError => "Runtime error",
                TrustformersMobileError.NullPointer => "Null pointer error",
                TrustformersMobileError.SerializationError => "Serialization error",
                _ => $"Unknown error code: {errorCode}"
            };
        }
    }
}