package com.trustformers;

/**
 * Exception thrown by TrustformeRS operations.
 * Maps to the TrustformersError enum from the C API.
 */
public class TrustformersException extends Exception {
    
    public enum ErrorCode {
        SUCCESS(0),
        NULL_POINTER(1),
        INVALID_PARAMETER(2),
        RUNTIME_ERROR(3),
        SERIALIZATION_ERROR(4),
        MEMORY_ERROR(5),
        IO_ERROR(6),
        NOT_IMPLEMENTED(7),
        UNKNOWN_ERROR(8);
        
        private final int code;
        
        ErrorCode(int code) {
            this.code = code;
        }
        
        public int getCode() {
            return code;
        }
        
        public static ErrorCode fromCode(int code) {
            for (ErrorCode errorCode : values()) {
                if (errorCode.code == code) {
                    return errorCode;
                }
            }
            return UNKNOWN_ERROR;
        }
    }
    
    private final ErrorCode errorCode;
    
    public TrustformersException(ErrorCode errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }
    
    public TrustformersException(ErrorCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }
    
    public TrustformersException(int errorCode, String message) {
        this(ErrorCode.fromCode(errorCode), message);
    }
    
    public ErrorCode getErrorCode() {
        return errorCode;
    }
    
    @Override
    public String toString() {
        return String.format("TrustformersException[%s]: %s", errorCode, getMessage());
    }
}