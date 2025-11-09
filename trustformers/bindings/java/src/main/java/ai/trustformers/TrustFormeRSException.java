package ai.trustformers;

/**
 * Exception class for TrustFormeRS library errors.
 * 
 * This exception is thrown when operations in the TrustFormeRS library fail,
 * such as model loading, tokenization, or inference errors.
 */
public class TrustFormeRSException extends Exception {
    
    private static final long serialVersionUID = 1L;
    
    private final String errorCode;
    private final String suggestion;
    
    /**
     * Constructs a new TrustFormeRSException with the specified detail message.
     * 
     * @param message the detail message
     */
    public TrustFormeRSException(String message) {
        super(message);
        this.errorCode = null;
        this.suggestion = null;
    }
    
    /**
     * Constructs a new TrustFormeRSException with the specified detail message and cause.
     * 
     * @param message the detail message
     * @param cause the cause of the exception
     */
    public TrustFormeRSException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = null;
        this.suggestion = null;
    }
    
    /**
     * Constructs a new TrustFormeRSException with the specified detail message, error code, and suggestion.
     * 
     * @param message the detail message
     * @param errorCode the error code from the native library
     * @param suggestion helpful suggestion for resolving the error
     */
    public TrustFormeRSException(String message, String errorCode, String suggestion) {
        super(message);
        this.errorCode = errorCode;
        this.suggestion = suggestion;
    }
    
    /**
     * Constructs a new TrustFormeRSException with the specified detail message, error code, suggestion, and cause.
     * 
     * @param message the detail message
     * @param errorCode the error code from the native library
     * @param suggestion helpful suggestion for resolving the error
     * @param cause the cause of the exception
     */
    public TrustFormeRSException(String message, String errorCode, String suggestion, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
        this.suggestion = suggestion;
    }
    
    /**
     * Gets the error code associated with this exception.
     * 
     * @return the error code, or null if no error code was provided
     */
    public String getErrorCode() {
        return errorCode;
    }
    
    /**
     * Gets the suggestion for resolving this error.
     * 
     * @return the suggestion, or null if no suggestion was provided
     */
    public String getSuggestion() {
        return suggestion;
    }
    
    /**
     * Returns whether this exception has an error code.
     * 
     * @return true if an error code is available
     */
    public boolean hasErrorCode() {
        return errorCode != null && !errorCode.isEmpty();
    }
    
    /**
     * Returns whether this exception has a suggestion.
     * 
     * @return true if a suggestion is available
     */
    public boolean hasSuggestion() {
        return suggestion != null && !suggestion.isEmpty();
    }
    
    /**
     * Returns a detailed string representation of this exception,
     * including error code and suggestion if available.
     * 
     * @return detailed exception information
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(getClass().getSimpleName());
        
        if (getMessage() != null) {
            sb.append(": ").append(getMessage());
        }
        
        if (hasErrorCode()) {
            sb.append(" [Error Code: ").append(errorCode).append("]");
        }
        
        if (hasSuggestion()) {
            sb.append(" [Suggestion: ").append(suggestion).append("]");
        }
        
        return sb.toString();
    }
    
    /**
     * Returns a user-friendly error message that includes the suggestion if available.
     * 
     * @return user-friendly error message
     */
    public String getUserFriendlyMessage() {
        StringBuilder sb = new StringBuilder();
        
        if (getMessage() != null) {
            sb.append(getMessage());
        }
        
        if (hasSuggestion()) {
            if (sb.length() > 0) {
                sb.append("\n\n");
            }
            sb.append("💡 Suggestion: ").append(suggestion);
        }
        
        return sb.toString();
    }
}