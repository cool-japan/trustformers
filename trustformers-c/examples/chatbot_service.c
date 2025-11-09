/*
 * Real-world Chatbot Service Example for TrustformeRS C API
 * 
 * This example demonstrates how to build a production-ready chatbot service using:
 * - Conversational AI pipeline
 * - Session management and conversation history
 * - HTTP server integration for web API
 * - Performance monitoring and metrics
 * - Error handling and recovery
 * - Memory management for long-running service
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include "trustformers.h"

// Global state for signal handling
static volatile int running = 1;
static char* server_id = NULL;
static TrustformersPipeline chatbot_pipeline = NULL;

// Configuration
typedef struct {
    const char* model_name;
    const char* host;
    int port;
    int max_sessions;
    int max_history_length;
    double temperature;
    int max_tokens;
    int timeout_seconds;
} ChatbotConfig;

// Session management
typedef struct ChatSession {
    char session_id[64];
    char user_name[128];
    time_t created_at;
    time_t last_activity;
    char** conversation_history;
    int history_count;
    int max_history;
    struct ChatSession* next;
} ChatSession;

static ChatSession* active_sessions = NULL;
static int session_count = 0;

void print_error(TrustformersError error) {
    if (error != TrustformersError_Success) {
        const char* error_msg = trustformers_error_message(error);
        fprintf(stderr, "[ERROR] %s\n", error_msg);
    }
}

void signal_handler(int sig) {
    printf("\n[INFO] Received signal %d, shutting down gracefully...\n", sig);
    running = 0;
}

// Generate unique session ID
void generate_session_id(char* session_id, size_t size) {
    srand(time(NULL));
    snprintf(session_id, size, "chat_%ld_%d", time(NULL), rand() % 10000);
}

// Create new chat session
ChatSession* create_session(const char* user_name, int max_history) {
    ChatSession* session = malloc(sizeof(ChatSession));
    if (!session) return NULL;
    
    generate_session_id(session->session_id, sizeof(session->session_id));
    strncpy(session->user_name, user_name ? user_name : "Anonymous", sizeof(session->user_name) - 1);
    session->user_name[sizeof(session->user_name) - 1] = '\0';
    
    session->created_at = time(NULL);
    session->last_activity = time(NULL);
    session->max_history = max_history;
    session->history_count = 0;
    session->conversation_history = malloc(max_history * sizeof(char*));
    session->next = NULL;
    
    if (!session->conversation_history) {
        free(session);
        return NULL;
    }
    
    // Add to active sessions list
    session->next = active_sessions;
    active_sessions = session;
    session_count++;
    
    printf("[INFO] Created new session: %s for user: %s\n", 
           session->session_id, session->user_name);
    
    return session;
}

// Find session by ID
ChatSession* find_session(const char* session_id) {
    ChatSession* current = active_sessions;
    while (current) {
        if (strcmp(current->session_id, session_id) == 0) {
            current->last_activity = time(NULL);
            return current;
        }
        current = current->next;
    }
    return NULL;
}

// Add message to conversation history
void add_to_history(ChatSession* session, const char* role, const char* message) {
    if (session->history_count >= session->max_history) {
        // Remove oldest message to make room
        free(session->conversation_history[0]);
        for (int i = 1; i < session->history_count; i++) {
            session->conversation_history[i-1] = session->conversation_history[i];
        }
        session->history_count--;
    }
    
    // Format message with role
    size_t msg_len = strlen(role) + strlen(message) + 10;
    char* formatted_msg = malloc(msg_len);
    snprintf(formatted_msg, msg_len, "%s: %s", role, message);
    
    session->conversation_history[session->history_count] = formatted_msg;
    session->history_count++;
}

// Build conversation context
char* build_conversation_context(ChatSession* session) {
    if (session->history_count == 0) {
        return strdup("You are a helpful AI assistant. Please respond naturally and helpfully to the user's questions.");
    }
    
    // Calculate total length needed
    size_t total_len = 200; // Base context
    for (int i = 0; i < session->history_count; i++) {
        total_len += strlen(session->conversation_history[i]) + 2; // +2 for \n
    }
    
    char* context = malloc(total_len);
    strcpy(context, "You are a helpful AI assistant. Here's our conversation so far:\n\n");
    
    for (int i = 0; i < session->history_count; i++) {
        strcat(context, session->conversation_history[i]);
        strcat(context, "\n");
    }
    
    strcat(context, "\nAssistant:");
    return context;
}

// Generate chatbot response
char* generate_response(ChatSession* session, const char* user_message) {
    TrustformersError error;
    
    // Add user message to history
    add_to_history(session, "User", user_message);
    
    // Build conversation context
    char* context = build_conversation_context(session);
    
    // Configure generation parameters
    const char* generation_config = "{"
        "\"max_length\": 200,"
        "\"temperature\": 0.7,"
        "\"top_k\": 50,"
        "\"top_p\": 0.9,"
        "\"do_sample\": true,"
        "\"no_repeat_ngram_size\": 3"
    "}";
    
    // Generate response
    char* generated_text = NULL;
    error = trustformers_pipeline_generate_text_with_options(
        chatbot_pipeline, context, generation_config, &generated_text
    );
    
    free(context);
    
    if (error != TrustformersError_Success) {
        print_error(error);
        return strdup("I'm sorry, I'm having trouble responding right now. Please try again.");
    }
    
    if (generated_text) {
        // Add response to history
        add_to_history(session, "Assistant", generated_text);
        
        // Create a copy to return (original will be freed)
        char* response = strdup(generated_text);
        trustformers_free_string(generated_text);
        return response;
    }
    
    return strdup("I'm sorry, I couldn't generate a response.");
}

// Clean up expired sessions
void cleanup_expired_sessions(int max_idle_minutes) {
    time_t now = time(NULL);
    ChatSession* current = active_sessions;
    ChatSession* prev = NULL;
    
    while (current) {
        double idle_minutes = difftime(now, current->last_activity) / 60.0;
        
        if (idle_minutes > max_idle_minutes) {
            printf("[INFO] Cleaning up expired session: %s (idle for %.1f minutes)\n", 
                   current->session_id, idle_minutes);
            
            // Remove from list
            if (prev) {
                prev->next = current->next;
            } else {
                active_sessions = current->next;
            }
            
            // Free session data
            for (int i = 0; i < current->history_count; i++) {
                free(current->conversation_history[i]);
            }
            free(current->conversation_history);
            
            ChatSession* to_delete = current;
            current = current->next;
            free(to_delete);
            session_count--;
        } else {
            prev = current;
            current = current->next;
        }
    }
}

// HTTP endpoint handler simulation
void handle_chat_request(const char* request_json) {
    printf("[INFO] Processing chat request: %s\n", request_json);
    
    // In a real implementation, you would parse the JSON request
    // For this demo, we'll simulate the request processing
    
    // Extract session_id, user_message, and user_name from JSON
    // This is a simplified version - use a proper JSON parser in production
    char session_id[64] = {0};
    char user_message[512] = {0};
    char user_name[128] = "DemoUser";
    
    // Simulate JSON parsing
    sscanf(request_json, "{\"session_id\":\"%63[^\"]\",\"message\":\"%511[^\"]\"}",
           session_id, user_message);
    
    if (strlen(session_id) == 0 || strlen(user_message) == 0) {
        printf("[ERROR] Invalid request format\n");
        return;
    }
    
    // Find or create session
    ChatSession* session = find_session(session_id);
    if (!session) {
        session = create_session(user_name, 20); // Max 20 messages in history
        if (!session) {
            printf("[ERROR] Failed to create session\n");
            return;
        }
        strcpy(session->session_id, session_id); // Use provided session ID
    }
    
    // Generate response
    char* response = generate_response(session, user_message);
    
    // Format response JSON (simplified)
    printf("[RESPONSE] {"
           "\"session_id\":\"%s\","
           "\"response\":\"%s\","
           "\"timestamp\":%ld"
           "}\n", 
           session->session_id, response, time(NULL));
    
    free(response);
}

// Initialize chatbot pipeline
int initialize_chatbot(const ChatbotConfig* config) {
    TrustformersError error;
    
    printf("[INFO] Initializing chatbot with model: %s\n", config->model_name);
    
    // Load model and tokenizer
    void* model = trustformers_load_model_from_hub(config->model_name, &error);
    if (error != TrustformersError_Success) {
        printf("[ERROR] Failed to load model: %s\n", config->model_name);
        print_error(error);
        return 0;
    }
    
    void* tokenizer = trustformers_load_tokenizer_from_hub(config->model_name, &error);
    if (error != TrustformersError_Success) {
        printf("[ERROR] Failed to load tokenizer: %s\n", config->model_name);
        trustformers_model_free(model);
        print_error(error);
        return 0;
    }
    
    // Create conversational pipeline
    chatbot_pipeline = trustformers_create_conversational_pipeline(model, tokenizer, &error);
    if (error != TrustformersError_Success) {
        printf("[ERROR] Failed to create conversational pipeline\n");
        trustformers_tokenizer_free(tokenizer);
        trustformers_model_free(model);
        print_error(error);
        return 0;
    }
    
    printf("[INFO] Chatbot initialized successfully\n");
    return 1;
}

// Initialize HTTP server
int initialize_server(const ChatbotConfig* config) {
    TrustformersError error;
    
    // Create server configuration
    char server_config[512];
    snprintf(server_config, sizeof(server_config),
        "{"
        "\"host\":\"%s\","
        "\"port\":%d,"
        "\"max_connections\":100,"
        "\"enable_cors\":true,"
        "\"timeout_seconds\":%d"
        "}",
        config->host, config->port, config->timeout_seconds
    );
    
    // Create HTTP server
    error = trustformers_http_server_create_with_config(server_config, &server_id);
    if (error != TrustformersError_Success) {
        printf("[ERROR] Failed to create HTTP server\n");
        print_error(error);
        return 0;
    }
    
    // Add chat endpoint
    const char* endpoint_config = "{"
        "\"name\":\"chatbot\","
        "\"endpoint_path\":\"/chat\","
        "\"method\":\"POST\","
        "\"content_type\":\"application/json\","
        "\"max_request_size\":4096"
    "}";
    
    error = trustformers_http_server_add_model(server_id, endpoint_config);
    if (error != TrustformersError_Success) {
        printf("[ERROR] Failed to add chat endpoint\n");
        trustformers_http_server_destroy(server_id);
        trustformers_free_string(server_id);
        return 0;
    }
    
    // Start server
    error = trustformers_http_server_start(server_id);
    if (error != TrustformersError_Success) {
        printf("[ERROR] Failed to start HTTP server\n");
        trustformers_http_server_destroy(server_id);
        trustformers_free_string(server_id);
        return 0;
    }
    
    printf("[INFO] HTTP server started on %s:%d\n", config->host, config->port);
    printf("[INFO] Chat endpoint available at: http://%s:%d/chat\n", config->host, config->port);
    return 1;
}

// Print service statistics
void print_statistics() {
    TrustformersMemoryUsage memory_usage;
    TrustformersError error = trustformers_get_memory_usage(&memory_usage);
    
    printf("\n=== Chatbot Service Statistics ===\n");
    printf("Active sessions: %d\n", session_count);
    
    if (error == TrustformersError_Success) {
        printf("Memory usage: %.2f MB\n", memory_usage.total_memory_bytes / (1024.0 * 1024.0));
        printf("Allocated objects: %llu models, %llu tokenizers, %llu pipelines\n",
               memory_usage.allocated_models, memory_usage.allocated_tokenizers,
               memory_usage.allocated_pipelines);
    }
    
    // List active sessions
    if (session_count > 0) {
        printf("\nActive sessions:\n");
        ChatSession* current = active_sessions;
        int index = 1;
        while (current && index <= 5) { // Show max 5 sessions
            double idle_minutes = difftime(time(NULL), current->last_activity) / 60.0;
            printf("  %d. %s (%s) - %d messages, idle %.1f min\n",
                   index, current->session_id, current->user_name,
                   current->history_count, idle_minutes);
            current = current->next;
            index++;
        }
        if (session_count > 5) {
            printf("  ... and %d more sessions\n", session_count - 5);
        }
    }
    printf("==================================\n\n");
}

// Demo conversation simulation
void run_demo_conversations() {
    printf("[INFO] Running demo conversations...\n");
    
    // Simulate several chat requests
    const char* demo_requests[] = {
        "{\"session_id\":\"demo_001\",\"message\":\"Hello! How are you today?\"}",
        "{\"session_id\":\"demo_001\",\"message\":\"Can you help me with a coding problem?\"}",
        "{\"session_id\":\"demo_002\",\"message\":\"What's the weather like?\"}",
        "{\"session_id\":\"demo_001\",\"message\":\"I'm working on a Python script that reads CSV files.\"}",
        "{\"session_id\":\"demo_003\",\"message\":\"Tell me a joke!\"}",
        "{\"session_id\":\"demo_002\",\"message\":\"Actually, can you explain machine learning?\"}",
    };
    
    int num_requests = sizeof(demo_requests) / sizeof(demo_requests[0]);
    
    for (int i = 0; i < num_requests; i++) {
        printf("\n[DEMO] Request %d:\n", i + 1);
        handle_chat_request(demo_requests[i]);
        
        // Small delay between requests
        usleep(500000); // 0.5 seconds
    }
    
    printf("\n[INFO] Demo conversations completed\n");
}

int main() {
    printf("TrustformeRS Chatbot Service Example\n");
    printf("===================================\n\n");
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Initialize TrustformeRS
    TrustformersError error = trustformers_init();
    if (error != TrustformersError_Success) {
        print_error(error);
        return 1;
    }
    
    printf("[INFO] TrustformeRS initialized successfully\n");
    
    // Configuration
    ChatbotConfig config = {
        .model_name = "microsoft/DialoGPT-medium",
        .host = "127.0.0.1",
        .port = 8080,
        .max_sessions = 100,
        .max_history_length = 20,
        .temperature = 0.7,
        .max_tokens = 200,
        .timeout_seconds = 30
    };
    
    // Initialize chatbot
    if (!initialize_chatbot(&config)) {
        printf("[ERROR] Failed to initialize chatbot\n");
        trustformers_cleanup();
        return 1;
    }
    
    // Initialize HTTP server
    if (!initialize_server(&config)) {
        printf("[ERROR] Failed to initialize server\n");
        if (chatbot_pipeline) {
            trustformers_pipeline_free(chatbot_pipeline);
        }
        trustformers_cleanup();
        return 1;
    }
    
    printf("\n[INFO] Chatbot service is running!\n");
    printf("[INFO] Try sending POST requests to http://%s:%d/chat\n", config.host, config.port);
    printf("[INFO] Example request: curl -X POST -H \"Content-Type: application/json\" \\\n");
    printf("       -d '{\"session_id\":\"test123\",\"message\":\"Hello!\"}' \\\n");
    printf("       http://%s:%d/chat\n", config.host, config.port);
    printf("[INFO] Press Ctrl+C to stop the service\n\n");
    
    // Run demo conversations if no real traffic
    run_demo_conversations();
    
    // Main service loop
    time_t last_cleanup = time(NULL);
    time_t last_stats = time(NULL);
    
    while (running) {
        sleep(1);
        
        time_t now = time(NULL);
        
        // Cleanup expired sessions every 5 minutes
        if (difftime(now, last_cleanup) >= 300) {
            cleanup_expired_sessions(30); // Clean sessions idle for 30+ minutes
            last_cleanup = now;
        }
        
        // Print statistics every 60 seconds
        if (difftime(now, last_stats) >= 60) {
            print_statistics();
            last_stats = now;
        }
    }
    
    // Cleanup
    printf("\n[INFO] Shutting down chatbot service...\n");
    
    if (server_id) {
        trustformers_http_server_stop(server_id);
        trustformers_http_server_destroy(server_id);
        trustformers_free_string(server_id);
    }
    
    if (chatbot_pipeline) {
        trustformers_pipeline_free(chatbot_pipeline);
    }
    
    // Clean up all sessions
    ChatSession* current = active_sessions;
    while (current) {
        ChatSession* next = current->next;
        for (int i = 0; i < current->history_count; i++) {
            free(current->conversation_history[i]);
        }
        free(current->conversation_history);
        free(current);
        current = next;
    }
    
    trustformers_cleanup();
    printf("[INFO] Chatbot service shutdown complete\n");
    
    return 0;
}