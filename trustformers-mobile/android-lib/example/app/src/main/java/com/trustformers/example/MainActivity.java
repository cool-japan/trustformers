package com.trustformers.example;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.trustformers.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Example Android app demonstrating TrustformeRS library usage
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "TrustformersExample";
    private static final int PERMISSION_REQUEST_CODE = 100;
    
    private TrustformersEngine engine;
    private Model model;
    private ExecutorService executor;
    private Handler mainHandler;
    
    // UI elements
    private EditText inputText;
    private Button classifyButton;
    private Button deviceInfoButton;
    private TextView resultText;
    private TextView performanceText;
    private ProgressBar progressBar;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        // Initialize UI
        initializeUI();
        
        // Initialize executor and handler
        executor = Executors.newSingleThreadExecutor();
        mainHandler = new Handler(Looper.getMainLooper());
        
        // Check permissions
        checkPermissions();
        
        // Initialize TrustformersEngine
        initializeEngine();
    }
    
    private void initializeUI() {
        inputText = findViewById(R.id.inputText);
        classifyButton = findViewById(R.id.classifyButton);
        deviceInfoButton = findViewById(R.id.deviceInfoButton);
        resultText = findViewById(R.id.resultText);
        performanceText = findViewById(R.id.performanceText);
        progressBar = findViewById(R.id.progressBar);
        
        classifyButton.setOnClickListener(v -> classifyText());
        deviceInfoButton.setOnClickListener(v -> showDeviceInfo());
    }
    
    private void checkPermissions() {
        String[] permissions = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        };
        
        boolean allGranted = true;
        for (String permission : permissions) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                allGranted = false;
                break;
            }
        }
        
        if (!allGranted) {
            ActivityCompat.requestPermissions(this, permissions, PERMISSION_REQUEST_CODE);
        }
    }
    
    private void initializeEngine() {
        showProgress(true);
        
        executor.execute(() -> {
            try {
                // Create optimized configuration
                TrustformersEngine.EngineConfig config = TrustformersEngine.EngineConfig.createOptimized(this);
                config.setEnableProfiling(true);
                
                // Log configuration
                Log.i(TAG, "Engine configuration: " + config.toJson());
                
                // Create engine
                engine = new TrustformersEngine(this, config);
                
                // Load model from assets
                model = engine.loadModelFromAssets("models/text_classifier.tfm");
                
                runOnUiThread(() -> {
                    showProgress(false);
                    Toast.makeText(this, "Engine initialized successfully", Toast.LENGTH_SHORT).show();
                    classifyButton.setEnabled(true);
                });
                
            } catch (Exception e) {
                Log.e(TAG, "Failed to initialize engine", e);
                runOnUiThread(() -> {
                    showProgress(false);
                    Toast.makeText(this, "Failed to initialize: " + e.getMessage(), Toast.LENGTH_LONG).show();
                });
            }
        });
    }
    
    private void classifyText() {
        String text = inputText.getText().toString().trim();
        if (text.isEmpty()) {
            Toast.makeText(this, "Please enter some text", Toast.LENGTH_SHORT).show();
            return;
        }
        
        showProgress(true);
        classifyButton.setEnabled(false);
        
        executor.execute(() -> {
            try {
                long startTime = System.currentTimeMillis();
                
                // Tokenize text (simplified - in practice use real tokenizer)
                float[] tokens = tokenizeText(text);
                Tensor input = new Tensor(tokens, new int[]{1, tokens.length});
                
                // Perform inference
                Tensor output = engine.inference(model, input);
                
                // Apply softmax
                Tensor probabilities = output.softmax();
                
                // Get top predictions
                Tensor.TopKResult topK = probabilities.topK(3);
                
                long inferenceTime = System.currentTimeMillis() - startTime;
                
                // Prepare results
                String[] labels = {"Positive", "Negative", "Neutral"};
                StringBuilder results = new StringBuilder("Classification Results:\n\n");
                
                for (int i = 0; i < topK.getIndices().length; i++) {
                    int index = topK.getIndices()[i];
                    float score = topK.getValues()[i];
                    results.append(String.format(Locale.US, "%s: %.2f%%\n", 
                        labels[index], score * 100));
                }
                
                // Get performance stats
                String perfStats = "";
                if (engine.getPerformanceStats() != null) {
                    perfStats = engine.getPerformanceStats().getSummary();
                }
                
                String finalResults = results.toString();
                String finalPerfStats = perfStats;
                
                runOnUiThread(() -> {
                    showProgress(false);
                    classifyButton.setEnabled(true);
                    resultText.setText(finalResults);
                    performanceText.setText(String.format(Locale.US,
                        "Inference time: %d ms\n\n%s", inferenceTime, finalPerfStats));
                });
                
            } catch (Exception e) {
                Log.e(TAG, "Classification failed", e);
                runOnUiThread(() -> {
                    showProgress(false);
                    classifyButton.setEnabled(true);
                    Toast.makeText(this, "Classification failed: " + e.getMessage(), 
                        Toast.LENGTH_LONG).show();
                });
            }
        });
    }
    
    private void showDeviceInfo() {
        DeviceInfo deviceInfo = engine.getDeviceInfo();
        
        StringBuilder info = new StringBuilder();
        info.append("Device Information:\n\n");
        info.append(deviceInfo.getDeviceSummary());
        info.append("\n");
        info.append("Performance Class: ").append(deviceInfo.getPerformanceClass()).append("\n");
        info.append("Available Memory: ").append(deviceInfo.getAvailableMemoryMB()).append(" MB\n");
        info.append("Battery Level: ").append(deviceInfo.getBatteryLevel()).append("%\n");
        info.append("Charging: ").append(deviceInfo.isCharging() ? "Yes" : "No").append("\n");
        
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
            info.append("Thermal Status: ").append(deviceInfo.getThermalStatus()).append("\n");
        }
        
        info.append("\nRecommended Threads: ").append(deviceInfo.getRecommendedThreadCount()).append("\n");
        
        resultText.setText(info.toString());
    }
    
    private float[] tokenizeText(String text) {
        // Simplified tokenization - in practice, use proper tokenizer
        String[] words = text.toLowerCase().split("\\s+");
        float[] tokens = new float[Math.min(words.length * 10, 512)];
        
        // Generate mock embeddings
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = (float) (Math.random() * 2 - 1);
        }
        
        return tokens;
    }
    
    private void showProgress(boolean show) {
        progressBar.setVisibility(show ? View.VISIBLE : View.GONE);
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        
        // Clean up resources
        if (engine != null) {
            engine.close();
        }
        
        if (executor != null) {
            executor.shutdown();
        }
    }
}