package com.trustformers.example

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.trustformers.TrustformersEngine
import kotlinx.coroutines.launch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Camera Translation Activity
 * 
 * This activity demonstrates real-time text recognition from camera feed
 * and translation using TrustformeRS engine.
 */
class CameraTranslationActivity : AppCompatActivity() {
    
    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
    
    private lateinit var previewView: PreviewView
    private lateinit var tvRecognizedText: TextView
    private lateinit var tvTranslatedText: TextView
    private lateinit var tvLanguageInfo: TextView
    
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var trustformersEngine: TrustformersEngine
    
    private var fromLanguage: String = "en"
    private var toLanguage: String = "es"
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera_translation)
        
        // Get language parameters
        fromLanguage = intent.getStringExtra("fromLanguage") ?: "en"
        toLanguage = intent.getStringExtra("toLanguage") ?: "es"
        
        initializeUI()
        setupTrustformersEngine()
        
        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
        
        cameraExecutor = Executors.newSingleThreadExecutor()
    }
    
    private fun initializeUI() {
        previewView = findViewById(R.id.previewView)
        tvRecognizedText = findViewById(R.id.tvRecognizedText)
        tvTranslatedText = findViewById(R.id.tvTranslatedText)
        tvLanguageInfo = findViewById(R.id.tvLanguageInfo)
        
        tvLanguageInfo.text = "Translating from $fromLanguage to $toLanguage"
    }
    
    private fun setupTrustformersEngine() {
        lifecycleScope.launch {
            try {
                // Initialize TrustformersEngine with camera-optimized settings
                val config = com.trustformers.TrustformersConfig.Builder()
                    .setBackend(com.trustformers.MobileBackend.GPU)
                    .setQuantization(com.trustformers.MobileQuantization.FP16)
                    .setMaxBatchSize(1)
                    .setEnableOptimization(true)
                    .setNumThreads(2)
                    .build()
                
                trustformersEngine = com.trustformers.TrustformersEngine.Builder(this@CameraTranslationActivity)
                    .setConfig(config)
                    .build()
                
                // Load translation model
                trustformersEngine.loadModel("translation_model.onnx")
                
            } catch (e: Exception) {
                runOnUiThread {
                    Toast.makeText(this@CameraTranslationActivity, "Failed to initialize translation engine", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
            
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, TextAnalyzer { recognizedText ->
                        runOnUiThread {
                            tvRecognizedText.text = recognizedText
                            
                            // Translate recognized text
                            if (recognizedText.isNotEmpty()) {
                                translateText(recognizedText)
                            }
                        }
                    })
                }
            
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Toast.makeText(this, "Failed to bind camera use cases", Toast.LENGTH_SHORT).show()
            }
            
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun translateText(text: String) {
        lifecycleScope.launch {
            try {
                // Simplified translation using TrustformersEngine
                val translatedText = performTranslation(text)
                runOnUiThread {
                    tvTranslatedText.text = translatedText
                }
            } catch (e: Exception) {
                runOnUiThread {
                    tvTranslatedText.text = "Translation error: ${e.message}"
                }
            }
        }
    }
    
    private suspend fun performTranslation(text: String): String {
        return kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Default) {
            try {
                // Tokenize input text
                val inputTokens = tokenizeText(text)
                
                // Run inference
                val outputs = trustformersEngine.inference(inputTokens)
                
                // Decode output
                decodeTranslation(outputs)
            } catch (e: Exception) {
                "Translation failed: ${e.message}"
            }
        }
    }
    
    private fun tokenizeText(text: String): FloatArray {
        // Simplified tokenization for demonstration
        val tokens = text.lowercase().split(" ")
        val tokenIds = tokens.map { it.hashCode() % 1000 }
        return tokenIds.map { it.toFloat() }.toFloatArray()
    }
    
    private fun decodeTranslation(outputs: FloatArray): String {
        // Simplified decoding for demonstration
        val words = listOf("hello", "world", "translation", "camera", "text", "mobile", "app")
        val selectedWords = outputs.take(5).map { 
            val index = (it.toInt().absoluteValue) % words.size
            words[index]
        }
        return selectedWords.joinToString(" ")
    }
    
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
    
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
    
    private class TextAnalyzer(private val onTextRecognized: (String) -> Unit) : ImageAnalysis.Analyzer {
        
        private val textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
        
        override fun analyze(imageProxy: ImageProxy) {
            val mediaImage = imageProxy.image
            if (mediaImage != null) {
                val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
                
                textRecognizer.process(image)
                    .addOnSuccessListener { visionText ->
                        val recognizedText = visionText.text
                        onTextRecognized(recognizedText)
                    }
                    .addOnFailureListener { e ->
                        // Handle text recognition failure
                        onTextRecognized("Text recognition failed: ${e.message}")
                    }
                    .addOnCompleteListener {
                        imageProxy.close()
                    }
            } else {
                imageProxy.close()
            }
        }
    }
}