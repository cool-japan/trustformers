package com.trustformers.example

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.trustformers.TrustformersEngine
import com.trustformers.TrustformersConfig
import com.trustformers.MobileBackend
import com.trustformers.MobileQuantization
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.*
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Real-time Translation Android Demo
 * 
 * This activity demonstrates real-time translation capabilities using TrustformeRS
 * with support for text, voice, and camera-based translation.
 */
class RealTimeTranslationActivity : AppCompatActivity(), TextToSpeech.OnInitListener {
    
    companion object {
        private const val REQUEST_RECORD_AUDIO_PERMISSION = 200
        private const val REQUEST_CAMERA_PERMISSION = 201
        private const val SPEECH_REQUEST_CODE = 100
        
        // Supported languages for translation
        private val SUPPORTED_LANGUAGES = mapOf(
            "en" to "English",
            "es" to "Spanish",
            "fr" to "French",
            "de" to "German",
            "it" to "Italian",
            "pt" to "Portuguese",
            "ru" to "Russian",
            "ja" to "Japanese",
            "ko" to "Korean",
            "zh" to "Chinese",
            "ar" to "Arabic",
            "hi" to "Hindi"
        )
    }
    
    // UI Components
    private lateinit var etInputText: EditText
    private lateinit var tvTranslatedText: TextView
    private lateinit var spinnerFromLanguage: Spinner
    private lateinit var spinnerToLanguage: Spinner
    private lateinit var btnVoiceInput: Button
    private lateinit var btnCameraTranslate: Button
    private lateinit var btnPlayTranslation: Button
    private lateinit var btnSwapLanguages: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var tvTranslationInfo: TextView
    private lateinit var switchRealTimeMode: Switch
    private lateinit var tvConfidenceScore: TextView
    private lateinit var btnClearText: Button
    private lateinit var layoutTranslationHistory: LinearLayout
    private lateinit var scrollViewHistory: ScrollView
    
    // TrustformeRS Engine
    private lateinit var trustformersEngine: TrustformersEngine
    private lateinit var translationModel: TranslationModel
    
    // Speech Recognition and TTS
    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var textToSpeech: TextToSpeech
    private var isTTSReady = false
    
    // Audio Recording
    private var audioRecord: AudioRecord? = null
    private var isRecording = AtomicBoolean(false)
    private var audioBufferSize = 0
    
    // Translation History
    private val translationHistory = mutableListOf<TranslationEntry>()
    
    // Real-time translation
    private var isRealTimeEnabled = false
    private var lastTranslationTime = 0L
    private val translationThrottleMs = 1000 // 1 second throttle
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_real_time_translation)
        
        initializeUI()
        setupLanguageSpinners()
        setupTrustformersEngine()
        setupSpeechRecognition()
        setupTextToSpeech()
        setupEventListeners()
        requestPermissions()
    }
    
    private fun initializeUI() {
        etInputText = findViewById(R.id.etInputText)
        tvTranslatedText = findViewById(R.id.tvTranslatedText)
        spinnerFromLanguage = findViewById(R.id.spinnerFromLanguage)
        spinnerToLanguage = findViewById(R.id.spinnerToLanguage)
        btnVoiceInput = findViewById(R.id.btnVoiceInput)
        btnCameraTranslate = findViewById(R.id.btnCameraTranslate)
        btnPlayTranslation = findViewById(R.id.btnPlayTranslation)
        btnSwapLanguages = findViewById(R.id.btnSwapLanguages)
        progressBar = findViewById(R.id.progressBar)
        tvTranslationInfo = findViewById(R.id.tvTranslationInfo)
        switchRealTimeMode = findViewById(R.id.switchRealTimeMode)
        tvConfidenceScore = findViewById(R.id.tvConfidenceScore)
        btnClearText = findViewById(R.id.btnClearText)
        layoutTranslationHistory = findViewById(R.id.layoutTranslationHistory)
        scrollViewHistory = findViewById(R.id.scrollViewHistory)
    }
    
    private fun setupLanguageSpinners() {
        val languages = SUPPORTED_LANGUAGES.values.toList()
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, languages)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        
        spinnerFromLanguage.adapter = adapter
        spinnerToLanguage.adapter = adapter
        
        // Set default languages
        spinnerFromLanguage.setSelection(0) // English
        spinnerToLanguage.setSelection(1) // Spanish
    }
    
    private fun setupTrustformersEngine() {
        lifecycleScope.launch {
            try {
                val config = TrustformersConfig.Builder()
                    .setBackend(MobileBackend.NNAPI)
                    .setQuantization(MobileQuantization.INT8)
                    .setMaxBatchSize(1)
                    .setEnableOptimization(true)
                    .setUseFp16(true)
                    .setNumThreads(4)
                    .build()
                
                trustformersEngine = TrustformersEngine.Builder(this@RealTimeTranslationActivity)
                    .setConfig(config)
                    .build()
                
                // Load translation model
                translationModel = TranslationModel(trustformersEngine)
                translationModel.loadModel("translation_model.onnx")
                
                runOnUiThread {
                    updateTranslationInfo("Translation engine ready")
                }
            } catch (e: Exception) {
                runOnUiThread {
                    updateTranslationInfo("Error initializing engine: ${e.message}")
                }
            }
        }
    }
    
    private fun setupSpeechRecognition() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizer.setRecognitionListener(object : android.speech.RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                runOnUiThread {
                    btnVoiceInput.text = "Listening..."
                    btnVoiceInput.isEnabled = false
                }
            }
            
            override fun onBeginningOfSpeech() {}
            
            override fun onRmsChanged(rmsdB: Float) {}
            
            override fun onBufferReceived(buffer: ByteArray?) {}
            
            override fun onEndOfSpeech() {}
            
            override fun onError(error: Int) {
                runOnUiThread {
                    btnVoiceInput.text = "Voice Input"
                    btnVoiceInput.isEnabled = true
                    updateTranslationInfo("Speech recognition error: $error")
                }
            }
            
            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) {
                    val recognizedText = matches[0]
                    runOnUiThread {
                        etInputText.setText(recognizedText)
                        btnVoiceInput.text = "Voice Input"
                        btnVoiceInput.isEnabled = true
                        
                        // Auto-translate if real-time mode is enabled
                        if (isRealTimeEnabled) {
                            performTranslation(recognizedText)
                        }
                    }
                }
            }
            
            override fun onPartialResults(partialResults: Bundle?) {
                if (isRealTimeEnabled) {
                    val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    if (!matches.isNullOrEmpty()) {
                        val partialText = matches[0]
                        runOnUiThread {
                            etInputText.setText(partialText)
                        }
                        
                        // Throttled real-time translation
                        val currentTime = System.currentTimeMillis()
                        if (currentTime - lastTranslationTime > translationThrottleMs) {
                            performTranslation(partialText)
                            lastTranslationTime = currentTime
                        }
                    }
                }
            }
            
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
    }
    
    private fun setupTextToSpeech() {
        textToSpeech = TextToSpeech(this, this)
    }
    
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            isTTSReady = true
            updateTranslationInfo("Text-to-Speech ready")
        } else {
            updateTranslationInfo("Text-to-Speech initialization failed")
        }
    }
    
    private fun setupEventListeners() {
        // Real-time text translation
        etInputText.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            
            override fun afterTextChanged(s: Editable?) {
                if (isRealTimeEnabled && !s.isNullOrEmpty()) {
                    val currentTime = System.currentTimeMillis()
                    if (currentTime - lastTranslationTime > translationThrottleMs) {
                        performTranslation(s.toString())
                        lastTranslationTime = currentTime
                    }
                }
            }
        })
        
        // Voice input button
        btnVoiceInput.setOnClickListener {
            if (hasAudioPermission()) {
                startVoiceRecognition()
            } else {
                requestAudioPermission()
            }
        }
        
        // Camera translation button
        btnCameraTranslate.setOnClickListener {
            if (hasCameraPermission()) {
                startCameraTranslation()
            } else {
                requestCameraPermission()
            }
        }
        
        // Play translation button
        btnPlayTranslation.setOnClickListener {
            playTranslation()
        }
        
        // Swap languages button
        btnSwapLanguages.setOnClickListener {
            swapLanguages()
        }
        
        // Clear text button
        btnClearText.setOnClickListener {
            etInputText.text.clear()
            tvTranslatedText.text = ""
            tvConfidenceScore.text = ""
        }
        
        // Real-time mode switch
        switchRealTimeMode.setOnCheckedChangeListener { _, isChecked ->
            isRealTimeEnabled = isChecked
            if (isChecked) {
                updateTranslationInfo("Real-time mode enabled")
                // Translate current text if available
                val currentText = etInputText.text.toString()
                if (currentText.isNotEmpty()) {
                    performTranslation(currentText)
                }
            } else {
                updateTranslationInfo("Real-time mode disabled")
            }
        }
    }
    
    private fun performTranslation(text: String) {
        if (text.isEmpty()) return
        
        val fromLanguage = getSelectedLanguageCode(spinnerFromLanguage)
        val toLanguage = getSelectedLanguageCode(spinnerToLanguage)
        
        lifecycleScope.launch {
            try {
                showProgress(true)
                updateTranslationInfo("Translating...")
                
                val result = withContext(Dispatchers.Default) {
                    translationModel.translate(text, fromLanguage, toLanguage)
                }
                
                runOnUiThread {
                    tvTranslatedText.text = result.translatedText
                    tvConfidenceScore.text = "Confidence: ${String.format("%.2f", result.confidence)}"
                    
                    // Add to history
                    val entry = TranslationEntry(
                        originalText = text,
                        translatedText = result.translatedText,
                        fromLanguage = fromLanguage,
                        toLanguage = toLanguage,
                        confidence = result.confidence,
                        timestamp = System.currentTimeMillis()
                    )
                    addToHistory(entry)
                    
                    updateTranslationInfo("Translation complete (${result.inferenceTime}ms)")
                    showProgress(false)
                }
            } catch (e: Exception) {
                runOnUiThread {
                    updateTranslationInfo("Translation error: ${e.message}")
                    showProgress(false)
                }
            }
        }
    }
    
    private fun startVoiceRecognition() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, getSelectedLanguageCode(spinnerFromLanguage))
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }
        
        speechRecognizer.startListening(intent)
    }
    
    private fun startCameraTranslation() {
        // Launch camera translation activity
        val intent = Intent(this, CameraTranslationActivity::class.java).apply {
            putExtra("fromLanguage", getSelectedLanguageCode(spinnerFromLanguage))
            putExtra("toLanguage", getSelectedLanguageCode(spinnerToLanguage))
        }
        startActivity(intent)
    }
    
    private fun playTranslation() {
        val translatedText = tvTranslatedText.text.toString()
        if (translatedText.isNotEmpty() && isTTSReady) {
            val toLanguage = getSelectedLanguageCode(spinnerToLanguage)
            val locale = Locale(toLanguage)
            
            textToSpeech.language = locale
            textToSpeech.speak(translatedText, TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }
    
    private fun swapLanguages() {
        val fromPosition = spinnerFromLanguage.selectedItemPosition
        val toPosition = spinnerToLanguage.selectedItemPosition
        
        spinnerFromLanguage.setSelection(toPosition)
        spinnerToLanguage.setSelection(fromPosition)
        
        // Swap text content
        val originalText = etInputText.text.toString()
        val translatedText = tvTranslatedText.text.toString()
        
        etInputText.setText(translatedText)
        tvTranslatedText.text = originalText
        
        // Re-translate if real-time mode is enabled
        if (isRealTimeEnabled && translatedText.isNotEmpty()) {
            performTranslation(translatedText)
        }
    }
    
    private fun addToHistory(entry: TranslationEntry) {
        translationHistory.add(0, entry) // Add to beginning
        if (translationHistory.size > 50) {
            translationHistory.removeAt(translationHistory.size - 1) // Keep only last 50
        }
        
        // Update UI
        updateHistoryView()
    }
    
    private fun updateHistoryView() {
        layoutTranslationHistory.removeAllViews()
        
        for (entry in translationHistory.take(10)) { // Show only last 10 in UI
            val historyItem = layoutInflater.inflate(R.layout.item_translation_history, null)
            
            val tvOriginal = historyItem.findViewById<TextView>(R.id.tvOriginalText)
            val tvTranslated = historyItem.findViewById<TextView>(R.id.tvTranslatedText)
            val tvLanguages = historyItem.findViewById<TextView>(R.id.tvLanguages)
            val tvTimestamp = historyItem.findViewById<TextView>(R.id.tvTimestamp)
            
            tvOriginal.text = entry.originalText
            tvTranslated.text = entry.translatedText
            tvLanguages.text = "${entry.fromLanguage} → ${entry.toLanguage}"
            tvTimestamp.text = formatTimestamp(entry.timestamp)
            
            // Click to reuse translation
            historyItem.setOnClickListener {
                etInputText.setText(entry.originalText)
                tvTranslatedText.text = entry.translatedText
                
                // Update language spinners
                val fromIndex = SUPPORTED_LANGUAGES.keys.indexOf(entry.fromLanguage)
                val toIndex = SUPPORTED_LANGUAGES.keys.indexOf(entry.toLanguage)
                if (fromIndex >= 0) spinnerFromLanguage.setSelection(fromIndex)
                if (toIndex >= 0) spinnerToLanguage.setSelection(toIndex)
            }
            
            layoutTranslationHistory.addView(historyItem)
        }
    }
    
    private fun getSelectedLanguageCode(spinner: Spinner): String {
        val selectedLanguage = spinner.selectedItem.toString()
        return SUPPORTED_LANGUAGES.entries.find { it.value == selectedLanguage }?.key ?: "en"
    }
    
    private fun updateTranslationInfo(info: String) {
        tvTranslationInfo.text = info
    }
    
    private fun showProgress(show: Boolean) {
        progressBar.visibility = if (show) View.VISIBLE else View.GONE
    }
    
    private fun formatTimestamp(timestamp: Long): String {
        val date = Date(timestamp)
        return android.text.format.DateFormat.getTimeFormat(this).format(date)
    }
    
    // Permission handling
    private fun requestPermissions() {
        val permissions = mutableListOf<String>()
        
        if (!hasAudioPermission()) {
            permissions.add(Manifest.permission.RECORD_AUDIO)
        }
        
        if (!hasCameraPermission()) {
            permissions.add(Manifest.permission.CAMERA)
        }
        
        if (permissions.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, permissions.toTypedArray(), REQUEST_RECORD_AUDIO_PERMISSION)
        }
    }
    
    private fun hasAudioPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
    }
    
    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }
    
    private fun requestAudioPermission() {
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO_PERMISSION)
    }
    
    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA_PERMISSION)
    }
    
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        when (requestCode) {
            REQUEST_RECORD_AUDIO_PERMISSION -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    updateTranslationInfo("Audio permission granted")
                } else {
                    updateTranslationInfo("Audio permission denied")
                    btnVoiceInput.isEnabled = false
                }
            }
            REQUEST_CAMERA_PERMISSION -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    updateTranslationInfo("Camera permission granted")
                } else {
                    updateTranslationInfo("Camera permission denied")
                    btnCameraTranslate.isEnabled = false
                }
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        
        speechRecognizer.destroy()
        textToSpeech.shutdown()
        
        if (isRecording.get()) {
            stopRecording()
        }
    }
    
    private fun stopRecording() {
        isRecording.set(false)
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }
    
    // Data classes
    data class TranslationResult(
        val translatedText: String,
        val confidence: Float,
        val inferenceTime: Long
    )
    
    data class TranslationEntry(
        val originalText: String,
        val translatedText: String,
        val fromLanguage: String,
        val toLanguage: String,
        val confidence: Float,
        val timestamp: Long
    )
    
    // Translation model wrapper
    inner class TranslationModel(private val engine: TrustformersEngine) {
        private var isModelLoaded = false
        
        suspend fun loadModel(modelPath: String) {
            withContext(Dispatchers.Default) {
                try {
                    // Load translation model using TrustformersEngine
                    engine.loadModel(modelPath)
                    isModelLoaded = true
                } catch (e: Exception) {
                    throw RuntimeException("Failed to load translation model: ${e.message}")
                }
            }
        }
        
        suspend fun translate(text: String, fromLang: String, toLang: String): TranslationResult {
            if (!isModelLoaded) {
                throw RuntimeException("Translation model not loaded")
            }
            
            return withContext(Dispatchers.Default) {
                val startTime = System.currentTimeMillis()
                
                // Prepare input with language tokens
                val inputText = "[${fromLang}] ${text} [${toLang}]"
                
                // Tokenize input
                val tokenizedInput = tokenizeText(inputText)
                
                // Run inference
                val outputs = engine.inference(tokenizedInput)
                
                // Decode output
                val translatedText = decodeOutput(outputs)
                
                val inferenceTime = System.currentTimeMillis() - startTime
                
                // Calculate confidence (simplified)
                val confidence = calculateConfidence(outputs)
                
                TranslationResult(
                    translatedText = translatedText,
                    confidence = confidence,
                    inferenceTime = inferenceTime
                )
            }
        }
        
        private fun tokenizeText(text: String): FloatArray {
            // Simplified tokenization - in real implementation this would use proper tokenizer
            val tokens = text.lowercase().split(" ")
            val vocabulary = getVocabulary()
            
            val tokenIds = tokens.map { token ->
                vocabulary[token] ?: vocabulary["<unk>"] ?: 0
            }.toIntArray()
            
            // Convert to float array and pad/truncate to model input size
            val maxLength = 512
            val paddedTokens = IntArray(maxLength) { 0 }
            
            for (i in tokenIds.indices) {
                if (i < maxLength) {
                    paddedTokens[i] = tokenIds[i]
                }
            }
            
            return paddedTokens.map { it.toFloat() }.toFloatArray()
        }
        
        private fun decodeOutput(outputs: FloatArray): String {
            // Simplified decoding - in real implementation this would use proper decoder
            val vocabulary = getVocabulary()
            val reverseVocab = vocabulary.entries.associate { it.value to it.key }
            
            val tokens = mutableListOf<String>()
            var i = 0
            
            while (i < outputs.size) {
                val tokenId = outputs[i].toInt()
                val token = reverseVocab[tokenId]
                
                if (token != null && token != "<pad>" && token != "<unk>") {
                    tokens.add(token)
                }
                
                i++
                
                // Stop at end token or max length
                if (token == "<eos>" || tokens.size >= 100) {
                    break
                }
            }
            
            return tokens.joinToString(" ").trim()
        }
        
        private fun calculateConfidence(outputs: FloatArray): Float {
            // Simplified confidence calculation
            // In real implementation, this would be based on softmax probabilities
            val avgLogProb = outputs.filter { it != 0f }.average().toFloat()
            return kotlin.math.exp(avgLogProb).coerceIn(0f, 1f)
        }
        
        private fun getVocabulary(): Map<String, Int> {
            // Simplified vocabulary - in real implementation this would be loaded from model
            return mapOf(
                "<pad>" to 0,
                "<unk>" to 1,
                "<eos>" to 2,
                "the" to 3,
                "and" to 4,
                "to" to 5,
                "of" to 6,
                "a" to 7,
                "in" to 8,
                "is" to 9,
                "it" to 10,
                "you" to 11,
                "that" to 12,
                "he" to 13,
                "was" to 14,
                "for" to 15,
                "on" to 16,
                "are" to 17,
                "as" to 18,
                "with" to 19,
                "his" to 20,
                "they" to 21,
                "i" to 22,
                "at" to 23,
                "be" to 24,
                "this" to 25,
                "have" to 26,
                "from" to 27,
                "or" to 28,
                "one" to 29,
                "had" to 30,
                "by" to 31,
                "word" to 32,
                "but" to 33,
                "not" to 34,
                "what" to 35,
                "all" to 36,
                "were" to 37,
                "we" to 38,
                "when" to 39,
                "your" to 40,
                "can" to 41,
                "said" to 42,
                "there" to 43,
                "each" to 44,
                "which" to 45,
                "do" to 46,
                "how" to 47,
                "their" to 48,
                "if" to 49,
                "will" to 50,
                "up" to 51,
                "other" to 52,
                "about" to 53,
                "out" to 54,
                "many" to 55,
                "then" to 56,
                "them" to 57,
                "these" to 58,
                "so" to 59,
                "some" to 60,
                "her" to 61,
                "would" to 62,
                "make" to 63,
                "like" to 64,
                "into" to 65,
                "him" to 66,
                "time" to 67,
                "has" to 68,
                "two" to 69,
                "more" to 70,
                "go" to 71,
                "no" to 72,
                "way" to 73,
                "could" to 74,
                "my" to 75,
                "than" to 76,
                "first" to 77,
                "been" to 78,
                "call" to 79,
                "who" to 80,
                "oil" to 81,
                "its" to 82,
                "now" to 83,
                "find" to 84,
                "long" to 85,
                "down" to 86,
                "day" to 87,
                "did" to 88,
                "get" to 89,
                "come" to 90,
                "made" to 91,
                "may" to 92,
                "part" to 93,
                "over" to 94,
                "new" to 95,
                "sound" to 96,
                "take" to 97,
                "only" to 98,
                "little" to 99,
                "work" to 100
            )
        }
    }
}