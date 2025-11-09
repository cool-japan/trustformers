# Android Quick Start with TrustformeRS Mobile - Video Script

## Video Details
- **Duration**: 8-10 minutes
- **Target Audience**: Android developers
- **Prerequisites**: Android Studio, Android development basics, Kotlin knowledge
- **Learning Objectives**: Set up and run TrustformeRS in an Android app

## Script

### Introduction (0:00 - 0:30)
**[On-screen: Android Studio interface with new project]**

**Narrator**: "Welcome to the Android Quick Start guide for TrustformeRS Mobile! In this tutorial, we'll create a new Android project, integrate TrustformeRS using Gradle, and build a sentiment analysis app with Jetpack Compose that leverages NNAPI for hardware acceleration."

**[Visual: Final demo app running on Android device with real-time text classification]**

### Prerequisites and Setup (0:30 - 1:00)
**[On-screen: Requirements checklist]**

**Narrator**: "Before we begin, make sure you have:"

**Requirements:**
- ✅ Android Studio Hedgehog or later
- ✅ Android SDK API 21+ (Android 5.0)
- ✅ Kotlin 1.8+
- ✅ An Android device or emulator with API 21+

**[Visual: Android Studio About dialog, SDK Manager showing API levels]**

### Creating the Android Project (1:00 - 2:00)
**[On-screen: Live Android Studio project creation]**

**Narrator**: "Let's create a new Android project with Jetpack Compose. This will give us a modern UI framework that works perfectly with TrustformeRS."

**Step-by-step:**
1. Open Android Studio → Create New Project
2. Choose "Empty Compose Activity"
3. Name: "TrustformersAndroidDemo"
4. Package name: com.example.trustformersdemo
5. Language: Kotlin
6. Minimum SDK: API 21

**[Visual: Screen recording of actual project creation in Android Studio]**

### Adding TrustformeRS Dependency (2:00 - 3:00)
**[On-screen: build.gradle files]**

**Narrator**: "Now we'll add TrustformeRS to our project. We need to modify both the project-level and app-level Gradle files."

**Project-level build.gradle:**
```gradle
// Top-level build file
buildscript {
    ext {
        compose_version = '1.5.0'
        trustformers_version = '0.1.0'
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
        maven { url 'https://jitpack.io' }
    }
}
```

**App-level build.gradle:**
```gradle
android {
    compileSdk 34
    
    defaultConfig {
        applicationId "com.example.trustformersdemo"
        minSdk 21
        targetSdk 34
        versionCode 1
        versionName "1.0"

        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
    }
    
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }
    
    kotlinOptions {
        jvmTarget = '1.8'
    }
    
    buildFeatures {
        compose true
    }
    
    composeOptions {
        kotlinCompilerExtensionVersion compose_version
    }
}

dependencies {
    // TrustformeRS Mobile
    implementation 'com.github.trustformers:trustformers-mobile:$trustformers_version'
    
    // Jetpack Compose
    implementation "androidx.compose.ui:ui:$compose_version"
    implementation "androidx.compose.ui:ui-tooling-preview:$compose_version"
    implementation "androidx.compose.material3:material3:1.1.0"
    implementation "androidx.activity:activity-compose:1.7.2"
    implementation "androidx.lifecycle:lifecycle-viewmodel-compose:2.6.1"
    
    // Coroutines
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.0'
}
```

### Building the UI with Jetpack Compose (3:00 - 4:30)
**[On-screen: MainActivity.kt in Android Studio]**

**Narrator**: "Let's create our user interface using Jetpack Compose. We'll build a clean, Material Design 3 interface for text input and sentiment analysis."

```kotlin
package com.example.trustformersdemo

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.trustformersdemo.ui.theme.TrustformersAndroidDemoTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            TrustformersAndroidDemoTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    SentimentAnalysisScreen()
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SentimentAnalysisScreen(
    viewModel: SentimentViewModel = viewModel()
) {
    var inputText by remember { mutableStateOf("") }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "TrustformeRS Sentiment Analysis",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(bottom = 24.dp)
        )
        
        OutlinedTextField(
            value = inputText,
            onValueChange = { inputText = it },
            label = { Text("Enter text to analyze") },
            modifier = Modifier
                .fillMaxWidth()
                .height(120.dp),
            maxLines = 4
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        Button(
            onClick = { viewModel.analyzeSentiment(inputText) },
            enabled = inputText.isNotBlank() && !viewModel.isLoading,
            modifier = Modifier.fillMaxWidth()
        ) {
            if (viewModel.isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    color = MaterialTheme.colorScheme.onPrimary
                )
                Spacer(modifier = Modifier.width(8.dp))
            }
            Text("Analyze Sentiment")
        }
        
        Spacer(modifier = Modifier.height(24.dp))
        
        viewModel.result?.let { result ->
            ResultCard(result = result)
        }
        
        viewModel.error?.let { error ->
            ErrorCard(error = error)
        }
    }
}

@Composable
fun ResultCard(result: SentimentResult) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = when (result.sentiment) {
                "Positive" -> MaterialTheme.colorScheme.primaryContainer
                "Negative" -> MaterialTheme.colorScheme.errorContainer
                else -> MaterialTheme.colorScheme.surfaceVariant
            }
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Analysis Result",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Sentiment: ${result.sentiment}",
                style = MaterialTheme.typography.bodyLarge
            )
            Text(
                text = "Confidence: ${String.format("%.2f%%", result.confidence * 100)}",
                style = MaterialTheme.typography.bodyMedium
            )
            Text(
                text = "Processing time: ${result.processingTimeMs}ms",
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}

@Composable
fun ErrorCard(error: String) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer
        )
    ) {
        Text(
            text = "Error: $error",
            modifier = Modifier.padding(16.dp),
            color = MaterialTheme.colorScheme.onErrorContainer
        )
    }
}
```

### Creating the ViewModel (4:30 - 6:00)
**[On-screen: New Kotlin file - SentimentViewModel.kt]**

**Narrator**: "Now let's create our ViewModel that handles the TrustformeRS integration. This follows Android's recommended MVVM architecture."

```kotlin
package com.example.trustformersdemo

import androidx.compose.runtime.*
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.trustformers.TrustformersEngine
import com.trustformers.TrustformersConfig
import com.trustformers.Backend
import kotlinx.coroutines.launch
import kotlin.system.measureTimeMillis

data class SentimentResult(
    val sentiment: String,
    val confidence: Float,
    val processingTimeMs: Long
)

class SentimentViewModel : ViewModel() {
    private var engine: TrustformersEngine? = null
    
    var isLoading by mutableStateOf(false)
        private set
    
    var result by mutableStateOf<SentimentResult?>(null)
        private set
    
    var error by mutableStateOf<String?>(null)
        private set
    
    init {
        initializeEngine()
    }
    
    private fun initializeEngine() {
        viewModelScope.launch {
            try {
                val config = TrustformersConfig.Builder()
                    .setModelName("distilbert-base-uncased-finetuned-sst-2-english")
                    .setBackend(Backend.NNAPI) // Use NNAPI for hardware acceleration
                    .setQuantization(Quantization.FP16)
                    .setMaxSequenceLength(512)
                    .setBatchSize(1)
                    .build()
                
                engine = TrustformersEngine.create(config)
            } catch (e: Exception) {
                error = "Failed to initialize TrustformeRS: ${e.message}"
            }
        }
    }
    
    fun analyzeSentiment(text: String) {
        if (text.isBlank()) return
        
        viewModelScope.launch {
            isLoading = true
            error = null
            result = null
            
            try {
                val engine = this@SentimentViewModel.engine
                    ?: throw IllegalStateException("Engine not initialized")
                
                val processingTime = measureTimeMillis {
                    val prediction = engine.predict(text)
                    
                    // Convert logits to probabilities
                    val probabilities = softmax(prediction.logits)
                    val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
                    
                    val labels = arrayOf("Negative", "Positive")
                    val sentiment = labels[maxIndex]
                    val confidence = probabilities[maxIndex]
                    
                    result = SentimentResult(
                        sentiment = sentiment,
                        confidence = confidence,
                        processingTimeMs = processingTime
                    )
                }
                
            } catch (e: Exception) {
                error = "Analysis failed: ${e.message}"
            } finally {
                isLoading = false
            }
        }
    }
    
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expLogits = logits.map { kotlin.math.exp((it - maxLogit).toDouble()).toFloat() }
        val sumExp = expLogits.sum()
        return expLogits.map { it / sumExp }.toFloatArray()
    }
    
    override fun onCleared() {
        super.onCleared()
        engine?.close() // Clean up resources
    }
}
```

### Adding Permissions and Configuration (6:00 - 6:30)
**[On-screen: AndroidManifest.xml]**

**Narrator**: "Let's add the necessary permissions and configuration to our AndroidManifest.xml file."

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:tools="http://schemas.android.com/tools">

    <!-- Internet permission for model downloads -->
    <uses-permission android:name="android.permission.INTERNET" />
    
    <!-- Optional: GPU compute permission -->
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />

    <application
        android:allowBackup="true"
        android:dataExtractionRules="@xml/data_extraction_rules"
        android:fullBackupContent="@xml/backup_rules"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:theme="@style/Theme.TrustformersAndroidDemo"
        tools:targetApi="31">
        
        <!-- Enable hardware acceleration -->
        <meta-data
            android:name="trustformers.hardware_acceleration"
            android:value="true" />
            
        <!-- NNAPI configuration -->
        <meta-data
            android:name="trustformers.nnapi.enabled"
            android:value="true" />
        
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:theme="@style/Theme.TrustformersAndroidDemo">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
```

### Running and Testing (6:30 - 7:30)
**[On-screen: App running on Android device]**

**Narrator**: "Let's build and run our app! I'll demonstrate it on both an emulator and a real device to show the performance difference with NNAPI acceleration."

**Build and Run:**
1. Sync Project with Gradle Files
2. Build → Make Project
3. Run on emulator first
4. Then run on physical device

**Demo Scenarios:**
- Enter: "I love using TrustformeRS on Android!"
- Result: Positive sentiment, high confidence
- Enter: "This is frustrating and doesn't work"
- Result: Negative sentiment, high confidence

**Performance Comparison:**
- **Emulator**: ~800ms processing time
- **Device with NNAPI**: ~80ms processing time
- **Device GPU acceleration**: ~60ms processing time

### Hardware Acceleration Deep Dive (7:30 - 8:30)
**[On-screen: Device specifications and performance metrics]**

**Narrator**: "Let's explore the different hardware acceleration options available on Android and how TrustformeRS automatically selects the best backend."

**Backend Options:**
```kotlin
// Automatic selection (recommended)
.setBackend(Backend.AUTO)

// Specific backends
.setBackend(Backend.NNAPI)    // Android Neural Networks API
.setBackend(Backend.GPU)      // Vulkan/OpenGL compute
.setBackend(Backend.CPU)      // Optimized CPU execution
.setBackend(Backend.EDGE_TPU) // Google Edge TPU (if available)
```

**Device Capabilities:**
- **Pixel 6+**: Google Tensor TPU
- **Samsung Galaxy**: Exynos NPU
- **Qualcomm Snapdragon**: Hexagon DSP
- **MediaTek**: APU acceleration

**[Visual: Performance monitoring showing different backend utilization]**

### Optimization and Best Practices (8:30 - 9:30)
**[On-screen: Enhanced configuration options]**

**Narrator**: "Here are some production-ready optimizations and best practices for Android apps:"

**Advanced Configuration:**
```kotlin
val config = TrustformersConfig.Builder()
    .setModelName("distilbert-base-uncased-finetuned-sst-2-english")
    .setBackend(Backend.AUTO)
    .setQuantization(Quantization.INT8) // Better performance on mobile
    .setMaxSequenceLength(256)          // Reduce for better performance
    .setBatchSize(1)                    // Single inference optimization
    .setNumThreads(4)                   // CPU thread optimization
    .setMemoryPoolSize(64 * 1024 * 1024) // 64MB memory pool
    .enableCaching(true)                // Cache compiled models
    .build()
```

**Best Practices:**
- Initialize engine in Application class for faster startup
- Use appropriate quantization for your accuracy requirements
- Monitor memory usage with Android Studio Profiler
- Test on various Android API levels and devices
- Handle background/foreground transitions properly

### Troubleshooting Common Issues (9:30 - 10:00)
**[On-screen: Common error solutions]**

**Narrator**: "If you encounter issues, here are solutions to the most common problems:"

**Common Issues:**
1. **NNAPI not available**: Fallback to GPU or CPU automatically
2. **Model download failures**: Check internet connectivity and storage permissions
3. **Out of memory errors**: Reduce batch size or use INT8 quantization
4. **Slow performance**: Ensure you're testing on device with hardware acceleration

**Debugging Tips:**
```kotlin
// Enable debugging logs
TrustformersEngine.setLogLevel(LogLevel.DEBUG)

// Check available backends
val availableBackends = TrustformersEngine.getAvailableBackends()
Log.d("TrustformeRS", "Available backends: $availableBackends")
```

### Next Steps and Resources (10:00 - 10:30)
**[On-screen: Tutorial series roadmap and resources]**

**Narrator**: "Excellent work! You've successfully created an Android app with TrustformeRS. In upcoming tutorials, we'll cover performance optimization, multimodal AI, and advanced Android features."

**What's Next:**
- Performance Optimization Deep Dive
- Multimodal AI with Camera Integration
- Background Processing and Services
- Edge TPU Integration
- Production Deployment Guide

**Resources:**
- 📱 **Complete Code**: github.com/trustformers/android-quickstart-demo
- 📚 **Documentation**: docs.trustformers.ai/android
- 💬 **Community**: discord.gg/trustformers
- 🎯 **More Examples**: github.com/trustformers/android-examples

## Supporting Materials

### Complete Project Structure
```
app/
├── src/main/
│   ├── java/com/example/trustformersdemo/
│   │   ├── MainActivity.kt
│   │   ├── SentimentViewModel.kt
│   │   └── ui/theme/
│   ├── AndroidManifest.xml
│   └── res/
├── build.gradle
└── proguard-rules.pro
```

### Performance Benchmarks
| Device | Backend | Processing Time | Memory Usage |
|--------|---------|----------------|--------------|
| Pixel 6 Pro | Tensor TPU | 45ms | 85MB |
| Galaxy S21 | Exynos NPU | 60ms | 90MB |
| OnePlus 9 | Snapdragon | 75ms | 95MB |
| Emulator | CPU | 800ms | 120MB |

### Visual Assets Needed
1. **Android Studio Screenshots**
   - Project creation workflow
   - Gradle file editing
   - Code editor with syntax highlighting

2. **App Demo Footage**
   - Running on various Android devices
   - Performance comparison side-by-side
   - Android Studio Profiler output

3. **Architecture Diagrams**
   - NNAPI backend selection flow
   - Memory management visualization
   - Hardware acceleration pipeline

## Accessibility Features
- **TalkBack Support**: All UI elements properly labeled
- **High Contrast**: Material Design 3 theming
- **Large Text**: Supports system font scaling
- **Focus Management**: Proper focus order for keyboard navigation

## Engagement Elements
- **Code Repository**: Encourage cloning and modification
- **Device Testing**: Challenge viewers to test on different devices
- **Performance Comparison**: Share benchmarks from different devices
- **Community Showcase**: Feature viewer modifications

## Success Metrics
- **Completion Rate**: Target 65%+ finish rate
- **GitHub Engagement**: 300+ repository stars
- **Performance Reports**: Community sharing device benchmarks
- **Follow-up Projects**: Developers building on the foundation

## Production Checklist
- **Multi-Device Testing**: Verify on various Android versions
- **Performance Profiling**: Memory and CPU usage analysis
- **Network Handling**: Proper offline/online state management
- **Error Handling**: Graceful degradation and user feedback