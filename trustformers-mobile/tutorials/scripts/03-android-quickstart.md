# Tutorial 3: Android Quick Start with TrustformeRS
**Duration**: 9 minutes  
**Target Audience**: Android developers  
**Prerequisites**: Android Studio, Kotlin knowledge, API 21+

## Video Script

### Opening (0:00 - 0:30)
**[Screen: Android Studio welcome screen]**

**Narrator**: "Welcome to the Android Quick Start tutorial! Today we're building an intelligent Android app using TrustformeRS that performs AI inference completely on-device. We'll use Jetpack Compose for a modern UI and Kotlin coroutines for smooth async operations."

**[Screen: Final app preview on Android device]**

"This is our finished app - a smart text analyzer that works offline with hardware acceleration. Let's build it step by step!"

### Project Setup (0:30 - 2:00)
**[Screen: Android Studio new project dialog]**

**Narrator**: "First, let's create a new Android project. Open Android Studio and select 'Create New Project'. Choose 'Empty Compose Activity' for a modern UI foundation."

**[Screen: Project configuration]**

"I'll name this 'TrustformeRSDemo', set the package name to 'com.example.trustformersdemo', and make sure we're targeting API 21 or higher for broad device compatibility."

**[Screen: Project structure view]**

"Perfect! Our project is created with Jetpack Compose already configured. Now let's add the TrustformeRS dependency."

### Adding TrustformeRS Dependency (2:00 - 3:00)
**[Screen: app/build.gradle.kts file]**

**Narrator**: "Open the app-level build.gradle.kts file and add TrustformeRS to our dependencies section:"

```kotlin
dependencies {
    implementation("com.trustformers:trustformers-android:1.0.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    
    // Existing Compose dependencies...
    implementation("androidx.compose.ui:ui:$compose_version")
    implementation("androidx.compose.ui:ui-tooling-preview:$compose_version")
    implementation("androidx.compose.material3:material3:1.1.2")
    implementation("androidx.activity:activity-compose:1.8.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.6.2")
}
```

**[Screen: Sync project button]**

"Click 'Sync Now' to download the dependencies. This might take a moment."

**[Screen: Sync successful notification]**

"Great! TrustformeRS is now ready to use in our project."

### Setting Up the ViewModel (3:00 - 4:30)
**[Screen: Creating new Kotlin file]**

**Narrator**: "Let's create a ViewModel to manage our AI engine and app state. Right-click on your package and create a new Kotlin file called 'MainViewModel'."

**[Screen: ViewModel code]**

```kotlin
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.trustformers.android.TrustformersEngine
import com.trustformers.android.TrustformersConfig
import com.trustformers.android.Backend
import com.trustformers.android.Quantization
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class MainViewModel : ViewModel() {
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()
    
    private var engine: TrustformersEngine? = null
    
    data class UiState(
        val inputText: String = "",
        val outputText: String = "Ready to analyze your text!",
        val isLoading: Boolean = false,
        val isModelLoaded: Boolean = false
    )
    
    init {
        loadModel()
    }
    
    private fun loadModel() {
        viewModelScope.launch {
            try {
                _uiState.value = _uiState.value.copy(
                    isLoading = true,
                    outputText = "Loading AI model..."
                )
                
                val config = TrustformersConfig.Builder()
                    .setModelName("bert-base-uncased")
                    .setQuantization(Quantization.INT8)
                    .setBackend(Backend.NNAPI)
                    .build()
                
                engine = TrustformersEngine.create(config)
                
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    isModelLoaded = true,
                    outputText = "Model loaded successfully! Enter text to analyze."
                )
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    outputText = "Error loading model: ${e.message}"
                )
            }
        }
    }
    
    fun updateInputText(text: String) {
        _uiState.value = _uiState.value.copy(inputText = text)
    }
    
    fun analyzeText() {
        val currentText = _uiState.value.inputText
        if (currentText.isBlank() || engine == null) return
        
        viewModelScope.launch {
            try {
                _uiState.value = _uiState.value.copy(isLoading = true)
                
                val result = engine!!.classify(currentText)
                
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    outputText = """
                        Classification Results:
                        
                        Label: ${result.label}
                        Confidence: ${"%.2f".format(result.confidence * 100)}%
                        
                        Processing Time: ${result.processingTimeMs}ms
                        Model: ${result.modelInfo.name}
                        Backend: ${result.modelInfo.backend}
                    """.trimIndent()
                )
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    outputText = "Error: ${e.message}"
                )
            }
        }
    }
    
    override fun onCleared() {
        super.onCleared()
        engine?.close()
    }
}
```

### Building the Compose UI (4:30 - 6:30)
**[Screen: MainActivity.kt file]**

**Narrator**: "Now let's build our Compose UI in MainActivity. We'll create a clean, Material 3 design with text input and output areas:"

```kotlin
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TrustformeRSDemo(
    viewModel: MainViewModel = viewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            text = "TrustformeRS Android Demo",
            style = MaterialTheme.typography.headlineMedium
        )
        
        OutlinedTextField(
            value = uiState.inputText,
            onValueChange = { viewModel.updateInputText(it) },
            label = { Text("Enter text to analyze") },
            modifier = Modifier
                .fillMaxWidth()
                .height(120.dp),
            maxLines = 4
        )
        
        Button(
            onClick = { viewModel.analyzeText() },
            modifier = Modifier.fillMaxWidth(),
            enabled = !uiState.isLoading && uiState.isModelLoaded && uiState.inputText.isNotBlank()
        ) {
            if (uiState.isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    strokeWidth = 2.dp
                )
                Spacer(modifier = Modifier.width(8.dp))
            }
            Text(if (uiState.isLoading) "Analyzing..." else "Analyze Text")
        }
        
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Results",
                    style = MaterialTheme.typography.titleMedium
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = uiState.outputText,
                    modifier = Modifier
                        .fillMaxWidth()
                        .verticalScroll(rememberScrollState()),
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }
    }
}
```

**[Screen: MainActivity class update]**

"And let's update our MainActivity to use this composable:"

```kotlin
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                TrustformeRSDemo()
            }
        }
    }
}
```

### Adding Permissions (6:30 - 7:00)
**[Screen: AndroidManifest.xml]**

**Narrator**: "TrustformeRS might need internet permission for initial model downloads. Let's add it to our AndroidManifest.xml:"

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    
    <uses-permission android:name="android.permission.INTERNET" />
    
    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:theme="@style/Theme.TrustformeRSDemo">
        
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:theme="@style/Theme.TrustformeRSDemo">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
        
    </application>
</manifest>
```

### Testing the App (7:00 - 8:30)
**[Screen: Android device/emulator selection]**

**Narrator**: "Time to test our app! I'll run it on an Android emulator first. Make sure you have an AVD with API 21 or higher."

**[Screen: App compilation and installation]**

"The app is building and installing. This first launch might take a bit longer as TrustformeRS downloads and caches the AI model."

**[Screen: App running on emulator]**

"Great! The app is loading the BERT model. You can see the loading indicator and status message."

**[Screen: Text input demonstration]**

"Now the model is loaded. Let me enter some text: 'I absolutely love this new Android app!' and tap 'Analyze Text'."

**[Screen: Results display]**

"Excellent! The model classified this as positive sentiment with 92% confidence, and it only took 31 milliseconds. That's incredibly fast for on-device processing!"

**[Screen: Testing on physical device]**

"Let me also test this on a real Android device to show you the performance with hardware acceleration..."

### Performance and Next Steps (8:30 - 9:00)
**[Screen: Performance comparison chart]**

**Narrator**: "Notice how much faster this runs on a real device with NNAPI acceleration. TrustformeRS automatically chooses the best backend for your hardware."

**[Screen: Feature highlights]**

"What makes this special? We're using INT8 quantization for efficiency, NNAPI for hardware acceleration, and everything runs completely offline. Your users' data never leaves their device."

**[Screen: Next tutorial preview]**

"In our next tutorial, we'll explore performance optimization techniques, including GPU acceleration and advanced quantization strategies. Don't forget to subscribe and I'll see you next time!"

## Supporting Materials

### Complete Project Files

#### build.gradle.kts (app)
```kotlin
dependencies {
    implementation("com.trustformers:trustformers-android:1.0.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.2")
    implementation("androidx.activity:activity-compose:1.8.0")
    implementation("androidx.compose.ui:ui:$compose_version")
    implementation("androidx.compose.ui:ui-tooling-preview:$compose_version")
    implementation("androidx.compose.material3:material3:1.1.2")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.6.2")
    
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
    androidTestImplementation("androidx.compose.ui:ui-test-junit4:$compose_version")
    debugImplementation("androidx.compose.ui:ui-tooling:$compose_version")
}
```

#### MainViewModel.kt (Complete)
[See above in video script]

#### MainActivity.kt (Complete)
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
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.trustformersdemo.ui.theme.TrustformeRSDemoTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            TrustformeRSDemoTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    TrustformeRSDemo()
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TrustformeRSDemo(
    viewModel: MainViewModel = viewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            text = "TrustformeRS Android Demo",
            style = MaterialTheme.typography.headlineMedium
        )
        
        OutlinedTextField(
            value = uiState.inputText,
            onValueChange = { viewModel.updateInputText(it) },
            label = { Text("Enter text to analyze") },
            modifier = Modifier
                .fillMaxWidth()
                .height(120.dp),
            maxLines = 4
        )
        
        Button(
            onClick = { viewModel.analyzeText() },
            modifier = Modifier.fillMaxWidth(),
            enabled = !uiState.isLoading && uiState.isModelLoaded && uiState.inputText.isNotBlank()
        ) {
            if (uiState.isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    strokeWidth = 2.dp
                )
                Spacer(modifier = Modifier.width(8.dp))
            }
            Text(if (uiState.isLoading) "Analyzing..." else "Analyze Text")
        }
        
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Results",
                    style = MaterialTheme.typography.titleMedium
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = uiState.outputText,
                    modifier = Modifier
                        .fillMaxWidth()
                        .verticalScroll(rememberScrollState()),
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }
    }
}
```

### Demo Text Examples
1. **Positive**: "I absolutely love this new Android app!"
2. **Negative**: "This experience was frustrating and disappointing."
3. **Neutral**: "The Android documentation explains the API."
4. **Technical**: "Kotlin coroutines provide asynchronous programming."
5. **Mixed**: "While the UI is beautiful, the performance could be better."

### Key Teaching Points
1. **Kotlin coroutines** with viewModelScope
2. **Jetpack Compose** state management
3. **StateFlow and collectAsState** patterns
4. **Material 3** design components
5. **MVVM architecture** with ViewModel
6. **Resource cleanup** with onCleared()

### Performance Optimization Tips
- Use NNAPI backend for hardware acceleration
- Enable INT8 quantization for memory efficiency
- Test on real devices for accurate performance metrics
- Monitor memory usage with Android Profiler
- Consider model size vs accuracy trade-offs
- Use background threads for model operations

### Troubleshooting Guide
- **Model loading slow**: Normal on first load, cached afterwards
- **OutOfMemoryError**: Use quantization or smaller models
- **Backend not available**: Falls back to CPU automatically
- **Permissions**: Internet only needed for initial model download
- **Emulator performance**: Much slower than real devices
- **Coroutine errors**: Ensure proper scope management