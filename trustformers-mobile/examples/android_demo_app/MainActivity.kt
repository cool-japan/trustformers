package com.trustformers.demo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.trustformers.TrustformersEngine
import com.trustformers.model.*
import kotlinx.coroutines.launch
import java.io.InputStream

class MainActivity : ComponentActivity() {
    private lateinit var trustformersEngine: TrustformersEngine
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize TrustformersRS Engine
        initializeTrustformers()
        
        setContent {
            TrustformersDemoTheme {
                TrustformersDemoApp(trustformersEngine)
            }
        }
    }
    
    private fun initializeTrustformers() {
        val config = TrustformersConfig.Builder()
            .enableNNAPI(true)
            .enableGPU(true)
            .enableVulkan(true)
            .setMaxConcurrentInferences(2)
            .setBatchSize(1)
            .build()
            
        trustformersEngine = TrustformersEngine.initialize(this, config)
        
        // Load default models
        lifecycleScope.launch {
            try {
                trustformersEngine.loadModel("bert-base", "models/bert_base.tflite")
                trustformersEngine.loadModel("yolo-v5", "models/yolo_v5.tflite")
                trustformersEngine.loadModel("pose-net", "models/pose_estimation.tflite")
                
                Toast.makeText(this@MainActivity, "🚀 TrustformersRS initialized successfully!", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, "❌ Failed to load models: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TrustformersDemoApp(engine: TrustformersEngine) {
    var selectedTab by remember { mutableStateOf(0) }
    
    val tabs = listOf(
        "Text AI" to Icons.Default.TextFields,
        "Vision AI" to Icons.Default.Camera,
        "Performance" to Icons.Default.Analytics
    )
    
    Column {
        // Top App Bar
        TopAppBar(
            title = { 
                Text(
                    "TrustformersRS Demo",
                    fontWeight = FontWeight.Bold
                )
            },
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = MaterialTheme.colorScheme.primary,
                titleContentColor = MaterialTheme.colorScheme.onPrimary
            )
        )
        
        // Tab Bar
        TabRow(
            selectedTabIndex = selectedTab,
            containerColor = MaterialTheme.colorScheme.surface
        ) {
            tabs.forEachIndexed { index, (title, icon) ->
                Tab(
                    selected = selectedTab == index,
                    onClick = { selectedTab = index },
                    text = { Text(title) },
                    icon = { Icon(icon, contentDescription = title) }
                )
            }
        }
        
        // Content
        when (selectedTab) {
            0 -> TextClassificationScreen(engine)
            1 -> VisionAIScreen(engine)
            2 -> PerformanceScreen(engine)
        }
    }
}

@Composable
fun TextClassificationScreen(engine: TrustformersEngine) {
    val context = LocalContext.current
    var inputText by remember { mutableStateOf("") }
    var isAnalyzing by remember { mutableStateOf(false) }
    var analysisResult by remember { mutableStateOf<TextClassificationResult?>(null) }
    
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            // Input Section
            Card(
                modifier = Modifier.fillMaxWidth(),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        "Enter text to analyze:",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    
                    BasicTextField(
                        value = inputText,
                        onValueChange = { inputText = it },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(120.dp)
                            .background(
                                Color.Gray.copy(alpha = 0.1f),
                                RoundedCornerShape(8.dp)
                            )
                            .padding(12.dp),
                        decorationBox = { innerTextField ->
                            if (inputText.isEmpty()) {
                                Text(
                                    "Type your text here...",
                                    color = Color.Gray
                                )
                            }
                            innerTextField()
                        }
                    )
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    Button(
                        onClick = {
                            if (inputText.isNotEmpty()) {
                                analyzeText(engine, inputText) { result ->
                                    analysisResult = result
                                    isAnalyzing = false
                                }
                                isAnalyzing = true
                            }
                        },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = inputText.isNotEmpty() && !isAnalyzing
                    ) {
                        if (isAnalyzing) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(16.dp),
                                color = MaterialTheme.colorScheme.onPrimary
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Analyzing...")
                        } else {
                            Icon(Icons.Default.Psychology, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Analyze Text")
                        }
                    }
                }
            }
        }
        
        // Results Section
        analysisResult?.let { result ->
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Text(
                            "Analysis Results:",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(bottom = 12.dp)
                        )
                        
                        result.predictions.forEach { prediction ->
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 4.dp),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    prediction.label,
                                    fontWeight = FontWeight.Medium
                                )
                                Text(
                                    "${(prediction.confidence * 100).toInt()}%",
                                    color = MaterialTheme.colorScheme.primary,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                "Processing Time:",
                                fontWeight = FontWeight.Medium
                            )
                            Text(
                                "${String.format("%.2f", result.processingTimeMs)} ms",
                                color = MaterialTheme.colorScheme.secondary,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun VisionAIScreen(engine: TrustformersEngine) {
    var selectedImage by remember { mutableStateOf<Bitmap?>(null) }
    var isDetecting by remember { mutableStateOf(false) }
    var detectionResult by remember { mutableStateOf<ObjectDetectionResult?>(null) }
    
    val imagePickerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val context = LocalContext.current
            val inputStream: InputStream? = context.contentResolver.openInputStream(uri)
            selectedImage = BitmapFactory.decodeStream(inputStream)
        }
    }
    
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            // Image Selection
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(250.dp)
                    .clickable { imagePickerLauncher.launch("image/*") },
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
            ) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    if (selectedImage != null) {
                        Image(
                            bitmap = selectedImage!!.asImageBitmap(),
                            contentDescription = "Selected image",
                            modifier = Modifier
                                .fillMaxSize()
                                .clip(RoundedCornerShape(8.dp))
                        )
                    } else {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Icon(
                                Icons.Default.PhotoCamera,
                                contentDescription = "Select image",
                                modifier = Modifier.size(48.dp),
                                tint = Color.Gray
                            )
                            Text(
                                "Tap to select image",
                                color = Color.Gray,
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                }
            }
        }
        
        item {
            // Detection Button
            Button(
                onClick = {
                    selectedImage?.let { bitmap ->
                        detectObjects(engine, bitmap) { result ->
                            detectionResult = result
                            isDetecting = false
                        }
                        isDetecting = true
                    }
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = selectedImage != null && !isDetecting
            ) {
                if (isDetecting) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        color = MaterialTheme.colorScheme.onPrimary
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Detecting...")
                } else {
                    Icon(Icons.Default.Visibility, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Detect Objects")
                }
            }
        }
        
        // Detection Results
        detectionResult?.let { result ->
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Text(
                            "Detected Objects:",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(bottom = 12.dp)
                        )
                        
                        result.detections.forEach { detection ->
                            Card(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 4.dp),
                                colors = CardDefaults.cardColors(
                                    containerColor = Color.White.copy(alpha = 0.8f)
                                )
                            ) {
                                Column(
                                    modifier = Modifier.padding(12.dp)
                                ) {
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween
                                    ) {
                                        Text(
                                            detection.className,
                                            fontWeight = FontWeight.Bold
                                        )
                                        Text(
                                            "${(detection.confidence * 100).toInt()}%",
                                            color = MaterialTheme.colorScheme.primary
                                        )
                                    }
                                    Text(
                                        "Position: (${detection.bbox.x.toInt()}, ${detection.bbox.y.toInt()})",
                                        fontSize = 12.sp,
                                        color = Color.Gray
                                    )
                                }
                            }
                        }
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                "Processing Time:",
                                fontWeight = FontWeight.Medium
                            )
                            Text(
                                "${String.format("%.2f", result.processingTimeMs)} ms",
                                color = MaterialTheme.colorScheme.secondary,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun PerformanceScreen(engine: TrustformersEngine) {
    var performanceMetrics by remember { mutableStateOf<PerformanceMetrics?>(null) }
    
    // Simulate performance monitoring
    LaunchedEffect(Unit) {
        performanceMetrics = PerformanceMetrics(
            cpuUsage = 45.2,
            memoryUsageMB = 512.0,
            gpuUsage = 32.1,
            batteryLevel = 0.78,
            thermalState = "Normal",
            processingQueueSize = 2
        )
    }
    
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        performanceMetrics?.let { metrics ->
            item {
                Card(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Text(
                            "System Performance",
                            fontSize = 20.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(bottom = 12.dp)
                        )
                        
                        MetricRow("CPU Usage", "${metrics.cpuUsage}%", Icons.Default.Memory)
                        MetricRow("Memory Usage", "${metrics.memoryUsageMB} MB", Icons.Default.Storage)
                        MetricRow("GPU Usage", "${metrics.gpuUsage}%", Icons.Default.Videocam)
                        MetricRow("Battery Level", "${(metrics.batteryLevel * 100).toInt()}%", Icons.Default.Battery4Bar)
                        MetricRow("Thermal State", metrics.thermalState, Icons.Default.Thermostat)
                        MetricRow("Queue Size", "${metrics.processingQueueSize}", Icons.Default.Queue)
                    }
                }
            }
            
            item {
                Card(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Text(
                            "Loaded Models",
                            fontSize = 20.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(bottom = 12.dp)
                        )
                        
                        ModelRow("BERT-Base", "Text Classification", "500 MB", "NNAPI")
                        ModelRow("YOLOv5", "Object Detection", "300 MB", "GPU")
                        ModelRow("PoseNet", "Pose Estimation", "200 MB", "CPU")
                    }
                }
            }
        }
    }
}

@Composable
fun MetricRow(label: String, value: String, icon: androidx.compose.ui.graphics.vector.ImageVector) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                icon,
                contentDescription = null,
                modifier = Modifier.size(20.dp),
                tint = MaterialTheme.colorScheme.primary
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                label,
                fontWeight = FontWeight.Medium
            )
        }
        Text(
            value,
            color = MaterialTheme.colorScheme.secondary,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
fun ModelRow(name: String, type: String, size: String, backend: String) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier.padding(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    name,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    size,
                    color = MaterialTheme.colorScheme.primary
                )
            }
            Text(
                "$type • $backend",
                fontSize = 12.sp,
                color = Color.Gray
            )
        }
    }
}

// Helper functions
private fun analyzeText(
    engine: TrustformersEngine,
    text: String,
    callback: (TextClassificationResult) -> Unit
) {
    // Simulate text analysis
    val predictions = listOf(
        Prediction("Positive", 0.85),
        Prediction("Neutral", 0.12),
        Prediction("Negative", 0.03)
    )
    
    val result = TextClassificationResult(
        predictions = predictions,
        processingTimeMs = (15..25).random().toDouble()
    )
    
    callback(result)
}

private fun detectObjects(
    engine: TrustformersEngine,
    bitmap: Bitmap,
    callback: (ObjectDetectionResult) -> Unit
) {
    // Simulate object detection
    val detections = listOf(
        Detection("Person", 0.92, BoundingBox(120.0, 80.0, 200.0, 400.0)),
        Detection("Chair", 0.78, BoundingBox(50.0, 300.0, 150.0, 180.0))
    )
    
    val result = ObjectDetectionResult(
        detections = detections,
        processingTimeMs = (25..35).random().toDouble()
    )
    
    callback(result)
}

@Composable
fun TrustformersDemoTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = Color(0xFF2196F3),
            secondary = Color(0xFF4CAF50),
            tertiary = Color(0xFFFF9800)
        )
    ) {
        content()
    }
}