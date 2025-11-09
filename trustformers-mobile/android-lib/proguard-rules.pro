# TrustformeRS Android ProGuard Rules

# Keep all public API classes
-keep public class com.trustformers.** {
    public protected *;
}

# Keep JNI methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep enums
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# Keep inner classes
-keepattributes InnerClasses
-keep class com.trustformers.**$* {
    *;
}

# Keep Kotlin metadata
-keepattributes *Annotation*
-keepattributes RuntimeVisibleAnnotations
-keepattributes RuntimeInvisibleAnnotations
-keepattributes RuntimeVisibleParameterAnnotations
-keepattributes RuntimeInvisibleParameterAnnotations
-keepattributes EnclosingMethod
-keepattributes Signature
-keepattributes Exceptions

# Kotlin specific
-dontwarn kotlin.**
-keep class kotlin.Metadata { *; }
-keepclassmembers class **$WhenMappings {
    <fields>;
}
-keepclassmembers class kotlin.Metadata {
    public <methods>;
}

# Coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
-keepclassmembernames class kotlinx.** {
    volatile <fields>;
}

# Keep performance monitoring classes
-keep class com.trustformers.PerformanceMonitor { *; }
-keep class com.trustformers.PerformanceMonitor$* { *; }

# Keep device info
-keep class com.trustformers.DeviceInfo { *; }
-keep class com.trustformers.DeviceInfo$* { *; }

# Keep tensor operations
-keep class com.trustformers.Tensor { *; }
-keep class com.trustformers.Tensor$* { *; }

# Suppress warnings for missing classes (will be provided by app)
-dontwarn androidx.**
-dontwarn android.**