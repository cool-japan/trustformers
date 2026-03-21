# Consumer ProGuard rules for TrustformeRS Android
# These rules will be applied to apps that use this library

# Keep all public API
-keep class com.trustformers.** { *; }

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep enums used in the API
-keepclassmembers enum com.trustformers.** {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# Keep Kotlin extensions
-keep class com.trustformers.TrustformersKt { *; }
-keep class com.trustformers.TrustformersKtKt { *; }