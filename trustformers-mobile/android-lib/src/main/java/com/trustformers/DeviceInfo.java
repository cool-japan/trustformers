package com.trustformers;

import android.app.ActivityManager;
import android.content.Context;
import android.content.pm.FeatureInfo;
import android.content.pm.PackageManager;
import android.graphics.Point;
import android.os.BatteryManager;
import android.os.Build;
import android.os.Environment;
import android.os.PowerManager;
import android.os.StatFs;
import android.os.HardwarePropertiesManager;
import android.view.Display;
import android.view.WindowManager;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Device information and capability detection for Android
 * 
 * This class provides comprehensive information about the Android device
 * including hardware capabilities, thermal status, and performance characteristics.
 */
public class DeviceInfo {
    private static DeviceInfo instance;
    private final Context context;
    
    // Device properties
    private final String manufacturer;
    private final String model;
    private final String device;
    private final int apiLevel;
    private final String androidVersion;
    private final long totalMemoryMB;
    private final int cpuCores;
    private final String cpuArchitecture;
    private final boolean hasNEON;
    private final boolean hasFP16;
    private final boolean hasNNAPI;
    private final boolean hasGPU;
    private final GPUInfo gpuInfo;
    private final List<String> systemFeatures;
    
    /**
     * GPU information
     */
    public static class GPUInfo {
        private final String vendor;
        private final String renderer;
        private final String version;
        private final boolean supportsVulkan;
        private final boolean supportsOpenGLES3;
        
        GPUInfo(String vendor, String renderer, String version, 
                boolean supportsVulkan, boolean supportsOpenGLES3) {
            this.vendor = vendor;
            this.renderer = renderer;
            this.version = version;
            this.supportsVulkan = supportsVulkan;
            this.supportsOpenGLES3 = supportsOpenGLES3;
        }
        
        public String getVendor() {
            return vendor;
        }
        
        public String getRenderer() {
            return renderer;
        }
        
        public String getVersion() {
            return version;
        }
        
        public boolean supportsVulkan() {
            return supportsVulkan;
        }
        
        public boolean supportsOpenGLES3() {
            return supportsOpenGLES3;
        }
        
        /**
         * Check if this is a high-performance GPU
         * @return True if GPU is high-performance
         */
        public boolean isHighPerformance() {
            // Check for known high-performance GPUs
            String lowerRenderer = renderer.toLowerCase();
            return lowerRenderer.contains("adreno") && 
                   (lowerRenderer.contains("6") || lowerRenderer.contains("7")) ||
                   lowerRenderer.contains("mali-g7") ||
                   lowerRenderer.contains("mali-g8") ||
                   lowerRenderer.contains("powervr");
        }
    }
    
    private DeviceInfo(Context context) {
        this.context = context.getApplicationContext();
        
        // Basic device info
        this.manufacturer = Build.MANUFACTURER;
        this.model = Build.MODEL;
        this.device = Build.DEVICE;
        this.apiLevel = Build.VERSION.SDK_INT;
        this.androidVersion = Build.VERSION.RELEASE;
        
        // Memory info
        this.totalMemoryMB = getTotalMemory();
        
        // CPU info
        this.cpuCores = Runtime.getRuntime().availableProcessors();
        this.cpuArchitecture = getCpuArchitecture();
        this.hasNEON = hasNEONSupport();
        this.hasFP16 = hasFP16Support();
        
        // API support
        this.hasNNAPI = hasNNAPISupport();
        this.hasGPU = hasGPUSupport();
        
        // GPU info
        this.gpuInfo = detectGPUInfo();
        
        // System features
        this.systemFeatures = getSystemFeatures();
    }
    
    /**
     * Get singleton instance
     * @param context Application context
     * @return DeviceInfo instance
     */
    public static synchronized DeviceInfo getInstance(@NonNull Context context) {
        if (instance == null) {
            instance = new DeviceInfo(context);
        }
        return instance;
    }
    
    // Getters for device properties
    
    public String getManufacturer() {
        return manufacturer;
    }
    
    public String getModel() {
        return model;
    }
    
    public String getDevice() {
        return device;
    }
    
    public int getApiLevel() {
        return apiLevel;
    }
    
    public String getAndroidVersion() {
        return androidVersion;
    }
    
    public long getTotalMemoryMB() {
        return totalMemoryMB;
    }
    
    public int getCpuCores() {
        return cpuCores;
    }
    
    public String getCpuArchitecture() {
        return cpuArchitecture;
    }
    
    public boolean hasNEON() {
        return hasNEON;
    }
    
    public boolean hasFP16() {
        return hasFP16;
    }
    
    public boolean hasNNAPI() {
        return hasNNAPI;
    }
    
    public boolean hasGPU() {
        return hasGPU;
    }
    
    public GPUInfo getGPUInfo() {
        return gpuInfo;
    }
    
    /**
     * Check if device supports FP16 operations
     * @return True if FP16 is supported
     */
    public boolean supportsFP16() {
        return hasFP16 || (apiLevel >= Build.VERSION_CODES.O_MR1);
    }
    
    /**
     * Get available memory in MB
     * @return Available memory
     */
    public long getAvailableMemoryMB() {
        ActivityManager activityManager = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo memInfo = new ActivityManager.MemoryInfo();
        activityManager.getMemoryInfo(memInfo);
        return memInfo.availMem / (1024 * 1024);
    }
    
    /**
     * Get current memory pressure level
     * @return Memory pressure level (0.0 = no pressure, 1.0 = critical)
     */
    public float getMemoryPressure() {
        long available = getAvailableMemoryMB();
        long total = getTotalMemoryMB();
        float used = 1.0f - ((float) available / total);
        return Math.max(0.0f, Math.min(1.0f, used));
    }
    
    /**
     * Get thermal status
     * @return Thermal status
     */
    @RequiresApi(api = Build.VERSION_CODES.Q)
    public ThermalStatus getThermalStatus() {
        if (apiLevel >= Build.VERSION_CODES.Q) {
            PowerManager powerManager = (PowerManager) context.getSystemService(Context.POWER_SERVICE);
            int status = powerManager.getCurrentThermalStatus();
            
            switch (status) {
                case PowerManager.THERMAL_STATUS_NONE:
                    return ThermalStatus.NONE;
                case PowerManager.THERMAL_STATUS_LIGHT:
                    return ThermalStatus.LIGHT;
                case PowerManager.THERMAL_STATUS_MODERATE:
                    return ThermalStatus.MODERATE;
                case PowerManager.THERMAL_STATUS_SEVERE:
                    return ThermalStatus.SEVERE;
                case PowerManager.THERMAL_STATUS_CRITICAL:
                    return ThermalStatus.CRITICAL;
                case PowerManager.THERMAL_STATUS_EMERGENCY:
                    return ThermalStatus.EMERGENCY;
                case PowerManager.THERMAL_STATUS_SHUTDOWN:
                    return ThermalStatus.SHUTDOWN;
            }
        }
        return ThermalStatus.UNKNOWN;
    }
    
    /**
     * Get battery level
     * @return Battery level (0-100)
     */
    public int getBatteryLevel() {
        BatteryManager batteryManager = (BatteryManager) context.getSystemService(Context.BATTERY_SERVICE);
        return batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY);
    }
    
    /**
     * Check if device is charging
     * @return True if charging
     */
    public boolean isCharging() {
        BatteryManager batteryManager = (BatteryManager) context.getSystemService(Context.BATTERY_SERVICE);
        return batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_STATUS) == 
               BatteryManager.BATTERY_STATUS_CHARGING;
    }
    
    /**
     * Get device performance class
     * @return Performance class
     */
    @RequiresApi(api = Build.VERSION_CODES.S)
    public PerformanceClass getPerformanceClass() {
        if (apiLevel >= Build.VERSION_CODES.S) {
            int mediaPerformanceClass = Build.VERSION.MEDIA_PERFORMANCE_CLASS;
            
            if (mediaPerformanceClass >= 33) { // Android 13
                return PerformanceClass.CLASS_33;
            } else if (mediaPerformanceClass >= 31) { // Android 12
                return PerformanceClass.CLASS_31;
            } else if (mediaPerformanceClass >= 30) { // Android 11
                return PerformanceClass.CLASS_30;
            }
        }
        
        // Estimate based on hardware
        if (totalMemoryMB >= 8192 && cpuCores >= 8) {
            return PerformanceClass.HIGH_END;
        } else if (totalMemoryMB >= 4096 && cpuCores >= 6) {
            return PerformanceClass.MID_RANGE;
        } else {
            return PerformanceClass.ENTRY_LEVEL;
        }
    }
    
    /**
     * Get recommended thread count for inference
     * @return Recommended thread count
     */
    public int getRecommendedThreadCount() {
        // Use half of available cores for efficiency
        int cores = cpuCores;
        
        // Adjust based on thermal status if available
        if (apiLevel >= Build.VERSION_CODES.Q) {
            ThermalStatus thermal = getThermalStatus();
            switch (thermal) {
                case CRITICAL:
                case EMERGENCY:
                    return 1;
                case SEVERE:
                    return Math.max(1, cores / 4);
                case MODERATE:
                    return Math.max(1, cores / 2);
                default:
                    return Math.max(1, cores / 2);
            }
        }
        
        return Math.max(1, cores / 2);
    }
    
    /**
     * Get device summary
     * @return Human-readable device summary
     */
    public String getDeviceSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("Device: ").append(manufacturer).append(" ").append(model).append("\n");
        sb.append("Android: ").append(androidVersion).append(" (API ").append(apiLevel).append(")\n");
        sb.append("CPU: ").append(cpuArchitecture).append(" (").append(cpuCores).append(" cores)\n");
        sb.append("Memory: ").append(totalMemoryMB).append(" MB\n");
        
        if (gpuInfo != null) {
            sb.append("GPU: ").append(gpuInfo.renderer).append("\n");
        }
        
        if (hasNNAPI) {
            sb.append("NNAPI: Supported\n");
        }
        
        return sb.toString();
    }
    
    // Private helper methods
    
    private long getTotalMemory() {
        ActivityManager activityManager = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo memInfo = new ActivityManager.MemoryInfo();
        activityManager.getMemoryInfo(memInfo);
        return memInfo.totalMem / (1024 * 1024);
    }
    
    private String getCpuArchitecture() {
        String[] abis = Build.SUPPORTED_ABIS;
        if (abis.length > 0) {
            return abis[0];
        }
        return "unknown";
    }
    
    private boolean hasNEONSupport() {
        // NEON is standard on ARMv8 (arm64)
        if (cpuArchitecture.contains("arm64") || cpuArchitecture.contains("aarch64")) {
            return true;
        }
        
        // Check for NEON on ARMv7
        if (cpuArchitecture.contains("armeabi-v7a")) {
            try {
                File cpuInfo = new File("/proc/cpuinfo");
                RandomAccessFile reader = new RandomAccessFile(cpuInfo, "r");
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.toLowerCase().contains("neon")) {
                        reader.close();
                        return true;
                    }
                }
                reader.close();
            } catch (IOException e) {
                // Ignore
            }
        }
        
        return false;
    }
    
    private boolean hasFP16Support() {
        // FP16 is supported on newer ARM processors
        return cpuArchitecture.contains("arm64") || cpuArchitecture.contains("aarch64");
    }
    
    private boolean hasNNAPISupport() {
        // NNAPI requires Android 8.1 (API 27) or higher
        return apiLevel >= Build.VERSION_CODES.O_MR1;
    }
    
    private boolean hasGPUSupport() {
        PackageManager pm = context.getPackageManager();
        
        // Check for OpenGL ES 3.0 support
        FeatureInfo[] features = pm.getSystemAvailableFeatures();
        for (FeatureInfo feature : features) {
            if (feature.name != null) {
                if (feature.name.equals("android.hardware.opengles.aep")) {
                    return true;
                }
            }
        }
        
        // Check for Vulkan support
        if (apiLevel >= Build.VERSION_CODES.N) {
            return pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL) ||
                   pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_VERSION);
        }
        
        return true; // Most devices have some GPU
    }
    
    private GPUInfo detectGPUInfo() {
        // In a real implementation, this would use EGL to query GPU info
        // For now, return placeholder data
        
        boolean supportsVulkan = false;
        boolean supportsOpenGLES3 = true;
        
        if (apiLevel >= Build.VERSION_CODES.N) {
            PackageManager pm = context.getPackageManager();
            supportsVulkan = pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL);
        }
        
        return new GPUInfo(
            "Unknown",
            "Unknown GPU",
            "OpenGL ES 3.0",
            supportsVulkan,
            supportsOpenGLES3
        );
    }
    
    private List<String> getSystemFeatures() {
        List<String> features = new ArrayList<>();
        PackageManager pm = context.getPackageManager();
        
        FeatureInfo[] featureInfos = pm.getSystemAvailableFeatures();
        for (FeatureInfo feature : featureInfos) {
            if (feature.name != null) {
                features.add(feature.name);
            }
        }
        
        return features;
    }
    
    /**
     * Thermal status levels
     */
    public enum ThermalStatus {
        UNKNOWN,
        NONE,
        LIGHT,
        MODERATE,
        SEVERE,
        CRITICAL,
        EMERGENCY,
        SHUTDOWN
    }
    
    /**
     * Performance class
     */
    public enum PerformanceClass {
        ENTRY_LEVEL,
        MID_RANGE,
        HIGH_END,
        CLASS_30,  // Android 11 performance class
        CLASS_31,  // Android 12 performance class
        CLASS_33   // Android 13 performance class
    }
}