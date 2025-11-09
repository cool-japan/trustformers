package ai.trustformers;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;

/**
 * Main TrustFormeRS Java API entry point providing access to transformer models,
 * tokenizers, and ML pipelines through JNI bindings.
 */
public class TrustFormeRS {
    
    // Native library loading
    private static boolean libraryLoaded = false;
    
    // Native function declarations
    private static native void initializeNative();
    private static native String getVersionNative();
    private static native boolean checkCudaAvailableNative();
    private static native boolean checkMetalAvailableNative();
    private static native int getDeviceCountNative();
    private static native String getDeviceInfoNative(int deviceId);
    
    static {
        loadNativeLibrary();
    }
    
    /**
     * Load the native TrustFormeRS library with automatic platform detection
     */
    private static synchronized void loadNativeLibrary() {
        if (libraryLoaded) {
            return;
        }
        
        try {
            // Try system library first
            System.loadLibrary("trustformers_java");
            libraryLoaded = true;
            initializeNative();
            return;
        } catch (UnsatisfiedLinkError e) {
            // Fall back to bundled library
        }
        
        // Determine platform and architecture
        String osName = System.getProperty("os.name").toLowerCase();
        String osArch = System.getProperty("os.arch").toLowerCase();
        
        String platform;
        if (osName.contains("win")) {
            platform = "windows";
        } else if (osName.contains("mac")) {
            platform = "macos";
        } else if (osName.contains("linux")) {
            platform = "linux";
        } else {
            throw new RuntimeException("Unsupported platform: " + osName);
        }
        
        String arch;
        if (osArch.contains("amd64") || osArch.contains("x86_64")) {
            arch = "x64";
        } else if (osArch.contains("aarch64") || osArch.contains("arm64")) {
            arch = "arm64";
        } else {
            throw new RuntimeException("Unsupported architecture: " + osArch);
        }
        
        // Load bundled library
        String libraryName = platform.equals("windows") ? 
            "trustformers_java.dll" : 
            (platform.equals("macos") ? "libtrusformers_java.dylib" : "libtrusformers_java.so");
        
        String resourcePath = "/native/" + platform + "/" + arch + "/" + libraryName;
        
        try {
            InputStream is = TrustFormeRS.class.getResourceAsStream(resourcePath);
            if (is == null) {
                throw new RuntimeException("Native library not found: " + resourcePath);
            }
            
            // Extract to temporary file
            Path tempFile = Files.createTempFile("trustformers_java", 
                platform.equals("windows") ? ".dll" : ".so");
            tempFile.toFile().deleteOnExit();
            
            Files.copy(is, tempFile, StandardCopyOption.REPLACE_EXISTING);
            is.close();
            
            // Load the extracted library
            System.load(tempFile.toAbsolutePath().toString());
            libraryLoaded = true;
            initializeNative();
            
        } catch (IOException e) {
            throw new RuntimeException("Failed to load native library", e);
        }
    }
    
    /**
     * Get TrustFormeRS library version
     * @return Version string
     */
    public static String getVersion() {
        return getVersionNative();
    }
    
    /**
     * Check if CUDA is available for GPU acceleration
     * @return true if CUDA is available
     */
    public static boolean isCudaAvailable() {
        return checkCudaAvailableNative();
    }
    
    /**
     * Check if Metal is available for GPU acceleration (macOS/iOS)
     * @return true if Metal is available
     */
    public static boolean isMetalAvailable() {
        return checkMetalAvailableNative();
    }
    
    /**
     * Get the number of available devices
     * @return Number of devices
     */
    public static int getDeviceCount() {
        return getDeviceCountNative();
    }
    
    /**
     * Get information about a specific device
     * @param deviceId Device ID (0-based)
     * @return Device information string
     */
    public static String getDeviceInfo(int deviceId) {
        return getDeviceInfoNative(deviceId);
    }
    
    /**
     * Get comprehensive device information for all available devices
     * @return Map of device ID to device information
     */
    public static Map<Integer, String> getAllDeviceInfo() {
        Map<Integer, String> deviceInfo = new HashMap<>();
        int deviceCount = getDeviceCount();
        
        for (int i = 0; i < deviceCount; i++) {
            deviceInfo.put(i, getDeviceInfo(i));
        }
        
        return deviceInfo;
    }
    
    /**
     * Print system information including devices and capabilities
     */
    public static void printSystemInfo() {
        System.out.println("TrustFormeRS Version: " + getVersion());
        System.out.println("CUDA Available: " + isCudaAvailable());
        System.out.println("Metal Available: " + isMetalAvailable());
        System.out.println("Device Count: " + getDeviceCount());
        
        Map<Integer, String> devices = getAllDeviceInfo();
        if (!devices.isEmpty()) {
            System.out.println("\nAvailable Devices:");
            for (Map.Entry<Integer, String> entry : devices.entrySet()) {
                System.out.println("  Device " + entry.getKey() + ": " + entry.getValue());
            }
        }
    }
}