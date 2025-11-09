package com.trustformers;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.*;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Utility class for loading the TrustformeRS native library.
 * Handles platform-specific library loading and extraction from JAR resources.
 */
public class NativeLibraryLoader {
    
    private static final Logger logger = LoggerFactory.getLogger(NativeLibraryLoader.class);
    private static final AtomicBoolean loaded = new AtomicBoolean(false);
    
    // Native library names for different platforms
    private static final String WINDOWS_LIB = "trustformers_c.dll";
    private static final String MACOS_LIB = "libtrustformers_c.dylib";
    private static final String LINUX_LIB = "libtrustformers_c.so";
    
    /**
     * Load the TrustformeRS native library.
     * This method is thread-safe and will only load the library once.
     * 
     * @throws UnsatisfiedLinkError if the library cannot be loaded
     */
    public static void loadLibrary() throws UnsatisfiedLinkError {
        if (loaded.get()) {
            return;
        }
        
        synchronized (NativeLibraryLoader.class) {
            if (loaded.get()) {
                return;
            }
            
            try {
                loadNativeLibrary();
                loaded.set(true);
                logger.info("TrustformeRS native library loaded successfully");
            } catch (Exception e) {
                logger.error("Failed to load TrustformeRS native library", e);
                throw new UnsatisfiedLinkError("Failed to load TrustformeRS native library: " + e.getMessage());
            }
        }
    }
    
    /**
     * Check if the native library has been loaded.
     * 
     * @return true if the library is loaded
     */
    public static boolean isLoaded() {
        return loaded.get();
    }
    
    private static void loadNativeLibrary() throws IOException {
        String osName = System.getProperty("os.name").toLowerCase();
        String osArch = System.getProperty("os.arch").toLowerCase();
        
        String libraryName = getLibraryName(osName);
        String libraryPath = getLibraryPath(osName, osArch);
        
        // Try to load from system paths first
        if (tryLoadFromSystem(libraryName)) {
            return;
        }
        
        // Try to load from JAR resources
        if (tryLoadFromJar(libraryPath, libraryName)) {
            return;
        }
        
        // Try to load from relative path (development environment)
        if (tryLoadFromDevelopmentPath(libraryName)) {
            return;
        }
        
        throw new UnsatisfiedLinkError("Could not locate TrustformeRS native library for " + osName + "/" + osArch);
    }
    
    private static String getLibraryName(String osName) {
        if (osName.contains("win")) {
            return WINDOWS_LIB;
        } else if (osName.contains("mac")) {
            return MACOS_LIB;
        } else {
            return LINUX_LIB;
        }
    }
    
    private static String getLibraryPath(String osName, String osArch) {
        String platform = getPlatformString(osName, osArch);
        return "/native/" + platform + "/" + getLibraryName(osName);
    }
    
    private static String getPlatformString(String osName, String osArch) {
        String os;
        if (osName.contains("win")) {
            os = "windows";
        } else if (osName.contains("mac")) {
            os = "macos";
        } else {
            os = "linux";
        }
        
        String arch;
        if (osArch.contains("64")) {
            arch = "x86_64";
        } else if (osArch.contains("arm") || osArch.contains("aarch64")) {
            arch = "aarch64";
        } else {
            arch = "x86";
        }
        
        return os + "-" + arch;
    }
    
    private static boolean tryLoadFromSystem(String libraryName) {
        try {
            // Try loading by name (assumes library is in system PATH/LD_LIBRARY_PATH)
            System.loadLibrary(getLibraryNameWithoutExtension(libraryName));
            logger.debug("Loaded native library from system path: {}", libraryName);
            return true;
        } catch (UnsatisfiedLinkError e) {
            logger.debug("Could not load from system path: {}", e.getMessage());
            return false;
        }
    }
    
    private static boolean tryLoadFromJar(String resourcePath, String libraryName) {
        try (InputStream is = NativeLibraryLoader.class.getResourceAsStream(resourcePath)) {
            if (is == null) {
                logger.debug("Library not found in JAR resources: {}", resourcePath);
                return false;
            }
            
            // Extract to temporary file
            Path tempDir = Files.createTempDirectory("trustformers-native");
            Path tempLib = tempDir.resolve(libraryName);
            
            Files.copy(is, tempLib, StandardCopyOption.REPLACE_EXISTING);
            
            // Load the temporary file
            System.load(tempLib.toAbsolutePath().toString());
            
            // Schedule cleanup on exit
            tempLib.toFile().deleteOnExit();
            tempDir.toFile().deleteOnExit();
            
            logger.debug("Loaded native library from JAR: {}", resourcePath);
            return true;
        } catch (IOException | UnsatisfiedLinkError e) {
            logger.debug("Could not load from JAR: {}", e.getMessage());
            return false;
        }
    }
    
    private static boolean tryLoadFromDevelopmentPath(String libraryName) {
        // Try common development paths
        String[] devPaths = {
            "../target/release/" + libraryName,
            "../../target/release/" + libraryName,
            "../../../target/release/" + libraryName,
            "target/release/" + libraryName
        };
        
        for (String path : devPaths) {
            try {
                File libFile = new File(path);
                if (libFile.exists()) {
                    System.load(libFile.getAbsolutePath());
                    logger.debug("Loaded native library from development path: {}", path);
                    return true;
                }
            } catch (UnsatisfiedLinkError e) {
                logger.debug("Could not load from development path {}: {}", path, e.getMessage());
            }
        }
        
        return false;
    }
    
    private static String getLibraryNameWithoutExtension(String libraryName) {
        if (libraryName.startsWith("lib") && !System.getProperty("os.name").toLowerCase().contains("win")) {
            // Remove "lib" prefix for Unix-like systems
            libraryName = libraryName.substring(3);
        }
        
        int dotIndex = libraryName.lastIndexOf('.');
        if (dotIndex > 0) {
            return libraryName.substring(0, dotIndex);
        }
        
        return libraryName;
    }
}