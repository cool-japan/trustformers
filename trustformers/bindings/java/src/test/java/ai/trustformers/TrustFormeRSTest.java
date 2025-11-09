package ai.trustformers;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

import java.util.Map;

/**
 * Test suite for TrustFormeRS main API functionality.
 */
public class TrustFormeRSTest {
    
    @BeforeAll
    static void checkNativeLibrary() {
        // Assume native library is available for testing
        // In real tests, this would check if the native library loads successfully
        assumeTrue(true, "Native library should be available for testing");
    }
    
    @Test
    @DisplayName("Should get library version")
    void testGetVersion() {
        String version = TrustFormeRS.getVersion();
        assertNotNull(version, "Version should not be null");
        assertFalse(version.isEmpty(), "Version should not be empty");
        assertTrue(version.matches("\\d+\\.\\d+\\.\\d+.*"), "Version should follow semantic versioning");
    }
    
    @Test
    @DisplayName("Should check hardware capabilities")
    void testHardwareCapabilities() {
        // These methods should not throw exceptions
        assertDoesNotThrow(() -> {
            boolean cudaAvailable = TrustFormeRS.isCudaAvailable();
            boolean metalAvailable = TrustFormeRS.isMetalAvailable();
            int deviceCount = TrustFormeRS.getDeviceCount();
            
            // Device count should be non-negative
            assertTrue(deviceCount >= 0, "Device count should be non-negative");
            
            // If we have devices, we should be able to get info about them
            if (deviceCount > 0) {
                for (int i = 0; i < deviceCount; i++) {
                    String deviceInfo = TrustFormeRS.getDeviceInfo(i);
                    assertNotNull(deviceInfo, "Device info should not be null");
                    assertFalse(deviceInfo.isEmpty(), "Device info should not be empty");
                }
            }
        });
    }
    
    @Test
    @DisplayName("Should get all device information")
    void testGetAllDeviceInfo() {
        Map<Integer, String> deviceInfo = TrustFormeRS.getAllDeviceInfo();
        assertNotNull(deviceInfo, "Device info map should not be null");
        
        int deviceCount = TrustFormeRS.getDeviceCount();
        assertEquals(deviceCount, deviceInfo.size(), "Device info map size should match device count");
        
        // Check that all device IDs are present and have non-null info
        for (int i = 0; i < deviceCount; i++) {
            assertTrue(deviceInfo.containsKey(i), "Device info should contain device " + i);
            assertNotNull(deviceInfo.get(i), "Device info should not be null for device " + i);
        }
    }
    
    @Test
    @DisplayName("Should print system info without errors")
    void testPrintSystemInfo() {
        // This should not throw any exceptions
        assertDoesNotThrow(() -> TrustFormeRS.printSystemInfo());
    }
    
    @Test
    @DisplayName("Should handle invalid device ID gracefully")
    void testInvalidDeviceId() {
        int deviceCount = TrustFormeRS.getDeviceCount();
        
        // Test with device ID beyond available range
        assertDoesNotThrow(() -> {
            String info = TrustFormeRS.getDeviceInfo(deviceCount + 10);
            // Should either return null or empty string for invalid device
            if (info != null) {
                assertTrue(info.isEmpty() || info.contains("invalid") || info.contains("not found"));
            }
        });
        
        // Test with negative device ID
        assertDoesNotThrow(() -> {
            String info = TrustFormeRS.getDeviceInfo(-1);
            if (info != null) {
                assertTrue(info.isEmpty() || info.contains("invalid") || info.contains("not found"));
            }
        });
    }
}