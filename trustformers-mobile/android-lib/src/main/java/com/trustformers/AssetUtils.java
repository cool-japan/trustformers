package com.trustformers;

import android.content.Context;
import android.content.res.AssetManager;
import androidx.annotation.NonNull;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Utility class for handling assets in TrustformeRS Android
 */
public class AssetUtils {
    
    private AssetUtils() {
        // Private constructor to prevent instantiation
    }
    
    /**
     * Copy an asset file to the file system
     * @param context Application context
     * @param assetPath Path to asset file
     * @param destFile Destination file
     * @throws IOException if copy fails
     */
    public static void copyAssetToFile(@NonNull Context context, 
                                      @NonNull String assetPath, 
                                      @NonNull File destFile) throws IOException {
        AssetManager assetManager = context.getAssets();
        
        try (InputStream in = assetManager.open(assetPath);
             OutputStream out = new FileOutputStream(destFile)) {
            
            byte[] buffer = new byte[8192];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
            
            out.flush();
        }
    }
    
    /**
     * Copy an asset directory to the file system
     * @param context Application context
     * @param assetDir Asset directory path
     * @param destDir Destination directory
     * @throws IOException if copy fails
     */
    public static void copyAssetDirectory(@NonNull Context context,
                                         @NonNull String assetDir,
                                         @NonNull File destDir) throws IOException {
        AssetManager assetManager = context.getAssets();
        
        // Create destination directory if it doesn't exist
        if (!destDir.exists() && !destDir.mkdirs()) {
            throw new IOException("Failed to create directory: " + destDir);
        }
        
        String[] assets = assetManager.list(assetDir);
        if (assets == null || assets.length == 0) {
            // It's a file, not a directory
            File destFile = new File(destDir.getParentFile(), destDir.getName());
            copyAssetToFile(context, assetDir, destFile);
            return;
        }
        
        // Copy all files in the directory
        for (String asset : assets) {
            String assetPath = assetDir + "/" + asset;
            File destFile = new File(destDir, asset);
            
            String[] subAssets = assetManager.list(assetPath);
            if (subAssets != null && subAssets.length > 0) {
                // It's a directory
                copyAssetDirectory(context, assetPath, destFile);
            } else {
                // It's a file
                copyAssetToFile(context, assetPath, destFile);
            }
        }
    }
    
    /**
     * Check if an asset exists
     * @param context Application context
     * @param assetPath Path to check
     * @return True if asset exists
     */
    public static boolean assetExists(@NonNull Context context, @NonNull String assetPath) {
        AssetManager assetManager = context.getAssets();
        try (InputStream stream = assetManager.open(assetPath)) {
            return true;
        } catch (IOException e) {
            return false;
        }
    }
    
    /**
     * Get asset size in bytes
     * @param context Application context
     * @param assetPath Asset path
     * @return Size in bytes or -1 if not found
     */
    public static long getAssetSize(@NonNull Context context, @NonNull String assetPath) {
        AssetManager assetManager = context.getAssets();
        try (InputStream stream = assetManager.open(assetPath)) {
            return stream.available();
        } catch (IOException e) {
            return -1;
        }
    }
    
    /**
     * Extract model from assets if needed
     * @param context Application context
     * @param modelAssetPath Path to model in assets
     * @return File pointing to extracted model
     * @throws IOException if extraction fails
     */
    public static File extractModelIfNeeded(@NonNull Context context, 
                                           @NonNull String modelAssetPath) throws IOException {
        // Use app's cache directory for extracted models
        File modelsDir = new File(context.getCacheDir(), "models");
        if (!modelsDir.exists() && !modelsDir.mkdirs()) {
            throw new IOException("Failed to create models directory");
        }
        
        // Generate unique filename based on asset path
        String fileName = modelAssetPath.replace('/', '_');
        File modelFile = new File(modelsDir, fileName);
        
        // Check if already extracted and up to date
        if (modelFile.exists()) {
            long assetSize = getAssetSize(context, modelAssetPath);
            if (assetSize > 0 && modelFile.length() == assetSize) {
                // File already extracted and same size
                return modelFile;
            }
        }
        
        // Extract the model
        copyAssetToFile(context, modelAssetPath, modelFile);
        return modelFile;
    }
    
    /**
     * Clean up extracted models
     * @param context Application context
     */
    public static void cleanupExtractedModels(@NonNull Context context) {
        File modelsDir = new File(context.getCacheDir(), "models");
        if (modelsDir.exists()) {
            deleteRecursive(modelsDir);
        }
    }
    
    private static void deleteRecursive(File file) {
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteRecursive(child);
                }
            }
        }
        file.delete();
    }
}