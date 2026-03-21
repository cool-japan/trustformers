package com.trustformers;

import androidx.annotation.NonNull;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

/**
 * Tensor representation for TrustformeRS Android
 * 
 * This class represents multi-dimensional arrays used for model input/output.
 * Supports various data types and operations commonly needed for inference.
 */
public class Tensor {
    private final float[] data;
    private final int[] shape;
    private final int totalElements;
    
    /**
     * Create a tensor from float array with specified shape
     * @param data Float array containing tensor data
     * @param shape Shape of the tensor
     */
    public Tensor(@NonNull float[] data, @NonNull int[] shape) {
        if (data == null || shape == null) {
            throw new IllegalArgumentException("Data and shape cannot be null");
        }
        
        this.totalElements = calculateTotalElements(shape);
        if (data.length != totalElements) {
            throw new IllegalArgumentException(
                String.format("Data length (%d) doesn't match shape %s (total elements: %d)",
                    data.length, Arrays.toString(shape), totalElements)
            );
        }
        
        this.data = data.clone();
        this.shape = shape.clone();
    }
    
    /**
     * Create a tensor filled with zeros
     * @param shape Shape of the tensor
     * @return Zero-filled tensor
     */
    public static Tensor zeros(@NonNull int[] shape) {
        int totalElements = calculateTotalElements(shape);
        float[] data = new float[totalElements];
        return new Tensor(data, shape);
    }
    
    /**
     * Create a tensor filled with ones
     * @param shape Shape of the tensor
     * @return One-filled tensor
     */
    public static Tensor ones(@NonNull int[] shape) {
        int totalElements = calculateTotalElements(shape);
        float[] data = new float[totalElements];
        Arrays.fill(data, 1.0f);
        return new Tensor(data, shape);
    }
    
    /**
     * Create a tensor filled with a constant value
     * @param shape Shape of the tensor
     * @param value Fill value
     * @return Constant-filled tensor
     */
    public static Tensor constant(@NonNull int[] shape, float value) {
        int totalElements = calculateTotalElements(shape);
        float[] data = new float[totalElements];
        Arrays.fill(data, value);
        return new Tensor(data, shape);
    }
    
    /**
     * Create a tensor from ByteBuffer
     * @param buffer ByteBuffer containing float data
     * @param shape Shape of the tensor
     * @return Tensor created from buffer
     */
    public static Tensor fromByteBuffer(@NonNull ByteBuffer buffer, @NonNull int[] shape) {
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        int totalElements = calculateTotalElements(shape);
        float[] data = new float[totalElements];
        
        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        floatBuffer.get(data);
        
        return new Tensor(data, shape);
    }
    
    /**
     * Create tensor from 2D array
     * @param array 2D float array
     * @return Tensor representation
     */
    public static Tensor fromArray2D(@NonNull float[][] array) {
        if (array.length == 0 || array[0].length == 0) {
            throw new IllegalArgumentException("Array cannot be empty");
        }
        
        int rows = array.length;
        int cols = array[0].length;
        float[] data = new float[rows * cols];
        
        for (int i = 0; i < rows; i++) {
            if (array[i].length != cols) {
                throw new IllegalArgumentException("All rows must have the same length");
            }
            System.arraycopy(array[i], 0, data, i * cols, cols);
        }
        
        return new Tensor(data, new int[]{rows, cols});
    }
    
    /**
     * Get tensor data as float array
     * @return Float array containing tensor data
     */
    @NonNull
    public float[] getData() {
        return data.clone();
    }
    
    /**
     * Get tensor shape
     * @return Array representing tensor dimensions
     */
    @NonNull
    public int[] getShape() {
        return shape.clone();
    }
    
    /**
     * Get total number of elements
     * @return Total elements in tensor
     */
    public int getTotalElements() {
        return totalElements;
    }
    
    /**
     * Get number of dimensions
     * @return Number of dimensions
     */
    public int getRank() {
        return shape.length;
    }
    
    /**
     * Get element at specified index
     * @param indices Indices for each dimension
     * @return Element value
     */
    public float getElement(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(
                "Number of indices must match tensor rank: " + shape.length
            );
        }
        
        int flatIndex = 0;
        int stride = 1;
        
        for (int i = shape.length - 1; i >= 0; i--) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException(
                    String.format("Index %d out of bounds for dimension %d (size: %d)",
                        indices[i], i, shape[i])
                );
            }
            flatIndex += indices[i] * stride;
            stride *= shape[i];
        }
        
        return data[flatIndex];
    }
    
    /**
     * Set element at specified index
     * @param value Value to set
     * @param indices Indices for each dimension
     */
    public void setElement(float value, int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(
                "Number of indices must match tensor rank: " + shape.length
            );
        }
        
        int flatIndex = 0;
        int stride = 1;
        
        for (int i = shape.length - 1; i >= 0; i--) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException(
                    String.format("Index %d out of bounds for dimension %d (size: %d)",
                        indices[i], i, shape[i])
                );
            }
            flatIndex += indices[i] * stride;
            stride *= shape[i];
        }
        
        data[flatIndex] = value;
    }
    
    /**
     * Reshape tensor to new shape
     * @param newShape New shape for the tensor
     * @return Reshaped tensor (new instance)
     */
    @NonNull
    public Tensor reshape(@NonNull int[] newShape) {
        int newTotalElements = calculateTotalElements(newShape);
        if (newTotalElements != totalElements) {
            throw new IllegalArgumentException(
                String.format("Cannot reshape tensor of size %d to shape %s (size: %d)",
                    totalElements, Arrays.toString(newShape), newTotalElements)
            );
        }
        
        return new Tensor(data.clone(), newShape);
    }
    
    /**
     * Flatten tensor to 1D
     * @return Flattened tensor
     */
    @NonNull
    public Tensor flatten() {
        return reshape(new int[]{totalElements});
    }
    
    /**
     * Convert tensor to ByteBuffer
     * @return ByteBuffer containing tensor data
     */
    @NonNull
    public ByteBuffer toByteBuffer() {
        ByteBuffer buffer = ByteBuffer.allocate(totalElements * 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        floatBuffer.put(data);
        
        buffer.rewind();
        return buffer;
    }
    
    /**
     * Apply softmax to tensor (assumes last dimension)
     * @return New tensor with softmax applied
     */
    @NonNull
    public Tensor softmax() {
        if (shape.length == 0) {
            throw new IllegalStateException("Cannot apply softmax to scalar");
        }
        
        float[] result = new float[totalElements];
        int lastDimSize = shape[shape.length - 1];
        int numBatches = totalElements / lastDimSize;
        
        for (int batch = 0; batch < numBatches; batch++) {
            int offset = batch * lastDimSize;
            
            // Find max for numerical stability
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < lastDimSize; i++) {
                maxVal = Math.max(maxVal, data[offset + i]);
            }
            
            // Compute exp and sum
            float sum = 0;
            for (int i = 0; i < lastDimSize; i++) {
                result[offset + i] = (float) Math.exp(data[offset + i] - maxVal);
                sum += result[offset + i];
            }
            
            // Normalize
            for (int i = 0; i < lastDimSize; i++) {
                result[offset + i] /= sum;
            }
        }
        
        return new Tensor(result, shape);
    }
    
    /**
     * Get argmax along last dimension
     * @return Indices of maximum values
     */
    @NonNull
    public int[] argmax() {
        if (shape.length == 0) {
            return new int[]{0};
        }
        
        int lastDimSize = shape[shape.length - 1];
        int numBatches = totalElements / lastDimSize;
        int[] indices = new int[numBatches];
        
        for (int batch = 0; batch < numBatches; batch++) {
            int offset = batch * lastDimSize;
            float maxVal = Float.NEGATIVE_INFINITY;
            int maxIndex = 0;
            
            for (int i = 0; i < lastDimSize; i++) {
                if (data[offset + i] > maxVal) {
                    maxVal = data[offset + i];
                    maxIndex = i;
                }
            }
            
            indices[batch] = maxIndex;
        }
        
        return indices;
    }
    
    /**
     * Get top K values and indices
     * @param k Number of top values to return
     * @return Pair of values and indices arrays
     */
    @NonNull
    public TopKResult topK(int k) {
        if (shape.length == 0) {
            throw new IllegalStateException("Cannot get top-k from scalar");
        }
        
        int lastDimSize = shape[shape.length - 1];
        k = Math.min(k, lastDimSize);
        
        // For simplicity, only handling 1D case
        if (shape.length == 1) {
            // Create index-value pairs
            IndexedValue[] indexedValues = new IndexedValue[lastDimSize];
            for (int i = 0; i < lastDimSize; i++) {
                indexedValues[i] = new IndexedValue(i, data[i]);
            }
            
            // Sort by value descending
            Arrays.sort(indexedValues, (a, b) -> Float.compare(b.value, a.value));
            
            // Extract top k
            float[] values = new float[k];
            int[] indices = new int[k];
            
            for (int i = 0; i < k; i++) {
                values[i] = indexedValues[i].value;
                indices[i] = indexedValues[i].index;
            }
            
            return new TopKResult(values, indices);
        }
        
        throw new UnsupportedOperationException("Top-k only supported for 1D tensors currently");
    }
    
    /**
     * Slice tensor along specified dimension
     * @param dimension Dimension to slice
     * @param start Start index (inclusive)
     * @param end End index (exclusive)
     * @return Sliced tensor
     */
    @NonNull
    public Tensor slice(int dimension, int start, int end) {
        if (dimension < 0 || dimension >= shape.length) {
            throw new IllegalArgumentException("Invalid dimension: " + dimension);
        }
        
        if (start < 0 || end > shape[dimension] || start >= end) {
            throw new IllegalArgumentException(
                String.format("Invalid slice range [%d, %d) for dimension %d (size: %d)",
                    start, end, dimension, shape[dimension])
            );
        }
        
        // Calculate new shape
        int[] newShape = shape.clone();
        newShape[dimension] = end - start;
        
        // Only handle simple cases for now
        if (shape.length == 1) {
            float[] slicedData = Arrays.copyOfRange(data, start, end);
            return new Tensor(slicedData, newShape);
        }
        
        throw new UnsupportedOperationException("Slicing only supported for 1D tensors currently");
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor(shape=").append(Arrays.toString(shape));
        sb.append(", data=[");
        
        int maxElements = Math.min(10, totalElements);
        for (int i = 0; i < maxElements; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.4f", data[i]));
        }
        
        if (totalElements > maxElements) {
            sb.append(", ...");
        }
        
        sb.append("])");
        return sb.toString();
    }
    
    // Helper methods
    
    private static int calculateTotalElements(int[] shape) {
        int total = 1;
        for (int dim : shape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("Shape dimensions must be positive");
            }
            total *= dim;
        }
        return total;
    }
    
    // Helper classes
    
    private static class IndexedValue {
        final int index;
        final float value;
        
        IndexedValue(int index, float value) {
            this.index = index;
            this.value = value;
        }
    }
    
    /**
     * Result of top-k operation
     */
    public static class TopKResult {
        private final float[] values;
        private final int[] indices;
        
        TopKResult(float[] values, int[] indices) {
            this.values = values;
            this.indices = indices;
        }
        
        public float[] getValues() {
            return values.clone();
        }
        
        public int[] getIndices() {
            return indices.clone();
        }
    }
}