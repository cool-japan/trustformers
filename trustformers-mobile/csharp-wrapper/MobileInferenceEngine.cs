using System;
using System.Runtime.InteropServices;
using TrustformersMobile.Native;

namespace TrustformersMobile
{
    /// <summary>
    /// TrustformeRS Mobile inference engine for running ML models on mobile devices
    /// </summary>
    public class MobileInferenceEngine : IDisposable
    {
        private UIntPtr _handle;
        private bool _disposed = false;

        /// <summary>
        /// Creates a new inference engine with the specified configuration and model
        /// </summary>
        /// <param name="config">Mobile configuration</param>
        /// <param name="modelPath">Path to the model file</param>
        /// <exception cref="ArgumentNullException">Thrown if config or modelPath is null</exception>
        /// <exception cref="TrustformersMobileException">Thrown if engine creation fails</exception>
        public MobileInferenceEngine(MobileConfig config, string modelPath)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));
            if (string.IsNullOrEmpty(modelPath))
                throw new ArgumentNullException(nameof(modelPath));

            // Validate configuration before creating engine
            config.Validate();

            var error = TrustformersMobileNative.trustformers_mobile_engine_create(
                config.Handle, 
                modelPath, 
                out _handle);
            
            ThrowIfError(error);
        }

        /// <summary>
        /// Runs inference with float32 input data
        /// </summary>
        /// <param name="inputData">Input data as float array</param>
        /// <returns>Output data as float array</returns>
        /// <exception cref="ArgumentNullException">Thrown if inputData is null</exception>
        /// <exception cref="TrustformersMobileException">Thrown if inference fails</exception>
        public unsafe float[] InferenceF32(float[] inputData)
        {
            if (inputData == null)
                throw new ArgumentNullException(nameof(inputData));
            
            ThrowIfDisposed();

            // Estimate output size (this could be made more sophisticated)
            var estimatedOutputSize = inputData.Length * 2; // Conservative estimate
            var outputData = new float[estimatedOutputSize];

            fixed (float* inputPtr = inputData)
            fixed (float* outputPtr = outputData)
            {
                var error = TrustformersMobileNative.trustformers_mobile_engine_inference_f32(
                    _handle,
                    (IntPtr)inputPtr,
                    (UIntPtr)inputData.Length,
                    (IntPtr)outputPtr,
                    (UIntPtr)outputData.Length,
                    out UIntPtr actualOutputSize);

                ThrowIfError(error);

                // Return only the actual output data
                var actualSize = (int)actualOutputSize;
                if (actualSize <= outputData.Length)
                {
                    Array.Resize(ref outputData, actualSize);
                    return outputData;
                }
                else
                {
                    // Need to reallocate with larger buffer
                    var largerOutput = new float[actualSize];
                    fixed (float* largerOutputPtr = largerOutput)
                    {
                        error = TrustformersMobileNative.trustformers_mobile_engine_inference_f32(
                            _handle,
                            (IntPtr)inputPtr,
                            (UIntPtr)inputData.Length,
                            (IntPtr)largerOutputPtr,
                            (UIntPtr)largerOutput.Length,
                            out actualOutputSize);

                        ThrowIfError(error);
                        return largerOutput;
                    }
                }
            }
        }

        /// <summary>
        /// Runs inference with float32 input data and writes output to provided buffer
        /// </summary>
        /// <param name="inputData">Input data as float array</param>
        /// <param name="outputData">Output buffer to write results to</param>
        /// <returns>Actual number of output elements written</returns>
        /// <exception cref="ArgumentNullException">Thrown if inputData or outputData is null</exception>
        /// <exception cref="TrustformersMobileException">Thrown if inference fails</exception>
        public unsafe int InferenceF32(float[] inputData, float[] outputData)
        {
            if (inputData == null)
                throw new ArgumentNullException(nameof(inputData));
            if (outputData == null)
                throw new ArgumentNullException(nameof(outputData));
            
            ThrowIfDisposed();

            fixed (float* inputPtr = inputData)
            fixed (float* outputPtr = outputData)
            {
                var error = TrustformersMobileNative.trustformers_mobile_engine_inference_f32(
                    _handle,
                    (IntPtr)inputPtr,
                    (UIntPtr)inputData.Length,
                    (IntPtr)outputPtr,
                    (UIntPtr)outputData.Length,
                    out UIntPtr actualOutputSize);

                ThrowIfError(error);
                return (int)actualOutputSize;
            }
        }

        /// <summary>
        /// Runs inference with float32 input data using Span&lt;T&gt; for zero-copy operations
        /// </summary>
        /// <param name="inputData">Input data as ReadOnlySpan&lt;float&gt;</param>
        /// <param name="outputData">Output buffer as Span&lt;float&gt;</param>
        /// <returns>Actual number of output elements written</returns>
        /// <exception cref="TrustformersMobileException">Thrown if inference fails</exception>
        public unsafe int InferenceF32(ReadOnlySpan<float> inputData, Span<float> outputData)
        {
            ThrowIfDisposed();

            fixed (float* inputPtr = inputData)
            fixed (float* outputPtr = outputData)
            {
                var error = TrustformersMobileNative.trustformers_mobile_engine_inference_f32(
                    _handle,
                    (IntPtr)inputPtr,
                    (UIntPtr)inputData.Length,
                    (IntPtr)outputPtr,
                    (UIntPtr)outputData.Length,
                    out UIntPtr actualOutputSize);

                ThrowIfError(error);
                return (int)actualOutputSize;
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(MobileInferenceEngine));
        }

        private static void ThrowIfError(TrustformersMobileError error)
        {
            if (error != TrustformersMobileError.Success)
                throw new TrustformersMobileException(error);
        }

        /// <summary>
        /// Disposes the inference engine and frees native resources
        /// </summary>
        public void Dispose()
        {
            if (!_disposed && _handle != UIntPtr.Zero)
            {
                TrustformersMobileNative.trustformers_mobile_engine_free(_handle);
                _handle = UIntPtr.Zero;
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Finalizer
        /// </summary>
        ~MobileInferenceEngine()
        {
            Dispose();
        }
    }
}