using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;

namespace TrustformeRS
{
    /// <summary>
    /// High-level wrapper for TrustformersRS tokenizer operations
    /// </summary>
    public class Tokenizer : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed = false;

        /// <summary>
        /// Gets the native handle for this tokenizer
        /// </summary>
        public IntPtr Handle => _handle;

        /// <summary>
        /// Gets whether the tokenizer is loaded and ready for use
        /// </summary>
        public bool IsLoaded => _handle != IntPtr.Zero;

        /// <summary>
        /// Create a tokenizer from the specified path
        /// </summary>
        /// <param name="tokenizerPath">Path to the tokenizer file</param>
        /// <returns>Loaded tokenizer instance</returns>
        public static Tokenizer Create(string tokenizerPath)
        {
            if (string.IsNullOrEmpty(tokenizerPath))
                throw new ArgumentException("Tokenizer path cannot be null or empty", nameof(tokenizerPath));

            var tokenizer = new Tokenizer();
            var error = TrustformeRS.trustformers_create_tokenizer(tokenizerPath, out tokenizer._handle);
            TrustformeRS.ThrowIfError(error, $"Failed to create tokenizer from {tokenizerPath}");

            return tokenizer;
        }

        private Tokenizer()
        {
            _handle = IntPtr.Zero;
        }

        /// <summary>
        /// Encode text into token IDs
        /// </summary>
        /// <param name="text">Text to encode</param>
        /// <returns>Array of token IDs</returns>
        public int[] Encode(string text)
        {
            ThrowIfNotLoaded();
            
            if (string.IsNullOrEmpty(text))
                return new int[0];

            var error = TrustformeRS.trustformers_tokenizer_encode(_handle, text, out IntPtr tokenIdsPtr, out int numTokens);
            TrustformeRS.ThrowIfError(error, "Failed to encode text");

            if (tokenIdsPtr == IntPtr.Zero || numTokens <= 0)
                return new int[0];

            try
            {
                // Copy token IDs from unmanaged memory
                var tokenIds = new int[numTokens];
                Marshal.Copy(tokenIdsPtr, tokenIds, 0, numTokens);
                return tokenIds;
            }
            finally
            {
                // Free the unmanaged memory
                if (tokenIdsPtr != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(tokenIdsPtr);
                }
            }
        }

        /// <summary>
        /// Encode multiple texts into token IDs
        /// </summary>
        /// <param name="texts">Texts to encode</param>
        /// <returns>Array of token ID arrays</returns>
        public int[][] EncodeBatch(string[] texts)
        {
            if (texts == null)
                throw new ArgumentNullException(nameof(texts));

            var results = new int[texts.Length][];
            for (int i = 0; i < texts.Length; i++)
            {
                results[i] = Encode(texts[i]);
            }
            return results;
        }

        /// <summary>
        /// Encode multiple texts into token IDs
        /// </summary>
        /// <param name="texts">Texts to encode</param>
        /// <returns>List of token ID arrays</returns>
        public List<int[]> EncodeBatch(IEnumerable<string> texts)
        {
            if (texts == null)
                throw new ArgumentNullException(nameof(texts));

            return texts.Select(Encode).ToList();
        }

        /// <summary>
        /// Decode token IDs back into text
        /// </summary>
        /// <param name="tokenIds">Token IDs to decode</param>
        /// <returns>Decoded text</returns>
        public string Decode(int[] tokenIds)
        {
            ThrowIfNotLoaded();
            
            if (tokenIds == null || tokenIds.Length == 0)
                return string.Empty;

            // Copy token IDs to unmanaged memory
            IntPtr tokenIdsPtr = Marshal.AllocHGlobal(tokenIds.Length * sizeof(int));
            try
            {
                Marshal.Copy(tokenIds, 0, tokenIdsPtr, tokenIds.Length);

                var error = TrustformeRS.trustformers_tokenizer_decode(_handle, tokenIdsPtr, tokenIds.Length, out IntPtr decodedTextPtr);
                TrustformeRS.ThrowIfError(error, "Failed to decode token IDs");

                return TrustformeRS.PtrToStringAndFree(decodedTextPtr) ?? string.Empty;
            }
            finally
            {
                Marshal.FreeHGlobal(tokenIdsPtr);
            }
        }

        /// <summary>
        /// Decode token IDs back into text
        /// </summary>
        /// <param name="tokenIds">Token IDs to decode</param>
        /// <returns>Decoded text</returns>
        public string Decode(IEnumerable<int> tokenIds)
        {
            if (tokenIds == null)
                throw new ArgumentNullException(nameof(tokenIds));

            return Decode(tokenIds.ToArray());
        }

        /// <summary>
        /// Decode multiple token ID arrays back into texts
        /// </summary>
        /// <param name="tokenIdArrays">Arrays of token IDs to decode</param>
        /// <returns>Array of decoded texts</returns>
        public string[] DecodeBatch(int[][] tokenIdArrays)
        {
            if (tokenIdArrays == null)
                throw new ArgumentNullException(nameof(tokenIdArrays));

            var results = new string[tokenIdArrays.Length];
            for (int i = 0; i < tokenIdArrays.Length; i++)
            {
                results[i] = Decode(tokenIdArrays[i]);
            }
            return results;
        }

        /// <summary>
        /// Decode multiple token ID arrays back into texts
        /// </summary>
        /// <param name="tokenIdArrays">Arrays of token IDs to decode</param>
        /// <returns>List of decoded texts</returns>
        public List<string> DecodeBatch(IEnumerable<int[]> tokenIdArrays)
        {
            if (tokenIdArrays == null)
                throw new ArgumentNullException(nameof(tokenIdArrays));

            return tokenIdArrays.Select(Decode).ToList();
        }

        /// <summary>
        /// Get the length of encoded text (number of tokens)
        /// </summary>
        /// <param name="text">Text to measure</param>
        /// <returns>Number of tokens</returns>
        public int GetTokenCount(string text)
        {
            var tokens = Encode(text);
            return tokens.Length;
        }

        /// <summary>
        /// Check if text fits within the given token limit
        /// </summary>
        /// <param name="text">Text to check</param>
        /// <param name="maxTokens">Maximum number of tokens allowed</param>
        /// <returns>True if text fits within limit</returns>
        public bool FitsWithinLimit(string text, int maxTokens)
        {
            return GetTokenCount(text) <= maxTokens;
        }

        /// <summary>
        /// Truncate text to fit within the given token limit
        /// </summary>
        /// <param name="text">Text to truncate</param>
        /// <param name="maxTokens">Maximum number of tokens</param>
        /// <returns>Truncated text</returns>
        public string Truncate(string text, int maxTokens)
        {
            if (string.IsNullOrEmpty(text) || maxTokens <= 0)
                return string.Empty;

            var tokens = Encode(text);
            if (tokens.Length <= maxTokens)
                return text;

            var truncatedTokens = new int[maxTokens];
            Array.Copy(tokens, truncatedTokens, maxTokens);
            return Decode(truncatedTokens);
        }

        /// <summary>
        /// Split text into chunks that fit within the token limit
        /// </summary>
        /// <param name="text">Text to split</param>
        /// <param name="maxTokensPerChunk">Maximum tokens per chunk</param>
        /// <param name="overlapTokens">Number of overlapping tokens between chunks</param>
        /// <returns>List of text chunks</returns>
        public List<string> SplitIntoChunks(string text, int maxTokensPerChunk, int overlapTokens = 0)
        {
            if (string.IsNullOrEmpty(text) || maxTokensPerChunk <= 0)
                return new List<string>();

            if (overlapTokens < 0)
                overlapTokens = 0;

            if (overlapTokens >= maxTokensPerChunk)
                throw new ArgumentException("Overlap tokens must be less than max tokens per chunk");

            var tokens = Encode(text);
            var chunks = new List<string>();

            if (tokens.Length <= maxTokensPerChunk)
            {
                chunks.Add(text);
                return chunks;
            }

            int start = 0;
            while (start < tokens.Length)
            {
                int end = Math.Min(start + maxTokensPerChunk, tokens.Length);
                var chunkTokens = new int[end - start];
                Array.Copy(tokens, start, chunkTokens, 0, end - start);
                
                string chunk = Decode(chunkTokens);
                chunks.Add(chunk);

                // Move start position, accounting for overlap
                start = end - overlapTokens;
                if (start >= tokens.Length)
                    break;
            }

            return chunks;
        }

        /// <summary>
        /// Destroy the tokenizer and free resources
        /// </summary>
        public void Destroy()
        {
            if (_handle != IntPtr.Zero)
            {
                var error = TrustformeRS.trustformers_destroy_tokenizer(_handle);
                _handle = IntPtr.Zero;
                
                // Don't throw on destroy errors during cleanup
                if (!TrustformeRS.IsSuccess(error))
                {
                    Console.WriteLine($"Warning: Error destroying tokenizer: {error}");
                }
            }
        }

        /// <summary>
        /// Dispose of the tokenizer and free resources
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                Destroy();
                _disposed = true;
            }
        }

        ~Tokenizer()
        {
            Dispose(false);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Tokenizer));
        }

        private void ThrowIfNotLoaded()
        {
            ThrowIfDisposed();
            if (!IsLoaded)
                throw new InvalidOperationException("Tokenizer is not loaded");
        }

        /// <summary>
        /// Create a string representation of the tokenizer
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            if (_disposed)
                return "Tokenizer [Disposed]";
            
            if (!IsLoaded)
                return "Tokenizer [Not Loaded]";

            return "Tokenizer [Loaded]";
        }
    }

    /// <summary>
    /// Tokenization result with additional metadata
    /// </summary>
    public class TokenizationResult
    {
        /// <summary>
        /// The original input text
        /// </summary>
        public string InputText { get; set; }

        /// <summary>
        /// The encoded token IDs
        /// </summary>
        public int[] TokenIds { get; set; }

        /// <summary>
        /// The decoded text (should match input text in most cases)
        /// </summary>
        public string DecodedText { get; set; }

        /// <summary>
        /// Number of tokens
        /// </summary>
        public int TokenCount => TokenIds?.Length ?? 0;

        /// <summary>
        /// Whether the tokenization was successful
        /// </summary>
        public bool IsSuccessful { get; set; }

        /// <summary>
        /// Any error message if tokenization failed
        /// </summary>
        public string ErrorMessage { get; set; }

        /// <summary>
        /// Time taken for tokenization in milliseconds
        /// </summary>
        public double ProcessingTimeMs { get; set; }
    }
}