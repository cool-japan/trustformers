# frozen_string_literal: true

module TrustFormeRS
  # Tokenizer for converting text to tokens and vice versa
  class Tokenizer
    attr_reader :client, :vocab_size

    # Tokenization result structure
    TokenizationResult = Struct.new(:token_ids, :tokens, :attention_mask, :special_tokens_mask) do
      def initialize(token_ids:, tokens: nil, attention_mask: nil, special_tokens_mask: nil)
        super(token_ids, tokens, attention_mask, special_tokens_mask)
      end
    end

    # Batch tokenization result structure
    BatchTokenizationResult = Struct.new(:token_ids, :attention_masks, :max_length) do
      def initialize(token_ids:, attention_masks: nil)
        max_len = token_ids.map(&:size).max || 0
        super(token_ids, attention_masks, max_len)
      end
    end

    # Model input structure for prepared tokenization
    ModelInput = Struct.new(:token_ids, :attention_mask, :original_length) do
      def initialize(token_ids:, attention_mask:, original_length:)
        super(token_ids, attention_mask, original_length)
      end
    end

    # Batch model input structure
    BatchModelInput = Struct.new(:inputs, :batch_size) do
      def initialize(inputs:)
        super(inputs, inputs.size)
      end

      def token_ids
        inputs.map(&:token_ids)
      end

      def attention_masks
        inputs.map(&:attention_mask)
      end
    end

    # Padding strategies
    module PaddingStrategy
      NONE = :none
      MAX_LENGTH = :max_length
    end

    # Initialize tokenizer
    # @param client [Client] TrustFormeRS client
    # @param tokenizer_id [String] Tokenizer identifier or path
    # @param options [Hash] Tokenizer options
    def initialize(client, tokenizer_id, options = {})
      @client = client
      @tokenizer_id = tokenizer_id
      @options = options

      error = Native::TrustformersError.new
      @handle = Native.tokenizer_load(client.native_handle, tokenizer_id, error)
      
      if @handle.null?
        Native.check_error!(error)
        raise TokenizationError, "Failed to load tokenizer: #{tokenizer_id}"
      end

      @vocab_size = Native.tokenizer_get_vocab_size(@handle)
    end

    # Clean up native resources
    def finalize
      return unless @handle && !@handle.null?
      
      Native.tokenizer_free(@handle)
      @handle = nil
    end

    # Get vocabulary as array of tokens
    # @return [Array<String>] Vocabulary tokens
    def vocabulary
      @vocabulary ||= begin
        size_ptr = FFI::MemoryPointer.new(:int)
        vocab_ptr = Native.tokenizer_get_vocab(@handle, size_ptr)
        size = size_ptr.read_int

        return [] if vocab_ptr.null?

        vocab = []
        (0...size).each do |i|
          token_ptr = vocab_ptr[i].read_pointer
          vocab << token_ptr.read_string unless token_ptr.null?
        end

        Native.trustformers_free_string_array(vocab_ptr, size)
        vocab
      end
    end

    # Encode text to token IDs
    # @param text [String] Input text to tokenize
    # @param add_special_tokens [Boolean] Whether to add special tokens
    # @param return_tokens [Boolean] Whether to return token strings
    # @return [TokenizationResult] Tokenization result
    def encode(text, add_special_tokens: true, return_tokens: false)
      length_ptr = FFI::MemoryPointer.new(:int)
      error = Native::TrustformersError.new

      token_ids_ptr = Native.tokenizer_encode(@handle, text, length_ptr, error)
      
      if token_ids_ptr.null?
        Native.check_error!(error)
        raise TokenizationError, "Failed to encode text"
      end

      length = length_ptr.read_int
      token_ids = token_ids_ptr.read_array_of_int32(length)

      # Free the allocated memory
      token_ids_ptr.free

      tokens = return_tokens ? token_ids.map { |id| id_to_token(id) } : nil

      TokenizationResult.new(
        token_ids: token_ids,
        tokens: tokens
      )
    end

    # Encode multiple texts
    # @param texts [Array<String>] Array of input texts
    # @param padding [Symbol] Padding strategy (:none, :max_length)
    # @param truncation [Boolean] Whether to truncate sequences
    # @param max_length [Integer, nil] Maximum sequence length
    # @param add_special_tokens [Boolean] Whether to add special tokens
    # @return [BatchTokenizationResult] Batch tokenization result
    def encode_batch(texts, padding: :none, truncation: true, max_length: nil, add_special_tokens: true)
      all_token_ids = texts.map do |text|
        result = encode(text, add_special_tokens: add_special_tokens)
        token_ids = result.token_ids

        # Apply truncation if needed
        if max_length && truncation && token_ids.size > max_length
          token_ids = token_ids[0, max_length]
        end

        token_ids
      end

      # Apply padding if needed
      if padding == :max_length
        target_length = max_length || all_token_ids.map(&:size).max
        
        all_token_ids.each do |token_ids|
          while token_ids.size < target_length
            token_ids << 0 # Pad with 0 (typically pad token)
          end
        end

        # Create attention masks
        attention_masks = all_token_ids.map do |token_ids|
          token_ids.map { |id| id == 0 ? 0 : 1 }
        end

        return BatchTokenizationResult.new(
          token_ids: all_token_ids,
          attention_masks: attention_masks
        )
      end

      BatchTokenizationResult.new(token_ids: all_token_ids)
    end

    # Decode token IDs to text
    # @param token_ids [Array<Integer>] Token IDs to decode
    # @param skip_special_tokens [Boolean] Whether to skip special tokens
    # @return [String] Decoded text
    def decode(token_ids, skip_special_tokens: true)
      error = Native::TrustformersError.new
      
      # Convert Ruby array to C array
      token_ids_ptr = FFI::MemoryPointer.new(:int32, token_ids.size)
      token_ids_ptr.write_array_of_int32(token_ids)

      text_ptr = Native.tokenizer_decode(@handle, token_ids_ptr, token_ids.size, error)
      
      if text_ptr.null?
        Native.check_error!(error)
        raise TokenizationError, "Failed to decode token IDs"
      end

      text = text_ptr.read_string
      Native.trustformers_free_string(text_ptr)
      
      text
    end

    # Decode multiple sequences
    # @param token_ids_batch [Array<Array<Integer>>] Array of token ID sequences
    # @param skip_special_tokens [Boolean] Whether to skip special tokens
    # @return [Array<String>] Array of decoded texts
    def decode_batch(token_ids_batch, skip_special_tokens: true)
      token_ids_batch.map { |token_ids| decode(token_ids, skip_special_tokens: skip_special_tokens) }
    end

    # Get token string for specific ID
    # @param token_id [Integer] Token ID
    # @return [String, nil] Token string or nil if invalid
    def id_to_token(token_id)
      return nil if token_id < 0 || token_id >= vocab_size
      vocabulary[token_id]
    end

    # Get ID for specific token
    # @param token [String] Token string
    # @return [Integer, nil] Token ID or nil if not found
    def token_to_id(token)
      vocabulary.index(token)
    end

    # Convert texts to model inputs with proper padding and attention masks
    # @param text [String] Input text
    # @param max_length [Integer] Maximum sequence length
    # @param padding [Symbol] Padding strategy
    # @param truncation [Boolean] Whether to truncate
    # @return [ModelInput] Model-ready input
    def prepare_for_model(text, max_length: 512, padding: :max_length, truncation: true)
      result = encode(text)
      token_ids = result.token_ids

      # Truncate if necessary
      original_length = token_ids.size
      if truncation && token_ids.size > max_length
        token_ids = token_ids[0, max_length]
      end

      # Apply padding
      case padding
      when :max_length
        while token_ids.size < max_length
          token_ids << 0 # Pad token
        end
      when :none
        # No padding
      else
        raise ArgumentError, "Unsupported padding strategy: #{padding}"
      end

      # Create attention mask
      attention_mask = token_ids.map.with_index { |_, i| i < original_length ? 1 : 0 }

      ModelInput.new(
        token_ids: token_ids,
        attention_mask: attention_mask,
        original_length: [original_length, max_length].min
      )
    end

    # Convert multiple texts to batch model inputs
    # @param texts [Array<String>] Input texts
    # @param max_length [Integer] Maximum sequence length
    # @param padding [Symbol] Padding strategy
    # @param truncation [Boolean] Whether to truncate
    # @return [BatchModelInput] Batch model input
    def prepare_for_model_batch(texts, max_length: 512, padding: :max_length, truncation: true)
      inputs = texts.map do |text|
        prepare_for_model(text, max_length: max_length, padding: padding, truncation: truncation)
      end

      BatchModelInput.new(inputs: inputs)
    end

    # Get tokenizer configuration and information
    # @return [Hash] Tokenizer information
    def info
      {
        tokenizer_id: @tokenizer_id,
        vocab_size: @vocab_size,
        model_max_length: @options[:model_max_length] || 512,
        padding_side: @options[:padding_side] || "right",
        truncation_side: @options[:truncation_side] || "right",
        pad_token: @options[:pad_token] || "[PAD]",
        unk_token: @options[:unk_token] || "[UNK]",
        cls_token: @options[:cls_token] || "[CLS]",
        sep_token: @options[:sep_token] || "[SEP]",
        mask_token: @options[:mask_token] || "[MASK]"
      }
    end

    # Get special token IDs
    # @return [Hash] Special token ID mappings
    def special_tokens
      info.transform_values { |token| token_to_id(token) if token.is_a?(String) }.compact
    end

    # Check if tokenizer is fast (uses Rust tokenizers library)
    # @return [Boolean] True for fast tokenizers
    def fast?
      true # TrustFormeRS tokenizers are always fast
    end

    # String representation
    # @return [String] Human-readable tokenizer information
    def to_s
      "Tokenizer(id: #{@tokenizer_id}, vocab_size: #{@vocab_size})"
    end

    # Detailed inspection
    # @return [String] Detailed tokenizer information
    def inspect
      info_str = info.map { |k, v| "#{k}: #{v.inspect}" }.join(", ")
      "#{to_s}\nInfo: {#{info_str}}"
    end

    private

    # Get native handle for FFI calls
    def native_handle
      @handle
    end
  end
end