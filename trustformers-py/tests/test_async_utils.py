"""
Tests for async utilities in TrustformeRS
"""

import asyncio
import pytest
import tempfile
import os
from unittest.mock import Mock, patch

from trustformers import (
    AsyncAutoModel, AsyncAutoTokenizer, AsyncPipeline,
    AsyncModelWrapper, AsyncTokenizerWrapper, AsyncBatchProcessor,
    AsyncResourceManager, AsyncStreamProcessor,
    async_download_file, async_cached_path, async_pipeline,
    wrap_model_async, wrap_tokenizer_async, safe_async_call,
    async_wrapper, get_async_executor, cleanup_async_executor,
    AsyncOperationError,
    BertModel, WordPieceTokenizer
)


class TestAsyncWrapper:
    """Test async wrapper decorator."""
    
    def test_async_wrapper_basic(self):
        """Test basic async wrapper functionality."""
        
        def sync_function(x: int) -> int:
            return x * 2
        
        async_function = async_wrapper(sync_function)
        
        async def test():
            result = await async_function(5)
            assert result == 10
        
        asyncio.run(test())
    
    def test_async_wrapper_with_args_kwargs(self):
        """Test async wrapper with args and kwargs."""
        
        def sync_function(a: int, b: int, multiplier: int = 1) -> int:
            return (a + b) * multiplier
        
        async_function = async_wrapper(sync_function)
        
        async def test():
            result = await async_function(2, 3, multiplier=4)
            assert result == 20
        
        asyncio.run(test())


class TestAsyncAutoModel:
    """Test AsyncAutoModel functionality."""
    
    @patch('trustformers.AutoModel.from_pretrained')
    def test_async_model_loading(self, mock_from_pretrained):
        """Test async model loading."""
        mock_model = Mock(spec=BertModel)
        mock_from_pretrained.return_value = mock_model
        
        async def test():
            model = await AsyncAutoModel.from_pretrained("bert-base-uncased")
            assert model == mock_model
            mock_from_pretrained.assert_called_once_with("bert-base-uncased")
        
        asyncio.run(test())


class TestAsyncAutoTokenizer:
    """Test AsyncAutoTokenizer functionality."""
    
    @patch('trustformers.AutoTokenizer.from_pretrained')
    def test_async_tokenizer_loading(self, mock_from_pretrained):
        """Test async tokenizer loading."""
        mock_tokenizer = Mock(spec=WordPieceTokenizer)
        mock_from_pretrained.return_value = mock_tokenizer
        
        async def test():
            tokenizer = await AsyncAutoTokenizer.from_pretrained("bert-base-uncased")
            assert tokenizer == mock_tokenizer
            mock_from_pretrained.assert_called_once_with("bert-base-uncased")
        
        asyncio.run(test())


class TestAsyncPipeline:
    """Test AsyncPipeline functionality."""
    
    @patch('trustformers.async_utils.pipeline')
    def test_async_pipeline_creation(self, mock_pipeline):
        """Test async pipeline creation."""
        mock_sync_pipeline = Mock()
        mock_pipeline.return_value = mock_sync_pipeline
        
        async def test():
            async_pipeline_obj = await AsyncPipeline.create(
                "text-generation",
                model="gpt2"
            )
            assert isinstance(async_pipeline_obj, AsyncPipeline)
            assert async_pipeline_obj._pipeline == mock_sync_pipeline
        
        asyncio.run(test())
    
    @patch('trustformers.async_utils.pipeline')
    def test_async_pipeline_call(self, mock_pipeline):
        """Test async pipeline call."""
        mock_sync_pipeline = Mock()
        mock_sync_pipeline.return_value = {"generated_text": "Hello world"}
        mock_pipeline.return_value = mock_sync_pipeline
        
        async def test():
            async_pipeline_obj = await AsyncPipeline.create(
                "text-generation",
                model="gpt2"
            )
            result = await async_pipeline_obj("Hello")
            assert result == {"generated_text": "Hello world"}
            mock_sync_pipeline.assert_called_once_with("Hello")
        
        asyncio.run(test())


class TestAsyncModelWrapper:
    """Test AsyncModelWrapper functionality."""
    
    def test_async_model_wrapper_forward(self):
        """Test async model wrapper forward pass."""
        mock_model = Mock(spec=BertModel)
        mock_model.forward.return_value = {"last_hidden_state": "tensor_output"}
        
        wrapper = AsyncModelWrapper(mock_model)
        
        async def test():
            result = await wrapper.forward(input_ids=[1, 2, 3])
            assert result == {"last_hidden_state": "tensor_output"}
            mock_model.forward.assert_called_once_with(input_ids=[1, 2, 3])
        
        asyncio.run(test())
    
    def test_async_model_wrapper_generate(self):
        """Test async model wrapper generate."""
        mock_model = Mock()
        mock_model.generate.return_value = "generated_output"
        
        wrapper = AsyncModelWrapper(mock_model)
        
        async def test():
            result = await wrapper.generate(input_ids=[1, 2, 3])
            assert result == "generated_output"
            mock_model.generate.assert_called_once_with(input_ids=[1, 2, 3])
        
        asyncio.run(test())
    
    def test_async_model_wrapper_generate_no_method(self):
        """Test async model wrapper generate when method doesn't exist."""
        mock_model = Mock(spec=BertModel)
        # BertModel doesn't have generate method
        
        wrapper = AsyncModelWrapper(mock_model)
        
        async def test():
            with pytest.raises(AttributeError):
                await wrapper.generate(input_ids=[1, 2, 3])
        
        asyncio.run(test())


class TestAsyncTokenizerWrapper:
    """Test AsyncTokenizerWrapper functionality."""
    
    def test_async_tokenizer_encode(self):
        """Test async tokenizer encode."""
        mock_tokenizer = Mock(spec=WordPieceTokenizer)
        mock_tokenizer.encode.return_value = [1, 2, 3, 4]
        
        wrapper = AsyncTokenizerWrapper(mock_tokenizer)
        
        async def test():
            result = await wrapper.encode("Hello world")
            assert result == [1, 2, 3, 4]
            mock_tokenizer.encode.assert_called_once_with("Hello world")
        
        asyncio.run(test())
    
    def test_async_tokenizer_decode(self):
        """Test async tokenizer decode."""
        mock_tokenizer = Mock(spec=WordPieceTokenizer)
        mock_tokenizer.decode.return_value = "Hello world"
        
        wrapper = AsyncTokenizerWrapper(mock_tokenizer)
        
        async def test():
            result = await wrapper.decode([1, 2, 3, 4])
            assert result == "Hello world"
            mock_tokenizer.decode.assert_called_once_with([1, 2, 3, 4])
        
        asyncio.run(test())
    
    def test_async_tokenizer_batch_encode_plus(self):
        """Test async tokenizer batch encode plus."""
        mock_tokenizer = Mock(spec=WordPieceTokenizer)
        mock_tokenizer.batch_encode_plus.return_value = {
            "input_ids": [[1, 2], [3, 4]],
            "attention_mask": [[1, 1], [1, 1]]
        }
        
        wrapper = AsyncTokenizerWrapper(mock_tokenizer)
        
        async def test():
            result = await wrapper.batch_encode_plus(["Hello", "World"])
            expected = {
                "input_ids": [[1, 2], [3, 4]],
                "attention_mask": [[1, 1], [1, 1]]
            }
            assert result == expected
            mock_tokenizer.batch_encode_plus.assert_called_once_with(["Hello", "World"])
        
        asyncio.run(test())


class TestAsyncBatchProcessor:
    """Test AsyncBatchProcessor functionality."""
    
    def test_batch_processor_simple(self):
        """Test batch processor with simple function."""
        
        def process_batch(batch):
            return [x * 2 for x in batch]
        
        processor = AsyncBatchProcessor(batch_size=2)
        
        async def test():
            items = [1, 2, 3, 4, 5]
            results = await processor.process_batches(items, process_batch)
            assert results == [2, 4, 6, 8, 10]
        
        asyncio.run(test())
    
    def test_batch_processor_with_kwargs(self):
        """Test batch processor with kwargs."""
        
        def process_batch(batch, multiplier=1):
            return [x * multiplier for x in batch]
        
        processor = AsyncBatchProcessor(batch_size=3)
        
        async def test():
            items = [1, 2, 3, 4, 5, 6]
            results = await processor.process_batches(items, process_batch, multiplier=3)
            assert results == [3, 6, 9, 12, 15, 18]
        
        asyncio.run(test())


class TestAsyncResourceManager:
    """Test AsyncResourceManager functionality."""
    
    def test_resource_manager_context(self):
        """Test resource manager as context manager."""
        
        async def test():
            async with AsyncResourceManager() as manager:
                # Test that context manager works
                assert manager is not None
        
        asyncio.run(test())
    
    def test_resource_manager_cleanup(self):
        """Test resource manager cleanup."""
        
        class MockResource:
            def __init__(self):
                self.cleaned = False
                
            def cleanup(self):
                self.cleaned = True
        
        async def test():
            resource = MockResource()
            
            async with AsyncResourceManager() as manager:
                manager.add_resource(resource)
                assert not resource.cleaned
            
            # After context exit, resource should be cleaned
            assert resource.cleaned
        
        asyncio.run(test())


class TestAsyncStreamProcessor:
    """Test AsyncStreamProcessor functionality."""
    
    def test_stream_processor_basic(self):
        """Test basic stream processor functionality."""
        
        def process_item(item):
            return item * 2
        
        async def test():
            processor = AsyncStreamProcessor(process_item)
            await processor.start_processing()
            
            # Add items
            await processor.add_item(5)
            await processor.add_item(10)
            
            # Get results
            result1 = await processor.get_result()
            result2 = await processor.get_result()
            
            assert result1 == 10
            assert result2 == 20
            
            await processor.stop_processing()
        
        asyncio.run(test())


class TestAsyncFileOperations:
    """Test async file operations."""
    
    @patch('trustformers.async_utils.download_file')
    def test_async_download_file(self, mock_download):
        """Test async file download."""
        mock_download.return_value = "/tmp/downloaded_file.txt"
        
        async def test():
            result = await async_download_file("http://example.com/file.txt", "/tmp/local_file.txt")
            assert result == "/tmp/downloaded_file.txt"
            mock_download.assert_called_once_with("http://example.com/file.txt", "/tmp/local_file.txt")
        
        asyncio.run(test())
    
    @patch('trustformers.async_utils.cached_path')
    def test_async_cached_path(self, mock_cached_path):
        """Test async cached path."""
        mock_cached_path.return_value = "/tmp/cached_file.txt"
        
        async def test():
            result = await async_cached_path("http://example.com/file.txt")
            assert result == "/tmp/cached_file.txt"
            mock_cached_path.assert_called_once()
        
        asyncio.run(test())


class TestSafeAsyncCall:
    """Test safe async call functionality."""
    
    def test_safe_async_call_success(self):
        """Test safe async call with successful operation."""
        
        async def success_coro():
            return "success"
        
        async def test():
            result = await safe_async_call(success_coro())
            assert result == "success"
        
        asyncio.run(test())
    
    def test_safe_async_call_timeout(self):
        """Test safe async call with timeout."""
        
        async def slow_coro():
            await asyncio.sleep(2)
            return "success"
        
        async def test():
            with pytest.raises(AsyncOperationError):
                await safe_async_call(slow_coro(), timeout=0.1)
        
        asyncio.run(test())
    
    def test_safe_async_call_timeout_with_default(self):
        """Test safe async call with timeout and default value."""
        
        async def slow_coro():
            await asyncio.sleep(2)
            return "success"
        
        async def test():
            result = await safe_async_call(
                slow_coro(), 
                timeout=0.1, 
                fallback="default"
            )
            assert result == "default"
        
        asyncio.run(test())
    
    def test_safe_async_call_exception(self):
        """Test safe async call with exception."""
        
        async def error_coro():
            raise ValueError("Test error")
        
        async def test():
            with pytest.raises(AsyncOperationError):
                await safe_async_call(error_coro())
        
        asyncio.run(test())
    
    def test_safe_async_call_exception_with_default(self):
        """Test safe async call with exception and default value."""
        
        async def error_coro():
            raise ValueError("Test error")
        
        async def test():
            result = await safe_async_call(
                error_coro(), 
                fallback="default"
            )
            assert result == "default"
        
        asyncio.run(test())


class TestAsyncExecutorManagement:
    """Test async executor management."""
    
    def test_get_async_executor(self):
        """Test getting async executor."""
        executor = get_async_executor()
        assert executor is not None
        
        # Should return same executor on subsequent calls
        executor2 = get_async_executor()
        assert executor is executor2
    
    def test_cleanup_async_executor(self):
        """Test cleanup of async executor."""
        # Get executor first
        executor = get_async_executor()
        assert executor is not None
        
        # Cleanup
        cleanup_async_executor()
        
        # Should create new executor after cleanup
        new_executor = get_async_executor()
        assert new_executor is not executor


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_wrap_model_async(self):
        """Test wrapping model for async operations."""
        mock_model = Mock(spec=BertModel)
        wrapper = wrap_model_async(mock_model)
        
        assert isinstance(wrapper, AsyncModelWrapper)
        assert wrapper.model == mock_model
    
    def test_wrap_tokenizer_async(self):
        """Test wrapping tokenizer for async operations."""
        mock_tokenizer = Mock(spec=WordPieceTokenizer)
        wrapper = wrap_tokenizer_async(mock_tokenizer)
        
        assert isinstance(wrapper, AsyncTokenizerWrapper)
        assert wrapper.tokenizer == mock_tokenizer
    
    @patch('trustformers.async_utils.pipeline')
    def test_async_pipeline_convenience(self, mock_pipeline):
        """Test async pipeline convenience function."""
        mock_sync_pipeline = Mock()
        mock_pipeline.return_value = mock_sync_pipeline
        
        async def test():
            async_pipeline_obj = await async_pipeline("text-generation", model="gpt2")
            assert isinstance(async_pipeline_obj, AsyncPipeline)
            assert async_pipeline_obj._pipeline == mock_sync_pipeline
        
        asyncio.run(test())


if __name__ == "__main__":
    # Run a simple test to verify functionality
    async def quick_test():
        print("Testing async wrapper...")
        
        def sync_func(x):
            return x * 2
        
        async_func = async_wrapper(sync_func)
        result = await async_func(5)
        print(f"Async wrapper result: {result}")
        assert result == 10
        
        print("✅ Async utilities working correctly!")
    
    asyncio.run(quick_test())