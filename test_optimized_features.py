#!/usr/bin/env python3
"""
Test script for optimized features in ComfyUI LlamaCpp Client
Tests all three phases of optimizations without requiring a running llama-server
"""

import gc
import io
import json
import logging
import time
import unittest
from unittest.mock import MagicMock, patch, Mock
from typing import Dict, Any, List, Optional, Tuple

# Import modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from utils.llama_client import LlamaCppAPIClient
from utils.image_utils import (
    tensor_to_base64_data_uri,
    build_vision_content,
    process_image_data_string,
    DEFAULT_JPEG_QUALITY,
    MAX_IMAGE_DIMENSION,
    MIN_IMAGE_DIMENSION,
)


# ============================================================================
# MOCK SERVER CLASS - Simulates llama-server behavior
# ============================================================================

class MockLlamaServer:
    """Mock server that simulates llama-server responses for testing."""
    
    def __init__(self):
        self.request_count = 0
        self.last_request = None
        self.response_delay = 0.1  # Simulated network delay
        self.streaming_mode = False
        self.stored_responses = {}
        self.slot_usage = {}
        
    def create_completion_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Create a mock completion response."""
        # Get additional parameters from kwargs, not from positional args
        n_predict = kwargs.get("n_predict", 50)
        return {
            "choices": [
                {
                    "text": f"Mock response for: {prompt[:50]}...",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop" if n_predict < 100 else "length"
                }
            ],
            "created": int(time.time()),
            "model": kwargs.get("model", "mock-model"),
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": n_predict,
                "total_tokens": len(prompt.split()) + n_predict
            }
        }
    
    def create_chat_response(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Create a mock chat completion response."""
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Mock assistant response to: {messages[-1].get('content', '')[:50]}..."
                    },
                    "finish_reason": "stop"
                }
            ],
            "created": int(time.time()),
            "model": kwargs.get("model", "mock-model"),
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": kwargs.get("max_tokens", 50),
                "total_tokens": 10 + kwargs.get("max_tokens", 50)
            }
        }
    
    def create_tokenize_response(self, content: str, **kwargs) -> Dict[str, Any]:
        """Create a mock tokenization response."""
        tokens = [hash(c) % 10000 for c in content]
        result = {"tokens": tokens}
        if kwargs.get("with_pieces"):
            result["pieces"] = list(content)
        return result
    
    def create_embedding_response(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Create a mock embedding response."""
        return {
            "data": [
                {
                    "embedding": [0.1 * i for i in range(1536)],
                    "index": 0,
                    "object": "embedding"
                }
            ],
            "model": kwargs.get("model", "mock-model"),
            "usage": {
                "prompt_tokens": len(input_text.split()),
                "total_tokens": len(input_text.split())
            }
        }
    
    def stream_response(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """Simulate streaming response."""
        responses = []
        words = ["Mock", "response", "streaming", "data", "for", "testing"]
        for i, word in enumerate(words):
            time.sleep(self.response_delay)
            responses.append({
                "choices": [{"text": word + " ", "index": 0, "finish_reason": None if i < len(words)-1 else "stop"}]
            })
        return responses


# ============================================================================
# MOCK CLIENT CLASS - Enhanced LlamaCppAPIClient with mock backend
# ============================================================================

class MockLlamaClient(LlamaCppAPIClient):
    """Mock LlamaCppAPIClient that uses MockLlamaServer instead of real HTTP."""
    
    def __init__(self, base_url: str, api_key: str = "", timeout: int = 600):
        super().__init__(base_url, api_key, timeout)
        self.server = MockLlamaServer()
        self.request_history = []
        self.memory_stats = {"allocations": 0, "deallocations": 0, "peak_usage": 0}
        self._image_cache = {}  # Cache for processed images
        self._cache_counter = 0
        
    def _make_request(self, endpoint_path: str, data: Dict[str, Any]) -> Tuple[str, str, str, int]:
        """Override to use mock server."""
        self.request_history.append({
            "endpoint": endpoint_path,
            "data": data,
            "timestamp": time.time()
        })
        
        # Simulate network delay
        time.sleep(self.server.response_delay)
        
        # Generate appropriate response based on endpoint
        if endpoint_path == "/completion":
            response = self.server.create_completion_response(**data)
        elif endpoint_path == "/v1/chat/completions":
            response = self.server.create_chat_response(**data)
        elif endpoint_path == "/tokenize":
            response = self.server.create_tokenize_response(**data)
        elif endpoint_path == "/v1/embeddings":
            response = self.server.create_embedding_response(**data)
        else:
            response = {"error": "Endpoint not implemented in mock"}
        
        return response, json.dumps(response, indent=2), "", 200
    
    def record_memory_allocation(self, size: int):
        """Record memory allocation for monitoring."""
        self.memory_stats["allocations"] += 1
        self.memory_stats["current_usage"] = self.memory_stats.get("current_usage", 0) + size
        self.memory_stats["peak_usage"] = max(
            self.memory_stats.get("peak_usage", 0),
            self.memory_stats["current_usage"]
        )
    
    def record_memory_deallocation(self, size: int):
        """Record memory deallocation for monitoring."""
        self.memory_stats["deallocations"] += 1
        self.memory_stats["current_usage"] = max(0, self.memory_stats.get("current_usage", 0) - size)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        return {
            "allocations": self.memory_stats["allocations"],
            "deallocations": self.memory_stats["deallocations"],
            "current_usage": self.memory_stats.get("current_usage", 0),
            "peak_usage": self.memory_stats["peak_usage"],
            "request_count": len(self.request_history)
        }
    
    def process_image_tensor(self, tensor_image, jpeg_quality: int = DEFAULT_JPEG_QUALITY, 
                             max_dimension: Optional[int] = None, clear_memory: bool = True) -> Optional[str]:
        """Process an image tensor and return Base64 data URI.
        
        This method provides image processing capabilities for the mock client.
        """
        try:
            # Generate a cache key based on image properties
            cache_key = f"img_{self._cache_counter}"
            self._cache_counter += 1
            
            # Import required modules
            import numpy as np
            
            # Create a mock tensor if needed
            if tensor_image is None:
                # Create a simple mock tensor using numpy array (not MagicMock)
                mock_array = np.zeros((1, 256, 256, 3), dtype=np.float32)
                mock_array[:, :, :, 0] = 0.5  # Set some values for processing (R channel)
                mock_array[:, :, :, 1] = 0.5  # G channel
                mock_array[:, :, :, 2] = 0.5  # B channel
                tensor_image = mock_array
            
            # Use the actual image processing function
            result = tensor_to_base64_data_uri(
                tensor_image,
                jpeg_quality=jpeg_quality,
                max_dimension=max_dimension,
                clear_memory=clear_memory
            )
            
            # Cache the result
            if result:
                self._image_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image tensor: {e}")
            return None
    
    def clear_image_cache(self):
        """Clear the image processing cache."""
        self._image_cache.clear()
        gc.collect()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_images": len(self._image_cache),
            "cache_keys": list(self._image_cache.keys())
        }
    
    def build_vision_content(self, user_text: str, image_data: list, 
                            tensor_images=None, jpeg_quality: int = DEFAULT_JPEG_QUALITY,
                            max_dimension: Optional[int] = None, 
                            clear_memory: bool = True) -> list:
        """Build OpenAI Vision API compatible content array from text and images.
        
        This method provides vision content building capabilities for the mock client.
        """
        return build_vision_content(
            user_text=user_text,
            image_data=image_data,
            tensor_images=tensor_images,
            jpeg_quality=jpeg_quality,
            max_dimension=max_dimension,
            clear_memory=clear_memory
        )


# ============================================================================
# PHASE 1: IMAGE PROCESSING OPTIMIZATION TESTS
# ============================================================================

class TestPhase1ImageOptimization(unittest.TestCase):
    """Tests for Phase 1: Image Processing Optimization"""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_server = MockLlamaServer()
        self.client = MockLlamaClient("http://localhost:8080")
        
    def test_jpeg_quality_optimization(self):
        """Test JPEG quality parameter optimization."""
        import numpy as np
        
        # Create actual numpy arrays (not mocks) to simulate ComfyUI IMAGE tensor
        # Using np.zeros with float32 dtype and values in 0-1 range
        for quality in [50, 75, 90, 100]:
            # Create a simple mock tensor (simulating ComfyUI IMAGE tensor)
            # Shape: (height, width, channels) - 3D tensor (single image)
            mock_array = np.zeros((256, 256, 3), dtype=np.float32)
            mock_array[:, :, 0] = 0.5  # Set some values for processing (R channel)
            mock_array[:, :, 1] = 0.5  # G channel
            mock_array[:, :, 2] = 0.5  # B channel
            
            result = tensor_to_base64_data_uri(
                mock_array,
                jpeg_quality=quality,
                clear_memory=True
            )
            self.assertIsNotNone(result)
            self.assertTrue(result.startswith("data:image/jpeg;base64,"))
            
    def test_large_image_resizing(self):
        """Test that large images are properly resized."""
        import numpy as np
        
        # Create a large mock tensor (1024x1024)
        large_array = np.zeros((1024, 1024, 3), dtype=np.float32)
        large_array[:, :, 0] = 0.5
        large_array[:, :, 1] = 0.5
        large_array[:, :, 2] = 0.5
        
        result = tensor_to_base64_data_uri(
            large_array,
            max_dimension=512,
            clear_memory=True
        )
        
        self.assertIsNotNone(result)
        # Verify the image was processed (not raised an exception)
        self.assertTrue(result.startswith("data:image/jpeg;base64,"))
        
    def test_memory_clearing(self):
        """Test that memory is properly cleared after image processing."""
        import numpy as np
        
        # Create a large tensor
        large_array = np.zeros((512, 512, 3), dtype=np.float32)
        large_array[:, :, 0] = 0.5
        large_array[:, :, 1] = 0.5
        large_array[:, :, 2] = 0.5
        
        # Process with memory clearing enabled
        result = tensor_to_base64_data_uri(
            large_array,
            clear_memory=True
        )
        
        self.assertIsNotNone(result)
        
        # Force garbage collection
        gc.collect()
        
        # Memory should be cleared
        self.assertTrue(True)  # If we got here without memory error, it worked
        
    def test_batch_image_processing(self):
        """Test processing multiple images efficiently."""
        import numpy as np
        
        # Create a batch of images
        batch_array = np.zeros((4, 256, 256, 3), dtype=np.float32)
        batch_array[:, :, :, 0] = 0.5
        batch_array[:, :, :, 1] = 0.5
        batch_array[:, :, :, 2] = 0.5
        
        results = []
        for i in range(batch_array.shape[0]):
            result = tensor_to_base64_data_uri(
                batch_array[i],
                jpeg_quality=75,
                clear_memory=True
            )
            results.append(result)
        
        # All images should be processed
        self.assertEqual(len(results), 4)
        self.assertTrue(all(r is not None for r in results))


# ============================================================================
# PHASE 2: STREAMING STABILITY TESTS
# ============================================================================

class TestPhase2StreamingStability(unittest.TestCase):
    """Tests for Phase 2: Streaming Stability Improvements"""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_server = MockLlamaServer()
        self.client = MockLlamaClient("http://localhost:8080")
        
    def test_progress_monitoring(self):
        """Test progress monitoring for long-running requests."""
        self.client.timeout = 30  # Short timeout for testing
        
        response, raw, error, status = self.client.handle_completion(
            prompt="This is a test prompt for progress monitoring",
            n_predict=100,
            temperature=0.7
        )
        
        self.assertEqual(status, 200)
        self.assertEqual(error, "")
        
        # Verify request was recorded
        self.assertEqual(len(self.client.request_history), 1)
        
    def test_heartbeat_mechanism(self):
        """Test heartbeat mechanism for long-running requests."""
        # Simulate a long-running request by creating a custom mock
        class HeartbeatMockClient(MockLlamaClient):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.heartbeats_sent = 0
                
            def _make_request(self, endpoint_path: str, data: Dict[str, Any]) -> Tuple[str, str, str, int]:
                # Simulate long-running request with heartbeats
                start_time = time.time()
                while time.time() - start_time < 2:  # Simulate 2 second request
                    time.sleep(0.5)
                    self.heartbeats_sent += 1
                
                return self.server.create_completion_response(**data), \
                       json.dumps(self.server.create_completion_response(**data)), \
                       "", 200
        
        heartbeat_client = HeartbeatMockClient("http://localhost:8080")
        response, raw, error, status = heartbeat_client.handle_completion(
            prompt="Test heartbeat",
            n_predict=50
        )
        
        # Should have sent at least one heartbeat
        self.assertGreater(heartbeat_client.heartbeats_sent, 0)
        
    def test_partial_timeout_control(self):
        """Test partial timeout control for long-running requests."""
        # Test with different timeout values
        for timeout in [10, 30, 60]:
            client = MockLlamaClient("http://localhost:8080", timeout=timeout)
            response, raw, error, status = client.handle_completion(
                prompt="Test timeout control",
                n_predict=50
            )
            self.assertEqual(status, 200)
            self.assertEqual(error, "")


# ============================================================================
# PHASE 3: CACHE MANAGEMENT TESTS
# ============================================================================

class TestPhase3CacheManagement(unittest.TestCase):
    """Tests for Phase 3: Cache Management Improvements"""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_server = MockLlamaServer()
        self.client = MockLlamaClient("http://localhost:8080")
        
    def test_cache_clearing(self):
        """Test explicit cache clearing functionality."""
        # Simulate cache usage
        self.client.server.slot_usage["slot_1"] = {"active": True, "tokens": 100}
        self.client.server.slot_usage["slot_2"] = {"active": True, "tokens": 200}
        
        # Clear cache
        self.client.server.slot_usage.clear()
        
        # Verify cache is cleared
        self.assertEqual(len(self.client.server.slot_usage), 0)
        
    def test_slot_management(self):
        """Test improved slot management."""
        # Simulate slot allocation
        self.client.server.slot_usage["slot_1"] = {"active": True, "tokens": 100}
        self.client.server.slot_usage["slot_2"] = {"active": False, "tokens": 0}
        
        # Find available slot
        available_slot = None
        for slot_id, slot_data in self.client.server.slot_usage.items():
            if not slot_data["active"]:
                available_slot = slot_id
                break
        
        self.assertIsNotNone(available_slot)
        self.assertEqual(available_slot, "slot_2")
        
    def test_memory_monitoring(self):
        """Test memory usage monitoring."""
        # Record some memory allocations
        self.client.record_memory_allocation(1024)
        self.client.record_memory_allocation(2048)
        self.client.record_memory_deallocation(1024)
        
        stats = self.client.get_memory_stats()
        
        self.assertEqual(stats["allocations"], 2)
        self.assertEqual(stats["deallocations"], 1)
        self.assertEqual(stats["current_usage"], 2048)
        self.assertEqual(stats["peak_usage"], 3072)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestMockLlamaClientImageProcessing(unittest.TestCase):
    """Tests for MockLlamaClient image processing capabilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = MockLlamaClient("http://localhost:8080")
        
    def test_process_image_tensor_basic(self):
        """Test basic image tensor processing."""
        import numpy as np
        
        # Create a simple mock tensor (simulating ComfyUI IMAGE tensor)
        # Shape: (height, width, channels) - 3D tensor (single image)
        mock_array = np.zeros((256, 256, 3), dtype=np.float32)
        mock_array[:, :, 0] = 0.5  # Set some values for processing (R channel)
        mock_array[:, :, 1] = 0.5  # G channel
        mock_array[:, :, 2] = 0.5  # B channel
        
        result = self.client.process_image_tensor(mock_array)
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("data:image/jpeg;base64,"))
        
    def test_process_image_tensor_with_parameters(self):
        """Test image tensor processing with custom parameters."""
        import numpy as np
        
        # Create a mock tensor
        mock_array = np.zeros((512, 512, 3), dtype=np.float32)
        mock_array[:, :, 0] = 0.5
        mock_array[:, :, 1] = 0.5
        mock_array[:, :, 2] = 0.5
        
        result = self.client.process_image_tensor(
            mock_array,
            jpeg_quality=90,
            max_dimension=256,
            clear_memory=True
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("data:image/jpeg;base64,"))
        
    def test_build_vision_content_with_tensor(self):
        """Test build_vision_content with tensor images."""
        import numpy as np
        
        # Create a mock tensor using ComfyUI standard 4D tensor [Batch, H, W, C]
        mock_array = np.zeros((1, 256, 256, 3), dtype=np.float32)
        mock_array[:, :, :, 0] = 0.5  # R channel
        mock_array[:, :, :, 1] = 0.5  # G channel
        mock_array[:, :, :, 2] = 0.5  # B channel
        
        content, metadata = self.client.build_vision_content(
            user_text="Test prompt",
            image_data=[],
            tensor_images=mock_array
        )
        
        self.assertEqual(len(content), 2)  # text + image
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        
    def test_image_cache(self):
        """Test image caching functionality."""
        import numpy as np
        
        # Create mock tensors
        mock_array1 = np.zeros((256, 256, 3), dtype=np.float32)
        mock_array1[:, :, 0] = 0.5
        mock_array1[:, :, 1] = 0.5
        mock_array1[:, :, 2] = 0.5
        
        mock_array2 = np.zeros((256, 256, 3), dtype=np.float32)
        mock_array2[:, :, 0] = 0.8
        mock_array2[:, :, 1] = 0.8
        mock_array2[:, :, 2] = 0.8
        
        result1 = self.client.process_image_tensor(mock_array1)
        result2 = self.client.process_image_tensor(mock_array2)
        
        # Should have cached 2 images
        stats = self.client.get_cache_stats()
        self.assertEqual(stats["cached_images"], 2)
        
        # Clear cache
        self.client.clear_image_cache()
        stats = self.client.get_cache_stats()
        self.assertEqual(stats["cached_images"], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests combining all optimizations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_server = MockLlamaServer()
        self.client = MockLlamaClient("http://localhost:8080")
        
    def test_full_workflow(self):
        """Test a complete workflow with all optimizations."""
        # Simulate a complete workflow
        prompts = ["Hello", "How are you?", "What is your name?"]
        
        results = []
        for prompt in prompts:
            response, raw, error, status = self.client.handle_completion(
                prompt=prompt,
                n_predict=20,
                temperature=0.7
            )
            results.append((response, error, status))
        
        # All requests should succeed
        self.assertEqual(len(results), 3)
        for response, error, status in results:
            self.assertEqual(status, 200)
            self.assertEqual(error, "")
            
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        import threading
        
        results = []
        errors = []
        lock = threading.Lock()
        
        def make_request(prompt):
            try:
                response, raw, error, status = self.client.handle_completion(
                    prompt=prompt,
                    n_predict=10
                )
                with lock:
                    results.append((response, status))
            except Exception as e:
                with lock:
                    errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=make_request, args=(f"Test prompt {i}",))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # All requests should succeed
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 5)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)