#!/usr/bin/env python3
"""
Unit tests for error handling in LlamaCppAPIClient
These tests verify error handling without requiring a running server
"""

import json
import unittest
from unittest.mock import patch, MagicMock
import requests

# Import the client
from utils.llama_client import LlamaCppAPIClient


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling in LlamaCppAPIClient"""

    def setUp(self):
        """Set up test fixtures"""
        self.client = LlamaCppAPIClient("http://localhost:8080", timeout=30)

    @patch('utils.llama_client.requests.post')
    def test_connection_error_handling(self, mock_post):
        """Test that connection errors are handled properly"""
        # Simulate connection error - should retry and then raise exception
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        with self.assertRaises(requests.exceptions.ConnectionError):
            self.client.handle_completion("test prompt")
        
        # Should have been called 3 times (max_retries)
        self.assertEqual(mock_post.call_count, 3)

    @patch('utils.llama_client.requests.post')
    def test_timeout_error_handling(self, mock_post):
        """Test that timeout errors are handled properly"""
        # Simulate timeout - should retry and then raise exception
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        with self.assertRaises(requests.exceptions.Timeout):
            self.client.handle_completion("test prompt")
        
        # Should have been called 3 times (max_retries)
        self.assertEqual(mock_post.call_count, 3)

    @patch('utils.llama_client.requests.post')
    def test_http_400_error_handling(self, mock_post):
        """Test that HTTP 400 errors are handled properly"""
        # Simulate 400 Bad Request
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request: invalid parameter"
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_post.return_value = mock_response
        
        response, raw, error, status = self.client.handle_completion("test prompt")
        
        self.assertEqual(error, "HTTP 400 (client_error)")
        self.assertEqual(status, 400)
        self.assertIn("Bad request", raw)

    @patch('utils.llama_client.requests.post')
    def test_http_401_error_handling(self, mock_post):
        """Test that HTTP 401 errors are handled properly"""
        # Simulate 401 Unauthorized
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_post.return_value = mock_response
        
        response, raw, error, status = self.client.handle_completion("test prompt")
        
        self.assertEqual(error, "HTTP 401 (client_error)")
        self.assertEqual(status, 401)

    @patch('utils.llama_client.requests.post')
    def test_http_500_error_handling(self, mock_post):
        """Test that HTTP 500 errors are handled properly"""
        # Simulate 500 Internal Server Error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_post.return_value = mock_response
        
        response, raw, error, status = self.client.handle_completion("test prompt")
        
        self.assertEqual(error, "HTTP 500 (server_error)")
        self.assertEqual(status, 500)

    @patch('utils.llama_client.requests.post')
    def test_empty_choices_response_handling(self, mock_post):
        """Test that empty choices response is handled properly"""
        # Simulate empty choices response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        mock_post.return_value = mock_response
        
        response, raw, error, status = self.client.handle_completion("test prompt")
        
        self.assertEqual(error, "Empty or invalid response")
        self.assertEqual(status, 200)
        self.assertIn("choices", raw)

    @patch('utils.llama_client.requests.post')
    def test_invalid_json_response_handling(self, mock_post):
        """Test that invalid JSON response is handled properly"""
        # Simulate invalid JSON response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Not valid JSON {{{"
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_post.return_value = mock_response
        
        response, raw, error, status = self.client.handle_completion("test prompt")
        
        self.assertEqual(error, "Invalid JSON response")
        self.assertEqual(status, 502)

    @patch('utils.llama_client.requests.post')
    def test_retry_on_transient_error(self, mock_post):
        """Test that transient errors trigger retries"""
        # Simulate transient error that succeeds on second attempt
        mock_post.side_effect = [
            requests.exceptions.Timeout("Timeout"),
            MagicMock(status_code=200, json=lambda: {"choices": [{"text": "success"}]})
        ]
        
        response, raw, error, status = self.client.handle_completion("test prompt")
        
        # Should have been called twice (retry)
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(error, "")
        self.assertEqual(status, 200)

    @patch('utils.llama_client.requests.post')
    def test_max_retries_exceeded(self, mock_post):
        """Test that max retries are respected"""
        # Simulate persistent transient error
        mock_post.side_effect = requests.exceptions.Timeout("Timeout")
        
        with self.assertRaises(requests.exceptions.Timeout):
            self.client.handle_completion("test prompt")
        
        # Should have been called 3 times (max_retries)
        self.assertEqual(mock_post.call_count, 3)

    @patch('utils.llama_client.requests.post')
    def test_chat_completions_error_handling(self, mock_post):
        """Test error handling in chat completions endpoint"""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service unavailable"
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_post.return_value = mock_response
        
        response, raw, error, status = self.client.handle_chat_completions(
            system_message="Test",
            user_message="Hello"
        )
        
        self.assertEqual(error, "HTTP 503 (server_error)")
        self.assertEqual(status, 503)

    @patch('utils.llama_client.requests.post')
    def test_request_exception_handling(self, mock_post):
        """Test handling of general request exceptions"""
        # Simulate a general request exception
        mock_post.side_effect = requests.exceptions.RequestException("Network error")
        
        response, raw, error, status = self.client.handle_completion("test prompt")
        
        self.assertIn("Request error", error)
        self.assertEqual(status, 500)


if __name__ == '__main__':
    unittest.main(verbosity=2)