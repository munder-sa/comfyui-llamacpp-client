import unittest
from unittest.mock import patch

import numpy as np

from llamacpp_client_node import LlamaCppClientNode
from utils.llama_client import ApiResponse


class TestLlamaCppClientNodeStructure(unittest.TestCase):
    def setUp(self):
        self.node = LlamaCppClientNode()

    def test_input_types_has_required_keys(self):
        input_types = self.node.INPUT_TYPES()
        required = input_types["required"]
        self.assertIn("server_url", required)
        self.assertIn("endpoint", required)
        self.assertEqual(required["server_url"][0], "STRING")

    def test_return_types_5_tuple(self):
        self.assertEqual(len(self.node.RETURN_TYPES), 5)
        # (response_text, raw_response, error, status_code, metadata)
        self.assertEqual(self.node.RETURN_TYPES[0], "STRING")
        self.assertEqual(self.node.RETURN_TYPES[2], "STRING")
        self.assertEqual(self.node.RETURN_TYPES[3], "INT")

    def test_return_names_5_tuple(self):
        self.assertEqual(len(self.node.RETURN_NAMES), 5)
        self.assertEqual(self.node.RETURN_NAMES[0], "response")
        self.assertEqual(self.node.RETURN_NAMES[2], "error")

    def test_node_category(self):
        self.assertEqual(self.node.CATEGORY, "AI/LlamaCpp")


class TestProcessRequest(unittest.TestCase):
    def setUp(self):
        self.node = LlamaCppClientNode()
        # Common valid arguments for process_request
        self.common_args = {
            "server_url": "http://localhost:8080",
            "endpoint": "completion",
            "prompt": "test prompt",
            "system_message": "",
            "user_message": "",
            "assistant_message": "",
            "messages": "[]",
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.95,
            "min_p": 0.05,
            "n_predict": 128,
            "stop_sequences": "[]",
            "stream": False,
            "cache_prompt": True,
            "api_key": "",
            "timeout": 60,
            "images": None,
            "image_data": "[]",
            "extract_metadata": False,
        }

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_process_request_completion_success(self, MockClientClass):
        # Setup mock client
        mock_client = MockClientClass.return_value
        mock_response = ApiResponse(
            data={"content": "Hello world"},
            raw='{"content": "Hello world"}',
            error="",
            status_code=200,
        )
        mock_client.handle_completion.return_value = mock_response

        # Execute
        result = self.node.process_request(**self.common_args)

        # Verify
        self.assertEqual(result[0], "Hello world")
        self.assertEqual(result[2], "")
        self.assertEqual(result[3], 200)
        mock_client.handle_completion.assert_called_once()

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_process_request_chat_completions_with_images(self, MockClientClass):
        # Update args for chat_completions
        args = self.common_args.copy()
        args["endpoint"] = "chat_completions"
        args["user_message"] = "What is in this image?"
        args["images"] = np.zeros((1, 64, 64, 3), dtype=np.float32)

        # Setup mock client
        mock_client = MockClientClass.return_value
        mock_response = ApiResponse(
            data={"choices": [{"message": {"content": "An image"}}]},
            raw='{"choices": [...]}',
            error="",
            status_code=200,
            metadata=[{"image_0": "meta"}],
        )
        mock_client.handle_chat_completions.return_value = mock_response

        # Execute
        result = self.node.process_request(**args)

        # Verify
        self.assertEqual(result[0], "An image")
        self.assertIn("image_0", result[4])  # metadata dict
        mock_client.handle_chat_completions.assert_called_once()

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_process_request_error_status(self, MockClientClass):
        mock_client = MockClientClass.return_value
        mock_response = ApiResponse(data={}, raw="", error="Connection Refused", status_code=503)
        mock_client.handle_completion.return_value = mock_response

        result = self.node.process_request(**self.common_args)

        self.assertEqual(result[2], "Connection Refused")
        self.assertEqual(result[3], 503)

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_process_request_exception_handling(self, MockClientClass):
        mock_client = MockClientClass.return_value
        mock_client.handle_completion.side_effect = Exception("Unexpected crash")

        result = self.node.process_request(**self.common_args)

        self.assertIn("Unexpected crash", result[2])
        self.assertEqual(result[3], 500)


class TestProcessRequestOtherEndpoints(unittest.TestCase):
    def setUp(self):
        self.node = LlamaCppClientNode()
        self.base_args = {
            "server_url": "http://localhost:8080",
            "prompt": "test",
            "system_message": "",
            "user_message": "",
            "assistant_message": "",
            "messages": "[]",
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "min_p": 0.05,
            "n_predict": 128,
            "stop_sequences": "[]",
            "stream": False,
            "cache_prompt": True,
            "api_key": "",
            "timeout": 60,
            "images": None,
            "image_data": "[]",
            "extract_metadata": False,
        }

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_embeddings_endpoint(self, MockClientClass):
        args = self.base_args.copy()
        args["endpoint"] = "embeddings"

        mock_client = MockClientClass.return_value
        mock_client.handle_embeddings.return_value = ApiResponse({"data": [[0.1]]}, "", "", 200)

        result = self.node.process_request(**args)

        mock_client.handle_embeddings.assert_called_once()
        self.assertEqual(result[0], "[[0.1]]")

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_tokenize_endpoint(self, MockClientClass):
        args = self.base_args.copy()
        args["endpoint"] = "tokenize"

        mock_client = MockClientClass.return_value
        mock_client.handle_tokenize.return_value = ApiResponse({"tokens": [1, 2, 3]}, "", "", 200)

        result = self.node.process_request(**args)

        mock_client.handle_tokenize.assert_called_once()
        self.assertEqual(result[0], "[1, 2, 3]")

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_reranking_endpoint(self, MockClientClass):
        args = self.base_args.copy()
        args["endpoint"] = "reranking"

        mock_client = MockClientClass.return_value
        mock_client.handle_reranking.return_value = ApiResponse(
            {"results": [{"index": 0}]}, "", "", 200
        )

        result = self.node.process_request(**args)

        mock_client.handle_reranking.assert_called_once()
        self.assertEqual(result[0], '[{"index": 0}]')


if __name__ == "__main__":
    unittest.main()
