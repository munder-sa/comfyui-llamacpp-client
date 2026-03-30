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

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_detokenize_endpoint(self, MockClientClass):
        args = self.base_args.copy()
        args["endpoint"] = "detokenize"

        mock_client = MockClientClass.return_value
        mock_client.handle_detokenize.return_value = ApiResponse(
            {"content": "decoded text"}, "", "", 200
        )

        result = self.node.process_request(**args)

        mock_client.handle_detokenize.assert_called_once()
        self.assertEqual(result[0], "decoded text")

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_apply_template_endpoint(self, MockClientClass):
        args = self.base_args.copy()
        args["endpoint"] = "apply_template"

        mock_client = MockClientClass.return_value
        mock_client.handle_apply_template.return_value = ApiResponse(
            {"content": "<|user|>\nhello"}, "", "", 200
        )

        result = self.node.process_request(**args)

        mock_client.handle_apply_template.assert_called_once()
        self.assertEqual(result[0], "<|user|>\nhello")

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_infill_endpoint(self, MockClientClass):
        args = self.base_args.copy()
        args["endpoint"] = "infill"

        mock_client = MockClientClass.return_value
        mock_client.handle_infill.return_value = ApiResponse(
            {"content": "infilled code"}, "", "", 200
        )

        result = self.node.process_request(**args)

        mock_client.handle_infill.assert_called_once()
        self.assertEqual(result[0], "infilled code")


class TestProcessRequestMetadata(unittest.TestCase):
    def setUp(self):
        self.node = LlamaCppClientNode()
        self.base_args = {
            "server_url": "http://localhost:8080",
            "endpoint": "chat_completions",
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
            "extract_metadata": True,
        }

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_metadata_populated_from_chat(self, MockClientClass):
        mock_client = MockClientClass.return_value
        mock_response = ApiResponse(
            data={"choices": [{"message": {"content": "ok"}}]},
            raw='{"choices": [{"message": {"content": "ok"}}]}',
            error="",
            status_code=200,
            metadata=[{"metrics": "data1"}, {"model_info": "data2"}],
        )
        mock_client.handle_chat_completions.return_value = mock_response

        result = self.node.process_request(**self.base_args)

        metadata = result[4]
        self.assertIn("image_0", metadata)
        self.assertIn("image_1", metadata)
        self.assertEqual(metadata["image_0"], {"metrics": "data1"})
        self.assertEqual(metadata["image_1"], {"model_info": "data2"})

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_metadata_empty_if_no_metrics(self, MockClientClass):
        mock_client = MockClientClass.return_value
        mock_response = ApiResponse(
            data={"choices": [{"message": {"content": "ok"}}]},
            raw='{"choices": [{"message": {"content": "ok"}}]}',
            error="",
            status_code=200,
            metadata=[],
        )
        mock_client.handle_chat_completions.return_value = mock_response

        result = self.node.process_request(**self.base_args)

        metadata = result[4]
        self.assertEqual(metadata, {})


class TestMoENodeFeatures(unittest.TestCase):
    """Tests for moe_mode preset logic and timings extraction in process_request."""

    def setUp(self):
        self.node = LlamaCppClientNode()
        # Minimal valid args; only the fields relevant to each test are overridden
        self.base_args = {
            "server_url": "http://localhost:8080",
            "endpoint": "completion",
            "prompt": "test prompt",
            "system_message": "",
            "user_message": "",
            "assistant_message": "",
            "messages": "[]",
            "temperature": 0.8,
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
    def test_moe_mode_disables_heavy_stats(self, MockClientClass):
        """moe_mode=True forces n_probs=0 and timings_per_token=False."""
        mock_client = MockClientClass.return_value
        captured_kwargs = {}

        def capture(**kwargs):
            captured_kwargs.update(kwargs)
            return ApiResponse(
                data={"content": "ok"}, raw='{"content":"ok"}', error="", status_code=200
            )

        mock_client.handle_completion.side_effect = capture

        args = {**self.base_args, "moe_mode": True, "n_probs": 10, "timings_per_token": True}
        self.node.process_request(**args)

        self.assertEqual(captured_kwargs.get("n_probs"), 0)
        self.assertFalse(captured_kwargs.get("timings_per_token"))

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_moe_mode_disables_post_sampling_probs(self, MockClientClass):
        """moe_mode=True forces post_sampling_probs=False."""
        mock_client = MockClientClass.return_value
        captured_kwargs = {}

        def capture(**kwargs):
            captured_kwargs.update(kwargs)
            return ApiResponse(
                data={"content": "ok"}, raw='{"content":"ok"}', error="", status_code=200
            )

        mock_client.handle_completion.side_effect = capture

        args = {**self.base_args, "moe_mode": True, "post_sampling_probs": True}
        self.node.process_request(**args)

        self.assertFalse(captured_kwargs.get("post_sampling_probs"))

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_moe_mode_simplifies_samplers(self, MockClientClass):
        """moe_mode=True replaces default sampler chain with lightweight version."""
        mock_client = MockClientClass.return_value
        captured_kwargs = {}

        def capture(**kwargs):
            captured_kwargs.update(kwargs)
            return ApiResponse(
                data={"content": "ok"}, raw='{"content":"ok"}', error="", status_code=200
            )

        mock_client.handle_completion.side_effect = capture

        # Pass the default samplers string — moe_mode should simplify it
        default_samplers = '["dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]'
        args = {**self.base_args, "moe_mode": True, "samplers": default_samplers}
        self.node.process_request(**args)

        # After simplification, only lightweight samplers should remain
        resultant = captured_kwargs.get("samplers", [])
        if isinstance(resultant, str):
            import json

            resultant = json.loads(resultant)
        self.assertNotIn("dry", resultant)
        self.assertNotIn("xtc", resultant)
        self.assertIn("temperature", resultant)

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_moe_mode_false_preserves_samplers(self, MockClientClass):
        """moe_mode=False must not alter the user-specified samplers."""
        mock_client = MockClientClass.return_value
        captured_kwargs = {}

        def capture(**kwargs):
            captured_kwargs.update(kwargs)
            return ApiResponse(
                data={"content": "ok"}, raw='{"content":"ok"}', error="", status_code=200
            )

        mock_client.handle_completion.side_effect = capture

        default_samplers = '["dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]'
        args = {**self.base_args, "moe_mode": False, "samplers": default_samplers}
        self.node.process_request(**args)

        # samplers must not have been simplified — dry and xtc should still be there
        resultant = captured_kwargs.get("samplers", [])
        if isinstance(resultant, str):
            import json

            resultant = json.loads(resultant)
        self.assertIn("dry", resultant)
        self.assertIn("xtc", resultant)

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_timings_extracted_to_metadata(self, MockClientClass):
        """If response contains 'timings', it must appear in the node's metadata output."""
        timings_data = {"prompt_n": 5, "predicted_n": 20, "predicted_per_second": 45.0}
        mock_client = MockClientClass.return_value
        mock_client.handle_completion.return_value = ApiResponse(
            data={"content": "hi", "timings": timings_data},
            raw='{"content":"hi","timings":{}}',
            error="",
            status_code=200,
        )

        result = self.node.process_request(**self.base_args)

        metadata = result[4]  # 5th element is metadata dict
        self.assertIn("timings", metadata)
        self.assertAlmostEqual(metadata["timings"]["predicted_per_second"], 45.0)

    @patch("llamacpp_client_node.LlamaCppAPIClient")
    def test_timings_absent_does_not_error(self, MockClientClass):
        """If response has no 'timings' field, it must not raise and metadata has no timings key."""
        mock_client = MockClientClass.return_value
        mock_client.handle_completion.return_value = ApiResponse(
            data={"content": "hi"},
            raw='{"content":"hi"}',
            error="",
            status_code=200,
        )

        result = self.node.process_request(**self.base_args)

        metadata = result[4]
        self.assertNotIn("timings", metadata)


if __name__ == "__main__":
    unittest.main()
