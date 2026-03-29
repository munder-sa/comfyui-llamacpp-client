import unittest
from unittest.mock import patch

from llamacpp_client_node import LlamaCppClientNode


class TestNode(unittest.TestCase):
    def setUp(self):
        self.node = LlamaCppClientNode()

    @patch("llamacpp_client_node.LlamaCppClientNode.process_request")
    def test_process_request_success(self, mock_process_request):
        mock_process_request.return_value = {"status": "success", "data": "processed"}

        result = self.node.process_request({"key": "value"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"], "processed")

    @patch("llamacpp_client_node.LlamaCppClientNode.process_request")
    def test_process_request_failure(self, mock_process_request):
        mock_process_request.side_effect = Exception("Processing error")

        with self.assertRaises(Exception):
            self.node.process_request({"key": "value"})

    def test_input_types(self):
        input_types = self.node.INPUT_TYPES()
        self.assertIsInstance(input_types, dict)
        self.assertIn("required", input_types)
        self.assertIn("optional", input_types)

    def test_extract_response_text_completion(self):
        response = {"content": "hello"}
        result = self.node._extract_response_text(response, "completion")
        self.assertEqual(result, "hello")

    def test_extract_response_text_chat(self):
        response = {"choices": [{"message": {"content": "world"}}]}
        result = self.node._extract_response_text(response, "chat_completions")
        self.assertEqual(result, "world")

    def test_extract_response_text_scalar(self):
        result = self.node._extract_response_text("plain text", "completion")
        self.assertEqual(result, "plain text")


if __name__ == "__main__":
    unittest.main()
