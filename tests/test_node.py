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

    def test_extract_response_text_embeddings(self):
        response = {"data": [[0.1, 0.2], [0.3, 0.4]]}
        result = self.node._extract_response_text(response, "embeddings")
        self.assertEqual(result, "[[0.1, 0.2], [0.3, 0.4]]")

    def test_extract_response_text_tokenize(self):
        response = {"tokens": [1, 2, 3, 4, 5]}
        result = self.node._extract_response_text(response, "tokenize")
        self.assertEqual(result, "[1, 2, 3, 4, 5]")

    def test_extract_response_text_reranking(self):
        response = {"results": [{"index": 0, "score": 0.95}]}
        result = self.node._extract_response_text(response, "reranking")
        self.assertEqual(result, '[{"index": 0, "score": 0.95}]')

    def test_extract_response_text_detokenize(self):
        response = {"content": "hello world"}
        result = self.node._extract_response_text(response, "detokenize")
        self.assertEqual(result, "hello world")

    def test_extract_response_text_apply_template(self):
        response = {"content": "<system>prompt</system>"}
        result = self.node._extract_response_text(response, "apply_template")
        self.assertEqual(result, "<system>prompt</system>")

    def test_extract_response_text_infill(self):
        response = {"content": "completed code"}
        result = self.node._extract_response_text(response, "infill")
        self.assertEqual(result, "completed code")

    def test_extract_response_text_with_text_field(self):
        response = {"text": "fallback text"}
        result = self.node._extract_response_text(response, "completion")
        self.assertEqual(result, "fallback text")

    def test_extract_response_text_empty_response(self):
        response = {}
        result = self.node._extract_response_text(response, "completion")
        self.assertEqual(result, "")

    def test_extract_response_text_non_dict_response(self):
        result = self.node._extract_response_text(None, "completion")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
