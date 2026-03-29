import unittest
from unittest.mock import MagicMock, patch

from utils.llama_client import LlamaCppAPIClient


class TestLlamaClient(unittest.TestCase):
    def setUp(self):
        self.client = LlamaCppAPIClient(base_url="http://localhost:8000", api_key="test_key")

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_make_request_success(self, mock_get_session):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": "success"}

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        response = self.client._make_request("/test", {"key": "value"})
        self.assertEqual(response[0], {"content": "success"})
        mock_session.post.assert_called_once()

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_make_request_failure(self, mock_get_session):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = '{"error": "server error"}'
        mock_response.json.return_value = {"error": "server error"}

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        response = self.client._make_request("/test", {"key": "value"})
        self.assertEqual(response[2], "HTTP 500")
        self.assertEqual(response[3], 500)

    def test_handle_completion(self):
        with patch.object(
            self.client, "_make_request", return_value=({}, "", "", 200)
        ) as mock_method:
            result = self.client.handle_completion("Test prompt")
            self.assertEqual(result[3], 200)
            mock_method.assert_called_once()

    def test_handle_chat_completions(self):
        with patch.object(
            self.client, "_make_request", return_value=({}, "", "", 200)
        ) as mock_method:
            result = self.client.handle_chat_completions(
                messages=[{"role": "user", "content": "Hello"}]
            )
            self.assertEqual(result[3], 200)
            mock_method.assert_called_once()


if __name__ == "__main__":
    unittest.main()
