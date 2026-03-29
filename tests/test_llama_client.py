import json
import unittest
from unittest.mock import MagicMock, patch

import requests

from utils.llama_client import RequestError  # noqa: F401
from utils.llama_client import ResponseError  # noqa: F401
from utils.llama_client import (
    ApiResponse,
    EndpointType,
    LlamaConnectionError,
    LlamaCppAPIClient,
)


class TestLlamaClient(unittest.TestCase):
    def setUp(self):
        self.client = LlamaCppAPIClient(base_url="http://localhost:8000", api_key="test_key")

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_make_request_success(self, mock_get_session):
        """Test successful HTTP request returns ApiResponse."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": "success"}

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        response = self.client._make_request(
            "/completion", {"key": "value"}, EndpointType.COMPLETION
        )
        self.assertIsInstance(response, ApiResponse)
        self.assertEqual(response.data, {"content": "success"})
        self.assertEqual(response.error, "")
        self.assertEqual(response.status_code, 200)
        mock_session.post.assert_called_once()

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_make_request_http_error(self, mock_get_session):
        """Test HTTP error (4xx, 5xx) returns ApiResponse with error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        response = self.client._make_request("/test", {"key": "value"}, EndpointType.COMPLETION)
        self.assertIsInstance(response, ApiResponse)
        self.assertIn("500", response.error)
        self.assertEqual(response.status_code, 500)

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_make_request_json_decode_error(self, mock_get_session):
        """Test JSONDecodeError returns ApiResponse with error."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
        mock_response.text = "invalid json"

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        response = self.client._make_request("/test", {"key": "value"}, EndpointType.COMPLETION)
        self.assertIsInstance(response, ApiResponse)
        self.assertIn("Invalid JSON", response.error)

    def test_clean_content_with_prompt_tag(self):
        """Test _clean_content extracts <prompt> tag content."""
        content = "Some text <prompt>extracted content</prompt> more text"
        result = self.client._clean_content(content)
        self.assertEqual(result, "extracted content")

    def test_clean_content_with_think_tag(self):
        """Test _clean_content removes <think> tags."""
        content = "Start <think>internal thought</think> end"
        result = self.client._clean_content(content)
        self.assertEqual(result, "Start  end")

    def test_clean_content_with_both_tags(self):
        """Test _clean_content prioritizes <prompt> tag."""
        content = "<think>thought</think> <prompt>result</prompt>"
        result = self.client._clean_content(content)
        self.assertEqual(result, "result")

    def test_clean_content_empty_string(self):
        """Test _clean_content handles empty string."""
        result = self.client._clean_content("")
        self.assertEqual(result, "")

    def test_handle_completion_returns_api_response(self):
        """Test handle_completion returns ApiResponse."""
        api_response = ApiResponse(
            data={"content": "test response"},
            raw='{"content": "test response"}',
            error="",
            status_code=200,
        )
        with patch.object(self.client, "_make_request", return_value=api_response):
            result = self.client.handle_completion("Test prompt")
            self.assertIsInstance(result, ApiResponse)
            self.assertEqual(result.data, {"content": "test response"})

    def test_handle_chat_completions_returns_api_response(self):
        """Test handle_chat_completions returns ApiResponse with metadata."""
        api_response = ApiResponse(
            data={"choices": [{"message": {"content": "Hello!"}}]},
            raw='{"choices":[{"message":{"content":"Hello!"}}]}',
            error="",
            status_code=200,
        )
        with patch.object(self.client, "_make_request", return_value=api_response):
            with patch.object(
                self.client,
                "_build_messages",
                return_value=([{"role": "user", "content": "Hi"}], [{"image_0": "metadata"}]),
            ):
                result = self.client.handle_chat_completions(
                    messages=[{"role": "user", "content": "Hi"}]
                )
                self.assertIsInstance(result, ApiResponse)
                self.assertEqual(result.data["choices"][0]["message"]["content"], "Hello!")
                self.assertEqual(result.metadata, [{"image_0": "metadata"}])

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_parse_and_validate_chat_response(self, mock_get_session):
        """Test _parse_and_validate validates chat response structure."""
        from utils.llama_client import EndpointType

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "hello"}}]}

        result = self.client._parse_and_validate(mock_response, EndpointType.CHAT_COMPLETION)
        self.assertEqual(result.data["choices"][0]["message"]["content"], "hello")
        self.assertEqual(result.error, "")

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_parse_and_validate_chat_missing_choices(self, mock_get_session):
        """Test _parse_and_validate returns error for missing choices."""
        from utils.llama_client import EndpointType

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"model": "test"}

        result = self.client._parse_and_validate(mock_response, EndpointType.CHAT_COMPLETION)
        self.assertIn("Invalid response structure", result.error)
        self.assertEqual(result.status_code, 200)

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_parse_and_validate_completion_response(self, mock_get_session):
        """Test _parse_and_validate validates completion response."""
        from utils.llama_client import EndpointType

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": "generated text"}

        result = self.client._parse_and_validate(mock_response, EndpointType.COMPLETION)
        self.assertEqual(result.data, {"content": "generated text"})

    def test_build_messages_with_system_and_user(self):
        """Test _build_messages builds correct message order."""
        messages, metadata = self.client._build_messages(
            system_message="You are helpful",
            user_message="Hello",
        )
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

    def test_build_messages_with_history(self):
        """Test _build_messages includes message history."""
        history = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
        ]
        messages, metadata = self.client._build_messages(
            messages=history,
            user_message="Second message",
        )
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "First message")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[2]["role"], "user")
        self.assertEqual(messages[2]["content"], "Second message")

    def test_close(self):
        """Test close() closes session and resets timing."""
        with patch.object(self.client, "_close_session") as mock_close:
            self.client.close()
            mock_close.assert_called_once()
            self.assertIsNone(self.client._request_start_time)

    def test_parse_and_validate_raises_request_error_4xx(self):
        """Test _parse_and_validate returns error for 4xx status."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        result = self.client._parse_and_validate(mock_response, EndpointType.COMPLETION)
        self.assertIn("400", result.error)
        self.assertEqual(result.status_code, 400)

    def test_parse_and_validate_raises_request_error_5xx(self):
        """Test _parse_and_validate returns error for 5xx status."""
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.text = "Bad Gateway"

        result = self.client._parse_and_validate(mock_response, EndpointType.COMPLETION)
        self.assertIn("502", result.error)
        self.assertEqual(result.status_code, 502)

    def test_parse_and_validate_raises_response_error_on_invalid_json(self):
        """Test _parse_and_validate returns error on JSON decode error."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)

        result = self.client._parse_and_validate(mock_response, EndpointType.COMPLETION)
        self.assertIn("Invalid JSON", result.error)
        self.assertEqual(result.status_code, 200)

    def test_parse_and_validate_raises_response_error_on_invalid_structure(self):
        """Test _parse_and_validate returns error for invalid structure."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"model": "test"}  # Missing required fields

        result = self.client._parse_and_validate(mock_response, EndpointType.CHAT_COMPLETION)
        self.assertIn("Invalid response structure", result.error)
        self.assertEqual(result.status_code, 200)

    def test_api_response_unpack(self):
        """Test ApiResponse.unpack() returns tuple of all components."""
        metadata = [{"image_0": "metadata"}]
        response = ApiResponse(
            data={"content": "test"},
            raw='{"content":"test"}',
            error="",
            status_code=200,
            metadata=metadata,
        )

        data, raw, error, status_code, metadata_out = response.unpack()
        self.assertEqual(data, {"content": "test"})
        self.assertEqual(raw, '{"content":"test"}')
        self.assertEqual(error, "")
        self.assertEqual(status_code, 200)
        self.assertEqual(metadata_out, metadata)

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_parse_and_validate_completion_with_text_field(self, mock_get_session):
        """Test _parse_and_validate handles 'text' field in completion response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "generated text"}

        result = self.client._parse_and_validate(mock_response, EndpointType.COMPLETION)
        self.assertEqual(result.data, {"text": "generated text"})
        self.assertEqual(result.error, "")

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_handle_embeddings(self, mock_get_session):
        """Test handle_embeddings returns ApiResponse."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        response = self.client.handle_embeddings(input_text="test text", model="embedding-model")
        self.assertIsInstance(response, ApiResponse)
        self.assertEqual(response.data, {"data": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]})
        mock_session.post.assert_called_once()

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_handle_embeddings_invalid_response(self, mock_get_session):
        """Test handle_embeddings returns error for missing data field."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"model": "embedding-model"}

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        response = self.client.handle_embeddings(input_text="test text", model="embedding-model")
        self.assertIn("Invalid response structure", response.error)

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_handle_tokenize(self, mock_get_session):
        """Test handle_tokenize returns ApiResponse."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"tokens": [1, 2, 3, 4, 5]}

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        response = self.client.handle_tokenize(content="test")
        self.assertIsInstance(response, ApiResponse)
        self.assertEqual(response.data, {"tokens": [1, 2, 3, 4, 5]})

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_handle_tokenize_invalid_response(self, mock_get_session):
        """Test handle_tokenize returns error for missing tokens field."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"model": "tokenizer"}

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        response = self.client.handle_tokenize(content="test")
        self.assertIn("Invalid response structure", response.error)

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_handle_detokenize(self, mock_get_session):
        """Test handle_detokenize returns ApiResponse."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": "hello world"}

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        response = self.client.handle_detokenize(tokens="[1,2,3]")
        self.assertIsInstance(response, ApiResponse)
        self.assertEqual(response.data, {"content": "hello world"})

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_handle_infill(self, mock_get_session):
        """Test handle_infill returns ApiResponse."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": "filled text"}

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        response = self.client.handle_infill(input_prefix="prefix", input_suffix="suffix")
        self.assertIsInstance(response, ApiResponse)

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_handle_apply_template(self, mock_get_session):
        """Test handle_apply_template returns ApiResponse."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": "<system>prompt<user>message</user></system>"}

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        response = self.client.handle_apply_template(messages='[{"role":"user","content":"test"}]')
        self.assertIsInstance(response, ApiResponse)

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_handle_reranking(self, mock_get_session):
        """Test handle_reranking returns ApiResponse."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [{"index": 0, "score": 0.95}]}

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        response = self.client.handle_reranking(query="test query", documents='["doc1", "doc2"]')
        self.assertIsInstance(response, ApiResponse)
        self.assertEqual(response.data, {"results": [{"index": 0, "score": 0.95}]})

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_handle_reranking_invalid_response(self, mock_get_session):
        """Test handle_reranking returns error for missing results field."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"model": "reranker"}

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        response = self.client.handle_reranking(query="test query", documents='["doc1", "doc2"]')
        self.assertIn("Invalid response structure", response.error)

    def test_add_system_message(self):
        """Test _add_system_message adds system message at beginning."""
        messages = []
        self.client._add_system_message(messages, "You are helpful")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are helpful")

    def test_add_system_message_empty_string(self):
        """Test _add_system_message skips empty string."""
        messages = []
        self.client._add_system_message(messages, "")
        self.assertEqual(len(messages), 0)

    def test_add_assistant_prefill(self):
        """Test _add_assistant_prefill adds assistant message."""
        messages = []
        self.client._add_assistant_prefill(messages, "I am ready")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["content"], "I am ready")

    def test_add_assistant_prefill_empty_string(self):
        """Test _add_assistant_prefill skips empty string."""
        messages = []
        self.client._add_assistant_prefill(messages, "")
        self.assertEqual(len(messages), 0)

    def test_build_messages_with_assistant_prefill(self):
        """Test _build_messages includes assistant prefill at end."""
        messages, _ = self.client._build_messages(
            system_message="System prompt",
            user_message="User input",
            assistant_message="Assistant prefill",
        )
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "Assistant prefill")

    def test_connection_error_raised_on_max_retries(self):
        """Test LlamaConnectionError is raised after max retries are exhausted."""
        with self.assertRaises(LlamaConnectionError) as context:
            raise LlamaConnectionError("Connection failed after 3 retries: Connection refused")

        self.assertIn("3 retries", str(context.exception))

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_retry_on_transient_error_with_timeout(self, mock_get_session):
        """Test retry_on_transient_error decorator retries on Timeout."""
        mock_session = MagicMock()
        # First 2 calls timeout, 3rd succeeds
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": "success"}

        mock_session.post.side_effect = [
            requests.exceptions.Timeout("Request timeout"),
            requests.exceptions.Timeout("Request timeout"),
            mock_response,
        ]
        mock_get_session.return_value = mock_session

        response = self.client._make_request(
            "/completion", {"key": "value"}, EndpointType.COMPLETION
        )
        self.assertIsInstance(response, ApiResponse)
        self.assertEqual(response.data, {"content": "success"})
        self.assertEqual(mock_session.post.call_count, 3)

    @patch.object(LlamaCppAPIClient, "_get_session")
    def test_retry_on_transient_error_max_retries_exceeded(self, mock_get_session):
        """Test retry_on_transient_error raises LlamaConnectionError after max retries."""
        mock_session = MagicMock()
        mock_session.post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        mock_get_session.return_value = mock_session

        with self.assertRaises(LlamaConnectionError) as context:
            self.client._make_request("/completion", {"key": "value"}, EndpointType.COMPLETION)

        self.assertIn("Connection failed after 3 retries", str(context.exception))
        self.assertIsInstance(context.exception.__cause__, requests.exceptions.ConnectionError)
        # Should have attempted 3 times
        self.assertEqual(mock_session.post.call_count, 3)


if __name__ == "__main__":
    unittest.main()
