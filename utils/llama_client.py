import json
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

# Local imports
try:
    from image_utils import build_vision_content, process_image_data_string
    from logger import log_debug, log_error, log_info
    from param_utils import (
        CHAT_COMPLETION_PARAMS,
        COMMON_COMPLETION_PARAMS,
        clean_params,
        map_parameters,
    )
    from typing import Tuple, List, Dict, Any
except ImportError:
    from .image_utils import build_vision_content, process_image_data_string
    from .logger import log_debug, log_error, log_info
    from .param_utils import (
        CHAT_COMPLETION_PARAMS,
        COMMON_COMPLETION_PARAMS,
        clean_params,
        map_parameters,
    )
    from typing import Tuple, List, Dict, Any
except ImportError:
    from .image_utils import build_vision_content, process_image_data_string
    from .logger import log_debug, log_error, log_info
    from .param_utils import (
        CHAT_COMPLETION_PARAMS,
        COMMON_COMPLETION_PARAMS,
        clean_params,
        map_parameters,
    )

class LlamaCppClientError(Exception):
    """Base exception for LlamaCppClient errors."""
    pass


class ConnectionError(LlamaCppClientError):
    """Raised when connection to llama-server fails."""
    pass


class RequestError(LlamaCppClientError):
    """Raised when a request to llama-server fails."""
    pass


class ResponseError(LlamaCppClientError):
    """Raised when response from llama-server is invalid."""
    pass


def retry_on_transient_error(max_retries: int = 3, base_delay: float = 1.0) -> Callable[[Callable], Callable]:
    """Decorator to retry on transient network errors with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        log_info(f"Transient error ({type(e).__name__}), retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        log_error(f"Max retries ({max_retries}) exceeded after transient error")
                        raise
            return None  # Should not reach here
        return wrapper
    return decorator


class LlamaCppAPIClient:
    """Client for handling requests to llama-server."""

    def __init__(self, base_url: str, api_key: str = "", timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = None
        self._request_start_time = None

    def _get_session(self):
        """Get or create a requests session for connection pooling."""
        if self._session is None:
            self._session = requests.Session()
            # Configure session for better performance
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                pool_block=False,
            )
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
        return self._session

    def _close_session(self):
        """Close the session and release resources."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def _reset_timing(self):
        """Reset timing information for the next request."""
        self._request_start_time = None

    @retry_on_transient_error(max_retries=3, base_delay=1.0)
    def _make_request(self, endpoint_path: str, data: Dict[str, Any]) -> Tuple[str, str, str, int]:
        """Make HTTP POST request to llama-server."""
        url = f"{self.base_url}{endpoint_path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        log_info(f"Sending request to {url}")
        # Log parameter values for debugging
        log_debug(f"Request Payload (before sending): | Data: {json.dumps(data, indent=2)[:500]}")
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                log_debug(f"  {key}: {type(value).__name__} with {len(value)} items")
            elif isinstance(value, str):
                log_debug(f"  {key}: str (length: {len(value)})")
            else:
                log_debug(f"  {key}: {type(value).__name__} = {value}")

        try:
            # Start timing
            self._request_start_time = time.time()
            
            session = self._get_session()
            response = session.post(url, json=data, headers=headers, timeout=self.timeout)
            
            # Calculate elapsed time
            elapsed_time = time.time() - self._request_start_time if self._request_start_time else 0
            log_debug(f"Request completed in {elapsed_time:.2f}s")

            # Check for HTTP errors first
            if response.status_code >= 400:
                error_type = "server_error" if response.status_code >= 500 else "client_error"
                error_detail = response.text[:500] if response.text else "No response body"
                log_error(f"HTTP {response.status_code} ({error_type}): {error_detail}")
                return "", response.text, f"HTTP {response.status_code} ({error_type})", response.status_code

            try:
                response_json = response.json()
                log_debug(f"Response JSON preview:", response_json)
                
                # Check for empty response
                if not response_json or (isinstance(response_json, dict) and not response_json.get("choices")):
                    log_error("Empty or invalid response structure")
                    return "", json.dumps(response_json, indent=2), "Empty or invalid response", 200
                
                return response_json, json.dumps(response_json, indent=2), "", response.status_code
            except json.JSONDecodeError as e:
                log_error(
                    f"Failed to decode JSON from response. Status: {response.status_code}, Text: {response.text[:200]}"
                )
                return "", response.text, "Invalid JSON response", 502

        except requests.exceptions.Timeout as e:
            log_error(f"Request timeout: {str(e)}")
            raise  # Re-raise to allow retry decorator
        except requests.exceptions.ConnectionError as e:
            log_error(f"Connection error: {str(e)}")
            raise  # Re-raise to allow retry decorator
        except requests.exceptions.RequestException as e:
            log_error(f"Request exception: {type(e).__name__} - {str(e)}")
            return "", "", f"Request error: {str(e)}", 500

    def handle_completion(self, prompt: str, **kwargs) -> Tuple[str, str, str, int]:
        """Handle /completion endpoint.
        
        Args:
            prompt: The text prompt for generation
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (response, raw_response, error, status_code)
        """
        params: Dict[str, Any] = {"prompt": prompt}
        mapped_params = map_parameters(kwargs, COMMON_COMPLETION_PARAMS)
        params.update(mapped_params)
        params = clean_params(params)
        return self._make_request("/completion", params)

    def handle_chat_completions(self, **kwargs) -> Tuple[str, str, str, int, Dict[str, Any]]:
        """Handle /v1/chat/completions endpoint.
        
        Args:
            **kwargs: Chat-specific parameters including messages, system_message, etc.
            
        Returns:
            Tuple of (response, raw_response, error, status_code, metadata)
        """
        messages: List[Dict[str, str]] = []

        # Parse existing messages if provided
        messages_input = kwargs.get("messages")
        if messages_input:
            if isinstance(messages_input, str):
                # 文字列の場合は JSON パース
                try:
                    messages = json.loads(messages_input)
                except json.JSONDecodeError:
                    pass
            elif isinstance(messages_input, list):
                # リストの場合はそのまま使用
                messages = messages_input
            else:
                messages = []

        # Add individual messages if provided
        system_message = kwargs.get("system_message")
        if isinstance(system_message, str) and system_message.strip():
            messages.append({"role": "system", "content": system_message})

        # Build user content (support for multimodal / vision)
        user_text = kwargs.get("user_message") or kwargs.get("prompt", "")
        image_data = kwargs.get("image_data")
        tensor_images = kwargs.get("images")

        # If image_data is still string, parse it
        if isinstance(image_data, str) and image_data.strip():
            image_data = process_image_data_string(image_data)

        # Extract metadata if requested
        extract_metadata = kwargs.get("extract_metadata", False)
        user_content_items, metadata_list = build_vision_content(
            user_text, image_data, tensor_images, extract_metadata=extract_metadata
        )

        if user_content_items:
            # If we only have text and no images, send as simple string (OpenAI compat)
            if len(user_content_items) == 1 and user_content_items[0]["type"] == "text":
                messages.append({"role": "user", "content": user_content_items[0]["text"]})
            else:
                messages.append({"role": "user", "content": user_content_items})

        if kwargs.get("assistant_message") and kwargs["assistant_message"].strip():
            messages.append({"role": "assistant", "content": kwargs["assistant_message"]})

        params: Dict[str, Any] = {
            "messages": messages,
            "model": kwargs.get("model", "default"),
        }

        mapped_params = map_parameters(kwargs, CHAT_COMPLETION_PARAMS)
        params.update(mapped_params)
        params = clean_params(params)
        
        # Make the request and return metadata along with response
        response, raw_response, error, status_code = self._make_request("/v1/chat/completions", params)
        return response, raw_response, error, status_code, metadata_list

    def handle_embeddings(self, **kwargs) -> Tuple[str, str, str, int]:
        """Handle /v1/embeddings endpoint.
        
        Args:
            **kwargs: Embedding-specific parameters
            
        Returns:
            Tuple of (response, raw_response, error, status_code)
        """
        input_text = kwargs.get("input_text") or kwargs.get("content") or kwargs.get("prompt", "")
        params: Dict[str, Any] = {
            "input": input_text,
            "model": kwargs.get("model", "default"),
            "encoding_format": kwargs.get("encoding_format", "float"),
        }
        params = clean_params(params)
        return self._make_request("/v1/embeddings", params)

    def handle_tokenize(self, **kwargs) -> Tuple[str, str, str, int]:
        """Handle /tokenize endpoint.
        
        Args:
            **kwargs: Tokenization parameters
            
        Returns:
            Tuple of (response, raw_response, error, status_code)
        """
        content = kwargs.get("content") or kwargs.get("prompt", "")
        params: Dict[str, Any] = {
            "content": content,
            "add_special": kwargs.get("add_special", False),
            "parse_special": kwargs.get("parse_special", True),
            "with_pieces": kwargs.get("with_pieces", False),
        }
        return self._make_request("/tokenize", params)

    def handle_detokenize(self, **kwargs) -> Tuple[str, str, str, int]:
        """Handle /detokenize endpoint.
        
        Args:
            **kwargs: Detokenization parameters
            
        Returns:
            Tuple of (response, raw_response, error, status_code)
        """
        tokens_str = kwargs.get("tokens", "[]")
        tokens: List[int] = []
        if isinstance(tokens_str, str):
            try:
                tokens = json.loads(tokens_str)
            except json.JSONDecodeError:
                tokens = []
        params: Dict[str, List[int]] = {"tokens": tokens}
        return self._make_request("/detokenize", params)

    def handle_apply_template(self, **kwargs) -> Tuple[str, str, str, int]:
        """Handle /apply-template endpoint.
        
        Args:
            **kwargs: Template application parameters
            
        Returns:
            Tuple of (response, raw_response, error, status_code)
        """
        messages_str = kwargs.get("messages", "[]")
        messages: List[Dict[str, str]] = []
        if isinstance(messages_str, str):
            try:
                messages = json.loads(messages_str)
            except json.JSONDecodeError:
                messages = []
        params: Dict[str, List[Dict[str, str]]] = {"messages": messages}
        return self._make_request("/apply-template", params)

    def handle_infill(self, **kwargs) -> Tuple[str, str, str, int]:
        """Handle /infill endpoint.
        
        Args:
            **kwargs: Infill-specific parameters
            
        Returns:
            Tuple of (response, raw_response, error, status_code)
        """
        params: Dict[str, Any] = {
            "input_prefix": kwargs.get("input_prefix", ""),
            "input_suffix": kwargs.get("input_suffix", ""),
        }
        if kwargs.get("input_extra"):
            try:
                params["input_extra"] = json.loads(kwargs["input_extra"])
            except json.JSONDecodeError:
                pass
        if kwargs.get("prompt"):
            params["prompt"] = kwargs["prompt"]

        completion_params = [
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "seed",
            "stream",
            "n_predict",
            "stop_sequences",
            "repeat_penalty",
            "repeat_last_n",
        ]

        for param in completion_params:
            if param in kwargs and kwargs[param] is not None:
                if param == "stop_sequences":
                    params["stop"] = kwargs[param]
                else:
                    params[param] = kwargs[param]

        params = clean_params(params)
        return self._make_request("/infill", params)

    def handle_reranking(self, **kwargs) -> Tuple[str, str, str, int]:
        """Handle /v1/rerank endpoint.
        
        Args:
            **kwargs: Reranking-specific parameters
            
        Returns:
            Tuple of (response, raw_response, error, status_code)
        """
        query = kwargs.get("query", "")
        documents_str = kwargs.get("documents", "[]")
        documents: List[str] = []
        if isinstance(documents_str, str):
            try:
                documents = json.loads(documents_str)
            except json.JSONDecodeError:
                documents = []

        params: Dict[str, Any] = {
            "model": kwargs.get("model", "default"),
            "query": query,
            "documents": documents,
            "top_n": kwargs.get("top_n", 10),
        }
        return self._make_request("/v1/rerank", params)

    def close(self):
        """Close the client and release resources."""
        self._close_session()
        self._reset_timing()