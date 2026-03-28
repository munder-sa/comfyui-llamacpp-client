import json
import re
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Tuple  # Removed unused Optional

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


def retry_on_transient_error(
    max_retries: int = 3, base_delay: float = 1.0
) -> Callable[[Callable], Callable]:
    """Decorator to retry on transient network errors with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        msg = (
                            f"Transient error ({type(e).__name__}), "
                            f"retrying in {delay}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        log_info(msg)
                        time.sleep(delay)
                    else:
                        log_error(f"Max retries ({max_retries}) exceeded after transient error")
                        raise
            return None

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
        """Get or create a requests session."""
        if self._session is None:
            self._session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
        return self._session

    def _close_session(self):
        """Close the session."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def _reset_timing(self):
        self._request_start_time = None

    @retry_on_transient_error(max_retries=3, base_delay=1.0)
    def _make_request(self, endpoint_path: str, data: Dict[str, Any]) -> Tuple[Any, str, str, int]:
        """Make HTTP POST request to llama-server."""
        url = f"{self.base_url}{endpoint_path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        log_info(f"Sending request to {url}")
        try:
            self._request_start_time = time.time()
            session = self._get_session()
            response = session.post(url, json=data, headers=headers, timeout=self.timeout)

            elapsed_time = time.time() - self._request_start_time if self._request_start_time else 0
            log_debug(f"Request completed in {elapsed_time: .2f}s")  # Added missing whitespace

            if response.status_code >= 400:
                return {}, response.text, f"HTTP {response.status_code}", response.status_code

            try:
                response_json = response.json()

                # Validation logic
                is_valid = False
                is_chat = "/chat/" in endpoint_path

                if isinstance(response_json, dict):
                    if is_chat:
                        is_valid = bool(response_json.get("choices"))
                    else:
                        is_valid = "content" in response_json or "text" in response_json

                if not is_valid:
                    return {}, json.dumps(response_json), "Invalid structure", 200

                # Tag extraction depending on endpoint
                content_str = ""
                if is_chat and is_valid:
                    choices = response_json.get("choices", [])
                    if choices:
                        content_str = choices[0].get("message", {}).get("content", "")
                elif "content" in response_json:
                    content_str = response_json["content"]

                if content_str:
                    match = re.search(r"<prompt>(.*?)</prompt>", content_str, re.DOTALL)
                    if match:
                        cleaned_content = match.group(1).strip()
                    else:
                        cleaned_content = re.sub(
                            r"<think>.*?</think>", "", content_str, flags=re.DOTALL
                        ).strip()

                    # Put it back where it belongs
                    if is_chat:
                        response_json["choices"][0]["message"]["content"] = cleaned_content
                    else:
                        response_json["content"] = cleaned_content

                return response_json, json.dumps(response_json, indent=2), "", response.status_code

            except json.JSONDecodeError:
                return {}, response.text, "Invalid JSON", 502

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            raise
        except Exception as e:
            return {}, "", str(e), 500

    def handle_completion(self, prompt: str, **kwargs) -> Tuple[Any, str, str, int]:
        """Handle /completion endpoint. THIS MUST BE AT THE SAME LEVEL AS _make_request"""
        params: Dict[str, Any] = {"prompt": prompt}
        mapped_params = map_parameters(kwargs, COMMON_COMPLETION_PARAMS)
        params.update(mapped_params)
        params = clean_params(params)

        # 5090 God Speed Mode Stop sequence
        params.update({"stop": ["</prompt>", "</s>", "\n\n"]})

        return self._make_request("/completion", params)

    def handle_chat_completions(self, **kwargs) -> Tuple[Any, str, str, int, Dict[str, Any]]:
        """Handle /v1/chat/completions endpoint."""
        messages: List[Dict[str, Any]] = []

        # 1. System message goes first
        system_message = kwargs.get("system_message")
        if isinstance(system_message, str) and system_message.strip():
            messages.append({"role": "system", "content": system_message})

        # 2. Add history
        messages_input = kwargs.get("messages")
        if messages_input:
            if isinstance(messages_input, str):
                try:
                    parsed_messages = json.loads(messages_input)
                    if isinstance(parsed_messages, list):
                        messages.extend(parsed_messages)
                except Exception:
                    pass
            elif isinstance(messages_input, list):
                messages.extend(messages_input)

        # 3. Add current user text and images
        user_text = kwargs.get("user_message") or kwargs.get("prompt", "")
        image_data = kwargs.get("image_data")
        tensor_images = kwargs.get("images")

        if isinstance(image_data, str) and image_data.strip():
            image_data = process_image_data_string(image_data)

        user_content_items, metadata_list = build_vision_content(
            user_text,
            image_data,
            tensor_images,
            extract_metadata=kwargs.get("extract_metadata", False),
        )

        if user_content_items:
            if len(user_content_items) == 1 and user_content_items[0]["type"] == "text":
                messages.append({"role": "user", "content": user_content_items[0]["text"]})
            else:
                messages.append({"role": "user", "content": user_content_items})

        # 4. Add assistant message (if pre-filling)
        if kwargs.get("assistant_message") and kwargs["assistant_message"].strip():
            messages.append({"role": "assistant", "content": kwargs["assistant_message"]})

        params: Dict[str, Any] = {"messages": messages, "model": kwargs.get("model", "default")}
        params.update(map_parameters(kwargs, CHAT_COMPLETION_PARAMS))
        params = clean_params(params)

        response, raw_response, error, status_code = self._make_request(
            "/v1/chat/completions", params
        )
        return response, raw_response, error, status_code, metadata_list

    def close(self):
        self._close_session()
        self._reset_timing()
